"""Access control for the HTTP API.

Three independent guards, each cheap enough to run before any real work:

* ``SessionMiddleware`` resolves the caller's session (cookie or bearer token)
  and turns away a request without one before its body is read, so an anonymous
  client can neither store a file nor start an analysis that spends budget.
* ``UploadSizeMiddleware`` stops an upload as soon as it passes the size cap,
  from ``Content-Length`` when the client sends one and by counting bytes as
  they arrive when it does not, so an oversized file is never spooled whole.
* ``RateLimiter`` caps how many uploads and analysis starts one client makes
  in a window.

The first two are plain ASGI middleware so they run ahead of FastAPI's body
parsing; the limiter is a route dependency, applied to the costly routes only.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

from fastapi import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..accounts import AccountService
from ..errors import AuthenticationError, FileTooLargeError, RateLimitError

SESSION_COOKIE = "pm_session"
#: Served without a session: the UI shells (they show the sign-in screen themselves),
#: the sign-up and log-in calls, the liveness probe, and the API description.
PUBLIC_PATHS = frozenset(
    {
        "/", "/tracing", "/health", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json",
        "/auth/config", "/auth/signup", "/auth/login", "/auth/logout",
    }
)
#: Multipart framing around the file: boundaries and part headers.
_MULTIPART_OVERHEAD = 64 * 1024


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", ()):
        if key == name:
            return value.decode("latin-1")
    return None


def session_token(scope: Scope) -> str | None:
    """The session token from ``Authorization: Bearer`` or the session cookie."""
    scheme, _, token = (_header(scope, b"authorization") or "").partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    for part in (_header(scope, b"cookie") or "").split(";"):
        name, _, value = part.strip().partition("=")
        if name == SESSION_COOKIE and value:
            return value
    return None


class SessionMiddleware:
    """Attach the signed-in user to the request; refuse non-public paths without one."""

    def __init__(self, app: ASGIApp, *, accounts: AccountService) -> None:
        self.app = app
        self.accounts = accounts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        user = self.accounts.user_for(session_token(scope))
        scope.setdefault("state", {})["user"] = user
        if user is not None or scope["path"] in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        error = AuthenticationError("please sign in")
        response = JSONResponse(
            error.to_dict(), status_code=error.http_status, headers={"WWW-Authenticate": "Bearer"}
        )
        await response(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


class UploadSizeMiddleware:
    """Reject an upload body larger than ``max_bytes`` while it is still arriving."""

    def __init__(self, app: ASGIApp, *, max_bytes: int, paths: frozenset[str] = frozenset({"/papers"})) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.limit = max_bytes + _MULTIPART_OVERHEAD
        self.paths = paths

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        error = FileTooLargeError(
            f"upload exceeds the {self.max_bytes / (1024 * 1024):g} MB limit",
            details={"max_bytes": self.max_bytes},
        )
        await JSONResponse(error.to_dict(), status_code=error.http_status)(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in self.paths:
            await self.app(scope, receive, send)
            return

        declared = _header(scope, b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.limit:
            await self._reject(scope, receive, send)
            return

        received = 0
        exceeded = False
        started = False

        async def counted_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            # Whatever the app makes of a body cut short (FastAPI answers 400)
            # is replaced by the 413 below.
            if exceeded:
                return
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counted_receive, guarded_send)
        except _BodyTooLarge:
            pass
        if exceeded and not started:
            await self._reject(scope, receive, send)


class RateLimiter:
    """Sliding-window count of costly requests per client, in process."""

    def __init__(self, limit: int, window_seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, client: str) -> int | None:
        """Record a request; return seconds to wait if it is over the limit."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window - now))
            hits.append(now)
            if len(self._hits) > 10_000:
                self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - self.window}
            return None


def _client(request: Request) -> str:
    user = getattr(request.state, "user", None)
    if user is not None:
        return user.user_id
    return request.client.host if request.client else "unknown"


def _limit(limiter: RateLimiter | None, request: Request, what: str) -> None:
    if limiter is None:
        return
    retry_after = limiter.hit(_client(request))
    if retry_after is not None:
        raise RateLimitError(
            f"too many {what}; at most {limiter.limit} per {limiter.window:g} s",
            retry_after=retry_after,
        )


def rate_limited(request: Request) -> None:
    """Route dependency: count an upload or analysis start against the account."""
    _limit(request.app.state.rate_limiter, request, "uploads or analyses")


def auth_rate_limited(request: Request) -> None:
    """Route dependency: slow down password guessing, per client address."""
    _limit(request.app.state.auth_limiter, request, "sign-in attempts")
