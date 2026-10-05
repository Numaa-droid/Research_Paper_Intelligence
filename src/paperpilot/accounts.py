"""User accounts: sign-up behind an invite code, password login, sessions.

Passwords are hashed with scrypt from the standard library. A session is a
random token handed to the browser as an HttpOnly cookie (or to a script as a
bearer token); only its SHA-256 is stored, so a leaked database does not leak
live sessions.

The first account created takes ownership of every paper uploaded before
accounts existed, so an existing installation keeps its history.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from .config import Settings
from .errors import AuthenticationError, ForbiddenError, ValidationError
from .observability import info
from .storage.repository import DocumentRepository, User, UserRepository

SESSION_DAYS = 30
MIN_PASSWORD = 8
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt, digest = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    candidate = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), **_SCRYPT)
    return hmac.compare_digest(candidate, base64.b64decode(digest))


#: Checked against when the email is unknown, so a wrong email costs as much
#: time as a wrong password and response timing does not reveal accounts.
_DUMMY_HASH = hash_password(secrets.token_hex(8))


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class AccountService:
    def __init__(
        self, users: UserRepository, documents: DocumentRepository, *, settings: Settings
    ) -> None:
        self.users = users
        self.documents = documents
        self.settings = settings

    @property
    def invite_code(self) -> str | None:
        return self.settings.invite_code or None

    @property
    def invite_required(self) -> bool:
        return self.invite_code is not None

    def sign_up(self, *, name: str, email: str, password: str, invite_code: str | None) -> tuple[User, str]:
        if self.invite_code is not None and not hmac.compare_digest(
            (invite_code or "").strip().encode(), self.invite_code.encode()
        ):
            raise ForbiddenError("that invite code is not valid")
        name, email = name.strip(), email.strip().lower()
        if not name:
            raise ValidationError("please enter your name")
        if not _EMAIL.match(email):
            raise ValidationError("please enter a valid email address")
        if len(password) < MIN_PASSWORD:
            raise ValidationError(f"the password must be at least {MIN_PASSWORD} characters")

        first = self.users.count() == 0
        user = User(
            user_id=f"user-{uuid.uuid4().hex[:12]}",
            email=email,
            name=name[:80],
            password_hash=hash_password(password),
            created_at=datetime.now(UTC),
        )
        self.users.create(user)   # raises ConflictError for an email already registered
        if first:
            claimed = self.documents.claim_unowned(user.user_id)
            info("accounts.first_user", user_id=user.user_id, claimed_papers=claimed)
        info("accounts.signed_up", user_id=user.user_id)
        return user, self._start_session(user)

    def log_in(self, *, email: str, password: str) -> tuple[User, str]:
        user = self.users.get_by_email(email.strip().lower())
        if not verify_password(password, user.password_hash if user else _DUMMY_HASH) or user is None:
            raise AuthenticationError("wrong email or password")
        return user, self._start_session(user)

    def log_out(self, token: str) -> None:
        self.users.delete_session(_token_hash(token))

    def user_for(self, token: str | None) -> User | None:
        if not token:
            return None
        return self.users.session_user(_token_hash(token), datetime.now(UTC))

    def change_password(self, user: User, *, current: str, new: str, token: str) -> None:
        if not verify_password(current, user.password_hash):
            raise AuthenticationError("the current password is wrong")
        if len(new) < MIN_PASSWORD:
            raise ValidationError(f"the password must be at least {MIN_PASSWORD} characters")
        self.users.update_password(user.user_id, hash_password(new))
        # Other devices signed in with the old password are signed out.
        self.users.delete_sessions(user.user_id, keep=_token_hash(token))

    def _start_session(self, user: User) -> str:
        token = secrets.token_urlsafe(32)
        expires = datetime.now(UTC) + timedelta(days=SESSION_DAYS)
        self.users.create_session(_token_hash(token), user.user_id, expires)
        return token
