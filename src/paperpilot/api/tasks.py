"""Background task abstraction.

Analysis is long-running, so the API starts it on a worker thread and clients
poll. The abstraction exists so the executor can later become Celery, RQ or
arq without touching the routes.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from ..observability import error, info


class TaskRunner(ABC):
    """Runs a callable out of band and reports whether it is still running."""

    @abstractmethod
    def submit(self, task_id: str, fn: Callable[[], None]) -> None: ...

    @abstractmethod
    def is_running(self, task_id: str) -> bool: ...

    @abstractmethod
    def shutdown(self) -> None: ...


class ThreadPoolTaskRunner(TaskRunner):
    """In-process thread pool. Adequate for the MVP and for local use."""

    def __init__(self, *, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="paperpilot")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, task_id: str, fn: Callable[[], None]) -> None:
        def wrapped() -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - the record already holds the failure
                error("task.failed", task_id=task_id, error=str(exc)[:300])

        with self._lock:
            existing = self._futures.get(task_id)
            if existing is not None and not existing.done():
                info("task.already_running", task_id=task_id)
                return
            self._futures[task_id] = self._pool.submit(wrapped)
        info("task.submitted", task_id=task_id)

    def is_running(self, task_id: str) -> bool:
        with self._lock:
            future = self._futures.get(task_id)
        return future is not None and not future.done()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class InlineTaskRunner(TaskRunner):
    """Runs tasks synchronously. Used by tests, where determinism matters more."""

    def __init__(self) -> None:
        self._done: set[str] = set()

    def submit(self, task_id: str, fn: Callable[[], None]) -> None:
        fn()
        self._done.add(task_id)

    def is_running(self, task_id: str) -> bool:
        return False

    def shutdown(self) -> None:
        return None
