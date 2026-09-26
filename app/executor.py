"""Bounded-concurrency task executor.

* A fixed-size :class:`~concurrent.futures.ThreadPoolExecutor` guarantees a
  limited number of concurrent evaluations; excess tasks wait in the queue,
  and a slow task never blocks independent work (others run on free workers).
* Every task has a :class:`~app.evaluator.CancellationToken`. Cancelling a
  queued task flips the flag before it starts so it aborts immediately;
  cancelling a running task is observed cooperatively at sample boundaries.
* Per-task deadlines are enforced by a watchdog thread that cancels the token
  on timeout. Timeouts and cancellations never write partial results — the
  worker only persists an outcome via compare-and-swap.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from app.evaluator import CancellationToken

log = logging.getLogger(__name__)

#: A runnable unit; receives the task id and cancellation token.
RunFn = Callable[[str, CancellationToken], None]


@dataclass
class _Handle:
    task_id: str
    token: CancellationToken
    future: Future
    timeout_seconds: float | None
    timer: threading.Timer | None = None
    done: threading.Event = field(default_factory=threading.Event)


class Executor:
    def __init__(self, max_workers: int = 4, default_timeout: float | None = 60.0,
                 on_timeout: Callable[[str], None] | None = None):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="eval-worker",
        )
        self._default_timeout = default_timeout
        self._on_timeout_cb = on_timeout
        self._handles: dict[str, _Handle] = {}
        self._lock = threading.Lock()
        self._shutdown = False

    # ------------------------------------------------------------- lifecycle
    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True
            handles = list(self._handles.values())
        for h in handles:
            h.token.cancel()
            if h.timer is not None:
                h.timer.cancel()
        self._pool.shutdown(wait=wait, cancel_futures=True)

    # ------------------------------------------------------------------ enqueue
    def submit(
        self,
        task_id: str,
        run_fn: RunFn,
        *,
        timeout_seconds: float | None = None,
    ) -> CancellationToken:
        """Schedule ``run_fn`` and return its cancellation token.

        Re-submitting the same task id replaces no state — the service layer
        is responsible for de-duplication.
        """
        timeout = (
            timeout_seconds if timeout_seconds is not None
            else self._default_timeout
        )
        token = CancellationToken()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("executor has been shut down")
            future = self._pool.submit(self._run, task_id, run_fn, token)
            timer: threading.Timer | None = None
            if timeout is not None and timeout > 0:
                timer = threading.Timer(timeout, self._on_timeout, (task_id,))
                timer.daemon = True
                timer.start()
            self._handles[task_id] = _Handle(
                task_id=task_id,
                token=token,
                future=future,
                timeout_seconds=timeout,
                timer=timer,
            )
        return token

    def _run(self, task_id: str, run_fn: RunFn, token: CancellationToken) -> None:
        try:
            run_fn(task_id, token)
        except Exception:  # worker exceptions are expected to be handled in run_fn
            log.exception("task %s raised an unhandled exception", task_id)
        finally:
            with self._lock:
                handle = self._handles.pop(task_id, None)
            if handle is not None:
                if handle.timer is not None:
                    handle.timer.cancel()
                handle.done.set()

    def _on_timeout(self, task_id: str) -> None:
        log.warning("task %s timed out; requesting cancellation", task_id)
        with self._lock:
            handle = self._handles.get(task_id)
        if handle is not None:
            handle.token.request_timeout()
        cb = self._on_timeout_cb
        if cb is not None:
            try:
                cb(task_id)
            except Exception:
                log.exception("timeout callback failed for task %s", task_id)

    # ----------------------------------------------------------------- control
    def cancel(self, task_id: str) -> bool:
        """Signal cancellation. Returns True if the task was active/queued."""
        with self._lock:
            handle = self._handles.get(task_id)
        if handle is None:
            return False
        handle.token.cancel()
        return True

    def is_active(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._handles

    def wait(self, task_id: str, timeout: float | None = None) -> bool:
        """Block until the task's worker finishes. Returns True if finished."""
        with self._lock:
            handle = self._handles.get(task_id)
        if handle is None:
            return True
        return handle.done.wait(timeout)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._handles)
