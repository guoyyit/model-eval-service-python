"""Evaluation service: dedup, quota, scheduling, cancel, recovery.

Concurrency model: a bounded ThreadPoolExecutor executes tasks; every
state change goes through the store's atomic compare-and-swap, so the
race between cancellation and result writing is decided by whichever
transition commits first -- the loser simply does nothing.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
import uuid
from pathlib import Path

from .fingerprint import request_fingerprint
from .metrics import get_metric
from .models import EvalRequest, TaskRecord, TaskStatus
from .quota import QuotaManager
from .runner import EvaluationCancelled, EvaluationTimeout, run_evaluation
from .store import Store

_ACTIVE = {TaskStatus.QUEUED, TaskStatus.RUNNING}


class EvalService:
    def __init__(
        self,
        db_path: str | Path,
        *,
        max_workers: int = 4,
        default_quota: int = 100,
        default_timeout_seconds: float = 60.0,
    ) -> None:
        self._store = Store(db_path)
        self._quota = QuotaManager(self._store, default_quota)
        self._max_workers = max_workers
        self._default_timeout = default_timeout_seconds
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._accepting = False

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        """Recover in-flight tasks and start the worker pool."""
        with self._state_lock:
            if self._executor is not None:
                return
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="eval-worker"
            )
            self._accepting = True
        # Crash recovery: RUNNING tasks are reset to QUEUED by the store
        # and every queued task is rescheduled. Re-execution is idempotent
        # (results are written atomically; quota was charged once).
        for task in self._store.recover_inflight():
            self._schedule(task.task_id)

    def shutdown(self, wait: bool = True) -> None:
        with self._state_lock:
            self._accepting = False
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=False)
        # Unfinished tasks stay QUEUED/RUNNING in the store and will be
        # recovered on the next start().

    def __enter__(self) -> "EvalService":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # -- API ------------------------------------------------------------
    def submit(self, request: EvalRequest) -> tuple[TaskRecord, bool]:
        """Submit an evaluation. Returns (task, created).

        Duplicate submissions of the same request (per tenant) return the
        existing task without charging quota again.
        """
        if not self._accepting:
            raise RuntimeError("service is not running")
        for spec in request.metrics:  # fail fast on unknown metrics
            get_metric(spec.name)
        fingerprint = request_fingerprint(request)

        existing = self._store.find_by_fingerprint(request.tenant, fingerprint)
        if existing is not None:
            return existing, False

        record, created = self._store.create_task(
            request,
            fingerprint,
            task_id=uuid.uuid4().hex,
            quota_limit=self._quota.limit_for(request.tenant),
        )
        if created:
            self._schedule(record.task_id)
        return record, created

    def get(self, task_id: str) -> TaskRecord | None:
        return self._store.get_task(task_id)

    def list(self, tenant: str | None = None) -> list[TaskRecord]:
        return self._store.list_tasks(tenant)

    def cancel(self, task_id: str) -> TaskRecord | None:
        """Cancel a task. Terminal tasks are returned unchanged."""
        with self._cancel_lock:
            event = self._cancel_events.get(task_id)
        if event is not None:
            event.set()
        # Atomic CAS: if the success transition already won, this is a
        # no-op; otherwise the task lands in CANCELLED with no result.
        self._store.transition(
            task_id,
            TaskStatus.CANCELLED,
            _ACTIVE,
            error="cancelled by user",
        )
        return self._store.get_task(task_id)

    def wait_terminal(self, task_id: str, timeout: float = 10.0) -> TaskRecord | None:
        """Poll until the task reaches a terminal state (test/demo helper)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(task_id)
            if record is None or record.status not in _ACTIVE:
                return record
            time.sleep(0.01)
        return self.get(task_id)

    # -- internals ------------------------------------------------------
    @property
    def store(self) -> Store:
        return self._store

    @property
    def quota(self) -> QuotaManager:
        return self._quota

    def _schedule(self, task_id: str) -> None:
        with self._state_lock:
            executor = self._executor
        if executor is None:
            return
        event = threading.Event()
        with self._cancel_lock:
            self._cancel_events[task_id] = event
        executor.submit(self._execute, task_id, event)

    def _execute(self, task_id: str, cancel_event: threading.Event) -> None:
        try:
            # CAS queued -> running; loses against a cancel that landed
            # while the task was still queued.
            if not self._store.transition(
                task_id, TaskStatus.RUNNING, {TaskStatus.QUEUED}
            ):
                return
            record = self._store.get_task(task_id)
            if record is None:
                return
            request = record.request
            timeout = (
                request.timeout_seconds
                if request.timeout_seconds is not None
                else self._default_timeout
            )
            deadline = time.monotonic() + timeout if timeout and timeout > 0 else None
            try:
                result = run_evaluation(
                    request, record.fingerprint, cancel_event, deadline
                )
            except EvaluationCancelled:
                self._store.transition(
                    task_id, TaskStatus.CANCELLED, {TaskStatus.RUNNING},
                    error="cancelled by user",
                )
                return
            except EvaluationTimeout:
                self._store.transition(
                    task_id, TaskStatus.TIMEOUT, {TaskStatus.RUNNING},
                    error=f"evaluation exceeded timeout of {timeout}s",
                )
                return
            except Exception as exc:  # noqa: BLE001 - record any failure
                self._store.transition(
                    task_id, TaskStatus.FAILED, {TaskStatus.RUNNING},
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            # Atomic success: if a cancel won the CAS in the meantime the
            # result is dropped and the task stays CANCELLED -- no partial
            # or post-cancel results are ever persisted.
            self._store.transition(
                task_id, TaskStatus.SUCCEEDED, {TaskStatus.RUNNING}, result=result
            )
        finally:
            with self._cancel_lock:
                self._cancel_events.pop(task_id, None)
