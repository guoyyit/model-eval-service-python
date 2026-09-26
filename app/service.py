"""Evaluation service facade: orchestrates storage, quotas and the executor.

Thread-safety: a single :class:`threading.RLock` guards submission/cancel
orchestration; the storage layer has its own atomic transactions, and the
executor is independently thread-safe.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from typing import Self

from app.evaluator import (
    CancellationToken,
    TaskCancelled,
    TaskTimeout,
    evaluate,
)
from app.executor import Executor
from app.models import (
    EvalRequest,
    QuotaInfo,
    SubmitResponse,
    TaskResult,
    TaskState,
    TaskStatus,
    canonical_json,
    quota_scope,
    request_fingerprint,
)
from app.quotas import QuotaService
from app.storage import Storage

log = logging.getLogger(__name__)


class TaskNotFoundError(KeyError):
    pass


class CrossTenantBaselineError(ValueError):
    pass


def _serialize_result(result: TaskResult) -> str:
    """Deterministic JSON for storage; non-finite floats become null."""
    def sanitize(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        return value

    return canonical_json(sanitize(result.model_dump()))


class EvalService:
    def __init__(
        self,
        db_path: str = "eval.db",
        max_workers: int = 4,
        default_quota: int = 100,
        default_timeout: float | None = 60.0,
        autostart: bool = True,
        enqueue_on_submit: bool = True,
    ):
        self.storage = Storage(db_path, default_quota=default_quota)
        self.quotas = QuotaService(self.storage, default_limit=default_quota)
        self.executor = Executor(
            max_workers=max_workers,
            default_timeout=default_timeout,
            on_timeout=self._on_timeout,
        )
        self._lock = threading.RLock()
        self._closed = False
        self._enqueue_on_submit = enqueue_on_submit
        # Recovery: completed tasks remain readable; interrupted RUNNING tasks
        # are re-queued idempotently (no new task rows, no quota changes).
        self.storage.reset_interrupted()
        if autostart:
            self.recover_pending()

    # ---------------------------------------------------------------- lifecycle
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.executor.shutdown(wait=True)
        self.storage.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def recover_pending(self) -> int:
        """Re-enqueue all PENDING tasks (e.g. after a crash)."""
        count = 0
        for task in self.storage.list_pending():
            req = EvalRequest(**json.loads(task["request_json"]))
            self._enqueue(task["task_id"], req)
            count += 1
        return count

    # ----------------------------------------------------------------- submit
    def submit(self, req: EvalRequest) -> SubmitResponse:
        fingerprint = request_fingerprint(req)
        task_id = f"task-{fingerprint[:16]}"
        scope = quota_scope(req.tenant_id, req.project_id)
        request_json = canonical_json(req.model_dump())

        with self._lock:
            self._ensure_open()
            # The storage transaction is authoritative: it deduplicates first
            # (so a repeated submission succeeds even when quota is exhausted
            # and never re-charges), then atomically enforces quota.
            real_id, reused = self.storage.submit_task(
                task_id=task_id,
                fingerprint=fingerprint,
                scope=scope,
                tenant_id=req.tenant_id,
                project_id=req.project_id,
                model_version=req.model_version,
                request_json=request_json,
            )
            if reused:
                existing = self.storage.get_task(real_id)
                state = existing["state"] if existing else TaskState.PENDING
                return SubmitResponse(task_id=real_id, state=state, reused=True)
            if self._enqueue_on_submit:
                self._enqueue(real_id, req)
            return SubmitResponse(task_id=real_id, state=TaskState.PENDING,
                                  reused=False)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("service is closed")

    def _enqueue(self, task_id: str, req: EvalRequest) -> None:
        timeout = req.timeout_seconds
        self.executor.submit(task_id, self._run_task, timeout_seconds=timeout)

    # ------------------------------------------------------------------- run
    def _run_task(self, task_id: str, token: CancellationToken) -> None:
        task = self.storage.get_task(task_id)
        if task is None:
            log.warning("task %s vanished before execution", task_id)
            return
        # A cancellation/timeout may have been requested while queued.
        queued_intent = self.storage.get_cancel_reason(task_id)
        if queued_intent is not None:
            self._finalize_by_intent(task_id, queued_intent)
            return

        req = EvalRequest(**json.loads(task["request_json"]))
        if not self.storage.mark_running(task_id):
            return  # cancelled before start, or already running elsewhere

        baseline_metrics: dict[str, float] | None = None
        if req.baseline is not None:
            try:
                baseline_metrics = self._resolve_baseline(req)
            except Exception as exc:
                self.storage.finish_task(
                    task_id, TaskState.FAILED, error=str(exc)
                )
                return

        try:
            result = evaluate(req, token, baseline_metrics=baseline_metrics)
        except (TaskCancelled, TaskTimeout):
            reason = token.reason or self.storage.get_cancel_reason(task_id) \
                or "cancelled"
            self._finalize_by_intent(task_id, reason)
            return
        except Exception as exc:  # unexpected engine failure
            log.exception("task %s failed", task_id)
            self.storage.finish_task(task_id, TaskState.FAILED,
                                     error=f"{type(exc).__name__}: {exc}")
            return

        # First-intent-wins: a cancel/timeout recorded before completion
        # suppresses the result so no partial/unwanted output is persisted.
        outcome = self.storage.complete_if_not_cancelled(
            task_id, result_json=_serialize_result(result)
        )
        if outcome not in (None, TaskState.SUCCEEDED.value):
            log.info("task %s result discarded due to earlier %s intent",
                     task_id, outcome)

    def _on_timeout(self, task_id: str) -> None:
        """Persist a timeout intent (first intent wins) so it survives a
        crash; finalize queued tasks immediately, signal running ones."""
        recorded = self.storage.request_cancel(task_id, "timeout")
        if not recorded:
            return
        task = self.storage.get_task(task_id)
        if task is None:
            return
        if task["state"] == TaskState.PENDING:
            self.storage.finish_task(
                task_id, TaskState.TIMEOUT, error="evaluation timed out"
            )

    def _finalize_by_intent(self, task_id: str, reason: str) -> None:
        state = TaskState.TIMEOUT if reason == "timeout" else TaskState.CANCELLED
        error = ("evaluation timed out" if reason == "timeout"
                 else "cancelled by request")
        self.storage.finish_task(task_id, state, error=error)

    def _resolve_baseline(self, req: EvalRequest) -> dict[str, float]:
        assert req.baseline is not None
        if req.baseline.metrics is not None:
            return dict(req.baseline.metrics)
        if req.baseline.task_id is None:
            raise ValueError("baseline requires task_id or metrics")
        base_task = self.storage.get_task(req.baseline.task_id)
        if base_task is None:
            raise ValueError(f"baseline task not found: {req.baseline.task_id}")
        if base_task["tenant_id"] != req.tenant_id:
            raise CrossTenantBaselineError(
                "baseline task belongs to a different tenant"
            )
        if base_task["state"] is not TaskState.SUCCEEDED:
            raise ValueError(
                f"baseline task {req.baseline.task_id} is not succeeded "
                f"(state={base_task['state'].value})"
            )
        result = json.loads(base_task["result_json"])
        return {k: v for k, v in result["metrics"].items() if v is not None}

    # ------------------------------------------------------------------ read
    def get_status(self, task_id: str) -> TaskStatus:
        task = self.storage.get_task(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        return self._to_status(task)

    def get_result(self, task_id: str) -> TaskResult:
        task = self.storage.get_task(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        if task["state"] is not TaskState.SUCCEEDED or not task["result_json"]:
            raise ValueError(
                f"task {task_id} has no result (state={task['state'].value})"
            )
        return TaskResult(**json.loads(task["result_json"]))

    def list_tasks(
        self, tenant_id: str, project_id: str | None = None
    ) -> list[TaskStatus]:
        return [
            self._to_status(t)
            for t in self.storage.list_tasks(tenant_id, project_id)
        ]

    @staticmethod
    def _to_status(task: dict) -> TaskStatus:
        result = None
        if task["result_json"]:
            result = TaskResult(**json.loads(task["result_json"]))
        return TaskStatus(
            task_id=task["task_id"],
            tenant_id=task["tenant_id"],
            project_id=task["project_id"],
            model_version=task["model_version"],
            state=task["state"],
            created_at=task["created_at"],
            started_at=task["started_at"],
            finished_at=task["finished_at"],
            error=task["error"],
            result=result,
        )

    # ------------------------------------------------------------------ cancel
    def cancel(self, task_id: str) -> TaskStatus:
        """Cancel a task deterministically.

        The cancellation intent is persisted first (first writer wins), then:
        a queued/pending task is finalized immediately; a running task is
        signalled and aborts at the next sample boundary. If a running
        evaluation has *already* completed, its SUCCEEDED result wins and is
        returned unchanged — the outcome always reflects whichever happened
        first. Terminal tasks are returned as-is.
        """
        with self._lock:
            task = self.storage.get_task(task_id)
            if task is None:
                raise TaskNotFoundError(task_id)
            if task["state"] in (
                TaskState.SUCCEEDED, TaskState.FAILED,
                TaskState.CANCELLED, TaskState.TIMEOUT,
            ):
                return self._to_status(task)

            recorded = self.storage.request_cancel(task_id, "cancelled")
            current = self.storage.get_task(task_id)
            if current["state"] == TaskState.PENDING:
                # Queued task: finalize immediately, it never executes.
                self.storage.finish_task(
                    task_id, TaskState.CANCELLED,
                    error="cancelled by request",
                )
            elif recorded:
                # Running task with a fresh cancel intent: signal the worker;
                # it finalizes at the next sample boundary (first intent wins).
                self.executor.cancel(task_id)
            # else: another intent (e.g. timeout) already exists — leave the
            # worker to finalize deterministically via that intent.
            return self.get_status(task_id)

    # ------------------------------------------------------------------ quotas
    def set_quota(self, tenant_id: str, project_id: str | None,
                  limit: int) -> QuotaInfo:
        return self.quotas.set_limit(tenant_id, project_id, limit)

    def get_quota(self, tenant_id: str,
                  project_id: str | None = None) -> QuotaInfo:
        return self.quotas.get_info(quota_scope(tenant_id, project_id))

    def list_quotas(self) -> list[QuotaInfo]:
        return self.quotas.list_infos()
