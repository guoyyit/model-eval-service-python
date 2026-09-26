"""SQLite-backed persistent store for tasks, results and quota.

Correctness guarantees:

* Task dedup is enforced by a partial UNIQUE index on
  (tenant, fingerprint) covering only live states (queued, running,
  succeeded): concurrent identical submissions create exactly one task,
  a succeeded task acts as a result cache, and a cancelled/failed/
  timed-out request can be resubmitted as a fresh task.
* Quota is charged inside the same transaction as task creation and the
  ledger is keyed by (tenant, task_id), so a task can never be charged
  twice -- including across crash recovery.
* Status changes are atomic compare-and-swap UPDATEs. A result is only
  ever written in the same statement that flips the status to SUCCEEDED,
  so cancelled/timed-out tasks can never carry partial results.
* WAL + FULL synchronous give durability across process crashes.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from .models import EvalRequest, EvalResult, TaskRecord, TaskStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id      TEXT PRIMARY KEY,
    tenant       TEXT NOT NULL,
    project      TEXT,
    fingerprint  TEXT NOT NULL,
    status       TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json  TEXT,
    error        TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
-- Dedup applies only to live work: an identical request dedups against a
-- queued/running task (single execution) or a succeeded one (result
-- cache), while a cancelled/failed/timed-out request may be resubmitted.
CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_live_fingerprint
    ON tasks (tenant, fingerprint)
    WHERE status IN ('queued', 'running', 'succeeded');
CREATE TABLE IF NOT EXISTS quota_ledger (
    tenant     TEXT NOT NULL,
    task_id    TEXT NOT NULL,
    amount     INTEGER NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (tenant, task_id)
);
"""


class QuotaExceeded(Exception):
    """Raised when a tenant has exhausted its task quota."""


class Store:
    """Durable store. All state transitions are atomic SQL operations."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- mapping --------------------------------------------------------
    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            tenant=row["tenant"],
            project=row["project"],
            fingerprint=row["fingerprint"],
            status=TaskStatus(row["status"]),
            request=EvalRequest.model_validate_json(row["request_json"]),
            result=(
                EvalResult.model_validate_json(row["result_json"])
                if row["result_json"]
                else None
            ),
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- task lifecycle -------------------------------------------------
    def find_by_fingerprint(self, tenant: str, fingerprint: str) -> TaskRecord | None:
        """Find the live (queued/running/succeeded) task for a request."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE tenant=? AND fingerprint=?"
                " AND status IN ('queued','running','succeeded')",
                (tenant, fingerprint),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def create_task(
        self,
        request: EvalRequest,
        fingerprint: str,
        task_id: str,
        quota_limit: int,
    ) -> tuple[TaskRecord, bool]:
        """Atomically dedup, charge quota and insert a queued task.

        Returns (record, created). When an identical request already
        exists, returns the existing record with created=False and does
        NOT charge quota again. Raises QuotaExceeded when the tenant is
        over its limit; in that case no task row is left behind.
        """
        now = time.time()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT * FROM tasks WHERE tenant=? AND fingerprint=?"
                    " AND status IN ('queued','running','succeeded')",
                    (request.tenant, fingerprint),
                ).fetchone()
                if row is not None:
                    self._conn.commit()
                    return self._row_to_record(row), False

                # Idempotent charge: ledger PK (tenant, task_id) makes a
                # retried charge for the same task a no-op.
                self._conn.execute(
                    "INSERT OR IGNORE INTO quota_ledger"
                    " (tenant, task_id, amount, created_at) VALUES (?,?,1,?)",
                    (request.tenant, task_id, now),
                )
                usage = self._conn.execute(
                    "SELECT COALESCE(SUM(amount),0) AS u FROM quota_ledger WHERE tenant=?",
                    (request.tenant,),
                ).fetchone()["u"]
                if usage > quota_limit:
                    self._conn.rollback()
                    raise QuotaExceeded(
                        f"tenant {request.tenant!r} quota exceeded: limit={quota_limit}"
                    )
                self._conn.execute(
                    "INSERT INTO tasks (task_id, tenant, project, fingerprint,"
                    " status, request_json, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (
                        task_id,
                        request.tenant,
                        request.project,
                        fingerprint,
                        TaskStatus.QUEUED.value,
                        request.model_dump_json(),
                        now,
                        now,
                    ),
                )
                self._conn.commit()
            except QuotaExceeded:
                raise
            except Exception:
                self._conn.rollback()
                raise
        record = self.get_task(task_id)
        assert record is not None  # noqa: S101 - internal invariant
        return record, True

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_tasks(self, tenant: str | None = None) -> list[TaskRecord]:
        with self._lock:
            if tenant is None:
                rows = self._conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM tasks WHERE tenant=? ORDER BY created_at",
                    (tenant,),
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def transition(
        self,
        task_id: str,
        to: TaskStatus,
        from_states: set[TaskStatus],
        *,
        result: EvalResult | None = None,
        error: str | None = None,
    ) -> bool:
        """Compare-and-swap status transition; optionally persist a result.

        Returns True when this call performed the transition. Whoever
        wins the CAS owns the final state, which eliminates the race
        between cancellation and result writing.
        """
        now = time.time()
        placeholders = ",".join("?" for _ in from_states)
        result_json = result.model_dump_json() if result is not None else None
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE tasks SET status=?,"
                f" result_json=COALESCE(?, result_json), error=?, updated_at=?"
                f" WHERE task_id=? AND status IN ({placeholders})",
                (
                    to.value,
                    result_json,
                    error,
                    now,
                    task_id,
                    *[s.value for s in from_states],
                ),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def recover_inflight(self) -> list[TaskRecord]:
        """Reset crashed RUNNING tasks to QUEUED and return all queued tasks.

        Re-execution is idempotent: results are only ever written by the
        atomic success transition, and quota was charged once at creation.
        """
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE status=?",
                (TaskStatus.QUEUED.value, now, TaskStatus.RUNNING.value),
            )
            self._conn.commit()
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY created_at",
                (TaskStatus.QUEUED.value,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # -- quota ledger ---------------------------------------------------
    def quota_usage(self, tenant: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(amount),0) AS u FROM quota_ledger WHERE tenant=?",
                (tenant,),
            ).fetchone()
        return int(row["u"])
