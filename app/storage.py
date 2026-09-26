"""SQLite-backed persistence for tasks, dedup index and quotas.

Concurrency model
-----------------
* A single shared connection guarded by an :class:`threading.RLock`; WAL mode
  is enabled and all write paths use short, explicit transactions.
* Task submission (dedup check + quota check + row inserts) happens in one
  ``IMMEDIATE`` transaction, which makes duplicate concurrent submissions and
  quota accounting atomic.
* State transitions are compare-and-swap (``UPDATE ... WHERE state IN ...``),
  so a late cancellation can never overwrite a finished result.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any

from app.models import TaskState

TERMINAL_STATE_VALUES = tuple(s.value for s in (
    TaskState.SUCCEEDED,
    TaskState.FAILED,
    TaskState.CANCELLED,
    TaskState.TIMEOUT,
))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task (
    task_id       TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    project_id    TEXT,
    scope         TEXT NOT NULL,
    model_version TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    state         TEXT NOT NULL,
    request_json  TEXT NOT NULL,
    error         TEXT,
    result_json   TEXT,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_task_tenant ON task(tenant_id);
CREATE INDEX IF NOT EXISTS idx_task_scope_state ON task(scope, state);

CREATE TABLE IF NOT EXISTS dedup_index (
    fingerprint TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quota (
    scope  TEXT PRIMARY KEY,
    tenant_id  TEXT NOT NULL,
    project_id TEXT,
    "limit"    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cancel_request (
    task_id    TEXT PRIMARY KEY,
    reason     TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class QuotaExceededError(Exception):
    """Raised when a new submission would exceed the scope's quota."""

    def __init__(self, scope: str, used: int, limit: int):
        self.scope = scope
        self.used = used
        self.limit = limit
        super().__init__(
            f"quota exceeded for scope '{scope}': used={used} limit={limit}"
        )


class Storage:
    def __init__(self, db_path: str, default_quota: int = 100):
        self._db_path = db_path
        self.default_quota = default_quota
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA busy_timeout={5000}")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ utils
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _get_task_row(self, task_id: str) -> sqlite3.Row | None:
        cur = self._conn.execute(
            "SELECT * FROM task WHERE task_id = ?", (task_id,)
        )
        return cur.fetchone()

    @staticmethod
    def row_to_task(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"],
            "tenant_id": row["tenant_id"],
            "project_id": row["project_id"],
            "scope": row["scope"],
            "model_version": row["model_version"],
            "fingerprint": row["fingerprint"],
            "state": TaskState(row["state"]),
            "request_json": row["request_json"],
            "error": row["error"],
            "result_json": row["result_json"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._get_task_row(task_id)
            return self.row_to_task(row) if row else None

    def list_tasks(
        self, tenant_id: str, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            if project_id is None:
                cur = self._conn.execute(
                    "SELECT * FROM task WHERE tenant_id = ? ORDER BY created_at",
                    (tenant_id,),
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM task WHERE tenant_id = ? AND project_id = ?"
                    " ORDER BY created_at",
                    (tenant_id, project_id),
                )
            return [self.row_to_task(r) for r in cur.fetchall()]

    # --------------------------------------------------------------- submit
    def submit_task(
        self,
        *,
        task_id: str,
        fingerprint: str,
        scope: str,
        tenant_id: str,
        project_id: str | None,
        model_version: str,
        request_json: str,
    ) -> tuple[str, bool]:
        """Atomically deduplicate + quota-check + insert a new PENDING task.

        Returns ``(task_id, reused)``. ``reused`` is True when an existing
        task with the same fingerprint was found and no quota was consumed.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT task_id FROM dedup_index WHERE fingerprint = ?",
                    (fingerprint,),
                ).fetchone()
                if row is not None:
                    self._conn.execute("COMMIT")
                    return row["task_id"], True

                # Tenant-level quota: counts every task owned by the tenant
                # (with or without a project).
                tenant_used = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM task WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()["c"]
                tenant_limit = self.get_quota_limit(tenant_id)
                if tenant_used >= tenant_limit:
                    self._conn.execute("COMMIT")
                    raise QuotaExceededError(tenant_id, tenant_used, tenant_limit)

                # Project-level quota (only for project-scoped tasks).
                if project_id:
                    proj_used = self._conn.execute(
                        "SELECT COUNT(*) AS c FROM task WHERE scope = ?",
                        (scope,),
                    ).fetchone()["c"]
                    proj_limit = self.get_quota_limit(scope)
                    if proj_used >= proj_limit:
                        self._conn.execute("COMMIT")
                        raise QuotaExceededError(scope, proj_used, proj_limit)

                now = time.time()
                self._conn.execute(
                    "INSERT INTO task (task_id, tenant_id, project_id, scope,"
                    " model_version, fingerprint, state, request_json,"
                    " created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        task_id, tenant_id, project_id, scope, model_version,
                        fingerprint, TaskState.PENDING.value, request_json, now,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO dedup_index (fingerprint, task_id)"
                    " VALUES (?, ?)",
                    (fingerprint, task_id),
                )
                self._conn.execute("COMMIT")
                return task_id, False
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------ transitions
    def mark_running(self, task_id: str) -> bool:
        """PENDING -> RUNNING. Returns False if task no longer pending."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE task SET state = ?, started_at = COALESCE(started_at, ?)"
                " WHERE task_id = ? AND state = ?",
                (TaskState.RUNNING.value, time.time(), task_id,
                 TaskState.PENDING.value),
            )
            return cur.rowcount == 1

    def finish_task(
        self,
        task_id: str,
        state: TaskState,
        *,
        result_json: str | None = None,
        error: str | None = None,
    ) -> bool:
        """Move a non-terminal task to a terminal state and atomically store
        its result/error. Returns False if the task was already terminal
        (e.g. a cancellation won the race) — in that case nothing is written.
        """
        if state not in (
            TaskState.SUCCEEDED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.TIMEOUT,
        ):
            raise ValueError(f"finish_task requires a terminal state, got {state}")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT state FROM task WHERE task_id = ?", (task_id,)
                ).fetchone()
                if row is None or row["state"] in TERMINAL_STATE_VALUES:
                    self._conn.execute("COMMIT")
                    return False
                self._conn.execute(
                    "UPDATE task SET state = ?, result_json = COALESCE(?, result_json),"
                    " error = ?, finished_at = ? WHERE task_id = ?",
                    (state.value, result_json, error, time.time(), task_id),
                )
                self._conn.execute("COMMIT")
                return True
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    # -------------------------------------------------------------- cancels
    def request_cancel(self, task_id: str, reason: str = "cancelled") -> bool:
        """Persist a cancellation/timeout intent (first writer wins).

        Returns True if this call recorded the intent, False if a task with
        the id does not exist or an intent already existed.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                task = self._conn.execute(
                    "SELECT state FROM task WHERE task_id = ?", (task_id,)
                ).fetchone()
                if task is None:
                    self._conn.execute("COMMIT")
                    return False
                if task["state"] in TERMINAL_STATE_VALUES:
                    self._conn.execute("COMMIT")
                    return False
                existing = self._conn.execute(
                    "SELECT task_id FROM cancel_request WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if existing is not None:
                    self._conn.execute("COMMIT")
                    return False
                self._conn.execute(
                    "INSERT INTO cancel_request (task_id, reason, created_at)"
                    " VALUES (?,?,?)",
                    (task_id, reason, time.time()),
                )
                self._conn.execute("COMMIT")
                return True
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    def get_cancel_reason(self, task_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT reason FROM cancel_request WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            return row["reason"] if row else None

    def complete_if_not_cancelled(
        self, task_id: str, *, result_json: str
    ) -> str | None:
        """Persist a successful result unless a cancel/timeout was requested
        earlier (or the task is already terminal).

        Returns the winning terminal state value: ``succeeded`` when the
        result was stored, otherwise ``cancelled``/``timeout``. This gives a
        deterministic, first-intent-wins race between cancellation and
        result writing.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                task = self._conn.execute(
                    "SELECT state FROM task WHERE task_id = ?", (task_id,)
                ).fetchone()
                if task is None:
                    self._conn.execute("COMMIT")
                    return None
                if task["state"] in TERMINAL_STATE_VALUES:
                    self._conn.execute("COMMIT")
                    return task["state"]
                cancel = self._conn.execute(
                    "SELECT reason FROM cancel_request WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                now = time.time()
                if cancel is not None:
                    state_value = (
                        TaskState.TIMEOUT.value
                        if cancel["reason"] == "timeout"
                        else TaskState.CANCELLED.value
                    )
                    error = (
                        "evaluation timed out"
                        if cancel["reason"] == "timeout"
                        else "cancelled by request"
                    )
                    self._conn.execute(
                        "UPDATE task SET state = ?, error = ?, finished_at = ?"
                        " WHERE task_id = ?",
                        (state_value, error, now, task_id),
                    )
                    self._conn.execute("COMMIT")
                    return state_value
                self._conn.execute(
                    "UPDATE task SET state = ?, result_json = ?, error = NULL,"
                    " finished_at = ? WHERE task_id = ?",
                    (TaskState.SUCCEEDED.value, result_json, now, task_id),
                )
                self._conn.execute("COMMIT")
                return TaskState.SUCCEEDED.value
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    def reset_interrupted(self) -> list[str]:
        """Recovery: move RUNNING tasks back to PENDING for re-execution.

        Tasks that already had a cancellation/timeout intent are finalized
        directly instead of being re-run. Completed tasks remain readable; no
        rows are inserted/removed, so quota usage (derived from task rows) is
        never double counted.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self._conn.execute(
                    "SELECT t.task_id AS task_id, c.reason AS reason"
                    " FROM task t LEFT JOIN cancel_request c"
                    " ON c.task_id = t.task_id WHERE t.state = ?",
                    (TaskState.RUNNING.value,),
                ).fetchall()
                rerun_ids = [
                    r["task_id"] for r in rows if r["reason"] is None
                ]
                now = time.time()
                for r in rows:
                    if r["reason"] is None:
                        continue
                    state_value = (
                        TaskState.TIMEOUT.value
                        if r["reason"] == "timeout"
                        else TaskState.CANCELLED.value
                    )
                    error = (
                        "evaluation timed out"
                        if r["reason"] == "timeout"
                        else "cancelled by request"
                    )
                    self._conn.execute(
                        "UPDATE task SET state = ?, started_at = NULL,"
                        " error = ?, finished_at = ? WHERE task_id = ?",
                        (state_value, error, now, r["task_id"]),
                    )
                self._conn.execute(
                    "UPDATE task SET state = ?, started_at = NULL,"
                    " finished_at = NULL, error = NULL"
                    " WHERE state = ? AND task_id NOT IN"
                    " (SELECT task_id FROM cancel_request)",
                    (TaskState.PENDING.value, TaskState.RUNNING.value),
                )
                self._conn.execute("COMMIT")
                return rerun_ids
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    def list_pending(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM task WHERE state = ? ORDER BY created_at",
                (TaskState.PENDING.value,),
            )
            return [self.row_to_task(r) for r in cur.fetchall()]

    # ---------------------------------------------------------------- quotas
    def get_quota_limit(self, scope: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT \"limit\" AS lim FROM quota WHERE scope = ?", (scope,)
            ).fetchone()
            return row["lim"] if row else self.default_quota

    def set_quota(self, scope: str, tenant_id: str, project_id: str | None,
                  limit: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO quota (scope, tenant_id, project_id, \"limit\")"
                " VALUES (?,?,?,?) ON CONFLICT(scope) DO UPDATE SET"
                " \"limit\" = excluded.\"limit\"",
                (scope, tenant_id, project_id, limit),
            )

    def quota_usage(self, scope: str) -> int:
        """Usage for an exact scope (tenant scope counts project-less tasks;
        use :meth:`tenant_usage` for total tasks owned by a tenant)."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) AS c FROM task WHERE scope = ?", (scope,)
            ).fetchone()["c"]

    def tenant_usage(self, tenant_id: str) -> int:
        """Total number of tasks owned by a tenant (all projects included)."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) AS c FROM task WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()["c"]

    def get_quota_info(self, scope: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM quota WHERE scope = ?", (scope,)
            ).fetchone()
            if row is None:
                return None
            return {
                "scope": row["scope"],
                "tenant_id": row["tenant_id"],
                "project_id": row["project_id"],
                "limit": row["limit"],
                "used": self.quota_usage(scope),
            }

    def scopes_with_usage(self) -> list[dict[str, Any]]:
        """Union of configured quota scopes and scopes that have tasks.

        Emulates a FULL OUTER JOIN (unavailable in SQLite) so the listing
        includes configured-but-unused limits as well as implicit limits.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT scope, tenant_id, project_id,"
                " SUM(task_count) AS used,"
                " MAX(lim) AS lim FROM ("
                " SELECT t.scope AS scope, t.tenant_id AS tenant_id,"
                " t.project_id AS project_id, 1 AS task_count,"
                " COALESCE(q.\"limit\", ?) AS lim"
                " FROM task t LEFT JOIN quota q ON q.scope = t.scope"
                " UNION ALL"
                " SELECT scope, tenant_id, project_id, 0 AS task_count,"
                " \"limit\" AS lim FROM quota"
                ") GROUP BY scope ORDER BY scope",
                (self.default_quota,),
            )
            return [dict(r) for r in cur.fetchall()]

    def delete_task(self, task_id: str) -> None:
        """Test/admin helper: remove a task and its dedup entry."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._get_task_row(task_id)
                if row is not None:
                    self._conn.execute(
                        "DELETE FROM cancel_request WHERE task_id = ?",
                        (task_id,),
                    )
                    self._conn.execute(
                        "DELETE FROM dedup_index WHERE task_id = ?", (task_id,)
                    )
                    self._conn.execute(
                        "DELETE FROM task WHERE task_id = ?", (task_id,)
                    )
                self._conn.execute("COMMIT")
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise
