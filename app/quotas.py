"""Tenant/project quota service.

Quota semantics
---------------
* Quota is charged exactly once at successful submission and never refunded.
  Usage is derived from task rows, so it survives restarts without counters
  that could drift or double-count.
* A project-scoped task is counted at **two** levels: the owning tenant
  (total tasks across all of that tenant's projects) and the project scope.
  Submission must fit within both limits; the database transaction performs
  the authoritative atomic check.
* A limit change only affects future submissions; queued/running tasks are
  unaffected.
"""
from __future__ import annotations

from app.models import QuotaInfo, quota_scope
from app.storage import Storage


class QuotaService:
    def __init__(self, storage: Storage, default_limit: int = 100):
        self._storage = storage
        self.default_limit = default_limit

    def set_limit(
        self, tenant_id: str, project_id: str | None, limit: int
    ) -> QuotaInfo:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        scope = quota_scope(tenant_id, project_id)
        self._storage.set_quota(scope, tenant_id, project_id, limit)
        return self.get_info(scope)

    def get_info(self, scope: str) -> QuotaInfo:
        tenant_id, _, project_id = scope.partition(":")
        project_id = project_id or None
        row = self._storage.get_quota_info(scope)
        limit = row["limit"] if row else self.default_limit
        if project_id is None:
            used = self._storage.tenant_usage(tenant_id)
        else:
            used = self._storage.quota_usage(scope)
        return QuotaInfo(
            scope=scope,
            tenant_id=tenant_id,
            project_id=project_id,
            limit=limit,
            used=used,
        )

    def list_infos(self) -> list[QuotaInfo]:
        """One entry per (tenant, project) scope that has tasks, plus any
        explicitly configured tenant scope."""
        infos: list[QuotaInfo] = []
        for row in self._storage.scopes_with_usage():
            scope = row["scope"]
            if row["project_id"] is None:
                used = self._storage.tenant_usage(row["tenant_id"])
            else:
                used = row["used"]
            infos.append(QuotaInfo(
                scope=scope,
                tenant_id=row["tenant_id"],
                project_id=row["project_id"],
                limit=row["lim"],
                used=used,
            ))
        return infos
