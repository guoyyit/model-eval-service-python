"""Per-tenant quota management.

Quota *charging* happens atomically inside ``Store.create_task`` so that
dedup and charging can never race; this manager only owns the policy:
per-tenant limits and usage reporting. Limit changes apply only to new
submissions -- tasks already charged keep their ledger entries.
"""

from __future__ import annotations

import threading

from .store import Store


class QuotaManager:
    """Enforces per-tenant quotas; charges are idempotent per task id."""

    def __init__(self, store: Store, default_limit: int) -> None:
        self._store = store
        self._default_limit = default_limit
        self._overrides: dict[str, int] = {}
        self._lock = threading.Lock()

    def set_limit(self, tenant: str, limit: int) -> None:
        """Override the quota limit for a tenant (affects new submissions)."""
        if limit < 0:
            raise ValueError("quota limit must be >= 0")
        with self._lock:
            self._overrides[tenant] = limit

    def limit_for(self, tenant: str) -> int:
        with self._lock:
            return self._overrides.get(tenant, self._default_limit)

    def usage(self, tenant: str) -> int:
        return self._store.quota_usage(tenant)

    def remaining(self, tenant: str) -> int:
        return max(0, self.limit_for(tenant) - self.usage(tenant))
