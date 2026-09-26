"""FastAPI application wiring."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .metrics import MetricError, metric_names
from .models import EvalRequest, TaskRecord
from .service import EvalService
from .store import QuotaExceeded


class SubmitResponse(BaseModel):
    task: TaskRecord
    created: bool = Field(description="False when an identical live task was reused")


class QuotaView(BaseModel):
    tenant: str
    limit: int
    used: int
    remaining: int


class QuotaUpdate(BaseModel):
    limit: int = Field(ge=0)


def _service_from_env() -> EvalService:
    return EvalService(
        db_path=os.environ.get("EVAL_DB_PATH", "./data/eval.db"),
        max_workers=int(os.environ.get("EVAL_MAX_WORKERS", "4")),
        default_quota=int(os.environ.get("EVAL_DEFAULT_QUOTA", "100")),
        default_timeout_seconds=float(os.environ.get("EVAL_DEFAULT_TIMEOUT", "60")),
    )


def create_app(service: EvalService | None = None) -> FastAPI:
    """Build the FastAPI app around an EvalService instance."""
    svc = service or _service_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc.start()
        yield
        svc.shutdown()

    app = FastAPI(title="model-eval-service", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(QuotaExceeded)
    async def quota_handler(request: Request, exc: QuotaExceeded) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(MetricError)
    async def metric_handler(request: Request, exc: MetricError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def list_metrics() -> dict[str, list[str]]:
        return {"metrics": metric_names()}

    @app.post("/evaluations", response_model=SubmitResponse)
    def submit(request: EvalRequest) -> SubmitResponse:
        task, created = svc.submit(request)
        return SubmitResponse(task=task, created=created)

    @app.get("/evaluations", response_model=list[TaskRecord])
    def list_evaluations(tenant: str | None = Query(default=None)) -> list[TaskRecord]:
        return svc.list(tenant)

    @app.get("/evaluations/{task_id}", response_model=TaskRecord)
    def get_evaluation(task_id: str) -> TaskRecord:
        task = svc.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task

    @app.post("/evaluations/{task_id}/cancel", response_model=TaskRecord)
    def cancel_evaluation(task_id: str) -> TaskRecord:
        task = svc.cancel(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task

    @app.get("/tenants/{tenant}/quota", response_model=QuotaView)
    def get_quota(tenant: str) -> QuotaView:
        limit = svc.quota.limit_for(tenant)
        used = svc.quota.usage(tenant)
        return QuotaView(tenant=tenant, limit=limit, used=used,
                         remaining=max(0, limit - used))

    @app.put("/tenants/{tenant}/quota", response_model=QuotaView)
    def set_quota(tenant: str, update: QuotaUpdate) -> QuotaView:
        svc.quota.set_limit(tenant, update.limit)
        return get_quota(tenant)

    return app
