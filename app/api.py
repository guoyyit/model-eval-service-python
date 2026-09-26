"""HTTP API for the evaluation service.

The service instance is stored on the FastAPI app state (``app.state.service``)
so tests can inject an isolated, temporary-database instance via
:func:`create_app`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from app.models import (
    CancelResponse,
    EvalRequest,
    QuotaInfo,
    QuotaUpdate,
    SampleScore,
    SubmitResponse,
    TaskResult,
    TaskState,
    TaskStatus,
)
from app.service import (
    EvalService,
    TaskNotFoundError,
)
from app.storage import QuotaExceededError

router = APIRouter(prefix="/api/v1", tags=["eval"])


class MetricsResponse(BaseModel):
    metrics: list[str]


def get_service(request: Request) -> EvalService:
    service: EvalService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="service not initialized")
    return service


def _require_tenant(task: TaskStatus, tenant_id: str) -> None:
    if task.tenant_id != tenant_id:
        # Do not reveal the existence of another tenant's task.
        raise HTTPException(status_code=404, detail="task not found")


def _state_filter(value: str | None) -> frozenset[TaskState] | None:
    if value is None:
        return None
    try:
        return frozenset({TaskState(value)})
    except ValueError:
        raise HTTPException(status_code=400, detail=f"unknown state: {value}")


@router.post("/evaluations", response_model=SubmitResponse,
             status_code=status.HTTP_201_CREATED)
def submit_evaluation(
    payload: EvalRequest,
    service: EvalService = Depends(get_service),
) -> SubmitResponse:
    try:
        return service.submit(payload)
    except QuotaExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "scope": exc.scope,
                "used": exc.used,
                "limit": exc.limit,
            },
        ) from exc


@router.get("/evaluations/{task_id}", response_model=TaskStatus)
def get_evaluation(
    task_id: str,
    tenant_id: str,
    project_id: str | None = None,
    service: EvalService = Depends(get_service),
) -> TaskStatus:
    try:
        task = service.get_status(task_id)
    except TaskNotFoundError:
        raise HTTPException(status_code=404, detail="task not found") from None
    _require_tenant(task, tenant_id)
    if project_id is not None and task.project_id != project_id:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@router.get("/evaluations/{task_id}/result", response_model=TaskResult)
def get_result(
    task_id: str,
    tenant_id: str,
    service: EvalService = Depends(get_service),
) -> TaskResult:
    try:
        task = service.get_status(task_id)
    except TaskNotFoundError:
        raise HTTPException(status_code=404, detail="task not found") from None
    _require_tenant(task, tenant_id)
    if task.state is not TaskState.SUCCEEDED:
        raise HTTPException(
            status_code=409,
            detail={"error": "not_succeeded", "state": task.state.value,
                    "reason": task.error},
        )
    try:
        return service.get_result(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/evaluations/{task_id}/samples",
            response_model=list[SampleScore])
def get_sample_scores(
    task_id: str,
    tenant_id: str,
    service: EvalService = Depends(get_service),
) -> list[SampleScore]:
    """Sample-level scores and per-sample failure reasons."""
    try:
        task = service.get_status(task_id)
    except TaskNotFoundError:
        raise HTTPException(status_code=404, detail="task not found") from None
    _require_tenant(task, tenant_id)
    if task.result is None:
        raise HTTPException(
            status_code=409,
            detail={"error": "no_result", "state": task.state.value},
        )
    return task.result.sample_scores


@router.post("/evaluations/{task_id}/cancel", response_model=CancelResponse)
def cancel_evaluation(
    task_id: str,
    tenant_id: str,
    service: EvalService = Depends(get_service),
) -> CancelResponse:
    try:
        task = service.get_status(task_id)
    except TaskNotFoundError:
        raise HTTPException(status_code=404, detail="task not found") from None
    _require_tenant(task, tenant_id)
    updated = service.cancel(task_id)
    return CancelResponse(task_id=task_id, state=updated.state)


@router.get("/tenants/{tenant_id}/evaluations",
            response_model=list[TaskStatus])
def list_tenant_evaluations(
    tenant_id: str,
    state: str | None = None,
    service: EvalService = Depends(get_service),
) -> list[TaskStatus]:
    wanted = _state_filter(state)
    tasks = service.list_tasks(tenant_id)
    if wanted is not None:
        tasks = [t for t in tasks if t.state in wanted]
    return tasks


@router.put("/quotas", response_model=QuotaInfo)
def put_quota(
    payload: QuotaUpdate,
    service: EvalService = Depends(get_service),
) -> QuotaInfo:
    return service.set_quota(payload.tenant_id, payload.project_id, payload.limit)


@router.get("/quotas", response_model=list[QuotaInfo])
def list_quotas(service: EvalService = Depends(get_service)) -> list[QuotaInfo]:
    return service.list_quotas()


@router.get("/metrics", response_model=MetricsResponse)
def list_metrics() -> MetricsResponse:
    from app.metrics import available_metrics

    return MetricsResponse(metrics=available_metrics())


def create_app(
    *,
    service: EvalService | None = None,
    db_path: str = "eval.db",
    max_workers: int = 4,
    default_quota: int = 100,
    default_timeout: float | None = 60.0,
) -> FastAPI:
    """Build the FastAPI app, injecting a service (tests) or creating one."""
    app = FastAPI(title="Local Model Evaluation Service")

    @app.get("/", tags=["meta"])
    def health() -> dict[str, str]:
        return {"service": "model-eval", "status": "ok"}

    app.include_router(router)

    if service is not None:
        app.state.service = service
        app.state.owns_service = False
    else:
        app.state.service = EvalService(
            db_path=db_path,
            max_workers=max_workers,
            default_quota=default_quota,
            default_timeout=default_timeout,
        )
        app.state.owns_service = True

    @app.on_event("shutdown")
    def _shutdown() -> None:  # pragma: no cover - exercised via server lifecycle
        if getattr(app.state, "owns_service", False):
            app.state.service.close()

    return app
