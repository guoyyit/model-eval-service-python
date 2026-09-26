"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.service import EvalService


@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    return str(tmp_path / "eval.db")


@pytest.fixture
def service(tmp_db_path) -> EvalService:
    svc = EvalService(
        db_path=tmp_db_path,
        max_workers=4,
        default_quota=100,
        default_timeout=30.0,
    )
    yield svc
    svc.close()


@pytest.fixture
def client(service: EvalService) -> TestClient:
    app = create_app(service=service)
    with TestClient(app) as c:
        yield c
