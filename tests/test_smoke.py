"""Smoke tests: package imports and app wiring."""

from fastapi.testclient import TestClient

import main


def test_imports():
    import app.api
    import app.evaluator
    import app.executor
    import app.metrics
    import app.models
    import app.quotas
    import app.service
    import app.storage  # noqa: F401


def test_root_endpoint():
    with TestClient(main.app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
