"""HTTP API surface tests."""

import pytest
from fastapi.testclient import TestClient

from evalsvc.app import create_app
from evalsvc.service import EvalService


@pytest.fixture()
def client(tmp_path):
    svc = EvalService(tmp_path / "api.db", max_workers=2, default_quota=3)
    with TestClient(create_app(svc)) as c:
        c.eval_service = svc  # type: ignore[attr-defined]
        yield c


def payload(**overrides):
    base = {
        "tenant": "acme",
        "model_version": "v1",
        "samples": [{"id": "s1", "input": "hello", "expected": "hello"}],
        "metrics": [{"name": "exact_match"}],
        "baseline": {"exact_match": 0.5},
    }
    base.update(overrides)
    return base


def test_submit_get_and_dedup(client):
    r1 = client.post("/evaluations", json=payload())
    assert r1.status_code == 200
    body = r1.json()
    assert body["created"] is True
    tid = body["task"]["task_id"]

    r2 = client.post("/evaluations", json=payload())
    assert r2.json()["created"] is False
    assert r2.json()["task"]["task_id"] == tid

    client.eval_service.wait_terminal(tid)
    got = client.get(f"/evaluations/{tid}")
    assert got.status_code == 200
    task = got.json()
    assert task["status"] == "succeeded"
    assert task["result"]["aggregate"] == {"exact_match": 1.0}
    assert task["result"]["delta_vs_baseline"] == {"exact_match": 0.5}
    assert task["result"]["failed_samples"] == []


def test_get_unknown_task_404(client):
    assert client.get("/evaluations/missing").status_code == 404


def test_unknown_metric_400(client):
    r = client.post("/evaluations", json=payload(metrics=[{"name": "nope"}]))
    assert r.status_code == 400


def test_quota_endpoints_and_429(client):
    q = client.get("/tenants/acme/quota").json()
    assert q == {"tenant": "acme", "limit": 3, "used": 0, "remaining": 3}

    for i in range(3):
        r = client.post("/evaluations", json=payload(model_version=f"v{i}"))
        assert r.status_code == 200
    r = client.post("/evaluations", json=payload(model_version="v3"))
    assert r.status_code == 429

    q = client.get("/tenants/acme/quota").json()
    assert q["used"] == 3 and q["remaining"] == 0

    client.put("/tenants/acme/quota", json={"limit": 4})
    r = client.post("/evaluations", json=payload(model_version="v3"))
    assert r.status_code == 200


def test_cancel_endpoint(client):
    slow = payload(
        model_version="slow",
        sample_delay_seconds=0.05,
        samples=[{"id": f"s{i}", "input": "x", "expected": "x"} for i in range(50)],
    )
    tid = client.post("/evaluations", json=slow).json()["task"]["task_id"]
    r = client.post(f"/evaluations/{tid}/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    final = client.eval_service.wait_terminal(tid)
    assert final.result is None
    assert client.post("/evaluations/missing/cancel").status_code == 404


def test_list_filter_by_tenant(client):
    client.post("/evaluations", json=payload(tenant="a"))
    client.post("/evaluations", json=payload(tenant="b"))
    assert len(client.get("/evaluations", params={"tenant": "a"}).json()) == 1
    assert len(client.get("/evaluations").json()) == 2
