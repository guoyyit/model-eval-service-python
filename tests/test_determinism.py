"""Requirement 2/8: deterministic scoring and cache invalidation."""
from __future__ import annotations

from helpers import make_request, wait_terminal

from app.models import Sample, TaskState

SAMPLES = [
    Sample(id="a", reference="hello world", prediction="hello world"),
    Sample(id="b", reference="a b c", prediction="a b d"),
]


def _result(service, req):
    r = service.submit(req)
    status = wait_terminal(service, r.task_id)
    assert status.state is TaskState.SUCCEEDED
    return service.get_result(r.task_id)


def test_same_input_same_config_identical_result(service):
    req = make_request(samples=SAMPLES, metrics=["exact_match", "token_f1"])
    res1 = _result(service, req)
    res2 = _result(service, req)
    assert res1.model_dump() == res2.model_dump()


def test_metric_order_does_not_change_fingerprint(service):
    r1 = service.submit(make_request(samples=SAMPLES,
                                     metrics=["exact_match", "token_f1"]))
    r2 = service.submit(make_request(samples=SAMPLES,
                                     metrics=["token_f1", "exact_match"]))
    assert r1.task_id == r2.task_id


def test_different_model_version_does_not_reuse_cache(service):
    r1 = service.submit(make_request(model_version="v1"))
    r2 = service.submit(make_request(model_version="v2"))
    assert r1.task_id != r2.task_id


def test_different_metric_config_does_not_reuse_cache(service):
    r1 = service.submit(make_request(
        samples=[Sample(id="a", reference="Hello", prediction="hello")],
        config={"case_sensitive": True}))
    r2 = service.submit(make_request(
        samples=[Sample(id="a", reference="Hello", prediction="hello")],
        config={"case_sensitive": False}))
    assert r1.task_id != r2.task_id
    wait_terminal(service, r1.task_id)
    res2 = _result(service, make_request(
        samples=[Sample(id="a", reference="Hello", prediction="hello")],
        config={"case_sensitive": False}))
    assert res2.metrics["exact_match"] == 1.0


def test_different_metric_set_does_not_reuse_cache(service):
    r1 = service.submit(make_request(metrics=["exact_match"]))
    r2 = service.submit(make_request(metrics=["token_f1"]))
    assert r1.task_id != r2.task_id


def test_different_baseline_does_not_reuse_cache(service):
    r_none = service.submit(make_request(
        idempotency_key="k", baseline={"metrics": {"exact_match": 0.5}}))
    r_other = service.submit(make_request(
        idempotency_key="k", baseline={"metrics": {"exact_match": 0.9}}))
    assert r_none.task_id != r_other.task_id


def test_reproducible_across_process_restart(tmp_db_path):
    # First process/lifetime.
    from app.service import EvalService
    svc1 = EvalService(db_path=tmp_db_path, default_timeout=30.0)
    req = make_request(samples=SAMPLES, metrics=["exact_match", "token_f1"])
    r = svc1.submit(req)
    wait_terminal(svc1, r.task_id)
    expected = svc1.get_result(r.task_id).model_dump()
    svc1.close()

    # Reopen: same fingerprint must map to same task with identical result.
    svc2 = EvalService(db_path=tmp_db_path, default_timeout=30.0)
    r2 = svc2.submit(req)
    assert r2.reused is True
    actual = svc2.get_result(r2.task_id).model_dump()
    assert actual == expected
    svc2.close()


def test_comparison_against_inline_baseline(service):
    req = make_request(
        samples=SAMPLES,
        metrics=["exact_match"],
        baseline={"metrics": {"exact_match": 0.0}},
    )
    res = _result(service, req)
    comp = res.comparison
    assert comp is not None
    assert comp["exact_match"].baseline == 0.0
    assert comp["exact_match"].current == 0.5
    assert comp["exact_match"].delta == 0.5
