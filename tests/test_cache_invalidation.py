"""Cache invalidation: different versions/configs never share results."""

from evalsvc.fingerprint import request_fingerprint
from evalsvc.models import MetricSpec, TaskStatus

from .conftest import make_request


def test_same_input_same_config_same_result(service):
    req = make_request()
    t1, _ = service.submit(req)
    r1 = service.wait_terminal(t1.task_id).result

    # force a fresh execution of an identical request by using a second
    # tenant (same fingerprint space, no shared cache entry)
    t2, _ = service.submit(make_request(tenant="t2"))
    r2 = service.wait_terminal(t2.task_id).result

    assert r1 is not None and r2 is not None
    assert r1.model_dump_json() == r2.model_dump_json(), (
        "identical input and config must reproduce identical results"
    )


def test_different_model_version_does_not_reuse_cache(service):
    t1, _ = service.submit(make_request(model_version="v1"))
    t2, created2 = service.submit(make_request(model_version="v2"))
    assert created2 is True, "different model version must be a new task"
    assert t1.task_id != t2.task_id

    f1 = service.wait_terminal(t1.task_id)
    f2 = service.wait_terminal(t2.task_id)
    assert f1.result.fingerprint != f2.result.fingerprint
    assert f1.result.model_version == "v1"
    assert f2.result.model_version == "v2"


def test_different_metric_config_does_not_reuse_cache(service):
    t1, _ = service.submit(make_request(metrics=[MetricSpec(name="exact_match")]))
    t2, created2 = service.submit(
        make_request(metrics=[MetricSpec(name="exact_match"),
                              MetricSpec(name="token_f1")])
    )
    assert created2 is True
    assert t1.task_id != t2.task_id

    f1 = service.wait_terminal(t1.task_id)
    f2 = service.wait_terminal(t2.task_id)
    assert set(f1.result.aggregate) == {"exact_match"}
    assert set(f2.result.aggregate) == {"exact_match", "token_f1"}


def test_metric_params_are_part_of_fingerprint():
    loose = make_request(metrics=[MetricSpec(name="numeric_close",
                                             params={"tolerance": 0.1})])
    tight = make_request(metrics=[MetricSpec(name="numeric_close",
                                             params={"tolerance": 1e-9})])
    assert request_fingerprint(loose) != request_fingerprint(tight)


def test_execution_params_do_not_affect_fingerprint():
    a = make_request(timeout_seconds=1.0, sample_delay_seconds=0.0)
    b = make_request(timeout_seconds=99.0, sample_delay_seconds=0.5)
    assert request_fingerprint(a) == request_fingerprint(b)


def test_resubmit_after_cancel_creates_fresh_task(service):
    req = make_request(sample_delay_seconds=0.02,
                       samples=[{"id": f"s{i}", "input": "x", "expected": "x"}
                                for i in range(50)])
    t1, _ = service.submit(req)
    service.cancel(t1.task_id)
    final1 = service.wait_terminal(t1.task_id)
    assert final1.status == TaskStatus.CANCELLED

    t2, created = service.submit(req)
    assert created is True, "a cancelled request must be re-submittable"
    assert t2.task_id != t1.task_id
    final2 = service.wait_terminal(t2.task_id)
    assert final2.status == TaskStatus.SUCCEEDED
