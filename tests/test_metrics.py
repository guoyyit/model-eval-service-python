"""Metric determinism and failure accounting."""

import threading

from evalsvc.metrics import get_metric, simulate_model_output
from evalsvc.models import MetricSpec, Sample, TaskStatus
from evalsvc.runner import run_evaluation

from .conftest import make_request


def test_simulated_model_is_deterministic():
    for version in ("v1", "v2", "gpt-x"):
        for text in ("hello", "world", "1.0"):
            assert simulate_model_output(version, text) == simulate_model_output(
                version, text
            )


def test_repeated_runs_produce_identical_results():
    req = make_request(
        metrics=[MetricSpec(name="exact_match"), MetricSpec(name="token_f1")],
        baseline={"exact_match": 0.5},
    )
    r1 = run_evaluation(req, "fp", threading.Event(), None)
    r2 = run_evaluation(req, "fp", threading.Event(), None)
    assert r1.model_dump_json() == r2.model_dump_json()


def test_failed_samples_are_recorded_not_counted(service):
    # numeric_close fails on non-numeric model output -> sample failure
    req = make_request(
        samples=[
            Sample(id="ok", input="1.0", expected="1.0"),
            Sample(id="bad", input="hello", expected="2.0"),
        ],
        metrics=[MetricSpec(name="numeric_close")],
    )
    # make the failure deterministic: 'hello' never maps to a numeric output
    task, _ = service.submit(req)
    final = service.wait_terminal(task.task_id)
    assert final.status == TaskStatus.SUCCEEDED
    result = final.result
    assert "bad" in result.failed_samples
    assert "not numeric" in result.failure_reasons["bad"]
    bad_score = next(s for s in result.sample_scores if s.sample_id == "bad")
    assert bad_score.error is not None
    assert bad_score.scores == {}, "failed sample must carry no scores"
    # aggregate only reflects successful samples
    ok_score = next(s for s in result.sample_scores if s.sample_id == "ok")
    if "numeric_close" in ok_score.scores:
        assert result.aggregate["numeric_close"] == ok_score.scores["numeric_close"]


def test_sample_without_expected_fails_for_reference_metrics(service):
    req = make_request(
        samples=[Sample(id="noexp", input="hello")],
        metrics=[MetricSpec(name="exact_match")],
    )
    task, _ = service.submit(req)
    final = service.wait_terminal(task.task_id)
    assert final.status == TaskStatus.SUCCEEDED
    assert final.result.failed_samples == ["noexp"]
    assert "expected" in final.result.failure_reasons["noexp"]
    # zero successful samples -> aggregate defaults to 0.0, not an exception
    assert final.result.aggregate == {"exact_match": 0.0}


def test_unknown_metric_rejected_at_submit(service):
    import pytest

    from evalsvc.metrics import MetricError

    with pytest.raises(MetricError):
        service.submit(make_request(metrics=[MetricSpec(name="nope")]))


def test_aggregate_reflects_all_samples(service):
    req = make_request(
        samples=[Sample(id=f"s{i}", input="x", expected="x") for i in range(10)],
        metrics=[MetricSpec(name="exact_match")],
    )
    task, _ = service.submit(req)
    final = service.wait_terminal(task.task_id)
    result = final.result
    assert len(result.sample_scores) == 10
    scored = [s.scores["exact_match"] for s in result.sample_scores if not s.error]
    assert abs(result.aggregate["exact_match"] - sum(scored) / len(scored)) < 1e-12


def test_baseline_delta_is_computed(service):
    req = make_request(baseline={"exact_match": 0.0})
    task, _ = service.submit(req)
    result = service.wait_terminal(task.task_id).result
    assert result.baseline == {"exact_match": 0.0}
    assert result.delta_vs_baseline["exact_match"] == result.aggregate["exact_match"]


def test_metric_functions_are_pure():
    em = get_metric("exact_match")
    assert em("a", "a", {}) == 1.0
    assert em("a", "b", {}) == 0.0
    f1 = get_metric("token_f1")
    assert f1("the cat", "the cat", {}) == 1.0
    assert 0.0 < f1("the cat sat", "the dog", {}) < 1.0
    lr = get_metric("length_ratio")
    assert lr("abcd", "ab", {}) == 0.5
