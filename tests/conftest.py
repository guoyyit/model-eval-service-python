import pytest

from evalsvc.models import EvalRequest, MetricSpec, Sample
from evalsvc.service import EvalService


@pytest.fixture()
def service(tmp_path):
    svc = EvalService(tmp_path / "eval.db", max_workers=4, default_quota=100)
    svc.start()
    yield svc
    svc.shutdown()


def make_request(**overrides) -> EvalRequest:
    base = {
        "tenant": "t1",
        "model_version": "v1",
        "samples": [
            Sample(id="s1", input="hello", expected="hello"),
            Sample(id="s2", input="world", expected="WORLD"),
        ],
        "metrics": [MetricSpec(name="exact_match")],
    }
    base.update(overrides)
    return EvalRequest(**base)
