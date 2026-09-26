"""Requirement 6/8: randomized cancel/finish race invariants."""
from __future__ import annotations

import random
import threading

from helpers import make_request

from app.models import Sample, TaskState


def test_cancel_race_invariants(service):
    """For many tasks with random cancellation, every terminal state must be
    self-consistent: succeeded iff a result exists; cancelled/timeout never
    carry a result."""
    random.seed(1234)
    task_ids = []
    for i in range(24):
        latency = random.uniform(0.0, 0.12)
        req = make_request(
            tenant_id="fuzz",
            idempotency_key=f"t{i}",
            samples=[Sample(id=str(j), reference="same", prediction="same")
                     for j in range(3)],
            metrics=["exact_match"],
            config={"latency": latency},
        )
        task_ids.append(service.submit(req).task_id)

    def canceller(tid):
        if random.random() < 0.5:
            service.executor.wait(tid, timeout=random.uniform(0.0, 0.06))
            service.cancel(tid)

    threads = [threading.Thread(target=canceller, args=(tid,))
               for tid in task_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for tid in task_ids:
        service.executor.wait(tid, timeout=15)
        st = service.get_status(tid)
        assert st.state in (
            TaskState.SUCCEEDED, TaskState.CANCELLED,
            TaskState.FAILED, TaskState.TIMEOUT,
        ), (tid, st.state)
        if st.state is TaskState.SUCCEEDED:
            assert st.result is not None, f"{tid} succeeded without result"
            assert st.error is None
        else:
            assert st.result is None, (
                f"{tid} state={st.state.value} must not carry a result")
            assert st.error, f"{tid} non-success must explain why"

    # Task rows exactly match submissions (no duplication/loss).
    assert service.storage.tenant_usage("fuzz") == len(task_ids)
