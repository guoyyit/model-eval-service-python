"""Deterministic fingerprinting of evaluation requests.

The fingerprint decides task deduplication and result caching. It must
cover everything that determines the *result* of an evaluation:
model version, samples, metric configuration and baseline. Execution
parameters (tenant, timeout) are deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json

from .models import EvalRequest


def canonical_json(obj: object) -> str:
    """Serialize an object to a canonical JSON string.

    Sorted keys and tight separators make the output stable across
    processes and Python versions, which is what makes fingerprints
    reproducible.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_fingerprint(request: EvalRequest) -> str:
    """Compute a deterministic fingerprint for an evaluation request.

    Different model versions or different metric configurations produce
    different fingerprints, so their cached results can never be mixed.
    """
    payload = {
        "model_version": request.model_version,
        "samples": [s.model_dump(mode="json") for s in request.samples],
        "metrics": [m.model_dump(mode="json") for m in request.metrics],
        "baseline": request.baseline,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
