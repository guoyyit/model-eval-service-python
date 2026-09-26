"""Deterministic metric registry and simulated model outputs.

There is no real model behind this local service; instead a deterministic
pseudo-model maps (model_version, input) -> output. The mapping is a pure
function of its arguments, so identical requests always score identically
and different model versions genuinely produce different outputs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

MetricFn = Callable[[str, str | None, dict[str, Any]], float]

_REGISTRY: dict[str, MetricFn] = {}


class MetricError(ValueError):
    """Raised when a metric cannot score a sample (recorded as failure)."""


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    """Register a metric function under a name."""

    def deco(fn: MetricFn) -> MetricFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def get_metric(name: str) -> MetricFn:
    """Look up a metric function by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise MetricError(f"unknown metric: {name}") from None


def metric_names() -> list[str]:
    """Return all registered metric names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Simulated model
# ---------------------------------------------------------------------------

_TRANSFORMS: tuple[Callable[[str], str], ...] = (
    lambda s: s,  # identity
    str.upper,
    str.lower,
    lambda s: s[::-1],  # reversed
)


def simulate_model_output(model_version: str, sample_input: str) -> str:
    """Deterministically simulate a model output for a version and input.

    A hash of (version, input) selects a transformation, so each model
    version has its own stable "behaviour" and outputs are reproducible.
    """
    digest = hashlib.sha256(f"{model_version}::{sample_input}".encode("utf-8")).digest()
    transform = _TRANSFORMS[digest[0] % len(_TRANSFORMS)]
    return transform(sample_input)


# ---------------------------------------------------------------------------
# Built-in metrics (pure functions of their arguments)
# ---------------------------------------------------------------------------


def _require_expected(expected: str | None) -> str:
    if expected is None:
        raise MetricError("metric requires 'expected' but sample has none")
    return expected


@register("exact_match")
def _exact_match(output: str, expected: str | None, params: dict[str, Any]) -> float:
    return 1.0 if output == _require_expected(expected) else 0.0


@register("contains")
def _contains(output: str, expected: str | None, params: dict[str, Any]) -> float:
    return 1.0 if _require_expected(expected) in output else 0.0


@register("length_ratio")
def _length_ratio(output: str, expected: str | None, params: dict[str, Any]) -> float:
    exp = _require_expected(expected)
    if not output and not exp:
        return 1.0
    longer = max(len(output), len(exp))
    return 0.0 if longer == 0 else min(len(output), len(exp)) / longer


@register("numeric_close")
def _numeric_close(output: str, expected: str | None, params: dict[str, Any]) -> float:
    exp = _require_expected(expected)
    tol = float(params.get("tolerance", 1e-6))
    try:
        got = float(output.strip())
    except ValueError:
        raise MetricError(f"output is not numeric: {output!r}") from None
    try:
        want = float(exp.strip())
    except ValueError:
        raise MetricError(f"expected is not numeric: {exp!r}") from None
    return 1.0 if abs(got - want) <= tol else 0.0


@register("token_f1")
def _token_f1(output: str, expected: str | None, params: dict[str, Any]) -> float:
    exp = _require_expected(expected)
    out_tokens = output.split()
    exp_tokens = exp.split()
    if not out_tokens and not exp_tokens:
        return 1.0
    if not out_tokens or not exp_tokens:
        return 0.0
    common: dict[str, int] = {}
    for tok in out_tokens:
        if tok in exp_tokens:
            common[tok] = common.get(tok, 0) + 1
    # count overlap respecting multiplicity
    overlap = 0
    remaining: dict[str, int] = {}
    for tok in exp_tokens:
        remaining[tok] = remaining.get(tok, 0) + 1
    for tok in out_tokens:
        if remaining.get(tok, 0) > 0:
            overlap += 1
            remaining[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(out_tokens)
    recall = overlap / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)
