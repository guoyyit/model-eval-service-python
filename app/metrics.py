"""Deterministic evaluation metric registry.

Every metric is a pure function of ``(prediction, reference, config)``: the
same inputs always yield the same score. New metrics can be registered via
:func:`register_metric`.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

MetricFn = Callable[[str, str, dict], float]


class Metric(Protocol):
    def __call__(self, prediction: str, reference: str, config: dict) -> float: ...


def _normalize(text: str, config: dict) -> str:
    if config.get("strip", False):
        text = text.strip()
    if not config.get("case_sensitive", True):
        text = text.lower()
    return text


def exact_match(prediction: str, reference: str, config: dict) -> float:
    """1.0 when prediction equals reference exactly, else 0.0.

    Config: ``case_sensitive`` (default true), ``strip`` (default false).
    """
    return float(
        _normalize(prediction, config) == _normalize(reference, config)
    )


def token_f1(prediction: str, reference: str, config: dict) -> float:
    """Token-set F1: order-insensitive, duplicate-insensitive F1 score.

    Tokenization is a deterministic whitespace split with the same
    ``case_sensitive``/``strip`` normalization as ``exact_match``. Empty
    prediction and reference together score 1.0; otherwise empty side 0.0.
    """
    pred_tokens = set(_normalize(prediction, config).split())
    ref_tokens = set(_normalize(reference, config).split())
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = len(pred_tokens & ref_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def numeric_abs_error(prediction: str, reference: str, config: dict) -> float:
    """Absolute error between numeric prediction and reference.

    Both sides must parse as floats via ``float()``; otherwise ``ValueError``
    is raised and recorded as a failed sample (never silently scored).
    """
    try:
        pred = float(prediction)
        ref = float(reference)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric value for numeric_abs_error: {exc}") from exc
    return abs(pred - ref)


METRICS: dict[str, MetricFn] = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "numeric_abs_error": numeric_abs_error,
}


def register_metric(name: str, fn: MetricFn) -> None:
    """Register (or replace) a metric function under ``name``."""
    if not name:
        raise ValueError("metric name must be non-empty")
    METRICS[name] = fn


def compute_metric(name: str, prediction: str, reference: str, config: dict) -> float:
    """Compute one metric. Raises KeyError for unknown metric names."""
    try:
        fn = METRICS[name]
    except KeyError:
        raise KeyError(f"unknown metric: {name}") from None
    return fn(prediction, reference, config)


def available_metrics() -> list[str]:
    """Registered metric names, sorted for deterministic listing."""
    return sorted(METRICS)
