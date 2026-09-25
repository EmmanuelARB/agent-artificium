"""Real/estimate token calibration (B3).

A character-based estimate is corrected by an EMA of (provider-reported real
input tokens / raw character estimate) observed on completed requests,
clamped so one bad sample cannot swing the ratio wildly. Persisted per
(provider, model) in the runtime directory so a restart keeps it, and so
switching models discards a now-irrelevant ratio.

This is a small, standalone module (rather than living inside memory.py,
where it originated) specifically so that memory.py, context_budget.py, and
runtime.py can each depend on it directly: a workspace overlay of just one of
those modules should not have to also know about calibration to keep the
rest of the harness importable. memory.py still re-exports these names for
any code (including tests) written against the pre-split layout.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from .filesystem import Paths, atomic_write_json, read_json, utc_now


CALIBRATION_MIN_RATIO = 0.8
CALIBRATION_MAX_RATIO = 2.0
CALIBRATION_EMA_ALPHA = 0.3


@dataclass(frozen=True)
class CalibrationState:
    ratio: float = 1.0
    samples: int = 0
    provider: str = ""
    model: str = ""
    updated_at: str = ""


def _calibration_path(paths: Paths) -> Path:
    return paths.runtime / "token_calibration.json"


def load_calibration(paths: Paths, provider: str, model: str) -> CalibrationState:
    """Read the persisted real/estimate ratio; a provider or model change resets it."""
    value = read_json(_calibration_path(paths), {})
    if (
        isinstance(value, dict)
        and value.get("provider") == provider
        and value.get("model") == model
        and isinstance(value.get("ratio"), (int, float))
        and not isinstance(value.get("ratio"), bool)
    ):
        return CalibrationState(
            ratio=float(value["ratio"]),
            samples=int(value.get("samples", 0) or 0),
            provider=provider,
            model=model,
            updated_at=str(value.get("updated_at") or ""),
        )
    return CalibrationState(provider=provider, model=model)


def update_calibration(
    paths: Paths,
    *,
    provider: str,
    model: str,
    real_tokens: int | None,
    raw_estimate: int,
) -> CalibrationState:
    """Fold one (real, estimate) pair from a completed request into the ratio."""
    if real_tokens is None or raw_estimate <= 0:
        return load_calibration(paths, provider, model)
    prior = load_calibration(paths, provider, model)
    sample_ratio = max(
        CALIBRATION_MIN_RATIO, min(CALIBRATION_MAX_RATIO, real_tokens / raw_estimate)
    )
    ratio = (
        sample_ratio
        if prior.samples <= 0
        else CALIBRATION_EMA_ALPHA * sample_ratio + (1 - CALIBRATION_EMA_ALPHA) * prior.ratio
    )
    ratio = max(CALIBRATION_MIN_RATIO, min(CALIBRATION_MAX_RATIO, ratio))
    state = CalibrationState(
        ratio=ratio, samples=prior.samples + 1, provider=provider, model=model,
        updated_at=utc_now(),
    )
    atomic_write_json(_calibration_path(paths), dataclasses.asdict(state))
    return state
