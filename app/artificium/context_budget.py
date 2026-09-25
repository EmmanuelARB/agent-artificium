"""Count input and check capacity without changing generation settings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import Config
from .engine import Engine, EngineError, PreparedRequest
from .usage import normalize_usage

if TYPE_CHECKING:
    # Only used in type annotations below (postponed by `from __future__
    # import annotations`), so this never needs to be a real import: a
    # workspace overlay of just memory.py (old or new) cannot break this
    # module's own import, and `apply_calibration` only ever reads
    # `calibration.samples` / `.ratio` (duck-typed, no class check).
    from .memory import CalibrationState, TokenEstimator


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    source: str


def measure(engine: Engine, prepared: PreparedRequest,
            messages: list[dict[str, Any]], estimator: TokenEstimator) -> TokenCount:
    count = engine.count_input_tokens(prepared)
    if count is not None:
        return TokenCount(count, "provider")
    return TokenCount(estimator.messages(messages), "estimate")


def apply_calibration(count: TokenCount, calibration: CalibrationState) -> TokenCount:
    """Correct a character-based estimate with the persisted real/estimate ratio (B3).

    Only ``"estimate"`` counts are adjusted: a provider preflight count is
    already exact, and an already-``"calibrated"`` count must not be corrected
    twice.
    """
    if count.source != "estimate" or calibration.samples <= 0:
        return count
    return TokenCount(max(1, round(count.tokens * calibration.ratio)), "calibrated")


def skip_provider_count(config: Config, calibrated_tokens: int, *, has_images: bool) -> bool:
    """Whether a calibrated estimate is trustworthy enough to skip the preflight count (B11).

    Requires headroom on two independent measures (the raw serving window and
    the mandatory-offload target with its own margin) and never applies when
    images are present, since their token cost is only a rough guess.
    """
    if has_images:
        return False
    if calibrated_tokens >= int(config.context_window_tokens * 0.60):
        return False
    if config.mandatory_offload:
        threshold = config.working_memory_limit * config.offload_threshold_percent / 100
        buffer = max(256, config.context_window_tokens // 33)
        if calibrated_tokens >= threshold - buffer:
            return False
    return True


def minimum_generation_room(config: Config) -> int:
    """Minimum room to proceed, never a maximum length for the response."""
    return max(config.max_output_tokens or 0, (config.reasoning_budget_tokens or 0) + 256)


# A calibrated estimate is corrected by real usage but still not exact, so it
# keeps a tighter margin than a plain character estimate without claiming the
# certainty of a provider preflight count.
_MARGIN_DIVISORS = {"provider": 100, "calibrated": 33, "estimate": 20}


def margin(config: Config, count: TokenCount) -> int:
    # Even native counts can differ slightly from a provider's final accounting.
    return max(256, config.context_window_tokens // _MARGIN_DIVISORS.get(count.source, 20))


def available_output(config: Config, count: TokenCount) -> int:
    return config.context_window_tokens - count.tokens - margin(config, count)


def check_input(config: Config, count: TokenCount) -> None:
    available = available_output(config, count)
    if available < minimum_generation_room(config):
        raise EngineError(
            f"Input ({count.tokens} tokens, {count.source}) leaves insufficient generation room "
            f"inside the {config.context_window_tokens}-token serving context.",
            kind="context", hint="Offload working memory or enable automatic repair; history is retained.",
        )


def context_exhausted(error: EngineError, config: Config, count: TokenCount | None) -> bool:
    if error.kind == "context":
        return True
    reply = error.reply
    if not reply or reply.finish_reason not in {"length", "incomplete", "max_tokens", "MAX_TOKENS"}:
        return False
    normalized = normalize_usage(reply.usage, reply.raw)
    prompt, generated = normalized["input_tokens"], normalized["output_tokens"]
    if isinstance(prompt, int) and isinstance(generated, int):
        if prompt + generated >= config.context_window_tokens - max(256, config.context_window_tokens // 100):
            return True
    return False
