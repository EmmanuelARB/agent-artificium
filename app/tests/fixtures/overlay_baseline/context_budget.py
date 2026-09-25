"""Count input and check capacity without changing generation settings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config
from .engine import Engine, EngineError, PreparedRequest
from .memory import TokenEstimator


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


def minimum_generation_room(config: Config) -> int:
    """Minimum room to proceed, never a maximum length for the response."""
    return max(config.max_output_tokens or 0, (config.reasoning_budget_tokens or 0) + 256)


def margin(config: Config, count: TokenCount) -> int:
    # Even native counts can differ slightly from a provider's final accounting.
    return max(256, config.context_window_tokens // (100 if count.source == "provider" else 20))


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
    usage = reply.usage
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", usage.get("promptTokenCount", usage.get("prompt_eval_count"))))
    generated = usage.get("completion_tokens", usage.get("output_tokens", usage.get("eval_count")))
    if generated is None and isinstance(usage.get("candidatesTokenCount"), int):
        generated = usage["candidatesTokenCount"] + (usage.get("thoughtsTokenCount") or 0)
    if isinstance(prompt, int) and isinstance(generated, int):
        if prompt + generated >= config.context_window_tokens - max(256, config.context_window_tokens // 100):
            return True
    return False
