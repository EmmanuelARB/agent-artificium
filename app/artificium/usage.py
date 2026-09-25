"""One provider-neutral reading of a model reply's token accounting.

Adapters keep the provider's usage object verbatim for evidence; everything
that reasons about tokens (context budgeting, throughput, the console, watch)
reads it through :func:`normalize_usage` instead of guessing field names.
"""
from __future__ import annotations

from typing import Any


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _first(source: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _int(source.get(key))
        if value is not None:
            return value
    return None


def _details(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    return value if isinstance(value, dict) else {}


def normalize_usage(
    usage: dict[str, Any] | None, raw: dict[str, Any] | None = None
) -> dict[str, int | None]:
    """Return input, output, reasoning, cache and total token counts.

    Every value is ``None`` when the provider did not report it.

    * ``input_tokens`` is the *whole* prompt, cached or not. Anthropic reports
      uncached input separately from cache reads and writes, so they are added.
    * ``output_tokens`` counts every generated token, reasoning included.
    * ``reasoning_tokens`` is the part of the output spent thinking.
    * ``cache_read_tokens`` is the part of the input served from a prompt cache.
    * ``cache_write_tokens`` is the part of the input written to a prompt cache
      (Anthropic only).
    """

    usage = usage if isinstance(usage, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else {}

    cache_read: int | None = None
    cache_write: int | None = None
    reasoning: int | None = None

    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        # Anthropic Messages: input_tokens excludes both cache buckets.
        uncached = _first(usage, "input_tokens") or 0
        cache_read = _first(usage, "cache_read_input_tokens") or 0
        cache_write = _first(usage, "cache_creation_input_tokens") or 0
        input_tokens: int | None = uncached + cache_read + cache_write
        output_tokens = _first(usage, "output_tokens")
    elif "promptTokenCount" in usage or "candidatesTokenCount" in usage:
        # Gemini: thoughts are reported beside, not inside, candidates.
        input_tokens = _first(usage, "promptTokenCount")
        cache_read = _first(usage, "cachedContentTokenCount")
        reasoning = _first(usage, "thoughtsTokenCount")
        candidates = _first(usage, "candidatesTokenCount")
        output_tokens = (
            None if candidates is None and reasoning is None
            else (candidates or 0) + (reasoning or 0)
        )
    elif "prompt_eval_count" in usage or "eval_count" in usage:
        # Ollama native: prompt_eval_count covers only the evaluated (uncached) part.
        input_tokens = _first(usage, "prompt_eval_count")
        output_tokens = _first(usage, "eval_count")
    else:
        # OpenAI Chat Completions, OpenAI Responses, OpenRouter, vLLM, llama.cpp.
        input_tokens = _first(usage, "prompt_tokens", "input_tokens")
        prompt_details = _details(usage, "prompt_tokens_details") or _details(
            usage, "input_tokens_details"
        )
        cache_read = _first(prompt_details, "cached_tokens")
        cache_write = _first(prompt_details, "cache_write_tokens")
        completion_details = _details(usage, "completion_tokens_details") or _details(
            usage, "output_tokens_details"
        )
        output_tokens = _first(usage, "completion_tokens", "output_tokens")
        reasoning = _first(completion_details, "reasoning_tokens")
        if reasoning is None:
            # A standalone total is excluded from the output count by the
            # providers that report it this way.
            separate = _first(usage, "reasoning_tokens", "thinking_tokens")
            if separate is not None:
                reasoning = separate
                output_tokens = (output_tokens or 0) + separate

    if cache_read is None and timings:
        # llama.cpp: prompt_n is what was evaluated, cache_n what was reused.
        cache_read = _first(timings, "cache_n")
        if input_tokens is None:
            evaluated = _first(timings, "prompt_n")
            if evaluated is not None or cache_read is not None:
                input_tokens = (evaluated or 0) + (cache_read or 0)
        if output_tokens is None:
            output_tokens = _first(timings, "predicted_n")

    total = _first(usage, "total_tokens", "totalTokenCount")
    if total is None and (input_tokens is not None or output_tokens is not None):
        total = (input_tokens or 0) + (output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "total_tokens": total,
    }


def cache_hit_ratio(normalized: dict[str, int | None]) -> float | None:
    """Share of the input served from cache, or None when not reported."""

    read = normalized.get("cache_read_tokens")
    total = normalized.get("input_tokens")
    if read is None or not total:
        return None
    return max(0.0, min(1.0, read / total))


def generation_token_count(usage: dict[str, Any] | None) -> int:
    """Count every token the model *generated*, including thinking/reasoning.

    Reasoning (a.k.a. thinking) tokens are decoded like any other output token,
    so excluding them understates the generation rate. Providers disagree on
    where they land:

    * OpenAI-style usage keeps reasoning *inside* ``completion_tokens`` and also
      breaks it out under ``completion_tokens_details.reasoning_tokens`` - so it
      is already counted and must not be added again.
    * Anthropic-style usage keeps thinking inside ``output_tokens`` with no
      separate field - also already counted.
    * Some providers report a separate ``reasoning_tokens`` / ``thinking_tokens``
      total that is *excluded* from the output count - that one must be added.

    We therefore add a standalone reasoning total only when it is not already
    folded into the output count via an OpenAI-style details block.
    """

    usage = usage if isinstance(usage, dict) else {}
    base = usage.get("completion_tokens")
    if base is None:
        base = usage.get("output_tokens")
    base = int(base) if isinstance(base, (int, float)) else 0

    details = usage.get("completion_tokens_details")
    reasoning_already_included = isinstance(details, dict) and isinstance(
        details.get("reasoning_tokens"), (int, float)
    )
    reasoning = None
    for key in ("reasoning_tokens", "thinking_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            reasoning = int(value)
            break
    if reasoning is not None and not reasoning_already_included:
        return base + reasoning
    return base


def throughput_stats(
    usage: dict[str, Any] | None,
    raw: dict[str, Any] | None,
    elapsed: float | None,
) -> dict[str, Any]:
    """Prefill and generation token rates for one completed model request.

    Prefers exact server-reported timings when the provider includes them
    (a llama.cpp-style ``timings`` block with ``prompt_*`` / ``predicted_*``
    fields; ``predicted_n`` already counts thinking tokens). When the provider
    reports no timings, only the total wall time is known, so the
    prefill/generation split is not recoverable; we then report an approximate
    generation rate from the generated-token count (output + any separately
    reported thinking tokens, see :func:`generation_token_count`) over
    ``elapsed`` (seconds). ``source`` records which path produced the numbers.
    """

    usage = usage if isinstance(usage, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    timings = raw.get("timings")
    if isinstance(timings, dict):

        def _rate(per_second: str, n_key: str, ms_key: str) -> float | None:
            value = timings.get(per_second)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
            n, ms = timings.get(n_key), timings.get(ms_key)
            if isinstance(n, (int, float)) and isinstance(ms, (int, float)) and ms > 0:
                return float(n) / (float(ms) / 1000.0)
            return None

        prefill = _rate("prompt_per_second", "prompt_n", "prompt_ms")
        generation = _rate("predicted_per_second", "predicted_n", "predicted_ms")
        if prefill is not None or generation is not None:
            return {
                "prefill_tok_s": prefill,
                "generation_tok_s": generation,
                "source": "server",
            }

    generated = generation_token_count(usage)
    if generated > 0 and elapsed and elapsed > 0:
        return {
            "prefill_tok_s": None,
            "generation_tok_s": float(generated) / float(elapsed),
            "source": "approx",
        }
    return {"prefill_tok_s": None, "generation_tok_s": None, "source": "none"}


def format_throughput(stats: dict[str, Any] | None) -> str:
    """Render a ``" (prefill Y tok/s, generation Z tok/s)"`` suffix.

    Returns an empty string when no rate is known. An approximate generation
    rate (no server timings) is marked with a ``~``.
    """

    if not isinstance(stats, dict):
        return ""
    parts: list[str] = []
    prefill = stats.get("prefill_tok_s")
    if isinstance(prefill, (int, float)):
        parts.append(f"prefill {prefill:.1f} tok/s")
    generation = stats.get("generation_tok_s")
    if isinstance(generation, (int, float)):
        tilde = "" if stats.get("source") == "server" else "~"
        parts.append(f"generation {tilde}{generation:.1f} tok/s")
    return " (" + ", ".join(parts) + ")" if parts else ""
