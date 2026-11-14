"""A short readiness check for the oversight connection.

Separate from connection.py so that a workspace overlay of that module,
written before the oversight connection existed, keeps working.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .config import Config
from .connection import _input_tokens, _progress
from .context_budget import check_input, measure
from .engine import make_engine
from .filesystem import utc_now
from .memory import TokenEstimator


def verify_oversight_connection(config: Config, key: str | None, *,
                                report: Callable[[str], None] = print) -> dict[str, Any]:
    """Prove an oversight connection with one short real request.

    Oversight requests are plain text with no tools, so the check does not need
    the life-loop prompt, the mind, or an image test.
    """
    started = time.monotonic()
    messages = [
        {"role": "system", "content": "This is a connection check for a harness. Answer in one short sentence."},
        {"role": "user", "content": "Reply with the word OK to confirm you received this."},
    ]
    engine = make_engine(config, key)
    prepared = engine.prepare(messages)
    count = measure(engine, prepared, messages, TokenEstimator(config.chars_per_token))
    check_input(config, count)
    report(f"Checking the oversight connection ({config.provider}/{config.model}). This sends a short real request.")
    with _progress(report, "Oversight check"):
        reply = engine.complete_prepared(prepared)
    report("Oversight connection passed.")
    return {"checked_at": utc_now(), "model": config.model, "adapter": config.adapter,
            "context_window_tokens": config.context_window_tokens,
            "input_tokens": _input_tokens(reply) or count.tokens,
            "text_passed": True, "reply_characters": len(reply.content or ""),
            "seconds": round(time.monotonic() - started, 2)}
