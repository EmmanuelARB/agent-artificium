"""A bounded, tool-free summarizer. It never runs an Artificium life-loop."""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from .config import Config
from .context_budget import TokenCount, available_output, check_input, context_exhausted, measure
from .engine import EngineError, make_engine
from .filesystem import json_dumps, sortable_id
from .memory import TokenEstimator
from .prompts import PromptPack
from .records import Records


MAX_RECOVERY_ATTEMPTS = 3


def repairable(error: EngineError, config: Config, count: TokenCount | None) -> bool:
    """Only failures that changing conversation input might fix."""
    if error.status in {401, 402, 403, 404, 405, 429} or (error.status or 0) >= 500:
        return False
    return context_exhausted(error, config, count) or error.kind in {
        "tokenization", "template", "vision", "empty", "response",
    }


def helper_config(config: Config) -> Config:
    if config.adapter == "custom_json":
        raise EngineError("Emergency summarization requires a built-in API adapter; custom JSON history was retained.", kind="context")
    # Keep routing and the model's template, but no tools, conversation IDs,
    # continuation IDs, response schemas, or other life-loop request options.
    options = {key: copy.deepcopy(config.request_options[key]) for key in (
        "provider", "service_tier", "chat_template", "chat_template_kwargs",
    ) if key in config.request_options}
    return replace(
        config, request_options=options, reasoning_effort=None,
        reasoning_budget_tokens=None, reasoning_mode=None,
        stop_sequences=[], auto_repair=False,
    )


def source_excerpt(history: list[dict[str, Any]], byte_limit: int) -> tuple[str, bool]:
    # No image loading. Image paths and tool results remain inert source data.
    raw = json_dumps(history).encode("utf-8")
    if len(raw) <= byte_limit:
        return raw.decode("utf-8"), False
    start = byte_limit // 3
    text = (raw[:start].decode("utf-8", errors="ignore")
            + "\n[Middle of record omitted; consult the original request log.]\n"
            + raw[-(byte_limit - start):].decode("utf-8", errors="ignore"))
    return text, True


def summarize(*, config: Config, api_key: str | None, history: list[dict[str, Any]],
              original_log: str, attempt: int, prompts: PromptPack,
              records: Records) -> tuple[str, bool]:
    helper = helper_config(config)
    engine = make_engine(helper, api_key)
    engine.request_attempts = 1
    request_id = sortable_id("emergency_summary_")
    messages: list[dict[str, Any]] = []
    parameters: dict[str, Any] = {}
    try:
        # Leave at least half the context for the summary. Shrinking here only
        # repeats token counting, never inference or tool execution.
        byte_limit = int(config.context_window_tokens * config.chars_per_token / 2)
        while True:
            excerpt, omitted = source_excerpt(history, byte_limit)
            messages = [
                {"role": "system", "content": prompts.runtime("emergency_summary")},
                {"role": "user", "content": json_dumps({
                    "original_request_log": original_log, "excerpt_omitted_material": omitted,
                    "execution_record": excerpt,
                })},
            ]
            prepared = engine.prepare(messages)
            try:
                count = measure(engine, prepared, messages, TokenEstimator(config.chars_per_token))
                if count.source == "estimate":
                    count = TokenCount(max(count.tokens, len(json_dumps(messages).encode("utf-8"))), "estimate")
                if count.tokens <= config.context_window_tokens // 2:
                    break
            except EngineError as exc:
                if exc.kind != "context":
                    raise
            byte_limit //= 2
            if byte_limit < 128:
                raise EngineError("The summary helper's input cannot fit; original records were retained.", kind="context")
        room = available_output(helper, count)
        model_limit = config.model_capabilities.get("max_output_tokens")
        if not isinstance(model_limit, int) or model_limit < 1:
            model_limit = room
        # This limit belongs only to the isolated helper, and scales with its
        # actual remaining context. It is not a fixed summary-length default.
        helper = replace(helper, max_output_tokens=min(config.max_output_tokens or room, model_limit, room))
        engine.config = helper
        prepared = engine.prepare(messages)
        check_input(helper, count)
        parameters = prepared.safe_summary()
        parameters["token_count"] = {"tokens": count.tokens, "source": count.source}
        parameters["repair_attempt"] = attempt
        reply = engine.complete_prepared(prepared)
        records.model_exchange(request_id=request_id, messages=messages,
                               request_parameters=parameters, response=vars(reply))
        if reply.finish_reason in {"length", "incomplete", "max_tokens", "MAX_TOKENS"} or len(reply.content.strip()) < 40:
            raise EngineError("The emergency summary was empty, too short, or truncated; original working memory is retained.", kind="context", reply=reply)
        return reply.content.strip(), omitted
    except EngineError as exc:
        records.model_exchange(request_id=request_id, messages=messages,
                               request_parameters=parameters, error=str(exc),
                               response=vars(exc.reply) if exc.reply else None)
        raise
