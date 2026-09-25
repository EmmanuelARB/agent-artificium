"""Read-only readiness checks using the harness prompt and real engine adapter.

Nothing here starts an agent, loads mutable tools, consumes a notification,
executes model output, or writes working memory.
"""
from __future__ import annotations

import contextlib
import dataclasses
import struct
import tempfile
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .context_budget import TokenCount, check_input, margin, measure, minimum_generation_room
from .engine import EngineError, EngineReply, make_engine
from .filesystem import Paths, read_json, read_jsonl, utc_now
from .memory import TokenEstimator
from .prompts import PromptPack
from .records import Records
from .vision import VisualContext


def preview_messages(paths: Paths, config: Config, *, self_directive: str | None = None,
                     include_images: bool = True) -> list[dict[str, Any]]:
    """Use the same prompt pack, pinned mind and history as the life-loop.

    The state header identifies this as a connection check. A diagnostic
    instruction replaces acting on the wake; returned tool text is never run.
    """
    prompts = PromptPack(paths)
    estimator = TokenEstimator(config.chars_per_token)
    self_text = self_directive or paths.self_file.read_text(encoding="utf-8", errors="replace")
    if len(self_text) > 50_000:
        self_text = self_text[:50_000].rstrip() + f"\n\n[PINNED FILE TRUNCATED: {len(self_text):,} characters; inspect {paths.self_file}]"
    meta = paths.meta_memory.read_text(encoding="utf-8", errors="replace")
    pinned = prompts.runtime("pinned_mind", self_content=self_text.rstrip(), meta_memory_content=meta.rstrip())
    parts = [prompts.always(), pinned, prompts.tool_catalog()]
    history = read_jsonl(paths.working_context)
    visual = VisualContext(paths, config, Records(paths))
    visual_message = visual.request_message() if include_images else None
    images = visual.list() if visual_message else []
    estimated = estimator.text("\n".join(parts)) + estimator.messages(history + ([visual_message] if visual_message else []))
    header = prompts.runtime(
        "state_header", timestamp=utc_now(), wake_reason="connection_check",
        engine_name=f"{config.provider}/{config.model}", context_tokens=estimated,
        context_window_tokens=config.context_window_tokens,
        working_memory_tokens=config.working_memory_limit,
        context_percent=f"{estimated / config.working_memory_limit * 100:.1f}",
        tokens_since_last_notice=0, context_status="connection check",
        current_interaction_or_none="none", pending_event_count=0,
        active_stream_or_none="none", pre_sleep_issued=False,
        last_checkpoint_or_none="none", meta_memory_tokens=estimator.text(meta),
        meta_memory_words=len(meta.split()), meta_memory_guidance_tokens=config.meta_memory_guidance_tokens,
        vision_mode=config.vision, vision_guidance="Check only; do not load or release files.",
        active_image_count=len(images), active_images_or_none=images or "none",
    )
    inputs = []
    state = read_json(paths.initialization_state, {})
    if not isinstance(state, dict) or state.get("status") != "completed":
        inputs.append(prompts.event("first_wake"))
    inputs.append("[CONNECTION CHECK]\nThis is a setup diagnostic, not an active wake. "
                  "Do not perform the orientation or any task. Do not call tools. "
                  "Reply briefly with OK to confirm that you can receive this context.")
    messages = [{"role": "system", "content": "\n\n---\n\n".join([*parts, header])}, *history,
                {"role": config.runtime_message_role,
                 "content": prompts.runtime("input_batch", runtime_records="\n\n---\n\n".join(inputs))}]
    if visual_message:
        messages.append(visual_message)
    return messages


def prompt_tokens(config: Config, key: str | None, messages: list[dict[str, Any]]) -> tuple[int, str]:
    engine = make_engine(config, key)
    prepared = engine.prepare(messages)
    count = measure(engine, prepared, messages, TokenEstimator(config.chars_per_token))
    return count.tokens, count.source


def check_capacity(config: Config, tokens: int, source: str) -> None:
    # Reserve space for a useful answer (including reasoning). Never silently
    # truncate pinned memory/history or claim a larger server allocation.
    count = TokenCount(tokens, "estimate" if source == "estimate" else "provider")
    reserve = minimum_generation_room(config) + margin(config, count)
    if tokens + reserve > config.context_window_tokens:
        approximate = "approximately " if source == "estimate" else ""
        required = tokens + reserve
        suggestion = max(32768, ((required + 8191) // 8192) * 8192)
        hint = f"Use a serving context of at least {required:,} tokens; {suggestion:,} leaves more room for work."
        if config.adapter == "llamacpp":
            hint += f" Restart llama-server with --ctx-size {suggestion} --parallel 1 (if your model and hardware support it), then choose Retry."
        hint += " Your existing context and memory have not been changed."
        raise EngineError(
            f"Artificium needs {approximate}{tokens:,} input tokens plus {reserve:,} for output, "
            f"but this connection is configured for {config.context_window_tokens:,}.",
            kind="context", hint=hint,
        )


def _input_tokens(reply: EngineReply) -> int | None:
    for name in ("prompt_tokens", "input_tokens", "prompt_eval_count", "promptTokenCount"):
        value = reply.usage.get(name)
        if isinstance(value, int) and value > 0:
            return value
    return None


@contextlib.contextmanager
def _progress(report: Callable[[str], None], label: str):
    stop = threading.Event()
    started = time.monotonic()
    def tick():
        while not stop.wait(15):
            report(f"{label}: still waiting ({time.monotonic() - started:.0f}s). Ctrl-C cancels without saving.")
    worker = threading.Thread(target=tick, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=1)


def _test_image(path: Path) -> None:
    """A generated 32px RGB PNG; no dependency or user attachment required."""
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data) & 0xffffffff)
    pixels = b"".join(b"\x00" + bytes((40, 100, 180)) * 32 for _ in range(32))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 32, 32, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def _carries_image(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("data:image/")
    if isinstance(value, dict):
        return any(_carries_image(item) for item in value.values())
    if isinstance(value, list):
        return any(_carries_image(item) for item in value)
    return False


def verify_connection(paths: Paths, config: Config, key: str | None, *,
                      report: Callable[[str], None] = print,
                      self_directive: str | None = None) -> tuple[Config, dict[str, Any]]:
    """Only return readiness after real inference with the complete harness input."""
    started = time.monotonic()
    messages = preview_messages(paths, config, self_directive=self_directive, include_images=False)
    engine = make_engine(config, key)
    prepared = engine.prepare(messages)
    count = measure(engine, prepared, messages, TokenEstimator(config.chars_per_token))
    tokens, source = count.tokens, count.source
    check_capacity(config, tokens, source)
    report(f"Checking Artificium's full prompt: {tokens:,} input tokens ({source}). This sends a real request.")
    with _progress(report, "Model check"):
        check_input(config, count)
        reply = engine.complete_prepared(prepared)
    actual = _input_tokens(reply)
    if reply.raw.get("truncated") is True:
        raise EngineError("The server truncated the connection-check prompt.", kind="context",
                          hint="Increase the server context allocation. Setup will not accept silently lost context.")
    if actual:
        check_capacity(config, actual, "server usage")
        # Ollama truncates overlong input internally. Reject a clear discrepancy
        # when exact tokenization is available; estimates are not exact counts.
        if source == "provider" and actual < tokens * 0.95:
            raise EngineError("The server processed fewer tokens than its tokenizer reported.", kind="context",
                              hint="Check for server-side prompt truncation and increase its context allocation.")
    report("Text connection passed.")
    vision_checked = False
    if config.vision_preference != "no" and config.model_supports_vision is not False:
        report("Checking image input with a generated test image.")
        vision_config = dataclasses.replace(config, vision="yes")
        with tempfile.TemporaryDirectory(prefix="artificium-vision-check-") as directory:
            path = Path(directory) / "check.png"
            _test_image(path)
            image_messages = preview_messages(paths, vision_config, self_directive=self_directive)
            image_messages.append({"role": "user", "content": [
                {"type": "text", "text": "Connection check only: confirm receipt of this test image briefly. Do not call tools."},
                {"type": "artificium_image", "path": str(path), "mime": "image/png", "detail": "low"},
            ]})
            try:
                image_engine = make_engine(vision_config, key)
                if config.adapter == "custom_json" and not _carries_image(image_engine.prepare(image_messages).payload):
                    raise EngineError("This custom JSON contract does not transmit image data.", kind="vision",
                                      hint="Use a contract that maps multimodal messages, or keep image input disabled.")
                with _progress(report, "Image check"):
                    image_prepared = image_engine.prepare(image_messages)
                    image_count = measure(image_engine, image_prepared, image_messages, TokenEstimator(config.chars_per_token))
                    check_input(vision_config, image_count)
                    image_engine.complete_prepared(image_prepared)
                config = dataclasses.replace(config, model_supports_vision=True)
                vision_checked = True
                report("Image connection passed.")
            except EngineError as exc:
                if config.vision_preference == "auto" and (exc.kind == "vision" or
                    (exc.status in {400, 415, 422} and exc.kind not in {"context", "template"})):
                    config = dataclasses.replace(config, vision="no", model_supports_vision=False)
                    report(f"Image check was rejected: {exc}. Continuing with the verified text connection; images are disabled.")
                else:
                    raise
    elif config.model_supports_vision is False:
        report("The server reports text-only input; images stay disabled.")
    result = {"checked_at": utc_now(), "model": config.model, "adapter": config.adapter,
              "context_window_tokens": config.context_window_tokens,
              "input_tokens": actual or tokens, "token_count_source": "server usage" if actual else source,
              "text_passed": True, "image_passed": vision_checked,
              "effective_vision": config.vision, "seconds": round(time.monotonic() - started, 2)}
    return config, result
