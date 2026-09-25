"""One event renderer, shared by the foreground :class:`~artificium.records.Console`
and the ``watch`` command (``artificium/cli.py``).

Rendering a life-record (see :mod:`artificium.events`) is split in two layers:

* :func:`render_event_body` turns one record into a list of plain-text lines
  (colorized when ``options.color`` is set), with no timestamp. This is what
  ``Console`` uses directly, so a thought, tool call, or context line looks
  the same whether it scrolls by live in the foreground or is replayed later
  through ``watch``.
* :func:`render_record_lines` adds a local ``HH:MM:SS`` timestamp in front,
  which is what ``watch`` wants but a foreground console (already printing in
  real time, one line among many other prints) does not need repeated on
  every line.

Every renderer is defensive: a malformed or unexpected-shape record never
raises out of this module. Unknown kinds fall back to a generic key=value
rendering (see :mod:`artificium.events`).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

from . import events
from .filesystem import json_dumps
from .usage import cache_hit_ratio, normalize_usage

# --------------------------------------------------------------------------
# options / color
# --------------------------------------------------------------------------


@dataclass
class RenderOptions:
    color: bool = False
    width: int = 100
    max_lines: int = 6
    full: bool = False
    reasoning: bool = False
    timestamps: bool = True


def colors_enabled(*, no_color: bool = False, stream: Any = None) -> bool:
    """Whether ANSI colour is appropriate right now.

    Off when explicitly disabled, when ``NO_COLOR`` is set (see
    https://no-color.org), or when the destination is not a terminal.
    """

    if no_color or os.environ.get("NO_COLOR"):
        return False
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


_RESET = "\x1b[0m"
_CATEGORY_COLOR: dict[str, str] = {
    events.CATEGORY_ENGINE: "\x1b[36m",
    events.CATEGORY_TOOLS: "\x1b[33m",
    events.CATEGORY_THOUGHTS: "\x1b[35m",
    events.CATEGORY_CONTEXT: "\x1b[94m",
    events.CATEGORY_MEMORY: "\x1b[32m",
    events.CATEGORY_TURNS: "\x1b[1m",
    events.CATEGORY_NOTIFICATIONS: "\x1b[93m",
    events.CATEGORY_RECOVERY: "\x1b[31m",
    events.CATEGORY_OTHER: "\x1b[2m",
}


# --------------------------------------------------------------------------
# small formatting helpers
# --------------------------------------------------------------------------


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_count(value: Any) -> str:
    if value is None:
        return "?"
    number = _num(value)
    if abs(number) >= 1000:
        return f"{number / 1000:.1f}k"
    return f"{int(number)}"


def _fmt_pct(ratio: float | None) -> str:
    if ratio is None:
        return "?"
    return f"{ratio * 100:.0f}%"


def _fmt_duration(seconds: Any) -> str:
    if seconds is None:
        return "?"
    value = _num(seconds)
    if value >= 60:
        minutes, secs = divmod(int(round(value)), 60)
        return f"{minutes}m{secs:02d}s"
    if value >= 10:
        return f"{value:.0f}s"
    return f"{value:.1f}s"


def _fmt_duration_long(seconds: Any) -> str:
    value = max(0, int(round(_num(seconds))))
    minutes, secs = divmod(value, 60)
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _short_id(request_id: Any) -> str:
    text = str(request_id or "").strip()
    return text[-4:] if len(text) >= 4 else (text or "????")


def _bar(ratio: float, width: int = 10) -> str:
    ratio = max(0.0, min(1.5, ratio))
    filled = min(width, int(round(ratio * width)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _parse_timestamp(value: Any) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def local_time(value: Any) -> str:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return str(value or "")
    return parsed.astimezone().strftime("%H:%M:%S")


def parse_since(value: str) -> dt.datetime:
    """Parse ``watch --since``: either ``HH:MM`` (today, local) or an ISO timestamp."""

    text = value.strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        now_local = dt.datetime.now().astimezone()
        candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate > now_local:
            candidate -= dt.timedelta(days=1)
        return candidate.astimezone(dt.timezone.utc)
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone(dt.timezone.utc)


def record_passes_filters(
    record: dict[str, Any],
    *,
    only_categories: set[str] | None = None,
    no_thoughts: bool = False,
    since: dt.datetime | None = None,
) -> bool:
    kind = str(record.get("kind") or "event")
    if only_categories:
        if events.category_for(kind) not in only_categories:
            return False
    if no_thoughts and kind in (
        events.KIND_THOUGHT,
        events.KIND_LIFE_LOOP_OUTPUT,
        events.KIND_PROVIDER_REASONING,
    ):
        return False
    if since is not None:
        when = _parse_timestamp(record.get("timestamp"))
        if when is not None and when < since:
            return False
    return True


def _wrap_text(text: str, width: int, max_lines: int, full: bool) -> list[str]:
    width = max(20, width)
    paragraphs = text.splitlines() or [""]
    lines: list[str] = []
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=width, break_long_words=True) or [""]
        lines.extend(wrapped)
    if not full and max_lines and max_lines > 0 and len(lines) > max_lines:
        hidden = len(lines) - max_lines
        lines = lines[:max_lines]
        lines.append(f"… (+{hidden} lines)")
    return lines


def _wrap_tagged(tag: str, text: str, options: RenderOptions) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return [f"[{tag}] (empty)"]
    width = max(20, options.width - (len(tag) + 3))
    lines = _wrap_text(text, width, options.max_lines, options.full)
    pad = " " * (len(tag) + 3)
    return [f"[{tag}] {lines[0]}"] + [pad + line for line in lines[1:]]


def _compact(value: Any, limit: int = 80) -> str:
    if isinstance(value, (dict, list)):
        text = json_dumps(value)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


_SKIP_KEYS = {"id", "event_id", "timestamp", "run_id", "turn_id", "kind"}


def _render_generic(
    record: dict[str, Any], options: RenderOptions, session: "WatchSession | None" = None,
    *, tag: str | None = None,
) -> list[str]:
    kind = str(record.get("kind") or "event")
    parts = [
        f"{key}={_compact(value)}"
        for key, value in record.items()
        if key not in _SKIP_KEYS and value not in (None, "", [], {})
    ]
    body = " ".join(parts)
    label = tag or kind
    return [f"[{label}] {body}".rstrip()]


# --------------------------------------------------------------------------
# session: turn/cache bookkeeping shared across a watch run (or one
# foreground Console instance)
# --------------------------------------------------------------------------


@dataclass
class WatchSession:
    last_response_at: dt.datetime | None = None
    turn_started_at: dt.datetime | None = None
    turn_request_count: int = 0
    cumulative_requests: int = 0
    cumulative_input: int = 0
    cumulative_cached: int = 0
    cumulative_output: int = 0
    cumulative_reasoning: int = 0

    def observe_response(
        self, record: dict[str, Any], normalized: dict[str, Any]
    ) -> list[str]:
        warnings: list[str] = []
        now = _parse_timestamp(record.get("timestamp"))
        input_tokens = int(normalized.get("input_tokens") or 0)
        cache_read = int(normalized.get("cache_read_tokens") or 0)
        ratio = cache_hit_ratio(normalized)
        if (
            self.last_response_at is not None
            and now is not None
            and ratio is not None
            and ratio < 0.10
            and input_tokens > 8000
        ):
            gap = (now - self.last_response_at).total_seconds()
            gap_text = _fmt_duration_long(gap)
            if gap > 240:
                warnings.append(
                    f"  ⚠ cache miss {_fmt_count(cache_read)}/{_fmt_count(input_tokens)} — "
                    f"idle {gap_text} since previous request (provider cache likely expired)"
                )
            else:
                warnings.append(
                    f"  ⚠ cache miss {_fmt_count(cache_read)}/{_fmt_count(input_tokens)} — "
                    f"backend may have re-routed (idle {gap_text})"
                )
        if now is not None:
            self.last_response_at = now
        self.turn_request_count += 1
        self.cumulative_requests += 1
        self.cumulative_input += input_tokens
        self.cumulative_cached += cache_read
        self.cumulative_output += int(normalized.get("output_tokens") or 0)
        self.cumulative_reasoning += int(normalized.get("reasoning_tokens") or 0)
        return warnings

    def observe_turn_started(self, record: dict[str, Any]) -> list[str]:
        self.turn_started_at = _parse_timestamp(record.get("timestamp"))
        self.turn_request_count = 0
        trigger = record.get("trigger")
        label = f" ({trigger})" if trigger else ""
        return [f"── turn started{label} " + "─" * 12]

    def observe_turn_completed(self, record: dict[str, Any]) -> list[str]:
        now = _parse_timestamp(record.get("timestamp"))
        duration = None
        if self.turn_started_at is not None and now is not None:
            duration = (now - self.turn_started_at).total_seconds()
        summary = f"{self.turn_request_count} request(s)"
        if duration is not None:
            summary += f", {_fmt_duration(duration)}"
        lines = [f"── turn completed — {summary} " + "─" * 12]
        if self.cumulative_requests:
            hit = (
                self.cumulative_cached / self.cumulative_input * 100
                if self.cumulative_input
                else 0.0
            )
            lines.append(
                "   session so far: "
                f"{self.cumulative_requests} req · in {_fmt_count(self.cumulative_input)} "
                f"· cached {_fmt_count(self.cumulative_cached)} ({hit:.0f}%) "
                f"· out {_fmt_count(self.cumulative_output)} "
                f"(think {_fmt_count(self.cumulative_reasoning)})"
            )
        return lines


# --------------------------------------------------------------------------
# per-kind renderers
# --------------------------------------------------------------------------


def render_turn_started(record, options, session=None):
    if session is not None:
        return session.observe_turn_started(record)
    trigger = record.get("trigger")
    return [f"── turn started" + (f" ({trigger})" if trigger else "")]


def render_turn_completed(record, options, session=None):
    if session is not None:
        return session.observe_turn_completed(record)
    return ["── turn completed ──"]


def render_turn_interrupted(record, options, session=None):
    reason = record.get("reason") or "?"
    rid = record.get("request_id")
    suffix = f" request={_short_id(rid)}" if rid else ""
    return [f"[turn] interrupted ({reason}){suffix}"]


def render_context_usage(record, options, session=None):
    tokens = int(_num(record.get("estimated_tokens")))
    window = int(_num(record.get("context_window_tokens")))
    percent = record.get("context_percent")
    percent = _num(percent) if percent is not None else (tokens / window * 100 if window else 0.0)
    source = record.get("token_count_source", "estimate")
    prefix = "~" if source == "estimate" else ""
    line = f"[context] {prefix}{tokens:,} / {window:,} tokens ({percent:.1f}%; {source})"
    working_memory_tokens = record.get("working_memory_tokens")
    if working_memory_tokens:
        target = int(_num(working_memory_tokens))
        ratio = tokens / target if target else 0.0
        line += f"  wm {tokens:,}/{target:,} {_bar(ratio)}"
    return [line]


def render_engine_request(record, options, session=None):
    rid = record.get("request_id")
    estimated = int(_num(record.get("estimated_tokens")))
    source = record.get("token_count_source", "estimate")
    provider = record.get("provider")
    model = record.get("model")
    return [
        f"[engine] request {rid} sent to {provider}/{model} "
        f"({estimated:,} input tokens, {source})"
    ]


def render_engine_waiting(record, options, session=None):
    rid = record.get("request_id")
    elapsed = _num(record.get("elapsed_seconds"))
    quiet = record.get("seconds_since_stream_data")
    detail = f", no streamed data for {_num(quiet):.0f}s" if quiet is not None else ""
    return [f"[engine] request {rid} still waiting for provider ({elapsed:.0f}s elapsed{detail})"]


def _render_engine_failure(record, label: str) -> list[str]:
    rid = record.get("request_id")
    duration = _num(record.get("duration_seconds"))
    lines = [
        f"[engine] request {rid} {label} after {duration:.1f}s; "
        f"log={record.get('model_log_path')}"
    ]
    if record.get("error"):
        lines.append("[engine] " + " ".join(str(record["error"]).splitlines())[:1600])
    if record.get("hint"):
        lines.append("[engine] " + " ".join(str(record["hint"]).splitlines())[:1600])
    return lines


def render_engine_request_failed(record, options, session=None):
    return _render_engine_failure(record, "failed")


def render_engine_request_cancelled(record, options, session=None):
    return _render_engine_failure(record, "cancelled")


def render_engine_blocked(record, options, session=None):
    return [
        "[engine] Model requests paused; history and incoming messages are retained. "
        "Correct the reported problem, then run: "
        + str(record.get("recovery_command") or "python3 artificium.py restart")
    ]


def render_engine_response(record, options, session=None):
    request_id = record.get("request_id")
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    normalized = record.get("usage_normalized")
    if not isinstance(normalized, dict):
        normalized = normalize_usage(usage)
    input_tokens = normalized.get("input_tokens")
    output_tokens = normalized.get("output_tokens")
    reasoning_tokens = normalized.get("reasoning_tokens")
    cache_read = normalized.get("cache_read_tokens")
    cache_write = normalized.get("cache_write_tokens")
    duration = record.get("duration_seconds")
    throughput = record.get("throughput") if isinstance(record.get("throughput"), dict) else {}
    rate = throughput.get("generation_tok_s")
    approx = throughput.get("source") != "server"
    finish = record.get("finish_reason") or "?"
    estimated = record.get("estimated_tokens")

    parts = [f"⇄ req {_short_id(request_id)}"]
    if input_tokens is not None:
        segment = f"in {_fmt_count(input_tokens)}"
        if cache_read:
            ratio = cache_hit_ratio(normalized)
            segment += f" · cache {_fmt_count(cache_read)}"
            if ratio is not None:
                segment += f" ({_fmt_pct(ratio)})"
        if cache_write:
            segment += f" +write {_fmt_count(cache_write)}"
        parts.append(segment)
    if estimated is not None and input_tokens:
        diff_pct = (_num(estimated) - input_tokens) / input_tokens * 100
        parts.append(f"est {_fmt_count(estimated)} ({diff_pct:+.0f}%)")
    if output_tokens is not None:
        segment = f"out {_fmt_count(output_tokens)}"
        if reasoning_tokens:
            segment += f" (think {_fmt_count(reasoning_tokens)})"
        parts.append(segment)
    if duration is not None:
        parts.append(_fmt_duration(duration))
    if isinstance(rate, (int, float)):
        parts.append(f"{'~' if approx else ''}{rate:.0f} tok/s")
    parts.append(str(finish))
    lines = ["  ".join(parts)]
    if session is not None:
        lines.extend(session.observe_response(record, normalized))
    return lines


def render_engine_progress(record, options, session=None):
    rid = record.get("request_id")
    generated = int(_num(record.get("generated_chars")))
    reasoning = int(_num(record.get("reasoning_chars")))
    elapsed = _num(record.get("elapsed_seconds"))
    text = f"[engine] request {_short_id(rid)} generating… {generated:,} chars"
    if reasoning:
        text += f" (thinking {reasoning:,})"
    text += f" · {elapsed:.0f}s"
    return [text]


def render_provider_reasoning(record, options, session=None):
    if not options.reasoning:
        return None
    return _wrap_tagged("reasoning", record.get("content"), options)


def render_thought(record, options, session=None):
    return _wrap_tagged("think", record.get("content"), options)


def render_life_loop_output(record, options, session=None):
    return _wrap_tagged("output", record.get("content"), options)


_TOOL_FOCUS_KEYS = (
    "command",
    "path",
    "source",
    "interaction_id",
    "event_id",
    "stream_id",
    "session_id",
    "title",
)


def render_tool_call(record, options, session=None):
    name = record.get("name") or "invalid_tool"
    arguments = record.get("arguments") if isinstance(record.get("arguments"), dict) else {}
    focus = ""
    for key in _TOOL_FOCUS_KEYS:
        if key in arguments and arguments[key] not in (None, ""):
            value = " ".join(str(arguments[key]).split())
            limit = max(20, options.width - 24)
            if len(value) > limit:
                value = value[:limit].rstrip() + "…"
            focus = f" {key}={value}"
            break
    return [f"[tool] {name}{focus}".rstrip()]


_TOOL_RESULT_EXTRA_KEYS = (
    "path",
    "result_path",
    "session_id",
    "stream_id",
    "before_tokens",
    "after_tokens",
    "before_words",
    "after_words",
    "before_characters",
    "after_characters",
    "source_exhausted",
)


def render_tool_result(record, options, session=None):
    name = record.get("name") or "invalid_tool"
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    status = result.get("status") or "ok"
    summary = result.get("summary") or status
    extras: list[str] = []
    returncode = result.get("returncode")
    if returncode is not None:
        extras.append(f"rc={returncode}")
    duration = result.get("duration_seconds")
    if duration is not None:
        extras.append(_fmt_duration(duration))
    for key in _TOOL_RESULT_EXTRA_KEYS:
        if key in result and result[key] is not None:
            extras.append(f"{key}={result[key]}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    if summary != status:
        return [f"[result] {name}: {status} — {summary}{suffix}"]
    return [f"[result] {name}: {status}{suffix}"]


def render_tool_request_rejected(record, options, session=None):
    errors = record.get("errors")
    count = len(errors) if isinstance(errors, list) else 0
    first = ""
    if isinstance(errors, list) and errors:
        item = errors[0]
        first = str(item.get("error") or "") if isinstance(item, dict) else str(item)
        first = " ".join(first.split())
        limit = max(20, options.width - 30)
        if len(first) > limit:
            first = first[:limit].rstrip() + "…"
    suffix = f" — {first}" if first else ""
    return [f"[reject] {count} invalid tool call(s){suffix}"]


_MEMORY_LABELS = {
    events.KIND_WORKING_MEMORY_OFFLOADED: "offloaded",
    events.KIND_CONTEXT_COMPACTED: "compacted",
    events.KIND_MEMORY_SAVED: "saved",
    events.KIND_ATTENTION_CONTEXT_COMPRESSED: "attention-compressed",
}


def render_memory_event(record, options, session=None):
    kind = str(record.get("kind") or "")
    label = _MEMORY_LABELS.get(kind, kind)
    path = record.get("path") or record.get("memory_path") or record.get("stream_id") or ""
    before = record.get("before_tokens", "")
    after = record.get("after_tokens", "")
    return [f"[memory] {label} {path} {before} -> {after}".rstrip()]


def render_pinned_mind_refreshed(record, options, session=None):
    return _render_generic(record, options, tag="memory")


def render_guidance_notification(record, options, session=None):
    return [
        f"[guidance] {record.get('guidance_type')}: "
        f"~{int(_num(record.get('estimated_tokens'))):,} tokens"
    ]


def render_request_repair(record, options, session=None):
    kind = str(record.get("kind") or "")
    suffix = kind.removeprefix("request_repair_")
    detail = record.get("archive") or record.get("error") or record.get("attempts")
    return [f"[recovery] {suffix}: {detail}"]


_RENDERERS: dict[str, Callable[[dict, RenderOptions, "WatchSession | None"], list[str] | None]] = {
    events.KIND_TURN_STARTED: render_turn_started,
    events.KIND_TURN_COMPLETED: render_turn_completed,
    events.KIND_TURN_INTERRUPTED: render_turn_interrupted,
    events.KIND_CONTEXT_USAGE: render_context_usage,
    events.KIND_CONTEXT_COMPACTED: render_memory_event,
    events.KIND_ENGINE_REQUEST: render_engine_request,
    events.KIND_ENGINE_WAITING: render_engine_waiting,
    events.KIND_ENGINE_RESPONSE: render_engine_response,
    events.KIND_ENGINE_REQUEST_FAILED: render_engine_request_failed,
    events.KIND_ENGINE_REQUEST_CANCELLED: render_engine_request_cancelled,
    events.KIND_ENGINE_BLOCKED: render_engine_blocked,
    events.KIND_ENGINE_PROGRESS: render_engine_progress,
    events.KIND_PROVIDER_REASONING: render_provider_reasoning,
    events.KIND_THOUGHT: render_thought,
    events.KIND_LIFE_LOOP_OUTPUT: render_life_loop_output,
    events.KIND_TOOL_CALL: render_tool_call,
    events.KIND_TOOL_RESULT: render_tool_result,
    events.KIND_TOOL_REQUEST_REJECTED: render_tool_request_rejected,
    events.KIND_WORKING_MEMORY_OFFLOADED: render_memory_event,
    events.KIND_ATTENTION_CONTEXT_COMPRESSED: render_memory_event,
    events.KIND_MEMORY_SAVED: render_memory_event,
    events.KIND_PINNED_MIND_REFRESHED: render_pinned_mind_refreshed,
    events.KIND_GUIDANCE_NOTIFICATION: render_guidance_notification,
    events.KIND_REQUEST_REPAIR_STARTED: render_request_repair,
    events.KIND_REQUEST_REPAIR_FAILED: render_request_repair,
    events.KIND_REQUEST_REPAIR_READY: render_request_repair,
}


def render_event_body(
    record: dict[str, Any], options: RenderOptions, session: "WatchSession | None" = None
) -> list[str]:
    """Render one life-record to plain-text lines, colorized, no timestamp.

    Never raises: a renderer that trips over an unexpected shape falls back
    to the generic key=value rendering, and if even that fails the record is
    still surfaced (never silently dropped).
    """

    kind = str(record.get("kind") or "event")
    renderer = _RENDERERS.get(kind, _render_generic)
    try:
        body = renderer(record, options, session)
    except Exception:
        try:
            body = _render_generic(record, options, session)
        except Exception:
            body = [f"[{kind}] <unrenderable record>"]
    if body is None:
        return []
    body = list(body)
    if options.color and body:
        color = _CATEGORY_COLOR.get(events.category_for(kind))
        if color:
            body = [f"{color}{line}{_RESET}" for line in body]
    return body


def render_record_lines(
    record: dict[str, Any], options: RenderOptions, session: "WatchSession | None" = None
) -> list[str]:
    """Render one life-record with a leading local timestamp (``watch``'s view)."""

    body = render_event_body(record, options, session)
    if not body or not options.timestamps:
        return body
    prefix = f"{local_time(record.get('timestamp'))} "
    pad = " " * len(prefix)
    return [f"{prefix}{body[0]}"] + [f"{pad}{line}" for line in body[1:]]
