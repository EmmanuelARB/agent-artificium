from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .filesystem import (
    Paths,
    append_jsonl,
    atomic_write_json,
    file_lock,
    json_dumps,
    read_json,
    read_jsonl,
    sortable_id,
    utc_now,
)


class Records:
    _FEATURE_PREFIXES: tuple[tuple[str, str], ...] = (
        ("attention_", "infinite-attention"),
        ("long_term_memory_", "memory"),
        ("memory_", "memory"),
        ("working_memory_", "working-memory"),
        ("working_context_", "working-memory"),
        ("context_", "working-memory"),
        ("interaction_", "interactions"),
        ("notification_", "notifications"),
        ("guidance_", "notifications"),
        ("scheduler_", "scheduler"),
        ("self_", "self"),
        ("sleep_", "sleep"),
        ("model_", "engine"),
        ("engine_", "engine"),
        ("api_key_", "engine"),
        ("tool_", "tools"),
        ("prompt_pack_", "promptgramming"),
        ("initialization_", "initialization"),
        ("setup_", "initialization"),
        ("turn_", "life-loop"),
        ("automatic_no_action_", "sleep"),
        ("images_", "vision"),
    )

    def __init__(self, paths: Paths):
        self.paths = paths
        self.run_id = sortable_id("run_")
        self._lock = threading.Lock()

    def emit(self, kind: str, **data: Any) -> dict[str, Any]:
        record = {
            "event_id": sortable_id("evt_"),
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "kind": kind,
            **data,
        }
        with self._lock, file_lock(self.paths.records_lock):
            append_jsonl(self.paths.lifetime_log, record)
            self._record_feature(record)
        return record

    @classmethod
    def feature_for(cls, kind: str) -> str:
        for prefix, feature in cls._FEATURE_PREFIXES:
            if kind.startswith(prefix):
                return feature
        return "runtime"

    def _record_feature(self, record: dict[str, Any]) -> None:
        """Write one routed copy plus compact cumulative usage metadata.

        The lifetime log remains authoritative. Feature logs are deliberately
        redundant views for humans and for later Infinite Attention review.
        Callers hold the cross-process records lock.
        """

        feature = self.feature_for(str(record.get("kind") or "runtime"))
        append_jsonl(self.paths.feature_log(feature), record)
        summary = read_json(self.paths.feature_summary, {})
        summary = summary if isinstance(summary, dict) else {}
        feature_counts = summary.get("features")
        feature_counts = feature_counts if isinstance(feature_counts, dict) else {}
        kind_counts = summary.get("kinds")
        kind_counts = kind_counts if isinstance(kind_counts, dict) else {}
        kind = str(record.get("kind") or "runtime")
        feature_counts[feature] = int(feature_counts.get(feature, 0) or 0) + 1
        kind_counts[kind] = int(kind_counts.get(kind, 0) or 0) + 1
        summary.update(
            {
                "updated_at": record.get("timestamp"),
                "total_operational_events": int(
                    summary.get("total_operational_events", 0) or 0
                )
                + 1,
                "features": dict(sorted(feature_counts.items())),
                "kinds": dict(sorted(kind_counts.items())),
            }
        )
        atomic_write_json(self.paths.feature_summary, summary)

    def life(self, kind: str, *, turn_id: str | None = None, **data: Any) -> dict[str, Any]:
        record = {
            "id": sortable_id("life_"),
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "turn_id": turn_id,
            "kind": kind,
            **data,
        }
        with self._lock:
            append_jsonl(self.paths.life_loop_log, record)
        return record

    def model_exchange(
        self,
        *,
        request_id: str,
        messages: list[dict[str, Any]],
        request_parameters: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "request_id": request_id,
            "timestamp": utc_now(),
            "messages": messages,
            "request_parameters": request_parameters,
            "response": response,
            "error": error,
        }
        atomic_write_json(self.paths.model_log / f"{request_id}.json", payload)

    def recent_life(self, limit: int = 50) -> list[dict[str, Any]]:
        return read_jsonl(self.paths.life_loop_log)[-max(1, limit) :]

    def recent_operational(self, limit: int = 50) -> list[dict[str, Any]]:
        return read_jsonl(self.paths.lifetime_log)[-max(1, limit) :]

    def recent_feature(self, feature: str, limit: int = 50) -> list[dict[str, Any]]:
        return read_jsonl(self.paths.feature_log(feature))[-max(1, limit) :]

    def feature_usage(self) -> dict[str, Any]:
        value = read_json(self.paths.feature_summary, {})
        return value if isinstance(value, dict) else {}


class Console:
    def __init__(self, *, verbose: bool = False, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet

    def line(self, label: str, message: str, *, detail: bool = False) -> None:
        if self.quiet or (detail and not self.verbose):
            return
        message = " ".join(message.strip().splitlines())
        limit = 8_000 if self.verbose else 2_000
        if len(message) > limit:
            message = message[:limit].rstrip() + " … [full content in structured logs]"
        print(f"[{label}] {message}", flush=True)

    def thought(self, content: str) -> None:
        self.line("think", content)

    def tool(self, name: str, arguments: dict[str, Any]) -> None:
        focus = ""
        for key in (
            "path",
            "source",
            "interaction_id",
            "event_id",
            "stream_id",
            "session_id",
            "command",
            "title",
        ):
            if key in arguments and arguments[key] not in (None, ""):
                focus = f" {key}={arguments[key]}"
                break
        self.line("tool", f"{name}{focus}")

    def result(self, name: str, result: dict[str, Any]) -> None:
        if self.quiet:
            return
        summary = result.get("summary") or result.get("status") or "ok"
        extras: list[str] = []
        for key in (
            "path",
            "result_path",
            "session_id",
            "before_tokens",
            "after_tokens",
            "before_words",
            "after_words",
            "before_characters",
            "after_characters",
            "source_exhausted",
            "stream_id",
        ):
            if key in result and result[key] is not None:
                extras.append(f"{key}={result[key]}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        self.line("result", f"{name}: {summary}{suffix}")

    def context(self, tokens: int, window: int, *, source: str = "estimate") -> None:
        percent = tokens / window * 100 if window else 0
        prefix = "~" if source == "estimate" else ""
        self.line("context", f"{prefix}{tokens:,} / {window:,} tokens ({percent:.1f}%; {source})")

    def usage(self, usage: dict[str, Any]) -> None:
        if not usage:
            return
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        total = usage.get("total_tokens")
        details = []
        if prompt is not None:
            details.append(f"input={prompt}")
        if completion is not None:
            details.append(f"output={completion}")
        if total is not None:
            details.append(f"total={total}")
        if details:
            self.line("usage", "tokens " + ", ".join(details))

    def error(self, message: str) -> None:
        self.line("error", message)
