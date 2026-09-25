from __future__ import annotations

import atexit
import weakref
import gzip
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from . import display
from .filesystem import (
    Paths,
    append_jsonl,
    atomic_write_json,
    file_lock,
    json_dumps,
    read_json,
    read_jsonl,
    sha256_text,
    sortable_id,
    utc_now,
)
# Throughput/token-generation accounting lives in usage.py alongside
# normalize_usage(); re-exported here (rather than only imported by new
# callers) so a workspace overlay of just this module -- or old code that
# still does `from .records import format_throughput` -- keeps working.
from .usage import format_throughput, generation_token_count, throughput_stats


def _atomic_write_gzip_json(path: Path, value: Any) -> None:
    """Write ``value`` as gzip-compressed, pretty-printed JSON, atomically.

    Model request/response logs are the largest thing this harness writes
    (the whole conversation context, per request); gzip alone gives roughly a
    10x reduction with no format redesign. The write goes through a sibling
    temp file plus ``os.replace`` so a reader never observes a partial file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_dumps(value, pretty=True).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as archive:
                archive.write(payload)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_model_exchange(path: str | Path) -> dict[str, Any]:
    """Read one :meth:`Records.model_exchange` payload.

    Transparently reads both the current gzip-compressed logs
    (``request_*.json.gz``) and plain ``request_*.json`` files written by
    versions before B7 - by content (the gzip magic bytes), not by
    extension, so a caller that only has the old ``.json`` path (as older
    ``model_log_path`` values recorded in the lifetime/life logs do) still
    finds the file under its new ``.gz`` name.
    """

    given = Path(path)
    candidates: list[Path] = [given]
    if given.suffix == ".gz":
        candidates.append(given.with_suffix(""))
    else:
        candidates.append(given.with_suffix(given.suffix + ".gz"))
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("rb") as handle:
                magic = handle.read(2)
            if magic == b"\x1f\x8b":
                with gzip.open(candidate, "rt", encoding="utf-8") as handle:
                    return json.loads(handle.read())
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(given)


def _flush_records_at_exit(reference: "weakref.ref[Records]") -> None:
    records = reference()
    if records is not None:
        records._flush_at_exit()


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

    # How much feature-summary staleness (B10) other processes may see before
    # this one merges its buffered counters back to disk.
    _FLUSH_EVERY_EVENTS = 25
    _FLUSH_INTERVAL_SECONDS = 5.0

    # Newest model logs kept as plain JSON; older ones are gzip-compressed.
    PLAIN_MODEL_LOGS = 50
    MODEL_LOG_COMPRESS_EVERY = 10

    def __init__(self, paths: Paths):
        self.paths = paths
        self.run_id = sortable_id("run_")
        self._model_logs_written = 0
        self._lock = threading.Lock()
        self._feature_deltas: dict[str, int] = {}
        self._kind_deltas: dict[str, int] = {}
        self._pending_events = 0
        self._last_flush_at = time.monotonic()
        self._last_timestamp: str | None = None
        # A weak reference: registration must not keep every Records alive.
        atexit.register(_flush_records_at_exit, weakref.ref(self))

    def emit(self, kind: str, **data: Any) -> dict[str, Any]:
        if kind == "working_context_appended":
            data = self._compact_working_context_payload(data)
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

    @staticmethod
    def _compact_working_context_payload(data: dict[str, Any]) -> dict[str, Any]:
        """Replace a full working-context message with a compact reference.

        Nothing in this codebase reads ``working_context_appended`` messages
        back out of the lifetime/feature logs - the working context itself
        (``paths.working_context``) is the durable source of truth for that.
        Duplicating the whole message into two more logs on every append was
        pure overhead; a hash, role, length, and short preview is enough to
        audit or spot-check a run without paying that cost.
        """

        message = data.get("message")
        if not isinstance(message, dict):
            return data
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif content is None:
            text = ""
        else:
            text = json_dumps(content)
        compacted = dict(data)
        compacted["message"] = {
            "role": message.get("role"),
            "message_sha256": sha256_text(text) if text else None,
            "characters": len(text),
            "preview": text[:200],
        }
        return compacted

    @classmethod
    def feature_for(cls, kind: str) -> str:
        for prefix, feature in cls._FEATURE_PREFIXES:
            if kind.startswith(prefix):
                return feature
        return "runtime"

    def _record_feature(self, record: dict[str, Any]) -> None:
        """Write one routed copy; buffer the cumulative usage counters (B10).

        The lifetime log remains authoritative. Feature logs are deliberately
        redundant views for humans and for later Infinite Attention review,
        and each event's append there is cheap (an O(1) file append), so it
        still happens on every call. ``feature_summary.json`` used to be
        read, updated, and rewritten whole on every single event under the
        cross-process lock - real overhead on a busy life-loop. Counters are
        now accumulated in memory and merged into the on-disk summary at most
        every :attr:`_FLUSH_EVERY_EVENTS` events or
        :attr:`_FLUSH_INTERVAL_SECONDS` seconds (and always on ``flush()``/
        process exit). The merge is a read-add-write under the same lock as
        before, so a concurrent writer's counts are never lost - only the
        *frequency* of that read-modify-write drops. Callers hold the
        cross-process records lock and ``self._lock``.
        """

        feature = self.feature_for(str(record.get("kind") or "runtime"))
        append_jsonl(self.paths.feature_log(feature), record)
        kind = str(record.get("kind") or "runtime")
        self._feature_deltas[feature] = self._feature_deltas.get(feature, 0) + 1
        self._kind_deltas[kind] = self._kind_deltas.get(kind, 0) + 1
        self._pending_events += 1
        self._last_timestamp = record.get("timestamp")
        due = (
            self._pending_events >= self._FLUSH_EVERY_EVENTS
            or (time.monotonic() - self._last_flush_at) >= self._FLUSH_INTERVAL_SECONDS
        )
        if due:
            self._flush_feature_summary_locked()

    def _flush_feature_summary_locked(self) -> None:
        """Merge buffered counters into ``feature_summary.json``.

        The caller must already hold ``self._lock`` and the cross-process
        ``records_lock`` (as :meth:`emit` and :meth:`flush` both do), so this
        read-modify-write is still race-free against other writer processes.
        """

        if not self._feature_deltas and not self._kind_deltas:
            return
        summary = read_json(self.paths.feature_summary, {})
        summary = summary if isinstance(summary, dict) else {}
        feature_counts = dict(summary.get("features") or {})
        kind_counts = dict(summary.get("kinds") or {})
        total = int(summary.get("total_operational_events", 0) or 0)
        for feature, delta in self._feature_deltas.items():
            feature_counts[feature] = int(feature_counts.get(feature, 0) or 0) + delta
            total += delta
        for kind, delta in self._kind_deltas.items():
            kind_counts[kind] = int(kind_counts.get(kind, 0) or 0) + delta
        summary.update(
            {
                "updated_at": self._last_timestamp or summary.get("updated_at"),
                "total_operational_events": total,
                "features": dict(sorted(feature_counts.items())),
                "kinds": dict(sorted(kind_counts.items())),
            }
        )
        atomic_write_json(self.paths.feature_summary, summary)
        self._feature_deltas.clear()
        self._kind_deltas.clear()
        self._pending_events = 0
        self._last_flush_at = time.monotonic()

    def flush(self) -> None:
        """Force any buffered feature-summary counters to disk now."""

        with self._lock, file_lock(self.paths.records_lock):
            self._flush_feature_summary_locked()

    def _flush_at_exit(self) -> None:
        # Best-effort: interpreter teardown can make logging/locking unusable
        # (and a killed process never runs atexit at all - buffered counts
        # from a hard crash are lost, same as any other in-memory counter).
        # A workspace deleted meanwhile (reset, a finished temporary run)
        # must not be recreated just to hold counters.
        if not self.paths.root.is_dir():
            return
        try:
            self.flush()
        except Exception:
            pass

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

    def model_log_path(self, request_id: str) -> Path:
        """Where :meth:`model_exchange` writes one request's log.

        Recent logs stay plain JSON so the agent can inspect them with its
        ordinary file tools; older ones are gzip-compressed in place (the
        name gains ``.gz``). :func:`load_model_exchange` reads either form
        from this path.
        """

        return self.paths.model_log / f"{request_id}.json"

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
        atomic_write_json(self.model_log_path(request_id), payload)
        self._model_logs_written += 1
        if (self._model_logs_written - 1) % max(1, self.MODEL_LOG_COMPRESS_EVERY) == 0:
            self.compress_old_model_logs()

    def compress_old_model_logs(self, keep: int | None = None) -> int:
        """Gzip every plain model log except the newest ``keep`` ones.

        Each log holds the whole request context, so plain logs grow with
        the square of a long session. Returns how many files were compressed.
        """

        keep = self.PLAIN_MODEL_LOGS if keep is None else max(0, keep)
        try:
            plain = sorted(
                self.paths.model_log.glob("*.json"),
                key=lambda item: (item.stat().st_mtime_ns, item.name),
            )
        except OSError:
            return 0
        compressed = 0
        for path in plain[: max(0, len(plain) - keep)]:
            target = path.with_name(path.name + ".gz")
            try:
                _atomic_write_gzip_json(target, json.loads(path.read_text(encoding="utf-8")))
                path.unlink()
                compressed += 1
            except (OSError, ValueError):
                continue
        return compressed

    def recent_life(self, limit: int = 50) -> list[dict[str, Any]]:
        return read_jsonl(self.paths.life_loop_log)[-max(1, limit) :]

    def recent_operational(self, limit: int = 50) -> list[dict[str, Any]]:
        return read_jsonl(self.paths.lifetime_log)[-max(1, limit) :]

    def recent_feature(self, feature: str, limit: int = 50) -> list[dict[str, Any]]:
        return read_jsonl(self.paths.feature_log(feature))[-max(1, limit) :]

    def feature_usage(self) -> dict[str, Any]:
        """Feature/kind usage counters, including this instance's unflushed deltas.

        Reading straight off disk here would make counters lag behind the
        very calls to :meth:`emit` an in-process caller (a test, ``logs
        --summary``, a status check) just made, since :meth:`_record_feature`
        now buffers most updates (B10). The on-disk snapshot is merged with
        this instance's pending deltas without writing them out, so reads
        stay immediately consistent for the process that produced them while
        other processes still only see committed, lock-protected writes.
        """

        with self._lock:
            disk = read_json(self.paths.feature_summary, {})
            disk = disk if isinstance(disk, dict) else {}
            if not self._feature_deltas and not self._kind_deltas:
                return disk
            feature_counts = dict(disk.get("features") or {})
            kind_counts = dict(disk.get("kinds") or {})
            total = int(disk.get("total_operational_events", 0) or 0)
            for feature, delta in self._feature_deltas.items():
                feature_counts[feature] = int(feature_counts.get(feature, 0) or 0) + delta
                total += delta
            for kind, delta in self._kind_deltas.items():
                kind_counts[kind] = int(kind_counts.get(kind, 0) or 0) + delta
            merged = dict(disk)
            merged.update(
                {
                    "total_operational_events": total,
                    "features": dict(sorted(feature_counts.items())),
                    "kinds": dict(sorted(kind_counts.items())),
                }
            )
            return merged


class Console:
    """The foreground, real-time twin of ``watch``.

    ``thought``/``tool``/``result``/``context`` build the same life-record
    shape ``records.life(...)`` uses and hand it to
    :func:`artificium.display.render_event_body` - the identical per-kind
    renderer ``watch`` uses on the durable log - so a thought or tool call
    reads the same whether it is seen live or replayed later (S1). ``line``
    stays a separate, deliberately generic label+message primitive: several
    call sites (``runtime.py``, ``tools.py``) use it directly for messages
    that have no corresponding life-record kind (waiting/shutdown/recovery
    notices), and its exact behaviour (truncation, quiet/verbose gating) is
    part of the public contract other modules already depend on.
    """

    def __init__(self, *, verbose: bool = False, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self._session = display.WatchSession()

    def _render_options(self) -> display.RenderOptions:
        try:
            width = shutil.get_terminal_size(fallback=(100, 24)).columns
        except OSError:
            width = 100
        return display.RenderOptions(
            color=display.colors_enabled(),
            width=width,
            max_lines=0 if self.verbose else 6,
            full=self.verbose,
            reasoning=self.verbose,
            timestamps=False,
        )

    def _emit(self, record: dict[str, Any], *, detail: bool = False) -> None:
        if self.quiet or (detail and not self.verbose):
            return
        limit = 8_000 if self.verbose else 2_000
        for text in display.render_event_body(record, self._render_options(), self._session):
            if len(text) > limit:
                text = text[:limit].rstrip() + " … [full content in structured logs]"
            print(text, flush=True)

    def line(self, label: str, message: str, *, detail: bool = False) -> None:
        if self.quiet or (detail and not self.verbose):
            return
        message = " ".join(message.strip().splitlines())
        limit = 8_000 if self.verbose else 2_000
        if len(message) > limit:
            message = message[:limit].rstrip() + " … [full content in structured logs]"
        print(f"[{label}] {message}", flush=True)

    def thought(self, content: str) -> None:
        self._emit({"kind": "thought", "content": content})

    def tool(self, name: str, arguments: dict[str, Any]) -> None:
        self._emit({"kind": "tool_call", "name": name, "arguments": arguments or {}})

    def result(self, name: str, result: dict[str, Any]) -> None:
        if self.quiet:
            return
        self._emit({"kind": "tool_result", "name": name, "result": result or {}})

    def context(self, tokens: int, window: int, *, source: str = "estimate") -> None:
        percent = tokens / window * 100 if window else 0
        self._emit(
            {
                "kind": "context_usage",
                "estimated_tokens": tokens,
                "context_window_tokens": window,
                "context_percent": percent,
                "token_count_source": source,
            }
        )

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
