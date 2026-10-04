from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


# The lifetime log of a long run reaches hundreds of megabytes; only its tail
# since the previous offload matters here.
MAX_SCAN_BYTES = 64 * 1024 * 1024
MAX_SCAN_RECORDS = 50_000


def _reverse_lines(path: Path, max_bytes: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        floor = max(0, position - max_bytes)
        remainder = b""
        while position > floor:
            size = min(1 << 20, position - floor)
            position -= size
            handle.seek(position)
            block = handle.read(size) + remainder
            lines = block.split(b"\n")
            remainder = lines.pop(0)
            for line in reversed(lines):
                if line.strip():
                    yield line
        if position == 0 and remainder.strip():
            yield remainder


def _parse_time(value: Any) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _hours(seconds: float) -> float:
    return round(seconds / 3600, 2)


def summarize(lifetime_log: Path, now: dt.datetime | None = None) -> dict[str, Any]:
    """Where wall time went since the last confirmed working-memory offload.

    Measured from the lifetime log, not from the agent's recollection: a
    long wait on the same rebuild, repeated many times, is invisible from
    inside a context that is offloaded every hour or two.
    """

    now = now or dt.datetime.now(dt.timezone.utc)
    records: list[dict[str, Any]] = []
    if lifetime_log.is_file():
        for line in _reverse_lines(lifetime_log, MAX_SCAN_BYTES):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("kind") == "working_memory_offloaded":
                break
            records.append(record)
            if len(records) >= MAX_SCAN_RECORDS:
                break
    records.reverse()
    start = _parse_time(records[0].get("timestamp")) if records else None
    model_seconds = tool_seconds = sleep_seconds = 0.0
    model_requests = tool_calls = failures = 0
    sleeping_since: dt.datetime | None = None
    slowest: list[tuple[float, str, str]] = []
    shell: dict[str, list[float]] = defaultdict(list)
    for record in records:
        kind = record.get("kind")
        if kind == "model_response":
            model_requests += 1
            model_seconds += float(record.get("duration_seconds") or 0)
        elif kind == "tool_executed":
            tool_calls += 1
            seconds = float(record.get("duration_seconds") or 0)
            tool_seconds += seconds
            result = record.get("result") if isinstance(record.get("result"), dict) else {}
            if result.get("status") in {"failed", "error"}:
                failures += 1
            name = str(record.get("name") or "unknown")
            target = ""
            if name == "run_shell":
                target = re.sub(r"\s+", " ", str(
                    result.get("command") or result.get("primary_target") or ""
                )).strip()
                if target:
                    shell[target].append(seconds)
            slowest.append((seconds, name, target[:160]))
        elif kind == "sleep_started":
            sleeping_since = _parse_time(record.get("timestamp"))
        elif kind == "sleep_ended" and sleeping_since is not None:
            ended = _parse_time(record.get("timestamp"))
            if ended is not None:
                sleep_seconds += max(0.0, (ended - sleeping_since).total_seconds())
            sleeping_since = None
    if sleeping_since is not None:
        sleep_seconds += max(0.0, (now - sleeping_since).total_seconds())
    slowest.sort(reverse=True)
    repeated = sorted(
        ((len(times), sum(times), command) for command, times in shell.items()
         if len(times) >= 3),
        reverse=True,
    )
    return {
        "since": records[0].get("timestamp") if records else None,
        "wall_hours": _hours((now - start).total_seconds()) if start else 0.0,
        "model_hours": _hours(model_seconds),
        "model_requests": model_requests,
        "tool_hours": _hours(tool_seconds),
        "tool_calls": tool_calls,
        "failed_tool_calls": failures,
        "sleep_hours": _hours(sleep_seconds),
        "slowest_tool_calls": [
            {"tool": name, "seconds": round(seconds, 1), "target": target or None}
            for seconds, name, target in slowest[:3]
        ],
        "repeated_shell_commands": [
            {"command": command[:160], "runs": runs, "seconds": round(total, 1)}
            for runs, total, command in repeated[:3]
        ],
    }
