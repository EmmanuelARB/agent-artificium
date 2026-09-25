"""The `watch` command: a growing-file follower and its record renderer.

`_print_detach_status` is resolved through `artificium.cli` at call time
(rather than imported directly) because tests patch it at
"artificium.cli._print_detach_status"-adjacent paths; only a lookup through
that module object at call time observes such a patch.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import cli as _cli
from . import display, events
from .filesystem import Paths


class _LogFollower:
    """Follows a growing JSONL file, reopening it on rotation or truncation.

    Detection is by inode (rotation: the path now names a different file) or
    by the file having shrunk below the last read position (truncation, e.g.
    an external ``> file`` or log-rotation copy-truncate). A partial line at
    EOF (the writer is mid-``append_jsonl``) is left unread and retried on
    the next call rather than returned broken.
    """

    def __init__(self, path: Path):
        self.path = path
        self._handle = path.open("r", encoding="utf-8", errors="replace")
        self._inode = self._stat_inode()
        self._position = 0

    def _stat_inode(self) -> int | None:
        try:
            return os.fstat(self._handle.fileno()).st_ino
        except OSError:
            return None

    def seek_end(self) -> None:
        self._handle.seek(0, os.SEEK_END)
        self._position = self._handle.tell()

    def read_new_lines(self) -> list[str]:
        lines = self._read_available()
        if self._maybe_reopen():
            # Whatever was read above came from the (now stale) old handle;
            # the reopened file may already hold lines of its own.
            lines.extend(self._read_available())
        return lines

    def _read_available(self) -> list[str]:
        lines: list[str] = []
        while True:
            start = self._handle.tell()
            line = self._handle.readline()
            if not line:
                break
            if not line.endswith("\n"):
                # A writer's partial append; wait for the rest of the line.
                self._handle.seek(start)
                break
            self._position = self._handle.tell()
            lines.append(line)
        return lines

    def _maybe_reopen(self) -> bool:
        try:
            stat = self.path.stat()
        except OSError:
            return False
        rotated = self._inode is not None and stat.st_ino != self._inode
        truncated = stat.st_size < self._position
        if not (rotated or truncated):
            return False
        try:
            self._handle.close()
        except OSError:
            pass
        self._handle = self.path.open("r", encoding="utf-8", errors="replace")
        self._inode = self._stat_inode()
        self._position = 0
        return True

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass


def _tail_lines_from_end(path: Path, count: int, *, block_size: int = 65536) -> list[str]:
    """The last ``count`` complete lines of ``path``, without reading it whole.

    Seeks backwards in blocks from the end of the file, counting newlines,
    until enough are found (or the start of the file is reached) - cheap even
    for a many-megabyte log, unlike ``readlines()``.
    """

    if count <= 0:
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        blocks: list[bytes] = []
        newline_count = 0
        while position > 0 and newline_count <= count:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            block = handle.read(read_size)
            blocks.append(block)
            newline_count += block.count(b"\n")
        data = b"".join(reversed(blocks))
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if position > 0 and lines:
        # The first line in this window is almost certainly a partial line;
        # the real first line of the file starts before our earliest block.
        lines = lines[1:]
    return lines[-count:]


def _parse_only(value: str | None) -> set[str] | None:
    if not value:
        return None
    categories = {item.strip() for item in value.split(",") if item.strip()}
    unknown = categories - set(events.ALL_CATEGORIES)
    if unknown:
        raise ValueError(
            f"Unknown --only categor{'y' if len(unknown) == 1 else 'ies'}: "
            f"{', '.join(sorted(unknown))}; choose from {', '.join(events.ALL_CATEGORIES)}"
        )
    return categories


def _watch_life_loop(
    paths: Paths,
    *,
    tail: int = 30,
    only: set[str] | None = None,
    no_thoughts: bool = False,
    since: str | None = None,
    reasoning: bool = False,
    full: bool = False,
    max_lines: int = 6,
    no_color: bool = False,
    json_mode: bool = False,
) -> None:
    path = paths.life_loop_log
    path.touch(exist_ok=True)
    print(f"Watching {path}. Ctrl-C detaches this viewer; it does not stop Artificium.\n")
    since_cutoff = display.parse_since(since) if since else None
    options = display.RenderOptions(
        color=display.colors_enabled(no_color=no_color),
        width=shutil.get_terminal_size(fallback=(100, 24)).columns,
        max_lines=max_lines,
        full=full,
        reasoning=reasoning,
        timestamps=True,
    )
    session = display.WatchSession()

    progress_active = {"value": False}

    def handle_line(line: str) -> None:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(record, dict):
            return
        if json_mode:
            print(line.rstrip("\n"), flush=True)
            return
        if not display.record_passes_filters(
            record, only_categories=only, no_thoughts=no_thoughts, since=since_cutoff
        ):
            return
        kind = str(record.get("kind") or "event")
        if kind in events.STREAMING_KINDS:
            if not sys.stdout.isatty():
                return
            body = display.render_event_body(record, options, session)
            if body:
                sys.stdout.write("\r\033[2K" + body[0])
                sys.stdout.flush()
                progress_active["value"] = True
            return
        if progress_active["value"]:
            # Commit the in-place progress line before scrolling past it.
            sys.stdout.write("\n")
            progress_active["value"] = False
        for text in display.render_record_lines(record, options, session):
            print(text, flush=True)

    follower = _LogFollower(path)
    try:
        for line in _tail_lines_from_end(path, tail):
            handle_line(line)
        follower.seek_end()
        while True:
            new_lines = follower.read_new_lines()
            if new_lines:
                for line in new_lines:
                    handle_line(line)
            else:
                time.sleep(0.25)
    except KeyboardInterrupt:
        _cli._print_detach_status(paths, client="life-loop viewer")
    finally:
        follower.close()


def _print_life_record(line: str) -> None:
    """Render one raw life-log line with the shared renderer.

    Kept as a small, stateless, public convenience: it is used by ``watch``
    when it briefly falls back to a per-line style, and existing callers
    (tests) construct a record and call it directly. A full ``watch``
    invocation instead calls :func:`artificium.display.render_record_lines`
    with one long-lived :class:`~artificium.display.WatchSession`, so
    turn separators, cache-miss detection, and cumulative summaries carry
    state across calls the way this single-line helper does not.
    """

    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(value, dict):
        return
    options = display.RenderOptions(color=display.colors_enabled(), timestamps=False)
    for text in display.render_event_body(value, options, display.WatchSession()):
        print(text, flush=True)

