"""Tests for artificium/display.py, artificium/events.py, and the ``watch``
side of artificium/cli.py (S1, S3, W1-W10).
"""
from __future__ import annotations

import datetime as dt
import glob
import io
import json
import os
import re
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from artificium import display, events
from artificium.cli import (
    _LogFollower,
    _parse_only,
    _print_life_record,
    _tail_lines_from_end,
    _watch_life_loop,
)
from artificium.filesystem import Paths, append_jsonl


def _iso(offset_seconds: float = 0.0, base: dt.datetime | None = None) -> str:
    base = base or dt.datetime(2026, 9, 23, 19, 44, 0, tzinfo=dt.timezone.utc)
    return (base + dt.timedelta(seconds=offset_seconds)).isoformat().replace("+00:00", "Z")


class EventRegistryCoverageCase(unittest.TestCase):
    """S3: every kind any module emits through records.life(...) is registered."""

    _LIFE_CALL = re.compile(r'\.life\(\s*\n?\s*"([a-zA-Z0-9_]+)"')

    def test_every_emitted_life_kind_has_a_category(self) -> None:
        root = Path(__file__).resolve().parent.parent / "artificium"
        emitted: set[str] = set()
        for path in root.glob("*.py"):
            if path.name in {"events.py", "display.py"}:
                continue  # the registry and renderer only *mention* kinds
            text = path.read_text(encoding="utf-8")
            emitted |= set(self._LIFE_CALL.findall(text))
        self.assertTrue(emitted, "expected to find at least one records.life(...) call")
        missing = emitted - set(events.EVENT_CATEGORIES)
        self.assertFalse(missing, f"kinds emitted but not registered in events.py: {missing}")

    def test_every_registered_kind_has_a_renderer_or_falls_back_safely(self) -> None:
        options = display.RenderOptions(color=False, timestamps=False)
        for kind in events.EVENT_CATEGORIES:
            record = {"kind": kind, "timestamp": _iso()}
            lines = display.render_event_body(record, options)
            # provider_reasoning is hidden unless options.reasoning; every
            # other registered kind must render to at least one line, and
            # never raise.
            if kind == events.KIND_PROVIDER_REASONING:
                self.assertEqual(lines, [])
            else:
                self.assertTrue(lines, f"{kind} rendered no lines")

    def test_unknown_kind_falls_back_to_generic_and_never_crashes(self) -> None:
        options = display.RenderOptions(color=False, timestamps=False)
        record = {"kind": "some_future_kind_nobody_registered", "detail": "x", "n": 3}
        lines = display.render_event_body(record, options)
        self.assertEqual(len(lines), 1)
        self.assertIn("some_future_kind_nobody_registered", lines[0])
        self.assertIn("detail=x", lines[0])

    def test_malformed_record_shapes_never_crash_the_renderer(self) -> None:
        options = display.RenderOptions(color=False, timestamps=False)
        malformed = [
            {"kind": "engine_response", "usage": "not-a-dict", "duration_seconds": "oops"},
            {"kind": "tool_call", "arguments": "not-a-dict"},
            {"kind": "context_usage", "estimated_tokens": None, "context_window_tokens": None},
            {"kind": "thought"},  # no content
            {"kind": None},
            {},
        ]
        for record in malformed:
            lines = display.render_event_body(record, options)
            self.assertIsInstance(lines, list)


class RenderingCase(unittest.TestCase):
    def setUp(self) -> None:
        self.options = display.RenderOptions(color=False, width=100, max_lines=6, full=False, timestamps=True)

    def test_engine_response_line_shows_cache_estimate_and_rate(self) -> None:
        record = {
            "kind": "engine_response",
            "timestamp": _iso(38),
            "request_id": "request_20260923T194400000000Z_aaaaaaaa6b9e",
            "duration_seconds": 38.2,
            "estimated_tokens": 105500,
            "usage_normalized": {
                "input_tokens": 133600,
                "output_tokens": 4600,
                "reasoning_tokens": 3700,
                "cache_read_tokens": 129400,
                "cache_write_tokens": None,
                "total_tokens": 138200,
            },
            "throughput": {"generation_tok_s": 121.3, "source": "server"},
            "finish_reason": "stop",
        }
        lines = display.render_record_lines(record, self.options)
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertIn("⇄ req 6b9e", line)
        self.assertIn("in 133.6k", line)
        self.assertIn("cache 129.4k (97%)", line)
        self.assertIn("out 4.6k (think 3.7k)", line)
        self.assertIn("38s", line)
        self.assertIn("121 tok/s", line)
        self.assertIn("stop", line)
        self.assertIn("est 105.5k (-21%)", line)

    def test_engine_response_without_normalized_usage_falls_back_to_normalize(self) -> None:
        # Old-format record: no usage_normalized, no estimated_tokens - as a
        # log written before those fields existed would have.
        record = {
            "kind": "engine_response",
            "timestamp": _iso(),
            "request_id": "request_x",
            "duration_seconds": 2.0,
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50},
            "finish_reason": "stop",
        }
        lines = display.render_record_lines(record, self.options)
        self.assertEqual(len(lines), 1)
        self.assertIn("in 1000", lines[0].replace("1.0k", "1000"))  # tolerate either formatting
        self.assertIn("out 50", lines[0])

    def test_cache_write_is_shown_when_reported(self) -> None:
        record = {
            "kind": "engine_response",
            "timestamp": _iso(),
            "request_id": "request_y",
            "usage_normalized": {
                "input_tokens": 20000, "output_tokens": 100, "reasoning_tokens": None,
                "cache_read_tokens": 0, "cache_write_tokens": 18000, "total_tokens": 20100,
            },
        }
        lines = display.render_record_lines(record, self.options)
        self.assertIn("+write 18.0k", lines[0])

    def test_context_usage_keeps_legacy_substrings(self) -> None:
        record = {
            "kind": "context_usage",
            "timestamp": _iso(),
            "estimated_tokens": 12345,
            "context_window_tokens": 20000,
            "context_percent": 61.7,
            "token_count_source": "estimate",
        }
        lines = display.render_record_lines(record, self.options)
        self.assertIn("[context] ~", lines[0])
        self.assertIn("/ 20,000 tokens", lines[0])

    def test_context_usage_shows_working_memory_bar_when_present(self) -> None:
        record = {
            "kind": "context_usage",
            "timestamp": _iso(),
            "estimated_tokens": 8000,
            "context_window_tokens": 20000,
            "context_percent": 40.0,
            "token_count_source": "provider",
            "working_memory_tokens": 10000,
        }
        lines = display.render_record_lines(record, self.options)
        self.assertIn("wm 8,000/10,000", lines[0])

    def test_engine_request_failed_preserves_error_and_hint_text(self) -> None:
        record = {
            "kind": "engine_request_failed", "timestamp": _iso(), "request_id": "test",
            "duration_seconds": 0, "error": "HTTP 400: Failed to tokenize prompt",
            "hint": "Inspect the request.", "model_log_path": "test.json",
        }
        output = io.StringIO()
        with redirect_stdout(output):
            _print_life_record(json.dumps(record))
        rendered = output.getvalue()
        self.assertIn("Failed to tokenize prompt", rendered)
        self.assertIn("Inspect the request.", rendered)

    def test_engine_blocked_preserves_restart_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            _print_life_record(json.dumps({"kind": "engine_blocked"}))
        self.assertIn("python3 artificium.py restart", output.getvalue())

    def test_guidance_notification_kept_compact(self) -> None:
        record = {"kind": "guidance_notification", "guidance_type": "meta_memory_size", "estimated_tokens": 4000}
        output = io.StringIO()
        with redirect_stdout(output):
            _print_life_record(json.dumps(record))
        self.assertIn("[guidance] meta_memory_size", output.getvalue())

    def test_tool_call_shows_command_for_run_shell(self) -> None:
        record = {"kind": "tool_call", "timestamp": _iso(), "name": "run_shell", "arguments": {"command": "ls -la"}}
        lines = display.render_record_lines(record, self.options)
        self.assertIn("command=ls -la", lines[0])

    def test_tool_result_shows_returncode_and_duration(self) -> None:
        record = {
            "kind": "tool_result", "timestamp": _iso(), "name": "run_shell",
            "result": {"status": "ok", "summary": "done", "returncode": 0, "duration_seconds": 1.25},
        }
        lines = display.render_record_lines(record, self.options)
        self.assertIn("rc=0", lines[0])
        self.assertIn("1.2s", lines[0])

    def test_thought_wraps_and_caps_lines(self) -> None:
        long_thought = " ".join(f"word{i}" for i in range(200))
        options = display.RenderOptions(color=False, width=40, max_lines=3, full=False, timestamps=False)
        lines = display.render_event_body({"kind": "thought", "content": long_thought}, options)
        self.assertLessEqual(len(lines), 4)  # 3 content lines + one "(+K lines)" marker
        self.assertTrue(lines[-1].strip().startswith("…"))

    def test_full_flag_disables_the_cap(self) -> None:
        long_thought = " ".join(f"word{i}" for i in range(200))
        capped = display.RenderOptions(color=False, width=40, max_lines=3, full=False, timestamps=False)
        uncapped = display.RenderOptions(color=False, width=40, max_lines=3, full=True, timestamps=False)
        capped_lines = display.render_event_body({"kind": "thought", "content": long_thought}, capped)
        full_lines = display.render_event_body({"kind": "thought", "content": long_thought}, uncapped)
        self.assertGreater(len(full_lines), len(capped_lines))
        self.assertFalse(full_lines[-1].strip().startswith("…"))

    def test_provider_reasoning_hidden_unless_reasoning_flag(self) -> None:
        record = {"kind": "provider_reasoning", "content": "internal deliberation"}
        hidden = display.RenderOptions(color=False, reasoning=False, timestamps=False)
        shown = display.RenderOptions(color=False, reasoning=True, timestamps=False)
        self.assertEqual(display.render_event_body(record, hidden), [])
        self.assertTrue(display.render_event_body(record, shown))

    def test_context_compacted_legacy_kind_still_renders(self) -> None:
        # Never emitted by current code, but old logs may still contain it.
        record = {"kind": "context_compacted", "path": "mind/x.txt", "before_tokens": 100, "after_tokens": 40}
        lines = display.render_event_body(record, self.options)
        self.assertIn("mind/x.txt", lines[0])
        self.assertIn("100", lines[0])
        self.assertIn("40", lines[0])

    def test_color_disabled_without_tty_or_with_no_color(self) -> None:
        self.assertFalse(display.colors_enabled(stream=io.StringIO()))
        self.assertFalse(display.colors_enabled(no_color=True, stream=io.StringIO()))
        os.environ["NO_COLOR"] = "1"
        try:
            self.assertFalse(display.colors_enabled(stream=io.StringIO()))
        finally:
            del os.environ["NO_COLOR"]

    def test_colored_output_wraps_line_in_category_color_and_reset(self) -> None:
        options = display.RenderOptions(color=True, timestamps=False)
        lines = display.render_event_body({"kind": "thought", "content": "hi"}, options)
        self.assertTrue(lines[0].startswith("\x1b["))
        self.assertTrue(lines[0].endswith("\x1b[0m"))


class CacheInsightAndSummaryCase(unittest.TestCase):
    def _response(self, seconds: float, *, input_tokens: int, cache_read: int) -> dict:
        return {
            "kind": "engine_response",
            "timestamp": _iso(seconds),
            "request_id": f"request_{int(seconds)}",
            "usage_normalized": {
                "input_tokens": input_tokens, "output_tokens": 100, "reasoning_tokens": None,
                "cache_read_tokens": cache_read, "cache_write_tokens": None,
                "total_tokens": input_tokens + 100,
            },
        }

    def test_cache_miss_after_long_idle_blames_ttl_expiry(self) -> None:
        session = display.WatchSession()
        options = display.RenderOptions(color=False, timestamps=False)
        display.render_event_body(self._response(0, input_tokens=130000, cache_read=126000), options, session)
        lines = display.render_event_body(
            self._response(400, input_tokens=134700, cache_read=0), options, session
        )
        joined = "\n".join(lines)
        self.assertIn("cache miss", joined)
        self.assertIn("provider cache likely expired", joined)
        self.assertIn("idle", joined)

    def test_cache_miss_after_short_gap_blames_rerouting(self) -> None:
        session = display.WatchSession()
        options = display.RenderOptions(color=False, timestamps=False)
        display.render_event_body(self._response(0, input_tokens=130000, cache_read=126000), options, session)
        lines = display.render_event_body(
            self._response(10, input_tokens=134700, cache_read=0), options, session
        )
        joined = "\n".join(lines)
        self.assertIn("cache miss", joined)
        self.assertIn("re-routed", joined)

    def test_no_warning_for_a_healthy_cache_hit(self) -> None:
        session = display.WatchSession()
        options = display.RenderOptions(color=False, timestamps=False)
        display.render_event_body(self._response(0, input_tokens=130000, cache_read=126000), options, session)
        lines = display.render_event_body(
            self._response(400, input_tokens=134700, cache_read=130000), options, session
        )
        self.assertFalse(any("cache miss" in line for line in lines))

    def test_no_warning_below_the_8k_input_floor(self) -> None:
        session = display.WatchSession()
        options = display.RenderOptions(color=False, timestamps=False)
        display.render_event_body(self._response(0, input_tokens=5000, cache_read=4000), options, session)
        lines = display.render_event_body(self._response(400, input_tokens=5000, cache_read=0), options, session)
        self.assertFalse(any("cache miss" in line for line in lines))

    def test_turn_completed_prints_cumulative_summary(self) -> None:
        session = display.WatchSession()
        options = display.RenderOptions(color=False, timestamps=False)
        display.render_event_body({"kind": "turn_started", "timestamp": _iso(0), "trigger": "manual"}, options, session)
        display.render_event_body(self._response(1, input_tokens=1000, cache_read=900), options, session)
        display.render_event_body(self._response(2, input_tokens=2000, cache_read=1800), options, session)
        lines = display.render_event_body(
            {"kind": "turn_completed", "timestamp": _iso(10)}, options, session
        )
        joined = "\n".join(lines)
        self.assertIn("turn completed", joined)
        self.assertIn("2 request(s)", joined)
        self.assertIn("session so far", joined)
        self.assertIn("2 req", joined)


class FilterCase(unittest.TestCase):
    def test_only_category_filters(self) -> None:
        engine_record = {"kind": "engine_response", "timestamp": _iso()}
        tool_record = {"kind": "tool_call", "timestamp": _iso()}
        self.assertTrue(display.record_passes_filters(engine_record, only_categories={"engine"}))
        self.assertFalse(display.record_passes_filters(tool_record, only_categories={"engine"}))

    def test_no_thoughts_hides_thought_output_and_reasoning(self) -> None:
        for kind in ("thought", "life_loop_output", "provider_reasoning"):
            self.assertFalse(display.record_passes_filters({"kind": kind}, no_thoughts=True))
        self.assertTrue(display.record_passes_filters({"kind": "tool_call"}, no_thoughts=True))

    def test_since_filters_out_earlier_records(self) -> None:
        cutoff = display.parse_since("2026-09-23T19:44:30Z")
        early = {"kind": "thought", "timestamp": "2026-09-23T19:44:00Z"}
        late = {"kind": "thought", "timestamp": "2026-09-23T19:45:00Z"}
        self.assertFalse(display.record_passes_filters(early, since=cutoff))
        self.assertTrue(display.record_passes_filters(late, since=cutoff))

    def test_parse_only_rejects_unknown_categories(self) -> None:
        with self.assertRaises(ValueError):
            _parse_only("engine,not-a-real-category")

    def test_parse_only_accepts_known_categories(self) -> None:
        self.assertEqual(_parse_only("engine, tools"), {"engine", "tools"})
        self.assertIsNone(_parse_only(None))
        self.assertIsNone(_parse_only(""))


class TailAndFollowCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "life-loop.jsonl"

    def _write_lines(self, count: int, *, start: int = 0) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for i in range(start, start + count):
                handle.write(json.dumps({"kind": "thought", "content": f"line {i}"}) + "\n")

    def test_tail_from_end_matches_readlines_on_a_large_file(self) -> None:
        self._write_lines(5000)
        expected = self.path.read_text(encoding="utf-8").splitlines()[-37:]
        got = _tail_lines_from_end(self.path, 37)
        self.assertEqual(got, expected)

    def test_tail_from_end_handles_fewer_lines_than_requested(self) -> None:
        self._write_lines(5)
        got = _tail_lines_from_end(self.path, 100)
        self.assertEqual(len(got), 5)

    def test_tail_from_end_zero_is_empty(self) -> None:
        self._write_lines(5)
        self.assertEqual(_tail_lines_from_end(self.path, 0), [])

    def test_log_follower_reads_newly_appended_lines(self) -> None:
        self._write_lines(3)
        follower = _LogFollower(self.path)
        self.addCleanup(follower.close)
        follower.seek_end()
        self.assertEqual(follower.read_new_lines(), [])
        self._write_lines(2, start=3)
        new_lines = follower.read_new_lines()
        self.assertEqual(len(new_lines), 2)
        self.assertIn("line 3", new_lines[0])

    def test_log_follower_ignores_a_partial_trailing_write(self) -> None:
        self._write_lines(1)
        follower = _LogFollower(self.path)
        self.addCleanup(follower.close)
        follower.seek_end()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"kind": "thought", "content": "unfinished')  # no closing / newline
        self.assertEqual(follower.read_new_lines(), [])
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('"}\n')
        finished = follower.read_new_lines()
        self.assertEqual(len(finished), 1)
        self.assertIn("unfinished", finished[0])

    def test_log_follower_reopens_on_truncation(self) -> None:
        self._write_lines(10)
        follower = _LogFollower(self.path)
        self.addCleanup(follower.close)
        follower.seek_end()
        # Copy-truncate rotation: same inode, shorter file.
        with self.path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "thought", "content": "after rotation"}) + "\n")
        lines = follower.read_new_lines()
        self.assertEqual(len(lines), 1)
        self.assertIn("after rotation", lines[0])

    def test_log_follower_reopens_on_inode_rotation(self) -> None:
        self._write_lines(10)
        follower = _LogFollower(self.path)
        self.addCleanup(follower.close)
        follower.seek_end()
        # rename+recreate rotation: a brand-new inode at the same path.
        rotated = self.path.with_suffix(".jsonl.1")
        os.replace(self.path, rotated)
        self.path.write_text(json.dumps({"kind": "thought", "content": "new file"}) + "\n", encoding="utf-8")
        lines = follower.read_new_lines()
        self.assertEqual(len(lines), 1)
        self.assertIn("new file", lines[0])


class WatchLoopIntegrationCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))

    def test_json_mode_passes_records_through_unrendered(self) -> None:
        record = {"kind": "thought", "content": "hello", "timestamp": _iso()}
        append_jsonl(self.paths.life_loop_log, record)
        output = io.StringIO()
        import time as time_module

        with redirect_stdout(output):
            from unittest import mock

            with mock.patch("artificium.cli.time.sleep", side_effect=KeyboardInterrupt):
                _watch_life_loop(self.paths, tail=5, json_mode=True)
        rendered = output.getvalue().strip().splitlines()
        payloads = [json.loads(line) for line in rendered if line.startswith("{")]
        self.assertTrue(any(p.get("content") == "hello" for p in payloads))

    def test_only_filter_hides_other_categories_in_a_real_watch_run(self) -> None:
        append_jsonl(self.paths.life_loop_log, {"kind": "thought", "content": "hidden", "timestamp": _iso()})
        append_jsonl(
            self.paths.life_loop_log,
            {"kind": "tool_call", "name": "run_shell", "arguments": {"command": "ls"}, "timestamp": _iso()},
        )
        output = io.StringIO()
        from unittest import mock

        with redirect_stdout(output):
            with mock.patch("artificium.cli.time.sleep", side_effect=KeyboardInterrupt):
                _watch_life_loop(self.paths, tail=5, only={"tools"})
        rendered = output.getvalue()
        self.assertNotIn("hidden", rendered)
        self.assertIn("run_shell", rendered)


if __name__ == "__main__":
    unittest.main()
