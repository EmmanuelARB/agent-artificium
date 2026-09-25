"""Tests for the records.py rework: B7 (gzip model logs) and B10 (feature
summary buffering, working_context_appended compaction), plus backward
compatibility of every public name records.py has always exported.
"""
from __future__ import annotations

import gzip
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from artificium.filesystem import Paths, atomic_write_json, read_json, sha256_text
from artificium.records import (
    Console,
    Records,
    format_throughput,
    generation_token_count,
    load_model_exchange,
    throughput_stats,
)


class RecordsStorageCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = Paths(self.root)
        self.records = Records(self.paths)
        self.addCleanup(self.records.flush)

    # -- backward compatibility -------------------------------------------------

    def test_every_previously_public_name_is_still_importable(self) -> None:
        self.assertTrue(callable(Records))
        self.assertTrue(callable(Console))
        self.assertTrue(callable(generation_token_count))
        self.assertTrue(callable(throughput_stats))
        self.assertTrue(callable(format_throughput))
        # Signatures unchanged.
        self.assertEqual(generation_token_count({"output_tokens": 3}), 3)
        self.assertEqual(
            throughput_stats(None, None, None), {"prefill_tok_s": None, "generation_tok_s": None, "source": "none"}
        )
        self.assertEqual(format_throughput(None), "")

    # -- B7: gzip-compressed model logs -----------------------------------------

    def test_model_log_path_stays_plain_json(self) -> None:
        path = self.records.model_log_path("request_abc")
        self.assertTrue(str(path).endswith("request_abc.json"))

    def test_recent_model_log_is_plain_and_round_trips(self) -> None:
        self.records.model_exchange(
            request_id="request_1",
            messages=[{"role": "user", "content": "hi"}],
            request_parameters={"body": {"model": "x"}},
            response={"content": "hello", "usage": {"input_tokens": 5, "output_tokens": 2}},
        )
        path = self.records.model_log_path("request_1")
        # The agent reads recent logs with ordinary file tools.
        self.assertEqual(json.loads(path.read_text())["request_id"], "request_1")
        payload = load_model_exchange(path)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(payload["response"]["content"], "hello")

    def test_older_model_logs_are_gzip_compressed_and_still_readable(self) -> None:
        big_messages = [{"role": "user", "content": "x" * 2000} for _ in range(20)]
        for index in range(5):
            self.records.model_exchange(request_id=f"request_{index}", messages=big_messages)
            path = self.records.model_log_path(f"request_{index}")
            os.utime(path, ns=(index * 10**9, index * 10**9))
        self.assertEqual(self.records.compress_old_model_logs(keep=2), 3)
        plain = sorted(item.name for item in self.paths.model_log.glob("*.json"))
        self.assertEqual(plain, ["request_3.json", "request_4.json"])
        old = self.paths.model_log / "request_0.json.gz"
        with old.open("rb") as handle:
            self.assertEqual(handle.read(2), b"\x1f\x8b")
        self.assertLess(old.stat().st_size, len(json.dumps({"messages": big_messages})) / 3)
        # A path recorded before compression still resolves.
        loaded = load_model_exchange(self.records.model_log_path("request_0"))
        self.assertEqual(loaded["messages"], big_messages)

    def test_model_exchange_compresses_beyond_the_plain_window(self) -> None:
        self.records.PLAIN_MODEL_LOGS = 3
        self.records.MODEL_LOG_COMPRESS_EVERY = 1
        for index in range(6):
            self.records.model_exchange(request_id=f"request_{index}", messages=[])
        self.assertLessEqual(len(list(self.paths.model_log.glob("*.json"))), 3)
        self.assertGreaterEqual(len(list(self.paths.model_log.glob("*.json.gz"))), 2)

    def test_load_model_exchange_reads_an_old_plain_json_log(self) -> None:
        # A log written by a version before B7: plain, uncompressed JSON,
        # sitting at the old `request_id.json` path with no `.gz`.
        old_path = self.paths.model_log / "request_old.json"
        payload = {"request_id": "request_old", "messages": [{"role": "user", "content": "legacy"}]}
        atomic_write_json(old_path, payload)
        loaded = load_model_exchange(old_path)
        self.assertEqual(loaded["request_id"], "request_old")
        self.assertEqual(loaded["messages"][0]["content"], "legacy")

    def test_load_model_exchange_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_model_exchange(self.paths.model_log / "does_not_exist.json")

    # -- B10: working_context_appended compaction --------------------------------

    def test_working_context_appended_is_compacted_not_duplicated_whole(self) -> None:
        big_content = "word " * 2000  # far longer than the 200-char preview
        record = self.records.emit(
            "working_context_appended",
            origin="life_loop_input",
            message={"role": "user", "content": big_content},
        )
        message = record["message"]
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["message_sha256"], sha256_text(big_content))
        self.assertEqual(message["characters"], len(big_content))
        self.assertEqual(message["preview"], big_content[:200])
        self.assertNotIn("content", message)
        # And it is what actually landed in both the lifetime and feature logs.
        from artificium.filesystem import read_jsonl

        lifetime = read_jsonl(self.paths.lifetime_log)
        self.assertEqual(lifetime[-1]["message"]["preview"], big_content[:200])
        self.assertLess(len(json.dumps(lifetime[-1])), len(big_content))

    def test_working_context_appended_tolerates_non_dict_message(self) -> None:
        # Never crash on an unexpected shape; pass it through unchanged.
        record = self.records.emit("working_context_appended", origin="x", message="not-a-dict")
        self.assertEqual(record["message"], "not-a-dict")

    # -- B10: feature_summary.json buffering --------------------------------------

    def test_feature_summary_writes_are_batched_not_per_event(self) -> None:
        with mock.patch("artificium.records.atomic_write_json") as write:
            for _ in range(5):
                self.records.emit("attention_opened", stream_id="s")
            # Below both the event-count and time thresholds: nothing flushed yet.
            write.assert_not_called()
            self.records.flush()
            self.assertEqual(write.call_count, 1)

    def test_feature_summary_flushes_after_enough_events(self) -> None:
        with mock.patch("artificium.records.atomic_write_json") as write:
            for _ in range(self.records._FLUSH_EVERY_EVENTS):
                self.records.emit("attention_opened", stream_id="s")
            self.assertEqual(write.call_count, 1)

    def test_feature_usage_reads_are_consistent_with_unflushed_emits(self) -> None:
        before = self.records.feature_usage().get("features", {}).get("infinite-attention", 0)
        self.records.emit("attention_opened", stream_id="s")
        # Not flushed yet, but a same-process read must still see it.
        usage = self.records.feature_usage()
        self.assertEqual(usage["features"]["infinite-attention"], before + 1)
        # The on-disk file itself was not necessarily touched yet.
        disk = read_json(self.paths.feature_summary, {})
        disk_count = (disk.get("features") or {}).get("infinite-attention", 0)
        self.assertLessEqual(disk_count, before + 1)

    def test_flush_merges_with_another_processes_counts_not_overwrite(self) -> None:
        # Simulate another process having already written committed counts.
        atomic_write_json(
            self.paths.feature_summary,
            {"features": {"memory": 7}, "kinds": {"long_term_memory_saved": 7}, "total_operational_events": 7},
        )
        self.records.emit("long_term_memory_saved", path="x")
        self.records.flush()
        summary = read_json(self.paths.feature_summary, {})
        self.assertEqual(summary["features"]["memory"], 8)
        self.assertEqual(summary["total_operational_events"], 8)

    def test_flush_runs_at_process_exit_via_atexit(self) -> None:
        import atexit as atexit_module

        with mock.patch.object(atexit_module, "register") as register:
            Records(self.paths)
        register.assert_called_once()

    def test_exit_flush_never_recreates_a_deleted_workspace(self) -> None:
        import shutil

        records = Records(self.paths)
        records.emit("tool_executed", name="x")
        shutil.rmtree(self.paths.root)
        records._flush_at_exit()
        self.assertFalse(self.paths.root.exists())


if __name__ == "__main__":
    unittest.main()
