"""Tests for B3 (token calibration), B6 (stable system prompt), B10 (fewer
redundant reads), B11 (skip the provider count when safe), and the richer
engine_response/engine_progress records.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from artificium.config import Config, ConfigStore
from artificium.context_budget import TokenCount
from artificium.engine import Engine, EngineError, EngineReply
from artificium.filesystem import Paths, read_json
from artificium.life_loop import ToolIntent
from artificium.memory import CalibrationState, load_calibration, update_calibration
from artificium.records import Console
from artificium.runtime import Artificium

ROOT = Path(__file__).resolve().parents[2]


def call(name, **arguments):
    return "<tool_call>" + json.dumps({"tool": name, **arguments}) + "</tool_call>"


class ScriptedEngine(Engine):
    """Same shape as the other suites' fixture: scripted replies, optional usage."""

    def __init__(self, *responses, usage=None):
        self.responses = list(responses)
        self.usage = usage or {}
        self.requests = []

    def complete(self, messages):
        self.requests.append(messages)
        response = self.responses.pop(0) if self.responses else "<think>Waiting.</think>"
        if isinstance(response, Exception):
            raise response
        return EngineReply(response, usage=dict(self.usage))


class CountingEngine(Engine):
    """Tracks every count_input_tokens call so B11's skip decision is observable."""

    def __init__(self, *responses, count=None, usage=None):
        self.responses = list(responses)
        self.count = count
        self.usage = usage or {}
        self.requests = []
        self.count_calls = 0

    def count_input_tokens(self, prepared):
        self.count_calls += 1
        return self.count

    def complete(self, messages):
        self.requests.append(messages)
        response = self.responses.pop(0) if self.responses else "<think>Waiting.</think>"
        if isinstance(response, Exception):
            raise response
        return EngineReply(response, usage=dict(self.usage))


class ProgressEngine(Engine):
    """Has a progress_callback attribute and calls it once per completion,
    mirroring how the streaming JSONEngine path calls it.
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.progress_callback = None
        self.progress_snapshots = []

    def complete(self, messages):
        self.requests.append(messages)
        # Record whether a callback was wired for this request before using it,
        # so the test can tell wiring-before-call apart from clearing-after.
        self.progress_snapshots.append(self.progress_callback is not None)
        if self.progress_callback is not None:
            self.progress_callback(
                {"generated_chars": 10, "reasoning_chars": 2, "elapsed_seconds": 0.01}
            )
        response = self.responses.pop(0) if self.responses else "<think>Waiting.</think>"
        if isinstance(response, Exception):
            raise response
        return EngineReply(response, usage={"input_tokens": 321, "output_tokens": 7})


class RuntimeTokensCacheCase(unittest.TestCase):
    def setUp(self):
        check = mock.patch(
            "artificium.setup.verify_connection",
            side_effect=lambda paths, config, key, **kw: (config, {}),
        )
        check.start()
        self.addCleanup(check.stop)
        network = mock.patch("urllib.request.urlopen", side_effect=OSError("offline fixture"))
        network.start()
        self.addCleanup(network.stop)
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        shutil.copytree(ROOT / "app/seed", self.paths.seed)
        self.paths.ensure_layout()
        self.config = Config(
            provider="custom", model="test-model", base_url="http://example.invalid/v1",
            context_window_tokens=200_000, max_life_loop_rounds=1,
        )
        ConfigStore(self.paths).save(self.config)

    def agent(self, engine=None, **settings):
        self.config = dataclasses.replace(self.config, **settings)
        ConfigStore(self.paths).save(self.config)
        agent = Artificium(self.paths, engine=engine or ScriptedEngine(), console=Console(quiet=True))
        agent.initialization.finish("Self, meta-memory, tools, environment, and pending events inspected.")
        return agent

    # -- B6a: tool_catalog is placed before the pinned Self/meta-memory block --

    def test_system_prompt_places_tool_catalog_before_pinned_block(self):
        agent = self.agent()
        prompt = agent.system_prompt()
        self.assertLess(prompt.index("Core textual tools"), prompt.index("PINNED SELF"))

    # -- The transient state-header message is tagged for cache placement --

    def test_state_header_message_is_tagged(self):
        agent = self.agent()
        messages = agent._request_messages([], wake_reason="test")
        self.assertEqual(messages[-1]["_artificium"], {"kind": "state_header"})
        self.assertIn("[SYSTEM STATE]", messages[-1]["content"])

    # -- B6b: pinned-mind snapshot freezes Self/meta-memory between rebuilds --

    def test_pinned_snapshot_freezes_and_notifies_change_exactly_once(self):
        agent = self.agent(pinned_mind_snapshot=True)
        refreshed = [
            item for item in agent.records.recent_life(50)
            if item.get("kind") == "pinned_mind_refreshed"
        ]
        self.assertEqual(len(refreshed), 1)
        self.assertEqual(refreshed[0]["reason"], "process_start")

        before = agent.system_prompt()
        agent.paths.self_file.write_text("A brand-new edited Self, never pinned yet.\n")
        # The live edit must not show up until a rebuild or a notice.
        after_edit = agent.system_prompt()
        self.assertEqual(before, after_edit)
        self.assertNotIn("brand-new edited Self", after_edit)

        first_notices = agent._pinned_mind_notices()
        self.assertEqual(len(first_notices), 1)
        self.assertIn("mind/self.txt", first_notices[0])
        self.assertIn("brand-new edited Self", first_notices[0])
        # A round that never committed (failed or interrupted request) shows
        # the same change again.
        self.assertEqual(agent._pinned_mind_notices(), first_notices)
        # Once a reply commits the round, an unchanged file gives no notice.
        agent._commit_pinned_mind_notices()
        second_notices = agent._pinned_mind_notices()
        self.assertEqual(second_notices, [])

        # Still frozen: the pinned block itself only changes at a rebuild.
        self.assertNotIn("brand-new edited Self", agent.system_prompt())

    def test_pinned_snapshot_refreshes_on_working_memory_offload(self):
        agent = self.agent(pinned_mind_snapshot=True)
        agent.paths.self_file.write_text("Self content visible only after offload.\n")
        agent.tools.offload_working_memory()  # first call: reflection_required
        intent = ToolIntent(
            id="tool_1", name="offload_working_memory",
            arguments={
                "path": "context/pinned-mind-offload-test",
                "checkpoint": "Objective done; evidence checked; next step is none; nothing pending.",
                "retrieve_when": "Retrieve when checking the pinned-mind offload test.",
                "reflection_complete": True,
            },
        )
        ended = agent._execute_tools([intent])
        self.assertFalse(ended)
        self.assertIn("Self content visible only after offload", agent.system_prompt())
        refreshed = [
            item for item in agent.records.recent_life(50)
            if item.get("kind") == "pinned_mind_refreshed"
        ]
        self.assertEqual(refreshed[-1]["reason"], "offload_working_memory")
        self.assertTrue(refreshed[-1]["self_changed"])

    def test_pinned_snapshot_disabled_behaves_exactly_as_live_reads(self):
        agent = self.agent(pinned_mind_snapshot=False)
        agent.paths.self_file.write_text("Immediately-visible Self, snapshotting is off.\n")
        self.assertIn("Immediately-visible Self", agent.system_prompt())
        self.assertEqual(agent._pinned_mind_notices(), [])
        self.assertIsNone(agent._load_pinned_snapshot())

    # -- B3: real/estimate calibration converges, persists, resets on model change --

    def test_calibration_converges_and_persists_across_restart(self):
        engine = ScriptedEngine("<think>One round.</think>", usage={"input_tokens": 50_000})
        agent = self.agent(engine)
        agent.run_turn(trigger="calibration-test")
        raw_estimate = agent.estimator.messages(engine.requests[-1])
        state = load_calibration(agent.paths, agent.config.provider, agent.config.model)
        self.assertEqual(state.samples, 1)
        expected_ratio = max(0.8, min(2.0, 50_000 / raw_estimate))
        self.assertAlmostEqual(state.ratio, expected_ratio, places=6)

        restarted = self.agent(ScriptedEngine("<think>Idle.</think>"))
        restarted_state = load_calibration(restarted.paths, restarted.config.provider, restarted.config.model)
        self.assertEqual(restarted_state.samples, 1)
        self.assertAlmostEqual(restarted_state.ratio, expected_ratio, places=6)

    def test_calibration_resets_when_the_model_changes(self):
        update_calibration(
            self.paths, provider="custom", model="test-model", real_tokens=9_000, raw_estimate=3_000,
        )
        seeded = load_calibration(self.paths, "custom", "test-model")
        self.assertEqual(seeded.samples, 1)
        after_model_change = load_calibration(self.paths, "custom", "a-different-model")
        self.assertEqual(after_model_change.samples, 0)
        self.assertEqual(after_model_change.ratio, 1.0)

    # -- B11: skip the provider preflight count once calibration exists and is safe --

    def test_measure_skips_provider_count_once_calibrated_and_comfortably_small(self):
        engine = CountingEngine(count=999_999)
        agent = self.agent(engine, mandatory_offload=False)
        update_calibration(
            agent.paths, provider=agent.config.provider, model=agent.config.model,
            real_tokens=1_000, raw_estimate=1_000,
        )
        small_messages = [{"role": "user", "content": "hi"}]
        prepared, count = agent._measure(small_messages)
        self.assertEqual(engine.count_calls, 0)
        self.assertEqual(count.source, "calibrated")

    def test_measure_still_calls_provider_count_when_estimate_is_not_safely_small(self):
        engine = CountingEngine(count=150_000)
        agent = self.agent(engine, mandatory_offload=False)
        update_calibration(
            agent.paths, provider=agent.config.provider, model=agent.config.model,
            real_tokens=1_000, raw_estimate=1_000,
        )
        # Comfortably above 60% of the 200,000-token window once calibrated.
        big_messages = [{"role": "user", "content": "x" * 900_000}]
        prepared, count = agent._measure(big_messages)
        self.assertEqual(engine.count_calls, 1)
        self.assertEqual(count, TokenCount(150_000, "provider"))

    def test_measure_calls_provider_count_before_any_calibration_exists(self):
        engine = CountingEngine(count=42_000)
        agent = self.agent(engine, mandatory_offload=False)
        small_messages = [{"role": "user", "content": "hi"}]
        prepared, count = agent._measure(small_messages)
        self.assertEqual(engine.count_calls, 1)
        self.assertEqual(count, TokenCount(42_000, "provider"))

    def test_measure_never_skips_the_provider_count_when_images_are_present(self):
        engine = CountingEngine(count=1_000)
        agent = self.agent(engine, mandatory_offload=False)
        update_calibration(
            agent.paths, provider=agent.config.provider, model=agent.config.model,
            real_tokens=1_000, raw_estimate=1_000,
        )
        messages = [{"role": "user", "content": [{"type": "artificium_image", "path": "/tmp/x.png"}]}]
        agent._measure(messages)
        self.assertEqual(engine.count_calls, 1)

    # -- Richer records: usage_normalized, estimated_tokens, token_count_source --

    def test_engine_response_life_record_carries_normalized_usage_and_count(self):
        engine = ScriptedEngine(
            "<think>Report usage.</think>", usage={"input_tokens": 555, "output_tokens": 12},
        )
        agent = self.agent(engine)
        agent.run_turn(trigger="usage-test")
        responses = [
            item for item in agent.records.recent_life(100) if item.get("kind") == "engine_response"
        ]
        self.assertEqual(len(responses), 1)
        record = responses[0]
        self.assertEqual(record["usage_normalized"]["input_tokens"], 555)
        self.assertEqual(record["usage_normalized"]["output_tokens"], 12)
        self.assertIn("estimated_tokens", record)
        self.assertIn("token_count_source", record)

    # -- Streaming progress records (Task 6) --

    def test_progress_callback_is_wired_and_emits_throttled_life_records(self):
        engine = ProgressEngine("<think>Streaming done.</think>")
        agent = self.agent(engine)
        agent.run_turn(trigger="progress-test")
        self.assertTrue(engine.progress_snapshots and engine.progress_snapshots[0])
        self.assertIsNone(engine.progress_callback)
        progress = [
            item for item in agent.records.recent_life(100) if item.get("kind") == "engine_progress"
        ]
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["generated_chars"], 10)
        self.assertEqual(progress[0]["reasoning_chars"], 2)

    def test_engines_without_progress_callback_are_left_untouched(self):
        engine = ScriptedEngine("<think>No callback attribute here.</think>")
        agent = self.agent(engine)
        agent.run_turn(trigger="no-progress-test")
        self.assertFalse(hasattr(engine, "progress_callback"))
        progress = [
            item for item in agent.records.recent_life(100) if item.get("kind") == "engine_progress"
        ]
        self.assertEqual(progress, [])

    # -- B10: WorkingMemory.load() is cached by (mtime, size) --

    def test_working_memory_load_is_cached_until_a_real_edit(self):
        agent = self.agent()
        agent.working.append({"role": "user", "content": "first"}, origin="test")
        first = agent.working.load()
        second = agent.working.load()
        self.assertIsNot(first, second)  # copy-safe: callers cannot mutate the cache
        self.assertEqual(len(first), 1)
        first.append({"role": "user", "content": "mutated by caller"})
        self.assertEqual(len(agent.working.load()), 1)  # cache unaffected by that mutation
        agent.working.append({"role": "user", "content": "second"}, origin="test")
        self.assertEqual(len(agent.working.load()), 2)  # a real edit is still visible


if __name__ == "__main__":
    unittest.main()
