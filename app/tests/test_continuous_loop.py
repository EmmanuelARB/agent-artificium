from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from artificium.config import Config, ConfigStore
from artificium.engine import Engine, EngineReply
from artificium.filesystem import Paths, read_json, read_jsonl
from artificium.records import Console
from artificium.runtime import Artificium


ROOT = Path(__file__).resolve().parents[2]


def call(name, **arguments):
    return "<tool_call>" + json.dumps({"tool": name, **arguments}) + "</tool_call>"


class RecordingEngine(Engine):
    def __init__(self, respond):
        self.respond = respond
        self.requests = []

    def complete(self, messages):
        self.requests.append(messages)
        return EngineReply(self.respond(len(self.requests), messages))


class ContinuousLoopCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        shutil.copytree(ROOT / "app/seed", self.paths.seed)
        self.paths.ensure_layout()
        self.clock = 0.0
        clock = mock.patch("artificium.runtime.time.monotonic", side_effect=lambda: self.clock)
        clock.start()
        self.addCleanup(clock.stop)

    def agent(self, respond, **settings):
        defaults = dict(provider="custom", model="test", base_url="http://example.invalid/v1", context_window_tokens=500_000,
                        mandatory_offload=False)
        ConfigStore(self.paths).save(Config(**(defaults | settings)))
        engine = RecordingEngine(respond)
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.initialization.finish("The test instance is initialized and ready for its task.")
        agent.working.append({"role": "user", "content": "KEEP_THIS_RESEARCH_CONTEXT"}, origin="test")
        return agent, engine

    def stop_after_tools(self, agent, count):
        execute = agent.tools.execute
        self.executions = 0

        def execute_and_stop(intent):
            result = execute(intent)
            self.executions += 1
            if self.executions >= count:
                agent._stop = True
            return result

        patch = mock.patch.object(agent.tools, "execute", side_effect=execute_and_stop)
        patch.start()
        self.addCleanup(patch.stop)

    def assert_no_limit_notices(self, agent):
        records = read_jsonl(self.paths.life_loop_log) + read_jsonl(self.paths.lifetime_log)
        self.assertNotIn("The prior turn reached", json.dumps(records))
        self.assertFalse(agent.notifications.has_new())

    def test_continuous_work_passes_64_rounds_with_history_and_new_events(self):
        def respond(number, messages):
            self.clock += 1
            if number == 64:
                agent.notifications.create(type="external_event", source="user",
                                           summary="NEW_USER_MESSAGE_AFTER_ROUND_64")
            return f"<output>RESEARCH_STEP_{number}</output>" + call("read_file", path="mind/self.txt")

        agent, engine = self.agent(respond)
        self.stop_after_tools(agent, 66)
        agent.run_forever(quiet=True)

        self.assertEqual(len(engine.requests), 66)
        request_65 = json.dumps(engine.requests[64])
        self.assertIn("KEEP_THIS_RESEARCH_CONTEXT", request_65)
        self.assertIn("RESEARCH_STEP_1", request_65)
        self.assertIn("RESEARCH_STEP_64", request_65)
        self.assertIn("NEW_USER_MESSAGE_AFTER_ROUND_64", request_65)
        life = read_jsonl(self.paths.life_loop_log)
        self.assertEqual(sum(r["kind"] == "turn_started" for r in life), 1)
        self.assertEqual(sum(r["kind"] == "engine_response" for r in life), 66)
        self.assertIn("RESEARCH_STEP_66", self.paths.working_context.read_text())
        self.assert_no_limit_notices(agent)

    def test_long_generations_continue_past_15_minutes_without_a_new_turn(self):
        def respond(number, messages):
            self.clock += 1600  # Simulate each successful inference taking over 26 minutes.
            self.assertEqual(read_json(self.paths.runtime_state)["round"], number)
            return call("read_file", path="mind/self.txt")

        agent, engine = self.agent(respond)
        self.stop_after_tools(agent, 3)
        agent.run_forever(quiet=True)

        self.assertEqual(len(engine.requests), 3)
        life = read_jsonl(self.paths.life_loop_log)
        self.assertEqual(sum(r["kind"] == "turn_started" for r in life), 1)
        self.assertTrue(all(r["duration_seconds"] == 1600 for r in life if r["kind"] == "engine_response"))
        self.assert_no_limit_notices(agent)

    def test_stop_during_a_tool_prevents_another_model_request(self):
        agent, engine = self.agent(lambda *_: call("read_file", path="mind/self.txt"))
        self.stop_after_tools(agent, 1)
        agent.run_forever(quiet=True)
        self.assertEqual(len(engine.requests), 1)
        self.assertIn("KEEP_THIS_RESEARCH_CONTEXT", self.paths.working_context.read_text())
        self.assertEqual(read_json(self.paths.runtime_state)["status"], "stopped")

    def test_text_only_responses_continue_immediately_with_context(self):
        def respond(number, messages):
            if number <= 2:
                return f"<think>REASONING_STEP_{number}</think>"
            return call("read_file", path="mind/self.txt")

        agent, engine = self.agent(respond)
        self.stop_after_tools(agent, 2)
        with mock.patch("artificium.runtime.time.sleep", side_effect=AssertionError("unexpected wait")):
            agent.run_forever(quiet=True)
        self.assertEqual(len(engine.requests), 4)
        self.assertIn("REASONING_STEP_1", json.dumps(engine.requests[-1]))
        self.assertIn("REASONING_STEP_2", json.dumps(engine.requests[-1]))
        self.assertEqual(json.dumps(engine.requests[-1]).count("Type: life_loop_continuation"), 2)
        turns = [r["trigger"] for r in read_jsonl(self.paths.life_loop_log) if r["kind"] == "turn_started"]
        self.assertEqual(turns, ["startup", "continuation", "continuation"])

    def test_stop_during_text_only_generation_prevents_continuation(self):
        def respond(number, messages):
            agent._stop = True
            return "<think>Finished generating.</think>"

        agent, engine = self.agent(respond)
        agent.run_forever(quiet=True)
        self.assertEqual(len(engine.requests), 1)
        self.assertEqual(read_json(self.paths.runtime_state)["status"], "stopped")

    def test_pending_notification_takes_priority_over_continuation(self):
        def respond(number, messages):
            if number == 1:
                agent.notifications.create(type="external_event", source="user", summary="NEW_EVENT")
                return "<think>Finished a reasoning step.</think>"
            return call("read_file", path="mind/self.txt")

        agent, engine = self.agent(respond)
        self.stop_after_tools(agent, 1)
        agent.run_forever(quiet=True)
        self.assertIn("NEW_EVENT", json.dumps(engine.requests[1]))
        turns = [r["trigger"] for r in read_jsonl(self.paths.life_loop_log) if r["kind"] == "turn_started"]
        self.assertEqual(turns, ["startup", "notification"])

    def test_repeated_no_action_continuations_still_back_off(self):
        agent, engine = self.agent(lambda *_: "<think>Nothing to do.</think>")

        def observe_backoff(_seconds):
            state = read_json(self.paths.sleep_state)
            self.assertTrue(state["active"])
            self.assertEqual(state["reason"], "repeated_no_action_output")
            agent._stop = True

        with mock.patch("artificium.runtime.time.sleep", side_effect=observe_backoff):
            agent.run_forever(quiet=True)
        self.assertEqual(len(engine.requests), 3)

    def test_timed_sleep_waits_for_its_deadline_before_continuing(self):
        request_times = []

        def respond(number, messages):
            request_times.append(self.clock)
            if number <= 2:
                return call("sleep", mode="timed", seconds=5, reflection_complete=number == 2)
            return call("read_file", path="mind/self.txt")

        agent, engine = self.agent(respond)
        self.stop_after_tools(agent, 3)

        def advance(seconds):
            self.clock += seconds
            self.assertLessEqual(self.clock, 5, "Timed sleep did not wake")

        with mock.patch("artificium.runtime.time.time", side_effect=lambda: 1000 + self.clock), \
             mock.patch("artificium.runtime.time.sleep", side_effect=advance):
            agent.run_forever(quiet=True)
        self.assertEqual(request_times, [0, 0, 5])
        self.assertIn("timer", json.dumps(engine.requests[-1]))

    def test_sleep_still_completes_reflection_and_waits_for_an_event(self):
        def respond(number, messages):
            self.assertLessEqual(number, 2)
            return call("sleep", mode="until_event", reflection_complete=number == 2)

        agent, engine = self.agent(respond, max_life_loop_rounds=1)

        def observe_sleep(_seconds):
            sleep = read_json(self.paths.sleep_state)
            self.assertTrue(sleep["active"])
            self.assertEqual(sleep["mode"], "until_event")
            agent._stop = True

        with mock.patch("artificium.runtime.time.sleep", side_effect=observe_sleep):
            agent.run_forever(quiet=True)
        self.assertEqual(len(engine.requests), 2)
        self.assertIn("reflection", json.dumps(engine.requests[1]).lower())
        self.assert_no_limit_notices(agent)

    def test_credentials_are_reloaded_between_rounds(self):
        agent, engine = self.agent(lambda *_: call("read_file", path="mind/self.txt"))
        replacement = RecordingEngine(lambda *_: call("read_file", path="mind/self.txt"))
        signatures = iter([None, None, (1, 1)])  # Outer loop, round 1, then changed key.
        self.stop_after_tools(agent, 2)
        with mock.patch.object(agent, "_secret_signature", side_effect=lambda: next(signatures, (1, 1))), \
             mock.patch.object(agent.secrets, "resolve_api_key", return_value="updated-key"), \
             mock.patch("artificium.runtime.make_engine", return_value=replacement) as make:
            agent.run_forever(quiet=True)
        self.assertEqual(len(engine.requests), 1)
        self.assertEqual(len(replacement.requests), 1)
        make.assert_called_once_with(agent.config, "updated-key")

    def test_memory_map_guidance_is_rechecked_when_the_map_changes(self):
        def respond(number, messages):
            if number == 1:
                self.paths.meta_memory.write_text("navigation " * 1000)
            return call("read_file", path="mind/self.txt")

        agent, engine = self.agent(respond, meta_memory_guidance_tokens=1000)
        self.paths.meta_memory.write_text("A small memory map.")
        self.stop_after_tools(agent, 3)
        agent.run_forever(quiet=True)
        guidance = [r for r in read_jsonl(self.paths.life_loop_log) if r["kind"] == "guidance_notification"]
        self.assertEqual(len(guidance), 1)
        self.assertIn("SYSTEM GUIDANCE NOTIFICATION — META-MEMORY SIZE", json.dumps(engine.requests[1], ensure_ascii=False))

    def test_one_off_round_budget_returns_without_a_continuation_notice(self):
        agent, engine = self.agent(lambda *_: call("read_file", path="mind/self.txt"),
                                   max_life_loop_rounds=2)
        agent.run_once()
        self.assertEqual(len(engine.requests), 2)
        self.assertIn("KEEP_THIS_RESEARCH_CONTEXT", self.paths.working_context.read_text())
        self.assert_no_limit_notices(agent)

    def test_one_off_time_budget_waits_for_the_current_response(self):
        def respond(number, messages):
            self.clock += 1600
            return "<output>THE_LONG_RESPONSE_FINISHED</output>" + call("read_file", path="mind/self.txt")

        agent, engine = self.agent(respond)
        result = agent.run_once()
        self.assertEqual(len(engine.requests), 1)
        self.assertIn("THE_LONG_RESPONSE_FINISHED", result)
        self.assertIn("THE_LONG_RESPONSE_FINISHED", self.paths.working_context.read_text())
        self.assert_no_limit_notices(agent)


if __name__ == "__main__":
    unittest.main()
