"""consult and request_review with read-only tools: the oversight model checks
the evidence itself, in bounded rounds, and the agent sees what it read."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from test_revolution import FakeEngine, _seed_install
except ImportError:  # run as tests.test_oversight_tools
    from tests.test_revolution import FakeEngine, _seed_install

from artificium import display
from artificium.cli import main
from artificium.config import Config, ConfigStore, SecretsStore
from artificium.cost import CostLimitReached
from artificium.engine import make_engine
from artificium.filesystem import Paths, read_json
from artificium.life_loop import parse_life_loop_output
from artificium.records import Console
from artificium.runtime import Artificium
from artificium.setup import SetupOptions, SetupWizard
from artificium.tool_review import OVERSIGHT_READ_TOOLS

TEXT_SECTION = {"provider": "anthropic", "model": "judge-model", "context_window_tokens": 30_000}
NATIVE_SECTION = {"provider": "mistral", "model": "judge-model", "context_window_tokens": 30_000}
QUESTION = "Should the parser be rewritten or patched?"
CLAIM = "The cache is never stale, I checked."


def call(name: str, **arguments) -> str:
    return "<tool_call>\n" + json.dumps({"tool": name, **arguments}) + "\n</tool_call>"


class SessionEngine(FakeEngine):
    """A fake that, like the OpenCode Go engine, has a session ID to set."""

    session_id = None

    def __init__(self, responses=()):
        super().__init__(list(responses))
        self.sessions: list[str | None] = []

    def prepare(self, messages):
        self.sessions.append(self.session_id)
        return super().prepare(messages)


class ToolsCase(unittest.TestCase):
    section: dict | None = dict(TEXT_SECTION)
    config_changes: dict = {"decision_review": False}
    engine_class = FakeEngine

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))
        _seed_install(self.paths)
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        config = Config(provider="custom", model="worker-model", base_url="http://worker.invalid/v1",
                        context_window_tokens=20_000, context_reminder_tokens=2_000,
                        oversight_connection=dict(self.section) if self.section else None,
                        **self.config_changes)
        ConfigStore(self.paths).save(config)
        self.secrets = SecretsStore(self.paths)
        self.secrets.save_api_key("worker-key")
        if self.section:
            self.secrets.save_oversight_api_key(
                "judge-key", provider=self.section["provider"],
                base_url=config.oversight_config().base_url)
        self.worker = self.engine_class([])
        self.judge = self.engine_class([])
        patch = mock.patch("artificium.runtime.make_engine", return_value=self.judge)
        patch.start()
        self.addCleanup(patch.stop)
        self.agent = Artificium(self.paths, engine=self.worker, console=Console(quiet=True))
        self.agent.initialization.finish("The test instance is initialized and ready for its task.")
        self.tools = self.agent.tools
        self.notes = self.paths.space / "notes.txt"
        self.notes.parent.mkdir(parents=True, exist_ok=True)
        self.notes.write_text("the cache is flushed at step 7\n")

    def lifetime(self, kind: str) -> list[dict]:
        return [json.loads(line) for line in self.paths.lifetime_log.read_text().splitlines()
                if json.loads(line).get("kind") == kind]

    def life(self, kind: str) -> list[dict]:
        return [json.loads(line) for line in self.paths.life_loop_log.read_text().splitlines()
                if json.loads(line).get("kind") == kind]

    def set_rounds(self, rounds: int) -> None:
        self.tools.config = dataclasses.replace(self.tools.config, oversight_tool_rounds=rounds)

    @staticmethod
    def text_of(messages: list[dict]) -> str:
        return json.dumps(messages)


class TextProtocolTests(ToolsCase):
    def test_a_consult_reads_then_answers(self) -> None:
        self.judge.responses = [call("read_file", path=str(self.notes)),
                                "RECOMMENDATION: patch it, see notes.txt"]
        result = self.tools.consult(QUESTION)
        self.assertEqual(result["status"], "consulted")
        self.assertEqual(result["answer"], "RECOMMENDATION: patch it, see notes.txt")
        self.assertEqual(len(self.judge.requests), 2)
        self.assertEqual(len(self.worker.requests), 0)
        first, second = self.judge.requests
        self.assertIn("read_file, list_directory and search_memory", first[0]["content"])
        self.assertIn("<tool_call>", first[0]["content"])
        self.assertNotIn("tools", first[0].get("_artificium", {}))
        # The reviewer sees the result of its call, in the next request.
        self.assertIn("the cache is flushed at step 7", self.text_of(second[len(first):]))
        self.assertEqual(second[len(first)]["role"], "assistant")
        checked = result["reviewer_checked"]
        self.assertEqual((checked["tool_rounds_used"], checked["tool_rounds_allowed"], checked["requests"]),
                         (1, 4, 2))
        self.assertEqual(len(checked["request_ids"]), 2)
        self.assertEqual(result["request_id"], checked["request_ids"][-1])
        [entry] = checked["calls"]
        self.assertEqual((entry["tool"], entry["status"], entry["round"]), ("read_file", "ok", 1))
        self.assertIn(str(self.notes), entry["arguments"])

    def test_a_review_lists_and_searches_before_it_answers(self) -> None:
        memory = self.paths.memory / "topics" / "cache.md"
        memory.parent.mkdir(parents=True, exist_ok=True)
        memory.write_text("The cache invalidation rule is documented here.\n")
        self.judge.responses = [
            call("list_directory", path=str(self.notes.parent)) + "\n"
            + call("search_memory", query="cache invalidation"),
            "VERDICT: supported",
        ]
        result = self.tools.request_review(CLAIM)
        self.assertEqual(result["status"], "reviewed")
        self.assertEqual(result["review"], "VERDICT: supported")
        self.assertEqual([item["tool"] for item in result["reviewer_checked"]["calls"]],
                         ["list_directory", "search_memory"])
        self.assertEqual(result["reviewer_checked"]["tool_rounds_used"], 1)
        # Both results come back as separate messages, in call order.
        results = self.judge.requests[1][-2:]
        self.assertIn("notes.txt", results[0]["content"])
        self.assertIn("cache.md", results[1]["content"])

    def test_the_cap_forces_an_answer_without_tools(self) -> None:
        self.set_rounds(2)
        self.judge.responses = [call("read_file", path=str(self.notes))] * 3
        result = self.tools.consult(QUESTION)
        self.assertEqual(len(self.judge.requests), 3)
        last = self.judge.requests[-1]
        self.assertIn("Answer now", last[-1]["content"])
        self.assertIn("used your tool rounds", last[-1]["content"])
        self.assertNotIn("tools", last[0].get("_artificium", {}))
        # The model ignored the instruction and called again: the call is not an answer.
        self.assertEqual(result["status"], "consulted")
        self.assertNotIn("<tool_call>", result["answer"])
        self.assertEqual(result["reviewer_checked"]["tool_rounds_used"], 2)
        self.assertEqual(result["reviewer_checked"]["requests"], 3)

    def test_the_final_answer_may_follow_a_last_ignored_call(self) -> None:
        self.set_rounds(1)
        self.judge.responses = [call("read_file", path=str(self.notes)),
                                "RECOMMENDATION: patch it.\n" + call("read_file", path="x")]
        result = self.tools.consult(QUESTION)
        self.assertEqual(result["answer"], "RECOMMENDATION: patch it.")

    def test_zero_rounds_is_a_single_request_as_before(self) -> None:
        self.set_rounds(0)
        self.judge.responses = ["RECOMMENDATION: patch it."]
        result = self.tools.consult(QUESTION)
        self.assertEqual(result["answer"], "RECOMMENDATION: patch it.")
        self.assertEqual(len(self.judge.requests), 1)
        self.assertEqual(self.judge.requests[0][0]["content"],
                         self.agent.prompts.runtime("consult_system"))
        self.assertNotIn("reviewer_checked", result)
        self.assertEqual(self.lifetime("oversight_tool_executed"), [])

    def test_an_answer_in_the_first_request_needs_no_further_request(self) -> None:
        self.judge.responses = ["VERDICT: supported"]
        result = self.tools.request_review(CLAIM)
        self.assertEqual(len(self.judge.requests), 1)
        self.assertEqual(result["reviewer_checked"]["calls"], [])
        self.assertEqual(result["reviewer_checked"]["tool_rounds_used"], 0)

    def test_the_agents_own_evidence_and_the_prompt_forms_are_unchanged(self) -> None:
        self.judge.responses = ["VERDICT: supported"]
        self.tools.request_review(CLAIM, evidence_paths=[str(self.notes)])
        system, user = self.judge.requests[0]
        self.assertTrue(system["content"].startswith(self.agent.prompts.runtime("review_system")))
        self.assertIn("the cache is flushed at step 7", user["content"])

    def test_every_request_of_the_loop_is_logged_on_the_oversight_connection(self) -> None:
        self.judge.responses = [call("read_file", path=str(self.notes)), "VERDICT: supported"]
        self.tools.request_review(CLAIM)
        requests = self.lifetime("model_request")
        self.assertEqual([(r["connection"]) for r in requests], ["oversight", "oversight"])
        self.assertEqual(len(self.lifetime("model_response")), 2)
        done = self.lifetime("review_completed")[0]
        self.assertEqual((done["tool_rounds"], done["requests"]), (1, 2))


class ReadOnlyTests(ToolsCase):
    def refused(self, name: str, **arguments) -> str:
        self.judge.responses = [call(name, **arguments), "VERDICT: done"]
        result = self.tools.request_review(CLAIM)
        self.assertEqual(result["status"], "reviewed")
        [entry] = result["reviewer_checked"]["calls"]
        self.assertEqual(entry["status"], "error")
        return self.text_of(self.judge.requests[1][-1:])

    def test_a_tool_that_is_not_read_only_is_refused_and_not_run(self) -> None:
        target = self.paths.space / "created.txt"
        seen = self.refused("write_file", path=str(target), content="x")
        self.assertFalse(target.exists())
        self.assertIn("not available to you", seen)
        marker = self.paths.space / "ran.txt"
        self.refused("run_shell", command=f"touch {marker}")
        self.assertFalse(marker.exists())
        self.refused("send_interaction", interaction_id="room", content="hello there")
        self.refused("no_such_tool")
        for name in ("write_file", "run_shell", "save_memory", "sleep", "consult", "request_review"):
            self.assertNotIn(name, OVERSIGHT_READ_TOOLS)

    def test_the_settings_and_the_key_are_never_read(self) -> None:
        secrets = self.paths.secrets
        self.assertTrue(secrets.exists() and self.paths.config.exists())
        key_text = secrets.read_text()
        link = self.paths.space / "innocent.txt"
        link.symlink_to(secrets)
        elsewhere = self.paths.space / "nested"
        elsewhere.mkdir()
        (elsewhere / "config.json").write_text('{"token": "elsewhere-secret"}')
        (elsewhere / ".secrets.json").write_text('{"token": "elsewhere-secret"}')
        for path in (secrets, self.paths.config, link, elsewhere / "config.json",
                     elsewhere / ".secrets.json", "../.secrets.json"):
            seen = self.refused("read_file", path=str(path))
            self.assertIn("withheld", seen, path)
            self.assertNotIn("elsewhere-secret", seen)
            for line in key_text.splitlines():
                if len(line.strip()) > 12:
                    self.assertNotIn(line.strip(), seen)
            self.judge.requests.clear()
        # Everything else the file tools allow is still readable.
        self.judge.responses = [call("read_file", path=str(self.notes)), "VERDICT: ok"]
        result = self.tools.request_review(CLAIM)
        self.assertEqual(result["reviewer_checked"]["calls"][0]["status"], "ok")

    def test_search_results_never_include_withheld_files(self) -> None:
        topics = self.paths.memory / "topics"
        topics.mkdir(parents=True, exist_ok=True)
        (topics / "config.json").write_text("zebra stripes zebra")
        (topics / "zebra.md").write_text("zebra notes")
        self.judge.responses = [call("search_memory", query="zebra"), "VERDICT: ok"]
        self.tools.request_review(CLAIM)
        seen = self.text_of(self.judge.requests[1][-1:])
        self.assertIn("zebra.md", seen)
        self.assertNotIn("config.json", seen)

    def test_a_malformed_call_gets_an_error_result_not_a_crash(self) -> None:
        self.judge.responses = [
            call("read_file", bogus=1) + "\n" + call("read_file") + "\n"
            + "<tool_call>\n{not json\n</tool_call>\n" + call("read_file", path=str(self.notes)),
            "VERDICT: ok",
        ]
        result = self.tools.request_review(CLAIM)
        statuses = [item["status"] for item in result["reviewer_checked"]["calls"]]
        self.assertEqual(statuses[:2], ["error", "error"])
        self.assertEqual(statuses[-1], "ok")
        self.assertEqual(result["status"], "reviewed")
        self.assertEqual(result["review"], "VERDICT: ok")

    def test_a_missing_file_is_a_tool_error_the_reviewer_sees(self) -> None:
        seen = self.refused("read_file", path=str(self.paths.space / "absent.txt"))
        self.assertIn("does not exist", seen)

    def test_the_reviewers_reads_leave_the_agents_partial_read_markers_alone(self) -> None:
        big = self.paths.space / "big.txt"
        big.write_text("line of text\n" * 20_000)
        before = read_json(self.paths.partial_reads_state, None)
        self.judge.responses = [call("read_file", path=str(big), max_characters=500), "VERDICT: ok"]
        result = self.tools.request_review(CLAIM)
        self.assertEqual(result["reviewer_checked"]["calls"][0]["status"], "ok")
        self.assertEqual(read_json(self.paths.partial_reads_state, None), before)


class BoundTests(ToolsCase):
    def test_each_result_is_bounded_by_the_tool_output_limit(self) -> None:
        big = self.paths.space / "big.txt"
        big.write_text("line of text\n" * 20_000)
        listing = self.paths.space / "many"
        listing.mkdir()
        for index in range(300):
            (listing / f"file-{index:04d}-with-a-long-name-to-fill-the-listing.txt").write_text("x")
        cap = self.tools.config.max_tool_output_chars
        self.judge.responses = [call("read_file", path=str(big), max_characters=10_000_000)
                                + call("list_directory", path=str(listing)), "VERDICT: ok"]
        self.tools.config = dataclasses.replace(self.tools.config, max_tool_output_chars=3_000)
        result = self.tools.request_review(CLAIM)
        self.assertEqual(len(result["reviewer_checked"]["calls"]), 2)
        for message in self.judge.requests[1][-2:]:
            self.assertLessEqual(len(message["content"]), 3_000, cap)
        self.assertIn("characters omitted", self.judge.requests[1][-1]["content"])

    def test_results_that_would_overflow_the_window_force_the_answer(self) -> None:
        window = self.agent._oversight_budget_config()
        limit = int(window.working_memory_limit * window.chars_per_token * 0.8)
        part = self.paths.space / "part.txt"
        part.write_text("0123456789" * 5_000)  # 50 000 characters per read
        reads = limit // 50_000 + 2
        self.set_rounds(reads + 2)
        self.judge.responses = [call("read_file", path=str(part))] * (reads + 2)
        self.judge.responses.append("never reached")
        result = self.tools.consult(QUESTION)
        statuses = [item["status"] for item in result["reviewer_checked"]["calls"]]
        self.assertIn("not_shown (window full)", statuses[-1])
        self.assertTrue(all(status == "ok" for status in statuses[:-1]))
        self.assertLess(len(statuses), reads + 2)
        self.assertEqual(self.lifetime("oversight_tool_executed")[-1]["status"], "not_shown (window full)")
        last = self.judge.requests[-1]
        self.assertIn("size limit", last[-1]["content"])
        self.assertIn("Answer now", last[-1]["content"])
        # What was left out is not sent, and nothing was sent past the window.
        self.assertLessEqual(sum(len(str(m.get("content"))) for m in last[:-1]), limit)


class NativeProtocolTests(ToolsCase):
    section = dict(NATIVE_SECTION)

    def wire(self, messages: list[dict]) -> dict:
        config = ConfigStore(self.paths).load().oversight_config()
        return make_engine(config, "judge-key").prepare(messages).payload

    def test_a_native_connection_gets_definitions_and_structured_rounds(self) -> None:
        self.judge.responses = [call("read_file", path=str(self.notes)), "VERDICT: supported"]
        result = self.tools.request_review(CLAIM)
        self.assertEqual(result["review"], "VERDICT: supported")
        first, second = self.judge.requests
        names = [item["function"]["name"] for item in first[0]["_artificium"]["tools"]]
        self.assertEqual(sorted(names), sorted(OVERSIGHT_READ_TOOLS))
        self.assertNotIn("<tool_call>", first[0]["content"])
        payload = self.wire(first)
        self.assertEqual([item["function"]["name"] for item in payload["tools"]], names)
        replay = self.wire(second)
        self.assertEqual([m["role"] for m in replay["messages"]], ["system", "user", "assistant", "tool"])
        [native_call] = replay["messages"][2]["tool_calls"]
        self.assertEqual(native_call["function"]["name"], "read_file")
        self.assertEqual(json.loads(native_call["function"]["arguments"])["path"], str(self.notes))
        self.assertEqual(replay["messages"][3]["tool_call_id"], native_call["id"])
        self.assertIn("the cache is flushed at step 7", replay["messages"][3]["content"])
        self.assertEqual(replay["tools"], payload["tools"])
        self.assertNotIn("_artificium", json.dumps(replay))

    def test_the_final_native_request_carries_no_tools_and_asks_for_the_answer(self) -> None:
        self.set_rounds(1)
        self.judge.responses = [call("read_file", path=str(self.notes)), "RECOMMENDATION: patch it."]
        result = self.tools.consult(QUESTION)
        self.assertEqual(result["answer"], "RECOMMENDATION: patch it.")
        final = self.wire(self.judge.requests[-1])
        self.assertNotIn("tools", final)
        self.assertNotIn("tool", [m["role"] for m in final["messages"]])
        self.assertNotIn("tool_calls", json.dumps(final["messages"]))
        self.assertIn("Answer now", final["messages"][-1]["content"])

    def test_a_native_adapter_with_the_text_tool_mode_still_gets_native_tools(self) -> None:
        # The oversight connection never has a tool mode of its own.
        self.assertEqual(self.agent._oversight_budget_config().tool_mode, "text")
        self.assertTrue(self.agent._oversight_native_tools())
        payload = self.wire([{"role": "system", "content": "S", "_artificium": {
            "kind": "system", "tools": [{"type": "function", "function": {"name": "x"}}]}},
            {"role": "user", "content": "q"}])
        self.assertNotIn("tools", payload, "definitions alone never switch a text-mode connection")

    def test_every_native_adapter_sends_the_definitions_a_request_asks_for(self) -> None:
        from artificium.config import NATIVE_TOOL_ADAPTERS
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
        messages = [{"role": "system", "content": "S",
                     "_artificium": {"kind": "system", "native": True, "tools": tools}},
                    {"role": "user", "content": "q"}]
        seen = set()
        for provider in ("custom", "mistral", "openrouter", "vllm", "llamacpp", "opencode-go"):
            config = Config(provider=provider, model="m",
                            **({"base_url": "http://x/v1"} if provider == "custom" else {}))
            seen.add(config.adapter)
            self.assertEqual(make_engine(config, "k").prepare(messages).payload["tools"], tools, provider)
        self.assertEqual(seen, set(NATIVE_TOOL_ADAPTERS))

    def test_a_text_adapter_uses_the_text_protocol(self) -> None:
        with mock.patch.object(self.agent, "_oversight_native_tools", return_value=False):
            self.tools.oversight_native_tools = self.agent._oversight_native_tools
            self.judge.responses = ["VERDICT: ok"]
            self.tools.request_review(CLAIM)
        self.assertIn("<tool_call>", self.judge.requests[0][0]["content"])
        self.assertNotIn("tools", self.judge.requests[0][0].get("_artificium", {}))


class WorkerFallbackTests(ToolsCase):
    """No oversight section: the worker serves oversight, and a native worker
    adapter gets native tools for these requests too."""

    section = None

    def test_the_worker_serves_the_loop(self) -> None:
        self.worker.responses = [call("read_file", path=str(self.notes)), "VERDICT: supported"]
        result = self.tools.request_review(CLAIM)
        self.assertEqual(result["review"], "VERDICT: supported")
        self.assertEqual(len(self.worker.requests), 2)
        self.assertEqual([r["connection"] for r in self.lifetime("model_request")], ["worker", "worker"])


class CostAndFailureTests(ToolsCase):
    def test_the_cost_limit_reached_mid_loop_stops_the_loop_with_the_error(self) -> None:
        self.judge.responses = [call("read_file", path=str(self.notes)), "never sent"]
        limited = CostLimitReached("daily cost limit reached (10 of 10 USD); no request was sent.")
        gate = mock.Mock(side_effect=[None, limited])
        with mock.patch.object(self.agent, "_cost_limit_gate", gate):
            result = self.tools.consult(QUESTION)
        self.assertEqual(len(self.judge.requests), 1)
        self.assertEqual(result["status"], "error")
        self.assertIn("daily cost limit", result["summary"])
        self.assertIn("reviewer_checked", result)
        self.assertEqual(result["reviewer_checked"]["calls"][0]["tool"], "read_file")
        self.assertEqual(self.lifetime("oversight_connection_failed"), [])

    def test_a_cost_limit_before_the_first_round_has_no_reads_to_list(self) -> None:
        limited = CostLimitReached("daily cost limit reached (10 of 10 USD); no request was sent.")
        with mock.patch.object(self.agent, "_cost_limit_gate", side_effect=limited):
            result = self.tools.request_review(CLAIM)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("reviewer_checked", result)

    def test_a_connection_failure_mid_loop_is_an_error_that_lists_the_reads(self) -> None:
        replies = [call("read_file", path=str(self.notes))]

        def complete(messages):
            self.judge.requests.append(messages)
            if replies:
                from artificium.engine import EngineReply
                return EngineReply(replies.pop(0), usage={"input_tokens": 1, "output_tokens": 1})
            from artificium.engine import EngineError
            raise EngineError("HTTP 500: down", status=500, kind="server")

        self.judge.complete = complete
        result = self.tools.request_review(CLAIM)
        self.assertEqual(result["status"], "error")
        self.assertIn("oversight", result["summary"])
        self.assertEqual(len(result["reviewer_checked"]["calls"]), 1)
        self.assertEqual(self.lifetime("oversight_connection_failed")[0]["connection"], "oversight")


class VisibilityTests(ToolsCase):
    def test_each_call_is_recorded_and_shown(self) -> None:
        self.judge.responses = [call("read_file", path=str(self.notes)) + call("write_file", path="x", content="y"),
                                "VERDICT: ok"]
        self.tools.request_review(CLAIM)
        records = self.lifetime("oversight_tool_executed")
        self.assertEqual([(r["tool"], r["status"], r["during"], r["round"]) for r in records],
                         [("read_file", "ok", "request_review", 1),
                          ("write_file", "error", "request_review", 1)])
        self.assertIn(str(self.notes), records[0]["arguments"])
        self.assertEqual(records[0]["request_id"], records[1]["request_id"])
        trace = self.life("oversight_tool_executed")
        self.assertEqual(len(trace), 2)
        lines = display.render_oversight_tool_executed(trace[0], display.RenderOptions())
        self.assertEqual(len(lines), 1)
        self.assertIn("read_file", lines[0])
        self.assertIn("ok", lines[0])
        self.assertIn("request_review", lines[0])

    def test_the_kind_is_known_to_watch(self) -> None:
        from artificium import events
        self.assertTrue(events.is_known_kind("oversight_tool_executed"))


class SessionTests(ToolsCase):
    engine_class = SessionEngine

    def test_all_rounds_of_one_review_share_one_session_and_the_next_review_has_its_own(self) -> None:
        self.judge.responses = [call("read_file", path=str(self.notes)),
                                call("read_file", path=str(self.notes)),
                                "VERDICT: ok", "RECOMMENDATION: keep going"]
        self.tools.request_review(CLAIM)
        self.assertEqual(len(self.judge.sessions), 3)
        self.assertEqual(len(set(self.judge.sessions)), 1)
        self.tools.consult(QUESTION)
        self.assertEqual(len(self.judge.sessions), 4)
        self.assertNotEqual(self.judge.sessions[-1], self.judge.sessions[0])
        self.assertIsNone(self.agent._review_session)
        self.assertIsNone(self.agent._fresh_session)

    def test_a_failure_inside_the_loop_ends_the_conversation(self) -> None:
        limited = CostLimitReached("daily cost limit reached (10 of 10 USD); no request was sent.")
        self.judge.responses = [call("read_file", path=str(self.notes))]
        with mock.patch.object(self.agent, "_cost_limit_gate", side_effect=[None, limited]):
            self.tools.consult(QUESTION)
        self.assertIsNone(self.agent._review_session)


class RegistryWithoutRuntimeTests(ToolsCase):
    def test_a_registry_without_the_runtime_hooks_uses_the_text_protocol_and_plain_sessions(self) -> None:
        sent = []

        def consultant(messages):
            sent.append(messages)
            return "request_1", ("VERDICT: ok" if len(sent) > 1 else call("read_file", path=str(self.notes)))

        for name in ("oversight_native_tools", "oversight_conversation"):
            delattr(self.tools, name)
        self.tools.consultant = consultant
        result = self.tools.consult(QUESTION)
        self.assertEqual(result["status"], "consulted")
        self.assertEqual(len(sent), 2)
        self.assertIn("<tool_call>", sent[0][0]["content"])


class ConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))
        _seed_install(self.paths)
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = Config(provider="custom", model="m", base_url="http://worker.invalid/v1")
        ConfigStore(self.paths).save(self.config)

    def test_the_default_is_four_and_every_save_writes_it(self) -> None:
        self.assertEqual(self.config.oversight_tool_rounds, 4)
        self.assertEqual(read_json(self.paths.config)["harness"]["oversight_tool_rounds"], 4)

    def test_a_file_without_the_key_means_the_current_default(self) -> None:
        saved = read_json(self.paths.config)
        del saved["harness"]["oversight_tool_rounds"]
        self.paths.config.write_text(json.dumps(saved))
        self.assertEqual(ConfigStore(self.paths).load().oversight_tool_rounds, 4)

    def test_only_whole_numbers_from_zero_to_twenty_are_valid(self) -> None:
        for value in (-1, 21, 1.5, "3", True, None):
            with self.assertRaises(ValueError, msg=repr(value)):
                dataclasses.replace(self.config, oversight_tool_rounds=value)
        for value in (0, 1, 20):
            self.assertEqual(dataclasses.replace(self.config, oversight_tool_rounds=value)
                             .oversight_tool_rounds, value)

    def test_configure_harness_sets_it_and_it_does_not_belong_to_the_other_scopes(self) -> None:
        updated = SetupWizard(self.paths).reconfigure(
            SetupOptions(scope="harness", oversight_tool_rounds=0))
        self.assertEqual(updated.oversight_tool_rounds, 0)
        self.assertEqual(ConfigStore(self.paths).load().oversight_tool_rounds, 0)
        for scope in ("model", "oversight"):
            with self.assertRaises(ValueError):
                SetupWizard(self.paths).reconfigure(
                    SetupOptions(scope=scope, oversight_tool_rounds=2))

    def test_the_command_line_flag(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--root", str(self.paths.install), "configure", "harness",
                         "--oversight-tool-rounds", "7"])
        self.assertEqual(code, 0, out.getvalue())
        self.assertEqual(ConfigStore(self.paths).load().oversight_tool_rounds, 7)


if __name__ == "__main__":
    unittest.main()
