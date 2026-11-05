"""Native tool calling: definitions, history replay, wire format and prompts."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from artificium.config import Config, ConfigStore, NATIVE_TOOL_ADAPTERS
from artificium.engine import Engine, EngineReply, make_engine
from artificium.engine_native import call_id, native_messages, tools_from_messages
from artificium.filesystem import Paths
from artificium.life_loop import parse_life_loop_output
from artificium.prompts import PromptPack
from artificium.records import Console
from artificium.runtime import Artificium
from artificium.tool_schemas import LEGACY_TOOLS, catalog_sections, native_tool_schemas
from artificium.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[2]
CATALOG = (ROOT / "app/prompts/tools/core_tools.md").read_text(encoding="utf-8")
TOOLS = [{"type": "function", "function": {"name": "run_shell", "description": "d",
                                           "parameters": {"type": "object", "properties": {}}}}]


def turn(calls: list[tuple[str, dict]], *, valid: bool = True, text: str = "Working.") -> dict:
    body = "\n".join([f"<think>{text}</think>"] + [
        "<tool_call>" + json.dumps({"tool": name, **arguments}) + "</tool_call>" for name, arguments in calls])
    return {"role": "assistant", "content": body, "_artificium": {
        "kind": "life_loop_output", "tool_names": [name for name, _ in calls], "tool_request_valid": valid}}


def result(name: str, tool_id: str, text: str = "[TOOL RESULT] ok") -> dict:
    return {"role": "user", "content": text, "_artificium": {"kind": "tool_result", "tool": name, "tool_id": tool_id}}


def system(tools: list | None = None) -> dict:
    message = {"role": "system", "content": "S"}
    if tools:
        message["_artificium"] = {"kind": "system", "tools": tools}
    return message


class ConfigTests(unittest.TestCase):
    def test_text_is_the_default_and_native_needs_a_chat_completions_adapter(self) -> None:
        self.assertEqual(Config(provider="custom", model="m", base_url="http://x/v1").tool_mode, "text")
        for provider in ("mistral", "openrouter", "vllm", "llamacpp"):
            self.assertEqual(Config(provider=provider, model="m", tool_mode="native").tool_mode, "native")
        for provider in ("anthropic", "openai", "gemini", "ollama"):
            with self.assertRaisesRegex(ValueError, "chat-completions adapter"):
                Config(provider=provider, model="m", tool_mode="native")
        with self.assertRaisesRegex(ValueError, "tool_mode must be"):
            Config(provider="mistral", model="m", tool_mode="both")
        self.assertIn("mistral", NATIVE_TOOL_ADAPTERS)


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schemas = native_tool_schemas(ToolRegistry, CATALOG)
        self.by_name = {item["function"]["name"]: item["function"] for item in self.schemas}

    def test_every_taught_tool_has_a_complete_definition(self) -> None:
        expected = set(ToolRegistry.CANONICAL_EXAMPLES) - LEGACY_TOOLS
        self.assertEqual(set(self.by_name), expected)
        sections = catalog_sections(CATALOG)
        for name, function in self.by_name.items():
            self.assertTrue(function["description"].strip(), name)
            self.assertIn(name, sections)
            self.assertNotIn('{"tool"', function["description"], name)
            self.assertEqual(function["parameters"]["type"], "object")
        json.dumps(self.schemas)

    def test_required_and_optional_arguments_follow_the_handler_signature(self) -> None:
        for name, function in self.by_name.items():
            parameters = list(inspect.signature(getattr(ToolRegistry, name)).parameters.values())[1:]
            self.assertEqual(set(function["parameters"]["properties"]), {p.name for p in parameters}, name)
            self.assertEqual(function["parameters"]["required"],
                             [p.name for p in parameters if p.default is inspect.Parameter.empty], name)

    def test_types_come_from_the_documented_examples(self) -> None:
        properties = self.by_name["write_file"]["parameters"]["properties"]
        self.assertEqual(properties["allow_shrink"], {"type": "boolean"})
        self.assertEqual(properties["path"], {"type": "string"})
        self.assertEqual(self.by_name["list_directory"]["parameters"]["properties"]["depth"], {"type": "integer"})
        self.assertEqual(self.by_name["load_images"]["parameters"]["properties"]["paths"]["type"], "array")


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        self.prompts = PromptPack(self.paths)

    def test_native_pack_swaps_the_protocol_and_drops_the_catalog(self) -> None:
        text, native = self.prompts.always(), self.prompts.always(native=True)
        self.assertIn("textual protocol", text.lower())
        self.assertNotIn("<tool_call>", native)
        self.assertIn("tool-calling interface", native)
        self.assertEqual(self.prompts.tool_catalog(native=True), "")
        self.assertIn("run_shell", self.prompts.tool_catalog())
        self.assertLess(len(native.split()), len(text.split()) + 50)
        self.assertEqual(sorted(self.prompts.fingerprints()),
                         sorted(set(self.prompts.fingerprints())))
        self.assertIn("always/tool_protocol_native.md", self.prompts.fingerprints())
        self.assertIn("events/tool_call_repair_native.md", self.prompts.fingerprints())

    def test_native_repair_event_does_not_teach_text_requests(self) -> None:
        rendered = self.prompts.event("tool_call_repair_native", repair_cases="[]", model_log_path="x")
        self.assertNotIn("<tool_call>", rendered)


class ReplayTests(unittest.TestCase):
    def test_a_paired_turn_becomes_structured_calls_and_tool_messages(self) -> None:
        messages = [system(TOOLS), {"role": "user", "content": "go"},
                    turn([("run_shell", {"command": "ls"}), ("read_file", {"path": "a"})]),
                    result("run_shell", "tool_1", "R1"), result("read_file", "tool_2", "R2"),
                    {"role": "user", "content": "state"}]
        converted = native_messages(messages)
        self.assertEqual([m["role"] for m in converted], ["system", "user", "assistant", "tool", "tool", "user"])
        calls = converted[2]["tool_calls"]
        self.assertEqual([c["function"]["name"] for c in calls], ["run_shell", "read_file"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls"})
        self.assertEqual([m["tool_call_id"] for m in converted[3:5]], [c["id"] for c in calls])
        self.assertEqual([m["content"] for m in converted[3:5]], ["R1", "R2"])
        self.assertIn("<think>Working.</think>", converted[2]["content"])
        for call in calls:
            self.assertRegex(call["id"], r"^[A-Za-z0-9]{9}$")
        self.assertEqual(len({c["id"] for c in calls}), 2)

    def test_ids_are_stable_and_distinct(self) -> None:
        self.assertEqual(call_id("tool_a"), call_id("tool_a"))
        self.assertNotEqual(call_id("tool_a"), call_id("tool_b"))

    def test_a_notice_between_the_turn_and_its_results_moves_after_them(self) -> None:
        notice = {"role": "user", "content": "images released", "_artificium": {"kind": "runtime_input"}}
        messages = [system(TOOLS), turn([("run_shell", {"command": "ls"})]), notice, result("run_shell", "t1")]
        converted = native_messages(messages)
        self.assertEqual([m["role"] for m in converted], ["system", "assistant", "tool", "user"])
        self.assertEqual(converted[3]["content"], "images released")

    def test_anything_not_exactly_paired_stays_text(self) -> None:
        first = turn([("run_shell", {"command": "ls"})])
        cases = {
            "missing result": [system(TOOLS), first, {"role": "user", "content": "x"}],
            "wrong tool": [system(TOOLS), first, result("read_file", "t1")],
            "next turn before result": [system(TOOLS), first, turn([("run_shell", {"command": "pwd"})]),
                                        result("run_shell", "t2")],
            "invalid request": [system(TOOLS), turn([("run_shell", {"command": "ls"})], valid=False),
                                result("run_shell", "t1")],
        }
        for label, messages in cases.items():
            converted = native_messages(messages)
            self.assertEqual(converted[1], messages[1], label)  # the first turn is replayed as it was recorded
            self.assertNotIn("tool_calls", converted[1], label)
            self.assertNotIn("tool", [m["role"] for m in converted[:2]], label)

    def test_requests_without_tool_definitions_are_untouched(self) -> None:
        messages = [system(), turn([("run_shell", {"command": "ls"})]), result("run_shell", "t1")]
        self.assertIs(native_messages(messages), messages)
        self.assertEqual(tools_from_messages(messages), [])


class WireTests(unittest.TestCase):
    PAIRED = [system(TOOLS), {"role": "user", "content": "go"},
              turn([("run_shell", {"command": "ls"})]), result("run_shell", "t1", "R"),
              {"role": "user", "content": "state"}]

    def prepare(self, provider: str, mode: str, messages: list[dict]) -> dict:
        config = Config(provider=provider, model="m", tool_mode=mode,
                        **({"base_url": "http://x/v1"} if provider == "custom" else {}))
        return make_engine(config, "key").prepare(messages).payload

    def test_chat_completions_adapters_send_tools_and_structured_history(self) -> None:
        for provider in ("custom", "mistral", "openrouter", "vllm", "llamacpp"):
            payload = self.prepare(provider, "native", self.PAIRED)
            self.assertEqual(payload["tools"], TOOLS, provider)
            roles = [m["role"] for m in payload["messages"]]
            self.assertEqual(roles, ["system", "user", "assistant", "tool", "user"], provider)
            self.assertIn("tool_calls", payload["messages"][2], provider)
            self.assertNotIn("_artificium", json.dumps(payload["messages"]), provider)

    def test_text_mode_sends_the_history_as_text(self) -> None:
        for provider in ("custom", "mistral", "openrouter", "vllm", "llamacpp"):
            payload = self.prepare(provider, "text", self.PAIRED)
            self.assertNotIn("tools", payload, provider)
            self.assertNotIn("tool_calls", json.dumps(payload["messages"]), provider)

    def test_a_request_without_definitions_never_declares_tools(self) -> None:
        payload = self.prepare("mistral", "native", [system(), {"role": "user", "content": "summarize"}])
        self.assertNotIn("tools", payload)

    def test_native_tool_calls_in_the_reply_become_runtime_readable_requests(self) -> None:
        raw = {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "a", "function": {"name": "run_shell", "arguments": '{"command": "ls"}'}}]},
            "finish_reason": "tool_calls"}]}
        engine = make_engine(Config(provider="mistral", model="m", tool_mode="native"), "key")
        with mock.patch.object(type(engine), "_post", return_value=raw):
            reply = engine.complete_prepared(engine.prepare(self.PAIRED))
        [tool] = parse_life_loop_output(reply.content).tools
        self.assertEqual((tool.name, tool.arguments, tool.parse_error), ("run_shell", {"command": "ls"}, None))


class ScriptedEngine(Engine):
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.requests: list[list[dict]] = []

    def complete(self, messages):
        self.requests.append(messages)
        return EngineReply(self.responses.pop(0) if self.responses else "<think>Waiting.</think>")


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        shutil.copytree(ROOT / "app/seed", self.paths.seed)
        self.paths.ensure_layout()
        environment = mock.patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def agent(self, engine: Engine, mode: str) -> Artificium:
        config = Config(provider="mistral", model="m", context_window_tokens=50000,
                        max_life_loop_rounds=1, tool_mode=mode)
        ConfigStore(self.paths).save(config)
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.initialization.finish("Self, meta-memory, tools, environment, and pending events inspected.")
        return agent

    def test_native_requests_carry_definitions_a_smaller_prompt_and_replayable_history(self) -> None:
        call = '<tool_call>{"tool":"list_directory","path":"mind"}</tool_call>'
        engine = ScriptedEngine(call, "<think>Done.</think>")
        agent = self.agent(engine, "native")
        agent.run_once()
        agent.run_once()
        first, second = engine.requests[0], engine.requests[-1]
        self.assertTrue(tools_from_messages(first))
        self.assertNotIn("<tool_call>", first[0]["content"])
        self.assertNotIn("### `read_file`", first[0]["content"])
        payload = make_engine(agent.config, "key").prepare(second).payload
        roles = [m["role"] for m in payload["messages"]]
        self.assertIn("tool", roles)
        paired = payload["messages"][roles.index("tool") - 1]
        self.assertEqual(paired["tool_calls"][0]["function"]["name"], "list_directory")
        self.assertEqual(payload["messages"][roles.index("tool")]["tool_call_id"], paired["tool_calls"][0]["id"])

    def test_text_mode_requests_are_unchanged(self) -> None:
        engine = ScriptedEngine("<think>Idle.</think>")
        self.agent(engine, "text").run_once()
        self.assertFalse(tools_from_messages(engine.requests[0]))
        self.assertIn("<tool_call>", engine.requests[0][0]["content"])
        self.assertIn("### `read_file`", engine.requests[0][0]["content"])


if __name__ == "__main__":
    unittest.main()
