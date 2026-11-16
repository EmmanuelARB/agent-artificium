"""The oversight connection: config section, key, routing, consult, setup, CLI."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from test_revolution import FakeEngine, _seed_install
    from test_connection_flow import TestServer
except ImportError:  # run as tests.test_oversight_connection
    from tests.test_revolution import FakeEngine, _seed_install
    from tests.test_connection_flow import TestServer

from artificium.cli import main
from artificium.config import (Config, ConfigStore, MODEL_FIELDS, OVERSIGHT_FIELDS,
                               SecretsStore)
from artificium.engine import EngineError
from artificium.filesystem import Paths, read_json
from artificium.operator import format_status, status_snapshot
from artificium.prompts import PromptPack
from artificium.records import Console
from artificium.runtime import Artificium
from artificium.setup import SetupOptions, SetupWizard
from artificium.tool_schemas import native_tool_schemas
from artificium.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[2]

SECTION = {"provider": "anthropic", "model": "judge-model", "context_window_tokens": 30_000}


def _worker(**changes) -> Config:
    return Config(provider="custom", model="worker-model", base_url="http://worker.invalid/v1",
                  context_window_tokens=20_000, context_reminder_tokens=2_000, **changes)


def _later(hours: float) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)


class ConfigSectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))

    def test_a_configuration_without_the_section_has_no_oversight_connection(self) -> None:
        config = _worker()
        self.assertIsNone(config.oversight_config())
        ConfigStore(self.paths).save(config)
        saved = read_json(self.paths.config)
        self.assertEqual(set(saved), {"schema_version", "harness", "model"})
        self.assertNotIn("oversight_connection", saved["harness"])
        self.assertEqual(ConfigStore(self.paths).load(), config)

    def test_the_section_round_trips_with_every_field_written(self) -> None:
        config = dataclasses.replace(_worker(), oversight_connection=dict(SECTION))
        ConfigStore(self.paths).save(config)
        saved = read_json(self.paths.config)
        self.assertEqual(set(saved["oversight"]), OVERSIGHT_FIELDS)
        self.assertNotIn("tool_mode", saved["oversight"])
        self.assertEqual(saved["oversight"]["base_url"], "https://api.anthropic.com/v1")
        self.assertEqual(saved["oversight"]["adapter"], "anthropic")
        # Keys the section left out mean the current default.
        self.assertEqual(saved["oversight"]["stall_timeout_seconds"], 180.0)
        loaded = ConfigStore(self.paths).load()
        self.assertEqual(loaded, config)
        oversight = loaded.oversight_config()
        self.assertEqual((oversight.provider, oversight.model, oversight.context_window_tokens),
                         ("anthropic", "judge-model", 30_000))
        # The worker's own connection is untouched.
        self.assertEqual((loaded.provider, loaded.model, loaded.context_window_tokens),
                         ("custom", "worker-model", 20_000))

    def test_the_oversight_config_shares_harness_settings_and_never_uses_tools(self) -> None:
        worker = _worker(tool_mode="text", chars_per_token=3.2, working_memory_tokens=19_000,
                         mandatory_offload=True, oversight_model="ignored")
        config = dataclasses.replace(worker, oversight_connection={
            **SECTION, "context_window_tokens": 8_000})
        oversight = config.oversight_config()
        self.assertEqual(oversight.chars_per_token, 3.2)
        self.assertEqual(oversight.tool_mode, "text")
        self.assertIsNone(oversight.oversight_connection)
        # A worker target larger than the oversight window must not make the
        # oversight config invalid; the budget is the oversight window's.
        self.assertEqual(oversight.working_memory_limit, 8_000)

    def test_a_field_in_the_wrong_section_is_a_load_error(self) -> None:
        grouped = dataclasses.replace(_worker(), oversight_connection=dict(SECTION)).grouped_dict()
        for place, name, value in (("oversight", "tool_mode", "text"),
                                   ("oversight", "decision_default_hours", 3),
                                   ("harness", "oversight_connection", dict(SECTION)),
                                   ("model", "oversight", {})):
            broken = json.loads(json.dumps(grouped))
            broken[place][name] = value
            with self.assertRaises(ValueError, msg=f"{place}.{name}"):
                Config.from_dict(broken)
        broken = json.loads(json.dumps(grouped))
        broken["oversight"] = "judge-model"
        with self.assertRaises(ValueError):
            Config.from_dict(broken)
        # And an incomplete connection is refused rather than guessed at.
        broken["oversight"] = {"provider": "anthropic"}
        with self.assertRaises(ValueError):
            Config.from_dict(broken)

    def test_a_section_with_missing_keys_loads_with_current_defaults(self) -> None:
        grouped = _worker().grouped_dict()
        grouped["oversight"] = {"provider": "mistral", "model": "m"}
        loaded = Config.from_dict(grouped)
        oversight = loaded.oversight_config()
        self.assertEqual((oversight.adapter, oversight.base_url), ("mistral", "https://api.mistral.ai/v1"))
        self.assertIsNone(oversight.request_timeout_seconds)
        # Loading does not rewrite the file; saving writes the whole section.
        self.assertEqual(set(loaded.grouped_dict()["oversight"]), OVERSIGHT_FIELDS)

    def test_the_legacy_oversight_model_setting_still_loads_and_saves(self) -> None:
        config = _worker(oversight_model="judge")
        ConfigStore(self.paths).save(config)
        loaded = ConfigStore(self.paths).load()
        self.assertEqual(loaded.oversight_model, "judge")
        self.assertIsNone(loaded.oversight_config())
        self.assertIn("oversight_model", read_json(self.paths.config)["harness"])

    def test_the_sections_fields_are_the_model_fields_without_tool_mode(self) -> None:
        self.assertEqual(OVERSIGHT_FIELDS, MODEL_FIELDS - {"tool_mode"})


class SecretsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))
        self.paths.install.mkdir(parents=True, exist_ok=True)
        self.secrets = SecretsStore(self.paths)
        self.worker = _worker()
        self.oversight = dataclasses.replace(
            self.worker, oversight_connection=dict(SECTION)).oversight_config()
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_both_keys_are_stored_apart_and_survive_each_other(self) -> None:
        self.secrets.save_api_key("worker-key", provider="custom", base_url=self.worker.base_url)
        self.secrets.save_oversight_api_key("judge-key", provider="anthropic",
                                            base_url=self.oversight.base_url)
        self.assertEqual(self.secrets.resolve_api_key(self.worker), "worker-key")
        self.assertEqual(self.secrets.resolve_oversight_key(self.oversight, self.worker), "judge-key")
        # Replacing the worker's key keeps the oversight key, and the reverse.
        self.secrets.save_api_key("worker-key-2", provider="custom", base_url=self.worker.base_url)
        self.assertEqual(self.secrets.resolve_oversight_key(self.oversight, self.worker), "judge-key")
        self.secrets.save_oversight_api_key("judge-key-2", provider="anthropic",
                                            base_url=self.oversight.base_url)
        self.assertEqual(self.secrets.resolve_api_key(self.worker), "worker-key-2")
        self.assertEqual(oct(self.paths.secrets.stat().st_mode & 0o777), "0o600")

    def test_a_file_that_predates_the_oversight_key_keeps_working(self) -> None:
        self.paths.secrets.write_text(json.dumps({"api_key": "old", "preferred": True}))
        self.assertEqual(self.secrets.resolve_api_key(self.worker), "old")
        self.assertIsNone(self.secrets.resolve_oversight_key(self.oversight, self.worker))

    def test_the_workers_key_is_never_sent_to_another_provider(self) -> None:
        self.secrets.save_api_key("worker-key")
        self.assertIsNone(self.secrets.resolve_oversight_key(self.oversight, self.worker))
        with mock.patch.dict(os.environ, {"ARTIFICIUM_API_KEY": "generic"}):
            self.assertIsNone(self.secrets.resolve_oversight_key(self.oversight, self.worker))

    def test_environment_variables_for_the_oversight_connection(self) -> None:
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "provider-key"}):
            self.assertEqual(self.secrets.resolve_oversight_key(self.oversight, self.worker), "provider-key")
            with mock.patch.dict(os.environ, {"ARTIFICIUM_OVERSIGHT_API_KEY": "specific"}):
                self.assertEqual(self.secrets.resolve_oversight_key(self.oversight, self.worker), "specific")
            # An explicitly entered key outranks the environment.
            self.secrets.save_oversight_api_key("typed", provider="anthropic",
                                                base_url=self.oversight.base_url)
            self.assertEqual(self.secrets.resolve_oversight_key(self.oversight, self.worker), "typed")

    def test_the_very_server_the_worker_uses_shares_its_key(self) -> None:
        same = dataclasses.replace(self.worker, oversight_connection={
            "provider": "custom", "model": "bigger", "base_url": self.worker.base_url}).oversight_config()
        self.secrets.save_api_key("worker-key", provider="custom", base_url=self.worker.base_url)
        self.assertEqual(self.secrets.resolve_oversight_key(same, self.worker), "worker-key")

    def test_removing_the_oversight_key_leaves_the_workers(self) -> None:
        self.secrets.save_api_key("worker-key")
        self.secrets.save_oversight_api_key("judge-key")
        self.assertTrue(self.secrets.remove_oversight_api_key())
        self.assertFalse(self.secrets.remove_oversight_api_key())
        self.assertEqual(self.secrets.resolve_api_key(self.worker), "worker-key")
        self.assertIsNone(self.secrets.resolve_oversight_key(self.oversight, self.worker))


class RuntimeCase(unittest.TestCase):
    """An agent whose engines are fakes: one per connection, built on demand."""

    section: dict | None = dict(SECTION)
    config_changes: dict = {}

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = Paths(self.root)
        _seed_install(self.paths)
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        config = _worker(oversight_connection=dict(self.section) if self.section else None,
                         **self.config_changes)
        ConfigStore(self.paths).save(config)
        self.secrets = SecretsStore(self.paths)
        self.secrets.save_api_key("worker-key")
        if self.section:
            self.secrets.save_oversight_api_key("judge-key", provider="anthropic",
                                                base_url="https://api.anthropic.com/v1")
        self.worker = FakeEngine([])
        self.judge = FakeEngine([])
        self.builds: list[tuple[str, str | None]] = []

        def build(cfg, key):
            self.builds.append((cfg.model, key))
            return self.judge

        patch = mock.patch("artificium.runtime.make_engine", side_effect=build)
        patch.start()
        self.addCleanup(patch.stop)
        self.agent = Artificium(self.paths, engine=self.worker, console=Console(quiet=True))

    def lifetime(self, kind: str) -> list[dict]:
        return [json.loads(line) for line in self.paths.lifetime_log.read_text().splitlines()
                if json.loads(line).get("kind") == kind]


class RoutingTests(RuntimeCase):
    # A decision's own review is routed in DecisionReviewRuntimeTests.
    config_changes = {"decision_review": False}

    def assert_served_by_oversight(self, action) -> None:
        before = (len(self.worker.requests), len(self.judge.requests))
        action()
        self.assertEqual(len(self.worker.requests), before[0], "the worker was used")
        self.assertEqual(len(self.judge.requests), before[1] + 1, "oversight was not used")

    def test_the_evaluation_goes_to_the_oversight_connection(self) -> None:
        self.judge.responses = ["VERDICT: on track"]
        self.assert_served_by_oversight(lambda: self.agent.oversight.evaluate(_later(0)))

    def test_the_decision_default_check_goes_to_the_oversight_connection(self) -> None:
        ArtificiumClient = __import__("artificium.interactions", fromlist=["x"]).ArtificiumClient
        ArtificiumClient(self.root).send("room", sender="entity_1", content="Please build it.")
        self.agent.tools.send_interaction("room", "Which one?", decision=True, default="proceed")
        self.judge.responses = ["VERDICT: apply\nREASON: reversible"]
        self.assert_served_by_oversight(lambda: self.agent.oversight.process_defaults(_later(25)))

    def test_request_review_goes_to_the_oversight_connection(self) -> None:
        self.judge.responses = ["VERDICT: doubtful"]
        self.assert_served_by_oversight(
            lambda: self.agent.tools.request_review("The cache is never stale, I checked."))

    def test_consult_goes_to_the_oversight_connection(self) -> None:
        self.judge.responses = ["RECOMMENDATION: keep going"]
        self.assert_served_by_oversight(
            lambda: self.agent.tools.consult("Should the parser be rewritten or patched?"))

    def test_the_worker_connection_is_untouched_by_an_oversight_request(self) -> None:
        engine, config = self.agent.engine, self.agent.config
        self.judge.responses = ["VERDICT: doubtful"]
        self.agent.tools.request_review("The cache is never stale, I checked.")
        self.assertIs(self.agent.engine, engine)
        self.assertIs(self.agent.config, config)
        self.worker.responses = ["<think>fine</think>"]
        self.agent._complete([{"role": "user", "content": "work"}], wake_reason="test")
        self.assertEqual(len(self.worker.requests), 1)

    def test_the_engine_is_built_once_with_its_own_key_and_rebuilt_when_the_key_changes(self) -> None:
        self.judge.responses = ["VERDICT: a", "VERDICT: b", "VERDICT: c"]
        self.agent._oversight_complete([{"role": "user", "content": "one"}])
        self.agent._oversight_complete([{"role": "user", "content": "two"}])
        self.assertEqual(self.builds, [("judge-model", "judge-key")])
        self.secrets.save_oversight_api_key("judge-key-2", provider="anthropic",
                                            base_url="https://api.anthropic.com/v1")
        self.agent._oversight_complete([{"role": "user", "content": "three"}])
        self.assertEqual(self.builds[-1], ("judge-model", "judge-key-2"))
        self.assertEqual(len(self.builds), 2)

    def test_token_budgets_use_the_oversight_connections_own_window(self) -> None:
        seen = []
        original = self.agent._measure

        def spy(messages, prepared=None):
            seen.append(self.agent.config.context_window_tokens)
            return original(messages, prepared)

        self.judge.responses = ["VERDICT: ok"]
        with mock.patch.object(self.agent, "_measure", spy):
            self.agent._oversight_complete([{"role": "user", "content": "x"}])
        self.assertEqual(seen, [30_000])
        # Too large for the oversight window, though it would fit the worker's.
        with self.assertRaises(EngineError):
            self.agent._oversight_complete([{"role": "user", "content": "word " * 40_000}])

    def test_every_request_is_logged_with_the_connection_that_served_it(self) -> None:
        self.judge.responses = ["VERDICT: ok"]
        self.worker.responses = ["<think>fine</think>"]
        self.agent._oversight_complete([{"role": "user", "content": "judge this"}])
        self.agent._complete([{"role": "user", "content": "work"}], wake_reason="test")
        requests = self.lifetime("model_request")
        responses = self.lifetime("model_response")
        self.assertEqual([r["connection"] for r in requests], ["oversight", "worker"])
        self.assertEqual([r["connection"] for r in responses], ["oversight", "worker"])
        life = [json.loads(line) for line in self.paths.life_loop_log.read_text().splitlines()]
        self.assertEqual([r["connection"] for r in life if r["kind"] == "engine_response"],
                         ["oversight", "worker"])
        logs = {request["request_id"]: read_json(self.paths.model_log / f"{request['request_id']}.json")
                for request in requests}
        self.assertEqual([logs[r["request_id"]]["request_parameters"]["connection"] for r in requests],
                         ["oversight", "worker"])

    def test_oversight_requests_do_not_overwrite_the_workers_calibration(self) -> None:
        self.worker.responses = ["<think>fine</think>"]
        self.agent._complete([{"role": "user", "content": "work"}], wake_reason="test")
        before = (self.paths.runtime / "token-calibration.json")
        stored = before.read_bytes() if before.exists() else None
        self.judge.responses = ["VERDICT: ok"]
        self.agent._oversight_complete([{"role": "user", "content": "judge this"}])
        self.assertEqual(before.read_bytes() if before.exists() else None, stored)


class LegacyOversightModelTests(RuntimeCase):
    section = None

    def test_without_a_section_requests_go_to_the_worker_as_before(self) -> None:
        self.worker.responses = ["VERDICT: ok"]
        self.agent._oversight_complete([{"role": "user", "content": "q"}])
        self.assertEqual((len(self.worker.requests), len(self.judge.requests)), (1, 0))
        self.assertEqual(self.lifetime("model_request")[-1]["connection"], "worker")

    def test_oversight_model_names_another_model_on_the_workers_endpoint_and_key(self) -> None:
        self.agent.config = dataclasses.replace(self.agent.config, oversight_model="judge")
        self.judge.responses = ["VERDICT: ok"]
        self.agent._oversight_complete([{"role": "user", "content": "q"}])
        self.assertEqual(self.builds, [("judge", "worker-key")])
        self.assertEqual(len(self.worker.requests), 0)
        self.assertEqual(self.agent.config.model, "worker-model")

    def test_the_section_wins_over_oversight_model(self) -> None:
        config = dataclasses.replace(self.agent.config, oversight_model="legacy",
                                     oversight_connection=dict(SECTION))
        self.agent.config = config
        self.judge.responses = ["VERDICT: ok"]
        self.agent._oversight_complete([{"role": "user", "content": "q"}])
        self.assertEqual(self.builds[0][0], "judge-model")


class OversightFailureTests(RuntimeCase):
    def failing(self, error: Exception | None = None) -> None:
        def complete(messages):
            raise error or EngineError("HTTP 401: Invalid API key", status=401, kind="auth",
                                       hint="Enter the key for this endpoint.")
        self.judge.complete = complete

    def test_a_failing_connection_gives_review_and_consult_an_actionable_error(self) -> None:
        self.failing()
        for result in (self.agent.tools.request_review("The cache is never stale, I checked."),
                       self.agent.tools.consult("Should the parser be rewritten or patched?")):
            self.assertEqual(result["status"], "error")
            self.assertIn("oversight", result["summary"])
            self.assertIn("Invalid API key", result["summary"])
            self.assertIn("continue the work", result["summary"])
        # Never silently answered by the worker in its place.
        self.assertEqual(len(self.worker.requests), 0)
        failed = self.lifetime("oversight_connection_failed")
        self.assertEqual([(f["connection"], f["wake_reason"]) for f in failed],
                         [("oversight", "review"), ("oversight", "consult")])
        self.assertEqual(len(self.lifetime("model_request_failed")), 2)
        self.assertEqual(self.lifetime("model_request_failed")[0]["connection"], "oversight")
        self.assertIs(self.agent.engine, self.worker)
        self.assertEqual(self.agent.config.model, "worker-model")

    def test_a_failing_connection_does_not_block_the_life_loop(self) -> None:
        self.failing()
        self.agent.oversight.min_tick_seconds = 0
        # An evaluation is due; its failure is recorded and the tick returns.
        with mock.patch.object(self.agent.oversight, "evaluation_due", return_value="interval"):
            self.assertEqual(self.agent.oversight.tick(force=True), [])
        self.assertEqual(self.lifetime("oversight_failed")[0]["step"], "maybe_evaluate")
        self.assertFalse(self.agent._request_blocked)
        # The worker carries on.
        self.worker.responses = ["<think>No action.</think>"]
        self.agent.run_once()
        self.assertGreaterEqual(len(self.worker.requests), 1)
        self.assertFalse(self.agent._request_blocked)

    def test_a_connection_that_cannot_be_built_is_an_error_too(self) -> None:
        with mock.patch("artificium.runtime.make_engine", side_effect=EngineError("bad settings", kind="settings")):
            result = self.agent.tools.consult("Should the parser be rewritten or patched?")
        self.assertEqual(result["status"], "error")
        self.assertIn("bad settings", result["summary"])
        self.assertEqual(len(self.lifetime("oversight_connection_failed")), 1)

    def test_an_oversight_request_failing_does_not_leave_the_swap_in_place(self) -> None:
        self.failing(RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            self.agent._oversight_complete([{"role": "user", "content": "q"}])
        self.assertEqual(self.agent._connection, "worker")
        self.assertIs(self.agent.engine, self.worker)


class ConsultToolTests(RuntimeCase):
    def test_the_question_context_directives_and_messages_reach_the_advisor_only(self) -> None:
        from artificium.interactions import ArtificiumClient
        client = ArtificiumClient(self.root)
        first, _ = client.send("room", sender="entity_1", content="Build the importer, quickly.")
        rule, _ = client.send("room", sender="entity_1", content="Never touch the vendor folder.")
        client.send("room", sender="entity_1", content="Latest word: prefer small commits.")
        self.agent.tools.record_directive(rule["id"], "Never touch the vendor folder.")
        evidence = self.paths.space / "notes.txt"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("the build fails at step 7\n")
        self.agent.working.append({"role": "assistant", "content": "PRIVATE WORKING DETAIL"}, origin="test")
        self.judge.responses = ["<think>weighing</think>RECOMMENDATION: patch it"]
        result = self.agent.tools.consult(
            "Should the parser be rewritten or patched in place?",
            evidence_paths=[str(evidence)], context="Three patches already failed.")
        self.assertEqual(result["status"], "consulted")
        self.assertEqual(result["answer"], "RECOMMENDATION: patch it")
        self.assertTrue(result["request_id"])
        sent = json.dumps(self.judge.requests[-1])
        for text in ("rewritten or patched in place", "Three patches already failed.",
                     "Never touch the vendor folder.", "Build the importer, quickly.",
                     "prefer small commits", "the build fails at step 7", "advisor"):
            self.assertIn(text, sent)
        self.assertNotIn("PRIVATE WORKING DETAIL", sent)
        self.assertEqual(len(self.lifetime("consult_completed")), 1)

    def test_evidence_is_bounded_by_the_oversight_models_window(self) -> None:
        big = self.paths.space / "big.txt"
        big.parent.mkdir(parents=True, exist_ok=True)
        big.write_text("line of evidence\n" * 20_000)
        self.judge.responses = ["RECOMMENDATION: ok"]
        result = self.agent.tools.consult("What should come first in this plan?", evidence_paths=[str(big)])
        self.assertTrue(result["evidence"][0]["truncated"])
        oversight = self.agent.config.oversight_config()
        limit = int(oversight.working_memory_limit * oversight.chars_per_token * 0.4)
        self.assertLessEqual(len(json.dumps(self.judge.requests[-1])), limit + 20_000)

    def test_bad_arguments_are_refused_with_what_to_do(self) -> None:
        with self.assertRaisesRegex(ValueError, "too short"):
            self.agent.tools.consult("Parser?")
        with self.assertRaisesRegex(FileNotFoundError, "not a file"):
            self.agent.tools.consult("Should the parser be rewritten or patched?", evidence_paths=["missing.txt"])
        self.assertEqual(len(self.judge.requests), 0)

    def test_consult_is_registered_with_its_example_and_arguments(self) -> None:
        tools = self.agent.tools
        self.assertIn("consult", tools._functions)
        self.assertEqual(tools.accepted_arguments("consult"),
                         {"required": ["question"], "optional": ["evidence_paths", "context"]})
        self.assertEqual(tools.canonical_example("consult")["tool"], "consult")

    def test_without_the_runtime_it_reports_itself_unavailable(self) -> None:
        tools = self.agent.tools
        tools.consultant = None
        self.assertEqual(tools.consult("Should the parser be rewritten or patched?")["status"], "error")

    def test_the_native_schema_names_the_question_and_its_types(self) -> None:
        catalog = PromptPack(self.paths).tool_catalog()
        schemas = {item["function"]["name"]: item["function"]
                   for item in native_tool_schemas(ToolRegistry, catalog)}
        function = schemas["consult"]
        parameters = function["parameters"]
        self.assertEqual(parameters["required"], ["question"])
        self.assertEqual(parameters["properties"], {
            "question": {"type": "string"},
            "evidence_paths": {"type": "array", "items": {"type": "string"}},
            "context": {"type": "string"},
        })
        self.assertIn("oversight model", function["description"])
        self.assertNotIn("<tool_call>", function["description"])
        self.assertNotIn('{"tool"', function["description"])

    def test_the_prompts_are_declared_and_fully_filled(self) -> None:
        prompts = PromptPack(self.paths)
        request = prompts.runtime("consult_request", question="Q", context="C", ask="A", evidence="E")
        self.assertNotIn("{{", request)
        self.assertIn("advisor", prompts.runtime("consult_system"))


class SetupCase(unittest.TestCase):
    """Configure the oversight connection against two loopback servers."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name) / "instance")
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        shutil.copytree(ROOT / "app/seed", self.paths.seed)
        self.paths.ensure_layout()
        self.worker = TestServer()
        self.judge = TestServer()
        self.judge.required_key = "judge-secret"
        self.addCleanup(self.worker.close)
        self.addCleanup(self.judge.close)
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        SetupWizard(self.paths).run(SetupOptions(provider="llamacpp", api_url=self.worker.url))
        self.secrets = SecretsStore(self.paths)

    def configure(self, **values) -> Config:
        options = SetupOptions(scope="oversight", **values)
        return SetupWizard(self.paths).reconfigure(options)

    def set_oversight(self) -> Config:
        return self.configure(provider="llamacpp", api_url=self.judge.url, api_key="judge-secret")

    def inference(self, server):
        # A streaming attempt the test server answers in plain JSON is retried
        # without streaming; count each logical request once.
        return [r for r in server.requests
                if r[1].endswith("/chat/completions") and not r[3].get("stream")]


class SetupTests(SetupCase):
    def test_setting_the_connection_checks_it_with_a_short_request_and_saves_its_key_apart(self) -> None:
        before = len(self.inference(self.worker))
        updated = self.set_oversight()
        self.assertEqual(updated.oversight_config().base_url, self.judge.url + "/v1")
        saved = read_json(self.paths.config)
        self.assertEqual(saved["oversight"]["model"], "local-model")
        self.assertEqual(set(saved["oversight"]), OVERSIGHT_FIELDS)
        # One short real request to the oversight server, none to the worker.
        sent = self.inference(self.judge)
        self.assertEqual(len(sent), 1)
        self.assertLess(len(json.dumps(sent[0][3]["messages"])), 1_000)
        self.assertEqual(sent[0][2].get("Authorization"), "Bearer judge-secret")
        self.assertEqual(len(self.inference(self.worker)), before)
        config = ConfigStore(self.paths).load()
        self.assertEqual(self.secrets.resolve_oversight_key(config.oversight_config(), config), "judge-secret")
        self.assertNotEqual(self.secrets.resolve_api_key(config), "judge-secret")
        # Reasoning/sampling settings and the window are the oversight model's own.
        self.assertEqual(config.oversight_config().context_window_tokens, 32768)
        self.assertEqual(config.oversight_config().tool_mode, "text")

    def test_a_failed_check_saves_nothing(self) -> None:
        before = self.paths.config.read_bytes()
        self.judge.inference_error = (500, {"error": {"message": "overloaded"}})
        with self.assertRaises(EngineError):
            self.set_oversight()
        self.assertEqual(self.paths.config.read_bytes(), before)
        self.assertNotIn("oversight", read_json(self.paths.secrets, {}))

    def test_changing_the_worker_keeps_the_oversight_connection_and_its_key(self) -> None:
        self.set_oversight()
        SetupWizard(self.paths).reconfigure(SetupOptions(scope="model", reasoning="on"))
        loaded = ConfigStore(self.paths).load()
        self.assertEqual(loaded.oversight_config().model, "local-model")
        self.assertEqual(self.secrets.resolve_oversight_key(loaded.oversight_config(), loaded), "judge-secret")
        SetupWizard(self.paths).reconfigure(SetupOptions(scope="harness", vision="no"))
        self.assertIsNotNone(ConfigStore(self.paths).load().oversight_config())

    def test_changing_the_oversight_connection_keeps_the_worker(self) -> None:
        before = ConfigStore(self.paths).load()
        self.set_oversight()
        self.configure(reasoning="high", context_window_tokens=16_000)
        after = ConfigStore(self.paths).load()
        self.assertEqual(after.oversight_config().reasoning_effort, "high")
        self.assertEqual(after.oversight_config().context_window_tokens, 16_000)
        self.assertEqual(after.oversight_config().base_url, self.judge.url + "/v1")
        self.assertEqual((after.provider, after.model, after.base_url, after.reasoning_effort),
                         (before.provider, before.model, before.base_url, before.reasoning_effort))

    def test_check_tests_the_saved_connection_and_changes_nothing(self) -> None:
        self.set_oversight()
        saved = self.paths.config.read_bytes()
        count = len(self.inference(self.judge))
        SetupWizard(self.paths).reconfigure(SetupOptions(scope="oversight", oversight_check=True))
        self.assertEqual(len(self.inference(self.judge)), count + 1)
        self.assertEqual(self.paths.config.read_bytes(), saved)
        self.judge.required_key = "rotated"
        with self.assertRaises(EngineError):
            SetupWizard(self.paths).reconfigure(SetupOptions(scope="oversight", oversight_check=True))

    def test_check_without_a_connection_says_how_to_set_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "configure oversight"):
            SetupWizard(self.paths).reconfigure(SetupOptions(scope="oversight", oversight_check=True))

    def test_off_removes_the_connection_and_its_key_only(self) -> None:
        self.set_oversight()
        SetupWizard(self.paths).reconfigure(SetupOptions(scope="oversight", oversight_off=True))
        loaded = ConfigStore(self.paths).load()
        self.assertIsNone(loaded.oversight_config())
        self.assertNotIn("oversight", read_json(self.paths.config))
        self.assertNotIn("oversight", read_json(self.paths.secrets, {}))
        self.assertEqual(loaded.model, "local-model")
        # Removing what is not there is harmless.
        SetupWizard(self.paths).reconfigure(SetupOptions(scope="oversight", oversight_off=True))

    def test_settings_that_do_not_apply_are_refused(self) -> None:
        for name, value in (("vision", "no"), ("tool_mode", "native"), ("auto_repair", True)):
            with self.assertRaisesRegex(ValueError, "do not apply"):
                self.configure(provider="llamacpp", api_url=self.judge.url, **{name: value})

    def test_the_connection_can_use_another_provider_preset(self) -> None:
        # A hosted provider needs a key; none was given, so nothing is saved.
        before = self.paths.config.read_bytes()
        with self.assertRaisesRegex(ValueError, "API key is required"):
            self.configure(provider="anthropic", model="judge-model")
        self.assertEqual(self.paths.config.read_bytes(), before)

    def test_an_interactive_menu_changes_checks_or_removes_an_existing_connection(self) -> None:
        self.set_oversight()
        answers = iter(["remove"])
        with mock.patch("builtins.input", side_effect=lambda prompt: next(answers)):
            SetupWizard(self.paths).reconfigure(SetupOptions(scope="oversight"), interactive=True)
        self.assertIsNone(ConfigStore(self.paths).load().oversight_config())

    def test_an_interactive_setup_of_a_new_connection_reuses_the_connection_screens(self) -> None:
        def answer(prompt):
            if prompt.startswith("Choose provider"):
                return "llamacpp"
            if prompt.startswith("Server address"):
                return self.judge.url
            return ""
        with mock.patch("builtins.input", side_effect=answer), \
                mock.patch("getpass.getpass", return_value="judge-secret"):
            SetupWizard(self.paths).reconfigure(SetupOptions(scope="oversight"), interactive=True)
        loaded = ConfigStore(self.paths).load()
        self.assertEqual(loaded.oversight_config().base_url, self.judge.url + "/v1")
        self.assertEqual(self.secrets.resolve_oversight_key(loaded.oversight_config(), loaded), "judge-secret")


class CliTests(SetupCase):
    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--root", str(self.paths.install), *argv])
        return code, out.getvalue(), err.getvalue()

    def test_configure_oversight_sets_checks_and_removes_from_the_command_line(self) -> None:
        key_file = Path(self.temporary.name) / "judge.key"
        key_file.write_text("judge-secret\n")
        code, out, _ = self.run_cli("configure", "oversight", "--provider", "llamacpp",
                                    "--api-url", self.judge.url, "--api-key-file", str(key_file))
        self.assertEqual(code, 0, out)
        self.assertIn("Oversight connection saved", out)
        self.assertIn("billed separately", out)
        self.assertNotIn("judge-secret", out)
        code, out, _ = self.run_cli("configure", "oversight", "--check")
        self.assertEqual((code, "works" in out), (0, True))
        code, out, _ = self.run_cli("configure", "oversight", "--off")
        self.assertEqual(code, 0)
        self.assertIn("removed", out)
        self.assertIsNone(ConfigStore(self.paths).load().oversight_config())

    def test_status_and_doctor_show_the_connection_and_never_a_key(self) -> None:
        self.set_oversight()
        state = status_snapshot(self.paths)
        self.assertEqual(state["oversight"]["model"], "local-model")
        self.assertTrue(state["oversight"]["separate_connection"])
        self.assertIn("Oversight: llamacpp / local-model (own connection", format_status(state))
        code, out, _ = self.run_cli("status", "--json")
        self.assertEqual(code, 0)
        self.assertNotIn("judge-secret", out)
        code, out, _ = self.run_cli("doctor")
        self.assertEqual(code, 0)
        report = json.loads(out)["oversight"]
        self.assertEqual((report["model"], report["api_key_available"]), ("local-model", True))
        self.assertNotIn("judge-secret", out)
        code, out, _ = self.run_cli("config", "oversight")
        self.assertEqual(json.loads(out)["model"], "local-model")

    def test_doctor_live_checks_the_oversight_connection_with_a_short_request(self) -> None:
        self.set_oversight()
        count = len(self.inference(self.judge))
        code, out, _ = self.run_cli("doctor", "--live")
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["oversight"]["connection_check"]["text_passed"])
        self.assertEqual(len(self.inference(self.judge)), count + 1)

    def test_the_plain_configure_menu_offers_the_oversight_connection(self) -> None:
        answers = iter(["2"])
        shown = []

        def fake_input(prompt):
            shown.append(prompt)
            return next(answers, "")

        with mock.patch("artificium.cli_admin.sys.stdin.isatty", return_value=True), \
                mock.patch("builtins.input", side_effect=fake_input), \
                mock.patch("artificium.setup.SetupWizard.reconfigure", return_value=ConfigStore(self.paths).load()) as reconfigure:
            code, out, _ = self.run_cli("configure")
        self.assertEqual(code, 0)
        self.assertIn("Oversight connection", out)
        self.assertEqual(reconfigure.call_args.args[0].scope, "oversight")

    def test_status_without_a_connection_says_the_worker_serves_oversight(self) -> None:
        self.assertIn("Oversight: the worker model", format_status(status_snapshot(self.paths)))


if __name__ == "__main__":
    unittest.main()
