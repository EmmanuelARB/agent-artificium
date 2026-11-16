"""OpenCode Go: the provider preset and adapter, the client requirements of its
usage terms (own User-Agent, a session ID per conversation), model discovery,
and the cost it reports (charged cost, plan allowance, list price).

Everything runs against a loopback server or fakes: no real network.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

try:
    import test_connection_flow as flow  # a module, so its cases are not collected here
    from test_revolution import FakeEngine, _seed_install
except ImportError:  # run as tests.test_opencode_go
    from tests import test_connection_flow as flow
    from tests.test_revolution import FakeEngine, _seed_install

from artificium import cost
from artificium.cli import main
from artificium.config import (
    NATIVE_TOOL_ADAPTERS, PROVIDERS, Config, ConfigStore, SecretsStore, requires_api_key,
)
from artificium.cost import CostLedger
from artificium.discovery import discover_provider_models, models_endpoint
from artificium.engine import EngineError, EngineReply, make_engine
from artificium.engine_adapters import OpenCodeGoEngine
from artificium.engine_stream import _STREAMING_ADAPTERS, request_json
from artificium.filesystem import Paths, read_json
from artificium.records import Console
from artificium.runtime import Artificium
from artificium.session import SESSION_HEADER, LifeLoopSession, new_session_id
from artificium.setup import (
    PROVIDER_ORDER, SetupOptions, SetupWizard, normalize_provider_endpoint,
)
from artificium.usage import normalize_usage
from artificium.version import USER_AGENT

SESSION = re.compile(r"ses_[0-9a-f]{26}")
MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "hello"}]
GO_USAGE = {
    "prompt_tokens": 1200, "completion_tokens": 30, "cost": 0,
    "opencode": {"billing_source": "go", "list_cost": 0.0042, "allowance": 0.0013,
                 "tokens": {"input": 1200, "output": 30}},
}


class Gateway:
    """A loopback stand-in for the gateway: records headers, answers plain or SSE."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.status = 200
        self.usage = GO_USAGE
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_args) -> None:
                pass

            def do_GET(self) -> None:
                self.answer(None)

            def do_POST(self) -> None:
                self.answer(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))

            def answer(self, body) -> None:
                owner.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if owner.status != 200:
                    raw = json.dumps({"error": {"message": "Model is not available to this client"}}).encode()
                    self.send_response(owner.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if body and body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for chunk in (
                        {"choices": [{"delta": {"role": "assistant", "content": "O"}}]},
                        {"choices": [{"delta": {"content": "K"}, "finish_reason": "stop"}]},
                        {"choices": [], "usage": owner.usage},
                    ):
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                raw = json.dumps({
                    "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                    "usage": owner.usage,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.http.server_port}"

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()


def _config(gateway: Gateway | None = None, **changes) -> Config:
    base = {"provider": "opencode-go", "model": "glm-5.3"}
    if gateway is not None:
        base["base_url"] = gateway.url + "/inference/go/openai/v1"
    return Config(**{**base, **changes})


class PresetTests(unittest.TestCase):
    def test_the_preset_and_its_adapter(self) -> None:
        preset = PROVIDERS["opencode-go"]
        self.assertEqual((preset.base_url, preset.endpoint, preset.adapter, preset.key_environment),
                         ("https://opencode.ai/inference/go/openai/v1", "chat/completions",
                          "opencode_go", "OPENCODE_API_KEY"))
        self.assertTrue(preset.requires_api_key)
        self.assertTrue(requires_api_key("opencode-go"))
        config = _config()
        self.assertEqual((config.adapter, config.endpoint), ("opencode_go", "chat/completions"))
        self.assertIsInstance(make_engine(config, "key"), OpenCodeGoEngine)
        self.assertIn("opencode_go", NATIVE_TOOL_ADAPTERS)
        self.assertIn("opencode_go", _STREAMING_ADAPTERS)
        self.assertEqual(_config(tool_mode="native").tool_mode, "native")
        self.assertIn(("opencode-go", "OpenCode Go"), PROVIDER_ORDER)

    def test_the_request_is_a_chat_completions_request_with_native_tools_and_streaming(self) -> None:
        prepared = make_engine(_config(stream_responses=True, temperature=0.2, seed=3), "oc_sk_x").prepare(MESSAGES)
        self.assertEqual(prepared.url, "https://opencode.ai/inference/go/openai/v1/chat/completions")
        self.assertEqual(prepared.headers["Authorization"], "Bearer oc_sk_x")
        self.assertEqual(prepared.payload["model"], "glm-5.3")
        self.assertEqual(prepared.payload["stream_options"], {"include_usage": True})
        self.assertEqual((prepared.payload["temperature"], prepared.payload["seed"]), (0.2, 3))
        with self.assertRaisesRegex(EngineError, "OpenCode Go does not define"):
            make_engine(_config(reasoning_mode="pro"), "k").prepare(MESSAGES)

    def test_the_cli_accepts_the_provider_and_the_endpoint_is_normalized(self) -> None:
        for pasted in ("https://opencode.ai", "https://opencode.ai/inference/go/openai/v1",
                       "https://opencode.ai/inference/go/openai/v1/chat/completions"):
            base, endpoint, adapter = normalize_provider_endpoint(pasted, provider="opencode-go")
            self.assertEqual(adapter, "opencode_go")
            self.assertEqual(endpoint, "chat/completions")
            if pasted != "https://opencode.ai":
                self.assertEqual(base, "https://opencode.ai/inference/go/openai/v1")
        self.assertEqual(normalize_provider_endpoint("https://opencode.ai", provider="opencode-go")[0],
                         "https://opencode.ai/inference/go/openai/v1")
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["setup", "--help"])
        self.assertIn("opencode-go", out.getvalue())


class HeaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = Gateway()
        self.addCleanup(self.gateway.close)

    def sent(self, index: int = -1) -> dict:
        return {name.lower(): value for name, value in self.gateway.requests[index]["headers"].items()}

    def test_a_plain_request_carries_the_product_user_agent_and_a_session(self) -> None:
        reply = make_engine(_config(self.gateway), "oc_sk_test").complete(MESSAGES)
        self.assertEqual(reply.content, "OK")
        headers = self.sent()
        self.assertEqual(headers["user-agent"], USER_AGENT)
        self.assertTrue(headers["user-agent"].startswith("Artificium-"))
        self.assertNotIn("python", headers["user-agent"].lower())
        self.assertRegex(headers[SESSION_HEADER], SESSION)
        self.assertEqual(headers["authorization"], "Bearer oc_sk_test")
        self.assertEqual(self.gateway.requests[-1]["path"], "/inference/go/openai/v1/chat/completions")

    def test_a_streamed_request_carries_them_too(self) -> None:
        reply = make_engine(_config(self.gateway, stream_responses=True), "oc_sk_test").complete(MESSAGES)
        self.assertEqual(reply.content, "OK")
        self.assertTrue(self.gateway.requests[-1]["body"]["stream"])
        headers = self.sent()
        self.assertEqual(headers["user-agent"], USER_AGENT)
        self.assertRegex(headers[SESSION_HEADER], SESSION)

    def test_an_engine_keeps_one_session_and_two_engines_never_share_one(self) -> None:
        first, second = make_engine(_config(self.gateway), "k"), make_engine(_config(self.gateway), "k")
        for engine in (first, first, second):
            engine.complete(MESSAGES)
        a, b, c = (self.sent(i)[SESSION_HEADER] for i in range(3))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_the_session_the_runtime_sets_wins_and_leaks_nothing(self) -> None:
        engine = make_engine(_config(self.gateway), "oc_sk_secret_key")
        engine.session_id = "ses_" + "ab" * 13
        engine.complete(MESSAGES)
        self.assertEqual(self.sent()[SESSION_HEADER], "ses_" + "ab" * 13)
        for _ in range(20):
            token = new_session_id()
            self.assertRegex(token, SESSION)
            self.assertNotIn("secret", token)

    def test_every_provider_sends_the_product_user_agent_on_every_transport(self) -> None:
        # The default of urllib ("Python-urllib/3.x") must reach no provider:
        # plain JSON, streamed, a bare GET (discovery, checks) and a count call.
        generic = make_engine(Config(provider="custom", model="m", base_url=self.gateway.url + "/v1",
                                     stream_responses=True), None)
        generic.complete(MESSAGES)
        request_json(self.gateway.url + "/v1/models")
        discover_provider_models("custom", self.gateway.url + "/v1")
        make_engine(Config(provider="custom", model="m", base_url=self.gateway.url + "/v1"), None).complete(MESSAGES)
        self.assertEqual(len(self.gateway.requests), 4)
        for index in range(4):
            self.assertEqual(self.sent(index)["user-agent"], USER_AGENT, index)

    def test_a_403_says_to_choose_a_paid_model(self) -> None:
        self.gateway.status = 403
        for streaming in (False, True):
            engine = make_engine(_config(self.gateway, stream_responses=streaming), "k")
            with self.assertRaises(EngineError) as caught:
                engine.complete(MESSAGES)
            error = caught.exception
            self.assertEqual((error.status, error.kind), (403, "permission"))
            self.assertIn("paid OpenCode Go model", str(error))
            self.assertIn("free-tier", error.hint)
            self.assertIn("Model is not available", str(error))
            self.assertTrue(error.requires_operator_action)

    def test_another_status_keeps_its_own_message(self) -> None:
        self.gateway.status = 401
        with self.assertRaises(EngineError) as caught:
            make_engine(_config(self.gateway), "k").complete(MESSAGES)
        self.assertEqual(caught.exception.kind, "auth")
        self.assertNotIn("paid", str(caught.exception))


class DiscoveryTests(unittest.TestCase):
    def test_the_model_list_is_not_under_the_openai_route(self) -> None:
        self.assertEqual(models_endpoint("opencode-go", "https://opencode.ai/inference/go/openai/v1"),
                         "https://opencode.ai/inference/go/v1/models")
        self.assertEqual(models_endpoint("opencode-go", "https://proxy.test/x/go/openai/v1/"),
                         "https://proxy.test/x/go/v1/models")
        self.assertEqual(models_endpoint("opencode-go", "https://proxy.test/go/v1"),
                         "https://proxy.test/go/v1/models")
        self.assertEqual(models_endpoint("opencode-go", "https://proxy.test/go"),
                         "https://proxy.test/go/v1/models")
        # Other providers are unchanged.
        self.assertEqual(models_endpoint("mistral", "https://api.mistral.ai/v1"),
                         "https://api.mistral.ai/v1/models")

    def test_models_and_their_metadata_are_listed_with_a_session_and_the_key(self) -> None:
        seen = {}

        def fake(url, **kwargs):
            seen.update(url=url, **kwargs)
            return {"data": [{"id": "glm-5.3", "owned_by": "z", "context_length": 200_000,
                              "max_completion_tokens": 32_000},
                             {"id": "kimi-k3", "architecture": {"input_modalities": ["text", "image"]}}]}

        with mock.patch("artificium.discovery.request_json", fake):
            found = discover_provider_models("opencode-go", "https://opencode.ai/inference/go/openai/v1",
                                             api_key="oc_sk_x")
        self.assertEqual(seen["url"], "https://opencode.ai/inference/go/v1/models")
        self.assertEqual(seen["headers"]["Authorization"], "Bearer oc_sk_x")
        self.assertRegex(seen["headers"][SESSION_HEADER], SESSION)
        self.assertEqual(found.models, ("glm-5.3", "kimi-k3"))
        self.assertEqual(found.details["glm-5.3"]["context_length"], 200_000)
        self.assertEqual(found.details["glm-5.3"]["max_output_tokens"], 32_000)
        self.assertTrue(found.details["kimi-k3"]["vision"])


class SetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name) / "instance")
        _seed_install(self.paths)
        self.server = flow.TestServer()
        self.addCleanup(self.server.close)
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def test_a_new_setup_defaults_to_native_tools_and_streaming(self) -> None:
        self.server.provider = "openrouter"
        SetupWizard(self.paths).run(SetupOptions(
            provider="opencode-go", api_url=self.server.url + "/inference/go/openai/v1",
            model="local-model", api_key="oc_sk_test"))
        config = ConfigStore(self.paths).load()
        self.assertEqual((config.provider, config.adapter), ("opencode-go", "opencode_go"))
        self.assertEqual(config.base_url, self.server.url + "/inference/go/openai/v1")
        self.assertEqual(config.tool_mode, "native")
        self.assertTrue(config.stream_responses)
        self.assertEqual(config.context_window_tokens, 32768)
        paths = [request[1] for request in self.server.requests]
        self.assertIn("/inference/go/v1/models", paths)
        for _, path, headers, _ in self.server.requests:
            lowered = {name.lower(): value for name, value in headers.items()}
            self.assertEqual(lowered["user-agent"], USER_AGENT, path)
            self.assertRegex(lowered[SESSION_HEADER], SESSION, path)
        self.assertEqual(SecretsStore(self.paths).resolve_api_key(config), "oc_sk_test")


class CostParsingTests(unittest.TestCase):
    def test_the_charged_cost_and_plan_usage_are_read_from_a_plain_response(self) -> None:
        gateway = Gateway()
        self.addCleanup(gateway.close)
        reply = make_engine(_config(gateway), "k").complete(MESSAGES)
        self.check(reply)

    def test_and_from_the_final_chunk_of_a_stream(self) -> None:
        gateway = Gateway()
        self.addCleanup(gateway.close)
        reply = make_engine(_config(gateway, stream_responses=True), "k").complete(MESSAGES)
        self.check(reply)

    def check(self, reply: EngineReply) -> None:
        config = _config()
        normalized = normalize_usage(reply.usage, reply.raw)
        self.assertEqual(cost.price_usage(config, reply.usage, normalized), (0.0, "reported"))
        self.assertEqual(cost.reported_plan_usage(config, reply.usage),
                         {"allowance": 0.0013, "list_cost": 0.0042, "billing_source": "go"})

    def test_a_charge_beyond_the_plan_is_the_reported_cost(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.031,
                 "opencode": {"billing_source": "balance", "list_cost": 0.031}}
        config = _config()
        self.assertEqual(cost.price_usage(config, usage, normalize_usage(usage))[0], 0.031)
        self.assertEqual(cost.reported_plan_usage(config, usage),
                         {"list_cost": 0.031, "billing_source": "balance"})

    def test_missing_malformed_or_foreign_figures_are_ignored(self) -> None:
        config = _config()
        for usage in (None, {}, {"opencode": "x"}, {"opencode": {"allowance": -1, "list_cost": "a"}},
                      {"opencode": {"allowance": True}}, {"opencode": {"allowance": float("nan")}}):
            self.assertEqual(cost.reported_plan_usage(config, usage), {}, usage)
        # Another adapter's usage is not read this way.
        self.assertEqual(cost.reported_plan_usage(_config(provider="openrouter"), GO_USAGE), {})
        # Without a reported cost, prices (if any) still apply, else it is unknown.
        bare = {"prompt_tokens": 10, "completion_tokens": 5}
        self.assertEqual(cost.price_usage(config, bare, normalize_usage(bare))[1], "unknown")

    def test_a_daily_limit_needs_no_prices_for_this_provider(self) -> None:
        self.assertIsNone(cost.cost_limit_problem(_config(daily_cost_limit=5.0)))


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))
        self.paths.ensure_layout()
        self.ledger = CostLedger(self.paths, _config(daily_cost_limit=1.0))

    def test_allowance_and_list_cost_are_summed_per_connection_and_day(self) -> None:
        usage = {"input_tokens": 10, "output_tokens": 5}
        plan = {"allowance": 0.25, "list_cost": 0.5, "billing_source": "go"}
        self.ledger.record("worker", usage, 0.0, plan_usage=plan)
        self.ledger.record("worker", usage, 0.0, plan_usage={"allowance": 0.125, "list_cost": 0.25})
        self.ledger.record("oversight", usage, 0.0, plan_usage=plan)
        self.ledger.record("worker", usage, 0.0)
        status = self.ledger.status()
        worker, oversight = status["connections"]["worker"], status["connections"]["oversight"]
        self.assertEqual((worker["allowance"], worker["list_cost"], worker["plan_usage_requests"]),
                         (0.375, 0.75, 2))
        self.assertEqual((oversight["allowance"], oversight["list_cost"]), (0.25, 0.5))
        self.assertEqual(worker["requests"], 3)

    def test_the_daily_limit_counts_charged_dollars_not_the_allowance(self) -> None:
        for _ in range(5):
            self.ledger.record("worker", {"input_tokens": 1, "output_tokens": 1}, 0.0,
                               plan_usage={"allowance": 0.9, "list_cost": 2.0})
        status = self.ledger.status()
        self.assertEqual(status["spent"], 0.0)
        self.assertFalse(status["limited"])
        self.ledger.record("worker", {"input_tokens": 1, "output_tokens": 1}, 1.0)
        self.assertTrue(self.ledger.status()["limited"])

    def test_the_report_shows_the_allowance_used(self) -> None:
        self.ledger.record("worker", {"input_tokens": 10, "output_tokens": 5}, 0.0,
                           plan_usage={"allowance": 0.0123, "list_cost": 0.5})
        text = cost.format_report(self.ledger.report())
        self.assertIn("plan allowance used 0.0123, 0.5000 at list price", text)
        self.assertIn("allowance is the figure to watch", text)
        other = CostLedger(self.paths, _config())
        other.record("worker", {"input_tokens": 1, "output_tokens": 1}, 0.1)
        # Days without any plan usage get no allowance wording.
        fresh = Paths(Path(self.temporary.name) / "other")
        fresh.ensure_layout()
        plain = CostLedger(fresh, _config())
        plain.record("worker", {"input_tokens": 1, "output_tokens": 1}, 0.1)
        self.assertNotIn("allowance", cost.format_report(plain.report()))


class SessionFake(FakeEngine):
    """A fake that, like the OpenCode Go engine, has a session to set."""

    session_id = None

    def __init__(self, responses=()):
        super().__init__(list(responses))
        self.sessions: list[str | None] = []

    def prepare(self, messages):
        self.sessions.append(self.session_id)
        return super().prepare(messages)


class RuntimeCase(unittest.TestCase):
    oversight: dict | None = None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name) / "instance one")
        _seed_install(self.paths)
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        ConfigStore(self.paths).save(_config(
            context_window_tokens=30_000, oversight_connection=self.oversight))
        SecretsStore(self.paths).save_api_key("oc_sk_the_secret_key")
        self.judge = SessionFake()
        patch = mock.patch("artificium.runtime.make_engine", return_value=self.judge)
        patch.start()
        self.addCleanup(patch.stop)
        self.agent = self.start()

    def start(self) -> Artificium:
        self.worker = SessionFake()
        agent = Artificium(self.paths, engine=self.worker, console=Console(quiet=True))
        agent.initialization.finish("The test instance is initialized and ready for its task.")
        return agent

    def ask(self, agent: Artificium | None = None) -> None:
        (agent or self.agent)._complete([{"role": "user", "content": "work"}], wake_reason="test")

    def add(self, text: str, agent: Artificium | None = None) -> None:
        (agent or self.agent).working.append({"role": "user", "content": text}, origin="test")

    def offload(self, agent: Artificium | None = None) -> None:
        (agent or self.agent).working.offload(
            title="Checkpoint of the test working context", compression="Everything that matters, in a form that carries on. " * 2,
            reason="test", memory_path=self.paths.mind / "checkpoint.md")

    def lifetime(self, kind: str) -> list[dict]:
        return [json.loads(line) for line in self.paths.lifetime_log.read_text().splitlines()
                if json.loads(line).get("kind") == kind]


class LifeLoopSessionTests(RuntimeCase):
    def test_the_life_loops_session_is_stable_across_turns(self) -> None:
        self.ask()
        self.add("one")
        self.ask()
        self.add("two")
        self.add("three")
        self.ask()
        first, *rest = self.worker.sessions
        self.assertRegex(first, SESSION)
        self.assertEqual(rest, [first, first])

    def test_it_survives_a_restart(self) -> None:
        self.add("one")
        self.ask()
        before = self.worker.sessions[-1]
        self.add("two")
        restarted = self.start()
        self.ask(restarted)
        self.assertEqual(self.worker.sessions, [before])
        self.add("three", restarted)
        self.ask(restarted)
        self.assertEqual(self.worker.sessions, [before, before])

    def test_it_is_kept_in_runtime_state_without_the_key_or_the_path(self) -> None:
        self.add("one")
        self.ask()
        state_file = self.paths.runtime / "life-loop-session.json"
        state = read_json(state_file)
        self.assertEqual(state["session_id"], self.worker.sessions[-1])
        text = state_file.read_text()
        self.assertNotIn("oc_sk_the_secret_key", text)
        self.assertNotIn("instance one", text)
        self.assertNotIn(str(self.paths.install), self.worker.sessions[-1])

    def test_a_working_memory_offload_renews_it_once(self) -> None:
        self.add("one")
        self.ask()
        self.add("two")
        self.offload()
        self.ask()
        self.add("after")
        self.ask()
        before, renewed, after = self.worker.sessions
        self.assertNotEqual(before, renewed)
        self.assertEqual(renewed, after)
        self.assertRegex(renewed, SESSION)

    def test_any_other_rebuild_of_the_working_context_renews_it(self) -> None:
        for text in ("one", "two"):
            self.add(text)
        self.ask()
        history = self.agent.working.load()
        # The context rewritten the way a repair or a compression does: not extended.
        self.paths.working_context.write_text("".join(json.dumps(m) + "\n" for m in history[:1]))
        self.ask()
        self.assertNotEqual(*self.worker.sessions)
        # A rebuild that keeps the length but changes a message.
        changed = [dict(history[0], content="different")]
        self.paths.working_context.write_text("".join(json.dumps(m) + "\n" for m in changed))
        self.ask()
        self.assertNotEqual(self.worker.sessions[1], self.worker.sessions[2])

    def test_a_request_that_fails_and_is_retried_keeps_its_session(self) -> None:
        self.add("one")
        self.worker.complete = mock.Mock(side_effect=[EngineError("busy", status=503), EngineReply("<think>ok</think>")])
        with self.assertRaises(EngineError):
            self.ask()
        self.ask()
        self.assertEqual(self.worker.sessions[0], self.worker.sessions[1])


class FreshContextSessionTests(RuntimeCase):
    def test_each_fresh_context_request_has_its_own_session_on_the_worker_engine(self) -> None:
        self.add("one")
        self.ask()
        life = self.worker.sessions[-1]
        for _ in range(3):
            self.agent._oversight_complete([{"role": "user", "content": "judge"}], wake_reason="review")
        fresh = self.worker.sessions[1:]
        self.assertEqual(len(set(fresh)), 3)
        self.assertNotIn(life, fresh)
        self.ask()
        self.assertEqual(self.worker.sessions[-1], life)
        # Fresh requests never touch the life-loop's saved state.
        self.assertEqual(read_json(self.paths.runtime / "life-loop-session.json")["session_id"], life)

    def test_review_consult_and_the_checks_all_get_one(self) -> None:
        self.worker.responses = ["VERDICT: ok"] * 2
        self.agent.tools.request_review("The cache is never stale, I checked.")
        self.agent.tools.consult("Should the parser be rewritten or patched?")
        a, b = self.worker.sessions
        self.assertNotEqual(a, b)
        self.assertRegex(a, SESSION)


    def test_a_decision_review_gets_one_too(self) -> None:
        self.worker.responses = ["NECESSARY: yes\nFRAMING: ok\nPOINTS: none\nDEFAULT: none\nREASONS: x"]
        result = self.agent.tools.send_interaction("room", "Which parser?", decision=True,
                                                   new_interaction=True)
        self.assertEqual(result["status"], "review_required")
        (session,) = self.worker.sessions
        self.assertRegex(session, SESSION)


class OversightConnectionSessionTests(RuntimeCase):
    oversight = {"provider": "opencode-go", "model": "kimi-k3", "context_window_tokens": 30_000}

    def test_the_oversight_engine_gets_a_new_session_per_request(self) -> None:
        self.add("one")
        self.ask()
        for _ in range(2):
            self.agent._oversight_complete([{"role": "user", "content": "judge"}])
        first, second = self.judge.sessions
        self.assertNotEqual(first, second)
        self.assertNotIn(self.worker.sessions[0], (first, second))
        self.assertEqual(len(self.worker.sessions), 1)


class AccountingTests(RuntimeCase):
    def test_the_response_record_and_the_ledger_carry_the_allowance(self) -> None:
        self.worker.complete = lambda messages: EngineReply("<think>ok</think>", usage=GO_USAGE)
        self.ask()
        record = self.lifetime("model_response")[-1]
        self.assertEqual((record["cost"], record["cost_source"]), (0.0, "reported"))
        self.assertEqual((record["allowance"], record["list_cost"], record["billing_source"]),
                         (0.0013, 0.0042, "go"))
        worker = self.agent.cost.status()["connections"]["worker"]
        self.assertEqual((worker["allowance"], worker["list_cost"], worker["cost"]), (0.0013, 0.0042, 0.0))

    def test_the_oversight_connection_is_accounted_apart(self) -> None:
        self.agent.config = dataclasses.replace(self.agent.config, oversight_connection={
            "provider": "opencode-go", "model": "kimi-k3"})
        self.judge.complete = lambda messages: EngineReply(
            "VERDICT: ok", usage=dict(GO_USAGE, opencode=dict(GO_USAGE["opencode"], allowance=0.5)))
        self.agent._oversight_complete([{"role": "user", "content": "judge"}])
        connections = self.agent.cost.status()["connections"]
        self.assertEqual(connections["oversight"]["allowance"], 0.5)
        self.assertNotIn("worker", connections)
        self.assertEqual(self.lifetime("model_response")[-1]["connection"], "oversight")

    def test_other_providers_records_have_no_plan_fields(self) -> None:
        self.agent.config = dataclasses.replace(self.agent.config, provider="openrouter",
                                                adapter="openrouter", model="m")
        self.worker.complete = lambda messages: EngineReply(
            "<think>ok</think>", usage={"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.01})
        self.ask()
        record = self.lifetime("model_response")[-1]
        self.assertNotIn("allowance", record)
        self.assertNotIn("allowance", self.agent.cost.status()["connections"]["worker"])

    def test_the_cost_command_shows_the_allowance(self) -> None:
        self.worker.complete = lambda messages: EngineReply("<think>ok</think>", usage=GO_USAGE)
        self.ask()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--root", str(self.paths.install), "cost"])
        self.assertEqual(code, 0)
        self.assertIn("plan allowance used 0.0013", out.getvalue())


if __name__ == "__main__":
    unittest.main()
