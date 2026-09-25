"""Connection acceptance tests over real loopback HTTP, without model credentials.

These test the wire contract and setup recovery. They do not impersonate a
live llama.cpp model or assert model quality.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

from artificium.cli import main
from artificium.config import Config, ConfigStore, SecretsStore
from artificium.connection import preview_messages, verify_connection
from artificium.engine import EngineError, make_engine
from artificium.filesystem import Paths, atomic_write_json
from artificium.runtime import Artificium
from artificium.setup import SetupOptions, SetupWizard, normalize_provider_endpoint, reasoning_choices
from artificium.setup_ui import SetupCancelled


class TestServer:
    __test__ = False

    def __init__(self):
        self.context = 32768
        self.provider = "llamacpp"
        self.vision = False
        self.template = "{% if enable_thinking %}thinking{% endif %}"
        self.requests = []
        self.required_key = None
        self.inference_error = None
        self.raw_response = None
        self.reasoning_only = False
        self.metadata_missing = False
        self.delay = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.respond(None)

            def do_POST(self):
                self.respond(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))

            def respond(self, body):
                owner.requests.append((self.command, self.path, dict(self.headers), body))
                status, value = owner.route(self.path, dict(self.headers), body)
                raw = value if isinstance(value, bytes) else json.dumps(value).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.http.server_port}"

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()

    def route(self, path, headers, body):
        if self.required_key and headers.get("Authorization") != f"Bearer {self.required_key}":
            return 401, {"error": {"message": "Invalid API key"}}
        if path.endswith("/models"):
            if self.metadata_missing:
                return 404, {"error": "Model list not available"}
            if self.provider == "gemini":
                return 200, {"models": [{"name": "models/local-model", "supportedGenerationMethods": ["generateContent"],
                                         "inputTokenLimit": self.context, "outputTokenLimit": 8192, "thinking": False}]}
            detail = {"id": "local-model", "owned_by": self.provider}
            if self.provider == "vllm":
                detail["max_model_len"] = self.context
            if self.provider == "openrouter":
                detail.update(context_length=self.context, architecture={"input_modalities": ["text"]})
            if self.provider == "anthropic":
                detail.update(max_input_tokens=self.context, max_tokens=8192,
                              capabilities={"image_input": {"supported": False}, "thinking": {"supported": True, "types": {"adaptive": {"supported": True}}}})
            return 200, {"data": [detail]}
        if path.endswith("/api/tags"):
            return 200, {"models": [{"name": "local-model"}]}
        if path.endswith("/api/ps"):
            return 200, {"models": [{"name": "local-model", "context_length": 4096}]}
        if path.endswith("/api/show"):
            return 200, {"capabilities": ["completion", "thinking"], "model_info": {"llama.context_length": 65536}}
        if "/props?" in path:
            modalities = {} if self.vision is None else {"vision": self.vision}
            return 200, {"default_generation_settings": {"n_ctx": self.context},
                         "modalities": modalities, "chat_template": self.template}
        if path.endswith("/apply-template"):
            return 200, {"prompt": "".join(str(x["content"]) for x in body["messages"])}
        if path.endswith("/tokenize"):
            return 200, {"tokens": [1] * 12000}
        if path.endswith("/api/chat"):
            return 200, {"message": {"role": "assistant", "content": "OK"}, "done": True, "prompt_eval_count": 12000}
        if path.endswith("/responses"):
            return 200, {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}], "usage": {"input_tokens": 12000}}
        if path.endswith(":generateContent"):
            return 200, {"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 12000}}
        if path.endswith("/messages"):
            return 200, {"content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn", "usage": {"input_tokens": 12000}}
        if path.endswith("/chat/completions"):
            if self.delay:
                time.sleep(self.delay)
            if self.inference_error:
                return self.inference_error
            if self.raw_response is not None:
                return 200, self.raw_response
            if any(isinstance(m["content"], list) for m in body["messages"]):
                if self.vision is not True:
                    return 500, {"error": {"message": "mmproj is not loaded; model does not support images"}}
            if body.get("temperature") == 2:
                return 400, {"error": {"message": "Unsupported temperature setting"}}
            if self.reasoning_only:
                return 200, {"choices": [{"message": {"content": "", "reasoning_content": "still thinking"}, "finish_reason": "length"}]}
            return 200, {"choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 12000, "completion_tokens": 1}, "model": "local-model"}
        return 404, {"error": "Missing route"}


class ConnectionFlowCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(self.temp.cleanup)
        self.paths = Paths(Path(self.temp.name) / "instance")
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        shutil.copytree(ROOT / "app/seed", self.paths.seed)
        self.paths.ensure_layout()
        self.server = TestServer()
        self.addCleanup(self.server.close)
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def options(self, **values):
        return SetupOptions(provider="llamacpp", api_url=self.server.url, **values)

    def setup(self, **values):
        SetupWizard(self.paths).run(self.options(**values))
        return ConfigStore(self.paths).load()

    def inference_requests(self):
        return [r for r in self.server.requests if r[1].endswith("/chat/completions")]

    def test_setup_then_runtime_use_complete_prompt_and_same_wire_contract(self):
        config = self.setup(reasoning="on")
        checked = self.inference_requests()[0][3]
        self.assertGreater(len(checked["messages"][0]["content"]), 30000)
        self.assertTrue(any("FIRST WAKE" in str(m) for m in checked["messages"]))
        self.assertEqual(checked["chat_template_kwargs"], {"enable_thinking": True})
        self.assertFalse(checked["stream"])
        self.assertNotIn("tools", checked)
        self.assertEqual(self.paths.working_context.read_text(), "")
        config.max_life_loop_rounds = 1
        ConfigStore(self.paths).save(config)
        Artificium(self.paths).run_once()
        runtime = self.inference_requests()[-1][3]
        self.assertEqual({k: v for k, v in checked.items() if k != "messages"},
                         {k: v for k, v in runtime.items() if k != "messages"})
        # Pinned mind and tool contracts are the same; only state/check wording differs.
        self.assertEqual(checked["messages"][0]["content"].split("[SYSTEM STATE]")[0]
                         .removesuffix("\n\n---\n\n"),
                         runtime["messages"][0]["content"])
        self.assertIn("[SYSTEM STATE]", runtime["messages"][-1]["content"])

    def test_too_small_server_context_is_not_saved_or_sent_for_inference(self):
        self.server.context = 4096
        with self.assertRaises(EngineError) as caught:
            self.setup()
        self.assertEqual(caught.exception.kind, "context")
        self.assertIn("--ctx-size", caught.exception.hint)
        self.assertFalse(self.paths.config.exists())
        self.assertEqual(self.inference_requests(), [])

    def test_metadata_success_does_not_mask_inference_failure_or_replace_config(self):
        self.setup()
        before = self.paths.config.read_bytes()
        self.server.inference_error = (400, {"error": {"message": "Rejected request parameter"}})
        with self.assertRaises(EngineError):
            SetupWizard(self.paths).reconfigure(SetupOptions(scope="model", reasoning="on"))
        self.assertEqual(self.paths.config.read_bytes(), before)

    def test_selected_key_is_the_key_runtime_uses_even_with_old_environment_key(self):
        self.server.required_key = "new-key"
        with mock.patch.dict(os.environ, {"ARTIFICIUM_API_KEY": "old-key"}):
            config = self.setup(api_key="new-key")
            self.assertEqual(SecretsStore(self.paths).resolve_api_key(config), "new-key")
            make_engine(config, SecretsStore(self.paths).resolve_api_key(config)).complete([{"role": "user", "content": "OK"}])
        self.assertTrue(all(r[2].get("Authorization") == "Bearer new-key" for r in self.server.requests))

    def test_unknown_image_support_is_tested_and_rejection_becomes_text_only(self):
        self.server.vision = None
        config = self.setup()
        self.assertEqual((config.vision_preference, config.vision), ("auto", "no"))
        self.assertFalse(config.model_supports_vision)
        self.assertEqual(len(self.inference_requests()), 2)  # no repeated 500 image request

    def test_vision_model_passes_actual_image_transport(self):
        self.server.vision = True
        config = self.setup()
        image_request = self.inference_requests()[-1][3]
        image = image_request["messages"][-1]["content"][-1]
        self.assertTrue(image["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(config.vision, "auto")

    def test_no_vision_sends_no_image_and_does_not_consume_saved_visual_state(self):
        config = self.setup(vision="no")
        atomic_write_json(self.paths.visual_context, {"images": [{"path": "missing.png", "retention": "once"}]})
        before = self.paths.visual_context.read_bytes()
        verify_connection(self.paths, config, None)
        self.assertEqual(before, self.paths.visual_context.read_bytes())
        self.assertTrue(all(not isinstance(m["content"], list) for r in self.inference_requests() for m in r[3]["messages"]))

    def test_empty_reasoning_only_reply_cannot_pass_setup(self):
        self.server.reasoning_only = True
        with self.assertRaises(EngineError) as caught:
            self.setup()
        self.assertEqual(caught.exception.kind, "empty")
        self.assertIn("reasoning", caught.exception.hint)
        self.assertFalse(self.paths.config.exists())

    def test_non_json_response_is_actionable_and_not_retried(self):
        self.server.raw_response = b"<html>Sign in to this proxy</html>"
        with self.assertRaises(EngineError) as caught:
            self.setup()
        self.assertEqual(caught.exception.kind, "response")
        self.assertEqual(len(self.inference_requests()), 1)

    def test_http200_error_is_not_a_success(self):
        self.server.raw_response = b'{"error":{"message":"upstream unavailable"}}'
        with self.assertRaisesRegex(EngineError, "upstream unavailable"):
            self.setup()
        self.assertFalse(self.paths.config.exists())

    def test_timeout_does_not_replay_a_maybe_running_inference(self):
        self.server.delay = 1.3
        with self.assertRaises(EngineError) as caught:
            self.setup(request_timeout_seconds=1)
        self.assertEqual(caught.exception.kind, "timeout")
        self.assertEqual(len(self.inference_requests()), 1)

    def test_interactive_retry_keeps_harness_and_rediscovers_changed_capacity(self):
        self.server.context = 4096
        prompts = []
        def answer(prompt):
            prompts.append(prompt)
            if prompt.startswith("Next step"):
                self.server.context = 32768
                return "1"
            return ""
        with mock.patch("builtins.input", side_effect=answer):
            SetupWizard(self.paths).run(self.options(), interactive=True)
        self.assertEqual(sum(p.startswith("Vision") for p in prompts), 1)
        self.assertEqual(sum(p.startswith("Context capacity") for p in prompts), 0)
        self.assertEqual(ConfigStore(self.paths).load().context_window_tokens, 32768)

    def test_interactive_cancel_keeps_existing_configuration_and_key(self):
        self.setup()
        before = self.paths.config.read_bytes()
        self.server.inference_error = (400, {"error": {"message": "Unsupported parameter"}})
        def answer(prompt):
            return "9" if prompt.startswith("Next step") else ""
        with mock.patch("builtins.input", side_effect=answer), self.assertRaises(SetupCancelled):
            SetupWizard(self.paths).reconfigure(SetupOptions(scope="model"), interactive=True)
        self.assertEqual(before, self.paths.config.read_bytes())

    def test_advanced_editor_changes_one_setting_without_repeating_context(self):
        prompts = []
        menu = iter(["3", "done"])
        sampling = iter(["2", "done"])
        def answer(prompt):
            prompts.append(prompt)
            if prompt.startswith("Adjust advanced"):
                return "yes"
            if prompt.startswith("Setting number or done"):
                return next(menu)
            if prompt.startswith("Setting number, reset"):
                return next(sampling)
            if prompt.startswith("Max output tokens"):
                return "4096"
            return ""
        with mock.patch("builtins.input", side_effect=answer):
            SetupWizard(self.paths).run(self.options(), interactive=True)
        self.assertEqual(ConfigStore(self.paths).load().max_output_tokens, 4096)
        self.assertEqual(sum(p.startswith("Context capacity") for p in prompts), 0)

    def test_rejected_generation_setting_can_be_reset_without_restarting_setup(self):
        prompts = []
        def answer(prompt):
            prompts.append(prompt)
            return "6" if prompt.startswith("Next step") else ""
        with mock.patch("builtins.input", side_effect=answer):
            SetupWizard(self.paths).run(self.options(temperature=2), interactive=True)
        self.assertIsNone(ConfigStore(self.paths).load().temperature)
        self.assertEqual(sum(p.startswith("Vision") for p in prompts), 1)
        self.assertEqual(len(self.inference_requests()), 2)

    def test_explicit_new_key_is_not_saved_when_inference_rejects_it(self):
        self.setup(api_key="good-key")
        config_before = self.paths.config.read_bytes()
        key_before = self.paths.secrets.read_bytes()
        self.server.required_key = "good-key"
        with self.assertRaises(EngineError):
            SetupWizard(self.paths).reconfigure(SetupOptions(scope="model", api_key="bad-key"))
        self.assertEqual(self.paths.config.read_bytes(), config_before)
        self.assertEqual(self.paths.secrets.read_bytes(), key_before)

    def test_reconnect_refreshes_vision_after_model_replacement_under_same_alias(self):
        self.setup()
        self.assertFalse(ConfigStore(self.paths).load().model_supports_vision)
        self.server.vision = True
        updated = SetupWizard(self.paths).reconfigure(SetupOptions(scope="model"))
        self.assertTrue(updated.model_supports_vision)
        self.assertEqual(updated.vision, "auto")

    def test_reconnect_adjusts_stale_inherited_window_to_actual_server_capacity(self):
        self.setup(context_window_tokens=32768)
        self.server.context = 16384
        updated = SetupWizard(self.paths).reconfigure(SetupOptions(scope="model"))
        self.assertEqual(updated.context_window_tokens, 16384)

    def test_reasoning_controls_follow_template_metadata(self):
        self.assertEqual(reasoning_choices("llamacpp", "unknown-name", {"thinking_toggle": True, "effort_control": False}), ["auto", "off", "on"])
        self.assertEqual(reasoning_choices("ollama", "qwen3", {"thinking": True}), ["auto", "off", "on"])

    def test_each_native_provider_goes_from_discovery_to_full_inference(self):
        for provider, suffix in (("openai", "/v1/responses"), ("anthropic", "/v1/messages"),
                                 ("gemini", "/v1beta/models/local-model:generateContent"),
                                 ("ollama", "/api/chat"), ("vllm", "/v1/chat/completions"),
                                 ("openrouter", "/api/v1/chat/completions")):
            with self.subTest(provider=provider):
                self.server.provider = provider
                self.server.requests.clear()
                self.paths.config.unlink(missing_ok=True)
                options = SetupOptions(provider=provider, api_url=self.server.url, api_key="test-key", vision="no")
                SetupWizard(self.paths).run(options)
                config = ConfigStore(self.paths).load()
                request = self.server.requests[-1]
                self.assertEqual(request[1], suffix)
                body = request[3]
                self.assertGreater(len(json.dumps(body)), 30000)
                if provider == "ollama":
                    self.assertEqual(body["options"]["num_ctx"], 32768)
                    self.assertNotEqual(config.context_window_tokens, 4096)
                if provider == "anthropic":
                    self.assertEqual({k.lower(): v for k, v in request[2].items()}.get("x-api-key"), "test-key")
                    self.assertEqual({k.lower(): v for k, v in request[2].items()}.get("anthropic-version"), "2023-06-01")
                if provider == "gemini":
                    self.assertEqual({k.lower(): v for k, v in request[2].items()}.get("x-goog-api-key"), "test-key")

    def test_check_command_does_not_change_working_state_or_import_scheduler(self):
        self.setup()
        (self.paths.created_tools / "scheduler.py").write_text("raise RuntimeError('must not execute')")
        before = {str(p): p.read_bytes() for p in self.paths.root.rglob("*") if p.is_file()}
        with contextlib.redirect_stderr(io.StringIO()):
            result = main(["--root", str(self.paths.install), "check"])
        self.assertEqual(result, 0)
        after = {str(p): p.read_bytes() for p in self.paths.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_url_forms_preserve_reverse_proxy_prefix(self):
        for suffix in ("/proxy/v1", "/proxy/v1/chat/completions/"):
            self.assertEqual(normalize_provider_endpoint(self.server.url + suffix, provider="llamacpp"),
                             (self.server.url + "/proxy/v1", "chat/completions", "llamacpp"))
        self.assertEqual(normalize_provider_endpoint("api.openai.com/v1/responses", provider="openai"),
                         ("https://api.openai.com/v1", "responses", "openai_responses"))

    def test_custom_text_contract_does_not_falsely_pass_an_image_check(self):
        self.server.raw_response = b'{"text":"OK"}'
        contract = {"url": self.server.url + "/v1/chat/completions", "auth": False,
                    "body": {"prompt": "$artificium.prompt"}, "response": {"content": "/text"}}
        config = Config(provider="custom", model="local", base_url=self.server.url,
                        adapter="custom_json", custom_contract=contract, context_window_tokens=32768)
        checked, report = verify_connection(self.paths, config, None)
        self.assertFalse(report["image_passed"])
        self.assertEqual(checked.vision, "no")
        self.assertEqual(len(self.inference_requests()), 1)


if __name__ == "__main__":
    unittest.main()
