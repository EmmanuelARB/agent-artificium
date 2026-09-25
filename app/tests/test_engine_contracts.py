from __future__ import annotations

import json
import io
import copy
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from artificium.config import Config, load_api_key_file
from artificium.engine import EngineError, make_engine, _server_error
from artificium.setup import ModelDiscovery, validate_discovered_settings


MESSAGES = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "hello"},
]


class DummyResponse:
    def __init__(self, value: dict):
        self.value = value

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode()


class EngineContractCase(unittest.TestCase):
    def test_llamacpp_quotes_marker_text_without_changing_images_or_history(self) -> None:
        marker = "<__media_0123456789abcdefghijklmnopqrstuv__>"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Server properties: " + marker},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}, {"role": "assistant", "content": marker, "reasoning_content": marker}]
        original = copy.deepcopy(messages)
        config = Config(provider="llamacpp", model="test", temperature=0.8, top_k=20)
        prepared = make_engine(config, None).prepare(messages)
        self.assertNotIn(marker, json.dumps(prepared.payload["messages"]))
        self.assertEqual(prepared.payload["messages"][0]["content"][1], original[0]["content"][1])
        self.assertEqual(prepared.payload["temperature"], 0.8)
        self.assertEqual(prepared.payload["top_k"], 20)
        self.assertEqual(messages, original)
        other = make_engine(Config(provider="vllm", model="test"), None).prepare(messages)
        self.assertEqual(other.payload["messages"], original)

    def test_tokenization_rejection_is_actionable_and_does_not_disable_vision(self) -> None:
        failure = _server_error('{"error":{"message":"Failed to tokenize prompt"}}', 400)
        self.assertEqual(failure.kind, "tokenization")
        self.assertTrue(failure.requires_operator_action)
        self.assertFalse(failure.image_input_unsupported)
        self.assertIn("reserved-marker", failure.hint)
        genuine = _server_error('{"error":{"message":"image input is unsupported"}}', 415)
        self.assertTrue(genuine.image_input_unsupported)
        self.assertFalse(_server_error("mmproj: out of memory", 500).image_input_unsupported)

    def test_openrouter_uses_reasoning_object_and_strict_parameter_routing(self) -> None:
        config = Config(
            provider="openrouter",
            model="openai/gpt-5.6-luna",
            reasoning_effort="low",
            reasoning_mode="pro",
            temperature=0.2,
            max_output_tokens=8192,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            repetition_penalty=1.05,
            seed=42,
            stop_sequences=["STOP"],
            request_options={"provider": {"order": ["OpenAI"]}},
        )
        prepared = make_engine(config, "secret").prepare(MESSAGES)
        self.assertEqual(prepared.url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(
            prepared.payload["reasoning"], {"effort": "low", "mode": "pro"}
        )
        self.assertEqual(prepared.payload["max_tokens"], 8192)
        self.assertEqual(prepared.payload["top_k"], 40)
        self.assertEqual(prepared.payload["min_p"], 0.05)
        self.assertEqual(prepared.payload["stop"], ["STOP"])
        self.assertEqual(
            prepared.payload["provider"],
            {"require_parameters": True, "order": ["OpenAI"]},
        )
        summary = prepared.safe_summary()
        self.assertNotIn("secret", json.dumps(summary))
        self.assertEqual(summary["body"]["messages"], "<runtime prompt omitted>")

    def test_openrouter_rejects_ambiguous_effort_and_budget_combination(self) -> None:
        config = Config(
            provider="openrouter",
            model="model",
            reasoning_effort="high",
            reasoning_budget_tokens=4096,
        )
        with self.assertRaisesRegex(EngineError, "effort or max_tokens, not both"):
            make_engine(config, "key").prepare(MESSAGES)

    def test_actual_openrouter_http_body_matches_inspected_contract(self) -> None:
        config = Config(
            provider="openrouter",
            model="openai/gpt-5.6-luna",
            reasoning_effort="medium",
            temperature=0.25,
            max_output_tokens=2048,
        )
        engine = make_engine(config, "do-not-log-this-key")
        expected = engine.prepare([{"role": "user", "content": "hello"}]).payload
        captured: dict = {}

        def fake_open(request, timeout=0):
            captured.update(json.loads(request.data.decode()))
            self.assertEqual(
                request.get_header("Authorization"),
                "Bearer do-not-log-this-key",
            )
            return DummyResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "OK"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        with mock.patch("urllib.request.urlopen", side_effect=fake_open):
            reply = engine.complete([{"role": "user", "content": "hello"}])
        self.assertEqual(captured, expected)
        self.assertEqual(captured["reasoning"], {"effort": "medium"})
        self.assertEqual(captured["provider"], {"require_parameters": True})
        self.assertEqual(reply.content, "OK")

    def test_ollama_uses_native_endpoint_and_native_option_names(self) -> None:
        config = Config(
            provider="ollama",
            model="qwen3:8b",
            context_window_tokens=65_536,
            reasoning_effort="low",
            temperature=0.3,
            max_output_tokens=4096,
            top_p=0.8,
            top_k=20,
            min_p=0.1,
            repetition_penalty=1.1,
            seed=7,
            stop_sequences=["END"],
        )
        prepared = make_engine(config, None).prepare(MESSAGES)
        self.assertEqual(prepared.url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(prepared.payload["think"], "low")
        self.assertEqual(
            prepared.payload["options"],
            {
                "num_ctx": 65_536,
                "temperature": 0.3,
                "num_predict": 4096,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.1,
                "repeat_penalty": 1.1,
                "seed": 7,
                "stop": ["END"],
            },
        )
        self.assertNotIn("Authorization", prepared.headers)

    def test_actual_ollama_http_body_matches_prepared_contract(self) -> None:
        config = Config(
            provider="ollama",
            model="qwen3:8b",
            context_window_tokens=32_768,
            reasoning_effort="none",
            max_output_tokens=1024,
        )
        engine = make_engine(config, None)
        expected = engine.prepare([{"role": "user", "content": "hello"}]).payload
        captured: dict = {}

        def fake_open(request, timeout=0):
            captured.update(json.loads(request.data.decode()))
            return DummyResponse({"message": {"content": "OK"}, "done": True})

        with mock.patch("urllib.request.urlopen", side_effect=fake_open):
            reply = engine.complete([{"role": "user", "content": "hello"}])
        self.assertEqual(captured, expected)
        self.assertEqual(reply.content, "OK")

    def test_vllm_uses_chat_extensions_without_faking_context_control(self) -> None:
        config = Config(
            provider="vllm",
            model="openai/gpt-oss-120b",
            context_window_tokens=50_000,
            reasoning_effort="medium",
            reasoning_budget_tokens=12_000,
            max_output_tokens=5000,
            top_k=50,
            min_p=0.03,
            repetition_penalty=1.08,
            seed=9,
        )
        body = make_engine(config, None).prepare(MESSAGES).payload
        self.assertEqual(body["reasoning_effort"], "medium")
        self.assertEqual(body["thinking_token_budget"], 12_000)
        self.assertEqual(body["max_tokens"], 5000)
        self.assertNotIn("context_window", body)
        self.assertNotIn("max_model_len", body)

    def test_vllm_rejects_reasoning_levels_outside_its_contract(self) -> None:
        config = Config(
            provider="vllm",
            model="openai/gpt-oss-120b",
            reasoning_effort="xhigh",
        )
        with self.assertRaisesRegex(EngineError, "none, low, medium, or high"):
            make_engine(config, None).prepare(MESSAGES)

    def test_openai_uses_responses_api_reasoning_and_output_fields(self) -> None:
        config = Config(
            provider="openai",
            model="gpt-5.6-luna",
            reasoning_effort="high",
            reasoning_mode="pro",
            max_output_tokens=6000,
            temperature=0.4,
            top_p=0.95,
        )
        prepared = make_engine(config, "key").prepare(MESSAGES)
        self.assertEqual(prepared.url, "https://api.openai.com/v1/responses")
        self.assertEqual(
            prepared.payload["reasoning"], {"effort": "high", "mode": "pro"}
        )
        self.assertEqual(prepared.payload["max_output_tokens"], 6000)
        self.assertIs(prepared.payload["store"], False)
        self.assertIn("input", prepared.payload)

    def test_gemini_uses_native_generate_content_and_thinking_level(self) -> None:
        config = Config(
            provider="gemini",
            model="gemini-3.7-flash",
            reasoning_effort="medium",
            max_output_tokens=4000,
            temperature=0.5,
            top_p=0.9,
            top_k=30,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            seed=4,
            stop_sequences=["DONE"],
        )
        prepared = make_engine(config, "key").prepare(MESSAGES)
        self.assertEqual(
            prepared.url,
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-3.7-flash:generateContent",
        )
        generation = prepared.payload["generationConfig"]
        self.assertEqual(generation["thinkingConfig"], {"thinkingLevel": "MEDIUM"})
        self.assertEqual(generation["maxOutputTokens"], 4000)
        self.assertEqual(generation["topP"], 0.9)
        self.assertEqual(generation["topK"], 30)
        self.assertEqual(generation["frequencyPenalty"], 0.2)
        self.assertEqual(generation["presencePenalty"], 0.3)
        self.assertEqual(generation["stopSequences"], ["DONE"])
        self.assertIn("systemInstruction", prepared.payload)

    def test_gemini_25_uses_exact_budget_instead_of_fake_effort_mapping(self) -> None:
        config = Config(
            provider="gemini",
            model="gemini-2.5-flash",
            reasoning_budget_tokens=2048,
        )
        body = make_engine(config, "key").prepare(MESSAGES).payload
        self.assertEqual(
            body["generationConfig"]["thinkingConfig"], {"thinkingBudget": 2048}
        )

    def test_anthropic_uses_effort_output_config_and_thinking_budget(self) -> None:
        config = Config(
            provider="anthropic",
            model="claude-opus-4-5",
            reasoning_effort="high",
            reasoning_budget_tokens=2048,
            max_output_tokens=8192,
            temperature=1.0,
            top_p=0.99,
            top_k=5,
            stop_sequences=["STOP"],
        )
        prepared = make_engine(config, "key").prepare(MESSAGES)
        self.assertEqual(prepared.url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(prepared.payload["output_config"], {"effort": "high"})
        self.assertEqual(
            prepared.payload["thinking"],
            {"type": "enabled", "budget_tokens": 2048},
        )
        self.assertEqual(prepared.payload["stop_sequences"], ["STOP"])

    def test_custom_endpoint_is_explicitly_server_defined(self) -> None:
        config = Config(
            provider="custom",
            model="local",
            base_url="https://custom.example/v1",
            reasoning_effort="low",
            top_k=10,
        )
        prepared = make_engine(config, "key").prepare(MESSAGES)
        self.assertEqual(prepared.url, "https://custom.example/v1/chat/completions")
        self.assertEqual(prepared.payload["reasoning_effort"], "low")
        self.assertEqual(prepared.payload["top_k"], 10)
        self.assertIn("defined by the custom server", prepared.guarantee)

    def test_advanced_custom_json_maps_typed_request_and_response(self) -> None:
        contract = {
            "url": "https://engine.example/v2/generate",
            "headers": {"X-API-Version": "2026-08-29"},
            "auth": {"header": "X-API-Key", "prefix": "", "required": True},
            "body": {
                "model_name": "$artificium.model",
                "conversation": "$artificium.messages",
                "plain_prompt": "$artificium.prompt",
                "sampling": {
                    "temperature": "$artificium.temperature",
                    "max_new_tokens": "$artificium.max_output_tokens",
                    "top_p": "$artificium.top_p",
                },
            },
            "response": {
                "content": ["/missing", "/result/segments/0/text"],
                "reasoning": "/result/reasoning",
                "usage": "/usage",
                "finish_reason": "/result/status",
            },
        }
        config = Config(
            provider="custom",
            model="engine-model",
            base_url="https://engine.example",
            endpoint="v2/generate",
            adapter="custom_json",
            custom_contract=contract,
            temperature=0.35,
            max_output_tokens=2048,
        )
        engine = make_engine(config, "custom-secret")
        prepared = engine.prepare(MESSAGES)
        self.assertEqual(prepared.url, contract["url"])
        self.assertEqual(prepared.headers["X-API-Key"], "custom-secret")
        self.assertEqual(prepared.payload["model_name"], "engine-model")
        self.assertEqual(prepared.payload["conversation"], MESSAGES)
        self.assertEqual(prepared.payload["sampling"]["temperature"], 0.35)
        self.assertEqual(prepared.payload["sampling"]["max_new_tokens"], 2048)
        self.assertNotIn("top_p", prepared.payload["sampling"])
        self.assertNotIn("custom-secret", json.dumps(prepared.safe_summary()))

        captured: dict = {}

        def fake_open(request, timeout=0):
            captured.update(json.loads(request.data.decode()))
            self.assertEqual(request.get_header("X-api-key"), "custom-secret")
            return DummyResponse(
                {
                    "result": {
                        "segments": [{"text": "OK"}],
                        "reasoning": "provider reasoning",
                        "status": "complete",
                    },
                    "usage": {"input": 12, "output": 2},
                }
            )

        with mock.patch("urllib.request.urlopen", side_effect=fake_open):
            reply = engine.complete(MESSAGES)
        self.assertEqual(captured, prepared.payload)
        self.assertEqual(reply.content, "OK")
        self.assertEqual(reply.provider_reasoning, "provider reasoning")
        self.assertEqual(reply.usage, {"input": 12, "output": 2})
        self.assertEqual(reply.finish_reason, "complete")

    def test_advanced_custom_json_rejects_unknown_placeholders(self) -> None:
        config = Config(
            provider="custom",
            model="model",
            base_url="https://engine.example",
            endpoint="generate",
            adapter="custom_json",
            custom_contract={
                "url": "https://engine.example/generate",
                "body": {"input": "$artificium.not_a_real_field"},
                "response": {"content": "/text"},
            },
        )
        with self.assertRaisesRegex(EngineError, "Unknown custom-contract placeholder"):
            make_engine(config, None).prepare(MESSAGES)

    def test_custom_contract_and_adapter_must_be_paired(self) -> None:
        contract = {
            "url": "https://engine.example/generate",
            "body": {"prompt": "$artificium.prompt"},
            "response": {"content": "/text"},
        }
        with self.assertRaisesRegex(ValueError, "requires the custom_json adapter"):
            Config(
                provider="custom",
                model="model",
                base_url="https://engine.example",
                custom_contract=contract,
            )

    def test_unsupported_field_is_rejected_instead_of_silently_ignored(self) -> None:
        config = Config(provider="openai", model="gpt-5.6", seed=42)
        with self.assertRaisesRegex(EngineError, "seed"):
            make_engine(config, "key").prepare(MESSAGES)

    def test_legacy_ollama_config_migrates_to_native_contract(self) -> None:
        config = Config.from_dict(
            {
                "schema_version": 1,
                "provider": "ollama",
                "model": "qwen3:8b",
                "base_url": "https://pod.example/v1",
                "endpoint": "chat/completions",
                "adapter": "openai_compatible",
            }
        )
        self.assertEqual(config.schema_version, 3)
        self.assertEqual(config.base_url, "https://pod.example")
        self.assertEqual(config.endpoint, "api/chat")
        self.assertEqual(config.adapter, "ollama")

    def test_openrouter_metadata_rejects_an_unsupported_effort(self) -> None:
        discovery = ModelDiscovery(
            models=("model",),
            details={
                "model": {
                    "context_length": 64_000,
                    "reasoning": {"supported_efforts": ["low", "medium", "high"]},
                }
            },
        )
        with self.assertRaisesRegex(ValueError, "not xhigh"):
            validate_discovered_settings(
                "openrouter",
                discovery,
                "model",
                context_window_tokens=64_000,
                max_output_tokens=None,
                reasoning_effort="xhigh",
                reasoning_budget_tokens=None,
            )

    def test_discovered_context_limit_rejects_unsafe_working_capacity(self) -> None:
        discovery = ModelDiscovery(
            models=("model",),
            details={"model": {"context_length": 50_000}},
        )
        with self.assertRaisesRegex(ValueError, "100,000.*50,000"):
            validate_discovered_settings(
                "openrouter",
                discovery,
                "model",
                context_window_tokens=100_000,
                max_output_tokens=None,
                reasoning_effort=None,
                reasoning_budget_tokens=None,
            )

    def test_api_key_file_accepts_artificium_secret_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".secrets.json"
            path.write_text(json.dumps({"api_key": "actual-key", "provider": "openrouter"}))
            self.assertEqual(load_api_key_file(path), "actual-key")

    def test_provider_error_cannot_copy_credentials_into_logs(self) -> None:
        config = Config(
            provider="custom",
            model="model",
            base_url="https://custom.example/v1",
            headers={"X-Private-Token": "header-secret"},
        )
        engine = make_engine(config, "api-secret")
        error = urllib.error.HTTPError(
            "https://custom.example/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b"echo api-secret and header-secret"),
        )
        with (
            mock.patch("urllib.request.urlopen", side_effect=error),
            self.assertRaises(EngineError) as raised,
        ):
            engine.complete([{"role": "user", "content": "hello"}])
        rendered = str(raised.exception)
        self.assertNotIn("api-secret", rendered)
        self.assertNotIn("header-secret", rendered)
        self.assertIn("<redacted>", rendered)


if __name__ == "__main__":
    unittest.main()
