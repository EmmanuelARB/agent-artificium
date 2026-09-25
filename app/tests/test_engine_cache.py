"""Tests for B2 (Anthropic prompt caching), B5 (prompt cache key), B9
(Anthropic max_tokens fallback), and B12 (opt-in chat-completions streaming).
"""
from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from artificium.config import Config
from artificium.engine import (
    AnthropicEngine,
    EngineError,
    LlamaCppEngine,
    OpenAICompatibleEngine,
    OpenAIResponsesEngine,
    OpenRouterEngine,
    VLLMEngine,
)


MESSAGES = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "hello"},
]


def _history_and_pending(*, with_pending: bool = True, with_visual: bool = False) -> list[dict]:
    """A realistic runtime-shaped message list: system, persisted alternating
    history (itself tagged runtime_input, as `_append_runtime` persists it),
    optionally a pending runtime-input message, optionally a transient visual
    message, and always a transient trailing state-header message."""
    messages = [
        {"role": "system", "content": "SYS PROMPT"},
        {"role": "assistant", "content": "hist-assistant-1"},
        {"role": "user", "content": "hist-user-1", "_artificium": {"kind": "runtime_input"}},
        {"role": "assistant", "content": "hist-assistant-2"},
    ]
    if with_pending:
        messages.append(
            {"role": "user", "content": "pending-input", "_artificium": {"kind": "runtime_input"}}
        )
    if with_visual:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "visual context"}],
                "_artificium": {"kind": "visual_context"},
            }
        )
    messages.append(
        {"role": "user", "content": "STATE HEADER", "_artificium": {"kind": "state_header"}}
    )
    return messages


def _cache_flags(payload_messages: list[dict]) -> list[tuple[str, bool]]:
    """(text, has_cache_control) for every content block, message by message,
    flattened for easy assertions."""
    flags = []
    for message in payload_messages:
        for block in message["content"]:
            flags.append((block.get("text"), "cache_control" in block))
    return flags


class AnthropicCacheBreakpointCase(unittest.TestCase):
    def test_breakpoints_land_on_system_persisted_boundary_and_pending_input(self) -> None:
        config = Config(provider="anthropic", model="claude-opus-4-5")
        prepared = AnthropicEngine(config, "key").prepare(_history_and_pending())
        system = prepared.payload["system"]
        self.assertEqual(system, [{"type": "text", "text": "SYS PROMPT",
                                    "cache_control": {"type": "ephemeral"}}])
        flags = _cache_flags(prepared.payload["messages"])
        self.assertEqual(
            flags,
            [
                ("hist-assistant-1", False),
                ("hist-user-1", False),
                ("hist-assistant-2", True),   # persisted-history/pending-input boundary
                ("pending-input", True),      # last stable block before transient tail
                ("STATE HEADER", False),      # transient: never cached
            ],
        )

    def test_visual_message_is_also_transient_and_never_cached(self) -> None:
        config = Config(provider="anthropic", model="claude-opus-4-5")
        prepared = AnthropicEngine(config, "key").prepare(
            _history_and_pending(with_visual=True)
        )
        flags = _cache_flags(prepared.payload["messages"])
        self.assertEqual(
            flags,
            [
                ("hist-assistant-1", False),
                ("hist-user-1", False),
                ("hist-assistant-2", True),
                ("pending-input", True),
                ("visual context", False),
                ("STATE HEADER", False),
            ],
        )

    def test_no_pending_input_still_caches_end_of_persisted_history(self) -> None:
        config = Config(provider="anthropic", model="claude-opus-4-5")
        prepared = AnthropicEngine(config, "key").prepare(
            _history_and_pending(with_pending=False)
        )
        flags = _cache_flags(prepared.payload["messages"])
        self.assertEqual(
            flags,
            [
                ("hist-assistant-1", False),
                ("hist-user-1", False),
                # hist-assistant-2 is the last stable block before the transient
                # tail; with no pending input there is only this one breakpoint.
                ("hist-assistant-2", True),
                ("STATE HEADER", False),
            ],
        )

    def test_consecutive_same_role_messages_are_merged_and_breakpoint_tracks_the_right_block(self) -> None:
        # Two adjacent user messages (persisted history ending on a user turn,
        # immediately followed by the pending user runtime-input message) must
        # merge into one Anthropic message while still placing cache_control on
        # the correct sub-block, not the merged message as a whole.
        config = Config(provider="anthropic", model="claude-opus-4-5")
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u1", "_artificium": {"kind": "runtime_input"}},
            {"role": "user", "content": "u2-pending", "_artificium": {"kind": "runtime_input"}},
            {"role": "user", "content": "STATE HEADER", "_artificium": {"kind": "state_header"}},
        ]
        prepared = AnthropicEngine(config, "key").prepare(messages)
        merged = prepared.payload["messages"]
        # u1, u2-pending and STATE HEADER all merge into a single user message.
        self.assertEqual([m["role"] for m in merged], ["assistant", "user"])
        blocks = merged[1]["content"]
        self.assertEqual([b["text"] for b in blocks], ["u1", "u2-pending", "STATE HEADER"])
        # u1 is the persisted-history/pending-input boundary; u2-pending is the
        # last stable block before the transient tail. Both land correctly
        # even though merging put all three text blocks in one message.
        self.assertIn("cache_control", blocks[0])
        self.assertIn("cache_control", blocks[1])
        self.assertNotIn("cache_control", blocks[2])

    def test_prompt_cache_false_reproduces_the_exact_old_payload_shape(self) -> None:
        config_cached = Config(provider="anthropic", model="claude-opus-4-5")
        config_plain = Config(provider="anthropic", model="claude-opus-4-5", prompt_cache=False)
        messages = _history_and_pending()
        cached = AnthropicEngine(config_cached, "key").prepare(messages)
        plain = AnthropicEngine(config_plain, "key").prepare(messages)
        self.assertIsInstance(plain.payload["system"], str)
        self.assertEqual(plain.payload["system"], "SYS PROMPT")
        for message in plain.payload["messages"]:
            for block in message["content"]:
                self.assertNotIn("cache_control", block)
        # The cached run differs only by added cache_control markers, not content.
        strip = lambda blocks: [{k: v for k, v in b.items() if k != "cache_control"} for b in blocks]
        for cached_message, plain_message in zip(cached.payload["messages"], plain.payload["messages"]):
            self.assertEqual(strip(cached_message["content"]), plain_message["content"])

    def test_count_tokens_payload_carries_cache_control_and_stays_shaped_for_the_endpoint(self) -> None:
        config = Config(provider="anthropic", model="claude-opus-4-5")
        engine = AnthropicEngine(config, "key")
        prepared = engine.prepare(_history_and_pending())
        captured = {}

        def fake_open(request, timeout=0):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode())
            return _JSONResponse({"input_tokens": 42})

        with mock.patch("urllib.request.urlopen", side_effect=fake_open):
            count = engine.count_input_tokens(prepared)
        self.assertEqual(count, 42)
        self.assertTrue(captured["url"].endswith("/count_tokens"))
        self.assertEqual(captured["body"]["system"], prepared.payload["system"])
        self.assertEqual(captured["body"]["messages"], prepared.payload["messages"])


class AnthropicMaxTokensFallbackCase(unittest.TestCase):
    def test_unknown_model_limit_falls_back_to_capped_32000(self) -> None:
        config = Config(provider="anthropic", model="m", context_window_tokens=200_000)
        prepared = AnthropicEngine(config, "key").prepare(MESSAGES)
        self.assertEqual(prepared.payload["max_tokens"], 32_000)

    def test_small_context_window_caps_fallback_below_32000(self) -> None:
        config = Config(provider="anthropic", model="m", context_window_tokens=10_000)
        prepared = AnthropicEngine(config, "key").prepare(MESSAGES)
        self.assertEqual(prepared.payload["max_tokens"], 10_000)

    def test_known_model_capability_is_still_preferred_over_the_cap(self) -> None:
        config = Config(
            provider="anthropic", model="m", context_window_tokens=200_000,
            model_capabilities={"max_output_tokens": 4096},
        )
        prepared = AnthropicEngine(config, "key").prepare(MESSAGES)
        self.assertEqual(prepared.payload["max_tokens"], 4096)

    def test_explicit_max_output_tokens_still_wins_over_any_fallback(self) -> None:
        config = Config(
            provider="anthropic", model="m", context_window_tokens=200_000, max_output_tokens=555,
        )
        prepared = AnthropicEngine(config, "key").prepare(MESSAGES)
        self.assertEqual(prepared.payload["max_tokens"], 555)

    def test_thinking_budget_validation_still_uses_the_capped_fallback(self) -> None:
        config = Config(
            provider="anthropic", model="m", context_window_tokens=10_000,
            reasoning_budget_tokens=10_000,
        )
        with self.assertRaisesRegex(EngineError, "greater than the thinking token budget"):
            AnthropicEngine(config, "key").prepare(MESSAGES)
        ok = Config(
            provider="anthropic", model="m", context_window_tokens=10_000,
            reasoning_budget_tokens=9_999,
        )
        prepared = AnthropicEngine(ok, "key").prepare(MESSAGES)
        self.assertEqual(prepared.payload["thinking"], {"type": "enabled", "budget_tokens": 9_999})


class PromptCacheKeyCase(unittest.TestCase):
    def test_auto_derives_from_instance_id_for_openai_compatible_and_responses(self) -> None:
        compatible = Config(
            provider="custom", adapter="openai_compatible", base_url="https://x", model="m",
            prompt_cache_key="auto", instance_id="myinst",
        )
        prepared = OpenAICompatibleEngine(compatible, "key").prepare(MESSAGES)
        self.assertEqual(prepared.payload["prompt_cache_key"], "artificium-myinst")

        responses = Config(provider="openai", model="m", prompt_cache_key="auto", instance_id="myinst")
        prepared = OpenAIResponsesEngine(responses, "key").prepare(MESSAGES)
        self.assertEqual(prepared.payload["prompt_cache_key"], "artificium-myinst")

    def test_explicit_string_is_used_verbatim(self) -> None:
        config = Config(
            provider="custom", adapter="openai_compatible", base_url="https://x", model="m",
            prompt_cache_key="fixed-key",
        )
        prepared = OpenAICompatibleEngine(config, "key").prepare(MESSAGES)
        self.assertEqual(prepared.payload["prompt_cache_key"], "fixed-key")

    def test_unset_key_is_absent_from_the_payload(self) -> None:
        config = Config(provider="custom", adapter="openai_compatible", base_url="https://x", model="m")
        prepared = OpenAICompatibleEngine(config, "key").prepare(MESSAGES)
        self.assertNotIn("prompt_cache_key", prepared.payload)

    def test_adapters_outside_the_spec_never_receive_a_cache_key(self) -> None:
        for cls, kwargs in (
            (VLLMEngine, dict(provider="vllm", model="m")),
            (OpenRouterEngine, dict(provider="openrouter", model="m")),
            (LlamaCppEngine, dict(provider="llamacpp", model="m")),
        ):
            config = Config(prompt_cache_key="auto", instance_id="myinst", **kwargs)
            prepared = cls(config, "key").prepare(MESSAGES)
            self.assertNotIn("prompt_cache_key", prepared.payload)


class StreamFlagCase(unittest.TestCase):
    def test_flag_off_reproduces_the_exact_old_non_streaming_payload(self) -> None:
        for cls, kwargs in (
            (OpenAICompatibleEngine, dict(provider="custom", adapter="openai_compatible", base_url="https://x", model="m")),
            (LlamaCppEngine, dict(provider="llamacpp", model="m")),
            (VLLMEngine, dict(provider="vllm", model="m")),
            (OpenRouterEngine, dict(provider="openrouter", model="m")),
        ):
            engine = cls(Config(**kwargs), "key")
            prepared = engine.prepare(MESSAGES)
            self.assertIs(prepared.payload["stream"], False)
            self.assertNotIn("stream_options", prepared.payload)

    def test_flag_on_adds_stream_and_include_usage(self) -> None:
        for cls, kwargs in (
            (OpenAICompatibleEngine, dict(provider="custom", adapter="openai_compatible", base_url="https://x", model="m")),
            (LlamaCppEngine, dict(provider="llamacpp", model="m")),
            (VLLMEngine, dict(provider="vllm", model="m")),
            (OpenRouterEngine, dict(provider="openrouter", model="m")),
        ):
            engine = cls(Config(stream_responses=True, **kwargs), "key")
            prepared = engine.prepare(MESSAGES)
            self.assertIs(prepared.payload["stream"], True)
            self.assertEqual(prepared.payload["stream_options"], {"include_usage": True})


class _JSONResponse:
    def __init__(self, value: dict):
        self.value = value

    def __enter__(self) -> "_JSONResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode()


class SSEServer:
    """A minimal local SSE fixture for chat-completions streaming, mirroring
    the shape TestServer in test_connection_flow.py uses for JSON servers."""

    __test__ = False

    def __init__(self) -> None:
        self.status = 200
        self.chunks: list[dict] = []
        self.delays: list[float] = []
        self.error_body: dict | None = None
        self.requests: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_a):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                owner.requests.append(body)
                if owner.status != 200:
                    raw = json.dumps(owner.error_body or {"error": {"message": "failed"}}).encode()
                    self.send_response(owner.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                delays = owner.delays or [0] * len(owner.chunks)
                for chunk, delay in zip(owner.chunks, delays):
                    if delay:
                        time.sleep(delay)
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.http.server_port}"

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()


class StreamingTransportCase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = SSEServer()
        self.addCleanup(self.server.close)

    def engine(self, **overrides) -> OpenAICompatibleEngine:
        config = Config(
            provider="custom", adapter="openai_compatible", base_url=self.server.url,
            endpoint="chat/completions", model="m", stream_responses=True,
            request_timeout_seconds=10, **overrides,
        )
        return OpenAICompatibleEngine(config, None)

    def test_content_deltas_and_final_usage_chunk_reassemble_like_a_plain_reply(self) -> None:
        self.server.chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
            {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        ]
        engine = self.engine()
        reply = engine.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.content, "Hello")
        self.assertEqual(reply.finish_reason, "stop")
        self.assertEqual(reply.usage, {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
        self.assertTrue(self.server.requests[0]["stream"])
        self.assertEqual(self.server.requests[0]["stream_options"], {"include_usage": True})

    def test_reasoning_deltas_are_assembled_into_provider_reasoning(self) -> None:
        self.server.chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "Let me "}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "think. "}}]},
            {"choices": [{"index": 0, "delta": {"content": "Answer"}, "finish_reason": "stop"}]},
        ]
        engine = self.engine()
        reply = engine.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.content, "Answer")
        self.assertEqual(reply.provider_reasoning, "Let me think. ")

    def test_llamacpp_timings_in_final_chunk_reach_the_raw_reply(self) -> None:
        self.server.chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]},
            {"choices": [], "timings": {"prompt_n": 10, "cache_n": 4, "predicted_n": 3}},
        ]
        config = Config(
            provider="llamacpp", base_url=self.server.url, model="m",
            stream_responses=True, request_timeout_seconds=10,
        )
        engine = LlamaCppEngine(config, None)
        reply = engine.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.content, "OK")
        self.assertEqual(reply.raw["timings"], {"prompt_n": 10, "cache_n": 4, "predicted_n": 3})

    def test_http_error_response_is_not_treated_as_a_stream(self) -> None:
        self.server.status = 400
        self.server.error_body = {"error": {"message": "Unsupported temperature setting"}}
        engine = self.engine()
        with self.assertRaises(EngineError) as caught:
            engine.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(caught.exception.kind, "settings")
        self.assertIn("Unsupported temperature setting", str(caught.exception))

    def test_progress_callback_fires_at_most_every_two_seconds_with_the_documented_shape(self) -> None:
        self.server.chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}]},
            {"choices": [{"index": 0, "delta": {"content": " there"}, "finish_reason": "stop"}]},
        ]
        # One real gap just over the 2-second throttle window, before the
        # second (final) chunk, so exactly one progress call is expected.
        self.server.delays = [0, 2.15]
        engine = self.engine()
        calls: list[dict] = []
        engine.progress_callback = calls.append
        reply = engine.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.content, "Hi there")
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(set(call), {"generated_chars", "reasoning_chars", "elapsed_seconds"})
        # The callback is checked once per processed chunk; by the time the
        # delayed second (final) chunk has arrived and crossed the 2-second
        # throttle window, its content is already folded into the tally.
        self.assertEqual(call["generated_chars"], len("Hi there"))
        self.assertEqual(call["reasoning_chars"], 0)
        self.assertGreaterEqual(call["elapsed_seconds"], 2.0)

    def test_progress_callback_exceptions_are_swallowed(self) -> None:
        self.server.chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}]},
            {"choices": [{"index": 0, "delta": {"content": " there"}, "finish_reason": "stop"}]},
        ]
        self.server.delays = [0, 2.1]
        engine = self.engine()

        def boom(_progress):
            raise RuntimeError("progress hook is broken")

        engine.progress_callback = boom
        reply = engine.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.content, "Hi there")

    def test_non_streaming_behaviour_is_unchanged_when_the_flag_is_false(self) -> None:
        config = Config(
            provider="custom", adapter="openai_compatible", base_url=self.server.url,
            endpoint="chat/completions", model="m", request_timeout_seconds=10,
        )
        engine = OpenAICompatibleEngine(config, None)
        # A non-SSE JSON server would normally answer this request; confirm the
        # prepared payload never asks for a stream when the flag is off.
        prepared = engine.prepare([{"role": "user", "content": "hi"}])
        self.assertIs(prepared.payload["stream"], False)


if __name__ == "__main__":
    unittest.main()
