"""Streaming stall (inactivity) and first-token watchdogs, and the automatic
non-streaming fallback when a server does not speak SSE for chat completions.

Uses a small local SSE fixture (a real loopback HTTP server on a thread), in
the same spirit as SSEServer in test_engine_cache.py, so timing is exercised
against a genuine socket rather than mocked delays.
"""
from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from artificium.config import Config
from artificium.engine import EngineError
from artificium.engine_adapters import OpenAICompatibleEngine


class ControllableSSEServer:
    """A chat-completions endpoint whose streaming/rejection behavior a test
    configures per attribute before each request."""

    __test__ = False

    def __init__(self) -> None:
        # Chunks (and matching per-chunk delays, in seconds) sent as SSE
        # `data: ...` lines when a request asks to stream.
        self.chunks: list[dict] = []
        self.delays: list[float] = []
        # After the configured chunks, hang instead of sending `data: [DONE]`.
        self.hang_seconds: float | None = None
        # Reject a streaming request outright (simulates a server that does
        # not implement `stream: true`).
        self.reject_stream = False
        # Answer a streaming request with a plain JSON body (wrong content
        # type) instead of SSE at all.
        self.non_sse_body = False
        # The body every non-streaming request receives (the initial
        # attempt when streaming is off, and any post-fallback retry).
        self.plain_response = {
            "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        }
        self.requests: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_a):
                pass

            def _send_json(self, status: int, value: dict) -> None:
                raw = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except OSError:
                    pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                owner.requests.append(body)
                if not body.get("stream"):
                    self._send_json(200, owner.plain_response)
                    return
                if owner.reject_stream:
                    self._send_json(400, {"error": {
                        "message": "Unknown parameter: 'stream' is not supported by this endpoint",
                    }})
                    return
                if owner.non_sse_body:
                    self._send_json(200, owner.plain_response)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                delays = owner.delays or [0] * len(owner.chunks)
                try:
                    for chunk, delay in zip(owner.chunks, delays):
                        if delay:
                            time.sleep(delay)
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    if owner.hang_seconds is not None:
                        time.sleep(owner.hang_seconds)
                        return
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    return  # the client gave up; nothing left to report

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.http.server_port}"

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()


class StallAndFirstTokenTimeoutCase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ControllableSSEServer()
        self.addCleanup(self.server.close)

    def engine(self, **overrides) -> OpenAICompatibleEngine:
        config = Config(
            provider="custom", adapter="openai_compatible", base_url=self.server.url,
            endpoint="chat/completions", model="m", stream_responses=True,
            **overrides,
        )
        return OpenAICompatibleEngine(config, None)

    # (a) Streams normally, including a tool call split across chunk
    # boundaries: the reassembled content must match a non-streamed reply
    # byte for byte, regardless of where the SSE frames happened to split it.
    def test_normal_stream_reassembles_content_usage_and_a_split_tool_call(self) -> None:
        tool_call = '<tool_call>{"tool": "write_file", "path": "x.txt", "content": "hi"}</tool_call>'
        split = len(tool_call) // 2
        self.server.chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": tool_call[:split]}}]},
            {"choices": [{"index": 0, "delta": {"content": tool_call[split:]}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}},
        ]
        engine = self.engine(request_timeout_seconds=5, stall_timeout_seconds=2)
        reply = engine.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.content, tool_call)
        self.assertEqual(reply.finish_reason, "stop")
        self.assertEqual(reply.usage, {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15})
        self.assertTrue(self.server.requests[0]["stream"])

    # (b) A server that goes silent mid-stream (after at least one chunk) is
    # caught by the stall watchdog quickly, well before the sleep it is
    # hanging on and well before the (much larger) total request timeout.
    def test_mid_stream_stall_aborts_quickly_with_a_clear_message(self) -> None:
        self.server.chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"}}]},
        ]
        self.server.hang_seconds = 1.5
        engine = self.engine(request_timeout_seconds=10, stall_timeout_seconds=0.15)
        started = time.monotonic()
        with self.assertRaises(EngineError) as caught:
            engine.complete([{"role": "user", "content": "hi"}])
        elapsed = time.monotonic() - started
        self.assertEqual(caught.exception.kind, "timeout")
        self.assertIn("stalled", str(caught.exception))
        self.assertNotIn("during prefill", str(caught.exception))
        self.assertLess(elapsed, 1.0)

    # (c) A long silent prefill, with no first-token timeout configured, is
    # not mistaken for a stall even though it is longer than the (separate,
    # tighter) inactivity budget that applies once tokens start arriving.
    def test_long_silent_prefill_then_streaming_is_not_a_stall(self) -> None:
        self.server.chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}}]},
            {"choices": [{"index": 0, "delta": {"content": " there"}, "finish_reason": "stop"}]},
        ]
        # First chunk delayed well past stall_timeout_seconds; only bounded
        # by first_token_timeout_seconds (None here) and the total timeout.
        self.server.delays = [0.6, 0]
        engine = self.engine(
            request_timeout_seconds=5, stall_timeout_seconds=0.2, first_token_timeout_seconds=None,
        )
        reply = engine.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.content, "Hello there")

    # The overall request timeout is a genuine wall-clock cap even when every
    # individual gap between chunks stays comfortably under the (larger)
    # stall timeout: a server that trickles data slowly enough to keep
    # resetting the inactivity clock must not be able to stream forever.
    def test_total_timeout_still_caps_a_slow_trickle_that_never_stalls(self) -> None:
        self.server.chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "x"}}]}
            for _ in range(20)
        ]
        self.server.delays = [0.25] * 20  # 5s of total trickle if left alone
        engine = self.engine(request_timeout_seconds=1, stall_timeout_seconds=5)
        started = time.monotonic()
        with self.assertRaises(EngineError) as caught:
            engine.complete([{"role": "user", "content": "hi"}])
        elapsed = time.monotonic() - started
        self.assertEqual(caught.exception.kind, "timeout")
        self.assertNotIn("stalled", str(caught.exception))
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertLess(elapsed, 2.0)

    # A configured first-token timeout, unlike the default None, does bound
    # prefill silence -- distinctly from the post-first-chunk stall message.
    def test_first_token_timeout_bounds_prefill_when_configured(self) -> None:
        self.server.chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "late"}}]},
        ]
        self.server.delays = [1.5]
        engine = self.engine(
            request_timeout_seconds=10, stall_timeout_seconds=5, first_token_timeout_seconds=0.15,
        )
        started = time.monotonic()
        with self.assertRaises(EngineError) as caught:
            engine.complete([{"role": "user", "content": "hi"}])
        elapsed = time.monotonic() - started
        self.assertEqual(caught.exception.kind, "timeout")
        self.assertIn("during prefill", str(caught.exception))
        self.assertLess(elapsed, 1.0)

    # (d) A server that rejects `stream: true` outright falls back to a
    # single non-streaming retry, and the engine remembers that for the rest
    # of the session (a later call skips straight to non-streaming).
    def test_stream_rejection_falls_back_once_and_is_remembered(self) -> None:
        self.server.reject_stream = True
        engine = self.engine(request_timeout_seconds=5)
        self.assertFalse(engine._stream_unsupported)
        reply = engine.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.content, "OK")
        self.assertTrue(engine._stream_unsupported)
        self.assertEqual(len(self.server.requests), 2)
        self.assertTrue(self.server.requests[0]["stream"])
        self.assertFalse(self.server.requests[1]["stream"])

        reply = engine.complete([{"role": "user", "content": "hi again"}])
        self.assertEqual(reply.content, "OK")
        self.assertEqual(len(self.server.requests), 3)
        self.assertFalse(self.server.requests[2]["stream"])

    # A 200 response that is not actually SSE (wrong content type) is the
    # same kind of "server doesn't really stream" signal as an outright
    # rejection, and falls back the same way.
    def test_non_sse_success_body_also_falls_back(self) -> None:
        self.server.non_sse_body = True
        engine = self.engine(request_timeout_seconds=5)
        reply = engine.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.content, "OK")
        self.assertTrue(engine._stream_unsupported)
        self.assertEqual(len(self.server.requests), 2)
        self.assertTrue(self.server.requests[0]["stream"])
        self.assertFalse(self.server.requests[1]["stream"])


if __name__ == "__main__":
    unittest.main()
