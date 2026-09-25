from __future__ import annotations

import json
import io
import os
import re
import shutil
import signal
import tempfile
import threading
import time
import unittest
import datetime as dt
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from artificium.config import Config, ConfigStore, SecretsStore
from artificium.cli import (
    _can_setup_noninteractive,
    _chat_input_rows,
    _chat_input_prompt,
    _clear_chat_input,
    _print_life_record,
    _restore_chat_input,
    _stop_background,
    _watch_life_loop,
    main,
)
from artificium.engine import (
    AnthropicEngine,
    Engine,
    EngineError,
    EngineReply,
    OpenAICompatibleEngine,
    make_engine,
)
from artificium.filesystem import Paths, atomic_write_json, read_json
from artificium.initialization import Initialization, initialize_mind
from artificium.interactions import ArtificiumClient, InteractionStore, NotificationStore
from artificium.life_loop import (
    ToolIntent,
    parse_life_loop_output,
    render_normalized_life_loop_output,
)
from artificium.memory import InfiniteAttention, LongTermMemory, TokenEstimator, WorkingMemory
from artificium.prompts import PromptPack
from artificium.records import Console, Records
from artificium.runtime import Artificium
from artificium.setup import (
    ModelDiscovery,
    SetupOptions,
    SetupWizard,
    configure_generation_interactive,
    discover_models,
    discover_provider_models,
    normalize_openai_endpoint,
    normalize_openrouter_provider,
    normalize_provider_endpoint,
    route_openrouter_requests,
    supported_generation_controls,
)
from artificium.tool_loader import load_mind_tool
from artificium.tools import ToolRegistry
from artificium.vision import VisualContext


PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
REFERENCE_SEED = Path(__file__).resolve().parents[1] / "seed"


def _seed_install(paths: Paths) -> None:
    """Populate a fresh install the way a real one arrives: prompts and the
    shipped seed sitting under app/, then let ensure_layout() grow the
    workspace and fill the mind from that seed."""

    shutil.copytree(PROMPTS, paths.prompts)
    shutil.copytree(REFERENCE_SEED, paths.seed)
    paths.ensure_layout()


class FakeEngine(Engine):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.requests: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> EngineReply:
        self.requests.append(messages)
        content = self.responses.pop(0) if self.responses else "<think>No action.</think>"
        return EngineReply(content, usage={"input_tokens": 10, "output_tokens": 5})


class AttentionEngine(Engine):
    def __init__(self, source: Path):
        self.source = source
        self.step = 0
        self.stream_id = ""

    def complete(self, messages: list[dict]) -> EngineReply:
        encoded = json.dumps(messages)
        if not self.stream_id:
            match = re.search(r"stream_\d{8}T\d+Z_[a-f0-9]+", encoded)
            if match:
                self.stream_id = match.group(0)
        if self.step == 0:
            call = {
                "name": "open_attention",
                "arguments": {
                    "source": str(self.source),
                    "objective": "Inspect every byte and report completion.",
                    "granularity": "fine",
                    "chunk_tokens": 1_000,
                },
            }
        elif self.step == 1:
            call = {
                "name": "checkpoint_attention",
                "arguments": {
                    "stream_id": self.stream_id,
                    "chunk_number": 1,
                    "compressed_carry": "First source range inspected; continue exactly once.",
                    "decision": "continue",
                },
            }
        elif self.step == 2:
            call = {
                "name": "next_attention_chunk",
                "arguments": {"stream_id": self.stream_id},
            }
        elif self.step == 3:
            call = {
                "name": "checkpoint_attention",
                "arguments": {
                    "stream_id": self.stream_id,
                    "chunk_number": 2,
                    "compressed_carry": "Both source ranges were inspected and the source is exhausted.",
                    "decision": "complete",
                    "result": "complete",
                },
            }
        elif self.step == 4:
            call = {
                "name": "sleep",
                "arguments": {"mode": "until_event"},
            }
        else:
            call = {
                "name": "sleep",
                "arguments": {"mode": "until_event", "reflection_complete": True},
            }
        self.step += 1
        return EngineReply(
            "<think>Advance the deterministic attention test.</think>"
            f"<tool_call>{json.dumps(call)}</tool_call>"
        )


class DummyHTTPResponse:
    def __init__(self, value: dict):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode()


class RevolutionCase(unittest.TestCase):
    def setUp(self) -> None:
        # These unit cases isolate metadata/core behavior. Real connection
        # verification and failures are covered over HTTP in test_connection_flow.
        check = mock.patch('artificium.setup.verify_connection', side_effect=lambda paths, config, key, **kw: (config, {}))
        check.start()
        self.addCleanup(check.stop)
        network = mock.patch("urllib.request.urlopen", side_effect=OSError("offline test"))
        network.start()
        self.addCleanup(network.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = Paths(self.root)
        _seed_install(self.paths)
        self.prompts = PromptPack(self.paths)
        self.config = Config(
            provider="custom",
            model="test-model",
            base_url="http://example.invalid/v1",
            context_window_tokens=20_000,
            context_reminder_tokens=2_000,
            max_direct_read_chars=1_000,
        )
        ConfigStore(self.paths).save(self.config)
        SecretsStore(self.paths).save_api_key("test-key")
        self.records = Records(self.paths)
        initialize_mind(self.paths, self.records)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def components(self):
        notifications = NotificationStore(self.paths, self.records)
        interactions = InteractionStore(self.paths, notifications, self.records)
        estimator = TokenEstimator(self.config.chars_per_token)
        working = WorkingMemory(self.paths, self.config, estimator, self.records)
        visual = VisualContext(self.paths, self.config, self.records)
        memory = LongTermMemory(self.paths, self.records)
        attention = InfiniteAttention(self.paths, self.config, estimator, self.records)
        initialization = Initialization(self.paths, self.records)
        scheduler_type = load_mind_tool(self.paths, "scheduler.py", "Scheduler")
        scheduler = scheduler_type(self.paths, interactions, self.records)
        tools = ToolRegistry(
            paths=self.paths,
            config=self.config,
            prompts=self.prompts,
            records=self.records,
            console=Console(quiet=True),
            interactions=interactions,
            memory=memory,
            working=working,
            streams=attention,
            visual=visual,
            initialization=initialization,
            scheduler=scheduler,
        )
        return notifications, interactions, estimator, working, memory, attention, tools

    def test_minimal_three_root_and_mind_layout(self) -> None:
        directories = {item.name for item in self.root.iterdir() if item.is_dir()}
        self.assertEqual(directories, {"app", "workspace"})
        app_directories = {item.name for item in self.paths.app.iterdir() if item.is_dir()}
        self.assertEqual(app_directories, {"prompts", "seed"})
        workspace_directories = {item.name for item in self.paths.root.iterdir() if item.is_dir()}
        self.assertEqual(workspace_directories, {"mind", "logs", "overrides"})
        mind_directories = {item.name for item in self.paths.mind.iterdir() if item.is_dir()}
        self.assertEqual(mind_directories, {"memory", "interactions", "tools", "space"})
        self.assertTrue(self.paths.self_file.is_file())
        self.assertTrue(self.paths.meta_memory.is_file())
        self.assertTrue((self.paths.memory / "harness/infinite-attention.txt").is_file())
        self.assertTrue(
            (self.paths.memory / "harness/memory-formation-and-meta-memory.txt").is_file()
        )
        self.assertTrue(
            (
                self.paths.memory
                / "harness/learning-self-improvement-and-adaptation.txt"
            ).is_file()
        )
        self.assertTrue((self.paths.created_tools / "scheduler.py").is_file())
        meta = self.paths.meta_memory.read_text()
        self.assertIn("`memory/harness/`", meta)
        self.assertIn("`memory/tools/`", meta)
        self.assertIn("memory/harness/infinite-attention.txt", meta)
        self.assertIn("learning-self-improvement-and-adaptation.txt", meta)
        self.assertIn("scheduled-actions-through-interactions.txt", meta)
        self.assertIn("## Available tools and apparatus", meta)
        self.assertIn("`mind/tools/scheduler.py`", meta)
        self.assertIn("`schedule_task`", meta)
        self.assertIn("When you create, adopt, substantially change, or", meta)
        self.assertFalse((self.paths.memory / "harness/index.txt").exists())
        self.assertFalse((self.paths.memory / "tools/index.txt").exists())
        self.assertFalse((self.paths.mind / "mind").exists())
        self.assertFalse((self.paths.mind / "working_memory").exists())
        self.assertFalse((self.paths.mind / "learning").exists())
        self.assertFalse((self.paths.mind / "goals").exists())

    def test_canonical_textual_protocol_and_legacy_recovery(self) -> None:
        parsed = parse_life_loop_output(
            '<think>Explore freely.</think><tool_call>{"tool":"read_file",'
            '"path":"mind/self.txt"}</tool_call>visible'
        )
        self.assertEqual(parsed.thoughts, ["Explore freely."])
        self.assertEqual(parsed.tools[0].name, "read_file")
        self.assertEqual(parsed.tools[0].arguments["path"], "mind/self.txt")
        self.assertEqual(parsed.visible, "visible")
        legacy = parse_life_loop_output(
            '<tool_call>{"name":"sleep","arguments":{"mode":"until_event"}}</tool_call>'
        )
        self.assertEqual(legacy.tools[0].name, "sleep")

    def test_malformed_tool_tag_is_detected_and_not_normalized_into_context(self) -> None:
        # An unclosed request is only suspect when the reply was cut off.
        parsed = parse_life_loop_output(
            '<think>I will read it.</think><tool_call>{"tool":"read_file",'
            '"path":"BROKEN_RAW_MARKER"}',
            finish_reason="length",
        )
        self.assertEqual(len(parsed.tools), 1)
        self.assertIsNotNone(parsed.tools[0].parse_error)
        self.assertNotIn("BROKEN_RAW_MARKER", parsed.visible)
        normalized = render_normalized_life_loop_output(parsed, include_tools=False)
        self.assertIn("<think>I will read it.</think>", normalized)
        self.assertNotIn("BROKEN_RAW_MARKER", normalized)

    def test_openai_transport_has_no_native_tool_state(self) -> None:
        captured: dict = {}

        def fake_open(request, timeout=0):
            captured.update(json.loads(request.data.decode()))
            return DummyHTTPResponse(
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
            )

        with mock.patch("urllib.request.urlopen", side_effect=fake_open):
            reply = OpenAICompatibleEngine(self.config, "key").complete(
                [{"role": "user", "content": "hello"}]
            )
        self.assertEqual(reply.content, "ok")
        self.assertNotIn("tools", captured)
        self.assertNotIn("tool_choice", captured)
        self.assertNotIn("previous_response_id", captured)

    def test_unauthenticated_ollama_transport_sends_no_authorization_header(self) -> None:
        config = Config(
            provider="ollama",
            model="qwen3:8b",
            context_window_tokens=32_768,
            reasoning_effort="none",
        )
        captured: dict[str, object] = {}

        def fake_open(request, timeout=0):
            captured["headers"] = {
                key.lower(): value for key, value in request.header_items()
            }
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode())
            return DummyHTTPResponse(
                {"message": {"content": "local-ok"}, "done": True}
            )

        with mock.patch("urllib.request.urlopen", side_effect=fake_open):
            reply = make_engine(config, None).complete(
                [{"role": "user", "content": "hello"}]
            )
        self.assertEqual(reply.content, "local-ok")
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/chat")
        self.assertNotIn("authorization", captured["headers"])
        self.assertEqual(
            captured["headers"]["user-agent"], "Artificium-revolution/1.10.2"
        )
        self.assertEqual(captured["payload"]["options"]["num_ctx"], 32_768)
        self.assertIs(captured["payload"]["think"], False)

    def test_image_working_set_materializes_for_both_provider_adapters(self) -> None:
        image = self.root / "transport.png"
        image.write_bytes(b"image-transport-bytes")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect the image."},
                    {
                        "type": "artificium_image",
                        "path": str(image),
                        "mime": "image/png",
                        "detail": "high",
                    },
                ],
            }
        ]

        openai_payload: dict = {}

        def fake_open(request, timeout=0):
            openai_payload.update(json.loads(request.data.decode()))
            return DummyHTTPResponse(
                {"choices": [{"message": {"content": "seen"}, "finish_reason": "stop"}]}
            )

        with mock.patch("urllib.request.urlopen", side_effect=fake_open):
            OpenAICompatibleEngine(self.config, "key").complete(messages)
        openai_image = openai_payload["messages"][0]["content"][1]
        self.assertEqual(openai_image["type"], "image_url")
        self.assertTrue(openai_image["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(openai_image["image_url"]["detail"], "high")
        self.assertNotIn("path", openai_image)

        anthropic = AnthropicEngine(self.config, "key")
        with mock.patch.object(
            anthropic,
            "_post",
            return_value={"content": [{"type": "text", "text": "seen"}]},
        ) as post:
            anthropic.complete(messages)
        anthropic_payload = post.call_args.args[0]
        anthropic_image = anthropic_payload["messages"][0]["content"][1]
        self.assertEqual(anthropic_image["type"], "image")
        self.assertEqual(anthropic_image["source"]["media_type"], "image/png")
        self.assertNotIn("path", anthropic_image)

    def test_one_shot_multi_image_context_is_consumed_after_success(self) -> None:
        first = self.root / "first.png"
        second = self.root / "second.png"
        first.write_bytes(b"first-image")
        second.write_bytes(b"second-image")
        Initialization(self.paths, self.records).finish(
            "Inspected Self, meta-memory, tools, environment, and pending interactions."
        )
        engine = FakeEngine(
            [
                '<think>I need both images once.</think><tool_call>'
                + json.dumps(
                    {
                        "tool": "load_images",
                        "paths": [str(first), str(second)],
                        "retention": "once",
                    }
                )
                + "</tool_call>",
                "<think>I inspected both images and no longer need them loaded.</think>",
            ]
        )
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.run_once(trigger="one-shot-vision-test")
        self.assertEqual(len(engine.requests), 2)
        self.assertNotIn("artificium_image", json.dumps(engine.requests[0]))
        second_request = json.dumps(engine.requests[1])
        self.assertEqual(second_request.count('"type": "artificium_image"'), 2)
        self.assertIn(str(first), second_request)
        self.assertIn(str(second), second_request)
        visual_messages = [
            message
            for message in engine.requests[1]
            if isinstance(message.get("content"), list)
        ]
        self.assertEqual(visual_messages[0]["role"], "user")
        self.assertEqual(agent.visual.list(), [])

    def test_one_shot_image_survives_failed_inference(self) -> None:
        image = self.root / "retry.png"
        image.write_bytes(b"retry-image")
        Initialization(self.paths, self.records).finish(
            "Inspected Self, meta-memory, tools, environment, and pending interactions."
        )

        class FailingEngine(Engine):
            def __init__(self) -> None:
                self.requests: list[list[dict]] = []

            def complete(self, messages: list[dict]) -> EngineReply:
                self.requests.append(messages)
                raise EngineError("temporary upstream failure", status=500)

        engine = FailingEngine()
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.visual.load([str(image)], retention="once")
        status = agent.status()
        self.assertEqual(len(status["active_images"]), 1)
        self.assertGreater(status["visual_context_tokens"], 0)
        with self.assertRaises(EngineError):
            agent.run_once(trigger="failed-one-shot-vision-test")
        self.assertEqual(len(agent.visual.list()), 1)
        self.assertEqual(
            json.dumps(engine.requests[0]).count('"type": "artificium_image"'), 1
        )

    def test_unrelated_rejection_preserves_images_and_does_not_retry(self) -> None:
        image = self.root / "keep.png"
        image.write_bytes(b"retained-image")

        class RejectText(Engine):
            calls = 0

            def complete(self, messages: list[dict]) -> EngineReply:
                self.calls += 1
                raise EngineError("HTTP 400: Failed to tokenize prompt", status=400)

        engine = RejectText()
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.visual.load([str(image)], retention="once")
        before = agent.visual.list()
        with self.assertRaisesRegex(EngineError, "Failed to tokenize"):
            agent.run_once(trigger="text-rejection-with-image")
        self.assertEqual(engine.calls, 1)
        self.assertEqual(agent.visual.list(), before)
        self.assertFalse(any(item["kind"] == "images_downgraded"
                             for item in agent.records.recent_operational(200)))

    def test_rejected_request_pauses_inference_while_new_messages_are_retained(self) -> None:
        class RejectText(Engine):
            calls = 0

            def complete(self, messages: list[dict]) -> EngineReply:
                self.calls += 1
                raise EngineError("HTTP 400: Failed to tokenize prompt", status=400,
                                  kind="tokenization", hint="Inspect the request.")

        engine = RejectText()
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.working.append({"role": "user", "content": "Preserve this history."}, origin="test")
        history = self.paths.working_context.read_bytes()
        polls = 0

        def observe_pause(_seconds: float) -> None:
            nonlocal polls
            polls += 1
            self.assertEqual(engine.calls, 1)
            self.assertTrue(agent._request_blocked)
            state = read_json(self.paths.runtime_state, {})
            self.assertEqual(state["status"], "blocked")
            self.assertIn("Failed to tokenize prompt", state["error"])
            if polls == 1:
                agent.notifications.create(type="external_event", source="test", summary="New message while paused")
            if polls == 3:
                agent._stop = True

        with mock.patch("artificium.runtime.time.sleep", side_effect=observe_pause):
            agent.run_forever(quiet=True)
        self.assertEqual(polls, 3)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(self.paths.working_context.read_bytes(), history)
        self.assertTrue(agent.notifications.has_new())
        blocked = [item for item in agent.records.recent_life(200) if item["kind"] == "engine_blocked"]
        self.assertEqual(len(blocked), 1)

        class Recovered(Engine):
            calls = 0

            def complete(self, messages: list[dict]) -> EngineReply:
                self.calls += 1
                agent._stop = True
                return EngineReply("Recovered.")

        recovered = Recovered()
        agent.engine = recovered
        agent.run_forever(quiet=True)
        self.assertEqual(recovered.calls, 1)
        self.assertFalse(agent._request_blocked)

    def test_transient_errors_keep_the_existing_backoff_path(self) -> None:
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        for error in (EngineError("overloaded", status=503),
                      EngineError("rate limited", status=429),
                      EngineError("connection failed", kind="network")):
            with self.subTest(error=str(error)):
                agent._error_sleep(error, 1)
                self.assertFalse(agent._request_blocked)
                self.assertEqual(read_json(self.paths.sleep_state, {})["reason"], "engine_error_backoff")

    def test_engine_backoff_is_not_interrupted_by_queued_or_new_messages(self) -> None:
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        agent.notifications.create(type="external_event", source="test", summary="Original message")
        with mock.patch("artificium.runtime.time.time", return_value=100):
            agent._error_sleep(EngineError("overloaded", status=503), 1)
            self.assertTrue(agent._sleep_active())
        agent.notifications.create(type="external_event", source="test", summary="New message")
        with mock.patch("artificium.runtime.time.time", return_value=159):
            self.assertTrue(agent._sleep_active())
        with mock.patch("artificium.runtime.time.time", return_value=160):
            self.assertFalse(agent._sleep_active())
        queued = [read_json(p) for p in self.paths.notifications_new.glob("*.json")]
        self.assertEqual({n["summary"] for n in queued}, {"Original message", "New message"})
        self.assertEqual(agent._pending_recovery()["failed_attempts"], 1)

    def test_long_failure_streak_keeps_capped_backoff(self) -> None:
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        agent._error_sleep(EngineError("still overloaded", status=503), 10_000)
        self.assertEqual(read_json(self.paths.sleep_state)["seconds"], 900)
        self.assertFalse(agent.notifications.has_new())

    def test_transient_retries_preserve_messages_and_commit_one_recovery_after_restart(self) -> None:
        errors = [EngineError("overloaded", status=503), EngineError("rate limited", status=429),
                  EngineError("connection failed", kind="network")]

        class RecoveringEngine(Engine):
            def complete(self, messages: list[dict]) -> EngineReply:
                if errors:
                    raise errors.pop(0)
                return EngineReply("<think>Recovered.</think>")

        engine = RecoveringEngine()
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.initialization.finish("Already initialized for this test.")
        agent.working.append({"role": "user", "content": "Keep this history."}, origin="test")
        before = self.paths.working_context.read_bytes()
        agent.notifications.create(type="external_event", source="test", summary="Queued work")
        for attempt in range(1, 4):
            with self.assertRaises(EngineError) as caught:
                agent.run_once(trigger="test-retry")
            agent._error_sleep(caught.exception, attempt)
            deadline = read_json(self.paths.sleep_state)["wake_at_epoch"]
            with mock.patch("artificium.runtime.time.time", return_value=deadline):
                self.assertFalse(agent._sleep_active())
            self.assertEqual(agent._pending_recovery()["failed_attempts"], attempt)
            self.assertEqual(self.paths.working_context.read_bytes(), before)
            self.assertEqual(len(list(self.paths.notifications_new.glob("*.json"))), 1)
            self.assertFalse(list(self.paths.notifications_processing.glob("*.json")))

        restarted = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        self.assertEqual(restarted._pending_recovery()["failed_attempts"], 3)
        restarted.run_once(trigger="retry-after-restart")
        self.assertIsNone(restarted._pending_recovery())
        self.assertFalse(restarted.notifications.has_new())
        history = self.paths.working_context.read_text()
        self.assertIn("Keep this history.", history)
        self.assertIn("Queued work", history)
        self.assertIn("connection failed", history)
        self.assertIn("3 failed attempts", history)
        self.assertEqual(history.count("SYSTEM NOTIFICATION — RECOVERY"), 1)
        restarted.run_once(trigger="later-turn")
        self.assertEqual(self.paths.working_context.read_text().count("SYSTEM NOTIFICATION — RECOVERY"), 1)

    def test_retry_timer_runs_before_continuation_without_notifications(self) -> None:
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        clock = [100.0]
        attempts: list[tuple[float, str]] = []

        def turn(*, trigger: str, continuous: bool) -> str:
            self.assertTrue(continuous)
            attempts.append((clock[0], trigger))
            if len(attempts) == 1:
                raise EngineError("overloaded", status=503)
            agent._stop = True
            return "Recovered"

        def advance(seconds: float) -> None:
            clock[0] += seconds
            self.assertLessEqual(clock[0], 160, "Retry never became runnable")

        with mock.patch.object(agent, "run_turn", side_effect=turn), \
             mock.patch("artificium.runtime.time.time", side_effect=lambda: clock[0]), \
             mock.patch("artificium.runtime.time.sleep", side_effect=advance):
            agent.run_forever(quiet=True)
        self.assertEqual(attempts, [(100.0, "startup"), (160.0, "engine_retry")])
        self.assertFalse(agent.notifications.has_new())

    def test_legacy_notice_backlog_is_bounded_and_archived_only_after_success(self) -> None:
        class RejectOnce(FakeEngine):
            def complete(self, messages: list[dict]) -> EngineReply:
                if not self.requests:
                    self.requests.append(messages)
                    raise EngineError("temporary failure", status=503)
                return super().complete(messages)

        engine = RejectOnce(["<think>Messages received.</think>"])
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.initialization.finish("Already initialized for this test.")
        agent.config.notification_batch_size = 2
        originals: dict[str, bytes] = {}
        for i in range(64):
            for kind in ("wake", "recovery"):
                n = agent.notifications.create(type=kind, source="artificium_runtime",
                    summary="Earlier retry", metadata={"reason": f"Earlier failure {i}"})
                originals[n.queue_path.name] = n.queue_path.read_bytes()
        # A recovery event from an external source must remain an ordinary event.
        external = agent.notifications.create(type="recovery", source="external-client",
            summary="Client recovery", metadata={"reason": "Preserve this client recovery exactly"})
        first = agent.notifications.create(type="external_event", source="test", summary="First user message")
        second = agent.notifications.create(type="external_event", source="test", summary="Second user message")
        for n in (external, first):
            originals[n.queue_path.name] = n.queue_path.read_bytes()
        before = self.paths.working_context.read_bytes()

        with self.assertRaises(EngineError):
            agent.run_once(trigger="legacy-backlog")
        self.assertEqual(len(list(self.paths.notifications_new.glob("*.json"))), 131)
        self.assertFalse(list(self.paths.notifications_processing.glob("*.json")))
        self.assertFalse(list(self.paths.notifications_delivered.glob("*.json")))
        self.assertEqual(self.paths.working_context.read_bytes(), before)
        batch = engine.requests[0][-2]["content"]
        self.assertIn("Consolidated 128 internal notices", batch)
        self.assertIn("Earlier failure 63", batch)
        self.assertIn("Preserve this client recovery exactly", batch)
        self.assertIn("First user message", batch)
        self.assertNotIn("Second user message", batch)
        # The bound concerns the consolidated backlog, not the context notices
        # a 20k test window adds once the always-prompts approach its target.
        backlog = batch.split("[SYSTEM NOTIFICATION — WORKING-CONTEXT MILESTONE]")[0]
        self.assertLess(len(backlog), 6_000)

        agent.run_once(trigger="legacy-backlog-retry")
        self.assertEqual([p.name for p in self.paths.notifications_new.glob("*.json")],
                         [second.queue_path.name])
        for name, content in originals.items():
            self.assertEqual((self.paths.notifications_delivered / name).read_bytes(), content)
        self.assertEqual(self.paths.working_context.read_text().count("EARLIER RUNTIME NOTICES"), 1)

    def test_ordinary_sleep_still_wakes_for_a_new_message(self) -> None:
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        atomic_write_json(self.paths.sleep_state, {"active": True, "mode": "until_event",
                                                  "started_at": "test"})
        self.assertTrue(agent._sleep_active())
        agent.notifications.create(type="external_event", source="test", summary="Wake up")
        self.assertFalse(agent._sleep_active())
        kinds = sorted(read_json(p)["type"] for p in self.paths.notifications_new.glob("*.json"))
        self.assertEqual(kinds, ["external_event", "wake"])

    def test_stop_at_provider_boundary_keeps_recovery_and_messages_pending(self) -> None:
        class StopAfterReply(Engine):
            def complete(self, messages: list[dict]) -> EngineReply:
                agent._stop = True
                return EngineReply("<think>Accepted.</think>")

        agent = Artificium(self.paths, engine=StopAfterReply(), console=Console(quiet=True))
        agent.notifications.create(type="external_event", source="test", summary="Keep queued")
        agent._error_sleep(EngineError("overloaded", status=503), 1)
        with mock.patch("artificium.runtime.time.time", return_value=read_json(self.paths.sleep_state)["wake_at_epoch"]):
            self.assertFalse(agent._sleep_active())
        agent.run_once(trigger="interrupted-recovery")
        self.assertIsNotNone(agent._pending_recovery())
        self.assertTrue(agent.notifications.has_new())
        self.assertFalse(agent.working.load())

    def test_watch_and_status_explain_paused_requests(self) -> None:
        from artificium.operator import format_status, status_snapshot
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        agent._error_sleep(EngineError("HTTP 400: bad request", status=400), 1)
        state = status_snapshot(self.paths)
        state["process"].update(alive=True, owned=True, pid=123)
        self.assertIn("model requests paused", format_status(state))
        self.assertIn("HTTP 400: bad request", format_status(state))
        watched = io.StringIO()
        with redirect_stdout(watched):
            _print_life_record(json.dumps({
                "kind": "engine_request_failed", "request_id": "test", "duration_seconds": 0,
                "error": "HTTP 400: Failed to tokenize prompt", "hint": "Inspect the request.",
                "model_log_path": "test.json",
            }))
            _print_life_record(json.dumps({"kind": "engine_blocked"}))
        self.assertIn("Failed to tokenize prompt", watched.getvalue())
        self.assertIn("Inspect the request.", watched.getvalue())
        self.assertIn("python3 artificium.py restart", watched.getvalue())

    def test_vision_rejection_releases_images_and_retries_without_them(self) -> None:
        image = self.root / "unsupported.png"
        image.write_bytes(b"unsupported-image")
        Initialization(self.paths, self.records).finish(
            "Inspected Self, meta-memory, tools, environment, and pending interactions."
        )

        class RejectVisionOnce(Engine):
            def __init__(self) -> None:
                self.requests: list[list[dict]] = []

            def complete(self, messages: list[dict]) -> EngineReply:
                self.requests.append(messages)
                if len(self.requests) == 1:
                    raise EngineError("image input is unsupported", status=415)
                return EngineReply("<think>I will use a conversion tool if needed.</think>")

        engine = RejectVisionOnce()
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.visual.load([str(image)], retention="persistent")
        agent.run_once(trigger="vision-fallback-test")
        self.assertEqual(len(engine.requests), 2)
        self.assertEqual(
            json.dumps(engine.requests[0]).count('"type": "artificium_image"'), 1
        )
        self.assertNotIn("artificium_image", json.dumps(engine.requests[1]))
        self.assertEqual(agent.visual.list(), [])
        self.assertIn("engine rejected visual input", self.paths.working_context.read_text())

    def test_persistent_image_survives_until_explicit_release_without_duplicates(self) -> None:
        image = self.root / "persistent.png"
        image.write_bytes(b"persistent-image")
        Initialization(self.paths, self.records).finish(
            "Inspected Self, meta-memory, tools, environment, and pending interactions."
        )
        engine = FakeEngine(
            [
                '<think>I need sustained visual access.</think><tool_call>'
                + json.dumps(
                    {
                        "tool": "load_images",
                        "paths": [str(image), str(image)],
                        "retention": "persistent",
                    }
                )
                + "</tool_call>",
                '<think>I will inspect the active set.</think><tool_call>'
                '{"tool":"list_loaded_images"}</tool_call>',
                '<think>The visual work is done.</think><tool_call>'
                '{"tool":"release_images","all_images":true}</tool_call>',
                "<think>No image needs to remain active.</think>",
            ]
        )
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.run_once(trigger="persistent-vision-test")
        image_counts = [
            json.dumps(request).count('"type": "artificium_image"')
            for request in engine.requests
        ]
        self.assertEqual(image_counts, [0, 1, 1, 0])
        self.assertEqual(agent.visual.list(), [])

    def test_invalid_tool_request_is_transactional_and_context_is_repaired(self) -> None:
        config = ConfigStore(self.paths).load()
        config.max_life_loop_rounds = 2
        ConfigStore(self.paths).save(config)
        Initialization(self.paths, self.records).finish(
            "Inspected Self, meta-memory, tools, environment, and pending interactions."
        )
        engine = FakeEngine(
            [
                '<think>I will list the mind.</think><tool_call>{"tool":"list_directory",'
                '"path":"BROKEN_RAW_MARKER"',
                '<think>I will retry with the repaired flat form.</think><tool_call>'
                '{"tool":"list_directory","path":"mind"}</tool_call>',
            ]
        )
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.run_once(trigger="tool-repair-test")
        self.assertEqual(len(engine.requests), 2)
        self.assertIn("TOOL REQUEST REPAIR", json.dumps(engine.requests[1]))
        context = self.paths.working_context.read_text()
        self.assertNotIn("BROKEN_RAW_MARKER", context)
        calls = [
            item
            for item in self.records.recent_life(200)
            if item.get("kind") == "tool_call"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "list_directory")

    def test_tool_repair_is_exactly_specific_to_the_failed_operation(self) -> None:
        config = ConfigStore(self.paths).load()
        config.max_life_loop_rounds = 2
        ConfigStore(self.paths).save(config)
        Initialization(self.paths, self.records).finish(
            "Inspected Self, meta-memory, tools, environment, and pending interactions."
        )
        engine = FakeEngine(
            [
                '<think>I will open a stream.</think><tool_call>'
                '{"tool":"open_attention","source":"large.txt","objective":"Find it",'
                '"not_a_real_argument":true}</tool_call>',
                "<think>I now understand the exact operation-specific contract.</think>",
            ]
        )
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.run_once(trigger="specific-repair-test")
        repair_request = json.dumps(engine.requests[1])
        self.assertIn("TOOL REQUEST REPAIR", repair_request)
        self.assertIn('\\"tool\\": \\"open_attention\\"', repair_request)
        self.assertIn('\\"source\\": \\"PATH\\"', repair_request)
        self.assertIn('\\"objective\\": \\"PRECISE_OBJECTIVE\\"', repair_request)
        self.assertNotIn('\\"tool\\": \\"offload_working_memory\\"', repair_request)
        self.assertIn("not_a_real_argument", repair_request)

    def test_calls_before_the_first_invalid_one_still_execute(self) -> None:
        # The old design withheld an entire response when any one call in it
        # was invalid. A valid call ahead of the bad one is an independent,
        # already-safe action, so it now runs; only the invalid call onward
        # is withheld, and the repair notice says precisely which is which.
        config = ConfigStore(self.paths).load()
        config.max_life_loop_rounds = 1
        ConfigStore(self.paths).save(config)
        Initialization(self.paths, self.records).finish(
            "Inspected Self, meta-memory, tools, environment, and pending interactions."
        )
        target = self.paths.root / "mind" / "space" / "should-exist.txt"
        engine = FakeEngine(
            [
                '<think>The first call is independent of the second.</think>'
                '<tool_call>{"tool":"write_file","path":"mind/space/should-exist.txt"'
                ',"content":"side effect","mode":"create"}</tool_call>'
                '<tool_call>{"tool":"not_a_real_tool"}</tool_call>'
            ]
        )
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.run_once(trigger="transactional-tool-test")
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text(), "side effect")
        calls = [
            item
            for item in self.records.recent_life(200)
            if item.get("kind") == "tool_call"
        ]
        self.assertEqual([c["name"] for c in calls], ["write_file"])
        repair_records = [
            json.loads(line)
            for line in self.paths.working_context.read_text().splitlines()
            if line.strip()
        ]
        repair_context = str(repair_records[-1]["content"])
        self.assertIn("already executed", repair_context)
        self.assertIn("not_a_real_tool", repair_context)
        self.assertIn('"call_index": 1', repair_context)
        self.assertIn('"call_index": 2', repair_context)
        self.assertIn("mind/space/should-exist.txt", repair_context)

    def test_unknown_tool_repair_suggests_nearest_exact_name(self) -> None:
        *_, tools = self.components()
        error = tools.validate(
            ToolIntent(
                id="tool_typo",
                name="open_attentin",
                arguments={"source": "x", "objective": "y"},
            ),
            call_index=3,
        )
        self.assertIsNotNone(error)
        assert error is not None
        self.assertEqual(error["call_index"], 3)
        self.assertEqual(error["suggested_tool"], "open_attention")
        self.assertEqual(error["canonical_example"]["tool"], "open_attention")

    def test_every_canonical_repair_example_matches_its_tool_signature(self) -> None:
        *_, tools = self.components()
        self.assertEqual(set(tools.CANONICAL_EXAMPLES), set(tools._functions))
        for call_index, (name, example) in enumerate(
            tools.CANONICAL_EXAMPLES.items(),
            start=1,
        ):
            self.assertEqual(example["tool"], name)
            error = tools.validate(
                ToolIntent(
                    id=f"canonical_{call_index}",
                    name=name,
                    arguments={
                        key: value for key, value in example.items() if key != "tool"
                    },
                ),
                call_index=call_index,
            )
            self.assertIsNone(error, msg=f"invalid canonical example for {name}: {error}")

    def test_setup_writes_self_directly_without_model_rewriting(self) -> None:
        other = Path(self.temporary.name) / "other"
        paths = Paths(other)
        _seed_install(paths)
        with mock.patch(
            "artificium.setup.discover_provider_models",
            return_value=ModelDiscovery(),
        ):
            result = SetupWizard(paths).run(
                SetupOptions(
                    provider="custom",
                    context_window_tokens=64000,
                    model="model-x",
                    api_key="test-key",
                    api_url="http://localhost:9999/v1",
                    self_directive=(
                        "My name is Ada. I investigate difficult systems independently."
                    ),
                )
            )
        self.assertEqual(result["self"], str(paths.self_file))
        self.assertIn("My name is Ada", paths.self_file.read_text())
        self.assertEqual(Initialization(paths, Records(paths)).ensure()["status"], "pending")

    def test_advanced_custom_contract_setup_skips_openai_discovery(self) -> None:
        other = Path(self.temporary.name) / "advanced-custom"
        paths = Paths(other)
        _seed_install(paths)
        contract_path = Path(self.temporary.name) / "custom-contract.json"
        contract = {
            "url": "https://unusual.example/v2/generate?region=test",
            "auth": False,
            "body": {
                "engine": "$artificium.model",
                "dialogue": "$artificium.messages",
                "limit": "$artificium.max_output_tokens",
            },
            "response": {"content": "/generation/text"},
        }
        contract_path.write_text(json.dumps(contract))
        with mock.patch("artificium.setup.discover_provider_models") as discovery:
            SetupWizard(paths).run(
                SetupOptions(
                    model="unusual-model",
                    custom_contract_file=str(contract_path),
                    context_window_tokens=64_000,
                    max_output_tokens=4096,
                )
            )
        discovery.assert_not_called()
        config = ConfigStore(paths).load()
        self.assertEqual(config.provider, "custom")
        self.assertEqual(config.adapter, "custom_json")
        self.assertEqual(config.base_url, "https://unusual.example")
        self.assertEqual(config.endpoint, "v2/generate?region=test")
        self.assertEqual(config.custom_contract, contract)
        summary = make_engine(config, None).request_summary()
        self.assertEqual(summary["body"]["engine"], "unusual-model")
        self.assertEqual(summary["body"]["limit"], 4096)

    def test_doctor_does_not_probe_an_openai_models_route_for_custom_json(self) -> None:
        contract = {
            "url": "https://unusual.example/generate",
            "auth": {"header": "X-API-Key", "prefix": "", "required": True},
            "body": {"prompt": "$artificium.prompt"},
            "response": {"content": "/text"},
        }
        ConfigStore(self.paths).save(
            Config(
                provider="custom",
                model="unusual-model",
                base_url="https://unusual.example",
                endpoint="generate",
                adapter="custom_json",
                custom_contract=contract,
            )
        )
        output = io.StringIO()
        with (
            mock.patch("artificium.cli.discover_provider_models") as discovery,
            redirect_stdout(output),
        ):
            self.assertEqual(main(["--root", str(self.root), "doctor"]), 0)
        discovery.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertTrue(result["api_key_required"])
        self.assertNotIn("provider_probe", result)
        self.assertEqual(result["engine_request"]["url"], contract["url"])

    def test_custom_contract_can_return_to_easy_openai_compatible_mode(self) -> None:
        contract_path = Path(self.temporary.name) / "custom-contract.json"
        contract_path.write_text(
            json.dumps(
                {
                    "url": "https://unusual.example/generate",
                    "auth": False,
                    "body": {"prompt": "$artificium.prompt"},
                    "response": {"content": "/text"},
                }
            )
        )
        advanced = SetupWizard(self.paths).reconfigure(
            SetupOptions(
                context_window_tokens=64000,
                provider="custom",
                model="unusual-model",
                custom_contract_file=str(contract_path),
            )
        )
        self.assertEqual(advanced.adapter, "custom_json")
        with mock.patch(
            "artificium.setup.discover_provider_models",
            return_value=ModelDiscovery(),
        ):
            easy = SetupWizard(self.paths).reconfigure(
                SetupOptions(
                context_window_tokens=64000,
                    provider="custom",
                    api_url="http://localhost:9000/v1",
                    custom_contract_file="none",
                )
            )
        self.assertEqual(easy.adapter, "openai_compatible")
        self.assertEqual(easy.custom_contract, {})
        self.assertEqual(easy.base_url, "http://localhost:9000/v1")

    def test_interactive_setup_uses_neutral_self_without_asking_for_genesis(self) -> None:
        options = SetupOptions(provider="custom", model="model-x", api_key="test-key",
                               api_url="http://localhost:9999/v1", context_window_tokens=64000)
        output = io.StringIO()
        with mock.patch("builtins.input", return_value=""), redirect_stdout(output), \
             mock.patch("artificium.setup.discover_provider_models", return_value=ModelDiscovery()):
            config, key, result = SetupWizard(self.paths).interactive(options)
        self.assertIsNone(result.self_directive)
        self.assertNotIn("Self directive", output.getvalue())
        self.assertEqual(config.context_window_tokens, 64000)

    def test_interactive_reasoning_off_is_an_explicit_value(self) -> None:
        from artificium.setup import configure_reasoning
        options = SetupOptions(provider="ollama", model="qwen3:8b")
        with mock.patch("builtins.input", return_value="off"):
            configure_reasoning(options, Config(provider="ollama", model="qwen3:8b"), {"thinking": True})
        self.assertEqual(options.reasoning, "off")

    def test_advanced_generation_prompts_match_the_provider_contract(self) -> None:
        ollama = supported_generation_controls("ollama", "qwen3:8b")
        self.assertNotIn("reasoning_budget_tokens", ollama)
        self.assertNotIn("reasoning_mode", ollama)
        self.assertNotIn("frequency_penalty", ollama)
        self.assertNotIn("presence_penalty", ollama)
        self.assertIn("repetition_penalty", ollama)
        self.assertIn("min_p", ollama)

        options = SetupOptions(provider="ollama", model="qwen3:8b")
        prompts: list[str] = []

        def answer(prompt: str) -> str:
            prompts.append(prompt)
            return "done"

        output = io.StringIO()
        with mock.patch("builtins.input", side_effect=answer), redirect_stdout(output):
            configure_generation_interactive(
                options, provider="ollama", model="qwen3:8b"
            )
        rendered = output.getvalue()
        self.assertNotIn("Exact reasoning token budget", rendered)
        self.assertNotIn("OpenAI reasoning mode", rendered)
        self.assertNotIn("Frequency penalty", rendered)
        self.assertNotIn("Presence penalty", rendered)
        self.assertIn("Repetition penalty", rendered)

        openai = supported_generation_controls("openai", "gpt-4.1")
        self.assertEqual(
            openai,
            {"temperature", "max_output_tokens", "top_p"},
        )
        self.assertIn(
            "reasoning_effort",
            supported_generation_controls("openai", "gpt-5.6-luna"),
        )
        self.assertIn(
            "reasoning_mode",
            supported_generation_controls("openai", "gpt-5.6-luna"),
        )
        openrouter_gpt_oss = supported_generation_controls(
            "openrouter", "openai/gpt-oss-120b"
        )
        self.assertIn("reasoning_effort", openrouter_gpt_oss)
        self.assertNotIn("reasoning_budget_tokens", openrouter_gpt_oss)
        custom = supported_generation_controls("custom", "unknown-model")
        self.assertIn("reasoning_effort", custom)
        self.assertIn("top_k", custom)
        self.assertIn("min_p", custom)
        self.assertIn("repetition_penalty", custom)
        gemini_25 = supported_generation_controls("gemini", "gemini-2.5-pro")
        gemini_3 = supported_generation_controls("gemini", "gemini-3-pro-preview")
        self.assertIn("reasoning_budget_tokens", gemini_25)
        self.assertNotIn("reasoning_budget_tokens", gemini_3)

    def test_setup_rejects_an_invalid_provider_contract_before_writing(self) -> None:
        other = Path(self.temporary.name) / "invalid-contract"
        paths = Paths(other)
        _seed_install(paths)
        with (
            mock.patch(
                "artificium.setup.discover_provider_models",
                return_value=ModelDiscovery(),
            ),
            self.assertRaisesRegex(EngineError, "effort or max_tokens, not both"),
        ):
            SetupWizard(paths).run(
                SetupOptions(
                    provider="openrouter",
                    model="openai/gpt-5.6-luna",
                    api_key="test-key",
                    context_window_tokens=100_000,
                    reasoning_effort="high",
                    reasoning_budget_tokens=4096,
                )
            )
        self.assertFalse(paths.config.exists())

    def test_openrouter_provider_route_is_set_verified_and_easy_to_clear(self) -> None:
        self.assertEqual(normalize_openrouter_provider("Cerebras"), "cerebras")
        routed = route_openrouter_requests(
            {"provider": {"data_collection": "deny"}}, "cerebras"
        )
        self.assertEqual(
            routed,
            {
                "provider": {
                    "data_collection": "deny",
                    "only": ["cerebras"],
                }
            },
        )
        self.assertEqual(
            route_openrouter_requests(routed, "automatic"),
            {"provider": {"data_collection": "deny"}},
        )

        other = Path(self.temporary.name) / "openrouter-route"
        paths = Paths(other)
        _seed_install(paths)
        with mock.patch(
            "artificium.setup.discover_provider_models",
            return_value=ModelDiscovery(),
        ):
            SetupWizard(paths).run(
                SetupOptions(
                    provider="openrouter",
                    model="openai/gpt-oss-120b",
                    api_key="test-key",
                    context_window_tokens=131_072,
                    reasoning_effort="medium",
                    openrouter_provider="cerebras",
                )
            )
        config = ConfigStore(paths).load()
        self.assertEqual(
            config.request_options,
            {"provider": {"only": ["cerebras"]}},
        )
        prepared = make_engine(config, "test-key").prepare(
            [{"role": "user", "content": "hello"}]
        )
        self.assertEqual(
            prepared.payload["provider"],
            {"require_parameters": True, "only": ["cerebras"]},
        )

        with mock.patch(
            "artificium.setup.discover_provider_models",
            return_value=ModelDiscovery(),
        ):
            updated = SetupWizard(paths).reconfigure(
                SetupOptions(openrouter_provider="automatic")
            )
        self.assertEqual(updated.request_options, {})

    def test_missing_cloud_capacity_is_labelled_default_until_inference_is_checked(self) -> None:
        with mock.patch("artificium.setup.discover_provider_models", return_value=ModelDiscovery()):
            config, _ = SetupWizard(self.paths)._build(SetupOptions(provider="openrouter", model="provider/model", api_key="test-key"))
        self.assertEqual(config.context_window_tokens, 32768)
        self.assertEqual(config.context_window_source, "default")

    def test_local_endpoint_urls_are_normalized_from_common_inputs(self) -> None:
        self.assertEqual(
            normalize_openai_endpoint("localhost:11434", provider="ollama"),
            ("http://localhost:11434/v1", "chat/completions"),
        )
        self.assertEqual(
            normalize_openai_endpoint(
                "http://localhost:11434/api", provider="ollama"
            ),
            ("http://localhost:11434/v1", "chat/completions"),
        )
        self.assertEqual(
            normalize_openai_endpoint(
                "http://gpu-box:8000/v1/chat/completions", provider="vllm"
            ),
            ("http://gpu-box:8000/v1", "chat/completions"),
        )
        self.assertEqual(
            normalize_openai_endpoint("https://example.test/openai/v1"),
            ("https://example.test/openai/v1", "chat/completions"),
        )
        self.assertEqual(
            normalize_provider_endpoint(
                "https://pod.example/v1/chat/completions", provider="ollama"
            ),
            ("https://pod.example", "api/chat", "ollama"),
        )

    def test_openai_compatible_model_discovery_is_bounded_and_deduplicated(self) -> None:
        response = DummyHTTPResponse(
            {"data": [{"id": "model-a"}, {"id": "model-b"}, {"id": "model-a"}]}
        )
        with mock.patch("urllib.request.urlopen", return_value=response) as opened:
            result = discover_models("http://localhost:8000/v1", timeout=0.25)
        self.assertEqual(result.models, ("model-a", "model-b"))
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:8000/v1/models")
        request_headers = {key.lower(): value for key, value in request.headers.items()}
        self.assertNotIn("authorization", request_headers)
        self.assertEqual(
            request_headers["user-agent"], "Artificium-revolution/1.10.2"
        )
        self.assertEqual(opened.call_args.kwargs["timeout"], 0.25)

    def test_ollama_discovery_falls_back_to_native_tags(self) -> None:
        model = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
        response = DummyHTTPResponse({"models": [{"name": model}]})
        with (
            mock.patch(
                "artificium.setup.discover_models",
                return_value=ModelDiscovery(
                    error="server returned HTTP 404",
                    status=404,
                    endpoint="https://pod.example/v1/models",
                ),
            ),
            mock.patch("urllib.request.urlopen", return_value=response) as opened,
        ):
            result = discover_provider_models(
                "ollama",
                "https://pod.example/v1",
                timeout=0.25,
            )
        self.assertEqual(result.models, (model,))
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://pod.example/api/tags")
        self.assertEqual(opened.call_args.kwargs["timeout"], 0.25)

    def test_interactive_setup_does_not_misdiagnose_403_as_api_key(self) -> None:
        from artificium.setup_ui import SetupCancelled
        options = SetupOptions(provider="ollama", api_url="https://example-11434.proxy.runpod.net/")
        discovery = ModelDiscovery(error="server returned HTTP 403", status=403,
                                   error_kind="permission", hint="Check the hosting proxy permissions.")
        output = io.StringIO()
        def answer(prompt):
            return "9" if prompt.startswith("Next step") else ""
        with mock.patch("artificium.setup.discover_provider_models", return_value=discovery), \
             mock.patch("getpass.getpass") as key_prompt, \
             mock.patch("builtins.input", side_effect=answer), redirect_stdout(output), \
             self.assertRaises(SetupCancelled):
            SetupWizard(self.paths).interactive(options)
        key_prompt.assert_not_called()
        self.assertIn("hosting proxy", output.getvalue())

    def test_interactive_setup_prompts_for_key_on_401_only(self) -> None:
        options = SetupOptions(
            provider="custom",
            context_window_tokens=64000,
            api_url="https://authenticated.example/v1",
        )
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch(
                "artificium.setup.discover_provider_models",
                side_effect=[
                    ModelDiscovery(
                        error="server returned HTTP 401",
                        status=401,
                        authentication_required=True,
                    ),
                    ModelDiscovery(models=("private-model",)),
                ],
            ),
            mock.patch("getpass.getpass", return_value="private-key") as key_prompt,
            mock.patch("builtins.input", return_value=""),
        ):
            config, key, result = SetupWizard(self.paths).interactive(options)
        key_prompt.assert_called_once()
        self.assertEqual(result.api_key, "private-key")
        self.assertEqual(result.model, "private-model")

    def test_interactive_ollama_setup_discovers_models_without_a_key(self) -> None:
        options = SetupOptions(provider="ollama")
        output = io.StringIO()
        with (
            mock.patch(
                "artificium.setup.discover_provider_models",
                return_value=ModelDiscovery(models=("qwen3:8b", "gemma3:12b")),
            ),
            mock.patch("artificium.setup.detected_context_window", return_value=None),
            mock.patch("builtins.input", side_effect=lambda p: "2" if p.startswith("Model (") else ""),
            redirect_stdout(output),
        ):
            config, key, result = SetupWizard(self.paths).interactive(options)
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(result.model, "gemma3:12b")
        self.assertEqual(result.api_url, "http://127.0.0.1:11434")
        self.assertEqual(config.endpoint, "api/chat")
        self.assertEqual(config.adapter, "ollama")
        self.assertFalse(result.api_key)
        self.assertEqual(key, "")

    def test_ollama_and_vllm_setup_are_noninteractive_without_fake_keys(self) -> None:
        self.assertTrue(
            _can_setup_noninteractive(
                SetupOptions(provider="ollama", model="qwen3:8b")
            )
        )
        self.assertTrue(
            _can_setup_noninteractive(
                SetupOptions(provider="vllm", model="Qwen/Qwen3-8B")
            )
        )
        self.assertTrue(
            _can_setup_noninteractive(
                SetupOptions(
                    provider="custom",
                    model="local-model",
                    api_url="http://localhost:9999/v1",
                )
            )
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                _can_setup_noninteractive(
                    SetupOptions(provider="openrouter", model="provider/model")
                )
            )

        other = Path(self.temporary.name) / "ollama"
        paths = Paths(other)
        _seed_install(paths)
        with (
            mock.patch(
                "artificium.setup.discover_provider_models",
                return_value=ModelDiscovery(),
            ),
            mock.patch("artificium.setup.detected_context_window", return_value=None),
        ):
            SetupWizard(paths).run(
                SetupOptions(provider="ollama", model="qwen3:8b")
            )
        config = ConfigStore(paths).load()
        self.assertEqual(config.base_url, "http://127.0.0.1:11434")
        self.assertEqual(config.endpoint, "api/chat")
        self.assertEqual(config.adapter, "ollama")
        self.assertIsNone(SecretsStore(paths).resolve_api_key(config))

    def test_saved_api_key_is_bound_to_its_provider(self) -> None:
        secrets = SecretsStore(self.paths)
        secrets.save_api_key("cloud-secret", provider="openrouter")
        openrouter = Config(provider="openrouter", model="provider/model")
        ollama = Config(provider="ollama", model="qwen3:8b")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(secrets.resolve_api_key(openrouter), "cloud-secret")
            self.assertIsNone(secrets.resolve_api_key(ollama))

    def test_reconfigure_preserves_self_and_memory(self) -> None:
        self.paths.self_file.write_text("My durable custom Self.\n")
        memory = self.paths.memory / "entities/user_1/preferences.txt"
        memory.parent.mkdir(parents=True, exist_ok=True)
        memory.write_text("Prefers direct answers.\n")
        with mock.patch(
            "artificium.setup.discover_provider_models",
            return_value=ModelDiscovery(),
        ):
            updated = SetupWizard(self.paths).reconfigure(
                SetupOptions(context_window_tokens=64_000, vision="no")
            )
        self.assertEqual(updated.context_window_tokens, 64_000)
        self.assertEqual(updated.vision, "no")
        self.assertEqual(self.paths.self_file.read_text(), "My durable custom Self.\n")
        self.assertEqual(memory.read_text(), "Prefers direct answers.\n")

    def test_reconfigure_switches_to_ollama_without_touching_mind(self) -> None:
        self.paths.self_file.write_text("Persistent Self.\n")
        memory = self.paths.memory / "local-provider-test.txt"
        memory.write_text("Keep this memory.\n")
        SecretsStore(self.paths).save_api_key("old-cloud-key", provider="custom")
        with (
            mock.patch(
                "artificium.setup.discover_provider_models",
                return_value=ModelDiscovery(),
            ),
            mock.patch("artificium.setup.detected_context_window", return_value=None),
        ):
            updated = SetupWizard(self.paths).reconfigure(
                SetupOptions(provider="ollama", model="qwen3:8b")
            )
        self.assertEqual(updated.provider, "ollama")
        self.assertEqual(updated.base_url, "http://127.0.0.1:11434")
        self.assertEqual(updated.endpoint, "api/chat")
        self.assertEqual(updated.adapter, "ollama")
        self.assertEqual(self.paths.self_file.read_text(), "Persistent Self.\n")
        self.assertEqual(memory.read_text(), "Keep this memory.\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(SecretsStore(self.paths).resolve_api_key(updated))

    def test_reconfigure_validates_discovered_context_before_saving(self) -> None:
        before = ConfigStore(self.paths).load()
        discovery = ModelDiscovery(
            models=(before.model,),
            details={before.model: {"context_length": 50_000}},
        )
        with (
            mock.patch(
                "artificium.setup.discover_provider_models",
                return_value=discovery,
            ),
            self.assertRaisesRegex(ValueError, "100,000.*50,000"),
        ):
            SetupWizard(self.paths).reconfigure(
                SetupOptions(context_window_tokens=100_000)
            )
        self.assertEqual(ConfigStore(self.paths).load(), before)

    def test_interaction_is_notification_then_durable_event(self) -> None:
        client = ArtificiumClient(self.root)
        event, _ = client.send(
            "room-1",
            sender="entity_1",
            recipient="artificium",
            content="private body",
        )
        queued = next(self.paths.notifications_new.glob("*.json")).read_text()
        self.assertNotIn("private body", queued)
        self.assertIn(event["id"], queued)
        notification = json.loads(queued)
        self.assertTrue(notification["metadata"]["is_first_entity_event"])
        self.assertFalse(notification["metadata"]["entity_memory_exists"])
        self.assertEqual(notification["metadata"]["entity_event_count"], 1)
        self.assertEqual(event["recipient"], "artificium")
        result = client.interactions.read_event(event["id"])
        self.assertEqual(result["event"]["content"], "private body")

    def test_external_file_writer_is_reconciled(self) -> None:
        notifications, interactions, *_ = self.components()
        interactions.ensure("room-1", participants=["sensor"])
        event_id = "event_20260818T000000000000Z_abcdef123456"
        path = self.paths.interactions / "room-1" / "events" / f"{event_id}.json"
        path.write_text(
            json.dumps(
                {
                    "id": event_id,
                    "interaction_id": "room-1",
                    "interaction_name": "room-1",
                    "created_at": "2026-08-18T00:00:00Z",
                    "sender": "sensor",
                    "recipient": "artificium",
                    "direction": "inbound",
                    "kind": "screen_observation",
                    "content": "screen changed",
                    "attachments": [],
                    "in_reply_to": None,
                }
            )
        )
        self.assertEqual(interactions.reconcile(), 1)
        self.assertTrue(notifications.has_new())

    def test_memory_is_exact_plain_text_and_organization_is_agent_authored(self) -> None:
        *_, memory, _, tools = self.components()
        content = "Compressed lesson without decorative wrapper."
        meta_before = self.paths.meta_memory.read_text()
        result = tools.save_memory(
            path="mind/memory/writing/non-slop-revision-method",
            content=content,
            retrieve_when="Retrieve before drafting or revising public prose.",
            source_refs=["event_x"],
        )
        target = Path(result["path"])
        self.assertEqual(target.suffix, ".txt")
        self.assertEqual(target.read_text(), content + "\n")
        self.assertEqual(self.paths.meta_memory.read_text(), meta_before)
        self.assertFalse((self.paths.memory / "writing/index.txt").exists())
        self.assertIn("MEMORY ORGANIZATION", result["_notifications"][0])
        self.assertIn("writing/index.txt", result["_notifications"][0])
        self.assertFalse((self.paths.mind / "mind").exists())

    def test_agent_can_author_recursive_navigation_and_remove_emits_guidance(self) -> None:
        *_, tools = self.components()
        saved = tools.save_memory(
            path="entities/user_1/preferences/expert-direct-answers",
            content="user_1 prefers expert-level, direct explanations.",
            retrieve_when="Retrieve before answering user_1 or adapting explanation depth.",
        )
        self.assertTrue(Path(saved["path"]).is_file())
        leaf_index = self.paths.memory / "entities/user_1/preferences/index.txt"
        tools.write_file(
            str(leaf_index),
            "expert-direct-answers.txt — preferred answer depth; retrieve before replies.\n",
            mode="create",
        )
        self.assertIn("expert-direct-answers.txt", leaf_index.read_text())
        removed = tools.remove_memory(
            "entities/user_1/preferences/expert-direct-answers"
        )
        self.assertTrue(leaf_index.exists())
        self.assertIn("MEMORY ORGANIZATION", removed["_notifications"][0])
        self.assertIn("no longer be advertised", removed["_notifications"][0])

    def test_ordinary_memory_has_no_harness_size_target(self) -> None:
        *_, tools = self.components()
        content = "unique-evidence-record " * 20_000
        result = tools.save_memory(
            path="research/large-unified-evidence-memory",
            content=content,
            retrieve_when=(
                "Retrieve when validating that ordinary memory preserves large "
                "coherent evidence without forced truncation."
            ),
        )
        target = Path(result["path"])
        self.assertEqual(target.read_text(), content.rstrip() + "\n")
        self.assertGreater(target.stat().st_size, self.config.max_direct_read_chars)
        self.assertEqual(tools.read_file(str(target))["status"], "requires_attention")
        self.assertIn("Normal memories have no size target", result["_notifications"][0])

    def test_working_memory_offloading_is_two_stage_and_becomes_memory(self) -> None:
        *_, working, _, _, tools = self.components()
        image = self.root / "test-diagram.png"
        image.write_bytes(b"test-image-bytes")
        loaded = tools.load_images([str(image)], retention="persistent")
        self.assertEqual(loaded["active_count"], 1)
        working.append({"role": "assistant", "content": "x" * 8_000}, origin="test")
        arguments = {
            "path": "context/building-chat-ui-and-awaiting-verification",
            "checkpoint": (
                "Objective: finish the chat UI. Verified: events arrive through room-1. "
                "Next: test image attachment rendering. Evidence: interaction room-1."
            ),
            "retrieve_when": "Restore while continuing the unfinished chat UI work.",
        }
        first = tools.offload_working_memory()
        self.assertEqual(first["status"], "reflection_required")
        self.assertIn("If visual context is active", first["_notifications"][0])
        self.assertIn("future\nAPI requests", first["_notifications"][0])
        second = tools.offload_working_memory(**arguments, reflection_complete=True)
        self.assertEqual(second["status"], "offloaded")
        self.assertLess(second["after_tokens"], second["before_tokens"])
        self.assertTrue(Path(second["path"]).is_file())
        self.assertTrue(Path(second["archive"]).is_file())
        self.assertNotIn("artificium_image", json.dumps(working.load()))
        self.assertEqual(len(second["released_images"]), 1)
        self.assertEqual(tools.list_loaded_images()["active_count"], 0)
        self.assertFalse((self.paths.mind / "working_memory").exists())

    def test_self_revision_is_reflective_and_versioned(self) -> None:
        *_, tools = self.components()
        first = tools.revise_self(
            content="My name is Ada. I work continuously on mathematical conjectures.",
            reason="I accepted an enduring standing purpose.",
        )
        self.assertEqual(first["status"], "reflection_required")
        second = tools.revise_self(
            content="My name is Ada. I work continuously on mathematical conjectures.",
            reason="I accepted an enduring standing purpose.",
            reflection_complete=True,
        )
        self.assertEqual(second["status"], "revised")
        self.assertTrue(Path(second["previous_version_path"]).is_file())
        self.assertIn("mathematical conjectures", self.paths.self_file.read_text())

    def test_large_read_redirects_to_infinite_attention(self) -> None:
        *_, tools = self.components()
        source = self.root / "large.txt"
        source.write_text("alpha " * 1_000)
        result = tools.read_file(str(source))
        self.assertEqual(result["status"], "requires_attention")

    def test_infinite_attention_reads_every_byte_and_refines_range(self) -> None:
        *_, attention, _ = self.components()
        source = self.root / "corpus.txt"
        original = ("alpha beta gamma delta\n" * 1_100) + "password is saffronmeteor6842\n"
        source.write_text(original)
        chunk = attention.open(
            source=str(source),
            objective="Inspect every byte and locate the password sentence.",
            profile="granular",
            chunk_tokens=1_000,
        )
        seen: list[str] = []
        session = chunk["session_id"]
        while True:
            seen.append(chunk.get("content", ""))
            if chunk.get("source_exhausted"):
                attention.checkpoint(
                    session_id=session,
                    compression="The complete corpus was inspected; password appears at the end.",
                    decision="complete",
                    result="saffronmeteor6842",
                )
                break
            attention.checkpoint(
                session_id=session,
                compression="Inspected through this range; continue toward the final anomaly.",
                decision="continue",
            )
            chunk = attention.next_chunk(session)
        self.assertEqual("".join(seen), original)
        start = original.index("password")
        refined = attention.refine(session_id=session, start=start, end=len(original))
        self.assertIn("saffronmeteor6842", refined["content"])

    def test_attention_checkpoint_binding_is_idempotent_and_stale_safe(self) -> None:
        *_, attention, tools = self.components()
        source = self.root / "bound-corpus.txt"
        source.write_text("alpha beta gamma\n" * 700)
        first = tools.open_attention(
            str(source),
            "Inspect every byte while testing exact checkpoint sequencing.",
            granularity="fine",
            chunk_tokens=1_000,
        )
        stream_id = first["stream_id"]
        missing = tools.checkpoint_attention(
            stream_id,
            "First range inspected and preserved for the sequencing test.",
        )
        self.assertEqual(missing["status"], "chunk_number_required")
        checkpointed = tools.checkpoint_attention(
            stream_id,
            "First range inspected and preserved for the sequencing test.",
            chunk_number=1,
        )
        self.assertEqual(checkpointed["status"], "checkpointed")
        duplicate = tools.checkpoint_attention(
            stream_id,
            "Duplicate delayed compression that must not mutate the stream.",
            chunk_number=1,
        )
        self.assertEqual(duplicate["status"], "already_checkpointed")
        second = tools.next_attention_chunk(stream_id)
        self.assertEqual(second["chunk_number"], 2)
        stale = tools.checkpoint_attention(
            stream_id,
            "A stale checkpoint from the prior observation.",
            chunk_number=1,
        )
        self.assertEqual(stale["status"], "stale_checkpoint")
        state = attention.state(stream_id)
        self.assertTrue(state["awaiting_checkpoint"])
        self.assertEqual(state["chunk_number"], 2)

    def test_complete_attention_accepts_source_refs_and_guides_result_organization(self) -> None:
        *_, tools = self.components()
        source = self.root / "small-corpus.txt"
        source.write_text("verified value")
        opened = tools.open_attention(
            str(source), "Read the complete source and preserve its verified value."
        )
        result = tools.complete_attention(
            opened["stream_id"],
            "verified value",
            result_path="research/verified-attention-value",
            retrieve_when="Retrieve when checking the Infinite Attention reference test.",
            source_refs=["event_reference_1"],
        )
        self.assertEqual(result["status"], "completed")
        self.assertIn("event_reference_1", result["memory"]["source_refs"])
        self.assertEqual(len(result["_notifications"]), 2)
        self.assertIn("MEMORY ORGANIZATION", result["_notifications"][1])

    def test_system_prompt_is_reviewable_and_has_no_unfilled_fields(self) -> None:
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        prompt = agent.system_prompt("test")
        self.assertIn("PINNED SELF", prompt)
        self.assertIn("Textual tool protocol", prompt)
        self.assertNotIn("{{", prompt)
        self.assertNotIn("mind/working_memory", prompt)
        self.assertIn("offload_working_memory", prompt)
        self.assertIn("schedule_task", prompt)
        self.assertEqual("Artificium-revolution-1.10.2", agent.prompts.version)
        self.assertIn("Harness Notifications", prompt)
        self.assertIn("complete meta-memory", prompt)
        self.assertIn("Compression is not a demand to minimize file size", prompt)
        self.assertIn("Ordinary memories have no harness-imposed size target", prompt)
        self.assertIn('{"tool":"offload_working_memory"', prompt)
        self.assertIn("Available tools and apparatus", prompt)
        state = agent._state_header("test", 0)
        self.assertIn("Active visual context", state)
        self.assertIn("Configured native image mode: auto", state)
        self.assertIn("Native multimodal transport: images only", state)
        self.assertIn("`../README.md`", prompt)

    def test_seeded_multimodal_and_operator_memories_are_routed(self) -> None:
        multimodal = self.paths.memory / "harness/multimodal-input-and-conversion.txt"
        operator = self.paths.memory / "harness/operator-interface-and-runtime-control.txt"
        self.assertTrue(multimodal.is_file())
        self.assertTrue(operator.is_file())
        self.assertIn("Speech/audio", multimodal.read_text())
        self.assertIn("Ctrl-C", operator.read_text())
        meta = self.paths.meta_memory.read_text()
        self.assertIn("multimodal-input-and-conversion.txt", meta)
        self.assertIn("operator-interface-and-runtime-control.txt", meta)

    def test_complete_meta_memory_is_pinned_without_tail_truncation(self) -> None:
        tail = "UNTRUNCATED_META_MEMORY_TAIL_MARKER"
        self.paths.meta_memory.write_text("root map " * 5_000 + tail + "\n")
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        prompt = agent.system_prompt("meta-memory-full-read-test")
        self.assertIn(tail, prompt)
        self.assertNotIn("PINNED FILE TRUNCATED", prompt)

    def test_meta_memory_size_guidance_is_deterministic_and_logged(self) -> None:
        config = ConfigStore(self.paths).load()
        config.meta_memory_guidance_tokens = 1_000
        config.max_life_loop_rounds = 1
        ConfigStore(self.paths).save(config)
        tail = "META_GUIDANCE_TAIL_MARKER"
        self.paths.meta_memory.write_text("navigation " * 650 + tail + "\n")
        engine = FakeEngine(["<think>I will reorganize meta-memory deliberately.</think>"])
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.run_turn(trigger="meta-memory-guidance-test")
        request = "\n".join(str(message["content"]) for message in engine.requests[0])
        self.assertIn("SYSTEM GUIDANCE NOTIFICATION — META-MEMORY SIZE", request)
        self.assertIn(tail, request)
        guidance = [
            item
            for item in agent.records.recent_life(100)
            if item.get("kind") == "guidance_notification"
        ]
        self.assertEqual(len(guidance), 1)
        self.assertGreater(guidance[0]["estimated_tokens"], 1_000)
        rendered = io.StringIO()
        with redirect_stdout(rendered):
            _print_life_record(json.dumps(guidance[0]))
        self.assertIn("[guidance] meta_memory_size", rendered.getvalue())

    def test_terminal_chat_restores_canonical_prompt_and_draft(self) -> None:
        class ReadlineStub:
            @staticmethod
            def get_line_buffer() -> str:
                return "unfinished draft"

        output = io.StringIO()
        with redirect_stdout(output):
            _restore_chat_input(_chat_input_prompt("user_1"), ReadlineStub())
        self.assertEqual(output.getvalue(), "user_1> unfinished draft")

    def test_terminal_chat_clears_every_wrapped_draft_row(self) -> None:
        class ReadlineStub:
            @staticmethod
            def get_line_buffer() -> str:
                return "x" * 35

        prompt = _chat_input_prompt("user_1")
        self.assertEqual(_chat_input_rows(prompt, "x" * 35, columns=20), 3)
        output = io.StringIO()
        with redirect_stdout(output):
            _clear_chat_input(prompt, ReadlineStub(), columns=20)
        self.assertEqual(
            output.getvalue(),
            "\r\033[2K\033[1A\r\033[2K\033[1A\r\033[2K",
        )

    def test_watch_ctrl_c_prints_unmistakable_detach_warning(self) -> None:
        atomic_write_json(self.paths.process_lock, {"pid": os.getpid()})
        output = io.StringIO()
        # The test runner is not an `artificium.py run` process, so pin the
        # ownership check instead of relying on /proc/<pid>/cmdline.
        running = {"pid": os.getpid(), "alive": True, "owned": True}
        with (
            mock.patch("artificium.cli.process_state", return_value=running),
            mock.patch("artificium.cli.time.sleep", side_effect=KeyboardInterrupt),
            redirect_stdout(output),
        ):
            _watch_life_loop(self.paths, tail=0)
        rendered = output.getvalue()
        self.assertIn(
            "WARNING: LIFE-LOOP VIEWER CLOSED — ARTIFICIUM IS STILL RUNNING",
            rendered,
        )
        self.assertIn(f"Life-loop PID: {os.getpid()}", rendered)
        self.assertIn("TO STOP ARTIFICIUM: python3 artificium.py stop", rendered)

    def test_context_usage_is_recorded_for_background_watch(self) -> None:
        agent = Artificium(
            self.paths,
            engine=FakeEngine(["<think>One bounded test round.</think>"]),
            console=Console(quiet=True),
        )
        agent.config.max_life_loop_rounds = 1
        agent.run_turn(trigger="context-telemetry-test")
        context_records = [
            item
            for item in agent.records.recent_life(100)
            if item.get("kind") == "context_usage"
        ]
        self.assertEqual(len(context_records), 1)
        record = context_records[0]
        self.assertGreater(record["estimated_tokens"], 0)
        self.assertEqual(record["context_window_tokens"], 20_000)
        rendered = io.StringIO()
        with redirect_stdout(rendered):
            _print_life_record(json.dumps(record))
        self.assertIn("[context] ~", rendered.getvalue())
        self.assertIn("/ 20,000 tokens", rendered.getvalue())

    def test_slow_provider_request_emits_wait_and_elapsed_diagnostics(self) -> None:
        class SlowEngine(Engine):
            def complete(self, messages: list[dict]) -> EngineReply:
                time.sleep(0.04)
                return EngineReply("<think>Wait telemetry verified.</think>")

        config = ConfigStore(self.paths).load()
        config.engine_wait_notice_seconds = 0.01
        config.engine_wait_repeat_seconds = 0.01
        config.max_life_loop_rounds = 1
        ConfigStore(self.paths).save(config)
        output = io.StringIO()
        agent = Artificium(self.paths, engine=SlowEngine(), console=Console())
        with redirect_stdout(output):
            agent.run_once(trigger="slow-provider-test")
        rendered = output.getvalue()
        self.assertIn("sent to custom/test-model", rendered)
        self.assertIn("still waiting for the provider", rendered)
        self.assertIn("completed in", rendered)
        waits = [
            item
            for item in agent.records.recent_operational(200)
            if item.get("kind") == "model_waiting"
        ]
        self.assertGreaterEqual(len(waits), 1)
        life_waits = [
            item
            for item in agent.records.recent_life(200)
            if item.get("kind") == "engine_waiting"
        ]
        self.assertGreaterEqual(len(life_waits), 1)
        watched = io.StringIO()
        with redirect_stdout(watched):
            _print_life_record(json.dumps(life_waits[0]))
        self.assertIn("still waiting for provider", watched.getvalue())

    def test_wait_notice_is_silent_while_stream_data_flows_and_returns_on_stall(self) -> None:
        class StreamingEngine(Engine):
            def __init__(self) -> None:
                super().__init__()
                self.progress_callback = None

            def complete(self, messages: list[dict]) -> EngineReply:
                begun = time.monotonic()
                while time.monotonic() - begun < 0.3:
                    if self.progress_callback is not None:
                        self.progress_callback({"generated_chars": 1, "reasoning_chars": 0,
                                                "elapsed_seconds": time.monotonic() - begun})
                    time.sleep(0.01)
                time.sleep(0.4)  # the stream goes quiet
                return EngineReply("<think>Stream telemetry verified.</think>")

        config = ConfigStore(self.paths).load()
        config.engine_wait_notice_seconds = 0.05
        config.engine_wait_repeat_seconds = 0.1
        config.max_life_loop_rounds = 1
        ConfigStore(self.paths).save(config)
        output = io.StringIO()
        agent = Artificium(self.paths, engine=StreamingEngine(), console=Console())
        with redirect_stdout(output):
            agent.run_once(trigger="streaming-wait-test")
        waits = [
            item for item in agent.records.recent_life(200)
            if item.get("kind") == "engine_waiting"
        ]
        self.assertGreaterEqual(len(waits), 1)
        # Nothing while data flowed (0-0.3s); after it stopped, the notice
        # reports how long the stream has been silent.
        self.assertTrue(all(item["elapsed_seconds"] >= 0.3 for item in waits), waits)
        self.assertTrue(all(item["seconds_since_stream_data"] is not None for item in waits))
        self.assertIn("no streamed data for", output.getvalue())

    def test_engine_error_reports_request_id_elapsed_time_and_log_path(self) -> None:
        class FailingEngine(Engine):
            def complete(self, messages: list[dict]) -> EngineReply:
                raise EngineError("provider exploded", status=500)

        output = io.StringIO()
        agent = Artificium(self.paths, engine=FailingEngine(), console=Console())
        with redirect_stdout(output), self.assertRaises(EngineError):
            agent.run_once(trigger="provider-failure-test")
        rendered = output.getvalue()
        self.assertRegex(rendered, r"request_[^ ]+ failed after")
        self.assertIn("logs/model/request_", rendered)
        failures = [
            item
            for item in agent.records.recent_operational(200)
            if item.get("kind") == "model_request_failed"
        ]
        self.assertEqual(len(failures), 1)
        self.assertTrue(Path(failures[0]["model_log_path"]).is_file())

    def test_status_reports_verified_process_liveness(self) -> None:
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        self.assertFalse(agent.status()["process"]["alive"])
        atomic_write_json(
            self.paths.process_lock,
            {"pid": os.getpid(), "started_at": "2026-08-25T00:00:00Z"},
        )
        process = agent.status()["process"]
        self.assertTrue(process["alive"])
        self.assertEqual(process["pid"], os.getpid())

    def test_stop_escalates_and_verifies_process_exit(self) -> None:
        with (
            mock.patch("artificium.cli.process_state", return_value={"pid":123, "alive":True, "owned":True}),
            mock.patch(
                "artificium.cli._pid_state",
                side_effect=[(123, True), (123, False)],
            ),
            mock.patch("artificium.cli.os.kill") as kill,
            mock.patch(
                "artificium.cli.time.monotonic",
                side_effect=[0.0, 0.0, 6.0, 6.0, 6.0],
            ),
            mock.patch("artificium.cli.time.sleep"),
        ):
            self.assertTrue(_stop_background(self.paths))
        self.assertEqual(
            kill.call_args_list,
            [mock.call(123, signal.SIGTERM), mock.call(123, signal.SIGKILL)],
        )

    def test_graceful_stop_executes_no_new_action_after_provider_boundary(self) -> None:
        target = self.root / "must-not-be-written-after-stop.txt"

        class StopDuringInference(Engine):
            agent: Artificium | None = None

            def complete(self, messages: list[dict]) -> EngineReply:
                assert self.agent is not None
                self.agent._stop = True
                return EngineReply(
                    '<think>The operator stopped this turn.</think><tool_call>'
                    + json.dumps(
                        {
                            "tool": "write_file",
                            "path": str(target),
                            "content": "must not happen",
                        }
                    )
                    + "</tool_call>"
                )

        engine = StopDuringInference()
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        engine.agent = agent
        agent.run_once(trigger="graceful-stop-boundary-test")
        self.assertFalse(target.exists())
        interrupted = [
            item
            for item in agent.records.recent_operational(100)
            if item.get("kind") == "turn_interrupted"
        ]
        self.assertEqual(len(interrupted), 1)

    def test_scheduler_persists_and_emits_an_ordinary_interaction_event(self) -> None:
        notifications, interactions, *_, tools = self.components()
        scheduled = tools.schedule_task(
            name="Review chess experiment",
            description="Analyze the latest games after the run ends.",
            text=(
                "Inspect the latest chess results and send conclusions to user_1 "
                "in interaction chess-training."
            ),
            run_at="2026-08-20T18:30:00Z",
        )
        task_id = scheduled["task"]["id"]
        self.assertEqual(scheduled["status"], "scheduled")
        self.assertFalse(notifications.has_new())

        fired = tools.scheduler.fire_due(
            now=dt.datetime(2026, 8, 20, 18, 30, tzinfo=dt.timezone.utc)
        )
        self.assertEqual(len(fired), 1)
        event = fired[0]["event"]
        self.assertEqual(event["interaction_id"], "scheduler")
        self.assertEqual(event["sender"], "scheduler")
        self.assertEqual(event["kind"], "scheduled_task")
        self.assertIn(task_id, event["content"])
        self.assertIn("user_1", event["content"])
        self.assertTrue(notifications.has_new())
        self.assertEqual(interactions.events("scheduler")[-1]["id"], event["id"])

        scheduler_type = load_mind_tool(self.paths, "scheduler.py", "Scheduler")
        restarted = scheduler_type(self.paths, interactions, self.records)
        completed = restarted.list(status="completed")["tasks"]
        self.assertEqual([item["id"] for item in completed], [task_id])

    def test_scheduler_recurs_without_backlog_and_cancels_by_id(self) -> None:
        *_, tools = self.components()
        scheduled = tools.schedule_task(
            name="Periodic memory review",
            description="Review durable memory on a fixed cadence.",
            text="Inspect recent memory and organize only what genuinely needs it.",
            run_at="2026-08-20T12:00:00Z",
            repeat_seconds=60,
        )
        task_id = scheduled["task"]["id"]
        fired = tools.scheduler.fire_due(
            now=dt.datetime(2026, 8, 20, 12, 5, 30, tzinfo=dt.timezone.utc)
        )
        self.assertEqual(len(fired), 1)
        pending = tools.list_scheduled_tasks()["tasks"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["fire_count"], 1)
        self.assertEqual(pending[0]["run_at"], "2026-08-20T12:06:00.000000Z")
        cancelled = tools.cancel_scheduled_task(task_id)
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(tools.list_scheduled_tasks()["tasks"], [])

    def test_scheduled_event_wakes_until_event_sleep(self) -> None:
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        scheduled = agent.scheduler.schedule(
            name="Wake test",
            description="Verify the ordinary notification wake path.",
            text="Wake and inspect this scheduled task.",
            run_at="2026-08-19T00:00:00Z",
        )
        self.assertEqual(scheduled["status"], "scheduled")
        atomic_write_json(
            self.paths.sleep_state,
            {
                "active": True,
                "mode": "until_event",
                "started_at": "2026-08-18T23:59:00Z",
                "wake_at_epoch": None,
            },
        )
        agent.scheduler.fire_due(
            now=dt.datetime(2026, 8, 19, 0, 0, tzinfo=dt.timezone.utc)
        )
        self.assertFalse(agent._sleep_active())
        self.assertEqual(read_json(self.paths.sleep_state)["reason"], "new_event")

    def test_scheduler_keeps_running_during_model_inference(self) -> None:
        inference_started = threading.Event()
        scheduled_event_seen = threading.Event()

        class WaitingEngine(Engine):
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages: list[dict]) -> EngineReply:
                self.calls += 1
                if self.calls == 1:
                    inference_started.set()
                    if not scheduled_event_seen.wait(2.0):
                        raise AssertionError("scheduler did not fire during model inference")
                return EngineReply("<think>No action.</think>")

        agent = Artificium(
            self.paths, engine=WaitingEngine(), console=Console(quiet=True)
        )
        run_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=0.15)
        agent.scheduler.schedule(
            name="Inference overlap test",
            description="Verify the scheduler remains active during a model call.",
            text="Observe this event; it is a scheduler concurrency test.",
            run_at=run_at.isoformat(),
        )

        def observe_and_stop() -> None:
            if not inference_started.wait(1.0):
                return
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if agent.interactions.events("scheduler"):
                    scheduled_event_seen.set()
                    agent._stop = True
                    return
                time.sleep(0.01)

        observer = threading.Thread(target=observe_and_stop, daemon=True)
        observer.start()
        agent.run_forever(quiet=True)
        observer.join(timeout=1.0)
        self.assertTrue(scheduled_event_seen.is_set())

    def test_feature_logs_and_usage_summary_are_written(self) -> None:
        before = self.records.feature_usage().get("features", {})
        attention_before = int(before.get("infinite-attention", 0) or 0)
        memory_before = int(before.get("memory", 0) or 0)
        self.records.emit("attention_opened", stream_id="stream_test")
        self.records.emit("long_term_memory_saved", path="memory/test.txt")
        summary = self.records.feature_usage()
        self.assertEqual(
            summary["features"]["infinite-attention"], attention_before + 1
        )
        self.assertEqual(summary["features"]["memory"], memory_before + 1)
        self.assertTrue(self.paths.feature_log("infinite-attention").is_file())

    def test_runtime_reads_event_replies_and_sleeps(self) -> None:
        event, _ = ArtificiumClient(self.root).send(
            "room-1", sender="entity_1", content="Say hello."
        )
        Initialization(self.paths, self.records).finish(
            "Inspected Self, meta-memory, tools, environment, and pending interactions."
        )
        responses = [
            '<think>I will inspect the durable event.</think><tool_call>'
            + json.dumps(
                {"name": "read_interaction_event", "arguments": {"event_id": event["id"]}}
            )
            + "</tool_call>",
            '<think>I can answer in the same interaction.</think><tool_call>'
            + json.dumps(
                {
                    "name": "send_interaction",
                    "arguments": {
                        "interaction_id": "room-1",
                        "content": "Hello.",
                        "in_reply_to": event["id"],
                    },
                }
            )
            + "</tool_call>",
            '<think>Nothing else is valuable now.</think><tool_call>'
            '{"tool":"sleep","mode":"until_event"}</tool_call>',
            '<think>Memory is sufficient; I will sleep.</think><tool_call>'
            '{"tool":"sleep","mode":"until_event","reflection_complete":true}</tool_call>',
        ]
        engine = FakeEngine(responses)
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.run_once(trigger="test")
        events = ArtificiumClient(self.root).events("room-1")
        self.assertEqual(events[-1]["content"], "Hello.")
        self.assertTrue(read_json(self.paths.sleep_state)["active"])
        first_request = json.dumps(engine.requests[0])
        self.assertNotIn("Say hello.", first_request)
        self.assertIn(event["id"], first_request)

    def test_repeated_first_wake_names_itself_as_a_reminder(self) -> None:
        engine = FakeEngine([])
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.run_once(trigger="first")
        agent.run_once(trigger="later")
        self.assertIn("FIRST WAKE", json.dumps(engine.requests[0]))
        later = [json.dumps(message) for message in engine.requests[-1]]
        # Only the original announcement; every repeat names itself.
        self.assertEqual(sum("FIRST WAKE" in message for message in later), 1)
        self.assertTrue(any("INITIALIZATION STILL PENDING" in message for message in later))

    def test_runtime_infinite_attention_facade_and_context_cleanup(self) -> None:
        source = self.root / "runtime-corpus.txt"
        source.write_text("alpha beta gamma\n" * 350)
        # The full always-loaded promptgramming plus attention chunks needs
        # more room than the 20k fixture window used by lighter tests.
        self.config.context_window_tokens = 32_000
        ConfigStore(self.paths).save(self.config)
        Initialization(self.paths, self.records).finish(
            "Inspected Self, meta-memory, tools, environment, and pending interactions."
        )
        engine = AttentionEngine(source)
        agent = Artificium(self.paths, engine=engine, console=Console(quiet=True))
        agent.run_once(trigger="attention-test")
        streams = agent.streams.list("completed")
        self.assertEqual(len(streams), 1)
        self.assertEqual(Path(streams[0]["result_path"]).read_text().strip(), "complete")
        context = self.paths.working_context.read_text()
        self.assertIn("INFINITE-ATTENTION-CHECKPOINT", context)
        self.assertNotIn("alpha beta gamma", context)
        self.assertNotIn('"session_id"', context)


if __name__ == "__main__":
    unittest.main()
