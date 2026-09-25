from __future__ import annotations

import hashlib
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from . import _bootstrap
from .config import ConfigStore, SecretsStore
from .engine import Engine, EngineError, EngineReply, PreparedRequest, make_engine
from .context_budget import (TokenCount, available_output, check_input, context_exhausted,
                             measure, minimum_generation_room)
from .filesystem import (
    Paths,
    atomic_write_json,
    atomic_write_text,
    json_dumps,
    read_json,
    read_jsonl,
    sortable_id,
    utc_now,
)
from .initialization import Initialization, initialize_mind
from .interactions import InteractionStore, Notification, NotificationStore
from .life_loop import ToolIntent, parse_life_loop_output, render_normalized_life_loop_output
from .memory import InfiniteAttention, LongTermMemory, TokenEstimator, WorkingMemory
from .prompts import PromptPack
from .records import Console, Records
from .recovery import MAX_RECOVERY_ATTEMPTS, repairable, summarize
from .tool_loader import load_mind_tool
from .tools import ToolExecution, ToolRegistry
from .vision import VisualContext


class ProcessLock:
    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def _alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def __enter__(self) -> "ProcessLock":
        existing = read_json(self.path, {})
        if isinstance(existing, dict):
            pid = int(existing.get("pid", 0) or 0)
            if pid and pid != os.getpid() and self._alive(pid):
                raise RuntimeError(f"Artificium is already running as PID {pid}")
        atomic_write_json(self.path, {"pid": os.getpid(), "started_at": utc_now()})
        return self

    def __exit__(self, *_: object) -> None:
        existing = read_json(self.path, {})
        if isinstance(existing, dict) and int(existing.get("pid", 0) or 0) == os.getpid():
            self.path.unlink(missing_ok=True)


class Artificium:
    """Persistent provider-neutral life-loop for Artificium-revolution."""

    VERSION = "1.9.3-revolution"

    def __init__(
        self,
        paths: Paths,
        *,
        engine: Engine | None = None,
        console: Console | None = None,
    ):
        self.paths = paths
        self.paths.ensure_layout()
        self.prompts = PromptPack(paths)
        self.store = ConfigStore(paths)
        self.config = self.store.load()
        self.secrets = SecretsStore(paths)
        self.api_key = self.secrets.resolve_api_key(self.config)
        self.engine = engine or make_engine(self.config, self.api_key)
        self.console = console or Console()
        self.records = Records(paths)
        self.prompt_fingerprints = self.prompts.fingerprints()
        self.prompt_pack_sha256 = hashlib.sha256(
            json_dumps(self.prompt_fingerprints).encode("utf-8")
        ).hexdigest()
        self.records.emit(
            "prompt_pack_loaded",
            version=self.prompts.version,
            sha256=self.prompt_pack_sha256,
            files=self.prompt_fingerprints,
            overridden=self.prompts.overridden(),
        )
        initialize_mind(paths, self.records)
        self.notifications = NotificationStore(paths, self.records)
        self.notifications.recover()
        self.interactions = InteractionStore(paths, self.notifications, self.records)
        scheduler_type = load_mind_tool(paths, "scheduler.py", "Scheduler")
        self.scheduler = scheduler_type(paths, self.interactions, self.records)
        self.estimator = TokenEstimator(self.config.chars_per_token)
        self.working = WorkingMemory(paths, self.config, self.estimator, self.records)
        self.visual = VisualContext(paths, self.config, self.records)
        self.memory = LongTermMemory(paths, self.records)
        self.streams = InfiniteAttention(paths, self.config, self.estimator, self.records)
        self.initialization = Initialization(paths, self.records)
        self.tools = ToolRegistry(
            paths=paths,
            config=self.config,
            prompts=self.prompts,
            records=self.records,
            console=self.console,
            interactions=self.interactions,
            memory=self.memory,
            working=self.working,
            streams=self.streams,
            visual=self.visual,
            initialization=self.initialization,
            scheduler=self.scheduler,
        )
        if not self.config.mandatory_offload:
            control = self.tools._control()
            if control.pop("mandatory_offload_pending", None):
                self.tools._save_control(control)
        self._key_signature = self._secret_signature()
        self._stop = False
        self._request_blocked = False
        # A start that reaches this point has imported and wired the whole
        # harness, overlay included, so the failed-start counter can reset.
        _bootstrap.clear_attempts(self.paths.root)

    def _secret_signature(self) -> tuple[int, int] | None:
        try:
            stat = self.paths.secrets.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _refresh_key(self) -> None:
        signature = self._secret_signature()
        if signature == self._key_signature:
            return
        self.api_key = self.secrets.resolve_api_key(self.config)
        self.engine = make_engine(self.config, self.api_key)
        self._key_signature = signature
        self.records.emit("api_key_reloaded", available=bool(self.api_key))
        self.console.line("engine", "API credentials reloaded")

    def _read_pinned(self, path: Path, limit: int) -> str:
        if not path.is_file():
            return "(missing)"
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) <= limit:
            return content.rstrip()
        return (
            content[:limit].rstrip()
            + f"\n\n[PINNED FILE TRUNCATED: {len(content):,} characters; inspect {path}]"
        )

    @staticmethod
    def _read_full(path: Path) -> str:
        if not path.is_file():
            return "(missing)"
        return path.read_text(encoding="utf-8", errors="replace").rstrip()

    def _meta_memory_metrics(self) -> dict[str, int]:
        content = self._read_full(self.paths.meta_memory)
        return {
            "tokens": self.estimator.text(content),
            "words": len(content.split()),
            "characters": len(content),
        }

    def _active_stream(self) -> str:
        active = self.streams.list("active")
        paused = self.streams.list("paused")
        available = active or paused
        return str(available[-1]["id"]) if available else "none"

    def _last_checkpoint(self) -> str:
        events = self.records.recent_life(200)
        for item in reversed(events):
            if item.get("kind") in {
                "working_memory_offloaded",
                "context_compacted",
            } and item.get("path"):
                return str(item["path"])
        return "none"

    def _prompt_overhead(self) -> int:
        self_text = self._read_pinned(self.paths.self_file, 50_000)
        meta = self._read_full(self.paths.meta_memory)
        text = "\n".join(
            [
                self.prompts.always(),
                self.prompts.tool_catalog(),
                self.prompts.runtime(
                    "pinned_mind",
                    self_content=self_text,
                    meta_memory_content=meta,
                ),
            ]
        )
        return self.estimator.text(text)

    def _state_header(self, wake_reason: str, context_tokens: int) -> str:
        percent = context_tokens / self.config.working_memory_limit * 100
        if percent >= self.config.context_hard_fraction * 100:
            status = "hard pressure"
        elif percent >= self.config.context_soft_fraction * 100:
            status = "soft pressure"
        else:
            status = "normal"
        pressure = read_json(self.paths.pressure_state, {})
        prior = int(pressure.get("tokens", 0) or 0) if isinstance(pressure, dict) else 0
        control = read_json(self.tools.control_path, {})
        pre_sleep = bool(control.get("sleep_reflection_pending")) if isinstance(control, dict) else False
        pending = self.interactions.pending_events(1_000)
        meta_metrics = self._meta_memory_metrics()
        visual_images = self.visual.list()
        vision_guidance = {
            "no": (
                "Native image transport is disabled for this engine. Image paths remain "
                "durable and inert; use or build OCR, computer-vision, or conversion "
                "apparatus when visual evidence matters."
            ),
            "yes": (
                "Native image transport is enabled. Call load_images deliberately; "
                "prefer one-shot retention unless repeated viewing is necessary."
            ),
            "auto": (
                "Native image transport will be attempted. If the provider rejects it, "
                "Artificium releases active images, records the failure, and continues "
                "without visual blocks so you can choose a conversion fallback."
            ),
        }[self.config.vision]
        interaction_focus = "none"
        if pending and pending[0].get("event_path"):
            try:
                interaction_focus = Path(str(pending[0]["event_path"])).parents[1].name
            except IndexError:
                interaction_focus = "unknown"
        return self.prompts.runtime(
            "state_header",
            timestamp=utc_now(),
            wake_reason=wake_reason,
            engine_name=f"{self.config.provider}/{self.config.model}",
            context_tokens=context_tokens,
            context_window_tokens=self.config.context_window_tokens,
            working_memory_tokens=self.config.working_memory_limit,
            context_percent=f"{percent:.1f}",
            tokens_since_last_notice=max(0, context_tokens - prior),
            context_status=status,
            current_interaction_or_none=interaction_focus,
            pending_event_count=len(pending),
            active_stream_or_none=self._active_stream(),
            pre_sleep_issued=pre_sleep,
            last_checkpoint_or_none=self._last_checkpoint(),
            meta_memory_tokens=meta_metrics["tokens"],
            meta_memory_words=meta_metrics["words"],
            meta_memory_guidance_tokens=self.config.meta_memory_guidance_tokens,
            vision_mode=self.config.vision,
            vision_guidance=vision_guidance,
            active_image_count=len(visual_images),
            active_images_or_none=(
                [
                    {
                        "id": item.get("id"),
                        "path": item.get("path"),
                        "retention": item.get("retention"),
                    }
                    for item in visual_images
                ]
                or "none"
            ),
        )

    def system_prompt(self, wake_reason: str = "runtime") -> str:
        self_text = self._read_pinned(self.paths.self_file, 50_000)
        meta = self._read_full(self.paths.meta_memory)
        pinned = self.prompts.runtime(
            "pinned_mind", self_content=self_text, meta_memory_content=meta
        )
        return "\n\n---\n\n".join(
            [
                self.prompts.always(),
                pinned,
                self.prompts.tool_catalog(),
            ]
        )

    def _runtime_batch(self, records: list[str]) -> str:
        return self.prompts.runtime(
            "input_batch", runtime_records="\n\n---\n\n".join(records)
        )

    def _request_messages(
        self, inputs: list[str], *, wake_reason: str,
        working: list[dict[str, Any]] | None = None, include_images: bool = True,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()}
        ]
        messages.extend(self.working.load() if working is None else working)
        if inputs:
            messages.append(
                {
                    "role": self.config.runtime_message_role,
                    "content": self._runtime_batch(inputs),
                    "_artificium": {"kind": "runtime_input"},
                }
            )
        visual_message = self.visual.request_message()
        if include_images and visual_message is not None:
            messages.append(visual_message)
        visual_tokens = (
            self.estimator.messages([visual_message]) if visual_message is not None else 0
        )
        estimated = self.working.estimated_tokens(self._prompt_overhead() + visual_tokens)
        messages.append(
            {
                "role": self.config.runtime_message_role,
                "content": self._state_header(wake_reason, estimated),
            }
        )
        return messages

    @staticmethod
    def _contains_image(messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(part, dict) and part.get("type") == "artificium_image"
                for part in content
            ):
                return True
        return False

    def _downgrade_context_images(self, reason: str) -> int:
        active = self.visual.list()
        if not active:
            return 0
        released = self.visual.release(
            all_images=True,
            reason="engine_vision_fallback",
        )
        paths = [item.get("path") for item in released["released"]]
        self._append_runtime(
            [
                self.prompts.event(
                    "external_event",
                    event_type="vision_fallback",
                    created_at=utc_now(),
                    source="engine_adapter",
                    summary=(
                        "The engine rejected visual input. Active images were released; "
                        "use or build OCR or another image-processing tool if they matter."
                    ),
                    path_or_none=paths or "none",
                    metadata_or_none={"reason": reason, "released_images": active},
                )
            ],
            origin="vision_fallback",
        )
        self.records.emit("images_downgraded", count=len(active), reason=reason, paths=paths)
        return len(active)

    def _complete(
        self, messages: list[dict[str, Any]], *, wake_reason: str,
        pending_inputs: list[str] | None = None,
        prepared: PreparedRequest | None = None, count: TokenCount | None = None,
    ) -> tuple[str, EngineReply]:
        request_id = sortable_id("request_")
        request_log_path = self.paths.model_log / f"{request_id}.json"
        prepared = prepared or self.engine.prepare(messages)
        count = count or measure(self.engine, prepared, messages, self.estimator)
        check_input(self.config, count)
        estimated = count.tokens
        request_parameters = prepared.safe_summary()
        request_parameters["token_count"] = {"tokens": count.tokens, "source": count.source}
        fingerprints = self.prompts.fingerprints()
        pack_sha256 = hashlib.sha256(json_dumps(fingerprints).encode("utf-8")).hexdigest()
        if pack_sha256 != self.prompt_pack_sha256:
            self.prompt_fingerprints = fingerprints
            self.prompt_pack_sha256 = pack_sha256
            self.records.emit(
                "prompt_pack_reloaded",
                version=self.prompts.version,
                sha256=pack_sha256,
                files=fingerprints,
            )
        self.records.emit(
            "model_request",
            request_id=request_id,
            estimated_tokens=estimated,
            token_count_source=count.source,
            message_count=len(messages),
            prompt_pack=self.prompts.version,
            prompt_pack_sha256=self.prompt_pack_sha256,
            engine_request=request_parameters,
        )
        self.records.life(
            "engine_request",
            request_id=request_id,
            estimated_tokens=estimated,
            provider=self.config.provider,
            model=self.config.model,
            adapter=self.config.adapter,
            token_count_source=count.source,
            request_parameters=request_parameters.get("body"),
        )
        self.console.line(
            "engine",
            (
                f"request {request_id} sent to {self.config.provider}/{self.config.model} "
                f"({estimated:,} input tokens, {count.source})"
            ),
        )
        self.console.line(
            "engine",
            "serialized parameters: "
            + json_dumps(request_parameters.get("body") or {}),
            detail=True,
        )
        started = time.monotonic()
        wait_stop = threading.Event()

        def report_wait() -> None:
            if wait_stop.wait(self.config.engine_wait_notice_seconds):
                return
            while not wait_stop.is_set():
                elapsed = time.monotonic() - started
                self.records.emit(
                    "model_waiting",
                    request_id=request_id,
                    elapsed_seconds=round(elapsed, 3),
                    provider=self.config.provider,
                    model=self.config.model,
                )
                self.records.life(
                    "engine_waiting",
                    request_id=request_id,
                    elapsed_seconds=round(elapsed, 3),
                    provider=self.config.provider,
                    model=self.config.model,
                )
                self.console.line(
                    "engine",
                    (
                        f"request {request_id} is still waiting for the provider "
                        f"({elapsed:.0f}s elapsed)"
                    ),
                )
                if wait_stop.wait(self.config.engine_wait_repeat_seconds):
                    return

        wait_thread = threading.Thread(
            target=report_wait,
            name=f"artificium-engine-wait-{request_id}",
            daemon=True,
        )
        wait_thread.start()
        try:
            reply = self.engine.complete_prepared(prepared)
        except KeyboardInterrupt:
            elapsed = time.monotonic() - started
            self.records.model_exchange(
                request_id=request_id,
                messages=messages,
                request_parameters=request_parameters,
                error="KeyboardInterrupt: operator forced request termination",
            )
            self.records.emit(
                "model_request_cancelled",
                request_id=request_id,
                duration_seconds=round(elapsed, 3),
                model_log_path=str(request_log_path),
            )
            self.records.life(
                "engine_request_cancelled",
                request_id=request_id,
                duration_seconds=round(elapsed, 3),
                model_log_path=str(request_log_path),
            )
            self.console.error(
                f"engine request {request_id} was force-stopped after {elapsed:.1f}s; "
                f"request evidence: {request_log_path}"
            )
            raise
        except EngineError as exc:
            exc.request_log_path = str(request_log_path)
            elapsed = time.monotonic() - started
            self.records.model_exchange(
                request_id=request_id,
                messages=messages,
                request_parameters=request_parameters,
                error=str(exc),
                response=vars(exc.reply) if exc.reply is not None else None,
            )
            self.records.emit(
                "model_request_failed",
                request_id=request_id,
                duration_seconds=round(elapsed, 3),
                error=f"{type(exc).__name__}: {exc}",
                model_log_path=str(request_log_path),
            )
            self.records.life(
                "engine_request_failed",
                request_id=request_id,
                duration_seconds=round(elapsed, 3),
                error=f"{type(exc).__name__}: {exc}",
                hint=exc.hint,
                http_status=exc.status,
                error_kind=exc.kind,
                model_log_path=str(request_log_path),
            )
            self.console.error(
                f"engine request {request_id} failed after {elapsed:.1f}s; "
                f"details: {request_log_path}; {type(exc).__name__}: {exc}"
            )
            if (
                self.config.vision == "auto"
                and self._contains_image(messages)
                and exc.status in {400, 413, 415, 422}
                and exc.image_input_unsupported
                and self._downgrade_context_images(str(exc))
            ):
                return self._complete(
                    self._request_messages(pending_inputs or [], wake_reason="vision_fallback"),
                    wake_reason="vision_fallback",
                    pending_inputs=pending_inputs,
                )
            raise
        finally:
            wait_stop.set()
            wait_thread.join(timeout=0.2)
        elapsed = time.monotonic() - started
        self.records.model_exchange(
            request_id=request_id,
            messages=messages,
            request_parameters=request_parameters,
            response={
                "content": reply.content,
                "usage": reply.usage,
                "finish_reason": reply.finish_reason,
                "provider_reasoning": reply.provider_reasoning,
                "raw": reply.raw,
            },
        )
        self.records.emit(
            "model_response",
            request_id=request_id,
            content=reply.content,
            usage=reply.usage,
            finish_reason=reply.finish_reason,
            duration_seconds=round(elapsed, 3),
            model_log_path=str(request_log_path),
        )
        self.records.life(
            "engine_response",
            request_id=request_id,
            duration_seconds=round(elapsed, 3),
            usage=reply.usage,
            finish_reason=reply.finish_reason,
            model_log_path=str(request_log_path),
        )
        self.console.line(
            "engine",
            f"request {request_id} completed in {elapsed:.1f}s",
        )
        return request_id, reply

    def _append_runtime(self, records: list[str], *, origin: str) -> None:
        if not records:
            return
        self.working.append(
            {
                "role": self.config.runtime_message_role,
                "content": self._runtime_batch(records),
                "_artificium": {"kind": "runtime_input"},
            },
            origin=origin,
        )

    def _append_tool_result(self, execution: ToolExecution) -> None:
        result = execution.result
        visible_result = dict(result)
        bounded = json_dumps(visible_result, pretty=True)
        target = next(
            (
                result.get(key)
                for key in ("path", "result_path", "memory_path", "primary_target")
                if result.get(key)
            ),
            "none",
        )
        rendered = self.prompts.runtime(
            "tool_result",
            call_id=execution.tool_id,
            tool_name=execution.name,
            status=result.get("status", "ok"),
            started_at=execution.started_at,
            finished_at=execution.finished_at,
            primary_target_or_none=target,
            output_characters=len(bounded),
            output_truncated=bool(result.get("output_truncated")),
            full_output_path_or_none=result.get("full_output_path") or "none",
            bounded_output=bounded,
        )
        self.working.append(
            {
                "role": self.config.runtime_message_role,
                "content": rendered,
                "_artificium": {
                    "kind": "tool_result",
                    "tool": execution.name,
                    "tool_id": execution.tool_id,
                    "stream_id": result.get("stream_id"),
                },
            },
            origin=f"tool_result:{execution.name}",
        )

    def _render_notification(self, item: Notification) -> str:
        metadata = item.metadata
        if item.type == "interaction_event":
            return self.prompts.event(
                "interaction_arrived",
                event_kind=metadata.get("event_kind", "message"),
                sender=metadata.get("entity_id", item.source),
                interaction_id=metadata.get("interaction_id"),
                event_id=metadata.get("event_id"),
                recipient=metadata.get("recipient", self.config.instance_id),
                created_at=metadata.get("created_at", item.created_at),
                in_reply_to_or_none=metadata.get("in_reply_to") or "none",
                event_path=item.path,
                attachment_count=metadata.get("attachment_count", 0),
                entity_event_count=metadata.get("entity_event_count", 1),
                entity_memory_exists=metadata.get("entity_memory_exists", False),
                is_first_entity_event=metadata.get("is_first_entity_event", False),
                memory_hints_or_none={
                    "possible_entity_memory": metadata.get("possible_entity_memory"),
                    "previous_interactions": metadata.get("previous_interactions", []),
                },
            )
        if item.type == "wake":
            return self.prompts.event(
                "wake",
                wake_reason=metadata.get("reason", item.summary),
                sleep_started_at=metadata.get("sleep_started_at", "unknown"),
                timestamp=item.created_at,
                pending_event_count=len(self.interactions.pending_events(1_000)),
                timed_sleep_interrupted=metadata.get("timed_sleep_interrupted", False),
            )
        if item.type == "recovery":
            reason = metadata.get("reason", item.summary)
            if metadata.get("failed_attempts"):
                reason = f"{metadata['failed_attempts']} failed attempts; latest error: {reason}"
            return self.prompts.event(
                "recovery",
                recovery_reason=reason,
                last_confirmed_action_or_none=metadata.get("last_confirmed_action") or "none",
                uncertain_action_or_none=metadata.get("uncertain_action") or "none",
                last_checkpoint_or_none=self._last_checkpoint(),
                pending_event_count=len(self.interactions.pending_events(1_000)),
                recovery_paths=metadata.get("paths") or [str(self.paths.lifetime_log)],
            )
        if item.type == "async_output_ready":
            return self.prompts.event(
                "async_output_ready",
                action_name=metadata.get("action_name", item.source),
                request_id=metadata.get("request_id", item.id),
                completed_at=item.created_at,
                status=metadata.get("status", "complete"),
                result_path=item.path or "none",
                interaction_id_or_none=metadata.get("interaction_id") or "none",
                objective_or_none=metadata.get("objective") or "none",
            )
        return self.prompts.event(
            "external_event",
            event_type=item.type,
            created_at=item.created_at,
            source=item.source,
            summary=item.summary,
            path_or_none=item.path or "none",
            metadata_or_none=metadata or "none",
        )

    def _claim_notifications(self) -> list[Notification]:
        self.interactions.reconcile()
        return self.notifications.claim(
            self.config.notification_batch_size, include_runtime_notices=True
        )

    def _render_notifications(self, claimed: list[Notification]) -> list[str]:
        notices = [item for item in claimed if item.is_runtime_notice]
        if len(notices) < 2:
            return [self._render_notification(item) for item in claimed]
        ordered = sorted(notices, key=lambda item: (item.created_at, item.id))
        latest = {item.type: item for item in ordered}
        summary = self.prompts.event(
            "runtime_notice_summary",
            notice_count=len(notices),
            wake_count=sum(item.type == "wake" for item in notices),
            recovery_count=sum(item.type == "recovery" for item in notices),
            first_notice_at=ordered[0].created_at,
            last_notice_at=ordered[-1].created_at,
            latest_wake=self._render_notification(latest["wake"]) if "wake" in latest else "none",
            latest_recovery=self._render_notification(latest["recovery"]) if "recovery" in latest else "none",
            notice_archive_path=str(self.paths.notifications_delivered),
        )
        rendered: list[str] = []
        summary_added = False
        for item in claimed:
            if item.is_runtime_notice:
                if not summary_added:
                    rendered.append(summary)
                    summary_added = True
            else:
                rendered.append(self._render_notification(item))
        return rendered

    def _pending_recovery(self) -> dict[str, Any] | None:
        state = read_json(self.paths.sleep_state, {})
        recovery = state.get("recovery")
        return recovery if not state.get("active") and isinstance(recovery, dict) else None

    def _acknowledge_recovery(self, recovery: dict[str, Any] | None) -> None:
        if recovery is None:
            return
        state = read_json(self.paths.sleep_state, {})
        if state.get("recovery") == recovery:
            state.pop("recovery")
            state["recovered_at"] = utc_now()
            atomic_write_json(self.paths.sleep_state, state)

    def _mandatory_offload_required(self, estimated: int, *, generation_room: int | None = None) -> bool:
        if not self.config.mandatory_offload:
            return False
        state = self.tools._control()
        pending = state.get("mandatory_offload_pending")
        if not pending and (
            estimated >= self.config.working_memory_limit * self.config.offload_threshold_percent / 100
            or (generation_room is not None and generation_room < minimum_generation_room(self.config))
        ):
            state["mandatory_offload_pending"] = {"requested_at": utc_now(), "estimated_tokens": estimated}
            self.tools._save_control(state)
            self.records.emit("mandatory_offload_required", estimated_tokens=estimated,
                              threshold_percent=self.config.offload_threshold_percent)
            pending = True
        return bool(pending)

    @property
    def _repair_state_path(self) -> Path:
        return self.paths.runtime / "request-repair.json"

    def _repair_request(self, error: EngineError, inputs: list[str], *, context_error: bool) -> bool:
        """Try earlier successful request inputs, never replay filesystem actions.

        Existing assistant records supply the boundaries. Only a failed request
        creates a recovery archive; normal execution needs no extra checkpoint.
        """
        state = read_json(self._repair_state_path, {})
        if not state:
            archive = self.paths.context_archive / f"{sortable_id('repair_')}.jsonl"
            atomic_write_text(archive, "".join(json_dumps(m) + "\n" for m in self.working.load()))
            state = {"attempts": 0, "archive": str(archive),
                     "original_log": getattr(error, "request_log_path", None) or str(self.paths.model_log)}
        history = read_jsonl(Path(state["archive"]))
        boundaries = [i for i, message in enumerate(history)
                      if message.get("_artificium", {}).get("kind") == "life_loop_output"]
        last_error = str(error)
        while int(state.get("attempts", 0)) < MAX_RECOVERY_ATTEMPTS and not self._stop:
            attempt = int(state.get("attempts", 0)) + 1
            state.update(attempts=attempt, error=last_error, started_at=utc_now())
            atomic_write_json(self._repair_state_path, state)
            self.console.line("recovery", f"Request repair {attempt}/{MAX_RECOVERY_ATTEMPTS}")
            self.records.life("request_repair_started", **state)
            before = self.working.load()
            try:
                # The final context-repair attempt uses the independent summary
                # helper. It can also be used when no earlier request remains.
                use_summary = context_error and (attempt == MAX_RECOVERY_ATTEMPTS or attempt > len(boundaries))
                cut = boundaries[-attempt] if attempt <= len(boundaries) else 0
                restored = history[:cut]
                summary, omitted = "", False
                if use_summary:
                    summary, omitted = summarize(
                        config=self.config, api_key=self.api_key, history=history,
                        original_log=state["original_log"], attempt=attempt,
                        prompts=self.prompts, records=self.records,
                    )
                    restored = []
                if self._stop:
                    return False
                # Keep a compact pointer to later actions, not their potentially
                # malformed/oversized output. Their full records stay in archive.
                tools = sorted({str(m.get("_artificium", {}).get("tool")) for m in history[cut:]
                                if m.get("_artificium", {}).get("kind") == "tool_result"})
                release_images = context_error or error.kind in {"vision", "tokenization"}
                notice = self.prompts.event(
                    "request_repair", error=last_error, original_log=state["original_log"],
                    archive=state["archive"], attempt=attempt,
                    method="isolated summary" if use_summary else "earlier successful request context",
                    later_tools=", ".join(tools) or "see archive",
                    images_released=release_images, omitted=omitted,
                    continuation=summary or ("Offload working memory before resuming the task." if context_error
                                            else "Inspect what changed and correct the input before continuing."),
                )
                restored = [*restored, {"role": self.config.runtime_message_role, "content": notice}]
                candidate = self._request_messages(inputs, wake_reason="request_repair", working=restored,
                                                   include_images=not release_images)
                prepared = self.engine.prepare(candidate)
                count = measure(self.engine, prepared, candidate, self.estimator)
                check_input(self.config, count)
                if use_summary and count.tokens >= self.config.working_memory_limit * self.config.offload_threshold_percent / 100:
                    raise EngineError("The rebuilt request still leaves too little room to continue. Pinned instructions or pending input may be too large.", kind="context")
                if before != self.working.load():
                    raise EngineError("Working context changed during recovery; refusing to overwrite it.", kind="context")
            except EngineError as exc:
                last_error = str(exc)
                context_error = context_error or exc.kind == "context"
                state["error"] = last_error
                atomic_write_json(self._repair_state_path, state)
                self.records.life("request_repair_failed", attempt=attempt, error=last_error)
                continue
            atomic_write_text(self.paths.working_context, "".join(json_dumps(m) + "\n" for m in restored))
            if release_images:
                self.visual.release(all_images=True, reason="request_repair")
            if use_summary:
                control = self.tools._control()
                for key in ("working_memory_offload_pending", "mandatory_offload_pending", "compaction_pending"):
                    control.pop(key, None)
                self.tools._save_control(control)
            self.records.life("request_repair_ready", method="summary" if use_summary else "earlier context", **state)
            return True
        if self._stop:
            return False
        raise EngineError(
            f"Automatic repair stopped after {state.get('attempts', 0)} attempts: {last_error}",
            kind="repair", hint=f"History is retained in {state['archive']}; incoming messages and files are kept. Inspect the logs and correct the cause before restarting.",
        )

    def _offload_tool_error(self, intent: ToolIntent, call_index: int) -> dict[str, Any] | None:
        allowed = intent.name in {
            "offload_working_memory", "compact_context", "save_memory", "search_memory",
            "remove_memory", "read_file", "list_directory", "list_loaded_images", "release_images",
        }
        if intent.name == "write_file":
            try:
                path = self.tools._resolve(str(intent.arguments.get("path", "")))
                allowed = path == self.paths.meta_memory or path.is_relative_to(self.paths.memory)
            except (ValueError, OSError):
                allowed = False
        if allowed:
            return None
        return {
            "call_index": call_index, "tool": intent.name,
            "error": "Mandatory offloading is active. Ordinary actions are withheld until offload_working_memory completes. Preserve learning and the continuation first.",
            "suggested_tool": "offload_working_memory",
            "accepted_arguments": self.tools.accepted_arguments("offload_working_memory"),
            "received_arguments": intent.arguments,
            "canonical_example": {"tool": "offload_working_memory"},
        }

    def _context_events(self) -> list[str]:
        overhead = self._prompt_overhead()
        notice = self.working.pressure_notice(overhead)
        total = self.working.estimated_tokens(overhead)
        fraction = total / self.config.working_memory_limit
        result: list[str] = []
        if notice:
            milestone = int(notice["milestone"])
            result.append(
                self.prompts.event(
                    "context_milestone",
                    milestone_tokens=self.config.context_reminder_tokens,
                    context_tokens=total,
                    context_window_tokens=self.config.working_memory_limit,
                    context_percent=f"{fraction * 100:.1f}",
                    next_milestone_tokens=milestone + self.config.context_reminder_tokens,
                )
            )
        if fraction >= self.config.context_soft_fraction and (
            notice or fraction >= self.config.context_hard_fraction
        ):
            result.append(
                self.prompts.event(
                    "context_pressure",
                    context_tokens=total,
                    context_window_tokens=self.config.working_memory_limit,
                    context_percent=f"{fraction * 100:.1f}",
                    soft_threshold_percent=f"{self.config.context_soft_fraction * 100:.0f}",
                    hard_threshold_percent=f"{self.config.context_hard_fraction * 100:.0f}",
                )
            )
        return result

    def _meta_memory_guidance(self, *, turn_id: str) -> str | None:
        metrics = self._meta_memory_metrics()
        threshold = self.config.meta_memory_guidance_tokens
        if metrics["tokens"] <= threshold:
            return None
        data = {
            "guidance_type": "meta_memory_size",
            "meta_memory_path": str(self.paths.meta_memory),
            "estimated_tokens": metrics["tokens"],
            "words": metrics["words"],
            "characters": metrics["characters"],
            "guidance_threshold_tokens": threshold,
        }
        self.records.emit("guidance_notification_issued", turn_id=turn_id, **data)
        self.records.life("guidance_notification", turn_id=turn_id, **data)
        return self.prompts.event(
            "meta_memory_pressure",
            meta_memory_path=str(self.paths.meta_memory),
            meta_memory_tokens=metrics["tokens"],
            meta_memory_words=metrics["words"],
            meta_memory_characters=metrics["characters"],
            guidance_threshold_tokens=threshold,
        )

    def _schedule_sleep(self) -> None:
        request = self.tools.sleep_request
        if not request:
            return
        wake_at = (
            time.time() + float(request.seconds or 0)
            if request.mode == "timed"
            else None
        )
        atomic_write_json(
            self.paths.sleep_state,
            {
                "active": True,
                "mode": request.mode,
                "seconds": request.seconds,
                "started_at": utc_now(),
                "wake_at_epoch": wake_at,
            },
        )
        self.records.emit("sleep_started", mode=request.mode, seconds=request.seconds)
        self.tools.sleep_request = None

    def _sleep_active(self) -> bool:
        state = read_json(self.paths.sleep_state, {})
        if not isinstance(state, dict) or not state.get("active"):
            return False
        if state.get("reason") == "engine_error_backoff":
            # Unconsumed messages must not wake their own failed request. Keep
            # one durable recovery record until a later inference accepts it.
            if time.time() < float(state.get("wake_at_epoch", 0)):
                return True
            state.update(active=False, woke_at=utc_now())
            atomic_write_json(self.paths.sleep_state, state)
            self.records.emit("sleep_ended", reason="engine_retry_timer")
            return False
        reason: str | None = None
        interrupted = False
        if self.notifications.has_new():
            reason = "new_event"
            interrupted = state.get("mode") == "timed"
        elif state.get("mode") == "timed" and time.time() >= float(state.get("wake_at_epoch", 0)):
            reason = "timer"
        if reason is None:
            return True
        woke = utc_now()
        state.update({"active": False, "woke_at": woke, "reason": reason})
        atomic_write_json(self.paths.sleep_state, state)
        self.records.emit("sleep_ended", reason=reason)
        self.notifications.create(
            type="wake",
            source="artificium_runtime",
            summary=f"Sleep ended because of {reason}.",
            metadata={
                "reason": reason,
                "sleep_started_at": state.get("started_at"),
                "timed_sleep_interrupted": interrupted,
            },
        )
        return False

    def _observe_no_action(self, content: str) -> tuple[int, bool]:
        path = self.paths.runtime / "no-action.json"
        state = read_json(path, {})
        state = state if isinstance(state, dict) else {}
        fingerprint = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
        repeats = (
            int(state.get("repeats", 0) or 0) + 1
            if state.get("fingerprint") == fingerprint
            else 1
        )
        atomic_write_json(
            path,
            {
                "fingerprint": fingerprint,
                "repeats": repeats,
                "updated_at": utc_now(),
                "preview": content[:500],
            },
        )
        return repeats, repeats >= 3

    def run_turn(self, *, trigger: str = "manual", continuous: bool = False) -> str:
        """Work until sleep, no further action, error, or an operator stop.

        Only one-off calls use the configured round/time budget. Continuous
        execution keeps the same working context without synthetic turn wakes.
        """
        turn_id = sortable_id("turn_")
        self.records.emit("turn_started", turn_id=turn_id, trigger=trigger)
        self.records.life("turn_started", turn_id=turn_id, trigger=trigger)
        pulse_used = False
        first_wake_issued = False
        meta_memory_guidance_issued = False
        guided_meta_memory: str | None = None
        last_visible = ""
        started = time.monotonic()
        round_number = 0

        while not self._stop:
            if not continuous and (
                round_number >= self.config.max_life_loop_rounds
                or time.monotonic() - started > self.config.max_turn_seconds
            ):
                break
            round_number += 1
            self._refresh_key()
            if continuous:
                self._write_state("running", turn_id=turn_id, round=round_number)
            claimed = self._claim_notifications()
            inputs = self._render_notifications(claimed)
            recovery = self._pending_recovery()
            if recovery is not None:
                inputs.insert(0, self._render_notification(Notification(
                    id="engine_recovery",
                    created_at=str(recovery.get("last_failed_at") or utc_now()),
                    type="recovery", source="artificium_runtime", path=None,
                    summary="Retrying after earlier provider or runtime failures.",
                    metadata=recovery,
                )))
            inputs.extend(self._context_events())
            # Recheck the memory map when it changes, without needing a timed
            # turn boundary or repeating guidance for an unchanged map.
            meta_memory = self._read_full(self.paths.meta_memory) if continuous else None
            if not meta_memory_guidance_issued or (continuous and meta_memory != guided_meta_memory):
                guidance = self._meta_memory_guidance(turn_id=turn_id)
                if guidance:
                    inputs.append(guidance)
                    meta_memory_guidance_issued = True
                guided_meta_memory = meta_memory
            if self.initialization.pending() and not first_wake_issued:
                inputs.append(self.prompts.event("first_wake"))
                first_wake_issued = True
            if not inputs and not pulse_used and (trigger != "continuation" or round_number == 1):
                inputs.append(
                    self.prompts.event(
                        "external_event",
                        event_type="life_loop_continuation" if trigger == "continuation" else "life_loop_wake",
                        created_at=utc_now(),
                        source="artificium_runtime",
                        summary=(
                            "The previous generation finished without tool calls or an explicit sleep."
                            if trigger == "continuation"
                            else f"The life-loop was triggered by {trigger}."
                        ),
                        path_or_none="none",
                        metadata_or_none={"trigger": trigger},
                    )
                )
                pulse_used = True
            messages = self._request_messages(inputs, wake_reason=trigger)
            count = None
            prepared = None
            try:
                prepared = self.engine.prepare(messages)
                count = measure(self.engine, prepared, messages, self.estimator)
                mandatory_offload = self._mandatory_offload_required(
                    count.tokens, generation_room=available_output(self.config, count))
                if mandatory_offload:
                    inputs.append(self.prompts.event(
                        "mandatory_offload", threshold_percent=f"{self.config.offload_threshold_percent:g}",
                        input_tokens=count.tokens, token_count_source=count.source,
                        working_memory_tokens=self.config.working_memory_limit))
                    messages = self._request_messages(inputs, wake_reason=trigger)
                    prepared = self.engine.prepare(messages)
                    count = measure(self.engine, prepared, messages, self.estimator)
                context_data = {
                    "round": round_number, "estimated_tokens": count.tokens,
                    "token_count_source": count.source,
                    "context_window_tokens": self.config.context_window_tokens,
                    "working_memory_tokens": self.config.working_memory_limit,
                    "context_percent": round(count.tokens / self.config.context_window_tokens * 100, 3),
                    "wake_reason": trigger,
                }
                self.records.emit("context_usage_measured", turn_id=turn_id, **context_data)
                self.records.life("context_usage", turn_id=turn_id, **context_data)
                self.console.context(count.tokens, self.config.context_window_tokens, source=count.source)
                request_id, reply = self._complete(messages, wake_reason=trigger, pending_inputs=inputs,
                                                   prepared=prepared, count=count)
            except EngineError as exc:
                for item in claimed:
                    self.notifications.release(item)
                if not getattr(exc, "request_log_path", None):
                    failed_id = sortable_id("request_")
                    exc.request_log_path = str(self.paths.model_log / f"{failed_id}.json")
                    self.records.model_exchange(request_id=failed_id, messages=messages,
                                                request_parameters=prepared.safe_summary() if prepared else {}, error=str(exc))
                if self.config.auto_repair and repairable(exc, self.config, count):
                    if self._repair_request(exc, inputs, context_error=context_exhausted(exc, self.config, count)):
                        pulse_used = first_wake_issued = meta_memory_guidance_issued = False
                        continue
                raise
            except KeyboardInterrupt:
                for item in claimed:
                    self.notifications.release(item)
                raise
            except Exception:
                for item in claimed:
                    self.notifications.release(item)
                raise
            consumed_images = self.visual.consume_once(request_id=request_id)
            self.console.usage(reply.usage)
            if self._stop:
                for item in claimed:
                    self.notifications.release(item)
                self.records.emit(
                    "turn_interrupted",
                    turn_id=turn_id,
                    request_id=request_id,
                    reason="operator_stop_at_provider_boundary",
                )
                self.records.life(
                    "turn_interrupted",
                    turn_id=turn_id,
                    request_id=request_id,
                    reason="operator_stop_at_provider_boundary",
                )
                self.console.line(
                    "shutdown",
                    "provider request finished; no new model-requested actions were executed",
                )
                break
            self._append_runtime(inputs, origin="life_loop_input")
            self._repair_state_path.unlink(missing_ok=True)
            self._acknowledge_recovery(recovery)
            for item in claimed:
                self.notifications.commit(item)

            parsed = parse_life_loop_output(reply.content)
            tool_validations = [
                (
                    call_index,
                    intent,
                    self.tools.validate(intent, call_index=call_index)
                    or (self._offload_tool_error(intent, call_index) if mandatory_offload else None),
                )
                for call_index, intent in enumerate(parsed.tools, start=1)
            ]
            validation_errors = [
                error for _, _, error in tool_validations if error is not None
            ]
            stream_ids = sorted(
                {
                    str(intent.arguments.get("stream_id") or intent.arguments.get("session_id"))
                    for intent in parsed.tools
                    if intent.arguments.get("stream_id") or intent.arguments.get("session_id")
                }
            )
            self.working.append(
                {
                    "role": "assistant",
                    "content": render_normalized_life_loop_output(
                        parsed, include_tools=not validation_errors
                    ),
                    "_artificium": {
                        "kind": "life_loop_output",
                        "turn_id": turn_id,
                        "tool_names": [intent.name for intent in parsed.tools],
                        "attention_stream_ids": stream_ids,
                        "tool_request_valid": not validation_errors,
                    },
                },
                origin="model",
            )
            if consumed_images["consumed"]:
                self._append_runtime(
                    [
                        self.prompts.event(
                            "visual_context_released",
                            release_reason="one-shot images were consumed by a successful inference",
                            released_images=consumed_images["consumed"],
                            active_image_count=consumed_images["active_count"],
                            active_images_or_none=(
                                consumed_images["active_images"] or "none"
                            ),
                        )
                    ],
                    origin="visual_context_consumed",
                )
            if reply.provider_reasoning:
                self.records.life(
                    "provider_reasoning",
                    turn_id=turn_id,
                    round=round_number,
                    content=reply.provider_reasoning,
                )
            for thought in parsed.thoughts:
                self.records.life(
                    "thought", turn_id=turn_id, round=round_number, content=thought
                )
                self.console.thought(thought)
            if parsed.visible:
                last_visible = parsed.visible
                self.records.life(
                    "life_loop_output",
                    turn_id=turn_id,
                    round=round_number,
                    content=parsed.visible,
                )
                self.console.line("output", parsed.visible, detail=True)

            if validation_errors:
                (self.paths.runtime / "no-action.json").unlink(missing_ok=True)
                repair_cases: list[dict[str, Any]] = []
                for call_index, intent, error in tool_validations:
                    if error is not None:
                        repair_cases.append(error)
                        continue
                    repair_cases.append(
                        {
                            "call_index": call_index,
                            "tool": intent.name,
                            "error": (
                                "This call was valid but was transactionally withheld "
                                "because another call in the same response was invalid."
                            ),
                            "suggested_tool": None,
                            "accepted_arguments": self.tools.accepted_arguments(
                                intent.name
                            ),
                            "received_arguments": intent.arguments,
                            "canonical_example": {
                                "tool": intent.name,
                                **intent.arguments,
                            },
                        }
                    )
                self.records.emit(
                    "tool_request_rejected",
                    turn_id=turn_id,
                    round=round_number,
                    errors=validation_errors,
                    repair_cases=repair_cases,
                    raw_response_sha256=hashlib.sha256(
                        reply.content.encode("utf-8")
                    ).hexdigest(),
                )
                self.records.life(
                    "tool_request_rejected",
                    turn_id=turn_id,
                    round=round_number,
                    errors=validation_errors,
                    repair_cases=repair_cases,
                )
                self.console.result(
                    "tool_protocol",
                    {
                        "status": "repair_required",
                        "summary": "invalid tool request was not executed or retained verbatim",
                    },
                )
                self._append_runtime(
                    [
                        self.prompts.event(
                            "tool_call_repair",
                            repair_cases=json_dumps(
                                repair_cases,
                                pretty=True,
                            ),
                            model_log_path=str(
                                self.paths.model_log / f"{request_id}.json"
                            ),
                        )
                    ],
                    origin="tool_call_repair",
                )
                continue

            if parsed.tools:
                (self.paths.runtime / "no-action.json").unlink(missing_ok=True)
                for intent in parsed.tools:
                    execution = self.tools.execute(intent)
                    stream_id = str(
                        execution.result.get("session_id")
                        or execution.result.get("stream_id")
                        or intent.arguments.get("stream_id")
                        or ""
                    )
                    if intent.name in {"open_attention", "next_attention_chunk"} and stream_id:
                        self.working.prepare_attention_chunk(stream_id)
                    self._append_tool_result(execution)
                    if (
                        intent.name in {"checkpoint_attention", "complete_attention"}
                        and stream_id
                        and execution.result.get("status")
                        in {"checkpointed", "paused", "refine_ready", "completed"}
                    ):
                        compacted = self.working.compress_attention_context(
                            session_id=stream_id,
                            compression=str(
                                intent.arguments.get("compressed_carry")
                                or intent.arguments.get("result")
                                or ""
                            ),
                            decision=str(intent.arguments.get("decision") or "complete"),
                            checkpoint_result=execution.result,
                        )
                        self.console.result("attention_context", compacted)
                    self._append_runtime(
                        execution.notifications,
                        origin=f"tool_notifications:{execution.name}",
                    )
                    if (
                        execution.name in {"offload_working_memory", "compact_context"}
                        and execution.result.get("status") == "offloaded"
                    ) or (
                        execution.name == "revise_self"
                        and execution.result.get("status") == "revised"
                    ):
                        break
                    if self.tools.sleep_request:
                        break
                if self.tools.sleep_request:
                    self._schedule_sleep()
                    break
                continue

            if mandatory_offload:
                # A plain response cannot satisfy the requirement or enter normal
                # sleep/backoff. The operator can still stop between rounds.
                continue
            repeats, forced_backoff = self._observe_no_action(reply.content)
            pending = self.interactions.pending_events(limit=20)
            if self.initialization.pending() and round_number < 3:
                self._append_runtime(
                    [self.prompts.event("first_wake")], origin="initialization_reminder"
                )
                continue
            if pending and round_number < 3:
                self._append_runtime(
                    [
                        self.prompts.event(
                            "external_event",
                            event_type="unhandled_interaction_reminder",
                            created_at=utc_now(),
                            source="artificium_runtime",
                            summary="One or more interaction events remain unresolved.",
                            path_or_none=str(self.paths.interactions),
                            metadata_or_none=pending,
                        )
                    ],
                    origin="interaction_reminder",
                )
                continue
            if forced_backoff:
                seconds = self.config.engine_error_backoff_seconds
                atomic_write_json(
                    self.paths.sleep_state,
                    {
                        "active": True,
                        "mode": "timed",
                        "seconds": seconds,
                        "started_at": utc_now(),
                        "wake_at_epoch": time.time() + seconds,
                        "reason": "repeated_no_action_output",
                    },
                )
                self.records.emit(
                    "automatic_no_action_backoff", repeats=repeats, seconds=seconds
                )
                self.console.line(
                    "sleep", f"repeated no-action output; backing off for {seconds:g}s"
                )
            break
        self.records.emit("turn_completed", turn_id=turn_id, visible=last_visible)
        self.records.life("turn_completed", turn_id=turn_id)
        self._write_state("idle", last_turn_id=turn_id)
        return last_visible

    def _write_state(self, status: str, **data: Any) -> None:
        atomic_write_json(
            self.paths.runtime_state,
            {
                "status": status,
                "pid": os.getpid(),
                "updated_at": utc_now(),
                "provider": self.config.provider,
                "model": self.config.model,
                "prompt_pack": self.prompts.version,
                **data,
            },
        )

    def status(self) -> dict[str, Any]:
        overhead = self._prompt_overhead()
        visual_message = self.visual.request_message()
        visual_tokens = (
            self.estimator.messages([visual_message]) if visual_message is not None else 0
        )
        meta_metrics = self._meta_memory_metrics()
        attention = self.streams.list()
        for state in attention:
            state["stream_id"] = state.pop("id", None)
            profile = str(state.pop("profile", "auto"))
            state["granularity"] = {"broad": "coarse", "granular": "fine"}.get(
                profile, profile
            )
        lock_state = read_json(self.paths.process_lock, {})
        process_pid = (
            int(lock_state.get("pid", 0) or 0)
            if isinstance(lock_state, dict)
            else 0
        )
        process_alive = ProcessLock._alive(process_pid)
        return {
            "version": self.VERSION,
            "codename": self.config.codename,
            "provider": self.config.provider,
            "model": self.config.model,
            "root": str(self.paths.root),
            "mind": str(self.paths.mind),
            "process": {
                "alive": process_alive,
                "pid": process_pid or None,
                "started_at": (
                    lock_state.get("started_at")
                    if isinstance(lock_state, dict)
                    else None
                ),
                "lock_path": str(self.paths.process_lock),
            },
            "estimated_request_tokens": self.working.estimated_tokens(
                overhead + visual_tokens
            ),
            "working_context_tokens": self.working.estimated_tokens(),
            "pinned_prompt_tokens": overhead,
            "visual_context_tokens": visual_tokens,
            "active_images": self.visual.list(),
            "context_window_tokens": self.config.context_window_tokens,
            "working_memory_tokens": self.config.working_memory_limit,
            "auto_repair": self.config.auto_repair,
            "meta_memory_tokens": meta_metrics["tokens"],
            "meta_memory_words": meta_metrics["words"],
            "meta_memory_characters": meta_metrics["characters"],
            "meta_memory_guidance_tokens": self.config.meta_memory_guidance_tokens,
            "initialization": self.initialization.ensure(),
            "pending_notifications": len(list(self.paths.notifications_new.glob("*.json"))),
            "unhandled_interactions": self.interactions.pending_events(limit=100),
            "scheduled_tasks": self.scheduler.list(status="pending", limit=100)["tasks"],
            "attention_streams": attention,
            "feature_usage": self.records.feature_usage(),
            "sleep": read_json(self.paths.sleep_state, {}),
            "runtime": read_json(self.paths.runtime_state, {}),
        }

    def run_once(self, *, trigger: str = "manual") -> str:
        self.scheduler.fire_due()
        return self.run_turn(trigger=trigger)

    def _error_sleep(self, exc: Exception, repeats: int) -> None:
        if isinstance(exc, EngineError) and exc.requires_operator_action:
            self._request_blocked = True
            detail = {
                "error": str(exc),
                "hint": exc.hint,
                "http_status": exc.status,
                "error_kind": exc.kind,
                "recovery_command": "python3 artificium.py restart",
            }
            self._write_state("blocked", **detail)
            self.records.life("engine_blocked", **detail)
            self.records.emit("engine_blocked", **detail)
            self.console.line(
                "engine",
                "Model requests paused; history and incoming messages are retained. "
                "Correct the reported request or connection problem, then run: "
                "python3 artificium.py restart",
            )
            return
        seconds = min(self.config.engine_error_backoff_seconds, 900.0)
        for _ in range(max(0, repeats - 1)):
            seconds = min(seconds * 2, 900.0)
            if seconds >= 900.0:
                break
        previous = read_json(self.paths.sleep_state, {}).get("recovery") or {}
        failed_at = utc_now()
        atomic_write_json(
            self.paths.sleep_state,
            {
                "active": True,
                "mode": "timed",
                "seconds": seconds,
                "started_at": utc_now(),
                "wake_at_epoch": time.time() + seconds,
                "reason": "engine_error_backoff",
                "recovery": {
                    "reason": f"{type(exc).__name__}: {exc}",
                    "failed_attempts": int(previous.get("failed_attempts", 0)) + 1,
                    "first_failed_at": previous.get("first_failed_at") or failed_at,
                    "last_failed_at": failed_at,
                    "paths": [str(self.paths.lifetime_log), str(self.paths.model_log)],
                },
            },
        )
        self.console.line("backoff", f"retrying after {seconds:g}s")

    def run_forever(self, *, verbose: bool = False, quiet: bool = False) -> None:
        self.console.verbose = verbose
        self.console.quiet = quiet
        self._stop = False
        self._request_blocked = False

        interrupt_count = 0

        def stop(signum: int, *_: object) -> None:
            nonlocal interrupt_count
            if signum == signal.SIGINT:
                interrupt_count += 1
                if interrupt_count >= 2:
                    self.console.line(
                        "shutdown",
                        "second Ctrl-C received; forcing the foreground process to exit",
                    )
                    raise KeyboardInterrupt
                self.console.line(
                    "shutdown",
                    (
                        "Ctrl-C requested a graceful stop. An in-flight provider request "
                        "may finish first; press Ctrl-C again to force exit."
                    ),
                )
            else:
                self.console.line(
                    "shutdown",
                    "stop requested; finishing the current safe boundary",
                )
            self._stop = True

        prior_int = signal.signal(signal.SIGINT, stop)
        prior_term = signal.signal(signal.SIGTERM, stop)
        self.console.line(
            "artificium",
            f"Artificium-revolution {self.VERSION} living at {self.paths.mind} "
            f"({self.config.provider}/{self.config.model})",
        )
        self.console.line(
            "life-loop",
            (
                f"online; full trace: {self.paths.life_loop_log}; "
                "Ctrl-C requests a graceful foreground stop"
            ),
        )
        initial_pulse = True
        error_repeats = 0
        scheduler_stop = threading.Event()

        def scheduler_worker() -> None:
            while not scheduler_stop.is_set():
                try:
                    fired = self.scheduler.fire_due()
                    for item in fired:
                        task = item["task"]
                        self.console.line(
                            "scheduler", f"due: {task['name']} ({task['id']})"
                        )
                except Exception as exc:
                    self.records.emit(
                        "scheduler_poll_failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    self.console.error(f"scheduler: {type(exc).__name__}: {exc}")
                scheduler_stop.wait(self.config.poll_seconds)

        try:
            with ProcessLock(self.paths.process_lock):
                self._write_state("running")
                scheduler_thread = threading.Thread(
                    target=scheduler_worker,
                    name="artificium-scheduler",
                    daemon=True,
                )
                scheduler_thread.start()
                while not self._stop:
                    self._refresh_key()
                    self.interactions.reconcile()
                    if self._request_blocked:
                        time.sleep(self.config.poll_seconds)
                        continue
                    if self._sleep_active():
                        time.sleep(self.config.poll_seconds)
                        continue
                    event_ready = self.notifications.has_new()
                    retry_ready = self._pending_recovery() is not None
                    # An awake run only returns after a response with no tools.
                    # Continue immediately; sleep, stop and error guards still apply.
                    trigger = (
                        "startup"
                        if initial_pulse
                        else "engine_retry"
                        if retry_ready
                        else "notification"
                        if event_ready
                        else "continuation"
                    )
                    initial_pulse = False
                    try:
                        self.run_turn(trigger=trigger, continuous=True)
                        error_repeats = 0
                    except Exception as exc:
                        error_repeats += 1
                        self.records.emit(
                            "turn_failed",
                            error=f"{type(exc).__name__}: {exc}",
                            repeats=error_repeats,
                        )
                        if not isinstance(exc, EngineError):
                            self.console.error(f"{type(exc).__name__}: {exc}")
                        self._error_sleep(exc, error_repeats)
                self._write_state("stopped")
                self.console.line("shutdown", "life-loop stopped")
        finally:
            scheduler_stop.set()
            thread = locals().get("scheduler_thread")
            if isinstance(thread, threading.Thread):
                thread.join(timeout=max(1.0, self.config.poll_seconds * 2))
            if self._stop:
                self._write_state("stopped")
            signal.signal(signal.SIGINT, prior_int)
            signal.signal(signal.SIGTERM, prior_term)
