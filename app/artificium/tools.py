from __future__ import annotations

import difflib
import inspect
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .filesystem import (
    Paths,
    atomic_write_json,
    atomic_write_text,
    json_dumps,
    read_json,
    utc_now,
)
from .initialization import Initialization
from .interactions import InteractionStore
from .life_loop import ToolIntent
from .memory import InfiniteAttention, LongTermMemory, WorkingMemory
from .prompts import PromptPack
from .records import Console, Records
from .tool_attention import AttentionToolsMixin
from .tool_files import FileToolsMixin
from .tool_interactions import InteractionToolsMixin
from .tool_memory import MemoryToolsMixin
from .vision import VisualContext


@dataclass
class SleepRequest:
    mode: str
    seconds: float | None


@dataclass
class ToolExecution:
    name: str
    tool_id: str
    result: dict[str, Any]
    started_at: str
    finished_at: str
    notifications: list[str] = field(default_factory=list)


class ToolRegistry(
    FileToolsMixin,
    MemoryToolsMixin,
    InteractionToolsMixin,
    AttentionToolsMixin,
):
    """Provider-neutral tools used only through Artificium's textual protocol.

    The tool method implementations are split by domain into mixins
    (``tool_files``, ``tool_memory``, ``tool_interactions``,
    ``tool_attention``); this class keeps the registration table,
    validation, execution, and helpers shared across domains (path
    resolution, result bounding, control-state persistence). Sleep and
    first-wake initialization tools stay here since they are harness-level
    rather than domain tools.
    """

    # These examples are the deterministic repair source of truth. They are
    # intentionally complete, flat, and tool-specific so a malformed call never
    # receives a generic or accidentally unrelated example.
    CANONICAL_EXAMPLES: dict[str, dict[str, Any]] = {
        "list_directory": {"tool": "list_directory", "path": "mind", "depth": 2},
        "read_file": {"tool": "read_file", "path": "PATH", "max_characters": 20000},
        "write_file": {
            "tool": "write_file",
            "path": "PATH",
            "content": "TEXT",
            "mode": "create",
            "allow_shrink": False,
        },
        "run_shell": {"tool": "run_shell", "command": "COMMAND", "cwd": "PATH"},
        "load_images": {
            "tool": "load_images",
            "paths": ["IMAGE_PATH"],
            "detail": "auto",
            "retention": "once",
        },
        "list_loaded_images": {"tool": "list_loaded_images"},
        "release_images": {"tool": "release_images", "all_images": True},
        "load_attachment": {"tool": "load_attachment", "path": "ATTACHMENT_PATH"},
        "save_memory": {
            "tool": "save_memory",
            "path": "DESCRIPTIVE_PATH_BELOW_MEMORY",
            "content": "MEMORY_CONTENT",
            "retrieve_when": "WHEN_THIS_MEMORY_WILL_HELP",
            "mode": "overwrite",
            "allow_shrink": False,
        },
        "search_memory": {"tool": "search_memory", "query": "SEARCH_TERMS"},
        "remove_memory": {"tool": "remove_memory", "path": "PATH_BELOW_MEMORY"},
        "offload_working_memory": {
            "tool": "offload_working_memory",
            "reflection_complete": False,
        },
        "compact_context": {"tool": "compact_context"},
        "revise_self": {
            "tool": "revise_self",
            "content": "COMPLETE_NEW_SELF",
            "reason": "WHY_THIS_CHANGE_SHOULD_PERSIST",
            "reflection_complete": False,
        },
        "list_interactions": {"tool": "list_interactions", "status": "all"},
        "read_interaction_event": {
            "tool": "read_interaction_event",
            "event_id": "EVENT_ID",
        },
        "set_interaction_event_status": {
            "tool": "set_interaction_event_status",
            "event_id": "EVENT_ID",
            "status": "handled",
        },
        "send_interaction": {
            "tool": "send_interaction",
            "interaction_id": "INTERACTION_ID",
            "content": "MESSAGE",
        },
        "schedule_task": {
            "tool": "schedule_task",
            "name": "TASK_NAME",
            "description": "TASK_DESCRIPTION",
            "text": "SELF_CONTAINED_INSTRUCTION",
            "run_at": "2030-01-01T12:00:00Z",
        },
        "list_scheduled_tasks": {"tool": "list_scheduled_tasks", "status": "pending"},
        "cancel_scheduled_task": {
            "tool": "cancel_scheduled_task",
            "task_id": "TASK_ID",
        },
        "open_attention": {
            "tool": "open_attention",
            "source": "PATH",
            "objective": "PRECISE_OBJECTIVE",
            "granularity": "auto",
        },
        "checkpoint_attention": {
            "tool": "checkpoint_attention",
            "stream_id": "STREAM_ID",
            "chunk_number": 1,
            "compressed_carry": "OBJECTIVE_SPECIFIC_COMPRESSION",
            "decision": "continue",
        },
        "next_attention_chunk": {
            "tool": "next_attention_chunk",
            "stream_id": "STREAM_ID",
        },
        "refine_attention": {
            "tool": "refine_attention",
            "stream_id": "STREAM_ID",
            "start": 0,
            "end": 100000,
        },
        "complete_attention": {
            "tool": "complete_attention",
            "stream_id": "STREAM_ID",
            "result": "FINAL_RESULT",
        },
        "list_attention_streams": {"tool": "list_attention_streams", "status": None},
        "finish_initialization": {
            "tool": "finish_initialization",
            "summary": "WHAT_WAS_INSPECTED_AND_VERIFIED",
        },
        "sleep": {"tool": "sleep", "mode": "until_event", "reflection_complete": False},
    }

    LARGE_RESULT_TOOLS = {
        "open_attention",
        "next_attention_chunk",
        "refine_attention",
        "load_images",
        "load_attachment",
    }

    def __init__(
        self,
        *,
        paths: Paths,
        config: Config,
        prompts: PromptPack,
        records: Records,
        console: Console,
        interactions: InteractionStore,
        memory: LongTermMemory,
        working: WorkingMemory,
        streams: InfiniteAttention,
        visual: VisualContext,
        initialization: Initialization,
        scheduler: Any,
    ):
        self.paths = paths
        self.config = config
        self.prompts = prompts
        self.records = records
        self.console = console
        self.interactions = interactions
        self.memory = memory
        self.working = working
        self.streams = streams
        self.visual = visual
        self.initialization = initialization
        self.scheduler = scheduler
        self.sleep_request: SleepRequest | None = None
        self.control_path = self.paths.runtime / "life-loop-control.json"
        self._functions: dict[str, Callable[..., dict[str, Any]]] = {
            "list_directory": self.list_directory,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "run_shell": self.run_shell,
            "load_images": self.load_images,
            "list_loaded_images": self.list_loaded_images,
            "release_images": self.release_images,
            # Compatibility operation for existing contexts. New promptgramming uses
            # load_images for visual input and read_file/Infinite Attention for text.
            "load_attachment": self.load_attachment,
            "save_memory": self.save_memory,
            "search_memory": self.search_memory,
            "remove_memory": self.remove_memory,
            "offload_working_memory": self.offload_working_memory,
            # Backward-compatible textual alias. Promptgramming teaches only the
            # canonical working-memory-offloading vocabulary.
            "compact_context": self.compact_context,
            "revise_self": self.revise_self,
            "list_interactions": self.list_interactions,
            "read_interaction_event": self.read_interaction_event,
            "set_interaction_event_status": self.set_interaction_event_status,
            "send_interaction": self.send_interaction,
            "schedule_task": self.schedule_task,
            "list_scheduled_tasks": self.list_scheduled_tasks,
            "cancel_scheduled_task": self.cancel_scheduled_task,
            "open_attention": self.open_attention,
            "checkpoint_attention": self.checkpoint_attention,
            "next_attention_chunk": self.next_attention_chunk,
            "refine_attention": self.refine_attention,
            "complete_attention": self.complete_attention,
            "list_attention_streams": self.list_attention_streams,
            "finish_initialization": self.finish_initialization,
            "sleep": self.sleep,
        }

    def accepted_arguments(self, name: str) -> dict[str, list[str]]:
        function = self._functions.get(name)
        if function is None:
            return {"required": [], "optional": []}
        required: list[str] = []
        optional: list[str] = []
        for parameter in inspect.signature(function).parameters.values():
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            target = required if parameter.default is inspect.Parameter.empty else optional
            target.append(parameter.name)
        return {"required": required, "optional": optional}

    def canonical_example(self, name: str) -> dict[str, Any]:
        example = self.CANONICAL_EXAMPLES.get(name)
        return dict(example) if example else {"tool": name or "TOOL_NAME"}

    def validate(
        self, intent: ToolIntent, *, call_index: int | None = None
    ) -> dict[str, Any] | None:
        requested_name = intent.name or ""
        suggestion = ""
        if requested_name and requested_name not in self._functions:
            matches = difflib.get_close_matches(
                requested_name, self._functions.keys(), n=1, cutoff=0.55
            )
            suggestion = matches[0] if matches else ""
        example_name = requested_name if requested_name in self._functions else suggestion

        def details(error: str) -> dict[str, Any]:
            return {
                "call_index": call_index,
                "tool": requested_name or "unknown",
                "error": error,
                "suggested_tool": suggestion or None,
                "accepted_arguments": self.accepted_arguments(example_name),
                "received_arguments": intent.arguments,
                "canonical_example": self.canonical_example(example_name),
            }

        if intent.parse_error:
            return details(intent.parse_error)
        function = self._functions.get(intent.name)
        if function is None:
            message = f"Unknown tool `{intent.name}`. Use an exact catalog name."
            if suggestion:
                message += f" The closest available tool is `{suggestion}`."
            return details(message)
        try:
            inspect.signature(function).bind(**intent.arguments)
        except TypeError as exc:
            return details(f"Invalid arguments: {exc}")
        return None

    def execute(self, intent: ToolIntent) -> ToolExecution:
        started_at = utc_now()
        clock_start = time.monotonic()
        self.console.tool(intent.name or "invalid_tool", intent.arguments)
        self.records.life(
            "tool_call",
            tool_id=intent.id,
            name=intent.name,
            arguments=intent.arguments,
            raw_arguments=intent.raw_arguments,
        )
        if intent.parse_error:
            result: dict[str, Any] = {"status": "error", "summary": intent.parse_error}
        elif intent.name not in self._functions:
            result = {
                "status": "error",
                "summary": f"Unknown tool `{intent.name}`. Use an exact catalog name.",
            }
        else:
            try:
                result = self._functions[intent.name](**intent.arguments)
            except subprocess.TimeoutExpired as exc:
                result = {"status": "error", "summary": f"TimeoutExpired: {exc}",
                          "guidance": self.prompts.event("shell_timeout")}
                # run_shell decodes whatever it had already read from the
                # pipes before the timeout fired and stashes it on the
                # exception, so a delayed/killed command still leaves a
                # trace instead of going dark.
                partial_stdout = getattr(exc, "stdout", None)
                partial_stderr = getattr(exc, "stderr", None)
                if isinstance(partial_stdout, str) and partial_stdout:
                    result["stdout"] = partial_stdout
                if isinstance(partial_stderr, str) and partial_stderr:
                    result["stderr"] = partial_stderr
            except Exception as exc:
                result = {"status": "error", "summary": f"{type(exc).__name__}: {exc}"}
        if not isinstance(result, dict):
            result = {"status": "ok", "value": result}
        notifications = [str(item) for item in result.pop("_notifications", [])]
        if result.get("status") in {"error", "failed"}:
            notifications.append(
                self.prompts.event(
                    "tool_execution_error",
                    tool_name=intent.name or "unknown",
                    error_summary=result.get("summary") or result.get("status"),
                    accepted_arguments=self.accepted_arguments(intent.name),
                    received_arguments=intent.arguments,
                    canonical_example=json_dumps(
                        self.canonical_example(intent.name)
                    ),
                )
            )
        result = self._bound_result(intent, result)
        finished_at = utc_now()
        duration_seconds = round(time.monotonic() - clock_start, 3)
        self.records.emit(
            "tool_executed",
            tool_id=intent.id,
            name=intent.name,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            result=result,
        )
        self.records.life(
            "tool_result",
            tool_id=intent.id,
            name=intent.name,
            duration_seconds=duration_seconds,
            result=result,
        )
        self.console.result(intent.name or "invalid_tool", result)
        return ToolExecution(
            intent.name, intent.id, result, started_at, finished_at, notifications
        )

    def _bound_result(self, intent: ToolIntent, result: dict[str, Any]) -> dict[str, Any]:
        encoded = json_dumps(result, pretty=True)
        if intent.name in self.LARGE_RESULT_TOOLS or len(encoded) <= self.config.max_tool_output_chars:
            return result
        output_path = self.paths.outputs / f"{intent.id}--{intent.name or 'tool'}.json"
        atomic_write_text(output_path, encoded + "\n")
        bounded: dict[str, Any] = {
            "status": "output_saved",
            "summary": "tool output exceeded the working-context limit",
            "full_output_path": str(output_path),
            "output_characters": len(encoded),
            "output_truncated": True,
            "primary_target": self._primary_target(intent.arguments),
        }
        if "returncode" in result:
            bounded["returncode"] = result["returncode"]
        # Budget inline previews so the agent is not left blind even though
        # the full output only lives on disk now. Roughly 40% of each
        # stream's share goes to a head preview and 40% to a tail preview,
        # leaving headroom below max_tool_output_chars for JSON escaping and
        # the keys above. When more than one stream needs a preview (stdout
        # and stderr both present), the available room is split between them
        # so the two previews together still fit the budget.
        overhead = len(json_dumps(bounded, pretty=True))
        available = max(0, self.config.max_tool_output_chars - overhead)
        stdout = result.get("stdout")
        stderr = result.get("stderr")
        streams: list[tuple[str, str]] = []
        if isinstance(stdout, str) and stdout:
            streams.append(("stdout", stdout))
        if isinstance(stderr, str) and stderr:
            streams.append(("stderr", stderr))
        if streams:
            per_stream = available // len(streams)
            head_budget = int(per_stream * 0.4)
            tail_budget = int(per_stream * 0.4)
            for key, text in streams:
                head, tail = self._preview(text, head_budget, tail_budget)
                bounded[f"{key}_head"] = head
                bounded[f"{key}_tail"] = tail
        else:
            head_budget = int(available * 0.4)
            tail_budget = int(available * 0.4)
            head, tail = self._preview(encoded, head_budget, tail_budget)
            bounded["output_head"] = head
            bounded["output_tail"] = tail
        return bounded

    @staticmethod
    def _preview(text: str, head_budget: int, tail_budget: int) -> tuple[str, str]:
        """Split ``text`` into a non-overlapping head/tail preview pair."""

        head_budget = max(0, head_budget)
        tail_budget = max(0, tail_budget)
        if len(text) <= head_budget:
            return text, ""
        head = text[:head_budget]
        remainder = text[head_budget:]
        if len(remainder) <= tail_budget:
            return head, remainder
        return head, (text[-tail_budget:] if tail_budget else "")

    @staticmethod
    def _primary_target(arguments: dict[str, Any]) -> str | None:
        for name in ("path", "source", "interaction_id", "event_id", "stream_id", "command"):
            value = arguments.get(name)
            if value not in (None, ""):
                return str(value)
        return None

    def _resolve(self, supplied: str) -> Path:
        path = Path(supplied).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.paths.root / path).resolve()

    def _resolve_write(self, supplied: str) -> Path:
        """Resolve a path to write, keeping the application out of reach.

        The application is the one thing this agent cannot repair from inside
        itself, so changing it is routed through the workspace overlay, where a
        mistake is reversible and reviewable.  This is a guarantee about the
        file tools only; the shell remains what the operating system allows.
        """

        target = self._resolve(supplied)
        app = self.paths.app.resolve()
        if target == app or app in target.parents:
            relative = self.paths.prompts.resolve()
            hint = ("overrides/prompts/<relative path>"
                    if target.is_relative_to(relative)
                    else "overrides/code/<module>.py")
            raise PermissionError(
                f"{target} belongs to the application, which is read-only. "
                f"Copy it to {hint} in the workspace and edit the copy; the "
                "overlay replaces the shipped file at the next restart."
            )
        # Beside app/, only the workspace and the operator's settings change.
        project = self.paths.install.resolve()
        workspace = self.paths.root.resolve()
        mutable = {self.paths.config.resolve(), self.paths.secrets.resolve()}
        if ((target == project or project in target.parents)
                and not (target == workspace or workspace in target.parents)
                and target not in mutable):
            raise PermissionError(
                f"{target} belongs to the Artificium project, which is read-only "
                "outside workspace/. Write inside the workspace instead."
            )
        return target

    def control(self) -> dict[str, Any]:
        value = read_json(self.control_path, {})
        return value if isinstance(value, dict) else {}

    def save_control(self, value: dict[str, Any]) -> None:
        atomic_write_json(self.control_path, value)

    # Pre-existing private spellings. The runtime (and a workspace overlay
    # copy of this file written against the current API) calls
    # ``self.tools._control()``/``self.tools._save_control(...)`` directly,
    # so these aliases must keep working alongside the public names.
    _control = control
    _save_control = save_control

    def finish_initialization(self, summary: str) -> dict[str, Any]:
        state = self.initialization.finish(summary)
        return {
            "status": "completed",
            "summary": "first-wake orientation completed",
            "initialization": state,
        }

    def sleep(
        self,
        mode: str = "until_event",
        seconds: float | None = None,
        reflection_complete: bool = False,
    ) -> dict[str, Any]:
        aliases = {"until_notification": "until_event", "for": "timed"}
        mode = aliases.get(mode, mode)
        if mode not in {"until_event", "timed"}:
            raise ValueError("sleep mode must be until_event or timed")
        if mode == "timed":
            if seconds is None:
                raise ValueError("seconds is required for timed sleep")
            minimum = self.config.poll_seconds
            if float(seconds) < minimum:
                raise ValueError(f"sleep cannot be shorter than the configured interval ({minimum}s)")
        state = self._control()
        if not reflection_complete:
            state["sleep_reflection_pending"] = {
                "mode": mode,
                "seconds": seconds,
                "requested_at": utc_now(),
            }
            self._save_control(state)
            return {
                "status": "reflection_required",
                "summary": "sleep paused for one pre-sleep reflection",
                "_notifications": [self.prompts.event("pre_sleep")],
            }
        if not state.get("sleep_reflection_pending"):
            return {
                "status": "reflection_required",
                "summary": "request sleep once before confirming reflection",
                "_notifications": [self.prompts.event("pre_sleep")],
            }
        state.pop("sleep_reflection_pending", None)
        self._save_control(state)
        self.sleep_request = SleepRequest(mode, float(seconds) if seconds is not None else None)
        return {
            "status": "sleeping",
            "summary": f"sleep scheduled: {mode}",
            "mode": mode,
            "seconds": seconds,
        }
