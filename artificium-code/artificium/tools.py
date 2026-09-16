from __future__ import annotations

import difflib
import inspect
import mimetypes
import subprocess
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
    sha256_file,
    sortable_id,
    utc_now,
)
from .initialization import Initialization
from .interactions import InteractionStore
from .life_loop import ToolIntent
from .memory import InfiniteAttention, LongTermMemory, WorkingMemory
from .prompts import PromptPack
from .records import Console, Records
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


class ToolRegistry:
    """Provider-neutral tools used only through Artificium's textual protocol."""

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
        self.records.emit(
            "tool_executed",
            tool_id=intent.id,
            name=intent.name,
            started_at=started_at,
            finished_at=finished_at,
            result=result,
        )
        self.records.life("tool_result", tool_id=intent.id, name=intent.name, result=result)
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
        return {
            "status": "output_saved",
            "summary": "tool output exceeded the working-context limit",
            "full_output_path": str(output_path),
            "output_characters": len(encoded),
            "output_truncated": True,
            "primary_target": self._primary_target(intent.arguments),
        }

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

    def list_directory(
        self,
        path: str = "mind",
        depth: int = 1,
        include_hidden: bool = False,
        max_entries: int = 500,
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(target)
        depth = max(0, min(int(depth), 10))
        maximum = max(1, min(int(max_entries), 5_000))
        entries: list[dict[str, Any]] = []
        root_depth = len(target.parts)
        for child in sorted(target.rglob("*")):
            relative_parts = child.parts[root_depth:]
            if len(relative_parts) > depth:
                continue
            if not include_hidden and any(part.startswith(".") for part in relative_parts):
                continue
            entries.append(
                {
                    "path": str(child),
                    "type": "directory" if child.is_dir() else "file",
                    "size_bytes": child.stat().st_size if child.is_file() else None,
                }
            )
            if len(entries) >= maximum:
                break
        return {
            "status": "ok",
            "summary": f"listed {len(entries)} entries",
            "path": str(target),
            "entries": entries,
            "truncated": len(entries) >= maximum,
        }

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        max_characters: int | None = None,
        start: int | None = None,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(target)
        size = target.stat().st_size
        explicit_bound = max_characters is not None or max_chars is not None or start is not None
        limit = min(
            int(max_characters or max_chars or self.config.max_direct_read_chars),
            self.config.max_direct_read_chars,
        )
        if not explicit_bound and int(start_line) <= 1 and size > limit:
            return {
                "status": "requires_attention",
                "summary": "file is too large for a safe direct read",
                "path": str(target),
                "size_bytes": size,
                "direct_read_limit_characters": limit,
            }
        if start is not None:
            offset = max(0, int(start))
            with target.open("rb") as handle:
                handle.seek(offset)
                raw = handle.read(limit)
                end = handle.tell()
            content = raw.decode("utf-8", errors="replace")
        else:
            with target.open("r", encoding="utf-8", errors="replace") as handle:
                for _ in range(max(0, int(start_line) - 1)):
                    if not handle.readline():
                        break
                offset = handle.tell()
                content = handle.read(limit)
                end = handle.tell()
        return {
            "status": "ok",
            "summary": "bounded file content read",
            "path": str(target),
            "start_byte": offset,
            "end_byte": end,
            "size_bytes": size,
            "complete": offset == 0 and end >= size,
            "truncated": end < size,
            "content": content,
        }

    def write_file(self, path: str, content: str, mode: str = "create") -> dict[str, Any]:
        target = self._resolve(path)
        if mode not in {"create", "overwrite", "append"}:
            raise ValueError("mode must be create, overwrite, or append")
        if mode == "create" and target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        final = content
        if mode == "append" and target.exists():
            final = target.read_text(encoding="utf-8", errors="replace") + content
        atomic_write_text(target, final)
        return {
            "status": "written",
            "summary": f"file {mode} complete",
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }

    def run_shell(
        self, command: str, cwd: str | None = None, timeout_seconds: float | None = None
    ) -> dict[str, Any]:
        if not command.strip():
            raise ValueError("command cannot be empty")
        timeout = min(
            float(timeout_seconds or self.config.shell_timeout_seconds),
            self.config.max_shell_timeout_seconds,
        )
        working = self._resolve(cwd) if cwd else self.paths.root
        completed = subprocess.run(
            command,
            cwd=working,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "status": "ok" if completed.returncode == 0 else "failed",
            "summary": f"shell exited {completed.returncode}",
            "command": command,
            "cwd": str(working),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def load_images(
        self,
        paths: list[str] | str,
        detail: str = "auto",
        retention: str = "once",
    ) -> dict[str, Any]:
        supplied = [paths] if isinstance(paths, str) else paths
        resolved = [str(self._resolve(path)) for path in supplied]
        return self.visual.load(resolved, detail=detail, retention=retention)

    def list_loaded_images(self) -> dict[str, Any]:
        images = self.visual.list()
        return {
            "status": "ok",
            "summary": f"{len(images)} image(s) are active in visual context",
            "active_count": len(images),
            "active_images": images,
        }

    def release_images(
        self,
        paths: list[str] | str | None = None,
        image_ids: list[str] | str | None = None,
        all_images: bool = False,
    ) -> dict[str, Any]:
        supplied_paths = [paths] if isinstance(paths, str) else (paths or [])
        supplied_ids = [image_ids] if isinstance(image_ids, str) else (image_ids or [])
        resolved = [str(self._resolve(path)) for path in supplied_paths]
        return self.visual.release(
            paths=resolved,
            image_ids=supplied_ids,
            all_images=all_images,
            reason="explicit_tool_release",
        )

    def load_attachment(
        self,
        path: str,
        detail: str = "auto",
        retention: str = "once",
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(target)
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if mime.startswith("image/"):
            result = self.visual.load(
                [str(target)], detail=detail, retention=retention
            )
            result["compatibility_operation"] = "load_attachment"
            return result
        if mime.startswith("text/") or target.suffix.lower() in {".json", ".md", ".txt", ".py"}:
            return self.read_file(str(target))
        return {
            "status": "conversion_required",
            "summary": "attachment is durable but not directly supported by this engine adapter",
            "path": str(target),
            "mime": mime,
            "size_bytes": target.stat().st_size,
        }

    def save_memory(
        self,
        path: str,
        content: str,
        retrieve_when: str,
        source_refs: list[str] | None = None,
        mode: str = "overwrite",
    ) -> dict[str, Any]:
        result = self.memory.save(
            path=path,
            content=content,
            retrieve_when=retrieve_when,
            source_refs=source_refs or [],
            mode=mode,
        )
        result["_notifications"] = [
            self.prompts.event(
                "memory_organization",
                action="saved or updated",
                memory_path=result["memory_path"],
                parent_index_path=result["suggested_parent_index"],
                meta_memory_path=result["meta_memory_path"],
                retrieve_when=retrieve_when,
            )
        ]
        return result

    def search_memory(self, query: str, limit: int = 20) -> dict[str, Any]:
        results = self.memory.search(query, limit=limit)
        return {
            "status": "ok",
            "summary": f"found {len(results)} matching memories",
            "query": query,
            "results": results,
        }

    def remove_memory(self, path: str) -> dict[str, Any]:
        result = self.memory.remove(path)
        result["_notifications"] = [
            self.prompts.event(
                "memory_organization",
                action="removed",
                memory_path=result["path"],
                parent_index_path=result["suggested_parent_index"],
                meta_memory_path=result["meta_memory_path"],
                retrieve_when="The removed memory must no longer be advertised.",
            )
        ]
        return result

    def _control(self) -> dict[str, Any]:
        value = read_json(self.control_path, {})
        return value if isinstance(value, dict) else {}

    def _save_control(self, value: dict[str, Any]) -> None:
        atomic_write_json(self.control_path, value)

    def offload_working_memory(
        self,
        path: str = "",
        checkpoint: str = "",
        retrieve_when: str = "",
        source_refs: list[str] | None = None,
        reflection_complete: bool = False,
        reason: str = "context_management",
    ) -> dict[str, Any]:
        state = self._control()
        if not reflection_complete:
            state["working_memory_offload_pending"] = {
                "proposed_path": path or None,
                "requested_at": utc_now(),
            }
            self._save_control(state)
            return {
                "status": "reflection_required",
                "summary": "working-memory offloading paused for conscious memory formation",
                "_notifications": [
                    self.prompts.event("working_memory_offload_reflection"),
                ],
            }
        if not state.get("working_memory_offload_pending"):
            return {
                "status": "reflection_required",
                "summary": "request offload_working_memory once before confirming reflection",
                "_notifications": [
                    self.prompts.event("working_memory_offload_reflection")
                ],
            }
        remembered = self.memory.save(
            path=path,
            content=checkpoint,
            retrieve_when=retrieve_when,
            source_refs=source_refs or [],
        )
        offloaded = self.working.offload(
            title=str(Path(path).with_suffix("")),
            compression=checkpoint,
            reason=reason,
            memory_path=Path(remembered["path"]),
            replacement_builder=lambda values: self.prompts.runtime(
                "restored_checkpoint", **values
            ),
        )
        active_images = self.visual.list()
        released_visual = (
            self.visual.release(
                all_images=True,
                reason="working_memory_offloading",
            )
            if active_images
            else {
                "released": [],
                "active_count": 0,
                "active_images": [],
            }
        )
        state.pop("working_memory_offload_pending", None)
        state.pop("mandatory_offload_pending", None)
        # Clean up the pre-Revolution-1.1 control key if an existing mind is
        # upgraded while an old reflection was pending.
        state.pop("compaction_pending", None)
        self._save_control(state)
        reduction = max(0, offloaded["before_tokens"] - offloaded["after_tokens"])
        percent = (
            reduction / offloaded["before_tokens"] * 100
            if offloaded["before_tokens"]
            else 0
        )
        offloaded.update(
            {
                "memory_path": remembered["memory_path"],
                "retrieve_when": remembered["retrieve_when"],
                "reduction_tokens": reduction,
                "reduction_percent": round(percent, 1),
                "released_images": released_visual["released"],
            }
        )
        offloaded["_notifications"] = [
            self.prompts.event(
                "working_memory_offloaded",
                before_tokens=offloaded["before_tokens"],
                after_tokens=offloaded["after_tokens"],
                reduction_tokens=reduction,
                reduction_percent=f"{percent:.1f}",
                checkpoint_path=remembered["memory_path"],
                source_archive_path=offloaded["archive"],
                meta_memory_entry=retrieve_when,
                pending_event_count=len(self.interactions.pending_events(limit=1_000)),
            ),
            self.prompts.event(
                "memory_organization",
                action="saved as a working-memory-offload checkpoint",
                memory_path=remembered["memory_path"],
                parent_index_path=remembered["suggested_parent_index"],
                meta_memory_path=remembered["meta_memory_path"],
                retrieve_when=retrieve_when,
            ),
        ]
        return offloaded

    def compact_context(self, **arguments: Any) -> dict[str, Any]:
        """Compatibility alias for pre-1.1 contexts and external tests."""

        return self.offload_working_memory(**arguments)

    def revise_self(
        self,
        content: str,
        reason: str,
        source_event_ids: list[str] | None = None,
        reflection_complete: bool = False,
    ) -> dict[str, Any]:
        state = self._control()
        source_ids = source_event_ids or []
        if not reflection_complete:
            state["self_revision_pending"] = {"reason": reason, "requested_at": utc_now()}
            self._save_control(state)
            return {
                "status": "reflection_required",
                "summary": "persistent Self revision paused for reflection",
                "_notifications": [
                    self.prompts.event(
                        "pre_self_revision",
                        reason=reason,
                        source_event_ids_or_none=source_ids or "none",
                        self_path=str(self.paths.self_file),
                    )
                ],
            }
        if not state.get("self_revision_pending"):
            return {
                "status": "reflection_required",
                "summary": "request revise_self once before confirming reflection",
            }
        if len(content.strip()) < 20:
            raise ValueError("Self replacement is too short to remain meaningful")
        previous = self.paths.self_file.read_text(encoding="utf-8", errors="replace")
        history = self.paths.self_history / f"self-before-{sortable_id()}.txt"
        atomic_write_text(history, previous)
        atomic_write_text(self.paths.self_file, content.rstrip() + "\n")
        state.pop("self_revision_pending", None)
        self._save_control(state)
        result = {
            "status": "revised",
            "summary": "pinned Self atomically revised",
            "path": str(self.paths.self_file),
            "previous_version_path": str(history),
            "before_characters": len(previous),
            "after_characters": len(content.rstrip()) + 1,
        }
        self.records.emit("self_revised", reason=reason, source_event_ids=source_ids, **result)
        result["_notifications"] = [
            self.prompts.event(
                "post_self_revision",
                previous_version_path=str(history),
                self_path=str(self.paths.self_file),
                reason=reason,
                source_event_ids_or_none=source_ids or "none",
                before_characters=len(previous),
                after_characters=len(content.rstrip()) + 1,
            )
        ]
        return result

    def list_interactions(
        self,
        status: str = "all",
        limit: int = 50,
        entity_id: str | None = None,
        include_events: bool = False,
    ) -> dict[str, Any]:
        items = self.interactions.list_interactions(entity_id)
        pending = self.interactions.pending_events(10_000)
        for item in items:
            item["pending_count"] = sum(
                1
                for event in pending
                if f"/{item['id']}/" in str(event.get("event_path"))
            )
            if include_events:
                item["events"] = self.interactions.events(str(item["id"]))
        if status == "pending":
            items = [item for item in items if item.get("pending_count")]
        elif status not in {"all", "open", "sleeping", "closed"}:
            raise ValueError("status must be pending, open, sleeping, closed, or all")
        elif status != "all":
            items = [item for item in items if item.get("status") == status]
        items = items[: max(1, min(int(limit), 1_000))]
        return {
            "status": "ok",
            "summary": f"found {len(items)} interactions",
            "interactions": items,
        }

    def read_interaction_event(self, event_id: str) -> dict[str, Any]:
        result = self.interactions.read_event(event_id)
        event = result.get("event")
        content = event.get("content") if isinstance(event, dict) else None
        if isinstance(content, str) and len(content) > self.config.max_direct_read_chars:
            bounded = dict(event)
            bounded["content"] = "[LARGE CONTENT OMITTED: inspect event_path with Infinite Attention]"
            result["event"] = bounded
            result["status"] = "requires_attention"
            result["content_characters"] = len(content)
        return result

    def set_interaction_event_status(
        self, event_id: str, status: str, reason: str | None = None
    ) -> dict[str, Any]:
        if status not in {"handled", "postponed", "ignored"}:
            raise ValueError("status must be handled, postponed, or ignored")
        receipt = self.interactions.mark_handled(event_id, decision=status)
        if reason:
            receipt["reason"] = reason
            atomic_write_json(self.paths.receipts / f"{event_id}.json", receipt)
        return {
            "status": status,
            "summary": f"interaction event marked {status}",
            "event_id": event_id,
            "receipt": receipt,
        }

    def send_interaction(
        self,
        interaction_id: str,
        content: str,
        in_reply_to: str | None = None,
        attachments: list[str] | None = None,
        recipient: str | None = None,
        sender: str | None = None,
    ) -> dict[str, Any]:
        event, path = self.interactions.add_event(
            interaction_id,
            sender=sender or self.config.instance_id,
            recipient=recipient,
            content=content,
            direction="outbound",
            attachments=[str(self._resolve(item)) for item in (attachments or [])],
            in_reply_to=in_reply_to,
        )
        return {
            "status": "sent",
            "summary": "outbound interaction event written",
            "path": str(path),
            "event": event,
            "_notifications": [
                self.prompts.event(
                    "memory_opportunity",
                    boundary_reason=f"an outbound event was sent in interaction {interaction_id}",
                )
            ],
        }

    def schedule_task(
        self,
        name: str,
        description: str,
        text: str,
        run_at: str,
        repeat_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self.scheduler.schedule(
            name=name,
            description=description,
            text=text,
            run_at=run_at,
            repeat_seconds=repeat_seconds,
        )

    def list_scheduled_tasks(
        self, status: str = "pending", limit: int = 100
    ) -> dict[str, Any]:
        return self.scheduler.list(status=status, limit=limit)

    def cancel_scheduled_task(self, task_id: str) -> dict[str, Any]:
        return self.scheduler.cancel(task_id)

    def _public_attention(self, result: dict[str, Any]) -> dict[str, Any]:
        internal = result.pop("session_id", None)
        if internal:
            result["stream_id"] = internal
        if "profile" in result:
            result["granularity"] = self._granularity(result.pop("profile"))
        return result

    @staticmethod
    def _granularity(profile: str | None) -> str:
        return {"broad": "coarse", "granular": "fine"}.get(
            str(profile or "auto"), str(profile or "auto")
        )

    def _attention_event(self, result: dict[str, Any], granularity: str | None = None) -> str:
        state = self.streams.state(str(result["stream_id"]))
        files = state.get("files") or []
        total_bytes = sum(int(item.get("snapshot_bytes", 0)) for item in files)
        chunk_bytes = max(1, int(state["chunk_tokens"] * self.config.chars_per_token))
        estimated_chunks = (
            max(1, (total_bytes + chunk_bytes - 1) // chunk_bytes)
            if total_bytes
            else "unknown"
        )
        return self.prompts.event(
            "attention_chunk",
            stream_id=result["stream_id"],
            objective=state["objective"],
            source_path=result.get("path") or state.get("source"),
            chunk_number=result.get("chunk_number", state.get("chunk_number")),
            chunk_count_or_unknown=estimated_chunks,
            source_range=(
                f"{result.get('byte_offset_start')}..{result.get('byte_offset_end')}"
                if result.get("byte_offset_start") is not None
                else "not applicable"
            ),
            granularity=granularity or self._granularity(state.get("profile")),
            source_exhausted=result.get("source_exhausted", state.get("source_exhausted")),
            pending_event_count=len(self.interactions.pending_events(1_000)),
        )

    def open_attention(
        self,
        source: str,
        objective: str,
        granularity: str = "auto",
        chunk_tokens: int | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        profile = {"auto": "broad", "coarse": "broad", "fine": "granular"}.get(
            granularity, granularity
        )
        result = self.streams.open(
            source=source,
            objective=objective,
            profile=profile,
            output_path=output_path,
            chunk_tokens=chunk_tokens,
        )
        internal_id = str(result["session_id"])
        state = self.streams.state(internal_id)
        total_bytes = sum(int(item.get("snapshot_bytes", 0)) for item in state.get("files", []))
        chunk_bytes = max(1, int(state["chunk_tokens"] * self.config.chars_per_token))
        estimated_chunks = (
            max(1, (total_bytes + chunk_bytes - 1) // chunk_bytes)
            if total_bytes
            else "unknown"
        )
        self._public_attention(result)
        result["_notifications"] = [
            self.prompts.event(
                "attention_opened",
                stream_id=result["stream_id"],
                source_path=state["source"],
                objective=state["objective"],
                context_window_tokens=self.config.context_window_tokens,
                usable_context_tokens=int(
                    self.config.context_window_tokens
                    * self.config.context_hard_fraction
                ),
                chunk_tokens=state["chunk_tokens"],
                carry_tokens=state["carry_tokens"],
                granularity=self._granularity(state.get("profile")),
                estimated_chunks_or_unknown=estimated_chunks,
            ),
            self._attention_event(result),
        ]
        return result

    def checkpoint_attention(
        self,
        stream_id: str,
        compressed_carry: str,
        decision: str = "continue",
        chunk_number: int | None = None,
        focus_ranges: list[dict[str, int]] | None = None,
        result: str = "",
    ) -> dict[str, Any]:
        if chunk_number is None:
            state = self.streams.state(stream_id)
            return {
                "status": "chunk_number_required",
                "summary": (
                    "checkpoint_attention requires the delivered chunk_number so a "
                    "delayed or repeated call cannot checkpoint the wrong chunk"
                ),
                "stream_id": stream_id,
                "current_chunk_number": state.get("chunk_number"),
                "awaiting_checkpoint": bool(state.get("awaiting_checkpoint")),
            }
        if decision == "refine":
            response = self.streams.checkpoint(
                session_id=stream_id,
                compression=compressed_carry,
                decision="pause",
                expected_chunk_number=chunk_number,
            )
            if response.get("status") in {
                "stale_checkpoint",
                "already_checkpointed",
                "already_completed",
            }:
                return self._public_attention(response)
            response.update(
                {
                    "status": "refine_ready",
                    "summary": "coarse carry saved; use refine_attention on a focus range",
                    "focus_ranges": focus_ranges or [],
                }
            )
            return self._public_attention(response)
        response = self.streams.checkpoint(
            session_id=stream_id,
            compression=compressed_carry,
            decision=decision,
            result=result,
            expected_chunk_number=chunk_number,
        )
        self._public_attention(response)
        if decision == "complete":
            response["_notifications"] = [self._attention_completed_event(stream_id, response)]
        return response

    def next_attention_chunk(self, stream_id: str) -> dict[str, Any]:
        result = self._public_attention(self.streams.next_chunk(stream_id))
        result["_notifications"] = [self._attention_event(result)]
        return result

    def refine_attention(
        self,
        stream_id: str,
        start: int,
        end: int,
        chunk_tokens: int | None = None,
        overlap_tokens: int = 0,
    ) -> dict[str, Any]:
        result = self.streams.refine(
            session_id=stream_id,
            start=start,
            end=end,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        self._public_attention(result)
        result["_notifications"] = [self._attention_event(result, "fine")]
        return result

    def _attention_completed_event(self, stream_id: str, response: dict[str, Any]) -> str:
        state = self.streams.state(stream_id)
        return self.prompts.event(
            "attention_completed",
            stream_id=stream_id,
            objective=state["objective"],
            source_path=state["source"],
            chunks_inspected=state.get("chunk_number", 0),
            source_exhausted=state.get("source_exhausted"),
            refined_ranges_or_none=state.get("refined_ranges") or "none",
            result_path=response.get("result_path") or state.get("result_path"),
            final_carry_path=str(self.paths.streams / stream_id / "carry.txt"),
            pending_event_count=len(self.interactions.pending_events(1_000)),
        )

    def complete_attention(
        self,
        stream_id: str,
        result: str,
        result_path: str | None = None,
        retrieve_when: str | None = None,
        source_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        state = self.streams.state(stream_id)
        if state.get("status") == "completed":
            response = {
                "status": "already_completed",
                "summary": "Infinite Attention stream was already completed",
                "stream_id": stream_id,
                "result_path": state.get("result_path"),
                "source_exhausted": bool(state.get("source_exhausted")),
            }
            response["_notifications"] = [
                self._attention_completed_event(stream_id, response)
            ]
            return response
        if state.get("awaiting_checkpoint"):
            response = self.streams.checkpoint(
                session_id=stream_id,
                compression=result,
                decision="complete",
                result=result,
            )
        else:
            response = self.streams.complete(stream_id, result)
        self._public_attention(response)
        if result_path:
            references = [
                stream_id,
                str(self.streams.state(stream_id).get("source")),
                *(source_refs or []),
            ]
            references = list(dict.fromkeys(item for item in references if item))
            remembered = self.memory.save(
                path=result_path,
                content=result,
                retrieve_when=retrieve_when
                or "Retrieve for the objective recorded by this Infinite Attention stream.",
                source_refs=references,
            )
            response["memory"] = remembered
        response["_notifications"] = [self._attention_completed_event(stream_id, response)]
        if result_path:
            response["_notifications"].append(
                self.prompts.event(
                    "memory_organization",
                    action="saved from an Infinite Attention result",
                    memory_path=remembered["memory_path"],
                    parent_index_path=remembered["suggested_parent_index"],
                    meta_memory_path=remembered["meta_memory_path"],
                    retrieve_when=remembered["retrieve_when"],
                )
            )
        return response

    def list_attention_streams(self, status: str | None = None) -> dict[str, Any]:
        streams = self.streams.list(status)
        for state in streams:
            state["stream_id"] = state.pop("id", None)
            state["granularity"] = self._granularity(state.pop("profile", None))
        return {
            "status": "ok",
            "summary": f"found {len(streams)} Infinite Attention streams",
            "streams": streams,
        }

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
