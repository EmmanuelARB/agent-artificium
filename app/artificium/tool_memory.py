from __future__ import annotations

from pathlib import Path
from typing import Any

from . import time_use
from .blockers import BlockerTracker
from .filesystem import atomic_write_text, sortable_id, utc_now


class MemoryToolsMixin:
    """Long-term memory, working-memory offloading, and Self revision tools.

    Mixed into :class:`ToolRegistry`; freely uses attributes/helpers defined
    there (``self.memory``, ``self.working``, ``self.visual``,
    ``self.interactions``, ``self.prompts``, ``self.paths``, ``self.records``,
    ``self._control``/``self._save_control``).
    """

    def save_memory(
        self,
        path: str,
        content: str,
        retrieve_when: str,
        source_refs: list[str] | None = None,
        mode: str = "overwrite",
        allow_shrink: bool = False,
    ) -> dict[str, Any]:
        result = self.memory.save(
            path=path,
            content=content,
            retrieve_when=retrieve_when,
            source_refs=source_refs or [],
            mode=mode,
            allow_shrink=allow_shrink,
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

    def offload_working_memory(
        self,
        path: str = "",
        checkpoint: str = "",
        retrieve_when: str = "",
        source_refs: list[str] | None = None,
        reflection_complete: bool = False,
        reason: str = "context_management",
        blocker: str = "",
        objective_progress: str = "",
    ) -> dict[str, Any]:
        state = self._control()
        BlockerTracker.validate(objective_progress)
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
                    *self._time_use_notice(),
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
        progress = BlockerTracker(self.paths, self.records).record(
            blocker, objective_progress
        )
        offloaded["blocker_status"] = progress
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
        if progress["stalled"]:
            offloaded["_notifications"].append(
                self.prompts.event(
                    "stalled_blocker",
                    blocker=progress["blocker"] or "(not named)",
                    same_blocker_checkpoints=progress["same_blocker_checkpoints"],
                    checkpoints_without_advance=progress["checkpoints_without_advance"],
                    since=progress["since"],
                    hours=progress["hours"],
                )
            )
        return offloaded

    def _time_use_notice(self) -> list[str]:
        """Measured time use since the last offload, for the reflection."""

        try:
            usage = time_use.summarize(self.paths.lifetime_log)
        except OSError:
            return []
        if not usage["since"]:
            return []
        lines = [
            f"- Since {usage['since']}: {usage['wall_hours']} h wall time; "
            f"{usage['model_hours']} h in {usage['model_requests']} model requests; "
            f"{usage['tool_hours']} h in {usage['tool_calls']} tool calls "
            f"({usage['failed_tool_calls']} failed); {usage['sleep_hours']} h asleep.",
        ]
        for item in usage["slowest_tool_calls"]:
            target = f": {item['target']}" if item["target"] else ""
            lines.append(f"- Slow call, {item['seconds']} s, {item['tool']}{target}")
        for item in usage["repeated_shell_commands"]:
            lines.append(
                f"- Run {item['runs']} times ({item['seconds']} s in total): {item['command']}"
            )
        return [self.prompts.event("time_use", time_use="\n".join(lines))]

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
