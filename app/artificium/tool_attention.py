from __future__ import annotations

from typing import Any


class AttentionToolsMixin:
    """Infinite Attention stream tools mixed into :class:`ToolRegistry`.

    Freely uses attributes/helpers defined there (``self.streams``,
    ``self.config``, ``self.prompts``, ``self.interactions``, ``self.paths``,
    ``self.memory``).
    """

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
