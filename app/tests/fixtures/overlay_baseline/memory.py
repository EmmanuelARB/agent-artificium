from __future__ import annotations

import mimetypes
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .filesystem import (
    Paths,
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    descriptive_slug,
    json_dumps,
    read_json,
    read_jsonl,
    resolve_inside,
    safe_identifier,
    sortable_id,
    utc_now,
)
from .records import Records


class TokenEstimator:
    def __init__(self, chars_per_token: float = 4.0, image_tokens: int = 1_500):
        self.chars_per_token = chars_per_token
        self.image_tokens = image_tokens

    def text(self, content: str) -> int:
        return max(1, int(len(content) / self.chars_per_token)) if content else 0

    def value(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            total = 0
            for item in value:
                if isinstance(item, dict) and item.get("type") in {
                    "artificium_image",
                    "image_url",
                    "image",
                }:
                    total += self.image_tokens
                else:
                    total += self.value(item)
            return total
        if isinstance(value, dict):
            return sum(self.text(str(key)) + self.value(item) for key, item in value.items())
        return self.text(str(value))

    def messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(4 + self.value(message) for message in messages)


class WorkingMemory:
    def __init__(
        self, paths: Paths, config: Config, estimator: TokenEstimator, records: Records
    ):
        self.paths = paths
        self.config = config
        self.estimator = estimator
        self.records = records

    def load(self) -> list[dict[str, Any]]:
        return read_jsonl(self.paths.working_context)

    def append(self, message: dict[str, Any], *, origin: str) -> None:
        append_jsonl(self.paths.working_context, message)
        self.records.emit("working_context_appended", origin=origin, message=message)

    def estimated_tokens(self, extra_tokens: int = 0) -> int:
        return self.estimator.messages(self.load()) + max(0, int(extra_tokens))

    def pressure_notice(self, extra_tokens: int = 0) -> dict[str, Any] | None:
        tokens = self.estimated_tokens(extra_tokens)
        state = read_json(self.paths.pressure_state, {})
        if not isinstance(state, dict):
            state = {}
        step = self.config.context_reminder_tokens
        bucket = tokens // step
        prior_tokens = int(state.get("tokens", 0) or 0)
        prior_bucket = int(state.get("bucket", 0) or 0)
        if tokens < prior_tokens:
            prior_bucket = bucket
        notice = None
        if bucket > prior_bucket and bucket > 0:
            notice = {
                "estimated_tokens": tokens,
                "context_window_tokens": self.config.context_window_tokens,
                "working_memory_tokens": self.config.working_memory_limit,
                "fraction": tokens / self.config.working_memory_limit,
                "milestone": bucket * step,
            }
            prior_bucket = bucket
        atomic_write_json(
            self.paths.pressure_state,
            {"updated_at": utc_now(), "tokens": tokens, "bucket": prior_bucket},
        )
        return notice

    def offload(
        self,
        *,
        title: str,
        compression: str,
        reason: str,
        memory_path: Path,
        replacement_content: str | None = None,
        replacement_builder: Callable[[dict[str, Any]], str] | None = None,
    ) -> dict[str, Any]:
        slug = descriptive_slug(title, label="working-memory offload title")
        if len(compression.strip()) < 40:
            raise ValueError("compression is too short to be a self-sufficient continuation")
        prior = self.load()
        before_tokens = self.estimator.messages(prior)
        before_characters = len("".join(json_dumps(message) for message in prior))
        before_words = sum(
            len(re.findall(r"\S+", json_dumps(message))) for message in prior
        )
        identifier = sortable_id()
        archive = self.paths.context_archive / f"{slug}--{identifier}.jsonl"
        atomic_write_text(
            archive, "".join(json_dumps(message) + "\n" for message in prior)
        )
        replacement_values: dict[str, Any] = {
            "checkpoint_path": str(memory_path),
            "created_at": utc_now(),
            "before_tokens": before_tokens,
            "after_tokens": 0,
            "source_archive_path": str(archive),
            "checkpoint_content": compression.strip(),
        }
        if replacement_builder is not None:
            replacement_content = replacement_builder(replacement_values)
            provisional = {
                "role": self.config.runtime_message_role,
                "content": replacement_content,
            }
            replacement_values["after_tokens"] = self.estimator.messages([provisional])
            replacement_content = replacement_builder(replacement_values)
        replacement = {
            "role": self.config.runtime_message_role,
            "content": replacement_content
            or (
                "[RESTORED MEMORY — WORKING-MEMORY-OFFLOAD CHECKPOINT]\n"
                f"Memory path: {memory_path}\n\n{compression.strip()}\n"
                "[END RESTORED MEMORY]"
            ),
            "_artificium": {
                "kind": "context_checkpoint",
                "path": str(memory_path),
            },
        }
        atomic_write_text(self.paths.working_context, json_dumps(replacement) + "\n")
        after_tokens = self.estimator.messages([replacement])
        replacement_text = json_dumps(replacement)
        after_characters = len(replacement_text)
        after_words = len(re.findall(r"\S+", replacement_text))
        atomic_write_json(
            self.paths.pressure_state,
            {"updated_at": utc_now(), "tokens": after_tokens, "bucket": 0},
        )
        result = {
            "status": "offloaded",
            "summary": "working memory compressed, archived, and replaced by its checkpoint",
            "path": str(memory_path),
            "archive": str(archive),
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "before_words": before_words,
            "after_words": after_words,
            "before_characters": before_characters,
            "after_characters": after_characters,
            "prior_messages": len(prior),
        }
        self.records.emit("working_memory_offloaded", **result)
        self.records.life("working_memory_offloaded", **result)
        return result

    def compress_attention_context(
        self,
        *,
        session_id: str,
        compression: str,
        decision: str,
        checkpoint_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Replace consumed attention material with its latest semantic carry.

        Context-stream chunks intentionally bypass ordinary tool-output limits. Once the
        model checkpoints a chunk, keeping that raw observation (or every older carry)
        would make sequential attention grow without bound. Only stream-specific records
        are removed; unrelated notifications and work remain intact.
        """

        prior = self.load()
        before_tokens = self.estimator.messages(prior)
        kept: list[dict[str, Any]] = []
        removed = 0
        stream_tools = {
            "open_attention",
            "next_attention_chunk",
            "checkpoint_attention",
            "refine_attention",
            "complete_attention",
        }
        for message in prior:
            metadata = message.get("_artificium")
            metadata = metadata if isinstance(metadata, dict) else {}
            kind = metadata.get("kind")
            same_session = metadata.get("stream_id") == session_id
            session_ids = metadata.get("attention_stream_ids")
            session_ids = session_ids if isinstance(session_ids, list) else []
            remove = False
            if same_session and kind in {
                "attention_checkpoint",
                "tool_result",
            }:
                remove = True
            if not remove and kind == "life_loop_output" and session_id in session_ids:
                tool_names = set(metadata.get("tool_names") or [])
                # Preserve a multi-purpose output; its non-stream action or deliberation
                # may matter. Single-purpose stream mechanics are replaced by the carry.
                remove = bool(tool_names) and tool_names.issubset(stream_tools)
            if remove:
                removed += 1
            else:
                kept.append(message)

        state = checkpoint_result.get("status") or decision
        marker = {
            "role": self.config.runtime_message_role,
            "content": (
                "<INFINITE-ATTENTION-CHECKPOINT>\n"
                f"Session: {session_id}\n"
                f"Decision/status: {decision}/{state}\n"
                f"Source exhausted: {checkpoint_result.get('source_exhausted')}\n"
                f"Result path: {checkpoint_result.get('result_path') or '(not final)'}\n\n"
                "Latest objective-specific carry:\n"
                f"{compression.strip()}\n"
                "</INFINITE-ATTENTION-CHECKPOINT>"
            ),
            "_artificium": {
                "kind": "attention_checkpoint",
                "stream_id": session_id,
                "decision": decision,
            },
        }
        kept.append(marker)
        atomic_write_text(
            self.paths.working_context,
            "".join(json_dumps(message) + "\n" for message in kept),
        )
        after_tokens = self.estimator.messages(kept)
        result = {
            "status": "compressed",
            "summary": "consumed stream chunk replaced by its latest compressed carry",
            "stream_id": session_id,
            "removed_messages": removed,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
        }
        self.records.emit("attention_context_compressed", **result)
        self.records.life("attention_context_compressed", **result)
        return result

    def prepare_attention_chunk(self, session_id: str) -> dict[str, Any]:
        """Remove the prior carry/raw delivery before one replacement chunk arrives.

        The next chunk observation already embeds the latest durable carry. Removing its
        prior live marker avoids paying for that compression twice. This also makes an
        idempotent re-delivery replace, rather than duplicate, a large raw observation.
        """

        prior = self.load()
        kept: list[dict[str, Any]] = []
        removed = 0
        for message in prior:
            metadata = message.get("_artificium")
            metadata = metadata if isinstance(metadata, dict) else {}
            same_session = metadata.get("stream_id") == session_id
            is_replaceable = metadata.get("kind") == "attention_checkpoint" or (
                metadata.get("kind") == "tool_result"
                and metadata.get("tool")
                in {"open_attention", "next_attention_chunk"}
            )
            if same_session and is_replaceable:
                removed += 1
            else:
                kept.append(message)
        if removed:
            atomic_write_text(
                self.paths.working_context,
                "".join(json_dumps(message) + "\n" for message in kept),
            )
            self.records.emit(
                "attention_chunk_context_prepared",
                stream_id=session_id,
                removed_messages=removed,
            )
        return {
            "status": "prepared",
            "stream_id": session_id,
            "removed_messages": removed,
        }


class LongTermMemory:
    def __init__(self, paths: Paths, records: Records):
        self.paths = paths
        self.records = records

    def ensure_meta_memory(self) -> None:
        if self.paths.meta_memory.exists():
            return
        atomic_write_text(
            self.paths.meta_memory,
            "# Meta-memory\n\n"
            "Always-present knowledge and a compact, agent-authored map of durable "
            "memory. Keep what must remain in context and enough navigation to find "
            "everything else.\n",
        )

    def _resolve_memory_path(self, supplied: str) -> Path:
        clean = supplied.strip().replace("\\", "/")
        if clean.startswith("mind/memory/"):
            clean = clean[len("mind/memory/") :]
        elif clean.startswith("memory/"):
            clean = clean[len("memory/") :]
        elif clean.startswith("mind/"):
            raise ValueError(
                "memory paths are relative to mind/memory; do not prefix them with mind/"
            )
        if not clean or clean.endswith("/"):
            raise ValueError("memory path must name a file")
        if not Path(clean).suffix:
            clean += ".txt"
        if Path(clean).name == "index.txt":
            raise ValueError(
                "index.txt is an agent-authored navigation file; use write_file for "
                "indexes and save_memory for semantic memory"
            )
        if len(Path(clean).as_posix()) < 10:
            raise ValueError("memory path must be descriptive, not a generic short name")
        return resolve_inside(self.paths.memory, clean)

    def save(
        self,
        *,
        path: str,
        content: str,
        retrieve_when: str,
        source_refs: Sequence[str] = (),
        mode: str = "overwrite",
    ) -> dict[str, Any]:
        target = self._resolve_memory_path(path)
        if mode not in {"overwrite", "append"}:
            raise ValueError("mode must be overwrite or append")
        if not content.strip():
            raise ValueError("memory content cannot be empty")
        if len(retrieve_when.strip()) < 12:
            raise ValueError("retrieve_when must explain when this memory is useful")
        rendered = content.rstrip()
        rendered += "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append" and target.exists():
            existing = target.read_text(encoding="utf-8", errors="replace").rstrip()
            rendered = existing + "\n\n" + rendered
        atomic_write_text(target, rendered)
        relative = str(target.relative_to(self.paths.mind))
        result = {
            "status": "remembered",
            "summary": "durable memory written; organization remains agent-authored",
            "path": str(target),
            "memory_path": relative,
            "suggested_parent_index": str(target.parent / "index.txt"),
            "meta_memory_path": str(self.paths.meta_memory),
            "retrieve_when": retrieve_when.strip(),
            "source_refs": list(source_refs),
        }
        self.records.emit("long_term_memory_saved", **result)
        self.records.life("memory_saved", **result)
        return result

    def remove(self, path: str) -> dict[str, Any]:
        target = self._resolve_memory_path(path)
        if not target.is_file():
            raise FileNotFoundError(target)
        target.unlink()
        self.records.emit("long_term_memory_removed", path=str(target))
        return {
            "status": "removed",
            "summary": "memory removed; related indexes remain agent-authored",
            "path": str(target),
            "suggested_parent_index": str(target.parent / "index.txt"),
            "meta_memory_path": str(self.paths.meta_memory),
        }

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        words = [item.lower() for item in re.findall(r"[\w-]+", query) if len(item) > 1]
        results: list[dict[str, Any]] = []
        for path in sorted(self.paths.memory.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            haystack = (str(path.relative_to(self.paths.memory)) + "\n" + text).lower()
            score = sum(haystack.count(word) for word in words)
            if score:
                first = min((haystack.find(word) for word in words if word in haystack), default=0)
                start = max(0, first - 200)
                results.append(
                    {
                        "path": str(path),
                        "score": score,
                        "preview": text[start : start + 800],
                    }
                )
        return sorted(results, key=lambda item: (-int(item["score"]), item["path"]))[:limit]


class InfiniteAttention:
    """Durable chunk -> objective compression -> next-chunk Infinite Attention."""

    def __init__(
        self,
        paths: Paths,
        config: Config,
        estimator: TokenEstimator,
        records: Records,
    ):
        self.paths = paths
        self.config = config
        self.estimator = estimator
        self.records = records

    def _directory(self, session_id: str) -> Path:
        return self.paths.streams / safe_identifier(session_id, label="stream id")

    def _state_path(self, session_id: str) -> Path:
        return self._directory(session_id) / "state.json"

    def _carry_path(self, session_id: str) -> Path:
        return self._directory(session_id) / "carry.txt"

    def _chunk_path(self, session_id: str) -> Path:
        return self._directory(session_id) / "current_chunk.json"

    def _resolve_source(self, supplied: str) -> Path:
        candidate = Path(supplied).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        root_candidate = (self.paths.root / candidate).resolve()
        if root_candidate.exists():
            return root_candidate
        return (self.paths.mind / candidate).resolve()

    @staticmethod
    def _binary(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                return b"\x00" in handle.read(4096)
        except OSError:
            return True

    def _files(self, source: Path) -> Iterator[Path]:
        if source.is_file():
            yield source
            return
        if not source.is_dir():
            raise FileNotFoundError(source)
        for path in sorted(source.rglob("*")):
            if path.is_file() and not path.is_symlink():
                yield path

    def open(
        self,
        *,
        source: str,
        objective: str,
        profile: str = "broad",
        output_path: str | None = None,
        chunk_tokens: int | None = None,
    ) -> dict[str, Any]:
        if len(objective.strip()) < 8:
            raise ValueError("stream objective must state what should be extracted or done")
        resolved = self._resolve_source(source)
        files = [
            {
                "path": str(path),
                "snapshot_bytes": path.stat().st_size,
                "binary": self._binary(path),
                "mime": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            }
            for path in self._files(resolved)
        ]
        session_id = sortable_id("stream_")
        directory = self._directory(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        aliases = {"auto": "broad", "coarse": "broad", "fine": "granular"}
        profile = aliases.get(profile, profile)
        configured_chunk = self.config.chunk_tokens(profile)
        if chunk_tokens is not None:
            configured_chunk = max(
                1_000,
                min(
                    int(chunk_tokens),
                    int(self.config.working_memory_limit * 0.50),
                ),
            )
        state = {
            "id": session_id,
            "status": "active",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "source": str(resolved),
            "objective": objective.strip(),
            "profile": profile,
            "output_path": output_path,
            "files": files,
            "file_index": 0,
            "byte_offset": 0,
            "chunk_number": 0,
            "chunk_tokens": configured_chunk,
            "carry_tokens": self.config.carry_tokens(profile),
            "context_window_tokens": self.config.context_window_tokens,
            "source_exhausted": not files,
            "awaiting_checkpoint": False,
        }
        atomic_write_json(self._state_path(session_id), state)
        atomic_write_text(self._carry_path(session_id), "")
        self.records.emit("attention_opened", state=state)
        return self.next_chunk(session_id)

    def state(self, session_id: str) -> dict[str, Any]:
        value = read_json(self._state_path(session_id))
        if not isinstance(value, dict):
            raise FileNotFoundError(f"Unknown Infinite Attention stream: {session_id}")
        return value

    def _save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self._state_path(str(state["id"])), state)

    @staticmethod
    def _decode(raw: bytes, *, final: bool) -> tuple[str, int]:
        if not raw:
            return "", 0
        try:
            return raw.decode("utf-8"), len(raw)
        except UnicodeDecodeError as exc:
            if not final and exc.reason == "unexpected end of data" and exc.start > 0:
                usable = raw[: exc.start]
                return usable.decode("utf-8", errors="replace"), len(usable)
            return raw.decode("utf-8", errors="replace"), len(raw)

    def next_chunk(self, session_id: str) -> dict[str, Any]:
        state = self.state(session_id)
        if state.get("status") not in {"active", "paused"}:
            raise ValueError(
                f"Infinite Attention stream is {state.get('status')}, not resumable"
            )
        if state.get("awaiting_checkpoint"):
            cached = read_json(self._chunk_path(session_id))
            if not isinstance(cached, dict):
                raise RuntimeError("stream awaits a checkpoint but its chunk is missing")
            return cached
        state["status"] = "active"
        files = state.get("files") if isinstance(state.get("files"), list) else []
        carry = (
            self._carry_path(session_id).read_text(encoding="utf-8", errors="replace")
            if self._carry_path(session_id).exists()
            else ""
        )
        max_bytes = max(1_000, int(state["chunk_tokens"] * self.config.chars_per_token))
        while int(state["file_index"]) < len(files):
            descriptor = files[int(state["file_index"])]
            source = Path(str(descriptor["path"]))
            offset = int(state.get("byte_offset", 0))
            snapshot_bytes = int(descriptor.get("snapshot_bytes", 0))
            if offset >= snapshot_bytes:
                state["file_index"] = int(state["file_index"]) + 1
                state["byte_offset"] = 0
                continue
            state["chunk_number"] = int(state.get("chunk_number", 0)) + 1
            number = int(state["chunk_number"])
            if descriptor.get("binary"):
                state["file_index"] = int(state["file_index"]) + 1
                state["byte_offset"] = 0
                state["source_exhausted"] = int(state["file_index"]) >= len(files)
                result = {
                    "session_id": session_id,
                    "objective": state["objective"],
                    "chunk_number": number,
                    "kind": "image"
                    if str(descriptor.get("mime", "")).startswith("image/")
                    else "binary",
                    "path": str(source),
                    "mime": descriptor.get("mime"),
                    "size_bytes": snapshot_bytes,
                    "carried_memory": carry,
                    "carry_target_tokens": state["carry_tokens"],
                    "source_exhausted": state["source_exhausted"],
                }
            else:
                remaining = snapshot_bytes - offset
                amount = min(max_bytes, remaining)
                with source.open("rb") as handle:
                    handle.seek(offset)
                    raw = handle.read(amount)
                text, consumed = self._decode(raw, final=amount >= remaining)
                if consumed <= 0:
                    raise RuntimeError(f"Could not advance through {source}")
                state["byte_offset"] = offset + consumed
                if int(state["byte_offset"]) >= snapshot_bytes:
                    state["file_index"] = int(state["file_index"]) + 1
                    state["byte_offset"] = 0
                state["source_exhausted"] = int(state["file_index"]) >= len(files)
                result = {
                    "session_id": session_id,
                    "objective": state["objective"],
                    "profile": state["profile"],
                    "chunk_number": number,
                    "kind": "text",
                    "path": str(source),
                    "byte_offset_start": offset,
                    "byte_offset_end": offset + consumed,
                    "estimated_chunk_tokens": self.estimator.text(text),
                    "configured_chunk_tokens": state["chunk_tokens"],
                    "carried_memory": carry,
                    "carry_target_tokens": state["carry_tokens"],
                    "source_exhausted": state["source_exhausted"],
                    "content": text,
                }
            state["awaiting_checkpoint"] = True
            atomic_write_json(self._chunk_path(session_id), result)
            self._save(state)
            self.records.emit(
                "attention_chunk",
                session_id=session_id,
                chunk_number=number,
                path=str(source),
                chunk_kind=result["kind"],
                source_exhausted=result["source_exhausted"],
                byte_offset_start=result.get("byte_offset_start"),
                byte_offset_end=result.get("byte_offset_end"),
            )
            return result
        state["source_exhausted"] = True
        state["awaiting_checkpoint"] = True
        result = {
            "session_id": session_id,
            "objective": state["objective"],
            "kind": "end",
            "source_exhausted": True,
            "carried_memory": carry,
        }
        atomic_write_json(self._chunk_path(session_id), result)
        self._save(state)
        return result

    def checkpoint(
        self,
        *,
        session_id: str,
        compression: str,
        decision: str,
        result: str = "",
        expected_chunk_number: int | None = None,
    ) -> dict[str, Any]:
        if decision not in {"continue", "pause", "complete"}:
            raise ValueError("decision must be continue, pause, or complete")
        if not compression.strip():
            raise ValueError("compressed carry cannot be empty")
        state = self.state(session_id)
        if state.get("status") == "completed":
            return {
                "status": "already_completed",
                "summary": "Infinite Attention stream was already completed",
                "session_id": session_id,
                "chunk_number": state.get("chunk_number"),
                "result_path": state.get("result_path"),
            }
        current_chunk = int(state.get("chunk_number", 0) or 0)
        if expected_chunk_number is not None and int(expected_chunk_number) != current_chunk:
            response = {
                "status": "stale_checkpoint",
                "summary": (
                    "checkpoint did not match the currently delivered chunk; "
                    "stream state was left unchanged"
                ),
                "session_id": session_id,
                "expected_chunk_number": int(expected_chunk_number),
                "current_chunk_number": current_chunk,
                "awaiting_checkpoint": bool(state.get("awaiting_checkpoint")),
                "next_action": (
                    "inspect and checkpoint the currently delivered chunk"
                    if state.get("awaiting_checkpoint")
                    else "request next_attention_chunk"
                ),
            }
            self.records.emit("attention_stale_checkpoint", **response)
            return response
        if not state.get("awaiting_checkpoint"):
            last = state.get("last_checkpoint")
            response = {
                "status": "already_checkpointed",
                "summary": (
                    "the latest delivered chunk already has a durable checkpoint; "
                    "no stream state was changed"
                ),
                "session_id": session_id,
                "chunk_number": current_chunk,
                "last_checkpoint": last,
                "next_action": "request next_attention_chunk",
            }
            self.records.emit("attention_checkpoint_replayed", **response)
            return response
        estimated = self.estimator.text(compression)
        atomic_write_text(self._carry_path(session_id), compression.rstrip() + "\n")
        append_jsonl(
            self._directory(session_id) / "checkpoints.jsonl",
            {
                "timestamp": utc_now(),
                "chunk_number": state.get("chunk_number"),
                "file_index": state.get("file_index"),
                "byte_offset": state.get("byte_offset"),
                "decision": decision,
                "estimated_carry_tokens": estimated,
                "compression": compression,
            },
        )
        state["awaiting_checkpoint"] = False
        state["estimated_carry_tokens"] = estimated
        state["last_checkpoint"] = {
            "chunk_number": current_chunk,
            "decision": decision,
            "timestamp": utc_now(),
            "estimated_carry_tokens": estimated,
        }
        warning = None
        if estimated > int(state["carry_tokens"]):
            warning = (
                f"Carry is ~{estimated} tokens, above the target of {state['carry_tokens']}; "
                "compress more aggressively before another large chunk."
            )
        if decision == "complete":
            state["status"] = "completed"
            state["completed_at"] = utc_now()
            final = result.strip() or compression.strip()
            destination = self._destination(state)
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(destination, final.rstrip() + "\n")
            state["result_path"] = str(destination)
            self._save(state)
            self._chunk_path(session_id).unlink(missing_ok=True)
            response = {
                "status": "completed",
                "summary": "Infinite Attention stream completed",
                "session_id": session_id,
                "result_path": str(destination),
                "stopped_early": not bool(state.get("source_exhausted")),
                "source_exhausted": bool(state.get("source_exhausted")),
                "warning": warning,
            }
            self.records.emit("attention_completed", **response)
            return response
        if decision == "pause":
            state["status"] = "paused"
            self._save(state)
            self._chunk_path(session_id).unlink(missing_ok=True)
            response = {
                "status": "paused",
                "summary": "Infinite Attention stream paused at a durable checkpoint",
                "session_id": session_id,
                "chunk_number": current_chunk,
                "warning": warning,
            }
            self.records.emit("attention_checkpointed", **response, decision=decision)
            return response
        state["status"] = "active"
        self._save(state)
        self._chunk_path(session_id).unlink(missing_ok=True)
        response = {
            "status": "checkpointed",
            "summary": "compression saved; stream is ready for another chunk",
            "session_id": session_id,
            "chunk_number": current_chunk,
            "yield_to_life_loop": True,
            "warning": warning,
        }
        self.records.emit("attention_checkpointed", **response, decision=decision)
        return response

    def _destination(self, state: dict[str, Any]) -> Path:
        supplied = state.get("output_path")
        if not supplied:
            return self._directory(str(state["id"])) / "result.txt"
        candidate = Path(str(supplied)).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (self.paths.root / candidate).resolve()

    def complete(self, session_id: str, result: str) -> dict[str, Any]:
        """Complete from a durable carry, including after fine-range refinement."""

        state = self.state(session_id)
        if state.get("status") == "completed":
            raise ValueError("Infinite Attention stream is already complete")
        if not result.strip():
            raise ValueError("final Infinite Attention result cannot be empty")
        atomic_write_text(self._carry_path(session_id), result.rstrip() + "\n")
        destination = self._destination(state)
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(destination, result.rstrip() + "\n")
        state.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "result_path": str(destination),
                "awaiting_checkpoint": False,
            }
        )
        self._save(state)
        self._chunk_path(session_id).unlink(missing_ok=True)
        response = {
            "status": "completed",
            "summary": "Infinite Attention stream completed from its durable carry",
            "session_id": session_id,
            "result_path": str(destination),
            "stopped_early": not bool(state.get("source_exhausted")),
            "source_exhausted": bool(state.get("source_exhausted")),
        }
        self.records.emit("attention_completed", **response)
        return response

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in sorted(self.paths.streams.glob("*/state.json")):
            state = read_json(path)
            if isinstance(state, dict) and (status is None or state.get("status") == status):
                result.append(state)
        return result

    def refine(
        self,
        *,
        session_id: str,
        start: int,
        end: int,
        chunk_tokens: int | None = None,
        overlap_tokens: int = 0,
    ) -> dict[str, Any]:
        """Reread a precise source range without disturbing the coarse cursor."""

        state = self.state(session_id)
        files = state.get("files") if isinstance(state.get("files"), list) else []
        if len(files) != 1 or files[0].get("binary"):
            raise ValueError("fine range refinement currently requires one text source file")
        source = Path(str(files[0]["path"]))
        size = int(files[0].get("snapshot_bytes", source.stat().st_size))
        start = max(0, int(start))
        end = min(size, int(end))
        if end <= start:
            raise ValueError("refinement end must be greater than start")
        tokens = chunk_tokens or self.config.chunk_tokens("granular")
        tokens = max(1_000, min(int(tokens), self.config.chunk_tokens("balanced")))
        max_bytes = max(1_000, int(tokens * self.config.chars_per_token))
        requested_end = end
        end = min(end, start + max_bytes)
        with source.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(end - start)
        text, consumed = self._decode(raw, final=end >= size)
        actual_end = start + consumed
        overlap_bytes = max(0, int(overlap_tokens * self.config.chars_per_token))
        next_start = None
        if actual_end < requested_end:
            next_start = max(start + 1, actual_end - overlap_bytes)
        ranges = state.get("refined_ranges")
        ranges = ranges if isinstance(ranges, list) else []
        ranges.append({"start": start, "end": actual_end, "timestamp": utc_now()})
        state["refined_ranges"] = ranges[-100:]
        self._save(state)
        result = {
            "status": "ok",
            "summary": "source range reread at fine granularity",
            "session_id": session_id,
            "path": str(source),
            "byte_offset_start": start,
            "byte_offset_end": actual_end,
            "requested_end": requested_end,
            "next_start": next_start,
            "content": text,
        }
        self.records.emit("attention_range_refined", **{k: v for k, v in result.items() if k != "content"})
        return result
