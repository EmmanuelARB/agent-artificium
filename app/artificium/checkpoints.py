"""Tiered working-memory checkpoints.

Every offload writes one checkpoint under ``mind/memory/context/``, so a
long-lived instance accumulates one file per offload forever. The newest
``harness.checkpoint_keep_verbatim`` checkpoints stay verbatim; older ones are
merged into a single older-tier checkpoint whose content is a fresh-context
summary of their summaries (``recovery.summarize``, one bounded request).

Only files the harness registered as offload checkpoints are merged, and only
inside the checkpoint directory. The raw context archives under
``logs/context/`` are never read, changed, or deleted: they remain the durable
fallback for anything a summary drops.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .filesystem import (
    Paths,
    atomic_write_json,
    atomic_write_text,
    read_json,
    sortable_id,
    utc_now,
)
from .records import Console, Records

# Sources per merge. Bounded input keeps each summarizer request cheap and
# small enough to fit any window without the helper having to truncate it.
MERGE_BATCH = 24

# Merges per startup backlog pass. The pass is resumable, so a large backlog
# never delays a start for long; the next start continues it.
BACKLOG_MERGES_PER_RUN = 10

# A backlog is only migrated when it is this many times the verbatim limit.
BACKLOG_FACTOR = 10

# The merged tier's file name prefix, below mind/memory/context/.
TIER_PREFIX = "older-tier"


def excerpt(text: str, limit: int) -> str:
    """Keep the head and tail of one source; the middle is left to the archive."""

    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = max(0, limit - head)
    omitted = len(text) - head - tail
    return (
        text[:head].rstrip()
        + f"\n[{omitted} characters omitted from this checkpoint]\n"
        + text[-tail:].lstrip()
    )


class CheckpointTiering:
    """The checkpoint index and the older-tier merge over it."""

    def __init__(
        self,
        paths: Paths,
        config: Config,
        records: Records,
        console: Console | None = None,
        summarize: Callable[[list[dict[str, Any]]], str] | None = None,
    ):
        self.paths = paths
        self.config = config
        self.records = records
        self.console = console
        self.summarize = summarize

    # -- the index -----------------------------------------------------

    @property
    def keep(self) -> int:
        return max(0, int(getattr(self.config, "checkpoint_keep_verbatim", 3) or 0))

    def _relative(self, path: Path) -> str | None:
        try:
            return str(path.resolve().relative_to(self.paths.root.resolve()))
        except (OSError, ValueError):
            return None

    def _managed(self, path: Path) -> bool:
        """Only files inside the checkpoint directory are ever merged or deleted."""

        try:
            path.resolve().relative_to(self.paths.checkpoint_dir.resolve())
        except (OSError, ValueError):
            return False
        return True

    def _read_index(self) -> list[dict[str, Any]]:
        value = read_json(self.paths.checkpoint_index, {})
        items = value.get("checkpoints") if isinstance(value, dict) else None
        entries: list[dict[str, Any]] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            path = self.paths.root / str(item["path"])
            if not path.is_file() or not self._managed(path):
                # A file the agent deleted or moved is no longer a checkpoint.
                continue
            entries.append(
                {
                    "path": path,
                    "relative_path": str(item["path"]),
                    "created_at": str(item.get("created_at") or ""),
                    "tokens": int(item.get("tokens") or 0),
                    "tier": "merged" if item.get("tier") == "merged" else "verbatim",
                    "sources": list(item.get("sources") or []),
                }
            )
        return entries

    def _write_index(self, entries: list[dict[str, Any]]) -> None:
        atomic_write_json(
            self.paths.checkpoint_index,
            {
                "version": 1,
                "updated_at": utc_now(),
                "checkpoints": [
                    {
                        "path": entry["relative_path"],
                        "created_at": entry["created_at"],
                        "tokens": entry["tokens"],
                        "tier": entry["tier"],
                        "sources": entry["sources"],
                    }
                    for entry in entries
                ],
            },
        )

    def entries(self) -> list[dict[str, Any]]:
        """Registered checkpoints that still exist, oldest first."""

        entries = self._read_index()
        return sorted(entries, key=lambda item: (item["created_at"], item["relative_path"]))

    def register(
        self,
        path: Path,
        *,
        tokens: int = 0,
        created_at: str | None = None,
        tier: str = "verbatim",
        sources: list[str] | None = None,
    ) -> bool:
        """Record one checkpoint the harness just wrote.

        Paths outside ``mind/memory/context/`` are not tracked: the harness
        never merges or deletes a memory it did not write as a checkpoint.
        """

        path = Path(path)
        relative = self._relative(path)
        if relative is None or not self._managed(path):
            return False
        entries = [entry for entry in self._read_index() if entry["relative_path"] != relative]
        entries.append(
            {
                "path": path,
                "relative_path": relative,
                "created_at": created_at or utc_now(),
                "tokens": int(tokens or 0),
                "tier": "merged" if tier == "merged" else "verbatim",
                "sources": list(sources or []),
            }
        )
        self._write_index(entries)
        return True

    def adopt(self, *, write: bool = True) -> list[dict[str, Any]]:
        """Treat files already in the checkpoint directory as checkpoints.

        This is the one-time migration path for an instance that predates the
        index. Created time is the file's modification time, the only ordering
        an unregistered checkpoint has; file names are never used.
        """

        known = {entry["relative_path"] for entry in self._read_index()}
        adopted: list[dict[str, Any]] = []
        if not self.paths.checkpoint_dir.is_dir():
            return adopted
        for path in sorted(self.paths.checkpoint_dir.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name.startswith(".") or path.name == "index.txt":
                continue
            relative = self._relative(path)
            if relative is None or relative in known:
                continue
            try:
                created = path.stat().st_mtime
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            adopted.append(
                {
                    "path": path,
                    "relative_path": relative,
                    "created_at": _iso(created),
                    "tokens": self._tokens(text),
                    "tier": "verbatim",
                    "sources": [],
                }
            )
        if adopted and write:
            self._write_index(self._read_index() + adopted)
        return adopted

    # -- merging -------------------------------------------------------

    def _tokens(self, text: str) -> int:
        per_token = float(getattr(self.config, "chars_per_token", 4.0) or 4.0)
        return int(len(text) / per_token)

    def _merge(self, sources: list[dict[str, Any]]) -> dict[str, Any]:
        """Write one older-tier checkpoint from ``sources``, then retire them.

        The target is written and verified before any source is deleted, and a
        source is deleted only when the target names its path. A failure at any
        step therefore leaves every checkpoint in place.
        """

        if self.summarize is None:
            raise RuntimeError("no summarizer is available for checkpoint merging")
        payload = []
        for entry in sources:
            try:
                content = entry["path"].read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise RuntimeError(f"cannot read {entry['relative_path']}: {exc}") from None
            payload.append(
                {
                    "path": entry["relative_path"],
                    "created_at": entry["created_at"],
                    "tokens": entry["tokens"],
                    "content": content,
                }
            )
        summary = str(self.summarize(payload) or "").strip()
        if len(summary) < 40:
            raise RuntimeError("the merged summary was empty or too short")

        target = self.paths.checkpoint_dir / f"{TIER_PREFIX}--{sortable_id()}.txt"
        listing = "\n".join(
            f"- {entry['relative_path']} (created {entry['created_at']}, ~{entry['tokens']} tokens)"
            for entry in sources
        )
        content = (
            f"[TIERED CHECKPOINT — merged from {len(sources)} older checkpoints]\n"
            f"Merged: {utc_now()}\n"
            "Sources, now represented by this file only:\n"
            f"{listing}\n"
            "[END SOURCE LIST]\n\n"
            f"{summary}\n"
        )
        atomic_write_text(target, content)
        written = target.read_text(encoding="utf-8", errors="replace")
        missing = [entry["relative_path"] for entry in sources
                   if entry["relative_path"] not in written]
        if len(written) < 40 or missing:
            raise RuntimeError(
                "the merged checkpoint did not name every source; nothing was deleted"
            )

        # Index first: the target becomes the merged tier and the sources stop
        # being checkpoints. Only then are the source files removed, so an
        # interrupted merge never loses content.
        remaining = [
            entry for entry in self._read_index()
            if entry["relative_path"] not in {item["relative_path"] for item in sources}
        ]
        entry = {
            "path": target,
            "relative_path": self._relative(target) or str(target),
            "created_at": utc_now(),
            "tokens": self._tokens(written),
            "tier": "merged",
            "sources": [item["relative_path"] for item in sources],
        }
        self._write_index(remaining + [entry])
        deleted: list[str] = []
        for item in sources:
            try:
                if self._managed(item["path"]):
                    item["path"].unlink()
                    deleted.append(item["relative_path"])
            except OSError:
                # The content is already in the target; a leftover file is
                # harmless and will be adopted or ignored later.
                continue
        result = {
            "status": "merged",
            "sources": [item["relative_path"] for item in sources],
            "deleted": deleted,
            "target": entry["relative_path"],
            "tokens": entry["tokens"],
            "source_tokens": sum(item["tokens"] for item in sources),
        }
        self.records.emit("checkpoint_merged", **result)
        self.records.life("checkpoint_merged", **result)
        if self.console is not None:
            self.console.line(
                "memory",
                f"merged {len(sources)} old checkpoints into {entry['relative_path']}",
            )
        return {**entry, **result}

    def tier(
        self,
        *,
        dry_run: bool = False,
        limit: int = 0,
        adopt: bool = False,
    ) -> dict[str, Any]:
        """Merge until at most ``keep`` verbatim checkpoints remain.

        ``limit`` stops after that many merges (0 is no limit); the next call
        continues where this one stopped.
        """

        keep = self.keep
        if keep <= 0:
            return {"status": "disabled", "merges": 0, "keep": 0, "dry_run": dry_run}
        adopted = self.adopt(write=not dry_run) if adopt else []
        entries = self.entries()
        if adopted:
            known = {entry["relative_path"] for entry in entries}
            entries = sorted(
                entries + [entry for entry in adopted if entry["relative_path"] not in known],
                key=lambda item: (item["created_at"], item["relative_path"]),
            )
        verbatim = [entry for entry in entries if entry["tier"] != "merged"]
        merged = [entry for entry in entries if entry["tier"] == "merged"]
        merges = 0
        sources_merged = 0
        sources_seen: list[str] = []
        targets: list[str] = []
        errors: list[str] = []
        while len(verbatim) > keep or len(merged) > 1:
            batch = verbatim[: max(0, min(len(verbatim) - keep, MERGE_BATCH))]
            sources = batch + merged
            if not sources:
                break
            sources_seen.extend(entry["relative_path"] for entry in sources)
            if dry_run:
                merges += 1
                sources_merged += len(batch)
                targets.append(f"{TIER_PREFIX}--<new>.txt")
                verbatim = verbatim[len(batch):]
                merged = [{"tier": "merged", "relative_path": f"{TIER_PREFIX}--<new>.txt"}]
            else:
                try:
                    result = self._merge(sources)
                except Exception as exc:  # a merge is maintenance, never fatal
                    errors.append(f"{type(exc).__name__}: {exc}")
                    break
                merges += 1
                sources_merged += len(batch)
                targets.append(str(result["target"]))
                verbatim = verbatim[len(batch):]
                merged = [result]
            if limit and merges >= limit:
                break
        if dry_run:
            verbatim_left, merged_left = len(verbatim), len(merged)
        else:
            remaining = self.entries()
            verbatim_left = len([entry for entry in remaining if entry["tier"] != "merged"])
            merged_left = len([entry for entry in remaining if entry["tier"] == "merged"])
        report = {
            "status": "failed" if errors else "planned" if dry_run and merges
                      else "merged" if merges else "unchanged",
            "merges": merges,
            "sources_merged": sources_merged,
            "sources": sources_seen,
            "targets": targets,
            "target": targets[-1] if targets else None,
            "keep": keep,
            "verbatim": verbatim_left,
            "merged": merged_left,
            "adopted": len(adopted),
            "dry_run": dry_run,
        }
        if errors:
            report["errors"] = errors
        return report

    # -- the one-time backlog pass -------------------------------------

    def backlog(self, *, dry_run: bool = False, limit: int = 0) -> dict[str, Any]:
        """Migrate a pre-existing backlog of checkpoints, once.

        The marker is written only when the backlog is fully reduced, so an
        interrupted or failed pass resumes at the next start. A fresh instance
        with nothing to migrate never gets a marker.
        """

        state = read_json(self.paths.checkpoint_tiering_state, {})
        state = state if isinstance(state, dict) else {}
        if state.get("completed_at"):
            return {"status": "already_completed", "merges": 0,
                    "completed_at": state["completed_at"], "dry_run": dry_run}
        keep = self.keep
        if keep <= 0:
            return {"status": "disabled", "merges": 0, "keep": 0, "dry_run": dry_run}
        existing = len(self.entries()) + len(self.adopt(write=False))
        if not state and existing <= keep * BACKLOG_FACTOR:
            return {"status": "not_needed", "checkpoints": existing, "merges": 0,
                    "threshold": keep * BACKLOG_FACTOR, "dry_run": dry_run}
        started = state.get("started_at") or utc_now()
        if not dry_run:
            atomic_write_json(
                self.paths.checkpoint_tiering_state,
                {**state, "started_at": started, "updated_at": utc_now(),
                 "checkpoints_at_start": state.get("checkpoints_at_start") or existing},
            )
            self.records.emit(
                "checkpoint_tiering_started",
                checkpoints=existing,
                threshold=keep * BACKLOG_FACTOR,
                resumed=bool(state),
            )
        report = self.tier(dry_run=dry_run, limit=limit, adopt=True)
        report["backlog"] = True
        report["checkpoints_at_start"] = state.get("checkpoints_at_start") or existing
        if dry_run:
            return report
        remaining = self.entries()
        verbatim = len([entry for entry in remaining if entry["tier"] != "merged"])
        complete = verbatim <= keep and not report.get("errors")
        atomic_write_json(
            self.paths.checkpoint_tiering_state,
            {
                **state,
                "started_at": started,
                "updated_at": utc_now(),
                "checkpoints_at_start": state.get("checkpoints_at_start") or existing,
                "merges": int(state.get("merges") or 0) + report["merges"],
                "completed_at": utc_now() if complete else None,
                "remaining_verbatim": verbatim,
                "last_error": (report.get("errors") or [None])[-1],
            },
        )
        self.records.emit(
            "checkpoint_tiering_completed" if complete else "checkpoint_tiering_skipped",
            merges=report["merges"],
            remaining_verbatim=verbatim,
            keep=keep,
            error=(report.get("errors") or [None])[-1],
        )
        return report

    def report(self) -> dict[str, Any]:
        entries = self.entries()
        state = read_json(self.paths.checkpoint_tiering_state, {})
        state = state if isinstance(state, dict) else {}
        return {
            "keep": self.keep,
            "verbatim": len([entry for entry in entries if entry["tier"] != "merged"]),
            "merged": len([entry for entry in entries if entry["tier"] == "merged"]),
            "checkpoints": [
                {
                    "path": entry["relative_path"],
                    "created_at": entry["created_at"],
                    "tokens": entry["tokens"],
                    "tier": entry["tier"],
                }
                for entry in entries[-20:]
            ],
            "backlog": {
                "started_at": state.get("started_at"),
                "completed_at": state.get("completed_at"),
                "merges": state.get("merges", 0),
                "last_error": state.get("last_error"),
            },
            "index": str(self.paths.checkpoint_index),
        }


def _iso(epoch: float) -> str:
    return (
        dt.datetime.fromtimestamp(epoch, dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
