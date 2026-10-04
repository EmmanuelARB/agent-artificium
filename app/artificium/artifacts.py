from __future__ import annotations

import datetime as dt
import os
import subprocess
from pathlib import Path
from typing import Any

from .filesystem import Paths, atomic_write_json, read_json, utc_now
from .records import Records


# Checking a large repository costs a `git status`; once every few minutes
# is enough for a reminder measured in hours.
CHECK_INTERVAL_SECONDS = 600
WORKSPACE_REPOSITORY_HOURS = 24.0


def _parse(value: Any) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _git(path: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *arguments],
            capture_output=True, text=True, timeout=20,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


class ArtifactTracker:
    """Reminds the agent of records it committed to keeping current.

    A file is stale when it has not changed for longer than its period. A
    Git repository is stale when it has uncommitted changes and its last
    commit is older than the period. The workspace root, when it is a
    repository, is watched without being registered: work left uncommitted
    for days is lost or unreviewable whatever the task.
    """

    def __init__(self, paths: Paths, records: Records):
        self.paths = paths
        self.records = records

    @property
    def path(self) -> Path:
        return self.paths.runtime / "tracked-artifacts.json"

    def _load(self) -> dict[str, Any]:
        value = read_json(self.path, {})
        if not isinstance(value, dict):
            value = {}
        value.setdefault("artifacts", {})
        return value

    def track(self, path: Path, update_every_hours: float, note: str,
              stop: bool) -> dict[str, Any]:
        state = self._load()
        key = str(path)
        if stop:
            removed = state["artifacts"].pop(key, None)
            atomic_write_json(self.path, state)
            return {"status": "untracked" if removed else "unchanged",
                    "summary": f"{key} is no longer tracked", "path": key}
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist; create the file (or initialize the "
                "repository) before tracking it"
            )
        if path.is_dir() and not (path / ".git").exists():
            raise ValueError(
                f"{path} is a directory without .git; track a file, or a Git "
                "repository to watch its uncommitted changes"
            )
        if not 0.25 <= float(update_every_hours) <= 24 * 30:
            raise ValueError("update_every_hours must be between 0.25 and 720")
        state["artifacts"][key] = {
            "kind": "repository" if path.is_dir() else "file",
            "update_every_hours": float(update_every_hours),
            "note": note.strip(),
            "tracked_at": utc_now(),
        }
        atomic_write_json(self.path, state)
        self.records.emit("artifact_tracked", path=key, artifact=state["artifacts"][key])
        return {
            "status": "tracked",
            "summary": f"{key} tracked; a reminder follows when it is more than "
                       f"{float(update_every_hours):g} h behind",
            "path": key,
        }

    def _watched(self, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        watched = {key: dict(value) for key, value in state["artifacts"].items()
                   if isinstance(value, dict)}
        root = str(self.paths.root)
        if root not in watched and (self.paths.root / ".git").exists():
            watched[root] = {"kind": "repository",
                             "update_every_hours": WORKSPACE_REPOSITORY_HOURS,
                             "note": "the workspace repository"}
        return watched

    @staticmethod
    def _behind_hours(key: str, item: dict[str, Any], now: dt.datetime) -> float | None:
        path = Path(key)
        if item.get("kind") == "repository":
            dirty = _git(path, "status", "--porcelain", "--untracked-files=no")
            if not dirty or not dirty.strip():
                return None
            last = _git(path, "log", "-1", "--format=%ct")
            if not last or not last.strip():
                return None
            since = dt.datetime.fromtimestamp(int(last.strip()), dt.timezone.utc)
        else:
            try:
                since = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
            except OSError:
                return None
        return max(0.0, (now - since).total_seconds() / 3600)

    def reminders(self, now: dt.datetime | None = None) -> list[str]:
        now = now or dt.datetime.now(dt.timezone.utc)
        state = self._load()
        checked = _parse(state.get("checked_at"))
        if checked and (now - checked).total_seconds() < CHECK_INTERVAL_SECONDS:
            return []
        state["checked_at"] = now.isoformat().replace("+00:00", "Z")
        notified = state.setdefault("notified", {})
        lines: list[str] = []
        for key, item in self._watched(state).items():
            period = float(item.get("update_every_hours") or WORKSPACE_REPOSITORY_HOURS)
            behind = self._behind_hours(key, item, now)
            if behind is None or behind < period:
                notified.pop(key, None)
                continue
            last = _parse(notified.get(key))
            if last and (now - last).total_seconds() < period * 3600:
                continue
            notified[key] = state["checked_at"]
            note = f" ({item['note']})" if item.get("note") else ""
            if item.get("kind") == "repository":
                lines.append(f"- {key}{note}: uncommitted changes; last commit "
                             f"{behind:.1f} h ago.")
            else:
                lines.append(f"- {key}{note}: unchanged for {behind:.1f} h; "
                             f"you meant to update it every {period:g} h.")
        atomic_write_json(self.path, state)
        if lines:
            self.records.emit("artifact_reminder", items=lines)
        return lines
