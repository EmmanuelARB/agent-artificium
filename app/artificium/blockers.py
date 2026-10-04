from __future__ import annotations

import datetime as dt
import re
from typing import Any

from .filesystem import Paths, atomic_write_json, read_json, utc_now
from .records import Records


PROGRESS_VALUES = ("advanced", "supporting", "none")
# A notice after this many consecutive checkpoints on one blocker (or without
# an advance), then again after as many more. Twelve hours on one blocker
# also qualifies once a few checkpoints confirm it is the same one.
STALL_CHECKPOINTS = 6
STALL_HOURS = 12.0
STALL_HOURS_MIN_CHECKPOINTS = 3
HISTORY_LIMIT = 200


def _normalize(blocker: str) -> str:
    return re.sub(r"\s+", " ", blocker).strip().lower()


def _hours_between(start: str, end: str) -> float:
    try:
        first = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
        last = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max(0.0, (last - first).total_seconds() / 3600)


class BlockerTracker:
    """Counts working-memory-offload checkpoints spent on the same blocker.

    An agent deep in one obstacle reframes it every few hours and reports
    each reframe as progress, so neither the checkpoint text nor its own
    sense of advance shows that days have gone into the same wall. The
    harness counts what the checkpoints declare instead.
    """

    def __init__(self, paths: Paths, records: Records):
        self.paths = paths
        self.records = records

    @property
    def path(self):
        return self.paths.runtime / "blocker-history.json"

    @staticmethod
    def validate(objective_progress: str) -> None:
        if objective_progress and objective_progress not in PROGRESS_VALUES:
            raise ValueError(
                "objective_progress must be advanced, supporting, or none "
                "(or omitted)"
            )

    def record(self, blocker: str, objective_progress: str) -> dict[str, Any]:
        state = read_json(self.path, {})
        state = state if isinstance(state, dict) else {}
        history = [item for item in state.get("history") or [] if isinstance(item, dict)]
        entry = {
            "at": utc_now(),
            "blocker": blocker.strip(),
            "objective_progress": objective_progress or None,
        }
        history = (history + [entry])[-HISTORY_LIMIT:]

        key = _normalize(blocker)
        same = 0
        if key:
            for item in reversed(history):
                if _normalize(str(item.get("blocker") or "")) != key:
                    break
                same += 1
        without_advance = 0
        for item in reversed(history):
            if item.get("objective_progress") not in ("supporting", "none"):
                break
            without_advance += 1
        streak = max(same, without_advance)
        first = history[-streak]["at"] if streak else entry["at"]
        hours = _hours_between(first, entry["at"])

        # A streak is identified by its first checkpoint, which stays fixed
        # while the streak lasts even when the blocker's wording changes.
        notified = state.get("notified_streak")
        if state.get("notified_since") != first or not isinstance(notified, int):
            notified = 0
        due = streak >= STALL_CHECKPOINTS or (
            hours >= STALL_HOURS and streak >= STALL_HOURS_MIN_CHECKPOINTS
        )
        stalled = due and (not notified or streak - notified >= STALL_CHECKPOINTS)
        if stalled:
            notified = streak
        atomic_write_json(self.path, {
            "history": history,
            "notified_since": first,
            "notified_streak": notified,
        })
        result = {
            "blocker": entry["blocker"] or None,
            "same_blocker_checkpoints": same,
            "checkpoints_without_advance": without_advance,
            "since": first,
            "hours": round(hours, 1),
            "stalled": stalled,
        }
        if stalled:
            self.records.emit("stalled_blocker_noticed", **result)
        return result
