"""Artificium's editable reference scheduler tool.

The harness loads this module from ``mind/tools`` at startup and only supplies
generic supervision.  Keeping the implementation in the mind makes it a real,
inspectable example of an Artificium-built tool rather than a hidden feature of
the architecture.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

from artificium.filesystem import (
    Paths,
    atomic_write_json,
    file_lock,
    read_json,
    safe_identifier,
    sortable_id,
    utc_now,
)
from artificium.interactions import InteractionStore
from artificium.records import Records


SCHEDULER_INTERACTION_ID = "scheduler"
SCHEDULER_ENTITY_ID = "scheduler"


def _parse_time(value: str) -> dt.datetime:
    supplied = value.strip()
    if not supplied:
        raise ValueError("run_at cannot be empty")
    try:
        parsed = dt.datetime.fromisoformat(supplied.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "run_at must be an ISO 8601 timestamp with a timezone, for example "
            "2026-08-19T18:30:00Z"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError("run_at must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _format_time(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


class Scheduler:
    """Persist tasks and emit ordinary inbound events when they become due.

    It has no model loop, entity routing, or second notification system. The
    Artificium runtime calls ``fire_due`` at its normal polling boundary.
    """

    def __init__(
        self,
        paths: Paths,
        interactions: InteractionStore,
        records: Records,
    ):
        self.paths = paths
        self.interactions = interactions
        self.records = records
        self.paths.scheduler_tasks.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        identifier = safe_identifier(task_id, label="scheduled task id")
        return self.paths.scheduler_tasks / f"{identifier}.json"

    @staticmethod
    def _clean(value: str, *, field: str, maximum: int) -> str:
        result = value.strip()
        if not result:
            raise ValueError(f"{field} cannot be empty")
        if len(result) > maximum:
            raise ValueError(f"{field} cannot exceed {maximum:,} characters")
        return result

    def schedule(
        self,
        *,
        name: str,
        description: str,
        text: str,
        run_at: str,
        repeat_seconds: float | None = None,
    ) -> dict[str, Any]:
        normalized_time = _format_time(_parse_time(run_at))
        repeat = None if repeat_seconds is None else float(repeat_seconds)
        if repeat is not None and repeat <= 0:
            raise ValueError("repeat_seconds must be positive or omitted")
        task_id = sortable_id("task_")
        created = utc_now()
        task = {
            "id": task_id,
            "name": self._clean(name, field="name", maximum=200),
            "description": self._clean(
                description, field="description", maximum=2_000
            ),
            "text": self._clean(text, field="text", maximum=100_000),
            "run_at": normalized_time,
            "repeat_seconds": repeat,
            "status": "pending",
            "created_at": created,
            "updated_at": created,
            "last_fired_at": None,
            "last_event_id": None,
            "fire_count": 0,
        }
        with file_lock(self.paths.scheduler_lock):
            atomic_write_json(self._path(task_id), task)
        self.records.emit(
            "scheduler_task_scheduled",
            task_id=task_id,
            name=task["name"],
            run_at=normalized_time,
            repeat_seconds=repeat,
        )
        return {
            "status": "scheduled",
            "summary": f"scheduled task `{task_id}` for {normalized_time}",
            "task": task,
        }

    def list(self, status: str = "pending", limit: int = 100) -> dict[str, Any]:
        if status not in {"pending", "completed", "cancelled", "all"}:
            raise ValueError("status must be pending, completed, cancelled, or all")
        maximum = max(1, min(int(limit), 1_000))
        tasks: list[dict[str, Any]] = []
        with file_lock(self.paths.scheduler_lock):
            for path in self.paths.scheduler_tasks.glob("task_*.json"):
                task = read_json(path, {})
                if not isinstance(task, dict) or not task.get("id"):
                    continue
                if status != "all" and task.get("status") != status:
                    continue
                tasks.append(task)
        tasks.sort(key=lambda item: (str(item.get("run_at") or ""), str(item["id"])))
        tasks = tasks[:maximum]
        return {
            "status": "ok",
            "summary": f"found {len(tasks)} {status} scheduled tasks",
            "tasks": tasks,
        }

    def cancel(self, task_id: str) -> dict[str, Any]:
        path = self._path(task_id)
        with file_lock(self.paths.scheduler_lock):
            task = read_json(path, {})
            if not isinstance(task, dict) or not task:
                raise FileNotFoundError(f"Unknown scheduled task: {task_id}")
            current = str(task.get("status") or "pending")
            if current == "cancelled":
                return {
                    "status": "already_cancelled",
                    "summary": f"scheduled task `{task_id}` is already cancelled",
                    "task": task,
                }
            if current == "completed":
                return {
                    "status": "already_completed",
                    "summary": f"scheduled task `{task_id}` has already completed",
                    "task": task,
                }
            task.update(
                {
                    "status": "cancelled",
                    "cancelled_at": utc_now(),
                    "updated_at": utc_now(),
                }
            )
            atomic_write_json(path, task)
        self.records.emit(
            "scheduler_task_cancelled", task_id=task_id, name=task.get("name")
        )
        return {
            "status": "cancelled",
            "summary": f"cancelled scheduled task `{task_id}`",
            "task": task,
        }

    @staticmethod
    def _event_content(task: dict[str, Any]) -> str:
        recurrence = (
            f"every {task['repeat_seconds']:g} seconds"
            if task.get("repeat_seconds") is not None
            else "one time"
        )
        return (
            "[SCHEDULED TASK DUE]\n"
            f"Task ID: {task['id']}\n"
            f"Name: {task['name']}\n"
            f"Description: {task['description']}\n"
            f"Scheduled for: {task['run_at']}\n"
            f"Recurrence: {recurrence}\n\n"
            "Instruction written when the task was scheduled:\n"
            f"{task['text']}"
        )

    @staticmethod
    def _next_run(task: dict[str, Any], now: dt.datetime) -> str:
        interval = float(task["repeat_seconds"])
        prior = _parse_time(str(task["run_at"]))
        elapsed = max(0.0, (now - prior).total_seconds())
        steps = max(1, math.floor(elapsed / interval) + 1)
        return _format_time(prior + dt.timedelta(seconds=steps * interval))

    def fire_due(
        self, *, now: dt.datetime | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        current = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
        maximum = max(1, min(int(limit), 1_000))
        fired: list[dict[str, Any]] = []
        with file_lock(self.paths.scheduler_lock):
            due: list[tuple[dt.datetime, Path, dict[str, Any]]] = []
            for path in self.paths.scheduler_tasks.glob("task_*.json"):
                task = read_json(path, {})
                if not isinstance(task, dict) or task.get("status") != "pending":
                    continue
                scheduled = _parse_time(str(task.get("run_at") or ""))
                if scheduled <= current:
                    due.append((scheduled, path, task))
            due.sort(key=lambda item: (item[0], str(item[2].get("id") or "")))

            for _, path, task in due[:maximum]:
                scheduled_for = str(task["run_at"])
                event, event_path = self.interactions.add_event(
                    SCHEDULER_INTERACTION_ID,
                    sender=SCHEDULER_ENTITY_ID,
                    recipient="artificium",
                    content=self._event_content(task),
                    direction="inbound",
                    kind="scheduled_task",
                    name="Scheduler",
                    participants=("artificium",),
                )
                fired_at = _format_time(current)
                task["fire_count"] = int(task.get("fire_count", 0) or 0) + 1
                task["last_fired_at"] = fired_at
                task["last_event_id"] = event["id"]
                task["updated_at"] = fired_at
                if task.get("repeat_seconds") is None:
                    task["status"] = "completed"
                    task["completed_at"] = fired_at
                else:
                    task["run_at"] = self._next_run(task, current)
                atomic_write_json(path, task)
                self.records.emit(
                    "scheduler_task_fired",
                    task_id=task["id"],
                    name=task["name"],
                    event_id=event["id"],
                    interaction_id=SCHEDULER_INTERACTION_ID,
                    scheduled_for=scheduled_for,
                    fired_at=fired_at,
                    next_run_at=task.get("run_at") if task.get("status") == "pending" else None,
                )
                fired.append(
                    {
                        "task": dict(task),
                        "event": event,
                        "event_path": str(event_path),
                    }
                )
        return fired
