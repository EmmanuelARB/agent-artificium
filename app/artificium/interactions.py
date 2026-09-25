from __future__ import annotations

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import ConfigStore, configured_paths
from .filesystem import (
    Paths,
    atomic_write_json,
    file_lock,
    read_json,
    safe_identifier,
    sortable_id,
    utc_now,
)
from .records import Records


@dataclass
class Notification:
    id: str
    created_at: str
    type: str
    summary: str
    source: str
    path: str | None
    metadata: dict[str, Any]
    queue_path: Path | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], path: Path | None = None) -> "Notification":
        return cls(
            id=str(value["id"]),
            created_at=str(value["created_at"]),
            type=str(value["type"]),
            summary=str(value["summary"]),
            source=str(value.get("source") or "unknown"),
            path=str(value["path"]) if value.get("path") else None,
            metadata=dict(value.get("metadata") or {}),
            queue_path=path,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"queue_path": None}

    @property
    def is_runtime_notice(self) -> bool:
        return self.source == "artificium_runtime" and self.type in {"wake", "recovery"}

class NotificationStore:
    def __init__(self, paths: Paths, records: Records):
        self.paths = paths
        self.records = records

    def create(
        self,
        *,
        type: str,
        summary: str,
        source: str,
        path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Notification:
        item = Notification(
            id=sortable_id("notification_"),
            created_at=utc_now(),
            type=type,
            summary=summary,
            source=source,
            path=path,
            metadata=metadata or {},
        )
        destination = self.paths.notifications_new / f"{item.id}.json"
        atomic_write_json(destination, item.to_dict())
        item.queue_path = destination
        self.records.emit("notification_created", notification=item.to_dict())
        return item

    def has_new(self) -> bool:
        return any(self.paths.notifications_new.glob("*.json"))

    def claim(self, limit: int, *, include_runtime_notices: bool = False) -> list[Notification]:
        """Claim a normal batch, optionally including all internal wake/recovery notices.

        The runtime renders those notices as one bounded summary. Their original
        queue files still follow the usual release/commit transaction.
        """
        claimed: list[Notification] = []
        if limit <= 0:
            return claimed
        ordinary_count = 0
        for source in sorted(self.paths.notifications_new.glob("*.json")):
            if ordinary_count >= limit and not include_runtime_notices:
                break
            value = read_json(source)
            item = Notification.from_dict(value, source) if isinstance(value, dict) else None
            extra_notice = include_runtime_notices and item is not None and item.is_runtime_notice
            if ordinary_count >= limit and not extra_notice:
                continue
            destination = self.paths.notifications_processing / source.name
            try:
                source.replace(destination)
            except FileNotFoundError:
                continue
            if item is not None:
                item.queue_path = destination
                claimed.append(item)
            else:
                destination.replace(self.paths.notifications_delivered / destination.name)
            if not extra_notice:
                ordinary_count += 1
        return claimed

    def commit(self, item: Notification) -> None:
        if item.queue_path and item.queue_path.exists():
            destination = self.paths.notifications_delivered / item.queue_path.name
            item.queue_path.replace(destination)
            item.queue_path = destination
        self.records.emit("notification_delivered", notification_id=item.id)
        event_id = item.metadata.get("event_id")
        if event_id:
            receipt_path = self.paths.receipts / f"{event_id}.json"
            receipt = read_json(receipt_path, {})
            if isinstance(receipt, dict) and receipt:
                receipt["delivered_at"] = receipt.get("delivered_at") or utc_now()
                atomic_write_json(receipt_path, receipt)

    def release(self, item: Notification) -> None:
        if item.queue_path and item.queue_path.exists():
            destination = self.paths.notifications_new / item.queue_path.name
            item.queue_path.replace(destination)
            item.queue_path = destination

    def recover(self) -> int:
        count = 0
        for source in self.paths.notifications_processing.glob("*.json"):
            source.replace(self.paths.notifications_new / source.name)
            count += 1
        return count


class InteractionStore:
    def __init__(self, paths: Paths, notifications: NotificationStore, records: Records):
        self.paths = paths
        self.notifications = notifications
        self.records = records

    def _directory(self, interaction_id: str) -> Path:
        return self.paths.interactions / safe_identifier(
            interaction_id, label="interaction id"
        )

    def _meta_path(self, interaction_id: str) -> Path:
        return self._directory(interaction_id) / "interaction.json"

    def _event_path(self, interaction_id: str, event_id: str) -> Path:
        return self._directory(interaction_id) / "events" / f"{safe_identifier(event_id)}.json"

    def _receipt_path(self, event_id: str) -> Path:
        return self.paths.receipts / f"{safe_identifier(event_id, label='event id')}.json"

    def ensure(
        self,
        interaction_id: str,
        *,
        name: str | None = None,
        participants: Sequence[str] = (),
    ) -> dict[str, Any]:
        directory = self._directory(interaction_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "events").mkdir(exist_ok=True)
        (directory / "attachments").mkdir(exist_ok=True)
        path = self._meta_path(interaction_id)
        existing = read_json(path, {})
        if not isinstance(existing, dict):
            existing = {}
        known = [str(item) for item in existing.get("participants", [])]
        for participant in participants:
            clean = safe_identifier(participant, label="entity id")
            if clean not in known:
                known.append(clean)
        value = {
            "id": interaction_id,
            "name": name or existing.get("name") or interaction_id,
            "participants": known,
            "created_at": existing.get("created_at") or utc_now(),
            "updated_at": utc_now(),
            "status": existing.get("status") or "open",
        }
        atomic_write_json(path, value)
        return value

    def _copy_attachments(
        self, interaction_id: str, event_id: str, attachments: Sequence[str | Path]
    ) -> list[str]:
        result: list[str] = []
        destination_root = self._directory(interaction_id) / "attachments"
        for index, supplied in enumerate(attachments):
            source = Path(supplied).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            safe_name = "".join(
                character if character.isalnum() or character in "._-" else "_"
                for character in source.name
            )
            destination = destination_root / f"{event_id}_{index}_{safe_name}"
            shutil.copy2(source, destination)
            result.append(str(destination.relative_to(self.paths.root)))
        return result

    def add_event(
        self,
        interaction_id: str,
        *,
        sender: str,
        content: str,
        direction: str = "inbound",
        kind: str = "message",
        name: str | None = None,
        participants: Sequence[str] = (),
        attachments: Sequence[str | Path] = (),
        in_reply_to: str | None = None,
        recipient: str | None = None,
    ) -> tuple[dict[str, Any], Path]:
        interaction_id = safe_identifier(interaction_id, label="interaction id")
        sender = safe_identifier(sender, label="entity id")
        if direction not in {"inbound", "outbound"}:
            raise ValueError("direction must be inbound or outbound")
        event_id = sortable_id("event_")
        with file_lock(self.paths.interaction_lock):
            metadata = self.ensure(
                interaction_id,
                name=name,
                participants=[*participants, sender],
            )
            copied = self._copy_attachments(interaction_id, event_id, attachments)
            event = {
                "id": event_id,
                "interaction_id": interaction_id,
                "interaction_name": metadata["name"],
                "created_at": utc_now(),
                "sender": sender,
                "recipient": recipient or ("artificium" if direction == "inbound" else None),
                "direction": direction,
                "kind": kind,
                "content": content,
                "attachments": copied,
                "in_reply_to": in_reply_to,
            }
            path = self._event_path(interaction_id, event_id)
            atomic_write_json(path, event)
            if direction == "inbound":
                self._enqueue_event(event, path)
            else:
                atomic_write_json(
                    self._receipt_path(event_id),
                    {
                        "event_id": event_id,
                        "event_path": str(path),
                        "direction": direction,
                        "created_at": event["created_at"],
                        "seen_at": event["created_at"],
                        "handled_at": event["created_at"],
                    },
                )
                if in_reply_to:
                    self.mark_handled(in_reply_to, reply_event_id=event_id)
        self.records.emit("interaction_event_created", event=event, path=str(path))
        return event, path

    def _entity_history(self, entity_id: str) -> dict[str, Any]:
        interaction_ids: list[str] = []
        event_count = 0
        for metadata_path in sorted(self.paths.interactions.glob("*/interaction.json")):
            metadata = read_json(metadata_path, {})
            if not isinstance(metadata, dict) or entity_id not in metadata.get("participants", []):
                continue
            interaction_ids.append(str(metadata.get("id") or metadata_path.parent.name))
            for event_path in (metadata_path.parent / "events").glob("*.json"):
                event = read_json(event_path, {})
                if isinstance(event, dict) and event.get("sender") == entity_id:
                    event_count += 1
        return {"interaction_ids": interaction_ids, "event_count": event_count}

    def _enqueue_event(self, event: dict[str, Any], path: Path) -> Notification:
        sender = str(event["sender"])
        history = self._entity_history(sender)
        entity_memory_root = self.paths.memory / "entities" / safe_identifier(
            sender, label="entity id"
        )
        entity_memory_exists = entity_memory_root.is_dir() and any(
            candidate.is_file() and candidate.name != "index.txt"
            for candidate in entity_memory_root.rglob("*")
        )
        receipt = {
            "event_id": event["id"],
            "event_path": str(path),
            "direction": "inbound",
            "created_at": event["created_at"],
            "notified_at": utc_now(),
            "seen_at": None,
            "handled_at": None,
            "reply_event_id": None,
        }
        item = self.notifications.create(
            type="interaction_event",
            source=sender,
            summary=(
                f"A new {event.get('kind', 'event')} arrived from entity `{sender}` in "
                f"interaction `{event['interaction_id']}`. Read it when appropriate."
            ),
            path=str(path.relative_to(self.paths.root)),
            metadata={
                "event_id": event["id"],
                "interaction_id": event["interaction_id"],
                "interaction_name": event.get("interaction_name"),
                "entity_id": sender,
                "recipient": event.get("recipient") or "artificium",
                "event_kind": event.get("kind", "message"),
                "in_reply_to": event.get("in_reply_to"),
                "attachment_count": len(event.get("attachments") or []),
                "entity_event_count": history["event_count"],
                "entity_interaction_count": len(history["interaction_ids"]),
                "previous_interactions": history["interaction_ids"][-10:],
                "possible_entity_memory": f"memory/entities/{sender}/",
                "entity_memory_exists": entity_memory_exists,
                "is_first_entity_event": history["event_count"] == 1,
                "created_at": event["created_at"],
            },
        )
        receipt["notification_id"] = item.id
        atomic_write_json(self._receipt_path(str(event["id"])), receipt)
        return item

    def read_event(self, event_id: str) -> dict[str, Any]:
        event_id = safe_identifier(event_id, label="event id")
        receipt_path = self._receipt_path(event_id)
        receipt = read_json(receipt_path, {})
        path: Path | None = None
        if isinstance(receipt, dict) and receipt.get("event_path"):
            path = Path(str(receipt["event_path"]))
        if path is None or not path.is_file():
            matches = list(self.paths.interactions.glob(f"*/events/{event_id}.json"))
            if not matches:
                raise FileNotFoundError(f"Unknown interaction event: {event_id}")
            path = matches[0]
            receipt = {
                "event_id": event_id,
                "event_path": str(path),
                "direction": "unknown",
                "created_at": utc_now(),
            }
        event = read_json(path)
        if not isinstance(event, dict):
            raise ValueError(f"Invalid interaction event file: {path}")
        receipt["seen_at"] = receipt.get("seen_at") or utc_now()
        atomic_write_json(receipt_path, receipt)
        self.records.emit("interaction_event_seen", event_id=event_id, path=str(path))
        return {
            "status": "seen",
            "event": event,
            "event_path": str(path),
            "receipt": receipt,
            "possible_entity_memory": f"memory/entities/{event.get('sender', 'unknown')}/",
        }

    def mark_handled(
        self, event_id: str, *, reply_event_id: str | None = None, decision: str = "handled"
    ) -> dict[str, Any]:
        path = self._receipt_path(event_id)
        receipt = read_json(path, {})
        if not isinstance(receipt, dict) or not receipt:
            raise FileNotFoundError(f"Unknown interaction receipt: {event_id}")
        if decision == "postponed":
            receipt["postponed_at"] = utc_now()
            receipt["handled_at"] = None
        else:
            receipt["handled_at"] = utc_now()
        receipt["decision"] = decision
        if reply_event_id:
            receipt["reply_event_id"] = reply_event_id
        atomic_write_json(path, receipt)
        self.records.emit(
            "interaction_event_handled",
            event_id=event_id,
            reply_event_id=reply_event_id,
            decision=decision,
        )
        return receipt

    def events(self, interaction_id: str) -> list[dict[str, Any]]:
        directory = self._directory(interaction_id) / "events"
        result: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            value = read_json(path)
            if isinstance(value, dict):
                result.append(value)
        return sorted(result, key=lambda item: str(item.get("created_at") or ""))

    def list_interactions(self, entity_id: str | None = None) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in sorted(self.paths.interactions.glob("*/interaction.json")):
            value = read_json(path)
            if not isinstance(value, dict):
                continue
            if entity_id and entity_id not in value.get("participants", []):
                continue
            value = dict(value)
            value["event_count"] = len(list((path.parent / "events").glob("*.json")))
            result.append(value)
        return result

    def pending_events(self, limit: int = 20) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in sorted(self.paths.receipts.glob("event_*.json")):
            receipt = read_json(path, {})
            if (
                not isinstance(receipt, dict)
                or receipt.get("direction") != "inbound"
                or receipt.get("handled_at")
            ):
                continue
            result.append(
                {
                    "event_id": receipt.get("event_id"),
                    "event_path": receipt.get("event_path"),
                    "notified_at": receipt.get("notified_at"),
                    "delivered_at": receipt.get("delivered_at"),
                    "seen_at": receipt.get("seen_at"),
                }
            )
            if len(result) >= limit:
                break
        return result

    def reconcile(self) -> int:
        """Recover inbound events written by a minimal client without a notification."""

        created = 0
        for path in self.paths.interactions.glob("*/events/*.json"):
            value = read_json(path)
            if not isinstance(value, dict) or value.get("direction") != "inbound":
                continue
            event_id = str(value.get("id") or "")
            if not event_id or self._receipt_path(event_id).exists():
                continue
            self._enqueue_event(value, path)
            created += 1
        if created:
            self.records.emit("interaction_events_reconciled", count=created)
        return created


class ArtificiumClient:
    """File-native interaction client. It never initializes a model engine."""

    def __init__(self, root: str | Path | None = None):
        # ``root`` is the installation directory that holds app/ and workspace/.
        self.paths = configured_paths(root)
        self.paths.ensure_layout()
        self.records = Records(self.paths)
        self.notifications = NotificationStore(self.paths, self.records)
        self.interactions = InteractionStore(self.paths, self.notifications, self.records)

    def send(
        self,
        interaction_id: str,
        *,
        sender: str,
        content: str,
        interaction_name: str | None = None,
        participants: Sequence[str] = (),
        attachments: Sequence[str | Path] = (),
        in_reply_to: str | None = None,
        kind: str = "message",
        recipient: str | None = "artificium",
    ) -> tuple[dict[str, Any], Path]:
        return self.interactions.add_event(
            interaction_id,
            sender=sender,
            content=content,
            direction="inbound",
            kind=kind,
            name=interaction_name,
            participants=participants,
            attachments=attachments,
            in_reply_to=in_reply_to,
            recipient=recipient,
        )

    def events(self, interaction_id: str) -> list[dict[str, Any]]:
        return self.interactions.events(interaction_id)

    def interactions_for(self, entity_id: str | None = None) -> list[dict[str, Any]]:
        return self.interactions.list_interactions(entity_id)
