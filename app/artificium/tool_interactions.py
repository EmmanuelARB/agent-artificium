from __future__ import annotations

from typing import Any

from .filesystem import atomic_write_json


class InteractionToolsMixin:
    """Interaction-store and scheduler tools mixed into :class:`ToolRegistry`.

    Freely uses attributes/helpers defined there (``self.interactions``,
    ``self.scheduler``, ``self.config``, ``self.paths``, ``self.prompts``,
    ``self._resolve``).
    """

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
