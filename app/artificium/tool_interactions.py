from __future__ import annotations

import difflib
import inspect
from typing import Any

from .directives import DirectiveStore
from .filesystem import atomic_write_json, read_json, safe_identifier


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
        if isinstance(event, dict):
            notice = self._brief_notice(event)
            if notice:
                result["_notifications"] = [notice]
        return result

    def _brief_notice(self, event: dict[str, Any]) -> str | None:
        """Point out, once per event, workspace files a message names.

        An instruction like "follow the rules in FILE" is easy to acknowledge
        and then lose: the rules sit in the file, not in the message.
        """
        store = DirectiveStore(self.paths, self.records)
        files = store.unrecorded_briefs(event)
        if not files:
            return None
        seen_path = self.paths.runtime / "brief-notices.json"
        seen = read_json(seen_path, [])
        seen = seen if isinstance(seen, list) else []
        if event.get("id") in seen:
            return None
        atomic_write_json(seen_path, (seen + [event.get("id")])[-500:])
        return self.prompts.event(
            "brief_referenced",
            event_id=event.get("id"),
            entity=event.get("sender"),
            files="\n".join(f"- {path}" for path in files),
        )

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
        new_interaction: bool = False,
        decision: bool = False,
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        refusal = self._check_interaction_target(
            interaction_id, in_reply_to, new_interaction
        ) or self._check_decision(decision, options)
        if refusal:
            return refusal
        extra: dict[str, Any] = {}
        if decision:
            extra["kind"] = "decision"
            if options:
                if "options" in inspect.signature(self.interactions.add_event).parameters:
                    extra["options"] = options
                else:
                    # An overlay of an older store cannot keep the field; the
                    # choices still reach the user as part of the text.
                    content += "\n\nOptions:\n" + "\n".join(
                        f"{number}. {item}" for number, item in enumerate(options, 1))
        event, path = self.interactions.add_event(
            interaction_id,
            sender=sender or self.config.instance_id,
            recipient=recipient,
            content=content,
            direction="outbound",
            attachments=[str(self._resolve(item)) for item in (attachments or [])],
            in_reply_to=in_reply_to,
            **extra,
        )
        return {
            "status": "sent",
            "summary": ("decision sent; it stays pinned for the user until they "
                        "answer it" if decision else "outbound interaction event written"),
            "path": str(path),
            "event": event,
            "_notifications": [
                self.prompts.event(
                    "memory_opportunity",
                    boundary_reason=f"an outbound event was sent in interaction {interaction_id}",
                )
            ],
        }

    @staticmethod
    def _check_decision(decision: bool, options: list[str] | None) -> dict[str, Any] | None:
        """Refuse malformed decision arguments before anything is sent."""

        if options is None:
            return None
        if not decision:
            return {
                "status": "error",
                "summary": ("options only apply to a decision; set decision to true, "
                            "or drop options. Nothing was sent."),
            }
        valid = (isinstance(options, list) and 2 <= len(options) <= 8
                 and all(isinstance(item, str) and 0 < len(item.strip()) <= 200
                         for item in options))
        if not valid:
            return {
                "status": "error",
                "summary": ("options must be 2 to 8 non-empty strings of at most 200 "
                            "characters; nothing was sent."),
            }
        return None

    def _check_interaction_target(
        self, interaction_id: str, in_reply_to: str | None, new_interaction: bool
    ) -> dict[str, Any] | None:
        """Refuse a send whose destination is probably a mistyped ID.

        A reply written to a misspelled interaction ID used to create a new
        stream silently and report success, so the entity never saw it.
        Starting a stream therefore has to be explicit.
        """

        interaction_id = safe_identifier(interaction_id, label="interaction id")
        root = self.paths.interactions
        if in_reply_to:
            event_id = safe_identifier(in_reply_to, label="event id")
            owners = sorted(path.parent.parent.name
                            for path in root.glob(f"*/events/{event_id}.json"))
            if not owners:
                return {
                    "status": "error",
                    "summary": (
                        f"in_reply_to `{in_reply_to}` is not a known interaction "
                        "event; nothing was sent. Copy the exact event ID from "
                        "the notification or read_interaction_event."
                    ),
                }
            if interaction_id not in owners:
                return {
                    "status": "error",
                    "summary": (
                        f"Event `{in_reply_to}` belongs to interaction "
                        f"`{owners[0]}`, not `{interaction_id}`; nothing was "
                        f"sent. Reply with interaction_id `{owners[0]}`."
                    ),
                    "expected_interaction_id": owners[0],
                }
        if new_interaction or (root / interaction_id / "interaction.json").is_file():
            return None
        known = sorted(path.parent.name for path in root.glob("*/interaction.json"))
        close = difflib.get_close_matches(interaction_id, known, n=3, cutoff=0.6)
        hint = (f" Did you mean {', '.join(f'`{item}`' for item in close)}?"
                if close else "")
        return {
            "status": "error",
            "summary": (
                f"Interaction `{interaction_id}` does not exist; nothing was sent."
                f"{hint} Copy the exact ID to reply, or set new_interaction "
                "to true to start a new stream deliberately."
            ),
            "similar_interaction_ids": close,
            "known_interaction_ids": known[:20],
        }

    def record_directive(
        self, event_id: str, quote: str, source_path: str | None = None
    ) -> dict[str, Any]:
        source = self._resolve(source_path) if source_path else None
        store = DirectiveStore(self.paths, self.records)
        result = store.record(event_id, quote, source)
        if source is None:
            # Recording "follow the file" without the file's rules is the
            # half-step this catches; the notice fires once per event.
            notice = self._brief_notice(store._inbound_event(event_id))
            if notice:
                result["_notifications"] = [notice]
        return result

    def retire_directive(self, directive_id: str, event_id: str) -> dict[str, Any]:
        return DirectiveStore(self.paths, self.records).retire(directive_id, event_id)

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
