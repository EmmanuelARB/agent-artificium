from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .filesystem import Paths, atomic_write_json, read_json, safe_identifier, utc_now
from .records import Records


_WHITESPACE = re.compile(r"\s+")
MAX_ACTIVE_DIRECTIVES = 40
MAX_QUOTE_CHARACTERS = 2_000


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


class DirectiveStore:
    """Standing instructions quoted verbatim from inbound interaction events.

    A paraphrased constraint drifts: an agent that summarized "no hand-seeded
    values" later carved itself an exception the entity never granted. The
    store therefore accepts only text that occurs in the cited event, and only
    a later event from the same entity can retire it.
    """

    def __init__(self, paths: Paths, records: Records):
        self.paths = paths
        self.records = records

    @property
    def path(self) -> Path:
        return self.paths.runtime / "directives.json"

    def _load(self) -> list[dict[str, Any]]:
        value = read_json(self.path, {})
        items = value.get("directives") if isinstance(value, dict) else None
        return [item for item in items or [] if isinstance(item, dict)]

    def _save(self, items: list[dict[str, Any]]) -> None:
        atomic_write_json(self.path, {"directives": items})

    def active(self) -> list[dict[str, Any]]:
        return [item for item in self._load() if item.get("status") == "active"]

    def _inbound_event(self, event_id: str) -> dict[str, Any]:
        event_id = safe_identifier(event_id, label="event id")
        matches = sorted(self.paths.interactions.glob(f"*/events/{event_id}.json"))
        event = read_json(matches[0], {}) if matches else {}
        if not isinstance(event, dict) or not event:
            raise ValueError(
                f"Unknown interaction event `{event_id}`. Cite the exact event ID "
                "in which the entity gave the instruction."
            )
        if event.get("direction") != "inbound":
            raise ValueError(
                f"Event `{event_id}` is outbound; a directive must come from an "
                "entity's inbound event, not from your own message."
            )
        return event

    def record(self, event_id: str, quote: str) -> dict[str, Any]:
        event = self._inbound_event(event_id)
        wanted = _normalize(quote)
        if len(wanted) < 3:
            raise ValueError("quote is empty; copy the instruction's exact words")
        if len(wanted) > MAX_QUOTE_CHARACTERS:
            raise ValueError(
                f"quote exceeds {MAX_QUOTE_CHARACTERS} characters; record the "
                "binding sentences as separate directives"
            )
        if wanted not in _normalize(str(event.get("content") or "")):
            raise ValueError(
                f"The quote does not occur in event `{event['id']}`. Copy the "
                "entity's exact words (whitespace may differ); a paraphrase is "
                "not accepted."
            )
        items = self._load()
        for item in items:
            if (item.get("status") == "active" and item.get("event_id") == event["id"]
                    and item.get("quote") == wanted):
                return {"status": "unchanged",
                        "summary": f"already recorded as {item['id']}",
                        "directive": item}
        if sum(1 for item in items if item.get("status") == "active") >= MAX_ACTIVE_DIRECTIVES:
            raise ValueError(
                f"{MAX_ACTIVE_DIRECTIVES} directives are already active; retire "
                "superseded ones with retire_directive first"
            )
        directive = {
            "id": f"D{len(items) + 1}",
            "status": "active",
            "quote": wanted,
            "entity": event.get("sender"),
            "interaction_id": event.get("interaction_id"),
            "event_id": event["id"],
            "given_at": event.get("created_at"),
            "recorded_at": utc_now(),
        }
        items.append(directive)
        self._save(items)
        self.records.emit("directive_recorded", directive=directive)
        return {
            "status": "recorded",
            "summary": (f"{directive['id']} recorded; it is shown in every request "
                        "until its entity lifts it"),
            "directive": directive,
        }

    def retire(self, directive_id: str, event_id: str) -> dict[str, Any]:
        items = self._load()
        directive = next((item for item in items if item.get("id") == directive_id), None)
        if directive is None or directive.get("status") != "active":
            known = [item["id"] for item in items if item.get("status") == "active"]
            raise ValueError(f"No active directive `{directive_id}`; active: {known or 'none'}")
        event = self._inbound_event(event_id)
        if event.get("sender") != directive.get("entity"):
            raise ValueError(
                f"{directive_id} was given by `{directive.get('entity')}`; only an "
                f"event from that entity can lift it, not one from `{event.get('sender')}`. "
                "Ask that entity if the directive no longer fits."
            )
        if str(event.get("created_at") or "") <= str(directive.get("given_at") or ""):
            raise ValueError(
                f"Event `{event['id']}` predates {directive_id}; cite the later "
                "event in which the entity lifted or replaced it."
            )
        directive.update(status="retired", retired_by_event_id=event["id"],
                         retired_at=utc_now())
        self._save(items)
        self.records.emit("directive_retired", directive=directive)
        return {"status": "retired", "summary": f"{directive_id} retired",
                "directive": directive}

    def render(self) -> str:
        return "\n".join(
            f"- {item['id']} — {item.get('entity')}, {item.get('given_at')}, "
            f"event {item.get('event_id')}: \"{item.get('quote')}\""
            for item in self.active()
        )
