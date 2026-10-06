from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .filesystem import Paths, atomic_write_json, read_json, safe_identifier, utc_now
from .records import Records


_WHITESPACE = re.compile(r"\s+")
# A word that looks like a file name or path: something.ext, dir/file.ext.
_FILE_TOKEN = re.compile(r"[\w./~-]*\w\.[A-Za-z0-9]{1,8}\b")
MAX_ACTIVE_DIRECTIVES = 40
MAX_QUOTE_CHARACTERS = 2_000
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


class DirectiveStore:
    """Standing instructions quoted verbatim from inbound interaction events.

    A paraphrased constraint drifts: an agent that summarized "no hand-seeded
    values" later carved itself an exception the entity never granted. The
    store therefore accepts only text that occurs in the cited event, and only
    a later event from the same entity can retire it.

    A directive may also quote a file the cited event names (a brief such as
    a task description): rules read once at the start of a long run fade
    from a context that is offloaded hundreds of times.
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

    @staticmethod
    def _source_text(source: Path) -> str | None:
        try:
            if source.stat().st_size > MAX_SOURCE_BYTES:
                return None
            return source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def referenced_files(self, text: str, limit: int = 5) -> list[Path]:
        """Workspace files a message names, so the agent can treat a brief
        as binding without being told to.

        Only the workspace root, `mind/space/`, and paths written out in the
        message are looked up; the workspace itself is never walked.
        """
        root = self.paths.root.resolve()
        found: list[Path] = []
        for token in dict.fromkeys(_FILE_TOKEN.findall(text)):
            token = token.rstrip(".")
            given = Path(token).expanduser()
            candidates = ([given] if given.is_absolute()
                          else [root / given, self.paths.space / given])
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                    resolved.relative_to(root)
                except (OSError, ValueError):
                    continue
                if resolved.is_file() and resolved not in found:
                    if self._source_text(resolved) is not None:
                        found.append(resolved)
                    break
            if len(found) >= limit:
                break
        return found

    def unrecorded_briefs(self, event: dict[str, Any]) -> list[Path]:
        """Files an inbound event names that no active directive quotes yet."""

        if event.get("direction") != "inbound":
            return []
        quoted = {str(item.get("source_path")) for item in self.active()
                  if item.get("source_path")}
        return [path for path in self.referenced_files(str(event.get("content") or ""))
                if str(path) not in quoted]

    def record(self, event_id: str, quote: str, source: Path | None = None) -> dict[str, Any]:
        event = self._inbound_event(event_id)
        wanted = _normalize(quote)
        if len(wanted) < 3:
            raise ValueError("quote is empty; copy the instruction's exact words")
        if len(wanted) > MAX_QUOTE_CHARACTERS:
            raise ValueError(
                f"quote exceeds {MAX_QUOTE_CHARACTERS} characters; record the "
                "binding sentences as separate directives"
            )
        content = str(event.get("content") or "")
        if source is None:
            if wanted not in _normalize(content):
                raise ValueError(
                    f"The quote does not occur in event `{event['id']}`. Copy the "
                    "entity's exact words (whitespace may differ); a paraphrase is "
                    "not accepted."
                )
        else:
            if source.name.lower() not in content.lower():
                raise ValueError(
                    f"Event `{event['id']}` does not name `{source.name}`. A file "
                    "can only be quoted through the message in which the entity "
                    "pointed you to it; cite that event."
                )
            text = self._source_text(source)
            if text is None:
                raise ValueError(
                    f"{source} is not a readable text file under "
                    f"{MAX_SOURCE_BYTES // 1024**2} MiB; pass the brief the entity named"
                )
            if wanted not in _normalize(text):
                raise ValueError(
                    f"The quote does not occur in {source}. Copy the file's exact "
                    "words (whitespace may differ); a paraphrase is not accepted."
                )
        source_path = str(source) if source is not None else None
        items = self._load()
        for item in items:
            if (item.get("status") == "active" and item.get("event_id") == event["id"]
                    and item.get("quote") == wanted
                    and item.get("source_path") == source_path):
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
        if source_path:
            directive["source_path"] = source_path
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
        lines = []
        for item in self.active():
            origin = f"event {item.get('event_id')}"
            note = ""
            if item.get("source_path"):
                source = Path(str(item["source_path"]))
                origin = f"{source.name} (via event {item.get('event_id')})"
                text = self._source_text(source)
                if text is None or str(item.get("quote")) not in _normalize(text):
                    # Kept, not dropped: whether the rule still holds is the
                    # entity's call, not a side effect of an edit.
                    note = f" (no longer in {source.name}; ask whether it still holds)"
            lines.append(f"- {item['id']} — {item.get('entity')}, {item.get('given_at')}, "
                         f"{origin}: \"{item.get('quote')}\"{note}")
        return "\n".join(lines)
