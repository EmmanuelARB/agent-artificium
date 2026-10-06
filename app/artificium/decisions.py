"""Which messages from the agent are waiting for an answer.

Pure functions over interaction events, shared by the store and the chat
client so both agree. An outbound event of kind ``decision`` is open until any
event replies to it; events written before decisions existed have no such
kind and never count.
"""

from __future__ import annotations

from typing import Any, Iterable


def open_decisions(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    events = [event for event in events if isinstance(event, dict)]
    answered = {event.get("in_reply_to") for event in events if event.get("in_reply_to")}
    return sorted(
        (event for event in events
         if event.get("direction") == "outbound"
         and event.get("kind") == "decision"
         and event.get("id") not in answered),
        key=lambda event: str(event.get("created_at") or ""),
    )


def decision_options(decision: dict[str, Any]) -> list[str]:
    options = decision.get("options")
    return [str(item) for item in options] if isinstance(options, list) else []


def answer_text(decision: dict[str, Any], text: str) -> str:
    """A bare option number stands for that option's text."""

    options = decision_options(decision)
    choice = text.strip()
    if options and choice.isdigit() and 1 <= int(choice) <= len(options):
        return options[int(choice) - 1]
    return text
