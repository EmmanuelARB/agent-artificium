"""Native tool calling for chat-completions adapters.

Working memory keeps every turn as text (assistant turns carry their tool
requests as ``<tool_call>`` blocks, results are runtime messages), so offload,
summaries and recovery work the same in both tool modes. Native mode changes
only the wire format: tool definitions travel in ``tools``, and history that is
completely paired is replayed as structured assistant ``tool_calls`` followed
by ``tool`` messages. Anything that cannot be paired exactly stays text.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .life_loop import parse_life_loop_output

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def tools_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tool definitions attached to the request's system message, if any.

    Only life-loop requests carry them, so reviews, summaries and checks that
    reuse the same engine never declare tools.
    """

    for message in messages:
        if message.get("role") != "system":
            continue
        meta = message.get("_artificium")
        tools = meta.get("tools") if isinstance(meta, dict) else None
        if isinstance(tools, list) and tools:
            return tools
    return []


def call_id(seed: str) -> str:
    """A stable nine-character alphanumeric id: the strictest format any provider asks for."""

    digest = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest(), "big")
    chars: list[str] = []
    for _ in range(9):
        digest, index = divmod(digest, len(_ALPHABET))
        chars.append(_ALPHABET[index])
    return "".join(chars)


def _meta(message: dict[str, Any]) -> dict[str, Any]:
    meta = message.get("_artificium")
    return meta if isinstance(meta, dict) else {}


def _paired(messages: list[dict[str, Any]], index: int) -> tuple[list[dict[str, Any]], int] | None:
    """Native messages for the assistant turn at ``index`` and the next index, or None."""

    message = messages[index]
    meta = _meta(message)
    names = meta.get("tool_names")
    if (message.get("role") != "assistant" or meta.get("kind") != "life_loop_output"
            or not names or meta.get("tool_request_valid") is False
            or not isinstance(message.get("content"), str)):
        return None
    parsed = parse_life_loop_output(message["content"])
    intents = [tool for tool in parsed.tools if tool.name and not tool.parse_error]
    if [tool.name for tool in intents] != list(names) or len(intents) != len(parsed.tools):
        return None

    results: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    cursor = index + 1
    while cursor < len(messages) and len(results) < len(names):
        following = messages[cursor]
        following_meta = _meta(following)
        if following.get("role") == "assistant" or following_meta.get("kind") == "life_loop_output":
            return None
        if following_meta.get("kind") == "tool_result":
            if following_meta.get("tool") != names[len(results)] or not isinstance(following.get("content"), str):
                return None
            results.append(following)
        else:
            # A notice recorded between the turn and its results (for example a
            # released-image notice) is replayed after them: APIs want the tool
            # messages to follow the calls directly.
            deferred.append(following)
        cursor += 1
    if len(results) != len(names):
        return None

    text = [f"<think>{thought}</think>" for thought in parsed.thoughts]
    if parsed.visible:
        text.append(parsed.visible)
    calls = []
    ids = []
    for position, (intent, result) in enumerate(zip(intents, results)):
        identifier = call_id(str(_meta(result).get("tool_id") or f"{index}:{position}"))
        ids.append(identifier)
        calls.append({
            "id": identifier,
            "type": "function",
            "function": {"name": intent.name, "arguments": json.dumps(intent.arguments, ensure_ascii=False)},
        })
    native: list[dict[str, Any]] = [
        {"role": "assistant", "content": "\n".join(text), "tool_calls": calls},
        *({"role": "tool", "tool_call_id": identifier, "content": result["content"]}
          for identifier, intent, result in zip(ids, intents, results)),
        *deferred,
    ]
    return native, cursor


def native_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """History with completely paired tool turns replayed as structured calls."""

    if not tools_from_messages(messages):
        return messages
    converted: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        paired = _paired(messages, index)
        if paired is None:
            converted.append(messages[index])
            index += 1
        else:
            native, index = paired
            converted.extend(native)
    return converted
