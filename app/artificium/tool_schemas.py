"""JSON-schema tool definitions for native tool calling.

Built from what the registry already knows, so the two protocols cannot drift:
argument names and required/optional come from the handlers' signatures,
types from the documented example of each tool, and the description from that
tool's section of the tool catalog.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

# Kept for old contexts only; the prompts never teach them.
LEGACY_TOOLS = frozenset({"compact_context", "load_attachment"})

_SECTION = re.compile(r"^### `(?P<name>[A-Za-z_][A-Za-z0-9_]*)`\s*$", re.MULTILINE)
_INLINE_REQUEST = re.compile(r"`(\{\s*\"tool\"\s*:.*?\})`")
_FENCE = re.compile(r"```[a-z]*\n(?P<body>.*?)```", re.DOTALL)


def _schema_type(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        item = _schema_type(value[0]) if value else {}
        return {"type": "array", "items": item}
    if isinstance(value, dict):
        return {"type": "object"}
    return {}


def _inline_call(match: re.Match[str]) -> str:
    """`{"tool":"x","a":1}` written in prose becomes `x(a=1)`: no text-protocol syntax in native mode."""

    try:
        value = json.loads(match.group(1))
    except ValueError:
        return match.group(0)
    if not isinstance(value, dict) or not isinstance(value.get("tool"), str):
        return match.group(0)
    arguments = ", ".join(f"{key}={json.dumps(item)}" for key, item in value.items() if key != "tool")
    return f"`{value['tool']}({arguments})`"


def catalog_sections(catalog: str) -> dict[str, tuple[str, dict[str, Any]]]:
    """Per tool: its prose (request example removed) and its documented example."""

    sections: dict[str, tuple[str, dict[str, Any]]] = {}
    matches = list(_SECTION.finditer(catalog))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(catalog)
        body = catalog[match.end():end]
        # A "## Group" heading belongs to the next tools, not to this one.
        body = re.split(r"^## ", body, maxsplit=1, flags=re.MULTILINE)[0]
        example: dict[str, Any] = {}

        def drop_request_example(block: re.Match[str]) -> str:
            # Only the flat text-protocol example goes; other code blocks are prose.
            nonlocal example
            try:
                value = json.loads(block.group("body"))
            except ValueError:
                return block.group(0)
            if isinstance(value, dict) and "tool" in value:
                example = example or value
                return ""
            return block.group(0)

        prose = _INLINE_REQUEST.sub(_inline_call, _FENCE.sub(drop_request_example, body))
        sections[match.group("name")] = (re.sub(r"\n{3,}", "\n\n", prose).strip(), example)
    return sections


def native_tool_schemas(registry: type, catalog: str) -> list[dict[str, Any]]:
    """Tool definitions in the chat-completions ``tools`` format."""

    sections = catalog_sections(catalog)
    examples = getattr(registry, "CANONICAL_EXAMPLES", {})
    schemas: list[dict[str, Any]] = []
    for name in examples:
        handler = getattr(registry, name, None)
        if name in LEGACY_TOOLS or not callable(handler):
            continue
        prose, documented = sections.get(name, ("", {}))
        types = {**{k: v for k, v in examples[name].items() if k != "tool"},
                 **{k: v for k, v in documented.items() if k != "tool"}}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter in list(inspect.signature(handler).parameters.values())[1:]:
            if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                continue
            properties[parameter.name] = _schema_type(types.get(parameter.name))
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": prose or name.replace("_", " "),
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        })
    return schemas
