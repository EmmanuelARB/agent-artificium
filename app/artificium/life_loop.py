from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .filesystem import json_dumps, sortable_id


THINK_PATTERN = re.compile(
    r"<think\b[^>]*>(?P<body>.*?)</think\s*>", re.IGNORECASE | re.DOTALL
)
THINK_OPEN_PATTERN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_PATTERN = re.compile(r"</think\s*>", re.IGNORECASE)
TOOL_OPEN_PATTERN = re.compile(r"<(?P<tag>tool_call|tool)\b(?P<attrs>[^>]*)>", re.IGNORECASE)
TOOL_CLOSE_PATTERN = re.compile(r"</(?:tool_call|tool)\s*>", re.IGNORECASE)
BARE_OPEN_PATTERN = re.compile(r"<tool_call\s*>", re.IGNORECASE)
FENCE_OPEN_PATTERN = re.compile(r"```(?:json)?[ \t]*\n?", re.IGNORECASE)
NAME_PATTERN = re.compile(
    r"\bname\s*=\s*(?:\"(?P<double>[^\"]+)\"|'(?P<single>[^']+)'|(?P<bare>[^\s>]+))",
    re.IGNORECASE,
)
# Qwen's native request: <function=NAME><parameter=KEY>VALUE</parameter></function>.
FUNCTION_PATTERN = re.compile(
    r"<function=(?P<name>[^>\s]+)\s*>(?P<body>.*?)</function\s*>", re.IGNORECASE | re.DOTALL
)
PARAMETER_PATTERN = re.compile(
    r"<parameter=(?P<key>[^>\s]+)\s*>\n?(?P<value>.*?)\n?</parameter\s*>", re.IGNORECASE | re.DOTALL
)
# A mangled start of the flat object, such as `<tool":"run_shell"` or `<tool="run_shell"`.
TOOL_KEY_PREFIX_PATTERN = re.compile(r"<\s*[\"']?tool[\"']?\s*[:=]\s*(?=\")", re.IGNORECASE)
# The flat object found a few characters after stray text such as `tool:` or `<function=tool>`.
OBJECT_START_PATTERN = re.compile(r"\{\s*\"(?:tool|name)\"\s*:")
OBJECT_SEARCH_WINDOW = 48
YAML_LINE_PATTERN = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s?(?P<value>.*)")
JSON_LITERAL_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|null")
# Finish reasons meaning generation was cut off rather than ended by the model.
TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens", "MAX_TOKENS", "incomplete"})
_CHUNK_HINT = (
    " If its content was too large to finish, split it into several "
    '`write_file` calls instead of one: the first with mode "create", each '
    'following one with mode "append", none larger than comfortably fits in '
    "one response."
)

# strict=False accepts raw newlines and tabs inside strings: models writing code
# into a JSON string routinely emit them, and their meaning is unambiguous.
_DECODER = json.JSONDecoder(strict=False)
_ESCAPE_PATTERN = re.compile(r"\\(.)", re.DOTALL)
_VALID_ESCAPES = frozenset('"\\/bfnrtu')
_AMBIGUOUS_ESCAPES = frozenset("'`")
_BARE_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\"\s*:")
_MAX_QUOTE_REPAIRS = 200


class _DecodeFailure(ValueError):
    """A request body that could not be read; ``offset`` is relative to it."""

    def __init__(self, message: str, offset: int = 0):
        super().__init__(message)
        self.message = message
        self.offset = offset


def _skip_whitespace(content: str, index: int) -> int:
    while index < len(content) and content[index].isspace():
        index += 1
    return index


class _RepairText:
    """Text being repaired, remembering which characters were inserted so a
    decoded end offset can be mapped back to the original content."""

    def __init__(self, prefix: str, original: str):
        self.text = prefix + original
        self.synthetic = [True] * len(prefix) + [False] * len(original)

    def insert_chars(self, positions: list[int], char: str) -> None:
        if not positions:
            return
        text: list[str] = []
        synthetic: list[bool] = []
        cursor = 0
        for position in sorted(positions):
            text.append(self.text[cursor:position])
            synthetic.extend(self.synthetic[cursor:position])
            text.append(char)
            synthetic.append(True)
            cursor = position
        text.append(self.text[cursor:])
        synthetic.extend(self.synthetic[cursor:])
        self.text = "".join(text)
        self.synthetic = synthetic

    def insert_backslashes(self, positions: list[int]) -> None:
        self.insert_chars(positions, "\\")

    def insert_quote(self, position: int) -> None:
        self.insert_chars([position], '"')

    def original_length(self, end: int) -> int:
        return end - sum(self.synthetic[:end])


def _invalid_escape_positions(text: str) -> list[int] | None:
    """Backslashes that do not start a valid JSON escape, or None when one is ambiguous.

    Shell and regex text such as ``grep 'a\\|b'`` or ``mp\\.dps`` reaches a JSON
    string with a lone backslash; the literal backslash is the only reading.
    Before a quote or a backtick it may instead be a needless escape of that
    character, and in shell the two readings run different commands.
    """

    positions = []
    for match in _ESCAPE_PATTERN.finditer(text):
        if match.group(1) in _VALID_ESCAPES:
            continue
        if match.group(1) in _AMBIGUOUS_ESCAPES:
            return None
        positions.append(match.start())
    return positions


def _ends_request(content: str, index: int) -> bool:
    rest = content[_skip_whitespace(content, index) :]
    return not rest or rest.startswith("```") or TOOL_OPEN_PATTERN.match(rest) is not None or (
        TOOL_CLOSE_PATTERN.match(rest) is not None
    )


def _decode_object(content: str, start: int, *, prefix: str = "") -> tuple[Any, int]:
    """Decode one JSON value at ``start``; return it and its end offset in ``content``.

    Decoding stops where the value ends, so the closing tag, text after it, or
    a following request never become part of this one. Two slips with a single
    reading are repaired: a lone backslash is literal, and a double quote that
    ends a string only to be followed by more text (`print("x")` or `"$f"` in
    code) was meant as part of that string. A quote repair is kept only when the
    repaired object ends exactly where the request does, so a string that
    merely looks like JSON (`{"a": 1}` in code) can never cut a value short.
    """

    repair = _RepairText(prefix, content[start:])
    first_error: json.JSONDecodeError | None = None
    escapes_repaired = False
    quote_repairs = 0
    while True:
        try:
            value, end = _DECODER.raw_decode(repair.text, 0)
            break
        except json.JSONDecodeError as exc:
            first_error = first_error or exc
            if "Invalid \\escape" in exc.msg and not escapes_repaired:
                escapes_repaired = True
                positions = _invalid_escape_positions(repair.text)
                if positions is not None:
                    repair.insert_backslashes(positions)
                    continue
            if exc.msg == "Expecting ',' delimiter" and quote_repairs < _MAX_QUOTE_REPAIRS:
                quote = repair.text.rfind('"', 0, exc.pos)
                # `""` is a stray doubled quote, not a quote inside the text.
                doubled = exc.pos == quote + 1 and repair.text.startswith('"', exc.pos)
                if quote > 0 and not doubled and not repair.text[quote + 1 : exc.pos].strip():
                    quote_repairs += 1
                    repair.insert_backslashes([quote])
                    continue
            if exc.msg == "Expecting property name enclosed in double quotes" and (
                quote_repairs < _MAX_QUOTE_REPAIRS
            ):
                # `...",mode":"create"`: the key already carries its closing
                # quote, so the only plausible repair is the missing opening
                # one — nothing else is legal JSON in a key position.
                if _BARE_KEY_PATTERN.match(repair.text, exc.pos):
                    quote_repairs += 1
                    repair.insert_quote(exc.pos)
                    continue
                # `print("x", y)`: the quote before the comma belongs to the code.
                quote = repair.text.rfind('"', 0, exc.pos)
                between = repair.text[quote + 1 : exc.pos]
                if quote > 0 and between.strip() == ",":
                    quote_repairs += 1
                    repair.insert_backslashes([quote])
                    continue
            raise _DecodeFailure(
                first_error.msg, max(0, first_error.pos - len(prefix))
            ) from None
    content_end = start + repair.original_length(end)
    if quote_repairs and not _ends_request(content, content_end):
        raise _DecodeFailure(first_error.msg, max(0, first_error.pos - len(prefix)))
    if isinstance(value, dict):
        merged = _merge_premature_close(content, start, content_end)
        if merged is not None:
            value, content_end = merged
    return value, content_end


def _merge_premature_close(content: str, start: int, end: int) -> tuple[Any, int] | None:
    """Recover ``{...}key":"value",...}`` where the model closed the object
    one field early, then kept writing fields as though it had not.

    The dangling key already carries its closing quote (``_BARE_KEY_PATTERN``),
    which is the same unambiguous signal ``_decode_object`` uses mid-decode:
    nothing else is legal JSON right after an object's closing brace. Merged
    only when the repaired object's end lands exactly on a request boundary
    (``_ends_request``), so trailing prose that merely resembles a key can
    never be absorbed.
    """

    # ``merged_text`` carries the already-repaired object across iterations
    # (chained premature closes), so a later round never re-parses the
    # original raw brace that a previous round already folded in.
    merged_text = content[start:end]
    total_end = end
    for _ in range(4):
        if not merged_text.endswith("}"):
            return None
        rest_start = _skip_whitespace(content, total_end)
        if not _BARE_KEY_PATTERN.match(content, rest_start):
            return None
        head = merged_text[:-1]
        try:
            value, tail_end = _DECODER.raw_decode(head + ',"' + content[rest_start:], 0)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        new_end = rest_start + (tail_end - len(head) - 2)
        if new_end <= total_end:
            return None
        if _ends_request(content, new_end):
            return value, new_end
        merged_text = (head + ',"' + content[rest_start:])[:tail_end]
        total_end = new_end
    return None


def _literal(value: str) -> Any:
    """A parameter value from a text format: JSON when it plainly is, else text."""

    stripped = value.strip()
    if JSON_LITERAL_PATTERN.fullmatch(stripped) or stripped[:1] in {"[", "{"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return value


def _intent_from_value(tag: str, attrs: str, decoded: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(decoded, dict):
        raise TypeError("tool call must be a JSON object")
    if tag == "tool_call":
        if "tool" in decoded:
            name = str(decoded.get("tool") or "").strip()
            arguments = {str(key): value for key, value in decoded.items() if key != "tool"}
            return name, arguments
        # Revolution 1.4 and earlier used a nested name/arguments object, which
        # is also the native form of several model families. Keep it readable
        # so old working contexts can recover while promptgramming teaches the
        # smaller flat request.
        supplied_arguments = decoded.get("arguments", {})
        if not isinstance(supplied_arguments, dict):
            raise TypeError("tool call `arguments` must be a JSON object")
        return str(decoded.get("name") or "").strip(), supplied_arguments
    # Legacy syntax remains readable so an existing context or a less capable
    # model cannot strand itself while learning the canonical protocol.
    name = ""
    name_match = NAME_PATTERN.search(attrs)
    if name_match:
        name = next(
            value for value in name_match.group("double", "single", "bare") if value is not None
        ).strip()
    return name, decoded


def _body_start(content: str, index: int) -> tuple[int, bool]:
    """Where a request's body starts, past whitespace, repeated bare opening
    tags (they carry no intent of their own), and a Markdown fence; and whether
    a fence was skipped."""

    index = _skip_whitespace(content, index)
    while (repeat := BARE_OPEN_PATTERN.match(content, index)) is not None:
        index = _skip_whitespace(content, repeat.end())
    fence = FENCE_OPEN_PATTERN.match(content, index)
    if fence is None:
        return index, False
    return _skip_whitespace(content, fence.end()), True


def _decoded_span_covering(content: str, position: int) -> int | None:
    """End of a request JSON object that contains ``position``, if any."""

    for match in TOOL_OPEN_PATTERN.finditer(content, 0, position):
        start, _ = _body_start(content, match.end())
        if start >= position:
            continue
        try:
            _, end = _decode_object(content, start)
        except _DecodeFailure:
            continue
        if start <= position < end:
            return end
    return None


def _leading_thought_end(content: str) -> int:
    """End of an unopened thought at the start of a response, or 0.

    Some chat templates open ``<think>`` in the generation prompt, so the reply
    starts inside a thought and only its ``</think>`` appears. Everything before
    that closing tag is thinking, including any tool call drafted there.
    """

    search_from = 0
    while (close := THINK_CLOSE_PATTERN.search(content, search_from)) is not None:
        if THINK_OPEN_PATTERN.search(content, 0, close.start()):
            return 0
        # A `</think>` quoted inside a request's JSON string is data, not a boundary.
        covering = _decoded_span_covering(content, close.start())
        if covering is None:
            return close.end()
        search_from = covering
    return 0


def _unclosed_thought_start(content: str, spans: list[tuple[int, int]], start: int) -> int | None:
    """Offset of a ``<think>`` that is never closed, or None.

    Models sometimes end a turn while still thinking. Such a thought runs to
    the end of the reply; a quoted mention such as "`<think>`" is prose.
    """

    for match in THINK_OPEN_PATTERN.finditer(content, start):
        if any(s <= match.start() < e for s, e in spans):
            continue
        if match.start() > 0 and content[match.start() - 1] == "`":
            continue
        if THINK_CLOSE_PATTERN.search(content, match.end()) is None:
            return match.start()
    return None


def _guess_name(raw: str) -> str:
    match = re.search(r'["\'](?:tool|name)["\']\s*:\s*["\']([^"\']+)', raw)
    if match is None:
        match = re.search(r"<function=([^>\s]+)", raw)
    return match.group(1).strip() if match else ""


def _read_json_objects(
    content: str, start: int, prefix: str = ""
) -> tuple[list[tuple[Any, int, int]], int]:
    """Decode one or more consecutive objects; a tag holding two objects is two
    requests, as the model plainly meant. Returns them and the end offset."""

    decoded, end = _decode_object(content, start, prefix=prefix)
    objects = [(decoded, start, end)]
    position = _skip_whitespace(content, end)
    while content.startswith("{", position):
        try:
            decoded, end = _decode_object(content, position)
        except _DecodeFailure:
            break
        objects.append((decoded, position, end))
        position = _skip_whitespace(content, end)
    return objects, position


def _read_body(
    content: str, body_start: int, limit: int, closing: re.Match[str] | None
) -> tuple[list[tuple[Any, int, int]], int]:
    """Read the requests in one tool tag, whichever form the model used.

    Returns ``(value, start, end)`` triples, each value being a flat request
    object, and the offset where the body ends. Raises ``_DecodeFailure``.
    """

    if content.startswith("{", body_start):
        return _read_json_objects(content, body_start)
    function = FUNCTION_PATTERN.match(content, body_start)
    if function is not None:
        arguments = {
            match.group("key"): _literal(match.group("value"))
            for match in PARAMETER_PATTERN.finditer(function.group("body"))
        }
        value = {"tool": function.group("name"), **arguments}
        return [(value, body_start, function.end())], _skip_whitespace(content, function.end())
    mangled = TOOL_KEY_PREFIX_PATTERN.match(content, body_start)
    if mangled is not None:
        return _read_json_objects(content, mangled.end(), prefix='{"tool":')
    window_end = min(limit, body_start + OBJECT_SEARCH_WINDOW)
    if closing is not None:
        window_end = min(window_end, closing.start())
    found = OBJECT_START_PATTERN.search(content, body_start, window_end)
    if found is not None:
        return _read_json_objects(content, found.start())
    if closing is not None:
        # `tool: read_file` then one `key: value` per line.
        lines = [line for line in content[body_start : closing.start()].splitlines() if line.strip()]
        pairs = [YAML_LINE_PATTERN.fullmatch(line.strip()) for line in lines]
        if pairs and all(pairs) and pairs[0].group("key") == "tool":
            value = {pair.group("key"): _literal(pair.group("value")) for pair in pairs}
            value["tool"] = str(value["tool"]).strip()
            return [(value, body_start, closing.start())], closing.start()
    raise _DecodeFailure("Expecting value", 0)


@dataclass
class ToolIntent:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class LifeLoopOutput:
    raw: str
    thoughts: list[str]
    tools: list[ToolIntent]
    visible: str


def parse_life_loop_output(content: str, *, finish_reason: str | None = None) -> LifeLoopOutput:
    """Split a reply into thoughts, tool requests, and visible text.

    Each request is read by decoding what follows its opening tag, so a closing
    tag is expected but not required: models regularly end their turn right
    after the object, omit the tag between two requests, or repeat the opening
    tag. Qwen's native `<function=...>` form is also read. Only a reply cut off
    by the output limit (``finish_reason``) keeps an unclosed request invalid,
    since its end may be missing. Tags inside a JSON string or a thought are
    never requests.
    """

    truncated = finish_reason in TRUNCATED_FINISH_REASONS
    thoughts: list[str] = []
    tools: list[ToolIntent] = []
    spans: list[tuple[int, int]] = []

    leading_end = _leading_thought_end(content)
    if leading_end:
        spans.append((0, leading_end))
        body = THINK_CLOSE_PATTERN.sub("", content[:leading_end]).strip()
        if body:
            thoughts.append(body)
    for match in THINK_PATTERN.finditer(content, leading_end):
        spans.append(match.span())
        body = match.group("body").strip()
        if body:
            thoughts.append(body)
    think_spans = list(spans)
    open_thought = _unclosed_thought_start(content, spans, leading_end)
    valid_request_starts: list[int] = []

    cursor = leading_end
    while True:
        match = TOOL_OPEN_PATTERN.search(content, cursor)
        if match is None:
            break
        start = match.start()
        inside_thought = next((end for s, end in think_spans if s <= start < end), None)
        if inside_thought is not None:
            cursor = inside_thought
            continue
        if start > 0 and content[start - 1] == "`" and content[match.end() : match.end() + 1] == "`":
            # A mention of the tag in prose, such as "`<tool_call>`", is not a request.
            cursor = match.end()
            continue
        tag = match.group("tag").lower()
        attrs = match.group("attrs")
        body_start, fenced = _body_start(content, match.end())
        close_now = TOOL_CLOSE_PATTERN.match(content, body_start)
        if close_now is not None or body_start >= len(content):
            # An empty request pair, or a dangling opening tag, requests nothing.
            end = close_now.end() if close_now is not None else len(content)
            spans.append((start, end))
            cursor = end
            continue
        next_open = TOOL_OPEN_PATTERN.search(content, body_start)
        limit = next_open.start() if next_open else len(content)
        closing = TOOL_CLOSE_PATTERN.search(content, body_start, limit)
        if closing is None and content[body_start : body_start + 1].isalpha() and not (
            OBJECT_START_PATTERN.search(content, body_start, min(limit, body_start + OBJECT_SEARCH_WINDOW))
        ):
            # "I will emit a <tool_call> next": prose naming the tag, with no
            # request body and no closing tag, is text rather than a request.
            cursor = match.end()
            continue

        try:
            objects, end = _read_body(content, body_start, limit, closing)
        except _DecodeFailure as failure:
            if open_thought is not None and start >= open_thought:
                # A request drafted inside a thought that never closed is
                # thinking, not a malformed call; it joins the thought below.
                cursor = closing.end() if closing else limit
                continue
            # Undecodable: the request ends at its closing tag when one comes
            # before the next request, otherwise at the next request.
            end = closing.end() if closing else limit
            raw = content[body_start : closing.start() if closing else limit].strip()
            error = f"invalid tool-call JSON: {failure.message} at character {failure.offset} of the request"
            if closing is None:
                error += "; the request also has no closing `</tool_call>`"
                if truncated:
                    error += _CHUNK_HINT
            spans.append((start, end))
            tools.append(ToolIntent(id=sortable_id("tool_"), name=_guess_name(raw),
                                    raw_arguments=raw, parse_error=error))
            cursor = end
            continue

        if fenced and content.startswith("```", end):
            end = _skip_whitespace(content, end + 3)
        closing_here = TOOL_CLOSE_PATTERN.match(content, end)
        if closing_here is not None:
            end = closing_here.end()
        for index, (decoded, object_start, object_end) in enumerate(objects):
            parse_error: str | None = None
            name = ""
            arguments: dict[str, Any] = {}
            try:
                name, arguments = _intent_from_value(tag, attrs, decoded)
                if not name:
                    parse_error = "tool call is missing a non-empty name"
            except TypeError as exc:
                parse_error = f"invalid tool-call JSON: {exc}"
            if closing_here is None and truncated and index == len(objects) - 1:
                parse_error = parse_error or (
                    f"the reply was cut off (finish_reason {finish_reason!r}) before this "
                    "request closed with `</tool_call>`; it may be incomplete." + _CHUNK_HINT
                )
            tools.append(
                ToolIntent(
                    id=sortable_id("tool_"),
                    name=name,
                    arguments=arguments if parse_error is None else {},
                    raw_arguments=content[object_start:object_end],
                    parse_error=parse_error,
                )
            )
        spans.append((start, end))
        valid_request_starts.append(start)
        cursor = end

    if open_thought is not None:
        # A well-formed request after an unclosed `<think>` still runs, since
        # the model may simply have forgotten `</think>`; the thought ends there.
        thought_end = min((s for s in valid_request_starts if s > open_thought), default=len(content))
        spans.append((open_thought, thought_end))
        body = THINK_OPEN_PATTERN.sub("", content[open_thought:thought_end], count=1).strip()
        if body:
            thoughts.append(body)

    visible_parts: list[str] = []
    cursor = 0
    merged_spans: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged_spans and start <= merged_spans[-1][1]:
            merged_spans[-1] = (merged_spans[-1][0], max(end, merged_spans[-1][1]))
        else:
            merged_spans.append((start, end))
    for start, end in merged_spans:
        if start > cursor:
            visible_parts.append(content[cursor:start])
        cursor = end
    visible_parts.append(content[cursor:])
    # A stray closing tag left by a request that ended at its JSON is not text.
    visible_parts = [TOOL_CLOSE_PATTERN.sub("", part) for part in visible_parts]
    visible = "\n".join(part.strip() for part in visible_parts if part.strip()).strip()
    return LifeLoopOutput(content, thoughts, tools, visible)


def render_normalized_life_loop_output(
    parsed: LifeLoopOutput, *, include_tools: bool = True
) -> str:
    """Return clean context evidence without preserving arbitrary tool formatting."""

    parts = [f"<think>{thought}</think>" for thought in parsed.thoughts]
    if parsed.visible:
        parts.append(parsed.visible)
    if include_tools:
        for intent in parsed.tools:
            if intent.parse_error or not intent.name:
                continue
            payload = {"tool": intent.name, **intent.arguments}
            parts.append(f"<tool_call>{json_dumps(payload)}</tool_call>")
    if not parts:
        return "[No valid life-loop output was retained from this inference.]"
    return "\n".join(parts)


def render_observation(
    *,
    kind: str,
    content: Any,
    tool_name: str | None = None,
    tool_id: str | None = None,
) -> str:
    attributes = [f'kind="{kind}"']
    if tool_name:
        attributes.append(f'tool="{tool_name}"')
    if tool_id:
        attributes.append(f'id="{tool_id}"')
    body = content if isinstance(content, str) else json_dumps(content, pretty=True)
    return (
        f"<life-loop-observation {' '.join(attributes)}>\n"
        f"{body.rstrip()}\n"
        "</life-loop-observation>"
    )


def concise_tool_catalog(specs: list[dict[str, Any]]) -> str:
    lines = []
    for spec in specs:
        signature = ", ".join(spec.get("arguments", []))
        lines.append(
            f"- `{spec['name']}({signature})` — {str(spec['description']).strip()}"
        )
    return "\n".join(lines)
