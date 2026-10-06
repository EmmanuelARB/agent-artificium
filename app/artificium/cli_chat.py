"""The interactive terminal chat REPL and its small display/input helpers.

`_pid_state` and `_print_detach_status` are resolved through `artificium.cli`
at call time rather than imported directly, since tests patch names such as
"artificium.cli._pid_state"; only a module-attribute lookup at call time sees
that patch.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unicodedata
from typing import Any, Callable

from . import cli as _cli
from .decisions import answer_text, decision_options, open_decisions
from .display import colors_enabled
from .filesystem import Paths, atomic_write_json, read_json, safe_identifier, sortable_id
from .interactions import ArtificiumClient

DEFAULT_ENTITY = "user_1"
CONTINUATION_PROMPT = "... "
PASTE_PROMPT = "paste> "


def _local_time(value: Any) -> str:
    # A conversation spans days and its history is replayed on reconnect, so
    # a time without its date is ambiguous.
    raw = str(value or "")
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw


_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"
PANEL_CONTENT_LINES = 10


def _print_event(
    event: dict[str, Any], *, local_entity: str | None = None, color: bool = False
) -> None:
    sender = str(event.get("sender") or "unknown")
    direction = str(event.get("direction") or "")
    if direction == "outbound":
        # Display the durable event identity rather than inventing a UI name.
        # A Self may use another name, and clients may override the sender.
        label = sender
    elif local_entity and sender == local_entity:
        label = "You"
    else:
        label = sender
    decision = direction == "outbound" and event.get("kind") == "decision"
    timestamp = _local_time(event.get("created_at"))
    content = str(event.get("content") or "")
    print(f"\n[{timestamp}] {label}" + (" · decision" if decision else ""))
    # Progress reports from the agent are dimmed so decisions and replies
    # stand out; the text itself is never shortened.
    update = direction == "outbound" and not decision
    print(f"{_DIM}{content}{_RESET}" if color and update else content)
    for number, option in enumerate(decision_options(event), 1):
        print(f"  {number}) {option}")
    attachments = event.get("attachments") or []
    if attachments:
        print("Attachments: " + ", ".join(str(item) for item in attachments))


def _decision_panel(
    decisions: list[dict[str, Any]], *, columns: int = 80, color: bool = False
) -> list[str]:
    """The block pinned above the prompt while the agent waits for an answer."""

    if not decisions:
        return []
    bold, dim, reset = (_BOLD, _DIM, _RESET) if color else ("", "", "")
    count = len(decisions)
    title = f"{count} decision{'s' if count != 1 else ''} waiting for you"
    lines = [f"{bold}━━ {title} " + "━" * max(0, columns - len(title) - 4) + reset]
    for number, decision in enumerate(decisions, 1):
        text = str(decision.get("content") or "").strip().splitlines() or [""]
        shown = text[:PANEL_CONTENT_LINES]
        stamp = _local_time(decision.get("created_at"))
        lines.append(f"{bold}[{number}]{reset} {dim}{stamp}{reset}  {shown[0]}")
        lines.extend(f"    {line}" for line in shown[1:])
        if len(text) > len(shown):
            lines.append(f"    {dim}… {len(text) - len(shown)} more lines "
                         f"(see /history){reset}")
        for index, option in enumerate(decision_options(decision), 1):
            lines.append(f"    {index}) {option}")
    hint = "answer with /reply N TEXT" + (
        "" if count > 1 else ", or just type your answer")
    lines.append(f"{dim}{hint}{reset}")
    return lines


def _panel_rows(lines: list[str], columns: int | None = None) -> int:
    """Physical terminal rows a block of printed lines occupies."""

    terminal_columns = max(
        1, int(columns or shutil.get_terminal_size(fallback=(80, 24)).columns))
    total = 0
    for line in lines:
        visible = "".join(_ANSI.split(line)) if "\x1b" in line else line
        width = _chat_display_width(visible)
        total += max(1, (width + terminal_columns - 1) // terminal_columns)
    return total


def _erase_rows_above(rows: int) -> None:
    """Remove the ``rows`` rows just above the cursor, and everything below."""

    if rows > 0:
        sys.stdout.write(f"\033[{rows}A")
    sys.stdout.write("\r\033[J")
    sys.stdout.flush()


def _route_message(
    content: str, decisions: list[dict[str, Any]]
) -> tuple[str, str | None, str | None]:
    """Decide which decision, if any, a typed message answers.

    Returns ``(content, in_reply_to, note)``. ``/reply N TEXT`` is explicit;
    a plain message answers the one open decision when there is exactly one,
    and is otherwise left unattached rather than guessed.
    """

    if content == "/reply" or content.startswith("/reply "):
        parts = content.split(maxsplit=2)
        if (len(parts) < 3 or not parts[1].isdigit()
                or not 1 <= int(parts[1]) <= len(decisions)):
            return content, None, (
                "Usage: /reply N TEXT, where N is a decision number shown above"
                if decisions else "No decision is waiting for an answer.")
        decision = decisions[int(parts[1]) - 1]
        text = answer_text(decision, parts[2])
        return text, str(decision["id"]), f"answering decision {parts[1]}"
    if len(decisions) == 1:
        text = answer_text(decisions[0], content)
        return text, str(decisions[0]["id"]), "answering the open decision"
    if len(decisions) > 1:
        return content, None, (
            f"{len(decisions)} decisions are open and this message answers none; "
            "use /reply N TEXT")
    return content, None, None


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _chat_input_prompt(entity: str) -> str:
    """Return the one canonical terminal-chat prompt."""

    return f"{entity}> "


def _chat_line_buffer(readline_module: Any | None) -> str:
    if readline_module is not None:
        try:
            return str(readline_module.get_line_buffer())
        except (AttributeError, RuntimeError):
            pass
    return ""


def _chat_display_width(value: str) -> int:
    """Approximate the terminal columns occupied by one input line."""

    columns = 0
    for character in value:
        if character == "\t":
            columns += 8 - (columns % 8)
        elif unicodedata.combining(character):
            continue
        else:
            columns += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return columns


def _chat_input_rows(prompt: str, buffer: str, columns: int | None = None) -> int:
    """Return the physical terminal rows occupied by a wrapped chat draft."""

    terminal_columns = max(
        1,
        int(columns or shutil.get_terminal_size(fallback=(80, 24)).columns),
    )
    width = _chat_display_width(prompt + buffer)
    return max(1, (width + terminal_columns - 1) // terminal_columns)


def _clear_chat_input(
    prompt: str,
    readline_module: Any | None,
    *,
    columns: int | None = None,
) -> None:
    """Clear every physical row occupied by the active wrapped input draft."""

    buffer = _chat_line_buffer(readline_module)
    rows = _chat_input_rows(prompt, buffer, columns)
    sys.stdout.write("\r\033[2K")
    for _ in range(rows - 1):
        sys.stdout.write("\033[1A\r\033[2K")
    sys.stdout.flush()


def _restore_chat_input(prompt: str, readline_module: Any | None) -> None:
    """Restore an active input line after asynchronous interaction output.

    GNU readline's ``redisplay`` is not reliable when called by the polling
    thread.  Repaint the prompt and its current buffer explicitly so the next
    user message always has a visible, correctly labelled input line.
    """

    buffer = _chat_line_buffer(readline_module)
    sys.stdout.write(f"{prompt}{buffer}")
    sys.stdout.flush()


def _chat_entity(paths: Paths, supplied: str | None) -> str:
    """Return who is chatting without asking: the flag, else the last entity
    used on this install, else the default. An explicit choice is remembered."""

    store = paths.runtime / "chat-client.json"
    if supplied:
        entity = safe_identifier(supplied, label="entity id")
        atomic_write_json(store, {"entity": entity})
        return entity
    saved = read_json(store, {})
    try:
        return safe_identifier(str(saved.get("entity") or ""), label="entity id")
    except (AttributeError, ValueError):
        return DEFAULT_ENTITY


def _choose_interaction(
    client: ArtificiumClient,
    entity: str,
    supplied: str | None,
    name: str | None,
    new: bool = False,
) -> str:
    """Pick the thread to open without prompting.

    Opens the given thread, else the most recent one of this entity; a new
    thread is made only when asked for (``new`` or a ``name``) or when the
    entity has none yet.
    """

    if supplied:
        client.interactions.ensure(supplied, name=name, participants=[entity])
        return supplied
    existing = client.interactions_for(entity)
    if existing and not new and not name:
        latest = sorted(existing, key=lambda item: str(item.get("updated_at") or ""))[-1]
        return str(latest["id"])
    interaction_name = name or "chat"
    base = "".join(
        char.lower() if char.isalnum() else "-" for char in interaction_name
    ).strip("-") or "chat"
    interaction_id = safe_identifier(
        f"{base[:60]}-{sortable_id()[-12:]}", label="interaction id"
    )
    client.interactions.ensure(
        interaction_id, name=interaction_name, participants=[entity]
    )
    return interaction_id


def _edit_message() -> str:
    """Compose a message in ``$VISUAL``/``$EDITOR`` (else nano or vi)."""

    command = shlex.split(os.environ.get("VISUAL") or os.environ.get("EDITOR") or "")
    if not command:
        found = next((tool for tool in ("nano", "vi") if shutil.which(tool)), None)
        if found is None:
            print("No editor found; set $EDITOR, or use /paste or a trailing backslash.")
            return ""
        command = [found]
    with tempfile.TemporaryDirectory() as directory:
        draft = os.path.join(directory, "message.txt")
        with open(draft, "w", encoding="utf-8"):
            pass
        try:
            subprocess.run([*command, draft], check=True)
            with open(draft, encoding="utf-8", errors="replace") as handle:
                return handle.read().strip()
        except (OSError, subprocess.SubprocessError) as error:
            print(f"Editor failed ({error}); nothing was sent.")
            return ""


def _read_chat_message(
    prompt: str,
    set_prompt: Callable[[str], None],
    *,
    read: Callable[[str], str] = input,
    edit: Callable[[], str] = _edit_message,
) -> str:
    """Read one message, which may span several lines.

    A line ending in a backslash continues on the next one; ``/paste`` takes
    lines until one holds only ``.``; ``/edit`` opens an editor. Ctrl-C while
    a multi-line draft is open discards the draft instead of closing the
    chat. ``set_prompt`` tells the asynchronous printer which prompt to
    repaint after the agent's output arrives.
    """

    lines: list[str] = []
    current = prompt
    try:
        while True:
            set_prompt(current)
            line = read(current)
            if not lines and line.strip() == "/edit":
                return edit()
            if not lines and line.strip() == "/paste":
                return _read_pasted(read, set_prompt)
            if line.endswith("\\"):
                lines.append(line[:-1])
                current = CONTINUATION_PROMPT
                continue
            lines.append(line)
            return "\n".join(lines).strip()
    except KeyboardInterrupt:
        if not lines:
            raise
        print("\n(draft discarded)")
        return ""
    finally:
        set_prompt(prompt)


def _read_pasted(read: Callable[[str], str], set_prompt: Callable[[str], None]) -> str:
    print("Paste your message, then finish with a line holding only `.` (or Ctrl-D).")
    set_prompt(PASTE_PROMPT)
    lines: list[str] = []
    while True:
        try:
            line = read(PASTE_PROMPT)
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _chat_repl(
    paths: Paths,
    entity: str | None,
    interaction: str | None,
    name: str | None,
    new: bool = False,
) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("chat requires an interactive terminal")
    client = ArtificiumClient(paths.install)
    entity = _chat_entity(paths, entity)
    interaction_id = _choose_interaction(client, entity, interaction, name, new)
    color = colors_enabled()
    seen = {str(item.get("id")) for item in client.events(interaction_id)}
    print("\n╭─ Artificium terminal interaction")
    print(f"│ Thread: {interaction_id}")
    print(f"│ You:    {entity}")
    print("│ /help lists commands; Ctrl-C or /quit closes ONLY this chat client.")
    print("╰─ TO STOP ARTIFICIUM: python3 artificium.py stop\n")
    history = client.events(interaction_id)
    for event in history:
        _print_event(event, local_entity=entity, color=color)

    stop = threading.Event()
    input_active = threading.Event()
    output_lock = threading.Lock()
    input_prompt = _chat_input_prompt(entity)
    try:
        import readline  # noqa: F401
    except ImportError:
        readline = None  # type: ignore[assignment]
    # The prompt currently on screen, which differs from `input_prompt`
    # while a multi-line draft is open; and the decision panel above it.
    shown = {"prompt": input_prompt, "panel_rows": 0, "panel_ids": [], "announced": set()}

    def draw_panel(decisions: list[dict[str, Any]]) -> None:
        """Print the pinned panel just above where the prompt will appear."""

        columns = shutil.get_terminal_size(fallback=(80, 24)).columns
        lines = _decision_panel(decisions, columns=columns, color=color)
        for line in lines:
            print(line)
        shown["panel_rows"] = _panel_rows(lines, columns)
        shown["panel_ids"] = [str(item["id"]) for item in decisions]
        fresh = set(shown["panel_ids"]) - shown["announced"]
        if fresh and color:
            sys.stdout.write("\a")
        shown["announced"] |= set(shown["panel_ids"])

    def poll() -> None:
        while not stop.wait(0.5):
            events = client.events(interaction_id)
            fresh = [item for item in events if str(item.get("id")) not in seen]
            decisions = open_decisions(events)
            ids = [str(item["id"]) for item in decisions]
            # Only the main prompt sits directly under the panel; while a
            # multi-line draft is open its earlier lines are in between, so
            # the panel is left alone until the next prompt.
            at_main = input_active.is_set() and shown["prompt"] == input_prompt
            if not fresh and (ids == shown["panel_ids"] or not at_main):
                continue
            with output_lock:
                at_main = input_active.is_set() and shown["prompt"] == input_prompt
                if at_main:
                    _clear_chat_input(shown["prompt"], readline)
                    _erase_rows_above(shown["panel_rows"])
                elif input_active.is_set():
                    _clear_chat_input(shown["prompt"], readline)
                else:
                    print("\r\033[2K", end="", flush=True)
                for event in fresh:
                    seen.add(str(event.get("id")))
                    # A new decision is shown by the panel, not in the log.
                    if (event.get("kind") == "decision"
                            and event.get("direction") == "outbound"):
                        continue
                    _print_event(event, local_entity=entity, color=color)
                if at_main:
                    draw_panel(decisions)
                if input_active.is_set():
                    _restore_chat_input(shown["prompt"], readline)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        while True:
            with output_lock:
                draw_panel(open_decisions(client.events(interaction_id)))
                input_active.set()
            try:
                # Give the prompt to Readline so its cursor and wrapping math
                # includes the visible label. Asynchronous output explicitly
                # clears and restores every physical row occupied by the draft.
                content = _read_chat_message(
                    input_prompt, lambda value: shown.update(prompt=value)
                )
            finally:
                input_active.clear()
                # What was on screen is scrollback now; never erase it.
                shown["panel_rows"] = 0
            if not content:
                continue
            if content == "/quit":
                break
            if content == "/history":
                for event in client.events(interaction_id):
                    _print_event(event, local_entity=entity, color=color)
                continue
            if content == "/help":
                print(
                    "Commands:\n"
                    "  /history                 show the complete thread\n"
                    "  /reply N TEXT            answer decision N (a bare option\n"
                    "                           number sends that option)\n"
                    "  /attach PATH MESSAGE     send one attachment\n"
                    "  /edit                    write a long message in $EDITOR\n"
                    "  /paste                   paste lines; finish with a line of `.`\n"
                    "  (a line ending in \\ continues on the next line)\n"
                    "  /status                  show runtime status path\n"
                    "  /quit                    close this client only"
                )
                continue
            if content == "/status":
                pid, alive = _cli._pid_state(paths)
                print(
                    f"Life-loop: {'running' if alive else 'stopped'}"
                    + (f" (PID {pid})" if alive and pid else "")
                    + f"\nTrace: {paths.life_loop_log}"
                )
                runtime = read_json(paths.runtime_state, {})
                if alive and runtime.get("status") == "blocked":
                    print("Model requests paused: " + str(runtime.get("error", "")))
                    print("Correct the problem, then run: python3 artificium.py restart")
                waiting = len(open_decisions(client.events(interaction_id)))
                if waiting:
                    print(f"{waiting} decision{'s' if waiting != 1 else ''} waiting for you.")
                continue
            attachments: list[str] = []
            if content.startswith("/attach "):
                parts = content.split(maxsplit=2)
                if len(parts) < 3:
                    print("Usage: /attach PATH MESSAGE")
                    continue
                attachments = [parts[1]]
                content = parts[2]
            content, reply_to, note = _route_message(
                content, open_decisions(client.events(interaction_id))
            )
            if note and reply_to is None:
                print(note)
                if content.startswith("/reply"):
                    continue
            elif note:
                print(f"({note})")
            event, _ = client.send(
                interaction_id,
                sender=entity,
                content=content,
                attachments=attachments,
                in_reply_to=reply_to,
            )
            seen.add(str(event["id"]))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop.set()
        thread.join(timeout=1)
        _cli._print_detach_status(paths, client="chat client")
