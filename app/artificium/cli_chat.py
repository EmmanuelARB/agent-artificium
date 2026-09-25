"""The interactive terminal chat REPL and its small display/input helpers.

`_pid_state` and `_print_detach_status` are resolved through `artificium.cli`
at call time rather than imported directly, since tests patch names such as
"artificium.cli._pid_state"; only a module-attribute lookup at call time sees
that patch.
"""

from __future__ import annotations

import datetime as dt
import shutil
import sys
import threading
import unicodedata
from typing import Any

from . import cli as _cli
from .filesystem import Paths, read_json, safe_identifier, sortable_id
from .interactions import ArtificiumClient


def _local_time(value: Any) -> str:
    raw = str(value or "")
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%H:%M:%S")
    except ValueError:
        return raw


def _print_event(event: dict[str, Any], *, local_entity: str | None = None) -> None:
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
    timestamp = _local_time(event.get("created_at"))
    content = str(event.get("content") or "")
    print(f"\n[{timestamp}] {label}")
    print(content)
    attachments = event.get("attachments") or []
    if attachments:
        print("Attachments: " + ", ".join(str(item) for item in attachments))


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


def _choose_interaction(
    client: ArtificiumClient,
    entity: str,
    supplied: str | None,
    name: str | None,
) -> str:
    if supplied:
        client.interactions.ensure(supplied, name=name, participants=[entity])
        return supplied
    existing = client.interactions_for(entity)
    if existing:
        latest = sorted(existing, key=lambda item: str(item.get("updated_at") or ""))[-1]
        answer = input(
            f"Resume `{latest['name']}` ({latest['id']})? [Y/n]: "
        ).strip().lower()
        if answer in {"", "y", "yes"}:
            return str(latest["id"])
    interaction_name = name or input("Interaction name [chat]: ").strip() or "chat"
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


def _chat_repl(paths: Paths, entity: str | None, interaction: str | None, name: str | None) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("chat requires an interactive terminal")
    client = ArtificiumClient(paths.install)
    entity = safe_identifier(
        entity or input("Your entity name/ID [user_1]: ").strip() or "user_1",
        label="entity id",
    )
    interaction_id = _choose_interaction(client, entity, interaction, name)
    seen = {str(item.get("id")) for item in client.events(interaction_id)}
    print("\n╭─ Artificium terminal interaction")
    print(f"│ Thread: {interaction_id}")
    print(f"│ You:    {entity}")
    print("│ Ctrl-C or /quit closes ONLY this chat client.")
    print("╰─ TO STOP ARTIFICIUM: python3 artificium.py stop\n")
    for event in client.events(interaction_id):
        _print_event(event, local_entity=entity)

    stop = threading.Event()
    input_active = threading.Event()
    output_lock = threading.Lock()
    input_prompt = _chat_input_prompt(entity)
    try:
        import readline  # noqa: F401
    except ImportError:
        readline = None  # type: ignore[assignment]

    def poll() -> None:
        while not stop.wait(0.5):
            for event in client.events(interaction_id):
                event_id = str(event.get("id"))
                if event_id in seen:
                    continue
                seen.add(event_id)
                with output_lock:
                    if input_active.is_set():
                        _clear_chat_input(input_prompt, readline)
                    else:
                        print("\r\033[2K", end="", flush=True)
                    _print_event(event, local_entity=entity)
                    if input_active.is_set():
                        _restore_chat_input(input_prompt, readline)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        while True:
            with output_lock:
                input_active.set()
            try:
                # Give the prompt to Readline so its cursor and wrapping math
                # includes the visible label. Asynchronous output explicitly
                # clears and restores every physical row occupied by the draft.
                content = input(input_prompt).strip()
            finally:
                input_active.clear()
            if not content:
                continue
            if content == "/quit":
                break
            if content == "/history":
                for event in client.events(interaction_id):
                    _print_event(event, local_entity=entity)
                continue
            if content == "/help":
                print(
                    "Commands:\n"
                    "  /history                 show the complete thread\n"
                    "  /attach PATH MESSAGE     send one attachment\n"
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
                continue
            attachments: list[str] = []
            if content.startswith("/attach "):
                parts = content.split(maxsplit=2)
                if len(parts) < 3:
                    print("Usage: /attach PATH MESSAGE")
                    continue
                attachments = [parts[1]]
                content = parts[2]
            event, _ = client.send(
                interaction_id,
                sender=entity,
                content=content,
                attachments=attachments,
            )
            seen.add(str(event["id"]))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop.set()
        thread.join(timeout=1)
        _cli._print_detach_status(paths, client="chat client")

