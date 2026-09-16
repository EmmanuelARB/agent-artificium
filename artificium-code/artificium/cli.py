from __future__ import annotations

import argparse
import datetime as dt
import dataclasses
import getpass
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

from . import VERSION
from .config import (
    PROVIDERS,
    ConfigStore,
    SecretsStore,
    configured_paths,
    load_api_key_file,
    requires_api_key,
)
from .engine import EngineError, make_engine
from .connection import verify_connection
from .filesystem import Paths, json_dumps, read_json, safe_identifier, sortable_id
from .interactions import ArtificiumClient
from .prompts import PromptPack
from .records import Console, Records
from .runtime import Artificium
from .operator import process_state, status_snapshot, format_status
from .setup import (
    SetupOptions,
    SetupWizard,
    custom_contract_requires_api_key,
    detected_context_window,
    discover_provider_models,
    load_custom_contract,
    reasoning_label,
)


def _generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reasoning", choices=["auto", "off", "on", "minimal", "low", "medium", "high", "xhigh", "max"],
                        help="Reasoning control; auto clears overrides, off disables it when supported")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "on", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="Provider/model reasoning effort; omitted preserves the provider default",
    )
    parser.add_argument("--reasoning-budget-tokens", type=int)
    parser.add_argument("--reasoning-mode", choices=["standard", "pro"])
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--frequency-penalty", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--stop-sequence",
        action="append",
        dest="stop_sequences",
        help="Repeat to configure multiple stop strings",
    )


def _harness_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("harness settings")
    group.add_argument("--vision", choices=["auto", "yes", "no"], help="Image preference; checked against model capability")
    group.add_argument("--mandatory-offload", choices=["on", "off"], help="Require memory offloading at the threshold (default: off)")
    group.add_argument("--offload-threshold", type=float, help="Working-memory percentage, 1–95 (default: 80)")
    group.add_argument("--working-memory-tokens", metavar="TOKENS|same", help="Offloading target; same tracks the model context (default)")
    group.add_argument("--auto-repair", "--emergency-offload", dest="auto_repair", choices=["on", "off"], help="Try earlier context after input failures; up to 3 attempts (default: off)")


def _connection_arguments(parser: argparse.ArgumentParser, *, runtime_settings: bool = True) -> None:
    parser = parser.add_argument_group("model/API settings")
    parser.add_argument("--provider", choices=[*PROVIDERS, "custom"])
    parser.add_argument("--model", help="Model ID; a single served model is selected automatically")
    secret = parser.add_mutually_exclusive_group()
    secret.add_argument("--api-key")
    secret.add_argument("--api-key-file")
    parser.add_argument("--api-url", "--url", help="Server root, /v1 base, or complete endpoint")
    parser.add_argument("--endpoint")
    parser.add_argument("--custom-contract", dest="custom_contract_file", help="Advanced JSON contract file, or none")
    parser.add_argument("--adapter", choices=["openai_compatible", "custom_json", *dict.fromkeys(p.adapter for p in PROVIDERS.values())])
    if not runtime_settings:
        return
    parser.add_argument("--openrouter-provider", help="Inference provider slug, or automatic")
    parser.add_argument("--context-window", type=int, help="Actual serving capacity in tokens; discovered when available")
    parser.add_argument("--request-timeout", metavar="SECONDS|off", help="Seconds to wait for inference, or off to wait indefinitely (default: off)")
    _generation_arguments(parser)


def _setup_arguments(parser: argparse.ArgumentParser) -> None:
    _harness_arguments(parser)
    _connection_arguments(parser)
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--self", dest="self_directive", help="Optional initial mutable Self")
    identity.add_argument("--self-file")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Use supplied values and detected defaults without prompts; still verify inference")


def _configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("scope", nargs="?", choices=["harness", "model"], help="Edit one settings group; omitted accepts legacy combined flags")
    _harness_arguments(parser)
    _connection_arguments(parser)
    parser.add_argument("--reset-generation-settings", action="store_true", help="Clear explicit generation controls before applying flags")
    parser.add_argument("--yes", action="store_true", help="Use supplied values without prompts; still verify model changes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="artificium",
        description="Artificium-revolution: continual learning and Infinite Attention.",
    )
    parser.add_argument(
        "--root",
        help="Root containing artificium-code, mind, and logs",
    )
    parser.add_argument("--version", action="version", version=f"Artificium {VERSION}")
    commands = parser.add_subparsers(dest="command")

    for name in ("setup", "init"):
        setup = commands.add_parser(
            name,
            help="Configure the model and prepare the shipped Artificium mind",
        )
        _setup_arguments(setup)
        setup.add_argument(
            "--no-launch",
            action="store_true",
            help="Configure only; do not open the chat/life-loop menu",
        )

    run = commands.add_parser("run", help="Run the life-loop daemon")
    run.add_argument("--once", action="store_true", help="Run one bounded turn using the configured round/time budget; continuous mode has no turn budget")
    run.add_argument("--verbose", action="store_true")
    run.add_argument("--quiet", action="store_true")

    commands.add_parser("start", help="Start the life-loop in the background")
    commands.add_parser("restart", help="Stop and start the life-loop to apply configuration")
    upgrade = commands.add_parser("upgrade", help="Upgrade code and prompts while preserving this instance's mind and settings")
    upgrade.add_argument("--check", action="store_true", help="Preview available code changes without applying them")
    upgrade.add_argument("--repo", default="https://github.com/officialgr/agent-artificium.git", help="Git repository to upgrade from")
    upgrade.add_argument("--ref", default="main", help="Branch or tag to install (default: main)")
    commands.add_parser(
        "stop",
        help="Stop the life-loop, escalating to a forced stop if needed",
    )
    watch = commands.add_parser(
        "watch", help="Attach to the life-loop trace without restarting Artificium"
    )
    watch.add_argument("--tail", type=int, default=30, help="Recent records to show first")

    chat = commands.add_parser("chat", help="Open a terminal interaction client")
    chat.add_argument("--entity")
    chat.add_argument("--interaction")
    chat.add_argument("--name")

    send = commands.add_parser("send", help="Write one inbound interaction event")
    send.add_argument("content")
    send.add_argument("--interaction", required=True)
    send.add_argument("--sender", required=True)
    send.add_argument("--name")
    send.add_argument("--attachment", action="append", default=[])
    send.add_argument("--in-reply-to")
    send.add_argument("--recipient", default="artificium")
    send.add_argument("--kind", default="message")

    show = commands.add_parser("show", help="Show one interaction")
    show.add_argument("interaction_id")

    notify = commands.add_parser("notify", help="Queue a generic temporal notification")
    notify.add_argument("summary")
    notify.add_argument("--type", default="external_event")
    notify.add_argument("--source", default="owner_cli")
    notify.add_argument("--path")

    attention = commands.add_parser(
        "attention", aliases=["stream"], help="Request Infinite Attention over a large source"
    )
    attention.add_argument("source")
    attention.add_argument("objective")
    attention.add_argument(
        "--granularity", choices=["auto", "coarse", "fine"], default="auto"
    )
    attention.add_argument("--output")

    commands.add_parser("status", help="Read-only runtime summary").add_argument("--json", action="store_true", help="Machine-readable detailed state")
    models = commands.add_parser("models", help="List served models without inference")
    _connection_arguments(models, runtime_settings=False)
    commands.add_parser("config", help="Show saved configuration groups").add_argument("section", nargs="?", choices=["harness", "model"])
    configure = commands.add_parser(
        "configure",
        aliases=["reconfigure"],
        help="Change engine/runtime settings without touching Self or memory",
    )
    _configure_arguments(configure)
    doctor = commands.add_parser("doctor", help="Inspect the connection; --live checks the full harness request")
    doctor.add_argument("--live", action="store_true", help="Send the full harness prompt and check image support without running tools")
    check = commands.add_parser("check", help="Test this instance's full model connection without changing it")
    check.set_defaults(live=True)
    connect = commands.add_parser("connect", help="Connect or reconnect a model while preserving harness settings and memory")
    _connection_arguments(connect)
    connect.add_argument("--reset-generation-settings", action="store_true")
    connect.add_argument("--yes", action="store_true")
    connect.set_defaults(scope="model")
    key = commands.add_parser("key", help="Replace the saved API key")
    key.add_argument("--api-key")
    key.add_argument("--api-key-file")
    logs = commands.add_parser("logs", help="Show recent structured life-loop events")
    logs.add_argument("--limit", type=int, default=50)
    logs.add_argument("--feature", help="Show one feature log, e.g. infinite-attention")
    logs.add_argument("--summary", action="store_true", help="Show feature usage counters")
    logs.add_argument("--lifetime", action="store_true", help="Show operational lifetime events")
    return parser


def _setup_options(args: argparse.Namespace) -> SetupOptions:
    return SetupOptions(
        scope=getattr(args, "scope", None) or "all",
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        api_key=getattr(args, "api_key", None),
        api_key_file=getattr(args, "api_key_file", None),
        api_url=getattr(args, "api_url", None),
        endpoint=getattr(args, "endpoint", None),
        adapter=getattr(args, "adapter", None),
        custom_contract_file=getattr(args, "custom_contract_file", None),
        openrouter_provider=getattr(args, "openrouter_provider", None),
        self_directive=getattr(args, "self_directive", None),
        self_file=getattr(args, "self_file", None),
        context_window_tokens=getattr(args, "context_window", None),
        request_timeout_seconds=getattr(args, "request_timeout", None),
        reasoning=getattr(args, "reasoning", None),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        reasoning_budget_tokens=getattr(args, "reasoning_budget_tokens", None),
        reasoning_mode=getattr(args, "reasoning_mode", None),
        temperature=getattr(args, "temperature", None),
        max_output_tokens=getattr(args, "max_output_tokens", None),
        top_p=getattr(args, "top_p", None),
        top_k=getattr(args, "top_k", None),
        min_p=getattr(args, "min_p", None),
        frequency_penalty=getattr(args, "frequency_penalty", None),
        presence_penalty=getattr(args, "presence_penalty", None),
        repetition_penalty=getattr(args, "repetition_penalty", None),
        seed=getattr(args, "seed", None),
        stop_sequences=getattr(args, "stop_sequences", None),
        reset_generation_settings=bool(
            getattr(args, "reset_generation_settings", False)
        ),
        vision=getattr(args, "vision", None),
        mandatory_offload=(getattr(args, "mandatory_offload") == "on" if getattr(args, "mandatory_offload", None) is not None else None),
        offload_threshold_percent=getattr(args, "offload_threshold", None),
        working_memory_tokens=getattr(args, "working_memory_tokens", None),
        auto_repair=(getattr(args, "auto_repair") == "on" if getattr(args, "auto_repair", None) is not None else None),
        force=bool(getattr(args, "force", False)),
    )


def _can_setup_noninteractive(options: SetupOptions) -> bool:
    provider = (
        options.provider
        or ("custom" if options.api_url or options.custom_contract_file else "")
    ).lower()
    preset = PROVIDERS.get(provider)
    environment_key = os.getenv("ARTIFICIUM_API_KEY") or (
        os.getenv(preset.key_environment) if preset else None
    )
    contract = {}
    if options.custom_contract_file:
        try:
            contract = load_custom_contract(options.custom_contract_file)
        except (OSError, ValueError, json.JSONDecodeError):
            # Let SetupWizard produce the precise contract error.
            return True
    adapter = (
        "custom_json"
        if contract
        else options.adapter or (preset.adapter if preset else "openai_compatible")
    )
    key_available = bool(options.api_key or options.api_key_file or environment_key)
    key_required = requires_api_key(provider, adapter) or (
        bool(contract) and custom_contract_requires_api_key(contract)
    )
    return bool(
        (options.provider or options.api_url or options.custom_contract_file)
        and (options.model or provider in SetupWizard.DISCOVERABLE)
        and (key_available or not key_required)
    )


def _setup(paths: Paths, args: argparse.Namespace) -> None:
    options = _setup_options(args)
    interactive = sys.stdin.isatty() and not getattr(args, "yes", False)
    if not interactive and not _can_setup_noninteractive(options):
        raise RuntimeError(
            "Interactive setup requires a terminal. Pass --provider and --model; "
            "cloud providers also require --api-key or their standard environment variable."
        )
    result = SetupWizard(paths).run(options, interactive=interactive)
    print(f"\n{result['name']} initialized.")
    print(f"Mind: {result['mind']}")
    print(f"Self: {result['self']}")
    print(f"Logs: {result['logs']}")
    print(
        f"Engine: {result['provider']}/{result['model']} · "
        f"model context window: {int(result['context_window_tokens']):,} tokens · "
        f"reasoning: {result['reasoning_effort']} · "
        f"native image input: {result['vision']}"
    )


def _has_reconfigure_values(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None) is not None
        for name in (
            "provider",
            "model",
            "api_url",
            "endpoint",
            "adapter",
            "custom_contract_file",
            "api_key",
            "api_key_file",
            "openrouter_provider",
            "context_window",
            "reasoning", "request_timeout", "reasoning_effort",
            "reasoning_budget_tokens",
            "reasoning_mode",
            "temperature",
            "max_output_tokens",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "seed",
            "stop_sequences",
            "vision",
            "mandatory_offload",
            "offload_threshold",
            "working_memory_tokens",
            "auto_repair",
        )
    ) or bool(
        getattr(args, "reset_generation_settings", False)
    )


def _reconfigure(paths: Paths, args: argparse.Namespace) -> None:
    options = _setup_options(args)
    interactive = sys.stdin.isatty() and not _has_reconfigure_values(args) and not getattr(args, "yes", False)
    if interactive and not sys.stdin.isatty():
        raise RuntimeError("interactive configure requires a terminal or explicit flags")
    updated = SetupWizard(paths).reconfigure(options, interactive=interactive)
    pid, alive = _pid_state(paths)
    print("Configuration saved. Self and memory were not changed.")
    print(
        f"Engine: {updated.provider}/{updated.model} · "
        f"model context window: {updated.context_window_tokens:,} tokens · "
        f"reasoning: {reasoning_label(updated)} · "
        f"native image input: {updated.vision}"
    )
    print(f"Vision preference: {updated.vision_preference}; mandatory offloading: "
          + (f"{updated.offload_threshold_percent:g}%" if updated.mandatory_offload else "off"))
    print(f"Working-memory target: {updated.working_memory_limit:,} tokens"
          + (" (same as model)" if updated.working_memory_tokens is None else "")
          + f"; automatic repair: {'on' if updated.auto_repair else 'off'}")
    if alive and pid:
        print(
            f"Artificium is currently running as PID {pid}. Restart it to apply "
            "configuration changes: python3 artificium.py restart"
        )


def _pid_state(paths: Paths) -> tuple[int | None, bool]:
    state = process_state(paths)
    return state["pid"], bool(state["alive"] and state["owned"] is not False)


def _print_detach_status(paths: Paths, *, client: str) -> None:
    """Make detaching from a client impossible to confuse with stopping the agent."""

    pid, alive = _pid_state(paths)
    label = client.upper()
    if alive and pid:
        print(f"\nWARNING: {label} CLOSED — ARTIFICIUM IS STILL RUNNING")
        print(f"Life-loop PID: {pid}")
        print("TO STOP ARTIFICIUM: python3 artificium.py stop")
    else:
        print(f"\n{label} CLOSED — ARTIFICIUM IS NOT RUNNING")


def _start_background(paths: Paths) -> int:
    pid, alive = _pid_state(paths)
    if alive and pid:
        return pid
    launcher = paths.code / "artificium.py"
    output = (paths.logs / "daemon.stdout.log").open("a", encoding="utf-8")
    error = (paths.logs / "daemon.stderr.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(launcher), "--root", str(paths.root), "run", "--quiet"],
        cwd=paths.root,
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=error,
        start_new_session=True,
    )
    output.close()
    error.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        found, running = _pid_state(paths)
        if running and found:
            return found
        if process.poll() is not None:
            raise RuntimeError(
                f"background life-loop exited; inspect {paths.logs / 'daemon.stderr.log'}"
            )
        time.sleep(0.1)
    raise RuntimeError(
        f"background life-loop did not become ready; inspect {paths.logs / 'daemon.stderr.log'}"
    )


def _stop_background(paths: Paths) -> bool:
    state = process_state(paths)
    if state["alive"] and state["owned"] is None:
        raise RuntimeError("Cannot verify the lock PID belongs to Artificium; inspect it before stopping manually.")
    pid, alive = state["pid"], state["alive"] and state["owned"] is True
    if not alive or not pid:
        paths.process_lock.unlink(missing_ok=True)
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        paths.process_lock.unlink(missing_ok=True)
        return False
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _, running = _pid_state(paths)
        if not running:
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        _, running = _pid_state(paths)
        if not running:
            paths.process_lock.unlink(missing_ok=True)
            return True
        time.sleep(0.05)
    raise RuntimeError(
        f"could not stop PID {pid}; inspect {paths.process_lock} and terminate it explicitly"
    )


def _terminal_command(paths: Paths, chat_args: list[str]) -> list[str] | None:
    base = [
        sys.executable,
        str(paths.code / "artificium.py"),
        "--root",
        str(paths.root),
        *chat_args,
    ]
    configured = os.getenv("TERMINAL")
    candidates: list[list[str]] = []
    if configured:
        candidates.append([*shlex.split(configured), "-e", *base])
    candidates.extend(
        [
            ["gnome-terminal", "--", *base],
            ["konsole", "-e", *base],
            ["kitty", *base],
            ["wezterm", "start", "--", *base],
            ["xterm", "-e", *base],
        ]
    )
    for command in candidates:
        if shutil.which(command[0]):
            return command
    return None


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
    client = ArtificiumClient(paths.root)
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
                pid, alive = _pid_state(paths)
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
        _print_detach_status(paths, client="chat client")


def _watch_life_loop(paths: Paths, *, tail: int = 30) -> None:
    path = paths.life_loop_log
    path.touch(exist_ok=True)
    print(f"Watching {path}. Ctrl-C detaches this viewer; it does not stop Artificium.\n")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
            recent = lines[-int(tail) :] if int(tail) > 0 else []
            for line in recent:
                _print_life_record(line)
            handle.seek(0, os.SEEK_END)
            while True:
                line = handle.readline()
                if line:
                    _print_life_record(line)
                else:
                    time.sleep(0.25)
    except KeyboardInterrupt:
        _print_detach_status(paths, client="life-loop viewer")


def _print_life_record(line: str) -> None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return
    kind = str(value.get("kind") or "event")
    if kind == "thought":
        print(f"[think] {' '.join(str(value.get('content') or '').splitlines())}")
    elif kind == "tool_call":
        arguments = value.get("arguments") or {}
        focus = arguments.get("path") or arguments.get("source") or arguments.get("interaction_id") or ""
        print(f"[tool] {value.get('name')} {focus}".rstrip())
    elif kind == "tool_result":
        result = value.get("result") or {}
        print(f"[result] {value.get('name')}: {result.get('summary') or result.get('status')}")
    elif kind in {"working_memory_offloaded", "context_compacted", "memory_saved"}:
        print(
            f"[{kind}] {value.get('path')} "
            f"{value.get('before_tokens', '')} -> {value.get('after_tokens', '')}"
        )
    elif kind == "life_loop_output":
        print(f"[output] {' '.join(str(value.get('content') or '').splitlines())}")
    elif kind == "context_usage":
        tokens = int(value.get("estimated_tokens", 0) or 0)
        window = int(value.get("context_window_tokens", 0) or 0)
        percent = float(value.get("context_percent", 0.0) or 0.0)
        source = value.get("token_count_source", "estimate")
        prefix = "~" if source == "estimate" else ""
        print(f"[context] {prefix}{tokens:,} / {window:,} tokens ({percent:.1f}%; {source})")
    elif kind.startswith("request_repair_"):
        print(f"[recovery] {kind.removeprefix('request_repair_')}: "
              f"{value.get('archive') or value.get('error') or value.get('attempts')}")
    elif kind == "guidance_notification":
        print(
            f"[guidance] {value.get('guidance_type')}: "
            f"~{int(value.get('estimated_tokens', 0) or 0):,} tokens"
        )
    elif kind == "engine_request":
        print(
            f"[engine] request {value.get('request_id')} sent to "
            f"{value.get('provider')}/{value.get('model')} "
            f"({int(value.get('estimated_tokens', 0) or 0):,} input tokens, {value.get('token_count_source', 'estimate')})"
        )
    elif kind == "engine_waiting":
        print(
            f"[engine] request {value.get('request_id')} still waiting for provider "
            f"({float(value.get('elapsed_seconds', 0) or 0):.0f}s elapsed)"
        )
    elif kind == "engine_response":
        print(
            f"[engine] request {value.get('request_id')} completed in "
            f"{float(value.get('duration_seconds', 0) or 0):.1f}s"
        )
    elif kind in {"engine_request_failed", "engine_request_cancelled"}:
        print(
            f"[engine] request {value.get('request_id')} "
            f"{'failed' if kind.endswith('failed') else 'cancelled'} after "
            f"{float(value.get('duration_seconds', 0) or 0):.1f}s; "
            f"log={value.get('model_log_path')}"
        )
        if value.get("error"):
            print("[engine] " + " ".join(str(value["error"]).splitlines())[:1600])
        if value.get("hint"):
            print("[engine] " + " ".join(str(value["hint"]).splitlines())[:1600])
    elif kind == "engine_blocked":
        print(
            "[engine] Model requests paused; history and incoming messages are retained. "
            "Correct the reported problem, then run: "
            + str(value.get("recovery_command") or "python3 artificium.py restart")
        )


def _launcher(paths: Paths) -> None:
    print("\nWhat would you like to do?\n")
    print("1. Start a chat")
    print("2. Watch the life-loop")
    choice = input("\nChoose [1]: ").strip() or "1"
    if choice == "1":
        entity = safe_identifier(
            input("Your entity name/ID [user_1]: ").strip() or "user_1",
            label="entity id",
        )
        client = ArtificiumClient(paths.root)
        interaction_id = _choose_interaction(client, entity, None, None)
        pid = _start_background(paths)
        print(f"Life-loop running as PID {pid}.")
        args = ["chat", "--entity", entity, "--interaction", interaction_id]
        command = _terminal_command(paths, args)
        if command:
            subprocess.Popen(command, cwd=paths.root, start_new_session=True)
            print("Chat opened in a new terminal.")
        else:
            print("No supported terminal launcher was detected; opening chat here.")
            _chat_repl(paths, entity, interaction_id, None)
        return
    if choice == "2":
        _, alive = _pid_state(paths)
        if not alive:
            pid = _start_background(paths)
            print(f"Life-loop started as PID {pid}.")
        _watch_life_loop(paths)
        return
    raise ValueError("choose 1 or 2")


def _replace_key(paths: Paths, args: argparse.Namespace) -> None:
    key = args.api_key or ""
    if args.api_key_file:
        key = load_api_key_file(args.api_key_file)
    if not key:
        if not sys.stdin.isatty():
            raise RuntimeError("pass --api-key or --api-key-file")
        key = getpass.getpass("New API key: ").strip()
    config = ConfigStore(paths).load()
    verify_connection(paths, config, key)
    SecretsStore(paths).save_api_key(key, provider=config.provider, base_url=config.base_url)
    print("API key saved. A running life-loop will reload it automatically.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = configured_paths(args.root)
    store = ConfigStore(paths)
    try:
        if args.command == "upgrade":
            from .upgrade import upgrade
            upgrade(paths, repository=args.repo, ref=args.ref, check=args.check)
            return 0
        if args.command in {"setup", "init"}:
            _setup(paths, args)
            if sys.stdin.isatty() and not args.no_launch:
                _launcher(paths)
            else:
                print(
                    "Start: "
                    f"{sys.executable} {paths.code / 'artificium.py'} --root "
                    f"{paths.root}"
                )
            return 0
        if args.command is None:
            if not store.exists():
                namespace = argparse.Namespace(
                    provider=None,
                    model=None,
                    api_key=None,
                    api_key_file=None,
                    api_url=None,
                    endpoint=None,
                    adapter=None,
                    self_directive=None,
                    self_file=None,
                    context_window=None,
                    temperature=None,
                    max_output_tokens=None,
                    vision=None,
                    force=False,
                )
                _setup(paths, namespace)
            _launcher(paths)
            return 0
        if args.command == "status":
            state = status_snapshot(paths)
            print(json_dumps(state, pretty=True) if args.json else format_status(state))
            return 0
        if args.command == "models":
            options = _setup_options(args)
            wizard = SetupWizard(paths)
            current = store.load() if store.exists() else None
            provider, base, _, _, contract, key = wizard._connection(options, current)
            if contract:
                raise ValueError("An arbitrary custom JSON contract has no model-discovery endpoint.")
            found = wizard._probe(provider, base, key)
            print(json_dumps(dataclasses.asdict(found), pretty=True))
            return 0 if found.models else 1
        if not store.exists():
            raise RuntimeError("Artificium is not configured. Run without a command first.")
        if args.command == "restart":
            config = store.load()
            make_engine(config, SecretsStore(paths).resolve_api_key(config)).request_summary()
            _stop_background(paths)
            print(f"Life-loop restarted as PID {_start_background(paths)}.")
            return 0
        if args.command == "start":
            print(f"Life-loop running as PID {_start_background(paths)}.")
            return 0
        if args.command == "stop":
            print("Life-loop stopped." if _stop_background(paths) else "Life-loop was not running.")
            return 0
        if args.command == "watch":
            _, alive = _pid_state(paths)
            if not alive:
                print(
                    "Life-loop is not currently running. Watching its durable trace "
                    "without starting or restarting it."
                )
            _watch_life_loop(paths, tail=args.tail)
            return 0
        if args.command == "chat":
            _start_background(paths)
            _chat_repl(paths, args.entity, args.interaction, args.name)
            return 0
        if args.command == "send":
            event, path = ArtificiumClient(paths.root).send(
                args.interaction,
                sender=args.sender,
                content=args.content,
                interaction_name=args.name,
                attachments=args.attachment,
                in_reply_to=args.in_reply_to,
                recipient=args.recipient,
                kind=args.kind,
            )
            print(json_dumps({"event": event, "path": str(path)}, pretty=True))
            return 0
        if args.command == "show":
            client = ArtificiumClient(paths.root)
            print(json_dumps(client.events(args.interaction_id), pretty=True))
            return 0
        if args.command == "notify":
            client = ArtificiumClient(paths.root)
            item = client.notifications.create(
                type=args.type,
                summary=args.summary,
                source=args.source,
                path=args.path,
            )
            print(json_dumps(item.to_dict(), pretty=True))
            return 0
        if args.command in {"attention", "stream"}:
            source = str(Path(args.source).expanduser().resolve())
            client = ArtificiumClient(paths.root)
            item = client.notifications.create(
                type="attention_request",
                source="owner_cli",
                summary="An operator requested Infinite Attention over a durable source.",
                path=source,
                metadata={
                    "objective": args.objective,
                    "granularity": args.granularity,
                    "output_path": args.output,
                },
            )
            print(json_dumps(item.to_dict(), pretty=True))
            return 0
        if args.command == "key":
            _replace_key(paths, args)
            return 0
        if args.command in {"configure", "reconfigure", "connect"}:
            _reconfigure(paths, args)
            return 0
        if args.command == "config":
            grouped = store.load().grouped_dict()
            print(json_dumps(grouped[args.section] if args.section else grouped, pretty=True))
            return 0
        if args.command in {"doctor", "check"}:
            config = store.load()
            key = SecretsStore(paths).resolve_api_key(config)
            prompt_pack = PromptPack(paths)
            engine = make_engine(config, key)
            result: dict[str, Any] = {
                "root": str(paths.root),
                "three_roots": {
                    "artificium-code": paths.code.is_dir(),
                    "mind": paths.mind.is_dir(),
                    "logs": paths.logs.is_dir(),
                },
                "provider": config.provider,
                "model": config.model,
                "adapter": config.adapter,
                "api_base_url": config.base_url,
                "api_key_required": (
                    requires_api_key(config.provider, config.adapter)
                    or custom_contract_requires_api_key(config.custom_contract)
                ),
                "api_key_available": bool(key),
                "self": paths.self_file.is_file(),
                "meta_memory": paths.meta_memory.is_file(),
                "scheduler_state": paths.scheduler_tasks.is_dir(),
                "prompt_pack": prompt_pack.version,
                "prompt_files": len(prompt_pack.fingerprints()),
                "engine_request": engine.request_summary(),
            }
            if (
                config.provider in SetupWizard.DISCOVERABLE
                and config.adapter != "custom_json"
            ):
                discovery = discover_provider_models(
                    config.provider,
                    config.base_url,
                    api_key=key,
                    timeout=3.0,
                )
                model_details = SetupWizard(paths)._details(config.provider, config.base_url, config.model, key or "", discovery)
                detected_context = model_details.get("context_length")
                result["provider_probe"] = {
                    "endpoint": discovery.endpoint,
                    "error": discovery.error,
                    "configured_model_found": config.model in discovery.models,
                    "models_found": len(discovery.models),
                    "configured_model_details": model_details,
                    "detected_context_window": detected_context,
                    "working_memory_fits_detected_context": (
                        config.context_window_tokens <= detected_context
                        if detected_context
                        else None
                    ),
                }
            if args.live:
                # Resolve metadata exactly as setup does, but never save or run
                # the agent while diagnosing an existing instance.
                wizard = SetupWizard(paths)
                candidate, candidate_key = wizard._build(SetupOptions(scope="model"), config)
                _, check_result = verify_connection(paths, candidate, candidate_key or None,
                                                     report=lambda text: print(text, file=sys.stderr))
                result["connection_check"] = check_result
            print(json_dumps(result, pretty=True))
            return 0
        if args.command == "logs":
            records = Records(paths)
            if args.summary:
                value = records.feature_usage()
            elif args.feature:
                value = records.recent_feature(args.feature, args.limit)
            elif args.lifetime:
                value = records.recent_operational(args.limit)
            else:
                value = records.recent_life(args.limit)
            print(json_dumps(value, pretty=True))
            return 0
        agent = Artificium(paths)
        if args.command == "run":
            if args.once:
                output = agent.run_once(trigger="manual")
                if output and not args.quiet:
                    print(output)
            else:
                agent.run_forever(verbose=args.verbose, quiet=args.quiet)
            return 0
        parser.error(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        print("artificium: cancelled by operator", file=sys.stderr)
        return 130
    except (
        EngineError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"artificium: error: {exc}", file=sys.stderr)
        if getattr(exc, "hint", None):
            print(exc.hint, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
