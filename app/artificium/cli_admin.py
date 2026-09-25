"""Setup, reconfigure, key, reset, and overrides admin commands for the CLI.

`_pid_state` and `process_state` are resolved through `artificium.cli` at
call time (not imported directly) because tests patch them at
"artificium.cli.process_state" / "artificium.cli._pid_state"; a direct import
here would capture the original function before any such patch is applied.
"""

from __future__ import annotations

import argparse
import difflib
import getpass
import json
import os
import shutil
import sys
from pathlib import Path

from . import cli as _cli
from .config import (
    PROVIDERS,
    ConfigStore,
    SecretsStore,
    load_api_key_file,
    requires_api_key,
)
from .connection import verify_connection
from .filesystem import Paths
from .setup import (
    SetupOptions,
    SetupWizard,
    custom_contract_requires_api_key,
    load_custom_contract,
    reasoning_label,
)


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
        stall_timeout_seconds=getattr(args, "stall_timeout", None),
        first_token_timeout_seconds=getattr(args, "first_token_timeout", None),
        stream_responses=(getattr(args, "stream_responses") == "on" if getattr(args, "stream_responses", None) is not None else None),
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
            "reasoning", "request_timeout", "stall_timeout", "first_token_timeout",
            "stream_responses", "reasoning_effort",
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
    pid, alive = _cli._pid_state(paths)
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
          + (f" ({updated.working_memory_fraction:.0%} of model)" if updated.working_memory_tokens is None and updated.working_memory_fraction
             else " (same as model)" if updated.working_memory_tokens is None else "")
          + f"; automatic repair: {'on' if updated.auto_repair else 'off'}")
    if alive and pid:
        print(
            f"Artificium is currently running as PID {pid}. Restart it to apply "
            "configuration changes: python3 artificium.py restart"
        )


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


def _confirm(question: str, *, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        raise RuntimeError(f"{question} Re-run with --yes to confirm without a terminal.")
    return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}


def _require_stopped(paths: Paths, action: str) -> None:
    if _cli.process_state(paths)["alive"]:
        raise RuntimeError(
            f"Stop Artificium before {action}: python3 artificium.py stop"
        )


def _reset(paths: Paths, args: argparse.Namespace) -> int:
    """Return the workspace to its shipped state, keeping the connection."""

    _require_stopped(paths, "resetting the workspace")
    if paths.root.is_dir():
        if not _confirm(
            f"This will delete everything the agent has learned in {paths.root}. "
            f"Settings in {paths.config.name} and {paths.secrets.name} are kept. Continue?",
            assume_yes=args.yes,
        ):
            print("Cancelled; nothing was changed.")
            return 1
        shutil.rmtree(paths.root)
        print(f"Removed {paths.root}")
    seeded = paths.ensure_layout()
    print(f"Grew a new workspace from the seed: {paths.root} ({len(seeded)} files)")
    print("Start: python3 artificium.py start")
    return 0


def _overlay_entries(paths: Paths) -> list[tuple[str, Path, Path | None]]:
    """Every overlay file, with the shipped file it shadows when there is one."""

    entries: list[tuple[str, Path, Path | None]] = []
    for overlay, shipped in (
        (paths.prompt_overrides, paths.prompts),
        (paths.code_overrides, paths.code / "artificium"),
    ):
        if not overlay.is_dir():
            continue
        for path in sorted(overlay.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(overlay)
            original = shipped / relative
            name = f"{overlay.name}/{relative.as_posix()}"
            entries.append((name, path, original if original.is_file() else None))
    return entries


def _overrides(paths: Paths, args: argparse.Namespace) -> int:
    entries = _overlay_entries(paths)
    selected = [item for item in entries if args.name in (None, item[0])]
    if args.name and not selected:
        raise RuntimeError(f"No such override: {args.name}")

    if args.action == "list":
        if paths.quarantined_overrides.is_dir():
            print(f"quarantined: {paths.quarantined_overrides}")
        if not entries:
            print(f"No overrides. The agent runs entirely on {paths.app}.")
            return 0
        for name, path, original in entries:
            shadows = "shadows the shipped file" if original else "shadows nothing shipped"
            print(f"{name}  ({path.stat().st_size} bytes, {shadows})")
        print("\nApplied at the next restart. Compare with: overrides diff [NAME]")
        return 0

    if args.action == "diff":
        shown = False
        for name, path, original in selected:
            if original is None:
                print(f"--- {name}: shadows nothing shipped; it is workspace-only\n")
                continue
            lines = list(difflib.unified_diff(
                original.read_text(encoding="utf-8", errors="replace").splitlines(True),
                path.read_text(encoding="utf-8", errors="replace").splitlines(True),
                fromfile=f"shipped/{name}", tofile=f"override/{name}",
            ))
            shown = True
            print("".join(lines) if lines else f"--- {name}: identical to the shipped file\n")
        if not shown and not selected:
            print("No overrides to compare.")
        return 0

    if not selected:
        print("No overrides to remove.")
        return 0
    target = "this override" if args.name else f"all {len(selected)} overrides"
    if not _confirm(f"Remove {target}?", assume_yes=args.yes):
        print("Cancelled; nothing was changed.")
        return 1
    for name, path, _ in selected:
        path.unlink()
        print(f"Removed {name}")
    print("Restart to run on the shipped harness: python3 artificium.py restart")
    return 0

