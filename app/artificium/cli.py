from __future__ import annotations

import argparse
import dataclasses
import json
import os  # re-exported so `artificium.cli.os` stays a patchable name
import sys
import time  # re-exported so `artificium.cli.time` stays a patchable name
from pathlib import Path
from typing import Any

from . import VERSION, _bootstrap
from .config import (
    PROVIDERS,
    ConfigStore,
    SecretsStore,
    configured_paths,
    requires_api_key,
)
from .engine import EngineError, make_engine
from .connection import verify_connection
from .filesystem import json_dumps
from .interactions import ArtificiumClient
from .prompts import PromptPack
from .records import Records
from .runtime import Artificium
from .operator import process_state, status_snapshot, format_status
from .setup import (
    SetupOptions,
    SetupWizard,
    custom_contract_requires_api_key,
    discover_provider_models,
)

# cli.py keeps argument parsing (build_parser) and top-level dispatch
# (main/_dispatch); the command groups live in cli_admin.py (setup,
# reconfigure, key, reset, overrides), cli_service.py (background process
# control and the terminal launcher), cli_chat.py (the terminal REPL), and
# cli_watch.py (the `watch` command). Every name they define is re-exported
# here so `artificium.cli.<name>` keeps resolving -- for callers, for tests
# that patch e.g. "artificium.cli.process_state" or "artificium.cli._pid_state",
# and for a workspace code overlay written against the pre-split module. Those
# submodules resolve a handful of cross-module/patch-sensitive names (like
# `process_state` and `_pid_state`) back through this module at call time
# rather than importing them directly; see their docstrings.
from .cli_admin import (
    _can_setup_noninteractive,
    _confirm,
    _has_reconfigure_values,
    _overlay_entries,
    _overrides,
    _reconfigure,
    _replace_key,
    _require_stopped,
    _reset,
    _setup,
    _setup_options,
)
from .cli_service import (
    _launcher,
    _pid_state,
    _print_detach_status,
    _start_background,
    _stop_background,
    _terminal_command,
)
from .cli_chat import (
    _chat_display_width,
    _chat_input_prompt,
    _chat_input_rows,
    _chat_line_buffer,
    _chat_repl,
    _choose_interaction,
    _clear_chat_input,
    _local_time,
    _print_event,
    _restore_chat_input,
)
from .cli_watch import (
    _LogFollower,
    _parse_only,
    _print_life_record,
    _tail_lines_from_end,
    _watch_life_loop,
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
    group.add_argument("--mandatory-offload", choices=["on", "off"], help="Require memory offloading at the threshold (default: on)")
    group.add_argument("--offload-threshold", type=float, help="Working-memory percentage, 1–95 (default: 80)")
    group.add_argument("--working-memory-tokens", metavar="TOKENS|auto|same", help="Offloading target; auto is 60%% of the model context (default for new setups), same tracks the whole context")
    group.add_argument("--auto-repair", "--emergency-offload", dest="auto_repair", choices=["on", "off"], help="Try earlier context after input failures; up to 3 attempts (default: on)")


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
    parser.add_argument("--request-timeout", metavar="SECONDS|off", help="Seconds to wait for inference, or off to wait indefinitely (default: 1800)")
    parser.add_argument("--stall-timeout", metavar="SECONDS|off", help="Abort a streamed response after this many seconds without new data (default: 180)")
    parser.add_argument("--first-token-timeout", metavar="SECONDS|off", help="Limit on waiting for the first streamed data; off leaves prefill bounded only by the request timeout (default: off)")
    parser.add_argument("--stream-responses", choices=["on", "off"], help="Stream model responses; required for the stall timeout")
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
        help="Project root to use instead of this clone (used by the background launcher and tests)",
    )
    parser.add_argument(
        "--no-overrides",
        action="store_true",
        help="Start on the shipped harness, ignoring workspace/overrides/code",
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
    reset = commands.add_parser(
        "reset",
        help="Delete the workspace and grow a factory-fresh mind from the seed",
    )
    reset.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    overrides = commands.add_parser(
        "overrides",
        help="Inspect or remove the workspace overlays on prompts and code",
    )
    overrides.add_argument(
        "action", nargs="?", default="list", choices=["list", "diff", "clear"]
    )
    overrides.add_argument(
        "name", nargs="?", help="One overlay path, such as code/runtime.py"
    )
    overrides.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    upgrade = commands.add_parser("upgrade", help="Stop-checked git pull; the workspace and settings are untouched")
    upgrade.add_argument("--check", action="store_true", help="Fetch and preview the changes without applying them")
    commands.add_parser(
        "stop",
        help="Stop the life-loop, escalating to a forced stop if needed",
    )
    watch = commands.add_parser(
        "watch", help="Attach to the life-loop trace without restarting Artificium"
    )
    watch.add_argument("--tail", type=int, default=30, help="Recent records to show first")
    watch.add_argument(
        "--only",
        help=(
            "Comma-separated categories to show, e.g. engine,tools,thoughts,context,"
            "memory,turns,notifications,recovery (default: all)"
        ),
    )
    watch.add_argument(
        "--no-thoughts", action="store_true", help="Hide thought/output/reasoning records"
    )
    watch.add_argument(
        "--since", help="Only show records at/after this local HH:MM or ISO timestamp"
    )
    watch.add_argument(
        "--reasoning", action="store_true", help="Show provider reasoning traces (hidden by default)"
    )
    watch.add_argument(
        "--full", action="store_true", help="Do not truncate long thoughts/outputs/reasoning"
    )
    watch.add_argument(
        "--max-lines", type=int, default=6,
        help="Lines to show before truncating a long thought/output block (default: 6)",
    )
    watch.add_argument("--no-color", action="store_true", help="Disable ANSI colour")
    watch.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Print raw JSON records instead of the formatted view",
    )

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


def main(argv: list[str] | None = None) -> int:
    """Run one command, then vouch for the overlay that was active during it.

    The failed-start counter is raised when the overlay loads and cleared by any
    command that reaches its own end.  That way a crash loop is caught, while
    ordinary use of the CLI never accumulates failures against a healthy overlay.
    """

    result = _dispatch(argv)
    try:
        _bootstrap.clear_attempts(configured_paths(_root_argument(argv)).root)
    except (RuntimeError, OSError):
        pass
    return result


def _root_argument(argv: list[str] | None) -> str | None:
    return _bootstrap._argument(sys.argv if argv is None else list(argv), "--root")


def _dispatch(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = configured_paths(args.root)
    store = ConfigStore(paths)
    try:
        if args.command == "reset":
            return _reset(paths, args)
        if args.command == "overrides":
            return _overrides(paths, args)
        if args.command == "upgrade":
            from .upgrade import upgrade
            upgrade(paths, check=args.check)
            return 0
        if args.command in {"setup", "init"}:
            _setup(paths, args)
            if sys.stdin.isatty() and not args.no_launch:
                _launcher(paths)
            else:
                print("Start: python3 artificium.py start")
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
            _watch_life_loop(
                paths,
                tail=args.tail,
                only=_parse_only(args.only),
                no_thoughts=args.no_thoughts,
                since=args.since,
                reasoning=args.reasoning,
                full=args.full,
                max_lines=args.max_lines,
                no_color=args.no_color,
                json_mode=args.json_output,
            )
            return 0
        if args.command == "chat":
            _start_background(paths)
            _chat_repl(paths, args.entity, args.interaction, args.name)
            return 0
        if args.command == "send":
            event, path = ArtificiumClient(paths.install).send(
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
            client = ArtificiumClient(paths.install)
            print(json_dumps(client.events(args.interaction_id), pretty=True))
            return 0
        if args.command == "notify":
            client = ArtificiumClient(paths.install)
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
            client = ArtificiumClient(paths.install)
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
                "install": str(paths.install),
                "root": str(paths.root),
                "layout": {
                    "app": paths.app.is_dir(),
                    "seed": paths.seed.is_dir(),
                    "workspace": paths.root.is_dir(),
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
