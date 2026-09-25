"""Terminal setup: keep entered values while a connection is corrected."""
from __future__ import annotations

import copy
import getpass

from .config import (Config, PROVIDERS, parse_first_token_timeout, parse_request_timeout,
                    parse_stall_timeout, requires_api_key)
from .engine import EngineError
from .setup import (ModelDiscovery, SetupOptions, _ask, _choice, _yes,
                    choose_model, choose_provider, configure_generation_interactive,
                    configure_reasoning, configured_openrouter_provider,
                    custom_contract_requires_api_key, reasoning_label)


class SetupCancelled(RuntimeError):
    pass


def edit_context(options: SetupOptions, config: Config | None) -> None:
    current = options.context_window_tokens or (config.context_window_tokens if config else 32768)
    while True:
        raw = _ask("Context capacity in tokens (or auto to rediscover)", str(current)).lower()
        if raw == "auto":
            options.context_window_tokens = None
            options.context_window_source = "detected"
            return
        try:
            number = int(raw.replace(",", "").replace("_", ""))
            if number < 4000:
                raise ValueError
            options.context_window_tokens, options.context_window_source = number, "manual"
            return
        except ValueError:
            print("Enter at least 4,000 tokens, or auto. This must match the server's allocation.")


def edit_request_timeout(options: SetupOptions, config: Config | None) -> None:
    current = options.request_timeout_seconds
    if current is None and config is not None:
        current = config.request_timeout_seconds
    while True:
        raw = _ask("Request timeout in seconds, or off to wait indefinitely",
                   "off" if current is None else str(current))
        try:
            seconds = parse_request_timeout(raw)
            options.request_timeout_seconds = seconds if seconds is not None else "off"
            return
        except ValueError as exc:
            print(exc)


def edit_stall_timeout(options: SetupOptions, config: Config | None) -> None:
    current = options.stall_timeout_seconds
    if current is None and config is not None:
        current = config.stall_timeout_seconds
    while True:
        raw = _ask("Streaming stall (inactivity) timeout in seconds, or off to rely only on the request timeout",
                   "off" if current is None else str(current))
        try:
            seconds = parse_stall_timeout(raw)
            options.stall_timeout_seconds = seconds if seconds is not None else "off"
            return
        except ValueError as exc:
            print(exc)


def edit_stream_responses(options: SetupOptions, config: Config | None) -> None:
    current = options.stream_responses
    if current is None and config is not None:
        current = config.stream_responses
    options.stream_responses = _yes(
        "Stream responses (progress visible as the model generates; "
        "falls back to non-streaming automatically if the server rejects it)",
        bool(current),
    )


def edit_first_token_timeout(options: SetupOptions, config: Config | None) -> None:
    current = options.first_token_timeout_seconds
    if current is None and config is not None:
        current = config.first_token_timeout_seconds
    while True:
        raw = _ask("First-token (prefill) timeout in seconds, or off to wait for the full request timeout",
                   "off" if current is None else str(current))
        try:
            seconds = parse_first_token_timeout(raw)
            options.first_token_timeout_seconds = seconds if seconds is not None else "off"
            return
        except ValueError as exc:
            print(exc)


def model_editor(wizard, options: SetupOptions, current: Config | None) -> None:
    while True:
        config, _ = wizard._build(options, current)
        print("\nModel settings")
        print(f"  1. Reasoning: {reasoning_label(config)}")
        print(f"  2. Context capacity: {config.context_window_tokens:,}")
        print("  3. Sampling and output limits")
        timeout = config.request_timeout_seconds
        print("  4. Request timeout: " + ("off (wait indefinitely)" if timeout is None else f"{timeout:g} seconds"))
        stall = config.stall_timeout_seconds
        print("  5. Streaming stall timeout: " + ("off (rely on request timeout)" if stall is None else f"{stall:g} seconds"))
        first_token = config.first_token_timeout_seconds
        print("  6. First-token (prefill) timeout: " + ("off (wait for request timeout)" if first_token is None else f"{first_token:g} seconds"))
        print(f"  7. Stream responses: {'on' if config.stream_responses else 'off'}")
        if config.provider == "openrouter":
            print("  8. Inference provider routing")
        print("  reset. Restore generation defaults")
        action = _ask("Setting number or done", "done").lower()
        if action in {"done", "back"}:
            return
        if action == "1":
            configure_reasoning(options, config, config.model_capabilities)
        elif action == "2":
            edit_context(options, config)
        elif action == "3":
            configure_generation_interactive(options, current=config,
                provider="llamacpp" if config.adapter == "llamacpp" else config.provider, model=config.model)
        elif action == "4":
            edit_request_timeout(options, config)
        elif action == "5":
            edit_stall_timeout(options, config)
        elif action == "6":
            edit_first_token_timeout(options, config)
        elif action == "7":
            edit_stream_responses(options, config)
        elif action == "8" and config.provider == "openrouter":
            options.openrouter_provider = _ask("Inference provider slug, or automatic", configured_openrouter_provider(config.request_options) or "automatic")
        elif action == "reset":
            wizard._clear_generation(options)
        else:
            print("Choose a setting from the menu.")


def interactive_setup(wizard, options: SetupOptions | None = None, *, current: Config | None = None):
    options = copy.deepcopy(options or SetupOptions())
    if options.scope != "model":
        wizard._edit_harness(options, current)
    settings_reviewed = False
    config = None
    while True:
        try:
            print("\nModel connection")
            if not options.provider:
                options.provider = choose_provider(current.provider if current else "openrouter")
            same = current is not None and current.provider == options.provider
            preset = PROVIDERS.get(options.provider)
            contract = options.custom_contract_file or (current.custom_contract if same else None)
            if options.provider in wizard.LOCAL and not contract and not options.api_url:
                options.api_url = _ask("Server address", current.base_url if same else preset.base_url if preset else "http://127.0.0.1:8000")
            provider, base, endpoint, adapter, contract, key = wizard._connection(options, current)
            if (requires_api_key(provider, adapter) or custom_contract_requires_api_key(contract)) and not key:
                options.api_key = getpass.getpass("API key (hidden): ").strip()
                key = options.api_key
            headers = current.headers if same and base == current.base_url else {}
            print(f"Connecting to {base} …")
            discovery = wizard._probe(provider, base, key, headers) if not contract else ModelDiscovery()
            if discovery.authentication_required and not key:
                options.api_key = getpass.getpass("This server requires an API key (hidden): ").strip()
                key = options.api_key
                discovery = wizard._probe(provider, base, key, headers)
            # Model-list routes are optional. Connectivity/auth failures are
            # corrected before asking for a model identifier or context limit.
            if discovery.error and discovery.status != 404 and discovery.error_kind != "response":
                raise EngineError(discovery.error, status=discovery.status,
                                  kind=discovery.error_kind, hint=discovery.hint)
            if not options.model:
                options.model = choose_model(discovery.models, default=current.model if same else "")
            config, key = wizard._build(options, current)
            if not settings_reviewed:
                if config.context_window_source == "detected":
                    print(f"Detected serving context: {config.context_window_tokens:,} tokens.")
                elif config.context_window_source == "default":
                    if config.provider == "ollama":
                        print(f"Ollama will allocate {config.context_window_tokens:,} context tokens for Artificium.")
                    else:
                        print("The server did not report its context limit. Use its serving allocation here.")
                        edit_context(options, config)
                        config, key = wizard._build(options, current)
                if options.reasoning is None and not any(getattr(options, field) is not None for field in ("reasoning_effort", "reasoning_budget_tokens", "reasoning_mode")):
                    configure_reasoning(options, config, config.model_capabilities)
                if _yes("Adjust advanced model settings", False):
                    model_editor(wizard, options, current)
                settings_reviewed = True
                config, key = wizard._build(options, current)
            print(f"\n{config.model} · context {config.context_window_tokens:,} · reasoning {reasoning_label(config)}")
            print(f"API: {config.base_url}/{config.endpoint}")
            print(f"Images: {config.vision_preference}; the connection check verifies support.")
            if not _yes("Check connection and save", True):
                model_editor(wizard, options, current)
                continue
            config = wizard._verify(options, config, key)
            return config, key, options
        except (EngineError, ValueError, OSError) as exc:
            print(f"\nConnection not saved: {exc}")
            if getattr(exc, "hint", None):
                print(exc.hint)
            print("\n  1. Retry after fixing/restarting the server")
            print("  2. Change server or provider")
            print("  3. Choose another model")
            print("  4. Change API key")
            print("  5. Change context capacity")
            print("  6. Reset generation settings to server defaults")
            print("  7. Change request timeout")
            print("  8. Edit model settings")
            print("  9. Cancel; keep existing configuration")
            default = "4" if getattr(exc, "kind", None) == "auth" else "1"
            action = _choice("Next step", [str(x) for x in range(1, 10)], default)
            if action == "9":
                raise SetupCancelled("Cancelled. Existing configuration and mind are unchanged.")
            if action == "2":
                options.provider = choose_provider(options.provider or "openrouter")
                options.api_url = options.endpoint = options.adapter = options.model = None
                options.api_key = options.api_key_file = None
                options.custom_contract_file = None
                selected = PROVIDERS.get(options.provider)
                options.api_url = _ask("Server/API address", selected.base_url if selected else "http://127.0.0.1:8000")
                options.context_window_tokens = None
                options.context_window_source = "detected"
                wizard._clear_generation(options)
                options.reasoning = None
                settings_reviewed = False
            elif action == "3":
                options.model = None
                options.context_window_tokens = None
                options.context_window_source = "detected"
                wizard._clear_generation(options)
                options.reasoning = None
                settings_reviewed = False
            elif action == "4":
                options.api_key = getpass.getpass("API key (hidden): ").strip()
                options.api_key_file = None
            elif action == "5":
                edit_context(options, config)
            elif action == "6":
                wizard._clear_generation(options)
            elif action == "7":
                edit_request_timeout(options, config or current)
            elif action == "8":
                settings_reviewed = False
            wizard._discoveries.clear()
            wizard._properties.clear()
