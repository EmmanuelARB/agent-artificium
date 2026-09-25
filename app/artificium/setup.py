from __future__ import annotations

import dataclasses
import copy
import getpass
import ipaddress
import json
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from . import config as _config
from .config import (
    GENERATION_FIELDS,
    Config,
    ConfigStore,
    PROVIDERS,
    SecretsStore,
    load_api_key_file,
    requires_api_key,
)

# NEW_SETUP_DEFAULTS is a newer addition to config.py; resolved through the
# module object with a fallback (rather than imported directly) so a
# workspace overlay of just config.py, predating it, still lets setup.py run
# -- with the same defaults a fresh setup would otherwise have written.
NEW_SETUP_DEFAULTS = getattr(_config, "NEW_SETUP_DEFAULTS", {
    "request_timeout_seconds": 1800.0,
    "mandatory_offload": True,
    "offload_threshold_percent": 80.0,
    "auto_repair": True,
    "working_memory_fraction": 0.6,
    "chars_per_token": 3.2,
    "max_tool_output_chars": 20_000,
})
from .engine import DEFAULT_USER_AGENT, EngineError, make_engine
from .engine_stream import _STREAMING_ADAPTERS
from .discovery import (ModelDiscovery, discover_models, discover_provider_models,
                        discover_ollama_models, discover_ollama_model_details,
                        discover_llamacpp_properties, detected_context_window,
                        _openai_base_root)
from .filesystem import Paths, atomic_write_text
from .initialization import initialize_mind, render_self
from .prompts import PromptPack
from .records import Records
from .connection import verify_connection


@dataclass
class SetupOptions:
    scope: str = "all"
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_key_file: str | None = None
    api_url: str | None = None
    endpoint: str | None = None
    adapter: str | None = None
    custom_contract_file: str | None = None
    openrouter_provider: str | None = None
    self_directive: str | None = None
    self_file: str | None = None
    context_window_tokens: int | None = None
    context_window_source: str | None = None
    # None leaves the saved setting alone; "off" explicitly disables it.
    request_timeout_seconds: float | str | None = None
    # None leaves the saved setting alone (or the default, for a new setup);
    # "off" explicitly disables the watchdog.
    stall_timeout_seconds: float | str | None = None
    first_token_timeout_seconds: float | str | None = None
    # None leaves stream_responses alone: a fresh setup then defaults it to
    # True for a streaming-capable adapter; reconfiguring an existing
    # workspace keeps whatever was already saved.
    stream_responses: bool | None = None
    reasoning: str | None = None
    reasoning_effort: str | None = None
    reasoning_budget_tokens: int | None = None
    reasoning_mode: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    stop_sequences: list[str] | None = None
    reset_generation_settings: bool = False
    vision: str | None = None
    mandatory_offload: bool | None = None
    offload_threshold_percent: float | None = None
    working_memory_tokens: str | None = None
    auto_repair: bool | None = None
    force: bool = False


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def _yes(prompt: str, default: bool = False) -> bool:
    marker = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{marker}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "n", "no"}:
            return value in {"y", "yes"}
        print("Please enter yes or no, or press Enter for the default.")


def _number(prompt: str, default: object, *, minimum: float = 0,
            maximum: float | None = None, integer: bool = False) -> int | float:
    while True:
        try:
            raw = _ask(prompt, str(default)).replace(",", "").replace("_", "")
            value = int(raw) if integer else float(raw)
            if not value >= minimum or (maximum is not None and not value <= maximum):
                raise ValueError
            return value
        except ValueError:
            print(f"Enter {'a whole number' if integer else 'a number'} from {minimum:g}"
                  + (f" to {maximum:g}." if maximum is not None else "."))


def _choice(prompt: str, choices: list[str], default: str) -> str:
    while True:
        value = _ask(prompt, default).lower().strip()
        if value in choices:
            return value
        print("Choose " + ", ".join(choices) + ".")


def normalize_openrouter_provider(value: str | None) -> str | None:
    """Normalize one optional OpenRouter inference-provider slug."""

    if value is None:
        return None
    provider = value.strip().lower()
    if provider in {"", "auto", "automatic", "default", "none"}:
        return None
    if not re.fullmatch(r"[a-z0-9][a-z0-9._/-]*", provider):
        raise ValueError(
            "OpenRouter inference provider must be a provider slug such as "
            "'cerebras' or 'deepinfra/turbo'"
        )
    return provider


def configured_openrouter_provider(request_options: dict[str, object]) -> str | None:
    """Return a single configured OpenRouter-only provider, when present."""

    preferences = request_options.get("provider")
    if not isinstance(preferences, dict):
        return None
    only = preferences.get("only")
    if (
        isinstance(only, list)
        and len(only) == 1
        and isinstance(only[0], str)
    ):
        return only[0]
    return None


def route_openrouter_requests(
    request_options: dict[str, object],
    provider: str | None,
) -> dict[str, object]:
    """Set or clear the simple OpenRouter provider-only routing preference."""

    result = copy.deepcopy(request_options)
    preferences = result.get("provider")
    if preferences is None:
        preferences = {}
    elif not isinstance(preferences, dict):
        raise ValueError("request_options.provider must be a JSON object")
    else:
        preferences = copy.deepcopy(preferences)
    selected = normalize_openrouter_provider(provider)
    if selected is None:
        preferences.pop("only", None)
    else:
        preferences["only"] = [selected]
    if preferences:
        result["provider"] = preferences
    else:
        result.pop("provider", None)
    return result


def load_custom_contract(value: str) -> dict[str, object]:
    """Load one advanced custom JSON-over-HTTP contract from disk."""

    if value.strip().lower() in {"none", "off", "openai", "openai-compatible"}:
        return {}
    path = Path(value).expanduser()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("custom contract file must contain one JSON object")
    return parsed


def custom_contract_url_parts(contract: dict[str, object]) -> tuple[str, str]:
    """Validate and split the contract's complete request URL for Config."""

    url = str(contract.get("url") or "").strip()
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("custom contract requires an absolute HTTP(S) `url`")
    if parsed.fragment:
        raise ValueError("custom contract URL cannot contain a fragment")
    base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    endpoint = parsed.path.lstrip("/")
    if parsed.query:
        endpoint += "?" + parsed.query
    return base.rstrip("/"), endpoint


def custom_contract_requires_api_key(contract: dict[str, object]) -> bool:
    """Return whether an advanced custom contract explicitly requires a key."""

    auth = contract.get("auth", {})
    return isinstance(auth, dict) and auth.get("required") is True


PROVIDER_ORDER = (
    ("openrouter", "OpenRouter"),
    ("openai", "OpenAI"),
    ("ollama", "Ollama (local or remote)"),
    ("vllm", "vLLM (local or remote)"),
    ("llamacpp", "llama.cpp / llama-server (local or remote)"),
    ("gemini", "Google Gemini"),
    ("anthropic", "Anthropic"),
    ("custom", "Other OpenAI-compatible endpoint"),
)



def choose_provider(default: str = "openrouter") -> str:
    """Present one stable provider menu and accept a number or provider name."""

    choices = {str(index): name for index, (name, _) in enumerate(PROVIDER_ORDER, 1)}
    names = {name for name, _ in PROVIDER_ORDER}
    default_number = next(
        (number for number, name in choices.items() if name == default),
        "1",
    )
    print("\nModel provider")
    for index, (_, label) in enumerate(PROVIDER_ORDER, 1):
        print(f"  {index}. {label}")
    while True:
        value = _ask("Choose provider", default_number).lower()
        provider = choices.get(value, value)
        if provider in names:
            return provider
        print("Choose a number from the menu or enter a provider name.")


def normalize_openai_endpoint(
    value: str,
    *,
    provider: str = "custom",
) -> tuple[str, str]:
    """Accept a server root, `/v1` base, or full chat-completions URL.

    The Ollama compatibility branch is retained for callers of this helper;
    normal provider setup uses `normalize_provider_endpoint` and native
    `/api/chat` instead.
    """

    parsed = _url_parts(value)

    path = parsed.path.rstrip("/")
    endpoint = "chat/completions"
    suffix = "/chat/completions"
    if path.endswith(suffix):
        path = path[: -len(suffix)].rstrip("/")
    elif provider == "ollama" and path == "/api":
        path = "/v1"
    elif not path:
        path = "/v1"
    base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return base.rstrip("/"), endpoint


def _url_parts(value: str):
    raw = value.strip()
    if not raw:
        raise ValueError("Enter the model server's address.")
    if "://" not in raw:
        candidate = urllib.parse.urlsplit("//" + raw)
        hostname = candidate.hostname or ""
        try:
            local = ipaddress.ip_address(hostname).is_private
        except ValueError:
            local = hostname == "localhost" or "." not in hostname or hostname.endswith(".local")
        raw = ("http://" if local else "https://") + raw
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or any(x.isspace() for x in parsed.netloc):
        raise ValueError("Enter a valid HTTP or HTTPS server address.")
    if parsed.username or parsed.password:
        raise ValueError("Enter an address without embedded credentials; use the API-key prompt.")
    if parsed.query or parsed.fragment:
        raise ValueError("API URL cannot contain a query string or fragment. Custom JSON contracts support query-based APIs.")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("The server port must be between 1 and 65535.")
    return parsed


def normalize_provider_endpoint(value: str, *, provider: str) -> tuple[str, str, str]:
    """Normalize one pasted server URL into its provider-native request route."""

    if provider in {"openai", "openrouter", "anthropic", "gemini"}:
        preset = PROVIDERS[provider]
        parsed = _url_parts(value)
        path = parsed.path.rstrip("/")
        suffix = "/" + preset.endpoint
        if provider == "gemini" and "/models/" in path and path.endswith(":generateContent"):
            path = path[:path.rfind("/models/")]
        elif path.endswith(suffix):
            path = path[:-len(suffix)]
        elif not path:
            path = urllib.parse.urlsplit(preset.base_url).path
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/"), preset.endpoint, preset.adapter
    if provider != "ollama":
        base, endpoint = normalize_openai_endpoint(value, provider=provider)
        adapter = provider if provider in {"vllm", "llamacpp"} else "openai_compatible"
        return base, endpoint, adapter

    parsed = _url_parts(value)
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/chat/completions", "/api/chat", "/v1", "/api"):
        if path == suffix or path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return base.rstrip("/"), "api/chat", "ollama"


def resolved_vision(provider: str, requested: str | None, details: dict[str, object]) -> str:
    """Auto uses explicit metadata; unverified local/custom vision is opt-in."""
    if requested not in {None, "auto", "yes", "no"}:
        raise ValueError("vision must be auto, yes, or no")
    if requested == "no" or details.get("vision") is False:
        return "no"
    if requested == "yes":
        return "yes"
    if details.get("vision") is True:
        return "auto"
    return "no" if provider in {"llamacpp", "ollama", "vllm", "custom"} else "auto"


def validate_discovered_settings(
    provider: str,
    discovery: ModelDiscovery,
    model: str,
    *,
    context_window_tokens: int,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
    reasoning_budget_tokens: int | None,
) -> None:
    details = discovery.details.get(model) or {}
    context = details.get("context_length")
    if isinstance(context, (int, float)) and context_window_tokens > int(context):
        raise ValueError(
            f"configured context window {context_window_tokens:,} exceeds the "
            f"{provider} server/model limit of {int(context):,} tokens"
        )
    output = details.get("max_output_tokens")
    if (
        max_output_tokens is not None
        and isinstance(output, (int, float))
        and max_output_tokens > int(output)
    ):
        raise ValueError(
            f"configured maximum output {max_output_tokens:,} exceeds the model "
            f"limit of {int(output):,} tokens"
        )
    reasoning = details.get("reasoning")
    if provider == "openrouter" and isinstance(reasoning, dict):
        efforts = reasoning.get("supported_efforts")
        if (
            reasoning_effort not in {None, "on"}
            and isinstance(efforts, list)
            and reasoning_effort not in efforts
        ):
            raise ValueError(
                f"OpenRouter reports that {model} supports reasoning efforts "
                f"{', '.join(str(item) for item in efforts)}, not {reasoning_effort}"
            )
        if reasoning_effort == "none" and reasoning.get("mandatory") is True:
            raise ValueError(f"OpenRouter reports that reasoning is mandatory for {model}")
        if reasoning_budget_tokens is not None and reasoning.get("supports_max_tokens") is False:
            raise ValueError(
                f"OpenRouter reports that {model} does not support an exact reasoning budget"
            )


def choose_model(models: tuple[str, ...], *, default: str = "") -> str:
    if not models:
        while True:
            value = _ask("Model identifier", default)
            if value:
                return value
            print("Enter the model ID shown by your provider or server.")
    if len(models) == 1:
        print(f"Detected model: {models[0]}")
        return models[0]
    choices = models
    while True:
        if len(choices) <= 20:
            for index, model in enumerate(choices, 1):
                print(f"  {index}. {model}" + (" (current)" if model == default else ""))
        else:
            print(f"{len(choices)} models available. Type an ID or search term.")
        selected = _ask("Model (number, ID, or search; * shows all)", default)
        if selected in models:
            return selected
        if selected.isdigit() and len(choices) <= 20 and 1 <= int(selected) <= len(choices):
            return choices[int(selected) - 1]
        if selected == "*":
            choices = models
        else:
            matches = tuple(model for model in models if selected and selected.lower() in model.lower())
            if len(matches) == 1:
                print(f"Selected model: {matches[0]}")
                return matches[0]
            if matches:
                choices = matches
            else:
                print("No matching model. Try another name or search term.")
        default = ""


def supported_generation_controls(provider: str, model: str) -> set[str]:
    """Return only controls serialized by the selected provider contract.

    This governs the friendly interactive UI. Explicit CLI values still pass
    through the engine's stricter validation so model/server-specific failures
    are reported rather than silently discarded.
    """

    common = {"temperature", "max_output_tokens", "top_p"}
    controls: dict[str, set[str]] = {
        "openrouter": common
        | {
            "reasoning_effort", "reasoning_budget_tokens", "top_k", "min_p",
            "frequency_penalty", "presence_penalty", "repetition_penalty", "seed",
            "stop_sequences",
        },
        "openai": common,
        "ollama": common
        | {
            "reasoning_effort", "top_k", "min_p", "repetition_penalty", "seed",
            "stop_sequences",
        },
        "vllm": common
        | {
            "reasoning_effort", "reasoning_budget_tokens", "top_k", "min_p",
            "frequency_penalty", "presence_penalty", "repetition_penalty", "seed",
            "stop_sequences",
        },
        "gemini": common
        | {
            "reasoning_effort", "top_k", "frequency_penalty", "presence_penalty",
            "seed", "stop_sequences",
        },
        "anthropic": common
        | {"reasoning_effort", "reasoning_budget_tokens", "top_k", "stop_sequences"},
        # Expose extensions as optional, explicitly server-defined controls.
        # Hidden settings prevented custom-server users from configuring them.
        "custom": common
        | {
            "reasoning_effort", "top_k", "min_p", "repetition_penalty",
            "frequency_penalty", "presence_penalty", "seed", "stop_sequences",
        },
        "llamacpp": common | {
            "reasoning_effort", "top_k", "min_p", "repetition_penalty",
            "frequency_penalty", "presence_penalty", "seed", "stop_sequences",
        },
    }
    selected = set(controls.get(provider, controls["custom"]))
    model_lower = model.lower()
    if provider == "openai" and re.match(r"^(?:gpt-5|o\d(?:-|$))", model_lower):
        selected.add("reasoning_effort")
    if provider in {"openrouter", "openai"} and "gpt-5.6" in model_lower:
        selected.add("reasoning_mode")
    if provider == "openrouter" and "gpt-oss" in model_lower:
        # GPT-OSS exposes the discrete low/medium/high effort control. Neither
        # OpenRouter's model metadata nor the Cerebras contract advertises an
        # exact reasoning-token budget for this model.
        selected.discard("reasoning_budget_tokens")
    if provider == "gemini":
        # Native generateContent uses exact thinking budgets for Gemini 2.5 and
        # thinking levels for Gemini 3+. Do not present both as universal knobs.
        if model_lower.startswith("gemini-2.5"):
            selected.add("reasoning_budget_tokens")
        else:
            selected.discard("reasoning_budget_tokens")
    if provider == "anthropic" and (
        re.search(r"claude-(?:sonnet|opus|haiku)-5(?:-|$)", model_lower)
        or re.search(r"claude-opus-4-(?:7|8)(?:-|$)", model_lower)
    ):
        # These generations use adaptive thinking; manual budgets and sampling
        # controls are rejected rather than merely ineffective.
        selected.difference_update(
            {"reasoning_budget_tokens", "temperature", "top_p", "top_k"}
        )
    return selected


def reasoning_choices(provider: str, model: str, details: dict | None = None) -> list[str]:
    details = details or {}
    controls = supported_generation_controls(provider, model)
    if details.get("thinking") is False:
        return ["auto"]
    reasoning = details.get("reasoning") or {}
    efforts = reasoning.get("supported_efforts") if isinstance(reasoning, dict) else None
    if isinstance(efforts, list):
        values = ["auto", *("off" if x == "none" else str(x) for x in efforts)]
        if reasoning.get("supports_max_tokens"):
            values.append("budget")
        return values
    if provider == "llamacpp":
        values = ["auto"]
        if details.get("thinking_toggle") is not False:
            values += ["off", "on"]
        if details.get("effort_control") is not False:
            values += ["low", "medium", "high"]
        return values
    if provider == "ollama":
        return ["auto", "low", "medium", "high"] if "gpt-oss" in model.lower() else ["auto", "off", "on"]
    if provider == "openrouter" and isinstance(details.get("supported_parameters"), list):
        if not any(x in details["supported_parameters"] for x in ("reasoning", "reasoning.effort", "include_reasoning")):
            return ["auto"]
    values = ["auto"]
    if "reasoning_effort" in controls:
        if provider == "gemini":
            values += ["off"] if model.lower().startswith("gemini-2.5") else ["minimal", "low", "medium", "high"]
        elif provider == "anthropic":
            values += ["off", "low", "medium", "high", "max"]
            if "adaptive" in details.get("thinking_types", []):
                values.append("on")
        else:
            values += ["off", "low", "medium", "high"]
    if "reasoning_budget_tokens" in controls and reasoning.get("supports_max_tokens") is not False:
        values.append("budget")
    return values


def reasoning_label(config: Config) -> str:
    if config.reasoning_budget_tokens is not None:
        return f"budget of {config.reasoning_budget_tokens:,} tokens"
    if config.reasoning_effort == "none":
        return "off"
    return config.reasoning_effort or "server default"


def configure_reasoning(options: SetupOptions, config: Config, details: dict | None = None) -> None:
    provider = "llamacpp" if config.adapter == "llamacpp" else config.provider
    choices = reasoning_choices(provider, config.model, details)
    if choices == ["auto"]:
        options.reasoning = "auto"
        options.reasoning_effort = options.reasoning_budget_tokens = options.reasoning_mode = None
        print("Reasoning: server default (no adjustable control advertised).")
        return
    default = "budget" if config.reasoning_budget_tokens else "off" if config.reasoning_effort == "none" else config.reasoning_effort or "auto"
    if default not in choices:
        choices.append(default)
    print("Reasoning controls how much the model thinks before answering. Auto keeps the server default.")
    if provider == "llamacpp" and not details:
        print("This server did not report reasoning controls; selected overrides will be checked with a real request.")
    value = _choice("Reasoning (" + "/".join(choices) + ")", choices, default)
    options.reasoning = value
    options.reasoning_effort = None
    options.reasoning_budget_tokens = None
    options.reasoning_mode = None
    if value == "budget":
        options.reasoning = "auto"
        options.reasoning_budget_tokens = int(_number("Reasoning budget in tokens", config.reasoning_budget_tokens or 2048, minimum=1, integer=True))


def configure_generation_interactive(options: SetupOptions, *, current: Config | None = None,
                                     provider: str | None = None, model: str | None = None) -> None:
    """An optional editor: ask only for the one control the operator selects."""
    provider = provider or options.provider or (current.provider if current else "custom")
    model = model or options.model or (current.model if current else "")
    supported = supported_generation_controls(provider, model)
    fields = [name for name in GENERATION_FIELDS if name in supported and not name.startswith("reasoning")]
    if "qwen" in model.lower() and "temperature" in fields:
        print("Tip: Qwen models behave best around temperature 0.3.")
    while True:
        print("\nSampling and output (unset values use server defaults)")
        for i, name in enumerate(fields, 1):
            value = getattr(options, name)
            if value is None and current and not options.reset_generation_settings:
                value = getattr(current, name)
            print(f"  {i}. {name.replace('_', ' ').capitalize()}: {value if value not in (None, []) else 'server default'}")
        choice = _ask("Setting number, reset, or done", "done").lower()
        if choice in {"done", "back"}:
            return
        if choice == "reset":
            options.reset_generation_settings = True
            for name in GENERATION_FIELDS:
                setattr(options, name, [] if name == "stop_sequences" else None)
            options.reasoning = "auto"
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(fields):
            print("Choose a setting number, reset, or done.")
            continue
        name = fields[int(choice) - 1]
        default = getattr(options, name)
        if default is None and current:
            default = getattr(current, name)
        try:
            raw = _ask(name.replace('_', ' ').capitalize() + " (or default)", str(default) if default is not None else "default")
            if raw.lower() in {"default", "provider default", "none"}:
                # Preserve every other current value while clearing this field.
                if current and not options.reset_generation_settings:
                    for field in GENERATION_FIELDS:
                        if field.startswith("reasoning") and options.reasoning is not None:
                            continue
                        if getattr(options, field) is None:
                            setattr(options, field, getattr(current, field))
                    options.reset_generation_settings = True
                value = [] if name == "stop_sequences" else None
            elif name == "stop_sequences":
                value = json.loads(raw)
                if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
                    raise ValueError("Use a JSON list of non-empty stop strings.")
            else:
                cast = int if name in {"max_output_tokens", "top_k", "seed"} else float
                value = cast(raw.replace(",", "").replace("_", ""))
            # Config validates ranges immediately, keeping the editor open on typos.
            dataclasses.replace(current or Config(), **{name: value})
            setattr(options, name, value)
        except (ValueError, TypeError) as exc:
            print(f"Invalid value: {exc}")


class SetupWizard:
    """One connection resolver and editor for initial setup and reconfiguration."""

    DISCOVERABLE = {*PROVIDERS, "custom"}
    AUTO_WORKING_MEMORY_FRACTION = NEW_SETUP_DEFAULTS["working_memory_fraction"]
    LOCAL = {"ollama", "vllm", "llamacpp", "custom"}

    @staticmethod
    def _harness_values(options: SetupOptions, current: Config | None) -> dict[str, object]:
        values: dict[str, object] = {}
        for name in ("mandatory_offload", "offload_threshold_percent", "auto_repair"):
            value = getattr(options, name)
            if value is not None:
                values[name] = value
        if options.working_memory_tokens is not None:
            raw = str(options.working_memory_tokens).strip().lower()
            values["working_memory_tokens"] = None if raw in {"same", "auto"} else int(raw)
            values["working_memory_fraction"] = (
                SetupWizard.AUTO_WORKING_MEMORY_FRACTION if raw == "auto" else None
            )
        if current is None:
            for name, default in NEW_SETUP_DEFAULTS.items():
                if name == "request_timeout_seconds" or name in values:
                    continue
                if name == "working_memory_fraction" and options.working_memory_tokens is not None:
                    continue
                values[name] = default
        values["vision_preference"] = options.vision if options.vision is not None else (
            current.vision_preference if current else "auto")
        return values

    @staticmethod
    def _edit_harness(options: SetupOptions, current: Config | None = None) -> None:
        print("\nHarness settings")
        if options.vision is None:
            options.vision = _choice("Vision (auto detects image support; yes requires it; no disables it)", ["auto", "yes", "no"], current.vision_preference if current else "auto")
        if options.working_memory_tokens is None:
            if current is None or current.working_memory_fraction:
                default = "auto"
            elif current.working_memory_tokens:
                default = str(current.working_memory_tokens)
            else:
                default = "same"
            percent = f"{SetupWizard.AUTO_WORKING_MEMORY_FRACTION:.0%}"
            while True:
                raw = _ask(f"Working-memory target: auto ({percent} of model context), tokens, or same as model context", default).lower()
                if raw in {"auto", "same"} or (raw.isdecimal() and int(raw) >= 4000):
                    options.working_memory_tokens = raw
                    break
                print("Enter auto, same, or at least 4000 tokens.")
        if options.mandatory_offload is None:
            options.mandatory_offload = _yes("Require offloading at a working-memory threshold", current.mandatory_offload if current else True)
        if options.mandatory_offload and options.offload_threshold_percent is None:
            options.offload_threshold_percent = _number("Offload threshold percent", current.offload_threshold_percent if current else 80, minimum=1, maximum=95)
        if options.auto_repair is None:
            options.auto_repair = _yes("Try automatic repair after a rejected input or empty answer (up to 3 attempts)", current.auto_repair if current else True)

    @staticmethod
    def _validate_scope(options: SetupOptions) -> None:
        if options.scope == "model" and any(
            getattr(options, name) is not None for name in ("vision", "mandatory_offload", "offload_threshold_percent", "working_memory_tokens", "auto_repair")
        ):
            raise ValueError("Use configure harness for vision and offloading settings")
        if options.scope == "harness" and (options.reset_generation_settings or any(
            getattr(options, name) is not None for name in (
                "provider", "model", "api_url", "api_key", "api_key_file", "endpoint", "adapter",
                "custom_contract_file", "openrouter_provider", "context_window_tokens", "request_timeout_seconds",
                "stall_timeout_seconds", "first_token_timeout_seconds", "stream_responses", "reasoning", *GENERATION_FIELDS,
            )
        )):
            raise ValueError("Use configure model for connection and generation settings")

    def __init__(self, paths: Paths):
        self.paths = paths
        self.store = ConfigStore(paths)
        self.prompts = PromptPack(paths)
        self._discoveries: dict[tuple, ModelDiscovery] = {}
        self._properties: dict[tuple, dict[str, object]] = {}

    @staticmethod
    def _text(value: str | None, file: str | None, default: str) -> str:
        if file:
            return Path(file).expanduser().read_text(encoding="utf-8")
        if value and value.startswith("@"):
            return Path(value[1:]).expanduser().read_text(encoding="utf-8")
        return value if value and value.strip() else default

    @staticmethod
    def _key(options: SetupOptions) -> str:
        if options.api_key_file:
            return load_api_key_file(options.api_key_file)
        return (options.api_key or "").strip()

    def _credential(self, options: SetupOptions, provider: str, current: Config | None,
                    base: str) -> str:
        explicit = self._key(options)
        if explicit:
            return explicit
        if current is not None and current.provider == provider and current.base_url == base:
            return SecretsStore(self.paths).resolve_api_key(current) or ""
        preset = PROVIDERS.get(provider)
        environment = os.getenv("ARTIFICIUM_API_KEY") or (
            os.getenv(preset.key_environment) if preset else None
        )
        if environment:
            return environment.strip()
        return ""

    def _probe(self, provider: str, base: str, key: str, headers: dict | None = None) -> ModelDiscovery:
        identity = (provider, base, key, json.dumps(headers or {}, sort_keys=True))
        if identity not in self._discoveries:
            self._discoveries[identity] = discover_provider_models(
                provider, base, api_key=key or None, timeout=10.0, headers=headers,
            )
        return self._discoveries[identity]

    def _details(self, provider: str, base: str, model: str, key: str,
                 discovery: ModelDiscovery, headers: dict | None = None) -> dict[str, object]:
        identity = (provider, base, model, key, json.dumps(headers or {}, sort_keys=True))
        if identity not in self._properties:
            details = dict(discovery.details.get(model) or {})
            is_llama = provider == "llamacpp" or "llama" in str(details.get("owned_by", "")).lower()
            if is_llama:
                details.update(discover_llamacpp_properties(base, model, api_key=key or None, headers=headers))
                details["llamacpp"] = True
            elif provider == "ollama":
                props = discover_ollama_model_details(base, model, api_key=key or None, headers=headers)
                details.update(props)
                # Ollama accepts num_ctx in each request. A currently loaded
                # 4k slot is not a ceiling on a new allocation.
                if props.get("model_context_length"):
                    details["context_length"] = props["model_context_length"]
            self._properties[identity] = details
        return self._properties[identity]

    def _connection(self, options: SetupOptions, current: Config | None = None) -> tuple:
        provider = (options.provider or (current.provider if current else None)
                    or ("custom" if options.api_url or options.custom_contract_file else "")).lower()
        if provider not in {*PROVIDERS, "custom"}:
            raise ValueError("Choose --provider from: " + ", ".join(name for name, _ in PROVIDER_ORDER))
        same = current is not None and current.provider == provider
        preset = PROVIDERS.get(provider)
        contract = (load_custom_contract(options.custom_contract_file)
                    if options.custom_contract_file is not None
                    else copy.deepcopy(current.custom_contract) if same else {})
        if contract and provider != "custom":
            raise ValueError("--custom-contract can only be used with --provider custom")
        base = options.api_url or (current.base_url if same else preset.base_url if preset else "")
        endpoint = options.endpoint or (current.endpoint if same else preset.endpoint if preset else "chat/completions")
        adapter = options.adapter or (current.adapter if same else preset.adapter if preset else "openai_compatible")
        if contract:
            if options.api_url or options.endpoint:
                raise ValueError("the custom contract defines its URL; do not also pass --api-url or --endpoint")
            if options.adapter not in {None, "custom_json"}:
                raise ValueError("an advanced custom contract uses the custom_json adapter")
            base, endpoint = custom_contract_url_parts(contract)
            adapter = "custom_json"
        else:
            if options.custom_contract_file is not None and adapter == "custom_json":
                adapter = "openai_compatible"
            if provider in self.LOCAL or options.api_url:
                base, derived_endpoint, derived_adapter = normalize_provider_endpoint(base, provider=provider)
                keep_contract = same and base == current.base_url and options.custom_contract_file is None
                endpoint = options.endpoint or (current.endpoint if keep_contract else derived_endpoint)
                adapter = options.adapter or (current.adapter if keep_contract else derived_adapter)
            if preset and adapter != preset.adapter:
                raise ValueError(f"provider {provider} uses the {preset.adapter} adapter; use custom for another contract")
        return provider, base, endpoint, adapter, contract, self._credential(options, provider, current, base)

    def _build(self, options: SetupOptions, current: Config | None = None) -> tuple[Config, str]:
        if options.reasoning is not None and options.reasoning_effort is not None:
            raise ValueError("Use --reasoning or --reasoning-effort, not both")
        if options.reasoning not in {None, "auto"} and options.reasoning_budget_tokens is not None:
            raise ValueError("Choose a reasoning level or a token budget, not both")
        provider, base, endpoint, adapter, contract, key = self._connection(options, current)
        if (requires_api_key(provider, adapter) or custom_contract_requires_api_key(contract)) and not key:
            raise ValueError(f"an API key is required for provider {provider}")
        headers = current.headers if current and provider == current.provider and base == current.base_url else {}
        discovery = self._probe(provider, base, key, headers) if provider in self.DISCOVERABLE and not contract else ModelDiscovery()
        if discovery.authentication_required and not key:
            raise ValueError("The server requires an API key; use --api-key-file or interactive setup.")
        same_server = current is not None and provider == current.provider and base == current.base_url
        model = options.model or (current.model if same_server else None)
        if not model and len(discovery.models) == 1:
            model = discovery.models[0]
        if not model and current and provider == current.provider and not discovery.models:
            model = current.model
        if not model:
            raise ValueError("Specify --model. Served models: " + (", ".join(discovery.models) or "unavailable; check the server URL"))
        if provider == "gemini":
            model = model.removeprefix("models/")
        if discovery.models and model not in discovery.models and not (
            provider == "ollama" and model + ":latest" in discovery.models
        ):
            raise ValueError(f"Model {model!r} is not served by this endpoint. Choose: " + ", ".join(discovery.models))
        details = self._details(provider, base, model, key, discovery, headers) if not contract else {}
        if provider == "custom" and details.get("llamacpp") and not options.adapter:
            adapter = "llamacpp"
        values = dataclasses.asdict(current) if current else {}
        changed = current is None or (provider, model, base, endpoint, adapter, contract) != (
            current.provider, current.model, current.base_url, current.endpoint, current.adapter, current.custom_contract)
        if current and changed:
            for name in GENERATION_FIELDS:
                values[name] = [] if name == "stop_sequences" else None
        if current and (provider != current.provider or base != current.base_url):
            values["request_options"] = {}
            values["headers"] = {}
        values.update(provider=provider, model=model, base_url=base, endpoint=endpoint,
                      adapter=adapter, custom_contract=contract)
        if options.reset_generation_settings:
            for name in GENERATION_FIELDS:
                values[name] = [] if name == "stop_sequences" else None
        if options.reasoning is not None:
            if options.reasoning not in {"auto", "off", "on", "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
                raise ValueError("Reasoning must be auto, off, on, or a supported effort level")
            values.update(reasoning_effort=None, reasoning_budget_tokens=None, reasoning_mode=None)
            if options.reasoning != "auto":
                values["reasoning_effort"] = "none" if options.reasoning == "off" else options.reasoning
        for name in GENERATION_FIELDS:
            value = getattr(options, name)
            if value is not None:
                values[name] = value
        values.update(self._harness_values(options, current))
        context = options.context_window_tokens
        source = options.context_window_source or ("manual" if context is not None else None)
        if context is None and source is None and not changed and current and current.context_window_source == "manual":
            advertised = details.get("context_length")
            if not advertised or current.context_window_tokens <= int(advertised):
                context = current.context_window_tokens
                source = "manual"
        if context is None:
            context = details.get("context_length")
            source = "detected" if context else None
            if provider == "ollama":
                context = min(int(context), 32768) if context else 32768
                source = "default"
        if context is None and not changed and current:
            context = current.context_window_tokens
            source = current.context_window_source
        if context is None:
            context, source = 32768, "default"
        values["context_window_tokens"] = int(context)
        values["context_window_source"] = source or "default"
        values["model_capabilities"] = details
        if options.request_timeout_seconds is not None:
            values["request_timeout_seconds"] = options.request_timeout_seconds
        elif current is None:
            values["request_timeout_seconds"] = NEW_SETUP_DEFAULTS["request_timeout_seconds"]
        if options.stall_timeout_seconds is not None:
            values["stall_timeout_seconds"] = options.stall_timeout_seconds
        if options.first_token_timeout_seconds is not None:
            values["first_token_timeout_seconds"] = options.first_token_timeout_seconds
        if options.stream_responses is not None:
            values["stream_responses"] = options.stream_responses
        # A reconnect rechecks capability even when a server keeps the same
        # alias after replacing its model. A prior image rejection is not a
        # permanent verdict on the replacement.
        values["model_supports_vision"] = details.get("vision")
        values["vision"] = resolved_vision(provider, values["vision_preference"], details)
        augmented = dataclasses.replace(discovery, details={**discovery.details, model: details})
        validate_discovered_settings(
            provider, augmented, model, context_window_tokens=int(context),
            max_output_tokens=values.get("max_output_tokens"),
            reasoning_effort=values.get("reasoning_effort"),
            reasoning_budget_tokens=values.get("reasoning_budget_tokens"),
        )
        if options.openrouter_provider is not None:
            if provider != "openrouter":
                raise ValueError("--openrouter-provider can only be used with --provider openrouter")
            values["request_options"] = route_openrouter_requests(
                values.get("request_options") or {}, options.openrouter_provider)
        config = Config(**values)
        make_engine(config, key or None).request_summary()  # Validate before any write.
        return config, key

    def interactive(self, options: SetupOptions | None = None, *, current: Config | None = None):
        from .setup_ui import interactive_setup
        return interactive_setup(self, options, current=current)

    @staticmethod
    def _clear_generation(options: SetupOptions) -> None:
        options.reset_generation_settings = True
        options.reasoning = "auto"
        for name in GENERATION_FIELDS:
            setattr(options, name, [] if name == "stop_sequences" else None)

    def _directive(self, options: SetupOptions) -> str | None:
        if options.self_directive is not None or options.self_file is not None:
            return self._text(options.self_directive, options.self_file, "")
        return None

    def _verify(self, options: SetupOptions, config: Config, key: str) -> Config:
        verified, result = verify_connection(self.paths, config, key or None,
                                             self_directive=self._directive(options))
        self.last_check = result
        return verified

    def _save_connection(self, config: Config, key: str, options: SetupOptions,
                         current: Config | None = None) -> None:
        previous = self.paths.secrets.read_bytes() if self.paths.secrets.is_file() else None
        changed_key = bool(key and (self._key(options) or current is None or
                                   (config.provider, config.base_url) != (current.provider, current.base_url)))
        try:
            if changed_key:
                SecretsStore(self.paths).save_api_key(key, provider=config.provider, base_url=config.base_url,
                                                      preferred=bool(self._key(options)))
            self.store.save(config)
        except OSError:
            if changed_key:
                if previous is None:
                    self.paths.secrets.unlink(missing_ok=True)
                else:
                    atomic_write_text(self.paths.secrets, previous.decode("utf-8"), mode=0o600)
            raise

    def run(self, options: SetupOptions, *, interactive: bool = False) -> dict[str, str]:
        if self.store.exists() and not options.force:
            raise FileExistsError("Already configured. Use configure to change the connection, or setup --force to replace configuration.")
        # Seed before anything reads the mind: a connection check previews
        # mind/self.txt, and on a fresh install the workspace does not exist yet.
        self.paths.ensure_layout()
        if interactive:
            config, key, options = self.interactive(options)
        else:
            config, key = self._build(options)
            config = self._verify(options, config, key)
        if options.stream_responses is None and config.adapter in _STREAMING_ADAPTERS and not config.stream_responses:
            # A brand-new setup defaults to streaming once the connection is
            # verified, without re-verifying it: the check above already used
            # the plain JSON transport, and a server that turns out not to
            # speak SSE is caught automatically at runtime (HTTPMixin._post
            # falls back to non-streaming and remembers that for the
            # session). Reconfiguring an existing workspace never reaches
            # this method, so a saved stream_responses=False is never touched.
            config = dataclasses.replace(config, stream_responses=True)
        directive = self._directive(options)
        if directive is None:
            directive = self.paths.self_file.read_text(encoding="utf-8")
        self_existed = self.paths.self_file.is_file()
        self._save_connection(config, key, options)
        records = Records(self.paths)
        initialize_mind(self.paths, records, self_directive=directive)
        if options.self_directive is not None or options.self_file is not None or not self_existed:
            atomic_write_text(self.paths.self_file, render_self(directive))
        records.emit("setup_completed", provider=config.provider, model=config.model,
                     self_path=str(self.paths.self_file), prompt_pack=self.prompts.version,
                     connection_check=self.last_check)
        return {"name": "Artificium", "root": str(self.paths.root), "mind": str(self.paths.mind),
                "self": str(self.paths.self_file), "logs": str(self.paths.logs),
                "provider": config.provider, "model": config.model,
                "context_window_tokens": str(config.context_window_tokens),
                "reasoning_effort": reasoning_label(config), "vision": config.vision}

    def reconfigure(self, options: SetupOptions, *, interactive: bool = False) -> Config:
        self.paths.ensure_layout()
        self._validate_scope(options)
        current = self.store.load()
        if options.scope == "harness":
            # Harness edits work offline, without credentials or model discovery.
            if interactive:
                self._edit_harness(options, current)
            values = self._harness_values(options, current)
            if options.vision is not None:
                values["vision"] = resolved_vision(current.provider, options.vision,
                                                  {"vision": current.model_supports_vision})
            updated = dataclasses.replace(current, **values)
            if interactive and not _yes("Save harness settings", True):
                print("No settings changed.")
                return current
            self.store.save(updated)
            Records(self.paths).emit("harness_configuration_updated", **values)
            return updated
        if interactive:
            updated, key, options = self.interactive(options, current=current)
        else:
            updated, key = self._build(options, current)
            updated = self._verify(options, updated, key)
        self._save_connection(updated, key, options, current)
        Records(self.paths).emit("configuration_updated", provider=updated.provider, model=updated.model,
                                context_window_tokens=updated.context_window_tokens,
                                vision=updated.vision,
                                connection_check=self.last_check)
        return updated
