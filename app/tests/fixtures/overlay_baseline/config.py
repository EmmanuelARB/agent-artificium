from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .filesystem import Paths, atomic_write_json, read_json


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str
    endpoint: str
    adapter: str
    key_environment: str
    requires_api_key: bool = True


PROVIDERS: dict[str, ProviderPreset] = {
    "llamacpp": ProviderPreset(
        "llamacpp", "http://127.0.0.1:8080/v1", "chat/completions",
        "llamacpp", "LLAMA_API_KEY", requires_api_key=False,
    ),
    "openrouter": ProviderPreset(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "chat/completions",
        "openrouter",
        "OPENROUTER_API_KEY",
    ),
    "openai": ProviderPreset(
        "openai",
        "https://api.openai.com/v1",
        "responses",
        "openai_responses",
        "OPENAI_API_KEY",
    ),
    "gemini": ProviderPreset(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        "models/{model}:generateContent",
        "gemini",
        "GEMINI_API_KEY",
    ),
    "anthropic": ProviderPreset(
        "anthropic",
        "https://api.anthropic.com/v1",
        "messages",
        "anthropic",
        "ANTHROPIC_API_KEY",
    ),
    "ollama": ProviderPreset(
        "ollama",
        "http://127.0.0.1:11434",
        "api/chat",
        "ollama",
        "OLLAMA_API_KEY",
        requires_api_key=False,
    ),
    "vllm": ProviderPreset(
        "vllm",
        "http://127.0.0.1:8000/v1",
        "chat/completions",
        "vllm",
        "VLLM_API_KEY",
        requires_api_key=False,
    ),
}


def requires_api_key(provider: str, adapter: str = "openai_compatible") -> bool:
    """Return whether setup must reject a missing credential.

    OpenAI-compatible custom servers are allowed to be unauthenticated because
    that is the normal local-development shape. If such a server does require a
    key, the operator can still supply one. Anthropic transport always requires
    authentication.
    """

    preset = PROVIDERS.get(provider.lower().strip())
    if preset is not None:
        return preset.requires_api_key
    return adapter in {"anthropic", "gemini", "openai_responses", "openrouter"}


REASONING_EFFORTS = ("none", "on", "minimal", "low", "medium", "high", "xhigh", "max")


GENERATION_FIELDS = (
    "reasoning_effort",
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
)

# Persist two independent sections in one atomic file; the runtime keeps its
# existing flat Config interface. Effective vision belongs to the connection.
MODEL_FIELDS = frozenset({
    "provider", "model", "base_url", "endpoint", "adapter",
    "context_window_tokens", "request_options", "headers", "custom_contract",
    "context_window_source", "request_timeout_seconds", "model_capabilities",
    "vision", "model_supports_vision", *GENERATION_FIELDS,
})


def parse_request_timeout(value: float | str | None) -> float | None:
    """None/off disables the inference timeout; a number keeps it bounded."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "off"):
        return None
    message = "Request timeout must be off or between 1 and 86,400 seconds"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if isinstance(value, bool) or not 1 <= seconds <= 86_400:
        raise ValueError(message)
    return seconds


@dataclass
class Config:
    schema_version: int = 3
    codename: str = "Artificium-revolution"
    instance_id: str = "artificium"
    provider: str = "custom"
    model: str = ""
    base_url: str = ""
    endpoint: str = "chat/completions"
    adapter: str = "openai_compatible"
    wake_on_interaction: bool = True
    poll_seconds: float = 1.0
    context_window_tokens: int = 100_000
    context_window_source: str = "manual"
    request_timeout_seconds: float | None = None
    model_capabilities: dict[str, Any] = field(default_factory=dict)
    context_reminder_tokens: int = 10_000
    context_soft_fraction: float = 0.70
    context_hard_fraction: float = 0.85
    mandatory_offload: bool = False
    offload_threshold_percent: float = 80.0
    working_memory_tokens: int | None = None
    auto_repair: bool = False
    chars_per_token: float = 4.0
    broad_chunk_fraction: float = 0.50
    balanced_chunk_fraction: float = 0.20
    granular_chunk_fraction: float = 0.05
    carry_fraction: float = 0.12
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
    stop_sequences: list[str] = field(default_factory=list)
    # Budgets for run --once / bounded run_turn calls only; continuous mode
    # ignores these legacy setting names and does not emit turn-limit notices.
    max_life_loop_rounds: int = 64
    max_turn_seconds: float = 900.0
    shell_timeout_seconds: float = 120.0
    max_shell_timeout_seconds: float = 3600.0
    max_tool_output_chars: int = 60_000
    max_direct_read_chars: int = 60_000
    notification_batch_size: int = 32
    meta_memory_guidance_tokens: int = 8_000
    vision: str = "auto"
    vision_preference: str | None = None
    model_supports_vision: bool | None = None
    max_image_bytes: int = 20_000_000
    engine_error_backoff_seconds: float = 60.0
    engine_wait_notice_seconds: float = 15.0
    engine_wait_repeat_seconds: float = 30.0
    runtime_message_role: str = "user"
    request_options: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    custom_contract: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_window_source not in {"manual", "detected", "default"}:
            raise ValueError("context_window_source must be manual, detected, or default")
        self.request_timeout_seconds = parse_request_timeout(self.request_timeout_seconds)
        if not isinstance(self.model_capabilities, dict):
            raise ValueError("model_capabilities must be an object")
        self.provider = self.provider.lower().strip()
        preset = PROVIDERS.get(self.provider)
        if preset:
            if not self.base_url:
                self.base_url = preset.base_url
            if not self.endpoint or (
                self.endpoint == "chat/completions"
                and preset.endpoint != "chat/completions"
                and self.adapter == "openai_compatible"
            ):
                self.endpoint = preset.endpoint
            if self.adapter == "openai_compatible" and preset.adapter != self.adapter:
                self.adapter = preset.adapter
        self.base_url = self.base_url.rstrip("/")
        self.endpoint = self.endpoint.lstrip("/")
        if not self.base_url:
            raise ValueError("An API base URL is required")
        if not self.model.strip():
            raise ValueError("A model identifier is required")
        adapters = {
            "llamacpp",
            "openai_compatible",
            "openrouter",
            "openai_responses",
            "gemini",
            "anthropic",
            "ollama",
            "vllm",
            "custom_json",
        }
        if self.adapter not in adapters:
            raise ValueError("adapter must be one of: " + ", ".join(sorted(adapters)))
        if self.context_window_tokens < 4_000:
            raise ValueError("context_window_tokens must be at least 4000")
        if self.working_memory_tokens is not None and (
            isinstance(self.working_memory_tokens, bool)
            or not isinstance(self.working_memory_tokens, int)
            or not 4000 <= self.working_memory_tokens <= self.context_window_tokens
        ):
            raise ValueError("working_memory_tokens must be null (same as model), or 4000 through context_window_tokens")
        if not isinstance(self.auto_repair, bool):
            raise ValueError("auto_repair must be boolean")
        if self.context_reminder_tokens < 1_000:
            raise ValueError("context_reminder_tokens must be at least 1000")
        if self.meta_memory_guidance_tokens < 1_000:
            raise ValueError("meta_memory_guidance_tokens must be at least 1000")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        fractions = (
            self.broad_chunk_fraction,
            self.balanced_chunk_fraction,
            self.granular_chunk_fraction,
            self.carry_fraction,
        )
        if not all(0.01 <= value <= 0.80 for value in fractions):
            raise ValueError("context fractions must be between 0.01 and 0.80")
        if not (
            self.granular_chunk_fraction
            <= self.balanced_chunk_fraction
            <= self.broad_chunk_fraction
        ):
            raise ValueError("chunk fractions must be granular <= balanced <= broad")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if self.engine_wait_notice_seconds <= 0:
            raise ValueError("engine_wait_notice_seconds must be positive")
        if self.engine_wait_repeat_seconds <= 0:
            raise ValueError("engine_wait_repeat_seconds must be positive")
        if self.vision not in {"auto", "yes", "no"}:
            raise ValueError("vision must be auto, yes, or no")
        if self.vision_preference is None:
            self.vision_preference = self.vision
        if self.vision_preference not in {"auto", "yes", "no"}:
            raise ValueError("vision_preference must be auto, yes, or no")
        if self.model_supports_vision is not None and not isinstance(self.model_supports_vision, bool):
            raise ValueError("model_supports_vision must be boolean or null")
        if self.vision_preference == "no" or self.model_supports_vision is False:
            self.vision = "no"
        elif self.vision_preference == "yes":
            self.vision = "yes"
        elif self.model_supports_vision is True:
            self.vision = "auto"
        if not isinstance(self.mandatory_offload, bool):
            raise ValueError("mandatory_offload must be boolean")
        if isinstance(self.offload_threshold_percent, bool) or not 1 <= self.offload_threshold_percent <= 95:
            raise ValueError("offload_threshold_percent must be between 1 and 95")
        if not 0.30 <= self.context_soft_fraction < self.context_hard_fraction <= 0.95:
            raise ValueError("context thresholds must satisfy 0.30 <= soft < hard <= 0.95")
        if self.runtime_message_role not in {"user", "system"}:
            raise ValueError("runtime_message_role must be user or system")
        if not isinstance(self.custom_contract, dict):
            raise ValueError("custom_contract must be a JSON object")
        if self.custom_contract and self.provider != "custom":
            raise ValueError("custom_contract can only be used with provider custom")
        if self.custom_contract and self.adapter != "custom_json":
            raise ValueError("custom_contract requires the custom_json adapter")
        if self.adapter == "custom_json" and not self.custom_contract:
            raise ValueError("the custom_json adapter requires a custom_contract")
        if self.reasoning_effort is not None:
            self.reasoning_effort = self.reasoning_effort.lower().strip()
            if self.reasoning_effort not in REASONING_EFFORTS:
                raise ValueError(
                    "reasoning_effort must be one of: " + ", ".join(REASONING_EFFORTS)
                )
        if self.reasoning_budget_tokens is not None and self.reasoning_budget_tokens < 1:
            raise ValueError("reasoning_budget_tokens must be positive")
        if self.reasoning_mode is not None:
            self.reasoning_mode = self.reasoning_mode.lower().strip()
            if self.reasoning_mode not in {"standard", "pro"}:
                raise ValueError("reasoning_mode must be standard or pro")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        for name in ("top_p", "min_p"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        for name in ("frequency_penalty", "presence_penalty"):
            value = getattr(self, name)
            if value is not None and not -2 <= value <= 2:
                raise ValueError(f"{name} must be between -2 and 2")
        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if not isinstance(self.stop_sequences, list) or not all(
            isinstance(item, str) and item for item in self.stop_sequences
        ):
            raise ValueError("stop_sequences must be a list of non-empty strings")

    @property
    def working_memory_limit(self) -> int:
        """An offloading target, never an increase to the server's capacity."""
        return self.working_memory_tokens or self.context_window_tokens

    def chunk_tokens(self, profile: str) -> int:
        fractions = {
            "broad": self.broad_chunk_fraction,
            "balanced": self.balanced_chunk_fraction,
            "granular": self.granular_chunk_fraction,
        }
        if profile not in fractions:
            raise ValueError("profile must be broad, balanced, or granular")
        return max(1_000, int(self.working_memory_limit * fractions[profile]))

    def carry_tokens(self, profile: str) -> int:
        return max(
            1_000,
            min(
                int(self.working_memory_limit * self.carry_fraction),
                int(self.chunk_tokens(profile) * 0.50),
            ),
        )

    def public_dict(self) -> dict[str, Any]:
        always = {"schema_version", "codename", "instance_id", "provider", "model"}
        result: dict[str, Any] = {}
        for item in dataclasses.fields(self):
            value = getattr(self, item.name)
            if item.default is not dataclasses.MISSING:
                default = item.default
            elif item.default_factory is not dataclasses.MISSING:
                default = item.default_factory()
            else:
                default = dataclasses.MISSING
            if item.name in always or value != default:
                result[item.name] = value
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Config":
        if int(value.get("schema_version", 1) or 1) >= 3:
            harness, model = value.get("harness"), value.get("model")
            if "harness" in value and not (isinstance(harness, dict) and isinstance(model, dict)):
                raise ValueError("configuration requires harness and model objects")
            if isinstance(harness, dict) and isinstance(model, dict):
                if any(key in MODEL_FIELDS or key == "schema_version" for key in harness):
                    raise ValueError("model settings must be in the model section")
                if any(key not in MODEL_FIELDS for key in model):
                    raise ValueError("harness settings must be in the harness section")
                value = {**harness, **model, "schema_version": 3}
        allowed = {item.name for item in dataclasses.fields(cls)}
        supplied = {key: item for key, item in value.items() if key in allowed}
        if "auto_repair" not in supplied and "emergency_offload" in value:
            supplied["auto_repair"] = value["emergency_offload"]
        old_schema = int(value.get("schema_version", 1) or 1)
        provider = str(value.get("provider") or "custom").lower().strip()
        if old_schema < 2 and provider in PROVIDERS:
            preset = PROVIDERS[provider]
            supplied["adapter"] = preset.adapter
            supplied["endpoint"] = preset.endpoint
            base_url = str(value.get("base_url") or preset.base_url).rstrip("/")
            if provider == "ollama" and base_url.endswith("/v1"):
                base_url = base_url[:-3].rstrip("/")
            elif provider == "gemini" and base_url.endswith("/openai"):
                base_url = base_url[:-7].rstrip("/")
            supplied["base_url"] = base_url
        supplied["schema_version"] = 3
        return cls(**supplied)

    def grouped_dict(self) -> dict[str, Any]:
        values = self.public_dict()
        values.pop("schema_version", None)
        values["request_timeout_seconds"] = self.request_timeout_seconds
        # Always show the small set of operator-facing harness policies.
        for name in ("vision_preference", "mandatory_offload", "offload_threshold_percent", "working_memory_tokens", "auto_repair"):
            values[name] = getattr(self, name)
        return {
            "schema_version": 3,
            "harness": {k: v for k, v in values.items() if k not in MODEL_FIELDS},
            "model": {k: v for k, v in values.items() if k in MODEL_FIELDS},
        }


class ConfigStore:
    def __init__(self, paths: Paths):
        self.paths = paths

    def exists(self) -> bool:
        return self.paths.config.is_file()

    def load(self) -> Config:
        value = read_json(self.paths.config)
        if not isinstance(value, dict):
            raise RuntimeError(f"Configuration is missing or invalid: {self.paths.config}")
        return Config.from_dict(value)

    def save(self, config: Config) -> None:
        self.paths.install.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.paths.config, config.grouped_dict())


class SecretsStore:
    def __init__(self, paths: Paths):
        self.paths = paths

    def save_api_key(self, api_key: str, *, provider: str | None = None,
                     base_url: str | None = None, preferred: bool = True) -> None:
        key = api_key.strip()
        if not key:
            raise ValueError("API key cannot be empty")
        value = {"api_key": key, "preferred": preferred}
        if provider:
            value["provider"] = provider.lower().strip()
        if base_url:
            value["base_url"] = base_url.rstrip("/")
        atomic_write_json(self.paths.secrets, value, mode=0o600)

    def resolve_api_key(self, config: Config) -> str | None:
        value = read_json(self.paths.secrets, {})
        saved = None
        if isinstance(value, dict) and isinstance(value.get("api_key"), str):
            provider = str(value.get("provider") or "").strip().lower()
            base = str(value.get("base_url") or "").rstrip("/")
            if (not provider or provider == config.provider) and (not base or base == config.base_url):
                saved = value["api_key"].strip() or None
        # A key explicitly entered by the operator must be the one runtime uses,
        # even when an older key remains in the shell environment. Legacy files
        # without this marker retain their original environment-first behavior.
        if saved and value.get("preferred") is True:
            return saved
        variables = ["ARTIFICIUM_API_KEY"]
        preset = PROVIDERS.get(config.provider)
        if preset:
            variables.append(preset.key_environment)
        for variable in variables:
            value = os.getenv(variable)
            if value:
                return value.strip()
        return saved


def load_api_key_file(path_value: str | Path) -> str:
    """Read either a plain key file or an Artificium `.secrets.json` file."""

    text = Path(path_value).expanduser().read_text(encoding="utf-8").strip()
    if not text:
        return ""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(value, dict) and isinstance(value.get("api_key"), str):
        return value["api_key"].strip()
    return text


def configured_paths(start: str | Path | None = None) -> Paths:
    """Resolve the installation root: an explicit choice, then the environment,
    then the directory that contains the running ``app/`` tree."""

    if start is not None:
        return Paths(Path(start).expanduser().resolve())
    override = os.getenv("ARTIFICIUM_ROOT")
    if override:
        return Paths(Path(override).expanduser().resolve())
    return Paths.from_code_file(__file__)
