from __future__ import annotations

import base64
import copy
import json
import logging
import mimetypes
import time  # re-exported so `artificium.engine.time` stays a patchable name
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Config, requires_api_key
from .version import USER_AGENT
from .filesystem import json_dumps

# engine.py holds the shared primitives (errors, base classes, the JSON
# transport mixin) that both the streaming transport (engine_stream.py) and
# the provider adapters (engine_adapters.py, engine_providers.py) build on.
# Those modules import their primitives back from here, so anything they need
# at *their* import time must be defined above the `from .engine_stream/...
# import` lines below; make_engine() then imports the adapter classes for
# re-export once the full adapter modules have loaded.


DEFAULT_USER_AGENT = USER_AGENT


_UNSUPPORTED_VISION_MESSAGES = (
    "does not support image", "images are not supported",
    "image input is not supported", "image input is unsupported",
    "vision is not supported", "does not support vision",
    "does not support multimodal", "multimodal is not supported",
    "multimodal support is not enabled", "image_url is not supported",
)


class EngineError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None,
                 kind: str | None = None, hint: str | None = None,
                 reply: EngineReply | None = None):
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.hint = hint
        self.reply = reply

    @property
    def image_input_unsupported(self) -> bool:
        return any(text in str(self).lower() for text in _UNSUPPORTED_VISION_MESSAGES)

    @property
    def requires_operator_action(self) -> bool:
        # Rejected requests need correction; waiting cannot repair their contents.
        return self.status in {400, 401, 403, 404, 405, 413, 415, 422} or (
            self.status is None and self.kind in {
                "settings", "auth", "permission", "not_found", "context",
                "vision", "template", "tokenization", "tls", "repair",
            }
        )


def _server_error(detail: str, status: int | None = None) -> EngineError:
    """Retain server evidence and classify only recognizable failures."""
    try:
        value = json.loads(detail)
        error = value.get("error", value) if isinstance(value, dict) else value
        if isinstance(error, dict):
            detail = str(error.get("message") or error.get("detail") or json_dumps(error))
        elif isinstance(error, str):
            detail = error
    except (ValueError, TypeError):
        pass
    lower = detail.lower()
    kind, hint = "server", "Check the model server's error above, then retry."
    if any(x in lower for x in ("context length", "context size", "context window", "n_ctx", "too many tokens", "prompt is too long", "exceed_context_size")):
        kind, hint = "context", "The request does not fit the serving context. Increase the model server's context allocation or choose a model with more capacity; changing Artificium's number alone cannot enlarge a server."
    elif any(x in lower for x in ("failed to tokenize prompt", "number of media markers")):
        kind, hint = "tokenization", "The server rejected the prompt during tokenization. Inspect its logs and the saved request for reserved-marker collisions or malformed input; retry after correcting the request."
    elif any(x in lower for x in (*_UNSUPPORTED_VISION_MESSAGES, "multimodal projector", "mmproj", "invalid image", "failed to decode image", "could not decode image")):
        kind, hint = "vision", "This server cannot accept the image request. Use text-only input or load a vision-capable model and its image projector."
    elif status == 401:
        kind, hint = "auth", "The server rejected the API key. Enter the key for this endpoint."
    elif status == 403:
        kind, hint = "permission", "Access was refused. Check endpoint permissions or the hosting proxy; this is not necessarily an API-key problem."
    elif status == 404:
        kind, hint = "not_found", "Check the server URL and served model ID. A web UI address is not always its API address."
    elif status in {402, 429}:
        kind, hint = "quota", "Check this provider's credits or request limit, then retry."
    elif any(x in lower for x in ("template", "roles must alternate", "role alternation")):
        kind, hint = "template", "The model's chat template rejected the conversation. Check the server's chat template or choose an instruction/chat model."
    elif status in {400, 422}:
        kind, hint = "settings", "Review the setting named by the server, or reset generation settings to server defaults and retry."
    prefix = f"HTTP {status}: " if status else "Server error: "
    return EngineError(prefix + detail[:4000], status=status, kind=kind, hint=hint)


def _redact(text: str, secrets: list[str | None]) -> str:
    result = text
    for secret in secrets:
        if secret:
            result = result.replace(secret, "<redacted>")
    return result


# The synchronous JSON transport and the chat-completions SSE streaming
# transport live in engine_stream.py; import them here so
# `artificium.engine.request_json` (etc.) keeps resolving for callers and for
# tests that patch it at that path. JSONEngine/HTTPMixin below call these
# through this module's own globals, so a patch on `artificium.engine.X`
# reaches them regardless of where the function is defined.
from .engine_stream import (  # noqa: E402
    _STREAMING_ADAPTERS,
    StreamingUnsupported,
    _stream_chat_completion,
    _stream_flag,
    request_json,
)


_logger = logging.getLogger(__name__)



@dataclass
class EngineReply:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    provider_reasoning: str | None = None


@dataclass(frozen=True)
class PreparedRequest:
    adapter: str
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]
    guarantee: str = "provider contract"

    def __getitem__(self, key: str) -> Any:
        """Keep payload-style inspection compatible with earlier adapter tests."""

        return self.payload[key]

    def safe_summary(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "method": "POST",
            "url": self.url,
            "guarantee": self.guarantee,
            "header_names": sorted(
                {"Accept", "Content-Type", "User-Agent", *self.headers}
            ),
            "body": _safe_body(self.payload),
        }


class Engine(ABC):
    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        return PreparedRequest(
            adapter=type(self).__name__,
            url="in-process://custom-engine",
            headers={},
            payload={"messages": copy.deepcopy(messages)},
            guarantee="custom in-process engine",
        )

    @abstractmethod
    def complete(self, messages: list[dict[str, Any]]) -> EngineReply:
        raise NotImplementedError

    def complete_prepared(self, prepared: PreparedRequest) -> EngineReply:
        return self.complete(prepared.payload["messages"])

    def count_input_tokens(self, prepared: PreparedRequest) -> int | None:
        """Optional preflight count. None keeps the character-based fallback."""
        return None

    def request_summary(self) -> dict[str, Any]:
        return self.prepare(
            [{"role": "user", "content": "<runtime messages omitted from preview>"}]
        ).safe_summary()


def _safe_body(payload: dict[str, Any]) -> dict[str, Any]:
    prompt_keys = {"messages", "input", "contents", "system", "systemInstruction"}
    secret_keys = {
        "api_key", "apikey", "authorization", "password", "secret",
        "access_token", "x_api_key", "x_goog_api_key",
    }

    def clean(value: Any, *, key: str = "") -> Any:
        normalized = key.lower().replace("-", "_")
        if key in prompt_keys:
            return "<runtime prompt omitted>"
        if normalized in secret_keys:
            return "<redacted>"
        if isinstance(value, dict):
            return {str(k): clean(v, key=str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return copy.deepcopy(value)

    return clean(payload)


def _merge_options(payload: dict[str, Any], options: dict[str, Any]) -> None:
    """Merge expert options without silently replacing normalized settings."""

    protected = {"model", "messages", "input", "contents", "system", "systemInstruction"}

    def merge(target: dict[str, Any], source: dict[str, Any], path: str = "") -> None:
        for key, value in source.items():
            location = f"{path}.{key}" if path else key
            if key in protected:
                raise EngineError(
                    f"request_options cannot override protected request field {location!r}"
                )
            if key not in target:
                target[key] = copy.deepcopy(value)
            elif isinstance(target[key], dict) and isinstance(value, dict):
                merge(target[key], value, location)
            else:
                raise EngineError(
                    f"request_options duplicates normalized request field {location!r}; "
                    "configure it through the corresponding Artificium setting"
                )

    merge(payload, options)


def _configured(config: Config, *names: str) -> list[str]:
    return [name for name in names if getattr(config, name) not in (None, [], {})]


def _reject(config: Config, provider: str, *names: str) -> None:
    values = _configured(config, *names)
    if values:
        raise EngineError(
            f"{provider} does not define these normalized request controls: "
            + ", ".join(values)
            + ". Remove them or use request_options only when your exact endpoint "
            "documents a provider-specific equivalent."
        )


def _image(path_value: str, supplied_mime: str | None = None) -> tuple[str, str]:
    path = Path(path_value).expanduser().resolve()
    mime = supplied_mime or mimetypes.guess_type(path.name)[0] or "image/png"
    try:
        return mime, base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise EngineError(f"Cannot read image {path}: {exc}", kind="vision") from exc


def _openai_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    result: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            result.append({"type": "text", "text": str(part)})
        elif part.get("type") != "artificium_image":
            result.append({k: copy.deepcopy(v) for k, v in part.items() if not k.startswith("_")})
        else:
            mime, data = _image(str(part.get("path") or ""), part.get("mime"))
            image_url: dict[str, Any] = {"url": f"data:{mime};base64,{data}"}
            if part.get("detail") in {"low", "high", "auto"}:
                image_url["detail"] = part["detail"]
            result.append({"type": "image_url", "image_url": image_url})
    return result


def materialize_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        clean = {k: copy.deepcopy(v) for k, v in message.items() if not k.startswith("_")}
        clean["content"] = _openai_content(clean.get("content"))
        rendered.append(clean)
    return rendered


def _openai_response_content(content: Any, role: str) -> Any:
    if not isinstance(content, list):
        return content
    result: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            result.append({"type": "input_text", "text": str(part)})
        elif part.get("type") == "artificium_image":
            mime, data = _image(str(part.get("path") or ""), part.get("mime"))
            item: dict[str, Any] = {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{data}",
            }
            if part.get("detail") in {"low", "high", "auto"}:
                item["detail"] = part["detail"]
            result.append(item)
        else:
            text = str(part.get("text") or part.get("content") or "")
            result.append({
                "type": "output_text" if role == "assistant" else "input_text",
                "text": text,
            })
    return result


def _materialize_openai_response_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "role": str(message.get("role") or "user"),
            "content": _openai_response_content(
                message.get("content"), str(message.get("role") or "user")
            ),
        }
        for message in messages
    ]


class HTTPMixin:
    config: Config
    api_key: str | None
    request_attempts: int = 3

    def _redaction_secrets(self, prepared: PreparedRequest) -> list[str | None]:
        values: list[str | None] = [self.api_key]
        values.extend(
            value
            for value in (*prepared.headers.values(), *self.config.headers.values())
            if isinstance(value, str) and value
        )
        return values

    # Set once a streaming request proves the server does not speak SSE for
    # this endpoint; every later call in the same engine instance (i.e. for
    # the rest of this run) skips straight to the non-streaming transport.
    _stream_unsupported: bool = False

    def _post(self, prepared: PreparedRequest) -> dict[str, Any]:
        try:
            wants_stream = (
                prepared.adapter in _STREAMING_ADAPTERS
                and prepared.payload.get("stream") is True
            )
            if wants_stream and not self._stream_unsupported:
                try:
                    return _stream_chat_completion(
                        prepared.url, payload=prepared.payload,
                        headers={**prepared.headers, **self.config.headers},
                        timeout=self.config.request_timeout_seconds, attempts=self.request_attempts,
                        secrets=self._redaction_secrets(prepared),
                        progress_callback=getattr(self, "progress_callback", None),
                        stall_timeout=self.config.stall_timeout_seconds,
                        first_token_timeout=self.config.first_token_timeout_seconds,
                    )
                except StreamingUnsupported as exc:
                    self._stream_unsupported = True
                    _logger.warning(
                        "Streaming request to %s was rejected or not served as SSE (%s); "
                        "falling back to a non-streaming request and disabling streaming "
                        "for the rest of this session.", prepared.url, exc,
                    )
                    # Fall through: send this same request again, without
                    # `stream`, exactly like every later call now will too.
            if wants_stream:
                # Either the fallback just above, or a session that already
                # learned (on an earlier call) that this server rejects
                # `stream: true`: both must actually ask for a non-streamed
                # reply, not merely skip the SSE reader over a payload that
                # still says `"stream": true`.
                payload = {k: v for k, v in prepared.payload.items() if k != "stream_options"}
                payload["stream"] = False
            else:
                payload = prepared.payload
            return request_json(
                prepared.url, payload=payload,
                headers={**prepared.headers, **self.config.headers},
                timeout=self.config.request_timeout_seconds, attempts=self.request_attempts,
                secrets=self._redaction_secrets(prepared),
            )
        except EngineError as exc:
            # Keep the existing runtime's auto-vision fallback contract.
            if exc.kind == "vision" and exc.status in {None, 400, 415, 422, 500}:
                exc.status = 415
            raise


class JSONEngine(HTTPMixin, Engine):
    def __init__(self, config: Config, api_key: str | None):
        self.config = config
        self.api_key = api_key
        self._count_unavailable = False
        self._legacy_count_unavailable = False
        # Optional streaming progress hook; see _stream_chat_completion.
        self.progress_callback: Callable[[dict[str, Any]], None] | None = None

    def count_input_tokens(self, prepared: PreparedRequest) -> int | None:
        if self._count_unavailable:
            return self._count_legacy_llama(prepared)
        body, adapter = prepared.payload, prepared.adapter
        url, field = prepared.url, "input_tokens"
        if adapter == "llamacpp":
            url += "/input_tokens"
            payload = body
        elif adapter == "vllm":
            url = self.config.base_url.removesuffix("/v1") + "/tokenize"
            payload = {key: body[key] for key in (
                "model", "messages", "tools", "tool_choice", "chat_template",
                "chat_template_kwargs", "add_generation_prompt", "continue_final_message",
                "add_special_tokens", "media_io_kwargs",
            ) if key in body}
            field = "count"
        elif adapter == "openai_responses":
            url += "/input_tokens"
            payload = {key: body[key] for key in (
                "model", "input", "instructions", "reasoning", "text", "tools",
                "tool_choice", "parallel_tool_calls", "conversation",
                "previous_response_id", "truncation",
            ) if key in body}
        elif adapter == "anthropic":
            url += "/count_tokens"
            payload = {key: body[key] for key in (
                "model", "messages", "system", "tools", "tool_choice", "thinking",
            ) if key in body}
        elif adapter == "gemini":
            url = url.removesuffix(":generateContent") + ":countTokens"
            payload = {"generateContentRequest": {
                **body, "model": "models/" + self.config.model.removeprefix("models/"),
            }}
            field = "totalTokens"
        else:
            return None
        try:
            raw = request_json(
                url, payload=payload, headers={**prepared.headers, **self.config.headers},
                timeout=min(10, self.config.request_timeout_seconds or 10), attempts=1,
                secrets=self._redaction_secrets(prepared),
            )
            count = raw.get(field)
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                return count
        except EngineError as exc:
            # A provider's input rejection is useful evidence, not a missing API.
            if exc.kind == "context":
                raise
            if exc.status in {404, 405, 501}:
                self._count_unavailable = True
        return self._count_legacy_llama(prepared)

    def _count_legacy_llama(self, prepared: PreparedRequest) -> int | None:
        """Older llama.cpp builds: render their template, then tokenize text.

        Media needs the native count endpoint; tokenizing a placeholder would
        miss the image embeddings and produce a falsely precise count.
        """
        if prepared.adapter != "llamacpp" or self._legacy_count_unavailable:
            return None
        if any(not isinstance(item.get("content"), str) for item in prepared.payload.get("messages", [])):
            return None
        root = self.config.base_url.removesuffix("/v1")
        kwargs = dict(headers={**prepared.headers, **self.config.headers},
                      timeout=min(10, self.config.request_timeout_seconds or 10), attempts=1,
                      secrets=self._redaction_secrets(prepared))
        try:
            formatted = request_json(root + "/apply-template", payload=prepared.payload, **kwargs)
            if not isinstance(formatted.get("prompt"), str):
                return None
            raw = request_json(root + "/tokenize", payload={
                "content": formatted["prompt"], "add_special": False, "parse_special": True,
            }, **kwargs)
            tokens = raw.get("tokens")
            if isinstance(tokens, list) and tokens:
                return len(tokens)
        except EngineError as exc:
            if exc.kind == "context":
                raise
            if exc.status in {404, 405, 501}:
                self._legacy_count_unavailable = True
        return None

    def _require_key(self) -> None:
        if requires_api_key(self.config.provider, self.config.adapter) and not self.api_key:
            raise EngineError(
                "No API key is available for this provider. Run "
                "`python3 artificium.py key` or set ARTIFICIUM_API_KEY."
            )

    def complete(self, messages: list[dict[str, Any]]) -> EngineReply:
        return self.complete_prepared(self.prepare(messages))

    def complete_prepared(self, prepared: PreparedRequest) -> EngineReply:
        self._require_key()
        raw = self._post(prepared)
        try:
            reply = self.parse(raw)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EngineError("The server response does not match the selected API format.",
                              kind="response", hint="Check the API type and endpoint in model settings.") from exc
        except EngineError as exc:
            raise EngineError(_redact(str(exc), self._redaction_secrets(prepared)),
                              kind=exc.kind or "response", status=exc.status, hint=exc.hint) from exc
        if not reply.content.strip():
            reason = reply.finish_reason or "no finish reason"
            hint = ("Generation ended before a usable answer. Inspect saved usage and the server log: reasoning may have consumed the output allowance, or remaining context may have been exhausted."
                    if reply.provider_reasoning or reason in {"length", "incomplete", "max_tokens", "MAX_TOKENS"}
                    else "The model returned no usable answer. Check the model, chat template, and server log.")
            raise EngineError(f"The model returned an empty answer ({reason}).", kind="empty", hint=hint, reply=reply)
        return reply

    def request_summary(self) -> dict[str, Any]:
        summary = super().request_summary()
        summary["header_names"] = sorted(
            set(summary["header_names"]) | set(self.config.headers)
        )
        return summary

    @abstractmethod
    def parse(self, raw: dict[str, Any]) -> EngineReply:
        raise NotImplementedError


def _chat_response(raw: dict[str, Any]) -> EngineReply:
    try:
        choice = raw["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EngineError(f"Unrecognized engine response: {json_dumps(raw)}") from exc
    content_value = message.get("content")
    if isinstance(content_value, str):
        content = content_value
    elif isinstance(content_value, list):
        content = "\n".join(
            str(item.get("text") or item.get("content") or "")
            for item in content_value if isinstance(item, dict)
        ).strip()
    else:
        content = ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if not reasoning and isinstance(message.get("reasoning_details"), list):
        reasoning = "\n".join(
            str(item.get("text") or item.get("content") or "")
            for item in message["reasoning_details"] if isinstance(item, dict)
        ).strip()
    return EngineReply(
        content=content,
        usage=dict(raw.get("usage") or {}),
        raw=raw,
        finish_reason=choice.get("finish_reason"),
        provider_reasoning=str(reasoning) if reasoning else None,
    )


def _prompt_cache_key_value(config: Config) -> str | None:
    """Resolve the configured prompt-cache routing hint, if any."""
    if not config.prompt_cache_key:
        return None
    if config.prompt_cache_key == "auto":
        return f"artificium-{config.instance_id}"
    return config.prompt_cache_key


def _chat_common(config: Config) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in (
        "temperature", "top_p", "top_k", "min_p", "frequency_penalty",
        "presence_penalty", "repetition_penalty", "seed",
    ):
        value = getattr(config, name)
        if value is not None:
            payload[name] = value
    if config.max_output_tokens is not None:
        payload["max_tokens"] = config.max_output_tokens
    if config.stop_sequences:
        payload["stop"] = list(config.stop_sequences)
    return payload


# Concrete provider adapters live in engine_adapters.py (chat-completions
# family: OpenAI-compatible, llama.cpp, custom JSON, OpenRouter, vLLM) and
# engine_providers.py (Ollama, OpenAI Responses, Gemini, Anthropic). Import
# them here so `artificium.engine.<AdapterClass>` keeps resolving, and so
# make_engine() below can look them up by adapter name. A test that patches a
# method directly on one of these classes (e.g.
# "artificium.engine.JSONEngine._post") patches the shared JSONEngine class
# object itself, so it applies no matter which module defines a subclass.
from .engine_adapters import (  # noqa: E402
    CustomJSONEngine,
    LlamaCppEngine,
    OpenAICompatibleEngine,
    OpenRouterEngine,
    VLLMEngine,
)
from .engine_providers import (  # noqa: E402
    AnthropicEngine,
    GeminiEngine,
    OllamaEngine,
    OpenAIResponsesEngine,
)



def make_engine(config: Config, api_key: str | None) -> Engine:
    adapters: dict[str, type[JSONEngine]] = {
        "llamacpp": LlamaCppEngine,
        "openai_compatible": OpenAICompatibleEngine,
        "custom_json": CustomJSONEngine,
        "openrouter": OpenRouterEngine,
        "vllm": VLLMEngine,
        "ollama": OllamaEngine,
        "openai_responses": OpenAIResponsesEngine,
        "gemini": GeminiEngine,
        "anthropic": AnthropicEngine,
    }
    try:
        engine_type = adapters[config.adapter]
    except KeyError as exc:
        raise EngineError(f"Unsupported engine adapter: {config.adapter}") from exc
    return engine_type(config, api_key)
