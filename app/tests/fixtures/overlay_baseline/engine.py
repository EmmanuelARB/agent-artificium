from __future__ import annotations

import base64
import copy
import json
import mimetypes
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, requires_api_key
from .filesystem import json_dumps


DEFAULT_USER_AGENT = "Artificium-revolution/1.9.3"


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


def request_json(url: str, *, payload: dict[str, Any] | None = None,
                 headers: dict[str, str] | None = None, timeout: float | None = 10,
                 attempts: int = 1, secrets: list[str | None] | None = None) -> dict[str, Any]:
    """One JSON transport for discovery, checks, and inference.

    Retry explicit transient HTTP failures, never blindly replay a timed-out
    inference or a malformed response. Those may already have incurred work.
    """
    supplied = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT,
                **(headers or {})}
    redactions = [*(secrets or []), *supplied.values()]
    for attempt in range(attempts):
        try:
            body = json_dumps(payload).encode("utf-8") if payload is not None else None
            if body is not None:
                supplied.setdefault("Content-Type", "application/json")
            request = urllib.request.Request(url, data=body, headers=supplied,
                                             method="POST" if body is not None else "GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeError) as exc:
                preview = _redact(raw[:240].decode("utf-8", errors="replace"), redactions)
                raise EngineError(
                    f"The endpoint returned invalid JSON: {preview!r}", kind="response",
                    hint="Check that the URL reaches the JSON API, rather than a login page, web UI, or streaming endpoint.",
                ) from exc
            if not isinstance(decoded, dict):
                raise EngineError("The endpoint returned a JSON value instead of an API object.", kind="response")
            if decoded.get("error"):
                raise _server_error(_redact(json_dumps(decoded), redactions))
            return decoded
        except urllib.error.HTTPError as exc:
            detail = _redact(exc.read(16_384).decode("utf-8", errors="replace"), redactions)
            failure = _server_error(detail, exc.code)
            if attempt + 1 < attempts and (exc.code == 429 or exc.code >= 500) and failure.kind not in {"context", "vision", "template"}:
                time.sleep(min(2 ** attempt, 4))
                continue
            raise failure from exc
        except (TimeoutError, socket.timeout) as exc:
            elapsed = f" after {timeout:g} seconds" if timeout is not None else ""
            raise EngineError(f"The model request timed out{elapsed}.", kind="timeout",
                              hint="Check that the server is still processing. Request timeout can be increased or set to off in model settings; the server or network may also impose timeouts.") from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                elapsed = f" after {timeout:g} seconds" if timeout is not None else ""
                raise EngineError(f"The connection timed out{elapsed}.", kind="timeout",
                                  hint="Check the server is reachable. Request timeout can be increased or set to off; server and network timeouts still apply.") from exc
            if isinstance(reason, ssl.SSLError):
                raise EngineError("TLS verification failed: " + _redact(str(reason), redactions), kind="tls",
                                  hint="Use the correct HTTPS address and a trusted server certificate.") from exc
            raise EngineError("Could not reach the model server: " + _redact(str(reason), redactions), kind="network",
                              hint="Start the model server and check the address and port. localhost refers to the machine running Artificium.") from exc
        except (OSError, ValueError) as exc:
            raise EngineError("Connection failed: " + _redact(str(exc), redactions), kind="network",
                              hint="Check the server address and connection.") from exc
    raise AssertionError("request attempts must be positive")


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


def _redact(text: str, secrets: list[str | None]) -> str:
    result = text
    for secret in secrets:
        if secret:
            result = result.replace(secret, "<redacted>")
    return result


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

    def _post(self, prepared: PreparedRequest) -> dict[str, Any]:
        try:
            return request_json(
                prepared.url, payload=prepared.payload,
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


class OpenAICompatibleEngine(JSONEngine):
    """Best-effort transport for an explicitly OpenAI-compatible custom server."""

    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        if self.config.reasoning_effort == "on":
            raise EngineError("This API defines effort levels, not a universal reasoning-on switch. Choose server default or a documented effort.", kind="settings")
        _reject(
            self.config, "generic OpenAI-compatible transport",
            "reasoning_budget_tokens", "reasoning_mode",
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": materialize_openai_messages(messages),
            "stream": False,
            **_chat_common(self.config),
        }
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort
        _merge_options(payload, self.config.request_options)
        return PreparedRequest(
            adapter="openai_compatible",
            url=f"{self.config.base_url}/{self.config.endpoint}",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            payload=payload,
            guarantee=(
                "OpenAI-standard fields plus conventional extensions; acceptance is "
                "defined by the custom server"
            ),
        )

    def parse(self, raw: dict[str, Any]) -> EngineReply:
        return _chat_response(raw)


_LLAMACPP_MEDIA_MARKER = re.compile(r"<__media(?:_[A-Za-z0-9]+)?__>")


def _llamacpp_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Quote literal media markers in text, preserving history and image parts.

    llama.cpp inserts its own markers for real image inputs. A marker copied
    from /props or logs into ordinary text must not consume another bitmap.
    This covers the generated marker format and the legacy <__media__> form.
    """
    def escape(text: str) -> str:
        return _LLAMACPP_MEDIA_MARKER.sub(
            lambda match: "&lt;" + match.group(0)[1:-1] + "&gt;", text
        )

    rendered = materialize_openai_messages(messages)
    for message in rendered:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = escape(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    part["text"] = escape(part["text"])
        for key in ("reasoning_content", "reasoning"):
            if isinstance(message.get(key), str):
                message[key] = escape(message[key])
    return rendered


class LlamaCppEngine(OpenAICompatibleEngine):
    """llama-server Chat Completions, with its native sampling field names.

    Effort is passed to the server's chat template; it is not a universal
    reasoning capability. Unset values preserve the server's own defaults.
    """

    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        _reject(self.config, "llama.cpp", "reasoning_budget_tokens", "reasoning_mode")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": _llamacpp_messages(messages),
            "stream": False,
            **_chat_common(self.config),
        }
        if "repetition_penalty" in payload:
            payload["repeat_penalty"] = payload.pop("repetition_penalty")
        if self.config.reasoning_effort == "on":
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        elif self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort
        _merge_options(payload, self.config.request_options)
        return PreparedRequest(
            adapter="llamacpp",
            url=f"{self.config.base_url}/{self.config.endpoint}",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            payload=payload,
            guarantee="llama-server contract; reasoning effort depends on the chat template",
        )


_CUSTOM_MISSING = object()


def _plain_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") in {"image_url", "artificium_image"}:
                parts.append("[image]")
            else:
                parts.append(str(item.get("text") or item.get("content") or ""))
        else:
            parts.append(str(item))
    return "\n".join(part for part in parts if part)


def _custom_template_context(
    config: Config,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    materialized = materialize_openai_messages(messages)
    transcript: list[str] = []
    system: list[str] = []
    last_user = ""
    for message in materialized:
        role = str(message.get("role") or "user")
        text = _plain_message_content(message.get("content"))
        transcript.append(f"[{role}]\n{text}")
        if role == "system":
            system.append(text)
        elif role == "user":
            last_user = text
    context: dict[str, Any] = {
        "model": config.model,
        "messages": materialized,
        "prompt": "\n\n".join(transcript),
        "system": "\n\n".join(system),
        "last_user": last_user,
        "context_window_tokens": config.context_window_tokens,
    }
    for name in (
        "reasoning_effort", "reasoning_budget_tokens", "reasoning_mode",
        "temperature", "max_output_tokens", "top_p", "top_k", "min_p",
        "frequency_penalty", "presence_penalty", "repetition_penalty", "seed",
        "stop_sequences",
    ):
        configured = getattr(config, name)
        context[name] = (
            None if name == "stop_sequences" and not configured
            else copy.deepcopy(configured)
        )
    return context


def _expand_custom_template(value: Any, context: dict[str, Any]) -> Any:
    """Expand exact `$artificium.NAME` values while preserving JSON types.

    A placeholder whose optional configured value is null is omitted from its
    containing object/list. Exact placeholders avoid string interpolation and
    its quoting/type ambiguities.
    """

    if isinstance(value, str) and value.startswith("$artificium."):
        name = value.removeprefix("$artificium.")
        if name not in context:
            raise EngineError(f"Unknown custom-contract placeholder: {value}")
        resolved = context[name]
        return _CUSTOM_MISSING if resolved is None else copy.deepcopy(resolved)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            resolved = _expand_custom_template(item, context)
            if resolved is not _CUSTOM_MISSING:
                result[str(key)] = resolved
        return result
    if isinstance(value, list):
        result_list: list[Any] = []
        for item in value:
            resolved = _expand_custom_template(item, context)
            if resolved is not _CUSTOM_MISSING:
                result_list.append(resolved)
        return result_list
    return copy.deepcopy(value)


def _json_pointer(value: Any, pointer: str) -> Any:
    """Resolve a small RFC 6901 JSON Pointer used by custom response mappings."""

    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise EngineError(
            f"Custom response path {pointer!r} must be a JSON Pointer beginning with /"
        )
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                raise KeyError(part)
        except (KeyError, IndexError, ValueError) as exc:
            raise EngineError(
                f"Custom response path {pointer!r} was not present in the provider response"
            ) from exc
    return current


def _custom_response_value(raw: dict[str, Any], paths: Any) -> Any:
    candidates = [paths] if isinstance(paths, str) else paths
    if not isinstance(candidates, list) or not candidates or not all(
        isinstance(item, str) for item in candidates
    ):
        raise EngineError("Custom response mappings must be JSON Pointer strings or lists")
    failures: list[str] = []
    for pointer in candidates:
        try:
            return _json_pointer(raw, pointer)
        except EngineError:
            failures.append(pointer)
    raise EngineError(
        "None of the custom response paths were present: " + ", ".join(failures)
    )


def _custom_response_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            part for part in (_custom_response_text(item) for item in value) if part
        )
    if isinstance(value, dict):
        for key in ("text", "content", "output_text"):
            if key in value:
                return _custom_response_text(value[key])
    return str(value)


class CustomJSONEngine(JSONEngine):
    """Declarative escape hatch for non-OpenAI JSON-over-HTTP model APIs."""

    @property
    def contract(self) -> dict[str, Any]:
        return self.config.custom_contract

    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        body_template = self.contract.get("body")
        if not isinstance(body_template, dict):
            raise EngineError("custom_contract.body must be a JSON object")
        response = self.contract.get("response")
        if not isinstance(response, dict) or "content" not in response:
            raise EngineError("custom_contract.response.content is required")
        payload = _expand_custom_template(
            body_template,
            _custom_template_context(self.config, messages),
        )
        if not isinstance(payload, dict):
            raise EngineError("the expanded custom request body must be a JSON object")
        if self.config.request_options:
            _merge_options(payload, self.config.request_options)

        url = str(
            self.contract.get("url")
            or f"{self.config.base_url}/{self.config.endpoint}"
        ).strip()
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise EngineError("custom_contract.url must be an absolute HTTP(S) URL")

        raw_headers = self.contract.get("headers") or {}
        if not isinstance(raw_headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_headers.items()
        ):
            raise EngineError("custom_contract.headers must contain string values")
        headers = dict(raw_headers)
        auth = self.contract.get("auth", {})
        if auth is not False:
            if not isinstance(auth, dict):
                raise EngineError("custom_contract.auth must be an object or false")
            if auth.get("required") and not self.api_key:
                raise EngineError(
                    "This custom contract requires an API key. Run "
                    "`python3 artificium.py key`."
                )
            if self.api_key:
                header = str(auth.get("header") or "Authorization")
                prefix = str(auth.get("prefix") if "prefix" in auth else "Bearer ")
                headers[header] = prefix + self.api_key
        return PreparedRequest(
            adapter="custom_json",
            url=url,
            headers=headers,
            payload=payload,
            guarantee="operator-defined custom JSON contract",
        )

    def parse(self, raw: dict[str, Any]) -> EngineReply:
        response = self.contract.get("response")
        if not isinstance(response, dict) or "content" not in response:
            raise EngineError("custom_contract.response.content is required")
        content = _custom_response_text(
            _custom_response_value(raw, response["content"])
        )
        reasoning = None
        if response.get("reasoning") is not None:
            reasoning = _custom_response_text(
                _custom_response_value(raw, response["reasoning"])
            ) or None
        usage: dict[str, Any] = {}
        if response.get("usage") is not None:
            usage_value = _custom_response_value(raw, response["usage"])
            if isinstance(usage_value, dict):
                usage = copy.deepcopy(usage_value)
        finish_reason = None
        if response.get("finish_reason") is not None:
            finish_value = _custom_response_value(raw, response["finish_reason"])
            finish_reason = str(finish_value) if finish_value is not None else None
        return EngineReply(
            content=content,
            usage=usage,
            raw=raw,
            finish_reason=finish_reason,
            provider_reasoning=reasoning,
        )


class OpenRouterEngine(JSONEngine):
    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": materialize_openai_messages(messages),
            "stream": False,
            **_chat_common(self.config),
        }
        if (
            self.config.reasoning_effort is not None
            and self.config.reasoning_budget_tokens is not None
        ):
            raise EngineError(
                "OpenRouter reasoning accepts effort or max_tokens, not both"
            )
        reasoning: dict[str, Any] = {}
        if self.config.reasoning_effort == "on":
            reasoning["enabled"] = True
        elif self.config.reasoning_effort is not None:
            reasoning["effort"] = self.config.reasoning_effort
        if self.config.reasoning_budget_tokens is not None:
            reasoning["max_tokens"] = self.config.reasoning_budget_tokens
        if self.config.reasoning_mode is not None:
            reasoning["mode"] = self.config.reasoning_mode
        if reasoning:
            payload["reasoning"] = reasoning
        explicit = _configured(
            self.config,
            "reasoning_effort", "reasoning_budget_tokens", "reasoning_mode", "temperature",
            "max_output_tokens", "top_p", "top_k", "min_p",
            "frequency_penalty", "presence_penalty", "repetition_penalty",
            "seed", "stop_sequences",
        )
        if explicit:
            payload["provider"] = {"require_parameters": True}
        _merge_options(payload, self.config.request_options)
        return PreparedRequest(
            adapter="openrouter",
            url=f"{self.config.base_url}/{self.config.endpoint}",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            payload=payload,
        )

    def parse(self, raw: dict[str, Any]) -> EngineReply:
        return _chat_response(raw)


class VLLMEngine(JSONEngine):
    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        _reject(self.config, "vLLM Chat Completions", "reasoning_mode")
        if self.config.reasoning_effort is not None and self.config.reasoning_effort not in {
            "none", "low", "medium", "high",
        }:
            raise EngineError(
                "vLLM reasoning_effort accepts none, low, medium, or high"
            )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": materialize_openai_messages(messages),
            "stream": False,
            **_chat_common(self.config),
        }
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if self.config.reasoning_budget_tokens is not None:
            payload["thinking_token_budget"] = self.config.reasoning_budget_tokens
        _merge_options(payload, self.config.request_options)
        return PreparedRequest(
            adapter="vllm",
            url=f"{self.config.base_url}/{self.config.endpoint}",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            payload=payload,
        )

    def parse(self, raw: dict[str, Any]) -> EngineReply:
        return _chat_response(raw)


def _ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        text: list[str] = []
        images: list[str] = []
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "artificium_image":
                    _, data = _image(str(part.get("path") or ""), part.get("mime"))
                    images.append(data)
                elif isinstance(part, dict):
                    text.append(str(part.get("text") or part.get("content") or ""))
                else:
                    text.append(str(part))
        else:
            text.append(str(content or ""))
        item: dict[str, Any] = {
            "role": str(message.get("role") or "user"),
            "content": "\n".join(text),
        }
        if images:
            item["images"] = images
        result.append(item)
    return result


class OllamaEngine(JSONEngine):
    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        _reject(
            self.config,
            "Ollama native /api/chat",
            "reasoning_budget_tokens", "reasoning_mode",
            "frequency_penalty", "presence_penalty",
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": _ollama_messages(messages),
            "stream": False,
        }
        effort = self.config.reasoning_effort
        is_gpt_oss = "gpt-oss" in self.config.model.lower()
        if effort is not None:
            if is_gpt_oss and effort not in {"low", "medium", "high"}:
                raise EngineError(
                    "Ollama GPT-OSS accepts reasoning effort low, medium, or high; "
                    "its thinking trace cannot be disabled"
                )
            if not is_gpt_oss and effort in {"minimal", "xhigh"}:
                raise EngineError(
                    "Ollama native thinking accepts none, low, medium, high, or max; "
                    f"it has no exact {effort!r} level"
                )
            payload["think"] = False if effort == "none" else True if effort == "on" else effort
        options: dict[str, Any] = {"num_ctx": self.config.context_window_tokens}
        mappings = {
            "temperature": "temperature",
            "max_output_tokens": "num_predict",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
            "repetition_penalty": "repeat_penalty",
            "seed": "seed",
        }
        for source, target in mappings.items():
            value = getattr(self.config, source)
            if value is not None:
                options[target] = value
        if self.config.stop_sequences:
            options["stop"] = list(self.config.stop_sequences)
        payload["options"] = options
        _merge_options(payload, self.config.request_options)
        root = self.config.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3].rstrip("/")
        return PreparedRequest(
            adapter="ollama",
            url=f"{root}/api/chat",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            payload=payload,
        )

    def parse(self, raw: dict[str, Any]) -> EngineReply:
        message = raw.get("message")
        if not isinstance(message, dict):
            raise EngineError(f"Unrecognized Ollama response: {json_dumps(raw)}")
        usage = {
            key: raw[key]
            for key in (
                "total_duration", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration",
            )
            if key in raw
        }
        return EngineReply(
            content=str(message.get("content") or ""),
            usage=usage,
            raw=raw,
            finish_reason=str(raw.get("done_reason") or "") or None,
            provider_reasoning=str(message.get("thinking") or "") or None,
        )


class OpenAIResponsesEngine(JSONEngine):
    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        if self.config.reasoning_effort == "on":
            raise EngineError("OpenAI uses reasoning effort levels. Choose server default or an effort level.", kind="settings")
        _reject(
            self.config,
            "OpenAI Responses API",
            "reasoning_budget_tokens", "top_k", "min_p",
            "frequency_penalty", "presence_penalty", "repetition_penalty",
            "seed", "stop_sequences",
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": _materialize_openai_response_input(messages),
            "store": False,
        }
        reasoning: dict[str, Any] = {}
        if self.config.reasoning_effort is not None:
            reasoning["effort"] = self.config.reasoning_effort
        if self.config.reasoning_mode is not None:
            reasoning["mode"] = self.config.reasoning_mode
        if reasoning:
            payload["reasoning"] = reasoning
        if self.config.max_output_tokens is not None:
            payload["max_output_tokens"] = self.config.max_output_tokens
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        _merge_options(payload, self.config.request_options)
        return PreparedRequest(
            adapter="openai_responses",
            url=f"{self.config.base_url}/{self.config.endpoint}",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            payload=payload,
        )

    def parse(self, raw: dict[str, Any]) -> EngineReply:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for item in raw.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text_parts.append(str(part.get("text") or ""))
            elif item.get("type") == "reasoning":
                for part in item.get("summary") or []:
                    if isinstance(part, dict):
                        reasoning_parts.append(str(part.get("text") or ""))
        if not text_parts and isinstance(raw.get("output_text"), str):
            text_parts.append(raw["output_text"])
        return EngineReply(
            content="\n".join(text_parts).strip(),
            usage=dict(raw.get("usage") or {}),
            raw=raw,
            finish_reason=str(raw.get("status") or "") or None,
            provider_reasoning="\n".join(reasoning_parts).strip() or None,
        )


def _gemini_parts(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"text": str(content or "")}]
    result: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "artificium_image":
            mime, data = _image(str(part.get("path") or ""), part.get("mime"))
            result.append({"inlineData": {"mimeType": mime, "data": data}})
        elif isinstance(part, dict):
            result.append({"text": str(part.get("text") or part.get("content") or "")})
        else:
            result.append({"text": str(part)})
    return result


class GeminiEngine(JSONEngine):
    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        _reject(
            self.config,
            "Gemini generateContent",
            "reasoning_mode", "min_p", "repetition_penalty",
        )
        system_parts: list[dict[str, Any]] = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            parts = _gemini_parts(message.get("content"))
            if role == "system":
                system_parts.extend(parts)
            else:
                contents.append({
                    "role": "model" if role == "assistant" else "user",
                    "parts": parts,
                })
        payload: dict[str, Any] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        generation: dict[str, Any] = {}
        mappings = {
            "temperature": "temperature",
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
            "frequency_penalty": "frequencyPenalty",
            "presence_penalty": "presencePenalty",
            "seed": "seed",
        }
        for source, target in mappings.items():
            value = getattr(self.config, source)
            if value is not None:
                generation[target] = value
        if self.config.stop_sequences:
            generation["stopSequences"] = list(self.config.stop_sequences)
        effort = self.config.reasoning_effort
        budget = self.config.reasoning_budget_tokens
        model_lower = self.config.model.lower()
        if effort is not None and budget is not None:
            raise EngineError(
                "Gemini generateContent requires either reasoning effort or a thinking "
                "token budget, not both"
            )
        if effort is not None:
            if model_lower.startswith("gemini-2.5"):
                if effort == "none":
                    generation["thinkingConfig"] = {"thinkingBudget": 0}
                else:
                    raise EngineError(
                        "Gemini 2.5 generateContent uses an exact thinking token budget, "
                        "not thinkingLevel. Set --reasoning-budget-tokens instead."
                    )
            elif effort in {"minimal", "low", "medium", "high"}:
                generation["thinkingConfig"] = {"thinkingLevel": effort.upper()}
            else:
                raise EngineError(
                    "Gemini thinkingLevel accepts minimal, low, medium, or high for "
                    "supported Gemini 3+ models; the selected value has no exact mapping"
                )
        elif budget is not None:
            generation["thinkingConfig"] = {"thinkingBudget": budget}
        if generation:
            payload["generationConfig"] = generation
        _merge_options(payload, self.config.request_options)
        model = self.config.model.removeprefix("models/")
        endpoint = self.config.endpoint.format(model=urllib.parse.quote(model, safe=""))
        return PreparedRequest(
            adapter="gemini",
            url=f"{self.config.base_url}/{endpoint}",
            headers={"x-goog-api-key": self.api_key or ""},
            payload=payload,
        )

    def parse(self, raw: dict[str, Any]) -> EngineReply:
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise EngineError(f"Unrecognized Gemini response: {json_dumps(raw)}")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for part in parts:
            if not isinstance(part, dict) or "text" not in part:
                continue
            target = reasoning_parts if part.get("thought") else text_parts
            target.append(str(part.get("text") or ""))
        return EngineReply(
            content="\n".join(text_parts).strip(),
            usage=dict(raw.get("usageMetadata") or {}),
            raw=raw,
            finish_reason=str(candidate.get("finishReason") or "") or None,
            provider_reasoning="\n".join(reasoning_parts).strip() or None,
        )


def _anthropic_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content or "")}]
    result: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            result.append({"type": "text", "text": str(part)})
        elif part.get("type") == "artificium_image":
            mime, data = _image(str(part.get("path") or ""), part.get("mime"))
            result.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data},
            })
        elif part.get("type") == "text":
            result.append({"type": "text", "text": str(part.get("text") or "")})
        elif part.get("type") == "image_url":
            result.append({
                "type": "text",
                "text": "[An image_url block could not be converted.]",
            })
    return result


class AnthropicEngine(JSONEngine):
    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        _reject(
            self.config,
            "Anthropic Messages API",
            "reasoning_mode", "min_p", "frequency_penalty",
            "presence_penalty", "repetition_penalty", "seed",
        )
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            if role == "system":
                system_parts.append(str(message.get("content") or ""))
                continue
            role = "assistant" if role == "assistant" else "user"
            content = _anthropic_content(message.get("content"))
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(content)
            else:
                converted.append({"role": role, "content": content})
        # Messages requires max_tokens. Prefer the model's reported maximum when
        # unset. Other adapters leave an unset output limit to the provider.
        model_limit = self.config.model_capabilities.get("max_output_tokens")
        if not isinstance(model_limit, int) or isinstance(model_limit, bool) or model_limit < 1:
            model_limit = self.config.context_window_tokens
        max_tokens = self.config.max_output_tokens or model_limit
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "system": "\n\n".join(system_parts),
            "messages": converted,
        }
        for name in ("temperature", "top_p", "top_k"):
            value = getattr(self.config, name)
            if value is not None:
                payload[name] = value
        if self.config.stop_sequences:
            payload["stop_sequences"] = list(self.config.stop_sequences)
        effort = self.config.reasoning_effort
        if effort is not None:
            if effort == "minimal":
                raise EngineError("Anthropic output_config.effort has no minimal level")
            if effort == "none":
                payload["thinking"] = {"type": "disabled"}
            elif effort == "on":
                payload["thinking"] = {"type": "adaptive"}
            else:
                payload["output_config"] = {"effort": effort}
                thinking = self.config.model_capabilities.get("thinking_types", [])
                if "adaptive" in thinking:
                    payload["thinking"] = {"type": "adaptive"}
        if self.config.reasoning_budget_tokens is not None:
            if self.config.reasoning_budget_tokens >= max_tokens:
                raise EngineError(
                    "Anthropic max_output_tokens must be greater than the thinking token budget"
                )
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.config.reasoning_budget_tokens,
            }
        _merge_options(payload, self.config.request_options)
        return PreparedRequest(
            adapter="anthropic",
            url=f"{self.config.base_url}/{self.config.endpoint}",
            headers={
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
            },
            payload=payload,
        )

    def parse(self, raw: dict[str, Any]) -> EngineReply:
        blocks = raw.get("content") if isinstance(raw.get("content"), list) else []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block.get("type") == "thinking":
                reasoning_parts.append(
                    str(block.get("thinking") or block.get("summary") or "")
                )
        return EngineReply(
            content="\n".join(text_parts).strip(),
            usage=dict(raw.get("usage") or {}),
            raw=raw,
            finish_reason=str(raw.get("stop_reason") or "") or None,
            provider_reasoning="\n".join(reasoning_parts).strip() or None,
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
