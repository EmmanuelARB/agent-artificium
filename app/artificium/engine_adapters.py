from __future__ import annotations

import copy
import re
import urllib.parse
from typing import Any

from .config import Config
from .engine import (
    Engine,
    EngineError,
    EngineReply,
    JSONEngine,
    PreparedRequest,
    _chat_common,
    _chat_response,
    _configured,
    _merge_options,
    _prompt_cache_key_value,
    _reject,
    _stream_flag,
    materialize_openai_messages,
)


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
            **_stream_flag(self.config),
            **_chat_common(self.config),
        }
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort
        cache_key = _prompt_cache_key_value(self.config)
        if cache_key:
            payload["prompt_cache_key"] = cache_key
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
            **_stream_flag(self.config),
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
            **_stream_flag(self.config),
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
            **_stream_flag(self.config),
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


