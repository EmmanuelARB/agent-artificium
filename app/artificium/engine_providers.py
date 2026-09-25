from __future__ import annotations

import urllib.parse
from typing import Any

from .engine import (
    EngineError,
    EngineReply,
    JSONEngine,
    PreparedRequest,
    _image,
    _materialize_openai_response_input,
    _merge_options,
    _prompt_cache_key_value,
    _reject,
)
from .filesystem import json_dumps


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
        cache_key = _prompt_cache_key_value(self.config)
        if cache_key:
            payload["prompt_cache_key"] = cache_key
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


def _anthropic_transient_tail(messages: list[dict[str, Any]]) -> set[int]:
    """Indices of the request-shaped tail that changes on every call.

    The runtime always sends a transient state-header message last, tagged
    ``state_header`` once the caller adds that marker; an untagged caller
    still gets the positional fallback (the last message is always treated as
    transient) so older call sites keep caching correctly. A transient
    visual-context message, when present, immediately precedes it.
    """
    if not messages:
        return set()
    last = len(messages) - 1
    tail = {last}
    if len(messages) >= 2:
        previous = messages[last - 1].get("_artificium")
        if isinstance(previous, dict) and previous.get("kind") == "visual_context":
            tail.add(last - 1)
    return tail


class AnthropicEngine(JSONEngine):
    def prepare(self, messages: list[dict[str, Any]]) -> PreparedRequest:
        _reject(
            self.config,
            "Anthropic Messages API",
            "reasoning_mode", "min_p", "frequency_penalty",
            "presence_penalty", "repetition_penalty", "seed",
        )
        transient = _anthropic_transient_tail(messages)
        stable_indices = [
            i for i, message in enumerate(messages)
            if i not in transient and str(message.get("role") or "user") != "system"
        ]
        stable_end_index = stable_indices[-1] if stable_indices else None
        # Persisted working-context history reuses the same `runtime_input`
        # marker as the pending message, so only its position (immediately
        # before the transient tail) identifies it as still pending -- it is
        # persisted verbatim after a successful reply, so its boundary with
        # older persisted history is a second good cache breakpoint: both
        # stay valid across the next turn.
        pending_marker = (
            messages[stable_end_index].get("_artificium")
            if stable_end_index is not None else None
        )
        is_pending = isinstance(pending_marker, dict) and pending_marker.get("kind") == "runtime_input"
        persisted_end_index = stable_end_index - 1 if is_pending else None

        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        # Maps an original (non-system) message index to the position of the
        # last content block it contributed, after same-role merging.
        block_end_by_index: dict[int, tuple[int, int]] = {}
        for i, message in enumerate(messages):
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
            block_end_by_index[i] = (len(converted) - 1, len(converted[-1]["content"]) - 1)

        if self.config.prompt_cache:
            for index in {persisted_end_index, stable_end_index}:
                if index is None or index not in block_end_by_index:
                    continue
                msg_idx, block_idx = block_end_by_index[index]
                converted[msg_idx]["content"][block_idx]["cache_control"] = {"type": "ephemeral"}

        # Messages requires max_tokens. Prefer the model's reported maximum when
        # unset. Other adapters leave an unset output limit to the provider.
        # When the model's own maximum is unknown, cap the fallback instead of
        # defaulting to the whole context window (which is rarely a valid
        # output limit and can be far larger than any model actually allows).
        model_limit = self.config.model_capabilities.get("max_output_tokens")
        if not isinstance(model_limit, int) or isinstance(model_limit, bool) or model_limit < 1:
            model_limit = min(self.config.context_window_tokens, 32_000)
        max_tokens = self.config.max_output_tokens or model_limit
        system_text = "\n\n".join(system_parts)
        if self.config.prompt_cache and system_text:
            system: Any = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
        else:
            system = system_text
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "system": system,
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

