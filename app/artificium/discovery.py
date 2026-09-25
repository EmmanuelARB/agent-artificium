"""Provider metadata, normalized without guessing capabilities from model names."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from .engine import EngineError, request_json


@dataclass(frozen=True)
class ModelDiscovery:
    models: tuple[str, ...] = ()
    error: str | None = None
    status: int | None = None
    endpoint: str | None = None
    authentication_required: bool = False
    details: dict[str, dict[str, Any]] = field(default_factory=dict)
    error_kind: str | None = None
    hint: str | None = None


def api_headers(provider: str, api_key: str | None,
                headers: dict[str, str] | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    if provider == "anthropic":
        result["anthropic-version"] = "2023-06-01"
    if api_key:
        name = {"anthropic": "x-api-key", "gemini": "x-goog-api-key"}.get(provider, "Authorization")
        result[name] = api_key if name != "Authorization" else f"Bearer {api_key}"
    return {**result, **(headers or {})}


def _positive(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def _supported(value: Any) -> bool | None:
    if isinstance(value, dict) and isinstance(value.get("supported"), bool):
        return value["supported"]
    return value if isinstance(value, bool) else None


def _model_detail(item: dict[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {"owned_by": str(item.get("owned_by") or "")}
    top = item.get("top_provider") if isinstance(item.get("top_provider"), dict) else {}
    context = _positive(item.get("max_model_len") or item.get("context_length") or item.get("max_input_tokens") or top.get("context_length"))
    output = _positive(item.get("max_completion_tokens") or item.get("max_tokens") or top.get("max_completion_tokens"))
    if context:
        detail["context_length"] = context
    if output:
        detail["max_output_tokens"] = output
    architecture = item.get("architecture") or {}
    inputs = architecture.get("input_modalities") if isinstance(architecture, dict) else None
    modalities = item.get("modalities") or {}
    if isinstance(modalities, dict) and isinstance(modalities.get("vision"), bool):
        detail["vision"] = modalities["vision"]
    elif isinstance(inputs, list):
        detail["vision"] = "image" in inputs
    parameters = item.get("supported_parameters")
    if isinstance(parameters, list):
        detail["supported_parameters"] = [str(x) for x in parameters]
    if isinstance(item.get("reasoning"), dict):
        detail["reasoning"] = dict(item["reasoning"])
    caps = item.get("capabilities")
    if isinstance(caps, list):
        detail["capabilities"] = caps
        if "vision" in caps or "multimodal" in caps:
            detail["vision"] = True
    elif isinstance(caps, dict):
        image = _supported(caps.get("image_input"))
        thinking = _supported(caps.get("thinking"))
        if image is not None:
            detail["vision"] = image
        if thinking is not None:
            detail["thinking"] = thinking
        effort = caps.get("effort")
        if isinstance(effort, dict):
            detail["reasoning"] = {"supported_efforts": [name for name, val in effort.items() if name != "supported" and _supported(val)]}
        thinking_caps = caps.get("thinking") or {}
        types = thinking_caps.get("types") if isinstance(thinking_caps, dict) else None
        if isinstance(types, dict):
            detail["thinking_types"] = [name for name, val in types.items() if _supported(val)]
    return detail


def _failed(error: EngineError, endpoint: str) -> ModelDiscovery:
    return ModelDiscovery(error=str(error), status=error.status, endpoint=endpoint,
                          authentication_required=error.status == 401,
                          error_kind=error.kind, hint=error.hint)


def _openai_base_root(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value[:-3].rstrip("/") if value.endswith("/v1") else value


def discover_models(base_url: str, *, api_key: str | None = None,
                    timeout: float = 10, headers: dict[str, str] | None = None) -> ModelDiscovery:
    return discover_provider_models("custom", base_url, api_key=api_key, timeout=timeout, headers=headers)


def discover_provider_models(provider: str, base_url: str, *, api_key: str | None = None,
                             timeout: float = 10, headers: dict[str, str] | None = None) -> ModelDiscovery:
    if provider == "ollama":
        return discover_ollama_models(base_url, api_key=api_key, timeout=timeout, headers=headers)
    endpoint = f"{base_url.rstrip('/')}/models"
    auth = api_headers(provider, api_key, headers)
    details: dict[str, dict[str, Any]] = {}
    query: dict[str, Any] = {}
    seen_pages: set[str] = set()
    for _ in range(100):
        url = endpoint + ("?" + urlencode(query) if query else "")
        try:
            value = request_json(url, headers=auth, timeout=timeout, secrets=[api_key])
        except EngineError as exc:
            return _failed(exc, url)
        data = value.get("models" if provider == "gemini" else "data")
        if not isinstance(data, list):
            return ModelDiscovery(error="The endpoint did not return a model list.", endpoint=url, error_kind="response")
        for item in data:
            if not isinstance(item, dict):
                continue
            model = str(item.get("name") if provider == "gemini" else item.get("id") or "").strip()
            if not model or model == "None":
                continue
            if provider == "gemini":
                methods = item.get("supportedGenerationMethods")
                if isinstance(methods, list) and "generateContent" not in methods:
                    continue
                model = model.removeprefix("models/")
                detail: dict[str, Any] = {}
                if _positive(item.get("inputTokenLimit")):
                    detail["context_length"] = int(item["inputTokenLimit"])
                if _positive(item.get("outputTokenLimit")):
                    detail["max_output_tokens"] = int(item["outputTokenLimit"])
                if isinstance(item.get("thinking"), bool):
                    detail["thinking"] = item["thinking"]
                if "topK" not in item:
                    detail["top_k_supported"] = False
            else:
                detail = _model_detail(item)
            details[model] = detail
        token = value.get("nextPageToken") if provider == "gemini" else value.get("last_id") if value.get("has_more") else None
        if not token or str(token) in seen_pages:
            return ModelDiscovery(models=tuple(details), details=details, endpoint=endpoint)
        seen_pages.add(str(token))
        query = {"pageToken" if provider == "gemini" else "after_id": str(token)}
    return ModelDiscovery(error="The server's model list did not finish. Enter the model ID directly.", endpoint=endpoint)


def discover_ollama_models(base_url: str, *, api_key: str | None = None,
                          timeout: float = 10, headers: dict[str, str] | None = None) -> ModelDiscovery:
    endpoint = f"{_openai_base_root(base_url)}/api/tags"
    try:
        value = request_json(endpoint, headers=api_headers("ollama", api_key, headers), timeout=timeout, secrets=[api_key])
    except EngineError as exc:
        # Only a missing native discovery route warrants trying compatibility.
        if exc.status == 404:
            return discover_models(f"{_openai_base_root(base_url)}/v1", api_key=api_key, timeout=timeout, headers=headers)
        return _failed(exc, endpoint)
    data = value.get("models")
    if not isinstance(data, list):
        return ModelDiscovery(error="The server did not return an Ollama model list.", endpoint=endpoint, error_kind="response")
    models = tuple(dict.fromkeys(str(x.get("name") or x.get("model")) for x in data if isinstance(x, dict) and (x.get("name") or x.get("model"))))
    return ModelDiscovery(models=models, endpoint=endpoint)


def discover_ollama_model_details(base_url: str, model: str, *, api_key: str | None = None,
                                 timeout: float = 10, headers: dict[str, str] | None = None) -> dict[str, Any]:
    root, auth = _openai_base_root(base_url), api_headers("ollama", api_key, headers)
    result: dict[str, Any] = {}
    try:
        value = request_json(f"{root}/api/ps", headers=auth, timeout=timeout, secrets=[api_key])
        for item in value.get("models", []):
            if isinstance(item, dict) and str(item.get("name") or item.get("model")) in {model, model + ":latest"}:
                context = _positive(item.get("context_length"))
                if context:
                    result["running_context_length"] = context
    except (EngineError, TypeError):
        pass
    try:
        value = request_json(f"{root}/api/show", payload={"model": model}, headers=auth, timeout=timeout, secrets=[api_key])
        capabilities = value.get("capabilities")
        if isinstance(capabilities, list):
            result["capabilities"] = capabilities
            result["vision"] = "vision" in capabilities
            result["thinking"] = "thinking" in capabilities
        info = value.get("model_info")
        if isinstance(info, dict):
            contexts = [_positive(val) for name, val in info.items() if str(name).endswith(".context_length")]
            if any(contexts):
                result["model_context_length"] = max(x for x in contexts if x)
    except EngineError:
        pass
    return result


def discover_llamacpp_properties(base_url: str, model: str, *, api_key: str | None = None,
                                timeout: float = 10, headers: dict[str, str] | None = None) -> dict[str, Any]:
    url = f"{_openai_base_root(base_url)}/props?{urlencode({'model': model})}"
    try:
        value = request_json(url, headers=api_headers("llamacpp", api_key, headers), timeout=timeout, secrets=[api_key])
    except EngineError:
        return {}
    result: dict[str, Any] = {}
    generation = value.get("default_generation_settings")
    context = _positive(generation.get("n_ctx")) if isinstance(generation, dict) else None
    if context:
        result["context_length"] = context  # per-slot allocation, not training length
    modalities = value.get("modalities")
    if isinstance(modalities, dict) and isinstance(modalities.get("vision"), bool):
        result["vision"] = modalities["vision"]
    # Template variables tell us which controls can have an effect. Do not
    # infer effort levels from the model filename or treat trace parsing as a toggle.
    template = value.get("chat_template")
    if isinstance(template, str) and template:
        result["thinking_toggle"] = "enable_thinking" in template
        result["effort_control"] = "reasoning_effort" in template
    if isinstance(value.get("chat_template_caps"), dict):
        result["chat_template_caps"] = value["chat_template_caps"]
    if value.get("build_info"):
        result["server_build"] = str(value["build_info"])
    return result


def detected_context_window(provider: str, discovery: ModelDiscovery, model: str, *,
                            base_url: str, api_key: str | None = None, timeout: float = 10) -> int | None:
    if provider == "ollama":
        return discover_ollama_model_details(base_url, model, api_key=api_key, timeout=timeout).get("running_context_length")
    return _positive(discovery.details.get(model, {}).get("context_length"))
