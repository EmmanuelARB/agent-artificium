from __future__ import annotations

import json
import queue
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .config import Config
from .filesystem import json_dumps
from .engine import DEFAULT_USER_AGENT, EngineError, _redact, _server_error


class StreamingUnsupported(EngineError):
    """Raised by the SSE transport when the server rejects `stream: true` or
    replies in a shape that is not an SSE stream at all (a plain JSON body,
    for example). This is an internal signal: `HTTPMixin._post` catches it,
    retries the same request once non-streaming, and remembers the fallback
    for the rest of the engine's session. It is never meant to reach the
    runtime's ordinary EngineError classification/recovery path."""


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


# Chat-completions-shaped adapters whose wire format defines `stream` and
# server-sent-event chunking. Other adapters (openai_responses, anthropic,
# gemini, ollama, custom_json) keep their own transport unchanged.
_STREAMING_ADAPTERS = frozenset({"openai_compatible", "llamacpp", "vllm", "openrouter"})


# HTTP statuses a server plausibly uses to reject an unsupported `stream`
# parameter, combined with the response body mentioning "stream" at all.
_STREAM_REJECTION_STATUSES = frozenset({400, 404, 415, 422})


def _stream_chat_completion(
    url: str, *, payload: dict[str, Any], headers: dict[str, str],
    timeout: float | None, attempts: int, secrets: list[str | None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    stall_timeout: float | None = 180.0,
    first_token_timeout: float | None = None,
) -> dict[str, Any]:
    """POST a chat-completions request and reassemble its SSE stream.

    Mirrors :func:`request_json`'s error classification, redaction, retry, and
    timeout behavior so streaming and non-streaming transport fail identically.
    Reassembles the same raw shape a non-streamed reply would have
    (``choices[0].message.content``/``reasoning_content``/``finish_reason``,
    plus the final chunk's ``usage`` and any llama.cpp ``timings``) so
    ``parse()`` and everything downstream is unaffected by the transport
    choice. An HTTP error response is not a stream: it is read and classified
    exactly as :func:`request_json` does.

    ``timeout`` (``request_timeout_seconds``) is enforced here as a genuine
    wall-clock deadline for the whole call, not merely as urllib's per-socket-
    operation timeout: a server that trickles one byte per read call, each
    well within the socket timeout, could otherwise stream forever. The body
    is read on a background thread into a queue; this thread applies its own
    budget to each ``queue.get`` -- ``first_token_timeout`` before anything at
    all has arrived (``None`` means bounded only by ``timeout``, since prefill
    can legitimately take minutes), then ``stall_timeout`` for every gap after
    that (``None`` disables the inactivity watchdog, again leaving only the
    total deadline) -- and never waits past the remaining total budget either
    way. Any received line counts as activity, including SSE comment/keep-
    alive lines that are not ``data:`` payloads.

    If the server replies to `stream: true` with a non-SSE content type, or
    with an HTTP 400/404/415/422 whose body mentions "stream", this raises
    :class:`StreamingUnsupported` instead of the usual classification so the
    caller can fall back to a single non-streaming retry.
    """
    supplied = {"Accept": "text/event-stream", "User-Agent": DEFAULT_USER_AGENT, **headers}
    redactions = [*(secrets or []), *supplied.values()]
    for attempt in range(attempts):
        try:
            body = json_dumps(payload).encode("utf-8")
            supplied.setdefault("Content-Type", "application/json")
            request = urllib.request.Request(url, data=body, headers=supplied, method="POST")
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            role: str | None = None
            finish_reason: str | None = None
            usage: dict[str, Any] | None = None
            timings: dict[str, Any] | None = None
            start = time.monotonic()
            last_progress = start
            deadline = start + timeout if timeout is not None else None
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = ""
                headers_obj = getattr(response, "headers", None)
                if headers_obj is not None:
                    content_type = (headers_obj.get("Content-Type") or "").lower()
                if content_type and "text/event-stream" not in content_type:
                    preview = _redact(response.read(2_048).decode("utf-8", errors="replace"), redactions)
                    raise StreamingUnsupported(
                        "The server replied to a streaming request with Content-Type "
                        f"{content_type!r} instead of an SSE stream: {preview[:200]!r}",
                        status=getattr(response, "status", None), kind="stream_unsupported",
                        hint="The server may not support server-sent-event streaming for this endpoint.",
                    )
                chunk_queue: queue.Queue = queue.Queue()
                done = object()

                def _pump() -> None:
                    try:
                        for raw_line in response:
                            chunk_queue.put(raw_line)
                    except BaseException as exc:  # noqa: BLE001 - surfaced on the main thread
                        chunk_queue.put(exc)
                    else:
                        chunk_queue.put(done)

                pump = threading.Thread(target=_pump, daemon=True)
                pump.start()
                received_first = False
                try:
                    while True:
                        now = time.monotonic()
                        remaining = deadline - now if deadline is not None else None
                        if remaining is not None and remaining <= 0:
                            raise TimeoutError("the total request timeout elapsed while streaming")
                        budget = stall_timeout if received_first else first_token_timeout
                        if budget is None:
                            wait_for = remaining
                        elif remaining is None:
                            wait_for = budget
                        else:
                            wait_for = min(remaining, budget)
                        try:
                            item = chunk_queue.get(timeout=wait_for) if wait_for is not None else chunk_queue.get()
                        except queue.Empty:
                            if budget is None or (remaining is not None and remaining <= budget):
                                # The total deadline, not the (longer) phase
                                # budget, is what actually elapsed here.
                                raise TimeoutError("the total request timeout elapsed while streaming") from None
                            if received_first:
                                raise EngineError(
                                    f"No streamed data for {budget:g} s; the server appears stalled.",
                                    kind="timeout",
                                    hint="The connection was live but produced no new tokens. Check that "
                                         "the server is still generating; stall_timeout_seconds can be "
                                         "raised or set to off in model settings.",
                                ) from None
                            raise EngineError(
                                f"No streamed data for {budget:g} s during prefill; the server appears stalled.",
                                kind="timeout",
                                hint="The server accepted the request but sent nothing before the "
                                     "first-token timeout. Check that it is still processing the prompt; "
                                     "first_token_timeout_seconds can be raised or set to off to wait for "
                                     "the full request timeout instead.",
                            ) from None
                        if item is done:
                            break
                        if isinstance(item, BaseException):
                            raise item
                        received_first = True
                        raw_line = item
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except ValueError:
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        choices = chunk.get("choices")
                        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                            choice = choices[0]
                            delta = choice.get("delta")
                            if isinstance(delta, dict):
                                if isinstance(delta.get("role"), str):
                                    role = delta["role"]
                                piece = delta.get("content")
                                if isinstance(piece, str):
                                    content_parts.append(piece)
                                reasoning_piece = delta.get("reasoning_content")
                                if not isinstance(reasoning_piece, str):
                                    reasoning_piece = delta.get("reasoning")
                                if isinstance(reasoning_piece, str):
                                    reasoning_parts.append(reasoning_piece)
                            if choice.get("finish_reason"):
                                finish_reason = str(choice["finish_reason"])
                        if isinstance(chunk.get("usage"), dict):
                            usage = chunk["usage"]
                        if isinstance(chunk.get("timings"), dict):
                            timings = chunk["timings"]
                        if progress_callback is not None:
                            now2 = time.monotonic()
                            if now2 - last_progress >= 2.0:
                                last_progress = now2
                                try:
                                    progress_callback({
                                        "generated_chars": sum(len(part) for part in content_parts),
                                        "reasoning_chars": sum(len(part) for part in reasoning_parts),
                                        "elapsed_seconds": now2 - start,
                                    })
                                except Exception:
                                    pass
                finally:
                    # A plain response.close() below would wait for the pump
                    # thread's in-flight blocked read (they share the
                    # buffered reader's lock), silently reintroducing the
                    # stall this function exists to cut short. Shutting down
                    # the raw socket first unblocks that read immediately;
                    # the socket internals are private, so this is
                    # best-effort and never masks the real outcome above.
                    try:
                        response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    pump.join(timeout=0.5)
            message: dict[str, Any] = {"role": role or "assistant", "content": "".join(content_parts)}
            if reasoning_parts:
                message["reasoning_content"] = "".join(reasoning_parts)
            raw: dict[str, Any] = {"choices": [{"message": message, "finish_reason": finish_reason}]}
            if usage is not None:
                raw["usage"] = usage
            if timings is not None:
                raw["timings"] = timings
            return raw
        except EngineError:
            # Includes StreamingUnsupported (a subclass): both are already
            # classified and must reach HTTPMixin._post unchanged.
            raise
        except urllib.error.HTTPError as exc:
            detail = _redact(exc.read(16_384).decode("utf-8", errors="replace"), redactions)
            failure = _server_error(detail, exc.code)
            if exc.code in _STREAM_REJECTION_STATUSES and "stream" in detail.lower():
                raise StreamingUnsupported(str(failure), status=exc.code, kind="stream_unsupported",
                                           hint=failure.hint) from exc
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


def _stream_flag(config: Config) -> dict[str, Any]:
    """The `stream`/`stream_options` fields for a chat-completions payload."""
    if config.stream_responses:
        return {"stream": True, "stream_options": {"include_usage": True}}
    return {"stream": False}
