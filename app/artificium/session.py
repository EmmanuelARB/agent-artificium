"""Conversation identity for providers that ask a client for a session ID.

Some gateways route and cache by a client-supplied session ID (OpenCode Go
reads the ``x-opencode-session`` header). What counts as one *conversation*
for Artificium:

* The life-loop's working context is one conversation. Its ID is stable across
  turns and across restarts, because the requests really do share a growing
  prefix. When the working context is replaced rather than extended (a
  working-memory offload, a request repair, an attention compression, an
  operator edit: any rebuild), the cached prefix is gone and the conversation
  gets a new ID.
* Every fresh-context request (``request_review``, ``consult``, the
  evaluation, the decision-default check, the emergency summary, a
  connection check) shares nothing with the working context and is its own
  conversation with its own ID.

An ID is ``ses_`` followed by random hex. It is opaque: it never carries the
API key, the instance name, a path, or anything the agent wrote.
"""
from __future__ import annotations

import hashlib
import re
import secrets
from pathlib import Path
from typing import Any

from .filesystem import atomic_write_json, json_dumps, read_json, utc_now

SESSION_HEADER = "x-opencode-session"
SESSION_PREFIX = "ses_"
_ID = re.compile(r"ses_[0-9a-f]{26}")


def new_session_id() -> str:
    return SESSION_PREFIX + secrets.token_hex(13)


def _valid(value: Any) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


class LifeLoopSession:
    """The life-loop's session ID, kept in runtime state across restarts.

    The state is the ID plus a fingerprint of the working context it was last
    used with (message count and a hash of those messages). The context is the
    same conversation while it still *starts with* those messages; anything
    else means it was replaced, so a new ID is issued. Comparing contents,
    rather than hooking every place that rewrites the context, also catches
    rebuilds nobody thought to announce.
    """

    def __init__(self, path: Path):
        self.path = path
        self._state: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any] | None:
        if self._state is None:
            value = read_json(self.path, None)
            if (isinstance(value, dict) and _valid(value.get("session_id"))
                    and isinstance(value.get("messages"), int) and value["messages"] >= 0
                    and isinstance(value.get("digest"), str)):
                self._state = value
        return self._state

    def current(self, history: list[dict[str, Any]]) -> str:
        """The ID for a request whose working-context history is ``history``."""
        state = self._load()
        known = state["messages"] if state else None
        digest = hashlib.sha256()
        prefix = digest.copy().hexdigest() if known == 0 else None
        for index, message in enumerate(history, 1):
            digest.update(json_dumps(message).encode("utf-8"))
            if index == known:
                prefix = digest.copy().hexdigest()
        same = state is not None and len(history) >= known and prefix == state["digest"]
        session_id = state["session_id"] if same else new_session_id()
        fingerprint = digest.hexdigest()
        if not same or (len(history), fingerprint) != (state["messages"], state["digest"]):
            self._state = {"session_id": session_id, "messages": len(history),
                           "digest": fingerprint, "updated_at": utc_now()}
            try:
                atomic_write_json(self.path, self._state)
            except OSError:
                pass  # an unsaved ID only costs one cache miss after a restart
        return session_id
