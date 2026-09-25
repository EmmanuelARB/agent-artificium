"""The registry of every life-loop record kind (:meth:`Records.life`).

``display.py`` renders one event per line (or a small block) for both the
foreground :class:`~artificium.records.Console` and the ``watch`` command.
Every kind that any module in this package emits through ``records.life(...)``
must be listed here with the category ``watch --only CATEGORY`` filters on.
Unknown kinds (a newer harness version, or a workspace overlay emitting its
own kind) fall back to :data:`CATEGORY_OTHER` and a generic renderer instead
of crashing or being silently dropped.
"""
from __future__ import annotations

# --- categories -------------------------------------------------------
#
# These are the values accepted by ``watch --only``.

CATEGORY_TURNS = "turns"
CATEGORY_ENGINE = "engine"
CATEGORY_TOOLS = "tools"
CATEGORY_THOUGHTS = "thoughts"
CATEGORY_CONTEXT = "context"
CATEGORY_MEMORY = "memory"
CATEGORY_NOTIFICATIONS = "notifications"
CATEGORY_RECOVERY = "recovery"
CATEGORY_OTHER = "other"

ALL_CATEGORIES = (
    CATEGORY_TURNS,
    CATEGORY_ENGINE,
    CATEGORY_TOOLS,
    CATEGORY_THOUGHTS,
    CATEGORY_CONTEXT,
    CATEGORY_MEMORY,
    CATEGORY_NOTIFICATIONS,
    CATEGORY_RECOVERY,
    CATEGORY_OTHER,
)

# --- kinds --------------------------------------------------------------
#
# One constant per string passed to ``records.life("...")`` anywhere in
# ``artificium/*.py``. ``tests/test_display_watch.py`` greps the source for
# every ``records.life("kind"`` / ``.life(\n    "kind"`` call site and asserts
# it appears here.

KIND_TURN_STARTED = "turn_started"
KIND_TURN_COMPLETED = "turn_completed"
KIND_TURN_INTERRUPTED = "turn_interrupted"

KIND_CONTEXT_USAGE = "context_usage"
# Rendered for old logs only; no current code emits it.
KIND_CONTEXT_COMPACTED = "context_compacted"

KIND_ENGINE_REQUEST = "engine_request"
KIND_ENGINE_WAITING = "engine_waiting"
KIND_ENGINE_RESPONSE = "engine_response"
KIND_ENGINE_REQUEST_FAILED = "engine_request_failed"
KIND_ENGINE_REQUEST_CANCELLED = "engine_request_cancelled"
KIND_ENGINE_BLOCKED = "engine_blocked"
# Streaming progress; not emitted by every engine adapter.
KIND_ENGINE_PROGRESS = "engine_progress"
KIND_PROVIDER_REASONING = "provider_reasoning"

KIND_THOUGHT = "thought"
KIND_LIFE_LOOP_OUTPUT = "life_loop_output"

KIND_TOOL_CALL = "tool_call"
KIND_TOOL_RESULT = "tool_result"
KIND_TOOL_REQUEST_REJECTED = "tool_request_rejected"

KIND_WORKING_MEMORY_OFFLOADED = "working_memory_offloaded"
KIND_ATTENTION_CONTEXT_COMPRESSED = "attention_context_compressed"
KIND_MEMORY_SAVED = "memory_saved"
KIND_PINNED_MIND_REFRESHED = "pinned_mind_refreshed"

KIND_GUIDANCE_NOTIFICATION = "guidance_notification"

KIND_REQUEST_REPAIR_STARTED = "request_repair_started"
KIND_REQUEST_REPAIR_FAILED = "request_repair_failed"
KIND_REQUEST_REPAIR_READY = "request_repair_ready"

EVENT_CATEGORIES: dict[str, str] = {
    KIND_TURN_STARTED: CATEGORY_TURNS,
    KIND_TURN_COMPLETED: CATEGORY_TURNS,
    KIND_TURN_INTERRUPTED: CATEGORY_TURNS,
    KIND_CONTEXT_USAGE: CATEGORY_CONTEXT,
    KIND_CONTEXT_COMPACTED: CATEGORY_CONTEXT,
    KIND_ENGINE_REQUEST: CATEGORY_ENGINE,
    KIND_ENGINE_WAITING: CATEGORY_ENGINE,
    KIND_ENGINE_RESPONSE: CATEGORY_ENGINE,
    KIND_ENGINE_REQUEST_FAILED: CATEGORY_ENGINE,
    KIND_ENGINE_REQUEST_CANCELLED: CATEGORY_ENGINE,
    KIND_ENGINE_BLOCKED: CATEGORY_ENGINE,
    KIND_ENGINE_PROGRESS: CATEGORY_ENGINE,
    KIND_PROVIDER_REASONING: CATEGORY_ENGINE,
    KIND_THOUGHT: CATEGORY_THOUGHTS,
    KIND_LIFE_LOOP_OUTPUT: CATEGORY_THOUGHTS,
    KIND_TOOL_CALL: CATEGORY_TOOLS,
    KIND_TOOL_RESULT: CATEGORY_TOOLS,
    KIND_TOOL_REQUEST_REJECTED: CATEGORY_TOOLS,
    KIND_WORKING_MEMORY_OFFLOADED: CATEGORY_MEMORY,
    KIND_ATTENTION_CONTEXT_COMPRESSED: CATEGORY_MEMORY,
    KIND_MEMORY_SAVED: CATEGORY_MEMORY,
    KIND_PINNED_MIND_REFRESHED: CATEGORY_MEMORY,
    KIND_GUIDANCE_NOTIFICATION: CATEGORY_NOTIFICATIONS,
    KIND_REQUEST_REPAIR_STARTED: CATEGORY_RECOVERY,
    KIND_REQUEST_REPAIR_FAILED: CATEGORY_RECOVERY,
    KIND_REQUEST_REPAIR_READY: CATEGORY_RECOVERY,
}

# Kinds whose most useful rendering is a single line that updates itself with
# a carriage return while streaming, instead of scrolling history.
STREAMING_KINDS = frozenset({KIND_ENGINE_PROGRESS})

# Kinds hidden by default (opt-in via a watch flag) because they are verbose
# or duplicate what a neighbouring event already reports.
HIDDEN_BY_DEFAULT_KINDS = frozenset({KIND_PROVIDER_REASONING})


def category_for(kind: str) -> str:
    """The ``watch --only`` bucket for one life-record kind.

    Unknown kinds (future harness versions, workspace overlays) fall back to
    :data:`CATEGORY_OTHER` rather than raising - the renderer must never
    crash or silently drop a record it does not recognize.
    """

    return EVENT_CATEGORIES.get(kind, CATEGORY_OTHER)


def is_known_kind(kind: str) -> bool:
    return kind in EVENT_CATEGORIES
