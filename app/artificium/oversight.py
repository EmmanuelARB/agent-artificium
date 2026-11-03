"""Checks the agent does not make on itself.

Two mechanisms share this module because both are a fresh-context model call
the harness starts on its own, and both deliver their result as an ordinary
event from the sender ``harness`` (which can never be quoted as a directive):

* **Decision defaults.** A decision sent with a default is final after a
  deadline only if a reviewer who sees the entity's own words finds that the
  default contradicts nothing, can be undone, and needs nothing from outside
  the workspace. A default that fails the check is not applied.
* **Evaluation.** Once per interval, and sooner when the stalled-blocker
  notice fires, a reviewer with none of the agent's context judges whether the
  work is advancing toward what the entity asked, from evidence the agent did
  not write as well as its own claims.

Neither acts on the agent's behalf beyond delivering the event: the agent
remains the one who decides what to do next.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .blockers import BlockerTracker
from .config import Config
from .directives import SYSTEM_SENDERS, DirectiveStore
from .filesystem import Paths, atomic_write_json, read_json
from .interactions import InteractionStore
from .prompts import PromptPack
from .records import Records


SYSTEM_SENDER = "harness"
OVERSIGHT_INTERACTION = "oversight"
# Interactions the harness itself writes to; a decision is never sent there.
INTERNAL_INTERACTIONS = frozenset({"scheduler", OVERSIGHT_INTERACTION})

CHECK_RETRY_SECONDS = 15 * 60
CHECK_ATTEMPTS = 3
EVALUATION_RETRY_SECONDS = 30 * 60
# A stalled-blocker notice brings the evaluation forward, but not more often
# than a quarter of the interval, so a long streak does not become a loop.
STALL_GAP_FRACTION = 0.25
# Without a state file the evaluation waits until the agent has this many
# checkpoints to judge; a fresh install has nothing to evaluate yet.
FIRST_EVALUATION_CHECKPOINTS = 3

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_VERDICT = re.compile(r"VERDICT\s*:\s*\**\s*([A-Za-z][A-Za-z -]*)", re.IGNORECASE)
_REASON = re.compile(r"REASONS?\s*:\s*\**\s*(.*)", re.IGNORECASE | re.DOTALL)


def _parse(value: Any) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _stamp(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


def _clip(text: Any, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return (f"{text[:head]}\n[... {len(text) - limit} characters omitted ...]\n"
            f"{text[-(limit - head):]}")


def default_deadline(now: dt.datetime, hours: float) -> str:
    return _stamp(now + dt.timedelta(hours=hours))


def workspace_signals(paths: Paths, *, now: dt.datetime | None = None,
                      timeout: float = 15.0) -> list[str]:
    """What the workspace's git repositories show, measured by the harness.

    Looks at the workspace root and its direct subdirectories. A tool that is
    missing, slow, or refuses a repository only removes that repository from
    the picture.
    """

    now = now or dt.datetime.now(dt.timezone.utc)
    candidates = [paths.root]
    try:
        candidates += sorted(item for item in paths.root.iterdir()
                             if item.is_dir() and item.name not in {"logs", "overrides"})
    except OSError:
        pass
    lines: list[str] = []
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(root: Path, *arguments: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-c", "safe.directory=*", *arguments], cwd=root, env=environment,
                capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    for root in candidates:
        if not (root / ".git").exists():
            continue
        last = git(root, "log", "-1", "--format=%cI")
        if not last or not last.strip():
            continue
        recent = (git(root, "rev-list", "--count", "--since=24 hours ago", "HEAD")
                  or "?").strip()
        status = git(root, "status", "--porcelain", "-uno") or ""
        changed = [line[3:].strip() for line in status.splitlines() if line[3:].strip()]
        oldest: float | None = None
        for name in changed:
            try:
                stamp = (root / name).stat().st_mtime
            except OSError:
                continue
            oldest = stamp if oldest is None else min(oldest, stamp)
        committed = _parse(last.strip())
        ago = (f" ({(now - committed).total_seconds() / 3600:.1f} h ago)"
               if committed else "")
        lines.append(f"{root.name}: last commit {last.strip()}{ago}; "
                     f"commits in the last 24 h: {recent}")
        if changed:
            age = ("" if oldest is None else
                   f", oldest edit {(now.timestamp() - oldest) / 3600:.1f} h ago")
            lines.append(f"{root.name}: {len(changed)} tracked files modified and not "
                         f"committed{age}")
        else:
            lines.append(f"{root.name}: no uncommitted changes to tracked files")
    return lines or ["no git repository found in the workspace"]


class Oversight:
    """Runs the checks that are due; ``tick`` is called from the life-loop.

    ``complete`` takes chat messages and returns ``(request_id, text)``; it is
    the runtime's fresh-context request, so it is budgeted, logged and counted
    like any inference but shares none of the agent's working memory.
    """

    def __init__(self, paths: Paths, config: Config, records: Records,
                 interactions: InteractionStore, prompts: PromptPack,
                 complete: Callable[[list[dict[str, Any]]], tuple[str, str]],
                 *, min_tick_seconds: float = 60.0):
        self.paths = paths
        self.config = config
        self.records = records
        self.interactions = interactions
        self.prompts = prompts
        self.complete = complete
        self.min_tick_seconds = min_tick_seconds
        self._last_tick = float("-inf")

    # -- state --------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.paths.runtime / "oversight.json"

    def _state(self) -> dict[str, Any]:
        value = read_json(self.state_path, {})
        value = value if isinstance(value, dict) else {}
        for key in ("defaults", "evaluation"):
            if not isinstance(value.get(key), dict):
                value[key] = {}
        return value

    def _save(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_path, state)

    # -- the tick -----------------------------------------------------------

    def tick(self, now: dt.datetime | None = None, *, force: bool = False) -> list[str]:
        """Do whatever is due; returns short descriptions for the console."""

        if not force and time.monotonic() - self._last_tick < self.min_tick_seconds:
            return []
        self._last_tick = time.monotonic()
        now = now or dt.datetime.now(dt.timezone.utc)
        done: list[str] = []
        for step in (self.process_defaults, self.maybe_evaluate):
            try:
                done += step(now)
            except Exception as exc:  # a failed check must never stop the life-loop
                self.records.emit("oversight_failed", step=getattr(step, "__name__", "step"),
                                  error=f"{type(exc).__name__}: {exc}")
        return done

    def pending_deadlines(self) -> list[str]:
        """Decision-default deadlines not yet resolved. They wake a sleeping
        agent, so a sleep that has one needs no separate idle fallback."""

        state = self._state()["defaults"]
        return [str(item["default_after"]) for item in self.open_decisions()
                if item.get("default") and item.get("default_after")
                and state.get(item["id"], {}).get("status") not in ("applied", "held")]

    # -- decision defaults --------------------------------------------------

    def open_decisions(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for item in self.interactions.list_interactions():
            identifier = str(item.get("id") or "")
            if identifier and identifier not in INTERNAL_INTERACTIONS:
                found.extend(self.interactions.open_decisions(identifier))
        return found

    def process_defaults(self, now: dt.datetime) -> list[str]:
        done: list[str] = []
        for decision in self.open_decisions():
            deadline = _parse(decision.get("default_after"))
            if not decision.get("default") or deadline is None or deadline > now:
                continue
            entry = self._state()["defaults"].get(decision["id"], {})
            if entry.get("status") in ("applied", "held"):
                continue
            retry = _parse(entry.get("next_try"))
            if retry is not None and retry > now:
                continue
            done.append(self._resolve_default(decision, entry, now))
        return done

    def _resolve_default(self, decision: dict[str, Any], entry: dict[str, Any],
                         now: dt.datetime) -> str:
        identifier = decision["id"]
        attempts = int(entry.get("attempts", 0)) + 1
        try:
            verdict, reason, request_id = self._check_default(decision)
        except Exception as exc:
            reason = f"the check could not run ({type(exc).__name__}: {exc})"
            if attempts < CHECK_ATTEMPTS:
                self._remember_default(identifier, {
                    "status": "retry", "attempts": attempts, "reason": reason,
                    "next_try": _stamp(now + dt.timedelta(seconds=CHECK_RETRY_SECONDS))})
                self.records.emit("decision_default_check_failed", decision_id=identifier,
                                  attempts=attempts, error=reason)
                return f"default check for {identifier} failed ({attempts}); will retry"
            verdict, request_id = "hold", None
        applied = verdict == "apply"
        self.interactions.add_event(
            str(decision["interaction_id"]), sender=SYSTEM_SENDER, direction="inbound",
            kind="default_applied" if applied else "default_held",
            content=self.prompts.event(
                "default_applied" if applied else "default_held",
                decision_id=identifier, hours=self._hours_open(decision, now),
                default=decision["default"], question=_clip(decision.get("content"), 600),
                reason=reason or "no reason given"),
            # A held default is not an answer: the question stays open.
            in_reply_to=identifier if applied else None,
            recipient=self.config.instance_id)
        status = "applied" if applied else "held"
        self._remember_default(identifier, {
            "status": status, "attempts": attempts, "decided_at": _stamp(now),
            "reason": reason, "request_id": request_id})
        self.records.emit(f"decision_default_{status}", decision_id=identifier,
                          default=decision["default"], reason=reason, request_id=request_id)
        return f"default for {identifier} {status}"

    def _remember_default(self, identifier: str, entry: dict[str, Any]) -> None:
        state = self._state()
        state["defaults"][identifier] = entry
        self._save(state)

    @staticmethod
    def _hours_open(decision: dict[str, Any], now: dt.datetime) -> str:
        asked = _parse(decision.get("created_at"))
        return f"{(now - asked).total_seconds() / 3600:.1f}" if asked else "several"

    def _entity_words(self, interaction_id: str, limit: int = 5) -> str:
        """The entity's recent messages in this interaction, verbatim: the
        constraints it voiced in conversation are not all recorded as
        directives."""

        spoken = [event for event in self.interactions.events(interaction_id)
                  if event.get("direction") == "inbound"
                  and event.get("sender") not in SYSTEM_SENDERS]
        lines = [f"[{event.get('created_at')}] {event.get('sender')}: "
                 f"{_clip(event.get('content'), 1_500)}" for event in spoken[-limit:]]
        return "\n\n".join(lines) or "none"

    def _check_default(self, decision: dict[str, Any]) -> tuple[str, str, str]:
        directives = DirectiveStore(self.paths, self.records).render()
        options = decision.get("options") or []
        messages = [
            {"role": "system", "content": self.prompts.runtime("default_check_system")},
            {"role": "user", "content": self.prompts.runtime(
                "default_check_request",
                question=_clip(decision.get("content"), 3_000),
                options="; ".join(str(item) for item in options) or "none (free answer)",
                default=str(decision["default"]),
                directives=directives or "none recorded",
                entity_words=self._entity_words(str(decision["interaction_id"])),
            )},
        ]
        request_id, text = self.complete(messages)
        text = _THINK.sub("", text).strip()
        match = _VERDICT.search(text)
        word = match.group(1).strip().lower() if match else ""
        found = _REASON.search(text)
        reason = _clip(found.group(1) if found else text, 1_000)
        if word.startswith("apply"):
            return "apply", reason, request_id
        if not word.startswith("hold"):
            # An unclear answer is never read as permission.
            reason = f"the reviewer's answer had no clear verdict: {_clip(text, 300)}"
        return "hold", reason, request_id

    # -- evaluation ---------------------------------------------------------

    def evaluation_due(self, now: dt.datetime) -> str | None:
        """Why an evaluation should run now: "interval", "stalled", or None."""

        hours = float(self.config.evaluation_interval_hours or 0)
        if hours <= 0:
            return None
        evaluation = self._state()["evaluation"]
        attempted = _parse(evaluation.get("attempt_at"))
        if attempted and (now - attempted).total_seconds() < EVALUATION_RETRY_SECONDS:
            return None
        history = read_json(BlockerTracker(self.paths, self.records).path, {})
        history = history if isinstance(history, dict) else {}
        checkpoints = [item for item in history.get("history") or []
                       if isinstance(item, dict)]
        last = _parse(evaluation.get("last_at"))
        if last is None:
            return "interval" if len(checkpoints) >= FIRST_EVALUATION_CHECKPOINTS else None
        since = (now - last).total_seconds() / 3600
        if since >= hours:
            return "interval"
        stalled = _parse(history.get("stalled_at"))
        if stalled and stalled > last and since >= hours * STALL_GAP_FRACTION:
            return "stalled"
        return None

    def maybe_evaluate(self, now: dt.datetime) -> list[str]:
        reason = self.evaluation_due(now)
        return [] if reason is None else [self.evaluate(now, reason=reason)]

    def _ask(self) -> str:
        """What the entity asked: its recorded directives, then its first and
        latest messages (the first one usually holds the brief)."""

        directives = DirectiveStore(self.paths, self.records).render()
        spoken: list[dict[str, Any]] = []
        for item in self.interactions.list_interactions():
            identifier = str(item.get("id") or "")
            if identifier in INTERNAL_INTERACTIONS:
                continue
            spoken += [event for event in self.interactions.events(identifier)
                       if event.get("direction") == "inbound"
                       and event.get("sender") not in SYSTEM_SENDERS]
        spoken.sort(key=lambda event: str(event.get("created_at") or ""))
        parts = [f"STANDING DIRECTIVES (verbatim)\n{directives or 'none recorded'}"]
        if spoken:
            parts.append("FIRST MESSAGE FROM AN ENTITY\n" + _clip(spoken[0].get("content"), 4_000))
            if len(spoken) > 1:
                parts.append("LATEST MESSAGES FROM ENTITIES\n" + "\n\n".join(
                    f"[{event.get('created_at')}] {event.get('sender')}: "
                    f"{_clip(event.get('content'), 1_200)}" for event in spoken[-4:]))
        return "\n\n".join(parts)

    def _claims(self) -> str:
        history = read_json(BlockerTracker(self.paths, self.records).path, {})
        entries = ([item for item in history.get("history") or [] if isinstance(item, dict)]
                   if isinstance(history, dict) else [])
        if not entries:
            return "no working-memory-offload checkpoints yet"
        rows = [f"{item.get('at')} | blocker: {item.get('blocker') or '-'} | "
                f"progress label: {item.get('objective_progress') or '-'} | "
                f"measure: {item.get('measure') or '-'}" for item in entries[-12:]]
        parts = ["CHECKPOINT HISTORY (the agent's own labels; last 12)\n" + "\n".join(rows)]
        for item in entries[-2:]:
            path = item.get("checkpoint_path")
            if not path:
                continue
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(f"CHECKPOINT TEXT written {item.get('at')}\n" + _clip(text, 3_500))
        return "\n\n".join(parts)

    def _situation(self, now: dt.datetime) -> str:
        lines = workspace_signals(self.paths, now=now)
        decisions = self.open_decisions()
        lines.append(f"open decisions waiting for an entity: {len(decisions)}")
        for item in decisions[:3]:
            lines.append(f"  asked {item.get('created_at')}: {_clip(item.get('content'), 200)}")
        outbound = 0
        cutoff = now - dt.timedelta(hours=24)
        for item in self.interactions.list_interactions():
            identifier = str(item.get("id") or "")
            if identifier in INTERNAL_INTERACTIONS:
                continue
            for event in self.interactions.events(identifier):
                made = _parse(event.get("created_at"))
                if event.get("direction") == "outbound" and made and made >= cutoff:
                    outbound += 1
        lines.append(f"messages the agent sent to entities in the last 24 h: {outbound}")
        previous = self._state()["evaluation"]
        if previous.get("last_at"):
            lines.append(f"previous evaluation: {previous['last_at']} "
                         f"(verdict: {previous.get('verdict') or 'unknown'})")
        return "\n".join(f"- {line}" if not line.startswith("  ") else line
                         for line in lines)

    def evaluate(self, now: dt.datetime, *, reason: str = "interval") -> str:
        messages = [
            {"role": "system", "content": self.prompts.runtime("evaluation_system")},
            {"role": "user", "content": self.prompts.runtime(
                "evaluation_request", ask=self._ask(), claims=self._claims(),
                measured=self._situation(now), trigger=reason)},
        ]
        state = self._state()
        # Noted before the call: a failing model is retried after a pause, not
        # on every tick.
        state["evaluation"]["attempt_at"] = _stamp(now)
        self._save(state)
        request_id, text = self.complete(messages)
        review = _THINK.sub("", text).strip() or "(the evaluator returned no text)"
        match = _VERDICT.search(review)
        verdict = match.group(1).strip().lower()[:40] if match else "unknown"
        state = self._state()
        state["evaluation"].update(
            last_at=_stamp(now), trigger=reason, verdict=verdict, request_id=request_id,
            count=int(state["evaluation"].get("count", 0)) + 1)
        self._save(state)
        event, path = self.interactions.add_event(
            OVERSIGHT_INTERACTION, name="Oversight", sender=SYSTEM_SENDER,
            direction="inbound", kind="evaluation", recipient=self.config.instance_id,
            content=self.prompts.event("evaluation_notice", trigger=reason,
                                       verdict=verdict, review=review))
        self.records.emit("evaluation_completed", trigger=reason, verdict=verdict,
                          request_id=request_id, event_id=event["id"], path=str(path))
        self.records.life("evaluation_completed", trigger=reason, verdict=verdict,
                          preview=" ".join(review.split())[:300])
        return f"evaluation ({reason}): {verdict}"
