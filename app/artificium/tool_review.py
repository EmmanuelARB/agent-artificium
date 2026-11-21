from __future__ import annotations

import contextlib
import re
from typing import Any

from .directives import DirectiveStore
from .filesystem import json_dumps, read_json, atomic_write_json
from .life_loop import ToolIntent, parse_life_loop_output


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clip(text: Any, limit: int) -> str:
    """Head and tail of a long text, so neither the opening nor the end is lost."""
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return (f"{text[:head]}\n[... {len(text) - limit} characters omitted ...]\n"
            f"{text[-(limit - head):]}")


# One label per line, in any case, with or without Markdown emphasis; a
# section runs to the next label.
_REVIEW_LABEL = re.compile(
    r"^[ \t>*#_-]*(NECESSARY|FRAMING|POINTS|DEFAULT|REASONS?)[ \t]*\**[ \t]*:[ \t]*\**[ \t]*",
    re.IGNORECASE | re.MULTILINE)
_NO_DEFAULT = re.compile(r"^(none\b(?!\s+of\b)|no (safe )?default\b|n/a\b|-+$)", re.IGNORECASE)


def parse_decision_review(text: str, options: list[str] | None = None) -> dict[str, Any] | None:
    """The reviewer's fixed answer form as fields, or None when it has no
    clear NECESSARY and FRAMING (an unclear answer is never read as a verdict).

    ``default`` is None when the reviewer found no safe default; when it names
    one of ``options`` it is returned written exactly as offered.
    """

    text = _THINK.sub("", text or "")
    found = list(_REVIEW_LABEL.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(found):
        end = found[index + 1].start() if index + 1 < len(found) else len(text)
        label = match.group(1).upper()
        sections.setdefault("REASONS" if label.startswith("REASON") else label,
                            text[match.end():end].strip())
    necessary = sections.get("NECESSARY", "").lower()
    framing = sections.get("FRAMING", "").lower()
    necessary = "yes" if necessary.startswith("yes") else "no" if necessary.startswith("no") else ""
    framing = ("ok" if framing.startswith("ok") else "revise" if framing.startswith("revise") else "")
    if not necessary or not framing:
        return None
    points = [line.strip(" \t-*•").strip() for line in sections.get("POINTS", "").splitlines()]
    points = [line for line in points if line and line.lower().rstrip(".") != "none"]
    default = sections.get("DEFAULT", "").strip().strip("`*\"' ").strip()
    if not default or _NO_DEFAULT.match(default):
        default = None
    else:
        exact = {item.strip().lower(): item.strip() for item in options or []}
        default = exact.get(default.lower().rstrip("."), _clip(default, 500))
    return {"necessary": necessary, "framing": framing, "points": points,
            "default": default, "reasons": _clip(sections.get("REASONS", ""), 1_500)}


# What the oversight model may call while it checks evidence: the registry's
# own read-only handlers, nothing that writes, runs, sends or sleeps.
OVERSIGHT_READ_TOOLS = ("read_file", "list_directory", "search_memory")
# Files the reviewer never reads, wherever they are (the operator's settings
# and the API key); the file tools themselves allow them.
_WITHHELD_NAMES = frozenset({".secrets.json", "config.json"})
# Share of the oversight window the conversation may fill before it must answer.
_WINDOW_SHARE = 0.8


class ReviewToolsMixin:
    """Fresh-context requests to the oversight model: ``request_review``,
    ``consult``, and the review a decision passes before it is sent.

    Mixed into :class:`ToolRegistry`. The runtime attaches ``reviewer``,
    ``consultant`` and ``decision_reviewer``, callables taking chat messages and
    returning ``(request_id, text)``, plus ``entity_ask`` (what the entity
    asked, for consult and the decision review), ``oversight_budget`` (the
    Config whose context window bounds what is sent) and
    ``oversight_connection_label``. Without them (a registry built outside the
    runtime) the tools report that they are unavailable, and a decision is sent
    without review. ``oversight_native_tools`` (does the oversight connection's
    adapter take native tool definitions) and ``oversight_conversation`` (a
    context manager that makes the requests inside it one conversation) are
    optional: without them the reviewer uses the text protocol and each request
    is its own conversation.
    """

    def _evidence_sections(
        self, paths: list[str], purpose: str,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Bounded contents of evidence files, sized to the window of the model
        that will read them."""
        budget_config = getattr(self, "oversight_budget", None)
        config = budget_config() if callable(budget_config) else self.config
        budget = int(config.working_memory_limit * config.chars_per_token * 0.4)
        share = budget // max(1, len(paths))
        sections: list[str] = []
        evidence: list[dict[str, Any]] = []
        for supplied in paths:
            path = self._resolve(supplied)
            if not path.is_file():
                raise FileNotFoundError(
                    f"evidence path {path} is not a file; pass files whose content "
                    f"{purpose}"
                )
            text = path.read_text(encoding="utf-8", errors="replace")
            truncated = len(text) > share
            if truncated:
                head = text[: share // 2]
                tail = text[-(share - share // 2):]
                text = (f"{head}\n[... {len(text) - share} characters omitted ...]\n"
                        f"{tail}")
            sections.append(f"=== {path} ===\n{text}")
            evidence.append({"path": str(path), "truncated": truncated})
        return sections, evidence

    # -- the reviewer's read-only tools -------------------------------------

    def _oversight_rounds(self) -> int:
        try:
            return max(0, int(getattr(self.config, "oversight_tool_rounds", 4)))
        except (TypeError, ValueError):
            return 4

    def _oversight_tool_schemas(self) -> list[dict[str, Any]]:
        from .tool_schemas import native_tool_schemas

        schemas = native_tool_schemas(type(self), self.prompts.tool_catalog())
        return [item for item in schemas if item["function"]["name"] in OVERSIGHT_READ_TOOLS]

    def _withheld(self, path: Any) -> bool:
        """Whether ``path`` is a file the reviewer never reads (see _WITHHELD_NAMES)."""
        try:
            target = self._resolve(str(path))
            if target.name in _WITHHELD_NAMES:
                return True
            if target.exists():
                return any(own.exists() and target.samefile(own)
                           for own in (self.paths.secrets, self.paths.config))
        except (OSError, ValueError):
            return True
        return False

    def _reviewer_tool(self, intent: ToolIntent) -> dict[str, Any]:
        """One call of the reviewer, through the registry's own handler.

        Anything but the read-only tools is refused, and so is the operator's
        configuration and the API key. Errors are returned to the reviewer as
        results, never raised.
        """
        name = intent.name
        if intent.parse_error:
            return {"status": "error", "summary": intent.parse_error}
        if name not in OVERSIGHT_READ_TOOLS:
            return {"status": "error", "summary": (
                f"`{name}` is not available to you and was not run; you can call only "
                + ", ".join(OVERSIGHT_READ_TOOLS) + ".")}
        problem = self.validate(intent)
        if problem is not None:
            return {"status": "error", "summary": problem["error"],
                    "accepted_arguments": problem["accepted_arguments"]}
        if name != "search_memory" and "path" in intent.arguments and self._withheld(
                intent.arguments["path"]):
            return {"status": "error", "summary": (
                f"{intent.arguments['path']} is withheld from reviewers (settings and "
                "credentials) and was not read; check something else.")}
        # read_file notes which paths the agent read only in part, for the
        # agent's own overwrite guard; the reviewer's reads are not the agent's.
        markers = read_json(self.paths.partial_reads_state, None)
        try:
            result = self._functions[name](**intent.arguments)
        except Exception as exc:
            return {"status": "error", "summary": f"{type(exc).__name__}: {exc}"}
        finally:
            if read_json(self.paths.partial_reads_state, None) != markers:
                try:
                    if markers is None:
                        self.paths.partial_reads_state.unlink(missing_ok=True)
                    else:
                        atomic_write_json(self.paths.partial_reads_state, markers)
                except OSError:
                    pass
        if not isinstance(result, dict):
            return {"status": "ok", "value": result}
        if name == "search_memory":
            result["results"] = [item for item in result.get("results") or []
                                 if not self._withheld(item.get("path", ""))]
        return result

    def _reviewer_system(self, system: str, rounds: int, native: bool) -> dict[str, Any]:
        text = system + "\n\n" + self.prompts.runtime("oversight_tools", rounds=rounds)
        if not native:
            text += self.prompts.runtime("oversight_tools_text")
        message: dict[str, Any] = {"role": "system", "content": text}
        if native:
            message["_artificium"] = {"kind": "system", "native": True,
                                      "tools": self._oversight_tool_schemas()}
        return message

    def _oversight_exchange(
        self, tool: str, requester: Any, messages: list[dict[str, Any]],
        checked: dict[str, Any],
    ) -> tuple[str, str]:
        """The request to the oversight model, with the read-only loop when allowed.

        ``checked`` is filled in as the loop runs, so a failure part-way still
        tells the agent what was read. Each round is an ordinary oversight
        request; the history is kept as text with the same markers the
        life-loop uses, so the native protocol replays it as structured calls
        (engine_native) and the text protocol sends it as written.
        """
        rounds = self._oversight_rounds()
        checked["tool_rounds_allowed"] = rounds
        if rounds <= 0 or not messages or messages[0].get("role") != "system":
            request_id, text = requester(messages)
            checked["requests"] += 1
            checked["request_ids"].append(request_id)
            return request_id, text
        native_asked = getattr(self, "oversight_native_tools", None)
        native = bool(native_asked()) if callable(native_asked) else False
        budget_config = getattr(self, "oversight_budget", None)
        window = budget_config() if callable(budget_config) else self.config
        limit = int(window.working_memory_limit * window.chars_per_token * _WINDOW_SHARE)
        role = getattr(self.config, "runtime_message_role", "user")
        opened = [self._reviewer_system(messages[0]["content"], rounds, native), *messages[1:]]
        # The final request keeps the same system text (a changed prefix would
        # lose the provider's cache) but declares no tools.
        plain = {**opened[0]}
        plain.pop("_artificium", None)
        history = list(opened)
        conversation = getattr(self, "oversight_conversation", None)
        reason = ""
        with conversation() if callable(conversation) else contextlib.nullcontext():
            while True:
                final = bool(reason) or checked["tool_rounds"] >= rounds
                if final and not reason:
                    reason = "You have used your tool rounds."
                request = history if not final else [
                    plain, *history[1:],
                    {"role": role, "content": self.prompts.runtime(
                        "oversight_answer_now", reason=reason)}]
                request_id, text = requester(request)
                checked["requests"] += 1
                checked["request_ids"].append(request_id)
                parsed = parse_life_loop_output(text)
                if final:
                    return request_id, parsed.visible if parsed.tools else text
                if not parsed.tools:
                    return request_id, text
                checked["tool_rounds"] += 1
                reason = self._reviewer_round(
                    tool, request_id, text, parsed.tools, history, checked, role, limit)

    def _reviewer_round(
        self, tool: str, request_id: str, text: str, intents: list[ToolIntent],
        history: list[dict[str, Any]], checked: dict[str, Any], role: str, limit: int,
    ) -> str:
        """Run one response's calls in order and extend the history with them.

        Returns why the next request must be the answer (the results would not
        fit the window), or an empty string to go on.
        """
        number = checked["tool_rounds"]
        cap = self.config.max_tool_output_chars
        results: list[tuple[ToolIntent, str, str]] = []
        for intent in intents:
            result = self._reviewer_tool(intent)
            body = json_dumps(result, pretty=True)
            if len(body) > cap:
                body = _clip(body, max(200, cap - 100))
            results.append((intent, body, str(result.get("status", "ok"))))
        used = sum(len(str(item.get("content") or "")) for item in history)
        full = used + len(text) + sum(len(body) for _, body, _ in results) > limit
        for intent, body, status in results:
            status = "not_shown (window full)" if full else status
            summary = self._arguments_summary(intent.arguments)
            checked["calls"].append({"round": number, "tool": intent.name or "?",
                                     "arguments": summary, "status": status})
            fields = dict(tool=intent.name or "?", during=tool, arguments=summary,
                          status=status, round=number, request_id=request_id,
                          result_characters=len(body))
            self.records.emit("oversight_tool_executed", **fields)
            self.records.life("oversight_tool_executed", **fields)
        if full:
            return "The conversation has reached the size limit and the last results were not added."
        valid = all(intent.name and not intent.parse_error for intent in intents)
        history.append({"role": "assistant", "content": text, "_artificium": {
            "kind": "life_loop_output",
            **({"tool_names": [intent.name for intent in intents]} if valid
               else {"tool_request_valid": False})}})
        for position, (intent, body, _) in enumerate(results):
            history.append({"role": role, "content": body, "_artificium": {
                "kind": "tool_result", "tool": intent.name,
                "tool_id": f"{request_id}:{position}"}})
        return ""

    @staticmethod
    def _arguments_summary(arguments: dict[str, Any]) -> str:
        return " ".join(
            f"{key}={_clip(value, 120)!r}" if isinstance(value, str)
            else f"{key}={_clip(json_dumps(value), 120)}"
            for key, value in arguments.items())

    @staticmethod
    def _oversight_unavailable(tool: str, exc: Exception) -> dict[str, Any]:
        if getattr(exc, "cost_limited", False):
            return {
                "status": "error",
                "summary": f"{tool} was not sent: {exc} Continue the work without it, "
                "and ask again after the reset only if it still matters.",
            }
        hint = getattr(exc, "hint", None)
        return {
            "status": "error",
            "summary": (
                f"{tool} could not reach the oversight model ({type(exc).__name__}: {exc})"
                + (f" {hint}" if hint else "")
                + " This is a problem with that connection, not with your request: "
                "continue the work without it, and retry later only if it still matters."
            ),
        }

    @staticmethod
    def _reviewer_checked(checked: dict[str, Any]) -> dict[str, Any]:
        """What the reviewer read itself, for the result the agent sees; empty
        when it was not allowed to read."""
        if not checked.get("tool_rounds_allowed"):
            return {}
        return {"reviewer_checked": {
            "tool_rounds_used": checked["tool_rounds"],
            "tool_rounds_allowed": checked["tool_rounds_allowed"],
            "requests": checked["requests"],
            "request_ids": checked["request_ids"],
            "calls": checked["calls"],
        }}

    def _oversight_failed(
        self, tool: str, exc: Exception, checked: dict[str, Any],
    ) -> dict[str, Any]:
        result = self._oversight_unavailable(tool, exc)
        if checked["calls"]:
            result.update(self._reviewer_checked(checked))
            result["summary"] += (" The reviewer had already read what `reviewer_checked` "
                                  "lists; the rest of its work is lost.")
        return result

    def request_review(
        self,
        claim: str,
        evidence_paths: list[str] | None = None,
        constraints: str = "",
    ) -> dict[str, Any]:
        reviewer = getattr(self, "reviewer", None)
        if reviewer is None:
            return {
                "status": "error",
                "summary": "request_review needs the running life-loop's model "
                "engine, which this context does not have",
            }
        if len(claim.strip()) < 10:
            raise ValueError(
                "claim is too short; state the claim and the reasoning behind it "
                "so a reviewer without your context can judge it"
            )
        sections, evidence = self._evidence_sections(
            list(evidence_paths or []), "supports or tests the claim")
        directives = DirectiveStore(self.paths, self.records).render()
        stated = "\n".join(item for item in (constraints.strip(), directives) if item)
        messages = [
            {"role": "system", "content": self.prompts.runtime("review_system")},
            {
                "role": "user",
                "content": self.prompts.runtime(
                    "review_request",
                    claim=claim.strip(),
                    constraints=stated or "none stated",
                    evidence="\n\n".join(sections) or "none supplied",
                ),
            },
        ]
        checked = {"tool_rounds": 0, "requests": 0, "request_ids": [], "calls": []}
        try:
            request_id, text = self._oversight_exchange(
                "request_review", reviewer, messages, checked)
        except Exception as exc:
            return self._oversight_failed("request_review", exc, checked)
        review = _THINK.sub("", text).strip()
        result = {
            "status": "reviewed",
            "summary": "fresh-context review completed; weigh it, it is not authoritative",
            "request_id": request_id,
            "review": review or "(the reviewer returned no text)",
            "evidence": evidence,
            **self._reviewer_checked(checked),
        }
        self.records.emit(
            "review_completed", request_id=request_id, evidence=evidence,
            claim_characters=len(claim), tool_rounds=checked["tool_rounds"],
            requests=checked["requests"],
        )
        return result

    def consult(
        self,
        question: str,
        evidence_paths: list[str] | None = None,
        context: str = "",
    ) -> dict[str, Any]:
        consultant = getattr(self, "consultant", None)
        if consultant is None:
            return {
                "status": "error",
                "summary": "consult needs the running life-loop's model engine, "
                "which this context does not have",
            }
        if len(question.strip()) < 10:
            raise ValueError(
                "question is too short; state the question, the options you see and "
                "what you already know, so an advisor without your context can answer it"
            )
        sections, evidence = self._evidence_sections(
            list(evidence_paths or []), "the question depends on")
        asker = getattr(self, "entity_ask", None)
        asked = asker() if callable(asker) else (
            "STANDING DIRECTIVES (verbatim)\n"
            + (DirectiveStore(self.paths, self.records).render() or "none recorded"))
        messages = [
            {"role": "system", "content": self.prompts.runtime("consult_system")},
            {
                "role": "user",
                "content": self.prompts.runtime(
                    "consult_request",
                    question=question.strip(),
                    context=context.strip() or "none supplied",
                    ask=asked,
                    evidence="\n\n".join(sections) or "none supplied",
                ),
            },
        ]
        checked = {"tool_rounds": 0, "requests": 0, "request_ids": [], "calls": []}
        try:
            request_id, text = self._oversight_exchange(
                "consult", consultant, messages, checked)
        except Exception as exc:
            return self._oversight_failed("consult", exc, checked)
        answer = _THINK.sub("", text).strip()
        result = {
            "status": "consulted",
            "summary": "the oversight model answered in a fresh context; weigh it, "
            "it is advice, not authority",
            "request_id": request_id,
            "answer": answer or "(the oversight model returned no text)",
            "evidence": evidence,
            **self._reviewer_checked(checked),
        }
        self.records.emit(
            "consult_completed", request_id=request_id, evidence=evidence,
            question_characters=len(question), tool_rounds=checked["tool_rounds"],
            requests=checked["requests"],
        )
        return result

    # -- decision review ---------------------------------------------------

    _REVIEW_SKIPS = {
        "cost_limit": "a cost limit is reached",
        "connection_failed": "the oversight model could not be reached",
        "unparseable": "its answer had no clear verdict",
        "unavailable": "this context has no oversight engine",
    }

    def _decision_gate(
        self, interaction_id: str, content: str, options: list[str] | None,
        default: str | None, reply_by_hours: float | None, review_complete: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """The review step of a decision, before anything is sent.

        Returns ``(held, review)``: ``held`` is the result to return instead of
        sending (the review, for the agent to weigh), else the decision goes out
        and ``review`` says how it was reviewed. The check never blocks
        communication: when it cannot run, the decision is sent at once.
        """

        # In memory only: a restart asks for a new review, which is cheap.
        pending = getattr(self, "_decision_reviews", None)
        if pending is None:
            pending = self._decision_reviews = {}
        earlier = pending.get(interaction_id) if review_complete else None
        # One review covers one decision: whichever way this call leaves, the
        # next decision starts over.
        if earlier is not None:
            pending.clear()
            return None, {"status": "reviewed", "request_id": earlier["request_id"]}
        if not getattr(self.config, "decision_review", True):
            pending.clear()
            self._decision_review_skipped("disabled", interaction_id)
            return None, None
        reviewer = getattr(self, "decision_reviewer", None)
        if reviewer is None:
            pending.clear()
            return None, self._decision_review_skipped("unavailable", interaction_id)
        asker = getattr(self, "entity_ask", None)
        asked = asker() if callable(asker) else (
            "STANDING DIRECTIVES (verbatim)\n"
            + (DirectiveStore(self.paths, self.records).render() or "none recorded"))
        messages = [
            {"role": "system", "content": self.prompts.runtime("decision_review_system")},
            {"role": "user", "content": self.prompts.runtime(
                "decision_review_request",
                ask=asked,
                open_decision=self._open_decision_text(),
                message=_clip(content.strip(), 6_000),
                options="; ".join(options) if options else "none (free answer)",
                default=default.strip() if default else "none",
                reply_by=f"{reply_by_hours:g} h" if reply_by_hours else "the configured wait",
            )},
        ]
        try:
            request_id, text = reviewer(messages)
        except Exception as exc:
            pending.clear()
            reason = "cost_limit" if getattr(exc, "cost_limited", False) else "connection_failed"
            return None, self._decision_review_skipped(
                reason, interaction_id, error=f"{type(exc).__name__}: {exc}")
        parsed = parse_decision_review(text, options)
        if parsed is None:
            pending.clear()
            return None, self._decision_review_skipped(
                "unparseable", interaction_id, request_id=request_id)
        label = getattr(self, "oversight_connection_label", None)
        connection = label() if callable(label) else "worker"
        pending[interaction_id] = {"request_id": request_id}
        fields = dict(request_id=request_id, connection=connection,
                      interaction_id=interaction_id, necessary=parsed["necessary"],
                      framing=parsed["framing"], default=parsed["default"])
        self.records.emit("decision_review_completed", points=parsed["points"],
                          reasons=parsed["reasons"], **fields)
        self.records.life("decision_review_completed",
                          preview=" ".join(parsed["reasons"].split())[:300], **fields)
        return {
            "status": "review_required",
            "summary": (("no review was pending for this interaction (a restart clears "
                         "them), so this call ran one. " if review_complete else "")
                        + "nothing was sent: the oversight model read this decision first, "
                        "in a fresh context. Weigh its verdict, then either drop the "
                        "question and act on your own judgement, or call send_interaction "
                        "again with review_complete=true, revised or not. It is advice: "
                        "you decide."),
            "request_id": request_id,
            "review": parsed,
        }, None

    def _decision_review_skipped(
        self, reason: str, interaction_id: str, *, error: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any] | None:
        self.records.emit("decision_review_skipped", reason=reason,
                          interaction_id=interaction_id, error=error, request_id=request_id)
        if reason == "disabled":
            return None
        self.records.life("decision_review_skipped", reason=reason,
                          interaction_id=interaction_id)
        why = self._REVIEW_SKIPS.get(reason, reason)
        return {"status": "skipped", "reason": reason, "error": error,
                "note": f"the decision review did not run ({why}), so it was sent without one"}

    def _open_decision_text(self) -> str:
        try:
            open_now = self.interactions.open_decisions()
        except OSError:
            open_now = []
        if not open_now:
            return "none"
        newest = open_now[-1]
        options = newest.get("options") or []
        return (f"[{newest.get('created_at')}] in interaction {newest.get('interaction_id')}: "
                f"{_clip(newest.get('content'), 2_000)}"
                + (f"\nOffered choices: {'; '.join(str(item) for item in options)}" if options else "")
                + "\n(Sending the new question withdraws this one.)")
