from __future__ import annotations

import re
from typing import Any

from .directives import DirectiveStore
from .oversight import _clip


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
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
    without review.
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
        try:
            request_id, text = reviewer(messages)
        except Exception as exc:
            return self._oversight_unavailable("request_review", exc)
        review = _THINK.sub("", text).strip()
        result = {
            "status": "reviewed",
            "summary": "fresh-context review completed; weigh it, it is not authoritative",
            "request_id": request_id,
            "review": review or "(the reviewer returned no text)",
            "evidence": evidence,
        }
        self.records.emit(
            "review_completed", request_id=request_id, evidence=evidence,
            claim_characters=len(claim),
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
        try:
            request_id, text = consultant(messages)
        except Exception as exc:
            return self._oversight_unavailable("consult", exc)
        answer = _THINK.sub("", text).strip()
        result = {
            "status": "consulted",
            "summary": "the oversight model answered in a fresh context; weigh it, "
            "it is advice, not authority",
            "request_id": request_id,
            "answer": answer or "(the oversight model returned no text)",
            "evidence": evidence,
        }
        self.records.emit(
            "consult_completed", request_id=request_id, evidence=evidence,
            question_characters=len(question),
        )
        return result

    # -- decision review ---------------------------------------------------

    _REVIEW_SKIPS = {
        "cost_limit": "the daily cost limit is reached",
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
