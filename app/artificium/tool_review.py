from __future__ import annotations

import re
from typing import Any

from .directives import DirectiveStore


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class ReviewToolsMixin:
    """Fresh-context requests to the oversight model: ``request_review`` and
    ``consult``.

    Mixed into :class:`ToolRegistry`. The runtime attaches ``reviewer`` and
    ``consultant``, callables taking chat messages and returning
    ``(request_id, text)``, plus ``entity_ask`` (what the entity asked, for
    consult) and ``oversight_budget`` (the Config whose context window bounds
    what is sent). Without them (a registry built outside the runtime) the
    tools report that they are unavailable.
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
