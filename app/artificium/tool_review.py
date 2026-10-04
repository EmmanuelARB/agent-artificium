from __future__ import annotations

import re
from typing import Any

from .directives import DirectiveStore


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class ReviewToolsMixin:
    """Independent review of a claim by the model in a fresh context.

    Mixed into :class:`ToolRegistry`. The runtime attaches ``reviewer``, a
    callable taking chat messages and returning the reply text; without it
    (a registry built outside the runtime) the tool reports that it is
    unavailable.
    """

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
        paths = list(evidence_paths or [])
        budget = int(
            self.config.working_memory_limit * self.config.chars_per_token * 0.4
        )
        share = budget // max(1, len(paths))
        sections: list[str] = []
        evidence: list[dict[str, Any]] = []
        for supplied in paths:
            path = self._resolve(supplied)
            if not path.is_file():
                raise FileNotFoundError(
                    f"evidence path {path} is not a file; pass files whose content "
                    "supports or tests the claim"
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
        request_id, text = reviewer(messages)
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
