from __future__ import annotations

import re

from .generator import ABSTAIN_TOKEN, GrokClient, SYSTEM_PROMPT
from .store import Hit


ABSTAIN_MESSAGE = (
    "I don't have enough information in the available documents to answer that."
)

# Accepts both standard [abc123] and full-width 【abc123】 brackets
_CITE_RE = re.compile(r"[\[【]([a-f0-9]{16})[\]】]")


def _build_context(hits: list[Hit]) -> str:
    lines: list[str] = []
    for h in hits:
        src = h.metadata.get("filename", "unknown")
        idx = h.metadata.get("chunk_index", "?")
        lines.append(f"[{h.chunk_id}] (source: {src}, chunk_index: {idx})")
        lines.append(h.text.strip())
        lines.append("")
    return "\n".join(lines)


def _build_prompt(question: str, hits: list[Hit]) -> str:
    return (
        f"Question: {question}\n\n"
        f"Context chunks:\n{_build_context(hits)}\n"
        f"Answer:"
    )


class Answerer:
    def __init__(self, client: GrokClient) -> None:
        self.client = client

    def answer(self, question: str, hits: list[Hit]) -> dict:
        if not hits:
            return self._abstain(0, 0, 0)

        result = self.client.complete(
            system=SYSTEM_PROMPT,
            user=_build_prompt(question, hits),
        )
        text = result.text

        # 1. Abstention
        if not text or ABSTAIN_TOKEN in text:
            return self._abstain(
                result.prompt_tokens, result.completion_tokens, result.total_tokens
            )

        # 2. Fix full-width brackets if the model output them
        text = text.replace("【", "[").replace("】", "]")

        # 3. Verify citations against the retrieved set
        retrieved_ids = {h.chunk_id for h in hits}
        raw_cites = _CITE_RE.findall(text)
        valid_cites = [cid for cid in raw_cites if cid in retrieved_ids]

        # Strip hallucinated ids from the visible answer text.
        def _strip(m: re.Match) -> str:
            return m.group(0) if m.group(1) in retrieved_ids else ""

        text = _CITE_RE.sub(_strip, text).strip()

        # 4. Build citation objects
        by_id = {h.chunk_id: h for h in hits}
        citations: list[dict] = []
        for cid in dict.fromkeys(valid_cites):
            h = by_id[cid]
            citations.append({
                "chunk_id": h.chunk_id,
                "doc_id": h.doc_id,
                "filename": h.metadata.get("filename", ""),
                "chunk_index": h.metadata.get("chunk_index"),
                "score": round(h.score, 4),
                "snippet": h.text[:200],
            })

        return {
            "answer": text,
            "abstained": False,
            "citations": citations,
            "citations_inferred": not citations,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            },
        }

    @staticmethod
    def _abstain(p: int, c: int, t: int) -> dict:
        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "citations": [],
            "citations_inferred": False,
            "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t},
        }