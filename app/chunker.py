import hashlib
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    metadata: dict


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _split(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("0 <= overlap < chunk_size required")

    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    out: list[str] = []
    cur = ""

    for sent in _sentences(text):
        # A single sentence that exceeds the budget gets force-split.
        if len(sent) > size:
            if cur:
                out.append(cur)
                cur = ""
            step = size - overlap
            for i in range(0, len(sent), step):
                piece = sent[i:i + size]
                if piece:
                    out.append(piece)
            continue

        cand = f"{cur} {sent}".strip() if cur else sent
        if len(cand) <= size:
            cur = cand
        else:
            out.append(cur)
            tail = cur[-overlap:] if overlap else ""
            if tail:
                m = re.search(r"\s", tail)
                tail = tail[m.end():] if m else tail
            cur = f"{tail} {sent}".strip() if tail else sent

    if cur:
        out.append(cur)
    return out


def _chunk_id(doc_id: str, index: int, text: str) -> str:
    return hashlib.sha256(
        f"{doc_id}|{index}|{text}".encode("utf-8")
    ).hexdigest()[:16]


def chunk_text(doc_id: str, text: str, filename: str,
               size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for i, body in enumerate(_split(text, size, overlap)):
        out.append(Chunk(
            chunk_id=_chunk_id(doc_id, i, body),
            doc_id=doc_id,
            index=i,
            text=body,
            metadata={"doc_id": doc_id, "filename": filename, "chunk_index": i},
        ))
    return out