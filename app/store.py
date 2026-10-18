from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict


def _normalize_where(where: dict | None) -> dict | None:
    """ChromaDB accepts a single-key exact-match dict, or an operator form.

    - {"filename": "x"}                          -> pass through
    - {"filename": "x", "chunk_index": 3}        -> {"$and": [{"filename": "x"}, {"chunk_index": 3}]}
    - None or {}                                 -> None
    """
    if not where:
        return None
    if len(where) == 1:
        return where
    return {"$and": [{k: v} for k, v in where.items()]}


class VectorStore:
    def __init__(self, persist_dir: Path, collection: str) -> None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def delete_by_doc(self, doc_id: str) -> int:
        found = self._collection.get(where={"doc_id": doc_id}, include=[])
        ids = found.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def upsert(self, ids: list[str], embeddings: list[list[float]],
               texts: list[str], metadatas: list[dict]) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], top_k: int,
              where: dict | None = None) -> list[Hit]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            where=_normalize_where(where),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for cid, text, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            meta = meta or {}
            hits.append(Hit(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                text=text,
                score=1.0 - float(dist),
                metadata=meta,
            ))
        return hits

    def count(self) -> int:
        return int(self._collection.count())