from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .chunker import chunk_text
from .config import Settings
from .embedder import get_embedder
from .loader import load
from .store import Hit, VectorStore


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = get_embedder(settings.embed_model)
        self.store = VectorStore(settings.chroma_dir, settings.collection)
        self.answerer = None
        if settings.grok_api_key:
            try:
                from .answerer import Answerer
                from .generator import GrokClient
                self.answerer = Answerer(GrokClient(settings))
            except Exception as e:  # bad key format, no network, etc.
                print(f"[warn] Grok client init failed: {e}")

    # --- ingest ---------------------------------------------------------

    def ingest_file(self, src: Path) -> dict:
        src = Path(src)
        raw = src.read_bytes()
        doc_id = hashlib.sha256(src.name.encode() + b"|" + raw).hexdigest()[:16]

        archived = self.settings.uploads_dir / f"{doc_id}_{src.name}"
        if not archived.exists():
            shutil.copyfile(src, archived)

        doc = load(archived)
        chunks = chunk_text(
            doc_id=doc_id,
            text=doc.text,
            filename=src.name,
            size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        embeddings = self.embedder.embed_chunks([c.text for c in chunks])

        replaced = self.store.delete_by_doc(doc_id)
        self.store.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            texts=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

        return {
            "doc_id": doc_id,
            "filename": src.name,
            "extension": doc.extension,
            "chunks": len(chunks),
            "replaced_chunks": replaced,
            "total_in_store": self.store.count(),
        }

    # --- retrieve -------------------------------------------------------

    def _retrieve_raw(self, query: str, top_k: int | None = None,
                      where: dict | None = None) -> list[Hit]:
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        k = top_k if top_k is not None else self.settings.top_k
        q_emb = self.embedder.embed_query(query)
        return self.store.query(q_emb, top_k=k, where=where)

    def retrieve(self, query: str, top_k: int | None = None,
                 where: dict | None = None) -> dict:
        hits = self._retrieve_raw(query, top_k, where)
        k = top_k if top_k is not None else self.settings.top_k
        return {
            "query": query,
            "top_k": k,
            "hits": [
                {
                    "chunk_id": h.chunk_id,
                    "doc_id": h.doc_id,
                    "text": h.text,
                    "score": round(h.score, 4),
                    "metadata": h.metadata,
                }
                for h in hits
            ],
        }

    # --- ask ------------------------------------------------------------

    def ask(self, question: str, top_k: int | None = None,
            where: dict | None = None) -> dict:
        if self.answerer is None:
            raise RuntimeError(
                "No LLM configured. Set RAG_GROK_API_KEY in .env and restart."
            )
        hits = self._retrieve_raw(question, top_k, where)
        gen = self.answerer.answer(question, hits)
        return {
            "question": question,
            "answer": gen["answer"],
            "abstained": gen["abstained"],
            "citations": gen["citations"],
            "citations_inferred": gen["citations_inferred"],
            "usage": gen["usage"],
            "retrieved_count": len(hits),
            "retrieved_chunk_ids": [h.chunk_id for h in hits],
            "model": self.answerer.client.model,
        }

    # --- stats ----------------------------------------------------------

    def stats(self) -> dict:
        return {
            "collection": self.settings.collection,
            "count": self.store.count(),
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "llm": self.answerer.client.model if self.answerer else None,
        }