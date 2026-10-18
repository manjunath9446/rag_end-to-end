from __future__ import annotations

from functools import lru_cache

import numpy as np


class Embedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

        # Newer sentence-transformers renamed the method; support both.
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim = int(self._model.get_embedding_dimension())
        else:
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        arr = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(arr[0], dtype=np.float32).tolist()


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)