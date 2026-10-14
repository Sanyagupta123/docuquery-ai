"""Embedding-based retrieval for document chunks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    """Indexes and searches document chunks using sentence embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as exc:  # pragma: no cover - depends on network access
                raise RuntimeError(
                    "The sentence-transformers model could not be downloaded or loaded. "
                    "Check internet access and storage permissions before indexing a document."
                ) from exc
        return self.model

    def index(self, chunks: list[str]) -> int:
        """Encode and store chunks in memory."""
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("At least one chunk is required to build the index.")

        cleaned_chunks: list[str] = []
        for chunk in chunks:
            if not isinstance(chunk, str):
                raise TypeError("Each chunk must be a string.")
            cleaned = chunk.strip()
            if not cleaned:
                raise ValueError("Chunk content cannot be empty.")
            cleaned_chunks.append(cleaned)

        model = self._load_model()
        embeddings = model.encode(cleaned_chunks, convert_to_numpy=True, normalize_embeddings=True)
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] != len(cleaned_chunks):
            raise ValueError("Embedding count does not match the stored chunk count.")

        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        array = array / norms

        self.chunks = cleaned_chunks
        self.embeddings = array
        return len(cleaned_chunks)

    def search(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the best matching chunks for a question."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A question is required.")
        if self.embeddings is None or not self.chunks:
            raise ValueError("No document has been indexed yet.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        model = self._load_model()
        question_embedding = model.encode(
            [question.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        question_vector = np.asarray(question_embedding, dtype=np.float32)
        if question_vector.ndim == 1:
            question_vector = question_vector.reshape(1, -1)

        question_norm = np.linalg.norm(question_vector, axis=1, keepdims=True)
        question_norm[question_norm == 0] = 1.0
        question_vector = question_vector / question_norm

        scores = self.embeddings @ question_vector[0]
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            chunk_index = int(index)
            results.append(
                {
                    "text": self.chunks[chunk_index],
                    "score": float(scores[chunk_index]),
                    "chunk_index": chunk_index,
                }
            )
        return results
