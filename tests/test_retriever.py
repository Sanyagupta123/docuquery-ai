"""Tests for retrieval logic and ranking."""

import numpy as np

import rag.retriever as retriever_module
from rag.retriever import Retriever


class DummyEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vec = np.zeros(3, dtype=np.float32)
            if "alpha" in lowered:
                vec[0] += 1.0
            if "beta" in lowered:
                vec[1] += 1.0
            if "gamma" in lowered:
                vec[2] += 1.0
            vectors.append(vec)
        return np.asarray(vectors, dtype=np.float32)


def test_retriever_indexes_and_searches(monkeypatch) -> None:
    monkeypatch.setattr(retriever_module, "SentenceTransformer", lambda _: DummyEmbeddingModel())
    retriever = Retriever(model_name="dummy-model")
    chunks = ["Alpha beta", "Beta gamma", "Gamma delta"]
    retriever.index(chunks)

    matches = retriever.search("alpha gamma", top_k=2)
    assert len(matches) == 2
    assert all("text" in match for match in matches)
    assert all("score" in match for match in matches)
    assert all("chunk_index" in match for match in matches)
    assert matches[0]["score"] >= matches[1]["score"]
    assert set(match["chunk_index"] for match in matches) <= {0, 1, 2}


def test_retriever_rejects_unindexed_question() -> None:
    retriever = Retriever(model_name="dummy-model")
    try:
        retriever.search("why")
        raise AssertionError("Expected an unindexed retriever to raise ValueError.")
    except ValueError as exc:
        assert "indexed" in str(exc).lower()
