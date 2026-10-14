"""Text chunking utilities for document retrieval."""

from __future__ import annotations


def split_into_chunks(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """Split a body of text into overlapping word-based chunks.

    The chunk size and overlap are measured in words, not tokens. This keeps the behavior
    deterministic and easy to test across document sizes.
    """
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    normalized = text.strip()
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = normalized.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        if chunk_text.strip():
            chunks.append(chunk_text)
        if end == len(words):
            break
        start += step

    return chunks
