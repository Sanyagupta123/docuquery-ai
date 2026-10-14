"""Tests for deterministic chunk splitting."""

import pytest

from rag.chunker import split_into_chunks


def test_chunker_splits_with_overlap() -> None:
    text = "A B C D E F G H"
    chunks = split_into_chunks(text, chunk_size=5, overlap=2)
    assert chunks == ["A B C D E", "D E F G H"]


def test_chunker_handles_short_text() -> None:
    chunks = split_into_chunks("Alpha beta gamma", chunk_size=10, overlap=2)
    assert chunks == ["Alpha beta gamma"]


def test_chunker_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        split_into_chunks("A B C", chunk_size=0, overlap=0)

    with pytest.raises(ValueError):
        split_into_chunks("A B C", chunk_size=4, overlap=4)

    with pytest.raises(ValueError):
        split_into_chunks("A B C", chunk_size=4, overlap=-1)


def test_chunker_handles_empty_input() -> None:
    assert split_into_chunks("   ") == []
