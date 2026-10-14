"""Tests for grounded answer generation."""

import os
from unittest.mock import patch

from rag.answerer import answer_question


class FakeResponse:
    def __init__(self, text: str):
        self.output_text = text


def test_answer_question_uses_passages_when_api_is_configured() -> None:
    passages = [
        {"text": "The proposal reduces waste in the supply chain.", "chunk_index": 0},
        {"text": "The report focuses on measurable impact over time.", "chunk_index": 1},
    ]

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"}, clear=False):
        with patch("rag.answerer.OpenAI") as mock_client:
            mock_client.return_value.responses.create.return_value = FakeResponse("The proposal reduces waste in the supply chain.")
            result = answer_question("What does the proposal do?", passages)

    assert result["answer"] == "The proposal reduces waste in the supply chain."
    assert result["sources"] == passages


def test_answer_question_handles_missing_api_key() -> None:
    passages = [{"text": "Evidence excerpt here.", "chunk_index": 0}]

    with patch.dict(os.environ, {}, clear=True):
        result = answer_question("What is the answer?", passages)

    assert "not configured" in result["answer"].lower()
    assert result["sources"] == passages
