"""Tests for PDF loading and validation."""

import fitz

from rag.pdf_loader import extract_text


def _make_valid_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_extract_text_from_valid_pdf() -> None:
    pdf_bytes = _make_valid_pdf("This is a sample report about AI safety and evaluation.")
    text = extract_text(pdf_bytes)
    assert "sample report" in text.lower()


def test_extract_text_rejects_empty_pdf() -> None:
    try:
        extract_text(b"")
        raise AssertionError("Expected empty PDF data to raise a ValueError.")
    except ValueError as exc:
        assert "empty" in str(exc).lower()


def test_extract_text_rejects_corrupt_pdf() -> None:
    try:
        extract_text(b"not-a-real-pdf")
        raise AssertionError("Expected invalid PDF data to raise a ValueError.")
    except ValueError as exc:
        assert "readable pdf" in str(exc).lower() or "not a readable pdf" in str(exc).lower()


def test_extract_text_rejects_pdf_without_text() -> None:
    document = fitz.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()

    try:
        extract_text(pdf_bytes)
        raise AssertionError("Expected a blank PDF to raise a ValueError.")
    except ValueError as exc:
        assert "no readable text" in str(exc).lower()
