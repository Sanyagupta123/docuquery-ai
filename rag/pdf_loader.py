"""Utilities for safely extracting text from uploaded PDFs."""

from __future__ import annotations

from typing import Final

import fitz

MAX_UPLOAD_BYTES: Final[int] = 20 * 1024 * 1024


def extract_text(pdf_bytes: bytes, max_size_bytes: int = MAX_UPLOAD_BYTES) -> str:
    """Return the readable text from a PDF upload.

    The function validates the payload before opening it, rejects empty or corrupt PDFs,
    and raises clear errors rather than exposing implementation details.
    """
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise TypeError("PDF content must be provided as bytes.")

    payload = bytes(pdf_bytes)
    if not payload:
        raise ValueError("The uploaded PDF is empty.")
    if len(payload) > max_size_bytes:
        raise ValueError(
            f"The uploaded PDF exceeds the {max_size_bytes // (1024 * 1024)} MB size limit."
        )

    try:
        document = fitz.open(stream=payload, filetype="pdf")
    except Exception as exc:  # pragma: no cover - exercised through integration tests
        raise ValueError("The uploaded file is not a readable PDF.") from exc

    try:
        page_texts: list[str] = []
        for page in document:
            text = page.get_text("text")
            if text and text.strip():
                page_texts.append(text.strip())

        combined = "\n\n".join(page_texts)
        if not combined.strip():
            raise ValueError(
                "No readable text could be extracted from this PDF. This may be a scanned or image-only document."
            )

        return combined
    finally:
        document.close()
