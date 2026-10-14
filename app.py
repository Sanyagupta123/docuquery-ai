"""Streamlit frontend for the DocuQuery document Q&A experience."""

from __future__ import annotations

from typing import Any

import streamlit as st
from dotenv import load_dotenv

from rag.answerer import answer_question
from rag.chunker import split_into_chunks
from rag.pdf_loader import MAX_UPLOAD_BYTES, extract_text
from rag.retriever import Retriever

load_dotenv()

MAX_UPLOAD_MB = 10


def _init_state() -> None:
    defaults = {
        "retriever": None,
        "document_name": "",
        "indexed_chunks": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _show_sources(passages: list[dict[str, Any]]) -> None:
    if not passages:
        st.info("No supporting passages are available for this question.")
        return

    st.subheader("Supporting passages")
    st.caption("Similarity scores indicate textual similarity only; they do not establish factual correctness.")
    for passage in passages:
        chunk_index = passage.get("chunk_index", "unknown")
        score = passage.get("score", 0.0)
        text = passage.get("text", "").strip()
        with st.expander(f"Chunk {chunk_index} · similarity {score:.3f}", expanded=False):
            st.write(text)


def main() -> None:
    _init_state()

    st.set_page_config(page_title="DocuQuery", page_icon="📄", layout="wide")
    st.title("DocuQuery")
    st.caption("AI-powered PDF question answering grounded in retrieved document passages.")

    st.info(
        "This prototype runs locally in memory and does not persist uploaded PDFs or extracted text beyond the current session."
    )

    with st.container():
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            help="Only text-based PDFs are supported in this MVP. Scanned image-only PDFs are not OCR-enabled.",
        )

        if uploaded_file is not None:
            st.write(f"Selected file: {uploaded_file.name}")

        if st.button("Index document", disabled=uploaded_file is None):
            try:
                pdf_bytes = uploaded_file.getvalue()
                text = extract_text(pdf_bytes, max_size_bytes=MAX_UPLOAD_MB * 1024 * 1024)
                chunks = split_into_chunks(text)
                retriever = Retriever()
                retriever.index(chunks)
                st.session_state.retriever = retriever
                st.session_state.document_name = uploaded_file.name
                st.session_state.indexed_chunks = len(chunks)
                st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}.")
            except (TypeError, ValueError) as exc:
                st.error(str(exc))
                st.session_state.retriever = None
                st.session_state.document_name = ""
                st.session_state.indexed_chunks = 0
            except Exception as exc:  # pragma: no cover - real model failures should surface clearly
                st.error(f"Document indexing failed: {exc}")
                st.session_state.retriever = None
                st.session_state.document_name = ""
                st.session_state.indexed_chunks = 0

    active_document = st.session_state.get("document_name") or "No document indexed"
    st.metric("Active document", active_document)
    st.metric("Indexed chunks", st.session_state.get("indexed_chunks", 0))

    question = st.text_area("Question", placeholder="Ask a question about the uploaded document...")
    if st.button("Ask", disabled=not question.strip() or st.session_state.get("retriever") is None):
        if not question.strip():
            st.warning("Enter a question before submitting.")
        if st.session_state.get("retriever") is None:
            st.warning("Please upload and index a PDF before asking questions.")
            st.stop()

        try:
            with st.spinner("Retrieving evidence and generating an answer..."):
                matches = st.session_state.retriever.search(question, top_k=3)
                answer_result = answer_question(question, matches)

            st.subheader("Answer")
            answer_text = answer_result.get("answer", "")
            if answer_text:
                st.write(answer_text)
            else:
                st.warning("The document excerpts do not provide enough information to answer this question.")

            _show_sources(answer_result.get("sources", matches))
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - provider issues handled in answerer
            st.error(f"Answer generation failed: {exc}")


if __name__ == "__main__":
    main()
