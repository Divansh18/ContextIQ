"""Tests for document upload behavior."""

from unittest.mock import Mock, patch
from uuid import UUID

import pymupdf
from fastapi.testclient import TestClient

from app.rag.embeddings import (
    MissingOpenAIAPIKeyError,
    UnexpectedEmbeddingDimensionError,
)
from app.rag.vector_store import BulkIndexingError
from app.services.document_service import process_pdf_document


def _create_pdf(page_texts: list[str]) -> bytes:
    document = pymupdf.open()
    try:
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


@patch("app.services.document_ingestion_service.bulk_index_chunk_documents")
@patch("app.services.document_ingestion_service.ensure_document_chunk_index")
@patch("app.services.document_ingestion_service.embed_texts")
@patch("app.services.document_ingestion_service.validate_embedding_configuration")
def test_upload_pdf_returns_extracted_summary(
    validate_configuration: Mock,
    embed_batch: Mock,
    ensure_index: Mock,
    bulk_index: Mock,
    client: TestClient,
) -> None:
    embed_batch.return_value = [[0.0] * 1_536]
    pdf_bytes = _create_pdf(["ContextIQ first page", "Knowledge assistant second page"])

    response = client.post(
        "/documents/upload",
        files={"file": ("knowledge.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert UUID(payload["document_id"])
    assert payload["filename"] == "knowledge.pdf"
    assert payload["page_count"] == 2
    assert payload["character_count"] > 0
    assert "ContextIQ first page" in payload["preview"]
    assert "Knowledge assistant second page" in payload["preview"]
    assert payload["chunk_count"] == 1
    assert len(payload["chunks"]) == payload["chunk_count"]
    assert payload["chunks"][0]["chunk_index"] == 0
    assert payload["chunks"][0]["character_count"] == len(
        payload["chunks"][0]["text"]
    )
    assert "embedding" not in payload["chunks"][0]
    validate_configuration.assert_called_once()
    embed_batch.assert_called_once()
    ensure_index.assert_called_once()
    bulk_index.assert_called_once()


def test_processing_assigns_a_unique_document_id_per_upload() -> None:
    pdf_bytes = _create_pdf(["ContextIQ"])

    first = process_pdf_document(
        filename="knowledge.pdf",
        content_type="application/pdf",
        content=pdf_bytes,
        preview_character_limit=200,
        chunk_size=1_000,
        chunk_overlap=200,
    )
    second = process_pdf_document(
        filename="knowledge.pdf",
        content_type="application/pdf",
        content=pdf_bytes,
        preview_character_limit=200,
        chunk_size=1_000,
        chunk_overlap=200,
    )

    assert first.document_id != second.document_id


def test_upload_rejects_non_pdf_file_type(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "The uploaded file must have a .pdf extension."
    }


def test_upload_rejects_non_pdf_content_type(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.pdf", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "The uploaded file must use the application/pdf content type."
    }


def test_upload_rejects_malformed_pdf(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("broken.pdf", b"%PDF-not-valid", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "The uploaded file is not a readable PDF."}


@patch(
    "app.api.routes.documents.ingest_pdf_document",
    side_effect=MissingOpenAIAPIKeyError("OPENAI_API_KEY is required."),
)
def test_upload_reports_missing_openai_api_key(
    ingest_document: Mock,
    client: TestClient,
) -> None:
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "knowledge.pdf",
                _create_pdf(["ContextIQ"]),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "OPENAI_API_KEY is required."}


@patch(
    "app.api.routes.documents.ingest_pdf_document",
    side_effect=UnexpectedEmbeddingDimensionError("unexpected dimensions"),
)
def test_upload_reports_embedding_failure(
    ingest_document: Mock,
    client: TestClient,
) -> None:
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "knowledge.pdf",
                _create_pdf(["ContextIQ"]),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "unexpected dimensions"}


@patch(
    "app.api.routes.documents.ingest_pdf_document",
    side_effect=BulkIndexingError("Elasticsearch bulk indexing failed."),
)
def test_upload_reports_indexing_failure(
    ingest_document: Mock,
    client: TestClient,
) -> None:
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "knowledge.pdf",
                _create_pdf(["ContextIQ"]),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Elasticsearch bulk indexing failed."
    }
