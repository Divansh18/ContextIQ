"""Tests for the document ingestion orchestration flow."""

from unittest.mock import Mock, patch
from uuid import UUID

from pydantic import SecretStr

from app.core.config import Settings
from app.schemas.document import DocumentChunk, DocumentUploadResponse
from app.services.document_ingestion_service import ingest_pdf_document

DOCUMENT_ID = UUID("12345678-1234-5678-1234-567812345678")


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key=SecretStr("test-key"),
        embedding_dimensions=2,
    )


def _processed_document() -> DocumentUploadResponse:
    chunks = [
        DocumentChunk(chunk_index=0, text="first", character_count=5),
        DocumentChunk(chunk_index=1, text="second", character_count=6),
    ]
    return DocumentUploadResponse(
        document_id=DOCUMENT_ID,
        filename="knowledge.pdf",
        page_count=1,
        character_count=11,
        preview="first second",
        chunk_count=len(chunks),
        chunks=chunks,
    )


@patch("app.services.document_ingestion_service.bulk_index_chunk_documents")
@patch("app.services.document_ingestion_service.embed_texts")
@patch("app.services.document_ingestion_service.ensure_document_chunk_index")
@patch("app.services.document_ingestion_service.process_pdf_document")
def test_ingestion_preserves_document_id_for_every_indexed_chunk(
    process_pdf: Mock,
    ensure_index: Mock,
    embed_batch: Mock,
    bulk_index: Mock,
) -> None:
    process_pdf.return_value = _processed_document()
    embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

    response = ingest_pdf_document(
        filename="knowledge.pdf",
        content_type="application/pdf",
        content=b"pdf bytes",
        settings=_settings(),
        elasticsearch_client=Mock(),
    )

    assert response.document_id == DOCUMENT_ID
    indexed_documents = bulk_index.call_args.kwargs["documents"]
    assert {document["document_id"] for document in indexed_documents} == {
        str(DOCUMENT_ID)
    }
    embed_batch.assert_called_once()
    ensure_index.assert_called_once()
    bulk_index.assert_called_once()


@patch("app.services.document_ingestion_service.bulk_index_chunk_documents")
@patch("app.services.document_ingestion_service.embed_texts")
@patch("app.services.document_ingestion_service.ensure_document_chunk_index")
@patch("app.services.document_ingestion_service.process_pdf_document")
def test_empty_document_skips_embedding_and_indexing(
    process_pdf: Mock,
    ensure_index: Mock,
    embed_batch: Mock,
    bulk_index: Mock,
) -> None:
    process_pdf.return_value = DocumentUploadResponse(
        document_id=DOCUMENT_ID,
        filename="empty.pdf",
        page_count=1,
        character_count=0,
        preview="",
        chunk_count=0,
        chunks=[],
    )

    response = ingest_pdf_document(
        filename="empty.pdf",
        content_type="application/pdf",
        content=b"pdf bytes",
        settings=Settings(_env_file=None),
        elasticsearch_client=Mock(),
    )

    assert response.chunk_count == 0
    ensure_index.assert_not_called()
    embed_batch.assert_not_called()
    bulk_index.assert_not_called()
