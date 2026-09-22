"""Orchestration for PDF processing, embedding, and chunk indexing."""

from elasticsearch import Elasticsearch

from app.core.config import Settings
from app.rag.embeddings import embed_texts, validate_embedding_configuration
from app.rag.vector_store import (
    build_chunk_documents,
    bulk_index_chunk_documents,
    ensure_document_chunk_index,
)
from app.schemas.document import DocumentUploadResponse
from app.services.document_service import process_pdf_document


def ingest_pdf_document(
    *,
    filename: str | None,
    content_type: str | None,
    content: bytes,
    settings: Settings,
    elasticsearch_client: Elasticsearch,
) -> DocumentUploadResponse:
    """Process, embed, and persist every meaningful chunk from one PDF."""
    document = process_pdf_document(
        filename=filename,
        content_type=content_type,
        content=content,
        preview_character_limit=settings.preview_character_limit,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not document.chunks:
        return document

    api_key = (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key is not None
        else None
    )
    validate_embedding_configuration(
        api_key=api_key,
        model=settings.embedding_model,
        expected_dimensions=settings.embedding_dimensions,
    )
    ensure_document_chunk_index(
        elasticsearch_client,
        index_name=settings.elasticsearch_index,
        embedding_dimensions=settings.embedding_dimensions,
    )

    embeddings = embed_texts(
        [chunk.text for chunk in document.chunks],
        api_key=api_key,
        model=settings.embedding_model,
        expected_dimensions=settings.embedding_dimensions,
    )
    chunk_documents = build_chunk_documents(
        document_id=document.document_id,
        filename=document.filename,
        chunks=document.chunks,
        embeddings=embeddings,
        expected_embedding_dimensions=settings.embedding_dimensions,
    )
    bulk_index_chunk_documents(
        elasticsearch_client,
        index_name=settings.elasticsearch_index,
        documents=chunk_documents,
    )

    return document
