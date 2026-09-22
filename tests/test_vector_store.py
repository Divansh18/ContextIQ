"""Unit tests for Elasticsearch chunk mapping and bulk indexing."""

from unittest.mock import Mock, patch
from uuid import UUID

import pytest

from app.rag.vector_store import (
    BulkIndexingError,
    IncompatibleIndexMappingError,
    SEARCH_RESULT_SOURCE_FIELDS,
    VectorSearchError,
    VectorStoreError,
    build_chunk_documents,
    build_document_chunk_mapping,
    bulk_index_chunk_documents,
    document_chunk_index_exists,
    ensure_document_chunk_index,
    search_similar_chunks,
)
from app.schemas.document import DocumentChunk

INDEX_NAME = "contextiq-document-chunks"
EMBEDDING_DIMENSIONS = 1_536


def _mapping_response(dimensions: int | None) -> dict[str, object]:
    embedding: dict[str, object] = {
        "type": "dense_vector",
        "index": True,
        "similarity": "cosine",
    }
    if dimensions is not None:
        embedding["dims"] = dimensions

    return {
        INDEX_NAME: {
            "mappings": {
                "properties": {
                    "embedding": embedding,
                }
            }
        }
    }


def test_mapping_defines_finalized_embedding_configuration() -> None:
    mapping = build_document_chunk_mapping(EMBEDDING_DIMENSIONS)

    assert mapping["dynamic"] == "strict"
    assert mapping["properties"]["embedding"] == {
        "type": "dense_vector",
        "dims": EMBEDDING_DIMENSIONS,
        "element_type": "float",
        "index": True,
        "similarity": "cosine",
    }


@pytest.mark.parametrize("dimensions", [0, -1, 4_097])
def test_mapping_rejects_invalid_embedding_dimensions(dimensions: int) -> None:
    with pytest.raises(ValueError):
        build_document_chunk_mapping(dimensions)


def test_index_exists_checks_configured_index() -> None:
    client = Mock()
    client.indices.exists.return_value = True

    assert document_chunk_index_exists(client, index_name=INDEX_NAME) is True
    client.indices.exists.assert_called_once_with(index=INDEX_NAME)


def test_compatible_existing_index_is_preserved() -> None:
    client = Mock()
    client.indices.exists.return_value = True
    client.indices.get_mapping.return_value = _mapping_response(
        EMBEDDING_DIMENSIONS
    )

    created = ensure_document_chunk_index(
        client,
        index_name=INDEX_NAME,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
    )

    assert created is False
    client.indices.create.assert_not_called()


def test_incompatible_existing_index_requires_migration() -> None:
    client = Mock()
    client.indices.exists.return_value = True
    client.indices.get_mapping.return_value = _mapping_response(None)

    with pytest.raises(IncompatibleIndexMappingError, match="recreation is required"):
        ensure_document_chunk_index(
            client,
            index_name=INDEX_NAME,
            embedding_dimensions=EMBEDDING_DIMENSIONS,
        )

    client.indices.create.assert_not_called()


def test_missing_index_is_created_with_explicit_mapping() -> None:
    client = Mock()
    client.indices.exists.return_value = False

    created = ensure_document_chunk_index(
        client,
        index_name=INDEX_NAME,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
    )

    assert created is True
    client.indices.create.assert_called_once_with(
        index=INDEX_NAME,
        mappings=build_document_chunk_mapping(EMBEDDING_DIMENSIONS),
    )


def test_index_connection_failure_is_reported() -> None:
    client = Mock()
    client.indices.exists.side_effect = RuntimeError("connection failed")

    with pytest.raises(VectorStoreError, match="Unable to prepare"):
        ensure_document_chunk_index(
            client,
            index_name=INDEX_NAME,
            embedding_dimensions=EMBEDDING_DIMENSIONS,
        )


def test_chunks_are_converted_to_elasticsearch_documents() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    chunks = [
        DocumentChunk(chunk_index=0, text="first", character_count=5),
        DocumentChunk(chunk_index=1, text="second", character_count=6),
    ]

    documents = build_chunk_documents(
        document_id=document_id,
        filename="knowledge.pdf",
        chunks=chunks,
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        expected_embedding_dimensions=2,
    )

    assert documents == [
        {
            "document_id": str(document_id),
            "filename": "knowledge.pdf",
            "chunk_index": 0,
            "text": "first",
            "character_count": 5,
            "embedding": [0.1, 0.2],
        },
        {
            "document_id": str(document_id),
            "filename": "knowledge.pdf",
            "chunk_index": 1,
            "text": "second",
            "character_count": 6,
            "embedding": [0.3, 0.4],
        },
    ]
    assert {document["document_id"] for document in documents} == {str(document_id)}


@patch("app.rag.vector_store.bulk")
def test_bulk_indexing_uses_one_bulk_helper_call(bulk: Mock) -> None:
    documents = build_chunk_documents(
        document_id="document-123",
        filename="knowledge.pdf",
        chunks=[DocumentChunk(chunk_index=0, text="first", character_count=5)],
        embeddings=[[0.1, 0.2]],
        expected_embedding_dimensions=2,
    )
    bulk.return_value = (1, [])

    indexed_count = bulk_index_chunk_documents(
        Mock(),
        index_name=INDEX_NAME,
        documents=documents,
    )

    assert indexed_count == 1
    bulk.assert_called_once()
    assert bulk.call_args.args[1] == [
        {
            "_index": INDEX_NAME,
            "_id": "document-123:0",
            "_source": documents[0],
        }
    ]


@patch("app.rag.vector_store.bulk")
def test_bulk_indexing_failure_is_reported(bulk: Mock) -> None:
    documents = build_chunk_documents(
        document_id="document-123",
        filename="knowledge.pdf",
        chunks=[DocumentChunk(chunk_index=0, text="first", character_count=5)],
        embeddings=[[0.1, 0.2]],
        expected_embedding_dimensions=2,
    )
    bulk.return_value = (0, [{"index": {"error": {"reason": "failed"}}}])

    with pytest.raises(BulkIndexingError, match="failed to index 1 chunk"):
        bulk_index_chunk_documents(
            Mock(),
            index_name=INDEX_NAME,
            documents=documents,
        )


@patch("app.rag.vector_store.bulk")
def test_bulk_transport_failure_is_wrapped(bulk: Mock) -> None:
    documents = build_chunk_documents(
        document_id="document-123",
        filename="knowledge.pdf",
        chunks=[DocumentChunk(chunk_index=0, text="first", character_count=5)],
        embeddings=[[0.1, 0.2]],
        expected_embedding_dimensions=2,
    )
    bulk.side_effect = RuntimeError("connection failed")

    with pytest.raises(BulkIndexingError, match="bulk indexing failed"):
        bulk_index_chunk_documents(
            Mock(),
            index_name=INDEX_NAME,
            documents=documents,
        )


def test_vector_search_uses_knn_and_converts_hits_without_embeddings() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    client = Mock()
    client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 0.91,
                    "_source": {
                        "document_id": str(document_id),
                        "filename": "knowledge.pdf",
                        "chunk_index": 1,
                        "text": "Containers run without managed virtual machines.",
                        "character_count": 48,
                        "embedding": [0.1, 0.2],
                    },
                }
            ]
        }
    }

    results = search_similar_chunks(
        client,
        index_name=INDEX_NAME,
        query_embedding=[0.4, 0.5],
        top_k=3,
        num_candidates=30,
    )

    client.search.assert_called_once_with(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": [0.4, 0.5],
            "k": 3,
            "num_candidates": 30,
        },
        size=3,
        source_includes=SEARCH_RESULT_SOURCE_FIELDS,
    )
    assert len(results) == 1
    assert results[0].document_id == document_id
    assert results[0].chunk_index == 1
    assert results[0].score == pytest.approx(0.91)
    assert "embedding" not in results[0].model_dump()


def test_vector_search_applies_document_id_as_knn_pre_filter() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")
    client = Mock()
    client.search.return_value = {"hits": {"hits": []}}

    search_similar_chunks(
        client,
        index_name=INDEX_NAME,
        query_embedding=[0.4, 0.5],
        top_k=2,
        num_candidates=20,
        document_id=document_id,
    )

    assert client.search.call_args.kwargs["knn"]["filter"] == {
        "term": {"document_id": str(document_id)}
    }


def test_vector_search_failure_is_wrapped() -> None:
    client = Mock()
    client.search.side_effect = RuntimeError("connection failed")

    with pytest.raises(VectorSearchError, match="vector search failed"):
        search_similar_chunks(
            client,
            index_name=INDEX_NAME,
            query_embedding=[0.4, 0.5],
            top_k=3,
            num_candidates=30,
        )


def test_vector_search_rejects_excessive_candidate_count() -> None:
    with pytest.raises(ValueError, match="must not exceed 10000"):
        search_similar_chunks(
            Mock(),
            index_name=INDEX_NAME,
            query_embedding=[0.4, 0.5],
            top_k=3,
            num_candidates=10_001,
        )
