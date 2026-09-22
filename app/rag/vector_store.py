"""Elasticsearch mapping, chunk conversion, and bulk indexing."""

from collections.abc import Sequence
from typing import Any, TypedDict
from uuid import UUID

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from app.core.config import (
    MAX_ELASTICSEARCH_EMBEDDING_DIMENSIONS,
    MAX_ELASTICSEARCH_NUM_CANDIDATES,
)
from app.schemas.document import DocumentChunk
from app.schemas.search import SemanticSearchResult

SEARCH_RESULT_SOURCE_FIELDS = (
    "document_id",
    "filename",
    "chunk_index",
    "text",
    "character_count",
)


class ElasticsearchChunkDocument(TypedDict):
    """Stored representation of one embedded document chunk."""

    document_id: str
    filename: str
    chunk_index: int
    text: str
    character_count: int
    embedding: list[float]


class VectorStoreError(RuntimeError):
    """Base error for Elasticsearch vector-store failures."""


class IncompatibleIndexMappingError(VectorStoreError):
    """Raised when an existing index cannot store configured embeddings."""


class BulkIndexingError(VectorStoreError):
    """Raised when one or more chunks cannot be bulk indexed."""


class VectorSearchError(VectorStoreError):
    """Raised when Elasticsearch vector retrieval fails."""


def build_document_chunk_mapping(embedding_dimensions: int) -> dict[str, Any]:
    """Build the finalized explicit mapping for ContextIQ document chunks."""
    if not 1 <= embedding_dimensions <= MAX_ELASTICSEARCH_EMBEDDING_DIMENSIONS:
        raise ValueError(
            "embedding_dimensions must be between 1 and "
            f"{MAX_ELASTICSEARCH_EMBEDDING_DIMENSIONS}"
        )

    return {
        "dynamic": "strict",
        "properties": {
            "document_id": {"type": "keyword"},
            "filename": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "text": {"type": "text"},
            "character_count": {"type": "integer"},
            "embedding": {
                "type": "dense_vector",
                "dims": embedding_dimensions,
                "element_type": "float",
                "index": True,
                "similarity": "cosine",
            },
        },
    }


def document_chunk_index_exists(
    client: Elasticsearch,
    *,
    index_name: str,
) -> bool:
    """Return whether the configured document chunk index already exists."""
    return bool(client.indices.exists(index=index_name))


def ensure_document_chunk_index(
    client: Elasticsearch,
    *,
    index_name: str,
    embedding_dimensions: int,
) -> bool:
    """Create a missing index or validate an existing one without changing it."""
    try:
        if document_chunk_index_exists(client, index_name=index_name):
            mapping_response = client.indices.get_mapping(index=index_name)
            _validate_embedding_mapping(
                mapping_response,
                index_name=index_name,
                expected_dimensions=embedding_dimensions,
            )
            return False

        client.indices.create(
            index=index_name,
            mappings=build_document_chunk_mapping(embedding_dimensions),
        )
    except IncompatibleIndexMappingError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"Unable to prepare Elasticsearch index '{index_name}'."
        ) from exc

    return True


def build_chunk_documents(
    *,
    document_id: UUID | str,
    filename: str,
    chunks: Sequence[DocumentChunk],
    embeddings: Sequence[Sequence[float]],
    expected_embedding_dimensions: int,
) -> list[ElasticsearchChunkDocument]:
    """Combine chunk metadata and embeddings into Elasticsearch documents."""
    if len(chunks) != len(embeddings):
        raise ValueError("each chunk must have exactly one embedding")

    documents: list[ElasticsearchChunkDocument] = []
    for chunk, embedding_values in zip(chunks, embeddings, strict=True):
        embedding = list(embedding_values)
        if len(embedding) != expected_embedding_dimensions:
            raise ValueError(
                "chunk embedding dimension does not match the configured dimension"
            )

        documents.append(
            ElasticsearchChunkDocument(
                document_id=str(document_id),
                filename=filename,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                character_count=chunk.character_count,
                embedding=embedding,
            )
        )

    return documents


def bulk_index_chunk_documents(
    client: Elasticsearch,
    *,
    index_name: str,
    documents: Sequence[ElasticsearchChunkDocument],
) -> int:
    """Index chunk documents through the Elasticsearch bulk API."""
    if not documents:
        return 0

    actions = [
        {
            "_index": index_name,
            "_id": f"{document['document_id']}:{document['chunk_index']}",
            "_source": document,
        }
        for document in documents
    ]

    try:
        indexed_count, errors = bulk(
            client,
            actions,
            raise_on_error=False,
            raise_on_exception=True,
        )
    except Exception as exc:
        raise BulkIndexingError("Elasticsearch bulk indexing failed.") from exc

    if errors or indexed_count != len(documents):
        failed_count = max(len(documents) - indexed_count, len(errors))
        raise BulkIndexingError(
            f"Elasticsearch failed to index {failed_count} chunk(s)."
        )

    return indexed_count


def search_similar_chunks(
    client: Elasticsearch,
    *,
    index_name: str,
    query_embedding: Sequence[float],
    top_k: int,
    num_candidates: int,
    document_id: UUID | str | None = None,
) -> list[SemanticSearchResult]:
    """Return the nearest stored chunks without exposing their vectors."""
    if not query_embedding:
        raise ValueError("query_embedding must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if num_candidates < top_k:
        raise ValueError("num_candidates must be greater than or equal to top_k")
    if num_candidates > MAX_ELASTICSEARCH_NUM_CANDIDATES:
        raise ValueError(
            "num_candidates must not exceed "
            f"{MAX_ELASTICSEARCH_NUM_CANDIDATES}"
        )

    knn: dict[str, Any] = {
        "field": "embedding",
        "query_vector": list(query_embedding),
        "k": top_k,
        "num_candidates": num_candidates,
    }
    if document_id is not None:
        knn["filter"] = {"term": {"document_id": str(document_id)}}

    try:
        response = client.search(
            index=index_name,
            knn=knn,
            size=top_k,
            source_includes=SEARCH_RESULT_SOURCE_FIELDS,
        )
        hits = response["hits"]["hits"]
        return [
            SemanticSearchResult(
                document_id=hit["_source"]["document_id"],
                filename=hit["_source"]["filename"],
                chunk_index=hit["_source"]["chunk_index"],
                text=hit["_source"]["text"],
                character_count=hit["_source"]["character_count"],
                score=hit["_score"],
            )
            for hit in hits
        ]
    except Exception as exc:
        raise VectorSearchError("Elasticsearch vector search failed.") from exc


def _validate_embedding_mapping(
    mapping_response: dict[str, Any],
    *,
    index_name: str,
    expected_dimensions: int,
) -> None:
    """Ensure an existing index matches the finalized embedding configuration."""
    try:
        embedding_mapping = mapping_response[index_name]["mappings"]["properties"][
            "embedding"
        ]
    except (KeyError, TypeError) as exc:
        raise _incompatible_mapping_error(
            index_name,
            "the embedding field is missing",
        ) from exc

    mismatches: list[str] = []
    if embedding_mapping.get("type") != "dense_vector":
        mismatches.append("embedding is not a dense_vector")
    if embedding_mapping.get("dims") != expected_dimensions:
        mismatches.append(
            f"embedding dims are {embedding_mapping.get('dims')!r}, "
            f"expected {expected_dimensions}"
        )
    if embedding_mapping.get("index", True) is not True:
        mismatches.append("embedding indexing is disabled")
    if embedding_mapping.get("similarity", "cosine") != "cosine":
        mismatches.append("embedding similarity is not cosine")

    if mismatches:
        raise _incompatible_mapping_error(index_name, "; ".join(mismatches))


def _incompatible_mapping_error(
    index_name: str,
    reason: str,
) -> IncompatibleIndexMappingError:
    return IncompatibleIndexMappingError(
        f"Elasticsearch index '{index_name}' has an incompatible mapping: {reason}. "
        "Migration or recreation is required; ContextIQ will not modify or delete "
        "the existing index automatically."
    )
