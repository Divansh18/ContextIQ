"""Semantic retrieval orchestration."""

from uuid import UUID

from elasticsearch import Elasticsearch

from app.core.config import (
    MAX_ELASTICSEARCH_NUM_CANDIDATES,
    MAX_SEARCH_TOP_K,
    Settings,
)
from app.rag.embeddings import embed_text
from app.rag.vector_store import search_similar_chunks
from app.schemas.search import SemanticSearchResult


def retrieve_semantic_chunks(
    *,
    query: str,
    top_k: int | None,
    document_id: UUID | None,
    settings: Settings,
    elasticsearch_client: Elasticsearch,
) -> list[SemanticSearchResult]:
    """Embed a query and retrieve its nearest document chunks."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")

    resolved_top_k = settings.search_default_top_k if top_k is None else top_k
    if not 1 <= resolved_top_k <= MAX_SEARCH_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_SEARCH_TOP_K}")

    api_key = (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key is not None
        else None
    )
    query_embedding = embed_text(
        normalized_query,
        api_key=api_key,
        model=settings.embedding_model,
        expected_dimensions=settings.embedding_dimensions,
    )
    num_candidates = min(
        resolved_top_k * settings.search_num_candidates_multiplier,
        MAX_ELASTICSEARCH_NUM_CANDIDATES,
    )

    return search_similar_chunks(
        elasticsearch_client,
        index_name=settings.elasticsearch_index,
        query_embedding=query_embedding,
        top_k=resolved_top_k,
        num_candidates=num_candidates,
        document_id=document_id,
    )
