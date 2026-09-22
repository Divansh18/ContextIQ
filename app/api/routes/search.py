"""Semantic-search endpoint."""

from typing import Annotated

from elasticsearch import Elasticsearch
from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.elasticsearch import get_elasticsearch_client
from app.rag.embeddings import EmbeddingError, MissingOpenAIAPIKeyError
from app.rag.retriever import retrieve_semantic_chunks
from app.rag.vector_store import VectorStoreError
from app.schemas.search import SemanticSearchRequest, SemanticSearchResponse

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SemanticSearchResponse,
    status_code=status.HTTP_200_OK,
)
async def semantic_search(
    request: SemanticSearchRequest,
    elasticsearch_client: Annotated[
        Elasticsearch,
        Depends(get_elasticsearch_client),
    ],
) -> SemanticSearchResponse:
    """Retrieve document chunks that are semantically related to a query."""
    settings = get_settings()

    try:
        results = await run_in_threadpool(
            retrieve_semantic_chunks,
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
            settings=settings,
            elasticsearch_client=elasticsearch_client,
        )
    except MissingOpenAIAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return SemanticSearchResponse(query=request.query, results=results)
