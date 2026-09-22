"""Grounded question-answering endpoint."""

from typing import Annotated

from elasticsearch import Elasticsearch
from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.elasticsearch import get_elasticsearch_client
from app.rag.embeddings import EmbeddingError, MissingOpenAIAPIKeyError
from app.rag.generation import GenerationError
from app.rag.vector_store import VectorStoreError
from app.schemas.answer import AskRequest, AskResponse
from app.services.question_service import answer_question

router = APIRouter(tags=["answers"])


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
)
async def ask_question(
    request: AskRequest,
    elasticsearch_client: Annotated[
        Elasticsearch,
        Depends(get_elasticsearch_client),
    ],
) -> AskResponse:
    """Answer a question using only retrieved ContextIQ sources."""
    settings = get_settings()

    try:
        return await run_in_threadpool(
            answer_question,
            question=request.question,
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
    except (EmbeddingError, GenerationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
