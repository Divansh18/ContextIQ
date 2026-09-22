"""Document upload endpoints."""

from typing import Annotated

from elasticsearch import Elasticsearch
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.elasticsearch import get_elasticsearch_client
from app.rag.embeddings import EmbeddingError, MissingOpenAIAPIKeyError
from app.rag.vector_store import IncompatibleIndexMappingError, VectorStoreError
from app.schemas.document import DocumentUploadResponse
from app.services.document_ingestion_service import ingest_pdf_document
from app.services.document_service import (
    DocumentProcessingError,
    UnsupportedDocumentTypeError,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_document(
    file: Annotated[UploadFile, File(description="The PDF document to process")],
    elasticsearch_client: Annotated[
        Elasticsearch,
        Depends(get_elasticsearch_client),
    ],
) -> DocumentUploadResponse:
    """Process, embed, and index one uploaded PDF."""
    settings = get_settings()

    try:
        content = await file.read()
        return await run_in_threadpool(
            ingest_pdf_document,
            filename=file.filename,
            content_type=file.content_type,
            content=content,
            settings=settings,
            elasticsearch_client=elasticsearch_client,
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except DocumentProcessingError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except MissingOpenAIAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except IncompatibleIndexMappingError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
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
    finally:
        await file.close()
