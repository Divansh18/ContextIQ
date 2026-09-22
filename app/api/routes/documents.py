"""Document upload endpoints."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.document import DocumentUploadResponse
from app.services.document_service import (
    DocumentProcessingError,
    UnsupportedDocumentTypeError,
    process_pdf_document,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_document(
    file: Annotated[UploadFile, File(description="The PDF document to process")],
) -> DocumentUploadResponse:
    """Validate and extract summary information from one uploaded PDF."""
    try:
        content = await file.read()
        return process_pdf_document(
            filename=file.filename,
            content_type=file.content_type,
            content=content,
            preview_character_limit=get_settings().preview_character_limit,
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
    finally:
        await file.close()
