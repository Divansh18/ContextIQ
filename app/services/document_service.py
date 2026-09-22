"""PDF validation and text extraction services."""

from pathlib import Path
from uuid import uuid4

import pymupdf

from app.rag.chunking import chunk_text
from app.schemas.document import DocumentUploadResponse

PDF_CONTENT_TYPES = frozenset({"application/pdf", "application/x-pdf"})
PDF_HEADER = b"%PDF-"
PDF_HEADER_SEARCH_LIMIT = 1_024


class UnsupportedDocumentTypeError(ValueError):
    """Raised when an upload is not identified as a PDF."""


class DocumentProcessingError(ValueError):
    """Raised when an uploaded PDF cannot be processed."""


def process_pdf_document(
    *,
    filename: str | None,
    content_type: str | None,
    content: bytes,
    preview_character_limit: int,
    chunk_size: int,
    chunk_overlap: int,
) -> DocumentUploadResponse:
    """Validate a PDF upload and return its extracted text summary."""
    validated_filename = _validate_pdf_upload(
        filename=filename,
        content_type=content_type,
        content=content,
    )

    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise DocumentProcessingError(
                    "Password-protected PDFs are not supported."
                )

            page_count = document.page_count
            extracted_text = "".join(page.get_text("text") for page in document)
    except DocumentProcessingError:
        raise
    except (RuntimeError, ValueError) as exc:
        raise DocumentProcessingError(
            "The uploaded file is not a readable PDF."
        ) from exc

    chunks = chunk_text(
        extracted_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return DocumentUploadResponse(
        document_id=uuid4(),
        filename=validated_filename,
        page_count=page_count,
        character_count=len(extracted_text),
        preview=_build_preview(extracted_text, preview_character_limit),
        chunk_count=len(chunks),
        chunks=chunks,
    )


def _validate_pdf_upload(
    *,
    filename: str | None,
    content_type: str | None,
    content: bytes,
) -> str:
    """Validate the upload metadata and PDF signature."""
    normalized_filename = (filename or "").strip()
    normalized_content_type = (content_type or "").split(";", maxsplit=1)[0].lower()

    if Path(normalized_filename).suffix.lower() != ".pdf":
        raise UnsupportedDocumentTypeError("The uploaded file must have a .pdf extension.")

    if normalized_content_type not in PDF_CONTENT_TYPES:
        raise UnsupportedDocumentTypeError(
            "The uploaded file must use the application/pdf content type."
        )

    if not content:
        raise DocumentProcessingError("The uploaded PDF is empty.")

    if PDF_HEADER not in content[:PDF_HEADER_SEARCH_LIMIT]:
        raise DocumentProcessingError("The uploaded file does not contain a PDF header.")

    return normalized_filename


def _build_preview(text: str, character_limit: int) -> str:
    """Create a compact, whitespace-normalized text preview."""
    normalized_text = " ".join(text.split())
    return normalized_text[:character_limit]
