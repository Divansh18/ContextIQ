"""Document endpoint schemas."""

from uuid import UUID

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """One character-based segment of extracted document text."""

    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    character_count: int = Field(ge=1)


class DocumentUploadResponse(BaseModel):
    """Summary returned after extracting text from a PDF."""

    document_id: UUID
    filename: str
    page_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    preview: str
    chunk_count: int = Field(ge=0)
    chunks: list[DocumentChunk]
