"""Document endpoint schemas."""

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Summary returned after extracting text from a PDF."""

    filename: str
    page_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    preview: str

