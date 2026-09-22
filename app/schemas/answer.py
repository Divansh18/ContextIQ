"""Grounded question-answering request and response schemas."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import MAX_SEARCH_TOP_K


class AskRequest(BaseModel):
    """Validated question for the RAG pipeline."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=MAX_SEARCH_TOP_K)
    document_id: UUID | None = None


class AnswerSource(BaseModel):
    """Retrieved source metadata used to generate an answer."""

    document_id: UUID
    filename: str
    chunk_index: int = Field(ge=0)
    score: float


class AskResponse(BaseModel):
    """Grounded answer and the retrieved chunks that supported it."""

    question: str
    answer: str = Field(min_length=1)
    sources: list[AnswerSource]
