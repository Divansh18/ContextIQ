"""Semantic-search request and response schemas."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import MAX_SEARCH_TOP_K


class SemanticSearchRequest(BaseModel):
    """Validated semantic-search input."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=MAX_SEARCH_TOP_K)
    document_id: UUID | None = None


class SemanticSearchResult(BaseModel):
    """One document chunk returned by vector similarity search."""

    document_id: UUID
    filename: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    character_count: int = Field(ge=1)
    score: float


class SemanticSearchResponse(BaseModel):
    """Semantic-search results for one query."""

    query: str
    results: list[SemanticSearchResult]
