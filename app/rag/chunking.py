"""Framework-independent document chunking."""

from app.core.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    validate_chunk_configuration,
)
from app.schemas.document import DocumentChunk


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split text into fixed-size, overlapping character chunks."""
    validate_chunk_configuration(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not text or not text.strip():
        return []

    chunks: list[DocumentChunk] = []
    step_size = chunk_size - chunk_overlap

    for start in range(0, len(text), step_size):
        chunk_text_value = text[start : start + chunk_size]
        if chunk_text_value.strip():
            chunks.append(
                DocumentChunk(
                    chunk_index=len(chunks),
                    text=chunk_text_value,
                    character_count=len(chunk_text_value),
                )
            )

        if start + chunk_size >= len(text):
            break

    return chunks
