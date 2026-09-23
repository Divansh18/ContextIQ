"""Application orchestration for grounded question answering."""

from uuid import UUID

from elasticsearch import Elasticsearch

from app.core.config import Settings
from app.rag.graph import run_rag_workflow
from app.schemas.answer import AnswerSource, AskResponse


def answer_question(
    *,
    question: str,
    top_k: int | None,
    document_id: UUID | None,
    settings: Settings,
    elasticsearch_client: Elasticsearch,
) -> AskResponse:
    """Retrieve supporting chunks and generate a grounded answer."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")

    workflow_state = run_rag_workflow(
        question=normalized_question,
        top_k=top_k,
        document_id=document_id,
        settings=settings,
        elasticsearch_client=elasticsearch_client,
    )
    sources = [
        AnswerSource(
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
        )
        for chunk in workflow_state["retrieved_chunks"]
    ]

    return AskResponse(
        question=normalized_question,
        answer=workflow_state["answer"],
        sources=sources,
    )
