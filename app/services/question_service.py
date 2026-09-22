"""Application orchestration for grounded question answering."""

from uuid import UUID

from elasticsearch import Elasticsearch

from app.core.config import Settings
from app.rag.generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    build_retrieved_context,
    generate_grounded_answer,
    to_langchain_documents,
)
from app.rag.retriever import retrieve_semantic_chunks
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

    retrieved_chunks = retrieve_semantic_chunks(
        query=normalized_question,
        top_k=top_k,
        document_id=document_id,
        settings=settings,
        elasticsearch_client=elasticsearch_client,
    )
    context_chunks = [chunk for chunk in retrieved_chunks if chunk.text.strip()]
    sources = [
        AnswerSource(
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
        )
        for chunk in context_chunks
    ]
    context_documents = to_langchain_documents(context_chunks)
    context = build_retrieved_context(context_documents)
    if not context:
        return AskResponse(
            question=normalized_question,
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            sources=[],
        )

    api_key = (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key is not None
        else None
    )
    answer = generate_grounded_answer(
        question=normalized_question,
        context=context,
        api_key=api_key,
        model=settings.generation_model,
        max_output_tokens=settings.generation_max_output_tokens,
    )

    return AskResponse(
        question=normalized_question,
        answer=answer,
        sources=sources,
    )
