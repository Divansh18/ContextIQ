"""Unit tests for the conditional LangGraph RAG workflow."""

from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.rag.generation import GenerationAPIError, INSUFFICIENT_CONTEXT_ANSWER
from app.rag.graph import (
    EVALUATE_CONTEXT_NODE,
    FALLBACK_NODE,
    GENERATE_NODE,
    INSUFFICIENT_ROUTE,
    RETRIEVE_NODE,
    SUFFICIENT_ROUTE,
    RAGWorkflowContext,
    RAGWorkflowState,
    build_rag_workflow,
    has_sufficient_context,
    run_rag_workflow,
    select_context_route,
)
from app.rag.vector_store import VectorSearchError
from app.schemas.search import SemanticSearchResult

DOCUMENT_ID = UUID("12345678-1234-5678-1234-567812345678")


def _settings(*, threshold: float = 0.60) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key=SecretStr("test-key"),
        embedding_dimensions=3,
        generation_model="gpt-test",
        generation_max_output_tokens=250,
        rag_relevance_threshold=threshold,
    )


def _chunk(
    *,
    score: float,
    text: str = "AWS Fargate runs containers without managing virtual machines.",
    index: int = 1,
) -> SemanticSearchResult:
    return SemanticSearchResult(
        document_id=DOCUMENT_ID,
        filename="knowledge.pdf",
        chunk_index=index,
        text=text,
        character_count=len(text),
        score=score,
    )


def _initial_state(
    *,
    top_k: int | None = 3,
    document_id: UUID | None = None,
) -> RAGWorkflowState:
    return RAGWorkflowState(
        question="Which service runs containers?",
        top_k=top_k,
        document_id=document_id,
        retrieved_chunks=[],
        enough_context=False,
        answer="",
    )


def _workflow_context() -> RAGWorkflowContext:
    return RAGWorkflowContext(
        settings=_settings(),
        elasticsearch_client=Mock(),
    )


@patch("app.rag.graph.generate_grounded_answer", return_value="Grounded answer")
@patch("app.rag.graph.retrieve_semantic_chunks")
def test_graph_starts_at_retrieval_and_updates_shared_state(
    retrieve_chunks: Mock,
    generate_answer: Mock,
) -> None:
    chunk = _chunk(score=0.75)
    retrieve_chunks.return_value = [chunk]
    workflow = build_rag_workflow()

    updates = list(
        workflow.stream(
            _initial_state(),
            context=_workflow_context(),
            stream_mode="updates",
        )
    )

    assert [next(iter(update)) for update in updates] == [
        RETRIEVE_NODE,
        EVALUATE_CONTEXT_NODE,
        GENERATE_NODE,
    ]
    assert updates[0][RETRIEVE_NODE]["retrieved_chunks"] == [chunk]
    generate_answer.assert_called_once()


@patch("app.rag.graph.generate_grounded_answer", return_value="Grounded answer")
@patch("app.rag.graph.retrieve_semantic_chunks")
def test_sufficient_context_routes_to_generation_and_passes_inputs(
    retrieve_chunks: Mock,
    generate_answer: Mock,
) -> None:
    chunk = _chunk(score=0.75)
    retrieve_chunks.return_value = [chunk]
    client = Mock()
    settings = _settings()

    result = run_rag_workflow(
        question="Which service runs containers?",
        top_k=2,
        document_id=DOCUMENT_ID,
        settings=settings,
        elasticsearch_client=client,
    )

    retrieve_chunks.assert_called_once_with(
        query="Which service runs containers?",
        top_k=2,
        document_id=DOCUMENT_ID,
        settings=settings,
        elasticsearch_client=client,
    )
    generation_call = generate_answer.call_args.kwargs
    assert generation_call["question"] == "Which service runs containers?"
    assert "AWS Fargate runs containers" in generation_call["context"]
    assert generation_call["api_key"] == "test-key"
    assert generation_call["model"] == "gpt-test"
    assert generation_call["max_output_tokens"] == 250
    assert result["retrieved_chunks"] == [chunk]
    assert result["enough_context"] is True
    assert result["answer"] == "Grounded answer"
    assert select_context_route(result) == SUFFICIENT_ROUTE


@patch("app.rag.graph.generate_grounded_answer")
@patch("app.rag.graph.retrieve_semantic_chunks")
def test_insufficient_score_routes_to_fallback_without_generation(
    retrieve_chunks: Mock,
    generate_answer: Mock,
) -> None:
    weak_chunk = _chunk(score=0.52)
    retrieve_chunks.return_value = [weak_chunk]

    result = run_rag_workflow(
        question="What is the capital of France?",
        top_k=3,
        document_id=None,
        settings=_settings(),
        elasticsearch_client=Mock(),
    )

    assert result["retrieved_chunks"] == [weak_chunk]
    assert result["enough_context"] is False
    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert select_context_route(result) == INSUFFICIENT_ROUTE
    generate_answer.assert_not_called()


@pytest.mark.parametrize(
    "retrieved_chunks",
    [[], [_chunk(score=0.80, text="   ")]],
)
@patch("app.rag.graph.generate_grounded_answer")
@patch("app.rag.graph.retrieve_semantic_chunks")
def test_empty_usable_retrieval_routes_to_fallback(
    retrieve_chunks: Mock,
    generate_answer: Mock,
    retrieved_chunks: list[SemanticSearchResult],
) -> None:
    retrieve_chunks.return_value = retrieved_chunks

    result = run_rag_workflow(
        question="Question",
        top_k=None,
        document_id=None,
        settings=_settings(),
        elasticsearch_client=Mock(),
    )

    assert result["retrieved_chunks"] == []
    assert result["enough_context"] is False
    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    generate_answer.assert_not_called()


def test_relevance_threshold_boundary_is_inclusive() -> None:
    assert has_sufficient_context(
        [_chunk(score=0.60)],
        relevance_threshold=0.60,
    )
    assert not has_sufficient_context(
        [_chunk(score=0.599999)],
        relevance_threshold=0.60,
    )


@patch("app.rag.graph.generate_grounded_answer")
@patch("app.rag.graph.retrieve_semantic_chunks")
def test_retrieval_failure_is_not_swallowed(
    retrieve_chunks: Mock,
    generate_answer: Mock,
) -> None:
    retrieve_chunks.side_effect = VectorSearchError("search failed")

    with pytest.raises(VectorSearchError, match="search failed"):
        run_rag_workflow(
            question="Question",
            top_k=3,
            document_id=None,
            settings=_settings(),
            elasticsearch_client=Mock(),
        )

    generate_answer.assert_not_called()


@patch("app.rag.graph.generate_grounded_answer")
@patch("app.rag.graph.retrieve_semantic_chunks")
def test_generation_failure_is_not_swallowed(
    retrieve_chunks: Mock,
    generate_answer: Mock,
) -> None:
    retrieve_chunks.return_value = [_chunk(score=0.75)]
    generate_answer.side_effect = GenerationAPIError("generation failed")

    with pytest.raises(GenerationAPIError, match="generation failed"):
        run_rag_workflow(
            question="Question",
            top_k=3,
            document_id=None,
            settings=_settings(),
            elasticsearch_client=Mock(),
        )


def test_graph_declares_both_conditional_destinations() -> None:
    workflow = build_rag_workflow()
    graph = workflow.get_graph()
    conditional_destinations = {
        edge.target
        for edge in graph.edges
        if edge.source == EVALUATE_CONTEXT_NODE and edge.conditional
    }

    assert conditional_destinations == {GENERATE_NODE, FALLBACK_NODE}
