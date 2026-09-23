"""Conditional LangGraph workflow for grounded question answering."""

from typing import Literal, TypedDict, cast
from uuid import UUID

from elasticsearch import Elasticsearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.core.config import Settings
from app.rag.generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    build_retrieved_context,
    generate_grounded_answer,
    to_langchain_documents,
)
from app.rag.retriever import retrieve_semantic_chunks
from app.schemas.search import SemanticSearchResult

RETRIEVE_NODE = "retrieve"
EVALUATE_CONTEXT_NODE = "evaluate_context"
GENERATE_NODE = "generate"
FALLBACK_NODE = "fallback"
SUFFICIENT_ROUTE = "sufficient"
INSUFFICIENT_ROUTE = "insufficient"

ContextRoute = Literal["sufficient", "insufficient"]
StateUpdate = dict[str, object]


class RAGWorkflowState(TypedDict):
    """Values shared between the conditional RAG workflow nodes."""

    question: str
    top_k: int | None
    document_id: UUID | None
    retrieved_chunks: list[SemanticSearchResult]
    enough_context: bool
    answer: str


class RAGWorkflowContext(TypedDict):
    """Run-scoped dependencies kept separate from workflow state."""

    settings: Settings
    elasticsearch_client: Elasticsearch


def retrieve_node(
    state: RAGWorkflowState,
    runtime: Runtime[RAGWorkflowContext],
) -> StateUpdate:
    """Retrieve usable chunks with the existing ContextIQ retriever."""
    settings = runtime.context["settings"]
    retrieved_chunks = retrieve_semantic_chunks(
        query=state["question"],
        top_k=state["top_k"],
        document_id=state["document_id"],
        settings=settings,
        elasticsearch_client=runtime.context["elasticsearch_client"],
    )
    return {
        "retrieved_chunks": [
            chunk for chunk in retrieved_chunks if chunk.text.strip()
        ]
    }


def has_sufficient_context(
    chunks: list[SemanticSearchResult],
    *,
    relevance_threshold: float,
) -> bool:
    """Apply the configured best-score retrieval heuristic."""
    best_score = max((chunk.score for chunk in chunks), default=None)
    return best_score is not None and best_score >= relevance_threshold


def evaluate_context_node(
    state: RAGWorkflowState,
    runtime: Runtime[RAGWorkflowContext],
) -> StateUpdate:
    """Record whether retrieval evidence is strong enough for generation."""
    settings = runtime.context["settings"]
    return {
        "enough_context": has_sufficient_context(
            state["retrieved_chunks"],
            relevance_threshold=settings.rag_relevance_threshold,
        )
    }


def select_context_route(state: RAGWorkflowState) -> ContextRoute:
    """Select the observable route after retrieval evidence evaluation."""
    return SUFFICIENT_ROUTE if state["enough_context"] else INSUFFICIENT_ROUTE


def generate_node(
    state: RAGWorkflowState,
    runtime: Runtime[RAGWorkflowContext],
) -> StateUpdate:
    """Generate an answer with the existing grounded LangChain pipeline."""
    settings = runtime.context["settings"]
    context = build_retrieved_context(
        to_langchain_documents(state["retrieved_chunks"])
    )
    api_key = (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key is not None
        else None
    )
    return {
        "answer": generate_grounded_answer(
            question=state["question"],
            context=context,
            api_key=api_key,
            model=settings.generation_model,
            max_output_tokens=settings.generation_max_output_tokens,
        )
    }


def insufficient_context_node(_: RAGWorkflowState) -> StateUpdate:
    """Return the deterministic fallback without calling the model."""
    return {"answer": INSUFFICIENT_CONTEXT_ANSWER}


def build_rag_workflow() -> CompiledStateGraph[
    RAGWorkflowState,
    RAGWorkflowContext,
    RAGWorkflowState,
    RAGWorkflowState,
]:
    """Compile the ContextIQ conditional RAG workflow."""
    workflow = StateGraph(
        RAGWorkflowState,
        context_schema=RAGWorkflowContext,
    )
    workflow.add_node(RETRIEVE_NODE, retrieve_node)
    workflow.add_node(EVALUATE_CONTEXT_NODE, evaluate_context_node)
    workflow.add_node(GENERATE_NODE, generate_node)
    workflow.add_node(FALLBACK_NODE, insufficient_context_node)

    workflow.add_edge(START, RETRIEVE_NODE)
    workflow.add_edge(RETRIEVE_NODE, EVALUATE_CONTEXT_NODE)
    workflow.add_conditional_edges(
        EVALUATE_CONTEXT_NODE,
        select_context_route,
        {
            SUFFICIENT_ROUTE: GENERATE_NODE,
            INSUFFICIENT_ROUTE: FALLBACK_NODE,
        },
    )
    workflow.add_edge(GENERATE_NODE, END)
    workflow.add_edge(FALLBACK_NODE, END)

    return workflow.compile(name="contextiq-rag")


RAG_WORKFLOW = build_rag_workflow()


def run_rag_workflow(
    *,
    question: str,
    top_k: int | None,
    document_id: UUID | None,
    settings: Settings,
    elasticsearch_client: Elasticsearch,
) -> RAGWorkflowState:
    """Run one conditional RAG workflow and return its final internal state."""
    result = RAG_WORKFLOW.invoke(
        RAGWorkflowState(
            question=question,
            top_k=top_k,
            document_id=document_id,
            retrieved_chunks=[],
            enough_context=False,
            answer="",
        ),
        context=RAGWorkflowContext(
            settings=settings,
            elasticsearch_client=elasticsearch_client,
        ),
    )
    return cast(RAGWorkflowState, result)
