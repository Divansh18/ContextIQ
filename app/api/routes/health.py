"""Service liveness and readiness endpoints."""

from typing import Annotated

from elasticsearch import Elasticsearch
from fastapi import APIRouter, Depends, Response, status

from app.core.config import get_settings
from app.core.elasticsearch import get_elasticsearch_client
from app.schemas.health import HealthResponse, ReadinessResponse
from app.services.health_service import is_elasticsearch_ready

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check() -> HealthResponse:
    """Report whether the API is available."""
    settings = get_settings()
    return HealthResponse(status="ok", service=settings.app_name)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
def readiness_check(
    response: Response,
    elasticsearch_client: Annotated[
        Elasticsearch,
        Depends(get_elasticsearch_client),
    ],
) -> ReadinessResponse:
    """Report whether external dependencies required for search are available."""
    settings = get_settings()
    if is_elasticsearch_ready(elasticsearch_client):
        return ReadinessResponse(
            status="ready",
            service=settings.app_name,
            elasticsearch="ok",
        )

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="not_ready",
        service=settings.app_name,
        elasticsearch="unavailable",
    )
