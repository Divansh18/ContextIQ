"""Service health endpoint."""

from fastapi import APIRouter, status

from app.core.config import get_settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check() -> HealthResponse:
    """Report whether the API is available."""
    settings = get_settings()
    return HealthResponse(status="ok", service=settings.app_name)

