"""Health endpoint schemas."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok"]
    service: str


class ReadinessResponse(BaseModel):
    """Application dependency readiness response."""

    status: Literal["ready", "not_ready"]
    service: str
    elasticsearch: Literal["ok", "unavailable"]
