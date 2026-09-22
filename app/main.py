"""FastAPI application initialization."""

from fastapi import FastAPI

from app.api.routes import documents, health
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(health.router)
app.include_router(documents.router)

