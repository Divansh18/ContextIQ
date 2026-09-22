"""FastAPI application initialization."""

from fastapi import FastAPI

from app.api.routes import ask, documents, health, search
from app.core.config import get_settings
from app.core.elasticsearch import elasticsearch_lifespan

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=elasticsearch_lifespan,
)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(ask.router)
