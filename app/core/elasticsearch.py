"""Elasticsearch client creation and application lifecycle management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from elasticsearch import Elasticsearch
from fastapi import FastAPI, Request

from app.core.config import get_settings


def create_elasticsearch_client(url: str) -> Elasticsearch:
    """Create a client whose connection pool can be reused across requests."""
    return Elasticsearch(url)


def get_elasticsearch_client(request: Request) -> Elasticsearch:
    """Provide the application-scoped Elasticsearch client to a route."""
    return cast(Elasticsearch, request.app.state.elasticsearch_client)


@asynccontextmanager
async def elasticsearch_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open one Elasticsearch client for the application process and close it."""
    client = create_elasticsearch_client(get_settings().elasticsearch_url)
    app.state.elasticsearch_client = client

    try:
        yield
    finally:
        client.close()
