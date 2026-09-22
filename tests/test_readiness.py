"""Tests for the Elasticsearch readiness endpoint."""

from unittest.mock import Mock

from elasticsearch import Elasticsearch
from fastapi.testclient import TestClient

from app.core.elasticsearch import get_elasticsearch_client
from app.main import app


def test_readiness_reports_available_elasticsearch(client: TestClient) -> None:
    elasticsearch_client = Mock(spec=Elasticsearch)
    elasticsearch_client.ping.return_value = True
    app.dependency_overrides[get_elasticsearch_client] = lambda: elasticsearch_client

    try:
        response = client.get("/ready")
    finally:
        app.dependency_overrides.pop(get_elasticsearch_client, None)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "ContextIQ",
        "elasticsearch": "ok",
    }


def test_readiness_reports_unavailable_elasticsearch(client: TestClient) -> None:
    elasticsearch_client = Mock(spec=Elasticsearch)
    elasticsearch_client.ping.return_value = False
    app.dependency_overrides[get_elasticsearch_client] = lambda: elasticsearch_client

    try:
        response = client.get("/ready")
    finally:
        app.dependency_overrides.pop(get_elasticsearch_client, None)

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "service": "ContextIQ",
        "elasticsearch": "unavailable",
    }
