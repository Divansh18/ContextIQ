"""Tests for Elasticsearch client lifecycle and connectivity."""

from unittest.mock import Mock, patch

from elasticsearch import Elasticsearch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.elasticsearch import (
    create_elasticsearch_client,
    elasticsearch_lifespan,
)
from app.services.health_service import is_elasticsearch_ready


@patch("app.core.elasticsearch.Elasticsearch")
def test_create_elasticsearch_client_uses_configured_url(
    elasticsearch_class: Mock,
) -> None:
    client = create_elasticsearch_client("http://elasticsearch.test:9200")

    elasticsearch_class.assert_called_once_with("http://elasticsearch.test:9200")
    assert client is elasticsearch_class.return_value


@patch("app.core.elasticsearch.create_elasticsearch_client")
@patch("app.core.elasticsearch.get_settings")
def test_lifespan_reuses_and_closes_one_client(
    get_settings: Mock,
    create_client: Mock,
) -> None:
    elasticsearch_client = Mock(spec=Elasticsearch)
    get_settings.return_value.elasticsearch_url = "http://elasticsearch.test:9200"
    create_client.return_value = elasticsearch_client
    test_app = FastAPI(lifespan=elasticsearch_lifespan)

    with TestClient(test_app):
        assert test_app.state.elasticsearch_client is elasticsearch_client
        create_client.assert_called_once_with("http://elasticsearch.test:9200")
        elasticsearch_client.close.assert_not_called()

    elasticsearch_client.close.assert_called_once_with()


def test_connectivity_check_uses_client_ping() -> None:
    client = Mock(spec=Elasticsearch)
    client.ping.return_value = True

    assert is_elasticsearch_ready(client) is True
    client.ping.assert_called_once_with()
