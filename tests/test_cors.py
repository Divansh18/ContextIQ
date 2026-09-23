"""Tests for the local frontend CORS policy."""

from fastapi.testclient import TestClient


def test_configured_frontend_origin_can_post(client: TestClient) -> None:
    response = client.options(
        "/ask",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:3000"
    )
    assert "POST" in response.headers["access-control-allow-methods"]


def test_unconfigured_origin_is_not_allowed(client: TestClient) -> None:
    response = client.options(
        "/ask",
        headers={
            "Origin": "http://malicious.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
