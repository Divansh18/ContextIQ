"""Tests for document upload behavior."""

import pymupdf
from fastapi.testclient import TestClient


def _create_pdf(page_texts: list[str]) -> bytes:
    document = pymupdf.open()
    try:
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


def test_upload_pdf_returns_extracted_summary(client: TestClient) -> None:
    pdf_bytes = _create_pdf(["ContextIQ first page", "Knowledge assistant second page"])

    response = client.post(
        "/documents/upload",
        files={"file": ("knowledge.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "knowledge.pdf"
    assert payload["page_count"] == 2
    assert payload["character_count"] > 0
    assert "ContextIQ first page" in payload["preview"]
    assert "Knowledge assistant second page" in payload["preview"]


def test_upload_rejects_non_pdf_file_type(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "The uploaded file must have a .pdf extension."
    }


def test_upload_rejects_non_pdf_content_type(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.pdf", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "The uploaded file must use the application/pdf content type."
    }


def test_upload_rejects_malformed_pdf(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("broken.pdf", b"%PDF-not-valid", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "The uploaded file is not a readable PDF."}
