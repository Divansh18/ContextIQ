# ContextIQ

ContextIQ is a lightweight knowledge assistant built incrementally to demonstrate a clean retrieval-augmented generation (RAG) architecture. The first milestone provides a FastAPI service that accepts a PDF and extracts its text metadata with PyMuPDF.

## Current capabilities

- Service health endpoint
- Single-PDF upload with file type and content validation
- PDF text extraction, page count, character count, and a short preview
- Environment-backed application configuration
- API tests

LangChain, LangGraph, embeddings, vector storage, semantic search, LLM calls, authentication, and a frontend are intentionally deferred to later milestones.

## Architecture

```text
app/
├── main.py                 # Application creation and router registration
├── api/routes/             # HTTP request handling
├── core/config.py          # Application settings
├── schemas/                # Pydantic API contracts
├── services/               # Application and document-processing logic
└── rag/                    # Future RAG pipeline modules
tests/                      # API tests
```

The document upload flow is deliberately one-directional:

```text
API route -> document service -> PyMuPDF
```

## Requirements

- Python 3.11 or newer

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
```

The defaults work without a `.env` file. Copying the example is only necessary when you want to change a setting.

## Run the API

```bash
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`, and its interactive documentation is at `http://127.0.0.1:8000/docs`.

## Example requests

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F 'file=@/absolute/path/to/document.pdf;type=application/pdf'
```

Successful uploads return this shape:

```json
{
  "filename": "document.pdf",
  "page_count": 3,
  "character_count": 8421,
  "preview": "The beginning of the extracted document text..."
}
```

## Run tests

```bash
pytest
```

# ContextIQ
