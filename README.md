# ContextIQ

ContextIQ is a lightweight knowledge assistant built incrementally to demonstrate
a clean retrieval-augmented generation (RAG) architecture. The backend ingests
PDFs, embeds their chunks, stores them in Elasticsearch, and retrieves
semantically related passages before generating grounded answers. A minimal
Next.js interface makes that FastAPI workflow easy to demonstrate.

## Current capabilities

- Service health endpoint
- Single-PDF upload with file type and content validation
- PDF text extraction, page count, character count, and a short preview
- Framework-independent character chunking with configurable overlap
- Elasticsearch 9.x local infrastructure and an explicit chunk index mapping
- Batched OpenAI embeddings with `text-embedding-3-small`
- Bulk indexing of embedded chunks into Elasticsearch
- Semantic vector search across all documents or within one document
- Grounded answer generation with retrieved source metadata and a focused
  LangChain LCEL pipeline
- Conditional LangGraph orchestration with deterministic weak-context fallback
- Document-scoped Next.js upload and grounded-question workspace
- Separate application liveness and Elasticsearch readiness checks
- Environment-backed application configuration
- API tests

Authentication and persistent user sessions are intentionally deferred to later
milestones. LangChain is used only at the answer-generation boundary, while
LangGraph orchestrates retrieval, evidence evaluation, and conditional routing.
Ingestion, indexing, and retrieval implementations remain explicit application
code.

## Architecture

```text
app/
├── main.py                 # Application creation and router registration
├── api/routes/             # HTTP request handling
├── core/                   # Application settings and client lifecycle
├── schemas/                # Pydantic API contracts
├── services/               # Application and document-processing logic
└── rag/                    # Chunking, embeddings, retrieval, and generation
tests/                      # API tests
frontend/
├── app/                    # Next.js App Router page and global styles
├── components/             # Upload, question, answer, and source UI
├── lib/                    # Typed backend API client
└── types/                  # FastAPI contract types
```

The document upload flow is deliberately one-directional:

```text
API route
  -> document ingestion service
  -> PDF extraction with PyMuPDF
  -> character chunking
  -> one batched OpenAI embedding request
  -> Elasticsearch bulk indexing
```

The semantic-search flow reuses those same infrastructure components:

```text
POST /search
  -> semantic retriever
  -> OpenAI query embedding
  -> Elasticsearch kNN vector search
  -> ranked document chunks with scores
```

The grounded answer flow builds directly on semantic search:

```text
POST /ask
  -> question service
  -> LangGraph retrieve node
  -> existing semantic retriever                 (ContextIQ)
  -> LangGraph evaluate-context node
  -> best score >= configured threshold?
       |-- yes -> generate node
       |          -> ChatPromptTemplate          (LangChain)
       |          -> ChatOpenAI                  (LangChain)
       |          -> StrOutputParser             (LangChain)
       |
       `-- no  -> deterministic fallback node    (no model call)
  -> answer plus source metadata                 (ContextIQ)
```

## Requirements

- Python 3.11 or newer
- Node.js 20.9 or newer
- Docker for local Elasticsearch

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
```

Set your API key in `.env` before uploading a PDF:

```dotenv
OPENAI_API_KEY=your-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
CONTEXTIQ_SEARCH_DEFAULT_TOP_K=3
CONTEXTIQ_SEARCH_NUM_CANDIDATES_MULTIPLIER=10
GENERATION_MODEL=gpt-5.6-luna
CONTEXTIQ_GENERATION_MAX_OUTPUT_TOKENS=300
CONTEXTIQ_RAG_RELEVANCE_THRESHOLD=0.60
CONTEXTIQ_FRONTEND_ORIGIN=http://localhost:3000
```

Never commit `.env`; it is excluded by `.gitignore`. The API can start without
an OpenAI key, but uploads containing extractable text, semantic searches, and
`/ask` requests that reach generation return a clear configuration error until
one is supplied. A weak-retrieval `/ask` request can take the deterministic
fallback branch without calling the generation model.

Install the frontend dependencies and create its local environment file:

```bash
cd frontend
npm install
cp .env.example .env.local
```

The browser-visible backend URL is configured separately in
`frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

`CONTEXTIQ_FRONTEND_ORIGIN` controls the one browser origin FastAPI permits for
cross-origin requests. It defaults to `http://localhost:3000`; wildcard origins
are not enabled.

## Elasticsearch

ContextIQ uses Elasticsearch as its search and vector store. It holds each
extracted chunk alongside searchable text, document metadata, and an embedding
vector. This allows later phases to combine conventional text search with
semantic nearest-neighbor retrieval.

Start the local single-node Elasticsearch service:

```bash
docker compose up -d elasticsearch
docker compose ps
```

The Compose setup binds Elasticsearch only to `127.0.0.1:9200`, disables
security for local development, and persists data in a named Docker volume.
Verify the cluster directly with:

```bash
curl http://localhost:9200
curl 'http://localhost:9200/_cluster/health?pretty'
```

Create the ContextIQ index if it does not already exist:

```bash
python -m app.scripts.initialize_elasticsearch
```

An Elasticsearch index is the named collection and schema that will contain
ContextIQ's chunk records. The explicit mapping is:

```json
{
  "document_id": "keyword",
  "filename": "keyword",
  "chunk_index": "integer",
  "text": "text",
  "character_count": "integer",
  "embedding": {
    "type": "dense_vector",
    "dims": 1536,
    "index": true,
    "similarity": "cosine"
  }
}
```

The intended stored document shape is:

```json
{
  "document_id": "7d2f4ea0-...",
  "filename": "document.pdf",
  "chunk_index": 0,
  "text": "Extracted document text...",
  "character_count": 950,
  "embedding": "<1536 floating-point values omitted>"
}
```

### Embeddings

An embedding is a numeric representation of a chunk's meaning. ContextIQ sends
all chunks from one upload to OpenAI in one batched embeddings request, validates
every returned vector, and stores each vector with the original text and
metadata. Keeping the source text allows later retrieval results to be cited and
used as RAG context; the vector is used for similarity matching rather than as a
replacement for the text.

ContextIQ uses `text-embedding-3-small` because it is OpenAI's focused small
embedding model and is suitable for search and relatedness tasks. Its default
output contains 1,536 values, so the application and Elasticsearch mapping both
use exactly 1,536 dimensions. See the
[official OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings).

Embedding vectors are internal and are never returned by the upload endpoint.

### Semantic search

Semantic search compares meaning rather than requiring the query and source text
to contain the same words. ContextIQ embeds the user's query with the same
`text-embedding-3-small` model and 1,536 dimensions used for document chunks.
Using the same model and dimensions places queries and chunks in the same vector
space, where their relative closeness is meaningful.

Elasticsearch performs an approximate k-nearest-neighbor (kNN) search over the
indexed `embedding` field. `top_k` is the maximum number of closest chunks to
return; it defaults to 3 and accepts values from 1 through 100. ContextIQ asks
Elasticsearch to consider `10 * top_k` candidates per shard. This modestly favors
retrieval recall for a small local project while remaining simple and bounded;
it is a starting point rather than production-scale tuning.
See Elasticsearch's
[search API documentation](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-search)
for the supported kNN parameters.

The optional `document_id` is added as an Elasticsearch kNN pre-filter. It can
restrict retrieval to one uploaded PDF without fetching unrelated chunks and
filtering them in Python.

The index uses cosine similarity, which compares vector direction at a high
level: chunks whose vectors point in similar directions represent more closely
related meaning. Elasticsearch converts that similarity into each result's
`_score`, so larger scores rank ahead of smaller scores.

Keyword search looks for literal terms or analyzed word matches. Semantic search
can retrieve a chunk about "object storage" for a question about where to keep
uploaded files even when the wording differs.

### Retrieval-augmented generation

RAG has three direct stages in ContextIQ, coordinated by a conditional graph:

1. **Retrieval:** the existing semantic retriever embeds the question and asks
   Elasticsearch for the nearest chunks.
2. **Augmentation:** ContextIQ formats those chunks into deterministic,
   clearly delimited `[SOURCE n]` blocks containing document metadata, retrieval
   score, and source text.
3. **Routing and generation:** a retrieval-score gate either sends the source
   blocks through the focused LangChain pipeline or returns the deterministic
   insufficient-context answer without a generation call.

ContextIQ uses `gpt-5.6-luna` by default because OpenAI describes it as the
cost-sensitive GPT-5.6 model and documents support for the Responses API. The
model remains configurable through `GENERATION_MODEL`. See the
[official model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
and [OpenAI API quickstart](https://developers.openai.com/api/docs/quickstart).

The grounding instructions tell the model not to use outside knowledge, not to
invent unsupported details, and to state when the retrieved context is
insufficient. Grounding reduces hallucination risk by limiting the evidence
available to the generation step, although application-level evaluation remains
important for any production use.

Every usable retrieved chunk is represented in the API response by its
`document_id`, filename, chunk index, and retrieval score—even when weak
evidence routes to the fallback. This preserves retrieval visibility without
returning chunk embeddings or exposing the internal generation prompt.

`POST /search` stops after retrieval and returns chunk text for inspecting search
quality. `POST /ask` performs the same retrieval, augments the prompt with those
chunks, and returns a concise generated answer plus source metadata.

### LangChain integration

ContextIQ first implemented the complete RAG flow manually so that PDF
extraction, chunking, embeddings, vector indexing, kNN retrieval, grounding, and
source attribution stayed visible. LangChain is introduced only after that
baseline, and only where it removes model-call plumbing without hiding the
retrieval mechanics.

The generation pipeline uses the focused `langchain-core` and
`langchain-openai` packages rather than the broader `langchain` package:

```python
GROUNDING_PROMPT | chat_model | StrOutputParser()
```

- `ChatPromptTemplate` owns the system grounding instructions and the human
  question/context placeholders.
- `ChatOpenAI` is the chat-model adapter. It uses the configured
  `GENERATION_MODEL`, API key, output-token limit, and OpenAI Responses API.
- `StrOutputParser` converts the returned AI message into the plain string used
  by the existing response schema.
- The LCEL `|` operator composes those pieces into one `Runnable` pipeline,
  keeping the stages independently testable and leaving room for standard
  runnable capabilities such as asynchronous, batch, or streaming execution in
  future phases.

A small adapter converts retrieved ContextIQ chunks into LangChain `Document`
objects. The chunk text becomes `page_content`; document ID, filename, chunk
index, and score are retained as metadata. ContextIQ still formats the final
source blocks itself and still constructs the public `sources` array from its
own retrieval models, so no embedding vector or internal prompt enters the API
response.

The LangChain boundary is intentionally narrow:

```text
Manual / framework-independent                 LangChain
--------------------------------------------   ---------------------------
PDF extraction and character chunking          ChatPromptTemplate
OpenAI embedding generation and validation      ChatOpenAI
Elasticsearch bulk indexing and kNN retrieval   LCEL Runnable composition
top_k and document_id filtering                 StrOutputParser
source selection, metadata, and API schemas
```

`POST /search` remains completely independent of LangChain. It continues to use
the manual query-embedding and Elasticsearch retriever path, so retrieval
quality can be inspected separately from answer generation.

See the official LangChain documentation for
[ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai),
[runnable sequences](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence),
[ChatPromptTemplate](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate),
and [StrOutputParser](https://reference.langchain.com/python/langchain-core/output_parsers/string/StrOutputParser).

### LangGraph conditional workflow

The Phase 6 LangChain pipeline was deliberately linear: once context existed,
the application formatted a prompt, called one model, and parsed one answer. A
graph would not have improved that sequence by itself. LangGraph is introduced
now because ContextIQ has a real branch: strong retrieval evidence proceeds to
generation, while weak or empty retrieval returns a deterministic fallback and
avoids an unnecessary model call.

The graph carries a small typed state containing only:

- the normalized question, requested `top_k`, and optional `document_id`
- retrieved ContextIQ chunks
- the `enough_context` routing decision
- the final answer

Settings and the Elasticsearch client are passed through LangGraph's run-scoped
context and are never put in graph state. The graph is compiled once, while
those dependencies can vary per invocation. Graph internals are also absent
from the public API response.

```text
START
  |
  v
retrieve -- existing embedding + Elasticsearch kNN retriever
  |
  v
evaluate_context -- compare best retrieval score with configured threshold
  |
  +-- sufficient --> generate -- existing LangChain LCEL pipeline --> END
  |
  `-- insufficient -> fallback -- deterministic answer, no model call --> END
```

The nodes have focused responsibilities:

1. `retrieve` delegates to the existing framework-independent semantic
   retriever and records usable chunks in state.
2. `evaluate_context` compares the best chunk score with
   `CONTEXTIQ_RAG_RELEVANCE_THRESHOLD`.
3. A LangGraph conditional edge chooses `generate` or `fallback`.
4. `generate` reuses the existing `ChatPromptTemplate | ChatOpenAI |
   StrOutputParser` pipeline; `fallback` returns the existing fixed
   insufficient-context answer.

The default threshold is `0.60`, and equality is considered sufficient. That
value conservatively separates the current known relevant examples (about
`0.62` for S3 and `0.75` for ECS/Fargate) from the unrelated France example
(about `0.52`). It is only a project-level heuristic—not a universal relevance
score. Elasticsearch scores depend on the embedding model, index contents, and
dataset, so this threshold must be evaluated and tuned on a larger
representative set before production use.

Responsibility boundaries remain explicit:

```text
Framework-independent          LangChain                 LangGraph
----------------------------   -----------------------   ----------------------
PDF extraction and chunking    ChatPromptTemplate        typed workflow state
embedding and bulk indexing    ChatOpenAI                node orchestration
Elasticsearch kNN retrieval    LCEL generation chain     relevance evaluation
filters and source metadata    StrOutputParser           conditional routing
```

`POST /search` does not invoke LangGraph or LangChain. It remains the direct way
to inspect the underlying manual semantic retrieval behavior. See the official
LangGraph documentation for the
[Graph API and conditional edges](https://docs.langchain.com/oss/python/langgraph/graph-api)
and [`StateGraph` reference](https://reference.langchain.com/python/langgraph/graph/state/StateGraph).

### Recreating an empty development index

Indices created before embedding dimensions were finalized do not contain an
explicit `dims: 1536` mapping. ContextIQ detects this and refuses to modify or
delete the index automatically.

For the local development index only, first confirm it is empty:

```bash
curl 'http://localhost:9200/contextiq-document-chunks/_count?pretty'
```

If and only if the response reports `"count": 0`, delete that exact local index
and initialize it again:

```bash
curl -X DELETE 'http://localhost:9200/contextiq-document-chunks'
python -m app.scripts.initialize_elasticsearch
curl 'http://localhost:9200/contextiq-document-chunks/_mapping?pretty'
```

Do not use this deletion process for a non-empty or shared index. Such an index
requires a migration and reindexing plan so existing data is preserved.

Stop the local service without deleting its data:

```bash
docker compose down
```

## Run the API

```bash
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`, and its interactive documentation is at `http://127.0.0.1:8000/docs`.

## Run the frontend

In a second terminal, with FastAPI and Elasticsearch already running:

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`. The local services use these ports:

| Service | URL | Purpose |
|---|---|---|
| Next.js | `http://localhost:3000` | Browser interface |
| FastAPI | `http://localhost:8000` | Upload, search, and answer APIs |
| Elasticsearch | `http://localhost:9200` | Local vector index |

### End-to-end usage

1. Start Elasticsearch with `docker compose up -d elasticsearch`.
2. Start FastAPI with `uvicorn app.main:app --reload`.
3. Start Next.js from `frontend/` with `npm run dev`.
4. Open the frontend and choose a PDF.
5. Select **Upload & index** and wait for the indexed document summary.
6. Ask a question. The frontend automatically includes that upload's
   `document_id`, keeping retrieval scoped to the active PDF.
7. Review the answer, chunk indexes, filenames, and rounded retrieval scores.

The deterministic insufficient-context response is displayed as a normal
answer. Embeddings and internal LangGraph state are never displayed.

## Example requests

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F 'file=@/absolute/path/to/document.pdf;type=application/pdf'
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"How can I deploy containers without maintaining virtual machines?","top_k":3}'
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which AWS service lets me run containers without managing virtual machines?","top_k":3}'
```

Successful uploads return this shape:

```json
{
  "document_id": "7d2f4ea0-...",
  "filename": "document.pdf",
  "page_count": 3,
  "character_count": 8421,
  "preview": "The beginning of the extracted document text...",
  "chunk_count": 11,
  "chunks": [
    {
      "chunk_index": 0,
      "text": "The beginning of the extracted document text...",
      "character_count": 1000
    }
  ]
}
```

The response deliberately contains chunk text and metadata for compatibility,
but never includes embedding vectors.

A successful semantic-search response has this shape:

```json
{
  "query": "How can I deploy containers without maintaining virtual machines?",
  "results": [
    {
      "document_id": "7d2f4ea0-...",
      "filename": "document.pdf",
      "chunk_index": 0,
      "text": "AWS Fargate lets developers run containers...",
      "character_count": 1000,
      "score": 0.91
    }
  ]
}
```

To search within one PDF, include its upload response ID as `document_id`. Search
responses deliberately omit the stored embedding vectors.

A successful grounded-answer response has this shape:

```json
{
  "question": "Which AWS service lets me run containers without managing virtual machines?",
  "answer": "Amazon ECS with AWS Fargate lets developers run containers without directly managing EC2 virtual machines.",
  "sources": [
    {
      "document_id": "7d2f4ea0-...",
      "filename": "document.pdf",
      "chunk_index": 1,
      "score": 0.67
    }
  ]
}
```

`POST /ask` also accepts an optional `document_id` to constrain both retrieval
and answer generation to one uploaded PDF. If retrieval produces no usable
context, ContextIQ returns a grounded insufficient-context response without
calling the generation model.

Chunking defaults to 1,000 characters per chunk with 200 characters of
overlap. Set `CONTEXTIQ_CHUNK_SIZE` and `CONTEXTIQ_CHUNK_OVERLAP` to override
those values. The overlap must always be smaller than the chunk size.

## Run tests

```bash
pytest
```

Frontend quality checks run from `frontend/`:

```bash
npm run lint
npm run typecheck
npm run build
```
