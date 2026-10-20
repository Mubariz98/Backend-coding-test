# PDF RAG API

A FastAPI service that lets you upload PDF documents and search them with natural-language queries. It works in any language: an English question finds German passages and the other way round.

Everything runs locally. There's no API key, no Docker and no external server.

## Quick start

Requires Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -r requirements-dev.txt                    # runtime + test dependencies
cp .env.example .env                                   # optional; defaults work as-is
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs, upload a PDF under **POST /documents**, then ask questions under **POST /query**. The first start downloads the embedding model (~220 MB) into `data/models/`.

On Debian/Ubuntu, `python3 -m venv` needs the `python3-venv` package. Alternatively, use [uv](https://docs.astral.sh/uv/).

## How it works

```
Upload:  PDF ─► hash (reject duplicates) ─► extract text per page ─► strip repeated headers/footers
             ─► split into overlapping, word-aligned chunks ─► embed ─► vector DB
Query:   question ─► embed (same model) ─► nearest-neighbour search ─► top-k chunks (+ file, page, score)
```

| Component | Choice | Why |
|---|---|---|
| Vector DB | **Qdrant** (default) or **ChromaDB**, chosen with `VECTOR_DB` | Both run embedded and save to `data/`, so no server is needed. Qdrant can later point at a real server with one line. |
| Embeddings | [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) via [FastEmbed](https://github.com/qdrant/fastembed) | Covers 50+ languages, runs on CPU through ONNX, with no PyTorch and no API key |
| PDF parsing | `pypdf[crypto]` | The `[crypto]` extra is needed for AES-encrypted PDFs |
| API / validation | FastAPI + Pydantic v2 | Typed request and response contracts, with OpenAPI docs for free |

## Architecture

The code is organised **by domain**, not by technical layer. It's a lightweight take on Domain-Driven Design combined with **Ports & Adapters** (hexagonal architecture).

```
app/
├── main.py                    # create_app(): lifespan, error handlers, routers. No logic.
│
├── documents/                 # Bounded context: getting PDFs IN
│   ├── router.py              #   POST /documents (HTTP only)
│   ├── schemas.py             #   UploadResponse (Pydantic, API contract)
│   ├── service.py             #   IngestionService: hash → parse → clean → chunk → embed → store
│   ├── models.py              #   IngestedDocument (domain object)
│   ├── pdf_parser.py          #   bytes → page texts (content check, encryption, broken pages)
│   ├── cleaner.py             #   drops header/footer lines repeated on most pages
│   ├── chunker.py             #   page texts → overlapping, word-aligned Chunks
│   └── exceptions.py          #   PDFParseError, EmptyDocumentError, DuplicateDocumentError
│
├── search/                    # Bounded context: getting answers OUT
│   ├── router.py              #   POST /query
│   ├── schemas.py             #   QueryRequest, QueryResponse, SearchResult
│   └── service.py             #   SearchService: embed query → search → min_score filter
│
├── health/router.py           # GET /health (also reports active backend + model)
│
├── shared/                    # Shared kernel: no framework imports at all
│   ├── models.py              #   Chunk, SearchHit (plain dataclasses)
│   └── ports.py               #   Embedder, VectorStore (Protocols = interfaces)
│
├── infrastructure/            # Adapters: the ONLY place third-party AI/DB libraries are imported
│   ├── fastembed_embedder.py  #   Embedder → FastEmbed
│   ├── qdrant_store.py        #   VectorStore → Qdrant
│   └── chroma_store.py        #   VectorStore → ChromaDB
│
└── core/
    ├── config.py              # Settings (pydantic-settings, validated at startup)
    ├── container.py           # Composition root: picks concrete adapters, builds services
    ├── dependencies.py        # FastAPI Depends(): hands services to routers
    └── error_handlers.py      # Domain error → HTTP status, in one place
```

### Dependency direction

```
            ┌──────────── HTTP edge (FastAPI + Pydantic) ────────────┐
 request ─► │ router ─► schemas                                       │
            └───┬─────────────────────────────────────────────────────┘
                ▼
          ┌─ services (plain Python) ─┐        ┌── infrastructure ───────────┐
          │ IngestionService           │──────►│ ports.Embedder    ◄── FastEmbedEmbedder
          │ SearchService              │       │ ports.VectorStore ◄── QdrantVectorStore
          └────────────────────────────┘       │                   ◄── ChromaVectorStore
                                               └─────────────────────────────┘
          core/container.py is the only module that knows which adapter is plugged in.
```

You can check this with `grep`. Services, domain models and ports import **no** FastAPI, Pydantic, Qdrant, Chroma or FastEmbed:

```bash
grep -lE "^(from|import) (fastapi|pydantic|chromadb|qdrant_client|fastembed)" app/*/service.py app/*/models.py app/shared/*.py
```

That command should print nothing.

### Why this design

| Principle | Where it shows |
|---|---|
| **Single responsibility** | Each file does one job. The parser only parses, the chunker only chunks, routers only translate HTTP, and services only orchestrate. |
| **Dependency inversion** | Services depend on the `Embedder` and `VectorStore` Protocols in `shared/ports.py`, never on a library. |
| **Open/closed** | A new vector DB means one new adapter file plus one branch in `container.py`. No service, router or schema changes. |
| **Liskov / contract tests** | `tests/test_vector_stores.py` runs the **same** tests against Qdrant and Chroma. Any new adapter must pass them unchanged. |
| **Low coupling** | Frameworks stay at the edges. FastAPI lives in `router.py`, `dependencies.py`, `error_handlers.py` and `main.py`. Pydantic lives in `schemas.py` and `config.py`. The domain uses plain dataclasses. |
| **High cohesion** | Everything about uploading is in `documents/`, and everything about searching is in `search/`. |
| **Explicit wiring** | `core/container.py` is the single composition root, built once at startup in FastAPI's `lifespan`. |
| **Fail fast** | Invalid settings (for example `chunk_overlap >= chunk_size`, or an unknown `VECTOR_DB`) stop the app at startup with a clear error. |
| **Kept deliberately small** | No repository layer, unit-of-work, event bus or DI framework. Those would be ceremony at this size. |

### What swapping things actually costs

| Change | Files touched |
|---|---|
| Qdrant ⇄ Chroma | **None.** Set `VECTOR_DB=chroma` in `.env`. |
| Add pgvector, Weaviate, Pinecone… | New `infrastructure/<db>_store.py` with 3 methods, plus one branch in `core/container.py` |
| Qdrant embedded → Qdrant server | `infrastructure/qdrant_store.py`: `QdrantClient(path=…)` → `QdrantClient(url=…)` |
| Another embedding model (FastEmbed) | **None.** Set `EMBEDDING_MODEL`. A fresh collection is created automatically, because vectors from different models can't be compared. |
| OpenAI or another embedding provider | New `infrastructure/<provider>_embedder.py` with 2 methods + `dimension`, plus one line in `container.py` |
| Pydantic → attrs or msgspec for the API | Only `*/schemas.py`. Services return dataclasses, not Pydantic models. |
| FastAPI → Flask, Litestar or a CLI | Only `router.py` files, `dependencies.py`, `error_handlers.py` and `main.py`. Services are plain Python classes. |
| Add LLM answer generation | New `answers/` module that uses `SearchService` and an `LLM` port, plus one `include_router` line |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness, plus the active `vector_db` and `embedding_model` |
| POST | `/documents` | Upload a PDF (multipart field `file`) |
| POST | `/query` | Semantic search over uploaded PDFs |

### Upload

```bash
curl -F "file=@sample.pdf" http://127.0.0.1:8000/documents
```

```json
{ "document_id": "f23d3db4f1aaf08b55b8e0797c84d74c", "filename": "sample.pdf", "pages": 212, "chunks": 913 }
```

| Status | When |
|---|---|
| `201` | Indexed |
| `400` | Not a PDF by content (the extension is ignored), corrupt, or password-protected |
| `409` | The same file content was already uploaded, under any filename |
| `413` | Larger than `MAX_UPLOAD_MB` |
| `422` | No extractable text (for example a scanned image, which would need OCR), or the `file` field is missing |

### Query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I fix a leaking faucet?", "top_k": 3, "min_score": 0.4}'
```

| Field | Required | Meaning |
|---|---|---|
| `query` | yes | 1–2000 characters, trimmed. Any language. |
| `top_k` | no | 1–50. Defaults to `DEFAULT_TOP_K`. |
| `document_id` | no | Search only within this document |
| `min_score` | no | Drop results below this cosine similarity (−1…1). Use it to avoid returning irrelevant chunks. |

```json
{
  "query": "How do I fix a leaking faucet?",
  "results": [
    { "text": "Fixing Ball-Type Faucets A ball faucet can leak…", "document_id": "f23d…", "filename": "Plumbing.pdf", "page": 100, "score": 0.70 }
  ]
}
```

## RAG design decisions and edge cases

| Concern | Handling |
|---|---|
| **Model input limit** | The model reads at most **128 tokens**, and longer text is silently truncated. 1000-character chunks lost about half their text this way. |
| **Chunk size (measured, not guessed)** | A small retrieval eval with 9 questions, known answer pages, English and German, and both PDFs compared three settings. **400/80**: hit@3 6/9 (too little context). **600/120**: hit@3 **8/9**. **1000/200**: hit@3 7/9 (truncated). 600/120 is the default. |
| **Chunk boundaries** | Chunks are word-aligned, so no word is ever cut in half. Consecutive chunks overlap, so a sentence crossing a boundary is still found. |
| **Page attribution** | Each page is chunked separately, so every result has an exact page number. |
| **Repeated headers/footers** | Lines found on ≥50% of pages are removed, ignoring digits so "page 6" matches "page 7". Otherwise company names and dates appear in every chunk and make them all look alike. |
| **Duplicates** | `document_id` is the SHA-256 of the file content. A re-upload returns `409` before any parsing or embedding. Chunk IDs are deterministic and stores upsert, so even two simultaneous uploads can't create duplicates. |
| **Model changes** | The collection name includes the model name, so changing `EMBEDDING_MODEL` never mixes incompatible vectors. |
| **Irrelevant queries** | The search always returns the nearest chunks, so use `min_score` to drop weak matches. Scores below ~0.35 usually mean nothing relevant was found. |
| **Encrypted PDFs** | Permission-only encryption with an empty password is supported. PDFs that need a real password get `400`. |
| **Broken pages** | A page that fails to parse is skipped with a warning instead of failing the whole document. |
| **Large uploads** | At most `MAX_UPLOAD_MB` + 1 bytes are read, so oversized files are never fully loaded into memory. |
| **Long ingestion** | Parsing and embedding run in a thread pool, so the server keeps answering other requests during a long upload. |
| **Untrusted filenames** | Paths are stripped (`C:\Users\x\a.pdf` → `a.pdf`). The filename is stored only as metadata. |

## Configuration

Set these as environment variables or in `.env`. They're validated at startup.

| Variable | Default | Meaning |
|---|---|---|
| `VECTOR_DB` | `qdrant` | `qdrant` or `chroma` |
| `QDRANT_PATH` / `CHROMA_PATH` | `./data/qdrant` / `./data/chroma` | Where vectors are persisted |
| `COLLECTION_NAME` | `pdf_chunks` | Prefix; the model name is appended automatically |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Any [FastEmbed text model](https://qdrant.github.io/fastembed/examples/Supported_Models/) |
| `MODEL_CACHE_DIR` | `./data/models` | Model download location |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `120` | In characters. Overlap must be smaller than size. |
| `DEFAULT_TOP_K` | `5` | 1–50 |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |

## Tests

```bash
pytest
```

The suite has 88 tests and runs in about 15 seconds. API and store tests run against **both** Qdrant and Chroma.

| File | What it covers |
|---|---|
| `test_chunker.py`, `test_cleaner.py`, `test_pdf_parser.py` | Text processing: word alignment, overlap, oversized words, header removal, non-PDF, corrupt, password-protected |
| `test_validation.py` | Pydantic rules for settings and requests: empty or whitespace query, `top_k` bounds, invalid backend, overlap ≥ size |
| `test_services.py` | Use cases with **fake** `Embedder` and `VectorStore`, so no DB and no model: dedup, header stripping, `min_score`, filters |
| `test_vector_stores.py` | **Contract tests** run on every adapter: ranking, scores, filters, idempotent upsert, empty store, persistence |
| `test_api.py` | End-to-end HTTP with the real model: every status code, cross-lingual search, `document_id` filter, `min_score`, path stripping |

## Limitations and next steps

- **No answer generation yet.** The service retrieves relevant chunks but doesn't generate an answer. That would be an `answers/` module calling an LLM with the top chunks.
- **No OCR.** Scanned PDFs return `422`.
- **Synchronous ingestion.** A 200-page PDF takes about 1.5 minutes on CPU. For production, ingestion should go to a background job queue, with a status endpoint.
- **No document management.** There are no list or delete endpoints yet, so a `409` can't be resolved by deleting and re-uploading.
- **Single process.** Embedded Qdrant allows one process per data folder. Use a Qdrant server for multiple workers.
