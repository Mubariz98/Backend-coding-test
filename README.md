# PDF RAG API

Upload PDF documents, search them in natural language, and get **answers with page citations** from a local LLM. It works across languages: an English question finds German passages and the other way round.

Everything runs locally and for free. There's no API key and no cloud service.

| Endpoint | Purpose |
|---|---|
| `POST /documents` | Upload and index a PDF |
| `GET /documents`, `GET`/`DELETE /documents/{id}` | List, inspect and delete documents |
| `POST /query` | Semantic search: the most relevant passages, with page and score |
| `POST /ask`, `POST /ask/stream` | Answer generation (the "G" in RAG), grounded in the passages, with `[n]` citations |
| `GET /health` | Status, including the active vector DB, models and whether the LLM is reachable |
| `GET /` | A minimal web UI for all of the above |

## Quick start

### Option A: Docker (one command, recommended for the demo)

```bash
docker compose up --build
```

Open **http://localhost:8000**. This starts the API + UI, a Qdrant server, and Ollama with `qwen2.5:1.5b`. The first run downloads about 1 GB for the LLM. Qdrant's dashboard is at http://localhost:6333/dashboard.

### Option B: Local Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
```

For answers, also install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:1.5b`. Without Ollama, upload and search work fully, and `/ask` returns `503`. Qdrant runs embedded in `data/`, with no server needed.

On Debian/Ubuntu, `python3 -m venv` needs the `python3-venv` package. [uv](https://docs.astral.sh/uv/) works too.

## How it works

```
Upload:  PDF ─► SHA-256 (reject duplicates) ─► text per page ─► strip repeated headers/footers
             ─► overlapping, word-aligned chunks ─► embed ─► Qdrant   (+ metadata row in SQLite)

Search:  question ─► embed (same model) ─► nearest chunks ─► min_score filter ─► passages + page + score

Ask:     question ─► search (top 4, score ≥ 0.4) ─┬─ nothing relevant ─► "not found" (LLM not called)
                                                   └─ prompt with numbered passages ─► local LLM ─► answer with [n] citations
```

| Component | Choice | Why |
|---|---|---|
| Vector DB | **Qdrant** (default) or **ChromaDB**, chosen with `VECTOR_DB` | Qdrant runs embedded (a folder) or as a server (Docker) with the same adapter. Chroma proves the interface is swappable. |
| Embeddings | [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) via [FastEmbed](https://github.com/qdrant/fastembed) | 50+ languages, including German. CPU-only through ONNX, ~220 MB, no PyTorch. |
| LLM | **Qwen2.5 1.5B** via [Ollama](https://ollama.com) | Free and local, ~1 GB RAM, good German. Fits an 8 GB laptop without a GPU. |
| Metadata | SQLite (Python standard library) | One row per document, for listing, duplicate checks and deletion |
| PDF parsing | `pypdf[crypto]` | The `[crypto]` extra handles AES-encrypted PDFs |
| API | FastAPI + Pydantic v2 | Typed contracts and OpenAPI docs at `/docs` |

## Architecture

The code is organised **by domain** (DDD-style modules) using **Ports & Adapters**. Business logic depends on small interfaces, and concrete libraries plug in at one place.

```
app/
├── main.py                     # create_app(): lifespan wiring, error handlers, routers. No logic.
│
├── documents/                  # Getting PDFs IN, and managing them
│   ├── router.py               #   POST/GET/DELETE /documents (HTTP only)
│   ├── schemas.py              #   DocumentResponse (Pydantic)
│   ├── service.py              #   DocumentService: ingest, list, get, delete
│   ├── models.py               #   Document (dataclass)
│   ├── ports.py                #   DocumentRepository (interface)
│   ├── pdf_parser.py           #   bytes → page texts (content check, encryption, broken pages)
│   ├── cleaner.py              #   removes header/footer lines repeated on most pages
│   ├── chunker.py              #   page texts → overlapping, word-aligned Chunks
│   └── exceptions.py
│
├── search/                     # Getting passages OUT
│   ├── router.py, schemas.py   #   POST /query
│   └── service.py              #   SearchService: embed query → search → min_score filter
│
├── answers/                    # Generating answers (builds on search)
│   ├── router.py, schemas.py   #   POST /ask, POST /ask/stream (NDJSON)
│   ├── service.py              #   AnswerService: retrieve → skip LLM if nothing relevant → generate
│   ├── prompt.py               #   grounded prompt with numbered passages
│   ├── ports.py                #   LLM (interface)
│   └── models.py, exceptions.py
│
├── frontend/                   # GET /: one static HTML page, a plain API client
├── health/                     # GET /health
│
├── shared/                     # Shared kernel: no framework imports
│   ├── models.py               #   Chunk, SearchHit (dataclasses)
│   └── ports.py                #   Embedder, VectorStore (interfaces)
│
├── infrastructure/             # Adapters: the ONLY place DB/AI libraries are imported
│   ├── qdrant_store.py         #   VectorStore → Qdrant (embedded or server)
│   ├── chroma_store.py         #   VectorStore → ChromaDB
│   ├── fastembed_embedder.py   #   Embedder → FastEmbed
│   ├── ollama_llm.py           #   LLM → Ollama HTTP API
│   └── sqlite_document_repository.py   # DocumentRepository → SQLite
│
└── core/
    ├── config.py               # Settings, validated at startup
    ├── container.py            # Composition root: picks adapters, builds services
    ├── dependencies.py         # FastAPI Depends() aliases
    └── error_handlers.py       # Domain error → HTTP status, in one place
```

### Dependency direction

```
 HTTP ─► router ─► service ─┬─► Embedder ◄──────────── FastEmbedEmbedder
   (FastAPI +      (plain   ├─► VectorStore ◄───────── QdrantVectorStore / ChromaVectorStore
    Pydantic)      Python)  ├─► DocumentRepository ◄── SqliteDocumentRepository
                            └─► LLM ◄───────────────── OllamaLLM

         Only core/container.py knows which adapter is plugged in.
```

You can check this with `grep`: services, domain models and ports import **no** FastAPI, Pydantic or DB/AI library.

```bash
grep -lE "^(from|import) (fastapi|pydantic|chromadb|qdrant_client|fastembed|httpx)" \
  app/*/service.py app/*/models.py app/*/ports.py app/shared/*.py
```

That command prints nothing.

### Design principles, and where they show

| Principle | Where |
|---|---|
| **Single responsibility** | One job per file: the parser parses, the chunker chunks, routers translate HTTP, services orchestrate, adapters talk to libraries. |
| **Dependency inversion** | Services depend on 4 Protocols (`Embedder`, `VectorStore`, `DocumentRepository`, `LLM`), never on a library. |
| **Open/closed** | A new DB or LLM means one new adapter file plus one line in `container.py`. Services, routers and schemas stay unchanged. |
| **Liskov / contract tests** | `tests/test_vector_stores.py` runs **identical** tests on Qdrant and Chroma. A new adapter must pass them unchanged. |
| **Low coupling, high cohesion** | Frameworks stay at the edges, and each domain folder holds everything about its feature. |
| **Fail fast** | Invalid settings stop startup. An unreachable LLM fails *before* streaming starts, with `503`. |
| **Kept small on purpose** | No DI framework, event bus or unit-of-work. That would be ceremony at this size. |

### What swapping costs

| Change | Files touched |
|---|---|
| Qdrant ⇄ Chroma | **None.** `VECTOR_DB=chroma` |
| Embedded Qdrant → Qdrant server | **None.** `QDRANT_URL=http://…`, as `docker-compose.yml` does |
| Bigger local LLM | **None.** `LLM_MODEL=qwen2.5:7b` |
| Hosted LLM (OpenAI, Claude, …) | New `infrastructure/<provider>_llm.py` (3 methods) + 1 line in `container.py` |
| pgvector, Weaviate, Pinecone… | New `infrastructure/<db>_store.py` (3 methods) + 1 branch in `container.py` |
| Other embedding model | **None** for FastEmbed models (`EMBEDDING_MODEL`); a fresh collection is created automatically |
| SQLite → Postgres for metadata | New `infrastructure/postgres_document_repository.py` (4 methods) |
| Pydantic → another validation library | Only `*/schemas.py` and `core/config.py`. Services use dataclasses. |
| FastAPI → another framework or a CLI | Only routers, `dependencies.py`, `error_handlers.py` and `main.py` |

## API details

### Upload: `POST /documents`

```bash
curl -F "file=@samples/Gewährleistungshandbuch.pdf" http://localhost:8000/documents
```

```json
{ "document_id": "c9caf8bcdb9c29420f7385e198938181", "filename": "Gewährleistungshandbuch.pdf",
  "pages": 155, "chunks": 504, "size_bytes": 5577348, "uploaded_at": "2026-10-20T21:33:47Z" }
```

| Status | When |
|---|---|
| `201` | Indexed |
| `400` | Not a PDF by content (the extension is ignored), corrupt, or password-protected |
| `409` | The same file content is already uploaded, under any filename |
| `413` | Larger than `MAX_UPLOAD_MB` |
| `422` | No extractable text (scanned images need OCR), or the `file` field is missing |

`GET /documents` lists documents newest first. `DELETE /documents/{id}` removes the chunks and the metadata (`204`, or `404` if unknown), after which the same file can be uploaded again.

### Search: `POST /query`

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "What warranty does Volkswagen offer on new cars?", "top_k": 3}'
```

| Field | Required | Meaning |
|---|---|---|
| `query` | yes | 1–2000 characters, trimmed. Any language. |
| `top_k` | no | 1–50. Defaults to `DEFAULT_TOP_K`. |
| `document_id` | no | Search one document only |
| `min_score` | no | Drop results below this cosine similarity (−1…1) |

### Answer: `POST /ask` and `POST /ask/stream`

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Welche Garantie bietet Volkswagen auf Neuwagen?"}'
```

```json
{ "answer": "Volkswagen gewährt 2 Jahre Fehlerfreiheit … [1]", "sources": [ { "filename": "…", "page": 6, "score": 0.74, "text": "…" } ] }
```

- **The answer** uses the question's language and cites passages as `[n]`, which means `sources[n-1]`.
- **No relevant passages** (none with score ≥ `ANSWER_MIN_SCORE`): it answers "not found" **without calling the LLM**. That's faster and leaves nothing to hallucinate from.
- **`/ask/stream`** sends newline-delimited JSON: `{"type":"sources"}`, then `{"type":"token"}`… and finally `{"type":"done"}`. On a laptop CPU an answer takes ~30 s, so streaming shows words as they're generated. The UI uses this endpoint.
- **LLM not reachable:** `503`. Search keeps working.

## Retrieval evaluation

**What it is:** an exam for the search step. A good answer needs the right passage, so retrieval quality decides answer quality.

1. [`eval/questions.json`](eval/questions.json) holds **17 questions** about the sample PDF, in German and English, with the **pages where the answer is** taken from the document's table of contents.
2. [`scripts/evaluate.py`](scripts/evaluate.py) indexes the PDF with the app's real pipeline, asks every question, and checks where the correct page shows up in the results:
   - **hit@3**: how often a correct page is in the top 3, "found it at all".
   - **MRR** (mean reciprocal rank): how *high* it ranks. 1st place scores 1, 2nd scores 0.5, 3rd scores 0.33, and a miss scores 0. The average is between 0 and 1.
3. Running it with different settings shows which setting is actually better, instead of guessing.

```bash
python scripts/evaluate.py                    # compares chunk sizes; ~1 min each
python scripts/evaluate.py --configs 800:160 --verbose   # per-question ✓/✗ with the pages found
```

Results on the sample PDF (`Gewährleistungshandbuch.pdf`, 155 pages):

| Chunk size / overlap (characters) | Chunks | hit@3 | MRR@3 |
|---|---|---|---|
| 400 / 80 | 962 | 65% | 0.46 |
| 600 / 120 | 649 | 65% | 0.58 |
| **800 / 160 (default)** | 504 | **71%** | **0.62** |
| 1000 / 200 | 412 | 71% | 0.63 |

**Decision:** 800/160 matches 1000/200's quality, but its chunks are 20% shorter. Each answer therefore sends a shorter prompt to the LLM, which matters on a laptop CPU. Small chunks (400) lose too much context.

The questions it still misses, such as "Wie ist die Fahrzeug-Identifizierungsnummer aufgebaut?", are where the next improvements would be measured: hybrid keyword + vector search, a reranker, or a larger embedding model.

## RAG edge cases handled

| Concern | Handling |
|---|---|
| Repeated headers/footers | Lines on ≥50% of pages are removed, ignoring digits so "page 6" matches "page 7". Otherwise "Volkswagen AG … intern" appears in every chunk and makes them all look alike. |
| Chunk boundaries | Word-aligned, so no word is cut in half. Overlap keeps sentences that cross a boundary findable. |
| Page attribution | Each page is chunked separately, so every passage has an exact page. |
| Duplicates | The content hash is the ID, so a re-upload returns `409` before any work. Deterministic chunk IDs and upserts mean even concurrent uploads can't duplicate. |
| Embedding model change | The collection name includes the model, so incompatible vectors are never mixed. |
| Irrelevant questions | `min_score` for search. `/ask` refuses to answer from weak passages. |
| Hallucination | The prompt says to use only the passages, cite `[n]`, and say when the answer isn't there. Temperature is 0.1. |
| Encrypted / broken PDFs | Permission-only encryption works. A real password gets `400`. Unreadable pages are skipped with a warning. |
| Large / slow uploads | Read up to the limit only. Parsing and embedding run in a thread pool, so the server stays responsive. |
| Untrusted filenames | Paths are stripped (`C:\x\a.pdf` → `a.pdf`). Stored as metadata only. The UI renders PDF text with `textContent`, never as HTML. |

## Configuration

Set these as environment variables or in `.env` (see [`.env.example`](.env.example)). They're validated at startup.

| Variable | Default | Meaning |
|---|---|---|
| `VECTOR_DB` | `qdrant` | `qdrant` or `chroma` |
| `QDRANT_URL` | *(unset)* | Qdrant server URL. If unset, embedded at `QDRANT_PATH`. |
| `QDRANT_PATH` / `CHROMA_PATH` | `./data/qdrant` / `./data/chroma` | Embedded storage folders |
| `SQLITE_PATH` | `./data/documents.db` | Document metadata |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Any [FastEmbed text model](https://qdrant.github.io/fastembed/examples/Supported_Models/) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `160` | Characters; chosen with the evaluation above |
| `LLM_BASE_URL` / `LLM_MODEL` | `http://localhost:11434` / `qwen2.5:1.5b` | Ollama server and model |
| `ANSWER_TOP_K` / `ANSWER_MIN_SCORE` | `4` / `0.4` | Passages given to the LLM, and the relevance threshold |
| `DEFAULT_TOP_K` | `5` | Search results when `top_k` is omitted |
| `MAX_UPLOAD_MB` | `20` | Upload limit (`100` in Docker) |

## Tests

```bash
pytest
```

There are **125 tests in ~20 s**. They need no Ollama, because the LLM is faked or mocked. API and store tests run on **both** Qdrant and Chroma.

| File | Covers |
|---|---|
| `test_chunker.py`, `test_cleaner.py`, `test_pdf_parser.py` | Text processing and every bad-input case (non-PDF, corrupt, password, empty) |
| `test_validation.py` | Settings and request validation rules |
| `test_services.py` | All use cases with in-memory fakes: dedup, delete, `min_score`, skip-LLM-when-irrelevant, streaming |
| `test_vector_stores.py` | **Contract tests** on every vector DB adapter |
| `test_document_repository.py` | SQLite adapter: round trip, ordering, persistence |
| `test_ollama_llm.py` | Ollama adapter against a mocked HTTP layer: requests, streaming, missing model, server down |
| `test_api.py` | End-to-end HTTP with the real embedding model: every status code, document management, `/ask`, `/ask/stream`, UI |

## Limitations and next steps

- **Retrieval quality:** hit@3 is about 71% with a small CPU embedding model. Next would be hybrid BM25 + vector search and a reranker, both measurable with `scripts/evaluate.py`.
- **LLM size:** 1.5B parameters suits the demo hardware. Larger local models or a hosted API give better answers, and switching is a config value or one adapter.
- **Synchronous ingestion:** a 155-page PDF takes ~1 minute. Production would use a job queue and a status endpoint.
- **No OCR** for scanned PDFs. **No auth** or rate limiting.
- **Embedded Qdrant** allows one process per folder. Use the Qdrant server (as in Docker) for several workers.

## Sample PDF

`samples/` holds the sample document provided with the task. It's marked internal, so it's **not committed** by default (see `.gitignore`). Place it in `samples/` to run the evaluation.
