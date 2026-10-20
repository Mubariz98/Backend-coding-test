"""Application settings, loaded from environment variables or a `.env` file."""

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Vector DB backend: switching is a config change, not a code change.
    vector_db: Literal["qdrant", "chroma"] = "qdrant"
    qdrant_url: str | None = None  # set to use a Qdrant server; otherwise embedded at qdrant_path
    qdrant_path: str = "./data/qdrant"
    chroma_path: str = "./data/chroma"
    collection_name: str = "pdf_chunks"

    # Document metadata (filename, pages, upload time)
    sqlite_path: str = "./data/documents.db"

    # Embedding model (multilingual: German, English and 50+ other languages)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_cache_dir: str = "./data/models"

    # Chunking, in characters. Chosen by scripts/evaluate.py on the sample PDF:
    # 800/160 matched 1000/200 (hit@3 71%) with smaller prompts for the LLM.
    chunk_size: int = Field(800, gt=0)
    chunk_overlap: int = Field(160, ge=0)

    default_top_k: int = Field(5, ge=1, le=50)

    # Answer generation (local LLM via Ollama). qwen2.5:1.5b: ~1 GB RAM, good German,
    # usable on a laptop CPU. Use qwen2.5:3b or larger on stronger hardware.
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:1.5b"
    llm_timeout_s: float = Field(300, gt=0)
    llm_context_tokens: int = Field(2048, ge=512)
    answer_top_k: int = Field(4, ge=1, le=10)  # passages given to the LLM; more = slower
    answer_min_score: float = Field(0.4, ge=-1.0, le=1.0)  # below this, answer "not found" without the LLM
    max_upload_mb: int = Field(20, gt=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
