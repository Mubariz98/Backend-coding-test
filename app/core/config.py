"""Application settings, loaded from environment variables or a `.env` file."""

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Vector DB backend: switching is a config change, not a code change.
    vector_db: Literal["qdrant", "chroma"] = "qdrant"
    qdrant_path: str = "./data/qdrant"
    chroma_path: str = "./data/chroma"
    collection_name: str = "pdf_chunks"

    # Embedding model (multilingual: German, English and 50+ other languages)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_cache_dir: str = "./data/models"

    # Chunking, in characters. The model reads at most 128 tokens (~500-600 chars) and
    # truncates the rest. 600/120 scored best in a retrieval eval (vs 400 and 1000).
    chunk_size: int = Field(600, gt=0)
    chunk_overlap: int = Field(120, ge=0)

    default_top_k: int = Field(5, ge=1, le=50)
    max_upload_mb: int = Field(20, gt=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
