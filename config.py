"""
config.py - Mipangilio ya mfumo wote wa RAG Multilingual
Inasoma vigeu kutoka .env na kutoa mipangilio iliyopangwa vizuri.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Pakia .env
load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class OllamaConfig(BaseModel):
    base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_LLM_MODEL", "deepseek-coder:6.7b-instruct-q4_K_M"))
    embed_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "granite-embedding:multilingual"))
    timeout: int = 120


class ChromaConfig(BaseModel):
    persist_dir: str = Field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"))
    nyaraka_collection: str = Field(default_factory=lambda: os.getenv("CHROMA_NYARAKA_COLLECTION", "nyaraka"))
    memory_collection: str = Field(default_factory=lambda: os.getenv("CHROMA_MEMORY_COLLECTION", "memory"))
    cache_collection: str = Field(default_factory=lambda: os.getenv("CHROMA_CACHE_COLLECTION", "cache"))


class DocumentConfig(BaseModel):
    docs_dir: str = Field(default_factory=lambda: os.getenv("DOCS_DIR", "./data/documents"))
    uploads_dir: str = Field(default_factory=lambda: os.getenv("UPLOADS_DIR", "./data/uploads"))
    processed_dir: str = Field(default_factory=lambda: os.getenv("PROCESSED_DIR", "./data/processed"))
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64")))
    supported_extensions: list = [".pdf", ".txt", ".md", ".docx"]


class MemoryConfig(BaseModel):
    max_episodic_memory: int = Field(default_factory=lambda: int(os.getenv("MAX_EPISODIC_MEMORY", "100")))
    cache_similarity_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
    )
    semantic_top_k: int = Field(default_factory=lambda: int(os.getenv("SEMANTIC_TOP_K", "5")))
    episodic_top_k: int = Field(default_factory=lambda: int(os.getenv("EPISODIC_TOP_K", "3")))


class AppConfig(BaseModel):
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    language: str = Field(default_factory=lambda: os.getenv("LANGUAGE", "auto"))
    version: str = "1.0.0"
    app_name: str = Field(default_factory=lambda: os.getenv("APP_NAME", "Binoblax"))

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    documents: DocumentConfig = Field(default_factory=DocumentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    def ensure_dirs(self):
        """Hakikisha saraka zote zinawepo."""
        dirs = [
            self.documents.docs_dir,
            self.documents.uploads_dir,
            self.documents.processed_dir,
            self.chroma.persist_dir,
            f"{self.chroma.persist_dir}/nyaraka",
            f"{self.chroma.persist_dir}/memory",
            f"{self.chroma.persist_dir}/cache",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# Singleton ya mipangilio
config = AppConfig()
config.ensure_dirs()
