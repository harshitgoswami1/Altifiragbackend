"""Runtime settings for the standalone backend."""

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:8b-q8_0"
DEFAULT_CHAT_MODEL = "qwen3:8b"


@dataclass(frozen=True)
class Settings:
    source_dir: Path
    vectorstore_dir: Path
    ollama_base_url: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chat_model: str = DEFAULT_CHAT_MODEL

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("OLLAMA_BASE_URL must be an http(s) URL, for example http://10.0.0.8:11434")
        root = Path(__file__).resolve().parents[1]
        return cls(
            source_dir=Path(os.environ.get("RAG_SOURCE_DIR", root / "data" / "source")),
            vectorstore_dir=Path(os.environ.get("RAG_VECTORSTORE_DIR", root / "data" / "vectorstore")),
            ollama_base_url=base_url,
            embedding_model=os.environ.get("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            chat_model=os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        )
