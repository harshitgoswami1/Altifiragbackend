"""Runtime settings for the standalone backend."""

from dataclasses import dataclass
import os
import math
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
    min_relevance_score: float | None = None
    router_model: str | None = None
    router_timeout_seconds: float = 15.0
    router_enabled: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("OLLAMA_BASE_URL must be an http(s) URL, for example http://10.0.0.8:11434")
        raw_min_score = os.environ.get("RAG_MIN_RELEVANCE_SCORE", "").strip()
        try:
            min_score = float(raw_min_score) if raw_min_score else None
        except ValueError as error:
            raise RuntimeError("RAG_MIN_RELEVANCE_SCORE must be a finite number") from error
        if min_score is not None and not math.isfinite(min_score):
            raise RuntimeError("RAG_MIN_RELEVANCE_SCORE must be a finite number")
        try:
            router_timeout = float(os.environ.get("RAG_ROUTER_TIMEOUT_SECONDS", "15"))
        except ValueError as error:
            raise RuntimeError("RAG_ROUTER_TIMEOUT_SECONDS must be positive and finite") from error
        if not math.isfinite(router_timeout) or router_timeout <= 0:
            raise RuntimeError("RAG_ROUTER_TIMEOUT_SECONDS must be positive and finite")
        router_enabled = os.environ.get("RAG_ROUTER_ENABLED", "false").casefold()
        if router_enabled not in {"true", "false"}:
            raise RuntimeError("RAG_ROUTER_ENABLED must be true or false")
        root = Path(__file__).resolve().parents[1]
        return cls(
            source_dir=Path(os.environ.get("RAG_SOURCE_DIR", root / "data" / "source")),
            vectorstore_dir=Path(os.environ.get("RAG_VECTORSTORE_DIR", root / "data" / "vectorstore")),
            ollama_base_url=base_url,
            embedding_model=os.environ.get("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            chat_model=os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL),
            min_relevance_score=min_score,
            router_model=os.environ.get("RAG_ROUTER_MODEL") or None,
            router_timeout_seconds=router_timeout,
            router_enabled=router_enabled == "true",
        )
