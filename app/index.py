"""Explicit, safe construction of the persistent local Chroma index."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.config import Settings
from app.corpus import Corpus, CorpusError, chroma_metadata, load_corpus


COLLECTION_NAME = "rag_chunks"
ACTIVE_MANIFEST = "active_index.json"
BATCH_SIZE = 64


class IndexError(RuntimeError):
    """No usable, current vector index is available."""


def index_fingerprint(corpus: Corpus, settings: Settings) -> str:
    return hashlib.sha256(f"{corpus.checksum}\0{settings.embedding_model}".encode()).hexdigest()[:24]


def _manifest_path(settings: Settings) -> Path:
    return settings.vectorstore_dir / ACTIVE_MANIFEST


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def active_manifest(settings: Settings) -> dict[str, Any]:
    path = _manifest_path(settings)
    if not path.is_file():
        raise IndexError("No active index; run python -m app.index")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise IndexError("The active index manifest is invalid") from error
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise IndexError("The active index manifest has no fingerprint")
    location = settings.vectorstore_dir / fingerprint
    if not location.is_dir():
        raise IndexError("The active index files are missing")
    return manifest


def ensure_current_index(settings: Settings, corpus: Corpus | None = None) -> dict[str, Any]:
    corpus = corpus or load_corpus(settings)
    manifest = active_manifest(settings)
    if manifest.get("corpus_checksum") != corpus.checksum or manifest.get("embedding_model") != settings.embedding_model:
        raise IndexError("The active index is stale; run python -m app.index")
    if manifest.get("chunk_count") != len(corpus.records):
        raise IndexError("The active index count does not match the copied corpus")
    return manifest


def _store(settings: Settings, fingerprint: str, embeddings: Any) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(settings.vectorstore_dir / fingerprint),
        embedding_function=embeddings,
    )


def open_active_store(settings: Settings, embeddings: Any | None = None) -> Chroma:
    manifest = ensure_current_index(settings)
    embeddings = embeddings or OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
    return _store(settings, manifest["fingerprint"], embeddings)


def _stored_count(store: Chroma) -> int:
    return len(store.get().get("ids", []))


def _activate(settings: Settings, corpus: Corpus, fingerprint: str) -> dict[str, Any]:
    manifest = {
        "fingerprint": fingerprint,
        "corpus_checksum": corpus.checksum,
        "embedding_model": settings.embedding_model,
        "chunk_count": len(corpus.records),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomically(_manifest_path(settings), manifest)
    return manifest


def build_index(settings: Settings, embeddings: Any | None = None) -> dict[str, Any]:
    """Build a new index before switching the active manifest to it."""
    corpus = load_corpus(settings)
    fingerprint = index_fingerprint(corpus, settings)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    embeddings = embeddings or OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
    destination = settings.vectorstore_dir / fingerprint
    if destination.exists():
        existing = _store(settings, fingerprint, embeddings)
        if _stored_count(existing) != len(corpus.records):
            raise IndexError(f"Existing index {fingerprint} has an unexpected record count; remove it before rebuilding")
        return _activate(settings, corpus, fingerprint)

    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}-", dir=settings.vectorstore_dir))
    try:
        store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(temporary),
            embedding_function=embeddings,
        )
        for start in range(0, len(corpus.records), BATCH_SIZE):
            batch = corpus.records[start:start + BATCH_SIZE]
            store.add_documents(
                [Document(page_content=row["embedding_text"], metadata=chroma_metadata(row)) for row in batch],
                ids=[row["id"] for row in batch],
            )
        if _stored_count(store) != len(corpus.records):
            raise IndexError("New index record count does not match the copied corpus")
        os.replace(temporary, destination)
        return _activate(settings, corpus, fingerprint)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    argparse.ArgumentParser(description="Build the standalone RAG vector index").parse_args()
    settings = Settings.from_env()
    manifest = build_index(settings)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
