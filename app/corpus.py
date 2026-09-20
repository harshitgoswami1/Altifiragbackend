"""Validation and conversion for the backend-owned corpus snapshot."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import Settings


class CorpusError(RuntimeError):
    """The copied corpus is missing, corrupt, or has not passed its producer audit."""


@dataclass(frozen=True)
class Corpus:
    records: list[dict[str, Any]]
    checksum: str
    report: dict[str, Any]


def _paths(settings: Settings) -> tuple[Path, Path]:
    return settings.source_dir / "chunks.jsonl", settings.source_dir / "chunk_report.json"


def load_corpus(settings: Settings) -> Corpus:
    chunks_path, report_path = _paths(settings)
    if not chunks_path.is_file() or not report_path.is_file():
        raise CorpusError("Copy chunks.jsonl and chunk_report.json into data/source before indexing")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CorpusError("chunk_report.json is not valid JSON") from error
    if report.get("audit") != "passed" or report.get("errors"):
        raise CorpusError("chunk_report.json does not describe an audit-passed corpus")

    payload = chunks_path.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    if report.get("output_sha256") != checksum:
        raise CorpusError("chunks.jsonl checksum does not match chunk_report.json")

    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CorpusError(f"chunks.jsonl line {line_number} is not valid JSON") from error
        if not isinstance(record.get("id"), str) or not record["id"]:
            raise CorpusError(f"chunks.jsonl line {line_number} has no chunk id")
        if not isinstance(record.get("embedding_text"), str) or not record["embedding_text"].strip():
            raise CorpusError(f"chunks.jsonl line {line_number} has no embedding_text")
        if record["id"] in ids:
            raise CorpusError(f"chunks.jsonl contains duplicate chunk id {record['id']}")
        ids.add(record["id"])
        records.append(record)
    if not records or len(records) != report.get("chunks"):
        raise CorpusError("chunks.jsonl record count does not match chunk_report.json")
    return Corpus(records=records, checksum=checksum, report=report)


def chroma_metadata(record: dict[str, Any]) -> dict[str, str]:
    """Chroma metadata is scalar-only, so quality flags are JSON encoded."""
    return {
        "chunk_id": record["id"],
        "document_id": str(record.get("document_id", "")),
        "document_type": str(record.get("document_type", "")),
        "title": str(record.get("title", "")),
        "category": str(record.get("category", "")),
        "url": str(record.get("url", "")),
        "isin": str(record.get("isin") or ""),
        "observed_at": str(record.get("observed_at", "")),
        "quality_flags": json.dumps(record.get("quality_flags", []), ensure_ascii=False),
    }
