import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.api import SYSTEM_PROMPT, create_app
from app.config import Settings
from app.corpus import CorpusError, chroma_metadata, load_corpus
from app.index import IndexError, active_manifest, build_index, ensure_current_index


def record(identifier="chunk_1"):
    return {
        "id": identifier,
        "document_id": "blog_example",
        "document_type": "blog",
        "title": "Example title",
        "category": "bonds",
        "url": "https://example.test/article",
        "isin": None,
        "observed_at": "2026-09-19T00:00:00Z",
        "quality_flags": ["stale_source"],
        "embedding_text": "Title: Example title\n\nUseful source text.",
    }


def write_corpus(source_dir: Path, rows: list[dict]) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (source_dir / "chunks.jsonl").write_bytes(payload)
    (source_dir / "chunk_report.json").write_text(json.dumps({
        "audit": "passed", "errors": [], "chunks": len(rows),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
    }))


class FakeStore:
    def similarity_search_with_relevance_scores(self, question, k):
        return [(
            Document(page_content="Useful source text.", metadata=chroma_metadata(record())),
            0.9,
        )]


class FakeChat:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="A grounded answer. [1]")


class TinyEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class FailingEmbeddings:
    def embed_documents(self, texts):
        raise RuntimeError("embedding host unavailable")


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.settings = Settings(root / "source", root / "vector", "http://127.0.0.1:11434")
        write_corpus(self.settings.source_dir, [record()])

    def tearDown(self):
        self.temporary.cleanup()

    def test_corpus_uses_embedding_text_and_scalar_metadata(self):
        corpus = load_corpus(self.settings)
        self.assertEqual(corpus.records[0]["embedding_text"], "Title: Example title\n\nUseful source text.")
        self.assertEqual(json.loads(chroma_metadata(corpus.records[0])["quality_flags"]), ["stale_source"])

    def test_corpus_rejects_checksum_mismatch(self):
        report_path = self.settings.source_dir / "chunk_report.json"
        report = json.loads(report_path.read_text())
        report["output_sha256"] = "wrong"
        report_path.write_text(json.dumps(report))
        with self.assertRaises(CorpusError):
            load_corpus(self.settings)

    def test_index_manifest_requires_current_source_and_files(self):
        self.settings.vectorstore_dir.mkdir()
        (self.settings.vectorstore_dir / "active_index.json").write_text(json.dumps({
            "fingerprint": "missing", "corpus_checksum": "anything",
            "embedding_model": self.settings.embedding_model, "chunk_count": 1,
        }))
        with self.assertRaises(Exception):
            active_manifest(self.settings)
        with self.assertRaises(Exception):
            ensure_current_index(self.settings)

    def test_index_activation_is_atomic_from_the_active_manifest_view(self):
        first = build_index(self.settings, TinyEmbeddings())
        self.assertEqual(ensure_current_index(self.settings)["fingerprint"], first["fingerprint"])

        write_corpus(self.settings.source_dir, [record("chunk_2")])
        with self.assertRaises(RuntimeError):
            build_index(self.settings, FailingEmbeddings())

        # The failed build never points active_index.json at an incomplete index.
        self.assertEqual(active_manifest(self.settings)["fingerprint"], first["fingerprint"])
        with self.assertRaises(IndexError):
            ensure_current_index(self.settings)

    def test_api_validates_input_and_returns_source_citations(self):
        fake_chat = FakeChat()
        app = create_app(
            self.settings,
            store_loader=lambda _: FakeStore(),
            chat_factory=lambda _: fake_chat,
            index_ready=lambda _: True,
            ollama_probe=lambda _: True,
        )
        client = TestClient(app)
        self.assertEqual(client.post("/v1/chat", json={"question": "   "}).status_code, 422)
        response = client.post("/v1/chat", json={"question": "What does this say?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"][0]["chunk_id"], "chunk_1")
        self.assertIn("Source [1]", str(fake_chat.messages[1].content))
        self.assertIn("untrusted reference data", str(fake_chat.messages[0].content))

    def test_api_returns_503_when_index_is_unavailable(self):
        app = create_app(self.settings, index_ready=lambda _: False, ollama_probe=lambda _: True)
        response = TestClient(app).post("/v1/chat", json={"question": "Hello"})
        self.assertEqual(response.status_code, 503)

    def test_prompt_excludes_advice_and_live_data(self):
        self.assertIn("personalized investment advice", SYSTEM_PROMPT)
        self.assertIn("live data", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
