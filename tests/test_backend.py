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
from app.retrieval import bond_catalog, deduplicate_matches, route_question


def record(
    identifier="chunk_1",
    *,
    document_id="blog_example",
    document_type="blog",
    title="Example title",
    category="bonds",
    url="https://example.test/article",
    isin=None,
    embedding_text="Useful source text.",
    quality_flags=None,
):
    return {
        "id": identifier,
        "document_id": document_id,
        "document_type": document_type,
        "title": title,
        "category": category,
        "url": url,
        "isin": isin,
        "observed_at": "2026-09-19T00:00:00Z",
        "quality_flags": ["stale_source"] if quality_flags is None else quality_flags,
        "embedding_text": f"Title: {title}\n\n{embedding_text}",
    }


def bond_record(isin="IN0020010081", title="10.18% Government Of India 11 Sep 2026"):
    return record(
        f"bond_{isin}",
        document_id=f"bond_{isin}",
        document_type="bond",
        title=title,
        category="government-securities",
        url=f"https://example.test/bonds/{isin}",
        isin=isin,
        quality_flags=[],
        embedding_text=(
            f"Bond: {title}\nISIN: {isin}\n"
            "Coupon rate: 10.18% p.a.\nObserved yield to maturity: 5.7% p.a."
        ),
    )


def write_corpus(source_dir: Path, rows: list[dict]) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (source_dir / "chunks.jsonl").write_bytes(payload)
    (source_dir / "chunk_report.json").write_text(json.dumps({
        "audit": "passed", "errors": [], "chunks": len(rows),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
    }))


class FakeStore:
    def __init__(self, rows=None):
        self.rows = rows or [record()]
        self.calls = []

    @staticmethod
    def _matches_filter(row, metadata_filter):
        if not metadata_filter:
            return True
        metadata = chroma_metadata(row)
        if "$and" in metadata_filter:
            return all(FakeStore._matches_filter(row, clause) for clause in metadata_filter["$and"])
        return all(
            metadata.get(key) == (value.get("$eq") if isinstance(value, dict) else value)
            for key, value in metadata_filter.items()
        )

    def get(self, where=None, include=None):
        return {
            "metadatas": [
                chroma_metadata(row)
                for row in self.rows
                if self._matches_filter(row, where)
            ]
        }

    def similarity_search_with_relevance_scores(self, question, k, filter=None):
        self.calls.append({"question": question, "k": k, "filter": filter})
        rows = [row for row in self.rows if self._matches_filter(row, filter)]
        return [
            (Document(page_content=row["embedding_text"], metadata=chroma_metadata(row)), 0.9 - index * 0.1)
            for index, row in enumerate(rows[:k])
        ]


class FakeChat:
    def __init__(self):
        self.messages = None
        self.invocations = 0

    def invoke(self, messages):
        self.messages = messages
        self.invocations += 1
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

    def routed_client(self, rows, settings=None):
        fake_chat = FakeChat()
        fake_store = FakeStore(rows)
        app = create_app(
            settings or self.settings,
            store_loader=lambda _: fake_store,
            chat_factory=lambda _: fake_chat,
            index_ready=lambda _: True,
            ollama_probe=lambda _: True,
        )
        return TestClient(app), fake_store, fake_chat

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

    def test_exact_isin_retrieves_only_the_matching_bond(self):
        bond = bond_record()
        blog = record("blog_chunk", title="What is yield to maturity?")
        client, store, _ = self.routed_client([blog, bond])

        response = client.post("/v1/chat", json={"question": "What is the YTM of IN0020010081?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["document_type"] for item in response.json()["citations"]], ["bond"])
        self.assertEqual(
            store.calls[0]["filter"],
            {
                "$and": [
                    {"document_type": {"$eq": "bond"}},
                    {"isin": {"$eq": "IN0020010081"}},
                ]
            },
        )

    def test_general_question_retrieves_only_blog_chunks(self):
        bond = bond_record()
        blog = record("blog_chunk", title="What is yield to maturity?")
        client, store, _ = self.routed_client([bond, blog])

        response = client.post("/v1/chat", json={"question": "What is yield to maturity?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["document_type"] for item in response.json()["citations"]], ["blog"])
        self.assertEqual(store.calls[0]["filter"], {"document_type": "blog"})

    def test_mixed_question_uses_bond_and_blog_lanes(self):
        bond = bond_record()
        blog = record("blog_chunk", title="What is yield to maturity?")
        client, store, _ = self.routed_client([blog, bond])

        response = client.post(
            "/v1/chat",
            json={
                "question": (
                    "What does 10.18% Government Of India 11 Sep 2026 mean, "
                    "and how does YTM work?"
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["document_type"] for item in response.json()["citations"]],
            ["bond", "blog"],
        )
        self.assertEqual(len(store.calls), 2)
        self.assertEqual(
            store.calls[0]["filter"],
            {
                "$and": [
                    {"document_type": {"$eq": "bond"}},
                    {"isin": {"$eq": "IN0020010081"}},
                ]
            },
        )
        self.assertEqual(store.calls[1]["filter"], {"document_type": "blog"})

    def test_ambiguous_bond_name_returns_candidates_without_calling_chat_model(self):
        first = bond_record("IN0020010081", "Tata Capital Limited 8.50% 2030")
        second = bond_record("IN0020010082", "Tata Capital Housing Finance 8.70% 2030")
        client, store, fake_chat = self.routed_client([first, second])

        response = client.post("/v1/chat", json={"question": "Tell me about the Tata Capital bond"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("multiple bond records", response.json()["answer"])
        self.assertEqual(len(response.json()["citations"]), 2)
        self.assertEqual(fake_chat.invocations, 0)
        self.assertEqual(store.calls[0]["filter"], {"document_type": "bond"})

    def test_unknown_isin_does_not_fall_back_to_blogs(self):
        bond = bond_record()
        blog = record("blog_chunk", title="What is yield to maturity?")
        client, store, fake_chat = self.routed_client([bond, blog])

        response = client.post("/v1/chat", json={"question": "What is the YTM of IN9999999999?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"], [])
        self.assertIn("does not contain a bond record", response.json()["answer"])
        self.assertEqual(store.calls, [])
        self.assertEqual(fake_chat.invocations, 0)

    def test_relevance_threshold_rejects_weak_blog_matches(self):
        settings = Settings(
            self.settings.source_dir,
            self.settings.vectorstore_dir,
            self.settings.ollama_base_url,
            min_relevance_score=0.95,
        )
        client, _, fake_chat = self.routed_client([record()], settings=settings)

        response = client.post("/v1/chat", json={"question": "What is yield to maturity?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"], [])
        self.assertEqual(fake_chat.invocations, 0)

    def test_route_question_resolves_exact_bond_title(self):
        candidate = bond_record()
        metadata = chroma_metadata(candidate)

        route = route_question(candidate["title"], bond_catalog([metadata]))

        self.assertEqual(route.route, "bond")
        self.assertEqual(route.resolved_bond.isin, candidate["isin"])

    def test_blog_results_are_deduplicated_by_document(self):
        first = Document(page_content="first", metadata={"document_id": "blog_1", "chunk_id": "chunk_1"})
        second = Document(page_content="second", metadata={"document_id": "blog_1", "chunk_id": "chunk_2"})
        third = Document(page_content="third", metadata={"document_id": "blog_2", "chunk_id": "chunk_3"})

        matches = deduplicate_matches([(first, 0.7), (second, 0.9), (third, 0.8)], 4)

        self.assertEqual([document.metadata["chunk_id"] for document, _ in matches], ["chunk_2", "chunk_3"])

    def test_prompt_excludes_advice_and_live_data(self):
        self.assertIn("personalized investment advice", SYSTEM_PROMPT)
        self.assertIn("live data", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
