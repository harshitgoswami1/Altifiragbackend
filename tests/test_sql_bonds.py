"""Database boundary checks without a PostgreSQL service."""

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api import create_app
from app.bonds import BondFilter, BondSort
from app.config import Settings
from app.retrieval import QueryRoute, route_question
from app.sql_bonds import BondRepository, observation, search_sql


def sample_row(isin="INE157D14EA9", active=False):
    return {
        "is_active": active, "bond_status": "Matured" if not active else "Active",
        "issuer_name": "Clix Capital Services Private Limited", "isin": isin,
        "final_url": None, "source_url": f"https://example.test/bonds/{isin}",
        "maturity_date": date(2022, 7, 28), "issue_date": date(2022, 2, 2),
        "fetched_at": datetime(2026, 10, 1, 7, 20, 23, tzinfo=timezone.utc),
        "face_value_inr": Decimal("500000.00"),
        "minimum_investment_inr": Decimal("499772.50"),
        "coupon_percent": Decimal("9.1000"), "ytm_percent": Decimal("8.6800"),
    }


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []
        self.last_query = ""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=None):
        self.last_query = query
        self.calls.append((query, params))
        return self

    def fetchone(self):
        return {"count": len(self.rows)}

    def fetchall(self):
        if "issuer_name AS title" in self.last_query:
            return [{"isin": row["isin"], "title": row["issuer_name"], "url": row["source_url"]}
                    for row in self.rows]
        return self.rows


class SQLBondTests(unittest.TestCase):
    def test_sql_filters_bind_values_and_keep_stable_order(self):
        route = QueryRoute(
            intent="discovery",
            filters=[BondFilter(field="ytm", op="gte", value="8.68", source="YTM at least 8.68%"),
                     BondFilter(field="issue_date", op="lt", value="2023-01-01", source="issued before 2023")],
            sorting=BondSort(field="face_value", descending=True, source="highest face value"),
        )
        where, values, order = search_sql(route)
        self.assertIn("is_active IS TRUE", where)
        self.assertIn("ytm_percent >= %s", where)
        self.assertIn("issue_date < %s", where)
        self.assertIn("face_value_inr IS NOT NULL", where)
        self.assertEqual(values, [Decimal("8.68"), date(2023, 1, 1)])
        self.assertEqual(order, "face_value_inr DESC, isin ASC")

    def test_explicit_status_search_includes_inactive_rows_without_sql_interpolation(self):
        malicious = "Matured' OR 1=1 --"
        route = QueryRoute(intent="discovery", filters=[BondFilter(
            field="status", op="eq", value=malicious, source="status request")])
        where, values, _ = search_sql(route)
        self.assertNotIn("is_active IS TRUE", where)
        self.assertIn("LOWER(bond_status) = %s", where)
        self.assertNotIn("OR 1=1", where)
        self.assertEqual(values, [malicious.casefold()])
        self.assertEqual(route_question("Show matured bonds", []).filters[0].value, "matured")

    def test_search_uses_count_and_limited_rows_with_bound_parameters(self):
        connection = FakeConnection([sample_row()])
        repository = BondRepository("postgres://example.invalid/db")
        repository._connect = lambda: connection
        route = QueryRoute(intent="discovery", filters=[BondFilter(
            field="status", op="eq", value="Matured", source="status Matured")])
        result = repository.search(route)
        self.assertEqual(result.count, 1)
        self.assertEqual(result.records[0].metadata["isin"], "INE157D14EA9")
        self.assertEqual(connection.calls[0][1], ["matured"])
        self.assertEqual(connection.calls[1][1], ["matured", 4])

    def test_inactive_lookup_preserves_citation_identity_and_fetched_time(self):
        connection = FakeConnection([sample_row()])
        repository = BondRepository("postgres://example.invalid/db")
        repository._connect = lambda: connection
        found = repository.by_isins(["INE157D14EA9"])[0]
        self.assertIn("inactive", found.flags)
        self.assertEqual(found.metadata["chunk_id"], "bond:INE157D14EA9")
        self.assertEqual(found.metadata["url"], sample_row()["source_url"])
        self.assertEqual(found.metadata["observed_at"], "2026-10-01T07:20:23+00:00")
        self.assertIn("9.1000% p.a.", found.text)
        self.assertEqual(connection.calls[0][1], (["INE157D14EA9"],))

    def test_new_field_phrases_route_to_structured_filters(self):
        for question, field in (
            ("Show bonds with face value above 5 lakh", "face_value"),
            ("Show bonds issued before 2024-01-01", "issue_date"),
            ("Show bonds with status Matured", "status"),
        ):
            with self.subTest(question=question):
                route = route_question(question, [])
                self.assertEqual(route.intent, "discovery")
                self.assertEqual(route.filters[0].field, field)

    def test_api_uses_database_rows_for_json_and_stream(self):
        connection = FakeConnection([sample_row()])
        repository = BondRepository("postgres://example.invalid/db")
        repository._connect = lambda: connection

        class Chat:
            def invoke(self, _):
                return AIMessage(content="The recorded coupon is 9.1000% p.a. [1]")

        settings = Settings(Path("/tmp/unused-source"), Path("/tmp/unused-index"), "http://localhost:11434")
        app = create_app(settings, bond_loader=lambda _: repository, bond_probe=lambda _: True,
                         chat_factory=lambda _: Chat(), index_ready=lambda _: False,
                         ollama_probe=lambda _: True)
        client = TestClient(app)
        question = {"question": "Show matured bonds"}
        answer = client.post("/v1/chat", json=question)
        stream = client.post("/v1/chat/stream", json=question)
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(stream.status_code, 200)
        self.assertEqual(answer.json()["citations"][0]["chunk_id"], "bond:INE157D14EA9")
        self.assertIn("inactive in source table", answer.json()["answer"])
        self.assertIn('"chunk_id": "bond:INE157D14EA9"', stream.text)
        self.assertIn("LOWER(bond_status) = %s", connection.calls[1][0])

        lookup = client.post("/v1/chat", json={"question": "What is the coupon of INE157D14EA9?"})
        self.assertEqual(lookup.status_code, 200)
        self.assertIn("inactive in the source table", lookup.json()["answer"])
        self.assertEqual(len(lookup.json()["citations"]), 1)


if __name__ == "__main__":
    unittest.main()
