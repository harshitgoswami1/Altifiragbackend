"""Read-only, parameterized queries against the scraped bond table."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.bonds import BondObservation, LABELS, SQL_FIELDS, normalize_words, typed_value
from app.retrieval import QueryRoute


SELECT_COLUMNS = (
    "is_active, bond_status, issuer_name, isin, final_url, source_url, "
    "maturity_date, issue_date, fetched_at, face_value_inr, "
    "minimum_investment_inr, coupon_percent, ytm_percent"
)
COLUMNS = {
    "ytm": "ytm_percent", "coupon": "coupon_percent",
    "minimum_investment": "minimum_investment_inr", "face_value": "face_value_inr",
    "maturity": "maturity_date", "issue_date": "issue_date",
    "title": "issuer_name", "status": "bond_status",
}
OPERATORS = {"eq": "=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


@dataclass(frozen=True)
class SearchResult:
    records: list[BondObservation]
    count: int


def observation(row: dict[str, Any]) -> BondObservation:
    fetched = row.get("fetched_at")
    observed_at = fetched.isoformat() if isinstance(fetched, datetime) else str(fetched or "")
    isin = str(row["isin"])
    issuer = str(row["issuer_name"])
    status = str(row.get("bond_status") or "")
    active = row.get("is_active") is True
    metadata = {
        "chunk_id": f"bond:{isin}", "document_type": "bond", "title": f"{issuer} ({isin})",
        "url": str(row.get("final_url") or row["source_url"]), "isin": isin,
        "observed_at": observed_at, "quality_flags": "[]", "bond_status": status,
        "is_active": active,
    }
    values: dict[str, Decimal | date | str | None] = {
        "title": normalize_words(issuer),
        "status": status.casefold() if status else None,
    }
    for field, column in COLUMNS.items():
        if field in {"title", "status"}:
            continue
        values[field] = row.get(column)
    def source_value(field: str, column: str) -> str:
        value = row.get(column)
        if value is None:
            return "not recorded"
        if field in {"ytm", "coupon"}:
            return f"{value}% p.a."
        if field in {"minimum_investment", "face_value"}:
            return f"{value} INR"
        return str(value)

    text = "\n".join([
        f"Issuer: {issuer}", f"ISIN: {isin}", f"Bond status: {status or 'not recorded'}",
        f"Active in source table: {'yes' if active else 'no'}", f"Fetched at: {observed_at or 'not recorded'}",
        *(
            f"{LABELS[field]}: {source_value(field, column)}"
            for field, column in COLUMNS.items() if field not in {"title", "status"}
        ),
    ])
    return BondObservation(metadata, text, values, ("inactive",) if not active else ())


def search_sql(route: QueryRoute) -> tuple[str, list[Any], str]:
    """Return approved WHERE, bound values, and ORDER BY fragments."""
    conditions: list[str] = []
    values: list[Any] = []
    if not any(item.field == "status" for item in route.filters) and not route.include_matured:
        conditions.append("is_active IS TRUE")
    if route.isins:
        conditions.append("isin = ANY(%s)")
        values.append(route.isins)
    for item in route.filters:
        if item.field not in SQL_FIELDS:
            raise ValueError(f"{item.field} is not available in the bond database")
        column = COLUMNS[item.field]
        value = typed_value(item.field, item.value)
        if item.field == "title":
            if route.isins:
                # The catalog has already resolved this issuer phrase to exact ISINs.
                continue
            conditions.append("LOWER(issuer_name) LIKE %s ESCAPE '\\'")
            escaped = str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append(f"%{escaped}%")
        elif item.field == "status":
            conditions.append("LOWER(bond_status) = %s")
            values.append(value)
        else:
            conditions.append(f"{column} {OPERATORS[item.op]} %s")
            values.append(value)
    if route.sorting:
        if route.sorting.field not in COLUMNS or route.sorting.field in {"title", "status"}:
            raise ValueError("Unsupported bond ordering field")
        column = COLUMNS[route.sorting.field]
        conditions.append(f"{column} IS NOT NULL")
        ordering = f"{column} {'DESC' if route.sorting.descending else 'ASC'}, isin ASC"
    else:
        ordering = "isin ASC"
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    return where, values, ordering


class BondRepository:
    def __init__(self, database_url: str | None):
        if not database_url:
            raise RuntimeError("DATABASE_URL is required for bond queries")
        self.database_url = database_url

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row, connect_timeout=5, prepare_threshold=None)

    def reachable(self) -> bool:
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1 FROM scraped_bonds LIMIT 1")
            return True
        except psycopg.Error:
            return False

    def catalog(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return list(connection.execute(
                "SELECT isin, issuer_name AS title, source_url AS url FROM scraped_bonds ORDER BY isin"
            ).fetchall())

    def by_isins(self, isins: list[str]) -> list[BondObservation]:
        if not isins:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {SELECT_COLUMNS} FROM scraped_bonds WHERE isin = ANY(%s)", (isins,)
            ).fetchall()
        by_isin = {row["isin"]: observation(row) for row in rows}
        return [by_isin[isin] for isin in isins if isin in by_isin]

    def search(self, route: QueryRoute) -> SearchResult:
        where, values, ordering = search_sql(route)
        limit = min(route.result_limit, 20)
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) AS count FROM scraped_bonds" + where, values).fetchone()["count"]
            rows = connection.execute(
                f"SELECT {SELECT_COLUMNS} FROM scraped_bonds{where} ORDER BY {ordering} LIMIT %s",
                [*values, limit],
            ).fetchall()
        return SearchResult([observation(row) for row in rows], count)
