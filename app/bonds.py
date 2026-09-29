"""Typed snapshot observations and deterministic bond screening."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import operator
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


BondField = Literal[
    "ytm", "coupon", "minimum_investment", "maturity", "rating", "security",
    "category", "payment_frequency", "title",
]
SortField = Literal["ytm", "coupon", "minimum_investment", "maturity"]
NUMERIC_FIELDS = {"ytm", "coupon", "minimum_investment"}
LABELS = {
    "ytm": "Observed yield to maturity", "coupon": "Coupon rate",
    "minimum_investment": "Observed minimum investment", "maturity": "Maturity date",
    "rating": "Credit rating", "security": "Security", "category": "Instrument category",
    "payment_frequency": "Interest payment frequency", "title": "Bond",
}
FREQUENCIES = {
    "monthly": "monthly", "quarterly": "quarterly", "yearly": "annually",
    "annual": "annually", "annually": "annually", "half yearly": "half yearly",
    "half_yearly": "half yearly", "semi annual": "half yearly", "semiannual": "half yearly",
    "semiannually": "half yearly", "maturity": "maturity", "at maturity": "maturity",
    "structured": "structured",
}
RATINGS = {
    "sovereign", "aaa", "aa+", "aa", "aa-", "a+", "a", "a-", "bbb+", "bbb",
    "bbb-", "bb+", "bb", "bb-", "b+", "b", "b-", "c", "d", "unrated",
    "a1+", "a1", "a2+", "a2", "a3", "a4",
}
SECURITIES = {"secured", "senior secured", "unsecured", "senior unsecured", "unsecured subordinate", "subordinated"}
CATEGORIES = {
    "government securities": "government-securities", "government bonds": "government-securities",
    "g sec": "government-securities", "g secs": "government-securities",
    "corporate bonds": "corporate-bonds", "commercial paper": "commercial-paper",
    "securitized debt instruments": "securitized-debt-instruments",
    "treasury bills": "treasury-bills", "t bills": "treasury-bills",
    "state development loans": "state-development-loans",
}
COMPARE = {"eq": operator.eq, "gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le}


def normalize_words(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def typed_value(field: str, value: str) -> Decimal | date | str:
    if field in NUMERIC_FIELDS:
        number = Decimal(value)
        if not number.is_finite() or number < 0:
            raise ValueError("Expected a finite nonnegative value")
        return number
    if field == "maturity":
        return date.fromisoformat(value)
    value = " ".join(value.casefold().split())
    if field == "rating":
        if re.sub(r"\s*\(ce\)$", "", value) not in RATINGS:
            raise ValueError("Unsupported rating")
    elif field == "security" and value not in SECURITIES:
        raise ValueError("Unsupported security")
    elif field == "category" and value not in set(CATEGORIES.values()):
        raise ValueError("Unsupported category")
    elif field == "payment_frequency":
        value = FREQUENCIES.get(value, "")
        if not value:
            raise ValueError("Unsupported payment frequency")
    elif field == "title":
        value = normalize_words(value)
        if not value:
            raise ValueError("Empty title constraint")
    return value


class BondFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: BondField
    op: Literal["eq", "gt", "gte", "lt", "lte"]
    value: str
    source: str

    @model_validator(mode="after")
    def valid_condition(self) -> "BondFilter":
        typed_value(self.field, self.value)
        if self.field not in NUMERIC_FIELDS | {"maturity"} and self.op != "eq":
            raise ValueError("Only exact categorical conditions are supported")
        if not self.source.strip():
            raise ValueError("A condition needs a source phrase")
        return self


class BondSort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: SortField
    descending: bool
    source: str


@dataclass(frozen=True)
class BondObservation:
    metadata: dict[str, Any]
    text: str
    values: dict[str, Decimal | date | str | None]
    flags: tuple[str, ...]


def parse_bond(metadata: dict[str, Any], text: str) -> BondObservation:
    values: dict[str, Decimal | date | str | None] = {}
    for field, label in LABELS.items():
        matches = re.findall(r"^" + re.escape(label) + r":\s*([^\n]+)$", text, re.MULTILINE)
        raw = matches[0].strip() if len(matches) == 1 else ""
        if field in {"title", "category"}:
            raw = str(metadata.get(field) or raw)
        if field in {"ytm", "coupon"}:
            match = re.fullmatch(r"(\d+(?:\.\d+)?)% p\.a\.", raw)
            raw = match[1] if match else ""
        elif field == "minimum_investment":
            match = re.fullmatch(r"(\d+(?:\.\d+)?) INR", raw)
            raw = match[1] if match else ""
        try:
            values[field] = typed_value(field, raw) if raw else None
        except (ValueError, InvalidOperation):
            values[field] = None
    try:
        flags = json.loads(metadata.get("quality_flags", "[]"))
    except (ValueError, TypeError):
        flags = []
    return BondObservation(metadata, text, values, tuple(flags) if isinstance(flags, list) else ())


def matches_condition(record: BondObservation, condition: BondFilter) -> bool:
    actual = record.values.get(condition.field)
    if actual is None:
        return False
    expected = typed_value(condition.field, condition.value)
    if condition.field == "title":
        return f" {expected} " in f" {actual} "
    if condition.field == "security":
        if expected == "secured":
            return actual in {"secured", "senior secured"}
        if expected == "unsecured":
            return actual in {"unsecured", "senior unsecured", "unsecured subordinate"}
    return COMPARE[condition.op](actual, expected)


def contradictory(filters: list[BondFilter]) -> bool:
    for field in LABELS:
        conditions = [condition for condition in filters if condition.field == field]
        if field in NUMERIC_FIELDS | {"maturity"}:
            # Any feasible interval has an endpoint satisfying all conditions, or
            # lies strictly between its strongest lower and upper bounds.
            lower = [(typed_value(field, c.value), c.op == "gt") for c in conditions if c.op in {"gt", "gte", "eq"}]
            upper = [(typed_value(field, c.value), c.op == "lt") for c in conditions if c.op in {"lt", "lte", "eq"}]
            if lower and upper:
                low = max(lower)
                high = min(upper, key=lambda item: (item[0], not item[1]))
                if low[0] > high[0] or (low[0] == high[0] and (low[1] or high[1])):
                    return True
        elif field != "title":
            allowed = None
            for condition in conditions:
                value = typed_value(field, condition.value)
                choices = {value}
                if field == "security" and value == "secured":
                    choices = {"secured", "senior secured"}
                elif field == "security" and value == "unsecured":
                    choices = {"unsecured", "senior unsecured", "unsecured subordinate"}
                allowed = choices if allowed is None else allowed & choices
            if allowed == set():
                return True
    return False


def screen_bonds(
    records: list[BondObservation], filters: list[BondFilter], sorting: BondSort | None,
    include_matured: bool, isins: list[str] | None = None,
) -> tuple[list[BondObservation], int]:
    eligible = []
    missing = 0
    required = {condition.field for condition in filters}
    if sorting:
        required.add(sorting.field)
    for record in records:
        if isins is not None and record.metadata.get("isin") not in isins:
            continue
        if not include_matured and "maturity_matured" in record.flags:
            continue
        if any(record.values.get(condition.field) is not None and not matches_condition(record, condition)
               for condition in filters):
            continue
        if any(record.values.get(field) is None for field in required):
            missing += 1
            continue
        eligible.append(record)
    eligible.sort(key=lambda record: (str(record.metadata.get("isin", "")), str(record.metadata.get("chunk_id", ""))))
    if sorting:
        eligible.sort(key=lambda record: record.values[sorting.field], reverse=sorting.descending)
    return eligible, missing


def display_value(record: BondObservation, field: str) -> str:
    value = record.values.get(field)
    if value is None:
        return "not recorded or malformed"
    if field in {"ytm", "coupon"}:
        return f"{value}% p.a."
    if field == "minimum_investment":
        return f"{value} INR"
    return str(value)
