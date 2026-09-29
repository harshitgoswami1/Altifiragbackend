"""Deterministic routing and result shaping for mixed blog/bond retrieval."""

from dataclasses import dataclass
from decimal import Decimal
import re
import unicodedata
from typing import Any, Iterable, Literal


ISIN_PATTERN = re.compile(r"(?<![A-Z0-9])IN[A-Z0-9]{10}(?![A-Z0-9])", re.IGNORECASE)

RouteName = Literal["bond", "bond_yield_filter", "bond_filter_clarification", "blog", "mixed", "ambiguous", "unknown_bond"]

BOND_YTM_PATTERN = re.compile(r"^Observed yield to maturity:\s*(\d+(?:\.\d+)?)% p\.a\.\s*$", re.MULTILINE)
PERCENT_PATTERN = r"(\d+(?:\.\d+)?)\s*(?:%|percent\b|per cent\b)"
COMPARISON_ALIASES = {
    "greater than or equal to": "gte", "more than or equal to": "gte",
    "at least": "gte", "no less than": "gte", "not less than": "gte", ">=": "gte", "≥": "gte",
    "less than or equal to": "lte", "at most": "lte", "no more than": "lte",
    "not more than": "lte", "<=": "lte", "≤": "lte",
    "greater than": "gt", "more than": "gt", "above": "gt", "over": "gt", ">": "gt",
    "less than": "lt", "below": "lt", "under": "lt", "<": "lt",
}
YIELD_THRESHOLD_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(word) for word in sorted(COMPARISON_ALIASES, key=len, reverse=True))
    + r")\s*" + PERCENT_PATTERN,
)
COUNT_WORDS = dict(zip(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
    range(1, 21),
))
BOND_COUNT_PATTERN = re.compile(r"\b(\d+|" + "|".join(COUNT_WORDS) + r")\s+bonds?\b")

GENERIC_TITLE_TOKENS = {
    "a", "about", "an", "and", "are", "can", "could", "details", "do", "does",
    "for", "from", "give", "how", "in", "information", "is", "me", "of", "on",
    "please", "show", "specific", "tell", "that", "the", "this", "to", "what",
    "which", "with", "would", "bond", "bonds",
}

MIXED_MARKERS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "difference between",
    "how does",
    "what does",
    "explain how",
    "explain why",
    "why does",
    "why is",
    "meaning of",
)


@dataclass(frozen=True)
class BondCandidate:
    chunk_id: str
    isin: str
    title: str
    normalized_title: str
    category: str
    url: str


@dataclass(frozen=True)
class QueryRoute:
    route: RouteName
    detected_isin: str | None = None
    resolved_bond: BondCandidate | None = None
    reason: str = ""
    yield_threshold: Decimal | None = None
    yield_comparison: str = "gt"
    result_limit: int = 4

    @property
    def metadata_filter(self) -> dict[str, Any] | None:
        if self.resolved_bond is not None:
            return {
                "$and": [
                    {"document_type": {"$eq": "bond"}},
                    {"isin": {"$eq": self.resolved_bond.isin}},
                ]
            }
        if self.route == "bond" or self.route == "ambiguous":
            return {"document_type": "bond"}
        if self.route == "blog":
            return {"document_type": "blog"}
        return None


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def bond_catalog(metadatas: Iterable[dict[str, Any]]) -> list[BondCandidate]:
    catalog: dict[str, BondCandidate] = {}
    for metadata in metadatas:
        isin = str(metadata.get("isin") or "").upper()
        title = str(metadata.get("title") or "").strip()
        if not isin or not title:
            continue
        catalog.setdefault(
            isin,
            BondCandidate(
                chunk_id=str(metadata.get("chunk_id") or ""),
                isin=isin,
                title=title,
                normalized_title=normalize_text(title),
                category=str(metadata.get("category") or ""),
                url=str(metadata.get("url") or ""),
            ),
        )
    return list(catalog.values())


def extract_isin(question: str) -> str | None:
    match = ISIN_PATTERN.search(question.upper())
    return match.group(0) if match else None


def _title_tokens(title: str) -> set[str]:
    return {
        token
        for token in normalize_text(title).split()
        if token not in GENERIC_TITLE_TOKENS
    }


def _query_tokens(question: str) -> set[str]:
    return {
        token
        for token in normalize_text(question).split()
        if token not in GENERIC_TITLE_TOKENS
    }


def _resolve_title(question: str, catalog: list[BondCandidate]) -> tuple[BondCandidate | None, bool]:
    normalized_question = normalize_text(question)
    exact = [candidate for candidate in catalog if candidate.normalized_title in normalized_question]
    if len(exact) == 1:
        return exact[0], False
    if len(exact) > 1:
        return None, True

    query_tokens = _query_tokens(question)
    scored: list[tuple[float, int, BondCandidate]] = []
    for candidate in catalog:
        title_tokens = _title_tokens(candidate.title)
        if len(title_tokens) < 2:
            continue
        overlap = len(title_tokens & query_tokens)
        recall = overlap / len(title_tokens)
        if overlap >= 2 and recall >= 0.8:
            scored.append((recall, overlap, candidate))
    if not scored:
        return None, False

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = scored[0]
    tied = [item for item in scored if item[0] == best[0] and item[1] == best[1]]
    if len(tied) > 1:
        return None, True
    if len(scored) > 1 and best[0] - scored[1][0] < 0.15:
        return None, True
    return best[2], False


def _has_bond_entity_hint(question: str, catalog: list[BondCandidate]) -> bool:
    normalized = normalize_text(question)
    if re.search(r"\b(this|the|that|specific|particular) bond\b", normalized):
        return True
    query_tokens = _query_tokens(question)
    return any(len(_title_tokens(candidate.title) & query_tokens) >= 2 for candidate in catalog)


def _is_mixed_question(question: str) -> bool:
    normalized = f" {normalize_text(question)} "
    return any(marker in normalized for marker in MIXED_MARKERS)


def _bond_yield_query(question: str) -> QueryRoute | None:
    text = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    if not re.search(r"\bbonds?\b", text) or not re.search(PERCENT_PATTERN, text):
        return None
    # Concept questions still use educational sources, regardless of numeric examples.
    opening = re.sub(r"^(?:(?:can|could|would|will) you\s+|please\s+|tell me\s+)+", "", text)
    if re.match(r"(?:why\b|explain\b|help me understand\b|what (?:is|does)\b|how (?:do|does|is|are)\b)", opening):
        return None
    matches = list(YIELD_THRESHOLD_PATTERN.finditer(text))
    if (len(matches) != 1 or len(re.findall(PERCENT_PATTERN, text)) != 1
            or re.search(r"\b(?:coupon|interest rate|between)\b", text)):
        return QueryRoute("bond_filter_clarification", reason="unsupported_yield_condition")
    match = matches[0]
    # Do not silently invert a negated condition such as 'not above 12%'.
    if re.search(r"\bnot\s*$", text[:match.start()]):
        return QueryRoute("bond_filter_clarification", reason="unsupported_yield_condition")
    count_match = BOND_COUNT_PATTERN.search(text)
    count = 4
    if count_match:
        value = count_match.group(1)
        count = int(value) if value.isdigit() else COUNT_WORDS[value]
        if count < 1:
            return QueryRoute("bond_filter_clarification", reason="invalid_result_count")
    return QueryRoute(
        "bond_yield_filter", reason="yield_threshold", yield_threshold=Decimal(match.group(2)),
        yield_comparison=COMPARISON_ALIASES[match.group(1)], result_limit=count,
    )


def observed_ytm(text: str) -> Decimal | None:
    values = BOND_YTM_PATTERN.findall(text)
    return Decimal(values[0]) if len(values) == 1 else None


def route_question(question: str, catalog: list[BondCandidate]) -> QueryRoute:
    detected_isin = extract_isin(question)
    if detected_isin:
        resolved = next((candidate for candidate in catalog if candidate.isin == detected_isin), None)
        if resolved is None:
            return QueryRoute("unknown_bond", detected_isin=detected_isin, reason="unknown_isin")
        route: RouteName = "mixed" if _is_mixed_question(question) else "bond"
        return QueryRoute(route, detected_isin=detected_isin, resolved_bond=resolved, reason="isin")

    resolved, ambiguous = _resolve_title(question, catalog)
    if resolved is not None:
        route = "mixed" if _is_mixed_question(question) else "bond"
        return QueryRoute(route, resolved_bond=resolved, reason="title")
    yield_query = _bond_yield_query(question)
    if yield_query is not None:
        return yield_query
    if ambiguous or _has_bond_entity_hint(question, catalog):
        return QueryRoute("ambiguous", reason="bond_name")
    return QueryRoute("blog", reason="general_question")


def deduplicate_matches(
    matches: list[tuple[Any, float]],
    limit: int,
) -> list[tuple[Any, float]]:
    best_by_document: dict[str, tuple[Any, float]] = {}
    for document, score in matches:
        metadata = document.metadata
        key = str(metadata.get("document_id") or metadata.get("chunk_id") or "")
        current = best_by_document.get(key)
        if current is None or score > current[1]:
            best_by_document[key] = (document, score)
    return sorted(best_by_document.values(), key=lambda item: item[1], reverse=True)[:limit]
