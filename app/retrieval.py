"""Deterministic routing and result shaping for mixed blog/bond retrieval."""

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Iterable, Literal


ISIN_PATTERN = re.compile(r"(?<![A-Z0-9])IN[A-Z0-9]{10}(?![A-Z0-9])", re.IGNORECASE)

RouteName = Literal["bond", "blog", "mixed", "ambiguous", "unknown_bond"]

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
