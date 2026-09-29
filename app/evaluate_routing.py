"""Evaluate routing against held-out decisions without retrieval or answer generation.

Usage: uv run python -m app.evaluate_routing tests/routing_eval.json
Uses the configured Ollama endpoint only for requests needing fallback.
"""

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from app.api import _route_request
from app.config import Settings
from app.retrieval import QueryRoute, bond_catalog


def decision_signature(route: QueryRoute) -> dict[str, Any]:
    return {
        "intent": route.intent,
        "isins": sorted(route.isins),
        "filters": sorted([[f.field, f.op, f.value] for f in route.filters]),
        "sorting": [route.sorting.field, route.sorting.descending] if route.sorting else None,
        "result_limit": route.result_limit,
        "include_matured": route.include_matured,
        "explanation": bool(route.explanation),
        "requested_fields": sorted(route.requested_fields),
        "limitations": sorted(route.limitations),
    }


def unsafe_decision(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if actual["intent"] in {"clarification", "capability"}:
        return False
    if actual["intent"] != expected["intent"] and expected["intent"] in {"clarification", "capability"}:
        return True
    # Returning facts after dropping a restriction is unsafe even if the remaining
    # conditions happen to select the same records in a small test snapshot.
    return any(actual[key] != expected[key] for key in (
        "isins", "filters", "sorting", "result_limit", "include_matured", "limitations",
    ))


async def evaluate(path: Path, settings: Settings) -> bool:
    fixture = json.loads(path.read_text())
    catalog = bond_catalog(fixture["catalog"])
    cases = fixture["cases"]
    if not cases:
        raise ValueError("Evaluation requires held-out cases")
    required = set(decision_signature(QueryRoute(intent="clarification")))
    for case in cases:
        if set(case["expected"]) != required:
            raise ValueError(f"Incomplete expected decision for {case['id']}")
    correct = unsafe = model_calls = 0
    current = replace(settings, router_enabled=True)
    for case in cases:
        decision = await _route_request(case["question"], catalog, current, None)
        actual = decision_signature(decision)
        expected = case["expected"]
        ok = actual == expected
        critical = unsafe_decision(actual, expected)
        correct += ok
        unsafe += critical
        model_calls += decision.method == "model"
        print(json.dumps({"case": case["id"], "correct": ok, "unsafe": critical,
                          "method": decision.method, "reason": decision.reason,
                          "latency_ms": round(decision.fallback_latency_ms, 1),
                          **({"actual": actual, "expected": expected} if not ok else {})}), flush=True)
    accuracy = correct / len(cases)
    passed = accuracy >= 0.95 and unsafe == 0 and model_calls > 0
    print(json.dumps({"passed": passed, "accuracy": accuracy, "unsafe_decisions": unsafe,
                      "model_calls": model_calls, "cases": len(cases)}))
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.cases, Settings.from_env())) else 1)


if __name__ == "__main__":
    main()
