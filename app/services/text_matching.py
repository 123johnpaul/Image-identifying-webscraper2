from __future__ import annotations

from typing import Iterable, List, Tuple

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None


def score_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if fuzz is None:
        # Simple fallback: token overlap ratio.
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        return 100.0 * len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))
    return float(fuzz.token_set_ratio(a, b))


def filter_matches(query: str, candidates: Iterable[str], threshold: float = 70.0) -> List[Tuple[str, float]]:
    scored = []
    for c in candidates:
        score = score_similarity(query, c)
        if score >= threshold:
            scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
