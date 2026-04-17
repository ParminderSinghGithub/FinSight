"""Utilities to improve code retrieval quality without changing models."""

from __future__ import annotations

import re
from pathlib import Path

# Keep boosts small and additive so semantic ranking remains dominant.
FUNCTION_NAME_BOOST = 0.06
FILENAME_BOOST = 0.04

# Lightweight stopword list for metadata keyword matching.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "to", "using",
    "with", "without", "find", "locate", "retrieve", "code", "function", "example",
    "implementation", "python",
}


def normalize_code_query(query: str) -> str:
    """Expand code queries with rule-based phrases before embedding/search."""
    q = (query or "").strip()
    if not q:
        return q

    q_lower = q.lower()

    if "sharpe ratio" in q_lower:
        return f"{q} python function to compute sharpe ratio implementation example"

    if "roc curve" in q_lower:
        return f"{q} sklearn roc curve example python implementation"

    return f"{q} python function example implementation"


def _query_keywords(query: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9_]+", (query or "").lower()))
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def apply_code_metadata_boost(
    query: str,
    ranked_results: list[tuple[str, float]],
    code_meta_by_chunk_id: dict[str, dict],
    function_name_boost: float = FUNCTION_NAME_BOOST,
    filename_boost: float = FILENAME_BOOST,
) -> list[tuple[str, float]]:
    """
    Add small metadata bonuses to code chunks before final ranking.

    - +function_name_boost if any query keyword appears in function_name
    - +filename_boost if any query keyword appears in filename

    Stable sort is used so ties preserve original ranking.
    """
    if not ranked_results:
        return ranked_results

    keywords = _query_keywords(query)
    if not keywords:
        return ranked_results

    scored: list[tuple[int, str, float]] = []

    for rank, (chunk_id, base_score) in enumerate(ranked_results):
        meta = code_meta_by_chunk_id.get(chunk_id, {})
        function_name = str(meta.get("function_name", "")).lower()
        file_name = Path(str(meta.get("source_file", ""))).name.lower()

        bonus = 0.0
        if function_name and any(kw in function_name for kw in keywords):
            bonus += function_name_boost
        if file_name and any(kw in file_name for kw in keywords):
            bonus += filename_boost

        scored.append((rank, chunk_id, float(base_score) + bonus))

    scored.sort(key=lambda x: (-x[2], x[0]))
    return [(chunk_id, boosted_score) for _, chunk_id, boosted_score in scored]
