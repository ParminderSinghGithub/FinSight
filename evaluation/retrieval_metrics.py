"""
evaluation/retrieval_metrics.py

Standard information-retrieval metrics for evaluating retrieval quality.

All functions assume binary relevance: a retrieved item is either relevant
(its ID appears in relevant_ids) or not.  All metrics are deterministic and
handle degenerate inputs (empty lists, k=0, no relevant items) safely by
returning 0.0 rather than raising exceptions.

Functions
---------
precision_at_k       — fraction of top-k results that are relevant
recall_at_k          — fraction of relevant items found in top-k results
mean_reciprocal_rank — average of reciprocal ranks of first relevant hit
ndcg_at_k            — normalised discounted cumulative gain at k
"""

import math


# ---------------------------------------------------------------------------
# Precision@k
# ---------------------------------------------------------------------------

def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Compute Precision@k.

    Precision@k = |relevant ∩ retrieved[:k]| / k

    The metric measures the proportion of the top-k retrieved items that
    are actually relevant.  It does not account for the rank position of
    relevant items within the top-k window.

    Args:
        retrieved_ids: Ordered list of retrieved item IDs (best first).
        relevant_ids:  Collection of IDs considered relevant for this query.
        k:             Cut-off depth.  Only the first k items are considered.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 when k <= 0, retrieved_ids is
        empty, or relevant_ids is empty.

    Examples:
        >>> precision_at_k(["a","b","c","d"], ["a","c"], k=2)
        0.5   # 1 relevant item in top-2 → 1/2
        >>> precision_at_k(["a","b","c","d"], ["a","c"], k=4)
        0.5   # 2 relevant items in top-4 → 2/4
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Compute Recall@k.

    Recall@k = |relevant ∩ retrieved[:k]| / |relevant|

    The metric measures what fraction of all relevant items were recovered
    within the top-k results.  Unlike Precision@k, the denominator is fixed
    to the total number of relevant items rather than k.

    Args:
        retrieved_ids: Ordered list of retrieved item IDs (best first).
        relevant_ids:  Collection of IDs considered relevant for this query.
        k:             Cut-off depth.  Only the first k items are considered.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 when k <= 0, retrieved_ids is
        empty, or relevant_ids is empty.

    Examples:
        >>> recall_at_k(["a","b","c","d"], ["a","c"], k=2)
        0.5   # found 1 of 2 relevant items in top-2
        >>> recall_at_k(["a","b","c","d"], ["a","c"], k=4)
        1.0   # found both relevant items in top-4
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank
# ---------------------------------------------------------------------------

def mean_reciprocal_rank(
    all_results: list[list[str]],
    all_relevant: list[list[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) across a set of queries.

    For each query q, the Reciprocal Rank (RR) is defined as:

        RR(q) = 1 / rank_of_first_relevant_item

    where rank is 1-indexed.  If no relevant item appears in the result
    list for a query, RR(q) = 0.

    MRR = (1 / |Q|) * Σ RR(q)  for q in Q

    MRR is sensitive only to the position of the *first* relevant item; it
    does not reward finding additional relevant items below that position.

    Args:
        all_results:  List of per-query ranked result lists (best first).
                      all_results[i] is the result list for query i.
        all_relevant: List of per-query relevance sets.
                      all_relevant[i] is the set of relevant IDs for query i.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 when the input lists are empty or
        no query has any relevant item in its result list.

    Raises:
        ValueError: If all_results and all_relevant have different lengths.

    Examples:
        >>> mrr = mean_reciprocal_rank(
        ...     [["b","a","c"], ["c","b","a"]],
        ...     [["a"],         ["a"]]
        ... )
        # Query 1: first relevant at rank 2 → RR=0.5
        # Query 2: first relevant at rank 3 → RR=0.333
        # MRR = (0.5 + 0.333) / 2 ≈ 0.417
    """
    if not all_results and not all_relevant:
        return 0.0

    if len(all_results) != len(all_relevant):
        raise ValueError(
            f"all_results (len={len(all_results)}) and all_relevant "
            f"(len={len(all_relevant)}) must have the same length."
        )

    total_rr = 0.0
    for retrieved_ids, relevant_ids in zip(all_results, all_relevant):
        relevant_set = set(relevant_ids)
        for rank, item in enumerate(retrieved_ids, start=1):
            if item in relevant_set:
                total_rr += 1.0 / rank
                break
        # if no relevant item found, RR contribution is 0 (no addition)

    return total_rr / len(all_results)


# ---------------------------------------------------------------------------
# NDCG@k
# ---------------------------------------------------------------------------

def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Compute Normalised Discounted Cumulative Gain at k (NDCG@k).

    With binary relevance, the gain g_i for the item at rank i (1-indexed)
    is 1 if the item is relevant and 0 otherwise.  The discount factor is
    1 / log2(i + 1).

    DCG@k  = Σ_{i=1}^{k}  g_i / log2(i + 1)

    IDCG@k is the DCG of a hypothetical perfect ranking, which places all
    min(|relevant|, k) relevant items at the top positions first.

    NDCG@k = DCG@k / IDCG@k

    NDCG@k = 1.0 means the ranking is perfect up to position k.

    Args:
        retrieved_ids: Ordered list of retrieved item IDs (best first).
        relevant_ids:  Collection of IDs considered relevant for this query.
        k:             Cut-off depth.  Only the first k items are considered.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 when k <= 0, retrieved_ids is
        empty, or relevant_ids is empty.

    Examples:
        >>> ndcg_at_k(["a","b","c","d"], ["a","c"], k=4)
        # DCG  = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) ≈ 1.0 + 0.6309 = 1.6309
        # NDCG ≈ 0.9197
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # DCG: sum gains for the actual ranking
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(top_k, start=1)
        if item in relevant_set
    )

    # IDCG: perfect ranking — relevant items fill top positions first
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
