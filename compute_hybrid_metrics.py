"""
compute_hybrid_metrics.py

Evaluates hybrid (semantic + BM25 via RRF) retrieval quality against the
ground-truth relevant_ids in evaluation_queries.json.

Metrics computed per query:
  - Precision@3
  - Recall@5
  - NDCG@5

Global aggregates:
  - Mean Precision@3
  - Mean Recall@5
  - Mean NDCG@5
  - MRR  (over full top-10 list)

Queries whose relevant_ids list is empty are skipped from metric
computation but still appear in the table as "no ground truth".

Run from project root:
    python compute_hybrid_metrics.py
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.hybrid_search      import HybridSearchEngine  # noqa: E402
from config.settings              import get_path, get_retrieval_top_k  # noqa: E402
from evaluation.retrieval_metrics import (                   # noqa: E402
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / Path(get_path("data"))
QUERIES_FILE = DATA_DIR / "processed" / "evaluation_queries.json"

TOP_K   = get_retrieval_top_k()   # retrieve this many candidates
P_K     = 3    # Precision cut-off
R_K     = 5    # Recall cut-off
NDCG_K  = 5    # NDCG cut-off

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def section(title: str) -> None:
    print(f"\n{title}")


def main() -> dict:
    # -----------------------------------------------------------------------
    # 1 – Load queries
    # -----------------------------------------------------------------------
    queries = load_json(QUERIES_FILE)
    print(f"\nLoaded {len(queries)} evaluation queries from {QUERIES_FILE.name}")

    # -----------------------------------------------------------------------
    # 2 – Instantiate hybrid engine (BM25 indexes must already exist)
    # -----------------------------------------------------------------------
    engine = HybridSearchEngine.from_disk(rebuild_bm25=False)

    # -----------------------------------------------------------------------
    # 3 – Retrieve and compute per-query metrics
    # -----------------------------------------------------------------------
    section(f"Per-query metrics  (P@{P_K} | R@{R_K} | NDCG@{NDCG_K})")

    scored_queries    = []
    all_retrieved     = []
    all_relevant_mrr  = []
    per_query_rows    = []

    col_w = 46
    header = (
        f"  {'#':<3}  {'Query':<{col_w}}  {'Mod':<5}  "
        f"{'P@'+str(P_K):>5}  {'R@'+str(R_K):>5}  {'NDCG@'+str(NDCG_K):>7}  {'GT':>4}"
    )
    print(f"\n{header}")

    for i, entry in enumerate(queries, start=1):
        query        = entry["query"]
        modality     = entry["modality"].lower()
        relevant_ids = entry.get("relevant_ids", [])

        hits          = engine.search(query, modality=modality, top_k=TOP_K)
        retrieved_ids = [chunk_id for chunk_id, _ in hits]

        has_gt = bool(relevant_ids)

        if has_gt:
            p3 = precision_at_k(retrieved_ids, relevant_ids, k=P_K)
            r5 = recall_at_k(retrieved_ids,    relevant_ids, k=R_K)
            n5 = ndcg_at_k(retrieved_ids,      relevant_ids, k=NDCG_K)

            scored_queries.append({"p3": p3, "r5": r5, "n5": n5})
            all_retrieved.append(retrieved_ids)
            all_relevant_mrr.append(relevant_ids)

            p3_s = f"{p3:.3f}"
            r5_s = f"{r5:.3f}"
            n5_s = f"{n5:.3f}"
            gt_s = str(len(relevant_ids))
        else:
            p3 = r5 = n5 = None
            p3_s = r5_s = n5_s = "  n/a"
            gt_s = "none"

        per_query_rows.append(
            {
                "qid": i,
                "query": query,
                "modality": modality,
                "p@3": p3,
                "r@5": r5,
                "ndcg@5": n5,
                "relevant_count": len(relevant_ids),
                "retrieved": retrieved_ids,
            }
        )

        q_display = query if len(query) <= col_w else query[:col_w - 3] + "..."
        print(
            f"{i:<3}  {q_display:<{col_w}}  {modality:<5}  "
            f"{p3_s:>5}  {r5_s:>5}  {n5_s:>7}  {gt_s:>4}"
        )

    # -----------------------------------------------------------------------
    # 4 – Global aggregates
    # -----------------------------------------------------------------------
    section("Global metrics  (hybrid retrieval)")

    n = len(scored_queries)

    if n == 0:
        print(
            "\nNo queries have ground-truth relevant_ids yet.\n"
            f"  Populate 'relevant_ids' in {QUERIES_FILE.relative_to(PROJECT_ROOT)}\n"
            "  and re-run this script."
        )
        result = {
            "global": {
                "mean_p@3": None,
                "mean_r@5": None,
                "mean_ndcg@5": None,
                "mean_mrr": None,
                "evaluated": 0,
                "total": len(queries),
            },
            "per_query": per_query_rows,
        }
    else:
        mean_p3 = sum(q["p3"] for q in scored_queries) / n
        mean_r5 = sum(q["r5"] for q in scored_queries) / n
        mean_n5 = sum(q["n5"] for q in scored_queries) / n
        mrr     = mean_reciprocal_rank(all_retrieved, all_relevant_mrr)

        print(f"\nEvaluated queries (with ground truth): {n} / {len(queries)}")
        print()
        print(f"{'Metric':<30} {'Value':>8}")
        print(f"{'Mean Precision@'+str(P_K):<30} {mean_p3:>8.4f}")
        print(f"{'Mean Recall@'+str(R_K):<30} {mean_r5:>8.4f}")
        print(f"{'Mean NDCG@'+str(NDCG_K):<30} {mean_n5:>8.4f}")
        print(f"{'MRR (over top-'+str(TOP_K)+')':<30} {mrr:>8.4f}")
        print()
        result = {
            "global": {
                "mean_p@3": mean_p3,
                "mean_r@5": mean_r5,
                "mean_ndcg@5": mean_n5,
                "mean_mrr": mrr,
                "evaluated": n,
                "total": len(queries),
            },
            "per_query": per_query_rows,
        }
    print()
    return result


if __name__ == "__main__":
    main()
