"""
compute_semantic_metrics.py

Evaluates semantic retrieval quality against ground-truth relevant_ids
stored in evaluation_queries.json.

Metrics computed per query:
  - Precision@3
  - Recall@5
  - NDCG@5

Global aggregates:
  - Mean Precision@3
  - Mean Recall@5
  - Mean Reciprocal Rank (MRR, over full top-10 list)
  - Mean NDCG@5

Queries whose relevant_ids list is empty are skipped from metric
computation but are listed in a summary table as "no ground truth".

Run from project root:
    python compute_semantic_metrics.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.text_embedder         import TextEmbedder   # noqa: E402
from embeddings.code_embedder         import CodeEmbedder   # noqa: E402
from embeddings.image_embedder        import ImageEmbedder  # noqa: E402
from indexing.faiss_index             import FaissIndex     # noqa: E402
from evaluation.retrieval_metrics     import (              # noqa: E402
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
QUERIES_FILE     = PROJECT_ROOT / "data" / "processed" / "evaluation_queries.json"
INDEX_DIR        = PROJECT_ROOT / "indexes"

TOP_K            = 10   # retrieve this many candidates
P_K              = 3    # Precision cut-off
R_K              = 5    # Recall cut-off
NDCG_K           = 5    # NDCG cut-off

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def section(title: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")


def _embed(query: str, modality: str,
           te: TextEmbedder, ce: CodeEmbedder, ie: ImageEmbedder):
    """Return a 1-D query vector using the correct encoder for the modality."""
    if modality == "text":
        vec = te.encode_query(query)       # (1, 1024)
    elif modality == "code":
        vec = ce.encode([query], batch_size=1)   # (1, 768)
    elif modality == "image":
        vec = ie.encode_text_query(query)  # (1, 512)
    else:
        raise ValueError(f"Unknown modality: '{modality}'")
    return vec[0] if vec.ndim == 2 else vec   # → (d,)


def _index_for(modality: str,
               text_idx: FaissIndex,
               code_idx: FaissIndex,
               image_idx: FaissIndex) -> FaissIndex:
    return {"text": text_idx, "code": code_idx, "image": image_idx}[modality]


# ---------------------------------------------------------------------------
# 1 – Load queries
# ---------------------------------------------------------------------------

queries = load_json(QUERIES_FILE)
print(f"\nLoaded {len(queries)} evaluation queries.")

# ---------------------------------------------------------------------------
# 2 – Load FAISS indexes
# ---------------------------------------------------------------------------

section("Loading FAISS indexes")

text_index  = FaissIndex(embedding_dim=1024)
text_index.load(INDEX_DIR / "faiss_text.index",  INDEX_DIR / "faiss_text_idmap.json")

code_index  = FaissIndex(embedding_dim=768)
code_index.load(INDEX_DIR / "faiss_code.index",  INDEX_DIR / "faiss_code_idmap.json")

image_index = FaissIndex(embedding_dim=512)
image_index.load(INDEX_DIR / "faiss_image.index", INDEX_DIR / "faiss_image_idmap.json")

# ---------------------------------------------------------------------------
# 3 – Load embedders
# ---------------------------------------------------------------------------

section("Loading embedders")

text_embedder  = TextEmbedder()
code_embedder  = CodeEmbedder()
image_embedder = ImageEmbedder()

# ---------------------------------------------------------------------------
# 4 – Retrieve and compute per-query metrics
# ---------------------------------------------------------------------------

section(f"Per-query metrics  (P@{P_K} | R@{R_K} | NDCG@{NDCG_K})")

# Accumulators for global aggregates
scored_queries   = []    # entries that have relevant_ids
all_retrieved    = []    # for MRR — includes every scored query
all_relevant_mrr = []

col_w = 46   # query column width
header = (
    f"  {'#':<3}  {'Query':<{col_w}}  {'Mod':<5}  "
    f"{'P@'+str(P_K):>5}  {'R@'+str(R_K):>5}  {'NDCG@'+str(NDCG_K):>7}  {'GT':>4}"
)
print(f"\n{header}")
print(f"  {'-'*3}  {'-'*col_w}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*4}")

for i, entry in enumerate(queries, start=1):
    query        = entry["query"]
    modality     = entry["modality"].lower()
    relevant_ids = entry.get("relevant_ids", [])

    # ── Embed + retrieve ──────────────────────────────────────────────────
    vec   = _embed(query, modality, text_embedder, code_embedder, image_embedder)
    idx   = _index_for(modality, text_index, code_index, image_index)
    hits  = idx.search(vec, top_k=TOP_K)
    retrieved_ids = [chunk_id for chunk_id, _ in hits]

    # ── Metrics ───────────────────────────────────────────────────────────
    has_gt = bool(relevant_ids)

    if has_gt:
        p3    = precision_at_k(retrieved_ids, relevant_ids, k=P_K)
        r5    = recall_at_k(retrieved_ids,    relevant_ids, k=R_K)
        n5    = ndcg_at_k(retrieved_ids,      relevant_ids, k=NDCG_K)

        scored_queries.append({"p3": p3, "r5": r5, "n5": n5})
        all_retrieved.append(retrieved_ids)
        all_relevant_mrr.append(relevant_ids)

        p3_s  = f"{p3:.3f}"
        r5_s  = f"{r5:.3f}"
        n5_s  = f"{n5:.3f}"
        gt_s  = str(len(relevant_ids))
    else:
        p3_s = r5_s = n5_s = "  n/a"
        gt_s = "none"

    # Truncate query for display
    q_display = query if len(query) <= col_w else query[:col_w - 3] + "..."
    print(
        f"  {i:<3}  {q_display:<{col_w}}  {modality:<5}  "
        f"{p3_s:>5}  {r5_s:>5}  {n5_s:>7}  {gt_s:>4}"
    )

# ---------------------------------------------------------------------------
# 5 – Global aggregates
# ---------------------------------------------------------------------------

section("Global metrics")

n = len(scored_queries)

if n == 0:
    print(
        "\n  No queries have ground-truth relevant_ids yet.\n"
        f"  Populate 'relevant_ids' in {QUERIES_FILE.relative_to(PROJECT_ROOT)}\n"
        "  and re-run this script."
    )
else:
    mean_p3   = sum(q["p3"] for q in scored_queries) / n
    mean_r5   = sum(q["r5"] for q in scored_queries) / n
    mean_n5   = sum(q["n5"] for q in scored_queries) / n
    mrr       = mean_reciprocal_rank(all_retrieved, all_relevant_mrr)

    print(f"\n  Evaluated queries (with ground truth) : {n} / {len(queries)}")
    print()
    print(f"  {'Metric':<30}  {'Value':>8}")
    print(f"  {'-'*30}  {'-'*8}")
    print(f"  {'Mean Precision@'+str(P_K):<30}  {mean_p3:>8.4f}")
    print(f"  {'Mean Recall@'+str(R_K):<30}  {mean_r5:>8.4f}")
    print(f"  {'Mean NDCG@'+str(NDCG_K):<30}  {mean_n5:>8.4f}")
    print(f"  {'MRR (over top-'+str(TOP_K)+')':<30}  {mrr:>8.4f}")
    print()

print(f"{'=' * 68}\n")
