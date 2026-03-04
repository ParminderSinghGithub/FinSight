"""
evaluate_semantic_baseline.py

Runs semantic retrieval for every query in evaluation_queries.json and prints
the top-10 ranked results for manual inspection.

No metrics are computed here — this is a qualitative baseline pass to allow
ground-truth relevant_ids to be assigned before quantitative evaluation.

Run from project root:
    python evaluate_semantic_baseline.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.text_embedder  import TextEmbedder   # noqa: E402
from embeddings.code_embedder  import CodeEmbedder   # noqa: E402
from embeddings.image_embedder import ImageEmbedder  # noqa: E402
from indexing.faiss_index      import FaissIndex     # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
QUERIES_FILE      = PROJECT_ROOT / "data" / "processed" / "evaluation_queries.json"

INDEX_DIR         = PROJECT_ROOT / "indexes"
TEXT_INDEX_PATH   = INDEX_DIR / "faiss_text.index"
TEXT_IDMAP_PATH   = INDEX_DIR / "faiss_text_idmap.json"
CODE_INDEX_PATH   = INDEX_DIR / "faiss_code.index"
CODE_IDMAP_PATH   = INDEX_DIR / "faiss_code_idmap.json"
IMAGE_INDEX_PATH  = INDEX_DIR / "faiss_image.index"
IMAGE_IDMAP_PATH  = INDEX_DIR / "faiss_image_idmap.json"

TOP_K = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def section_header(text: str) -> None:
    width = 66
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def print_query_block(
    rank_offset: int,
    query: str,
    modality: str,
    results: list[tuple[str, float]],
) -> None:
    print(f"\n  Query [{modality.upper()}]: \"{query}\"")
    print(f"  {'Rank':<5}  {'ID':<18}  {'Score':>8}")
    print(f"  {'-'*5}  {'-'*18}  {'-'*8}")
    for rank, (chunk_id, score) in enumerate(results, start=1):
        print(f"  {rank:<5}  {chunk_id:<18}  {score:>8.4f}")


# ---------------------------------------------------------------------------
# 1 – Load queries
# ---------------------------------------------------------------------------

queries = load_json(QUERIES_FILE)
print(f"\nLoaded {len(queries)} evaluation queries from {QUERIES_FILE.name}")

# ---------------------------------------------------------------------------
# 2 – Load FAISS indexes
# ---------------------------------------------------------------------------

section_header("Loading FAISS indexes")

text_index  = FaissIndex(embedding_dim=1024)
text_index.load(TEXT_INDEX_PATH, TEXT_IDMAP_PATH)

code_index  = FaissIndex(embedding_dim=768)
code_index.load(CODE_INDEX_PATH, CODE_IDMAP_PATH)

image_index = FaissIndex(embedding_dim=512)
image_index.load(IMAGE_INDEX_PATH, IMAGE_IDMAP_PATH)

# ---------------------------------------------------------------------------
# 3 – Instantiate embedders
# ---------------------------------------------------------------------------

section_header("Loading embedders")

text_embedder  = TextEmbedder()
code_embedder  = CodeEmbedder()
image_embedder = ImageEmbedder()

# ---------------------------------------------------------------------------
# 4 – Run retrieval for each query
# ---------------------------------------------------------------------------

section_header(f"Semantic retrieval  —  top {TOP_K} per query")

for i, entry in enumerate(queries, start=1):
    query    = entry["query"]
    modality = entry["modality"].lower()

    # ── Embed the query with the modality-appropriate encoder ────────────────
    if modality == "text":
        # BGE encode_query() prepends retrieval instruction prefix
        vec = text_embedder.encode_query(query)        # (1, 1024)
        index = text_index

    elif modality == "code":
        # CodeBERT: treat the query string like a short code snippet
        vec = code_embedder.encode([query], batch_size=1)  # (1, 768)
        index = code_index

    elif modality == "image":
        # CLIP text encoder → same embedding space as image vectors
        vec = image_embedder.encode_text_query(query)  # (1, 512)
        index = image_index

    else:
        print(f"  [WARN] Unknown modality '{modality}' for query {i} — skipping.")
        continue

    # ── Search ───────────────────────────────────────────────────────────────
    query_vector = vec[0] if vec.ndim == 2 else vec    # ensure 1-D (d,)
    results = index.search(query_vector, top_k=TOP_K)

    # ── Print results ─────────────────────────────────────────────────────────
    print_query_block(i, query, modality, results)

print(f"\n{'=' * 66}")
print(f"  Done. Inspect results above and populate relevant_ids in")
print(f"  {QUERIES_FILE.relative_to(PROJECT_ROOT)}")
print(f"{'=' * 66}\n")
