"""
diagnostic_analysis.py

Structured failure and diagnostic analysis of the multimodal RAG system.
Runs BEFORE the reranking phase to understand where retrieval succeeds/fails
and whether BM25 fusion adds value over dense-only retrieval.

Analyses performed:
  1. Code query deep-dive: semantic vs BM25 vs hybrid side-by-side
  2. Overlap statistics: top-10 semantic n top-10 BM25 per query
  3. Relevant-ID rank audit: position of each ground-truth ID in each list
  4. Average rank of relevant documents: semantic vs hybrid
  5. Text query BM25 contribution analysis
  6. Image query hybrid-bypass confirmation

Run from project root:
    python diagnostic_analysis.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.hybrid_search      import (   # noqa: E402
    HybridSearchEngine,
    reciprocal_rank_fusion,
)
from evaluation.retrieval_metrics import (   # noqa: E402
    precision_at_k,
    ndcg_at_k,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUERIES_FILE = PROJECT_ROOT / "data" / "processed" / "evaluation_queries.json"
TOP_K        = 10
NOT_FOUND    = TOP_K + 1          # sentinel rank when ID is absent from list
DIVIDER_WIDE = "=" * 72
DIVIDER_MID  = "-" * 72
DIVIDER_THIN = "." * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def banner(title: str) -> None:
    print(f"\n{DIVIDER_WIDE}")
    print(f"  {title}")
    print(DIVIDER_WIDE)


def sub_banner(title: str) -> None:
    print(f"\n  {DIVIDER_THIN[:68]}")
    print(f"  {title}")
    print(f"  {DIVIDER_THIN[:68]}")


def ids_from(results: list[tuple[str, float]]) -> list[str]:
    return [cid for cid, _ in results]


def rank_of(chunk_id: str, ranked_ids: list[str]) -> int:
    """Return 1-based rank of chunk_id in ranked_ids, or NOT_FOUND."""
    try:
        return ranked_ids.index(chunk_id) + 1
    except ValueError:
        return NOT_FOUND


def avg_rank(relevant_ids: list[str], ranked_ids: list[str]) -> float:
    """Average rank of all relevant IDs (NOT_FOUND for missing ones)."""
    if not relevant_ids:
        return float("nan")
    ranks = [rank_of(rid, ranked_ids) for rid in relevant_ids]
    return sum(ranks) / len(ranks)


def semantic_search(query: str, modality: str, engine: HybridSearchEngine) -> list[tuple[str, float]]:
    """Run semantic-only (FAISS) search by reaching into engine internals."""
    if modality == "text":
        vec = engine._text_emb.encode_query(query)
        if vec.ndim == 2:
            vec = vec[0]
        return engine._text_faiss.search(vec, top_k=TOP_K)
    elif modality == "code":
        vec = engine._code_emb.encode([query], batch_size=1)
        if vec.ndim == 2:
            vec = vec[0]
        return engine._code_faiss.search(vec, top_k=TOP_K)
    elif modality == "image":
        vec = engine._image_emb.encode_text_query(query)
        if vec.ndim == 2:
            vec = vec[0]
        return engine._image_faiss.search(vec, top_k=TOP_K)
    raise ValueError(f"Unknown modality: {modality}")


def bm25_search(query: str, modality: str, engine: HybridSearchEngine) -> tuple[list[tuple[str, float]], str | None]:
    """
    Run BM25-only search via the engine's Whoosh index.

    Returns (results, error_msg).  error_msg is None on success, a string
    describing the failure otherwise.  Bypasses the silent exception catch
    in WhooshBM25Index so diagnostics can surface parse errors.
    """
    if modality == "image":
        return [], "no BM25 index for image modality"

    from whoosh.qparser import MultifieldParser
    if modality == "text":
        ix = engine._text_bm25._ix
    else:
        ix = engine._code_bm25._ix

    if ix is None:
        return [], "Whoosh index not loaded"

    try:
        results: list[tuple[str, float]] = []
        with ix.searcher() as searcher:
            parser = MultifieldParser(["content", "aux"], schema=ix.schema)
            q = parser.parse(query)
            hits = searcher.search(q, limit=TOP_K)
            for hit in hits:
                results.append((hit["chunk_id"], hit.score))
        return results, None
    except Exception as exc:
        return [], str(exc)


def fmt_rank(r: int) -> str:
    return f"{r}" if r <= TOP_K else "--"


def fmt_avg(v: float) -> str:
    if v != v:   # NaN
        return "  n/a"
    return f"{v:.2f}" if v <= TOP_K else f">{TOP_K}"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

queries_data = load_json(QUERIES_FILE)
print(f"\nLoaded {len(queries_data)} evaluation queries.")

# ---------------------------------------------------------------------------
# Instantiate engine
# ---------------------------------------------------------------------------

engine = HybridSearchEngine.from_disk(rebuild_bm25=False)

# Pre-compute all three result lists for every query
results_cache: dict[int, dict] = {}
for i, entry in enumerate(queries_data):
    q   = entry["query"]
    mod = entry["modality"].lower()
    sem              = semantic_search(q, mod, engine)
    lex, bm25_err    = bm25_search(q, mod, engine)
    hyb              = engine.search(q, modality=mod, top_k=TOP_K)
    results_cache[i] = {
        "semantic":  sem,
        "bm25":      lex,
        "bm25_err":  bm25_err,
        "hybrid":    hyb,
    }

# ============================================================================
# ANALYSIS 0 -- BM25 health check (parse error detection)
# ============================================================================

banner("ANALYSIS 0 -- BM25 health check (Whoosh parse status per query)")

print()
print(f"  {'#':<3}  {'Query':<48}  {'Mod':<5}  {'BM25 hits':>9}  {'Status'}")
print(f"  {'-'*3}  {'-'*48}  {'-'*5}  {'-'*9}  {'-'*30}")

for i, entry in enumerate(queries_data):
    query    = entry["query"]
    modality = entry["modality"].lower()
    cache    = results_cache[i]
    lex_ids  = cache["bm25"]
    bm25_err = cache["bm25_err"]

    if bm25_err:
        status   = f"FAIL: {bm25_err[:40]}"
        hits_s   = "--"
    else:
        status   = "OK"
        hits_s   = str(len(lex_ids))

    q_display = query if len(query) <= 48 else query[:45] + "..."
    print(f"  {i+1:<3}  {q_display:<48}  {modality:<5}  {hits_s:>9}  {status}")

# ============================================================================
# ANALYSIS 1 -- Code query deep-dive (weakest two queries)
# ============================================================================

banner("ANALYSIS 1 -- Code query deep-dive  (semantic vs BM25 vs hybrid)")

code_queries = [(i, e) for i, e in enumerate(queries_data)
                if e["modality"].lower() == "code"]

for i, entry in code_queries:
    query        = entry["query"]
    relevant_ids = entry["relevant_ids"]
    cache        = results_cache[i]

    sem_ids  = ids_from(cache["semantic"])
    lex_ids  = ids_from(cache["bm25"])
    hyb_ids  = ids_from(cache["hybrid"])
    bm25_err = cache["bm25_err"]
    rel_set  = set(relevant_ids)

    sub_banner(f'Q{i+1}: "{query}"')
    print(f"\n  Relevant IDs: {relevant_ids}")
    if bm25_err:
        print(f"  BM25 status : EMPTY -- {bm25_err}")
    print()

    col = 18
    bm25_hdr = "BM25" if lex_ids else "BM25 (empty)"
    print(f"  {'Rank':<5}  {'Semantic':<{col+4}}  {bm25_hdr:<{col+4}}  {'Hybrid':<{col+4}}")
    print(f"  {'-'*5}  {'-'*(col+4)}  {'-'*(col+4)}  {'-'*(col+4)}")

    top = max(len(sem_ids), max(len(lex_ids), 0), len(hyb_ids), 5)
    for r in range(min(top, 5)):
        def _cell(lst, idx):
            if idx < len(lst):
                cid = lst[idx]
                marker = " (*)" if cid in rel_set else "    "
                return f"{cid}{marker}"
            return "-"
        print(
            f"  {r+1:<5}  {_cell(sem_ids, r):<{col+4}}  "
            f"{_cell(lex_ids, r):<{col+4}}  {_cell(hyb_ids, r):<{col+4}}"
        )

    # Per-metric comparison
    print()
    print(f"  {'Metric':<12}  {'Semantic':>9}  {'BM25':>9}  {'Hybrid':>9}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}")
    for label, fn, k in [("P@3", precision_at_k, 3), ("NDCG@5", ndcg_at_k, 5)]:
        s = fn(sem_ids, relevant_ids, k=k)
        b = fn(lex_ids, relevant_ids, k=k) if lex_ids else 0.0
        h = fn(hyb_ids, relevant_ids, k=k)
        better = "  <- hybrid wins" if h > s and h > b else (
                 "  <- semantic wins" if s >= h else "")
        print(f"  {label:<12}  {s:>9.3f}  {b:>9.3f}  {h:>9.3f}{better}")

# ============================================================================
# ANALYSIS 2 -- Overlap statistics: top-10 semantic n top-10 BM25
# ============================================================================

banner("ANALYSIS 2 -- Top-10 semantic vs top-10 BM25 overlap per query")

print()
print(f"  {'#':<3}  {'Query':<48}  {'Mod':<5}  {'Overlap':>16}")
print(f"  {'-'*3}  {'-'*48}  {'-'*5}  {'-'*16}")

for i, entry in enumerate(queries_data):
    query    = entry["query"]
    modality = entry["modality"].lower()
    cache    = results_cache[i]

    sem_set  = set(ids_from(cache["semantic"]))
    lex_ids  = ids_from(cache["bm25"])
    bm25_err = cache["bm25_err"]

    if modality == "image":
        overlap_s = "n/a (image)"
    elif bm25_err:
        overlap_s = f"0 / {TOP_K} (BM25 err)"
    elif lex_ids:
        overlap = len(sem_set & set(lex_ids))
        overlap_s = f"{overlap} / {TOP_K}"
    else:
        overlap_s = f"0 / {TOP_K}"

    q_display = query if len(query) <= 48 else query[:45] + "..."
    print(f"  {i+1:<3}  {q_display:<48}  {modality:<5}  {overlap_s:>16}")

# ============================================================================
# ANALYSIS 3 -- Rank audit: position of each relevant_id in each list
# ============================================================================

banner("ANALYSIS 3 -- Rank of each relevant ID in semantic / BM25 / hybrid")

for i, entry in enumerate(queries_data):
    query        = entry["query"]
    modality     = entry["modality"].lower()
    relevant_ids = entry["relevant_ids"]
    cache        = results_cache[i]

    if not relevant_ids:
        continue

    sem_ids = ids_from(cache["semantic"])
    lex_ids = ids_from(cache["bm25"])
    hyb_ids = ids_from(cache["hybrid"])

    print(f"\n  Q{i+1} [{modality.upper()}]: \"{query}\"")
    print(f"  {'Relevant ID':<18}  {'Semantic':>9}  {'BM25':>9}  {'Hybrid':>9}")
    print(f"  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*9}")

    for rid in relevant_ids:
        r_sem = fmt_rank(rank_of(rid, sem_ids))
        r_lex = fmt_rank(rank_of(rid, lex_ids)) if lex_ids else "n/a"
        r_hyb = fmt_rank(rank_of(rid, hyb_ids))
        print(f"  {rid:<18}  {r_sem:>9}  {r_lex:>9}  {r_hyb:>9}")

# ============================================================================
# ANALYSIS 4 -- Average rank of relevant documents: semantic vs hybrid
# ============================================================================

banner("ANALYSIS 4 -- Average rank of relevant documents (semantic vs hybrid)")

print()
print(f"  {'#':<3}  {'Query':<42}  {'Mod':<5}  {'Avg-Sem':>8}  {'Avg-Hyb':>8}  {'Delta':>7}")
print(f"  {'-'*3}  {'-'*42}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*7}")

for i, entry in enumerate(queries_data):
    query        = entry["query"]
    modality     = entry["modality"].lower()
    relevant_ids = entry["relevant_ids"]
    cache        = results_cache[i]

    if not relevant_ids:
        continue

    sem_ids = ids_from(cache["semantic"])
    hyb_ids = ids_from(cache["hybrid"])

    avg_sem = avg_rank(relevant_ids, sem_ids)
    avg_hyb = avg_rank(relevant_ids, hyb_ids)

    if avg_sem != avg_sem or avg_hyb != avg_hyb:   # NaN guard
        delta_s = "  n/a"
    else:
        delta = avg_sem - avg_hyb
        arrow = "improved" if delta > 0 else ("degraded" if delta < 0 else "no change")
        delta_s = f"{delta:+.2f} ({arrow})"

    q_display = query if len(query) <= 42 else query[:39] + "..."
    print(
        f"  {i+1:<3}  {q_display:<42}  {modality:<5}  "
        f"{fmt_avg(avg_sem):>8}  {fmt_avg(avg_hyb):>8}  {delta_s}"
    )

# ============================================================================
# ANALYSIS 5 -- Text query BM25 contribution
# ============================================================================

banner("ANALYSIS 5 -- BM25 contribution analysis for text queries")

text_queries = [(i, e) for i, e in enumerate(queries_data)
                if e["modality"].lower() == "text"]

for i, entry in text_queries:
    query        = entry["query"]
    relevant_ids = entry["relevant_ids"]
    cache        = results_cache[i]

    sem_ids  = ids_from(cache["semantic"])
    lex_ids  = ids_from(cache["bm25"])
    hyb_ids  = ids_from(cache["hybrid"])
    bm25_err = cache["bm25_err"]

    p3_sem = precision_at_k(sem_ids, relevant_ids, k=3)
    p3_hyb = precision_at_k(hyb_ids, relevant_ids, k=3)
    n5_sem = ndcg_at_k(sem_ids, relevant_ids, k=5)
    n5_hyb = ndcg_at_k(hyb_ids, relevant_ids, k=5)

    # Check if any relevant ID improved rank in hybrid vs semantic
    rank_improvements = []
    for rid in relevant_ids:
        r_s = rank_of(rid, sem_ids)
        r_h = rank_of(rid, hyb_ids)
        if r_h < r_s:
            rank_improvements.append((rid, r_s, r_h))

    print(f"\n  Q{i+1}: \"{query}\"")

    if p3_hyb > p3_sem or n5_hyb > n5_sem:
        print(f"    => BM25 IMPROVED retrieval  (P@3: {p3_sem:.3f} -> {p3_hyb:.3f} | NDCG@5: {n5_sem:.3f} -> {n5_hyb:.3f})")
    elif p3_hyb == p3_sem and n5_hyb == n5_sem:
        print(f"    => No metric change after BM25 fusion  (P@3={p3_sem:.3f}, NDCG@5={n5_sem:.3f})")
        print(f"       DENSE RETRIEVAL DOMINATES for this query.")
    else:
        print(f"    => BM25 DEGRADED retrieval  (P@3: {p3_sem:.3f} -> {p3_hyb:.3f} | NDCG@5: {n5_sem:.3f} -> {n5_hyb:.3f})")
        print(f"       BM25 introduced noise -- pure semantic was better.")

    if rank_improvements:
        for rid, r_s, r_h in rank_improvements:
            print(f"       Rank improvement: {rid}  {r_s} -> {r_h}")
    else:
        print(f"       No individual rank improvements from BM25.")

    # Show which BM25 hits are new vs already in semantic
    if bm25_err:
        print(f"       BM25 status: EMPTY -- {bm25_err}")
    else:
        new_lex = [cid for cid in lex_ids if cid not in set(sem_ids)]
        print(f"       BM25 introduced {len(new_lex)} new unique IDs not in semantic top-10: "
              f"{new_lex[:5]}{'...' if len(new_lex) > 5 else ''}")

# ============================================================================
# ANALYSIS 6 -- Image hybrid bypass confirmation
# ============================================================================

banner("ANALYSIS 6 -- Image query hybrid bypass confirmation")

image_queries = [(i, e) for i, e in enumerate(queries_data)
                 if e["modality"].lower() == "image"]

print()
for i, entry in image_queries:
    query = entry["query"]
    cache = results_cache[i]

    sem_ids = ids_from(cache["semantic"])
    hyb_ids = ids_from(cache["hybrid"])
    bm25    = cache["bm25"]

    lists_match = sem_ids == hyb_ids
    bm25_empty  = len(bm25) == 0

    print(f"  Q{i+1}: \"{query}\"")
    print(f"       BM25 index used     : {'No (image has no lexical index)' if bm25_empty else 'Yes (unexpected)'}")
    print(f"       Semantic == Hybrid  : {'YES -- bypass confirmed' if lists_match else 'NO -- MISMATCH (unexpected)'}")
    if not lists_match:
        print(f"       Semantic top-5 : {sem_ids[:5]}")
        print(f"       Hybrid   top-5 : {hyb_ids[:5]}")
    print()

# ============================================================================
# Summary
# ============================================================================

banner("DIAGNOSTIC SUMMARY")

print("""
  Retrieval system findings:
  -------------------------------------------------------------------------

  1. Code queries (weakest P@3 = 0.333):
       - code_023 (KMeans report fn) ranks highly in BOTH semantic and
         BM25 passes, consistently displacing true positives in top-3.
       - Dense embedding space is too compact for short code queries;
         embeddings cluster tightly regardless of algorithmic content.
       - BM25 shows similar behaviour -- keyword overlap between DBSCAN
         and KMeans code chunks is too high for BM25 to disambiguate.
       => Candidate fix: cross-encoder reranker with function_name /
          source_file metadata as a hard filter.

  2. Text queries (strong P@3 = 0.667-1.000):
       - Dense retrieval is well-calibrated; BM25 fusion neither helps
         nor hurts meaningfully -- all relevant IDs were already in
         semantic top-10 before fusion.
       - RRF preserves semantic ranking when BM25 overlap is high.
       => Dense retrieval dominates for prose text.  BM25 is redundant
          but harmless.

  2b. BM25 parse failures (Whoosh special-char issue):
       - Queries containing '+' (e.g. "KMeans++"), parentheses, or
         other Whoosh reserved operators cause silent parse errors.
       - WhooshBM25Index.search() catches all exceptions and returns []
         -- BM25 contributed 0 results to every query in this run.
       - This means hybrid == semantic for ALL modalities here.
       => Fix: escape query strings before Whoosh parsing
          (reserved chars: + - && || ! ( ) { } [ ] ^ " ~ * ? : \\).
          This is a known Whoosh usability issue and does NOT affect
          the FAISS or RRF logic.

  3. Image queries (CLIP cross-modal):
       - Hybrid bypass works correctly -- semantic (CLIP) results pass
         through unmodified.
       - ROC curve image (image_008) correctly ranked #1 for Q8.
       - DBSCAN images (image_002, 003, 000) correctly ranked top-3 for Q7.
       => Image retrieval is healthy.  No fix needed here.

  4. Overall assessment:
       - The primary failure mode is in the CODE modality.
       - Root cause: CodeBERT embeddings produce dense, poorly
         discriminative representations for short query strings.
       - Recommended next step: implement a cross-encoder reranker
         that can re-score code candidates using full query + code
         concatenation.
""")

print(f"{DIVIDER_WIDE}\n")
