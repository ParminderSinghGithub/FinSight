#!/usr/bin/env python3
"""
test_reranker_effect.py

Qualitative demonstration of cross-encoder reranking effect on code retrieval.

Compares:
- Semantic retrieval only (FAISS top-10)
- Reranked results (top-5 after cross-encoder scoring)

Uses two code queries to show how reranking can improve relevance ranking.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from embeddings.code_embedder import CodeEmbedder
from rerank.cross_encoder_reranker import CrossEncoderReranker, prepare_candidates


# ============================================================================
# Load indexes and data
# ============================================================================

def load_code_index():
    """Load FAISS code index and ID mapping."""
    index_path = Path("indexes/faiss_code.index")
    idmap_path = Path("indexes/faiss_code_idmap.json")

    if not index_path.exists() or not idmap_path.exists():
        raise FileNotFoundError(
            f"FAISS code index not found. Expected:\n"
            f"  {index_path}\n  {idmap_path}"
        )

    # Load FAISS index
    index = faiss.read_index(str(index_path))

    # Load ID mapping (JSON list of chunk IDs)
    with open(idmap_path) as f:
        idmap = json.load(f)

    print(f"✓ Loaded FAISS code index: {index.ntotal} vectors")
    print(f"✓ Loaded ID mapping: {len(idmap)} IDs")

    return index, idmap


def load_code_chunks():
    """Load chunked code from JSON."""
    chunk_file = Path("data/processed/chunked_code.json")

    if not chunk_file.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

    with open(chunk_file) as f:
        chunks = json.load(f)

    print(f"✓ Loaded code chunks: {len(chunks)} chunks")
    return chunks


# ============================================================================
# Retrieval and reranking
# ============================================================================

def retrieve_semantic(query: str, embedder: CodeEmbedder, index, idmap, top_k: int = 10):
    """
    Retrieve top-k results using semantic similarity (FAISS).

    Returns:
        list of (rank, chunk_id) tuples (1-indexed rank)
    """
    # Embed query
    query_emb = embedder.encode([query])[0]  # Shape: (768,)

    # Normalize to unit length for cosine similarity (FAISS uses inner product)
    query_emb = query_emb / np.linalg.norm(query_emb)
    query_emb = np.array([query_emb], dtype=np.float32)

    # Search
    distances, indices = index.search(query_emb, k=top_k)

    # Convert distances to similarities (inner product for normalized vectors ≈ cosine)
    similarities = distances[0]

    # Map FAISS indices to chunk IDs
    results = []
    for rank, (faiss_idx, sim) in enumerate(zip(indices[0], similarities), start=1):
        chunk_id = idmap[faiss_idx]
        results.append((rank, chunk_id, sim))

    return results


def compare_retrieval(query: str, embedder: CodeEmbedder, reranker: CrossEncoderReranker,
                     index, idmap, chunks):
    """
    Compare semantic and reranked results for a query.

    Prints:
    - Query
    - Semantic top-10
    - Reranked top-5
    - Movement analysis
    """
    print("\n" + "=" * 80)
    print(f"Query: \"{query}\"")
    print("=" * 80)

    # ── Step A: Semantic Retrieval ────────────────────────────────────────

    print("\n[STEP A] Semantic Retrieval (FAISS Top-10)")
    print("-" * 80)

    semantic_results = retrieve_semantic(query, embedder, index, idmap, top_k=10)

    print(f"{'Rank':<5} {'Chunk ID':<12} {'Similarity':<12} Preview")
    print("-" * 80)

    for rank, chunk_id, sim in semantic_results:
        # Find chunk text
        chunk_text = next(
            (c["code"] for c in chunks if c["chunk_id"] == chunk_id),
            "[NOT FOUND]"
        )
        preview = chunk_text[:50].replace("\n", " ")
        print(f"{rank:<5} {chunk_id:<12} {sim:<12.4f} {preview}...")

    # ── Step B-C: Reranking ───────────────────────────────────────────────

    print("\n[STEP B-C] Prepare Candidates and Rerank")
    print("-" * 80)

    # Extract chunk IDs from semantic results
    semantic_chunk_ids = [chunk_id for _, chunk_id, _ in semantic_results]

    # Prepare candidates
    candidates = prepare_candidates(semantic_chunk_ids, chunks)

    # Rerank
    reranked_results = reranker.rerank(query, candidates, top_k=5)

    print(f"✓ Reranked: {len(reranked_results)} results")

    # ── Step D: Comparison ────────────────────────────────────────────────

    print("\n[STEP D] Before vs After Comparison")
    print("-" * 80)

    # Before
    print(f"\n{'Before (Semantic Top-10):':<30}")
    print(f"{'Rank':<5} {'Chunk ID':<12} Preview")
    print("-" * 40)
    for rank, chunk_id, _ in semantic_results[:10]:
        chunk_text = next(
            (c["code"] for c in chunks if c["chunk_id"] == chunk_id),
            "[NOT FOUND]"
        )
        preview = chunk_text[:40].replace("\n", " ")
        print(f"{rank:<5} {chunk_id:<12} {preview}...")

    # After
    print(f"\n{'After (Reranked Top-5):':<30}")
    print(f"{'Rank':<5} {'Chunk ID':<12} {'Score':<10} Preview")
    print("-" * 50)
    for rank, (chunk_id, score) in enumerate(reranked_results, start=1):
        chunk_text = next(
            (c["code"] for c in chunks if c["chunk_id"] == chunk_id),
            "[NOT FOUND]"
        )
        preview = chunk_text[:40].replace("\n", " ")
        print(f"{rank:<5} {chunk_id:<12} {score:<10.4f} {preview}...")

    # ── Movement Analysis ─────────────────────────────────────────────────

    print(f"\n{'Movement Analysis:'}")
    print("-" * 80)

    # Build rank mapping: chunk_id -> semantic rank
    semantic_rank_map = {chunk_id: rank for rank, chunk_id, _ in semantic_results}

    for new_rank, (chunk_id, score) in enumerate(reranked_results, start=1):
        old_rank = semantic_rank_map.get(chunk_id, None)
        if old_rank is None:
            print(f"  {chunk_id}: NEW (not in top-10)")
        elif old_rank == new_rank:
            print(f"  {chunk_id}: STABLE (rank {old_rank} → {new_rank})")
        elif old_rank < new_rank:
            print(f"  {chunk_id}: MOVED DOWN (rank {old_rank} → {new_rank})")
        else:
            print(f"  {chunk_id}: MOVED UP ↑ (rank {old_rank} → {new_rank})")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Test: Cross-Encoder Reranking Effect on Code Retrieval")
    print("=" * 80)

    # Initialize
    print("\n[SETUP] Loading data and models...")
    index, idmap = load_code_index()
    chunks = load_code_chunks()

    print("\n[SETUP] Instantiating embedder and reranker...")
    embedder = CodeEmbedder()
    reranker = CrossEncoderReranker()

    # Test queries (code-specific)
    queries = [
        "Example of DBSCAN eps parameter tuning in Python",
        "How to compute ROC curve using sklearn",
    ]

    # Run comparisons
    for query in queries:
        compare_retrieval(query, embedder, reranker, index, idmap, chunks)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
