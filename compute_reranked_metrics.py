#!/usr/bin/env python3
"""
compute_reranked_metrics.py

Evaluates retrieval performance with cross-encoder reranking.

Pipeline:
1. Semantic retrieval (FAISS top-20)
2. Cross-encoder reranking (code only; text/image stay semantic)
3. Take top-10 reranked results
4. Compute metrics (P@3, R@5, NDCG@5, MRR) vs ground truth

Compares against evaluation_queries.json with ground-truth relevant_ids.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import faiss

warnings.filterwarnings("ignore", category=FutureWarning)

from embeddings.text_embedder import TextEmbedder
from embeddings.code_embedder import CodeEmbedder
from embeddings.image_embedder import ImageEmbedder
from rerank.cross_encoder_reranker import CrossEncoderReranker, prepare_candidates
from config.settings import get_path, get_rerank_top_k, get_retrieval_top_k
from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
)


# ============================================================================
# Load evaluation data
# ============================================================================

def load_evaluation_queries():
    """Load evaluation queries with ground-truth relevant_ids."""
    query_file = Path(get_path("data")) / "processed" / "evaluation_queries.json"
    if not query_file.exists():
        raise FileNotFoundError(f"Evaluation queries not found: {query_file}")

    with open(query_file) as f:
        queries = json.load(f)

    print(f"Loaded evaluation queries: {len(queries)}")
    return queries


def load_indexes():
    """Load all three FAISS indexes and ID mappings."""
    indexes = {}
    idmaps = {}

    for modality in ["text", "code", "image"]:
        index_path = Path(get_path("indexes")) / f"faiss_{modality}.index"
        idmap_path = Path(get_path("indexes")) / f"faiss_{modality}_idmap.json"

        if not index_path.exists() or not idmap_path.exists():
            raise FileNotFoundError(
                f"FAISS {modality} index not found: {index_path}"
            )

        index = faiss.read_index(str(index_path))
        with open(idmap_path) as f:
            idmap = json.load(f)

        indexes[modality] = index
        idmaps[modality] = idmap
        print(f"Loaded FAISS {modality} index vectors: {index.ntotal}")

    return indexes, idmaps


def load_chunk_stores():
    """Load text and code chunk stores."""
    stores = {}

    for modality, filename in [("text", "chunked_text.json"), ("code", "chunked_code.json")]:
        chunk_file = Path(get_path("data")) / "processed" / filename
        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

        with open(chunk_file) as f:
            chunks = json.load(f)

        stores[modality] = chunks
        print(f"Loaded {modality} chunks: {len(chunks)}")

    return stores


# ============================================================================
# Semantic retrieval
# ============================================================================

def retrieve_semantic(
    query: str,
    modality: str,
    embedder,
    index,
    idmap,
    top_k: int = 20,
) -> list[str]:
    """
    Retrieve top-k chunk IDs using semantic search.

    Args:
        query: Query string
            modality: "text", "code", or "image"
        embedder: Text/Code/ImageEmbedder instance
        index: FAISS index
        idmap: List of chunk IDs
        top_k: Number of results to retrieve

    Returns:
        List of chunk IDs, sorted by similarity (descending)
    """
    # Embed query
    # Determine embedder type and use appropriate method
    if modality == "image":
        query_emb = embedder.encode_text_query(query)
    elif modality == "text":
        query_emb = embedder.encode_query(query)
    elif modality == "code":
        query_emb = embedder.encode([query])[0]
    else:
        raise ValueError(f"Unknown modality: {modality}")


    # Ensure 2D shape (1, embedding_dim)
    if query_emb.ndim == 1:
        query_emb = np.array([query_emb], dtype=np.float32)
    else:
        query_emb = query_emb.astype(np.float32)
    # Search
    distances, indices = index.search(query_emb, k=top_k)

    # Map indices to chunk IDs
    chunk_ids = [idmap[idx] for idx in indices[0]]
    return chunk_ids


# ============================================================================
# Reranking (for code only)
# ============================================================================

def rerank_and_select(
    query: str,
    chunk_ids: list[str],
    chunk_store: list[dict],
    reranker: CrossEncoderReranker,
    top_k: int = 10,
) -> list[str]:
    """
    Rerank candidates and return top_k.

    Args:
        query: Query string
        chunk_ids: List of chunk IDs from semantic retrieval
        chunk_store: Loaded JSON chunk store
        reranker: CrossEncoderReranker instance
        top_k: Number of reranked results to return

    Returns:
        List of chunk IDs reranked and truncated to top_k
    """
    # Prepare candidates
    candidates = prepare_candidates(chunk_ids, chunk_store)

    # Rerank
    reranked = reranker.rerank(query, candidates, top_k=top_k)

    # Extract chunk IDs
    return [chunk_id for chunk_id, _ in reranked]


# ============================================================================
# Evaluation loop
# ============================================================================

def evaluate_reranked_retrieval(
    queries: list[dict],
    indexes: dict,
    idmaps: dict,
    chunk_stores: dict,
    embedders: dict,
    reranker: CrossEncoderReranker,
) -> dict:
    """
    Evaluate retrieval with reranking.

    Returns:
        Dictionary with per-query and global metrics.
    """
    metrics_per_query = {}
    all_p3 = []
    all_r5 = []
    all_ndcg5 = []
    all_retrieved_for_mrr = []
    all_relevant_for_mrr = []

    print("\nEvaluation: Semantic Retrieval + Reranking")

    for i, query_obj in enumerate(queries, start=1):
        query_id = f"Q{i}"
        query_text = query_obj["query"]
        modality = query_obj["modality"]
        relevant_ids = query_obj["relevant_ids"]

        print(f"\nQuery ID: {query_id}")
        print(f"Modality: {modality}")
        print(f"Query: {query_text}")

        # ── Step 1: Semantic retrieval (top 20) ─────────────────────────────

        embedder = embedders[modality]
        index = indexes[modality]
        idmap = idmaps[modality]

        semantic_results = retrieve_semantic(
            query_text,
            modality,
            embedder,
            index,
            idmap,
            top_k=get_rerank_top_k(),
        )

        # ── Step 2-4: Selective reranking by modality ────────────────────────

        if modality == "code":
            chunk_store = chunk_stores["code"]
            final_results = rerank_and_select(
                query_text,
                semantic_results,
                chunk_store,
                reranker,
                top_k=get_rerank_top_k(),
            )
            final_results = final_results[:get_retrieval_top_k()]
            print("Selection mode: code reranked top-10 from semantic top-20")
        elif modality == "text":
            final_results = semantic_results[:get_retrieval_top_k()]
            print("Selection mode: text semantic top-10")
        else:
            final_results = semantic_results[:get_retrieval_top_k()]
            print("Selection mode: image semantic top-10")

        # ── Step 5: Compute metrics ───────────────────────────────────────

        p3 = precision_at_k(final_results, relevant_ids, k=3)
        r5 = recall_at_k(final_results, relevant_ids, k=5)
        ndcg5 = ndcg_at_k(final_results, relevant_ids, k=5)

        # Store
        metrics_per_query[query_id] = {
            "p@3":  p3,
            "r@5":  r5,
            "ndcg@5": ndcg5,
            "retrieved": final_results[:3],  # First 3 for inspection
        }

        all_p3.append(p3)
        all_r5.append(r5)
        all_ndcg5.append(ndcg5)

        all_retrieved_for_mrr.append(final_results)
        all_relevant_for_mrr.append(relevant_ids)
        # Print
        print(f"Precision@3: {p3:.4f}")
        print(f"Recall@5: {r5:.4f}")
        print(f"NDCG@5: {ndcg5:.4f}")
        print(f"MRR: {mean_reciprocal_rank([final_results], [relevant_ids]):.4f}")

    # ── Global metrics ────────────────────────────────────────────────────

    print("\nGlobal metrics")

    mean_p3 = np.mean(all_p3)
    mean_r5 = np.mean(all_r5)
    mean_ndcg5 = np.mean(all_ndcg5)
    mean_mrr = mean_reciprocal_rank(all_retrieved_for_mrr, all_relevant_for_mrr)

    print(f"Mean Precision@3: {mean_p3:.4f}")
    print(f"Mean Recall@5: {mean_r5:.4f}")
    print(f"Mean NDCG@5: {mean_ndcg5:.4f}")
    print(f"Mean MRR: {mean_mrr:.4f}")

    print("\nPer-query results")

    for query_id, metrics in metrics_per_query.items():
        print(f"\nQuery ID: {query_id}")
        print(f"Precision@3: {metrics['p@3']:.4f}")
        print(f"Recall@5: {metrics['r@5']:.4f}")
        print(f"NDCG@5: {metrics['ndcg@5']:.4f}")
        print(f"Top-3 sources: {', '.join(metrics['retrieved'])}")

    return {
        "global": {
            "mean_p@3": mean_p3,
            "mean_r@5": mean_r5,
            "mean_ndcg@5": mean_ndcg5,
            "mean_mrr": mean_mrr,
        },
        "per_query": metrics_per_query,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nCompute Reranked Retrieval Metrics")

    # Load data
    print("\nSetup: loading evaluation data")
    queries = load_evaluation_queries()
    indexes, idmaps = load_indexes()
    chunk_stores = load_chunk_stores()

    # Instantiate models
    print("\nSetup: instantiating embedders and reranker")
    embedders = {
        "text": TextEmbedder(),
        "code": CodeEmbedder(),
        "image": ImageEmbedder(),
    }
    reranker = CrossEncoderReranker()

    # Evaluate
    results = evaluate_reranked_retrieval(
        queries,
        indexes,
        idmaps,
        chunk_stores,
        embedders,
        reranker,
    )

    print("\nEvaluation complete")
