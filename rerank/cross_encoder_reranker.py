"""
rerank/cross_encoder_reranker.py

Phase 7 — Cross-Encoder Reranking module.

Uses sentence-transformers CrossEncoder to rerank retrieval candidates by
computing query-candidate relevance scores. Useful for improving precision
after initial semantic or hybrid retrieval.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fast, lightweight cross-encoder trained on MS MARCO dataset
  - ~6 layers, ~22M parameters
  - ~50 ms latency per pair on CPU
  - Scores in range [0, 1] representing relevance probability

Typical usage:
    from rerank.cross_encoder_reranker import CrossEncoderReranker, prepare_candidates
    import json

    # Load chunk store
    text_chunks = json.load(open("data_sample/processed/chunked_text.json"))

    # Initial retrieval (e.g. from FAISS)
    initial_ids = ["text_016", "text_027", "text_026", ...]

    # Prepare candidates
    candidates = prepare_candidates(initial_ids, text_chunks)

    # Rerank
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("KMeans++ initialization", candidates, top_k=3)

    for chunk_id, score in reranked:
        print(f"{chunk_id}: {score:.4f}")
"""

import torch
from typing import Optional

from sentence_transformers import CrossEncoder

from config.settings import get_device, get_model


# ============================================================================
# CrossEncoderReranker
# ============================================================================

class CrossEncoderReranker:
    """
    Reranks retrieval candidates using a cross-encoder model.

    A cross-encoder takes a (query, document) pair and directly predicts
    the relevance score between 0 and 1, unlike bi-encoders which embed
    query and document separately and compute similarity.

    Cross-encoders are more accurate but slower; ideal for reranking a
    small set of initial retrieval candidates (top-50 to top-200).

    Attributes:
        model_name (str):  HuggingFace model identifier or path
        device (str):      "cuda" or "cpu"
        model (CrossEncoder): Loaded cross-encoder model
    """

    def __init__(
        self,
        model_name: str = get_model("reranker"),
        device: Optional[str] = None,
    ) -> None:
        """
        Load a cross-encoder model.

        Args:
            model_name: HuggingFace model identifier. Defaults to MiniLM-L-6-v2,
                        a fast model suitable for CPU inference in reranking.
            device:     Torch device string ("cuda" or "cpu").  If None, auto-detect.
        """
        if device is None:
            device = get_device()

        self.model_name = model_name
        self.device = device
        self.model = CrossEncoder(model_name, device=device)

        print(
            f"[CrossEncoderReranker] Loaded model '{model_name}' "
            f"on {device.upper()}"
        )

    # ── Reranking API ─────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        batch_size: int = 16,
    ) -> list[tuple[str, float]]:
        """
        Rerank retrieval candidates by relevance to the query.

        Args:
            query:       User query string.
            candidates:  List of candidate dicts with keys:
                         - "chunk_id" (str): unique identifier
                         - "text" (str):     chunk content (text or code)
            top_k:       Number of top candidates to return.
            batch_size:  Batch size for model inference.  Larger batches
                         are faster but use more memory.

        Returns:
            List of (chunk_id, relevance_score) tuples, sorted by score
            (descending). Length ≤ top_k. Scores are floats in [0, 1].

        Raises:
            ValueError: If candidates list is empty or missing required keys.
        """
        if not candidates:
            return []

        # Validate candidates format
        for cand in candidates:
            if "chunk_id" not in cand or "text" not in cand:
                raise ValueError(
                    "Each candidate must have 'chunk_id' and 'text' keys."
                )

        # Create query-document pairs
        pairs = [
            [query, cand["text"]]
            for cand in candidates
        ]

        # Score pairs in batches
        scores = self.model.predict(pairs, batch_size=batch_size)

        # Attach scores to candidates
        scored_candidates = [
            (cand["chunk_id"], float(score))
            for cand, score in zip(candidates, scores)
        ]

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        return scored_candidates[:top_k]

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model_name='{self.model_name}', "
            f"device='{self.device}')"
        )


# ============================================================================
# Candidate preparation helper
# ============================================================================

def prepare_candidates(
    chunk_ids: list[str],
    chunk_store: list[dict],
) -> list[dict]:
    """
    Build a list of candidate dicts from chunk IDs and a chunk store JSON.

    The chunk_store is expected to be a JSON-loaded list where each element
    has at least a "chunk_id" key and a "text" or "code" key containing the
    actual chunk content.

    Args:
        chunk_ids:   List of chunk IDs to retrieve from the store.
        chunk_store: Loaded JSON list of chunk records (e.g. from
                     configured data directory processed/chunked_text.json).

    Returns:
        List of candidate dicts ready for reranking, in input order.
        Each dict has "chunk_id" and "text" keys.

    Raises:
        KeyError: If a requested chunk_id is not found in the store.
    """
    # Build a lookup map for O(1) access
    store_map = {chunk["chunk_id"]: chunk for chunk in chunk_store}

    candidates = []
    for chunk_id in chunk_ids:
        if chunk_id not in store_map:
            raise KeyError(f"Chunk ID '{chunk_id}' not found in chunk store.")

        chunk = store_map[chunk_id]

        # Determine which field contains the text (usual case: "text" or "code")
        text_content = chunk.get("text") or chunk.get("code", "")

        candidates.append({
            "chunk_id": chunk_id,
            "text":     text_content,
        })

    return candidates
