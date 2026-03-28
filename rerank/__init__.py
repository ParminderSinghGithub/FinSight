"""
rerank — Reranking module for phase 7 of the multimodal RAG pipeline.
"""

from rerank.cross_encoder_reranker import CrossEncoderReranker, prepare_candidates

__all__ = [
    "CrossEncoderReranker",
    "prepare_candidates",
]
