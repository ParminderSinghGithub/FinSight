#!/usr/bin/env python3
"""
full_rag_system.py

End-to-end multimodal RAG flow:
Query -> Retrieval -> (Selective Rerank) -> Generation

Selective rerank rule:
- code: rerank semantic top-20 and keep top-5
- text: semantic top-5
- image: semantic top-5
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import faiss
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from embeddings.text_embedder import TextEmbedder
from embeddings.code_embedder import CodeEmbedder
from embeddings.image_embedder import ImageEmbedder
from rerank.cross_encoder_reranker import CrossEncoderReranker, prepare_candidates
from generation.rag_pipeline import RAGPipeline, prepare_text_chunks
from config.settings import get_path, get_retrieval_top_k, get_rerank_top_k


def load_indexes(index_dir: Path) -> tuple[dict[str, faiss.Index], dict[str, list[str]]]:
    """Load FAISS indexes and id maps for text/code/image modalities."""
    indexes: dict[str, faiss.Index] = {}
    idmaps: dict[str, list[str]] = {}

    for modality in ("text", "code", "image"):
        index_path = index_dir / f"faiss_{modality}.index"
        idmap_path = index_dir / f"faiss_{modality}_idmap.json"

        if not index_path.exists() or not idmap_path.exists():
            raise FileNotFoundError(
                f"Missing index assets for '{modality}': {index_path} / {idmap_path}"
            )

        indexes[modality] = faiss.read_index(str(index_path))
        with open(idmap_path, "r", encoding="utf-8") as f:
            idmaps[modality] = json.load(f)

    return indexes, idmaps


def load_chunk_stores(processed_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load text/code chunk stores used for reranking and generation."""
    text_path = processed_dir / "chunked_text.json"
    code_path = processed_dir / "chunked_code.json"

    if not text_path.exists() or not code_path.exists():
        raise FileNotFoundError(
            f"Missing chunk stores: {text_path} / {code_path}"
        )

    with open(text_path, "r", encoding="utf-8") as f:
        text_chunks = json.load(f)
    with open(code_path, "r", encoding="utf-8") as f:
        code_chunks = json.load(f)

    return {
        "text": text_chunks,
        "code": code_chunks,
    }


def load_evaluation_queries_optional(processed_dir: Path) -> list[dict[str, Any]] | None:
    """Optionally load evaluation queries when available."""
    eval_path = processed_dir / "evaluation_queries.json"
    if not eval_path.exists():
        return None

    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


class FullRAGSystem:
    """Full Query -> Retrieval -> Selective Rerank -> Generation pipeline."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent
        self.index_dir = self.project_root / Path(get_path("indexes"))
        self.processed_dir = self.project_root / Path(get_path("data")) / "processed"

        # 1) Load assets
        self.indexes, self.idmaps = load_indexes(self.index_dir)
        self.chunk_stores = load_chunk_stores(self.processed_dir)
        self.evaluation_queries = load_evaluation_queries_optional(self.processed_dir)

        # Helpful merged store for generation helper
        self.combined_store = self.chunk_stores["text"] + self.chunk_stores["code"]

        # 2) Instantiate models
        self.text_embedder = TextEmbedder()
        self.code_embedder = CodeEmbedder()
        self.image_embedder = ImageEmbedder()

        self.reranker = CrossEncoderReranker()
        self.generator = RAGPipeline()

    def _embed_query(self, query: str, modality: str) -> np.ndarray:
        """Embed query with the modality-specific encoder."""
        if modality == "text":
            vec = self.text_embedder.encode_query(query)
        elif modality == "code":
            vec = self.code_embedder.encode([query])[0]
        elif modality == "image":
            vec = self.image_embedder.encode_text_query(query)
        else:
            raise ValueError("modality must be one of: text, code, image")

        if vec.ndim == 1:
            return np.array([vec], dtype=np.float32)
        return vec.astype(np.float32)

    def _retrieve_semantic(self, query: str, modality: str, top_k: int | None = None) -> list[str]:
        """Retrieve top-k semantic chunk IDs from FAISS."""
        if top_k is None:
            top_k = get_retrieval_top_k()
        query_vec = self._embed_query(query, modality)
        _, indices = self.indexes[modality].search(query_vec, k=top_k)
        return [self.idmaps[modality][idx] for idx in indices[0]]

    @staticmethod
    def _dedupe_keep_order(chunk_ids: list[str]) -> list[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        deduped = []
        for chunk_id in chunk_ids:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            deduped.append(chunk_id)
        return deduped

    def _select_for_generation(
        self,
        query: str,
        modality: str,
        semantic_ids: list[str],
        supporting_text_ids: list[str] | None = None,
    ) -> list[str]:
        """
        Apply selective reranking policy and return final top-5 chunk IDs.

        Policy:
        - text: pass only text chunks (top-5 semantic)
        - code: rerank code top-20, then add top supporting text chunks
        - image: semantic top-5
        """
        if modality == "code":
            candidates = prepare_candidates(semantic_ids[:20], self.chunk_stores["code"])
            reranked = self.reranker.rerank(
                query,
                candidates,
                top_k=get_rerank_top_k(),
            )
            code_ids = [chunk_id for chunk_id, _ in reranked]

            text_support_limit = max(0, get_retrieval_top_k() - len(code_ids[:get_retrieval_top_k()]))
            text_support = (supporting_text_ids or [])[:text_support_limit]
            combined = self._dedupe_keep_order(code_ids + text_support)
            return combined[:get_retrieval_top_k()]

        if modality == "text":
            return semantic_ids[:get_retrieval_top_k()]

        return semantic_ids[:get_retrieval_top_k()]

    def _prepare_generation_chunks(self, chunk_ids: list[str], modality: str) -> list[dict[str, str]]:
        """
        Convert chunk IDs into generation chunks using prepare_text_chunks.

        For image modality, build lightweight textual placeholders because there is
        no dedicated image text chunk store.
        """
        if modality == "text":
            return prepare_text_chunks(chunk_ids, self.chunk_stores["text"])

        if modality == "code":
            return prepare_text_chunks(chunk_ids, self.combined_store)

        image_store = [{"chunk_id": cid, "text": f"Image result: {cid}"} for cid in chunk_ids]
        return prepare_text_chunks(chunk_ids, image_store)

    def run_query(self, query: str, modality: str, use_gemini: bool = False) -> dict[str, Any]:
        """
        Run one full query through the RAG system.

        Steps:
        1) Semantic top-20 retrieval
        2) Selective rerank (code only)
        3) Prepare chunks
        4) Generate answer
        5) Return query + answer + sources
        """
        retrieval_top_k = get_retrieval_top_k()
        rerank_top_k = get_rerank_top_k()

        semantic_k = rerank_top_k if modality == "code" else retrieval_top_k
        semantic_ids = self._retrieve_semantic(query, modality, top_k=semantic_k)

        supporting_text_ids = None
        if modality == "code":
            supporting_text_ids = self._retrieve_semantic(query, "text", top_k=retrieval_top_k)

        final_ids = self._select_for_generation(
            query,
            modality,
            semantic_ids,
            supporting_text_ids=supporting_text_ids,
        )
        generation_limit = min(retrieval_top_k, 5)
        generation_chunks = self._prepare_generation_chunks(final_ids[:generation_limit], modality)

        generation_result = self.generator.generate_answer(
            query,
            generation_chunks,
            use_gemini=use_gemini,
        )

        return {
            "query": query,
            "answer": generation_result["answer"],
            "sources": generation_result["used_chunks"],
        }


def print_result_block(result: dict[str, Any], idx: int) -> None:
    """Print readable CLI output for one query result."""
    print(f"\nResult {idx}")
    print("Query:")
    print(result["query"])
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    print(", ".join(result["sources"]))


if __name__ == "__main__":
    system = FullRAGSystem()

    queries = [
        ("Explain DBSCAN eps parameter", "text"),
        ("Example of DBSCAN eps tuning in Python", "code"),
    ]

    for i, (query, modality) in enumerate(queries, start=1):
        result = system.run_query(query, modality)
        print_result_block(result, i)
