"""
retrieval/hybrid_search.py

Phase 6 — Hybrid Retrieval pipeline combining:
  • Semantic search  — FAISS IndexFlatIP with L2-normalised embeddings
  • Lexical search   — BM25 via Whoosh full-text indexes
  • Fusion           — Reciprocal Rank Fusion (RRF)

Architecture
------------
WhooshBM25Index
    Builds, persists, and queries Whoosh indexes for a single modality
    (text or code).  Images are not indexed lexically.

reciprocal_rank_fusion(semantic_results, bm25_results, k=60)
    Pure function.  Fuses two ranked lists into a single score.

HybridSearchEngine
    Convenience class that bundles all FAISS indexes, embedders, and
    Whoosh indexes into a single `search()` entry-point.

    For image queries, BM25 is skipped and the semantic result is
    returned directly (CLIP already performs cross-modal matching).

Usage example
-------------
    from retrieval.hybrid_search import HybridSearchEngine

    engine = HybridSearchEngine.from_disk()     # loads everything
    results = engine.search(
        "DBSCAN eps parameter tuning", modality="code", top_k=10
    )
    for rank, (chunk_id, score) in enumerate(results, 1):
        print(rank, chunk_id, f"{score:.4f}")
"""

import json
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Whoosh imports
# ---------------------------------------------------------------------------
import re as _re

from whoosh import index as whoosh_index
from whoosh import query as whoosh_query
from whoosh.fields import ID, TEXT, Schema
from whoosh.qparser import MultifieldParser, OrGroup, QueryParser
from whoosh.writing import AsyncWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.text_embedder  import TextEmbedder   # noqa: E402
from embeddings.code_embedder  import CodeEmbedder   # noqa: E402
from embeddings.image_embedder import ImageEmbedder  # noqa: E402
from indexing.faiss_index      import FaissIndex     # noqa: E402

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
INDEX_DIR            = PROJECT_ROOT / "indexes"
WHOOSH_TEXT_DIR      = INDEX_DIR / "whoosh_text"
WHOOSH_CODE_DIR      = INDEX_DIR / "whoosh_code"
CHUNKED_TEXT_FILE    = PROJECT_ROOT / "data" / "processed" / "chunked_text.json"
CHUNKED_CODE_FILE    = PROJECT_ROOT / "data" / "processed" / "chunked_code.json"
FAISS_TEXT_INDEX     = INDEX_DIR / "faiss_text.index"
FAISS_TEXT_IDMAP     = INDEX_DIR / "faiss_text_idmap.json"
FAISS_CODE_INDEX     = INDEX_DIR / "faiss_code.index"
FAISS_CODE_IDMAP     = INDEX_DIR / "faiss_code_idmap.json"
FAISS_IMAGE_INDEX    = INDEX_DIR / "faiss_image.index"
FAISS_IMAGE_IDMAP    = INDEX_DIR / "faiss_image_idmap.json"


# ============================================================================
# WhooshBM25Index
# ============================================================================

class WhooshBM25Index:
    """
    Thin wrapper around a Whoosh on-disk index for BM25 lexical search.

    Supports two content fields:
      • ``"text"``  — full prose text (body field name: ``content``)
      • ``"code"``  — source code  (body field name: ``content``)

    Both expose the same public API so callers don't need to distinguish
    between them.

    Schema (both modalities)
    ------------------------
    chunk_id  : ID (stored, not analysed)
    content   : TEXT (stored, analysed with Whoosh StemmingAnalyzer)
    aux       : TEXT (stored) — section name for text, function name for code
    """

    # Whoosh schema shared by both modalities
    _SCHEMA = Schema(
        chunk_id = ID(stored=True, unique=True),
        content  = TEXT(stored=True),
        aux      = TEXT(stored=True),
    )

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self._ix: Optional[whoosh_index.FileIndex] = None

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self, records: list[dict], content_field: str, aux_field: str) -> None:
        """
        (Re-)create the Whoosh index from a list of chunk records.

        Args:
            records:       List of chunk dicts (from chunked_text.json or
                           chunked_code.json).
            content_field: Key in each record that holds the body text,
                           e.g. ``"text"`` or ``"code"``.
            aux_field:     Key for the auxiliary metadata field,
                           e.g. ``"section"`` or ``"function_name"``.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)

        ix = whoosh_index.create_in(str(self.index_dir), self._SCHEMA)
        writer = ix.writer()

        for record in records:
            writer.add_document(
                chunk_id = record["chunk_id"],
                content  = record.get(content_field, ""),
                aux      = record.get(aux_field, ""),
            )

        writer.commit()
        self._ix = ix
        print(
            f"[WhooshBM25Index] Built index with {len(records)} document(s) "
            f"→ {self.index_dir}"
        )

    # ── Load ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Open an existing on-disk Whoosh index."""
        if not whoosh_index.exists_in(str(self.index_dir)):
            raise FileNotFoundError(
                f"No Whoosh index found at {self.index_dir}. "
                "Call build() first."
            )
        self._ix = whoosh_index.open_dir(str(self.index_dir))
        print(
            f"[WhooshBM25Index] Loaded index ({self._ix.doc_count()} docs) "
            f"from {self.index_dir}"
        )

    # ── Search ────────────────────────────────────────────────────────────

    def search(self, query_str: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Run a BM25 query against the index.

        The query is searched across both ``content`` and ``aux`` fields
        using OR logic so that section headings / function names also
        contribute to matching.

        Args:
            query_str: Free-text query string.
            top_k:     Maximum number of results to return.

        Returns:
            List of ``(chunk_id, bm25_score)`` ordered by descending score.
            Returns an empty list on parse errors or empty index.
        """
        if self._ix is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        # Escape Whoosh reserved characters so that tokens like "KMeans++",
        # "C++", or operator-like punctuation are treated as literals rather
        # than query syntax.  Reserved set (Whoosh / Lucene convention):
        #   + - && || ! ( ) { } [ ] ^ " ~ * ? : \
        escaped = _re.sub(r'([+\-&|!(){}\[\]^"~*?:\\])', r'\\\1', query_str)

        results: list[tuple[str, float]] = []
        try:
            with self._ix.searcher() as searcher:
                # OrGroup.factory(0.9) scores documents that match more terms
                # higher but still returns partial matches (OR semantics).
                parser = QueryParser(
                    "content",
                    schema=self._ix.schema,
                    termclass=whoosh_query.Term,
                    group=OrGroup.factory(0.9),
                )
                # Also search the aux field by combining two parsed queries
                q_content = parser.parse(escaped)
                aux_parser = QueryParser(
                    "aux",
                    schema=self._ix.schema,
                    termclass=whoosh_query.Term,
                    group=OrGroup.factory(0.9),
                )
                q_aux = aux_parser.parse(escaped)
                q = whoosh_query.Or([q_content, q_aux])

                hits = searcher.search(q, limit=top_k)
                for hit in hits:
                    results.append((hit["chunk_id"], hit.score))
        except Exception as exc:
            print(
                f"[WhooshBM25Index] WARNING: query parse failed for "
                f"'{query_str}' (escaped: '{escaped}'): {exc}"
            )

        return results


# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================

def reciprocal_rank_fusion(
    semantic_results: list[tuple[str, float]],
    bm25_results:     list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion (RRF).

    For each document d in the union of both lists:

        RRF_score(d) = Σ_{list L}  1 / (k + rank_L(d))

    where ``rank_L(d)`` is the 1-based position of d in list L, and k is a
    smoothing constant (default 60, as in the original paper by Cormack et al.).
    Documents that appear in only one list receive a contribution only from
    that list.

    Args:
        semantic_results: Ranked list of ``(chunk_id, score)`` from FAISS.
        bm25_results:     Ranked list of ``(chunk_id, score)`` from Whoosh.
        k:                Smoothing constant.  Higher k reduces the impact of
                          high-rank differences.

    Returns:
        Merged list of ``(chunk_id, rrf_score)`` sorted by descending RRF score.
        Items from both lists are included (union); items present in only one
        list still receive their reciprocal rank contribution.
    """
    rrf_scores: dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(semantic_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================================
# HybridSearchEngine
# ============================================================================

class HybridSearchEngine:
    """
    Bundles FAISS semantic indexes, embedding models, and Whoosh BM25 indexes
    into a single retrieval interface.

    For text and code modalities, results from both systems are fused via RRF.
    For image modality, BM25 is skipped (lexical search over image embeddings
    is not meaningful); the CLIP semantic results are returned directly.

    Parameters
    ----------
    text_faiss   : FaissIndex for text chunks (dim=1024)
    code_faiss   : FaissIndex for code chunks (dim=768)
    image_faiss  : FaissIndex for image vectors (dim=512)
    text_embedder  : TextEmbedder
    code_embedder  : CodeEmbedder
    image_embedder : ImageEmbedder
    text_bm25    : WhooshBM25Index for text chunks
    code_bm25    : WhooshBM25Index for code chunks
    rrf_k        : RRF smoothing constant (default 60)
    """

    def __init__(
        self,
        text_faiss:     FaissIndex,
        code_faiss:     FaissIndex,
        image_faiss:    FaissIndex,
        text_embedder:  TextEmbedder,
        code_embedder:  CodeEmbedder,
        image_embedder: ImageEmbedder,
        text_bm25:      WhooshBM25Index,
        code_bm25:      WhooshBM25Index,
        rrf_k:          int = 60,
    ) -> None:
        self._text_faiss    = text_faiss
        self._code_faiss    = code_faiss
        self._image_faiss   = image_faiss
        self._text_emb      = text_embedder
        self._code_emb      = code_embedder
        self._image_emb     = image_embedder
        self._text_bm25     = text_bm25
        self._code_bm25     = code_bm25
        self._rrf_k         = rrf_k

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_disk(
        cls,
        *,
        rrf_k: int = 60,
        rebuild_bm25: bool = False,
    ) -> "HybridSearchEngine":
        """
        Instantiate a fully loaded HybridSearchEngine from default disk paths.

        Args:
            rrf_k:        RRF smoothing constant.
            rebuild_bm25: If True, rebuild Whoosh indexes from the chunked
                          JSON files even if they already exist on disk.

        Returns:
            Ready-to-use HybridSearchEngine instance.
        """
        # ── FAISS ──────────────────────────────────────────────────────────
        print("\n[HybridSearchEngine] Loading FAISS indexes ...")
        text_faiss  = FaissIndex(embedding_dim=1024)
        text_faiss.load(FAISS_TEXT_INDEX,  FAISS_TEXT_IDMAP)

        code_faiss  = FaissIndex(embedding_dim=768)
        code_faiss.load(FAISS_CODE_INDEX,  FAISS_CODE_IDMAP)

        image_faiss = FaissIndex(embedding_dim=512)
        image_faiss.load(FAISS_IMAGE_INDEX, FAISS_IMAGE_IDMAP)

        # ── Embedders ──────────────────────────────────────────────────────
        print("[HybridSearchEngine] Loading embedders ...")
        text_emb  = TextEmbedder()
        code_emb  = CodeEmbedder()
        image_emb = ImageEmbedder()

        # ── BM25 (Whoosh) ──────────────────────────────────────────────────
        text_bm25 = WhooshBM25Index(WHOOSH_TEXT_DIR)
        code_bm25 = WhooshBM25Index(WHOOSH_CODE_DIR)

        if rebuild_bm25 or not whoosh_index.exists_in(str(WHOOSH_TEXT_DIR)):
            print("[HybridSearchEngine] Building Whoosh text index ...")
            text_chunks = _load_json(CHUNKED_TEXT_FILE)
            text_bm25.build(text_chunks, content_field="text", aux_field="section")
        else:
            text_bm25.load()

        if rebuild_bm25 or not whoosh_index.exists_in(str(WHOOSH_CODE_DIR)):
            print("[HybridSearchEngine] Building Whoosh code index ...")
            code_chunks = _load_json(CHUNKED_CODE_FILE)
            code_bm25.build(code_chunks, content_field="code", aux_field="function_name")
        else:
            code_bm25.load()

        print("[HybridSearchEngine] Ready.\n")
        return cls(
            text_faiss, code_faiss, image_faiss,
            text_emb, code_emb, image_emb,
            text_bm25, code_bm25,
            rrf_k=rrf_k,
        )

    # ── Public search API ─────────────────────────────────────────────────

    def search(
        self,
        query:    str,
        modality: str,
        top_k:    int = 10,
    ) -> list[tuple[str, float]]:
        """
        Run hybrid retrieval for a query.

        For ``"text"`` and ``"code"`` modalities, semantic (FAISS) and
        lexical (BM25) results are fused with RRF.

        For ``"image"`` modality, only semantic (CLIP) search is performed
        because BM25 lexical matching is not applicable to image embeddings.

        Args:
            query:    Query string.
            modality: One of ``"text"``, ``"code"``, ``"image"``.
            top_k:    Number of final results to return.

        Returns:
            List of ``(chunk_id, score)`` sorted by descending fusion score.
            For image queries the score is the raw CLIP inner-product score.

        Raises:
            ValueError: If modality is not one of the supported values.
        """
        modality = modality.lower()

        if modality == "text":
            return self._hybrid_text_or_code(
                query, top_k,
                faiss_index=self._text_faiss,
                embed_fn=lambda q: self._text_emb.encode_query(q),
                bm25_index=self._text_bm25,
            )
        elif modality == "code":
            return self._hybrid_text_or_code(
                query, top_k,
                faiss_index=self._code_faiss,
                embed_fn=lambda q: self._code_emb.encode([q], batch_size=1),
                bm25_index=self._code_bm25,
            )
        elif modality == "image":
            return self._semantic_only(
                query, top_k,
                faiss_index=self._image_faiss,
                embed_fn=lambda q: self._image_emb.encode_text_query(q),
            )
        else:
            raise ValueError(
                f"Unsupported modality '{modality}'. "
                "Choose from: 'text', 'code', 'image'."
            )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _hybrid_text_or_code(
        self,
        query:       str,
        top_k:       int,
        faiss_index: FaissIndex,
        embed_fn,
        bm25_index:  WhooshBM25Index,
        fetch_k:     int = 20,
    ) -> list[tuple[str, float]]:
        """Fetch `fetch_k` candidates from each system, fuse, return top_k."""
        # Semantic (FAISS)
        vec = embed_fn(query)
        if vec.ndim == 2:
            vec = vec[0]
        semantic = faiss_index.search(vec, top_k=fetch_k)

        # Lexical (BM25)
        lexical = bm25_index.search(query, top_k=fetch_k)

        # Fuse
        fused = reciprocal_rank_fusion(semantic, lexical, k=self._rrf_k)
        return fused[:top_k]

    def _semantic_only(
        self,
        query:       str,
        top_k:       int,
        faiss_index: FaissIndex,
        embed_fn,
    ) -> list[tuple[str, float]]:
        """Run pure semantic search (no BM25 fusion)."""
        vec = embed_fn(query)
        if vec.ndim == 2:
            vec = vec[0]
        return faiss_index.search(vec, top_k=top_k)


# ============================================================================
# Module-level convenience functions
# ============================================================================

def bm25_search(
    query:      str,
    modality:   str,
    top_k:      int = 10,
    *,
    text_bm25:  Optional[WhooshBM25Index] = None,
    code_bm25:  Optional[WhooshBM25Index] = None,
) -> list[tuple[str, float]]:
    """
    Standalone BM25 search for a single modality.

    If ``text_bm25`` / ``code_bm25`` index objects are not provided, they
    are loaded from the default disk paths on each call (useful for quick
    one-off queries; prefer passing pre-loaded objects in a loop).

    Args:
        query:     Free-text query string.
        modality:  ``"text"`` or ``"code"``.  Image is not supported.
        top_k:     Maximum number of results.
        text_bm25: Optional pre-loaded WhooshBM25Index for text.
        code_bm25: Optional pre-loaded WhooshBM25Index for code.

    Returns:
        List of ``(chunk_id, bm25_score)`` ordered by descending score.

    Raises:
        ValueError: If modality is ``"image"`` or unknown.
    """
    modality = modality.lower()
    if modality == "text":
        idx = text_bm25 or _lazy_load_bm25(WHOOSH_TEXT_DIR)
    elif modality == "code":
        idx = code_bm25 or _lazy_load_bm25(WHOOSH_CODE_DIR)
    else:
        raise ValueError(
            f"bm25_search() does not support modality '{modality}'. "
            "Only 'text' and 'code' are indexed lexically."
        )
    return idx.search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _lazy_load_bm25(index_dir: Path) -> WhooshBM25Index:
    idx = WhooshBM25Index(index_dir)
    idx.load()
    return idx
