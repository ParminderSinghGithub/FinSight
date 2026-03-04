"""
indexing/faiss_index.py

Wraps a FAISS IndexFlatIP (inner-product / cosine similarity when vectors are
L2-normalised) with a string ID map so that search results return meaningful
chunk identifiers rather than raw integer positions.

Typical usage:
    from indexing.faiss_index import FaissIndex
    import numpy as np

    index = FaissIndex(embedding_dim=1024)
    index.add_embeddings(vectors, ids=["text_000", "text_001", ...])
    results = index.search(query_vec, top_k=5)
    # results → [("text_003", 0.91), ("text_007", 0.88), ...]

    index.save("indexing/text.index", "indexing/text_idmap.json")
    index.load("indexing/text.index", "indexing/text_idmap.json")
"""

import json
from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    """
    FAISS flat inner-product index with a parallel string ID map.

    IndexFlatIP computes exact (brute-force) inner products.  When the
    stored vectors and query vector are L2-normalised (unit norm), inner
    product equals cosine similarity.  No quantisation is applied, so
    retrieval is lossless — suitable for datasets up to ~1 M vectors on CPU.

    Attributes:
        embedding_dim (int):          Dimensionality expected for all vectors.
        index         (faiss.Index):  Underlying FAISS index object.
        id_map        (list[str]):    Position-aligned list of chunk IDs.
                                      id_map[i] is the chunk ID for the i-th
                                      vector stored in the index.
    """

    def __init__(self, embedding_dim: int) -> None:
        """
        Initialise an empty flat inner-product index.

        Args:
            embedding_dim: Vector dimensionality.  This must match the
                           dimensionality of every embedding passed to
                           add_embeddings() and search().
        """
        self.embedding_dim = embedding_dim
        self.index: faiss.Index = faiss.IndexFlatIP(embedding_dim)
        self.id_map: list[str] = []

    # ── Adding vectors ────────────────────────────────────────────────────────

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        ids: list[str],
    ) -> None:
        """
        Add a batch of embedding vectors and their corresponding IDs.

        Vectors are cast to float32 (C-contiguous) before being handed to
        FAISS, which requires exactly this layout.

        Args:
            embeddings: 2-D array of shape (n, embedding_dim).
            ids:        List of n chunk ID strings that correspond
                        positionally to the rows of `embeddings`.

        Raises:
            ValueError: If embeddings is not 2-D, if the second dimension
                        does not match embedding_dim, or if len(ids) does
                        not match the number of rows.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D, got shape {embeddings.shape}."
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {embeddings.shape[1]}."
            )
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"Length of ids ({len(ids)}) must match number of "
                f"embedding rows ({embeddings.shape[0]})."
            )

        # FAISS requires float32 in C-contiguous memory layout
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.id_map.extend(ids)

    # ── Searching ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Retrieve the top-k most similar vectors to a query.

        The query is reshaped to (1, embedding_dim) if a 1-D array is
        supplied, making it safe to pass single embedding rows directly
        without manual reshaping at the call site.

        Args:
            query_vector: 1-D array of shape (embedding_dim,) or 2-D array
                          of shape (1, embedding_dim).
            top_k:        Number of results to return.  Clamped to the
                          current index size if top_k exceeds it.

        Returns:
            List of (chunk_id, score) tuples ordered by descending score.
            Score is the inner product (≡ cosine similarity for unit vectors).

        Raises:
            ValueError: If the query dimensionality does not match the index.
        """
        if self.index.ntotal == 0:
            return []

        # Ensure 2-D float32 C-contiguous layout
        qv = np.ascontiguousarray(query_vector, dtype=np.float32)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)

        if qv.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query dim {qv.shape[1]} does not match index dim "
                f"{self.embedding_dim}."
            )

        # Clamp top_k to available vectors to avoid FAISS assertion errors
        k = min(top_k, self.index.ntotal)
        scores, positions = self.index.search(qv, k)

        results: list[tuple[str, float]] = []
        for pos, score in zip(positions[0], scores[0]):
            if pos == -1:
                # FAISS returns -1 for unfilled slots when k > ntotal
                continue
            results.append((self.id_map[pos], float(score)))

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        index_path: str | Path,
        idmap_path: str | Path,
    ) -> None:
        """
        Persist the FAISS index and ID map to disk.

        Args:
            index_path: Destination path for the FAISS binary index file
                        (e.g. 'indexing/text.index').
            idmap_path: Destination path for the JSON ID map
                        (e.g. 'indexing/text_idmap.json').
        """
        index_path = Path(index_path)
        idmap_path = Path(idmap_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        idmap_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with open(idmap_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, indent=2)

        print(
            f"[FaissIndex] Saved {self.index.ntotal} vector(s) "
            f"→ {index_path.name}  |  {idmap_path.name}"
        )

    def load(
        self,
        index_path: str | Path,
        idmap_path: str | Path,
    ) -> None:
        """
        Load a previously saved FAISS index and ID map from disk.

        Replaces the current index and id_map in-place.

        Args:
            index_path: Path to the FAISS binary index file.
            idmap_path: Path to the JSON ID map file.

        Raises:
            FileNotFoundError: If either file does not exist.
        """
        index_path = Path(index_path)
        idmap_path = Path(idmap_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not idmap_path.exists():
            raise FileNotFoundError(f"ID map file not found: {idmap_path}")

        self.index = faiss.read_index(str(index_path))

        with open(idmap_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        # Sync embedding_dim from the loaded index
        self.embedding_dim = self.index.d

        print(
            f"[FaissIndex] Loaded {self.index.ntotal} vector(s) "
            f"from {index_path.name}  |  {idmap_path.name}"
        )

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently stored in the index."""
        return self.index.ntotal

    def __repr__(self) -> str:
        return (
            f"FaissIndex(embedding_dim={self.embedding_dim}, "
            f"size={self.size})"
        )
