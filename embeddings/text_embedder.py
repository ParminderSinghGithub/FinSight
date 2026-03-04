"""
embeddings/text_embedder.py

Encodes plain-text chunks into dense vector embeddings using
BAAI/bge-large-en via sentence-transformers.

Typical usage:
    from embeddings.text_embedder import TextEmbedder
    embedder = TextEmbedder()
    vectors = embedder.encode(["Hello world", "Another sentence"])
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# BGE-large produces 1024-dim embeddings and ranks highly on MTEB benchmarks.
# It is well-suited for asymmetric retrieval (short query vs. long document).
MODEL_NAME = "BAAI/bge-large-en"


class TextEmbedder:
    """
    Sentence-level text encoder backed by BAAI/bge-large-en.

    The model is loaded once at construction time and reused across all
    encode() calls.  Embeddings are L2-normalised so that cosine similarity
    reduces to a dot product, which is faster to compute at retrieval time.

    Attributes:
        model_name (str):  HuggingFace model identifier.
        device     (str):  'cuda' if a GPU is available, otherwise 'cpu'.
        model      (SentenceTransformer): Loaded model instance.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """
        Load the sentence-transformer model onto the best available device.

        Args:
            model_name: HuggingFace model identifier.  Defaults to
                        BAAI/bge-large-en.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TextEmbedder] Loading '{model_name}' on {self.device} ...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[TextEmbedder] Ready. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        text_list: list[str],
        batch_size: int = 16,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised embedding vectors.

        BGE models perform best when a brief instruction prefix is prepended
        to the *query* side only.  For document-side encoding (this method),
        no prefix is needed.

        Args:
            text_list:     List of strings to embed.
            batch_size:    Number of texts processed per forward pass.
                           Reduce if GPU memory is limited.
            show_progress: Show a tqdm progress bar during encoding.

        Returns:
            Float32 numpy array of shape (len(text_list), embedding_dim).
            Each row is L2-normalised (unit norm).
        """
        if not text_list:
            return np.empty((0, self.model.get_sentence_embedding_dimension()),
                            dtype=np.float32)

        embeddings = self.model.encode(
            text_list,
            batch_size=batch_size,
            normalize_embeddings=True,   # L2 normalisation applied internally
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    # ── Convenience ───────────────────────────────────────────────────────────

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single retrieval query with the BGE instruction prefix.

        BGE-large-en documentation recommends prepending
        'Represent this sentence for searching relevant passages: '
        to queries (not to documents) for best retrieval accuracy.

        Args:
            query: Raw query string.

        Returns:
            Float32 array of shape (1, embedding_dim), L2-normalised.
        """
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.encode([prefixed], batch_size=1)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        return self.model.get_sentence_embedding_dimension()
