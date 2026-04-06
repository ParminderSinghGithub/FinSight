"""
embeddings/code_embedder.py

Encodes source-code chunks into dense vector embeddings using
microsoft/codebert-base via HuggingFace transformers.

Mean pooling over the last hidden state is applied to produce a
fixed-size vector for each code snippet regardless of its token length.

Typical usage:
    from embeddings.code_embedder import CodeEmbedder
    embedder = CodeEmbedder()
    vectors = embedder.encode(["def foo(): pass", "x = x + 1"])
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config.settings import get_batch_size, get_device, get_model


# CodeBERT is pre-trained on code-docstring pairs across 6 languages.
# 768-dim output.  Supports Python, Java, JavaScript, PHP, Ruby, Go.
MODEL_NAME = get_model("code_embedding")
MAX_LENGTH = 512   # CodeBERT's maximum positional encoding length


class CodeEmbedder:
    """
    Source-code encoder backed by microsoft/codebert-base.

    Produces 768-dimensional L2-normalised embeddings via mean pooling
    over the model's last hidden state, masking out padding tokens so
    they do not contribute to the pooled representation.

    Attributes:
        model_name (str):      HuggingFace model identifier.
        device     (str):      'cuda' or 'cpu'.
        tokenizer  (AutoTokenizer): Loaded tokenizer.
        model      (AutoModel):     Loaded transformer model in eval mode.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """
        Load CodeBERT tokenizer and model onto the best available device.

        Args:
            model_name: HuggingFace model identifier.  Defaults to
                        microsoft/codebert-base.
        """
        self.model_name = model_name
        self.device = get_device()
        print(f"[CodeEmbedder] Loading '{model_name}' on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        print(f"[CodeEmbedder] Ready. Embedding dim: {self.embedding_dim}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a masked mean over the sequence dimension.

        Padding tokens have attention_mask == 0 and are excluded from the
        average, ensuring that chunk length does not bias the embedding
        toward the zero vector.

        Args:
            last_hidden_state: Shape (batch, seq_len, hidden_dim).
            attention_mask:    Shape (batch, seq_len), 1 for real tokens.

        Returns:
            Tensor of shape (batch, hidden_dim).
        """
        # Expand mask to match hidden state dimensions
        mask_expanded = attention_mask.unsqueeze(-1).float()
        # Weighted sum over sequence positions
        sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)
        # Divide by the count of non-padding tokens (clamp to avoid /0)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / sum_mask

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        code_list: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Encode a list of code snippets into L2-normalised embedding vectors.

        Snippets longer than MAX_LENGTH tokens are silently truncated by
        the tokenizer.  For very long functions, consider pre-splitting at
        the chunking stage to avoid information loss.

        Args:
            code_list:  List of source-code strings.
            batch_size: Snippets per forward pass.  Reduce if OOM on GPU.

        Returns:
            Float32 numpy array of shape (len(code_list), 768).
            Each row is L2-normalised.
        """
        if not code_list:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if batch_size is None:
            batch_size = get_batch_size()

        all_embeddings = []

        for start in range(0, len(code_list), batch_size):
            batch = code_list[start: start + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids      = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)

            # L2 normalise so cosine similarity == dot product at retrieval time
            normalised = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normalised.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings (768 for codebert-base)."""
        return self.model.config.hidden_size
