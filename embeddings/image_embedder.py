"""
embeddings/image_embedder.py

Encodes images into dense vector embeddings using
openai/clip-vit-base-patch32 via HuggingFace transformers.

CLIP's vision encoder produces 512-dimensional embeddings that share the
same latent space as its text encoder, enabling cross-modal retrieval
(image ↔ text) in downstream pipeline stages.

Typical usage:
    from embeddings.image_embedder import ImageEmbedder
    embedder = ImageEmbedder()
    vectors = embedder.encode(["data/raw/images/roc_curve_example.png"])
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config.settings import get_batch_size, get_device, get_model


# CLIP ViT-B/32 is the standard lightweight CLIP checkpoint.
# Vision embedding dim: 512.  Weights ~350 MB.
MODEL_NAME = get_model("image_embedding")


class ImageEmbedder:
    """
    Image encoder backed by openai/clip-vit-base-patch32.

    Produces 512-dimensional L2-normalised embeddings from image file paths.
    The embeddings reside in the joint CLIP multimodal space, meaning they
    are directly comparable (via dot product) to text embeddings produced
    by a CLIP text encoder.

    Attributes:
        model_name (str):      HuggingFace model identifier.
        device     (str):      'cuda' or 'cpu'.
        processor  (CLIPProcessor): Image pre-processor (resize, normalise).
        model      (CLIPModel):     Loaded CLIP model in eval mode.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """
        Load CLIP processor and model onto the best available device.

        Args:
            model_name: HuggingFace model identifier.  Defaults to
                        openai/clip-vit-base-patch32.
        """
        self.model_name = model_name
        self.device = get_device()
        print(f"[ImageEmbedder] Loading '{model_name}' on {self.device} ...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        print(f"[ImageEmbedder] Ready. Embedding dim: {self.embedding_dim}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_image(path: str | Path) -> Image.Image:
        """
        Open an image file and convert it to RGB.

        CLIP's ViT patch embedding expects three-channel input, so grayscale
        and RGBA images are converted during loading.

        Args:
            path: File path to the image.

        Returns:
            PIL Image in RGB mode.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        image_paths: list[str | Path],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Encode a list of image file paths into L2-normalised embeddings.

        Each image is loaded, preprocessed to 224×224 pixels (CLIP's
        expected input resolution), passed through the vision encoder, and
        L2-normalised.

        Args:
            image_paths: List of paths to image files (PNG, JPEG, etc.).
            batch_size:  Images processed per forward pass.  Reduce if OOM.

        Returns:
            Float32 numpy array of shape (len(image_paths), 512).
            Each row is L2-normalised.

        Raises:
            FileNotFoundError: If any image path does not exist.
        """
        if not image_paths:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if batch_size is None:
            batch_size = get_batch_size()

        all_embeddings = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start: start + batch_size]
            images = [self._load_image(p) for p in batch_paths]

            # CLIPProcessor handles resizing, centre-cropping, and pixel
            # normalisation using the model's expected mean/std values.
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
            )
            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.no_grad():
                # get_image_features() extracts the [CLS]-like vision embedding
                # and projects it into the shared CLIP embedding space.
                image_features = self.model.get_image_features(
                    pixel_values=pixel_values
                )

            # L2 normalise for cosine-equivalent dot-product retrieval
            normalised = F.normalize(image_features, p=2, dim=1)
            all_embeddings.append(normalised.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    # ── Cross-modal helpers ───────────────────────────────────────────────────

    def encode_text_query(self, query: str) -> np.ndarray:
        """
        Encode a text string into the CLIP embedding space.

        Use this to compute cross-modal similarity between a text query and
        image embeddings produced by encode().  Both reside in the same
        512-dimensional normalised CLIP space.

        Args:
            query: Text string to encode.

        Returns:
            Float32 array of shape (1, 512), L2-normalised.
        """
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        normalised = F.normalize(text_features, p=2, dim=1)
        return normalised.cpu().numpy().astype(np.float32)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the vision projection output (512 for ViT-B/32)."""
        return self.model.config.projection_dim
