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
    vectors = embedder.encode(["data_sample/raw/images/roc_curve_example.png"])
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

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
    def _load_image(path: str | Path) -> Image.Image | None:
        """
        Open an image file and convert it to RGB.

        CLIP's ViT patch embedding expects three-channel input, so grayscale
        and RGBA images are converted during loading.

        Args:
            path: File path to the image.

        Returns:
            PIL Image in RGB mode, or None if the image cannot be loaded.
            Unsupported formats (e.g., SVG) and corrupted files return None
            instead of raising an exception.
        """
        path = Path(path)
        if not path.exists():
            print(f"[ImageEmbedder] Skipped (not found): {path}")
            return None
        
        try:
            img = Image.open(path)
            return img.convert("RGB")
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            print(f"[ImageEmbedder] Skipped (cannot load): {path} ({type(e).__name__})")
            return None
        except Exception as e:
            print(f"[ImageEmbedder] Skipped (unexpected error): {path} ({type(e).__name__}: {e})")
            return None

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        image_paths: list[str | Path],
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, list[Path]]:
        """
        Encode a list of image file paths into L2-normalised embeddings.

        Each image is loaded, preprocessed to 224×224 pixels (CLIP's
        expected input resolution), passed through the vision encoder, and
        L2-normalised.

        Invalid or unsupported image files (e.g., corrupted files, SVG format)
        are silently skipped with a log message. Only valid images are encoded.

        Args:
            image_paths: List of paths to image files (PNG, JPEG, etc.).
            batch_size:  Images processed per forward pass.  Reduce if OOM.

        Returns:
            A tuple of:
            1) Float32 numpy array of shape (num_valid_images, 512), L2-normalised.
            2) List of valid image paths aligned 1:1 with embedding rows.
        """
        if not image_paths:
            return np.empty((0, self.embedding_dim), dtype=np.float32), []

        if batch_size is None:
            batch_size = get_batch_size()

        all_embeddings = []
        valid_paths_all: list[Path] = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        iterator = range(0, len(image_paths), batch_size)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc="Image embedding batches",
                unit="batch",
            )

        for batch_idx, start in enumerate(iterator, 1):
            batch_paths = image_paths[start: start + batch_size]
            images = [self._load_image(p) for p in batch_paths]
            
            # Keep only valid image/path pairs so IDs can stay aligned upstream.
            valid_pairs = [
                (Path(path), img)
                for path, img in zip(batch_paths, images)
                if img is not None
            ]
            
            # Skip this batch if all images failed to load
            if not valid_pairs:
                continue

            valid_batch_paths = [p for p, _ in valid_pairs]
            valid_images = [img for _, img in valid_pairs]

            # CLIPProcessor handles resizing, centre-cropping, and pixel
            # normalisation using the model's expected mean/std values.
            inputs = self.processor(
                images=valid_images,
                return_tensors="pt",
                padding=True,
            )
            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.no_grad():
                # Full forward pass through CLIP vision encoder
                outputs = self.model.vision_model(pixel_values=pixel_values)
                
                # Extract the image embeddings tensor from the output object
                # CLIP returns a BaseModelOutputWithPooling which contains pooler_output
                image_embeds = outputs.pooler_output
                
                # Project into shared CLIP embedding space
                image_features = self.model.visual_projection(image_embeds)

            # L2 normalise for cosine-equivalent dot-product retrieval
            normalised = F.normalize(image_features, p=2, dim=1)
            all_embeddings.append(normalised.cpu().numpy())
            valid_paths_all.extend(valid_batch_paths)

            if tqdm is None and (batch_idx == total_batches or batch_idx % 10 == 0):
                print(f"[ImageEmbedder] Progress: {batch_idx}/{total_batches} batches", flush=True)

        # Return empty array if no valid images were encoded
        if not all_embeddings:
            return np.empty((0, self.embedding_dim), dtype=np.float32), []

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        if len(embeddings) != len(valid_paths_all):
            raise RuntimeError("Image embeddings/path alignment mismatch")
        return embeddings, valid_paths_all

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
            # Full forward pass through CLIP text encoder
            outputs = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Extract the text embeddings tensor from the output object
            # CLIP returns a BaseModelOutputWithPooling which contains pooler_output
            text_embeds = outputs.pooler_output
            
            # Project into shared CLIP embedding space
            text_features = self.model.text_projection(text_embeds)

        normalised = F.normalize(text_features, p=2, dim=1)
        return normalised.cpu().numpy().astype(np.float32)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the vision projection output (512 for ViT-B/32)."""
        return self.model.config.projection_dim
