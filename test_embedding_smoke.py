"""
test_embedding_smoke.py

Diagnostic smoke test for all three embedding modules.
Loads 5 samples per modality, encodes them, and prints shape + L2 norm.

Usage (from project root, with venv activated):
    python test_embedding_smoke.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.text_embedder  import TextEmbedder   # noqa: E402
from embeddings.code_embedder  import CodeEmbedder   # noqa: E402
from embeddings.image_embedder import ImageEmbedder  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
CHUNKED_TEXT  = PROJECT_ROOT / "data" / "processed" / "chunked_text.json"
CHUNKED_CODE  = PROJECT_ROOT / "data" / "processed" / "chunked_code.json"
IMAGE_META    = PROJECT_ROOT / "data" / "processed" / "image_metadata.json"

N_SAMPLES = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def report(label: str, embeddings: np.ndarray) -> None:
    """Print shape and L2 norm of the first row."""
    norm = float(np.linalg.norm(embeddings[0]))
    print(f"  {label}")
    print(f"    shape      : {embeddings.shape}")
    print(f"    dtype      : {embeddings.dtype}")
    print(f"    norm[0]    : {norm:.6f}  (expected ~1.0 if L2-normalised)")
    print(f"    min / max  : {embeddings.min():.4f} / {embeddings.max():.4f}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_text_samples() -> list[str]:
    records = load_json(CHUNKED_TEXT)
    if isinstance(records, dict):
        print("  [WARN] chunked_text.json is empty or not a list — no samples.")
        return []
    samples = records[:N_SAMPLES]
    print(f"  Loaded {len(samples)} text chunk(s) from {CHUNKED_TEXT.name}")
    for s in samples:
        preview = s["text"][:60].replace("\n", " ")
        print(f"    [{s['chunk_id']}] {s['section'][:30]:30s}  \"{preview}...\"")
    return [s["text"] for s in samples]


def load_code_samples() -> list[str]:
    records = load_json(CHUNKED_CODE)
    if isinstance(records, dict):
        print("  [WARN] chunked_code.json is empty or not a list — no samples.")
        return []
    samples = records[:N_SAMPLES]
    print(f"  Loaded {len(samples)} code chunk(s) from {CHUNKED_CODE.name}")
    for s in samples:
        preview = s["code"][:60].replace("\n", " ")
        print(f"    [{s['chunk_id']}] {s['function_name']:20s}  \"{preview}...\"")
    return [s["code"] for s in samples]


def load_image_samples() -> list[str]:
    records = load_json(IMAGE_META)
    if isinstance(records, dict):
        print("  [WARN] image_metadata.json is empty or not a list — no samples.")
        return []
    samples = records[:N_SAMPLES]
    print(f"  Loaded {len(samples)} image record(s) from {IMAGE_META.name}")
    paths = []
    for s in samples:
        p = Path(s["source_file"])
        exists = p.exists()
        status = "OK" if exists else "MISSING"
        print(f"    [{s['doc_id']}] {p.name:40s}  [{status}]")
        if exists:
            paths.append(str(p))
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          Multimodal RAG — Embedding Smoke Test           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Samples ───────────────────────────────────────────────────────────────
    section("1 / 3  Loading samples")
    text_samples  = load_text_samples()
    code_samples  = load_code_samples()
    image_paths   = load_image_samples()

    # ── TextEmbedder ──────────────────────────────────────────────────────────
    section("2 / 3  TextEmbedder  (BAAI/bge-large-en)")
    if text_samples:
        te = TextEmbedder()
        print(f"  Device : {te.device}")
        text_embeddings = te.encode(text_samples, batch_size=4)
        report("Text embeddings", text_embeddings)
    else:
        print("  Skipped — no text samples available.")

    # ── CodeEmbedder ──────────────────────────────────────────────────────────
    section("3a / 3  CodeEmbedder  (microsoft/codebert-base)")
    if code_samples:
        ce = CodeEmbedder()
        print(f"  Device : {ce.device}")
        code_embeddings = ce.encode(code_samples, batch_size=4)
        report("Code embeddings", code_embeddings)
    else:
        print("  Skipped — no code samples available.")

    # ── ImageEmbedder ─────────────────────────────────────────────────────────
    section("3b / 3  ImageEmbedder  (openai/clip-vit-base-patch32)")
    if image_paths:
        ie = ImageEmbedder()
        print(f"  Device : {ie.device}")
        image_embeddings = ie.encode(image_paths, batch_size=4)
        report("Image embeddings", image_embeddings)

        # Cross-modal bonus: encode a text query into CLIP space
        query = "DBSCAN clustering with noise points"
        q_emb = ie.encode_text_query(query)
        print()
        print(f"  Cross-modal query : \"{query}\"")
        print(f"    text query shape  : {q_emb.shape}")
        print(f"    text query norm   : {np.linalg.norm(q_emb[0]):.6f}")
        dots = image_embeddings @ q_emb[0]
        for i, (path, score) in enumerate(zip(image_paths, dots)):
            print(f"    cos_sim[{i}] = {score:.4f}  ({Path(path).name})")
    else:
        print("  Skipped — no valid image paths found.")

    section("Done")
    print("  All embedder smoke tests completed.")
    print()


if __name__ == "__main__":
    main()
