"""
build_and_test_indexes.py

End-to-end script that:
  1. Loads all chunked data from data/processed/
  2. Generates embeddings for text, code, and images
  3. Builds and saves three FAISS indexes
  4. Runs test semantic queries against each index

Run from project root:
    python build_and_test_indexes.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.text_embedder  import TextEmbedder   # noqa: E402
from embeddings.code_embedder  import CodeEmbedder   # noqa: E402
from embeddings.image_embedder import ImageEmbedder  # noqa: E402
from indexing.faiss_index      import FaissIndex     # noqa: E402
from config.settings           import get_batch_size, get_path, get_retrieval_top_k  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR           = PROJECT_ROOT / Path(get_path("data"))
INDEX_DIR          = PROJECT_ROOT / Path(get_path("indexes"))
CHUNKED_TEXT_FILE  = DATA_DIR / "processed" / "chunked_text.json"
CHUNKED_CODE_FILE  = DATA_DIR / "processed" / "chunked_code.json"
IMAGE_META_FILE    = DATA_DIR / "processed" / "image_metadata.json"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_TEXT_INDEX   = INDEX_DIR / "faiss_text.index"
FAISS_TEXT_IDMAP   = INDEX_DIR / "faiss_text_idmap.json"
FAISS_CODE_INDEX   = INDEX_DIR / "faiss_code.index"
FAISS_CODE_IDMAP   = INDEX_DIR / "faiss_code_idmap.json"
FAISS_IMAGE_INDEX  = INDEX_DIR / "faiss_image.index"
FAISS_IMAGE_IDMAP  = INDEX_DIR / "faiss_image_idmap.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print(f"\nStep: {msg}")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_results(results: list[tuple[str, float]], label: str) -> None:
    print(f"\nTop results for query: {label}")
    print(f"{'Rank':<5} {'Chunk ID':<15} {'Score':>8}")
    for rank, (chunk_id, score) in enumerate(results, 1):
        print(f"{rank:<5} {chunk_id:<15} {score:>8.4f}")


# ---------------------------------------------------------------------------
# 1 – Load data
# ---------------------------------------------------------------------------

banner("STEP 1 / 4  —  Loading chunked data")

raw_text  = load_json(CHUNKED_TEXT_FILE)
raw_code  = load_json(CHUNKED_CODE_FILE)
raw_image = load_json(IMAGE_META_FILE)

# Normalise: chunked files may be list or dict (empty = {})
text_chunks   = raw_text  if isinstance(raw_text,  list) else []
code_chunks   = raw_code  if isinstance(raw_code,  list) else []
image_records = raw_image if isinstance(raw_image, list) else []

text_ids   = [r["chunk_id"]   for r in text_chunks]
text_texts = [r["text"]       for r in text_chunks]

code_ids   = [r["chunk_id"]   for r in code_chunks]
code_texts = [r["code"]       for r in code_chunks]

image_ids   = [r["doc_id"]                     for r in image_records]
image_paths = [Path(r["source_file"])           for r in image_records]

print(f"Text chunks: {len(text_chunks)}")
print(f"Code chunks: {len(code_chunks)}")
print(f"Images: {len(image_records)}")

# ---------------------------------------------------------------------------
# 2 – Instantiate embedders
# ---------------------------------------------------------------------------

banner("STEP 2 / 4  —  Loading embedders")

text_embedder  = TextEmbedder()
code_embedder  = CodeEmbedder()
image_embedder = ImageEmbedder()

# ---------------------------------------------------------------------------
# 3 – Generate embeddings
# ---------------------------------------------------------------------------

banner("STEP 3 / 4  —  Generating embeddings")

print(f"\nEncoding text chunks: {len(text_texts)}")
text_vecs = text_embedder.encode(text_texts, batch_size=get_batch_size())
print(f"Text embedding shape: {text_vecs.shape}")

print(f"\nEncoding code chunks: {len(code_texts)}")
code_vecs = code_embedder.encode(code_texts, batch_size=get_batch_size())
print(f"Code embedding shape: {code_vecs.shape}")

print(f"\nEncoding images: {len(image_paths)}")
image_vecs = image_embedder.encode(image_paths, batch_size=get_batch_size())
print(f"Image embedding shape: {image_vecs.shape}")

# Sanity-check norms (should all be ~1.0 for unit vectors)
print(
    f"\nNorm checks: "
    f"text[0]={np.linalg.norm(text_vecs[0]):.4f}  "
    f"code[0]={np.linalg.norm(code_vecs[0]):.4f}  "
    f"image[0]={np.linalg.norm(image_vecs[0]):.4f}"
)

# ---------------------------------------------------------------------------
# 4 – Build, populate, and save FAISS indexes
# ---------------------------------------------------------------------------

banner("STEP 4a / 4  —  Building text index")

text_index = FaissIndex(embedding_dim=text_vecs.shape[1])
text_index.add_embeddings(text_vecs, text_ids)
text_index.save(FAISS_TEXT_INDEX, FAISS_TEXT_IDMAP)
print(f"Text index summary: {text_index}")

banner("STEP 4b / 4  —  Building code index")

code_index = FaissIndex(embedding_dim=code_vecs.shape[1])
code_index.add_embeddings(code_vecs, code_ids)
code_index.save(FAISS_CODE_INDEX, FAISS_CODE_IDMAP)
print(f"Code index summary: {code_index}")

banner("STEP 4c / 4  —  Building image index")

image_index = FaissIndex(embedding_dim=image_vecs.shape[1])
image_index.add_embeddings(image_vecs, image_ids)
image_index.save(FAISS_IMAGE_INDEX, FAISS_IMAGE_IDMAP)
print(f"Image index summary: {image_index}")

# ---------------------------------------------------------------------------
# 5 – Test queries
# ---------------------------------------------------------------------------

banner("STEP 5 / 4  —  Semantic search test queries")

# ── Text query ───────────────────────────────────────────────────────────────
TEXT_QUERY = "How does KMeans++ initialization reduce variance?"
print(f"\nText query: {TEXT_QUERY}")
text_q_vec = text_embedder.encode_query(TEXT_QUERY)          # (1, 1024)
text_results = text_index.search(text_q_vec[0], top_k=get_retrieval_top_k())
print_results(text_results, TEXT_QUERY)

# Enrich with snippet preview
print()
for chunk_id, score in text_results:
    match = next((r for r in text_chunks if r["chunk_id"] == chunk_id), None)
    if match:
        snippet = match["text"][:100].replace("\n", " ")
        print(f"Result chunk: {chunk_id}  section={match['section']}")
        print(f"Snippet: \"{snippet}...\"")

# ── Code query ────────────────────────────────────────────────────────────────
CODE_QUERY = "DBSCAN eps parameter tuning example"
print(f"\nCode query: {CODE_QUERY}")
code_q_vec = code_embedder.encode([CODE_QUERY], batch_size=get_batch_size())    # (1, 768)
code_results = code_index.search(code_q_vec[0], top_k=get_retrieval_top_k())
print_results(code_results, CODE_QUERY)

print()
for chunk_id, score in code_results:
    match = next((r for r in code_chunks if r["chunk_id"] == chunk_id), None)
    if match:
        snippet = match["code"][:100].replace("\n", " ")
        print(f"Result chunk: {chunk_id}  fn={match['function_name']}  type={match['chunk_type']}")
        print(f"Snippet: \"{snippet}...\"")

# ── Cross-modal image query (CLIP text → image) ───────────────────────────────
IMAGE_QUERY = "DBSCAN clustering with noise"
print(f"\nImage query: {IMAGE_QUERY}")
img_q_vec = image_embedder.encode_text_query(IMAGE_QUERY)        # (1, 512)
image_results = image_index.search(img_q_vec[0], top_k=get_retrieval_top_k())
print_results(image_results, IMAGE_QUERY)

print()
for img_id, score in image_results:
    match = next((r for r in image_records if r["doc_id"] == img_id), None)
    if match:
        filename = Path(match["source_file"]).name
        print(f"Result image: {img_id}  file={filename}")
        print(f"Caption: \"{match['caption'][:100]}...\"")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

banner("COMPLETE")
print(f"Indexes saved to: {INDEX_DIR}")
print(f"{'File':<35} {'Vectors':>8}")
for path in sorted(INDEX_DIR.glob("*.index")):
    idx = FaissIndex(1)
    idx.load(path, path.with_suffix("").parent / (path.stem + "_idmap.json"))
    print(f"{path.name:<35} {idx.size:>8}")
print()
