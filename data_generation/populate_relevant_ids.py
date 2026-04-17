"""
Populate evaluation_queries.json relevant_ids in a staged way.

Behavior:
1) Always tries to fill image-query relevant_ids from image_metadata.json
   using filename-keyword matching (works before indexing).
2) Optionally fills text/code relevant_ids from FAISS retrieval if indexes exist.

Usage:
    python data_generation/populate_relevant_ids.py --dataset main
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config, get_path, get_retrieval_top_k  # noqa: E402


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _image_keyword_bank() -> list[tuple[str, list[str]]]:
    return [
        ("architecture", ["architecture", "framework", "overview"]),
        ("compares algorithm performance", ["alg_compare", "performancealgs", "performance"]),
        ("cumulative return", ["cumulativereturn", "performance"]),
        ("confusion matrix", ["confusion_matrix", "cost_matrix"]),
        ("roc-curve", ["roc"]),
        ("model selection workflow", ["model_selection", "random_search"]),
        ("streaming or prequential", ["stream_prequential", "stream_train", "stream_valid", "prequential"]),
        ("baseline machine-learning workflow", ["baseline_ml_workflow"]),
        ("transaction-level fraud score", ["transaction_fraud_score", "hits_per_day"]),
        ("neural-network concept", ["attention", "embedding", "autoencoder", "convnet", "dropout", "activations"]),
    ]


def _fill_image_relevant_ids(queries: list[dict], image_meta_path: Path) -> int:
    if not image_meta_path.exists():
        return 0

    image_records = _load_json(image_meta_path)
    corpus = []
    for rec in image_records:
        doc_id = rec.get("doc_id")
        src = rec.get("extra", {}).get("source_original_path", "")
        name = Path(src).name.lower()
        corpus.append((doc_id, name))

    updated = 0
    bank = _image_keyword_bank()
    for entry in queries:
        if entry.get("modality") != "image" or entry.get("relevant_ids"):
            continue

        q = entry.get("query", "").lower()
        chosen: list[str] = []
        for anchor, kws in bank:
            if anchor in q:
                scores: list[tuple[int, str]] = []
                for doc_id, name in corpus:
                    score = sum(1 for kw in kws if kw in name)
                    if score > 0:
                        scores.append((score, doc_id))
                scores.sort(key=lambda x: x[0], reverse=True)
                chosen = [doc_id for _score, doc_id in scores[:3]]
                break

        if not chosen:
            # Conservative fallback: do not guess randomly.
            continue

        entry["relevant_ids"] = chosen
        updated += 1

    return updated


def _fill_text_code_from_faiss(queries: list[dict], top_n: int = 3) -> int:
    """Fill text/code relevant_ids from semantic top-k if indexes are present.

    This is a bootstrap helper, not final labeling. It deliberately skips
    modalities whose indexes are unavailable.
    """
    try:
        from embeddings.text_embedder import TextEmbedder  # noqa: E402
        from embeddings.code_embedder import CodeEmbedder  # noqa: E402
        from indexing.faiss_index import FaissIndex  # noqa: E402
    except Exception:
        return 0

    index_dir = PROJECT_ROOT / Path(get_path("indexes"))
    text_index_path = index_dir / "faiss_text.index"
    text_idmap_path = index_dir / "faiss_text_idmap.json"
    code_index_path = index_dir / "faiss_code.index"
    code_idmap_path = index_dir / "faiss_code_idmap.json"

    has_text = text_index_path.exists() and text_idmap_path.exists()
    has_code = code_index_path.exists() and code_idmap_path.exists()
    if not has_text and not has_code:
        return 0

    text_index = code_index = None
    text_embedder = code_embedder = None

    if has_text:
        text_index = FaissIndex(embedding_dim=1024)
        text_index.load(text_index_path, text_idmap_path)
        text_embedder = TextEmbedder()

    if has_code:
        code_index = FaissIndex(embedding_dim=768)
        code_index.load(code_index_path, code_idmap_path)
        code_embedder = CodeEmbedder()

    updated = 0
    for entry in queries:
        if entry.get("relevant_ids"):
            continue
        modality = str(entry.get("modality", "")).lower()
        query = str(entry.get("query", ""))

        if modality == "text" and has_text and text_index and text_embedder:
            vec = text_embedder.encode_query(query)
            qv = vec[0] if getattr(vec, "ndim", 1) == 2 else vec
            hits = text_index.search(qv, top_k=max(top_n, get_retrieval_top_k()))
            entry["relevant_ids"] = [cid for cid, _ in hits[:top_n]]
            updated += 1

        if modality == "code" and has_code and code_index and code_embedder:
            vec = code_embedder.encode([query])
            qv = vec[0] if getattr(vec, "ndim", 1) == 2 else vec
            hits = code_index.search(qv, top_k=max(top_n, get_retrieval_top_k()))
            entry["relevant_ids"] = [cid for cid, _ in hits[:top_n]]
            updated += 1

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate relevant_ids for evaluation queries")
    parser.add_argument("--dataset", choices=["sample", "main", "btp"], default="main")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    Config.DATASET = args.dataset
    data_dir = PROJECT_ROOT / Path(get_path("data"))
    # Local fallback when config points to Kaggle paths but script runs in repo.
    if not data_dir.exists():
        fallback_map = {
            "sample": "data_sample",
            "main": "data_main",
            "btp": "data_btp",
        }
        fallback = PROJECT_ROOT / fallback_map[args.dataset]
        data_dir = fallback
    queries_path = data_dir / "processed" / "evaluation_queries.json"
    image_meta_path = data_dir / "processed" / "image_metadata.json"

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    queries = _load_json(queries_path)

    image_updates = _fill_image_relevant_ids(queries, image_meta_path)
    text_code_updates = _fill_text_code_from_faiss(queries, top_n=max(1, args.top_n))

    _save_json(queries_path, queries)

    remaining_empty = sum(1 for q in queries if not q.get("relevant_ids"))
    print(f"image_updates={image_updates}")
    print(f"text_code_updates={text_code_updates}")
    print(f"remaining_empty={remaining_empty}")


if __name__ == "__main__":
    main()
