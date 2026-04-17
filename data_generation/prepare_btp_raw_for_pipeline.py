"""
Prepare data_btp as a medium-size dataset derived from main sources.

What this script does:
1) Copies SEC filings text only (no news extraction) into data_btp/raw/text.
2) Copies code files into data_btp/raw/code.
3) Copies discovered images from code tree into data_btp/raw/images.
4) Builds data_btp/processed/image_metadata.json.
5) Creates data_btp/processed/evaluation_queries.json by taking
   data_main queries and removing news-related text queries.

Default source priority:
- data_main/raw_ (original source snapshot)
- fallback: data_main/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}
VISUAL_HINT_KEYWORDS = {
    "plot",
    "plots",
    "figure",
    "fig",
    "chart",
    "graph",
    "visual",
    "viz",
    "roc",
    "pr_curve",
    "precision_recall",
    "heatmap",
    "confusion",
    "hist",
    "histogram",
    "scatter",
    "lineplot",
    "barplot",
    "dashboard",
}


NEWS_QUERY_MARKERS = {
    "market news",
    "news item",
    "financial news",
    "market report",
    "article",
    "coverage",
    "macro commentary",
    "crypto-market",
    "etf flows",
    "fed policy",
    "treasury yields",
    "inflation data",
    "earnings surprises",
    "oil price",
    "bank-sector",
    "volatility rose",
    "ipo market",
    "m&a activity",
    "recession probability",
    "sovereign or fiscal",
    "nafta",
    "bitcoin accumulation",
}


@dataclass
class PrepStats:
    sec_text_copied: int = 0
    code_copied: int = 0
    images_scanned: int = 0
    images_copied: int = 0
    images_skipped_non_visual: int = 0
    images_skipped_duplicate: int = 0
    queries_total_main: int = 0
    queries_kept_btp: int = 0
    queries_removed_news: int = 0


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "file"


def _sha1_bytes(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def _is_likely_visualization(path: Path) -> bool:
    low = str(path).lower().replace("\\", "/")
    return any(keyword in low for keyword in VISUAL_HINT_KEYWORDS)


def _image_dimensions(path: Path) -> tuple[int | None, int | None]:
    if Image is None:
        return None, None
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None, None


def _build_image_caption(source_path: Path) -> str:
    """
    Generate a meaningful caption from filename and visual hint keywords.
    
    Examples:
    - "roc_curve.png" → "ROC curve plot"
    - "confusion_matrix.png" → "Confusion matrix visualization"
    - "feature_importance.png" → "Feature importance chart"
    """
    filename = source_path.stem.lower()
    low = str(source_path).lower().replace("\\", "/")
    
    # Map visual hint keywords to human-readable descriptions
    keyword_descriptions = {
        "roc": "ROC curve",
        "pr_curve": "Precision-recall curve",
        "precision_recall": "Precision-recall plot",
        "confusion": "Confusion matrix",
        "heatmap": "Heatmap visualization",
        "scatter": "Scatter plot",
        "histogram": "Histogram",
        "hist": "Histogram",
        "lineplot": "Line plot",
        "barplot": "Bar plot",
        "plot": "Plot",
        "chart": "Chart",
        "graph": "Graph",
        "figure": "Figure",
        "viz": "Visualization",
        "visual": "Visualization",
        "dashboard": "Dashboard",
    }
    
    # Extract keywords from filename and path
    visual_type = None
    found_keywords = []
    
    for keyword in sorted(VISUAL_HINT_KEYWORDS):
        if keyword in low:
            found_keywords.append(keyword)
            if keyword in keyword_descriptions and not visual_type:
                visual_type = keyword_descriptions[keyword]
    
    # If we found a specific visual type, use it
    if visual_type:
        return f"{visual_type} visualization"
    
    # Otherwise use extracted keywords to build a description
    if found_keywords:
        primary_keyword = found_keywords[0]
        desc = keyword_descriptions.get(primary_keyword, primary_keyword.replace("_", " ").title())
        return f"{desc} visualization"
    
    # Fallback: use filename
    readable_name = filename.replace("_", " ").replace("-", " ").title()
    return f"{readable_name} visualization asset"


def _is_news_query(entry: dict) -> bool:
    if str(entry.get("modality", "")).lower() != "text":
        return False
    query = str(entry.get("query", "")).lower()
    return any(marker in query for marker in NEWS_QUERY_MARKERS)


def _copy_sec_text_only(text_src: Path, text_out: Path, stats: PrepStats) -> None:
    sec_root = text_src / "sec_filings"
    txt_files = sorted(_iter_files(sec_root, {".txt"}), key=lambda p: str(p).lower())

    for src in txt_files:
        rel = src.relative_to(text_src)
        digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
        stem = _slugify(src.stem)
        dest = text_out / f"{stem}_{digest}.txt"
        if not dest.exists():
            shutil.copy2(src, dest)
        stats.sec_text_copied += 1


def _copy_code_files(code_src: Path, code_out: Path, stats: PrepStats) -> None:
    py_files = sorted(_iter_files(code_src, {".py"}), key=lambda p: str(p).lower())
    for src in py_files:
        rel = src.relative_to(code_src)
        digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
        stem = _slugify(src.stem)
        dest = code_out / f"{stem}_{digest}.py"
        if not dest.exists():
            shutil.copy2(src, dest)
        stats.code_copied += 1


def _copy_images_and_metadata(
    code_src: Path,
    images_out: Path,
    metadata_path: Path,
    include_all_images: bool,
    stats: PrepStats,
) -> None:
    image_records: list[dict] = []
    seen_hashes: set[str] = set()

    image_candidates = sorted(_iter_files(code_src, IMAGE_EXTENSIONS), key=lambda p: str(p).lower())
    next_image_idx = 0

    for src in image_candidates:
        stats.images_scanned += 1
        if not include_all_images and not _is_likely_visualization(src):
            stats.images_skipped_non_visual += 1
            continue

        content_hash = _sha1_bytes(src)
        if content_hash in seen_hashes:
            stats.images_skipped_duplicate += 1
            continue
        seen_hashes.add(content_hash)

        ext = src.suffix.lower()
        idx = next_image_idx
        dest_name = f"image_{idx:06d}{ext}"
        dest = images_out / dest_name
        shutil.copy2(src, dest)

        width, height = _image_dimensions(dest)
        source_rel = dest.relative_to(_project_root()).as_posix()
        record = {
            "doc_id": f"image_{idx:06d}",
            "modality": "image",
            "source_file": source_rel,
            "format": ext.lstrip(".").upper(),
            "width": width,
            "height": height,
            "caption": _build_image_caption(src),
            "ocr_text": None,
            "extra": {
                "source": "repo-image-discovery",
                "source_original_path": str(src.relative_to(_project_root()).as_posix()),
                "sha1": content_hash,
            },
        }
        image_records.append(record)
        stats.images_copied += 1
        next_image_idx += 1

    metadata_path.write_text(json.dumps(image_records, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_btp_eval_queries(main_queries_path: Path, btp_queries_path: Path, stats: PrepStats) -> None:
    if not main_queries_path.exists():
        raise FileNotFoundError(f"Missing main evaluation queries: {main_queries_path}")

    queries = json.loads(main_queries_path.read_text(encoding="utf-8"))
    stats.queries_total_main = len(queries)

    kept: list[dict] = []
    for entry in queries:
        if _is_news_query(entry):
            stats.queries_removed_news += 1
            continue
        kept.append(entry)

    stats.queries_kept_btp = len(kept)
    btp_queries_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_btp_raw(
    source_root: Path,
    btp_data_root: Path,
    include_all_images: bool,
) -> dict:
    stats = PrepStats()

    text_src = source_root / "text"
    code_src = source_root / "code"

    if not text_src.exists() or not code_src.exists():
        raise FileNotFoundError(
            f"Expected source folders missing under {source_root}. Need text/ and code/."
        )

    raw_out = btp_data_root / "raw"
    processed_out = btp_data_root / "processed"

    if raw_out.exists():
        shutil.rmtree(raw_out)
    raw_out.mkdir(parents=True, exist_ok=True)

    text_out = raw_out / "text"
    code_out = raw_out / "code"
    images_out = raw_out / "images"
    text_out.mkdir(parents=True, exist_ok=True)
    code_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    processed_out.mkdir(parents=True, exist_ok=True)
    metadata_path = processed_out / "image_metadata.json"

    _copy_sec_text_only(text_src, text_out, stats)
    _copy_code_files(code_src, code_out, stats)
    _copy_images_and_metadata(code_src, images_out, metadata_path, include_all_images, stats)

    main_queries_path = _project_root() / "data_main" / "processed" / "evaluation_queries.json"
    btp_queries_path = processed_out / "evaluation_queries.json"
    _build_btp_eval_queries(main_queries_path, btp_queries_path, stats)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "btp_data_root": str(btp_data_root),
        "counts": {
            "sec_text_copied": stats.sec_text_copied,
            "code_copied": stats.code_copied,
            "images_scanned_from_code": stats.images_scanned,
            "images_copied": stats.images_copied,
            "images_skipped_non_visual": stats.images_skipped_non_visual,
            "images_skipped_duplicate": stats.images_skipped_duplicate,
            "image_metadata_records": stats.images_copied,
            "queries_total_main": stats.queries_total_main,
            "queries_kept_btp": stats.queries_kept_btp,
            "queries_removed_news": stats.queries_removed_news,
        },
    }

    report_path = processed_out / "btp_raw_prep_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[prep-btp] Completed btp raw preparation")
    print(json.dumps(report["counts"], indent=2))
    print(f"[prep-btp] Image metadata: {metadata_path}")
    print(f"[prep-btp] Queries: {btp_queries_path}")
    print(f"[prep-btp] Report: {report_path}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data_btp/raw for pipeline entry-point compatibility")
    parser.add_argument(
        "--source-root",
        default=None,
        help="Source folder containing text/ and code/. Defaults to data_main/raw_ if present, else data_main/raw.",
    )
    parser.add_argument(
        "--btp-root",
        default="data_btp",
        help="Target btp data root (default: data_btp)",
    )
    parser.add_argument(
        "--include-all-images",
        action="store_true",
        help="Include all discovered images from source code tree (default: only likely visualizations)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _project_root()

    if args.source_root:
        source_root = (root / args.source_root).resolve()
    else:
        preferred = (root / "data_main" / "raw_").resolve()
        fallback = (root / "data_main" / "raw").resolve()
        source_root = preferred if preferred.exists() else fallback

    btp_root = (root / args.btp_root).resolve()

    prepare_btp_raw(
        source_root=source_root,
        btp_data_root=btp_root,
        include_all_images=args.include_all_images,
    )


if __name__ == "__main__":
    main()
