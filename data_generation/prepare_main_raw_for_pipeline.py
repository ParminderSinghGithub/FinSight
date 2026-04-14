"""
Prepare data_main for pipeline entry-point compatibility without mutating source downloads.

What this script does:
1) Renames data_main/raw -> data_main/raw_ (backup source snapshot) if raw_ does not exist.
2) Recreates data_main/raw with pipeline-friendly folders:
   - raw/text   (top-level .txt files for chunker glob '*.txt')
   - raw/code   (top-level .py files for chunker glob '*.py')
   - raw/images (image files for image embed/index stage)
3) Copies (never modifies) source files from raw_ into new raw.
4) Builds data_main/processed/image_metadata.json for discovered images.

Notes:
- Original downloaded files are preserved in data_main/raw_.
- File copies use shutil.copy2 to preserve source timestamps/metadata where possible.
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
except Exception:  # pragma: no cover - optional runtime dependency
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


@dataclass
class PrepStats:
    sec_text_copied: int = 0
    other_text_copied: int = 0
    news_json_files_scanned: int = 0
    news_articles_extracted: int = 0
    code_copied: int = 0
    images_scanned: int = 0
    images_copied: int = 0
    images_skipped_non_visual: int = 0
    images_skipped_duplicate: int = 0


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


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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
    low = str(source_path).lower().replace("\\", "/")
    tags = [k for k in sorted(VISUAL_HINT_KEYWORDS) if k in low]
    if tags:
        return f"Auto-discovered visualization asset from repository path ({', '.join(tags[:4])})."
    return "Auto-discovered image asset from repository source code tree."


def _normalize_text(value: str) -> str:
    text = value.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _word_count(value: str) -> int:
    return len(re.findall(r"\w+", value))


def _extract_news_texts(payload: object) -> list[str]:
    """Extract article-like text blocks from heterogeneous news JSON structures."""
    articles: list[str] = []

    text_keys = (
        "text",
        "body",
        "content",
        "article",
        "description",
        "summary",
    )
    title_keys = ("title", "headline", "subject")

    def walk(node: object) -> None:
        if isinstance(node, dict):
            title = ""
            for k in title_keys:
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    title = v.strip()
                    break

            body = ""
            for k in text_keys:
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    body = v.strip()
                    break

            if body:
                merged = f"{title}\n\n{body}" if title else body
                merged = _normalize_text(merged)
                if _word_count(merged) >= 80:
                    articles.append(merged)

            for child in node.values():
                if isinstance(child, (dict, list)):
                    walk(child)

        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    walk(item)

    walk(payload)

    # De-duplicate article texts while preserving order.
    unique: list[str] = []
    seen: set[str] = set()
    for article in articles:
        digest = hashlib.sha1(article.lower().encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(article)
    return unique


def prepare_main_raw(
    data_root: Path,
    include_all_images: bool,
    overwrite_image_metadata: bool,
) -> dict:
    stats = PrepStats()

    raw_path = data_root / "raw"
    raw_backup = data_root / "raw_"
    processed_path = data_root / "processed"

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    # One-time safety rename requested by user.
    if not raw_backup.exists():
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Expected source folder missing: {raw_path}. Cannot create backup raw_."
            )
        raw_path.rename(raw_backup)
        print(f"[prep] Renamed '{raw_path}' -> '{raw_backup}'")
    else:
        print(f"[prep] Backup folder already present: {raw_backup}")

    # Create pipeline-friendly raw layout (preserve existing files).
    raw_path.mkdir(parents=True, exist_ok=True)
    text_out = raw_path / "text"
    code_out = raw_path / "code"
    images_out = raw_path / "images"
    text_out.mkdir(parents=True, exist_ok=True)
    code_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    text_src = raw_backup / "text"
    code_src = raw_backup / "code"
    
    # Prepare output directories
    processed_path.mkdir(parents=True, exist_ok=True)
    metadata_path = processed_path / "image_metadata.json"

    # 1) Text migration: copy all .txt found under raw_/text into top-level raw/text.
    txt_files = sorted(_iter_files(text_src, {".txt"}), key=lambda p: str(p).lower())
    for src in txt_files:
        rel = src.relative_to(text_src)
        rel_low = str(rel).lower().replace("\\", "/")
        digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
        stem = _slugify(src.stem)
        dest = text_out / f"{stem}_{digest}.txt"
        if not dest.exists():  # Skip if already copied
            shutil.copy2(src, dest)
            if "sec_filings/" in rel_low:
                stats.sec_text_copied += 1
            else:
                stats.other_text_copied += 1
        elif "sec_filings/" in rel_low:
            stats.sec_text_copied += 1
        else:
            stats.other_text_copied += 1

    # 1b) News migration: extract article text from raw_/text/news JSON archives.
    news_src = text_src / "news"
    news_json_files = sorted(_iter_files(news_src, {".json"}), key=lambda p: str(p).lower())
    article_index = stats.sec_text_copied + stats.other_text_copied
    for src in news_json_files:
        stats.news_json_files_scanned += 1
        try:
            payload = json.loads(src.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue

        articles = _extract_news_texts(payload)
        if not articles:
            continue

        rel = src.relative_to(text_src)
        rel_digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
        base = _slugify(src.stem)

        for local_idx, article in enumerate(articles):
            dest = text_out / f"news_{base}_{rel_digest}_{local_idx:03d}.txt"
            if not dest.exists():  # Skip if already extracted
                dest.write_text(article + "\n", encoding="utf-8")
                stats.other_text_copied += 1
                stats.news_articles_extracted += 1
            article_index += 1

    # 2) Code migration: copy all .py files under raw_/code into top-level raw/code.
    py_files = sorted(_iter_files(code_src, {".py"}), key=lambda p: str(p).lower())
    for src in py_files:
        rel = src.relative_to(code_src)
        digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
        stem = _slugify(src.stem)
        dest = code_out / f"{stem}_{digest}.py"
        if not dest.exists():  # Skip if already copied
            shutil.copy2(src, dest)
            stats.code_copied += 1
        else:
            stats.code_copied += 1

    # 3) Image discovery from raw_/code tree and copy to raw/images.
    # Load existing image metadata to avoid re-indexing.
    existing_records: list[dict] = []
    next_image_idx = 0
    if metadata_path.exists():
        try:
            existing_records = json.loads(metadata_path.read_text(encoding="utf-8"))
            next_image_idx = len(existing_records)
        except Exception:
            pass
    
    image_records: list[dict] = list(existing_records)  # Start with existing records
    seen_hashes: set[str] = set()
    
    # Track hashes from existing records
    for rec in existing_records:
        if "extra" in rec and "sha1" in rec["extra"]:
            seen_hashes.add(rec["extra"]["sha1"])

    image_candidates = sorted(_iter_files(code_src, IMAGE_EXTENSIONS), key=lambda p: str(p).lower())
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
        
        if not dest.exists():  # Skip if already copied
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

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "raw_backup": str(raw_backup),
        "new_raw": str(raw_path),
        "counts": {
            "sec_text_copied": stats.sec_text_copied,
            "other_text_copied": stats.other_text_copied,
            "news_json_files_scanned": stats.news_json_files_scanned,
            "news_articles_extracted": stats.news_articles_extracted,
            "code_copied": stats.code_copied,
            "images_scanned_from_code": stats.images_scanned,
            "images_copied": stats.images_copied,
            "images_skipped_non_visual": stats.images_skipped_non_visual,
            "images_skipped_duplicate": stats.images_skipped_duplicate,
            "image_metadata_records": len(image_records),
        },
    }

    report_path = processed_path / "main_raw_prep_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[prep] Completed main raw preparation")
    print(json.dumps(report["counts"], indent=2))
    print(f"[prep] Image metadata: {metadata_path}")
    print(f"[prep] Report: {report_path}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data_main/raw for pipeline entry-point compatibility")
    parser.add_argument(
        "--data-root",
        default="data_main",
        help="Path to data root containing raw/ and processed/ (default: data_main)",
    )
    parser.add_argument(
        "--include-all-images",
        action="store_true",
        help="Include all discovered images from raw_/code (default: only likely visualizations)",
    )
    parser.add_argument(
        "--overwrite-image-metadata",
        action="store_true",
        help="Overwrite processed/image_metadata.json if it exists (default: overwrite anyway with fresh records)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _project_root()
    data_root = (root / args.data_root).resolve()

    prepare_main_raw(
        data_root=data_root,
        include_all_images=args.include_all_images,
        overwrite_image_metadata=args.overwrite_image_metadata,
    )


if __name__ == "__main__":
    main()
