"""
chunking/text_chunker.py

Splits plain-text documents into overlapping chunks, keyed by ALL CAPS
section headings.  Chunk metadata is appended to
data/processed/chunked_text.json.

Usage (from project root):
    from chunking.text_chunker import chunk_text_file
    chunk_text_file("data/raw/text/dbscan_overview.txt")
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metadata_schema import generate_unique_id  # noqa: E402

CHUNKED_TEXT_FILE = PROJECT_ROOT / "data" / "processed" / "chunked_text.json"


# ── Token estimation ──────────────────────────────────────────────────────────

def estimate_token_count(text: str) -> int:
    """
    Approximate token count using whitespace splitting.

    This is a lightweight proxy for BPE token counts.  For English prose,
    whitespace tokens correlate reasonably well with transformer tokens
    (typically within 10-20 %).  Swap out for a real tokenizer in Phase 3.

    Args:
        text: Raw input string.

    Returns:
        Integer estimate of token count.
    """
    return len(text.split())


# ── Section splitting ─────────────────────────────────────────────────────────

# Matches a heading line: one or more ALL-CAPS words (and spaces/colons/hyphens),
# occupying the full line, with at least two consecutive caps words.
_HEADING_RE = re.compile(
    r"^([A-Z][A-Z0-9 :,\-]{4,})$",
    re.MULTILINE,
)


def split_into_sections(text: str) -> list[dict]:
    """
    Split document text into labelled sections using ALL CAPS headings.

    A section spans from one heading line to the next (or end of file).
    The heading text is stored as the section title; if no heading precedes
    the first content, the section is labelled 'PREAMBLE'.

    Args:
        text: Full document text.

    Returns:
        List of dicts with keys:
            'heading' (str): The ALL CAPS heading, or 'PREAMBLE'.
            'body'    (str): Section body text (heading line excluded).
    """
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        # No headings detected — treat the whole file as one section
        return [{"heading": "PREAMBLE", "body": text.strip()}]

    sections = []

    # Text before the first heading (if any)
    pre = text[: matches[0].start()].strip()
    if pre:
        sections.append({"heading": "PREAMBLE", "body": pre})

    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append({"heading": heading, "body": body})

    return sections


# ── Overlapping chunk builder ─────────────────────────────────────────────────

def _build_chunks(
    words: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Produce overlapping word-level chunks from a flat word list.

    Args:
        words:      Pre-split token list.
        chunk_size: Maximum tokens per chunk.
        overlap:    Number of tokens shared between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_existing() -> list:
    if CHUNKED_TEXT_FILE.exists() and CHUNKED_TEXT_FILE.stat().st_size > 2:
        with open(CHUNKED_TEXT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    return []


def _save(records: list) -> None:
    CHUNKED_TEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKED_TEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# ── Main entry ────────────────────────────────────────────────────────────────

def chunk_text_file(
    file_path: str | Path,
    chunk_size: int = 512,
    overlap_ratio: float = 0.2,
) -> list[dict]:
    """
    Read a plain-text file, split it into overlapping chunks, and persist
    structured records to data/processed/chunked_text.json.

    Chunking strategy:
        1. Split the document into sections at ALL CAPS headings.
        2. Within each section, slide a window of `chunk_size` tokens
           with `overlap_ratio * chunk_size` tokens of overlap.
        3. Assign a unique ID, section label, and token count to each chunk.

    Args:
        file_path:     Path to the .txt file (absolute or relative to cwd).
        chunk_size:    Maximum token count per chunk (whitespace-split tokens).
        overlap_ratio: Fraction of chunk_size that overlaps with the next chunk.
                       E.g. 0.2 means 20 % of tokens are repeated.

    Returns:
        List of newly created chunk records (not including pre-existing ones).
    """
    file_path = Path(file_path).resolve()
    text = file_path.read_text(encoding="utf-8")

    overlap = max(1, int(chunk_size * overlap_ratio))
    sections = split_into_sections(text)

    existing = _load_existing()
    start_idx = len(existing)
    new_records = []
    idx = start_idx

    for section in sections:
        words = section["body"].split()
        raw_chunks = _build_chunks(words, chunk_size, overlap)

        for chunk_text in raw_chunks:
            record = {
                "chunk_id":    generate_unique_id("text", idx),
                "source_file": str(file_path),
                "section":     section["heading"],
                "chunk_index": idx - start_idx,
                "token_count": estimate_token_count(chunk_text),
                "char_count":  len(chunk_text),
                "text":        chunk_text,
            }
            new_records.append(record)
            idx += 1

    _save(existing + new_records)
    print(
        f"[text_chunker] {file_path.name}: "
        f"{len(sections)} section(s), {len(new_records)} chunk(s) "
        f"→ {CHUNKED_TEXT_FILE.name}"
    )
    return new_records
