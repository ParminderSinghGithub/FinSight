"""
chunking/code_chunker.py

Splits Python source files into logical chunks: the top-level parameter
block, individual functions (each with its docstring), and main().
Chunk metadata is appended to the configured data directory's processed/chunked_code.json.

Usage (from project root):
    from chunking.code_chunker import split_code_by_functions
    split_code_by_functions("data_sample/raw/code/dbscan_basic.py")
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metadata_schema import generate_unique_id  # noqa: E402
from config.settings import get_path  # noqa: E402

CHUNKED_CODE_FILE = PROJECT_ROOT / Path(get_path("data")) / "processed" / "chunked_code.json"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_existing() -> list:
    if CHUNKED_CODE_FILE.exists() and CHUNKED_CODE_FILE.stat().st_size > 2:
        with open(CHUNKED_CODE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    return []


def _save(records: list) -> None:
    CHUNKED_CODE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKED_CODE_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# ── Raw text parsing ──────────────────────────────────────────────────────────

def _find_function_boundaries(lines: list[str]) -> list[tuple[int, str]]:
    """
    Return (line_index, function_name) for every 'def ' line at column 0.

    Only top-level functions (no leading whitespace) are captured, so
    nested helpers and class methods are grouped with their parent.

    Args:
        lines: File lines (with newlines stripped).

    Returns:
        Sorted list of (start_line_index, function_name) tuples.
    """
    boundaries = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if line.startswith("def ") and stripped.startswith("def "):
            # Extract name: 'def foo_bar(...)' → 'foo_bar'
            name = stripped[4:].split("(")[0].strip()
            boundaries.append((i, name))
    return boundaries


def _extract_parameter_block(lines: list[str]) -> str | None:
    """
    Extract the top-level parameter / constant block.

    Looks for a comment banner matching '# ── Parameters' (or similar) and
    collects all lines until the next comment banner or function definition.

    Args:
        lines: File lines.

    Returns:
        Joined string of the parameter block, or None if not found.
    """
    in_block = False
    block_lines = []

    for line in lines:
        stripped = line.strip()
        # Detect the parameters banner (case-insensitive)
        if not in_block and "parameter" in stripped.lower() and stripped.startswith("#"):
            in_block = True
            block_lines.append(line)
            continue
        if in_block:
            # Stop at the next section banner or function definition
            if (
                stripped.startswith("# ──")
                or stripped.startswith("def ")
                or stripped.startswith("class ")
            ):
                break
            block_lines.append(line)

    text = "\n".join(block_lines).strip()
    return text if text else None


# ── Chunk building ────────────────────────────────────────────────────────────

def _make_record(
    chunk_id: str,
    source_file: Path,
    chunk_type: str,
    function_name: str,
    start_line: int,
    end_line: int,
    code: str,
) -> dict:
    """Build a structured chunk record dict."""
    return {
        "chunk_id":          chunk_id,
        "source_file":       str(source_file),
        "programming_language": "python",
        "chunk_type":        chunk_type,   # 'parameters', 'function', or 'main'
        "function_name":     function_name,
        "start_line":        start_line + 1,  # 1-based for human readability
        "end_line":          end_line + 1,
        "token_count":       len(code.split()),
        "char_count":        len(code),
        "code":              code,
    }


# ── Main entry ────────────────────────────────────────────────────────────────

def split_code_by_functions(
    file_path: str | Path,
) -> list[dict]:
    """
    Parse a Python file and split it into logical code chunks.

    Chunking strategy:
        1. Parameter block  — constant/config section at the top of the file.
        2. Each top-level function — from its 'def' line to the line before
           the next top-level 'def', capturing its full body including
           any inline docstring.
        3. main() is captured as its own separate chunk (type='main').

    Chunk IDs are generated sequentially using generate_unique_id("code", n),
    continuing from however many code chunks already exist in the store.

    Args:
        file_path: Path to the .py source file.

    Returns:
        List of newly created chunk records.
    """
    file_path = Path(file_path).resolve()
    raw = file_path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    existing = _load_existing()
    idx = len(existing)
    new_records = []

    # ── 1. Parameter block ────────────────────────────────────────────────────
    param_text = _extract_parameter_block(lines)
    if param_text:
        # Determine approximate line range
        param_start = next(
            (i for i, l in enumerate(lines) if "parameter" in l.lower() and l.strip().startswith("#")),
            0,
        )
        param_end = param_start + len(param_text.splitlines()) - 1
        new_records.append(
            _make_record(
                generate_unique_id("code", idx),
                file_path,
                chunk_type="parameters",
                function_name="<module>",
                start_line=param_start,
                end_line=param_end,
                code=param_text,
            )
        )
        idx += 1

    # ── 2. Functions ──────────────────────────────────────────────────────────
    boundaries = _find_function_boundaries(lines)

    for pos, (start, name) in enumerate(boundaries):
        # End of this function = line before next top-level def, or EOF
        end = boundaries[pos + 1][0] - 1 if pos + 1 < len(boundaries) else len(lines) - 1

        # Trim trailing blank lines
        while end > start and not lines[end].strip():
            end -= 1

        code = "\n".join(lines[start: end + 1]).strip()
        chunk_type = "main" if name == "main" else "function"

        new_records.append(
            _make_record(
                generate_unique_id("code", idx),
                file_path,
                chunk_type=chunk_type,
                function_name=name,
                start_line=start,
                end_line=end,
                code=code,
            )
        )
        idx += 1

    _save(existing + new_records)
    fn_count = sum(1 for r in new_records if r["chunk_type"] == "function")
    print(
        f"[code_chunker]  {file_path.name}: "
        f"{fn_count} function(s)"
        + (f" + param block" if param_text else "")
        + (f" + main()" if any(r["chunk_type"] == "main" for r in new_records) else "")
        + f" → {len(new_records)} chunk(s) → {CHUNKED_CODE_FILE.name}"
    )
    return new_records
