"""
utils/metadata_schema.py

Defines base metadata structures and ID generation utilities
for all modalities in the multimodal RAG pipeline.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Base metadata structures
# ---------------------------------------------------------------------------

def text_metadata(
    doc_id: str,
    source_file: str,
    modality: str = "text",
    language: Optional[str] = None,
    page_number: Optional[int] = None,
    section: Optional[str] = None,
    char_count: Optional[int] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Base metadata record for a text document or chunk.

    Args:
        doc_id:       Unique document identifier (e.g. 'text_001').
        source_file:  Original file path or URL.
        modality:     Always 'text' for this schema.
        language:     Detected or declared language (e.g. 'en').
        page_number:  Page number within the source document, if applicable.
        section:      Section or heading label, if applicable.
        char_count:   Character count of the raw text.
        extra:        Any additional key-value pairs.

    Returns:
        dict: Populated metadata record.
    """
    return {
        "doc_id": doc_id,
        "modality": modality,
        "source_file": source_file,
        "language": language,
        "page_number": page_number,
        "section": section,
        "char_count": char_count,
        "extra": extra or {},
    }


def code_metadata(
    doc_id: str,
    source_file: str,
    modality: str = "code",
    programming_language: Optional[str] = None,
    function_name: Optional[str] = None,
    class_name: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    token_count: Optional[int] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Base metadata record for a code document or chunk.

    Args:
        doc_id:               Unique document identifier (e.g. 'code_005').
        source_file:          Original file path or URL.
        modality:             Always 'code' for this schema.
        programming_language: Detected or declared language (e.g. 'python').
        function_name:        Name of the enclosing function, if applicable.
        class_name:           Name of the enclosing class, if applicable.
        start_line:           Start line number in the source file.
        end_line:             End line number in the source file.
        token_count:          Approximate token count of the code snippet.
        extra:                Any additional key-value pairs.

    Returns:
        dict: Populated metadata record.
    """
    return {
        "doc_id": doc_id,
        "modality": modality,
        "source_file": source_file,
        "programming_language": programming_language,
        "function_name": function_name,
        "class_name": class_name,
        "start_line": start_line,
        "end_line": end_line,
        "token_count": token_count,
        "extra": extra or {},
    }


def image_metadata(
    doc_id: str,
    source_file: str,
    modality: str = "image",
    format: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    caption: Optional[str] = None,
    ocr_text: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Base metadata record for an image document.

    Args:
        doc_id:      Unique document identifier (e.g. 'image_012').
        source_file: Original file path or URL.
        modality:    Always 'image' for this schema.
        format:      Image format (e.g. 'JPEG', 'PNG').
        width:       Image width in pixels.
        height:      Image height in pixels.
        caption:     Human-provided or auto-generated caption.
        ocr_text:    Text extracted via OCR, if applicable.
        extra:       Any additional key-value pairs.

    Returns:
        dict: Populated metadata record.
    """
    return {
        "doc_id": doc_id,
        "modality": modality,
        "source_file": source_file,
        "format": format,
        "width": width,
        "height": height,
        "caption": caption,
        "ocr_text": ocr_text,
        "extra": extra or {},
    }


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

VALID_MODALITIES = {"text", "code", "image"}


def generate_unique_id(modality: str, index: int) -> str:
    """
    Generate a zero-padded unique document ID for a given modality.

    Format: <modality>_<index:03d>
    Examples:
        generate_unique_id("text",  1)  -> "text_001"
        generate_unique_id("code",  5)  -> "code_005"
        generate_unique_id("image", 12) -> "image_012"

    Args:
        modality: One of 'text', 'code', or 'image'.
        index:    Non-negative integer index for the document.

    Returns:
        str: Formatted unique ID string.

    Raises:
        ValueError: If modality is not recognised or index is negative.
    """
    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"Unknown modality '{modality}'. Must be one of {sorted(VALID_MODALITIES)}."
        )
    if index < 0:
        raise ValueError(f"Index must be a non-negative integer, got {index}.")

    return f"{modality}_{index:03d}"
