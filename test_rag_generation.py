#!/usr/bin/env python3
"""
test_rag_generation.py

Standalone qualitative test for Phase 8 generation layer.

Flow per query:
1) Select known-good chunk IDs manually
2) Prepare chunks with prepare_text_chunks
3) Generate answer with RAGPipeline
4) Print clean output (query, answer, sources)
"""

import json
from pathlib import Path

from generation.rag_pipeline import RAGPipeline, prepare_text_chunks


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_block(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    # 1) Load chunk stores
    text_store_path = Path("data_sample/processed/chunked_text.json")
    code_store_path = Path("data_sample/processed/chunked_code.json")

    if not text_store_path.exists() or not code_store_path.exists():
        raise FileNotFoundError(
            "Required chunk stores not found. Expected:\n"
            f"  {text_store_path}\n"
            f"  {code_store_path}"
        )

    text_chunks = load_json(text_store_path)
    code_chunks = load_json(code_store_path)

    # Combine stores so mixed text/code chunk IDs can be resolved.
    combined_store = text_chunks + code_chunks

    # 2) Manual test queries
    tests = [
        {
            "query": "Explain DBSCAN eps parameter with example",
            "chunk_ids": ["text_010", "code_000"],
        },
        {
            "query": "How does KMeans++ initialization work?",
            "chunk_ids": ["text_016", "text_027"],
        },
    ]

    pipeline = RAGPipeline()

    print_block("RAG Generation Test")

    # 3) Run each test query
    for i, test in enumerate(tests, start=1):
        query = test["query"]
        selected_ids = test["chunk_ids"]

        # STEP B — prepare chunks
        retrieved_chunks = prepare_text_chunks(selected_ids, combined_store)

        # STEP C — generate answer
        result = pipeline.generate_answer(query, retrieved_chunks)

        # STEP D — print clean output
        print(f"\n[Q{i}] Query:")
        print(query)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        print(result["used_chunks"])
        print("-" * 80)


if __name__ == "__main__":
    main()
