#!/usr/bin/env python3
"""
test_cross_encoder_reranker.py

Standalone test for Phase 7 CrossEncoderReranker module.

Runs reranking on mock candidates to verify:
- Model loading
- Batch scoring
- Score sorting
- Top-k return
- Helper function behavior
"""

import json
from pathlib import Path

from rerank.cross_encoder_reranker import CrossEncoderReranker, prepare_candidates


# ============================================================================
# Test 1: Model loading and device selection
# ============================================================================

def test_model_loading():
    """Verify model loads and device is correctly set."""
    print("\n" + "=" * 70)
    print("TEST 1: Model Loading and Device Selection")
    print("=" * 70)

    reranker = CrossEncoderReranker()
    print(f"✓ Model loaded: {reranker}")
    print(f"  Device: {reranker.device}")
    print(f"  Model name: {reranker.model_name}")
    assert reranker.device in ("cuda", "cpu"), "Device should be 'cuda' or 'cpu'"
    print("✓ Device selection OK")


# ============================================================================
# Test 2: Reranking with mock candidates
# ============================================================================

def test_rerank_mock():
    """Rerank a small mock set of candidates."""
    print("\n" + "=" * 70)
    print("TEST 2: Reranking Mock Candidates")
    print("=" * 70)

    reranker = CrossEncoderReranker()

    # Mock query and candidates
    query = "machine learning clustering algorithms"
    candidates = [
        {
            "chunk_id": "mock_001",
            "text": "KMeans++ is an algorithm for initializing cluster centers in KMeans clustering.",
        },
        {
            "chunk_id": "mock_002",
            "text": "DBSCAN is a density-based clustering algorithm useful for finding clusters of arbitrary shape.",
        },
        {
            "chunk_id": "mock_003",
            "text": "Python decorators are syntactic sugar for wrapping functions.",
        },
    ]

    print(f"Query: '{query}'")
    print(f"Candidates: {len(candidates)}")
    for cand in candidates:
        print(f"  - {cand['chunk_id']}: {cand['text'][:50]}...")

    # Rerank
    reranked = reranker.rerank(query, candidates, top_k=3)

    print(f"\nReranked (top 3):")
    for chunk_id, score in reranked:
        print(f"  {chunk_id}: {score:.4f}")

    # Verify output format and constraints
    assert len(reranked) <= 3, "Top-k constraint violated"
    assert all(isinstance(score, float) for _, score in reranked), \
        "Scores should be floats"
    assert all(isinstance(chunk_id, str) for chunk_id, _ in reranked), \
        "Chunk IDs should be strings"

    # Verify descending order
    scores = [score for _, score in reranked]
    assert scores == sorted(scores, reverse=True), "Scores not in descending order"

    print("✓ Reranking OK (format, constraints, ordering)")


# ============================================================================
# Test 3: Batching behavior
# ============================================================================

def test_batching():
    """Verify batching works with a larger candidate set."""
    print("\n" + "=" * 70)
    print("TEST 3: Batching Behavior (batch_size=4)")
    print("=" * 70)

    reranker = CrossEncoderReranker()

    query = "recursive algorithms"
    candidates = [
        {"chunk_id": f"batch_{i:03d}", "text": f"Mock candidate text {i}"}
        for i in range(10)
    ]

    reranked = reranker.rerank(query, candidates, top_k=10, batch_size=4)
    print(f"Batched reranking completed: {len(reranked)} results")
    assert len(reranked) == 10, "Should return all 10 candidates (≤ top_k)"
    print("✓ Batching OK")


# ============================================================================
# Test 4: prepare_candidates helper
# ============================================================================

def test_prepare_candidates():
    """Verify prepare_candidates correctly builds candidate dicts."""
    print("\n" + "=" * 70)
    print("TEST 4: prepare_candidates Helper Function")
    print("=" * 70)

    # Load real chunk store
    chunk_file = Path("data/processed/chunked_text.json")
    if not chunk_file.exists():
        print(f"⚠  Skipping: {chunk_file} not found")
        return

    with open(chunk_file) as f:
        chunk_store = json.load(f)

    print(f"Loaded chunk store: {len(chunk_store)} chunks")

    # Extract first 3 chunk IDs
    chunk_ids = [chunk["chunk_id"] for chunk in chunk_store[:3]]
    print(f"Selected chunk IDs: {chunk_ids}")

    # Prepare candidates
    candidates = prepare_candidates(chunk_ids, chunk_store)

    print(f"Prepared {len(candidates)} candidates:")
    for cand in candidates:
        text_preview = cand["text"][:50].replace("\n", " ")
        print(f"  - {cand['chunk_id']}: {text_preview}...")

    # Verify format
    assert len(candidates) == 3, "Should have 3 candidates"
    assert all("chunk_id" in c and "text" in c for c in candidates), \
        "Missing required keys"
    print("✓ prepare_candidates OK")


# ============================================================================
# Test 5: Error handling
# ============================================================================

def test_error_handling():
    """Verify proper error handling for invalid inputs."""
    print("\n" + "=" * 70)
    print("TEST 5: Error Handling")
    print("=" * 70)

    reranker = CrossEncoderReranker()

    # Empty candidates
    result = reranker.rerank("test query", [], top_k=5)
    assert result == [], "Empty candidates should return empty list"
    print("✓ Empty candidates handled")

    # Missing keys in candidate
    try:
        reranker.rerank("test", [{"chunk_id": "x"}], top_k=5)  # missing "text"
        assert False, "Should raise ValueError for missing 'text' key"
    except ValueError as e:
        print(f"✓ Missing key detected: {e}")

    # prepare_candidates with invalid ID
    chunk_store = [
        {"chunk_id": "valid", "text": "content"},
    ]
    try:
        prepare_candidates(["invalid_id"], chunk_store)
        assert False, "Should raise KeyError for missing chunk_id"
    except KeyError as e:
        print(f"✓ Missing chunk ID detected: {e}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Cross-Encoder Reranker — Unit Tests")
    print("=" * 70)

    test_model_loading()
    test_rerank_mock()
    test_batching()
    test_prepare_candidates()
    test_error_handling()

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
