"""
generation/rag_pipeline.py

Standalone generation layer for multimodal RAG (Phase 8).

Provides:
- RAGPipeline: context building + answer generation with FLAN-T5
- prepare_text_chunks: helper to map chunk IDs to chunk text/code
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config.settings import get_device, get_model


class RAGPipeline:
    """Generative layer for answering questions from retrieved chunks."""

    def __init__(self, model_name: str = get_model("llm")) -> None:
        """Load tokenizer and seq2seq model on the best available device."""
        self.model_name = model_name
        self.device = get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _is_low_quality_answer(answer: str) -> bool:
        """Detect very short outputs that should fallback."""
        cleaned = answer.strip()
        return len(cleaned) < 3

    @staticmethod
    def _select_context_chunks(retrieved_chunks: list[dict[str, str]]) -> list[dict[str, str]]:
        """Select up to 2 text and 2 code chunks while preserving relevance order."""
        text_chunks = [
            chunk for chunk in retrieved_chunks
            if chunk.get("chunk_id", "").startswith("text_")
        ]
        code_chunks = [
            chunk for chunk in retrieved_chunks
            if chunk.get("chunk_id", "").startswith("code_")
        ]

        # Fallback: treat unknown chunk types as text-like explanations.
        unknown_chunks = [
            chunk for chunk in retrieved_chunks
            if not chunk.get("chunk_id", "").startswith(("text_", "code_"))
        ]

        selected_text = (text_chunks + unknown_chunks)[:2]
        selected_code = code_chunks[:2]
        return selected_text + selected_code

    def build_context(self, retrieved_chunks: list[dict[str, str]]) -> str:
        """
        Build a single context string from retrieved chunks.

        Rules:
        - Up to 2 text chunks
        - Up to 2 code chunks
        - Truncate each chunk to ~300 tokens
        """
        if not retrieved_chunks:
            return ""

        selected_chunks = self._select_context_chunks(retrieved_chunks)
        text_parts: list[str] = []
        code_parts: list[str] = []

        for chunk in selected_chunks:
            chunk_id = chunk.get("chunk_id", "unknown_chunk")
            chunk_text = chunk.get("text", "")

            token_ids = self.tokenizer.encode(
                chunk_text,
                add_special_tokens=False,
                truncation=True,
                max_length=300,
            )
            truncated_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            if chunk_id.startswith("code_"):
                code_parts.append(f"[CODE {chunk_id}]\n{truncated_text}")
            else:
                text_parts.append(f"[TEXT {chunk_id}]\n{truncated_text}")

        sections: list[str] = ["The following context includes explanations and code."]

        sections.append("### TEXT EXPLANATIONS")
        if text_parts:
            sections.append("\n\n".join(text_parts))
        else:
            sections.append("[TEXT none]\nNo text explanation chunks provided.")

        sections.append("### CODE SNIPPETS")
        if code_parts:
            sections.append("\n\n".join(code_parts))
        else:
            sections.append("[CODE none]\nNo code snippets provided.")

        return "\n\n".join(sections)

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Generate an answer grounded in provided context.

        Returns:
            {
              "answer": str,
              "used_chunks": [chunk_ids]
            }
        """
        context = self.build_context(retrieved_chunks)
        selected_chunks = self._select_context_chunks(retrieved_chunks)

        prompt = (
            "You are a helpful machine learning assistant.\n\n"
            "Answer the question using the provided context.\n"
            "If relevant information is present, explain it clearly.\n\n"
            "Guidelines:\n"
            "- Use the context as primary source\n"
            "- You may paraphrase for clarity\n"
            "- If code is present, explain what it does\n"
            "- Keep answers clear and structured\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.5,
                do_sample=True,
            )

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if self._is_low_quality_answer(answer):
            answer = "Not enough information"

        used_chunks = [chunk.get("chunk_id", "unknown_chunk") for chunk in selected_chunks]

        return {
            "answer": answer,
            "used_chunks": used_chunks,
        }


def prepare_text_chunks(
    chunk_ids: list[str],
    chunk_store: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Map chunk IDs to text payloads for generation.

    Args:
        chunk_ids: List of chunk IDs from retrieval output.
        chunk_store: Loaded JSON chunk store containing chunk_id and text/code.

    Returns:
        List of dicts with:
            {
                "chunk_id": str,
                "text": str
            }

    Raises:
        KeyError: If a requested chunk_id is not found.
    """
    store_map = {chunk["chunk_id"]: chunk for chunk in chunk_store}

    prepared: list[dict[str, str]] = []
    for chunk_id in chunk_ids:
        if chunk_id not in store_map:
            raise KeyError(f"Chunk ID '{chunk_id}' not found in chunk store.")

        source = store_map[chunk_id]
        text = source.get("text") or source.get("code") or ""
        prepared.append({"chunk_id": chunk_id, "text": text})

    return prepared
