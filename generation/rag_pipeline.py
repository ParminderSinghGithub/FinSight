"""
generation/rag_pipeline.py

Standalone generation layer for multimodal RAG (Phase 8).

Provides:
- RAGPipeline: context building + answer generation with FLAN-T5
- prepare_text_chunks: helper to map chunk IDs to chunk text/code
"""

from __future__ import annotations

import os
import re
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
        """Detect answers that are too short to be useful."""
        cleaned = answer.strip()
        return len(cleaned) < 60

    @staticmethod
    def _chunk_modality(chunk: dict[str, str]) -> str:
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id.startswith("code_"):
            return "code"
        if chunk_id.startswith("image_"):
            return "image"
        return "text"

    @staticmethod
    def _tokenize_words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    @staticmethod
    def _is_distinct_chunk_text(candidate_text: str, existing_texts: list[str], threshold: float = 0.8) -> bool:
        """Keep text chunks diverse using token-overlap similarity."""
        cand_tokens = set(RAGPipeline._tokenize_words(candidate_text))
        if not cand_tokens:
            return False

        for existing in existing_texts:
            ex_tokens = set(RAGPipeline._tokenize_words(existing))
            if not ex_tokens:
                continue
            overlap = len(cand_tokens & ex_tokens) / max(1, len(cand_tokens | ex_tokens))
            if overlap >= threshold:
                return False
        return True

    @staticmethod
    def _query_intent(query: str) -> str:
        q = query.lower()
        code_hints = (
            "code", "function", "class", "implementation", "python", "api",
            "snippet", "algorithm", "script", "endpoint", "method",
        )
        image_hints = (
            "image", "figure", "plot", "chart", "diagram", "visual", "roc",
            "confusion matrix", "heatmap",
        )
        if any(h in q for h in image_hints):
            return "image"
        if any(h in q for h in code_hints):
            return "code"
        return "text"

    def _select_context_chunks(self, query: str, retrieved_chunks: list[dict[str, str]]) -> list[dict[str, str]]:
        """Select up to 5 chunks with at least 3 diverse text chunks when available."""
        intent = self._query_intent(query)
        text_chunks = [
            chunk for chunk in retrieved_chunks
            if self._chunk_modality(chunk) == "text"
        ]
        code_chunks = [
            chunk for chunk in retrieved_chunks
            if self._chunk_modality(chunk) == "code"
        ]
        image_chunks = [
            chunk for chunk in retrieved_chunks
            if self._chunk_modality(chunk) == "image"
        ]

        # Force multi-chunk text usage (no duplicates): pick up to 3 diverse text chunks first.
        selected: list[dict[str, str]] = []
        selected_texts: list[str] = []
        seen_ids: set[str] = set()
        min_text_target = min(3, len(text_chunks))
        for chunk in text_chunks:
            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("text", "")
            if chunk_id in seen_ids:
                continue
            if not self._is_distinct_chunk_text(chunk_text, selected_texts):
                continue
            selected.append(chunk)
            selected_texts.append(chunk_text)
            seen_ids.add(chunk_id)
            if len(selected_texts) >= min_text_target:
                break

        if intent == "code":
            order = [code_chunks, text_chunks, image_chunks]
            limits = {"code": 3, "text": 2, "image": 1}
        elif intent == "image":
            order = [image_chunks, text_chunks, code_chunks]
            limits = {"image": 3, "text": 2, "code": 1}
        else:
            order = [text_chunks, code_chunks, image_chunks]
            limits = {"text": 3, "code": 2, "image": 1}

        counts = {"text": 0, "code": 0, "image": 0}
        for chunk in selected:
            counts[self._chunk_modality(chunk)] += 1

        # Primary pass: honor intent-aware ordering and per-modality soft limits.
        for group in order:
            for chunk in group:
                modality = self._chunk_modality(chunk)
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id in seen_ids:
                    continue
                if counts[modality] >= limits[modality]:
                    continue
                selected.append(chunk)
                seen_ids.add(chunk_id)
                counts[modality] += 1
                if len(selected) >= 5:
                    return selected

        # Fill remaining slots by original retrieval order to preserve grounding quality.
        for chunk in retrieved_chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id in seen_ids:
                continue
            selected.append(chunk)
            seen_ids.add(chunk_id)
            if len(selected) >= 5:
                break

        return selected

    def build_context(self, query: str, retrieved_chunks: list[dict[str, str]]) -> str:
        """
        Build a single context string from retrieved chunks.

        Rules:
        - Up to 5 chunks total
        - Intent-aware modality prioritization
        - Truncate each chunk to ~300 tokens
        """
        if not retrieved_chunks:
            return ""

        selected_chunks = self._select_context_chunks(query, retrieved_chunks)
        context_parts: list[str] = []

        for chunk in selected_chunks:
            chunk_text = chunk.get("text", "")
            modality = self._chunk_modality(chunk)

            token_ids = self.tokenizer.encode(
                chunk_text,
                add_special_tokens=False,
                truncation=True,
                max_length=300,
            )
            truncated_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            if modality == "code":
                context_parts.append(f"[CODE]\n{truncated_text}")
            elif modality == "image":
                context_parts.append(f"[IMAGE]\n{truncated_text}")
            else:
                context_parts.append(f"[TEXT]\n{truncated_text}")

        return "\n\n".join(context_parts)

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[dict[str, str]],
        use_gemini: bool = False,
    ) -> dict[str, Any]:
        """
        Generate an answer grounded in provided context.

        Returns:
            {
              "answer": str,
              "used_chunks": [chunk_ids]
            }
        """
        selected_chunks = self._select_context_chunks(query, retrieved_chunks)
        context = self.build_context(query, retrieved_chunks)

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

        strict_prompt = self._build_strict_rag_prompt(query=query, context=context, selected_chunks=selected_chunks)

        if use_gemini:
            answer = self._generate_with_gemini(prompt=strict_prompt)
            if answer is not None and self._is_extractive_dump(answer, context):
                retry_prompt = strict_prompt + "\n\nYou will be penalized if you copy text directly."
                answer = self._generate_with_gemini(prompt=retry_prompt)
            if answer is None or self._is_extractive_dump(answer, context):
                answer = self._generate_with_local_model(prompt=prompt)
        else:
            answer = self._generate_with_local_model(prompt=prompt)

        if self._is_low_quality_answer(answer):
            answer = "Not enough information."

        used_chunks = [chunk.get("chunk_id", "unknown_chunk") for chunk in selected_chunks]

        return {
            "answer": answer,
            "used_chunks": used_chunks,
        }

    def _is_extractive_dump(self, answer: str, context: str) -> bool:
        """Reject copied outputs and unstructured single-paragraph dumps."""
        cleaned = answer.strip()
        if not cleaned:
            return True

        # Single-paragraph dump heuristic (must be structured by the prompt requirements).
        has_structure = any(marker in cleaned for marker in ("1.", "2.", "3."))
        if "\n" not in cleaned and len(cleaned.split(".")) >= 3 and not has_structure:
            return True

        # Long copied phrase heuristic: if any 21-token sequence from answer appears in context.
        ctx_tokens = self._tokenize_words(context)
        ans_tokens = self._tokenize_words(cleaned)
        if len(ctx_tokens) < 21 or len(ans_tokens) < 21:
            return False

        ctx_ngrams = {
            tuple(ctx_tokens[i : i + 21])
            for i in range(0, len(ctx_tokens) - 20)
        }
        for i in range(0, len(ans_tokens) - 20):
            if tuple(ans_tokens[i : i + 21]) in ctx_ngrams:
                return True
        return False

    @staticmethod
    def _build_strict_rag_prompt(query: str, context: str, selected_chunks: list[dict[str, str]]) -> str:
        code_count = sum(1 for c in selected_chunks if c.get("chunk_id", "").startswith("code_"))
        image_count = sum(1 for c in selected_chunks if c.get("chunk_id", "").startswith("image_"))

        modality_lines: list[str] = []
        if code_count > 0:
            modality_lines.append("- If code is present, include a short, relevant snippet and explain it")
        if image_count > 0:
            modality_lines.append("- If visual context is present, describe what the image represents")
        modality_block = "\n".join(modality_lines)
        if modality_block:
            modality_block += "\n"

        return (
            "You are an expert in financial risk analysis.\n\n"
            "Your task is to answer the question using the provided context.\n\n"
            "STRICT INSTRUCTIONS:\n"
            "- You MUST combine information from multiple context sections\n"
            "- You MUST explain relationships between concepts\n"
            "- You MUST NOT copy text directly\n"
            "- You MUST NOT return raw paragraphs from context\n"
            "- You MUST explain in your own words\n"
            "You will be penalized if you copy text directly.\n"
            f"{modality_block}"
            "\nRESPONSE STRUCTURE:\n"
            "1. Define key concepts\n"
            "2. Explain how they are related\n"
            "3. Explain why this relationship matters\n\n"
            "If the relationship is not clearly present, say:\n"
            '"Not enough information."\n\n'
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

    def _generate_with_local_model(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.5,
                do_sample=True,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    @staticmethod
    def _generate_with_gemini(prompt: str) -> str | None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 600,
                    "top_p": 0.95,
                },
            )
            text = getattr(response, "text", None)
            if text is None and getattr(response, "candidates", None):
                parts = response.candidates[0].content.parts
                text = "".join(getattr(p, "text", "") for p in parts)
            text = (text or "").strip()
            return text or "Not enough information."
        except Exception:
            return None


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
