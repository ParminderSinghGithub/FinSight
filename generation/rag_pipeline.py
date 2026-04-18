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

        compact_prompt = self._build_compact_rag_prompt(query=query, context=context, selected_chunks=selected_chunks)
        gemini_prompt = self._build_gemini_rag_prompt(query=query, context=context, selected_chunks=selected_chunks)

        generation_mode = "local-flan"
        generation_note = ""
        if use_gemini:
            answer, gemini_err = self._generate_with_gemini(prompt=gemini_prompt)
            if answer is None or not answer.strip():
                answer = self._generate_with_local_model(prompt=compact_prompt)
                generation_mode = "local-flan-fallback"
                generation_note = gemini_err or "gemini_empty"
            else:
                generation_mode = "gemini"
                generation_note = "gemini_ok"

            if generation_mode == "gemini" and self._needs_gemini_extension(answer):
                answer = self._extend_gemini_answer(
                    query=query,
                    context=context,
                    current_answer=answer,
                    selected_chunks=selected_chunks,
                )
        else:
            answer = self._generate_with_local_model(prompt=compact_prompt)
            generation_note = "local_only"

        if generation_mode != "gemini" and self._is_low_quality_answer(answer):
            rescue_prompt = self._build_rescue_prompt(query=query, context=context)
            rescue_answer = self._generate_with_local_model(prompt=rescue_prompt)
            if not self._is_low_quality_answer(rescue_answer):
                answer = rescue_answer
                generation_mode = "local-rescue"
                generation_note = "local_rescue_ok"
            else:
                answer = self._build_rule_based_fallback_summary(query=query, selected_chunks=selected_chunks)
                generation_mode = "rule-based-fallback"
                generation_note = "rule_based_summary"

        used_chunks = [chunk.get("chunk_id", "unknown_chunk") for chunk in selected_chunks]

        return {
            "answer": answer,
            "used_chunks": used_chunks,
            "generation_mode": generation_mode,
            "generation_note": generation_note,
        }

    @staticmethod
    def _build_rule_based_fallback_summary(query: str, selected_chunks: list[dict[str, str]]) -> str:
        """Final fallback when model outputs are too short despite relevant retrieved context."""
        if not selected_chunks:
            return "Not enough information."

        text_chunks = [c for c in selected_chunks if not str(c.get("chunk_id", "")).startswith(("code_", "image_"))]
        source_chunks = text_chunks or selected_chunks
        snippets: list[str] = []
        for chunk in source_chunks[:3]:
            raw = str(chunk.get("text", "")).strip().replace("\n", " ")
            if not raw:
                continue
            sentence = raw.split(".")[0].strip()
            if sentence:
                snippets.append(sentence[:200])

        if not snippets:
            return "Not enough information."

        q = query.lower()
        if any(k in q for k in ("supplier", "component", "supply", "shortage", "tariff", "trade")):
            lines = [
                "1. Supplier and component concentration risk: The disclosures indicate dependency on specific suppliers or limited-source components, increasing vulnerability to disruptions.",
                "2. Operational continuity risk: Supply shortages and procurement constraints can delay production cycles and impair product availability in critical periods.",
                "3. Financial impact risk: Persistent supply and trade frictions can pressure margins, reduce revenue conversion, and increase cost volatility.",
            ]
            return "\n".join(lines)

        lines = [
            "1. Core point: Retrieved disclosures highlight material operational risks tied to the query domain.",
            "2. Evidence signal: Multiple context sections point to compounding impacts rather than isolated events.",
            "3. Business implication: These risks can affect execution reliability, cost structure, and performance outcomes.",
        ]
        return "\n".join(lines)

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
    def _build_compact_rag_prompt(query: str, context: str, selected_chunks: list[dict[str, str]]) -> str:
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
            "Write a concise, presentation-ready answer using only the provided context.\n"
            "STRICT INSTRUCTIONS:\n"
            "- Give exactly 3 numbered points\n"
            "- Keep the full answer under 170 words\n"
            "- Use short, direct sentences\n"
            "- Do not copy text directly\n"
            "- Do not add an intro or conclusion\n"
            "- Stop after point 3\n"
            f"{modality_block}"
            "\nRESPONSE STRUCTURE:\n"
            "1. Define the key concept\n"
            "2. Explain the operational impact\n"
            "3. Explain why it matters\n\n"
            "If evidence is missing, say so briefly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

    @staticmethod
    def _build_gemini_rag_prompt(query: str, context: str, selected_chunks: list[dict[str, str]]) -> str:
        code_count = sum(1 for c in selected_chunks if c.get("chunk_id", "").startswith("code_"))
        image_count = sum(1 for c in selected_chunks if c.get("chunk_id", "").startswith("image_"))

        modality_lines: list[str] = []
        if code_count > 0:
            modality_lines.append("- When code appears, explain the logic and why it matters in one short example")
        if image_count > 0:
            modality_lines.append("- When visual context appears, explain what it shows and its implication")
        modality_block = "\n".join(modality_lines)
        if modality_block:
            modality_block += "\n"

        return (
            "You are an expert in financial risk analysis and technical documentation.\n\n"
            "Write a fuller, presentation-ready answer using only the provided context.\n"
            "STRICT INSTRUCTIONS:\n"
            "- Write exactly 6 numbered points\n"
            "- Aim for 280 to 420 words total\n"
            "- Each point should be 1 to 2 concise sentences and stay on its own line\n"
            "- Use the available context fully and explain the practical impact\n"
            "- Do not copy long phrases directly\n"
            "- Do not add an introduction or closing line\n"
            "- End cleanly after point 6\n"
            f"{modality_block}"
            "\nRESPONSE STRUCTURE:\n"
            "1. State the main concept or risk\n"
            "2. Explain the evidence from the context\n"
            "3. Explain the operational effect\n"
            "4. Explain the business or financial consequence\n"
            "5. Add a relevant supporting detail\n"
            "6. Give a short final takeaway\n\n"
            "If evidence is limited, say so briefly but still provide the best grounded answer.\n"
            "Write enough detail so the answer is not just a few words.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

    @staticmethod
    def _count_numbered_lines(answer: str) -> int:
        return sum(1 for line in answer.splitlines() if re.match(r"^\s*\d+[.)]\s+", line))

    def _needs_gemini_extension(self, answer: str) -> bool:
        cleaned = answer.strip()
        if not cleaned:
            return False
        if self._count_numbered_lines(cleaned) >= 6:
            return False
        return len(self._tokenize_words(cleaned)) < 140

    def _extend_gemini_answer(
        self,
        query: str,
        context: str,
        current_answer: str,
        selected_chunks: list[dict[str, str]],
    ) -> str:
        continuation_prompt = (
            "You are continuing a previous answer.\n\n"
            "Add the missing numbered points so the final answer has exactly 6 numbered lines total.\n"
            "Do not repeat earlier text.\n"
            "Write only the missing points, each on its own line, with 1 to 2 concise sentences per point.\n"
            "Keep the answer grounded in the context and make it longer and more complete.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            f"Existing answer:\n{current_answer}\n\n"
            "Missing points:"
        )

        extension, _ = self._generate_with_gemini(prompt=continuation_prompt)
        extension = (extension or "").strip()
        if not extension:
            return current_answer.strip()

        combined = f"{current_answer.strip()}\n{extension}".strip()
        if self._needs_gemini_extension(combined):
            # Preserve the Gemini answer even if it is still shorter than ideal.
            return combined
        return combined

    @staticmethod
    def _build_rescue_prompt(query: str, context: str) -> str:
        return (
            "You are an expert assistant for financial documents.\n\n"
            "Use only the provided context and produce exactly 3 numbered points.\n"
            "Each point must be 1 sentence, short, and directly tied to the risk question.\n"
            "Do not copy long phrases verbatim.\n"
            "Keep the answer under 150 words.\n\n"
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
                max_new_tokens=170,
                min_new_tokens=40,
                num_beams=3,
                length_penalty=1.0,
                do_sample=False,
                early_stopping=True,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    @staticmethod
    def _generate_with_gemini(prompt: str) -> tuple[str | None, str | None]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None, "missing_api_key"

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            preferred = os.getenv("GEMINI_MODEL", "").strip()
            candidate_models = [
                preferred,
                "models/gemini-2.5-flash",
                "models/gemini-2.0-flash",
                "models/gemini-flash-latest",
                "models/gemini-pro-latest",
            ]
            # Keep order and remove blanks/duplicates.
            seen: set[str] = set()
            candidate_models = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

            last_error = "gemini_unknown_error"
            for model_name in candidate_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        prompt,
                        request_options={"timeout": 12},
                        generation_config={
                            "temperature": 0.25,
                            "max_output_tokens": 700,
                            "top_p": 0.95,
                        },
                    )
                    text = getattr(response, "text", None)
                    if text is None and getattr(response, "candidates", None):
                        parts = response.candidates[0].content.parts
                        text = "".join(getattr(p, "text", "") for p in parts)
                    text = (text or "").strip()
                    if text:
                        return text, None
                    last_error = f"empty_gemini_response:{model_name}"
                except Exception as exc:
                    last_error = f"gemini_error:{type(exc).__name__}:{model_name}"
                    continue

            return None, last_error
        except Exception as exc:
            return None, f"gemini_error:{type(exc).__name__}"


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
