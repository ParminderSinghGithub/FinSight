from __future__ import annotations

import html
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import Config


FALLBACK_SEQ2SEQ_MODEL = "google/flan-t5-large"
HF_CACHE_DIR = Path(__file__).resolve().parent / ".hf_cache"


def _load_env_file() -> None:
	"""Load key=value pairs from .env into process env if not already set."""
	env_path = Path(__file__).resolve().parent / ".env"
	if not env_path.exists():
		return

	for raw_line in env_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key and key not in os.environ:
			os.environ[key] = value


_load_env_file()


def _runtime_signature() -> tuple[float, float]:
	base = Path(__file__).resolve().parent
	rag_path = base / "generation" / "rag_pipeline.py"
	system_path = base / "full_rag_system.py"
	rag_mtime = rag_path.stat().st_mtime if rag_path.exists() else 0.0
	system_mtime = system_path.stat().st_mtime if system_path.exists() else 0.0
	return rag_mtime, system_mtime


st.set_page_config(
	page_title="FinSight",
	page_icon="📈",
	layout="wide",
)


def _inject_styles() -> None:
	st.markdown(
		"""
		<style>
			:root {
				--bg: #0f172a;
				--card: #1e293b;
				--accent: #3b82f6;
				--text: #f8fafc;
				--muted: #cbd5e1;
			}

			.stApp {
				background: var(--bg);
				color: var(--text);
			}

			.main .block-container {
				max-width: 1100px;
				padding-top: 2.2rem;
				padding-bottom: 2rem;
			}

			.finsight-title {
				text-align: center;
				font-size: 3rem;
				line-height: 1.1;
				font-weight: 800;
				color: var(--text);
				margin: 0;
				letter-spacing: 0.01em;
			}

			.finsight-subtitle {
				text-align: center;
				font-size: 1rem;
				color: var(--muted);
				margin-top: 0.6rem;
				margin-bottom: 1.1rem;
			}

			.finsight-divider {
				height: 1px;
				border: 0;
				background: linear-gradient(90deg, rgba(59,130,246,0), rgba(59,130,246,0.8), rgba(59,130,246,0));
				margin: 0.3rem auto 1.8rem auto;
				width: 60%;
			}

			.finsight-card {
				background: var(--card);
				border-radius: 14px;
				padding: 1.1rem 1.2rem;
				margin-top: 1rem;
				margin-bottom: 1rem;
			}

			.finsight-label {
				color: var(--muted);
				font-size: 0.9rem;
				margin-bottom: 0.35rem;
				text-transform: uppercase;
				letter-spacing: 0.05em;
			}

			.finsight-answer {
				color: var(--text);
				font-size: 1.02rem;
				line-height: 1.65;
				white-space: pre-wrap;
				overflow-wrap: anywhere;
				word-break: break-word;
				max-height: 520px;
				overflow-y: auto;
				padding-right: 0.35rem;
				margin-top: 0.35rem;
			}

			section[data-testid="stSidebar"] {
				background: #0b1220;
			}

			section[data-testid="stSidebar"] .stMarkdown,
			section[data-testid="stSidebar"] .stText {
				color: var(--muted);
			}

			.stButton button {
				background: var(--accent);
				color: white;
				border: 0;
				border-radius: 10px;
				padding: 0.55rem 1.2rem;
				font-weight: 600;
			}

			.stButton button:hover {
				background: #2563eb;
				color: white;
			}

			div[data-baseweb="input"] input {
				background: #111c31;
				color: var(--text);
				border-radius: 12px;
				border: 1px solid transparent;
				padding: 0.9rem 1rem;
				font-size: 1rem;
			}

			div[data-baseweb="input"] input:focus {
				border-color: var(--accent);
				box-shadow: 0 0 0 1px rgba(59,130,246,0.35);
			}

			.finsight-source {
				color: var(--text);
				margin: 0.35rem 0;
			}

			.finsight-model-grid {
				display: grid;
				grid-template-columns: 1fr;
				gap: 0.5rem;
				margin-top: 0.4rem;
			}

			.finsight-model-item {
				background: #16233a;
				border-left: 3px solid var(--accent);
				border-radius: 10px;
				padding: 0.45rem 0.6rem;
			}

			.finsight-model-label {
				color: var(--muted);
				font-size: 0.75rem;
				text-transform: uppercase;
				letter-spacing: 0.05em;
				margin-bottom: 0.1rem;
			}

			.finsight-model-value {
				color: var(--text);
				font-size: 0.9rem;
				line-height: 1.35;
			}
		</style>
		""",
		unsafe_allow_html=True,
	)


def initialize_rag_system() -> Any | None:
	current_signature = _runtime_signature()
	cached_signature = st.session_state.get("rag_runtime_signature")
	if cached_signature != current_signature:
		st.session_state.pop("rag_system", None)
		st.session_state.pop("rag_init_error", None)
		st.session_state["rag_runtime_signature"] = current_signature

	if "rag_system" in st.session_state:
		existing_system = st.session_state["rag_system"]
		if existing_system is not None:
			st.session_state.pop("rag_init_error", None)
			return existing_system
		# A previous attempt failed and stored None; retry on the next rerun.
		st.session_state.pop("rag_system", None)
		st.session_state.pop("rag_init_error", None)

	try:
		_configure_hf_cache()
		model_cfg = Config.MODELS.get(Config.DATASET, {})
		llm_name = str(model_cfg.get("llm", ""))
		seq2seq_markers = ("t5", "bart", "pegasus", "mt5")
		if llm_name and not any(marker in llm_name.lower() for marker in seq2seq_markers):
			Config.MODELS[Config.DATASET]["llm"] = FALLBACK_SEQ2SEQ_MODEL
			st.session_state["llm_runtime_override"] = {
				"from": llm_name,
				"to": FALLBACK_SEQ2SEQ_MODEL,
			}

		# RAGPipeline binds default model at import time, so reload modules after override.
		if "generation.rag_pipeline" in sys.modules:
			importlib.reload(sys.modules["generation.rag_pipeline"])
		if "full_rag_system" in sys.modules:
			importlib.reload(sys.modules["full_rag_system"])

		from full_rag_system import FullRAGSystem

		with st.spinner("Initializing system..."):
			system = FullRAGSystem()
			st.session_state.rag_system = StreamlitRAGAdapter(system)
			st.session_state["rag_runtime_signature"] = current_signature
		st.session_state.pop("rag_init_error", None)
		return st.session_state.rag_system
	except Exception as exc:
		st.session_state["rag_system"] = None
		st.session_state["rag_init_error"] = str(exc)
		return None


def _configure_hf_cache() -> None:
	"""Route Hugging Face artifacts to a persistent local cache directory."""
	HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	cache_path = str(HF_CACHE_DIR)
	os.environ.setdefault("HF_HOME", cache_path)
	os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_path)
	os.environ.setdefault("TRANSFORMERS_CACHE", cache_path)


def _has_gemini_key() -> bool:
	key = os.getenv("GEMINI_API_KEY", "").strip()
	return bool(key and "your_" not in key.lower())


def _infer_modality(query: str) -> str:
	q = query.lower()
	code_hints = (
		"code", "function", "class", "python", "implementation", "api", "endpoint",
		"script", "algorithm", "backtest", "drawdown", "sharpe", "roc",
	)
	image_hints = (
		"image", "plot", "chart", "diagram", "figure", "visual", "heatmap",
	)

	if any(token in q for token in image_hints):
		return "image"
	if any(token in q for token in code_hints):
		return "code"
	return "text"


class StreamlitRAGAdapter:
	"""Frontend adapter exposing rag.query(user_query) for the Streamlit app."""

	def __init__(self, system: Any) -> None:
		self._system = system
		self._text_map = {row.get("chunk_id"): row for row in self._system.chunk_stores.get("text", [])}
		self._code_map = {row.get("chunk_id"): row for row in self._system.chunk_stores.get("code", [])}
		self._image_map = self._load_image_metadata()

	def _load_image_metadata(self) -> dict[str, dict[str, Any]]:
		image_file = self._system.processed_dir / "image_metadata.json"
		if not image_file.exists():
			return {}
		try:
			rows = json.loads(image_file.read_text(encoding="utf-8"))
		except Exception:
			return {}
		return {row.get("doc_id"): row for row in rows if row.get("doc_id")}

	def _materialize_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
		rows: list[dict[str, Any]] = []
		for cid in chunk_ids:
			if cid in self._code_map:
				row = self._code_map[cid]
				rows.append(
					{
						"chunk_id": cid,
						"modality": "code",
						"content": row.get("code", ""),
						"source_file": row.get("source_file", ""),
					}
				)
			elif cid in self._text_map:
				row = self._text_map[cid]
				rows.append(
					{
						"chunk_id": cid,
						"modality": "text",
						"content": row.get("text", ""),
						"source_file": row.get("source_file", ""),
					}
				)
			elif cid in self._image_map:
				row = self._image_map[cid]
				rows.append(
					{
						"chunk_id": cid,
						"modality": "image",
						"content": row.get("caption") or row.get("ocr_text") or "",
						"source_file": row.get("source_file", ""),
					}
				)
			else:
				rows.append({"chunk_id": cid, "modality": "unknown", "content": ""})
		return rows

	def query(self, user_query: str, use_gemini: bool = False) -> dict[str, Any]:
		modality = _infer_modality(user_query)
		result = self._system.run_query(user_query, modality, use_gemini=use_gemini)
		used_chunks = result.get("sources", [])
		return {
			"answer": result.get("answer", ""),
			"used_chunks": used_chunks,
			"retrieved_chunks": self._materialize_chunks(used_chunks),
			"modality": modality,
			"mode": result.get("generation_mode", "gemini" if use_gemini else "local-flan"),
		}


def render_header() -> None:
	st.markdown(
		"""
		<div style='display:flex; justify-content:center; align-items:center; gap:10px;'>
		  <svg width='30' height='30' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>
		    <rect x='2' y='2' width='20' height='20' rx='5' fill='#1e293b' stroke='#3b82f6' stroke-width='1.5'/>
		    <path d='M6 15L10 11L13 13L18 8' stroke='#3b82f6' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/>
		    <circle cx='18' cy='8' r='1.4' fill='#3b82f6'/>
		  </svg>
		  <h1 class='finsight-title' style='margin:0;'>FinSight</h1>
		</div>
		""",
		unsafe_allow_html=True,
	)
	st.markdown(
		"<div class='finsight-subtitle'>Multimodal RAG for Technical Documentation (Financial)</div>",
		unsafe_allow_html=True,
	)
	st.markdown("<hr class='finsight-divider' />", unsafe_allow_html=True)


def render_sidebar() -> None:
	st.sidebar.title("System Overview")
	st.sidebar.markdown(
		"""
This system enables unified search across:

- Financial filings and reports
- Code implementations and models
- Visual artifacts such as plots and diagrams

It automatically understands the intent of a query and retrieves the most relevant information across modalities.

Best suited for:

- Financial risk analysis
- Model understanding and implementation
- Technical documentation search
"""
	)

	rag_ready = st.session_state.get("rag_system") is not None
	if rag_ready:
		st.sidebar.success("System Ready")
	else:
		st.sidebar.warning("Loading...")

	st.sidebar.markdown("---")
	st.sidebar.markdown("### Models Used")

	model_cfg = Config.MODELS.get(Config.DATASET, {})
	if model_cfg:
		model_rows = [
			("Text Embedding", model_cfg.get("text_embedding", "n/a")),
			("Code Embedding", model_cfg.get("code_embedding", "n/a")),
			("Image Embedding", model_cfg.get("image_embedding", "n/a")),
			("Reranker", model_cfg.get("reranker", "n/a")),
			("LLM", model_cfg.get("llm", "n/a")),
		]
		html_blocks = [
			(
				"<div class='finsight-model-item'>"
				f"<div class='finsight-model-label'>{html.escape(label)}</div>"
				f"<div class='finsight-model-value'>{html.escape(str(value))}</div>"
				"</div>"
			)
			for label, value in model_rows
		]
		st.sidebar.markdown(
			"<div class='finsight-model-grid'>" + "".join(html_blocks) + "</div>",
			unsafe_allow_html=True,
		)

	if _has_gemini_key():
		st.sidebar.caption("Gemini API: configured")
	else:
		st.sidebar.caption("Gemini API: not configured (Gemini mode will fall back to local model)")

	error_msg = st.session_state.get("rag_init_error")
	if error_msg:
		st.sidebar.markdown("---")
		st.sidebar.error("Initialization failed. Check logs and model assets.")
		st.sidebar.caption(str(error_msg))


def _normalize_result_payload(result: dict[str, Any] | None) -> dict[str, Any] | None:
	if not result:
		return None

	answer = str(result.get("answer", "")).strip()
	used_chunks = result.get("used_chunks")

	if used_chunks is None:
		used_chunks = result.get("sources", [])

	if not isinstance(used_chunks, list):
		used_chunks = []

	return {
		"answer": answer,
		"used_chunks": used_chunks,
		"retrieved_chunks": result.get("retrieved_chunks", []),
		"modality": result.get("modality"),
		"mode": result.get("mode", "Standard Search"),
	}


def handle_query(rag: Any, user_query: str, use_gemini: bool = False) -> Any:
	return rag.query(user_query, use_gemini=use_gemini)


def display_answer(result: dict[str, Any]) -> None:
	answer = result.get("answer", "").strip()
	if not answer:
		st.info("No relevant information found.")
		return

	mode_used = result.get("mode", "Standard Search")

	st.markdown("<div class='finsight-card'>", unsafe_allow_html=True)
	st.markdown(f"<div class='finsight-source'><b>Mode:</b> {html.escape(str(mode_used))}</div>", unsafe_allow_html=True)
	st.markdown(
		f"<div class='finsight-answer'>{html.escape(answer)}</div>",
		unsafe_allow_html=True,
	)
	st.markdown("</div>", unsafe_allow_html=True)


def display_tabs(result: dict[str, Any], show_sources: bool = False) -> None:
	used_chunks = result.get("used_chunks", [])
	retrieved_chunks = result.get("retrieved_chunks", [])
	code_chunks = [
		c for c in retrieved_chunks
		if c.get("modality") == "code" and str(c.get("content", "")).strip()
	]
	image_chunks = [
		c for c in retrieved_chunks
		if c.get("modality") == "image" and str(c.get("content", "")).strip()
	]

	tab_names: list[str] = []
	if code_chunks:
		tab_names.append("Code")
	if image_chunks:
		tab_names.append("Image")
	if show_sources:
		tab_names.append("Sources")

	if not tab_names:
		return

	tabs = st.tabs(tab_names)

	idx = 0
	if code_chunks:
		with tabs[idx]:
			st.markdown("<div class='finsight-card'>", unsafe_allow_html=True)
			for chunk in code_chunks:
				snippet = str(chunk.get("content", "")).strip()
				if snippet:
					st.code(snippet, language="python")
			st.markdown("</div>", unsafe_allow_html=True)
		idx += 1

	if image_chunks:
		with tabs[idx]:
			st.markdown("<div class='finsight-card'>", unsafe_allow_html=True)
			for chunk in image_chunks:
				caption = str(chunk.get("content", "")).strip()
				if caption:
					st.write(caption)
			st.markdown("</div>", unsafe_allow_html=True)
		idx += 1

	if show_sources:
		with tabs[idx]:
			st.markdown("<div class='finsight-card'>", unsafe_allow_html=True)
			if used_chunks:
				for src in used_chunks:
					if isinstance(src, dict):
						source_id = src.get("chunk_id") or src.get("id") or src.get("source") or "unknown"
					else:
						source_id = src
					st.markdown(
						f"<div class='finsight-source'><b>Source:</b> {html.escape(str(source_id))}</div>",
						unsafe_allow_html=True,
					)
			else:
				st.write("No sources available.")
			st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
	_inject_styles()
	rag = initialize_rag_system()
	render_sidebar()
	render_header()

	if "last_result" not in st.session_state:
		st.session_state["last_result"] = None
	if "last_query" not in st.session_state:
		st.session_state["last_query"] = ""
	if "show_sources" not in st.session_state:
		st.session_state["show_sources"] = False
	if "llm_mode" not in st.session_state:
		st.session_state["llm_mode"] = "Standard Search"

	if rag is None:
		st.info("System is still loading. Please wait...")
		return

	if not hasattr(rag, "query"):
		st.error("Configured RAG object does not expose query(user_query).")
		return

	col_left, col_mid, col_right = st.columns([1, 3.2, 1])
	with col_mid:
		llm_mode = st.radio(
			"Choose Mode",
			["Standard Search", "Gemini Enhanced (⚡)"],
			horizontal=True,
			key="llm_mode",
		)
		user_query = st.text_area(
			"Search Query",
			placeholder="Ask about financial data, models, or implementations...",
			height=80,
		)
		submit = st.button("Search", use_container_width=True)

	if not user_query.strip():
		return

	if submit:
		use_gemini = llm_mode == "Gemini Enhanced (⚡)"
		with st.spinner("Retrieving and analyzing..."):
			raw_result = handle_query(rag, user_query.strip(), use_gemini=use_gemini)
		result = _normalize_result_payload(raw_result)
		if not result:
			st.info("No relevant information found.")
			return

		if not result.get("answer"):
			st.info("No relevant information found.")
			return

		st.session_state["last_result"] = result
		st.session_state["last_query"] = user_query.strip()

	result = st.session_state.get("last_result")
	if result:
		display_answer(result)
		st.toggle("Show sources", key="show_sources")
		display_tabs(result, show_sources=bool(st.session_state.get("show_sources", False)))


if __name__ == "__main__":
	main()
