from __future__ import annotations

import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from config.settings import Config


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".hf_cache"

# Skip alternate framework artifacts we do not use in this project.
IGNORE_PATTERNS = [
    "tf_model.h5",
    "flax_model.msgpack",
    "rust_model.ot",
    "*.onnx",
    "openvino*",
]

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5


def _set_cache_env() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = str(CACHE_DIR)
    os.environ.setdefault("HF_HOME", cache_path)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_path)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_path)
    # Windows commonly lacks symlink permissions; avoid noisy warnings.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _model_ids() -> list[str]:
    models = Config.MODELS.get(Config.DATASET, {})
    keys = ["text_embedding", "code_embedding", "image_embedding", "reranker", "llm"]
    ids = [models.get(k, "") for k in keys]
    return [m for m in ids if isinstance(m, str) and m.strip()]


def main() -> None:
    _set_cache_env()

    print(f"Dataset: {Config.DATASET}")
    print(f"Cache: {CACHE_DIR}")

    for model_id in _model_ids():
        print(f"\nPrefetching: {model_id}")
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(CACHE_DIR),
                    local_files_only=False,
                    ignore_patterns=IGNORE_PATTERNS,
                    max_workers=2,
                )
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                print(f"Attempt {attempt}/{MAX_RETRIES} failed for {model_id}: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        if last_error is not None:
            raise RuntimeError(f"Failed to prefetch '{model_id}' after {MAX_RETRIES} attempts") from last_error

    print("\nAll configured models are cached.")


if __name__ == "__main__":
    main()
