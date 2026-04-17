"""Central project configuration for local sample mode, Kaggle main mode, and local btp mode."""

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class Config:
    DATASET = "btp"  # "sample", "main", or "btp"

    PATHS = {
        "sample": {
            "data": "data_sample/",
            "indexes": "indexes/",
        },
        "main": {
            "data": "/kaggle/working/data_main/",
            "indexes": "/kaggle/working/indexes_main/",
        },
        "btp": {
            "data": "data_btp/",
            "indexes": "indexes_btp/",
        },
    }

    CHUNK_SIZE = {
        "sample": 512,
        "main": 256,
        "btp": 256,
    }

    BATCH_SIZE = {
        "sample": 8,
        "main": 32,
        "btp": 128,
    }

    DEVICE = {
        "sample": "cpu",
        "main": "cuda",
        "btp": "cpu",
    }

    MODELS = {
        "sample": {
            "text_embedding": "BAAI/bge-small-en",
            "code_embedding": "microsoft/codebert-base",
            "image_embedding": "openai/clip-vit-base-patch32",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "llm": "google/flan-t5-base",
        },
        "main": {
            "text_embedding": "BAAI/bge-large-en",
            "code_embedding": "microsoft/codebert-base",
            "image_embedding": "openai/clip-vit-base-patch32",
            "reranker": "BAAI/bge-reranker-large",
            "llm": "google/flan-t5-xl",
        },
        "btp": {
            "text_embedding": "BAAI/bge-small-en",
            "code_embedding": "microsoft/codebert-base",
            "image_embedding": "openai/clip-vit-base-patch32",
            "reranker": "BAAI/bge-reranker-large",
            "llm": "google/flan-t5-large",
        },
        # "main": {
        #     "text_embedding": "BAAI/bge-large-en",
        #     "code_embedding": "Salesforce/codet5p-110m-embedding",
        #     "image_embedding": "openai/clip-vit-base-patch32",
        #     "reranker": "BAAI/bge-reranker-large",
        #     "llm": "mistralai/Mistral-7B-Instruct-v0.2",
        # },
    }

    RETRIEVAL_TOP_K = {
        "sample": 5,
        "main": 10,
        "btp": 10,
    }

    RERANK_TOP_K = {
        "sample": 10,
        "main": 20,
        "btp": 20,
    }


def get_path(key):
    return Config.PATHS[Config.DATASET][key]


def get_model(name):
    return Config.MODELS[Config.DATASET][name]


def get_device():
    device = Config.DEVICE[Config.DATASET]
    if device == "cuda" and (torch is None or not torch.cuda.is_available()):
        return "cpu"
    return device


def get_batch_size():
    return Config.BATCH_SIZE[Config.DATASET]


def get_chunk_size():
    return Config.CHUNK_SIZE[Config.DATASET]


def get_retrieval_top_k():
    return Config.RETRIEVAL_TOP_K[Config.DATASET]


def get_rerank_top_k():
    return Config.RERANK_TOP_K[Config.DATASET]
