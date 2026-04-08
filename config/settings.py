"""Central project configuration for local sample mode and Kaggle main mode."""


class Config:
    DATASET = "sample"  # "sample" or "main"

    PATHS = {
        "sample": {
            "data": "data_sample/",
            "indexes": "indexes/",
        },
        "main": {
            "data": "/kaggle/working/data_main/",
            "indexes": "/kaggle/working/indexes_main/",
        },
    }

    CHUNK_SIZE = {
        "sample": 512,
        "main": 256,
    }

    BATCH_SIZE = {
        "sample": 8,
        "main": 32,
    }

    DEVICE = {
        "sample": "cpu",
        "main": "cuda",
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
            "code_embedding": "Salesforce/codet5p-110m-embedding",
            "image_embedding": "openai/clip-vit-base-patch32",
            "reranker": "BAAI/bge-reranker-large",
            "llm": "google/flan-t5-xl",
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
    }

    RERANK_TOP_K = {
        "sample": 10,
        "main": 20,
    }


def get_path(key):
    return Config.PATHS[Config.DATASET][key]


def get_model(name):
    return Config.MODELS[Config.DATASET][name]


def get_device():
    return Config.DEVICE[Config.DATASET]


def get_batch_size():
    return Config.BATCH_SIZE[Config.DATASET]


def get_chunk_size():
    return Config.CHUNK_SIZE[Config.DATASET]


def get_retrieval_top_k():
    return Config.RETRIEVAL_TOP_K[Config.DATASET]


def get_rerank_top_k():
    return Config.RERANK_TOP_K[Config.DATASET]
