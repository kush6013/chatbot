import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

from app.config import EMBEDDING_MODEL

EMBED_BATCH_SIZE = 32

MODEL_ALIASES = {
    "all-MiniLM-L6-v2": (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    "bge-small-en-v1.5": (
        "BAAI/bge-small-en-v1.5"
    ),
}


class EmbeddingService:

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            full = MODEL_ALIASES.get(
                EMBEDDING_MODEL,
                EMBEDDING_MODEL
            )

            print(
                f"Loading embedding model: "
                f"{full} (ONNX runtime)"
            )
            self._model = TextEmbedding(
                model_name=full
            )
        return self._model

    def generate(self, texts):
        model = self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        vectors = [
            np.asarray(
                vector,
                dtype="float32"
            )
            for vector in model.embed(
                texts,
                batch_size=EMBED_BATCH_SIZE
            )
        ]
        if not vectors:
            return np.zeros((0, 0), dtype="float32")
        return np.vstack(vectors)


embedding_service = EmbeddingService()