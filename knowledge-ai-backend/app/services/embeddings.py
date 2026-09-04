import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


class EmbeddingService:

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            print(
                f"Loading embedding model: "
                f"{EMBEDDING_MODEL}"
            )
            self._model = SentenceTransformer(
                EMBEDDING_MODEL
            )
        return self._model

    def generate(self, texts):
        model = self._load_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings


embedding_service = EmbeddingService()
