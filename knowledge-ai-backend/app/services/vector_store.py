import json
import os

import faiss
import numpy as np

from app.config import VECTOR_STORE_DIR


INDEX_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "index.faiss"
)

METADATA_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "metadata.json"
)


class VectorStore:

    def __init__(self):

        os.makedirs(
            VECTOR_STORE_DIR,
            exist_ok=True
        )

        self.index = None

        self.metadata = []

        self.load()

    def load(self):

        if os.path.exists(
            INDEX_FILE
        ):

            self.index = faiss.read_index(
                INDEX_FILE
            )

        if os.path.exists(
            METADATA_FILE
        ):

            with open(
                METADATA_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                self.metadata = json.load(
                    file
                )

    def save(self):

        if self.index is not None:

            faiss.write_index(
                self.index,
                INDEX_FILE
            )

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

    def add(
        self,
        embeddings,
        metadata
    ):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        if self.index is None:

            self.index = faiss.IndexFlatIP(
                dimension
            )

        self.index.add(
            embeddings
        )

        self.metadata.extend(
            metadata
        )

        self.save()

    def search(
        self,
        embedding,
        top_k
    ):

        if (
            self.index is None
            or self.index.ntotal == 0
        ):

            return []

        embedding = np.asarray(
            [embedding],
            dtype="float32"
        )

        actual_k = min(
            top_k,
            self.index.ntotal
        )

        scores, indexes = (
            self.index.search(
                embedding,
                actual_k
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indexes[0]
        ):

            if index < 0:
                continue

            item = dict(
                self.metadata[index]
            )

            item["score"] = float(
                score
            )

            results.append(item)

        return results

    def all_metadata(self):
        """Return all stored metadata items (for summary / structure)."""
        return [
            dict(item)
            for item in self.metadata
        ]

    def delete_source(self, filename):
        new_metadata = [
            item for item in self.metadata
            if item.get("source") != filename
        ]

        if len(new_metadata) == len(self.metadata):
            return

        self.metadata = new_metadata

        if not self.metadata:
            self.index = None
            if os.path.exists(INDEX_FILE):
                os.remove(INDEX_FILE)
            self.save()
            return

        from app.services.embeddings import embedding_service
        texts = [item["text"] for item in self.metadata]
        embeddings = np.asarray(
            embedding_service.generate(texts),
            dtype="float32"
        )
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.save()


vector_store = VectorStore()

