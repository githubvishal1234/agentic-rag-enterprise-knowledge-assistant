
import faiss
import numpy as np
import json
from collections import defaultdict
from pathlib import Path

from app.models.document import Document
from app.services.bm25 import BM25
from app.services.embedding_service import EmbeddingService

# Standard Reciprocal Rank Fusion constant; damps the weight of top ranks.
RRF_K = 60


class FaissVectorStore:
    """Store document embeddings and retrieve similar chunks."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.index = None
        self.documents: list[Document] = []
        self.keyword_index = BM25([])

    def _rebuild_keyword_index(self) -> None:
        # ponytail: rebuilt from scratch on every add/load, fine for small
        # corpora; make BM25 incremental if ingestion gets slow.
        self.keyword_index = BM25(
            [document.page_content for document in self.documents]
        )

    
    def save(self, directory: str) -> None:
        """Save the FAISS index and document metadata to disk."""
        if self.index is None or not self.documents:
            raise ValueError("Cannot save an empty vector store")

        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(save_path / "index.faiss"),
        )

        document_data = [
            {
                "text": document.page_content,
                "metadata": document.metadata,
            }
            for document in self.documents
        ]

        with (save_path / "documents.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(document_data, file, ensure_ascii=False, indent=2)

    def load(self, directory: str) -> None:
        """Load the FAISS index and document metadata from disk."""
        load_path = Path(directory)
        index_file = load_path / "index.faiss"
        documents_file = load_path / "documents.json"

        if not index_file.exists() or not documents_file.exists():
            raise FileNotFoundError(
                f"Saved vector store not found in: {load_path}"
            )

        loaded_index = faiss.read_index(str(index_file))

        with documents_file.open("r", encoding="utf-8") as file:
            document_data = json.load(file)

        loaded_documents = [
            Document(
                page_content=item["text"],
                metadata=item["metadata"],
            )
            for item in document_data
        ]

        if loaded_index.ntotal != len(loaded_documents):
            raise ValueError(
                "FAISS index and document metadata counts do not match"
            )

        self.index = loaded_index
        self.documents = loaded_documents
        self._rebuild_keyword_index()


    def add_documents(self, documents: list[Document]) -> None:
        """Embed documents and add their vectors to the index."""

        if not documents:
            return

        embeddings = self.embedding_service.embed_documents(documents)
        vectors = np.asarray(embeddings, dtype=np.float32)

        if self.index is None:
            dimension = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(vectors)
        self.documents.extend(documents)
        self._rebuild_keyword_index()

    def remove_source(self, source: str) -> int:
        """Remove every chunk from one source file; return how many."""

        indices = [
            index
            for index, document in enumerate(self.documents)
            if document.metadata["source"] == source
        ]

        if not indices:
            return 0

        # IndexFlat compacts in place and keeps the remaining vectors in
        # order, so they stay aligned with self.documents.
        self.index.remove_ids(np.asarray(indices, dtype=np.int64))
        self.documents = [
            document
            for document in self.documents
            if document.metadata["source"] != source
        ]

        if not self.documents:
            self.index = None

        self._rebuild_keyword_index()
        return len(indices)

    def _validate_search(self, query: str, k: int) -> None:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be greater than zero")

    def _vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (document index, cosine score) pairs, best first."""

        query_embedding = self.embedding_service.embed_query(query)
        query_vector = np.asarray(
            [query_embedding],
            dtype=np.float32,
        )

        k = min(k, len(self.documents))

        scores, indices = self.index.search(query_vector, k)

        return [
            (int(index), float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def _result(self, index: int, score: float) -> dict:
        return {
            "text": self.documents[index].page_content,
            "metadata": self.documents[index].metadata,
            "score": score,
        }

    def similarity_search(
        self,
        query: str,
        k: int = 3,
    ) -> list[dict]:
        """Return the top-k most similar document chunks."""

        self._validate_search(query, k)

        if self.index is None or not self.documents:
            return []

        return [
            self._result(index, score)
            for index, score in self._vector_search(query, k)
        ]

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[dict]:
        """Return the top-k chunks by fusing vector and BM25 rankings.

        Each chunk scores 1 / (RRF_K + rank) in every ranking it appears
        in, so exact keyword matches (names, codes, acronyms) can surface
        even when their embeddings are not the closest.
        """

        self._validate_search(query, k)

        if self.index is None or not self.documents:
            return []

        candidates = max(k, 20)

        vector_ranking = [
            index for index, _ in self._vector_search(query, candidates)
        ]

        keyword_scores = self.keyword_index.scores(query)
        keyword_ranking = sorted(
            (
                index
                for index, score in enumerate(keyword_scores)
                if score > 0
            ),
            key=lambda index: keyword_scores[index],
            reverse=True,
        )[:candidates]

        fused_scores = defaultdict(float)

        for ranking in (vector_ranking, keyword_ranking):
            for rank, index in enumerate(ranking, start=1):
                fused_scores[index] += 1 / (RRF_K + rank)

        best = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]

        return [self._result(index, fused_scores[index]) for index in best]
