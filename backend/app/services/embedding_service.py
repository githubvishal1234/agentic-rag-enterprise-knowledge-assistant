
from sentence_transformers import SentenceTransformer

from app.models.document import Document


class EmbeddingService:
    """Generate local vector embeddings for document chunks."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        documents: list[Document],
    ) -> list[list[float]]:
        """Convert document text into embedding vectors."""

        if not documents:
            return []

        texts = [document.page_content for document in documents]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert a user query into an embedding vector."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()
