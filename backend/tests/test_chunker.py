from app.services.document_loader import load_document
from app.services.chunker import chunk_documents


def test_chunker(docs_dir):
    documents = load_document(str(docs_dir / "company_it_policy.txt"))

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        chunk_overlap=40,
    )

    assert len(chunks) > 1, "Expected multiple chunks"

    for index, chunk in enumerate(chunks):
        assert chunk.page_content.strip(), f"Chunk {index} is empty"
        assert "source" in chunk.metadata
        assert "file_type" in chunk.metadata
        assert "chunk_id" in chunk.metadata

    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"
