
from pathlib import Path

from app.models.document import Document
from app.services.document_loader import load_document
from app.services.chunker import chunk_documents


def ingest_file(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Load, validate, and chunk a supported document."""

    path = Path(file_path)

    # Step 1: Load the document.
    documents = load_document(str(path))

    # Step 2: Validate extracted text.
    valid_documents = [
        document
        for document in documents
        if document.page_content and document.page_content.strip()
    ]

    if not valid_documents:
        raise ValueError(
            f"No readable text found in document: {path.name}"
        )

    # Step 3: Split documents into chunks.
    chunks = chunk_documents(
        valid_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(
            f"No chunks were generated for document: {path.name}"
        )

    return chunks
