import pytest

from app.models.document import Document
from app.services.chunker import chunk_documents


def make_document(text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"source": "test.txt", "file_type": "txt"},
    )


def test_short_document():
    document = make_document("This is a short document.")

    chunks = chunk_documents(
        [document],
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) == 1
    assert chunks[0].page_content == document.page_content


def test_empty_document():
    assert chunk_documents([make_document("")]) == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_documents(
            [make_document("Example text.")],
            chunk_size=0,
            chunk_overlap=0,
        )


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_documents(
            [make_document("Example text.")],
            chunk_size=100,
            chunk_overlap=100,
        )
