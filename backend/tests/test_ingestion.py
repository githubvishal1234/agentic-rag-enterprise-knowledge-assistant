import pytest

from app.services.ingestion import ingest_file


@pytest.mark.parametrize(
    "file_name",
    [
        "company_it_policy.txt",
        "leave_policy.pdf",
        "sample_policy.docx",
    ],
)
def test_ingest_file(docs_dir, file_name):
    chunks = ingest_file(
        str(docs_dir / file_name),
        chunk_size=200,
        chunk_overlap=40,
    )

    assert chunks, f"No chunks generated for {file_name}"

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert chunk.metadata["source"] == file_name
        assert "chunk_id" in chunk.metadata
        assert "start_index" in chunk.metadata
