import pytest

from app.services.document_loader import load_document


def test_load_pdf(docs_dir):
    documents = load_document(str(docs_dir / "leave_policy.pdf"))

    assert documents

    for document in documents:
        assert document.page_content.strip()
        assert document.metadata["source"] == "leave_policy.pdf"
        assert document.metadata["file_type"] == "pdf"
        assert document.metadata["page"] >= 1


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(str(tmp_path / "missing.txt"))


def test_unsupported_file_type(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(str(path))
