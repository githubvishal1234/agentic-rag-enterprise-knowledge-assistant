from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.services.embedding_service import EmbeddingService


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    """Tests ignore any provider or sign-in settings from a local .env."""
    for name in [
        "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY",
        "POLICY_DESK_PASSWORD", "POLICY_DESK_ADMIN_PASSWORD",
    ]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def docs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "sample"


@pytest.fixture(scope="session")
def embedding_service() -> EmbeddingService:
    # Loading the model is slow, so share one instance across tests.
    return EmbeddingService()


@pytest.fixture
def client(tmp_path, monkeypatch, embedding_service):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    # Sign-in off unless a test turns it on.
    monkeypatch.delenv("POLICY_DESK_PASSWORD", raising=False)
    monkeypatch.delenv("POLICY_DESK_ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(main, "EmbeddingService", lambda: embedding_service)
    monkeypatch.setattr(main, "has_credentials", lambda: True)
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, chunks: {"answer": "stub answer", "citations": []},
    )

    with TestClient(main.app) as client:
        yield client
