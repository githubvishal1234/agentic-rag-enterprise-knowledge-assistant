from langchain_core.messages import AIMessage

from app.rag import generator, llm

PASSAGES = {
    "leave.txt#0": {"text": "Employees get 22 days.", "metadata": {"source": "leave.txt", "chunk_id": 0}},
    "it.pdf#3": {"text": "Use the VPN.", "metadata": {"source": "it.pdf", "chunk_id": 3, "page": 2}},
}


def test_parse_citations():
    answer, citations, segments = llm.parse_citations(
        "You get 22 days [leave.txt#0]. Use the VPN [it.pdf#3, leave.txt#0][nope.txt#9]. Ask HR.",
        PASSAGES,
    )

    assert answer == "You get 22 days. Use the VPN. Ask HR."
    assert [c["source"] for c in citations] == ["leave.txt", "it.pdf"]
    assert citations[1]["title"] == "it.pdf, page 2 (chunk 3)"
    assert citations[1]["cited_text"] == "Use the VPN."
    assert segments == [
        {"text": "You get 22 days", "citations": [0]},
        {"text": ". Use the VPN", "citations": [1, 0]},  # unknown key dropped
        {"text": ". Ask HR.", "citations": []},
    ]


def test_parse_citations_full_width_brackets():
    # Real gpt-oss-120b output: full-width brackets and narrow no-break spaces.
    answer, citations, segments = llm.parse_citations(
        "You get up to 22 days 【leave.txt#0】. VPN too【it.pdf#3，leave.txt#0】.",
        PASSAGES,
    )

    assert answer == "You get up to 22 days. VPN too."
    assert [c["source"] for c in citations] == ["leave.txt", "it.pdf"]
    assert [s["citations"] for s in segments] == [[0], [1, 0], []]


def test_parse_citations_without_keys():
    assert llm.parse_citations("No sources.", PASSAGES) == (
        "No sources.", [], [{"text": "No sources.", "citations": []}]
    )


def test_format_passages_labels_each_chunk():
    text = llm.format_passages(list(PASSAGES.values()))

    assert text.startswith("[leave.txt#0] leave.txt (chunk 0)\nEmployees get 22 days.")
    assert "[it.pdf#3] it.pdf, page 2 (chunk 3)" in text


def test_has_credentials(monkeypatch):
    assert not llm.has_credentials()  # anthropic, no key

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert llm.has_credentials()

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert not llm.has_credentials()  # needs a model

    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    assert not llm.has_credentials()  # and a key or a URL

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    assert llm.has_credentials()  # Ollama: URL, no key


def test_chat_model_per_provider(monkeypatch):
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    claude = llm.get_chat_model()
    assert isinstance(claude, ChatAnthropic) and claude.model == "claude-opus-5-5"

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    other = llm.get_chat_model()
    assert isinstance(other, ChatOpenAI) and other.model_name == "gemini-2.5-flash"


def test_query_answer_with_citations(monkeypatch):
    sent = []

    class OneReply:
        def invoke(self, messages):
            sent.append(messages)
            return AIMessage("You get 22 days [leave.txt#0].")

    monkeypatch.setattr(llm, "get_chat_model", lambda: OneReply())
    chunks = list(PASSAGES.values())

    result = generator.generate_answer("How much leave?", chunks)

    assert result["answer"] == "You get 22 days."
    assert result["citations"] == [{
        "cited_text": "Employees get 22 days.", "title": "leave.txt (chunk 0)",
        "source": "leave.txt", "chunk_id": 0,
    }]
    system, question = sent[0]
    assert "citation key" in system.content
    assert "[leave.txt#0] leave.txt (chunk 0)" in question.content
