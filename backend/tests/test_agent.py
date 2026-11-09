import json
from threading import Lock

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from app.agents import rag_agent
from app.rag import llm
from app.services.ingestion import ingest_file
from app.services.vector_store import FaissVectorStore


class ScriptedModel(BaseChatModel):
    """A chat model that replies with queued AIMessages and records what it
    was sent. Streams text word by word and tool calls as chunks."""

    replies: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)
    bound_tools: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(messages)
        reply = self.replies.pop(0)

        words = reply.content.split(" ") if reply.content else []
        for index, word in enumerate(words):
            piece = word if index == len(words) - 1 else word + " "
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

        if reply.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(reply.tool_calls)
            ]))


def calls(*specs):
    """AIMessage asking for tools: calls(("search_documents", {"query": "x"}), ...)."""
    return AIMessage("", tool_calls=[
        {"name": name, "args": args, "id": f"call_{i}"} for i, (name, args) in enumerate(specs)
    ])


@pytest.fixture
def model(monkeypatch):
    scripted = ScriptedModel()
    monkeypatch.setattr(llm, "get_chat_model", lambda: scripted)
    return scripted


@pytest.fixture
def store(docs_dir, embedding_service):
    store = FaissVectorStore(embedding_service)
    store.add_documents(ingest_file(str(docs_dir / "company_it_policy.txt"), chunk_size=200, chunk_overlap=40))
    return store


@pytest.fixture
def agent(store):
    return rag_agent.build_agent(store, Lock())


def key_of(store, phrase):
    """The citation key of the chunk containing `phrase`."""
    chunk = next(d for d in store.documents if phrase in d.page_content)
    return llm.citation_key(chunk.metadata)


def ask(agent, message, thread_id="thread-1"):
    state = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": thread_id}},
    )
    return rag_agent.summarize_turn(state["messages"])


def tool_messages(request):
    return [m for m in request if isinstance(m, ToolMessage)]


def test_tool_loop_with_citations(agent, model, store):
    key = key_of(store, "lost or stolen")
    model.replies = [
        calls(("search_documents", {"query": "lost laptop"}), ("list_documents", {})),
        AIMessage(f"Report it to the IT department immediately [{key}]."),
    ]

    result = ask(agent, "What do I do if I lose my laptop?")

    assert result["answer"] == "Report it to the IT department immediately."
    assert result["segments"][0]["citations"] == [0]
    assert result["citations"][0]["source"] == "company_it_policy.txt"
    assert "lost or stolen" in result["citations"][0]["cited_text"]
    assert result["tool_calls"] == [
        {"name": "search_documents", "input": {"query": "lost laptop"}},
        {"name": "list_documents", "input": {}},
    ]

    assert model.bound_tools == ["search_documents", "read_document", "list_documents", "calculate"]
    first, second = model.requests
    assert isinstance(first[0], SystemMessage) and "citation key" in first[0].content

    # Both tool results came back before the model's next call.
    search, listing = tool_messages(second)
    assert f"[{key}]" in search.content
    assert listing.content == f"company_it_policy.txt ({len(store.documents)} chunks)"


def test_read_document_and_calculate(agent, model):
    model.replies = [
        calls(("read_document", {"source": "company_it_policy.txt", "start_chunk": 0}),
              ("calculate", {"expression": "22 - 7"})),
        AIMessage("You have 15 days left [company_it_policy.txt#0]."),
    ]

    result = ask(agent, "I took 7 of my 22 days. How many are left?")

    read, calculated = tool_messages(model.requests[1])
    assert read.content.startswith("company_it_policy.txt: chunks 0 to")
    assert "[company_it_policy.txt#0]" in read.content
    assert calculated.content == "15"
    assert result["citations"][0]["chunk_id"] == 0  # read results are citable too


def test_tool_errors_are_reported_to_the_model(agent, model):
    model.replies = [
        calls(("read_document", {"source": "missing.pdf"}),
              ("calculate", {"expression": "__import__('os').system('x')"}),
              ("search_documents", {"query": "   "})),
        AIMessage("Sorry, please rephrase."),
    ]

    assert ask(agent, "?")["answer"] == "Sorry, please rephrase."

    results = tool_messages(model.requests[1])
    assert [m.status for m in results] == ["error", "error", "error"]
    assert "No document named 'missing.pdf'" in results[0].content


def test_conversation_memory(agent, model):
    model.replies = [AIMessage("Hello!"), AIMessage("You said hi.")]

    ask(agent, "Hi")
    result = ask(agent, "What did I just say?")

    assert result["answer"] == "You said hi."
    assert result["tool_calls"] == []
    assert [m.type for m in model.requests[1]] == ["system", "human", "ai", "human"]

    model.replies = [AIMessage("New thread.")]
    ask(agent, "Hi again", thread_id="thread-2")
    assert [m.type for m in model.requests[2]] == ["system", "human"]


def test_endless_tool_calls_end_the_turn_cleanly(agent, model):
    model.replies = [
        calls(("search_documents", {"query": f"try {n}"}))
        for n in range(rag_agent.MAX_TOOL_ROUNDS + 1)
    ]

    result = ask(agent, "Impossible question?")

    assert result["answer"] == rag_agent.GAVE_UP_ANSWER
    assert len(result["tool_calls"]) == rag_agent.MAX_TOOL_ROUNDS + 1

    # The thread still works and every tool call has a result.
    model.replies = [AIMessage("Fine now.")]
    assert ask(agent, "Something simple?")["answer"] == "Fine now."

    history = model.requests[-1]
    call_ids = {c["id"] for m in history if isinstance(m, AIMessage) for c in m.tool_calls}
    result_ids = {m.tool_call_id for m in history if isinstance(m, ToolMessage)}
    assert call_ids == result_ids


def test_empty_reply_becomes_a_message(agent, model):
    model.replies = [AIMessage("")]

    assert ask(agent, "Something declined")["answer"] == rag_agent.NO_ANSWER


def test_calculator_only_does_arithmetic():
    assert rag_agent.evaluate("280 * 3 + 95 * 4") == 1220
    assert rag_agent.evaluate("2,500 / 2") == 1250
    assert rag_agent.evaluate("-(3 + 4) ** 2") == -49

    for bad in ["__import__('os')", "open('x')", "2 ** 1000", "a + 1"]:
        with pytest.raises(ValueError):
            rag_agent.evaluate(bad)


def test_conversation_turns_skips_unanswered_turns():
    messages = [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "One."},
        {"role": "user", "content": "Second?"},  # the LLM call failed here
        {"role": "user", "content": "Third?"},
        {"role": "assistant", "content": "Three."},
    ]
    from langchain_core.messages import convert_to_messages

    turns = rag_agent.conversation_turns(convert_to_messages(messages))

    assert [(t["question"], t["answer"]) for t in turns] == [("First?", "One."), ("Third?", "Three.")]


# --- Through the API ---------------------------------------------------------

def upload_it_policy(client, docs_dir):
    with (docs_dir / "company_it_policy.txt").open("rb") as file:
        client.post("/ingest", files={"file": ("company_it_policy.txt", file)})


def stream_events(client, body):
    response = client.post("/chat/stream", json=body)
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_chat_endpoint_and_history_survive_restart(client, model, docs_dir):
    upload_it_policy(client, docs_dir)
    model.replies = [calls(("search_documents", {"query": "VPN"})), AIMessage("Use the approved VPN.")]

    first = client.post("/chat", json={"message": "How do I work remotely?"}).json()
    assert first["answer"] == "Use the approved VPN."
    assert first["tool_calls"] == [{"name": "search_documents", "input": {"query": "VPN"}}]

    from app import main
    with TestClient(main.app) as restarted:
        history = restarted.get(f"/chat/{first['thread_id']}").json()
        model.replies = [AIMessage("Yes.")]
        restarted.post("/chat", json={"message": "Is that required?", "thread_id": first["thread_id"]})
        assert restarted.get("/chat/unknown-thread").status_code == 404

    assert [t["question"] for t in history["turns"]] == ["How do I work remotely?"]
    assert [m.type for m in model.requests[-1]] == ["system", "human", "ai", "tool", "ai", "human"]
    assert client.post("/chat", json={"message": "  "}).status_code == 422


def test_chat_stream(client, model, docs_dir, store):
    upload_it_policy(client, docs_dir)
    model.replies = [
        calls(("search_documents", {"query": "VPN"})),
        AIMessage("Use the approved VPN [company_it_policy.txt#2]."),
    ]

    events = stream_events(client, {"message": "How do I work remotely?"})

    assert events[0] == {"type": "tool", "name": "search_documents", "input": {"query": "VPN"}}
    assert {e["type"] for e in events[1:-1]} == {"text"}
    assert len(events) > 3  # text arrived in several pieces
    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "Use the approved VPN."


def test_chat_stream_reports_llm_errors(client, monkeypatch):
    def failing():
        raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))

    monkeypatch.setattr(llm, "get_chat_model", failing)

    events = stream_events(client, {"message": "Hi"})

    assert events[-1]["type"] == "error"
    assert "LLM request failed" in events[-1]["detail"]


# --- A real LangChain provider class over a mocked HTTP API -----------------

def test_chat_openai_provider_end_to_end(agent, store, monkeypatch):
    from langchain_openai import ChatOpenAI
    from openai import DefaultHttpx2Client

    key = key_of(store, "lost or stolen")
    responses = [
        {"tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "search_documents", "arguments": '{"query": "lost laptop"}'}}]},
        {"content": f"Tell IT immediately [{key}]."},
    ]
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": None, **responses.pop(0)}
        return httpx2.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}],
        })

    chat = ChatOpenAI(model="test-model", api_key="test", base_url="http://llm.test/v1",
                      http_client=DefaultHttpx2Client(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(llm, "get_chat_model", lambda: chat)

    result = ask(agent, "I lost my laptop")

    assert result["answer"] == "Tell IT immediately."
    assert result["citations"][0]["source"] == "company_it_policy.txt"
    assert [t["function"]["name"] for t in requests[0]["tools"]] == [
        "search_documents", "read_document", "list_documents", "calculate",
    ]
    roles = [m["role"] for m in requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert f"[{key}]" in requests[1]["messages"][3]["content"]
