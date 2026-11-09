import os
import secrets
import shutil
import sqlite3
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

import json

import anthropic
from dotenv import load_dotenv

# Keys and settings from the project's .env (see .env.example). Variables
# already set in the environment win, so Docker and CI can override them.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
import openai
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.sessions import SessionMiddleware

from app.core import auth
from app.core.auth import require_admin, require_user
from app.agents.rag_agent import build_agent, conversation_turns, summarize_turn
from app.rag.generator import generate_answer
from app.rag.llm import has_credentials
from app.services.embedding_service import EmbeddingService
from app.services.ingestion import ingest_file
from app.services.vector_store import FaissVectorStore

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
SUPPORTED_TYPES = {".txt", ".pdf", ".docx"}

# ponytail: one lock for every store read/write; ingesting a large file
# blocks queries until it finishes. Use a read/write lock if that hurts.
store_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = FaissVectorStore(EmbeddingService())

    if (DATA_DIR / "vectorstore" / "index.faiss").exists():
        store.load(str(DATA_DIR / "vectorstore"))

    # Conversations persist in SQLite so threads survive restarts. FastAPI
    # runs handlers in worker threads; SqliteSaver locks the connection.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        DATA_DIR / "conversations.sqlite", check_same_thread=False
    )

    app.state.store = store
    app.state.agent = build_agent(store, store_lock, SqliteSaver(connection))
    yield
    connection.close()


app = FastAPI(
    title="Agentic RAG Enterprise Knowledge Assistant",
    version="0.1.0",
    description="Enterprise AI knowledge and data assistant",
    lifespan=lifespan,
)

# Signed session cookie (HTTP-only). Without POLICY_DESK_SECRET a random key
# is used, so everyone is signed out whenever the server restarts.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("POLICY_DESK_SECRET") or secrets.token_urlsafe(32),
    same_site="lax",
    https_only=os.environ.get("POLICY_DESK_HTTPS_ONLY") == "1",
    max_age=8 * 60 * 60,
)
app.include_router(auth.router)


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=20)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1)
    thread_id: str | None = None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "agentic-rag-enterprise-knowledge-assistant",
        "llm_configured": has_credentials(),
        "auth_required": auth.auth_required(),
    }


def require_credentials() -> None:
    if not has_credentials():
        raise HTTPException(
            503,
            "The server has no LLM configured. Set ANTHROPIC_API_KEY, or "
            "LLM_PROVIDER=openai with LLM_MODEL (see RUNNING.md), and "
            "restart the server.",
        )


@app.get("/documents", dependencies=[Depends(require_user)])
def documents():
    with store_lock:
        counts = Counter(
            document.metadata["source"]
            for document in app.state.store.documents
        )

    return [
        {"source": source, "chunks": chunks}
        for source, chunks in sorted(counts.items())
    ]


@app.post("/ingest", dependencies=[Depends(require_admin)])
def ingest(file: UploadFile):
    # Keep only the base name so uploads cannot escape the documents folder.
    file_name = Path(file.filename or "").name

    if Path(file_name).suffix.lower() not in SUPPORTED_TYPES:
        raise HTTPException(400, "Supported types: .txt, .pdf, .docx")

    store = app.state.store

    with store_lock:
        if any(
            document.metadata["source"] == file_name
            for document in store.documents
        ):
            raise HTTPException(409, f"{file_name} is already ingested")

        path = DATA_DIR / "documents" / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.file.read())

        try:
            chunks = ingest_file(str(path))
        except Exception as error:
            path.unlink()
            raise HTTPException(422, f"Could not read {file_name}: {error}")

        store.add_documents(chunks)
        store.save(str(DATA_DIR / "vectorstore"))

    return {"source": file_name, "chunks": len(chunks)}


@app.delete("/documents/{source}", dependencies=[Depends(require_admin)])
def delete_document(source: str):
    store = app.state.store

    with store_lock:
        removed = store.remove_source(source)

        if not removed:
            raise HTTPException(404, f"{source} is not in the knowledge base")

        vectorstore_dir = DATA_DIR / "vectorstore"
        if store.documents:
            store.save(str(vectorstore_dir))
        else:
            shutil.rmtree(vectorstore_dir, ignore_errors=True)

        # Base name only, as in /ingest, so this can't reach outside the folder.
        (DATA_DIR / "documents" / Path(source).name).unlink(missing_ok=True)

    return {"source": source, "removed_chunks": removed}


@app.post("/query", dependencies=[Depends(require_user)])
def query(request: QueryRequest):
    with store_lock:
        chunks = app.state.store.hybrid_search(
            request.question,
            k=request.k,
        )

    if not chunks:
        return {
            "answer": "No documents have been ingested yet.",
            "citations": [],
            "sources": [],
        }

    require_credentials()

    try:
        result = generate_answer(request.question, chunks)
    except (anthropic.APIError, openai.APIError) as error:
        raise HTTPException(502, f"LLM request failed: {error}")

    return {
        **result,
        "sources": [
            {**chunk["metadata"], "score": chunk["score"]}
            for chunk in chunks
        ],
    }


@app.post("/chat", dependencies=[Depends(require_user)])
def chat(request: ChatRequest):
    require_credentials()
    thread_id = request.thread_id or str(uuid4())

    try:
        state = app.state.agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]},
            {"configurable": {"thread_id": thread_id}},
        )
    except (anthropic.APIError, openai.APIError) as error:
        raise HTTPException(502, f"LLM request failed: {error}")
    except GraphRecursionError:
        raise HTTPException(500, "The agent took too many steps")

    return {"thread_id": thread_id, **summarize_turn(state["messages"])}



@app.post("/chat/stream", dependencies=[Depends(require_user)])
def chat_stream(request: ChatRequest):
    """Like /chat, as Server-Sent Events while the agent works:
    {"type": "tool", "name", "input"} for each tool call, {"type": "text",
    "delta"} for answer text, then {"type": "done", ...same body as /chat}
    or {"type": "error", "detail"}.
    """

    require_credentials()
    thread_id = request.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id, "stream_text": True}}

    def event(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    def events():
        try:
            for mode, chunk in app.state.agent.stream(
                {"messages": [{"role": "user", "content": request.message}]},
                config,
                stream_mode=["custom", "updates"],
            ):
                if mode == "custom":
                    yield event(chunk)
                elif "model" in chunk:
                    for message in chunk["model"]["messages"]:
                        for call in message.tool_calls:
                            yield event({
                                "type": "tool",
                                "name": call["name"],
                                "input": call["args"],
                            })

            messages = app.state.agent.get_state(config).values["messages"]
            yield event({
                "type": "done",
                "thread_id": thread_id,
                **summarize_turn(messages),
            })
        except (anthropic.APIError, openai.APIError) as error:
            yield event({"type": "error", "detail": f"LLM request failed: {error}"})
        except GraphRecursionError:
            yield event({"type": "error", "detail": "The agent took too many steps"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Stop proxies from buffering the stream.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/chat/{thread_id}", dependencies=[Depends(require_user)])
def chat_history(thread_id: str):
    state = app.state.agent.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    if not messages:
        raise HTTPException(404, "Conversation not found")

    return {"thread_id": thread_id, "turns": conversation_turns(messages)}

# Serve the built frontend (npm run build) from the same server. Mounted
# last so the API routes above take precedence.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
