# Agentic RAG Enterprise Knowledge Assistant

[![CI](https://github.com/githubvishal1234/agentic-rag-enterprise-knowledge-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/githubvishal1234/agentic-rag-enterprise-knowledge-assistant/actions/workflows/ci.yml)

An agentic RAG assistant that answers questions over enterprise documents
(policies, handbooks) with grounded, cited answers. A LangGraph agent decides
when to retrieve, which tool to call and how to proceed across multiple steps,
instead of following a fixed retrieve-then-generate chain.

Built with **Python, LangChain, LangGraph, FAISS, embeddings and tool
calling**, served by FastAPI with a React frontend. Works with Claude or any
OpenAI-compatible model (OpenAI, Gemini, Groq, or Ollama running locally).

**→ Full setup and usage guide: [RUNNING.md](RUNNING.md)**  
**→ Improvement plan: [ROADMAP.md](ROADMAP.md)**

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
npm install --prefix frontend
copy .env.example .env          # then add your API key, see RUNNING.md section 3
npm run build --prefix frontend
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --port 8000
```

Open http://localhost:8000, add documents (samples in `data/sample` and
`data/eval_corpus`), and ask a question. Or with Docker:

```powershell
docker build -t knowledge-assistant .
docker run -p 8000:8000 --env-file .env -v knowledge-assistant-data:/app/data knowledge-assistant
```

## How it works

```
upload ─► load (txt/pdf/docx) ─► LangChain Documents ─► chunk ─► embed ─► FAISS + BM25
                                                                              │
question ─► LangGraph agent (LangChain chat model, tool calling)              │
              │   ▲                                                           │
              ▼   │   ToolNode: search_documents ── hybrid search ────────────┘
           tools ─┘            read_document, list_documents, calculate
              │
              ▼
           answer with citations [file#chunk] ─► parsed into cited segments ─► web app
```

**Ingestion pipeline.** Text, PDF and Word files are loaded into LangChain
`Document`s, split with LangChain's `RecursiveCharacterTextSplitter` (500
characters, 100 overlap), embedded locally with `all-MiniLM-L6-v2`
(sentence-transformers) and indexed in FAISS, with a BM25 keyword index over
the same chunks.

**LangGraph workflow.** The graph manages retrieval, tool execution,
response generation and workflow state:

| Node | Role |
|---|---|
| `model` | LangChain chat model with the tools bound; decides the next step |
| `tools` | LangGraph `ToolNode` running the chosen tools in parallel |
| `give_up` | Ends a turn cleanly if the agent exceeds 6 tool rounds |

State is LangGraph's `MessagesState`, persisted in SQLite so conversations
survive restarts and follow-up questions keep context.

**Tools** (LangChain `@tool`), chosen by the agent per question:

| Tool | Used for |
|---|---|
| `search_documents` | Hybrid search: FAISS vectors + BM25 keywords, merged with Reciprocal Rank Fusion |
| `read_document` | Reading consecutive passages when a search hit is cut off |
| `list_documents` | Questions about what the knowledge base contains |
| `calculate` | Arithmetic such as remaining leave days or trip costs (safe evaluator, no `eval`) |

**Grounded answers.** Every passage the model sees carries a citation key
such as `[leave_policy.txt#3]`. The agent is instructed to answer only from
tool results and cite each claim; the keys are parsed into citations shown
beside the answer. When the documents don't cover a question, the agent says
so instead of guessing.

## Features

- Agentic RAG with multi-step tool calling (one search per part of a
  multi-part question, follow-up reads, calculations).
- Hybrid retrieval with Reciprocal Rank Fusion and stopword filtering.
- Sentence-level citations, shown as margin notes with the quoted passage.
- Streaming: tool calls and answer text appear as the agent works.
- Conversation memory persisted in SQLite; recent conversations in the UI.
- Optional sign-in with employee and admin roles.
- Pluggable LLM through LangChain: Claude, OpenAI, Gemini, Groq or Ollama.
- Docker image, GitHub Actions CI, 58 automated tests.
- Evaluation sets for retrieval and answer quality.

| Path | Contents |
|---|---|
| `backend/app/agents/rag_agent.py` | LangGraph agent, tools and graph |
| `backend/app/rag/llm.py` | LangChain chat model selection and citation handling |
| `backend/app/rag/generator.py` | One-shot RAG answers (`/query`) |
| `backend/app/services/` | Loading, chunking, embeddings, FAISS store, BM25 |
| `backend/app/main.py` | FastAPI endpoints |
| `frontend/` | React web app |
| `evaluation/` | Eval cases and runners |
| `docs/` | BRD and FRD (Word) |

## Results

On 26 answerable questions over the eval corpus (lookups, paraphrases,
exact terms such as form codes, and multi-part questions), the passage
holding the answer is in the top 4 search results for:

| Retrieval | Found |
|---|---|
| Vector search only | 24 / 26 |
| Hybrid (vector + BM25) | 24 / 26 |
| Hybrid + stopword filtering | **26 / 26** |

The answer-quality eval (facts correct, right source cited, no invented
answers to unanswerable questions) runs through the real endpoints with any
configured model: see [RUNNING.md, section 9](RUNNING.md#9-evaluations).

## Tests

58 backend tests (scripted LLM responses, no key needed), including one
through LangChain's real `ChatOpenAI` class against a mocked API, plus a
frontend build and a Docker build-and-run check on every push:

```powershell
cd backend
..\.venv\Scripts\python -m pytest
```
