# Running the Agentic RAG Enterprise Knowledge Assistant

A step-by-step guide to installing, configuring and running this project:
the FastAPI backend, the web app, Docker, sign-in, tests and
evaluations. Commands are for **Windows PowerShell**; macOS/Linux
equivalents are noted where they differ.

**Contents**

1. [What you need](#1-what-you-need)
2. [Install (once)](#2-install-once)
3. [Choose an AI model and add your key](#3-choose-an-ai-model-and-add-your-key)
4. [Run the app](#4-run-the-app)
5. [Use it](#5-use-it)
6. [Optional: require sign-in](#6-optional-require-sign-in)
7. [Run with Docker](#7-run-with-docker)
8. [Tests](#8-tests)
9. [Evaluations](#9-evaluations)
10. [API reference](#10-api-reference)
11. [Where data is stored](#11-where-data-is-stored)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What you need

| Tool | Version | Check with |
|---|---|---|
| Python | 3.11 | `python --version` |
| Node.js | 20 or newer (24 recommended) | `node --version` |
| Git | any | `git --version` |
| An AI model | one of the options in [section 3](#3-choose-an-ai-model-and-add-your-key) | |
| Docker (optional) | any recent | `docker --version` |

About 3 GB of disk space for Python packages (PyTorch) and the embedding
model. No GPU is needed.

## 2. Install (once)

Get the code (skip if you already have the folder):

```powershell
git clone https://github.com/githubvishal1234/agentic-rag-policy-desk.git
cd agentic-rag-policy-desk
```

Create a Python virtual environment and install the backend:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
```

> macOS/Linux: `python3 -m venv .venv` then `.venv/bin/pip install -r backend/requirements.txt`.
> On Linux, install CPU-only PyTorch first to avoid a multi-GB GPU build:
> `.venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`

Install the frontend:

```powershell
npm install --prefix frontend
```

Create your settings file from the template:

```powershell
copy .env.example .env
```

> macOS/Linux: `cp .env.example .env`

## 3. Choose an AI model and add your key

Everything except answering questions works without a model: uploading
documents, search, the web app. To get answers, open `.env` in a text editor
and fill in **one** option. Each option is already written out in
`.env.example`; uncomment its lines (remove the leading `#`) and paste your key.

| Option | Cost | Key from | What to set in `.env` |
|---|---|---|---|
| **A. Claude** (default, best citations) | Paid | https://platform.claude.com | `ANTHROPIC_API_KEY=sk-ant-...` |
| **B. OpenAI** | Paid | https://platform.openai.com/api-keys | `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-5-mini`, `LLM_API_KEY=sk-...` |
| **C. Google Gemini** | Free tier | https://aistudio.google.com/apikey | `LLM_PROVIDER=openai`, `LLM_MODEL=gemini-2.5-flash`, `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, `LLM_API_KEY=...` |
| **D. Groq** | Free tier | https://console.groq.com/keys | `LLM_PROVIDER=openai`, `LLM_MODEL=llama-3.3-70b-versatile`, `LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_API_KEY=gsk_...` |
| **E. Ollama** (local) | Free | none, install from https://ollama.com | `LLM_PROVIDER=openai`, `LLM_MODEL=qwen2.5:7b`, `LLM_BASE_URL=http://localhost:11434/v1` |

For Ollama, download the model once before starting the backend:

```powershell
ollama pull qwen2.5:7b
```

How the options differ:

- Every provider runs through LangChain (`ChatAnthropic` for Claude,
  `ChatOpenAI` for the rest), so all of them get the same tools, streaming
  and citations.
- The model cites passages by key, such as `[leave_policy.txt#3]`; the app
  turns those into markers and margin notes. Capable models do this
  reliably; small local models sometimes skip citations or use the tools
  less well.
- Model names change over time. If one is rejected, check the provider's
  current model list and update `LLM_MODEL`.

The backend reads `.env` when it starts, so **restart it after editing**.
Settings can also be passed as normal environment variables, which take
priority over `.env`:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## 4. Run the app

### Option 1: development (two terminals, live reload)

**Terminal 1, the backend** (from the project folder):

```powershell
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --port 8000 --reload
```

Wait for `Application startup complete`. The first start downloads the
embedding model (about 90 MB) and takes a minute.

**Terminal 2, the frontend:**

```powershell
npm run dev --prefix frontend
```

Open **http://localhost:5173**. The dev server forwards API calls to port 8000.

### Option 2: one server

Build the web app once, then run only the backend; it serves the app too:

```powershell
npm run build --prefix frontend
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --port 8000
```

Open **http://localhost:8000**. Rebuild after changing anything in `frontend/`.

To let other computers on your network reach it, add `--host 0.0.0.0` and
open `http://<your-ip>:8000` (allow port 8000 in Windows Firewall), and
turn on sign-in first ([section 6](#6-optional-require-sign-in)).

## 5. Use it

1. **Add documents**: click *Add documents* in the left panel, or drag files
   onto it. Text (`.txt`), PDF and Word (`.docx`) files work. Sample files
   are in `data\sample` and `data\eval_corpus` (11 fictional policy documents).
2. **Ask a question**: type in the box at the bottom and press Enter
   (Shift+Enter for a new line).
3. **Check the answer**: numbered markers in the answer match the quoted
   passages in the margin. Hover a marker to highlight its passage.
4. **Follow up**: the conversation remembers earlier questions. *New
   conversation* starts fresh; *Recent conversations* reopens old ones.
5. **Remove a document**: hover it in the list and click *Remove*.

If a red notice says answers are off, the backend has no AI model set:
see [section 3](#3-choose-an-ai-model-and-add-your-key).

## 6. Optional: require sign-in

Sign-in is off by default. To turn it on, set passwords in `.env`:

```ini
POLICY_DESK_PASSWORD=choose-an-employee-password
POLICY_DESK_ADMIN_PASSWORD=choose-an-admin-password
POLICY_DESK_SECRET=any-long-random-text
```

- **Employees** can ask questions and read conversations.
- **Admins** can also add and remove documents.
- `POLICY_DESK_SECRET` signs the session cookie. Without it, everyone is
  signed out whenever the server restarts.
- Behind HTTPS, also set `POLICY_DESK_HTTPS_ONLY=1`.

Sessions last 8 hours. Wrong passwords are slowed down but there is no
lockout, so put the app behind a VPN or a reverse proxy before exposing it
to the internet.

## 7. Run with Docker

Build the image (includes the backend, the built web app and the embedding
model):

```powershell
docker build -t knowledge-assistant .
```

Run it with your `.env` settings and a volume that keeps documents,
the search index and conversations:

```powershell
docker run -p 8000:8000 --env-file .env -v knowledge-assistant-data:/app/data knowledge-assistant
```

Open **http://localhost:8000**.

Using Ollama from inside Docker: `localhost` means the container itself, so
set `LLM_BASE_URL=http://host.docker.internal:11434/v1` in `.env`.

## 8. Tests

The 58 backend tests use fake AI responses, so they need no key and cost
nothing:

```powershell
cd backend
..\.venv\Scripts\python -m pytest
cd ..
```

GitHub Actions runs the same tests, builds the web app and builds and
starts the Docker image on every push (see the CI badge in the README).

## 9. Evaluations

`evaluation/cases.jsonl` holds 31 questions over the fictional company
documents in `data/eval_corpus/`, each with the passage that answers it
and the facts the answer must contain. `evaluation/cases.md` lists them.

### Retrieval eval (free, no key)

Checks whether search finds each question's passage in its top 4 results:

```powershell
.venv\Scripts\python evaluation\run_retrieval_eval.py --variant baseline --method vector
.venv\Scripts\python evaluation\run_retrieval_eval.py --variant v2 --method hybrid
```

Current result: vector search 24/26, hybrid search 26/26.

### Answer eval (uses your AI model, costs money)

Runs the questions through the real app and checks each answer contains
the expected facts and cites the right document. For the 5 questions the
documents can't answer, it checks the model says so instead of inventing
an answer. Start with a 5-question pilot:

```powershell
.venv\Scripts\python evaluation\run_answer_eval.py --variant baseline --app query --limit 5
```

With Claude it prints the measured cost; multiply by about 6 for the
whole set. Then run everything, plain RAG (`query`) and the agent (`chat`):

```powershell
.venv\Scripts\python evaluation\run_answer_eval.py --variant baseline --app query
.venv\Scripts\python evaluation\run_answer_eval.py --variant v1 --app chat
```

Results land in `.claude/hillclimb/answers/`. If a run is interrupted, run
the same command again: finished questions are skipped. Token usage is
recorded for every provider; cost is calculated for Claude models and shows
as "not measured" for others (add their prices to `PRICES` in
`evaluation/run_answer_eval.py` to see it).

## 10. API reference

Interactive docs: **http://localhost:8000/docs**

| Endpoint | What it does | Who |
|---|---|---|
| `GET /health` | Status; whether a model and sign-in are configured | anyone |
| `POST /login`, `POST /logout`, `GET /me` | Sign-in | anyone |
| `GET /documents` | Files in the knowledge base | employee |
| `POST /ingest` | Upload a `.txt`, `.pdf` or `.docx` file | admin |
| `DELETE /documents/{name}` | Remove a file | admin |
| `POST /query` | One-shot answer: `{"question", "k"}` | employee |
| `POST /chat` | Agent answer: `{"message", "thread_id"?}` | employee |
| `POST /chat/stream` | Same as `/chat`, streamed as Server-Sent Events | employee |
| `GET /chat/{thread_id}` | A conversation so far | employee |

With sign-in off, everyone counts as admin.

Example without the web app:

```powershell
curl.exe -F "file=@data/sample/company_it_policy.txt" http://localhost:8000/ingest
curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{\"message\": \"What are the password rules?\"}'
```

## 11. Where data is stored

| Path | Contents |
|---|---|
| `data/documents/` | Uploaded files |
| `data/vectorstore/` | The search index (FAISS + chunk texts) |
| `data/conversations.sqlite` | Conversations |
| `.env` | Your keys and settings |

All of these are in `.gitignore`. To start over with an empty knowledge
base, stop the server and delete `data/documents`, `data/vectorstore` and
`data/conversations.sqlite`.

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| "Answers are off until the server has an AI model configured" | Fill in one option in `.env` ([section 3](#3-choose-an-ai-model-and-add-your-key)) and restart the backend. |
| "Can't reach the server" in the web app | Start the backend (section 4) and reload the page. |
| `LLM request failed: ... 401` | The API key is wrong or expired. |
| `LLM request failed: ... 404` / model not found | `LLM_MODEL` isn't a model your provider offers; check its model list. |
| `LLM request failed: Connection error` | No internet, or with Ollama: start it (`ollama serve`) and check `LLM_BASE_URL`. |
| Answers without citation markers | The model ignored the citation instructions; use a larger model. |
| First backend start is slow or fails offline | It downloads the embedding model once; connect to the internet for the first start. |
| `npm ci` fails with "Missing ... from lock file" | Run `npm install --prefix frontend`; the project's `frontend/.npmrc` keeps installs consistent. |
| Port 8000 or 5173 already in use | Add `--port 8001` to the backend command (and update `frontend/vite.config.js`), or stop the other program. |
| Signed out after every restart | Set `POLICY_DESK_SECRET` in `.env`. |
| `docker build` is slow | It downloads PyTorch and the embedding model; later builds reuse the cache. |
