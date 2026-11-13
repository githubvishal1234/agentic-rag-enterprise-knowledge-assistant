# Project audit (November 2026)

Audit of the Agentic RAG Enterprise Knowledge Assistant before the product
transformation on the `agency-transformation` branch. Findings come from the
code as it stands on `main`.

Measured baseline: 61 backend tests passing; retrieval 26/26 (hybrid);
answers 31/31 for `/query` and `/chat` on Groq `openai/gpt-oss-120b`,
including 5/5 correct declines.

## Frontend

`frontend/src/App.jsx` (751 lines) and `styles.css` (728 lines).

| Finding | Impact |
|---|---|
| One file, one screen, no router | No dashboard, documents page, settings or admin view exists |
| Citations show the quoted chunk only | No source document preview; can't open the document at the passage |
| No copy or feedback on answers | No signal about answer quality from users |
| Upload has no progress, search or filter; delete uses `window.confirm` | Weak document management |
| `STARTER_QUESTIONS` hardcoded to the sample corpus | Wrong suggestions for any other document set |
| No lint, type checks or frontend tests; CI only builds | Regressions go unnoticed |

## Backend

`backend/app/main.py` (303 lines) holds every endpoint.

| Finding | Impact |
|---|---|
| `api/`, `schemas/`, `tools/` are empty packages | Routes, models and startup are all in one file |
| Upload reads the whole file into memory, no size limit (`main.py:152`) | A large upload can exhaust memory |
| No rate limiting on `/login`, `/chat`, `/chat/stream`, `/ingest` | Password guessing and cost abuse are only slowed, not stopped |
| Conversations have no owner | Any signed-in user with a `thread_id` can read that conversation |
| Sign-in is two shared passwords | No individual accountability |
| Nothing recorded per request (latency, tokens, tool calls, errors) | A dashboard has no real data; no structured logging |
| Documents have no metadata beyond name and chunk count | No size, upload time, uploader or status to show |
| Ingestion is synchronous under one global lock | Large uploads block questions |
| Env vars still named `POLICY_DESK_*` | Leftover from the old product name |
| Conversations are never deleted | The SQLite file grows without limit |

## AI and RAG

| Finding | Evidence |
|---|---|
| Agent over-searches unanswerable questions | 7 tool calls, ~12.8K tokens, ~19x plain RAG (answer eval v1) |
| A prompt-only fix does not work | v2 experiment: no effect, reverted (`.claude/hillclimb/answers/v2/change.md`) |
| The model does not always use `calculate` | Seen in browser testing; answers were still correct |
| Citations are prompt-based keys | Three format variants found and handled in `llm.parse_citations` |
| DOCX tables are skipped | `document_loader.py` reads paragraphs only |
| BM25 has no stemming | "password" does not match "passwords" |
| Claude path never run | No Anthropic key available |
| Eval is saturated | Both apps score 31/31; it can't measure further gains |

## Sound and kept

Hybrid retrieval (`vector_store.py`, `bm25.py`), the LangGraph agent and its
four tools, citation parsing, SSE streaming, SQLite conversation checkpoints,
the eval runners and cases, Docker and CI.

## Plan

| Stage | Work |
|---|---|
| 2 | Design system; routed app shell; Assistant, Documents and Sign-in screens; source preview; copy and feedback; responsive layouts |
| 3 | Activity store (`data/app.sqlite`): document metadata, query log, feedback; Dashboard, Admin and Settings pages built on real data |
| 4 | Routers and schemas; logging; upload limit; rate limiting; conversation ownership; env var rename; code-level search limit; DOCX tables |
| 5 | Tests, lint, Docker, evals re-run, browser review at four widths, docs and delivery report |

Not planned, with reasons: SSO, per-user accounts and document-level access
control (need an identity provider); document versioning and external
integrations (need infrastructure and credentials); TypeScript migration
(large churn for little gain at this size).

Constraints: Groq's free daily token limit caps real-model testing; no
dashboard number is shown unless real requests produced it.
