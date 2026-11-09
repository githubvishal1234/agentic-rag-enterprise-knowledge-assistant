import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

const STARTER_QUESTIONS = [
  "How many days of annual leave do I get?",
  "What should I do if my laptop is lost or stolen?",
  "Who needs to approve a large expense claim?",
];

const listFormat = new Intl.ListFormat("en", { type: "conjunction" });

// The current conversation's id, so a reload picks it back up. Storage can
// be unavailable (private windows, blocked site data); then we just forget.
const THREAD_KEY = "knowledge-assistant-thread";

function savedThread() {
  try {
    return localStorage.getItem(THREAD_KEY);
  } catch {
    return null;
  }
}

function saveThread(threadId) {
  try {
    if (threadId) localStorage.setItem(THREAD_KEY, threadId);
    else localStorage.removeItem(THREAD_KEY);
  } catch {
    // Not saved; the conversation still works until the page closes.
  }
}

// Conversations started in this browser, newest first: [{ id, title }].
// Kept per browser, not on the server, because everyone shares a password.
const HISTORY_KEY = "knowledge-assistant-history";

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) ?? [];
  } catch {
    return [];
  }
}

function storeHistory(history) {
  try {
    if (history.length) localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    else localStorage.removeItem(HISTORY_KEY);
  } catch {
    // Not saved; the list still works until the page closes.
  }
}

async function errorFrom(response) {
  const body = await response.json().catch(() => null);
  const error = new Error(
    typeof body?.detail === "string"
      ? body.detail
      : `The server returned an error (${response.status}).`,
  );
  error.status = response.status;
  return error;
}

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw await errorFrom(response);
  return response.json();
}

// POST /chat/stream and call onEvent for each Server-Sent Event as it
// arrives. (EventSource can't send a POST body, so read the stream directly.)
async function streamChat(body, onEvent) {
  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await errorFrom(response);

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;

    buffer += value;
    const events = buffer.split("\n\n");
    buffer = events.pop();

    for (const event of events) {
      const data = event.split("\n").find((line) => line.startsWith("data: "));
      if (data) onEvent(JSON.parse(data.slice(6)));
    }
  }
}

export default function App() {
  // undefined while checking; then { role, auth_required } from /me.
  const [me, setMe] = useState(undefined);

  useEffect(() => {
    request("/me")
      .then(setMe)
      // Server unreachable: show the workspace, which explains the problem.
      .catch(() => setMe({ role: "admin", auth_required: false }));
  }, []);

  if (me === undefined) return null;

  if (!me.role) {
    return <SignIn onSignedIn={(role) => setMe({ ...me, role })} />;
  }

  async function signOut() {
    await request("/logout", { method: "POST" }).catch(() => {});
    saveThread(null);
    storeHistory([]);
    setMe({ ...me, role: null });
  }

  return (
    <Workspace
      role={me.role}
      authRequired={me.auth_required}
      onSignOut={signOut}
      onExpired={() => setMe((current) => ({ ...current, role: null }))}
    />
  );
}

function SignIn({ onSignedIn }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError(null);

    try {
      const { role } = await request("/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      onSignedIn(role);
    } catch (failure) {
      setError(failure.message);
      setBusy(false);
    }
  }

  return (
    <main className="signin">
      <form className="signin-form" onSubmit={submit}>
        <p className="brand">Enterprise Knowledge Assistant</p>
        <h1>Sign in</h1>
        <p className="muted">Use the password your administrator gave you.</p>

        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          autoComplete="current-password"
          autoFocus
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />

        {error && (
          <p className="signin-error" role="alert">
            {error}
          </p>
        )}

        <button className="ask-button" type="submit" disabled={busy || !password}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}

function Workspace({ role, authRequired, onSignOut, onExpired }) {
  // undefined while loading, null when the server can't be reached.
  const [health, setHealth] = useState(undefined);
  const [documents, setDocuments] = useState([]);
  const [turns, setTurns] = useState([]);
  const [threadId, setThreadId] = useState(savedThread);
  const [history, setHistory] = useState(loadHistory);

  function updateHistory(change) {
    setHistory((current) => {
      const next = change(current);
      storeHistory(next);
      return next;
    });
  }

  const loadDocuments = () =>
    request("/documents")
      .then(setDocuments)
      .catch((error) => error.status === 401 && onExpired());

  useEffect(() => {
    request("/health").then(setHealth).catch(() => setHealth(null));
    loadDocuments();

    const saved = savedThread();
    if (saved) openConversation(saved);
  }, []);

  async function openConversation(id) {
    try {
      const conversation = await request(`/chat/${encodeURIComponent(id)}`);
      setTurns(
        conversation.turns.map((result) => ({
          id: crypto.randomUUID(),
          question: result.question,
          status: "done",
          result,
        })),
      );
      setThreadId(id);
      saveThread(id);
    } catch (error) {
      if (error.status === 401) return onExpired();
      // The server no longer has it (e.g. its data was cleared).
      if (error.status === 404) {
        updateHistory((current) => current.filter((item) => item.id !== id));
        if (id === threadId) startOver();
      }
    }
  }

  const updateTurn = (id, changes) =>
    setTurns((all) =>
      all.map((turn) => (turn.id === id ? { ...turn, ...changes } : turn)),
    );

  async function ask(question) {
    const id = crypto.randomUUID();
    setTurns((all) => [...all, { id, question, status: "pending" }]);

    const changeTurn = (change) =>
      setTurns((all) => all.map((turn) => (turn.id === id ? change(turn) : turn)));

    try {
      let result = null;

      await streamChat({ message: question, thread_id: threadId }, (event) => {
        if (event.type === "tool") {
          changeTurn((turn) => ({ ...turn, tools: [...(turn.tools ?? []), event] }));
        } else if (event.type === "text") {
          changeTurn((turn) => ({ ...turn, streamText: (turn.streamText ?? "") + event.delta }));
        } else if (event.type === "done") {
          result = event;
        } else if (event.type === "error") {
          throw new Error(event.detail);
        }
      });

      if (!result) throw new Error("The answer stopped before it finished.");

      setThreadId(result.thread_id);
      saveThread(result.thread_id);
      updateHistory((current) => {
        const existing = current.find((item) => item.id === result.thread_id);
        const entry = existing ?? { id: result.thread_id, title: question };
        // Most recently used first; keep the list short.
        return [entry, ...current.filter((item) => item !== existing)].slice(0, 20);
      });
      updateTurn(id, { status: "done", result });
    } catch (error) {
      if (error.status === 401) onExpired();
      updateTurn(id, { status: "error", error: error.message });
    }
  }

  function retry(turn) {
    setTurns((all) => all.filter((other) => other.id !== turn.id));
    ask(turn.question);
  }

  function startOver() {
    setTurns([]);
    setThreadId(null);
    saveThread(null);
  }

  const busy = turns.some((turn) => turn.status === "pending");
  const canAsk = Boolean(health?.llm_configured) && documents.length > 0;

  return (
    <div className="app">
      <Sidebar
        documents={documents}
        onUploaded={loadDocuments}
        canEdit={role === "admin"}
        history={history}
        currentThread={threadId}
        onOpen={openConversation}
        busy={busy}
        account={authRequired && { role, onSignOut }}
      />

      <main className="main">
        <div className="topbar">
          {turns.length > 0 && (
            <button className="quiet-button" onClick={startOver} disabled={busy}>
              New conversation
            </button>
          )}
        </div>

        <section className="conversation" aria-live="polite">
          <ServerNotice health={health} />

          {turns.length === 0 ? (
            <Welcome
              hasDocuments={documents.length > 0}
              canAsk={canAsk}
              onAsk={ask}
            />
          ) : (
            turns.map((turn) => (
              <Turn key={turn.id} turn={turn} onRetry={() => retry(turn)} />
            ))
          )}
        </section>

        <Composer onAsk={ask} disabled={busy || !canAsk} />
      </main>
    </div>
  );
}

function ServerNotice({ health }) {
  if (health === null) {
    return (
      <p className="notice" role="alert">
        Can't reach the server. Start it from the <code>backend</code> folder
        with <code>uvicorn app.main:app</code>, then reload this page.
      </p>
    );
  }

  if (health && !health.llm_configured) {
    return (
      <p className="notice" role="alert">
        Answers are off until the server has an AI model configured. Set{" "}
        <code>ANTHROPIC_API_KEY</code>, or another provider as described in{" "}
        <code>RUNNING.md</code>, and restart the server.
      </p>
    );
  }

  return null;
}

function Sidebar({ documents, onUploaded, canEdit, account, history, currentThread, onOpen, busy }) {
  const [uploads, setUploads] = useState([]);
  const [dragging, setDragging] = useState(false);
  const input = useRef(null);

  async function upload(files) {
    if (!files.length) return;

    setUploads(files.map((file) => ({ name: file.name, state: "Adding…" })));

    // One at a time: the server indexes files sequentially anyway.
    for (const [index, file] of files.entries()) {
      const form = new FormData();
      form.append("file", file);

      let state;
      try {
        await request("/ingest", { method: "POST", body: form });
        state = "Added";
      } catch (error) {
        state = error.status === 409 ? "Already added" : error.message;
      }

      setUploads((all) =>
        all.map((item, i) => (i === index ? { ...item, state } : item)),
      );
    }

    onUploaded();
  }

  async function remove(source) {
    if (!window.confirm(`Remove ${source}? Answers will no longer use it.`)) return;

    try {
      await request(`/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
      setUploads([{ name: source, state: "Removed" }]);
    } catch (error) {
      setUploads([{ name: source, state: error.message }]);
    }

    onUploaded();
  }

  return (
    <aside className="sidebar">
      <div>
        <p className="brand">Enterprise Knowledge Assistant</p>
        <p className="brand-note">Answers from your company's documents</p>
      </div>

      {history.length > 0 && (
        <section aria-labelledby="history-heading">
          <h2 id="history-heading">Recent conversations</h2>
          <ul className="history-list">
            {history.map((item) => (
              <li key={item.id}>
                <button
                  className="history-item"
                  aria-current={item.id === currentThread ? "true" : undefined}
                  disabled={busy}
                  onClick={() => onOpen(item.id)}
                >
                  {item.title}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section aria-labelledby="documents-heading">
        <h2 id="documents-heading">
          Documents{documents.length > 0 && ` (${documents.length})`}
        </h2>

        {documents.length === 0 ? (
          <p className="muted">No documents yet.</p>
        ) : (
          <ul className="doc-list">
            {documents.map((doc) => (
              <li key={doc.source}>
                <span className="doc-name">{doc.source}</span>
                {canEdit && <button
                  className="remove-button"
                  aria-label={`Remove ${doc.source}`}
                  onClick={() => remove(doc.source)}
                >
                  Remove
                </button>}
              </li>
            ))}
          </ul>
        )}
      </section>

      {canEdit && <section>
        <button
          className={`dropzone${dragging ? " is-dragging" : ""}`}
          onClick={() => input.current.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            upload([...event.dataTransfer.files]);
          }}
        >
          <span className="dropzone-title">Add documents</span>
          <span className="dropzone-hint">
            Drop files here or choose them. Text, PDF or Word.
          </span>
        </button>
        <input
          ref={input}
          type="file"
          accept=".txt,.pdf,.docx"
          multiple
          hidden
          onChange={(event) => {
            upload([...event.target.files]);
            event.target.value = "";
          }}
        />

        {uploads.length > 0 && (
          <ul className="upload-list" aria-live="polite">
            {uploads.map((item) => (
              <li key={item.name}>
                <span className="upload-name">{item.name}</span>
                <span className="muted">{item.state}</span>
              </li>
            ))}
          </ul>
        )}
      </section>}

      {account && (
        <div className="account">
          <p className="muted">
            Signed in as {account.role === "admin" ? "an admin" : "an employee"}
          </p>
          <button className="quiet-button" onClick={account.onSignOut}>
            Sign out
          </button>
        </div>
      )}
    </aside>
  );
}

function Welcome({ hasDocuments, canAsk, onAsk }) {
  return (
    <div className="welcome">
      <h1>Ask about any company policy</h1>
      <p className="welcome-lead">
        Every answer quotes the passage it comes from, so you can check it
        yourself.
      </p>

      {!hasDocuments ? (
        <p className="muted">Add your policy documents on the left to start.</p>
      ) : (
        <ul className="starters">
          {STARTER_QUESTIONS.map((question) => (
            <li key={question}>
              <button
                className="starter"
                onClick={() => onAsk(question)}
                disabled={!canAsk}
              >
                {question}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Turn({ turn, onRetry }) {
  const ref = useRef(null);

  useEffect(() => {
    ref.current.scrollIntoView({ block: "start" });
  }, [turn.status]);

  return (
    <article className="turn" ref={ref} aria-busy={turn.status === "pending"}>
      <h2 className="question">{turn.question}</h2>

      {turn.status === "pending" && <PendingAnswer turn={turn} />}

      {turn.status === "error" && (
        <div className="turn-error" role="alert">
          <p>Couldn't get an answer. {turn.error}</p>
          <button className="quiet-button" onClick={onRetry}>
            Try again
          </button>
        </div>
      )}

      {turn.status === "done" && <Answer id={turn.id} result={turn.result} />}
    </article>
  );
}

// One plain sentence for the tools the agent used, e.g.
// "Searched for “leave”, read leave_policy.txt and calculated 22 - 7".
function describeTools(calls) {
  const searches = calls
    .filter((call) => call.name === "search_documents")
    .map((call) => `“${call.input.query}”`);
  const reads = [...new Set(calls
    .filter((call) => call.name === "read_document")
    .map((call) => call.input.source))];
  const sums = calls
    .filter((call) => call.name === "calculate")
    .map((call) => call.input.expression);

  const parts = [];
  if (searches.length) parts.push(`searched for ${listFormat.format(searches)}`);
  if (reads.length) parts.push(`read ${listFormat.format(reads)}`);
  if (sums.length) parts.push(`calculated ${listFormat.format(sums)}`);
  if (calls.some((call) => call.name === "list_documents")) parts.push("listed the documents");

  const sentence = listFormat.format(parts);
  return sentence && sentence[0].toUpperCase() + sentence.slice(1);
}

// Citation keys like [leave_policy.txt#3] are turned into markers once the
// answer is complete; while it streams, hide them (and a half-arrived one).
function withoutCitationKeys(text) {
  return text
    .replace(/\s*\[[^[\]#]+#\d+(?:\s*,\s*[^[\]#,]+#\d+)*\]/g, "")
    .replace(/\s*\[[^\]]*$/, "");
}

// What the agent has done so far: its searches, then the answer as it's
// written. Citations appear once the answer is complete.
function PendingAnswer({ turn }) {
  const calls = (turn.tools ?? []).map((tool) => ({ name: tool.name, input: tool.input }));
  const done = describeTools(calls);

  return (
    <>
      {done && <p className="searched">{done}</p>}

      {turn.streamText ? (
        <div className="answer answer-layout no-sources">
          <div className="streaming">
            <ReactMarkdown>{withoutCitationKeys(turn.streamText)}</ReactMarkdown>
          </div>
        </div>
      ) : (
        <p className="pending" role="status">
          <span>{calls.length ? "Reading what it found" : "Searching the documents"}</span>
        </p>
      )}
    </>
  );
}

// Puts a [n](#cite-n) link after each cited segment, before its trailing
// whitespace, so the markdown flows as one text with markers inline.
function withMarkers(segments) {
  return segments
    .map(({ text, citations }) => {
      if (!citations.length) return text;

      const trailing = text.match(/\s*$/)[0];
      const markers = citations
        .map((index) => `[${index + 1}](#cite-${index + 1})`)
        .join("");
      return text.slice(0, text.length - trailing.length) + markers + trailing;
    })
    .join("");
}

function Answer({ id, result }) {
  const [active, setActive] = useState(null);

  const done = describeTools(result.tool_calls);

  const markdownComponents = {
    a({ href, children }) {
      if (href?.startsWith("#cite-")) {
        const number = Number(href.slice(6));
        return (
          <sup>
            <a
              className="marker"
              href={`#${id}-source-${number}`}
              aria-label={`Source ${number}`}
              onMouseEnter={() => setActive(number)}
              onMouseLeave={() => setActive(null)}
              onFocus={() => setActive(number)}
              onBlur={() => setActive(null)}
            >
              {number}
            </a>
          </sup>
        );
      }

      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    },
  };

  return (
    <>
      {done && <p className="searched">{done}</p>}

      <div className={`answer-layout${result.citations.length ? "" : " no-sources"}`}>
        <div className="answer">
          <ReactMarkdown components={markdownComponents}>
            {withMarkers(result.segments)}
          </ReactMarkdown>
        </div>

        {result.citations.length > 0 && (
          <ol className="sources" aria-label="Sources">
            {result.citations.map((citation, index) => (
              <li
                key={index}
                id={`${id}-source-${index + 1}`}
                className={`source${active === index + 1 ? " is-active" : ""}`}
              >
                <span className="source-number">{index + 1}</span>
                <p className="source-quote">
                  <mark>{citation.cited_text.trim()}</mark>
                </p>
                <p className="source-title">{citation.title ?? citation.source}</p>
              </li>
            ))}
          </ol>
        )}
      </div>
    </>
  );
}

function Composer({ onAsk, disabled }) {
  const [text, setText] = useState("");

  function submit(event) {
    event.preventDefault();
    const question = text.trim();
    if (!question || disabled) return;
    onAsk(question);
    setText("");
  }

  return (
    <form className="composer" onSubmit={submit}>
      <div className="composer-inner">
        <label htmlFor="question" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="question"
          rows={1}
          value={text}
          placeholder="Ask a question about a policy"
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) submit(event);
          }}
        />
        <button className="ask-button" type="submit" disabled={disabled || !text.trim()}>
          Ask
        </button>
      </div>
    </form>
  );
}
