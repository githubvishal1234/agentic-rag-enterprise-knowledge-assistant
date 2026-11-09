r"""The agent: a LangGraph loop over a LangChain chat model with tools.

    START -> model --(tool calls)--> tools -> model ... --(answer)--> END
                   \--(too many rounds)--> give_up -> END

The model decides which tool to call (search, read a document, list
documents, calculate), how many times, and when it has enough to answer.
"""

import ast
import operator
from threading import Lock

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.rag import llm
from app.services.vector_store import FaissVectorStore

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer employee questions "
    "using only what your tools return from the company documents. Search "
    "before answering any question about company policy; for questions with "
    "several parts, search for each part. Use read_document when a search "
    "result is cut off or you need the surrounding section, and calculate "
    "for any arithmetic. If the documents do not contain the answer, say that "
    "you could not find it in the documents. Keep answers concise."
)

# Tool rounds allowed per question. Well under LangGraph's 25-step limit,
# which would otherwise stop the graph on an unanswered tool call and leave
# the saved conversation invalid for every later request.
MAX_TOOL_ROUNDS = 6

GAVE_UP_ANSWER = (
    "I couldn't finish looking this up. Try asking a narrower question."
)
NO_ANSWER = "I couldn't produce an answer to that. Try rephrasing the question."

READ_LIMIT = 5  # chunks per read_document call

# calculate(): the arithmetic it accepts, nothing else.
OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def evaluate(expression: str) -> float:
    """Safely evaluate + - * / // % ** and parentheses on numbers."""

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return OPERATORS[type(node.op)](left, right)
        raise ValueError("only numbers and + - * / // % ** ( ) are allowed")

    return walk(ast.parse(expression.replace(",", ""), mode="eval"))


def as_chunk(document) -> dict:
    return {"text": document.page_content, "metadata": document.metadata}


def build_agent(store: FaissVectorStore, store_lock: Lock, checkpointer=None):
    """Compile the agent graph.

    `checkpointer` stores conversations by thread_id; the default keeps them
    in memory only. The API passes a SqliteSaver so they survive restarts.
    """

    # Tools return (text for the model, chunks). The chunks are kept as the
    # ToolMessage artifact, which is how citations are resolved later.

    @tool(response_format="content_and_artifact")
    def search_documents(query: str) -> tuple[str, list[dict]]:
        """Search the company documents (policies, handbooks) and return the
        most relevant passages. Use specific queries; search again with
        different wording if the results miss."""

        with store_lock:
            chunks = store.hybrid_search(query, k=5)

        if not chunks:
            return "No matching passages found.", []

        return llm.format_passages(chunks), chunks

    @tool(response_format="content_and_artifact")
    def read_document(source: str, start_chunk: int = 0) -> tuple[str, list[dict]]:
        """Read consecutive passages of one document, starting at a chunk
        number, to see a whole section. `source` is the file name exactly as
        shown in search results or list_documents."""

        with store_lock:
            chunks = sorted(
                (as_chunk(d) for d in store.documents if d.metadata["source"] == source),
                key=lambda chunk: chunk["metadata"]["chunk_id"],
            )

        if not chunks:
            raise ValueError(f"No document named {source!r}. Use list_documents.")

        selected = chunks[max(start_chunk, 0):max(start_chunk, 0) + READ_LIMIT]
        if not selected:
            raise ValueError(f"{source} has chunks 0 to {len(chunks) - 1}.")

        last = selected[-1]["metadata"]["chunk_id"]
        more = f" Continue with start_chunk={last + 1}." if last < len(chunks) - 1 else ""
        header = f"{source}: chunks {selected[0]['metadata']['chunk_id']} to {last} of {len(chunks)}.{more}"
        return f"{header}\n\n{llm.format_passages(selected)}", selected

    @tool
    def list_documents() -> str:
        """List the documents in the knowledge base with their chunk counts."""

        with store_lock:
            counts = {}
            for document in store.documents:
                source = document.metadata["source"]
                counts[source] = counts.get(source, 0) + 1

        if not counts:
            return "The knowledge base is empty."

        return "\n".join(f"{source} ({n} chunks)" for source, n in sorted(counts.items()))

    @tool
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression, for example '22 - 7' or
        '280 * 3 + 95 * 4'. Use it for any arithmetic instead of computing
        in your head."""

        result = evaluate(expression)
        return str(round(result, 6) if isinstance(result, float) else result)

    tools = [search_documents, read_document, list_documents, calculate]

    def call_model(state: MessagesState, config: RunnableConfig) -> dict:
        model = llm.get_chat_model().bind_tools(tools)
        messages = [SystemMessage(SYSTEM_PROMPT + llm.CITATION_INSTRUCTIONS), *state["messages"]]

        if config["configurable"].get("stream_text"):
            # Forward answer text as it is generated; graph.stream() callers
            # receive it in stream_mode "custom".
            write = get_stream_writer()
            response = None

            for chunk in model.stream(messages):
                response = chunk if response is None else response + chunk
                if chunk.text:
                    write({"type": "text", "delta": chunk.text})

            response = message_chunk_to_message(response)
        else:
            response = model.invoke(messages)

        # An empty reply (e.g. a declined request) would leave the turn
        # without an answer; give the user something to act on.
        if not response.tool_calls and not response.text.strip():
            response = AIMessage(NO_ANSWER)

        return {"messages": [response]}

    def give_up(state: MessagesState) -> dict:
        # Answer the pending tool calls, then end the turn with a reply, so
        # the history stays valid and the conversation can continue.
        pending = [
            ToolMessage(
                "Stopped: too many tool calls for one question.",
                tool_call_id=call["id"],
                status="error",
            )
            for call in state["messages"][-1].tool_calls
        ]
        return {"messages": [*pending, AIMessage(GAVE_UP_ANSWER)]}

    def next_step(state: MessagesState) -> str:
        messages = state["messages"]

        if not getattr(messages[-1], "tool_calls", None):
            return END

        turn = messages[last_question(messages):]
        rounds = sum(1 for m in turn if isinstance(m, AIMessage) and m.tool_calls)

        return "tools" if rounds <= MAX_TOOL_ROUNDS else "give_up"

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    # Runs all of a turn's tool calls; a failing tool becomes an error
    # message the model can read and recover from.
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("give_up", give_up)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", next_step, ["tools", "give_up", END])
    graph.add_edge("tools", "model")
    graph.add_edge("give_up", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def last_question(messages: list[BaseMessage]) -> int:
    return max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))


def cited_passages(messages: list[BaseMessage]) -> dict[str, dict]:
    """Every passage a tool returned in the conversation, by citation key."""

    passages = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.artifact:
            for chunk in message.artifact:
                passages[llm.citation_key(chunk["metadata"])] = chunk
    return passages


def summarize_turn(messages: list[BaseMessage]) -> dict:
    """The latest turn's answer, citations, segments and tool calls."""

    turn = messages[last_question(messages):]

    tool_calls = [
        {"name": call["name"], "input": call["args"]}
        for message in turn
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]

    # The answer split where citations attach; each segment lists indexes
    # into `citations`, so a UI can place markers inline.
    answer, citations, segments = llm.parse_citations(
        messages[-1].text, cited_passages(messages)
    )

    return {
        "answer": answer,
        "segments": segments,
        "citations": citations,
        "tool_calls": tool_calls,
    }


def conversation_turns(messages: list[BaseMessage]) -> list[dict]:
    """Summarize every completed turn: the question plus summarize_turn()."""

    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    turns = []

    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(messages)
        last = messages[end - 1]

        # Skip turns that never got a final answer (e.g. the LLM call failed).
        if not isinstance(last, AIMessage) or last.tool_calls:
            continue

        turns.append({"question": messages[start].text, **summarize_turn(messages[:end])})

    return turns
