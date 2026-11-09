"""One-shot answers (/query): retrieved chunks in, cited answer out."""

from langchain_core.messages import HumanMessage, SystemMessage

from app.rag import llm

SYSTEM_PROMPT = (
    "You answer employee questions using only the provided company "
    "documents. If the documents do not contain the answer, say that you "
    "could not find it in the documents. Keep answers concise."
)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Answer a question from retrieved chunks, citing the chunks used."""

    response = llm.get_chat_model().invoke([
        SystemMessage(SYSTEM_PROMPT + llm.CITATION_INSTRUCTIONS),
        HumanMessage(
            f"Passages:\n\n{llm.format_passages(chunks)}\n\nQuestion: {question}"
        ),
    ])

    passages = {llm.citation_key(chunk["metadata"]): chunk for chunk in chunks}
    answer, citations, _ = llm.parse_citations(response.text, passages)

    return {"answer": answer, "citations": citations}
