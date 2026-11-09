"""The chat model (through LangChain) and the citation format.

Provider settings (from the environment or .env):
    LLM_PROVIDER  "anthropic" (default) or "openai" for any OpenAI-compatible
                  API: OpenAI, Google Gemini, Groq, Ollama, ...
    LLM_MODEL     model name; defaults to claude-opus-5-5 for anthropic
    LLM_BASE_URL  API address for Gemini, Groq, Ollama, ...
    LLM_API_KEY   key for openai-compatible providers (or OPENAI_API_KEY)
    ANTHROPIC_API_KEY  key for Claude

Every passage the model sees is labelled with a citation key such as
[leave_policy.txt#3] (file and chunk number). The model cites those keys
and parse_citations() turns them into citation records, so all providers
produce the same answer format.
"""

import os
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"

CITATION_INSTRUCTIONS = (
    " Each passage is labelled with a citation key in square brackets, for "
    "example [leave_policy.txt#3]. After each sentence that uses a passage, "
    "add its key exactly as written, for example [leave_policy.txt#3]. Cite "
    "only keys you were given."
)

# [file.txt#3] or [a.txt#1, b.pdf#4]. Some models (e.g. gpt-oss) write
# their trained-in full-width brackets and comma instead: 【file.txt#3】.
CITATION = re.compile(
    r"\s*[\[【]([^\[\]【】#]+#\d+(?:\s*[,，]\s*[^\[\]【】#,，]+#\d+)*)[\]】]"
)
SEPARATOR = re.compile(r"[,，]")


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "anthropic").lower()


def model_name() -> str | None:
    if provider() == "anthropic":
        return os.environ.get("LLM_MODEL") or DEFAULT_CLAUDE_MODEL
    return os.environ.get("LLM_MODEL")


def openai_api_key() -> str | None:
    return os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")


def has_credentials() -> bool:
    """True when the selected provider has what it needs to answer."""

    if provider() == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    # A local server such as Ollama needs a URL but no key.
    return bool(model_name() and (openai_api_key() or os.environ.get("LLM_BASE_URL")))


def get_chat_model() -> BaseChatModel:
    """The configured LangChain chat model."""

    if provider() == "anthropic":
        return ChatAnthropic(model=model_name(), max_tokens=16000, timeout=180)

    return ChatOpenAI(
        model=model_name(),
        api_key=openai_api_key() or "not-needed",
        base_url=os.environ.get("LLM_BASE_URL") or None,
        timeout=180,
    )


def citation_key(metadata: dict) -> str:
    return f"{metadata['source']}#{metadata['chunk_id']}"


def chunk_title(metadata: dict) -> str:
    title = metadata["source"]

    if "page" in metadata:
        title += f", page {metadata['page']}"

    return f"{title} (chunk {metadata['chunk_id']})"


def format_passages(chunks: list[dict]) -> str:
    """Passages for the model, each under its citation key."""

    return "\n\n".join(
        f"[{citation_key(chunk['metadata'])}] {chunk_title(chunk['metadata'])}\n{chunk['text']}"
        for chunk in chunks
    )


def parse_citations(text: str, passages: dict[str, dict]) -> tuple[str, list[dict], list[dict]]:
    """Split an answer at its citation keys.

    `passages` maps citation keys to chunks ({"text", "metadata"}). Returns
    the answer without the keys, the unique citations in order of first use
    ({"cited_text", "source", "title", **metadata}), and segments: the answer
    split where citations attach, each with indexes into the citations.
    Unknown keys are dropped.
    """

    citations: list[dict] = []
    segments: list[dict] = []
    position = 0

    for match in CITATION.finditer(text):
        references = []

        for key in (part.strip() for part in SEPARATOR.split(match.group(1))):
            chunk = passages.get(key)
            if chunk is None:
                continue

            citation = {
                "cited_text": chunk["text"],
                "title": chunk_title(chunk["metadata"]),
                **chunk["metadata"],
            }
            if citation not in citations:
                citations.append(citation)
            references.append(citations.index(citation))

        segment = text[position:match.start()]
        position = match.end()

        # "A [x#1][y#2]": a key straight after another cites the same text.
        if not segment and segments:
            segments[-1]["citations"] += references
        else:
            segments.append({"text": segment, "citations": references})

    if position < len(text):
        segments.append({"text": text[position:], "citations": []})

    answer = "".join(segment["text"] for segment in segments)
    return answer, citations, segments
