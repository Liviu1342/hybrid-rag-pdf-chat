"""Streamlit chat interface: upload a PDF, then ask questions about it."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from embeddings import EmbeddingError, OllamaEmbeddings
from documents import Chunk
from providers import (
    AnthropicProvider,
    BedrockProvider,
    GeminiProvider,
    LLMError,
    LLMProvider,
    OllamaProvider,
)
from rag_engine import NOT_FOUND, answer_question, rewrite_question
from vector_store import load_or_build_index

load_dotenv()  # read .env once, when the app starts

UPLOAD_FOLDER = Path("data") / "uploads"  # git-ignored, like the whole data/ folder

OLLAMA = "🖥️ Ollama (local & private)"
BEDROCK = "🪨 Bedrock (Claude, AWS cloud)"
GEMINI = "☁️ Gemini (Google cloud)"
CLAUDE = "🧠 Claude API (Anthropic, cloud)"


def build_provider(choice: str) -> LLMProvider:
    """Create the provider selected in the sidebar, configured from .env."""
    if choice == OLLAMA:
        return OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    if choice == BEDROCK:
        return BedrockProvider(
            access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            model=os.getenv("BEDROCK_MODEL", "eu.anthropic.claude-sonnet-4-6"),
            region=os.getenv("AWS_REGION", "eu-north-1"),
        )
    if choice == CLAUDE:
        return AnthropicProvider(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        )
    return GeminiProvider(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        model=os.getenv("GEMINI_MODEL", ""),
    )


def show_sources(sources: list[Chunk]) -> None:
    """Show the exact excerpts the answer was based on."""
    if not sources:
        return
    with st.expander("📚 Sources from the document"):
        for chunk in sources:
            st.markdown(f"**{chunk.location.capitalize()}**")
            st.caption(chunk.text)


st.set_page_config(page_title="Private PDF Chat", page_icon="📚")
st.title("📚 Chat with your PDF")
st.caption("Answers come only from your document, with page citations.")

# --- Sidebar: settings and upload ---
with st.sidebar:
    st.header("⚙️ Settings")
    engine = st.radio("AI engine", [OLLAMA, BEDROCK, GEMINI, CLAUDE])
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is None:
    st.info("Upload a PDF in the sidebar to start.")
    st.stop()  # nothing else to draw until there is a document

# --- Index the PDF once per new file ---
if st.session_state.get("pdf_name") != uploaded.name:
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    # Keep only the file name: a crafted name like "../../x.pdf" must not escape the folder
    pdf_path = UPLOAD_FOLDER / Path(uploaded.name).name
    pdf_path.write_bytes(uploaded.getvalue())
    try:
        with st.spinner("Indexing the PDF (only the first time, may take a minute)..."):
            embedder = OllamaEmbeddings(
                model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                host=os.getenv("OLLAMA_HOST"),
            )
            store = load_or_build_index(pdf_path, embedder)
    except EmbeddingError as error:
        st.error(str(error))
        st.stop()
    if len(store) == 0:
        st.error("No text found in this PDF (is it a scanned image?).")
        st.stop()
    st.session_state["store"] = store
    st.session_state["pdf_name"] = uploaded.name
    st.session_state["messages"] = []  # new document, new conversation

st.success(f"Ready: {uploaded.name} ({len(st.session_state['store'])} chunks)")

# --- Chat history: Streamlit redraws the page on every interaction ---
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("searched_for"):
            st.caption(f"🔎 Searched for: {message['searched_for']}")
        show_sources(message.get("sources", []))

# --- New question ---
if question := st.chat_input("Ask something about the document"):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            provider = build_provider(engine)
            # Everything said before this question (the question itself is the last item)
            history = [(m["role"], m["content"]) for m in st.session_state["messages"][:-1]]
            with st.spinner("Understanding the question..."):
                search_query = rewrite_question(question, history, provider)
            stream, sources = answer_question(search_query, st.session_state["store"], provider)
            answer = str(st.write_stream(stream))  # shows the text as it arrives
        except (LLMError, EmbeddingError) as error:
            st.error(str(error))
            st.stop()
        searched_for = search_query if search_query != question else ""
        if searched_for:
            st.caption(f"🔎 Searched for: {searched_for}")
        if answer.strip() == NOT_FOUND:
            sources = []  # nothing relevant was found: don't show unrelated pages
        show_sources(sources)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources, "searched_for": searched_for}
    )