"""Streamlit chat interface: add documents and links, then ask questions about them."""

import os
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from dotenv import load_dotenv

from documents import Chunk
from embeddings import EmbeddingError, OllamaEmbeddings
from loaders import SUPPORTED_EXTENSIONS, UnsupportedFileError
from providers import (
    AnthropicProvider,
    BedrockProvider,
    GeminiProvider,
    LLMError,
    LLMProvider,
    OllamaProvider,
)
from rag_engine import NOT_FOUND, answer_question, rewrite_question
from subtitle_loader import location_to_seconds
from vector_store import (
    VectorStore,
    load_or_build_index,
    load_or_build_web_index,
    load_or_build_youtube_index,
)
from web_loader import WebPageError
from youtube_loader import TranscriptError, extract_video_id

load_dotenv()  # read .env once, when the app starts

UPLOAD_FOLDER = Path("data") / "uploads"  # git-ignored, like the whole data/ folder
MAX_SOURCES = 10  # each source is indexed once; this keeps the first run reasonable

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
            if chunk.source.startswith("https://youtu.be/"):
                # A link that opens the video at the cited minute
                seconds = location_to_seconds(chunk.location)
                st.markdown(f"**[▶️ {chunk.location}]({chunk.source}?t={seconds})** · {chunk.source}")
            elif chunk.source.startswith(("http://", "https://")):
                st.markdown(f"**{chunk.location}** · [{chunk.source}]({chunk.source})")
            else:
                st.markdown(f"**{chunk.source}** · {chunk.location}")
            st.caption(chunk.text)


def index_source(source: UploadedFile | str, embedder: OllamaEmbeddings) -> VectorStore:
    """Load or build the index of one uploaded file or one link."""
    if isinstance(source, str):  # a link
        video_id = extract_video_id(source)  # None if it is not a YouTube link
        if video_id is not None:
            return load_or_build_youtube_index(video_id, embedder)
        return load_or_build_web_index(source, embedder)
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    # Keep only the file name: "../../x.pdf" must not escape the folder
    file_path = UPLOAD_FOLDER / Path(source.name).name
    file_path.write_bytes(source.getvalue())
    return load_or_build_index(file_path, embedder)


st.set_page_config(page_title="Private Document Chat", page_icon="📚")
st.title("📚 Chat with your documents")
st.caption("Documents, e-books, subtitles, YouTube and web pages. Answers come only from your source, with citations.")

# --- Sidebar: settings and sources ---
with st.sidebar:
    st.header("⚙️ Settings")
    engine = st.radio("AI engine", [OLLAMA, BEDROCK, GEMINI, CLAUDE])
    uploaded = st.file_uploader(
        "Upload documents", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True
    )
    links_text = st.text_area("...or paste links, one per line (YouTube videos or web pages)")

links = [line.strip() for line in links_text.splitlines() if line.strip()]
sources: list[UploadedFile | str] = [*uploaded, *links]
if not sources:
    st.info("Upload documents or paste links in the sidebar to start.")
    st.stop()  # nothing else to draw until there is a source
if len(sources) > MAX_SOURCES:
    st.error(f"Please use at most {MAX_SOURCES} sources at a time.")
    st.stop()

# --- Index every new source once, then search them all together ---
names = [source if isinstance(source, str) else source.name for source in sources]
if st.session_state.get("source_names") != names:
    embedder = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        host=os.getenv("OLLAMA_HOST"),
    )
    stores: list[VectorStore] = []
    ready: list[str] = []
    for source, name in zip(sources, names):
        try:
            with st.spinner(f"Indexing {name} (only the first time, may take a minute)..."):
                store = index_source(source, embedder)
        except (EmbeddingError, UnsupportedFileError, TranscriptError, WebPageError) as error:
            st.warning(f"Skipped {name}: {error}")  # one bad source must not block the others
            continue
        if len(store) == 0:
            st.warning(f"Skipped {name}: no text found (a scanned PDF, or a page built by JavaScript?).")
            continue
        stores.append(store)
        ready.append(f"{name} ({len(store)} chunks)")
    if not stores:
        st.error("None of the sources could be used.")
        st.stop()
    st.session_state["store"] = VectorStore.combine(stores)
    st.session_state["ready"] = ready
    st.session_state["source_names"] = names
    st.session_state["messages"] = []  # new sources, new conversation

with st.sidebar:
    st.subheader("📚 Sources in this chat")
    for line in st.session_state["ready"]:
        st.caption(f"✅ {line}")

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
            # With several sources, 4 excerpts could all come from one of them
            k = 4 if len(st.session_state["ready"]) == 1 else 6
            stream, found = answer_question(search_query, st.session_state["store"], provider, k=k)
            answer = str(st.write_stream(stream))  # shows the text as it arrives
        except (LLMError, EmbeddingError) as error:
            st.error(str(error))
            st.stop()
        searched_for = search_query if search_query != question else ""
        if searched_for:
            st.caption(f"🔎 Searched for: {searched_for}")
        if answer.strip() == NOT_FOUND:
            found = []  # nothing relevant was found: don't show unrelated pages
        show_sources(found)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": found, "searched_for": searched_for}
    )