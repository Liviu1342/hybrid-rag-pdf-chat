"""Tests for prompt building and answering, with a fake LLM and a fake embedder."""

from collections.abc import Iterator

from pdf_loader import Chunk
from rag_engine import NOT_FOUND, SYSTEM_PROMPT, answer_question, build_user_prompt
from vector_store import VectorStore


class FakeProvider:
    """Records the prompts it receives and streams a fixed answer (no network)."""

    name = "Fake"

    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "".join(self.stream(system_prompt, user_prompt))

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        yield "Refunds are accepted "
        yield "within 30 days (page 3)."


def test_each_chunk_is_wrapped_in_a_tag_with_its_page() -> None:
    chunks = [Chunk(text="Alpha", page=4, source="a.pdf"), Chunk(text="Beta", page=9, source="a.pdf")]
    prompt = build_user_prompt("What is alpha?", chunks)
    assert '<excerpt page="4">\nAlpha\n</excerpt>' in prompt
    assert '<excerpt page="9">\nBeta\n</excerpt>' in prompt


def test_question_comes_after_the_context() -> None:
    prompt = build_user_prompt("What is alpha?", [Chunk(text="Alpha", page=1, source="a.pdf")])
    assert prompt.index("</context>") < prompt.index("Question: What is alpha?")


def test_system_prompt_defines_the_not_found_answer() -> None:
    assert NOT_FOUND in SYSTEM_PROMPT


def test_answer_streams_the_provider_text_and_returns_sources(store: VectorStore) -> None:
    provider = FakeProvider()
    stream, sources = answer_question("How do I get a refund?", store, provider, k=1)

    assert "".join(stream) == "Refunds are accepted within 30 days (page 3)."
    assert [chunk.page for chunk in sources] == [3]


def test_provider_receives_the_rules_and_the_retrieved_chunk(store: VectorStore) -> None:
    provider = FakeProvider()
    stream, _sources = answer_question("How do I get a refund?", store, provider, k=1)
    list(stream)  # a generator runs only when consumed: this sends the "request"

    assert provider.system_prompt == SYSTEM_PROMPT
    assert '<excerpt page="3">' in provider.user_prompt
    assert "Shipping" not in provider.user_prompt  # only the top-k chunks are sent