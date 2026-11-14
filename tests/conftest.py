"""Shared test helpers: pytest loads this file automatically before the tests."""

import pytest

from pdf_loader import Chunk
from vector_store import VectorStore


class FakeEmbedder:
    """Turns text into a tiny 3-number vector by counting keywords (no Ollama)."""

    model_name = "fake-embedder"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [text.lower().count("refund"), text.lower().count("shipping"), 0.1]
            for text in texts  # 0.1 above keeps every vector non-zero
        ]


@pytest.fixture
def store() -> VectorStore:
    """A small store with three chunks on three different pages."""
    vector_store = VectorStore(FakeEmbedder())
    vector_store.add(
        [
            Chunk(text="Refund policy: refund within 30 days.", page=3, source="shop.pdf"),
            Chunk(text="Shipping takes 5 days.", page=7, source="shop.pdf"),
            Chunk(text="Our company was founded in 1999.", page=1, source="shop.pdf"),
        ]
    )
    return vector_store