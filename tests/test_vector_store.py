"""Tests for the in-memory vector store, using a fake embedder instead of Ollama."""

from pathlib import Path

import pytest

from conftest import FakeEmbedder
from documents import Chunk
from vector_store import VectorStore, VectorStoreError


def test_search_returns_the_most_similar_chunk_first(store: VectorStore) -> None:
    results = store.search("How do I get a refund?", k=1)
    chunk, score = results[0]
    assert chunk.location == "page 3"
    assert score > 0.9


def test_search_returns_at_most_k_results(store: VectorStore) -> None:
    assert len(store.search("shipping", k=2)) == 2


def test_search_on_empty_store_returns_nothing() -> None:
    assert VectorStore(FakeEmbedder()).search("anything") == []


def test_adding_no_chunks_keeps_the_store_empty() -> None:
    empty = VectorStore(FakeEmbedder())
    empty.add([])
    assert len(empty) == 0


def test_saved_index_can_be_loaded_back(store: VectorStore, tmp_path: Path) -> None:
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, FakeEmbedder())
    assert len(loaded) == 3
    assert loaded.search("shipping", k=1)[0][0].location == "page 7"


def test_loading_with_another_model_is_refused(store: VectorStore, tmp_path: Path) -> None:
    store.save(tmp_path)

    class OtherEmbedder(FakeEmbedder):
        model_name = "other-model"

    with pytest.raises(VectorStoreError, match="Re-index"):
        VectorStore.load(tmp_path, OtherEmbedder())


def test_combined_store_searches_every_source(store: VectorStore) -> None:
    other = VectorStore(FakeEmbedder())
    other.add([Chunk(text="Refunds for shipping costs.", location="slide 2", source="deck.pptx")])
    combined = VectorStore.combine([store, other])

    assert len(combined) == 4
    best, _score = combined.search("refund of shipping", k=1)[0]  # only the deck has both words
    assert (best.source, best.location) == ("deck.pptx", "slide 2")


def test_combining_stores_from_different_models_is_refused(store: VectorStore) -> None:
    class OtherEmbedder(FakeEmbedder):
        model_name = "other-model"

    with pytest.raises(VectorStoreError, match="same embedding model"):
        VectorStore.combine([store, VectorStore(OtherEmbedder())])