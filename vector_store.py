"""In-memory vector store: cosine-similarity search, saved to disk as plain files."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Self

import numpy as np

from embeddings import EmbeddingProvider
from documents import Chunk, Section, split_into_chunks
from loaders import load_document
from web_loader import fetch_page, load_web_page
from youtube_loader import load_youtube

INDEX_ROOT = Path("data") / "index"  # git-ignored: it contains the document text
INDEX_VERSION = 2  # bump when the saved format changes: old indexes are rebuilt


class VectorStoreError(Exception):
    """Raised when a saved index cannot be used."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)  # avoid division by zero


class VectorStore:
    """Keeps chunks and their vectors side by side: row i belongs to chunk i."""

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        """Embed the chunks and append them to the store."""
        if not chunks:  # e.g. a scanned PDF with no extractable text
            return
        raw = self._embedder.embed([chunk.text for chunk in chunks])
        new_vectors = _normalize(np.array(raw, dtype=np.float32))
        if len(self._chunks) == 0:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])
        self._chunks.extend(chunks)

    @classmethod
    def combine(cls, stores: list[Self]) -> Self:
        """Join several stores into one, so a single search covers all of them.

        Each source keeps its own cached index on disk; combining only stacks
        the vectors already computed, so nothing is embedded again.
        """
        if not stores:
            raise VectorStoreError("There is nothing to combine.")
        embedder = stores[0]._embedder
        if any(store._embedder.model_name != embedder.model_name for store in stores):
            raise VectorStoreError("All sources must be indexed with the same embedding model.")
        combined = cls(embedder)
        filled = [store for store in stores if len(store) > 0]
        if filled:
            combined._vectors = np.vstack([store._vectors for store in filled])
            combined._chunks = [chunk for store in filled for chunk in store._chunks]
        return combined

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the k chunks most similar to the query, best first."""
        if len(self._chunks) == 0:
            return []
        query_vector = _normalize(np.array(self._embedder.embed([query]), dtype=np.float32))[0]
        scores = self._vectors @ query_vector  # one cosine score per chunk
        best = np.argsort(scores)[::-1][:k]  # indexes of the highest scores
        return [(self._chunks[i], float(scores[i])) for i in best]

    def save(self, folder: Path) -> None:
        """Write vectors (.npy) and chunks + model name (.json). No pickle."""
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self._vectors)
        metadata = {
            "model_name": self._embedder.model_name,
            "chunks": [asdict(chunk) for chunk in self._chunks],
        }
        (folder / "chunks.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, folder: Path, embedder: EmbeddingProvider) -> Self:
        """Read a saved index; refuse it if it was built with another model."""
        metadata = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        if metadata["model_name"] != embedder.model_name:
            raise VectorStoreError(
                f"Index was built with '{metadata['model_name']}', "
                f"but the current model is '{embedder.model_name}'. Re-index the document."
            )
        store = cls(embedder)
        store._chunks = [Chunk(**item) for item in metadata["chunks"]]
        store._vectors = np.load(folder / "vectors.npy")
        return store


def _load_or_build(
    fingerprint: bytes,
    source: str,
    get_sections: Callable[[], list[Section]],
    embedder: EmbeddingProvider,
) -> VectorStore:
    """Load the saved index for this fingerprint, or build and save it.

    get_sections is only called when there is no saved index, so a cached
    PDF is not parsed again and a cached video is not downloaded again.
    """
    content_hash = hashlib.sha256(fingerprint).hexdigest()[:16]
    index_folder = INDEX_ROOT / f"v{INDEX_VERSION}-{content_hash}"
    if index_folder.exists():
        return VectorStore.load(index_folder, embedder)
    store = VectorStore(embedder)
    store.add(split_into_chunks(get_sections(), source=source))
    store.save(index_folder)
    return store


def load_or_build_index(file_path: Path, embedder: EmbeddingProvider) -> VectorStore:
    """Index an uploaded file, keyed by its exact content."""
    data = file_path.read_bytes()
    return _load_or_build(
        data, file_path.name, lambda: load_document(file_path.name, data), embedder
    )


def load_or_build_youtube_index(video_id: str, embedder: EmbeddingProvider) -> VectorStore:
    """Index a YouTube video's transcript, keyed by its video ID."""
    return _load_or_build(
        f"youtube:{video_id}".encode(),
        f"https://youtu.be/{video_id}",  # the source doubles as a link to the video
        lambda: load_youtube(video_id),
        embedder,
    )


def load_or_build_web_index(url: str, embedder: EmbeddingProvider) -> VectorStore:
    """Index a public web page, keyed by its content: a changed page is indexed again."""
    final_url, html = fetch_page(url)  # always downloaded, to notice changes
    return _load_or_build(html, final_url, lambda: load_web_page(html), embedder)


if __name__ == "__main__":
    import sys
    import time

    from embeddings import OllamaEmbeddings

    started = time.perf_counter()
    store = load_or_build_index(Path(sys.argv[1]), OllamaEmbeddings())
    print(f"Index ready: {len(store)} chunks in {time.perf_counter() - started:.1f} s")

    for chunk, score in store.search(sys.argv[2]):
        print(f"\n--- {chunk.location}, score {score:.2f} ---")
        print(chunk.text[:300])