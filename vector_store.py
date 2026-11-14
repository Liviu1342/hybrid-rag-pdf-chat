"""In-memory vector store: cosine-similarity search, saved to disk as plain files."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Self

import numpy as np

from embeddings import EmbeddingProvider
from pdf_loader import Chunk, load_pages, split_into_chunks

INDEX_ROOT = Path("data") / "index"  # git-ignored: it contains the PDF text


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
                f"but the current model is '{embedder.model_name}'. Re-index the PDF."
            )
        store = cls(embedder)
        store._chunks = [Chunk(**item) for item in metadata["chunks"]]
        store._vectors = np.load(folder / "vectors.npy")
        return store


def load_or_build_index(pdf_path: Path, embedder: EmbeddingProvider) -> VectorStore:
    """Load the saved index for this exact PDF content, or build and save it."""
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
    index_folder = INDEX_ROOT / pdf_hash
    if index_folder.exists():
        return VectorStore.load(index_folder, embedder)
    store = VectorStore(embedder)
    store.add(split_into_chunks(load_pages(pdf_path), source=pdf_path.name))
    store.save(index_folder)
    return store


if __name__ == "__main__":
    import sys
    import time

    from embeddings import OllamaEmbeddings

    started = time.perf_counter()
    store = load_or_build_index(Path(sys.argv[1]), OllamaEmbeddings())
    print(f"Index ready: {len(store)} chunks in {time.perf_counter() - started:.1f} s")

    for chunk, score in store.search(sys.argv[2]):
        print(f"\n--- page {chunk.page}, score {score:.2f} ---")
        print(chunk.text[:300])