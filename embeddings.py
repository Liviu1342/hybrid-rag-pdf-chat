"""Turn text into embedding vectors (local Ollama by default)."""

from typing import Protocol

import ollama


class EmbeddingError(Exception):
    """Raised when texts cannot be turned into vectors."""


class EmbeddingProvider(Protocol):
    """Anything that can turn a list of texts into vectors."""

    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...


class OllamaEmbeddings:
    """Embeddings computed locally by Ollama: the PDF never leaves the machine."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model
        self._batch_size = batch_size
        self._client = ollama.Client(host=host)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches, so one request never gets too large."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.embed(model=self.model_name, input=batch)
            except ollama.ResponseError as error:
                raise EmbeddingError(f"Ollama error: {error.error}") from error
            except ConnectionError as error:
                raise EmbeddingError("Cannot reach Ollama. Is it running?") from error
            vectors.extend(list(vector) for vector in response.embeddings)
        return vectors


if __name__ == "__main__":
    import numpy as np

    sentences = [
        "How much does the subscription cost?",
        "What is the monthly price of the service?",
        "My dog barks at night.",
    ]
    vectors = np.array(OllamaEmbeddings().embed(sentences))
    print(f"Shape: {vectors.shape}")

    # Cosine similarity: normalize every vector to length 1, then dot products.
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    print(np.round(unit @ unit.T, 2))