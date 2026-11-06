"""
Meaning-based search support: turns text into vectors with a small local
Ollama embedding model, so "extra hours pay" can find a section titled
"Overtime". If the embedding model isn't available, callers get None and
fall back to plain keyword search.
"""
import time

import numpy as np
import ollama

from config import EMBED_MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_TIMEOUT_SECONDS

_client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)
_retry_after = 0.0
BATCH_SIZE = 16


def embed(texts, kind="document"):
    """
    Return an (n, dim) float32 array of unit-length vectors, or None if the
    embedding model can't be used right now (not installed, Ollama down).
    kind is "document" for passages or "query" for questions.
    """
    global _retry_after

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    # After a failure, don't make every question wait on a broken model.
    if time.time() < _retry_after:
        return None

    # nomic-embed-text works best with these task prefixes.
    prefix = "search_query: " if kind == "query" else "search_document: "

    try:
        vectors = []

        for start in range(0, len(texts), BATCH_SIZE):
            batch = [prefix + text for text in texts[start:start + BATCH_SIZE]]
            response = _client.embed(
                model=EMBED_MODEL,
                input=batch,
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
            vectors.extend(response.embeddings)

        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    except Exception as exc:
        print(
            f"Embedding model unavailable ({exc}). Using keyword search "
            f"only. To enable meaning-based search: "
            f"ollama pull {EMBED_MODEL}"
        )
        _retry_after = time.time() + 60
        return None