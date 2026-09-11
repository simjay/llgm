"""Exact cosine retrieval with an explicitly supplied asynchronous embedder."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Sequence

from llgm.core.errors import ConfigurationError, ProviderError
from llgm.retrieval.base import (
    Embedder,
    SearchHit,
    SearchPassage,
    check_passages,
    corpus_fingerprint,
)


def _unit(vector: Sequence[float], expected: int | None = None) -> tuple[float, ...]:
    """Normalize a finite, nonzero embedding with optional dimension validation."""
    try:
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in vector):
            raise ValueError
        values = tuple(float(v) for v in vector)
    except (TypeError, ValueError, OverflowError):
        raise ProviderError("Embedding vector must contain finite numbers") from None
    if (
        not values
        or (expected is not None and len(values) != expected)
        or not all(math.isfinite(v) for v in values)
    ):
        raise ProviderError("Embedding vector is empty, nonfinite, or has inconsistent dimensions")
    scale = max(abs(v) for v in values)
    if scale == 0:
        raise ProviderError("Cosine similarity is undefined for a zero embedding")
    # Scale first so finite vectors neither overflow nor underflow while squaring.
    scaled = tuple(v / scale for v in values)
    norm = math.sqrt(math.fsum(v * v for v in scaled))
    return tuple(v / norm for v in scaled)


class ExactDenseRetriever:
    """Exact cosine ranking over normalized vectors from an injected embedder."""

    def __init__(self, passages, vectors, embedder, metadata=None):
        """Validate passage/vector alignment and normalize all stored embeddings."""
        check_passages(passages)
        if len(passages) != len(vectors):
            raise ProviderError("Embedding response count differs from corpus length")
        self.passages, self.embedder = list(passages), embedder
        dimension = None
        self.vectors = []
        for vector in vectors:
            normalized = _unit(vector, dimension)
            dimension = len(normalized)
            self.vectors.append(normalized)
        self.dimensions = dimension
        self.metadata = dict(metadata or {})
        self.events: list[dict] = []

    @classmethod
    async def build(
        cls, passages: Sequence[SearchPassage], embedder: Embedder, batch_size: int = 64
    ):
        """Encode passage batches and retain encoder identity and indexing time."""
        if type(batch_size) is not int or batch_size <= 0:
            raise ConfigurationError("Embedding batch size must be a positive integer")
        started = time.perf_counter()
        vectors = []
        for offset in range(0, len(passages), batch_size):
            batch = list(passages[offset : offset + batch_size])
            response = await embedder.embed([p.text for p in batch])
            if len(response) != len(batch):
                raise ProviderError("Embedding response count differs from request")
            vectors.extend(response)
        descriptor = (
            embedder.descriptor()
            if hasattr(embedder, "descriptor")
            else {"model": getattr(embedder, "model", "injected")}
        )
        return cls(
            passages,
            vectors,
            embedder,
            {
                "embedding": descriptor,
                "encoded_at": datetime.now(timezone.utc).isoformat(),
                "index_seconds": time.perf_counter() - started,
            },
        )

    async def search(self, query: str, k: int) -> list[SearchHit]:
        """Embed the query and rank every stored vector with deterministic tie breaks."""
        if type(k) is not int or k < 0:
            raise ConfigurationError("k must be a non-negative integer")
        if not self.passages or not k:
            return []
        start = time.perf_counter()
        response = await self.embedder.embed([query])
        if len(response) != 1:
            raise ProviderError("Query embedding response count must be one")
        vector = _unit(response[0], self.dimensions)
        scores = [
            (sum(a * b for a, b in zip(vector, stored)), i) for i, stored in enumerate(self.vectors)
        ]
        scores.sort(key=lambda pair: (-pair[0], self.passages[pair[1]].passage_id))
        self.events.append(
            {
                "operation": "exact_cosine_search",
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "scored_vectors": len(scores),
                "requested_k": k,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        return [
            SearchHit(self.passages[i], score, rank)
            for rank, (score, i) in enumerate(scores[:k], 1)
        ]

    def save(self, path: str | Path) -> None:
        """Write a vector snapshot bound to the ordered corpus fingerprint."""
        Path(path).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "corpus_sha256": corpus_fingerprint(self.passages),
                    "vectors": self.vectors,
                    "metadata": self.metadata,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def open(cls, path: str | Path, passages: Sequence[SearchPassage], embedder: Embedder):
        """Restore vectors only when the supplied corpus and known model IDs match."""
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or value.get("corpus_sha256") != corpus_fingerprint(
            passages
        ):
            raise ConfigurationError("Dense snapshot schema or corpus fingerprint mismatch")
        result = cls(passages, value["vectors"], embedder, value["metadata"])
        expected = result.metadata.get("embedding", {}).get("model")
        actual = getattr(embedder, "model", None)
        if expected and actual and expected != actual:
            raise ConfigurationError("Query embedding model differs from stored-vector model")
        return result

    def descriptor(self) -> dict:
        """Report vector dimensions, corpus identity, and encoding provenance."""
        return {
            "backend": "D",
            "implementation": "exact-cosine",
            "dimensions": self.dimensions,
            "corpus_sha256": corpus_fingerprint(self.passages),
            "passage_count": len(self.passages),
            **self.metadata,
        }
