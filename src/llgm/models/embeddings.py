"""Hosted dense embeddings, separate from chat and ColBERT representations."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from llgm.core.errors import ConfigurationError, ProviderError
from llgm.models.base import Usage
from llgm.models.hosted import _get, _HostedClient, _usage


class EmbeddingClient(Protocol):
    """Async dense-vector generation with reproducibility metadata."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input string in the same order."""
        ...

    def descriptor(self) -> Mapping[str, Any]:
        """Describe the encoder configuration without credentials or input text."""
        ...


@dataclass(frozen=True)
class EmbeddingEvent:
    """Usage and completion metadata for one physical embedding batch request."""

    usage: Usage
    provider: str
    model: str
    request_id: str | None
    input_count: int
    latency_seconds: float
    status: str


class OpenAIEmbeddingClient(_HostedClient):
    """Dense embedding batches with metadata-only physical-call accounting.

    This adapter makes no truncation, retries, or local-model substitutions.
    A provider-rejected oversized item is an error. Passage construction should
    enforce its selected encoder's input limit before starting an experiment.
    """

    provider = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        *,
        dimensions: int | None = None,
        batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        """Validate batching and dimensions, then initialize the hosted transport."""
        if type(batch_size) is not int or batch_size < 1:
            raise ConfigurationError("Embedding batch_size must be a positive integer")
        if dimensions is not None and (type(dimensions) is not int or dimensions < 1):
            raise ConfigurationError("Embedding dimensions must be a positive integer")
        super().__init__(model, supports_structured_output=False, **kwargs)
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.events: list[EmbeddingEvent] = []
        self._observed_dimensions: int | None = dimensions

    def descriptor(self) -> dict[str, Any]:
        """Extend transport metadata with the dense encoder and batch settings."""
        return {
            **super().descriptor(),
            "representation": "dense",
            "dimensions": self.dimensions,
            "batch_size": self.batch_size,
        }

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ordered batches and retain usage even when response validation fails."""
        if not isinstance(texts, list) or any(not isinstance(t, str) or not t for t in texts):
            raise ConfigurationError("Embedding inputs must be a list of nonempty strings")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": batch,
                "encoding_format": "float",
            }
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            started = time.monotonic()
            usage = Usage()
            request_id = None
            actual_model = self.model
            status = "failed"
            try:
                result = await self._call(self._request_client.embeddings.create, kwargs)
                raw_usage = _usage(_get(result, "usage"), input_key="prompt_tokens")
                # Embedding endpoints have no generated output tokens by API
                # contract. Missing input usage still remains unknown.
                usage = Usage(raw_usage.input_tokens, 0, raw_usage.extra)
                request_id = _get(result, "_request_id") or _get(result, "id")
                actual_model = _get(result, "model") or self.model
                rows = _get(result, "data", ()) or ()
                if not isinstance(rows, (list, tuple)) or len(rows) != len(batch):
                    raise ProviderError("Embedding response indices do not match the input batch")
                by_index = {}
                for row in rows:
                    index = _get(row, "index")
                    if type(index) is not int or not 0 <= index < len(batch) or index in by_index:
                        raise ProviderError(
                            "Embedding response indices do not match the input batch"
                        )
                    by_index[index] = _get(row, "embedding")
                batch_vectors: list[list[float]] = []
                for index in range(len(batch)):
                    vector = by_index[index]
                    if (
                        not isinstance(vector, (list, tuple))
                        or not vector
                        or any(type(n) not in {int, float} or not math.isfinite(n) for n in vector)
                    ):
                        raise ProviderError("Embedding response contains an invalid numeric vector")
                    if self._observed_dimensions is None:
                        self._observed_dimensions = len(vector)
                    if len(vector) != self._observed_dimensions:
                        raise ProviderError(
                            "Embedding dimensions changed or do not match configuration"
                        )
                    batch_vectors.append([float(n) for n in vector])
                vectors.extend(batch_vectors)
                status = "completed"
            finally:
                self.events.append(
                    EmbeddingEvent(
                        usage,
                        self.provider,
                        actual_model,
                        request_id,
                        len(batch),
                        time.monotonic() - started,
                        status,
                    )
                )
        return vectors
