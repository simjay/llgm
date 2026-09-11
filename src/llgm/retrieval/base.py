"""Search contracts. Passage identities never replace canonical source spans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence

from llgm.core.types import SourceSpan


@dataclass(frozen=True)
class SearchPassage:
    """An indexed text view tied to immutable canonical source spans."""

    passage_id: str
    text: str
    refs: tuple[SourceSpan, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    """A passage with a backend-specific score and one-based result rank."""

    passage: SearchPassage
    score: float
    rank: int


class Retriever(Protocol):
    """Async passage search with inspectable backend and corpus identity."""

    async def search(self, query: str, k: int) -> list[SearchHit]:
        """Return up to k ranked passage hits for the supplied query."""
        ...

    def descriptor(self) -> dict[str, Any]:
        """Describe the backend configuration and indexed corpus identity."""
        ...


class Embedder(Protocol):
    """Minimal async encoding contract required by dense retrieval."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one numeric vector for each input text, preserving input order."""
        ...


def passage_to_dict(passage: SearchPassage) -> dict[str, Any]:
    """Serialize a passage together with its canonical source references."""
    return asdict(passage)


def passage_from_dict(value: dict[str, Any]) -> SearchPassage:
    """Restore a passage and typed source spans from its serialized form."""
    return SearchPassage(
        value["passage_id"],
        value["text"],
        tuple(SourceSpan(**ref) for ref in value["refs"]),
        dict(value.get("metadata", {})),
    )


def corpus_fingerprint(passages: Sequence[SearchPassage]) -> str:
    """Hash ordered passages, references, and metadata for index compatibility."""
    digest = hashlib.sha256()
    for passage in passages:
        digest.update(
            json.dumps(
                passage_to_dict(passage), sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def check_passages(passages: Sequence[SearchPassage]) -> None:
    """Reject duplicate passage identities within a single case corpus."""
    from llgm.core.errors import ConfigurationError

    ids = [p.passage_id for p in passages]
    if len(ids) != len(set(ids)):
        raise ConfigurationError("Passage IDs must be unique inside a case corpus")
