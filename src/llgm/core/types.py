"""Immutable public evidence records. Offsets count Python Unicode code points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

from llgm.core.errors import SchemaError
from llgm.core.time import validate_instant_ms


def _nonempty(value: str, name: str) -> None:
    """Require a nonempty text identifier without normalizing its bytes."""
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{name} must be a nonempty string")


def _range(start: int, end: int) -> None:
    """Validate an ordered, nonnegative Unicode-code-point range."""
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise SchemaError("A range requires integer offsets 0 <= start <= end")


@dataclass(frozen=True)
class Turn:
    """One exact source utterance with a stable ID and speaker role."""

    turn_id: str
    role: str
    text: str

    def __post_init__(self) -> None:
        """Validate turn identity, role, and exact text without stripping whitespace."""
        _nonempty(self.turn_id, "turn_id")
        _nonempty(self.role, "role")
        if not isinstance(self.text, str):
            raise SchemaError("Turn text must be a string")


@dataclass(frozen=True)
class Conversation:
    """Ordered source turns and metadata to publish as one immutable node."""

    turns: tuple[Turn, ...]
    node_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        """Freeze turn order and reject missing, duplicate, or malformed turn records."""
        object.__setattr__(self, "turns", tuple(self.turns))
        if not self.turns or any(not isinstance(turn, Turn) for turn in self.turns):
            raise SchemaError("Conversation requires at least one Turn")
        if len({turn.turn_id for turn in self.turns}) != len(self.turns):
            raise SchemaError("Turn IDs must be unique within a source node")
        if self.node_id is not None:
            _nonempty(self.node_id, "node_id")
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("Conversation metadata must be a mapping")
        validate_instant_ms(self.timestamp_ms, "timestamp_ms")

    @classmethod
    def from_turns(
        cls,
        turns: list[Mapping[str, Any] | Turn],
        *,
        node_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        timestamp_ms: int | None = None,
    ) -> Conversation:
        """Preserve text exactly. Synthesize stable turn IDs when omitted."""
        converted = []
        for index, turn in enumerate(turns):
            if isinstance(turn, Turn):
                converted.append(turn)
            else:
                if "text" in turn and "content" in turn and turn["text"] != turn["content"]:
                    raise SchemaError("Conflicting text and content fields")
                converted.append(
                    Turn(
                        turn_id=turn.get("turn_id", f"turn-{index:06d}"),
                        role=turn.get("role", ""),
                        text=turn.get("text", turn.get("content")),
                    )
                )
        return cls(
            tuple(converted), node_id=node_id, metadata=metadata or {}, timestamp_ms=timestamp_ms
        )


@dataclass(frozen=True)
class SourceSpan:
    """A half-open text range in an immutable turn of a source node."""

    node_id: str
    turn_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate source identity, turn identity, and range coordinates."""
        _nonempty(self.node_id, "node_id")
        _nonempty(self.turn_id, "turn_id")
        _range(self.start, self.end)


@dataclass(frozen=True)
class NodeRef:
    """A stable node handle whose current view includes appended immutable turns."""

    node_id: str

    def __post_init__(self) -> None:
        """Validate immutable node identity."""
        _nonempty(self.node_id, "node_id")


@dataclass(frozen=True)
class JournalRef:
    """A journal-record handle or a half-open range within its inline text value."""

    node_id: str
    entry_id: str
    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        """Validate journal identity and require either both range bounds or neither."""
        _nonempty(self.node_id, "node_id")
        _nonempty(self.entry_id, "entry_id")
        if (self.start is None) != (self.end is None):
            raise SchemaError("JournalRef requires both range offsets or neither")
        if self.start is not None:
            _range(self.start, self.end)


EvidenceRef: TypeAlias = SourceSpan | NodeRef | JournalRef


@dataclass(frozen=True)
class SourceNode:
    """A loaded immutable source node with an optional event timestamp."""

    node_id: str
    turns: tuple[Turn, ...]
    metadata: Mapping[str, Any]
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class IngestResult:
    """Published source identity and whether this call created it or retried it."""

    node_id: str
    created: bool


@dataclass(frozen=True)
class ResolvedEvidence:
    """Resolved text with its canonical reference and stored attribution metadata."""

    text: str
    reference: EvidenceRef
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Provenance:
    """Origin, producer, supporting evidence, and optional model/policy identifiers."""

    origin: str
    producer: str
    supporting_references: tuple[EvidenceRef, ...] = ()
    model: str | None = None
    prompt_version: str | None = None
    policy_version: str | None = None

    def __post_init__(self) -> None:
        """Validate origin and producer while freezing supporting-reference order."""
        if self.origin not in {"user", "source", "model", "system"}:
            raise SchemaError("Provenance origin must be user, source, model, or system")
        _nonempty(self.producer, "producer")
        object.__setattr__(self, "supporting_references", tuple(self.supporting_references))


@dataclass(frozen=True)
class JournalEntry:
    """An immutable assertion, overwrite, or correction in a node journal."""

    entry_id: str
    owning_node_id: str
    journal_sequence: int
    recorded_at_ms: int
    subject: EvidenceRef
    record_kind: str
    relation: str
    value: str | EvidenceRef
    provenance: Provenance
    applicability: Mapping[str, Any] | None = None
    schema_version: int = 2


@dataclass(frozen=True)
class Edge:
    """An independently stored directed relationship with an explicit withdrawal state."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    provenance: Provenance
    recorded_at_ms: int
    applicability: Mapping[str, Any] | None = None
    withdrawn_at_ms: int | None = None
    withdrawal_provenance: Provenance | None = None

    def __post_init__(self) -> None:
        """Validate independent edge identity, endpoints, and auditable lifecycle fields."""
        for name in ("edge_id", "source_node_id", "target_node_id"):
            _nonempty(getattr(self, name), name)
        if self.source_node_id == self.target_node_id:
            raise SchemaError("Edge endpoints must differ")
        if not isinstance(self.provenance, Provenance):
            raise SchemaError("Edge provenance must be a Provenance record")
        if type(self.recorded_at_ms) is not int:
            raise SchemaError("Edge recorded_at_ms must be integer Unix milliseconds")
        validate_instant_ms(self.withdrawn_at_ms, "withdrawn_at_ms")
        if (self.withdrawn_at_ms is None) != (self.withdrawal_provenance is None):
            raise SchemaError("Edge withdrawal requires both time and provenance")
        if self.withdrawal_provenance is not None and not isinstance(
            self.withdrawal_provenance, Provenance
        ):
            raise SchemaError("Edge withdrawal provenance must be a Provenance record")
        if self.applicability is not None and not isinstance(self.applicability, Mapping):
            raise SchemaError("Edge applicability must be a mapping or None")


def reference_to_dict(ref: EvidenceRef) -> dict[str, Any]:
    """Encode a typed evidence reference with an explicit representation tag."""
    if isinstance(ref, SourceSpan):
        return {
            "type": "source_span",
            "node_id": ref.node_id,
            "turn_id": ref.turn_id,
            "start": ref.start,
            "end": ref.end,
        }
    if isinstance(ref, NodeRef):
        return {"type": "node", "node_id": ref.node_id}
    if isinstance(ref, JournalRef):
        return {
            "type": "journal",
            "node_id": ref.node_id,
            "entry_id": ref.entry_id,
            "start": ref.start,
            "end": ref.end,
        }
    raise SchemaError(f"Unsupported evidence reference: {type(ref).__name__}")


def reference_from_dict(value: Mapping[str, Any]) -> EvidenceRef:
    """Decode a tagged reference and reject unknown or malformed representations."""
    data = dict(value)
    kind = data.pop("type", None)
    classes = {"source_span": SourceSpan, "node": NodeRef, "journal": JournalRef}
    if kind not in classes:
        raise SchemaError(f"Unknown evidence reference type: {kind!r}")
    try:
        return classes[kind](**data)
    except TypeError as exc:
        raise SchemaError(f"Malformed {kind} reference") from exc
