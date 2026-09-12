"""Eager operational journals, lazy amended source reads, and independent adjacency."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any, Mapping

from llgm.core.errors import BudgetExceeded, ConfigurationError, ReferenceResolutionError
from llgm.core.time import (
    legacy_iso_to_ms,
    match_applicability,
    normalize_applicability,
    validate_instant_ms,
)
from llgm.core.types import (
    EvidenceRef,
    JournalEntry,
    JournalRef,
    NodeRef,
    ResolvedEvidence,
    SourceSpan,
    reference_to_dict,
)
from llgm.memory.evidence import Evidence, EvidenceSearchHit, interpret_journal

PATCH_RELATIONS = frozenset({"replace", "current_evidence"})


def _record(entry: JournalEntry) -> dict[str, Any]:
    """Serialize complete amendment content and typed provenance for local reasoning."""
    provenance = asdict(entry.provenance)
    provenance["supporting_references"] = [
        reference_to_dict(ref) for ref in entry.provenance.supporting_references
    ]
    return {
        "reference": reference_to_dict(JournalRef(entry.owning_node_id, entry.entry_id)),
        "entry_id": entry.entry_id,
        "owning_node_id": entry.owning_node_id,
        "schema_version": entry.schema_version,
        "journal_sequence": entry.journal_sequence,
        "recorded_at_ms": entry.recorded_at_ms,
        "subject": reference_to_dict(entry.subject),
        "record_kind": entry.record_kind,
        "relation": entry.relation,
        "value": entry.value if isinstance(entry.value, str) else reference_to_dict(entry.value),
        "provenance": provenance,
        "applicability": normalize_applicability(entry.applicability),
    }


def _key(reference: EvidenceRef) -> str:
    """Identify an exact source or journal range for bounded amendment-cycle detection."""
    return json.dumps(reference_to_dict(reference), sort_keys=True)


class QueryEvidence:
    """Apply explicit local amendments before exposing canonical source segments.

    Complete operational journals have finite serialized-byte limits per node and
    across this handle. Reads refresh current records. Original workspace resolution
    remains unchanged, and unrelated primary edges never inherit journal precedence.
    """

    def __init__(
        self,
        evidence: Evidence,
        scope: Mapping[str, Any],
        query_date: str | None,
        as_of: str | None = None,
        *,
        as_of_ms: int | None = None,
        max_journal_bytes: int = 65536,
        max_total_journal_bytes: int = 4194304,
        max_amendment_hops: int = 32,
    ):
        """Bind selectors. The positional ISO argument is an explicit historical adapter."""
        if as_of is not None and as_of_ms is not None:
            raise ConfigurationError("Specify as_of_ms or historical as_of, not both")
        for name, value in (
            ("max_journal_bytes", max_journal_bytes),
            ("max_total_journal_bytes", max_total_journal_bytes),
            ("max_amendment_hops", max_amendment_hops),
        ):
            if type(value) is not int or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        self.evidence = evidence
        self.scope = json.loads(json.dumps(dict(scope), allow_nan=False))
        self.query_date = query_date
        self.as_of_ms = (
            legacy_iso_to_ms(as_of)
            if as_of is not None
            else validate_instant_ms(as_of_ms, "as_of_ms")
        )
        self.max_journal_bytes = max_journal_bytes
        self.max_total_journal_bytes = max_total_journal_bytes
        self.max_amendment_hops = max_amendment_hops
        self._states: dict[str, tuple[list[JournalEntry], dict[str, Any]]] = {}
        self._journal_bytes = 0

    async def initialize_node(self, node_id: str) -> dict[str, Any]:
        """Load the entire operational journal before local model execution, never truncate."""
        entries = await self.evidence.workspace.operational_journal(
            node_id, max_bytes=self.max_journal_bytes
        )
        view = interpret_journal(entries, scope=self.scope, as_of_ms=self.as_of_ms)
        payload = {
            "node_id": node_id,
            "query_scope": self.scope,
            "query_date": self.query_date,
            "as_of_ms": self.as_of_ms,
            "journal": [_record(entry) for entry in entries],
            "active": [entry.entry_id for entry in view.active],
            "inactive": [entry.entry_id for entry in view.inactive],
            "proposed": [entry.entry_id for entry in view.proposed],
            "unresolved_entry_ids": [entry.entry_id for entry in view.unresolved],
            "conflicts": view.conflicts,
            "reasons": dict(view.reasons),
            "unresolved": [
                f"Journal {entry.entry_id}: {view.reasons[entry.entry_id]}"
                for entry in view.unresolved
            ]
            + [f"Conflicting journal entries: {', '.join(group)}" for group in view.conflicts],
        }
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if size > self.max_journal_bytes:
            raise BudgetExceeded(
                f"Complete operational journal for {node_id} requires {size} bytes; limit is {self.max_journal_bytes}"
            )
        previous = self._states.get(node_id)
        total = self._journal_bytes - (previous[1]["journal_bytes"] if previous else 0) + size
        if total > self.max_total_journal_bytes:
            raise BudgetExceeded(
                "Complete operational journals exceed the query's total journal-byte allowance"
            )
        payload["journal_bytes"] = size
        self._journal_bytes = total
        self._states[node_id] = (entries, payload)
        return json.loads(json.dumps(payload, ensure_ascii=False))

    def _patches(self, state) -> list[tuple[JournalEntry, EvidenceRef, str | None]]:
        """Select explicit replacements, including targeted superseding corrections."""
        entries, view = state
        by_id = {entry.entry_id: entry for entry in entries}
        active, unresolved = set(view["active"]), set(view["unresolved_entry_ids"])
        result = []
        for entry in entries:
            if entry.entry_id not in active | unresolved:
                continue
            original = entry
            if entry.record_kind == "correction":
                if entry.relation != "supersedes" or not isinstance(entry.subject, JournalRef):
                    continue
                original = by_id.get(entry.subject.entry_id)
                if original is None:
                    continue
            if original.relation not in PATCH_RELATIONS or not isinstance(
                original.subject, (SourceSpan, NodeRef)
            ):
                continue
            reason = view["reasons"][entry.entry_id] if entry.entry_id in unresolved else None
            result.append((entry, original.subject, reason))
        return result

    def _metadata(
        self, reference: EvidenceRef, *, entries=(), unresolved=(), view=None
    ) -> dict[str, Any]:
        """Keep complete local notes, amendment identity, and unresolved outcomes explicit."""
        return {
            "requested_reference": reference_to_dict(reference),
            "journal_view": json.loads(
                json.dumps(view if view is not None else self._states[reference.node_id][1])
            ),
            "amendments": [_record(entry) for entry in entries],
            "unresolved": list(unresolved),
        }

    def _blocked(
        self, reference: EvidenceRef, reason: str, entries=(), *, view=None
    ) -> list[ResolvedEvidence]:
        """Represent unavailable effective evidence without exposing overwritten source text."""
        return [
            ResolvedEvidence(
                "",
                reference,
                {
                    **self._metadata(reference, entries=entries, unresolved=(reason,), view=view),
                    "effective_status": "unresolved",
                },
            )
        ]

    async def read_segments(self, reference: EvidenceRef) -> list[ResolvedEvidence]:
        """Return original or replacement segments with their own canonical citations.

        A selected range intersecting a replaced passage returns that complete
        replacement because old and new offsets have no implied alignment. Empty
        blocked segments carry explicit unresolved metadata instead of stale text.
        """
        return await self._read_segments(reference, ())

    async def source_info(
        self, node_id: str, *, offset: int = 0, limit: int = 32
    ) -> dict[str, Any]:
        """Page canonical turn handles without exposing original source text to the model.

        Appended topic turns use indexed coordinates without loading their text.
        Legacy imported base blobs still require a whole-blob read.
        """
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 128:
            raise ConfigurationError("source_info requires offset >= 0 and 1 <= limit <= 128")
        await self.initialize_node(node_id)
        return await self.evidence.workspace.source_info(node_id, offset=offset, limit=limit)

    async def _read_segments(
        self, reference: EvidenceRef, path: tuple[str, ...]
    ) -> list[ResolvedEvidence]:
        """Resolve bounded replacement chains while keeping original offsets authoritative."""
        await self.initialize_node(reference.node_id)
        state = self._states[reference.node_id]
        view = state[1]
        identity = _key(reference)
        if identity in path or len(path) >= self.max_amendment_hops:
            return self._blocked(
                reference, "Amendment cycle or hop limit prevents effective resolution", view=view
            )
        path = (*path, identity)
        if isinstance(reference, JournalRef):
            record = await self.evidence.read(reference)
            return [
                replace(
                    record, metadata={**record.metadata, **self._metadata(reference, view=view)}
                )
            ]
        patches = self._patches(state)
        if isinstance(reference, NodeRef):
            whole = [patch for patch in patches if isinstance(patch[1], NodeRef)]
            if whole:
                if len(patches) != 1:
                    return self._blocked(
                        reference,
                        "Overlapping whole-node and passage amendments",
                        [patch[0] for patch in patches],
                        view=view,
                    )
                return await self._replacement(
                    reference, whole[0], path, view=view, allow_node=True
                )
            source = await self.evidence.source(reference.node_id)
            result = []
            for turn in source.turns:
                span = SourceSpan(reference.node_id, turn.turn_id, 0, len(turn.text))
                result.extend(await self._span(span, patches, path, view))
            return result
        return await self._span(reference, patches, path, view)

    async def _span(self, reference: SourceSpan, patches, path, view) -> list[ResolvedEvidence]:
        """Partition the requested original range and amend only intersecting coordinates."""
        # Validate even when a patch would otherwise hide an invalid caller range.
        original = await self.evidence.read(reference)
        applicable = []
        for patch in patches:
            subject = patch[1]
            if isinstance(subject, NodeRef):
                return self._blocked(
                    reference,
                    "Whole-node amendment has no mapping for this original passage",
                    [patch[0]],
                    view=view,
                )
            if (
                subject.turn_id == reference.turn_id
                and subject.start < reference.end
                and subject.end > reference.start
            ):
                applicable.append(patch)
        if not applicable:
            return [
                replace(
                    original, metadata={**original.metadata, **self._metadata(reference, view=view)}
                )
            ]
        # Replacement offsets cannot be split against another original-span
        # patch. Block the complete intersecting patches rather than leaking a
        # whole replacement through a superficially non-overlapping prefix.
        applicable = [
            (
                entry,
                subject,
                reason
                or (
                    "Overlapping applicable passage amendments"
                    if any(
                        other.entry_id != entry.entry_id
                        and other_subject.start < subject.end
                        and other_subject.end > subject.start
                        for other, other_subject, _ in applicable
                    )
                    else None
                ),
            )
            for entry, subject, reason in applicable
        ]
        bounds = sorted(
            {
                reference.start,
                reference.end,
                *(max(reference.start, patch[1].start) for patch in applicable),
                *(min(reference.end, patch[1].end) for patch in applicable),
            }
        )
        result, emitted = [], set()
        for start, end in zip(bounds, bounds[1:]):
            selected = [
                patch for patch in applicable if patch[1].start < end and patch[1].end > start
            ]
            segment = replace(reference, start=start, end=end)
            if not selected:
                result.append(
                    replace(
                        original,
                        reference=segment,
                        text=original.text[start - reference.start : end - reference.start],
                        metadata={**original.metadata, **self._metadata(segment, view=view)},
                    )
                )
            elif len(selected) > 1:
                result.extend(
                    self._blocked(
                        segment,
                        "Overlapping applicable passage amendments",
                        [patch[0] for patch in selected],
                        view=view,
                    )
                )
            elif selected[0][0].entry_id not in emitted:
                emitted.add(selected[0][0].entry_id)
                result.extend(await self._replacement(segment, selected[0], path, view=view))
        return result

    async def _replacement(
        self, requested, patch, path, *, view, allow_node=False
    ) -> list[ResolvedEvidence]:
        """Resolve an explicit exact replacement directly, with the amendment as provenance."""
        entry, subject, reason = patch
        if reason:
            return self._blocked(
                requested, f"Amendment {entry.entry_id} unresolved: {reason}", (entry,), view=view
            )
        target = (
            JournalRef(entry.owning_node_id, entry.entry_id, 0, len(entry.value))
            if isinstance(entry.value, str)
            else entry.value
        )
        if (
            isinstance(target, NodeRef)
            and not allow_node
            or isinstance(target, JournalRef)
            and target.start is None
        ):
            return self._blocked(
                requested,
                "Replacement requires an exact source or inline-journal span",
                (entry,),
                view=view,
            )
        try:
            if entry.record_kind == "assertion" and target == subject:
                raw = await self.evidence.read(requested)
                segments = [
                    replace(raw, metadata={**raw.metadata, **self._metadata(requested, view=view)})
                ]
            else:
                segments = await self._read_segments(target, path)
        except ReferenceResolutionError as error:
            return self._blocked(
                requested, f"Replacement unavailable: {error}", (entry,), view=view
            )
        result = []
        for segment in segments:
            metadata = {
                **segment.metadata,
                "requested_reference": reference_to_dict(requested),
                "amendments": [_record(entry), *segment.metadata.get("amendments", [])],
                "amended_from": reference_to_dict(subject),
                "journal_views": {
                    **segment.metadata.get("journal_views", {}),
                    requested.node_id: json.loads(json.dumps(view)),
                },
            }
            result.append(replace(segment, metadata=metadata))
        return result

    async def read(self, reference: EvidenceRef) -> ResolvedEvidence:
        """Return one effective segment. Multi-segment reads require explicit read_segments."""
        segments = await self.read_segments(reference)
        if len(segments) != 1:
            raise ConfigurationError(
                "Effective read has multiple canonical segments; use read_segments"
            )
        return segments[0]

    async def search(self, query: str, k: int) -> list[EvidenceSearchHit]:
        """Amend retrieved text while retaining the original owner for seed selection."""
        hits = await self.evidence.search(query, k)
        results = []
        for hit in hits:
            segments = []
            for reference in hit.passage.refs:
                segments.extend(await self.read_segments(reference))
            owners = list(dict.fromkeys(reference.node_id for reference in hit.passage.refs))
            metadata = {
                **hit.passage.metadata,
                "owner_node_id": owners[0],
                "owner_node_ids": owners,
                "segments": [
                    {
                        "text": segment.text,
                        "reference": reference_to_dict(segment.reference),
                        "metadata": dict(segment.metadata),
                    }
                    for segment in segments
                ],
                "unresolved": [
                    message
                    for segment in segments
                    for message in segment.metadata.get("unresolved", [])
                ],
                "journal_views": {owner: self._states[owner][1] for owner in owners},
            }
            results.append(
                replace(
                    hit,
                    passage=replace(
                        hit.passage,
                        text="\n".join(segment.text for segment in segments),
                        refs=tuple(dict.fromkeys(segment.reference for segment in segments)),
                        metadata=metadata,
                    ),
                )
            )
        return results

    async def journal(self, node_id: str) -> list[JournalEntry]:
        """Expose raw retained history for inspection, separate from local operational state."""
        return await self.evidence.journal(node_id)

    async def edge_descriptions(
        self, node_id: str, relation: str | None = None
    ) -> list[dict[str, Any]]:
        """Describe applicable primary edges without losing relation or provenance metadata."""
        await self.initialize_node(node_id)
        edges = await self.evidence.edge_descriptions(node_id, relation)
        descriptions = []
        for edge in edges:
            status, _ = match_applicability(
                edge["applicability"], scope=self.scope, as_of_ms=self.as_of_ms
            )
            if status == "active":
                descriptions.append(edge)
        return descriptions

    async def neighbors(self, node_id: str, relation: str | None = None) -> list[EvidenceRef]:
        """Return applicable primary neighbors. Semantic journal pointers never add adjacency."""
        descriptions = await self.edge_descriptions(node_id, relation)
        return [
            NodeRef(node)
            for node in sorted({edge["reference"]["node_id"] for edge in descriptions})
        ]
