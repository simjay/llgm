"""Measure passage retrieval and the application's existing seed admission separately.

Each invocation uses a fresh, journal-free workspace and a caller-owned real
retriever over the supplied corpus. Gold records enter scoring only. This stops
after ``LLGM._seeds``; direct seed coverage is not an answer-quality measurement.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from llgm.core.errors import ConfigurationError
from llgm.core.types import Conversation, SourceSpan, reference_to_dict
from llgm.evaluation.longmemeval import EvaluationCase, GoldRecord
from llgm.inference.budget import Budget, RunLedger, byte_token_bound
from llgm.llgm import LLGM
from llgm.memory.evidence import Evidence
from llgm.memory.workspace import Workspace
from llgm.retrieval.base import Retriever, SearchPassage, check_passages, corpus_fingerprint


def anonymize_case(case: EvaluationCase, gold: GoldRecord) -> tuple[EvaluationCase, GoldRecord]:
    """Remove benchmark identifier cues while preserving source text and scoring.

    Each node ID becomes ``n-`` plus the first 20 hexadecimal characters of
    SHA256(``llgm-node-search-opaque-v1:`` + case ID + ``:`` + original node ID).
    Only date metadata survives. Turn IDs, roles, text, source order, and source
    timestamps remain exact; this is identifier normalization, not text redaction.
    Original annotated session IDs stay exclusively in evaluator aliases/labels.
    """
    if case.case_id != gold.case_id:
        raise ConfigurationError("Case and evaluator identities must match")
    mapping = {
        source.node_id: "n-"
        + hashlib.sha256(
            f"llgm-node-search-opaque-v1:{case.case_id}:{source.node_id}".encode("utf-8")
        ).hexdigest()[:20]
        for source in case.sources
    }
    if len(mapping) != len(case.sources) or len(set(mapping.values())) != len(mapping):
        raise ConfigurationError("Original and anonymized node IDs must each be unique")
    if any(node_id not in mapping for node_id, _ in gold.evidence_turn_ids):
        raise ConfigurationError("Annotated evidence turn refers to an absent source node")
    sources = tuple(
        replace(
            source,
            node_id=mapping[source.node_id],
            metadata={"date": source.metadata["date"]} if "date" in source.metadata else {},
        )
        for source in case.sources
    )
    return replace(case, sources=sources), replace(
        gold,
        source_aliases={
            mapping[source.node_id]: gold.source_aliases.get(source.node_id, source.node_id)
            for source in case.sources
        },
        evidence_turn_ids=tuple(
            (mapping[node_id], turn_id) for node_id, turn_id in gold.evidence_turn_ids
        ),
    )


def node_coverage_metrics(
    candidate_node_ids: Sequence[str],
    selected_node_ids: Sequence[str],
    gold: GoldRecord,
    *,
    max_seed_nodes: int = 3,
) -> dict[str, Any]:
    """Score one owner per candidate hit, preserving alias and annotation limits.

    Repeated session occurrences count once against the original annotated node.
    Unlabeled selections are not necessarily irrelevant: annotations need not be
    exhaustive. Abstention and absent positive labels have no recall denominator.
    """
    candidates = list(dict.fromkeys(candidate_node_ids))
    selected = list(dict.fromkeys(selected_node_ids))
    aliases = gold.source_aliases
    candidate_labels = {aliases.get(node, node) for node in candidates}
    selected_labels = {aliases.get(node, node) for node in selected}
    required = set(gold.evidence_node_ids)
    scorable = gold.ability != "abstention" and bool(required)
    counts = Counter(candidate_node_ids)
    total = len(candidate_node_ids)
    return {
        "required_node_count": len(required) if gold.ability != "abstention" else None,
        "capacity_exceeded": len(required) > max_seed_nodes if scorable else None,
        "candidate_node_count": len(candidates),
        "selected_node_count": len(selected),
        "candidate_node_recall": len(required & candidate_labels) / len(required)
        if scorable
        else None,
        "selected_node_recall": len(required & selected_labels) / len(required)
        if scorable
        else None,
        "all_required_candidates": required <= candidate_labels if scorable else None,
        "all_required_selected": required <= selected_labels if scorable else None,
        "candidate_misses": sorted(required - candidate_labels) if scorable else None,
        "selection_misses": sorted((required & candidate_labels) - selected_labels)
        if scorable
        else None,
        "unlabeled_selected_node_ids": [
            node for node in selected if aliases.get(node, node) not in required
        ],
        "duplicate_concentration": 1 - len(candidates) / total if total else None,
        "top_owner_share": max(counts.values()) / total if total else None,
    }


def _hit_record(hit) -> dict[str, Any]:
    """Retain rankings and canonical references without duplicating source text."""
    return {
        "passage_id": hit.passage.passage_id,
        "retrieved_passage_id": hit.passage.metadata.get("retrieved_passage_id"),
        "rank": hit.rank,
        "score": hit.score,
        "references": [reference_to_dict(ref) for ref in hit.passage.refs],
    }


class _RecordingRetriever:
    """Observe actual backend results and require the declared indexed passages."""

    def __init__(self, retriever: Retriever, passages: Sequence[SearchPassage]):
        """Keep the underlying retriever caller-owned and index passage identities."""
        self.retriever = retriever
        self.passages = {passage.passage_id: passage for passage in passages}
        self.hits = []
        self.seconds = 0.0

    def descriptor(self) -> dict[str, Any]:
        """Forward the real backend descriptor without changing its identity."""
        return self.retriever.descriptor()

    async def search(self, query: str, k: int):
        """Call the backend independently for this cutoff and preserve its ranking."""
        started = time.perf_counter()
        self.hits = await self.retriever.search(query, k)
        self.seconds = time.perf_counter() - started
        if len(self.hits) > k:
            raise ConfigurationError("Backend exceeded requested passage count")
        for hit in self.hits:
            if self.passages.get(hit.passage.passage_id) != hit.passage:
                raise ConfigurationError("Backend returned a passage outside its declared corpus")
        return self.hits


class _RecordingEvidence(Evidence):
    """Observe canonical hits while retaining the application's actual search path."""

    async def search(self, query: str, k: int = 10):
        """Delegate search and canonical resolution unchanged, retaining elapsed time."""
        started = time.perf_counter()
        self.hits = await super().search(query, k)
        self.search_seconds = time.perf_counter() - started
        return self.hits


def _validate_corpus(case: EvaluationCase, passages: Sequence[SearchPassage]) -> None:
    """Require unique source identities and valid single-owner canonical passages."""
    check_passages(passages)
    if len({source.node_id for source in case.sources}) != len(case.sources):
        raise ConfigurationError("Case source node IDs must be unique")
    turns = {
        (source.node_id, turn.turn_id): turn.text
        for source in case.sources
        for turn in source.turns
    }
    for passage in passages:
        if not passage.refs or any(not isinstance(ref, SourceSpan) for ref in passage.refs):
            raise ConfigurationError("Controlled node search requires source-span passages")
        if len({ref.node_id for ref in passage.refs}) != 1:
            raise ConfigurationError("Controlled node search requires one owning node per passage")
        for ref in passage.refs:
            original = turns.get((ref.node_id, ref.turn_id))
            if original is None or not 0 <= ref.start <= ref.end <= len(original):
                raise ConfigurationError("Passage reference falls outside this case's sources")
            if original[ref.start : ref.end] not in passage.text:
                raise ConfigurationError(
                    "Indexed passage does not contain its canonical source span"
                )


async def evaluate_node_search(
    case: EvaluationCase,
    gold: GoldRecord,
    passages: Sequence[SearchPassage],
    retriever: Retriever,
    workspace_path: str | Path,
    *,
    retrieval_k: int,
    max_seed_nodes: int = 3,
    warm_repetitions: int = 5,
) -> dict[str, Any]:
    """Run real seed retrieval once, then independently repeat for warm timing.

    ``workspace_path`` must be new or empty. Original node IDs, turns and metadata
    are imported unchanged; evaluator labels are never ingested. The injected
    backend must report the same ordered corpus fingerprint. Setup covers local
    import and evidence preparation, excluding caller-owned index build/open.
    First and warm timings include canonical resolution separately from backend
    search. Each repeat calls the backend at ``retrieval_k``; none slices top 40.

    No generation clients are constructed: ``None`` is safe because the invoked
    application method ends before inference. Any exception propagates so the
    caller can preserve the failed trial. The retriever remains caller-owned.
    """
    started = time.perf_counter()
    if case.case_id != gold.case_id:
        raise ConfigurationError("Case and evaluator identities must match")
    if type(warm_repetitions) is not int or warm_repetitions < 0:
        raise ConfigurationError("warm_repetitions must be a nonnegative integer")
    directory = Path(workspace_path)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ConfigurationError("Node-search workspace must be new or empty")
    _validate_corpus(case, passages)
    fingerprint = corpus_fingerprint(passages)
    descriptor = retriever.descriptor()
    recorded_fingerprint = descriptor.get("corpus_fingerprint", descriptor.get("corpus_sha256"))
    if recorded_fingerprint != fingerprint or descriptor.get("passage_count") != len(passages):
        raise ConfigurationError("Retriever descriptor does not match the supplied passage corpus")
    recorded = _RecordingRetriever(retriever, passages)
    async with Workspace.open(directory) as workspace:
        for source in case.sources:
            result = await workspace.ingest(
                Conversation(
                    source.turns,
                    node_id=source.node_id,
                    metadata=source.metadata,
                    timestamp_ms=source.timestamp_ms,
                )
            )
            if result.node_id != source.node_id or await workspace.source(result.node_id) != source:
                raise ConfigurationError(
                    "Workspace import changed a source identity or its content"
                )
        application = LLGM(
            workspace, None, None, retrieval_k=retrieval_k, max_seed_nodes=max_seed_nodes
        )
        async with await _RecordingEvidence.open(workspace, recorded) as evidence:
            setup_seconds = time.perf_counter() - started
            samples = []
            for repetition in range(warm_repetitions + 1):
                ledger = RunLedger(Budget(max_searches=1), byte_token_bound)
                selection_started = time.perf_counter()
                seeds = await application._seeds(case.question, None, evidence, ledger)
                selection_seconds = time.perf_counter() - selection_started
                if ledger.calls or ledger.searches != 1:
                    raise ConfigurationError("Retrieval-only evaluation invoked unexpected work")
                owners = [hit.passage.refs[0].node_id for hit in evidence.hits]
                selected = [seed.node_id for seed in seeds]
                samples.append(
                    {
                        "repetition": repetition,
                        "raw_hits": [_hit_record(hit) for hit in recorded.hits],
                        "hits": [_hit_record(hit) for hit in evidence.hits],
                        "candidate_node_ids": list(dict.fromkeys(owners)),
                        "candidate_owner_ids_by_hit": owners,
                        "selected_node_ids": selected,
                        "selected_seeds": [
                            {
                                "node_id": seed.node_id,
                                "references": [reference_to_dict(ref) for ref in seed.references],
                            }
                            for seed in seeds
                        ],
                        "metrics": node_coverage_metrics(
                            owners, selected, gold, max_seed_nodes=max_seed_nodes
                        ),
                        "backend_search_seconds": recorded.seconds,
                        "evidence_search_seconds": evidence.search_seconds,
                        "search_and_selection_seconds": selection_seconds,
                        "ledger": {
                            "model_calls": ledger.calls,
                            "searches": ledger.searches,
                            "events": ledger.events,
                        },
                    }
                )
    first, warm = samples[0], samples[1:]
    first_ids = [hit["passage_id"] for hit in first["raw_hits"]]
    return {
        "schema_version": 1,
        "case_id": case.case_id,
        "question": case.question,
        "ability": gold.ability,
        "status": "completed",
        "retrieval_k": retrieval_k,
        "max_seed_nodes": max_seed_nodes,
        "corpus_fingerprint": fingerprint,
        "source_count": len(case.sources),
        "passage_count": len(passages),
        "retriever": descriptor,
        "workspace_setup_seconds": setup_seconds,
        "first": first,
        "warm": warm,
        "rank_consistent": all(
            [hit["passage_id"] for hit in sample["raw_hits"]] == first_ids for sample in warm
        ),
        "selection_consistent": all(
            sample["selected_node_ids"] == first["selected_node_ids"] for sample in warm
        ),
        "warm_median_seconds": {
            key: statistics.median(sample[key] for sample in warm) if warm else None
            for key in (
                "backend_search_seconds",
                "evidence_search_seconds",
                "search_and_selection_seconds",
            )
        },
        "generation_calls": 0,
        "search_calls": len(samples),
        "currency_cost": None,
    }
