"""Compare frozen seed choices through the application's complete answer pipeline.

Only initial retrieval is replayed. Delegates use the ordinary workspace BM25
index for subsequent searches, actual Docker interpreters, and the unchanged
recursive runtime. Evaluator labels and selector explanations never enter it.
"""

from __future__ import annotations

from typing import Any

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.core.types import SourceSpan, reference_from_dict, reference_to_dict
from llgm.evaluation.answer_judging import cited_source_evidence
from llgm.evaluation.longmemeval import EvaluationCase
from llgm.evaluation.node_selection import canonical_key, hydrate_candidates
from llgm.inference.budget import Budget
from llgm.inference.nodes import NodeSeed
from llgm.inference.repl import DockerREPLConfig
from llgm.llgm import LLGM
from llgm.memory.workspace import Workspace
from llgm.models.base import ModelClient

_RUNTIME_OPTIONS = {
    "max_depth",
    "max_steps",
    "max_operations",
    "max_concurrency",
    "passage_chars",
    "max_journal_bytes",
    "retrieval_k",
    "max_seed_nodes",
}


def _frozen_seeds(
    case: EvaluationCase, selected_node_ids: list[str], hits: list[dict[str, Any]]
) -> tuple[NodeSeed, ...]:
    """Validate the pool and retain every exact deduplicated span in rank order."""
    candidates = hydrate_candidates(case, hits)
    if not isinstance(selected_node_ids, list) or any(
        not isinstance(node_id, str) or not node_id for node_id in selected_node_ids
    ):
        raise ConfigurationError("Selected node IDs must be a list of nonempty strings")
    if len(selected_node_ids) > 3 or len(set(selected_node_ids)) != len(selected_node_ids):
        raise ConfigurationError("Select at most three distinct source nodes")
    if set(selected_node_ids) - {candidate["node_id"] for candidate in candidates}:
        raise ConfigurationError("Selected node is absent from the frozen candidate pool")
    turns = {
        (source.node_id, turn.turn_id): turn.text
        for source in case.sources
        for turn in source.turns
    }
    by_node: dict[str, list[SourceSpan]] = {node_id: [] for node_id in selected_node_ids}
    for hit in sorted(hits, key=lambda item: item["rank"]):
        canonical_key(hit)
        for value in hit["references"]:
            reference = reference_from_dict(value)
            if "text" in hit and (
                not isinstance(hit["text"], str)
                or turns[(reference.node_id, reference.turn_id)][reference.start : reference.end]
                not in hit["text"]
            ):
                raise ConfigurationError("Saved passage text disagrees with its canonical span")
            references = by_node.get(reference.node_id)
            if references is not None and reference not in references:
                references.append(reference)
    return tuple(NodeSeed(node_id, tuple(by_node[node_id])) for node_id in selected_node_ids)


class _FrozenSeedLLGM(LLGM):
    """Replace only initial seed admission with validated experimental choices."""

    def __init__(self, *args, seeds: tuple[NodeSeed, ...], **kwargs):
        """Keep the frozen handles separate from all ordinary runtime options."""
        super().__init__(*args, **kwargs)
        self.frozen_seeds = seeds

    async def _seeds(self, question, node_id, evidence, ledger):
        """Charge one initial search while recording its explicit replay provenance."""
        if node_id is not None:
            raise ConfigurationError("Frozen seed trials cannot override the starting node")
        if ledger.searches >= ledger.budget.max_searches:
            raise BudgetExceeded("Initial retrieval search allowance exhausted")
        ledger.searches += 1
        ledger.events.append(
            {
                "kind": "seed_selection",
                "source": "frozen_replay",
                "selected": [seed.node_id for seed in self.frozen_seeds],
                "skipped": [],
            }
        )
        return list(self.frozen_seeds)


async def _validate_workspace(workspace: Workspace, case: EvaluationCase) -> None:
    """Require the full exact case history without labels, links, or journal edits."""
    if set(await workspace.source_ids()) != {source.node_id for source in case.sources}:
        raise ConfigurationError("Answer workspace must contain the full case history only")
    for source in case.sources:
        if set(source.metadata) - {"date"}:
            raise ConfigurationError("Experimental source metadata may contain only its date")
        if await workspace.source(source.node_id) != source:
            raise ConfigurationError("Answer workspace changed a case source or its metadata")
        if await workspace.inspect_journal(source.node_id) or await workspace.edges(source.node_id):
            raise ConfigurationError("Frozen seed comparison requires an unmodified source graph")


async def answer_from_seeds(
    workspace: Workspace,
    case: EvaluationCase,
    selected_node_ids: list[str],
    hits: list[dict[str, Any]],
    root_model: ModelClient,
    sidecar_model: ModelClient,
    *,
    budget: Budget,
    runtime_options: dict[str, Any],
    repl_config: DockerREPLConfig,
) -> dict[str, Any]:
    """Answer from frozen seeds with caller-owned sources, clients, and one budget.

    The caller supplies the full opaque, date-only case workspace. Empty selection
    remains empty. Every selected owner's saved span is supplied to its delegate;
    the ordinary runtime may subsequently read, search, and recurse. Initial
    retrieval consumes one search allowance but incurs no replayed retrieval time.
    This helper never runs maintenance or accepts alternate evidence/interpreters.
    Returned evidence contains only records actually cited by the final root.
    """
    if not isinstance(budget, Budget) or not isinstance(repl_config, DockerREPLConfig):
        raise ConfigurationError("Answer trials require Budget and DockerREPLConfig records")
    if not isinstance(runtime_options, dict) or set(runtime_options) - _RUNTIME_OPTIONS:
        raise ConfigurationError("Unsupported frozen-answer runtime option")
    if runtime_options.get("max_seed_nodes", 3) != 3:
        raise ConfigurationError("Frozen answer comparison requires a three-node seed cap")
    seeds = _frozen_seeds(case, selected_node_ids, hits)
    await _validate_workspace(workspace, case)
    application = _FrozenSeedLLGM(
        workspace,
        root_model,
        sidecar_model,
        seeds=seeds,
        inference_budget=budget,
        capture_text=True,
        repl_config=repl_config,
        **runtime_options,
    )
    result = await application.answer(case.question, query_date=case.question_date, budget=budget)
    return {
        "status": result.status,
        "answer": result.answer,
        "unresolved": list(result.evidence.unresolved),
        "references": [reference_to_dict(reference) for reference in result.references],
        "cited_evidence": cited_source_evidence(case, result),
        "usage": result.usage,
        "trace": result.trace,
        "seed_node_ids": list(selected_node_ids),
    }
