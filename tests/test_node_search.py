"""Exercise the actual seed path with real SQLite retrieval and canonical storage."""

import asyncio
import json
from dataclasses import asdict, replace

import pytest

from llgm.core.errors import ConfigurationError
from llgm.core.types import SourceNode, SourceSpan, Turn
from llgm.evaluation.longmemeval import EvaluationCase, GoldRecord
from llgm.evaluation.node_search import anonymize_case, evaluate_node_search, node_coverage_metrics
from llgm.memory.workspace import Workspace
from llgm.retrieval.base import SearchPassage
from llgm.retrieval.bm25 import SQLiteBM25Retriever
from llgm.retrieval.passages import split_nodes
from llgm.retrieval.tokenizers import DiagnosticTokenizer


def _annotated_occurrences():
    """Retain repeated physical sessions whose original identifiers reveal labels."""
    sources = (
        SourceNode(
            "answer_session#occurrence-0",
            (Turn("t00000", "user", "Quartz café: 한글 🦋.\n  Preserve spacing."),),
            {"date": "2026-08-01", "benchmark_session_id": "answer_session", "has_answer": True},
            1785542400000,
        ),
        SourceNode(
            "answer_session#occurrence-1",
            (Turn("t00000", "assistant", "Quartz follow-up confirmed."),),
            {
                "date": "2026-08-02",
                "benchmark_session_id": "answer_session",
                "extra": "answer_hint",
            },
        ),
        SourceNode(
            "plain_session",
            (Turn("t00000", "user", "Rainy weather."),),
            {"benchmark_session_id": "plain_session"},
        ),
    )
    case = EvaluationCase("identity-case", "quartz", "2026-09-11", sources)
    gold = GoldRecord(
        case.case_id,
        "EVALUATOR_ONLY_VALUE",
        "single-session-user",
        ("answer_session",),
        ((sources[0].node_id, "t00000"), (sources[1].node_id, "t00000")),
        {sources[0].node_id: "answer_session", sources[1].node_id: "answer_session"},
    )
    return case, gold


def test_anonymization_preserves_sources_and_repeated_session_scoring():
    """Opaque physical IDs remove header cues while aliases retain original gold meaning."""
    original, original_gold = _annotated_occurrences()
    before = json.dumps(asdict(original), ensure_ascii=False, sort_keys=True)
    case, gold = anonymize_case(original, original_gold)
    assert anonymize_case(original, original_gold) == (case, gold)
    assert (case.case_id, case.question, case.question_date) == (
        original.case_id,
        original.question,
        original.question_date,
    )
    opaque_ids = {source.node_id for source in case.sources}
    assert len(opaque_ids) == len(original.sources)
    assert opaque_ids.isdisjoint(source.node_id for source in original.sources)
    assert all(isinstance(node_id, str) and node_id.strip() for node_id in opaque_ids)
    for old, new in zip(original.sources, case.sources):
        assert new.turns == old.turns
        assert new.timestamp_ms == old.timestamp_ms
        assert new.metadata == ({"date": old.metadata["date"]} if "date" in old.metadata else {})
        assert gold.source_aliases[new.node_id] == original_gold.source_aliases.get(
            old.node_id, old.node_id
        )
    assert gold.evidence_node_ids == original_gold.evidence_node_ids
    assert gold.evidence_turn_ids == tuple(
        (source.node_id, "t00000") for source in case.sources[:2]
    )
    assert (gold.answer, gold.ability) == (original_gold.answer, original_gold.ability)
    assert json.dumps(asdict(original), ensure_ascii=False, sort_keys=True) == before
    visible = json.dumps(asdict(case), ensure_ascii=False)
    assert "answer_" not in visible
    assert "benchmark_session_id" not in visible
    assert "EVALUATOR_ONLY_VALUE" not in visible
    for source in original.sources:
        assert source.node_id not in visible
    metrics = node_coverage_metrics(
        [source.node_id for source in case.sources[:2]], [case.sources[1].node_id], gold
    )
    assert metrics["required_node_count"] == 1
    assert metrics["candidate_node_recall"] == metrics["selected_node_recall"] == 1


def test_anonymized_case_runs_through_real_bm25_seed_path(tmp_path):
    """The production passage renderer and seed path resolve opaque IDs to original labels."""
    original, original_gold = _annotated_occurrences()
    case, gold = anonymize_case(original, original_gold)
    passages = split_nodes(case.sources, DiagnosticTokenizer())
    for passage in passages:
        assert "answer_" not in passage.text
        assert all(source.node_id not in passage.text for source in original.sources)
    retriever = SQLiteBM25Retriever.from_passages(passages)
    try:
        result = asyncio.run(
            evaluate_node_search(
                case,
                gold,
                passages,
                retriever,
                tmp_path / "opaque",
                retrieval_k=12,
                warm_repetitions=0,
            )
        )
    finally:
        retriever.close()
    assert result["first"]["metrics"]["all_required_selected"] is True
    assert result["first"]["metrics"]["required_node_count"] == 1
    assert result["generation_calls"] == 0


@pytest.mark.parametrize("fault", ["duplicate-node", "wrong-case", "absent-turn-node"])
def test_anonymization_rejects_ambiguous_or_unresolvable_labels(fault):
    """Invalid input cannot silently collapse source identities or misattribute scoring."""
    case, gold = _annotated_occurrences()
    if fault == "duplicate-node":
        case = replace(case, sources=case.sources + (case.sources[0],))
    elif fault == "wrong-case":
        gold = replace(gold, case_id="different")
    else:
        gold = replace(gold, evidence_turn_ids=(("missing-node", "t00000"),))
    with pytest.raises(ConfigurationError):
        anonymize_case(case, gold)


def _crowded_case():
    """Build a tied lexical ranking whose later candidates include labeled nodes."""
    sources, passages = [], []
    for node, count in (("crowded", 20), ("a", 1), ("b", 1), ("c", 1), ("gold", 1)):
        turns = tuple(
            Turn(f"t{index}", "user", "quartz decision evidence") for index in range(count)
        )
        sources.append(SourceNode(node, turns, {"date": "2026-09-11"}))
        for turn in turns:
            passages.append(
                SearchPassage(
                    f"{node}-{turn.turn_id}",
                    turn.text,
                    (SourceSpan(node, turn.turn_id, 0, len(turn.text)),),
                )
            )
    case = EvaluationCase("crowding", "quartz", "2026-09-11", tuple(sources))
    gold = GoldRecord("crowding", "EVALUATOR_ONLY_SECRET", "multi-session", ("a", "gold"), ())
    return case, gold, passages


def test_real_seed_path_distinguishes_candidate_and_selection_misses(tmp_path):
    """Actual top-12 crowding and top-40 admission expose distinct failure locations."""
    case, gold, passages = _crowded_case()
    retriever = SQLiteBM25Retriever.from_passages(passages)

    async def run():
        """Execute both requested cutoffs independently over the same live index."""
        results = []
        for k in (12, 40):
            results.append(
                await evaluate_node_search(
                    case,
                    gold,
                    passages,
                    retriever,
                    tmp_path / f"k{k}",
                    retrieval_k=k,
                    warm_repetitions=2,
                )
            )
        return results

    try:
        small, large = asyncio.run(run())
        assert [event["requested_k"] for event in retriever.events] == [12, 12, 12, 40, 40, 40]
    finally:
        retriever.close()
    assert small["first"]["selected_node_ids"] == ["crowded"]
    assert small["first"]["metrics"]["candidate_misses"] == ["a", "gold"]
    assert small["first"]["metrics"]["selection_misses"] == []
    assert large["first"]["selected_node_ids"] == ["crowded", "a", "b"]
    assert large["first"]["metrics"]["candidate_node_recall"] == 1
    assert large["first"]["metrics"]["selected_node_recall"] == 0.5
    assert large["first"]["metrics"]["selection_misses"] == ["gold"]
    assert large["first"]["metrics"]["all_required_candidates"] is True
    assert large["first"]["metrics"]["all_required_selected"] is False
    assert large["first"]["metrics"]["duplicate_concentration"] == pytest.approx(1 - 5 / 24)
    assert large["first"]["metrics"]["top_owner_share"] == pytest.approx(20 / 24)
    assert large["first"]["metrics"]["unlabeled_selected_node_ids"] == ["crowded", "b"]
    for result in (small, large):
        assert result["rank_consistent"] and result["selection_consistent"]
        assert result["generation_calls"] == 0
        assert result["search_calls"] == 3
        assert result["currency_cost"] is None
        assert result["workspace_setup_seconds"] > 0
        assert result["warm_median_seconds"]["evidence_search_seconds"] > 0
        for sample in [result["first"], *result["warm"]]:
            assert sample["ledger"]["model_calls"] == 0
            assert sample["ledger"]["searches"] == 1
            assert sample["ledger"]["events"][0]["selected"] == sample["selected_node_ids"]
            assert sample["hits"][0]["retrieved_passage_id"] == sample["raw_hits"][0]["passage_id"]
            assert sample["search_and_selection_seconds"] >= sample["evidence_search_seconds"]
        assert "EVALUATOR_ONLY_SECRET" not in json.dumps(result, allow_nan=False)

    async def read_import():
        """Check original metadata and turn identities survived real persistence."""
        async with Workspace.open(tmp_path / "k40") as workspace:
            return {source.node_id: source for source in await workspace.sources()}

    restored = asyncio.run(read_import())
    assert restored == {source.node_id: source for source in case.sources}


def test_aliases_count_original_sessions_without_merging_physical_seeds():
    """Two occurrences can occupy separate seeds but share one annotated session."""
    gold = GoldRecord(
        "case",
        None,
        "multi-session",
        ("session", "other"),
        (),
        {"session#0": "session", "session#1": "session"},
    )
    metrics = node_coverage_metrics(
        ["session#0", "session#0", "session#1", "other"],
        ["session#0", "session#1"],
        gold,
    )
    assert metrics["required_node_count"] == 2
    assert metrics["candidate_node_count"] == 3
    assert metrics["selected_node_count"] == 2
    assert metrics["candidate_node_recall"] == 1
    assert metrics["selected_node_recall"] == 0.5
    assert metrics["selection_misses"] == ["other"]
    assert metrics["duplicate_concentration"] == 0.25


def test_capacity_limit_is_distinguished_from_retrieval_failure():
    """Four required nodes cannot all be direct seeds under a three-node cap."""
    gold = GoldRecord("case", None, "multi-session", ("a", "b", "c", "d"), ())
    metrics = node_coverage_metrics(["a", "b", "c", "d"], ["a", "b", "c"], gold)
    assert metrics["capacity_exceeded"] is True
    assert metrics["candidate_node_recall"] == 1
    assert metrics["selected_node_recall"] == 0.75
    assert metrics["candidate_misses"] == []
    assert metrics["selection_misses"] == ["d"]


@pytest.mark.parametrize("ability, required", [("abstention", ("a",)), ("single-session", ())])
def test_unscorable_cases_do_not_become_perfect_recall(ability, required):
    """Abstention or missing positive annotations never contributes recall success."""
    gold = GoldRecord("case", None, ability, required, ())
    metrics = node_coverage_metrics([], [], gold)
    for key in (
        "candidate_node_recall",
        "selected_node_recall",
        "all_required_candidates",
        "all_required_selected",
        "capacity_exceeded",
        "candidate_misses",
        "selection_misses",
        "duplicate_concentration",
        "top_owner_share",
    ):
        assert metrics[key] is None


def test_empty_real_search_returns_zero_positive_recall(tmp_path):
    """No lexical match is a measured empty candidate set and triggers no generation."""
    case, gold, passages = _crowded_case()
    case = replace(case, question="unmatchedword")
    retriever = SQLiteBM25Retriever.from_passages(passages)
    try:
        result = asyncio.run(
            evaluate_node_search(
                case,
                gold,
                passages,
                retriever,
                tmp_path / "empty",
                retrieval_k=12,
                warm_repetitions=0,
            )
        )
    finally:
        retriever.close()
    assert result["first"]["raw_hits"] == []
    assert result["first"]["selected_seeds"] == []
    assert result["first"]["metrics"]["candidate_node_recall"] == 0
    assert result["first"]["metrics"]["all_required_selected"] is False
    assert all(value is None for value in result["warm_median_seconds"].values())


def test_mismatched_index_corpus_is_rejected_before_search(tmp_path):
    """A real index cannot silently stand in for different supplied passages."""
    case, gold, passages = _crowded_case()
    retriever = SQLiteBM25Retriever.from_passages(passages[:-1])
    try:
        with pytest.raises(ConfigurationError, match="descriptor"):
            asyncio.run(
                evaluate_node_search(
                    case,
                    gold,
                    passages,
                    retriever,
                    tmp_path / "mismatch",
                    retrieval_k=12,
                )
            )
        assert retriever.events == []
    finally:
        retriever.close()


@pytest.mark.parametrize("fault", ["unknown-node", "past-end", "wrong-text", "multiple-owners"])
def test_invalid_canonical_corpus_is_rejected(tmp_path, fault):
    """Reject references or rendered text that cannot represent the declared source."""
    case, gold, passages = _crowded_case()
    original = passages[0]
    if fault == "unknown-node":
        changed = replace(original, refs=(SourceSpan("missing", "t0", 0, 2),))
    elif fault == "past-end":
        changed = replace(original, refs=(SourceSpan("crowded", "t0", 0, 999),))
    elif fault == "wrong-text":
        changed = replace(original, text="unsupported")
    else:
        changed = replace(original, refs=original.refs + passages[-1].refs)
    passages = [changed, *passages[1:]]
    retriever = SQLiteBM25Retriever.from_passages(passages)
    try:
        with pytest.raises(ConfigurationError):
            asyncio.run(
                evaluate_node_search(
                    case,
                    gold,
                    passages,
                    retriever,
                    tmp_path / fault,
                    retrieval_k=12,
                )
            )
        assert retriever.events == []
    finally:
        retriever.close()


def test_nonempty_workspace_cannot_add_journals_or_other_sources(tmp_path):
    """The controlled comparison refuses a workspace whose state is not isolated."""
    case, gold, passages = _crowded_case()
    (tmp_path / "existing").write_text("preserve")
    retriever = SQLiteBM25Retriever.from_passages(passages)
    try:
        with pytest.raises(ConfigurationError, match="new or empty"):
            asyncio.run(
                evaluate_node_search(
                    case,
                    gold,
                    passages,
                    retriever,
                    tmp_path,
                    retrieval_k=12,
                )
            )
        assert (tmp_path / "existing").read_text() == "preserve"
    finally:
        retriever.close()
