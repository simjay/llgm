"""Real-corpus persistence/retrieval checks; no answer model and no paid calls."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

import pytest

from llgm.core.errors import ConfigurationError
from llgm.core.types import Conversation, JournalRef, NodeRef, Provenance, SourceSpan, Turn
from llgm.evaluation.scoring import evidence_coverage
from llgm.memory.evidence import Evidence, interpret_journal
from llgm.memory.workspace import Workspace
from llgm.retrieval.bm25 import SQLiteBM25Retriever
from llgm.retrieval.passages import split_nodes
from llgm.retrieval.tokenizers import DiagnosticTokenizer

pytestmark = [pytest.mark.integration, pytest.mark.dataset]
CASE_IDS = ("001be529", "01493427", "00ca467f", "031748ae_abs")
EVALUATOR_KEYS = {
    "has_answer",
    "answer",
    "answer_session_ids",
    "question_type",
    "evidence_node_ids",
    "evidence_turn_ids",
}


def _keys(value):
    """Collect structural keys without treating conversation text as evaluator metadata."""
    if isinstance(value, dict):
        return set(value).union(*(_keys(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(_keys(item) for item in value))
    return set()


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_full_case_ingest_search_restart_and_spans(
    longmemeval, case_id, tmp_path, integration_record
):
    """Full real histories preserve source spans and retrieval identities across restart."""
    record, _ = integration_record
    case = longmemeval.cases[case_id]
    record["dataset"] = longmemeval.manifest
    record["case_id"] = case_id
    record["physical_model_requests"] = 0
    record["tokenizer"] = DiagnosticTokenizer().descriptor()
    assert not (_keys([asdict(source) for source in case.sources]) & EVALUATOR_KEYS)
    assert all(set(source.metadata) == {"date", "benchmark_session_id"} for source in case.sources)

    async def scenario():
        """Ingest the complete case and audit every derived span before reopening both stores."""
        directory = tmp_path / "workspace"
        async with Workspace.open(directory) as workspace:
            for source in case.sources:
                result = await workspace.ingest(
                    Conversation(
                        source.turns, source.node_id, source.metadata, source.timestamp_ms
                    ),
                    idempotency_key=source.node_id,
                )
                assert result.created and result.node_id == source.node_id
            sources = await workspace.sources()
            expected = {source.node_id: source for source in case.sources}
            assert len(sources) == len(expected)
            for source in sources:
                assert source.turns == expected[source.node_id].turns
                assert source.metadata == expected[source.node_id].metadata
                assert source.timestamp_ms == expected[source.node_id].timestamp_ms
            passages = split_nodes(sources, DiagnosticTokenizer(), window=180, overlap=32)
            assert len(passages) > 40, "A top-40 check must not retrieve the complete tiny corpus"
            assert not (_keys([asdict(p) for p in passages]) & EVALUATOR_KEYS)
            record["source_count"] = len(sources)
            record["passage_count"] = len(passages)
            record["evidence_schema"] = "immutable-node-v2"
            record["read_policy"] = "current"
            index_path = tmp_path / "bm25.sqlite3"
            retriever = SQLiteBM25Retriever.from_passages(passages, index_path)
            try:
                hits = await retriever.search(case.question, 40)
                identities = [(hit.passage.passage_id, hit.score, hit.rank) for hit in hits]
                assert len(hits) <= 40
                assert len({hit.passage.passage_id for hit in hits}) == len(hits)
                async with await Evidence.open(workspace, retriever) as evidence:
                    wrapped_hits = await evidence.search(case.question, 40)
                    assert [hit.passage.refs for hit in wrapped_hits] == [
                        hit.passage.refs for hit in hits
                    ]
                    for hit in wrapped_hits:
                        expected_text = "\n\n".join(
                            [(await evidence.read(ref)).text for ref in hit.passage.refs]
                        )
                        assert hit.passage.text == expected_text
                record["retriever"] = retriever.descriptor()
                record["operations"].extend(retriever.events)
            finally:
                retriever.close()
            # Validate every generated span, not only successful retrieved hits.
            checked = 0
            for passage in passages:
                for ref in passage.refs:
                    resolved = await workspace.resolve(ref)
                    turn = next(
                        turn for turn in expected[ref.node_id].turns if turn.turn_id == ref.turn_id
                    )
                    assert resolved.text == turn.text[ref.start : ref.end]
                    assert passage.text.endswith(resolved.text)
                    checked += 1
            record["resolved_source_spans"] = checked
            record["coverage"] = {
                str(k): evidence_coverage(
                    [ref for hit in hits[:k] for ref in hit.passage.refs],
                    asdict(longmemeval.gold[case_id]),
                )
                for k in (10, 40)
            }
            # Retrieval scores are diagnostics, not a release-wide quality gate.
            for coverage in record["coverage"].values():
                for key in ("source_recall", "answer_turn_recall"):
                    assert coverage[key] is None or 0 <= coverage[key] <= 1
            record["operations"].append(
                {
                    "operation": "ingest_and_resolve",
                    "ingested": len(sources),
                    "spans_resolved": checked,
                }
            )

        async with Workspace.open(directory) as restarted:
            reopened_sources = await restarted.sources()
            assert reopened_sources == sources
            reopened_passages = split_nodes(
                reopened_sources, DiagnosticTokenizer(), window=180, overlap=32
            )
            reopened = SQLiteBM25Retriever.from_passages(reopened_passages, index_path)
            try:
                actual = await reopened.search(case.question, 40)
                assert [
                    (hit.passage.passage_id, hit.score, hit.rank) for hit in actual
                ] == identities
                record["operations"].extend(reopened.events)
            finally:
                reopened.close()
            # A derived index must reject different corpus identities after restart.
            with pytest.raises(ConfigurationError, match="different corpus"):
                SQLiteBM25Retriever.from_passages(reopened_passages[:-1], index_path)
            record["operations"].append(
                {
                    "operation": "restart",
                    "source_identity_preserved": True,
                    "retrieval_identity_preserved": True,
                    "changed_corpus_rejected": True,
                }
            )

    asyncio.run(scenario())


def test_real_source_survives_new_node_and_journal_overwrite(
    longmemeval, tmp_path, integration_record
):
    """New evidence changes journal interpretation without replacing the original real text."""
    record, _ = integration_record
    source = longmemeval.cases["01493427"].sources[0]
    record.update(dataset=longmemeval.manifest, case_id="01493427", physical_model_requests=0)

    async def scenario():
        """Publish a synthetic update and observe it through an already-open evidence handle."""
        path = tmp_path / "workspace"
        async with Workspace.open(path) as workspace:
            await workspace.ingest(
                Conversation(source.turns, source.node_id, source.metadata, source.timestamp_ms)
            )
            evidence = await Evidence.open(workspace)
            first_turn = source.turns[0]
            old_ref = SourceSpan(source.node_id, first_turn.turn_id, 0, len(first_turn.text))
            try:
                provenance = Provenance("user", "synthetic-storage-contract-v2")
                original = await workspace.append_journal(
                    source.node_id,
                    subject=old_ref,
                    relation="value",
                    value=old_ref,
                    provenance=provenance,
                )
                update_text = "Synthetic zirconium update for immutable-node journal testing."
                update = await workspace.ingest(
                    Conversation(
                        (Turn("update", "user", update_text),),
                        metadata={"purpose": "synthetic contract check; not a benchmark fact"},
                    )
                )
                assert update.created and update.node_id != source.node_id
                new_ref = SourceSpan(update.node_id, "update", 0, len(update_text))
                overwrite = await workspace.append_journal(
                    source.node_id,
                    subject=old_ref,
                    relation="value",
                    value=new_ref,
                    record_kind="overwrite",
                    provenance=provenance,
                )
                interpreted = interpret_journal(await evidence.journal(source.node_id))
                assert interpreted.active == (overwrite,)
                assert interpreted.inactive == (original,)
                assert (await evidence.read(old_ref)).text == first_turn.text
                assert (await evidence.read(new_ref)).text == update_text
                assert (await evidence.source(source.node_id)).turns == source.turns
                hits = await evidence.search("zirconium", 10)
                assert any(new_ref in hit.passage.refs for hit in hits)
                overwrite_ref = JournalRef(source.node_id, overwrite.entry_id)
                assert (await evidence.read(overwrite_ref)).metadata["record_kind"] == "overwrite"
            finally:
                await evidence.close()
        async with Workspace.open(path) as restarted:
            assert (await restarted.resolve(old_ref)).text == first_turn.text
            assert (await restarted.source(source.node_id)).turns == source.turns
            assert (await restarted.resolve(new_ref)).text == update_text
            assert (await restarted.resolve(NodeRef(update.node_id))).reference == NodeRef(
                update.node_id
            )
            assert (await restarted.resolve(overwrite_ref)).metadata["record_kind"] == "overwrite"
            interpreted = interpret_journal(await restarted.inspect_journal(source.node_id))
            assert interpreted.active == (overwrite,)
            assert interpreted.inactive == (original,)
        record["operations"].append(
            {
                "operation": "new_node_journal_overwrite_restart",
                "source_node_ids": [source.node_id, update.node_id],
                "original_source_retained": True,
                "open_handle_observes_new_evidence": True,
                "journal_overwrite_preserved": True,
                "synthetic_update_is_benchmark_evidence": False,
            }
        )

    asyncio.run(scenario())
