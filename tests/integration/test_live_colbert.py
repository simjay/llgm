"""Real optional ColBERT build/search/reopen, with explicit local asset pins."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

import pytest

from llgm.core.types import SourceSpan
from llgm.evaluation.scoring import evidence_coverage
from llgm.retrieval.colbert import ColBERTConfig, ColBERTRetriever, preflight_colbert
from llgm.retrieval.passages import split_nodes
from llgm.retrieval.tokenizers import ColBERTTokenizer

pytestmark = [pytest.mark.integration, pytest.mark.dataset, pytest.mark.colbert]


def test_small_workspace_hybrid_refresh_and_reopen(live_colbert, tmp_path):
    """Real ColBERT serves a tiny workspace and preserves hybrid ranks across an append and restart."""
    from llgm.retrieval.live import WorkspaceIndexService

    config = ColBERTConfig(**live_colbert, index_root=tmp_path, index_name="live", query_maxlen=128)
    service = WorkspaceIndexService(tmp_path / "workspaces", config)
    records = [
        {
            "node_id": "atlas",
            "turns": [
                {
                    "turn_id": "t1",
                    "role": "user",
                    "text": "Atlas database backups are retained for seven days.",
                }
            ],
            "metadata": {},
            "timestamp_ms": None,
        }
    ]
    prepared = service.prepare(records)
    assert prepared["descriptor"]["components"][1]["backend"] == "colbertv2_exact"
    first = service.search(prepared["index_id"], "How long are backups kept?", 12)
    assert len(first["hits"]) == 1
    assert first["hits"][0]["passage"]["refs"][0]["turn_id"] == "t1"
    records[0]["turns"].append(
        {"turn_id": "t2", "role": "user", "text": "Restore tests run every Friday."}
    )
    changed = service.prepare(records)
    assert changed["index_id"] != prepared["index_id"]
    newest = service.search(changed["index_id"], "When are restore tests?", 12)
    assert len(newest["hits"]) == 2
    restarted = WorkspaceIndexService(service.root, config)
    reopened = restarted.search(changed["index_id"], "When are restore tests?", 12)
    assert [h["passage_id"] for h in reopened["hits"]] == [h["passage_id"] for h in newest["hits"]]
    assert [h["score"] for h in reopened["hits"]] == pytest.approx(
        [h["score"] for h in newest["hits"]]
    )


def test_colbert_build_search_reopen(live_colbert, longmemeval, tmp_path, integration_record):
    """A real pinned PLAID index preserves top-40 passage identities after reopen."""
    record, _ = integration_record
    config = ColBERTConfig(
        **live_colbert, index_root=tmp_path, index_name="longmemeval-sanity", query_maxlen=128
    )
    problems = preflight_colbert(config)
    assert not problems, "Opted-in ColBERT preflight failed: " + "; ".join(problems)
    tokenizer = ColBERTTokenizer(config.checkpoint_path)
    case = longmemeval.cases["001be529"]
    passages = split_nodes(case.sources, tokenizer, window=config.doc_maxlen, overlap=32)
    assert len(passages) > 40
    record.update(
        dataset=longmemeval.manifest,
        case_id=case.case_id,
        passage_count=len(passages),
        configuration=asdict(config),
        physical_model_requests=0,
    )
    built = ColBERTRetriever.build(passages, config=config, tokenizer=tokenizer)
    first = asyncio.run(built.search(case.question, 40))
    assert len(first) == 40
    assert len({hit.passage.passage_id for hit in first}) == 40
    corpus_ids = {passage.passage_id for passage in passages}
    assert all(hit.passage.passage_id in corpus_ids for hit in first)
    record["operations"].append(
        {"operation": "build_search", "descriptor": built.descriptor(), "events": built.events}
    )
    # Open the persisted index through the public API; do not silently rebuild.
    reopened = ColBERTRetriever.open(passages, config=config, tokenizer=tokenizer)
    second = asyncio.run(reopened.search(case.question, 40))
    assert [hit.passage.passage_id for hit in second] == [hit.passage.passage_id for hit in first]
    assert [hit.score for hit in second] == pytest.approx([hit.score for hit in first])
    source_turns = {
        (node.node_id, turn.turn_id): turn.text for node in case.sources for turn in node.turns
    }
    for hit in (*first, *second):
        assert len(hit.passage.refs) == 1
        reference = hit.passage.refs[0]
        assert isinstance(reference, SourceSpan)
        original = source_turns[(reference.node_id, reference.turn_id)]
        assert 0 <= reference.start < reference.end <= len(original)
        _, separator, body = hit.passage.text.partition("\n")
        assert separator, "Rendered source passage must have a metadata header"
        assert body == original[reference.start : reference.end]
    record["retrieved_source_spans_verified"] = len(first) + len(second)
    record["operations"].append(
        {
            "operation": "reopen_search",
            "descriptor": reopened.descriptor(),
            "events": reopened.events,
        }
    )
    record["coverage"] = evidence_coverage(
        [ref for hit in second for ref in hit.passage.refs], asdict(longmemeval.gold[case.case_id])
    )
