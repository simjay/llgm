"""Frozen seed contracts, finite transport replay, and opted-in actual Docker execution."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from dataclasses import replace

import pytest

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.core.types import Conversation, SourceNode, SourceSpan, Turn, reference_to_dict
from llgm.evaluation.longmemeval import EvaluationCase
from llgm.evaluation.seed_answers import (
    _frozen_seeds,
    _FrozenSeedLLGM,
    answer_from_seeds,
)
from llgm.inference.budget import Budget, RunLedger, byte_token_bound
from llgm.inference.repl import DockerREPLConfig
from llgm.memory.workspace import Workspace
from llgm.models import CallableModelClient, ModelResponse
from tests.node_support import CHILD, READ, SEARCH, Models, ReplayFactory, finish


def _case():
    """Keep source role/date and exact Unicode text separate from evaluator identity."""
    return EvaluationCase(
        "EVALUATOR_CASE_ID",
        "Which region is jade in?",
        "2031-04-07",
        (
            SourceNode(
                "a",
                (
                    Turn("first", "user", "🦋 café\n\nexact  spacing"),
                    Turn("later", "assistant", "jade"),
                ),
                {"date": "2031-04-01"},
            ),
            SourceNode("b", (Turn("turn", "user", "jade eu-west-1"),), {"date": "2031-04-02"}),
            SourceNode("c", (Turn("turn", "user", "Unrelated document"),), {}),
            SourceNode("d", (Turn("turn", "user", "Fourth document"),), {}),
        ),
    )


def _hit(node_id="a", turn_id="first", start=0, end=6, rank=1):
    """Encode an actual saved-hit shape, including data that must not reach the model."""
    return {
        "passage_id": f"p-{rank}",
        "rank": rank,
        "score": 1.0,
        "references": [reference_to_dict(SourceSpan(node_id, turn_id, start, end))],
        "pool": "EVALUATOR_POOL_ONLY",
        "reason": "SELECTOR_REASON_ONLY",
    }


def _budget():
    """Allow complete local metadata and nested calls under a finite shared budget."""
    return Budget(
        max_model_calls=13,
        max_sidecar_calls=12,
        max_searches=8,
        max_evidence_tokens=65536,
        max_context_tokens=65536,
        max_bundle_tokens=16000,
        max_output_tokens=1536,
        timeout_seconds=120,
    )


async def _ingest(workspace, case):
    """Publish exactly the supplied model-visible sources without maintenance."""
    for source in case.sources:
        await workspace.ingest(
            Conversation(
                source.turns,
                node_id=source.node_id,
                metadata=source.metadata,
                timestamp_ms=source.timestamp_ms,
            )
        )


async def _answer(workspace, case, selected, hits, models, **options):
    """Supply real application defaults while allowing contract-specific arguments."""
    return await answer_from_seeds(
        workspace,
        case,
        selected,
        hits,
        models.root,
        models.sidecar,
        budget=options.pop("budget", _budget()),
        runtime_options=options.pop("runtime_options", {"max_steps": 8}),
        repl_config=options.pop("repl_config", DockerREPLConfig()),
        **options,
    )


def test_frozen_spans_preserve_selection_rank_and_reference_order():
    """Repeated handles deduplicate exactly while overlapping spans remain distinct."""
    first = _hit(rank=1)
    first["references"].append(reference_to_dict(SourceSpan("a", "later", 1, 4)))
    later = _hit(start=2, end=8, rank=3)
    seeds = _frozen_seeds(
        _case(), ["b", "a"], [later, first, _hit("b", "turn", 0, 4, 2), copy.deepcopy(first)]
    )
    assert [seed.node_id for seed in seeds] == ["b", "a"]
    assert seeds[1].references == (
        SourceSpan("a", "first", 0, 6),
        SourceSpan("a", "later", 1, 4),
        SourceSpan("a", "first", 2, 8),
    )
    assert _frozen_seeds(_case(), [], [first]) == ()


@pytest.mark.parametrize(
    "selection", [["missing"], ["b"], ["a", "a"], "a", [None], [""], ["a", "b", "c", "d"]]
)
def test_invalid_seed_choices_fail_before_inference(selection):
    """Absent, out-of-pool, duplicate and excessive selections cannot alter the trial."""
    with pytest.raises(ConfigurationError):
        _frozen_seeds(_case(), selection, [_hit()])


@pytest.mark.parametrize(
    "failure",
    ["node", "turn", "bounds", "type", "rank", "text", "malformed_text", "empty", "mixed"],
)
def test_invalid_candidate_references_and_saved_text_fail(failure):
    """Every candidate is checked, including a malformed unselected passage."""
    hit = _hit(rank=2)
    if failure in {"node", "turn"}:
        hit["references"][0][failure + "_id"] = "missing"
    elif failure == "bounds":
        hit["references"][0]["end"] = 10000
    elif failure == "type":
        hit["references"] = [{"type": "node", "node_id": "a"}]
    elif failure == "rank":
        hit["rank"] = True
    elif failure in {"text", "malformed_text"}:
        hit["text"] = "incorrect text" if failure == "text" else None
    elif failure == "empty":
        hit["references"] = []
    else:
        hit["references"].append(reference_to_dict(SourceSpan("b", "turn", 0, 4)))
    with pytest.raises(ConfigurationError):
        _frozen_seeds(_case(), [], [_hit(), hit])


def test_initial_replay_charges_one_search_and_preserves_empty_selection():
    """Even empty replay consumes its declared initial search without falling back."""

    async def scenario():
        """Exercise the isolated override without opening evidence or constructing models."""
        application = _FrozenSeedLLGM(None, None, None, seeds=())
        ledger = RunLedger(Budget(max_searches=1), byte_token_bound)
        assert await application._seeds("question", None, None, ledger) == []
        assert ledger.searches == 1 and ledger.calls == 0
        assert ledger.events == [
            {"kind": "seed_selection", "source": "frozen_replay", "selected": [], "skipped": []}
        ]
        with pytest.raises(BudgetExceeded):
            await application._seeds("question", None, None, ledger)
        with pytest.raises(ConfigurationError):
            await application._seeds("question", "a", None, ledger)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["missing", "extra", "content", "metadata"])
def test_workspace_must_match_full_case_without_hidden_metadata(tmp_path, failure):
    """Validation rejects altered histories before any model or interpreter starts."""

    async def scenario():
        """Change exactly one workspace property, keeping all source handles valid."""
        case = _case()
        imported = case
        if failure == "missing":
            imported = replace(case, sources=case.sources[:-1])
        elif failure == "extra":
            imported = replace(
                case, sources=(*case.sources, replace(case.sources[0], node_id="extra"))
            )
        elif failure == "content":
            imported = replace(
                case,
                sources=(
                    replace(case.sources[0], turns=(Turn("first", "user", "Changed"),)),
                    *case.sources[1:],
                ),
            )
        else:
            case = imported = replace(
                case,
                sources=(
                    replace(case.sources[0], metadata={"answer_hint": "hidden"}),
                    *case.sources[1:],
                ),
            )
        models = Models()
        async with Workspace.open(tmp_path / "workspace") as workspace:
            await _ingest(workspace, imported)
            with pytest.raises(ConfigurationError):
                await _answer(workspace, case, ["a"], [_hit()], models)
        assert not models.child_requests and not models.root_requests

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "options",
    [
        {"repl_factory": lambda: None},
        {"evidence_factory": None},
        {"capture_text": False},
        {"max_seed_nodes": 4},
        [],
    ],
)
def test_runtime_options_cannot_replace_interpreters_evidence_or_seed_cap(tmp_path, options):
    """Trials accept runtime limits while rejecting alternate execution machinery."""

    async def scenario():
        """Reject forbidden options before requiring an initialized workspace."""
        with pytest.raises(ConfigurationError):
            await _answer(None, _case(), [], [], Models(), runtime_options=options)

    asyncio.run(scenario())


def test_empty_selection_runs_real_application_without_model_calls(tmp_path):
    """An abstaining selector stays empty instead of receiving hidden retrieval fallback."""

    async def scenario():
        """Run full preparation and ordinary no-seed return without creating Docker sessions."""
        case, models = _case(), Models()
        async with Workspace.open(tmp_path / "workspace") as workspace:
            await _ingest(workspace, case)
            result = await _answer(workspace, case, [], [_hit()], models)
            assert await workspace.source("a") == case.sources[0]
        assert result["status"] == "partial" and result["answer"] == ""
        assert result["unresolved"] == ["No seed nodes retrieved"]
        assert result["references"] == result["cited_evidence"] == result["seed_node_ids"] == []
        assert result["usage"]["searches"] == 1
        assert not models.child_requests and not models.root_requests

    asyncio.run(scenario())


def test_replay_transport_keeps_only_final_citations_and_forwards_question_date(
    tmp_path, monkeypatch
):
    """Real workspace/runtime callbacks expose two reads but the final record contains only one."""

    async def scenario():
        """Replace only the transport with finite replay; no generated Python is executed."""
        replay, models, case = ReplayFactory(), Models(), _case()
        monkeypatch.setattr("llgm.inference.nodes.DockerREPL", replay)

        async def root(request):
            """Cite one branch and deliberately omit the other branch's already-read record."""
            models.root_requests.append(request)
            payload = json.loads(request.messages[1].content)
            assert payload["query_date"] == case.question_date
            return ModelResponse(
                finish("First source only", [payload["branches"][0]["citations"][0]])
            )

        models.root = CallableModelClient(root)
        async with Workspace.open(tmp_path / "workspace") as workspace:
            await _ingest(workspace, case)
            result = await _answer(
                workspace, case, ["a", "b"], [_hit(), _hit("b", "turn", 0, 13, 2)], models
            )
        assert result["status"] == "completed"
        assert len(result["cited_evidence"]) == 1
        record = result["cited_evidence"][0]
        assert record["text"] == case.sources[0].turns[0].text[:6]
        assert record["role"] == "user" and record["date"] == "2031-04-01"
        assert record["references"] == result["references"]
        assert {session.context["node_id"] for session in replay.sessions} == {"a", "b"}
        assert all(session.closed for session in replay.sessions)
        assert all(
            session.context["query_date"] == case.question_date for session in replay.sessions
        )
        requests = "\n".join(
            message.content
            for request in models.child_requests + models.root_requests
            for message in request.messages
        )
        assert all(
            sentinel not in requests
            for sentinel in (case.case_id, "EVALUATOR_POOL_ONLY", "SELECTOR_REASON_ONLY")
        )
        assert result["usage"]["searches"] == 1
        assert any(event["kind"] == "root_return" for event in result["trace"])

    asyncio.run(scenario())


def test_real_bm25_followup_and_nested_source_access_survive_frozen_seeds(tmp_path, monkeypatch):
    """Initial replay leaves ordinary BM25 search and recursive node callbacks available."""

    async def scenario():
        """Search and delegate to an unselected source using the finite transport adapter."""
        replay, models, case = ReplayFactory(), Models({"a": [READ, SEARCH, CHILD]}), _case()
        monkeypatch.setattr("llgm.inference.nodes.DockerREPL", replay)
        async with Workspace.open(tmp_path / "workspace") as workspace:
            await _ingest(workspace, case)
            result = await _answer(workspace, case, ["a"], [_hit()], models)
        assert result["status"] == "completed"
        assert result["usage"]["searches"] == 2
        assert [session.context["node_id"] for session in replay.sessions] == ["a", "b"]
        assert {reference["node_id"] for reference in result["references"]} == {"b"}
        assert result["seed_node_ids"] == ["a"]

    asyncio.run(scenario())


@pytest.mark.integration
@pytest.mark.docker
def test_actual_docker_answer_uses_saved_span_and_cleans_up(tmp_path):
    """Opted-in real Python execution reads exact sources through the full LLGM pipeline."""
    gate = os.environ.get("LLGM_TEST_DOCKER", "0")
    if gate == "0":
        pytest.skip("Set LLGM_TEST_DOCKER=1 for the actual Docker pipeline")
    if gate != "1":
        raise ConfigurationError("LLGM_TEST_DOCKER must be exactly 0 or 1")
    config = DockerREPLConfig(image=os.environ.get("LLGM_REPL_DOCKER_IMAGE", "python:3.12-slim"))

    async def scenario():
        """Use deterministic model decisions and actual Docker, without hosted model calls."""
        case = _case()
        models = Models({"a": ['import json\nprint(json.dumps(read(context["references"][0])))']})
        async with Workspace.open(tmp_path / "workspace") as workspace:
            await _ingest(workspace, case)
            result = await _answer(workspace, case, ["a"], [_hit()], models, repl_config=config)
        assert result["status"] == "completed", result["unresolved"]
        assert result["cited_evidence"][0]["text"] == case.sources[0].turns[0].text[:6]
        assert result["usage"]["searches"] == 1
        assert any(event["kind"] == "repl_closed" for event in result["trace"])

    asyncio.run(scenario())
