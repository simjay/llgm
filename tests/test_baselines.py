"""Reader citation integrity and explicit input admission without hosted calls."""

import asyncio
import json

import pytest

from llgm.core.types import Conversation, SourceNode, Turn
from llgm.evaluation import baselines
from llgm.evaluation.longmemeval import EvaluationCase
from llgm.memory.workspace import Workspace
from llgm.models import ScriptedModelClient


def _case():
    """Use speaker-distinct history with an irrelevant late turn and evaluator-only identity."""
    return EvaluationCase(
        "evaluator-only-case-id",
        "Which region uses jade?",
        "2031-04-07",
        (
            SourceNode(
                "opaque-node",
                (
                    Turn("first", "user", "Jade uses eu-west-1. Exact  spacing 🦋."),
                    Turn("later", "assistant", "An unrelated long note. " * 100),
                ),
                {"date": "2031-04-01"},
            ),
        ),
    )


async def _workspace(path, case):
    """Persist real immutable sources for local FTS and canonical reads."""
    workspace = Workspace.open(path)
    for source in case.sources:
        await workspace.ingest(
            Conversation(source.turns, node_id=source.node_id, metadata=source.metadata)
        )
    return workspace


def _character_count(request):
    """Substitute a deterministic counter to test admission, not tokenizer accuracy."""
    return sum(len(message.content) for message in request.messages)


@pytest.mark.parametrize("citation,expected_status", [("e1", "completed"), ("e999", "failed")])
def test_bm25_citations_must_match_delivered_exact_source_records(
    tmp_path, monkeypatch, citation, expected_status
):
    """A reader cannot mint evidence; accepted quotes preserve canonical roles, dates and text."""
    monkeypatch.setattr(baselines, "_input_tokens", _character_count)

    async def scenario():
        """Exercise the actual SQLite retrieval and source resolution path."""
        case = _case()
        model = ScriptedModelClient([json.dumps({"answer": "eu-west-1", "citations": [citation]})])
        async with await _workspace(tmp_path / "workspace", case) as workspace:
            result = await baselines.answer_baseline(
                "bm25",
                workspace,
                case,
                model,
                {
                    "retrieval_k": 5,
                    "passage_chars": 1024,
                    "max_input_tokens": 10000,
                    "max_output_tokens": 64,
                },
            )
        assert result["status"] == expected_status
        delivered = json.loads(model.requests[0].messages[-1].content)
        assert case.case_id not in model.requests[0].messages[-1].content
        first = delivered["evidence"][0]
        assert first["text"] == case.sources[0].turns[0].text
        assert (first["role"], first["date"]) == ("user", "2031-04-01")
        assert first["reference"] == {
            "type": "source_span",
            "node_id": "opaque-node",
            "turn_id": "first",
            "start": 0,
            "end": len(first["text"]),
        }
        assert result["evidence"] == ([first] if citation == "e1" else [])
        if citation != "e1":
            assert result["answer"] == ""
            assert result["trace"][-1]["kind"] == "invalid_response"

    asyncio.run(scenario())


def test_full_context_does_not_silently_clip_history(tmp_path, monkeypatch):
    """An oversized complete history produces an explicit failure before any model dispatch."""
    monkeypatch.setattr(baselines, "_input_tokens", _character_count)

    async def scenario():
        """Keep a short first turn that could fit while the complete second turn cannot."""
        case = _case()
        model = ScriptedModelClient([])
        async with await _workspace(tmp_path / "workspace", case) as workspace:
            result = await baselines.answer_baseline(
                "full_context",
                workspace,
                case,
                model,
                {
                    "retrieval_k": 5,
                    "passage_chars": 1024,
                    "max_input_tokens": 1200,
                    "max_output_tokens": 64,
                },
            )
        assert result["status"] == "budget_exceeded"
        assert result["answer"] == "" and result["evidence"] == []
        assert model.requests == []
        admission = result["trace"][0]
        assert admission["original_input_tokens"] > admission["max_input_tokens"]
        assert admission["available_records"] == 2
        assert admission["omitted_records"] == 0

    asyncio.run(scenario())


def test_reader_can_explain_abstention_without_a_citation(tmp_path, monkeypatch):
    """Insufficient evidence can produce a judgeable explanation without invented support."""
    monkeypatch.setattr(baselines, "_input_tokens", _character_count)

    async def scenario():
        """Accept natural-language insufficiency while preserving an empty citation set."""
        case = _case()
        answer = "The supplied conversation does not establish which city Jade is based in."
        model = ScriptedModelClient([json.dumps({"answer": answer, "citations": []})])
        async with await _workspace(tmp_path / "workspace", case) as workspace:
            result = await baselines.answer_baseline(
                "bm25",
                workspace,
                case,
                model,
                {
                    "retrieval_k": 5,
                    "passage_chars": 1024,
                    "max_input_tokens": 10000,
                    "max_output_tokens": 64,
                },
            )
        assert result["status"] == "completed"
        assert result["answer"] == answer
        assert result["evidence"] == []

    asyncio.run(scenario())
