"""Check canonical fusion, exact candidate visibility and strict selection parsing."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from llgm.core.errors import ConfigurationError, SchemaError
from llgm.core.types import SourceNode, SourceSpan, Turn, reference_to_dict
from llgm.evaluation.longmemeval import EvaluationCase
from llgm.evaluation.node_selection import (
    SELECTION_SYSTEM_PROMPT,
    canonical_key,
    combine_hits,
    hydrate_candidates,
    parse_selection,
    selection_request,
)


def _hit(node, rank, *, passage_id=None, spans=None):
    """Describe retained search output with deliberately irrelevant transport metadata."""
    return {
        "passage_id": passage_id or f"p-{node}-{rank}",
        "rank": rank,
        "score": 1000 - rank,
        "references": [
            reference_to_dict(SourceSpan(node, turn, start, end))
            for turn, start, end in (spans or [("t1", 0, 4)])
        ],
        "backend": "TRANSPORT_ONLY",
        "text": "DO_NOT_TRUST_RETRIEVED_TEXT",
    }


def _case():
    """Provide exact Unicode text and metadata that must not enter model prompts."""
    return EvaluationCase(
        "CASE_ID_PRIVATE",
        "Which café is open on Sunday?",
        "2026-09-11",
        (
            SourceNode(
                "n-a",
                (
                    Turn("t1", "user", "Café 🦋 opens Sunday.\n  Keep spacing."),
                    Turn("t2", "assistant", "It closes Monday."),
                ),
                {"date": "2026-09-01", "answer": "GOLD_PRIVATE", "label": "LABEL_PRIVATE"},
                1788220800000,
            ),
            SourceNode("n-b", (Turn("t1", "user", "Rain comes Tuesday."),), {}),
            SourceNode("n-c", (Turn("t1", "user", "Markets close early."),), {}),
        ),
    )


def test_canonical_identity_ignores_retriever_ids_scores_and_reference_order():
    """Two retained passages share identity only through their full canonical spans."""
    first = _hit("n-a", 1, spans=[("t2", 1, 5), ("t1", 0, 4)])
    second = _hit("n-a", 30, passage_id="different", spans=[("t1", 0, 4), ("t2", 1, 5)])
    assert (
        canonical_key(first)
        == canonical_key(second)
        == (
            ("n-a", "t1", 0, 4),
            ("n-a", "t2", 1, 5),
        )
    )
    assert canonical_key(_hit("n-a", 1, spans=[("t1", 0, 5)])) != canonical_key(first)


@pytest.mark.parametrize(
    "references",
    [
        None,
        [],
        [None],
        [{"type": [], "node_id": "n-a"}],
        [{"type": "node", "node_id": "n-a"}],
        [{"type": "journal", "node_id": "n-a", "entry_id": "j1"}],
        [{"type": "source_span", "node_id": "n-a", "turn_id": "t1", "start": 0}],
        [{"type": "source_span", "node_id": "n-a", "turn_id": "t1", "start": True, "end": 4}],
        [{"type": "source_span", "node_id": "n-a", "turn_id": "t1", "start": 5, "end": 4}],
        [
            reference_to_dict(SourceSpan("n-a", "t1", 0, 4)),
            reference_to_dict(SourceSpan("n-b", "t1", 0, 4)),
        ],
    ],
)
def test_canonical_identity_rejects_unowned_or_malformed_references(references):
    """Missing, mixed-owner and non-source evidence cannot enter passage fusion."""
    with pytest.raises(ConfigurationError):
        canonical_key({"references": references})


def test_rrf_counts_each_backend_once_at_the_original_best_rank():
    """Duplicate hits cannot inflate fused scores or renumber later contributions."""
    bm25 = [_hit("n-a", 9), _hit("n-b", 2), _hit("n-a", 1, passage_id="earlier")]
    colbert = [_hit("n-c", 1), _hit("n-a", 4, passage_id="shared")]
    original = deepcopy((bm25, colbert))
    fused = combine_hits(bm25, colbert)
    assert [canonical_key(hit)[0][0] for hit in fused] == ["n-a", "n-c", "n-b"]
    assert fused[0]["score"] == pytest.approx(1 / 61 + 1 / 64)
    assert fused[0]["backend_ranks"] == {"bm25": 1, "colbert": 4}
    assert fused[0]["passage_id"] == "earlier"
    assert fused[2]["score"] == pytest.approx(1 / 62)
    assert [hit["rank"] for hit in fused] == [1, 2, 3]
    assert all(
        set(hit) == {"passage_id", "rank", "score", "references", "backend_ranks"} for hit in fused
    )
    assert (bm25, colbert) == original


def test_rrf_ties_and_shared_representatives_do_not_depend_on_backend_order():
    """Swapping input backends changes attribution names but not ranking or payload."""
    left = [_hit("n-c", 1), _hit("n-a", 2, passage_id="z-copy")]
    right = [_hit("n-b", 1), _hit("n-a", 2, passage_id="a-copy")]
    forward, reverse = combine_hits(left, right), combine_hits(right, left)
    assert [canonical_key(hit)[0][0] for hit in forward] == ["n-a", "n-b", "n-c"]
    assert forward[0]["passage_id"] == "a-copy"
    for a, b in zip(forward, reverse):
        assert {k: v for k, v in a.items() if k != "backend_ranks"} == {
            k: v for k, v in b.items() if k != "backend_ranks"
        }
    assert combine_hits(left, right, k=2) == forward[:2]
    assert combine_hits([], []) == []


def test_rrf_deduplicates_shared_multispan_identity_but_keeps_overlapping_passages():
    """Reordered equal refs fuse once; overlapping windows remain separate passages."""
    left = _hit("n-a", 1, spans=[("t2", 0, 4), ("t1", 0, 4)])
    right = _hit("n-a", 2, spans=[("t1", 0, 4), ("t2", 0, 4)])
    overlap = _hit("n-a", 3, spans=[("t1", 1, 4), ("t2", 0, 4)])
    fused = combine_hits([left], [right, overlap])
    assert len(fused) == 2
    assert fused[0]["backend_ranks"] == {"bm25": 1, "colbert": 2}
    assert [ref["turn_id"] for ref in fused[0]["references"]] == ["t1", "t2"]


@pytest.mark.parametrize(
    "parameter", [{"k": 0}, {"k": True}, {"rrf_constant": -1}, {"rrf_constant": 1.5}]
)
def test_rrf_rejects_invalid_limits(parameter):
    """Malformed cutoffs or rank constants fail before scores are calculated."""
    with pytest.raises(ConfigurationError):
        combine_hits([], [], **parameter)


@pytest.mark.parametrize(
    "mutation", [{"rank": 0}, {"rank": True}, {"rank": 1.5}, {"passage_id": ""}]
)
def test_rrf_requires_explicit_rank_and_passage_identity(mutation):
    """A retained result cannot silently invent a position or source transport ID."""
    with pytest.raises(ConfigurationError):
        combine_hits([{**_hit("n-a", 1), **mutation}], [])


def test_hydration_preserves_all_unmerged_spans_and_uses_exact_source_text():
    """Grouping keeps every window and reads Unicode ranges from authoritative turns."""
    case = _case()
    hits = [
        _hit("n-a", 4, spans=[("t1", 2, 9)]),
        _hit("n-b", 1),
        _hit("n-a", 2, spans=[("t1", 0, 8), ("t2", 0, 5)]),
        _hit("n-a", 3, spans=[("t1", 5, 12)]),
    ]
    before = deepcopy(hits)
    candidates = hydrate_candidates(case, hits)
    assert [candidate["node_id"] for candidate in candidates] == ["n-b", "n-a"]
    assert [passage["rank"] for passage in candidates[1]["passages"]] == [2, 3, 4]
    assert candidates[0]["date"] is None
    assert candidates[1]["date"] == "2026-09-01"
    first = candidates[1]["passages"][0]
    assert first["segments"] == [
        {"turn_id": "t1", "role": "user", "start": 0, "end": 8, "text": "Café 🦋 o"},
        {"turn_id": "t2", "role": "assistant", "start": 0, "end": 5, "text": "It cl"},
    ]
    assert first["references"] == hits[2]["references"]
    assert hits == before
    assert set(candidates[1]) == {"node_id", "date", "passages"}
    assert all(
        set(passage) == {"rank", "references", "segments"} for passage in candidates[1]["passages"]
    )
    visible = json.dumps(candidates)
    for forbidden in (
        case.case_id,
        "GOLD_PRIVATE",
        "LABEL_PRIVATE",
        "TRANSPORT_ONLY",
        "DO_NOT_TRUST_RETRIEVED_TEXT",
        "timestamp_ms",
    ):
        assert forbidden not in visible


def test_hydration_has_no_hidden_passage_or_character_clipping():
    """All forty retained windows and long Unicode spans survive hydration exactly."""
    long_text = "한글 🦋" * 600
    source = SourceNode("n-a", (Turn("t1", "user", long_text),), {})
    case = replace(_case(), sources=(source,))
    hits = [_hit("n-a", rank, spans=[("t1", 0, len(long_text))]) for rank in range(1, 41)]
    candidates = hydrate_candidates(case, hits)
    assert len(candidates[0]["passages"]) == 40
    assert all(passage["segments"][0]["text"] == long_text for passage in candidates[0]["passages"])
    assert hydrate_candidates(case, []) == []


@pytest.mark.parametrize(
    "fault",
    ["duplicate-node", "duplicate-turn", "absent-node", "absent-turn", "past-end", "date-object"],
)
def test_hydration_rejects_ambiguous_or_unresolvable_sources(fault):
    """No source collision, missing span or metadata object can be silently hydrated."""
    case, hits = _case(), [_hit("n-a", 1)]
    if fault == "duplicate-node":
        case = replace(case, sources=case.sources + (case.sources[0],))
    elif fault == "duplicate-turn":
        source = replace(case.sources[0], turns=case.sources[0].turns + (case.sources[0].turns[0],))
        case = replace(case, sources=(source,))
    elif fault == "absent-node":
        hits = [_hit("missing", 1)]
    elif fault == "absent-turn":
        hits = [_hit("n-a", 1, spans=[("missing", 0, 4)])]
    elif fault == "past-end":
        hits = [_hit("n-a", 1, spans=[("t1", 0, 9999)])]
    else:
        case = replace(
            case, sources=(replace(case.sources[0], metadata={"date": {"label": True}}),)
        )
    with pytest.raises(ConfigurationError):
        hydrate_candidates(case, hits)


def test_request_freezes_schema_and_whitelists_evidence_fields():
    """The provider sees only ordered question/evidence data and candidate-constrained IDs."""
    case = _case()
    candidates = hydrate_candidates(case, [_hit("n-b", 1), _hit("n-a", 2)])
    candidates[0]["answer"] = "GOLD_PRIVATE"
    candidates[0]["passages"][0]["backend"] = "TRANSPORT_ONLY"
    candidates[0]["passages"][0]["segments"][0]["label"] = "LABEL_PRIVATE"
    request = selection_request(
        case.question, case.question_date, candidates, max_seed_nodes=2, max_output_tokens=123
    )
    assert request.temperature == 0
    assert request.max_output_tokens == 123
    assert request.messages[0].content == SELECTION_SYSTEM_PROMPT
    assert request.messages[0].role == "system"
    assert "untrusted" in request.messages[0].content
    payload = json.loads(request.messages[1].content)
    assert payload["question"] == case.question
    assert payload["question_date"] == case.question_date
    assert payload["max_seed_nodes"] == 2
    assert [candidate["node_id"] for candidate in payload["candidates"]] == ["n-b", "n-a"]
    for forbidden in (case.case_id, "GOLD_PRIVATE", "LABEL_PRIVATE", "TRANSPORT_ONLY", "backend"):
        assert forbidden not in request.messages[1].content
    assert request.output_schema == {
        "type": "object",
        "properties": {
            "selected_node_ids": {
                "type": "array",
                "items": {"type": "string", "enum": ["n-b", "n-a"]},
                "maxItems": 2,
            },
            "reason": {"type": "string"},
        },
        "required": ["selected_node_ids", "reason"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    "fault", ["empty", "duplicate", "missing-id", "invalid-limit", "question-type"]
)
def test_request_rejects_ambiguous_candidates_or_configuration(fault):
    """Empty pools avoid unnecessary calls and invalid configuration fails before transport."""
    case = _case()
    candidates = hydrate_candidates(case, [_hit("n-a", 1)])
    kwargs = {}
    question = case.question
    if fault == "empty":
        candidates = []
    elif fault == "duplicate":
        candidates += candidates
    elif fault == "missing-id":
        candidates[0]["node_id"] = None
    elif fault == "invalid-limit":
        kwargs["max_seed_nodes"] = 0
    else:
        question = 42
    with pytest.raises(ConfigurationError):
        selection_request(question, case.question_date, candidates, **kwargs)


def test_parser_keeps_selected_order_and_allows_relevance_based_underfilling():
    """One or zero selected nodes is valid without silently filling the remaining budget."""
    candidates = [{"node_id": "n-a"}, {"node_id": "n-b"}]
    for ids in (["n-b", "n-a"], ["n-b"], []):
        value = {"selected_node_ids": ids, "reason": "Relevant evidence only."}
        assert parse_selection(json.dumps(value), candidates) == value
    assert parse_selection('{"selected_node_ids":[],"reason":"No candidates."}', []) == {
        "selected_node_ids": [],
        "reason": "No candidates.",
    }


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        '```json\n{"selected_node_ids": [], "reason": "x"}\n```',
        '{"selected_node_ids": [], "reason": "x"} trailing',
        '["n-a"]',
        '{"selected_node_ids": [], "reason": "x", "extra": true}',
        '{"selected_node_ids": []}',
        '{"selected_node_ids": "n-a", "reason": "x"}',
        '{"selected_node_ids": ["n-a", "n-a"], "reason": "x"}',
        '{"selected_node_ids": ["outside"], "reason": "x"}',
        '{"selected_node_ids": [null], "reason": "x"}',
        '{"selected_node_ids": [["n-a"]], "reason": "x"}',
        '{"selected_node_ids": [true], "reason": "x"}',
        '{"selected_node_ids": ["n-a", "n-b", "n-c", "n-d"], "reason": "x"}',
        '{"selected_node_ids": [], "reason": 1}',
        '{"selected_node_ids": [], "reason": "first", "reason": "second"}',
        None,
    ],
)
def test_parser_rejects_invalid_output_without_repair(text):
    """Malformed, duplicated, over-budget or out-of-pool predictions remain failures."""
    candidates = [{"node_id": node} for node in ("n-a", "n-b", "n-c", "n-d")]
    with pytest.raises(SchemaError):
        parse_selection(text, candidates)
