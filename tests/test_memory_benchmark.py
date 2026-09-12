"""Protect benchmark denominators, cost uncertainty and history-only construction."""

import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from uuid import UUID

import pytest

from llgm import LLGM, Workspace
from llgm.core.errors import ConfigurationError
from llgm.evaluation.artifacts import write_json
from llgm.evaluation.costs import Allowance
from llgm.evaluation.memory_benchmark import ingest_history, prepare, report, run_trial, summarize
from llgm.models.base import (
    CallableModelClient,
    Message,
    ModelRequest,
    ModelResponse,
    ScriptedModelClient,
)


@pytest.fixture
def pinned_history(tmp_path):
    """Pin two independent questions with repeated occurrences and evaluator-only markers."""
    repeated = [
        {"role": "user", "content": "My Atlas parcel uses a linen wrapping.", "has_answer": True},
        {
            "role": "assistant",
            "content": "I recorded the Atlas wrapping.",
            "judge_note": "LABEL_ONLY",
        },
    ]
    raw = [
        {
            "question_id": "question-one",
            "question": "QUESTION_ONLY: What wrapping did I select?",
            "question_date": "2026/09/11 (Fri) 09:00",
            "question_type": "ABILITY_ONLY",
            "answer": "EVALUATOR_GOLD_ONLY",
            "haystack_session_ids": ["answer_support", "answer_support", "background"],
            "haystack_dates": ["2026/09/01", "2026/09/02", "2026/09/03"],
            "haystack_sessions": [
                repeated,
                deepcopy(repeated),
                [{"role": "user", "content": "The Atlas parcel travels on Monday."}],
            ],
            "answer_session_ids": ["answer_support"],
        },
        {
            "question_id": "question-two",
            "question": "SECOND_QUESTION_ONLY: When does the gallery open?",
            "question_date": "2026/09/10 (Thu) 10:00",
            "question_type": "single-session-user",
            "answer": "SECOND_GOLD_ONLY",
            "haystack_session_ids": ["gallery-note"],
            "haystack_dates": ["2026/08/30"],
            "haystack_sessions": [[{"role": "user", "content": "The gallery opens at noon."}]],
            "answer_session_ids": ["gallery-note"],
        },
    ]
    dataset = tmp_path / "history.json"
    dataset.write_text(json.dumps(raw), encoding="utf-8")
    judge = tmp_path / "judge.py"
    judge.write_text("# A checksum-only fixture; preparation must not execute it.\n")
    protocol = {
        "schema_version": 1,
        "repetitions": 1,
        "claim": "synthetic-development",
        "arms": ["llgm", "bm25"],
        "case_ids": "all",
        "dataset": {
            "path": dataset.name,
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "case_count": 2,
        },
        "judge": {"path": judge.name, "sha256": hashlib.sha256(judge.read_bytes()).hexdigest()},
        "limits": {"generation_cost_cap_usd": 1.0, "judge_cost_cap_usd": 1.0},
    }
    return protocol, tmp_path


def planned_trials():
    """Supply three paired questions so missing work cannot shrink either denominator."""
    return [
        {"trial_id": f"{case_id}-{arm}", "case_id": case_id, "arm": arm}
        for case_id in ("one", "two", "three")
        for arm in ("llgm", "bm25")
    ]


def trial(case_id, arm, *, status="completed", calls=()):
    """Represent retained generation work without importing any evaluator truth."""
    return {"case_id": case_id, "arm": arm, "status": status, "model_calls": list(calls)}


def judgment(case_id, arm, correct):
    """Represent a separate completed or unresolved correctness judgment."""
    return {"case_id": case_id, "arm": arm, "correct": correct, "model_calls": []}


def test_summary_preserves_missing_and_failed_planned_questions():
    """Failed answers and absent trials stay in the full planned accuracy denominator."""
    trials = [
        trial("one", "llgm"),
        trial("two", "llgm", status="failed"),
        trial("one", "bm25"),
        trial("two", "bm25"),
        trial("three", "bm25"),
    ]
    judgments = [
        judgment("one", "llgm", True),
        judgment("two", "llgm", False),
        judgment("one", "bm25", True),
        judgment("two", "bm25", False),
        judgment("three", "bm25", None),
    ]
    results = summarize(planned_trials(), trials, judgments)
    for arm in ("llgm", "bm25"):
        assert results[arm]["planned_questions"] == 3
        assert results[arm]["correct"] == 1
        assert results[arm]["unresolved_judgments"] == 1
        assert results[arm]["accuracy"] is None
        assert results[arm]["accuracy_lower_bound"] == pytest.approx(1 / 3)
        assert results[arm]["accuracy_upper_bound"] == pytest.approx(2 / 3)
    assert results["llgm"]["attempted_questions"] == 2
    assert results["llgm"]["status_counts"] == {"completed": 1, "failed": 1}
    assert results["llgm"]["total_api_cost_usd"] is None
    assert results["bm25"]["attempted_questions"] == 3


def test_all_missing_trials_do_not_become_an_empty_successful_cohort():
    """An unstarted run retains every planned question and the full uncertainty interval."""
    results = summarize(planned_trials(), [], [])
    for row in results.values():
        assert row["planned_questions"] == row["unresolved_judgments"] == 3
        assert row["attempted_questions"] == row["correct"] == 0
        assert row["accuracy"] is None
        assert (row["accuracy_lower_bound"], row["accuracy_upper_bound"]) == (0, 1)
        assert row["total_api_cost_usd"] is None
        assert row["mean_total_api_cost_usd"] is None


def test_canceled_token_wait_is_not_a_physical_or_unknown_cost_call():
    """Queued work remains visible without charging an undispatched provider request."""
    planned = [{"trial_id": "one-llgm", "case_id": "one", "arm": "llgm"}]
    calls = [
        {"estimated_cost_usd": 0.02, "phase": "query"},
        {
            "dispatched": False,
            "status": "not_dispatched",
            "estimated_cost_usd": 0.0,
            "pacing_wait_seconds": 12.5,
            "phase": "query",
        },
    ]
    result = summarize(
        planned,
        [trial("one", "llgm", status="failed", calls=calls)],
        [judgment("one", "llgm", False)],
    )["llgm"]
    assert result["model_calls"] == result["undispatched_requests"] == 1
    assert result["recorded_requests"] == 2
    assert result["unknown_cost_calls"] == 0
    assert result["total_api_cost_usd"] == pytest.approx(0.02)
    assert result["pacing_wait_seconds"] == 12.5


def test_failed_dispatch_and_unknown_usage_remain_in_phase_costs():
    """Known failed-call charges and unknown liabilities cannot disappear from totals."""
    planned = [{"trial_id": "one-llgm", "case_id": "one", "arm": "llgm"}]
    calls = [
        {"phase": "construction", "status": "completed", "estimated_cost_usd": 0.25},
        {"phase": "query", "status": "failed", "estimated_cost_usd": 0.50},
        {
            "phase": "query",
            "status": "failed",
            "estimated_cost_usd": None,
            "reserved_cost_usd": 0.75,
        },
    ]
    row = summarize(
        planned,
        [trial("one", "llgm", status="failed", calls=calls)],
        [judgment("one", "llgm", False)],
    )["llgm"]
    assert row["accuracy"] == 0
    assert row["model_calls"] == 3
    assert row["unknown_cost_calls"] == 1
    assert row["known_api_cost_usd"] == 0.75
    assert row["known_api_cost_by_phase_usd"] == {"construction": 0.25, "query": 0.50}
    assert row["total_api_cost_usd"] is None
    assert row["mean_total_api_cost_usd"] is None
    assert row["total_cost_usd"] is None


def test_known_generation_cost_does_not_claim_total_cost_parity():
    """Measured generation prices exclude judging and leave unpriced local work unknown."""
    planned = [row for row in planned_trials() if row["case_id"] == "one"]
    trials = [
        trial("one", arm, calls=[{"phase": "query", "estimated_cost_usd": 0.5}])
        for arm in ("llgm", "bm25")
    ]
    judgments = [judgment("one", arm, True) for arm in ("llgm", "bm25")]
    for record in judgments:
        record["model_calls"] = [{"phase": "judging", "estimated_cost_usd": 4.0}]
    results = summarize(planned, trials, judgments)
    for row in results.values():
        assert row["total_api_cost_usd"] == row["mean_total_api_cost_usd"] == 0.5
        assert row["total_cost_usd"] is None
        assert "unpriced" in row["total_cost_limitation"]


@pytest.mark.parametrize("duplicated", ["planned", "trials", "judgments"])
def test_summary_rejects_duplicate_attempt_selection(duplicated):
    """Duplicate schedule entries, attempts or judgments cannot be selected after execution."""
    values = {
        "planned": [{"trial_id": "one-llgm", "case_id": "one", "arm": "llgm"}],
        "trials": [trial("one", "llgm")],
        "judgments": [judgment("one", "llgm", True)],
    }
    values[duplicated].append(deepcopy(values[duplicated][0]))
    with pytest.raises(ConfigurationError):
        summarize(**values)


@pytest.mark.parametrize("outside", ["trials", "judgments"])
def test_summary_rejects_out_of_schedule_results(outside):
    """Unscheduled answers and judgments cannot alter the frozen comparison."""
    values = {"planned": planned_trials(), "trials": [], "judgments": []}
    values[outside].append(
        trial("extra", "llgm") if outside == "trials" else judgment("extra", "llgm", True)
    )
    with pytest.raises(ConfigurationError):
        summarize(**values)


def test_summary_rejects_judgment_without_an_attempt():
    """A scheduled but unattempted question cannot gain correctness from an orphan judgment."""
    with pytest.raises(ConfigurationError):
        summarize(planned_trials(), [], [judgment("one", "llgm", True)])


@pytest.mark.parametrize("selected", [["question-one"], "all"])
def test_prepare_rejects_small_cohort_claiming_all_longmemeval_s(pinned_history, selected):
    """A valid small pinned release cannot be reported as the complete 500-question benchmark."""
    protocol, root = pinned_history
    protocol["claim"] = "LongMemEval-S"
    protocol["case_ids"] = selected
    with pytest.raises(ConfigurationError, match="500"):
        prepare(protocol, root)


def test_prepare_rejects_dataset_checksum_drift(pinned_history):
    """Even harmless byte changes invalidate the frozen dataset identity before ingestion."""
    protocol, root = pinned_history
    dataset = root / protocol["dataset"]["path"]
    dataset.write_bytes(dataset.read_bytes() + b"\n")
    with pytest.raises(ConfigurationError, match="Pinned dataset"):
        prepare(protocol, root)


def test_execute_rejects_invalid_pinned_judge_before_setup(pinned_history, monkeypatch):
    """A matching checksum cannot admit paid generation without a usable judge function."""
    from llgm.evaluation import memory_benchmark

    def unexpected_setup(*args, **kwargs):
        """Fail if invalid judging configuration reaches dependencies or provider setup."""
        pytest.fail("Invalid judge reached benchmark setup")

    for name in ("preflight_runtime", "OpenAIModelClient", "OpenAICompatibleModelClient"):
        monkeypatch.setattr(memory_benchmark, name, unexpected_setup)
    protocol, root = pinned_history
    output = root / "must-not-start"
    with pytest.raises(ConfigurationError, match="top-level prompt function"):
        asyncio.run(memory_benchmark.execute(protocol, root, output))
    assert not output.exists()


def test_prepare_separates_annotations_and_retains_explicit_selection(pinned_history):
    """Gold markers and answer-prefixed IDs stay outside normalized history records."""
    protocol, root = pinned_history
    protocol["case_ids"] = ["question-two", "question-one"]
    cases, gold = prepare(protocol, root)
    assert [case.case_id for case in cases] == protocol["case_ids"]
    first = cases[1]
    assert len(first.sources) == 3
    assert len({source.node_id for source in first.sources}) == 3
    for source in first.sources:
        assert isinstance(source.node_id, str) and source.node_id
        assert set(source.metadata) == {"date"}
    payload = json.dumps([asdict(source) for source in first.sources])
    for hidden in (
        "QUESTION_ONLY",
        "EVALUATOR_GOLD_ONLY",
        "ABILITY_ONLY",
        "LABEL_ONLY",
        "has_answer",
        "answer_support",
        "benchmark_session_id",
    ):
        assert hidden not in payload
    assert gold[first.case_id].answer == "EVALUATOR_GOLD_ONLY"
    assert len(gold[first.case_id].evidence_turn_ids) == 2


@pytest.mark.parametrize("organize", [False, True])
def test_ingest_history_preserves_occurrences_roles_dates_and_no_question(pinned_history, organize):
    """Normal ingestion keeps repeated source occurrences and exposes only source data to maintenance."""
    protocol, root = pinned_history
    cases, _ = prepare(protocol, root)
    sources = cases[0].sources
    sources = (replace(sources[0], timestamp_ms=1788220800000), *sources[1:])

    async def scenario():
        """Exercise the real workspace and facade with generation replaced only at its boundary."""
        main = ScriptedModelClient([])
        maintenance = ScriptedModelClient(['{"links": []}'] * 3)
        async with Workspace.open(root / "workspace") as workspace:
            app = LLGM(workspace, main, main, maintenance_model=maintenance)
            stored, outcomes = await ingest_history(app, sources, organize=organize)
            assert len(stored) == len(outcomes) == 3
            assert len(set(await workspace.source_ids())) == 3
            assert len({source.node_id for source in stored}) == 3
            assert not {source.node_id for source in stored} & {
                source.node_id for source in sources
            }
            for expected, actual in zip(sources, stored):
                persisted = await workspace.source(actual.node_id)
                UUID(actual.node_id)
                assert persisted.turns == expected.turns
                assert persisted.metadata == expected.metadata
                assert persisted.timestamp_ms == expected.timestamp_ms
            assert stored[0].turns == stored[1].turns
            assert stored[0].metadata["date"] != stored[1].metadata["date"]
            assert [turn.role for turn in stored[0].turns] == ["user", "assistant"]
        assert not main.requests
        assert bool(maintenance.requests) is organize
        assert all(row["status"] == ("completed" if organize else "disabled") for row in outcomes)
        payload = json.dumps([asdict(request) for request in maintenance.requests])
        for hidden in (
            "QUESTION_ONLY",
            "EVALUATOR_GOLD_ONLY",
            "ABILITY_ONLY",
            "LABEL_ONLY",
            "has_answer",
            "answer_support",
            "question-one",
        ):
            assert hidden not in payload

    asyncio.run(scenario())


def test_ingest_cancellation_retains_prior_maintenance_outcomes(pinned_history):
    """An interrupted later source cannot erase earlier completed construction records."""
    protocol, root = pinned_history
    cases, _ = prepare(protocol, root)

    async def scenario():
        """Cancel the second maintenance call while retaining committed source occurrences."""
        main = ScriptedModelClient([])
        requests = []

        async def respond(request):
            """Complete the first proposal and interrupt the next without provider access."""
            requests.append(request)
            if len(requests) == 2:
                raise asyncio.CancelledError()
            return ModelResponse('{"links": []}')

        outcomes = []
        async with Workspace.open(root / "workspace") as workspace:
            app = LLGM(workspace, main, main, maintenance_model=CallableModelClient(respond))
            with pytest.raises(asyncio.CancelledError):
                await ingest_history(app, cases[0].sources, organize=True, maintenance=outcomes)
            assert len(outcomes) == 2
            assert all(row["status"] == "completed" for row in outcomes)
            assert len(await workspace.source_ids()) == 3
            assert app.last_maintenance.status == "cancelled"
        assert not main.requests

    asyncio.run(scenario())


def _report_fixture(directory):
    """Create a frozen one-question schedule and stale allowance checkpoints."""
    write_json(directory / "protocol.json", {"claim": "synthetic", "arms": ["llgm"]})
    write_json(
        directory / "schedule.json", [{"trial_id": "0000-llgm", "case_id": "one", "arm": "llgm"}]
    )
    write_json(directory / "gold.json", {"one": {"ability": "multi-session"}})
    write_json(
        directory / "allowances.json",
        {
            "generation": {
                "cap_usd": 2,
                "known_estimated_cost_usd": 0,
                "stopped_reason": "local_cost_admission_limit",
            },
            "judging": {"cap_usd": 1, "known_estimated_cost_usd": 0, "stopped_reason": None},
        },
    )
    return directory / "trials" / "0000-llgm"


def test_report_recovers_interrupted_trial_calls_without_double_counting(tmp_path):
    """Durable responses and unfinished reservations survive a stale aggregate and repeated reports."""
    folder = _report_fixture(tmp_path)
    known = {
        "call_id": "known",
        "role": "maintenance",
        "status": "completed",
        "estimated_cost_usd": 0.2,
        "reserved_cost_usd": 0.5,
    }
    pending = {
        "call_id": "pending",
        "role": "root",
        "status": "dispatched",
        "estimated_cost_usd": None,
        "reserved_cost_usd": 0.7,
    }
    write_json(
        folder / "trial.json",
        {**trial("one", "llgm", status="started", calls=[known]), "answer": ""},
    )
    write_json(folder / "calls" / "call-001.json", known)
    write_json(folder / "calls" / "call-002.json", pending)
    result = report(tmp_path)
    assert result["status"] == "incomplete"
    row = result["results"]["llgm"]
    assert row["model_calls"] == 2
    assert row["known_api_cost_usd"] == 0.2
    assert row["unknown_cost_calls"] == 1 and row["total_api_cost_usd"] is None
    assert result["generation"]["known_estimated_cost_usd"] == 0.2
    assert result["generation"]["pending_reserved_cost_usd"] == 0.7
    assert result["generation"]["estimated_cost_usd"] is None
    assert result["generation"]["stopped_reason"] == "local_cost_admission_limit"
    assert result["recorded_allowances"]["generation"]["known_estimated_cost_usd"] == 0
    assert report(tmp_path) == result


@pytest.mark.parametrize("cost", [0.03, None])
def test_report_retains_orphan_judge_cost_without_inventing_verdict(tmp_path, cost):
    """A paid judge response missing its enclosing checkpoint remains charged and unresolved."""
    folder = _report_fixture(tmp_path)
    write_json(
        folder / "trial.json", {**trial("one", "llgm"), "answer": "An answer", "query_seconds": 2.5}
    )
    identity = {"case_id": "one", "arm": "llgm", "phase": "judging"}
    write_json(tmp_path / "judgments" / "0000" / "identity.json", identity)
    write_json(
        tmp_path / "judgments" / "0000" / "call-001.json",
        {
            **identity,
            "call_id": "judge-one",
            "role": "judge",
            "status": "completed" if cost else "dispatched",
            "estimated_cost_usd": cost,
            "reserved_cost_usd": 0.04,
        },
    )
    result = report(tmp_path)
    assert result["judge_known_api_cost_usd"] == (cost or 0)
    assert result["judging"]["known_estimated_cost_usd"] == (cost or 0)
    assert result["judging"]["unknown_cost_calls"] == (cost is None)
    assert result["judging"]["pending_reserved_cost_usd"] == (0.04 if cost is None else 0)
    assert result["results"]["llgm"]["accuracy"] is None
    assert result["results"]["llgm"]["unresolved_judgments"] == 1
    assert result["results"]["llgm"]["query_latency_samples"] == 1
    assert result["unattributed_judge_call_ids"] == []


@pytest.mark.parametrize("effort", [None, "medium"])
def test_failed_query_preserves_phase_latency_and_call_identity(
    pinned_history, monkeypatch, effort
):
    """A failing provider call retains query duration and predispatch identity on disk."""
    from pathlib import Path

    from llgm.evaluation import memory_benchmark

    async def failing_reader(arm, workspace, case, model, config):
        """Exercise recorded dispatch before the finite client raises ProviderError."""
        await model.complete(ModelRequest((Message("user", case.question),)))

    monkeypatch.setattr(memory_benchmark, "answer_baseline", failing_reader)
    pinned, root = pinned_history
    cases, _ = prepare(pinned, root)
    protocol_path = Path(__file__).resolve().parents[1] / "experiments/longmemeval_smoke.json"
    protocol = json.loads(protocol_path.read_text())
    if effort is not None:
        protocol["root_reasoning_effort"] = effort

    async def scenario():
        """Use ordinary ingestion and failure accounting without a provider or Docker."""
        clients = {role: ScriptedModelClient([]) for role in ("root", "sidecar", "maintenance")}
        folder = root / "failed-trial"
        record = await run_trial(cases[0], "bm25", protocol, folder, clients, Allowance(1))
        assert record["status"] == "failed" and record["failed_phase"] == "query"
        assert record["query_seconds"] > 0
        call = json.loads((folder / "calls" / "call-001.json").read_text())
        assert call["phase"] == "query"
        assert call["case_id"] == cases[0].case_id and call["arm"] == "bm25"
        assert call["call_id"] == record["model_calls"][0]["call_id"]
        assert call["estimated_cost_usd"] is None
        expected_temperature = None if effort is not None else 0
        assert call["request"]["temperature"] == expected_temperature
        assert clients["root"].requests[-1].temperature == expected_temperature

    asyncio.run(scenario())
