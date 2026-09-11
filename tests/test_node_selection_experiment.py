"""Verify selection-run accounting and failures without invoking paid providers."""

import asyncio
import json
from dataclasses import asdict

import pytest

from llgm.evaluation.longmemeval import GoldRecord
from llgm.evaluation.node_search import node_coverage_metrics
from llgm.models.base import Message, ModelRequest, ModelResponse, ScriptedModelClient, Usage
from tools.node_selection_experiment import (
    POOLS,
    execute,
    prepare,
    restore_request,
    usage_cost,
)

PRICING = {
    "usd_per_million_input_tokens": 0.4,
    "usd_per_million_cached_input_tokens": 0.1,
    "usd_per_million_output_tokens": 1.6,
}


def fixture_run():
    """Supply one bounded case whose three pools have the same source."""
    gold = GoldRecord("q", None, "fact", ("n-a",), ())
    request = ModelRequest((Message("user", "Choose evidence."),), max_output_tokens=512)
    pools = [
        {
            "pool_id": pool_id,
            "prompt_sha256": "fixture",
            "request": asdict(request),
            "reserved_input_tokens": 3000,
            "candidates": [{"node_id": "n-a"}],
            "hits": [
                {
                    "references": [
                        {
                            "type": "source_span",
                            "node_id": "n-a",
                            "turn_id": "t0",
                            "start": 0,
                            "end": 1,
                        }
                    ]
                }
            ],
            "baseline": {
                "selected_node_ids": ["n-a"],
                "metrics": node_coverage_metrics(["n-a"], ["n-a"], gold),
            },
        }
        for pool_id in POOLS
    ]
    plan = {"case_id": "q", "cohort": "controlled"}
    prepared = {
        "planned_cases": [plan],
        "reserved_cost_usd": 1.0,
        "cases": [
            {**plan, "union_metrics": node_coverage_metrics(["n-a"], [], gold), "pools": pools}
        ],
    }
    protocol = {
        "repetitions": 3,
        "model": {"model": "pinned", "max_output_tokens": 512},
        "pricing": PRICING,
        "limits": {"max_failed_calls": 3, "run_timeout_seconds": 60},
    }
    return prepared, {"q": gold}, protocol


def response(text=None, usage=None, **kwargs):
    """Create a measured response for a deterministic transport contract."""
    return ModelResponse(
        text
        if text is not None
        else json.dumps({"selected_node_ids": ["n-a"], "reason": "Relevant"}),
        usage
        if usage is not None
        else Usage(100, 20, {"input_tokens_details": {"cached_tokens": 40}}),
        model="pinned",
        **kwargs,
    )


def test_cached_tokens_are_not_double_charged():
    """Total input includes cached tokens and each category is priced once."""
    assert usage_cost(response().usage, PRICING) == pytest.approx(
        (60 * 0.4 + 40 * 0.1 + 20 * 1.6) / 1e6
    )


@pytest.mark.parametrize(
    "usage", [Usage(), Usage(10, 2), Usage(10, 2, {"input_tokens_details": {"cached_tokens": 11}})]
)
def test_unknown_or_invalid_billing_categories_stay_unknown(usage):
    """Unavailable category counts never appear as free model work."""
    assert usage_cost(usage, PRICING) is None


def test_restore_request_preserves_provider_neutral_fields():
    """Frozen requests retain multibyte input, schema and output settings."""
    request = ModelRequest(
        (Message("user", "한글"),), max_output_tokens=5, output_schema={"type": "object"}
    )
    assert restore_request(asdict(request)) == request


def test_success_records_all_repeats_and_real_usage_fields(tmp_path):
    """Each attempt has a persisted completion, score and cost without best-of filtering."""
    prepared, gold, protocol = fixture_run()
    client = ScriptedModelClient([response() for _ in range(9)])
    result = asyncio.run(execute(prepared, gold, protocol, tmp_path, client))
    assert result["status"] == "completed"
    assert result["attempted_model_calls"] == len(client.requests) == 9
    assert result["unknown_cost_calls"] == 0
    assert result["estimated_cost_usd"] == pytest.approx(9 * usage_cost(response().usage, PRICING))
    assert json.loads((tmp_path / "node-selection.json").read_text()) == result
    for pool in result["cases"][0]["pools"]:
        assert [trial["repetition"] for trial in pool["trials"]] == [0, 1, 2]
        assert all(trial["metrics"]["all_required_selected"] for trial in pool["trials"])


def test_invalid_selection_stops_after_three_and_retains_responses(tmp_path):
    """No implicit retries, fallback nodes or removal of failed attempts occurs."""
    prepared, gold, protocol = fixture_run()
    client = ScriptedModelClient(
        [response('{"selected_node_ids":["unknown"],"reason":"x"}') for _ in range(9)]
    )
    result = asyncio.run(execute(prepared, gold, protocol, tmp_path, client))
    assert result["status"] == "stopped"
    assert result["attempted_model_calls"] == len(client.requests) == 3
    for pool in result["cases"][0]["pools"]:
        trial = pool["trials"][0]
        assert trial["status"] == "failed" and "response" in trial
        assert "selected_node_ids" not in trial and "metrics" not in trial


@pytest.mark.parametrize(
    "usage", [Usage(), Usage(3001, 1, {"input_tokens_details": {"cached_tokens": 0}})]
)
def test_unknown_or_excess_usage_stops_admission(tmp_path, usage):
    """The local reservation is checked against reported provider counts."""
    prepared, gold, protocol = fixture_run()
    client = ScriptedModelClient([response(usage=usage)])
    result = asyncio.run(execute(prepared, gold, protocol, tmp_path, client))
    assert result["status"] == "stopped"
    assert result["attempted_model_calls"] == 1


def test_truncation_is_failure_even_if_text_looks_parseable(tmp_path):
    """Provider completion state is checked before a valid-looking JSON payload."""
    prepared, gold, protocol = fixture_run()
    client = ScriptedModelClient([response(status="incomplete") for _ in range(3)])
    result = asyncio.run(execute(prepared, gold, protocol, tmp_path, client))
    assert result["status"] == "stopped"
    assert all(pool["trials"][0]["status"] == "failed" for pool in result["cases"][0]["pools"])


def test_unsupported_protocol_is_rejected_before_reading_inputs(tmp_path):
    """Changed comparison shape cannot dispatch the declared experiment."""
    with pytest.raises(ValueError, match="Unsupported"):
        prepare({"schema_version": 2}, tmp_path)
