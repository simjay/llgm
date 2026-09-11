"""Summarize repeated node selectors with explicit failures and cohort denominators."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

COHORTS = ("expanded_longmemeval", "controlled")
POOLS = ("bm25_40", "colbert_40", "rrf_40")


def _number(value, label: str, *, maximum: float | None = None) -> None:
    """Reject nonfinite, negative and boolean measurements."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"Invalid {label}: {value!r}")


def _ids(values, label: str) -> None:
    """Require a list of distinct, nonempty string identities."""
    if (
        not isinstance(values, list)
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"Invalid or duplicate {label}")


def _rate(values: list[bool]) -> dict:
    """Keep counts visible when reporting a completion rate."""
    count = sum(values)
    return {
        "numerator": count,
        "denominator": len(values),
        "rate": count / len(values) if values else None,
    }


def _distribution(values: list[float | None]) -> dict:
    """Describe recorded calls without silently removing unknown values."""
    known = [value for value in values if value is not None]
    return {
        "observed_count": len(known),
        "unknown_count": len(values) - len(known),
        "mean": statistics.mean(known) if known else None,
        "median": statistics.median(known) if known else None,
        "min": min(known) if known else None,
        "max": max(known) if known else None,
    }


def _metrics(metrics: dict, maximum: int, selected: list[str] | None = None) -> None:
    """Check coverage arithmetic and seed limits, including unscorable cases."""
    required = metrics["required_node_count"]
    if required is not None and (type(required) is not int or required < 0):
        raise ValueError("Required node count must be a nonnegative integer or null")
    for key in ("candidate_node_count", "selected_node_count"):
        if type(metrics[key]) is not int or metrics[key] < 0:
            raise ValueError(f"Invalid {key}")
    if metrics["selected_node_count"] > min(maximum, metrics["candidate_node_count"]):
        raise ValueError("Selected node count exceeds candidate count or capacity")
    if selected is not None:
        _ids(selected, "selected node IDs")
        if len(selected) != metrics["selected_node_count"]:
            raise ValueError("Selected IDs disagree with metrics")
    if not required:
        if any(
            metrics[field] is not None
            for field in (
                "candidate_node_recall",
                "selected_node_recall",
                "all_required_candidates",
                "all_required_selected",
                "capacity_exceeded",
                "candidate_misses",
                "selection_misses",
            )
        ):
            raise ValueError("Cases without positive labels must have null coverage")
        return
    for key in ("candidate_node_recall", "selected_node_recall"):
        _number(metrics[key], key, maximum=1)
    for key in ("all_required_candidates", "all_required_selected", "capacity_exceeded"):
        if type(metrics[key]) is not bool:
            raise ValueError(f"Invalid {key}")
    for key in ("candidate_misses", "selection_misses"):
        _ids(metrics[key], key)
    candidate = required * metrics["candidate_node_recall"]
    chosen = required * metrics["selected_node_recall"]
    if (
        not math.isclose(candidate, round(candidate))
        or not math.isclose(chosen, round(chosen))
        or chosen > candidate
        or round(candidate) > metrics["candidate_node_count"]
        or round(chosen) > metrics["selected_node_count"]
        or required - round(candidate) != len(metrics["candidate_misses"])
        or round(candidate - chosen) != len(metrics["selection_misses"])
        or metrics["all_required_candidates"] != (round(candidate) == required)
        or metrics["all_required_selected"] != (round(chosen) == required)
        or metrics["capacity_exceeded"] != (required > maximum)
    ):
        raise ValueError("Coverage scores, missing labels and capacity disagree")


def _validate(result: dict) -> tuple[dict, dict, int]:
    """Validate run identities and observations while accepting explicitly partial runs."""
    if result["schema_version"] != 1 or result["status"] not in {"completed", "running", "stopped"}:
        raise ValueError("Unsupported result schema or status")
    protocol = result["protocol"]
    repeats, maximum = protocol["repetitions"], protocol["max_seed_nodes"]
    if (
        type(repeats) is not int
        or not 1 <= repeats <= 100
        or type(maximum) is not int
        or maximum < 1
    ):
        raise ValueError("Invalid repetition count or seed capacity")
    planned = result["planned_case_ids"]
    _ids(planned, "planned case IDs")
    if not planned:
        raise ValueError("No planned cases")
    rows = {}
    for case in result["cases"]:
        case_id = case["case_id"]
        if case_id not in planned or case_id in rows or case["cohort"] not in COHORTS:
            raise ValueError("Duplicate, unexpected or misclassified case")
        _metrics(case["union_metrics"], maximum)
        required = case["union_metrics"]["required_node_count"]
        pools = {}
        for pool in case["pools"]:
            key = pool["pool_id"]
            if key not in POOLS or key in pools:
                raise ValueError("Duplicate or unexpected pool")
            baseline = pool["baseline"]
            _metrics(baseline["metrics"], maximum, baseline["selected_node_ids"])
            if baseline["metrics"]["required_node_count"] != required:
                raise ValueError("Pool gold denominator differs from its case")
            trials = {}
            for trial in pool["trials"]:
                repetition = trial["repetition"]
                if (
                    type(repetition) is not int
                    or not 0 <= repetition < repeats
                    or repetition in trials
                ):
                    raise ValueError("Duplicate or unexpected repetition")
                if trial["status"] not in {"completed", "failed"}:
                    raise ValueError("Invalid trial status")
                _number(trial["elapsed_seconds"], "elapsed seconds")
                if trial["estimated_cost_usd"] is not None:
                    _number(trial["estimated_cost_usd"], "estimated cost")
                if trial["status"] == "completed":
                    _metrics(trial["metrics"], maximum, trial["selected_node_ids"])
                    for field in (
                        "required_node_count",
                        "candidate_node_count",
                        "candidate_node_recall",
                        "candidate_misses",
                    ):
                        if trial["metrics"][field] != baseline["metrics"][field]:
                            raise ValueError(
                                "Selector changed its candidate pool or gold denominator"
                            )
                elif "metrics" in trial:
                    raise ValueError("Failed trials must not carry scored selections")
                trials[repetition] = trial
            pools[key] = (pool, trials)
        rows[case_id] = (case, pools)
    declared = result.get("planned_cases")
    if declared is not None:
        if [case["case_id"] for case in declared] != planned:
            raise ValueError("Planned cohort records disagree with planned case IDs")
        cohorts = {case["case_id"]: case["cohort"] for case in declared}
        if any(cohort not in COHORTS for cohort in cohorts.values()):
            raise ValueError("Invalid planned cohort")
        if any(case["cohort"] != cohorts[case_id] for case_id, (case, _) in rows.items()):
            raise ValueError("Observed cohort differs from planned cohort")
    else:
        cohorts = {case_id: case["cohort"] for case_id, (case, _) in rows.items()}
    observed = sum(len(trials) for _, pools in rows.values() for _, trials in pools.values())
    if (
        type(result["attempted_model_calls"]) is not int
        or result["attempted_model_calls"] != observed
    ):
        raise ValueError("Model call total differs from recorded trial attempts")
    if result["estimated_cost_usd"] is not None:
        _number(result["estimated_cost_usd"], "run estimated cost")
    return cohorts, rows, repeats


def _comparison(pairs: list[tuple[str, dict, dict]], expected: int) -> dict:
    """Compare matched completed selections without scoring unavailable calls as evidence."""
    details = []
    counts = {
        metric: {"gains": 0, "losses": 0, "ties": 0}
        for metric in ("complete_coverage", "fractional_recall")
    }
    for case_id, left, right in pairs:
        row = {"case_id": case_id}
        for name, field in (
            ("complete_coverage", "all_required_selected"),
            ("fractional_recall", "selected_node_recall"),
        ):
            delta = left[field] - right[field]
            outcome = "gains" if delta > 0 else "losses" if delta < 0 else "ties"
            counts[name][outcome] += 1
            row[name] = outcome
        details.append(row)
    return {
        "expected_positive_feasible_pairs": expected,
        "completed_pair_count": len(pairs),
        "unavailable_pair_count": expected - len(pairs),
        "counts": counts,
        "cases": details,
    }


def _pool_summary(case_ids: list[str], rows: dict, key: str, repeats: int) -> dict:
    """Summarize one candidate pool with fixed per-repetition case denominators."""
    cases = {case_id: rows[case_id] for case_id in case_ids if case_id in rows}
    positive = [
        case_id
        for case_id, (case, _) in cases.items()
        if case["union_metrics"]["required_node_count"]
    ]
    feasible = [
        case_id
        for case_id in positive
        if not cases[case_id][0]["union_metrics"]["capacity_exceeded"]
    ]
    pools = {case_id: pools[key] for case_id, (_, pools) in cases.items() if key in pools}
    baseline = {case_id: pool["baseline"]["metrics"] for case_id, (pool, _) in pools.items()}
    trials = [trial for _, values in pools.values() for trial in values.values()]
    result = {
        "expected_case_count": len(case_ids),
        "observed_baseline_count": len(pools),
        "missing_baseline_count": len(case_ids) - len(pools),
        "unknown_gold_case_count": len(case_ids) - len(cases),
        "positive_case_count": len(positive),
        "positive_feasible_case_count": len(feasible),
        "capacity_case_ids": [case_id for case_id in positive if case_id not in feasible],
        "no_positive_label_case_ids": [case_id for case_id in cases if case_id not in positive],
        "expected_trial_count": len(case_ids) * repeats,
        "observed_trial_count": len(trials),
        "completed_trial_count": sum(trial["status"] == "completed" for trial in trials),
        "failed_trial_count": sum(trial["status"] == "failed" for trial in trials),
        "missing_trial_count": len(case_ids) * repeats - len(trials),
        "baseline": {
            "positive_feasible_candidate_complete": _rate(
                [
                    baseline[case_id]["all_required_candidates"] if case_id in baseline else False
                    for case_id in feasible
                ]
            ),
            "positive_feasible_selected_complete": _rate(
                [
                    baseline[case_id]["all_required_selected"] if case_id in baseline else False
                    for case_id in feasible
                ]
            ),
            "positive_macro_recall_observed": {
                stage: _distribution(
                    [
                        baseline[case_id][f"{stage}_node_recall"] if case_id in baseline else None
                        for case_id in positive
                    ]
                )
                for stage in ("candidate", "selected")
            },
        },
        "latency_seconds_per_attempt": _distribution(
            [trial["elapsed_seconds"] for trial in trials]
        ),
        "estimated_cost_usd_per_attempt": _distribution(
            [trial["estimated_cost_usd"] for trial in trials]
        ),
        "repetitions": [],
    }
    for repetition in range(repeats):
        attempted = {
            case_id: values[repetition]
            for case_id, (_, values) in pools.items()
            if repetition in values
        }
        completed = {
            case_id: trial["metrics"]
            for case_id, trial in attempted.items()
            if trial["status"] == "completed"
        }
        result["repetitions"].append(
            {
                "repetition": repetition,
                "expected_trial_count": len(case_ids),
                "observed_trial_count": len(attempted),
                "completed_trial_count": len(completed),
                "failed_trial_count": len(attempted) - len(completed),
                "missing_trial_count": len(case_ids) - len(attempted),
                "positive_feasible_selected_complete": _rate(
                    [
                        completed[case_id]["all_required_selected"]
                        if case_id in completed
                        else False
                        for case_id in feasible
                    ]
                ),
                "positive_macro_recall_completed": _distribution(
                    [
                        completed[case_id]["selected_node_recall"] if case_id in completed else None
                        for case_id in positive
                    ]
                ),
                "positive_macro_recall_zero_for_unavailable": statistics.mean(
                    [
                        completed[case_id]["selected_node_recall"] if case_id in completed else 0
                        for case_id in positive
                    ]
                )
                if positive
                else None,
                "paired_vs_baseline": _comparison(
                    [
                        (case_id, completed[case_id], baseline[case_id])
                        for case_id in feasible
                        if case_id in completed and case_id in baseline
                    ],
                    len(feasible),
                ),
            }
        )
    complete_repeats, stable, all_coverage, any_coverage = [], [], [], []
    for case_id in case_ids:
        values = pools[case_id][1] if case_id in pools else {}
        successes = [trial for trial in values.values() if trial["status"] == "completed"]
        if len(successes) == repeats:
            complete_repeats.append(case_id)
            if len({frozenset(trial["selected_node_ids"]) for trial in successes}) == 1:
                stable.append(case_id)
        if case_id in feasible:
            count = sum(trial["metrics"]["all_required_selected"] for trial in successes)
            if count:
                any_coverage.append(case_id)
            if count == repeats:
                all_coverage.append(case_id)
    result["repeat_consistency"] = {
        "all_repetitions_completed_case_count": len(complete_repeats),
        "all_repetitions_completed_case_ids": complete_repeats,
        "selected_set_stable_case_count": len(stable),
        "selected_set_stable_case_ids": stable,
        "stable_among_completed": {
            "numerator": len(stable),
            "denominator": len(complete_repeats),
            "rate": len(stable) / len(complete_repeats) if complete_repeats else None,
        },
        "positive_feasible_case_count": len(feasible),
        "complete_coverage_all_repetitions_case_count": len(all_coverage),
        "complete_coverage_all_repetitions_case_ids": all_coverage,
        "complete_coverage_any_repetition_case_count": len(any_coverage),
        "complete_coverage_any_repetition_case_ids": any_coverage,
    }
    return result


def summarize(result: dict) -> dict:
    """Return descriptive selector results without pooling cohorts or inventing successes."""
    cohorts, rows, repeats = _validate(result)
    planned = result["planned_case_ids"]
    trials = [
        trial
        for _, pools in rows.values()
        for _, values in pools.values()
        for trial in values.values()
    ]
    costs = [trial["estimated_cost_usd"] for trial in trials]
    known_cost = sum(cost for cost in costs if cost is not None)
    failures = sum(trial["status"] == "failed" for trial in trials)
    expected = len(planned) * len(POOLS) * repeats
    summary = {
        "schema_version": 1,
        "run_id": result.get("run_id"),
        "status": "completed"
        if result["status"] == "completed" and len(trials) == expected and not failures
        else "incomplete",
        "source_status": result["status"],
        "scope": "Initial selected-source coverage; not answer accuracy or statistical significance.",
        "expected_case_count": len(planned),
        "observed_case_count": len(rows),
        "missing_case_ids": [case_id for case_id in planned if case_id not in rows],
        "unclassified_missing_case_ids": [case_id for case_id in planned if case_id not in cohorts],
        "expected_trial_count": expected,
        "observed_trial_count": len(trials),
        "completed_trial_count": len(trials) - failures,
        "failed_trial_count": failures,
        "missing_trial_count": expected - len(trials),
        "cost": {
            "priced_attempt_count": sum(cost is not None for cost in costs),
            "unknown_cost_attempt_count": sum(cost is None for cost in costs),
            "known_estimated_cost_usd_subtotal": known_cost,
            "estimated_cost_usd_total": known_cost
            if all(cost is not None for cost in costs)
            else None,
            "recorded_run_estimated_cost_usd": result["estimated_cost_usd"],
        },
        "cohorts": {},
    }
    for cohort in COHORTS:
        case_ids = [case_id for case_id in planned if cohorts.get(case_id) == cohort]
        feasible = [
            case_id
            for case_id in case_ids
            if case_id in rows
            and rows[case_id][0]["union_metrics"]["required_node_count"]
            and not rows[case_id][0]["union_metrics"]["capacity_exceeded"]
        ]
        union = [rows[case_id][0]["union_metrics"] for case_id in feasible]
        data = {
            "expected_case_count": len(case_ids),
            "observed_case_count": sum(case_id in rows for case_id in case_ids),
            "union_positive_feasible_candidate_complete": _rate(
                [metrics["all_required_candidates"] for metrics in union]
            ),
            "pools": {key: _pool_summary(case_ids, rows, key, repeats) for key in POOLS},
            "paired_model_hybrid_vs_single": {},
        }
        for single in POOLS[:2]:
            comparisons = []
            for repetition in range(repeats):
                pairs = []
                for case_id in feasible:
                    pools = rows[case_id][1]
                    if any(
                        key not in pools
                        or repetition not in pools[key][1]
                        or pools[key][1][repetition]["status"] != "completed"
                        for key in ("rrf_40", single)
                    ):
                        continue
                    pairs.append(
                        (
                            case_id,
                            pools["rrf_40"][1][repetition]["metrics"],
                            pools[single][1][repetition]["metrics"],
                        )
                    )
                comparisons.append({"repetition": repetition, **_comparison(pairs, len(feasible))})
            data["paired_model_hybrid_vs_single"][single] = comparisons
        summary["cohorts"][cohort] = data
    return summary


def main(argv: list[str] | None = None) -> int:
    """Write a JSON report; incomplete trials and malformed inputs have distinct exits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = summarize(json.loads(args.result.read_text()))
        rendered = json.dumps(summary, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        else:
            print(rendered, end="")
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}))
        return 2
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
