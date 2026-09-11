"""Summarize frozen node-search trials without treating timing repeats as cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

COHORTS = ("replication", "expanded_longmemeval", "controlled")
LATENCIES = (
    "backend_search_seconds",
    "evidence_search_seconds",
    "search_and_selection_seconds",
)


def _distribution(values: list[float | None]) -> dict:
    """Describe one value per case, preserving unavailable measurements."""
    observed = [value for value in values if value is not None]
    return {
        "count": len(observed),
        "unavailable_count": len(values) - len(observed),
        "mean": statistics.mean(observed) if observed else None,
        "median": statistics.median(observed) if observed else None,
        "min": min(observed) if observed else None,
        "max": max(observed) if observed else None,
    }


def _rate(values: list[bool]) -> dict:
    """Keep a completion numerator and denominator visible alongside its rate."""
    return {
        "numerator": sum(values),
        "denominator": len(values),
        "rate": sum(values) / len(values) if values else None,
    }


def _key(arm: dict) -> tuple[str, int]:
    """Identify one declared backend and retrieval cutoff."""
    return arm["backend"], arm["retrieval_k"]


def _name(key: tuple[str, int]) -> str:
    """Give a stable JSON key to a backend/cutoff pair."""
    return f"{key[0]}@{key[1]}"


def _number(value, label: str, *, maximum: float | None = None) -> None:
    """Reject invalid numerical observations rather than dropping them."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"Invalid {label}: {value!r}")


def _validate_passed(arm: dict) -> None:
    """Require the fields needed for honest coverage, pairing and timing summaries."""
    first = arm["first"]
    metrics = first["metrics"]
    required = metrics["required_node_count"]
    scorable = required is not None and required > 0
    if required is not None and (type(required) is not int or required < 0):
        raise ValueError("Required node count must be a nonnegative integer or null")
    for field in ("candidate_node_recall", "selected_node_recall"):
        if scorable:
            _number(metrics[field], field, maximum=1)
        elif metrics[field] is not None:
            raise ValueError("Unlabeled cases must not carry recall scores")
    for field in ("all_required_candidates", "all_required_selected", "capacity_exceeded"):
        if scorable and type(metrics[field]) is not bool:
            raise ValueError(f"Scorable case requires boolean {field}")
        if not scorable and metrics[field] is not None:
            raise ValueError(f"Unlabeled case requires null {field}")
    if scorable:
        candidate_covered = round(required * metrics["candidate_node_recall"])
        selected_covered = round(required * metrics["selected_node_recall"])
        if (
            not math.isclose(candidate_covered, required * metrics["candidate_node_recall"])
            or not math.isclose(selected_covered, required * metrics["selected_node_recall"])
            or selected_covered > candidate_covered
            or required - candidate_covered != len(metrics["candidate_misses"])
            or candidate_covered - selected_covered != len(metrics["selection_misses"])
            or metrics["all_required_candidates"] != (candidate_covered == required)
            or metrics["all_required_selected"] != (selected_covered == required)
            or metrics["capacity_exceeded"] != (required > arm["max_seed_nodes"])
        ):
            raise ValueError("Coverage scores, missing nodes and capacity disagree")
    selected = first["selected_node_ids"]
    candidates = first["candidate_node_ids"]
    if (
        len(set(selected)) != len(selected)
        or len(selected) > arm["max_seed_nodes"]
        or not set(selected) <= set(candidates)
        or [seed["node_id"] for seed in first["selected_seeds"]] != selected
    ):
        raise ValueError("Selected seed identities are inconsistent")
    for field in ("raw_hits", "hits"):
        for hit in first[field]:
            if not isinstance(hit["passage_id"], str) or not isinstance(hit["references"], list):
                raise ValueError("Missing passage identities or canonical references")
    for field in ("duplicate_concentration", "top_owner_share"):
        if metrics[field] is not None:
            _number(metrics[field], field, maximum=1)
    for field in LATENCIES:
        value = arm["warm_median_seconds"][field]
        if value is not None:
            _number(value, field)
    if arm["generation_calls"] != 0:
        raise ValueError("Retrieval-only results report generation calls")


def _validated_rows(result: dict) -> tuple[dict, dict, list[tuple[str, int]]]:
    """Validate identities and successful trials while retaining failed/missing work."""
    if result["schema_version"] != 1:
        raise ValueError("Unsupported node-search result schema")
    protocol = result["protocol"]
    expected = {_key(arm): arm for arm in protocol["arms"]}
    if len(expected) != len(protocol["arms"]) or set(expected) != {
        (backend, k) for backend in ("bm25", "colbertv2_plaid") for k in (12, 40)
    }:
        raise ValueError("Expected the four distinct frozen backend/cutoff arms")
    if any(
        type(arm["max_seed_nodes"]) is not int or arm["max_seed_nodes"] != 3
        for arm in expected.values()
    ):
        raise ValueError("The frozen experiment requires a three-node seed capacity")
    planned = result["planned_case_ids"]
    if not planned or len(set(planned)) != len(planned):
        raise ValueError("Planned case IDs are empty or duplicated")
    replication = set(protocol["replication_indexes"])
    expanded = {
        case_id for ids in protocol["expanded_selection"]["strata"].values() for case_id in ids
    }
    if replication & expanded or not (replication | expanded) <= set(planned):
        raise ValueError("Declared cohorts disagree with the planned cases")
    cohorts = {
        case_id: "replication"
        if case_id in replication
        else "expanded_longmemeval"
        if case_id in expanded
        else "controlled"
        for case_id in planned
    }
    rows = {}
    for case in result["cases"]:
        case_id = case["case_id"]
        if case_id in rows or case_id not in cohorts or case["cohort"] != cohorts[case_id]:
            raise ValueError("Duplicate, unexpected or misclassified case")
        arms = {}
        for arm in case["arms"]:
            key = _key(arm)
            if key in arms or key not in expected:
                raise ValueError("Duplicate or unexpected trial")
            if arm["max_seed_nodes"] != expected[key]["max_seed_nodes"]:
                raise ValueError("Trial seed capacity differs from its protocol")
            if arm["status"] == "passed":
                _validate_passed(arm)
            arms[key] = arm
        passed = [arm for arm in arms.values() if arm["status"] == "passed"]
        if len({arm["first"]["metrics"]["required_node_count"] for arm in passed}) > 1:
            raise ValueError("Paired arms disagree about the gold denominator")
        rows[case_id] = (case, arms)
    unattempted = [case_id for case_id in planned if case_id not in rows]
    if "unattempted_case_ids" in result and result["unattempted_case_ids"] != unattempted:
        raise ValueError("Recorded unattempted cases disagree with observed trials")
    return cohorts, rows, list(expected)


def _aggregate(case_ids: list[str], rows: dict, key: tuple[str, int]) -> dict:
    """Aggregate passed observations and retain failed or missing trial counts separately."""
    observed = [
        rows[case_id][1][key] for case_id in case_ids if case_id in rows and key in rows[case_id][1]
    ]
    passed = [arm for arm in observed if arm["status"] == "passed"]
    metrics = [arm["first"]["metrics"] for arm in passed]
    scorable = [item for item in metrics if item["candidate_node_recall"] is not None]
    feasible = [item for item in scorable if not item["capacity_exceeded"]]
    required = sum(item["required_node_count"] for item in scorable)
    covered = {
        stage: sum(
            round(item["required_node_count"] * item[f"{stage}_node_recall"]) for item in scorable
        )
        for stage in ("candidate", "selected")
    }
    return {
        "planned_case_count": len(case_ids),
        "attempted_trial_count": len(observed),
        "passed_trial_count": len(passed),
        "failed_trial_count": len(observed) - len(passed),
        "missing_trial_count": len(case_ids) - len(observed),
        "scorable_case_count": len(scorable),
        "no_label_case_count": len(metrics) - len(scorable),
        "capacity_exceeded_case_count": len(scorable) - len(feasible),
        "macro_recall": {
            stage: statistics.mean(item[f"{stage}_node_recall"] for item in scorable)
            if scorable
            else None
            for stage in ("candidate", "selected")
        },
        "cap_feasible_macro_recall": {
            stage: statistics.mean(item[f"{stage}_node_recall"] for item in feasible)
            if feasible
            else None
            for stage in ("candidate", "selected")
        },
        "all_required_completion": {
            stage: _rate([item[f"all_required_{stage}"] for item in scorable])
            for stage in ("candidates", "selected")
        },
        "cap_feasible_completion": {
            stage: _rate([item[f"all_required_{stage}"] for item in feasible])
            for stage in ("candidates", "selected")
        },
        "micro_recall": {
            "required_nodes": required,
            **{
                stage: {"covered_nodes": value, "recall": value / required if required else None}
                for stage, value in covered.items()
            },
        },
        "candidate_missing_nodes": sum(len(item["candidate_misses"]) for item in scorable),
        "candidate_present_unselected_nodes": sum(
            len(item["selection_misses"]) for item in scorable
        ),
        "cases_with_candidate_misses": sum(bool(item["candidate_misses"]) for item in scorable),
        "cases_with_selection_misses": sum(bool(item["selection_misses"]) for item in scorable),
        "unlabeled_selected_nodes": sum(
            len(item["unlabeled_selected_node_ids"]) for item in metrics
        ),
        "concentration": {
            field: _distribution([item[field] for item in metrics])
            for field in (
                "duplicate_concentration",
                "top_owner_share",
                "candidate_node_count",
                "selected_node_count",
            )
        },
        "warm_case_median_latency_seconds": {
            field: _distribution([arm["warm_median_seconds"][field] for arm in passed])
            for field in LATENCIES
        },
        "ranking_unstable_case_count": sum(not arm["rank_consistent"] for arm in passed),
        "selection_unstable_case_count": sum(not arm["selection_consistent"] for arm in passed),
    }


def _references(seed: dict) -> set[str]:
    """Compare canonical references independently of dictionary key ordering."""
    return {json.dumps(ref, sort_keys=True, separators=(",", ":")) for ref in seed["references"]}


def _cutoff_pair(case_id: str, small: dict, large: dict) -> dict:
    """Compare actual independent searches and evidence retained for common seeds."""
    left, right = small["first"], large["first"]
    old = {seed["node_id"]: _references(seed) for seed in left["selected_seeds"]}
    new = {seed["node_id"]: _references(seed) for seed in right["selected_seeds"]}
    common = [node for node in old if node in new]
    return {
        "case_id": case_id,
        "selected_order_equal": left["selected_node_ids"] == right["selected_node_ids"],
        "selected_set_equal": set(old) == set(new),
        **{
            f"{label}_prefix_exact": left[field] == right[field][: len(left[field])]
            for label, field in (("raw_hits", "raw_hits"), ("canonical_hits", "hits"))
        },
        **{
            f"{label}_id_prefix_equal": [hit["passage_id"] for hit in left[field]]
            == [hit["passage_id"] for hit in right[field]][: len(left[field])]
            for label, field in (("raw", "raw_hits"), ("canonical", "hits"))
        },
        "common_seed_reference_changes": [
            {
                "node_id": node,
                "k12_count": len(old[node]),
                "k40_count": len(new[node]),
                "added_count": len(new[node] - old[node]),
                "removed_count": len(old[node] - new[node]),
            }
            for node in common
        ],
        "recall_changes": {
            stage: right["metrics"][f"{stage}_node_recall"]
            - left["metrics"][f"{stage}_node_recall"]
            if left["metrics"][f"{stage}_node_recall"] is not None
            else None
            for stage in ("candidate", "selected")
        },
    }


def _paired(case_ids: list[str], rows: dict) -> dict:
    """Keep complete pairs and list exclusions instead of imputing failed results."""
    cutoff = {}
    for backend in ("bm25", "colbertv2_plaid"):
        pairs, unavailable = [], []
        for case_id in case_ids:
            arms = rows.get(case_id, ({}, {}))[1]
            small, large = arms.get((backend, 12)), arms.get((backend, 40))
            if not small or not large or small["status"] != "passed" or large["status"] != "passed":
                unavailable.append(case_id)
                continue
            pairs.append(_cutoff_pair(case_id, small, large))
        cutoff[backend] = {
            "paired_case_count": len(pairs),
            "unavailable_case_ids": unavailable,
            **{
                field: sum(pair[field] for pair in pairs)
                for field in (
                    "selected_order_equal",
                    "selected_set_equal",
                    "raw_hits_prefix_exact",
                    "canonical_hits_prefix_exact",
                    "raw_id_prefix_equal",
                    "canonical_id_prefix_equal",
                )
            },
            "cases_with_added_common_seed_references": sum(
                any(seed["added_count"] for seed in pair["common_seed_reference_changes"])
                for pair in pairs
            ),
            "cases": pairs,
        }
    backends = {}
    for k in (12, 40):
        outcomes = {"bm25_wins": [], "colbertv2_plaid_wins": [], "ties": []}
        excluded = []
        for case_id in case_ids:
            arms = rows.get(case_id, ({}, {}))[1]
            pair = [arms.get((backend, k)) for backend in ("bm25", "colbertv2_plaid")]
            if any(arm is None or arm["status"] != "passed" for arm in pair):
                excluded.append({"case_id": case_id, "reason": "incomplete_pair"})
                continue
            left, right = [arm["first"]["metrics"] for arm in pair]
            if left["selected_node_recall"] is None:
                excluded.append({"case_id": case_id, "reason": "no_positive_labels"})
            elif left["capacity_exceeded"]:
                excluded.append({"case_id": case_id, "reason": "capacity_exceeded"})
            else:
                difference = right["selected_node_recall"] - left["selected_node_recall"]
                category = (
                    "ties"
                    if abs(difference) < 1e-12
                    else "colbertv2_plaid_wins"
                    if difference > 0
                    else "bm25_wins"
                )
                outcomes[category].append(case_id)
        backends[str(k)] = {
            "metric": "selected_node_recall, scorable capacity-feasible paired cases",
            "paired_case_count": sum(len(ids) for ids in outcomes.values()),
            "counts": {category: len(ids) for category, ids in outcomes.items()},
            "case_ids": outcomes,
            "excluded": excluded,
        }
    return {"cutoffs": cutoff, "backends": backends}


def summarize(result: dict) -> dict:
    """Produce descriptive cohort summaries with explicit incomplete-work accounting."""
    cohorts, rows, keys = _validated_rows(result)
    details, failures = [], []
    for case_id, cohort in cohorts.items():
        if case_id not in rows:
            failures.append({"case_id": case_id, "cohort": cohort, "reason": "unattempted"})
            details.append(
                {"case_id": case_id, "cohort": cohort, "status": "unattempted", "arms": {}}
            )
            continue
        case, arms = rows[case_id]
        complete = len(arms) == len(keys) and all(
            arm["status"] == "passed" for arm in arms.values()
        )
        if case["status"] != "passed" or not complete:
            failures.append(
                {
                    "case_id": case_id,
                    "cohort": cohort,
                    "reason": "failed_or_incomplete_case",
                    "declared_status": case["status"],
                    "error_type": case.get("error_type"),
                    "error": case.get("error"),
                    "failed_arms": [
                        {
                            "arm": _name(key),
                            "status": arms[key]["status"],
                            "error_type": arms[key].get("error_type"),
                            "error": arms[key].get("error"),
                        }
                        for key in keys
                        if key in arms and arms[key]["status"] != "passed"
                    ],
                    "missing_arms": [_name(key) for key in keys if key not in arms],
                }
            )
        passed = [arm for arm in arms.values() if arm["status"] == "passed"]
        metrics = passed[0]["first"]["metrics"] if passed else {}
        details.append(
            {
                "case_id": case_id,
                "cohort": cohort,
                "ability": case.get("ability"),
                "status": case["status"],
                "required_node_count": metrics.get("required_node_count"),
                "capacity_exceeded": metrics.get("capacity_exceeded"),
                "arms": {
                    _name(key): {
                        "metrics": arm["first"]["metrics"],
                        "selected_node_ids": arm["first"]["selected_node_ids"],
                        "candidate_node_ids": arm["first"]["candidate_node_ids"],
                        "warm_median_seconds": arm["warm_median_seconds"],
                    }
                    for key, arm in arms.items()
                    if arm["status"] == "passed"
                },
            }
        )
    return {
        "schema_version": 1,
        "run_id": result["run_id"],
        "source_status": result["status"],
        "status": "passed"
        if result["status"] == "passed" and not failures
        else "failed_or_incomplete",
        "protocol_sha256": result["protocol_sha256"],
        "planned_case_count": len(cohorts),
        "observed_case_count": len(rows),
        "failed_or_incomplete_case_count": len(failures),
        "failures": failures,
        "cohorts": {
            cohort: {
                "planned_case_count": sum(value == cohort for value in cohorts.values()),
                "arms": {
                    _name(key): _aggregate(
                        [case_id for case_id, value in cohorts.items() if value == cohort],
                        rows,
                        key,
                    )
                    for key in keys
                },
                "paired": _paired(
                    [case_id for case_id, value in cohorts.items() if value == cohort], rows
                ),
            }
            for cohort in COHORTS
        },
        "capacity_cases": [
            {key: row[key] for key in ("case_id", "cohort", "required_node_count")}
            for row in details
            if row.get("capacity_exceeded")
        ],
        "cases": details,
        "interpretation": [
            "One first observation per case; warm repetitions contribute timing only.",
            "Coverage denominators exclude failed, missing and unlabeled trials; their counts remain explicit.",
            "Labeled-node coverage does not measure answer-span discovery or answer accuracy.",
            "Unlabeled nodes are not necessarily irrelevant. Extra seed references do not establish useful evidence.",
            "These are descriptive diagnostics, with no independence, generalization or significance claim.",
        ],
        "cost_usd": result.get("cost_usd"),
    }


def main(argv: list[str] | None = None) -> int:
    """Write an auditable summary and return nonzero for failed or incomplete runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.result.resolve() == args.output.resolve():
        parser.error("Output must not overwrite the source result")
    try:
        raw = args.result.read_bytes()
        summary = summarize(json.loads(raw))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Cannot summarize node-search result: {error}")
    summary["source_result_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
