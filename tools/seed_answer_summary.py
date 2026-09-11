"""Summarize paired final answers with planned denominators and separate judge costs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

COHORT_FIELDS = {
    "expanded_longmemeval": "benchmark_case_ids",
    "controlled": "controlled_case_ids",
}
POOLS = ("bm25_40", "colbert_40", "rrf_40")
POLICIES = ("first_owner", "model_selector")
JUDGE_ROLES = {"accuracy_judge", "support_judge"}
VERDICTS = {"supported", "unsupported", "insufficient", "not_applicable"}


def _number(value, label: str, *, integer: bool = False) -> None:
    """Accept unknown measurements but reject negative, boolean, and nonfinite values."""
    if value is not None and (
        type(value) not in ({int} if integer else {int, float})
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid {label}: {value!r}")


def _ids(values, label: str) -> None:
    """Require distinct, nonempty string identifiers before building paired cells."""
    if (
        not isinstance(values, list)
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"Invalid or duplicate {label}")


def _validate(result: dict) -> tuple[dict, dict]:
    """Validate recorded observations without dropping missing or unsuccessful trials."""
    if (
        not isinstance(result, dict)
        or type(result["schema_version"]) is not int
        or result["schema_version"] != 1
        or not isinstance(result["status"], str)
        or not result["status"]
    ):
        raise ValueError("Unsupported result schema or status")
    protocol = result["protocol"]
    for field, expected_values in (("pools", POOLS), ("policies", POLICIES)):
        if field in protocol:
            _ids(protocol[field], field)
            if set(protocol[field]) != set(expected_values):
                raise ValueError("Declared arms differ from the fixed six-arm comparison")
    if protocol.get("answer_repetitions", 1) != 1:
        raise ValueError("Answer summary requires one declared trial per case and arm")
    cohorts, ownership = {}, {}
    for cohort, field in COHORT_FIELDS.items():
        ids = protocol[field]
        _ids(ids, field)
        if set(ids) & ownership.keys():
            raise ValueError("Case IDs must be disjoint across cohorts")
        cohorts[cohort] = ids
        ownership.update({case_id: cohort for case_id in ids})
    expected = len(ownership) * len(POOLS) * len(POLICIES)
    if type(result["planned_trials"]) is not int or result["planned_trials"] != expected:
        raise ValueError("Planned trial count differs from the declared six-arm cohort")
    if not isinstance(result["trials"], list):
        raise ValueError("Trials must be a list")
    rows, identities = {}, set()
    for trial in result["trials"]:
        if not isinstance(trial, dict):
            raise ValueError("Each trial must be an object")
        case_id, pool, policy = trial["case_id"], trial["pool_id"], trial["policy"]
        key = (case_id, pool, policy)
        if (
            case_id not in ownership
            or trial["cohort"] != ownership[case_id]
            or pool not in POOLS
            or policy not in POLICIES
            or key in rows
            or not isinstance(trial["trial_id"], str)
            or not trial["trial_id"]
            or trial["trial_id"] in identities
            or not isinstance(trial["status"], str)
            or not trial["status"]
        ):
            raise ValueError("Duplicate, unexpected, or misclassified trial")
        identities.add(trial["trial_id"])
        _number(trial.get("elapsed_seconds"), "trial elapsed seconds")
        if "seed_node_ids" in trial:
            _ids(trial["seed_node_ids"], "seed node IDs")
        if "references" in trial and (
            not isinstance(trial["references"], list)
            or any(
                not isinstance(ref, dict)
                or not isinstance(ref.get("node_id"), str)
                or not ref["node_id"]
                for ref in trial["references"]
            )
        ):
            raise ValueError("Final citations require source node identities")
        if not isinstance(trial.get("judgments", {}), dict):
            raise ValueError("Judgments must be an object")
        for kind in ("accuracy", "support"):
            judgment = trial.get("judgments", {}).get(kind)
            if judgment is None:
                continue
            if not isinstance(judgment, dict) or judgment.get("status") not in {
                "completed",
                "failed",
            }:
                raise ValueError("Unsupported judgment status")
            if judgment["status"] == "completed":
                if kind == "accuracy" and type(judgment.get("correct")) is not bool:
                    raise ValueError("Completed accuracy judgment requires a boolean verdict")
                if kind == "support" and (
                    not isinstance(judgment.get("verdict"), str)
                    or judgment["verdict"] not in VERDICTS
                    or not isinstance(judgment.get("reason"), str)
                ):
                    raise ValueError("Completed support judgment requires a verdict and reason")
            elif any(field in judgment for field in ("correct", "verdict")):
                raise ValueError("Failed judgments cannot carry scored verdicts")
        history = trial.get("historical_selector")
        if history is not None:
            if not isinstance(history, dict):
                raise ValueError("Historical selector accounting must be an object")
            for field in ("elapsed_seconds", "estimated_cost_usd"):
                _number(history.get(field), f"historical selector {field}")
                if policy == "first_owner" and history.get(field) != 0:
                    raise ValueError(
                        "First-owner control cannot carry historical model-selection work"
                    )
        if not isinstance(trial.get("model_calls"), list):
            raise ValueError("Observed trials require an explicit model-call list")
        for call in trial["model_calls"]:
            if not isinstance(call, dict):
                raise ValueError("Each model-call record must be an object")
            if any(
                not isinstance(call.get(field), str) or not call[field]
                for field in ("role", "status")
            ):
                raise ValueError("Model calls require a role and status")
            for field in ("estimated_cost_usd", "reserved_cost_usd", "elapsed_seconds"):
                _number(call.get(field), f"model-call {field}")
            response = call.get("response")
            if response is not None:
                if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
                    raise ValueError("Returned model responses require provider usage records")
                for field in ("input_tokens", "output_tokens"):
                    _number(response["usage"].get(field), field, integer=True)
        rows[key] = trial
    _number(result.get("elapsed_seconds"), "run elapsed seconds")
    return cohorts, rows


def _total(values: list[float | int | None]) -> dict:
    """Retain a known subtotal while leaving totals with unknown components unset."""
    known = [value for value in values if value is not None]
    return {
        "total": sum(known) if len(known) == len(values) else None,
        "known_subtotal": sum(known),
        "known_count": len(known),
        "unknown_count": len(values) - len(known),
    }


def _distribution(values: list[float | None]) -> dict:
    """Use linear interpolation between observed order statistics for p95 latency."""
    known = sorted(value for value in values if value is not None)
    position = (len(known) - 1) * 0.95
    lower, upper = math.floor(position), math.ceil(position)
    return {
        "observed_count": len(known),
        "unknown_count": len(values) - len(known),
        "mean": statistics.mean(known) if known else None,
        "p50": statistics.median(known) if known else None,
        "p95": (
            known[lower] + (known[upper] - known[lower]) * (position - lower) if known else None
        ),
    }


def _judgment(trial: dict | None, kind: str):
    """Return only completed judgments; absent and failed judges remain unknown."""
    judgment = trial.get("judgments", {}).get(kind) if trial is not None else None
    if not judgment or judgment["status"] != "completed":
        return None
    return judgment["correct" if kind == "accuracy" else "verdict"]


def _discovery(trial: dict | None) -> list[str] | None:
    """Identify final citation owners outside initial seeds without claiming recovery."""
    if trial is None or "seed_node_ids" not in trial or "references" not in trial:
        return None
    return sorted({ref["node_id"] for ref in trial["references"]} - set(trial["seed_node_ids"]))


def _accounting(trials: list[dict | None]) -> dict:
    """Separate current generation/judging from reused selector cost and provider tokens."""
    missing = sum(trial is None for trial in trials)
    calls = [call for trial in trials if trial is not None for call in trial["model_calls"]]
    groups = {
        "generation": [call for call in calls if call["role"] not in JUDGE_ROLES],
        "judging": [call for call in calls if call["role"] in JUDGE_ROLES],
        "all_current_calls": calls,
    }
    costs, tokens = {}, {}
    for group, selected in groups.items():
        costs[group] = {
            **_total([call.get("estimated_cost_usd") for call in selected] + [None] * missing),
            "recorded_call_count": len(selected),
            "missing_trial_count": missing,
        }
        tokens[group] = {
            field: _total(
                [(call.get("response") or {}).get("usage", {}).get(field) for call in selected]
                + [None] * missing
            )
            for field in ("input_tokens", "output_tokens")
        }
    history = [
        (trial.get("historical_selector") or {}).get("estimated_cost_usd") if trial else None
        for trial in trials
    ]
    costs["historical_selector_increment"] = _total(history)
    generation, selector = costs["generation"], costs["historical_selector_increment"]
    costs["inference_plus_historical_selection"] = {
        "total": (
            generation["total"] + selector["total"]
            if generation["total"] is not None and selector["total"] is not None
            else None
        ),
        "known_subtotal": generation["known_subtotal"] + selector["known_subtotal"],
    }
    return {
        "estimated_cost_usd": costs,
        "provider_tokens": tokens,
        "model_call_status_counts": dict(sorted(Counter(call["status"] for call in calls).items())),
    }


def _arm(case_ids: list[str], rows: dict, pool: str, policy: str) -> dict:
    """Describe one cohort arm using every planned case, including absent trials."""
    trials = [rows.get((case_id, pool, policy)) for case_id in case_ids]
    accuracies = [_judgment(trial, "accuracy") for trial in trials]
    support = [_judgment(trial, "support") for trial in trials]
    discoveries = [_discovery(trial) for trial in trials]
    correct, incorrect = accuracies.count(True), accuracies.count(False)
    inference = [trial.get("elapsed_seconds") if trial else None for trial in trials]
    selection = [
        (trial.get("historical_selector") or {}).get("elapsed_seconds") if trial else None
        for trial in trials
    ]
    return {
        "expected_case_count": len(case_ids),
        "observed_trial_count": sum(trial is not None for trial in trials),
        "missing_case_ids": [case_id for case_id, trial in zip(case_ids, trials) if trial is None],
        "runtime_status_counts": dict(
            sorted(Counter(trial["status"] if trial else "missing" for trial in trials).items())
        ),
        "accuracy": {
            "correct_count": correct,
            "incorrect_count": incorrect,
            "unknown_count": accuracies.count(None),
            "planned_denominator": len(case_ids),
            "correct_over_planned": correct / len(case_ids) if case_ids else None,
            "accuracy_among_judged": correct / (correct + incorrect)
            if correct + incorrect
            else None,
        },
        "support": {
            "verdict_counts": {verdict: support.count(verdict) for verdict in sorted(VERDICTS)},
            "unknown_count": support.count(None),
            "factual_supported_correct_count": sum(
                accuracy is True and verdict == "supported"
                for accuracy, verdict in zip(accuracies, support)
            ),
            "correct_no_material_facts_count": sum(
                accuracy is True and verdict == "not_applicable"
                for accuracy, verdict in zip(accuracies, support)
            ),
            "planned_denominator": len(case_ids),
        },
        "final_citation_discovery": {
            "case_count": sum(bool(nodes) for nodes in discoveries),
            "unknown_case_count": discoveries.count(None),
            "nodes_by_case": {
                case_id: nodes for case_id, nodes in zip(case_ids, discoveries) if nodes
            },
        },
        "latency_seconds": {
            "inference": _distribution(inference),
            "historical_selection_increment": _distribution(selection),
            "inference_plus_historical_selection": _distribution(
                [
                    left + right if left is not None and right is not None else None
                    for left, right in zip(inference, selection)
                ]
            ),
        },
        "cases": [
            {"case_id": case_id, "accuracy_correct": accuracy, "support_verdict": verdict}
            for case_id, accuracy, verdict in zip(case_ids, accuracies, support)
        ],
        **_accounting(trials),
    }


def _paired(case_ids: list[str], rows: dict, pool: str) -> dict:
    """Compare selector and control on the same questions without treating unknowns as losses."""
    outcomes = {
        key: []
        for key in (
            "selector_only_wins",
            "control_only_wins",
            "both_correct",
            "both_wrong",
            "unknown",
        )
    }
    for case_id in case_ids:
        control = _judgment(rows.get((case_id, pool, "first_owner")), "accuracy")
        selector = _judgment(rows.get((case_id, pool, "model_selector")), "accuracy")
        if control is None or selector is None:
            outcome = "unknown"
        elif control == selector:
            outcome = "both_correct" if control else "both_wrong"
        else:
            outcome = "selector_only_wins" if selector else "control_only_wins"
        outcomes[outcome].append(case_id)
    return {
        "planned_pair_count": len(case_ids),
        "counts": {key: len(ids) for key, ids in outcomes.items()},
        "case_ids": outcomes,
    }


def summarize(result: dict) -> dict:
    """Aggregate a frozen six-arm answer diagnostic without implying independent samples."""
    cohorts, rows = _validate(result)
    missing = result["planned_trials"] - len(rows)
    unknown_judgments = sum(
        _judgment(trial, kind) is None
        for trial in rows.values()
        for kind in ("accuracy", "support")
    )
    return {
        "schema_version": 1,
        "status": (
            "completed"
            if result["status"] == "completed" and not missing and not unknown_judgments
            else "incomplete"
        ),
        "source_status": result["status"],
        "planned_trial_count": result["planned_trials"],
        "observed_trial_count": len(rows),
        "missing_trial_count": missing,
        "unknown_judgment_count": unknown_judgments + 2 * missing,
        "elapsed_seconds": result.get("elapsed_seconds"),
        "interpretation": {
            "scope": "Exposed diagnostic questions; cohorts are separate and not independent held-out samples.",
            "accuracy": "Official-template model judgment, with an explicitly adapted rubric for controlled cases; unknown judgments are reported separately.",
            "support": "Blind model judgment of final cited quotes, not proof of truth or answer completeness.",
            "discovery": "A final citation outside the initial seeds establishes citation discovery, not recovery of a missed required fact.",
            "cost": "Current recorded provider-call estimates exclude historical selector work; inference comparison adds that historical increment separately. Reserved costs are not measured charges.",
            "latency": "Inference latency excludes prior selection and judging. Added historical selection is a comparison estimate; p95 uses linear interpolation over observed case latencies.",
        },
        "cohorts": {
            cohort: {
                "expected_case_count": len(case_ids),
                "pools": {
                    pool: {
                        "policies": {
                            policy: _arm(case_ids, rows, pool, policy) for policy in POLICIES
                        },
                        "paired_selector_vs_control": _paired(case_ids, rows, pool),
                    }
                    for pool in POOLS
                },
            }
            for cohort, case_ids in cohorts.items()
        },
        **_accounting(list(rows.values()) + [None] * missing),
    }


def main(argv: list[str] | None = None) -> int:
    """Write a JSON summary, distinguishing complete, incomplete, and malformed runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = summarize(json.loads(args.result.read_text(encoding="utf-8")))
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}))
        return 2
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
