"""Qualify a stronger delegate on eight fixed cases after the frozen parent run."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from llgm.core.environment import load_env_file
from llgm.core.types import Conversation
from llgm.evaluation.answer_judging import (
    accuracy_request,
    load_official_prompt,
    parse_accuracy,
    parse_support,
    support_request,
)
from llgm.evaluation.seed_answers import answer_from_seeds
from llgm.inference.budget import Budget
from llgm.inference.repl import DockerREPLConfig
from llgm.memory.evidence import Evidence
from llgm.memory.workspace import Workspace
from llgm.models.hosted import OpenAICompatibleModelClient, OpenAIModelClient
from tools.node_selection_experiment import digest, write_json
from tools.seed_answer_experiment import Allowance, freeze_sources, recorded_model
from tools.seed_answer_experiment import prepare as prepare_parent

ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = (
    "dad224aa",
    "gpt4_731e37d7",
    "cc539528",
    "1da05512",
    "21436231",
    "gpt4_7f6b06db",
    "ns-01",
    "ns-03",
)


def _amount(value, label: str) -> float:
    """Require finite nonnegative measured or reserved dollar estimates."""
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {label}")
    return float(value)


def parent_cost(parent: dict, limits: dict) -> dict:
    """Reconcile known costs and unknown liabilities before admitting new work."""
    if parent.get("status") != "completed":
        raise ValueError("Qualification requires the completed parent run")
    costs, unknown, unknown_counts, pending = {}, {}, {}, {}
    for group in ("generation", "judging"):
        allowance = parent["allowances"][group]
        pending[group] = _amount(allowance.get("pending_reserved_cost_usd"), "pending parent cost")
        if not math.isclose(pending[group], 0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Parent cost must have no pending requests")
        if allowance.get("stopped_reason") == "returned_usage_exceeds_reservation":
            raise ValueError("Parent usage exceeded its reservation")
        unknown[group] = _amount(allowance.get("unknown_reserved_cost_usd"), "unknown reservation")
        count = allowance.get("unknown_cost_calls")
        if type(count) is not int or count < 0:
            raise ValueError("Invalid unknown parent call count")
        unknown_counts[group] = count
        known = _amount(allowance.get("known_estimated_cost_usd"), "parent cost")
        if count:
            if allowance.get("estimated_cost_usd") is not None:
                raise ValueError("Unknown parent cost must retain a null total estimate")
        else:
            estimate = _amount(allowance.get("estimated_cost_usd"), "parent cost estimate")
            if not math.isclose(known, estimate, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Parent known and total cost estimates disagree")
        costs[group] = known
    recorded = {"generation": 0.0, "judging": 0.0}
    recorded_unknown = {"generation": 0.0, "judging": 0.0}
    recorded_counts = {"generation": 0, "judging": 0}
    for trial in parent["trials"]:
        for call in trial["model_calls"]:
            role = call.get("role")
            if role not in {"root", "sidecar", "accuracy_judge", "support_judge"}:
                raise ValueError("Parent contains an unknown model role")
            if not isinstance(call.get("status"), str) or call["status"] in {
                "dispatched",
                "running",
                "attempted",
            }:
                raise ValueError("Parent contains an in-flight model call")
            if call.get("usage_within_reservation") is not True:
                raise ValueError("Parent call does not retain an intact usage bound")
            group = "generation" if role in {"root", "sidecar"} else "judging"
            reservation = _amount(call.get("reserved_cost_usd"), "parent call reservation")
            if reservation <= 0:
                raise ValueError("Parent call requires a positive reservation")
            actual = call.get("estimated_cost_usd")
            if actual is None:
                recorded_unknown[group] += reservation
                recorded_counts[group] += 1
            else:
                actual = _amount(actual, "parent call cost")
                if actual > reservation + 1e-9:
                    raise ValueError("Parent known call cost exceeds its reservation")
                recorded[group] += actual
    if any(
        not math.isclose(recorded[group], costs[group], rel_tol=0, abs_tol=1e-8) for group in costs
    ):
        raise ValueError("Parent allowances disagree with retained model-call estimates")
    if recorded_counts != unknown_counts or any(
        not math.isclose(recorded_unknown[group], unknown[group], rel_tol=0, abs_tol=1e-9)
        for group in unknown
    ):
        raise ValueError("Parent unknown reservations disagree with retained unknown calls")
    total = math.fsum(costs.values())
    unknown_total = math.fsum(unknown.values())
    pending_total = math.fsum(pending.values())
    liability = math.fsum((total, unknown_total, pending_total))
    additional = math.fsum(
        _amount(limits[field], "qualification allowance")
        for field in ("generation_cost_cap_usd", "judge_cost_cap_usd")
    )
    cap = _amount(limits["combined_parent_and_qualification_cap_usd"], "combined cap")
    if liability + additional > cap:
        raise ValueError("Parent liability plus qualification allowance exceeds the combined cap")
    return {
        "known_estimated_cost_usd": total,
        "estimated_cost_usd": None if sum(unknown_counts.values()) else total,
        "unknown_reserved_cost_usd": unknown_total,
        "unknown_cost_calls": sum(unknown_counts.values()),
        "pending_reserved_cost_usd": pending_total,
        "admission_liability_usd": liability,
    }


def prepare(protocol: dict, root: Path = ROOT):
    """Freeze eight exact parent payloads while changing only delegate model and price."""
    if (
        protocol["schema_version"] != 1
        or protocol["case_ids"] != list(CASE_IDS)
        or protocol["pool_id"] != "bm25_40"
        or protocol["policy"] != "first_owner"
        or protocol["answer_repetitions"] != 1
        or protocol["sidecar_model"] != "gpt-4.1-2025-04-14"
        or protocol["parent_run"]["requires_completed"] is not True
        or protocol["limits"]["generation_cost_cap_usd"] != 2
        or protocol["limits"]["judge_cost_cap_usd"] != 0.5
        or protocol["limits"]["combined_parent_and_qualification_cap_usd"] != 20
    ):
        raise ValueError("Unsupported qualification protocol")
    base_spec = protocol["base_protocol"]
    if digest(root / base_spec["path"]) != base_spec["sha256"]:
        raise ValueError("Base answer protocol differs from its frozen hash")
    base = json.loads((root / base_spec["path"]).read_text())
    parent_bytes = (root / protocol["parent_run"]["path"]).read_bytes()
    parent = json.loads(parent_bytes)
    if parent["protocol"] != base:
        raise ValueError("Completed parent used a different protocol")
    parent_accounting = parent_cost(parent, protocol["limits"])
    original, cases, golds = prepare_parent(base, root)
    effective = copy.deepcopy(base)
    if base["models"]["root"] != protocol["sidecar_model"]:
        raise ValueError("The qualification delegate must match the existing full root model")
    effective["models"]["sidecar"] = protocol["sidecar_model"]
    effective["pricing"]["sidecar"] = copy.deepcopy(base["pricing"]["root"])
    by_id = {row["case_id"]: row for row in original["cases"]}
    completed = {trial["trial_id"]: trial for trial in parent["trials"]}
    planned = []
    for case_id in CASE_IDS:
        row = by_id[case_id]
        pool = next(pool for pool in row["pools"] if pool["pool_id"] == "bm25_40")
        trial_id = f"{case_id}__bm25_40__first_owner"
        prior = completed.get(trial_id)
        if (
            prior is None
            or prior["status"] in {"running", "not_started"}
            or prior["seed_node_ids"] != pool["seeds"]["first_owner"]
        ):
            raise ValueError("Qualification parent trial is missing or changed its seed payload")
        planned.append(
            {
                "trial_id": trial_id,
                "case_id": case_id,
                "cohort": row["cohort"],
                "pool_id": "bm25_40",
                "policy": "first_owner",
                "seed_node_ids": copy.deepcopy(pool["seeds"]["first_owner"]),
                "hits": copy.deepcopy(pool["hits"]),
            }
        )
    prepared = {
        "schema_version": 1,
        "planned_trials": len(planned),
        "trials": planned,
        "effective_protocol": effective,
        "parent_run": {
            "path": protocol["parent_run"]["path"],
            "sha256": hashlib.sha256(parent_bytes).hexdigest(),
            **parent_accounting,
        },
    }
    return prepared, {key: cases[key] for key in CASE_IDS}, {key: golds[key] for key in CASE_IDS}


def verify_parent(prepared: dict, protocol: dict, root: Path = ROOT) -> None:
    """Reject parent changes between preparation and the first provider dispatch."""
    spec = prepared["parent_run"]
    payload = (root / spec["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise ValueError("Parent run changed after qualification preparation")
    actual = parent_cost(json.loads(payload), protocol["limits"])
    if any(spec.get(key) != value for key, value in actual.items()):
        raise ValueError("Parent cost changed after preparation")


async def execute(prepared, cases, golds, protocol, directory: Path, clients: dict) -> dict:
    """Run eight actual node answers sequentially with durable failures and judgments."""
    verify_parent(prepared, protocol)
    effective = prepared["effective_protocol"]
    deadline = time.monotonic() + protocol["limits"]["run_timeout_seconds"]
    generation = Allowance(protocol["limits"]["generation_cost_cap_usd"], deadline=deadline)
    judging = Allowance(protocol["limits"]["judge_cost_cap_usd"], deadline=deadline)
    official = effective["inputs"]["official_judge"]
    prompt = load_official_prompt(ROOT / official["path"], official["sha256"])
    result = {
        "schema_version": 1,
        "status": "running",
        "protocol": protocol,
        "effective_protocol": effective,
        "parent_run": prepared["parent_run"],
        "planned_trials": 8,
        "workspace_preparation": [],
        "clients": {role: dict(client.descriptor()) for role, client in clients.items()},
        "trials": [
            {
                **{key: value for key, value in row.items() if key != "hits"},
                "status": "not_started",
                "model_calls": [],
                "judgments": {},
                "historical_selector": {"elapsed_seconds": 0.0, "estimated_cost_usd": 0.0},
            }
            for row in prepared["trials"]
        ],
    }
    started = time.perf_counter()

    def checkpoint():
        """Persist every trial, including absent judgments and stopped admissions."""
        result["allowances"] = {"generation": generation.record(), "judging": judging.record()}
        result["elapsed_seconds"] = time.perf_counter() - started
        result["combined_known_estimated_cost_usd"] = (
            prepared["parent_run"]["known_estimated_cost_usd"]
            + generation.known_cost
            + judging.known_cost
        )
        result["combined_unknown_reserved_cost_usd"] = (
            prepared["parent_run"]["unknown_reserved_cost_usd"]
            + generation.unresolved
            + judging.unresolved
        )
        result["combined_pending_reserved_cost_usd"] = (
            prepared["parent_run"]["pending_reserved_cost_usd"]
            + generation.pending
            + judging.pending
        )
        result["combined_estimated_cost_usd"] = (
            None
            if prepared["parent_run"]["unknown_cost_calls"]
            or generation.unknown_calls
            or judging.unknown_calls
            or generation.pending > 1e-9
            or judging.pending > 1e-9
            else result["combined_known_estimated_cost_usd"]
        )
        result["combined_admission_liability_usd"] = math.fsum(
            result[field]
            for field in (
                "combined_known_estimated_cost_usd",
                "combined_unknown_reserved_cost_usd",
                "combined_pending_reserved_cost_usd",
            )
        )
        result["completed_judgments"] = sum(
            judgment.get("status") == "completed"
            for trial in result["trials"]
            for judgment in trial["judgments"].values()
        )
        write_json(directory / "seed-answer-qualification.json", result)

    checkpoint()
    for row, trial in zip(prepared["trials"], result["trials"]):
        trial_path = directory / "trials" / row["trial_id"]
        trial_path.mkdir(parents=True)
        case = cases[row["case_id"]]
        if generation.stopped_reason or time.monotonic() >= deadline:
            trial["error_type"] = generation.stopped_reason or "run_admission_deadline"
            write_json(trial_path / "result.json", trial)
            checkpoint()
            continue
        trial["status"] = "running"
        checkpoint()
        try:
            setup_started = time.perf_counter()
            async with Workspace.open(directory / "workspaces" / case.case_id) as workspace:
                for source in case.sources:
                    await workspace.ingest(
                        Conversation(
                            source.turns, source.node_id, source.metadata, source.timestamp_ms
                        )
                    )
                evidence = await Evidence.open(
                    workspace, passage_chars=effective["runtime"]["passage_chars"]
                )
                await evidence.close()
                result["workspace_preparation"].append(
                    {
                        "case_id": case.case_id,
                        "elapsed_seconds": time.perf_counter() - setup_started,
                    }
                )
                models = {
                    role: recorded_model(
                        clients[role],
                        role,
                        effective["pricing"][role],
                        generation,
                        trial,
                        trial_path,
                    )
                    for role in ("root", "sidecar")
                }
                inference_started = time.perf_counter()
                try:
                    trial.update(
                        await answer_from_seeds(
                            workspace,
                            case,
                            row["seed_node_ids"],
                            row["hits"],
                            models["root"],
                            models["sidecar"],
                            budget=Budget(**effective["budget"]),
                            runtime_options=effective["runtime"],
                            repl_config=DockerREPLConfig(**effective["docker"]),
                        )
                    )
                finally:
                    trial["elapsed_seconds"] = time.perf_counter() - inference_started
        except Exception as exc:
            trial.update(status="failed", error_type=type(exc).__name__)
        checkpoint()
        if isinstance(trial.get("answer"), str):
            for kind in ("accuracy", "support"):
                judgment = {"status": "running"}
                trial["judgments"][kind] = judgment
                role = f"{kind}_judge"
                judge = recorded_model(
                    clients[role], role, effective["pricing"]["judge"], judging, trial, trial_path
                )
                try:
                    request = (
                        accuracy_request(case, golds[case.case_id], trial["answer"], prompt)
                        if kind == "accuracy"
                        else support_request(
                            case.question,
                            case.question_date,
                            trial["answer"],
                            trial["cited_evidence"],
                        )
                    )
                    response = (await judge.complete(request)).ensure_complete()
                    parsed = (
                        {"correct": parse_accuracy(response.text)}
                        if kind == "accuracy"
                        else parse_support(response.text)
                    )
                    judgment.update(status="completed", **parsed)
                except Exception as exc:
                    judgment.update(status="failed", error_type=type(exc).__name__)
                checkpoint()
        write_json(trial_path / "result.json", trial)
        checkpoint()
        print(json.dumps({"trial_id": trial["trial_id"], "status": trial["status"]}), flush=True)
    result["status"] = (
        "completed"
        if all(trial["status"] not in {"running", "not_started"} for trial in result["trials"])
        else "stopped"
    )
    checkpoint()
    return result


def main():
    """Prepare without credentials or explicitly execute a new qualification directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "experiments/seed_answer_qualification_v1.json"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    prepared, cases, golds = prepare(protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "prepared.json", prepared)
    write_json(args.output / "scoring.json", {key: asdict(value) for key, value in golds.items()})
    freeze_sources(args.output, protocol, args.protocol.resolve())
    if not args.execute:
        print(json.dumps({"status": "prepared", "planned_trials": 8, "model_calls": 0}))
        return
    if args.env_file:
        load_env_file(args.env_file)

    async def live():
        """Own real retry-free transports without changing the existing runtime clients."""
        effective = prepared["effective_protocol"]
        options = {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "timeout_seconds": effective["limits"]["timeout_seconds_per_call"],
        }
        clients = {
            "root": OpenAIModelClient(effective["models"]["root"], **options),
            "sidecar": OpenAIModelClient(effective["models"]["sidecar"], **options),
            "accuracy_judge": OpenAICompatibleModelClient(effective["models"]["judge"], **options),
            "support_judge": OpenAIModelClient(effective["models"]["judge"], **options),
        }
        try:
            return await execute(prepared, cases, golds, protocol, args.output, clients)
        finally:
            for client in clients.values():
                await client.aclose()

    result = asyncio.run(live())
    print(json.dumps({"status": result["status"], "allowances": result["allowances"]}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
