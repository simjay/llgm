"""Run matched real LLGM answers from previously frozen node selections."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import platform
import shutil
import sys
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
from llgm.evaluation.costs import Allowance as Allowance
from llgm.evaluation.costs import recorded_model as recorded_model
from llgm.evaluation.costs import request_reservation as request_reservation
from llgm.evaluation.costs import usage_cost as usage_cost
from llgm.evaluation.longmemeval import parse_longmemeval
from llgm.evaluation.node_search import anonymize_case
from llgm.evaluation.node_selection import hydrate_candidates, parse_selection
from llgm.evaluation.seed_answers import _frozen_seeds, answer_from_seeds
from llgm.inference.budget import Budget
from llgm.inference.repl import DockerREPLConfig
from llgm.memory.evidence import Evidence
from llgm.memory.workspace import Workspace
from llgm.models.hosted import OpenAICompatibleModelClient, OpenAIModelClient
from tools.node_selection_experiment import (
    digest,
    write_json,
)
from tools.node_selection_experiment import (
    prepare as prepare_selection,
)

ROOT = Path(__file__).resolve().parents[1]
POOLS = ("bm25_40", "colbert_40", "rrf_40")
POLICIES = ("first_owner", "model_selector")


def prepare(protocol: dict, root: Path = ROOT):
    """Verify historical inputs and separate all gold from executable case state."""
    from tools.node_search_cases import controlled_cases

    if (
        protocol["schema_version"] != 1
        or protocol["pools"] != list(POOLS)
        or protocol["policies"] != list(POLICIES)
        or protocol["selector_repetition"] != 0
        or protocol["answer_repetitions"] != 1
        or protocol["temperature"] != 0
        or protocol["limits"]["concurrent_cases"] != 2
        or protocol["limits"]["sdk_retries"] != 0
    ):
        raise ValueError("Unsupported seed-answer protocol")
    for spec in protocol["inputs"].values():
        if digest(root / spec["path"]) != spec["sha256"]:
            raise ValueError(f"Frozen input mismatch: {spec['path']}")
    selection_protocol = json.loads(
        (root / protocol["inputs"]["selection_protocol"]["path"]).read_text()
    )
    fresh, golds = prepare_selection(selection_protocol, root)
    saved = json.loads((root / protocol["inputs"]["selection_prepared"]["path"]).read_text())
    if saved != json.loads(json.dumps(fresh)):
        raise ValueError("Reconstructed selection inputs differ from the frozen run")
    selected = json.loads((root / protocol["inputs"]["selection_results"]["path"]).read_text())
    if selected["status"] != "completed" or selected["protocol"] != selection_protocol:
        raise ValueError("Selection run is incomplete or has a different protocol")
    for field in ("benchmark_case_ids", "controlled_case_ids"):
        if protocol[field] != selection_protocol[field]:
            raise ValueError("Answer cohort differs from the frozen selection cohort")
    raw = json.loads((root / selection_protocol["inputs"]["dataset"]["path"]).read_text())
    cases, labels = parse_longmemeval(
        [row for row in raw if row["question_id"] in protocol["benchmark_case_ids"]]
    )
    cases_by_id = {case.case_id: anonymize_case(case, labels[case.case_id])[0] for case in cases}
    cases_by_id.update({case.case_id: case for case, _ in controlled_cases()})
    selections = {row["case_id"]: row for row in selected["cases"]}
    prepared = {"schema_version": 1, "planned_trials": 192, "cases": []}
    for row in fresh["cases"]:
        case = cases_by_id[row["case_id"]]
        pools = {pool["pool_id"]: pool for pool in selections[case.case_id]["pools"]}
        output = {"case_id": case.case_id, "cohort": row["cohort"], "pools": []}
        for pool in row["pools"]:
            saved_pool = pools[pool["pool_id"]]
            prior = [trial for trial in saved_pool["trials"] if trial["repetition"] == 0]
            if len(prior) != 1 or prior[0]["status"] != "completed":
                raise ValueError("Missing successful frozen selector repetition zero")
            prior = prior[0]
            candidates = hydrate_candidates(case, pool["hits"])
            parsed = parse_selection(prior["response"]["text"], candidates, max_seed_nodes=3)[
                "selected_node_ids"
            ]
            if parsed != prior["selected_node_ids"] or saved_pool["baseline"] != pool["baseline"]:
                raise ValueError("Saved seed choices disagree with their original output")
            seeds = {
                "first_owner": pool["baseline"]["selected_node_ids"],
                "model_selector": parsed,
            }
            for ids in seeds.values():
                _frozen_seeds(case, ids, pool["hits"])
            output["pools"].append(
                {
                    "pool_id": pool["pool_id"],
                    "hits": pool["hits"],
                    "seeds": seeds,
                    "historical_selector": {
                        "elapsed_seconds": prior["elapsed_seconds"],
                        "estimated_cost_usd": prior["estimated_cost_usd"],
                    },
                }
            )
        prepared["cases"].append(output)
    if len(prepared["cases"]) != 32:
        raise ValueError("Expected 32 fixed cases")
    return prepared, cases_by_id, golds


def freeze_sources(directory: Path, protocol: dict, protocol_path: Path) -> None:
    """Snapshot executed Python and configuration before any paid inference."""
    paths = sorted((ROOT / "src/llgm").rglob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
    paths += [ROOT / "pyproject.toml", protocol_path]
    manifest = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = directory / "executed-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[str(relative)] = digest(path)
    write_json(directory / "protocol.json", protocol)
    write_json(
        directory / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("llgm", "openai")},
            "source_sha256": manifest,
            "protocol_sha256": digest(protocol_path),
        },
    )


async def execute(prepared, cases, golds, protocol, directory: Path, clients: dict) -> dict:
    """Run every declared attempt once, retaining errors and missing judgments."""
    deadline = time.monotonic() + protocol["limits"]["run_timeout_seconds"]
    generation = Allowance(protocol["limits"]["generation_cost_cap_usd"], deadline=deadline)
    judging = Allowance(protocol["limits"]["judge_cost_cap_usd"], deadline=deadline)
    official = protocol["inputs"]["official_judge"]
    prompt = load_official_prompt(ROOT / official["path"], official["sha256"])
    result = {
        "schema_version": 1,
        "status": "running",
        "protocol": protocol,
        "planned_trials": prepared["planned_trials"],
        "trials": [],
        "workspace_preparation": [],
        "clients": {name: dict(client.descriptor()) for name, client in clients.items()},
    }
    started = time.perf_counter()
    gate = asyncio.Semaphore(protocol["limits"]["concurrent_cases"])

    def checkpoint():
        """Atomically retain current results without model text in console output."""
        result["allowances"] = {"generation": generation.record(), "judging": judging.record()}
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(directory / "seed-answers.json", result)

    async def run_case(index, row):
        """Prepare one shared immutable history, then rotate its six arm attempts."""
        async with gate:
            case = cases[row["case_id"]]
            setup_started = time.perf_counter()
            async with Workspace.open(directory / "workspaces" / case.case_id) as workspace:
                for source in case.sources:
                    await workspace.ingest(
                        Conversation(
                            source.turns, source.node_id, source.metadata, source.timestamp_ms
                        )
                    )
                evidence = await Evidence.open(
                    workspace, passage_chars=protocol["runtime"]["passage_chars"]
                )
                await evidence.close()
                result["workspace_preparation"].append(
                    {
                        "case_id": case.case_id,
                        "elapsed_seconds": time.perf_counter() - setup_started,
                    }
                )
                arms = [(pool, policy) for pool in row["pools"] for policy in POLICIES]
                offset = index % len(arms)
                for pool, policy in arms[offset:] + arms[:offset]:
                    trial_id = f"{case.case_id}__{pool['pool_id']}__{policy}"
                    trial_path = directory / "trials" / trial_id
                    trial_path.mkdir(parents=True)
                    trial = {
                        "trial_id": trial_id,
                        "case_id": case.case_id,
                        "cohort": row["cohort"],
                        "pool_id": pool["pool_id"],
                        "policy": policy,
                        "status": "not_started",
                        "seed_node_ids": pool["seeds"][policy],
                        "historical_selector": pool["historical_selector"]
                        if policy == "model_selector"
                        else {"elapsed_seconds": 0.0, "estimated_cost_usd": 0.0},
                        "model_calls": [],
                        "judgments": {},
                    }
                    result["trials"].append(trial)
                    if generation.stopped_reason or (
                        time.perf_counter() - started > protocol["limits"]["run_timeout_seconds"]
                    ):
                        trial["error_type"] = generation.stopped_reason or "run_timeout"
                        checkpoint()
                        continue
                    trial["status"] = "running"
                    checkpoint()
                    models = {
                        role: recorded_model(
                            clients[role],
                            role,
                            protocol["pricing"][role],
                            generation,
                            trial,
                            trial_path,
                        )
                        for role in ("root", "sidecar")
                    }
                    inference_started = time.perf_counter()
                    try:
                        answered = await answer_from_seeds(
                            workspace,
                            case,
                            pool["seeds"][policy],
                            pool["hits"],
                            models["root"],
                            models["sidecar"],
                            budget=Budget(**protocol["budget"]),
                            runtime_options=protocol["runtime"],
                            repl_config=DockerREPLConfig(**protocol["docker"]),
                        )
                        trial.update(answered)
                    except Exception as exc:
                        trial.update(status="failed", error_type=type(exc).__name__)
                    trial["elapsed_seconds"] = time.perf_counter() - inference_started
                    checkpoint()
                    if isinstance(trial.get("answer"), str):
                        for kind in ("accuracy", "support"):
                            judgment = {"status": "running"}
                            trial["judgments"][kind] = judgment
                            role = f"{kind}_judge"
                            judge = recorded_model(
                                clients[role],
                                role,
                                protocol["pricing"]["judge"],
                                judging,
                                trial,
                                trial_path,
                            )
                            try:
                                request = (
                                    accuracy_request(
                                        case, golds[case.case_id], trial["answer"], prompt
                                    )
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
                    done = sum(
                        t["status"] not in {"running", "not_started"} for t in result["trials"]
                    )
                    print(
                        json.dumps(
                            {
                                "completed_trials": done,
                                "planned_trials": 192,
                                "trial_id": trial_id,
                                "status": trial["status"],
                                "generation_estimate_usd": generation.known_cost,
                                "judge_estimate_usd": judging.known_cost,
                            }
                        ),
                        flush=True,
                    )

    outcomes = await asyncio.gather(
        *(run_case(index, row) for index, row in enumerate(prepared["cases"])),
        return_exceptions=True,
    )
    result["case_errors"] = [
        {"case_id": row["case_id"], "error_type": type(outcome).__name__}
        for row, outcome in zip(prepared["cases"], outcomes)
        if isinstance(outcome, BaseException)
    ]
    present = {trial["trial_id"] for trial in result["trials"]}
    for row in prepared["cases"]:
        for pool in row["pools"]:
            for policy in POLICIES:
                trial_id = f"{row['case_id']}__{pool['pool_id']}__{policy}"
                if trial_id not in present:
                    result["trials"].append(
                        {
                            "trial_id": trial_id,
                            "case_id": row["case_id"],
                            "cohort": row["cohort"],
                            "pool_id": pool["pool_id"],
                            "policy": policy,
                            "status": "not_started",
                            "error_type": "case_setup_or_execution_failure",
                            "model_calls": [],
                            "judgments": {},
                            "seed_node_ids": pool["seeds"][policy],
                            "historical_selector": pool["historical_selector"]
                            if policy == "model_selector"
                            else {"elapsed_seconds": 0.0, "estimated_cost_usd": 0.0},
                        }
                    )
    result["completed_judgments"] = sum(
        value.get("status") == "completed"
        for trial in result["trials"]
        for value in trial["judgments"].values()
    )
    result["status"] = (
        "completed"
        if len(result["trials"]) == 192
        and not any(t["status"] in {"running", "not_started"} for t in result["trials"])
        and not result["case_errors"]
        else "stopped"
    )
    checkpoint()
    return result


def main():
    """Prepare without credentials, or explicitly execute a new paid run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/seed_answers_v1.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    prepared, cases, golds = prepare(protocol)
    write_json(args.output / "prepared.json", prepared)
    write_json(args.output / "scoring.json", {key: asdict(value) for key, value in golds.items()})
    freeze_sources(args.output, protocol, args.protocol.resolve())
    official = protocol["inputs"]["official_judge"]
    load_official_prompt(ROOT / official["path"], official["sha256"])
    if not args.execute:
        print(json.dumps({"status": "prepared", "planned_trials": 192, "model_calls": 0}))
        return
    if args.env_file:
        load_env_file(args.env_file)

    async def live():
        """Own the four stateless, retry-free transports for one fixed run."""
        options = {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "timeout_seconds": protocol["limits"]["timeout_seconds_per_call"],
        }
        clients = {
            "root": OpenAIModelClient(protocol["models"]["root"], **options),
            "sidecar": OpenAIModelClient(protocol["models"]["sidecar"], **options),
            "accuracy_judge": OpenAICompatibleModelClient(protocol["models"]["judge"], **options),
            "support_judge": OpenAIModelClient(protocol["models"]["judge"], **options),
        }
        try:
            return await execute(prepared, cases, golds, protocol, args.output, clients)
        finally:
            for client in clients.values():
                await client.aclose()

    result = asyncio.run(live())
    print(
        json.dumps(
            {
                "status": result["status"],
                "trials": len(result["trials"]),
                "allowances": result["allowances"],
            }
        )
    )
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
