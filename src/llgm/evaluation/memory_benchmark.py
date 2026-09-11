"""Evaluate ordinary memory construction and answers on pinned LongMemEval inputs.

Preparation accepts source histories only. Gold enters a separate judging phase.
Every scheduled attempt remains in the denominator, including failed dispatches.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import statistics
import time
from collections import Counter
from contextlib import AsyncExitStack
from dataclasses import asdict, replace
from pathlib import Path

from llgm import LLGM, Budget, Conversation, MaintenancePolicy, Workspace
from llgm.core.environment import load_env_file
from llgm.core.errors import ConfigurationError
from llgm.evaluation.answer_judging import (
    accuracy_request,
    cited_source_evidence,
    load_official_prompt,
    parse_accuracy,
)
from llgm.evaluation.artifacts import write_json, write_jsonl
from llgm.evaluation.baselines import answer_baseline
from llgm.evaluation.benchmark_preflight import preflight_runtime
from llgm.evaluation.costs import Allowance, make_token_pacer, recorded_model
from llgm.evaluation.longmemeval import EvaluationCase, load_longmemeval
from llgm.evaluation.node_search import anonymize_case
from llgm.inference.repl import DockerREPLConfig
from llgm.models.hosted import OpenAICompatibleModelClient, OpenAIModelClient

ARMS = ("llgm", "bm25", "full_context")


def digest(path: Path) -> str:
    """Hash exact file bytes without loading a complete dataset into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint(path: Path, value: dict) -> None:
    """Replace one record atomically so interrupted writes cannot erase an attempt."""
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def prepare(protocol: dict, root: Path) -> tuple[list[EvaluationCase], dict]:
    """Verify dataset and judge identities, then remove benchmark source-ID cues."""
    if protocol.get("schema_version") != 1 or protocol.get("repetitions") != 1:
        raise ConfigurationError("Expected the one-attempt LongMemEval protocol")
    arms = protocol.get("arms", [])
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ConfigurationError("Unknown or duplicate benchmark arms")
    effort = protocol.get("root_reasoning_effort")
    if effort is not None and (not isinstance(effort, str) or not effort.strip()):
        raise ConfigurationError("root_reasoning_effort must be a nonblank string")
    for key in ("dataset", "judge"):
        spec = protocol[key]
        if digest(root / spec["path"]) != spec["sha256"]:
            raise ConfigurationError(f"Pinned {key} bytes do not match")
    cases, gold = load_longmemeval(root / protocol["dataset"]["path"])
    if len(cases) != protocol["dataset"]["case_count"]:
        raise ConfigurationError("Dataset case count does not match protocol")
    selected = protocol["case_ids"]
    if selected != "all":
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
            raise ConfigurationError("case_ids must be all or unique explicit identifiers")
        by_id = {case.case_id: case for case in cases}
        if set(selected) - set(by_id):
            raise ConfigurationError("Selected question is absent from pinned dataset")
        cases = [by_id[identity] for identity in selected]
    if protocol["claim"] == "LongMemEval-S" and (selected != "all" or len(cases) != 500):
        raise ConfigurationError("A full LongMemEval-S claim requires all 500 questions")
    normalized, labels = [], {}
    for case in cases:
        clean, record = anonymize_case(case, gold[case.case_id])
        normalized.append(clean)
        labels[case.case_id] = record
    for key in ("generation_cost_cap_usd", "judge_cost_cap_usd"):
        amount = protocol["limits"][key]
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount <= 0:
            raise ConfigurationError("Cost allowances must be finite and positive")
    return normalized, labels


def schedule(cases: list[EvaluationCase], arms: list[str]) -> list[dict]:
    """Rotate arm order without selecting questions by evidence or answer quality."""
    rows = []
    for index, case in enumerate(cases):
        offset = index % len(arms)
        for arm in arms[offset:] + arms[:offset]:
            rows.append({"trial_id": f"{index:04d}-{arm}", "case_id": case.case_id, "arm": arm})
    return rows


async def ingest_history(
    app: LLGM, sources: tuple, *, organize: bool, maintenance: list | None = None
) -> tuple[tuple, list]:
    """Use normal ingestion for each supplied occurrence without receiving a question."""
    stored = []
    if maintenance is None:
        maintenance = []
    for source in sources:
        outcome = await app.ingest(
            Conversation(source.turns, metadata=source.metadata, timestamp_ms=source.timestamp_ms),
            organize=organize,
        )
        stored.append(replace(source, node_id=outcome.source.node_id))
        maintenance.append(asdict(outcome.maintenance))
    return tuple(stored), maintenance


async def run_trial(case, arm, protocol, directory, clients, allowance, *, pacing=None) -> dict:
    """Construct an isolated history and attempt an answer once, retaining all work."""
    record = {
        "case_id": case.case_id,
        "arm": arm,
        "status": "started",
        "answer": "",
        "model_calls": [],
        "evidence": [],
        "maintenance": [],
    }
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "calls").mkdir()
    checkpoint(directory / "trial_identity.json", {"case_id": case.case_id, "arm": arm})
    checkpoint(directory / "trial.json", record)
    started = time.perf_counter()
    cpu_started = time.process_time()
    phase = "construction"
    query_started = None
    call_metadata = {"case_id": case.case_id, "arm": arm, "phase": phase}
    try:
        models = {
            role: recorded_model(
                client,
                role,
                protocol["pricing"][role],
                allowance,
                record,
                directory / "calls",
                call_metadata=call_metadata,
                pacing=pacing,
                temperature=None
                if role == "root" and protocol.get("root_reasoning_effort") not in (None, "none")
                else 0,
            )
            for role, client in clients.items()
            if role != "judge"
        }
        async with Workspace.open(directory / "workspace") as workspace:
            policy = protocol["maintenance"]
            app = LLGM(
                workspace,
                models["root"],
                models["sidecar"],
                maintenance_model=models["maintenance"],
                maintenance_policy=MaintenancePolicy(
                    **{key: value for key, value in policy.items() if key != "budget"},
                    budget=Budget(**policy["budget"]),
                ),
                inference_budget=Budget(**protocol["budget"]),
                repl_config=DockerREPLConfig(**protocol["docker"]),
                capture_text=True,
                **protocol["runtime"],
            )
            sources, maintenance = await ingest_history(
                app, case.sources, organize=arm == "llgm", maintenance=record["maintenance"]
            )
            record["maintenance"] = maintenance
            record["construction_seconds"] = time.perf_counter() - started
            for call in record["model_calls"]:
                call["phase"] = "construction"
            construction_calls = len(record["model_calls"])
            record["construction_status_counts"] = dict(Counter(x["status"] for x in maintenance))
            record["sources"] = len(sources)
            record["primary_edges"] = sum([len(await workspace.edges(s.node_id)) for s in sources])
            record["journal_amendments"] = 0
            actual_case = replace(case, sources=sources)
            phase = "query"
            call_metadata["phase"] = phase
            query_started = time.perf_counter()
            if arm == "llgm":
                result = await app.answer(case.question, query_date=case.question_date)
                record.update(
                    answer=result.answer,
                    status=result.status,
                    trace=result.trace,
                    runtime_usage=result.usage,
                    unresolved=result.evidence.unresolved,
                )
                record["evidence"] = cited_source_evidence(actual_case, result)
            else:
                record.update(
                    await answer_baseline(
                        arm, workspace, actual_case, models["root"], protocol["reader"]
                    )
                )
            record["query_seconds"] = time.perf_counter() - query_started
            for call in record["model_calls"][construction_calls:]:
                call["phase"] = "query"
    except asyncio.CancelledError:
        record.update(status="interrupted", error_type="CancelledError", failed_phase=phase)
        raise
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, failed_phase=phase)
    finally:
        if query_started is not None:
            record.setdefault("query_seconds", time.perf_counter() - query_started)
        else:
            record.setdefault("construction_seconds", time.perf_counter() - started)
        for call in record["model_calls"]:
            call.setdefault("phase", phase)
        record["elapsed_seconds"] = time.perf_counter() - started
        record["process_cpu_seconds"] = time.process_time() - cpu_started
        record["workspace_bytes"] = sum(
            p.stat().st_size for p in directory.rglob("*") if p.is_file() and "workspace" in p.parts
        )
        checkpoint(directory / "trial.json", record)
    return record


def summarize(planned: list[dict], trials: list[dict], judgments: list[dict]) -> dict:
    """Keep failed and missing attempts visible and refuse incomplete accuracy claims."""
    indexed = {(x["case_id"], x["arm"]): x for x in trials}
    judged = {(x["case_id"], x["arm"]): x for x in judgments}
    if len(indexed) != len(trials) or len(judged) != len(judgments):
        raise ConfigurationError("Duplicate attempts or judgments cannot be selected post hoc")
    allowed = {(x["case_id"], x["arm"]) for x in planned}
    if len(allowed) != len(planned):
        raise ConfigurationError("Duplicate planned attempts are not repetitions")
    if set(indexed) - allowed or set(judged) - allowed:
        raise ConfigurationError("Result is outside the frozen schedule")
    if set(judged) - set(indexed):
        raise ConfigurationError("A judgment requires an attempted answer")
    output = {}
    for arm in dict.fromkeys(x["arm"] for x in planned):
        keys = [(x["case_id"], arm) for x in planned if x["arm"] == arm]
        observed = [indexed[key] for key in keys if key in indexed]
        decisions = [judged.get(key, {}) for key in keys]
        correct = sum(x.get("correct") is True for x in decisions)
        unresolved = sum(type(x.get("correct")) is not bool for x in decisions)
        calls = [call for trial in observed for call in trial["model_calls"]]
        unknown = sum(call.get("estimated_cost_usd") is None for call in calls)
        known = sum(call.get("estimated_cost_usd") or 0 for call in calls)
        latencies = [x["query_seconds"] for x in observed if "query_seconds" in x]
        by_phase = {
            phase: sum(x.get("estimated_cost_usd") or 0 for x in calls if x.get("phase") == phase)
            for phase in ("construction", "query")
        }
        output[arm] = {
            "planned_questions": len(keys),
            "attempted_questions": len(observed),
            "correct": correct,
            "unresolved_judgments": unresolved,
            "accuracy": None if unresolved else correct / len(keys),
            "accuracy_lower_bound": correct / len(keys),
            "accuracy_upper_bound": (correct + unresolved) / len(keys),
            "status_counts": dict(Counter(x["status"] for x in observed)),
            "model_calls": sum(call.get("dispatched") is not False for call in calls),
            "recorded_requests": len(calls),
            "undispatched_requests": sum(call.get("dispatched") is False for call in calls),
            "pacing_wait_seconds": sum(call.get("pacing_wait_seconds", 0) for call in calls),
            "unknown_cost_calls": unknown,
            "known_api_cost_usd": known,
            "known_api_cost_by_phase_usd": by_phase,
            "total_api_cost_usd": None if unknown or len(observed) != len(keys) else known,
            "mean_total_api_cost_usd": None
            if unknown or len(observed) != len(keys)
            else known / len(keys),
            "total_cost_usd": None,
            "total_cost_limitation": "Local CPU, Docker and storage are measured but unpriced",
            "median_query_seconds": statistics.median(latencies) if latencies else None,
            "p95_query_seconds": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
            if latencies
            else None,
        }
    return output


def report(directory: Path) -> dict:
    """Rebuild complete-denominator reports from durable records without new model calls."""

    def read_optional(path, default):
        """Read a published checkpoint while tolerating an interruption before creation."""
        return json.loads(path.read_text()) if path.exists() else default

    def reconcile_calls(folder, aggregate, identity):
        """Merge final aggregates with per-call files without charging either copy twice."""
        calls = {}
        for number, call in enumerate(aggregate, 1):
            key = call.get("call_id") or f"{folder}/call-{number:03d}.json"
            calls[key] = dict(call)
        for path in sorted(folder.glob("call-*.json")):
            call = json.loads(path.read_text())
            key = call.get("call_id") or str(path)
            prior = calls.get(key, {})
            # An aggregate may retain a returned response after the final call
            # file update failed; a dispatched checkpoint must not erase it.
            calls[key] = (
                {**call, **prior}
                if "response" in prior and "response" not in call
                else {**prior, **call}
            )
        for key, call in calls.items():
            call.setdefault("call_id", key)
            for name, value in identity.items():
                if name in call and call[name] != value:
                    raise ConfigurationError("Call identity disagrees with its owning attempt")
                call.setdefault(name, value)
            call.setdefault(
                "phase", "construction" if call.get("role") == "maintenance" else "query"
            )
        return list(calls.values())

    def metrics(planned, trials, judgments):
        """Add explicit latency denominators to complete-schedule correctness metrics."""
        result = summarize(planned, trials, judgments)
        for arm, row in result.items():
            sampled = [
                trial for trial in trials if trial["arm"] == arm and "query_seconds" in trial
            ]
            row["query_latency_samples"] = len(sampled)
            row["query_latency_status_counts"] = dict(Counter(trial["status"] for trial in sampled))
        return result

    protocol = json.loads((directory / "protocol.json").read_text())
    planned = json.loads((directory / "schedule.json").read_text())
    planned_by_id = {item["trial_id"]: item for item in planned}
    trials = []
    for folder in sorted(directory.glob("trials/*")):
        if not folder.is_dir():
            continue
        record = read_optional(folder / "trial.json", None)
        identity = read_optional(folder / "trial_identity.json", planned_by_id.get(folder.name))
        if record is None and (
            identity is None or not list((folder / "calls").glob("call-*.json"))
        ):
            continue
        if record is None:
            record = {**identity, "status": "interrupted", "answer": "", "model_calls": []}
        identity = {name: record[name] for name in ("case_id", "arm")}
        record["model_calls"] = reconcile_calls(folder / "calls", record["model_calls"], identity)
        trials.append(record)
    judgments = read_optional(directory / "judgments.json", {"records": []})["records"]
    saved_judgments = list(judgments)
    judged_by_key = {(item["case_id"], item["arm"]): item for item in judgments}
    unattributed_judge_calls = []
    for folder in sorted(directory.glob("judgments/*")):
        if not folder.is_dir():
            continue
        identity = read_optional(folder / "identity.json", {})
        files = sorted(folder.glob("call-*.json"))
        if not identity and files:
            first_call = json.loads(files[0].read_text())
            identity = {name: first_call[name] for name in ("case_id", "arm") if name in first_call}
        # Earlier runs identify completed judgment folders by their saved order.
        if not {"case_id", "arm"}.issubset(identity) and folder.name.isdecimal():
            index = int(folder.name)
            if index < len(saved_judgments):
                identity = {name: saved_judgments[index][name] for name in ("case_id", "arm")}
        if not {"case_id", "arm"}.issubset(identity):
            unattributed_judge_calls.extend(reconcile_calls(folder, [], {"phase": "judging"}))
            continue
        identity = {name: identity[name] for name in ("case_id", "arm")}
        key = (identity["case_id"], identity["arm"])
        record = judged_by_key.get(key)
        if record is None:
            record = {**identity, "correct": None, "status": "interrupted", "model_calls": []}
            judgments.append(record)
            judged_by_key[key] = record
        record["model_calls"] = reconcile_calls(
            folder, record["model_calls"], {**identity, "phase": "judging"}
        )
    indexed = {(trial["case_id"], trial["arm"]): trial for trial in trials}
    results = metrics(planned, trials, judgments)
    labels = json.loads((directory / "gold.json").read_text())
    for arm in protocol["arms"]:
        write_jsonl(
            directory / f"predictions-{arm}.jsonl",
            [
                {
                    "question_id": item["case_id"],
                    "hypothesis": indexed.get((item["case_id"], arm), {}).get("answer", ""),
                }
                for item in planned
                if item["arm"] == arm
            ],
        )
        results[arm]["categories"] = {}
        for ability in sorted({record["ability"] for record in labels.values()}):
            subset = [
                item
                for item in planned
                if item["arm"] == arm and labels[item["case_id"]]["ability"] == ability
            ]
            ids = {item["case_id"] for item in subset}
            results[arm]["categories"][ability] = metrics(
                subset,
                [x for x in trials if x["arm"] == arm and x["case_id"] in ids],
                [x for x in judgments if x["arm"] == arm and x["case_id"] in ids],
            )[arm]
    prior = read_optional(directory / "summary.json", {})
    saved_allowances = read_optional(directory / "allowances.json", None)
    if saved_allowances is None:
        saved_allowances = prior.get("recorded_allowances") or {
            name: prior[name] for name in ("generation", "judging") if name in prior
        }
        progress = read_optional(directory / "progress.json", {})
        if "generation" not in saved_allowances and "generation" in progress:
            saved_allowances["generation"] = progress["generation"]
    generation_calls = [call for trial in trials for call in trial["model_calls"]]
    judge_calls = [call for judgment in judgments for call in judgment["model_calls"]]
    judge_calls.extend(unattributed_judge_calls)

    def recovered_allowance(name, calls):
        """Preserve admission state while recovering known cost and unresolved call liabilities."""

        def reserved_total(records):
            """Keep missing reservation amounts unknown rather than reporting zero liability."""
            amounts = [record.get("reserved_cost_usd") for record in records]
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in amounts
            ):
                return None
            return sum(amounts)

        known = sum(call.get("estimated_cost_usd") or 0 for call in calls)
        unknown = [call for call in calls if call.get("estimated_cost_usd") is None]
        pending = [call for call in unknown if call.get("status") == "dispatched"]
        unresolved = [call for call in unknown if call.get("status") != "dispatched"]
        return {
            **saved_allowances.get(name, {}),
            "known_estimated_cost_usd": known,
            "pending_reserved_cost_usd": reserved_total(pending),
            "unknown_reserved_cost_usd": reserved_total(unresolved),
            "unknown_cost_calls": len(unknown),
            "estimated_cost_usd": None if unknown else known,
        }

    summary = {
        "status": "completed"
        if len(trials) == len(planned) and all(x["accuracy"] is not None for x in results.values())
        else "incomplete",
        "claim": protocol["claim"],
        "results": results,
        "generation": recovered_allowance("generation", generation_calls),
        "judging": recovered_allowance("judging", judge_calls),
        "recorded_allowances": saved_allowances,
        "judge_known_api_cost_usd": sum(
            call.get("estimated_cost_usd") or 0 for call in judge_calls
        ),
        "judge_unknown_cost_calls": sum(
            call.get("estimated_cost_usd") is None for call in judge_calls
        ),
        "unattributed_judge_call_ids": [call["call_id"] for call in unattributed_judge_calls],
        "protocol_sha256": digest(directory / "protocol.json"),
        "prediction_sha256": {
            arm: digest(directory / f"predictions-{arm}.jsonl") for arm in protocol["arms"]
        },
    }
    checkpoint(directory / "summary.json", summary)
    lines = [
        "# LongMemEval results",
        "",
        f"Status: {summary['status']}. {protocol['claim']}.",
        "",
        "| Arm | Questions | Correct | Accuracy | Known API cost | Unknown calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, row in results.items():
        accuracy = "incomplete" if row["accuracy"] is None else f"{row['accuracy']:.1%}"
        lines.append(
            f"| {arm} | {row['planned_questions']} | {row['correct']} | {accuracy} | "
            f"${row['known_api_cost_usd']:.4f} | {row['unknown_cost_calls']} |"
        )
    lines += [
        "",
        "API cost includes memory construction and answering. Judge costs are separate.",
        "Local compute and storage are unpriced. These results cannot establish equal total cost.",
        "All scheduled questions remain in the denominator. Missing judgments keep accuracy incomplete.",
        "The JSON summary includes category results, latency and failure counts.",
    ]
    (directory / "results.md").write_text("\n".join(lines) + "\n")
    return summary


async def execute(protocol: dict, root: Path, directory: Path) -> dict:
    """Freeze one run, finish answer attempts, then judge saved answers separately."""
    cases, gold = prepare(protocol, root)
    prompt = load_official_prompt(root / protocol["judge"]["path"], protocol["judge"]["sha256"])
    prerequisites = preflight_runtime(protocol)
    directory.mkdir(parents=True, exist_ok=False)
    planned = schedule(cases, protocol["arms"])
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "schedule.json", planned)
    write_json(directory / "gold.json", {key: asdict(value) for key, value in gold.items()})
    frozen = directory / "source" / "llgm"
    shutil.copytree(
        Path(__file__).resolve().parents[1], frozen, ignore=shutil.ignore_patterns("__pycache__")
    )
    write_json(
        directory / "source_hashes.json",
        {str(path.relative_to(frozen)): digest(path) for path in frozen.rglob("*.py")},
    )
    write_json(
        directory / "environment.json",
        {
            "python": platform.python_version(),
            "openai": importlib.metadata.version("openai"),
            "dataset_sha256": protocol["dataset"]["sha256"],
            "official_judge_sha256": protocol["judge"]["sha256"],
            "runtime_prerequisites": prerequisites,
        },
    )
    deadline = time.monotonic() + protocol["limits"]["run_timeout_seconds"]
    unknown_limit = protocol["limits"].get("max_unknown_calls", 16)
    generation = Allowance(
        protocol["limits"]["generation_cost_cap_usd"], unknown_limit, deadline=deadline
    )
    judging = Allowance(protocol["limits"]["judge_cost_cap_usd"], unknown_limit)
    token_limit = protocol["limits"].get("tokens_per_minute")
    generation_pacer = make_token_pacer(token_limit) if token_limit is not None else None
    judge_pacer = make_token_pacer(token_limit) if token_limit is not None else None
    trials, judgments = [], []
    by_id = {case.case_id: case for case in cases}

    def save_allowances():
        """Retain admission state separately from derived score tables."""
        checkpoint(
            directory / "allowances.json",
            {"generation": generation.record(), "judging": judging.record()},
        )

    save_allowances()
    async with AsyncExitStack() as stack:
        clients = {}
        for role, model in protocol["models"].items():
            factory = OpenAICompatibleModelClient if role == "judge" else OpenAIModelClient
            clients[role] = await stack.enter_async_context(
                factory(
                    model,
                    timeout_seconds=protocol["limits"]["timeout_seconds_per_call"],
                    **({"base_url": "https://api.openai.com/v1"} if role == "judge" else {}),
                    **(
                        {"reasoning_effort": protocol["root_reasoning_effort"]}
                        if role == "root" and protocol.get("root_reasoning_effort") is not None
                        else {}
                    ),
                )
            )
        slots = asyncio.Semaphore(protocol["limits"].get("concurrent_cases", 1))

        async def run_case(case):
            """Keep arm order fixed within each independently constructed case."""
            async with slots:
                for item in [item for item in planned if item["case_id"] == case.case_id]:
                    if generation.stopped_reason or time.monotonic() >= deadline:
                        return
                    trial = await run_trial(
                        case,
                        item["arm"],
                        protocol,
                        directory / "trials" / item["trial_id"],
                        clients,
                        generation,
                        pacing=generation_pacer,
                    )
                    trials.append(trial)
                    save_allowances()
                    checkpoint(
                        directory / "progress.json",
                        {
                            "completed_attempts": len(trials),
                            "planned": len(planned),
                            "generation": generation.record(),
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "attempt": len(trials),
                                "planned": len(planned),
                                "arm": item["arm"],
                                "status": trial["status"],
                            }
                        ),
                        flush=True,
                    )

        tasks = [asyncio.create_task(run_case(case)) for case in cases]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            save_allowances()
        order = {(item["case_id"], item["arm"]): index for index, item in enumerate(planned)}
        trials.sort(key=lambda trial: order[(trial["case_id"], trial["arm"])])
        for arm in protocol["arms"]:
            write_jsonl(
                directory / f"predictions-{arm}.jsonl",
                [
                    {"question_id": x["case_id"], "hypothesis": x["answer"]}
                    for x in trials
                    if x["arm"] == arm
                ],
            )
        judging.deadline = time.monotonic() + protocol["limits"].get("judge_timeout_seconds", 3600)
        for number, trial in enumerate(trials):
            judgment = {"case_id": trial["case_id"], "arm": trial["arm"], "model_calls": []}
            if (
                trial["status"] in {"failed", "budget_exhausted", "context_exceeded"}
                or not trial["answer"].strip()
            ):
                judgment.update(correct=False, status="operational_failure", official_label=None)
            else:
                (directory / "judgments" / f"{number:04d}").mkdir(parents=True)
                identity = {"case_id": trial["case_id"], "arm": trial["arm"], "phase": "judging"}
                write_json(directory / "judgments" / f"{number:04d}" / "identity.json", identity)
                model = recorded_model(
                    clients["judge"],
                    "judge",
                    protocol["pricing"]["judge"],
                    judging,
                    judgment,
                    directory / "judgments" / f"{number:04d}",
                    call_metadata=identity,
                    pacing=judge_pacer,
                )
                try:
                    response = await model.complete(
                        accuracy_request(
                            by_id[trial["case_id"]], gold[trial["case_id"]], trial["answer"], prompt
                        )
                    )
                    response.ensure_complete()
                    judgment["official_label"] = "yes" in response.text.lower()
                    judgment.update(correct=parse_accuracy(response.text), status="completed")
                except Exception as exc:
                    judgment.update(
                        correct=None, status="judge_failed", error_type=type(exc).__name__
                    )
            judgments.append(judgment)
            checkpoint(directory / "judgments.json", {"records": judgments})
            save_allowances()
    results = summarize(planned, trials, judgments)
    summary = {
        "status": "completed"
        if len(trials) == len(planned)
        and all(row["accuracy"] is not None for row in results.values())
        else "incomplete",
        "claim": protocol["claim"],
        "results": results,
        "generation": generation.record(),
        "judging": judging.record(),
    }
    checkpoint(directory / "summary.json", summary)
    return summary


def main() -> None:
    """Validate without paid calls by default, or explicitly execute the frozen protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/longmemeval_pilot_v6.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--report", action="store_true", help="Rebuild an existing run report offline"
    )
    args = parser.parse_args()
    if args.report:
        if args.execute:
            parser.error("--report cannot dispatch model calls")
        report(args.output)
        return
    root = Path.cwd()
    protocol = json.loads(args.protocol.read_text())
    cases, _ = prepare(protocol, root)
    if not args.execute:
        if args.output.exists():
            parser.error("Output must be a new directory")
        write_json(
            args.output / "preflight.json",
            {
                "protocol": protocol,
                "schedule": schedule(cases, protocol["arms"]),
                "model_calls": 0,
                "source_occurrences": sum(len(x.sources) for x in cases),
            },
        )
        print(f"Validated {len(cases)} questions. No model calls made.")
        return
    if args.env_file:
        load_env_file(args.env_file)
    try:
        asyncio.run(execute(protocol, root, args.output))
    finally:
        if (args.output / "schedule.json").exists():
            report(args.output)


if __name__ == "__main__":
    main()
