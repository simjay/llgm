"""Evaluate hosted node selection over frozen, real BM25 and ColBERT retrieval."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

from llgm.core.environment import load_env_file
from llgm.evaluation.costs import request_reservation
from llgm.evaluation.longmemeval import GoldRecord, parse_longmemeval
from llgm.evaluation.node_search import anonymize_case, node_coverage_metrics
from llgm.evaluation.node_selection import (
    canonical_key,
    combine_hits,
    hydrate_candidates,
    parse_selection,
    selection_request,
)
from llgm.models.base import Message, ModelRequest, Usage
from llgm.models.hosted import OpenAIModelClient

ROOT = Path(__file__).resolve().parents[1]
POOLS = ("bm25_40", "colbert_40", "rrf_40")


def digest(path: Path) -> str:
    """Hash exact input bytes without interpreting their contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Replace a checkpoint atomically within its existing directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def usage_cost(usage: Usage, pricing: dict) -> float | None:
    """Price actual tokens once, applying cached-input pricing only when known."""
    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    details = usage.extra.get("input_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if type(cached) is not int or not 0 <= cached <= usage.input_tokens:
        return None
    return (
        (usage.input_tokens - cached) * pricing["usd_per_million_input_tokens"]
        + cached * pricing["usd_per_million_cached_input_tokens"]
        + usage.output_tokens * pricing["usd_per_million_output_tokens"]
    ) / 1_000_000


def restore_request(record: dict) -> ModelRequest:
    """Restore only the declared provider-neutral request fields."""
    return ModelRequest(
        tuple(Message(**message) for message in record["messages"]),
        max_output_tokens=record["max_output_tokens"],
        temperature=record["temperature"],
        output_schema=record["output_schema"],
    )


def prepare(protocol: dict, root: Path = ROOT) -> tuple[dict, dict[str, GoldRecord]]:
    """Verify pinned retrieval and rebuild model requests without provider calls."""
    from tools.node_search_cases import controlled_cases, controlled_manifest

    if (
        protocol["schema_version"] != 1
        or protocol["pools"] != list(POOLS)
        or protocol["retrieval_k"] != 40
        or protocol["max_seed_nodes"] != 3
        or protocol["repetitions"] != 3
        or protocol["limits"]["concurrency"] != 1
        or protocol["limits"]["sdk_retries"] != 0
    ):
        raise ValueError("Unsupported selection protocol")
    for spec in protocol["inputs"].values():
        if digest(root / spec["path"]) != spec["sha256"]:
            raise ValueError(f"Frozen input mismatch: {spec['path']}")

    def read(name: str):
        """Read a previously verified named JSON input."""
        return json.loads((root / protocol["inputs"][name]["path"]).read_text())

    if read("controlled_manifest") != controlled_manifest():
        raise ValueError("Controlled source generator differs from frozen manifest")
    benchmark_ids = protocol["benchmark_case_ids"]
    parent = read("benchmark_protocol")
    if benchmark_ids != [x for ids in parent["expanded_selection"]["strata"].values() for x in ids]:
        raise ValueError("Benchmark case selection changed")
    cases, gold = parse_longmemeval(
        [row for row in read("dataset") if row["question_id"] in benchmark_ids]
    )
    by_id = {case.case_id: anonymize_case(case, gold[case.case_id]) for case in cases}
    controlled = dict((case.case_id, (case, labels)) for case, labels in controlled_cases())
    if set(controlled) != set(protocol["controlled_case_ids"]):
        raise ValueError("Controlled case selection changed")
    sources = {
        "expanded_longmemeval": {
            row["case_id"]: row for row in read("benchmark_retrieval")["cases"]
        },
        "controlled": {
            row["case_id"]: row
            for row in read("controlled_retrieval")["cases"]
            if row["cohort"] == "controlled"
        },
    }
    planned = [
        {"case_id": case_id, "cohort": cohort}
        for cohort, ids in (
            ("expanded_longmemeval", benchmark_ids),
            ("controlled", protocol["controlled_case_ids"]),
        )
        for case_id in ids
    ]
    if len(planned) != 32 or len({item["case_id"] for item in planned}) != 32:
        raise ValueError("Expected 32 unique histories")
    prepared = {"schema_version": 1, "planned_cases": planned, "cases": []}
    labels_by_id = {}
    for plan in planned:
        case_id, cohort = plan["case_id"], plan["cohort"]
        case, labels = (by_id if cohort == "expanded_longmemeval" else controlled)[case_id]
        labels_by_id[case_id] = labels
        original = sources[cohort][case_id]
        arms = {row["backend"]: row for row in original["arms"] if row["retrieval_k"] == 40}
        if (
            len(arms) != 2
            or any(row["status"] != "passed" for row in arms.values())
            or len({row["corpus_fingerprint"] for row in arms.values()}) != 1
            or any(row["question"] != case.question for row in arms.values())
        ):
            raise ValueError(f"Invalid source retrieval arms for {case_id}")
        bm25, colbert = arms["bm25"]["first"]["hits"], arms["colbertv2_plaid"]["first"]["hits"]
        union = combine_hits(bm25, colbert, k=80, rrf_constant=protocol["rrf_constant"])
        pools = (bm25, colbert, union[:40])
        row = {
            **plan,
            "union_metrics": node_coverage_metrics(
                [canonical_key(hit)[0][0] for hit in union], [], labels
            ),
            "pools": [],
        }
        for pool_id, hits in zip(POOLS, pools):
            candidates = hydrate_candidates(case, hits)
            owners = [canonical_key(hit)[0][0] for hit in hits]
            selected = list(dict.fromkeys(owners))[:3]
            metrics = node_coverage_metrics(owners, selected, labels)
            if pool_id != "rrf_40":
                backend = "bm25" if pool_id == "bm25_40" else "colbertv2_plaid"
                saved = arms[backend]["first"]
                if selected != saved["selected_node_ids"] or metrics != saved["metrics"]:
                    raise ValueError(f"Original application control changed: {case_id}/{pool_id}")
            request = selection_request(
                case.question,
                case.question_date,
                candidates,
                max_seed_nodes=protocol["max_seed_nodes"],
                max_output_tokens=protocol["model"]["max_output_tokens"],
            )
            if request.temperature != protocol["model"]["temperature"]:
                raise ValueError("Request temperature differs from protocol")
            serialized = asdict(request)
            encoded = json.dumps(serialized, ensure_ascii=False, sort_keys=True).encode("utf-8")
            if len(encoded) > protocol["limits"]["max_prompt_bytes"]:
                raise ValueError(f"Prompt exceeds fixed byte cap: {case_id}/{pool_id}")
            input_bound, reserved_cost = request_reservation(request, protocol["pricing"])
            row["pools"].append(
                {
                    "pool_id": pool_id,
                    "hits": hits,
                    "candidates": candidates,
                    "request": serialized,
                    "prompt_sha256": hashlib.sha256(encoded).hexdigest(),
                    "prompt_bytes": len(encoded),
                    "reserved_input_tokens": input_bound,
                    "reserved_cost_usd": reserved_cost,
                    "baseline": {"selected_node_ids": selected, "metrics": metrics},
                }
            )
        prepared["cases"].append(row)
    prepared["reserved_cost_usd"] = protocol["repetitions"] * sum(
        pool["reserved_cost_usd"] for case in prepared["cases"] for pool in case["pools"]
    )
    if prepared["reserved_cost_usd"] > protocol["limits"]["max_estimated_cost_usd"]:
        raise ValueError("Complete frozen run exceeds conservative local cost reservation")
    return prepared, labels_by_id


async def execute(prepared: dict, labels: dict, protocol: dict, directory: Path, client) -> dict:
    """Retain every real attempt, stop on limits, and never retry failed selections."""
    result = {
        "schema_version": 1,
        "status": "running",
        "protocol": protocol,
        "planned_cases": prepared["planned_cases"],
        "planned_case_ids": [case["case_id"] for case in prepared["cases"]],
        "cases": [
            {
                "case_id": case["case_id"],
                "cohort": case["cohort"],
                "union_metrics": case["union_metrics"],
                "pools": [
                    {key: pool[key] for key in ("pool_id", "prompt_sha256", "baseline")}
                    | {"trials": []}
                    for pool in case["pools"]
                ],
            }
            for case in prepared["cases"]
        ],
        "attempted_model_calls": 0,
        "estimated_cost_usd": 0.0,
        "known_estimated_cost_usd": 0.0,
        "unknown_cost_calls": 0,
        "reserved_cost_usd": prepared["reserved_cost_usd"],
        "client": dict(client.descriptor()),
        "retrieval_execution": "frozen actual retrieval replay, no new GPU requests",
    }
    started, failed = time.perf_counter(), 0
    output = directory / "node-selection.json"
    for case_index, case in enumerate(prepared["cases"]):
        for repetition in range(protocol["repetitions"]):
            offset = (case_index + repetition) % len(POOLS)
            for pool_index in [(offset + index) % len(POOLS) for index in range(len(POOLS))]:
                if (
                    failed >= protocol["limits"]["max_failed_calls"]
                    or time.perf_counter() - started >= protocol["limits"]["run_timeout_seconds"]
                ):
                    result["status"] = "stopped"
                    result["elapsed_seconds"] = time.perf_counter() - started
                    write_json(output, result)
                    return result
                pool = case["pools"][pool_index]
                trial = {"repetition": repetition, "status": "running", "estimated_cost_usd": None}
                result["cases"][case_index]["pools"][pool_index]["trials"].append(trial)
                result["attempted_model_calls"] += 1
                write_json(output, result)
                call_started = time.perf_counter()
                try:
                    response = await client.complete(restore_request(pool["request"]))
                    trial["response"] = asdict(response)
                    trial["estimated_cost_usd"] = usage_cost(response.usage, protocol["pricing"])
                    response.ensure_complete()
                    if response.model != protocol["model"]["model"]:
                        raise ValueError("Provider returned a different model identity")
                    if (
                        response.usage.input_tokens is None
                        or response.usage.output_tokens is None
                        or response.usage.input_tokens > pool["reserved_input_tokens"]
                        or response.usage.output_tokens > protocol["model"]["max_output_tokens"]
                    ):
                        failed = protocol["limits"]["max_failed_calls"]
                        raise ValueError("Provider usage is unknown or exceeds local reservation")
                    selected = parse_selection(response.text, pool["candidates"], max_seed_nodes=3)
                    trial.update(selected)
                    trial["metrics"] = node_coverage_metrics(
                        [canonical_key(hit)[0][0] for hit in pool["hits"]],
                        selected["selected_node_ids"],
                        labels[case["case_id"]],
                    )
                    trial["status"] = "completed"
                except Exception as exc:
                    failed += 1
                    trial["status"] = "failed"
                    trial["error_type"] = type(exc).__name__
                trial["elapsed_seconds"] = time.perf_counter() - call_started
                if trial["estimated_cost_usd"] is None:
                    result["unknown_cost_calls"] += 1
                else:
                    result["known_estimated_cost_usd"] += trial["estimated_cost_usd"]
                result["estimated_cost_usd"] = (
                    None if result["unknown_cost_calls"] else result["known_estimated_cost_usd"]
                )
                result["elapsed_seconds"] = time.perf_counter() - started
                write_json(output, result)
                print(
                    json.dumps(
                        {
                            "case": case["case_id"],
                            "pool": pool["pool_id"],
                            "repeat": repetition,
                            "status": trial["status"],
                            "calls": result["attempted_model_calls"],
                            "known_estimated_cost_usd": round(
                                result["known_estimated_cost_usd"], 6
                            ),
                        }
                    ),
                    flush=True,
                )
    result["status"] = "completed" if failed == 0 else "stopped"
    write_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Prepare reproducible requests, then optionally execute the authorized run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "experiments/node_selection_v1.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory must be new. Preserve every previous attempt.")
    protocol = json.loads(args.protocol.read_text())
    prepared, labels = prepare(protocol)
    args.output.mkdir(parents=True)
    write_json(args.output / "prepared.json", prepared)
    write_json(
        args.output / "scoring.json",
        {case_id: asdict(value) | {"answer": None} for case_id, value in labels.items()},
    )
    shutil.copyfile(args.protocol, args.output / "protocol.json")
    files = sorted((ROOT / "src/llgm").rglob("*.py")) + [
        Path(__file__),
        ROOT / "tools/node_search_cases.py",
    ]
    source_records = []
    for path in files:
        relative = path.relative_to(ROOT)
        target = args.output / "executed-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_records.append({"path": str(relative), "sha256": digest(path)})
    write_json(
        args.output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "openai_sdk": importlib.metadata.version("openai"),
            "protocol_sha256": digest(args.protocol),
            "sources": source_records,
        },
    )
    print(
        json.dumps(
            {
                "prepared_cases": len(prepared["cases"]),
                "planned_model_calls": 288,
                "reserved_cost_usd": prepared["reserved_cost_usd"],
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    if not args.execute:
        return 0
    if args.env_file:
        load_env_file(args.env_file)

    async def run():
        """Own the configured hosted client for this one bounded experiment."""
        async with OpenAIModelClient(
            protocol["model"]["model"],
            api_key_env=protocol["model"]["api_key_env"],
            timeout_seconds=protocol["limits"]["timeout_seconds_per_call"],
        ) as client:
            return await execute(prepared, labels, protocol, args.output, client)

    result = asyncio.run(run())
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
