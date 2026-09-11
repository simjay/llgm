"""Run frozen retrieval and seed-selection diagnostics in the pinned GPU environment."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

from llgm.evaluation.longmemeval import parse_longmemeval
from llgm.retrieval.base import corpus_fingerprint


def selected_longmemeval_ids(protocol: dict) -> list[str]:
    """Preserve frozen replication order followed by the six declared strata."""
    return list(protocol["replication_indexes"]) + [
        case_id for ids in protocol["expanded_selection"]["strata"].values() for case_id in ids
    ]


def verify_selection(rows: list[dict], protocol: dict) -> None:
    """Reject a changed dataset selection before running any retrieval."""
    selection = protocol["expanded_selection"]
    for category, expected in selection["strata"].items():
        eligible = [
            str(row["question_id"])
            for row in rows
            if row["question_type"] == category
            and not str(row["question_id"]).endswith("_abs")
            and str(row["question_id"]) not in selection["exclude"]
        ]
        actual = sorted(
            eligible,
            key=lambda case_id: hashlib.sha256(
                (selection["salt"] + ":" + case_id).encode()
            ).hexdigest(),
        )[:4]
        if actual != expected:
            raise ValueError(f"Frozen case selection differs for {category}")
    ids = selected_longmemeval_ids(protocol)
    expected_count = protocol.get("expected_longmemeval_cases", 27)
    if len(ids) != len(set(ids)) or len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} distinct frozen LongMemEval questions")


def cutoff_comparison(arms: list[dict]) -> dict:
    """Check observed ranking-prefix stability without assuming how the backend uses k."""
    result = {}
    for backend in ("bm25", "colbertv2_plaid"):
        rows = {
            row["retrieval_k"]: row
            for row in arms
            if row["backend"] == backend and row["status"] == "passed"
        }
        if set(rows) != {12, 40}:
            result[backend] = {"status": "incomplete"}
            continue
        small, large = rows[12]["first"], rows[40]["first"]
        result[backend] = {
            "status": "passed",
            "selected_order_equal": small["selected_node_ids"] == large["selected_node_ids"],
            "candidate_prefix_equal": [hit["passage_id"] for hit in small["hits"]]
            == [hit["passage_id"] for hit in large["hits"][: len(small["hits"])]],
        }
    return result


def run_experiment(run_id: str, protocol_name: str = "node_search_v1.json") -> dict:
    """Retain every attempted arm and stop after three failed histories without retries."""
    import colbert_worker as worker
    from node_search_cases import controlled_cases, controlled_manifest

    from llgm.evaluation.node_search import anonymize_case, evaluate_node_search
    from llgm.retrieval.bm25 import SQLiteBM25Retriever
    from llgm.retrieval.passages import split_nodes
    from llgm.retrieval.tokenizers import ColBERTTokenizer

    worker._name(run_id, "run_id")
    directory = worker._path("runs", run_id)
    directory.mkdir(parents=True, exist_ok=True)
    if protocol_name not in {"node_search_v1.json", "node_search_opaque_v1.json"}:
        raise ValueError("Unsupported node-search protocol file")
    protocol_path = worker.ROOT / "experiments" / protocol_name
    protocol = worker._read_json(protocol_path)
    pins = worker.load_pins()
    if (
        worker._digest(worker.ROOT / "experiments/colbert_modal.json")
        != protocol["colbert_pins_sha256"]
    ):
        raise ValueError("ColBERT configuration differs from the frozen node-search protocol")
    manifest_path = worker.ROOT / protocol["controlled_manifest"]
    if (
        worker._digest(manifest_path) != protocol["controlled_manifest_sha256"]
        or worker._read_json(manifest_path) != controlled_manifest()
    ):
        raise ValueError("Controlled source text or labels differ from the frozen manifest")
    dataset_path = worker._path("datasets", pins["dataset"]["file"])
    if worker._digest(dataset_path) != pins["dataset"]["sha256"]:
        raise ValueError("LongMemEval bytes differ from the pinned release")
    raw = json.loads(dataset_path.read_text())
    verify_selection(raw, protocol)
    selected_ids = selected_longmemeval_ids(protocol)
    cases, gold = parse_longmemeval([row for row in raw if row["question_id"] in selected_ids])
    by_id = {case.case_id: case for case in cases}
    cohort = [(by_id[case_id], gold[case_id]) for case_id in selected_ids]
    identity_policy = protocol.get("source_identity_policy", "original")
    if identity_policy == "opaque_case_node_v1":
        cohort = [anonymize_case(case, labels) for case, labels in cohort]
    elif identity_policy != "original":
        raise ValueError("Unknown source identity policy")
    if protocol.get("include_controlled", True):
        cohort += controlled_cases()
    del raw, cases, by_id, gold
    for relative in (
        f"experiments/{protocol_name}",
        "experiments/node_search_controlled.json",
        "tools/node_search_cases.py",
        "tools/node_search_worker.py",
        "src/llgm/evaluation/node_search.py",
    ):
        target = directory / "frozen" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(worker.ROOT / relative, target)
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "protocol_sha256": worker._digest(protocol_path),
        "protocol": protocol,
        "planned_case_ids": [case.case_id for case, _ in cohort],
        "cases": [],
        "cost_usd": None,
        "model_calls": 0,
    }
    worker._write_json(directory / "node-search.json", result)
    tokenizer = ColBERTTokenizer(
        worker._path("checkpoint"), revision=pins["checkpoint"]["revision"]
    )
    failures = 0
    started = time.perf_counter()
    for index, (case, gold) in enumerate(cohort):
        row = {
            "case_id": case.case_id,
            "question": case.question,
            "ability": gold.ability,
            "cohort": (
                "replication"
                if case.case_id in protocol["replication_indexes"]
                else "expanded_longmemeval"
                if index < len(selected_ids)
                else "controlled"
            ),
            "status": "failed",
            "arms": [],
        }
        print(f"History {index + 1}/{len(cohort)}: {case.case_id}", flush=True)
        try:
            passages = split_nodes(case.sources, tokenizer, **protocol["chunking"])
            row.update(
                passage_count=len(passages),
                source_count=len(case.sources),
                corpus_fingerprint=corpus_fingerprint(passages),
            )
            index_id = protocol["replication_indexes"].get(case.case_id)
            opening = time.perf_counter()
            if index_id:
                record = worker._record(index_id)
                if record["corpus_fingerprint"] != row["corpus_fingerprint"]:
                    raise ValueError("Cached ColBERT corpus differs from current canonical chunks")
                colbert = worker._open(index_id, record)
                row["colbert_index"] = {
                    "index_id": index_id,
                    "reused": True,
                    "open_seconds": time.perf_counter() - opening,
                    "descriptor": colbert.descriptor(),
                }
            else:
                built = worker.build_history(case, gold, run_id, pins)
                index_id = built["index_id"]
                colbert = worker._open(index_id, worker._record(index_id))
                row["colbert_index"] = built
            with tempfile.TemporaryDirectory(prefix="llgm-node-search-") as temporary:
                temporary_path = Path(temporary)
                bm25_started = time.perf_counter()
                bm25 = SQLiteBM25Retriever.from_passages(passages, temporary_path / "bm25.sqlite3")
                row["bm25_build_seconds"] = time.perf_counter() - bm25_started
                try:
                    arms = list(protocol["arms"])
                    if index % 2:
                        arms.reverse()
                    for arm in arms:
                        trial = {**arm, "status": "failed"}
                        backend = arm["backend"]
                        retriever = bm25 if backend == "bm25" else colbert
                        try:
                            trial.update(
                                asyncio.run(
                                    evaluate_node_search(
                                        case,
                                        gold,
                                        passages,
                                        retriever,
                                        temporary_path / f"{backend}-{arm['retrieval_k']}",
                                        retrieval_k=arm["retrieval_k"],
                                        max_seed_nodes=arm["max_seed_nodes"],
                                        warm_repetitions=protocol["warm_repetitions"],
                                    )
                                )
                            )
                            trial["status"] = "passed"
                        except Exception as error:
                            trial.update(error_type=type(error).__name__, error=str(error))
                        row["arms"].append(trial)
                        worker._write_json(directory / f"case-{case.case_id}.json", row)
                        print(f"  {backend} k={arm['retrieval_k']}: {trial['status']}", flush=True)
                finally:
                    bm25.close()
            row["cutoff_comparison"] = cutoff_comparison(row["arms"])
            row["status"] = (
                "passed" if all(arm["status"] == "passed" for arm in row["arms"]) else "failed"
            )
        except Exception as error:
            row.update(error_type=type(error).__name__, error=str(error))
        result["cases"].append(row)
        failures += row["status"] != "passed"
        worker._write_json(directory / f"case-{case.case_id}.json", row)
        result["elapsed_seconds"] = time.perf_counter() - started
        worker._write_json(directory / "node-search.json", result)
        if failures >= protocol["limits"]["max_failed_histories"]:
            break
    result["unattempted_case_ids"] = [case.case_id for case, _ in cohort[len(result["cases"]) :]]
    result["status"] = "passed" if not failures and not result["unattempted_case_ids"] else "failed"
    result["worker"] = worker._process()
    worker._write_json(directory / "node-search.json", result)
    return result
