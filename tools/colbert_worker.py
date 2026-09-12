"""Durable ColBERTv2/PLAID jobs for the pinned Modal experiment environment.

Importing this module does not load the encoder or require Modal. Completed
records are immutable and bind one corpus, checkpoint, implementation and run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import resource
import socket
import statistics
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llgm.core.errors import ConfigurationError
from llgm.evaluation.longmemeval import parse_longmemeval
from llgm.evaluation.scoring import evidence_coverage
from llgm.retrieval.base import (
    SearchHit,
    corpus_fingerprint,
    passage_from_dict,
    passage_to_dict,
)

ROOT = Path("/opt/llgm") if Path("/opt/llgm").is_dir() else Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("LLGM_COLBERT_ASSETS", "/assets"))
_LOCK = threading.RLock()
_CACHE: tuple[str, Any] | None = None


def _canonical(value: Any) -> bytes:
    """Encode finite JSON deterministically for durable content identities."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(path: Path) -> str:
    """Hash a file incrementally without loading model or dataset bytes into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _name(value: str, label: str) -> str:
    """Require a bounded single path component for run and dataset names."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ConfigurationError(
            f"{label} must be 1–80 letters, digits, dots, underscores or hyphens"
        )
    return value


def _index_id(value: str) -> str:
    """Reject index identifiers that are not canonical content digests."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ConfigurationError("index_id must be 64 lowercase hexadecimal characters")
    return value


def _path(*parts: str) -> Path:
    """Resolve asset paths while rejecting symlinks outside the configured volume."""
    root = ASSETS.resolve()
    result = root.joinpath(*parts).resolve()
    if not result.is_relative_to(root):
        raise ConfigurationError("Asset path escapes the configured volume")
    return result


def _read_json(path: Path) -> dict:
    """Require an object-shaped JSON artifact and give corruption an explicit error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Missing or invalid JSON artifact: {path}") from exc


def _write_json(path: Path, value: dict) -> None:
    """Publish a complete JSON artifact by atomic replacement on its filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f".{os.getpid()}.tmp")
    pending.write_bytes(_canonical(value) + b"\n")
    pending.replace(path)


def load_pins() -> dict:
    """Load the checked-in protocol without importing the optional GPU stack."""
    pins = _read_json(ROOT / "experiments" / "colbert_modal.json")
    if type(pins.get("schema_version")) is not int or pins["schema_version"] != 1:
        raise ConfigurationError("Unsupported ColBERT experiment pin schema")
    _name(pins["dataset"]["file"], "dataset filename")
    return pins


def _case(case_id: str, pins: dict):
    """Verify release bytes before parsing exactly one full history and its separate gold."""
    _name(case_id, "case_id")
    dataset = _path("datasets", pins["dataset"]["file"])
    if (
        dataset.stat().st_size != pins["dataset"]["bytes"]
        or _digest(dataset) != pins["dataset"]["sha256"]
    ):
        raise ConfigurationError("LongMemEval dataset does not match the pinned release")
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    selected = [row for row in rows if str(row.get("question_id")) == case_id]
    if len(selected) != 1:
        raise ConfigurationError(f"Expected exactly one LongMemEval case {case_id!r}")
    cases, gold = parse_longmemeval(selected)
    return cases[0], gold[case_id]


def _configuration(index_id: str, pins: dict):
    """Construct the official adapter's explicit local configuration from frozen pins."""
    from llgm.retrieval.colbert import ColBERTConfig

    return ColBERTConfig(
        checkpoint_path=_path("checkpoint"),
        checkpoint_sha256=pins["checkpoint"]["sha256"],
        repository_revision=pins["repository"]["revision"],
        index_root=_path("indexes"),
        index_name=_index_id(index_id),
        **pins["configuration"],
    )


def _process() -> dict:
    """Record worker identity and peak resident memory without exposing credentials."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
    }


def _hits(hits: list[SearchHit]) -> list[dict]:
    """Serialize ranking values while keeping passage content in its canonical artifact."""
    return [
        {"passage_id": hit.passage.passage_id, "score": hit.score, "rank": hit.rank} for hit in hits
    ]


def _compare_hits(actual: list[dict], expected: list[dict]) -> None:
    """Require identical ranking identities and numerically equivalent floating scores."""
    if len(actual) != len(expected):
        raise ConfigurationError("Reopened index returned a different number of hits")
    for left, right in zip(actual, expected):
        if (
            left["passage_id"] != right["passage_id"]
            or left["rank"] != right["rank"]
            or not math.isclose(left["score"], right["score"], rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise ConfigurationError("Reopened index changed passage IDs, ranks or scores")


def _record(index_id: str) -> dict:
    """Validate immutable record identities and artifact checksums before index access."""
    index_id = _index_id(index_id)
    directory = _path("records", index_id)
    record = _read_json(directory / "manifest.json")
    try:
        identity = record["identity"]
        descriptor = record["descriptor"]
        pins = load_pins()
        if (
            type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or record["status"] != "complete"
            or record["index_id"] != index_id
            or hashlib.sha256(_canonical(identity)).hexdigest() != index_id
            or identity["pins"] != pins
            or record["corpus_fingerprint"] != identity["corpus_fingerprint"]
            or descriptor["corpus_fingerprint"] != identity["corpus_fingerprint"]
            or descriptor["backend"] != "colbertv2_plaid"
            or descriptor["configuration"]["checkpoint_sha256"] != pins["checkpoint"]["sha256"]
            or descriptor["configuration"]["repository_revision"] != pins["repository"]["revision"]
        ):
            raise ValueError("Record identity differs from its path, corpus or current pins")
        for name in ("passages.jsonl", "baseline.json"):
            if _digest(directory / name) != record["checksums"][name]:
                raise ValueError(f"Corrupt {name}")
        if (
            _digest(_path("indexes", index_id, "llgm-manifest.json"))
            != record["checksums"]["llgm-manifest.json"]
        ):
            raise ValueError("Corrupt official index manifest")
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise ConfigurationError(
            f"Incompatible or corrupt ColBERT record {index_id}: {exc}"
        ) from exc
    return record


def _passages(index_id: str, record: dict) -> list:
    """Restore canonical references and require their complete ordered corpus digest."""
    try:
        values = (
            _path("records", index_id, "passages.jsonl").read_text(encoding="utf-8").splitlines()
        )
        passages = [passage_from_dict(json.loads(value)) for value in values]
        if (
            corpus_fingerprint(passages) != record["corpus_fingerprint"]
            or len(passages) != record["descriptor"]["passage_count"]
            or len({passage.passage_id for passage in passages}) != len(passages)
        ):
            raise ValueError("Passage corpus or count differs from its manifest")
        return passages
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ConfigurationError("Corrupt canonical passage collection") from exc


def describe_index(index_id: str) -> dict:
    """Return a verified protocol descriptor without importing torch or ColBERT."""
    record = _record(index_id)
    return {
        key: record[key]
        for key in ("schema_version", "index_id", "corpus_fingerprint", "descriptor")
    }


def _open(index_id: str, record: dict):
    """Reuse one process-local encoder after every request validates durable identity."""
    global _CACHE
    if _CACHE is None or _CACHE[0] != index_id:
        from llgm.retrieval.colbert import ColBERTRetriever

        _CACHE = None
        retriever = ColBERTRetriever.open(
            _passages(index_id, record), config=_configuration(index_id, record["identity"]["pins"])
        )
        _CACHE = (index_id, retriever)
    return _CACHE[1]


def search_index(index_id: str, query: str, k: int = 40) -> dict:
    """Search the pinned index with serialized native operations and explicit server timing."""
    if not isinstance(query, str) or not query.strip() or type(k) is not int or k < 1:
        raise ConfigurationError("Require a nonempty query and a positive integer k")
    with _LOCK:
        record = _record(index_id)
        started = time.perf_counter()
        retriever = _open(index_id, record)
        open_seconds = time.perf_counter() - started
        started = time.perf_counter()
        hits = asyncio.run(retriever.search(query, k))
        return {
            "schema_version": 1,
            "index_id": index_id,
            "corpus_fingerprint": record["corpus_fingerprint"],
            "hits": _hits(hits),
            "search_seconds": time.perf_counter() - started,
            "open_seconds": open_seconds,
        }


def build_case(case_id: str, run_id: str) -> dict:
    """Build and atomically publish a real full-history index, or verify an exact completed reuse."""
    pins = load_pins()
    case, gold = _case(case_id, pins)
    return build_history(case, gold, run_id, pins)


def build_history(case, gold, run_id: str, pins: dict) -> dict:
    """Index supplied source text; evaluator labels are used only in the retained report."""
    from llgm.retrieval.colbert import ColBERTRetriever
    from llgm.retrieval.passages import split_nodes
    from llgm.retrieval.tokenizers import ColBERTTokenizer

    global _CACHE
    _name(run_id, "run_id")
    started = time.perf_counter()
    case_id = _name(case.case_id, "case_id")
    tokenizer = ColBERTTokenizer(_path("checkpoint"), revision=pins["checkpoint"]["revision"])
    passages = split_nodes(
        case.sources, tokenizer, window=pins["configuration"]["doc_maxlen"], overlap=32
    )
    if len(passages) < 40:
        raise ConfigurationError("The sanity protocol requires at least 40 real passages")
    identity = {
        "pins": pins,
        "case_id": case_id,
        "run_id": run_id,
        "corpus_fingerprint": corpus_fingerprint(passages),
    }
    index_id = hashlib.sha256(_canonical(identity)).hexdigest()
    directory = _path("records", index_id)
    with _LOCK:
        if directory.exists():
            record = _record(index_id)
            response = search_index(index_id, case.question, 40)
            baseline = _read_json(directory / "baseline.json")
            _compare_hits(response["hits"], baseline["hits"])
            return {
                **describe_index(index_id),
                **record["build"],
                "hits": response["hits"],
                "reused": True,
            }
        if _path("indexes", index_id).exists():
            raise ConfigurationError(
                "Refusing an index without a completed worker record. Use a new run_id"
            )
        directory.mkdir(parents=True, exist_ok=False)
        try:
            _CACHE = None
            (directory / "passages.jsonl").write_bytes(
                b"".join(_canonical(passage_to_dict(passage)) + b"\n" for passage in passages)
            )
            build_started = time.perf_counter()
            retriever = ColBERTRetriever.build(
                passages, config=_configuration(index_id, pins), tokenizer=tokenizer
            )
            build_seconds = time.perf_counter() - build_started
            _CACHE = (index_id, retriever)
            search_started = time.perf_counter()
            hits = asyncio.run(retriever.search(case.question, 40))
            cold_query_seconds = time.perf_counter() - search_started
            serialized = _hits(hits)
            if len(hits) != 40 or len({hit.passage.passage_id for hit in hits}) != 40:
                raise ConfigurationError(
                    "Real sanity retrieval must return 40 distinct canonical passages"
                )
            _write_json(
                directory / "baseline.json", {"query": case.question, "k": 40, "hits": serialized}
            )
            build = {
                "case_id": case_id,
                "run_id": run_id,
                "passage_count": len(passages),
                "source_count": len(case.sources),
                "build_seconds": build_seconds,
                "cold_query_seconds": cold_query_seconds,
                "index_bytes": sum(
                    path.stat().st_size
                    for path in _path("indexes", index_id).rglob("*")
                    if path.is_file()
                ),
                "coverage": evidence_coverage(
                    [ref for hit in hits for ref in hit.passage.refs], asdict(gold)
                ),
                "wall_seconds": time.perf_counter() - started,
                "worker": _process(),
                "cost_usd": None,
                "hosted_generation_requests": 0,
            }
            record = {
                "schema_version": 1,
                "status": "complete",
                "index_id": index_id,
                "identity": identity,
                "corpus_fingerprint": identity["corpus_fingerprint"],
                "descriptor": retriever.descriptor(),
                "checksums": {
                    "passages.jsonl": _digest(directory / "passages.jsonl"),
                    "baseline.json": _digest(directory / "baseline.json"),
                    "llgm-manifest.json": _digest(_path("indexes", index_id, "llgm-manifest.json")),
                },
                "build": build,
            }
            _write_json(directory / "manifest.json", record)
            return {**describe_index(index_id), **build, "hits": serialized, "reused": False}
        except Exception as exc:
            _write_json(
                directory / "failure.json",
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "worker": _process(),
                },
            )
            raise


def reopen_index(index_id: str) -> dict:
    """Open a persisted index anew and compare its top 40 with the original measured ranking."""
    global _CACHE
    with _LOCK:
        _CACHE = None
        record = _record(index_id)
        baseline = _read_json(_path("records", index_id, "baseline.json"))
        response = search_index(index_id, baseline["query"], baseline["k"])
        _compare_hits(response["hits"], baseline["hits"])
        return {
            **response,
            "status": "passed",
            "worker": _process(),
            "build_worker": record["build"]["worker"],
            "score_tolerance": {"relative": 1e-6, "absolute": 1e-6},
        }


def _timings(samples: list[float]) -> dict:
    """Summarize retained warm repetitions using median and nearest-rank p95."""
    return {
        "samples_seconds": samples,
        "p50_seconds": statistics.median(samples),
        "p95_seconds": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
        "p95_method": "nearest-rank",
    }


def benchmark(run_id: str) -> dict:
    """Measure raw BM25 and PLAID on identical real passages, retaining every case failure."""
    from llgm.retrieval.bm25 import SQLiteBM25Retriever

    _name(run_id, "run_id")
    pins = load_pins()
    rows = []
    for case_id in pins["benchmark_case_ids"]:
        row = {"case_id": case_id, "status": "failed"}
        try:
            built = build_case(case_id, run_id)
            index_id = built["index_id"]
            record = _record(index_id)
            passages = _passages(index_id, record)
            case, gold = _case(case_id, pins)
            colbert_samples = []
            for _ in range(pins["benchmark_repeats"]):
                response = search_index(index_id, case.question, 40)
                _compare_hits(response["hits"], built["hits"])
                colbert_samples.append(response["search_seconds"])
            bm25_path = _path("records", index_id, f"bm25-{time.time_ns()}.sqlite3")
            started = time.perf_counter()
            bm25 = SQLiteBM25Retriever.from_passages(passages, bm25_path)
            bm25_build_seconds = time.perf_counter() - started
            try:
                started = time.perf_counter()
                first = asyncio.run(bm25.search(case.question, 40))
                bm25_cold_seconds = time.perf_counter() - started
                bm25_samples = []
                for _ in range(pins["benchmark_repeats"]):
                    started = time.perf_counter()
                    hits = asyncio.run(bm25.search(case.question, 40))
                    bm25_samples.append(time.perf_counter() - started)
                    _compare_hits(_hits(hits), _hits(first))
                bm25_result = {
                    "descriptor": bm25.descriptor(),
                    "build_seconds": bm25_build_seconds,
                    "cold_query_seconds": bm25_cold_seconds,
                    "index_bytes": bm25_path.stat().st_size,
                    "hits": _hits(first),
                    "coverage": evidence_coverage(
                        [ref for hit in first for ref in hit.passage.refs], asdict(gold)
                    ),
                    "warm_query": _timings(bm25_samples),
                }
            finally:
                bm25.close()
            row.update(
                status="passed",
                index_id=index_id,
                corpus_fingerprint=built["corpus_fingerprint"],
                colbert={**built, "warm_query": _timings(colbert_samples)},
                bm25=bm25_result,
            )
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error=str(exc))
        rows.append(row)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
        "pins": pins,
        "cases": rows,
        "measurement_scope": "Three exposed LongMemEval histories, original queries, top 40, raw retrieval. Diagnostic replication, not independent benchmark evidence. Warm server search excludes dispatch and index-open time. No hosted model calls.",
        "worker": _process(),
        "cost_usd": None,
    }


def main() -> int:
    """Write a machine-readable job result, including failures, and set the process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "reopen", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "reopen":
            command.add_argument("--index-id", required=True)
        else:
            command.add_argument("--run-id", required=True)
        if name == "build":
            command.add_argument("--case-id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build_case(args.case_id, args.run_id)
        elif args.command == "reopen":
            result = reopen_index(args.index_id)
        else:
            result = benchmark(args.run_id)
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "worker": _process(),
        }
    _write_json(args.output, result)
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
