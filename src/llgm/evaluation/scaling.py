"""Local indexing measurements with explicit cold, reuse, update and restart phases."""

from __future__ import annotations

import asyncio
import json
import time
import tracemalloc
from pathlib import Path

from llgm.core.errors import ConfigurationError
from llgm.core.types import Conversation, reference_to_dict
from llgm.evaluation.artifacts import write_json
from llgm.evaluation.runner import _code_provenance
from llgm.memory.evidence import Evidence
from llgm.memory.workspace import Workspace
from llgm.storage.blobs import LocalBlobStore


class _CountedBlobs:
    """Count source reads at the actual blob interface without retaining payload copies."""

    def __init__(self, directory):
        """Wrap the local content-addressed store used by the measured workspace."""
        self.store = LocalBlobStore(directory)
        self.gets = 0
        self.bytes_read = 0

    def put(self, data):
        """Delegate durable publication unchanged."""
        return self.store.put(data)

    def get(self, digest):
        """Charge verified source bytes returned by one blob read."""
        data = self.store.get(digest)
        self.gets += 1
        self.bytes_read += len(data)
        return data


def _conversation(number, chars, *, updated=False):
    """Generate deterministic distinct source bytes, giving additional evidence its own identity."""
    prefix = f"Record {number:06d} unique{number:06d} release evidence version {'two' if updated else 'one'}. "
    text = (
        prefix
        + ("measured local corpus content " * (chars // 20 + 1))[: max(0, chars - len(prefix))]
    )
    return Conversation.from_turns(
        [{"turn_id": "t", "role": "user", "text": text}],
        node_id=f"node-{number:06d}" + ("-update" if updated else ""),
    )


async def _measure_open(workspace, blobs, query, *, repeats=3):
    """Measure preparation Python allocation separately from indexed search and one canonical read."""
    before_gets, before_bytes = blobs.gets, blobs.bytes_read
    if tracemalloc.is_tracing():
        raise ConfigurationError("Scaling probe requires ownership of Python allocation tracing")
    tracemalloc.start()
    evidence = None
    try:
        started = time.perf_counter()
        evidence = await Evidence.open(workspace)
        seconds = time.perf_counter() - started
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    try:
        result = {
            "preparation_seconds": seconds,
            "preparation_python_current_bytes": current,
            "preparation_python_peak_bytes": peak,
            "preparation_blob_gets": blobs.gets - before_gets,
            "preparation_blob_bytes": blobs.bytes_read - before_bytes,
            "index": dict(evidence.preparation),
            "retrieval": evidence.descriptor(),
        }
        latencies = []
        hits = []
        for _ in range(repeats):
            started = time.perf_counter()
            hits = await evidence.search(query, 10)
            latencies.append(time.perf_counter() - started)
        result["query_seconds"] = latencies
        result["hits"] = [
            {"references": [reference_to_dict(ref) for ref in hit.passage.refs], "score": hit.score}
            for hit in hits
        ]
        result["search_blob_gets"] = blobs.gets - before_gets - result["preparation_blob_gets"]
        before_read = blobs.gets
        if hits:
            await evidence.read(hits[0].passage.refs[0])
        result["canonical_read_blob_gets"] = blobs.gets - before_read
        return result
    finally:
        await evidence.close()


async def run_scaling_probe(
    output: str | Path, *, sizes=(100, 1000, 10000), source_chars=256, repeats=3
) -> dict:
    """Measure independent cold corpora and their unchanged, one-update and restarted index phases."""
    sizes = list(sizes)
    if (
        not sizes
        or any(type(size) is not int or size < 1 for size in sizes)
        or len(sizes) != len(set(sizes))
    ):
        raise ConfigurationError("Probe sizes must be distinct positive integers")
    if (
        type(source_chars) is not int
        or source_chars < 128
        or type(repeats) is not int
        or repeats < 1
    ):
        raise ConfigurationError("source_chars must be at least 128 and repeats must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ConfigurationError("Scaling output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "local-index-scaling-diagnostic",
        "schema_version": 2,
        "sizes": sizes,
        "source_chars": source_chars,
        "query_repeats": repeats,
        "code": _code_provenance(),
        "model_calls": 0,
        "benchmark_result": False,
        "conditions": {
            "corpus": "independent deterministic workspace per size; one new immutable source after warm reuse",
            "memory": "Python traced allocations during open only; excludes SQLite/native allocations and is not RSS",
            "timing": "wall times on the current host; not isolated from concurrent work",
            "search": "literal unique-source term, top10; a functional retrieval probe, not answer quality",
            "storage": "durable local SQLite and content-addressed blobs; source ingest measured separately",
        },
    }
    write_json(output / "manifest.json", manifest)
    summary = {"status": "running", "model_calls": 0, "benchmark_result": False, "measurements": []}
    write_json(output / "summary.json", summary)
    for size in sizes:
        directory = output / f"n-{size}"
        blobs = _CountedBlobs(directory / "blobs")
        query = f"unique{size - 1:06d}"
        async with Workspace.open(directory, blob_store=blobs) as workspace:
            started = time.perf_counter()
            for number in range(size):
                await workspace.ingest(_conversation(number, source_chars))
            measurement = {
                "source_count": size,
                "ingest_seconds": time.perf_counter() - started,
                "source_text_chars": source_chars * size,
                "phases": {},
            }
            phases = measurement["phases"]
            phases["cold"] = await _measure_open(workspace, blobs, query, repeats=repeats)
            phases["warm"] = await _measure_open(workspace, blobs, query, repeats=repeats)
            await workspace.ingest(_conversation(size - 1, source_chars, updated=True))
            phases["one_update"] = await _measure_open(workspace, blobs, query, repeats=repeats)
        async with Workspace.open(directory, blob_store=blobs) as workspace:
            phases["restart"] = await _measure_open(workspace, blobs, query, repeats=repeats)
        measurement["restart_ranking_unchanged"] = (
            phases["one_update"]["hits"] == phases["restart"]["hits"]
        )
        measurement["storage_bytes"] = sum(
            path.stat().st_size for path in directory.rglob("*") if path.is_file()
        )
        summary["measurements"].append(measurement)
        write_json(output / "summary.json", summary)
    summary["status"] = "completed"
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    """Run a local-only scaling probe with explicit output and corpus sizes."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--source-chars", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args(argv)
    result = asyncio.run(
        run_scaling_probe(
            arguments.output,
            sizes=arguments.sizes,
            source_chars=arguments.source_chars,
            repeats=arguments.repeats,
        )
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_counts": [row["source_count"] for row in result["measurements"]],
                "output": str(arguments.output.resolve()),
                "model_calls": 0,
            },
            indent=2,
        )
    )
    return 0
