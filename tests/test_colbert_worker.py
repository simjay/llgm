"""Durable worker artifact contracts without loading or substituting an encoder."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from llgm.core.errors import ConfigurationError
from llgm.core.types import SourceSpan
from llgm.retrieval.base import SearchPassage, corpus_fingerprint, passage_to_dict

_spec = importlib.util.spec_from_file_location(
    "colbert_worker", Path(__file__).resolve().parents[1] / "tools" / "colbert_worker.py"
)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


@pytest.fixture
def record(tmp_path, monkeypatch):
    """Create a structurally valid storage record containing no claimed retrieval result."""
    pins = worker.load_pins()
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    worker._write_json(root / "experiments" / "colbert_modal.json", pins)
    monkeypatch.setattr(worker, "ROOT", root)
    monkeypatch.setattr(worker, "ASSETS", tmp_path / "assets")
    monkeypatch.setattr(worker, "_CACHE", None)
    passages = [SearchPassage("p-one", "Source text", (SourceSpan("node", "turn", 0, 11),))]
    identity = {
        "pins": pins,
        "case_id": "case",
        "run_id": "test",
        "corpus_fingerprint": corpus_fingerprint(passages),
    }
    index_id = hashlib.sha256(worker._canonical(identity)).hexdigest()
    directory = worker._path("records", index_id)
    directory.mkdir(parents=True)
    (directory / "passages.jsonl").write_bytes(
        worker._canonical(passage_to_dict(passages[0])) + b"\n"
    )
    worker._write_json(directory / "baseline.json", {"query": "source", "k": 40, "hits": []})
    official = worker._path("indexes", index_id, "llgm-manifest.json")
    worker._write_json(official, {"storage_fixture": True})
    value = {
        "schema_version": 1,
        "status": "complete",
        "index_id": index_id,
        "identity": identity,
        "corpus_fingerprint": identity["corpus_fingerprint"],
        "descriptor": {
            "backend": "colbertv2_plaid",
            "corpus_fingerprint": identity["corpus_fingerprint"],
            "passage_count": 1,
            "configuration": {
                "checkpoint_sha256": pins["checkpoint"]["sha256"],
                "repository_revision": pins["repository"]["revision"],
            },
        },
        "checksums": {
            "passages.jsonl": worker._digest(directory / "passages.jsonl"),
            "baseline.json": worker._digest(directory / "baseline.json"),
            "llgm-manifest.json": worker._digest(official),
        },
    }
    worker._write_json(directory / "manifest.json", value)
    return index_id, directory, value, pins


@pytest.mark.parametrize(
    "value", ["../escape", "/tmp/escape", "a/b", "a\\b", "", "x" * 81, ".", "..", "a\x00b"]
)
def test_run_names_reject_paths(value):
    """Run identifiers cannot become paths or unbounded artifact names."""
    with pytest.raises(ConfigurationError):
        worker._name(value, "run_id")


@pytest.mark.parametrize("value", ["../" + "a" * 64, "A" * 64, "g" * 64, "a" * 63, "", None])
def test_index_ids_reject_noncanonical_digests(value):
    """Only lowercase SHA256 identifiers reach index storage."""
    with pytest.raises(ConfigurationError):
        worker.describe_index(value)


def test_volume_path_rejects_symlink_escape(tmp_path, monkeypatch):
    """An existing symlink cannot redirect a worker artifact outside its volume."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "records").symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(worker, "ASSETS", assets)
    with pytest.raises(ConfigurationError, match="escapes"):
        worker._path("records", "outside")


def test_describe_is_lightweight_and_preserves_canonical_passages(record):
    """Describing a completed storage record restores its identity without an encoder import."""
    index_id, _, value, _ = record
    before = set(sys.modules)
    result = worker.describe_index(index_id)
    assert result["index_id"] == index_id
    assert result["descriptor"] == value["descriptor"]
    assert not ({"torch", "colbert"} & (set(sys.modules) - before))
    passages = worker._passages(index_id, value)
    assert passages[0].refs == (SourceSpan("node", "turn", 0, 11),)
    assert passages[0].text == "Source text"


@pytest.mark.parametrize("name", ["passages.jsonl", "baseline.json", "llgm-manifest.json"])
def test_modified_artifacts_are_rejected(record, name):
    """Checksums detect changed passage, baseline and official manifest bytes before search."""
    index_id, directory, _, _ = record
    path = (
        worker._path("indexes", index_id, name)
        if name == "llgm-manifest.json"
        else directory / name
    )
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ConfigurationError, match="corrupt"):
        worker.describe_index(index_id)


@pytest.mark.parametrize("field", ["status", "index_id", "corpus_fingerprint", "schema_version"])
def test_record_identity_mismatch_is_rejected(record, field):
    """Path, completion, corpus and protocol identity cannot diverge silently."""
    index_id, directory, value, _ = record
    value[field] = "changed"
    worker._write_json(directory / "manifest.json", value)
    with pytest.raises(ConfigurationError, match="corrupt"):
        worker.describe_index(index_id)


def test_changed_pins_require_new_index(record):
    """Changing search configuration invalidates the old immutable run record."""
    index_id, _, _, pins = record
    changed = copy.deepcopy(pins)
    changed["configuration"]["nbits"] = 4
    worker._write_json(worker.ROOT / "experiments" / "colbert_modal.json", changed)
    with pytest.raises(ConfigurationError, match="corrupt"):
        worker.describe_index(index_id)


def test_incomplete_record_is_not_opened(record):
    """A interrupted build has no completed manifest and cannot masquerade as reusable."""
    index_id, directory, _, _ = record
    (directory / "manifest.json").unlink()
    worker._write_json(directory / "failure.json", {"status": "failed"})
    with pytest.raises(ConfigurationError, match="Missing or invalid"):
        worker.describe_index(index_id)


def test_changed_corpus_is_rejected_even_with_replaced_file_checksum(record):
    """Ordered passage fingerprints provide a second check on canonical content."""
    index_id, directory, value, _ = record
    path = directory / "passages.jsonl"
    passage = json.loads(path.read_text())
    passage["text"] = "Changed text"
    path.write_bytes(worker._canonical(passage) + b"\n")
    value["checksums"]["passages.jsonl"] = worker._digest(path)
    with pytest.raises(ConfigurationError, match="Corrupt canonical"):
        worker._passages(index_id, value)


@pytest.mark.parametrize(
    "query,k", [("", 40), ("  ", 1), (None, 1), ("q", 0), ("q", True), ("q", 1.2)]
)
def test_invalid_queries_fail_before_storage_or_gpu(query, k):
    """Invalid query inputs do not load an index or GPU dependency."""
    with pytest.raises(ConfigurationError, match="nonempty query"):
        worker.search_index("a" * 64, query, k)


def test_ranking_comparison_rejects_changed_identity_and_score():
    """Reopen verification detects changed identities and materially different scores."""
    expected = [{"passage_id": "p", "rank": 1, "score": 12.0}]
    worker._compare_hits([{"passage_id": "p", "rank": 1, "score": 12.000001}], expected)
    for actual in (
        [],
        [{"passage_id": "other", "rank": 1, "score": 12.0}],
        [{"passage_id": "p", "rank": 1, "score": 13.0}],
    ):
        with pytest.raises(ConfigurationError):
            worker._compare_hits(actual, expected)


def test_timing_summary_retains_samples():
    """The declared nearest-rank percentile remains auditable from its raw samples."""
    result = worker._timings([0.1, 0.5, 0.2, 0.4, 0.3])
    assert result["p50_seconds"] == 0.3
    assert result["p95_seconds"] == 0.5
    assert len(result["samples_seconds"]) == 5
