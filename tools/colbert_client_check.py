"""Validate a local client against an existing index through authenticated Modal RPCs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from llgm.core.errors import ConfigurationError
from llgm.retrieval.base import SearchPassage, passage_from_dict
from llgm.retrieval.modal import ModalColBERTRetriever, SearchRPC

DescribeRPC = Callable[[str], Awaitable[dict[str, Any]]]


def _load_run(directory: Path) -> tuple[dict, dict, list[SearchPassage]]:
    """Require the original top-40 baseline and canonical passages before dispatch."""
    try:
        built = json.loads((directory / "build.json").read_text(encoding="utf-8"))
        baseline = json.loads((directory / "baseline.json").read_text(encoding="utf-8"))
        passages = [
            passage_from_dict(json.loads(line))
            for line in (directory / "passages.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if (
            not isinstance(built, dict)
            or not isinstance(baseline, dict)
            or not isinstance(baseline["query"], str)
            or not baseline["query"].strip()
            or type(baseline["k"]) is not int
            or baseline["k"] != 40
            or not isinstance(baseline["hits"], list)
            or len(baseline["hits"]) != 40
        ):
            raise ValueError("Require a saved query and exactly 40 baseline hits")
        known = {passage.passage_id for passage in passages}
        seen = set()
        for rank, hit in enumerate(baseline["hits"], 1):
            if (
                hit["passage_id"] not in known
                or hit["passage_id"] in seen
                or type(hit["rank"]) is not int
                or hit["rank"] != rank
                or type(hit["score"]) not in (int, float)
                or not math.isfinite(hit["score"])
            ):
                raise ValueError("Saved baseline contains invalid IDs, ranks or scores")
            seen.add(hit["passage_id"])
        if built["hits"] != baseline["hits"]:
            raise ValueError("Baseline differs from the retained build result")
        built["descriptor"]["configuration"]["checkpoint_sha256"]
        built["descriptor"]["configuration"]["repository_revision"]
        return built, baseline, passages
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ConfigurationError(f"Invalid saved ColBERT run: {exc}") from exc


async def check_saved_index(
    directory: Path,
    *,
    describe_rpc: DescribeRPC,
    search_rpc: SearchRPC,
) -> dict[str, Any]:
    """Verify real RPC ranking and local evidence, saving client.json even on failure.

    Pass ``describe_index.remote.aio`` and ``search_index.remote.aio`` from the
    active Modal app. This checks the authenticated SDK transport without a
    permanent deployment. It neither builds an index nor substitutes retrieval.
    """
    started = time.perf_counter()
    retriever = None
    result: dict[str, Any] = {"status": "failed", "cost_usd": None}
    try:
        built, baseline, passages = _load_run(directory)
        configuration = built["descriptor"]["configuration"]
        arguments = {
            "index_id": built["index_id"],
            "search_rpc": search_rpc,
            "expected_checkpoint_sha256": configuration["checkpoint_sha256"],
            "expected_repository_revision": configuration["repository_revision"],
        }
        # Validate the saved corpus and provenance before starting paid work.
        ModalColBERTRetriever(passages, index_metadata=built, **arguments)
        result["index_id"] = built["index_id"]
        describe_started = time.perf_counter()
        metadata = await describe_rpc(built["index_id"])
        result["describe_elapsed_seconds"] = time.perf_counter() - describe_started
        retriever = ModalColBERTRetriever(passages, index_metadata=metadata, **arguments)
        hits = await retriever.search(baseline["query"], baseline["k"])
        result["hits"] = [
            {"passage_id": hit.passage.passage_id, "rank": hit.rank, "score": hit.score}
            for hit in hits
        ]
        result["same_passage_order"] = [hit.passage.passage_id for hit in hits] == [
            hit["passage_id"] for hit in baseline["hits"]
        ]
        result["scores_match"] = len(hits) == 40 and all(
            math.isclose(hit.score, saved["score"], rel_tol=1e-6, abs_tol=1e-6)
            for hit, saved in zip(hits, baseline["hits"])
        )
        originals = {passage.passage_id: passage for passage in passages}
        result["canonical_evidence_matches"] = len(hits) == 40 and all(
            hit.passage == originals[hit.passage.passage_id] for hit in hits
        )
        result["score_tolerance"] = {"relative": 1e-6, "absolute": 1e-6}
        if not all(
            result[key]
            for key in ("same_passage_order", "scores_match", "canonical_evidence_matches")
        ):
            raise ConfigurationError("Remote client result differs from saved ranking or evidence")
        result["status"] = "passed"
        return result
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        if retriever is not None:
            result["descriptor"] = retriever.descriptor()
            result["events"] = retriever.events
        if directory.is_dir():
            (directory / "client.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
