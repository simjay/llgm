"""Query a deployed ColBERT index using canonical passages from a completed GPU test."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from llgm.retrieval.base import passage_from_dict
from llgm.retrieval.modal import ModalColBERTRetriever


async def query_saved_index(directory: Path, query: str | None = None) -> dict:
    """Verify the saved corpus and query its corresponding authenticated remote index."""
    built = json.loads((directory / "build.json").read_text())
    baseline = json.loads((directory / "baseline.json").read_text())
    passages = [
        passage_from_dict(json.loads(line))
        for line in (directory / "passages.jsonl").read_text().splitlines()
    ]
    configuration = built["descriptor"]["configuration"]
    retriever = await ModalColBERTRetriever.connect(
        passages,
        index_id=built["index_id"],
        expected_checkpoint_sha256=configuration["checkpoint_sha256"],
        expected_repository_revision=configuration["repository_revision"],
    )
    hits = await retriever.search(query or baseline["query"], baseline["k"])
    return {
        "index_id": built["index_id"],
        "passage_ids": [hit.passage.passage_id for hit in hits],
        "scores": [hit.score for hit in hits],
        "matches_saved_ranking": (
            [hit.passage.passage_id for hit in hits]
            == [item["passage_id"] for item in baseline["hits"]]
        )
        if query is None
        else None,
        "events": retriever.events,
    }


def main() -> None:
    """Print remote retrieval identities and timing for an explicit local run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--query")
    arguments = parser.parse_args()
    print(json.dumps(asyncio.run(query_saved_index(arguments.run_dir, arguments.query)), indent=2))


if __name__ == "__main__":
    main()
