"""Local artifact rejection checks that never dispatch a retrieval request."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from llgm.core.errors import ConfigurationError
from llgm.core.types import SourceSpan
from llgm.retrieval.base import SearchPassage, passage_to_dict

_spec = importlib.util.spec_from_file_location(
    "colbert_client_check",
    Path(__file__).resolve().parents[1] / "tools" / "colbert_client_check.py",
)
client_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client_check)


@pytest.fixture
def saved_run(tmp_path):
    """Write a parser fixture with no claim of model execution or retrieval quality."""
    passages = [
        SearchPassage(
            f"p-{number}",
            "source text",
            (SourceSpan(f"node-{number}", "turn", 0, 11),),
            {"labels": ["artifact parser fixture"]},
        )
        for number in range(40)
    ]
    baseline = {
        "query": "source",
        "k": 40,
        "hits": [
            {"passage_id": passage.passage_id, "score": 1.0, "rank": rank}
            for rank, passage in enumerate(passages, 1)
        ],
    }
    built = {
        "hits": baseline["hits"],
        "descriptor": {
            "configuration": {"checkpoint_sha256": "b" * 64, "repository_revision": "c" * 40}
        },
    }
    (tmp_path / "passages.jsonl").write_text(
        "".join(json.dumps(passage_to_dict(passage)) + "\n" for passage in passages)
    )
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))
    (tmp_path / "build.json").write_text(json.dumps(built))
    return tmp_path, built, baseline, passages


def test_load_preserves_saved_passages_and_baseline(saved_run):
    """Parser reconstruction retains source spans and nested metadata exactly."""
    directory, built, baseline, passages = saved_run
    assert client_check._load_run(directory) == (built, baseline, passages)


@pytest.mark.parametrize("filename", ["build.json", "baseline.json", "passages.jsonl"])
def test_missing_artifacts_never_dispatch(tmp_path, filename):
    """An incomplete download fails locally and retains the reason in client.json."""
    for name in ("build.json", "baseline.json", "passages.jsonl"):
        if name != filename:
            (tmp_path / name).write_text("{}")

    async def forbidden_rpc(*args):
        """Fail if local validation unexpectedly reaches a remote call."""
        raise AssertionError("No RPC should occur")

    with pytest.raises(ConfigurationError, match="Invalid saved ColBERT run"):
        asyncio.run(
            client_check.check_saved_index(
                tmp_path, describe_rpc=forbidden_rpc, search_rpc=forbidden_rpc
            )
        )
    result = json.loads((tmp_path / "client.json").read_text())
    assert result["status"] == "failed"
    assert result["error"]["type"] == "ConfigurationError"
    assert "events" not in result


@pytest.mark.parametrize("defect", ["duplicate", "score", "count", "rank", "changed_build"])
def test_corrupt_baseline_is_rejected(saved_run, defect):
    """Reject invalid or inconsistent stored ranks before a cloud request can start."""
    directory, _, baseline, _ = saved_run
    if defect == "duplicate":
        baseline["hits"][1]["passage_id"] = baseline["hits"][0]["passage_id"]
    elif defect == "score":
        baseline["hits"][0]["score"] = float("nan")
    elif defect == "count":
        baseline["hits"].pop()
    elif defect == "rank":
        baseline["hits"][0]["rank"] = True
    else:
        baseline["hits"][0]["score"] = 2.0
    (directory / "baseline.json").write_text(json.dumps(baseline))
    with pytest.raises(ConfigurationError, match="Invalid saved ColBERT run"):
        client_check._load_run(directory)
