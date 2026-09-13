"""Optional real-data fixtures and explicit gates; never download or infer consent."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import uuid
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

from llgm.evaluation.artifacts import write_json
from llgm.evaluation.longmemeval import EvaluationCase, GoldRecord, parse_longmemeval

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    """Hash the complete local file without retaining its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_env(name: str) -> str:
    """Require a named setting without exposing its value in failure output."""
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.fail(f"Opted-in integration test requires {name}", pytrace=False)
    return value


def require_opt_in(flag: str) -> None:
    """Skip disabled external work and reject malformed opt-in values."""
    value = os.environ.get(flag, "0")
    if value == "0":
        pytest.skip(f"Set {flag}=1 to enable this explicitly opted-in test")
    if value != "1":
        pytest.fail(f"{flag} must be exactly 0 or 1", pytrace=False)


def pinned_model(name: str) -> str:
    """Require a declared model identity without a moving latest alias."""
    model = required_env(name)
    if "latest" in model.lower():
        pytest.fail(f"{name} must identify an explicit model, not latest", pytrace=False)
    return model


@dataclass(frozen=True)
class LocalLongMemEval:
    """Hold verified diagnostic inputs separately from evaluator-only labels."""

    path: Path
    manifest: dict
    cases: dict[str, EvaluationCase]
    gold: dict[str, GoldRecord]


@pytest.fixture(scope="session")
def longmemeval_manifest():
    """Load the committed release checksum and fixed diagnostic case IDs."""
    return json.loads((PROJECT_ROOT / "tests/fixtures/longmemeval-s.json").read_text())


@pytest.fixture(scope="session")
def longmemeval(longmemeval_manifest):
    """Verify the complete local release before selecting a few diagnostic cases."""
    configured = os.environ.get("LLGM_TEST_LONGMEMEVAL_PATH")
    path = (
        Path(configured)
        if configured
        else PROJECT_ROOT / "data/longmemeval" / longmemeval_manifest["file"]
    )
    path = path.expanduser().resolve()
    explicitly_required = configured is not None or os.environ.get("LLGM_TEST_COLBERT") == "1"
    if not path.is_file():
        if explicitly_required:
            pytest.fail(
                "Configured integration test requires the local pinned LongMemEval-S file",
                pytrace=False,
            )
        pytest.skip("Optional pinned LongMemEval-S data is absent; tests never download it")
    assert path.stat().st_size == longmemeval_manifest["bytes"], (
        "LongMemEval byte-size pin mismatch"
    )
    assert file_sha256(path) == longmemeval_manifest["sha256"], "LongMemEval SHA256 pin mismatch"
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) == longmemeval_manifest["case_count"]
    selected = set(longmemeval_manifest["case_ids"])
    cases, gold = parse_longmemeval([row for row in rows if str(row["question_id"]) in selected])
    assert {case.case_id for case in cases} == selected
    return LocalLongMemEval(
        path, dict(longmemeval_manifest), {case.case_id: case for case in cases}, gold
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Retain pytest outcomes and exception types for sanitized artifact summaries."""
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"integration_report_{report.when}", report)
    if call.excinfo is not None and report.failed:
        # Exception messages can contain provider bodies or credentials.
        item.integration_error_type = call.excinfo.typename


@pytest.fixture(scope="session")
def integration_artifact_root():
    """Allocate a unique ignored directory so earlier integration results survive."""
    path = PROJECT_ROOT / "runs/integration" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture
def integration_record(request, integration_artifact_root):
    """Record test provenance, attempted work and outcome without exception bodies."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.node.nodeid).strip("-")
    directory = integration_artifact_root / name
    directory.mkdir()
    packages = {}
    for name in ("llgm", "pytest", "openai", "anthropic", "colbert-ai", "torch", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    record = {
        "schema_version": 1,
        "test_id": request.node.nodeid,
        "scope": "integration sanity; no benchmark performance claim",
        "benchmark_result": False,
        "python": platform.python_version(),
        "packages": packages,
        "operations": [],
        "models": [],
        "errors": [],
    }
    yield record, directory
    report = getattr(request.node, "integration_report_call", None)
    record["status"] = report.outcome if report is not None else "setup_failed_or_skipped"
    error_type = getattr(request.node, "integration_error_type", None)
    if error_type:
        record["errors"].append(
            {
                "type": error_type,
                "detail": "See local pytest output; exception text is not persisted",
            }
        )
    write_json(directory / "summary.json", record)


@pytest.fixture
def live_openai():
    """Require explicit consent, credentials and an OpenAI model identity."""
    require_opt_in("LLGM_TEST_OPENAI")
    required_env("OPENAI_API_KEY")
    return pinned_model("LLGM_TEST_OPENAI_MODEL")


@pytest.fixture
def live_anthropic():
    """Require explicit consent, credentials and an Anthropic model identity."""
    require_opt_in("LLGM_TEST_ANTHROPIC")
    required_env("ANTHROPIC_API_KEY")
    return pinned_model("LLGM_TEST_ANTHROPIC_MODEL")


@pytest.fixture
def live_embedding():
    """Require explicit consent and the hosted embedding model identity."""
    require_opt_in("LLGM_TEST_EMBEDDING")
    required_env("OPENAI_API_KEY")
    return pinned_model("LLGM_TEST_EMBEDDING_MODEL")


@pytest.fixture
def live_colbert():
    """Require local checkpoint and code pins before optional ColBERT work."""
    require_opt_in("LLGM_TEST_COLBERT")
    return {
        "checkpoint_path": required_env("LLGM_TEST_COLBERT_CHECKPOINT"),
        "checkpoint_sha256": required_env("LLGM_TEST_COLBERT_CHECKPOINT_SHA256"),
        "repository_revision": required_env("LLGM_TEST_COLBERT_REVISION"),
        "gpus": int(os.environ.get("LLGM_TEST_COLBERT_GPUS", "0")),
    }
