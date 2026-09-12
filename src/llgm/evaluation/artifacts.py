"""Auditable experiment files; no evaluator label is written to model traces."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from llgm.core.errors import ConfigurationError


def json_default(value: Any):
    """Convert supported dataclasses, paths, and collections for artifact JSON."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Unsupported artifact value: {type(value).__name__}")


def write_json(path: str | Path, value: Any) -> None:
    """Write stable, readable JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, values) -> None:
    """Replace a JSONL file with one serialized record per input value."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default) + "\n"
            )


def read_jsonl(path: str | Path) -> list[dict]:
    """Load JSONL records in order, ignoring blank lines."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class RunArtifacts:
    """Keep one run's manifest, append-only event streams, and final summary."""

    STREAMS = ("cases", "traces", "usage", "predictions", "judgments")

    def __init__(self, directory: str | Path, manifest: dict):
        """Initialize artifact files, refusing to overwrite a nonempty run directory."""
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ConfigurationError(
                "Run directory must be new or empty; existing artifacts will not be overwritten"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "manifest.json", manifest)
        for name in self.STREAMS:
            (self.directory / f"{name}.jsonl").touch()

    def append(self, stream: str, value: dict) -> None:
        """Append a record to a declared stream; reject unknown stream names."""
        if stream not in self.STREAMS:
            raise ConfigurationError(f"Unknown artifact stream: {stream}")
        with (self.directory / f"{stream}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, default=json_default) + "\n")

    def finish(self, metrics: dict, decision: str) -> None:
        """Write the final metrics and research decision alongside recorded events."""
        write_json(self.directory / "metrics.json", metrics)
        (self.directory / "decision.md").write_text(decision + "\n", encoding="utf-8")


def _code_provenance() -> dict:
    """Capture source hashes, runtime packages, and available Git state without network calls."""
    root = Path(__file__).resolve().parents[3]
    result = {
        "source_sha256": {},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
    }
    package = Path(__file__).resolve().parents[1]
    for source in sorted(package.rglob("*.py")):
        result["source_sha256"][str(source.relative_to(package))] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    for name in ("llgm", "openai", "anthropic", "transformers", "colbert-ai", "torch"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    for key, command in (
        ("git_commit", ["rev-parse", "HEAD"]),
        ("git_status", ["status", "--porcelain"]),
    ):
        try:
            result[key] = subprocess.run(
                ["git", "-C", str(root), *command],
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            result[key] = None
    return result
