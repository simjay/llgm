"""Offline Modal job packaging, asset integrity and local subprocess contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(scope="module")
def job_module():
    """Import the installed SDK's image definitions with network connections prohibited."""
    modal = pytest.importorskip("modal", reason="Install llgm[modal] for job packaging checks")
    captured = []
    original_add_file = modal.Image.add_local_file

    def add_file(image, local_path, remote_path, **kwargs):
        """Record exactly which paths the real SDK attaches to the image definition."""
        captured.append((Path(local_path), remote_path))
        return original_add_file(image, local_path, remote_path, **kwargs)

    path = Path(__file__).resolve().parents[1] / "tools/colbert_modal.py"
    spec = importlib.util.spec_from_file_location("_llgm_colbert_modal_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with (
        patch("socket.socket.connect", side_effect=AssertionError("Unexpected network access")),
        patch.object(modal.Image, "add_local_file", add_file),
    ):
        spec.loader.exec_module(module)
    yield module, captured
    sys.modules.pop(spec.name, None)


def pin(data: bytes) -> dict:
    """Bind synthetic downloaded bytes to an exact length and digest."""
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_upload_allowlist_excludes_private_checkout_files(job_module):
    """The image uploads only library Python and its explicit runnable job inputs."""
    job, captured = job_module
    expected = {
        "tools/colbert_modal.py",
        "tools/colbert_worker.py",
        "tests/conftest.py",
        "tests/integration/test_live_colbert.py",
        "tests/fixtures/longmemeval-s.json",
        "experiments/colbert_modal.json",
        "experiments/colbert-requirements.txt",
        "pyproject.toml",
    }
    expected.update(str(path.relative_to(job.ROOT)) for path in job.ROOT.glob("src/**/*.py"))
    actual = set()
    for local, remote in captured:
        assert local.is_file()
        assert local.resolve().is_relative_to(job.ROOT.resolve())
        relative = local.relative_to(job.ROOT).as_posix()
        assert remote == f"/opt/llgm/{relative}"
        assert not (
            {"research", "agent-context", ".agents", ".git", ".env"} & set(Path(relative).parts)
        )
        actual.add(relative)
    assert actual == expected
    assert len(captured) == len(actual)


def test_download_publishes_verified_bytes_once(job_module, tmp_path, monkeypatch):
    """A complete valid stream is published atomically and a verified cache is reused offline."""
    job, _ = job_module
    data = b"checkpoint-data" * 1000
    request = Mock(return_value=io.BytesIO(data))
    monkeypatch.setattr(job, "urlopen", request)
    destination = tmp_path / "checkpoint" / "weights.bin"
    job.download_pinned("https://example.invalid/weights", destination, pin(data))
    assert destination.read_bytes() == data
    assert job.file_hash(destination) == pin(data)["sha256"]
    assert not list(destination.parent.glob("*.partial"))
    request.assert_called_once()
    job.download_pinned("https://example.invalid/weights", destination, pin(data))
    request.assert_called_once()


@pytest.mark.parametrize("data", [b"short", b"same-length-bad", b"too-long-and-not-correct"])
def test_bad_download_never_becomes_a_cached_asset(job_module, tmp_path, monkeypatch, data):
    """Length or digest mismatch removes temporary bytes and leaves no completed asset."""
    job, _ = job_module
    expected = pin(b"expected-value")
    monkeypatch.setattr(job, "urlopen", Mock(return_value=io.BytesIO(data)))
    destination = tmp_path / "weights.bin"
    with pytest.raises(ValueError, match="pin"):
        job.download_pinned("https://example.invalid/weights", destination, expected)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("data", [b"wrong-size", b"same-size!"])
def test_corrupt_cached_asset_is_preserved_and_rejected(job_module, tmp_path, monkeypatch, data):
    """A corrupt existing asset is not silently overwritten or trusted from its filename."""
    job, _ = job_module
    destination = tmp_path / "weights.bin"
    destination.write_bytes(data)
    request = Mock(side_effect=AssertionError("No repair download authorized"))
    monkeypatch.setattr(job, "urlopen", request)
    with pytest.raises(ValueError, match="Existing asset"):
        job.download_pinned("https://example.invalid/weights", destination, pin(b"good-data!"))
    assert destination.read_bytes() == data
    request.assert_not_called()


def test_interrupted_download_cleans_partial_bytes(job_module, tmp_path, monkeypatch):
    """A stream failure retains neither a partial cache nor a falsely complete artifact."""
    job, _ = job_module
    stream = Mock()
    stream.read.side_effect = [b"partial-data", OSError("connection interrupted")]
    context = Mock()
    context.__enter__ = Mock(return_value=stream)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(job, "urlopen", Mock(return_value=context))
    with pytest.raises(OSError, match="interrupted"):
        job.download_pinned(
            "https://example.invalid/weights", tmp_path / "weights.bin", pin(b"partial-data-more")
        )
    assert list(tmp_path.iterdir()) == []


def test_oversized_download_stops_without_consuming_the_rest(job_module, tmp_path, monkeypatch):
    """An oversized remote stream stops at the first chunk beyond its pinned length."""
    job, _ = job_module
    stream = Mock()
    stream.read.side_effect = [b"x" * 11, AssertionError("Read beyond pinned size")]
    context = Mock()
    context.__enter__ = Mock(return_value=stream)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(job, "urlopen", Mock(return_value=context))
    with pytest.raises(ValueError, match="pinned length"):
        job.download_pinned(
            "https://example.invalid/weights", tmp_path / "weights.bin", pin(b"x" * 10)
        )
    assert stream.read.call_count == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../escape", "/tmp/escape", "a/b", "a\\b", "x" * 81, "a\x00b"]
)
def test_run_labels_cannot_become_paths(job_module, tmp_path, monkeypatch, name):
    """Invalid run labels fail before creating any directories."""
    job, _ = job_module
    monkeypatch.setattr(job, "ASSETS", tmp_path)
    with pytest.raises(ValueError):
        job.run_directory(name)
    assert list(tmp_path.iterdir()) == []


def test_run_directory_is_local_to_the_configured_assets(job_module, tmp_path, monkeypatch):
    """A valid repeated run label resolves to the same directory within the test volume."""
    job, _ = job_module
    monkeypatch.setattr(job, "ASSETS", tmp_path)
    directory = job.run_directory("sanity_2026-09-11")
    assert directory == tmp_path / "runs" / "sanity_2026-09-11"
    assert directory.is_dir()
    assert job.run_directory("sanity_2026-09-11") == directory


def test_run_directory_rejects_existing_symlink_escape(job_module, tmp_path, monkeypatch):
    """A preexisting symlink cannot redirect run logs outside the asset volume."""
    job, _ = job_module
    assets = tmp_path / "assets"
    outside = tmp_path / "outside"
    assets.mkdir()
    outside.mkdir()
    (assets / "runs").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(job, "ASSETS", assets)
    with pytest.raises(ValueError, match="escapes"):
        job.run_directory("sanity")
    assert list(outside.iterdir()) == []


@pytest.fixture
def worker_script(job_module, tmp_path, monkeypatch):
    """Create a real local child script without importing any model or GPU dependency."""
    job, _ = job_module
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    script = root / "tools/colbert_worker.py"
    monkeypatch.setattr(job, "ROOT", root)
    monkeypatch.setenv("MODAL_TASK_ID", "test-container")
    return job, script, tmp_path / "result.json"


def test_worker_captures_output_and_augments_result(worker_script):
    """A real child returns structured results while preserving both output streams in its log."""
    job, script, output = worker_script
    script.write_text(
        '"""Emit a small subprocess result for the local transport test."""\n'
        "import json, sys\n"
        "from pathlib import Path\n"
        "print('worker stdout')\n"
        "print('worker stderr', file=sys.stderr)\n"
        "Path(sys.argv[-1]).write_text(json.dumps({'status': 'passed', 'arguments': sys.argv[1:-2]}))\n"
    )
    result = job.run_worker(["build", "--run-id", "sanity"], output, timeout=5)
    assert result["status"] == "passed"
    assert result["arguments"] == ["build", "--run-id", "sanity"]
    assert result["job_seconds"] >= 0
    assert result["container_id"] == "test-container"
    assert json.loads(output.read_text()) == result
    assert "worker stdout" in output.with_suffix(".log").read_text()
    assert "worker stderr" in output.with_suffix(".log").read_text()


def test_worker_nonzero_exit_retains_failure_log(worker_script, capsys):
    """A failed child exposes its retained log and cannot return stale success data."""
    job, script, output = worker_script
    script.write_text(
        '"""Fail the local child-process contract check."""\nimport sys\nprint("native failure", flush=True)\nsys.exit(7)\n'
    )
    with pytest.raises(RuntimeError, match="exited 7"):
        job.run_worker(["build"], output, timeout=5)
    assert "native failure" in output.with_suffix(".log").read_text()
    assert "native failure" in capsys.readouterr().out
    assert not output.exists()


def test_worker_timeout_stops_the_direct_child_and_preserves_log(worker_script):
    """The configured subprocess deadline interrupts a blocking local worker."""
    job, script, output = worker_script
    script.write_text(
        '"""Wait past the bounded local worker deadline."""\nimport time\nprint("started", flush=True)\ntime.sleep(30)\n'
    )
    with pytest.raises(subprocess.TimeoutExpired):
        job.run_worker(["build"], output, timeout=0.2)
    assert "started" in output.with_suffix(".log").read_text()
    assert not output.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Modal native jobs use Linux process groups")
def test_worker_timeout_also_stops_term_resistant_grandchild(worker_script, tmp_path):
    """A spawned compiler-like process cannot keep running after its parent reaches the deadline."""
    job, script, output = worker_script
    ready = tmp_path / "child-pid"
    survived = tmp_path / "survived"
    child = (
        "import os, signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(1.8)\n"
        f"Path({str(survived)!r}).touch()\n"
        "time.sleep(30)\n"
    )
    script.write_text(
        '"""Spawn a child that requires process-group cleanup."""\n'
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "time.sleep(30)\n"
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            job.run_worker(["build"], output, timeout=1)
        assert ready.exists(), "The descendant must start before checking timeout cleanup"
        time.sleep(1.1)
        assert not survived.exists()
    finally:
        if ready.exists():
            try:
                os.kill(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("prefix", ["", "/"])
def test_collect_run_preserves_only_requested_artifact_paths(
    job_module, tmp_path, monkeypatch, prefix
):
    """Artifact download accepts both Modal path forms and keeps the run's relative tree."""
    job, _ = job_module
    file_type = job.modal.volume.FileEntryType.FILE
    directory_type = job.modal.volume.FileEntryType.DIRECTORY
    paths = [
        f"{prefix}runs/sanity/environment.json",
        f"{prefix}runs/sanity/integration/summary.json",
    ]
    volume = SimpleNamespace(
        iterdir=Mock(
            return_value=[
                SimpleNamespace(type=directory_type, path=f"{prefix}runs/sanity/integration"),
                *(SimpleNamespace(type=file_type, path=path) for path in paths),
            ]
        ),
        read_file=Mock(return_value=iter([b'{"status":', b'"passed"}'])),
    )
    volume.read_file.side_effect = lambda path: iter([b'{"path":', json.dumps(path).encode(), b"}"])
    monkeypatch.setattr(job, "volume", volume)
    job.collect_run("sanity", tmp_path)
    volume.iterdir.assert_called_once_with("/runs/sanity", recursive=True)
    assert json.loads((tmp_path / "environment.json").read_text())["path"] == paths[0]
    assert json.loads((tmp_path / "integration/summary.json").read_text())["path"] == paths[1]


def test_runtime_manifest_hashes_source_and_excludes_credentials(job_module, tmp_path, monkeypatch):
    """Source changes alter provenance while unrelated environment values remain absent."""
    job, _ = job_module
    for relative, content in {
        "src/llgm/example.py": '"""A source identity fixture."""\n',
        "tools/colbert_worker.py": '"""A worker identity fixture."""\n',
        "experiments/colbert-requirements.txt": "pytest==8.3.3\n",
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(job.subprocess, "check_output", Mock(return_value="pytest==8.3.3\n"))
    monkeypatch.setattr(
        job.subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 1, stdout=""))
    )
    monkeypatch.setenv("MODAL_TASK_ID", "container")
    monkeypatch.setenv("UNRELATED_API_KEY", "not-in-the-manifest")
    first = job.runtime_manifest()
    assert first["installed_packages"] == ["pytest==8.3.3"]
    assert first["container_id"] == "container"
    assert first["billed_usd"] is None
    assert set(first["source_files"]) == {"src/llgm/example.py", "tools/colbert_worker.py"}
    assert "not-in-the-manifest" not in json.dumps(first)
    (tmp_path / "src/llgm/example.py").write_text('"""Changed source identity."""\n')
    assert job.runtime_manifest()["source_sha256"] != first["source_sha256"]
