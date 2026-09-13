"""Run pinned ColBERTv2/PLAID jobs and authenticated retrieval on Modal."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import modal

ROOT = (
    Path("/opt/llgm")
    if Path("/opt/llgm/experiments").is_dir()
    else Path(__file__).resolve().parents[1]
)
PINS = json.loads((ROOT / "experiments/colbert_modal.json").read_text())
ASSETS = Path("/assets")
app = modal.App("llgm-colbert")
volume = modal.Volume.from_name("llgm-colbert-assets", create_if_missing=True)

image = (
    modal.Image.from_registry(PINS["cuda_image"], add_python="3.11")
    .run_commands("sed -i 's|http://|https://|g' /etc/apt/sources.list")
    .apt_install("git", "g++")
    .pip_install_from_requirements(str(ROOT / "experiments/colbert-requirements.txt"))
    .run_commands(
        "python -m pip install --no-deps "
        f"git+{PINS['repository']['url']}.git@{PINS['repository']['revision']}"
    )
    .env(
        {
            "PYTHONPATH": "/opt/llgm/src:/opt/llgm/tools",
            "LLGM_COLBERT_ASSETS": "/assets",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "MAX_JOBS": "4",
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "TORCH_EXTENSIONS_DIR": "/tmp/llgm-torch-extensions",
        }
    )
    .workdir("/opt/llgm")
)
# Upload library code and an explicit job allowlist, never the whole checkout.
PAYLOAD_FILES = (
    "tools/colbert_modal.py",
    "tools/colbert_worker.py",
    "tests/conftest.py",
    "tests/integration/test_live_colbert.py",
    "tests/fixtures/longmemeval-s.json",
    "experiments/colbert_modal.json",
    "experiments/colbert-requirements.txt",
    "pyproject.toml",
)
for payload_path in (*sorted(ROOT.glob("src/**/*.py")), *(ROOT / name for name in PAYLOAD_FILES)):
    image = image.add_local_file(payload_path, "/opt/llgm/" + str(payload_path.relative_to(ROOT)))


def file_hash(path: Path) -> str:
    """Hash a complete file without loading the model or dataset into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pinned(url: str, destination: Path, expected: dict) -> None:
    """Publish only a complete download matching independently recorded pins."""
    if destination.exists():
        if (
            destination.stat().st_size != expected["bytes"]
            or file_hash(destination) != expected["sha256"]
        ):
            raise ValueError(f"Existing asset does not match its pin: {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    try:
        request = Request(url, headers={"User-Agent": "llgm-colbert-validation"})
        with urlopen(request, timeout=120) as response, temporary.open("xb") as stream:
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > expected["bytes"]:
                    raise ValueError(f"Download exceeds its pinned length: {destination.name}")
                stream.write(chunk)
        if (
            temporary.stat().st_size != expected["bytes"]
            or file_hash(temporary) != expected["sha256"]
        ):
            raise ValueError(f"Download does not match its pin: {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_directory(run_id: str) -> Path:
    """Resolve a caller-supplied run label without allowing path traversal."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("run_id must be 1–80 letters, digits, underscores or hyphens")
    directory = (ASSETS / "runs" / run_id).resolve()
    if not directory.is_relative_to(ASSETS.resolve()):
        raise ValueError("Run directory escapes the asset volume")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def runtime_manifest() -> dict:
    """Record observed environment and source identities without credentials."""
    command = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if shutil.which("nvidia-smi")
        else None
    )
    paths = sorted(ROOT.glob("src/**/*.py")) + sorted(ROOT.glob("tools/colbert*.py"))
    files = {str(path.relative_to(ROOT)): file_hash(path) for path in paths}
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=30)
    return {
        "python": sys.version,
        "pins": PINS,
        "dependency_lock_sha256": file_hash(ROOT / "experiments/colbert-requirements.txt"),
        "source_files": files,
        "source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "installed_packages": frozen.splitlines(),
        "gpu": command.stdout.strip() if command and command.returncode == 0 else None,
        "container_id": os.environ.get("MODAL_TASK_ID"),
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
        "pid": os.getpid(),
        "billed_usd": None,
    }


def run_process(command: list[str], *, stream, timeout: int, environment=None) -> int:
    """Wait for native work and terminate its process group on timeout or interruption."""
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The group can retain compiler workers after its immediate child exits.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def run_worker(arguments: list[str], output: Path, timeout: int = 1200) -> dict:
    """Run native indexing in a child process and retain logs on failure."""
    log = output.with_suffix(".log")
    started = time.perf_counter()
    with log.open("w") as stream:
        returncode = run_process(
            [
                sys.executable,
                str(ROOT / "tools/colbert_worker.py"),
                *arguments,
                "--output",
                str(output),
            ],
            stream=stream,
            timeout=timeout,
        )
    if returncode:
        print(log.read_text()[-12000:])
        raise RuntimeError(f"ColBERT worker exited {returncode}; see {log.name}")
    payload = json.loads(output.read_text())
    payload["job_seconds"] = time.perf_counter() - started
    payload["container_id"] = os.environ.get("MODAL_TASK_ID")
    output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


@app.function(
    image=image,
    volumes={"/assets": volume},
    cpu=2,
    memory=4096,
    timeout=900,
    max_containers=1,
    retries=0,
    scaledown_window=2,
)
def prepare_assets() -> dict:
    """Download and verify the pinned released weights and diagnostic dataset."""
    from llgm.retrieval.colbert import checkpoint_sha256

    volume.reload()
    checkpoint = PINS["checkpoint"]
    for name, expected in checkpoint["files"].items():
        url = f"https://huggingface.co/{checkpoint['repository']}/resolve/{checkpoint['revision']}/{name}"
        download_pinned(url, ASSETS / "checkpoint" / name, expected)
    actual = checkpoint_sha256(ASSETS / "checkpoint")
    if actual != checkpoint["sha256"]:
        raise ValueError("Checkpoint directory has unexpected or changed files")
    dataset = PINS["dataset"]
    download_pinned(dataset["url"], ASSETS / "datasets" / dataset["file"], dataset)
    result = {
        "checkpoint_sha256": actual,
        "dataset_sha256": dataset["sha256"],
        "status": "verified",
    }
    (ASSETS / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(
    image=image,
    gpu=PINS["gpu"],
    cpu=PINS["cpu"],
    memory=PINS["memory_mib"],
    volumes={"/assets": volume},
    timeout=2400,
    max_containers=1,
    retries=0,
    scaledown_window=2,
)
def gpu_test(run_id: str) -> dict:
    """Run the existing genuine GPU test and publish a durable real-history index."""
    volume.reload()
    directory = run_directory(run_id)
    manifest = runtime_manifest()
    (directory / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environment = dict(os.environ)
    environment.update(
        {
            "LLGM_TEST_COLBERT": "1",
            "LLGM_TEST_COLBERT_GPUS": "1",
            "LLGM_TEST_COLBERT_CHECKPOINT": str(ASSETS / "checkpoint"),
            "LLGM_TEST_COLBERT_CHECKPOINT_SHA256": PINS["checkpoint"]["sha256"],
            "LLGM_TEST_COLBERT_REVISION": PINS["repository"]["revision"],
            "LLGM_TEST_LONGMEMEVAL_PATH": str(ASSETS / "datasets" / PINS["dataset"]["file"]),
        }
    )
    try:
        with (directory / "pytest.log").open("w") as stream:
            returncode = run_process(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/integration/test_live_colbert.py",
                    "-q",
                    f"--junitxml={directory / 'junit.xml'}",
                ],
                environment=environment,
                stream=stream,
                timeout=900,
            )
        if returncode:
            print((directory / "pytest.log").read_text()[-14000:])
            raise RuntimeError("Official ColBERT integration test failed")
        payload = run_worker(
            ["build", "--case-id", PINS["sanity_case_id"], "--run-id", run_id],
            directory / "build.json",
        )
        for name in ("passages.jsonl", "baseline.json"):
            shutil.copyfile(ASSETS / "records" / payload["index_id"] / name, directory / name)
        return {"build": payload, "environment": manifest, "pytest_exit_code": returncode}
    finally:
        if (ROOT / "runs/integration").exists():
            shutil.copytree(
                ROOT / "runs/integration", directory / "integration", dirs_exist_ok=True
            )
        volume.commit()


@app.function(
    image=image,
    gpu=PINS["gpu"],
    cpu=PINS["cpu"],
    memory=PINS["memory_mib"],
    volumes={"/assets": volume},
    timeout=900,
    max_containers=1,
    retries=0,
    scaledown_window=2,
)
def gpu_reopen(index_id: str, run_id: str) -> dict:
    """Reopen the persisted index in a container distinct from the builder."""
    volume.reload()
    try:
        return run_worker(
            ["reopen", "--index-id", index_id], run_directory(run_id) / "reopen.json", 600
        )
    finally:
        volume.commit()


@app.function(
    image=image,
    gpu=PINS["gpu"],
    cpu=PINS["cpu"],
    memory=PINS["memory_mib"],
    volumes={"/assets": volume},
    timeout=2700,
    max_containers=1,
    retries=0,
    scaledown_window=2,
)
def gpu_benchmark(run_id: str) -> dict:
    """Measure the declared BM25 and PLAID diagnostic cases without hosted generation."""
    volume.reload()
    directory = run_directory(run_id)
    (directory / "environment.json").write_text(json.dumps(runtime_manifest(), indent=2) + "\n")
    try:
        return run_worker(["benchmark", "--run-id", run_id], directory / "benchmark.json", 2400)
    finally:
        volume.commit()


@app.function(
    image=image,
    volumes={"/assets": volume},
    timeout=60,
    max_containers=1,
    retries=0,
    scaledown_window=2,
)
def describe_index(index_id: str) -> dict:
    """Return verified index metadata through authenticated Modal RPC."""
    from colbert_worker import describe_index as describe

    volume.reload()
    return describe(index_id)


@app.function(
    image=image,
    gpu=PINS["gpu"],
    cpu=PINS["cpu"],
    memory=PINS["memory_mib"],
    volumes={"/assets": volume},
    timeout=180,
    max_containers=1,
    retries=0,
    scaledown_window=15,
)
def search_index(index_id: str, query: str, k: int) -> dict:
    """Search an immutable saved index using the official encoder and PLAID engine."""
    from colbert_worker import search_index as search

    volume.reload()
    return search(index_id, query, k)


_workspace_service = None


def workspace_service():
    """Reuse one live workspace service separately from the frozen benchmark indexes."""
    global _workspace_service
    if _workspace_service is None:
        from colbert_worker import _configuration

        from llgm.retrieval.live import WorkspaceIndexService

        _workspace_service = WorkspaceIndexService(
            ASSETS / "workspaces", _configuration("0" * 64, PINS)
        )
    return _workspace_service


@app.function(
    image=image,
    gpu=PINS["gpu"],
    cpu=PINS["cpu"],
    memory=PINS["memory_mib"],
    volumes={"/assets": volume},
    timeout=2400,
    max_containers=1,
    retries=0,
    scaledown_window=60,
)
def prepare_workspace(records: list[dict]) -> dict:
    """Index an uploaded source snapshot for application and viewer hybrid search."""
    volume.reload()
    result = workspace_service().prepare(records)
    volume.commit()
    return result


@app.function(
    image=image,
    gpu=PINS["gpu"],
    cpu=PINS["cpu"],
    memory=PINS["memory_mib"],
    volumes={"/assets": volume},
    timeout=180,
    max_containers=1,
    retries=0,
    scaledown_window=60,
)
def search_workspace(index_id: str, query: str, k: int) -> dict:
    """Search the current uploaded generation with matching BM25 and ColBERT corpora."""
    volume.reload()
    return workspace_service().search(index_id, query, k)


def collect_run(run_id: str, destination: Path) -> None:
    """Copy each retained artifact from this run without exposing unrelated data."""
    for entry in volume.iterdir(f"/runs/{run_id}", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = Path(entry.path.lstrip("/")).relative_to(f"runs/{run_id}")
        local = destination / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as stream:
            for chunk in volume.read_file(entry.path):
                stream.write(chunk)


@app.local_entrypoint()
def main(action: str = "test", run_id: str = "") -> None:
    """Prepare assets or execute one bounded live job and retain local evidence."""
    if action not in {"prepare", "test", "benchmark"}:
        raise ValueError("Unsupported ColBERT experiment action")
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run_id")
    directory = ROOT / "runs/colbert-modal" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    result = {"action": action, "run_id": run_id, "status": "running", "pins": PINS}
    started = time.perf_counter()
    artifacts_collected = False
    try:
        result["preparation"] = prepare_assets.remote()
        if action == "test":
            result["test"] = gpu_test.remote(run_id)
            index_id = result["test"]["build"]["index_id"]
            result["reopen"] = gpu_reopen.remote(index_id, run_id)
            if result["test"]["environment"]["container_id"] == result["reopen"]["container_id"]:
                raise RuntimeError("Reopen test did not use a distinct container")
            from tools.colbert_client_check import check_saved_index

            collect_run(run_id, directory)
            artifacts_collected = True
            result["client"] = asyncio.run(
                check_saved_index(
                    directory,
                    describe_rpc=describe_index.remote.aio,
                    search_rpc=search_index.remote.aio,
                )
            )
        elif action == "benchmark":
            result["benchmark"] = gpu_benchmark.remote(run_id)
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        (directory / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        if action != "prepare" and not artifacts_collected:
            try:
                collect_run(run_id, directory)
            except Exception as error:
                print(
                    f"Artifact download failed: {type(error).__name__}. Retained on Modal Volume."
                )
        print(f"Run artifacts: {directory}")
