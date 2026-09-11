"""Fail execution prerequisites before model setup without external tools or downloads."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from llgm.core.errors import ConfigurationError
from llgm.evaluation import benchmark_preflight

IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def protocol():
    """Supply independently chosen minimal runtime and reader settings."""
    return {
        "limits": {"concurrent_cases": 2},
        "budget": {"max_model_calls": 5},
        "maintenance": {"mode": "disabled", "budget": {"max_model_calls": 1}},
        "docker": {"image": IMAGE, "startup_timeout_seconds": 3},
        "runtime": {"retrieval_k": 4, "max_seed_nodes": 2},
        "reader": {
            "retrieval_k": 4,
            "passage_chars": 64,
            "max_input_tokens": 1000,
            "max_output_tokens": 100,
        },
    }


@pytest.fixture
def local_prerequisites(monkeypatch):
    """Replace only tokenizer asset loading, package metadata and Docker subprocess inspection."""
    observed = {"tokenizers": [], "commands": []}

    def encoding(name):
        """Record tokenizer initialization without reading or downloading assets."""
        observed["tokenizers"].append(name)
        return SimpleNamespace(name=name)

    def inspect(command, **options):
        """Return an exact local image identity without starting Docker or a container."""
        observed["commands"].append((command, options))
        return subprocess.CompletedProcess(command, 0, stdout=IMAGE + "\n", stderr="")

    monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(get_encoding=encoding))
    monkeypatch.setattr(benchmark_preflight.subprocess, "run", inspect)
    monkeypatch.setattr(benchmark_preflight.importlib.metadata, "version", lambda name: "fixture")
    return observed


def test_preflight_returns_verified_prerequisites_without_models(protocol, local_prerequisites):
    """Configuration needs no model identities, credentials or constructed client to validate."""
    result = benchmark_preflight.preflight_runtime(protocol)
    assert result == {
        "docker_image_id": IMAGE,
        "tokenizer": "cl100k_base",
        "package_versions": {"llgm": "fixture", "tiktoken": "fixture", "openai": "fixture"},
    }
    assert local_prerequisites["tokenizers"] == ["cl100k_base"]
    assert local_prerequisites["commands"] == [
        (
            ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE],
            {"capture_output": True, "text": True, "timeout": 3, "check": False},
        )
    ]


@pytest.mark.parametrize("concurrency", [0, -1, 9, True, 1.5, "2"])
def test_invalid_concurrency_fails_before_external_prerequisites(
    protocol, local_prerequisites, concurrency
):
    """Invalid case fan-out cannot reach tokenizer preparation or Docker inspection."""
    protocol["limits"]["concurrent_cases"] = concurrency
    with pytest.raises(ConfigurationError, match="concurrent_cases"):
        benchmark_preflight.preflight_runtime(protocol)
    assert local_prerequisites == {"tokenizers": [], "commands": []}


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("limits", "max_unknown_calls", 0),
        ("limits", "max_unknown_calls", True),
        ("limits", "tokens_per_minute", 0),
        ("limits", "tokens_per_minute", 1.5),
        ("budget", "max_model_calls", 0),
        ("maintenance", "mode", "invented"),
        ("docker", "memory_mb", -1),
        ("runtime", "retrieval_k", 1),
        ("runtime", "unknown_option", 1),
        ("reader", "max_input_tokens", True),
        ("reader", "unknown_option", 1),
    ],
)
def test_invalid_runtime_settings_fail_before_dependencies(
    protocol, local_prerequisites, section, field, value
):
    """The public runtime constructors reject invalid limits and coupled seed/retrieval settings."""
    protocol[section][field] = value
    with pytest.raises(ConfigurationError):
        benchmark_preflight.preflight_runtime(protocol)
    assert local_prerequisites == {"tokenizers": [], "commands": []}


@pytest.mark.parametrize("image", ["python:3.12-slim", "sha256:short", "sha256:" + "G" * 64])
def test_unpinned_images_are_rejected_without_inspection(protocol, local_prerequisites, image):
    """A mutable tag or malformed image ID cannot masquerade as an immutable runtime pin."""
    protocol["docker"]["image"] = image
    with pytest.raises(ConfigurationError, match="pinned"):
        benchmark_preflight.preflight_runtime(protocol)
    assert local_prerequisites == {"tokenizers": [], "commands": []}


@pytest.mark.parametrize("failure", ["missing_docker", "missing_image", "mismatch", "timeout"])
def test_docker_failure_requires_no_model_setup(
    protocol, local_prerequisites, monkeypatch, failure
):
    """Missing or mismatched local Docker prerequisites fail without any model configuration."""

    def unavailable(command, **options):
        """Inject local inspection failures without contacting an actual daemon."""
        if failure == "missing_docker":
            raise FileNotFoundError()
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return subprocess.CompletedProcess(
            command, 1 if failure == "missing_image" else 0, stdout="sha256:" + "b" * 64, stderr=""
        )

    monkeypatch.setattr(benchmark_preflight.subprocess, "run", unavailable)
    with pytest.raises(ConfigurationError, match="Docker"):
        benchmark_preflight.preflight_runtime(protocol)
    assert "models" not in protocol


@pytest.mark.parametrize("failure", ["missing_package", "asset_load"])
def test_tokenizer_unavailable_fails_before_docker(
    protocol, local_prerequisites, monkeypatch, failure
):
    """A missing tokenizer package or unloadable asset stops execution before model setup."""
    if failure == "missing_package":
        monkeypatch.setitem(sys.modules, "tiktoken", None)
    else:

        def fail(name):
            """Simulate a failed tokenizer asset load without accessing the network."""
            raise OSError("Asset unavailable")

        monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(get_encoding=fail))
    with pytest.raises(ConfigurationError, match="cl100k_base"):
        benchmark_preflight.preflight_runtime(protocol)
    assert local_prerequisites["commands"] == []
