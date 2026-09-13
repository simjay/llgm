"""Fail execution prerequisites before model setup without external tools or downloads."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from llgm.core.errors import ConfigurationError
from llgm.evaluation import benchmark_preflight

IDENTITY = {"implementation": "dspy", "version": "3.3.1", "deno_version": "2.9.6"}


@pytest.fixture
def protocol():
    """Supply independently chosen minimal runtime and reader settings."""
    return {
        "limits": {"concurrent_cases": 2},
        "budget": {"max_model_calls": 5},
        "maintenance": {"mode": "disabled", "budget": {"max_model_calls": 1}},
        "sandbox": {"startup_timeout_seconds": 3},
        "rlm": dict(IDENTITY),
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
    """Replace only tokenizer asset loading, package metadata and Deno executable inspection."""
    observed = {"tokenizers": [], "commands": []}

    def encoding(name):
        """Record tokenizer initialization without reading or downloading assets."""
        observed["tokenizers"].append(name)
        return SimpleNamespace(name=name)

    def inspect(command, **options):
        """Return the pinned executable version without starting a sandbox."""
        observed["commands"].append((command, options))
        return subprocess.CompletedProcess(command, 0, stdout="deno 2.9.6\n", stderr="")

    monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(get_encoding=encoding))
    monkeypatch.setattr(benchmark_preflight.subprocess, "run", inspect)
    monkeypatch.setattr(
        benchmark_preflight.importlib.metadata,
        "version",
        lambda name: "3.3.1" if name == "dspy" else "fixture",
    )
    monkeypatch.setitem(sys.modules, "deno", SimpleNamespace(find_deno_bin=lambda: "deno"))
    return observed


def test_preflight_returns_verified_prerequisites_without_models(protocol, local_prerequisites):
    """Configuration needs no model identities, credentials or constructed client to validate."""
    result = benchmark_preflight.preflight_runtime(protocol)
    assert result == {
        "rlm": IDENTITY,
        "tokenizer": "cl100k_base",
        "package_versions": {
            "llgm": "fixture",
            "tiktoken": "fixture",
            "openai": "fixture",
            "dspy": "3.3.1",
            "deno": "fixture",
        },
    }
    assert local_prerequisites["tokenizers"] == ["cl100k_base"]
    assert local_prerequisites["commands"] == [
        (
            ["deno", "--version"],
            {"capture_output": True, "text": True, "timeout": 3, "check": False},
        )
    ]


@pytest.mark.parametrize("concurrency", [0, -1, 9, True, 1.5, "2"])
def test_invalid_concurrency_fails_before_external_prerequisites(
    protocol, local_prerequisites, concurrency
):
    """Invalid case fan-out cannot reach tokenizer preparation or Deno inspection."""
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
        ("sandbox", "max_code_bytes", -1),
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


def test_historical_docker_protocol_is_rejected(protocol, local_prerequisites):
    """A frozen Docker experiment cannot silently acquire DSPy's controller semantics."""
    protocol["docker"] = {"image": "sha256:historical"}
    with pytest.raises(ConfigurationError, match="Historical Docker protocol"):
        benchmark_preflight.preflight_runtime(protocol)
    assert local_prerequisites == {"tokenizers": [], "commands": []}


@pytest.mark.parametrize("version", [None, "latest", "old"])
def test_unpinned_runtime_is_rejected(protocol, local_prerequisites, version):
    """A protocol must name the implemented DSPy and Deno versions."""
    protocol["rlm"]["version"] = version
    with pytest.raises(ConfigurationError, match="pinned"):
        benchmark_preflight.preflight_runtime(protocol)


@pytest.mark.parametrize("failure", ["missing_deno", "bad_exit", "mismatch", "timeout"])
def test_deno_failure_requires_no_model_setup(protocol, local_prerequisites, monkeypatch, failure):
    """Missing or mismatched local Deno prerequisites fail without any model configuration."""

    def unavailable(command, **options):
        """Inject local inspection failures without contacting an actual daemon."""
        if failure == "missing_deno":
            raise FileNotFoundError()
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return subprocess.CompletedProcess(
            command, 1 if failure == "bad_exit" else 0, stdout="sha256:" + "b" * 64, stderr=""
        )

    monkeypatch.setattr(benchmark_preflight.subprocess, "run", unavailable)
    with pytest.raises(ConfigurationError, match="Deno"):
        benchmark_preflight.preflight_runtime(protocol)
    assert "models" not in protocol


@pytest.mark.parametrize("failure", ["missing_package", "asset_load"])
def test_tokenizer_unavailable_fails_before_deno(
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
