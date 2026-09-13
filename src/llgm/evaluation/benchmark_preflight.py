"""Validate benchmark execution prerequisites before any cost-bearing model setup."""

import importlib.metadata
import subprocess

from llgm import LLGM, Budget, MaintenancePolicy
from llgm.core.errors import ConfigurationError
from llgm.inference.repl import SandboxConfig


def preflight_runtime(protocol: dict) -> dict:
    """Validate DSPy and Deno identities before any cost-bearing model setup.

    Runtime assets may be fetched on first sandbox startup. This preflight
    checks installed packages and the executable without starting generated code.
    Historical Docker protocols require their original checkout.
    """
    if "docker" in protocol:
        raise ConfigurationError(
            "Historical Docker protocol requires its original checkout. Use a DSPy protocol with sandbox and rlm fields."
        )
    try:
        concurrency = protocol["limits"].get("concurrent_cases", 1)
        if type(concurrency) is not int or not 1 <= concurrency <= 8:
            raise ConfigurationError("concurrent_cases must be an integer from 1 to 8")
        unknown_limit = protocol["limits"].get("max_unknown_calls", 16)
        if type(unknown_limit) is not int or unknown_limit < 1:
            raise ConfigurationError("max_unknown_calls must be a positive integer")
        token_limit = protocol["limits"].get("tokens_per_minute")
        if token_limit is not None and (type(token_limit) is not int or token_limit < 1):
            raise ConfigurationError("tokens_per_minute must be a positive integer")
        budget = Budget(**protocol["budget"])
        maintenance = protocol["maintenance"]
        policy = MaintenancePolicy(
            **{key: value for key, value in maintenance.items() if key != "budget"},
            budget=Budget(**maintenance["budget"]),
        )
        sandbox = SandboxConfig(**protocol["sandbox"])
        LLGM(
            None,
            None,
            None,
            graph_model=None,
            maintenance_policy=policy,
            inference_budget=budget,
            repl_config=sandbox,
            capture_text=True,
            **protocol["runtime"],
        )
        reader = protocol["reader"]
        fields = {"retrieval_k", "passage_chars", "max_input_tokens", "max_output_tokens"}
        if (
            not isinstance(reader, dict)
            or set(reader) != fields
            or any(type(value) is not int or value < 1 for value in reader.values())
        ):
            raise ConfigurationError("Reader requires exactly four positive integer limits")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ConfigurationError("Invalid benchmark execution configuration") from exc
    identity = protocol.get("rlm")
    if identity != {"implementation": "dspy", "version": "3.3.1", "deno_version": "2.9.6"}:
        raise ConfigurationError("Benchmark requires pinned DSPy and Deno identities")
    try:
        from deno import find_deno_bin

        if importlib.metadata.version("dspy") != identity["version"]:
            raise ConfigurationError("Installed DSPy does not match the protocol")
        executable = find_deno_bin()
    except (ImportError, FileNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
        raise ConfigurationError("Benchmark requires llgm[rlm]") from exc
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        raise ConfigurationError(
            "Benchmark reader requires a loadable cl100k_base tokenizer"
        ) from exc
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=sandbox.startup_timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigurationError("Benchmark requires an available Deno executable") from exc
    if result.returncode or (result.stdout.splitlines() or [""])[0].split()[:2] != [
        "deno",
        identity["deno_version"],
    ]:
        raise ConfigurationError("Deno executable does not match the protocol")
    versions = {}
    for package in ("llgm", "tiktoken", "openai", "dspy", "deno"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"rlm": identity, "tokenizer": encoding.name, "package_versions": versions}
