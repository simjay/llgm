"""Validate benchmark execution prerequisites before any cost-bearing model setup."""

import importlib.metadata
import re
import subprocess

from llgm import LLGM, Budget, MaintenancePolicy
from llgm.core.errors import ConfigurationError
from llgm.inference.repl import DockerREPLConfig


def preflight_runtime(protocol: dict) -> dict:
    """Validate configuration, load the reader tokenizer and inspect a pinned local image.

    No models, workspaces or containers are created, and no Docker image is
    pulled. Normal tiktoken loading may populate its verified asset cache.
    The synchronous Docker inspection is bounded by the configured startup timeout.
    """
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
        docker = DockerREPLConfig(**protocol["docker"])
        LLGM(
            None,
            None,
            None,
            graph_model=None,
            maintenance_policy=policy,
            inference_budget=budget,
            repl_config=docker,
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
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", docker.image):
        raise ConfigurationError("Benchmark Docker image must be pinned to its full sha256 ID")
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        raise ConfigurationError(
            "Benchmark reader requires a loadable cl100k_base tokenizer"
        ) from exc
    try:
        result = subprocess.run(
            [docker.docker_executable, "image", "inspect", "--format", "{{.Id}}", docker.image],
            capture_output=True,
            text=True,
            timeout=docker.startup_timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigurationError("Benchmark requires an available local Docker image") from exc
    identity = result.stdout.strip()
    if result.returncode or identity != docker.image:
        raise ConfigurationError("Local Docker image does not match the pinned image ID")
    versions = {}
    for package in ("llgm", "tiktoken", "openai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"docker_image_id": identity, "tokenizer": encoding.name, "package_versions": versions}
