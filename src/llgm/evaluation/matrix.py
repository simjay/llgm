"""The committed B/D/H/C × S/U/A matrix and explicit local preflight."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
from pathlib import Path

from llgm.core.errors import ConfigurationError, SchemaError
from llgm.evaluation.artifacts import read_jsonl
from llgm.retrieval import ColBERTTokenizer, validate_encoder_text
from llgm.retrieval.base import passage_from_dict

BACKENDS = ("B", "D", "H", "C")
POLICIES = {"S": "single", "U": "upfront", "A": "adaptive"}
REQUIRED_ARMS = tuple(f"{backend}-{policy}" for backend in BACKENDS for policy in POLICIES)


def matrix_template() -> dict:
    """Return the committed twelve-arm configuration with model and asset pins unset."""
    return {
        "schema_version": 1,
        "objective": "E03",
        "stage": "development",
        "seed": 1729,
        "arms": [
            {"id": arm, "backend": arm[0], "policy": POLICIES[arm[-1]]} for arm in REQUIRED_ARMS
        ],
        "models": {
            role: {"provider": "openai", "model": None, "api_key_env": "OPENAI_API_KEY"}
            for role in ("root", "sidecar")
        },
        "dense": {
            "model": "text-embedding-3-large",
            "dimensions": 3072,
            "api_key_env": "OPENAI_API_KEY",
        },
        "hybrid": {"rank_constant": 60, "pool_size": 40},
        "passages": {"tokenizer": "colbert-local-fast", "window": 180, "overlap": 32},
        "tokenizer": {"local_path": None, "revision": None},
        "colbert": {
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "repository_revision": None,
            "doc_maxlen": 180,
            "query_maxlen": 128,
            "nbits": 2,
            "ncells": 2,
            "centroid_score_threshold": 0.45,
            "ndocs": 1024,
            "nranks": 1,
            "gpus": 0,
        },
        "budgets": {
            "max_model_calls": 9,
            "max_sidecar_calls": 8,
            "max_searches": 4,
            "max_evidence_tokens": 8000,
            "max_bundle_tokens": 4000,
            "max_output_tokens": 1024,
            "max_context_tokens": 16000,
            "timeout_seconds": 120,
        },
        "overall": {
            "max_case_arm_runs": 600,
            "max_generation_calls": 5400,
            "max_embedding_requests": 10000,
            "timeout_seconds": 86400,
        },
        "accounting_tokenizer": "shared-colbert-tokenizer",
        "sampling": {"temperature": None},
        "prompts": "llgm.inference.iterative iterative prompts; code and module hash recorded at run time",
        "price_table": None,
        "currency_cap": None,
        "scorer": "official LongMemEval judge invoked separately",
        "decision_thresholds": None,
        "stopping_rule": "declared case set or shared run cap, include all failures",
    }


def validate_matrix(value: dict) -> None:
    """Enforce the committed comparison settings and positive overall execution caps."""
    if not isinstance(value, dict):
        raise ConfigurationError("Experiment matrix must be a JSON object")
    if value.get("schema_version") != 1:
        raise ConfigurationError("Unsupported experiment matrix schema")
    for name in (
        "passages",
        "hybrid",
        "dense",
        "colbert",
        "models",
        "budgets",
        "overall",
        "sampling",
        "tokenizer",
    ):
        if name in value and not isinstance(value[name], dict):
            raise ConfigurationError(f"Matrix {name} must be a JSON object")
    tokenizer_settings(value)
    for role, model in value.get("models", {}).items():
        if not isinstance(model, dict):
            raise ConfigurationError(f"Matrix models.{role} must be a JSON object")
        for name in ("model", "provider", "api_key_env", "base_url"):
            item = model.get(name)
            if item is not None and not isinstance(item, str):
                raise ConfigurationError(f"Matrix models.{role}.{name} must be text or null")
    arms = value.get("arms", [])
    if not isinstance(arms, list) or any(not isinstance(arm, dict) for arm in arms):
        raise ConfigurationError("Matrix arms must be a list of JSON objects")
    ids = [arm.get("id") for arm in arms]
    if (
        len(ids) != 12
        or any(not isinstance(identity, str) for identity in ids)
        or set(ids) != set(REQUIRED_ARMS)
    ):
        raise ConfigurationError(
            "The E03 matrix must contain every B/D/H/C × S/U/A arm exactly once"
        )
    for arm in arms:
        if arm.get("backend") != arm["id"][0] or arm.get("policy") != POLICIES[arm["id"][-1]]:
            raise ConfigurationError(f"Arm identity does not match its backend/policy: {arm['id']}")
    if (
        value.get("passages", {}).get("window") != 180
        or value.get("passages", {}).get("overlap") != 32
    ):
        raise ConfigurationError(
            "Committed first matrix requires shared 180/32 passages; declare sweeps separately"
        )
    if value.get("hybrid") != {"rank_constant": 60, "pool_size": 40}:
        raise ConfigurationError(
            "Committed H uses equal-weight RRF with constant 60 and top 40 components"
        )
    dense = value.get("dense", {})
    if (
        dense.get("model") != "text-embedding-3-large"
        or type(dense.get("dimensions")) is not int
        or not 1 <= dense["dimensions"] <= 3072
    ):
        raise ConfigurationError(
            "Committed D requires text-embedding-3-large and explicit valid dimensions"
        )
    if value.get("currency_cap") is not None:
        raise ConfigurationError(
            "This runner does not claim a strict currency cap; use its enforceable call/time limits"
        )
    if value.get("sampling", {"temperature": None}) != {"temperature": None}:
        raise ConfigurationError(
            "Current runner uses provider-default sampling, recorded per role; other sampling is unsupported"
        )
    from llgm.inference.budget import Budget

    budget = Budget(**value.get("budgets", {}))
    if (budget.max_searches, budget.max_sidecar_calls, budget.max_model_calls) != (4, 8, 9):
        raise ConfigurationError(
            "Committed matrix requires at most 4 searches, 8 sidecar calls and 1 root call"
        )
    for key in (
        "max_case_arm_runs",
        "max_generation_calls",
        "max_embedding_requests",
        "timeout_seconds",
    ):
        cap = value.get("overall", {}).get(key)
        if (
            isinstance(cap, bool)
            or not isinstance(cap, (int, float))
            or not (0 < cap < float("inf"))
        ):
            raise ConfigurationError(f"A finite positive overall.{key} is required")


def load_matrix(path: str | Path) -> dict:
    """Read and validate a local matrix, reporting unreadable JSON as configuration failure."""
    try:
        matrix = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Cannot load matrix: {path}") from exc
    validate_matrix(matrix)
    return matrix


def _available(module: str) -> bool:
    """Check module discoverability without importing the target module."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def resolve_backend_scope(required_backends: list[str] | None) -> list[str]:
    """Resolve an explicit readiness scope while retaining all four backends by default."""
    scope = list(BACKENDS) if required_backends is None else list(required_backends)
    if (
        not scope
        or any(not isinstance(backend, str) for backend in scope)
        or len(scope) != len(set(scope))
        or not set(scope).issubset(BACKENDS)
    ):
        raise ConfigurationError("Required backends must be unique nonempty members of B/D/H/C")
    return scope


def tokenizer_settings(matrix: dict) -> dict:
    """Use complete standalone pins, falling back only for an absent or unset configuration."""
    config = matrix.get("tokenizer", {})
    if not isinstance(config, dict):
        raise ConfigurationError("Matrix tokenizer must be a JSON object")
    if config:
        if set(config) != {"local_path", "revision"}:
            raise ConfigurationError("Tokenizer requires explicit local_path and revision strings")
        if any(value is not None for value in config.values()):
            if any(not isinstance(value, str) or not value.strip() for value in config.values()):
                raise ConfigurationError(
                    "Tokenizer requires explicit local_path and revision strings"
                )
            return dict(config)
    legacy = matrix.get("colbert", {})
    return {
        "local_path": legacy.get("checkpoint_path"),
        "revision": legacy.get("repository_revision"),
    }


def preflight(
    prepared: str | Path,
    matrix: dict,
    *,
    diagnostic: bool = False,
    retrieval_only: bool = False,
    required_backends: list[str] | None = None,
) -> dict:
    """Check the full matrix or an explicitly narrower backend scope without dispatch.

    A selected scope still requires the shared pinned passage tokenizer. Backends
    outside it are marked unchecked, and cannot establish full-matrix readiness.
    """
    validate_matrix(matrix)
    scope = resolve_backend_scope(required_backends)
    if diagnostic:
        if required_backends is not None and scope != ["B"]:
            raise ConfigurationError("Offline diagnostic scope must be B only")
        scope = ["B"]
    prepared = Path(prepared)
    report = {
        "mode": "diagnostic-B-S-only"
        if diagnostic
        else (
            "retrieval-only-all4-backends"
            if retrieval_only and set(scope) == set(BACKENDS)
            else "full-required-12-arm-matrix"
            if not retrieval_only and set(scope) == set(BACKENDS)
            else "retrieval-only-selected-backends"
            if retrieval_only
            else "selected-backend-matrix"
        ),
        "required_backends": scope,
        "excluded_backends": [backend for backend in BACKENDS if backend not in scope],
        "full_matrix_required": not diagnostic
        and not retrieval_only
        and set(scope) == set(BACKENDS),
        "network_calls": 0,
        "errors": [],
        "warnings": [],
        "arms": [],
    }
    common = report["errors"]
    try:
        manifest = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        common.append("Missing or malformed prepared.json; run experiment prepare first")
        manifest = {}
    supported_preparation = (
        isinstance(manifest, dict)
        and manifest.get("schema_version") == 2
        and manifest.get("evidence_schema") == "immutable-node-v2"
    )
    if not supported_preparation:
        common.append(
            "Unsupported prepared evidence format: require schema_version=2 and "
            "evidence_schema=immutable-node-v2. Regenerate preparation in a new directory; "
            "legacy source-version artifacts remain unchanged."
        )
        manifest = {}
    if manifest and manifest.get("selection", {}).get("status") != "ready":
        common.append("History isolation did not produce a usable development/held-out split")
    if diagnostic and manifest.get("dataset", {}).get("kind") != "controlled-offline-fixture":
        common.append(
            "The fake diagnostic reader is restricted to dedicated offline-smoke fixtures"
        )
    if not diagnostic:
        if manifest.get("selection", {}).get("freeze_required"):
            if (
                matrix.get("stage") != "held-out"
                or not matrix.get("configuration_frozen_at")
                or not matrix.get("development_dataset_sha256")
            ):
                common.append(
                    "Reserved benchmark evaluation requires stage=held-out, configuration_frozen_at, and separate development_dataset_sha256"
                )
        if manifest.get("tokenizer", {}).get("implementation") != "colbert-local-fast":
            common.append(
                "Substantive matrix requires shared passages made with the pinned local ColBERT tokenizer"
            )
        if not manifest.get("dataset", {}).get("revision"):
            common.append("Pin the cleaned LongMemEval dataset revision")
        kind = manifest.get("dataset", {}).get("kind")
        if kind not in {"longmemeval-s-cleaned", "controlled-development"}:
            common.append(
                "Full matrix requires cleaned LongMemEval-S or explicitly separate controlled development data"
            )
        elif kind == "controlled-development":
            if matrix.get("stage") != "development":
                common.append(
                    "Synthetic controlled histories are for development diagnostics, not held-out benchmark claims"
                )
            report["warnings"].append(
                "Controlled development workload: this run cannot establish benchmark performance"
            )
        for role in () if retrieval_only else ("root", "sidecar"):
            model = matrix.get("models", {}).get(role, {})
            identity = model.get("model")
            if not identity or "latest" in identity.lower():
                common.append(f"Pin an explicit non-latest {role} model ID")
            provider = model.get("provider")
            package = {
                "openai": "openai",
                "openai_compatible": "openai",
                "anthropic": "anthropic",
            }.get(provider)
            if not package or not _available(package):
                common.append(f"Missing supported provider SDK for {role}: {provider}")
            key = model.get("api_key_env") or (
                "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            )
            if not os.environ.get(key):
                common.append(f"Missing {role} credential environment variable {key}")
        if matrix.get("decision_thresholds") is None:
            report["warnings"].append(
                "Decision thresholds are unset: development diagnostics may run, confirmatory claims may not"
            )
    backend_errors = {key: [] for key in BACKENDS}
    if set(scope) & {"B", "H"}:
        try:
            with sqlite3.connect(":memory:") as connection:
                connection.execute("CREATE VIRTUAL TABLE fts USING fts5(text)")
            connection.close()
        except sqlite3.OperationalError:
            backend_errors["B"].append("SQLite lacks FTS5")
    if set(scope) & {"D", "H"}:
        if not _available("openai"):
            backend_errors["D"].append("OpenAI SDK is unavailable for dense embeddings")
        dense_key = matrix["dense"].get("api_key_env", "OPENAI_API_KEY")
        if not os.environ.get(dense_key):
            backend_errors["D"].append(
                f"Missing dense embedding credential environment variable {dense_key}"
            )
    backend_errors["H"] = [*backend_errors["B"], *backend_errors["D"]]
    tokenizer = None
    if "C" in scope:
        try:
            from llgm.retrieval.colbert import ColBERTConfig, preflight_colbert

            config = ColBERTConfig(
                **matrix["colbert"], index_root=prepared / "indexes", index_name="preflight"
            )
            backend_errors["C"].extend(preflight_colbert(config))
        except (ConfigurationError, TypeError, ValueError, OSError, ImportError) as exc:
            backend_errors["C"].append(f"ColBERT configuration/assets not ready: {exc}")
    if (
        not diagnostic
        and manifest.get("tokenizer", {}).get("implementation") == "colbert-local-fast"
    ):
        try:
            tokenizer = ColBERTTokenizer(**tokenizer_settings(matrix))
            if tokenizer.descriptor()["sha256"] != manifest["tokenizer"].get("sha256"):
                common.append(
                    "Prepared passage tokenizer hash differs from configured ColBERT tokenizer"
                )
        except (ConfigurationError, TypeError, ValueError, OSError, ImportError) as exc:
            common.append(f"Shared passage tokenizer is not ready: {exc}")
    if supported_preparation:
        try:
            queries = read_jsonl(prepared / "queries.jsonl")
        except (OSError, ValueError) as exc:
            common.append(f"Prepared queries are unreadable: {exc}")
            queries = []
        for query in queries:
            try:
                if tokenizer is not None:
                    validate_encoder_text(
                        query["question"], tokenizer, matrix["colbert"]["query_maxlen"], "query"
                    )
                for raw in read_jsonl(prepared / query["passages_path"]):
                    passage = passage_from_dict(raw)
                    if tokenizer is not None:
                        validate_encoder_text(
                            passage.text, tokenizer, matrix["colbert"]["doc_maxlen"]
                        )
            except (
                ConfigurationError,
                SchemaError,
                TypeError,
                KeyError,
                ValueError,
                OSError,
            ) as exc:
                common.append(
                    f"Invalid prepared case or source-span reference ({query.get('case_id', 'unknown')}): "
                    f"{exc}. Regenerate incompatible preparation in a new directory."
                )
    for arm in matrix["arms"]:
        selected = arm["backend"] in scope and (
            arm["id"] == "B-S"
            if diagnostic
            else (arm["id"].endswith("-S") if retrieval_only else True)
        )
        errors = (
            [*common, *backend_errors[arm["backend"]]]
            if selected
            else backend_errors[arm["backend"]]
        )
        report["arms"].append(
            {
                **arm,
                "selected": selected,
                "status": ("blocked" if errors else "ready") if selected else "not_checked",
                "errors": errors,
            }
        )
    report["ready"] = not common and all(
        arm["status"] == "ready" for arm in report["arms"] if arm["selected"]
    )
    report["full_matrix_ready"] = report["full_matrix_required"] and report["ready"]
    report["full_retrieval_matrix_ready"] = (
        not diagnostic and set(scope) == set(BACKENDS) and report["ready"]
    )
    report["benchmark_ready"] = report["full_matrix_ready"]
    return report
