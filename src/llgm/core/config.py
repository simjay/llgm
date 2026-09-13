"""Typed, per-instance configuration. Parsing never opens a connection."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from llgm.core.errors import ConfigurationError

# Integration fixtures own this environment namespace.
_EXTERNAL_ENV_PREFIXES = ("LLGM_TEST_",)


def redact_url(value: str) -> str:
    """Remove userinfo, query parameters and fragments from endpoint diagnostics."""
    parts = urlsplit(value)
    host = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


@dataclass(frozen=True)
class Settings:
    """Validated per-instance storage, model, search, and budget configuration."""

    workspace_path: str = "./memory"
    blob_backend: str = "local"
    blob_uri: str = ""
    metadata_backend: str = "sqlite"
    database_url: str = field(default="", repr=False)
    retriever_backend: str = "hybrid"
    retrieval_modal_app: str = "llgm-colbert"
    retrieval_modal_environment: str = ""
    main_provider: str = "openai"
    main_model: str | None = None
    main_base_url: str | None = field(default=None, repr=False)
    main_api_key_env: str | None = None
    reader_provider: str = "openai"
    reader_model: str | None = None
    reader_base_url: str | None = field(default=None, repr=False)
    reader_api_key_env: str | None = None
    graph_provider: str = "openai"
    graph_model: str | None = None
    graph_base_url: str | None = field(default=None, repr=False)
    graph_api_key_env: str | None = None
    max_seed_nodes: int = 3
    retrieval_k: int = 12
    max_concurrency: int = 3
    max_journal_bytes: int = 65536
    max_model_calls: int = 40
    max_reader_calls: int = 36
    max_graph_calls: int = 8
    max_searches: int = 8
    max_evidence_tokens: int = 65536
    max_bundle_tokens: int = 8000
    max_context_tokens: int = 65536
    max_output_tokens: int = 2048
    timeout_seconds: float = 120.0
    field_sources: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate backend compatibility and resolve default local storage locations."""
        optional = {
            "main_model",
            "reader_model",
            "graph_model",
            "graph_base_url",
            "graph_api_key_env",
            "main_base_url",
            "reader_base_url",
            "main_api_key_env",
            "reader_api_key_env",
        }
        for item in fields(self):
            if item.name in _INT_FIELDS or item.name in {"timeout_seconds", "field_sources"}:
                continue
            value = getattr(self, item.name)
            if value is None and item.name in optional:
                continue
            if not isinstance(value, str):
                raise ConfigurationError(f"{item.name} must be text")
        if self.blob_backend not in {"local", "s3"}:
            raise ConfigurationError("blob_backend must be local or s3")
        if self.metadata_backend not in {"sqlite", "postgres"}:
            raise ConfigurationError("metadata_backend must be sqlite or postgres")
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if self.retrieval_k > 40 or self.max_seed_nodes > self.retrieval_k:
            raise ConfigurationError("Require max_seed_nodes <= retrieval_k <= 40")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not (0 < self.timeout_seconds < float("inf"))
        ):
            raise ConfigurationError("timeout_seconds must be finite and positive")
        base = Path(self.workspace_path).expanduser().resolve()
        if not self.blob_uri:
            if self.blob_backend != "local":
                raise ConfigurationError("s3 requires an explicit blob_uri")
            object.__setattr__(self, "blob_uri", (base / "blobs").as_uri())
        if not self.database_url:
            if self.metadata_backend != "sqlite":
                raise ConfigurationError("postgres requires an explicit database_url")
            object.__setattr__(self, "database_url", "sqlite:///" + str(base / "metadata.sqlite3"))
        if urlsplit(self.blob_uri).scheme != {"local": "file", "s3": "s3"}[self.blob_backend]:
            raise ConfigurationError("blob_uri scheme does not match blob_backend")
        db_scheme = urlsplit(self.database_url).scheme
        if (
            db_scheme
            not in {"sqlite": {"sqlite"}, "postgres": {"postgres", "postgresql"}}[
                self.metadata_backend
            ]
        ):
            raise ConfigurationError("database_url scheme does not match metadata_backend")
        if self.metadata_backend == "postgres" and self.retriever_backend in {
            "sqlite_fts5",
            "hybrid",
        }:
            raise ConfigurationError("postgres requires an explicitly compatible retriever_backend")
        if not self.retrieval_modal_app.strip():
            raise ConfigurationError("retrieval_modal_app must be nonempty text")
        for name in ("main_provider", "reader_provider", "graph_provider"):
            if getattr(self, name) not in {"openai", "anthropic", "openai_compatible"}:
                raise ConfigurationError(f"Unsupported {name}; inject a custom client in Python")
        object.__setattr__(self, "field_sources", MappingProxyType(dict(self.field_sources)))

    @classmethod
    def from_env(
        cls, *, environ: Mapping[str, str] | None = None, overrides: Mapping[str, Any] | None = None
    ) -> Settings:
        """Load process or supplied environment settings with optional explicit overrides."""
        return cls.load(environ=environ, overrides=overrides)

    @classmethod
    def load(
        cls,
        *,
        config_file: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Settings:
        """Apply defaults, application environment, TOML, then explicit overrides.

        ``LLGM_TEST_`` environment values belong to integration fixtures,
        so they are neither parsed
        nor included in settings diagnostics. Other unknown ``LLGM_`` keys
        remain errors. TOML and explicit overrides accept only settings fields.
        """
        environment = dict(os.environ if environ is None else environ)
        known = {f.name for f in fields(cls)} - {"field_sources"}
        values: dict[str, Any] = {}
        sources = {name: "default" for name in known}
        for key, value in environment.items():
            if not key.startswith("LLGM_") or key.startswith(_EXTERNAL_ENV_PREFIXES):
                continue
            name = key[5:].lower()
            if name not in known:
                raise ConfigurationError(f"Unknown LLGM setting: {key}")
            values[name] = _convert(name, value)
            sources[name] = "environment"
        if config_file is not None:
            with Path(config_file).open("rb") as stream:
                document = tomllib.load(stream)
            if "llgm" in document:
                if set(document) != {"llgm"} or not isinstance(document["llgm"], dict):
                    raise ConfigurationError(
                        "Configuration file must contain only the [llgm] table"
                    )
                document = document["llgm"]
            _merge(values, sources, document, known, "file")
        _merge(values, sources, overrides or {}, known, "explicit")
        return cls(**values, field_sources=sources)

    def redacted(self) -> dict[str, Any]:
        """Return diagnostic settings with endpoint credentials and query strings removed."""
        result = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "field_sources"}
        for name in (
            "database_url",
            "blob_uri",
            "main_base_url",
            "reader_base_url",
            "graph_base_url",
        ):
            if result[name]:
                result[name] = redact_url(result[name])
        result["field_sources"] = dict(self.field_sources)
        return result


_INT_FIELDS = {
    "max_seed_nodes",
    "retrieval_k",
    "max_concurrency",
    "max_journal_bytes",
    "max_model_calls",
    "max_reader_calls",
    "max_graph_calls",
    "max_searches",
    "max_evidence_tokens",
    "max_bundle_tokens",
    "max_context_tokens",
    "max_output_tokens",
}


def _convert(name: str, value: Any) -> Any:
    """Parse one setting according to its declared scalar type without boolean coercion."""
    try:
        if name in _INT_FIELDS:
            if isinstance(value, bool) or isinstance(value, float):
                raise ValueError
            return int(value)
        if name == "timeout_seconds":
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if value is not None and not isinstance(value, str):
            raise ValueError
        return value
    except (ValueError, TypeError):
        raise ConfigurationError(f"Invalid value type for {name}") from None


def _merge(values, sources, supplied, known, origin):
    """Apply a validated configuration layer and record each supplied field's origin."""
    for name, value in supplied.items():
        if name not in known:
            raise ConfigurationError(f"Unknown LLGM setting: {name}")
        values[name] = _convert(name, value)
        sources[name] = origin
