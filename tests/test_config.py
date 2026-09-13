"""Configuration precedence, isolation, validation and credential-redaction contracts."""

import os
import tempfile
from pathlib import Path

import pytest

from llgm import Settings
from llgm.core.errors import ConfigurationError


def test_explicit_file_environment_default_precedence():
    """Explicit values outrank files and environment while omitted fields retain defaults."""
    with tempfile.TemporaryDirectory() as directory:
        file = Path(directory) / "llgm.toml"
        file.write_text('[llgm]\nmax_searches=3\nreader_model="from-file"\n')
        settings = Settings.load(
            config_file=file,
            environ={"LLGM_MAX_SEARCHES": "2", "LLGM_MAIN_MODEL": "from-env"},
            overrides={"max_searches": 4},
        )
    assert settings.max_searches == 4
    assert settings.main_model == "from-env"
    assert settings.reader_model == "from-file"
    assert settings.max_model_calls == 40
    assert settings.field_sources["max_searches"] == "explicit"
    assert settings.field_sources["reader_model"] == "file"
    assert settings.field_sources["main_model"] == "environment"


def test_independent_instances_and_snapshot_of_environment(monkeypatch):
    """Settings capture their environment without sharing mutable provenance state."""
    env = {"LLGM_MAIN_MODEL": "a"}
    first = Settings.from_env(environ=env)
    env["LLGM_MAIN_MODEL"] = "b"
    second = Settings.from_env(environ=env)
    assert first.main_model == "a" and second.main_model == "b"
    with pytest.raises(TypeError):
        first.field_sources["main_model"] = "altered"


@pytest.mark.parametrize("process_environment", [False, True])
def test_application_settings_coexist_with_integration_environment(
    monkeypatch, process_environment
):
    """Separate environment namespaces cannot enter application values or diagnostics."""
    environment = {
        "LLGM_MAIN_MODEL": "application-model",
        "LLGM_MAX_SEARCHES": "2",
        "LLGM_TEST_OPENAI": "0",
        "LLGM_TEST_APPLICATION_MAIN_MODEL": "integration-model",
        "LLGM_TEST_LONGMEMEVAL_PATH": "/private/local-dataset.json",
    }
    if process_environment:
        for name in tuple(os.environ):
            if name.startswith("LLGM_"):
                monkeypatch.delenv(name)
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        settings = Settings.from_env()
    else:
        settings = Settings.from_env(environ=environment)
    assert settings.main_model == "application-model"
    assert settings.max_searches == 2
    assert settings.field_sources["main_model"] == "environment"
    diagnostics = repr(settings.redacted()) + repr(settings)
    for name, value in environment.items():
        if name.startswith("LLGM_TEST_"):
            assert name not in diagnostics
            if len(value) > 1:
                assert value not in diagnostics


@pytest.mark.parametrize("name", ["test_openai", "test_application_main_model"])
def test_external_environment_names_are_not_application_file_or_override_fields(tmp_path, name):
    """Namespace separation applies to process variables, never unknown TOML or override keys."""
    with pytest.raises(ConfigurationError, match="Unknown LLGM setting"):
        Settings.load(environ={}, overrides={name: "external-value"})
    config = tmp_path / "llgm.toml"
    config.write_text(f'[llgm]\n{name} = "external-value"\n')
    with pytest.raises(ConfigurationError, match="Unknown LLGM setting"):
        Settings.load(config_file=config, environ={})


@pytest.mark.parametrize(
    "env",
    [
        {"LLGM_MAX_SEARCHS": "4"},
        {"LLGM_GRAPH_PROVIDER": "unsupported"},
        {"LLGM_MAX_GRAPH_CALLS": "0"},
        {"LLGM_MAIN_MODLE": "typo"},
        {"LLGM_TESTING_MODE": "1"},
        {"LLGM_REPLICA_COUNT": "2"},
        {"LLGM_SEARCH_POLICY": "adaptive"},
        {"LLGM_MAX_SEARCHES": "one"},
        {"LLGM_MAX_SEARCHES": "0"},
        {"LLGM_BLOB_BACKEND": "s3"},
        {"LLGM_BLOB_URI": "s3://bucket/key"},
        {"LLGM_METADATA_BACKEND": "postgres", "LLGM_DATABASE_URL": "postgresql://db/llgm"},
    ],
)
def test_invalid_configuration_fails_without_fallback(env):
    """Invalid or incompatible settings fail instead of selecting a fallback."""
    with pytest.raises(ConfigurationError):
        Settings.from_env(environ=env)


def test_secrets_redacted_from_descriptor_and_repr():
    """Credential-bearing URLs remain redacted in public settings representations."""
    settings = Settings.from_env(
        environ={
            "LLGM_METADATA_BACKEND": "postgres",
            "LLGM_RETRIEVER_BACKEND": "postgres_fts",
            "LLGM_DATABASE_URL": "postgresql://user:SECRET@db/llgm?token=PRIVATE",
            "LLGM_MAIN_BASE_URL": "https://u:SECRET@api.example/v1?key=PRIVATE",
            "LLGM_GRAPH_BASE_URL": "https://u:SECRET@maintenance.example/v1?key=PRIVATE",
        }
    )
    assert "SECRET" not in repr(settings)
    assert "PRIVATE" not in repr(settings.redacted())
    assert "SECRET" not in repr(settings.redacted())


def test_loading_settings_creates_no_directories(tmp_path):
    """Resolving default storage paths performs no filesystem writes."""
    path = tmp_path / "not-created"
    settings = Settings.from_env(environ={}, overrides={"workspace_path": str(path)})
    assert not path.exists()
    assert settings.blob_uri.startswith("file:///")


def test_context_allowance_is_configurable_with_recorded_precedence(tmp_path):
    """Explicit context limits override file and environment values without changing peers."""
    config = tmp_path / "llgm.toml"
    config.write_text("[llgm]\nmax_context_tokens = 48000\n")
    settings = Settings.load(
        config_file=config,
        environ={"LLGM_MAX_CONTEXT_TOKENS": "32000"},
        overrides={"max_context_tokens": 64000},
    )
    assert settings.max_context_tokens == 64000
    assert settings.field_sources["max_context_tokens"] == "explicit"
    assert settings.max_bundle_tokens == 8000


@pytest.mark.parametrize(
    "values",
    [
        {"max_seed_nodes": 0},
        {"retrieval_k": 41},
        {"max_seed_nodes": 4, "retrieval_k": 3},
        {"max_concurrency": True},
        {"max_journal_bytes": 0},
    ],
)
def test_node_execution_configuration_rejects_invalid_admission(values):
    """Seed, concurrency and journal admission cannot be disabled by malformed scalar settings."""
    with pytest.raises(ConfigurationError):
        Settings(**values)


@pytest.mark.parametrize("name", ["LLGM_NODE_REPL_IMAGE", "LLGM_REPL_DOCKER_IMAGE"])
def test_retired_docker_setting_is_not_silently_ignored(name):
    """Old image configuration fails explicitly after switching to the DSPy sandbox."""
    with pytest.raises(ConfigurationError):
        Settings.from_env(environ={name: "python:3.12-slim"})
    settings = Settings.from_env(environ={"LLGM_MAX_CONCURRENCY": "2"})
    assert settings.max_concurrency == 2
    assert settings.field_sources["max_concurrency"] == "environment"


def test_maintenance_settings_are_independent_of_reader_settings():
    """Environment model roles retain separate identities, credentials, and call limits."""
    settings = Settings.from_env(
        environ={
            "LLGM_READER_MODEL": "reader",
            "LLGM_GRAPH_MODEL": "organizer",
            "LLGM_GRAPH_PROVIDER": "anthropic",
            "LLGM_GRAPH_API_KEY_ENV": "ORGANIZER_KEY",
            "LLGM_MAX_GRAPH_CALLS": "2",
        }
    )
    assert settings.reader_model == "reader"
    assert settings.graph_model == "organizer"
    assert settings.graph_provider == "anthropic"
    assert settings.graph_api_key_env == "ORGANIZER_KEY"
    assert settings.max_graph_calls == 2
    assert settings.field_sources["graph_model"] == "environment"


@pytest.mark.parametrize(
    "name",
    [
        "root_model",
        "sidecar_model",
        "maintenance_model",
        "max_sidecar_calls",
        "max_maintenance_calls",
    ],
)
def test_legacy_role_configuration_is_rejected_in_every_layer(tmp_path, name):
    """Old names fail explicitly instead of coexisting with independent new model settings."""
    with pytest.raises(ConfigurationError, match="Unknown LLGM setting"):
        Settings.from_env(environ={"LLGM_" + name.upper(): "obsolete"})
    with pytest.raises(ConfigurationError, match="Unknown LLGM setting"):
        Settings.load(environ={}, overrides={name: "obsolete"})
    config = tmp_path / "old.toml"
    config.write_text(f'[llgm]\n{name} = "obsolete"\n')
    with pytest.raises(ConfigurationError, match="Unknown LLGM setting"):
        Settings.load(environ={}, config_file=config)
