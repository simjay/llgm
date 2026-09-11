"""Explicit local environment loading and provider credential handoff contracts."""

import os
import sys
from types import SimpleNamespace

import pytest

from llgm import Settings, load_env_file
from llgm.core.errors import ConfigurationError
from llgm.models.hosted import create_model


def test_env_file_accepts_assignments_quotes_and_comments(tmp_path):
    """Supported dotenv syntax preserves quoted content and ordinary unquoted values."""
    path = tmp_path / "local.env"
    path.write_text(
        "# Local synthetic configuration\n"
        "\n"
        "export LLGM_ROOT_MODEL = root-fixture  # selected model\n"
        "LLGM_SIDECAR_MODEL='sidecar fixture' # inline comment\n"
        'QUOTED="spaces and # literal hash"\n'
        "HASH=token#suffix\n"
        "LEADING_HASH=#literal\n"
        "EMPTY_COMMENT= # comment\n"
        "EMPTY=\n"
        'EMPTY_QUOTED=""\n'
        "UNICODE='café'\n",
        encoding="utf-8",
    )
    environment = {}

    assert load_env_file(path, environ=environment) is None

    assert environment == {
        "LLGM_ROOT_MODEL": "root-fixture",
        "LLGM_SIDECAR_MODEL": "sidecar fixture",
        "QUOTED": "spaces and # literal hash",
        "HASH": "token#suffix",
        "LEADING_HASH": "#literal",
        "EMPTY_COMMENT": "",
        "EMPTY": "",
        "EMPTY_QUOTED": "",
        "UNICODE": "café",
    }


def test_env_file_does_not_evaluate_shell_or_escape_syntax(tmp_path, monkeypatch):
    """Values remain literal even when they resemble interpolation or executable shell text."""
    path = tmp_path / "literal.env"
    payload = r"$HOME ${TOKEN} $(touch should-not-exist) `touch other-file` \n \t"
    path.write_text(
        f"UNQUOTED={payload}\nSINGLE='{payload}'\nDOUBLE=\"{payload}\"\n",
        encoding="utf-8",
    )
    environment = {"TOKEN": "exported-fixture"}
    monkeypatch.chdir(tmp_path)

    load_env_file(path, environ=environment)

    assert all(environment[key] == payload for key in ("UNQUOTED", "SINGLE", "DOUBLE"))
    assert tuple(tmp_path.iterdir()) == (path,)


def test_env_file_preserves_existing_values_including_empty_strings(tmp_path):
    """An exported empty value is intentional and cannot be replaced by a local file."""
    path = tmp_path / "local.env"
    path.write_text("PRESENT=file-value\nEMPTY=file-value\nNEW=local-value\n", encoding="utf-8")
    environment = {"PRESENT": "exported-value", "EMPTY": ""}

    load_env_file(path, environ=environment)

    assert environment == {"PRESENT": "exported-value", "EMPTY": "", "NEW": "local-value"}


def test_env_file_updates_only_the_requested_environment(tmp_path, monkeypatch):
    """A supplied mapping stays isolated while an omitted mapping loads process configuration."""
    path = tmp_path / "local.env"
    path.write_text("LLGM_ROOT_MODEL=root-fixture\n", encoding="utf-8")
    process_environment = {}
    monkeypatch.setattr(os, "environ", process_environment)
    separate_environment = {}

    load_env_file(path, environ=separate_environment)
    assert process_environment == {}
    assert separate_environment == {"LLGM_ROOT_MODEL": "root-fixture"}

    load_env_file(path)
    settings = Settings.from_env()
    assert settings.root_model == "root-fixture"
    assert settings.field_sources["root_model"] == "environment"


@pytest.mark.parametrize(
    "invalid_line",
    [
        "MISSING_EQUALS synthetic-private-value",
        "9INVALID=synthetic-private-value",
        "INVALID-NAME=synthetic-private-value",
        "UNTERMINATED='synthetic-private-value",
        'UNTERMINATED="synthetic-private-value',
        'TRAILING="synthetic-private-value" unexpected',
        "FIRST=synthetic-private-value",
        "NULL=synthetic-private-value\x00",
    ],
)
def test_invalid_env_file_is_atomic_and_does_not_disclose_values(tmp_path, invalid_line):
    """Malformed entries and duplicates fail before any earlier assignment changes the target."""
    path = tmp_path / "invalid.env"
    path.write_text(f"FIRST=valid-fixture\n{invalid_line}\n", encoding="utf-8")
    environment = {"PRESENT": "existing-fixture"}

    with pytest.raises(ConfigurationError) as error:
        load_env_file(path, environ=environment)

    assert environment == {"PRESENT": "existing-fixture"}
    assert "line 2" in str(error.value)
    assert "synthetic-private-value" not in str(error.value)
    assert invalid_line not in str(error.value)


@pytest.mark.parametrize("file_kind", ["missing", "directory", "invalid-utf8"])
def test_unreadable_env_file_raises_configuration_error_without_mutation(tmp_path, file_kind):
    """Explicit paths must resolve to readable UTF-8 files before environment changes occur."""
    path = tmp_path / "unreadable.env"
    if file_kind == "directory":
        path.mkdir()
    elif file_kind == "invalid-utf8":
        path.write_bytes(b"KEY=synthetic-private-value\xff\n")
    environment = {"PRESENT": "existing-fixture"}

    with pytest.raises(ConfigurationError) as error:
        load_env_file(path, environ=environment)

    assert environment == {"PRESENT": "existing-fixture"}
    assert "synthetic-private-value" not in str(error.value)


def test_settings_do_not_discover_local_or_parent_dotenv_files(tmp_path, monkeypatch):
    """Settings remain independent of nearby dotenv files unless loading is explicitly requested."""
    (tmp_path / ".env").write_text("LLGM_ROOT_MODEL=parent-fixture\n", encoding="utf-8")
    child = tmp_path / "child"
    child.mkdir()
    (child / ".env").write_text("LLGM_ROOT_MODEL=child-fixture\n", encoding="utf-8")
    monkeypatch.chdir(child)
    monkeypatch.setattr(os, "environ", {})

    settings = Settings.from_env()

    assert settings.root_model is None
    assert os.environ == {}


@pytest.mark.parametrize(
    "provider,sdk_name,credential_name,custom_name",
    [
        ("openai", "AsyncOpenAI", "OPENAI_API_KEY", None),
        ("anthropic", "AsyncAnthropic", "ANTHROPIC_API_KEY", None),
        ("openai", "AsyncOpenAI", "FIXTURE_PROVIDER_KEY", "FIXTURE_PROVIDER_KEY"),
    ],
)
def test_env_file_credentials_reach_sdk_construction_without_entering_settings(
    tmp_path, monkeypatch, provider, sdk_name, credential_name, custom_name
):
    """Loaded credentials cross the real adapter factory boundary without appearing in diagnostics."""
    path = tmp_path / "provider.env"
    path.write_text(
        f"{credential_name}=synthetic-private-value\n"
        "LLGM_ROOT_MODEL=root-fixture\n"
        f"LLGM_ROOT_PROVIDER={provider}\n"
        + (f"LLGM_ROOT_API_KEY_ENV={custom_name}\n" if custom_name else ""),
        encoding="utf-8",
    )
    monkeypatch.setattr(os, "environ", {})
    received = []

    def sdk_client(**kwargs):
        """Stand in for optional SDK initialization without opening a network transport."""
        received.append(kwargs.get("api_key", os.environ.get(credential_name)))
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, provider, SimpleNamespace(**{sdk_name: sdk_client}))
    load_env_file(path)
    settings = Settings.from_env()
    client = create_model(
        settings.root_provider,
        settings.root_model,
        api_key_env=settings.root_api_key_env,
    )

    assert received == ["synthetic-private-value"]
    assert "synthetic-private-value" not in repr(settings)
    assert "synthetic-private-value" not in repr(settings.redacted())
    assert "synthetic-private-value" not in repr(client.descriptor())
