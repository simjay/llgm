"""Public conversation CLI validation and serialization contracts."""

import json

import pytest

from llgm.cli import main


def test_ask_cli_forwards_scope_time_and_returns_canonical_evidence(tmp_path, capsys, monkeypatch):
    """The CLI uses the real LLGM facade with an explicit replay seam and no provider dispatch."""
    from contextlib import asynccontextmanager

    from llgm import LLGM, Conversation, Workspace
    from tests.node_support import Models, ReplayFactory

    received = []

    @asynccontextmanager
    async def configured(settings):
        """Replace only settings-owned resources with a real temporary workspace and scripted clients."""
        received.append(settings)
        models = Models()
        async with Workspace.open(settings.workspace_path) as workspace:
            app = LLGM(workspace, models.root, models.sidecar, repl_factory=ReplayFactory())
            await app.ingest(
                Conversation.from_turns(
                    [{"role": "user", "turn_id": "t", "text": "Orion region is eu-west-1"}],
                    node_id="n",
                ),
                organize=False,
            )
            yield app
        received.append(models)

    monkeypatch.setattr(LLGM, "from_settings", configured)
    assert (
        main(
            [
                "ask",
                "Orion region",
                "--workspace",
                str(tmp_path / "memory"),
                "--scope",
                '{"env":"production"}',
                "--as-of-ms",
                "0",
                "--query-date",
                "original date",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["references"][0]["node_id"] == "n"
    initial = json.loads(received[1].child_requests[0].messages[1].content)
    assert initial["query_scope"] == {"env": "production"}
    assert initial["journal"]["as_of_ms"] == 0
    assert initial["query_date"] == "original date"


def test_ask_cli_rejects_nonobject_scope_before_opening_resources(capsys, monkeypatch):
    """Invalid query selectors cannot create provider clients or a workspace."""
    from llgm import LLGM

    def forbidden(*args, **kwargs):
        """Fail if argument validation crosses into resource allocation."""
        raise AssertionError("No resources should open")

    monkeypatch.setattr(LLGM, "from_settings", forbidden)
    assert main(["ask", "question", "--scope", "[]"]) == 2
    assert "--scope must be a JSON object" in capsys.readouterr().err


def test_ask_cli_loads_explicit_env_file_before_settings(tmp_path, capsys, monkeypatch):
    """Explicit files supply models and credentials while exported, TOML and CLI values retain priority."""
    import os
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from llgm import LLGM

    environment = {"LLGM_ROOT_MODEL": "exported-root", "OPENAI_API_KEY": "exported-key-fixture"}
    monkeypatch.setattr(os, "environ", environment)
    env_file = tmp_path / "local.env"
    env_file.write_text(
        "LLGM_ROOT_MODEL=local-root\n"
        "LLGM_SIDECAR_MODEL=local-sidecar\n"
        "LLGM_MAX_SEARCHES=2\n"
        "OPENAI_API_KEY=local-key-fixture\n"
        "ANTHROPIC_API_KEY=local-sidecar-key-fixture\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "llgm.toml"
    config_file.write_text(
        '[llgm]\nmax_searches=3\nworkspace_path="toml-workspace"\n', encoding="utf-8"
    )
    received = []

    async def answer(question, **kwargs):
        """Return a completed synthetic answer without model or Docker execution."""
        return SimpleNamespace(
            answer="synthetic-answer",
            status="completed",
            references=[],
            evidence=SimpleNamespace(unresolved=[]),
            usage={},
        )

    @asynccontextmanager
    async def configured(settings):
        """Capture final configuration at the application resource boundary."""
        received.append(settings)
        assert os.environ["OPENAI_API_KEY"] == "exported-key-fixture"
        assert os.environ["ANTHROPIC_API_KEY"] == "local-sidecar-key-fixture"
        yield SimpleNamespace(answer=answer)

    monkeypatch.setattr(LLGM, "from_settings", configured)
    workspace = tmp_path / "cli-workspace"
    assert (
        main(
            [
                "ask",
                "question",
                "--env-file",
                str(env_file),
                "--config",
                str(config_file),
                "--workspace",
                str(workspace),
            ]
        )
        == 0
    )
    settings = received[0]
    assert settings.root_model == "exported-root"
    assert settings.sidecar_model == "local-sidecar"
    assert settings.max_searches == 3
    assert settings.workspace_path == str(workspace)
    assert settings.field_sources["root_model"] == "environment"
    assert settings.field_sources["sidecar_model"] == "environment"
    assert settings.field_sources["max_searches"] == "file"
    assert settings.field_sources["workspace_path"] == "explicit"
    output = capsys.readouterr()
    assert json.loads(output.out)["answer"] == "synthetic-answer"
    assert "key-fixture" not in output.out + output.err
    assert not workspace.exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_ask_cli_env_file_errors_are_explicit_and_precede_resources(
    tmp_path, capsys, monkeypatch, explicit
):
    """A nearby dotenv file is ignored unless requested, when malformed content fails before resources."""
    import os

    from llgm import LLGM
    from llgm.core.errors import ConfigurationError

    monkeypatch.setattr(os, "environ", {})
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("KEY=synthetic-private-value\nINVALID ASSIGNMENT\n", encoding="utf-8")
    called = []

    def configured(settings):
        """Record whether resource allocation was reached without constructing actual resources."""
        called.append(settings)
        raise ConfigurationError("synthetic resource boundary")

    monkeypatch.setattr(LLGM, "from_settings", configured)
    argv = ["ask", "question"]
    if explicit:
        argv.extend(["--env-file", str(env_file)])
    assert main(argv) == 2
    assert bool(called) is not explicit
    output = capsys.readouterr()
    assert "ConfigurationError" in output.err
    assert ("line 2" in output.err) is explicit
    assert "synthetic-private-value" not in output.err
    assert os.environ == {}
