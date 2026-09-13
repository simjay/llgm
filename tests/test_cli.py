"""Public conversation CLI validation and serialization contracts."""

import json

import pytest

from llgm.cli import main


def test_ask_cli_forwards_scope_time_and_returns_canonical_evidence(tmp_path, capsys, monkeypatch):
    """The CLI uses the real LLGM facade with an explicit replay seam and no provider dispatch."""
    from contextlib import asynccontextmanager

    from llgm import LLGM, Conversation, Workspace
    from tests.node_support import Models, ReplayFactory, node_context

    received = []

    @asynccontextmanager
    async def configured(settings):
        """Replace only settings-owned resources with a real temporary workspace and scripted clients."""
        received.append(settings)
        models = Models()
        async with Workspace.open(settings.workspace_path) as workspace:
            app = LLGM(
                workspace,
                models.main,
                models.reader,
                graph_model=models.reader,
                interpreter_factory=ReplayFactory(),
            )
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
                "--read-only",
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
    initial = node_context(received[1].child_requests[0])
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

    environment = {"LLGM_MAIN_MODEL": "exported-main", "OPENAI_API_KEY": "exported-key-fixture"}
    monkeypatch.setattr(os, "environ", environment)
    env_file = tmp_path / "local.env"
    env_file.write_text(
        "LLGM_MAIN_MODEL=local-main\n"
        "LLGM_READER_MODEL=local-reader\n"
        "LLGM_MAX_SEARCHES=2\n"
        "OPENAI_API_KEY=local-key-fixture\n"
        "ANTHROPIC_API_KEY=local-reader-key-fixture\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "llgm.toml"
    config_file.write_text(
        '[llgm]\nmax_searches=3\nworkspace_path="toml-workspace"\n', encoding="utf-8"
    )
    received = []

    async def answer(question, **kwargs):
        """Return a completed synthetic answer without model or sandbox execution."""
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
        assert os.environ["ANTHROPIC_API_KEY"] == "local-reader-key-fixture"
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
    assert settings.main_model == "exported-main"
    assert settings.reader_model == "local-reader"
    assert settings.max_searches == 3
    assert settings.workspace_path == str(workspace)
    assert settings.field_sources["main_model"] == "environment"
    assert settings.field_sources["reader_model"] == "environment"
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


def test_view_cli_rejects_missing_workspace_without_creating_it(tmp_path, capsys, monkeypatch):
    """A typo in the workspace path cannot silently create a new empty graph."""
    import os

    monkeypatch.setattr(os, "environ", {})
    missing = tmp_path / "absent"
    assert main(["view", "--workspace", str(missing), "--no-browser"]) == 2
    assert "Workspace does not exist" in capsys.readouterr().err
    assert not missing.exists()


def test_view_cli_rejects_custom_backend_instead_of_falling_back(tmp_path, capsys, monkeypatch):
    """The CLI directs custom search users to their application's evidence factory."""
    import os

    monkeypatch.setattr(os, "environ", {"LLGM_RETRIEVER_BACKEND": "custom"})
    assert main(["view", "--workspace", str(tmp_path), "--no-browser"]) == 2
    assert "GraphViewer.from_application" in capsys.readouterr().err


def test_view_cli_uses_storage_without_models(tmp_path, capsys, monkeypatch):
    """The command forwards viewer settings without constructing application model clients."""
    import asyncio
    import os

    from llgm import LLGM, Workspace
    from llgm.viewer import GraphViewer

    monkeypatch.setattr(os, "environ", {})

    async def initialize():
        """Create a genuine empty workspace before exercising the command."""
        async with Workspace.open(tmp_path):
            pass

    asyncio.run(initialize())
    received = []

    async def enter(viewer):
        """Replace only the listening socket while recording real CLI configuration."""
        received.append((viewer.workspace.database_path, viewer.conversation_id, viewer.port))
        viewer.url = "http://127.0.0.1:8765/test/"
        return viewer

    async def exit_viewer(viewer, *args):
        """Finish the no-socket viewer context."""

    async def serve(viewer):
        """Return immediately after verifying that an entered viewer is served."""
        assert viewer.url

    def no_models(*args, **kwargs):
        """Fail if viewing a workspace tries to allocate answer-model clients."""
        raise AssertionError("Viewer allocated models")

    monkeypatch.setattr(GraphViewer, "__aenter__", enter)
    monkeypatch.setattr(GraphViewer, "__aexit__", exit_viewer)
    monkeypatch.setattr(GraphViewer, "serve_forever", serve)
    monkeypatch.setattr(LLGM, "from_settings", no_models)
    assert (
        main(
            [
                "view",
                "--workspace",
                str(tmp_path),
                "--conversation-id",
                "atlas",
                "--port",
                "0",
                "--no-browser",
            ]
        )
        == 0
    )
    assert received == [(str(tmp_path / "metadata.sqlite3"), "atlas", 0)]
    assert "http://127.0.0.1:8765/test/" in capsys.readouterr().out
