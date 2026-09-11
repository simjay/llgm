"""CLI dispatch, explicit execution gates and durable local experiment outputs."""

import json

import pytest

from llgm.cli import main
from llgm.evaluation.artifacts import read_jsonl, write_json, write_jsonl
from llgm.evaluation.matrix import REQUIRED_ARMS, matrix_template
from llgm.evaluation.prepare import diagnostic_dataset, file_sha256, prepare


@pytest.fixture
def prepared_cli(tmp_path):
    """Build isolated offline histories and a complete, unresolved matrix on disk."""
    dataset = tmp_path / "input.json"
    write_json(dataset, diagnostic_dataset())
    directory = tmp_path / "prepared"
    prepare(dataset, directory, target=5, dataset_kind="controlled-offline-fixture")
    matrix = tmp_path / "matrix.json"
    write_json(matrix, matrix_template())
    return directory, matrix


def test_offline_smoke_cli_outputs_json_and_complete_artifacts(tmp_path, capsys):
    """The public smoke command returns parseable results and auditable local artifacts."""
    output = tmp_path / "smoke"
    assert main(["offline-smoke", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["benchmark_result"] is False
    assert result["physical_embedding_requests"] == 0
    assert result["arms"]["B-S"]["completed"] == 5
    assert len(read_jsonl(output / "run/B-S/predictions.jsonl")) == 5
    for name in ("manifest.json", "metrics.json", "decision.md", "usage.jsonl", "traces.jsonl"):
        assert (output / "run/B-S" / name).is_file()


def test_template_and_controlled_fixture_commands_refuse_overwrite(tmp_path, capsys):
    """Generated configurations and fixtures are usable and cannot replace existing files."""
    for command, name in (("matrix-template", "matrix.json"), ("controlled-fixtures", "data.json")):
        path = tmp_path / name
        argv = ["experiment", command, "--output", str(path)]
        assert main(argv) == 0
        result = json.loads(capsys.readouterr().out)
        before = path.read_bytes()
        assert main(argv) == 2
        assert "already exists" in capsys.readouterr().err
        assert path.read_bytes() == before
        if command == "matrix-template":
            assert result["arms"] == 12
            assert {arm["id"] for arm in json.loads(before)["arms"]} == set(REQUIRED_ARMS)
        else:
            assert result["benchmark"] is False
            assert result["cases"] == len(json.loads(before)) == 60


@pytest.mark.parametrize("tokenizer", ["word", "character"])
def test_prepare_cli_preserves_selection_and_tokenizer(tmp_path, capsys, tokenizer):
    """Preparation forwards declared selection settings and reports its actual tokenizer."""
    dataset = tmp_path / "data.json"
    write_json(dataset, diagnostic_dataset())
    output = tmp_path / "prepared"
    assert (
        main(
            [
                "experiment",
                "prepare",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--target",
                "3",
                "--smoke-count",
                "2",
                "--seed",
                "91",
                "--dataset-revision",
                "local-fixture-v1",
                "--dataset-kind",
                "controlled-development",
                "--diagnostic-tokenizer",
                tokenizer,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert len(result["selection"]["development_ids"]) == 3
    assert len(result["selection"]["smoke_ids"]) == 2
    assert result["selection"]["seed"] == 91
    assert result["tokenizer"]["mode"] == tokenizer
    assert result["tokenizer"]["benchmark_compatible"] is False
    assert result["dataset"]["revision"] == "local-fixture-v1"
    assert result["dataset"]["sha256"] == file_sha256(dataset)


def test_prepare_reports_blocked_history_isolation_as_nonzero(tmp_path, capsys):
    """A valid dataset with inseparable histories produces a reviewable blocked manifest."""
    raw = diagnostic_dataset()
    for row in raw:
        row["haystack_sessions"] = raw[0]["haystack_sessions"]
    dataset = tmp_path / "shared.json"
    output = tmp_path / "prepared"
    write_json(dataset, raw)
    assert (
        main(
            [
                "experiment",
                "prepare",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["selection"]["status"] == "blocked_history_isolation"
    assert (output / "prepared.json").is_file()
    assert read_jsonl(output / "queries.jsonl") == []


@pytest.mark.parametrize("diagnostic,expected", [(False, 2), (True, 0)])
def test_preflight_cli_reports_full_matrix_and_writes_same_json(
    tmp_path, capsys, prepared_cli, diagnostic, expected
):
    """Preflight is read-only and its file matches stdout, including blocked backend reasons."""
    prepared, matrix = prepared_cli
    output = tmp_path / "preflight.json"
    argv = [
        "experiment",
        "preflight",
        "--prepared",
        str(prepared),
        "--matrix",
        str(matrix),
        "--output",
        str(output),
    ]
    if diagnostic:
        argv.append("--diagnostic")
    assert main(argv) == expected
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text())
    assert result["network_calls"] == 0
    assert len(result["arms"]) == 12
    assert result["benchmark_ready"] is False
    assert result["ready"] is diagnostic
    assert not (prepared / "indexes").exists()


@pytest.mark.parametrize("command", ["run", "retrieve"])
def test_live_dispatch_requires_execute_before_reading_configuration(tmp_path, capsys, command):
    """Omitting the execution gate refuses dispatch even before missing inputs are opened."""
    output = tmp_path / "run"
    assert (
        main(
            [
                "experiment",
                command,
                "--prepared",
                str(tmp_path / "missing-prepared"),
                "--matrix",
                str(tmp_path / "missing-matrix"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "--execute" in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_diagnostic_run_dispatches_only_b_s_without_execute(tmp_path, capsys, prepared_cli):
    """The offline exception to the live gate executes only the declared deterministic arm."""
    prepared, matrix = prepared_cli
    output = tmp_path / "run"
    assert (
        main(
            [
                "experiment",
                "run",
                "--prepared",
                str(prepared),
                "--matrix",
                str(matrix),
                "--output",
                str(output),
                "--diagnostic",
                "--arms",
                "B-S",
                "--split",
                "smoke",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert set(result["arms"]) == {"B-S"}
    assert result["physical_embedding_requests"] == 0
    assert result["benchmark_result"] is False
    assert (output / "B-S/predictions.jsonl").is_file()
    assert not (output / "C-S").exists()


def test_retrieve_cli_forwards_explicit_subset_and_split(
    tmp_path, capsys, monkeypatch, prepared_cli
):
    """Authorized retrieval forwards paths and selection to the async runner without generation."""
    prepared, matrix = prepared_cli
    output = tmp_path / "run"
    calls = []

    async def record_dispatch(prepared_arg, output_arg, matrix_arg, **kwargs):
        """Capture orchestration arguments without invoking any retrieval backend."""
        calls.append((prepared_arg, output_arg, matrix_arg, kwargs))
        return {"benchmark_result": False, "generation_calls": 0}

    monkeypatch.setattr(
        "llgm.evaluation.retrieval_diagnostic.run_retrieval_diagnostic", record_dispatch
    )
    assert (
        main(
            [
                "experiment",
                "retrieve",
                "--prepared",
                str(prepared),
                "--matrix",
                str(matrix),
                "--output",
                str(output),
                "--execute",
                "--backends",
                "B",
                "H",
                "--split",
                "evaluation",
            ]
        )
        == 0
    )
    assert calls == [
        (
            prepared,
            output,
            matrix_template(),
            {"backends": ["B", "H"], "split": "evaluation", "required_backends": None},
        )
    ]
    assert json.loads(capsys.readouterr().out)["generation_calls"] == 0


def test_download_cli_forwards_pins_and_reports_actual_file_hash(tmp_path, capsys, monkeypatch):
    """Download dispatch preserves supplied pins and hashes returned bytes, without a network test."""
    output = tmp_path / "download.json"
    revision, checksum = "a" * 40, "b" * 64
    calls = []

    def write_download(path, **kwargs):
        """Stand in for the download boundary with a small local JSON file."""
        calls.append((path, kwargs))
        write_json(path, [])
        return path

    monkeypatch.setattr("llgm.evaluation.prepare.download_cleaned_s", write_download)
    assert (
        main(
            [
                "experiment",
                "download",
                "--output",
                str(output),
                "--dataset-revision",
                revision,
                "--sha256",
                checksum,
            ]
        )
        == 0
    )
    assert calls == [(output, {"revision": revision, "expected_sha256": checksum})]
    assert json.loads(capsys.readouterr().out)["sha256"] == file_sha256(output)


def test_export_cli_preserves_failed_case_denominator(tmp_path, capsys):
    """Export writes every question, including empty failed answers, without running a judge."""
    source, output = tmp_path / "predictions.jsonl", tmp_path / "official.jsonl"
    write_jsonl(
        source,
        [
            {"case_id": "ok", "hypothesis": "amber", "status": "completed"},
            {"case_id": "failed", "hypothesis": "", "status": "failed"},
        ],
    )
    assert (
        main(
            [
                "experiment",
                "export-official",
                "--predictions",
                str(source),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["official_judge_invoked"] is False
    assert read_jsonl(output) == [
        {"question_id": "ok", "hypothesis": "amber"},
        {"question_id": "failed", "hypothesis": ""},
    ]


@pytest.mark.parametrize("payload", ["{invalid json", "[]", "null"])
def test_malformed_matrix_returns_configuration_error(tmp_path, capsys, payload):
    """Malformed JSON and non-object matrices produce a controlled error without a traceback."""
    matrix = tmp_path / "matrix.json"
    matrix.write_text(payload)
    assert (
        main(
            [
                "experiment",
                "preflight",
                "--prepared",
                str(tmp_path / "missing"),
                "--matrix",
                str(matrix),
            ]
        )
        == 2
    )
    result = capsys.readouterr()
    assert "ConfigurationError" in result.err
    assert result.out == ""


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "limit",
    ["max_case_arm_runs", "max_generation_calls", "max_embedding_requests", "timeout_seconds"],
)
def test_nonfinite_overall_caps_are_rejected_before_dispatch(tmp_path, capsys, value, limit):
    """Nonfinite JSON resource caps cannot disable experiment admission limits."""
    matrix = matrix_template()
    matrix["overall"][limit] = value
    path = tmp_path / "matrix.json"
    write_json(path, matrix)
    assert (
        main(
            [
                "experiment",
                "preflight",
                "--prepared",
                str(tmp_path / "missing"),
                "--matrix",
                str(path),
            ]
        )
        == 2
    )
    assert f"overall.{limit}" in capsys.readouterr().err


@pytest.mark.parametrize(
    "path,value",
    [
        (("arms",), None),
        (("arms",), [1]),
        (("arms", 0, "id"), []),
        (("passages",), None),
        (("dense",), []),
        (("tokenizer",), []),
        (("tokenizer",), {"local_path": "only-one-pin"}),
        (("tokenizer", "revision"), ""),
        (("models",), []),
        (("models", "root"), []),
        (("models", "root", "model"), 42),
        (("models", "root", "provider"), []),
        (("models", "root", "api_key_env"), 42),
        (("budgets",), []),
        (("overall",), None),
    ],
)
def test_wrongly_typed_matrix_fields_return_controlled_errors(tmp_path, capsys, path, value):
    """Wrong JSON field shapes fail validation before indexing, provider checks or Python type errors."""
    matrix = matrix_template()
    target = matrix
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    filename = tmp_path / "matrix.json"
    write_json(filename, matrix)
    assert (
        main(
            [
                "experiment",
                "preflight",
                "--prepared",
                str(tmp_path / "missing"),
                "--matrix",
                str(filename),
            ]
        )
        == 2
    )
    result = capsys.readouterr()
    assert "ConfigurationError" in result.err
    assert result.out == ""


@pytest.mark.parametrize(
    "argv", [[], ["experiment"], ["experiment", "retrieve", "--backends", "X"]]
)
def test_argument_errors_exit_before_creating_files(argv, tmp_path, monkeypatch):
    """Missing commands and invalid enum values stop at argument parsing."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


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
