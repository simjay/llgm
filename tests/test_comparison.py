"""Frozen comparison, graph-operation ablation and evaluator-isolation contracts."""

import asyncio
import json
from pathlib import Path

import pytest

from llgm.core.errors import ConfigurationError
from llgm.core.types import JournalRef, NodeRef, SourceSpan
from llgm.evaluation.artifacts import read_jsonl, write_json
from llgm.evaluation.comparison import (
    ARMS,
    _covers,
    freeze_comparison,
    load_comparison,
    main,
    run_comparison,
)
from llgm.evaluation.prepare import file_sha256
from llgm.inference.recursive import RecursiveRuntime
from llgm.models import CallableModelClient, ModelResponse, ScriptedModelClient, Usage


@pytest.fixture
def frozen_case(tmp_path):
    """Freeze one independent lookup case with evaluator labels outside runtime evidence."""
    path = tmp_path / "protocol.json"
    freeze_comparison(
        path,
        models={
            role: {"provider": "controlled-test", "model": role + "-v1"}
            for role in ("root", "sidecar")
        },
        distractors=2,
    )
    value = json.loads(path.read_text())
    value["cases"] = value["cases"][:1]
    value["arm_orders"] = {"lookup": list(ARMS)}
    write_json(path, value)
    return path, file_sha256(path)


def controlled_clients(requests, *, fail_first_root=False, waiting=None):
    """Return fixed protocol decisions for storage/accounting tests, without claiming model accuracy."""
    root_calls = 0

    async def root(request):
        """Search then cite observed lookup evidence, or finish the fixed flat-reader answer."""
        nonlocal root_calls
        root_calls += 1
        requests.append(("root", request))
        if waiting is not None:
            waiting.set()
            await asyncio.Event().wait()
        if fail_first_root and root_calls == 1:
            raise RuntimeError("PRIVATE_PROVIDER_BODY")
        payload = json.loads(request.messages[-1].content)
        if "Resolve the task using external evidence" not in request.messages[0].content:
            text = "CEDAR"
        elif "hits" in payload:
            text = json.dumps(
                {
                    "op": "finish",
                    "answer": "CEDAR",
                    "citations": [payload["hits"][0]["id"]],
                    "unresolved": [],
                }
            )
        else:
            text = json.dumps({"op": "search", "query": "Alder approved release key", "k": 3})
        return ModelResponse(text, Usage(7, 3), provider="controlled-test", model="root-v1")

    async def sidecar(request):
        """Compose one admitted passage without reading evaluator fields."""
        requests.append(("sidecar", request))
        first = json.loads(request.messages[-1].content)["evidence"][0]
        return ModelResponse(
            json.dumps(
                {"text": first["text"], "passage_ids": [first["passage_id"]], "unresolved": []}
            ),
            Usage(5, 2),
            provider="controlled-test",
            model="sidecar-v1",
        )

    return (
        CallableModelClient(root, provider="controlled-test", model="root-v1"),
        CallableModelClient(sidecar, provider="controlled-test", model="sidecar-v1"),
    )


def test_comparison_uses_frozen_shared_evidence_and_records_every_arm(tmp_path, frozen_case):
    """All three arms read the same unchanged corpus with declared pins and counted model attempts."""
    requests = []
    models = controlled_clients(requests)
    path, digest = frozen_case
    output = tmp_path / "run"
    result = asyncio.run(run_comparison(path, output, *models, expected_sha256=digest))
    protocol = json.loads((output / "manifest.json").read_text())["protocol"]
    assert protocol["schema_version"] == 3
    assert protocol["evidence_schema"] == "primary-edges-v3"
    assert result["benchmark_result"] is False
    assert result["recorded_case_arm_runs"] == result["planned_case_arm_runs"] == 3
    assert set(result["arms"]) == set(ARMS)
    assert sum(arm["model_calls"] for arm in result["arms"].values()) == len(requests) == 6
    assert all(arm["exact_match_with_support"] == 1 for arm in result["arms"].values())
    cases = read_jsonl(output / "cases.jsonl")
    assert len({row["case_id"] for row in cases}) == 1
    assert result["setup"][0]["maintenance_origin"] == "curated"
    assert result["setup"][0]["maintenance_model_calls"] == 0
    assert (
        len(read_jsonl(output / "predictions.jsonl"))
        == len(read_jsonl(output / "judgments.jsonl"))
        == 3
    )
    recursive_prompts = [
        request.messages[0].content
        for role, request in requests
        if "Resolve the task using external evidence" in request.messages[0].content
    ]
    assert any('"op":"neighbors"' in prompt for prompt in recursive_prompts)
    assert any(
        '"op":"neighbors"' not in prompt and "unavailable" in prompt for prompt in recursive_prompts
    )
    assert all(
        "owner_journals" in request.messages[-1].content
        for role, request in requests
        if role == "sidecar"
    )


def test_changed_protocol_and_wrong_model_pins_fail_before_calls(tmp_path, frozen_case):
    """An unpinned cohort mutation or substituted client cannot silently change the comparison."""
    path, digest = frozen_case
    requests = []
    models = controlled_clients(requests)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ConfigurationError, match="SHA256"):
        asyncio.run(run_comparison(path, tmp_path / "hash-run", *models, expected_sha256=digest))
    models[0].model = "different-model"
    with pytest.raises(ConfigurationError, match="model pin"):
        asyncio.run(
            run_comparison(path, tmp_path / "model-run", *models, expected_sha256=file_sha256(path))
        )
    assert requests == []


def test_legacy_comparison_is_rejected_before_calls(tmp_path, frozen_case):
    """A correctly hashed v1 file remains historical and cannot silently run revised semantics."""
    path, _ = frozen_case
    value = json.loads(path.read_text())
    value["schema_version"] = 1
    value.pop("evidence_schema")
    write_json(path, value)
    original = path.read_bytes()
    requests = []
    with pytest.raises(ConfigurationError, match="Freeze a new v3 protocol"):
        asyncio.run(
            run_comparison(
                path,
                tmp_path / "run",
                *controlled_clients(requests),
                expected_sha256=file_sha256(path),
            )
        )
    assert requests == []
    assert path.read_bytes() == original
    assert not (tmp_path / "run").exists()


def test_new_protocol_preserves_historical_inputs_and_declares_new_graph_contract():
    """Frozen old artifacts stay byte-identical and cannot silently run under new semantics."""
    directory = Path(__file__).resolve().parents[1] / "experiments"
    for version, digest in (
        (1, "148549682d7ed456fca86c18c7a1848c33057e57475228abeaa05cff1c5c9000"),
        (2, "6b406e2172762a5721e2660d314b5a22f857114780461eb61d65e7d92c256971"),
    ):
        historical = directory / f"architecture_comparison_v{version}.json"
        assert file_sha256(historical) == digest
        with pytest.raises(ConfigurationError, match="Freeze a new v3 protocol"):
            load_comparison(historical, digest)
    old = json.loads((directory / "architecture_comparison_v2.json").read_text())
    path = directory / "architecture_comparison_v3.json"
    current = load_comparison(path, file_sha256(path))
    for key in (
        "models",
        "budget",
        "seed",
        "arm_orders",
        "arms",
        "answer_format",
        "max_depth",
        "max_steps",
        "max_operations",
        "passage_chars",
    ):
        assert current[key] == old[key], key
    aliases = next(case for case in current["cases"] if case["case_id"] == "aliases")
    assert len(aliases["edges"]) == 2 and aliases["journals"] == []
    assert "not the LLGM Python" in current["conditions"]["recursion"]


def test_gold_changes_scores_without_entering_model_inputs(tmp_path, frozen_case):
    """Evaluator-only answer changes affect judgments while provider requests retain identical evidence."""
    path, _ = frozen_case
    value = json.loads(path.read_text())
    value["cases"][0]["gold"]["answer"] = "EVALUATOR_ONLY_SECRET"
    write_json(path, value)
    requests = []
    result = asyncio.run(
        run_comparison(
            path, tmp_path / "run", *controlled_clients(requests), expected_sha256=file_sha256(path)
        )
    )
    assert all(arm["exact_match"] == 0 for arm in result["arms"].values())
    serialized = json.dumps(
        [[message.content for message in request.messages] for _, request in requests]
    )
    assert "EVALUATOR_ONLY_SECRET" not in serialized
    assert "required_sources" not in serialized


def test_failed_call_remains_in_scores_and_usage(tmp_path, frozen_case):
    """One failed provider attempt cannot disappear from case denominators or token accounting."""
    path, digest = frozen_case
    requests = []
    result = asyncio.run(
        run_comparison(
            path,
            tmp_path / "run",
            *controlled_clients(requests, fail_first_root=True),
            expected_sha256=digest,
        )
    )
    assert result["recorded_case_arm_runs"] == 3
    assert result["arms"]["flat_single"]["failed"] == 1
    assert result["arms"]["flat_single"]["exact_match"] == 0
    assert result["arms"]["flat_single"]["unknown_usage_calls"] == 1
    assert sum(arm["model_calls"] for arm in result["arms"].values()) == len(requests)
    assert "PRIVATE_PROVIDER_BODY" not in (tmp_path / "run/traces.jsonl").read_text()


def test_cancelled_comparison_preserves_partial_artifacts(tmp_path, frozen_case):
    """Cancellation records the attempted trial and does not score unstarted arms."""
    path, digest = frozen_case
    requests = []

    async def exercise():
        """Interrupt an in-flight counted model call and await durable cleanup."""
        waiting = asyncio.Event()
        task = asyncio.create_task(
            run_comparison(
                path,
                tmp_path / "run",
                *controlled_clients(requests, waiting=waiting),
                expected_sha256=digest,
            )
        )
        try:
            await asyncio.wait_for(waiting.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["status"] == "cancelled"
    assert summary["recorded_case_arm_runs"] == 1
    assert summary["planned_case_arm_runs"] == 3
    assert read_jsonl(tmp_path / "run/predictions.jsonl")[0]["status"] == "cancelled"
    assert (tmp_path / "run/decision.md").is_file()


def test_setup_cancellation_records_unstarted_trials_without_provider_calls(
    tmp_path, frozen_case, monkeypatch
):
    """Cancelling source preparation records its phase while leaving inference denominators unstarted."""
    from llgm.memory.workspace import Workspace

    path, digest = frozen_case
    requests = []

    async def exercise():
        """Interrupt preparation before a source can be published or a model can run."""
        entered = asyncio.Event()

        async def waiting_ingest(self, *args, **kwargs):
            """Hold the setup boundary until the run receives cancellation."""
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(Workspace, "ingest", waiting_ingest)
        task = asyncio.create_task(
            run_comparison(
                path, tmp_path / "run", *controlled_clients(requests), expected_sha256=digest
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["status"] == "cancelled"
    assert summary["failure_phase"] == "setup"
    assert summary["recorded_case_arm_runs"] == 0
    assert summary["planned_case_arm_runs"] == 3
    assert read_jsonl(tmp_path / "run/predictions.jsonl") == []
    assert requests == []
    assert (tmp_path / "run/decision.md").is_file()


def test_navigation_exclusion_is_advertised_and_enforced():
    """Disabled navigation is absent from the prompt and cannot silently return an empty graph."""
    root = ScriptedModelClient(['{"op":"neighbors","node_id":"a","relation":null}'])
    runtime = RecursiveRuntime(root, ScriptedModelClient([]), object(), allow_neighbors=False)
    result = asyncio.run(runtime.answer("Which evidence applies?"))
    assert result.status == "failed"
    assert '"op":"neighbors"' not in root.requests[0].messages[0].content
    assert "neighbors operation is unavailable" in root.requests[0].messages[0].content
    assert result.usage["model_calls"] == 1
    with pytest.raises(ConfigurationError, match="allow_neighbors"):
        RecursiveRuntime(root, root, object(), allow_neighbors="false")


def test_support_requires_full_labeled_range_and_correct_node():
    """Whole-node citations cover matching turns; partial ranges and different nodes do not."""
    target = SourceSpan("a", "t", 0, 20)
    assert _covers(NodeRef("a"), target)
    assert not _covers(NodeRef("another-node"), target)
    assert not _covers(SourceSpan("a", "t", 0, 10), target)
    assert _covers(JournalRef("a", "edit"), JournalRef("a", "edit", 0, 10))
    assert not _covers(JournalRef("a", "edit", 0, 5), JournalRef("a", "edit", 0, 10))


def test_missing_credentials_write_not_run_without_constructing_models(
    tmp_path, monkeypatch, capsys
):
    """An enabled hosted run with unavailable keys records an explicit blocker and never substitutes a fixture."""
    path = tmp_path / "protocol.json"
    frozen = freeze_comparison(
        path,
        models={
            role: {"provider": "openai", "model": role + "-v1"} for role in ("root", "sidecar")
        },
        distractors=0,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        """Fail if the readiness boundary attempts provider construction."""
        pytest.fail("Provider construction must wait for available credentials")

    monkeypatch.setattr("llgm.models.create_model", forbidden)
    assert (
        main(
            [
                "run",
                "--protocol",
                str(path),
                "--sha256",
                frozen["sha256"],
                "--output",
                str(tmp_path / "run"),
                "--execute",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "not_run"
    assert result["model_calls"] == 0
    assert result["planned_case_arm_runs"] == 15
    assert read_jsonl(tmp_path / "run/predictions.jsonl") == []
    assert load_comparison(path, frozen["sha256"])["conditions"]["setup"].startswith("curated")
