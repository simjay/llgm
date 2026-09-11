"""Retrieval-runner contracts with real local indexes and a controlled embedding transport.

These tests substitute external readiness and tokenizer loading to isolate
orchestration. The diagnostic tokenizer remains labeled as such; no test runs
ColBERT or treats these results as benchmark evidence.
"""

import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.evaluation import retrieval_diagnostic as diagnostic
from llgm.evaluation.artifacts import RunArtifacts, read_jsonl, write_json, write_jsonl
from llgm.evaluation.matrix import matrix_template, preflight, validate_matrix
from llgm.evaluation.prepare import diagnostic_dataset, file_sha256, prepare
from llgm.retrieval import DiagnosticTokenizer, SQLiteBM25Retriever


class FixtureTokenizer(DiagnosticTokenizer):
    """Diagnostic word tokenizer with a stable identity for runner integrity contracts."""

    def descriptor(self):
        """Retain the diagnostic label while exposing a deterministic fixture identity."""
        descriptor = super().descriptor()
        return {
            **descriptor,
            "sha256": hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        }


class EmbeddingTransport:
    """Controlled SDK boundary retaining request order, usage and owned cleanup."""

    def __init__(self, state):
        """Share invocation records and failure controls across per-case clients."""
        self.state = state
        self.embeddings = self
        self.closed = False

    def with_options(self, **kwargs):
        """Require the adapter to disable SDK retries on its request client."""
        assert kwargs["max_retries"] == 0
        return self

    async def create(self, **kwargs):
        """Return finite vectors or an explicitly requested query failure without network I/O."""
        self.state["requests"].append(kwargs)
        is_query = len(kwargs["input"]) == 1
        if is_query:
            self.state["queries"] += 1
            if self.state["mode"] == "first-query-fails" and self.state["queries"] == 1:
                raise RuntimeError("PRIVATE_PROVIDER_ERROR_BODY")
            if self.state["mode"] == "query-waits":
                self.state["query_started"].set()
                await asyncio.Event().wait()
        return {
            "model": kwargs["model"],
            "_request_id": f"local-{len(self.state['requests'])}",
            "usage": {"prompt_tokens": 7},
            "data": [
                {"index": index, "embedding": [1.0, float(len(text) % 13 + 1)]}
                for index, text in enumerate(kwargs["input"])
            ],
        }

    async def close(self):
        """Record transport release even when cleanup itself reports a failure."""
        self.closed = True
        if self.state["mode"] == "cleanup-fails":
            raise RuntimeError("PRIVATE_CLEANUP_ERROR_BODY")


@pytest.fixture
def local_diagnostic(tmp_path, monkeypatch):
    """Prepare three isolated 70-passage corpora, one with no positive evidence labels."""
    records = diagnostic_dataset()[:3]
    for index, row in enumerate(records):
        row["answer"] = f"EVALUATOR_ONLY_ANSWER_{index}"
        row["haystack_sessions"] = [
            [
                {
                    "role": "user" if turn % 2 == 0 else "assistant",
                    "content": f"Project {index} launch color record {turn}: naïve 🦋 한글.",
                    "has_answer": turn == 0 and index != 2,
                }
                for turn in range(70)
            ]
        ]
    records[2]["question_id"] += "_abs"
    records[2]["answer_session_ids"] = []
    source = tmp_path / "input.json"
    write_json(source, records)
    prepared = tmp_path / "prepared"
    tokenizer = FixtureTokenizer()
    manifest = prepare(
        source,
        prepared,
        tokenizer=tokenizer,
        selection_mode="evaluation",
        dataset_kind="controlled-offline-fixture",
        revision="local-contract-fixture-v1",
    )
    matrix = matrix_template()
    matrix["dense"]["dimensions"] = 2
    state = {"requests": [], "clients": [], "queries": 0, "mode": "success"}
    preflight_calls = []

    def controlled_readiness(path, configuration, **kwargs):
        """Keep matrix validation while bypassing external prerequisites for B/D/H unit contracts."""
        validate_matrix(configuration)
        preflight_calls.append((path, kwargs))
        assert kwargs == {"retrieval_only": True}
        return {"ready": True, "mode": "test-controlled-readiness", "network_calls": 0}

    def diagnostic_tokenizer(*args, **kwargs):
        """Use the same explicitly diagnostic tokenizer that produced the temporary passages."""
        return tokenizer

    def local_transport(*args, **kwargs):
        """Inject only the SDK transport, preserving production embedding validation and accounting."""
        client = EmbeddingTransport(state)
        state["clients"].append(client)
        return client

    monkeypatch.setattr(diagnostic, "preflight", controlled_readiness)
    monkeypatch.setattr("llgm.retrieval.ColBERTTokenizer", diagnostic_tokenizer)
    monkeypatch.setattr("llgm.models.hosted._sdk_client", local_transport)
    return {
        "prepared": prepared,
        "output": tmp_path / "run",
        "matrix": matrix,
        "manifest": manifest,
        "state": state,
        "preflight_calls": preflight_calls,
    }


def run_local(fixture, **kwargs):
    """Execute the production runner over the controlled local preparation."""
    return asyncio.run(
        diagnostic.run_retrieval_diagnostic(
            fixture["prepared"],
            fixture["output"],
            fixture["matrix"],
            split="evaluation",
            **kwargs,
        )
    )


def rehash_prepared(fixture):
    """Commit intentional fixture mutations to its integrity manifest for structural validation tests."""
    for name in fixture["manifest"]["files"]:
        fixture["manifest"]["files"][name] = file_sha256(fixture["prepared"] / name)
    write_json(fixture["prepared"] / "prepared.json", fixture["manifest"])


def backend_records(fixture, backend, stream):
    """Read an actual per-backend artifact stream after execution."""
    return read_jsonl(fixture["output"] / backend / f"{stream}.jsonl")


@pytest.fixture
def scoped_diagnostic(local_diagnostic, monkeypatch):
    """Exercise real scope validation with controlled dependency discovery and tokenizer identity."""
    fixture = local_diagnostic
    fixture["manifest"]["dataset"]["kind"] = "controlled-development"
    fixture["manifest"]["selection"]["freeze_required"] = False
    fixture["manifest"]["tokenizer"]["implementation"] = "colbert-local-fast"
    write_json(fixture["prepared"] / "prepared.json", fixture["manifest"])
    for role in ("root", "sidecar"):
        fixture["matrix"]["models"][role]["model"] = f"controlled-{role}-v1"
    fixture["matrix"]["tokenizer"] = {
        "local_path": "tokenizer-only-files",
        "revision": "standalone-tokenizer-revision",
    }
    fixture["tokenizer_loads"] = []
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-only-no-network")
    monkeypatch.setattr("llgm.evaluation.matrix._available", lambda name: name == "openai")

    def controlled_tokenizer(local_path, revision):
        """Record standalone tokenizer pins while substituting only local tokenizer loading."""
        fixture["tokenizer_loads"].append({"local_path": local_path, "revision": revision})
        return FixtureTokenizer()

    monkeypatch.setattr("llgm.evaluation.matrix.ColBERTTokenizer", controlled_tokenizer)
    monkeypatch.setattr("llgm.evaluation.runner.ColBERTTokenizer", controlled_tokenizer)
    monkeypatch.setattr("llgm.retrieval.ColBERTTokenizer", controlled_tokenizer)
    monkeypatch.setattr(diagnostic, "preflight", preflight)
    return fixture


def test_explicit_scope_keeps_full_matrix_blocked_and_requires_shared_tokenizer(scoped_diagnostic):
    """B/D/H readiness excludes unchecked C assets while retaining tokenizer identity checks."""
    fixture = scoped_diagnostic
    full = preflight(fixture["prepared"], fixture["matrix"], retrieval_only=True)
    assert full["ready"] is False
    assert next(row for row in full["arms"] if row["id"] == "C-S")["status"] == "blocked"
    partial = preflight(
        fixture["prepared"],
        fixture["matrix"],
        required_backends=["B", "D", "H"],
    )
    assert partial["ready"] is True
    assert partial["required_backends"] == ["B", "D", "H"]
    assert partial["excluded_backends"] == ["C"]
    assert partial["full_matrix_required"] is partial["full_matrix_ready"] is False
    assert partial["benchmark_ready"] is False
    assert all(row["status"] == "not_checked" for row in partial["arms"] if row["backend"] == "C")
    assert len([row for row in partial["arms"] if row["selected"]]) == 9
    fixture["manifest"]["tokenizer"]["sha256"] = "wrong-shared-tokenizer"
    write_json(fixture["prepared"] / "prepared.json", fixture["manifest"])
    blocked = preflight(fixture["prepared"], fixture["matrix"], required_backends=["B", "D", "H"])
    assert blocked["ready"] is False
    assert any("tokenizer hash differs" in error for error in blocked["errors"])
    assert fixture["state"]["requests"] == []


def test_explicit_retrieval_scope_records_only_completed_non_c_backends(scoped_diagnostic):
    """A non-C retrieval pilot records its declared and completed scope without full-matrix claims."""
    fixture = scoped_diagnostic
    summary = run_local(fixture, required_backends=["B", "D", "H"])
    for name in (
        "required_backends",
        "selected_backends",
        "recorded_backends",
        "completed_backends",
    ):
        assert summary[name] == ["B", "D", "H"]
    assert set(summary["backends"]) == {"B", "D", "H"}
    assert summary["full_retrieval_matrix_complete"] is False
    assert summary["full_matrix_complete"] is summary["full_matrix_ready"] is False
    assert summary["physical_embedding_requests"] == 18
    assert not (fixture["output"] / "C").exists()
    for backend in ("B", "D", "H"):
        manifest = json.loads((fixture["output"] / backend / "manifest.json").read_text())
        assert manifest["required_backends"] == ["B", "D", "H"]
        assert manifest["selected_backends"] == ["B", "D", "H"]
        assert manifest["full_matrix_ready"] is False


def test_explicit_answering_scope_runs_all_nine_non_c_arms(scoped_diagnostic, monkeypatch):
    """The answering runner dispatches B/D/H × S/U/A and preserves incomplete full-matrix status."""
    from llgm.evaluation.runner import run_matrix
    from llgm.models import CallableModelClient, ModelResponse

    fixture = scoped_diagnostic

    def controlled_generation(provider, model, **kwargs):
        """Provide deterministic protocol responses while real retrieval and accounting execute."""

        async def complete(request):
            """Respond to the requested query-planning or evidence-composition operation."""
            if model == "controlled-root-v1":
                return ModelResponse("Controlled answer; no semantic quality is asserted.")
            instructions = request.messages[0].content
            if 'Return {"queries":' in instructions:
                return ModelResponse(
                    json.dumps(
                        {
                            "queries": [
                                "launch followup one",
                                "launch followup two",
                                "launch followup three",
                            ]
                        }
                    )
                )
            if 'Return {"query":' in instructions:
                return ModelResponse('{"query": null}')
            evidence = json.loads(request.messages[-1].content)["evidence"]
            return ModelResponse(
                json.dumps(
                    {
                        "text": evidence[0]["text"],
                        "passage_ids": [evidence[0]["passage_id"]],
                        "unresolved": [],
                    }
                )
            )

        return CallableModelClient(complete, provider="controlled-test", model=model)

    monkeypatch.setattr("llgm.models.create_model", controlled_generation)
    summary = asyncio.run(
        run_matrix(
            fixture["prepared"],
            fixture["output"],
            fixture["matrix"],
            split="evaluation",
            required_backends=["B", "D", "H"],
        )
    )
    expected = [f"{backend}-{policy}" for backend in ("B", "D", "H") for policy in ("S", "U", "A")]
    assert summary["required_backends"] == ["B", "D", "H"]
    assert (
        summary["selected_arms"]
        == summary["recorded_arms"]
        == summary["completed_arms"]
        == expected
    )
    assert set(summary["arms"]) == set(expected)
    assert summary["full_matrix_ready"] is summary["full_matrix_complete"] is False
    assert summary["case_arm_runs"] == 27
    assert summary["physical_generation_calls"] == 72
    assert summary["physical_embedding_requests"] == 72
    assert not (fixture["output"] / "C-S").exists()
    for arm in expected:
        predictions = read_jsonl(fixture["output"] / arm / "predictions.jsonl")
        assert len(predictions) == 3
        assert all(row["status"] == "completed" for row in predictions)


@pytest.mark.parametrize("runner", ["retrieval", "answering"])
def test_explicit_non_c_scope_refuses_c_dispatch_before_preflight(local_diagnostic, runner):
    """Explicitly excluded C execution is rejected before any backend or provider is constructed."""
    fixture = local_diagnostic
    with pytest.raises(ConfigurationError, match="outside the explicit"):
        if runner == "retrieval":
            run_local(fixture, backends=["C"], required_backends=["B", "D", "H"])
        else:
            from llgm.evaluation.runner import run_matrix

            asyncio.run(
                run_matrix(
                    fixture["prepared"],
                    fixture["output"],
                    fixture["matrix"],
                    arms=["C-S"],
                    required_backends=["B", "D", "H"],
                )
            )
    assert fixture["preflight_calls"] == []
    assert fixture["state"]["clients"] == []
    assert not fixture["output"].exists()


@pytest.mark.parametrize("failed_role", ["root", "sidecar"])
def test_answer_failure_results_retain_provider_error_and_attempts(
    scoped_diagnostic, monkeypatch, failed_role
):
    """Returned runtime failures retain the same error and cost records as raised failures."""
    from llgm.core.errors import ProviderError
    from llgm.evaluation.runner import run_matrix
    from llgm.models import CallableModelClient, ModelResponse

    fixture = scoped_diagnostic

    def controlled_generation(provider, model, **kwargs):
        """Replace only the remote transport while real indexing and runtime execution continue."""
        role = "root" if model == "controlled-root-v1" else "sidecar"

        async def complete(request):
            """Fail the selected role or compose exactly the provided evidence fixture."""
            if role == failed_role:
                raise ProviderError("Controlled provider unavailable")
            payload = json.loads(request.messages[-1].content)
            first = payload["evidence"][0]
            return ModelResponse(
                json.dumps(
                    {"text": first["text"], "passage_ids": [first["passage_id"]], "unresolved": []}
                )
            )

        return CallableModelClient(complete, provider="controlled-test", model=model)

    monkeypatch.setattr("llgm.models.create_model", controlled_generation)
    asyncio.run(
        run_matrix(
            fixture["prepared"],
            fixture["output"],
            fixture["matrix"],
            arms=["B-S"],
            split="evaluation",
            required_backends=["B", "D", "H"],
        )
    )
    predictions = read_jsonl(fixture["output"] / "B-S" / "predictions.jsonl")
    assert len(predictions) == 3
    assert all(
        row["status"] == "failed" and row["error_type"] == "ProviderError" for row in predictions
    )
    attempts = read_jsonl(fixture["output"] / "B-S" / "usage.jsonl")
    assert len(attempts) == 3 * (2 if failed_role == "root" else 1)
    assert sum(row["status"] == "failed" for row in attempts) == 3


@pytest.mark.parametrize("backend", ["B", "D", "H"])
def test_local_index_execution_preserves_case_isolation_and_artifacts(local_diagnostic, backend):
    """Real BM25/cosine/RRF runs expose at most 40 source-bound hits and retain complete records."""
    fixture = local_diagnostic
    summary = run_local(fixture, backends=[backend])
    assert summary["benchmark_result"] is False
    assert summary["selected_backends"] == [backend]
    assert summary["required_backends"] == ["B", "D", "H", "C"]
    assert fixture["preflight_calls"] == [(fixture["prepared"], {"retrieval_only": True})]
    metrics = summary["backends"][backend]
    assert metrics["case_count"] == metrics["completed"] == 3
    assert metrics["source_recall_denominator"] == 2
    assert metrics["turn_recall_denominator"] == 2
    assert metrics["abstention_cases"] == 1
    assert metrics["official_answer_score"] is None
    assert metrics["generation_calls"] == 0
    assert summary["physical_embedding_requests"] == (0 if backend == "B" else 9)
    assert metrics["embedding_requests"] == summary["physical_embedding_requests"]
    assert metrics["embedding_input_tokens"] == summary["physical_embedding_requests"] * 7
    assert metrics["currency_cost"] is None
    assert summary == json.loads((fixture["output"] / "summary.json").read_text())
    directory = fixture["output"] / backend
    for name in (*RunArtifacts.STREAMS,):
        assert (directory / f"{name}.jsonl").is_file()
    for name in ("manifest.json", "metrics.json", "decision.md"):
        assert (directory / name).is_file()
    predictions = backend_records(fixture, backend, "predictions")
    queries = {row["case_id"]: row for row in read_jsonl(fixture["prepared"] / "queries.jsonl")}
    assert len(predictions) == len(backend_records(fixture, backend, "judgments")) == 3
    for prediction in predictions:
        query = queries[prediction["case_id"]]
        passages = read_jsonl(fixture["prepared"] / query["passages_path"])
        expected = {passage["passage_id"]: passage for passage in passages}
        assert len(expected) == 70
        assert prediction["returned_hit_count"] == 40
        assert len({hit["passage_id"] for hit in prediction["hits"]}) == 40
        assert [hit["rank"] for hit in prediction["hits"]] == list(range(1, 41))
        for hit in prediction["hits"]:
            assert hit["references"] == expected[hit["passage_id"]]["refs"]
        case_hash = hashlib.sha256(prediction["case_id"].encode()).hexdigest()[:24]
        expected_index = "bm25.sqlite3" if backend == "B" else "dense.json"
        assert (directory / "indexes" / case_hash / expected_index).is_file()
    serialized_inputs = json.dumps(fixture["state"]["requests"])
    assert "EVALUATOR_ONLY_ANSWER" not in serialized_inputs
    assert "has_answer" not in serialized_inputs
    assert "answer_session_ids" not in serialized_inputs
    assert all(client.closed for client in fixture["state"]["clients"])
    if backend != "B":
        assert [len(call["input"]) for call in fixture["state"]["requests"]] == [64, 6, 1] * 3
        query_calls = [
            call["input"][0] for call in fixture["state"]["requests"] if len(call["input"]) == 1
        ]
        assert query_calls == [query["question"] for query in queries.values()]
    traces = backend_records(fixture, backend, "traces")
    operations = [row.get("operation") for row in traces]
    if backend == "H":
        assert operations.count("rrf_search") == 3
        assert operations.count("exact_cosine_search") == 3


def test_actual_preflight_refuses_fixture_even_for_b_subset(local_diagnostic, monkeypatch):
    """Selecting BM25 cannot silently bypass the required matrix's missing real tokenizer and C assets."""
    fixture = local_diagnostic
    monkeypatch.setattr(diagnostic, "preflight", preflight)
    with pytest.raises(ConfigurationError, match="all four backend requirements"):
        run_local(fixture, backends=["B"])
    assert fixture["state"]["requests"] == []
    assert not fixture["output"].exists()


@pytest.mark.parametrize("backends", [[], ["B", "B"], ["X"], ["b"]])
def test_invalid_backend_selection_precedes_preflight(local_diagnostic, backends):
    """Empty, duplicate or unknown backend requests fail before readiness checks or writes."""
    with pytest.raises(ConfigurationError, match="unique nonempty"):
        run_local(local_diagnostic, backends=backends)
    assert local_diagnostic["preflight_calls"] == []
    assert not local_diagnostic["output"].exists()


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "escape",
        "no-selection",
        "duplicate-query",
        "missing-gold",
        "tokenizer",
        "case-cap",
    ],
)
def test_invalid_preparation_fails_before_execution(local_diagnostic, damage):
    """Corrupt, incomplete or incompatible preparation never starts indexing or paid transports."""
    fixture = local_diagnostic
    prepared = fixture["prepared"]
    if damage == "checksum":
        with (prepared / "queries.jsonl").open("a") as handle:
            handle.write("\n")
    elif damage == "escape":
        fixture["manifest"]["files"]["../input.json"] = file_sha256(prepared.parent / "input.json")
        write_json(prepared / "prepared.json", fixture["manifest"])
    elif damage == "no-selection":
        fixture["manifest"]["selection"]["evaluation_ids"] = []
        write_json(prepared / "prepared.json", fixture["manifest"])
    elif damage == "duplicate-query":
        rows = read_jsonl(prepared / "queries.jsonl")
        write_jsonl(prepared / "queries.jsonl", [*rows, rows[0]])
        rehash_prepared(fixture)
    elif damage == "missing-gold":
        write_jsonl(prepared / "evaluator/gold.jsonl", [])
        rehash_prepared(fixture)
    elif damage == "tokenizer":
        fixture["manifest"]["tokenizer"]["sha256"] = "incompatible-tokenizer"
        write_json(prepared / "prepared.json", fixture["manifest"])
    else:
        fixture["matrix"]["retrieval_diagnostic"] = {"max_embedding_requests_per_case": True}
    with pytest.raises(ConfigurationError):
        run_local(fixture, backends=["D"])
    assert fixture["state"]["clients"] == []
    assert not fixture["output"].exists()


def test_existing_output_is_not_overwritten(local_diagnostic):
    """A nonempty output directory retains its existing artifacts when a run is refused."""
    fixture = local_diagnostic
    fixture["output"].mkdir()
    sentinel = fixture["output"] / "summary.json"
    sentinel.write_text("original")
    with pytest.raises(ConfigurationError, match="new or empty"):
        run_local(fixture, backends=["B"])
    assert sentinel.read_text() == "original"
    assert fixture["state"]["clients"] == []


@pytest.mark.parametrize("damage", ["question", "reference", "query-overflow", "document-overflow"])
def test_invalid_case_records_failure_and_keeps_other_cases(local_diagnostic, damage):
    """A structurally invalid case fails before encoding while later isolated cases remain runnable."""
    fixture = local_diagnostic
    prepared = fixture["prepared"]
    queries = read_jsonl(prepared / "queries.jsonl")
    first = queries[0]
    sources = json.loads((prepared / first["sources_path"]).read_text())
    passages = read_jsonl(prepared / first["passages_path"])
    if damage == "question":
        sources["question"] = "different case question"
    elif damage == "reference":
        passages[0]["refs"][0]["node_id"] = "another-cases-node"
    elif damage == "query-overflow":
        first["question"] = sources["question"] = "overflow " * 129
    else:
        passages[0]["text"] = "overflow " * 181
    write_json(prepared / first["sources_path"], sources)
    write_jsonl(prepared / first["passages_path"], passages)
    write_jsonl(prepared / "queries.jsonl", queries)
    rehash_prepared(fixture)
    summary = run_local(fixture, backends=["D"])
    metrics = summary["backends"]["D"]
    assert (metrics["case_count"], metrics["completed"], metrics["failed"]) == (3, 2, 1)
    assert summary["physical_embedding_requests"] == 6
    predictions = backend_records(fixture, "D", "predictions")
    assert predictions[0]["status"] == "failed"
    assert predictions[0]["failure_phase"] == "preparation"
    assert predictions[0]["hits"] == []
    judgments = backend_records(fixture, "D", "judgments")
    assert judgments[0]["source_recall"] == judgments[0]["turn_recall"] == 0
    assert metrics["source_recall_denominator"] == 2


@pytest.mark.parametrize("maximum,phase", [(1, "index"), (2, "query")])
def test_per_case_embedding_cap_preserves_failed_denominators(local_diagnostic, maximum, phase):
    """Index batches and query encoding share one case allowance without dropping failed questions."""
    fixture = local_diagnostic
    fixture["matrix"]["retrieval_diagnostic"] = {"max_embedding_requests_per_case": maximum}
    summary = run_local(fixture, backends=["D"])
    assert summary["physical_embedding_requests"] == maximum * 3
    metrics = summary["backends"]["D"]
    assert (metrics["completed"], metrics["failed"]) == (0, 3)
    assert metrics["source_recall"] == metrics["turn_recall"] == 0
    assert metrics["source_recall_denominator"] == metrics["turn_recall_denominator"] == 2
    for prediction in backend_records(fixture, "D", "predictions"):
        assert prediction["status"] == "budget_exhausted"
        assert prediction["failure_phase"] == phase
        assert prediction["usage"]["embedding_requests"] == maximum
    assert all(client.closed for client in fixture["state"]["clients"])


def test_overall_caps_are_shared_across_backends(local_diagnostic):
    """Case and embedding caps apply to the complete run, retaining failures in later backend artifacts."""
    fixture = local_diagnostic
    fixture["matrix"]["overall"]["max_embedding_requests"] = 2
    fixture["matrix"]["overall"]["max_case_arm_runs"] = 4
    summary = run_local(fixture, backends=["D", "H"])
    assert summary["physical_embedding_requests"] == len(fixture["state"]["requests"]) == 2
    assert summary["case_backend_runs"] == 4
    for backend in ("D", "H"):
        assert summary["backends"][backend]["case_count"] == 3
        assert summary["backends"][backend]["failed"] == 3
        assert len(backend_records(fixture, backend, "predictions")) == 3
    assert sum(summary["backends"][backend]["embedding_requests"] for backend in ("D", "H")) == 2
    assert all(client.closed for client in fixture["state"]["clients"])


def test_partial_provider_failure_retains_unknown_usage_and_continues(local_diagnostic):
    """A failed physical query preserves unknown usage, redacts its error body and does not erase later results."""
    fixture = local_diagnostic
    fixture["state"]["mode"] = "first-query-fails"
    summary = run_local(fixture, backends=["D"])
    metrics = summary["backends"]["D"]
    assert (metrics["completed"], metrics["failed"]) == (2, 1)
    assert summary["physical_embedding_requests"] == metrics["embedding_requests"] == 9
    assert metrics["embedding_input_tokens"] is None
    assert metrics["known_embedding_input_tokens"] == 56
    assert metrics["embedding_requests_with_unknown_usage"] == 1
    assert metrics["source_recall"] == 0.5
    predictions = backend_records(fixture, "D", "predictions")
    assert predictions[0]["failure_phase"] == "query"
    assert predictions[0]["error_type"] == "ProviderError"
    usage = backend_records(fixture, "D", "usage")
    assert len([row for row in usage if row["category"] == "embedding"]) == 9
    assert all(client.closed for client in fixture["state"]["clients"])
    for path in fixture["output"].rglob("*.json*"):
        assert "PRIVATE_PROVIDER_ERROR_BODY" not in path.read_text()


def test_timeout_preserves_attempted_query_usage_and_closes_transports(local_diagnostic):
    """A waiting query is bounded by the case deadline and remains an unknown-usage physical request."""
    fixture = local_diagnostic
    fixture["state"]["mode"] = "query-waits"
    fixture["state"]["query_started"] = asyncio.Event()
    fixture["matrix"]["budgets"]["timeout_seconds"] = 0.1
    summary = run_local(fixture, backends=["D"])
    assert fixture["state"]["query_started"].is_set()
    assert summary["backends"]["D"]["failed"] == 3
    assert summary["physical_embedding_requests"] == 9
    assert summary["backends"]["D"]["embedding_requests_with_unknown_usage"] == 3
    assert all(
        row["status"] == "budget_exhausted" for row in backend_records(fixture, "D", "predictions")
    )
    assert all(client.closed for client in fixture["state"]["clients"])


def test_cancelled_query_preserves_partial_artifacts_and_usage(local_diagnostic):
    """Cancellation propagates after the attempted request, partial prediction and summary are durable."""
    fixture = local_diagnostic
    fixture["state"]["mode"] = "query-waits"

    async def exercise():
        """Cancel the first active query after two real local indexing batches complete."""
        fixture["state"]["query_started"] = asyncio.Event()
        task = asyncio.create_task(
            diagnostic.run_retrieval_diagnostic(
                fixture["prepared"],
                fixture["output"],
                fixture["matrix"],
                backends=["D"],
                split="evaluation",
            )
        )
        try:
            await asyncio.wait_for(fixture["state"]["query_started"].wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
    assert len(fixture["state"]["requests"]) == 3
    assert all(client.closed for client in fixture["state"]["clients"])
    predictions = backend_records(fixture, "D", "predictions")
    assert len(predictions) == 1
    assert predictions[0]["status"] == "cancelled"
    assert predictions[0]["usage"]["embedding_requests"] == 3
    assert predictions[0]["usage"]["embedding_requests_with_unknown_usage"] == 1
    assert backend_records(fixture, "D", "judgments")[0]["source_recall"] == 0
    summary = json.loads((fixture["output"] / "summary.json").read_text())
    assert summary["cancelled"] is True
    assert summary["physical_embedding_requests"] == 3
    assert summary["completed_backends"] == []
    assert summary["full_matrix_complete"] is False
    assert (fixture["output"] / "D/metrics.json").is_file()
    assert (fixture["output"] / "D/decision.md").is_file()


@pytest.mark.parametrize("runner_kind", ["retrieval", "answering"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_cancellation_during_cleanup_retains_the_case_and_finishes_release(
    local_diagnostic, monkeypatch, runner_kind, close_fails
):
    """Both runners finish owned cleanup and retain the interrupted case before returning cancellation."""
    from llgm.evaluation.runner import run_matrix

    fixture = local_diagnostic
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_close = SQLiteBM25Retriever.close

    async def delayed_close(self):
        """Release a real index after controlled delay, optionally reporting a sanitized failure."""
        started.set()
        await release.wait()
        original_close(self)
        finished.set()
        if close_fails:
            raise RuntimeError("PRIVATE_CLEANUP_ERROR_BODY")

    monkeypatch.setattr(SQLiteBM25Retriever, "close", delayed_close)

    async def exercise():
        """Deliver repeated cancellation only after a completed query enters resource release."""
        operation = (
            diagnostic.run_retrieval_diagnostic(
                fixture["prepared"],
                fixture["output"],
                fixture["matrix"],
                backends=["B"],
                split="evaluation",
            )
            if runner_kind == "retrieval"
            else run_matrix(
                fixture["prepared"],
                fixture["output"],
                fixture["matrix"],
                diagnostic=True,
                split="evaluation",
            )
        )
        task = asyncio.create_task(operation)
        try:
            await asyncio.wait_for(started.wait(), 3)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert not finished.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
    assert finished.is_set()
    directory = fixture["output"] / ("B" if runner_kind == "retrieval" else "B-S")
    assert len(read_jsonl(directory / "cases.jsonl")) == 1
    predictions = read_jsonl(directory / "predictions.jsonl")
    assert len(predictions) == 1
    assert predictions[0]["status"] == "cancelled"
    assert len(read_jsonl(directory / "judgments.jsonl")) == 1
    traces = read_jsonl(directory / "traces.jsonl")
    assert any(row["kind"] == "cancelled" and row.get("phase") == "cleanup" for row in traces)
    failures = [row for row in traces if row["kind"] == "cleanup_failure"]
    assert len(failures) == int(close_fails)
    if close_fails:
        assert failures[0]["error_type"] == "RuntimeError"
    assert "PRIVATE_CLEANUP_ERROR_BODY" not in json.dumps(traces)
    assert json.loads((directory / "metrics.json").read_text())["cancelled"] is True
    summary = json.loads((fixture["output"] / "summary.json").read_text())
    assert summary["cancelled"] is True
    assert summary["full_matrix_complete"] is False


def test_completed_hits_after_deadline_are_audited_but_not_scored(local_diagnostic, monkeypatch):
    """Late results retain source references for audit while successful-evidence credit stays zero."""
    fixture = local_diagnostic
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(diagnostic, "time", SimpleNamespace(monotonic=lambda: clock.value))
    original = SQLiteBM25Retriever.search

    async def late_search(self, question, k):
        """Return real lexical hits and advance only the diagnostic's case clock."""
        hits = await original(self, question, k)
        clock.value += 121
        return hits

    monkeypatch.setattr(SQLiteBM25Retriever, "search", late_search)
    summary = run_local(fixture, backends=["B"])
    assert summary["backends"]["B"]["completed"] == 0
    assert summary["backends"]["B"]["source_recall"] == 0
    for row in backend_records(fixture, "B", "predictions"):
        assert row["status"] == "budget_exhausted"
        assert row["returned_hit_count"] == 40
        assert row["hits"]


def test_cleanup_failure_is_recorded_without_hiding_completed_queries(local_diagnostic):
    """Cleanup exceptions are visible, sanitized and do not suppress prediction or summary artifacts."""
    fixture = local_diagnostic
    fixture["state"]["mode"] = "cleanup-fails"
    summary = run_local(fixture, backends=["D"])
    assert summary["backends"]["D"]["completed"] == 3
    failures = [
        row for row in backend_records(fixture, "D", "traces") if row["kind"] == "cleanup_failure"
    ]
    assert len(failures) == 3
    assert all(row["error_type"] == "RuntimeError" for row in failures)
    assert "PRIVATE_CLEANUP_ERROR_BODY" not in json.dumps(failures)
    assert all(client.closed for client in fixture["state"]["clients"])


@pytest.mark.parametrize("worker_fails", [False, True])
def test_local_native_work_is_drained_before_cancellation_returns(worker_fails):
    """Repeated cancellation drains native work and retains cancellation even if the worker fails."""
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def worker():
        """Model an already-running native operation that cannot be interrupted by task cancellation."""
        started.set()
        assert release.wait(5)
        finished.set()
        if worker_fails:
            raise RuntimeError("native failure after cancellation")

    async def exercise():
        """Cancel the wrapper while the worker is blocked, then release and observe orderly completion."""
        limits = SimpleNamespace(remaining=lambda: 60)
        task = asyncio.create_task(
            diagnostic._local_call(
                worker,
                deadline=diagnostic.time.monotonic() + 60,
                limits=limits,
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 5)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_empty_embedding_batch_does_not_spend_budget():
    """An empty batch returns before expired deadlines or physical-call accounting."""
    embedder = diagnostic._CaseEmbedder(SimpleNamespace(model="local"), 1, 0)
    assert asyncio.run(embedder.embed([])) == []
    assert embedder.calls == 0
    with pytest.raises(BudgetExceeded, match="deadline"):
        asyncio.run(embedder.embed(["source"]))
    assert embedder.calls == 0


def test_matrix_c_build_drains_native_worker_on_repeated_cancellation(
    local_diagnostic, monkeypatch
):
    """The matrix retains its cancelled case only after its native index builder finishes."""
    from llgm.evaluation import runner
    from llgm.models import ScriptedModelClient
    from llgm.retrieval import colbert

    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def build(passages, **kwargs):
        """Stand in for an uninterruptible native builder without loading ColBERT assets."""
        started.set()
        assert release.wait(5)
        finished.set()
        raise RuntimeError("native failure after cancellation")

    def readiness(*args, **kwargs):
        """Permit entry into the C build path without installing or probing native dependencies."""
        return {"ready": True, "required_backends": ["C"], "full_matrix_ready": False}

    monkeypatch.setattr(runner, "preflight", readiness)
    monkeypatch.setattr(runner, "ColBERTTokenizer", lambda **kwargs: FixtureTokenizer())
    monkeypatch.setattr("llgm.models.create_model", lambda *args, **kwargs: ScriptedModelClient([]))
    monkeypatch.setattr(colbert, "ColBERTConfig", lambda **kwargs: None)
    monkeypatch.setattr(colbert.ColBERTRetriever, "build", build)

    async def exercise():
        """Cancel while the builder owns work, then verify the runner drains before returning."""
        task = asyncio.create_task(
            runner.run_matrix(
                local_diagnostic["prepared"],
                local_diagnostic["output"],
                local_diagnostic["matrix"],
                arms=["C-S"],
                split="evaluation",
                required_backends=["C"],
            )
        )
        try:
            builder_started = await asyncio.to_thread(started.wait, 5)
            if task.done():
                task.result()
            assert builder_started
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
    predictions = read_jsonl(local_diagnostic["output"] / "C-S" / "predictions.jsonl")
    assert len(predictions) == 1
    assert predictions[0]["status"] == "cancelled"
