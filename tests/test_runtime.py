"""Iterative evidence-policy budgets, citation validation and observable failure contracts."""

import asyncio
import json

import pytest

from llgm import Budget, SourceSpan
from llgm.core.errors import ConfigurationError, ProviderError, SchemaError
from llgm.inference.iterative import EvidenceSidecar, IterativeRuntime
from llgm.models import CallableModelClient, ModelResponse, ScriptedModelClient, Usage
from llgm.retrieval import SearchHit, SearchPassage


def hit(pid, text="Atlas uses PostgreSQL.", metadata=None):
    """Build a retrieval hit with a source span matching its supplied text."""
    return SearchHit(
        SearchPassage(pid, text, (SourceSpan(pid, "t", 0, len(text)),), metadata or {}),
        1.0,
        1,
    )


class Retriever:
    """Ordered retrieval double that records query text and visible hit limits."""

    def __init__(self, responses):
        """Queue search results and initialize an empty request ledger."""
        self.responses = iter(responses)
        self.calls = []

    async def search(self, query, k):
        """Record a query and consume its next fixed retrieval result."""
        self.calls.append((query, k))
        return next(self.responses)


def composition(ids, text="Atlas uses PostgreSQL."):
    """Encode a sidecar evidence bundle with explicit passage citations."""
    return json.dumps({"text": text, "passage_ids": ids, "unresolved": []})


def test_single_policy_answers_with_verified_references_and_unknown_usage():
    """Single-pass answers retain verified references and distinguish unknown token usage."""

    async def run():
        """Compose one retrieved passage before the root supplies the final answer."""
        retriever = Retriever([[hit("a")]])
        sidecar = EvidenceSidecar(
            model=ScriptedModelClient([composition(["a"])]), retriever=retriever, policy="single"
        )
        root = ScriptedModelClient([ModelResponse("PostgreSQL.", Usage(10, 3))])
        result = await IterativeRuntime(root=root, sidecar=sidecar).answer("Atlas database?")
        assert result.answer == "PostgreSQL."
        assert result.references[0].node_id == "a"
        assert result.usage["model_calls"] == 2
        assert result.usage["unknown_usage_calls"] == 1
        assert result.usage["known_input_tokens"] == 10
        assert retriever.calls == [("Atlas database?", 40)]
        assert all("query" not in event for event in result.trace)

    asyncio.run(run())


def test_adaptive_followup_depends_on_first_evidence_and_deduplicates():
    """Adaptive search uses observed evidence and deduplicates repeated passages."""

    async def run():
        """Expose the initial choice before requesting its production update."""
        model = ScriptedModelClient(
            [
                '{"query":"Atlas production update"}',
                '{"query":null}',
                composition(
                    ["a", "b"], "Original choice was SQLite; production now uses PostgreSQL."
                ),
            ]
        )
        original = hit("a", "Original choice: SQLite.")
        retriever = Retriever([[original], [original, hit("b")]])
        sidecar = EvidenceSidecar(model=model, retriever=retriever, policy="adaptive")
        result = await sidecar.gather("Atlas database?")
        assert len(result.hits) == 2
        assert retriever.calls == [("Atlas database?", 10), ("Atlas production update", 10)]
        assert "SQLite" in model.requests[0].messages[1].content
        assert len(result.references) == 2

    asyncio.run(run())


def test_upfront_generation_sees_no_results_and_queries_use_same_hit_ceiling():
    """Upfront query generation sees only the original question and shares the hit budget."""

    async def run():
        """Generate three follow-ups before executing the four bounded searches."""
        model = ScriptedModelClient(['{"queries":["q2","q3","q4"]}', composition(["a"])])
        retriever = Retriever([[hit("a")], [], [], []])
        result = await EvidenceSidecar(model=model, retriever=retriever, policy="upfront").gather(
            "q1"
        )
        assert model.requests[0].messages[1].content == "q1"
        assert retriever.calls == [(q, 10) for q in ["q1", "q2", "q3", "q4"]]
        assert result.usage["searches"] == 4

    asyncio.run(run())


def test_root_call_reserved_when_sidecar_runs_out_of_allowance():
    """A depleted sidecar allowance still reserves the root's final response."""

    async def run():
        """Use a one-call budget to expose evidence without invoking the sidecar model."""
        sidecar_model = ScriptedModelClient([])
        sidecar = EvidenceSidecar(
            model=sidecar_model, retriever=Retriever([[hit("a")]]), policy="single"
        )
        root = ScriptedModelClient(["PostgreSQL."])
        result = await IterativeRuntime(root=root, sidecar=sidecar).answer(
            "database?", Budget(max_model_calls=1)
        )
        assert result.status == "partial"
        assert result.usage["model_calls"] == 1
        assert len(sidecar_model.requests) == 0
        assert result.evidence.stop_reason == "budget_exhausted"
        assert result.references

    asyncio.run(run())


def test_unseen_citation_rejected_before_root_answer():
    """An unseen sidecar citation returns an explicit failure before root inference."""

    async def run():
        """Retain admitted raw evidence and attempted work after an invented citation."""
        sidecar = EvidenceSidecar(
            model=ScriptedModelClient([composition(["invented"])]),
            retriever=Retriever([[hit("a")]]),
            policy="single",
        )
        root = ScriptedModelClient(["must not be called"])
        result = await IterativeRuntime(root=root, sidecar=sidecar).answer("database?")
        assert not root.requests
        assert result.status == "failed" and result.answer == ""
        assert result.usage["model_calls"] == 1
        assert result.trace[-1]["error_type"] == "SchemaError"
        assert [ref.node_id for ref in result.references] == ["a"]
        assert "invented" not in result.evidence.text

    asyncio.run(run())


def test_overlarge_passage_never_exposed_to_reader():
    """An over-budget passage never reaches the reader's prompt."""

    async def run():
        """Reject a large hit before any model invocation."""
        model = ScriptedModelClient([])
        result = await EvidenceSidecar(
            model=model, retriever=Retriever([[hit("a", "x" * 1000)]]), policy="single"
        ).gather("q", Budget(max_evidence_tokens=20))
        assert result.stop_reason == "no_evidence"
        assert not model.requests and not result.hits

    asyncio.run(run())


def test_deadline_counts_attempt_and_stops_model():
    """A timed-out model call remains an accounted attempt."""

    async def slow(request):
        """Delay the provider response beyond the declared run deadline."""
        await asyncio.sleep(0.05)
        return ModelResponse("unreachable")

    async def run():
        """Spend a short deadline on one observable sidecar request."""
        model = CallableModelClient(slow)
        result = await EvidenceSidecar(
            model=model, retriever=Retriever([[hit("a")]]), policy="single"
        ).gather("q", Budget(timeout_seconds=0.01))
        assert result.stop_reason == "budget_exhausted"
        assert result.usage["model_calls"] == 1
        assert result.usage["unknown_usage_calls"] == 1
        assert result.trace[-1]["status"] == "timeout"

    asyncio.run(run())


def test_query_date_reaches_both_models_without_changing_original_search():
    """The query date reaches both models without rewriting the search question."""

    async def run():
        """Carry one explicit date through evidence composition and root inference."""
        reader = ScriptedModelClient([composition(["a"])])
        root = ScriptedModelClient(["PostgreSQL."])
        retriever = Retriever([[hit("a")]])
        sidecar = EvidenceSidecar(model=reader, retriever=retriever, policy="single")
        await IterativeRuntime(root=root, sidecar=sidecar).answer(
            "database?", question_date="2026-09-09"
        )
        assert retriever.calls == [("database?", 40)]
        assert "2026-09-09" in reader.requests[0].messages[1].content
        assert "2026-09-09" in root.requests[0].messages[1].content

    asyncio.run(run())


def test_upfront_duplicate_queries_fail_without_search_or_hidden_repair():
    """Duplicate upfront queries fail before retrieval and receive no hidden repair."""

    async def run():
        """Submit a follow-up list containing the original query."""
        model = ScriptedModelClient(['{"queries":["q1","q2","q3"]}'])
        retriever = Retriever([])
        with pytest.raises(SchemaError) as caught:
            await EvidenceSidecar(model=model, retriever=retriever, policy="upfront").gather("q1")
        assert not retriever.calls
        assert caught.value.llgm_usage["model_calls"] == 1

    asyncio.run(run())


def test_duplicate_json_fields_cannot_replace_an_iterative_query_plan():
    """An ambiguous query plan fails before retrieval instead of silently choosing the last queries field."""

    async def scenario():
        """Offer two conflicting plans in one object and require an explicit schema failure."""
        reader = ScriptedModelClient(
            ['{"queries":["first"],"queries":["second","third","fourth"]}']
        )
        retriever = Retriever([])
        with pytest.raises(SchemaError, match="JSON object"):
            await EvidenceSidecar(model=reader, retriever=retriever, policy="upfront").gather(
                "question"
            )
        assert len(reader.requests) == 1
        assert not retriever.calls

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", [-1, True, 1.5, float("nan"), None])
def test_invalid_accounting_cannot_admit_a_model_call(invalid):
    """Invalid custom counter results fail before model work or a charged attempt."""
    from llgm.inference.budget import RunLedger
    from llgm.models import Message

    async def scenario():
        """Reject malformed accounting even when it would make the input appear to fit."""
        client = ScriptedModelClient(["must not dispatch"])
        ledger = RunLedger(Budget(), lambda text: invalid)
        with pytest.raises(ConfigurationError, match="nonnegative integer"):
            await ledger.call(client, [Message("user", "Question")])
        assert not client.requests
        assert ledger.calls == 0 and not ledger.events

    asyncio.run(scenario())


def test_cache_usage_categories_survive_the_runtime_ledger():
    """Cached-input billing categories survive without doubling total input tokens."""

    async def run():
        """Compose evidence from a response carrying explicit cached-input usage."""
        response = ModelResponse(
            composition(["a"]), Usage(100, 20, {"cache_read_input_tokens": 80})
        )
        bundle = await EvidenceSidecar(
            model=ScriptedModelClient([response]),
            retriever=Retriever([[hit("a")]]),
            policy="single",
        ).gather("q")
        event = [item for item in bundle.trace if item["kind"] == "model"][0]
        assert event["usage_extra"]["cache_read_input_tokens"] == 80
        assert bundle.usage["known_input_tokens"] == 100

    asyncio.run(run())


def test_cancellation_retains_the_dispatched_attempt():
    """Cancellation preserves the already dispatched model attempt and its status."""

    async def run():
        """Cancel evidence gathering only after the injected model has started."""
        entered = asyncio.Event()

        async def waiting(request):
            """Signal dispatch and remain suspended until cancellation arrives."""
            entered.set()
            await asyncio.Event().wait()

        model = CallableModelClient(waiting)
        sidecar = EvidenceSidecar(model=model, retriever=Retriever([[hit("a")]]), policy="single")
        task = asyncio.create_task(sidecar.gather("q"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.llgm_usage["unknown_usage_calls"] == 1
        assert caught.value.llgm_trace[-1]["status"] == "cancelled"

    asyncio.run(run())


def test_native_schema_is_charged_before_dispatch_at_the_context_boundary():
    """A schema that crosses the input allowance consumes no model call or request."""
    from llgm.core.errors import BudgetExceeded
    from llgm.inference.budget import RunLedger, byte_token_bound
    from llgm.models import Message, ModelCapabilities

    async def scenario():
        """Reject one unit over budget and accept the exact combined input boundary."""
        requests = []

        async def respond(request):
            """Record admitted requests and return a completed synthetic schema response."""
            requests.append(request)
            return ModelResponse("{}")

        client = CallableModelClient(
            respond, capabilities=ModelCapabilities(structured_output=True)
        )
        messages = [Message("user", "A")]
        schema = {"type": "object", "description": "x" * 5000}
        size = len(json.dumps([{"role": "user", "content": "A"}]).encode())
        size += len(json.dumps(schema).encode())
        denied = RunLedger(Budget(max_context_tokens=size, max_output_tokens=1), byte_token_bound)
        with pytest.raises(BudgetExceeded, match="context allowance"):
            await denied.call(client, messages, output_schema=schema)
        assert not requests and denied.calls == 0 and not denied.events
        admitted = RunLedger(
            Budget(max_context_tokens=size + 1, max_output_tokens=1), byte_token_bound
        )
        assert await admitted.call(client, messages, output_schema=schema) == "{}"
        assert len(requests) == 1 and requests[0].output_schema == schema
        assert admitted.events[0]["context_accounting_units"] == size
        assert admitted.events[0]["structured_output"] is True

    asyncio.run(scenario())


def test_current_source_metadata_reaches_the_iterative_reader(tmp_path):
    """Source speaker, date, and scope survive current evidence retrieval into reader input."""
    from llgm import Conversation, Workspace
    from llgm.memory.evidence import Evidence

    async def scenario():
        """Search one persisted source whose applicability exists only in its metadata."""
        async with Workspace.open(tmp_path) as workspace:
            await workspace.ingest(
                Conversation.from_turns(
                    [{"role": "assistant", "content": "Atlas uses PostgreSQL."}],
                    node_id="atlas",
                    metadata={"date": "2031-05-16", "scope": {"environment": "staging"}},
                )
            )
            async with await Evidence.open(workspace) as evidence:
                requests = []

                async def compose(request):
                    """Return citations chosen from the actual retrieved passage identifiers."""
                    payload = json.loads(request.messages[1].content)
                    requests.append(payload)
                    return ModelResponse(
                        composition([p["passage_id"] for p in payload["evidence"]])
                    )

                result = await EvidenceSidecar(
                    model=CallableModelClient(compose),
                    retriever=evidence,
                    policy="single",
                ).gather("Atlas")
                metadata = requests[0]["evidence"][0]["metadata"]
                assert metadata["role"] == "assistant"
                assert metadata["source_metadata"] == {
                    "date": "2031-05-16",
                    "scope": {"environment": "staging"},
                }
                assert result.references[0].node_id == "atlas"

    asyncio.run(scenario())


def test_metadata_is_charged_before_passage_admission():
    """A short source with oversized applicability metadata cannot bypass evidence limits."""

    async def scenario():
        """Reject the entire passage before exposing it to the reader."""
        model = ScriptedModelClient([])
        result = await EvidenceSidecar(
            model=model,
            policy="single",
            retriever=Retriever([[hit("a", "Atlas", {"scope": "x" * 1000})]]),
        ).gather("Atlas", Budget(max_evidence_tokens=100))
        assert result.stop_reason == "no_evidence"
        assert not result.hits and not model.requests
        assert result.usage["evidence_accounting_units"] == 0

    asyncio.run(scenario())


def test_budget_fallback_preserves_the_admitted_metadata_snapshot():
    """Raw root evidence retains admitted metadata even if a later callback mutates the hit."""

    async def scenario():
        """Spend the sidecar call allowance after observing a source with scoped metadata."""
        metadata = {"role": "assistant", "date": "2031-05-16", "scope": {"env": "staging"}}

        async def followup(request):
            """Mutate caller-owned metadata after it has already been admitted."""
            metadata["scope"]["env"] = "production"
            return ModelResponse('{"query":"Atlas update"}')

        root = ScriptedModelClient(["The staging setting is PostgreSQL; production is unresolved."])
        result = await IterativeRuntime(
            root=root,
            sidecar=EvidenceSidecar(
                model=CallableModelClient(followup),
                policy="adaptive",
                retriever=Retriever([[hit("a", metadata=metadata)], []]),
            ),
        ).answer("Atlas", Budget(max_model_calls=2))
        supplied = json.loads(root.requests[0].messages[1].content)
        record = json.loads(supplied["evidence"])
        assert record["text"] == "Atlas uses PostgreSQL."
        assert record["metadata"] == {
            "role": "assistant",
            "date": "2031-05-16",
            "scope": {"env": "staging"},
        }
        assert result.status == "partial"
        assert result.evidence.stop_reason == "budget_exhausted"
        assert result.references[0].node_id == "a"

    asyncio.run(scenario())


def test_metadata_cannot_bypass_the_fallback_bundle_allowance():
    """Fallback admission includes applicability metadata when choosing raw excerpts."""

    async def scenario():
        """Admit the source for reading but exclude its oversized raw return to the root."""
        root = ScriptedModelClient(["Evidence is unavailable within the allowance."])
        result = await IterativeRuntime(
            root=root,
            sidecar=EvidenceSidecar(
                model=ScriptedModelClient([]),
                policy="single",
                retriever=Retriever([[hit("a", "Atlas", {"scope": "x" * 2000})]]),
            ),
        ).answer("Atlas", Budget(max_model_calls=1, max_bundle_tokens=1000))
        assert len(result.evidence.hits) == 1
        assert not result.references
        assert json.loads(root.requests[0].messages[1].content)["evidence"] == ""
        assert result.status == "partial"

    asyncio.run(scenario())


@pytest.mark.parametrize("has_evidence", [False, True])
def test_unresolved_or_missing_evidence_returns_partial_status(has_evidence):
    """An executed root call does not turn explicit evidence gaps into a completed answer."""

    async def scenario():
        """Keep both no-hit and reader-declared gaps visible in the shared answer status."""
        response = json.dumps(
            {
                "text": "Atlas uses PostgreSQL.",
                "passage_ids": ["a"],
                "unresolved": ["Production setting remains unknown."],
            }
        )
        reader = ScriptedModelClient([response] if has_evidence else [])
        result = await IterativeRuntime(
            root=ScriptedModelClient(["Insufficient production evidence."]),
            sidecar=EvidenceSidecar(
                model=reader,
                policy="single",
                retriever=Retriever([[hit("a")] if has_evidence else []]),
            ),
        ).answer("Atlas production database?")
        assert result.status == "partial"
        assert result.evidence.unresolved
        assert result.answer == "Insufficient production evidence."

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["search", "reader", "root"])
@pytest.mark.parametrize("expected_failure", [False, True])
def test_answer_failure_boundary_retains_attempts_and_available_evidence(stage, expected_failure):
    """Expected operational failures return results; unexpected errors raise with attempted usage."""

    async def scenario():
        """Inject a failure before retrieval, during reading, or after successful gathering."""
        error_type = ProviderError if expected_failure else RuntimeError

        async def fail(request):
            """Fail one admitted provider request without inventing usage."""
            raise error_type("Controlled failure")

        class FailingRetriever(Retriever):
            """Expose the same failure boundary before any model has been dispatched."""

            async def search(self, query, k):
                """Fail a counted retrieval attempt."""
                raise error_type("Controlled failure")

        retriever = FailingRetriever([]) if stage == "search" else Retriever([[hit("a")]])
        reader = (
            CallableModelClient(fail)
            if stage == "reader"
            else ScriptedModelClient([composition(["a"])])
        )
        root = (
            CallableModelClient(fail) if stage == "root" else ScriptedModelClient(["unreachable"])
        )
        runtime = IterativeRuntime(
            root=root,
            sidecar=EvidenceSidecar(
                model=reader,
                retriever=retriever,
                policy="single",
            ),
        )
        expected_calls = {"search": 0, "reader": 1, "root": 2}[stage]
        if expected_failure:
            result = await runtime.answer("Atlas database?")
            assert result.status == "failed" and result.answer == ""
            assert result.usage["model_calls"] == expected_calls
            assert result.usage["unknown_usage_calls"] == expected_calls
            assert result.usage["searches"] == 1
            assert result.trace[-1]["error_type"] == "ProviderError"
            assert result.evidence.unresolved == ["ProviderError: Controlled failure"]
            assert bool(result.references) == (stage != "search")
            if stage != "search":
                assert "Atlas uses PostgreSQL." in result.evidence.text
        else:
            with pytest.raises(RuntimeError) as caught:
                await runtime.answer("Atlas database?")
            assert caught.value.llgm_usage["model_calls"] == expected_calls
            assert caught.value.llgm_usage["searches"] == 1
            if expected_calls:
                assert caught.value.llgm_trace[-1]["status"] == "failed"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "options",
    [
        {"question": ""},
        {"question": None},
        {"question_date": []},
        {"question_date": " "},
        {"budget": {}},
    ],
)
def test_answer_rejects_invalid_public_arguments_before_dispatch(options):
    """Invalid questions, dates, and budgets raise before starting a run or retrieval."""
    reader = ScriptedModelClient([])
    root = ScriptedModelClient([])
    retriever = Retriever([])
    runtime = IterativeRuntime(
        root=root,
        sidecar=EvidenceSidecar(
            model=reader,
            retriever=retriever,
            policy="single",
        ),
    )
    with pytest.raises((SchemaError, ConfigurationError)):
        asyncio.run(runtime.answer(**{"question": "Atlas", **options}))
    assert not reader.requests and not root.requests and not retriever.calls


def test_answer_deadline_retains_admitted_evidence_and_the_timed_out_attempt():
    """An exhausted reader deadline returns a failure result without discarding prior retrieval."""

    async def scenario():
        """Let the reader consume the run deadline before a final root call can start."""
        cancelled = asyncio.Event()

        async def slow(request):
            """Signal cancellation cleanup for the provider request that exceeds its deadline."""
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        root = ScriptedModelClient([])
        result = await IterativeRuntime(
            root=root,
            sidecar=EvidenceSidecar(
                model=CallableModelClient(slow),
                retriever=Retriever([[hit("a")]]),
                policy="single",
            ),
        ).answer("Atlas", Budget(timeout_seconds=0.01))
        assert result.status == "budget_exhausted" and result.answer == ""
        assert result.references[0].node_id == "a"
        assert result.usage["model_calls"] == 1 and result.usage["unknown_usage_calls"] == 1
        assert any(event.get("status") == "timeout" for event in result.trace)
        assert result.trace[-1]["error_type"] == "BudgetExceeded"
        assert cancelled.is_set() and not root.requests

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["reader", "root"])
def test_answer_cancellation_still_raises_and_retains_dispatched_work(stage):
    """Result-based operational failures do not swallow caller cancellation of either model."""

    async def scenario():
        """Cancel a known active provider call and inspect the attached shared ledger."""
        started = asyncio.Event()

        async def waiting(request):
            """Wait for cancellation only after signaling actual provider dispatch."""
            started.set()
            await asyncio.Event().wait()

        reader = (
            CallableModelClient(waiting)
            if stage == "reader"
            else ScriptedModelClient([composition(["a"])])
        )
        root = CallableModelClient(waiting) if stage == "root" else ScriptedModelClient([])
        task = asyncio.create_task(
            IterativeRuntime(
                root=root,
                sidecar=EvidenceSidecar(
                    model=reader,
                    retriever=Retriever([[hit("a")]]),
                    policy="single",
                ),
            ).answer("Atlas")
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.llgm_usage["model_calls"] == (1 if stage == "reader" else 2)
        assert caught.value.llgm_trace[-1]["status"] == "cancelled"

    asyncio.run(scenario())
