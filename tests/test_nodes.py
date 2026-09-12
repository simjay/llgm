"""Node orchestration contracts with explicit replay transport and no Python execution.

The replay adapter recognizes a finite set of test commands and invokes real
host callbacks. It never evaluates model-generated code. Actual Python transport
is covered separately by opted-in Docker tests.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import replace

import pytest

from llgm.core.errors import BudgetExceeded, ProviderError, ReferenceResolutionError
from llgm.core.types import NodeRef, ResolvedEvidence, SourceSpan, reference_to_dict
from llgm.inference.budget import Budget, RunLedger
from llgm.inference.nodes import NodeRuntime, NodeSeed
from llgm.inference.repl import DockerREPLConfig, REPLResult
from llgm.models import CallableModelClient, ModelCapabilities, ModelResponse
from tests.node_support import (
    BAD_READ,
    CHILD,
    CYCLE,
    EDGES,
    READ,
    Models,
    ReplayFactory,
    ReplayREPL,
    finish,
)

THIRD_CHILD = 'print(query_node("c", "Find the date"))'


def third_node_replay(*, exhaust_after=None):
    """Replay a declared C query and optional exhaustion after a completed callback."""
    owner = ReplayFactory()

    class ThirdNodeReplay(ReplayREPL):
        """Recognize one additional command without evaluating generated Python."""

        async def execute(self, code):
            """Complete callback delivery before injecting an explicitly selected budget failure."""
            if code == THIRD_CHILD:
                payload = await self.callback(
                    {"op": "query_node", "node_id": "c", "question": "Find the date"}
                )
                result = REPLResult(json.dumps(payload), None, False, 1, ())
            else:
                result = await super().execute(code)
            if (exhaust_after or {}).get(self.context["node_id"]) == code:
                raise BudgetExceeded("Replay continuation allowance exhausted after callback")
            return result

    def create(context, *, config, node_callback):
        """Retain each interpreter's independent context and cleanup status."""
        session = ThirdNodeReplay(context, config=config, node_callback=node_callback, owner=owner)
        owner.sessions.append(session)
        return session

    return owner, create


class EvidenceFixture:
    """Canonical source and complete-journal provider with observable lazy access."""

    def __init__(self):
        """Separate source content from journal metadata and track callback ordering."""
        self.sources = {
            "a": "Long uncited local source sentinel. " * 30,
            "b": "eu-west-1",
            "c": "2031-04-07",
        }
        self.initialized, self.reads = [], []

    async def source_info(self, node_id, *, offset=0, limit=32):
        """Expose canonical turn handles without making source text model-visible."""
        assert node_id in self.initialized
        text = self.sources[node_id]
        turns = [
            {
                "turn_id": "turn",
                "role": "user",
                "length": len(text),
                "reference": reference_to_dict(SourceSpan(node_id, "turn", 0, len(text))),
            }
        ]
        return {
            "node_id": node_id,
            "turns": turns[offset : offset + limit],
            "offset": offset,
            "next_offset": None,
            "total_turns": 1,
        }

    async def initialize_node(self, node_id):
        """Return the entire small operational journal before any lazy read."""
        if node_id not in self.sources:
            raise ReferenceResolutionError("Missing node")
        self.initialized.append(node_id)
        return {
            "node_id": node_id,
            "journal": [{"note": "Complete journal sentinel"}],
            "unresolved": [],
            "journal_bytes": 47,
        }

    async def read_segments(self, reference):
        """Resolve exact immutable source coordinates only after journal initialization."""
        assert reference.node_id in self.initialized
        self.reads.append(reference)
        text = self.sources[reference.node_id]
        span = (
            SourceSpan(reference.node_id, "turn", 0, len(text))
            if isinstance(reference, NodeRef)
            else reference
        )
        return [ResolvedEvidence(text[span.start : span.end], span, {"role": "user"})]

    async def neighbors(self, node_id):
        """Return a primary edge independently of the journal's inline note."""
        return [NodeRef("b")] if node_id == "a" else []

    async def edge_descriptions(self, node_id):
        """Describe the fixture's stored relationship without adding inferred relevance."""
        if node_id != "a":
            return []
        return [
            {
                "edge_id": "edge-a-b",
                "source_node_id": "a",
                "reference": reference_to_dict(NodeRef("b")),
                "provenance": {
                    "origin": "user",
                    "producer": "node-test",
                    "supporting_references": [reference_to_dict(NodeRef("a"))],
                    "model": None,
                    "prompt_version": None,
                    "policy_version": None,
                },
                "applicability": {},
                "recorded_at_ms": 1,
            }
        ]


def runtime(models=None, evidence=None, factory=None, **options):
    """Use roomy character-count limits so tests can tighten one relevant boundary."""
    defaults = {
        "budget": Budget(
            max_model_calls=40,
            max_reader_calls=36,
            max_evidence_tokens=50000,
            max_context_tokens=50000,
            max_bundle_tokens=12000,
        ),
        "token_counter": len,
    }
    defaults.update(options)
    models, evidence, factory = (
        models or Models(),
        evidence or EvidenceFixture(),
        factory or ReplayFactory(),
    )
    return (
        NodeRuntime(models.main, models.reader, evidence, repl_factory=factory, **defaults),
        models,
        evidence,
        factory,
    )


def test_all_seeds_contribute_to_one_main_without_raw_local_history():
    """All admitted seeds return attributed findings before the single final main call."""

    async def scenario():
        """Collect a recursive A-to-B branch and an independent C branch under one registry."""
        engine, models, evidence, factory = runtime(Models({"a": [READ, EDGES, CHILD]}))
        result = await engine.answer("Where and when?", seeds=[NodeSeed("a"), NodeSeed("c")])
        assert result.status == "completed"
        assert {ref.node_id for ref in result.references} == {"b", "c"}
        assert len(models.main_requests) == 1
        main_context = json.loads(models.main_requests[0].messages[1].content)
        assert [branch["node_id"] for branch in main_context["branches"]] == ["a", "c"]
        assert (
            "Long uncited local source sentinel" not in models.main_requests[0].messages[1].content
        )
        initial = [request for request in models.child_requests if len(request.messages) == 2]
        assert all(
            "Complete journal sentinel" in request.messages[1].content for request in initial
        )
        assert all(
            "Long uncited local source sentinel" not in request.messages[1].content
            for request in initial
        )
        assert all(repl.closed for repl in factory.sessions)
        assert {event["target_node_id"] for event in result.trace if event["kind"] == "enter"} == {
            "a",
            "b",
            "c",
        }
        assert result.usage["model_calls"] == len(models.child_requests) + 1
        assert {
            event["operation"]["op"] for event in result.trace if event["kind"] == "node_operation"
        } == {"read", "edges", "query_node"}
        discovery = next(
            event
            for event in result.trace
            if event["kind"] == "node_result" and event["operation"] == "edges"
        )
        assert discovery["references"] == [reference_to_dict(NodeRef("b"))]
        edge_observation = next(
            json.loads(observation["stdout"])
            for request in models.child_requests
            for message in request.messages
            if message.role == "user"
            and (observation := json.loads(message.content)).get("stdout")
            and "edges" in json.loads(observation["stdout"])
        )
        assert edge_observation["edges"] == await evidence.edge_descriptions("a")
        assert evidence.reads

    asyncio.run(scenario())


def test_three_seed_limit_two_queues_third_and_bounds_model_calls():
    """A concurrency cap queues extra seeds without dropping them or sharing model permits with children."""

    async def scenario():
        """Hold the first two model requests and prove the third seed starts only after a slot opens."""
        models = Models()
        original = models.child
        entered, release = asyncio.Event(), asyncio.Event()
        active = peak = 0

        async def held(request):
            """Measure in-flight generation while delaying the first admitted pair."""
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                entered.set()
            await release.wait()
            try:
                return await original(request)
            finally:
                active -= 1

        models.reader = CallableModelClient(held)
        engine, _, evidence, factory = runtime(models, max_concurrency=2)
        task = asyncio.create_task(
            engine.answer("Question", seeds=[NodeSeed(n) for n in ("a", "b", "c")])
        )
        await asyncio.wait_for(entered.wait(), 1)
        assert set(factory.started) == {"a", "b"}
        assert "c" not in evidence.initialized
        release.set()
        result = await asyncio.wait_for(task, 2)
        assert result.status == "completed"
        assert peak == 2 and set(factory.started) == {"a", "b", "c"}

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", ["shared_calls", "last_step"])
def test_three_seeds_keep_finish_calls_and_finalize_instead_of_repeating_reads(limit):
    """Every admitted seed can select its local evidence before the shared allowance or last step ends."""

    async def scenario():
        """Keep requesting reads until the native schema explicitly restricts the next response to finish."""
        models = Models()
        finish_requests = []

        async def investigate(request):
            """Obey finish-only admission while otherwise attempting another exploratory read."""
            models.child_requests.append(request)
            context = json.loads(request.messages[1].content)
            schema = request.output_schema["properties"]["operation"]
            finish_only = schema.get("properties", {}).get("op", {}).get("enum") == ["finish"]
            if finish_only:
                finish_requests.append(context["node_id"])
                observations = [
                    json.loads(message.content)
                    for message in request.messages
                    if message.role == "user"
                ]
                payload = next(
                    json.loads(observation["stdout"])
                    for observation in reversed(observations)
                    if observation.get("stdout")
                )
                operation = json.loads(
                    finish(
                        context["node_id"] + " local finding",
                        [record["id"] for record in payload["evidence"]],
                    )
                )
            else:
                operation = {"op": "python", "code": READ}
            return ModelResponse(json.dumps({"operation": operation}))

        models.reader = CallableModelClient(
            investigate, capabilities=ModelCapabilities(structured_output=True)
        )
        budget = Budget(
            max_model_calls=7 if limit == "shared_calls" else 40,
            max_reader_calls=6 if limit == "shared_calls" else 36,
            max_context_tokens=50000,
            max_evidence_tokens=50000,
            max_bundle_tokens=12000,
        )
        engine, _, evidence, factory = runtime(
            models, budget=budget, max_concurrency=2, max_steps=2 if limit == "last_step" else 12
        )
        result = await engine.answer("Combine local facts", seeds=[NodeSeed(n) for n in "abc"])
        assert result.status == "completed"
        assert {reference.node_id for reference in result.references} == set("abc")
        assert Counter(finish_requests) == Counter("abc")
        assert len(models.child_requests) == 6 and len(models.main_requests) == 1
        assert result.usage["reader_calls"] == 6 and result.usage["model_calls"] == 7
        assert Counter(reference.node_id for reference in evidence.reads) == Counter("abc")
        assert set(factory.started) == set("abc") and all(s.closed for s in factory.sessions)

    asyncio.run(scenario())


def test_child_admission_preserves_parent_finish_when_two_child_calls_do_not_fit():
    """A child cannot consume its parent's finish reservation without its own read-and-finish allowance."""

    async def scenario():
        """Attempt recursion with only one unreserved child call and retain an explicit admission gap."""
        budget = Budget(
            max_model_calls=4,
            max_reader_calls=3,
            max_context_tokens=50000,
            max_evidence_tokens=50000,
            max_bundle_tokens=12000,
        )
        engine, models, _, factory = runtime(Models({"a": [CHILD]}), budget=budget)
        result = await engine.answer("Question", seeds=[NodeSeed("a")])
        assert result.status == "partial" and not result.references
        assert factory.started == ["a"]
        assert len(models.main_requests) == 1
        assert result.usage["reader_calls"] <= 3
        assert any("call" in gap.lower() for gap in result.evidence.unresolved)
        assert not any(
            event["kind"] == "repl_open" and event["target_node_id"] == "b"
            for event in result.trace
        )

    asyncio.run(scenario())


def test_concurrency_one_still_allows_recursive_child():
    """A parent never holds the model permit while a recursive child needs it."""

    async def scenario():
        """Complete nested execution with a single generation slot and independent REPLs."""
        engine, models, _, factory = runtime(Models({"a": [CHILD]}), max_concurrency=1)
        result = await asyncio.wait_for(engine.answer("Question", seeds=[NodeSeed("a")]), 2)
        assert result.status == "completed"
        assert [session.context["node_id"] for session in factory.sessions] == ["a", "b"]
        child_context = next(
            json.loads(request.messages[1].content)
            for request in models.child_requests
            if len(request.messages) == 2
            and json.loads(request.messages[1].content)["node_id"] == "b"
        )
        assert child_context["references"] == [reference_to_dict(SourceSpan("b", "turn", 0, 9))]
        assert "eu-west-1" not in json.dumps(child_context)
        assert all(session.closed for session in factory.sessions)

    asyncio.run(scenario())


def test_parent_budget_exhaustion_preserves_delivered_child_findings_only():
    """Exhaustion after child delivery preserves selected findings without promoting uncited local reads."""

    async def scenario():
        """Use roomy model limits and inject failure only after the child callback returns its result."""
        factory, create = third_node_replay(exhaust_after={"a": CHILD})
        engine, models, evidence, _ = runtime(Models({"a": [READ, CHILD]}), factory=create)
        result = await engine.answer("Question", seeds=[NodeSeed("a")])
        assert result.status == "partial"
        assert result.usage["reader_calls"] == 4 and result.usage["model_calls"] == 5
        assert {reference.node_id for reference in evidence.reads} == {"a", "b"}
        assert {reference.node_id for reference in result.references} == {"b"}
        assert len(models.main_requests) == 1
        main_context = json.loads(models.main_requests[0].messages[1].content)
        branch = main_context["branches"][0]
        child = next(
            event
            for event in result.trace
            if event["kind"] == "branch_return" and event["node_id"] == "b"
        )
        assert child["status"] == "completed"
        assert branch["status"] == "budget_exhausted"
        assert branch["citations"] == [record["id"] for record in child["evidence"]]
        assert child["answer"] in branch["findings"]
        assert child["invocation_id"] in branch["findings"]
        assert any("allowance" in gap for gap in branch["unresolved"])
        assert (
            "Long uncited local source sentinel" not in models.main_requests[0].messages[1].content
        )
        assert all(session.closed for session in factory.sessions)

    asyncio.run(scenario())


def test_exhausted_ancestors_forward_the_same_delivered_grandchild_findings():
    """A selected grandchild result survives budget exhaustion in both ancestors without an extra model call."""

    async def scenario():
        """Inject post-delivery failures in A and B after C finishes with its selected evidence."""

        owner, factory = third_node_replay(exhaust_after={"a": CHILD, "b": THIRD_CHILD})
        engine, models, _, _ = runtime(Models({"a": [CHILD], "b": [THIRD_CHILD]}), factory=factory)
        result = await engine.answer("Question", seeds=[NodeSeed("a")])
        assert result.status == "partial" and {ref.node_id for ref in result.references} == {"c"}
        returns = {
            event["node_id"]: event for event in result.trace if event["kind"] == "branch_return"
        }
        assert returns["c"]["status"] == "completed"
        assert returns["a"]["status"] == returns["b"]["status"] == "budget_exhausted"
        assert returns["a"]["evidence"] == returns["b"]["evidence"] == returns["c"]["evidence"]
        assert sum(bool(returns[node]["evidence"]) for node in ("b", "c")) == 2
        assert result.usage["model_calls"] == 5 and len(models.main_requests) == 1
        assert len(owner.sessions) == 3 and all(session.closed for session in owner.sessions)

    asyncio.run(scenario())


def test_local_reads_without_a_selected_child_are_not_preserved_on_exhaustion():
    """Budget fallback cannot manufacture a selected finding from a merely visible source record."""

    async def scenario():
        """Stop after the first local read before the model has selected any answer evidence."""
        _, factory = third_node_replay(exhaust_after={"a": READ})
        engine, models, evidence, _ = runtime(factory=factory)
        result = await engine.answer("Question", seeds=[NodeSeed("a")])
        assert evidence.reads and not result.references
        assert result.status == "partial"
        assert engine.last_branches[0]["status"] == "budget_exhausted"
        assert not engine.last_branches[0]["evidence"]
        assert (
            "Long uncited local source sentinel" not in models.main_requests[0].messages[1].content
        )

    asyncio.run(scenario())


def test_parent_schema_error_does_not_activate_budget_preservation():
    """Invalid parent citations still fail the branch even after a valid child result was delivered."""

    async def scenario():
        """Try to finish with an invented citation after receiving a legitimate child finding."""
        models = Models({"a": [CHILD]})
        original = models.child

        async def invalid_finish(request):
            """Inject a protocol violation only after A receives the child's actual result."""
            context = json.loads(request.messages[1].content)
            if context["node_id"] == "a" and len(request.messages) > 2:
                return ModelResponse(finish("Unsupported parent finding", ["not-visible"]))
            return await original(request)

        models.reader = CallableModelClient(invalid_finish)
        engine, _, _, _ = runtime(models)
        result = await engine.answer("Question", seeds=[NodeSeed("a")])
        assert result.status == "partial" and not result.references
        assert engine.last_branches[0]["status"] == "failed"
        assert not engine.last_branches[0]["evidence"]
        assert any(
            event["kind"] == "node_result"
            and event["operation"] == "query_node"
            and event["citations"]
            for event in result.trace
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["transport", "aggregate_bundle", "main_context"])
def test_budget_preservation_respects_delivery_and_aggregate_return_limits(boundary):
    """Undelivered or oversized child findings cannot bypass transport, bundle, or main-context admission."""

    async def scenario():
        """Make child answers fit their local bundle, then deny their parent transfer or aggregate fallback."""
        plans = {"a": [CHILD, THIRD_CHILD] if boundary == "aggregate_bundle" else [CHILD]}
        models = Models(plans)
        original = models.child
        answer = "\\" * 1700 if boundary == "main_context" else "Child finding " * 40

        async def long_findings(request):
            """Select actual child evidence while varying only the returned finding size."""
            context = json.loads(request.messages[1].content)
            if context["node_id"] != "a" and len(request.messages) > 2:
                payload = json.loads(json.loads(request.messages[-1].content)["stdout"])
                return ModelResponse(
                    finish(answer, [record["id"] for record in payload["evidence"]])
                )
            return await original(request)

        models.reader = CallableModelClient(long_findings)
        owner, factory = third_node_replay(
            exhaust_after={"a": THIRD_CHILD if boundary == "aggregate_bundle" else CHILD}
        )
        budget = Budget(
            max_model_calls=40,
            max_reader_calls=36,
            max_context_tokens=8000 if boundary == "main_context" else 50000,
            max_evidence_tokens=50000,
            max_bundle_tokens=1000 if boundary == "aggregate_bundle" else 7000,
            max_output_tokens=1000,
        )
        config = (
            DockerREPLConfig(max_response_bytes=512)
            if boundary == "transport"
            else DockerREPLConfig()
        )
        engine, _, _, _ = runtime(models, factory=factory, budget=budget, repl_config=config)
        result = await engine.answer("Question", seeds=[NodeSeed("a")])
        assert result.status == "partial" and not result.references
        assert len(models.main_requests) == 1
        branch = engine.last_branches[0]
        assert branch["status"] == "budget_exhausted" and not branch["evidence"]
        delivered = [
            event
            for event in result.trace
            if event["kind"] == "node_result"
            and event["operation"] == "query_node"
            and event["citations"]
        ]
        if boundary == "transport":
            assert not delivered
            assert any(event["kind"] == "node_operation_failed" for event in result.trace)
        else:
            assert len(delivered) == (2 if boundary == "aggregate_bundle" else 1)
            assert any("omitt" in gap.lower() for gap in branch["unresolved"])
        assert answer not in models.main_requests[0].messages[1].content
        assert all(session.closed for session in owner.sessions)

    asyncio.run(scenario())


def test_overlapping_seed_branches_keep_isolated_invocations_and_shared_evidence():
    """A direct B seed and A's B child remain distinct investigations of the same canonical evidence."""

    async def scenario():
        """Revisit B without caching its result or treating two paths as independent sources."""
        engine, models, evidence, factory = runtime(Models({"a": [CHILD]}))
        result = await engine.answer("Question", seeds=[NodeSeed("a"), NodeSeed("b")])
        assert result.status == "completed"
        b_sessions = [session for session in factory.sessions if session.context["node_id"] == "b"]
        assert len(b_sessions) == 2
        assert b_sessions[0].context is not b_sessions[1].context
        assert {session.context["question"] for session in b_sessions} == {
            "Question",
            "Find the region",
        }
        assert len([reference for reference in evidence.reads if reference.node_id == "b"]) == 2
        main_request = models.main_requests[0]
        payload = json.loads(main_request.messages[1].content)
        branches = payload["branches"]
        assert [branch["node_id"] for branch in branches] == ["a", "b"]
        assert branches[0]["invocation_id"] != branches[1]["invocation_id"]
        assert branches[0]["citations"] == branches[1]["citations"]
        assert len(payload["evidence"]) == 1
        assert len(result.references) == 1
        assert all(session.closed for session in factory.sessions)

    asyncio.run(scenario())


def test_main_presents_attributed_sources_before_findings_without_changing_evidence():
    """Lift dates and roles for synthesis while preserving every canonical record and metadata field."""

    async def scenario():
        """Present a user statement before an earlier-listed assistant echo and retain both citations."""
        evidence = EvidenceFixture()
        original = evidence.read_segments
        metadata = {
            "b": {
                "role": "assistant",
                "source_metadata": {"date": "2031-04-06", "scope": {"project": "cedar"}},
                "timestamp_ms": 1933200000000,
                "amendments": [{"entry_id": "retained-provenance", "record_kind": "overwrite"}],
            },
            "c": {"role": "user", "source_metadata": {"date": "2031-04-07"}},
        }
        before = json.loads(json.dumps(metadata))

        async def attributed(reference):
            """Attach source attribution and amendment metadata to exact fixture spans."""
            return [
                replace(record, metadata=metadata[reference.node_id])
                for record in await original(reference)
            ]

        evidence.read_segments = attributed
        engine, models, _, _ = runtime(evidence=evidence)
        result = await engine.answer("Which date applies?", seeds=[NodeSeed("b"), NodeSeed("c")])
        assert result.status == "completed"
        payload = json.loads(models.main_requests[0].messages[1].content)
        assert list(payload).index("evidence") < list(payload).index("branches")
        assert [record["role"] for record in payload["evidence"]] == ["user", "assistant"]
        originals = {
            record["id"]: record for branch in engine.last_branches for record in branch["evidence"]
        }
        for record in payload["evidence"]:
            source = originals[record["id"]]
            assert list(record).index("source_date") < list(record).index("text")
            assert record["source_date"] == source["metadata"]["source_metadata"]["date"]
            assert record["timestamp_ms"] == source["metadata"].get("timestamp_ms")
            assert {key: record[key] for key in source} == source
            assert set(source) == {"id", "text", "references", "metadata"}
        for branch in payload["branches"]:
            original_branch = next(
                item
                for item in engine.last_branches
                if item["invocation_id"] == branch["invocation_id"]
            )
            assert branch["findings"] == original_branch["answer"]
            assert branch["citations"] == [record["id"] for record in original_branch["evidence"]]
            assert "evidence" not in branch and "answer" not in branch
        returned = {
            record["id"]: record for record in map(json.loads, result.evidence.text.split("\n\n"))
        }
        assert returned == originals
        assert metadata == before

    asyncio.run(scenario())


@pytest.mark.parametrize("metadata_key", ["note", "date"])
def test_main_reserves_the_expanded_attribution_presentation(metadata_key):
    """Lifted source dates count toward main context even when the canonical branch fits its bundle."""

    async def scenario():
        """Keep the same metadata bytes, lifting them only when they are the source date."""
        evidence, models, owner = EvidenceFixture(), Models(), ReplayFactory()
        original = evidence.read_segments

        async def attributed(reference):
            """Return a short exact passage with large escaped source metadata."""
            return [
                replace(
                    record, metadata={"role": "user", "source_metadata": {metadata_key: "\\" * 900}}
                )
                for record in await original(reference)
            ]

        class QuietReplay(ReplayREPL):
            """Keep source metadata in Python and print only learned citation IDs."""

            async def execute(self, code):
                """Deliver a real read without using its full metadata as the reader observation."""
                assert code == READ
                payload = await self.callback(
                    {"op": "read", "reference": self.context["references"][0]}
                )
                visible = {"evidence": [{"id": record["id"]} for record in payload["evidence"]]}
                return REPLResult(json.dumps(visible), None, False, 1, ())

        def create(context, *, config, node_callback):
            """Inject the finite transport while preserving the runtime's actual callback admission."""
            session = QuietReplay(context, config=config, node_callback=node_callback, owner=owner)
            owner.sessions.append(session)
            return session

        evidence.read_segments = attributed
        engine, _, _, _ = runtime(
            models,
            evidence,
            create,
            budget=Budget(max_context_tokens=10000, max_output_tokens=1000, max_bundle_tokens=7000),
        )
        result = await engine.answer("Region?", seeds=[NodeSeed("b")])
        assert len(models.main_requests) == 1
        request = models.main_requests[0]
        assert RunLedger(engine.budget, len).context_size(request.messages) + 1000 <= 10000
        if metadata_key == "date":
            assert result.status == "partial" and not result.references
            assert "synthesis allowance" in str(result.evidence.unresolved)
            assert json.loads(request.messages[1].content)["evidence"] == []
        else:
            assert result.status == "completed"
            assert result.references == (SourceSpan("b", "turn", 0, 9),)
        assert all(session.closed for session in owner.sessions)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["later_segment", "read_transport", "child_transport"])
def test_failed_callback_payload_does_not_make_registered_citations_visible(failure):
    """Guessed IDs from undelivered partial reads or oversized returns are never admissible citations."""

    async def scenario():
        """Register evidence before failing admission, then deliberately try to cite the hidden ID."""
        models, evidence = Models(), EvidenceFixture()

        async def child(request):
            """Guess the first registry ID after an error instead of learning it from delivered evidence."""
            context = json.loads(request.messages[1].content)
            if len(request.messages) == 2:
                code = CHILD if failure == "child_transport" and context["node_id"] == "a" else READ
                return ModelResponse(json.dumps({"op": "python", "code": code}))
            if failure == "child_transport" and context["node_id"] == "b":
                observed = json.loads(json.loads(request.messages[-1].content)["stdout"])
                return ModelResponse(
                    finish(
                        "Detailed child findings " * 60,
                        [record["id"] for record in observed["evidence"]],
                    )
                )
            return ModelResponse(finish("Guessed hidden evidence", ["e1"]))

        models.reader = CallableModelClient(child)
        options = {}
        if failure == "later_segment":

            async def segments(reference):
                """The first short segment fits; the later large segment exhausts the same read."""
                return [
                    ResolvedEvidence("ok", SourceSpan("b", "turn", 0, 2), {}),
                    ResolvedEvidence("x" * 3000, SourceSpan("b", "turn", 2, 3002), {}),
                ]

            evidence.read_segments = segments
            options["budget"] = Budget(
                max_model_calls=10,
                max_reader_calls=9,
                max_evidence_tokens=1000,
                max_context_tokens=16000,
                max_bundle_tokens=6000,
            )
        else:
            options["repl_config"] = DockerREPLConfig(max_response_bytes=512)
            if failure == "read_transport":
                evidence.sources["b"] = "source sentinel " * 80
        engine, _, _, _ = runtime(models, evidence, **options)
        result = await engine.answer(
            "Question", seeds=[NodeSeed("a" if failure == "child_transport" else "b")]
        )
        assert result.status == "partial"
        assert not result.references
        assert "Citation was not accessed" in str(result.evidence.unresolved)
        assert any(
            event["kind"] == "node_operation_failed" and event["error_type"] == "BudgetExceeded"
            for event in result.trace
        )
        assert all(not branch["evidence"] for branch in engine.last_branches)
        assert len(models.main_requests) == 1

    asyncio.run(scenario())


def test_failed_branch_keeps_successful_findings_and_cannot_be_hidden_by_main():
    """A branch's provider failure survives main wording while successful sibling evidence remains usable."""

    async def scenario():
        """Fail one branch and make the main omit its unresolved field intentionally."""
        models = Models()
        original = models.child

        async def child(request):
            """Inject a known provider failure into only node A."""
            if json.loads(request.messages[1].content)["node_id"] == "a":
                raise ProviderError("deliberate branch failure")
            return await original(request)

        async def main(request):
            """Return the good citation without volunteering the failed branch's gap."""
            models.main_requests.append(request)
            records = json.loads(request.messages[1].content)["evidence"]
            return ModelResponse(
                finish("Only supported findings", [record["id"] for record in records])
            )

        models.reader, models.main = CallableModelClient(child), CallableModelClient(main)
        engine, _, _, factory = runtime(models)
        result = await engine.answer("Question", seeds=[NodeSeed("a"), NodeSeed("b")])
        assert result.status == "partial"
        assert {ref.node_id for ref in result.references} == {"b"}
        assert "deliberate branch failure" in str(result.evidence.unresolved)
        assert {branch["status"] for branch in engine.last_branches} == {"completed", "failed"}
        assert all(session.closed for session in factory.sessions)

    asyncio.run(scenario())


def test_repaired_read_schema_error_does_not_force_a_final_gap():
    """A malformed reference remains in the trace without tainting a later supported answer."""

    async def scenario():
        """Recover by copying the complete supplied reference after one missing-type callback."""
        engine, models, evidence, _ = runtime(Models({"b": [BAD_READ, READ]}))
        result = await engine.answer("Region?", seeds=[NodeSeed("b")])
        assert result.status == "completed" and not result.evidence.unresolved
        assert evidence.reads == [SourceSpan("b", "turn", 0, 9)]
        assert result.references == (SourceSpan("b", "turn", 0, 9),)
        assert any(
            event["kind"] == "node_operation_failed" and event["error_type"] == "SchemaError"
            for event in result.trace
        )
        assert engine.last_branches[0]["required_gaps"] == []
        assert len(models.main_requests) == 1

    asyncio.run(scenario())


def test_main_can_resolve_a_local_missing_fact_from_another_seed():
    """A delegate's local absence statement is visible to synthesis without forcing global incompleteness."""

    async def scenario():
        """Leave A's local gap in its branch while the main answers from B's observed region."""
        models = Models()
        original = models.child

        async def local(request):
            """Report a local limitation only after actually inspecting A's source."""
            context = json.loads(request.messages[1].content)
            if context["node_id"] == "a" and len(request.messages) > 2:
                return ModelResponse(
                    finish("No region in this node", [], ["This node has no region."])
                )
            return await original(request)

        async def synthesize(request):
            """Use B's evidence and explicitly resolve the irrelevant local absence in A."""
            models.main_requests.append(request)
            branches = json.loads(request.messages[1].content)["branches"]
            selected = next(branch for branch in branches if branch["node_id"] == "b")
            return ModelResponse(finish("eu-west-1", selected["citations"]))

        models.reader, models.main = CallableModelClient(local), CallableModelClient(synthesize)
        engine, _, _, _ = runtime(models)
        result = await engine.answer("Region?", seeds=[NodeSeed("a"), NodeSeed("b")])
        assert result.status == "completed" and not result.evidence.unresolved
        assert {reference.node_id for reference in result.references} == {"b"}
        branch = next(branch for branch in engine.last_branches if branch["node_id"] == "a")
        assert "This node has no region." in branch["unresolved"]
        assert branch["required_gaps"] == []
        assert "This node has no region." in models.main_requests[0].messages[1].content

    asyncio.run(scenario())


def test_journal_gap_is_attributed_and_cannot_be_hidden_by_final_main():
    """Host-detected journal uncertainty remains mandatory even when the model omits it."""

    async def scenario():
        """Return a real local citation while refusing to let synthesis erase its unresolved journal."""
        evidence, models = EvidenceFixture(), Models()
        initialize = evidence.initialize_node

        async def unresolved_journal(node_id):
            """Keep the complete fixture journal and add an explicit operational uncertainty."""
            return {
                **await initialize(node_id),
                "unresolved": ["Replacement authority is unresolved"],
            }

        async def synthesize(request):
            """Deliberately omit the mandatory gap from otherwise valid cited synthesis."""
            models.main_requests.append(request)
            evidence = json.loads(request.messages[1].content)["evidence"]
            return ModelResponse(
                finish(
                    "eu-west-1",
                    [record["id"] for record in evidence],
                )
            )

        evidence.initialize_node = unresolved_journal
        models.main = CallableModelClient(synthesize)
        engine, _, _, _ = runtime(models, evidence)
        result = await engine.answer("Region?", seeds=[NodeSeed("b")])
        assert result.status == "partial"
        required = engine.last_branches[0]["required_gaps"]
        assert any(
            "Replacement authority" in gap and "b" in gap and "n1" in gap for gap in required
        )
        assert set(required).issubset(result.evidence.unresolved)

    asyncio.run(scenario())


def test_failed_child_gap_reaches_main_when_parent_and_main_omit_it():
    """A failed recursive investigation remains an attributed mandatory gap through parent synthesis."""

    async def scenario():
        """Make a child provider fail, then have both surviving models omit the failure in their finishes."""
        models = Models({"a": [CHILD]})
        original = models.child

        async def child(request):
            """Fail B and deliberately suppress its operational gap in A's model-selected findings."""
            context = json.loads(request.messages[1].content)
            if context["node_id"] == "b":
                raise ProviderError("Child provider unavailable")
            if len(request.messages) > 2:
                return ModelResponse(finish("Unknown from this investigation"))
            return await original(request)

        async def main(request):
            """Try to return an apparently complete answer despite the failed child branch."""
            models.main_requests.append(request)
            return ModelResponse(finish("Unknown"))

        models.reader, models.main = CallableModelClient(child), CallableModelClient(main)
        engine, _, _, _ = runtime(models)
        result = await engine.answer("Region?", seeds=[NodeSeed("a")])
        assert result.status == "partial" and not result.references
        assert any(
            "Child provider unavailable" in gap and "b" in gap and "n2" in gap
            for gap in result.evidence.unresolved
        )
        assert engine.last_branches[0]["required_gaps"]
        assert len(models.main_requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("native_main", [False, True])
def test_empty_answer_with_explicit_abstention_is_valid_after_inspection(native_main):
    """An inspected but insufficient source permits an empty answer with an explicit unresolved reason."""

    async def scenario():
        """Abstain in the delegate and final main without treating empty text as a schema failure."""
        models = Models()
        original = models.child

        async def abstain(request):
            """Inspect first, then explain why no supported answer is available."""
            if len(request.messages) > 2:
                return ModelResponse(finish("", [], ["No requested fact in this source"]))
            return await original(request)

        async def main(request):
            """Return the same explicit abstention as a valid final result."""
            models.main_requests.append(request)
            return ModelResponse(finish("", [], ["No supported answer available"]))

        models.reader = CallableModelClient(abstain)
        models.main = CallableModelClient(
            main, capabilities=ModelCapabilities(structured_output=native_main)
        )
        engine, _, evidence, _ = runtime(models)
        result = await engine.answer("Missing fact?", seeds=[NodeSeed("b")])
        assert result.status == "partial" and result.answer == "" and not result.references
        assert evidence.reads and result.evidence.unresolved == ["No supported answer available"]
        assert engine.last_branches[0]["status"] == "partial"
        assert not any(event.get("error_type") == "SchemaError" for event in result.trace)
        assert len(models.main_requests) == 1
        schema = models.main_requests[0].output_schema
        if native_main:
            assert schema["properties"]["citations"] == {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 0,
            }
        else:
            assert schema is None

    asyncio.run(scenario())


def test_native_main_schema_admits_only_selected_branch_evidence_ids():
    """Native synthesis can cite selected returned evidence without exposing unselected local reads."""

    async def scenario():
        """Read A, select its child's B finding, and restrict the sole main request accordingly."""
        models = Models({"a": [READ, CHILD]})
        models.main = CallableModelClient(
            models.synthesize, capabilities=ModelCapabilities(structured_output=True)
        )
        engine, _, evidence, _ = runtime(models)
        result = await engine.answer("Where?", seeds=[NodeSeed("a")])
        assert result.status == "completed"
        assert {reference.node_id for reference in evidence.reads} == {"a", "b"}
        assert {reference.node_id for reference in result.references} == {"b"}
        assert len(models.main_requests) == 1
        request = models.main_requests[0]
        branch = json.loads(request.messages[1].content)["branches"][0]
        identifiers = branch["citations"]
        assert request.output_schema["properties"]["citations"]["items"] == {
            "type": "string",
            "enum": identifiers,
        }
        assert branch["node_id"] not in identifiers
        assert "Long uncited local source sentinel" not in request.messages[1].content
        event = next(
            event for event in result.trace if event["kind"] == "model" and event["role"] == "main"
        )
        counter = RunLedger(engine.budget, len)
        assert event["context_accounting_units"] == counter.context_size(
            request.messages, output_schema=request.output_schema
        )
        assert result.usage["model_calls"] == len(models.child_requests) + 1

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_citation", ["node_id", "unselected_evidence"])
def test_native_main_citation_schema_keeps_host_visibility_validation(invalid_citation):
    """A native adapter returning a node handle or unselected evidence still fails without repair."""

    async def scenario():
        """Deliberately violate native output constraints after observing a real local read."""
        models = Models({"a": [READ, CHILD]})

        async def wrong_citation(request):
            """Return an identifier outside the main schema while retaining the actual request."""
            models.main_requests.append(request)
            branch = json.loads(request.messages[1].content)["branches"][0]
            if invalid_citation == "node_id":
                citation = branch["node_id"]
            else:
                records = [
                    record
                    for child_request in models.child_requests
                    for message in child_request.messages
                    if message.role == "user"
                    for observation in [json.loads(message.content)]
                    if observation.get("stdout")
                    for record in json.loads(observation["stdout"]).get("evidence", [])
                    if record["references"][0]["node_id"] == "a"
                ]
                citation = records[0]["id"]
            assert citation not in request.output_schema["properties"]["citations"]["items"]["enum"]
            return ModelResponse(finish("eu-west-1", [citation]))

        models.main = CallableModelClient(
            wrong_citation, capabilities=ModelCapabilities(structured_output=True)
        )
        engine, _, _, _ = runtime(models)
        result = await engine.answer("Where?", seeds=[NodeSeed("a")])
        assert result.status == "failed" and not result.references
        assert result.answer == ""
        assert result.evidence.unresolved == ["Main citation was not returned by a node branch"]
        assert len(models.main_requests) == 1
        assert result.usage["model_calls"] == len(models.child_requests) + 1

    asyncio.run(scenario())


def test_native_citation_schema_growth_is_reserved_before_branch_return():
    """Schema admission units can reject selected findings while preserving one final abstention call."""

    def counter(text):
        """Make a nonempty citation enum costly to isolate schema growth from source-message size."""
        value = json.loads(text) if text.startswith("{") else {}
        citations = value.get("properties", {}).get("citations", {})
        return len(text) + (50000 if "enum" in citations.get("items", {}) else 0)

    async def scenario():
        """Keep all messages small but make the selected-ID schema exceed synthesis capacity."""
        models = Models()
        models.main = CallableModelClient(
            models.synthesize, capabilities=ModelCapabilities(structured_output=True)
        )
        engine, _, evidence, _ = runtime(models, token_counter=counter)
        result = await engine.answer("Where?", seeds=[NodeSeed("b")])
        assert evidence.reads
        assert result.status == "partial" and not result.references
        assert any("synthesis allowance" in gap for gap in result.evidence.unresolved)
        assert len(models.main_requests) == 1
        request = models.main_requests[0]
        assert request.output_schema["properties"]["citations"]["maxItems"] == 0
        assert json.loads(request.messages[1].content)["evidence"] == []
        ledger = RunLedger(engine.budget, counter)
        assert (
            ledger.context_size(request.messages, output_schema=request.output_schema)
            + engine.budget.max_output_tokens
            <= engine.budget.max_context_tokens
        )

    asyncio.run(scenario())


def test_premature_finish_receives_a_corrective_observation_then_reads():
    """A model cannot abandon an available source before attempting access when exploration remains."""

    async def scenario():
        """Recover from an initial false-unavailability finish within the same bounded invocation."""
        models = Models()
        calls = 0

        async def premature(request):
            """Stop once prematurely, then use the host correction to inspect the supplied handle."""
            nonlocal calls
            calls += 1
            models.child_requests.append(request)
            if calls == 1:
                return ModelResponse(finish("Unavailable", [], ["Source result was not supplied"]))
            if calls == 2:
                observation = json.loads(request.messages[-1].content)
                assert observation.get("error") or observation.get("instruction")
                return ModelResponse(json.dumps({"op": "python", "code": READ}))
            observation = json.loads(json.loads(request.messages[-1].content)["stdout"])
            return ModelResponse(
                finish("eu-west-1", [record["id"] for record in observation["evidence"]])
            )

        models.reader = CallableModelClient(premature)
        engine, _, evidence, _ = runtime(models)
        result = await engine.answer("Region?", seeds=[NodeSeed("b")])
        assert result.status == "completed" and not result.evidence.unresolved
        assert calls == 3 and len(models.main_requests) == 1
        assert evidence.reads == [SourceSpan("b", "turn", 0, 9)]

    asyncio.run(scenario())


def test_active_cycle_is_explicit_and_does_not_start_another_interpreter():
    """Repeated active node questions stop recursively and preserve the unresolved branch."""

    async def scenario():
        """Request the same node and question through the real callback dispatcher."""
        engine, _, _, factory = runtime(Models({"a": [CYCLE]}))
        result = await engine.answer("Question", seeds=[NodeSeed("a")])
        assert result.status == "partial"
        assert "Repeated active node request" in str(result.evidence.unresolved)
        assert len(factory.sessions) == 1

    asyncio.run(scenario())


def test_initial_retrieval_ledger_and_final_call_reservation_are_shared():
    """Injected retrieval work remains charged and two node calls leave one final synthesis call."""

    async def scenario():
        """Use exactly the remaining three calls without resetting retrieval time or counters."""
        budget = Budget(
            max_model_calls=3,
            max_reader_calls=2,
            max_searches=1,
            max_evidence_tokens=10000,
            max_bundle_tokens=6000,
        )
        ledger = RunLedger(budget, len, reserve_main=True)
        ledger.searches = 1
        ledger.events.append(
            {
                "kind": "seed_selection",
                "selected": ["b"],
                "skipped": [{"node_id": "c", "reason": "seed_limit"}],
            }
        )
        started = ledger.started
        engine, models, _, _ = runtime(budget=replace(budget, timeout_seconds=100))
        result = await engine.answer("Question", seeds=[NodeSeed("b")], ledger=ledger)
        assert result.status == "partial"
        assert "seed_limit" in str(result.evidence.unresolved)
        assert result.usage["model_calls"] == 3 and result.usage["searches"] == 1
        assert ledger.started == started
        assert "seed_limit" in models.main_requests[0].messages[1].content

    asyncio.run(scenario())


def test_node_collection_deadline_leaves_time_for_final_synthesis():
    """A slow node is stopped before the shared deadline so the main can report partial findings."""

    async def scenario():
        """Spend the collection slice in a hanging generation call and still invoke the main once."""
        budget = Budget(
            max_model_calls=4,
            max_reader_calls=3,
            timeout_seconds=0.4,
            max_evidence_tokens=10000,
            max_bundle_tokens=6000,
        )
        engine, models, _, factory = runtime(budget=budget)

        async def slow(request):
            """Require collection cancellation instead of voluntarily returning."""
            await asyncio.Event().wait()

        engine.reader_model = CallableModelClient(slow)
        result = await engine.answer("Question", seeds=[NodeSeed("b")])
        assert result.status == "partial"
        assert len(models.main_requests) == 1
        assert result.usage["model_calls"] == 2
        assert any(event.get("status") == "timeout" for event in result.trace)
        assert (
            next(event for event in result.trace if event["kind"] == "scheduling")[
                "final_time_reserve_seconds"
            ]
            > 0
        )
        assert all(session.closed for session in factory.sessions)

    asyncio.run(scenario())


def test_unresolved_amendment_cannot_expose_stale_text_or_disappear_from_result():
    """A blocked effective source region stays blank and forces a visible unresolved result."""

    async def scenario():
        """Provide an unavailable replacement as an explicit gap instead of old source bytes."""
        evidence = EvidenceFixture()

        async def blocked(reference):
            """Return the query layer's canonical empty-region and amendment-gap contract."""
            return [
                ResolvedEvidence(
                    "",
                    SourceSpan(reference.node_id, "turn", 0, 4),
                    {"unresolved": ["Replacement unavailable"]},
                )
            ]

        evidence.read_segments = blocked
        engine, models, _, _ = runtime(evidence=evidence)
        result = await engine.answer("Question", seeds=[NodeSeed("a")])
        assert result.status == "partial"
        assert "Replacement unavailable" in str(result.evidence.unresolved)
        assert "Long uncited local source sentinel" not in str(models.main_requests)

    asyncio.run(scenario())


def test_programming_error_and_cancellation_propagate_after_all_cleanup():
    """Unexpected host failures and cancellation never become successful or ordinary partial answers."""

    async def scenario():
        """Exercise both a callback bug and cancellation while another node is active."""
        engine, _, evidence, factory = runtime()

        async def broken(reference):
            """Inject an unexpected host implementation failure."""
            raise RuntimeError("programming bug")

        evidence.read_segments = broken
        with pytest.raises(RuntimeError, match="programming bug"):
            await engine.answer("Question", seeds=[NodeSeed("a"), NodeSeed("b")])
        assert all(session.closed for session in factory.sessions)
        entered = asyncio.Event()

        async def slow(request):
            """Block model work until the owning answer is cancelled."""
            entered.set()
            await asyncio.Event().wait()

        engine, _, _, factory = runtime()
        engine.reader_model = CallableModelClient(slow)
        task = asyncio.create_task(engine.answer("Question", seeds=[NodeSeed("a"), NodeSeed("b")]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(session.closed for session in factory.sessions)
        assert any(event.get("status") == "cancelled" for event in engine.last_trace)

    asyncio.run(scenario())


def test_complete_journal_overflow_stops_before_model_or_interpreter_start():
    """An oversized complete journal fails explicitly instead of truncating local guidance."""

    async def scenario():
        """Reject the node initializer's size boundary while preserving its failed seed outcome."""
        evidence = EvidenceFixture()

        async def oversized(node_id):
            """Model the query layer's complete-journal size rejection."""
            raise BudgetExceeded("Complete journal exceeds max_journal_bytes")

        evidence.initialize_node = oversized
        engine, models, _, factory = runtime(evidence=evidence)
        result = await engine.answer("Question", seeds=[NodeSeed("a")])
        assert result.status == "partial"
        assert not models.child_requests and not factory.sessions
        assert "max_journal_bytes" in str(result.evidence.unresolved)

    asyncio.run(scenario())


def test_final_operation_is_reserved_after_later_branch_exhaustion():
    """A later seed cannot spend the final synthesis operation after an earlier seed succeeds."""

    async def scenario():
        """Exhaust collection operations in branch two and synthesize branch one's retained evidence."""
        engine, models, _, _ = runtime(max_operations=6, max_concurrency=1)
        result = await engine.answer("Question", seeds=[NodeSeed("b"), NodeSeed("c")])
        assert result.status == "partial"
        assert result.usage["operations"] == 6
        assert len(models.main_requests) == 1
        assert {ref.node_id for ref in result.references} == {"b"}
        assert "operation allowance" in str(result.evidence.unresolved)

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["\\" * 2500, "🧭" * 1600], ids=["escaped", "utf8"])
def test_oversized_serialized_seed_return_leaves_final_main_context(source):
    """Escaping and UTF-8 accounting cannot turn an admitted branch into an oversized final prompt."""

    async def scenario():
        """Keep source in the interpreter, print only citation IDs, and reject its oversized return explicitly."""
        evidence, models, factory = EvidenceFixture(), Models(), ReplayFactory()
        evidence.sources["b"] = source

        class QuietReplay(ReplayREPL):
            """Replay a read whose full text stays external to model messages."""

            async def execute(self, code):
                """Deliver the evidence to the interpreter but print only its learned citation IDs."""
                assert code == READ
                payload = await self.callback(
                    {"op": "read", "reference": self.context["references"][0]}
                )
                observation = {"evidence": [{"id": record["id"]} for record in payload["evidence"]]}
                return REPLResult(json.dumps(observation), None, False, 1, ())

        def create(context, *, config, node_callback):
            """Use an explicit no-execution replay session for the hidden-source scenario."""
            session = QuietReplay(
                context, config=config, node_callback=node_callback, owner=factory
            )
            factory.sessions.append(session)
            return session

        engine = NodeRuntime(
            models.main,
            models.reader,
            evidence,
            budget=Budget(
                max_context_tokens=8000,
                max_output_tokens=1000,
                max_bundle_tokens=7000,
                max_evidence_tokens=16000,
            ),
            token_counter=lambda text: len(text.encode("utf-8")),
            repl_factory=create,
        )
        result = await engine.answer("Question", seeds=[NodeSeed("b")])
        assert result.status == "partial"
        assert "synthesis allowance" in str(result.evidence.unresolved)
        assert len(models.main_requests) == 1
        assert not result.references
        assert all(session.closed for session in factory.sessions)

    asyncio.run(scenario())


@pytest.mark.parametrize("with_passage", [False, True])
def test_source_info_allows_small_read_without_loading_huge_node(with_passage):
    """Node and retrieved-passage seeds receive turn roles without loading a huge source."""

    async def scenario():
        """Replay metadata pagination and a narrow read while the full node exceeds every text allowance."""
        info_code = "info = source_info(limit=1); print(info)"
        slice_code = 'ref = dict(info["turns"][0]["reference"]); ref["end"] = 9; print(read(ref))'
        evidence = EvidenceFixture()
        evidence.sources["b"] = "eu-west-1" + " huge unread source sentinel" * 50000

        async def source_info(node_id, *, offset, limit):
            """Return only canonical turn coordinates, with no source text copied into metadata."""
            assert node_id in evidence.initialized
            return {
                "node_id": node_id,
                "turns": [
                    {
                        "turn_id": "turn",
                        "role": "user",
                        "length": len(evidence.sources[node_id]),
                        "reference": reference_to_dict(
                            SourceSpan(node_id, "turn", 0, len(evidence.sources[node_id]))
                        ),
                    }
                ],
                "offset": offset,
                "next_offset": None,
                "total_turns": 1,
            }

        evidence.source_info = source_info

        class MetadataReplay(ReplayREPL):
            """Replay two declared metadata/read commands without evaluating Python."""

            async def execute(self, code):
                """Use the returned address to request a source slice through the real callback."""
                if code == info_code:
                    payload = await self.callback(
                        {"op": "source_info", "node_id": None, "offset": 0, "limit": 1}
                    )
                    self.address = payload["turns"][0]["reference"]
                elif code == slice_code:
                    payload = await self.callback(
                        {"op": "read", "reference": {**self.address, "end": 9}}
                    )
                else:
                    raise AssertionError("Unknown replay command")
                return REPLResult(json.dumps(payload), None, False, 1, ())

        factory = ReplayFactory()

        def create(context, *, config, node_callback):
            """Inject metadata transport only for this explicit deterministic test."""
            session = MetadataReplay(
                context, config=config, node_callback=node_callback, owner=factory
            )
            factory.sessions.append(session)
            return session

        budget = Budget(
            max_model_calls=5,
            max_reader_calls=4,
            max_evidence_tokens=5000,
            max_context_tokens=16000,
        )
        models = Models({"b": [info_code, slice_code]})
        engine = NodeRuntime(
            models.main,
            models.reader,
            evidence,
            repl_factory=create,
            budget=budget,
            token_counter=len,
        )
        references = (SourceSpan("b", "turn", 0, 9),) if with_passage else ()
        result = await engine.answer("Region?", seeds=[NodeSeed("b", references)])
        assert result.status == "completed"
        context = factory.sessions[0].context
        assert context["source_page"]["turns"][0]["role"] == "user"
        if with_passage:
            assert context["references"] == [reference_to_dict(references[0])]
        assert evidence.reads == [SourceSpan("b", "turn", 0, 9)]
        assert "huge unread source sentinel" not in str(models.child_requests) + str(
            models.main_requests
        )
        assert result.usage["evidence_accounting_units"] < 5000

    asyncio.run(scenario())
