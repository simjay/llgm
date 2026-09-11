"""Executor contracts: deterministic choices exercise real persisted evidence.

Scripted models control only operation choices and failure injection. These
tests establish routing/accounting, not model reasoning accuracy. Live model
behavior is tested separately against pinned benchmark sources.
"""

import asyncio
import json
from dataclasses import replace

import pytest

from llgm import Budget, Conversation, JournalRef, NodeRef, Provenance, Workspace
from llgm.core.errors import ConfigurationError
from llgm.core.types import reference_to_dict
from llgm.inference.recursive import RecursiveRuntime
from llgm.models import CallableModelClient, ModelResponse, ScriptedModelClient, Usage


def response(op, **fields):
    """Serialize a single scripted runtime operation with its explicit fields."""
    return json.dumps({"op": op, **fields})


def read(node):
    """Request one complete immutable source node."""
    return response("read", reference=reference_to_dict(NodeRef(node)))


def delegate(question, *nodes):
    """Delegate a subquestion with explicit immutable source handles."""
    return response(
        "delegate", question=question, references=[reference_to_dict(NodeRef(n)) for n in nodes]
    )


def query_node(node_id, question):
    """Address a recursive subquestion to one node's external evidence context."""
    return response("query_node", node_id=node_id, question=question)


def finish(answer, citations=(), unresolved=()):
    """Encode a final answer with citations and unresolved evidence gaps."""
    return response("finish", answer=answer, citations=list(citations), unresolved=list(unresolved))


def model(*steps):
    """Queue model decisions without contacting a provider."""
    return ScriptedModelClient(steps)


def run(coro):
    """Execute one async scenario in an independent event loop."""
    return asyncio.run(coro)


async def setup(path):
    """Persist a relay chain, a separate date source and an unread distractor."""
    from llgm.memory.evidence import Evidence

    workspace = Workspace.open(path)
    await workspace.__aenter__()
    for node, text in {
        "a": "Project Cedar uses release r17. Lookup the release in node b.",
        "b": "Release r17 uses registry node c. Its scope is production only.",
        "c": "Registry r17: production region is eu-west-1. Staging is us-east-1.",
        "d": "Cedar rollout date is 2031-04-07.",
        "secret": "Never automatically expose this distractor: hidden corpus sentinel.",
    }.items():
        await workspace.ingest(
            Conversation.from_turns([{"role": "user", "content": text}], node_id=node)
        )
    evidence = await Evidence.open(workspace)
    return workspace, evidence


def make(root, sidecar, evidence, **options):
    """Construct a runtime with controlled budgets and character-count accounting."""
    defaults = dict(
        budget=Budget(
            max_model_calls=30,
            max_sidecar_calls=25,
            max_bundle_tokens=8000,
            max_context_tokens=20000,
        ),
        token_counter=len,
    )
    defaults.update(options)
    return RecursiveRuntime(root, sidecar, evidence, **defaults)


def test_duplicate_operation_fields_fail_before_evidence_access():
    """A repeated operation key cannot replace a requested read with a finish command."""

    async def scenario():
        """Reject an otherwise valid final operation whose duplicate key makes its intent ambiguous."""
        root = model(
            '{"op":"read","op":"finish","answer":"guess","citations":[],"unresolved":["unknown"]}'
        )
        result = await make(root, model(), None).answer("Question")
        assert result.status == "failed"
        assert result.answer == "" and not result.references
        assert result.usage["model_calls"] == 1
        assert not any(event["kind"] == "operation" for event in result.trace)

    run(scenario())


def test_external_context_is_read_only_on_request_and_canonical_citations(tmp_path):
    """Source handles neither preload private context nor weaken citation validation."""

    async def scenario():
        """Read and cite one canonical source while leaving the distractor external."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(delegate("Read production registry", "c"), finish("eu-west-1", ["e1"]))
            sidecar = model(read("c"), finish("Production: eu-west-1; staging: us-east-1", ["e1"]))
            result = await make(root, sidecar, evidence).answer(
                "Production region?", initial_refs=[NodeRef("c")]
            )
            assert result.status == "completed"
            assert result.references == (NodeRef("c"),)
            for client in (root, sidecar):
                initial = " ".join(m.content for m in client.requests[0].messages)
                assert "Registry r17:" not in initial
            prompts = str(root.requests) + str(sidecar.requests)
            assert "hidden corpus sentinel" not in prompts
            assert "Registry r17:" in sidecar.requests[1].messages[-1].content
            assert result.usage["model_calls"] == 4
            assert result.usage["unknown_usage_calls"] == 4
        finally:
            await ws.close()

    run(scenario())


def test_node_queries_follow_graph_and_return_only_selected_span(tmp_path):
    """Nested node queries follow stored links and return only selected exact evidence."""

    async def scenario():
        """Keep a long local read outside ancestors while propagating a small descendant span."""
        import re

        from llgm.core.types import SourceSpan

        ws, evidence = await setup(tmp_path)
        try:
            long_text = "Unrelated local history sentinel. " * 80
            await ws.ingest(
                Conversation.from_turns([{"role": "user", "content": long_text}], node_id="local-a")
            )
            for owner, target in (("local-a", "b"), ("b", "c")):
                await ws.publish_edge(
                    owner,
                    target,
                    relation="depends_on",
                    provenance=Provenance("user", "node-query-test"),
                )
            requests = []

            def query_neighbor(request):
                """Address the next recursive call using a returned graph reference."""
                payload = json.loads(request.messages[-1].content)
                return query_node(payload["references"][0]["node_id"], "Find the production region")

            def read_turn(request):
                """Discover an exact turn address from the host's whole-node read metadata."""
                payload = json.loads(request.messages[-1].content)
                return response("read", reference=payload["metadata"]["turns"][0]["reference"])

            def read_region(request):
                """Select Unicode offsets in returned turn text without fixture-side coordinates."""
                payload = json.loads(request.messages[-1].content)
                match = re.search(r"production region is ([^.]+)", payload["text"])
                reference = dict(payload["references"][0])
                reference.update(start=match.start(1), end=match.end(1))
                return response("read", reference=reference)

            decisions = iter(
                (
                    read("local-a"),
                    response("neighbors", node_id="local-a", relation="depends_on"),
                    query_neighbor,
                    read("b"),
                    response("neighbors", node_id="b", relation="depends_on"),
                    query_neighbor,
                    read("c"),
                    read_turn,
                    read_region,
                    finish("Production region is eu-west-1", ["e5"]),
                    finish("The release uses eu-west-1", ["e5"]),
                    finish("Production region is eu-west-1", ["e5"]),
                )
            )

            async def decide(request):
                """Script the route while deriving all selected-span coordinates from observed evidence."""
                requests.append(request)
                decision = next(decisions)
                return ModelResponse(decision(request) if callable(decision) else decision)

            root = model(
                query_node("local-a", "Find the production region"), finish("eu-west-1", ["e5"])
            )
            sidecar = CallableModelClient(decide)
            result = await make(root, sidecar, evidence).answer("Which region?")
            assert result.status == "completed"
            assert len(result.references) == 1 and isinstance(result.references[0], SourceSpan)
            assert result.references[0].node_id == "c"
            assert (await ws.resolve(result.references[0])).text == "eu-west-1"
            assert long_text in str(requests)
            assert "Unrelated local history sentinel" not in str(root.requests)
            returned = json.loads(root.requests[1].messages[-1].content)
            assert [item["text"] for item in returned["evidence"]] == ["eu-west-1"]
            assert returned["evidence"][0]["metadata"]["role"] == "user"
            initial = [
                json.loads(request.messages[1].content)
                for request in requests
                if len(request.messages) == 2
            ]
            assert [item["target_node_id"] for item in initial] == ["local-a", "b", "c"]
            assert all(
                item["references"] == [reference_to_dict(NodeRef(item["target_node_id"]))]
                for item in initial
            )
            assert "Unrelated local history sentinel" not in json.dumps(initial)
            entries = [event for event in result.trace if event["kind"] == "enter"]
            assert [(event["target_node_id"], event["parent_id"]) for event in entries] == [
                (None, None),
                ("local-a", "q1"),
                ("b", "q2"),
                ("c", "q3"),
            ]
            assert [
                event["target_node_id"] for event in result.trace if event["kind"] == "return"
            ] == ["c", "b", "local-a", None]
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


def test_answer_can_start_at_one_node_without_loading_its_source(tmp_path):
    """A public node target starts the root with a local handle and no preloaded text."""

    async def scenario():
        """Read the target explicitly and retain its identity throughout the trace."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(read("c"), finish("eu-west-1", ["e1"]))
            result = await make(root, model(), evidence).answer("Which region?", node_id="c")
            initial = json.loads(root.requests[0].messages[1].content)
            assert initial["target_node_id"] == "c"
            assert initial["references"] == [reference_to_dict(NodeRef("c"))]
            assert "Registry r17:" not in str(root.requests[0].messages)
            assert "read_basis" not in initial
            assert result.status == "completed"
            assert all(
                event["target_node_id"] == "c"
                for event in result.trace
                if event["kind"] in {"enter", "model", "return"}
            )
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


def test_node_operations_preserve_exact_identifier_bytes(tmp_path):
    """Node targeting, traversal, and journal reads never normalize a valid stored identity."""

    async def scenario():
        """Keep a padded source distinct from a normalized decoy while trimming question text."""
        ws, evidence = await setup(tmp_path)
        try:
            for node_id, text, target in (
                (" padded ", "Exact node sentinel", "c"),
                ("padded", "Normalized decoy sentinel", "d"),
            ):
                await ws.ingest(
                    Conversation.from_turns(
                        [{"role": "user", "content": text}],
                        node_id=node_id,
                    )
                )
                await ws.append_journal(
                    node_id,
                    subject=NodeRef(node_id),
                    relation="depends_on",
                    value=NodeRef(target),
                    provenance=Provenance("user", "identity-test"),
                )
                await ws.publish_edge(
                    node_id,
                    target,
                    relation="depends_on",
                    provenance=Provenance("user", "identity-test"),
                )
            root = model(
                read(" padded "),
                response("neighbors", node_id=" padded ", relation="depends_on"),
                response("journal", node_id=" padded "),
                query_node(" padded ", " Child question "),
                finish("Exact node", ["e1", "e2"]),
            )
            sidecar = model(read(" padded "), finish("Exact node", ["e1"]))
            result = await make(root, sidecar, evidence).answer(" Question ", node_id=" padded ")
            assert result.status == "completed"
            initial = json.loads(root.requests[0].messages[1].content)
            child_initial = json.loads(sidecar.requests[0].messages[1].content)
            assert initial["question"] == "Question"
            assert child_initial["question"] == "Child question"
            for payload in (initial, child_initial):
                assert payload["target_node_id"] == " padded "
                assert payload["references"] == [reference_to_dict(NodeRef(" padded "))]
            neighbors = json.loads(root.requests[2].messages[-1].content)
            assert neighbors["references"] == [reference_to_dict(NodeRef("c"))]
            journal = json.loads(root.requests[3].messages[-1].content)
            assert all(
                entry["references"][0]["node_id"] == " padded " for entry in journal["entries"]
            )
            assert {reference.node_id for reference in result.references} == {" padded "}
            assert "Normalized decoy sentinel" not in str(root.requests) + str(sidecar.requests)
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


@pytest.mark.parametrize("node_id", [True, 3, {}, [], ""])
def test_invalid_node_target_fails_before_model_dispatch(node_id):
    """Malformed node targets fail before spending any model call."""
    from llgm.core.errors import SchemaError

    root = model()
    with pytest.raises(SchemaError, match="node_id"):
        run(make(root, model(), None).answer("Question", node_id=node_id))
    assert not root.requests


def test_node_target_cannot_mix_with_experimental_initial_references():
    """A node target has exactly one initial handle and rejects ambiguous extra context."""
    from llgm.core.errors import SchemaError

    root = model()
    with pytest.raises(SchemaError, match="cannot be combined"):
        run(make(root, model(), None).answer("Question", node_id="a", initial_refs=[NodeRef("b")]))
    assert not root.requests


def test_node_query_repeated_active_request_stops_before_child_dispatch(tmp_path):
    """Node-targeted and generic frames share the active-request cycle guard."""

    async def scenario():
        """Attempt to recursively ask the same question of the current target."""
        ws, evidence = await setup(tmp_path)
        try:
            sidecar = model()
            result = await make(model(query_node("c", "Question")), sidecar, evidence).answer(
                "Question", node_id="c"
            )
            assert result.status == "budget_exhausted"
            assert "Repeated active recursive request" in result.evidence.unresolved[0]
            assert result.usage["model_calls"] == 1
            assert not sidecar.requests
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


@pytest.mark.parametrize(
    "options", [{"max_depth": 0}, {"max_operations": 1}, {"budget": Budget(max_model_calls=1)}]
)
def test_node_queries_obey_shared_limits(tmp_path, options):
    """Addressing a node does not bypass global call, operation, or depth limits."""

    async def scenario():
        """Stop a node query at the boundary without dispatching a sidecar call."""
        ws, evidence = await setup(tmp_path)
        try:
            sidecar = model()
            result = await make(
                model(query_node("c", "Read region")), sidecar, evidence, **options
            ).answer("Question")
            assert result.status == "budget_exhausted"
            assert result.usage["model_calls"] == 1
            assert not sidecar.requests
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


def test_node_query_child_cannot_cite_sibling_evidence(tmp_path):
    """A node query never inherits another node invocation's citation authority."""

    async def scenario():
        """Reject a second node's citation to an ID exposed only to its sibling."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(query_node("c", "First"), query_node("d", "Second"))
            sidecar = model(read("c"), finish("region", ["e1"]), finish("stolen", ["e1"]))
            result = await make(root, sidecar, evidence).answer("Question")
            assert result.status == "failed"
            assert "not exposed" in result.evidence.unresolved[0]
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


def test_node_query_missing_source_returns_explicit_gap(tmp_path):
    """A missing target remains an unresolved node query instead of fabricated evidence."""

    async def scenario():
        """Resolve the target on demand and propagate the child's explicit missing-source gap."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(
                query_node("missing", "Find its region"),
                finish("Cannot determine", unresolved=["Target source unavailable"]),
            )
            sidecar = model(
                read("missing"),
                finish("Cannot determine", unresolved=["Target source unavailable"]),
            )
            result = await make(root, sidecar, evidence).answer("Question")
            assert result.status == "partial"
            assert not result.references
            assert json.loads(sidecar.requests[1].messages[-1].content)["error"] == "unavailable"
            assert json.loads(root.requests[1].messages[-1].content)["unresolved"] == [
                "Target source unavailable"
            ]
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


def test_recursive_relay_returns_along_actual_parent_chain(tmp_path):
    """Recursive findings return through the actual parent chain in reverse entry order."""

    async def scenario():
        """Follow a three-level release lookup and propagate the same query date to every child."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(
                delegate("Resolve Cedar", "a"), finish("Production eu-west-1", ["e1", "e2", "e3"])
            )
            sidecar = model(
                read("a"),
                delegate("Resolve release r17", "b"),
                read("b"),
                delegate("Read production registry", "c"),
                read("c"),
                finish("Production eu-west-1", ["e3"]),
                finish("r17 production eu-west-1", ["e2", "e3"]),
                finish("Cedar production eu-west-1", ["e1", "e2", "e3"]),
            )
            result = await make(root, sidecar, evidence).answer(
                "Where?", query_date="2031-05-01", query_scope={"environment": "production"}
            )
            enters = [e for e in result.trace if e["kind"] == "enter"]
            assert [(e["invocation_id"], e["parent_id"], e["depth"]) for e in enters] == [
                ("q1", None, 0),
                ("q2", "q1", 1),
                ("q3", "q2", 2),
                ("q4", "q3", 3),
            ]
            returns = [e["invocation_id"] for e in result.trace if e["kind"] == "return"]
            assert returns == ["q4", "q3", "q2", "q1"]
            assert result.status == "completed"
            assert {r.node_id for r in result.references} == {"a", "b", "c"}
            for request in root.requests + sidecar.requests:
                assert json.loads(request.messages[1].content)["query_date"] == "2031-05-01"
                assert json.loads(request.messages[1].content)["query_scope"] == {
                    "environment": "production"
                }
        finally:
            await ws.close()

    run(scenario())


def test_root_continues_and_combines_siblings_without_inheriting_context(tmp_path):
    """The root can combine siblings without exposing one child's context to another."""

    async def scenario():
        """Gather region and rollout date in separate child invocations."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(
                delegate("Get region", "c"),
                delegate("Get date", "d"),
                finish("eu-west-1 on 2031-04-07", ["e1", "e2"]),
            )
            sidecar = model(
                read("c"), finish("eu-west-1", ["e1"]), read("d"), finish("2031-04-07", ["e2"])
            )
            result = await make(root, sidecar, evidence).answer("Where and when?")
            assert result.status == "completed"
            assert len(root.requests) == 3
            second_child = sidecar.requests[2]
            assert "Registry r17:" not in str(second_child.messages)
            assert "eu-west-1" not in str(second_child.messages)
            assert [e["parent_id"] for e in result.trace if e["kind"] == "enter"][1:] == [
                "q1",
                "q1",
            ]
        finally:
            await ws.close()

    run(scenario())


def test_child_cannot_cite_evidence_seen_only_by_its_sibling(tmp_path):
    """A sibling cannot cite evidence that only another child has observed."""

    async def scenario():
        """Make the second child return an evidence ID it never received."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(delegate("First", "c"), delegate("Second", "d"))
            sidecar = model(read("c"), finish("region", ["e1"]), finish("stolen evidence", ["e1"]))
            result = await make(root, sidecar, evidence).answer("Question")
            assert result.status == "failed"
            assert "not exposed" in result.evidence.unresolved[0]
        finally:
            await ws.close()

    run(scenario())


@pytest.mark.parametrize(
    "finish_response",
    [
        finish("invented", ["e999"]),
        finish("invented"),
        response("finish", answer="a", citations="e1", unresolved=[]),
        response("finish", answer="a", citations=[], unresolved="unknown"),
        '{"op":"delete","node_id":"c"}',
        "[]",
        "not JSON",
        '{"op":"search","query":"Cedar","k":true}',
        '{"op":"read","reference":null}',
        '{"op":"query_node","node_id":true,"question":"Find it"}',
        '{"op":"query_node","node_id":"c","question":""}',
        '{"op":"query_node","node_id":"c","question":"Find it","references":[]}',
    ],
)
def test_invalid_operation_and_unsupported_citations_fail_without_repair(tmp_path, finish_response):
    """Malformed decisions and unsupported citations fail without a repair call."""

    async def scenario():
        """Submit one invalid operation and verify the runtime returns no invented answer."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(finish_response)
            result = await make(root, model(), evidence).answer("Question")
            assert result.status == "failed"
            assert len(root.requests) == 1
            assert result.answer == ""
            assert not any(e["kind"] == "return" for e in result.trace)
        finally:
            await ws.close()

    run(scenario())


def test_missing_evidence_can_be_reported_without_invented_answer(tmp_path):
    """Missing sources can produce an explicit unresolved result without fabricated evidence."""

    async def scenario():
        """Read an absent node before reporting the remaining information gap."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(
                read("missing"), finish("Cannot determine", unresolved=["Source unavailable"])
            )
            result = await make(root, model(), evidence).answer("Question")
            assert result.status == "partial"
            assert result.references == ()
            assert '"error": "unavailable"' in root.requests[1].messages[-1].content
        finally:
            await ws.close()

    run(scenario())


def test_shared_source_reads_deduplicate_exposure_but_not_model_cost(tmp_path):
    """Repeated evidence exposure deduplicates text while every model call remains charged."""

    async def scenario():
        """Read the same source in two children and compare exposure with a single read."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(delegate("First", "c"), delegate("Second", "c"), finish("region", ["e1"]))
            sidecar = model(
                read("c"), finish("region", ["e1"]), read("c"), finish("region", ["e1"])
            )
            result = await make(root, sidecar, evidence).answer("Question")
            assert result.status == "completed"
            single = await make(
                model(read("c"), finish("region", ["e1"])), model(), evidence
            ).answer("Question")
            assert (
                result.usage["evidence_accounting_units"]
                == single.usage["evidence_accounting_units"]
            )
            assert result.usage["model_calls"] == 7
        finally:
            await ws.close()

    run(scenario())


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"max_depth": 0}, "depth"),
        ({"max_steps": 1}, "step"),
        ({"max_operations": 1}, "Operation"),
    ],
)
def test_recursive_limits_terminate_explicitly(tmp_path, options, reason):
    """Depth, per-invocation steps and total operations stop recursion explicitly."""

    async def scenario():
        """Attempt one delegation under each independently tightened execution limit."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(delegate("child", "c"))
            sidecar = model(read("c"), finish("region", ["e1"]))
            result = await make(root, sidecar, evidence, **options).answer("Question")
            assert result.status == "budget_exhausted"
            assert reason.lower() in result.evidence.unresolved[0].lower()
        finally:
            await ws.close()

    run(scenario())


def test_active_request_cycle_stops_without_infinite_recursion(tmp_path):
    """A repeated active request terminates before recursive self-invocation can continue."""

    async def scenario():
        """Delegate the same question and handles already present on the active ancestry."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(delegate("Question", "c"))
            result = await make(root, model(), evidence).answer(
                "Question", initial_refs=[NodeRef("c")]
            )
            assert result.status == "budget_exhausted"
            assert "Repeated active" in result.evidence.unresolved[0]
        finally:
            await ws.close()

    run(scenario())


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("max_model_calls", 2, "Model-call"),
        ("max_sidecar_calls", 1, "Sidecar-call"),
        ("max_evidence_tokens", 1, "Evidence exposure"),
        ("max_bundle_tokens", 1, "bundle"),
        ("max_context_tokens", 1, "context"),
    ],
)
def test_shared_budget_boundaries(tmp_path, field, value, reason):
    """Model, sidecar, evidence, bundle and context allowances share one run ledger."""

    async def scenario():
        """Tighten one resource bound around a child read and inspect the stopping reason."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(delegate("child", "c"), finish("region", ["e1"]))
            sidecar = model(read("c"), finish("region", ["e1"]))
            budget = replace(Budget(), **{field: value})
            result = await make(root, sidecar, evidence, budget=budget).answer("Question")
            assert result.status == "budget_exhausted"
            assert reason.lower() in result.evidence.unresolved[0].lower()
            if field == "max_context_tokens":
                assert result.usage["model_calls"] == 0
            if field == "max_evidence_tokens":
                assert result.usage["evidence_accounting_units"] == 0
                assert "Registry r17:" not in str(sidecar.requests)
        finally:
            await ws.close()

    run(scenario())


def test_timeout_in_descendant_counts_attempt_and_stops_execution(tmp_path):
    """A descendant timeout cancels its provider work and retains the attempted call."""

    async def scenario():
        """Let the child exceed a short deadline after root delegation."""
        ws, evidence = await setup(tmp_path)
        try:
            stopped = asyncio.Event()

            async def slow(request):
                """Remain pending until deadline cancellation and signal that cleanup ran."""
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

            root = model(delegate("child", "c"))
            runtime = make(
                root, CallableModelClient(slow), evidence, budget=Budget(timeout_seconds=0.05)
            )
            result = await runtime.answer("Question")
            assert result.status == "budget_exhausted"
            assert stopped.is_set()
            assert result.usage["model_calls"] == 2
            assert result.trace[-2]["status"] == "timeout"
        finally:
            await ws.close()

    run(scenario())


def test_cancellation_propagates_to_child_and_runtime_can_be_reused(tmp_path):
    """Cancellation reaches descendants and leaves the runtime reusable after the run ends."""

    async def scenario():
        """Cancel a running child, reject simultaneous reuse, then start a fresh question."""
        ws, evidence = await setup(tmp_path)
        try:
            entered, stopped = asyncio.Event(), asyncio.Event()

            async def slow(request):
                """Signal child dispatch and record cancellation cleanup while remaining pending."""
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

            runtime = make(model(delegate("child", "c")), CallableModelClient(slow), evidence)
            task = asyncio.create_task(runtime.answer("Question"))
            await entered.wait()
            with pytest.raises(ConfigurationError):
                await runtime.answer("Concurrent question")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stopped.is_set()
            assert runtime.last_usage["model_calls"] == 2
            assert any(e.get("status") == "cancelled" for e in runtime.last_trace)
            runtime.root_model = model(finish("Unknown", unresolved=["No source"]))
            assert (await runtime.answer("New question")).status == "partial"
        finally:
            await ws.close()

    run(scenario())


def test_provider_failure_and_known_cached_usage_survive_recursion(tmp_path):
    """Recursive failures retain known usage and provider-specific cached-input categories."""

    async def scenario():
        """Return a billed incomplete child response after a billed root delegation."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(
                ModelResponse(delegate("child", "c"), Usage(10, 5, {"cached_input_tokens": 4}))
            )
            sidecar = model(ModelResponse("", Usage(7, 2), provider="test", status="incomplete"))
            result = await make(root, sidecar, evidence).answer("Question")
            assert result.status == "failed"
            assert result.usage["known_input_tokens"] == 17
            assert result.usage["known_output_tokens"] == 7
            events = [e for e in result.trace if e["kind"] == "model"]
            assert events[0]["usage_extra"]["cached_input_tokens"] == 4
            assert events[1]["status"] == "incomplete"
        finally:
            await ws.close()

    run(scenario())


def test_distinct_correction_node_preserves_original_evidence(tmp_path):
    """Publishing a correction under a new identity leaves original references stable."""

    async def scenario():
        """Publish a distinct correction, then resolve the original node handle."""
        ws, evidence = await setup(tmp_path)
        try:
            await ws.ingest(
                Conversation.from_turns(
                    [{"role": "user", "content": "Production changed to ap-south-1"}],
                    node_id="correction",
                )
            )
            root = model(
                response("read", reference=reference_to_dict(NodeRef("c"))),
                finish("eu-west-1", ["e1"]),
            )
            result = await make(root, model(), evidence).answer("Question")
            assert result.status == "completed"
            assert result.references == (NodeRef("c"),)
            assert "ap-south-1" not in str(root.requests)
        finally:
            await ws.close()

    run(scenario())


def test_journal_inline_and_pointer_evidence_can_be_read_and_cited(tmp_path):
    """Inline notes and pointer records remain distinct citable journal evidence."""

    async def scenario():
        """Read a suggestion and a structural link from one node's journal."""
        from llgm.memory.evidence import Evidence

        ws, _ = await setup(tmp_path)
        try:
            entry = await ws.append_journal(
                "a",
                subject=NodeRef("a"),
                relation="note",
                value="This is a suggestion, not an adopted production change.",
                provenance=Provenance("user", "test"),
            )
            link = await ws.append_journal(
                "a",
                subject=NodeRef("a"),
                relation="depends_on",
                value=NodeRef("b"),
                provenance=Provenance("user", "test"),
            )
            evidence = await Evidence.open(ws)
            root = model(response("journal", node_id="a"), finish("Suggestion only", ["e1", "e2"]))
            result = await make(root, model(), evidence).answer("Is it adopted?")
            assert result.status == "completed"
            assert set(result.references) == {
                JournalRef("a", entry.entry_id),
                JournalRef("a", link.entry_id),
            }
            assert "suggestion" in result.evidence.text
        finally:
            await ws.close()

    run(scenario())


def test_graph_neighbors_are_handles_until_the_model_reads_them(tmp_path):
    """Graph discovery returns handles without exposing source text until an explicit read."""

    async def scenario():
        """Discover a typed neighbor before reading and citing its source content."""
        from llgm.memory.evidence import Evidence

        ws, _ = await setup(tmp_path)
        try:
            await ws.publish_edge(
                "a", "c", relation="depends_on", provenance=Provenance("user", "test")
            )
            evidence = await Evidence.open(ws)
            root = model(
                response("neighbors", node_id="a", relation="depends_on"),
                read("c"),
                finish("production eu-west-1", ["e1"]),
            )
            result = await make(root, model(), evidence).answer("Question")
            assert result.status == "completed"
            assert json.loads(root.requests[1].messages[-1].content)["references"] == [
                reference_to_dict(NodeRef("c"))
            ]
            assert "Registry r17:" not in root.requests[1].messages[-1].content
            assert "Registry r17:" in root.requests[2].messages[-1].content
        finally:
            await ws.close()

    run(scenario())


def test_search_uses_real_index_and_passage_references(tmp_path):
    """Search results from the actual evidence index carry resolvable passage spans."""

    async def scenario():
        """Retrieve the rollout fact and validate its returned source coordinates."""
        from llgm.core.types import SourceSpan

        ws, evidence = await setup(tmp_path)
        try:
            root = model(response("search", query="rollout", k=1), finish("2031-04-07", ["e1"]))
            result = await make(root, model(), evidence).answer("When is rollout?")
            assert result.status == "completed"
            assert result.usage["searches"] == 1
            assert len(result.references) == 1
            assert isinstance(result.references[0], SourceSpan)
            assert result.references[0].node_id == "d"
            assert (await ws.resolve(result.references[0])).text in result.evidence.text
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


def test_search_budget_is_global_across_sibling_invocations(tmp_path):
    """Sibling searches consume the same global search allowance."""

    async def scenario():
        """Exhaust the one-search allowance in the first child before the second searches."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(delegate("First"), delegate("Second"))
            sidecar = model(
                response("search", query="rollout", k=1),
                finish("date", ["e1"]),
                response("search", query="registry", k=1),
            )
            result = await make(root, sidecar, evidence, budget=Budget(max_searches=1)).answer(
                "Question"
            )
            assert result.status == "budget_exhausted"
            assert result.usage["searches"] == 1
            assert "Search allowance" in result.evidence.unresolved[0]
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


def test_child_budget_leaves_one_call_for_root_final_answer(tmp_path):
    """A completed child can leave exactly one remaining call for the root's final answer."""

    async def scenario():
        """Fit delegation, child read/return and root completion into four calls."""
        ws, evidence = await setup(tmp_path)
        try:
            root = model(delegate("child", "c"), finish("region", ["e1"]))
            sidecar = model(read("c"), finish("region", ["e1"]))
            result = await make(root, sidecar, evidence, budget=Budget(max_model_calls=4)).answer(
                "Question"
            )
            assert result.status == "completed"
            assert result.usage["model_calls"] == 4
            assert result.usage["sidecar_calls"] == 2
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


def test_read_timeout_cancels_tool_and_retains_model_attempt(tmp_path):
    """A timed-out evidence operation is cancelled without losing its preceding model call."""

    async def scenario():
        """Suspend a source read after the root has requested it."""
        ws, evidence = await setup(tmp_path)
        try:
            stopped = asyncio.Event()

            class SlowEvidence:
                """Evidence double whose read stays pending until the run deadline cancels it."""

                async def read(self, reference):
                    """Record cancellation of a deliberately nonreturning source read."""
                    try:
                        await asyncio.Event().wait()
                    finally:
                        stopped.set()

            root = model(read("c"))
            result = await make(
                root, model(), SlowEvidence(), budget=Budget(timeout_seconds=0.05)
            ).answer("Question")
            assert result.status == "budget_exhausted"
            assert stopped.is_set()
            assert result.usage["model_calls"] == 1
        finally:
            await evidence.close()
            await ws.close()

    run(scenario())


@pytest.mark.parametrize("operation", ["read", "search"])
def test_dates_and_speaker_metadata_survive_child_evidence_return(tmp_path, operation):
    """Dates and speaker attribution survive evidence transport through recursive returns."""

    async def scenario():
        """Read or search a relative-date source in the child and inspect both continuation prompts."""
        from llgm.core.types import SourceSpan
        from llgm.memory.evidence import Evidence

        async with Workspace.open(tmp_path) as ws:
            await ws.ingest(
                Conversation.from_turns(
                    [{"role": "user", "content": "Deployment is tomorrow."}],
                    node_id="dated",
                    metadata={"date": "2031-04-06"},
                )
            )
            async with await Evidence.open(ws) as evidence:
                op = (
                    response(
                        "read",
                        reference=reference_to_dict(
                            SourceSpan("dated", "turn-000000", 0, len("Deployment is tomorrow."))
                        ),
                    )
                    if operation == "read"
                    else response("search", query="Deployment", k=1)
                )
                root = model(
                    delegate("Resolve relative date", "dated"), finish("2031-04-07", ["e1"])
                )
                sidecar = model(op, finish("2031-04-07", ["e1"]))
                result = await make(root, sidecar, evidence).answer("When?")
                assert result.status == "completed"
                for request in (sidecar.requests[1], root.requests[1]):
                    payload = request.messages[-1].content
                    assert "2031-04-06" in payload
                    assert '"role": "user"' in payload
                assert result.usage["evidence_accounting_units"] > len("Deployment is tomorrow.")

    run(scenario())


@pytest.mark.parametrize("bad_date", [True, 17, {}, [], ""])
def test_invalid_query_date_is_rejected_before_any_model_call(bad_date):
    """Malformed query dates fail before any root request is dispatched."""
    from llgm.core.errors import SchemaError

    root = model()
    runtime = make(root, model(), None)
    with pytest.raises(SchemaError, match="query_date"):
        run(runtime.answer("Question", query_date=bad_date))
    assert not root.requests


def test_expired_tool_deadline_does_not_start_or_leak_a_coroutine():
    """An already expired deadline rejects a tool before creating its coroutine."""
    from llgm.core.errors import BudgetExceeded
    from llgm.inference.recursive import _Execution

    async def scenario():
        """Age the execution ledger and attempt to dispatch a tracked tool factory."""
        execution = _Execution(make(model(), model(), None), None)
        execution.ledger.started -= 1000
        called = False

        async def operation():
            """Expose whether an expired execution accidentally started the evidence operation."""
            nonlocal called
            called = True

        with pytest.raises(BudgetExceeded):
            await execution.tool(operation)
        assert not called

    run(scenario())


@pytest.mark.parametrize(
    "scope", [[], "production", {1: "value"}, {"v": float("nan")}, {"v": object()}]
)
def test_invalid_query_scope_is_rejected_before_model_calls(scope):
    """Invalid scope metadata fails before evidence access or provider dispatch."""
    from llgm.core.errors import SchemaError

    root = model()
    runtime = make(root, model(), None)
    with pytest.raises(SchemaError, match="query_scope"):
        run(runtime.answer("Question", query_scope=scope))
    assert root.requests == []


@pytest.mark.parametrize("capture", [False, True])
def test_multiple_operations_fail_with_auditable_output(capture):
    """Concatenated operations fail without executing either and retain optional diagnostic text."""
    raw = finish("No evidence", unresolved=["missing"]) + finish(
        "Second answer", unresolved=["missing"]
    )
    runtime = RecursiveRuntime(model(raw), model(), None, capture_text=capture)
    result = run(runtime.answer("Question"))
    assert result.status == "failed"
    assert result.usage["model_calls"] == 1
    assert not any(event["kind"] == "operation" for event in result.trace)
    event = next(event for event in result.trace if event["kind"] == "model_output")
    assert len(event["output_sha256"]) == 64
    assert ("text" in event) is capture
    if capture:
        assert event["text"] == raw
    assert "JSON object" in result.evidence.unresolved[0]
