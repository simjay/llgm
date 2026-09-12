"""Application contracts over real persisted sources, primary edges, and journals.

Model fixtures control operation choices and proposal errors; they do not
measure hosted-model reasoning or semantic link accuracy.
"""

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from llgm import parse_instant_ms
from llgm.core.config import Settings
from llgm.core.errors import ConfigurationError, ProviderError
from llgm.core.types import Conversation, JournalRef, NodeRef, Provenance, SourceSpan
from llgm.inference.budget import Budget
from llgm.llgm import LLGM, MaintenancePolicy
from llgm.memory.evidence import Evidence, LinkProposal
from llgm.memory.query import QueryEvidence
from llgm.memory.workspace import Workspace
from llgm.models import CallableModelClient, ModelResponse, ScriptedModelClient, Usage
from llgm.models.hosted import OpenAIModelClient
from llgm.retrieval.base import SearchPassage
from llgm.retrieval.bm25 import SQLiteBM25Retriever
from tests.node_support import READ, SEARCH, Models, ReplayFactory
from tests.test_models import FakeSDK


def conversation(node_id, text):
    """Create exact deterministic source text without evaluator annotations."""
    return Conversation.from_turns(
        [{"role": "user", "turn_id": "t", "text": text}], node_id=node_id
    )


def operation(name, **fields):
    """Encode one structured runtime operation for contract fixtures."""
    return json.dumps({"op": name, **fields})


def proposer(*, relation="related_to", usage=Usage(20, 10), invalid=False):
    """Propose links only from the exact endpoint spans presented by maintenance."""

    async def complete(request):
        """Build structurally valid fixture proposals without access to Workspace internals."""
        if invalid:
            return ModelResponse(
                '{"links":"invalid"}', usage, provider="fixture", model="maintenance"
            )
        payload = json.loads(request.messages[-1].content)
        source = payload["source_node_id"]
        passages = payload["passages"]
        source_passage = next(item for item in passages if item["reference"]["node_id"] == source)
        links = []
        for candidate in payload["candidate_node_ids"]:
            target = next(item for item in passages if item["reference"]["node_id"] == candidate)
            if "Orion" not in source_passage["text"] or "Orion" not in target["text"]:
                continue
            links.append(
                {
                    "target_node_id": candidate,
                    "relation": relation,
                    "supporting_references": [source_passage["reference"], target["reference"]],
                    "rationale": "Both fixture passages describe Orion.",
                }
            )
        return ModelResponse(
            json.dumps({"links": links}), usage, provider="fixture", model="maintenance"
        )

    return CallableModelClient(complete, provider="fixture", model="maintenance")


class ApplicationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise primary-edge publication and retrieval-first node inference boundaries."""

    async def asyncSetUp(self):
        """Open real local storage for every independent application contract."""
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.workspace = await Workspace.open(self.path).__aenter__()

    async def asyncTearDown(self):
        """Release the caller-owned workspace and temporary immutable blobs."""
        await self.workspace.close()
        self.temp.cleanup()

    def app(self, *, root=None, sidecar=None, maintenance=None, policy=None, **kwargs):
        """Construct an application with deterministic clients and generous local accounting limits."""
        models = Models()
        kwargs.setdefault("repl_factory", ReplayFactory())
        return LLGM(
            self.workspace,
            root or models.root,
            sidecar or models.sidecar,
            maintenance_model=maintenance or proposer(),
            maintenance_policy=policy,
            inference_budget=Budget(
                max_model_calls=12,
                max_sidecar_calls=10,
                max_context_tokens=64000,
                max_bundle_tokens=16000,
            ),
            **kwargs,
        )

    async def seed(self, *sources):
        """Publish input text through the application with explicitly deferred maintenance."""
        app = self.app()
        for node_id, text in sources:
            await app.ingest(conversation(node_id, text), organize=False)

    async def test_ingestion_automatically_discovers_and_publishes_supported_links(self):
        """A newly ingested source links to a previously disconnected matching node."""
        app = self.app()
        first = await app.ingest(conversation("a", "Orion registry region is eu-west-1."))
        self.assertEqual(first.maintenance.usage["model_calls"], 0)
        await app.ingest(
            conversation("distractor", "Harbor fiction describes an unrelated blue ship.")
        )
        second = await app.ingest(
            conversation("b", "Orion uses the registry for production deployments.")
        )
        self.assertEqual(second.maintenance.status, "completed")
        self.assertEqual(second.maintenance.usage["maintenance_calls"], 1)
        self.assertEqual(second.maintenance.usage["known_input_tokens"], 20)
        self.assertEqual(second.maintenance.usage["known_output_tokens"], 10)
        self.assertIsNone(second.maintenance.usage["currency_cost"])
        self.assertEqual([entry.target_node_id for entry in second.maintenance.accepted], ["a"])
        entry = second.maintenance.accepted[0]
        self.assertEqual(entry.provenance.model, "maintenance")
        self.assertEqual(
            {ref.node_id for ref in entry.provenance.supporting_references}, {"a", "b"}
        )
        self.assertEqual(await self.workspace.edges("b"), [entry])

    async def test_retry_does_not_duplicate_links_or_source_nodes(self):
        """An ingestion retry can repeat proposal work but cannot duplicate accepted evidence."""
        app = self.app()
        await app.ingest(conversation("a", "Orion region is eu-west-1."))
        first = await app.ingest(
            conversation("b", "Orion uses the same region."), idempotency_key="b"
        )
        again = await app.ingest(
            conversation("b", "Orion uses the same region."), idempotency_key="b"
        )
        self.assertFalse(again.source.created)
        self.assertEqual(first.source.node_id, again.source.node_id)
        self.assertEqual(len(await self.workspace.edges("b")), 1)
        self.assertEqual(again.maintenance.accepted, ())
        self.assertEqual(again.maintenance.decisions[0]["decision"], "already_recorded")
        self.assertEqual(again.maintenance.usage["model_calls"], 1)

    async def test_maintenance_preserves_late_retrieval_matches_and_attribution(self):
        """Automatic publication retains late supporting spans, source roles, dates, and their cost."""
        await self.workspace.ingest(
            Conversation.from_turns(
                [
                    {
                        "role": "assistant",
                        "turn_id": "intro",
                        "text": "Unrelated background. " * 250,
                    },
                    {
                        "role": "user",
                        "turn_id": "decision",
                        "text": "Orion cobalt production runs in eu-west-1.",
                    },
                ],
                node_id="registry",
                metadata={"date": "2026-09-09"},
            )
        )
        requests = []

        async def link_from_presented_evidence(request):
            """Publish only when the model request contains the actual matched decision."""
            requests.append(request)
            payload = json.loads(request.messages[-1].content)
            source = next(
                item for item in payload["passages"] if item["reference"]["node_id"] == "release"
            )
            matched = next(
                item
                for item in payload["passages"]
                if item["reference"]["node_id"] == "registry" and "eu-west-1" in item["text"]
            )
            self.assertEqual(matched["reference"]["turn_id"], "decision")
            self.assertEqual(matched["metadata"]["role"], "user")
            self.assertEqual(matched["metadata"]["source_metadata"]["date"], "2026-09-09")
            return ModelResponse(
                json.dumps(
                    {
                        "links": [
                            {
                                "target_node_id": "registry",
                                "relation": "depends_on",
                                "supporting_references": [
                                    source["reference"],
                                    matched["reference"],
                                ],
                                "rationale": "The registry identifies the production region.",
                            }
                        ]
                    }
                ),
                Usage(30, 15),
                provider="fixture",
                model="maintenance",
            )

        app = self.app(
            maintenance=CallableModelClient(link_from_presented_evidence),
            policy=MaintenancePolicy(max_context_chars=1024),
            passage_chars=256,
        )
        result = await app.ingest(
            conversation("release", "Orion cobalt production uses the registry.")
        )
        self.assertEqual(result.maintenance.status, "completed")
        self.assertEqual(result.maintenance.usage["searches"], 1)
        self.assertEqual(result.maintenance.usage["model_calls"], 1)
        self.assertEqual(len(result.maintenance.accepted), 1)
        support = result.maintenance.accepted[0].provenance.supporting_references
        target = next(ref for ref in support if ref.node_id == "registry")
        self.assertIn("eu-west-1", (await self.workspace.resolve(target)).text)
        payload = json.loads(requests[0].messages[-1].content)
        presented = {
            json.dumps(item, ensure_ascii=False, sort_keys=True) for item in payload["passages"]
        }
        self.assertEqual(
            result.maintenance.usage["evidence_accounting_units"],
            sum(len(item.encode("utf-8")) for item in presented),
        )

    async def test_maintenance_tracks_same_endpoints_in_distinct_scopes(self):
        """Link retries deduplicate within a scope without colliding with another scope."""
        await self.seed(("a", "Orion origin"), ("b", "Orion target"))
        for environment in ("production", "staging"):
            app = self.app()
            proposal = LinkProposal(
                NodeRef("a"),
                NodeRef("b"),
                "related_to",
                (SourceSpan("a", "t", 0, 12), SourceSpan("b", "t", 0, 12)),
                "Shared Orion context",
                "fixture",
                applicability={"scope": {"env": environment}},
            )
            with patch("llgm.llgm.propose_links", return_value=[proposal]):
                first = await app.organize(["a"])
                retry = await app.organize(["a"])
            self.assertEqual(first.status, "completed")
            self.assertEqual(len(first.accepted), 1)
            self.assertEqual(retry.accepted, ())
            self.assertEqual(retry.decisions[0]["decision"], "already_recorded")
        self.assertEqual(
            {edge.applicability["scope"]["env"] for edge in await self.workspace.edges("a")},
            {"production", "staging"},
        )

    async def test_proposal_and_relation_policies_do_not_publish_rejected_links(self):
        """Review-only and relation-rejection decisions retain proposals without graph mutation."""
        await self.seed(("a", "Orion origin"), ("b", "Orion target"))
        review = await self.app(policy=MaintenancePolicy(mode="propose")).organize(["a"])
        self.assertEqual(review.decisions[0]["decision"], "review_required")
        self.assertEqual(len(review.proposals), 1)
        self.assertFalse(review.accepted)
        rejected = await self.app(maintenance=proposer(relation="suggests")).organize(["a"])
        self.assertEqual(rejected.decisions[0]["decision"], "relation_not_allowed")
        self.assertEqual(await self.workspace.inspect_journal("a"), [])

    async def test_maintenance_model_receives_the_configured_relation_vocabulary(self):
        """Automatic proposals are asked to use the same relation vocabulary publication enforces."""
        await self.seed(("a", "Orion origin"), ("b", "Orion target"))
        requests = []
        underlying = proposer(relation="depends_on")

        async def observing(request):
            """Retain the actual proposal request before producing a structurally valid fixture response."""
            requests.append(request)
            return await underlying.complete(request)

        policy = MaintenancePolicy(allowed_relations=("depends_on",))
        result = await self.app(maintenance=CallableModelClient(observing), policy=policy).organize(
            ["a"]
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual([entry.relation for entry in result.accepted], ["depends_on"])
        self.assertIn(json.dumps(policy.allowed_relations), requests[0].messages[0].content)
        self.assertEqual(json.loads(requests[0].messages[-1].content)["source_node_id"], "a")

    async def test_maintenance_native_schema_reaches_sdk_and_host_still_checks_references(self):
        """Native structure reaches the provider while wrapped passages still fail host validation."""
        await self.seed(("a", "Orion origin"), ("b", "Orion target"))
        support = [
            {"type": "source_span", "node_id": node_id, "turn_id": "t", "start": 0, "end": 12}
            for node_id in ("a", "b")
        ]
        policy = MaintenancePolicy(allowed_relations=("depends_on",))
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                payload = {
                    "links": [
                        {
                            "target_node_id": "b",
                            "relation": "depends_on",
                            "supporting_references": [
                                {"reference": ref, "text": "Orion passage", "metadata": {}}
                                if wrapped
                                else ref
                                for ref in support
                            ],
                            "rationale": "Both discuss Orion.",
                        }
                    ]
                }
                sdk = FakeSDK(
                    NS(
                        id="maintenance-response",
                        model="maintenance-model",
                        status="completed",
                        usage=NS(input_tokens=20, output_tokens=10),
                        output=[
                            NS(
                                type="message",
                                content=[NS(type="output_text", text=json.dumps(payload))],
                            )
                        ],
                    )
                )
                async with OpenAIModelClient("maintenance-model", client=sdk) as model:
                    result = await self.app(maintenance=model, policy=policy).organize(["a"])
                self.assertEqual(len(sdk.requests), 1)
                request = sdk.requests[0]
                native = request["text"]["format"]
                self.assertTrue(native["strict"])
                link = native["schema"]["properties"]["links"]["items"]
                self.assertEqual(
                    set(link["required"]),
                    {"target_node_id", "relation", "supporting_references", "rationale"},
                )
                refs = link["properties"]["supporting_references"]["items"]["anyOf"]
                self.assertEqual(
                    {ref["properties"]["type"]["enum"][0] for ref in refs},
                    {"source_span", "journal"},
                )
                for ref in refs:
                    self.assertEqual(set(ref["required"]), set(ref["properties"]))
                    self.assertFalse(ref["additionalProperties"])
                    self.assertNotIn("text", ref["properties"])
                    self.assertNotIn("reference", ref["properties"])
                self.assertIn("copy only the value", request["input"][0]["content"])
                self.assertIn(json.dumps(policy.allowed_relations), request["input"][0]["content"])
                self.assertEqual(request["max_output_tokens"], policy.budget.max_output_tokens)
                self.assertTrue(result.trace[0]["structured_output"])
                self.assertEqual(result.usage["model_calls"], 1)
                self.assertEqual(result.status, "failed" if wrapped else "completed")
                if wrapped:
                    self.assertEqual(result.error_type, "SchemaError")
                    self.assertEqual(result.accepted, ())
                else:
                    self.assertEqual(len(result.accepted), 1)
                    self.assertEqual(
                        result.accepted[0].provenance.prompt_version, "link-proposal-v4"
                    )

    async def test_node_link_and_model_call_limits_preserve_partial_work(self):
        """Explicit bounds defer nodes, limit publications, and retain accepted work at exhaustion."""
        await self.seed(("a", "Orion alpha"), ("b", "Orion beta"), ("c", "Orion gamma"))
        limited = await self.app(
            policy=MaintenancePolicy(max_nodes=1, max_links_per_node=1)
        ).organize()
        self.assertEqual(limited.deferred_node_ids, ("b", "c"))
        self.assertEqual(len(limited.accepted), 1)
        self.assertIn("link_limit", [row["decision"] for row in limited.decisions])
        policy = MaintenancePolicy(budget=replace(MaintenancePolicy().budget, max_model_calls=1))
        exhausted = await self.app(policy=policy).organize(["b", "c"])
        self.assertEqual(exhausted.status, "budget_exhausted")
        self.assertEqual(exhausted.usage["model_calls"], 1)
        self.assertTrue(exhausted.accepted)
        self.assertEqual(exhausted.error_type, "BudgetExceeded")

    async def test_invalid_proposal_keeps_ingested_source_and_usage(self):
        """Malformed model output fails maintenance without rolling back durable ingestion."""
        await self.seed(("a", "Orion origin"))
        app = self.app(maintenance=proposer(invalid=True))
        result = await app.ingest(conversation("b", "Orion target"))
        self.assertTrue(result.source.created)
        self.assertEqual(result.maintenance.status, "failed")
        self.assertEqual(result.maintenance.error_type, "SchemaError")
        self.assertEqual(result.maintenance.usage["model_calls"], 1)
        self.assertIn("Orion target", (await self.workspace.resolve(NodeRef("b"))).text)
        self.assertEqual(await self.workspace.inspect_journal("b"), [])

    async def test_maintenance_evidence_budget_prevents_provider_admission(self):
        """Source evidence cannot reach maintenance models after its declared allowance is exhausted."""
        await self.seed(("a", "Orion origin"))
        policy = MaintenancePolicy(
            budget=replace(MaintenancePolicy().budget, max_evidence_tokens=1)
        )
        result = await self.app(policy=policy).ingest(conversation("b", "Orion target"))
        self.assertEqual(result.maintenance.status, "budget_exhausted")
        self.assertEqual(result.maintenance.usage["model_calls"], 0)
        self.assertEqual(result.maintenance.usage["evidence_accounting_units"], 0)
        self.assertIn("Orion target", (await self.workspace.resolve(NodeRef("b"))).text)

    async def test_maintenance_metadata_counts_before_provider_admission(self):
        """Short source text cannot bypass evidence limits through large provenance metadata."""
        await self.workspace.ingest(
            Conversation.from_turns(
                [{"role": "user", "turn_id": "t", "text": "Orion registry region."}],
                node_id="a",
                metadata={"source_description": "recorded provenance " * 300},
            )
        )
        client = ScriptedModelClient([])
        policy = MaintenancePolicy(
            budget=replace(MaintenancePolicy().budget, max_evidence_tokens=2000)
        )
        result = await self.app(maintenance=client, policy=policy).ingest(
            conversation("b", "Orion target region."),
        )
        self.assertEqual(result.maintenance.status, "budget_exhausted")
        self.assertEqual(result.maintenance.usage["model_calls"], 0)
        self.assertEqual(result.maintenance.usage["evidence_accounting_units"], 0)
        self.assertEqual(client.requests, [])
        self.assertTrue(result.source.created)

    async def test_repeated_maintenance_evidence_is_deduplicated_but_calls_are_counted(self):
        """A shared maintenance ledger charges repeated identical spans once and both provider calls."""
        await self.seed(("a", "Orion origin"), ("b", "Orion target"))
        result = await self.app(policy=MaintenancePolicy(mode="propose")).organize(["a", "b"])
        events = [event for event in result.trace if event["kind"] == "model"]
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage["model_calls"], 2)
        self.assertGreater(events[0]["new_evidence_accounting_units"], 0)
        self.assertEqual(events[1]["new_evidence_accounting_units"], 0)
        self.assertEqual(
            result.usage["evidence_accounting_units"], events[0]["new_evidence_accounting_units"]
        )
        self.assertEqual(result.usage["known_input_tokens"], 40)

    async def test_provider_failure_closes_evidence_and_retains_unknown_usage(self):
        """A failed proposal call leaves its attempt visible and releases the local index."""

        async def fail(request):
            """Inject an unavailable provider without modifying storage behavior."""
            raise ProviderError("fixture unavailable")

        await self.seed(("a", "Orion origin"), ("b", "Orion target"))
        opened = []
        original = Evidence.open

        async def track(*args, **kwargs):
            """Retain evidence handles to verify closure through their public methods."""
            handle = await original(*args, **kwargs)
            opened.append(handle)
            return handle

        with patch("llgm.llgm.Evidence.open", side_effect=track):
            result = await self.app(maintenance=CallableModelClient(fail)).organize(["a"])
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.usage["unknown_usage_calls"], 1)
        self.assertEqual(result.trace[0]["status"], "failed")
        with self.assertRaises(ConfigurationError):
            await opened[0].search("Orion", 1)
        self.assertEqual(len(await self.workspace.sources()), 2)

    async def test_cancellation_preserves_publication_and_maintenance_attempt(self):
        """Cancelling maintenance propagates while preserving the committed source and attempted cost."""
        started = asyncio.Event()

        async def wait(request):
            """Hold a model call until the application task is cancelled."""
            started.set()
            await asyncio.Event().wait()

        await self.seed(("a", "Orion origin"))
        app = self.app(maintenance=CallableModelClient(wait))
        task = asyncio.create_task(app.ingest(conversation("b", "Orion target")))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(app.last_maintenance.status, "cancelled")
        self.assertEqual(app.last_maintenance.usage["model_calls"], 1)
        self.assertEqual(app.last_maintenance.trace[0]["status"], "cancelled")
        self.assertIn("Orion target", (await self.workspace.resolve(NodeRef("b"))).text)

    async def test_initial_retrieval_deduplicates_seeds_then_root_receives_branch_returns(self):
        """An ordinary question triggers retrieval before any delegate and one final root call."""
        models = Models()
        app = self.app(root=models.root, sidecar=models.sidecar)
        await app.ingest(
            conversation("release", "Orion rollout date is 2031-04-07."), organize=False
        )
        await app.ingest(
            conversation("distractor", "A hidden sentinel is unrelated."), organize=False
        )
        result = await app.answer("Orion rollout date")
        self.assertEqual(result.status, "completed")
        selection = next(event for event in result.trace if event["kind"] == "seed_selection")
        self.assertEqual(selection["selected"], ["release"])
        self.assertEqual(result.usage["searches"], 1)
        self.assertEqual(len(models.root_requests), 1)
        payload = json.loads(models.root_requests[0].messages[1].content)
        self.assertEqual([branch["node_id"] for branch in payload["branches"]], ["release"])
        self.assertNotIn("hidden sentinel", str(models.root_requests) + str(models.child_requests))
        self.assertNotIn("2031-04-07", models.child_requests[0].messages[1].content)
        self.assertEqual({ref.node_id for ref in result.references}, {"release"})

    async def test_inflight_search_observes_new_nodes_without_changing_old_references(self):
        """A newly ingested source enters later search while earlier citations remain immutable."""
        started, release = asyncio.Event(), asyncio.Event()
        models = Models({"a": [SEARCH, READ]})
        original = models.child
        observations = []

        async def reading_model(request):
            """Pause the seeded delegate so an ordinary ingestion precedes its additional search."""
            if len(request.messages) == 2:
                started.set()
                await release.wait()
            elif len(request.messages) == 4:
                observations.append(json.loads(json.loads(request.messages[-1].content)["stdout"]))
            return await original(request)

        app = self.app(root=models.root, sidecar=CallableModelClient(reading_model))
        await app.ingest(conversation("a", "Orion color is cobalt."), organize=False)
        task = asyncio.create_task(app.answer("Orion color"))
        await asyncio.wait_for(started.wait(), 1)
        await app.ingest(conversation("b", "Orion color is jade."), organize=False)
        release.set()
        result = await asyncio.wait_for(task, 2)
        self.assertEqual(result.status, "completed")
        self.assertEqual(observations[0]["hits"][0]["references"][0]["node_id"], "b")
        self.assertIn("cobalt", (await self.workspace.resolve(NodeRef("a"))).text)
        self.assertEqual(result.usage["searches"], 2)

    async def test_restart_preserves_automatic_edges_independently_of_journals(self):
        """A reopened workspace retains primary connections without journal link assertions."""
        app = self.app()
        await app.ingest(conversation("a", "Orion registry region is eu-west-1."))
        await app.ingest(conversation("b", "Orion registry is used for deployment."))
        await self.workspace.close()
        self.workspace = await Workspace.open(self.path).__aenter__()
        self.assertEqual([edge.target_node_id for edge in await self.workspace.edges("b")], ["a"])
        self.assertEqual(await self.workspace.inspect_journal("b"), [])
        result = await self.app().answer("Orion registry region")
        self.assertEqual(result.status, "completed")
        self.assertIn("a", {ref.node_id for ref in result.references})

    async def test_scoped_primary_edges_ignore_journal_pointer_corrections(self):
        """Scope filters primary edges while a correction to a journal pointer cannot withdraw them."""
        await self.seed(("a", "Orion deployment"), ("b", "Orion registry"))
        edge = await self.workspace.publish_edge(
            "a",
            "b",
            relation="related_to",
            applicability={"scope": {"env": "production"}},
            provenance=Provenance("user", "fixture"),
        )
        note = await self.workspace.append_journal(
            "a",
            subject=NodeRef("a"),
            relation="note",
            value=NodeRef("b"),
            provenance=Provenance("user", "fixture"),
        )
        await self.workspace.append_journal(
            "a",
            subject=JournalRef("a", note.entry_id),
            relation="retract",
            value="This note no longer applies",
            record_kind="correction",
            provenance=Provenance("user", "fixture"),
        )
        evidence = await Evidence.open(self.workspace)
        try:
            production = QueryEvidence(evidence, {"env": "production"}, None)
            staging = QueryEvidence(evidence, {"env": "staging"}, None)
            self.assertEqual(await production.neighbors("a"), [NodeRef("b")])
            self.assertEqual(await staging.neighbors("a"), [])
            await self.workspace.withdraw_edge(
                edge.edge_id, provenance=Provenance("user", "fixture")
            )
            self.assertEqual(await production.neighbors("a"), [])
        finally:
            await evidence.close()
        self.assertEqual(len(await self.workspace.inspect_journal("a")), 2)

    async def test_patch_changes_effective_evidence_without_rewiring_primary_edges(self):
        """An exact local amendment is applied before reasoning and cites its replacement source."""
        await self.seed(
            ("a", "Orion color is cobalt."), ("b", "Orion related deployment"), ("c", "jade")
        )
        await self.workspace.publish_edge(
            "a", "b", relation="related_to", provenance=Provenance("user", "fixture")
        )
        await self.workspace.append_journal(
            "a",
            subject=SourceSpan("a", "t", 15, 21),
            record_kind="overwrite",
            relation="replace",
            value=SourceSpan("c", "t", 0, 4),
            provenance=Provenance("user", "fixture"),
        )
        result = await self.app().answer("Orion color", node_id="a")
        self.assertEqual(result.status, "completed")
        self.assertIn(SourceSpan("c", "t", 0, 4), result.references)
        self.assertEqual([edge.target_node_id for edge in await self.workspace.edges("a")], ["b"])
        self.assertIn("cobalt", (await self.workspace.resolve(NodeRef("a"))).text)
        self.assertEqual(result.usage["searches"], 0)
        self.assertEqual(result.usage["node_invocations"], 1)

    async def test_disabled_maintenance_and_invalid_selection_make_no_model_calls(self):
        """Disabled policy and invalid node selection leave provider clients untouched."""
        model = ScriptedModelClient([])
        app = self.app(maintenance=model, policy=MaintenancePolicy(mode="disabled"))
        result = await app.ingest(conversation("a", "Orion deployment"))
        self.assertEqual(result.maintenance.status, "disabled")
        self.assertEqual(result.maintenance.usage["model_calls"], 0)
        self.assertEqual(model.requests, [])
        failed = await self.app(maintenance=model).organize(["missing"])
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error_type, "ConfigurationError")
        self.assertEqual(model.requests, [])

    async def test_configuration_and_query_scope_fail_before_provider_calls(self):
        """Invalid policy limits or query applicability are rejected before inference."""
        for kwargs in ({"mode": "unknown"}, {"max_nodes": 0}, {"allowed_relations": "related_to"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                MaintenancePolicy(**kwargs)
        with self.assertRaises(ConfigurationError):
            self.app(max_depth=-1)
        with self.assertRaises(ConfigurationError):
            await self.app().answer("Question", scope="production")

    async def test_synchronous_preparation_exhaustion_never_dispatches_a_model(self):
        """A blocking index build consumes the application deadline before provider admission."""
        await self.seed(("a", "Orion source"))
        root = ScriptedModelClient(
            [operation("finish", answer="Unknown", citations=[], unresolved=["No evidence"])]
        )
        app = self.app(root=root)
        evidence = await Evidence.open(self.workspace, passage_chars=app.passage_chars)
        self.addAsyncCleanup(evidence.close)
        clock = NS(now=0.0)

        async def slow_open(*args, **kwargs):
            """Advance application time without yielding, then return the real prepared index."""
            clock.now += 0.04
            return evidence

        with (
            patch("llgm.llgm.Evidence.open", side_effect=slow_open),
            patch("llgm.llgm.time", NS(monotonic=lambda: clock.now)),
        ):
            result = await app.answer("Question", node_id="a", budget=Budget(timeout_seconds=0.01))
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(root.requests, [])
        self.assertEqual(result.usage["model_calls"], 0)
        self.assertEqual(result.usage["preparation_seconds"], 0.04)
        self.assertEqual(result.usage["inference_seconds"], 0)
        self.assertGreaterEqual(result.usage["total_seconds"], result.usage["preparation_seconds"])
        with self.assertRaises(ConfigurationError):
            await evidence.search("Orion", 1)

    async def test_asynchronous_preparation_timeout_retains_timing_and_no_model_attempt(self):
        """An awaiting preparation task is cancelled at the shared deadline and recorded explicitly."""
        cancelled = asyncio.Event()

        async def blocked_open(*args, **kwargs):
            """Wait before allocating an index to test cancellation without leaking a resource."""
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        app = self.app()
        with patch("llgm.llgm.Evidence.open", side_effect=blocked_open):
            result = await app.answer("Question", budget=Budget(timeout_seconds=0.01))
        self.assertTrue(cancelled.is_set())
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.usage["model_calls"], 0)
        self.assertGreaterEqual(result.usage["total_seconds"], 0.01)
        self.assertEqual(result.trace[0]["status"], "budget_exhausted")

    async def test_failed_preparation_retains_timing_before_raising(self):
        """A storage preparation failure preserves attempt timing without inventing model usage."""
        app = self.app()
        with patch("llgm.llgm.Evidence.open", side_effect=ConfigurationError("index unavailable")):
            with self.assertRaises(ConfigurationError):
                await app.answer("Question")
        self.assertEqual(app.last_usage["model_calls"], 0)
        self.assertEqual(app.last_usage["inference_seconds"], 0)
        self.assertGreaterEqual(app.last_usage["total_seconds"], 0)
        self.assertEqual(app.last_trace[0]["status"], "failed")

    async def test_original_date_text_does_not_invent_a_machine_validity_instant(self):
        """The complete operational read plan exposes ambiguous time as unresolved before model use."""
        await self.seed(("a", "Orion region applies here"))
        entry = await self.workspace.append_journal(
            "a",
            subject=NodeRef("a"),
            relation="value",
            value="Orion temporary region",
            provenance=Provenance("user", "fixture"),
            applicability={"valid_from_ms": parse_instant_ms("2023-05-30T00:00:00Z")},
        )
        date = "2023/05/30 (Tue) 10:18"
        models = Models()
        result = await self.app(root=models.root, sidecar=models.sidecar).answer(
            "Orion region", query_date=date
        )
        self.assertEqual(result.status, "partial")
        context = json.loads(models.child_requests[0].messages[1].content)
        self.assertEqual(context["query_date"], date)
        self.assertIsNone(context["journal"]["as_of_ms"])
        self.assertIn(entry.entry_id, context["journal"]["unresolved_entry_ids"])

    async def test_settings_factory_owns_clients_and_workspace_and_inherits_budget(self):
        """The configured application closes factory-owned resources and uses selected limits."""

        class OwnedClient(ScriptedModelClient):
            """Track adapter closure without replacing the application's real storage path."""

            def __init__(self):
                """Initialize a response-free model for lifecycle-only checks."""
                super().__init__([])
                self.closed = False

            async def aclose(self):
                """Record closure of this factory-owned model adapter."""
                self.closed = True

        clients = [OwnedClient(), OwnedClient()]
        settings = Settings(
            workspace_path=str(self.path / "factory"),
            root_model="root",
            sidecar_model="small",
            max_model_calls=7,
            max_output_tokens=512,
        )
        with patch("llgm.models.create_model", side_effect=clients) as create:
            async with LLGM.from_settings(settings, max_depth=2) as app:
                self.assertEqual(app.inference_budget.max_model_calls, 7)
                self.assertEqual(app.inference_budget.max_output_tokens, 512)
                self.assertEqual(app.max_depth, 2)
                self.assertIs(app.maintenance_model, clients[1])
                await app.ingest(conversation("a", "Exact factory source"), organize=False)
                workspace = app.workspace
            self.assertEqual(create.call_count, 2)
        self.assertTrue(all(client.closed for client in clients))
        with self.assertRaises(ConfigurationError):
            await workspace.sources()
        async with Workspace.open(self.path / "factory") as reopened:
            self.assertIn("Exact factory source", (await reopened.resolve(NodeRef("a"))).text)

    async def test_factory_closes_created_resources_when_second_model_construction_fails(self):
        """A partial factory failure closes the first model and opened metadata connection."""
        closed = []

        class FirstClient(ScriptedModelClient):
            """Observe cleanup after another adapter fails to initialize."""

            async def aclose(self):
                """Record that the first constructed client was released."""
                closed.append("first")

        settings = Settings(
            workspace_path=str(self.path / "factory"), root_model="root", sidecar_model="small"
        )
        opened = []
        original = Workspace.open

        def track(*args, **kwargs):
            """Keep the actual Workspace handle so closure is checked by public operations."""
            workspace = original(*args, **kwargs)
            opened.append(workspace)
            return workspace

        with (
            patch("llgm.llgm.Workspace.open", side_effect=track),
            patch(
                "llgm.models.create_model",
                side_effect=[FirstClient([]), ConfigurationError("second adapter unavailable")],
            ),
        ):
            with self.assertRaises(ConfigurationError):
                async with LLGM.from_settings(settings):
                    self.fail("Factory must not yield after partial construction failure")
        self.assertEqual(closed, ["first"])
        with self.assertRaises(ConfigurationError):
            await opened[0].sources()

    async def test_factory_rejects_missing_models_and_unsupported_retriever_before_io(self):
        """Invalid factory configurations cannot create a workspace or model clients."""
        configurations = [
            Settings(workspace_path=str(self.path / "missing")),
            Settings(
                workspace_path=str(self.path / "unsupported"),
                root_model="root",
                sidecar_model="small",
                retriever_backend="dense",
            ),
        ]
        for settings in configurations:
            with (
                self.subTest(settings=settings),
                patch("llgm.llgm.Workspace.open") as opening,
                patch("llgm.models.create_model") as creating,
            ):
                with self.assertRaises(ConfigurationError):
                    async with LLGM.from_settings(settings):
                        self.fail("Invalid configuration must not enter an application context")
                opening.assert_not_called()
                creating.assert_not_called()

    async def test_factory_closes_owned_resources_after_an_inference_exception(self):
        """A model exception escaping inference closes the factory context and preserves its source."""

        class FailingOwnedClient(ScriptedModelClient):
            """Record owned adapter cleanup after an unexpected model-client failure."""

            def __init__(self):
                """Initialize a client whose completion always fails."""
                super().__init__([])
                self.closed = False

            async def complete(self, request):
                """Inject an unexpected client exception after model admission."""
                raise RuntimeError("fixture transport failure")

            async def aclose(self):
                """Record release by the settings factory's exit stack."""
                self.closed = True

        clients = [FailingOwnedClient(), FailingOwnedClient()]
        settings = Settings(
            workspace_path=str(self.path / "factory-inference"),
            root_model="root",
            sidecar_model="small",
        )
        with patch("llgm.models.create_model", side_effect=clients):
            with self.assertRaises(RuntimeError):
                async with LLGM.from_settings(settings, repl_factory=ReplayFactory()) as app:
                    await app.ingest(conversation("a", "Exact persisted source"), organize=False)
                    await app.answer("Exact persisted source")
        self.assertTrue(all(client.closed for client in clients))
        self.assertEqual(app.last_usage["model_calls"], 1)
        self.assertGreaterEqual(
            app.last_usage["total_seconds"], app.last_usage["preparation_seconds"]
        )
        with self.assertRaises(ConfigurationError):
            await app.workspace.sources()
        async with Workspace.open(self.path / "factory-inference") as reopened:
            self.assertIn("Exact persisted source", (await reopened.resolve(NodeRef("a"))).text)

    async def test_custom_evidence_factory_serves_maintenance_and_answers(self):
        """One injected real search backend serves both workflows and remains caller-owned."""
        await self.seed(
            ("a", "Orion region is eu-west-1."), ("b", "Orion deployment uses the region.")
        )
        passages = [
            SearchPassage(
                source.node_id,
                source.turns[0].text,
                (SourceSpan(source.node_id, "t", 0, len(source.turns[0].text)),),
            )
            for source in await self.workspace.sources()
        ]
        retriever = SQLiteBM25Retriever.from_passages(passages)
        opened = []

        async def factory(workspace, *, passage_chars):
            """Attach a caller-owned retriever to each application evidence handle."""
            handle = await Evidence.open(workspace, retriever, passage_chars=passage_chars)
            opened.append(handle)
            return handle

        models = Models()
        root = models.root
        try:
            app = self.app(root=root, sidecar=models.sidecar, evidence_factory=factory)
            maintenance = await app.organize(["b"])
            result = await app.answer("Which region?")
            self.assertEqual(maintenance.status, "completed")
            self.assertEqual(result.status, "completed")
            self.assertIn("a", {ref.node_id for ref in result.references})
            self.assertEqual(len(opened), 2)
            for handle in opened:
                with self.assertRaises(ConfigurationError):
                    await handle.search("Orion", 1)
            self.assertTrue(await retriever.search("Orion", 1))
        finally:
            retriever.close()

    async def test_settings_custom_retrieval_requires_and_uses_an_explicit_factory(self):
        """Custom retrieval configuration is usable without replacing model or storage setup."""
        settings = Settings(
            workspace_path=str(self.path / "injected"),
            root_model="root",
            sidecar_model="small",
            retriever_backend="application-search",
        )
        calls = []

        async def factory(workspace, *, passage_chars):
            """Represent the application's selected backend through the evidence contract."""
            calls.append(workspace)
            return await Evidence.open(workspace, passage_chars=passage_chars)

        models = Models()

        async def close():
            """Match the factory-owned adapter lifecycle in this deterministic test."""

        clients = [models.root, models.sidecar]
        for client in clients:
            client.aclose = close
        with patch("llgm.models.create_model", side_effect=clients):
            async with LLGM.from_settings(
                settings, evidence_factory=factory, repl_factory=ReplayFactory()
            ) as app:
                await app.ingest(conversation("a", "Orion region is eu-west-1."), organize=False)
                self.assertEqual((await app.answer("Which region?")).status, "completed")
        self.assertEqual(len(calls), 1)
        with patch("llgm.llgm.Workspace.open") as opening:
            with self.assertRaises(ConfigurationError):
                async with LLGM.from_settings(settings, evidence_factory=17):
                    self.fail("Invalid factory must not open the application")
            opening.assert_not_called()

    async def test_repeated_answers_reuse_index_and_update_only_changed_sources(self):
        """Application preparation indexes each committed source once across repeated questions."""
        await self.seed(("a", "Orion old region"), ("b", "Atlas other region"))
        root = ScriptedModelClient(
            [
                operation(
                    "finish", answer="Unknown", citations=[], unresolved=["No evidence inspected"]
                )
            ]
            * 3
        )
        app = self.app(root=root)
        first = await app.answer("Which region?")
        second = await app.answer("Which region?")
        await app.ingest(conversation("a-update", "Orion new region"), organize=False)
        third = await app.answer("Which region?")
        self.assertEqual(
            [
                result.trace[0]["index"]["source_records_indexed"]
                for result in (first, second, third)
            ],
            [2, 0, 1],
        )
        self.assertEqual(len(await self.workspace.source_ids()), 3)

    async def test_seed_cap_reports_skipped_nodes_and_queues_all_admitted_nodes(self):
        """A declared seed subset stays visible even if the final model omits its coverage limit."""
        await self.seed(("a", "Orion region A"), ("b", "Orion region B"), ("c", "Orion region C"))
        models = Models()
        result = await self.app(
            root=models.root,
            sidecar=models.sidecar,
            max_seed_nodes=2,
            retrieval_k=6,
            max_concurrency=1,
        ).answer("Orion region")
        selection = next(event for event in result.trace if event["kind"] == "seed_selection")
        self.assertEqual(len(selection["selected"]), 2)
        self.assertEqual(len(selection["skipped"]), 1)
        entered = [event["target_node_id"] for event in result.trace if event["kind"] == "enter"]
        self.assertEqual(entered, selection["selected"])
        self.assertEqual(result.status, "partial")
        self.assertIn("seed_limit", str(result.evidence.unresolved))
        self.assertEqual(len(models.root_requests), 1)

    async def test_empty_retrieval_returns_explicit_gap_without_model_calls(self):
        """A missing seed does not turn into an unsupported root guess or implicit search policy."""
        await self.seed(("a", "Orion region"))
        models = Models()
        result = await self.app(root=models.root, sidecar=models.sidecar).answer("zqxunmatched")
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.usage["model_calls"], 0)
        self.assertEqual(result.usage["searches"], 1)
        self.assertIn("No seed nodes", str(result.evidence.unresolved))
        self.assertEqual(models.root_requests + models.child_requests, [])


if __name__ == "__main__":
    unittest.main()
