"""Real SQLite/blob integration and deterministic evidence/organization contracts.

Scripted models below validate proposal protocol failures. They do not establish
whether a hosted model discovers correct semantic relationships.
"""

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llgm.core.errors import ConfigurationError, ReferenceResolutionError, SchemaError
from llgm.core.types import (
    Conversation,
    JournalRef,
    NodeRef,
    Provenance,
    SourceSpan,
    reference_to_dict,
)
from llgm.memory.evidence import (
    Evidence,
    LinkProposal,
    accept_link,
    interpret_journal,
    propose_links,
)
from llgm.memory.workspace import Workspace
from llgm.models.base import ModelResponse, ScriptedModelClient
from llgm.retrieval.base import SearchHit, SearchPassage
from llgm.retrieval.bm25 import SQLiteBM25Retriever


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    """Current evidence, graph traversal and explicit journal-interpretation contracts."""

    async def asyncSetUp(self):
        """Create isolated storage and track evidence handles for cleanup."""
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.workspace = await Workspace.open(self.path).__aenter__()
        self.handles = []
        self.retrievers = []

    async def asyncTearDown(self):
        """Close owned handles, injected retrievers and storage before removing files."""
        for handle in self.handles:
            await handle.close()
        for retriever in self.retrievers:
            retriever.close()
        await self.workspace.close()
        self.temp.cleanup()

    async def source(self, node_id, text):
        """Publish one user turn and return its exact full-turn source span."""
        await self.workspace.ingest(
            Conversation.from_turns(
                [{"role": "user", "text": text, "turn_id": "t"}], node_id=node_id
            )
        )
        return SourceSpan(node_id, "t", 0, len(text))

    async def open(self, **kwargs):
        """Open current evidence access and retain its handle for teardown."""
        handle = await Evidence.open(self.workspace, **kwargs)
        self.handles.append(handle)
        return handle

    async def assertion(
        self,
        node_id,
        value,
        *,
        applicability=None,
        relation="note",
        subject=None,
        record_kind="assertion",
    ):
        """Append a user-attributed record with a precise target and explicit operation kind."""
        return await self.workspace.append_journal(
            node_id,
            subject=subject or NodeRef(node_id),
            relation=relation,
            value=value,
            record_kind=record_kind,
            provenance=Provenance("user", "unit-contract"),
            applicability=applicability,
        )

    async def edge(self, source, target):
        """Publish primary adjacency without writing semantic journal records."""
        return await self.workspace.publish_edge(
            source, target, provenance=Provenance("user", "unit-contract")
        )

    async def correction(self, target, value, *, relation="supersedes", applicability=None):
        """Append an explicit correction pointing to a prior journal record."""
        return await self.workspace.append_journal(
            target.owning_node_id,
            subject=JournalRef(target.owning_node_id, target.entry_id),
            relation=relation,
            value=value,
            record_kind="correction",
            provenance=Provenance("user", "unit-contract"),
            applicability=applicability,
        )

    async def test_source_and_inline_journal_share_search_and_exact_read_interface(self):
        """Sources and inline journal values expose the same exact-read search contract."""
        ref = await self.source("a", "Production uses harbor-blue.")
        entry = await self.assertion("a", "Staging uses harbor-green.", relation="value")
        evidence = await self.open()
        hits = await evidence.search("harbor", 10)
        refs = [hit.passage.refs[0] for hit in hits]
        self.assertIn(ref, refs)
        self.assertIn(JournalRef("a", entry.entry_id, 0, len(entry.value)), refs)
        for hit in hits:
            self.assertEqual((await evidence.read(hit.passage.refs[0])).text, hit.passage.text)

    async def test_unicode_span_boundaries_and_rechunking_keep_durable_references(self):
        """Unicode slicing and changed passage boundaries preserve canonical coordinates."""
        text = "  e\u0301 é 👩🏽‍💻\r\n雪 harbor  "
        await self.source("a", text)
        for width in (1, 3, 8):
            evidence = await self.open(passage_chars=width)
            # Searchable slices are exact references, including partial Unicode
            # graphemes: persisted coordinates count Python code points.
            for hit in await evidence.search("harbor 雪", 100):
                ref = hit.passage.refs[0]
                self.assertEqual(hit.passage.text, text[ref.start : ref.end])
                self.assertEqual((await evidence.read(ref)).text, hit.passage.text)
            self.assertEqual((await evidence.read(SourceSpan("a", "t", 2, 4))).text, "e\u0301")

    async def test_current_reads_include_new_sources_journals_and_links(self):
        """An existing evidence handle refreshes new publications without altering old sources."""
        await self.source("a", "original azure")
        await self.source("b", "neighbor cobalt")
        evidence = await self.open()
        async with Workspace.open(self.path) as writer:
            await writer.ingest(
                Conversation.from_turns(
                    [{"role": "user", "text": "additional magenta", "turn_id": "t"}], node_id="c"
                )
            )
            await writer.publish_edge("a", "b", provenance=Provenance("user", "writer"))
            await writer.append_journal(
                "a",
                subject=NodeRef("a"),
                relation="value",
                value="future magenta",
                provenance=Provenance("user", "writer"),
            )
        self.assertIn("original azure", (await evidence.read(NodeRef("a"))).text)
        self.assertEqual(len(await evidence.search("magenta", 10)), 2)
        self.assertEqual(len(await evidence.journal("a")), 1)
        self.assertEqual(await evidence.neighbors("a"), [NodeRef("b")])
        self.assertIn("additional magenta", (await evidence.read(NodeRef("c"))).text)
        self.assertEqual(evidence.descriptor()["visibility"], "current")

    async def test_immutable_citations_and_journal_history_survive_restart(self):
        """New evidence nodes leave original source and journal citations resolvable after restart."""
        ref = await self.source("a", "harbor exact source")
        entry = await self.assertion("a", "harbor journal evidence")
        await self.open()
        await self.source("b", "new evidence")
        await self.workspace.close()
        self.workspace = await Workspace.open(self.path).__aenter__()
        reopened = await self.open(passage_chars=7)
        self.assertEqual((await reopened.read(ref)).text, "harbor exact source")
        self.assertEqual(
            (await reopened.read(JournalRef("a", entry.entry_id, 0, 6))).text, "harbor"
        )
        self.assertEqual((await reopened.read(NodeRef("a"))).reference, NodeRef("a"))

    async def test_dangling_and_out_of_range_references_fail_explicitly(self):
        """Missing nodes, unknown entries and invalid ranges fail explicitly."""
        await self.source("a", "text")
        evidence = await self.open()
        refs = (
            NodeRef("missing"),
            SourceSpan("a", "t", 0, 5),
            JournalRef("a", "missing"),
            SourceSpan("a", "wrong-turn", 0, 1),
        )
        for ref in refs:
            with self.subTest(reference=ref), self.assertRaises(ReferenceResolutionError):
                await evidence.read(ref)
        with self.assertRaises(ReferenceResolutionError):
            await evidence.neighbors("missing")

    async def test_evidence_requires_sources_to_exist_in_its_workspace(self):
        """A reference to evidence stored only in another workspace cannot be read here."""
        async with Workspace.open(self.path / "other") as other:
            node = await other.ingest(
                Conversation.from_turns([{"role": "user", "text": "elsewhere"}])
            )
        evidence = await self.open()
        with self.assertRaises(ReferenceResolutionError):
            await evidence.read(NodeRef(node.node_id))

    async def test_pointer_and_inline_journal_are_distinct_resolvable_evidence(self):
        """Pointer records and inline journal values retain distinct readable forms."""
        await self.source("a", "origin")
        target = await self.source("b", "target statement")
        pointer = await self.assertion("a", target)
        inline = await self.assertion("a", "qualification: staging only")
        evidence = await self.open()
        self.assertEqual(await evidence.neighbors("a"), [])
        record = json.loads((await evidence.read(JournalRef("a", pointer.entry_id))).text)
        self.assertEqual(record["value"], reference_to_dict(target))
        self.assertEqual((await evidence.read(target)).text, "target statement")
        self.assertEqual(
            (await evidence.read(JournalRef("a", inline.entry_id, 15, 22))).text, "staging"
        )
        with self.assertRaises(ReferenceResolutionError):
            await evidence.read(JournalRef("a", pointer.entry_id, 0, 1))

    async def test_graph_links_are_directed_filtered_and_deduplicated(self):
        """Neighbor discovery follows generic directed connections without duplicate endpoints."""
        for node in "abc":
            await self.source(node, node)
        await self.edge("a", "b")
        await self.edge("a", "b")
        await self.edge("a", "c")
        evidence = await self.open()
        self.assertEqual(await evidence.neighbors("a"), [NodeRef("b"), NodeRef("c")])
        self.assertEqual(await evidence.neighbors("b"), [])
        self.assertEqual(await evidence.neighbors("a"), [NodeRef("b"), NodeRef("c")])

    async def test_edge_descriptions_preserve_generic_connections_and_stored_attribution(self):
        """Neighbor descriptions retain real edge metadata without loading target text or inventing relevance."""
        source = await self.source("a", "entry source sentinel")
        target = await self.source("b", "credential source sentinel")
        await self.source("c", "contact source sentinel")
        support = await self.assertion("a", "Relationship approved")
        provenance = Provenance(
            "model",
            "maintenance",
            (source, target, JournalRef("a", support.entry_id)),
            model="test-model",
            prompt_version="test-prompt",
            policy_version="reviewed",
        )
        applicability = {"scope": {"env": "prod"}, "valid_from_ms": 100}
        credential = await self.workspace.publish_edge(
            "a", "b", provenance=provenance, applicability=applicability
        )
        corroboration = await self.edge("a", "b")
        contact = await self.edge("a", "c")
        evidence = await self.open()
        with patch.object(
            self.workspace.blob_store,
            "get",
            side_effect=AssertionError("Edge descriptions must not read source text"),
        ):
            descriptions = await evidence.edge_descriptions("a")
            self.assertEqual(await evidence.neighbors("a"), [NodeRef("b"), NodeRef("c")])
        records = {record["edge_id"]: record for record in descriptions}
        self.assertEqual(set(records), {credential.edge_id, corroboration.edge_id, contact.edge_id})
        expected = {
            "edge_id": credential.edge_id,
            "source_node_id": "a",
            "reference": {"type": "node", "node_id": "b"},
            "provenance": {
                "origin": "model",
                "producer": "maintenance",
                "supporting_references": [
                    reference_to_dict(reference) for reference in provenance.supporting_references
                ],
                "model": "test-model",
                "prompt_version": "test-prompt",
                "policy_version": "reviewed",
            },
            "applicability": applicability,
            "recorded_at_ms": credential.recorded_at_ms,
        }
        self.assertEqual(records[credential.edge_id], expected)
        self.assertEqual(
            next(
                row
                for row in await evidence.edge_descriptions("a")
                if row["edge_id"] == credential.edge_id
            ),
            expected,
        )
        self.assertEqual(records[contact.edge_id]["reference"], {"type": "node", "node_id": "c"})
        self.assertNotIn("relation", records[contact.edge_id])
        self.assertEqual(await evidence.edge_descriptions("b"), [])
        self.assertNotIn("source sentinel", json.dumps(descriptions))
        records[credential.edge_id]["applicability"]["scope"]["env"] = "changed"
        records[credential.edge_id]["provenance"]["supporting_references"][0]["node_id"] = "c"
        self.assertEqual(
            next(
                row
                for row in await evidence.edge_descriptions("a")
                if row["edge_id"] == credential.edge_id
            ),
            expected,
        )

    async def test_edge_descriptions_refresh_withdrawal_and_survive_reopen(self):
        """Description reads follow current edge publication and withdrawal across workspace handles."""
        for node in "abc":
            await self.source(node, node)
        edge = await self.edge("a", "b")
        evidence = await self.open()
        original = await evidence.edge_descriptions("a")
        self.assertEqual([record["edge_id"] for record in original], [edge.edge_id])
        async with Workspace.open(self.path) as writer:
            await writer.withdraw_edge(edge.edge_id, provenance=Provenance("user", "reviewer"))
            replacement = await writer.publish_edge(
                "a", "c", provenance=Provenance("user", "reviewer")
            )
        current = await evidence.edge_descriptions("a")
        self.assertEqual([record["edge_id"] for record in current], [replacement.edge_id])
        await self.workspace.close()
        self.workspace = await Workspace.open(self.path).__aenter__()
        reopened = await self.open()
        self.assertEqual(await reopened.edge_descriptions("a"), current)
        with self.assertRaises(ReferenceResolutionError):
            await reopened.edge_descriptions("missing")
        await reopened.close()
        with self.assertRaises(ConfigurationError):
            await reopened.edge_descriptions("a")

    async def test_traversal_handles_cycle_diamond_and_both_bounds(self):
        """Traversal terminates on cycles and respects depth and node-count bounds."""
        for node in "abcd":
            await self.source(node, node)
        for source, target in (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "a")):
            await self.edge(source, target)
        evidence = await self.open()
        with patch.object(
            self.workspace.blob_store,
            "get",
            side_effect=AssertionError("Graph discovery must not read source text"),
        ):
            complete = await evidence.traverse("a", max_depth=3, max_nodes=4)
        self.assertEqual([ref.node_id for ref in complete.references], list("abcd"))
        self.assertFalse(complete.truncated)
        depth_limited = await evidence.traverse("a", max_depth=1)
        self.assertEqual([ref.node_id for ref in depth_limited.references], list("abc"))
        self.assertTrue(depth_limited.truncated)
        node_limited = await evidence.traverse("a", max_nodes=2)
        self.assertEqual([ref.node_id for ref in node_limited.references], list("ab"))
        self.assertTrue(node_limited.truncated)
        zero = await evidence.traverse("a", max_depth=0)
        self.assertEqual(zero.references, (NodeRef("a"),))
        self.assertTrue(zero.truncated)

    async def test_graph_corrections_are_exposed_as_history_not_new_edges(self):
        """Graph correction records remain inspectable history rather than new structural links."""
        await self.source("a", "origin")
        await self.source("b", "target")
        link = await self.assertion("a", NodeRef("b"))
        correction = await self.correction(link, "Wrong relationship", relation="retract")
        evidence = await self.open()
        # A journal pointer never creates primary adjacency.
        self.assertEqual(await evidence.neighbors("a"), [])
        result = interpret_journal(await evidence.journal("a"))
        self.assertEqual(result.inactive, (link,))
        self.assertEqual(result.active, (correction,))

    async def test_external_retriever_text_is_replaced_by_canonical_evidence(self):
        """Injected search text is replaced by the referenced immutable source bytes."""
        ref = await self.source("a", "correct cobalt setting")
        retriever = SQLiteBM25Retriever.from_passages(
            [SearchPassage("metadata-passage", "invented cobalt setting (metadata)", (ref,))]
        )
        self.retrievers.append(retriever)
        evidence = await self.open(retriever=retriever)
        hits = await evidence.search("cobalt", 5)
        self.assertEqual(hits[0].passage.text, "correct cobalt setting")
        self.assertNotIn("invented", hits[0].passage.text)
        self.assertEqual(hits[0].passage.refs, (ref,))

    async def test_external_retrieval_rejects_missing_and_out_of_range_references(self):
        """External hits cannot introduce missing sources or nonexistent text ranges."""
        await self.source("a", "old cobalt")
        for ref in (SourceSpan("a", "t", 0, 100), SourceSpan("missing", "t", 0, 1)):
            retriever = SQLiteBM25Retriever.from_passages([SearchPassage("p", "cobalt", (ref,))])
            self.retrievers.append(retriever)
            evidence = await self.open(retriever=retriever)
            with self.assertRaises(ReferenceResolutionError):
                await evidence.search("cobalt", 5)

    async def test_external_retrieval_requires_refs_and_deduplicates_evidence(self):
        """External retrieval requires canonical references and deduplicates repeated evidence."""
        ref = await self.source("a", "cobalt")
        for references, expected in (((), 0), ((ref,), 1)):
            retriever = SQLiteBM25Retriever.from_passages(
                [
                    SearchPassage("p1", "cobalt", references),
                    SearchPassage("p2", "cobalt", references),
                ]
            )
            self.retrievers.append(retriever)
            evidence = await self.open(retriever=retriever)
            if not expected:
                with self.assertRaises(ReferenceResolutionError):
                    await evidence.search("cobalt", 5)
            else:
                self.assertEqual(len(await evidence.search("cobalt", 5)), 1)

    async def test_inline_journals_remain_searchable_with_injected_source_backend(self):
        """Injecting a source retriever does not hide searchable inline journal values."""
        ref = await self.source("a", "cobalt source")
        entry = await self.assertion("a", "cobalt journal")
        retriever = SQLiteBM25Retriever.from_passages([SearchPassage("p", "cobalt source", (ref,))])
        self.retrievers.append(retriever)
        evidence = await self.open(retriever=retriever)
        hits = await evidence.search("cobalt", 5)
        self.assertEqual({ref.node_id for hit in hits for ref in hit.passage.refs}, {"a"})
        self.assertEqual(len(hits), 2)
        self.assertTrue(
            any(
                isinstance(hit.passage.refs[0], JournalRef)
                and hit.passage.refs[0].entry_id == entry.entry_id
                for hit in hits
            )
        )

    async def test_search_literal_queries_limits_and_close(self):
        """Search treats syntax literally, validates limits and rejects use after close."""
        await self.source("a", "cobalt")
        evidence = await self.open()
        self.assertEqual(await evidence.search("", 5), [])
        self.assertEqual(await evidence.search("cobalt", 0), [])
        self.assertEqual(len(await evidence.search('cobalt OR "', 5)), 1)
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                await evidence.search("cobalt", value)
        await evidence.close()
        with self.assertRaises(ConfigurationError):
            await evidence.read(NodeRef("a"))
        with self.assertRaises(ConfigurationError):
            await evidence.search("cobalt", 1)
        await evidence.close()

    async def test_concurrent_searches_have_consistent_results(self):
        """Concurrent searches without writes return consistent results from the shared index."""
        await self.source("a", "cobalt")
        evidence = await self.open()
        results = await asyncio.gather(*(evidence.search("cobalt", 3) for _ in range(8)))
        self.assertTrue(all(result == results[0] for result in results))

    async def test_scope_mismatch_missing_scope_and_suggestions_do_not_override_facts(self):
        """Scope and proposal status prevent unrelated values from replacing active facts."""
        await self.source("a", "deployment")
        production = await self.assertion(
            "a", "blue", relation="value", applicability={"scope": {"env": "production"}}
        )
        staging = await self.assertion(
            "a", "green", relation="value", applicability={"scope": {"env": "staging"}}
        )
        suggestion = await self.assertion(
            "a", "Perhaps red", relation="suggests", applicability={"scope": {"env": "production"}}
        )
        records = [production, staging, suggestion]
        result = interpret_journal(records, scope={"env": "production"})
        self.assertEqual(result.active, (production,))
        self.assertEqual(result.inactive, (staging,))
        self.assertEqual(result.proposed, (suggestion,))
        self.assertEqual(result.conflicts, ())
        unspecified = interpret_journal(records)
        self.assertEqual(unspecified.unresolved, tuple(records))

    async def test_explicit_temporal_correction_preserves_historical_answer(self):
        """A dated correction affects only its applicable time and scope while preserving history."""
        await self.source("a", "setting")
        old = await self.assertion("a", "blue", relation="value")
        edit = await self.correction(
            old, "green", applicability={"scope": {"env": "production"}, "valid_from": "2025-02-01"}
        )
        historical = interpret_journal([old, edit], scope={"env": "production"}, as_of="2025-01-31")
        self.assertEqual(historical.active, (old,))
        self.assertEqual(historical.inactive, (edit,))
        current = interpret_journal([old, edit], scope={"env": "production"}, as_of="2025-02-01")
        self.assertEqual(current.active, (edit,))
        self.assertEqual(current.inactive, (old,))
        staging = interpret_journal([old, edit], scope={"env": "staging"}, as_of="2025-02-01")
        self.assertEqual(staging.active, (old,))
        self.assertEqual(staging.inactive, (edit,))
        self.assertEqual(
            (await self.workspace.resolve(JournalRef("a", old.entry_id, 0, 4))).text, "blue"
        )

    async def test_valid_time_is_half_open_and_requires_query_time(self):
        """Valid-time intervals include their start, exclude their end and need a query time."""
        await self.source("a", "setting")
        entry = await self.assertion(
            "a",
            "temporary",
            applicability={"valid_from": "2025-01-01", "valid_until": "2025-02-01"},
        )
        self.assertEqual(interpret_journal([entry]).unresolved, (entry,))
        self.assertEqual(
            interpret_journal([entry], as_of="2025-01-01T01:00:00+01:00").active, (entry,)
        )
        self.assertEqual(interpret_journal([entry], as_of="2025-02-01").inactive, (entry,))

    async def test_correction_cannot_broaden_target_scope(self):
        """An unscoped correction cannot expand its target's declared applicability."""
        await self.source("a", "deployment")
        production = await self.assertion(
            "a", "blue", relation="value", applicability={"scope": {"env": "production"}}
        )
        edit = await self.correction(production, "green")
        staging = interpret_journal([production, edit], scope={"env": "staging"})
        self.assertEqual(staging.active, ())
        self.assertEqual(staging.inactive, (production, edit))
        unspecified = interpret_journal([production, edit])
        self.assertEqual(unspecified.unresolved, (production, edit))

    async def test_retraction_of_correction_restores_original_without_deleting_history(self):
        """Retracting an edit restores prior interpretation without removing journal records."""
        await self.source("a", "setting")
        original = await self.assertion("a", "blue", relation="value")
        update = await self.correction(original, "green")
        retract_update = await self.correction(update, "Update was wrong", relation="retract")
        result = interpret_journal([original, update, retract_update])
        self.assertEqual(result.active, (original, retract_update))
        self.assertEqual(result.inactive, (update,))
        retract_retraction = await self.correction(
            retract_update, "Update was correct", relation="retract"
        )
        again = interpret_journal([original, update, retract_update, retract_retraction])
        self.assertEqual(again.active, (update, retract_retraction))
        self.assertEqual(again.inactive, (original, retract_update))
        self.assertEqual(len(await self.workspace.inspect_journal("a")), 4)

    async def test_ambiguous_replacement_of_correction_preserves_base_fact(self):
        """An ambiguous edit of an edit leaves the base assertion available."""
        await self.source("a", "setting")
        original = await self.assertion("a", "blue", relation="value")
        update = await self.correction(original, "green")
        ambiguous = await self.correction(update, "orange")
        result = interpret_journal([original, update, ambiguous])
        self.assertEqual(result.active, (original,))
        self.assertEqual(result.unresolved, (update, ambiguous))

    async def test_unknown_applicability_and_undated_historical_edits_stay_unresolved(self):
        """Unknown conditions and undated historical corrections remain unresolved."""
        await self.source("a", "setting")
        old = await self.assertion("a", "blue")
        edit = await self.correction(old, "green")
        unknown = await self.assertion(
            "a", "conditional", applicability={"arbitrary_semantics": "maybe"}
        )
        result = interpret_journal([old, edit, unknown], as_of="2020-01-01")
        self.assertEqual(result.active, (old,))
        self.assertEqual(result.unresolved, (edit, unknown))

    async def test_contradictory_values_and_competing_edits_remain_visible(self):
        """Contradictory assertions and competing corrections retain visible conflicts."""
        await self.source("a", "setting")
        first = await self.assertion("a", "blue", relation="value")
        second = await self.assertion("a", "green", relation="value")
        result = interpret_journal([first, second])
        self.assertEqual(result.active, (first, second))
        self.assertEqual(result.conflicts, ((first.entry_id, second.entry_id),))
        edit1 = await self.correction(first, "red")
        edit2 = await self.correction(first, "orange")
        competing = interpret_journal([first, edit1, edit2])
        self.assertEqual(competing.active, (first,))
        self.assertEqual(competing.unresolved, (edit1, edit2))
        self.assertEqual(competing.conflicts, ((edit1.entry_id, edit2.entry_id),))

    async def test_latest_explicit_overwrite_uses_append_sequence_not_time_or_input_order(self):
        """The newest explicit same-slot overwrite wins even when timestamps and input order disagree."""
        span = await self.source("a", "production setting")
        original = await self.assertion("a", "blue", relation="value", subject=span)
        middle = await self.assertion(
            "a", "green", relation="value", subject=span, record_kind="overwrite"
        )
        latest = await self.assertion(
            "a", "gold", relation="value", subject=span, record_kind="overwrite"
        )
        original = replace(original, recorded_at_ms=3000)
        middle = replace(middle, recorded_at_ms=2000)
        latest = replace(latest, recorded_at_ms=1000)
        result = interpret_journal([latest, original, middle])
        self.assertEqual(result.active, (latest,))
        self.assertEqual(result.inactive, (original, middle))
        self.assertEqual(result.conflicts, ())
        self.assertEqual(result.reasons[original.entry_id], f"overwritten by {latest.entry_id}")
        self.assertEqual(result, interpret_journal([original, middle, latest]))

    async def test_overwrite_key_preserves_exact_spans_relations_and_declared_scopes(self):
        """Overwriting one production span cannot replace adjacent, overlapping, or differently scoped facts."""
        await self.source("a", "alpha beta")
        precise = SourceSpan("a", "t", 0, 5)
        production = {"scope": {"env": "production", "team": "core"}}
        original = await self.assertion(
            "a", "old", subject=precise, relation="value", applicability=production
        )
        adjacent = await self.assertion(
            "a",
            "beta",
            subject=SourceSpan("a", "t", 6, 10),
            relation="value",
            applicability=production,
        )
        overlapping = await self.assertion(
            "a",
            "whole statement",
            subject=SourceSpan("a", "t", 0, 10),
            relation="value",
            applicability=production,
        )
        other_relation = await self.assertion(
            "a", "support", subject=precise, relation="supported_by", applicability=production
        )
        unscoped = await self.assertion("a", "broad", subject=precise, relation="value")
        staging = await self.assertion(
            "a",
            "staging",
            subject=precise,
            relation="value",
            applicability={"scope": {"env": "staging", "team": "core"}},
        )
        latest = await self.assertion(
            "a",
            "new",
            subject=precise,
            relation="value",
            record_kind="overwrite",
            applicability={"scope": {"team": "core", "env": "production"}},
        )
        entries = await self.workspace.inspect_journal("a")
        result = interpret_journal(entries, scope={"env": "production", "team": "core"})
        self.assertEqual(result.active, (adjacent, overlapping, other_relation, unscoped, latest))
        self.assertEqual(result.inactive, (original, staging))
        self.assertEqual(result.conflicts, ())
        other_scope = interpret_journal(entries, scope={"env": "staging", "team": "core"})
        self.assertEqual(other_scope.active, (unscoped, staging))
        missing = interpret_journal(entries, scope={"env": "production"})
        self.assertIn(original, missing.unresolved)
        self.assertIn(latest, missing.unresolved)

    async def test_support_assertions_remain_multivalued_until_an_explicit_overwrite(self):
        """Ordinary links append before and after an overwrite; only a later overwrite replaces its slot."""
        for node_id in ("a", "b", "c", "d"):
            await self.source(node_id, node_id)
        first = await self.assertion("a", NodeRef("b"), relation="supported_by")
        second = await self.assertion("a", NodeRef("c"), relation="supported_by")
        self.assertEqual(interpret_journal([first, second]).active, (first, second))
        overwrite = await self.assertion(
            "a", NodeRef("d"), relation="supported_by", record_kind="overwrite"
        )
        after = await self.assertion("a", NodeRef("b"), relation="supported_by")
        result = interpret_journal(await self.workspace.inspect_journal("a"))
        self.assertEqual(result.active, (overwrite, after))
        self.assertEqual(result.inactive, (first, second))
        latest = await self.assertion(
            "a", NodeRef("c"), relation="supported_by", record_kind="overwrite"
        )
        result = interpret_journal(await self.workspace.inspect_journal("a"))
        self.assertEqual(result.active, (latest,))
        self.assertEqual(result.inactive, (first, second, overwrite, after))
        evidence = await self.open()
        self.assertEqual(await evidence.neighbors("a"), [])

    async def test_ineligible_overwrites_never_suppress_applicable_evidence(self):
        """Future, expired, differently scoped, unknown, and suggested edits leave the current fact available."""
        span = await self.source("a", "setting")
        original = await self.assertion(
            "a",
            "blue",
            subject=span,
            relation="value",
            applicability={"scope": {"env": "production"}},
        )
        cases = (
            ({"scope": {"env": "production"}, "valid_from": "2026-02-01"}, "value", "inactive"),
            ({"scope": {"env": "production"}, "valid_until": "2025-12-01"}, "value", "inactive"),
            ({"scope": {"env": "staging"}}, "value", "inactive"),
            ({"scope": {"env": "production"}, "unknown": True}, "value", "unresolved"),
            ({"scope": {"env": "production"}}, "suggests", "proposed"),
        )
        for applicability, relation, category in cases:
            overwrite = await self.assertion(
                "a",
                "green",
                subject=span,
                relation=relation,
                applicability=applicability,
                record_kind="overwrite",
            )
            with self.subTest(applicability=applicability, relation=relation):
                result = interpret_journal(
                    [overwrite, original], scope={"env": "production"}, as_of="2026-01-01"
                )
                self.assertEqual(result.active, (original,))
                self.assertEqual(getattr(result, category), (overwrite,))

    async def test_temporary_overwrite_preserves_history_and_restores_outside_its_interval(self):
        """An overwrite applies only during its valid interval and all original records survive restart."""
        span = await self.source("a", "setting")
        original = await self.assertion("a", "blue", subject=span, relation="value")
        temporary = await self.assertion(
            "a",
            "green",
            subject=span,
            relation="value",
            record_kind="overwrite",
            applicability={"valid_from": "2026-02-01", "valid_until": "2026-03-01"},
        )
        await self.workspace.close()
        self.workspace = await Workspace.open(self.path).__aenter__()
        records = await self.workspace.inspect_journal("a")
        self.assertEqual(records, [original, temporary])
        for date, winner in (
            ("2026-01-01", original),
            ("2026-02-01", temporary),
            ("2026-03-01", original),
        ):
            with self.subTest(date=date):
                self.assertEqual(interpret_journal(records, as_of=date).active, (winner,))
        self.assertEqual((await self.workspace.resolve(span)).text, "setting")
        self.assertEqual(
            (await self.workspace.resolve(JournalRef("a", original.entry_id, 0, 4))).text, "blue"
        )

    async def test_retracting_an_overwrite_restores_its_predecessor(self):
        """Explicit correction of an overwrite can restore the earlier interpretation without deleting records."""
        span = await self.source("a", "setting")
        original = await self.assertion("a", "blue", subject=span, relation="value")
        overwrite = await self.assertion(
            "a", "green", subject=span, relation="value", record_kind="overwrite"
        )
        correction = await self.correction(overwrite, "unsupported update", relation="retract")
        result = interpret_journal([correction, overwrite, original])
        self.assertEqual(result.active, (original, correction))
        self.assertEqual(result.inactive, (overwrite,))

    async def test_reducer_preserves_every_record_and_rejects_invalid_intervals(self):
        """Interpretation accounts for every record and rejects invalid temporal inputs."""
        await self.source("a", "setting")
        valid = await self.assertion("a", "blue")
        malformed = replace(
            valid, applicability={"valid_from": "2025-02-01", "valid_until": "2025-01-01"}
        )
        with self.assertRaises(SchemaError):
            interpret_journal([malformed], as_of="2025-01-01")
        with self.assertRaises(SchemaError):
            interpret_journal([valid, valid])
        with self.assertRaises(SchemaError):
            interpret_journal(
                [valid, replace(valid, entry_id="duplicate-sequence", record_kind="overwrite")]
            )
        with self.assertRaises(SchemaError):
            interpret_journal([replace(valid, journal_sequence=0)])
        with self.assertRaises(SchemaError):
            interpret_journal([valid], as_of="2025-01-01T00:00:00")
        unknown_target = replace(
            valid,
            entry_id="correction",
            journal_sequence=valid.journal_sequence + 1,
            subject=JournalRef("a", "missing"),
            record_kind="correction",
            relation="retract",
        )
        result = interpret_journal([valid, unknown_target])
        self.assertEqual(result.unresolved, (unknown_target,))
        self.assertEqual(
            len(result.active + result.inactive + result.proposed + result.unresolved), 2
        )

    async def proposal_fixture(self):
        """Create two disconnected sources and a fully supported proposed-link payload."""
        source = await self.source("a", "Orion uses cobalt storage.")
        target = await self.source("b", "Orion storage is deployed in eu-west.")
        evidence = await self.open()
        payload = {
            "links": [
                {
                    "target_node_id": "b",
                    "supporting_references": [reference_to_dict(source), reference_to_dict(target)],
                    "rationale": "Both describe Orion storage.",
                }
            ]
        }
        return evidence, payload, source, target

    async def test_model_proposes_disconnected_links_without_publication(self):
        """Model-generated link proposals do not alter the persistent graph."""
        evidence, payload, source, target = await self.proposal_fixture()
        model = ScriptedModelClient([ModelResponse(json.dumps(payload), model="small-model")])
        proposals = await propose_links(evidence, "a", model)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].source, NodeRef("a"))
        self.assertEqual(proposals[0].target, NodeRef("b"))
        self.assertEqual(proposals[0].supporting_references, (source, target))
        self.assertEqual(await evidence.neighbors("a"), [])
        self.assertEqual(await self.workspace.inspect_journal("a"), [])

    async def test_link_proposals_retain_late_search_matches_and_speaker_dates(self):
        """Late matched evidence and its attribution survive discovery and caller-owned retrieval."""
        source_text = "Orion uses cobalt storage."
        target_text = "Consider cobalt storage for Orion."
        await self.workspace.ingest(
            Conversation.from_turns(
                [{"role": "user", "text": source_text, "turn_id": "decision"}],
                node_id="a",
                metadata={"date": "2026-09-10", "scope": {"service": "Orion"}},
                timestamp_ms=1788998400000,
            )
        )
        await self.workspace.ingest(
            Conversation.from_turns(
                [
                    {
                        "role": "user",
                        "text": "Unrelated introductory background. " * 10,
                        "turn_id": "intro",
                    },
                    {"role": "assistant", "text": target_text, "turn_id": "suggestion"},
                ],
                node_id="b",
                metadata={"date": "2026-09-09", "scope": {"service": "Orion"}},
            )
        )
        evidence = await self.open(passage_chars=80)
        hits = await evidence.search(source_text, 8)
        source_ref = SourceSpan("a", "decision", 0, len(source_text))
        target_ref = SourceSpan("b", "suggestion", 0, len(target_text))
        self.assertTrue(any(target_ref in hit.passage.refs for hit in hits))
        payload = {
            "links": [
                {
                    "target_node_id": "b",
                    "supporting_references": [
                        reference_to_dict(source_ref),
                        reference_to_dict(target_ref),
                    ],
                    "rationale": "Both passages discuss cobalt storage for Orion.",
                }
            ]
        }
        for supplied in (False, True):
            with self.subTest(supplied=supplied):
                model = ScriptedModelClient([json.dumps(payload)])
                if supplied:
                    with patch.object(
                        evidence, "search", side_effect=AssertionError("Search already charged")
                    ):
                        proposals = await propose_links(
                            evidence,
                            "a",
                            model,
                            candidate_node_ids=["b"],
                            candidate_hits=hits,
                            max_context_chars=160,
                        )
                else:
                    proposals = await propose_links(evidence, "a", model, max_context_chars=160)
                self.assertEqual(proposals[0].supporting_references, (source_ref, target_ref))
                presented = json.loads(model.requests[0].messages[-1].content)["passages"]
                by_node = {item["reference"]["node_id"]: item for item in presented}
                self.assertEqual(by_node["b"]["text"], target_text)
                self.assertEqual(by_node["b"]["metadata"]["role"], "assistant")
                self.assertEqual(by_node["b"]["metadata"]["source_metadata"]["date"], "2026-09-09")
                self.assertEqual(by_node["a"]["metadata"]["role"], "user")
                self.assertEqual(by_node["a"]["metadata"]["timestamp_ms"], 1788998400000)
                self.assertIsNone(by_node["b"]["metadata"]["timestamp_ms"])
                self.assertEqual(
                    by_node["a"]["metadata"]["source_metadata"]["scope"], {"service": "Orion"}
                )
                self.assertLessEqual(sum(len(item["text"]) for item in presented), 160)
                accepted = await accept_link(evidence, proposals[0])
                self.assertIn(target_ref, accepted.provenance.supporting_references)

    async def test_supplied_link_hits_use_canonical_text_and_metadata(self):
        """Caller-provided hit text and attribution cannot replace immutable source evidence."""
        evidence, _, _, target = await self.proposal_fixture()
        hit = next(hit for hit in await evidence.search("storage", 8) if target in hit.passage.refs)
        forged = replace(
            hit,
            passage=replace(
                hit.passage,
                text="Invented source text",
                metadata={"role": "system", "date": "invented date"},
            ),
        )
        model = ScriptedModelClient(['{"links": []}'])
        with patch.object(
            evidence, "search", side_effect=AssertionError("Unexpected repeated retrieval")
        ):
            await propose_links(evidence, "a", model, candidate_hits=[forged])
        payload = json.loads(model.requests[0].messages[-1].content)
        self.assertEqual(payload["candidate_node_ids"], ["b"])
        passage = next(item for item in payload["passages"] if item["reference"]["node_id"] == "b")
        resolved = await evidence.read(target)
        self.assertEqual(passage["text"], resolved.text)
        self.assertEqual(passage["metadata"], resolved.metadata)
        self.assertNotIn("invented", json.dumps(passage).lower())

    async def test_link_match_trimming_preserves_nonzero_unicode_offsets(self):
        """Trimming a matched passage keeps its original start and valid Unicode source coordinates."""
        await self.source("a", "cobalt")
        target = await self.source("b", "prefix cobalt 雪😺 storage with more evidence")
        reference = replace(target, start=7)
        retriever = SQLiteBM25Retriever.from_passages(
            [
                SearchPassage("late-match", "cobalt", (reference,)),
            ]
        )
        self.retrievers.append(retriever)
        evidence = await self.open(retriever=retriever)
        hits = await evidence.search("cobalt", 8)
        model = ScriptedModelClient(['{"links": []}'])
        await propose_links(evidence, "a", model, candidate_hits=hits, max_context_chars=24)
        presented = json.loads(model.requests[0].messages[-1].content)["passages"]
        passage = next(item for item in presented if item["reference"]["node_id"] == "b")
        self.assertEqual(passage["reference"]["start"], 7)
        self.assertEqual(passage["reference"]["end"], 19)
        self.assertEqual(passage["text"], (await evidence.read(replace(reference, end=19))).text)
        self.assertLessEqual(sum(len(item["text"]) for item in presented), 24)

    async def test_acceptance_persists_exact_provenance_and_is_idempotent(self):
        """Accepted links preserve model provenance and support while retries stay idempotent."""
        evidence, payload, source, target = await self.proposal_fixture()
        model = ScriptedModelClient([ModelResponse(json.dumps(payload), model="small-model")])
        proposal = (await propose_links(evidence, "a", model, candidate_node_ids=["b"]))[0]
        with patch.object(
            self.workspace.blob_store, "get", wraps=self.workspace.blob_store.get
        ) as read_blob:
            entry = await accept_link(evidence, proposal, idempotency_key="accepted-link")
            retry = await accept_link(evidence, proposal, idempotency_key="accepted-link")
        self.assertEqual(read_blob.call_count, 2)
        self.assertEqual(entry, retry)
        self.assertEqual(entry.provenance.origin, "model")
        self.assertEqual(entry.provenance.model, "small-model")
        self.assertEqual(entry.provenance.supporting_references, (source, target))
        self.assertEqual(entry.provenance.prompt_version, "link-proposal-v6")
        self.assertEqual(await evidence.neighbors("a"), [NodeRef("b")])

    async def test_acceptance_is_independent_of_unrelated_journal_appends(self):
        """Ordinary link publication does not depend on another mechanism's append sequence."""
        evidence, payload, _, _ = await self.proposal_fixture()
        proposal = (
            await propose_links(
                evidence, "a", ScriptedModelClient([json.dumps(payload)]), candidate_node_ids=["b"]
            )
        )[0]
        note = await self.assertion("a", "concurrent update")
        accepted = await accept_link(evidence, proposal)
        self.assertEqual(accepted.source_node_id, "a")
        self.assertEqual(accepted.target_node_id, "b")
        self.assertEqual(await self.workspace.inspect_journal("a"), [note])
        self.assertEqual(await self.workspace.edges("a"), [accepted])

    async def test_proposal_rejects_unsupported_endpoints_invented_spans_and_partial_support(self):
        """Unsupported endpoints, fabricated spans and incomplete evidence invalidate proposals."""
        evidence, payload, source, _ = await self.proposal_fixture()
        malformed = []
        missing = json.loads(json.dumps(payload))
        missing["links"][0]["target_node_id"] = "missing"
        malformed.append(missing)
        invented = json.loads(json.dumps(payload))
        invented["links"][0]["supporting_references"][0]["end"] = source.end + 1
        malformed.append(invented)
        wrapped = json.loads(json.dumps(payload))
        wrapped["links"][0]["supporting_references"] = [
            {"reference": reference, "text": "An echoed passage", "metadata": {"role": "user"}}
            for reference in wrapped["links"][0]["supporting_references"]
        ]
        malformed.append(wrapped)
        partial = json.loads(json.dumps(payload))
        partial["links"][0]["supporting_references"] = [reference_to_dict(source)]
        malformed.append(partial)
        malformed.append({"links": "not-an-array"})
        malformed.append({"links": [], "execute": "unrequested-operation"})
        for data in malformed:
            with self.subTest(data=data), self.assertRaises(SchemaError):
                await propose_links(
                    evidence, "a", ScriptedModelClient([json.dumps(data)]), candidate_node_ids=["b"]
                )
        self.assertEqual(await self.workspace.inspect_journal("a"), [])

    async def test_no_candidates_and_explicit_abstention_publish_nothing(self):
        """Empty candidates and explicit model abstention never publish a link."""
        evidence, _, _, _ = await self.proposal_fixture()
        unused = ScriptedModelClient([])
        self.assertEqual(await propose_links(evidence, "a", unused, candidate_node_ids=[]), [])
        self.assertEqual(unused.requests, [])
        abstaining = ScriptedModelClient(['{"links": []}'])
        self.assertEqual(
            await propose_links(evidence, "a", abstaining, candidate_node_ids=["b"]), []
        )
        self.assertEqual(await self.workspace.inspect_journal("a"), [])

    async def test_manual_proposal_requires_node_endpoints_and_exact_support(self):
        """Manual proposals require node endpoints and exact support from both sources."""
        evidence, _, source, target = await self.proposal_fixture()
        proposal = LinkProposal(
            NodeRef("a"),
            NodeRef("b"),
            (source, target),
            "rationale",
            "model",
        )
        variants = (
            replace(proposal, source=source),
            replace(proposal, supporting_references=(source,)),
            replace(proposal, supporting_references=(NodeRef("a"), NodeRef("b"))),
        )
        for value in variants:
            with self.subTest(value=value), self.assertRaises(SchemaError):
                await accept_link(evidence, value)
        with self.assertRaises(ReferenceResolutionError):
            await accept_link(
                evidence,
                replace(
                    proposal, supporting_references=(replace(source, end=source.end + 1), target)
                ),
            )
        self.assertEqual(await self.workspace.edges("a"), [])
        await evidence.close()
        with self.assertRaises(ConfigurationError):
            await accept_link(evidence, proposal)

    async def test_proposal_support_cannot_silently_substitute_another_node(self):
        """Support from one immutable source cannot justify another endpoint."""
        _, _, source, target = await self.proposal_fixture()
        await self.source("c", "New conflicting source")
        evidence = await self.open()
        proposal = LinkProposal(
            NodeRef("a"),
            NodeRef("c"),
            (source, target),
            "rationale",
            "model",
        )
        with self.assertRaises(SchemaError):
            await accept_link(evidence, proposal)

    async def test_search_and_read_preserve_trusted_date_role_scope_and_provenance(self):
        """Canonical dates, speaker roles, scope and provenance survive read and search adapters."""
        text = "I moved there yesterday."
        await self.workspace.ingest(
            Conversation.from_turns(
                [{"role": "user", "text": text, "turn_id": "t"}],
                node_id="a",
                metadata={"session_date": "2025-02-03", "speaker": "user-1"},
                timestamp_ms=1738540800123,
            )
        )
        ref = SourceSpan("a", "t", 0, len(text))
        entry = await self.assertion(
            "a", "Yesterday applies to staging.", applicability={"scope": {"env": "staging"}}
        )
        evidence = await self.open()
        resolved = await evidence.read(ref)
        self.assertEqual(resolved.metadata["source_metadata"]["session_date"], "2025-02-03")
        self.assertEqual(resolved.metadata["role"], "user")
        self.assertEqual(resolved.metadata["timestamp_ms"], 1738540800123)
        source_hit = next(
            hit
            for hit in await evidence.search("yesterday", 10)
            if isinstance(hit.passage.refs[0], SourceSpan)
        )
        self.assertEqual(
            source_hit.passage.metadata["source_metadata"]["session_date"], "2025-02-03"
        )
        self.assertEqual(source_hit.passage.metadata["timestamp_ms"], 1738540800123)
        journal_hit = next(
            hit
            for hit in await evidence.search("yesterday", 10)
            if isinstance(hit.passage.refs[0], JournalRef)
        )
        self.assertEqual(
            journal_hit.passage.metadata["applicability"], {"scope": {"env": "staging"}}
        )
        self.assertEqual(journal_hit.passage.metadata["provenance"]["origin"], "user")
        journal_read = await evidence.read(JournalRef("a", entry.entry_id, 0, len(entry.value)))
        self.assertEqual(journal_read.metadata["applicability"], {"scope": {"env": "staging"}})
        self.assertEqual(journal_read.metadata["recorded_at_ms"], entry.recorded_at_ms)
        retriever = SQLiteBM25Retriever.from_passages(
            [SearchPassage("p", "Yesterday", (ref,), {"session_date": "invented date"})]
        )
        self.retrievers.append(retriever)
        injected = await self.open(retriever=retriever)
        retrieved = next(
            hit
            for hit in await injected.search("yesterday", 10)
            if isinstance(hit.passage.refs[0], SourceSpan)
        )
        self.assertEqual(
            retrieved.passage.metadata["evidence_metadata"][0]["source_metadata"]["session_date"],
            "2025-02-03",
        )
        self.assertNotIn("invented date", str(retrieved.passage.metadata))
        self.assertEqual(
            retrieved.passage.metadata["evidence_metadata"][0]["timestamp_ms"], 1738540800123
        )

    async def test_nonfinite_external_score_fails(self):
        """A nonfinite score from an injected retriever fails evidence validation."""
        ref = await self.source("a", "cobalt")

        class InvalidRetriever:
            """External-retrieval double that emits a deliberately invalid score."""

            async def search(self, query, k):
                """Return valid source evidence paired with a NaN score."""
                return [SearchHit(SearchPassage("p", "cobalt", (ref,)), float("nan"), 1)]

        evidence = await self.open(retriever=InvalidRetriever())
        with self.assertRaises(SchemaError):
            await evidence.search("cobalt", 1)


if __name__ == "__main__":
    unittest.main()
