"""Canonical lazy reads with eagerly loaded amendments and independent primary edges."""

import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta, timezone
from unittest.mock import patch

from llgm import Conversation, JournalRef, NodeRef, Provenance, SourceSpan, Workspace
from llgm.core.errors import (
    BudgetExceeded,
    ConfigurationError,
    ReferenceResolutionError,
    SchemaError,
)
from llgm.core.time import legacy_iso_to_ms, normalize_applicability, parse_instant_ms
from llgm.memory.evidence import Evidence, interpret_journal
from llgm.memory.query import QueryEvidence


class EffectiveReadTests(unittest.IsolatedAsyncioTestCase):
    """Use real local publication and deterministic replacement plans without a model."""

    async def asyncSetUp(self):
        """Create one workspace and query handle with no external dependencies."""
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = await Workspace.open(self.temp.name).__aenter__()
        self.evidence = await Evidence.open(self.workspace)
        self.query = QueryEvidence(self.evidence, {}, None)

    async def asyncTearDown(self):
        """Close local handles before removing temporary evidence."""
        await self.evidence.close()
        await self.workspace.close()
        self.temp.cleanup()

    async def source(self, node, text):
        """Publish exact text and return its full-turn canonical span."""
        await self.workspace.ingest(
            Conversation.from_turns([{"role": "user", "turn_id": "t", "text": text}], node_id=node)
        )
        return SourceSpan(node, "t", 0, len(text))

    async def amend(
        self, subject, target, *, applicability=None, relation="replace", kind="overwrite"
    ):
        """Append an explicit scoped source patch with supporting provenance."""
        support = (target,) if isinstance(target, SourceSpan) else ()
        return await self.workspace.append_journal(
            subject.node_id,
            subject=subject,
            record_kind=kind,
            relation=relation,
            value=target,
            applicability=applicability,
            provenance=Provenance("user", "effective-read-test", support),
        )

    async def test_partial_patch_returns_canonical_replacement_and_untouched_source(self):
        """Production changes without attributing replacement text to old offsets or changing staging."""
        old = await self.source("old", "blue | green")
        target = await self.source("new", "red")
        subject = replace(old, end=4)
        amendment = await self.amend(subject, target)
        segments = await self.query.read_segments(old)
        self.assertEqual(
            [(part.text, part.reference) for part in segments],
            [("red", target), (" | green", replace(old, start=4))],
        )
        self.assertEqual(segments[0].metadata["amendments"][0]["entry_id"], amendment.entry_id)
        self.assertEqual(segments[0].metadata["amended_from"]["end"], 4)
        self.assertEqual((await self.workspace.resolve(old)).text, "blue | green")
        self.assertEqual(
            "".join(part.text for part in await self.query.read_segments(NodeRef("old"))),
            "red | green",
        )
        with self.assertRaises(ConfigurationError):
            await self.query.read(old)

    async def test_source_info_pages_handles_without_exposing_large_source_text(self):
        """A delegate can discover Unicode coordinates and speaker roles before selecting text."""
        await self.workspace.ingest(
            Conversation.from_turns(
                [
                    {"turn_id": "first", "role": "user", "text": "雪" * 5000},
                    {"turn_id": "second", "role": "assistant", "text": "secret second text"},
                ],
                node_id="large",
            )
        )
        first = await self.query.source_info("large", limit=1)
        self.assertEqual(first["total_turns"], 2)
        self.assertEqual(first["next_offset"], 1)
        self.assertEqual(
            first["turns"][0]["reference"],
            {
                "type": "source_span",
                "node_id": "large",
                "turn_id": "first",
                "start": 0,
                "end": 5000,
            },
        )
        self.assertNotIn("雪", str(first))
        second = await self.query.source_info("large", offset=1, limit=1)
        self.assertIsNone(second["next_offset"])
        self.assertEqual(second["turns"][0]["role"], "assistant")
        self.assertNotIn("secret", str(second))
        for kwargs in ({"offset": -1}, {"offset": True}, {"limit": 0}, {"limit": 129}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                await self.query.source_info("large", **kwargs)

    async def test_unchanged_fragments_reuse_the_validated_original_read(self):
        """One partial read preserves Unicode offsets and attribution without refetching each unchanged fragment."""
        old = await self.source("old", "xx 雪blue🍃 yy")
        target = await self.source("new", "red")
        await self.amend(replace(old, start=4, end=8), target)
        requested = replace(old, start=3, end=9)
        with patch.object(
            self.workspace.blob_store, "get", wraps=self.workspace.blob_store.get
        ) as read_blob:
            parts = await self.query.read_segments(requested)
        self.assertEqual(read_blob.call_count, 2)
        self.assertEqual(
            [(part.text, part.reference) for part in parts],
            [
                ("雪", replace(old, start=3, end=4)),
                ("red", target),
                ("🍃", replace(old, start=8, end=9)),
            ],
        )
        self.assertTrue(all(part.metadata["role"] == "user" for part in parts))
        for part in (parts[0], parts[2]):
            self.assertEqual(part.text, (await self.workspace.resolve(part.reference)).text)

    async def test_read_inside_old_patch_returns_complete_exact_replacement(self):
        """Changed lengths never cause guessed offset mappings into replacement text."""
        old = await self.source("old", "blue")
        target = await self.source("new", "a longer replacement")
        await self.amend(old, target)
        parts = await self.query.read_segments(replace(old, start=1, end=2))
        self.assertEqual(
            [(part.text, part.reference) for part in parts], [("a longer replacement", target)]
        )

    async def test_eager_initialization_contains_all_notes_and_no_source_text(self):
        """Full local notes are available before any large-source read or model decision."""
        source = await self.source("old", "large original source")
        note = await self.amend(
            source, "Final note outside any prefix", relation="note", kind="assertion"
        )
        with (
            patch.object(
                self.evidence, "source", side_effect=AssertionError("source must remain lazy")
            ),
            patch.object(
                self.evidence, "read", side_effect=AssertionError("source must remain lazy")
            ),
        ):
            state = await self.query.initialize_node("old")
        self.assertEqual(state["journal"][0]["value"], "Final note outside any prefix")
        self.assertEqual(state["journal"][0]["entry_id"], note.entry_id)
        self.assertNotIn("large original source", str(state))
        self.assertEqual((await self.query.read_segments(source))[0].text, "large original source")

    async def test_latest_scoped_patch_and_half_open_numeric_validity(self):
        """Numeric time and exact scope choose applicable patches without changing unrelated contexts."""
        old = await self.source("old", "blue")
        target = await self.source("new", "red")
        await self.amend(
            old,
            target,
            applicability={"scope": {"env": "prod"}, "valid_from_ms": 100, "valid_until_ms": 200},
        )
        for instant, env, expected in (
            (99, "prod", "blue"),
            (100, "prod", "red"),
            (199, "prod", "red"),
            (200, "prod", "blue"),
            (100, "stage", "blue"),
        ):
            with self.subTest(instant=instant, env=env):
                query = QueryEvidence(self.evidence, {"env": env}, None, as_of_ms=instant)
                self.assertEqual((await query.read_segments(old))[0].text, expected)
        unknown = QueryEvidence(self.evidence, {"env": "prod"}, "some day")
        blocked = (await unknown.read_segments(old))[0]
        self.assertEqual(blocked.text, "")
        self.assertTrue(blocked.metadata["unresolved"])

    async def test_unknown_patch_scope_or_selector_never_exposes_stale_source(self):
        """Ambiguous applicability blocks its affected source passage instead of falling back."""
        source = await self.source("old", "blue")
        target = await self.source("new", "red")
        await self.amend(source, target, applicability={"scope": {"env": "prod"}})
        blocked = await self.query.read_segments(source)
        self.assertEqual(blocked[0].text, "")
        self.assertIn("scope missing", blocked[0].metadata["unresolved"][0])

    async def test_unavailable_replacement_never_falls_back_to_original(self):
        """A storage read failure remains explicit when the current patch target cannot be fetched."""
        source = await self.source("old", "blue")
        target = await self.source("new", "red")
        await self.amend(source, target)
        original_read = self.evidence.read

        async def unavailable(reference):
            """Simulate a temporarily missing replacement blob after valid publication."""
            if reference == target:
                raise ReferenceResolutionError("replacement blob unavailable")
            return await original_read(reference)

        with patch.object(self.evidence, "read", side_effect=unavailable):
            parts = await self.query.read_segments(source)
        self.assertEqual(parts[0].text, "")
        self.assertIn("Replacement unavailable", parts[0].metadata["unresolved"][0])

    async def test_replacement_chains_and_cycles_have_explicit_bounded_outcomes(self):
        """Transitive patches resolve exact targets and stop cycles without stale source leakage."""
        a, b, c = [await self.source(name, name) for name in "abc"]
        await self.amend(a, b)
        await self.amend(b, c)
        parts = await self.query.read_segments(a)
        self.assertEqual([(part.text, part.reference) for part in parts], [("c", c)])
        self.assertEqual(len(parts[0].metadata["amendments"]), 2)
        await self.amend(c, a)
        parts = await self.query.read_segments(a)
        self.assertEqual(parts[0].text, "")
        self.assertIn("cycle", parts[0].metadata["unresolved"][0])

    async def test_overlapping_patches_block_full_replacement_not_only_shared_offsets(self):
        """Conflicting coordinates cannot leak a complete replacement through an unaffected prefix."""
        source = await self.source("old", "abcdef unchanged")
        b, c = [await self.source(name, name) for name in "bc"]
        await self.amend(replace(source, end=4), b)
        await self.amend(replace(source, start=2, end=6), c)
        parts = await self.query.read_segments(source)
        self.assertEqual("".join(part.text for part in parts), " unchanged")
        self.assertTrue(all(part.metadata["unresolved"] for part in parts if not part.text))

    async def test_primary_neighbors_ignore_journal_pointers_and_patch_scope(self):
        """Primary connectivity works with empty journals and survives unrelated amendments."""
        a, b, c = [await self.source(name, name) for name in "abc"]
        await self.workspace.publish_edge(
            "a", "b", relation="related_to", provenance=Provenance("user", "test")
        )
        self.assertEqual(await self.workspace.inspect_journal("a"), [])
        self.assertEqual(await self.query.neighbors("a"), [NodeRef("b")])
        await self.amend(a, c)
        self.assertEqual(await self.query.neighbors("a"), [NodeRef("b")])
        descriptions = await self.query.edge_descriptions("a")
        self.assertEqual(len(descriptions), 1)
        self.assertEqual(descriptions[0]["reference"], {"type": "node", "node_id": "b"})
        self.assertEqual(descriptions[0]["relation"], "related_to")
        self.assertEqual((await self.query.read_segments(a))[0].reference, c)

    async def test_edge_applicability_filters_primary_relationship(self):
        """Primary edges have their own explicit scope and time selectors."""
        for name in "ab":
            await self.source(name, name)
        edge = await self.workspace.publish_edge(
            "a",
            "b",
            relation="related_to",
            provenance=Provenance("user", "test"),
            applicability={"scope": {"env": "prod"}, "valid_from_ms": 100, "valid_until_ms": 200},
        )
        raw = await self.evidence.edge_descriptions("a")
        self.assertEqual(raw[0]["edge_id"], edge.edge_id)
        cases = (
            ({}, 100, False),
            ({"env": "stage"}, None, False),
            ({"env": "prod"}, None, False),
            ({"env": "prod"}, 99, False),
            ({"env": "prod"}, 100, True),
            ({"env": "prod"}, 199, True),
            ({"env": "prod"}, 200, False),
        )
        for scope, instant, active in cases:
            with self.subTest(scope=scope, instant=instant):
                query = QueryEvidence(self.evidence, scope, None, as_of_ms=instant)
                self.assertEqual(await query.neighbors("a"), [NodeRef("b")] if active else [])
                with patch.object(
                    self.workspace.blob_store,
                    "get",
                    side_effect=AssertionError("Edge descriptions must not read source text"),
                ):
                    self.assertEqual(await query.edge_descriptions("a"), raw if active else [])
                self.assertEqual(await query.edge_descriptions("a", "different_relation"), [])

    async def test_edge_descriptions_keep_applicable_relations_separate_and_hide_withdrawals(self):
        """One target can have several applicable edges while unresolved scope and withdrawal stay excluded."""
        for node in "ab":
            await self.source(node, node)
        edges = []
        for relation, applicability in (
            ("credential", {}),
            ("supported_by", {"scope": {"env": "prod"}}),
            ("incident_contact", {"scope": {"env": "stage"}}),
            ("conditional", {"unrecognized_rule": "unknown"}),
        ):
            edges.append(
                await self.workspace.publish_edge(
                    "a",
                    "b",
                    relation=relation,
                    provenance=Provenance("user", "test", (NodeRef("a"),)),
                    applicability=applicability,
                )
            )
        query = QueryEvidence(self.evidence, {"env": "prod"}, None)
        descriptions = await query.edge_descriptions("a")
        self.assertEqual(
            {record["edge_id"] for record in descriptions}, {edge.edge_id for edge in edges[:2]}
        )
        self.assertEqual(
            {record["relation"] for record in descriptions}, {"credential", "supported_by"}
        )
        self.assertEqual(await query.neighbors("a"), [NodeRef("b")])
        await self.workspace.withdraw_edge(edges[0].edge_id, provenance=Provenance("user", "test"))
        remaining = await query.edge_descriptions("a")
        self.assertEqual([record["edge_id"] for record in remaining], [edges[1].edge_id])
        self.assertEqual(remaining[0]["applicability"], {"scope": {"env": "prod"}})
        self.assertEqual(
            remaining[0]["provenance"]["supporting_references"], [{"type": "node", "node_id": "a"}]
        )

    async def test_retrieved_old_passage_keeps_owner_but_exposes_effective_citations(self):
        """Seed identity survives replacement so the old node's amendment remains inspectable."""
        old = await self.source("old", "distinctive blue")
        target = await self.source("new", "red")
        await self.amend(old, target)
        hit = (await self.query.search("distinctive", 5))[0]
        self.assertEqual(hit.passage.metadata["owner_node_id"], "old")
        self.assertEqual(hit.passage.refs, (target,))
        self.assertEqual(hit.passage.text, "red")
        self.assertEqual(hit.passage.metadata["segments"][0]["reference"]["node_id"], "new")

    async def test_compacted_operational_state_keeps_history_and_equivalent_reads(self):
        """Repeated same-slot replacements compact without deleting old canonical journal evidence."""
        source = await self.source("old", "blue")
        for number in range(12):
            await self.amend(source, f"replacement {number}")
        state = await self.query.initialize_node("old")
        self.assertEqual(len(state["journal"]), 1)
        self.assertEqual(len(await self.workspace.inspect_journal("old")), 12)
        part = (await self.query.read_segments(source))[0]
        self.assertEqual(part.text, "replacement 11")
        self.assertEqual((await self.workspace.resolve(part.reference)).text, part.text)

    async def test_distinct_operational_notes_overflow_without_truncation(self):
        """Irreducible local state exceeding its finite bound cannot become a partial journal."""
        source = await self.source("old", "blue")
        for number in range(5):
            await self.amend(source, str(number) + "x" * 250, relation="note", kind="assertion")
        query = QueryEvidence(self.evidence, {}, None, max_journal_bytes=1000)
        with self.assertRaises(BudgetExceeded):
            await query.initialize_node("old")
        with self.assertRaises(BudgetExceeded):
            await query.read_segments(source)
        self.assertEqual(len(await self.workspace.inspect_journal("old")), 5)

    async def test_total_journal_cache_is_bounded_and_returns_isolated_payloads(self):
        """Many distinct nodes cannot grow the local journal cache beyond its declared allowance."""
        for name in "ab":
            await self.source(name, name)
        state = await self.query.initialize_node("a")
        state["journal"].append({"invented": True})
        self.assertEqual((await self.query.initialize_node("a"))["journal"], [])
        query = QueryEvidence(
            self.evidence, {}, None, max_total_journal_bytes=state["journal_bytes"] + 1
        )
        await query.initialize_node("a")
        with self.assertRaises(BudgetExceeded):
            await query.initialize_node("b")

    async def test_explicit_retraction_restores_prior_effective_patch(self):
        """A correction can restore an archived predecessor without source mutation."""
        source = await self.source("old", "original")
        first = await self.amend(source, "first")
        latest = await self.amend(source, "latest")
        await self.workspace.append_journal(
            "old",
            subject=JournalRef("old", latest.entry_id),
            record_kind="correction",
            relation="retract",
            value="unsupported",
            provenance=Provenance("user", "test"),
        )
        part = (await self.query.read_segments(source))[0]
        self.assertEqual(part.text, "first")
        self.assertEqual(part.reference.entry_id, first.entry_id)


class TimeContractTests(unittest.TestCase):
    """Numeric instants preserve declared offsets and reject invented precision."""

    def test_equivalent_timezone_inputs_and_negative_milliseconds(self):
        """Explicit offsets identify one instant without floating-point rounding."""
        self.assertEqual(
            parse_instant_ms("2026-09-10T09:00:00-04:00"), parse_instant_ms("2026-09-10T13:00:00Z")
        )
        self.assertEqual(parse_instant_ms("1969-12-31T23:59:59.999Z"), -1)

    def test_date_only_requires_explicit_conversion_rule(self):
        """A source date alone does not silently acquire an exact timestamp."""
        with self.assertRaises(SchemaError):
            parse_instant_ms("2026-09-10")
        self.assertEqual(
            parse_instant_ms("2026-09-10", date_only_timezone=timezone(timedelta(hours=-4))),
            parse_instant_ms("2026-09-10T04:00:00Z"),
        )
        self.assertEqual(legacy_iso_to_ms("2026-09-10"), parse_instant_ms("2026-09-10T00:00:00Z"))

    def test_machine_bounds_are_strict_and_explicit_legacy_conversion_matches(self):
        """New fields use integer milliseconds and old bound keys convert only at the boundary."""
        instant = parse_instant_ms("2026-09-10T00:00:00Z")
        self.assertEqual(
            normalize_applicability({"valid_from": "2026-09-10"}), {"valid_from_ms": instant}
        )
        for payload in (
            {"valid_from_ms": True},
            {"valid_from_ms": "123"},
            {"valid_from_ms": None},
            {"valid_from": "2026-09-10", "valid_from_ms": instant},
            {"valid_from_ms": 2, "valid_until_ms": 2},
        ):
            with self.subTest(payload=payload), self.assertRaises(SchemaError):
                normalize_applicability(payload)
        for value in ("2026-09-10T00:00:00", "2026-09-10T00:00:00.000001Z"):
            with self.subTest(value=value), self.assertRaises(SchemaError):
                parse_instant_ms(value)
        with self.assertRaises(SchemaError):
            interpret_journal([], as_of_ms=True)
