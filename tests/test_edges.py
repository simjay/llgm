"""Independent primary-edge publication, withdrawal and retry contracts."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llgm.core.errors import ConflictError, ReferenceResolutionError, SchemaError
from llgm.core.types import Conversation, NodeRef, Provenance, SourceSpan
from llgm.memory.workspace import Workspace


class EdgeTests(unittest.IsolatedAsyncioTestCase):
    """Primary relationships exist and retain provenance independently of semantic journals."""

    async def asyncSetUp(self):
        """Publish two immutable endpoints in an isolated real SQLite workspace."""
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.workspace = await Workspace.open(self.path).__aenter__()
        for node in ("a", "b", "c"):
            await self.workspace.ingest(
                Conversation.from_turns(
                    [{"role": "user", "turn_id": "t", "text": node + " evidence"}], node_id=node
                )
            )
        self.provenance = Provenance("user", "edge-contract", (SourceSpan("a", "t", 0, 1),))

    async def asyncTearDown(self):
        """Close all storage before removing the isolated workspace."""
        await self.workspace.close()
        self.temporary.cleanup()

    async def publish(self, **kwargs):
        """Create the standard directed relationship with caller-selected retry conditions."""
        return await self.workspace.publish_edge("a", "b", provenance=self.provenance, **kwargs)

    async def test_independent_directed_edges_and_reopen(self):
        """Primary adjacency needs no journal and retains exact provenance across reopen."""
        edge = await self.publish()
        self.assertEqual(len(edge.edge_id), 32)
        self.assertEqual(edge.provenance, self.provenance)
        self.assertEqual(await self.workspace.edges("a"), [edge])
        self.assertEqual(await self.workspace.edges("b"), [])
        self.assertFalse(hasattr(edge, "relation"))
        self.assertEqual(await self.workspace.inspect_journal("a"), [])
        self.assertEqual(await self.workspace.operational_journal("a"), [])
        self.assertEqual(
            self.workspace._connection.execute("SELECT COUNT(*) FROM _index_changes").fetchone()[0],
            3,
        )
        await self.workspace.close()
        self.workspace = await Workspace.open(self.path).__aenter__()
        self.assertEqual(await self.workspace.edge(edge.edge_id), edge)
        self.assertEqual(await self.workspace.edges("a"), [edge])

    async def test_concurrent_duplicates_and_retry_conflicts(self):
        """Concurrent publication deduplicates active identity and checks retry content independently."""
        async with Workspace.open(self.path) as other:
            results = await asyncio.gather(
                self.publish(idempotency_key="left"),
                other.publish_edge(
                    "a",
                    "b",
                    provenance=self.provenance,
                    idempotency_key="right",
                ),
            )
        self.assertEqual(results[0], results[1])
        self.assertEqual(await self.publish(idempotency_key="left"), results[0])
        with self.assertRaises(ConflictError):
            await self.workspace.publish_edge(
                "a", "c", provenance=self.provenance, idempotency_key="left"
            )
        self.assertEqual(len(await self.workspace.edges("a")), 1)

    async def test_withdrawal_preserves_record_and_republication_uses_new_id(self):
        """Withdrawal targets one relationship and never erases history or mutates source text."""
        original = await self.publish(idempotency_key="create")
        reason = Provenance("user", "reviewer", (NodeRef("a"),))
        withdrawn = await self.workspace.withdraw_edge(
            original.edge_id, provenance=reason, idempotency_key="remove"
        )
        self.assertIsNotNone(withdrawn.withdrawn_at_ms)
        self.assertEqual(withdrawn.withdrawal_provenance, reason)
        self.assertEqual(withdrawn.provenance, original.provenance)
        self.assertEqual(await self.workspace.edges("a"), [])
        self.assertEqual(await self.workspace.edges("a", include_withdrawn=True), [withdrawn])
        self.assertEqual(
            await self.workspace.withdraw_edge(
                original.edge_id, provenance=reason, idempotency_key="remove"
            ),
            withdrawn,
        )
        with self.assertRaises(ConflictError):
            await self.workspace.withdraw_edge(
                original.edge_id, provenance=self.provenance, idempotency_key="remove"
            )
        # Creation retries retain the original publication receipt, while edge()
        # always exposes its current withdrawal state.
        self.assertEqual(await self.publish(idempotency_key="create"), original)
        self.assertEqual(await self.workspace.edge(original.edge_id), withdrawn)
        replacement = await self.publish()
        self.assertNotEqual(replacement.edge_id, original.edge_id)
        self.assertEqual((await self.workspace.resolve(SourceSpan("a", "t", 0, 1))).text, "a")
        self.assertEqual(await self.workspace.inspect_journal("a"), [])

    async def test_scope_identity_and_caller_mapping_independence(self):
        """Different declared applicability stays distinct and returned mappings cannot change stored edges."""
        applicability = {"scope": {"env": "production"}, "valid_from": "2026-01-01"}
        production = await self.publish(applicability=applicability)
        staging = await self.publish(applicability={"scope": {"env": "staging"}})
        self.assertNotEqual(production.edge_id, staging.edge_id)
        self.assertIn("valid_from_ms", production.applicability)
        applicability["scope"]["env"] = "changed"
        production.applicability["scope"]["env"] = "another change"
        self.assertEqual(
            (await self.workspace.edge(production.edge_id)).applicability["scope"],
            {"env": "production"},
        )

    async def test_invalid_endpoints_support_and_atomic_failure(self):
        """Invalid references or a transaction failure cannot leave a partially published edge."""
        with self.assertRaises(SchemaError):
            await self.workspace.publish_edge("a", "a", provenance=self.provenance)
        with self.assertRaises(ReferenceResolutionError):
            await self.workspace.publish_edge("a", "missing", provenance=self.provenance)
        with self.assertRaises(ReferenceResolutionError):
            await self.workspace.publish_edge(
                "a",
                "b",
                provenance=Provenance("user", "invalid", (SourceSpan("a", "t", 0, 999),)),
            )
        with patch.object(
            self.workspace, "_record_retry", side_effect=OSError("injected commit failure")
        ):
            with self.assertRaises(OSError):
                await self.publish(idempotency_key="retryable")
        self.assertEqual(await self.workspace.edges("a"), [])
        self.assertEqual((await self.publish(idempotency_key="retryable")).source_node_id, "a")
        with self.assertRaises(ReferenceResolutionError):
            await self.workspace.withdraw_edge("missing", provenance=self.provenance)
