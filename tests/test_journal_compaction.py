"""Complete operational journals preserve interpretation and stable archived evidence."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llgm.core.errors import BudgetExceeded
from llgm.core.types import Conversation, JournalRef, NodeRef, Provenance, SourceSpan
from llgm.memory.evidence import interpret_journal
from llgm.memory.workspace import Workspace, journal_to_dict


class JournalCompactionTests(unittest.IsolatedAsyncioTestCase):
    """Exact reconciliation reduces repeated writes without making current-only history assumptions."""

    async def asyncSetUp(self):
        """Open one immutable source whose archived and operational records can be compared."""
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.workspace = await Workspace.open(self.path).__aenter__()
        await self.workspace.ingest(
            Conversation.from_turns(
                [{"role": "user", "turn_id": "t", "text": "production staging"}], node_id="a"
            )
        )
        self.provenance = Provenance("user", "compaction-contract")

    async def asyncTearDown(self):
        """Close real storage before releasing its temporary directory."""
        await self.workspace.close()
        self.temporary.cleanup()

    async def append(self, value, **kwargs):
        """Append to one exact source slot with independently supplied applicability."""
        return await self.workspace.append_journal(
            "a",
            subject=SourceSpan("a", "t", 0, 10),
            relation="value",
            value=value,
            provenance=self.provenance,
            **kwargs,
        )

    async def test_repeated_updates_compact_and_original_references_retry_reopen_survive(self):
        """Operational reconciliation archives superseded entries while exact references and retries survive."""
        first = await self.append("initial", idempotency_key="first", expected_sequence=0)
        for index in range(20):
            latest = await self.append(f"replacement {index}", record_kind="overwrite")
        later_assertion = await self.append("independent later assertion")
        operational = await self.workspace.operational_journal("a")
        self.assertEqual(operational, [latest, later_assertion])
        stats = await self.workspace.compact_journal("a")
        self.assertEqual(
            stats, {"history_entries": 22, "operational_entries": 2, "archived_entries": 20}
        )
        self.assertEqual(
            await self.append("initial", idempotency_key="first", expected_sequence=0), first
        )
        await self.workspace.close()
        self.workspace = await Workspace.open(self.path).__aenter__()
        self.assertEqual(await self.workspace.operational_journal("a"), operational)
        self.assertEqual(
            (await self.workspace.resolve(JournalRef("a", first.entry_id, 0, 7))).text, "initial"
        )
        self.assertEqual(
            (await self.workspace.resolve(SourceSpan("a", "t", 0, 10))).text, "production"
        )

    async def test_distinct_scopes_and_validity_intervals_keep_historical_and_future_meaning(self):
        """Timed and scoped differences remain available rather than being reduced to today's winner."""
        entries = []
        for scope, start, end, text in (
            ("production", 0, 100, "old"),
            ("production", 100, 200, "middle"),
            ("production", 200, 300, "future"),
            ("staging", 0, 300, "staging"),
        ):
            entries.append(
                await self.append(
                    text,
                    record_kind="overwrite",
                    applicability={
                        "scope": {"env": scope},
                        "valid_from_ms": start,
                        "valid_until_ms": end,
                    },
                )
            )
        await self.workspace.compact_journal("a")
        operational = await self.workspace.operational_journal("a")
        self.assertEqual(operational, entries)
        for scope, instant in (
            ("production", 50),
            ("production", 150),
            ("production", 250),
            ("staging", 250),
        ):
            full = interpret_journal(
                await self.workspace.inspect_journal("a"), scope={"env": scope}, as_of_ms=instant
            )
            compact = interpret_journal(operational, scope={"env": scope}, as_of_ms=instant)
            self.assertEqual(full.active, compact.active)

    async def test_later_correction_restores_archived_predecessors(self):
        """Correcting an archived chain restores every record needed to interpret its dependencies."""
        first = await self.append("original interpretation")
        second = await self.append("overwritten interpretation", record_kind="overwrite")
        self.assertEqual(await self.workspace.operational_journal("a"), [second])
        correction = await self.workspace.append_journal(
            "a",
            subject=JournalRef("a", second.entry_id),
            relation="retract",
            value="withdraw mistaken update",
            provenance=self.provenance,
            record_kind="correction",
        )
        operational = await self.workspace.operational_journal("a")
        self.assertEqual(operational, [first, second, correction])
        full = interpret_journal(await self.workspace.inspect_journal("a"))
        self.assertEqual(full, interpret_journal(operational))
        self.assertIn(first, full.active)
        await self.workspace.compact_journal("a")
        self.assertEqual(await self.workspace.operational_journal("a"), operational)

    async def test_unknown_rules_and_suggestions_are_not_discarded(self):
        """Unresolved and proposed records stay complete rather than being treated as safe replacements."""
        for value in ("first", "second"):
            await self.append(
                value, record_kind="overwrite", applicability={"unknown_selector": "x"}
            )
            await self.workspace.append_journal(
                "a",
                subject=NodeRef("a"),
                relation="suggests",
                value=value,
                record_kind="overwrite",
                provenance=self.provenance,
            )
        self.assertEqual(len(await self.workspace.operational_journal("a")), 4)

    async def test_size_bound_counts_complete_utf8_and_overflow_never_truncates(self):
        """Irreducibly large journals raise before returning a prefix or committing partial compaction."""
        entry = await self.append("한글 " * 30)
        encoded = json.dumps(
            [journal_to_dict(entry)], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        self.assertEqual(
            await self.workspace.operational_journal("a", max_bytes=len(encoded)), [entry]
        )
        with self.assertRaises(BudgetExceeded):
            await self.workspace.operational_journal("a", max_bytes=len(encoded) - 1)
        before = self.workspace._connection.execute(
            "SELECT entry_id FROM _journal_operational"
        ).fetchall()
        with self.assertRaises(BudgetExceeded):
            await self.workspace.compact_journal("a", max_bytes=2)
        self.assertEqual(
            self.workspace._connection.execute(
                "SELECT entry_id FROM _journal_operational"
            ).fetchall(),
            before,
        )
        self.assertEqual(await self.workspace.inspect_journal("a"), [entry])

    async def test_concurrent_append_and_compaction_never_drop_the_tail(self):
        """SQLite publication keeps the latest append when another connection compacts concurrently."""
        await self.append("baseline")
        async with Workspace.open(self.path) as other:
            _, latest = await asyncio.gather(
                self.workspace.compact_journal("a"),
                other.append_journal(
                    "a",
                    subject=SourceSpan("a", "t", 0, 10),
                    relation="value",
                    value="concurrent replacement",
                    record_kind="overwrite",
                    provenance=self.provenance,
                ),
            )
        self.assertEqual(await self.workspace.operational_journal("a"), [latest])
        self.assertEqual(len(await self.workspace.inspect_journal("a")), 2)

    async def test_reconciliation_failure_rolls_back_append_history_and_projection(self):
        """An interrupted operational publication leaves no half-written journal append or index event."""
        first = await self.append("baseline")
        changes = self.workspace._connection.execute(
            "SELECT COUNT(*) FROM _index_changes"
        ).fetchone()[0]
        with patch.object(
            self.workspace,
            "_reconcile_journal",
            side_effect=OSError("injected reconciliation failure"),
        ):
            with self.assertRaises(OSError):
                await self.append("unpublished", record_kind="overwrite")
        self.assertEqual(await self.workspace.inspect_journal("a"), [first])
        self.assertEqual(await self.workspace.operational_journal("a"), [first])
        self.assertEqual(
            self.workspace._connection.execute("SELECT COUNT(*) FROM _index_changes").fetchone()[0],
            changes,
        )
