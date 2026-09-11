"""Explicit schema conversion preserves historical evidence and refuses guessed topology."""

import asyncio
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from llgm.core.errors import CapabilityError, ConfigurationError, ReferenceResolutionError
from llgm.core.types import Conversation, JournalRef, NodeRef, Provenance, SourceSpan
from llgm.memory.migration import copy_schema2_workspace
from llgm.memory.workspace import Workspace
from llgm.storage import LocalBlobStore


class StorageMigrationTests(unittest.IsolatedAsyncioTestCase):
    """A narrowly supported schema-2 copy never overwrites the original workspace."""

    async def asyncSetUp(self):
        """Construct the exact earlier tables and stable evidence IDs without old package imports."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "original"
        self.destination = self.root / "converted"
        async with Workspace.open(self.source) as workspace:
            self.conversation = Conversation.from_turns(
                [{"role": "user", "turn_id": "t", "text": "old production; unchanged staging"}],
                node_id="a",
            )
            await workspace.ingest(self.conversation, idempotency_key="source-retry")
            await workspace.ingest(
                Conversation.from_turns(
                    [{"role": "user", "turn_id": "t", "text": "updated production"}], node_id="b"
                )
            )
            self.link = await workspace.append_journal(
                "a",
                subject=NodeRef("a"),
                relation="related_to",
                value=NodeRef("b"),
                provenance=Provenance("model", "legacy-proposal", (SourceSpan("a", "t", 0, 3),)),
            )
            self.patch = await workspace.append_journal(
                "a",
                subject=SourceSpan("a", "t", 0, 14),
                relation="value",
                record_kind="overwrite",
                value=SourceSpan("b", "t", 0, 18),
                provenance=Provenance("user", "legacy-amendment"),
                idempotency_key="patch-retry",
            )
            self.history = await workspace.inspect_journal("a")
            self.raw_link = (await workspace.resolve(JournalRef("a", self.link.entry_id))).text
        with closing(sqlite3.connect(self.source / "metadata.sqlite3")) as connection:
            with connection:
                for table in ("_journal_classification", "_journal_operational", "edges"):
                    connection.execute("DROP TABLE " + table)
                connection.execute("PRAGMA user_version=2")
        self.original_bytes = (self.source / "metadata.sqlite3").read_bytes()

    async def asyncTearDown(self):
        """Remove only the isolated fixture directories after each conversion attempt."""
        self.temporary.cleanup()

    def roles(self):
        """Declare the fixture author's explicit distinction between a relationship and an amendment."""
        return {self.link.entry_id: "edge", self.patch.entry_id: "amendment"}

    async def test_copy_preserves_ids_raw_history_source_bytes_and_retry_records(self):
        """Reviewed topology becomes independent while original journal references and retry receipts survive."""
        report = await copy_schema2_workspace(
            self.source, self.destination, journal_roles=self.roles()
        )
        self.assertEqual(report["workspace_schema"], 3)
        self.assertTrue(report["history_preserved"])
        self.assertEqual((self.source / "metadata.sqlite3").read_bytes(), self.original_bytes)
        self.assertFalse(Path(str(self.destination / "metadata.sqlite3") + ".indexes").exists())
        async with Workspace.open(self.destination) as copied:
            self.assertEqual(await copied.source_ids(), ["a", "b"])
            self.assertEqual(await copied.inspect_journal("a"), self.history)
            self.assertEqual(
                (await copied.resolve(JournalRef("a", self.link.entry_id))).text, self.raw_link
            )
            self.assertEqual(
                (await copied.resolve(SourceSpan("a", "t", 16, 33))).text, "unchanged staging"
            )
            self.assertEqual(await copied.operational_journal("a"), [self.patch])
            edges = await copied.edges("a")
            self.assertEqual(len(edges), 1)
            self.assertEqual((edges[0].source_node_id, edges[0].target_node_id), ("a", "b"))
            self.assertEqual(edges[0].provenance, self.link.provenance)
            self.assertFalse(
                (await copied.ingest(self.conversation, idempotency_key="source-retry")).created
            )
            retry = await copied.append_journal(
                "a",
                subject=SourceSpan("a", "t", 0, 14),
                relation="value",
                record_kind="overwrite",
                value=SourceSpan("b", "t", 0, 18),
                provenance=Provenance("user", "legacy-amendment"),
                idempotency_key="patch-retry",
            )
            self.assertEqual(retry, self.patch)
        manifest = json.loads((self.destination / "conversion.json").read_text())
        self.assertEqual(manifest, report)
        with self.assertRaises(CapabilityError):
            async with Workspace.open(self.source):
                pass
        self.assertEqual((self.source / "metadata.sqlite3").read_bytes(), self.original_bytes)

    async def test_missing_or_invalid_classification_never_publishes_a_destination(self):
        """Pointer shape never silently decides whether old journal evidence is a primary edge."""
        for mapping in (
            {},
            {**self.roles(), "unknown": "edge"},
            {self.link.entry_id: "edge", self.patch.entry_id: "edge"},
        ):
            with self.subTest(mapping=mapping), self.assertRaises(ConfigurationError):
                await copy_schema2_workspace(self.source, self.destination, journal_roles=mapping)
            self.assertFalse(self.destination.exists())
            self.assertEqual((self.source / "metadata.sqlite3").read_bytes(), self.original_bytes)

    async def test_readonly_backup_copies_committed_wal_without_checkpointing_source(self):
        """A live writer's committed WAL data enters the copy while its source files remain unchanged."""
        with closing(
            sqlite3.connect(self.source / "metadata.sqlite3", isolation_level=None)
        ) as writer:
            writer.execute(
                "INSERT INTO sources SELECT 'c',blob_digest FROM sources WHERE node_id='a'"
            )
            writer.execute("INSERT INTO _index_changes(kind,record_id) VALUES ('source','c')")
            main_before = (self.source / "metadata.sqlite3").read_bytes()
            wal = self.source / "metadata.sqlite3-wal"
            wal_before = wal.read_bytes()
            await copy_schema2_workspace(self.source, self.destination, journal_roles=self.roles())
            self.assertEqual((self.source / "metadata.sqlite3").read_bytes(), main_before)
            self.assertEqual(wal.read_bytes(), wal_before)
            async with Workspace.open(self.destination) as copied:
                self.assertEqual(await copied.source_ids(), ["a", "b", "c"])
                self.assertEqual((await copied.source("c")).turns, self.conversation.turns)

    async def test_unresolved_classification_retains_raw_records_but_blocks_effective_read(self):
        """Ambiguous legacy records remain accessible without gaining new operational authority."""
        roles = {self.link.entry_id: "unresolved", self.patch.entry_id: "amendment"}
        await copy_schema2_workspace(self.source, self.destination, journal_roles=roles)
        async with Workspace.open(self.destination) as copied:
            self.assertEqual(await copied.edges("a"), [])
            self.assertEqual(await copied.inspect_journal("a"), self.history)
            with self.assertRaisesRegex(ConfigurationError, "unresolved legacy"):
                await copied.operational_journal("a")
            self.assertEqual(
                (await copied.resolve(JournalRef("a", self.link.entry_id))).text, self.raw_link
            )

    async def test_repeated_cancellation_waits_for_backup_before_removing_staging(self):
        """Cancellation cannot delete the staging directory underneath a still-running backup worker."""
        entered, release = threading.Event(), threading.Event()
        original_get = LocalBlobStore.get

        def held_get(store, digest):
            """Hold verified source reading until both cancellation requests have arrived."""
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release the backup worker")
            return original_get(store, digest)

        with patch.object(LocalBlobStore, "get", held_get):
            task = asyncio.create_task(
                copy_schema2_workspace(
                    self.source,
                    self.destination,
                    journal_roles=self.roles(),
                )
            )
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                await asyncio.sleep(0.01)
                task.cancel()
                await asyncio.sleep(0.01)
                self.assertFalse(task.done())
                self.assertEqual(len(list(self.root.glob(".llgm-convert-*/workspace"))), 1)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob(".llgm-convert-*")), [])
        self.assertEqual((self.source / "metadata.sqlite3").read_bytes(), self.original_bytes)

    async def test_existing_destination_and_corrupt_blob_do_not_damage_source(self):
        """Conversion refuses overwrite and checksum failures leave no published partial workspace."""
        self.destination.mkdir()
        marker = self.destination / "retained.txt"
        marker.write_text("retain this output")
        with self.assertRaises(ConfigurationError):
            await copy_schema2_workspace(self.source, self.destination, journal_roles=self.roles())
        self.assertEqual(marker.read_text(), "retain this output")
        blob = next(path for path in (self.source / "blobs").rglob("*") if path.is_file())
        blob.write_bytes(b"deliberately corrupt fixture")
        with self.assertRaises(ReferenceResolutionError):
            await copy_schema2_workspace(
                self.source, self.root / "failed-copy", journal_roles=self.roles()
            )
        self.assertFalse((self.root / "failed-copy").exists())
        self.assertEqual((self.source / "metadata.sqlite3").read_bytes(), self.original_bytes)
