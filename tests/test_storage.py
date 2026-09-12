"""Persistence contracts: exact evidence identity, publication, and journal history."""

import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from llgm.core.errors import CapabilityError, ConflictError, ReferenceResolutionError, SchemaError
from llgm.core.types import (
    Conversation,
    JournalRef,
    NodeRef,
    Provenance,
    SourceSpan,
    reference_from_dict,
    reference_to_dict,
)
from llgm.memory.workspace import Workspace
from llgm.storage import LocalBlobStore, S3BlobStore


class StorageTests(unittest.IsolatedAsyncioTestCase):
    """SQLite publication, immutable node identities and journal concurrency contracts."""

    async def asyncSetUp(self):
        """Open a temporary workspace for independently isolated persistence checks."""
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.memory = await Workspace.open(self.path).__aenter__()

    async def asyncTearDown(self):
        """Close the workspace before removing its temporary files."""
        await self.memory.close()
        self.temp.cleanup()

    async def source(self, text="original", node_id="a", key=None):
        """Publish one user turn with a stable node/turn identity and optional retry key."""
        return await self.memory.ingest(
            Conversation.from_turns(
                [
                    {"role": "user", "text": text, "turn_id": "t"},
                ],
                node_id=node_id,
                metadata={"origin": "fixture"},
            ),
            idempotency_key=key,
        )

    async def test_unicode_exact_text_and_restart(self):
        """Exact Unicode offsets and node identities survive closing and reopening storage."""
        text = "  e\u0301 / é 👩🏽‍💻\n\r\n終  "
        record = await self.source(text)
        span = SourceSpan(record.node_id, "t", 2, len(text) - 2)
        self.assertEqual((await self.memory.resolve(span)).text, text[2:-2])
        self.assertEqual(reference_from_dict(reference_to_dict(span)), span)
        await self.memory.close()
        self.memory = await Workspace.open(self.path).__aenter__()
        self.assertEqual((await self.memory.resolve(span)).text, text[2:-2])
        self.assertEqual((await self.memory.sources())[0].turns[0].text, text)

    async def test_rechunking_does_not_change_reference_coordinates(self):
        """Changing search-window boundaries cannot redefine durable source coordinates."""
        await self.source("zero one two three")
        ref = SourceSpan("a", "t", 5, 8)
        # Search windows are independent derived choices over the same source.
        text = (await self.memory.sources())[0].turns[0].text
        for width in (3, 7, 12):
            windows = [text[start : start + width] for start in range(0, len(text), width)]
            self.assertEqual("".join(windows), text)
            self.assertEqual((await self.memory.resolve(ref)).text, "one")

    async def test_whole_node_read_exposes_canonical_turn_handles(self):
        """Whole-node projections carry turn references so models need not guess coordinate boundaries."""
        result = await self.memory.ingest(
            Conversation.from_turns(
                [
                    {"role": "user", "turn_id": "question", "text": "  Where? 😺"},
                    {"role": "assistant", "turn_id": "answer", "text": "Here."},
                ]
            )
        )
        node = await self.memory.resolve(NodeRef(result.node_id))
        self.assertEqual(
            [turn["turn_id"] for turn in node.metadata["turns"]], ["question", "answer"]
        )
        for turn, expected in zip(node.metadata["turns"], ("  Where? 😺", "Here."), strict=True):
            self.assertNotIn("text", turn)
            ref = reference_from_dict(turn["reference"])
            self.assertEqual((ref.start, ref.end), (0, len(expected)))
            self.assertEqual((await self.memory.resolve(ref)).text, expected)

    async def test_idempotency_and_immutable_node_identity(self):
        """Retries reuse a node, while different evidence must receive a new identity."""
        first = await self.source(key="request-1")
        duplicate = await self.source(key="request-1")
        self.assertTrue(first.created)
        self.assertFalse(duplicate.created)
        self.assertEqual(first.node_id, duplicate.node_id)
        self.assertFalse((await self.source()).created)
        with self.assertRaises(ConflictError):
            await self.source("different", key="request-1")
        with self.assertRaises(ConflictError):
            await self.source("replacement", key="request-2")
        second = await self.source("replacement", node_id="b", key="request-2")
        old = await self.memory.resolve(NodeRef("a"))
        new = await self.memory.resolve(NodeRef(second.node_id))
        self.assertEqual(old.reference, NodeRef("a"))
        self.assertEqual(new.reference, NodeRef("b"))
        self.assertIn("original", old.text)
        self.assertIn("replacement", new.text)
        self.assertEqual((await self.memory.resolve(SourceSpan("a", "t", 0, 8))).text, "original")
        self.assertEqual(set(await self.memory.source_ids()), {"a", "b"})

    async def test_source_timestamp_is_explicit_and_unknown_stays_unknown(self):
        """Unix-millisecond event time survives persistence without inventing missing source time."""
        await self.source()
        self.assertIsNone((await self.memory.source("a")).timestamp_ms)
        event_time = 1_700_000_000_123
        result = await self.memory.ingest(
            Conversation.from_turns(
                [{"role": "user", "text": "dated"}],
                timestamp_ms=event_time,
            )
        )
        self.assertEqual((await self.memory.source(result.node_id)).timestamp_ms, event_time)
        self.assertEqual(
            (await self.memory.resolve(NodeRef(result.node_id))).metadata["timestamp_ms"],
            event_time,
        )
        for invalid in (True, 1.25, "2026-09-10"):
            with self.subTest(invalid=invalid), self.assertRaises(SchemaError):
                Conversation.from_turns([{"role": "user", "text": "dated"}], timestamp_ms=invalid)

    async def test_journal_links_inline_corrections_and_current_reads(self):
        """Journal pointers, inline values and corrections retain append history and current reads."""
        await self.source(node_id="a")
        await self.source("other", node_id="b")
        before_append = time.time_ns() // 1_000_000
        provenance = Provenance("user", "fixture", (NodeRef("a"),))
        link = await self.memory.append_journal(
            "a",
            subject=NodeRef("a"),
            relation="supported_by",
            value=NodeRef("b"),
            provenance=provenance,
            expected_sequence=0,
            idempotency_key="link",
        )
        note = await self.memory.append_journal(
            "a",
            subject=NodeRef("a"),
            relation="clarified_by",
            value="only staging",
            provenance=provenance,
        )
        correction = await self.memory.append_journal(
            "a",
            subject=JournalRef("a", link.entry_id),
            relation="retract",
            value="Wrong scope",
            record_kind="correction",
            provenance=provenance,
        )
        self.assertEqual(
            [entry.journal_sequence for entry in await self.memory.inspect_journal("a")], [1, 2, 3]
        )
        self.assertIs(type(link.recorded_at_ms), int)
        self.assertLessEqual(before_append, link.recorded_at_ms)
        self.assertLessEqual(link.recorded_at_ms, time.time_ns() // 1_000_000)
        self.assertEqual(
            (await self.memory.resolve(JournalRef("a", note.entry_id, 5, 12))).text, "staging"
        )
        structural = json.loads(
            (await self.memory.resolve(JournalRef("a", correction.entry_id))).text
        )
        self.assertEqual(structural["subject"]["entry_id"], link.entry_id)
        self.assertEqual((await self.memory.inspect_journal("a"))[0], link)
        with self.assertRaises(ReferenceResolutionError):
            await self.memory.resolve(JournalRef("a", link.entry_id, 0, 1))
        retry = await self.memory.append_journal(
            "a",
            subject=NodeRef("a"),
            relation="supported_by",
            value=NodeRef("b"),
            provenance=provenance,
            expected_sequence=0,
            idempotency_key="link",
        )
        self.assertEqual(retry, link)

    async def test_journal_rejects_dangling_and_wrong_owner(self):
        """Dangling and incorrectly owned journal references never publish."""
        await self.source()
        common = {"relation": "note", "provenance": Provenance("user", "fixture")}
        with self.assertRaises(ReferenceResolutionError):
            await self.memory.append_journal(
                "a", subject=NodeRef("a"), value=NodeRef("missing"), **common
            )
        with self.assertRaises(SchemaError):
            await self.memory.append_journal("a", subject=NodeRef("b"), value="note", **common)
        with self.assertRaises(SchemaError):
            await self.memory.append_journal(
                "a", subject=NodeRef("a"), value="note", record_kind="correction", **common
            )
        self.assertEqual(await self.memory.inspect_journal("a"), [])

    async def test_overwrite_appends_without_mutating_prior_journal_text(self):
        """An overwrite record preserves the earlier entry and immutable source span."""
        await self.source("original")
        fields = {
            "subject": SourceSpan("a", "t", 0, 8),
            "relation": "value",
            "provenance": Provenance("user", "fixture"),
        }
        first = await self.memory.append_journal("a", value="first interpretation", **fields)
        second = await self.memory.append_journal(
            "a", value="new interpretation", record_kind="overwrite", **fields
        )
        self.assertEqual(await self.memory.inspect_journal("a"), [first, second])
        self.assertEqual(
            (await self.memory.resolve(JournalRef("a", first.entry_id, 0, 5))).text, "first"
        )
        self.assertEqual((await self.memory.resolve(fields["subject"])).text, "original")

    async def test_concurrent_writers_compare_journal_sequence(self):
        """Competing writers at one expected sequence admit exactly one journal append."""
        await self.source()
        async with Workspace.open(self.path) as other:

            async def append(memory, text):
                """Attempt a journal append against the same expected initial sequence."""
                return await memory.append_journal(
                    "a",
                    subject=NodeRef("a"),
                    value=text,
                    relation="note",
                    provenance=Provenance("user", "fixture"),
                    expected_sequence=0,
                )

            results = await asyncio.gather(
                append(self.memory, "first"), append(other, "second"), return_exceptions=True
            )
        self.assertEqual(sum(isinstance(result, ConflictError) for result in results), 1)
        self.assertEqual(len(await self.memory.inspect_journal("a")), 1)

    async def test_concurrent_idempotent_ingest_publishes_once(self):
        """Concurrent retries of one ingest publish a single source commit."""
        async with Workspace.open(self.path) as other:
            conversation = Conversation.from_turns([{"role": "user", "text": "a"}])
            results = await asyncio.gather(
                self.memory.ingest(conversation, idempotency_key="same"),
                other.ingest(conversation, idempotency_key="same"),
            )
        self.assertEqual(results[0].node_id, results[1].node_id)
        self.assertEqual(sum(record.created for record in results), 1)
        self.assertEqual(len(await self.memory.sources()), 1)

    async def test_concurrent_first_open_shares_one_workspace_identity(self):
        """Concurrent first openers observe one durable workspace identity."""
        path = self.path / "concurrent-initialization"
        workspaces = [Workspace.open(path) for _ in range(4)]
        try:
            await asyncio.gather(*(workspace.__aenter__() for workspace in workspaces))
            self.assertEqual(len({workspace._workspace_id for workspace in workspaces}), 1)
        finally:
            await asyncio.gather(*(workspace.close() for workspace in workspaces))

    async def test_first_open_retries_transient_wal_busy_without_hiding_other_errors(self):
        """A WAL lock race retries successfully; an unrelated SQLite failure stays explicit."""
        connect = sqlite3.connect
        for code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_IOERR):
            with self.subTest(code=code):
                attempts = []

                class FirstPragmaFailure(sqlite3.Connection):
                    """Inject one journal-mode failure around an otherwise real SQLite connection."""

                    def execute(self, sql, *args, **kwargs):
                        """Fail the first WAL attempt with the selected SQLite result code."""
                        if sql == "PRAGMA journal_mode = WAL":
                            attempts.append(sql)
                            if len(attempts) == 1:
                                error = sqlite3.OperationalError("injected journal-mode failure")
                                error.sqlite_errorcode = code
                                raise error
                        return super().execute(sql, *args, **kwargs)

                def connection(*args, **kwargs):
                    """Preserve real storage while injecting the first pragma result."""
                    return connect(*args, **kwargs, factory=FirstPragmaFailure)

                workspace = Workspace.open(self.path / f"pragma-{code}")
                try:
                    with patch("llgm.memory.workspace.sqlite3.connect", side_effect=connection):
                        if code == sqlite3.SQLITE_BUSY:
                            async with workspace:
                                self.assertEqual(await workspace.source_ids(), [])
                            self.assertEqual(len(attempts), 2)
                        else:
                            with self.assertRaises(sqlite3.OperationalError):
                                await workspace.__aenter__()
                            self.assertEqual(len(attempts), 1)
                finally:
                    await workspace.close()

    async def test_metadata_failure_retry_leaves_only_unpublished_blob(self):
        """A conflicting retry leaves existing metadata and private indexing progress unchanged."""
        await self.source("existing", key="same")
        before = self.memory._connection.execute("SELECT COUNT(*) FROM _index_changes").fetchone()[
            0
        ]
        with self.assertRaises(ConflictError):
            await self.source("unpublished changed bytes", key="same")
        self.assertEqual(
            self.memory._connection.execute("SELECT COUNT(*) FROM _index_changes").fetchone()[0],
            before,
        )
        self.assertEqual(len(await self.memory.sources()), 1)
        self.assertEqual((await self.memory.sources())[0].turns[0].text, "existing")

    async def test_resolution_rejects_missing_nodes_and_bad_ranges(self):
        """Missing source identities and invalid spans fail resolution."""
        await self.source("abc")
        with self.assertRaises(ReferenceResolutionError):
            await self.memory.resolve(NodeRef("missing"))
        with self.assertRaises(ReferenceResolutionError):
            await self.memory.resolve(SourceSpan("a", "t", 0, 4))
        with self.assertRaises(ReferenceResolutionError):
            await self.memory.resolve(SourceSpan("a", "wrong-turn", 0, 0))

    async def test_blob_failure_never_publishes_source(self):
        """A failed blob upload cannot publish source metadata."""

        class BrokenBlobs:
            """Blob-store double that rejects writes and exposes accidental reads."""

            def put(self, data):
                """Simulate an unavailable blob backend before metadata publication."""
                raise OSError("upload failed")

            def get(self, digest):
                """Reject any read because the failed upload published no blob."""
                raise AssertionError("No source should be visible")

        async with Workspace.open(self.path / "failure", blob_store=BrokenBlobs()) as memory:
            with self.assertRaises(OSError):
                await memory.ingest(
                    Conversation.from_turns([{"role": "user", "text": "unpublished"}])
                )
            self.assertEqual(await memory.sources(), [])
            self.assertEqual(
                memory._connection.execute("SELECT COUNT(*) FROM _index_changes").fetchone()[0], 0
            )

    async def test_unsupported_schema_is_not_opened(self):
        """An unsupported database schema fails before normal workspace use."""
        await self.memory.close()
        database = sqlite3.connect(self.path / "metadata.sqlite3")
        database.execute("PRAGMA user_version=999")
        database.close()
        with self.assertRaises(CapabilityError):
            async with Workspace.open(self.path):
                pass

    async def test_legacy_schema_is_rejected_without_rewriting_existing_data(self):
        """The old versioned schema is rejected before any automatic data or journal-mode rewrite."""
        legacy = self.path / "legacy.sqlite3"
        with sqlite3.connect(legacy) as database:
            database.execute(
                "CREATE TABLE sources(node_id TEXT, source_version INTEGER, blob_digest TEXT)"
            )
            database.execute("INSERT INTO sources VALUES ('old-node',1,'old-digest')")
            database.execute("PRAGMA user_version=1")
        before = legacy.read_bytes()
        workspace = Workspace(legacy, LocalBlobStore(self.path / "legacy-blobs"))
        with self.assertRaisesRegex(CapabilityError, "schema 1"):
            await workspace.__aenter__()
        await workspace.close()
        self.assertEqual(legacy.read_bytes(), before)
        self.assertFalse((self.path / "legacy-blobs").exists())
        self.assertFalse(Path(str(legacy) + ".indexes").exists())


class BlobTests(unittest.TestCase):
    """Content-addressed local and S3 blob-store boundary contracts."""

    def test_local_deduplication_checksum_and_path_validation(self):
        """Local blobs deduplicate content and reject checksum/path violations."""
        with tempfile.TemporaryDirectory() as directory:
            store = LocalBlobStore(directory)
            digest = store.put(b"exact bytes")
            self.assertEqual(store.put(b"exact bytes"), digest)
            self.assertEqual(store.get(digest), b"exact bytes")
            with self.assertRaises(ReferenceResolutionError):
                store.get("../../secret")
            (Path(directory) / digest[:2] / digest[2:]).write_bytes(b"corrupt")
            with self.assertRaises(ReferenceResolutionError):
                store.get(digest)

    def test_s3_uses_conditional_publication_and_verifies_readback(self):
        """S3 conditional publication verifies existing bytes instead of overwriting blindly."""

        class AlreadyExists(Exception):
            """Represent S3's precondition failure for an already published object."""

            response = {"Error": {"Code": "PreconditionFailed"}}

        class Client:
            """Minimal S3 double for conditional publication and checksum validation."""

            def __init__(self):
                """Initialize one in-memory object store and record conditional write attempts."""
                self.objects = {}
                self.conditions = []

            def put_object(self, **kwargs):
                """Honor the absent-object precondition and preserve existing content."""
                self.conditions.append(kwargs["IfNoneMatch"])
                key = (kwargs["Bucket"], kwargs["Key"])
                if key in self.objects:
                    raise AlreadyExists()
                self.objects[key] = kwargs["Body"]

            def get_object(self, **kwargs):
                """Return stored bytes through the SDK's streaming-body response shape."""
                return {"Body": BytesIO(self.objects[(kwargs["Bucket"], kwargs["Key"])])}

        client = Client()
        store = S3BlobStore("s3://test/prefix/", client=client)
        digest = store.put(b"value")
        self.assertEqual(store.put(b"value"), digest)
        self.assertEqual(client.conditions, ["*", "*"])
        self.assertEqual(store.get(digest), b"value")


if __name__ == "__main__":
    unittest.main()
