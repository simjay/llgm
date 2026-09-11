"""Real derived-index contracts: incremental work, current reads and metadata isolation."""

import asyncio
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from llgm.core.types import Conversation, JournalRef, NodeRef, Provenance, SourceSpan
from llgm.memory.evidence import Evidence
from llgm.memory.workspace import Workspace
from llgm.retrieval.base import SearchHit, SearchPassage
from llgm.retrieval.bm25 import SQLiteBM25Retriever
from llgm.storage import LocalBlobStore


class CountingBlobs:
    """Count real local blob reads, with an optional injected read failure."""

    def __init__(self, directory):
        """Keep actual content-addressed storage behind the observation counter."""
        self.store = LocalBlobStore(directory)
        self.reads = 0
        self.fail_at = None

    def put(self, data):
        """Publish original source bytes using the real local adapter."""
        return self.store.put(data)

    def get(self, digest):
        """Count and optionally fail one real source read."""
        self.reads += 1
        if self.reads == self.fail_at:
            raise OSError("injected source read failure")
        return self.store.get(digest)


class IncrementalEvidenceTests(unittest.IsolatedAsyncioTestCase):
    """Exercise persistent lexical projections through their public evidence handles."""

    async def asyncSetUp(self):
        """Open an isolated workspace and track its caller-owned evidence handles."""
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.blobs = CountingBlobs(self.path / "blobs")
        self.workspace = await Workspace.open(self.path, blob_store=self.blobs).__aenter__()
        self.handles = []

    async def asyncTearDown(self):
        """Close all handles before removing the workspace and derived databases."""
        for evidence in self.handles:
            await evidence.close()
        await self.workspace.close()
        self.temp.cleanup()

    async def source(self, node, text, *, metadata=None):
        """Publish one stable turn and return the resulting immutable node identity."""
        return await self.workspace.ingest(
            Conversation.from_turns(
                [{"role": "user", "text": text, "turn_id": "t"}],
                node_id=node,
                metadata=metadata,
            )
        )

    async def note(self, node="a", text="cobalt journal"):
        """Publish an independently scoped journal value for a visible source."""
        return await self.workspace.append_journal(
            node,
            subject=NodeRef(node),
            relation="value",
            value=text,
            provenance=Provenance("user", "fixture"),
            applicability={"scope": {"env": "staging"}},
        )

    async def open(self, **kwargs):
        """Retain an initialized evidence handle for deterministic cleanup."""
        evidence = await Evidence.open(self.workspace, **kwargs)
        self.handles.append(evidence)
        return evidence

    async def test_repeated_open_loads_no_sources_and_refresh_loads_only_changed_source(self):
        """Warm opens avoid corpus reads; new nodes and journals refresh only their append records."""
        for node in "abc":
            await self.source(node, f"cobalt source {node}")
        await self.note()
        self.blobs.reads = 0
        with patch.object(
            self.workspace, "sources", side_effect=AssertionError("eager corpus load")
        ):
            first = await self.open()
        self.assertEqual(self.blobs.reads, 3)
        self.assertEqual(first.preparation["source_records_indexed"], 3)
        self.assertEqual(first.preparation["journal_records_indexed"], 1)
        self.blobs.reads = 0
        await first.close()
        second = await self.open()
        self.assertEqual(self.blobs.reads, 0)
        self.assertEqual(second.preparation["passages_indexed"], 0)
        await self.source("new-b", "magenta replacement")
        await self.note("new-b", "magenta update")
        self.blobs.reads = 0
        changed = await self.open()
        self.assertEqual(self.blobs.reads, 1)
        self.assertEqual(changed.preparation["source_records_indexed"], 1)
        self.assertEqual(changed.preparation["journal_records_indexed"], 1)
        self.assertEqual(changed.preparation["passages_indexed"], 2)
        self.assertEqual(len(await second.search("magenta")), 2)
        self.assertEqual(len(await changed.search("magenta")), 2)

    async def test_source_ids_journal_and_source_reads_are_on_demand(self):
        """Metadata navigation never rereads source blobs; explicit source access reads just one."""
        await self.source("a", "cobalt")
        entry = await self.note()
        evidence = await self.open()
        self.blobs.reads = 0
        self.assertEqual(await self.workspace.source_ids(), ["a"])
        self.assertEqual(await evidence.journal("a"), [entry])
        self.assertEqual(self.blobs.reads, 0)
        self.assertEqual((await evidence.source("a")).turns[0].text, "cobalt")
        self.assertEqual(self.blobs.reads, 1)

    async def test_open_handles_see_new_evidence_and_restart_reuses_current_index(self):
        """Current searches discover new nodes and journals while immutable old sources remain readable."""
        await self.source("a", "cobalt cobalt zircon")
        await self.source("b", "zircon distant words words words")
        await self.source("c", "irrelevant background")
        original = await self.open()
        before = await original.search("cobalt zircon")
        for i in range(20):
            await self.source(f"future-{i}", "cobalt zircon")
        await self.source("replacement-a", "replacement magenta")
        await self.note("a", "future zircon journal")
        newer = await self.open()
        current = await original.search("cobalt zircon", 100)
        self.assertGreater(len(current), len(before))
        self.assertTrue(
            any(
                ref == SourceSpan("a", "t", 0, 20)
                for hit in await newer.search("cobalt", 100)
                for ref in hit.passage.refs
            )
        )
        self.assertEqual(len(await original.search("magenta")), 1)
        await self.workspace.close()
        self.workspace = await Workspace.open(self.path, blob_store=self.blobs).__aenter__()
        self.blobs.reads = 0
        reopened = await self.open()
        self.assertEqual(self.blobs.reads, 0)
        self.assertEqual(reopened.preparation["source_records_indexed"], 0)
        self.assertEqual(await reopened.search("cobalt zircon", 100), current)

    async def test_new_empty_node_does_not_retire_existing_source_passages(self):
        """An empty new source is indexed once without hiding any existing immutable evidence."""
        await self.source("a", "cobalt old")
        old = await self.open()
        await self.source("empty", "")
        current = await self.open()
        self.assertEqual(current.preparation["source_records_indexed"], 1)
        self.assertEqual(current.preparation["passages_indexed"], 0)
        self.assertEqual(len(await current.search("cobalt")), 1)
        self.assertEqual(len(await old.search("cobalt")), 1)
        self.assertEqual((await current.read(SourceSpan("a", "t", 0, 10))).text, "cobalt old")

    async def test_incremental_bm25_matches_fresh_fts5_reference_scores(self):
        """The reusable index uses ordinary FTS5 ranking on its current passage corpus."""
        texts = ["cobalt cobalt zircon", "zircon background words words words", "cobalt"] + [
            f"unrelated document {number}" for number in range(9)
        ]
        passages = []
        for number, text in enumerate(texts):
            node = f"n{number:03d}"
            await self.source(node, text)
            passages.append(SearchPassage(node, text, (SourceSpan(node, "t", 0, len(text)),)))
        evidence = await self.open()
        baseline = SQLiteBM25Retriever.from_passages(passages)
        try:
            actual = await evidence.search("cobalt zircon", 20)
            expected = await baseline.search("cobalt zircon", 20)
            self.assertEqual(
                [hit.passage.refs for hit in actual], [hit.passage.refs for hit in expected]
            )
            for observed, reference in zip(actual, expected, strict=True):
                self.assertAlmostEqual(observed.score, reference.score, places=12)
            self.assertEqual(evidence.descriptor()["implementation"], "sqlite-fts5-incremental")
        finally:
            baseline.close()

    async def test_failed_incremental_refresh_rolls_back_before_retry(self):
        """A failed source load cannot advance the derived cursor or publish a partial index."""
        await self.source("a", "cobalt first")
        await self.source("b", "cobalt second")
        self.blobs.reads = 0
        self.blobs.fail_at = 2
        with self.assertRaises(OSError):
            await self.open()
        self.blobs.fail_at = None
        self.blobs.reads = 0
        retry = await self.open()
        self.assertEqual(self.blobs.reads, 2)
        self.assertEqual(retry.preparation["source_records_indexed"], 2)
        self.assertEqual(len(await retry.search("cobalt")), 2)

    async def test_concurrent_workspace_refreshes_publish_each_source_once(self):
        """Independent workspace instances serialize one shared persistent index refresh."""
        for node in "abc":
            await self.source(node, "cobalt source")
        self.blobs.reads = 0
        async with Workspace.open(self.path, blob_store=self.blobs) as other:
            first, second = await asyncio.gather(
                self.open(),
                Evidence.open(other),
            )
            try:
                self.assertEqual(self.blobs.reads, 3)
                self.assertEqual(
                    first.preparation["source_records_indexed"]
                    + second.preparation["source_records_indexed"],
                    3,
                )
                self.assertEqual(await first.search("cobalt"), await second.search("cobalt"))
            finally:
                await second.close()

    async def test_cancelled_open_keeps_refresh_owned_until_workspace_cleanup(self):
        """Cancellation cannot detach the storage worker or close its connection mid-publication."""
        await self.source("a", "cobalt source")
        started, release = threading.Event(), threading.Event()
        original_get = self.blobs.get

        def blocked(digest):
            """Pause source loading inside the real refresh transaction until the test releases it."""
            started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release source read")
            return original_get(digest)

        with patch.object(self.blobs, "get", side_effect=blocked):
            opening = asyncio.create_task(Evidence.open(self.workspace))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                opening.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await opening
                closing = asyncio.create_task(self.workspace.close())
                await asyncio.sleep(0)
                self.assertFalse(closing.done())
            finally:
                release.set()
            await closing
        self.workspace = await Workspace.open(self.path, blob_store=self.blobs).__aenter__()
        self.blobs.reads = 0
        reopened = await self.open()
        self.assertEqual(self.blobs.reads, 0)
        self.assertEqual(reopened.preparation["source_records_indexed"], 0)
        self.assertEqual(len(await reopened.search("cobalt")), 1)

    async def test_injected_source_retriever_prepares_only_journals(self):
        """External source retrieval needs no source corpus materialization during open."""
        await self.source("a", "cobalt source")
        entry = await self.note()

        class Retriever:
            """Return one canonical source handle with deliberately untrusted display text."""

            async def search(self, query, k):
                """Exercise canonical resolution independently of local journal retrieval."""
                return [
                    SearchHit(
                        SearchPassage("external", "forged", (SourceSpan("a", "t", 0, 13),)), 1, 1
                    )
                ]

        self.blobs.reads = 0
        evidence = await self.open(retriever=Retriever())
        self.assertEqual(self.blobs.reads, 0)
        self.assertEqual(evidence.preparation["source_records_indexed"], 0)
        self.assertEqual(evidence.preparation["journal_records_indexed"], 1)
        hits = await evidence.search("cobalt")
        self.assertEqual({hit.passage.text for hit in hits}, {"cobalt source", entry.value})

    async def test_nested_search_and_read_metadata_cannot_mutate_storage_or_index(self):
        """Returned source and journal metadata remain independent across reads, searches and handles."""
        await self.source("a", "cobalt source", metadata={"nested": {"date": "original"}})
        entry = await self.note()
        evidence = await self.open()
        hits = await evidence.search("cobalt")
        source = next(hit for hit in hits if isinstance(hit.passage.refs[0], SourceSpan))
        journal = next(hit for hit in hits if isinstance(hit.passage.refs[0], JournalRef))
        source.passage.metadata["source_metadata"]["nested"]["date"] = "forged"
        journal.passage.metadata["applicability"]["scope"]["env"] = "production"
        journal.passage.metadata["provenance"]["origin"] = "system"
        ref = JournalRef("a", entry.entry_id, 0, len(entry.value))
        read = await evidence.read(ref)
        self.assertEqual(read.metadata["applicability"], {"scope": {"env": "staging"}})
        read.metadata["applicability"]["scope"]["env"] = "also forged"
        for handle in (evidence, await self.open()):
            repeated = await handle.search("cobalt")
            source = next(hit for hit in repeated if isinstance(hit.passage.refs[0], SourceSpan))
            journal = next(hit for hit in repeated if isinstance(hit.passage.refs[0], JournalRef))
            self.assertEqual(
                source.passage.metadata["source_metadata"]["nested"]["date"], "original"
            )
            self.assertEqual(
                journal.passage.metadata["applicability"], {"scope": {"env": "staging"}}
            )
            self.assertEqual(journal.passage.metadata["provenance"]["origin"], "user")
            self.assertEqual(
                (await handle.read(ref)).metadata["applicability"], {"scope": {"env": "staging"}}
            )

    async def test_derived_schema_mismatch_rebuilds_from_durable_records(self):
        """A version mismatch discards only disposable derived data and rebuilds canonical passages."""
        await self.source("a", "cobalt source")
        evidence = await self.open()
        before = await evidence.search("cobalt")
        await self.workspace.close()
        path = self.path / "metadata.sqlite3.indexes" / "evidence-2048.sqlite3"
        with sqlite3.connect(path) as connection:
            identity = json.loads(
                connection.execute("SELECT identity FROM index_info").fetchone()[0]
            )
            identity["schema_version"] = 999
            connection.execute("UPDATE index_info SET identity=?", (json.dumps(identity),))
        self.workspace = await Workspace.open(self.path, blob_store=self.blobs).__aenter__()
        self.blobs.reads = 0
        rebuilt = await self.open()
        self.assertEqual(self.blobs.reads, 1)
        self.assertEqual(rebuilt.preparation["source_records_indexed"], 1)
        self.assertEqual(await rebuilt.search("cobalt"), before)


if __name__ == "__main__":
    unittest.main()
