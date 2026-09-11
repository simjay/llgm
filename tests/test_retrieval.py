"""Offline retrieval contracts; fake encoders are explicitly test doubles.

No test downloads model assets or exercises a hosted provider. The ColBERT
tests verify adapter boundaries, not real PLAID retrieval quality.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llgm.core.errors import ConfigurationError, ProviderError
from llgm.core.types import SourceNode, SourceSpan, Turn
from llgm.retrieval.base import SearchHit, SearchPassage, corpus_fingerprint
from llgm.retrieval.bm25 import SQLiteBM25Retriever
from llgm.retrieval.colbert import (
    ColBERTConfig,
    ColBERTRetriever,
    checkpoint_sha256,
    preflight_colbert,
)
from llgm.retrieval.dense import ExactDenseRetriever
from llgm.retrieval.hybrid import HybridRetriever
from llgm.retrieval.passages import split_nodes
from llgm.retrieval.tokenizers import DiagnosticTokenizer, validate_encoder_text


def passage(pid: str, text: str, *, node: str = "session-a") -> SearchPassage:
    """Build a passage whose canonical span covers exactly the supplied text."""
    return SearchPassage(
        pid,
        text,
        (SourceSpan(node, "turn-a", 0, len(text)),),
        {"role": "user", "date": "2026-09-09"},
    )


class FakeEmbedder:
    """Lookup vectors make tests deterministic; they are not a retrieval model."""

    def __init__(self, vectors: dict[str, list[float]], model: str = "test-encoder-v1"):
        """Bind fixed text-to-vector values to an explicit fake encoder identity."""
        self.lookup = vectors
        self.model = model
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Record one embedding batch and return the matching lookup vectors."""
        self.calls.append(list(texts))
        return [self.lookup[text] for text in texts]

    def descriptor(self) -> dict:
        """Identify the injected encoder without claiming a learned retrieval model."""
        return {"model": self.model, "implementation": "offline-test-double"}


class PassageTests(unittest.TestCase):
    """Passage rendering and source-coordinate preservation contracts."""

    def test_unicode_rendering_covers_original_codepoints_and_stable_refs(self):
        """Rendered windows cover Unicode codepoints and retain stable canonical references."""
        text = "  café 👩🏽‍💻 한글 e\u0301 😀\n" * 12 + "tail  "
        turns = (Turn("turn-a", "user", text), Turn("turn-b", "assistant", "另一个回复 😀" * 9))
        first = SourceNode("session-a", turns, {"date": "2026-09-09"})
        second = SourceNode("session-b", (Turn("turn-a", "user", text),), {})
        tokenizer = DiagnosticTokenizer("character")
        chunks = split_nodes([first, second], tokenizer, window=96, overlap=7)
        originals = {
            (node.node_id, turn.turn_id): turn.text
            for node in (first, second)
            for turn in node.turns
        }
        covered = {key: set() for key in originals}
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(tokenizer.count(chunk.text), 96)
            self.assertEqual(len(chunk.refs), 1)
            ref = chunk.refs[0]
            key = (ref.node_id, ref.turn_id)
            self.assertEqual(chunk.text.split("\n", 1)[1], originals[key][ref.start : ref.end])
            self.assertIn(f"role {chunk.metadata['role']}", chunk.text.split("\n", 1)[0])
            covered[key].update(range(ref.start, ref.end))
        for key, original in originals.items():
            self.assertEqual(covered[key], set(range(len(original))))
        old_refs = [(chunk.refs[0], chunk.text.split("\n", 1)[1]) for chunk in chunks]
        changed = split_nodes([first, second], tokenizer, window=110, overlap=7)
        self.assertNotEqual([p.passage_id for p in chunks], [p.passage_id for p in changed])
        for ref, old_text in old_refs:
            self.assertEqual(
                originals[(ref.node_id, ref.turn_id)][ref.start : ref.end],
                old_text,
            )
        self.assertEqual(chunks, split_nodes([first, second], tokenizer, window=96, overlap=7))

    def test_overlap_retains_boundary_evidence_without_crossing_turns(self):
        """Overlap preserves boundary evidence without combining different speaker turns."""
        node = SourceNode("s", (Turn("t", "user", " ".join((f"token{i}" for i in range(30)))),), {})
        chunks = split_nodes([node], DiagnosticTokenizer(), window=13, overlap=2)
        self.assertGreater(len(chunks), 1)
        for left, right in zip(chunks, chunks[1:]):
            self.assertGreater(left.refs[0].end, right.refs[0].start)
            self.assertGreater(right.refs[0].start, left.refs[0].start)

    def test_invalid_windows_or_unfit_metadata_are_explicit_failures(self):
        """Invalid windows and metadata that cannot fit fail before partial passage creation."""
        node = SourceNode("s", (Turn("t", "user", "evidence"),), {})
        for window, overlap in ((0, 0), (10, -1), (10, 10)):
            with (
                self.subTest(window=window, overlap=overlap),
                self.assertRaises(ConfigurationError),
            ):
                split_nodes([node], DiagnosticTokenizer(), window=window, overlap=overlap)
        with self.assertRaisesRegex(ConfigurationError, "Metadata"):
            split_nodes([node], DiagnosticTokenizer("character"), window=10, overlap=0)

    def test_encoder_overflow_is_rejected_at_actual_boundary(self):
        """Encoder limits are enforced against the selected tokenizer's actual count."""
        tokenizer = DiagnosticTokenizer()
        self.assertEqual(validate_encoder_text("one two three", tokenizer, 3), 3)
        with self.assertRaisesRegex(ConfigurationError, "truncation prohibited"):
            validate_encoder_text("one two three four", tokenizer, 3, kind="query")


class BM25Tests(unittest.IsolatedAsyncioTestCase):
    """SQLite FTS5 literal-query, persistence and accounting contracts."""

    async def test_fts_operators_are_literal_terms_and_sql_remains_intact(self):
        """FTS operators remain literal text and cannot modify the backing database."""
        passages = [
            passage("a", "alpha road"),
            passage("b", "beta river"),
            passage("c", "gamma hill"),
        ]
        index = SQLiteBM25Retriever.from_passages(passages)
        self.addCleanup(index.close)
        hits = await index.search("alpha AND beta", 10)
        self.assertEqual({hit.passage.passage_id for hit in hits}, {"a", "b"})
        injected = await index.search('alpha"); DROP TABLE passages; --', 10)
        self.assertEqual([hit.passage.passage_id for hit in injected], ["a"])
        self.assertEqual(
            (await index.search('"beta"* NEAR(title:absent)', 10))[0].passage, passages[1]
        )
        self.assertEqual(await index.search('"() * --', 10), [])
        self.assertEqual(await index.search("alpha", 0), [])
        with self.assertRaises(ConfigurationError):
            await index.search("alpha", -1)

    async def test_persistent_index_reuses_exact_corpus_and_preserves_unicode_refs(self):
        """A reopened BM25 index retains its corpus identity and exact Unicode references."""
        passages = [passage("a", "café 한글 🦉", node="node-한글"), passage("b", "different text")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bm25.sqlite3"
            index = SQLiteBM25Retriever.from_passages(passages, path)
            index.close()
            reopened = SQLiteBM25Retriever.from_passages(passages, path)
            try:
                hits = await reopened.search("cafe 한글", 10)
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0].passage, passages[0])
                self.assertEqual(hits[0].rank, 1)
                self.assertEqual(
                    reopened.descriptor()["corpus_sha256"], corpus_fingerprint(passages)
                )
            finally:
                reopened.close()
            for changed in (
                [passage("a", "changed content"), passages[1]],
                [passage("a", passages[0].text, node="different-node"), passages[1]],
            ):
                with (
                    self.subTest(changed=changed),
                    self.assertRaisesRegex(ConfigurationError, "different corpus"),
                ):
                    SQLiteBM25Retriever.from_passages(changed, path)

    async def test_duplicate_passage_ids_rejected(self):
        """A corpus cannot contain repeated passage identities."""
        with self.assertRaises(ConfigurationError):
            SQLiteBM25Retriever.from_passages([passage("same", "first"), passage("same", "second")])

    async def test_events_hash_private_query(self):
        """Search telemetry hashes query text instead of storing it in cleartext."""
        private = "private appointment address"
        index = SQLiteBM25Retriever.from_passages([passage("a", private)])
        self.addCleanup(index.close)
        await index.search(private, 1)
        self.assertNotIn(private, json.dumps(index.events))
        self.assertEqual(
            index.events[-1]["query_sha256"], hashlib.sha256(private.encode()).hexdigest()
        )


class DenseTests(unittest.IsolatedAsyncioTestCase):
    """Exact dense retrieval, encoder identity and numeric validation contracts."""

    async def test_injected_semantics_use_exact_normalized_cosine_and_stable_ties(self):
        """Normalized cosine scores and deterministic ties match the injected vectors."""
        passages = [
            passage("z", "automobile"),
            passage("a", "motor vehicle"),
            passage("n", "fruit"),
        ]
        embedder = FakeEmbedder(
            {"automobile": [10, 0], "motor vehicle": [2, 0], "fruit": [0, 5], "car": [7, 0]}
        )
        index = await ExactDenseRetriever.build(passages, embedder, batch_size=2)
        self.assertEqual(embedder.calls, [["automobile", "motor vehicle"], ["fruit"]])
        hits = await index.search("car", 3)
        self.assertEqual([hit.passage.passage_id for hit in hits], ["a", "z", "n"])
        self.assertEqual([hit.rank for hit in hits], [1, 2, 3])
        self.assertAlmostEqual(hits[0].score, 1.0)
        self.assertAlmostEqual(hits[-1].score, 0.0)
        self.assertEqual(hits[0].passage.refs, passages[1].refs)
        self.assertEqual(index.descriptor()["dimensions"], 2)
        self.assertEqual(index.events[-1]["scored_vectors"], 3)
        self.assertNotIn('"query":', json.dumps(index.events))
        self.assertEqual(index.events[-1]["query_sha256"], hashlib.sha256(b"car").hexdigest())

    async def test_snapshots_require_identical_evidence_and_query_model(self):
        """Dense snapshots reject a changed corpus or a different query encoder."""
        passages = [passage("a", "automobile")]
        embedder = FakeEmbedder({"automobile": [1, 0], "car": [1, 0]})
        index = await ExactDenseRetriever.build(passages, embedder)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.json"
            index.save(path)
            restored = ExactDenseRetriever.open(path, passages, embedder)
            self.assertEqual((await restored.search("car", 1))[0].passage, passages[0])
            with self.assertRaisesRegex(ConfigurationError, "model"):
                ExactDenseRetriever.open(path, passages, FakeEmbedder({}, model="other-encoder"))
            with self.assertRaisesRegex(ConfigurationError, "fingerprint"):
                ExactDenseRetriever.open(path, [passage("a", "changed source")], embedder)

    async def test_provider_shape_zero_and_nonfinite_fail_explicitly(self):
        """Invalid vector cardinality, dimensions, zero norms and nonfinite values fail explicitly."""
        passages = [passage("a", "first"), passage("b", "second")]
        for vectors in ([[1, 0]], [[1, 0], [1]], [[0, 0], [1, 0]], [[float("nan"), 0], [1, 0]]):
            with self.subTest(vectors=vectors), self.assertRaises(ProviderError):
                ExactDenseRetriever(passages, vectors, FakeEmbedder({}))
        index = ExactDenseRetriever([passages[0]], [[1, 0]], FakeEmbedder({"wrong": [1, 0, 0]}))
        with self.assertRaises(ProviderError):
            await index.search("wrong", 1)

    async def test_cosine_normalization_handles_extreme_finite_scales(self):
        """Equivalent directions rank identically without numeric overflow or underflow."""
        passages = [passage("aligned", "same direction"), passage("orthogonal", "other direction")]
        for scale in (1e308, 1e-308):
            with self.subTest(scale=scale):
                embedder = FakeEmbedder({"query": [scale, scale]})
                index = ExactDenseRetriever(passages, [[scale, scale], [scale, -scale]], embedder)
                hits = await index.search("query", 2)
                self.assertEqual(
                    [hit.passage.passage_id for hit in hits], ["aligned", "orthogonal"]
                )
                self.assertAlmostEqual(hits[0].score, 1.0)
                self.assertAlmostEqual(hits[1].score, 0.0)

    async def test_malformed_vectors_and_batch_sizes_fail_before_encoding(self):
        """Numeric-looking text, booleans and invalid batch sizes never become embeddings."""
        passages = [passage("a", "source")]
        embedder = FakeEmbedder({})
        for vector in (None, "12", ["1", "2"], [True, 0], [10**400, 0]):
            with self.subTest(vector=vector), self.assertRaises(ProviderError):
                ExactDenseRetriever(passages, [vector], embedder)
        for batch_size in (True, 1.5, "2", 0, -1):
            with self.subTest(batch_size=batch_size), self.assertRaises(ConfigurationError):
                await ExactDenseRetriever.build(passages, embedder, batch_size=batch_size)
        self.assertEqual(embedder.calls, [])

    async def test_invalid_search_limits_do_not_start_backend_work(self):
        """Every lightweight retriever rejects noninteger limits before issuing a query."""
        passages = [passage("a", "source")]
        embedder = FakeEmbedder({})
        lexical = SQLiteBM25Retriever.from_passages(passages)
        dense = ExactDenseRetriever(passages, [[1, 0]], embedder)
        try:
            for retriever in (lexical, dense, HybridRetriever(lexical, dense)):
                for k in (True, 1.5, "2", -1):
                    with self.subTest(retriever=type(retriever).__name__, k=k):
                        with self.assertRaises(ConfigurationError):
                            await retriever.search("source", k)
                self.assertEqual(retriever.events, [])
            self.assertEqual(embedder.calls, [])
        finally:
            lexical.close()

    async def test_empty_requests_do_not_invoke_encoder(self):
        """Empty corpora and zero-hit requests do not spend embedding calls."""
        embedder = FakeEmbedder({})
        empty = ExactDenseRetriever([], [], embedder)
        self.assertEqual(await empty.search("query", 10), [])
        nonempty = ExactDenseRetriever([passage("a", "first")], [[1]], embedder)
        self.assertEqual(await nonempty.search("query", 0), [])
        self.assertEqual(embedder.calls, [])
        with self.assertRaises(ConfigurationError):
            await nonempty.search("query", -1)


class StubRetriever:
    """Ordered retrieval double with a declared corpus fingerprint."""

    def __init__(self, passages: list[SearchPassage], fingerprint: str = "same-corpus"):
        """Store fixed passages and initialize a query ledger for hybrid checks."""
        self.passages, self.fingerprint = passages, fingerprint
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, k: int) -> list[SearchHit]:
        """Return at most the requested passages with deterministic ranks and scores."""
        self.calls.append((query, k))
        return [
            SearchHit(item, 999.0 - rank, rank) for rank, item in enumerate(self.passages[:k], 1)
        ]

    def descriptor(self) -> dict:
        """Expose the corpus identity required for compatible hybrid components."""
        return {"corpus_sha256": self.fingerprint, "implementation": "offline-test-double"}


class HybridTests(unittest.IsolatedAsyncioTestCase):
    """Hybrid rank fusion and separate component-work accounting contracts."""

    async def test_top40_internal_searches_rrf_identity_and_visible_budget(self):
        """Top-40 component searches preserve evidence identity within the visible hit limit."""
        shared, lexical_only, dense_only = (
            passage("shared", "shared evidence"),
            passage("lex", "exact ID"),
            passage("dense", "paraphrase"),
        )
        lexical = StubRetriever([shared, lexical_only])
        dense = StubRetriever([dense_only, shared])
        hybrid = HybridRetriever(lexical, dense)
        hits = await hybrid.search("private query", 2)
        self.assertEqual(lexical.calls, [("private query", 40)])
        self.assertEqual(dense.calls, [("private query", 40)])
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].passage, shared)
        self.assertAlmostEqual(hits[0].score, 1 / 61 + 1 / 62)
        self.assertEqual(hits[1].passage, dense_only)
        self.assertAlmostEqual(hits[1].score, 1 / 61)
        self.assertEqual(hits[0].passage.refs, shared.refs)
        event = hybrid.events[-1]
        self.assertEqual(event["component_hit_counts"], [2, 2])
        self.assertTrue(event["internal_work_charged_separately"])
        self.assertNotIn("private query", json.dumps(event))
        self.assertEqual(event["query_sha256"], hashlib.sha256(b"private query").hexdigest())

    async def test_corpus_mismatch_and_invalid_visible_budgets_rejected(self):
        """Hybrid construction rejects different corpora and unsupported visible budgets."""
        with self.assertRaises(ConfigurationError):
            HybridRetriever(StubRetriever([], "one"), StubRetriever([], "two"))
        left, right = StubRetriever([]), StubRetriever([])
        hybrid = HybridRetriever(left, right)
        self.assertEqual(await hybrid.search("query", 0), [])
        self.assertEqual(left.calls + right.calls, [])
        for k in (-1, 41):
            with self.subTest(k=k), self.assertRaises(ConfigurationError):
                await hybrid.search("query", k)

    async def test_failed_component_drains_sibling_before_returning(self):
        """A failed hybrid query owns sibling cleanup despite repeated caller cancellation."""
        started, cleaning, release, finished = (asyncio.Event() for _ in range(4))

        async def fail(query, k):
            """Fail only after the sibling has entered its request."""
            await started.wait()
            raise ProviderError("component failed")

        async def wait(query, k):
            """Keep resource release pending until the test explicitly permits completion."""
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                finished.set()

        lexical, dense = StubRetriever([]), StubRetriever([])
        lexical.search, dense.search = fail, wait
        hybrid = HybridRetriever(lexical, dense)
        task = asyncio.create_task(hybrid.search("query", 1))
        try:
            await asyncio.wait_for(cleaning.wait(), 2)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
            release.set()
            with self.assertRaisesRegex(ProviderError, "component failed"):
                await task
            self.assertTrue(finished.is_set())
            self.assertEqual(hybrid.events, [])
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_repeated_cancellation_does_not_interrupt_component_cleanup(self):
        """Cancelling the pending fused query cannot detach either component's release."""
        started, cleaning, release, finished = (asyncio.Event() for _ in range(4))

        async def wait(query, k):
            """Expose a blocked request followed by an independently blocked cleanup."""
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                finished.set()

        dense = StubRetriever([])
        dense.search = wait
        task = asyncio.create_task(HybridRetriever(StubRetriever([]), dense).search("query", 1))
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            await asyncio.wait_for(cleaning.wait(), 2)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(finished.is_set())
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


class ColBERTBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """ColBERT asset provenance, tokenizer limits and persisted-index boundaries."""

    def config(self, directory: str, **kwargs) -> ColBERTConfig:
        """Build isolated ColBERT configuration with explicit placeholder pins and overrides."""
        values = {
            "checkpoint_path": Path(directory) / "checkpoint",
            "checkpoint_sha256": "a" * 64,
            "repository_revision": "b" * 40,
            "index_root": Path(directory) / "indexes",
            "index_name": "case-a",
        }
        values.update(kwargs)
        return ColBERTConfig(**values)

    def test_import_does_not_load_optional_model_stack(self):
        """Importing the adapter does not load optional encoder or indexing libraries."""
        program = (
            "import sys; import llgm.retrieval.colbert; "
            "assert not {'colbert', 'torch', 'transformers', 'faiss'} & set(sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=15
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checkpoint_digest_depends_on_names_and_contents_not_location(self):
        """Checkpoint digests track relative filenames and bytes rather than absolute location."""
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            left.mkdir()
            right.mkdir()
            for folder in (left, right):
                (folder / "weights.bin").write_bytes(b"offline-fixture")
                (folder / "config.json").write_text('{"model_type":"bert"}', encoding="utf-8")
            original = checkpoint_sha256(left)
            self.assertEqual(original, checkpoint_sha256(right))
            (right / "config.json").write_text('{"model_type":"other"}', encoding="utf-8")
            self.assertNotEqual(original, checkpoint_sha256(right))
            (left / "weights.bin").rename(left / "different-name.bin")
            self.assertNotEqual(original, checkpoint_sha256(left))

    def test_missing_dependencies_and_assets_are_reported_without_loading_models(self):
        """Preflight reports absent capabilities without importing or loading model weights."""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            with patch("llgm.retrieval.colbert.importlib.util.find_spec", return_value=None):
                errors = preflight_colbert(config)
            self.assertTrue(any("local directory" in message for message in errors))
            self.assertTrue(any("ColBERTv2/PLAID is unavailable" in message for message in errors))
            self.assertTrue(any("faiss" in message for message in errors))

    def test_build_rejects_checksum_mismatch_before_optional_import(self):
        """A wrong checkpoint checksum blocks construction before optional imports."""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            Path(config.checkpoint_path).mkdir()
            (Path(config.checkpoint_path) / "weights.bin").write_bytes(b"offline")
            with patch("llgm.retrieval.colbert.importlib.import_module") as importer:
                with self.assertRaisesRegex(ConfigurationError, "checksum mismatch"):
                    ColBERTRetriever.build([passage("p", "text")], config=config)
                importer.assert_not_called()

    def test_repository_revision_mismatch_is_not_silently_accepted(self):
        """An official repository URL cannot compensate for a mismatched code revision."""
        from llgm.retrieval.colbert import _check_repository_revision

        def fake_git(_directory, *args):
            """Report a legitimate origin with a deliberately different checked-out commit."""
            if args == ("rev-parse", "HEAD"):
                return "c" * 40
            if args == ("remote", "-v"):
                return "origin https://github.com/stanford-futuredata/ColBERT.git (fetch)"
            return ""

        with (
            patch(
                "llgm.retrieval.colbert.importlib.util.find_spec",
                return_value=SimpleNamespace(origin="/offline/colbert/__init__.py"),
            ),
            patch("llgm.retrieval.colbert._git", side_effect=fake_git),
        ):
            with self.assertRaisesRegex(ConfigurationError, "does not match pin"):
                _check_repository_revision("b" * 40)

    def test_open_requires_matching_completed_index_manifest(self):
        """Opening requires a completed manifest matching the corpus and encoder pins."""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            with self.assertRaisesRegex(ConfigurationError, "completed LLGM"):
                ColBERTRetriever.open([passage("p", "text")], config=config)
            config.index_path.mkdir(parents=True)
            (config.index_path / "llgm-manifest.json").write_text(
                '{"identity": {}}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ConfigurationError, "pins do not match"):
                ColBERTRetriever.open([passage("p", "text")], config=config)

    async def test_special_marker_overflow_precedes_encoder_and_events_are_hashed(self):
        """Marker-inclusive overflow fails before search and query telemetry stays hashed."""

        class RawTokenizer:
            """Tokenizer double containing normal encoder boundary tokens."""

            def __call__(self, text, **_kwargs):
                """Represent one token per whitespace-delimited word plus encoder boundaries."""
                return {"input_ids": [101] + list(range(len(text.split()))) + [102]}

        class SharedTokenizer:
            """Shared tokenizer double that also accounts for the ColBERT marker."""

            def count(self, text, kind="document"):
                """Count word tokens plus encoder boundaries and one additional marker."""
                return len(text.split()) + 3

        class Searcher:
            """Searcher double exposing tokenization and observable encoder invocations."""

            checkpoint = SimpleNamespace(query_tokenizer=SimpleNamespace(tok=RawTokenizer()))

            def __init__(self):
                """Start with no physical search calls recorded."""
                self.calls = []

            def search(self, query, **kwargs):
                """Record the request and return one fixed ColBERT PID, rank and score."""
                self.calls.append((query, kwargs))
                return [0], [1], [4.25]

        with tempfile.TemporaryDirectory() as directory:
            searcher = Searcher()
            config = self.config(directory, query_maxlen=5)
            source = passage("p", "original evidence")
            retriever = ColBERTRetriever(
                config=config,
                passages=(source,),
                tokenizer=SharedTokenizer(),
                searcher=searcher,
                repository={"revision": config.repository_revision},
            )
            hits = await retriever.search("private query", 1)
            self.assertEqual(hits[0].passage, source)
            self.assertEqual(hits[0].score, 4.25)
            self.assertEqual(retriever.events[-1]["query_tokens"], 5)
            self.assertNotIn("private query", json.dumps(retriever.events))
            self.assertEqual(
                retriever.events[-1]["query_sha256"], hashlib.sha256(b"private query").hexdigest()
            )
            with self.assertRaisesRegex(ConfigurationError, "Silent truncation is forbidden"):
                await retriever.search("private query overflow", 1)
            self.assertEqual(len(searcher.calls), 1)
            self.assertEqual(searcher.calls[0][1]["full_length_search"], False)

    def test_shared_tokenizer_mismatch_is_explicit(self):
        """Disagreement between shared and raw token counts fails instead of truncating."""

        def raw(text, **kwargs):
            """Return a raw encoder sequence whose marker-adjusted count exposes the mismatch."""
            return {"input_ids": [101, 42, 102]}

        shared = SimpleNamespace(count=lambda text, kind="document": 3)
        with self.assertRaisesRegex(ConfigurationError, "counts disagree"):
            ColBERTRetriever._check_tokens(shared, raw, "text", "document", 180)

    def test_pins_and_index_path_must_be_explicit(self):
        """Unverified pins, unsafe index paths and invalid token limits fail configuration."""
        with tempfile.TemporaryDirectory() as directory:
            for override in (
                {"repository_revision": "main"},
                {"checkpoint_sha256": "unverified"},
                {"index_name": "../elsewhere"},
                {"query_maxlen": 2},
            ):
                with self.subTest(override=override), self.assertRaises(ConfigurationError):
                    self.config(directory, **override)


if __name__ == "__main__":
    unittest.main()
