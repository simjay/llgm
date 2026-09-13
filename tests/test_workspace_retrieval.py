"""Hybrid defaults, live corpus refresh and canonical evidence through the viewer."""

import asyncio
from copy import deepcopy

import pytest

from llgm import Conversation, Evidence, Settings, Workspace
from llgm.core.errors import CapabilityError
from llgm.core.types import SourceNode, Turn
from llgm.retrieval.base import passage_to_dict
from llgm.retrieval.passages import split_nodes
from llgm.retrieval.tokenizers import DiagnosticTokenizer
from llgm.retrieval.workspace import (
    WorkspaceHybridRetriever,
    configured_evidence_factory,
    snapshot_identity,
)
from llgm.viewer import GraphViewer


def conversation(text):
    """Build one canonical user turn for a storage-only fixture."""
    return Conversation.from_turns([{"role": "user", "text": text}])


class Transport:
    """Deterministic RPC fixture with real source spans and observable snapshot preparation."""

    def __init__(self):
        """Retain independent prepared generations and search requests."""
        self.prepared, self.queries = [], []
        self.fail = False

    async def prepare(self, records):
        """Bind a source generation without executing a model or native index builder."""
        if self.fail:
            raise RuntimeError("fixture unavailable")
        self.prepared.append(deepcopy(records))
        identity = snapshot_identity(records)
        return {"index_id": identity, "snapshot_sha256": identity, "descriptor": {"backend": "H"}}

    async def search(self, index_id, query, k):
        """Return newest source passages with deliberately untrusted display text."""
        self.queries.append((index_id, query, k))
        records = self.prepared[-1]
        nodes = [
            SourceNode(
                r["node_id"], tuple(Turn(**t) for t in r["turns"]), r["metadata"], r["timestamp_ms"]
            )
            for r in records
        ]
        passages = split_nodes(nodes, DiagnosticTokenizer())
        hits = []
        for rank, passage in enumerate(reversed(passages[-k:]), 1):
            payload = passage_to_dict(passage)
            payload["text"] = "untrusted display text"
            hits.append(
                {
                    "passage_id": passage.passage_id,
                    "passage": payload,
                    "rank": rank,
                    "score": 1 / rank,
                }
            )
        return {"index_id": index_id, "snapshot_sha256": index_id, "hits": hits}


def test_hybrid_is_the_configured_default_and_bm25_is_explicit():
    """Environment parsing selects hybrid unless a caller explicitly requests offline BM25."""
    assert Settings.from_env(environ={}).retriever_backend == "hybrid"
    assert (
        Settings.from_env(environ={"LLGM_RETRIEVER_BACKEND": "sqlite_fts5"}).retriever_backend
        == "sqlite_fts5"
    )


def test_viewer_hybrid_refreshes_appends_and_reuses_unchanged_generation(tmp_path):
    """Viewer search sees appended turns, reuses snapshots and preserves the current pointer."""

    async def run():
        """Exercise actual storage and canonical search over a controlled remote transport."""
        async with Workspace.open(tmp_path) as workspace:
            await workspace.append_conversation("chat", conversation("Atlas backups"), node_id=None)
            current = await workspace.conversation_node("chat")
            transport = Transport()
            retriever = WorkspaceHybridRetriever(
                workspace, Settings(), prepare_rpc=transport.prepare, search_rpc=transport.search
            )

            async def factory(target, *, passage_chars):
                """Share the same hybrid retriever between evidence and viewer handles."""
                return await Evidence.open(target, retriever=retriever, passage_chars=passage_chars)

            viewer = GraphViewer(workspace, evidence_factory=factory, conversation_id="chat")
            first = await viewer.search("backups")
            assert "Atlas backups" in str(first)
            assert "untrusted display text" not in str(first)
            await viewer.search("Atlas")
            assert len(transport.prepared) == 1
            await workspace.append_conversation(
                "chat", conversation("Keep them for seven days"), node_id=current
            )
            newest = await viewer.search("retention")
            assert "seven days" in str(newest)
            assert len(transport.prepared) == 2
            assert await workspace.conversation_node("chat") == current
            assert transport.queries[-1][0] != transport.queries[0][0]

    asyncio.run(run())


def test_failed_refresh_never_searches_a_stale_generation(tmp_path):
    """A failed new-corpus preparation propagates without serving a previous source index."""

    async def run():
        """Publish two generations and inject a failure preparing the second."""
        async with Workspace.open(tmp_path) as workspace:
            await workspace.ingest(conversation("Initial topic"))
            transport = Transport()
            retriever = WorkspaceHybridRetriever(
                workspace, Settings(), prepare_rpc=transport.prepare, search_rpc=transport.search
            )
            await retriever.search("topic", 12)
            await workspace.ingest(conversation("New topic"))
            transport.fail = True
            with pytest.raises(CapabilityError, match="Modal worker"):
                await retriever.search("topic", 12)
            assert len(transport.queries) == 1
            transport.fail = False
            await retriever.search("topic", 12)
            assert len(transport.prepared[-1]) == 2

    asyncio.run(run())


def test_empty_workspace_needs_no_remote_dependencies(tmp_path):
    """An empty source corpus returns no hits without importing Modal."""

    async def run():
        """Use the default factory against an actual empty workspace."""
        async with Workspace.open(tmp_path) as workspace:
            evidence = await configured_evidence_factory(workspace, Settings())(workspace)
            try:
                assert await evidence.search("anything") == []
            finally:
                await evidence.close()

    asyncio.run(run())


def test_missing_modal_dependency_has_an_explicit_offline_option(tmp_path, monkeypatch):
    """Missing semantic dependencies fail with setup guidance instead of returning BM25 hits."""

    def missing(name):
        """Simulate only the optional Modal dependency being absent."""
        assert name == "modal"
        raise ImportError(name)

    monkeypatch.setattr("llgm.retrieval.workspace.importlib.import_module", missing)

    async def run():
        """Attempt a source search through the configured default."""
        async with Workspace.open(tmp_path) as workspace:
            await workspace.ingest(conversation("Searchable lexical match"))
            retriever = WorkspaceHybridRetriever(workspace, Settings())
            with pytest.raises(CapabilityError, match="LLGM_RETRIEVER_BACKEND=sqlite_fts5"):
                await retriever.search("lexical", 12)

    asyncio.run(run())
