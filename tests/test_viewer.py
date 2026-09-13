"""Graph inspection contracts over actual workspace storage and evidence search."""

import asyncio
import json
from types import SimpleNamespace
from urllib.parse import urlencode, urlsplit

import pytest

from llgm import Conversation, Evidence, NodeRef, Provenance, SourceSpan, Workspace
from llgm.core.errors import ConfigurationError, ReferenceResolutionError
from llgm.retrieval.base import SearchHit, SearchPassage
from llgm.viewer import GraphViewer, inspection


async def seed(workspace):
    """Create connected topics and a saved conversation using production publication APIs."""
    for name, text in (("alpha", "Atlas uses PostgreSQL."), ("beta", "Backups stay seven days.")):
        await workspace.ingest(
            Conversation.from_turns([{"role": "user", "turn_id": "t", "text": text}], node_id=name)
        )
    await workspace.append_conversation(
        "chat",
        Conversation.from_turns([{"role": "user", "text": "The current topic is Atlas."}]),
        node_id="alpha",
    )
    await workspace.publish_edge("alpha", "beta", provenance=Provenance("user", "test"))
    return await workspace.append_journal(
        "alpha",
        subject=NodeRef("alpha"),
        record_kind="assertion",
        relation="note",
        value="Canary rollback instructions",
        provenance=Provenance("user", "test"),
    )


def test_graph_is_metadata_only_and_refreshes_conversation_pointers(tmp_path, monkeypatch):
    """Listing huge topics reads no blobs and follows only the selected conversation pointer."""

    async def scenario():
        """Change another handle and observe a coherent refreshed topology."""
        async with Workspace.open(tmp_path) as workspace:
            await seed(workspace)
            async with Workspace.open(tmp_path) as other:
                await other.append_conversation(
                    "other",
                    Conversation.from_turns([{"role": "user", "text": "Other chat"}]),
                    node_id="beta",
                )

                def no_blobs(digest):
                    """Fail if rendering graph metadata materializes any topic text."""
                    raise AssertionError("Graph overview read source text")

                monkeypatch.setattr(workspace.blob_store, "get", no_blobs)
                value = await inspection.graph(workspace, "chat")
                assert value["current_node_id"] == "alpha"
                assert len(value["nodes"]) == 2 and len(value["edges"]) == 1
                assert (
                    next(n for n in value["nodes"] if n["node_id"] == "alpha")["journal_count"] == 1
                )
                assert (await inspection.graph(workspace, "missing"))["current_node_id"] is None
                await other.append_conversation(
                    "chat",
                    Conversation.from_turns([{"role": "user", "text": "Clear topic change"}]),
                    node_id="beta",
                )
                assert (await inspection.graph(workspace, "chat"))["current_node_id"] == "beta"
                edge = (await other.edges("alpha"))[0]
                await other.withdraw_edge(edge.edge_id, provenance=Provenance("user", "test"))
                assert (await inspection.graph(workspace, "chat"))["edges"] == []

    asyncio.run(scenario())


def test_node_and_journal_browsing_pages_exact_records(tmp_path):
    """Large topic lists page separately from explicit canonical text and file reads."""

    async def scenario():
        """Read appended Unicode text, retained journals and imported base turns."""
        async with Workspace.open(tmp_path) as workspace:
            entry = await seed(workspace)
            await workspace.append_conversation(
                "chat",
                Conversation.from_turns(
                    [{"role": "user", "turn_id": str(i), "text": "🙂" * 17000} for i in range(35)]
                ),
                node_id="alpha",
            )
            viewer = GraphViewer(workspace)
            page = await viewer._api("node", {"node_id": "alpha"})
            assert (
                page["total_turns"] == 37 and len(page["turns"]) == 32 and page["next_offset"] == 32
            )
            assert len((await inspection.node(workspace, "alpha", 32))["turns"]) == 5
            turn = page["turns"][2]
            assert (
                await viewer._api(
                    "turn",
                    {
                        "node_id": "alpha",
                        "turn_id": turn["turn_id"],
                        "offset": "16384",
                        "end": "17000",
                    },
                )
            )["text"] == "🙂" * 616
            assert (
                await inspection.file(workspace, "alpha", "turn", turn["turn_id"])
                == ("🙂" * 17000).encode()
            )
            assert "turns" in json.loads(await inspection.file(workspace, "alpha", "manifest", ""))
            value = json.loads(await inspection.file(workspace, "alpha", "journal", entry.entry_id))
            assert value["entry_id"] == entry.entry_id
            assert (await inspection.journals(workspace, "alpha"))["entries"][0][
                "entry_id"
            ] == entry.entry_id
            with pytest.raises(ReferenceResolutionError):
                await inspection.file(workspace, "beta", "journal", entry.entry_id)
            with pytest.raises(ReferenceResolutionError):
                await inspection.file(workspace, "../../outside", "manifest", "")
            with pytest.raises(ConfigurationError):
                await viewer._api(
                    "turn", {"node_id": "alpha", "turn_id": turn["turn_id"], "end": "17000"}
                )
            assert await workspace.conversation_node("chat") == "alpha"

    asyncio.run(scenario())


def test_search_uses_default_backend_and_keeps_current_node(tmp_path):
    """BM25 source and journal matches navigate to canonical owners without a chat write."""

    async def scenario():
        """Compare viewer search ranks and scores directly with evidence search."""
        async with Workspace.open(tmp_path) as workspace:
            await seed(workspace)
            viewer = GraphViewer(workspace, conversation_id="chat")
            async with await Evidence.open(workspace) as evidence:
                expected = await evidence.search("PostgreSQL", 12)
            value = await viewer.search("PostgreSQL")
            assert value["nodes"][0]["node_id"] == "alpha"
            assert value["nodes"][0]["matches"][0]["score"] == expected[0].score
            assert value["nodes"][0]["matches"][0]["references"][0]["node_id"] == "alpha"
            assert (await viewer.search("Canary"))["nodes"][0]["node_id"] == "alpha"
            assert (await viewer.search("unfindableword"))["nodes"] == []
            assert await workspace.conversation_node("chat") == "alpha"

    asyncio.run(scenario())


def test_application_viewer_reuses_custom_backend_and_closes_evidence(tmp_path):
    """Injected retrieval supplies the ranking while canonical text and resource ownership survive."""

    async def scenario():
        """Share application search settings without creating any model client."""
        async with Workspace.open(tmp_path) as workspace:
            await seed(workspace)
            calls, handles = [], []

            class Retriever:
                """Return one ranked source reference with deliberately untrusted passage text."""

                async def search(self, query, k):
                    """Record the query and return the source selected by this backend."""
                    calls.append((query, k))
                    return [
                        SearchHit(
                            SearchPassage("custom", "invented", (SourceSpan("beta", "t", 0, 7),)),
                            8.0,
                            1,
                        )
                    ]

                def descriptor(self):
                    """Identify the custom scorer independently of the local journal index."""
                    return {"backend": "custom-test"}

            retriever = Retriever()

            async def factory(current, *, passage_chars):
                """Bind the same caller-owned retriever and record disposable handles."""
                assert current is workspace and passage_chars == 512
                evidence = await Evidence.open(current, retriever, passage_chars=passage_chars)
                handles.append(evidence)
                return evidence

            app = SimpleNamespace(
                workspace=workspace, evidence_factory=factory, passage_chars=512, retrieval_k=7
            )
            viewer = GraphViewer.from_application(app)
            result = await viewer.search("semantic request")
            assert calls == [("semantic request", 7)]
            assert result["nodes"][0]["matches"][0]["text"] == "Backups"
            assert result["backend"]["source_retriever"]["backend"] == "custom-test"
            assert handles[0]._closed
            assert await workspace.source_ids() == ["alpha", "beta"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method,host,origin,path,status",
    [
        ("GET", "127.0.0.1:8765", None, "api/graph", 200),
        ("POST", "127.0.0.1:8765", None, "api/graph", 405),
        ("GET", "attacker.example", None, "api/graph", 403),
        ("GET", "127.0.0.1:8765", "https://attacker.example", "api/graph", 403),
        ("GET", "127.0.0.1:8765", None, "../pyproject.toml", 404),
        ("GET", "127.0.0.1:8765", None, "api/node?node_id=missing", 404),
        ("GET", "127.0.0.1:8765", None, "api/search?q=", 400),
    ],
)
def test_http_boundary_is_local_read_only_and_scoped(tmp_path, method, host, origin, path, status):
    """The real request parser rejects writes, foreign origins and non-record file access."""

    async def scenario():
        """Drive HTTP bytes through in-memory streams without binding a test port."""
        async with Workspace.open(tmp_path) as workspace:
            viewer = GraphViewer(workspace)
            viewer.url = "http://127.0.0.1:8765" + viewer._prefix
            reader = asyncio.StreamReader()
            request = f"{method} {viewer._prefix}{path} HTTP/1.1\r\nHost: {host}\r\n"
            if origin:
                request += f"Origin: {origin}\r\n"
            reader.feed_data((request + "\r\n").encode())
            reader.feed_eof()

            class Writer:
                """Collect a complete HTTP response while honoring the stream lifecycle."""

                data = b""

                def write(self, data):
                    """Keep emitted response bytes for status and header assertions."""
                    self.data += data

                async def drain(self):
                    """Complete the in-memory write immediately."""

                def close(self):
                    """Close the in-memory transport without external resources."""

                async def wait_closed(self):
                    """Complete transport cleanup immediately."""

            writer = Writer()
            await viewer._handle(reader, writer)
            assert writer.data.startswith(f"HTTP/1.1 {status}".encode())
            assert b"Content-Security-Policy:" in writer.data
            assert b"Cache-Control: no-store" in writer.data
            assert not viewer._requests

    asyncio.run(scenario())


def test_packaged_viewer_serves_assets_and_capability_scoped_records(tmp_path):
    """HTML and scripts ship locally, and untrusted source text is served as plain text."""

    async def scenario():
        """Read packaged assets through the same routes used by a browser."""
        async with Workspace.open(tmp_path) as workspace:
            await seed(workspace)
            viewer = GraphViewer(workspace)
            for path, mime in (
                ("", "text/html"),
                ("app.js", "text/javascript"),
                ("style.css", "text/css"),
                ("file", "text/html"),
                ("record.js", "text/javascript"),
            ):
                status, content_type, body = await viewer._response(viewer._prefix + path)
                assert status == 200 and content_type == mime and body
            assert (await viewer._response("/api/graph"))[0] == 404
            status, mime, body = await viewer._response(
                viewer._prefix + "api/file?" + urlencode({"kind": "manifest", "node_id": "alpha"})
            )
            assert status == 200 and mime == "text/plain" and json.loads(body)

    asyncio.run(scenario())


@pytest.mark.integration
def test_live_viewer_http_and_shutdown(tmp_path):
    """An opted-in local HTTP check exercises server binding, JSON reads and socket cleanup."""
    from tests.conftest import require_opt_in

    require_opt_in("LLGM_TEST_VIEWER")

    async def scenario():
        """Use an actual local socket with production workspace and HTTP handlers."""
        async with Workspace.open(tmp_path) as workspace:
            await seed(workspace)
            async with GraphViewer(workspace, conversation_id="chat") as viewer:
                url = urlsplit(viewer.url)
                reader, writer = await asyncio.open_connection(url.hostname, url.port)
                writer.write(
                    f"GET {url.path}api/graph HTTP/1.1\r\nHost: {url.netloc}\r\n\r\n".encode()
                )
                await writer.drain()
                response = await reader.read()
                assert json.loads(response.split(b"\r\n\r\n", 1)[1])["current_node_id"] == "alpha"
                writer.close()
                await writer.wait_closed()
            with pytest.raises(OSError):
                await asyncio.open_connection(url.hostname, url.port)
            assert await workspace.source_ids() == ["alpha", "beta"]

    asyncio.run(scenario())
