"""Local graph inspection with the application's own evidence search backend."""

from __future__ import annotations

import asyncio
import json
import secrets
from importlib.resources import files
from urllib.parse import parse_qs, urlsplit

from llgm.core.errors import ConfigurationError, LLGMError, ReferenceResolutionError
from llgm.core.types import JournalRef, SourceSpan, reference_to_dict
from llgm.memory.evidence import Evidence
from llgm.viewer import inspection


class GraphViewer:
    """Serve a local graph, paged records and evidence search without invoking answer models.

    Use as an async context manager. The workspace and injected evidence factory
    remain caller-owned. Search may refresh derived indexes. A configured hybrid
    factory can upload source snapshots and build indexes on Modal.
    Browsing and searching never move conversation pointers or publish evidence.
    """

    def __init__(
        self,
        workspace,
        *,
        evidence_factory=None,
        passage_chars=2048,
        retrieval_k=12,
        conversation_id="default",
        port=0,
    ):
        """Configure a loopback-only viewer and its evidence search factory."""
        if type(port) is not int or not 0 <= port <= 65535:
            raise ConfigurationError("port must be between 0 and 65535")
        if type(retrieval_k) is not int or not 1 <= retrieval_k <= 40:
            raise ConfigurationError("retrieval_k must be between 1 and 40")
        if type(passage_chars) is not int or passage_chars < 1:
            raise ConfigurationError("passage_chars must be positive")
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ConfigurationError("conversation_id must be nonempty text")
        if evidence_factory is not None and not callable(evidence_factory):
            raise ConfigurationError("evidence_factory must be an async callable")
        self.workspace = workspace
        self.evidence_factory = evidence_factory or Evidence.open
        self.passage_chars, self.retrieval_k = passage_chars, retrieval_k
        self.conversation_id, self.port = conversation_id, port
        self.url = ""
        self._server = None
        self._requests = set()
        self._prefix = "/" + secrets.token_urlsafe(24) + "/"

    @classmethod
    def from_application(cls, application, *, conversation_id="default", port=0):
        """Reuse the exact workspace, search factory and passage settings of an LLGM instance."""
        return cls(
            application.workspace,
            evidence_factory=application.evidence_factory,
            passage_chars=application.passage_chars,
            retrieval_k=application.retrieval_k,
            conversation_id=conversation_id,
            port=port,
        )

    async def __aenter__(self):
        """Listen on an available loopback port without opening a browser automatically."""
        if self._server is not None:
            raise ConfigurationError("Viewer is already running")
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", self.port, limit=16384)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}{self._prefix}"
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        """Stop accepting requests and drain active work before releasing caller resources."""
        self._server.close()
        await self._server.wait_closed()
        for task in self._requests:
            task.cancel()
        await asyncio.gather(*self._requests, return_exceptions=True)
        self._server = None
        self.url = ""

    async def serve_forever(self):
        """Keep an entered viewer alive until the caller cancels it."""
        if self._server is None:
            raise ConfigurationError("Enter the viewer before serving")
        await self._server.serve_forever()

    async def search(self, query):
        """Group ranked canonical evidence hits by node without changing the current topic."""
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise ConfigurationError("Search requires 1 to 4096 characters")
        evidence = await self.evidence_factory(self.workspace, passage_chars=self.passage_chars)
        try:
            hits = await evidence.search(query, self.retrieval_k)
            descriptor = evidence.descriptor()
            retriever = getattr(evidence, "retriever", None)
            if retriever is not None:
                descriptor = {**descriptor, "source_retriever": retriever.descriptor()}
            nodes = {}
            for hit in hits:
                owners = dict.fromkeys(ref.node_id for ref in hit.passage.refs)
                for owner in owners:
                    row = nodes.setdefault(owner, {"node_id": owner, "matches": []})
                    row["matches"].append(
                        {
                            "rank": hit.rank,
                            "score": hit.score,
                            "text": hit.passage.text[:1200],
                            "references": [reference_to_dict(ref) for ref in hit.passage.refs],
                        }
                    )
            return {"query": query, "nodes": list(nodes.values()), "backend": descriptor}
        finally:
            await evidence.close()

    async def _api(self, route, params):
        """Dispatch bounded read operations with explicitly validated identities and offsets."""
        node_id = params.get("node_id", "")
        offset = int(params.get("offset", "0"))
        if offset < 0:
            raise ConfigurationError("offset must be nonnegative")
        if route == "graph":
            return await inspection.graph(
                self.workspace, params.get("conversation_id", self.conversation_id)
            )
        if route == "search":
            return await self.search(params.get("q", ""))
        if route == "node":
            return await inspection.node(self.workspace, node_id, offset)
        if route == "journals":
            return await inspection.journals(self.workspace, node_id, offset)
        if route == "journal":
            value = await self.workspace.resolve(JournalRef(node_id, params.get("entry_id", "")))
            return {"text": value.text[:16384], "truncated": len(value.text) > 16384}
        if route == "turn":
            end = int(params.get("end", "0"))
            if end < offset or end - offset > 16384:
                raise ConfigurationError("Read at most 16384 characters per turn page")
            ref = SourceSpan(node_id, params.get("turn_id", ""), offset, end)
            value = await self.workspace.resolve(ref)
            return {"text": value.text, "reference": reference_to_dict(value.reference)}
        raise ReferenceResolutionError("Unknown viewer route")

    async def _response(self, target):
        """Serve packaged assets and canonical records below the per-session capability URL."""
        parsed = urlsplit(target)
        if not parsed.path.startswith(self._prefix):
            return 404, "text/plain", b"Not found"
        route = parsed.path[len(self._prefix) :]
        params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        if route in {"", "file", "app.js", "record.js", "style.css"}:
            asset = {"": "index.html", "file": "record.html"}.get(route, route)
            mime = (
                "text/javascript"
                if route.endswith(".js")
                else ("text/css" if route.endswith(".css") else "text/html")
            )
            return 200, mime, files("llgm.viewer").joinpath(asset).read_bytes()
        if route == "api/file":
            kind = params.get("kind")
            if kind not in {"manifest", "turn", "journal"}:
                raise ConfigurationError("Unknown record kind")
            content = await inspection.file(
                self.workspace, params.get("node_id", ""), kind, params.get("record_id", "")
            )
            return 200, "text/plain", content
        if route.startswith("api/"):
            payload = await self._api(route[4:], params)
            return 200, "application/json", json.dumps(payload, ensure_ascii=False).encode()
        return 404, "text/plain", b"Not found"

    async def _handle(self, reader, writer):
        """Handle one bounded local request with no arbitrary file access or cross-origin API."""
        task = asyncio.current_task()
        self._requests.add(task)
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            lines = raw.decode("iso-8859-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = dict(line.split(":", 1) for line in lines[1:] if ":" in line)
            headers = {key.lower(): value.strip() for key, value in headers.items()}
            origin = self.url.split(self._prefix)[0]
            if headers.get("host") != urlsplit(self.url).netloc or (
                "origin" in headers and headers["origin"] != origin
            ):
                status, mime, body = 403, "text/plain", b"Local requests only"
            elif method != "GET":
                status, mime, body = 405, "text/plain", b"Read-only viewer"
            else:
                try:
                    status, mime, body = await asyncio.wait_for(self._response(target), 60)
                except ReferenceResolutionError as exc:
                    status, mime, body = (
                        404,
                        "application/json",
                        json.dumps({"error": str(exc)}).encode(),
                    )
                except (LLGMError, ValueError) as exc:
                    status, mime, body = (
                        400,
                        "application/json",
                        json.dumps({"error": str(exc)}).encode(),
                    )
                except TimeoutError:
                    status, mime, body = 504, "application/json", b'{"error":"Request timed out"}'
                except Exception:
                    status, mime, body = (
                        500,
                        "application/json",
                        b'{"error":"Unable to read workspace"}',
                    )
            response = (
                f"HTTP/1.1 {status} Response\r\nContent-Type: {mime}; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\nCache-Control: no-store\r\n"
                "X-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n"
                "Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'\r\n\r\n"
            )
            writer.write(response.encode() + body)
            await writer.drain()
        except (
            ValueError,
            OSError,
            TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            finally:
                self._requests.discard(task)
