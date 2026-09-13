"""Configured hybrid retrieval over growing workspace sources.

Source snapshots are sent to the authenticated Modal worker on their first
search and after source appends. Unchanged searches send only an index ID and
query. Journals continue to use the local evidence projection.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from copy import deepcopy
from dataclasses import asdict

from llgm.core.errors import CapabilityError, ConfigurationError, LLGMError
from llgm.core.types import SourceNode
from llgm.memory.evidence import Evidence
from llgm.retrieval._colbert import ranked_hits
from llgm.retrieval.base import passage_from_dict


def snapshot_identity(records):
    """Bind a complete ordered source snapshot to a stable transport identity."""
    return hashlib.sha256(
        json.dumps(
            records, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class WorkspaceHybridRetriever:
    """Refresh source snapshots lazily and validate remote hybrid search responses."""

    def __init__(self, workspace, settings, *, prepare_rpc=None, search_rpc=None):
        """Bind one workspace without connecting to Modal or opening an index."""
        self.workspace, self.settings = workspace, settings
        self._prepare_rpc, self._search_rpc = prepare_rpc, search_rpc
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._records = []
        self._prepared = None
        self._descriptor = {
            "backend": "hybrid",
            "implementation": "BM25 + ColBERTv2",
            "deployment": "modal",
        }

    def _changes(self):
        """Read committed new sources and turns under the workspace transaction lock."""
        with self.workspace._transaction(write=False):
            cutoff = self.workspace._connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM _index_changes WHERE kind IN ('source','turn')"
            ).fetchone()[0]
            records = [
                asdict(record)
                for record in self.workspace._changed_records(self._sequence, cutoff)
                if isinstance(record, SourceNode)
            ]
            return cutoff, records

    def _connect(self):
        """Load the optional SDK only when source search needs remote execution."""
        if self._prepare_rpc is not None and self._search_rpc is not None:
            return
        try:
            modal = importlib.import_module("modal")
        except ImportError as exc:
            raise CapabilityError(
                "Hybrid search requires llgm[modal] and the llgm-colbert deployment. "
                "In a checkout run make setup-colbert and make colbert-deploy. "
                "Set LLGM_RETRIEVER_BACKEND=sqlite_fts5 for explicit offline BM25 search."
            ) from exc
        options = {"environment_name": self.settings.retrieval_modal_environment or None}
        self._prepare_rpc = modal.Function.from_name(
            self.settings.retrieval_modal_app, "prepare_workspace", **options
        ).remote.aio
        self._search_rpc = modal.Function.from_name(
            self.settings.retrieval_modal_app, "search_workspace", **options
        ).remote.aio

    async def search(self, query, k):
        """Search a current source generation, with no lexical fallback on remote failure."""
        if (
            not isinstance(query, str)
            or not query.strip()
            or type(k) is not int
            or not 0 <= k <= 40
        ):
            raise ConfigurationError("Hybrid search requires a nonempty query and 0 <= k <= 40")
        if not k:
            return []
        async with self._lock:
            cutoff, additions = await self.workspace._run(self._changes)
            records = self._records + additions
            if not records:
                return []
            self._connect()
            if self._prepared is None or cutoff != self._sequence:
                identity = snapshot_identity(records)
                prepared = await self._call(self._prepare_rpc, records)
                if (
                    not isinstance(prepared, dict)
                    or prepared.get("snapshot_sha256") != identity
                    or not isinstance(prepared.get("index_id"), str)
                    or not isinstance(prepared.get("descriptor"), dict)
                    or prepared["descriptor"].get("backend") != "H"
                ):
                    raise CapabilityError("Hybrid worker returned an incompatible source snapshot")
                self._records, self._sequence = records, cutoff
                self._prepared = prepared
                self._descriptor = prepared["descriptor"]
            response = await self._call(self._search_rpc, self._prepared["index_id"], query, k)
            if (
                not isinstance(response, dict)
                or response.get("index_id") != self._prepared["index_id"]
                or response.get("snapshot_sha256") != self._prepared["snapshot_sha256"]
            ):
                raise CapabilityError("Hybrid search returned a different source generation")
            try:
                values = response["hits"]
                passages = [passage_from_dict(value["passage"]) for value in values]
                if len({p.passage_id for p in passages}) != len(passages):
                    raise ValueError("Duplicate passage")
                return ranked_hits(values, {p.passage_id: p for p in passages}, k)
            except (KeyError, TypeError, ValueError) as exc:
                raise CapabilityError("Hybrid worker returned invalid passages") from exc

    async def _call(self, rpc, *args):
        """Expose actionable transport failures while preserving deadlines and cancellation."""
        try:
            return await rpc(*args)
        except (LLGMError, TimeoutError):
            raise
        except Exception as exc:
            raise CapabilityError(
                "Hybrid search could not use the Modal worker. Check Modal authentication, "
                "LLGM_RETRIEVAL_MODAL_APP, and deployment of prepare_workspace and search_workspace. "
                "In a checkout run make colbert-deploy after preparing the pinned assets. "
                "Use LLGM_RETRIEVER_BACKEND=sqlite_fts5 for explicit offline BM25."
            ) from exc

    def descriptor(self):
        """Report the prepared hybrid generation and its actual semantic engine."""
        return deepcopy(self._descriptor)


def configured_evidence_factory(workspace, settings):
    """Select the shared application and viewer backend without allocating remote resources."""
    if settings.retriever_backend == "sqlite_fts5":
        return Evidence.open
    if settings.retriever_backend != "hybrid":
        raise CapabilityError("Custom retrieval requires an explicit evidence_factory")
    retriever = WorkspaceHybridRetriever(workspace, settings)

    async def open_evidence(target, *, passage_chars=2048):
        """Open a short-lived evidence handle sharing this workspace's hybrid retriever."""
        if target is not workspace:
            raise ConfigurationError("Configured retrieval belongs to a different workspace")
        return await Evidence.open(target, retriever=retriever, passage_chars=passage_chars)

    return open_evidence
