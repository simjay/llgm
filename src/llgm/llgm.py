"""Persistent ingestion, bounded graph maintenance, and question-driven inference.

The application owns per-answer evidence handles, while the caller owns its
Workspace, reusable index and model clients. Evidence reads apply complete local journals.
Automatic publication validates references and a declared relation allowlist.
It does not certify the semantic truth of model-proposed links.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, fields, replace
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Sequence

from llgm.core.config import Settings
from llgm.core.errors import BudgetExceeded, CapabilityError, ConfigurationError
from llgm.core.time import validate_instant_ms
from llgm.core.types import (
    Conversation,
    Edge,
    IngestResult,
    NodeRef,
    reference_to_dict,
)
from llgm.inference.budget import Budget, RunLedger, byte_token_bound
from llgm.inference.nodes import NodeRuntime, NodeSeed
from llgm.inference.repl import DockerREPLConfig
from llgm.inference.results import AnswerResult, EvidenceBundle
from llgm.memory.evidence import (
    Evidence,
    LinkProposal,
    accept_link,
    propose_links,
)
from llgm.memory.maintenance import MaintenancePolicy, MaintenanceResult, _MaintenanceClient
from llgm.memory.query import QueryEvidence
from llgm.memory.workspace import Workspace
from llgm.models.base import ModelClient


@dataclass(frozen=True)
class IngestionOutcome:
    """An ingested source and the outcome of its optional edge discovery."""

    source: IngestResult
    maintenance: MaintenanceResult


class LLGM:
    """Retrieve seed nodes, query local Python delegates, and synthesize their findings.

    Workspace and model clients are caller-owned. Maintenance failure does not
    roll back source ingestion. Initial retrieval runs before model inference.
    Selected seed branches share one run budget and citation registry. Delegates
    inspect journal-corrected source slices through isolated Python interpreters.
    Primary edges organize discovery independently of journal amendments.
    """

    def __init__(
        self,
        workspace: Workspace,
        root_model: ModelClient,
        sidecar_model: ModelClient,
        *,
        maintenance_model: ModelClient | None = None,
        maintenance_policy: MaintenancePolicy | None = None,
        inference_budget: Budget | None = None,
        max_depth: int = 3,
        max_steps: int = 16,
        max_operations: int = 128,
        passage_chars: int = 2048,
        max_seed_nodes: int = 3,
        retrieval_k: int = 12,
        max_concurrency: int = 3,
        max_journal_bytes: int = 65536,
        capture_text: bool = False,
        evidence_factory: Callable[..., Awaitable[Evidence]] | None = None,
        repl_config: DockerREPLConfig | None = None,
        repl_factory: Callable[..., Any] | None = None,
    ):
        """Bind clients and an optional async evidence factory without performing I/O.

        The factory receives ``workspace`` and ``passage_chars``
        and returns an open evidence handle. The application
        closes each returned handle. Injected retrievers remain caller-owned.
        """
        for name, value in (
            ("passage_chars", passage_chars),
            ("max_seed_nodes", max_seed_nodes),
            ("retrieval_k", retrieval_k),
            ("max_concurrency", max_concurrency),
            ("max_journal_bytes", max_journal_bytes),
        ):
            if type(value) is not int or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        if retrieval_k > 40:
            raise ConfigurationError("retrieval_k must not exceed 40")
        if max_seed_nodes > retrieval_k:
            raise ConfigurationError("max_seed_nodes must not exceed retrieval_k")
        if type(capture_text) is not bool:
            raise ConfigurationError("capture_text must be a boolean")
        if evidence_factory is not None and not callable(evidence_factory):
            raise ConfigurationError("evidence_factory must be an async callable")
        if repl_factory is not None and not callable(repl_factory):
            raise ConfigurationError("repl_factory must be callable")
        if repl_config is not None and not isinstance(repl_config, DockerREPLConfig):
            raise ConfigurationError("repl_config must be a DockerREPLConfig")
        for name, value in (
            ("max_depth", max_depth),
            ("max_steps", max_steps),
            ("max_operations", max_operations),
        ):
            if type(value) is not int or value < (0 if name == "max_depth" else 1):
                raise ConfigurationError(f"Invalid {name}")
        self.workspace = workspace
        self.root_model, self.sidecar_model = root_model, sidecar_model
        self.maintenance_model = (
            maintenance_model if maintenance_model is not None else sidecar_model
        )
        self.maintenance_policy = maintenance_policy or MaintenancePolicy()
        self.inference_budget = inference_budget or Budget(
            max_model_calls=40,
            max_sidecar_calls=36,
            max_searches=8,
            max_evidence_tokens=65536,
            max_bundle_tokens=8000,
            max_context_tokens=65536,
            max_output_tokens=2048,
        )
        self.max_depth, self.max_steps, self.max_operations = max_depth, max_steps, max_operations
        self.passage_chars = passage_chars
        self.max_seed_nodes, self.retrieval_k = max_seed_nodes, retrieval_k
        self.max_concurrency, self.max_journal_bytes = max_concurrency, max_journal_bytes
        self.capture_text = capture_text
        self.evidence_factory = evidence_factory
        self.repl_config, self.repl_factory = repl_config, repl_factory
        self.last_maintenance: MaintenanceResult | None = None
        self.last_trace: list[dict] = []
        self.last_usage: dict = {}
        self._maintenance_lock = asyncio.Lock()

    @classmethod
    @asynccontextmanager
    async def from_settings(
        cls,
        settings: Settings,
        *,
        maintenance_policy: MaintenancePolicy | None = None,
        evidence_factory: Callable[..., Awaitable[Evidence]] | None = None,
        **runtime_options,
    ) -> AsyncIterator[LLGM]:
        """Own configured workspace and model clients for one async application context.

        Native and explicitly configured compatible model adapters are supported.
        The default evidence factory uses the workspace's local lexical index.
        Other retrieval selections require an explicit ``evidence_factory``.
        Construction or user-code failure closes every resource already created.

        Use ``async with LLGM.from_settings(settings) as memory``. Both model
        IDs must be configured. ``runtime_options`` supplies constructor options
        such as ``max_depth`` and ``max_steps``. Settings supply storage, providers,
        retrieval limits, concurrency, and the answer budget.
        """
        from llgm.models import create_model

        if not all(
            isinstance(getattr(settings, role + "_model", None), str)
            and getattr(settings, role + "_model").strip()
            for role in ("root", "sidecar")
        ):
            raise ConfigurationError(
                "Set explicit root_model and sidecar_model for application inference"
            )
        if evidence_factory is not None and not callable(evidence_factory):
            raise ConfigurationError("evidence_factory must be an async callable")
        if settings.metadata_backend != "sqlite":
            raise CapabilityError("The application settings factory requires sqlite metadata")
        if settings.retriever_backend != "sqlite_fts5" and evidence_factory is None:
            raise CapabilityError("Custom retrieval requires an explicit evidence_factory")
        options = {
            "inference_budget": Budget(
                **{item.name: getattr(settings, item.name) for item in fields(Budget)}
            ),
            **{
                name: getattr(settings, name)
                for name in (
                    "max_seed_nodes",
                    "retrieval_k",
                    "max_concurrency",
                    "max_journal_bytes",
                )
            },
            "repl_config": DockerREPLConfig(image=settings.node_repl_image),
            **runtime_options,
        }
        async with AsyncExitStack() as stack:
            workspace = await stack.enter_async_context(Workspace.open(settings=settings))
            clients = []
            for role in ("root", "sidecar"):
                client = create_model(
                    getattr(settings, role + "_provider"),
                    getattr(settings, role + "_model"),
                    base_url=getattr(settings, role + "_base_url"),
                    api_key_env=getattr(settings, role + "_api_key_env"),
                    timeout_seconds=settings.timeout_seconds,
                )
                stack.push_async_callback(client.aclose)
                clients.append(client)
            yield cls(
                workspace,
                clients[0],
                clients[1],
                maintenance_policy=maintenance_policy,
                evidence_factory=evidence_factory,
                **options,
            )

    async def _open_evidence(self) -> Evidence:
        """Acquire a current evidence handle whose lifetime belongs to this operation."""
        factory = self.evidence_factory or Evidence.open
        return await factory(self.workspace, passage_chars=self.passage_chars)

    async def ingest(
        self,
        conversation: Conversation,
        *,
        idempotency_key: str | None = None,
        organize: bool = True,
    ) -> IngestionOutcome:
        """Store a conversation and optionally discover relationships to other sources.

        :param conversation: Exact turns and optional source metadata to retain.
        :param idempotency_key: Reuse the source when retrying identical input.
            Reusing a key with different input raises a conflict.
        :param organize: Run maintenance after storing the source. False skips
            model calls for this ingestion. The maintenance policy also applies.
        :returns: The stored source ID and a separate maintenance outcome.
            Maintenance failure does not remove the source. Retrying ingestion
            can run maintenance again even when the source already exists.
        """
        source = await self.workspace.ingest(conversation, idempotency_key=idempotency_key)
        maintenance = await self.organize([source.node_id], disabled=not organize)
        return IngestionOutcome(source, maintenance)

    async def organize(
        self, node_ids: Sequence[str] | None = None, *, disabled: bool = False
    ) -> MaintenanceResult:
        """Discover bounded candidates and retain partial publications and failed-call accounting."""
        async with self._maintenance_lock:
            return await self._organize(node_ids, disabled=disabled)

    async def _organize(
        self, node_ids: Sequence[str] | None, *, disabled: bool
    ) -> MaintenanceResult:
        """Discover primary edges and retain accepted work if a later proposal fails."""
        policy = self.maintenance_policy
        ledger = RunLedger(policy.budget, byte_token_bound)
        proposals, accepted, decisions = [], [], []
        deferred = ()
        status, error_type = "completed", None
        try:
            if disabled or policy.mode == "disabled":
                status = "disabled"
            else:
                async with asyncio.timeout(policy.budget.timeout_seconds):
                    async with await self._open_evidence() as evidence:
                        visible = set(await self.workspace.source_ids())
                        if isinstance(node_ids, str) or (
                            node_ids is not None
                            and any(
                                not isinstance(node_id, str) or not node_id for node_id in node_ids
                            )
                        ):
                            raise ConfigurationError(
                                "node_ids must contain nonempty node identifiers"
                            )
                        selected = (
                            sorted(visible) if node_ids is None else list(dict.fromkeys(node_ids))
                        )
                        if set(selected) - visible:
                            raise ConfigurationError("Maintenance node does not exist")
                        deferred = tuple(selected[policy.max_nodes :])
                        model = _MaintenanceClient(
                            self.maintenance_model, ledger, policy.allowed_relations
                        )
                        for node_id in selected[: policy.max_nodes]:
                            ledger.remaining_seconds()
                            if len(visible) < 2:
                                continue
                            if ledger.searches >= policy.budget.max_searches:
                                raise BudgetExceeded("Maintenance search allowance exhausted")
                            source = await evidence.read(NodeRef(node_id))
                            ledger.searches += 1
                            hits = await evidence.search(
                                source.text[: policy.max_context_chars], policy.max_candidates * 4
                            )
                            candidates = list(
                                dict.fromkeys(
                                    reference.node_id
                                    for hit in hits
                                    for reference in hit.passage.refs
                                    if reference.node_id != node_id
                                )
                            )[: policy.max_candidates]
                            node_proposals = await propose_links(
                                evidence,
                                node_id,
                                model,
                                candidate_node_ids=candidates,
                                candidate_hits=hits,
                                max_candidates=policy.max_candidates,
                                max_context_chars=policy.max_context_chars,
                            )
                            proposals.extend(node_proposals)
                            edges = list(await self.workspace.edges(node_id))
                            published = 0
                            for proposal in node_proposals:
                                reason = self._decision(proposal, edges, published)
                                decision = {
                                    "source_node_id": node_id,
                                    "target_node_id": proposal.target.node_id,
                                    "relation": proposal.relation,
                                    "decision": reason,
                                }
                                if reason == "publish":
                                    key = hashlib.sha256(
                                        json.dumps(
                                            {
                                                "source": reference_to_dict(proposal.source),
                                                "target": reference_to_dict(proposal.target),
                                                "relation": proposal.relation,
                                                "applicability": dict(proposal.applicability or {}),
                                                "support": [
                                                    reference_to_dict(reference)
                                                    for reference in proposal.supporting_references
                                                ],
                                            },
                                            sort_keys=True,
                                        ).encode()
                                    ).hexdigest()
                                    edge = await accept_link(
                                        evidence, proposal, idempotency_key="auto-link:" + key
                                    )
                                    accepted.append(edge)
                                    edges.append(edge)
                                    published += 1
                                    decision["edge_id"] = edge.edge_id
                                decisions.append(decision)
        except asyncio.CancelledError:
            status, error_type = "cancelled", "CancelledError"
            raise
        except (TimeoutError, BudgetExceeded) as error:
            status, error_type = "budget_exhausted", type(error).__name__
        except Exception as error:
            status, error_type = ("partial" if accepted else "failed"), type(error).__name__
        finally:
            usage = ledger.usage()
            usage["maintenance_calls"] = usage.pop("sidecar_calls")
            usage["evidence_accounting_policy"] = "unique-presented-passage-json-utf8-bytes"
            self.last_maintenance = MaintenanceResult(
                status,
                tuple(proposals),
                tuple(accepted),
                tuple(decisions),
                deferred,
                usage,
                tuple(ledger.events),
                error_type,
            )
        return self.last_maintenance

    def _decision(self, proposal: LinkProposal, edges: Sequence[Edge], published: int) -> str:
        """Apply explicit publication policy without assigning semantic confidence."""
        policy = self.maintenance_policy
        if proposal.relation not in policy.allowed_relations:
            return "relation_not_allowed"
        if any(
            edge.source_node_id == proposal.source.node_id
            and edge.target_node_id == proposal.target.node_id
            and edge.relation == proposal.relation
            and dict(edge.applicability or {}) == dict(proposal.applicability or {})
            for edge in edges
        ):
            return "already_recorded"
        if policy.mode == "propose":
            return "review_required"
        if published >= policy.max_links_per_node:
            return "link_limit"
        return "publish"

    async def answer(
        self,
        question: str,
        *,
        scope: Mapping[str, Any] | None = None,
        query_date: str | None = None,
        as_of_ms: int | None = None,
        budget: Budget | None = None,
        node_id: str | None = None,
    ) -> AnswerResult:
        """Read relevant sources and combine their findings into a cited answer.

        :param question: Nonempty question text.
        :param scope: Values used to select applicable journal amendments and
            edges, for example ``{"env": "production"}``. This does not restrict
            source search or provide an access-control boundary.
        :param query_date: Original date text supplied as context to the models.
            It does not set a machine instant or imply ``as_of_ms``.
        :param as_of_ms: Evidence validity instant in integer Unix milliseconds.
            This does not reconstruct a historical workspace snapshot.
        :param budget: Complete budget for this answer, replacing the application
            budget. Omit to use ``memory.inference_budget``. Use
            ``dataclasses.replace`` to change only selected limits.
        :param node_id: Start at this source alone and skip initial retrieval.
            Its delegate can still search and follow links.
        :returns: Answer text, canonical evidence references, execution status,
            usage, and trace. Inspect ``result.evidence.unresolved`` for gaps.

        Invalid arguments, preparation errors, unexpected errors, and cancellation
        can raise instead of returning a result. Completion records execution
        status, not an independent judgment of answer correctness.
        """
        if not isinstance(question, str) or not question.strip():
            raise ConfigurationError("question must be nonempty text")
        question = question.strip()
        if scope is not None and not isinstance(scope, Mapping):
            raise ConfigurationError("scope must be a mapping")
        if query_date is not None and (not isinstance(query_date, str) or not query_date.strip()):
            raise ConfigurationError("query_date must be nonempty text or None")
        validate_instant_ms(as_of_ms, "as_of_ms")
        if node_id is not None:
            NodeRef(node_id)
        selected_budget = budget or self.inference_budget
        started = time.monotonic()
        preparation_ledger = RunLedger(selected_budget, byte_token_bound, reserve_root=True)
        preparation = {"kind": "preparation", "status": "started"}
        self.last_trace, self.last_usage = [preparation], {}
        evidence, runtime, result = None, None, None
        prepared_at, inference_finished = None, None
        try:
            async with asyncio.timeout(selected_budget.timeout_seconds):
                evidence = await self._open_evidence()
                scoped = QueryEvidence(
                    evidence,
                    scope or {},
                    query_date,
                    as_of_ms=as_of_ms,
                    max_journal_bytes=self.max_journal_bytes,
                )
                seeds = await self._seeds(question, node_id, evidence, preparation_ledger)
                prepared_at = time.monotonic()
                preparation.update(status="completed", index=dict(evidence.preparation))
                remaining = selected_budget.timeout_seconds - (prepared_at - started)
                # A synchronous index build cannot be interrupted by asyncio;
                # recheck wall time before dispatching any model request.
                if remaining <= 0:
                    preparation["status"] = "budget_exhausted"
                    raise BudgetExceeded("Answer preparation exhausted the run deadline")
                runtime = NodeRuntime(
                    self.root_model,
                    self.sidecar_model,
                    scoped,
                    budget=replace(selected_budget, timeout_seconds=remaining),
                    max_depth=self.max_depth,
                    max_steps=self.max_steps,
                    max_operations=self.max_operations,
                    max_concurrency=self.max_concurrency,
                    repl_config=self.repl_config,
                    repl_factory=self.repl_factory,
                    capture_text=self.capture_text,
                )
                result = await runtime.answer(
                    question,
                    seeds=seeds,
                    query_date=query_date,
                    query_scope=scope,
                    ledger=preparation_ledger,
                )
                inference_finished = time.monotonic()
        except (TimeoutError, BudgetExceeded):
            if runtime is None:
                preparation["status"] = "budget_exhausted"
            result = AnswerResult(
                "",
                EvidenceBundle(
                    unresolved=[
                        "BudgetExceeded: application preparation or inference deadline exhausted"
                    ],
                    stop_reason="budget_exhausted",
                ),
                {},
                [],
                "budget_exhausted",
            )
        except asyncio.CancelledError:
            if runtime is None:
                preparation["status"] = "cancelled"
            raise
        except Exception as error:
            if runtime is None:
                preparation.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            stopped = time.monotonic()
            if evidence is not None:
                await evidence.close()
            finished = time.monotonic()
            preparation_seconds = (prepared_at if prepared_at is not None else stopped) - started
            inference_seconds = (
                0.0
                if runtime is None
                else (
                    (inference_finished if inference_finished is not None else stopped)
                    - prepared_at
                )
            )
            preparation["elapsed_seconds"] = preparation_seconds
            self.last_trace = [preparation, *(runtime.last_trace if runtime is not None else [])]
            self.last_usage = dict(
                runtime.last_usage if runtime is not None else preparation_ledger.usage()
            )
            self.last_usage.update(
                preparation_seconds=preparation_seconds,
                inference_seconds=inference_seconds,
                cleanup_seconds=finished - stopped,
                total_seconds=finished - started,
                elapsed_seconds=finished - started,
            )
            if result is not None:
                result.usage = result.evidence.usage = self.last_usage
                result.trace = result.evidence.trace = self.last_trace
        return result

    async def _seeds(self, question, node_id, evidence, ledger):
        """Rank unique source owners without exposing raw retrieval text to models."""
        if node_id is not None:
            ledger.events.append(
                {
                    "kind": "seed_selection",
                    "source": "explicit",
                    "selected": [node_id],
                    "skipped": [],
                }
            )
            return [NodeSeed(node_id, (NodeRef(node_id),))]
        if ledger.searches >= ledger.budget.max_searches:
            raise BudgetExceeded("Initial retrieval search allowance exhausted")
        ledger.searches += 1
        hits = await evidence.search(question, self.retrieval_k)
        if len(hits) > self.retrieval_k:
            raise ConfigurationError("Retriever exceeded requested hit count")
        by_node = {}
        for hit in hits:
            owner = hit.passage.metadata.get("owner_node_id")
            owners = (
                [owner]
                if owner is not None
                else list(dict.fromkeys(reference.node_id for reference in hit.passage.refs))
            )
            for owner in owners:
                NodeRef(owner)
                by_node.setdefault(owner, [])
                for reference in hit.passage.refs:
                    if reference.node_id == owner and reference not in by_node[owner]:
                        by_node[owner].append(reference)
        selected = list(by_node)[: self.max_seed_nodes]
        ledger.events.append(
            {
                "kind": "seed_selection",
                "source": "retrieval",
                "retrieval_k": self.retrieval_k,
                "passage_hits": len(hits),
                "selected": selected,
                "skipped": [
                    {"node_id": owner, "reason": "seed_limit"}
                    for owner in list(by_node)[self.max_seed_nodes :]
                ],
            }
        )
        return [NodeSeed(owner, tuple(by_node[owner])) for owner in selected]
