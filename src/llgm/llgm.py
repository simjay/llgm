"""Persistent ingestion, bounded graph maintenance, and question-driven inference.

The application owns per-answer evidence handles, while the caller owns its
Workspace, reusable index and model clients. Evidence reads apply complete local journals.
Automatic publication validates evidence for generic connections.
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
from llgm.core.errors import BudgetExceeded, CapabilityError, ConfigurationError, SchemaError
from llgm.core.time import validate_instant_ms
from llgm.core.types import (
    Conversation,
    Edge,
    IngestResult,
    NodeRef,
    Turn,
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
from llgm.memory.maintenance import MaintenancePolicy, MaintenanceResult, _GraphClient
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
        main_model: ModelClient,
        reader_model: ModelClient,
        *,
        graph_model: ModelClient | None,
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
        Supply a graph client explicitly. With maintenance disabled,
        ``graph_model=None`` is allowed.
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
        self.main_model, self.reader_model = main_model, reader_model
        self.graph_model = graph_model
        self.maintenance_policy = maintenance_policy or MaintenancePolicy()
        self.inference_budget = inference_budget or Budget(
            max_model_calls=40,
            max_reader_calls=36,
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
        self._conversation_lock = getattr(workspace, "_conversation_lock", asyncio.Lock())

    @classmethod
    @asynccontextmanager
    async def from_settings(
        cls,
        settings: Settings | None = None,
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

        Use ``async with LLGM.from_settings() as memory`` to read environment
        settings when the context is entered. An explicit ``Settings`` instance
        is used unchanged. All three model IDs must be configured.
        ``runtime_options`` supplies constructor options
        such as ``max_depth`` and ``max_steps``. Settings supply storage, providers,
        retrieval limits, concurrency, and the answer budget.
        """
        from llgm.models import create_model

        if settings is None:
            settings = Settings.from_env()
        if not all(
            isinstance(getattr(settings, role + "_model", None), str)
            and getattr(settings, role + "_model").strip()
            for role in ("main", "reader", "graph")
        ):
            raise ConfigurationError(
                "Set explicit main_model, reader_model, and graph_model for the application"
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
            for role in ("main", "reader", "graph"):
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
                graph_model=clients[2],
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
        conversation: Conversation | str | Sequence[Mapping[str, Any] | Turn],
        *,
        idempotency_key: str | None = None,
        organize: bool = True,
        conversation_id: str = "default",
    ) -> IngestionOutcome:
        """Catch up on an earlier conversation batch with automatic topic routing.

        :param conversation: A string, a sequence of chat turns, or a
            ``Conversation`` with source metadata. A string becomes one user
            turn. Sequences accept ``Turn`` records or mappings with ``role``
            and either ``text`` or ``content``. Text is preserved exactly.
        :param idempotency_key: Reuse the source when retrying identical input.
            Reusing a key with different input raises a conflict.
        :param organize: Discover connections after storing the batch. False
            skips connection discovery but still permits model topic routing.
        :param conversation_id: Persistent chat whose topic this import continues.
            A supplied Conversation.node_id bypasses routing for an exact import.
        :returns: The stored source ID and a separate maintenance outcome.
            Maintenance failure does not remove the source. Retrying ingestion
            can run maintenance again even when the source already exists.
        """
        if isinstance(conversation, str):
            conversation = Conversation.from_turns([{"role": "user", "text": conversation}])
        elif not isinstance(conversation, Conversation):
            if (
                not isinstance(conversation, Sequence)
                or isinstance(conversation, (bytes, bytearray))
                or any(not isinstance(turn, (Mapping, Turn)) for turn in conversation)
            ):
                raise SchemaError("ingest requires text, a sequence of turns, or a Conversation")
            conversation = Conversation.from_turns(list(conversation))
        if conversation.node_id is None:
            async with self._conversation_lock:
                source, routing = await self._remember(
                    conversation, conversation_id, idempotency_key=idempotency_key
                )
            maintenance = await self.organize([source.node_id], disabled=not organize)
            maintenance = replace(
                maintenance,
                usage={**maintenance.usage, "topic_routing": routing.usage()},
                trace=(*routing.events, *maintenance.trace),
            )
            return IngestionOutcome(source, maintenance)
        source = await self.workspace.ingest(conversation, idempotency_key=idempotency_key)
        maintenance = await self.organize([source.node_id], disabled=not organize)
        return IngestionOutcome(source, maintenance)

    async def _remember(self, conversation, conversation_id, *, idempotency_key=None, ledger=None):
        """Route and append incoming turns while retaining their original attribution."""
        from llgm.memory.conversation import append_fingerprint
        from llgm.memory.topics import choose_topic

        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ConfigurationError("conversation_id must be nonempty text")
        ledger = ledger or RunLedger(self.maintenance_policy.budget, byte_token_bound)
        self.workspace._key(idempotency_key)
        if idempotency_key is not None:
            prior = await self.workspace._run(
                lambda: self.workspace._retry(
                    "conversation_append",
                    idempotency_key,
                    append_fingerprint(conversation_id, conversation),
                )
            )
            if prior is not None:
                return IngestResult(**prior, created=False), ledger
        async with asyncio.timeout(ledger.remaining_seconds()):
            if self.maintenance_policy.mode == "disabled":
                selected = await self.workspace.conversation_node(conversation_id)
            else:
                async with await self._open_evidence() as evidence:
                    selected = await choose_topic(
                        self.workspace,
                        evidence,
                        self.graph_model,
                        ledger,
                        conversation_id,
                        conversation.turns,
                    )
            source = await self.workspace.append_conversation(
                conversation_id, conversation, node_id=selected, idempotency_key=idempotency_key
            )
        return source, ledger

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
                        model = _GraphClient(self.graph_model, ledger)
                        for node_id in selected[: policy.max_nodes]:
                            ledger.remaining_seconds()
                            if len(visible) < 2:
                                continue
                            if ledger.searches >= policy.budget.max_searches:
                                raise BudgetExceeded("Maintenance search allowance exhausted")
                            prefix = await evidence.source_prefix(node_id, policy.max_context_chars)
                            ledger.searches += 1
                            hits = await evidence.search(
                                " ".join(record.text for record in prefix),
                                policy.max_candidates * 4,
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
                                    "decision": reason,
                                }
                                if reason == "publish":
                                    key = hashlib.sha256(
                                        json.dumps(
                                            {
                                                "source": reference_to_dict(proposal.source),
                                                "target": reference_to_dict(proposal.target),
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
        if any(
            edge.source_node_id == proposal.source.node_id
            and edge.target_node_id == proposal.target.node_id
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
        question: str | Sequence[Mapping[str, Any] | Turn],
        *,
        conversation_id: str = "default",
        remember: bool = True,
        scope: Mapping[str, Any] | None = None,
        query_date: str | None = None,
        as_of_ms: int | None = None,
        budget: Budget | None = None,
        node_id: str | None = None,
    ) -> AnswerResult:
        """Continue a persistent conversation and return an answer with evidence.

        :param question: New user text or new role/content turns ending in a user
            message. Send only new turns, not the accumulated transcript.
        :param conversation_id: Stable chat identity, resumed across application restarts.
            Topic changes can select a different node within the same chat.
        :param remember: Save incoming turns and the returned assistant text.
            False performs a read-only memory question and requires plain text.
        :param scope: Declared evidence applicability values, not access control.
        :param query_date: Human-readable date for interpreting evidence.
        :param as_of_ms: Declared evidence validity instant in Unix milliseconds.
        :param budget: Complete shared routing and inference allowance for this call.
        :param node_id: Explicit initial reading node for a read-only answer.
        :returns: Answer, source references, status, usage and active topic identity.

        Stored user turns survive generation failure. Only nonempty returned
        assistant text is appended. Calls on this instance are serialized.
        A repeated call is a new message, including after a failed generation.
        """
        if type(remember) is not bool:
            raise ConfigurationError("remember must be boolean")
        if not remember:
            return await self._answer(
                question,
                scope=scope,
                query_date=query_date,
                as_of_ms=as_of_ms,
                budget=budget,
                node_id=node_id,
            )
        if node_id is not None:
            raise ConfigurationError("node_id requires remember=False")
        if scope is not None and not isinstance(scope, Mapping):
            raise ConfigurationError("scope must be a mapping")
        if query_date is not None and (not isinstance(query_date, str) or not query_date.strip()):
            raise ConfigurationError("query_date must be nonempty text or None")
        validate_instant_ms(as_of_ms, "as_of_ms")
        if isinstance(question, str):
            turns = [{"role": "user", "content": question}]
        elif isinstance(question, Sequence) and not isinstance(question, (bytes, bytearray)):
            turns = list(question)
        else:
            raise SchemaError("answer requires text or new chat turns")
        if any(not isinstance(turn, (Mapping, Turn)) for turn in turns):
            raise SchemaError("answer requires role/content turns")
        conversation = Conversation.from_turns(
            turns, metadata={"date": query_date} if query_date else {}
        )
        if (
            conversation.turns[-1].role != "user"
            or not conversation.turns[-1].text.strip()
            or any(turn.role not in {"user", "assistant"} for turn in conversation.turns)
        ):
            raise SchemaError("answer requires user/assistant turns ending with nonempty user text")
        async with self._conversation_lock:
            conversation_started = time.monotonic()
            ledger = RunLedger(budget or self.inference_budget, byte_token_bound, reserve_main=True)
            try:
                source, ledger = await self._remember(conversation, conversation_id, ledger=ledger)
            except BaseException:
                self.last_usage, self.last_trace = ledger.usage(), list(ledger.events)
                raise
            routing_seconds = time.monotonic() - conversation_started
            result = await self._answer(
                conversation.turns[-1].text,
                scope=scope,
                query_date=query_date,
                as_of_ms=as_of_ms,
                budget=budget,
                active_node_id=source.node_id,
                ledger=ledger,
            )
            result.conversation_id, result.node_id = conversation_id, source.node_id
            if result.answer:
                await self.workspace.append_conversation(
                    conversation_id,
                    Conversation.from_turns(
                        [{"role": "assistant", "content": result.answer}],
                        metadata=conversation.metadata,
                    ),
                    node_id=source.node_id,
                )
            if source.created:
                maintenance = await self.organize([source.node_id])
                result.usage["maintenance"] = dict(maintenance.usage)
                result.trace.append(
                    {
                        "kind": "conversation_maintenance",
                        "status": maintenance.status,
                        "node_id": source.node_id,
                        "error_type": maintenance.error_type,
                    }
                )
            result.usage.update(
                routing_seconds=routing_seconds,
                total_seconds=time.monotonic() - conversation_started,
                elapsed_seconds=time.monotonic() - conversation_started,
            )
            return result

    async def _answer(
        self,
        question: str,
        *,
        scope: Mapping[str, Any] | None = None,
        query_date: str | None = None,
        as_of_ms: int | None = None,
        budget: Budget | None = None,
        node_id: str | None = None,
        active_node_id: str | None = None,
        ledger: RunLedger | None = None,
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
        preparation_ledger = ledger or RunLedger(
            selected_budget, byte_token_bound, reserve_main=True
        )
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
                seeds = await self._seeds(
                    question, node_id, evidence, preparation_ledger, active_node_id=active_node_id
                )
                if active_node_id is not None:
                    page = await self.workspace.source_info(active_node_id, limit=1)
                    page = await self.workspace.source_info(
                        active_node_id, offset=max(0, page["total_turns"] - 4), limit=4
                    )
                    from llgm.core.types import reference_from_dict

                    refs = tuple(reference_from_dict(item["reference"]) for item in page["turns"])
                    existing = next(
                        (seed.references for seed in seeds if seed.node_id == active_node_id), ()
                    )
                    seeds = [
                        NodeSeed(active_node_id, tuple(dict.fromkeys((*refs, *existing)))),
                        *(seed for seed in seeds if seed.node_id != active_node_id),
                    ]
                    # The active topic occupies one seed slot even for a follow-up
                    # whose wording has no lexical match with earlier evidence.
                    seeds = seeds[: self.max_seed_nodes]
                    preparation_ledger.events.append(
                        {"kind": "conversation_seed", "node_id": active_node_id}
                    )
                prepared_at = time.monotonic()
                preparation.update(status="completed", index=dict(evidence.preparation))
                remaining = selected_budget.timeout_seconds - (prepared_at - started)
                # A synchronous index build cannot be interrupted by asyncio;
                # recheck wall time before dispatching any model request.
                if remaining <= 0:
                    preparation["status"] = "budget_exhausted"
                    raise BudgetExceeded("Answer preparation exhausted the run deadline")
                runtime = NodeRuntime(
                    self.main_model,
                    self.reader_model,
                    scoped,
                    budget=replace(selected_budget, timeout_seconds=remaining),
                    max_depth=self.max_depth,
                    max_steps=self.max_steps,
                    max_operations=self.max_operations,
                    max_concurrency=self.max_concurrency,
                    repl_config=self.repl_config,
                    repl_factory=self.repl_factory,
                    capture_text=self.capture_text,
                    conversational=active_node_id is not None,
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

    async def _seeds(self, question, node_id, evidence, ledger, *, active_node_id=None):
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
        ranked = list(by_node)
        if active_node_id is not None:
            ranked = [active_node_id, *(owner for owner in ranked if owner != active_node_id)]
            by_node.setdefault(active_node_id, [])
        selected = ranked[: self.max_seed_nodes]
        ledger.events.append(
            {
                "kind": "seed_selection",
                "source": "retrieval",
                "retrieval_k": self.retrieval_k,
                "passage_hits": len(hits),
                "selected": selected,
                "skipped": [
                    {"node_id": owner, "reason": "seed_limit"}
                    for owner in ranked[self.max_seed_nodes :]
                ],
            }
        )
        return [NodeSeed(owner, tuple(by_node[owner])) for owner in selected]
