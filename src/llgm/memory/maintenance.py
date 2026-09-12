"""Primary-edge proposal policy, results, and bounded graph model calls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.core.types import Edge
from llgm.inference.budget import Budget, RunLedger, byte_token_bound
from llgm.memory.evidence import LinkProposal
from llgm.models.base import ModelClient, ModelRequest, ModelResponse, Usage


@dataclass(frozen=True)
class MaintenancePolicy:
    """Declared limits and acceptance rules for one graph-maintenance operation.

    ``validated`` publishes structurally supported generic connections.
    ``propose`` returns candidates for caller review. ``disabled`` makes no model
    or search calls. Original sources remain committed if maintenance fails.
    """

    mode: str = "validated"
    max_nodes: int = 8
    max_candidates: int = 8
    max_links_per_node: int = 4
    max_context_chars: int = 16000
    budget: Budget = field(
        default_factory=lambda: Budget(
            max_model_calls=8,
            max_reader_calls=8,
            max_searches=8,
            max_context_tokens=65536,
            max_output_tokens=2048,
            timeout_seconds=120,
        )
    )

    def __post_init__(self) -> None:
        """Reject undefined policy modes and nonpositive work limits."""
        if self.mode not in {"validated", "propose", "disabled"}:
            raise ConfigurationError("Maintenance mode must be validated, propose, or disabled")
        for name in ("max_nodes", "max_candidates", "max_links_per_node", "max_context_chars"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        if not isinstance(self.budget, Budget):
            raise ConfigurationError("Maintenance budget must be a Budget")


@dataclass(frozen=True)
class MaintenanceResult:
    """Retained proposals, committed links, policy decisions, and attempted usage."""

    status: str
    proposals: tuple[LinkProposal, ...] = ()
    accepted: tuple[Edge, ...] = ()
    decisions: tuple[Mapping[str, Any], ...] = ()
    deferred_node_ids: tuple[str, ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    trace: tuple[Mapping[str, Any], ...] = ()
    error_type: str | None = None


class _GraphClient:
    """Route proposal generation through the shared maintenance admission ledger."""

    def __init__(self, model: ModelClient, ledger: RunLedger):
        """Bind an existing client without taking ownership of its connection."""
        self.model, self.ledger = model, ledger
        self.capabilities = model.capabilities
        self._exposed: set[str] = set()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Admit unique presented evidence before a counted proposal call.

        Repeated identical spans count once against evidence transfer allowance.
        Provider input usage still accounts for every repeated presentation.
        Failed or cancelled admitted calls retain the evidence charge.
        """
        payload = json.loads(request.messages[-1].content)
        newly_exposed = {}
        for passage in payload["passages"]:
            encoded = json.dumps(passage, ensure_ascii=False, sort_keys=True)
            identity = hashlib.sha256(encoded.encode()).hexdigest()
            if identity not in self._exposed:
                newly_exposed[identity] = byte_token_bound(encoded)
        amount = sum(newly_exposed.values())
        if self.ledger.exposed_tokens + amount > self.ledger.budget.max_evidence_tokens:
            raise BudgetExceeded("Maintenance evidence exposure allowance exhausted")
        before_calls = self.ledger.calls
        messages = list(request.messages)
        try:
            text = await self.ledger.call(
                self.model, messages, role="graph", output_schema=request.output_schema
            )
        finally:
            # Rejected admission transfers nothing. An attempted request may
            # already have reached the provider even if its response is lost.
            if self.ledger.calls > before_calls:
                self._exposed.update(newly_exposed)
                self.ledger.exposed_tokens += amount
                event = next(
                    event for event in reversed(self.ledger.events) if event["kind"] == "model"
                )
                event["new_evidence_accounting_units"] = amount
        event = next(event for event in reversed(self.ledger.events) if event["kind"] == "model")
        return ModelResponse(
            text,
            Usage(event["input_tokens"], event["output_tokens"], event.get("usage_extra", {})),
            provider=event["provider"],
            model=event["model"],
            request_id=event.get("request_id"),
        )

    def descriptor(self) -> Mapping[str, Any]:
        """Expose the underlying model identity without credentials."""
        return self.model.descriptor()
