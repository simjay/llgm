"""Shared model admission, execution limits, and observable usage accounting."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.models import ModelRequest


def byte_token_bound(text: str) -> int:
    """Conservative byte-count allowance, explicitly not a provider tokenizer."""
    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class Budget:
    """Positive per-run limits. Token allowances use the configured counter."""

    max_model_calls: int = 9
    max_reader_calls: int = 8
    max_graph_calls: int = 8
    max_searches: int = 4
    max_evidence_tokens: int = 8000
    max_bundle_tokens: int = 4000
    timeout_seconds: float = 120.0
    max_output_tokens: int = 1024
    max_context_tokens: int = 16000

    def __post_init__(self):
        """Reject nonpositive, nonfinite, or incorrectly typed execution limits."""
        for name, value in asdict(self).items():
            if name == "timeout_seconds":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not (0 < value < float("inf"))
                ):
                    raise ConfigurationError("timeout_seconds must be finite and positive")
            elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")


class RunLedger:
    """Shared admission limits and accounting for every model invocation in a run."""

    def __init__(self, budget: Budget, token_counter: Callable[[str], int], reserve_main=False):
        """Start run accounting, optionally reserving a model call for the main."""
        self.budget = budget
        self.counter = token_counter
        self.reserve_main = reserve_main
        self.started = time.monotonic()
        self.calls = 0
        self.reader_calls = 0
        self.graph_calls = 0
        self.searches = 0
        self.exposed_tokens = 0
        self.events: list[dict] = []

    def remaining_seconds(self):
        """Return remaining wall time or reject work after the deadline."""
        remaining = self.budget.timeout_seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            raise BudgetExceeded("Run deadline exhausted")
        return remaining

    def count(self, text):
        """Count admission units, rejecting counter results that could bypass a limit."""
        amount = self.counter(text)
        if type(amount) is not int or amount < 0:
            raise ConfigurationError("Token counter must return a nonnegative integer")
        return amount

    def context_size(self, messages, *, output_schema=None):
        """Count serialized messages and schema under the declared admission counter.

        Provider-specific framing is not modeled. Actual token usage is recorded
        separately. A native schema still contributes to the input allowance.
        """
        amount = self.count(json.dumps([asdict(m) for m in messages], ensure_ascii=False))
        if output_schema is not None:
            amount += self.count(json.dumps(output_schema, ensure_ascii=False, allow_nan=False))
        return amount

    def check_admission(self, messages, role="reader", *, output_schema=None):
        """Check a possible invocation without spending a call, returning its input size."""
        if not isinstance(role, str) or role not in {"main", "reader", "graph"}:
            raise ConfigurationError("Model role must be main, reader, or graph")
        self.remaining_seconds()
        allowance = self.budget.max_model_calls - (self.reserve_main and role != "main")
        if self.calls >= allowance:
            raise BudgetExceeded("Model-call allowance exhausted")
        if role == "reader" and self.reader_calls >= self.budget.max_reader_calls:
            raise BudgetExceeded("Reader-call allowance exhausted")
        if role == "graph" and self.graph_calls >= self.budget.max_graph_calls:
            raise BudgetExceeded("Graph-call allowance exhausted")
        context = self.context_size(messages, output_schema=output_schema)
        if context + self.budget.max_output_tokens > self.budget.max_context_tokens:
            raise BudgetExceeded("Invocation context allowance exceeded")
        return context

    async def call(self, model, messages, role="reader", *, output_schema=None, event_context=None):
        """Admit one model call and retain usage or failure status without retrying."""
        context = self.check_admission(messages, role, output_schema=output_schema)
        self.calls += 1
        self.reader_calls += role == "reader"
        self.graph_calls += role == "graph"
        event = {
            "kind": "model",
            "role": role,
            "attempt": self.calls,
            "input_tokens": None,
            "output_tokens": None,
            "status": "attempted",
            "context_accounting_units": context,
            **dict(event_context or {}),
        }
        self.events.append(event)
        try:
            async with asyncio.timeout(self.remaining_seconds()):
                response = await model.complete(
                    ModelRequest(
                        messages=tuple(messages),
                        max_output_tokens=self.budget.max_output_tokens,
                        output_schema=output_schema,
                    )
                )
            event.update(
                provider=response.provider,
                model=response.model,
                request_id=response.request_id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                usage_extra=response.usage.extra,
                response_metadata=response.metadata,
                structured_output=output_schema is not None,
                status=response.status,
            )
            response.ensure_complete()
            return response.text
        except TimeoutError:
            event["status"] = "timeout"
            raise BudgetExceeded("Run deadline exhausted during model call") from None
        except asyncio.CancelledError:
            event["status"] = "cancelled"
            raise
        except Exception as error:
            if event["status"] == "attempted":
                event["status"] = "failed"
            event["error_type"] = type(error).__name__
            raise

    def usage(self):
        """Aggregate known usage while keeping unknown counts and unpriced cost explicit."""
        calls = [event for event in self.events if event["kind"] == "model"]
        return {
            "model_calls": self.calls,
            "reader_calls": self.reader_calls,
            "graph_calls": self.graph_calls,
            "searches": self.searches,
            "evidence_accounting_units": self.exposed_tokens,
            "accounting_tokenizer": getattr(
                self.counter, "name", getattr(self.counter, "__name__", "custom")
            ),
            "known_input_tokens": sum(e["input_tokens"] or 0 for e in calls),
            "known_output_tokens": sum(e["output_tokens"] or 0 for e in calls),
            "unknown_usage_calls": sum(
                e["input_tokens"] is None or e["output_tokens"] is None for e in calls
            ),
            "elapsed_seconds": time.monotonic() - self.started,
            "currency_cost": None,
        }
