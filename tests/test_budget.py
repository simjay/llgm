"""Shared model admission, provider usage and cancellation accounting."""

import asyncio
import json

import pytest

from llgm import Budget
from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.inference.budget import RunLedger
from llgm.models import CallableModelClient, Message, ModelResponse, ScriptedModelClient, Usage


def test_cache_usage_categories_survive_the_runtime_ledger():
    """Cached-input usage stays visible without doubling total input tokens."""

    async def scenario():
        """Account for one cached response and one response with unavailable usage."""
        model = ScriptedModelClient(
            [ModelResponse("cached", Usage(100, 20, {"cache_read_input_tokens": 80})), "unknown"]
        )
        ledger = RunLedger(Budget(), len)
        for _ in range(2):
            await ledger.call(model, [Message("user", "Question")])
        assert ledger.events[0]["usage_extra"]["cache_read_input_tokens"] == 80
        assert ledger.usage()["known_input_tokens"] == 100
        assert ledger.usage()["known_output_tokens"] == 20
        assert ledger.usage()["unknown_usage_calls"] == 1

    asyncio.run(scenario())


def test_cancellation_retains_the_dispatched_attempt():
    """Cancellation preserves the attempted call and records unavailable provider usage."""

    async def scenario():
        """Cancel after dispatch and retain accounting when no response arrives."""
        entered = asyncio.Event()

        async def waiting(request):
            """Signal provider dispatch and wait for cancellation."""
            entered.set()
            await asyncio.Event().wait()

        ledger = RunLedger(Budget(), len)
        task = asyncio.create_task(
            ledger.call(CallableModelClient(waiting), [Message("user", "Question")])
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ledger.usage()["model_calls"] == 1
        assert ledger.usage()["unknown_usage_calls"] == 1
        assert ledger.events[-1]["status"] == "cancelled"

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", [-1, True, 1.5, float("nan"), None])
def test_invalid_accounting_cannot_admit_a_model_call(invalid):
    """Invalid custom counter results fail before model work or a charged attempt."""

    async def scenario():
        """Reject malformed accounting even when it would make the input appear to fit."""
        client = ScriptedModelClient(["must not dispatch"])
        ledger = RunLedger(Budget(), lambda text: invalid)
        with pytest.raises(ConfigurationError, match="nonnegative integer"):
            await ledger.call(client, [Message("user", "Question")])
        assert not client.requests
        assert ledger.calls == 0 and not ledger.events

    asyncio.run(scenario())


def test_native_schema_is_charged_before_dispatch_at_the_context_boundary():
    """A schema that crosses the input allowance consumes no model call or request."""
    from llgm.core.errors import BudgetExceeded
    from llgm.inference.budget import RunLedger, byte_token_bound
    from llgm.models import Message, ModelCapabilities

    async def scenario():
        """Reject one unit over budget and accept the exact combined input boundary."""
        requests = []

        async def respond(request):
            """Record admitted requests and return a completed synthetic schema response."""
            requests.append(request)
            return ModelResponse("{}")

        client = CallableModelClient(
            respond, capabilities=ModelCapabilities(structured_output=True)
        )
        messages = [Message("user", "A")]
        schema = {"type": "object", "description": "x" * 5000}
        size = len(json.dumps([{"role": "user", "content": "A"}]).encode())
        size += len(json.dumps(schema).encode())
        denied = RunLedger(Budget(max_context_tokens=size, max_output_tokens=1), byte_token_bound)
        with pytest.raises(BudgetExceeded, match="context allowance"):
            await denied.call(client, messages, output_schema=schema)
        assert not requests and denied.calls == 0 and not denied.events
        admitted = RunLedger(
            Budget(max_context_tokens=size + 1, max_output_tokens=1), byte_token_bound
        )
        assert await admitted.call(client, messages, output_schema=schema) == "{}"
        assert len(requests) == 1 and requests[0].output_schema == schema
        assert admitted.events[0]["context_accounting_units"] == size
        assert admitted.events[0]["structured_output"] is True

    asyncio.run(scenario())


def test_maintenance_and_reader_allowances_are_independent():
    """Each helper role has its own cap while both spend the total run allowance."""

    async def scenario():
        """Admit one call per role and reject further calls before reaching the client."""
        ledger = RunLedger(Budget(max_model_calls=3, max_reader_calls=1, max_graph_calls=1), len)
        model = ScriptedModelClient(["ok", "ok", "ok"])
        await ledger.call(model, [Message("user", "test")], role="graph")
        await ledger.call(model, [Message("user", "test")], role="reader")
        for role in ("graph", "reader"):
            with pytest.raises(BudgetExceeded):
                await ledger.call(model, [Message("user", "test")], role=role)
        await ledger.call(model, [Message("user", "test")], role="main")
        assert len(model.requests) == 3
        assert ledger.usage()["graph_calls"] == 1
        assert ledger.usage()["reader_calls"] == 1
        with pytest.raises(BudgetExceeded):
            await ledger.call(model, [Message("user", "test")], role="main")

    asyncio.run(scenario())


@pytest.mark.parametrize("role", ["root", "sidecar", "maintenance", "query", "", None])
def test_unknown_model_roles_fail_before_spending_budget(role):
    """Old or misspelled role names cannot bypass per-role admission limits."""

    async def scenario():
        """Reject a malformed role before any client dispatch or ledger mutation."""
        ledger = RunLedger(Budget(), len)
        client = ScriptedModelClient(["unreachable"])
        with pytest.raises(ConfigurationError, match="Model role"):
            await ledger.call(client, [Message("user", "test")], role=role)
        assert client.requests == []
        assert ledger.calls == 0 and ledger.events == []

    asyncio.run(scenario())
