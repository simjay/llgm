"""Verify real-run admission, uncertainty accounting and durable call records."""

import asyncio
import json
import time

import pytest

from llgm.core.errors import BudgetExceeded
from llgm.evaluation.costs import (
    Allowance,
    make_token_pacer,
    pacing_reservation,
    recorded_model,
    request_reservation,
    usage_cost,
)
from llgm.models.base import (
    CallableModelClient,
    Message,
    ModelRequest,
    ModelResponse,
    Usage,
)

PRICING = {
    "usd_per_million_input_tokens": 2.0,
    "usd_per_million_cached_input_tokens": 0.5,
    "usd_per_million_output_tokens": 8.0,
}
REQUEST = ModelRequest((Message("user", "hello"),), max_output_tokens=20)


@pytest.mark.parametrize("field", ["input_tokens_details", "prompt_tokens_details"])
def test_pricing_counts_cached_input_once(field):
    """Both actual OpenAI transports retain the same complete input accounting."""
    assert usage_cost(Usage(1000, 200, {field: {"cached_tokens": 400}}), PRICING) == pytest.approx(
        0.003
    )


@pytest.mark.parametrize(
    "usage",
    [Usage(), Usage(100, 20), Usage(100, 20, {"input_tokens_details": {"cached_tokens": 101}})],
)
def test_unknown_pricing_stays_unknown(usage):
    """Missing or invalid billing categories do not become a zero estimate."""
    assert usage_cost(usage, PRICING) is None


def test_request_reservation_counts_utf8_schema_and_output():
    """Admission reserves uncached input bytes, framing and the full output limit."""
    request = ModelRequest(
        (Message("user", "한글"),), max_output_tokens=5, output_schema={"type": "object"}
    )
    bound, cost = request_reservation(request, PRICING)
    assert bound == len("한글".encode("utf-8")) + len(b'{"type": "object"}') + 2048
    assert cost == pytest.approx((bound * 2.0 + 5 * 8.0) / 1e6)


def test_concurrent_reservations_prevent_overspending():
    """In-flight requests consume allowance before any provider results return."""

    async def scenario():
        """Race two requests that individually fit but cannot both be admitted."""
        allowance = Allowance(1)
        outcomes = await asyncio.gather(
            allowance.reserve(0.6), allowance.reserve(0.6), return_exceptions=True
        )
        assert sum(isinstance(value, BudgetExceeded) for value in outcomes) == 1
        assert allowance.pending == 0.6
        await allowance.settle(0.6, 0.2, True)
        assert allowance.record()["known_estimated_cost_usd"] == 0.2
        with pytest.raises(BudgetExceeded):
            await allowance.reserve(0.1)

    asyncio.run(scenario())


def test_unknown_reservation_and_broken_bound_stop_new_work():
    """Failed transport liabilities remain counted and cannot be reused."""

    async def scenario():
        """Settle an unknown call, then retain an over-bound actual charge."""
        allowance = Allowance(1)
        await allowance.reserve(0.4)
        await allowance.settle(0.4, None, True)
        assert allowance.record()["estimated_cost_usd"] is None
        assert allowance.unresolved == 0.4
        await allowance.reserve(0.2)
        await allowance.settle(0.2, 0.3, False)
        assert allowance.known_cost == 0.3
        assert allowance.stopped_reason == "returned_usage_exceeds_reservation"

    asyncio.run(scenario())


@pytest.mark.parametrize("cap", [0, -1, float("inf"), float("nan")])
def test_invalid_allowance_is_rejected(cap):
    """Cost admission never accepts an unbounded or nonpositive cap."""
    with pytest.raises(ValueError):
        Allowance(cap)


def test_recording_preserves_failed_response_usage(tmp_path):
    """Truncated provider responses retain priced usage and their exact request."""

    async def scenario():
        """Return a genuine adapter-shaped failure without retry or replacement."""

        async def respond(request):
            """Validate temperature pin at the transport boundary."""
            assert request.temperature == 0
            return ModelResponse(
                "partial",
                Usage(100, 20, {"input_tokens_details": {"cached_tokens": 0}}),
                status="incomplete",
            )

        allowance, trial = Allowance(1), {"model_calls": []}
        client = recorded_model(
            CallableModelClient(respond), "main", PRICING, allowance, trial, tmp_path
        )
        response = await client.complete(REQUEST)
        assert response.status == "incomplete"
        record = json.loads((tmp_path / "call-001.json").read_text())
        assert record["response"]["text"] == "partial"
        assert record["status"] == "incomplete"
        assert record["estimated_cost_usd"] == pytest.approx(0.00036)
        assert allowance.pending == 0
        assert len(trial["model_calls"]) == 1

    asyncio.run(scenario())


def test_recording_explicit_none_temperature_preserves_actual_and_durable_request(tmp_path):
    """Reasoning clients can omit sampling without a recorder silently restoring temperature zero."""

    async def scenario():
        """Check the pre-dispatch file, observed request and settled record against the explicit setting."""
        original = ModelRequest(REQUEST.messages, max_output_tokens=20, temperature=0.8)

        async def respond(request):
            """Read the durable request before returning measured usage from the fake provider."""
            assert request.temperature is None
            pending = json.loads((tmp_path / "call-001.json").read_text())
            assert pending["request"]["temperature"] is None
            return ModelResponse("ok", Usage(2, 1, {"input_tokens_details": {"cached_tokens": 0}}))

        allowance, trial = Allowance(1), {"model_calls": []}
        client = recorded_model(
            CallableModelClient(respond),
            "main",
            PRICING,
            allowance,
            trial,
            tmp_path,
            temperature=None,
        )
        await client.complete(original)
        record = json.loads((tmp_path / "call-001.json").read_text())
        assert record["request"]["temperature"] is None
        assert record["dispatched"] and record["status"] == "completed"
        assert len(trial["model_calls"]) == 1 and allowance.unknown_calls == 0
        assert original.temperature == 0.8

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_transport_failure_and_cancellation_retain_unknown_cost(tmp_path, cancel):
    """Every dispatched failure has a durable request and unreleased reservation."""

    async def scenario():
        """Raise once after admission with no returned provider usage."""

        async def respond(request):
            """Simulate the transport exception boundary, including task cancellation."""
            dispatched = json.loads((tmp_path / "call-001.json").read_text())
            assert dispatched["status"] == "dispatched"
            assert dispatched["request"]["messages"] == [{"role": "user", "content": "hello"}]
            if cancel:
                raise asyncio.CancelledError()
            raise RuntimeError("Do not copy provider exceptions")

        allowance, trial = Allowance(1), {"model_calls": []}
        client = recorded_model(
            CallableModelClient(respond), "reader", PRICING, allowance, trial, tmp_path
        )
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await client.complete(REQUEST)
        record = json.loads((tmp_path / "call-001.json").read_text())
        assert record["status"] == "failed"
        assert record["estimated_cost_usd"] is None
        assert "Do not copy" not in json.dumps(record)
        assert allowance.pending == 0
        assert allowance.unresolved == record["reserved_cost_usd"]
        assert allowance.unknown_calls == 1

    asyncio.run(scenario())


def test_completed_response_with_unknown_usage_keeps_full_reservation(tmp_path):
    """Successful generation does not turn missing provider usage into free work."""

    async def scenario():
        """Return usable text while preserving the request's unresolved liability."""

        async def respond(request):
            """Supply a completed response whose provider omitted token counts."""
            return ModelResponse("answer")

        allowance, trial = Allowance(1, max_unknown_calls=1), {"model_calls": []}
        client = recorded_model(
            CallableModelClient(respond), "main", PRICING, allowance, trial, tmp_path
        )
        response = await client.complete(REQUEST)
        assert response.text == "answer"
        record = json.loads((tmp_path / "call-001.json").read_text())
        assert record["status"] == "completed"
        assert record["estimated_cost_usd"] is None
        assert allowance.pending == 0
        assert allowance.unresolved == record["reserved_cost_usd"]
        assert allowance.record()["estimated_cost_usd"] is None
        with pytest.raises(BudgetExceeded, match="unknown_usage_call_limit"):
            await client.complete(REQUEST)
        assert len(trial["model_calls"]) == 1
        assert not (tmp_path / "call-002.json").exists()

    asyncio.run(scenario())


def test_denied_request_never_reaches_adapter(tmp_path):
    """A rejected reservation is not counted as a provider call."""

    async def scenario():
        """Keep the adapter and request artifact untouched when admission fails."""

        async def respond(request):
            """Fail the test if denied work is dispatched."""
            pytest.fail("Denied request reached provider")

        allowance, trial = Allowance(0.000001), {"model_calls": []}
        client = recorded_model(
            CallableModelClient(respond), "main", PRICING, allowance, trial, tmp_path
        )
        with pytest.raises(BudgetExceeded):
            await client.complete(REQUEST)
        assert not trial["model_calls"]
        assert not list(tmp_path.iterdir())

    asyncio.run(scenario())


def test_deadline_rejects_each_new_request():
    """The global deadline applies to judgment and delegate calls within a trial."""

    async def scenario():
        """Use an expired monotonic deadline with unused dollar allowance."""
        allowance = Allowance(10, deadline=time.monotonic() - 1)
        with pytest.raises(BudgetExceeded):
            await allowance.reserve(0.001)
        assert allowance.pending == 0
        assert allowance.stopped_reason == "run_admission_deadline"

    asyncio.run(scenario())


def test_pacing_reservation_counts_real_tokens_schema_and_output():
    """Unicode and schema are tokenized without confusing bytes with token usage."""
    tiktoken = pytest.importorskip("tiktoken")
    encoding = tiktoken.get_encoding("cl100k_base")
    request = ModelRequest(
        (Message("user", "한글 <|endoftext|> " * 20),),
        max_output_tokens=100,
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    tokens = len(encoding.encode(request.messages[0].content, disallowed_special=()))
    reservation = pacing_reservation(request)
    assert reservation > tokens * 1.1 + request.max_output_tokens
    assert reservation < request_reservation(request, PRICING)[0]
    bigger_output = ModelRequest(
        request.messages, max_output_tokens=200, output_schema=request.output_schema
    )
    assert pacing_reservation(bigger_output) == reservation + 100
    no_schema = ModelRequest(request.messages, max_output_tokens=100)
    assert pacing_reservation(no_schema) < reservation


def test_pacer_waits_for_enough_sliding_window_expirations():
    """Staggered reservations expire independently rather than at minute boundaries."""

    async def scenario():
        """Advance an injected clock instead of sleeping for actual rate windows."""
        now, waits = [0.0], []

        async def sleep(delay):
            """Advance immediately to the requested wake time."""
            waits.append(delay)
            now[0] += delay

        pacer = make_token_pacer(100, clock=lambda: now[0], sleep=sleep)
        assert await pacer(60) == 0
        now[0] = 10
        assert await pacer(20) == 0
        assert await pacer(50) == 50
        assert await pacer(40) == 10
        assert waits == [50, 10]
        with pytest.raises(BudgetExceeded, match="request_exceeds_limit"):
            await pacer(101)

    asyncio.run(scenario())


def test_waiting_large_request_does_not_block_small_request_or_leak_on_cancel():
    """A sleeping full-context admission holds neither the lock nor future tokens."""

    async def scenario():
        """Cancel a large waiter after a smaller request uses the remaining window."""
        now = [0.0]
        sleeping = asyncio.Event()

        async def sleep(delay):
            """Keep the large request waiting until its task is cancelled."""
            assert delay == 50
            sleeping.set()
            await asyncio.Future()

        pacer = make_token_pacer(100, clock=lambda: now[0], sleep=sleep)
        await pacer(80)
        now[0] = 10
        large = asyncio.create_task(pacer(30))
        await sleeping.wait()
        assert await asyncio.wait_for(pacer(20), 1) == 0
        large.cancel()
        with pytest.raises(asyncio.CancelledError):
            await large
        now[0] = 60
        assert await pacer(80) == 0

    asyncio.run(scenario())


def test_pacer_deadline_ends_wait_without_admission():
    """The run deadline can expire before the next 60-second capacity release."""

    async def scenario():
        """Stop waiting exactly at the injected deadline with no wall-clock delay."""
        now, waits = [0.0], []

        async def sleep(delay):
            """Advance the clock to the clipped deadline."""
            waits.append(delay)
            now[0] += delay

        pacer = make_token_pacer(100, clock=lambda: now[0], sleep=sleep)
        await pacer(100)
        with pytest.raises(BudgetExceeded, match="run_admission_deadline"):
            await pacer(10, deadline=20)
        assert waits == [20]
        now[0] = 60
        assert await pacer(100) == 0

    asyncio.run(scenario())


def test_cancellation_while_pacing_is_durable_and_has_no_unknown_charge(tmp_path):
    """A queued request releases dollars and cannot appear as a paid provider call."""
    pytest.importorskip("tiktoken")

    async def scenario():
        """Cancel after the durable wait record, before any adapter invocation."""
        sleeping = asyncio.Event()

        async def sleep(delay):
            """Expose the pacing wait for deterministic cancellation."""
            sleeping.set()
            await asyncio.Future()

        async def respond(request):
            """Reject accidental dispatch of cancelled queued work."""
            pytest.fail("Cancelled pacing request reached provider")

        pacer = make_token_pacer(1000, clock=lambda: 0.0, sleep=sleep)
        await pacer(1000)
        allowance, trial = Allowance(1), {"model_calls": []}
        client = recorded_model(
            CallableModelClient(respond), "main", PRICING, allowance, trial, tmp_path, pacing=pacer
        )
        task = asyncio.create_task(client.complete(REQUEST))
        await sleeping.wait()
        waiting = json.loads((tmp_path / "call-001.json").read_text())
        assert waiting["status"] == "waiting_for_tokens"
        assert waiting["dispatched"] is False
        assert waiting["estimated_cost_usd"] == 0
        assert allowance.pending == waiting["reserved_cost_usd"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        record = json.loads((tmp_path / "call-001.json").read_text())
        assert record["status"] == "not_dispatched"
        assert record["dispatched"] is False
        assert record["pacing_reserved_tokens"] == pacing_reservation(REQUEST)
        assert record["pacing_wait_seconds"] >= 0
        assert allowance.pending == allowance.unresolved == allowance.known_cost == 0
        assert allowance.unknown_calls == 0

    asyncio.run(scenario())


def test_recorded_roles_share_pacing_and_retain_actual_waits(tmp_path):
    """Graph, reader and main requests use the same token admission window."""
    pytest.importorskip("tiktoken")

    async def scenario():
        """Admit two calls, then delay the third until the same window frees."""
        now = [0.0]
        seen = []

        async def sleep(delay):
            """Advance only the injected admission clock."""
            now[0] += delay

        async def respond(request):
            """Verify a durable dispatch and return known usage once per admission."""
            seen.append(now[0])
            record = json.loads((tmp_path / f"call-{len(seen):03d}.json").read_text())
            assert record["status"] == "dispatched"
            assert record["dispatched"] is True
            assert record["estimated_cost_usd"] is None
            return ModelResponse("ok", Usage(10, 1, {"input_tokens_details": {"cached_tokens": 0}}))

        pacer = make_token_pacer(pacing_reservation(REQUEST) * 2, clock=lambda: now[0], sleep=sleep)
        allowance, trial = Allowance(1), {"model_calls": []}
        for role in ("graph", "reader", "main"):
            client = recorded_model(
                CallableModelClient(respond),
                role,
                PRICING,
                allowance,
                trial,
                tmp_path,
                pacing=pacer,
            )
            await client.complete(REQUEST)
        assert seen == [0, 0, 60]
        assert [call["pacing_wait_seconds"] for call in trial["model_calls"]] == [0, 0, 60]
        assert allowance.pending == allowance.unresolved == 0
        assert allowance.known_cost == pytest.approx(0.000084)

    asyncio.run(scenario())


def test_paced_request_rechecks_dollar_stop_before_dispatch(tmp_path):
    """A request queued before another call stops admission must not reach the SDK."""
    pytest.importorskip("tiktoken")

    async def scenario():
        """Trigger the uncertainty guard while the queued request waits for tokens."""
        allowance, trial = Allowance(1, max_unknown_calls=1), {"model_calls": []}

        async def pacing(tokens, *, deadline):
            """Settle a separate already-dispatched call with missing usage."""
            await allowance.reserve(0.1)
            await allowance.settle(0.1, None, True)
            return 5.0

        async def respond(request):
            """Reject dispatch after the shared dollar allowance stopped admission."""
            pytest.fail("Stopped allowance reached provider")

        client = recorded_model(
            CallableModelClient(respond), "main", PRICING, allowance, trial, tmp_path, pacing=pacing
        )
        with pytest.raises(BudgetExceeded, match="unknown_usage_call_limit"):
            await client.complete(REQUEST)
        record = trial["model_calls"][0]
        assert record["dispatched"] is False
        assert record["estimated_cost_usd"] == 0
        assert record["pacing_wait_seconds"] == 5
        assert allowance.pending == 0
        assert allowance.unknown_calls == 1
        assert allowance.unresolved == 0.1

    asyncio.run(scenario())


def test_dispatched_failure_keeps_token_window_and_unknown_dollar_reservation(tmp_path):
    """A provider failure refunds neither estimated rate tokens nor unknown cost."""
    pytest.importorskip("tiktoken")

    async def scenario():
        """Make one failed physical call, then require the next call to wait a minute."""
        now, calls = [0.0], []

        async def sleep(delay):
            """Advance the pacing window without a real delay."""
            now[0] += delay

        async def respond(request):
            """Fail after the first dispatch and provide known usage on the second."""
            calls.append(now[0])
            if len(calls) == 1:
                raise RuntimeError("transport failure")
            return ModelResponse("ok", Usage(10, 1, {"input_tokens_details": {"cached_tokens": 0}}))

        pacer = make_token_pacer(pacing_reservation(REQUEST), clock=lambda: now[0], sleep=sleep)
        allowance, trial = Allowance(1), {"model_calls": []}
        client = recorded_model(
            CallableModelClient(respond), "main", PRICING, allowance, trial, tmp_path, pacing=pacer
        )
        with pytest.raises(RuntimeError):
            await client.complete(REQUEST)
        failed = trial["model_calls"][0]
        assert failed["dispatched"] is True
        assert failed["estimated_cost_usd"] is None
        assert allowance.unresolved == failed["reserved_cost_usd"]
        await client.complete(REQUEST)
        assert calls == [0, 60]
        assert trial["model_calls"][1]["pacing_wait_seconds"] == 60
        assert allowance.unknown_calls == 1
        assert allowance.unresolved == failed["reserved_cost_usd"]

    asyncio.run(scenario())
