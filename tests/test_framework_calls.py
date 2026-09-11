"""Verify physical framework-call accounting with real SDK parsing and no network."""

import asyncio
import json
import threading
from importlib import import_module

import pytest

from llgm.evaluation.costs import Allowance

httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")
pytest.importorskip("tiktoken")
framework_openai_client = import_module("llgm.evaluation.framework_calls").framework_openai_client

PRICING = {
    "usd_per_million_input_tokens": 2,
    "usd_per_million_cached_input_tokens": 0.5,
    "usd_per_million_output_tokens": 8,
}


def generation_response(**overrides):
    """Return a real Chat Completions envelope with independently known billing."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "not JSON"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
        **overrides,
    }


def client_for(path, handler, allowance, latch, *, kind="generation", **kwargs):
    """Create the configured SDK against only the supplied local mock transport."""
    return framework_openai_client(
        api_key="test-secret-key",
        model="test-model",
        kind=kind,
        pricing=PRICING if kind == "generation" else {"usd_per_million_input_tokens": 0.02},
        allowance=allowance,
        pacing=kwargs.pop("pacing", None),
        loop=asyncio.get_running_loop(),
        path=path,
        call_metadata={"case_id": "case-opaque", "arm": "mem0", "phase": "construction"},
        failure_latch=latch,
        max_output_tokens=20,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def complete(client, **kwargs):
    """Perform one bounded synchronous SDK call from a test worker."""
    return client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=20,
        **kwargs,
    )


def records(path):
    """Read the durable physical attempts rather than an in-memory aggregate."""
    return [json.loads(p.read_text()) for p in sorted(path.glob("call-*.json"))]


def test_raw_call_survives_later_framework_json_parse_failure(tmp_path):
    """A framework parser cannot erase a successful provider response or its cost."""

    async def scenario():
        """Parse deliberately invalid generated JSON after the transport checkpoint."""
        seen = []

        def handler(request):
            """Require a predispatch record and return a standard SDK response."""
            seen.append(json.loads(request.content))
            assert records(tmp_path)[0]["status"] == "dispatched"
            return httpx.Response(200, json=generation_response(), headers={"x-secret": "hidden"})

        allowance, latch = Allowance(1), threading.Event()
        with client_for(tmp_path, handler, allowance, latch) as client:
            assert client.max_retries == 0
            response = await asyncio.to_thread(complete, client)
            with pytest.raises(ValueError):
                json.loads(response.choices[0].message.content)
        record = records(tmp_path)[0]
        assert record["request"] == seen[0]
        assert json.loads(record["request_body"]) == seen[0]
        assert record["response"] == generation_response()
        assert record["estimated_cost_usd"] == pytest.approx(0.00022)
        assert allowance.known_cost == pytest.approx(0.00022)
        assert allowance.pending == allowance.unresolved == 0
        assert "test-secret-key" not in json.dumps(record)
        assert "hidden" not in json.dumps(record)
        assert not latch.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["http", "transport", "non_json", "unknown_usage", "body_read"])
def test_physical_failure_keeps_liability_and_denies_framework_fallback(tmp_path, failure):
    """A swallowed physical failure cannot trigger another paid fallback request."""

    async def scenario():
        """Try the SDK twice but observe only one actual transport dispatch."""
        calls = []

        class BrokenBody(httpx.SyncByteStream):
            """Fail while the already-dispatched response body is being consumed."""

            def __iter__(self):
                """Raise a transport read failure before a usable body arrives."""
                raise httpx.ReadError("do not retain transport error details")
                yield b""

        def handler(request):
            """Return one failure below the real OpenAI SDK's parser."""
            calls.append(request)
            if failure == "transport":
                raise httpx.ConnectError("do not retain transport error details")
            if failure == "http":
                return httpx.Response(
                    429, json={"error": {"message": "test-secret-key", "type": "rate_limit"}}
                )
            if failure == "non_json":
                return httpx.Response(
                    200, content="not JSON", headers={"content-type": "application/json"}
                )
            if failure == "body_read":
                return httpx.Response(200, stream=BrokenBody())
            return httpx.Response(200, json=generation_response(usage=None))

        allowance, latch = Allowance(1), threading.Event()
        with client_for(tmp_path, handler, allowance, latch) as client:
            try:
                await asyncio.to_thread(complete, client)
            except Exception:
                pass
            assert latch.is_set()
            with pytest.raises(openai.APIConnectionError):
                await asyncio.to_thread(complete, client)
        assert len(calls) == len(records(tmp_path)) == 1
        record = records(tmp_path)[0]
        assert record["status"] == "failed"
        assert record["dispatched"] is True
        assert record["estimated_cost_usd"] is None
        assert allowance.unknown_calls == 1
        assert allowance.unresolved == record["reserved_cost_usd"]
        assert allowance.pending == 0
        assert "test-secret-key" not in json.dumps(record)
        assert "do not retain transport error details" not in json.dumps(record)
        if failure == "http":
            assert "response_body" not in record
            assert record["response"] == {"error": {"type": "rate_limit"}}

    asyncio.run(scenario())


def test_embedding_usage_needs_no_generation_or_cached_token_fields(tmp_path):
    """Embedding prompt_tokens alone determines its input-only physical-call price."""

    async def scenario():
        """Complete an actual SDK embedding parse with no cached/output usage categories."""

        def handler(request):
            """Return a small numeric vector and provider embedding usage."""
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "test-model",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "usage": {"prompt_tokens": 12, "total_tokens": 12},
                },
            )

        allowance, latch = Allowance(1), threading.Event()
        with client_for(tmp_path, handler, allowance, latch, kind="embedding") as client:
            result = await asyncio.to_thread(
                client.embeddings.create, model="test-model", input="hello", encoding_format="float"
            )
        assert result.data[0].embedding == [0.1, 0.2]
        record = records(tmp_path)[0]
        assert record["reserved_output_tokens"] == 0
        assert record["estimated_cost_usd"] == pytest.approx(0.00000024)
        assert allowance.known_cost == pytest.approx(0.00000024)
        assert not latch.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("input_value", [[1] * 8193, [[1] * 8192] * 37])
def test_oversized_embedding_is_denied_before_paid_dispatch(tmp_path, input_value):
    """Neither per-item nor aggregate embedding overflow silently clips source input."""

    async def scenario():
        """Send oversized token arrays through the actual embedding SDK serialization."""

        def handler(request):
            """Reject any physical dispatch for inadmissible embedding input."""
            pytest.fail("Oversized embedding reached HTTP")

        allowance, latch = Allowance(1), threading.Event()
        with client_for(tmp_path, handler, allowance, latch, kind="embedding") as client:
            with pytest.raises(openai.APIConnectionError):
                await asyncio.to_thread(
                    client.embeddings.create, model="test-model", input=input_value
                )
        assert not records(tmp_path)
        assert allowance.pending == allowance.known_cost == allowance.unresolved == 0
        assert latch.is_set()

    asyncio.run(scenario())


def test_concurrent_worker_clients_have_unique_calls_and_shared_admission(tmp_path):
    """Independent SDK worker threads serialize shared reservations without losing records."""

    async def scenario():
        """Overlap two physical calls using a barrier outside the accounting loop."""
        barrier = threading.Barrier(2)
        paced = []

        async def pacing(tokens, *, deadline):
            """Verify pacing runs on the owner's event loop, not in a provider worker."""
            assert asyncio.get_running_loop() is loop
            paced.append(tokens)
            return 0.5

        def handler(request):
            """Keep both transports in flight until both dollar reservations exist."""
            barrier.wait(timeout=2)
            return httpx.Response(200, json=generation_response())

        loop = asyncio.get_running_loop()
        allowance, latch = Allowance(1), threading.Event()
        with (
            client_for(tmp_path, handler, allowance, latch, pacing=pacing) as a,
            client_for(tmp_path, handler, allowance, latch, pacing=pacing) as b,
        ):
            await asyncio.gather(asyncio.to_thread(complete, a), asyncio.to_thread(complete, b))
        saved = records(tmp_path)
        assert len(saved) == len({r["call_id"] for r in saved}) == 2
        assert len(paced) == 2
        assert all(r["pacing_wait_seconds"] == 0.5 for r in saved)
        assert allowance.known_cost == pytest.approx(0.00044)
        assert allowance.pending == allowance.unresolved == 0

    asyncio.run(scenario())


def test_latched_cancellation_while_queued_never_becomes_unknown_cost(tmp_path):
    """A wrapper cancellation latch denies dispatch after a queued worker finishes waiting."""

    async def scenario():
        """Set the latch during pacing, then drain the worker before closing resources."""
        allowance, latch = Allowance(1), threading.Event()

        async def pacing(tokens, *, deadline):
            """Signal cancellation while a admitted request is still unsent."""
            latch.set()
            return 1.0

        def handler(request):
            """Reject physical dispatch after the caller cancelled queued work."""
            pytest.fail("Cancelled framework worker reached HTTP")

        with client_for(tmp_path, handler, allowance, latch, pacing=pacing) as client:
            with pytest.raises(openai.APIConnectionError):
                await asyncio.to_thread(complete, client)
        record = records(tmp_path)[0]
        assert record["dispatched"] is False
        assert record["estimated_cost_usd"] == 0
        assert record["status"] == "not_dispatched"
        assert allowance.pending == allowance.unresolved == allowance.known_cost == 0
        assert allowance.unknown_calls == 0

    asyncio.run(scenario())


def test_missing_generation_output_bound_fails_before_dispatch(tmp_path):
    """A framework may not omit the bound used for generation cost reservation."""

    async def scenario():
        """Use the SDK's omitted default and require the transport to deny admission."""

        def handler(request):
            """Reject accidental dispatch without an explicit generation bound."""
            pytest.fail("Unbounded generation reached HTTP")

        allowance, latch = Allowance(1), threading.Event()
        with client_for(tmp_path, handler, allowance, latch) as client:
            with pytest.raises(openai.APIConnectionError):
                await asyncio.to_thread(
                    client.chat.completions.create,
                    model="test-model",
                    messages=[{"role": "user", "content": "hello"}],
                )
        assert not records(tmp_path)
        assert allowance.pending == 0

    asyncio.run(scenario())


def test_failed_response_checkpoint_still_settles_and_blocks_later_dispatch(tmp_path, monkeypatch):
    """A disk failure after paid return cannot leave pending dollars or allow fallback work."""
    module = import_module("llgm.evaluation.framework_calls")
    original = module._RecordedTransport._write

    def fail_final(self, record):
        """Preserve predispatch evidence but simulate unavailable disk after completion."""
        if record["status"] == "completed":
            raise OSError("disk unavailable")
        return original(self, record)

    monkeypatch.setattr(module._RecordedTransport, "_write", fail_final)

    async def scenario():
        """Settle measured usage even though its final artifact cannot be replaced."""
        calls = []

        def handler(request):
            """Return a successfully billed provider response exactly once."""
            calls.append(request)
            return httpx.Response(200, json=generation_response())

        allowance, latch = Allowance(1), threading.Event()
        with client_for(tmp_path, handler, allowance, latch) as client:
            with pytest.raises(openai.APIConnectionError):
                await asyncio.to_thread(complete, client)
            assert latch.is_set()
            with pytest.raises(openai.APIConnectionError):
                await asyncio.to_thread(complete, client)
        assert len(calls) == 1
        assert allowance.known_cost == pytest.approx(0.00022)
        assert allowance.pending == allowance.unresolved == 0
        assert records(tmp_path)[0]["status"] == "dispatched"

    asyncio.run(scenario())


def test_native_sdk_structured_parser_failure_preserves_physical_usage(tmp_path):
    """The SDK's own structured parser runs only after raw response cost is durable."""
    pydantic = pytest.importorskip("pydantic")

    class Answer(pydantic.BaseModel):
        """Require JSON that the deliberately invalid generated content cannot satisfy."""

        value: str

    async def scenario():
        """Use the SDK parse endpoint with invalid content and valid billed usage."""

        def handler(request):
            """Return valid transport JSON containing invalid structured model output."""
            return httpx.Response(200, json=generation_response())

        allowance, latch = Allowance(1), threading.Event()
        with client_for(tmp_path, handler, allowance, latch) as client:
            with pytest.raises(pydantic.ValidationError):
                await asyncio.to_thread(
                    client.beta.chat.completions.parse,
                    model="test-model",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=20,
                    response_format=Answer,
                )
        record = records(tmp_path)[0]
        assert record["request"]["response_format"]["type"] == "json_schema"
        assert record["response"]["choices"][0]["message"]["content"] == "not JSON"
        assert record["estimated_cost_usd"] == pytest.approx(0.00022)
        assert allowance.known_cost == pytest.approx(0.00022)
        assert allowance.pending == allowance.unresolved == 0

    asyncio.run(scenario())
