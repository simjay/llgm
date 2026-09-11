"""Record synchronous framework SDK calls before provider-response parsing.

The returned OpenAI client belongs to the caller. Use it in worker threads while
its accounting event loop remains alive. On cancellation, set the shared failure
latch and drain those workers before closing the client or event loop.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from pathlib import Path
from uuid import uuid4

import httpx

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.evaluation.costs import Allowance, usage_cost
from llgm.models.base import Usage


def _embedding_tokens(value, encoding) -> int:
    """Validate each embedding input and the combined request without clipping."""
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list) and value:
        items = [value] if all(type(token) is int for token in value) else value
    else:
        raise ConfigurationError("Embedding input must be nonempty text or token arrays")
    if len(items) > 2048:
        raise ConfigurationError("Embedding request exceeds 2048 inputs")
    counts = []
    for item in items:
        if isinstance(item, str):
            count = len(encoding.encode(item, disallowed_special=()))
        elif isinstance(item, list) and all(type(token) is int and token >= 0 for token in item):
            count = len(item)
        else:
            raise ConfigurationError("Embedding inputs must be text or nonnegative token arrays")
        if not 1 <= count <= 8192:
            raise ConfigurationError("Embedding input must contain 1 to 8192 tokens")
        counts.append(count)
    total = sum(counts)
    if total > 300_000:
        raise ConfigurationError("Embedding request exceeds 300000 tokens")
    return total


def _request_limits(request, *, model, kind, max_output_tokens, pricing):
    """Require a priced endpoint, model and output bound before any paid dispatch."""
    import tiktoken

    raw = request.read()
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ConfigurationError("Framework request must contain JSON") from exc
    if request.method != "POST" or not isinstance(body, dict) or body.get("model") != model:
        raise ConfigurationError("Framework request differs from its configured model")
    endpoint = request.url.path
    if request.url.host != "api.openai.com" or request.url.scheme != "https" or request.url.query:
        raise ConfigurationError("Framework client supports only the configured OpenAI endpoint")
    allowed = (
        {"/v1/embeddings"} if kind == "embedding" else {"/v1/chat/completions", "/v1/responses"}
    )
    if endpoint not in allowed or body.get("stream") or body.get("n", 1) != 1:
        raise ConfigurationError("Framework request uses an unsupported endpoint or response mode")
    encoding = tiktoken.get_encoding("cl100k_base")
    if kind == "embedding":
        prompt_tokens = _embedding_tokens(body.get("input"), encoding)
        output_tokens = 0
    else:
        fields = [
            name
            for name in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            if name in body
        ]
        allowed_fields = (
            {"max_output_tokens"}
            if endpoint == "/v1/responses"
            else {"max_tokens", "max_completion_tokens"}
        )
        if len(fields) != 1 or fields[0] not in allowed_fields:
            raise ConfigurationError("Generation requires one explicit output-token limit")
        output_tokens = body[fields[0]]
        if type(output_tokens) is not int or not 1 <= output_tokens <= max_output_tokens:
            raise ConfigurationError("Generation output exceeds its configured reservation")
        prompt_tokens = len(encoding.encode(raw.decode("utf-8"), disallowed_special=()))
    input_bound = len(raw) + 2048
    reserved = (
        input_bound * pricing["usd_per_million_input_tokens"]
        + output_tokens * pricing.get("usd_per_million_output_tokens", 0)
    ) / 1_000_000
    return {
        "endpoint": endpoint,
        "request": body,
        "request_body": raw.decode("utf-8"),
        "reserved_input_tokens": input_bound,
        "reserved_output_tokens": output_tokens,
        "reserved_cost_usd": reserved,
        "pacing_reserved_tokens": (prompt_tokens * 11 + 9) // 10 + 256 + output_tokens,
    }


def _response_cost(body, kind, pricing):
    """Price embeddings separately and retain missing generation categories as unknown."""
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None, None, None
    if kind == "embedding":
        tokens = usage.get("prompt_tokens")
        if type(tokens) is not int or tokens < 0:
            return None, None, None
        return tokens * pricing["usd_per_million_input_tokens"] / 1_000_000, tokens, 0
    inputs = usage.get("input_tokens", usage.get("prompt_tokens"))
    outputs = usage.get("output_tokens", usage.get("completion_tokens"))
    if any(type(count) is not int or count < 0 for count in (inputs, outputs)):
        return None, None, None
    return usage_cost(Usage(inputs, outputs, usage), pricing), inputs, outputs


def _error_summary(body):
    """Keep error classifications and numeric usage without provider messages."""
    if not isinstance(body, dict):
        return None
    result = {}
    error = body.get("error")
    if isinstance(error, dict):
        result["error"] = {
            key: error[key] for key in ("code", "type") if isinstance(error.get(key), str)
        }
    usage = body.get("usage")
    if isinstance(usage, dict):
        result["usage"] = {}
        for key, value in usage.items():
            if type(value) is int and value >= 0:
                result["usage"][key] = value
            elif isinstance(value, dict):
                result["usage"][key] = {
                    name: count
                    for name, count in value.items()
                    if type(count) is int and count >= 0
                }
    return result


class _RecordedTransport(httpx.BaseTransport):
    """Own the synchronous transport and serialize accounting on one async loop."""

    def __init__(
        self,
        *,
        transport,
        loop,
        allowance,
        pacing,
        path,
        call_metadata,
        failure_latch,
        stop_after_failure,
        model,
        kind,
        pricing,
        max_output_tokens,
        api_key,
    ):
        """Bind explicit transport ownership, pricing and a shared per-case failure latch."""
        self.transport = transport
        self.loop = loop
        self.allowance = allowance
        self.pacing = pacing
        self.path = path
        self.call_metadata = call_metadata
        self.failure_latch = failure_latch
        self.stop_after_failure = stop_after_failure
        self.model = model
        self.kind = kind
        self.pricing = pricing
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key

    def _bridge(self, function, *args):
        """Run accounting on its owning loop without blocking that same loop."""
        try:
            active = asyncio.get_running_loop()
        except RuntimeError:
            active = None
        if active is self.loop or not self.loop.is_running():
            raise ConfigurationError("Use the synchronous framework client in a worker thread")
        return asyncio.run_coroutine_threadsafe(function(*args), self.loop).result()

    def _write(self, record):
        """Publish raw bodies and accounting without headers or the configured credential."""
        text = json.dumps(record, ensure_ascii=False, indent=2)
        if self.api_key and self.api_key in text:
            text = text.replace(self.api_key, "[redacted]")
        target = self.path / f"call-{record['call_id']}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(text + "\n")
        temporary.replace(target)

    async def _admit(self, limits):
        """Reserve dollars and tokens before exposing the physical request to HTTP."""
        if self.failure_latch.is_set():
            raise BudgetExceeded("framework_failure_latch")
        await self.allowance.reserve(limits["reserved_cost_usd"])
        record = {
            **self.call_metadata,
            **limits,
            "call_id": uuid4().hex,
            "kind": self.kind,
            "model": self.model,
            "status": "waiting_for_tokens",
            "dispatched": False,
            "estimated_cost_usd": 0.0,
            "pacing_wait_seconds": 0.0,
            "admitted_at_unix_seconds": time.time(),
        }
        started = time.monotonic()
        try:
            self._write(record)
            if self.pacing is not None:
                record["pacing_wait_seconds"] = await self.pacing(
                    limits["pacing_reserved_tokens"], deadline=self.allowance.deadline
                )
            if self.failure_latch.is_set():
                raise BudgetExceeded("framework_failure_latch")
            await self.allowance.reserve(0)
            record.update(
                status="dispatched",
                dispatched=True,
                estimated_cost_usd=None,
                dispatched_at_unix_seconds=time.time(),
            )
            self._write(record)
            return record
        except BaseException as exc:
            await self.allowance.settle(limits["reserved_cost_usd"], 0.0, True)
            record.update(
                status="not_dispatched",
                dispatched=False,
                estimated_cost_usd=0.0,
                error_type=type(exc).__name__,
                pacing_wait_seconds=time.monotonic() - started,
            )
            self._write(record)
            raise

    async def _finish(self, record, response, error, elapsed):
        """Persist raw response and reconcile once before the SDK can parse it."""
        actual, inputs, outputs = None, None, None
        within_bound = True
        failed = error is not None
        if response is not None:
            record["http_status"] = response.status_code
            try:
                if response.is_success:
                    record["response_body"] = response.content.decode("utf-8", errors="replace")
                body = response.json()
            except (ValueError, httpx.ResponseNotRead):
                body = None
            if not response.is_success:
                body = _error_summary(body)
            record["response"] = body
            actual, inputs, outputs = _response_cost(body, self.kind, self.pricing)
            choices = body.get("choices") if isinstance(body, dict) else None
            choices = choices if isinstance(choices, list) else []
            incomplete = isinstance(body, dict) and (
                body.get("status") in {"failed", "incomplete", "cancelled"}
                or any(
                    choice.get("finish_reason") == "length"
                    for choice in choices
                    if isinstance(choice, dict)
                )
            )
            failed = (
                failed or not response.is_success or body is None or actual is None or incomplete
            )
            within_bound = (
                (inputs is None or inputs <= record["reserved_input_tokens"])
                and (outputs is None or outputs <= record["reserved_output_tokens"])
                and (actual is None or actual <= record["reserved_cost_usd"] + 1e-12)
            )
            failed = failed or not within_bound
        if failed and self.stop_after_failure:
            self.failure_latch.set()
        record.update(
            status="failed" if failed else "completed",
            estimated_cost_usd=actual,
            usage_within_reservation=within_bound,
            elapsed_seconds=elapsed,
        )
        if error is not None:
            record["error_type"] = type(error).__name__
        # Save returned usage before another parser or aggregate writer can fail.
        try:
            self._write(record)
        except BaseException:
            if self.stop_after_failure:
                self.failure_latch.set()
            raise
        finally:
            await self.allowance.settle(record["reserved_cost_usd"], actual, within_bound)

    def handle_request(self, request):
        """Dispatch once, fully read the body, and record before SDK parsing starts."""
        record, response, error = None, None, None
        started = time.monotonic()
        try:
            limits = _request_limits(
                request,
                model=self.model,
                kind=self.kind,
                max_output_tokens=self.max_output_tokens,
                pricing=self.pricing,
            )
            record = self._bridge(self._admit, limits)
            response = self.transport.handle_request(request)
            response.read()
            return response
        except BaseException as exc:
            error = exc
            if self.stop_after_failure:
                self.failure_latch.set()
            if response is not None:
                response.close()
            raise
        finally:
            if record is not None:
                self._bridge(self._finish, record, response, error, time.monotonic() - started)

    def close(self):
        """Close the caller-owned physical connection pool after workers have drained."""
        self.transport.close()


def framework_openai_client(
    *,
    api_key: str,
    model: str,
    kind: str,
    pricing: dict,
    allowance: Allowance,
    pacing,
    loop,
    path: Path,
    call_metadata: dict,
    failure_latch: threading.Event,
    max_output_tokens: int = 2048,
    timeout_seconds: float = 90,
    stop_after_failure: bool = True,
    transport=None,
):
    """Create a caller-owned synchronous OpenAI client with physical retries disabled.

    ``kind`` is generation or embedding; model and prices apply to that endpoint
    only. Generation requests must provide an explicit output limit no larger
    than ``max_output_tokens``. ``path`` must already exist. Metadata is copied
    on the owner loop per admission. Successful bodies retain raw JSON. HTTP
    failures retain only status, error type/code and numeric usage because
    provider error messages may echo credentials. Headers are never recorded;
    occurrences of the configured credential are also redacted.

    The failure latch blocks subsequent requests after physical failures by
    default. Wrappers must also latch surfaced SDK/framework parsing errors and
    validate results when a framework swallows its own parsing exceptions.
    """
    from openai import OpenAI

    fields = {"usd_per_million_input_tokens"}
    if kind == "generation":
        fields.update({"usd_per_million_cached_input_tokens", "usd_per_million_output_tokens"})
    if kind not in {"generation", "embedding"} or not isinstance(model, str) or not model:
        raise ConfigurationError("Framework client requires a fixed model and endpoint kind")
    if any(
        type(pricing.get(key)) not in (int, float)
        or not math.isfinite(pricing[key])
        or pricing[key] < 0
        for key in fields
    ):
        raise ConfigurationError("Framework client requires finite nonnegative token prices")
    if type(max_output_tokens) is not int or max_output_tokens <= 0 or not path.is_dir():
        raise ConfigurationError(
            "Framework client requires an output limit and existing record directory"
        )
    wrapped = _RecordedTransport(
        transport=transport
        if transport is not None
        else httpx.HTTPTransport(retries=0, trust_env=False),
        loop=loop,
        allowance=allowance,
        pacing=pacing,
        path=path,
        call_metadata=call_metadata,
        failure_latch=failure_latch,
        stop_after_failure=stop_after_failure,
        model=model,
        kind=kind,
        pricing=dict(pricing),
        max_output_tokens=max_output_tokens,
        api_key=api_key,
    )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        max_retries=0,
        timeout=timeout_seconds,
        http_client=httpx.Client(
            transport=wrapped, timeout=timeout_seconds, follow_redirects=False, trust_env=False
        ),
    )
