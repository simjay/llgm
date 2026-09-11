"""Local priced-request admission and durable provider-call accounting.

Prices are supplied by the caller in USD per million tokens. Reservations are
estimates, not provider billing limits; incomplete usage keeps cost unknown.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.models.base import CallableModelClient, ModelClient, ModelRequest, ModelResponse, Usage


def request_reservation(request: ModelRequest, pricing: dict) -> tuple[int, float]:
    """Reserve uncached cost using UTF-8 bytes plus schema and framing allowance.

    This is a conservative local admission estimate, not provider billing
    enforcement. Returned usage is checked against the reserved token counts.
    """
    input_bound = (
        sum(len(message.content.encode("utf-8")) for message in request.messages)
        + len(json.dumps(request.output_schema, ensure_ascii=False).encode("utf-8"))
        + 2048
    )
    cost = (
        input_bound * pricing["usd_per_million_input_tokens"]
        + request.max_output_tokens * pricing["usd_per_million_output_tokens"]
    ) / 1_000_000
    return input_bound, cost


def usage_cost(usage: Usage, pricing: dict) -> float | None:
    """Price measured Responses or Chat Completions counts without double counting."""
    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    details = usage.extra.get("input_tokens_details", usage.extra.get("prompt_tokens_details"))
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if type(cached) is not int or not 0 <= cached <= usage.input_tokens:
        return None
    return (
        (usage.input_tokens - cached) * pricing["usd_per_million_input_tokens"]
        + cached * pricing["usd_per_million_cached_input_tokens"]
        + usage.output_tokens * pricing["usd_per_million_output_tokens"]
    ) / 1_000_000


def pacing_reservation(request: ModelRequest) -> int:
    """Estimate rate-limit tokens with cl100k, 10% framing margin and full output.

    Tokenize the actual message/schema JSON, then add 256 framing tokens. This
    local estimate is separate from the more conservative monetary reservation.
    The provider's rate-limit accounting may differ from returned billed usage.
    """
    try:
        import tiktoken
    except ImportError as exc:
        raise ConfigurationError("Token pacing requires tiktoken") from exc
    rendered = json.dumps(
        {
            "messages": [asdict(message) for message in request.messages],
            "output_schema": request.output_schema,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    tokens = len(tiktoken.get_encoding("cl100k_base").encode(rendered, disallowed_special=()))
    return (tokens * 11 + 9) // 10 + 256 + request.max_output_tokens


def make_token_pacer(
    tokens_per_minute: int,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Callable[..., Awaitable[float]]:
    """Share a conservative 60-second token window across recorded clients.

    Reservations expire by admission time and are never reduced by returned or
    missing usage. Waiting does not hold the lock, so smaller requests can use
    remaining capacity. Cancellation before admission consumes no tokens.
    """
    if type(tokens_per_minute) is not int or tokens_per_minute <= 0:
        raise ValueError("tokens_per_minute must be a positive integer")
    admissions: deque[tuple[float, int]] = deque()
    lock = asyncio.Lock()

    async def admit(tokens: int, *, deadline: float | None = None) -> float:
        """Reserve one request or wait until enough earlier admissions expire."""
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("Pacing tokens must be a positive integer")
        if tokens > tokens_per_minute:
            raise BudgetExceeded("token_pacing_request_exceeds_limit")
        started = clock()
        while True:
            async with lock:
                now = clock()
                if deadline is not None and now >= deadline:
                    raise BudgetExceeded("run_admission_deadline")
                while admissions and admissions[0][0] + 60 <= now:
                    admissions.popleft()
                used = sum(amount for _, amount in admissions)
                if used + tokens <= tokens_per_minute:
                    admissions.append((now, tokens))
                    return now - started
                for admitted_at, amount in admissions:
                    used -= amount
                    if used + tokens <= tokens_per_minute:
                        delay = admitted_at + 60 - now
                        break
                if deadline is not None:
                    delay = min(delay, deadline - now)
            await sleep(delay)

    return admit


class Allowance:
    """Admit concurrent requests against known cost and unresolved reservations.

    These local estimates do not enforce provider billing. Unknown usage retains
    its full reservation. A returned count beyond its bound stops new admission.
    """

    def __init__(self, cap: float, max_unknown_calls: int = 16, *, deadline: float | None = None):
        """Start one independent generation or judging allowance."""
        if not 0 < cap < float("inf"):
            raise ValueError("Allowance cap must be finite and positive")
        self.cap = cap
        self.known_cost = 0.0
        self.pending = 0.0
        self.unresolved = 0.0
        self.unknown_calls = 0
        self.max_unknown_calls = max_unknown_calls
        self.deadline = deadline
        self.stopped_reason = None
        self.lock = asyncio.Lock()

    async def reserve(self, cost: float) -> None:
        """Reserve before dispatch, including requests already in flight."""
        async with self.lock:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self.stopped_reason = "run_admission_deadline"
            if self.stopped_reason:
                raise BudgetExceeded(self.stopped_reason)
            if self.known_cost + self.pending + self.unresolved + cost > self.cap:
                self.stopped_reason = "local_cost_admission_limit"
                raise BudgetExceeded(self.stopped_reason)
            self.pending += cost

    async def settle(self, reserved: float, actual: float | None, within_bound: bool) -> None:
        """Reconcile once, retaining uncertainty and halting after a broken bound."""
        async with self.lock:
            self.pending = max(0.0, self.pending - reserved)
            if actual is None:
                self.unresolved += reserved
                self.unknown_calls += 1
            else:
                self.known_cost += actual
            if not within_bound:
                self.stopped_reason = "returned_usage_exceeds_reservation"
            elif self.unknown_calls >= self.max_unknown_calls:
                self.stopped_reason = "unknown_usage_call_limit"

    def record(self) -> dict:
        """Expose estimated known cost separately from unresolved liabilities."""
        return {
            "cap_usd": self.cap,
            "known_estimated_cost_usd": self.known_cost,
            "pending_reserved_cost_usd": self.pending,
            "unknown_reserved_cost_usd": self.unresolved,
            "unknown_cost_calls": self.unknown_calls,
            "estimated_cost_usd": None if self.unknown_calls else self.known_cost,
            "stopped_reason": self.stopped_reason,
        }


def recorded_model(
    client: ModelClient,
    role: str,
    pricing: dict,
    allowance: Allowance,
    trial: dict,
    path: Path,
    *,
    call_metadata: dict | None = None,
    pacing: Callable[..., Awaitable[float]] | None = None,
    temperature: float | None = 0,
) -> CallableModelClient:
    """Record one call per request with a declared temperature and shared allowance.

    The caller owns the client and existing output directory. ``trial`` must
    contain a ``model_calls`` list shared by clients writing to that directory.
    A request record is atomically published before adapter dispatch and updated
    after return or failure. Provider exceptions retain only their type. Unknown
    usage retains the reservation; no retries or client cleanup are implicit.
    Optional call metadata is copied before each dispatch, allowing the caller
    to identify the current trial and construction/query phase durably.
    A shared token pacer delays dispatch within the allowance deadline. Waiting
    attempts are durable but explicitly unpaid; cancellation releases their
    monetary reservation. Dispatched failures retain unknown cost as usual.
    Temperature defaults to zero. Explicit None leaves sampling unset at the
    provider and is retained as None in the recorded request.
    """

    async def complete(request: ModelRequest) -> ModelResponse:
        """Call the real adapter once with fixed temperature and durable accounting."""
        request = replace(request, temperature=temperature)
        bound, reserved = request_reservation(request, pricing)
        pacing_tokens = pacing_reservation(request) if pacing is not None else None
        await allowance.reserve(reserved)
        record = {
            **(call_metadata or {}),
            "call_id": uuid4().hex,
            "role": role,
            "status": "waiting_for_tokens" if pacing is not None else "admitted",
            "dispatched": False,
            "request": asdict(request),
            "reserved_input_tokens": bound,
            "reserved_cost_usd": reserved,
            "estimated_cost_usd": 0.0,
        }
        trial["model_calls"].append(record)
        number = len(trial["model_calls"])
        output = path / f"call-{number:03d}.json"
        started, actual, within_bound = time.perf_counter(), 0.0, True
        dispatched = False
        try:
            if pacing is not None:
                record.update(pacing_reserved_tokens=pacing_tokens, pacing_wait_seconds=0.0)
                _write_call_record(output, record)
                waiting_started = time.monotonic()
                try:
                    waited = await pacing(pacing_tokens, deadline=allowance.deadline)
                except BaseException as exc:
                    record["pacing_wait_seconds"] = time.monotonic() - waiting_started
                    if isinstance(exc, BudgetExceeded):
                        await allowance.reserve(0.0)
                    raise
                record["pacing_wait_seconds"] = waited
                await allowance.reserve(0.0)
            record.update(status="dispatched", dispatched=True, estimated_cost_usd=None)
            _write_call_record(output, record)
            dispatched, actual = True, None
            response = await client.complete(request)
            actual = usage_cost(response.usage, pricing)
            within_bound = (
                (response.usage.input_tokens is None or response.usage.input_tokens <= bound)
                and (
                    response.usage.output_tokens is None
                    or response.usage.output_tokens <= request.max_output_tokens
                )
                and (actual is None or actual <= reserved + 1e-12)
            )
            record.update(
                response=asdict(response), status=response.status, estimated_cost_usd=actual
            )
            return response
        except BaseException as exc:
            record.update(
                status="failed" if dispatched else "not_dispatched",
                dispatched=dispatched,
                estimated_cost_usd=actual,
                error_type=type(exc).__name__,
            )
            raise
        finally:
            await allowance.settle(reserved, actual, within_bound)
            record["elapsed_seconds"] = time.perf_counter() - started
            record["usage_within_reservation"] = within_bound
            _write_call_record(output, record)

    descriptor = client.descriptor()
    return CallableModelClient(
        complete,
        provider=descriptor["provider"],
        model=descriptor["model"],
        capabilities=client.capabilities,
    )


def _write_call_record(path: Path, value: dict) -> None:
    """Replace a checkpoint atomically within its existing directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
