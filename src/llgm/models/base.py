"""Small, provider-neutral contracts for hosted text generation.

Requests carry text messages. LLGM validates runtime operations separately from
the provider transport.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from llgm.core.errors import CapabilityError, ConfigurationError, ProviderError


@dataclass(frozen=True)
class Message:
    """A text message with a provider-neutral instruction or conversation role."""

    role: str
    content: str

    def __post_init__(self) -> None:
        """Reject unsupported roles and non-text content."""
        if not isinstance(self.role, str) or self.role not in {
            "system",
            "developer",
            "user",
            "assistant",
        }:
            raise ConfigurationError(f"Unsupported text message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise ConfigurationError("Message content must be text")


@dataclass(frozen=True)
class ModelRequest:
    """An ordered message sequence with output limits and an optional JSON schema."""

    messages: tuple[Message, ...]
    max_output_tokens: int = 1024
    temperature: float | None = None
    output_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Freeze message order and validate generation settings before dispatch."""
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages or not all(isinstance(m, Message) for m in self.messages):
            raise ConfigurationError("A model request requires Message objects")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ConfigurationError("max_output_tokens must be a positive integer")
        if self.temperature is not None and (
            type(self.temperature) not in {int, float}
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ConfigurationError("temperature must be between 0 and 2")
        if self.output_schema is not None and not isinstance(self.output_schema, dict):
            raise ConfigurationError("output_schema must be a JSON Schema object")


@dataclass(frozen=True)
class Usage:
    """Actual provider counts. Unknown usage must never be represented as zero.

    input_tokens is the total input count, including cached tokens. Provider
    billing categories are retained separately in extra, and must be priced
    according to the provider's rules rather than charging every category twice.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Accept unknown counts while rejecting invalid measured token totals."""
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ConfigurationError("Usage counts must be nonnegative integers or None")

    @property
    def total_tokens(self) -> int | None:
        """Return the total only when both input and output counts are known."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ModelCapabilities:
    """Features explicitly supported by a configured generation adapter."""

    structured_output: bool = False
    tool_calls: bool = False
    streaming: bool = False
    token_counting: bool = False


@dataclass(frozen=True)
class ModelResponse:
    """Normalized provider output, completion state, and measured usage."""

    text: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    provider: str = "unknown"
    model: str = "unknown"
    request_id: str | None = None
    status: str = "completed"
    refusal: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def ensure_complete(self) -> ModelResponse:
        """Reject refusals, truncation, and provider failures before parsing."""
        if self.status != "completed" or self.refusal is not None:
            raise ProviderError(
                f"{self.provider} response is {self.status}; finish_reason={self.finish_reason!r}"
            )
        return self


@runtime_checkable
class ModelClient(Protocol):
    """Async text generation with explicit capabilities and reproducibility metadata."""

    capabilities: ModelCapabilities

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Perform one logical generation request and retain its completion state."""
        ...

    def descriptor(self) -> Mapping[str, Any]:
        """Return reproducibility metadata without credentials or message text."""
        ...


class ScriptedModelClient:
    """Consume deterministic responses once each. Never contact a provider.

    String fixtures have unknown token usage. Supply ModelResponse fixtures when
    tests require known counts. Exhaustion raises instead of inventing an answer.
    """

    capabilities = ModelCapabilities()

    def __init__(
        self, responses: Sequence[str | ModelResponse], *, model: str = "scripted"
    ) -> None:
        """Keep a finite response sequence and a record of received requests."""
        self.model = model
        self._responses = iter(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Consume the next fixture, raising when the response sequence is exhausted."""
        _check_schema(self.capabilities, request)
        self.requests.append(request)
        try:
            response = next(self._responses)
        except StopIteration:
            raise ProviderError("Scripted model has no remaining responses") from None
        if isinstance(response, ModelResponse):
            return response
        if not isinstance(response, str):
            raise ProviderError("Scripted response must be text or ModelResponse")
        return ModelResponse(response, provider="scripted", model=self.model, finish_reason="stop")

    def descriptor(self) -> dict[str, Any]:
        """Identify the deterministic model and its lack of network calls."""
        return {"provider": "scripted", "model": self.model, "network": False}

    async def aclose(self) -> None:
        """Satisfy the client lifecycle contract without owning external resources."""
        pass


class CallableModelClient:
    """Inject an application's async SDK wrapper without inheriting its types.

    The callable owns any internal retries and must account for them itself.
    LLGM makes one invocation and never silently retries it.
    """

    def __init__(
        self,
        function: Callable[[ModelRequest], Awaitable[ModelResponse]],
        *,
        provider: str = "callable",
        model: str = "application",
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        """Bind an application-owned async function and its declared capabilities."""
        self._function = function
        self.provider = provider
        self.model = model
        self.capabilities = capabilities or ModelCapabilities()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Invoke the injected function once and validate its response contract."""
        _check_schema(self.capabilities, request)
        result = self._function(request)
        if not inspect.isawaitable(result):
            raise ProviderError("CallableModelClient requires an async callable")
        response = await result
        if not isinstance(response, ModelResponse):
            raise ProviderError("Model callable must return ModelResponse")
        return response

    def descriptor(self) -> dict[str, Any]:
        """Describe the injected model and assign retry accounting to its application."""
        return {
            "provider": self.provider,
            "model": self.model,
            "capabilities": asdict(self.capabilities),
            "retry_owner": "application",
        }


def _check_schema(capabilities: ModelCapabilities, request: ModelRequest) -> None:
    """Reject native schema requests unless the adapter explicitly supports them."""
    if request.output_schema is not None and not capabilities.structured_output:
        raise CapabilityError(
            "This adapter has no enabled native JSON Schema capability; "
            "use a supported adapter or explicitly prompt and validate JSON in the runtime"
        )
