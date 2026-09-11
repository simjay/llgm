"""Lazy optional adapters for native and OpenAI-compatible hosted APIs.

Only text and native JSON-schema requests are implemented here. Schema support
is an adapter capability, subject to the selected model's provider restrictions.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from llgm.core.errors import CapabilityError, ConfigurationError, ProviderError
from llgm.models.base import ModelCapabilities, ModelRequest, ModelResponse, Usage, _check_schema


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read a response field from either an SDK object or a mapping."""
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _plain(value: Any) -> Any:
    """Convert provider metadata to JSON-compatible values without private attributes."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return None


def _count(value: Any) -> int | None:
    """Return a valid nonnegative token count, or mark it unknown."""
    return value if type(value) is int and value >= 0 else None


def _usage(
    value: Any, *, input_key: str = "input_tokens", output_key: str = "output_tokens"
) -> Usage:
    """Normalize provider token fields while preserving raw billing categories."""
    raw = _plain(value) or {}
    return Usage(_count(_get(value, input_key)), _count(_get(value, output_key)), raw)


def _anthropic_usage(value: Any) -> Usage:
    """Combine uncached and cached input categories without double counting."""
    raw = _plain(value) or {}
    parts = [
        _count(_get(value, "input_tokens")),
        _count(_get(value, "cache_creation_input_tokens", 0)),
        _count(_get(value, "cache_read_input_tokens", 0)),
    ]
    total = None if any(p is None for p in parts) else sum(parts)  # type: ignore[arg-type]
    return Usage(total, _count(_get(value, "output_tokens")), raw)


def _safe_url(value: str | None) -> str | None:
    """Validate an HTTP endpoint and reject embedded secrets or request parameters."""
    if value is None:
        return None
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigurationError("Model base_url must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigurationError(
            "Model base_url must not contain credentials, queries, or fragments"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _sdk_client(
    provider: str, base_url: str | None, api_key_env: str | None, timeout: float
) -> Any:
    """Load an optional provider SDK with explicit credentials and retries disabled."""
    kwargs: dict[str, Any] = {"max_retries": 0, "timeout": timeout}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key_env is not None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ConfigurationError(f"Credential environment variable {api_key_env!r} is not set")
        kwargs["api_key"] = api_key
    try:
        if provider == "anthropic":
            from anthropic import AsyncAnthropic

            return AsyncAnthropic(**kwargs)
        from openai import AsyncOpenAI

        return AsyncOpenAI(**kwargs)
    except ImportError:
        extra = "anthropic" if provider == "anthropic" else "openai"
        raise ConfigurationError(
            f"Install the optional provider dependency: pip install 'llgm[{extra}]'"
        ) from None
    except Exception as exc:
        raise ConfigurationError(
            f"Could not initialize {provider} SDK ({type(exc).__name__})"
        ) from None


class _HostedClient:
    """Shared SDK ownership, endpoint validation, and sanitized transport failures."""

    provider: str

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout_seconds: float = 60,
        supports_structured_output: bool = True,
    ) -> None:
        """Validate transport settings and disable retries on the request client."""
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("A hosted model identifier is required")
        if (
            type(timeout_seconds) not in {int, float}
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ConfigurationError("timeout_seconds must be finite and positive")
        if type(supports_structured_output) is not bool:
            raise ConfigurationError("supports_structured_output must be a boolean")
        self.model = model
        self.base_url = _safe_url(base_url)
        self.capabilities = ModelCapabilities(structured_output=supports_structured_output)
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else _sdk_client(self.provider, self.base_url, api_key_env, timeout_seconds)
        )
        # SDK clones share their transport. The adapter never closes a borrowed
        # client, while its per-request clone disables hidden SDK retry attempts.
        self._request_client = (
            self._client.with_options(max_retries=0, timeout=timeout_seconds)
            if hasattr(self._client, "with_options")
            else self._client
        )

    def descriptor(self) -> dict[str, Any]:
        """Expose transport and capability metadata without credentials."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "capabilities": asdict(self.capabilities),
            "sdk_max_retries": 0,
            "injected_client": not self._owns_client,
        }

    async def aclose(self) -> None:
        """Close only SDK clients created and owned by this adapter."""
        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> _HostedClient:
        """Enter the context without changing SDK ownership."""
        return self

    async def __aexit__(self, *_: Any) -> None:
        """Release owned transport resources when the context exits."""
        await self.aclose()

    async def _call(self, method: Any, kwargs: dict[str, Any]) -> Any:
        """Await one SDK invocation and redact provider exception details."""
        try:
            return await method(**kwargs)
        except Exception as exc:
            # Provider errors can contain prompts, credentials, or full HTTP
            # requests. Do not copy their message or chain into public errors.
            raise ProviderError(f"{self.provider} request failed ({type(exc).__name__})") from None


class OpenAIModelClient(_HostedClient):
    """Native Responses API, with schema negotiation and no SDK retries."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout_seconds: float = 60,
        supports_structured_output: bool = True,
        reasoning_effort: str | None = None,
    ) -> None:
        """Configure optional provider reasoning without inferring a model's supported efforts."""
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
        ):
            raise ConfigurationError("reasoning_effort must be a nonblank string or None")
        super().__init__(
            model,
            client=client,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
            supports_structured_output=supports_structured_output,
        )
        self.reasoning_effort = reasoning_effort

    def descriptor(self) -> dict[str, Any]:
        """Retain an explicitly configured effort alongside existing transport metadata."""
        descriptor = super().descriptor()
        if self.reasoning_effort is not None:
            descriptor["reasoning_effort"] = self.reasoning_effort
        return descriptor

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Call Responses and preserve usage, refusals, truncation, and output types."""
        _check_schema(self.capabilities, request)
        if self.reasoning_effort not in (None, "none") and request.temperature is not None:
            raise CapabilityError("OpenAI reasoning effort requires temperature=None")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": m.role,
                    "content": m.content,
                    **({"phase": "final_answer"} if m.role == "assistant" else {}),
                }
                for m in request.messages
            ],
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.output_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "llgm_output",
                    "strict": True,
                    "schema": request.output_schema,
                }
            }
        result = await self._call(self._request_client.responses.create, kwargs)
        text_parts: list[str] = []
        refusals: list[str] = []
        unsupported: list[str] = []
        output = _get(result, "output", ()) or ()
        phases = [_get(item, "phase") for item in output if _get(item, "type") == "message"]
        phased = any(phase is not None for phase in phases)
        missing_final = phased and "final_answer" not in phases
        for item in output:
            kind = _get(item, "type")
            if kind == "message":
                phase = _get(item, "phase")
                if phase not in {None, "commentary", "final_answer"}:
                    unsupported.append(f"message_phase:{phase}")
                for block in _get(item, "content", ()) or ():
                    if _get(block, "type") == "output_text":
                        # Commentary and the final answer can contain separate
                        # JSON operations. Only the completed answer is executable.
                        if not phased or phase == "final_answer":
                            text_parts.append(_get(block, "text", ""))
                    elif _get(block, "type") == "refusal":
                        refusals.append(_get(block, "refusal", "Provider refusal"))
                    else:
                        unsupported.append(str(_get(block, "type")))
            elif kind != "reasoning":
                unsupported.append(str(kind))
        text = "".join(text_parts)
        if not phased and not text:
            text = _get(result, "output_text", "") or ""
        provider_status = _get(result, "status")
        if refusals:
            status = "refused"
        elif missing_final and not unsupported:
            status = "incomplete"
        elif provider_status == "completed" and not unsupported:
            status = "completed"
        elif provider_status in {"incomplete", "cancelled", "queued", "in_progress"}:
            status = "incomplete"
        else:
            status = "failed"
        details = _get(result, "incomplete_details")
        finish_reason = _get(details, "reason") or (
            "missing_final_answer" if missing_final else provider_status
        )
        return ModelResponse(
            text,
            _usage(_get(result, "usage")),
            finish_reason,
            self.provider,
            _get(result, "model") or self.model,
            _get(result, "_request_id") or _get(result, "id"),
            status,
            "\n".join(refusals) if refusals else None,
            {
                "response_id": _get(result, "id"),
                "provider_status": provider_status,
                "incomplete_details": _plain(details),
                "unsupported_output_types": unsupported,
                "output_message_phases": phases,
                "discarded_commentary_messages": phases.count("commentary"),
            },
        )


class AnthropicModelClient(_HostedClient):
    """Native Messages API. Leading system instructions remain separate."""

    provider = "anthropic"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Map supported roles to Messages and normalize output and cache usage."""
        _check_schema(self.capabilities, request)
        system: list[str] = []
        messages: list[dict[str, str]] = []
        for message in request.messages:
            if message.role == "developer":
                raise CapabilityError(
                    "Anthropic has no separate developer role; use an explicit system prompt"
                )
            if message.role == "system":
                if messages:
                    raise CapabilityError(
                        "Anthropic system instructions must precede conversation messages"
                    )
                system.append(message.content)
            else:
                messages.append({"role": message.role, "content": message.content})
        if not messages:
            raise ConfigurationError("Anthropic requests require a user or assistant message")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
        }
        if system:
            kwargs["system"] = "\n\n".join(system)
        if request.temperature is not None:
            if request.temperature > 1:
                raise CapabilityError("Anthropic temperature must be between 0 and 1")
            kwargs["temperature"] = request.temperature
        if request.output_schema is not None:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": request.output_schema,
                }
            }
        result = await self._call(self._request_client.messages.create, kwargs)
        content = _get(result, "content", ()) or ()
        text = "".join(_get(b, "text", "") for b in content if _get(b, "type") == "text")
        stop = _get(result, "stop_reason")
        unsupported = [
            str(_get(b, "type"))
            for b in content
            if _get(b, "type") not in {"text", "thinking", "redacted_thinking"}
        ]
        if stop == "refusal":
            status = "refused"
        elif stop in {"max_tokens", "model_context_window_exceeded", "pause_turn"}:
            status = "incomplete"
        elif stop in {"end_turn", "stop_sequence"} and not unsupported:
            status = "completed"
        else:
            status = "failed"
        return ModelResponse(
            text,
            _anthropic_usage(_get(result, "usage")),
            stop,
            self.provider,
            _get(result, "model") or self.model,
            _get(result, "_request_id") or _get(result, "id"),
            status,
            text if status == "refused" else None,
            {
                "response_id": _get(result, "id"),
                "stop_sequence": _get(result, "stop_sequence"),
                "unsupported_output_types": unsupported,
            },
        )


class OpenAICompatibleModelClient(_HostedClient):
    """Chat Completions transport for a declared compatible endpoint.

    Schema support is opt-in: compatibility alone does not imply that an
    endpoint/model honors JSON Schema. Token-limit parameter is configurable
    because older servers accept max_tokens, newer APIs max_completion_tokens.
    """

    provider = "openai_compatible"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        supports_structured_output: bool = False,
        max_tokens_parameter: str = "max_tokens",
        **kwargs: Any,
    ) -> None:
        """Bind an explicit endpoint, schema capability, and token-limit parameter."""
        if max_tokens_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ConfigurationError("Unsupported Chat Completions token-limit parameter")
        super().__init__(
            model,
            base_url=base_url,
            supports_structured_output=supports_structured_output,
            **kwargs,
        )
        self.max_tokens_parameter = max_tokens_parameter

    def descriptor(self) -> dict[str, Any]:
        """Include the endpoint-specific token-limit parameter in transport metadata."""
        return {**super().descriptor(), "max_tokens_parameter": self.max_tokens_parameter}

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Call Chat Completions and normalize text, usage, and completion status."""
        _check_schema(self.capabilities, request)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            self.max_tokens_parameter: request.max_output_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.output_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "llgm_output",
                    "strict": True,
                    "schema": request.output_schema,
                },
            }
        result = await self._call(self._request_client.chat.completions.create, kwargs)
        choices = _get(result, "choices", ()) or ()
        choice = choices[0] if choices else None
        message = _get(choice, "message")
        content = _get(message, "content")
        refusal = _get(message, "refusal")
        stop = _get(choice, "finish_reason")
        if refusal or stop == "content_filter":
            status = "refused"
        elif stop == "length":
            status = "incomplete"
        elif stop == "stop" and isinstance(content, str) and not _get(message, "tool_calls"):
            status = "completed"
        else:
            status = "failed"
        return ModelResponse(
            content if isinstance(content, str) else "",
            _usage(
                _get(result, "usage"), input_key="prompt_tokens", output_key="completion_tokens"
            ),
            stop,
            self.provider,
            _get(result, "model") or self.model,
            _get(result, "_request_id") or _get(result, "id"),
            status,
            refusal,
            {
                "response_id": _get(result, "id"),
                "system_fingerprint": _get(result, "system_fingerprint"),
            },
        )


def create_model(
    provider: str,
    model: str,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    **kwargs: Any,
) -> OpenAIModelClient | AnthropicModelClient | OpenAICompatibleModelClient:
    """Construct a configured adapter. Never infer a provider from a model ID."""
    common = {"api_key_env": api_key_env, **kwargs}
    if provider == "openai":
        return OpenAIModelClient(model, base_url=base_url, **common)
    if provider == "anthropic":
        return AnthropicModelClient(model, base_url=base_url, **common)
    if provider in {"openai_compatible", "openai-compatible"}:
        if not base_url:
            raise ConfigurationError("An OpenAI-compatible provider requires base_url")
        return OpenAICompatibleModelClient(model, base_url=base_url, **common)
    raise ConfigurationError(f"Unsupported model provider: {provider!r}")
