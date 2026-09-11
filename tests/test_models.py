"""Provider contract tests using SDK-shaped fakes; no API keys or network."""

from __future__ import annotations

import asyncio
import inspect
import json
import unittest
from types import SimpleNamespace as NS

from llgm.core.errors import CapabilityError, ConfigurationError, ProviderError
from llgm.models import (
    AnthropicModelClient,
    CallableModelClient,
    Message,
    ModelRequest,
    ModelResponse,
    OpenAICompatibleModelClient,
    OpenAIEmbeddingClient,
    OpenAIModelClient,
    ScriptedModelClient,
    Usage,
    create_model,
)


class FakeSDK:
    """SDK-shaped transport double with explicit request, option and lifecycle records."""

    def __init__(self, *responses):
        """Queue provider responses behind the endpoint shapes used by all adapters."""
        self.results = iter(responses)
        self.requests = []
        self.options = []
        self.closed = False
        self.responses = NS(create=self.create)
        self.messages = NS(create=self.create)
        self.chat = NS(completions=NS(create=self.create))
        self.embeddings = NS(create=self.create)

    def with_options(self, **kwargs):
        """Record per-request SDK options while retaining this borrowed client."""
        self.options.append(kwargs)
        return self

    async def create(self, **kwargs):
        """Capture one endpoint request and return or raise its queued outcome."""
        self.requests.append(kwargs)
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result

    async def close(self):
        """Expose whether an adapter incorrectly closes the borrowed transport."""
        self.closed = True


REQUEST = ModelRequest((Message("system", "Be precise."), Message("user", "Find evidence.")))
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class ModelTests(unittest.IsolatedAsyncioTestCase):
    """Provider-neutral requests, status handling and usage-accounting contracts."""

    async def test_scripted_unknown_usage_and_exhaustion(self):
        """Scripted strings have unknown usage and exhaustion never invents a response."""
        client = ScriptedModelClient(["first", ModelResponse("second", Usage(2, 3))])
        first = await client.complete(REQUEST)
        self.assertIsNone(first.usage.total_tokens)
        second = await client.complete(REQUEST)
        self.assertEqual(second.usage.total_tokens, 5)
        with self.assertRaises(ProviderError):
            await client.complete(REQUEST)
        self.assertEqual(len(client.requests), 3)

    async def test_callable_contract_and_cancellation(self):
        """Callable adapters require awaitable typed responses and preserve cancellation."""

        async def generate(request):
            """Return the final input message through the async model contract."""
            return ModelResponse(request.messages[-1].content)

        client = CallableModelClient(generate)
        self.assertEqual((await client.complete(REQUEST)).text, "Find evidence.")

        async def cancel(request):
            """Raise cancellation directly from the injected provider callback."""
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await CallableModelClient(cancel).complete(REQUEST)
        with self.assertRaises(ProviderError):
            await CallableModelClient(lambda request: "not awaitable").complete(REQUEST)

    async def test_openai_native_schema_usage_and_actual_identity(self):
        """Native OpenAI schema requests retain actual model identity and cached-token details."""
        sdk = FakeSDK(
            NS(
                id="response-1",
                _request_id="http-1",
                model="resolved-model",
                status="completed",
                usage=NS(
                    input_tokens=11, output_tokens=7, input_tokens_details=NS(cached_tokens=4)
                ),
                output=[
                    NS(type="reasoning"),
                    NS(type="message", content=[NS(type="output_text", text='{"answer":"yes"}')]),
                ],
            )
        )
        model = OpenAIModelClient("configured-model", client=sdk)
        result = await model.complete(ModelRequest(REQUEST.messages, 256, output_schema=SCHEMA))
        result.ensure_complete()
        self.assertEqual(result.model, "resolved-model")
        self.assertEqual(result.request_id, "http-1")
        self.assertEqual(result.usage.total_tokens, 18)
        self.assertEqual(result.usage.extra["input_tokens_details"]["cached_tokens"], 4)
        self.assertEqual(sdk.requests[0]["text"]["format"]["schema"], SCHEMA)
        self.assertFalse(sdk.requests[0]["store"])
        self.assertNotIn("temperature", sdk.requests[0])
        self.assertNotIn("reasoning", sdk.requests[0])
        self.assertNotIn("reasoning_effort", model.descriptor())
        self.assertEqual(sdk.options[0]["max_retries"], 0)
        await model.aclose()
        self.assertFalse(sdk.closed, "Borrowed SDK lifecycle belongs to caller")

    async def test_openai_explicit_reasoning_preserves_transport_and_ownership(self):
        """Configured reasoning reaches Responses while existing SDK settings and ownership remain intact."""
        sdk = FakeSDK({"status": "completed", "output_text": "{}"})
        model = OpenAIModelClient(
            "reasoning-model",
            client=sdk,
            base_url="https://example.test/v1",
            timeout_seconds=17,
            reasoning_effort="medium",
        )
        await model.complete(ModelRequest(REQUEST.messages, 256, output_schema=SCHEMA))
        self.assertEqual(sdk.requests[0]["reasoning"], {"effort": "medium"})
        self.assertNotIn("temperature", sdk.requests[0])
        self.assertEqual(sdk.requests[0]["text"]["format"]["schema"], SCHEMA)
        self.assertEqual(sdk.options, [{"max_retries": 0, "timeout": 17}])
        self.assertEqual(model.descriptor()["reasoning_effort"], "medium")
        self.assertEqual(model.descriptor()["base_url"], "https://example.test/v1")
        await model.aclose()
        self.assertFalse(sdk.closed)

    async def test_openai_reasoning_temperature_conflict_precedes_sdk_dispatch(self):
        """Explicit reasoning rejects sampling instead of silently altering a configured request."""
        sdk = FakeSDK()
        model = OpenAIModelClient("reasoning-model", client=sdk, reasoning_effort="medium")
        with self.assertRaisesRegex(CapabilityError, "temperature=None"):
            await model.complete(ModelRequest(REQUEST.messages, temperature=0))
        self.assertFalse(sdk.requests)
        for effort in (None, "none"):
            with self.subTest(effort=effort):
                sdk = FakeSDK({"status": "completed", "output_text": "ok"})
                model = OpenAIModelClient("model", client=sdk, reasoning_effort=effort)
                await model.complete(ModelRequest(REQUEST.messages, temperature=0))
                self.assertEqual(sdk.requests[0]["temperature"], 0)
                self.assertEqual(
                    sdk.requests[0].get("reasoning"), {"effort": effort} if effort else None
                )

    def test_openai_invalid_reasoning_is_rejected_before_sdk_configuration(self):
        """Non-string and blank effort settings cannot initialize a request transport."""
        for effort in ("", " \t", True, 1, {}):
            with self.subTest(effort=effort):
                sdk = FakeSDK()
                with self.assertRaises(ConfigurationError):
                    OpenAIModelClient("model", client=sdk, reasoning_effort=effort)
                self.assertFalse(sdk.options)
                self.assertFalse(sdk.requests)

    async def test_openai_refusal_and_incomplete_are_not_ordinary_text(self):
        """Refusals and incomplete outputs cannot masquerade as completed responses."""
        sdk = FakeSDK(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "Cannot comply"}],
                    }
                ],
            },
            {
                "status": "incomplete",
                "output_text": "partial",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        )
        model = OpenAIModelClient("model", client=sdk)
        refused = await model.complete(REQUEST)
        self.assertEqual(refused.status, "refused")
        self.assertEqual(refused.refusal, "Cannot comply")
        self.assertIsNone(refused.usage.input_tokens)
        with self.assertRaises(ProviderError):
            refused.ensure_complete()
        partial = await model.complete(REQUEST)
        self.assertEqual(partial.text, "partial")
        self.assertEqual(partial.finish_reason, "max_output_tokens")
        with self.assertRaises(ProviderError):
            partial.ensure_complete()

    async def test_unexpected_output_is_not_silently_discarded(self):
        """Unsupported provider output blocks force an explicit failure status."""
        sdk = FakeSDK(
            {"status": "completed", "output": [{"type": "function_call", "name": "external"}]}
        )
        result = await OpenAIModelClient("model", client=sdk).complete(REQUEST)
        self.assertEqual(result.status, "failed")
        with self.assertRaises(ProviderError):
            result.ensure_complete()

    async def test_anthropic_native_system_schema_and_cache_accounting(self):
        """Anthropic schema transport preserves system instructions and cache billing categories."""
        sdk = FakeSDK(
            NS(
                id="message-1",
                model="claude-resolved",
                stop_reason="end_turn",
                content=[NS(type="text", text='{"answer":"yes"}')],
                usage=NS(
                    input_tokens=10,
                    cache_creation_input_tokens=20,
                    cache_read_input_tokens=30,
                    output_tokens=4,
                ),
            )
        )
        model = AnthropicModelClient("claude-selected", client=sdk)
        result = await model.complete(ModelRequest(REQUEST.messages, output_schema=SCHEMA))
        self.assertEqual(result.usage.input_tokens, 60)
        self.assertEqual(result.usage.total_tokens, 64)
        self.assertEqual(sdk.requests[0]["system"], "Be precise.")
        self.assertEqual(
            sdk.requests[0]["messages"], [{"role": "user", "content": "Find evidence."}]
        )
        self.assertEqual(sdk.requests[0]["output_config"]["format"]["schema"], SCHEMA)
        result.ensure_complete()

    async def test_anthropic_rejects_unrepresentable_instruction_roles(self):
        """Unsupported Anthropic instruction-role layouts fail before any SDK request."""
        sdk = FakeSDK()
        model = AnthropicModelClient("model", client=sdk)
        for messages in [
            (Message("developer", "rules"), Message("user", "question")),
            (Message("user", "question"), Message("system", "late rules")),
        ]:
            with self.assertRaises(CapabilityError):
                await model.complete(ModelRequest(messages))
        self.assertFalse(sdk.requests)

    async def test_anthropic_truncation_and_missing_usage(self):
        """Anthropic truncation remains incomplete and absent usage stays unknown."""
        for reason in ("max_tokens", "model_context_window_exceeded", "pause_turn"):
            model = AnthropicModelClient(
                "model", client=FakeSDK({"content": [], "stop_reason": reason})
            )
            result = await model.complete(REQUEST)
            self.assertEqual(result.status, "incomplete")
            self.assertIsNone(result.usage.total_tokens)

    async def test_compatible_schema_requires_explicit_capability(self):
        """Compatible endpoints require declared native-schema capability."""
        sdk = FakeSDK()
        model = OpenAICompatibleModelClient(
            "model", base_url="http://localhost:8000/v1", client=sdk
        )
        with self.assertRaises(CapabilityError):
            await model.complete(ModelRequest(REQUEST.messages, output_schema=SCHEMA))
        self.assertFalse(sdk.requests)

    async def test_compatible_schema_payload_and_length_status(self):
        """Compatible schema payloads use the configured token-limit field and retain truncation."""
        sdk = FakeSDK(
            {
                "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )
        model = OpenAICompatibleModelClient(
            "model",
            base_url="https://example.test/v1",
            client=sdk,
            supports_structured_output=True,
            max_tokens_parameter="max_completion_tokens",
        )
        result = await model.complete(ModelRequest(REQUEST.messages, 20, output_schema=SCHEMA))
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.usage.total_tokens, 120)
        self.assertEqual(sdk.requests[0]["max_completion_tokens"], 20)
        self.assertEqual(sdk.requests[0]["response_format"]["json_schema"]["schema"], SCHEMA)

    async def test_sdk_errors_redact_secrets_and_are_not_retried(self):
        """Provider exceptions are sanitized and never trigger hidden adapter retries."""
        sdk = FakeSDK(RuntimeError("secret-api-key and private prompt"))
        model = OpenAIModelClient("model", client=sdk)
        with self.assertRaises(ProviderError) as caught:
            await model.complete(REQUEST)
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("private", str(caught.exception))
        self.assertEqual(len(sdk.requests), 1)
        self.assertNotIn("api_key", json.dumps(model.descriptor()))

    async def test_sdk_cancellation_propagates(self):
        """SDK cancellation escapes the adapter rather than becoming an ordinary error."""
        sdk = FakeSDK(asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await OpenAIModelClient("model", client=sdk).complete(REQUEST)

    async def test_installed_openai_sdk_accepts_native_payload(self):
        """The installed OpenAI SDK accepts the adapter's native request shape."""
        try:
            from openai.resources.responses.responses import AsyncResponses
        except ImportError:
            self.skipTest("Optional OpenAI SDK is not installed")
        sdk = FakeSDK({"status": "completed", "output_text": "{}"})
        model = OpenAIModelClient("model", client=sdk, reasoning_effort="medium")
        await model.complete(ModelRequest(REQUEST.messages, output_schema=SCHEMA))
        inspect.signature(AsyncResponses.create).bind(None, **sdk.requests[0])

    async def test_installed_anthropic_sdk_accepts_native_payload(self):
        """The installed Anthropic SDK accepts the adapter's native request shape."""
        try:
            from anthropic.resources.messages.messages import AsyncMessages
        except ImportError:
            self.skipTest("Optional Anthropic SDK is not installed")
        sdk = FakeSDK({"stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}]})
        model = AnthropicModelClient("model", client=sdk)
        await model.complete(ModelRequest(REQUEST.messages, temperature=0.1, output_schema=SCHEMA))
        inspect.signature(AsyncMessages.create).bind(None, **sdk.requests[0])

    def test_configuration_rejects_unknown_provider_and_credential_urls(self):
        """Unknown providers, missing endpoint details and credential URLs fail configuration."""
        with self.assertRaises(ConfigurationError):
            create_model("made-up", "model")
        with self.assertRaises(ConfigurationError):
            create_model("openai_compatible", "model")
        with self.assertRaises(ConfigurationError):
            OpenAIModelClient("model", client=FakeSDK(), base_url="https://secret@example.test/v1")
        with self.assertRaises(ConfigurationError):
            ModelRequest(REQUEST.messages, max_output_tokens=0)
        for value in (True, float("inf"), float("nan")):
            with self.assertRaises(ConfigurationError):
                ModelRequest(REQUEST.messages, temperature=value)
            with self.assertRaises(ConfigurationError):
                OpenAIModelClient("model", client=FakeSDK(), timeout_seconds=value)

    async def test_embedding_batch_order_dimensions_and_usage_events(self):
        """Embedding outputs follow input indices and record every physical batch."""
        sdk = FakeSDK(
            {
                "model": "resolved-embedding",
                "data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}],
                "usage": {"prompt_tokens": 8},
            },
            {"data": [{"index": 0, "embedding": [1, 1]}]},
        )
        client = OpenAIEmbeddingClient(client=sdk, dimensions=2, batch_size=2)
        vectors = await client.embed(["one", "two", "three"])
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.assertEqual(len(client.events), 2)
        self.assertEqual(client.events[0].usage.total_tokens, 8)
        self.assertIsNone(client.events[1].usage.total_tokens)
        self.assertEqual(client.events[0].model, "resolved-embedding")
        self.assertEqual(sdk.requests[0]["dimensions"], 2)
        self.assertEqual(await client.embed([]), [])
        self.assertEqual(len(sdk.requests), 2)

    async def test_embedding_bad_response_preserves_billed_usage(self):
        """Malformed embedding vectors still retain the provider's billed input usage."""
        sdk = FakeSDK({"data": [{"index": 1, "embedding": [1, 0]}], "usage": {"prompt_tokens": 7}})
        client = OpenAIEmbeddingClient(client=sdk)
        with self.assertRaises(ProviderError):
            await client.embed(["text"])
        self.assertEqual(client.events[0].status, "failed")
        self.assertEqual(client.events[0].usage.input_tokens, 7)

    async def test_embedding_failure_has_unknown_usage(self):
        """Failed embedding requests retain an attempt with unknown usage."""
        client = OpenAIEmbeddingClient(client=FakeSDK(RuntimeError("remote failure")))
        with self.assertRaises(ProviderError):
            await client.embed(["text"])
        self.assertIsNone(client.events[0].usage.total_tokens)
        self.assertEqual(client.events[0].status, "failed")

    async def test_embedding_indices_require_unique_bounded_integers(self):
        """Malformed provider indexes fail consistently while preserving observed usage."""
        for indices in ([False], [0.0], [None], [[]], ["0"], [0, 0], [-1], [1]):
            with self.subTest(indices=indices):
                sdk = FakeSDK(
                    {
                        "data": [{"index": index, "embedding": [1, 0]} for index in indices],
                        "usage": {"prompt_tokens": 7},
                    }
                )
                client = OpenAIEmbeddingClient(client=sdk)
                with self.assertRaisesRegex(ProviderError, "indices"):
                    await client.embed(["source"] * len(indices))
                self.assertEqual(client.events[0].status, "failed")
                self.assertEqual(client.events[0].usage.input_tokens, 7)


class OpenAIPhaseTests(unittest.IsolatedAsyncioTestCase):
    """Responses commentary stays separate from the completed model protocol output."""

    async def test_final_phase_is_selected_and_normalized_turn_replayed(self):
        """Only final output reaches the executor and replay identifies it as final."""
        sdk = FakeSDK(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": '{"op":"delegate"}'}],
                    },
                    {
                        "type": "message",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": '{"op":"finish"}'}],
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        )
        messages = (
            *REQUEST.messages,
            Message("assistant", '{"op":"read"}'),
            Message("user", "result"),
        )
        result = await OpenAIModelClient("model", client=sdk).complete(ModelRequest(messages))
        self.assertEqual(result.text, '{"op":"finish"}')
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.metadata["output_message_phases"], ["commentary", "final_answer"])
        self.assertEqual(result.metadata["discarded_commentary_messages"], 1)
        self.assertEqual(result.usage.total_tokens, 30)
        self.assertEqual(sdk.requests[0]["input"][2]["phase"], "final_answer")
        self.assertTrue(
            all(
                "phase" not in item
                for item in sdk.requests[0]["input"]
                if item["role"] != "assistant"
            )
        )

    async def test_commentary_only_cannot_become_final_through_sdk_fallback(self):
        """A completed transport with only commentary remains an incomplete model answer."""
        sdk = FakeSDK(
            {
                "status": "completed",
                "output_text": "unexecuted commentary",
                "output": [
                    {
                        "type": "message",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": "unexecuted commentary"}],
                    }
                ],
            }
        )
        result = await OpenAIModelClient("model", client=sdk).complete(REQUEST)
        self.assertEqual(result.text, "")
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.finish_reason, "missing_final_answer")
        with self.assertRaises(ProviderError):
            result.ensure_complete()

    async def test_legacy_output_keeps_text_blocks_and_unknown_phases_fail(self):
        """Unphased legacy blocks combine, while unknown provider phases fail explicitly."""
        legacy = {"type": "message", "content": [{"type": "output_text", "text": "part"}]}
        sdk = FakeSDK({"status": "completed", "output": [legacy, legacy]})
        result = await OpenAIModelClient("model", client=sdk).complete(REQUEST)
        self.assertEqual(result.text, "partpart")
        self.assertEqual(result.status, "completed")
        sdk = FakeSDK(
            {"status": "completed", "output": [{**legacy, "phase": "new-unsupported-phase"}]}
        )
        result = await OpenAIModelClient("model", client=sdk).complete(REQUEST)
        self.assertEqual(result.status, "failed")


if __name__ == "__main__":
    unittest.main()
