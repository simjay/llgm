"""Explicitly opted-in provider contracts. No dataset or evaluator labels used."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict

import pytest

from llgm.models import (
    AnthropicModelClient,
    Message,
    ModelRequest,
    OpenAIEmbeddingClient,
    OpenAIModelClient,
)

pytestmark = [pytest.mark.integration, pytest.mark.live]
SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "count": {"type": "integer", "enum": [2]},
    },
    "required": ["status", "count"],
    "additionalProperties": False,
}


async def _schema_check(client_class, model, record):
    """A hosted response must satisfy the schema and report its pinned identity and usage."""
    record["models"].append({"configured": model, "provider": client_class.provider})
    async with client_class(model, timeout_seconds=60) as client:
        result = await client.complete(
            ModelRequest(
                (Message("user", 'Return the object with status "ok" and count 2.'),),
                max_output_tokens=1024,
                output_schema=SCHEMA,
            )
        )
        record["operations"].append(
            {
                "operation": "native_schema",
                "model": result.model,
                "provider": result.provider,
                "request_id": result.request_id,
                "status": result.status,
                "usage": asdict(result.usage),
            }
        )
        result.ensure_complete()
        assert result.model == model, (
            "Provider resolved a different model; pin the actual returned model ID"
        )
        assert json.loads(result.text) == {"status": "ok", "count": 2}
        assert result.request_id
        assert result.usage.input_tokens is not None and result.usage.input_tokens > 0
        assert result.usage.output_tokens is not None and result.usage.output_tokens > 0


@pytest.mark.openai
def test_openai_native_schema(live_openai, integration_record):
    """The configured OpenAI adapter honors native schema and accounting contracts."""
    asyncio.run(_schema_check(OpenAIModelClient, live_openai, integration_record[0]))


@pytest.mark.anthropic
def test_anthropic_native_schema(live_anthropic, integration_record):
    """The configured Anthropic adapter honors native schema and accounting contracts."""
    asyncio.run(_schema_check(AnthropicModelClient, live_anthropic, integration_record[0]))


@pytest.mark.embedding
def test_openai_embedding_batch_contract(live_embedding, integration_record):
    """Embedding batches preserve input cardinality, dimensions and physical-call accounting."""
    record, _ = integration_record
    record["models"].append({"configured": live_embedding, "provider": "openai"})

    async def scenario():
        """Send three synthetic strings through two actual embedding requests."""
        async with OpenAIEmbeddingClient(
            live_embedding, batch_size=2, timeout_seconds=60
        ) as client:
            try:
                vectors = await client.embed(
                    ["A blue circle.", "A red square.", "A green triangle."]
                )
                assert len(vectors) == 3
                assert len({len(vector) for vector in vectors}) == 1
                assert len(vectors[0]) > 0
                assert all(math.isfinite(value) for vector in vectors for value in vector)
                assert [event.input_count for event in client.events] == [2, 1]
                assert all(event.model == live_embedding for event in client.events)
                assert all(
                    event.status == "completed" and event.request_id for event in client.events
                )
                assert all(
                    event.usage.input_tokens is not None and event.usage.input_tokens > 0
                    for event in client.events
                )
            finally:
                record["operations"].extend(
                    {"operation": "embedding", **asdict(event)} for event in client.events
                )

    asyncio.run(scenario())
