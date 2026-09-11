"""Opt-in hosted-model and Docker check of the executable recursive RLM loop.

Set both LLGM_TEST_RLM=1 and LLGM_TEST_DOCKER=1. Declare ROOT and CHILD
provider/model pairs through LLGM_TEST_RLM_<ROLE>_PROVIDER and
LLGM_TEST_RLM_<ROLE>_MODEL, plus their normal provider credentials. Both roles
may use the same explicit model ID. LLGM_REPL_DOCKER_IMAGE must name a trusted
local Python image; its inspected immutable image ID is used for execution.
The test never pulls images or downloads data. Disabled gates make no calls.

The synthetic prompt prescribes selection, delegation, and continuation. This
is an executable protocol diagnostic, not an autonomous-planning benchmark.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
from contextlib import AsyncExitStack
from dataclasses import asdict

import pytest

from llgm.evaluation.artifacts import write_json, write_jsonl

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.recursive, pytest.mark.docker]


def _required_env(name):
    """Require explicit integration settings without exposing credential values."""
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.fail(f"Opted-in RLM integration requires {name}", pytrace=False)
    return value


def _docker_json(*arguments):
    """Read bounded local Docker metadata without running or pulling an image."""
    try:
        result = subprocess.run(
            ["docker", *arguments], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        pytest.fail(
            "Opted-in RLM integration requires a working local Docker daemon", pytrace=False
        )
    if result.returncode:
        pytest.fail(
            "Opted-in RLM integration requires the local daemon and configured image; "
            "no image was pulled",
            pytrace=False,
        )
    try:
        return json.loads(result.stdout)
    except ValueError:
        pytest.fail("Docker inspection returned malformed metadata", pytrace=False)


@pytest.fixture
def live_rlm_config():
    """Validate both gates, hosted identities, credentials, and a local image pin."""
    gates = {name: os.environ.get(name, "0") for name in ("LLGM_TEST_RLM", "LLGM_TEST_DOCKER")}
    for name, value in gates.items():
        if value not in {"0", "1"}:
            pytest.fail(f"{name} must be exactly 0 or 1", pytrace=False)
    if any(value == "0" for value in gates.values()):
        pytest.skip("Set both LLGM_TEST_RLM=1 and LLGM_TEST_DOCKER=1")
    models = {}
    for role in ("ROOT", "CHILD"):
        provider = _required_env(f"LLGM_TEST_RLM_{role}_PROVIDER")
        if provider not in {"openai", "anthropic"}:
            pytest.fail("RLM live test requires an openai or anthropic provider", pytrace=False)
        model = _required_env(f"LLGM_TEST_RLM_{role}_MODEL")
        if "latest" in model.lower():
            pytest.fail(f"LLGM_TEST_RLM_{role}_MODEL must not use a latest alias", pytrace=False)
        _required_env("OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY")
        if importlib.util.find_spec(provider) is None:
            pytest.fail(f"Opted-in RLM integration requires the {provider} SDK", pytrace=False)
        models[role.lower()] = {"provider": provider, "model": model}
    image = _required_env("LLGM_REPL_DOCKER_IMAGE")
    server_version = _docker_json("version", "--format", "{{json .Server.Version}}")
    image_id = _docker_json("image", "inspect", "--format", "{{json .Id}}", image)
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        pytest.fail("Docker image inspection did not return an immutable image ID", pytrace=False)
    return {
        "models": models,
        "docker": {
            "requested_image": image,
            "image_id": image_id,
            "server_version": server_version,
        },
    }


def _synthetic_context():
    """Build seeded revisions with the expected token available only in source records."""
    rng = random.Random(1729)
    records = []
    for project in range(24):
        for revision, state in ((1, "confirmed"), (2, "confirmed"), (3, "proposed")):
            records.append(
                {
                    "project_id": f"cedar-{project:02d}",
                    "revision": revision,
                    "state": state,
                    "release_token": f"release-{rng.getrandbits(128):032x}",
                }
            )
    rng.shuffle(records)
    expected = next(
        row["release_token"]
        for row in records
        if row["project_id"] == "cedar-17" and row["revision"] == 2
    )
    return {"records": records}, expected


def _trace_metadata(trace):
    """Preserve execution identities and counts without prompts or provider error bodies."""
    allowed = {
        "kind",
        "invocation_id",
        "parent_id",
        "depth",
        "role",
        "attempt",
        "step",
        "status",
        "provider",
        "model",
        "request_id",
        "input_tokens",
        "output_tokens",
        "context_accounting_units",
        "error_type",
        "reason",
        "code_sha256",
        "code_bytes",
        "stdout_bytes",
        "stdout_truncated",
        "llm_queries",
        "prompt_bytes",
        "response_bytes",
        "context_sha256",
        "context_bytes",
        "execution",
        "sha256",
        "bytes",
        "accounting_units",
        "container_name",
        "image",
    }
    return [{key: value for key, value in event.items() if key in allowed} for event in trace]


def test_hosted_rlm_selects_external_context_and_recurses(live_rlm_config, integration_record):
    """Generated Python delegates to a real child REPL and resumes with the hidden token."""
    from llgm.inference.budget import Budget
    from llgm.inference.repl import DockerREPLConfig
    from llgm.inference.rlm import RLMRuntime
    from llgm.models import create_model

    record, directory = integration_record
    context, expected = _synthetic_context()
    context_json = json.dumps(context, ensure_ascii=False, allow_nan=False)
    budget = Budget(
        max_model_calls=12,
        max_sidecar_calls=8,
        max_searches=1,
        max_evidence_tokens=32000,
        max_bundle_tokens=8000,
        max_context_tokens=48000,
        max_output_tokens=2048,
        timeout_seconds=300,
    )
    repl_config = DockerREPLConfig(
        image=live_rlm_config["docker"]["image_id"],
        execution_timeout_seconds=180,
        max_output_bytes=8192,
        max_llm_queries=2,
        max_session_llm_queries=2,
    )
    prompt = (
        "This is a prescribed RLM protocol diagnostic. Determine the release_token of the "
        "highest confirmed revision for project cedar-17. Proposed revisions do not count. "
        "The source data exists only in your external Python context['records']; each record "
        "has project_id, revision, state, and release_token. Complete these steps in separate "
        "python operations, using persistent variables between operations. First, select only "
        "cedar-17 records into a variable called selected_rows and print only their count. "
        "Second, import json and call llm_query with json.dumps of an object containing "
        "'records': selected_rows and an 'instructions' string. Those instructions must tell "
        "the child to execute Python, parse json.loads(context['prompt']), select the highest "
        "revision whose state is confirmed from its records, print its release_token, and "
        "finish with only that token. Require the child to use Python and not delegate again. "
        "Save the llm_query return value in child_answer; print only 'child returned'. "
        "Third, in another python operation print the saved child_answer variable. Finally, "
        "finish with only that returned release token. Do not print all source records or "
        "solve the token directly at the root. Use the required python/finish JSON operations."
    )
    assert expected not in prompt
    record.update(
        scope="prescribed synthetic RLM code/recursion protocol; not autonomous planning or benchmark quality",
        models=live_rlm_config["models"],
        docker=live_rlm_config["docker"],
        budget=asdict(budget),
        repl=asdict(repl_config),
        fixture={
            "name": "seeded-release-revisions-v1",
            "seed": 1729,
            "source_records": len(context["records"]),
            "context_sha256": hashlib.sha256(context_json.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "answer_source": "external context only",
        },
        runtime_limits={"max_depth": 1, "max_steps": 8, "max_executions": 12},
    )

    async def scenario():
        """Run both hosted roles and retain auditable attempted work before assertions."""
        async with AsyncExitStack() as stack:
            models = {
                role: await stack.enter_async_context(create_model(**config, timeout_seconds=90))
                for role, config in live_rlm_config["models"].items()
            }
            runtime = RLMRuntime(
                models["root"],
                models["child"],
                budget=budget,
                repl_config=repl_config,
                max_depth=1,
                max_steps=8,
                max_executions=12,
                capture_text=True,
            )
            try:
                result = await runtime.answer(prompt, context=context)
            finally:
                record["operations"] = _trace_metadata(runtime.last_trace)
                record["usage"] = runtime.last_usage
                record["provenance"] = runtime.last_provenance
                write_json(directory / "runtime-attempt.json", record)
                # This fixture is public synthetic data; keep generated code and
                # malformed responses for protocol diagnosis, not just their hashes.
                write_json(directory / "protocol-transcript.json", runtime.last_trace)
            record["result"] = {
                "status": result.status,
                "unresolved_count": len(result.unresolved),
                "unresolved": result.unresolved,
                "provenance": result.provenance,
            }
            predictions = directory / "predictions.jsonl"
            write_jsonl(
                predictions,
                [{"question_id": "seeded-release-revisions-v1", "hypothesis": result.answer}],
            )
            write_json(
                directory / "prediction-manifest.json",
                {
                    "schema_version": 1,
                    "fixture": record["fixture"],
                    "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                    "models": live_rlm_config["models"],
                    "docker": live_rlm_config["docker"],
                    "scope": record["scope"],
                    "benchmark_result": False,
                    "status": result.status,
                },
            )
            write_json(directory / "runtime-attempt.json", record)
            assert result.status == "completed", "Inspect the recorded RLM protocol attempt"
            assert result.answer.strip() == expected
            assert not result.unresolved
            assert result.provenance["root_context_sha256"] == record["fixture"]["context_sha256"]
            assert result.provenance["repl"]["image"] == live_rlm_config["docker"]["image_id"]
            events = result.trace
            entered = [event for event in events if event.get("kind") == "enter"]
            roots = [event for event in entered if event["depth"] == 0]
            children = [event for event in entered if event["depth"] == 1]
            assert len(roots) == len(children) == 1
            assert len(entered) == 2
            root_id, child_id = roots[0]["invocation_id"], children[0]["invocation_id"]
            assert children[0]["parent_id"] == root_id
            assert children[0]["context_bytes"] < roots[0]["context_bytes"]
            python_events = [event for event in events if event.get("kind") == "python"]
            assert sum(event["invocation_id"] == root_id for event in python_events) >= 3
            assert any(event["invocation_id"] == child_id for event in python_events)
            assert all(event["bytes"] > 0 and event["sha256"] for event in python_events)
            returned = next(
                index
                for index, event in enumerate(events)
                if event.get("kind") == "child_return" and event["invocation_id"] == root_id
            )
            assert any(
                event.get("kind") == "delegate" and event["invocation_id"] == root_id
                for event in events[:returned]
            )
            assert any(
                event.get("kind") == "return" and event["invocation_id"] == child_id
                for event in events[:returned]
            )
            assert any(
                event.get("kind") == "python" and event["invocation_id"] == root_id
                for event in events[returned + 1 :]
            )
            assert any(
                event.get("kind") == "model" and event["invocation_id"] == root_id
                for event in events[returned + 1 :]
            )
            calls = [event for event in events if event.get("kind") == "model"]
            assert {event["role"] for event in calls} == {"root", "sidecar"}
            for event in calls:
                role = "root" if event["role"] == "root" else "child"
                declared = live_rlm_config["models"][role]
                assert event["provider"] == declared["provider"]
                assert event["model"] == declared["model"]
                assert event["status"] == "completed"
                assert event["input_tokens"] is not None
                assert event["output_tokens"] is not None
            assert result.usage["model_calls"] == len(calls)
            assert result.usage["sidecar_calls"] == sum(
                event["role"] == "sidecar" for event in calls
            )
            assert result.usage["python_executions"] == len(python_events)
            assert result.usage["recursive_invocations"] == 1
            assert result.usage["known_input_tokens"] > 0
            assert result.usage["known_output_tokens"] > 0
            assert result.usage["unknown_usage_calls"] == 0
            opened = [event for event in events if event.get("kind") == "repl_open"]
            closed = [event for event in events if event.get("kind") == "repl_closed"]
            assert {event["invocation_id"] for event in opened} == {root_id, child_id}
            assert len({event["container_name"] for event in opened}) == 2
            assert {event["container_name"] for event in opened} == {
                event["container_name"] for event in closed
            }

    asyncio.run(scenario())
