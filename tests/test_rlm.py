"""RLM orchestration contracts with scripted models and a controlled REPL boundary.

The fake interpreter never evaluates Python. Real generated-code execution is
covered separately by the explicitly opted-in Docker and hosted-model test.
"""

from __future__ import annotations

import asyncio
import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from llgm.core.errors import ConfigurationError, SchemaError
from llgm.inference.budget import Budget
from llgm.inference.repl import (
    DockerREPL,
    DockerREPLConfig,
    REPLError,
    REPLResult,
    REPLTimeoutError,
)
from llgm.inference.rlm import RLMRuntime
from llgm.models import CallableModelClient, ModelResponse, ScriptedModelClient, Usage


def python(code):
    """Build a controller response that requests Python execution."""
    return {"op": "python", "code": code}


def finish(answer):
    """Build a controller response that explicitly ends its invocation."""
    return {"op": "finish", "answer": answer}


def model(*operations, name="scripted", usage=None):
    """Provide finite model outputs with known identity and optional measured usage."""
    return ScriptedModelClient(
        [
            ModelResponse(
                json.dumps(operation) if isinstance(operation, dict) else operation,
                usage or Usage(),
                provider="test",
                model=name,
            )
            for operation in operations
        ]
    )


class FakeREPL:
    """Record interpreter calls and return controlled outcomes without executing code."""

    def __init__(self, owner, context, config, llm_query):
        """Capture an independent context and this frame's recursive callback."""
        self.owner = owner
        self.context = copy.deepcopy(context)
        self.config, self.callback = config, llm_query
        self.index = len(owner.sessions)
        self.container_name = f"fake-container-{self.index}"
        self.plan = iter(owner.plans[self.index] if self.index < len(owner.plans) else [])
        self.code = []
        self.closed = False
        owner.sessions.append(self)

    async def start(self):
        """Allow tests to fail interpreter startup before a paid model invocation."""
        if self.owner.start_error:
            raise self.owner.start_error

    async def execute(self, code):
        """Return the next observation or await the test's explicit callback action."""
        self.code.append(code)
        operation = next(self.plan)
        if isinstance(operation, BaseException):
            raise operation
        if callable(operation):
            return await operation(self)
        return REPLResult(operation, None, False, 0)

    async def aclose(self):
        """Record cleanup order for ordinary exits, failures, and cancellation."""
        self.closed = True
        self.owner.closed.append(self.index)
        if self.owner.close_error:
            raise self.owner.close_error


class REPLFactory:
    """Supply one controlled execution plan per newly created recursive frame."""

    def __init__(self, *plans, start_error=None, close_error=None):
        """Retain interpreter plans and lifecycle records for contract assertions."""
        self.plans, self.start_error = plans, start_error
        self.close_error = close_error
        self.sessions, self.closed = [], []

    def __call__(self, context, *, config, llm_query):
        """Construct the next fake frame at the real DockerREPL injection boundary."""
        return FakeREPL(self, context, config, llm_query)


class RLMTests(unittest.IsolatedAsyncioTestCase):
    """Check model-driven loops, recursive isolation, and whole-run failure boundaries."""

    async def test_external_context_is_not_initial_model_context(self):
        """A source is exposed only through interpreter observations, not initial prompts."""
        source = "private evidence value 7392"
        root = model(python("print(context['document'])"), finish("7392"), name="root-v1")
        factory = REPLFactory([source])
        with patch("llgm.inference.rlm.DockerREPL", factory):
            runtime = RLMRuntime(root)
            actual = await runtime.answer("What is the value?", context={"document": source})
        self.assertEqual(actual.status, "completed")
        self.assertEqual(actual.answer, "7392")
        self.assertNotIn(source, str(root.requests[0].messages))
        self.assertIn(source, root.requests[1].messages[-1].content)
        self.assertEqual(factory.sessions[0].context, {"document": source})
        self.assertEqual(factory.sessions[0].code, ["print(context['document'])"])
        self.assertEqual(actual.usage["python_executions"], 1)
        self.assertEqual(actual.usage["unknown_usage_calls"], 2)
        self.assertNotIn(source, json.dumps(actual.trace))
        self.assertFalse(actual.provenance["paper_reproduction"])
        self.assertEqual(factory.closed, [0])

    async def test_root_child_grandchild_share_budget_and_keep_independent_contexts(self):
        """Recursive Python callbacks run child model loops and resume both ancestors."""

        async def root_relay(repl):
            """Delegate selected text without copying the root's other context keys."""
            answer = await repl.callback("Find the value in selected evidence")
            return REPLResult(answer, None, False, 1)

        async def child_relay(repl):
            """Create a grandchild request containing only the child's explicit prompt."""
            answer = await repl.callback("Find grandchild value 917")
            return REPLResult(answer, None, False, 1)

        root = model(python("relay_root"), finish("917"), name="root-v1", usage=Usage(10, 2))
        child = model(
            python("inspect_prompt"),
            python("relay_child"),
            python("inspect_grandchild"),
            finish("917"),
            finish("917"),
            name="child-v2",
            usage=Usage(7, 1),
        )
        factory = REPLFactory([root_relay], ["task inspected", child_relay], ["917"])
        with patch("llgm.inference.rlm.DockerREPL", factory):
            runtime = RLMRuntime(
                root,
                child,
                budget=Budget(max_model_calls=10, max_context_tokens=32000),
            )
            actual = await runtime.answer("Find value", context={"unshared": "root-only-secret"})
        self.assertEqual(actual.status, "completed", actual.unresolved)
        self.assertEqual(actual.usage["model_calls"], 7)
        self.assertEqual(actual.usage["sidecar_calls"], 5)
        self.assertEqual(actual.usage["known_input_tokens"], 55)
        self.assertEqual(actual.usage["known_output_tokens"], 9)
        self.assertEqual(actual.usage["recursive_invocations"], 2)
        self.assertEqual(actual.usage["python_executions"], 4)
        self.assertEqual(
            factory.sessions[1].context, {"prompt": "Find the value in selected evidence"}
        )
        self.assertEqual(factory.sessions[2].context, {"prompt": "Find grandchild value 917"})
        self.assertNotIn("root-only-secret", str(child.requests))
        self.assertEqual(factory.closed, [2, 1, 0])
        calls = [event for event in actual.trace if event["kind"] == "model"]
        self.assertEqual([event["depth"] for event in calls], [0, 1, 1, 2, 2, 1, 0])
        self.assertEqual(
            [event["model"] for event in calls],
            ["root-v1", "child-v2", "child-v2", "child-v2", "child-v2", "child-v2", "root-v1"],
        )
        transfers = [event for event in actual.trace if "accounting_units" in event]
        self.assertEqual(
            actual.usage["evidence_accounting_units"],
            sum(event["accounting_units"] for event in transfers),
        )
        self.assertEqual([event["kind"] for event in transfers].count("delegate"), 2)
        self.assertEqual([event["kind"] for event in transfers].count("child_return"), 2)

    async def test_python_error_and_truncation_reach_next_model_turn(self):
        """Python failures are observations that permit bounded model correction."""

        async def error(repl):
            """Expose a recoverable Python exception with explicit output truncation."""
            return REPLResult("prefix", "NameError: missing", True, 0)

        root = model(python("bad_name"), python("corrected"), finish("fixed"))
        factory = REPLFactory([error, "recovered"])
        with patch("llgm.inference.rlm.DockerREPL", factory):
            actual = await RLMRuntime(root, capture_text=True).answer("Fix", context={})
        self.assertEqual(actual.status, "completed")
        observation = json.loads(root.requests[1].messages[-1].content)
        self.assertEqual(observation["error"], "NameError: missing")
        self.assertTrue(observation["stdout_truncated"])
        self.assertEqual(factory.sessions[0].code, ["bad_name", "corrected"])
        self.assertTrue(any(event.get("text") == "corrected" for event in actual.trace))

    async def test_invalid_operations_never_execute_python(self):
        """Malformed or ambiguous envelopes fail instead of guessing model intent."""
        for operation in (
            "not JSON",
            "[]",
            "```json\n{}\n```",
            {"op": "python", "code": ""},
            '{"op":"finish","answer":"first","answer":"second"}',
            '{"op":"python","code":"pass"}{"op":"finish","answer":"invented"}',
            {"op": "read", "reference": "x"},
            {"op": "finish", "answer": "", "extra": 1},
            {"op": "python", "code": "pass", "extra": 1},
            {"op": "finish", "answer": 1},
        ):
            with self.subTest(operation=operation):
                factory = REPLFactory()
                with patch("llgm.inference.rlm.DockerREPL", factory):
                    actual = await RLMRuntime(model(operation)).answer("Question", context={})
                self.assertEqual(actual.status, "failed")
                self.assertEqual(actual.answer, "")
                self.assertTrue(actual.unresolved)
                self.assertEqual(factory.sessions[0].code, [])
                self.assertTrue(factory.sessions[0].closed)
                self.assertTrue(any(event["kind"] == "model_output" for event in actual.trace))

    async def test_context_and_public_arguments_are_validated_before_work(self):
        """Invalid JSON, UTF-8, and limits cannot trigger container or model calls."""
        cyclic = {}
        cyclic["self"] = cyclic
        cases = [
            [],
            {1: "one"},
            {"bad": float("nan")},
            {"bad": "\ud800"},
            cyclic,
            {"tuple": (1, 2)},
        ]
        for context in cases:
            with self.subTest(context=type(context)):
                factory = REPLFactory()
                root = model(finish("unused"))
                with patch("llgm.inference.rlm.DockerREPL", factory):
                    with self.assertRaises(ConfigurationError):
                        await RLMRuntime(root).answer("Question", context=context)
                self.assertFalse(factory.sessions)
                self.assertFalse(root.requests)
        for kwargs in (
            {"max_depth": -1},
            {"max_steps": 0},
            {"max_executions": True},
            {"capture_text": 1},
            {"budget": "x"},
            {"repl_config": {}},
            {"token_counter": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                RLMRuntime(model(), **kwargs)
        with self.assertRaises(SchemaError):
            await RLMRuntime(model()).answer("\ud800", context={})
        with self.assertRaises(ConfigurationError):
            await RLMRuntime(model(), repl_config=DockerREPLConfig(max_context_bytes=20)).answer(
                "Question",
                context={"text": "x" * 100},
            )

    async def test_oversized_code_answer_and_model_envelope_are_rejected(self):
        """UTF-8 limits reject oversized outputs before execution or answer publication."""
        config = DockerREPLConfig(max_code_bytes=5, max_response_bytes=5)
        for operation in (python("é" * 3), finish("é" * 3), " " * 5000):
            factory = REPLFactory()
            with patch("llgm.inference.rlm.DockerREPL", factory):
                actual = await RLMRuntime(model(operation), repl_config=config).answer(
                    "Question",
                    context={},
                )
            self.assertEqual(actual.status, "failed")
            self.assertEqual(factory.sessions[0].code, [])

    async def test_model_context_overflow_and_invalid_counter_prevent_startup(self):
        """Initial context admission and accounting validation precede Docker startup."""
        for kwargs in (
            {"budget": Budget(max_context_tokens=20)},
            {"token_counter": lambda text: -1},
            {"token_counter": lambda text: True},
        ):
            factory = REPLFactory()
            root = model(finish("unused"))
            with patch("llgm.inference.rlm.DockerREPL", factory):
                actual = await RLMRuntime(root, **kwargs).answer("Question", context={})
            self.assertNotEqual(actual.status, "completed")
            self.assertFalse(root.requests)
            self.assertFalse(factory.sessions)

    async def test_depth_and_global_call_budget_prevent_child_startup(self):
        """A forbidden recursive call consumes no child container or provider request."""

        async def relay(repl):
            """Attempt a callback whose admission should fail at the shared boundary."""
            await repl.callback("child prompt")
            raise AssertionError("unreachable")

        for kwargs in ({"max_depth": 0}, {"budget": Budget(max_model_calls=2)}):
            factory = REPLFactory([relay])
            child = model(finish("unused"))
            with patch("llgm.inference.rlm.DockerREPL", factory):
                actual = await RLMRuntime(model(python("relay")), child, **kwargs).answer(
                    "Question",
                    context={},
                )
            self.assertEqual(actual.status, "budget_exhausted")
            self.assertEqual(len(factory.sessions), 1)
            self.assertFalse(child.requests)

    async def test_evidence_transfer_limit_applies_before_child_or_continuation(self):
        """Large selected prompts and printed observations cannot bypass exposure limits."""

        async def relay(repl):
            """Attempt to send selected evidence larger than the shared allowance."""
            await repl.callback("x" * 100)
            raise AssertionError("unreachable")

        for plan in ([relay], ["x" * 100]):
            factory = REPLFactory(plan)
            root = model(python("inspect"), finish("unused"))
            with patch("llgm.inference.rlm.DockerREPL", factory):
                actual = await RLMRuntime(root, budget=Budget(max_evidence_tokens=50)).answer(
                    "Question",
                    context={},
                )
            self.assertEqual(actual.status, "budget_exhausted")
            self.assertEqual(len(root.requests), 1)
            self.assertEqual(len(factory.sessions), 1)
            self.assertEqual(actual.usage["evidence_accounting_units"], 0)

    async def test_sidecar_limit_spans_sibling_callbacks_before_container_start(self):
        """A completed child cannot reset the shared sidecar allowance for its sibling."""

        async def siblings(repl):
            """Attempt two sequential children under a one-child-model-call allowance."""
            await repl.callback("first child")
            await repl.callback("second child")
            raise AssertionError("unreachable")

        factory = REPLFactory([siblings], [])
        child = model(finish("first result"), finish("unused"))
        with patch("llgm.inference.rlm.DockerREPL", factory):
            actual = await RLMRuntime(
                model(python("two_callbacks")),
                child,
                budget=Budget(max_sidecar_calls=1),
            ).answer("Question", context={})
        self.assertEqual(actual.status, "budget_exhausted")
        self.assertEqual(len(factory.sessions), 2)
        self.assertEqual(len(child.requests), 1)
        self.assertEqual(actual.usage["sidecar_calls"], 1)
        self.assertEqual(factory.closed, [1, 0])

    async def test_returned_answer_steps_and_python_work_have_separate_limits(self):
        """Final answer, per-frame turn, and shared execution caps stop excess work."""
        cases = [
            ([finish("too long")], [], {"budget": Budget(max_bundle_tokens=3)}, 0),
            ([python("first"), finish("unused")], ["ok"], {"max_steps": 1}, 1),
            ([python("first"), python("second")], ["ok"], {"max_executions": 1}, 1),
        ]
        for operations, plan, kwargs, expected_executions in cases:
            factory = REPLFactory(plan)
            with patch("llgm.inference.rlm.DockerREPL", factory):
                actual = await RLMRuntime(model(*operations), **kwargs).answer(
                    "Question", context={}
                )
            self.assertEqual(actual.status, "budget_exhausted")
            self.assertEqual(actual.usage["python_executions"], expected_executions)
            self.assertTrue(factory.sessions[0].closed)

    async def test_active_identical_recursive_request_is_rejected(self):
        """A repeated active child prompt stops recursion even when depth remains."""

        async def repeat(repl):
            """Return the same delegated request at every level."""
            return REPLResult(await repl.callback("same prompt"), None, False, 1)

        factory = REPLFactory([repeat], [repeat])
        with patch("llgm.inference.rlm.DockerREPL", factory):
            actual = await RLMRuntime(
                model(python("relay")),
                model(python("relay")),
                max_depth=5,
            ).answer("Question", context={})
        self.assertEqual(actual.status, "budget_exhausted")
        self.assertIn("Repeated active", actual.unresolved[0])
        self.assertEqual(len(factory.sessions), 2)
        self.assertEqual(factory.closed, [1, 0])

    async def test_provider_failure_keeps_unknown_usage_without_secret_text(self):
        """A failed provider invocation is counted and its arbitrary exception is redacted."""

        async def fail(request):
            """Emulate a host transport failure containing data unsuitable for artifacts."""
            raise RuntimeError("provider-secret-content")

        factory = REPLFactory()
        with patch("llgm.inference.rlm.DockerREPL", factory):
            actual = await RLMRuntime(CallableModelClient(fail)).answer("Question", context={})
        self.assertEqual(actual.status, "failed")
        self.assertEqual(actual.usage["model_calls"], 1)
        self.assertEqual(actual.usage["unknown_usage_calls"], 1)
        self.assertNotIn("provider-secret", json.dumps(actual.trace) + str(actual.unresolved))
        self.assertTrue(factory.sessions[0].closed)

    async def test_refusal_preserves_billed_usage_but_never_executes_text(self):
        """Refused model output remains a failure even when it resembles valid Python JSON."""
        root = ScriptedModelClient(
            [
                ModelResponse(
                    json.dumps(python("must_not_execute")),
                    Usage(4, 2),
                    status="refused",
                    refusal="no",
                )
            ]
        )
        factory = REPLFactory()
        with patch("llgm.inference.rlm.DockerREPL", factory):
            actual = await RLMRuntime(root).answer("Question", context={})
        self.assertEqual(actual.status, "failed")
        self.assertEqual(actual.usage["known_input_tokens"], 4)
        self.assertEqual(factory.sessions[0].code, [])

    async def test_docker_failure_precedes_model_call_and_has_no_host_fallback(self):
        """Missing isolation infrastructure fails explicitly before provider work."""
        factory = REPLFactory(start_error=REPLError("Docker unavailable"))
        root = model(finish("unused"))
        with patch("llgm.inference.rlm.DockerREPL", factory):
            actual = await RLMRuntime(root).answer("Question", context={})
        self.assertEqual(actual.status, "failed")
        self.assertFalse(root.requests)
        self.assertEqual(factory.closed, [0])

    async def test_interpreter_and_whole_run_timeouts_are_explicit_budget_failures(self):
        """Execution and shared wall-clock deadlines terminate work and clean up."""

        async def wait(repl):
            """Keep the interpreter pending until the controller cancels the run."""
            await asyncio.Event().wait()

        for plan, budget in (
            ([REPLTimeoutError("code deadline")], Budget()),
            ([wait], replace(Budget(), timeout_seconds=0.02)),
        ):
            factory = REPLFactory(plan)
            with patch("llgm.inference.rlm.DockerREPL", factory):
                actual = await RLMRuntime(model(python("long_work")), budget=budget).answer(
                    "Question",
                    context={},
                )
            self.assertEqual(actual.status, "budget_exhausted")
            self.assertTrue(factory.sessions[0].closed)

    async def test_cancellation_closes_child_then_parent_and_retains_usage(self):
        """Cancellation interrupts the child model and unwinds every active interpreter."""
        child_started = asyncio.Event()

        async def relay(repl):
            """Hold the parent interpreter while a recursive child is active."""
            return REPLResult(await repl.callback("child task"), None, False, 1)

        async def pending(request):
            """Signal that the child was invoked before awaiting cancellation."""
            child_started.set()
            await asyncio.Event().wait()

        factory = REPLFactory([relay], [])
        with patch("llgm.inference.rlm.DockerREPL", factory):
            runtime = RLMRuntime(model(python("relay")), CallableModelClient(pending))
            task = asyncio.create_task(runtime.answer("Question", context={}))
            await child_started.wait()
            with self.assertRaises(ConfigurationError):
                await runtime.answer("Concurrent", context={})
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(factory.closed, [1, 0])
        self.assertEqual(runtime.last_usage["model_calls"], 2)
        self.assertTrue(any(event.get("status") == "cancelled" for event in runtime.last_trace))
        self.assertFalse(runtime._active)

    async def test_cleanup_failure_cannot_mask_cancellation(self):
        """Failed Docker cleanup remains visible while the original cancellation propagates."""
        started = asyncio.Event()

        async def pending(request):
            """Wait until the test cancels a model request in an active container."""
            started.set()
            await asyncio.Event().wait()

        factory = REPLFactory(close_error=REPLError("unconfirmed cleanup"))
        with patch("llgm.inference.rlm.DockerREPL", factory):
            runtime = RLMRuntime(CallableModelClient(pending))
            task = asyncio.create_task(runtime.answer("Question", context={}))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(any(event["kind"] == "cleanup_failed" for event in runtime.last_trace))
        self.assertFalse(any(event["kind"] == "repl_closed" for event in runtime.last_trace))
        self.assertFalse(runtime._active)

    async def test_successful_answer_requires_confirmed_cleanup(self):
        """A cleanup failure cannot publish a completed result with an orphaned session."""
        factory = REPLFactory(close_error=REPLError("unconfirmed cleanup"))
        with patch("llgm.inference.rlm.DockerREPL", factory):
            actual = await RLMRuntime(model(finish("answer"))).answer("Question", context={})
        self.assertEqual(actual.status, "failed")
        self.assertEqual(actual.answer, "")
        self.assertTrue(any(event["kind"] == "cleanup_failed" for event in actual.trace))

    async def test_context_copy_and_separate_answers_do_not_rewrite_prior_state(self):
        """Caller mutation and subsequent runs cannot alter an earlier context or trace."""
        original = {"items": ["initial"]}

        async def mutate(request):
            """Mutate caller state after the runtime has already snapped its evidence."""
            original["items"].append("later")
            return ModelResponse(json.dumps(finish("done")))

        factory = REPLFactory([], [])
        with patch("llgm.inference.rlm.DockerREPL", factory):
            runtime = RLMRuntime(CallableModelClient(mutate))
            first = await runtime.answer("First", context=original)
            trace_before = copy.deepcopy(first.trace)
            second = await runtime.answer("Second", context=original)
        self.assertEqual(factory.sessions[0].context, {"items": ["initial"]})
        self.assertEqual(factory.sessions[1].context, {"items": ["initial", "later"]})
        self.assertEqual(first.trace, trace_before)
        self.assertIsNot(first.trace, second.trace)
        self.assertEqual(first.usage["model_calls"], 1)


class RLMCleanupLifetimeTests(unittest.IsolatedAsyncioTestCase):
    """Cancellation and deadlines cannot detach container cleanup from the answer lifetime."""

    async def test_child_cleanup_failure_is_fatal_across_callback_error_normalization(self):
        """A recoverable guest callback error cannot conceal an unremoved child container."""
        replies = []

        async def relay(repl):
            """Exercise the real callback error boundary without starting a container."""
            transport = DockerREPL({}, config=repl.config, llm_query=repl.callback)
            transport._send = AsyncMock()
            for query_id in (1, 2):
                await transport._query(
                    {
                        "type": "query",
                        "execution_id": 1,
                        "query_id": query_id,
                        "prompt": "Inspect selected evidence",
                    },
                    1,
                    query_id,
                )
                replies.append(transport._send.call_args.args[0])
            return REPLResult("", replies[-1]["error"], False, 2)

        async def close(repl):
            """Fail child cleanup while allowing its parent's cleanup to complete."""
            repl.owner.closed.append(repl.index)
            if repl.index == 1:
                raise REPLError("Child container removal could not be confirmed")
            repl.closed = True

        factory = REPLFactory([relay])
        root = model(python("delegate"), finish("must not complete"))
        child = model(finish("child answer"))
        with (
            patch("llgm.inference.rlm.DockerREPL", factory),
            patch.object(FakeREPL, "aclose", close),
        ):
            runtime = RLMRuntime(root, child)
            result = await runtime.answer("Question", context={})

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.answer, "")
        self.assertIn("cleanup failed", result.unresolved[0])
        self.assertTrue(all(reply["error"] is not None for reply in replies))
        self.assertEqual(len(factory.sessions), 2)
        self.assertEqual(len(root.requests), 1)
        self.assertEqual(len(child.requests), 1)
        self.assertEqual(result.usage["model_calls"], 2)
        self.assertEqual(factory.closed, [1, 0])
        self.assertTrue(factory.sessions[0].closed)
        self.assertFalse(factory.sessions[1].closed)
        self.assertFalse(runtime._active)
        failed = [event for event in result.trace if event["kind"] == "cleanup_failed"]
        self.assertEqual([event["invocation_id"] for event in failed], ["r2"])

    async def test_repeated_cancellation_during_cleanup_waits_before_reuse(self):
        """The answer remains active until delayed cleanup completes despite repeated cancellation."""
        started, release = asyncio.Event(), asyncio.Event()
        original = FakeREPL.aclose

        async def delayed(repl):
            """Hold cleanup so cancellation can arrive after a successful model answer."""
            started.set()
            await release.wait()
            await original(repl)

        factory = REPLFactory()
        with (
            patch("llgm.inference.rlm.DockerREPL", factory),
            patch.object(FakeREPL, "aclose", delayed),
        ):
            runtime = RLMRuntime(model(finish("answer")))
            task = asyncio.create_task(runtime.answer("Question", context={}))
            await started.wait()
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertTrue(runtime._active)
            self.assertFalse(factory.sessions[0].closed)
            with self.assertRaises(ConfigurationError):
                await runtime.answer("Premature reuse", context={})
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(factory.sessions[0].closed)
        self.assertFalse(runtime._active)

    async def test_global_timeout_during_cleanup_is_drained(self):
        """A run deadline waits for cleanup and reports exhaustion without leaking a session."""
        original = FakeREPL.aclose

        async def delayed(repl):
            """Let the run deadline expire while the owned close operation remains active."""
            await asyncio.sleep(0.04)
            await original(repl)

        factory = REPLFactory()
        with (
            patch("llgm.inference.rlm.DockerREPL", factory),
            patch.object(FakeREPL, "aclose", delayed),
        ):
            runtime = RLMRuntime(
                model(finish("answer")), budget=replace(Budget(), timeout_seconds=0.01)
            )
            result = await runtime.answer("Question", context={})
        self.assertEqual(result.status, "budget_exhausted")
        self.assertTrue(factory.sessions[0].closed)
        self.assertGreaterEqual(result.usage["elapsed_seconds"], 0.04)


class RLMStructuredOutputTests(unittest.IsolatedAsyncioTestCase):
    """Native output schemas enforce one executable operation without text repair."""

    async def test_native_schema_wraps_and_validates_every_controller_turn(self):
        """A capable model receives the operation union and returns through its strict wrapper."""
        from llgm.models import ModelCapabilities

        responses = iter([{"operation": python("inspect")}, {"operation": finish("found")}])
        seen = []

        async def complete(request):
            """Record native schema requests and provide schema-shaped decisions."""
            seen.append(request)
            return ModelResponse(json.dumps(next(responses)))

        client = CallableModelClient(
            complete, capabilities=ModelCapabilities(structured_output=True)
        )
        with patch("llgm.inference.rlm.DockerREPL", REPLFactory(["evidence"])):
            result = await RLMRuntime(client).answer("Question", context={"source": "evidence"})
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "found")
        self.assertTrue(all(request.output_schema["required"] == ["operation"] for request in seen))
        self.assertTrue(all(e["structured_output"] for e in result.trace if e["kind"] == "model"))

    async def test_native_schema_adapter_cannot_bypass_runtime_envelope_validation(self):
        """Claiming schema support cannot make a malformed unwrapped operation executable."""
        from llgm.models import ModelCapabilities

        async def complete(request):
            """Simulate a transport violating its advertised native schema contract."""
            return ModelResponse(json.dumps(python("must not execute")))

        client = CallableModelClient(
            complete, capabilities=ModelCapabilities(structured_output=True)
        )
        factory = REPLFactory()
        with patch("llgm.inference.rlm.DockerREPL", factory):
            result = await RLMRuntime(client).answer("Question", context={})
        self.assertEqual(result.status, "failed")
        self.assertFalse(factory.sessions[0].code)

    async def test_oversized_native_schema_prevents_container_startup(self):
        """Schema conditioning must fit the frame allowance before Docker is started."""
        from llgm.models import ModelCapabilities

        client = model(finish("unused"))
        client.capabilities = ModelCapabilities(structured_output=True)
        factory = REPLFactory()
        schema = {"type": "object", "description": "x" * 16000}
        with (
            patch("llgm.inference.rlm.DockerREPL", factory),
            patch("llgm.inference.rlm._SCHEMA", schema),
        ):
            result = await RLMRuntime(client).answer("Question", context={})
        self.assertEqual(result.status, "budget_exhausted")
        self.assertFalse(factory.sessions)
        self.assertFalse(client.requests)
        self.assertEqual(result.usage["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
