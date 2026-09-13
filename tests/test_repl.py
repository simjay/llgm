"""Contracts around DSPy's actual loop with a finite CodeInterpreter boundary."""

import asyncio
import json
from collections import deque

import pytest

from llgm.core.errors import BudgetExceeded, ConfigurationError, SchemaError
from llgm.inference.repl import DSPySession, SandboxConfig


class Interpreter:
    """Return declared interpreter outcomes without executing generated host Python."""

    def __init__(self, outcomes):
        """Keep local bindings, output metadata and observable lifecycle state."""
        self.outcomes = deque(outcomes)
        self.tools, self.output_fields = {}, None
        self.closed, self.started = False, False
        self.variables, self.codes = [], []

    def start(self):
        """Record admission without starting an external process."""
        self.started = True

    def execute(self, code, variables=None):
        """Return or raise one declared boundary outcome for each DSPy action."""
        self.codes.append(code)
        self.variables.append(variables)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome(self) if callable(outcome) else outcome

    def shutdown(self):
        """Record resource closure on every completion path."""
        self.closed = True


def final(answer="found", **fields):
    """Construct DSPy's public typed interpreter result without a protocol imitation."""
    from dspy.primitives.code_interpreter import FinalOutput

    return FinalOutput({"answer": answer, **fields})


def generator(responses, calls):
    """Create an async model boundary with finite outputs and actual DSPy prompt capture."""
    remaining = iter(responses)

    async def generate(messages, *, schema, phase):
        """Return a fixture while exposing DSPy's signature and generation phase."""
        calls.append((messages, schema, phase))
        return json.dumps(next(remaining))

    return generate


async def run(session, generate, **options):
    """Use a small typed signature while retaining DSPy's real orchestration."""
    return await session.run(
        generate=generate,
        instructions="Find the answer in context.",
        max_steps=options.pop("max_steps", 3),
        max_llm_calls=2,
        outputs={"answer": str},
        **options,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_code_bytes", 0),
        ("max_response_bytes", True),
        ("execution_timeout_seconds", float("nan")),
        ("startup_timeout_seconds", -1),
    ],
)
def test_config_validation(field, value):
    """Invalid limits fail before DSPy can open an interpreter."""
    with pytest.raises(ConfigurationError):
        SandboxConfig(**{field: value})


def test_context_is_snapshotted_and_bounded_before_start():
    """Caller mutations cannot change the JSON admitted to an RLM session."""
    context = {"rows": ["original"]}
    session = DSPySession(context)
    context["rows"].append("later")
    assert session.context == {"rows": ["original"]}
    with pytest.raises(BudgetExceeded):
        DSPySession(context, config=SandboxConfig(max_context_bytes=4))
    with pytest.raises(SchemaError):
        DSPySession({"value": float("nan")})


def test_dspy_recovers_execution_errors_and_keeps_external_context():
    """DSPy supplies a real error observation and reuses the same isolated interpreter."""

    async def scenario():
        """Correct one failed Python action before a successful typed submission."""
        from dspy.primitives.code_interpreter import CodeExecutionError

        interpreter = Interpreter([CodeExecutionError("bad reference"), final()])
        calls = []
        result = await run(
            DSPySession({"text": "external"}, interpreter_factory=lambda: interpreter),
            generator(
                [
                    {"reasoning": "inspect", "code": "read"},
                    {"reasoning": "correct", "code": "submit"},
                ],
                calls,
            ),
        )
        assert result == {"answer": "found"}
        assert "bad reference" in calls[1][0][1].content
        assert len(interpreter.variables) == 2 and interpreter.variables[0] == {
            "context": {"text": "external"}
        }
        assert interpreter.closed
        assert all(set(call[1]["properties"]) == {"reasoning", "code"} for call in calls)

    asyncio.run(scenario())


def test_dspy_extraction_is_a_separate_admitted_call():
    """Reaching max_iters invokes DSPy's extractor through the same host model boundary."""

    async def scenario():
        """Observe an action followed by extraction, with no hidden fallback provider."""
        interpreter, calls = Interpreter(["observed value"]), []
        result = await run(
            DSPySession({}, interpreter_factory=lambda: interpreter),
            generator([{"reasoning": "inspect", "code": "print"}, {"answer": "extracted"}], calls),
            max_steps=1,
        )
        assert result == {"answer": "extracted"}
        assert [call[2] for call in calls] == ["final_action", "extract"]
        assert interpreter.closed

    asyncio.run(scenario())


def test_validation_errors_reach_dspy_before_final_acceptance():
    """LLGM evidence rejection becomes a DSPy correction observation without another custom loop."""

    async def scenario():
        """Reject a fabricated answer, then accept a corrected typed submission."""
        interpreter, calls = Interpreter([final("invented"), final("supported")]), []

        async def validate(output):
            """Apply a host-owned evidence constraint to every submission."""
            if output["answer"] != "supported":
                raise SchemaError("Citation was not accessed")

        result = await run(
            DSPySession({}, interpreter_factory=lambda: interpreter),
            generator(
                [{"reasoning": "guess", "code": "bad"}, {"reasoning": "repair", "code": "good"}],
                calls,
            ),
            validate=validate,
        )
        assert result["answer"] == "supported"
        assert "Citation was not accessed" in calls[1][0][1].content
        assert interpreter.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["code", "output", "return", "invalid_json"])
def test_failed_admission_closes_the_interpreter(boundary):
    """Oversized or malformed results cannot reach another model or a successful caller."""

    async def scenario():
        """Reject each transfer boundary without retrying a provider call."""
        interpreter = Interpreter(["x" * 100 if boundary == "output" else final("x" * 100)])
        config = SandboxConfig(
            max_code_bytes=4 if boundary == "code" else 100,
            max_output_bytes=4 if boundary == "output" else 100,
            max_response_bytes=4 if boundary == "return" else 200,
        )
        calls = []
        generate = generator([{"reasoning": "inspect", "code": "12345"}], calls)
        if boundary == "invalid_json":

            async def generate(messages, *, schema, phase):
                """Supply a wrong DSPy signature without a hidden format-retry call."""
                calls.append(phase)
                return '{"unexpected": "field"}'

        with pytest.raises((BudgetExceeded, SchemaError)):
            await run(
                DSPySession({}, config=config, interpreter_factory=lambda: interpreter), generate
            )
        assert len(calls) == 1 and interpreter.closed
        if boundary in {"code", "invalid_json"}:
            assert not interpreter.codes

    asyncio.run(scenario())


def test_cancellation_drains_host_callback_cleanup():
    """Returning cancellation waits for both the worker and delayed async callback cleanup."""

    async def scenario():
        """Cancel during a model call and again while its finally block owns cleanup."""
        interpreter = Interpreter([])
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        completed = False

        async def generate(messages, *, schema, phase):
            """Model an async provider that must drain its cancellation cleanup."""
            nonlocal completed
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                completed = True

        session = DSPySession({}, interpreter_factory=lambda: interpreter)
        task = asyncio.create_task(run(session, generate))
        await entered.wait()
        task.cancel()
        await cleaning.wait()
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert completed and interpreter.closed and not session.host_tasks

    asyncio.run(scenario())
