"""Opt-in real DSPy Deno/Pyodide tests with fixed model outputs and no API calls."""

import asyncio
import json
import os

import pytest

from llgm.inference.repl import DSPySession, REPLTimeoutError, SandboxConfig

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]


@pytest.fixture(autouse=True)
def sandbox_opt_in():
    """Require permission to start Deno and populate its runtime asset cache."""
    enabled = os.environ.get("LLGM_TEST_SANDBOX", "0")
    if enabled == "0":
        pytest.skip("Set LLGM_TEST_SANDBOX=1 for the real DSPy sandbox")
    if enabled != "1":
        pytest.fail("LLGM_TEST_SANDBOX must be exactly 0 or 1")


def test_real_python_persistence_errors_tools_and_isolation(tmp_path, monkeypatch):
    """Actual WASM Python persists state and invokes host tools without gaining host permissions."""
    sentinel = tmp_path / "private.txt"
    sentinel.write_text("host-only")
    monkeypatch.setenv("LLGM_SANDBOX_SENTINEL", "host-only")

    async def scenario():
        """Use two real executions, recovering from an exception after a host callback."""
        session = DSPySession({"path": str(sentinel)})
        calls = []

        async def host(value):
            """Return an explicitly authorized value to generated Python."""
            calls.append(value)
            return value * 2

        def twice(value: int) -> dict:
            """Double a value through the async host bridge."""
            return {"value": session.bridge(host, value), "next_offset": None}

        codes = iter(
            [
                "payload = twice(21)\nassert payload['next_offset'] is None\nresult = payload['value']\nraise ValueError('recoverable sentinel')",
                """import json, os, pathlib, js
assert result == 42
assert not pathlib.Path(context['path']).exists()
assert os.environ.get('LLGM_SANDBOX_SENTINEL') is None
permission = js.Deno.permissions.querySync(js.JSON.parse(json.dumps({'name': 'net', 'host': 'example.com:443'})))
assert str(permission.state) != 'granted'
try:
    js.Deno.readTextFileSync(context['path'])
except Exception:
    pass
else:
    raise AssertionError('Host path was accessible')
SUBMIT(answer=str(result))""",
            ]
        )
        requests = []

        async def generate(messages, *, schema, phase):
            """Provide fixed generated code while capturing the actual DSPy error observation."""
            requests.append(messages)
            return json.dumps({"reasoning": "Check sandbox behavior", "code": next(codes)})

        result = await session.run(
            generate=generate,
            instructions="Check supplied code",
            max_steps=3,
            max_llm_calls=2,
            tools=[twice],
            outputs={"answer": str},
        )
        assert result == {"answer": "42"} and calls == [21]
        assert "recoverable sentinel" in requests[1][1].content
        assert session.future.done() and session.interpreter.deno_process is None

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
def test_real_infinite_python_is_aborted_and_reaped(stop):
    """Deadlines and cancellation terminate a real busy sandbox before returning control."""

    async def scenario():
        """Stop infinite generated code while retaining its actual subprocess identity."""
        session = DSPySession(
            {}, config=SandboxConfig(execution_timeout_seconds=0.15 if stop == "timeout" else 30)
        )
        entered = asyncio.Event()

        async def generate(messages, *, schema, phase):
            """Produce nonterminating Python without making any model call."""
            entered.set()
            return json.dumps({"reasoning": "Test interruption", "code": "while True: pass"})

        task = asyncio.create_task(
            session.run(
                generate=generate,
                instructions="Test interruption",
                max_steps=1,
                max_llm_calls=1,
                outputs={"answer": str},
            )
        )
        await asyncio.wait_for(entered.wait(), 30)
        process = session.interpreter.deno_process
        if stop == "cancel":
            await asyncio.sleep(0.05)
            task.cancel()
        with pytest.raises(REPLTimeoutError if stop == "timeout" else asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert process.poll() is not None
        assert session.future.done() and session.interpreter.deno_process is None

    asyncio.run(scenario())
