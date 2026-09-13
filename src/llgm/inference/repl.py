"""DSPy RLM execution with an async host bridge and a Deno/Pyodide sandbox.

DSPy owns prompting, Python state, tool transport and the action loop. LLGM owns
admission, deadlines and cancellation. No host paths, environment or network
permissions are granted to the default interpreter. Imports are lazy so storage
and retrieval do not require DSPy.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import math
import threading
import time
from dataclasses import dataclass

from llgm.core.errors import BudgetExceeded, ConfigurationError, LLGMError, SchemaError
from llgm.inference._json import validate_json_types
from llgm.models import Message


class REPLError(LLGMError):
    """DSPy's interpreter could not start, execute or shut down."""


class REPLTimeoutError(REPLError, TimeoutError):
    """An interpreter deadline expired and its process was aborted."""


@dataclass(frozen=True)
class SandboxConfig:
    """Application admission limits around DSPy's default WASM interpreter.

    Byte limits bound inputs and admitted outputs, not process memory. The
    sandbox does not impose operating-system CPU, memory or process quotas.
    """

    startup_timeout_seconds: float = 30.0
    execution_timeout_seconds: float = 30.0
    max_context_bytes: int = 8 * 1024 * 1024
    max_code_bytes: int = 64 * 1024
    max_output_bytes: int = 32 * 1024
    max_query_bytes: int = 64 * 1024
    max_response_bytes: int = 64 * 1024

    def __post_init__(self):
        """Reject nonpositive, nonfinite and incorrectly typed limits."""
        for name, value in vars(self).items():
            if name.endswith("seconds"):
                valid = type(value) in {int, float} and math.isfinite(value) and value > 0
            else:
                valid = type(value) is int and value > 0
            if not valid:
                raise ConfigurationError(f"{name} must be finite and positive")


def _bounded(value, limit, name):
    """Validate JSON transfer size before admitting it to another boundary."""
    try:
        text = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
        if len(text.encode("utf-8")) > limit:
            raise BudgetExceeded(f"{name} exceeds its byte allowance")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise SchemaError(f"{name} must be finite UTF-8 JSON") from error
    return value


# DSPy 3.3.1 converts JSON null to Pyodide's JsNull in tool results. Normalize
# inside WASM so the documented Python tool contract retains ordinary None.
# This setup never runs on the host and does not implement an execution loop.
_SANDBOX_SETUP = """
'''Normalize JSON metadata returned through the DSPy Pyodide tool bridge.'''
from functools import wraps as _llgm_wraps
from pyodide.ffi import JsNull as _llgm_JsNull

def _llgm_plain(value):
    '''Preserve normal Python JSON values across DSPy's JavaScript tool bridge.'''
    if isinstance(value, _llgm_JsNull):
        return None
    if isinstance(value, dict):
        return {key: _llgm_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_llgm_plain(item) for item in value]
    return value

def _llgm_wrap(function):
    '''Normalize returned metadata without changing a tool's arguments or authority.'''
    @_llgm_wraps(function)
    def invoke(*args, **kwargs):
        '''Return Python-native data from an already registered DSPy tool.'''
        return _llgm_plain(function(*args, **kwargs))
    return invoke

for _llgm_name in _llgm_tool_names:
    globals()[_llgm_name] = _llgm_wrap(globals()[_llgm_name])
"""


class DSPySession:
    """Own one DSPy RLM worker and marshal all authority to its async caller.

    A dedicated thread prevents recursive readers from exhausting an executor
    pool while their parents wait. Each interpreter stays on its owning thread.
    A caller-supplied factory must return a synchronous DSPy CodeInterpreter.
    Custom interpreters own their isolation and must support bounded shutdown.
    """

    def __init__(self, context, *, config=None, interpreter_factory=None):
        """Snapshot explicit JSON inputs without importing DSPy or starting a process."""
        self.config = config or SandboxConfig()
        if not isinstance(context, dict):
            raise ConfigurationError("RLM context must be a JSON object")
        _bounded(context, self.config.max_context_bytes, "RLM context")
        validate_json_types(context)
        self.context = json.loads(json.dumps(context, allow_nan=False))
        self.factory = interpreter_factory
        self.interpreter = None
        self.future = None
        self.pending = set()
        self.host_tasks = set()
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.deadline = None
        self.timeout_error = None

    def bridge(self, function, *args, **kwargs):
        """Run one async host operation on the owning event loop, never in the sandbox."""
        with self.lock:
            if self.stopped.is_set():
                raise REPLError("RLM session stopped")
            future = asyncio.run_coroutine_threadsafe(
                self._host_call(function, args, kwargs), self.loop
            )
            self.pending.add(future)
        try:
            return future.result()
        finally:
            with self.lock:
                self.pending.discard(future)

    async def _host_call(self, function, args, kwargs):
        """Retain async callback ownership until nested cancellation cleanup finishes."""
        task = asyncio.current_task()
        self.host_tasks.add(task)
        try:
            return await function(*args, **kwargs)
        finally:
            self.host_tasks.discard(task)

    def abort(self):
        """Cancel host work and kill a running Deno process to unblock synchronous I/O."""
        with self.lock:
            self.stopped.set()
            for future in tuple(self.pending):
                future.cancel()
        process = getattr(self.interpreter, "deno_process", None)
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    async def run(
        self,
        *,
        generate,
        instructions,
        max_steps,
        max_llm_calls,
        tools=(),
        on_execute=None,
        validate=None,
        outputs=None,
    ):
        """Run DSPy's loop and drain the worker and host callbacks before returning or raising."""
        if self.future is not None:
            raise ConfigurationError("DSPy sessions are single-use")
        self.loop = asyncio.get_running_loop()
        self.future = concurrent.futures.Future()

        def worker():
            """Keep DSPy context and interpreter operations confined to this thread."""
            try:
                result = self._run(
                    generate,
                    instructions,
                    max_steps,
                    max_llm_calls,
                    tools,
                    on_execute,
                    validate,
                    outputs,
                )
            except BaseException as error:
                self.future.set_exception(error)
            else:
                self.future.set_result(result)

        context = contextvars.copy_context()
        threading.Thread(
            target=context.run, args=(worker,), name="llgm-dspy-reader", daemon=True
        ).start()
        try:
            while not self.future.done():
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    self.timeout_error = REPLTimeoutError("DSPy interpreter deadline exhausted")
                    self.abort()
                await asyncio.sleep(0.01)
            if self.timeout_error is not None:
                raise self.timeout_error
            return self.future.result()
        finally:
            await self.aclose()

    async def aclose(self):
        """Wait for worker cleanup even when cancellation arrives repeatedly."""
        cancellation = None
        while (self.future is not None and not self.future.done()) or self.host_tasks:
            self.abort()
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError as error:
                cancellation = error
        if cancellation is not None:
            raise cancellation

    def _run(
        self,
        generate,
        instructions,
        max_steps,
        max_llm_calls,
        tools,
        on_execute,
        validate,
        outputs,
    ):
        """Adapt LLGM model calls and evidence tools to DSPy's public RLM interface."""
        try:
            import dspy
            from dspy.primitives.code_interpreter import (
                CodeExecutionError,
                CodeInterpreterError,
                FinalOutput,
            )
            from pydantic import create_model
        except ImportError as error:
            raise ConfigurationError("Install llgm[rlm] to use DSPy node readers") from error
        session = self

        class HostLM(dspy.BaseLM):
            """Send DSPy subqueries through LLGM's shared model admission path."""

            def __init__(self):
                """Provide the DSPy model contract without another provider or cache."""
                super().__init__("llgm/reader", cache=False)

            def __call__(self, prompt=None, messages=None, **kwargs):
                """Charge each plain DSPy subquery as a reader call."""
                _bounded(prompt or messages, session.config.max_query_bytes, "LLM subquery")
                values = (
                    [Message(**item) for item in messages]
                    if messages
                    else [Message("user", prompt)]
                )
                raw = session.bridge(generate, values, schema=None, phase="subquery")
                return [_bounded(raw, session.config.max_response_bytes, "LLM subquery response")]

        class HostAdapter(dspy.JSONAdapter):
            """Use DSPy's signature formatting and parsing with one admitted provider call."""

            def __call__(self, lm, lm_kwargs, signature, demos, inputs):
                """Avoid implicit provider fallback calls and global DSPy configuration."""
                messages = [Message(**item) for item in self.format(signature, demos, inputs)]
                schema = create_model(
                    "ReaderOutput",
                    **{
                        name: (field.annotation, ...)
                        for name, field in signature.output_fields.items()
                    },
                ).model_json_schema()
                schema["additionalProperties"] = False
                phase = "action" if "code" in signature.output_fields else "extract"
                if phase == "action" and inputs.get("iteration") == f"{max_steps}/{max_steps}":
                    phase = "final_action"
                raw = session.bridge(generate, messages, schema=schema, phase=phase)
                try:
                    return [self.parse(signature, raw)]
                except dspy.utils.exceptions.AdapterParseError as error:
                    raise SchemaError("Invalid DSPy reader output") from error

        class Interpreter:
            """Enforce LLGM admission around an upstream synchronous interpreter."""

            execution_instructions = dspy.PythonInterpreter.execution_instructions

            def __init__(self):
                """Create only the explicitly selected interpreter, with no permission grants."""
                session.deadline = time.monotonic() + session.config.startup_timeout_seconds
                session.interpreter = (
                    session.factory() if session.factory is not None else dspy.PythonInterpreter()
                )
                self.tools = session.interpreter.tools
                self.output_fields = None

            def start(self):
                """Start and register tools under the startup deadline."""
                if session.stopped.is_set():
                    raise REPLError("RLM session stopped")
                if hasattr(session.interpreter, "output_fields"):
                    session.interpreter.output_fields = self.output_fields
                session.interpreter.start()
                if isinstance(session.interpreter, dspy.PythonInterpreter):
                    session.interpreter.execute(
                        _SANDBOX_SETUP, variables={"_llgm_tool_names": list(self.tools)}
                    )
                session.deadline = None

            def execute(self, code, variables=None):
                """Execute generated code and validate cited submissions before returning them."""
                _bounded(code, session.config.max_code_bytes, "Python code")
                if on_execute is not None:
                    session.bridge(on_execute, code)
                session.deadline = time.monotonic() + session.config.execution_timeout_seconds
                try:
                    result = session.interpreter.execute(code, variables=variables)
                    if isinstance(result, FinalOutput):
                        _bounded(result.output, session.config.max_response_bytes, "RLM return")
                        if validate is not None:
                            try:
                                session.bridge(validate, result.output)
                            except SchemaError as error:
                                raise CodeExecutionError(str(error)) from error
                    else:
                        _bounded(result, session.config.max_output_bytes, "Python output")
                    return result
                finally:
                    session.deadline = None

            def shutdown(self):
                """Reap the process on its owning thread, including after an abort."""
                session.deadline = time.monotonic() + session.config.startup_timeout_seconds
                session.interpreter.shutdown()
                session.deadline = None

        fields = {"context": (dict, dspy.InputField())}
        fields.update(
            {
                name: (annotation, dspy.OutputField())
                for name, annotation in (
                    outputs or {"answer": str, "citations": list[str], "unresolved": list[str]}
                ).items()
            }
        )
        signature = dspy.Signature(fields, instructions)
        lm = HostLM()
        try:
            with dspy.context(lm=lm, adapter=HostAdapter(), callbacks=[]):
                rlm = dspy.RLM(
                    signature,
                    max_iters=max_steps,
                    max_llm_calls=max_llm_calls,
                    max_output_chars=self.config.max_output_bytes,
                    tools=list(tools),
                    sub_lm=lm,
                    interpreter_factory=Interpreter,
                )
                prediction = rlm(context=self.context)
            result = {name: getattr(prediction, name) for name in signature.output_fields}
            _bounded(result, self.config.max_response_bytes, "RLM return")
            if validate is not None:
                self.bridge(validate, result)
            return result
        except CodeInterpreterError as error:
            raise REPLError(str(error)) from error
        finally:
            # Factory setup may fail before DSPy enters its managed context.
            if self.interpreter is not None and getattr(self.interpreter, "deno_process", None):
                self.abort()
                self.interpreter.shutdown()
