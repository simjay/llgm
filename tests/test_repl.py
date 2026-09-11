"""REPL protocol/config tests: no Docker daemon, network, or model provider.

Host control paths use a fake subprocess transport. WorkerProtocolTests run only
fixed, trusted test snippets in an ordinary local subprocess; those tests verify
Python/stdio behavior and explicitly do NOT test or provide isolation.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import unittest
from unittest.mock import patch

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.inference.repl import _WORKER, DockerREPL, DockerREPLConfig, REPLError, REPLTimeoutError


def frame(value):
    """Encode a JSON protocol message with its four-byte payload-length header."""
    payload = json.dumps(value, ensure_ascii=False).encode()
    return struct.pack("!I", len(payload)) + payload


READY = {"type": "ready", "protocol": 1}


def result(execution_id=1, **kwargs):
    """Build a completed worker-result frame with optional field overrides."""
    return {
        "type": "result",
        "execution_id": execution_id,
        "stdout": "answer\n",
        "stdout_truncated": False,
        "error": None,
        **kwargs,
    }


def query(execution_id=1, query_id=1, **kwargs):
    """Build a worker callback request with explicit execution and query IDs."""
    return {
        "type": "query",
        "execution_id": execution_id,
        "query_id": query_id,
        "prompt": "Which source supports this?",
        **kwargs,
    }


class Writer:
    """In-memory stdin double that decodes and records protocol writes."""

    def __init__(self):
        """Initialize an empty ledger of frames sent to the worker."""
        self.frames = []

    def write(self, data):
        """Validate the payload length and retain the decoded host message."""
        size = struct.unpack("!I", data[:4])[0]
        assert len(data[4:]) == size
        self.frames.append(json.loads(data[4:]))

    async def drain(self):
        """Satisfy the async stream contract without introducing backpressure."""
        pass


class Process:
    """Subprocess double with scripted output, lifecycle state and stderr."""

    def __init__(self, responses=(), *, eof=False, returncode=None, stderr=b""):
        """Feed fixed worker frames and optionally terminate the output stream."""
        self.stdin = Writer()
        self.stdout = asyncio.StreamReader()
        for response in responses:
            self.stdout.feed_data(response if isinstance(response, bytes) else frame(response))
        if eof:
            self.stdout.feed_eof()
        self.returncode, self.stderr = returncode, stderr
        self.killed = False

    def kill(self):
        """Record forced termination and close the simulated stdout stream."""
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()

    async def wait(self):
        """Return the simulated process exit status."""
        return self.returncode

    async def communicate(self):
        """Expose the configured stderr body through the subprocess result shape."""
        return None, self.stderr


class Factory:
    """Subprocess factory separating Docker launch from cleanup calls."""

    def __init__(self, responses=(), *, eof=False, cleanup=None):
        """Prepare independent launch and cleanup outcomes with a call ledger."""
        self.process = Process(responses, eof=eof)
        self.cleanup = cleanup or Process(returncode=0)
        self.calls = []

    async def __call__(self, *args, **kwargs):
        """Record subprocess arguments and return the selected lifecycle double."""
        self.calls.append((args, kwargs))
        return self.process if args[1] == "run" else self.cleanup


class REPLConfigTests(unittest.TestCase):
    """REPL configuration and generated container-boundary contracts."""

    def test_config_rejects_invalid_numeric_values(self):
        """Resource and timeout settings reject invalid types and nonpositive or nonfinite values."""
        for name in ("cpus", "startup_timeout_seconds", "execution_timeout_seconds"):
            for value in (0, -1, True, "1", float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ConfigurationError):
                    DockerREPLConfig(**{name: value})
        for name in ("memory_mb", "pids_limit", "max_context_bytes", "max_llm_queries"):
            for value in (0, -1, True, 0.5, "1"):
                with self.subTest(name=name, value=value), self.assertRaises(ConfigurationError):
                    DockerREPLConfig(**{name: value})

    def test_image_and_executable_cannot_inject_options(self):
        """Image and executable values cannot smuggle command-line options or whitespace."""
        for value in ("", "--privileged", "image --volume=/", "python\n", None):
            for name in ("image", "docker_executable"):
                with self.subTest(value=value, name=name), self.assertRaises(ConfigurationError):
                    DockerREPLConfig(**{name: value})

    def test_context_requires_bounded_json_with_string_keys(self):
        """Initial context must be finite, acyclic, bounded JSON with string mapping keys."""
        cyclic = {}
        cyclic["cycle"] = cyclic
        for value in (
            None,
            [],
            {1: "one"},
            {"nested": [{1: 2}]},
            {"tuple": (1, 2)},
            {"number": float("nan")},
            {"number": float("inf")},
            {"unknown": object()},
            {"bad": "\ud800"},
            cyclic,
        ):
            with self.subTest(value=type(value)), self.assertRaises(ConfigurationError):
                DockerREPL(value)
        with self.assertRaises(ConfigurationError):
            DockerREPL({"text": "x" * 50}, config=DockerREPLConfig(max_context_bytes=20))

    def test_context_snapshot_and_callback_configuration(self):
        """Construction snapshots caller context and validates callback and size configuration."""
        context = {"items": [1]}
        repl = DockerREPL(context)
        context["items"].append(2)
        self.assertEqual(repl._context, {"items": [1]})
        with self.assertRaises(ConfigurationError):
            DockerREPL({}, llm_query="not callable")
        with self.assertRaises(ConfigurationError):
            DockerREPLConfig(max_context_bytes=128 * 1024 * 1024)

    def test_isolation_command_has_no_mounts_or_host_environment(self):
        """Container arguments exclude host mounts and environment while declaring resource limits."""
        repl = DockerREPL({"secret": "context-is-not-a-command-argument"})
        command = repl._command()
        for flag in (
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--user=65534:65534",
            "--cpus=0.5",
            "--memory=256m",
            "--memory-swap=256m",
            "--pids-limit=32",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m",
            "--workdir=/tmp",
        ):
            self.assertIn(flag, command)
        for argument in command[:-1]:
            self.assertFalse(argument.startswith(("--volume", "--mount", "--env", "--privileged")))
            self.assertNotIn("docker.sock", argument)
            self.assertNotIn("context-is-not-a-command-argument", argument)
        self.assertEqual(command[-4:-1], ["-I", "-u", "-c"])


class REPLHostTests(unittest.IsolatedAsyncioTestCase):
    """Host-side protocol, callback accounting and cleanup behavior with fake processes."""

    async def test_start_and_close_are_idempotent(self):
        """Repeated start/close calls do not duplicate container launch or successful cleanup."""
        factory = Factory([READY, result()])
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            async with DockerREPL({"document": "abc"}) as repl:
                await repl.start()
                actual = await repl.execute("print(context['document'])")
                self.assertEqual(actual.stdout, "answer\n")
                self.assertEqual(actual.llm_queries, 0)
                self.assertIsNone(actual.error)
            await repl.aclose()
            with self.assertRaises(REPLError):
                await repl.execute("print(1)")
        self.assertEqual([call[0][1] for call in factory.calls], ["run", "rm"])
        self.assertTrue(factory.process.killed)
        self.assertEqual(factory.process.stdin.frames[0]["context"], {"document": "abc"})
        self.assertEqual(factory.process.stdin.frames[1]["execution_id"], 1)
        self.assertNotIn("shell", factory.calls[0][1])
        self.assertEqual(factory.calls[0][1]["stderr"], asyncio.subprocess.DEVNULL)

    async def test_host_callback_is_awaited_and_has_invocation_id(self):
        """Async callback completion retains an invocation identity and byte accounting."""
        prompts = []

        async def callback(prompt):
            """Yield once before recording the callback prompt and returning a source handle."""
            await asyncio.sleep(0)
            prompts.append(prompt)
            return "source:12"

        factory = Factory([READY, query(), result()])
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            async with DockerREPL({}, llm_query=callback) as repl:
                actual = await repl.execute("print(llm_query('question'))")
                self.assertEqual(actual.llm_queries, 1)
                self.assertEqual(actual.query_events[0].status, "completed")
                self.assertEqual(actual.query_events[0].response_bytes, 9)
                self.assertTrue(actual.query_events[0].invocation_id.endswith(":1:1"))
                self.assertEqual(repl.query_events, list(actual.query_events))
        self.assertEqual(prompts, [query()["prompt"]])
        reply = factory.process.stdin.frames[-1]
        self.assertEqual(reply["type"], "query_result")
        self.assertEqual(reply["text"], "source:12")
        self.assertIsNone(reply["error"])

    async def test_host_exception_details_are_not_sent_into_guest(self):
        """Private callback exception details never cross into the guest response."""

        async def callback(prompt):
            """Raise an exception containing a sentinel that must remain host-private."""
            raise RuntimeError("private-key-do-not-leak")

        factory = Factory([READY, query(), result(error="RuntimeError: callback failed")])
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            async with DockerREPL({}, llm_query=callback) as repl:
                actual = await repl.execute("llm_query('x')")
                self.assertEqual(actual.query_events[0].status, "failed")
        self.assertNotIn("private-key", json.dumps(factory.process.stdin.frames))
        self.assertIsNotNone(factory.process.stdin.frames[-1]["error"])

    async def test_missing_sync_and_invalid_callbacks_return_explicit_errors(self):
        """Missing, synchronous, nontext and oversized callbacks produce explicit guest errors."""

        async def bad_type(prompt):
            """Return an invalid mapping instead of callback text."""
            return {"not": "text"}

        async def oversized(prompt):
            """Return multibyte text exceeding the configured response-byte ceiling."""
            return "é" * 3

        for callback in (None, lambda prompt: "sync", bad_type, oversized):
            with self.subTest(callback=callback):
                factory = Factory([READY, query(), result()])
                with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
                    async with DockerREPL(
                        {},
                        llm_query=callback,
                        config=DockerREPLConfig(max_response_bytes=5),
                    ) as repl:
                        actual = await repl.execute("llm_query('x')")
                        self.assertEqual(actual.query_events[0].status, "failed")
                self.assertIsNotNone(factory.process.stdin.frames[-1]["error"])
                self.assertIsNone(factory.process.stdin.frames[-1]["text"])

    async def test_callback_budget_aborts_before_excess_call(self):
        """The per-execution callback cap stops before an extra host invocation."""
        prompts = []

        async def callback(prompt):
            """Record permitted callback invocations before returning a short response."""
            prompts.append(prompt)
            return "ok"

        factory = Factory([READY, query(), query(query_id=2), result()])
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            repl = DockerREPL({}, llm_query=callback, config=DockerREPLConfig(max_llm_queries=1))
            with self.assertRaises(BudgetExceeded):
                await repl.execute("pass")
            self.assertTrue(factory.process.killed)
        self.assertEqual(len(prompts), 1)

    async def test_session_callback_budget_spans_executions(self):
        """Session callback allowances remain spent across separate execute calls."""

        async def callback(prompt):
            """Return one successful callback response for each permitted execution."""
            return "ok"

        factory = Factory([READY, query(), result(), query(execution_id=2), result(2)])
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            repl = DockerREPL(
                {},
                llm_query=callback,
                config=DockerREPLConfig(max_session_llm_queries=1),
            )
            self.assertEqual((await repl.execute("pass")).llm_queries, 1)
            with self.assertRaises(BudgetExceeded):
                await repl.execute("pass")

    async def test_callback_global_budget_failure_aborts_session(self):
        """A shared model-budget failure aborts the REPL rather than becoming recoverable output."""

        async def callback(prompt):
            """Raise the outer runtime's budget exhaustion from a callback."""
            raise BudgetExceeded("global model budget exhausted")

        factory = Factory([READY, query(), result()])
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            repl = DockerREPL({}, llm_query=callback)
            with self.assertRaises(BudgetExceeded):
                await repl.execute("pass")
            self.assertEqual(repl.query_events[0].status, "budget_exceeded")
            self.assertTrue(factory.process.killed)

    async def test_execution_timeout_includes_callback_and_kills_container(self):
        """The execution deadline covers suspended callbacks and forces container cleanup."""

        async def callback(prompt):
            """Keep the callback pending so the execution timeout must cancel it."""
            await asyncio.Event().wait()

        factory = Factory([READY, query()])
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            repl = DockerREPL(
                {},
                llm_query=callback,
                config=DockerREPLConfig(execution_timeout_seconds=0.02),
            )
            with self.assertRaises(REPLTimeoutError):
                await repl.execute("pass")
            self.assertEqual(repl.query_events[0].status, "cancelled")
            self.assertTrue(factory.process.killed)
            self.assertEqual(factory.calls[-1][0][1:3], ("rm", "--force"))
            with self.assertRaises(REPLError):
                await repl.start()

    async def test_startup_timeout_removes_named_container(self):
        """A missing startup handshake times out and removes the named container."""
        factory = Factory()
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            repl = DockerREPL({}, config=DockerREPLConfig(startup_timeout_seconds=0.01))
            with self.assertRaises(REPLTimeoutError):
                await repl.start()
            self.assertTrue(factory.process.killed)
            self.assertEqual(factory.calls[-1][0][-1], repl.container_name)

    async def test_cancellation_propagates_after_cleanup(self):
        """External cancellation propagates only after the container cleanup attempt."""
        factory = Factory([READY])
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            repl = DockerREPL({})
            task = asyncio.create_task(repl.execute("while True: pass"))
            while len(factory.process.stdin.frames) < 2:
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(factory.process.killed)
            self.assertEqual(factory.calls[-1][0][1], "rm")

    async def test_repeated_cancellation_drains_start_execute_and_close_cleanup(self):
        """Public operations retain their lock until removal finishes despite repeated cancellation."""
        for operation in ("start", "execute", "close"):
            for cleanup_fails in (False, True):
                with self.subTest(operation=operation, cleanup_fails=cleanup_fails):
                    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

                    class DelayedCleanup(Process):
                        """Hold the removal response while repeated cancellation reaches its owner."""

                        async def communicate(self):
                            """Signal removal completion only when the test releases the response."""
                            started.set()
                            await release.wait()
                            finished.set()
                            return None, self.stderr

                    cleanup = DelayedCleanup(
                        returncode=1 if cleanup_fails else 0,
                        stderr=b"daemon unavailable",
                    )
                    factory = Factory([] if operation == "start" else [READY], cleanup=cleanup)
                    with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
                        repl = DockerREPL({})
                        if operation == "close":
                            await repl.start()
                            task = asyncio.create_task(repl.aclose())
                        else:
                            task = asyncio.create_task(
                                repl.start() if operation == "start" else repl.execute("pass"),
                            )
                            expected_frames = 1 if operation == "start" else 2
                            while len(factory.process.stdin.frames) < expected_frames:
                                await asyncio.sleep(0)
                            task.cancel()
                        await started.wait()
                        for _ in range(2):
                            task.cancel()
                            await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        self.assertFalse(finished.is_set())
                        self.assertTrue(repl._lock.locked())
                        self.assertEqual([call[0][1] for call in factory.calls], ["run", "rm"])
                        release.set()
                        with self.assertRaises(asyncio.CancelledError) as caught:
                            await task
                        self.assertTrue(finished.is_set())
                        self.assertFalse(repl._lock.locked())
                        self.assertEqual(repl._cleanup_pending, cleanup_fails)
                        if cleanup_fails:
                            self.assertIn(repl.container_name, " ".join(caught.exception.__notes__))
                            cleanup.returncode = 0
                            await repl.aclose()
                            self.assertFalse(repl._cleanup_pending)

    async def test_same_task_callback_reentry_fails_without_deadlock(self):
        """A callback cannot reenter its own active execute call and deadlock the session."""

        async def callback(prompt):
            """Attempt a nested execute call from the currently awaited callback."""
            await repl.execute("print('nested')")
            return "unreachable"

        factory = Factory([READY, query(), result()])
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            async with DockerREPL({}, llm_query=callback) as repl:
                actual = await repl.execute("pass")
                self.assertEqual(actual.query_events[0].status, "failed")

    async def test_protocol_rejects_malformed_and_mismatched_frames(self):
        """Malformed frames and mismatched execution/query IDs terminate the protocol."""
        bad_frames = [
            struct.pack("!I", 0),
            struct.pack("!I", 2**31),
            b"\x00\x00\x00\x01{",
            struct.pack("!I", DockerREPLConfig().max_incoming_frame_bytes + 1),
            frame([]),
            b'{"type":"query"}',
            result(execution_id=99),
            result(execution_id=True),
            result(stdout_truncated=1),
            result(stdout=2),
            result(error={}),
            {"type": "unknown", "execution_id": 1},
            query(query_id=2),
            query(query_id=True),
            query(prompt=[]),
            query(unexpected="extra"),
            result(unexpected="extra"),
        ]
        for bad in bad_frames:
            with self.subTest(bad=bad):
                factory = Factory([READY, bad])
                with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
                    repl = DockerREPL({})
                    with self.assertRaises(REPLError):
                        await repl.execute("pass")
                    self.assertTrue(factory.process.killed)

    async def test_utf8_byte_limits_apply_to_code_output_query_and_error(self):
        """Code, output, query and error ceilings count UTF-8 bytes rather than characters."""
        cfg = DockerREPLConfig(
            max_code_bytes=5,
            max_output_bytes=5,
            max_query_bytes=5,
            max_error_bytes=5,
        )
        with self.assertRaises(REPLError):
            await DockerREPL({}, config=cfg).execute("é" * 3)
        for bad in (result(stdout="é" * 3), query(prompt="é" * 3), result(error="é" * 3)):
            factory = Factory([READY, bad])
            with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
                with self.assertRaises(REPLError):
                    await DockerREPL({}, config=cfg).execute("pass")

    async def test_eof_at_start_and_mid_execution_are_explicit(self):
        """EOF during startup or a partial frame is an explicit protocol failure."""
        for responses in ([], [READY], [READY, struct.pack("!I", 10) + b"ab"]):
            factory = Factory(responses, eof=True)
            with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
                with self.assertRaises(REPLError):
                    await DockerREPL({}).execute("pass")

    async def test_bad_handshake_aborts(self):
        """An incompatible worker protocol version aborts startup."""
        factory = Factory([{"type": "ready", "protocol": 2}])
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            with self.assertRaises(REPLError):
                await DockerREPL({}).start()
        self.assertTrue(factory.process.killed)

    async def test_missing_executable_is_explicit_and_never_falls_back(self):
        """A missing Docker executable cannot trigger a local-execution fallback."""
        calls = []

        async def unavailable(*args, **kwargs):
            """Record each attempted executable before simulating FileNotFoundError."""
            calls.append(args)
            raise FileNotFoundError("docker is not installed")

        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", unavailable):
            with self.assertRaisesRegex(REPLError, "could not start"):
                await DockerREPL({}).start()
        self.assertTrue(all(call[0] == "docker" for call in calls))

    async def test_cleanup_failure_is_visible_and_retryable(self):
        """Failed container removal remains visible and can be retried explicitly."""
        cleanup = Process(returncode=1, stderr=b"daemon unavailable")
        factory = Factory([READY], cleanup=cleanup)
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            repl = DockerREPL({})
            await repl.start()
            with self.assertRaisesRegex(REPLError, repl.container_name):
                await repl.aclose()
            cleanup.returncode = 0
            await repl.aclose()
            self.assertEqual([call[0][1] for call in factory.calls], ["run", "rm", "rm"])

    async def test_already_removed_container_is_clean(self):
        """An already removed container counts as successful cleanup."""
        factory = Factory(
            [READY],
            cleanup=Process(returncode=1, stderr=b"Error: No such container: name"),
        )
        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", factory):
            async with DockerREPL({}):
                pass

    async def test_removal_in_progress_requires_confirmed_exact_container_absence(self):
        """An observed removal conflict succeeds only after that container disappears."""
        for visible_first in (False, True):
            with self.subTest(visible_first=visible_first):
                repl = DockerREPL({})
                cleanup = Process(
                    returncode=1,
                    stderr=(
                        f"Error response from daemon: removal of container {repl.container_name} "
                        "is already in progress\n"
                    ).encode(),
                )
                factory = Factory([READY], cleanup=cleanup)
                inspections = [Process(returncode=0)] if visible_first else []
                inspections.append(
                    Process(
                        returncode=1,
                        stderr=(
                            f"Error response from daemon: No such container: {repl.container_name}\n"
                        ).encode(),
                    )
                )
                commands = []

                async def inspect_exact(*args, **kwargs):
                    """Script only exact-container inspection after the recorded removal conflict."""
                    if args[1] == "inspect":
                        commands.append(args)
                        return inspections.pop(0)
                    return await factory(*args, **kwargs)

                with patch("llgm.inference.repl.asyncio.create_subprocess_exec", inspect_exact):
                    await repl.start()
                    await repl.aclose()
                    await repl.aclose()
                self.assertFalse(repl._cleanup_pending)
                self.assertEqual([call[0][1] for call in factory.calls], ["run", "rm"])
                self.assertEqual(len(commands), 2 if visible_first else 1)
                self.assertTrue(
                    all(
                        command
                        == (
                            "docker",
                            "inspect",
                            "--type",
                            "container",
                            "--format",
                            "{{.Id}}",
                            repl.container_name,
                        )
                        for command in commands
                    )
                )

    async def test_removal_conflict_does_not_hide_visible_container_or_daemon_failure(self):
        """Verification timeout, daemon failure and a different missing name retain cleanup liability."""
        for failure in ("visible", "daemon", "wrong_container"):
            with self.subTest(failure=failure):
                repl = DockerREPL({}, config=DockerREPLConfig(cleanup_timeout_seconds=0.02))
                factory = Factory(
                    [READY],
                    cleanup=Process(
                        returncode=1,
                        stderr=(
                            f"Error response from daemon: removal of container {repl.container_name} "
                            "is already in progress\n"
                        ).encode(),
                    ),
                )
                inspection = Process(
                    returncode=0 if failure == "visible" else 1,
                    stderr=(
                        b"Error response from daemon: No such container: another-container\n"
                        if failure == "wrong_container"
                        else b"daemon unavailable"
                    ),
                )
                calls = []

                async def unconfirmed(*args, **kwargs):
                    """Never report absence of this session's exact container."""
                    if args[1] == "inspect":
                        calls.append(args)
                        return inspection
                    return await factory(*args, **kwargs)

                with patch("llgm.inference.repl.asyncio.create_subprocess_exec", unconfirmed):
                    await repl.start()
                    with self.assertRaisesRegex(REPLError, repl.container_name):
                        await repl.aclose()
                self.assertTrue(repl._cleanup_pending)
                self.assertTrue(calls)
                self.assertEqual([call[0][1] for call in factory.calls], ["run", "rm"])

    async def test_cancellation_during_removal_verification_keeps_cleanup_lock(self):
        """Repeated cancellation cannot release ownership while removal confirmation is pending."""
        repl = DockerREPL({})
        started, release = asyncio.Event(), asyncio.Event()
        factory = Factory(
            [READY],
            cleanup=Process(
                returncode=1,
                stderr=(
                    f"Error response from daemon: removal of container {repl.container_name} "
                    "is already in progress\n"
                ).encode(),
            ),
        )

        class PendingInspection(Process):
            """Keep the daemon confirmation pending until the test allows removal to finish."""

            async def communicate(self):
                """Report this container absent only after repeated owner cancellation."""
                started.set()
                await release.wait()
                return None, (
                    f"Error response from daemon: No such container: {repl.container_name}\n"
                ).encode()

        async def verify(*args, **kwargs):
            """Hold only the absence query after a removal conflict."""
            if args[1] == "inspect":
                return PendingInspection(returncode=1)
            return await factory(*args, **kwargs)

        with patch("llgm.inference.repl.asyncio.create_subprocess_exec", verify):
            await repl.start()
            closing = asyncio.create_task(repl.aclose())
            await started.wait()
            for _ in range(2):
                closing.cancel()
                await asyncio.sleep(0)
            self.assertFalse(closing.done())
            self.assertTrue(repl._lock.locked())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await closing
        self.assertFalse(repl._cleanup_pending)
        self.assertFalse(repl._lock.locked())


class WorkerProtocolTests(unittest.IsolatedAsyncioTestCase):
    """Trusted fixed snippets only; this subprocess is NOT an isolation backend."""

    async def asyncSetUp(self):
        """Start the worker with only fixed trusted context and validate its handshake."""
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-u",
            "-c",
            _WORKER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self.send(
            {
                "context": {"document": "zero one two", "nested": [1, 2]},
                "max_output_bytes": 12,
                "max_query_bytes": 40,
                "max_error_bytes": 100,
            }
        )
        self.assertEqual(await self.receive(), READY)

    async def asyncTearDown(self):
        """Terminate the local worker and drain its pipes after each protocol check."""
        if self.process.returncode is None:
            self.process.kill()
        await self.process.communicate()

    async def send(self, value):
        """Send a length-prefixed JSON message to the worker's stdin."""
        self.process.stdin.write(frame(value))
        await self.process.stdin.drain()

    async def receive(self):
        """Read one complete worker frame under a short transport deadline."""
        async with asyncio.timeout(3):
            header = await self.process.stdout.readexactly(4)
            payload = await self.process.stdout.readexactly(struct.unpack("!I", header)[0])
            return json.loads(payload)

    async def execute(self, code, execution_id=1):
        """Submit trusted test code and receive its next protocol response."""
        await self.send({"type": "execute", "execution_id": execution_id, "code": code})
        return await self.receive()

    async def test_context_persistence_and_partial_state_after_exception(self):
        """Worker globals persist across calls, including changes made before an exception."""
        first = await self.execute("words = context['document'].split(); count = len(words)")
        self.assertIsNone(first["error"])
        second = await self.execute("count += 1; raise ValueError('expected')", 2)
        self.assertEqual(second["error"], "ValueError: expected")
        third = await self.execute("print(count, words[1])", 3)
        self.assertEqual(third["stdout"], "4 one\n")

    async def test_printed_json_is_output_not_a_callback(self):
        """Printed JSON remains captured output rather than being interpreted as RPC."""
        actual = await self.execute('print(\'{"type":"query"}\')')
        self.assertEqual(actual["type"], "result")
        self.assertTrue(actual["stdout_truncated"])
        self.assertEqual(actual["stdout"], '{"type":"que')

    async def test_output_is_byte_bounded_and_does_not_accumulate_between_calls(self):
        """Captured output is byte-bounded and resets between executions."""
        actual = await self.execute("print('é' * 100000)")
        self.assertEqual(actual["stdout"], "é" * 6)
        self.assertTrue(actual["stdout_truncated"])
        self.assertEqual(len(actual["stdout"].encode()), 12)
        next_result = await self.execute("print('ok')", 2)
        self.assertEqual(next_result["stdout"], "ok\n")
        self.assertFalse(next_result["stdout_truncated"])

    async def test_utf8_cut_keeps_a_prefix_without_later_shorter_characters(self):
        """Truncation preserves a UTF-8 prefix without appending later shorter characters."""
        actual = await self.execute("print('x' * 11 + 'é')")
        self.assertEqual(actual["stdout"], "x" * 11)
        self.assertTrue(actual["stdout_truncated"])

    async def test_callback_protocol_roundtrip_and_persistent_answer(self):
        """A callback response returns through RPC and remains available in worker globals."""
        await self.send(
            {
                "type": "execute",
                "execution_id": 7,
                "code": "answer = llm_query(context['document']); print(answer)",
            }
        )
        request = await self.receive()
        self.assertEqual(request, query(execution_id=7, prompt="zero one two"))
        await self.send(
            {
                "type": "query_result",
                "execution_id": 7,
                "query_id": 1,
                "text": "three words",
                "error": None,
            }
        )
        actual = await self.receive()
        self.assertEqual(actual["stdout"], "three words\n")
        next_result = await self.execute("print(answer[:5])", 8)
        self.assertEqual(next_result["stdout"], "three\n")

    async def test_invalid_query_is_rejected_before_rpc(self):
        """Invalid callback arguments fail in the worker before sending any host request."""
        for code in ("llm_query(123)", "llm_query('x' * 41)", "llm_query('é' * 21)"):
            actual = await self.execute(code)
            self.assertEqual(actual["type"], "result")
            self.assertIsNotNone(actual["error"])

    async def test_host_callback_error_reaches_python_code(self):
        """A host callback error becomes an explicit exception in executed Python."""
        await self.send({"type": "execute", "execution_id": 1, "code": "llm_query('x')"})
        self.assertEqual((await self.receive())["type"], "query")
        await self.send(
            {
                "type": "query_result",
                "execution_id": 1,
                "query_id": 1,
                "text": None,
                "error": "callback unavailable",
            }
        )
        actual = await self.receive()
        self.assertEqual(actual["error"], "RuntimeError: callback unavailable")

    async def test_stdin_and_system_exit_do_not_escape_execution_control(self):
        """Input and SystemExit cannot terminate or escape the worker's execution loop."""
        actual = await self.execute("input()")
        self.assertTrue(actual["error"].startswith("EOFError:"))
        actual = await self.execute("raise SystemExit(9)", 2)
        self.assertEqual(actual["error"], "SystemExit: 9")
        self.assertEqual((await self.execute("print('alive')", 3))["stdout"], "alive\n")

    async def test_syntax_error_is_returned_and_stderr_is_capped(self):
        """Syntax errors become results and redirected stderr obeys the same output cap."""
        actual = await self.execute("if:")
        self.assertTrue(actual["error"].startswith("SyntaxError:"))
        actual = await self.execute("import sys; sys.stderr.write('e' * 100)", 2)
        self.assertEqual(actual["stdout"], "e" * 12)
        self.assertTrue(actual["stdout_truncated"])


if __name__ == "__main__":
    unittest.main()
