"""Persistent Python REPL with Docker isolation and bounded host callbacks.

Only an explicit JSON ``context`` crosses into the container. Python code runs
with ordinary Python builtins *inside Docker*, never via host ``exec``. Docker
is the isolation boundary. The Python namespace and framing are not sandboxes.
Use a trusted, locally available image (preferably pinned by digest). Images are
never pulled. The caller owns the authority, cost, and recursion policy of the
optional async callbacks. ``llm_query`` receives a prompt string. ``node_callback``
receives a structured evidence operation. A session accepts one callback mode.

The guest calls ``llm_query(prompt)`` synchronously while the host awaits the
callback. A callback may start another model/REPL, but cannot re-enter the same
busy REPL. Each execution's deadline includes callbacks. Timeouts, cancellation,
and protocol failures destroy the session. Python exceptions preserve state.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from llgm.core.errors import BudgetExceeded, ConfigurationError, LLGMError
from llgm.inference._json import validate_json_types


class REPLError(LLGMError):
    """The isolated interpreter could not start or its protocol failed."""


class REPLTimeoutError(REPLError, TimeoutError):
    """The execution deadline expired. The session has been aborted."""


@dataclass(frozen=True)
class DockerREPLConfig:
    """Docker resource limits and protocol bounds for one isolated REPL session."""

    image: str = "python:3.12-slim"
    docker_executable: str = "docker"
    cpus: float = 0.5
    memory_mb: int = 256
    pids_limit: int = 32
    tmp_mb: int = 16
    startup_timeout_seconds: float = 15.0
    execution_timeout_seconds: float = 30.0
    cleanup_timeout_seconds: float = 5.0
    max_context_bytes: int = 8 * 1024 * 1024
    max_code_bytes: int = 64 * 1024
    max_output_bytes: int = 32 * 1024
    max_query_bytes: int = 64 * 1024
    max_response_bytes: int = 64 * 1024
    max_error_bytes: int = 4096
    max_llm_queries: int = 16
    max_session_llm_queries: int = 64

    def __post_init__(self) -> None:
        """Validate Docker arguments, positive resource limits, and maximum frame size."""
        for name in ("image", "docker_executable"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.startswith("-"):
                raise ConfigurationError(f"{name} must be a nonempty non-option string")
            if any(character.isspace() or ord(character) < 32 for character in value):
                raise ConfigurationError(
                    f"{name} must not contain whitespace or control characters"
                )
        for name in (
            "cpus",
            "startup_timeout_seconds",
            "execution_timeout_seconds",
            "cleanup_timeout_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigurationError(f"{name} must be finite and positive")
        for name in (
            "memory_mb",
            "pids_limit",
            "tmp_mb",
            "max_context_bytes",
            "max_code_bytes",
            "max_output_bytes",
            "max_query_bytes",
            "max_response_bytes",
            "max_error_bytes",
            "max_llm_queries",
            "max_session_llm_queries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if self.max_frame_bytes > 128 * 1024 * 1024:
            raise ConfigurationError("REPL configured frame limits must fit within 128 MiB")

    @property
    def max_frame_bytes(self) -> int:
        # JSON control-character escaping may expand text by six bytes per byte.
        """Bound outgoing JSON frames including worst-case character escaping."""
        return max(
            self.max_context_bytes + 4096,
            6 * self.max_code_bytes + 4096,
            6 * (self.max_output_bytes + self.max_error_bytes) + 4096,
            6 * max(self.max_query_bytes, self.max_response_bytes) + 4096,
        )

    @property
    def max_incoming_frame_bytes(self) -> int:
        # The guest never sends the full context back as a protocol field.
        # Its untrusted frames therefore get the much smaller output/query cap.
        """Bound guest responses independently of the larger initial context allowance."""
        return 6 * max(self.max_output_bytes + self.max_error_bytes, self.max_query_bytes) + 4096


@dataclass(frozen=True)
class REPLQueryEvent:
    """Callback identity and byte counts without storing prompt or response content."""

    invocation_id: str
    prompt_bytes: int
    status: str
    response_bytes: int | None = None


@dataclass(frozen=True)
class REPLResult:
    """Bounded captured output, Python error, and callback events for one execution."""

    stdout: str
    error: str | None
    stdout_truncated: bool
    llm_queries: int
    query_events: tuple[REPLQueryEvent, ...] = ()


# This source is passed as a command argument to Python *inside the container*.
# Neither evidence nor generated code is placed on the process command line.
_WORKER = r'''
"""Container-only framed interpreter with bounded output and synchronous host callbacks."""
import contextlib, io, json, struct, sys

wire_in, wire_out = sys.stdin.buffer, sys.stdout.buffer

def read_frame():
    """Read one length-prefixed JSON request from the host pipe."""
    header = wire_in.read(4)
    if len(header) != 4:
        raise EOFError("Missing frame")
    size = struct.unpack("!I", header)[0]
    if size < 1 or size > 128 * 1024 * 1024:
        raise ValueError("Invalid frame size")
    payload = wire_in.read(size)
    if len(payload) != size:
        raise EOFError("Incomplete frame")
    return json.loads(payload)

def send(value):
    """Flush one length-prefixed JSON response to the host pipe."""
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    wire_out.write(struct.pack("!I", len(payload)) + payload)
    wire_out.flush()

settings = read_frame()
output_limit = settings["max_output_bytes"]
query_limit = settings["max_query_bytes"]
error_limit = settings["max_error_bytes"]

class Capture(io.TextIOBase):
    """Capture stdout and stderr within a UTF-8 byte allowance."""
    def __init__(self):
        """Start an empty output capture with the configured byte allowance."""
        self.remaining, self.parts, self.truncated = output_limit, [], False
    def writable(self):
        """Advertise the writable stream interface used by redirected print calls."""
        return True
    def write(self, value):
        """Retain a bounded UTF-8 prefix while reporting the original write length."""
        if not isinstance(value, str):
            raise TypeError("write expects str")
        # Encode only a bounded prefix: huge print arguments cannot grow capture.
        chunk = value[:self.remaining].encode("utf-8", "replace")[:self.remaining]
        text = chunk.decode("utf-8", "ignore")
        self.remaining -= len(text.encode("utf-8"))
        if text:
            self.parts.append(text)
        self.truncated |= len(text) < len(value)
        if self.truncated:
            self.remaining = 0
        return len(value)
    def flush(self):
        """Satisfy the stream contract; captured output is emitted after execution."""
        pass

execution_id, query_id = 0, 0

def llm_query(prompt):
    """Request one host callback and validate the matching synchronous response."""
    global query_id
    if not isinstance(prompt, str):
        raise TypeError("llm_query expects a string")
    if len(prompt) > query_limit or len(prompt.encode("utf-8")) > query_limit:
        raise ValueError("llm_query prompt exceeds its byte limit")
    query_id += 1
    send({"type": "query", "execution_id": execution_id,
          "query_id": query_id, "prompt": prompt})
    reply = read_frame()
    if (reply.get("type") != "query_result" or
        reply.get("execution_id") != execution_id or reply.get("query_id") != query_id):
        raise RuntimeError("Invalid callback response")
    if reply.get("error") is not None:
        raise RuntimeError(reply["error"])
    return reply["text"]

scope = {"__name__": "__llgm_repl__", "context": settings["context"],
         "llm_query": llm_query, "__builtins__": __builtins__}
if settings.get("node_api", False):
    def node_call(operation, **arguments):
        """Exchange one finite JSON evidence operation through the existing callback transport."""
        return json.loads(llm_query(json.dumps({"op": operation, **arguments}, allow_nan=False)))
    def read(reference):
        """Read amended evidence lazily through a canonical reference dictionary."""
        return node_call("read", reference=reference)
    def source_info(node_id=None, offset=0, limit=32):
        """List bounded canonical turn handles without loading source text into Python."""
        return node_call("source_info", node_id=node_id, offset=offset, limit=limit)
    def edges(node_id=None, relation=None):
        """Discover primary graph neighbors without reading their source text."""
        return node_call("edges", node_id=node_id, relation=relation)
    def query_node(node_id, question):
        """Request an isolated recursive node computation and its cited findings."""
        return node_call("query_node", node_id=node_id, question=question)
    def search(query, k=5):
        """Discover bounded candidate handles without copying source text into the interpreter."""
        return node_call("search", query=query, k=k)
    scope.update(read=read, source_info=source_info, edges=edges, query_node=query_node, search=search)
send({"type": "ready", "protocol": 1})
while True:
    try:
        request = read_frame()
    except EOFError:
        break
    execution_id, query_id = request["execution_id"], 0
    capture, error = Capture(), None
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO("")
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            try:
                exec(compile(request["code"], "<llgm-repl>", "exec"), scope, scope)
            except BaseException as exc:
                error = (type(exc).__name__ + ": " + str(exc))
                error = error[:error_limit].encode("utf-8", "replace")[:error_limit]
                error = error.decode("utf-8", "ignore")
    finally:
        sys.stdin = old_stdin
    send({"type": "result", "execution_id": execution_id,
          "stdout": "".join(capture.parts), "stdout_truncated": capture.truncated,
          "error": error})
'''


def _json(value: Any) -> bytes:
    """Serialize finite UTF-8 JSON inputs and normalize encoding failures."""
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ConfigurationError("REPL inputs must be finite, UTF-8 JSON data") from exc


class DockerREPL:
    """A persistent, serialized Python session in a disposable Docker container.

    Use ``async with DockerREPL({"document": text}) as repl`` then
    ``await repl.execute('print(context["document"][:200])')``. Outputs are
    UTF-8 byte bounded. Variables and partial assignments survive Python errors.
    Closed/aborted sessions cannot be restarted. No host directory, environment
    variable, credential, or Docker socket is mounted/passed into the container.
    The Docker CLI itself uses the caller's Docker connection configuration.
    """

    def __init__(
        self,
        evidence: Mapping[str, Any],
        *,
        config: DockerREPLConfig | None = None,
        llm_query: Callable[[str], Awaitable[str]] | None = None,
        node_callback: Callable[[dict], Awaitable[dict]] | None = None,
    ) -> None:
        """Snapshot JSON context and configure a session without launching Docker."""
        self.config = config or DockerREPLConfig()
        if not isinstance(evidence, Mapping):
            raise ConfigurationError("REPL evidence must be a JSON object")
        context = dict(evidence)
        # Serialize first to reject cycles before recursively validating key types.
        encoded = _json(context)
        if len(encoded) > self.config.max_context_bytes:
            raise ConfigurationError("REPL context exceeds max_context_bytes")
        try:
            validate_json_types(context)
        except RecursionError as exc:
            raise ConfigurationError("REPL context nesting is too deep") from exc
        self._context = json.loads(encoded)  # Own a snapshot, not caller-mutated state.
        if llm_query is not None and not callable(llm_query):
            raise ConfigurationError("llm_query must be an async callable")
        if node_callback is not None and not callable(node_callback):
            raise ConfigurationError("node_callback must be an async callable")
        if node_callback is not None and llm_query is not None:
            raise ConfigurationError("Configure either llm_query or node_callback")
        self._node_callback = node_callback
        self._callback = llm_query
        self._name = "llgm-repl-" + uuid.uuid4().hex
        self._process: asyncio.subprocess.Process | None = None
        self._closed = False
        self._cleanup_pending = False
        self._lock = asyncio.Lock()
        self._executing_task: asyncio.Task[Any] | None = None
        self._execution_id = 0
        self._session_queries = 0
        self.query_events: list[REPLQueryEvent] = []

    @property
    def container_name(self) -> str:
        """Return the unique container identifier used for exact-session cleanup."""
        return self._name

    def _command(self) -> list[str]:
        """Build an isolated Docker command without pulling an image or mounting host data."""
        cfg = self.config
        return [
            cfg.docker_executable,
            "run",
            "--rm",
            "--interactive",
            "--pull=never",
            "--name",
            self._name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--user=65534:65534",
            f"--cpus={cfg.cpus}",
            f"--memory={cfg.memory_mb}m",
            f"--memory-swap={cfg.memory_mb}m",
            f"--pids-limit={cfg.pids_limit}",
            f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={cfg.tmp_mb}m",
            "--workdir=/tmp",
            "--entrypoint=python",
            cfg.image,
            "-I",
            "-u",
            "-c",
            _WORKER,
        ]

    async def __aenter__(self) -> DockerREPL:
        """Start the isolated interpreter and return its session handle."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Destroy the session when its async context ends."""
        await self.aclose()

    async def start(self) -> None:
        """Serialize startup with execution and cleanup operations."""
        async with self._lock:
            await self._start()

    async def _start(self) -> None:
        """Launch once, send context, and require the worker protocol handshake."""
        if self._closed:
            raise REPLError("The REPL session is closed")
        if self._process is not None:
            return
        try:
            async with asyncio.timeout(self.config.startup_timeout_seconds):
                self._cleanup_pending = True
                self._process = await asyncio.create_subprocess_exec(
                    *self._command(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await self._send(
                    {
                        "type": "init",
                        "context": self._context,
                        "max_output_bytes": self.config.max_output_bytes,
                        "max_query_bytes": self.config.max_query_bytes,
                        "max_error_bytes": self.config.max_error_bytes,
                        "node_api": self._node_callback is not None,
                    }
                )
                if await self._receive() != {"type": "ready", "protocol": 1}:
                    raise REPLError("Unsupported REPL worker handshake")
        except BaseException as exc:
            await self._drain_cleanup(exc)
            if isinstance(exc, TimeoutError):
                raise REPLTimeoutError("Docker REPL startup timed out") from exc
            if isinstance(exc, (OSError, asyncio.IncompleteReadError)):
                raise REPLError(
                    "Docker REPL could not start; check the daemon and local image. "
                    "Images are never pulled automatically."
                ) from exc
            raise

    async def _send(self, value: dict[str, Any]) -> None:
        """Write one bounded length-prefixed JSON frame to the guest."""
        payload = _json(value)
        if len(payload) > self.config.max_frame_bytes:
            raise REPLError("Outgoing REPL frame exceeds its byte limit")
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(struct.pack("!I", len(payload)) + payload)
        await self._process.stdin.drain()

    async def _receive(self) -> dict[str, Any]:
        """Read and validate a bounded guest JSON-object frame."""
        assert self._process is not None and self._process.stdout is not None
        header = await self._process.stdout.readexactly(4)
        size = struct.unpack("!I", header)[0]
        if size < 1 or size > self.config.max_incoming_frame_bytes:
            raise REPLError("Incoming REPL frame exceeds its byte limit or is empty")
        payload = await self._process.stdout.readexactly(size)
        try:
            result = json.loads(payload)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise REPLError("Invalid JSON in REPL frame") from exc
        if not isinstance(result, dict):
            raise REPLError("REPL frame must be an object")
        return result

    @staticmethod
    def _text_size(value: Any, maximum: int, field: str) -> int:
        """Validate a text field and return its UTF-8 byte size."""
        if not isinstance(value, str) or len(value) > maximum:
            raise REPLError(f"REPL {field} is invalid or exceeds its byte limit")
        try:
            size = len(value.encode("utf-8"))
        except UnicodeError as exc:
            raise REPLError(f"REPL {field} is not valid UTF-8") from exc
        if size > maximum:
            raise REPLError(f"REPL {field} exceeds its byte limit")
        return size

    async def execute(self, code: str) -> REPLResult:
        """Run code serially. Preserve state on Python errors and abort on transport failure."""
        self._text_size(code, self.config.max_code_bytes, "code")
        if self._executing_task is asyncio.current_task():
            raise REPLError("A callback cannot re-enter the same REPL")
        async with self._lock:
            await self._start()
            self._execution_id += 1
            execution_id, count, start_event = self._execution_id, 0, len(self.query_events)
            self._executing_task = asyncio.current_task()
            try:
                async with asyncio.timeout(self.config.execution_timeout_seconds):
                    await self._send(
                        {
                            "type": "execute",
                            "execution_id": execution_id,
                            "code": code,
                        }
                    )
                    while True:
                        frame = await self._receive()
                        if (
                            type(frame.get("execution_id")) is not int
                            or frame["execution_id"] != execution_id
                        ):
                            raise REPLError("REPL frame has an invalid execution ID")
                        if frame.get("type") == "query":
                            count += 1
                            await self._query(frame, execution_id, count)
                            continue
                        if (
                            set(frame)
                            != {
                                "type",
                                "execution_id",
                                "stdout",
                                "stdout_truncated",
                                "error",
                            }
                            or frame["type"] != "result"
                        ):
                            raise REPLError("Unexpected REPL frame")
                        self._text_size(frame["stdout"], self.config.max_output_bytes, "stdout")
                        if type(frame["stdout_truncated"]) is not bool:
                            raise REPLError("REPL stdout_truncated must be boolean")
                        if frame["error"] is not None:
                            self._text_size(frame["error"], self.config.max_error_bytes, "error")
                        return REPLResult(
                            frame["stdout"],
                            frame["error"],
                            frame["stdout_truncated"],
                            count,
                            tuple(self.query_events[start_event:]),
                        )
            except BaseException as exc:
                await self._drain_cleanup(exc)
                if isinstance(exc, TimeoutError):
                    raise REPLTimeoutError("Docker REPL execution timed out") from exc
                if isinstance(exc, (OSError, asyncio.IncompleteReadError)):
                    raise REPLError("Docker REPL connection ended during execution") from exc
                raise
            finally:
                self._executing_task = None

    async def _query(self, frame: dict[str, Any], execution_id: int, count: int) -> None:
        """Validate and count a guest callback before dispatching its configured host operation."""
        if (
            set(frame) != {"type", "execution_id", "query_id", "prompt"}
            or type(frame.get("query_id")) is not int
            or frame["query_id"] != count
        ):
            raise REPLError("Invalid REPL callback frame or query ID")
        size = self._text_size(frame["prompt"], self.config.max_query_bytes, "query prompt")
        if (
            count > self.config.max_llm_queries
            or self._session_queries >= self.config.max_session_llm_queries
        ):
            raise BudgetExceeded("REPL callback count budget exhausted")
        self._session_queries += 1
        invocation_id = f"{self._name}:{execution_id}:{count}"
        event_index = len(self.query_events)
        self.query_events.append(REPLQueryEvent(invocation_id, size, "running"))
        reply: dict[str, Any] = {
            "type": "query_result",
            "execution_id": execution_id,
            "query_id": count,
            "text": None,
            "error": None,
        }
        try:
            if self._node_callback is not None:
                request = json.loads(frame["prompt"])
                if not isinstance(request, dict):
                    raise REPLError("Node callback request must be a JSON object")
                response = _json(await self._node_callback(request)).decode("utf-8")
            elif self._callback is None:
                raise REPLError("No LLM callback configured")
            else:
                response = await self._callback(frame["prompt"])
            response_size = self._text_size(
                response,
                self.config.max_response_bytes,
                "callback response",
            )
            reply["text"] = response
            self.query_events[event_index] = REPLQueryEvent(
                invocation_id,
                size,
                "completed",
                response_size,
            )
        except asyncio.CancelledError:
            self.query_events[event_index] = REPLQueryEvent(invocation_id, size, "cancelled")
            raise
        except BudgetExceeded:
            self.query_events[event_index] = REPLQueryEvent(invocation_id, size, "budget_exceeded")
            raise
        except Exception:
            # Do not put host exception messages (possibly secrets) into the guest.
            reply["error"] = "Host LLM callback failed or returned invalid output"
            self.query_events[event_index] = REPLQueryEvent(invocation_id, size, "failed")
        await self._send(reply)

    async def _drain_cleanup(self, original: BaseException | None = None) -> None:
        """Retain cleanup ownership through repeated cancellation, preserving failure details."""
        cleanup_task = asyncio.create_task(self._destroy())
        cancellation = None
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as cancelled:
                cancellation = cancelled
            except Exception:
                break
        try:
            cleanup_task.result()
        except Exception as cleanup:
            failure = cancellation or original
            if failure is None:
                raise
            failure.add_note(str(cleanup))
        if cancellation is not None:
            raise cancellation

    async def aclose(self) -> None:
        """Close the session without allowing a callback to close its own active execution."""
        if self._executing_task is asyncio.current_task():
            raise REPLError("A callback cannot close its own active REPL")
        async with self._lock:
            await self._drain_cleanup()

    async def _destroy(self) -> None:
        """Stop the local process and confirm removal of this session's container."""
        self._closed = True
        if self._process is not None and self._process.returncode is None:
            try:
                self._process.kill()
                await asyncio.wait_for(self._process.wait(), self.config.cleanup_timeout_seconds)
            except (TimeoutError, ProcessLookupError):
                pass
        if not self._cleanup_pending:
            return
        cleanup = None
        try:
            async with asyncio.timeout(self.config.cleanup_timeout_seconds):
                cleanup = await asyncio.create_subprocess_exec(
                    self.config.docker_executable,
                    "rm",
                    "--force",
                    self._name,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await cleanup.communicate()
                if cleanup.returncode != 0 and b"No such container" not in (stderr or b""):
                    race = f"removal of container {self._name} is already in progress".encode()
                    if race not in (stderr or b""):
                        raise REPLError("Docker did not confirm container removal")
                    # --rm may already own removal after the attached CLI exits.
                    # Confirm absence within this deadline without issuing another rm.
                    while True:
                        cleanup = await asyncio.create_subprocess_exec(
                            self.config.docker_executable,
                            "inspect",
                            "--type",
                            "container",
                            "--format",
                            "{{.Id}}",
                            self._name,
                            stdin=asyncio.subprocess.DEVNULL,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        _, stderr = await cleanup.communicate()
                        if cleanup.returncode != 0:
                            missing = f"No such container: {self._name}".encode()
                            if (stderr or b"").strip() not in {
                                b"Error response from daemon: " + missing,
                                b"Error: " + missing,
                            }:
                                raise REPLError("Docker did not confirm container absence")
                            break
                        await asyncio.sleep(0.05)
                self._cleanup_pending = False
        except (OSError, TimeoutError, REPLError) as exc:
            if cleanup is not None and cleanup.returncode is None:
                cleanup.kill()
                await cleanup.wait()
            raise REPLError(
                f"Could not confirm removal of {self._name}; explicit Docker cleanup may be needed"
            ) from exc
