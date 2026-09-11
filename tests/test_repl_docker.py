"""Opt-in integration tests for the actual Docker REPL boundary.

Run ``LLGM_TEST_DOCKER=1 python -m pytest tests/test_repl_docker.py`` with a
running local Docker daemon and an already available ``python:3.12-slim`` image.
``LLGM_REPL_DOCKER_IMAGE`` can select another trusted existing Python image,
including a digest-pinned image. No test pulls images. Once opted in, missing
Docker/daemon/image is a failure, not a skip. No hosted model calls are made.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from llgm.core.errors import ConfigurationError
from llgm.inference.repl import DockerREPL, DockerREPLConfig, REPLError, REPLTimeoutError

pytestmark = [pytest.mark.integration, pytest.mark.docker]


class DockerREPLIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Opted-in checks of the actual Docker isolation and protocol boundary."""

    @classmethod
    def setUpClass(cls):
        """Require explicit opt-in, a running daemon and an existing trusted Python image."""
        gate = os.environ.get("LLGM_TEST_DOCKER", "0")
        if gate == "0":
            raise unittest.SkipTest("Set LLGM_TEST_DOCKER=1")
        if gate != "1":
            raise ConfigurationError("LLGM_TEST_DOCKER must be exactly 0 or 1")
        cls.image = os.environ.get("LLGM_REPL_DOCKER_IMAGE", "python:3.12-slim")
        cls.config = DockerREPLConfig(image=cls.image)
        for command in (
            ["docker", "version", "--format", "{{json .Server.Version}}"],
            ["docker", "image", "inspect", cls.image],
        ):
            completed = subprocess.run(command, capture_output=True, timeout=15, check=False)
            if completed.returncode:
                raise RuntimeError(
                    "Docker integration was explicitly requested, but the local daemon or "
                    "configured Python image is unavailable. No image was pulled."
                )

    async def docker(self, *arguments):
        """Execute a bounded Docker inspection command without shell interpretation."""
        process = await asyncio.create_subprocess_exec(
            "docker",
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        except BaseException:
            process.kill()
            await process.wait()
            raise
        return process.returncode, stdout, stderr

    async def test_actual_container_isolation_and_resource_configuration(self):
        """The actual container lacks host files, secrets, network access and writable root state."""
        with tempfile.TemporaryDirectory(prefix="llgm-host-only-") as directory:
            host_file = Path(directory) / "host-secret.txt"
            host_file.write_text("do not expose to container")
            with patch.dict(os.environ, {"LLGM_REPL_HOST_SECRET": "not-for-the-guest"}):
                async with DockerREPL(
                    {"host_path": str(host_file), "source": "authorized text"},
                    config=self.config,
                ) as repl:
                    actual = await repl.execute(
                        "import errno, json, os, socket\n"
                        "try:\n"
                        "    open('/llgm-disallowed-write', 'w').write('no')\n"
                        "    root_write = True\n"
                        "except OSError:\n"
                        "    root_write = False\n"
                        "with socket.socket() as probe:\n"
                        "    probe.settimeout(0.5)\n"
                        "    egress_error = probe.connect_ex(('192.0.2.1', 443))\n"
                        "print(json.dumps({'uid': os.getuid(), "
                        "'secret': os.environ.get('LLGM_REPL_HOST_SECRET'), "
                        "'host_file': os.path.exists(context['host_path']), "
                        "'interfaces': os.listdir('/sys/class/net'), "
                        "'egress_error': errno.errorcode.get(egress_error), "
                        "'routes': open('/proc/net/route').read().splitlines()[1:], "
                        "'root_write': root_write, "
                        "'source': context['source']}))"
                    )
                    self.assertIsNone(actual.error)
                    observed = json.loads(actual.stdout)
                    self.assertEqual(observed["uid"], 65534)
                    self.assertIsNone(observed["secret"])
                    self.assertFalse(observed["host_file"])
                    self.assertFalse(observed["root_write"])
                    # Linux may expose inactive tunnel interfaces even with network=none.
                    self.assertIn("lo", observed["interfaces"])
                    self.assertEqual(observed["routes"], [])
                    self.assertEqual(observed["egress_error"], "ENETUNREACH")
                    self.assertEqual(observed["source"], "authorized text")
                    status, stdout, _ = await self.docker("inspect", repl.container_name)
                    self.assertEqual(status, 0)
                    container = json.loads(stdout)[0]
                    host = container["HostConfig"]
                    self.assertEqual(host["NetworkMode"], "none")
                    self.assertTrue(host["ReadonlyRootfs"])
                    self.assertFalse(host["Privileged"])
                    self.assertIn("ALL", host["CapDrop"])
                    self.assertIn("no-new-privileges:true", host["SecurityOpt"])
                    self.assertEqual(host["Memory"], self.config.memory_mb * 1024 * 1024)
                    self.assertEqual(host["MemorySwap"], host["Memory"])
                    self.assertEqual(host["PidsLimit"], self.config.pids_limit)
                    self.assertEqual(host["NanoCpus"], int(self.config.cpus * 1_000_000_000))
                    self.assertFalse(any(mount["Type"] == "bind" for mount in container["Mounts"]))
                status, _, _ = await self.docker("inspect", repl.container_name)
                self.assertNotEqual(status, 0)

    async def test_persistent_context_and_real_async_callback_roundtrip(self):
        """Persistent worker state survives an actual async host-callback roundtrip."""
        seen = []

        async def callback(prompt):
            """Yield to the event loop and retain the prompt seen by the real RPC callback."""
            await asyncio.sleep(0)
            seen.append(prompt)
            return "child saw: " + prompt

        async with DockerREPL(
            {"document": "alpha beta gamma"},
            llm_query=callback,
            config=self.config,
        ) as repl:
            first = await repl.execute("parts = context['document'].split(); first = parts[0]")
            self.assertIsNone(first.error)
            second = await repl.execute("child_answer = llm_query(first); print(child_answer)")
            self.assertIsNone(second.error)
            self.assertEqual(second.stdout, "child saw: alpha\n")
            self.assertEqual(second.llm_queries, 1)
            self.assertEqual(second.query_events[0].status, "completed")
            third = await repl.execute("print(parts[-1], child_answer)")
            self.assertEqual(third.stdout, "gamma child saw: alpha\n")
        self.assertEqual(seen, ["alpha"])

    async def test_output_cap_and_timeout_destroy_container(self):
        """Output truncation and execution timeout apply inside the actual container."""
        config = DockerREPLConfig(
            image=self.image,
            max_output_bytes=17,
            execution_timeout_seconds=0.3,
        )
        async with DockerREPL({}, config=config) as repl:
            actual = await repl.execute("print('x' * 1000000)")
            self.assertEqual(actual.stdout, "x" * 17)
            self.assertTrue(actual.stdout_truncated)
            with self.assertRaises(REPLTimeoutError):
                await repl.execute("while True: pass")
            status, _, _ = await self.docker("inspect", repl.container_name)
            self.assertNotEqual(status, 0)

    async def test_raw_stdout_protocol_corruption_is_rejected(self):
        """Raw writes that corrupt stdout framing terminate and remove the container."""
        async with DockerREPL({}, config=self.config) as repl:
            with self.assertRaises(REPLError):
                await repl.execute("import os; os.write(1, b'not-a-protocol-frame')")
            status, _, _ = await self.docker("inspect", repl.container_name)
            self.assertNotEqual(status, 0)

    async def test_node_runtime_recurses_through_actual_python_and_canonical_evidence(self):
        """Actual isolated Python discovers a primary neighbor, queries it, and returns cited findings."""
        from llgm.core.types import Conversation, NodeRef, Provenance
        from llgm.inference.budget import Budget
        from llgm.inference.nodes import NodeRuntime, NodeSeed
        from llgm.memory.evidence import Evidence
        from llgm.memory.query import QueryEvidence
        from llgm.memory.workspace import Workspace
        from llgm.models import CallableModelClient, ModelResponse

        sessions, root_requests, child_requests = [], [], []

        def factory(context, *, config, node_callback):
            """Record real container identities without replacing execution or callback transport."""
            session = DockerREPL(context, config=config, node_callback=node_callback)
            sessions.append(session)
            return session

        def finish(records, unresolved=()):
            """Cite IDs learned from actual transport results rather than predicting their order."""
            return ModelResponse(
                json.dumps(
                    {
                        "op": "finish",
                        "answer": "Region and date found",
                        "citations": list(dict.fromkeys(record["id"] for record in records)),
                        "unresolved": list(unresolved),
                    }
                )
            )

        async def child(request):
            """Generate bounded Python whose callbacks enter the same node runtime recursively."""
            child_requests.append(request)
            context = json.loads(request.messages[1].content)
            if len(request.messages) == 2:
                if context["node_id"] == "a":
                    code = (
                        'import json\nneighbor = edges()["references"][0]\n'
                        'finding = query_node(neighbor["node_id"], "Find the region")\n'
                        "print(json.dumps(finding))"
                    )
                else:
                    code = (
                        "import json\ninfo = source_info(limit=1)\n"
                        'reference = info["turns"][0]["reference"]\n'
                        "print(json.dumps(read(reference)))"
                    )
                return ModelResponse(json.dumps({"op": "python", "code": code}))
            observation = json.loads(request.messages[-1].content)
            self.assertIsNone(observation["error"])
            result = json.loads(observation["stdout"])
            return finish(result["evidence"], result.get("unresolved", []))

        async def root(request):
            """Synthesize once from completed branches after real descendants have returned."""
            root_requests.append(request)
            payload = json.loads(request.messages[1].content)
            return finish(
                payload["evidence"],
                [gap for branch in payload["branches"] for gap in branch["unresolved"]],
            )

        with tempfile.TemporaryDirectory(prefix="llgm-node-docker-") as directory:
            async with Workspace.open(directory) as workspace:
                for node_id, text in (
                    ("a", "Unneeded long source " * 50),
                    ("b", "Production region: eu-west-1"),
                    ("c", "Rollout date: 2031-04-07"),
                ):
                    await workspace.ingest(
                        Conversation.from_turns(
                            [{"role": "user", "text": text}],
                            node_id=node_id,
                        )
                    )
                await workspace.publish_edge(
                    "a",
                    "b",
                    relation="depends_on",
                    provenance=Provenance("user", "docker-contract"),
                )
                await workspace.append_journal(
                    "a",
                    subject=NodeRef("a"),
                    relation="note",
                    value="Complete operational journal sentinel",
                    provenance=Provenance("user", "docker-contract"),
                )
                async with await Evidence.open(workspace) as evidence:
                    engine = NodeRuntime(
                        CallableModelClient(root),
                        CallableModelClient(child),
                        QueryEvidence(evidence, {}, None),
                        budget=Budget(
                            max_model_calls=20,
                            max_sidecar_calls=18,
                            max_context_tokens=50000,
                            max_evidence_tokens=50000,
                            max_bundle_tokens=12000,
                        ),
                        max_concurrency=2,
                        repl_config=self.config,
                        repl_factory=factory,
                    )
                    result = await engine.answer(
                        "Region and date?", seeds=[NodeSeed("a"), NodeSeed("c")]
                    )
                    self.assertEqual(result.status, "completed", str(result.evidence.unresolved))
                    self.assertEqual({ref.node_id for ref in result.references}, {"b", "c"})
                    self.assertEqual(len(root_requests), 1)
                    self.assertEqual(len(sessions), 3)
                    self.assertEqual(
                        sum(
                            event.get("kind") == "node_result"
                            and event.get("operation") == "source_info"
                            for event in result.trace
                        ),
                        2,
                    )
                    self.assertNotIn("Unneeded long source", str(root_requests))
                    initial_a = next(
                        request
                        for request in child_requests
                        if len(request.messages) == 2
                        and json.loads(request.messages[1].content)["node_id"] == "a"
                    )
                    self.assertIn(
                        "Complete operational journal sentinel", initial_a.messages[1].content
                    )
                    self.assertNotIn("Unneeded long source", initial_a.messages[1].content)
        for session in sessions:
            status, _, _ = await self.docker("inspect", session.container_name)
            self.assertNotEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
