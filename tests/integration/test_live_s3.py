"""Opted-in real S3 publication and Workspace restart contracts.

Set LLGM_TEST_S3=1 and LLGM_TEST_S3_URI=s3://bucket/test-prefix/.
Authentication uses boto3's normal credential chain. The test creates a unique
child prefix and deletes only exact keys it attempted to publish. Versioned
objects are deleted by their exact VersionId; no bucket/prefix listing or broad
delete operation is used. The account needs PutObject, GetObject, DeleteObject,
and (for a versioned bucket) DeleteObjectVersion for this test prefix.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from urllib.parse import urlparse

import pytest

from llgm.core.types import Conversation, NodeRef, SourceSpan
from llgm.memory.workspace import Workspace
from llgm.storage import S3BlobStore

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.s3]


@pytest.fixture
def live_s3_config():
    """Require cloud-write consent and isolate attempted objects beneath a UUID prefix."""
    enabled = os.environ.get("LLGM_TEST_S3", "0")
    if enabled == "0":
        pytest.skip("Set LLGM_TEST_S3=1 to enable real S3 writes and exact-object cleanup")
    if enabled != "1":
        pytest.fail("LLGM_TEST_S3 must be exactly 0 or 1", pytrace=False)
    uri = os.environ.get("LLGM_TEST_S3_URI", "").strip()
    if not uri:
        pytest.fail("Opted-in S3 integration requires LLGM_TEST_S3_URI", pytrace=False)
    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
    ):
        pytest.fail("LLGM_TEST_S3_URI must be s3://bucket/nonempty-test-prefix/", pytrace=False)
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        pytest.fail("Opted-in S3 integration requires the llgm[s3] extra", pytrace=False)
    # UUID isolation makes every attempted key test-owned, including ambiguous
    # upload failures. A configured parent prefix is never itself deleted.
    child_uri = uri.rstrip("/") + "/llgm-integration-" + uuid.uuid4().hex + "/"
    client = boto3.client(
        "s3",
        config=Config(
            connect_timeout=10,
            read_timeout=30,
            retries={"mode": "standard", "total_max_attempts": 1},
        ),
    )
    return child_uri, client


class _TrackedS3BlobStore(S3BlobStore):
    """Instrument publication inputs while all I/O remains the real adapter."""

    def __init__(self, uri, *, client, attempts):
        """Attach an exact-publication ledger to the real S3 adapter."""
        super().__init__(uri, client=client)
        self.attempts = attempts

    def put(self, data):
        """Record the content-addressed key before attempting publication."""
        digest = hashlib.sha256(data).hexdigest()
        self.attempts.append((self.bucket, self._key(digest), digest))
        return super().put(data)


def _cleanup_exact_objects(client, attempts, record):
    """Delete only this test's attempted keys, preserving exact version identities."""
    failures = []
    keys = dict.fromkeys((bucket, key) for bucket, key, _ in attempts)
    deleted = 0
    for bucket, key in keys:
        try:
            try:
                metadata = client.head_object(Bucket=bucket, Key=key)
            except Exception as error:
                code = str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))
                if code in {"404", "NoSuchKey", "NotFound"}:
                    continue
                raise
            arguments = {"Bucket": bucket, "Key": key}
            if metadata.get("VersionId") is not None:
                arguments["VersionId"] = metadata["VersionId"]
            client.delete_object(**arguments)
            deleted += 1
        except Exception as error:
            # Persist no provider bodies, credentials, bucket names or paths.
            failures.append(type(error).__name__)
    record["operations"].append(
        {
            "operation": "cleanup_exact_objects",
            "attempted_keys": len(keys),
            "deleted": deleted,
            "error_types": failures,
        }
    )
    if failures:
        pytest.fail(
            "Exact-object S3 cleanup failed; inspect the isolated llgm-integration prefix. "
            "Error types: " + ", ".join(failures),
            pytrace=False,
        )


def test_real_s3_content_addressing_and_workspace_restart(
    live_s3_config, tmp_path, integration_record
):
    """Real S3 blobs support duplicate publication and historical workspace reads."""
    uri, client = live_s3_config
    record, _ = integration_record
    record["storage_backend"] = "real-s3-conditional-put"
    attempts = []
    store = _TrackedS3BlobStore(uri, client=client, attempts=attempts)

    async def scenario():
        """Publish two immutable sources and resolve each across a SQLite restart."""
        payload = "  e\u0301 é 👩🏽‍💻\r\nS3 immutable source sanity.  ".encode("utf-8")
        digest = await asyncio.to_thread(store.put, payload)
        assert digest == hashlib.sha256(payload).hexdigest()
        duplicate = await asyncio.to_thread(store.put, payload)
        assert duplicate == digest
        assert await asyncio.to_thread(store.get, digest) == payload
        record["operations"].append(
            {
                "operation": "blob_roundtrip_and_duplicate",
                "sha256": digest,
                "bytes": len(payload),
                "puts": 2,
            }
        )
        original = "Production uses cobalt. Staging remains amber."
        changed = "Production now uses jade. Staging remains amber."
        first_conversation = Conversation.from_turns(
            [{"role": "user", "turn_id": "t", "text": original}],
            node_id="deployment",
            metadata={"origin": "s3-integration-supplied-text"},
        )
        async with Workspace.open(tmp_path / "workspace", blob_store=store) as workspace:
            first = await workspace.ingest(first_conversation, idempotency_key="source-1")
            retry = await workspace.ingest(first_conversation, idempotency_key="source-1")
            assert first.created and not retry.created
            assert first.node_id == retry.node_id
            ref = SourceSpan("deployment", "t", 16, 22)
            assert (await workspace.resolve(ref)).text == "cobalt"
            second = await workspace.ingest(
                Conversation.from_turns(
                    [{"role": "user", "turn_id": "t", "text": changed}], node_id="deployment-update"
                ),
                idempotency_key="source-2",
            )
            assert second.node_id != first.node_id
        # New adapter and Workspace objects read S3 and persisted SQLite metadata;
        # no in-memory source cache from the original instance is reused.
        reopened_store = _TrackedS3BlobStore(uri, client=client, attempts=attempts)
        async with Workspace.open(tmp_path / "workspace", blob_store=reopened_store) as reopened:
            assert (await reopened.resolve(ref)).text == "cobalt"
            old = await reopened.resolve(NodeRef(first.node_id))
            current = await reopened.resolve(NodeRef(second.node_id))
            assert old.reference == NodeRef(first.node_id)
            assert current.reference == NodeRef(second.node_id)
            assert original in old.text
            assert changed in current.text
        record["operations"].append(
            {
                "operation": "workspace_restart_with_s3",
                "source_nodes": 2,
                "exact_span": True,
            }
        )

    try:
        asyncio.run(scenario())
    finally:
        try:
            _cleanup_exact_objects(client, attempts, record)
        finally:
            client.close()
