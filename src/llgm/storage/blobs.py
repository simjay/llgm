"""Content-addressed immutable blobs, independent of logical evidence IDs."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from llgm.core.errors import CapabilityError, ConfigurationError, ReferenceResolutionError


@runtime_checkable
class BlobStore(Protocol):
    """Synchronous I/O contract. Workspace invokes it on its storage worker."""

    def put(self, data: bytes) -> str:
        """Publish immutable bytes and return their lowercase SHA-256 identifier."""
        ...

    def get(self, digest: str) -> bytes:
        """Return stored bytes only after verifying their requested SHA-256 identifier."""
        ...


def _validate_digest(digest: str) -> None:
    """Reject blob identifiers outside the lowercase SHA-256 hexadecimal format."""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ReferenceResolutionError("Invalid SHA-256 blob identifier")


def _verify(data: bytes, digest: str) -> bytes:
    """Return blob bytes only when their content matches the expected digest."""
    if hashlib.sha256(data).hexdigest() != digest:
        raise ReferenceResolutionError(f"Blob checksum mismatch: {digest}")
    return data


class LocalBlobStore:
    """Durable immutable blobs stored in directories partitioned by content hash."""

    def __init__(self, directory: str | Path):
        """Select the local blob root without creating it until publication."""
        self.directory = Path(directory)

    def _path(self, digest: str) -> Path:
        """Validate a digest before deriving its path within the blob root."""
        _validate_digest(digest)
        return self.directory / digest[:2] / digest[2:]

    def put(self, data: bytes) -> str:
        """Publish flushed bytes atomically without replacing an existing content object."""
        digest = hashlib.sha256(data).hexdigest()
        destination = self._path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self.get(digest)
            return digest
        descriptor, temporary = tempfile.mkstemp(prefix=".upload-", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Linking publishes without ever replacing a previously published object.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                self.get(digest)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return digest

    def get(self, digest: str) -> bytes:
        """Read a local blob and reject missing content or checksum mismatches."""
        try:
            data = self._path(digest).read_bytes()
        except FileNotFoundError as exc:
            raise ReferenceResolutionError(f"Missing source blob: {digest}") from exc
        return _verify(data, digest)


class S3BlobStore:
    """Optional S3 adapter. Credentials use the injected client's or SDK's chain."""

    def __init__(self, uri: str, *, client=None):
        """Parse a bucket/prefix location and optionally retain an injected S3 client."""
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
            raise ConfigurationError("S3 blob URI must be s3://bucket/optional-prefix/")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self._client = client

    def _sdk(self):
        """Lazily obtain the injected or optional SDK client for S3 operations."""
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise CapabilityError(
                    "S3 storage requires the llgm[s3] extra or an injected client"
                ) from exc
            self._client = boto3.client("s3")
        return self._client

    def _key(self, digest: str) -> str:
        """Validate a digest before deriving its bucket-prefix object key."""
        _validate_digest(digest)
        return "/".join(part for part in (self.prefix, digest[:2], digest[2:]) if part)

    def put(self, data: bytes) -> str:
        """Conditionally create the content object and verify it before returning its ID."""
        digest = hashlib.sha256(data).hexdigest()
        try:
            self._sdk().put_object(
                Bucket=self.bucket,
                Key=self._key(digest),
                Body=data,
                IfNoneMatch="*",
                Metadata={"sha256": digest},
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if str(code) not in {"PreconditionFailed", "412"}:
                raise
        # Completion and checksum are checked before publishing a metadata reference.
        self.get(digest)
        return digest

    def get(self, digest: str) -> bytes:
        """Read and verify object bytes while always closing the response stream."""
        response = self._sdk().get_object(Bucket=self.bucket, Key=self._key(digest))
        stream = response["Body"]
        try:
            return _verify(stream.read(), digest)
        finally:
            stream.close()
