"""Persistence adapters. Blob locations are never canonical evidence identities."""

from llgm.storage.blobs import BlobStore, LocalBlobStore, S3BlobStore

__all__ = ["BlobStore", "LocalBlobStore", "S3BlobStore"]
