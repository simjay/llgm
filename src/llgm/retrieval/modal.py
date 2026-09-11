"""Authenticated Modal calls to a pinned official ColBERTv2/PLAID index.

Import and construction perform no network access. ``connect`` describes an
existing deployment without uploading passages or building an index. Search
responses carry only passage identities, scores and timing. Canonical text,
references and metadata always come from the caller's verified corpus.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import math
import re
import time
from copy import deepcopy
from typing import Any, Awaitable, Callable, Iterable

from llgm.core.errors import CapabilityError, ConfigurationError
from llgm.retrieval.base import SearchHit, SearchPassage, check_passages, corpus_fingerprint

_OFFICIAL_REPOSITORY = "https://github.com/stanford-futuredata/ColBERT"
SearchRPC = Callable[[str, str, int], Awaitable[dict[str, Any]]]


def _digest(value: Any, length: int) -> bool:
    """Recognize a lowercase hexadecimal digest without coercing other values."""
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _target(
    index_id: str,
    app_name: str,
    environment: str | None,
    expected_checkpoint_sha256: str | None,
    expected_repository_revision: str | None,
) -> None:
    """Reject malformed deployment names and pins before any remote lookup."""
    if not _digest(index_id, 64):
        raise ConfigurationError("Modal ColBERT index_id must be a lowercase SHA256 digest")
    if not isinstance(app_name, str) or not app_name.strip():
        raise ConfigurationError("Modal app_name must be nonempty text")
    if environment is not None and (not isinstance(environment, str) or not environment.strip()):
        raise ConfigurationError("Modal environment must be nonempty text when supplied")
    for name, value, length in (
        ("checkpoint SHA256", expected_checkpoint_sha256, 64),
        ("repository revision", expected_repository_revision, 40),
    ):
        if value is not None and not _digest(value, length):
            raise ConfigurationError(
                f"Expected ColBERT {name} must be {length} lowercase hex digits"
            )


def _snapshot(passages: Iterable[SearchPassage]) -> tuple[SearchPassage, ...]:
    """Own passage metadata independently of the caller's mutable dictionaries."""
    result = tuple(deepcopy(tuple(passages)))
    if not result or any(
        not isinstance(passage, SearchPassage)
        or not isinstance(passage.passage_id, str)
        or not passage.passage_id
        for passage in result
    ):
        raise ConfigurationError("Modal ColBERT requires a nonempty corpus with text passage IDs")
    check_passages(result)
    try:
        corpus_fingerprint(result)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "Modal ColBERT passages must have JSON-serializable metadata"
        ) from exc
    return result


def _number(value: Any) -> bool:
    """Accept finite JSON numbers while excluding booleans and numeric strings."""
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class ModalColBERTRetriever:
    """A remote index bound to an immutable local corpus and explicit provenance.

    Use ``connect`` for an existing Modal deployment. Direct construction accepts
    a describe response and an async search callable for other authenticated
    transports and deterministic contract tests. Neither path retries or falls
    back to another retriever. Cancellation propagates to the caller, but does
    not establish that an already dispatched remote computation stopped.
    """

    def __init__(
        self,
        passages: Iterable[SearchPassage],
        *,
        index_id: str,
        index_metadata: dict[str, Any],
        search_rpc: SearchRPC,
        app_name: str = "llgm-colbert",
        environment: str | None = None,
        expected_checkpoint_sha256: str | None = None,
        expected_repository_revision: str | None = None,
    ) -> None:
        """Validate a describe response and retain independent canonical passages."""
        _target(
            index_id,
            app_name,
            environment,
            expected_checkpoint_sha256,
            expected_repository_revision,
        )
        if not callable(search_rpc):
            raise ConfigurationError("Modal ColBERT search_rpc must be an async callable")
        passages = _snapshot(passages)
        self._passages = {passage.passage_id: passage for passage in passages}
        self._fingerprint = corpus_fingerprint(passages)
        self._index_id = index_id
        self._identity(index_metadata)
        descriptor = index_metadata.get("descriptor")
        if not isinstance(descriptor, dict):
            raise CapabilityError("Modal ColBERT describe response is missing its descriptor")
        configuration = descriptor.get("configuration")
        implementation = descriptor.get("implementation")
        if not isinstance(configuration, dict) or not isinstance(implementation, dict):
            raise CapabilityError(
                "Modal ColBERT descriptor is missing configuration or implementation"
            )
        checkpoint = configuration.get("checkpoint_sha256")
        revision = configuration.get("repository_revision")
        if (
            descriptor.get("backend") != "colbertv2_plaid"
            or descriptor.get("corpus_fingerprint") != self._fingerprint
            or type(descriptor.get("passage_count")) is not int
            or descriptor["passage_count"] != len(passages)
            or implementation.get("repository") != _OFFICIAL_REPOSITORY
            or not _digest(checkpoint, 64)
            or not _digest(revision, 40)
            or implementation.get("revision") != revision
            or descriptor.get("silent_truncation") is not False
            or descriptor.get("token_limits_include_special_and_marker_tokens") is not True
        ):
            raise CapabilityError(
                "Modal ColBERT descriptor has incompatible corpus, backend or pins"
            )
        if expected_checkpoint_sha256 is not None and checkpoint != expected_checkpoint_sha256:
            raise ConfigurationError("Modal ColBERT checkpoint differs from the expected SHA256")
        if expected_repository_revision is not None and revision != expected_repository_revision:
            raise ConfigurationError(
                "Modal ColBERT implementation differs from the expected revision"
            )
        self._remote_descriptor = deepcopy(descriptor)
        self._search_rpc = search_rpc
        self._app_name = app_name
        self._environment = environment
        self._connect_seconds: float | None = None
        self.events: list[dict[str, Any]] = []

    @classmethod
    async def connect(
        cls,
        passages: Iterable[SearchPassage],
        *,
        index_id: str,
        app_name: str = "llgm-colbert",
        environment: str | None = None,
        expected_checkpoint_sha256: str | None = None,
        expected_repository_revision: str | None = None,
    ) -> ModalColBERTRetriever:
        """Describe and validate a deployed index using the optional Modal SDK.

        Reads credentials through Modal's configured authentication. Only the
        content-addressed index ID is sent while connecting. The local corpus
        must exactly match the indexed passage order, text, references and metadata.
        """
        _target(
            index_id,
            app_name,
            environment,
            expected_checkpoint_sha256,
            expected_repository_revision,
        )
        passages = _snapshot(passages)
        try:
            modal = importlib.import_module("modal")
        except ImportError as exc:
            raise CapabilityError(
                "Install llgm[modal] to connect to a Modal ColBERT deployment"
            ) from exc
        started = time.perf_counter()
        describe = modal.Function.from_name(
            app_name, "describe_index", environment_name=environment
        )
        search = modal.Function.from_name(app_name, "search_index", environment_name=environment)
        metadata = await describe.remote.aio(index_id)
        result = cls(
            passages,
            index_id=index_id,
            index_metadata=metadata,
            search_rpc=search.remote.aio,
            app_name=app_name,
            environment=environment,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_repository_revision=expected_repository_revision,
        )
        result._connect_seconds = time.perf_counter() - started
        return result

    def _identity(self, response: Any) -> None:
        """Require protocol and index/corpus identity on every remote response."""
        if (
            not isinstance(response, dict)
            or type(response.get("schema_version")) is not int
            or response["schema_version"] != 1
            or response.get("index_id") != self._index_id
            or response.get("corpus_fingerprint") != self._fingerprint
        ):
            raise CapabilityError(
                "Modal ColBERT response has an incompatible schema, index or corpus"
            )

    async def search(self, query: str, k: int = 10) -> list[SearchHit]:
        """Validate remote ranks and scores before returning local canonical evidence."""
        if not isinstance(query, str) or not query.strip():
            raise ConfigurationError("Modal ColBERT query must be nonempty text")
        if type(k) is not int or k < 1:
            raise ConfigurationError("Modal ColBERT k must be a positive integer")
        started = time.perf_counter()
        event: dict[str, Any] = {
            "operation": "modal_colbert_plaid_search",
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "requested_k": k,
            "returned": 0,
            "status": "failed",
            "server_search_seconds": None,
            "cost_usd": None,
        }
        try:
            response = await self._search_rpc(self._index_id, query, k)
            self._identity(response)
            elapsed = response.get("search_seconds")
            if not _number(elapsed) or elapsed < 0:
                raise CapabilityError("Modal ColBERT response has invalid server search timing")
            event["server_search_seconds"] = float(elapsed)
            values = response.get("hits")
            if not isinstance(values, list) or len(values) > min(k, len(self._passages)):
                raise CapabilityError("Modal ColBERT response has an invalid hit count")
            hits: list[SearchHit] = []
            seen: set[str] = set()
            for rank, value in enumerate(values, 1):
                if not isinstance(value, dict):
                    raise CapabilityError("Modal ColBERT response contains a malformed hit")
                passage_id = value.get("passage_id")
                score = value.get("score")
                if (
                    not isinstance(passage_id, str)
                    or passage_id not in self._passages
                    or passage_id in seen
                    or type(value.get("rank")) is not int
                    or value["rank"] != rank
                    or not _number(score)
                ):
                    raise CapabilityError(
                        "Modal ColBERT response contains invalid passage IDs, ranks or scores"
                    )
                seen.add(passage_id)
                hits.append(SearchHit(deepcopy(self._passages[passage_id]), float(score), rank))
            event.update(returned=len(hits), status="succeeded")
            return hits
        except asyncio.CancelledError:
            event["status"] = "cancelled"
            raise
        finally:
            event["client_elapsed_seconds"] = time.perf_counter() - started
            self.events.append(event)

    def descriptor(self) -> dict[str, Any]:
        """Report pinned backend identity and separate client/server search measurements.

        Client elapsed time includes dispatch, startup and response validation.
        It is not network latency alone. Server time is reported only when a
        valid response supplied it. Neither value establishes billable usage.
        """
        remote = deepcopy(self._remote_descriptor)
        # A describe response may include work from other clients on that server.
        remote.pop("search_seconds", None)
        return {
            **remote,
            "deployment": "modal",
            "index_id": self._index_id,
            "transport": {"app_name": self._app_name, "environment": self._environment},
            "connect_elapsed_seconds": self._connect_seconds,
            "search_calls": len(self.events),
            "client_search_seconds": sum(event["client_elapsed_seconds"] for event in self.events),
            "server_search_seconds": sum(
                event["server_search_seconds"]
                for event in self.events
                if event["server_search_seconds"] is not None
            )
            if any(event["server_search_seconds"] is not None for event in self.events)
            else None,
            "cost_usd": None,
        }
