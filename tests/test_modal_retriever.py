"""Offline Modal wire-contract tests, independent of GPU retrieval quality."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llgm.core.errors import CapabilityError, ConfigurationError
from llgm.core.types import SourceSpan
from llgm.retrieval.base import SearchPassage, corpus_fingerprint
from llgm.retrieval.modal import ModalColBERTRetriever

INDEX_ID = "a" * 64
CHECKPOINT = "b" * 64
REVISION = "c" * 40


@pytest.fixture
def passages():
    """Expose two distinct canonical sources with mutable nested metadata."""
    return [
        SearchPassage(
            "one", "café recipe", (SourceSpan("node-1", "turn-1", 4, 15),), {"tags": ["food"]}
        ),
        SearchPassage(
            "two", "train ticket", (SourceSpan("node-2", "turn-2", 0, 12),), {"tags": ["travel"]}
        ),
    ]


@pytest.fixture
def metadata(passages):
    """Supply explicit synthetic provenance for wire validation, not model execution."""
    fingerprint = corpus_fingerprint(passages)
    return {
        "schema_version": 1,
        "index_id": INDEX_ID,
        "corpus_fingerprint": fingerprint,
        "descriptor": {
            "backend": "colbertv2_plaid",
            "deployment": "local",
            "corpus_fingerprint": fingerprint,
            "passage_count": 2,
            "configuration": {"checkpoint_sha256": CHECKPOINT, "repository_revision": REVISION},
            "implementation": {
                "repository": "https://github.com/stanford-futuredata/ColBERT",
                "revision": REVISION,
                "verification": "test-double",
            },
            "silent_truncation": False,
            "token_limits_include_special_and_marker_tokens": True,
        },
    }


@pytest.fixture
def response(metadata):
    """Return a ranking whose order differs from the local passage collection."""
    return {
        **{key: value for key, value in metadata.items() if key != "descriptor"},
        "hits": [
            {"passage_id": "two", "score": 8.5, "rank": 1},
            {"passage_id": "one", "score": -2, "rank": 2},
        ],
        "search_seconds": 0.012,
    }


def retriever(passages, metadata, response):
    """Construct a client with a recorded async transport and no Modal dependency."""
    rpc = AsyncMock(return_value=response)
    return ModalColBERTRetriever(
        passages, index_id=INDEX_ID, index_metadata=metadata, search_rpc=rpc
    ), rpc


def test_constructor_and_search_preserve_canonical_evidence(passages, metadata, response):
    """Returned evidence comes from an independent local snapshot, never response text."""
    original = deepcopy(passages)
    response["hits"][0].update(text="remote injection", refs=[], metadata={"tags": ["remote"]})
    with patch(
        "llgm.retrieval.modal.importlib.import_module", side_effect=AssertionError("SDK import")
    ):
        client, rpc = retriever(passages, metadata, response)
    passages[1].metadata["tags"].append("caller change")
    metadata["descriptor"]["configuration"]["checkpoint_sha256"] = "changed"
    hits = asyncio.run(client.search("Where is my train ticket?", 2))
    assert [hit.passage for hit in hits] == original[::-1]
    assert [hit.rank for hit in hits] == [1, 2]
    assert [hit.score for hit in hits] == [8.5, -2.0]
    rpc.assert_awaited_once_with(INDEX_ID, "Where is my train ticket?", 2)
    hits[0].passage.metadata["tags"].append("consumer change")
    assert asyncio.run(client.search("ticket", 2))[0].passage == original[1]
    descriptor = client.descriptor()
    assert descriptor["corpus_fingerprint"] == corpus_fingerprint(original)
    assert descriptor["configuration"]["checkpoint_sha256"] == CHECKPOINT
    descriptor["configuration"].clear()
    assert client.descriptor()["configuration"]["checkpoint_sha256"] == CHECKPOINT


@pytest.mark.parametrize(
    "query,k", [("", 1), (" \n", 1), (None, 1), ("q", 0), ("q", -1), ("q", True), ("q", 1.5)]
)
def test_invalid_search_never_dispatches(passages, metadata, response, query, k):
    """Invalid query and limit values incur no remote calls or search events."""
    client, rpc = retriever(passages, metadata, response)
    with pytest.raises(ConfigurationError):
        asyncio.run(client.search(query, k))
    rpc.assert_not_awaited()
    assert client.events == []


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"index_id": "d" * 64},
        {"corpus_fingerprint": "e" * 64},
    ],
)
def test_index_identity_checked_on_describe_and_search(passages, metadata, response, change):
    """Both connection and every search reject another index or protocol version."""
    with pytest.raises(CapabilityError):
        retriever(passages, {**metadata, **change}, response)
    client, _ = retriever(passages, metadata, {**response, **change})
    with pytest.raises(CapabilityError):
        asyncio.run(client.search("q", 2))
    assert client.events[-1]["status"] == "failed"


@pytest.mark.parametrize("variant", ["reversed", "text", "refs", "metadata", "duplicate", "empty"])
def test_corpus_mismatch_is_rejected(passages, metadata, response, variant):
    """Order, evidence coordinates and metadata are part of the immutable corpus identity."""
    changed = deepcopy(passages)
    if variant == "reversed":
        changed.reverse()
    elif variant == "text":
        changed[0] = SearchPassage("one", "new text", changed[0].refs, changed[0].metadata)
    elif variant == "refs":
        changed[0] = SearchPassage(
            "one",
            changed[0].text,
            (SourceSpan("other-node", "turn-1", 0, 11),),
            changed[0].metadata,
        )
    elif variant == "metadata":
        changed[0].metadata["tags"].append("new tag")
    elif variant == "duplicate":
        changed.append(changed[0])
    else:
        changed.clear()
    with pytest.raises(ConfigurationError):
        retriever(changed, metadata, response)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "bm25"),
        ("passage_count", 3),
        ("passage_count", True),
        ("silent_truncation", True),
        ("token_limits_include_special_and_marker_tokens", False),
        ("corpus_fingerprint", "f" * 64),
        ("configuration", None),
        ("implementation", {}),
    ],
)
def test_incompatible_backend_descriptor_is_rejected(passages, metadata, response, field, value):
    """The deployment cannot substitute a backend, corpus or truncation policy."""
    metadata["descriptor"][field] = value
    with pytest.raises(CapabilityError):
        retriever(passages, metadata, response)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("implementation", "repository", "https://example.com/ColBERT"),
        ("implementation", "revision", "f" * 40),
        ("configuration", "checkpoint_sha256", "unpinned"),
        ("configuration", "repository_revision", "main"),
    ],
)
def test_unpinned_or_inconsistent_provenance_is_rejected(
    passages, metadata, response, section, field, value
):
    """Remote repository and weight identities must be complete and internally consistent."""
    metadata["descriptor"][section][field] = value
    with pytest.raises(CapabilityError):
        retriever(passages, metadata, response)


def test_caller_expected_pins_are_enforced(passages, metadata, response):
    """A valid remote descriptor still must match explicitly requested artifact pins."""
    kwargs = dict(
        index_id=INDEX_ID, index_metadata=metadata, search_rpc=AsyncMock(return_value=response)
    )
    ModalColBERTRetriever(
        passages,
        **kwargs,
        expected_checkpoint_sha256=CHECKPOINT,
        expected_repository_revision=REVISION,
    )
    for expected in (
        {"expected_checkpoint_sha256": "f" * 64},
        {"expected_repository_revision": "f" * 40},
    ):
        with pytest.raises(ConfigurationError, match="expected"):
            ModalColBERTRetriever(passages, **kwargs, **expected)


@pytest.mark.parametrize(
    "bad_hit",
    [
        None,
        {"passage_id": "absent", "score": 3, "rank": 1},
        {"passage_id": ["two"], "score": 3, "rank": 1},
        {"passage_id": "two", "score": True, "rank": 1},
        {"passage_id": "two", "score": "3.0", "rank": 1},
        {"passage_id": "two", "score": float("inf"), "rank": 1},
        {"passage_id": "two", "score": float("nan"), "rank": 1},
        {"passage_id": "two", "score": 3, "rank": 0},
        {"passage_id": "two", "score": 3, "rank": True},
        {"passage_id": "two", "score": 3, "rank": 2},
    ],
)
def test_invalid_hit_is_rejected_without_partial_results(passages, metadata, response, bad_hit):
    """Unknown identities, nonfinite scores and invalid ranks cannot become evidence."""
    response["hits"][0] = bad_hit
    client, rpc = retriever(passages, metadata, response)
    with pytest.raises(CapabilityError):
        asyncio.run(client.search("q", 2))
    rpc.assert_awaited_once()
    assert client.events[-1]["returned"] == 0


def test_duplicate_and_excess_hits_are_rejected(passages, metadata, response):
    """Duplicate passage IDs and responses exceeding requested k fail explicitly."""
    client, _ = retriever(passages, metadata, response)
    with pytest.raises(CapabilityError, match="count"):
        asyncio.run(client.search("q", 1))
    response["hits"][1]["passage_id"] = "two"
    with pytest.raises(CapabilityError, match="passage IDs"):
        asyncio.run(client.search("q", 2))


@pytest.mark.parametrize("timing", [None, True, -1, "0.1", float("inf"), float("nan")])
def test_invalid_server_time_is_rejected(passages, metadata, response, timing):
    """Missing, nonfinite or negative server timing is not silently counted as zero."""
    response["search_seconds"] = timing
    client, _ = retriever(passages, metadata, response)
    with pytest.raises(CapabilityError, match="timing"):
        asyncio.run(client.search("q", 2))
    assert client.descriptor()["server_search_seconds"] is None


def test_timings_and_unknown_billing_remain_separate(passages, metadata, response):
    """Client duration is measured independently from reported kernel work and unknown cost."""
    metadata["descriptor"].update(search_seconds=999.0, search_calls=100)
    client, _ = retriever(passages, metadata, response)
    with patch("llgm.retrieval.modal.time.perf_counter", side_effect=[10.0, 12.0]):
        asyncio.run(client.search("private query", 2))
    event = client.events[-1]
    assert event["client_elapsed_seconds"] == 2.0
    assert event["server_search_seconds"] == 0.012
    assert "private query" not in str(event)
    assert event["cost_usd"] is None
    descriptor = client.descriptor()
    assert descriptor["deployment"] == "modal"
    assert descriptor["search_calls"] == 1
    assert "search_seconds" not in descriptor
    assert descriptor["client_search_seconds"] == 2.0
    assert descriptor["server_search_seconds"] == 0.012
    assert descriptor["cost_usd"] is None


def test_transport_failure_and_cancellation_propagate_without_retry(passages, metadata, response):
    """Failed or cancelled calls preserve their exceptions and never retry or fall back."""
    for failure, status in [
        (RuntimeError("connection lost"), "failed"),
        (asyncio.CancelledError(), "cancelled"),
    ]:
        client, rpc = retriever(passages, metadata, response)
        rpc.side_effect = failure
        with pytest.raises(type(failure)):
            asyncio.run(client.search("q", 2))
        rpc.assert_awaited_once()
        assert client.events[-1]["status"] == status
        assert client.events[-1]["server_search_seconds"] is None


def test_connect_uses_async_modal_functions_without_uploading_corpus(passages, metadata, response):
    """Connect and search dispatch only index identity, query and result limit over async RPC."""
    describe = AsyncMock(return_value=metadata)
    search = AsyncMock(return_value=response)
    from_name = Mock(
        side_effect=[
            SimpleNamespace(remote=SimpleNamespace(aio=describe)),
            SimpleNamespace(remote=SimpleNamespace(aio=search)),
        ]
    )
    sdk = SimpleNamespace(Function=SimpleNamespace(from_name=from_name))
    with patch("llgm.retrieval.modal.importlib.import_module", return_value=sdk):
        client = asyncio.run(
            ModalColBERTRetriever.connect(
                passages, index_id=INDEX_ID, app_name="custom-app", environment="development"
            )
        )
        assert describe.await_args.args == (INDEX_ID,)
        search.assert_not_awaited()
        asyncio.run(client.search("q", 2))
    assert [call.args for call in from_name.call_args_list] == [
        ("custom-app", "describe_index"),
        ("custom-app", "search_index"),
    ]
    assert all(
        call.kwargs == {"environment_name": "development"} for call in from_name.call_args_list
    )
    search.assert_awaited_once_with(INDEX_ID, "q", 2)
    assert client.descriptor()["connect_elapsed_seconds"] >= 0


def test_missing_sdk_explains_optional_capability(passages):
    """The absent optional SDK fails before any remote request is possible."""
    with patch("llgm.retrieval.modal.importlib.import_module", side_effect=ImportError("no modal")):
        with pytest.raises(CapabilityError, match=r"llgm\[modal\]"):
            asyncio.run(ModalColBERTRetriever.connect(passages, index_id=INDEX_ID))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index_id": "../index"},
        {"index_id": "A" * 64},
        {"app_name": ""},
        {"environment": ""},
        {"expected_checkpoint_sha256": "bad"},
        {"expected_repository_revision": "main"},
    ],
)
def test_invalid_connection_settings_fail_before_sdk_import(passages, kwargs):
    """Malformed deployment identities and expected pins are rejected locally."""
    with patch(
        "llgm.retrieval.modal.importlib.import_module", side_effect=AssertionError("SDK import")
    ):
        with pytest.raises(ConfigurationError):
            asyncio.run(ModalColBERTRetriever.connect(passages, **{"index_id": INDEX_ID, **kwargs}))
