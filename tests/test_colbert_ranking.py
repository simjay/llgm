"""Local ColBERT ranking admission and canonical corpus ownership."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llgm.core.errors import CapabilityError
from llgm.core.types import SourceSpan
from llgm.retrieval import SearchPassage
from llgm.retrieval.colbert import ColBERTConfig, ColBERTRetriever


@pytest.fixture
def local_index(tmp_path):
    """Bind a fixed official-search boundary without importing encoders or loading weights."""
    passages = [
        SearchPassage(name, name, (SourceSpan(name, "turn", 0, len(name)),), {"tags": [name]})
        for name in ("one", "two")
    ]
    searcher = SimpleNamespace(
        checkpoint=SimpleNamespace(
            query_tokenizer=SimpleNamespace(tok=lambda *args, **kwargs: {"input_ids": [1, 2, 3]})
        ),
        search=Mock(return_value=([1, 0], [1, 2], [8.5, -2])),
    )
    retriever = ColBERTRetriever(
        config=ColBERTConfig(
            checkpoint_path=tmp_path,
            checkpoint_sha256="a" * 64,
            repository_revision="b" * 40,
            index_root=tmp_path,
            index_name="test",
        ),
        passages=passages,
        tokenizer=SimpleNamespace(count=lambda *args, **kwargs: 4),
        searcher=searcher,
        repository={"revision": "b" * 40},
    )
    return retriever, passages, searcher


def test_local_results_preserve_canonical_corpus_and_derive_accounting(local_index):
    """Caller and consumer mutations cannot change later hits or the index fingerprint."""
    retriever, passages, searcher = local_index
    original = deepcopy(passages)
    fingerprint = retriever.descriptor()["corpus_fingerprint"]
    passages[1].metadata["tags"].append("caller edit")
    retriever.passages[1].metadata["tags"].append("view edit")
    hits = asyncio.run(retriever.search("question", 2))
    assert [hit.passage for hit in hits] == original[::-1]
    assert [hit.score for hit in hits] == [8.5, -2.0]
    hits[0].passage.metadata["tags"].append("consumer edit")
    assert asyncio.run(retriever.search("question", 2))[0].passage == original[1]
    assert retriever.descriptor()["corpus_fingerprint"] == fingerprint
    assert retriever.descriptor()["search_calls"] == searcher.search.call_count == 2
    assert retriever.descriptor()["search_seconds"] == sum(
        event["elapsed_seconds"] for event in retriever.events
    )


@pytest.mark.parametrize(
    "ranking,k",
    [
        (([0, 1], [1], [1.0, 0.5]), 2),
        (([0], [1], []), 2),
        (([0, 0], [1, 2], [1.0, 0.5]), 2),
        (([0, 1], [1, 2], [1.0, 0.5]), 1),
        (([0.5], [1], [1.0]), 2),
        (([True], [1], [1.0]), 2),
        ((["0"], [1], [1.0]), 2),
        (([2], [1], [1.0]), 2),
        (([-1], [1], [1.0]), 2),
        (([0], [2], [1.0]), 2),
        (([0], [True], [1.0]), 2),
        (([0], [1], [True]), 2),
        (([0], [1], ["1.0"]), 2),
        (([0], [1], [float("nan")]), 2),
        (([0], [1], [float("inf")]), 2),
    ],
)
def test_local_invalid_ranking_cannot_return_partial_evidence(local_index, ranking, k):
    """Malformed ranks, coerced identities, duplicates and overflow fail as complete results."""
    retriever, _, searcher = local_index
    searcher.search.return_value = ranking
    with pytest.raises(CapabilityError):
        asyncio.run(retriever.search("question", k))
    assert retriever.events[-1]["status"] == "failed"
    assert retriever.events[-1]["returned"] == 0
    assert retriever.descriptor()["search_calls"] == 1


def test_upstream_numpy_scalars_remain_supported(local_index):
    """Native encoder scalar types keep their integral identities and finite scores."""
    np = pytest.importorskip("numpy")
    retriever, _, searcher = local_index
    searcher.search.return_value = ([np.int64(0)], [np.int64(1)], [np.float32(0.5)])
    hits = asyncio.run(retriever.search("question", 1))
    assert hits[0].passage.passage_id == "one" and hits[0].score == 0.5
