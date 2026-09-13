"""Canonical corpus and ranking checks shared by local and remote ColBERT."""

import math
from copy import deepcopy
from numbers import Integral, Real

from llgm.core.errors import CapabilityError, ConfigurationError
from llgm.retrieval.base import SearchHit, SearchPassage, check_passages, corpus_fingerprint

OFFICIAL_REPOSITORY = "https://github.com/stanford-futuredata/ColBERT"


def snapshot_passages(passages):
    """Retain a nonempty canonical corpus independently of caller-owned metadata."""
    result = deepcopy(tuple(passages))
    if not result or any(
        not isinstance(passage, SearchPassage)
        or not isinstance(passage.passage_id, str)
        or not passage.passage_id
        for passage in result
    ):
        raise ConfigurationError("ColBERT requires a nonempty corpus with text passage IDs")
    check_passages(result)
    try:
        corpus_fingerprint(result)
    except (TypeError, ValueError) as error:
        raise ConfigurationError("ColBERT passages must have JSON-serializable metadata") from error
    return result


def finite_number(value):
    """Accept finite numeric scores and timings, including upstream NumPy scalars."""
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def ranked_hits(values, passages, k):
    """Resolve a complete valid ranking against canonical local evidence.

    Local ColBERT uses integer passage offsets. Remote results use passage IDs.
    Both must be unique, have sequential ranks and fit the requested hit count.
    Returned metadata is copied so consumers cannot change the indexed corpus.
    """
    if not isinstance(values, list) or len(values) > min(k, len(passages)):
        raise CapabilityError("ColBERT response has an invalid hit count")
    hits, seen = [], set()
    for rank, value in enumerate(values, 1):
        if not isinstance(value, dict):
            raise CapabilityError("ColBERT response contains a malformed hit")
        passage_id, score = value.get("passage_id"), value.get("score")
        declared_rank = value.get("rank")
        if (
            not isinstance(passage_id, (str, Integral))
            or isinstance(passage_id, bool)
            or passage_id not in passages
            or passage_id in seen
            or not isinstance(declared_rank, Integral)
            or isinstance(declared_rank, bool)
            or declared_rank != rank
            or not finite_number(score)
        ):
            raise CapabilityError("ColBERT response contains invalid passage IDs, ranks or scores")
        seen.add(passage_id)
        hits.append(SearchHit(deepcopy(passages[passage_id]), float(score), rank))
    return hits
