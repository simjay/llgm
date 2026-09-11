"""Equal-weight RRF: a logical search always accounts for both component searches."""

from __future__ import annotations

import asyncio
import hashlib

from llgm.core.errors import ConfigurationError
from llgm.retrieval.base import SearchHit


class HybridRetriever:
    """Equal-weight reciprocal-rank fusion over matching lexical and dense corpora."""

    def __init__(self, lexical, dense, rank_constant: int = 60, pool_size: int = 40):
        """Validate fusion settings and require both components to index one corpus."""
        if (
            type(rank_constant) is not int
            or rank_constant < 0
            or type(pool_size) is not int
            or pool_size <= 0
        ):
            raise ConfigurationError("Invalid RRF rank constant or pool size")
        left, right = lexical.descriptor(), dense.descriptor()
        if left.get("corpus_sha256") != right.get("corpus_sha256"):
            raise ConfigurationError("Hybrid backends must index identical passages")
        self.lexical, self.dense = lexical, dense
        self.rank_constant, self.pool_size = rank_constant, pool_size
        self.events: list[dict] = []

    async def search(self, query: str, k: int) -> list[SearchHit]:
        """Search both component pools and fuse their ranks into the visible top k."""
        if type(k) is not int or k < 0 or k > self.pool_size:
            raise ConfigurationError(
                "Hybrid visible k must be between zero and component pool size"
            )
        if not k:
            return []
        pending = [
            asyncio.create_task(component.search(query, self.pool_size))
            for component in (self.lexical, self.dense)
        ]
        combined = asyncio.gather(*pending)
        try:
            lists = await asyncio.shield(combined)
        except BaseException:
            # Both searches belong to this call, including a sibling still running
            # after the other fails. Drain cleanup before the caller releases them.
            for task in pending:
                task.cancel()
            cleanup = asyncio.gather(*pending, return_exceptions=True)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            if not combined.cancelled():
                combined.exception()
            raise
        scores, passages = {}, {}
        for hits in lists:
            for rank, hit in enumerate(hits, 1):
                key = hit.passage.passage_id
                passages[key] = hit.passage
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rank_constant + rank)
        ranking = sorted(scores, key=lambda key: (-scores[key], key))[:k]
        self.events.append(
            {
                "operation": "rrf_search",
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "component_pool_size": self.pool_size,
                "component_hit_counts": [len(x) for x in lists],
                "visible_hits": len(ranking),
                "internal_work_charged_separately": True,
            }
        )
        return [SearchHit(passages[key], scores[key], rank) for rank, key in enumerate(ranking, 1)]

    def descriptor(self) -> dict:
        """Describe fusion settings and preserve both component configurations."""
        return {
            "backend": "H",
            "implementation": "equal-weight-rrf",
            "rank_constant": self.rank_constant,
            "pool_size": self.pool_size,
            "corpus_sha256": self.lexical.descriptor()["corpus_sha256"],
            "components": [self.lexical.descriptor(), self.dense.descriptor()],
        }
