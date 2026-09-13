"""Exact official ColBERT scoring for corpora too small to train a PLAID index."""

from __future__ import annotations

import asyncio
import importlib

from llgm.core.errors import ConfigurationError
from llgm.retrieval._colbert import snapshot_passages
from llgm.retrieval.base import SearchHit, corpus_fingerprint
from llgm.retrieval.colbert import _OFFICIAL_LOCK, _check_repository_revision, _checkpoint, _offline
from llgm.retrieval.tokenizers import ColBERTTokenizer


class ExactColBERTRetriever:
    """Use the pinned encoder and uncompressed MaxSim for fewer than 64 passages."""

    def __init__(self, passages, *, config):
        """Verify model provenance and encode each small-corpus document once."""
        self.passages = snapshot_passages(passages)
        if len(self.passages) >= 64:
            raise ConfigurationError("Exact workspace ColBERT is limited to fewer than 64 passages")
        self.config = config
        path = _checkpoint(config)
        with _OFFICIAL_LOCK, _offline():
            self.repository = _check_repository_revision(config.repository_revision)
            self.tokenizer = ColBERTTokenizer(str(path))
            for passage in self.passages:
                if self.tokenizer.count(passage.text) > config.doc_maxlen:
                    raise ConfigurationError("ColBERT passage exceeds the document token limit")
            infra = importlib.import_module("colbert.infra")
            checkpoint = importlib.import_module("colbert.modeling.checkpoint")
            self.encoder = checkpoint.Checkpoint(
                str(path),
                colbert_config=infra.ColBERTConfig(
                    checkpoint=str(path),
                    doc_maxlen=config.doc_maxlen,
                    query_maxlen=config.query_maxlen,
                    gpus=config.gpus,
                ),
            )
            if config.gpus:
                self.encoder = self.encoder.cuda()
            embeddings, lengths = self.encoder.docFromText(
                [p.text for p in self.passages],
                bsize=config.index_bsize,
                keep_dims="flatten",
                to_cpu=True,
            )
            self.documents = embeddings.float().split(lengths)

    async def search(self, query, k):
        """Rank every encoded document by the sum of per-query-token maximum similarities."""
        return await asyncio.to_thread(self._search, query, k)

    def _search(self, query, k):
        """Serialize encoder access and reject queries that would be silently truncated."""
        if not isinstance(query, str) or not query.strip() or type(k) is not int or k < 0:
            raise ConfigurationError("ColBERT requires a nonempty query and nonnegative k")
        with _OFFICIAL_LOCK, _offline():
            if self.tokenizer.count(query, kind="query") > self.config.query_maxlen:
                raise ConfigurationError("ColBERT query exceeds the query token limit")
            query_vectors = self.encoder.queryFromText([query], bsize=1, to_cpu=True)[0].float()
            scores = [
                (document @ query_vectors.T).max(dim=0).values.sum().item()
                for document in self.documents
            ]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.passages[i].passage_id))[
            :k
        ]
        return [SearchHit(self.passages[i], scores[i], rank) for rank, i in enumerate(order, 1)]

    def descriptor(self):
        """Distinguish exact semantic ranking from compressed PLAID retrieval."""
        return {
            "backend": "colbertv2_exact",
            "implementation": self.repository,
            "corpus_sha256": corpus_fingerprint(self.passages),
            "passage_count": len(self.passages),
            "checkpoint_sha256": self.config.checkpoint_sha256,
            "silent_truncation": False,
        }
