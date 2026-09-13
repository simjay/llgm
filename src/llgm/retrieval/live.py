"""Durable hybrid workspace indexes for the authenticated GPU deployment."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import asdict, replace
from pathlib import Path

from llgm.core.errors import ConfigurationError
from llgm.core.types import SourceNode, Turn
from llgm.retrieval.base import passage_from_dict, passage_to_dict
from llgm.retrieval.bm25 import SQLiteBM25Retriever
from llgm.retrieval.hybrid import HybridRetriever
from llgm.retrieval.workspace import snapshot_identity


class WorkspaceIndexService:
    """Own one warm hybrid index and immutable generations on a persistent volume."""

    def __init__(self, root, config):
        """Retain explicit asset configuration without loading the encoder."""
        self.root, self.config = Path(root), config
        self._lock = threading.RLock()
        self._cached = None
        self._identity = json.loads(json.dumps(asdict(config), default=str))
        self._identity.pop("index_name")

    def _directory(self, index_id):
        """Accept only content identities beneath the workspace index root."""
        if not isinstance(index_id, str) or re.fullmatch("[0-9a-f]{64}", index_id) is None:
            raise ConfigurationError("Workspace index ID must be a SHA256 digest")
        path = (self.root / index_id).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ConfigurationError("Workspace index path escapes its root")
        return path

    def _load(self, index_id, passages):
        """Build or reopen one verified semantic generation and its matching BM25 corpus."""
        if self._cached is not None and self._cached[0] == index_id:
            return self._cached[1]
        if self._cached is not None:
            self._cached[1].lexical.close()
            self._cached = None
        from llgm.retrieval.colbert import ColBERTRetriever
        from llgm.retrieval.colbert_exact import ExactColBERTRetriever

        config = replace(self.config, index_root=self._directory(index_id), index_name="plaid")
        if len(passages) < 64:
            dense = ExactColBERTRetriever(passages, config=config)
        elif (config.index_path / "llgm-manifest.json").is_file():
            dense = ColBERTRetriever.open(passages, config=config)
        else:
            dense = ColBERTRetriever.build(passages, config=config)
        lexical = SQLiteBM25Retriever.from_passages(passages)
        try:
            hybrid = HybridRetriever(lexical, dense)
        except BaseException:
            lexical.close()
            raise
        self._cached = (index_id, hybrid)
        return hybrid

    def prepare(self, records):
        """Persist a source generation only after both real retrieval components are ready."""
        from llgm.retrieval.passages import split_nodes
        from llgm.retrieval.tokenizers import ColBERTTokenizer

        snapshot = snapshot_identity(records)
        identity = {
            "schema_version": 1,
            "snapshot_sha256": snapshot,
            "configuration": self._identity,
        }
        index_id = snapshot_identity(identity)
        with self._lock:
            directory = self._directory(index_id)
            manifest = directory / "workspace.json"
            if manifest.exists():
                saved = self._read(index_id)
                return {key: saved[key] for key in ("index_id", "snapshot_sha256", "descriptor")}
            nodes = [
                SourceNode(
                    record["node_id"],
                    tuple(Turn(**turn) for turn in record["turns"]),
                    record["metadata"],
                    record["timestamp_ms"],
                )
                for record in records
            ]
            tokenizer = ColBERTTokenizer(str(self.config.checkpoint_path))
            passages = split_nodes(nodes, tokenizer, window=self.config.doc_maxlen)
            descriptor = (
                self._load(index_id, passages).descriptor()
                if passages
                else {"backend": "H", "implementation": "equal-weight-rrf", "passage_count": 0}
            )
            payload = [passage_to_dict(passage) for passage in passages]
            saved = {
                **identity,
                "index_id": index_id,
                "descriptor": descriptor,
                "passages": payload,
                "passages_sha256": snapshot_identity(payload),
            }
            directory.mkdir(parents=True, exist_ok=True)
            pending = directory / "workspace.pending.json"
            pending.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
            pending.replace(manifest)
            return {key: saved[key] for key in ("index_id", "snapshot_sha256", "descriptor")}

    def _read(self, index_id):
        """Verify persisted corpus bytes, deployment configuration and generation identity."""
        saved = json.loads(
            (self._directory(index_id) / "workspace.json").read_text(encoding="utf-8")
        )
        identity = {
            key: saved[key] for key in ("schema_version", "snapshot_sha256", "configuration")
        }
        if (
            saved["schema_version"] != 1
            or saved["configuration"] != self._identity
            or snapshot_identity(identity) != index_id
            or saved["index_id"] != index_id
            or snapshot_identity(saved["passages"]) != saved["passages_sha256"]
        ):
            raise ConfigurationError("Workspace index identity or passage checksum does not match")
        return saved

    def search(self, index_id, query, k):
        """Search both matching components and return canonical source spans with fused ranks."""
        if (
            not isinstance(query, str)
            or not query.strip()
            or type(k) is not int
            or not 1 <= k <= 40
        ):
            raise ConfigurationError("Workspace search requires a query and 1 <= k <= 40")
        with self._lock:
            saved = self._read(index_id)
            passages = [passage_from_dict(value) for value in saved["passages"]]
            hits = asyncio.run(self._load(index_id, passages).search(query, k)) if passages else []
            return {
                "index_id": index_id,
                "snapshot_sha256": saved["snapshot_sha256"],
                "hits": [
                    {
                        "passage_id": hit.passage.passage_id,
                        "passage": passage_to_dict(hit.passage),
                        "score": hit.score,
                        "rank": hit.rank,
                    }
                    for hit in hits
                ],
            }
