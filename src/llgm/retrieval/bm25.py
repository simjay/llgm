"""SQLite FTS5 BM25 with a pinned tokenizer and literal term escaping."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Sequence

from llgm.core.errors import CapabilityError, ConfigurationError
from llgm.retrieval.base import (
    SearchHit,
    SearchPassage,
    check_passages,
    corpus_fingerprint,
    passage_from_dict,
    passage_to_dict,
)


class SQLiteBM25Retriever:
    """FTS5 lexical retrieval over a corpus-checked SQLite passage index."""

    TOKENIZER = "unicode61 remove_diacritics 2"

    def __init__(self, connection: sqlite3.Connection, fingerprint: str, count: int):
        """Bind an open index and serialize access to its shared connection."""
        self.connection, self.fingerprint, self.count = connection, fingerprint, count
        self.events: list[dict] = []
        self._lock = threading.Lock()

    @classmethod
    def from_passages(cls, passages: Sequence[SearchPassage], path: str | Path = ":memory:"):
        """Create an FTS5 index or reuse one only when its corpus fingerprint matches."""
        check_passages(passages)
        connection = sqlite3.connect(str(path), check_same_thread=False)
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(text, payload UNINDEXED, tokenize='unicode61 remove_diacritics 2')"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS index_info (fingerprint TEXT PRIMARY KEY, count INTEGER)"
            )
            fingerprint = corpus_fingerprint(passages)
            existing = connection.execute("SELECT fingerprint, count FROM index_info").fetchone()
            if existing and existing[0] != fingerprint:
                raise ConfigurationError(
                    "Existing BM25 index belongs to a different corpus; choose a new per-case path"
                )
            if not existing:
                connection.executemany(
                    "INSERT INTO passages(text,payload) VALUES(?,?)",
                    [
                        (p.text, json.dumps(passage_to_dict(p), ensure_ascii=False))
                        for p in passages
                    ],
                )
                connection.execute(
                    "INSERT INTO index_info VALUES(?,?)", (fingerprint, len(passages))
                )
                connection.commit()
            return cls(connection, fingerprint, len(passages))
        except sqlite3.OperationalError as exc:
            connection.close()
            raise CapabilityError(
                "SQLite FTS5 is required for BM25; no fallback is enabled"
            ) from exc
        except Exception:
            connection.close()
            raise

    @staticmethod
    def escaped_query(query: str) -> str:
        """Join literal Unicode terms with OR without accepting caller FTS syntax."""
        terms = list(dict.fromkeys(re.findall(r"\w+", query, flags=re.UNICODE)))
        return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)

    async def search(self, query: str, k: int) -> list[SearchHit]:
        """Run lexical search on a worker thread after validating the result limit."""
        if type(k) is not int or k < 0:
            raise ConfigurationError("k must be a non-negative integer")
        return await asyncio.to_thread(self._search, query, k)

    def _search(self, query: str, k: int) -> list[SearchHit]:
        """Serialize a search against the shared SQLite connection."""
        with self._lock:
            return self._search_locked(query, k)

    def _search_locked(self, query: str, k: int) -> list[SearchHit]:
        """Execute BM25 ranking and record query work without retaining query text."""
        start = time.perf_counter()
        escaped = self.escaped_query(query)
        rows = (
            []
            if not escaped or not k
            else self.connection.execute(
                "SELECT payload, bm25(passages) AS score FROM passages WHERE passages MATCH ? ORDER BY score ASC, rowid ASC LIMIT ?",
                (escaped, k),
            ).fetchall()
        )
        self.events.append(
            {
                "operation": "bm25_search",
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "requested_k": k,
                "returned": len(rows),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        return [
            SearchHit(passage_from_dict(json.loads(payload)), -float(score), rank)
            for rank, (payload, score) in enumerate(rows, 1)
        ]

    def descriptor(self) -> dict:
        """Report SQLite, tokenizer, query policy, and indexed corpus provenance."""
        return {
            "backend": "B",
            "implementation": "sqlite-fts5-bm25",
            "sqlite_version": sqlite3.sqlite_version,
            "tokenizer": self.TOKENIZER,
            "query_policy": "literal-unicode-terms-or",
            "corpus_sha256": self.fingerprint,
            "passage_count": self.count,
        }

    def close(self) -> None:
        """Close the SQLite connection after any active locked search completes."""
        with self._lock:
            self.connection.close()
