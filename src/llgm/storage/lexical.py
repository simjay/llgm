"""Reusable SQLite FTS5 passages refreshed from private append progress.

This disposable search projection contains published source spans and inline
journal history. The workspace retains the canonical records.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from llgm.core.errors import ConfigurationError
from llgm.core.types import JournalEntry, JournalRef, SourceNode, SourceSpan, reference_to_dict
from llgm.retrieval.bm25 import SQLiteBM25Retriever
from llgm.storage._sqlite import enable_wal

_INDEX_VERSION = 2
_TOKENIZER = "unicode61 remove_diacritics 2"


def _json(value: Any) -> str:
    """Encode derived references and metadata independently of caller mappings."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class LexicalIndex:
    """A workspace-owned, rebuildable FTS5 index shared by evidence handles."""

    def __init__(self, path: str | Path, workspace_id: str, passage_chars: int):
        """Open one width-specific projection and initialize its disposable schema."""
        self.path = str(path)
        self.passage_chars = passage_chars
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        try:
            enable_wal(self._connection)
            self._connection.execute("BEGIN IMMEDIATE")
            expected = {
                "schema_version": _INDEX_VERSION,
                "workspace_id": workspace_id,
                "passage_chars": passage_chars,
                "tokenizer": _TOKENIZER,
            }
            existing = self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='index_info'"
            ).fetchone()
            info = (
                self._connection.execute("SELECT identity FROM index_info").fetchone()
                if existing
                else None
            )
            if info is None or json.loads(info[0]) != expected:
                for table in ("terms", "passages", "index_info"):
                    self._connection.execute(f"DROP TABLE IF EXISTS {table}")
                self._connection.execute(
                    "CREATE TABLE index_info(identity TEXT NOT NULL, through_sequence INTEGER NOT NULL)"
                )
                self._connection.execute("INSERT INTO index_info VALUES (?,0)", (_json(expected),))
                self._connection.execute(f"""CREATE VIRTUAL TABLE passages USING fts5(
                    text, passage_id UNINDEXED, kind UNINDEXED, reference UNINDEXED, metadata UNINDEXED,
                    tokenize='{_TOKENIZER}')""")
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.close()
            raise

    def _ensure_open(self) -> None:
        """Reject use after the owning workspace has closed the projection."""
        if self._closed:
            raise ConfigurationError("Workspace lexical index is closed")

    def descriptor(self) -> dict[str, Any]:
        """Identify the reusable FTS5 index independently of benchmark passage construction."""
        return {
            "implementation": "sqlite-fts5-incremental",
            "index_schema_version": _INDEX_VERSION,
            "sqlite_version": sqlite3.sqlite_version,
            "tokenizer": _TOKENIZER,
            "passage_chars": self.passage_chars,
            "query_policy": "literal-unicode-terms-or",
            "read_policy": "current",
            "tie_break": "passage-insertion-order",
        }

    def _add(self, text: str, reference, metadata: dict[str, Any], kind: str) -> None:
        """Store one canonical passage and independent attribution metadata."""
        encoded = _json(reference_to_dict(reference))
        self._connection.execute(
            "INSERT INTO passages VALUES (?,?,?,?,?)",
            (text, hashlib.sha256(encoded.encode()).hexdigest(), kind, encoded, _json(metadata)),
        )

    def refresh(self, through_sequence: int, records) -> dict[str, int]:
        """Atomically add unseen source/journal records and advance private indexing progress."""
        with self._lock:
            self._ensure_open()
            connection = self._connection
            through = connection.execute("SELECT through_sequence FROM index_info").fetchone()[0]
            stats = {
                "source_records_indexed": 0,
                "journal_records_indexed": 0,
                "passages_indexed": 0,
            }
            if through_sequence <= through:
                return stats
            connection.execute("BEGIN IMMEDIATE")
            try:
                through = connection.execute("SELECT through_sequence FROM index_info").fetchone()[
                    0
                ]
                for record in (
                    records(through, through_sequence) if through_sequence > through else ()
                ):
                    if isinstance(record, SourceNode):
                        stats["source_records_indexed"] += 1
                        for turn in record.turns:
                            for start in range(0, len(turn.text), self.passage_chars):
                                end = min(start + self.passage_chars, len(turn.text))
                                self._add(
                                    turn.text[start:end],
                                    SourceSpan(record.node_id, turn.turn_id, start, end),
                                    {
                                        "role": turn.role,
                                        "source_metadata": record.metadata,
                                        "timestamp_ms": record.timestamp_ms,
                                    },
                                    "source",
                                )
                                stats["passages_indexed"] += 1
                    elif isinstance(record, JournalEntry):
                        stats["journal_records_indexed"] += 1
                        if not isinstance(record.value, str):
                            continue
                        provenance = record.provenance
                        for start in range(0, len(record.value), self.passage_chars):
                            end = min(start + self.passage_chars, len(record.value))
                            self._add(
                                record.value[start:end],
                                JournalRef(record.owning_node_id, record.entry_id, start, end),
                                {
                                    "relation": record.relation,
                                    "record_kind": record.record_kind,
                                    "journal_sequence": record.journal_sequence,
                                    "recorded_at_ms": record.recorded_at_ms,
                                    "applicability": record.applicability,
                                    "provenance": {
                                        "origin": provenance.origin,
                                        "producer": provenance.producer,
                                        "model": provenance.model,
                                        "prompt_version": provenance.prompt_version,
                                        "policy_version": provenance.policy_version,
                                        "supporting_references": [
                                            reference_to_dict(ref)
                                            for ref in provenance.supporting_references
                                        ],
                                    },
                                },
                                "journal",
                            )
                            stats["passages_indexed"] += 1
                connection.execute(
                    "UPDATE index_info SET through_sequence=?", (max(through, through_sequence),)
                )
                connection.execute("COMMIT")
                return stats
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    def search(self, query: str, k: int, *, journals_only: bool = False) -> list[dict[str, Any]]:
        """Rank the indexed current corpus using SQLite's ordinary FTS5 BM25."""
        with self._lock:
            self._ensure_open()
            escaped = SQLiteBM25Retriever.escaped_query(query)
            if not escaped or not k:
                return []
            condition = " AND kind='journal'" if journals_only else ""
            rows = self._connection.execute(
                "SELECT passage_id,text,reference,metadata,bm25(passages) AS score FROM passages "
                "WHERE passages MATCH ?" + condition + " ORDER BY score,rowid LIMIT ?",
                (escaped, k),
            ).fetchall()
            return [
                {
                    "passage_id": row[0],
                    "text": row[1],
                    "reference": json.loads(row[2]),
                    "metadata": json.loads(row[3]),
                    "score": -float(row[4]),
                }
                for row in rows
            ]

    def close(self) -> None:
        """Close the shared projection after pending indexed operations finish."""
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True
