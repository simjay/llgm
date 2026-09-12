"""Async workspace over immutable source blobs and transactional SQLite metadata."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar
from urllib.parse import unquote, urlparse

from llgm.core.errors import (
    BudgetExceeded,
    CapabilityError,
    ConfigurationError,
    ConflictError,
    ReferenceResolutionError,
    SchemaError,
)
from llgm.core.types import (
    Conversation,
    Edge,
    EvidenceRef,
    IngestResult,
    JournalEntry,
    JournalRef,
    NodeRef,
    Provenance,
    ResolvedEvidence,
    SourceNode,
    SourceSpan,
    Turn,
    reference_from_dict,
    reference_to_dict,
)
from llgm.storage import BlobStore, LocalBlobStore, S3BlobStore
from llgm.storage._sqlite import enable_wal

T = TypeVar("T")
_SCHEMA_VERSION = 4


def _create_conversation_tables(connection: sqlite3.Connection) -> None:
    """Add immutable turn segments and persistent conversation routing state."""
    connection.execute("""CREATE TABLE IF NOT EXISTS source_turns (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT NOT NULL REFERENCES sources(node_id),
        turn_id TEXT NOT NULL, role TEXT NOT NULL, length INTEGER NOT NULL,
        blob_digest TEXT NOT NULL, metadata TEXT NOT NULL, timestamp_ms INTEGER,
        UNIQUE(node_id,turn_id))""")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS source_turn_order ON source_turns(node_id,sequence)"
    )
    connection.execute("""CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY, node_id TEXT NOT NULL REFERENCES sources(node_id))""")


def _json(value: Any) -> str:
    """Encode stable record bytes, rejecting nonfinite or non-JSON values."""
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError("Record values must be finite JSON-serializable data") from exc


def _provenance_dict(value: Provenance) -> dict[str, Any]:
    """Serialize provenance and its typed supporting references."""
    if not isinstance(value, Provenance):
        raise SchemaError("provenance must be a Provenance record")
    return {
        "origin": value.origin,
        "producer": value.producer,
        "supporting_references": [reference_to_dict(ref) for ref in value.supporting_references],
        "model": value.model,
        "prompt_version": value.prompt_version,
        "policy_version": value.policy_version,
    }


def journal_to_dict(entry: JournalEntry) -> dict[str, Any]:
    """Versioned journal representation used by structural JournalRef reads."""
    return {
        "schema_version": entry.schema_version,
        "entry_id": entry.entry_id,
        "owning_node_id": entry.owning_node_id,
        "journal_sequence": entry.journal_sequence,
        "recorded_at_ms": entry.recorded_at_ms,
        "subject": reference_to_dict(entry.subject),
        "record_kind": entry.record_kind,
        "relation": entry.relation,
        "value": {"type": "inline_text", "text": entry.value}
        if isinstance(entry.value, str)
        else reference_to_dict(entry.value),
        "provenance": _provenance_dict(entry.provenance),
        "applicability": entry.applicability,
    }


def _journal_from_dict(value: dict[str, Any]) -> JournalEntry:
    """Restore a persisted journal entry and its typed references."""
    data = dict(value)
    data["subject"] = reference_from_dict(data["subject"])
    data["value"] = (
        data["value"]["text"]
        if data["value"]["type"] == "inline_text"
        else reference_from_dict(data["value"])
    )
    provenance = dict(data["provenance"])
    provenance["supporting_references"] = tuple(
        reference_from_dict(ref) for ref in provenance["supporting_references"]
    )
    data["provenance"] = Provenance(**provenance)
    return JournalEntry(**data)


def edge_to_dict(edge: Edge) -> dict[str, Any]:
    """Encode relationship identity and both creation and withdrawal provenance."""
    return {
        "edge_id": edge.edge_id,
        "source_node_id": edge.source_node_id,
        "target_node_id": edge.target_node_id,
        "relation": edge.relation,
        "provenance": _provenance_dict(edge.provenance),
        "recorded_at_ms": edge.recorded_at_ms,
        "applicability": edge.applicability,
        "withdrawn_at_ms": edge.withdrawn_at_ms,
        "withdrawal_provenance": _provenance_dict(edge.withdrawal_provenance)
        if edge.withdrawal_provenance is not None
        else None,
    }


def _edge_from_dict(value: dict[str, Any]) -> Edge:
    """Restore an edge without losing typed source or journal support references."""
    data = dict(value)
    for field in ("provenance", "withdrawal_provenance"):
        if data[field] is not None:
            provenance = dict(data[field])
            provenance["supporting_references"] = tuple(
                reference_from_dict(ref) for ref in provenance["supporting_references"]
            )
            data[field] = Provenance(**provenance)
    return Edge(**data)


def _create_edge_tables(connection: sqlite3.Connection) -> None:
    """Create only schema-3 additions, within an existing publication transaction."""
    statements = (
        "CREATE TABLE edges(edge_id TEXT PRIMARY KEY, source_node_id TEXT NOT NULL REFERENCES sources(node_id), "
        "target_node_id TEXT NOT NULL REFERENCES sources(node_id), relation TEXT NOT NULL, identity TEXT NOT NULL, "
        "withdrawn_at_ms INTEGER, payload TEXT NOT NULL)",
        "CREATE INDEX edges_outgoing ON edges(source_node_id,relation)",
        "CREATE UNIQUE INDEX edges_active_identity ON edges(identity) WHERE withdrawn_at_ms IS NULL",
        "CREATE TABLE _journal_operational(entry_id TEXT PRIMARY KEY REFERENCES journal_entries(entry_id))",
        "CREATE TABLE _journal_classification(entry_id TEXT PRIMARY KEY REFERENCES journal_entries(entry_id), "
        "classification TEXT NOT NULL, edge_id TEXT REFERENCES edges(edge_id))",
    )
    for statement in statements:
        connection.execute(statement)


def _compact_entries(entries: list[JournalEntry]) -> list[JournalEntry]:
    """Discard only dominated exact-applicability entries without correction dependencies.

    Full history remains separately addressable. Conservatively preserve every
    slot touched by a correction, all proposals and unknown applicability rules.
    Different validity intervals never collapse into a current-only projection.
    """
    from llgm.core.time import normalize_applicability

    slots = {}
    applicable = {}
    for entry in entries:
        scope = (entry.applicability or {}).get("scope", {})
        slots[entry.entry_id] = _json((reference_to_dict(entry.subject), entry.relation, scope))
        try:
            normalized = normalize_applicability(entry.applicability)
            if set(normalized) - {"scope", "valid_from_ms", "valid_until_ms"}:
                continue
            applicable[entry.entry_id] = _json(normalized)
        except (SchemaError, TypeError, ValueError):
            continue
    protected = {
        slots[entry.subject.entry_id]
        for entry in entries
        if entry.record_kind == "correction"
        and isinstance(entry.subject, JournalRef)
        and entry.subject.entry_id in slots
    }
    retained = {}
    groups = {}
    for entry in entries:
        slot = slots[entry.entry_id]
        can_compact = (
            entry.record_kind in {"assertion", "overwrite"}
            and slot not in protected
            and entry.relation not in {"suggests", "suggestion", "proposed"}
            and entry.entry_id in applicable
        )
        if can_compact:
            key = slot, applicable[entry.entry_id]
            if entry.record_kind == "overwrite":
                for predecessor in groups.get(key, ()):
                    retained.pop(predecessor, None)
                groups[key] = []
            groups.setdefault(key, []).append(entry.entry_id)
        retained[entry.entry_id] = entry
    return list(retained.values())


def _local_path(uri: str, kind: str) -> Path:
    """Resolve a path or local file URI without accepting remote hosts."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ConfigurationError(f"{kind} requires a local file URI")
        return Path(unquote(parsed.path)).expanduser()
    if parsed.scheme:
        raise ConfigurationError(f"{kind} must be a local path or file URI")
    return Path(uri).expanduser()


class Workspace:
    """A single logical evidence workspace.

    Use ``async with Workspace.open(path)``. Blocking storage operations run in a
    thread and a per-instance lock. SQLite serializes writers across instances.
    Canceling the await does not roll back a storage operation already running.
    Explicit idempotency keys make publication retries safe.
    """

    def __init__(self, database_path: str | Path, blob_store: BlobStore):
        """Bind storage adapters. Defer database initialization until first use."""
        self.database_path = str(database_path)
        self.blob_store = blob_store
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._workspace_id: str | None = None
        self._closed = False
        self._lexical_indexes = {}
        self._conversation_lock = asyncio.Lock()

    @classmethod
    def open(
        cls, path: str | Path | None = None, *, settings=None, blob_store: BlobStore | None = None
    ) -> Workspace:
        """Select persistence once. Postgres fails explicitly until implemented.

        An explicit workspace path overrides configured local database/blob
        locations. An injected BlobStore overrides configured blob selection.
        Environment variables are read only by the Settings factory.
        """
        if settings is None and path is None:
            from llgm.core.config import Settings

            settings = Settings.from_env()
        if settings is not None and settings.metadata_backend != "sqlite":
            raise CapabilityError(
                f"Metadata backend {settings.metadata_backend!r} is not implemented; select sqlite"
            )
        if path is not None:
            directory = Path(path).expanduser()
            database_path = directory / "metadata.sqlite3"
            selected_blob = blob_store or LocalBlobStore(directory / "blobs")
        else:
            database_url = settings.database_url
            if database_url == "sqlite:///:memory:":
                database_path = ":memory:"
            elif database_url.startswith("sqlite:///"):
                database_path = Path(unquote(database_url[len("sqlite:///") :])).expanduser()
            else:
                raise ConfigurationError(
                    "SQLite database_url must start with sqlite:/// (use four slashes for an absolute path)"
                )
            if blob_store is not None:
                selected_blob = blob_store
            elif settings.blob_backend == "local":
                selected_blob = LocalBlobStore(_local_path(settings.blob_uri, "Local blob storage"))
            elif settings.blob_backend == "s3":
                selected_blob = S3BlobStore(settings.blob_uri)
            else:
                raise ConfigurationError(f"Unknown blob backend: {settings.blob_backend!r}")
        return cls(database_path, selected_blob)

    async def __aenter__(self) -> Workspace:
        """Initialize the metadata connection and return this workspace."""
        await self._run(lambda: None)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        """Close this workspace when its context ends."""
        await self.close()

    async def close(self) -> None:
        """Close the connection permanently after pending locked operations."""

        def close_sync():
            """Serialize connection shutdown with storage operations."""
            with self._lock:
                for index in self._lexical_indexes.values():
                    index.close()
                self._lexical_indexes.clear()
                if self._connection is not None:
                    self._connection.close()
                    self._connection = None
                self._closed = True

        await asyncio.to_thread(close_sync)

    def _ensure_open(self) -> sqlite3.Connection:
        """Open or initialize a compatible SQLite schema under the caller's lock."""
        if self._closed:
            raise ConfigurationError("Workspace is closed; open a new Workspace instance")
        if self._connection is not None:
            return self._connection
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.database_path, timeout=30, isolation_level=None, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, _SCHEMA_VERSION}:
                raise CapabilityError(
                    f"Workspace schema {version} is unsupported; create a new workspace or explicitly export old data"
                )
            connection.execute("PRAGMA foreign_keys = ON")
            enable_wal(connection)
            connection.execute("PRAGMA synchronous = FULL")
            # Serialize first-open schema creation across workspace instances.
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, _SCHEMA_VERSION}:
                raise CapabilityError(
                    f"Workspace schema {version} is unsupported; explicit migration is required"
                )
            if version == 0:
                # Refuse to adopt an unrelated database with unversioned tables.
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                if tables:
                    raise CapabilityError(
                        "Unversioned nonempty database requires explicit migration"
                    )
                schema = """
                    CREATE TABLE IF NOT EXISTS workspace_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS sources (
                        node_id TEXT PRIMARY KEY, blob_digest TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS journal_entries (
                        node_id TEXT NOT NULL, entry_id TEXT UNIQUE NOT NULL, journal_sequence INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        PRIMARY KEY(node_id, journal_sequence)
                    );
                    CREATE TABLE IF NOT EXISTS _index_changes (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, record_id TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS idempotency (
                        operation TEXT NOT NULL, key TEXT NOT NULL, fingerprint TEXT NOT NULL, result TEXT NOT NULL,
                        PRIMARY KEY(operation, key)
                    );
                    PRAGMA user_version = 4;
                """
                # executescript would implicitly end our transaction first.
                for statement in schema.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                _create_edge_tables(connection)
                _create_conversation_tables(connection)
                connection.execute(
                    "INSERT INTO workspace_metadata(key,value) VALUES ('workspace_id',?)",
                    (uuid.uuid4().hex,),
                )
            identity = connection.execute(
                "SELECT value FROM workspace_metadata WHERE key='workspace_id'"
            ).fetchone()
            if identity is None or not identity[0]:
                raise CapabilityError(
                    "Workspace identity is missing; explicit recovery is required"
                )
            self._workspace_id = identity[0]
            connection.execute("COMMIT")
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        return connection

    async def _run(self, operation: Callable[[], T]) -> T:
        """Run synchronous storage work off-loop under the workspace lock."""

        def locked() -> T:
            """Initialize the connection and execute one serialized operation."""
            with self._lock:
                self._ensure_open()
                return operation()

        return await asyncio.to_thread(locked)

    @contextmanager
    def _transaction(self, *, write: bool = True):
        """Commit one operation or roll it back while the workspace lock is held."""
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _key(key: str | None) -> None:
        """Reject empty or nontext idempotency keys."""
        if key is not None and (not isinstance(key, str) or not key):
            raise SchemaError("Idempotency key must be a nonempty string")

    def _retry(self, operation: str, key: str | None, fingerprint: str) -> dict[str, Any] | None:
        """Return a prior result only when retry content matches exactly."""
        if key is None:
            return None
        row = self._connection.execute(
            "SELECT fingerprint,result FROM idempotency WHERE operation=? AND key=?",
            (operation, key),
        ).fetchone()
        if row is None:
            return None
        if row["fingerprint"] != fingerprint:
            raise ConflictError(
                f"Idempotency key {key!r} was already used for different {operation} content"
            )
        return json.loads(row["result"])

    def _record_retry(
        self, operation: str, key: str | None, fingerprint: str, result: dict[str, Any]
    ) -> None:
        """Store a retry result within the caller's publication transaction."""
        if key is not None:
            self._connection.execute(
                "INSERT INTO idempotency VALUES (?,?,?,?)",
                (operation, key, fingerprint, _json(result)),
            )

    async def ingest(
        self, conversation: Conversation, *, idempotency_key: str | None = None
    ) -> IngestResult:
        """Publish one immutable node. An existing identity cannot receive different content."""
        if not isinstance(conversation, Conversation):
            raise SchemaError("ingest requires a Conversation")
        self._key(idempotency_key)
        payload = {
            "schema_version": 2,
            "turns": [
                {"turn_id": turn.turn_id, "role": turn.role, "text": turn.text}
                for turn in conversation.turns
            ],
            "metadata": dict(conversation.metadata),
            "timestamp_ms": conversation.timestamp_ms,
        }
        encoded = _json(payload).encode("utf-8")
        fingerprint = hashlib.sha256(
            _json({"node_id": conversation.node_id, "payload": payload}).encode("utf-8")
        ).hexdigest()

        def operation() -> IngestResult:
            """Publish the blob before atomically committing its source reference."""
            # Publish immutable bytes first. A failed metadata transaction may
            # leave an orphan blob, but never a visible partial source.
            digest = self.blob_store.put(encoded)
            with self._transaction() as connection:
                prior = self._retry("ingest", idempotency_key, fingerprint)
                if prior is not None:
                    return IngestResult(**prior, created=False)
                node_id = conversation.node_id or uuid.uuid4().hex
                existing = connection.execute(
                    "SELECT blob_digest FROM sources WHERE node_id=?", (node_id,)
                ).fetchone()
                if existing is not None:
                    if existing["blob_digest"] != digest:
                        raise ConflictError(
                            f"Source node {node_id!r} is immutable; use a new node ID"
                        )
                    result = {"node_id": node_id}
                    self._record_retry("ingest", idempotency_key, fingerprint, result)
                    return IngestResult(**result, created=False)
                connection.execute("INSERT INTO sources VALUES (?,?)", (node_id, digest))
                connection.execute(
                    "INSERT INTO _index_changes(kind,record_id) VALUES ('source',?)", (node_id,)
                )
                result = {"node_id": node_id}
                self._record_retry("ingest", idempotency_key, fingerprint, result)
                return IngestResult(**result, created=True)

        return await self._run(operation)

    def _base_source(self, node_id: str) -> SourceNode:
        """Resolve a node identity and verify its immutable source blob."""
        row = self._connection.execute(
            "SELECT * FROM sources WHERE node_id=?", (node_id,)
        ).fetchone()
        if row is None:
            raise ReferenceResolutionError(f"Source {node_id!r} does not exist")
        data = json.loads(self.blob_store.get(row["blob_digest"]))
        return SourceNode(
            row["node_id"],
            tuple(Turn(**turn) for turn in data["turns"]),
            data["metadata"],
            data["timestamp_ms"],
        )

    def _source(self, node_id: str) -> SourceNode:
        """Materialize a complete node only for callers requesting its full history."""
        source = self._base_source(node_id)
        turns = tuple(
            Turn(
                row["turn_id"], row["role"], self.blob_store.get(row["blob_digest"]).decode("utf-8")
            )
            for row in self._connection.execute(
                "SELECT * FROM source_turns WHERE node_id=? ORDER BY sequence", (node_id,)
            )
        )
        return SourceNode(node_id, source.turns + turns, source.metadata, source.timestamp_ms)

    async def append_conversation(
        self, conversation_id, conversation, *, node_id=None, idempotency_key=None
    ):
        """Append immutable turns to a topic and persist the conversation's active node.

        A missing node ID creates a topic. Existing turn IDs never change. Input
        turn IDs are namespaced per append so separate imported sessions can use
        the same IDs. A retry key deduplicates the entire append transaction.
        """
        from llgm.memory.conversation import append_conversation

        return await self._run(
            lambda: append_conversation(
                self,
                conversation_id,
                conversation,
                node_id=node_id,
                idempotency_key=idempotency_key,
            )
        )

    async def conversation_node(self, conversation_id: str) -> str | None:
        """Find the persisted active topic without loading its source text."""

        def operation():
            """Read one routing record under the workspace lock."""
            row = self._connection.execute(
                "SELECT node_id FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
            return row[0] if row else None

        return await self._run(operation)

    async def source_info(self, node_id: str, *, offset=0, limit=32) -> dict:
        """Page turn coordinates without loading appended conversation text."""
        from llgm.memory.conversation import source_info

        return await self._run(lambda: source_info(self, node_id, offset=offset, limit=limit))

    async def sources(self) -> list[SourceNode]:
        """Load all currently published nodes, ordered by node ID."""

        def operation():
            """Load each immutable node in a serialized metadata read."""
            rows = self._connection.execute(
                "SELECT node_id FROM sources ORDER BY node_id"
            ).fetchall()
            return [self._source(row["node_id"]) for row in rows]

        return await self._run(operation)

    async def source_ids(self) -> list[str]:
        """List published node identities without loading their source blobs."""

        def operation():
            """Select immutable node IDs without materializing their text."""
            return [
                row[0]
                for row in self._connection.execute("SELECT node_id FROM sources ORDER BY node_id")
            ]

        return await self._run(operation)

    async def source(self, node_id: str) -> SourceNode:
        """Load just one immutable node from the currently published records."""
        return await self._run(lambda: self._source(node_id))

    async def publish_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        relation: str,
        provenance: Provenance,
        applicability: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Edge:
        """Publish a directed edge independently of journals, deduplicating active relationships.

        Source, target, relation and applicability define duplicate identity. The
        first publication retains its provenance. A withdrawn edge is never
        reactivated. A later publication can create a new relationship ID.
        """
        from llgm.core.time import normalize_applicability

        self._key(idempotency_key)
        candidate = Edge(
            uuid.uuid4().hex,
            source_node_id,
            target_node_id,
            relation,
            provenance,
            time.time_ns() // 1_000_000,
            normalize_applicability(applicability),
        )
        request = edge_to_dict(candidate)
        request.pop("edge_id")
        request.pop("recorded_at_ms")
        fingerprint = hashlib.sha256(_json(request).encode()).hexdigest()
        identity = _json(
            {
                key: request[key]
                for key in ("source_node_id", "target_node_id", "relation", "applicability")
            }
        )
        # Decode the captured representation so caller-owned mappings cannot
        # mutate provenance or applicability while publication waits for a lock.
        candidate = _edge_from_dict(json.loads(_json(edge_to_dict(candidate))))

        def operation() -> Edge:
            """Validate support and insert one edge or return the transaction's existing result."""
            with self._transaction() as connection:
                prior = self._retry("edge", idempotency_key, fingerprint)
                if prior is not None:
                    return _edge_from_dict(prior)
                for node_id in (source_node_id, target_node_id):
                    if (
                        connection.execute(
                            "SELECT 1 FROM sources WHERE node_id=?", (node_id,)
                        ).fetchone()
                        is None
                    ):
                        raise ReferenceResolutionError(f"Source {node_id!r} does not exist")
                for ref in candidate.provenance.supporting_references:
                    self._resolve(ref)
                prior = connection.execute(
                    "SELECT payload FROM edges WHERE identity=? AND withdrawn_at_ms IS NULL",
                    (identity,),
                ).fetchone()
                result = json.loads(prior[0]) if prior is not None else edge_to_dict(candidate)
                if prior is None:
                    connection.execute(
                        "INSERT INTO edges VALUES (?,?,?,?,?,?,?)",
                        (
                            candidate.edge_id,
                            source_node_id,
                            target_node_id,
                            relation,
                            identity,
                            None,
                            _json(result),
                        ),
                    )
                self._record_retry("edge", idempotency_key, fingerprint, result)
                return _edge_from_dict(result)

        return await self._run(operation)

    async def edge(self, edge_id: str) -> Edge:
        """Read an edge's current withdrawal state by its stable opaque ID."""

        def operation() -> Edge:
            """Return the independently persisted edge without reading its source text."""
            row = self._connection.execute(
                "SELECT payload FROM edges WHERE edge_id=?", (edge_id,)
            ).fetchone()
            if row is None:
                raise ReferenceResolutionError(f"Edge {edge_id!r} does not exist")
            return _edge_from_dict(json.loads(row[0]))

        return await self._run(operation)

    async def edges(
        self,
        node_id: str,
        relation: str | None = None,
        *,
        include_withdrawn: bool = False,
    ) -> list[Edge]:
        """List independently stored outgoing edges, excluding withdrawn records by default."""
        if type(include_withdrawn) is not bool:
            raise SchemaError("include_withdrawn must be a boolean")

        def operation() -> list[Edge]:
            """Query indexed outgoing relationships without interpreting semantic journals."""
            if (
                self._connection.execute(
                    "SELECT 1 FROM sources WHERE node_id=?", (node_id,)
                ).fetchone()
                is None
            ):
                raise ReferenceResolutionError(f"Source {node_id!r} does not exist")
            query = "SELECT payload FROM edges WHERE source_node_id=?"
            parameters = [node_id]
            if relation is not None:
                query += " AND relation=?"
                parameters.append(relation)
            if not include_withdrawn:
                query += " AND withdrawn_at_ms IS NULL"
            return [
                _edge_from_dict(json.loads(row[0]))
                for row in self._connection.execute(query + " ORDER BY edge_id", parameters)
            ]

        return await self._run(operation)

    async def withdraw_edge(
        self,
        edge_id: str,
        *,
        provenance: Provenance,
        idempotency_key: str | None = None,
    ) -> Edge:
        """Withdraw exactly one edge while retaining its creation and withdrawal provenance."""
        self._key(idempotency_key)
        request = {"edge_id": edge_id, "provenance": _provenance_dict(provenance)}
        fingerprint = hashlib.sha256(_json(request).encode()).hexdigest()
        captured = json.loads(_json(request))["provenance"]

        def operation() -> Edge:
            """Publish an irreversible withdrawal of this edge ID in one SQLite transaction."""
            with self._transaction() as connection:
                retry = self._retry("withdraw_edge", idempotency_key, fingerprint)
                if retry is not None:
                    return _edge_from_dict(retry)
                row = connection.execute(
                    "SELECT payload FROM edges WHERE edge_id=?", (edge_id,)
                ).fetchone()
                if row is None:
                    raise ReferenceResolutionError(f"Edge {edge_id!r} does not exist")
                result = json.loads(row[0])
                if result["withdrawn_at_ms"] is None:
                    for raw in captured["supporting_references"]:
                        self._resolve(reference_from_dict(raw))
                    result.update(
                        withdrawn_at_ms=time.time_ns() // 1_000_000, withdrawal_provenance=captured
                    )
                    connection.execute(
                        "UPDATE edges SET withdrawn_at_ms=?,payload=? WHERE edge_id=?",
                        (result["withdrawn_at_ms"], _json(result), edge_id),
                    )
                self._record_retry("withdraw_edge", idempotency_key, fingerprint, result)
                return _edge_from_dict(result)

        return await self._run(operation)

    def _changed_records(self, after: int, cutoff: int, *, include_sources: bool = True):
        """Stream unseen source/journal appends using a private indexing cursor."""
        rows = self._connection.execute(
            "SELECT kind,record_id FROM _index_changes WHERE sequence>? AND sequence<=? ORDER BY sequence",
            (after, cutoff),
        )
        for row in rows:
            if row["kind"] == "source":
                if include_sources:
                    yield self._base_source(row["record_id"])
            elif row["kind"] == "turn":
                if include_sources:
                    turn = self._connection.execute(
                        "SELECT * FROM source_turns WHERE sequence=?", (row["record_id"],)
                    ).fetchone()
                    yield SourceNode(
                        turn["node_id"],
                        (
                            Turn(
                                turn["turn_id"],
                                turn["role"],
                                self.blob_store.get(turn["blob_digest"]).decode("utf-8"),
                            ),
                        ),
                        json.loads(turn["metadata"]),
                        turn["timestamp_ms"],
                    )
            else:
                payload = self._connection.execute(
                    "SELECT payload FROM journal_entries WHERE entry_id=?", (row["record_id"],)
                ).fetchone()[0]
                yield _journal_from_dict(json.loads(payload))

    async def _lexical_index(self, passage_chars: int, *, include_sources: bool = True):
        """Refresh and return the workspace-owned width-specific derived index."""

        def operation():
            """Serialize metadata access while streaming only new append records."""
            from llgm.storage.lexical import LexicalIndex

            cutoff = self._connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM _index_changes"
            ).fetchone()[0]
            key = passage_chars, include_sources
            index = self._lexical_indexes.get(key)
            if index is None:
                path = ":memory:"
                if self.database_path != ":memory:":
                    directory = Path(self.database_path + ".indexes")
                    directory.mkdir(parents=True, exist_ok=True)
                    kind = "evidence" if include_sources else "journals"
                    path = directory / f"{kind}-{passage_chars}.sqlite3"
                index = LexicalIndex(path, self._workspace_id, passage_chars)
                self._lexical_indexes[key] = index

            def records(after, through):
                """Load source blobs only when the local index supplies source retrieval."""
                return self._changed_records(after, through, include_sources=include_sources)

            return index, index.refresh(cutoff, records)

        return await self._run(operation)

    def _journal_entry(self, ref: JournalRef) -> JournalEntry:
        """Load one published journal record by its owning node and immutable ID."""
        row = self._connection.execute(
            "SELECT payload FROM journal_entries WHERE node_id=? AND entry_id=?",
            (ref.node_id, ref.entry_id),
        ).fetchone()
        if row is None:
            raise ReferenceResolutionError(
                f"Journal entry {ref.entry_id!r} does not exist on node {ref.node_id!r}"
            )
        return _journal_from_dict(json.loads(row["payload"]))

    def _resolve(self, ref: EvidenceRef) -> ResolvedEvidence:
        """Resolve canonical source text or a published journal entry."""
        if isinstance(ref, (NodeRef, SourceSpan)):
            if isinstance(ref, SourceSpan):
                row = self._connection.execute(
                    "SELECT * FROM source_turns WHERE node_id=? AND turn_id=?",
                    (ref.node_id, ref.turn_id),
                ).fetchone()
                if row is not None:
                    if ref.end > row["length"]:
                        raise ReferenceResolutionError("Source span exceeds turn length")
                    text = self.blob_store.get(row["blob_digest"]).decode("utf-8")
                    return ResolvedEvidence(
                        text[ref.start : ref.end],
                        ref,
                        {
                            "role": row["role"],
                            "source_metadata": json.loads(row["metadata"]),
                            "timestamp_ms": row["timestamp_ms"],
                        },
                    )
            source = self._source(ref.node_id)
            if isinstance(ref, NodeRef):
                # This is a display projection, not a source-offset coordinate
                # space. SourceSpan always addresses an exact individual turn.
                text = "\n\n".join(f"[{turn.role}]\n{turn.text}" for turn in source.turns)
                return ResolvedEvidence(
                    text,
                    NodeRef(source.node_id),
                    {
                        "source_metadata": source.metadata,
                        "timestamp_ms": source.timestamp_ms,
                        "projection": "role_labeled_turns",
                        "turns": [
                            {
                                "turn_id": turn.turn_id,
                                "role": turn.role,
                                "reference": reference_to_dict(
                                    SourceSpan(source.node_id, turn.turn_id, 0, len(turn.text))
                                ),
                            }
                            for turn in source.turns
                        ],
                    },
                )
            turn = next((turn for turn in source.turns if turn.turn_id == ref.turn_id), None)
            if turn is None or ref.end > len(turn.text):
                raise ReferenceResolutionError(
                    "Source span refers to an unknown turn or out-of-bounds range"
                )
            return ResolvedEvidence(
                turn.text[ref.start : ref.end],
                ref,
                {
                    "role": turn.role,
                    "source_metadata": source.metadata,
                    "timestamp_ms": source.timestamp_ms,
                },
            )
        if isinstance(ref, JournalRef):
            entry = self._journal_entry(ref)
            if ref.start is None:
                text = _json(journal_to_dict(entry))
            else:
                if not isinstance(entry.value, str):
                    raise ReferenceResolutionError(
                        "Ranged JournalRef requires an inline-text value"
                    )
                if ref.end > len(entry.value):
                    raise ReferenceResolutionError("JournalRef range exceeds inline-text length")
                text = entry.value[ref.start : ref.end]
            return ResolvedEvidence(
                text,
                ref,
                {
                    "journal_sequence": entry.journal_sequence,
                    "record_kind": entry.record_kind,
                    "provenance": _provenance_dict(entry.provenance),
                    "relation": entry.relation,
                    "recorded_at_ms": entry.recorded_at_ms,
                    "applicability": entry.applicability,
                },
            )
        raise ReferenceResolutionError("Unsupported reference type")

    async def resolve(self, reference: EvidenceRef) -> ResolvedEvidence:
        """Resolve an immutable reference against currently published records."""
        return await self._run(lambda: self._resolve(reference))

    async def append_journal(
        self,
        node_id: str,
        *,
        subject: EvidenceRef,
        relation: str,
        value: str | EvidenceRef,
        provenance: Provenance,
        record_kind: str = "assertion",
        applicability: Mapping[str, Any] | None = None,
        expected_sequence: int | None = None,
        idempotency_key: str | None = None,
    ) -> JournalEntry:
        """Append an assertion, overwrite, or correction without changing original source bytes.

        ``expected_sequence=0`` means an empty journal. Idempotent retries return
        the original record even if the journal has advanced in the meantime.
        """
        from llgm.core.time import normalize_applicability

        self._key(idempotency_key)
        if not isinstance(relation, str) or not relation:
            raise SchemaError("relation must be a nonempty string")
        if record_kind not in {"assertion", "overwrite", "correction"}:
            raise SchemaError("record_kind must be assertion, overwrite, or correction")
        if not isinstance(subject, (NodeRef, SourceSpan, JournalRef)):
            raise SchemaError("Journal subject must be an evidence reference")
        if subject.node_id != node_id:
            raise SchemaError("Journal subject must belong to the owning node")
        if record_kind == "correction" and not isinstance(subject, JournalRef):
            raise SchemaError("A journal correction must target an earlier JournalRef")
        if expected_sequence is not None and (
            type(expected_sequence) is not int or expected_sequence < 0
        ):
            raise SchemaError("expected_sequence must be a nonnegative integer")
        request = {
            "node_id": node_id,
            "subject": reference_to_dict(subject),
            "relation": relation,
            "value": {"type": "inline_text", "text": value}
            if isinstance(value, str)
            else reference_to_dict(value),
            "provenance": _provenance_dict(provenance),
            "record_kind": record_kind,
            "applicability": dict(applicability) if applicability is not None else None,
            "expected_sequence": expected_sequence,
        }
        fingerprint = hashlib.sha256(_json(request).encode("utf-8")).hexdigest()
        # Capture mutable caller mappings before yielding to the storage worker.
        applicability_copy = json.loads(_json(normalize_applicability(request["applicability"])))

        def operation() -> JournalEntry:
            """Commit a validated journal append with sequence and retry checks."""
            with self._transaction() as connection:
                prior = self._retry("journal", idempotency_key, fingerprint)
                if prior is not None:
                    return _journal_from_dict(prior)
                self._resolve(subject)
                if not isinstance(value, str):
                    self._resolve(value)
                for ref in provenance.supporting_references:
                    self._resolve(ref)
                sequence = connection.execute(
                    "SELECT COALESCE(MAX(journal_sequence),0) FROM journal_entries WHERE node_id=?",
                    (node_id,),
                ).fetchone()[0]
                if expected_sequence is not None and expected_sequence != sequence:
                    raise ConflictError(
                        f"Journal {node_id!r} expected sequence {expected_sequence}, actual sequence {sequence}"
                    )
                entry = JournalEntry(
                    uuid.uuid4().hex,
                    node_id,
                    sequence + 1,
                    time.time_ns() // 1_000_000,
                    subject,
                    record_kind,
                    relation,
                    value,
                    provenance,
                    applicability_copy,
                )
                serialized = journal_to_dict(entry)
                connection.execute(
                    "INSERT INTO journal_entries VALUES (?,?,?,?)",
                    (node_id, entry.entry_id, entry.journal_sequence, _json(serialized)),
                )
                connection.execute(
                    "INSERT INTO _index_changes(kind,record_id) VALUES ('journal',?)",
                    (entry.entry_id,),
                )
                connection.execute("INSERT INTO _journal_operational VALUES (?)", (entry.entry_id,))
                self._reconcile_journal(node_id, from_history=record_kind == "correction")
                self._record_retry("journal", idempotency_key, fingerprint, serialized)
                return entry

        return await self._run(operation)

    async def inspect_journal(self, node_id: str) -> list[JournalEntry]:
        """Return the node's current append history without interpreting it."""

        def operation():
            """Validate the owning node and load its current journal in append order."""
            if (
                self._connection.execute(
                    "SELECT 1 FROM sources WHERE node_id=? LIMIT 1", (node_id,)
                ).fetchone()
                is None
            ):
                raise ReferenceResolutionError(f"Source {node_id!r} does not exist")
            rows = self._connection.execute(
                "SELECT payload FROM journal_entries WHERE node_id=? ORDER BY journal_sequence",
                (node_id,),
            ).fetchall()
            return [_journal_from_dict(json.loads(row["payload"])) for row in rows]

        return await self._run(operation)

    def _reconcile_journal(self, node_id: str, *, from_history: bool = False) -> dict[str, int]:
        """Publish conservative operational membership inside the caller's transaction."""
        query = "SELECT j.payload FROM journal_entries j "
        if not from_history:
            query += "JOIN _journal_operational o ON o.entry_id=j.entry_id "
        query += (
            "WHERE j.node_id=? AND NOT EXISTS (SELECT 1 FROM _journal_classification c "
            "WHERE c.entry_id=j.entry_id AND c.classification='edge') ORDER BY j.journal_sequence"
        )
        entries = [
            _journal_from_dict(json.loads(row[0]))
            for row in self._connection.execute(query, (node_id,))
        ]
        retained = _compact_entries(entries)
        self._connection.execute(
            "DELETE FROM _journal_operational WHERE entry_id IN (SELECT entry_id FROM journal_entries WHERE node_id=?)",
            (node_id,),
        )
        self._connection.executemany(
            "INSERT INTO _journal_operational VALUES (?)", ((entry.entry_id,) for entry in retained)
        )
        total = self._connection.execute(
            "SELECT COUNT(*) FROM journal_entries WHERE node_id=?", (node_id,)
        ).fetchone()[0]
        return {
            "history_entries": total,
            "operational_entries": len(retained),
            "archived_entries": total - len(retained),
        }

    def _operational_rows(self, node_id: str, max_bytes: int) -> list[str]:
        """Check complete encoded size before loading operational records into memory."""
        if type(max_bytes) is not int or max_bytes <= 0:
            raise SchemaError("max_bytes must be a positive integer")
        if (
            self._connection.execute("SELECT 1 FROM sources WHERE node_id=?", (node_id,)).fetchone()
            is None
        ):
            raise ReferenceResolutionError(f"Source {node_id!r} does not exist")
        if (
            self._connection.execute(
                "SELECT 1 FROM _journal_classification c JOIN journal_entries j ON j.entry_id=c.entry_id "
                "WHERE j.node_id=? AND c.classification='unresolved' LIMIT 1",
                (node_id,),
            ).fetchone()
            is not None
        ):
            raise ConfigurationError(
                f"Node {node_id!r} has unresolved legacy journal classifications"
            )
        query = "FROM journal_entries j JOIN _journal_operational o ON o.entry_id=j.entry_id WHERE j.node_id=?"
        # Include JSON array brackets and separators in the encoded byte bound.
        size = self._connection.execute(
            "SELECT 2 + COALESCE(SUM(length(CAST(j.payload AS BLOB))),0) + MAX(COUNT(*)-1,0) "
            + query,
            (node_id,),
        ).fetchone()[0]
        if size > max_bytes:
            raise BudgetExceeded(
                f"Complete operational journal for {node_id!r} needs {size} bytes, exceeding max_bytes={max_bytes}; "
                "distinct live patches or retained correction/validity history require an explicit larger limit"
            )
        return [
            row[0]
            for row in self._connection.execute(
                "SELECT j.payload " + query + " ORDER BY j.journal_sequence", (node_id,)
            )
        ]

    async def operational_journal(
        self, node_id: str, *, max_bytes: int = 65536
    ) -> list[JournalEntry]:
        """Load the complete compact sidecar or fail before silently omitting any required record."""

        def operation() -> list[JournalEntry]:
            """Capture the size check and complete sidecar under one local SQLite read transaction."""
            with self._transaction(write=False):
                rows = self._operational_rows(node_id, max_bytes)
                return [_journal_from_dict(json.loads(row)) for row in rows]

        return await self._run(operation)

    async def compact_journal(self, node_id: str, *, max_bytes: int = 65536) -> dict[str, int]:
        """Rebuild exact operational membership from retained history with an explicit size limit.

        Overflow leaves the previous membership intact. Full history and every
        JournalRef remain available. No source copy or history deletion occurs.
        """

        def operation() -> dict[str, int]:
            """Serialize compaction with concurrent appends and publish all membership changes together."""
            with self._transaction():
                result = self._reconcile_journal(node_id, from_history=True)
                self._operational_rows(node_id, max_bytes)
                return result

        return await self._run(operation)
