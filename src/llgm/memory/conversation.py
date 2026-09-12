"""Append-only topic histories with stable source spans and bounded metadata reads."""

import hashlib
import uuid

from llgm.core.errors import ConfigurationError, SchemaError
from llgm.core.types import Conversation, IngestResult, SourceSpan, reference_to_dict


def append_fingerprint(conversation_id, conversation):
    """Identify exact imported bytes independently of a model's routing decision."""
    from llgm.memory.workspace import _json

    return hashlib.sha256(_json({"conversation_id": conversation_id,
        "turns": [(t.turn_id, t.role, t.text) for t in conversation.turns],
        "metadata": dict(conversation.metadata), "timestamp_ms": conversation.timestamp_ms,
    }).encode()).hexdigest()


def append_conversation(workspace, conversation_id, conversation, *, node_id, idempotency_key):
    """Publish turn blobs before atomically attaching them to one persistent topic."""
    from llgm.memory.workspace import _json

    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ConfigurationError("conversation_id must be nonempty text")
    if not isinstance(conversation, Conversation):
        raise SchemaError("append_conversation requires a Conversation")
    workspace._key(idempotency_key)
    fingerprint = append_fingerprint(conversation_id, conversation)
    blobs = [workspace.blob_store.put(turn.text.encode("utf-8")) for turn in conversation.turns]
    with workspace._transaction() as connection:
        prior = workspace._retry("conversation_append", idempotency_key, fingerprint)
        if prior is not None:
            return IngestResult(**prior, created=False)
        created = node_id is None
        if created:
            node_id = uuid.uuid4().hex
            base = _json(
                {
                    "schema_version": 2,
                    "turns": [],
                    "metadata": dict(conversation.metadata),
                    "timestamp_ms": conversation.timestamp_ms,
                }
            )
            digest = workspace.blob_store.put(base.encode("utf-8"))
            connection.execute("INSERT INTO sources VALUES (?,?)", (node_id, digest))
            connection.execute(
                "INSERT INTO _index_changes(kind,record_id) VALUES ('source',?)", (node_id,)
            )
        else:
            workspace._base_source(node_id)
        batch = uuid.uuid4().hex
        for turn, digest in zip(conversation.turns, blobs):
            cursor = connection.execute(
                "INSERT INTO source_turns(node_id,turn_id,role,length,blob_digest,metadata,timestamp_ms) VALUES (?,?,?,?,?,?,?)",
                (
                    node_id,
                    batch + ":" + turn.turn_id,
                    turn.role,
                    len(turn.text),
                    digest,
                    _json(dict(conversation.metadata)),
                    conversation.timestamp_ms,
                ),
            )
            connection.execute(
                "INSERT INTO _index_changes(kind,record_id) VALUES ('turn',?)",
                (str(cursor.lastrowid),),
            )
        connection.execute(
            "INSERT INTO conversations VALUES (?,?) ON CONFLICT(conversation_id) DO UPDATE SET node_id=excluded.node_id",
            (conversation_id, node_id),
        )
        result = {"node_id": node_id}
        workspace._record_retry("conversation_append", idempotency_key, fingerprint, result)
        return IngestResult(node_id, created)


def source_info(workspace, node_id, *, offset, limit):
    """Read a bounded page from legacy base coordinates and appended turn metadata."""
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 128:
        raise ConfigurationError("source_info requires offset >= 0 and 1 <= limit <= 128")
    base = workspace._base_source(node_id)
    connection = workspace._connection
    count = connection.execute(
        "SELECT count(*) FROM source_turns WHERE node_id=?", (node_id,)
    ).fetchone()[0]
    total = len(base.turns) + count
    rows = [
        {"turn_id": turn.turn_id, "role": turn.role, "length": len(turn.text)}
        for turn in base.turns[offset : offset + limit]
    ]
    if len(rows) < limit:
        rows.extend(
            dict(row)
            for row in connection.execute(
                "SELECT turn_id,role,length FROM source_turns WHERE node_id=? ORDER BY sequence LIMIT ? OFFSET ?",
                (node_id, limit - len(rows), max(0, offset - len(base.turns))),
            )
        )
    for row in rows:
        row["reference"] = reference_to_dict(SourceSpan(node_id, row["turn_id"], 0, row["length"]))
    return {
        "node_id": node_id,
        "offset": offset,
        "total_turns": total,
        "next_offset": offset + limit if offset + limit < total else None,
        "turns": rows,
    }
