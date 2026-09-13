"""Metadata-first inspection of workspace records without loading whole topic histories."""

from __future__ import annotations

import json
from pathlib import Path

from llgm.core.errors import ReferenceResolutionError
from llgm.memory.conversation import source_info
from llgm.storage import LocalBlobStore


def _source(workspace, node_id):
    """Resolve a source identity before exposing any associated storage record."""
    row = workspace._connection.execute(
        "SELECT blob_digest FROM sources WHERE node_id=?", (node_id,)
    ).fetchone()
    if row is None:
        raise ReferenceResolutionError(f"Unknown node: {node_id}")
    return row[0]


def _location(workspace, digest):
    """Describe a content-addressed blob without assuming every store is a filesystem."""
    if isinstance(workspace.blob_store, LocalBlobStore):
        return str((workspace.blob_store.directory / digest[:2] / digest[2:]).resolve())
    return f"sha256:{digest}"


async def graph(workspace, conversation_id):
    """Read nodes, active edges and conversation pointers in one metadata transaction."""

    def read():
        """Keep the topology internally consistent without reading source blobs."""
        with workspace._transaction(write=False) as connection:
            nodes = [
                dict(row)
                for row in connection.execute(
                    "SELECT s.node_id, coalesce(j.count,0) AS journal_count FROM sources s "
                    "LEFT JOIN (SELECT node_id,count(*) AS count FROM journal_entries GROUP BY node_id) j "
                    "ON j.node_id=s.node_id ORDER BY s.node_id"
                )
            ]
            edges = [
                dict(row)
                for row in connection.execute(
                    "SELECT edge_id,source_node_id,target_node_id FROM edges "
                    "WHERE withdrawn_at_ms IS NULL ORDER BY edge_id"
                )
            ]
            conversations = [
                dict(row)
                for row in connection.execute(
                    "SELECT conversation_id,node_id FROM conversations ORDER BY conversation_id"
                )
            ]
            current = next(
                (
                    row["node_id"]
                    for row in conversations
                    if row["conversation_id"] == conversation_id
                ),
                None,
            )
            return {
                "nodes": nodes,
                "edges": edges,
                "conversations": conversations,
                "conversation_id": conversation_id,
                "current_node_id": current,
                "database": str(Path(workspace.database_path).resolve())
                if workspace.database_path != ":memory:"
                else ":memory:",
            }

    return await workspace._run(read)


async def node(workspace, node_id, offset=0):
    """Page turn coordinates and physical locations, leaving text to explicit reads."""

    def read():
        """Read one page and its blob identities under the workspace lock."""
        digest = _source(workspace, node_id)
        page = source_info(workspace, node_id, offset=offset, limit=32)
        page["manifest_location"] = _location(workspace, digest)
        for turn in page["turns"]:
            row = workspace._connection.execute(
                "SELECT blob_digest FROM source_turns WHERE node_id=? AND turn_id=?",
                (node_id, turn["turn_id"]),
            ).fetchone()
            turn["location"] = _location(workspace, row[0] if row else digest)
            turn["storage"] = "turn" if row else "manifest"
        return page

    return await workspace._run(read)


async def journals(workspace, node_id, offset=0):
    """Page journal identities, including retained history, without returning their values."""

    def read():
        """Project journal headers in SQL so a large value stays out of the listing."""
        _source(workspace, node_id)
        rows = [
            dict(row)
            for row in workspace._connection.execute(
                "SELECT entry_id,journal_sequence,json_extract(payload,'$.record_kind') AS record_kind "
                "FROM journal_entries WHERE node_id=? ORDER BY journal_sequence LIMIT 33 OFFSET ?",
                (node_id, offset),
            )
        ]
        return {"entries": rows[:32], "next_offset": offset + 32 if len(rows) > 32 else None}

    return await workspace._run(read)


async def file(workspace, node_id, kind, record_id):
    """Open only a canonical source blob or journal record belonging to the selected node."""

    def read():
        """Resolve storage membership before loading the explicitly requested content."""
        digest = _source(workspace, node_id)
        if kind == "journal":
            row = workspace._connection.execute(
                "SELECT payload FROM journal_entries WHERE node_id=? AND entry_id=?",
                (node_id, record_id),
            ).fetchone()
            if row is None:
                raise ReferenceResolutionError("Unknown journal entry")
            return json.dumps(json.loads(row[0]), ensure_ascii=False, indent=2).encode()
        if kind == "turn":
            row = workspace._connection.execute(
                "SELECT blob_digest FROM source_turns WHERE node_id=? AND turn_id=?",
                (node_id, record_id),
            ).fetchone()
            if row is None:
                raise ReferenceResolutionError("This turn is stored inside the source manifest")
            digest = row[0]
        return workspace.blob_store.get(digest)

    return await workspace._run(read)
