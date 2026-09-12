"""Explicit schema-2 local-workspace copying with reviewed journal-link classification."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Mapping

from llgm.core.errors import ConfigurationError
from llgm.core.types import JournalRef, NodeRef
from llgm.memory.workspace import (
    Workspace,
    _create_conversation_tables,
    _create_edge_tables,
    _journal_from_dict,
)
from llgm.storage import LocalBlobStore


async def copy_schema3_workspace(source: str | Path, destination: str | Path) -> dict:
    """Copy local schema-3 evidence into schema 4 without modifying the original.

    Source and journal identities remain unchanged. Existing primary edges keep
    their recorded provenance and labels as historical data. New automatic
    connections use only related_to. The destination must not exist.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ConfigurationError("Migration destination must be new and outside the source")
    if not (source / "metadata.sqlite3").is_file():
        raise ConfigurationError("Migration requires a local workspace directory")

    def copy():
        """Stage a consistent database backup and verified blobs before publication."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".llgm-convert-", dir=destination.parent
        ) as temporary:
            staged = Path(temporary) / "workspace"
            staged.mkdir()
            original = sqlite3.connect(
                (source / "metadata.sqlite3").as_uri() + "?mode=ro", uri=True
            )
            copied = sqlite3.connect(staged / "metadata.sqlite3", isolation_level=None)
            try:
                if original.execute("PRAGMA user_version").fetchone()[0] != 3:
                    raise ConfigurationError("This converter accepts workspace schema 3 only")
                original.backup(copied)
                if copied.execute("PRAGMA user_version").fetchone()[0] != 3:
                    raise ConfigurationError("Source format changed during conversion")
                before, after = LocalBlobStore(source / "blobs"), LocalBlobStore(staged / "blobs")
                count = 0
                for (digest,) in copied.execute("SELECT blob_digest FROM sources"):
                    if after.put(before.get(digest)) != digest:
                        raise ConfigurationError("Copied blob identity changed")
                    count += 1
                copied.execute("BEGIN IMMEDIATE")
                _create_conversation_tables(copied)
                copied.execute(
                    "UPDATE workspace_metadata SET value=? WHERE key='workspace_id'",
                    (uuid.uuid4().hex,),
                )
                copied.execute("PRAGMA user_version=4")
                copied.execute("COMMIT")
            finally:
                copied.close()
                original.close()
            report = {
                "source_schema": 3,
                "workspace_schema": 4,
                "source_count": count,
                "history_preserved": True,
            }
            (staged / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
            if destination.exists():
                raise ConfigurationError("Migration destination appeared before publication")
            staged.rename(destination)
            return report

    return await asyncio.to_thread(copy)


async def copy_schema2_workspace(
    source: str | Path,
    destination: str | Path,
    *,
    journal_roles: Mapping[str, str],
) -> dict:
    """Copy a local schema-2 workspace, preserving original bytes and evidence identities.

    Every reference-valued journal record needs an explicit entry-ID mapping to
    ``edge``, ``amendment``, or ``unresolved``. Unresolved records remain readable
    in history but block effective operational reads. Only uncorrected whole-node
    assertion links can be promoted. Source blobs and original journals retain
    schema 2. The copied workspace metadata uses schema 4. Existing destinations
    are refused and source metadata is opened read-only.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not isinstance(journal_roles, Mapping) or any(
        not isinstance(key, str) or value not in {"edge", "amendment", "unresolved"}
        for key, value in journal_roles.items()
    ):
        raise ConfigurationError(
            "journal_roles must map entry IDs to edge, amendment, or unresolved"
        )
    journal_roles = dict(journal_roles)
    if destination.exists() or destination == source or source in destination.parents:
        raise ConfigurationError(
            "Migration destination must be new and outside the source workspace"
        )
    if not (source / "metadata.sqlite3").is_file():
        raise ConfigurationError("Schema-2 conversion requires a local workspace directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".llgm-convert-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "workspace"
        staged.mkdir()

        def copy_records():
            """Take a consistent SQLite backup and verify every copied immutable blob."""
            original = sqlite3.connect(
                (source / "metadata.sqlite3").as_uri() + "?mode=ro", uri=True
            )
            copied = sqlite3.connect(staged / "metadata.sqlite3", isolation_level=None)
            try:
                if original.execute("PRAGMA user_version").fetchone()[0] != 2:
                    raise ConfigurationError("This converter accepts workspace schema 2 only")
                original.backup(copied)
                if copied.execute("PRAGMA user_version").fetchone()[0] != 2:
                    raise ConfigurationError("Source format changed during conversion")
                entries = [
                    _journal_from_dict(json.loads(row[0]))
                    for row in copied.execute(
                        "SELECT payload FROM journal_entries ORDER BY node_id,journal_sequence"
                    )
                ]
                records = {entry.entry_id: entry for entry in entries}
                pointers = {entry.entry_id for entry in entries if not isinstance(entry.value, str)}
                missing = pointers - journal_roles.keys()
                if missing:
                    raise ConfigurationError(
                        "Explicit classification required for journal pointers: "
                        + ", ".join(sorted(missing))
                    )
                if journal_roles.keys() - records.keys():
                    raise ConfigurationError("Journal classification names an unknown entry ID")
                corrected = {
                    entry.subject.entry_id
                    for entry in entries
                    if entry.record_kind == "correction" and isinstance(entry.subject, JournalRef)
                }
                for entry_id, role in journal_roles.items():
                    entry = records[entry_id]
                    if role == "edge" and (
                        entry.record_kind != "assertion"
                        or not isinstance(entry.subject, NodeRef)
                        or not isinstance(entry.value, NodeRef)
                        or entry_id in corrected
                    ):
                        raise ConfigurationError(
                            f"Journal {entry_id!r} cannot be promoted automatically: require an uncorrected whole-node assertion"
                        )
                source_blobs, copied_blobs = (
                    LocalBlobStore(source / "blobs"),
                    LocalBlobStore(staged / "blobs"),
                )
                for (digest,) in copied.execute("SELECT DISTINCT blob_digest FROM sources"):
                    if copied_blobs.put(source_blobs.get(digest)) != digest:
                        raise ConfigurationError("Copied source blob identity changed")
                copied.execute("PRAGMA foreign_keys=ON")
                copied.execute("BEGIN IMMEDIATE")
                _create_edge_tables(copied)
                _create_conversation_tables(copied)
                copied.execute(
                    "INSERT INTO _journal_operational SELECT entry_id FROM journal_entries"
                )
                copied.execute(
                    "UPDATE workspace_metadata SET value=? WHERE key='workspace_id'",
                    (uuid.uuid4().hex,),
                )
                copied.execute("PRAGMA user_version=4")
                copied.execute("COMMIT")
                return records
            finally:
                copied.close()
                original.close()

        copying = asyncio.create_task(asyncio.to_thread(copy_records))
        try:
            records = await asyncio.shield(copying)
        except asyncio.CancelledError as cancellation:
            # Finish the read-only source backup before deleting its staging
            # directory; worker threads do not stop when their await is cancelled.
            while not copying.done():
                try:
                    await asyncio.shield(copying)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                copying.result()
            except Exception as error:
                cancellation.add_note(f"Backup cleanup also failed: {type(error).__name__}")
            raise cancellation
        mappings = []
        async with Workspace.open(staged) as workspace:
            for entry_id, role in sorted(journal_roles.items()):
                edge_id = None
                if role == "edge":
                    entry = records[entry_id]
                    edge = await workspace.publish_edge(
                        entry.owning_node_id,
                        entry.value.node_id,
                        relation=entry.relation,
                        provenance=entry.provenance,
                        applicability=entry.applicability,
                        idempotency_key="schema2-journal:" + entry_id,
                    )
                    edge_id = edge.edge_id
                mappings.append({"entry_id": entry_id, "classification": role, "edge_id": edge_id})

            def classify():
                """Publish reviewed classifications and rebuild complete operational membership."""
                with workspace._transaction() as connection:
                    connection.executemany(
                        "INSERT INTO _journal_classification VALUES (?,?,?)",
                        (
                            (item["entry_id"], item["classification"], item["edge_id"])
                            for item in mappings
                        ),
                    )
                    stats = {}
                    for (node_id,) in connection.execute(
                        "SELECT node_id FROM sources ORDER BY node_id"
                    ):
                        stats[node_id] = workspace._reconcile_journal(node_id, from_history=True)
                    return stats

            stats = await workspace._run(classify)
        report = {
            "source_schema": 2,
            "workspace_schema": 4,
            "source_path": str(source),
            "source_count": len(stats),
            "journal_classifications": mappings,
            "operational_journals": stats,
            "history_preserved": True,
        }
        (staged / "conversion.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if destination.exists():
            raise ConfigurationError("Migration destination appeared before publication")
        staged.rename(destination)
        return report
