"""Shared SQLite initialization for metadata and derived indexes."""

import sqlite3
import time


def enable_wal(connection: sqlite3.Connection) -> None:
    """Retry transient first-open journal-mode contention within the connection timeout."""
    timeout_ms = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    deadline = time.monotonic() + timeout_ms / 1000
    connection.execute("PRAGMA busy_timeout = 0")
    try:
        while True:
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as exc:
                # Concurrent mode changes can return BUSY before the schema's
                # BEGIN IMMEDIATE transaction can serialize first-openers.
                code = getattr(exc, "sqlite_errorcode", None)
                remaining = deadline - time.monotonic()
                if code is None or code & 0xFF != sqlite3.SQLITE_BUSY or remaining <= 0:
                    raise
                time.sleep(min(0.01, remaining))
    finally:
        connection.execute(f"PRAGMA busy_timeout = {timeout_ms}")
