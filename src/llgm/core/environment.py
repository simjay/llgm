"""Explicit loading of local environment files without shell evaluation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import MutableMapping

from llgm.core.errors import ConfigurationError


def load_env_file(path: str | Path, *, environ: MutableMapping[str, str] | None = None) -> None:
    """Fill missing environment variables from one explicitly selected UTF-8 file.

    The destination is ``os.environ`` unless a mutable mapping is supplied.
    Existing values, including empty strings, are preserved. Call this at
    application startup before constructing settings or provider clients.
    Loading never searches for a file or runs shell commands.

    Each assignment uses ``KEY=value`` with an optional ``export`` prefix.
    Blank lines, comments, single or double quoted single-line values and
    whitespace-prefixed inline comments are accepted. Values are literal,
    with no variable interpolation or escape processing. Duplicate keys,
    malformed syntax and unreadable files raise ``ConfigurationError``.
    The entire file is validated before changing the destination. Errors
    identify line numbers without exposing file contents or values.
    """
    try:
        document = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        raise ConfigurationError("Cannot read env file as UTF-8") from None

    values: dict[str, str] = {}
    for number, raw in enumerate(document.split("\n"), start=1):
        line = raw.strip()
        if "\x00" in raw:
            raise ConfigurationError(f"Invalid env file syntax at line {number}")
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)[ \t]*=(.*)", line)
        if match is None:
            raise ConfigurationError(f"Invalid env file assignment at line {number}")
        name, raw_value = match.groups()
        if name in values:
            raise ConfigurationError(f"Duplicate env file assignment at line {number}")
        values[name] = _parse_value(raw_value, number)

    destination = os.environ if environ is None else environ
    for name, value in values.items():
        destination.setdefault(name, value)


def _parse_value(raw: str, number: int) -> str:
    """Read a literal value and discard only comments outside enclosing quotes."""
    value = raw.strip()
    if not value or (raw[0] in " \t" and value.startswith("#")):
        return ""
    if value[0] in {"'", '"'}:
        end = value.find(value[0], 1)
        if end != -1:
            suffix = value[end + 1 :]
            if not suffix or (suffix[0] in " \t" and suffix.lstrip().startswith("#")):
                return value[1:end]
        raise ConfigurationError(f"Invalid env file quoted value at line {number}")
    return re.split(r"[ \t]+#", value, maxsplit=1)[0].rstrip()
