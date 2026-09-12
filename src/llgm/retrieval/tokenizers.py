"""Local-only tokenization. Diagnostic counters are never ColBERT substitutes."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Protocol

from llgm.core.errors import CapabilityError, ConfigurationError


class OffsetTokenizer(Protocol):
    """Token offsets and encoder-aware counts for bounded passage construction."""

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Return source-relative character spans for content tokens."""
        ...

    def count(self, text: str, kind: str = "document") -> int:
        """Count encoded tokens, including any query or document overhead."""
        ...

    def descriptor(self) -> dict[str, Any]:
        """Describe tokenizer provenance and benchmark compatibility."""
        ...


class DiagnosticTokenizer:
    """Word or character counters for offline passage-boundary and budget tests."""

    def __init__(self, mode: str = "word"):
        """Select a diagnostic counting rule with no model-token equivalence claim."""
        if mode not in {"word", "character"}:
            raise ConfigurationError("Diagnostic tokenizer must be word or character")
        self.mode = mode

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Return character spans using the selected diagnostic splitting rule."""
        if self.mode == "character":
            return [(i, i + 1) for i in range(len(text))]
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def count(self, text: str, kind: str = "document") -> int:
        """Count diagnostic units without encoder-specific special tokens."""
        return len(self.offsets(text))

    def descriptor(self) -> dict[str, Any]:
        """Describe the diagnostic tokenizer and its segmentation rule."""
        return {
            "implementation": "diagnostic",
            "mode": self.mode,
            "benchmark_compatible": False,
            "special_tokens": 0,
        }


class ColBERTTokenizer:
    """Pinned local fast tokenizer with offsets and ColBERT marker accounting.

    ColBERT inserts a query/document marker after tokenization. Count that token
    plus the encoder's normal special tokens. Loading
    requires a local directory and uses local_files_only, even in online runs.
    """

    def __init__(self, local_path: str | Path, revision: str | None = None):
        """Load and fingerprint a local fast tokenizer without downloading files."""
        path = Path(local_path).expanduser().resolve()
        if not path.is_dir():
            raise CapabilityError(f"Local ColBERT tokenizer directory is missing: {path}")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise CapabilityError("Install the colbert extra to load its local tokenizer") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), use_fast=True, local_files_only=True
        )
        if not self.tokenizer.is_fast:
            raise CapabilityError("A fast tokenizer is required for exact source offsets")
        self.path = str(path)
        self.revision = revision
        digest = hashlib.sha256()
        names = ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json")
        for name in names:
            candidate = path / name
            if candidate.exists():
                digest.update(name.encode())
                digest.update(candidate.read_bytes())
        self.sha256 = digest.hexdigest()

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Return untruncated content-token offsets without special-token spans."""
        result = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
            return_offsets_mapping=True,
        )
        return [(int(a), int(b)) for a, b in result["offset_mapping"] if b > a]

    def count(self, text: str, kind: str = "document") -> int:
        """Count full encoder tokens plus the inserted ColBERT query/document marker."""
        if kind not in {"document", "query"}:
            raise ConfigurationError("Token kind must be document or query")
        return 1 + len(
            self.tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)[
                "input_ids"
            ]
        )

    def descriptor(self) -> dict[str, Any]:
        """Report local tokenizer provenance and special-token accounting."""
        return {
            "implementation": "colbert-local-fast",
            "path": self.path,
            "revision": self.revision,
            "sha256": self.sha256,
            "benchmark_compatible": True,
            "marker_accounting": "special-tokens-plus-one-marker",
        }


def validate_encoder_text(
    text: str, tokenizer: OffsetTokenizer, limit: int, kind: str = "document"
) -> int:
    """Return the encoded size or reject overflow instead of truncating text."""
    size = tokenizer.count(text, kind=kind)
    if size > limit:
        raise ConfigurationError(
            f"{kind} encoder overflow: {size} tokens exceeds {limit}; truncation prohibited"
        )
    return size
