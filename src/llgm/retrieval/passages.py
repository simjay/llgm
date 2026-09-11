"""Render shared passages while retaining exact, turn-relative Unicode offsets."""

from __future__ import annotations

import hashlib
from typing import Iterable

from llgm.core.errors import ConfigurationError
from llgm.core.types import SourceNode, SourceSpan
from llgm.retrieval.base import SearchPassage
from llgm.retrieval.tokenizers import OffsetTokenizer


def split_nodes(
    nodes: Iterable[SourceNode], tokenizer: OffsetTokenizer, window: int = 180, overlap: int = 32
) -> list[SearchPassage]:
    """Build overlapping, metadata-prefixed passages within the rendered token limit."""
    if window <= 0 or overlap < 0 or overlap >= window:
        raise ConfigurationError("Require 0 <= overlap < window")
    result = []
    for node in nodes:
        date = str(node.metadata.get("date", "unknown"))
        for turn in node.turns:
            header = f"[session {node.node_id}; date {date}; role {turn.role}]\n"
            offsets = tokenizer.offsets(turn.text)
            if not offsets:
                if turn.text:
                    rendered = header + turn.text
                    if tokenizer.count(rendered) > window:
                        raise ConfigurationError("Passage metadata exceeds window")
                    result.append(_passage(node, turn, 0, len(turn.text), rendered, tokenizer))
                continue
            position = 0
            while position < len(offsets):
                start = 0 if position == 0 else offsets[position][0]
                last = min(len(offsets), position + window)
                while last > position:
                    end = offsets[last][0] if last < len(offsets) else len(turn.text)
                    rendered = header + turn.text[start:end]
                    if tokenizer.count(rendered) <= window:
                        break
                    last -= 1
                if last == position:
                    raise ConfigurationError(
                        "Metadata plus one source token exceeds passage window"
                    )
                result.append(_passage(node, turn, start, end, rendered, tokenizer))
                if last == len(offsets):
                    break
                position = max(position + 1, last - overlap)
    return result


def _passage(node, turn, start, end, rendered, tokenizer):
    """Bind rendered passage text to its exact turn-relative source span and identity."""
    span = SourceSpan(node.node_id, turn.turn_id, start, end)
    identity = f"{node.node_id}:{turn.turn_id}:{start}:{end}:{rendered}"
    return SearchPassage(
        "p-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
        rendered,
        (span,),
        {
            "node_id": node.node_id,
            "turn_id": turn.turn_id,
            "role": turn.role,
            "date": node.metadata.get("date"),
            "timestamp_ms": node.timestamp_ms,
            "rendered_tokens": tokenizer.count(rendered),
        },
    )
