"""Large Language Graphical Model: persistent conversations and recursive reading.

Continue conversations with ``LLGM.answer()``. Import earlier history with
``LLGM.ingest()``.
Local model readers return selected findings and source excerpts to a final
answer model. References keep the supporting original text inspectable.

Importing the package does not create files, load model weights, or access a
network. Provider and storage resources are created explicitly by applications.
"""

from llgm.core.config import Settings
from llgm.core.environment import load_env_file
from llgm.core.time import parse_instant_ms
from llgm.core.types import (
    Conversation,
    Edge,
    IngestResult,
    JournalEntry,
    JournalRef,
    NodeRef,
    Provenance,
    ResolvedEvidence,
    SourceNode,
    SourceSpan,
    Turn,
)
from llgm.inference.budget import Budget
from llgm.inference.results import AnswerResult
from llgm.llgm import LLGM, IngestionOutcome
from llgm.memory.evidence import Evidence
from llgm.memory.maintenance import MaintenancePolicy, MaintenanceResult
from llgm.memory.workspace import Workspace

__all__ = [
    "LLGM",
    "Settings",
    "load_env_file",
    "parse_instant_ms",
    "Workspace",
    "Conversation",
    "Edge",
    "Turn",
    "SourceNode",
    "SourceSpan",
    "NodeRef",
    "JournalRef",
    "JournalEntry",
    "Provenance",
    "ResolvedEvidence",
    "IngestResult",
    "Budget",
    "AnswerResult",
    "Evidence",
    "MaintenancePolicy",
    "MaintenanceResult",
    "IngestionOutcome",
]
