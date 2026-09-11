"""Answers and evidence returned by bounded inference."""

from dataclasses import dataclass, field


@dataclass
class EvidenceBundle:
    """Collected evidence, canonical references, unresolved needs, and execution records."""

    text: str = ""
    references: tuple = ()
    hits: list = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    stop_reason: str = "completed"
    usage: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)


@dataclass
class AnswerResult:
    """An answer with evidence and observable execution status.

    Unresolved evidence produces ``partial``. Expected operational failures
    produce an empty answer with ``failed`` or ``budget_exhausted``. Invalid
    public arguments, unexpected errors, and cancellation still raise.
    """

    answer: str
    evidence: EvidenceBundle
    usage: dict
    trace: list[dict]
    status: str = "completed"

    @property
    def references(self):
        """Expose the canonical references carried by the evidence bundle."""
        return self.evidence.references
