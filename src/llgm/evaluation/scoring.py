"""Diagnostic scores and explicit export for the separately invoked official judge."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from llgm.core.errors import ConfigurationError
from llgm.evaluation.artifacts import read_jsonl, write_jsonl


def exact_match_diagnostic(prediction: str, expected) -> bool:
    """Match a normalized answer against one or more accepted alternatives."""

    def normalize(value):
        """Fold Unicode and case, retaining word tokens for a punctuation-free comparison."""
        return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", str(value)).casefold()))

    alternatives = expected if isinstance(expected, list) else [expected]
    return any(normalize(prediction) == normalize(item) for item in alternatives)


def source_recall(retrieved_node_ids, expected_node_ids) -> float | None:
    """Return recall over unique expected sources, or None when none are labeled."""
    expected = set(expected_node_ids)
    if not expected:
        return None
    return len(set(retrieved_node_ids) & expected) / len(expected)


def evidence_coverage(references, gold: dict) -> dict:
    """Session/answer-turn hit coverage; no claim of exact answer-span recall."""
    aliases = gold.get("source_aliases", {})
    nodes = {aliases.get(ref.node_id, ref.node_id) for ref in references}
    turns = {(ref.node_id, ref.turn_id) for ref in references}
    abstention = gold.get("ability") == "abstention"
    expected_nodes = set() if abstention else set(gold["evidence_node_ids"])
    expected_turns = set() if abstention else {tuple(pair) for pair in gold["evidence_turn_ids"]}
    return {
        "source_recall": source_recall(nodes, expected_nodes),
        "answer_turn_recall": len(turns & expected_turns) / len(expected_turns)
        if expected_turns
        else None,
        "all_required_sources_covered": expected_nodes.issubset(nodes) if expected_nodes else None,
        "all_answer_turns_hit": expected_turns.issubset(turns) if expected_turns else None,
        "source_denominator": len(expected_nodes),
        "answer_turn_denominator": len(expected_turns),
        "annotation_granularity": "session and turn labels; partial passage hits do not prove answer-span coverage",
    }


def export_official_predictions(predictions_path: str | Path, output_path: str | Path) -> None:
    """Export one arm's answers for the official judge, rejecting missing or duplicate IDs."""
    predictions = read_jsonl(predictions_path)
    seen = set()
    exported = []
    for row in predictions:
        case_id = row.get("question_id", row.get("case_id"))
        if not case_id or case_id in seen:
            raise ConfigurationError(
                "Official export requires exactly one prediction per question, from one arm"
            )
        seen.add(case_id)
        exported.append(
            {"question_id": case_id, "hypothesis": row.get("hypothesis", row.get("answer", ""))}
        )
    write_jsonl(output_path, exported)


def official_scorer_command(
    script_path: str | Path,
    judge_model: str,
    predictions_path: str | Path,
    original_dataset_path: str | Path,
) -> list[str]:
    """Return arguments; never launch a paid judge implicitly."""
    import sys

    if not Path(script_path).is_file() or not judge_model:
        raise ConfigurationError("Provide a local official evaluate_qa.py and a pinned judge model")
    return [
        sys.executable,
        str(script_path),
        judge_model,
        str(predictions_path),
        str(original_dataset_path),
    ]
