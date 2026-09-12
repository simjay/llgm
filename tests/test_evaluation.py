"""Evaluation preparation, evidence-label isolation and diagnostic scoring contracts."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from llgm.core.errors import ConfigurationError, SchemaError
from llgm.core.types import SourceSpan
from llgm.evaluation.artifacts import read_jsonl, write_jsonl
from llgm.evaluation.longmemeval import history_groups, parse_longmemeval, select_development
from llgm.evaluation.prepare import controlled_dataset, diagnostic_dataset
from llgm.evaluation.scoring import (
    evidence_coverage,
    exact_match_diagnostic,
    export_official_predictions,
    source_recall,
)


class LongMemEvalTests(unittest.TestCase):
    """LongMemEval parsing and history-isolated selection invariants."""

    def test_gold_labels_do_not_enter_sources(self):
        """Only source text and allowed metadata cross the evaluator boundary."""
        raw = diagnostic_dataset()
        raw[0]["haystack_sessions"][0][0]["private_evaluator_metadata"] = "secret-label"
        cases, gold = parse_longmemeval(raw)
        serialized = json.dumps(
            [
                {"text": turn.text, "metadata": source.metadata}
                for case in cases
                for source in case.sources
                for turn in source.turns
            ]
        )
        self.assertNotIn("has_answer", serialized)
        self.assertNotIn("private_evaluator_metadata", serialized)
        self.assertNotIn("secret-label", serialized)
        self.assertEqual(
            gold[cases[0].case_id].evidence_turn_ids, (("diagnostic-session-0", "t00000"),)
        )

    def test_parser_preserves_unicode_and_roles(self):
        """Parsing leaves Unicode, whitespace and assistant roles intact."""
        raw = diagnostic_dataset()
        raw[0]["haystack_sessions"][0][0]["content"] = "  naïve 🦋 한글\n"
        cases, _ = parse_longmemeval(raw)
        self.assertEqual(cases[0].sources[0].turns[0].text, "  naïve 🦋 한글\n")
        self.assertEqual(cases[0].sources[0].turns[1].role, "assistant")

    def test_abstention_suffix_overrides_type(self):
        """The abstention suffix determines ability even when the nominal type differs."""
        cases, gold = parse_longmemeval(controlled_dataset(6))
        self.assertEqual(gold["controlled-0003_abs"].ability, "abstention")
        self.assertEqual(gold["controlled-0003_abs"].evidence_node_ids, ())

    def test_rejects_misaligned_arrays_and_duplicate_ids(self):
        """Inconsistent haystack arrays and repeated question IDs are rejected."""
        raw = diagnostic_dataset()
        raw[0]["haystack_dates"] = []
        with self.assertRaises(SchemaError):
            parse_longmemeval(raw)
        raw = diagnostic_dataset()
        raw.append(raw[0])
        with self.assertRaises(SchemaError):
            parse_longmemeval(raw)

    def test_rejects_gold_outside_case_corpus(self):
        """Gold evidence must belong to the question's supplied history."""
        raw = diagnostic_dataset()
        raw[0]["answer_session_ids"] = ["other-case-node"]
        with self.assertRaises(SchemaError):
            parse_longmemeval(raw)

    def test_repeated_session_ids_preserve_occurrences_and_credit_one_source(self):
        """Repeated session occurrences keep dates distinct while source recall counts the alias once."""
        from dataclasses import asdict

        raw = diagnostic_dataset()[:1]
        row = raw[0]
        row["haystack_session_ids"].append(row["haystack_session_ids"][0])
        row["haystack_dates"].append("2026/09/09 (Wed) 10:00")
        row["haystack_sessions"].append(copy.deepcopy(row["haystack_sessions"][0]))
        cases, evaluator = parse_longmemeval(raw)
        left, right = cases[0].sources
        self.assertNotEqual(left.node_id, right.node_id)
        self.assertNotEqual(left.metadata["date"], right.metadata["date"])
        self.assertEqual(left.turns, right.turns)
        gold = asdict(evaluator[cases[0].case_id])
        metrics = evidence_coverage([SourceSpan(left.node_id, "t00000", 0, 5)], gold)
        self.assertEqual(metrics["source_recall"], 1.0)
        self.assertEqual(metrics["source_denominator"], 1)
        self.assertEqual(metrics["answer_turn_recall"], 0.5)
        self.assertEqual(len(gold["source_aliases"]), 2)

    def test_history_components_are_transitive(self):
        """Shared sessions connect history groups transitively before split selection."""
        raw = diagnostic_dataset()[:3]
        for target, source in ((0, 1), (1, 2)):
            raw[target]["haystack_session_ids"].append(raw[source]["haystack_session_ids"][0])
            raw[target]["haystack_dates"].append(raw[source]["haystack_dates"][0])
            raw[target]["haystack_sessions"].append(raw[source]["haystack_sessions"][0])
        cases, gold = parse_longmemeval(raw)
        self.assertEqual(len(history_groups(cases)), 1)
        selection = select_development(cases, gold)
        self.assertEqual(selection["status"], "blocked_history_isolation")
        self.assertEqual(selection["development_ids"], [])
        self.assertTrue(selection["requires_separate_development_data"])

    def test_identical_content_under_new_id_is_same_history(self):
        """Renaming identical session content cannot create an independent history group."""
        raw = diagnostic_dataset()[:2]
        raw[1]["haystack_sessions"] = copy.deepcopy(raw[0]["haystack_sessions"])
        cases, _ = parse_longmemeval(raw)
        self.assertEqual(len(history_groups(cases)), 1)

    def test_split_is_deterministic_and_group_isolated(self):
        """Selection is stable under input reordering and keeps whole history groups together."""
        cases, gold = parse_longmemeval(controlled_dataset())
        first = select_development(cases, gold)
        second = select_development(list(reversed(cases)), gold)
        self.assertEqual(first, second)
        self.assertEqual(len(first["development_ids"]), 50)
        self.assertEqual(len(first["smoke_ids"]), 5)
        self.assertEqual(len(first["held_out_ids"]), 10)
        self.assertFalse(set(first["development_ids"]) & set(first["held_out_ids"]))
        self.assertTrue(set(first["smoke_ids"]).issubset(first["development_ids"]))
        self.assertIn("abstention", first["ability_counts"])


class ScoringTests(unittest.TestCase):
    """Diagnostic scoring boundaries and official prediction export contracts."""

    def test_exact_match_is_only_normalized_diagnostic(self):
        """Normalized exact match remains literal rather than inferring semantic equivalence."""
        self.assertTrue(exact_match_diagnostic("  VIOLET!", "violet"))
        self.assertTrue(exact_match_diagnostic("blue", ["azure", "blue"]))
        self.assertFalse(exact_match_diagnostic("It is violet.", "violet"))

    def test_source_recall_has_explicit_empty_denominator(self):
        """Recall with no positive evidence has no numeric denominator."""
        self.assertIsNone(source_recall(["x"], []))
        self.assertEqual(source_recall(["x", "x"], ["x", "y"]), 0.5)

    def test_official_export_keeps_failures_and_rejects_mixed_arms(self):
        """Official export retains failed cases and rejects duplicate predictions per question."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(
                root / "predictions.jsonl",
                [
                    {"case_id": "a", "hypothesis": "answer", "status": "completed"},
                    {"case_id": "b", "hypothesis": "", "status": "failed"},
                ],
            )
            export_official_predictions(root / "predictions.jsonl", root / "official.jsonl")
            self.assertEqual(
                read_jsonl(root / "official.jsonl"),
                [
                    {"question_id": "a", "hypothesis": "answer"},
                    {"question_id": "b", "hypothesis": ""},
                ],
            )
            write_jsonl(root / "predictions.jsonl", [{"case_id": "a"}, {"case_id": "a"}])
            with self.assertRaises(ConfigurationError):
                export_official_predictions(root / "predictions.jsonl", root / "official.jsonl")


if __name__ == "__main__":
    unittest.main()
