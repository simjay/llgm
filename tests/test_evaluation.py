"""Evaluation preparation, evidence-label isolation and diagnostic scoring contracts."""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path

from llgm.core.errors import ConfigurationError, SchemaError
from llgm.core.types import SourceSpan
from llgm.evaluation.artifacts import read_jsonl, write_json, write_jsonl
from llgm.evaluation.longmemeval import history_groups, parse_longmemeval, select_development
from llgm.evaluation.matrix import (
    REQUIRED_ARMS,
    matrix_template,
    preflight,
    tokenizer_settings,
    validate_matrix,
)
from llgm.evaluation.prepare import controlled_dataset, diagnostic_dataset, prepare
from llgm.evaluation.runner import offline_smoke
from llgm.evaluation.scoring import (
    evidence_coverage,
    exact_match_diagnostic,
    export_official_predictions,
    source_recall,
)
from llgm.retrieval.base import passage_from_dict


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


class PreparationAndMatrixTests(unittest.TestCase):
    """Experiment matrix validation and auditable preparation outputs."""

    def test_required_matrix_has_twelve_unique_arms(self):
        """The required matrix contains every declared retrieval/policy arm exactly once."""
        template = matrix_template()
        validate_matrix(template)
        self.assertEqual({arm["id"] for arm in template["arms"]}, set(REQUIRED_ARMS))
        template["arms"] = [arm for arm in template["arms"] if arm["backend"] != "C"]
        with self.assertRaises(ConfigurationError):
            validate_matrix(template)

    def test_backend_identity_and_resource_contract_are_checked(self):
        """Backend labels cannot silently change the frozen experiment resource contract."""
        for modify in (
            lambda x: x["arms"][0].update(backend="D"),
            lambda x: x["hybrid"].update(pool_size=10),
            lambda x: x["passages"].update(window=200),
            lambda x: x["budgets"].update(max_searches=1),
        ):
            value = matrix_template()
            modify(value)
            with self.assertRaises(ConfigurationError):
                validate_matrix(value)

    def test_standalone_tokenizer_pins_override_encoder_assets(self):
        """Tokenizer identity stays distinct from optional encoder checkpoint and code pins."""
        matrix = matrix_template()
        matrix["colbert"].update(
            checkpoint_path="encoder-checkpoint", repository_revision="encoder-code-revision"
        )
        matrix["tokenizer"] = {
            "local_path": "tokenizer-only-files",
            "revision": "tokenizer-release-revision",
        }
        validate_matrix(matrix)
        resolved = tokenizer_settings(matrix)
        self.assertEqual(resolved, matrix["tokenizer"])
        resolved["revision"] = "mutated-copy"
        self.assertEqual(matrix["tokenizer"]["revision"], "tokenizer-release-revision")
        self.assertEqual(matrix["colbert"]["repository_revision"], "encoder-code-revision")

    def test_unset_tokenizer_preserves_legacy_matrix_compatibility(self):
        """Absent, empty and fully unset tokenizer sections retain legacy checkpoint loading."""
        for config in (None, {}, {"local_path": None, "revision": None}):
            with self.subTest(config=config):
                matrix = matrix_template()
                matrix["colbert"].update(
                    checkpoint_path="legacy-checkpoint", repository_revision="legacy-revision"
                )
                if config is None:
                    matrix.pop("tokenizer")
                else:
                    matrix["tokenizer"] = config
                validate_matrix(matrix)
                self.assertEqual(
                    tokenizer_settings(matrix),
                    {"local_path": "legacy-checkpoint", "revision": "legacy-revision"},
                )

    def test_partial_tokenizer_pins_fail_before_preflight(self):
        """Partial, wrongly typed and unknown tokenizer pins cannot select legacy assets silently."""
        for config in (
            None,
            [],
            {"local_path": "tokenizer"},
            {"revision": "revision"},
            {"local_path": None},
            {"local_path": "tokenizer", "revision": None},
            {"local_path": None, "revision": "revision"},
            {"local_path": "tokenizer", "revision": " "},
            {"local_path": 5, "revision": "revision"},
            {"local_path": "tokenizer", "revision": "revision", "extra": None},
            {"unexpected": None},
        ):
            with self.subTest(config=config):
                matrix = matrix_template()
                matrix["tokenizer"] = config
                for check in (validate_matrix, tokenizer_settings):
                    with self.assertRaises(ConfigurationError):
                        check(matrix)

    def test_preparation_separates_gold_and_preserves_per_case_corpus(self):
        """Prepared case corpora omit evaluator labels and retain separate gold records."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "input.json", diagnostic_dataset())
            manifest = prepare(root / "input.json", root / "prepared", target=5)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["evidence_schema"], "immutable-node-v2")
            queries = read_jsonl(root / "prepared/queries.jsonl")
            self.assertEqual(len(queries), 5)
            for query in queries:
                passages = read_jsonl(root / "prepared" / query["passages_path"])
                self.assertEqual(len({p["refs"][0]["node_id"] for p in passages}), 1)
                self.assertNotIn("has_answer", json.dumps(passages))
                self.assertNotIn("answer_session_ids", json.dumps(passages))
                for raw in passages:
                    self.assertNotIn("source_version", raw["refs"][0])
                    decoded = passage_from_dict(raw)
                    self.assertEqual(decoded.refs[0], SourceSpan(**raw["refs"][0]))
            self.assertEqual(len(read_jsonl(root / "prepared/evaluator/gold.jsonl")), 6)
            self.assertIn("queries.jsonl", manifest["files"])
            with self.assertRaises(ConfigurationError):
                prepare(root / "input.json", root / "prepared")

    def test_evaluation_reservation_does_not_create_development(self):
        """Reserving evaluation cases does not turn held-out data into development data."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "input.json", diagnostic_dataset())
            manifest = prepare(
                root / "input.json", root / "prepared", selection_mode="evaluation", case_limit=2
            )
            self.assertEqual(len(manifest["selection"]["held_out_ids"]), 6)
            self.assertEqual(len(manifest["selection"]["evaluation_ids"]), 2)
            self.assertEqual(manifest["selection"]["development_ids"], [])
            self.assertTrue(manifest["selection"]["freeze_required"])
            self.assertEqual(len(read_jsonl(root / "prepared/queries.jsonl")), 2)

    def test_preflight_includes_colbert_and_labels_diagnostic(self):
        """Preflight exposes missing ColBERT assets and distinguishes diagnostic readiness."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "input.json", diagnostic_dataset())
            prepare(
                root / "input.json",
                root / "prepared",
                target=5,
                dataset_kind="controlled-offline-fixture",
            )
            full = preflight(root / "prepared", matrix_template())
            self.assertFalse(full["ready"])
            self.assertEqual(len(full["arms"]), 12)
            self.assertTrue(
                all(arm["status"] == "blocked" for arm in full["arms"] if arm["backend"] == "C")
            )
            diagnostic = preflight(root / "prepared", matrix_template(), diagnostic=True)
            self.assertTrue(diagnostic["ready"])
            self.assertFalse(diagnostic["benchmark_ready"])
            self.assertEqual([a["id"] for a in diagnostic["arms"] if a["selected"]], ["B-S"])

    def test_preflight_rejects_legacy_manifest_without_decoding_or_rewriting(self):
        """A v1 preparation cannot be treated as current evidence even when its files are accessible."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "input.json", diagnostic_dataset())
            manifest = prepare(
                root / "input.json",
                root / "prepared",
                target=5,
                dataset_kind="controlled-offline-fixture",
            )
            manifest["schema_version"] = 1
            manifest.pop("evidence_schema")
            path = root / "prepared/prepared.json"
            write_json(path, manifest)
            original = path.read_bytes()
            with patch("llgm.evaluation.matrix.passage_from_dict") as decoder:
                report = preflight(root / "prepared", matrix_template(), diagnostic=True)
            self.assertFalse(report["ready"])
            self.assertTrue(any("require schema_version=2" in error for error in report["errors"]))
            self.assertEqual(report["network_calls"], 0)
            decoder.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_legacy_reference_is_rejected_even_under_a_relabelled_manifest(self):
        """Changing a manifest number cannot silently discard an old source-edition coordinate."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "input.json", diagnostic_dataset())
            prepare(
                root / "input.json",
                root / "prepared",
                target=5,
                dataset_kind="controlled-offline-fixture",
            )
            query = read_jsonl(root / "prepared/queries.jsonl")[0]
            path = root / "prepared" / query["passages_path"]
            passages = read_jsonl(path)
            passages[0]["refs"][0]["source_version"] = 1
            with self.assertRaisesRegex(TypeError, "source_version"):
                passage_from_dict(passages[0])
            write_jsonl(path, passages)
            report = preflight(root / "prepared", matrix_template(), diagnostic=True)
            self.assertFalse(report["ready"])
            self.assertTrue(
                any(
                    "source_version" in error and "Regenerate" in error
                    for error in report["errors"]
                )
            )
            self.assertEqual(read_jsonl(path)[0]["refs"][0]["source_version"], 1)

    def test_offline_smoke_writes_all_artifacts(self):
        """Offline smoke produces complete artifacts without embedding requests."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = asyncio.run(offline_smoke(root / "smoke"))
            self.assertEqual(result["physical_embedding_requests"], 0)
            self.assertEqual(result["arms"]["B-S"]["completed"], 5)
            self.assertFalse(result["benchmark_result"])
            for filename in (
                "manifest.json",
                "cases.jsonl",
                "traces.jsonl",
                "usage.jsonl",
                "predictions.jsonl",
                "judgments.jsonl",
                "metrics.json",
                "decision.md",
            ):
                self.assertTrue((root / "smoke/run/B-S" / filename).exists())


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
