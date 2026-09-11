"""Check official prompt isolation, blinded support inputs, and strict judge parsing."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from llgm.core.errors import ConfigurationError, SchemaError
from llgm.core.types import SourceNode, SourceSpan, Turn, reference_to_dict
from llgm.evaluation.answer_judging import (
    CONTROLLED_ABILITIES,
    OFFICIAL_ABILITIES,
    SUPPORT_SYSTEM_PROMPT,
    SUPPORT_VERDICTS,
    accuracy_request,
    cited_source_evidence,
    load_official_prompt,
    parse_accuracy,
    parse_support,
    support_request,
)
from llgm.evaluation.longmemeval import EvaluationCase, GoldRecord
from llgm.inference.results import AnswerResult, EvidenceBundle
from llgm.models.base import Message

PROMPT_SOURCE = '''
import deliberately_unavailable_evaluator_dependency
raise RuntimeError("Top-level code must not execute")

@decorator_that_must_not_execute()
def unrelated():
    """Stand in for the evaluator's decorated provider-call helper."""
    raise RuntimeError("Unrelated function must not execute")

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    """Expose every official prompt argument in a dependency-free fixture."""
    if abstention:
        return "Abstention: {} | {} | {}".format(question, answer, response)
    return "{}: {} | {} | {}".format(task, question, answer, response)
'''
OFFICIAL_SHA256 = "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"


def _write_prompt(tmp_path, source=PROMPT_SOURCE):
    """Create trusted fixture bytes and the checksum expected by the loader."""
    path = tmp_path / "evaluate_qa.py"
    path.write_text(source, encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _case_gold(ability="multi-session"):
    """Keep labels, question identity, and the question date observably separate."""
    case = EvaluationCase("PRIVATE_CASE_ID", "Which café?\n  Keep spacing.", "PRIVATE_DATE", ())
    gold = GoldRecord(
        case.case_id,
        "GOLD_PRIVATE 🦋",
        ability,
        ("GOLD_NODE_PRIVATE",),
        (),
        {"OPAQUE": "GOLD_ALIAS_PRIVATE"},
    )
    return case, gold


def _evidence():
    """Return canonical quotes carrying metadata that judges must never receive."""
    quote = "Café 🦋 opens Sunday.\n  Keep spacing."
    return [
        {
            "id": "e1",
            "text": quote,
            "references": [reference_to_dict(SourceSpan("n-a", "t1", 3, 3 + len(quote)))],
            "role": "user",
            "date": "2026-09-01",
            "metadata": {"gold": "GOLD_PRIVATE", "backend": "BACKEND_PRIVATE"},
            "selector_reason": "SELECTOR_PRIVATE",
            "score": 42,
        },
        {
            "id": "e2",
            "text": "Try Monday?",
            "references": [reference_to_dict(SourceSpan("n-b", "t2", 0, 11))],
            "role": "assistant",
            "date": None,
        },
    ]


def test_loader_executes_only_the_pinned_function(tmp_path):
    """Evaluator imports, CLI code, and unrelated decorators cannot run during loading."""
    path, checksum = _write_prompt(tmp_path)
    prompt = load_official_prompt(path, checksum)
    assert prompt("multi-session", "Q", "A", "R") == "multi-session: Q | A | R"
    assert prompt("ignored", "Q", None, "R", abstention=True) == "Abstention: Q | None | R"
    assert set(prompt.__globals__) == {"__builtins__", "get_anscheck_prompt"}


def test_loader_checks_file_bytes_before_parsing(tmp_path):
    """An edited file fails its checksum even when it also contains invalid Python."""
    path, checksum = _write_prompt(tmp_path)
    path.write_text("not valid Python !!!")
    with pytest.raises(ConfigurationError, match="SHA256"):
        load_official_prompt(path, checksum)


@pytest.mark.parametrize(
    "source",
    [
        "not valid Python !!!",
        "def other():\n    pass\n",
        PROMPT_SOURCE + PROMPT_SOURCE,
        PROMPT_SOURCE.replace("def get_anscheck_prompt", "@untrusted()\ndef get_anscheck_prompt"),
        PROMPT_SOURCE.replace("abstention=False", "abstention=untrusted()"),
        PROMPT_SOURCE.replace("task, question", "task: untrusted(), question"),
        PROMPT_SOURCE.replace("response, abstention", "response, extra, abstention"),
        PROMPT_SOURCE.replace("    if abstention:", "    import os\n    if abstention:"),
    ],
)
def test_loader_rejects_changed_function_contracts(tmp_path, source):
    """Pins do not silently admit duplicate functions or executable definition metadata."""
    path, checksum = _write_prompt(tmp_path, source)
    with pytest.raises(ConfigurationError):
        load_official_prompt(path, checksum)


def test_loader_reports_missing_source(tmp_path):
    """Missing external evaluator data fails without importing a replacement scorer."""
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_official_prompt(tmp_path / "missing.py", OFFICIAL_SHA256)


@pytest.mark.parametrize("ability", sorted(OFFICIAL_ABILITIES))
def test_accuracy_preserves_official_arguments_and_chat_shape(tmp_path, ability):
    """All released abilities use exactly one original-format user prompt without a schema."""
    path, checksum = _write_prompt(tmp_path)
    prompt = load_official_prompt(path, checksum)
    case, gold = _case_gold(ability)
    answer = "The answer\n  with unchanged formatting."
    request = accuracy_request(case, gold, answer, prompt)
    assert request.messages == (
        Message("user", f"{ability}: {case.question} | {gold.answer} | {answer}"),
    )
    assert request.max_output_tokens == 10
    assert request.temperature == 0
    assert request.output_schema is None
    for private in (case.case_id, case.question_date, "GOLD_NODE_PRIVATE", "GOLD_ALIAS_PRIVATE"):
        assert private not in request.messages[0].content


@pytest.mark.parametrize("ability", sorted(CONTROLLED_ABILITIES))
def test_controlled_positive_abilities_use_explicit_template_adaptation(tmp_path, ability):
    """Authored diagnostic abilities map explicitly to the multi-session rubric."""
    path, checksum = _write_prompt(tmp_path)
    case, gold = _case_gold(ability)
    request = accuracy_request(case, gold, "A", load_official_prompt(path, checksum))
    assert request.messages[0].content == f"multi-session: {case.question} | {gold.answer} | A"


def test_abstention_comes_from_gold_ability_not_identifier(tmp_path):
    """Opaque case naming cannot change the abstention rubric or invent a gold explanation."""
    path, checksum = _write_prompt(tmp_path)
    prompt = load_official_prompt(path, checksum)
    case, gold = _case_gold("abstention")
    request = accuracy_request(case, replace(gold, answer=None), "Unknown", prompt)
    assert request.messages[0].content == f"Abstention: {case.question} | None | Unknown"
    case = replace(case, case_id="looks_abs")
    gold = replace(gold, case_id=case.case_id, ability="multi-session")
    assert (
        accuracy_request(case, gold, "A", prompt).messages[0].content.startswith("multi-session:")
    )


@pytest.mark.parametrize("problem", ["case", "ability", "answer"])
def test_accuracy_rejects_mismatched_or_unsupported_inputs(tmp_path, problem):
    """Invalid records cannot silently receive another question's gold or an invented rubric."""
    path, checksum = _write_prompt(tmp_path)
    case, gold = _case_gold()
    answer = "A"
    if problem == "case":
        gold = replace(gold, case_id="other")
    elif problem == "ability":
        gold = replace(gold, ability="unknown")
    else:
        answer = None
    with pytest.raises(ConfigurationError):
        accuracy_request(case, gold, answer, load_official_prompt(path, checksum))


@pytest.mark.parametrize("text, expected", [("yes", True), (" NO.\n", False), ("Yes.", True)])
def test_accuracy_accepts_only_normalized_single_verdicts(text, expected):
    """Whitespace, case, and one terminal period do not alter a standalone verdict."""
    assert parse_accuracy(text) is expected


@pytest.mark.parametrize(
    "text", [None, True, "", "yes..", "yes or no", "not yes", "yesterday", '"yes"']
)
def test_accuracy_rejects_substrings_and_extra_explanation(text):
    """Unexpected judge text fails instead of being counted as an incorrect or correct answer."""
    with pytest.raises(SchemaError):
        parse_accuracy(text)


def test_support_is_blind_to_labels_rankings_and_runtime_metadata():
    """Only exact final citations and source provenance enter the support judgment."""
    evidence = _evidence()
    before = deepcopy(evidence)
    request = support_request("Which café?", "2026-09-11", "Sunday [e1].", evidence)
    assert request.messages[0] == Message("system", SUPPORT_SYSTEM_PROMPT)
    payload = json.loads(request.messages[1].content)
    assert payload == {
        "question": "Which café?",
        "question_date": "2026-09-11",
        "answer": "Sunday [e1].",
        "cited_evidence": [
            {key: record[key] for key in ("id", "text", "references", "role", "date")}
            for record in evidence
        ],
    }
    assert evidence == before
    for private in ("GOLD_PRIVATE", "BACKEND_PRIVATE", "SELECTOR_PRIVATE", '"score"'):
        assert private not in request.messages[1].content
    assert request.temperature == 0
    assert request.max_output_tokens == 512
    assert request.output_schema["required"] == ["verdict", "reason"]
    assert request.output_schema["additionalProperties"] is False
    assert request.output_schema["properties"]["verdict"]["enum"] == list(SUPPORT_VERDICTS)


def test_support_prompt_distinguishes_no_citations_from_no_asserted_facts():
    """An empty citation list remains visible to the judge without assigning a fake verdict."""
    substantive = support_request("Q", "", "The code is 123456.", [])
    abstention = support_request("Q", "", "I do not have the code.", [])
    assert substantive.messages[0] == abstention.messages[0]
    assert json.loads(substantive.messages[1].content)["cited_evidence"] == []
    assert "a substantive factual answer with no citations is unsupported" in SUPPORT_SYSTEM_PROMPT
    assert "empty response with no asserted material facts" in SUPPORT_SYSTEM_PROMPT
    assert "suggestion" in SUPPORT_SYSTEM_PROMPT
    assert "calculations" in SUPPORT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "mutation",
    [
        {"id": ""},
        {"text": None},
        {"text": "wrong length"},
        {"role": None},
        {"date": 123},
        {"references": []},
        {"references": [None]},
        {"references": [{"type": "node", "node_id": "n-a"}]},
        {"references": [{"type": "source_span", "node_id": "n-a"}]},
    ],
)
def test_support_rejects_missing_or_inconsistent_provenance(mutation):
    """Malformed citations cannot be silently treated as grounded exact quotes."""
    evidence = _evidence()
    evidence[0].update(mutation)
    with pytest.raises(ConfigurationError):
        support_request("Q", "D", "A", evidence)


@pytest.mark.parametrize("evidence", [None, [None], _evidence()[:1] * 2])
def test_support_rejects_invalid_lists_and_duplicate_citation_ids(evidence):
    """Support payloads require a real list with unambiguous citation identities."""
    with pytest.raises(ConfigurationError):
        support_request("Q", "D", "A", evidence)


def test_support_rejects_nontext_question_inputs():
    """Missing dates or answers are not silently converted into model-visible claims."""
    with pytest.raises(ConfigurationError):
        support_request("Q", None, "A", [])


@pytest.mark.parametrize("verdict", SUPPORT_VERDICTS)
def test_support_parser_preserves_each_valid_verdict(verdict):
    """All declared categories survive parsing without merging ambiguity into correctness."""
    value = {"verdict": verdict, "reason": "Exact reason."}
    assert parse_support(json.dumps(value)) == value


@pytest.mark.parametrize(
    "text",
    [
        None,
        "not JSON",
        "[]",
        "{}",
        '{"verdict":"yes","reason":"bad enum"}',
        '{"verdict":null,"reason":"bad type"}',
        '{"verdict":"supported","reason":false}',
        '{"verdict":"supported","reason":"ok","gold":"private"}',
        '{"verdict":"supported","verdict":"unsupported","reason":"duplicate"}',
        '```json\n{"verdict":"supported","reason":"fenced"}\n```',
    ],
)
def test_support_parser_rejects_repairs_missing_fields_and_ambiguous_outputs(text):
    """Malformed responses remain failures rather than repaired or coerced judgments."""
    with pytest.raises(SchemaError):
        parse_support(text)


@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    os.environ.get("LLGM_TEST_OFFICIAL_PROMPT") != "1",
    reason="Set LLGM_TEST_OFFICIAL_PROMPT=1 to verify the pinned local evaluator without model calls",
)
def test_actual_pinned_official_prompt_matches_extracted_source_verbatim():
    """Real released prompts match across every ability and the abstention adaptation."""
    path = (
        Path(__file__).resolve().parents[1]
        / "data/evaluators/LongMemEval/src/evaluation/evaluate_qa.py"
    )
    prompt = load_official_prompt(path, OFFICIAL_SHA256)
    source = path.read_text(encoding="utf-8")
    definition = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_anscheck_prompt"
    )
    namespace = {}
    exec(ast.get_source_segment(source, definition), namespace)
    expected = namespace["get_anscheck_prompt"]
    for ability in sorted(OFFICIAL_ABILITIES | CONTROLLED_ABILITIES | {"abstention"}):
        case, gold = _case_gold(ability)
        task = "multi-session" if ability in CONTROLLED_ABILITIES else ability
        answer = "An answer with Unicode 🦋 and spacing.\n"
        request = accuracy_request(case, gold, answer, prompt)
        assert request.messages == (
            Message(
                "user",
                expected(
                    task, case.question, gold.answer, answer, abstention=ability == "abstention"
                ),
            ),
        )


@pytest.mark.parametrize(
    "failure", ["json", "text", "bounds", "absent", "type", "duplicate", "union", "role", "shape"]
)
def test_source_evidence_rejects_corrupt_citations(failure):
    """Artifact projection cannot invent citation support from malformed or unrelated evidence."""
    text = "Jade uses eu-west-1.\n\nExact spacing 🦋."
    case = EvaluationCase(
        "q",
        "Which region?",
        "2031-04-07",
        (SourceNode("a", (Turn("first", "user", text),), {"date": "2031-04-01"}),),
    )
    span = SourceSpan("a", "first", 0, len(text))
    record = {
        "id": "e1",
        "text": text,
        "references": [reference_to_dict(span)],
        "metadata": {"role": "user"},
    }
    result = AnswerResult("Answer", EvidenceBundle(json.dumps(record), (span,)), {}, [])
    if failure == "text":
        record["text"] = "wrong"
    elif failure == "bounds":
        record["references"][0]["end"] = 10000
    elif failure == "absent":
        record["references"][0]["node_id"] = "missing"
    elif failure == "type":
        record["references"] = [{"type": "node", "node_id": "a"}]
    elif failure == "union":
        result.evidence.references = ()
    elif failure == "role":
        record["metadata"]["role"] = "assistant"
    elif failure == "shape":
        record["id"] = None
    result.evidence.text = "not JSON" if failure == "json" else json.dumps(record)
    if failure == "duplicate":
        result.evidence.text += "\n\n" + json.dumps(record)
    with pytest.raises(ConfigurationError):
        cited_source_evidence(case, result)
