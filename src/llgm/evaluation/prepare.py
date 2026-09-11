"""Deterministic, per-case preparation; downloads require an explicit function call."""

from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

from llgm.core.errors import ConfigurationError
from llgm.evaluation.artifacts import write_json, write_jsonl
from llgm.evaluation.longmemeval import (
    case_to_dict,
    history_groups,
    load_longmemeval,
    select_development,
)
from llgm.retrieval import DiagnosticTokenizer, split_nodes


def file_sha256(path: str | Path) -> str:
    """Hash file bytes in bounded chunks for preparation and execution provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    dataset_path: str | Path,
    output: str | Path,
    *,
    tokenizer=None,
    revision: str | None = None,
    target: int = 50,
    smoke_count: int = 5,
    seed: int = 1729,
    dataset_kind: str = "longmemeval-s-cleaned",
    selection_mode: str = "development",
    case_limit: int | None = None,
) -> dict:
    """Materialize selected case corpora, shared passages, and evaluator-only labels.

    Record dataset and artifact hashes in a new or empty directory. Development
    selection requires isolated histories; reserved evaluation requires a later
    frozen configuration. The default tokenizer is for diagnostics only.
    """
    output = Path(output)
    started = time.perf_counter()
    if output.exists() and any(output.iterdir()):
        raise ConfigurationError("Preparation directory must be new or empty")
    tokenizer = tokenizer or DiagnosticTokenizer()
    cases, gold = load_longmemeval(dataset_path)
    if selection_mode == "development":
        selection = select_development(
            cases, gold, target=target, smoke_count=smoke_count, seed=seed
        )
    elif selection_mode == "evaluation":
        ids = sorted(case.case_id for case in cases)
        random.Random(seed).shuffle(ids)
        if case_limit is not None and case_limit <= 0:
            raise ConfigurationError("case_limit must be positive")
        selection = {
            "status": "ready",
            "selection_mode": "reserved_evaluation",
            "seed": seed,
            "evaluation_ids": ids[:case_limit] if case_limit else ids,
            "held_out_ids": sorted(ids),
            "development_ids": [],
            "smoke_ids": [],
            "groups": history_groups(cases),
            "requires_separate_development_data": True,
            "method": "benchmark reserved for evaluation; no within-release development split",
            "freeze_required": True,
        }
    else:
        raise ConfigurationError("selection_mode must be development or evaluation")
    manifest = {
        "schema_version": 2,
        "evidence_schema": "immutable-node-v2",
        "dataset": {
            "path": str(Path(dataset_path).resolve()),
            "sha256": file_sha256(dataset_path),
            "revision": revision,
            "kind": dataset_kind,
        },
        "selection": selection,
        "tokenizer": tokenizer.descriptor(),
        "representation": {
            "window": 180,
            "overlap": 32,
            "boundaries": "session-and-turn",
            "metadata": "session ID, date, role",
            "canonical_offsets": "Unicode code points",
        },
        "files": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    selected = set(selection.get("evaluation_ids", selection["development_ids"]))
    queries = []
    for case in cases:
        if case.case_id not in selected:
            continue
        directory = "corpora/" + hashlib.sha256(case.case_id.encode()).hexdigest()[:24]
        source_path = f"{directory}/sources.json"
        passage_path = f"{directory}/passages.jsonl"
        write_json(output / source_path, case_to_dict(case))
        passages = split_nodes(case.sources, tokenizer, window=180, overlap=32)
        write_jsonl(output / passage_path, passages)
        queries.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "question_date": case.question_date,
                "sources_path": source_path,
                "passages_path": passage_path,
                "passage_count": len(passages),
            }
        )
        manifest["files"][source_path] = file_sha256(output / source_path)
        manifest["files"][passage_path] = file_sha256(output / passage_path)
    write_jsonl(output / "queries.jsonl", queries)
    # Gold is never serialized into corpora, passage metadata, or queries.
    write_jsonl(output / "evaluator/gold.jsonl", [asdict(gold[key]) for key in sorted(gold)])
    manifest["files"]["queries.jsonl"] = file_sha256(output / "queries.jsonl")
    manifest["files"]["evaluator/gold.jsonl"] = file_sha256(output / "evaluator/gold.jsonl")
    manifest["preparation_seconds"] = time.perf_counter() - started
    write_json(output / "prepared.json", manifest)
    return manifest


def download_cleaned_s(
    output: str | Path, *, revision: str, expected_sha256: str | None = None
) -> Path:
    """Explicit public-data download from a pinned official dataset revision."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ConfigurationError("Dataset download requires a full Hugging Face commit hash")
    path = Path(output)
    if path.exists():
        raise ConfigurationError("Dataset destination already exists")
    from urllib.request import urlopen

    url = f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/{revision}/longmemeval_s_cleaned.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("xb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
        if expected_sha256 and file_sha256(temporary) != expected_sha256.lower():
            raise ConfigurationError("Downloaded dataset SHA256 does not match supplied pin")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def diagnostic_dataset() -> list[dict]:
    """Return six isolated lookup cases for offline ingestion, retrieval and scoring checks."""
    records = []
    for index, color in enumerate(("amber", "blue", "green", "violet", "silver", "copper")):
        node = f"diagnostic-session-{index}"
        records.append(
            {
                "question_id": f"diagnostic-{index}",
                "question_type": "single-session-user",
                "question": f"What is project {index}'s launch color?",
                "answer": color,
                "question_date": "2026/09/09 (Wed) 12:00",
                "haystack_session_ids": [node],
                "haystack_dates": ["2026/09/08 (Tue) 10:00"],
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": f"The launch color for project {index} is {color}.",
                            "has_answer": True,
                        },
                        {
                            "role": "assistant",
                            "content": f"I recorded the launch decision for project {index}.",
                        },
                    ]
                ],
                "answer_session_ids": [node],
            }
        )
    return records


def controlled_dataset(count: int = 60) -> list[dict]:
    """Small hand-authored template workload for separate development.

    Contains lookup, updates, scoped decisions, temporal queries, dependencies,
    and abstention. These templates are behavioral diagnostics, not a calibrated
    substitute for the benchmark or evidence of generalization.
    """
    if count < 6:
        raise ConfigurationError("Controlled workload needs at least six cases")
    result = []
    for index in range(count):
        project = f"Orchid-{index}"
        case_id = f"controlled-{index:04d}"
        first = {"role": "user", "content": f"For {project}, the initial launch color is amber."}
        sessions = [[first]]
        dates = ["2026/09/01 (Tue) 10:00"]
        kind = index % 6
        question = f"What is the launch color for {project}?"
        answer, ability, required, answer_turns = "amber", "single-session-user", [0], [(0, 0)]
        if kind == 1:
            sessions.append(
                [
                    {
                        "role": "user",
                        "content": f"For {project}, replace the earlier launch color: the final launch color is violet.",
                    }
                ]
            )
            dates.append("2026/09/03 (Thu) 10:00")
            answer, ability, required, answer_turns = "violet", "knowledge-update", [1], [(1, 0)]
        elif kind == 2:
            sessions[0] = [
                {"role": "user", "content": f"The public alias for {project} is Comet-{index}."}
            ]
            sessions.append(
                [{"role": "user", "content": f"Comet-{index} uses silver as its launch color."}]
            )
            dates.append("2026/09/03 (Thu) 10:00")
            answer, ability, required, answer_turns = (
                "silver",
                "multi-session",
                [0, 1],
                [(0, 0), (1, 0)],
            )
        elif kind == 3:
            sessions[0] = [
                {
                    "role": "user",
                    "content": f"The {project} launch date is September 20. No launch color has been selected.",
                }
            ]
            case_id += "_abs"
            answer, ability, required, answer_turns = (
                "The launch color has not been selected.",
                "single-session-user",
                [],
                [],
            )
        elif kind == 4:
            sessions.append(
                [
                    {
                        "role": "user",
                        "content": f"On September 3, {project} changed its launch color from amber to blue.",
                    }
                ]
            )
            dates.append("2026/09/03 (Thu) 10:00")
            question = f"What was {project}'s launch color on September 1, before the later change?"
            ability = "temporal-reasoning"
        else:
            if kind == 5:
                sessions[0] = [
                    {
                        "role": "user",
                        "content": f"For {project}, production uses amber. The testing environment uses green.",
                    }
                ]
                sessions.append(
                    [
                        {
                            "role": "assistant",
                            "content": f"Suggestion for {project}: consider violet for testing; this is not a confirmed change.",
                        }
                    ]
                )
                dates.append("2026/09/03 (Thu) 10:00")
                question = f"What is the confirmed testing environment color for {project}?"
                answer, ability = "green", "single-session-preference"
        ids = [f"{case_id}-session-{i}" for i in range(len(sessions))]
        for session, turn in answer_turns:
            sessions[session][turn]["has_answer"] = True
        result.append(
            {
                "question_id": case_id,
                "question_type": ability,
                "question": question,
                "answer": answer,
                "question_date": "2026/09/09 (Wed) 12:00",
                "haystack_session_ids": ids,
                "haystack_dates": dates,
                "haystack_sessions": sessions,
                "answer_session_ids": [ids[i] for i in required],
            }
        )
    return result
