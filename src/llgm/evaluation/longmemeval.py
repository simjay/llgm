"""LongMemEval ingestion with evaluator labels kept out of the model corpus."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llgm.core.errors import ConfigurationError, SchemaError
from llgm.core.types import SourceNode, Turn


@dataclass(frozen=True)
class EvaluationCase:
    """A question and its isolated source history, containing no evaluator labels."""

    case_id: str
    question: str
    question_date: str
    sources: tuple[SourceNode, ...]


@dataclass(frozen=True)
class GoldRecord:
    """Evaluator-only answers, ability labels, and evidence annotations for one case."""

    case_id: str
    answer: Any
    ability: str
    evidence_node_ids: tuple[str, ...]
    evidence_turn_ids: tuple[tuple[str, str], ...]
    source_aliases: dict[str, str] = field(default_factory=dict)


def load_longmemeval(path: str | Path) -> tuple[list[EvaluationCase], dict[str, GoldRecord]]:
    """Read a local release and separate model-visible cases from evaluator records."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise SchemaError(f"Cannot read LongMemEval JSON: {path}") from exc
    return parse_longmemeval(raw)


def parse_longmemeval(raw: Any) -> tuple[list[EvaluationCase], dict[str, GoldRecord]]:
    """Validate a release and preserve repeated sessions with distinct node IDs.

    Only role and text are copied from source turns; answer labels stay in gold records.
    """
    if not isinstance(raw, list):
        raise SchemaError("LongMemEval input must be a JSON array")
    cases, gold = [], {}
    for row in raw:
        if not isinstance(row, dict):
            raise SchemaError("Each LongMemEval case must be an object")
        required = {
            "question_id",
            "question",
            "question_date",
            "question_type",
            "answer",
            "haystack_session_ids",
            "haystack_dates",
            "haystack_sessions",
            "answer_session_ids",
        }
        if not required.issubset(row):
            raise SchemaError(f"LongMemEval case missing fields: {sorted(required - row.keys())}")
        case_id = str(row["question_id"])
        if case_id in gold:
            raise SchemaError(f"Duplicate question ID: {case_id}")
        ids, dates, sessions = (
            row["haystack_session_ids"],
            row["haystack_dates"],
            row["haystack_sessions"],
        )
        if (
            not all(isinstance(x, list) for x in (ids, dates, sessions))
            or len(ids) != len(dates)
            or len(ids) != len(sessions)
        ):
            raise SchemaError(f"Mismatched haystack arrays in {case_id}")
        id_counts = Counter(str(x) for x in ids)
        occurrences = Counter()
        aliases = {}
        occupied = {str(x) for x in ids}
        sources, answer_turns = [], []
        for session_id, date, session in zip(ids, dates, sessions):
            original_id = str(session_id)
            node_id = original_id
            if id_counts[original_id] > 1:
                occurrence = occurrences[original_id]
                occurrences[original_id] += 1
                node_id = f"{original_id}#occurrence-{occurrence:04d}"
                while node_id in occupied:
                    node_id += "-occurrence"
                occupied.add(node_id)
            aliases[node_id] = original_id
            if not isinstance(session, list):
                raise SchemaError(f"Session {session_id} is not a list of turns")
            turns = []
            for index, turn in enumerate(session):
                if (
                    not isinstance(turn, dict)
                    or not isinstance(turn.get("content"), str)
                    or not isinstance(turn.get("role"), str)
                ):
                    raise SchemaError(f"Invalid turn in session {session_id}")
                turn_id = f"t{index:05d}"
                # Explicit whitelist: has_answer and all other annotations stop here.
                turns.append(Turn(turn_id=turn_id, role=turn["role"], text=turn["content"]))
                if turn.get("has_answer") is True:
                    answer_turns.append((node_id, turn_id))
            sources.append(
                SourceNode(
                    node_id=node_id,
                    turns=tuple(turns),
                    metadata={"date": str(date), "benchmark_session_id": original_id},
                )
            )
        answers = tuple(dict.fromkeys(str(x) for x in row["answer_session_ids"]))
        if not set(answers).issubset({str(x) for x in ids}):
            raise SchemaError(f"Gold evidence session absent from supplied haystack in {case_id}")
        ability = "abstention" if case_id.endswith("_abs") else str(row["question_type"])
        gold[case_id] = GoldRecord(
            case_id, row["answer"], ability, answers, tuple(answer_turns), aliases
        )
        cases.append(
            EvaluationCase(case_id, str(row["question"]), str(row["question_date"]), tuple(sources))
        )
    return cases, gold


def _source_content_hash(source: SourceNode) -> str:
    """Hash role/text pairs so identical histories group together across session IDs."""
    payload = [(turn.role, turn.text) for turn in source.turns]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def history_groups(cases: list[EvaluationCase]) -> dict[str, list[str]]:
    """Connected components of shared session IDs OR identical session content.

    Shared distractors count too. If that collapses the release to one group,
    preparation reports the isolation problem instead of silently weakening it.
    """
    parent = list(range(len(cases)))

    def find(index):
        """Resolve a history component with path compression for repeated membership checks."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, case in enumerate(cases):
        for source in case.sources:
            for key in (
                ("id", source.metadata.get("benchmark_session_id", source.node_id)),
                ("content", _source_content_hash(source)),
            ):
                if key in owners:
                    left, right = find(index), find(owners[key])
                    parent[max(left, right)] = min(left, right)
                else:
                    owners[key] = index
    grouped = defaultdict(list)
    for index, case in enumerate(cases):
        grouped[find(index)].append(case.case_id)
    return {
        "history-" + hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]: sorted(ids)
        for ids in grouped.values()
    }


def select_development(
    cases: list[EvaluationCase],
    gold: dict[str, GoldRecord],
    target: int = 50,
    smoke_count: int = 5,
    seed: int = 1729,
) -> dict[str, Any]:
    """Choose seeded, ability-balanced whole histories while reserving a held-out group.

    Return a blocked selection when shared histories cannot support isolation.
    """
    if target <= 0 or smoke_count <= 0:
        raise ConfigurationError("Development target and smoke count must be positive")
    groups = history_groups(cases)
    if len(groups) < 2:
        return {
            "status": "blocked_history_isolation",
            "seed": seed,
            "target": target,
            "groups": groups,
            "development_ids": [],
            "smoke_ids": [],
            "held_out_ids": [],
            "requires_separate_development_data": True,
            "reason": "Fewer than two disjoint history groups; tune on separate controlled histories and freeze before benchmark evaluation",
        }
    rng = random.Random(seed)
    order = sorted(groups)
    rng.shuffle(order)
    tie = {group: rank for rank, group in enumerate(order)}
    abilities = sorted({record.ability for record in gold.values()})
    selected, chosen, counts = [], [], Counter()
    remaining = set(groups)
    # Greedy balanced stratification, selecting entire history components. Keep
    # at least one untouched group even if that means missing the target count.
    while remaining and len(selected) < target and len(remaining) > 1:

        def score(group):
            """Prefer underrepresented abilities, then target size, then seeded tie order."""
            group_abilities = {gold[case_id].ability for case_id in groups[group]}
            underrepresented = sum(1 / (1 + counts[a]) for a in group_abilities)
            return (-underrepresented, abs(target - len(selected) - len(groups[group])), tie[group])

        group = min(remaining, key=score)
        remaining.remove(group)
        chosen.append(group)
        selected.extend(groups[group])
        counts.update(gold[case_id].ability for case_id in groups[group])
    smoke = []
    for ability in abilities:
        candidate = [case_id for case_id in selected if gold[case_id].ability == ability]
        rng.shuffle(candidate)
        if candidate and len(smoke) < smoke_count:
            smoke.append(candidate[0])
    rest = sorted(set(selected) - set(smoke))
    rng.shuffle(rest)
    smoke.extend(rest[: max(0, smoke_count - len(smoke))])
    return {
        "status": "ready",
        "method": "project-defined shared-session connected components, balanced greedy selection",
        "seed": seed,
        "target": target,
        "actual_development_count": len(selected),
        "development_ids": sorted(selected),
        "smoke_ids": smoke,
        "held_out_ids": sorted(case_id for group in remaining for case_id in groups[group]),
        "development_groups": chosen,
        "held_out_groups": sorted(remaining),
        "groups": groups,
        "ability_counts": dict(sorted(counts.items())),
    }


def case_to_dict(case: EvaluationCase) -> dict[str, Any]:
    """Serialize a label-free case and its nested source records."""
    return asdict(case)


def case_from_dict(value: dict[str, Any]) -> EvaluationCase:
    """Restore case, source, and turn records from prepared artifact JSON."""
    sources = []
    for node in value["sources"]:
        if "source_version" in node or "commit_id" in node:
            raise SchemaError(
                "Legacy source editions require fresh preparation with immutable node IDs"
            )
        sources.append(
            SourceNode(
                node_id=node["node_id"],
                turns=tuple((Turn(**turn) for turn in node["turns"])),
                metadata=node.get("metadata", {}),
                timestamp_ms=node.get("timestamp_ms"),
            )
        )
    return EvaluationCase(
        value["case_id"], value["question"], value["question_date"], tuple(sources)
    )
