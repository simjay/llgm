"""Frozen controlled comparisons of flat reading, recursion and graph navigation.

Curated links and journal text are shared by every arm. These diagnostics do
not measure learned maintenance quality or establish benchmark superiority.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

from llgm.core.errors import ConfigurationError
from llgm.core.time import legacy_iso_to_ms, normalize_applicability
from llgm.core.types import (
    Conversation,
    JournalRef,
    NodeRef,
    Provenance,
    SourceSpan,
    reference_to_dict,
)
from llgm.evaluation.artifacts import RunArtifacts, write_json
from llgm.evaluation.prepare import file_sha256
from llgm.evaluation.runner import _code_provenance
from llgm.evaluation.scoring import exact_match_diagnostic
from llgm.inference.budget import Budget, byte_token_bound
from llgm.inference.iterative import EvidenceSidecar, IterativeRuntime
from llgm.inference.recursive import RecursiveRuntime
from llgm.inference.results import AnswerResult, EvidenceBundle
from llgm.memory.evidence import Evidence
from llgm.memory.query import QueryEvidence
from llgm.memory.workspace import Workspace, journal_to_dict

ARMS = ("flat_single", "recursive_search", "recursive_graph")
ANSWER_FORMAT = (
    "Return only the requested key as the answer. Use UNKNOWN when the evidence does not "
    "determine it; use UNRESOLVED when applicable claims conflict without a correction. "
    "Keep supporting references in the runtime's citation fields."
)


def controlled_cases(distractors: int = 40) -> list[dict]:
    """Build independent lookup, alias-chain, scoped-update, conflict and abstention histories."""
    if type(distractors) is not int or distractors < 0:
        raise ConfigurationError("distractors must be a nonnegative integer")
    specifications = [
        (
            "lookup",
            "What is Alder's release key?",
            {},
            "CEDAR",
            [("decision", "Alder's approved release key is CEDAR.")],
            [],
            ["decision"],
            [],
        ),
        (
            "aliases",
            "What release key does Birch's deployment destination use?",
            {},
            "NORTHSTAR",
            [
                ("plan", "Birch deploys through the channel Beacon."),
                ("channel", "Beacon's deployment destination is Slate."),
                ("destination", "Slate uses the release key NORTHSTAR."),
            ],
            [
                {
                    "connection_kind": "edge",
                    "key": "route1",
                    "owner": "plan",
                    "target": "channel",
                    "relation": "depends_on",
                },
                {
                    "connection_kind": "edge",
                    "key": "route2",
                    "owner": "channel",
                    "target": "destination",
                    "relation": "depends_on",
                },
            ],
            ["plan", "channel", "destination"],
            [],
        ),
        (
            "update",
            "As of 2026-01-10, what is Cypress's production release key?",
            {"env": "production"},
            "VIOLET",
            [
                (
                    "decision",
                    "Cypress previously used release key AMBER; subsequent journal corrections govern each environment.",
                )
            ],
            [
                {
                    "key": "production",
                    "owner": "decision",
                    "text": "Cypress production now uses release key VIOLET, replacing AMBER.",
                    "relation": "updates",
                    "applicability": {"scope": {"env": "production"}, "valid_from": "2026-01-04"},
                },
                {
                    "key": "staging",
                    "owner": "decision",
                    "text": "Cypress staging now uses release key BLUE.",
                    "relation": "updates",
                    "applicability": {"scope": {"env": "staging"}, "valid_from": "2026-01-04"},
                },
            ],
            [],
            ["production"],
        ),
        (
            "conflict",
            "What is Dogwood's approved release key?",
            {},
            "UNRESOLVED",
            [
                ("claim1", "Dogwood's release owner approves key RUBY for the same deployment."),
                (
                    "claim2",
                    "Dogwood's release owner approves key JADE for the same deployment. Neither approval supersedes the other.",
                ),
            ],
            [],
            ["claim1", "claim2"],
            [],
        ),
        (
            "abstention",
            "What is Elm's approved release key?",
            {},
            "UNKNOWN",
            [("note", "Elm's release key has not been decided. A planning meeting is scheduled.")],
            [],
            ["note"],
            [],
        ),
    ]
    cases = []
    for (
        case_id,
        question,
        scope,
        answer,
        sources,
        journals,
        required_sources,
        required_journals,
    ) in specifications:
        prefix = case_id + ":"
        records = [
            {"node_id": prefix + name, "text": text, "role": "user"} for name, text in sources
        ]
        records.extend(
            {
                "node_id": prefix + f"distractor-{number:03d}",
                "role": "assistant",
                "text": f"Independent {case_id} archive project {number} uses release key LOCAL{number:03d}. It has no stated relationship to the project in this case.",
            }
            for number in range(distractors)
        )
        entries, edges = [], []
        for entry in journals:
            value = {**entry, "owner": prefix + entry["owner"]}
            if "target" in value:
                value["target"] = prefix + value["target"]
            if "applicability" in value:
                value["applicability"] = normalize_applicability(value["applicability"])
            if value.pop("connection_kind", None) == "edge":
                edges.append(value)
            else:
                entries.append(value)
        cases.append(
            {
                "case_id": case_id,
                "question": question,
                "query_date": "2026-01-10",
                "as_of_ms": legacy_iso_to_ms("2026-01-10"),
                "scope": scope,
                "sources": records,
                "journals": entries,
                "edges": edges,
                "gold": {
                    "answer": answer,
                    "required_sources": [prefix + name for name in required_sources],
                    "required_journals": required_journals,
                },
            }
        )
    return cases


def freeze_comparison(
    path: str | Path,
    *,
    models: dict,
    budget: Budget | None = None,
    distractors: int = 40,
    seed: int = 1729,
) -> dict:
    """Write model pins, cohort, scoring and exposure rules before any inference request."""
    path = Path(path)
    if path.exists():
        raise ConfigurationError("Frozen comparison destination already exists")
    for role in ("root", "sidecar"):
        pin = models.get(role, {})
        if not all(
            isinstance(pin.get(key), str) and pin[key].strip() for key in ("provider", "model")
        ):
            raise ConfigurationError(f"Pin explicit {role} provider and model IDs")
        if "latest" in pin["model"].lower() or set(pin) != {"provider", "model"}:
            raise ConfigurationError(
                "Model pins require provider/model only, with no latest alias or credentials"
            )
    selected_budget = budget or Budget(
        max_model_calls=16,
        max_sidecar_calls=12,
        max_searches=4,
        max_evidence_tokens=65536,
        max_bundle_tokens=8192,
        max_context_tokens=65536,
        max_output_tokens=2048,
        timeout_seconds=120,
    )
    if type(seed) is not int:
        raise ConfigurationError("Comparison seed must be an integer")
    cases = controlled_cases(distractors)
    randomizer = random.Random(seed)
    orders = {}
    for case in cases:
        order = list(ARMS)
        randomizer.shuffle(order)
        orders[case["case_id"]] = order
    value = {
        "schema_version": 3,
        "evidence_schema": "primary-edges-v3",
        "kind": "controlled-development-comparison",
        "benchmark_result": False,
        "models": models,
        "budget": asdict(selected_budget),
        "seed": seed,
        "arm_orders": orders,
        "arms": list(ARMS),
        "cases": cases,
        "answer_format": ANSWER_FORMAT,
        "max_depth": 2,
        "max_steps": 16,
        "max_operations": 64,
        "passage_chars": 2048,
        "conditions": {
            "flat": "one original-question search, top40, sidecar composition then root answer",
            "recursion": "structured-operation ablation with same root/sidecar and Budget; not the LLGM Python seed-delegate pipeline",
            "navigation": "only primary-edge neighbors changes; graph-off retains journal/read/search opportunities",
            "journals": "complete compact operational journals accompany retrieved/read owners as charged metadata in every arm",
            "setup": "curated independent primary edges and journal assertions, shared immutable sources/index with no writes during trials; no model maintenance calls",
            "reads": "current journal/source reads; no source-version or public snapshot contract",
            "search": "ordinary SQLite FTS5 ranking through sqlite-fts5-incremental",
            "time": "index preparation is shared setup; each arm receives the same inference deadline",
            "order": "seeded per-case order, no outcome-based retries or exclusions",
            "scoring": "normalized exact match plus complete coverage of labeled source turns/journal values",
            "partial": "partial answers may match an abstention/conflict key; errors never receive correctness credit",
            "support_limit": "reference coverage is not an independent entailment judgment",
            "cost": "reported provider tokens/calls; currency remains unknown without a price basis",
        },
    }
    write_json(path, value)
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "case_count": len(cases)}


def load_comparison(path: str | Path, expected_sha256: str) -> dict:
    """Verify the frozen file hash and basic experiment identity before using any labels or clients."""
    if file_sha256(path) != expected_sha256:
        raise ConfigurationError("Frozen comparison SHA256 differs from the declared pin")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != 3
        or value.get("evidence_schema") != "primary-edges-v3"
        or value.get("kind") != "controlled-development-comparison"
        or value.get("arms") != list(ARMS)
    ):
        raise ConfigurationError(
            "Unsupported comparison protocol: require schema_version=3 with primary-edges-v3 "
            "evidence. Freeze a new v3 protocol; retain v1/v2 files and results unchanged."
        )
    Budget(**value["budget"])
    ids = [case["case_id"] for case in value["cases"]]
    if not ids or len(ids) != len(set(ids)):
        raise ConfigurationError("Comparison requires unique nonempty cases")
    seen_nodes = set()
    for case in value["cases"]:
        nodes = [source["node_id"] for source in case["sources"]]
        if len(nodes) != len(set(nodes)) or seen_nodes.intersection(nodes):
            raise ConfigurationError("Controlled case source identities must be disjoint")
        seen_nodes.update(nodes)
        if sorted(value["arm_orders"][case["case_id"]]) != sorted(ARMS):
            raise ConfigurationError("Each frozen case must include all three arms exactly once")
    return value


class _SharedEvidence:
    """Expose the same canonical text, owner journals and applicability information to every arm."""

    def __init__(self, evidence, case):
        """Apply the frozen query scope to the shared corpus; trials do not write evidence."""
        self.evidence = QueryEvidence(
            evidence, case["scope"], case["query_date"], as_of_ms=case["as_of_ms"]
        )

    async def _metadata(self, owners):
        """Expose complete operational owner journals equally without requiring edge navigation."""
        return {
            owner: [
                journal_to_dict(entry)
                for entry in await self.evidence.evidence.workspace.operational_journal(owner)
            ]
            for owner in dict.fromkeys(owners)
        }

    async def search(self, query, k):
        """Attach journal records to each canonical hit; runtimes charge the resulting text and metadata."""
        hits = await self.evidence.search(query, k)
        return [
            replace(
                hit,
                passage=replace(
                    hit.passage,
                    metadata={
                        **hit.passage.metadata,
                        "owner_journals": await self._metadata(
                            ref.node_id for ref in hit.passage.refs
                        ),
                    },
                ),
            )
            for hit in hits
        ]

    async def read(self, reference):
        """Return a canonical read with the same owner-journal opportunity as search."""
        result = await self.evidence.read(reference)
        return replace(
            result,
            metadata={
                **result.metadata,
                "owner_journals": await self._metadata([reference.node_id]),
            },
        )

    async def journal(self, node_id):
        """Retain journal access in both recursive arms."""
        return await self.evidence.journal(node_id)

    async def neighbors(self, node_id, relation=None):
        """Follow applicable primary edges in the graph-enabled arm."""
        return await self.evidence.neighbors(node_id, relation)


class _FlatSearch:
    """Keep flat retrieval on the original question while preserving shared evidence metadata."""

    def __init__(self, evidence, question):
        """Borrow the exact evidence view used by the recursive arms."""
        self.evidence = evidence
        self.question = question

    async def search(self, query, k):
        """Pass canonical text and metadata to the reader's ordinary evidence admission."""
        return await self.evidence.search(self.question, k)


async def _setup_case(workspace, case):
    """Publish frozen source and curator journal records without consulting evaluator answers."""
    journal_refs = {}
    for source in case["sources"]:
        await workspace.ingest(
            Conversation.from_turns(
                [
                    {"turn_id": "t", "role": source["role"], "text": source["text"]},
                ],
                node_id=source["node_id"],
                metadata={"date": "2026-01-01"},
            )
        )
    for specification in case["edges"]:
        await workspace.publish_edge(
            specification["owner"],
            specification["target"],
            relation=specification["relation"],
            provenance=Provenance("user", "frozen-comparison-curator"),
            applicability=specification.get("applicability"),
            idempotency_key=specification["key"],
        )
    for specification in case["journals"]:
        owner = specification["owner"]
        value = (
            NodeRef(specification["target"]) if "target" in specification else specification["text"]
        )
        entry = await workspace.append_journal(
            owner,
            subject=NodeRef(owner),
            relation=specification["relation"],
            value=value,
            provenance=Provenance("user", "frozen-comparison-curator"),
            applicability=specification.get("applicability"),
            idempotency_key=specification["key"],
        )
        journal_refs[specification["key"]] = JournalRef(owner, entry.entry_id)
    return journal_refs


def _covers(reference, required) -> bool:
    """Require an entire labeled turn or journal value, accepting canonical whole-node/entry citations."""
    if isinstance(required, SourceSpan):
        if isinstance(reference, NodeRef):
            return reference.node_id == required.node_id
        return (
            isinstance(reference, SourceSpan)
            and (reference.node_id, reference.turn_id) == (required.node_id, required.turn_id)
            and reference.start <= required.start
            and reference.end >= required.end
        )
    return (
        isinstance(reference, JournalRef)
        and (reference.node_id, reference.entry_id) == (required.node_id, required.entry_id)
        and (
            reference.start is None
            or (
                required.start is not None
                and reference.start <= required.start
                and reference.end >= required.end
            )
        )
    )


def _valid_reference(reference, sources, specifications, journal_refs):
    """Validate citations against the exact curated records published once for this frozen case."""
    if isinstance(reference, (NodeRef, SourceSpan)):
        source = sources.get(reference.node_id)
        if source is None:
            return False
        return isinstance(reference, NodeRef) or (
            reference.turn_id == "t"
            and 0 <= reference.start <= reference.end <= len(source["text"])
        )
    if isinstance(reference, JournalRef):
        for key, actual in journal_refs.items():
            if (reference.node_id, reference.entry_id) == (actual.node_id, actual.entry_id):
                text = specifications[key].get("text")
                return reference.start is None or (
                    text is not None and 0 <= reference.start <= reference.end <= len(text)
                )
    return False


def _score(result, case, journal_refs):
    """Score from frozen canonical records without an await between cancelled work and durable artifacts."""
    sources = {source["node_id"]: source for source in case["sources"]}
    required = [
        SourceSpan(node_id, "t", 0, len(sources[node_id]["text"]))
        for node_id in case["gold"]["required_sources"]
    ]
    specifications = {entry["key"]: entry for entry in case["journals"]}
    for key in case["gold"]["required_journals"]:
        reference = journal_refs[key]
        text = specifications[key].get("text")
        required.append(
            replace(reference, start=0, end=len(text)) if text is not None else reference
        )
    valid = all(
        _valid_reference(reference, sources, specifications, journal_refs)
        for reference in result.references
    )
    credited = result.status in {"completed", "partial"}
    covered = (
        sum(
            any(_covers(reference, target) for reference in result.references)
            for target in required
        )
        if valid and credited
        else 0
    )
    exact = credited and exact_match_diagnostic(result.answer, case["gold"]["answer"])
    support = covered == len(required) if required else None
    return {
        "exact_match": exact,
        "citations_valid": valid,
        "required_support_count": len(required),
        "covered_support_count": covered,
        "all_required_support": support,
        "exact_match_with_support": exact and valid and support is not False,
        "support_granularity": "complete labeled source turns or journal values; no semantic entailment judge",
    }


async def run_comparison(
    path: str | Path, output: str | Path, root_model, sidecar_model, *, expected_sha256: str
) -> dict:
    """Run the frozen three-arm comparison using caller-owned real or explicitly injected model clients."""
    protocol = load_comparison(path, expected_sha256)
    for role, client in (("root", root_model), ("sidecar", sidecar_model)):
        descriptor = client.descriptor()
        if any(
            descriptor.get(key) != protocol["models"][role][key] for key in ("provider", "model")
        ):
            raise ConfigurationError(f"Actual {role} client differs from the frozen model pin")
    output = Path(output)
    artifacts = RunArtifacts(
        output,
        {
            "protocol_sha256": expected_sha256,
            "protocol": protocol,
            "code": _code_provenance(),
            "status": "running",
            "benchmark_result": False,
            "model_clients": {
                "root": root_model.descriptor(),
                "sidecar": sidecar_model.descriptor(),
            },
        },
    )
    observations = []
    summary = {
        "status": "running",
        "benchmark_result": False,
        "protocol_sha256": expected_sha256,
        "planned_case_arm_runs": len(protocol["cases"]) * len(ARMS),
        "recorded_case_arm_runs": 0,
        "arms": {},
        "setup": [],
        "currency_cost": None,
    }
    write_json(output / "summary.json", summary)
    phase = "setup"
    try:
        for case in protocol["cases"]:
            phase = "setup"
            started = time.monotonic()
            async with Workspace.open(output / "workspaces" / case["case_id"]) as workspace:
                journal_refs = await _setup_case(workspace, case)
                async with await Evidence.open(
                    workspace, passage_chars=protocol["passage_chars"]
                ) as evidence:
                    setup = {
                        "case_id": case["case_id"],
                        "elapsed_seconds": time.monotonic() - started,
                        "source_count": len(case["sources"]),
                        "journal_count": len(case["journals"]),
                        "edge_count": len(case["edges"]),
                        "maintenance_model_calls": 0,
                        "maintenance_origin": "curated",
                        "index": getattr(evidence, "preparation", {}),
                        "retrieval": evidence.descriptor(),
                    }
                    summary["setup"].append(setup)
                    artifacts.append("usage", {"category": "shared_setup", **setup})
                    shared = _SharedEvidence(evidence, case)
                    for arm in protocol["arm_orders"][case["case_id"]]:
                        phase = "inference"
                        cancelled = None
                        caught_error_type = None
                        runtime = None
                        started = time.monotonic()
                        artifacts.append(
                            "cases",
                            {
                                "case_id": case["case_id"],
                                "arm": arm,
                                "question": case["question"],
                                "scope": case["scope"],
                                "query_date": case["query_date"],
                            },
                        )
                        try:
                            budget = Budget(**protocol["budget"])
                            question = case["question"] + "\n" + protocol["answer_format"]
                            if arm == "flat_single":
                                runtime = IterativeRuntime(
                                    root=root_model,
                                    sidecar=EvidenceSidecar(
                                        model=sidecar_model,
                                        retriever=_FlatSearch(shared, case["question"]),
                                        policy="single",
                                        token_counter=byte_token_bound,
                                    ),
                                )
                                result = await runtime.answer(
                                    question, budget, question_date=case["query_date"]
                                )
                            else:
                                runtime = RecursiveRuntime(
                                    root_model,
                                    sidecar_model,
                                    shared,
                                    budget=budget,
                                    max_depth=protocol["max_depth"],
                                    max_steps=protocol["max_steps"],
                                    max_operations=protocol["max_operations"],
                                    allow_neighbors=arm == "recursive_graph",
                                )
                                result = await runtime.answer(
                                    question,
                                    query_date=case["query_date"],
                                    query_scope=case["scope"],
                                )
                        except BaseException as error:
                            if not isinstance(error, (Exception, asyncio.CancelledError)):
                                raise
                            cancelled = error if isinstance(error, asyncio.CancelledError) else None
                            caught_error_type = type(error).__name__
                            result = AnswerResult(
                                "",
                                EvidenceBundle(),
                                getattr(error, "llgm_usage", getattr(runtime, "last_usage", {})),
                                getattr(error, "llgm_trace", getattr(runtime, "last_trace", [])),
                                "cancelled" if cancelled else "failed",
                            )
                            artifacts.append(
                                "traces",
                                {
                                    "case_id": case["case_id"],
                                    "arm": arm,
                                    "kind": "failure",
                                    "error_type": type(error).__name__,
                                },
                            )
                        row = {
                            "case_id": case["case_id"],
                            "arm": arm,
                            "status": result.status,
                            "error_type": caught_error_type
                            or next(
                                (
                                    event["error_type"]
                                    for event in reversed(result.trace)
                                    if event.get("error_type")
                                ),
                                None,
                            ),
                            "answer": result.answer,
                            "references": [reference_to_dict(ref) for ref in result.references],
                            "elapsed_seconds": time.monotonic() - started,
                            "usage": result.usage,
                        }
                        judgment = _score(result, case, journal_refs)
                        artifacts.append("predictions", row)
                        artifacts.append(
                            "judgments", {"case_id": case["case_id"], "arm": arm, **judgment}
                        )
                        artifacts.append(
                            "usage",
                            {
                                "case_id": case["case_id"],
                                "arm": arm,
                                "category": "inference",
                                **result.usage,
                            },
                        )
                        for event in result.trace:
                            artifacts.append(
                                "traces", {"case_id": case["case_id"], "arm": arm, **event}
                            )
                        observations.append({**row, **judgment})
                        summary.update(_summary(observations))
                        if cancelled:
                            summary["status"] = "cancelled"
                        write_json(output / "summary.json", summary)
                        if cancelled:
                            artifacts.finish(
                                summary,
                                "Cancelled; unstarted trials are not scored, and attempted work is retained.",
                            )
                            raise cancelled
    except BaseException as error:
        if not isinstance(error, (Exception, asyncio.CancelledError)):
            raise
        summary["status"] = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
        summary["failure_phase"] = phase
        summary["error_type"] = type(error).__name__
        write_json(output / "summary.json", summary)
        artifacts.finish(
            summary,
            "Execution stopped; recorded trials retain their denominators and unstarted trials were not scored.",
        )
        raise
    summary["status"] = "completed"
    write_json(output / "summary.json", summary)
    artifacts.finish(
        summary,
        "Controlled development diagnostics only. Curated setup is shared; results do not establish maintenance quality, generalization or benchmark superiority.",
    )
    return summary


def _summary(observations):
    """Keep every attempted trial in per-arm scores while separating unknown token usage from zero."""
    arms = {}
    for arm in ARMS:
        rows = [row for row in observations if row["arm"] == arm]
        if not rows:
            continue
        arms[arm] = {
            "case_count": len(rows),
            "completed": sum(row["status"] == "completed" for row in rows),
            "partial": sum(row["status"] == "partial" for row in rows),
            "failed": sum(row["status"] not in {"completed", "partial"} for row in rows),
            "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
            "exact_match_with_support": sum(row["exact_match_with_support"] for row in rows)
            / len(rows),
            "model_calls": sum(row["usage"].get("model_calls", 0) for row in rows),
            "known_input_tokens": sum(row["usage"].get("known_input_tokens", 0) for row in rows),
            "known_output_tokens": sum(row["usage"].get("known_output_tokens", 0) for row in rows),
            "unknown_usage_calls": sum(row["usage"].get("unknown_usage_calls", 0) for row in rows),
            "inference_seconds": sum(row["elapsed_seconds"] for row in rows),
            "currency_cost": None,
        }
    return {"recorded_case_arm_runs": len(observations), "arms": arms}


def main(argv=None) -> int:
    """Freeze conditions or explicitly dispatch pinned hosted clients; missing readiness writes not_run."""
    import argparse
    import os
    import sys
    from contextlib import AsyncExitStack

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--distractors", type=int, default=40)
    freeze.add_argument("--seed", type=int, default=1729)
    for role in ("root", "sidecar"):
        freeze.add_argument(f"--{role}-provider", choices=("openai", "anthropic"), default="openai")
        freeze.add_argument(f"--{role}-model", required=True)
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "freeze":
            result = freeze_comparison(
                arguments.output,
                models={
                    role: {
                        "provider": getattr(arguments, role + "_provider"),
                        "model": getattr(arguments, role + "_model"),
                    }
                    for role in ("root", "sidecar")
                },
                distractors=arguments.distractors,
                seed=arguments.seed,
            )
        else:
            protocol = load_comparison(arguments.protocol, arguments.sha256)
            errors = []
            if not arguments.execute:
                errors.append("Explicit --execute is required for hosted model calls")
            for role, pin in protocol["models"].items():
                key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(
                    pin["provider"]
                )
                if key is None:
                    errors.append(f"Unsupported hosted {role} provider")
                elif not os.environ.get(key):
                    errors.append(f"Missing {role} credential environment variable {key}")
            if errors:
                artifacts = RunArtifacts(
                    arguments.output,
                    {
                        "protocol_sha256": arguments.sha256,
                        "protocol": protocol,
                        "status": "not_run",
                        "benchmark_result": False,
                    },
                )
                result = {
                    "status": "not_run",
                    "benchmark_result": False,
                    "errors": errors,
                    "model_calls": 0,
                    "recorded_case_arm_runs": 0,
                    "planned_case_arm_runs": len(protocol["cases"]) * len(ARMS),
                }
                artifacts.finish(
                    result,
                    "No inference was dispatched; fix the recorded prerequisites and use a new output directory.",
                )
                write_json(arguments.output / "summary.json", result)
                print(json.dumps(result, indent=2))
                return 2

            async def execute():
                """Own hosted adapters for one explicitly authorized invocation."""
                from llgm.models import create_model

                async with AsyncExitStack() as stack:
                    clients = []
                    for role in ("root", "sidecar"):
                        pin = protocol["models"][role]
                        client = create_model(
                            pin["provider"],
                            pin["model"],
                            api_key_env="ANTHROPIC_API_KEY"
                            if pin["provider"] == "anthropic"
                            else "OPENAI_API_KEY",
                        )
                        stack.push_async_callback(client.aclose)
                        clients.append(client)
                    return await run_comparison(
                        arguments.protocol,
                        arguments.output,
                        *clients,
                        expected_sha256=arguments.sha256,
                    )

            result = asyncio.run(execute())
        print(json.dumps(result, indent=2))
        return 0
    except (ConfigurationError, OSError, ValueError) as error:
        print(f"comparison: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
