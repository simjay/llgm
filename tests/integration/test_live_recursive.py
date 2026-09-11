"""Real-model protocol smoke with evaluator-selected source handles, not answers."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import AsyncExitStack
from dataclasses import asdict

import pytest

from llgm.core.types import Conversation, NodeRef, SourceSpan, reference_to_dict
from llgm.evaluation.artifacts import write_json, write_jsonl
from llgm.evaluation.scoring import evidence_coverage
from llgm.memory.workspace import Workspace

pytestmark = [pytest.mark.integration, pytest.mark.dataset, pytest.mark.live, pytest.mark.recursive]


def _trace_metadata(trace):
    """Keep operations/provenance; avoid persisting prompts, evidence or error bodies."""
    records = []
    for event in trace:
        row = {
            key: value
            for key, value in event.items()
            if key
            in {
                "kind",
                "invocation_id",
                "parent_id",
                "depth",
                "role",
                "attempt",
                "status",
                "provider",
                "model",
                "request_id",
                "input_tokens",
                "output_tokens",
                "context_accounting_units",
                "error_type",
                "reason",
                "references",
                "citations",
            }
        }
        operation = event.get("operation")
        if isinstance(operation, str):
            row["operation"] = operation
        elif isinstance(operation, dict):
            row["operation"] = {
                key: value
                for key, value in operation.items()
                if key in {"op", "reference", "references", "node_id", "relation", "k", "citations"}
            }
            for key in ("question", "query", "answer"):
                if isinstance(operation.get(key), str):
                    row["operation"][key + "_sha256"] = hashlib.sha256(
                        operation[key].encode()
                    ).hexdigest()
        records.append(row)
    return records


def test_recursive_oracle_source_protocol(
    live_recursive, longmemeval, tmp_path, integration_record
):
    """An oracle-source run delegates twice and returns resolvable cited evidence."""
    from llgm.inference.iterative import Budget
    from llgm.inference.recursive import RecursiveRuntime
    from llgm.memory.evidence import Evidence
    from llgm.models import create_model

    record, directory = integration_record
    case = longmemeval.cases["001be529"]
    gold = longmemeval.gold[case.case_id]
    # Only source identities are used for this declared oracle-reader diagnostic.
    # No has_answer turns, answer text, ability label or question ID enters prompts.
    initial_refs = tuple(
        NodeRef(source.node_id)
        for source in case.sources
        if gold.source_aliases[source.node_id] in gold.evidence_node_ids
    )
    assert initial_refs
    budget = Budget(
        max_model_calls=12,
        max_sidecar_calls=10,
        max_searches=2,
        max_evidence_tokens=64000,
        max_bundle_tokens=64000,
        max_context_tokens=131072,
        max_output_tokens=2048,
        timeout_seconds=180,
    )
    record.update(
        dataset=longmemeval.manifest,
        case_id=case.case_id,
        scope="oracle-source recursive protocol sanity; root is instructed to delegate; not retrieval quality or autonomous planning performance",
        budget=asdict(budget),
        models=live_recursive,
    )
    prompt = (
        "This is a recursive protocol check. Your first operation must delegate the original "
        "question with all provided references to a child at depth 1. Your delegate question "
        "must require that child's first operation to delegate again, passing the question "
        "and all source references to a child at depth 2. In that second delegate question, "
        "require the depth-2 child to read the supplied source nodes directly and finish "
        "with a cited answer, without further delegation. The depth-1 child must then "
        "return the answer supported by its child's evidence. Finally, use that returned "
        "evidence to finish at the root. Complete both delegation levels; do not substitute "
        "a direct root answer or stop at depth 1. Original question: " + case.question
    )

    async def scenario():
        """Exercise both hosted roles against a frozen real-history workspace."""
        async with Workspace.open(tmp_path / "workspace") as workspace:
            for source in case.sources:
                await workspace.ingest(Conversation(source.turns, source.node_id, source.metadata))
            async with await Evidence.open(workspace) as evidence:
                async with AsyncExitStack() as stack:
                    models = {
                        role: await stack.enter_async_context(
                            create_model(**config, timeout_seconds=60)
                        )
                        for role, config in live_recursive.items()
                    }
                    runtime = RecursiveRuntime(
                        models["root"],
                        models["sidecar"],
                        evidence,
                        budget=budget,
                        max_depth=2,
                        max_steps=8,
                        capture_text=True,
                    )
                    try:
                        result = await runtime.answer(
                            prompt, initial_refs=initial_refs, query_date=case.question_date
                        )
                    finally:
                        record["operations"] = _trace_metadata(runtime.last_trace)
                        record["usage"] = runtime.last_usage
                        # Persist attempted work before assertions or any quality check.
                        write_json(directory / "runtime-attempt.json", record)
                        write_json(directory / "protocol-transcript.json", runtime.last_trace)
                    # Full-node reads exposed every turn. Expand those references
                    # only inside this evaluator; no gold spans are given to models.
                    covered_spans = []
                    sources_by_id = {source.node_id: source for source in case.sources}
                    for ref in result.references:
                        if isinstance(ref, SourceSpan):
                            covered_spans.append(ref)
                        elif isinstance(ref, NodeRef):
                            source = sources_by_id[ref.node_id]
                            covered_spans.extend(
                                SourceSpan(
                                    source.node_id,
                                    turn.turn_id,
                                    0,
                                    len(turn.text),
                                )
                                for turn in source.turns
                            )
                    record["coverage"] = evidence_coverage(covered_spans, asdict(gold))
                    record["coverage"]["annotation_granularity"] = (
                        "source and labeled-turn coverage; cited whole-node reads expand to all "
                        "their actually exposed turns only in the evaluator; no answer-span correctness claim"
                    )
                    record["result"] = {
                        "status": result.status,
                        "references": [reference_to_dict(ref) for ref in result.references],
                        "unresolved_count": len(result.evidence.unresolved),
                        "unresolved": result.evidence.unresolved,
                    }
                    predictions = directory / "predictions.jsonl"
                    write_jsonl(
                        predictions, [{"question_id": case.case_id, "hypothesis": result.answer}]
                    )
                    write_json(
                        directory / "prediction-manifest.json",
                        {
                            "schema_version": 1,
                            "dataset_sha256": longmemeval.manifest["sha256"],
                            "predictions_sha256": hashlib.sha256(
                                predictions.read_bytes()
                            ).hexdigest(),
                            "case_ids": [case.case_id],
                            "models": live_recursive,
                            "scope": record["scope"],
                            "benchmark_result": False,
                            "status": result.status,
                        },
                    )
                    write_json(directory / "runtime-attempt.json", record)
                    assert result.status == "completed", (
                        "Protocol run did not complete; inspect metadata artifacts"
                    )
                    events = result.trace
                    invocations = {
                        event["invocation_id"]: event
                        for event in events
                        if event.get("kind") == "enter"
                    }
                    assert {event["depth"] for event in invocations.values()} == {0, 1, 2}
                    assert any(
                        event["depth"] == 2 and invocations[event["parent_id"]]["depth"] == 1
                        for event in invocations.values()
                    )
                    assert any(
                        event.get("kind") == "enter" and event.get("depth") == 1 for event in events
                    )
                    assert any(
                        event.get("kind") == "return" and event.get("depth") == 1
                        for event in events
                    )
                    assert any(
                        event.get("kind") == "return" and event.get("depth") == 2
                        for event in events
                    )
                    assert any(
                        event.get("kind") == "operation"
                        and event.get("operation", {}).get("op") == "read"
                        and invocations[event["invocation_id"]]["depth"] == 2
                        for event in events
                    )
                    assert result.answer.strip() and result.references
                    for ref in result.references:
                        assert (await evidence.read(ref)).text
                    calls = [event for event in events if event.get("kind") == "model"]
                    assert {event["role"] for event in calls} == {"root", "sidecar"}
                    assert all(
                        event.get("model") == live_recursive[event["role"]]["model"]
                        for event in calls
                    )
                    assert all(event.get("status") == "completed" for event in calls)

    asyncio.run(scenario())
