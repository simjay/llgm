"""Opted-in autonomous application sanity over a complete pinned LongMemEval history.

This test supplies the original question and ordinary source metadata only.
It does not select gold handles or prescribe delegate code. The library's seed
dispatch, graph proposals, Python choices, failures and predictions remain observable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
from contextlib import AsyncExitStack
from dataclasses import asdict
from datetime import timezone

import pytest

from llgm.core.time import parse_instant_ms
from llgm.core.types import Conversation, NodeRef, Provenance, SourceSpan, reference_to_dict
from llgm.evaluation.artifacts import write_json, write_jsonl
from llgm.evaluation.scoring import evidence_coverage
from llgm.inference.budget import Budget
from llgm.inference.repl import DockerREPLConfig
from llgm.llgm import LLGM, MaintenancePolicy
from llgm.memory.evidence import Evidence
from llgm.memory.query import QueryEvidence
from llgm.memory.workspace import Workspace
from llgm.models import create_model

pytestmark = [pytest.mark.integration, pytest.mark.dataset, pytest.mark.live, pytest.mark.docker]


@pytest.fixture
def live_application():
    """Require explicit service opt-in and exact models before any dataset or provider work."""
    enabled = os.environ.get("LLGM_TEST_APPLICATION", "0")
    if enabled == "0":
        pytest.skip("Set LLGM_TEST_APPLICATION=1 for the autonomous hosted application check")
    if enabled != "1":
        pytest.fail("LLGM_TEST_APPLICATION must be exactly 0 or 1", pytrace=False)
    configurations = {}
    for role in ("MAIN", "READER", "GRAPH"):
        prefix = f"LLGM_TEST_APPLICATION_{role}_"
        provider, model = (
            os.environ.get(prefix + key, "").strip() for key in ("PROVIDER", "MODEL")
        )
        if provider not in {"openai", "anthropic"}:
            pytest.fail(prefix + "PROVIDER must be openai or anthropic", pytrace=False)
        if not model or "latest" in model.lower():
            pytest.fail(prefix + "MODEL must be an explicit model identifier", pytrace=False)
        credential = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        if not os.environ.get(credential):
            pytest.fail("Opted-in application integration requires " + credential, pytrace=False)
        configurations[role.lower()] = {"provider": provider, "model": model}
    return configurations


@pytest.fixture
def application_docker(live_application):
    """Require the Docker gate and pin an already-local image before constructing hosted clients."""
    enabled = os.environ.get("LLGM_TEST_DOCKER", "0")
    if enabled == "0":
        pytest.skip("Set LLGM_TEST_DOCKER=1 for Python node delegates")
    if enabled != "1":
        pytest.fail("LLGM_TEST_DOCKER must be exactly 0 or 1", pytrace=False)
    requested = os.environ.get("LLGM_REPL_DOCKER_IMAGE", "").strip()
    if not requested:
        pytest.fail("Set LLGM_REPL_DOCKER_IMAGE to a trusted local image", pytrace=False)
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", requested],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.fail(
            "Application integration requires an available local Docker daemon", pytrace=False
        )
    identity = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
        pytest.fail("Configured Docker image is unavailable; no image was pulled", pytrace=False)
    return {
        "requested_image": requested,
        "image_id": identity,
        "config": DockerREPLConfig(image=identity),
    }


def _safe_trace(trace):
    """Retain operation identities and usage without copying source text or provider error bodies."""
    records = []
    for event in trace:
        record = {
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
                "usage_extra",
                "context_accounting_units",
                "error_type",
                "reason",
                "references",
                "citations",
                "source",
                "selected",
                "skipped",
                "target_node_id",
                "admitted_seed_nodes",
                "max_concurrency",
                "sha256",
                "passage_hits",
                "retrieval_k",
            }
        }
        operation = event.get("operation")
        if isinstance(operation, str):
            record["operation"] = operation
        elif isinstance(operation, dict):
            record["operation"] = {
                key: value
                for key, value in operation.items()
                if key
                in {
                    "op",
                    "reference",
                    "references",
                    "node_id",
                    "relation",
                    "k",
                    "citations",
                }
            }
            for key in ("question", "query", "answer"):
                if isinstance(operation.get(key), str):
                    record["operation"][key + "_sha256"] = hashlib.sha256(
                        operation[key].encode()
                    ).hexdigest()
        records.append(record)
    return records


def test_autonomous_memory_application(
    live_application, application_docker, longmemeval, tmp_path, integration_record
):
    """Real models maintain a full history and answer its original question without oracle routing."""
    case = longmemeval.cases["001be529"]
    gold = longmemeval.gold[case.case_id]
    record, directory = integration_record
    maintenance_policy = MaintenancePolicy(
        max_nodes=2,
        max_candidates=4,
        max_links_per_node=2,
        budget=Budget(
            max_model_calls=2,
            max_reader_calls=2,
            max_searches=2,
            max_context_tokens=65536,
            max_evidence_tokens=60000,
            max_output_tokens=2048,
            timeout_seconds=180,
        ),
    )
    inference_budget = Budget(
        max_model_calls=16,
        max_reader_calls=12,
        max_searches=8,
        max_evidence_tokens=60000,
        max_bundle_tokens=24000,
        max_context_tokens=131072,
        max_output_tokens=2048,
        timeout_seconds=180,
    )
    record.update(
        {
            "dataset": longmemeval.manifest,
            "case_id": case.case_id,
            "evidence_schema": "immutable-node-v2",
            "read_policy": "current",
            "workspace_schema": 3,
            "execution": "python-node-delegates-v1",
            "docker": {key: value for key, value in application_docker.items() if key != "config"},
            "seed_policy": {"max_seed_nodes": 3, "retrieval_k": 10, "max_concurrency": 2},
            "scope": "autonomous application sanity; full history; ordinary seed retrieval and Python delegates; no oracle handles; semantic quality scored separately",
            "models": live_application,
            "maintenance_policy": asdict(maintenance_policy),
            "inference_budget": asdict(inference_budget),
            "maintenance_node_selection": "first two input source occurrences; no evaluator labels",
        }
    )

    async def scenario():
        """Publish real source histories, run bounded maintenance, and retain every inference attempt."""
        async with Workspace.open(tmp_path / "workspace") as workspace:
            async with AsyncExitStack() as stack:
                clients = {
                    role: await stack.enter_async_context(
                        create_model(**config, timeout_seconds=60)
                    )
                    for role, config in live_application.items()
                }
                app = LLGM(
                    workspace,
                    clients["main"],
                    clients["reader"],
                    graph_model=clients["graph"],
                    maintenance_policy=maintenance_policy,
                    inference_budget=inference_budget,
                    max_depth=3,
                    max_steps=12,
                    max_operations=32,
                    max_seed_nodes=3,
                    retrieval_k=10,
                    max_concurrency=2,
                    repl_config=application_docker["config"],
                )
                for source in case.sources:
                    await app.ingest(
                        Conversation(
                            source.turns, source.node_id, source.metadata, source.timestamp_ms
                        ),
                        organize=False,
                    )
                maintenance = await app.organize([source.node_id for source in case.sources[:2]])
                record["maintenance"] = {
                    "status": maintenance.status,
                    "usage": dict(maintenance.usage),
                    "proposal_count": len(maintenance.proposals),
                    "accepted_count": len(maintenance.accepted),
                    "decisions": maintenance.decisions,
                    "error_type": maintenance.error_type,
                    "operations": _safe_trace(maintenance.trace),
                    "support": [
                        [reference_to_dict(ref) for ref in entry.provenance.supporting_references]
                        for entry in maintenance.accepted
                    ],
                    "accepted_edge_ids": [edge.edge_id for edge in maintenance.accepted],
                }
                for edge in maintenance.accepted:
                    assert (
                        await workspace.edge(edge.edge_id)
                    ).target_node_id == edge.target_node_id
                assert all(
                    [
                        not await workspace.inspect_journal(source.node_id)
                        for source in case.sources[:2]
                    ]
                )
                write_json(directory / "application-attempt.json", record)
                assert maintenance.status == "completed", (
                    "Maintenance failed; attempted usage is retained"
                )
                assert maintenance.usage["model_calls"] > 0
                try:
                    result = await app.answer(
                        case.question, query_date=case.question_date, remember=False
                    )
                finally:
                    record["operations"], record["usage"] = (
                        _safe_trace(app.last_trace),
                        app.last_usage,
                    )
                    write_json(directory / "application-attempt.json", record)
                predictions = directory / "predictions.jsonl"
                write_jsonl(
                    predictions, [{"question_id": case.case_id, "hypothesis": result.answer}]
                )
                write_json(
                    directory / "prediction-manifest.json",
                    {
                        "schema_version": 1,
                        "dataset_sha256": longmemeval.manifest["sha256"],
                        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                        "case_ids": [case.case_id],
                        "models": live_application,
                        "evidence_schema": "immutable-node-v2",
                        "read_policy": "current",
                        "workspace_schema": 3,
                        "execution": "python-node-delegates-v1",
                        "scope": record["scope"],
                        "benchmark_result": False,
                        "status": result.status,
                    },
                )
                covered_spans = []
                sources = {source.node_id: source for source in case.sources}
                for reference in result.references:
                    assert (await workspace.resolve(reference)).text
                    if isinstance(reference, SourceSpan):
                        covered_spans.append(reference)
                    elif isinstance(reference, NodeRef):
                        source = sources[reference.node_id]
                        covered_spans.extend(
                            SourceSpan(source.node_id, turn.turn_id, 0, len(turn.text))
                            for turn in source.turns
                        )
                record["coverage"] = evidence_coverage(covered_spans, asdict(gold))
                record["coverage"]["annotation_granularity"] = (
                    "source and labeled-turn coverage, not answer-span correctness"
                )
                record["result"] = {
                    "status": result.status,
                    "references": [reference_to_dict(ref) for ref in result.references],
                    "unresolved_count": len(result.evidence.unresolved),
                }
                record["delegation_depths"] = sorted(
                    {event["depth"] for event in result.trace if event.get("kind") == "enter"}
                )
                write_json(directory / "application-attempt.json", record)
                assert result.status in {"completed", "partial"}, (
                    "Autonomous answer failed; inspect retained attempt and prediction"
                )
                assert result.answer.strip() and result.references
                selection = next(
                    event for event in result.trace if event.get("kind") == "seed_selection"
                )
                assert selection["source"] == "retrieval" and selection["selected"]
                entered = [
                    event
                    for event in result.trace
                    if event.get("kind") == "enter" and event["parent_id"] == "main"
                ]
                assert {event["target_node_id"] for event in entered} == set(selection["selected"])
                assert all(event["question"] == case.question for event in entered)
                assert any(event.get("kind") == "python" for event in result.trace)
                assert any(event.get("kind") == "repl_open" for event in result.trace)
                calls = [event for event in result.trace if event.get("kind") == "model"]
                assert sum(event["role"] == "main" for event in calls) == 1
                assert all(
                    event["model"] == live_application[event["role"]]["model"] for event in calls
                )
                assert json.loads(json.dumps(record))["benchmark_result"] is False

    asyncio.run(scenario())


def test_controlled_application_updates_restart_and_scopes(
    live_application, application_docker, tmp_path, integration_record
):
    """Python delegates use amended source spans and independent edges across scoped updates and restart."""
    record, directory = integration_record
    sources = [
        (
            "orion-service",
            "Orion service O-16. Production archive key is COBALT-41. Staging archive key is AMBER-26.",
        ),
        (
            "orion-runbook",
            "For Orion archive-key questions, follow the related Orion service evidence. Production and staging are separate environments. Draft suggestions are not adopted changes.",
        ),
        (
            "boreal-service",
            "Boreal production archive key is RUBY-98. Boreal is a separate service and its key does not apply to Orion.",
        ),
        (
            "orion-draft",
            "A fictional draft suggested VIOLET-12 for Orion production. The draft was never approved or adopted; it does not replace the adopted service decision.",
        ),
        ("recipe-notes", "A cooking note recommends one teaspoon of cumin for lentil soup."),
    ]
    update_node_id = "orion-service-update"
    update_text = (
        "Orion service is now labeled O-17. Its adopted production archive key is JADE-73, "
        "effective 2025-06-01. Staging is unchanged; drafts are not adopted facts."
    )
    cases = [
        {
            "id": "production-historical",
            "question": "What was Orion's adopted production archive key on 2025-05-15? Return the exact key.",
            "scope": {"env": "production"},
            "date": "2025-05-15",
            "expected": "COBALT-41",
            "evidence": "production_original",
        },
        {
            "id": "production-current",
            "question": "What was Orion's adopted production archive key on 2025-07-01? Return the exact key.",
            "scope": {"env": "production"},
            "date": "2025-07-01",
            "expected": "JADE-73",
            "evidence": "production_update",
        },
        {
            "id": "staging-current",
            "question": "What was Orion's adopted staging archive key on 2025-07-01? Return the exact key.",
            "scope": {"env": "staging"},
            "date": "2025-07-01",
            "expected": "AMBER-26",
            "evidence": "staging",
        },
    ]
    for case in cases:
        case["as_of_ms"] = parse_instant_ms(case["date"], date_only_timezone=timezone.utc)
    policy = MaintenancePolicy(
        max_nodes=2,
        max_candidates=3,
        max_links_per_node=2,
        budget=Budget(
            max_model_calls=2,
            max_reader_calls=2,
            max_searches=2,
            max_context_tokens=65536,
            max_output_tokens=2048,
            timeout_seconds=120,
        ),
    )
    budget = Budget(
        max_model_calls=12,
        max_reader_calls=9,
        max_searches=5,
        max_evidence_tokens=60000,
        max_bundle_tokens=24000,
        max_context_tokens=131072,
        max_output_tokens=2048,
        timeout_seconds=180,
    )
    record.update(
        {
            "scope": "controlled autonomous application sanity; immutable source nodes, current journal reads, and explicit journal valid-time",
            "models": live_application,
            "source_count": len(sources) + 1,
            "evidence_schema": "immutable-node-v2",
            "read_policy": "current",
            "workspace_schema": 3,
            "execution": "python-node-delegates-v1",
            "docker": {key: value for key, value in application_docker.items() if key != "config"},
            "seed_policy": {"max_seed_nodes": 3, "retrieval_k": 10, "max_concurrency": 2},
            "maintenance_policy": asdict(policy),
            "inference_budget_per_answer": asdict(budget),
            "maintenance": [],
            "answers": [],
            "benchmark_result": False,
        }
    )
    # Freeze workload and expected checks before constructing hosted clients.
    write_json(
        directory / "controlled-workload.json",
        {
            "sources": sources,
            "source_update": {"node_id": update_node_id, "text": update_text},
            "questions": cases,
            "acceptance": "Every answer completes or reports partial evidence under the declared seed/budget policy, contains its expected key, and cites canonical supporting source text. Direct effective reads verify amendment provenance before inference. Maintenance completes under declared call limits. All attempts are retained.",
            "transcript_notice": "Full transcripts contain only this public synthetic workload and configured model outputs; no credentials are written.",
        },
    )

    def save_maintenance(result):
        """Retain automatic maintenance decisions and attempted usage before outcome checks."""
        record["maintenance"].append(
            {
                "status": result.status,
                "usage": dict(result.usage),
                "error_type": result.error_type,
                "decisions": result.decisions,
                "accepted_count": len(result.accepted),
                "proposal_count": len(result.proposals),
                "operations": _safe_trace(result.trace),
                "accepted_edge_ids": [edge.edge_id for edge in result.accepted],
            }
        )
        write_json(directory / "controlled-attempt.json", record)

    async def scenario():
        """Append a new source and a journal overwrite, restart, and ask three unmodified questions."""
        async with AsyncExitStack() as stack:
            clients = {
                role: await stack.enter_async_context(create_model(**config, timeout_seconds=60))
                for role, config in live_application.items()
            }
            async with Workspace.open(tmp_path / "controlled-workspace") as workspace:
                app = LLGM(
                    workspace,
                    clients["main"],
                    clients["reader"],
                    graph_model=clients["graph"],
                    maintenance_policy=policy,
                    inference_budget=budget,
                    max_depth=3,
                    max_steps=12,
                    capture_text=True,
                    max_seed_nodes=3,
                    retrieval_k=10,
                    max_concurrency=2,
                    repl_config=application_docker["config"],
                )
                for node_id, text in sources:
                    await app.ingest(
                        Conversation.from_turns(
                            [{"role": "user", "turn_id": "t", "text": text}], node_id=node_id
                        ),
                        organize=False,
                    )
                provenance = Provenance("user", "controlled-workload-v3")
                production_ref = SourceSpan(
                    "orion-service",
                    "t",
                    sources[0][1].index("Production"),
                    sources[0][1].index(" Staging"),
                )
                staging_ref = SourceSpan(
                    "orion-service", "t", sources[0][1].index("Staging"), len(sources[0][1])
                )
                curated_edge = await workspace.publish_edge(
                    "orion-runbook", "orion-service", provenance=provenance
                )
                record["curated_primary_edge_id"] = curated_edge.edge_id
                original = await workspace.append_journal(
                    "orion-service",
                    subject=production_ref,
                    relation="current_evidence",
                    value=production_ref,
                    provenance=provenance,
                    applicability={
                        "scope": {"env": "production"},
                        "valid_from_ms": parse_instant_ms("2025-01-01T00:00:00Z"),
                    },
                )
                staging = await workspace.append_journal(
                    "orion-service",
                    subject=staging_ref,
                    relation="current_evidence",
                    value=staging_ref,
                    provenance=provenance,
                    applicability={
                        "scope": {"env": "staging"},
                        "valid_from_ms": parse_instant_ms("2025-01-01T00:00:00Z"),
                    },
                )
                save_maintenance(await app.organize(["orion-service", "orion-runbook"]))
                update = await app.ingest(
                    Conversation.from_turns(
                        [{"role": "user", "turn_id": "t", "text": update_text}],
                        node_id=update_node_id,
                        timestamp_ms=parse_instant_ms("2025-06-01T00:00:00Z"),
                    )
                )
                save_maintenance(update.maintenance)
                update_ref = SourceSpan(update_node_id, "t", 0, len(update_text))
                production_update = await workspace.append_journal(
                    "orion-service",
                    subject=production_ref,
                    relation="current_evidence",
                    record_kind="overwrite",
                    value=update_ref,
                    provenance=Provenance(
                        "user", "controlled-workload-v3", supporting_references=(update_ref,)
                    ),
                    applicability={
                        "scope": {"env": "production"},
                        "valid_from_ms": parse_instant_ms("2025-06-01T00:00:00Z"),
                    },
                )
                required_entries = {
                    "production_original": original.entry_id,
                    "production_update": production_update.entry_id,
                    "staging": staging.entry_id,
                }
                record["source_update"] = asdict(update.source)
                record["expected_journal_entries"] = required_entries
                write_json(directory / "controlled-attempt.json", record)

            async with Workspace.open(tmp_path / "controlled-workspace") as reopened:
                app = LLGM(
                    reopened,
                    clients["main"],
                    clients["reader"],
                    graph_model=clients["graph"],
                    maintenance_policy=policy,
                    inference_budget=budget,
                    max_depth=3,
                    max_steps=12,
                    capture_text=True,
                    max_seed_nodes=3,
                    retrieval_k=10,
                    max_concurrency=2,
                    repl_config=application_docker["config"],
                )
                record["reopened_source_node_ids"] = list(await reopened.source_ids())
                record["original_source_retained"] = (
                    await reopened.resolve(SourceSpan("orion-service", "t", 0, len(sources[0][1])))
                ).text == sources[0][1]
                record["new_source_retained"] = (
                    await reopened.resolve(update_ref)
                ).text == update_text
                record["historical_source_statement_retained"] = (
                    "COBALT-41" in (await reopened.resolve(production_ref)).text
                )
                assert (await reopened.edge(curated_edge.edge_id)).target_node_id == "orion-service"
                assert await reopened.inspect_journal("orion-runbook") == []
                for case in cases:
                    attempt = {
                        "case_id": case["id"],
                        "expected": case["expected"],
                        "required_entry_id": required_entries[case["evidence"]],
                    }
                    try:
                        async with await Evidence.open(reopened) as evidence:
                            scoped = QueryEvidence(
                                evidence, case["scope"], case["date"], as_of_ms=case["as_of_ms"]
                            )
                            subject = (
                                staging_ref if case["scope"]["env"] == "staging" else production_ref
                            )
                            segments = await scoped.read_segments(subject)
                            attempt["effective_read_contains_expected"] = any(
                                case["expected"] in part.text for part in segments
                            )
                            attempt["effective_read_amendment_provenance"] = any(
                                amendment["entry_id"] == required_entries[case["evidence"]]
                                for part in segments
                                for amendment in part.metadata.get("amendments", [])
                            )
                        result = await app.answer(
                            case["question"],
                            scope=case["scope"],
                            query_date=case["date"],
                            as_of_ms=case["as_of_ms"],
                            remember=False,
                        )
                        attempt.update(
                            status=result.status,
                            answer=result.answer,
                            citations=[reference_to_dict(ref) for ref in result.references],
                            unresolved=result.evidence.unresolved,
                        )
                        attempt["contains_expected_key"] = (
                            case["expected"].casefold() in result.answer.casefold()
                        )
                        expected_owner = (
                            update_node_id
                            if case["evidence"] == "production_update"
                            else "orion-service"
                        )
                        attempt["cites_expected_source_text"] = any(
                            [
                                ref.node_id == expected_owner
                                and case["expected"] in (await reopened.resolve(ref)).text
                                for ref in result.references
                                if isinstance(ref, (SourceSpan, NodeRef))
                            ]
                        )
                        attempt["citations_resolve"] = all(
                            [(await reopened.resolve(ref)).text != "" for ref in result.references]
                        )
                    except Exception as error:
                        attempt.update(status="exception", error_type=type(error).__name__)
                    finally:
                        attempt["usage"] = app.last_usage
                        attempt["operations"] = _safe_trace(app.last_trace)
                        write_json(directory / (case["id"] + "-transcript.json"), app.last_trace)
                        record["answers"].append(attempt)
                        write_json(directory / "controlled-attempt.json", record)
        assert {"orion-service", update_node_id}.issubset(record["reopened_source_node_ids"])
        assert record["original_source_retained"] and record["new_source_retained"]
        assert record["historical_source_statement_retained"]
        assert all(row["status"] == "completed" for row in record["maintenance"])
        assert 1 <= sum(row["usage"]["model_calls"] for row in record["maintenance"]) <= 4
        assert len(record["answers"]) == len(cases)
        for attempt in record["answers"]:
            assert attempt["status"] in {"completed", "partial"}, attempt["case_id"]
            assert attempt["contains_expected_key"], attempt["case_id"]
            assert attempt["cites_expected_source_text"] and attempt["citations_resolve"], attempt[
                "case_id"
            ]
            assert (
                attempt["effective_read_contains_expected"]
                and attempt["effective_read_amendment_provenance"]
            ), attempt["case_id"]

    asyncio.run(scenario())
