"""Explicit original-question retrieval diagnostics over isolated case corpora.

This runner makes no generation calls. D/H still make explicitly requested
hosted embedding calls, and C uses the real official ColBERT/PLAID stack.
Importing the module neither loads a checkpoint nor initializes a provider.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.evaluation.artifacts import RunArtifacts, read_jsonl, write_json
from llgm.evaluation.matrix import BACKENDS, preflight, resolve_backend_scope, tokenizer_settings
from llgm.evaluation.prepare import file_sha256
from llgm.evaluation.runner import (
    _CappedEmbedder,
    _close_resources,
    _code_provenance,
    _drain_local,
    _redacted,
    _RunLimits,
)
from llgm.evaluation.scoring import evidence_coverage


class _CaseEmbedder:
    """Bound physical embedding requests at both case and experiment levels."""

    def __init__(self, embedder: _CappedEmbedder, maximum: int, deadline: float):
        """Add a case request allowance and monotonic deadline around the run-capped embedder."""
        self.embedder, self.maximum, self.deadline = embedder, maximum, deadline
        self.model = embedder.model
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Charge nonempty attempts to the case allowance before applying run-level limits."""
        if not texts:
            return []
        if self.calls >= self.maximum:
            raise BudgetExceeded("Per-case embedding-request allowance exhausted")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExceeded("Per-case retrieval diagnostic deadline exhausted")
        self.calls += 1
        async with asyncio.timeout(remaining):
            return await self.embedder.embed(texts)

    def descriptor(self) -> dict:
        """Preserve the underlying embedding model and provider descriptor."""
        return self.embedder.descriptor()

    @property
    def events(self):
        """Expose provider events for complete physical-request accounting."""
        return self.embedder.events


def _remaining(deadline: float, limits: _RunLimits) -> float:
    """Return the tighter case or run time allowance, raising when either is exhausted."""
    remaining = min(deadline - time.monotonic(), limits.remaining())
    if remaining <= 0:
        raise BudgetExceeded("Per-case retrieval diagnostic deadline exhausted")
    return remaining


async def _local_call(function, *args, deadline: float, limits: _RunLimits, **kwargs):
    """Run local native work to completion, checking its cooperative deadline afterward."""
    _remaining(deadline, limits)
    result = await _drain_local(asyncio.create_task(asyncio.to_thread(function, *args, **kwargs)))
    _remaining(deadline, limits)
    return result


def _prepared_path(prepared: Path, relative: str) -> Path:
    """Resolve an artifact path and reject escapes outside its preparation directory."""
    path = (prepared / relative).resolve()
    if not path.is_relative_to(prepared.resolve()):
        raise ConfigurationError(
            "Prepared artifact paths must stay inside their preparation directory"
        )
    return path


def _coverage(hits: list, gold: dict) -> dict[str, Any]:
    """Score only after retrieval; labels never influence candidate selection."""
    references = [ref for hit in hits for ref in hit.passage.refs]
    coverage = evidence_coverage(references, gold)
    aliases = gold.get("source_aliases", {})
    retrieved_nodes = {aliases.get(ref.node_id, ref.node_id) for ref in references}
    retrieved_turns = {(ref.node_id, ref.turn_id) for ref in references}
    abstention = gold.get("ability") == "abstention"
    source, turn = coverage["source_recall"], coverage["answer_turn_recall"]
    all_sources, all_turns = (
        coverage["all_required_sources_covered"],
        coverage["all_answer_turns_hit"],
    )
    available = [value for value in (all_sources, all_turns) if value is not None]
    return {
        **coverage,
        "ability": gold.get("ability"),
        "abstention": abstention,
        "source_recall": source,
        "turn_recall": turn,
        "all_required_source_coverage": all_sources,
        "all_required_turn_coverage": all_turns,
        "all_required_coverage": all(available) if available else None,
        "required_source_count": None if abstention else coverage["source_denominator"],
        "required_turn_count": None if abstention else coverage["answer_turn_denominator"],
        "retrieved_source_count": len(retrieved_nodes),
        "retrieved_turn_count": len(retrieved_turns),
    }


def _embedding_usage(events: list[dict]) -> dict[str, Any]:
    """Aggregate observed requests while keeping unknown token usage distinct from zero."""
    values = [event.get("usage", {}).get("input_tokens") for event in events]
    known = [value for value in values if value is not None]
    return {
        "embedding_requests": len(events),
        "embedding_input_tokens": sum(known) if len(known) == len(values) else None,
        "known_embedding_input_tokens": sum(known),
        "embedding_requests_with_unknown_usage": len(values) - len(known),
        "generation_calls": 0,
        "currency_cost": None,
    }


def _metrics(observations: list[dict]) -> dict[str, Any]:
    """Aggregate retrieval coverage, latency, and usage with explicit scoring denominators."""
    metrics: dict[str, Any] = {
        "case_count": len(observations),
        "completed": sum(row["status"] == "completed" for row in observations),
        "failed": sum(row["status"] != "completed" for row in observations),
        "abstention_cases": sum(row["abstention"] for row in observations),
        "generation_calls": 0,
        "official_answer_score": None,
        "currency_cost": None,
        "currency_note": "Unpriced; all physical embedding usage, including failed calls, is retained",
        "scope": "Original-question top40 retrieval; no generation or answer-quality evaluation",
    }
    for name in (
        "source_recall",
        "turn_recall",
        "all_required_source_coverage",
        "all_required_turn_coverage",
        "all_required_coverage",
    ):
        values = [row[name] for row in observations if row[name] is not None]
        metrics[name] = sum(values) / len(values) if values else None
        metrics[name + "_denominator"] = len(values)
    for name in ("index_seconds", "query_seconds", "elapsed_seconds"):
        values = sorted(row[name] for row in observations if row[name] is not None)
        metrics["total_" + name] = sum(values)
        metrics["mean_" + name] = sum(values) / len(values) if values else None
        metrics["p95_" + name] = (
            values[max(0, math.ceil(0.95 * len(values)) - 1)] if values else None
        )
        metrics[name + "_denominator"] = len(values)
    metrics["embedding_requests"] = sum(row["usage"]["embedding_requests"] for row in observations)
    metrics["known_embedding_input_tokens"] = sum(
        row["usage"]["known_embedding_input_tokens"] for row in observations
    )
    metrics["embedding_requests_with_unknown_usage"] = sum(
        row["usage"]["embedding_requests_with_unknown_usage"] for row in observations
    )
    metrics["embedding_input_tokens"] = (
        None
        if metrics["embedding_requests_with_unknown_usage"]
        else metrics["known_embedding_input_tokens"]
    )
    return metrics


async def run_retrieval_diagnostic(
    prepared: str | Path,
    output: str | Path,
    matrix: dict,
    *,
    backends: list[str] | None = None,
    split: str = "smoke",
    required_backends: list[str] | None = None,
) -> dict:
    """Run real B/D/H/C single-query retrieval with no text generation.

    Calling this function explicitly authorizes the configured embedding API
    requests. The CLI separately requires ``--execute``. Read-only preflight
    checks all four backends unless ``required_backends`` explicitly narrows
    readiness. Dispatched backends must stay inside that declared scope.
    Every backend builds an isolated cold index for each question's haystack.
    """
    scope = resolve_backend_scope(required_backends)
    chosen = list(scope) if backends is None else list(backends)
    if (
        not chosen
        or any(not isinstance(backend, str) for backend in chosen)
        or len(set(chosen)) != len(chosen)
        or not set(chosen).issubset(BACKENDS)
    ):
        raise ConfigurationError("Diagnostic backends must be unique nonempty members of B/D/H/C")
    if not set(chosen).issubset(scope):
        raise ConfigurationError(
            "Requested backends fall outside the explicit required backend scope"
        )
    if split not in {"smoke", "development", "evaluation"}:
        raise ConfigurationError("Choose smoke, development, or explicitly reserved evaluation")
    prepared, output = Path(prepared), Path(output)
    report = preflight(
        prepared,
        matrix,
        retrieval_only=True,
        **({"required_backends": scope} if required_backends is not None else {}),
    )
    if not report["ready"]:
        raise ConfigurationError(
            "Retrieval diagnostic preflight is blocked; inspect all four backend requirements or the explicitly declared scope"
        )
    if output.exists() and any(output.iterdir()):
        raise ConfigurationError("Retrieval diagnostic output directory must be new or empty")
    manifest = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.get("files", {}).items():
        if file_sha256(_prepared_path(prepared, relative)) != expected:
            raise ConfigurationError(f"Prepared artifact changed: {relative}")
    selection_key = {
        "smoke": "smoke_ids",
        "development": "development_ids",
        "evaluation": "evaluation_ids",
    }[split]
    selected = set(manifest["selection"].get(selection_key, ()))
    if not selected:
        raise ConfigurationError(f"Preparation has no selected {split} cases")
    queries = [row for row in read_jsonl(prepared / "queries.jsonl") if row["case_id"] in selected]
    if len(queries) != len(selected) or {row["case_id"] for row in queries} != selected:
        raise ConfigurationError("Prepared queries must cover each selected case exactly once")
    # Evaluator data is only accessed by _coverage after each retrieval attempt.
    evaluator = {row["case_id"]: row for row in read_jsonl(prepared / "evaluator/gold.jsonl")}
    if not selected.issubset(evaluator):
        raise ConfigurationError("Prepared evaluator labels do not cover the selected cases")

    from llgm.inference.budget import Budget
    from llgm.models import OpenAIEmbeddingClient
    from llgm.retrieval import (
        ColBERTTokenizer,
        ExactDenseRetriever,
        HybridRetriever,
        SQLiteBM25Retriever,
        validate_encoder_text,
    )
    from llgm.retrieval.base import corpus_fingerprint, passage_from_dict

    tokenizer = ColBERTTokenizer(**tokenizer_settings(matrix))
    if tokenizer.descriptor()["sha256"] != manifest["tokenizer"].get("sha256"):
        raise ConfigurationError(
            "Retrieval diagnostic tokenizer differs from the prepared passage tokenizer"
        )
    budget = Budget(**matrix["budgets"])
    limits = _RunLimits(matrix["overall"])
    controls = matrix.get("retrieval_diagnostic", {})
    configured_embedding_cap = controls.get("max_embedding_requests_per_case")
    if configured_embedding_cap is not None and (
        type(configured_embedding_cap) is not int or configured_embedding_cap < 1
    ):
        raise ConfigurationError(
            "retrieval_diagnostic.max_embedding_requests_per_case must be a positive integer"
        )
    output.mkdir(parents=True, exist_ok=True)
    provenance = _code_provenance()
    prepared_sha256 = file_sha256(prepared / "prepared.json")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "mode": "retrieval-only",
        "benchmark_result": False,
        "prepared_sha256": prepared_sha256,
        "split": split,
        "backends": {},
        "all_matrix_backends": list(BACKENDS),
        "required_backends": scope,
        "selected_backends": chosen,
        "recorded_backends": [],
        "completed_backends": [],
        "full_matrix_ready": report.get("full_matrix_ready", False),
        "full_matrix_complete": False,
        "full_retrieval_matrix_complete": False,
        "query_policy": "original question once",
        "visible_hit_limit": 40,
        "generation_calls": 0,
    }
    write_json(output / "preflight.json", report)
    for backend in chosen:
        artifacts = RunArtifacts(
            output / backend,
            {
                "schema_version": 1,
                "run_id": output.name,
                "mode": "retrieval-only",
                "backend": backend,
                "required_backends": scope,
                "selected_backends": chosen,
                "full_matrix_ready": report.get("full_matrix_ready", False),
                "query_policy": "original question once",
                "visible_hit_limit": 40,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "matrix": _redacted(matrix),
                "prepared": manifest,
                "prepared_sha256": prepared_sha256,
                "case_ids": [query["case_id"] for query in queries],
                "split": split,
                "code": provenance,
                "cache_policy": "cold per case and backend; D and H encoding charged independently",
                "source_cutoff": "only each question's supplied haystack",
                "generation_calls": 0,
                "retries": "no application or SDK retries",
                "query_preprocessing": "none; overflow fails before indexing or encoding",
                "embedding_request_cap_per_case": configured_embedding_cap
                or "ceil(passage_count / 64) + 1",
                "case_timeout_seconds": budget.timeout_seconds,
                "local_deadline_semantics": "cooperative: native ColBERT work is drained; overruns fail and prevent subsequent work",
                "recall_semantics": "annotated source/turn IDs only; abstention and missing labels have no denominator",
                "all_required_coverage_semantics": "all available required source AND turn labels retrieved",
                "turn_recall_limitation": "a retrieved passage intersects a labeled turn; no gold passage-span labels are inferred",
            },
        )
        observations: list[dict] = []
        for query in queries:
            case_id = query["case_id"]
            case_index_name = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:24]
            started = time.monotonic()
            deadline = started + budget.timeout_seconds
            artifacts.append(
                "cases",
                {
                    **query,
                    "backend": backend,
                    "prepared_inputs": {
                        "prepared_sha256": prepared_sha256,
                        "sources_sha256": manifest["files"][query["sources_path"]],
                    },
                },
            )
            resources: list[Any] = []
            retriever = lexical = dense = embedding = None
            hits: list = []
            status, error_type, phase = "failed", None, "preparation"
            cancelled = None
            index_seconds = query_seconds = None
            embedding_events: list[dict] = []
            descriptor = None
            try:
                limits.take("case_runs", "max_case_arm_runs")
                _remaining(deadline, limits)
                source = json.loads(
                    _prepared_path(prepared, query["sources_path"]).read_text(encoding="utf-8")
                )
                if source.get("case_id") != case_id or source.get("question") != query["question"]:
                    raise ConfigurationError("Prepared query differs from its isolated source case")
                passages = [
                    passage_from_dict(row)
                    for row in read_jsonl(_prepared_path(prepared, query["passages_path"]))
                ]
                allowed_refs = {
                    (node["node_id"], turn["turn_id"]): turn["text"]
                    for node in source["sources"]
                    for turn in node["turns"]
                }
                validate_encoder_text(
                    query["question"], tokenizer, matrix["colbert"]["query_maxlen"], "query"
                )
                for passage in passages:
                    validate_encoder_text(
                        passage.text, tokenizer, matrix["colbert"]["doc_maxlen"], "document"
                    )
                    for ref in passage.refs:
                        original = allowed_refs.get((ref.node_id, ref.turn_id))
                        if original is None or not 0 <= ref.start <= ref.end <= len(original):
                            raise ConfigurationError(
                                "A retrieved passage reference falls outside this case's original sources"
                            )
                artifacts.append(
                    "traces",
                    {
                        "case_id": case_id,
                        "kind": "corpus_validated",
                        "passage_count": len(passages),
                        "corpus_sha256": corpus_fingerprint(passages),
                        "query_sha256": hashlib.sha256(
                            query["question"].encode("utf-8")
                        ).hexdigest(),
                        "query_tokens": tokenizer.count(query["question"], kind="query"),
                    },
                )
                _remaining(deadline, limits)
                phase = "index"
                index_started = time.monotonic()
                try:
                    index_directory = artifacts.directory / "indexes" / case_index_name
                    if backend in {"B", "H"}:
                        index_directory.mkdir(parents=True, exist_ok=False)
                        lexical = SQLiteBM25Retriever.from_passages(
                            passages, index_directory / "bm25.sqlite3"
                        )
                        resources.append(lexical)
                    if backend in {"D", "H"}:
                        client = OpenAIEmbeddingClient(**matrix["dense"], batch_size=128)
                        resources.append(client)
                        maximum = configured_embedding_cap or math.ceil(len(passages) / 64) + 1
                        embedding = _CaseEmbedder(
                            _CappedEmbedder(client, limits), maximum, deadline
                        )
                        dense = await ExactDenseRetriever.build(passages, embedding, batch_size=64)
                        index_directory.mkdir(parents=True, exist_ok=True)
                        dense.save(index_directory / "dense.json")
                    if backend == "B":
                        retriever = lexical
                    elif backend == "D":
                        retriever = dense
                    elif backend == "H":
                        retriever = HybridRetriever(lexical, dense, **matrix["hybrid"])
                    else:
                        from llgm.retrieval.colbert import ColBERTConfig, ColBERTRetriever

                        config = ColBERTConfig(
                            **matrix["colbert"],
                            index_root=artifacts.directory / "indexes",
                            index_name=case_index_name,
                        )
                        retriever = await _local_call(
                            ColBERTRetriever.build,
                            passages,
                            config=config,
                            tokenizer=tokenizer,
                            deadline=deadline,
                            limits=limits,
                        )
                    descriptor = retriever.descriptor()
                finally:
                    index_seconds = time.monotonic() - index_started
                artifacts.append(
                    "traces",
                    {
                        "case_id": case_id,
                        "kind": "index_ready",
                        "descriptor": descriptor,
                        "index_seconds": index_seconds,
                    },
                )
                _remaining(deadline, limits)
                phase = "query"
                query_started = time.monotonic()
                try:
                    if backend == "C":
                        hits = await _drain_local(
                            asyncio.create_task(retriever.search(query["question"], 40))
                        )
                    else:
                        async with asyncio.timeout(_remaining(deadline, limits)):
                            hits = await retriever.search(query["question"], 40)
                    _remaining(deadline, limits)
                    if len(hits) > 40:
                        raise ConfigurationError(
                            "Backend exceeded the shared visible hit allowance"
                        )
                finally:
                    query_seconds = time.monotonic() - query_started
                status = "completed"
            except asyncio.CancelledError as exc:
                cancelled = exc
                status, error_type = "cancelled", "CancelledError"
                artifacts.append(
                    "traces", {"case_id": case_id, "kind": "cancelled", "phase": phase}
                )
            except Exception as exc:
                error_type = type(exc).__name__
                status = (
                    "budget_exhausted"
                    if isinstance(exc, (BudgetExceeded, TimeoutError))
                    else "failed"
                )
                # Partially completed native work may have returned results after
                # the deadline. Such hits are retained in predictions for audit,
                # but never count as successful evidence in recall metrics.
                artifacts.append(
                    "traces",
                    {
                        "case_id": case_id,
                        "kind": "failure",
                        "phase": phase,
                        "error_type": error_type,
                    },
                )
            finally:
                if embedding is not None:
                    for event in embedding.events:
                        event_dict = asdict(event) if is_dataclass(event) else dict(event)
                        embedding_events.append(event_dict)
                        artifacts.append(
                            "usage", {"case_id": case_id, "category": "embedding", **event_dict}
                        )
                seen: set[int] = set()
                for item in (retriever, lexical, dense):
                    if item is None or id(item) in seen:
                        continue
                    seen.add(id(item))
                    for event in getattr(item, "events", ()):
                        artifacts.append(
                            "traces", {"case_id": case_id, "kind": "retrieval_backend", **event}
                        )
                cleanup_cancelled = await _close_resources(resources, artifacts, case_id)
                if cleanup_cancelled is not None:
                    cancelled = cancelled or cleanup_cancelled
                    status, error_type, phase = "cancelled", "CancelledError", "cleanup"
                    artifacts.append(
                        "traces", {"case_id": case_id, "kind": "cancelled", "phase": phase}
                    )
            usage = _embedding_usage(embedding_events)
            artifacts.append(
                "usage",
                {
                    "case_id": case_id,
                    "category": "retrieval_summary",
                    "backend": backend,
                    **usage,
                    "index_seconds": index_seconds,
                    "query_seconds": query_seconds,
                },
            )
            prediction = {
                "case_id": case_id,
                "question_id": case_id,
                "backend": backend,
                "status": status,
                "error_type": error_type,
                "failure_phase": phase if status != "completed" else None,
                "hits": [
                    {
                        "passage_id": hit.passage.passage_id,
                        "rank": hit.rank,
                        "score": hit.score,
                        "references": hit.passage.refs,
                    }
                    for hit in hits
                ],
                "returned_hit_count": len(hits),
                "usage": usage,
                "index_seconds": index_seconds,
                "query_seconds": query_seconds,
                "elapsed_seconds": time.monotonic() - started,
            }
            artifacts.append("predictions", prediction)
            judgment = {
                "case_id": case_id,
                "scorer": "annotated_source_and_turn_retrieval",
                "official_answer_score": None,
                **_coverage(hits if status == "completed" else [], evaluator[case_id]),
            }
            artifacts.append("judgments", judgment)
            observations.append({**prediction, **judgment})
            if cancelled is not None:
                metrics = {**_metrics(observations), "cancelled": True}
                artifacts.finish(
                    metrics,
                    "Run cancelled; partial retrieval predictions and observed embedding usage are preserved. Unstarted cases were not evaluated.",
                )
                summary["backends"][backend] = metrics
                summary["cancelled"] = True
                summary["physical_embedding_requests"] = limits.embeddings
                summary["case_backend_runs"] = limits.case_runs
                summary["elapsed_seconds"] = time.monotonic() - limits.started
                write_json(output / "summary.json", summary)
                raise cancelled
        metrics = _metrics(observations)
        metrics["by_ability"] = {
            ability: _metrics([row for row in observations if row["ability"] == ability])
            for ability in sorted({row["ability"] for row in observations})
        }
        artifacts.finish(
            metrics,
            "Original-question retrieval diagnostic only. No generation calls or answer-quality "
            "scores were produced. Source/turn coverage does not establish IterativeRuntime benchmark superiority.",
        )
        summary["backends"][backend] = metrics
        summary["recorded_backends"].append(backend)
        if metrics["completed"] == len(observations):
            summary["completed_backends"].append(backend)
        summary["full_retrieval_matrix_complete"] = set(summary["completed_backends"]) == set(
            BACKENDS
        )
        summary["physical_embedding_requests"] = limits.embeddings
        summary["case_backend_runs"] = limits.case_runs
        summary["elapsed_seconds"] = time.monotonic() - limits.started
        write_json(output / "summary.json", summary)
    return summary
