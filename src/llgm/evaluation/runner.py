"""Explicit experiment execution through the public IterativeRuntime runtime.

Imports, preparation, and preflight make no paid calls. ``run_matrix`` is the
dispatch boundary. Diagnostic execution uses deterministic local callables.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import platform
import re
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.evaluation.artifacts import RunArtifacts, read_jsonl, write_json
from llgm.evaluation.matrix import (
    REQUIRED_ARMS,
    matrix_template,
    preflight,
    resolve_backend_scope,
    tokenizer_settings,
)
from llgm.evaluation.prepare import diagnostic_dataset, file_sha256, prepare
from llgm.evaluation.scoring import evidence_coverage, exact_match_diagnostic
from llgm.retrieval import (
    ColBERTTokenizer,
    DiagnosticTokenizer,
    ExactDenseRetriever,
    HybridRetriever,
    SQLiteBM25Retriever,
    validate_encoder_text,
)
from llgm.retrieval.base import passage_from_dict


class _RunLimits:
    """Share one deadline and call counters across every case and arm in a run."""

    def __init__(self, config):
        """Start the experiment clock with zero charged case, generation, and embedding calls."""
        self.config = config
        self.started = time.monotonic()
        self.generations = 0
        self.embeddings = 0
        self.case_runs = 0

    def remaining(self):
        """Return seconds left on the shared deadline, raising when it has expired."""
        left = self.config["timeout_seconds"] - (time.monotonic() - self.started)
        if left <= 0:
            raise BudgetExceeded("Overall experiment deadline exhausted")
        return left

    def take(self, name, maximum):
        """Charge an attempted operation only if its count and the shared deadline permit it."""
        self.remaining()
        if getattr(self, name) >= self.config[maximum]:
            raise BudgetExceeded(f"Overall experiment {maximum} exhausted")
        setattr(self, name, getattr(self, name) + 1)


async def _close_resources(
    resources: list[object], artifacts: RunArtifacts, case_id: str
) -> asyncio.CancelledError | None:
    """Release owned resources and defer cancellation until the caller records its case.

    Both experiment runners await synchronous or asynchronous close methods in
    reverse ownership order. Cleanup errors retain only their type; repeated
    cancellation cannot detach an in-flight release or skip remaining resources.
    """
    cancellation = None
    for resource in reversed(resources):
        close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if close is None:
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                pending = asyncio.ensure_future(result)
                while not pending.done():
                    try:
                        await asyncio.shield(pending)
                    except asyncio.CancelledError as exc:
                        cancellation = cancellation or exc
                pending.result()
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
        except Exception as exc:
            artifacts.append(
                "traces",
                {
                    "case_id": case_id,
                    "kind": "cleanup_failure",
                    "error_type": type(exc).__name__,
                },
            )
    return cancellation


async def _drain_local(task: asyncio.Task):
    """Finish owned native work before propagating even repeated cancellation.

    Cancelling ``to_thread`` does not terminate CUDA/FAISS work. Wait for the
    worker before releasing its resources or starting another case. Local
    deadlines are cooperative and may overrun during a native operation.
    """
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception:
            if cancelled is None:
                raise
            break
    if cancelled is not None:
        if not task.cancelled():
            task.exception()
        raise cancelled
    return task.result()


class _CappedModel:
    """Apply experiment-wide call and time limits to a generation client."""

    def __init__(self, model, limits):
        """Wrap a generation client with shared run accounting."""
        self.model, self.limits = model, limits

    async def complete(self, request):
        """Charge the attempt before dispatch and bound it by the remaining run time."""
        self.limits.take("generations", "max_generation_calls")
        async with asyncio.timeout(self.limits.remaining()):
            return await self.model.complete(request)

    def descriptor(self):
        """Preserve the underlying generation client's identity for run provenance."""
        return self.model.descriptor()


class _CappedEmbedder:
    """Charge each allowed embedding batch as one physical provider request."""

    def __init__(self, model, limits):
        """Wrap the embedding client and expose its model identity to retrievers."""
        self.client, self.limits = model, limits
        self.model = model.model

    async def embed(self, texts):
        """Embed a single physical batch within the shared request and time allowances."""
        if not texts:
            return []
        # ExactDense uses <=64-item batches; underlying client's batch_size=128.
        if len(texts) > 128:
            raise ConfigurationError("Embedding call exceeds single physical request batch")
        self.limits.take("embeddings", "max_embedding_requests")
        async with asyncio.timeout(self.limits.remaining()):
            return await self.client.embed(texts)

    def descriptor(self):
        """Preserve the embedding provider's model and configuration descriptor."""
        return self.client.descriptor()

    @property
    def events(self):
        """Expose provider events so accounting includes usage from failed attempts."""
        return self.client.events


class _QueryValidatedRetriever:
    """Enforce a shared query length limit and retain hits for later evidence scoring."""

    def __init__(self, retriever, tokenizer, query_limit):
        """Wrap a retriever with its query tokenizer, optional limit, and an empty history."""
        self.retriever, self.tokenizer, self.query_limit = retriever, tokenizer, query_limit
        self.history = []

    async def search(self, query, k):
        """Validate before dispatch and record successful hits with a query hash."""
        if self.query_limit is not None:
            validate_encoder_text(query, self.tokenizer, self.query_limit, "query")
        hits = await self.retriever.search(query, k)
        self.history.append(
            {"query_sha256": hashlib.sha256(query.encode()).hexdigest(), "hits": hits}
        )
        return hits

    def descriptor(self):
        """Expose the wrapped backend's descriptor without changing its identity."""
        return self.retriever.descriptor()


def _code_provenance() -> dict:
    """Capture source hashes, runtime packages, and available Git state without network calls."""
    root = Path(__file__).resolve().parents[3]
    result = {
        "source_sha256": {},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
    }
    package = Path(__file__).resolve().parents[1]
    for source in sorted(package.rglob("*.py")):
        result["source_sha256"][str(source.relative_to(package))] = file_sha256(source)
    for name in ("llgm", "openai", "anthropic", "transformers", "colbert-ai", "torch"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    for key, command in (
        ("git_commit", ["rev-parse", "HEAD"]),
        ("git_status", ["status", "--porcelain"]),
    ):
        try:
            result[key] = subprocess.run(
                ["git", "-C", str(root), *command],
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            result[key] = None
    return result


def _redacted(value):
    """Recursively redact known secret keys and credential-bearing base URLs in configuration."""
    from llgm.core.config import redact_url

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"api_key", "password", "token", "secret"}:
                result[key] = "[redacted]"
            elif key.endswith("base_url") and isinstance(item, str):
                result[key] = redact_url(item)
            else:
                result[key] = _redacted(item)
        return result
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return value


def _diagnostic_models():
    """Create deterministic local readers for the controlled launch-color smoke fixture."""
    from llgm.models import CallableModelClient, ModelResponse

    async def sidecar(request):
        """Echo fixture evidence and its passage IDs in the runtime's expected message shape."""
        payload = json.loads(request.messages[-1].content)
        evidence = payload["evidence"]
        text = "\n".join(item["text"] for item in evidence)
        return ModelResponse(
            json.dumps(
                {"text": text, "passage_ids": [p["passage_id"] for p in evidence], "unresolved": []}
            ),
            provider="offline",
            model="deterministic-excerpt-reader",
        )

    async def root(request):
        """Extract the fixture's launch color, abstaining when its fixed pattern is absent."""
        payload = json.loads(request.messages[-1].content)
        matches = re.findall(r"launch color for project \d+ is ([a-z]+)", payload["evidence"])
        return ModelResponse(
            matches[-1] if matches else "insufficient evidence",
            provider="offline",
            model="deterministic-fixture-reader",
        )

    return (
        CallableModelClient(root, provider="offline", model="deterministic-fixture-reader"),
        CallableModelClient(sidecar, provider="offline", model="deterministic-excerpt-reader"),
    )


async def run_matrix(
    prepared: str | Path,
    output: str | Path,
    matrix: dict,
    *,
    arms: list[str] | None = None,
    split: str = "smoke",
    diagnostic: bool = False,
    required_backends: list[str] | None = None,
) -> dict:
    """Execute selected matrix arms with isolated indexes and auditable per-case artifacts.

    Readiness requires all four backends unless ``required_backends`` explicitly
    narrows it; dispatched arms must stay inside that scope. Hosted clients are
    dispatched after preflight and hash checks. Failures remain in metrics,
    observed usage survives cancellation, and the official judge runs separately.
    """
    prepared, output = Path(prepared), Path(output)
    scope = resolve_backend_scope(required_backends)
    chosen_arms = (
        list(arms)
        if arms is not None
        else ["B-S"]
        if diagnostic
        else [arm for arm in REQUIRED_ARMS if arm[0] in scope]
    )
    if (
        not chosen_arms
        or any(not isinstance(arm, str) for arm in chosen_arms)
        or len(chosen_arms) != len(set(chosen_arms))
        or not set(chosen_arms).issubset(REQUIRED_ARMS)
    ):
        raise ConfigurationError(
            "Requested arms must be unique nonempty members of the required matrix"
        )
    if any(arm[0] not in scope for arm in chosen_arms):
        raise ConfigurationError("Requested arms fall outside the explicit required backend scope")
    report = preflight(
        prepared,
        matrix,
        diagnostic=diagnostic,
        **({"required_backends": scope} if required_backends is not None else {}),
    )
    if not report["ready"]:
        raise ConfigurationError(
            "Experiment preflight is blocked; inspect `llgm experiment preflight` output"
        )
    if diagnostic and chosen_arms != ["B-S"]:
        raise ConfigurationError(
            "Offline smoke is B-S plumbing only; it cannot substitute for the 12-arm experiment"
        )
    if split not in {"smoke", "development", "evaluation"}:
        raise ConfigurationError("Choose smoke, development, or explicitly reserved evaluation")
    if output.exists() and any(output.iterdir()):
        raise ConfigurationError("Experiment output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.get("files", {}).items():
        if file_sha256(prepared / relative) != expected:
            raise ConfigurationError(f"Prepared artifact changed: {relative}")
    key = {"smoke": "smoke_ids", "development": "development_ids", "evaluation": "evaluation_ids"}[
        split
    ]
    selected = set(manifest["selection"].get(key, []))
    if not selected:
        raise ConfigurationError(f"Preparation has no selected {split} cases")
    queries = [row for row in read_jsonl(prepared / "queries.jsonl") if row["case_id"] in selected]
    if len(queries) != len(selected):
        raise ConfigurationError("Prepared questions do not cover the declared selection")
    evaluator = {row["case_id"]: row for row in read_jsonl(prepared / "evaluator/gold.jsonl")}
    tokenizer = (
        DiagnosticTokenizer() if diagnostic else ColBERTTokenizer(**tokenizer_settings(matrix))
    )
    counter = tokenizer.count
    limits = _RunLimits(matrix["overall"])
    provenance = _code_provenance()
    summary = {
        "diagnostic": diagnostic,
        "benchmark_result": False,
        "arms": {},
        "prepared_sha256": file_sha256(prepared / "prepared.json"),
        "all_required_arms": list(REQUIRED_ARMS),
        "required_backends": report["required_backends"],
        "selected_arms": chosen_arms,
        "recorded_arms": [],
        "completed_arms": [],
        "full_matrix_ready": report["full_matrix_ready"],
        "full_matrix_complete": False,
    }
    write_json(output / "preflight.json", report)
    from llgm.inference.budget import Budget
    from llgm.inference.iterative import EvidenceSidecar, IterativeRuntime
    from llgm.models import OpenAIEmbeddingClient, create_model

    for arm_id in chosen_arms:
        arm = next(item for item in matrix["arms"] if item["id"] == arm_id)
        artifacts = RunArtifacts(
            output / arm_id,
            {
                "schema_version": 1,
                "run_id": output.name,
                "arm": arm,
                "diagnostic": diagnostic,
                "required_backends": report["required_backends"],
                "selected_arms": chosen_arms,
                "full_matrix_ready": report["full_matrix_ready"],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "matrix": _redacted(matrix),
                "prepared": manifest,
                "case_ids": [q["case_id"] for q in queries],
                "split": split,
                "code": provenance,
                "cache_policy": "cold per case and arm; isolated indexes",
                "local_index_deadlines": "cooperative; in-flight PLAID/native indexing cannot be interrupted; deadline checked before further calls",
                "source_cutoff": "only each question's supplied haystack",
                "retries": "no application retries",
                "scorer": "normalized-exact-match diagnostic; official judge not run",
            },
        )
        observations = []
        for query in queries:
            case_id = query["case_id"]
            started = time.monotonic()
            artifacts.append(
                "cases",
                {
                    **query,
                    "arm": arm_id,
                    "prepared_inputs": {
                        "prepared_sha256": summary["prepared_sha256"],
                        "sources_sha256": manifest["files"][query["sources_path"]],
                    },
                },
            )
            resources, retriever, embedding, result, active_retriever = [], None, None, None, None
            passages = []
            usage, trace, status, answer, references = {}, [], "failed", "", ()
            error_type = None
            cancelled = None
            try:
                limits.take("case_runs", "max_case_arm_runs")
                passages = [
                    passage_from_dict(row) for row in read_jsonl(prepared / query["passages_path"])
                ]
                if diagnostic:
                    root, sidecar_model = _diagnostic_models()
                else:
                    clients = []
                    for role in ("root", "sidecar"):
                        settings = matrix["models"][role]
                        client = create_model(
                            settings["provider"],
                            settings["model"],
                            base_url=settings.get("base_url"),
                            api_key_env=settings.get("api_key_env"),
                        )
                        resources.append(client)
                        clients.append(client)
                    root, sidecar_model = clients
                lexical = None
                if arm["backend"] in {"B", "H"}:
                    lexical = SQLiteBM25Retriever.from_passages(passages)
                    resources.append(lexical)
                dense = None
                if arm["backend"] in {"D", "H"}:
                    client = OpenAIEmbeddingClient(**matrix["dense"], batch_size=128)
                    resources.append(client)
                    embedding = _CappedEmbedder(client, limits)
                    dense = await ExactDenseRetriever.build(passages, embedding)
                if arm["backend"] == "B":
                    retriever = lexical
                elif arm["backend"] == "D":
                    retriever = dense
                elif arm["backend"] == "H":
                    retriever = HybridRetriever(lexical, dense, **matrix["hybrid"])
                else:
                    from llgm.retrieval.colbert import ColBERTConfig, ColBERTRetriever

                    config = ColBERTConfig(
                        **matrix["colbert"],
                        index_root=artifacts.directory / "indexes",
                        index_name=hashlib.sha256(case_id.encode()).hexdigest()[:24],
                    )
                    retriever = await _drain_local(
                        asyncio.create_task(
                            asyncio.to_thread(
                                ColBERTRetriever.build, passages, config=config, tokenizer=tokenizer
                            )
                        )
                    )
                artifacts.append(
                    "traces",
                    {
                        "case_id": case_id,
                        "kind": "index_ready",
                        "descriptor": retriever.descriptor(),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                active_retriever = _QueryValidatedRetriever(
                    retriever, tokenizer, None if diagnostic else matrix["colbert"]["query_maxlen"]
                )
                sidecar = EvidenceSidecar(
                    model=_CappedModel(sidecar_model, limits),
                    retriever=active_retriever,
                    policy=arm["policy"],
                    token_counter=counter,
                    capture_text=True,
                )
                llgm = IterativeRuntime(root=_CappedModel(root, limits), sidecar=sidecar)
                budget = Budget(**matrix["budgets"])
                budget = replace(
                    budget, timeout_seconds=min(budget.timeout_seconds, limits.remaining())
                )
                result = await llgm.answer(
                    query["question"], budget, question_date=query["question_date"]
                )
                usage, trace, status, answer = (
                    result.usage,
                    result.trace,
                    result.status,
                    result.answer,
                )
                references = result.evidence.references
                error_type = next(
                    (
                        event.get("error_type")
                        for event in reversed(trace)
                        if event.get("kind") == "run_stopped"
                    ),
                    None,
                )
            except asyncio.CancelledError as exc:
                cancelled = exc
                error_type, status = "CancelledError", "cancelled"
                usage, trace = getattr(exc, "llgm_usage", {}), getattr(exc, "llgm_trace", [])
                artifacts.append("traces", {"case_id": case_id, "kind": "cancelled"})
            except Exception as exc:
                error_type = type(exc).__name__
                status = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "failed"
                usage, trace = getattr(exc, "llgm_usage", {}), getattr(exc, "llgm_trace", [])
                # Exceptions may contain provider response payloads. Retain the
                # error category here; backend preflight has actionable details.
                artifacts.append(
                    "traces", {"case_id": case_id, "kind": "failure", "error_type": error_type}
                )
            finally:
                for event in trace:
                    artifacts.append("traces", {"case_id": case_id, **event})
                    if event.get("kind") == "model":
                        artifacts.append(
                            "usage", {"case_id": case_id, "category": "generation", **event}
                        )
                if embedding is not None:
                    for event in embedding.events:
                        artifacts.append(
                            "usage", {"case_id": case_id, "category": "embedding", **asdict(event)}
                        )
                if retriever is not None:
                    for item in [
                        retriever,
                        *(
                            [retriever.lexical, retriever.dense]
                            if isinstance(retriever, HybridRetriever)
                            else []
                        ),
                    ]:
                        for event in getattr(item, "events", []):
                            artifacts.append(
                                "traces", {"case_id": case_id, "kind": "retrieval_backend", **event}
                            )
                if active_retriever is not None:
                    for search_number, record in enumerate(active_retriever.history, 1):
                        artifacts.append(
                            "traces",
                            {
                                "case_id": case_id,
                                "kind": "retrieval_results",
                                "search_number": search_number,
                                "query_sha256": record["query_sha256"],
                                "hits": [
                                    {
                                        "passage_id": hit.passage.passage_id,
                                        "rank": hit.rank,
                                        "score": hit.score,
                                        "references": hit.passage.refs,
                                    }
                                    for hit in record["hits"]
                                ],
                            },
                        )
                cleanup_cancelled = await _close_resources(resources, artifacts, case_id)
                if cleanup_cancelled is not None:
                    cancelled = cancelled or cleanup_cancelled
                    status, error_type = "cancelled", "CancelledError"
                    artifacts.append(
                        "traces", {"case_id": case_id, "kind": "cancelled", "phase": "cleanup"}
                    )
            prediction = {
                "case_id": case_id,
                "question_id": case_id,
                "hypothesis": answer,
                "status": status,
                "error_type": error_type,
                "references": references,
                "usage": usage,
                "elapsed_seconds": time.monotonic() - started,
            }
            artifacts.append("predictions", prediction)
            gold = evaluator[case_id]
            history = active_retriever.history if active_retriever else []
            first = (
                [ref for hit in history[0]["hits"] for ref in hit.passage.refs] if history else []
            )
            retrieved = [
                ref for record in history for hit in record["hits"] for ref in hit.passage.refs
            ]
            admitted_ids = {
                pid
                for event in trace
                if event.get("kind") == "search"
                for pid in event.get("admitted_ids", [])
            }
            admitted = [
                ref
                for passage in passages
                if passage.passage_id in admitted_ids
                for ref in passage.refs
            ]
            judgment = {
                "case_id": case_id,
                "scorer": "normalized_exact_match_diagnostic",
                "official": False,
                "exact_match": status == "completed"
                and exact_match_diagnostic(answer, gold["answer"]),
                "source_recall": evidence_coverage(admitted, gold)["source_recall"],
                "ability": gold["ability"],
                "first_retrieval": evidence_coverage(first, gold),
                "final_retrieval": evidence_coverage(retrieved, gold),
                "admitted_evidence": evidence_coverage(admitted, gold),
                "final_bundle": evidence_coverage(references, gold),
            }
            artifacts.append("judgments", judgment)
            observations.append({**prediction, **judgment})
            if cancelled is not None:
                artifacts.finish(
                    {
                        "cancelled": True,
                        "recorded_case_count": len(observations),
                        "official_score": None,
                    },
                    "Run cancelled; partial predictions and observed usage are preserved.",
                )
                summary["cancelled"] = True
                summary["physical_generation_calls"] = limits.generations
                summary["physical_embedding_requests"] = limits.embeddings
                write_json(output / "summary.json", summary)
                raise cancelled
        count = len(observations)
        recalls = [row["source_recall"] for row in observations if row["source_recall"] is not None]
        metrics = {
            "case_count": count,
            "completed": sum(row["status"] == "completed" for row in observations),
            "failed_or_partial": sum(row["status"] != "completed" for row in observations),
            "diagnostic_exact_match": sum(row["exact_match"] for row in observations) / count
            if count
            else None,
            "source_recall": sum(recalls) / len(recalls) if recalls else None,
            "source_recall_denominator": len(recalls),
            "official_score": None,
            "currency_cost": None,
            "currency_note": "Unpriced; raw physical generation/embedding usage is retained",
            "scope": "offline plumbing" if diagnostic else f"{split}, official judgment pending",
        }
        for stage in ("first_retrieval", "final_retrieval", "admitted_evidence", "final_bundle"):
            metrics[stage] = {}
            for key in (
                "source_recall",
                "answer_turn_recall",
                "all_required_sources_covered",
                "all_answer_turns_hit",
            ):
                values = [row[stage][key] for row in observations if row[stage][key] is not None]
                metrics[stage][key] = sum(values) / len(values) if values else None
                metrics[stage][key + "_case_denominator"] = len(values)
        artifacts.finish(
            metrics,
            "No benchmark superiority claim. "
            + (
                "Deterministic offline smoke only; complete all 12 hosted arms before substantive comparison."
                if diagnostic
                else "Official LongMemEval judgment and a reviewed research decision remain pending."
            ),
        )
        summary["arms"][arm_id] = metrics
        summary["recorded_arms"].append(arm_id)
        if metrics["completed"] == count:
            summary["completed_arms"].append(arm_id)
        summary["full_matrix_complete"] = set(summary["completed_arms"]) == set(REQUIRED_ARMS)
        write_json(output / "summary.json", summary)
    summary["physical_generation_calls"] = limits.generations
    summary["physical_embedding_requests"] = limits.embeddings
    summary["case_arm_runs"] = limits.case_runs
    summary["elapsed_seconds"] = time.monotonic() - limits.started
    write_json(output / "summary.json", summary)
    return summary


async def offline_smoke(output: str | Path) -> dict:
    """Prepare and run the deterministic B-S fixture in a new or empty directory."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ConfigurationError("Offline-smoke output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "fixture.json", diagnostic_dataset())
    prepare(
        output / "fixture.json",
        output / "prepared",
        tokenizer=DiagnosticTokenizer(),
        target=5,
        smoke_count=5,
        revision="controlled-v1",
        dataset_kind="controlled-offline-fixture",
    )
    matrix = matrix_template()
    write_json(output / "matrix.json", matrix)
    return await run_matrix(output / "prepared", output / "run", matrix, diagnostic=True)
