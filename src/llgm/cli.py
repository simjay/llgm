"""Thin CLI over LLGM's public preparation and runtime APIs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from llgm.core.errors import LLGMError


def _parser():
    """Define CLI commands and explicit gates for downloads and provider execution."""
    parser = argparse.ArgumentParser(
        prog="llgm", description="Persistent evidence and bounded LLM inference"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser(
        "ask", help="Answer with configured models and local Docker node delegates"
    )
    ask.add_argument("question")
    ask.add_argument("--config", type=Path)
    ask.add_argument(
        "--env-file", type=Path, help="Load a local env file without replacing exported variables"
    )
    ask.add_argument("--workspace", type=Path)
    ask.add_argument(
        "--node-id", help="Use one explicit starting node instead of initial retrieval"
    )
    ask.add_argument("--scope", default="{}", help="JSON object selecting evidence applicability")
    ask.add_argument("--as-of-ms", type=int, help="Journal validity instant in Unix milliseconds")
    ask.add_argument("--query-date", help="Original date text for model context")
    migrate = commands.add_parser(
        "migrate", help="Copy a schema-2 workspace with explicit pointer classification"
    )
    migrate.add_argument("source", type=Path)
    migrate.add_argument("destination", type=Path)
    migrate.add_argument(
        "--journal-roles",
        type=Path,
        required=True,
        help="JSON mapping of journal entry IDs to edge, amendment, or unresolved",
    )
    smoke = commands.add_parser(
        "offline-smoke", help="Run the deterministic B-S retrieval fixture without network access"
    )
    smoke.add_argument("--output", type=Path, required=True)
    experiment = commands.add_parser(
        "experiment", help="Prepare and run the required E03 retrieval matrix"
    )
    actions = experiment.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser(
        "prepare", help="Separate corpus/gold and make a deterministic history-group split"
    )
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--dataset-revision")
    prepare.add_argument("--tokenizer-directory", type=Path)
    prepare.add_argument("--tokenizer-revision")
    prepare.add_argument("--diagnostic-tokenizer", choices=("word", "character"), default="word")
    prepare.add_argument("--target", type=int, default=50)
    prepare.add_argument("--smoke-count", type=int, default=5)
    prepare.add_argument("--seed", type=int, default=1729)
    prepare.add_argument(
        "--selection", choices=("development", "evaluation"), default="development"
    )
    prepare.add_argument(
        "--case-limit", type=int, help="Materialize at most this many reserved evaluation cases"
    )
    prepare.add_argument(
        "--dataset-kind",
        choices=("longmemeval-s-cleaned", "controlled-development"),
        default="longmemeval-s-cleaned",
    )
    controlled = actions.add_parser(
        "controlled-fixtures",
        help="Write separate synthetic development histories; not benchmark results",
    )
    controlled.add_argument("--output", type=Path, required=True)
    controlled.add_argument("--count", type=int, default=60)
    download = actions.add_parser(
        "download", help="Explicitly download the public cleaned-S dataset at a pinned commit"
    )
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--dataset-revision", required=True)
    download.add_argument("--sha256")
    for name in ("preflight", "run"):
        action = actions.add_parser(name)
        action.add_argument("--prepared", type=Path, required=True)
        action.add_argument("--matrix", type=Path, required=True)
        action.add_argument("--output", type=Path, required=name == "run")
        action.add_argument(
            "--required-backends",
            choices=("B", "D", "H", "C"),
            nargs="+",
            help="Explicit readiness scope; omitted requires the complete matrix",
        )
        action.add_argument(
            "--diagnostic",
            action="store_true",
            help="Only deterministic offline B-S; not a benchmark run",
        )
        if name == "preflight":
            action.add_argument(
                "--retrieval-only",
                action="store_true",
                help="Check all four backends without generation-provider requirements",
            )
        if name == "run":
            action.add_argument(
                "--arms",
                nargs="+",
                help="Subset to dispatch within the required readiness scope",
            )
            action.add_argument(
                "--split", choices=("smoke", "development", "evaluation"), default="smoke"
            )
            action.add_argument(
                "--execute",
                action="store_true",
                help="Explicitly dispatch configured providers; may incur charges",
            )
    retrieve = actions.add_parser(
        "retrieve", help="Original-question top40 diagnostic across B/D/H/C; no generation calls"
    )
    retrieve.add_argument("--prepared", type=Path, required=True)
    retrieve.add_argument("--matrix", type=Path, required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--backends", choices=("B", "D", "H", "C"), nargs="+")
    retrieve.add_argument(
        "--required-backends",
        choices=("B", "D", "H", "C"),
        nargs="+",
        help="Explicit readiness scope; omitted requires all four backends",
    )
    retrieve.add_argument(
        "--split", choices=("smoke", "development", "evaluation"), default="smoke"
    )
    retrieve.add_argument(
        "--execute", action="store_true", help="Explicitly dispatch embedding and local index work"
    )
    export = actions.add_parser(
        "export-official", help="Export one arm for a separately invoked official scorer"
    )
    export.add_argument("--predictions", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    template = actions.add_parser(
        "matrix-template", help="Write all 12 required arms with unresolved model/checkpoint pins"
    )
    template.add_argument("--output", type=Path, required=True)
    return parser


async def _ask(args) -> tuple[dict, int]:
    """Keep configured resources open through inference and serialize the cited answer."""
    from llgm import LLGM, Settings, load_env_file
    from llgm.core.types import reference_to_dict

    scope = json.loads(args.scope)
    if not isinstance(scope, dict):
        raise LLGMError("--scope must be a JSON object")
    if args.env_file is not None:
        load_env_file(args.env_file)
    settings = Settings.load(
        config_file=args.config,
        overrides={"workspace_path": str(args.workspace)} if args.workspace else {},
    )
    async with LLGM.from_settings(settings) as application:
        response = await application.answer(
            args.question,
            node_id=args.node_id,
            scope=scope,
            as_of_ms=args.as_of_ms,
            query_date=args.query_date,
        )
    result = {
        "answer": response.answer,
        "status": response.status,
        "references": [reference_to_dict(ref) for ref in response.references],
        "unresolved": response.evidence.unresolved,
        "usage": response.usage,
    }
    return result, 0 if response.status == "completed" else 2


def _experiment(args) -> tuple[dict, int]:
    """Dispatch experiment commands with their explicit download and execution gates."""
    from llgm.evaluation.artifacts import write_json
    from llgm.evaluation.matrix import load_matrix, matrix_template, preflight

    if args.action == "prepare":
        from llgm.evaluation.prepare import prepare
        from llgm.retrieval import ColBERTTokenizer, DiagnosticTokenizer

        tokenizer = (
            ColBERTTokenizer(args.tokenizer_directory, args.tokenizer_revision)
            if args.tokenizer_directory
            else DiagnosticTokenizer(args.diagnostic_tokenizer)
        )
        manifest = prepare(
            args.dataset,
            args.output,
            tokenizer=tokenizer,
            revision=args.dataset_revision,
            target=args.target,
            smoke_count=args.smoke_count,
            seed=args.seed,
            selection_mode=args.selection,
            case_limit=args.case_limit,
            dataset_kind=args.dataset_kind,
        )
        result = {
            "output": str(args.output.resolve()),
            "selection": manifest["selection"],
            "tokenizer": manifest["tokenizer"],
            "dataset": manifest["dataset"],
        }
        return result, 0 if manifest["selection"]["status"] == "ready" else 2
    elif args.action == "download":
        from llgm.evaluation.prepare import download_cleaned_s, file_sha256

        path = download_cleaned_s(
            args.output, revision=args.dataset_revision, expected_sha256=args.sha256
        )
        result = {"dataset": str(path.resolve()), "sha256": file_sha256(path)}
    elif args.action == "controlled-fixtures":
        from llgm.evaluation.prepare import controlled_dataset

        if args.output.exists():
            raise LLGMError("Controlled fixture destination already exists")
        write_json(args.output, controlled_dataset(args.count))
        result = {
            "dataset": str(args.output.resolve()),
            "cases": args.count,
            "benchmark": False,
        }
    elif args.action == "matrix-template":
        if args.output.exists():
            raise LLGMError("Matrix destination already exists")
        write_json(args.output, matrix_template())
        result = {"matrix": str(args.output.resolve()), "arms": 12}
    elif args.action == "preflight":
        result = preflight(
            args.prepared,
            load_matrix(args.matrix),
            diagnostic=args.diagnostic,
            retrieval_only=args.retrieval_only,
            required_backends=args.required_backends,
        )
        if args.output:
            write_json(args.output, result)
        return result, 0 if result["ready"] else 2
    elif args.action == "run":
        if not args.diagnostic and not args.execute:
            raise LLGMError(
                "Run is a live dispatch boundary. Use --execute after preflight to invoke configured providers"
            )
        from llgm.evaluation.runner import run_matrix

        result = asyncio.run(
            run_matrix(
                args.prepared,
                args.output,
                load_matrix(args.matrix),
                arms=args.arms,
                split=args.split,
                diagnostic=args.diagnostic,
                required_backends=args.required_backends,
            )
        )
    elif args.action == "retrieve":
        if not args.execute:
            raise LLGMError(
                "Retrieval diagnostic can call paid embeddings; use --execute after preflight"
            )
        from llgm.evaluation.retrieval_diagnostic import run_retrieval_diagnostic

        result = asyncio.run(
            run_retrieval_diagnostic(
                args.prepared,
                args.output,
                load_matrix(args.matrix),
                backends=args.backends,
                split=args.split,
                required_backends=args.required_backends,
            )
        )
    elif args.action == "export-official":
        from llgm.evaluation.scoring import export_official_predictions

        export_official_predictions(args.predictions, args.output)
        result = {"output": str(args.output.resolve()), "official_judge_invoked": False}
    else:
        raise LLGMError("Unknown command")
    return result, 0


def main(argv=None) -> int:
    """Print one JSON result and translate operational failures to a process status."""
    args = _parser().parse_args(argv)
    try:
        status = 0
        if args.command == "ask":
            result, status = asyncio.run(_ask(args))
        elif args.command == "migrate":
            from llgm.memory.migration import copy_schema2_workspace

            result = asyncio.run(
                copy_schema2_workspace(
                    args.source,
                    args.destination,
                    journal_roles=json.loads(args.journal_roles.read_text(encoding="utf-8")),
                )
            )
        elif args.command == "offline-smoke":
            from llgm.evaluation.runner import offline_smoke

            result = asyncio.run(offline_smoke(args.output))
        else:
            result, status = _experiment(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return status
    except (LLGMError, OSError, ValueError) as exc:
        print(f"llgm: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
