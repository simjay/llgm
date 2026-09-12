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
        "migrate", help="Copy a schema-3 workspace, or schema 2 with --journal-roles"
    )
    migrate.add_argument("source", type=Path)
    migrate.add_argument("destination", type=Path)
    migrate.add_argument(
        "--journal-roles",
        type=Path,
        help="JSON mapping of journal entry IDs to edge, amendment, or unresolved",
    )
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


def main(argv=None) -> int:
    """Print one JSON result and translate operational failures to a process status."""
    args = _parser().parse_args(argv)
    try:
        status = 0
        if args.command == "ask":
            result, status = asyncio.run(_ask(args))
        elif args.command == "migrate":
            from llgm.memory.migration import copy_schema2_workspace, copy_schema3_workspace

            result = asyncio.run(
                copy_schema2_workspace(
                    args.source,
                    args.destination,
                    journal_roles=json.loads(args.journal_roles.read_text(encoding="utf-8")),
                )
                if args.journal_roles
                else copy_schema3_workspace(args.source, args.destination)
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return status
    except (LLGMError, OSError, ValueError) as exc:
        print(f"llgm: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
