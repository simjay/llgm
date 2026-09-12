"""Hosted persistent-memory example that makes configured model API calls.

Requires Docker and its configured local Python image.
Configure LLGM_ROOT_MODEL, LLGM_SIDECAR_MODEL, provider credentials, and optionally
LLGM_WORKSPACE_PATH before running, or supply an explicit --env-file path.
LLGM opens and closes its workspace and provider connections.
"""

import argparse
import asyncio
import json

from llgm import LLGM, load_env_file
from llgm.core.types import reference_to_dict


async def main():
    """Store two related notes and ask which region runs the production release."""
    async with LLGM.from_settings() as memory:
        for source_label, text in (
            (
                "orion-release",
                "Orion production runs release r17. Its deployment region is recorded in the Orion registry.",
            ),
            (
                "orion-registry",
                "The Orion registry places production release r17 in eu-west-1. Staging runs in us-east-1.",
            ),
        ):
            outcome = await memory.ingest(
                text,
                idempotency_key="example:notes:" + source_label,
            )
            print(
                json.dumps(
                    {
                        "source_id": outcome.source.node_id,
                        "maintenance_status": outcome.maintenance.status,
                        "accepted_edges": len(outcome.maintenance.accepted),
                        "maintenance_usage": dict(outcome.maintenance.usage),
                    },
                    indent=2,
                )
            )
            if outcome.maintenance.status not in {"completed", "disabled"}:
                raise RuntimeError(
                    "Source is persisted, but maintenance needs inspection before continuing"
                )
        answer = await memory.answer(
            "Which region runs Orion production?", scope={"env": "production"}
        )
        print(answer.answer)
        print(
            json.dumps(
                {
                    "status": answer.status,
                    "references": [reference_to_dict(reference) for reference in answer.references],
                    "usage": answer.usage,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Load local values without replacing exported variables")
    args = parser.parse_args()
    if args.env_file is not None:
        load_env_file(args.env_file)
    asyncio.run(main())
