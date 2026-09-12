"""Hosted persistent-memory example that makes configured model API calls.

Requires Docker and its configured local Python image.
Configure LLGM_MAIN_MODEL, LLGM_READER_MODEL, LLGM_GRAPH_MODEL,
provider credentials, and optionally
LLGM_WORKSPACE_PATH before running, or supply an explicit --env-file path.
LLGM opens and closes its workspace and provider connections.
"""

import argparse
import asyncio
import json

from llgm import LLGM, load_env_file
from llgm.core.types import reference_to_dict


async def main():
    """Continue one topic and let LLGM retain the conversation automatically."""
    async with LLGM.from_settings() as memory:
        await memory.answer(
            "Orion production runs release r17 in eu-west-1. Staging runs in us-east-1.",
            conversation_id="orion",
        )
        answer = await memory.answer(
            "Which region runs production?",
            conversation_id="orion",
        )
        print(answer.answer)
        print(
            json.dumps(
                {
                    "conversation_id": answer.conversation_id,
                    "node_id": answer.node_id,
                    "status": answer.status,
                    "references": [reference_to_dict(ref) for ref in answer.references],
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
