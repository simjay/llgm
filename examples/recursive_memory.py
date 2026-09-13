"""Hosted persistent-memory example that makes configured model API calls.

Requires the rlm extra with DSPy and Deno.
Configure LLGM_MAIN_MODEL, LLGM_READER_MODEL, LLGM_GRAPH_MODEL,
provider credentials and a search backend before running, or supply --env-file.
Use LLGM_RETRIEVER_BACKEND=sqlite_fts5 for local BM25. Hybrid search needs the
configured Modal service and can upload stored sources. LLGM_WORKSPACE_PATH
optionally selects the saved memory directory. Each rerun adds chat turns.
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
