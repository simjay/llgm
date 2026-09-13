"""Run the full LLGM node pipeline with DSPy and real Deno/Pyodide and scripted models, without API calls.

Install llgm[rlm] first. The first sandbox run can download runtime assets.
This script creates and removes only a temporary workspace.
"""

import asyncio
import json
from tempfile import TemporaryDirectory

from llgm import LLGM, Budget, Conversation, MaintenancePolicy, Provenance, SourceSpan, Workspace
from llgm.core.time import parse_instant_ms
from llgm.models import CallableModelClient, ModelResponse


async def node_response(request):
    """Script local Python inspection and child return while deriving findings from observations."""
    code = """import json
page = source_info()
records = []
while True:
    for turn in page["turns"]:
        records.extend(read(turn["reference"])["evidence"])
    if page["next_offset"] is None:
        break
    page = source_info(offset=page["next_offset"])
findings = [record["text"] for record in records]
citations = [record["id"] for record in records if record["text"]]
unresolved = [gap for record in records for gap in record["metadata"].get("unresolved", [])]
for neighbor in edges()["references"]:
    child = query_node(neighbor["node_id"], context["question"])
    findings.append(" " + child["answer"])
    citations.extend(record["id"] for record in child["evidence"])
    unresolved.extend(child["unresolved"])
SUBMIT(answer="".join(findings), citations=citations, unresolved=unresolved)"""
    return ModelResponse(
        json.dumps({"reasoning": "Inspect local evidence and useful children", "code": code})
    )


async def main_response(request):
    """Combine actual branch findings and their returned citation IDs in one final call."""
    payload = json.loads(request.messages[-1].content)
    operation = {
        "answer": " ".join(branch["findings"] for branch in payload["branches"]),
        "citations": [record["id"] for record in payload["evidence"]],
        "unresolved": [gap for branch in payload["branches"] for gap in branch["unresolved"]],
    }
    return ModelResponse(json.dumps(operation))


async def main():
    """Retrieve two seeds, follow a primary edge, apply a scoped patch, and synthesize."""
    with TemporaryDirectory(prefix="llgm-example-") as directory:
        async with Workspace.open(directory) as workspace:
            memory = LLGM(
                workspace,
                CallableModelClient(main_response),
                CallableModelClient(node_response),
                graph_model=None,
                maintenance_policy=MaintenancePolicy(mode="disabled"),
                max_seed_nodes=2,
                max_concurrency=2,
                inference_budget=Budget(
                    max_model_calls=12,
                    max_reader_calls=10,
                    max_searches=4,
                    max_evidence_tokens=131072,
                    max_bundle_tokens=32768,
                    max_context_tokens=131072,
                    max_output_tokens=2048,
                    timeout_seconds=120,
                ),
            )
            sources = {}
            texts = {
                "database": "The production database uses PostgreSQL. Staging uses SQLite.",
                "backups": "Production backups are retained for seven days.",
                "registry": "The deployment registry says eu-west-1.",
                "update": "MySQL",
            }
            for label, text in texts.items():
                outcome = await memory.ingest(
                    Conversation.from_turns(
                        [{"role": "user", "turn_id": "note", "text": text}],
                        node_id=label,
                    )
                )
                sources[label] = outcome.source.node_id
            await workspace.publish_edge(
                sources["database"],
                sources["registry"],
                provenance=Provenance("user", "offline-example"),
            )
            start = texts["database"].index("PostgreSQL")
            old = SourceSpan(sources["database"], "note", start, start + len("PostgreSQL"))
            replacement = SourceSpan(sources["update"], "note", 0, len(texts["update"]))
            await workspace.append_journal(
                sources["database"],
                subject=old,
                record_kind="overwrite",
                relation="replace",
                value=replacement,
                provenance=Provenance("user", "offline-example", (replacement,)),
                applicability={"scope": {"env": "production"}},
            )
            result = await memory.answer(
                "Production database, backups, and region",
                remember=False,
                scope={"env": "production"},
                as_of_ms=parse_instant_ms("2026-09-11T00:00:00Z"),
            )
            assert result.status == "completed", result.evidence.unresolved
            assert all(
                value in result.answer for value in ("MySQL", "SQLite", "seven days", "eu-west-1")
            ), result.answer
            assert "PostgreSQL" not in result.answer
            assert (await workspace.resolve(old)).text == "PostgreSQL"
            selected = next(event for event in result.trace if event["kind"] == "seed_selection")
            assert set(selected["selected"]) == {sources["database"], sources["backups"]}
            assert any(
                event["kind"] == "enter"
                and event["depth"] == 1
                and event["target_node_id"] == sources["registry"]
                for event in result.trace
            )
            assert sum(event["kind"] == "main_return" for event in result.trace) == 1
            print(result.answer)
            print("Two retrieved seeds, one recursive descendant, one final main call.")
            print("Replacement citations are canonical. The original source remains readable.")


if __name__ == "__main__":
    asyncio.run(main())
