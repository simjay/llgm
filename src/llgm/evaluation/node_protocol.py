"""Declared transport interventions for focused node-pipeline mechanism checks.

These protocols are experimental controls, not autonomous-navigation evidence.
They preserve model responses and never execute generated Python on the host.
"""

from __future__ import annotations

from llgm.core.errors import BudgetExceeded, ConfigurationError
from llgm.inference.repl import DockerREPL

PROTOCOLS = frozenset(
    {
        "guided_recursion",
        "child_return_exhaustion",
        "transient_read_error",
        "transient_operation_error",
        "relation_order",
    }
)

RECURSION_INSTRUCTIONS = (
    "This is a prescribed recursive-mechanism diagnostic. Inspect your own node, "
    "then use edge relation descriptions to choose a relevant linked node. "
    "Investigate another node only through query_node, not through direct remote "
    "reads or global search. Apply the same rule in children. Return the facts "
    "and citation IDs actually obtained; do not infer missing evidence. This "
    "instruction prescribes navigation mechanics, not an answer."
)


def protocol_factory(name, *, record_event, repl_factory=DockerREPL):
    """Wrap explicit REPL transport with one frozen, trial-scoped intervention.

    The factory adds protocol context before NodeRuntime builds its model
    messages, so ordinary context admission includes the declared instructions.
    Faults fire once across a trial's interpreters. The exhaustion intervention
    waits for the real execute result after a completed child was delivered.
    """
    if name not in PROTOCOLS:
        raise ConfigurationError(f"Unknown node diagnostic protocol: {name!r}")
    injected = False

    def create(context, *, config, node_callback):
        """Attach counted diagnostic instructions and retain the underlying interpreter owner."""
        nonlocal injected
        context["protocol"] = {"name": name, "autonomous_navigation": False}
        if name in {"guided_recursion", "child_return_exhaustion"}:
            context["protocol"]["instructions"] = RECURSION_INSTRUCTIONS
        record_event({"kind": "protocol_context", "protocol": name, "node_id": context["node_id"]})
        delivered_child = None

        async def callback(operation):
            """Apply the declared single fault or edge ordering and observe admitted child evidence."""
            nonlocal injected, delivered_child
            forwarded = operation
            if (
                not injected
                and name in {"transient_read_error", "transient_operation_error"}
                and operation.get("op") == "read"
            ):
                injected = True
                if name == "transient_read_error":
                    reference = operation["reference"]
                    forwarded = {
                        "op": "read",
                        "reference": {
                            "type": "source_span",
                            "node_id": reference["node_id"],
                            "turn_id": reference.get("turn_id", "diagnostic-missing-turn"),
                            "start": 0,
                            "end": 10**12,
                        },
                    }
                else:
                    forwarded = {"op": "diagnostic_unknown_operation"}
                record_event(
                    {
                        "kind": "protocol_fault",
                        "protocol": name,
                        "node_id": context["node_id"],
                        "requested_operation": operation,
                        "forwarded_operation": forwarded,
                    }
                )
            payload = await node_callback(forwarded)
            if name == "relation_order" and operation.get("op") == "edges" and payload.get("edges"):
                descriptions = sorted(
                    payload["edges"], key=lambda edge: (edge["relation"], edge["edge_id"])
                )
                references = {
                    edge["reference"]["node_id"]: edge["reference"] for edge in descriptions
                }
                payload = {
                    **payload,
                    "edges": descriptions,
                    "references": list(references.values()),
                }
                record_event(
                    {
                        "kind": "protocol_edge_order",
                        "protocol": name,
                        "node_id": context["node_id"],
                        "edges": [
                            {
                                "edge_id": edge["edge_id"],
                                "relation": edge["relation"],
                                "reference": edge["reference"],
                            }
                            for edge in descriptions
                        ],
                    }
                )
            if (
                operation.get("op") == "query_node"
                and payload.get("status") == "completed"
                and payload.get("evidence")
            ):
                delivered_child = {
                    "node_id": payload["node_id"],
                    "invocation_id": payload["invocation_id"],
                    "citations": [record["id"] for record in payload["evidence"]],
                }
            return payload

        inner = repl_factory(context, config=config, node_callback=callback)

        class ProtocolREPL:
            """Delegate interpreter ownership while observing callback delivery."""

            @property
            def container_name(self):
                """Expose the actual interpreter identity for diagnostics and cleanup."""
                return inner.container_name

            async def start(self):
                """Start the owned interpreter without creating a host execution fallback."""
                return await inner.start()

            async def execute(self, code):
                """Inject exhaustion only after Python returned with a completed child finding."""
                nonlocal injected
                output = await inner.execute(code)
                if name == "child_return_exhaustion" and not injected and delivered_child:
                    injected = True
                    record_event(
                        {
                            "kind": "protocol_fault",
                            "protocol": name,
                            "node_id": context["node_id"],
                            "completed_child": delivered_child,
                            "after_execute_returned": True,
                        }
                    )
                    raise BudgetExceeded("Diagnostic exhaustion after completed child delivery")
                return output

            async def aclose(self):
                """Close the underlying interpreter even after an injected failure."""
                return await inner.aclose()

        return ProtocolREPL()

    return create
