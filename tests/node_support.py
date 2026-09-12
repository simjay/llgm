"""Finite callback replay and scripted models for deterministic node tests."""

import json

from llgm.inference.repl import REPLResult
from llgm.models import CallableModelClient, ModelResponse

READ = 'print(read(context["references"][0]))'
BAD_READ = (
    'print(read({key: value for key, value in context["references"][0].items() if key != "type"}))'
)
EDGES = "print(edges())"
SEARCH = 'print(search("jade", k=1))'
CHILD = 'print(query_node("b", "Find the region"))'
CYCLE = 'print(query_node(context["node_id"], context["question"]))'


def finish(answer="Findings", citations=(), unresolved=()):
    """Encode explicit cited findings for the real runtime parser."""
    return json.dumps(
        {
            "op": "finish",
            "answer": answer,
            "citations": list(citations),
            "unresolved": list(unresolved),
        }
    )


class ReplayREPL:
    """Finite command replay adapter that never executes generated Python."""

    def __init__(self, context, *, config, node_callback, owner):
        """Capture isolated context and the real runtime's host callback."""
        self.context, self.callback, self.owner = context, node_callback, owner
        self.closed = False
        self.container_name = "replay-" + context["node_id"]

    async def start(self):
        """Record startup without launching a process or reading source data."""
        self.owner.started.append(self.context["node_id"])

    async def execute(self, code):
        """Replay one predeclared transport interaction, rejecting every other command."""
        self.owner.codes.append(code)
        if code == READ:
            operation = {"op": "read", "reference": self.context["references"][0]}
        elif code == BAD_READ:
            operation = {
                "op": "read",
                "reference": {
                    key: value
                    for key, value in self.context["references"][0].items()
                    if key != "type"
                },
            }
        elif code == SEARCH:
            operation = {"op": "search", "query": "jade", "k": 1}
        elif code == EDGES:
            operation = {"op": "edges", "node_id": None}
        elif code == CHILD:
            operation = {"op": "query_node", "node_id": "b", "question": "Find the region"}
        elif code == CYCLE:
            operation = {
                "op": "query_node",
                "node_id": self.context["node_id"],
                "question": self.context["question"],
            }
        else:
            raise AssertionError("Replay does not execute arbitrary Python")
        payload = await self.callback(operation)
        return REPLResult(json.dumps(payload), None, False, 1, ())

    async def aclose(self):
        """Record resource cleanup for successful, failed, and cancelled branches."""
        self.closed = True


class ReplayFactory:
    """Create separate replay sessions while retaining lifecycle evidence."""

    def __init__(self):
        """Start with no interpreters, source reads, or generated commands."""
        self.sessions, self.started, self.codes = [], [], []

    def __call__(self, context, *, config, node_callback):
        """Provide an explicit test transport with isolated context per invocation."""
        repl = ReplayREPL(context, config=config, node_callback=node_callback, owner=self)
        self.sessions.append(repl)
        return repl


class Models:
    """Deterministic decisions based on actual callback results, with request retention."""

    def __init__(self, plans=None):
        """Specify local operation plans without prescribing runtime evidence IDs."""
        self.plans = plans or {}
        self.child_requests, self.main_requests = [], []
        self.reader = CallableModelClient(self.child)
        self.main = CallableModelClient(self.synthesize)

    async def child(self, request):
        """Follow the declared local plan, then cite IDs received from its last observation."""
        self.child_requests.append(request)
        context = json.loads(request.messages[1].content)
        node_id = context["node_id"]
        plan = self.plans.get(node_id, [READ])
        step = (len(request.messages) - 2) // 2
        if step < len(plan):
            return ModelResponse(json.dumps({"op": "python", "code": plan[step]}))
        payload = json.loads(json.loads(request.messages[-1].content)["stdout"])
        citations = [record["id"] for record in payload.get("evidence", [])]
        return ModelResponse(
            finish(node_id + " findings", citations, payload.get("unresolved", []))
        )

    async def synthesize(self, request):
        """Combine attributed branch findings and preserve their supplied canonical citations."""
        self.main_requests.append(request)
        payload = json.loads(request.messages[1].content)
        branches = payload["branches"]
        citations = [record["id"] for record in payload["evidence"]]
        unresolved = [gap for branch in branches for gap in branch["unresolved"]]
        return ModelResponse(finish("Combined findings", citations, unresolved))
