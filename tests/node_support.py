"""Finite callback replay and scripted models for deterministic node tests."""

import json

from dspy.primitives.code_interpreter import FinalOutput

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
    """Encode cited findings for the main model's final synthesis schema."""
    return json.dumps(
        {
            "answer": answer,
            "citations": list(citations),
            "unresolved": list(unresolved),
        }
    )


def action(code):
    """Encode the output fields of DSPy's generated action signature."""
    return json.dumps({"reasoning": "Inspect the supplied evidence", "code": code})


def submit(answer="Findings", citations=(), unresolved=()):
    """Submit typed RLM output through Python without evaluating fixture code on the host."""
    payload = json.dumps(
        {"answer": answer, "citations": list(citations), "unresolved": list(unresolved)}
    )
    return action("SUBMIT(**json.loads(" + repr(payload) + "))")


def node_context(request):
    """Read the complete node metadata embedded in DSPy's signature instructions."""
    if "LLGM node context:\n" not in request.messages[0].content:
        return json.loads(request.messages[1].content)
    return json.JSONDecoder().raw_decode(
        request.messages[0].content.split("LLGM node context:\n", 1)[1].lstrip()
    )[0]


def history(request):
    """Decode DSPy's actual REPL history field from its formatted input message."""
    text = request.messages[1].content.split("[[ ## repl_history ## ]]\n", 1)[1]
    return text.split("\n\n[[ ##", 1)[0]


def step(request):
    """Read the action iteration without assuming one conversation turn per action."""
    marker = "[[ ## iteration ## ]]\n"
    if marker not in request.messages[1].content:
        return 1000
    return int(request.messages[1].content.split(marker)[1].split("/")[0]) - 1


def observation(request):
    """Find the latest JSON evidence payload in DSPy's rendered execution history."""
    text = history(request)
    decoder = json.JSONDecoder()
    values = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except ValueError:
            continue
        if isinstance(value, dict) and any(
            key in value for key in ("evidence", "error", "hits", "neighbors", "edges")
        ):
            values.append(value)
    return values[-1] if values else {}


class ReplayREPL:
    """Implement DSPy's CodeInterpreter using a finite command table, never host exec."""

    def __init__(self, owner):
        """Retain separate tool bindings and lifecycle state per invocation."""
        self.owner, self.tools = owner, {}
        self.output_fields = None
        self.closed = False
        self.context = {}

    def start(self):
        """Record interpreter admission without opening a process."""
        self.owner.started.append(self)

    def execute(self, code, variables=None):
        """Replay only explicitly declared commands and typed SUBMIT fixtures."""
        from ast import literal_eval

        self.context = variables["context"]
        self.owner.codes.append(code)
        if code.startswith("SUBMIT(**json.loads(") and code.endswith("))"):
            return FinalOutput(json.loads(literal_eval(code[len("SUBMIT(**json.loads(") : -2])))
        if code == READ:
            payload = self.tools["read"](reference=self.context["references"][0])
        elif code == BAD_READ:
            payload = self.tools["read"](
                reference={
                    key: value
                    for key, value in self.context["references"][0].items()
                    if key != "type"
                }
            )
        elif code == SEARCH:
            payload = self.tools["search"](query="jade", k=1)
        elif code == EDGES:
            payload = self.tools["edges"]()
        elif code == CHILD:
            payload = self.tools["query_node"](node_id="b", question="Find the region")
        elif code == CYCLE:
            payload = self.tools["query_node"](
                node_id=self.context["node_id"], question=self.context["question"]
            )
        else:
            raise AssertionError("Replay does not execute arbitrary Python: " + code)
        return json.dumps(payload)

    def shutdown(self):
        """Record cleanup on the same thread that owns the interpreter."""
        self.closed = True


class ReplayFactory:
    """Provide DSPy's public interpreter factory contract and retain lifecycle evidence."""

    def __init__(self):
        """Start without sessions or executed commands."""
        self.sessions, self.started, self.codes = [], [], []

    def __call__(self):
        """Create an independent finite interpreter for one RLM invocation."""
        repl = ReplayREPL(self)
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
        context = node_context(request)
        node_id = context["node_id"]
        plan = self.plans.get(node_id, [READ])
        index = step(request)
        if index < len(plan):
            return ModelResponse(action(plan[index]))
        payload = observation(request)
        citations = [record["id"] for record in payload.get("evidence", [])]
        return ModelResponse(
            submit(node_id + " findings", citations, payload.get("unresolved", []))
        )

    async def synthesize(self, request):
        """Combine attributed branch findings and preserve their supplied canonical citations."""
        self.main_requests.append(request)
        payload = json.loads(request.messages[1].content)
        branches = payload["branches"]
        citations = [record["id"] for record in payload["evidence"]]
        unresolved = [gap for branch in branches for gap in branch["unresolved"]]
        return ModelResponse(finish("Combined findings", citations, unresolved))
