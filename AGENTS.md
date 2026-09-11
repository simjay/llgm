# Agent instructions

Before changing this repository, read [agent-context/README.md](agent-context/README.md).
Start with [project context](agent-context/PROJECT_CONTEXT.md) for settled decisions,
then consult [remaining tasks](agent-context/REMAINING_TASKS.md) when the request
involves unfinished work. The current user request determines the task.

Agent context is internal working context, not contributor documentation. `docs/`
is exclusively for library end users: installation, usage, configuration, concepts,
architecture and API reference. Contributor setup, tests, CI, publishing, branding
and documentation maintenance belong in repository-only `development/`, reached
from `CONTRIBUTING.md`. Research proposals and
dated measurements belong in internal `research/`. Do not publish that directory
through documentation links, includes, downloads, mirrors, or package archives.
Keep benchmark run instructions in `experiments/`. Follow the ownership rules in
[documentation maintenance](development/documentation.md).

User docs must read and build independently of research and agent context.
Teach concepts and architecture through concrete examples. Use plain prose with
no em dashes or semicolons, including docstrings published in the API reference.
Remove obsolete or unrelated user guidance when updating behavior.

Update the relevant context when a decision, task status, or local prerequisite
changes. Update the owning end-user guide when an implementation contract changes,
and the development guide when a validation command changes. Preserve historical measurements and distinguish
proposed experiments from implemented and verified behavior.
