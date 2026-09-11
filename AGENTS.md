# Agent instructions

Before changing this repository, read [Contributing](CONTRIBUTING.md) and
[documentation maintenance](development/documentation.md). These tracked guides
are the canonical contributor instructions. The current user request determines
the task.

If local `agent-context/` files are present, read `agent-context/README.md` and
`agent-context/PROJECT_CONTEXT.md` for additional working context. Consult
`agent-context/REMAINING_TASKS.md` when the request involves unfinished work.
Their absence must not block work in a fresh clone.

Agent context is internal working context, not contributor documentation. `docs/`
is exclusively for library end users: installation, usage, configuration, concepts,
architecture and API reference. Contributor setup, tests, CI, publishing, branding
and documentation maintenance belong in repository-only `development/`, reached
from `CONTRIBUTING.md`. Research proposals and dated measurements belong in local
`research/`. Both `research/` and `agent-context/` are ignored, untracked local
directories. Do not stage, commit, push, mirror, or package their contents.
Tracked guides, tests, and builds must work without them. Do not link to them
from tracked documentation or include them in published pages or downloads.
Keep benchmark run instructions in `experiments/`. Follow the ownership rules in
[documentation maintenance](development/documentation.md).

User docs must read and build independently of research and agent context.
Teach concepts and architecture through concrete examples. Use plain prose with
no em dashes or semicolons, including docstrings published in the API reference.
Remove obsolete or unrelated user guidance when updating behavior.

Update existing local context when a decision, task status, or local prerequisite
changes. Update the owning end-user guide when an implementation contract changes,
and the development guide when a validation command changes. Preserve historical
measurements and distinguish proposed experiments from implemented and verified
behavior.
