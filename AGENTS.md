# Agent instructions

Before changing this repository, read [Contributing](CONTRIBUTING.md) and
[documentation maintenance](docs/contributing/documentation.md). These tracked guides
are the canonical contributor instructions. The current user request determines
the task.

Read [agent context](agent-context/README.md) and
[project context](agent-context/PROJECT_CONTEXT.md) for shared project knowledge.
Load the architecture, decisions, testing, and remaining-task files only as needed
for the current request. Inspect the relevant implementation and tests before
changing a contract. Start with `git status --short` and preserve unrelated edits.

## Setup and validation

Use `make setup` for the local development environment. Run affected tests first,
then `make check` for deterministic validation. Use `make docs` for the user site,
`make docs-links` for repository guidance, and `make build` for packaging changes.
The [testing guide](docs/contributing/testing.md) owns optional integration prerequisites
and commands. Test availability does not establish that a service was exercised.

## Code and documentation

Keep interfaces small and explicit. Add docstrings to every Python definition
and comments where they explain a constraint or non-obvious choice. Follow
[code standards](docs/contributing/standards.md). Prefer observable contract tests to
tests that repeat the implementation.

Agent context supports coding agents and does not replace contributor guidance.
`docs/guide/` and `docs/reference/` serve library end users: installation, usage,
configuration, concepts, architecture and API reference. Contributor setup, tests,
CI, publishing, branding and documentation maintenance belong in repository-only
`docs/contributing/`, reached
from `CONTRIBUTING.md`. Research proposals and dated measurements belong in local
`research/`. The six curated Markdown files allowed by `.gitignore` under
`agent-context/` are tracked repository context. Other files there, including
`local/` and `LOCAL_STATE.md`, remain private. Never force-add private context or
research, and never copy their contents into tracked files. Tracked guides,
tests, and builds must work without private notes. Do not include any agent
context in the user site, its downloads, or package archives.
Keep benchmark run instructions in `experiments/`. Follow the ownership rules in
[documentation maintenance](docs/contributing/documentation.md).

User docs must read and build independently of research and agent context.
Teach concepts and architecture through concrete examples. Use plain prose with
no em dashes or semicolons, including docstrings published in the API reference.
Remove obsolete or unrelated user guidance when updating behavior.

## Keep context current

Update shared context when a settled decision, code boundary, or public task status
changes. Keep machine prerequisites and session details in ignored local notes.
Do not treat a backlog item or past session as authorization for a new action.
Update the owning end-user guide when an implementation contract changes,
and the development guide when a validation command changes. Preserve historical
measurements and distinguish proposed experiments from implemented and verified
behavior.
