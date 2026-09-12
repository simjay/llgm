# Testing methodology

Use this page to choose evidence for a change. Commands, dependency setup and
integration gates belong in the [testing guide](../development/testing.md).
The [code map](../development/README.md#find-the-owning-code) identifies owners.

## Match the check to the claim

| Check | What it establishes |
| --- | --- |
| Deterministic tests | Storage, scheduling, configuration and error-handling contracts |
| Local dataset checks | Those contracts on pinned conversation histories, without model answers |
| Real Docker checks | Python execution, callbacks, isolation, output limits and cleanup |
| Provider and service checks | The configured adapter works across an actual external boundary |
| Frozen answer evaluation | Answer quality under the declared sources, models, prompts and limits |

Scripted clients and replay transports make behavior reproducible. They do not
establish model reasoning quality. Replay transports do not execute generated
Python, so use the real Docker checks for that boundary.

## Protect the observable contracts

- Preserve exact source text, Unicode spans, immutable identities and retry
  behavior across publication and restart.
- Keep primary edges independent of journals. Check scoped and timed patches,
  replacement provenance, conflicts, withdrawal and missing-target failures.
- Schedule every admitted seed. Isolate child context, preserve delivered
  findings, and reject citations that were not returned to their recipient.
- Exercise exhausted limits, cancellation and cleanup. Distinguish a branch's
  missing fact from required failures that the root must retain.
- Check ownership, configuration precedence, provider response validation and
  the difference between known usage, missing usage and estimated cost.

Start with [storage tests](../tests/test_storage.py),
[effective reads](../tests/test_effective_reads.py),
[node execution](../tests/test_nodes.py) and
[application tests](../tests/test_application.py) for these boundaries.
Prefer a regression reproducing an observed failure to assertions about private
helper layout, exact prompts or an implementation recipe.

## Keep evaluation separate

The [LongMemEval runbook](../experiments/longmemeval.md) owns the primary answer
protocol. Preparation records readiness, not completed model work. Freeze inputs
and planned trials before generation, and judge saved predictions separately.
Keep gold answers and label-bearing identifiers out of model-visible evidence.

Trace evidence through retrieval, seed selection, local reading, branch return
and final synthesis. Source coverage is not answer accuracy. Valid references
identify text but do not prove that it supports a claim. A recursion claim needs
an observed child invocation and delivered findings.

Retain failed and unstarted attempts in reporting. Preserve unknown usage and
unavailable judgments. Repeated exposed questions are development diagnostics,
not independent held-out evidence. Compare models and budgets explicitly.

## Choose and report checks

Run focused checks first, then `make check` for an integrated code change.
Use `make coverage` to find untested branches, not to claim semantic quality.
Use `make docs` for the user site, `make docs-links` for shared context and other
repository guidance, and `make build` for package changes.
The testing guide covers additional installed-package and integration checks.

Optional checks require their declared inputs and services. A skip does not
validate a boundary. Report what ran, what failed and what was skipped, and
distinguish syntax checks from executed examples and hosted model trials.
