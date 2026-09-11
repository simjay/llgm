# Research design

Use the [working roadmap](../ROADMAP.md) for the benchmark goal and next concrete
deliverables. The design documents below provide background and broader
experimental options. The delivered September 11 node/edge design governs
primary connectivity and delegates. Research hypotheses remain unproven.
Current support and validation limits are recorded in the
[public documentation](../../docs/reference/implementation-status.md).

| Document | Status | Scope |
|---|---|---|
| [Architecture update plan](update-plan.md) | Delivered, September 11, within its recorded validation scope | Primary edges, local amendments, recursive delegates, seed dispatch, time consistency and delivery checks |
| [Overview](overview.md) | Current architecture, September 11 | Implemented primary edges, local journals, deterministic seeds, recursive delegates and final root synthesis |
| [Simplification decision](simplification.md) | Implemented decision, September 10 | Local reasoning, bounded node messages, immutable nodes, explicit journal overwrites, and current reads |
| [Research directions](directions.md) | Open research choices, reviewed September 11 | Alternatives and comparisons compatible with the adopted architecture |
| [Experiment design](experiments.md) | Broader experiment options, subordinate to the current roadmap and delivered contracts | Hypotheses, controls, measurements, data eligibility and decision criteria |
| [September 9 technical baseline](technical-spec.md) | Superseded historical proposal | Original interfaces, storage guarantees and delivery plan retained to explain earlier implementations and reports |

The historical technical specification is not the current API or an additional
set of contributor requirements. Its source-version and mandatory snapshot
schemas were removed from the implementation. Schema-1 workspaces require explicit
handling and are rejected without automatic migration. The original E03
configuration and dated measurements retain their original conditions. They do
not validate the revised source model or node recursion.

See [reports](../reports/README.md) for completed measurements and the
[research index](../README.md) for the full collection.
