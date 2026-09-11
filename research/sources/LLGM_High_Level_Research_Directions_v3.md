---
title: "Large Language Graphical Models (LLGM)"
subtitle: "High-Level Research Directions"
author: "Working research brief"
date: "September 2026"
geometry: margin=1in
fontsize: 11pt
---

# Purpose

This document is a starting point for research, not a finalized architecture. It records the central LLGM motivation, the few design choices currently considered essential, and the questions that should remain open until they are tested.

# Core motivation

Long-running LLM sessions usually become harder to manage as they grow. The transcript competes for a bounded context window, recent turns displace old ones, and compaction can discard exact qualifications, revisions, and evidence.

LLGM explores a different experience:

> A long interaction should become a collection of addressable information nodes, while each new inference uses only the information needed for that task.

The total retained experience may grow continuously, but the active context supplied to an expensive model should remain bounded. This does not guarantee that quality always improves with time. It creates the possibility that useful experience can accumulate without being repeatedly compressed into one shrinking summary.

# Working hypothesis

LLGM combines three ideas:

1. **Graphical-model-style inference.** A global task is decomposed into local information states and local computations. Relevant information is passed between them rather than exposing the entire global history to every computation.
2. **RLM-style context querying.** A large information node is treated as an external environment that can be searched, inspected, decomposed, and recursively queried. The node does not need to fit inside a model context window.
3. **Language-model edge functions.** The connection between a source node and a receiving computation is executable. A language model decides whether the source is useful for the current query, determines what the receiver needs, and returns a bounded message.

The resulting system is not simply a memory database. It is an attempt to make persistent context participate in a factorized inference process.

# Core objects

## Full-fidelity node

A node contains a complete, contiguous Q&A segment. The source text is canonical. The node is not replaced by a generated summary.

Small routing metadata or derived indexes may exist, but they are access mechanisms rather than alternate memories.

## Relevant-node search

Before message passing begins, the system needs a fast way to propose nodes that may matter for the current query. The exact method is deliberately open. Candidate directions include lexical retrieval, embedding retrieval, late-interaction retrieval, graph-aware retrieval, learned routing, or combinations of these.

This component answers only:

> Which nodes are worth examining?

It does not determine what message they should send or whether their information is current, obsolete, or sufficient.

## Functional edge

For a query \(q\), source node \(N_i\), receiving computation \(F_j\), and message budget \(B_{ij}\), the working abstraction is:

$$
E^{(q)}_{i\rightarrow j}(N_i,q,F_j,B_{ij})
\rightarrow
\{\text{inactive}\}\;\text{or}\;m_{ij}.
$$

The edge function may be implemented primarily by a smaller LLM. It decides whether information should flow and formulates a receiver-specific request to the source node.

In conventional factor-graph notation, functions normally appear as factor nodes rather than on bare wires. The exact formal mapping between LLGM's executable-edge abstraction and classical factor graphs remains part of the research agenda.

## RLM-style message generation

When an edge is active, it recursively queries the full source node:

$$
m_{ij}=\operatorname{RQuery}(N_i,r_{ij},B_{ij}),
$$

where \(r_{ij}\) expresses what the receiver needs. The result is a bounded, source-grounded message rather than a general summary of the node.

The same node may produce different messages for different receivers:

$$
m_{i\rightarrow F_1}\neq m_{i\rightarrow F_2}.
$$

## Inference factor

The difficult task computation consumes the messages. It may be one frontier LLM, multiple models, a tool-assisted model, or another compound computation. This is where most of the reasoning budget should be spent.

# Model roles

LLGM proposes a division between control work and task work.

**Smaller LLMs act as glue.** Candidate roles include conversation grouping, query clarification, node relevance judgment, edge activation, receiver-requirement generation, routine extraction, and lightweight recursive calls.

**Larger models perform high-value inference.** Candidate roles include difficult analysis, coding, planning, synthesis, verification, and hard recursive subproblems.

**Deterministic systems handle mechanics.** Storage, indexing, access control, graph execution, caching, provenance, budgets, and telemetry should not require a frontier model.

This allocation is a hypothesis to evaluate. A small model may be insufficient for some edge or RLM operations, and escalation policies remain open.

# Query-time structure

The persistent state can remain simple: full-Q&A nodes, chronology, and whatever derived index is needed to find candidates. The more meaningful graph is induced for a particular query.

```text
                     current query
                           |
                  relevant-node search
                    /       |       \
                  N1        N2       N3
                   |         |        |
             LLM edge    LLM edge  LLM edge
                   |         |        |
             RLM message     |    no activation
                    \        /
                     inference factor
                           |
                         answer
```

The query-time topology depends on the question, the receiving computation, and the available budget. It may be discarded after the request, aside from traces or caches.

# Four phases

## 1. User query received

The system stores the exact request, loads the recent local context, and initializes the inference run. This phase should be mostly deterministic.

## 2. Beginning of inference

The system searches for potentially relevant nodes. Smaller models may clarify the query, rerank candidates, and activate initial edge functions. RLM-style node queries produce the first bounded messages.

## 3. During inference

The main factor performs the task. It may request more information, activate additional edges, recursively query nodes, use tools, or run another model. Cycles and iterative refinement are possible, but their scheduling and stopping rules are open research questions.

## 4. After inference

The exact Q&A is retained. A small model may decide whether the interaction continues the current node or forms a new logical checkpoint. Derived search indexes can then be updated. Grouping is segmentation, not summarization.

# Long-running interaction and correction

The long-term goal is a session in which accumulated experience remains available without being loaded in full on every turn. The context window becomes temporary working memory; the node collection becomes durable experience.

Forgetting should be studied as at least three different operations:

- **Selective recall:** unrelated nodes are not activated.
- **Logical suppression:** relevant but outdated or contradicted information is qualified or withheld for the current task.
- **Physical deletion:** source content and its derived indexes are removed under user or policy control.

How corrections, contradictions, time, and authority should affect edge functions is not fixed yet.

# What is currently fixed

The research direction currently assumes:

- canonical nodes retain full Q&A rather than summary-only memory;
- node grouping is incremental and may use smaller models;
- a separate mechanism searches for candidate nodes;
- edge functions are query-conditioned LLM computations;
- selected nodes are recursively queried to form bounded messages;
- larger models are reserved primarily for difficult inference;
- all control-plane compute must be counted when evaluating efficiency.

# What remains open

The following should not be treated as settled design:

- the best node-boundary policy and typical node size;
- the best relevant-node search technique;
- whether retrieval should operate over whole nodes, passages, or both;
- the exact relationship between executable edges and classical factor nodes;
- how many models or factors should participate in one inference;
- when cycles improve quality and how convergence should be defined;
- how to detect obsolete, contradictory, or superseded information;
- whether the query-time graph should be planned, discovered incrementally, or both;
- how small the glue models can be;
- which tasks benefit enough to justify the added compute;
- which benchmarks measure the intended long-running experience.

# Initial research questions

1. Can bounded, receiver-specific messages preserve answer quality better than generic summaries at the same context budget?
2. Does RLM-style querying of full Q&A nodes outperform direct passage retrieval or RLM over one linear transcript?
3. Does graphical locality reduce heavy-model context and total work as retained history grows?
4. Can smaller LLMs reliably perform grouping and edge functions without becoming the main source of errors?
5. When do iterative or cyclic message exchanges improve inference?
6. Can the system handle correction and changing user intent without destructively rewriting history?
7. Does LLGM provide value beyond strong long-context, retrieval, and agent-memory baselines?

# Initial empirical direction

Existing long-term conversational benchmarks such as LongMemEval and LoCoMo can test recall, multi-session reasoning, temporal reasoning, and updates. They are useful starting points but may not fully test the LLGM hypothesis.

A custom longitudinal study may be needed in which a project evolves over many interactions, earlier decisions are revised, several topics are interleaved, and later questions require different views of the same source nodes. The key measurements should include answer quality, relevant-node recall, source grounding, heavy-model context, total model work, latency, and performance as total history grows.

# Positioning

LLGM should not be positioned as a digital twin. A digital twin is one possible application. The broader proposal is:

> **A persistent inference substrate that uses graphical locality, executable language-model edges, and recursive context queries to turn large accumulated experience into bounded task-specific messages.**

This is the idea to investigate. The retrieval backend, formal graphical semantics, and production architecture should follow the evidence rather than being fixed in advance.
