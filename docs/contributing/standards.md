# Code standards

## Formatting

Run `make format` to apply Ruff formatting and import ordering. `make lint` and
CI check the same source, test, tool, example and Sphinx configuration files.
The development dependency group pins Ruff so local and CI formatting agree.
Keep this mechanical style separate from decisions about naming and abstractions.

## Comments

Explain why a constraint exists, what can fail, or which invariant a non-obvious operation preserves. Place the explanation beside the code it governs. Keep it short enough to remain accurate when the implementation changes.

```python
# Publish bytes first so committed metadata cannot reference a missing blob.
```

Do not narrate assignments and loops, restate a function name, add progress notes, praise an implementation, or make unsupported claims. Avoid decorative separators and repeated disclaimers. Remove comments made obsolete by a code change. Clear names and a short function are usually better than a paragraph explaining tangled control flow.

## Docstrings

Every Python module, class, function and method has a meaningful docstring. This includes private helpers, constructors, nested functions, test fixtures, test methods and embedded sandbox setup. Lambdas and generated third-party code are outside this rule.

- Start with one sentence stating the contract or tested behavior.
- Add details only where needed: units, ownership, side effects, visibility, meaningful failure modes or a return value that types alone do not explain.
- Do not repeat the signature or describe each implementation step.
- A test docstring describes the invariant, such as “A restart preserves passage IDs and ranking.” It does not promise overall correctness.
- Keep historical design discussions in internal repository notes, not API docstrings.

`python tools/check_docstrings.py` checks presence across `src`, `tests`, `examples`, `tools`, and `docs/conf.py`, including private definitions and embedded sandbox setup. Review still determines whether the wording helps.

## Abstractions

Use the smallest interface that captures a real boundary. Prefer functions for stateless transformations, records for data and explicit objects for owned resources. Add an abstraction when concrete implementations or a clear invariant require it. Do not invent extension points for hypothetical backends.

Keep provider SDKs inside adapters. Keep experimental policy choices visible in configuration and manifests. Do not hide work, retries, fallback methods or unknown costs behind a generic helper. Reuse a helper when behavior is actually shared, not merely because two blocks look alike.

## Validation

Test observable contracts and failure paths. Do not add assertion-free tests or mocks that merely replay the implementation. Coverage identifies code to inspect. A percentage is not proof of correctness. Follow [testing methodology](testing.md) and record live prerequisites and skipped checks accurately.

Documentation placement, navigation, and update responsibilities are defined in
the [documentation conventions](documentation.md).
