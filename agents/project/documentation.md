# Documentation playbook

Read before editing user or developer documentation.

## Scope

- Docs are reStructuredText under `docs/`, built on Read the Docs.
- User-visible behavior changes update the guide that covers them
  (`docs/guide/`). Check documented behavior against the code, not memory.
- `docs/guide/testing.rst` maps the test tiers, markers, and gates.
- Never edit `CHANGELOG.md`; it is generated at release.

## Style

- Write and edit with the slop-mop skill. The rules below add what it does
  not cover.
- Write like a developer explaining to a colleague. No headline-style
  headings and no news-article cadence.
- Examples must grep-hit the codebase unless marked simplified. Cite
  symbols, never line numbers.
- Developer docs cite AGENTS.md rule IDs or AGENTS.project.md contract
  names instead of copying process rules.
