# Development Guidelines

Read this file before work. It is the single source of truth for how to work in
this repo; `CLAUDE.md` just points here.

`pyzm` is the Python library for ZoneMinder — API client, ML detection pipeline,
and logging. It is the **source of truth** that `zmeventnotification` (ES, at
`~/fiddle/zmeventnotificationNg`) and its `zm_detect.py` depend on. A change to a
public interface here (a `DetectionResult` key, a `detect_event` signature, an
error contract) can break ES in production. Treat public shape as a contract.

| Work | Read first |
|---|---|
| Any code, test, or config change | This file, then run the gate (see Verification) |
| Public API used by ES (`Detector`, `DetectionResult`, `StreamConfig`, `ZMClient`) | Check the ES call sites at `~/fiddle/zmeventnotificationNg/hook` before changing shape |
| Docs | `docs/guide/testing.rst` for the test map |

## Core rules

1. Write plain, factual prose. No marketing claims, filler, or recap sections.
2. Create or use a GitHub issue before feature or bug work, on
   `ZoneMinder/pyzmNg` (the canonical repo). Label it. A user instruction to use
   an existing issue overrides creating one.
3. **Test first.** Write the failing test before the production code. Watch it
   fail for the right reason, then write the minimal code to pass. If you did not
   see the test fail, you do not know it tests anything.
4. Every new feature or bugfix ships with a test that fails before the change and
   passes after. Bug fixes start with a test that reproduces the bug. If you
   cannot write one, say why in the PR.
5. **Run the gate before every commit** (see Verification) and confirm it is
   green. Never commit after a failed or unrun gate.
6. A test that stays green when the code it covers is broken is worse than no
   test. No tautologies (assertions that pass regardless of correctness, e.g.
   `assert isinstance(x, list)`), no re-implementing the logic under test, no
   `if result.matched:`-style guards that let an empty result pass silently.
   A useful test fails when the behavior breaks.
7. **Mock fidelity.** A mock must match the real interface it stands in for. When
   you mock a class, verify its attributes, return shapes, and method signatures
   against the real source. Prefer exercising real code; mock only external I/O.
   The public shape ES consumes is verified by the contract tests — keep them
   passing and add to them when you add a consumed field.
8. Follow DRY. Write simple code. Match the style of the file you are editing.
9. Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`,
   `test:`. Scope optional (`feat(serve):`). One logical change per commit.
   Reference the issue in the body (`refs #<id>`); close only after the user
   confirms.
10. Never edit `CHANGELOG.md`. It is auto-generated.
11. Do not commit plan files (`PLAN.md`, `*.plan.md`, implementation plans).
    They are temporary; delete them when the task is done.
12. When bumping the pyzm version, update `setup.py` here AND the pinned version
    in `~/fiddle/zmeventnotificationNg/hook/setup.py`.
13. When responding to issues or PRs from others, add comments, never overwrite
    anyone's (including an AI agent's). Identify yourself as Claude.
14. Access DB, configs, and secrets as the ZM user: `sudo -u www-data`.
15. Read failures. Fix the cause. Do not blindly retry or weaken a test to make
    it pass.

## Verification (the test gate)

Install the pre-push hook once per clone:

```bash
make hooks        # git config core.hooksPath -> .githooks
```

Before every commit, run the gate. It runs on `git push` automatically, and
blocks the push when red (override a genuine emergency with `git push
--no-verify`).

```bash
# Tier-1: unit + integration. ~20s, no models / GPU / live ZM. This is the
# per-commit / per-push gate. Green here = nothing that runs without external
# deps was broken.
make gate

# Pre-release: Tier-1 + real e2e (ML models on disk + live ZoneMinder).
# PYZM_E2E_REQUIRE=1 turns a missing prerequisite into a FAILURE instead of a
# silent skip, so a green release-gate proves e2e actually ran. Run this on a
# box that has models and a reachable ZM before cutting a release.
make release-gate
```

Verification runs the real commands and reads the real output. Do not claim
green from memory. State which tiers you ran in the handoff.

The suite is a regression net: a green gate should mean a new change broke
nothing that worked. Tiers must not silently skip themselves green — that is
what `PYZM_E2E_REQUIRE=1` guards against. Do not weaken a test to make the gate
pass.

`docs/guide/testing.rst` maps the tiers and markers. E2E tests use real models
and a real server; do not set `ZM_E2E_WRITE` in test logic — write-tier ZM tests
are run manually by the user.

## Documentation

Docs live in `docs/` (Read the Docs / reStructuredText). Write as an expert who
cares that documentation is clear, easy to follow, comprehensive, user-forward,
and CORRECT. The docs must fully represent the system's real capabilities with
nothing outdated or incomplete. When you touch a subsystem, check its docs still
match reality and fix drift — validate documented behavior by reading the code,
not by assuming. Never edit `CHANGELOG.md`; it is auto-generated.
