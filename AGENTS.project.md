# pyzmNg Project Instructions

Read `AGENTS.md` first. Contracts, project rules, verification, playbooks.
The sanctioned path is the only path; a bypass is a bug even when it works.

pyzm is the Python library for ZoneMinder: the API client (`pyzm/zm/`,
`pyzm/client.py`), the ML detection pipeline (`pyzm/ml/`), the remote ML
gateway (`pyzm/serve/`), model training (`pyzm/train/`), and logging
(`pyzm/log.py`). zmeventnotificationNg (ES) and its `zm_detect.py` depend on
it; ES lives at ~/fiddle/zmeventnotificationNg.

## Architecture contracts

### ES interface
Owns: the public shape ES consumes: the `DetectionResult` wire dict, the
`detect_event` signature, and the classes `zm_detect.py` imports.
Path: `DetectionResult` with `to_dict` and `from_dict`
(`pyzm/models/detection.py`); `Detector` and `detect_event`
(`pyzm/ml/detector.py`).
Never: renaming or removing a wire key, or changing a signature ES calls,
without the matching ES change; a test mock of a pyzm class that differs from
the real one. Before changing this shape, read the ES call sites in
hook/zm_detect.py and hook/zmes_hook_helpers/ in ES.
Gate: `tests/test_models/test_wire_contract.py`; the es-contract CI job
runs ES's hook/tests/test_pyzm_contract.py against this checkout; review for
mock fidelity.

### Secrets in logs
Owns: credentials and API keys in log lines, error strings, and responses.
Path: `redact_secrets` (`pyzm/zm/auth.py`) on every URL, request params, or
request exception that reaches a log line or an error string.
Never: an authenticated URL, params dict, or requests exception logged or
returned unredacted. ZM tokens and ALPR keys leaked this way twice
(ab1464b, afcb373).
Gate: `tests/test_zm/test_auth.py`; review for new log lines.

### Outbound HTTP
Owns: every HTTP request pyzm makes.
Path: `request` on `ZMAPI` (`pyzm/zm/api.py`) for ZoneMinder; any other
requests call passes timeout=.
Never: a requests call without a timeout; a hung server then hangs
detection for the event.
Gate: the ratchet holds http_calls_without_timeout.

### Logging
Owns: diagnostic output from library code.
Path: a module logger from logging.getLogger; ZoneMinder log setup through
`setup_zm_logging` (`pyzm/log.py`).
Never: print() in library code. `__main__.py` entry points and the
Streamlit app in `pyzm/train/` are exempt.
Gate: `tests/test_instruction_gate.py` (no print call outside the exempt
files).

## Project rules

- Run commands from the repo root. Run `make hooks` once per clone so the
  pre-commit and pre-push gates exist.
- Issues and PRs go to `ZoneMinder/pyzmNg`, with a label. Gate: review.
- Never edit `CHANGELOG.md`; it is generated at release. Gate: the
  `pr-acceptance` CI job.
- Never commit plan files (`PLAN.md`, `*.plan.md`). Gate:
  `tests/test_instruction_gate.py`.
- A version bump updates `setup.py` here and the pyzm pin in ES's
  hook/setup.py. Gate: review.
- Write-tier ZoneMinder tests (ZM_E2E_WRITE) run only by hand; never set
  ZM_E2E_WRITE in test logic. Gate: review.
- Run `make release-gate` before a release. Gate: review.
- Match the style of the file being edited. Gate: review.
- Prose people read (docs, commit bodies, PR and issue bodies, review
  comments) is written with the slop-mop skill; prose reviews run its
  detect mode on the harness's top coding model.
- A `feat` PR links its spec or says in its `## Spec` section why it
  needs none. Gate: the `pr-acceptance` CI job.
- GitHub comments by an agent end with `Posted by <agent>, assisting
  @<login>.` where `<login>` comes from `gh api user --jq .login`. Add
  comments; never edit anyone else's.

## Verification

```
make gate            # Tier-1 unit + integration, instruction gate, ratchet; ~20s
make release-gate    # gate + ML and ZM e2e with PYZM_E2E_REQUIRE=1
make mutation        # mutation smoke
```

Per commit, run what the change touches; `make gate` before push or PR.
Ratchet baselines lower with `sh scripts/gates/ratchet.sh --update`, which
refuses to write a rise; raising one is a hand edit to `.ratchet-baseline`
with a reason in the commit message (C7). CI also runs proven red (P2,
`scripts/gates/proven-red.sh`), the mutation smoke, ES's pyzm contract
test, and `pr-acceptance` (PR body needs `## Acceptance` content, `## Spec`
on a `feat`, and no `CHANGELOG.md` edit). State completed checks in handoff.

## Playbooks

Read each listed playbook before work in that area.

| Work | Read first |
|---|---|
| Multi-agent or long-running work | `agents/generic/agent-workflows.md` |
| Naming, briefs, docs, proposing work | `agents/project/glossary.md`, `agents/project/out-of-scope.md` |
| Tests | `agents/project/testing.md` |
| Developer or user documentation | `agents/project/documentation.md` |
| ZoneMinder API, ML backends, serve, quirks | `agents/project/domain-context.md` |

Portable playbooks live in `agents/generic/`; project ones in
`agents/project/`.
