# Testing playbook

Read before writing or changing tests.

## Test design

- Test outcomes: the returned value, the request sent, the file written,
  the log line emitted, errors, and edge cases. Never type or existence
  alone; the ratchet counts `assert isinstance(` lines (C6).
- No `if result.matched:` style guard around assertions; it lets an empty
  result pass.
- Name the seam under test before writing a test: the highest interface
  that reaches the behavior. Mock only external I/O (ZoneMinder HTTP, the
  database, model files). A mock of a pyzm class must match the real one.
- Prove red before green: run the new test against the pre-change code and
  show it fail (`sh scripts/gates/proven-red.sh <base> <head>`; CI runs it
  on every push and PR).
- A new gate assertion (anything under `scripts/gates/` or
  `tests/test_instruction_gate.py`) is proven red against a scratch
  violation, then the scratch is removed in the same commit.
- A change to a field ES reads updates `tests/test_models/test_wire_contract.py`
  and, in ES, hook/tests/test_pyzm_contract.py.
- No fixed sleeps. Wait on a condition.

## Commands

```bash
make gate                                         # Tier-1, ~20s
python3 -m pytest tests/test_zm/test_auth.py -q   # one file
make release-gate                                 # ML and ZM e2e; needs models and a live ZM
make -C ~/fiddle/zmeventnotificationNg test-all   # ES gate plus this gate
```

`make release-gate` sets PYZM_E2E_REQUIRE=1, so a missing model or
unreachable ZM fails instead of skipping. ZM e2e reads its server from
`.env.zm_e2e` in the repo root (sample: `tests/.env.zm_e2e.sample`). Write-tier ZM tests
run only by hand.
