# Glossary

One name per concept, for code identifiers, briefs, docs, and commit
messages. Each entry says what the thing is, not what it does, and lists
the words to stop using for it. User-facing copy follows the product
wording and is exempt. A quality ratchet may count avoided terms in agent
and developer prose.

## Process

**Contract**:
An architecture entry in `AGENTS.project.md`: what it owns, the sanctioned
path, forbidden bypasses, and its gate.
_Avoid_: architecture rule, architecture convention

**Gate**:
A command that fails a commit, push, or CI run when a rule breaks.
_Avoid_: blocking check, guard script

**Brief**:
The requirements file handed to a subagent for one task.
_Avoid_: task prompt, instructions file

**Spec**:
A design document approved before a plan is written.
_Avoid_: design doc

**Seam**:
The interface a test exercises; the highest one that reaches the behavior.
_Avoid_: test boundary, layer under test

**Proven red**:
A new test that was run against the pre-change code and shown to fail there.
_Avoid_: verified failing, shown failing

## pyzm

**ES**:
zmeventnotificationNg, the Event Server whose `zm_detect.py` hook calls pyzm.
_Avoid_: event server (lowercase), notification server

**Gateway**:
A pyzm.serve process that runs models for remote clients (ml_gateway).
_Avoid_: ML server, inference server, mlapi

**Wire dict**:
The dict `DetectionResult.to_dict` returns and ES reads by key name.
_Avoid_: matched_data (except in code that already uses it), payload

**Tier-1**:
Tests that run with no models, GPU, or live ZoneMinder (`make gate`).
_Avoid_: unit tests (alone), fast tests
