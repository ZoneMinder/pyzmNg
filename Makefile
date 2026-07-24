# pyzm developer test gate.
#
# Run `make hooks` once after cloning to install the pre-push gate.
# `make gate`         -> fast Tier-1 (unit + integration), no models/GPU/ZM.
# `make release-gate` -> Tier-1 + real e2e (models + live ZM), skips forbidden.

PY ?= python3
PYTEST := $(PY) -m pytest
# Tier-1: everything that needs no models, no GPU, no live ZM, no subprocess.
TIER1_SELECT := -m "not e2e and not zm_e2e and not serve"

.PHONY: gate release-gate test hooks help

help:
	@echo "make gate          - fast pre-push gate (Tier-1, ~20s)"
	@echo "make release-gate  - full gate incl. e2e (needs models + live ZM)"
	@echo "make test          - run everything pytest can collect"
	@echo "make hooks         - install the git pre-push hook (run once)"

# The pre-push gate. Fast, hermetic, deterministic. This is the contract:
# green here means the change broke nothing that ran without external deps.
gate:
	$(PYTEST) tests/ $(TIER1_SELECT) -q

# Pre-release gate. PYZM_E2E_REQUIRE=1 turns "prereq missing" from a silent
# skip into a hard failure, so a green release-gate proves e2e actually ran.
release-gate: gate
	PYZM_E2E_REQUIRE=1 $(PYTEST) tests/test_ml_e2e/ tests/test_zm_e2e/ -v

test:
	$(PYTEST) tests/ -q

# Version-controlled hooks live in .githooks/. One command wires them up.
hooks:
	git config core.hooksPath .githooks
	@echo "pre-push gate installed (core.hooksPath -> .githooks)"
