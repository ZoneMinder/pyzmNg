#!/usr/bin/env python3
"""Mutation smoke (AGENTS.md P2, M2): break one line in each risky module and
require that module's tests to fail. Proves the existing tests can catch a
wrong value, which proven red cannot show for tests that were already there.

    python3 scripts/gates/mutation_smoke.py

Each target replaces one exact line fragment, runs its tests, and restores the
file whether or not they failed. Exit 0 every mutant was caught, 1 a mutant
survived, 2 a target no longer matches the source (update the target).
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (file, from, to, test file)
TARGETS = [
    # Secrets in logs: redaction does nothing
    ("pyzm/zm/auth.py", 'return _SECRET_RE.sub(r"\\1\\2***", str(text))', "return str(text)",
     "tests/test_zm/test_auth.py"),
    # ES interface: a wire key is renamed
    ("pyzm/models/detection.py", '"labels": self.labels,', '"label": self.labels,',
     "tests/test_models/test_wire_contract.py"),
    # Push throttling: the comparison flips
    ("pyzm/models/zm.py", "return elapsed < self.interval", "return elapsed > self.interval",
     "tests/test_models/test_zm.py"),
]


def main():
    survived = []
    for rel, old, new, test in TARGETS:
        path = REPO / rel
        src = path.read_text()
        if src.count(old) != 1:
            print(f"mutation-smoke: {rel}: '{old}' found {src.count(old)} times, expected 1; update the target")
            return 2
        path.write_text(src.replace(old, new))
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", test],
                               cwd=REPO, capture_output=True, text=True)
        finally:
            path.write_text(src)
        if r.returncode == 0:
            survived.append(f"{rel}: '{old}' -> '{new}'")
            print(f"SURVIVED {rel}: tests still pass")
        else:
            print(f"caught   {rel}")
    if survived:
        print("mutation-smoke: these mutants survived; the tests cannot catch them:\n  " + "\n  ".join(survived))
        return 1
    print(f"mutation-smoke: {len(TARGETS)} of {len(TARGETS)} mutants caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
