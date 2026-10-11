"""Tier-1 (`make gate`) must select no test that needs models or a live ZM.

The e2e directories rely on markers to stay out of Tier-1. A test there
without its marker runs real models on a machine that has them and skips in
CI, so the two greens measure different things.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TIER1 = "not e2e and not zm_e2e and not serve"  # same as TIER1_SELECT in the Makefile


@pytest.mark.parametrize("path", ["tests/test_ml_e2e", "tests/test_zm_e2e"])
def test_tier1_selects_nothing_from_e2e_dirs(path):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-m", TIER1, "--co", "-q", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    )
    # Exit 5 is "no tests collected"; "deselected" proves the directory had
    # tests to deselect, so an empty or moved directory cannot pass.
    assert r.returncode == 5 and "deselected" in r.stdout, f"Tier-1 selects tests from {path}:\n{r.stdout[-1500:]}"
