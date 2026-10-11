"""Instruction gate (AGENTS.md M1, M2), adapted from gap-trap's pytest reference.

Checks the instruction files themselves: every name a contract cites exists
where the contract says it lives, the portable core stays portable, the
always-loaded files stay small, cited commits exist, knowledge files hold no
private data, docs cite rule IDs that exist, no plan file is tracked, and
library code has no print() call (Logging contract).

Every check asserts its own denominator (tokens resolved, files scanned), so a
check that quietly measured nothing cannot pass (M2).
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

# ---- config ---------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ["pyzm", "scripts"]
SOURCE_EXT = (".py", ".sh")
TEST_PARTS = {"tests", "build"}
FORBIDDEN_IN_CORE = ["pyzm", "ZoneMinder", "zmeventnotification", "Streamlit",
                     "OpenCV", "agents/project"]
WORD_BUDGET = 2500  # 1563 at setup plus half, rounded up to 500 (C7)
MIN_CONTRACTS = 4
ALWAYS_LOADED = ["AGENTS.md", "AGENTS.project.md", "CLAUDE.md"]
KNOWLEDGE_FILES = [
    "agents/project/domain-context.md",
    "agents/project/glossary.md",
    "agents/project/out-of-scope.md",
    "agents/generic/agent-workflows.md",
]
DOCS_DIR = "docs"
# Logging contract: print() is allowed only in these library files.
PRINT_EXEMPT = lambda rel: rel.endswith("/__main__.py") or rel.startswith("pyzm/train/")
PLAN_FILE_RE = re.compile(r"(^|/)PLAN\.md$|\.plan\.md$|^docs/plans/", re.I)
# ---- end config -----------------------------------------------------------

FIELD_RE = re.compile(r"^(Owns|Path|Never|Gate):")


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    return out.splitlines()


def is_test_path(rel: str) -> bool:
    parts = rel.split("/")
    return (any(p in TEST_PARTS for p in parts[:-1])
            or parts[-1].startswith("test_"))


def source_files() -> list[str]:
    return [rel for rel in tracked_files()
            if rel.endswith(SOURCE_EXT)
            and any(rel == r or rel.startswith(r + "/") for r in SOURCE_ROOTS)
            and not is_test_path(rel)]


def parse_contracts(md: str) -> list[tuple[str, str]]:
    section = md.split("## Architecture contracts", 1)[1].split("\n## ", 1)[0] if "## Architecture contracts" in md else ""
    out = []
    for block in section.split("\n### ")[1:]:
        name, _, body = block.partition("\n")
        out.append((name.strip(), body))
    return out


def fold_fields(body: str) -> list[str]:
    """Fold wrapped lines back into the field they continue."""
    out: list[str] = []
    open_field = False
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            open_field = False
        elif FIELD_RE.match(line):
            out.append(line)
            open_field = True
        elif open_field:
            out[-1] += " " + line
    return out


def backtick_tokens(line: str) -> list[str]:
    return [t.removesuffix("()") for t in re.findall(r"`([^`]+)`", line)]


def text_under(rel: str) -> str:
    p = REPO / rel
    if not p.exists():
        return ""
    if p.is_dir():
        return "\n".join(read(f) for f in p.rglob("*") if f.suffix in SOURCE_EXT and f.is_file())
    return read(p)


CONTRACTS = parse_contracts(read(REPO / "AGENTS.project.md"))
# A contract symbol found only in a test file is not a symbol the code uses.
NON_TEST_SOURCE_TEXT = "\n".join(read(REPO / rel) for rel in source_files())


def test_contracts_have_all_four_lines():
    assert len(CONTRACTS) >= MIN_CONTRACTS, f"{len(CONTRACTS)} contracts read"
    for name, body in CONTRACTS:
        fields = fold_fields(body)
        for field in ("Owns:", "Path:", "Never:", "Gate:"):
            assert any(l.startswith(field) for l in fields), f"{name} missing {field}"


def test_every_symbol_and_path_in_contracts_exists():
    checked_overall = 0
    for name, body in CONTRACTS:
        lines = [l for l in fold_fields(body) if l.startswith(("Path:", "Gate:"))]
        checked = 0
        for line in lines:
            tokens = backtick_tokens(line)
            paths = [t for t in tokens if "/" in t]
            symbols = [t for t in tokens if "/" not in t]
            for token in paths:
                assert (REPO / token).exists(), f"{name}: path {token} missing"
                checked += 1
            # A Path line's symbols are looked for in the paths it names.
            scoped = line.startswith("Path:") and paths
            haystack = "\n".join(text_under(p) for p in paths) if scoped else NON_TEST_SOURCE_TEXT
            where = ", ".join(paths) if scoped else "non-test source"
            for token in symbols:
                assert re.search(rf"\b{re.escape(token)}\b", haystack), f"{name}: symbol {token} not found in {where}"
                checked += 1
        assert checked > 0, f"{name}: its Path/Gate lines name nothing the gate can check"
        checked_overall += checked
    assert checked_overall > 0, "no contract token was checked at all"


def test_core_is_portable():
    core = read(REPO / "AGENTS.md").lower()
    for token in FORBIDDEN_IN_CORE:
        assert token.lower() not in core, f'AGENTS.md contains "{token}"'


def test_always_loaded_files_within_word_budget():
    for f in ALWAYS_LOADED:
        assert (REPO / f).exists(), f"{f} is missing; the budget would fall for free"
    total = sum(len(read(REPO / f).split()) for f in ALWAYS_LOADED)
    assert total <= WORD_BUDGET, f"combined {total} words > budget {WORD_BUDGET}"


def test_cited_commit_hashes_exist():
    cited = set()
    for f in ("agents/project/domain-context.md", "AGENTS.project.md"):
        md = read(REPO / f)
        # 7 to 40 hex chars with a digit keeps words made of a-f out.
        cited |= {h for h in re.findall(r"\b[0-9a-f]{7,40}\b", md) if re.search(r"\d", h)}
    assert cited, "no commit hash cited; the check measured nothing"
    for h in sorted(cited):
        r = subprocess.run(["git", "cat-file", "-e", f"{h}^{{commit}}"], cwd=REPO, capture_output=True)
        assert r.returncode == 0, f"cited commit {h} not found in history"


def test_knowledge_files_hold_no_private_data():
    scanned = 0
    for f in KNOWLEDGE_FILES:
        p = REPO / f
        if not p.exists():
            continue
        scanned += 1
        text = read(p)
        assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text), f"{f} contains an IP address"
        assert not re.search(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b", text), f"{f} contains an email address"
    assert scanned == len(KNOWLEDGE_FILES), f"{scanned} of {len(KNOWLEDGE_FILES)} knowledge files found"


def test_docs_cite_rule_ids_that_exist():
    valid = set(re.findall(r"^- ([IPCM]\d+)\.", read(REPO / "AGENTS.md"), re.M))
    assert valid
    docs = [p for p in (REPO / DOCS_DIR).rglob("*") if p.suffix in (".md", ".rst")]
    assert docs, "no docs scanned"
    for p in docs:
        for m in re.finditer(r"\brules? ([IPCM]\d+)\b", read(p), re.I):
            assert m.group(1).upper() in valid, f"{p.name}: unknown rule id {m.group(1)}"


def test_no_plan_files_tracked():
    files = tracked_files()
    assert len(files) > 100, f"only {len(files)} tracked files listed"
    plans = [f for f in files if PLAN_FILE_RE.search(f)]
    assert plans == [], f"plan files are tracked: {plans}"


def _code_files() -> list[str]:
    return source_files()


def test_grep_gates_scan_files_at_all():
    files = _code_files()
    assert len(files) > 20, f"only {len(files)} source files scanned"
    assert "pyzm/zm/auth.py" in files


def test_no_print_in_library_code():
    # AST, not grep: print( inside a docstring example is not a call.
    scanned, offenders = 0, []
    for rel in source_files():
        if not rel.startswith("pyzm/") or not rel.endswith(".py") or PRINT_EXEMPT(rel):
            continue
        scanned += 1
        tree = ast.parse(read(REPO / rel))
        offenders += [f"{rel}:{n.lineno}" for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"]
    assert scanned > 20, f"only {scanned} library files scanned"
    assert offenders == [], f"Logging: print() in library code: {offenders}"
