#!/usr/bin/env python3
"""Outbound HTTP contract (AGENTS.project.md): count requests calls in pyzm/
that pass no timeout=. A hung server then hangs detection for the event.

    python3 scripts/gates/count_http_without_timeout.py           list them
    python3 scripts/gates/count_http_without_timeout.py --count   number only

Matches requests.get/post/put/delete/head/request(...) and the same on any
name bound by `import requests as <name>`. Exits 2 when it scanned too few
files to have measured anything (M2).
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VERBS = {"get", "post", "put", "delete", "head", "patch", "request"}


def offenders():
    found, scanned = [], 0
    for path in sorted((REPO / "pyzm").rglob("*.py")):
        scanned += 1
        tree = ast.parse(path.read_text())
        names = {"requests"} | {a.asname for n in ast.walk(tree) if isinstance(n, ast.Import)
                                for a in n.names if a.name == "requests" and a.asname}
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in VERBS and isinstance(n.func.value, ast.Name)
                    and n.func.value.id in names
                    and not any(k.arg == "timeout" for k in n.keywords)):
                found.append(f"{path.relative_to(REPO)}:{n.lineno}")
    if scanned < 20:
        print(f"count_http_without_timeout: only {scanned} files scanned", file=sys.stderr)
        sys.exit(2)
    return found


if __name__ == "__main__":
    found = offenders()
    print(len(found) if sys.argv[1:] == ["--count"] else ("\n".join(found) or "none"))
