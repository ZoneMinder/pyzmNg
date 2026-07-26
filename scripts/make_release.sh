#!/bin/bash
set -e

SKIP_PYPI=false
for arg in "$@"; do
    case "$arg" in
        --skip-pypi) SKIP_PYPI=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

GH_REPO="ZoneMinder/pyzmNg"

# --- Read version from pyzm/__init__.py ---
INIT_FILE="pyzm/__init__.py"
if [ ! -f "$INIT_FILE" ]; then
    echo "ERROR: $INIT_FILE not found"
    exit 1
fi
VER=$(grep -Po '(?<=^__version__ = ["\x27])[^"\x27]+' "$INIT_FILE")
if [ -z "$VER" ]; then
    echo "ERROR: could not parse __version__ from $INIT_FILE"
    exit 1
fi

echo "=== Release v${VER} ==="
echo

# --- Preflight checks ---
for cmd in git-cliff gh; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found."
        exit 1
    fi
done
if [ "$SKIP_PYPI" = false ]; then
    if ! python3 -m build --help &>/dev/null; then
        echo "ERROR: python3 -m build not available. Install with: pip install build"
        exit 1
    fi
    if ! command -v twine &>/dev/null; then
        echo "ERROR: twine not found. Install with: pip install twine"
        exit 1
    fi
fi
export GITHUB_TOKEN=$(gh auth token)

# --- Cross-repo (ES + pyzm) e2e: must pass before releasing ---
# Runs the ES hook chain against THIS pyzm checkout on real models, plus the
# pyzm<->ES contract test. Aborts the release on any failure.
# Overrides: ES_DIR=/path (ES checkout), SKIP_E2E=1 (emergency bypass).
if [ "${SKIP_E2E:-}" = "1" ]; then
    echo "WARNING: SKIP_E2E=1 -- skipping validation gates"
else
    # 1. pyzm's own Tier-1 suite (the cross-repo e2e below does NOT run it).
    echo "Running pyzm Tier-1 gate ..."
    if ! make gate; then
        echo "ERROR: pyzm gate FAILED -- aborting release"
        exit 1
    fi
    # 1b. Real-model local<->remote parity (needs models, not live ZM).
    # PYZM_E2E_REQUIRE=1 => missing models FAIL instead of silently skipping,
    # so a green release proves remote detection matches local on a real model.
    echo "Running real-model remote parity e2e ..."
    if ! PYZM_E2E_REQUIRE=1 python3 -m pytest tests/test_ml_e2e/test_remote_serve.py -m serve -v; then
        echo "ERROR: remote parity e2e FAILED -- aborting release"
        exit 1
    fi
    # 2. Cross-repo e2e (ES hook chain vs this pyzm + contract test).
    ES_DIR="${ES_DIR:-$(cd "$REPO_DIR/../zmeventnotificationNg" 2>/dev/null && pwd || echo "$REPO_DIR/../zmeventnotificationNg")}"
    if [ ! -d "$ES_DIR" ]; then
        echo "ERROR: ES repo not found at $ES_DIR (set ES_DIR=... or SKIP_E2E=1)"
        exit 1
    fi
    echo "Running cross-repo e2e (ES hook chain vs this pyzm) ..."
    if ! ( cd "$ES_DIR" && make release-gate PYZM_SRC="$REPO_DIR" ); then
        echo "ERROR: cross-repo e2e FAILED -- aborting release"
        exit 1
    fi
    echo "Cross-repo e2e passed."
    echo
fi

# --- Step 1: Check if tag already exists ---
if git rev-parse "v${VER}" &>/dev/null; then
    # Compute bumped patch version
    IFS='.' read -r V_MAJOR V_MINOR V_PATCH <<< "$VER"
    BUMPED_PATCH=$((V_PATCH + 1))
    BUMPED_VER="${V_MAJOR}.${V_MINOR}.${BUMPED_PATCH}"

    echo "Tag v${VER} already exists."
    echo "  1) Overwrite existing release v${VER}"
    echo "  2) Bump version: v${VER} -> v${BUMPED_VER}"
    read -p "Choose [1/2] (or anything else to abort): " choice
    case "$choice" in
        1)
            echo "  Deleting old release and tag v${VER} ..."
            gh release delete "v${VER}" --repo "$GH_REPO" --yes 2>/dev/null || true
            git tag -d "v${VER}"
            git push --no-verify origin --delete "v${VER}" 2>/dev/null || true
            ;;
        2)
            echo "  Bumping version: v${VER} -> v${BUMPED_VER} ..."
            sed -i "s/^__version__ = [\"']${VER}[\"']/__version__ = \"${BUMPED_VER}\"/" "$INIT_FILE"
            git add "$INIT_FILE"
            git commit -m "chore: bump version to v${BUMPED_VER}"
            git push --no-verify origin "$(git rev-parse --abbrev-ref HEAD)"
            VER="$BUMPED_VER"
            echo "  Done. Continuing with v${VER}."
            ;;
        *)
            echo "Aborted."
            exit 0
            ;;
    esac
    echo
fi

# --- Step 2: Check for uncommitted files ---
DIRTY_FILES=$(git status --porcelain)
if [ -n "$DIRTY_FILES" ]; then
    # Allow only pyzm/__init__.py (version bump) to be dirty
    NON_INIT=$(echo "$DIRTY_FILES" | grep -v " ${INIT_FILE}$" || true)
    if [ -n "$NON_INIT" ]; then
        echo "ERROR: Uncommitted files besides ${INIT_FILE}:"
        echo "$NON_INIT"
        exit 1
    fi
    echo "Committing ${INIT_FILE} (version bump) ..."
    git add "$INIT_FILE"
    git commit -m "chore: bump version to v${VER}"
    git push --no-verify origin master
    echo "  Done."
    echo
fi

# --- Confirm before proceeding ---
BRANCH=$(git rev-parse --abbrev-ref HEAD)
REMOTE_URL=$(git remote get-url origin)
echo "--- Release summary ---"
echo "  Version:      v${VER}"
echo "  Branch:       ${BRANCH}"
echo "  Remote:       ${REMOTE_URL}"
echo "  GitHub repo:  ${GH_REPO}"
echo "  PyPI upload:  $([ "$SKIP_PYPI" = true ] && echo "SKIPPED" || echo "yes")"
echo
if [ "$SKIP_PYPI" = true ]; then
    echo "This will: generate CHANGELOG, commit, tag, push, and create GitHub release (PyPI skipped)."
else
    echo "This will: generate CHANGELOG, commit, tag, push, build & upload to PyPI, and create GitHub release."
fi
read -p "Proceed? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi
echo

# --- Step 3: Generate and commit changelog ---
echo "Generating CHANGELOG.md ..."
git-cliff --tag "v${VER}" -o CHANGELOG.md
echo "  Done."

echo "Committing CHANGELOG.md ..."
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for v${VER}"
git push --no-verify origin master
echo "  Done."
echo

# --- Step 4: Tag ---
echo "Creating tag v${VER} ..."
git tag -a "v${VER}" -m "v${VER}"
git push --no-verify origin --tags
echo "  Done."
echo

# --- Step 5: Build and upload to PyPI ---
if [ "$SKIP_PYPI" = false ]; then
    echo "Building PyPI packages ..."
    rm -rf dist
    python3 -m build
    echo "  Done."

    echo "Uploading to PyPI ..."
    twine upload dist/pyzm-"${VER}"* --verbose
    echo "  Done."
    echo
else
    echo "Skipping PyPI build & upload (--skip-pypi)."
    echo
fi

# --- Step 6: Create GitHub Release ---
echo "Creating GitHub Release v${VER} ..."
NOTES_FILE=$(mktemp)
git-cliff --latest --strip header > "$NOTES_FILE" 2>/dev/null
gh release create "v${VER}" --repo "$GH_REPO" --title "v${VER}" --notes-file "$NOTES_FILE"
rm -f "$NOTES_FILE"
echo "  Done."

echo
echo "=== Release v${VER} complete ==="
