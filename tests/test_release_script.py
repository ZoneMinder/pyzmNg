"""scripts/make_release.sh: what happens when the version's tag already exists.

Runs the real script hermetically: a throwaway git repo with a local bare
origin, stub gh / git-cliff on PATH, --skip-pypi and SKIP_E2E=1. Answers
are fed on stdin; the run stops at the next prompt when stdin runs out.
"""

import os
import shutil
import subprocess

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _stub(path, body):
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)


@pytest.fixture
def release(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _stub(stubs / "git-cliff", "exit 0")
    _stub(stubs / "gh", 'if [[ "$1 $2" == "auth token" ]]; then echo tok; fi; exit 0')

    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    shutil.copy(os.path.join(REPO, "scripts", "make_release.sh"), work / "scripts")
    (work / "pyzm").mkdir()
    (work / "pyzm" / "__init__.py").write_text('__version__ = "1.2.3"\n')

    env = {
        "PATH": "{}:/usr/bin:/bin".format(stubs),
        "HOME": str(tmp_path),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "SKIP_E2E": "1",
    }

    def git(*args, cwd=work):
        return subprocess.run(["git"] + list(args), cwd=cwd, env=env,
                              capture_output=True, text=True)

    git("init", "-q", "--bare", str(tmp_path / "origin.git"), cwd=tmp_path)
    git("init", "-q", "-b", "master")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    git("remote", "add", "origin", str(tmp_path / "origin.git"))
    git("push", "-q", "origin", "master")
    git("tag", "v1.2.3")

    def run(answers):
        return subprocess.run(["bash", "scripts/make_release.sh", "--skip-pypi"],
                              cwd=work, env=env, input=answers,
                              capture_output=True, text=True, timeout=60)

    def version():
        return (work / "pyzm" / "__init__.py").read_text()

    def tag_exists():
        return git("rev-parse", "-q", "--verify", "refs/tags/v1.2.3").returncode == 0

    return {"run": run, "version": version, "tag_exists": tag_exists}


def test_option_1_bumps_to_new_version(release):
    r = release["run"]("1\n")
    assert "1) Bump version: v1.2.3 -> v1.2.4" in r.stdout
    assert '__version__ = "1.2.4"' in release["version"]()
    assert release["tag_exists"](), "the published tag must be kept"


def test_option_2_overwrites(release):
    r = release["run"]("2\n")
    assert "2) Overwrite existing release v1.2.3" in r.stdout
    assert '__version__ = "1.2.3"' in release["version"]()
    assert not release["tag_exists"]()


def test_other_answer_aborts(release):
    r = release["run"]("x\n")
    assert "Aborted." in r.stdout
    assert '__version__ = "1.2.3"' in release["version"]()
    assert release["tag_exists"]()
