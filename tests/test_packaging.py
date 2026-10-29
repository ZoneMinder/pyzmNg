"""Install requirements declared in setup.py (#85)."""

from __future__ import annotations

import os
import runpy
import sys
import types
from unittest.mock import MagicMock, patch

_SETUP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "setup.py")


def _setup_kwargs() -> dict:
    # A stand-in setuptools: Python 3.12+ venvs don't ship the real one.
    fake = types.ModuleType("setuptools")
    fake.setup = MagicMock()
    fake.find_packages = MagicMock(return_value=[])
    with patch.dict(sys.modules, {"setuptools": fake}):
        runpy.run_path(_SETUP)
    return fake.setup.call_args.kwargs


def _names(reqs: list[str]) -> set[str]:
    return {r.split("[")[0].split(">")[0].split("=")[0].strip().lower() for r in reqs}


def test_base_install_has_no_mysql_driver():
    assert not _names(_setup_kwargs()["install_requires"]) & {"mysql-connector-python", "pymysql"}


def test_base_install_is_requests_and_pydantic():
    assert _names(_setup_kwargs()["install_requires"]) == {"requests", "pydantic"}


def test_dates_extra_installs_dateparser_and_full_includes_it():
    extras = _setup_kwargs()["extras_require"]
    assert _names(extras["dates"]) == {"dateparser"}
    assert "dateparser" in _names(extras["full"])


def test_license_classifier_names_gpl_v2():
    assert "License :: OSI Approved :: GNU General Public License v2 (GPLv2)" in \
        _setup_kwargs()["classifiers"]


def test_db_extra_installs_pymysql():
    assert _names(_setup_kwargs()["extras_require"]["db"]) == {"pymysql"}


def test_on_box_extras_include_db_driver():
    """ES depends on pyzm[ml]; tagging and DB logging must keep working there."""
    extras = _setup_kwargs()["extras_require"]
    for name in ("ml", "full"):
        assert "pymysql" in _names(extras[name]), name


def test_remote_extras_leave_db_driver_out():
    """serve and train run on boxes that never touch the ZM database."""
    extras = _setup_kwargs()["extras_require"]
    for name in ("serve", "train"):
        assert "pymysql" not in _names(extras[name]), name


def _doc_extras_table() -> dict[str, set[str]]:
    """{extra: packages} from the 'What each extra installs' table."""
    import re
    path = os.path.join(os.path.dirname(_SETUP), "docs", "guide", "installation.rst")
    text = open(path).read().split("What each extra installs", 1)[1]
    rows, current, cell = {}, None, 0
    for line in text.splitlines():
        start = re.match(r"\s*\* - (.*)", line)
        item = re.match(r"\s{5}- (.*)", line)
        if start:
            current, cell = start.group(1).strip("`*[] ").replace("(base)", "base"), 0
        elif item and current:
            cell += 1
            if cell == 1:
                rows[current] = item.group(1)
        elif current and cell == 1 and line.startswith("       ") and line.strip():
            rows[current] += " " + line.strip()
        if line.startswith("Some packages aren't"):
            break
    return {k: {p.strip().lower() for p in v.split(",")} for k, v in rows.items() if k != "Extra"}


def test_installation_doc_lists_each_extras_packages():
    """The docs table matches what pip installs for every extra but full."""
    kwargs = _setup_kwargs()
    table = _doc_extras_table()
    assert table.pop("base") == _names(kwargs["install_requires"])
    assert table.pop("full") == {"everything in the rows above"}
    assert table == {name: _names(reqs) for name, reqs in kwargs["extras_require"].items()
                     if name != "full"}


def test_setup_reads_its_files_without_codecs_open():
    """Python 3.14 deprecates codecs.open(); with warnings as errors every test here failed on it."""
    with patch("codecs.open", side_effect=AssertionError("setup.py called codecs.open()")):
        assert _setup_kwargs()["version"]


def test_full_is_the_union_of_the_other_extras():
    extras = {k: _names(v) for k, v in _setup_kwargs()["extras_require"].items()}
    full = extras.pop("full")
    assert full == set().union(*extras.values())
