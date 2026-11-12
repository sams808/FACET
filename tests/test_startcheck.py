"""``--selfcheck``: a copy of FACET checks that it is whole (facet.ui.startcheck).

The continuous integration runs this in a pip-installed copy and in the
frozen application, the two places the test suite cannot see. Here: that it
passes in a checkout, that a check that does not hold is counted and turns
the exit code, and that the command itself exits with that code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_every_check_holds_in_a_checkout_and_the_report_says_what_ran(qapp, tmp_path, capsys):
    from facet.ui import startcheck
    from facet.version import __version__

    out = tmp_path / "report.json"
    assert startcheck.run(qapp, str(out)) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["holds"] is True
    assert [c["check"] for c in report["checks"]] == [name for name, _, _ in startcheck.CHECKS]
    assert all(c["holds"] for c in report["checks"])
    env = report["environment"]
    assert env["facet"] == __version__ and env["frozen"] is False
    assert env["qt_platform"] and env["python"] and env["numpy"]
    printed = capsys.readouterr().out
    assert f"{len(startcheck.CHECKS)} of {len(startcheck.CHECKS)} checks hold" in printed
    # the numbers of the manual were recomputed, not skipped
    numbers = next(c for c in report["checks"] if c["check"] == "reference numbers")
    done, total = (int(n) for n in numbers["detail"].split(" checks")[0].split(" of "))
    assert done == total and total >= 50


def test_a_check_that_does_not_hold_is_named_and_turns_the_exit_code(qapp, tmp_path, monkeypatch, capsys):
    from facet.ui import startcheck

    def missing_table():
        raise FileNotFoundError("bvparm2020.cif")

    monkeypatch.setattr(startcheck, "CHECKS", (
        ("bond-valence table", missing_table, False),
        ("bundled structures", startcheck._examples, False)))
    out = tmp_path / "report.json"
    assert startcheck.run(qapp, str(out)) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["holds"] is False
    first, second = report["checks"]
    assert first["holds"] is False and "bvparm2020.cif" in first["detail"]
    assert "FileNotFoundError" in first["traceback"]
    assert second["holds"] is True                 # the others still ran
    printed = capsys.readouterr().out
    assert "FAILS  bond-valence table" in printed and "1 of 2 checks hold" in printed


def test_the_file_after_out_is_found_wherever_it_stands():
    from facet.ui.startcheck import out_path

    assert out_path(["facet", "--selfcheck", "--out", "r.json"]) == "r.json"
    assert out_path(["facet", "--out", "r.json", "--selfcheck"]) == "r.json"
    assert out_path(["facet", "--selfcheck"]) is None
    assert out_path(["facet", "--selfcheck", "--out"]) is None


def test_the_command_exits_with_the_result_and_leaves_no_window(tmp_path):
    out = tmp_path / "report.json"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "facet", "--selfcheck", "--out", str(out)],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "checks hold" in done.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["holds"] is True
