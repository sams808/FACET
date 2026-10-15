"""The IUCr accumulated table ships with FACET: it reads, carries its
author's notice unchanged, names the paper behind each pair, is one click
away in the window, a preference at start, and a flag on the report
command line. And no window asks a question at start.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import dispose

from facet.core import bv, bv_files, prefs

EXAMPLES = Path(__file__).resolve().parent.parent / "facet" / "data" / "examples"
QUARTZ = EXAMPLES / "quartz_SiO2_cod9013321.cif"


def test_the_bundled_table_is_the_iucr_file_unchanged():
    path = bv_files.bundled_path()
    assert path.is_file() and path.name == "bvparm2020.cif"
    head = path.read_text(encoding="utf-8", errors="replace")[:3000]
    assert "COPYRIGHT NOTICE" in head and "I. David Brown" in head
    assert "data_BOND_VALENCE_PARAMETERS_2020-11-25" in head
    # carried as distributed: nothing of FACET's in the file's own header
    text = path.read_text(encoding="utf-8", errors="replace")
    assert "FACET" not in text.split("loop_")[0]


def test_the_bundled_table_reads_and_names_each_pairs_source():
    params = bv_files.bundled_compilation()
    assert params.name == bv_files.BUNDLED_NAME
    assert params.n_fitted > 1300
    si = params.get("Si", 4, "O")
    assert si.fitted and si.r0 == pytest.approx(1.624)
    assert "Brese and O'Keeffe" in si.source and "bvparm2020" in si.source
    # a pair whose first row is Brown & Altermatt's
    ag = params.get("Ag", 1, "O")
    assert ag.fitted and ag.r0 == pytest.approx(1.842)
    assert "Brown and Altermatt" in ag.source
    # a pair the table lacks falls to the estimator, labelled as such
    assert bv_files.bundled_compilation() is params           # read once
    assert bv_files.named_set("built-in") is bv.DEFAULT
    assert bv_files.named_set(bv_files.BUNDLED_KEY) is params
    with pytest.raises((OSError, ValueError)):
        bv_files.named_set("no-such-file.cif")


def test_the_bundled_table_is_in_the_menu_and_the_preference(qapp):
    from PySide6.QtGui import QAction
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        action = next(a for a in win.findChildren(QAction)
                      if "IUCr" in a.text())
        win.load(str(QUARTZ))
        before = win.project.params
        action.trigger()
        assert win.project.params.name == bv_files.BUNDLED_NAME
        assert win.project.params is not before
        assert win.results and win.results[0].cn_valence == 4
        reset = next(a for a in win.findChildren(QAction) if "built-in" in a.text())
        reset.trigger()
        assert win.project.params is bv.DEFAULT
    finally:
        win.close()
        dispose(win)
    prefs.set("parameter_set", "bvparm2020")
    again = preview.PreviewWindow()
    try:
        assert again.project.params.name == bv_files.BUNDLED_NAME
    finally:
        again.close()
        dispose(again)


def test_the_preferences_dialog_offers_the_set_and_no_tour(qapp):
    from facet.ui.preferences_dialog import PreferencesDialog
    dialog = PreferencesDialog()
    try:
        assert not hasattr(dialog, "offer_tour")
        dialog.parameter_set.setCurrentIndex(1)
        assert dialog.values()["parameter_set"] == "bvparm2020"
        dialog.parameter_set.setCurrentIndex(0)
        assert dialog.values()["parameter_set"] == "built-in"
    finally:
        dialog.deleteLater()


def test_the_report_command_takes_the_bundled_set(tmp_path):
    from facet.report.cli import main
    assert main([str(QUARTZ), "-o", str(tmp_path / "r"), "--format", "json",
                 "--params", "bvparm2020", "--quiet"]) == 0
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    assert bv_files.BUNDLED_NAME in text
    assert main([str(QUARTZ), "-o", str(tmp_path / "x"), "--params", "nope.cif"]) == 2


def test_the_build_bundles_the_table():
    text = (Path(__file__).resolve().parent.parent / "tools" / "build_exe.py"
            ).read_text(encoding="utf-8")
    assert "bvparm2020.cif" in text
