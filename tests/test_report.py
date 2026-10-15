"""Reports on a set of CIFs, in six formats, from the core, the command line
and the window.

The numbers in a report are the window's numbers: a site row carries what
``coordination.analyse_structure`` found at the same thresholds, and every
format writes the same five tables with the same provenance. A file that
cannot be read stays in the report with its reason.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from conftest import dispose

from facet.core import bv, cif, coordination, report

EXAMPLES = Path(__file__).resolve().parent.parent / "facet" / "data" / "examples"
QUARTZ = EXAMPLES / "quartz_SiO2_cod9013321.cif"
BIPO4 = EXAMPLES / "bismuth_phosphate_BiPO4_cod9008088.cif"


@pytest.fixture(scope="module")
def two_files():
    return report.analyse_files([QUARTZ, BIPO4])


# -- collecting ----------------------------------------------------------------

def test_a_folder_contributes_every_cif_once():
    files = report.collect_cifs([EXAMPLES, QUARTZ])
    assert all(f.suffix == ".cif" for f in files)
    assert len(files) == len(set(str(f).lower() for f in files))
    assert QUARTZ in files
    assert len(files) == len(list(EXAMPLES.glob("*.cif")))


# -- the numbers are the window's numbers -------------------------------------

def test_site_rows_match_a_direct_analysis(two_files):
    rows = [r for r in two_files.site_rows() if r["structure"] == QUARTZ.name]
    direct = coordination.analyse_structure(cif.read(str(QUARTZ)), bv.DEFAULT,
                                            v_bond=bv.V_BOND_DEFAULT,
                                            v_list=bv.V_LIST_DEFAULT,
                                            cations_only=True)
    assert len(rows) == len(direct) == 1
    assert rows[0]["cn"] == direct[0].cn_valence == 4
    assert abs(rows[0]["bvs"] - direct[0].bvs) < 1e-12
    assert abs(rows[0]["phi"] - direct[0].phi) < 1e-12
    assert rows[0]["plateau_cn"] == 4
    assert rows[0]["plateau_decades"] == pytest.approx(
        direct[0].current_plateau.width_decades)


def test_contact_rows_carry_a_running_sum_that_ends_at_the_listed_sum(two_files):
    rows = [r for r in two_files.contact_rows()
            if r["structure"] == BIPO4.name and r["site"] == "Bi"]
    assert rows == sorted(rows, key=lambda r: r["distance"])
    assert sum(r["is_bond"] for r in rows) == 8
    site = next(r for r in two_files.site_rows()
                if r["structure"] == BIPO4.name and r["site"] == "Bi")
    assert rows[-1]["running_sum"] == pytest.approx(site["bvs_listed"])


def test_plateau_rows_are_the_staircase(two_files):
    rows = [r for r in two_files.plateau_rows()
            if r["structure"] == BIPO4.name and r["site"] == "Bi"]
    assert [r["cn"] for r in rows] == list(range(1, len(rows) + 1))
    assert sum(r["holds_v_bond"] for r in rows) == 1
    held = next(r for r in rows if r["holds_v_bond"])
    assert held["cn"] == 8


def test_cutoff_rows_give_the_distance_each_threshold_means(two_files):
    rows = [r for r in two_files.cutoff_rows() if r["structure"] == BIPO4.name]
    pairs = {r["pair"] for r in rows}
    assert "Bi3+–O" in pairs and "P5+–O" in pairs
    bi = next(r for r in rows if r["pair"] == "Bi3+–O")
    assert bi["d_bond"] < bi["d_list"]
    assert bi["window"] == pytest.approx(bi["d_list"] - bi["d_bond"])
    assert bi["n_bonds"] == 8


def test_a_file_that_cannot_be_read_is_reported_not_dropped(tmp_path):
    bad = tmp_path / "broken.cif"
    bad.write_text("this is not a cif", encoding="utf-8")
    rep = report.analyse_files([QUARTZ, bad])
    assert len(rep.structures) == 2
    broken = rep.structures[1]
    assert broken.error and broken.results == []
    assert any("not read" in line for line in rep.provenance())
    balance = {r["structure"]: r for r in rep.balance_rows()}
    assert balance["broken.cif"]["read"] is False
    assert balance[QUARTZ.name]["read"] is True


def test_a_bond_threshold_below_the_listing_threshold_is_refused():
    with pytest.raises(ValueError):
        report.analyse_files([QUARTZ], v_bond=0.01, v_list=0.02)


def test_progress_can_stop_the_run():
    seen = []

    def progress(done, total, name):
        seen.append(name)
        return done < 1

    rep = report.analyse_files([QUARTZ, BIPO4], progress=progress)
    assert len(rep.structures) == 1 and len(seen) == 2


# -- the six formats ----------------------------------------------------------

@pytest.mark.parametrize("fmt", report.FORMATS)
def test_every_format_writes_the_same_tables(two_files, tmp_path, fmt):
    written = report.write_report(two_files, tmp_path / "out", fmt)
    assert written and all(p.exists() and p.stat().st_size > 0 for p in written)
    written = [p for p in written if not p.name.endswith(".provenance.json")]
    text = "".join(p.read_text(encoding="utf-8", errors="replace")
                   for p in written if p.suffix != ".xlsx")
    if fmt == "csv":
        assert {p.name for p in written} == {
            f"out_{t}.csv" for t in report.TABLES}
        with open(written[0], encoding="utf-8") as handle:
            lines = [ln for ln in handle if not ln.startswith("#")]
        assert len(list(csv.reader(lines))) > 1
    elif fmt == "xlsx":
        openpyxl = pytest.importorskip("openpyxl")
        book = openpyxl.load_workbook(written[0])
        assert set(report.TABLES) <= set(book.sheetnames)
    elif fmt == "json":
        data = json.loads(text)
        assert set(data["tables"]) == set(report.TABLES)
        assert data["v_bond"] == bv.V_BOND_DEFAULT
    else:
        for name in report.TABLES:
            assert name in text
        assert "v_bond = 0.0750" in text
        assert "quartz_SiO2" in text


def test_an_unknown_format_is_refused(two_files, tmp_path):
    with pytest.raises(ValueError):
        report.write_report(two_files, tmp_path / "out", "docx")


# -- the command line -----------------------------------------------------------

def test_the_cli_writes_a_report_and_imports_no_qt(tmp_path, monkeypatch):
    import subprocess
    import sys
    out = tmp_path / "r"
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from facet.report.cli import main; "
         "code = main(sys.argv[1:]); "
         "assert not any(m.startswith('PySide6') for m in sys.modules), 'Qt imported'; "
         "sys.exit(code)",
         str(QUARTZ), str(BIPO4), "-o", str(out), "--format", "csv", "--quiet"],
        capture_output=True, text=True, cwd=str(EXAMPLES.parents[2]))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "r_sites.csv").exists()


def test_the_cli_refuses_bad_thresholds_and_missing_files(tmp_path):
    from facet.report.cli import main
    assert main([str(QUARTZ), "-o", str(tmp_path / "x"), "--v-bond", "0.01",
                 "--v-list", "0.02"]) == 2
    assert main([str(tmp_path / "nothing.cif"), "-o", str(tmp_path / "x")]) == 2


def test_the_cli_reports_a_broken_file_with_exit_code_one(tmp_path):
    from facet.report.cli import main
    bad = tmp_path / "broken.cif"
    bad.write_text("nope", encoding="utf-8")
    code = main([str(QUARTZ), str(bad), "-o", str(tmp_path / "r"),
                 "--format", "json", "--quiet"])
    assert code == 1
    assert (tmp_path / "r.json").exists()


# -- the window ----------------------------------------------------------------

def test_the_dialog_takes_the_windows_thresholds_and_the_open_files(qapp):
    from facet.ui.report_dialog import ReportDialog
    d = ReportDialog(v_bond=0.1, v_list=0.03, paths=[str(QUARTZ), str(EXAMPLES)])
    try:
        assert d.v_bond.value() == pytest.approx(0.1)
        assert d.v_list.value() == pytest.approx(0.03)
        assert d.count.text().startswith(str(len(list(EXAMPLES.glob("*.cif")))))
        assert d.run.isEnabled()
        d.add_paths([str(QUARTZ)])          # a duplicate changes nothing
        assert d.files.count() == 2
        assert [d.format.itemData(i) for i in range(d.format.count())] == list(report.FORMATS)
    finally:
        dispose(d)


def test_the_file_menu_holds_the_report_item_and_the_explorer_defaults(qapp):
    from facet.ui import preview
    from facet.ui.cutoff_explorer import AXIS_DISTANCE
    win = preview.PreviewWindow()
    try:
        # the bar action is held while its menu is read: PySide6 drops the
        # menu's wrapper with the action's
        bar = next(a for a in win.menuBar().actions()
                   if a.text().replace("&", "") == "File")
        export = next(a for a in bar.menu().actions()
                      if a.text().replace("&", "") == "Export")
        texts = [a.text().replace("&", "") for a in export.menu().actions()]
        assert texts[-1] == "Report on CIFs…"
        assert win.report_action in export.menu().actions()
        ex = win.explorer
        assert ex.axis == AXIS_DISTANCE and ex.stacked and ex.show_bars \
            and ex.fill_sum and not ex.sum_as_fraction
        assert win.explorer_axis_actions[AXIS_DISTANCE].isChecked()
        assert win.running_sum_column.isChecked()
    finally:
        win.close()
        dispose(win)
