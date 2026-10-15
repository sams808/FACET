"""The reports-and-exports batch: report templates, a report over two
thresholds, figures in the HTML report, the Site tab as Markdown and LaTeX,
a FEFF input and a P1 CIF about an MD atom, the environment as extended
XYZ, the whole frame unwrapped, provenance as JSON beside every export, and
the site command line.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import dispose, sample_cif

from facet.core import bv, cif, coordination, md_model, md_readers, prefs, report
from facet.core import md_environment as env_mod
from facet.core.site_text import site_text

EXAMPLES = Path(__file__).resolve().parent.parent / "facet" / "data" / "examples"
QUARTZ = EXAMPLES / "quartz_SiO2_cod9013321.cif"
BIPO4 = EXAMPLES / "bismuth_phosphate_BiPO4_cod9008088.cif"
GLASS = Path(__file__).resolve().parent / "data" / "md" / "glass.extxyz"
SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


# -- the report ---------------------------------------------------------------------

def test_a_second_threshold_lists_the_sites_whose_cn_changes(tmp_path):
    rep = report.analyse_files([SAMPLE], v_bond=0.075, v_bond_other=0.03)
    tables = rep.tables()
    assert list(tables)[:2] == ["cn_changes", "sites_at_second"]
    second = tables["sites_at_second"]
    assert len(second) == 2 and all(r["v_bond_second"] == 0.03 for r in second)
    changes = tables["cn_changes"]
    assert changes and all(r["cn_at_first"] != r["cn_at_second"] for r in changes)
    assert any("second bond threshold" in line for line in rep.provenance())
    with pytest.raises(ValueError):
        report.analyse_files([SAMPLE], v_bond=0.075, v_list=0.02, v_bond_other=0.01)


@pytest.mark.parametrize("fmt", report.FORMATS)
def test_every_format_writes_a_provenance_json_beside_it(tmp_path, fmt):
    rep = report.analyse_files([QUARTZ])
    written = report.write_report(rep, tmp_path / "out", fmt)
    sidecar = [p for p in written if p.name.endswith(".provenance.json")]
    assert len(sidecar) == 1
    data = json.loads(sidecar[0].read_text(encoding="utf-8"))
    assert data["version"] and data["v_bond_vu"] == bv.V_BOND_DEFAULT
    assert data["files"][0]["name"] == QUARTZ.name


def test_a_template_limits_the_tables_and_the_cli_reads_it(tmp_path):
    fields = report.template_fields("md", 0.1, 0.02, False, None,
                                    tables=["sites", "cutoffs"])
    report.save_template("sites only", fields)
    assert "sites only" in report.template_names()
    assert report.load_template("sites only")["tables"] == ["sites", "cutoffs"]
    from facet.report.cli import main
    code = main([str(QUARTZ), "-o", str(tmp_path / "r"), "--template",
                 "sites only", "--quiet"])
    assert code == 0
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "## sites" in text and "## cutoffs" in text and "## contacts" not in text
    assert "v_bond = 0.1000" in text
    assert main([str(QUARTZ), "-o", str(tmp_path / "x"), "--template", "nothing"]) == 2


def test_the_html_report_takes_figures(qapp, tmp_path):
    from facet.ui.report_figures import add_figures
    rep = report.analyse_files([QUARTZ])
    written = report.write_report(rep, tmp_path / "out", "html")
    page = written[0]
    before = page.read_text(encoding="utf-8")
    assert "data:image/png;base64," not in before
    add_figures(page, rep)
    after = page.read_text(encoding="utf-8")
    assert after.count("data:image/png;base64,") >= 1
    assert "<h4>Si</h4>" in after


# -- the Site tab as text -----------------------------------------------------------

@pytest.fixture(scope="module")
def site():
    return coordination.analyse_structure(cif.read(str(QUARTZ)), bv.DEFAULT,
                                          v_bond=0.075, v_list=0.02,
                                          cations_only=True)[0]


def test_site_text_in_three_formats(site):
    txt = site_text(site, 0.075, fmt="txt", source="quartz")
    assert "Coordination number" in txt and "Plateaus" in txt and "quartz" in txt
    md = site_text(site, 0.075, fmt="md")
    assert md.startswith("## Si") and "|---|---:|" in md and "**Contacts**" in md
    tex = site_text(site, 0.075, fmt="tex")
    assert r"\begin{tabular}" in tex and r"\pm" in tex and "_" not in tex.split("%")[0]


def test_the_site_command_prints_and_refuses(capsys):
    from facet.site.cli import main
    assert main([str(QUARTZ)]) == 0
    out = capsys.readouterr().out
    assert "Coordination number" in out and "4.18" in out
    assert main([str(QUARTZ), "--site", "Nope"]) == 2
    assert "no site labelled" in capsys.readouterr().err
    assert main([str(BIPO4), "--all", "--format", "md"]) == 0
    assert capsys.readouterr().out.count("## ") == 2


def test_the_window_exports_the_site_tab_as_markdown_and_latex(qapp, tmp_path):
    prefs.set("tour_offered", True)
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        win.load_many([str(QUARTZ)])
        qapp.processEvents()
        entry = win.project.current
        entry.selected_site = win.project.results_for(entry)[0].site_index
        win._rebuild()
        md = win.export_site_text("md", tmp_path / "site.md")
        assert md.read_text(encoding="utf-8").startswith("## Si")
        tex = win.export_site_text("tex", tmp_path / "site.tex")
        assert r"\begin{tabular}" in tex.read_text(encoding="utf-8")
    finally:
        win.close()
        dispose(win)


# -- the MD environment exports ---------------------------------------------------------

@pytest.fixture
def small_frame():
    trajectory = md_readers.read_trajectory(str(GLASS))
    frame = trajectory.frame(0)
    from facet.core import bulk
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT, v_bond_vu=0.075,
                                  v_list_vu=0.02)
    return frame, table, ox


def test_extended_xyz_carries_the_box_and_the_rows(small_frame):
    frame, table, _ox = small_frame
    keep = np.ones(frame.n_atoms, bool)
    text = env_mod.environment_extxyz(frame, keep, 0, comment="all")
    lines = text.splitlines()
    assert lines[0] == str(frame.n_atoms)
    assert 'Lattice="' in lines[1] and "Properties=species:S:1:pos:R:3:row:I:1" in lines[1]
    assert lines[2].split()[4] == "0"           # the frame row, after xyz


def test_a_cluster_becomes_a_p1_structure_feff_and_cif(small_frame, tmp_path):
    frame, table, ox = small_frame
    from facet.core import exporters
    keep = np.ones(frame.n_atoms, bool)
    structure = env_mod.cluster_structure(frame, keep, 0, name="test", ox=ox)
    assert len(structure.sites) == frame.n_atoms
    assert structure.sites[0].label.startswith(str(frame.elements[0]))
    cart = np.array([a.cart for a in structure.atoms])
    assert (cart > 0).all()
    assert any("cut from an MD frame" in n for n in structure.notes)
    out = exporters.write_cif(structure, tmp_path / "cluster.cif")
    assert out.exists() and cif.read(str(out)).sites
    feff = exporters.write_feff(structure, 0, tmp_path / "feff.inp")
    text = feff.read_text(encoding="utf-8")
    assert "ATOMS" in text or "POTENTIALS" in text


def test_the_csv_export_writes_provenance_json(tmp_path):
    from facet.core import md_analysis as ma, md_export
    trajectory = md_readers.read_trajectory(str(GLASS))
    result = ma.analyse(trajectory, ma.AnalysisRequest(
        analyses=("glass",), formers=frozenset({"Si"})))
    written = md_export.write_csv_files(result, tmp_path / "csv")
    names = {p.name for p in written}
    assert "provenance.json" in names
    data = json.loads((tmp_path / "csv" / "provenance.json").read_text(encoding="utf-8"))
    assert data["lines"] and isinstance(data["rows"], list)
