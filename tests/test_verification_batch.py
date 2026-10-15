"""The verification batch: the reference numbers of the bundled examples
recomputed and matched to what the tutorials state, the self-check
command, the parity harness against another code's table, the "est." mark
on the explorer and in the report, the oxidation-state source in the
report, and the build record.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from facet.core import bv, cif, coordination, examples, md_readers, selfcheck
from facet.core import md_parity

ROOT = Path(__file__).resolve().parent.parent
GLASS = Path(__file__).resolve().parent / "data" / "md" / "glass.extxyz"


# -- the reference file -----------------------------------------------------------------

def test_every_bundled_example_is_in_the_reference_and_recomputes():
    reference = selfcheck.load_reference()
    names = {ex.path.name for ex in examples.available()}
    assert set(reference["examples"]) == names
    rows = selfcheck.check_examples(reference)
    failed = [r for r in rows if not r.agrees]
    assert not failed, selfcheck.format_rows(failed)
    # every quantity of every site was checked
    quantities = {r.quantity for r in rows}
    assert quantities == set(selfcheck.TOLERANCES)
    sums = selfcheck.check_sums(reference)
    assert sums and all(r.agrees for r in sums)


def test_the_tutorials_state_the_reference_numbers():
    """The numbers the tutorials quote are read from the reference file, so a
    changed engine or a changed tutorial fails here rather than in a
    reader's hands."""
    from facet.ui import help as help_mod
    text = re.sub(r"<[^>]+>", "", help_mod.manual_html())
    text = text.replace("&Aring;", "Å").replace("&phi;", "phi").replace("&mdash;", "—")
    ref = selfcheck.load_reference()["examples"]
    quartz = ref["quartz_SiO2_cod9013321.cif"]["sites"]["Si"]
    assert f"near {quartz['bvs_vu']:.2f}" in text
    assert f"{quartz['plateau_decades']:.2f} decades" in text
    assert f"up to {quartz['plateau_to_vu']:.3f}" in text
    eul = ref["eulytite_Bi4SiO4_3_cod9012894.cif"]["sites"]
    assert f"{eul['Bi']['phi']:.3f}" in text
    assert f"{eul['Bi']['plateau_to_vu']:.3f}" in text
    assert f"{eul['Si']['plateau_decades']:.2f} decades" in text
    assert f"{eul['Bi']['bond_distances_ang'][0]:.3f}" in text
    assert f"{eul['Bi']['bond_distances_ang'][-1]:.3f}" in text
    sen = ref["senarmontite_Sb2O3_cod9009747.cif"]["sites"]["Sb"]
    val = ref["valentinite_Sb2O3_cod9007587.cif"]["sites"]["Sb"]
    assert f"{sen['plateau_decades']:.2f} decades" in text
    assert f"{val['plateau_decades']:.2f} decades" in text
    assert f"{sen['bvs_vu']:.2f} and {val['bvs_vu']:.2f}" in text
    cry = ref["cryolite_Na3AlF6_cod9004097.cif"]["sites"]
    assert f"{cry['Al']['plateau_decades']:.2f} decades" in text
    assert f"{cry['Na2']['plateau_from_vu']:.4f} to {cry['Na2']['plateau_to_vu']:.4f}" in text
    bipo = ref["bismuth_phosphate_BiPO4_cod9008088.cif"]["sites"]["Bi"]
    assert f"{bipo['phi']:.3f}" in text
    assert f"{bipo['bvs_vu']:.2f} against {eul['Bi']['bvs_vu']:.2f}" in text


def test_the_self_check_reports_a_changed_reference(tmp_path, capsys):
    from facet.check.cli import main
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "within tolerance" in out and " NO" not in out
    ref = selfcheck.load_reference()
    ref["examples"]["quartz_SiO2_cod9013321.cif"]["sites"]["Si"]["cn"] = 5
    bad = tmp_path / "ref.json"
    bad.write_text(json.dumps(ref), encoding="utf-8")
    assert main(["--reference", str(bad), "--failures", "--json",
                 str(tmp_path / "rows.json")]) == 1
    out = capsys.readouterr().out
    assert "quartz" in out and " NO" in out
    rows = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    assert any(r["quantity"] == "cn" and not r["agrees"] for r in rows)
    # a missing example is a row, not a crash
    assert selfcheck.check_examples(ref, folder=tmp_path)[0].computed == "missing"


# -- the parity harness -----------------------------------------------------------------

def test_the_parity_harness_agrees_with_facet_on_its_own_numbers(tmp_path, capsys):
    trajectory = md_readers.read_trajectory(str(GLASS))
    results, frame = md_parity.facet_values(trajectory, 0)
    table = tmp_path / "theirs.csv"
    lines = ["id,cn,sum,other"]
    for i in range(frame.n_atoms):
        bvs = results.bvs_vu[i]
        lines.append(f"{i + 1},{results.cn[i]},{bvs + 0.01 if i % 2 else bvs:.6f},x")
    table.write_text("\n".join(lines), encoding="utf-8")
    rows = md_parity.compare_files(table, GLASS, 0, "cn=CN,sum=BVS",
                                   atom_column="id", one_based=True, tolerance=1e-3)
    by = {r.quantity: r for r in rows}
    assert by["CN"].n_compared == frame.n_atoms
    assert by["CN"].max_abs_difference == 0 and by["CN"].fraction_within == 1.0
    finite = int(np.isfinite(results.bvs_vu).sum())
    assert by["BVS"].n_compared == finite
    assert by["BVS"].max_abs_difference == pytest.approx(0.01, abs=1e-6)
    assert 0.4 < by["BVS"].fraction_within < 0.6
    assert by["BVS"].mean_difference > 0           # theirs minus FACET
    with pytest.raises(ValueError):
        md_parity.parse_mapping("cn=NOPE")
    with pytest.raises(ValueError):
        md_parity.compare_files(table, GLASS, 0, "cn=CN", atom_column="nope")
    from facet.check.cli import main
    assert main(["--parity", str(table), str(GLASS), "--map", "cn=CN,sum=BVS",
                 "--atom-column", "id", "--one-based"]) == 0
    out = capsys.readouterr().out
    assert "theirs minus FACET" in out and "100.0%" in out
    assert main(["--parity", str(table), str(tmp_path / "none.xyz")]) == 2


# -- marks and sources ------------------------------------------------------------------

@pytest.fixture
def estimated_site():
    """Quartz Si under a set with no fitted pair: every parameter estimated."""
    structure = cif.read(str(examples.folder() / "quartz_SiO2_cod9013321.cif"))
    params = bv.ParameterSet(name="estimates only", fitted={}, allow_estimated=True)
    return coordination.analyse_structure(structure, params, v_bond=0.075,
                                          v_list=0.02)[0]


def test_the_explorer_marks_estimated_parameters(qapp, estimated_site):
    from PySide6.QtGui import QImage
    from facet.ui.cutoff_explorer import CutoffExplorer
    ex = CutoffExplorer()
    ex.resize(640, 320)
    ex.set_result(estimated_site)
    ex.set_threshold(0.075)
    marked = ex.estimated_contacts()
    assert marked and marked == {i for i, c in enumerate(estimated_site.contacts)
                                 if c.has_valence}
    image = QImage(640, 320, QImage.Format_ARGB32)
    image.fill(0xFFFFFFFF)
    ex.render(image)
    structure = cif.read(str(examples.folder() / "quartz_SiO2_cod9013321.cif"))
    fitted = coordination.analyse_structure(structure, bv.DEFAULT, v_bond=0.075,
                                            v_list=0.02)[0]
    ex.set_result(fitted)
    assert ex.estimated_contacts() == set()
    ex.deleteLater()


def test_the_report_carries_the_mark_the_source_row_and_the_oxidation_source(tmp_path):
    from facet.core import report
    rep = report.analyse_files([examples.folder() / "quartz_SiO2_cod9013321.cif"])
    tables = rep.tables()
    contact = tables["contacts"][0]
    assert contact["estimated"] == "" and contact["parameter_fitted"] is True
    assert contact["parameter_row"] == "Si+4-O" and contact["parameter_source"]
    site = tables["sites"][0]
    assert site["oxidation_source"] and site["estimated_parameters"] is False
    written = report.write_report(rep, tmp_path / "r", "md")
    text = written[0].read_text(encoding="utf-8")
    assert "oxidation_source" in text and "parameter_row" in text


def test_the_build_record_lists_the_packages(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_exe", ROOT / "tools" / "build_exe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.write_build_record(tmp_path)
    text = out.read_text(encoding="utf-8")
    from facet.version import __version__
    assert text.startswith(f"FACET {__version__} -- build record")
    assert "commit " in text and "python " in text
    names = {ln.split("==")[0].lower() for ln in text.splitlines() if "==" in ln}
    assert {"numpy", "pyside6"} <= names
