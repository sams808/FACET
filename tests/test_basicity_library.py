"""2.0.0: two corrections to the microscopic basicity, the quantities added
beside it, and the library and command line that give all of it without
the window.

Corrections, each against something outside the code:

* a fluoride ion's lambda is the oxide expression divided by 2.3, the share
  its contacts do not carry included (Duffy 1989: "the optical basicities
  for fluorides are obtained by dividing those for oxide systems by 2.3");
  before, only the bonded part was divided, so an under-bonded fluoride ion
  read too basic;
* "host only" leaves out this atom's own bond, not every bond of its site
  (pinned on quartz in test_basicity.py).

Added: the probe-ion bands and the oxide-ion polarisability (the printed
equations), a composition written as on a batch sheet (Duffy's 1996 Table 2
and the species route), lambda by linkage, the spatial correlation of
lambda (against a direct double loop), the cube file (read back).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from facet import basicity as fb
from facet.core import basicity as B
from facet.core import basicity_composition as C
from facet.core import basicity_field as F
from facet.core import bulk, bv, md_model, md_site, readers

EXAMPLES = Path(__file__).resolve().parent.parent / "facet" / "data" / "examples"
QUARTZ = EXAMPLES / "quartz_SiO2_cod9013321.cif"
CRYOLITE = EXAMPLES / "cryolite_Na3AlF6_cod9004097.cif"
BIPO4 = EXAMPLES / "bismuth_phosphate_BiPO4_cod9008088.cif"


def _anion(element, ox, contacts, v_bond=0.075):
    rows = [SimpleNamespace(label=f"{e}{k}", element=e, site_index=k,
                            atom_index=k, distance=2.0 + 0.01 * k,
                            occupancy=1.0, valence=s, has_valence=True,
                            param=SimpleNamespace(cation_ox=z))
            for k, (e, z, s) in enumerate(contacts)]
    return SimpleNamespace(label=f"{element}1", element=element, ox=ox,
                           site_index=99, contacts=rows, v_bond=v_bond)


# -- the fluoride ion ------------------------------------------------------------------

def test_a_fluoride_ion_is_the_oxide_expression_over_2_3_whatever_its_sum():
    gamma_na = B.gamma_from_electronegativity(0.9)
    # exactly bonded: NaF, 0.50 in Duffy's 1989 Table 1
    full = B.micro_basicity(_anion("F", -1, [("Na", 1, 1 / 6)] * 6))
    assert full.value == pytest.approx(1 / (2.3 * gamma_na), abs=1e-12)
    assert full.value == pytest.approx(0.50, abs=0.005)
    # under-bonded by a quarter: the missing share counts at gamma = 1 on
    # the fluoride scale (1 / 2.3), not as a free oxide ion (1)
    under = B.micro_basicity(_anion("F", -1, [("Na", 1, 0.125)] * 6))
    assert under.value == pytest.approx(
        (1 - 0.75 * (1 - 1 / gamma_na)) / 2.3, abs=1e-12)
    assert under.value < 0.52                       # it read 0.62 before
    # calcium is the scale's gamma = 1: bonded to it or to nothing, the same
    # (on Duffy's table, where gamma(Ca) is 1.00 exactly)
    assert B.micro_basicity(_anion("F", -1, [("Ca", 2, 0.2)] * 4),
                            "duffy").value == pytest.approx(1 / 2.3)
    assert B.micro_basicity(_anion("F", -1, [])).value == pytest.approx(1 / 2.3)
    # and an oxide ion is unchanged by the correction
    o = B.micro_basicity(_anion("O", -2, [("Si", 4, 1.0)]))
    assert o.value == pytest.approx(1 - 0.5 * (1 - 1 / B.gamma_from_electronegativity(1.8)))


def test_a_fluoride_frame_gives_the_crystal_paths_lambda_atom_for_atom():
    structure = readers.read(str(CRYOLITE))
    frame, _ = md_model.supercell_frame(structure, (2, 2, 2))
    ox = md_model.model_oxidation(frame.species)
    ox_atom = ox.per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox_atom, bv.DEFAULT, v_list_vu=0.03)
    found = B.frame_basicity(table, "pauling", v_bond=0.075)
    anions = np.flatnonzero(np.asarray(table.is_anion))
    assert np.isfinite(found.value[anions]).all()
    for row in anions[:6]:
        site = md_site.site_result_for_atom(frame, table, int(row), v_bond=0.075)
        micro = B.micro_basicity(site, "pauling", v_bond=0.075)
        assert found.value[row] == pytest.approx(micro.value, abs=1e-12)
        assert found.value_listed[row] == pytest.approx(micro.value_listed,
                                                        abs=1e-12)


# -- the published relations ---------------------------------------------------------------

def test_the_probe_ion_equations_are_duffy_and_ingrams():
    # Lambda = 1 is CaO: the Pb2+ band there is 60700 - 31000 = 29700 cm-1
    assert B.probe_band(1.0, "Pb2+") == pytest.approx(29700.0)
    assert B.probe_band(0.0, "Bi3+") == pytest.approx(56000.0)
    assert B.probe_band(1.0, "Tl+") == pytest.approx(55300.0 - 18300.0)
    # Duffy 1989: Pb2+ in molten LiF-NaF-KF absorbs at 44600 cm-1, Lambda 0.52
    assert B.basicity_from_band(44600.0, "Pb2+") == pytest.approx(0.52, abs=0.005)
    for ion in B.PROBE_IONS:
        assert B.basicity_from_band(B.probe_band(0.63, ion), ion) == \
            pytest.approx(0.63)
    assert B.probe_wavelength(1.0, "Pb2+") == pytest.approx(1e7 / 29700.0)
    with pytest.raises(ValueError):
        B.probe_band(0.5, "Sn2+")


def test_the_polarisability_relation_and_its_inverse():
    # Lambda = 1.67 (1 - 1 / alpha): CaO at 1 gives 1.67 / 0.67
    assert B.oxide_polarisability(1.0) == pytest.approx(1.67 / 0.67)
    assert B.basicity_from_polarisability(B.oxide_polarisability(0.48)) == \
        pytest.approx(0.48)
    assert math.isnan(B.oxide_polarisability(1.67))
    assert math.isnan(B.oxide_polarisability(float("nan")))


# -- a written composition -----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Na2Si2O5", 0.61), ("Na2O-2SiO2", 0.61), ("CaSiO3", 0.65),
    ("Ca2SiO4", 0.74), ("Ca3(PO4)2", 0.58), ("NaPO3", 0.47)])
def test_compositions_on_duffys_scale_are_his_table_2(text, expected):
    # J. A. Duffy, J. Chem. Educ. 73 (1996) 1138, Table 2
    assert C.composition_basicity(text, "duffy").value == pytest.approx(
        expected, abs=0.016)


def test_a_composition_reads_the_same_however_it_is_written():
    forms = ["30Na2O-70SiO2", "30 Na2O 70 SiO2", "Na2O: 30, SiO2: 70",
             "0.3Na2O·0.7SiO2", "Na2O 30; SiO2 70", "30%Na2O + 70%SiO2",
             "3Na2O-7SiO2", {"Na2O": 30, "SiO2": 70}]
    values = [C.composition_basicity(f).value for f in forms]
    assert max(values) - min(values) < 1e-12
    # by hand: X = anion charge share; Na2O 0.3 of 1.7 oxygens, SiO2 1.4
    g_na, g_si = (B.gamma_from_electronegativity(x) for x in (0.9, 1.8))
    assert values[0] == pytest.approx((0.3 / g_na + 1.4 / g_si) / 1.7)
    # the species route (what a structure of this composition gives)
    species = {("Na", 1): 0.6, ("Si", 4): 0.7, ("O", -2): 1.7}
    assert values[0] == pytest.approx(B.bulk_basicity(species).value)


def test_weights_and_moles_agree_and_fluorides_keep_their_cation():
    by_mole = C.composition_basicity("30Na2O-70SiO2")
    weights = {c.formula: c.fraction * c.weight for c in by_mole.components}
    assert weights["Na2O"] / 0.3 == pytest.approx(61.979, abs=0.01)
    assert C.composition_basicity(weights, unit="wt").value == \
        pytest.approx(by_mole.value)
    # Duffy 1989, eq. 4: each fluoride with its own cation
    mixed = C.composition_basicity("20CaF2-80SiO2")
    x_f = 0.2 * 2 / (0.2 * 2 + 0.8 * 4)
    g_si = B.gamma_from_electronegativity(1.8)
    g_ca = B.gamma_from_electronegativity(1.0)
    assert mixed.value == pytest.approx(x_f / (2.3 * g_ca) + (1 - x_f) / g_si)
    assert any("eq. 4" in n for n in mixed.notes)
    assert C.composition_basicity("FeO", "duffy").value == pytest.approx(1.0)


@pytest.mark.parametrize("text,word", [
    ("Fe3O4", "whole oxidation state"), ("NaCl", "no oxide or fluoride"),
    ("Na2O 30 20 SiO2", "alternate"), ("2 3 Na2O SiO2 CaO", "two amounts"), ("Na2O-SiO2 5 CaO 3", "no formula after"),
    ("Xx2O", "not an element"), ("", "no formula"),
    ("Na2(O", "not closed")])
def test_a_composition_that_cannot_be_read_says_why(text, word):
    with pytest.raises(C.CompositionError, match=word):
        C.composition_basicity(text)


def test_a_series_runs_between_the_two_end_members():
    x, values = fb.series("Na2O", "SiO2", steps=5)
    assert x[0] == 0 and x[-1] == 1
    assert values[0] == pytest.approx(fb.oxide("Na", 1))
    assert values[-1] == pytest.approx(fb.oxide("Si", 4))
    assert (np.diff(values) < 0).all()


# -- linkages --------------------------------------------------------------------------------

def test_linkage_names():
    name = B.linkage_pattern
    assert name("O", {"Si": 2}, ["Si"]) == "Si-O-Si"
    assert name("O", {"Si": 1, "Al": 1}, ["Si", "Al"]) == "Al-O-Si"
    assert name("O", {"Si": 1, "Na": 3}, ["Si"]) == "Si-O-Na3"
    assert name("O", {"Si": 1}, ["Si"]) == "Si-O"
    assert name("O", {"Ca": 4, "Na": 2}, ["Si"]) == "O-Ca4 Na2"
    assert name("O", {"Si": 3}, ["Si"]) == "O(Si3)"
    assert name("F", {}, ["Si"]) == "F alone"


def test_linkages_of_a_crystal_and_of_its_supercell_frame_agree():
    crystal = fb.structure(QUARTZ, formers=["Si"])
    (link,) = crystal.linkages()
    assert link.pattern == "Si-O-Si" and link.atoms == pytest.approx(6.0)
    assert link.mean == pytest.approx(crystal.value) and link.sd == pytest.approx(0)
    frame, _ = md_model.supercell_frame(crystal.structure, (2, 2, 2))
    model = fb.frame(frame, formers=["Si"])
    (again,) = model.linkages()
    assert again.pattern == "Si-O-Si" and again.atoms == 48
    assert again.mean == pytest.approx(link.mean, abs=1e-4)
    assert model.by_class() and model.by_class()[0][1] == "BO"


# -- the spatial correlation ----------------------------------------------------------------

def test_the_correlation_is_the_direct_sum_over_minimum_images():
    structure = readers.read(str(BIPO4))
    frame, _ = md_model.supercell_frame(structure, (2, 2, 2))
    rng = np.random.default_rng(7)
    values = np.where(np.asarray(frame.elements) == "O",
                      rng.normal(0.5, 0.05, frame.n_atoms), np.nan)
    found = F.spatial_correlation(frame, values, r_max_ang=5.0, bin_ang=0.5)
    rows = np.flatnonzero(np.isfinite(values))
    box = np.asarray(frame.box_ang, float)
    cart = bulk._lattice(np.asarray(frame.frac)[rows], box)
    shifts = bulk._lattice(np.array([[i, j, k] for i in (-1, 0, 1)
                                     for j in (-1, 0, 1) for k in (-1, 0, 1)],
                                    float), box)
    d = values[rows] - values[rows].mean()
    n_bins = len(found.r_ang)
    total, count = np.zeros(n_bins), np.zeros(n_bins)
    for a in range(len(rows)):
        for shift in shifts:
            dist = np.linalg.norm(cart + shift - cart[a], axis=1)
            ok = (dist > 1e-6) & (dist < found.r_max_ang)
            which = np.minimum((dist[ok] / 0.5).astype(int), n_bins - 1)
            np.add.at(total, which, d[a] * d[ok])
            np.add.at(count, which, 1)
    assert np.array_equal(found.pairs, count.astype(int))
    filled = count > 0
    assert found.value[filled] == pytest.approx(
        total[filled] / (count[filled] * values[rows].var()), abs=1e-9)
    # a shift of every value changes nothing; a constant has no correlation
    again = F.spatial_correlation(frame, values + 3.0, r_max_ang=5.0, bin_ang=0.5)
    assert again.value[filled] == pytest.approx(found.value[filled], abs=1e-9)
    flat = F.spatial_correlation(frame, np.where(np.isfinite(values), 0.5, np.nan),
                                 r_max_ang=5.0, bin_ang=0.5)
    assert np.isnan(flat.value).all() and flat.notes


def test_the_correlation_range_stops_at_half_the_box():
    structure = readers.read(str(QUARTZ))
    frame, _ = md_model.supercell_frame(structure, (2, 2, 2))
    values = np.where(np.asarray(frame.elements) == "O", 0.5, np.nan)
    values[np.flatnonzero(np.isfinite(values))[::2]] = 0.6
    found = F.spatial_correlation(frame, values, r_max_ang=50.0)
    half = 0.5 * float(np.min(frame.perpendicular_widths_ang))
    assert found.r_max_ang == pytest.approx(half) and "half" in found.notes[0]


# -- the library: a structure, a frame, a field ------------------------------------------------

def test_a_structure_result_carries_the_rows_and_writes_them(tmp_path):
    s = fb.structure(BIPO4, "lebouteiller", formers=["P"])
    assert s.scale == "lebouteiller" and math.isfinite(s.value)
    rows = s.anions()
    assert {r[0] for r in rows} == {"O1", "O2"}
    assert all(r[7].startswith("P-O-Bi") for r in rows)
    assert s.site("O1").value == pytest.approx(rows[0][5])
    assert s.site("Bi").value == pytest.approx(s.cations()[0][3])
    with pytest.raises(KeyError):
        s.site("Zz9")
    data = json.loads(json.dumps(s.to_dict()))          # plain values only
    assert data["scale"] == "lebouteiller" and len(data["anions"]) == 2
    written = s.to_csv(tmp_path / "sites.csv").read_text(encoding="utf-8")
    assert written.splitlines()[0].startswith("site,element,atoms")
    assert len(written.splitlines()) == 3
    text = str(s)
    assert "Lambda from the composition" in text and "P-O-Bi" in text
    assert "alpha(O2-)" in text
    assert "alpha(O2-)" not in str(fb.structure(CRYOLITE))   # no oxide ion


def _extxyz(tmp_path, source=BIPO4):
    frame, _ = md_model.supercell_frame(readers.read(str(source)), (2, 2, 2))
    box = " ".join(f"{x:.6f}" for x in np.asarray(frame.box_ang).ravel())
    lines = [str(frame.n_atoms),
             f'Lattice="{box}" Properties=species:S:1:pos:R:3 pbc="T T T"']
    lines += [f"{e} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"
              for e, c in zip(frame.elements, np.asarray(frame.cart_ang))]
    path = tmp_path / "model.extxyz"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_frame_from_a_file_is_the_crystals_numbers(tmp_path):
    path = _extxyz(tmp_path)
    assert fb.is_md_file(path) and not fb.is_md_file(BIPO4)
    g = fb.load(path, formers=["P"])
    crystal = fb.structure(BIPO4)
    assert isinstance(g, fb.FrameResult) and g.frame.n_atoms == 144
    assert g.value == pytest.approx(crystal.value, abs=2e-4)
    assert g.bulk == pytest.approx(crystal.bulk)
    assert np.isnan(g.values[g.elements == "Bi"]).all()
    # a Bi3+ ion's band in the basicity of its own oxygens
    band = g.band("Bi")
    env = g.environment[g.elements == "Bi"]
    assert band[g.elements == "Bi"] == pytest.approx(56000.0 - 28800.0 * env)
    assert np.isnan(band[g.elements == "O"]).all()
    assert len(g.atoms()) == 144
    json.dumps(g.to_dict())
    with pytest.raises(IndexError):
        fb.frame(path, index=5)
    assert [r.index for r in fb.frames(path)] == [0]


def test_a_field_slices_and_writes_a_cube_that_reads_back(tmp_path):
    s = fb.structure(BIPO4)
    field = s.field("Pb", resolution=0.5)
    na, nb, nc = field.values.shape
    cut = field.slice("c", 0.5, 2.0)
    assert cut["values"].shape == (na + 1, nb + 1)
    assert cut["x"].shape == (na + 2, nb + 2)
    assert np.allclose(cut["values"][-1], cut["values"][0], equal_nan=True)
    assert (np.abs(cut["atom_distance"]) <= 2.0).all() and len(cut["atom_x"])
    smooth = field.smoothed(1.0)
    assert np.ptp(smooth.range) < np.ptp(field.range)

    path = field.to_cube(tmp_path / "pb.cube")
    lines = path.read_text().splitlines()
    n_atoms = int(lines[2].split()[0])
    assert n_atoms == len(s.structure.atoms)
    counts, voxels = [], []
    for line in lines[3:6]:
        parts = line.split()
        counts.append(int(parts[0]))
        voxels.append([float(x) for x in parts[1:]])
    assert tuple(counts) == (na, nb, nc)
    cell = np.array(voxels) * np.array(counts)[:, None] * F.BOHR_ANG
    assert cell == pytest.approx(field.vectors, abs=1e-4)
    numbers = np.array(" ".join(lines[6 + n_atoms:]).split(), float)
    assert numbers.size == na * nb * nc
    expected = np.where(np.isfinite(field.values), field.values, 0.0)
    assert numbers.reshape(na, nb, nc) == pytest.approx(expected, abs=1e-5)
    undefined = int(lines[1].split()[0])
    assert undefined == int((~np.isfinite(field.values)).sum())


def test_the_library_imports_no_window_toolkit():
    code = ("import sys, facet.basicity as fb;"
            f"s = fb.structure(r'{QUARTZ}'); fb.composition('Na2O-SiO2');"
            "s.field('Na', resolution=0.6);"
            "bad = [m for m in sys.modules if m.startswith(('PySide6', 'matplotlib'))];"
            "print(bad); sys.exit(1 if bad else 0)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(Path(__file__).resolve().parent.parent))
    assert done.returncode == 0, done.stdout + done.stderr


# -- the command line ------------------------------------------------------------------------

def _run(*argv):
    from facet.basicity import cli

    return cli.main([str(a) for a in argv])


def test_the_commands_print_their_tables(tmp_path, capsys):
    assert _run("scales") == 0
    assert "lebouteiller" in capsys.readouterr().out
    assert _run("oxide", "Si") == 0
    assert "0.477" in capsys.readouterr().out
    assert _run("composition", "30Na2O-70SiO2", "--scale", "all") == 0
    out = capsys.readouterr().out
    assert all(key in out for key in B.SCALE_KEYS)
    assert _run("composition", "30Na2O-70SiO2", "--json") == 0
    assert json.loads(capsys.readouterr().out)["Lambda"] == pytest.approx(
        C.composition_basicity("30Na2O-70SiO2").value)
    assert _run("series", "Na2O", "SiO2", "--steps", "3") == 0
    assert len(capsys.readouterr().out.splitlines()) == 5
    assert _run("band", "44600", "--from-band", "--ion", "Pb2+") == 0
    assert "0.519" in capsys.readouterr().out
    csv_path = tmp_path / "s.csv"
    assert _run("structure", QUARTZ, "--formers", "Si", "--csv", csv_path) == 0
    assert "Si-O-Si" in capsys.readouterr().out and csv_path.exists()
    model = _extxyz(tmp_path)
    assert _run("md", model, "--formers", "P", "--correlation",
                "--r-max", "4") == 0
    out = capsys.readouterr().out
    assert "C(r)" in out and "NBO" in out
    cube = tmp_path / "f.cube"
    assert _run("field", model, "--probe", "Pb", "--grid", "1.0",
                "--cube", cube) == 0
    assert cube.stat().st_size > 1000 and "Pb2+" in capsys.readouterr().out


def test_a_command_that_cannot_run_exits_2_with_the_reason(capsys):
    assert _run("composition", "Fe3O4") == 2
    assert "whole oxidation state" in capsys.readouterr().err
    assert _run("structure", "no_such_file.cif") == 2
    assert _run("band", "0.5", "--ion", "Pb2+") == 0


def test_every_quick_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    model = _extxyz(tmp_path)
    jobs = [("series", "Na2O", "SiO2"),
            ("sites", BIPO4, "--formers", "P"),
            ("histogram", model, "--formers", "P"),
            ("histogram", BIPO4, "--by", "linkage"),
            ("linkages", model, "--formers", "P"),
            ("slice", BIPO4, "--probe", "Pb", "--grid", "0.5"),
            ("slice", model, "--probe", "Pb", "--axis", "a", "--at", "0.25"),
            ("correlation", model, "--r-max", "4"),
            ("band", model, "--probe", "Bi")]
    for k, job in enumerate(jobs):
        out = tmp_path / f"fig{k}.png"
        assert _run("plot", *job, "-o", out) == 0, job
        assert out.stat().st_size > 5000, job
    assert _run("plot", "sites", model, "-o", tmp_path / "no.png") == 2


# -- the window's Tools tab -------------------------------------------------------------------

def test_the_tools_tab_reads_a_composition_and_lists_the_linkages(qapp):
    from facet.ui import preview

    win = preview.PreviewWindow()
    try:
        tools = win.utilities
        tools.basicity_composition.setText("30Na2O-70SiO2")
        expected = C.composition_basicity("30Na2O-70SiO2").value
        assert f"{expected:.4f}" in tools.basicity_composition_text.text()
        tools.set_basicity_scale("duffy")
        assert "duffy" in tools.basicity_composition_text.text()
        tools.basicity_composition.setText("15 Na2O 85 SiO2 wt")
        assert "weights" in tools.basicity_composition_text.text()
        tools.basicity_composition.setText("Fe3O4")
        assert "whole oxidation state" in tools.basicity_composition_text.text()
        tools.basicity_composition.setText("")
        assert tools.basicity_composition_text.text() == ""
        win.load_many([str(BIPO4)])
        qapp.processEvents()
        tools.refresh_tab("Basicity")
        html = tools.basicity_text.toPlainText()
        assert "by what the anion is bonded to" in html
        assert "Probe-ion" in html and "polarisability" in html
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_view_opens_the_window_coloured_by_basicity(qapp, tmp_path):
    import time

    from facet.core import theme as theme_mod
    from facet.ui import app as app_mod
    from facet.ui import preview
    from facet.ui.md_highlight import COLOUR_MODES

    win = preview.PreviewWindow(str(BIPO4))
    try:
        qapp.processEvents()
        app_mod._colour_by_basicity(win)
        qapp.processEvents()
        assert win.theme.color_mode is theme_mod.ColorMode.BASICITY
        assert win.anion_check.isChecked()
        assert win.scene is not None and win.scene.value_legend is not None
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()

    win = preview.PreviewWindow()
    try:
        controller = win.open_md_models([str(_extxyz(tmp_path))])[0]
        app_mod._colour_by_basicity(win)            # before the model is read
        wanted = COLOUR_MODES.index("microscopic basicity")
        deadline = time.time() + 120
        while time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.05)
            page = controller.highlight
            if page is not None and page.colour.currentIndex() == wanted:
                break
        else:
            pytest.fail("the model's colour mode was not set to the basicity")
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()
