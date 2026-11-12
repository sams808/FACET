"""FACET as a library (``import facet``): the crystal side, the tables, the
figures and the picture.

A wrapper earns its place by returning the engine's number, so each test
either recomputes the number outside the wrapper (a bond valence from its
formula, Bragg's law, the reference file the manual's numbers are held to)
or compares with the engine called directly. The model side is
tests/test_library_model.py.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import facet
from conftest import has_vector_marks

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "facet" / "data" / "examples"
REFERENCE = json.loads((EXAMPLES / "REFERENCE.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bipo4():
    return facet.example("bipo4")


@pytest.fixture(scope="module")
def cryolite():
    return facet.example("cryolite")


# --- the front door -------------------------------------------------------------------

def test_importing_facet_loads_no_window_toolkit_and_no_heavy_package():
    code = ("import sys, facet; c = facet.example('quartz'); c.sites(); "
            "c.diffraction(); c.pdf(r_max=5); "
            "print(sorted(m for m in ('PySide6', 'matplotlib', 'pymatgen', "
            "'pandas') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_every_public_name_is_reachable_from_facet():
    import facet.library as lib

    for name in lib.__all__:
        assert getattr(facet, name) is getattr(lib, name), name
    assert facet.basicity.composition("30Na2O-70SiO2").value == pytest.approx(0.596, abs=1e-3)
    with pytest.raises(AttributeError):
        facet.no_such_name


def test_no_library_name_is_the_name_of_a_sub_package():
    """facet.report is the report command's package: a library function of
    that name is replaced by the module as soon as the package is imported
    (it was, and the full suite found it). Every name is checked against
    the folders of the package, and after importing each command."""
    import importlib

    import facet.library as lib

    package = ROOT / "facet"
    folders = {p.name for p in package.iterdir() if p.is_dir()} |         {p.stem for p in package.glob("*.py")}
    clashes = sorted(set(lib.__all__) & (folders - {"basicity"}))
    assert not clashes, clashes
    for command in ("report", "site", "md", "check", "basicity"):
        importlib.import_module(f"facet.{command}")
    for name in lib.__all__:
        assert getattr(facet, name) is getattr(lib, name), name


def test_the_examples_are_the_bundled_files():
    found = facet.examples()
    assert {"quartz", "eulytite", "senarmontite", "valentinite", "cryolite",
            "bipo4"} <= set(found)
    assert all(path.is_file() for path in found.values())
    with pytest.raises(KeyError, match="quartz"):
        facet.example("no such example")


def test_a_file_that_is_missing_or_is_a_model_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        facet.read(tmp_path / "absent.cif")
    dump = ROOT / "tests" / "data" / "md" / "dump_general.lammpstrj"
    with pytest.raises(ValueError, match=r"facet\.model"):
        facet.read(dump)
    assert isinstance(facet.load(EXAMPLES / "quartz_SiO2_cod9013321.cif"),
                      facet.Crystal)


# --- the sites ------------------------------------------------------------------------

@pytest.mark.parametrize("filename", sorted(REFERENCE["examples"]))
def test_the_site_table_holds_the_numbers_the_manual_states(filename):
    """REFERENCE.json is what the tutorials state and ``facet.check``
    recomputes; the library's table is held to the same numbers."""
    expected = REFERENCE["examples"][filename]["sites"]
    crystal = facet.read(EXAMPLES / filename)
    rows = {row["site"]: row for row in crystal.sites()}
    assert set(expected) <= set(rows)
    for label, want in expected.items():
        row = rows[label]
        assert row["CN"] == want["cn"]
        assert row["BVS / v.u."] == pytest.approx(want["bvs_vu"], abs=5e-4)
        assert row["phi"] == pytest.approx(want["phi"], abs=5e-4)
        assert row["plateau / decades"] == pytest.approx(want["plateau_decades"], abs=5e-3)
        assert row["ox"] == want["oxidation_state"]


def test_the_sites_are_the_engines_results(bipo4):
    from facet.core import bv, cif, coordination

    direct = coordination.analyse_structure(
        cif.read(EXAMPLES / "bismuth_phosphate_BiPO4_cod9008088.cif"), bv.DEFAULT)
    assert [r.label for r in direct] == [row["site"] for row in bipo4.sites()]
    for r in direct:
        site = bipo4.site(r.label)
        assert (site.cn, site.bvs, site.phi) == (r.cn_valence, r.bvs, r.phi)
        assert site.result.contacts[0].distance == r.contacts[0].distance
    assert len(bipo4.sites(anions=True)) == bipo4.n_sites
    assert bipo4["Bi"].cn == bipo4.site("Bi").cn == bipo4.site(0).cn
    with pytest.raises(KeyError, match="Bi, P, O1, O2"):
        bipo4.site("Xx9")


def test_a_contacts_valence_is_the_formula_and_the_sum_is_the_sites(bipo4):
    site = bipo4.site("Bi")
    rows = site.contacts()
    for row in rows:
        formula = math.exp((row["R0 / Å"] - row["distance / Å"]) / row["b / Å"])
        assert row["valence / v.u."] == pytest.approx(formula, rel=1e-12)
        assert row["bonded"] == (formula > bipo4.v_bond)
        assert facet.bond_valence(row["distance / Å"], row["R0 / Å"], row["b / Å"]) \
            == pytest.approx(formula, rel=1e-12)
    bonded = rows.where(bonded=True)
    assert len(bonded) == site.cn
    assert bonded.column("valence / v.u.").sum() == pytest.approx(site.bvs, rel=1e-12)
    distances = rows.column("distance / Å")
    assert (np.diff(distances) >= 0).all()                # nearest first
    cut = facet.cutoff_distance(2.09, bipo4.v_bond)
    assert cut == pytest.approx(2.09 - 0.37 * math.log(bipo4.v_bond))


def test_the_cn_function_steps_where_the_plateaus_say(bipo4):
    site = bipo4.site("Bi")
    plateaus = site.plateaus()
    holding = plateaus.where(**{"holds the bond threshold": True})
    assert len(holding) == 1 and holding[0]["CN"] == site.cn
    assert site.plateau.cn == site.cn
    assert site.plateau_decades == pytest.approx(holding[0]["width / decades"])
    v, cn = site.cn_function(n=300)
    for threshold, count in zip(v[::17], cn[::17]):
        assert count == site.cn_at(threshold)
    assert (np.diff(cn) <= 0).all()                       # a higher bar, fewer bonds
    lower = site.at(0.5)
    assert lower.cn == site.cn_at(0.5) < site.cn
    assert lower.bvs < site.bvs and site.at(bipo4.v_bond).bvs == site.bvs


def test_angles_shells_methods_and_the_text_of_a_site(bipo4):
    site = bipo4.site("P")
    angles = site.angles()
    assert len(angles) == site.cn * (site.cn - 1) // 2 == 6
    assert angles.column("angle / deg").mean() == pytest.approx(109.47, abs=0.2)
    shells = site.shells()
    assert shells[site.cn - 1]["CN after"] == site.cn
    assert shells[site.cn - 1]["valence sum after / v.u."] == pytest.approx(site.bvs)
    methods = site.cn_methods()
    by = {row["method"]: row for row in methods}
    assert by["Bond valence"]["CN"] == site.cn
    assert by["ECoN (Hoppe)"]["CN"] == pytest.approx(site.econ)
    assert any("spread over the methods" in note for note in methods.notes)
    text = site.text()
    assert "Coordination number" in text and f"{site.bvs:.2f}" in text
    assert site.text("md").startswith("## P") and "\\" in site.text("tex")
    assert site.shape()["n"] == 4
    lone = bipo4.site("Bi").lone_pair()
    assert lone["phi"] == bipo4.site("Bi").phi
    assert np.linalg.norm(lone["bond-valence vector / v.u."]) == pytest.approx(bipo4.site("Bi").bvv)


def test_exafs_shells_and_the_feff_input(bipo4, tmp_path):
    table = bipo4.site("Bi").exafs_shells(r_max=4.0)
    rows = facet.table(table.shells)
    assert rows[0]["element"] == "O" and rows[0]["r_mean"] == pytest.approx(2.32, abs=0.01)
    first = bipo4.site("Bi").radial_shells(4.0)
    assert sum(r["count"] for r in first if r["element"] == "O" and r["distance"] < 3.0) == 8
    written = bipo4.site("Bi").write_feff(tmp_path / "feff.inp")
    text = written.read_text(encoding="utf-8", errors="replace")
    assert "POTENTIALS" in text and "ATOMS" in text and "Bi" in text


# --- the structure as a whole ----------------------------------------------------------

def test_composition_density_and_the_file_check(bipo4):
    comp = bipo4.composition()
    assert comp["counts"] == {"Bi": 3.0, "P": 3.0, "O": 12.0} and comp["balanced"]
    density = bipo4.density()
    mass = 3 * (208.9804 + 30.973762 + 4 * 15.999)
    assert density["density"] == pytest.approx(
        mass / 6.02214076e23 / (bipo4.cell.volume * 1e-24), rel=2e-4)
    check = bipo4.check()
    assert set(check.columns) >= {"level", "code", "message"}
    states = {row["site"]: row["ox"] for row in bipo4.oxidation_states()}
    assert states == {"Bi": 3, "P": 5}
    assert bipo4.disorder().present is False


def test_braggs_law_and_the_powder_pattern(bipo4):
    d = bipo4.d_spacing(1, 0, 1)
    two_theta = bipo4.two_theta(1, 0, 1, wavelength=1.5406)
    assert 1.5406 == pytest.approx(2 * d * math.sin(math.radians(two_theta / 2)))
    assert bipo4.plane(1, 0, 1).spacing == pytest.approx(d)
    pattern = bipo4.diffraction(two_theta_max=40)
    lines = facet.table(pattern.reflections)
    assert lines.column("intensity").max() == pytest.approx(100.0)
    strongest = lines.sorted_by("intensity", reverse=True)[0]
    assert (strongest["h"], strongest["k"], strongest["l"]) == (1, 0, 1)
    assert strongest["d"] == pytest.approx(d, abs=1e-6)
    from facet.core import diffraction

    assert bipo4.structure_factor(1, 0, 1) == diffraction.structure_factor(
        bipo4.structure, 1, 0, 1)
    assert abs(bipo4.structure_factor(1, 0, 1)) > 0
    neutron = bipo4.diffraction(radiation="neutron", two_theta_max=40)
    assert neutron.radiation == "neutron"
    assert len(bipo4.reflections(two_theta_max=20)) >= 2


def test_the_pdf_puts_its_first_peak_at_the_shortest_bond(bipo4):
    found = bipo4.pdf(r_max=6.0, dr=0.01)
    assert found.r.shape == found.G.shape == found.g.shape
    window = (found.r > 1.2) & (found.r < 1.9)
    peak = found.r[window][np.argmax(found.g[window])]
    shortest = bipo4.site("P").contacts()[0]["distance / Å"]
    assert peak == pytest.approx(shortest, abs=0.03)
    assert found.notes


def test_cutoffs_sums_and_indices(bipo4, cryolite):
    for row in bipo4.cutoffs():
        assert row["bond cutoff / Å"] == pytest.approx(
            row["R0 / Å"] - row["b / Å"] * math.log(bipo4.v_bond))
    anions = bipo4.anion_sums()
    cations = bipo4.sites()
    weight = {s.label: s.multiplicity * s.occupancy for s in bipo4.structure.sites}
    total_anions = sum(r["BVS / v.u."] * weight[r["site"]] for r in anions)
    total_cations = sum(r["BVS / v.u."] * weight[r["site"]] for r in cations)
    assert total_anions == pytest.approx(total_cations, rel=1e-5)
    balance = bipo4.charge_balance()
    assert balance["cation_valence"] == pytest.approx(total_cations, rel=1e-9)
    discrepancies = cations.column("BVS - ox / v.u.")
    assert bipo4.gii()["gii"] == pytest.approx(
        math.sqrt(float(np.mean(discrepancies ** 2))), rel=1e-9)
    for row in bipo4.a_priori():
        assert sum(row["a priori / v.u."]) == pytest.approx(row["ox"])
        assert len(row["observed / v.u."]) == row["CN"]
    assert bipo4.bond_strain() > 0
    joins = bipo4.connectivity()
    assert {"site_a", "site_b", "sharing"} <= set(joins.columns)
    scan = cryolite.stability()
    assert set(scan.per_site) == {"Al", "Na1", "Na2"}
    low, high = scan.stable_range("Al", cryolite.v_bond)
    assert low < cryolite.v_bond < high


def test_planes_atoms_and_torsions(bipo4):
    plane = bipo4.plane(0, 0, 1)
    assert plane.spacing == pytest.approx(bipo4.cell.c)
    assert plane.angle_to(1, 0, 0) == pytest.approx(90.0)
    assert sum(plane.occupancy(0.3).values()) == len(plane.atoms(0.3))
    atoms = bipo4.atoms()
    assert len(atoms) == 18 and set(atoms.column("element")) == {"Bi", "P", "O"}
    assert -180.0 <= bipo4.torsion(0, 1, 2, 3) <= 180.0
    with pytest.raises(ValueError):
        bipo4.plane(0, 0, 0)


def test_the_valence_map_gives_a_site_its_own_valence_sum(cryolite):
    """Outside check: where the probe sits on its own site, the map reads
    the bond-valence sum the site analysis gives for that site."""
    grid = cryolite.valence_map("Na", resolution=0.2)
    site = cryolite.site("Na1")
    frac = cryolite.structure.sites[site.result.site_index].frac
    at_site = float(grid.at_fractional(frac)[0])
    assert at_site == pytest.approx(site.result.bvs_listed, rel=0.15)
    energy = cryolite.energy_map("Na", resolution=0.4)
    assert energy.shape == cryolite.valence_map("Na", resolution=0.4).shape
    with pytest.raises(ValueError, match="oxidation state"):
        cryolite._probe_ox("He", None)


def test_thresholds_and_parameter_sets(bipo4):
    lower = bipo4.with_thresholds(v_bond=0.2)
    assert lower.site("Bi").cn == bipo4.site("Bi").cn_at(0.2)
    assert lower.structure is bipo4.structure
    iucr = bipo4.with_thresholds(params="bvparm2020")
    assert iucr.params.name != bipo4.params.name
    assert iucr.site("Bi").bvs != bipo4.site("Bi").bvs
    assert facet.parameters("built-in") is facet.parameters(None)
    assert facet.parameters(bipo4.params) is bipo4.params


def test_basicity_is_the_basicity_librarys(bipo4):
    found = bipo4.basicity()
    direct = facet.basicity.structure(EXAMPLES / "bismuth_phosphate_BiPO4_cod9008088.cif")
    assert found.value == pytest.approx(direct.value, rel=1e-12)
    assert found.bulk == pytest.approx(direct.bulk, rel=1e-12)


# --- writing, several structures --------------------------------------------------------

@pytest.mark.parametrize("name", ["out.cif", "POSCAR_out.vasp", "out.xyz", "out.vesta"])
def test_a_structure_written_reads_back_as_the_same_atoms(bipo4, tmp_path, name):
    written = bipo4.write(tmp_path / name)
    assert written.stat().st_size > 500
    if name.endswith(".xyz"):
        return                              # read back as a cluster, by design
    again = facet.read(written)
    assert again.composition()["counts"] == {"Bi": 3.0, "P": 3.0, "O": 12.0}
    assert sorted(set(np.round(again.sites().column("BVS / v.u."), 3))) == \
        sorted(set(np.round(bipo4.sites().column("BVS / v.u."), 3)))
    with pytest.raises(ValueError, match="formats written"):
        bipo4.write(tmp_path / "out.docx")


def test_the_report_and_the_comparison_cover_every_file(tmp_path):
    paths = [EXAMPLES / "quartz_SiO2_cod9013321.cif",
             EXAMPLES / "eulytite_Bi4SiO4_3_cod9012894.cif"]
    report = facet.report_on(paths, tmp_path / "report.xlsx")
    assert (tmp_path / "report.xlsx").stat().st_size > 5000
    tables = report.tables()
    assert len(tables["sites"]) == 3
    both = facet.compare(paths)
    eulytite = facet.read(paths[1]).name
    assert [row["structure"] for row in both] == ["Quartz", eulytite, eulytite]
    assert both.where(structure="Quartz")[0]["CN"] == 4
    html = facet.report_on(paths, tmp_path / "report.html")
    assert html.v_bond == facet.V_BOND and (tmp_path / "report.html").is_file()


def test_the_selfcheck_recomputes_the_reference_numbers():
    rows = facet.selfcheck()
    assert len(rows) >= 50 and all(row["agrees"] for row in rows)
    assert "within tolerance" in rows.title


# --- tables ---------------------------------------------------------------------------

def test_a_table_is_a_list_of_rows_with_helpers(bipo4, tmp_path):
    table = bipo4.sites(anions=True)
    assert isinstance(table, list) and isinstance(table[0], dict)
    assert table.columns[:3] == ["site", "element", "ox"]
    assert table.column("CN").dtype == float
    assert list(table.column("site")) == ["Bi", "P", "O1", "O2"]
    oxygens = table.where(element="O")
    assert len(oxygens) == 2 and oxygens.title == table.title
    by_cn = table.sorted_by("CN", reverse=True)
    assert by_cn[0]["site"] == "Bi" and len(table.head(1)) == 1
    with pytest.raises(KeyError, match="columns are"):
        table.column("nothing")
    text = str(table)
    assert text.splitlines()[0] == "sites of Ximengite" and "note:" in text
    assert "row(s)" in repr(table) and "<table>" in table._repr_html_()
    written = table.to_csv(tmp_path / "sites.csv")
    with open(written, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    back = list(csv.DictReader(lines))
    assert [row["site"] for row in back] == ["Bi", "P", "O1", "O2"]
    assert float(back[0]["BVS / v.u."]) == pytest.approx(table[0]["BVS / v.u."])
    pandas = pytest.importorskip("pandas")
    frame = table.to_frame()
    assert isinstance(frame, pandas.DataFrame) and list(frame["site"]) == ["Bi", "P", "O1", "O2"]
    assert pandas.DataFrame(table).shape == frame.shape
    assert str(facet.Table()) == "(no rows)"


def test_values_in_a_table_are_plain_python(bipo4):
    for row in bipo4.site("Bi").cn_methods():
        for value in row["parameters"].values():
            assert type(value) in (int, float, str, bool, list, type(None))
    row = facet.row_of(bipo4.site("Bi").plateau, {"cn": "CN", "v_low": "from"})
    assert row == {"CN": 8, "from": pytest.approx(0.02)}
    json.dumps(list(bipo4.sites(anions=True)))            # serialisable as it is


# --- figures and the picture -------------------------------------------------------------

def test_every_quick_figure_is_written(bipo4, cryolite, tmp_path):
    pytest.importorskip("matplotlib")
    site = bipo4.site("Bi")
    made = [
        facet.plot.cn_function(site, save=tmp_path / "cn.png"),
        facet.plot.contacts(site, save=tmp_path / "contacts.svg"),
        facet.plot.stability(cryolite, save=tmp_path / "stability.png"),
        facet.plot.pattern(bipo4.diffraction(two_theta_max=60), save=tmp_path / "pattern.pdf"),
        facet.plot.pdf(bipo4.pdf(r_max=8), save=tmp_path / "pdf.png"),
        facet.plot.grid(cryolite.valence_map("Na", resolution=0.5), "c", 0.25,
                        save=tmp_path / "grid.png"),
    ]
    assert all(figure.axes for figure in made)
    for name in ("cn.png", "contacts.svg", "stability.png", "pattern.pdf", "pdf.png",
                 "grid.png"):
        assert (tmp_path / name).stat().st_size > 2000, name
    assert "<svg" in (tmp_path / "contacts.svg").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        facet.plot.pdf(bipo4.pdf(r_max=5), which="S")
    with pytest.raises(ValueError):
        facet.plot.grid(cryolite.valence_map("Na", resolution=0.6), axis="q")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_a_picture_of_the_structure_is_drawn_without_a_window(qapp, tmp_path):
    from PySide6.QtGui import QImage

    quartz = facet.example("quartz")
    png = quartz.render(tmp_path / "quartz.png", size=(640, 480))
    image = QImage(str(png))
    assert (image.width(), image.height()) == (640, 480)
    colours = {image.pixel(x, y) for x in range(0, 640, 8) for y in range(0, 480, 8)}
    assert len(colours) > 12, "the picture is a flat colour"
    assert image.pixelColor(2, 2).name() == "#ffffff"       # the white ground
    along = quartz.render(tmp_path / "along_c.png", size=(640, 480), along="c",
                          style="polyhedral", cell_range=(2, 2, 1), polyhedra=["Si"])
    assert QImage(str(along)) != image
    svg = quartz.render(tmp_path / "quartz.svg")
    text = svg.read_text(encoding="utf-8", errors="replace")
    assert has_vector_marks(text) and "image/png" not in text
    assert quartz.render(tmp_path / "quartz.pdf").read_bytes()[:5] == b"%PDF-"
    with pytest.raises(ValueError, match="style"):
        quartz.render(tmp_path / "x.png", style="cartoon")
    with pytest.raises(ValueError, match="png"):
        quartz.render(tmp_path / "x.jpg")
    with pytest.raises(KeyError):
        quartz.render(tmp_path / "x.png", polyhedra=["Zz"])
