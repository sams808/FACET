"""FACET as a library, the MD model side, on the practice glass bundled with
FACET (20Na2O-10CaO-5Al2O3-65SiO2, 1200 atoms; a final frame as a LAMMPS data
file and six frames as a dump).

The wrappers are held to the engine called directly, and the model to what
is known of it from outside FACET: the atom counts it was built with, the
density LAMMPS printed, the stoichiometry of its non-bridging oxygens.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import facet

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"O": 700, "Si": 260, "Al": 40, "Na": 160, "Ca": 40}
FORMERS = ["Si", "Al"]

pytestmark = pytest.mark.skipif("glass" not in facet.examples(),
                                reason="the practice glass is not bundled")


@pytest.fixture(scope="module")
def glass():
    with facet.example("glass") as model:
        yield model


@pytest.fixture(scope="module")
def trajectory():
    with facet.example("glass-trajectory") as model:
        yield model


@pytest.fixture(scope="module")
def results(trajectory):
    return trajectory.analyse("glass", "rings", formers=FORMERS,
                              network={"ring_criterion": "king",
                                       "ring_max_size": 10})


# --- what was read ----------------------------------------------------------------------

def test_the_model_is_the_one_that_was_built(glass, trajectory):
    assert glass.n_atoms == trajectory.n_atoms == 1200
    assert glass.n_frames == 1 and trajectory.n_frames == 6
    assert glass.composition() == trajectory.composition() == COUNTS
    assert sorted(glass.elements) == sorted(COUNTS)
    assert glass.oxidation_states() == {"O": -2, "Si": 4, "Al": 3, "Na": 1, "Ca": 2}
    described = str(glass)
    assert "1200 atoms" in described and "type map (user)" in described
    assert repr(glass).startswith("<Model 'practice_glass.data'")
    # the density LAMMPS printed for its last step, from the log kept with
    # the model's input files
    assert glass.density() == pytest.approx(PRINTED_DENSITY, abs=0.01)
    frame = glass.frame()
    assert frame.n_atoms == 1200 and frame.box_is_diagonal
    with pytest.raises(IndexError):
        glass.frame(3)


def _printed_density() -> float:
    """g/cm3: the last thermo line of the log LAMMPS wrote for the run
    (step, temperature, energy, pressure, volume, density)."""
    log = ROOT / "docs" / "practice_glass" / "log.lammps"
    last = None
    for line in log.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if len(words) == 6 and words[0].isdigit():
            try:
                last = float(words[5])
            except ValueError:
                pass
    return last


PRINTED_DENSITY = _printed_density()


def test_a_lammps_file_names_its_elements_by_mass_or_by_a_type_map(tmp_path):
    path = facet.examples()["glass"]
    # the masses are the elements' own: the types are named from them
    with facet.model(path) as by_mass:
        assert by_mass.composition() == COUNTS
    # one mass written short (22.9 for sodium, as some scripts do): the type
    # can no longer be named, and the reader asks for the map
    short = tmp_path / "short_mass.data"
    short.write_text(path.read_text().replace("\n4 22.99\n", "\n4 22.9\n"))
    with pytest.raises(ValueError, match="type_map"):
        facet.model(short)
    with facet.model(short, type_map=facet.GLASS_TYPE_MAP) as mapped:
        assert mapped.composition() == COUNTS
    with pytest.raises(FileNotFoundError):
        facet.model(path.with_name("absent.data"))
    assert isinstance(facet.load(path), facet.Model)


# --- the catalogue of analyses ------------------------------------------------------------

def test_the_analyses_and_presets_are_the_engines():
    from facet.core import md_analysis as ma, md_presets

    listed = facet.analyses()
    assert list(listed.column("analysis")) == list(ma.ANALYSES)
    by = {row["analysis"]: row for row in listed}
    assert "formers" in by["rings"]["needs"]
    assert by["msd"]["runs"] == "whole trajectory"
    assert by["warren-cowley"]["reads the results of"] == "glass"
    assert list(facet.presets().column("preset")) == \
        [p.name for p in md_presets.PRESETS if not p.custom]


def test_a_request_is_checked_before_anything_is_read(glass):
    assert glass.missing("glass", formers=FORMERS) == []
    lacking = glass.missing("rings")
    assert any("formers" in line for line in lacking)
    with pytest.raises(ValueError, match="unknown analysis"):
        glass.request("no-such-analysis")
    with pytest.raises(ValueError, match="scattering.radiations"):
        glass.analyse("scattering", scattering={"radiations": ["xray"],
                                                "r_window": "Lorch"})
    with pytest.raises(ValueError, match="unknown request key"):
        glass.request("glass", former=["Si"])
    request = glass.request("glass", formers=FORMERS, frames=slice(0, 1), v_bond=0.1)
    assert request.v_bond_vu == 0.1 and request.formers == frozenset(FORMERS)
    runnable, left = glass.available(formers=FORMERS)
    assert "glass" in runnable and "msd" in left


# --- a run ------------------------------------------------------------------------------

def test_the_results_are_the_engines_run(trajectory, results):
    from facet.core import md_analysis as ma

    direct = ma.analyse(trajectory.trajectory, trajectory.request(
        "glass", "rings", formers=FORMERS,
        network={"ring_criterion": "king", "ring_max_size": 10}))
    assert results.analyses == list(direct.outputs) == ["glass", "rings"]
    assert results.failed == {}
    listed = results.descriptors()
    assert len(listed) == sum(1 for _ in direct.descriptors())
    name = "Qn Si (BV)"
    mine = results.get("glass", name)
    assert np.array_equal(mine.mean, direct.outputs["glass"].tables[name].mean)
    assert results["glass"][name] is mine
    assert type(results.raw("glass")).__name__ == "GlassResult"
    assert any("FACET" in line for line in results.provenance)
    assert len(results.find("qn")) >= 2
    with pytest.raises(KeyError, match="did not run"):
        results.get("msd", "anything")
    with pytest.raises(KeyError, match="no descriptor"):
        results.get("glass", "nothing by that name")


def test_the_speciation_is_the_stoichiometry_of_the_glass(results):
    """Outside check. Every Al is taken as a tetrahedral former compensated
    by one modifier charge, so the composition fixes the non-bridging
    oxygens: NBO = Na + 2 Ca - Al = 200 of 700, less two per tricluster or
    five-fold Al formed instead. And every former's Qn fractions sum to 1."""
    speciation = {row["key"]: row["mean (fraction)"]
                  for row in results.table("glass", "O speciation (BV)")}
    assert sum(speciation.values()) == pytest.approx(1.0)
    expected = (COUNTS["Na"] + 2 * COUNTS["Ca"] - COUNTS["Al"]) / COUNTS["O"]
    assert speciation["NBO"] + 2 * speciation["free"] == pytest.approx(expected, abs=0.03)
    for former in FORMERS:
        qn = results.table("glass", f"Qn {former} (BV)")
        assert qn.column("mean (fraction)").sum() == pytest.approx(1.0)
    cn = {row["key"]: row["mean (fraction)"]
          for row in results.table("glass", "CN Si (BV)")}
    assert cn[4] > 0.98
    # the frames one by one: one more column per frame, whose mean is the mean
    per_frame = results.table("glass", "Qn Si (BV)", per_frame=True)
    frames = [c for c in per_frame.columns if c.startswith("frame ")]
    assert len(frames) == 6
    for row in per_frame:
        assert np.mean([row[c] for c in frames]) == pytest.approx(row["mean (fraction)"])
    assert all(isinstance(row["key"], int) for row in per_frame)      # n of Qn, a number
    assert results.table("glass", "O speciation (BV)")[1]["key"] == "NBO"


def test_the_minima_and_the_export(results, tmp_path):
    minima = results.minima
    assert {"pair", "r_ang"} <= set(minima.columns)
    si_o = [row for row in minima if row["pair"] in ("Si-O", "O-Si")]
    assert si_o and 1.8 < si_o[0]["r_ang"] < 2.6
    workbook = results.export(tmp_path / "glass.xlsx")
    assert workbook[0].stat().st_size > 20_000
    files = results.export(tmp_path / "csv", analyses=["rings"])
    assert files and all(path.is_file() for path in files)
    names = [path.name for path in files if path.suffix == ".csv"]
    assert any(name.startswith("rings") for name in names)
    assert not any(name.startswith("glass") for name in names)


def test_analyses_of_time_need_a_time_axis_and_then_run(trajectory):
    with pytest.raises(ValueError, match="timestep_fs|frame_interval_ps"):
        trajectory.analyse("msd", dynamics={})
    found = trajectory.analyse("msd", frame_interval_ps=2.0)
    assert found.failed == {}
    assert any("msd" in row["descriptor"].lower() for row in found.descriptors())


def test_a_preset_becomes_the_keywords_of_analyse(glass):
    preset = glass.preset("Aluminosilicate")
    assert preset["formers"] == ["Al", "Si"] and "glass" in preset["analyses"]
    assert preset["network"]["ring_criterion"] == "king"
    names = [n for n in preset.pop("analyses") if n in ("glass", "components")]
    preset.pop("notes")
    found = glass.analyse(*names, **preset)
    assert found.analyses == ["glass", "components"]
    with pytest.raises(KeyError):
        glass.preset("No such preset")


# --- atom by atom -----------------------------------------------------------------------

def test_the_atom_table_and_one_atom_read_as_a_site(glass):
    atoms = glass.atoms()
    assert len(atoms) == 1200
    silicon = atoms.where(element="Si")
    assert silicon.column("CN").mean() == pytest.approx(4.0, abs=0.05)
    assert silicon.column("BVS / v.u.").mean() == pytest.approx(4.0, abs=0.35)
    number = silicon[0]["atom"]
    site = glass.site(number)
    assert site.element == "Si" and site.cn == silicon[0]["CN"]
    assert site.bvs == pytest.approx(silicon[0]["BVS / v.u."], rel=1e-9)
    assert site.phi == pytest.approx(silicon[0]["phi"], abs=1e-9)
    contacts = site.contacts()
    for row in contacts.where(bonded=True):
        assert row["valence / v.u."] == pytest.approx(
            math.exp((row["R0 / Å"] - row["distance / Å"]) / row["b / Å"]), rel=1e-9)
    assert len(site.angles()) == site.cn * (site.cn - 1) // 2
    assert len(site.plateaus()) >= 1 and site.cn_function()[1].max() >= site.cn
    with pytest.raises(ValueError, match="crystal"):
        site.radial_shells()
    with pytest.raises(IndexError):
        glass.site(5000)
    lower = glass.atoms(v_bond=0.3).where(element="Na").column("CN").mean()
    assert lower < atoms.where(element="Na").column("CN").mean()


def test_environments_and_the_cluster_written_about_an_atom(glass, tmp_path):
    groups, per_atom = glass.environments("Al", formers=FORMERS)
    assert len(per_atom) == COUNTS["Al"]
    assert sum(row["n"] for row in groups) == COUNTS["Al"]
    number = glass.atoms().where(element="Al")[0]["atom"]
    written = glass.write_environment(number, tmp_path / "al.xyz", shells=1)
    lines = written.read_text().splitlines()
    assert int(lines[0]) == len(lines) - 2 == glass.site(number).cn + 1
    wider = glass.write_environment(number, tmp_path / "al2.extxyz", shells=2)
    assert int(wider.read_text().splitlines()[0]) > int(lines[0])


def test_basicity_density_map_and_parity(glass, tmp_path):
    found = glass.basicity(formers=FORMERS)
    direct = facet.basicity.frame(facet.examples()["glass"], formers=FORMERS,
                                  type_map=facet.GLASS_TYPE_MAP)
    assert found.value == pytest.approx(direct.value, rel=1e-12)
    assert found.source == "practice_glass.data"
    composition = facet.basicity.composition("20Na2O-10CaO-5Al2O3-65SiO2")
    assert found.bulk == pytest.approx(composition.value, abs=1e-9)
    hot = glass.density_map(["Na", "Ca"], grid=2.0, sigma=1.5)
    assert hot.n_atoms == COUNTS["Na"] + COUNTS["Ca"]
    box = glass.frame().volume_ang3
    assert hot.mean_density == pytest.approx(hot.n_atoms / box, rel=0.02)
    # another code's table that agrees with FACET's own numbers
    atoms = glass.atoms()
    theirs = tmp_path / "theirs.csv"
    theirs.write_text("atom,CN,BVS\n" + "\n".join(
        f"{row['atom']},{row['CN']},{row['BVS / v.u.']:.6f}" for row in atoms))
    rows = glass.parity(theirs, k=0, one_based=True, tolerance=1e-4)
    assert len(rows) == 2 and all(row["fraction_within"] == 1.0 for row in rows), rows
    assert all(row["n_compared"] == 1200 for row in rows)


def test_the_descriptor_figures_are_written(results, tmp_path):
    pytest.importorskip("matplotlib")
    for name in ("Qn Si (BV)", "g O-Si"):
        figure = facet.plot.descriptor(results, "glass", name,
                                       save=tmp_path / f"{name[:2]}.png")
        assert figure.axes
    histogram = next(row for row in results.descriptors()
                     if row["kind"] == "histogram")
    facet.plot.descriptor(results.get(histogram["analysis"], histogram["descriptor"]),
                          save=tmp_path / "h.png")
    assert (tmp_path / "h.png").stat().st_size > 2000
    table = next(row for row in results.descriptors() if row["kind"] == "table")
    with pytest.raises(ValueError, match="results.table"):
        facet.plot.descriptor(results, table["analysis"], table["descriptor"])
