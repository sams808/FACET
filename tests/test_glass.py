"""Glass descriptors of MD models: what is pinned, and against what.

Every expected value here comes from outside the module under test: a fact
about a crystal (quartz is fully polymerised, eulytite is an orthosilicate,
cryolite holds AlF6 octahedra, BiB3O6 has BO3 and BO4 in the ratio 2:1), a
geometry built by hand with a bond length measured on quartz (never typed),
a second implementation in FACET written for another purpose
(``pdf.pair_distribution``, ``bulk.distance_cn``, ``utilities.bond_angles``,
``utilities.composition_summary``), the input's own statement of itself (each
CIF's ``_exptl_crystal_density_diffrn``), or a counting identity that must
hold whatever the structure (the bridges counted from the formers equal the
bridges counted from the anions).

* **Quartz**, 3 x 3 x 3 supercell, formers {'Si'} given explicitly: 100 % Q4,
  100 % BO, Si CN 4 and O CN 2 under the bond-valence and the distance
  definition alike, and every atom in the (4, 4) or (2, 2) cell of the
  BV-against-distance cross-table.
* **An isolated SiO4 in a large box**: Q0 and 4 NBO; **a corner-sharing
  dimer**: Q1 twice, 1 BO, 6 NBO. Exact integers. The Si-O length is the mean
  bond length of the quartz analysis.
* **A Q3 reference**: the reference collection was searched (``conftest``,
  252 files when this was written) for a silicate M_xSi2O5 whose only
  former is Si; none exists, so that test skips and says so. No structure is
  made up in its place.
* **The first minimum on a noisy g(r)**: a seeded Poisson-sampled g(r) at
  the pair counts of a 3 000-atom glass, and a quartz model jittered by
  0.08 Å, where each Si's 4th and 5th O distances (measured on the model)
  bound the gap the cutoff has to fall in. Judged against the counting error
  the valley rule lands in the floor; compared as they stand it lands on the
  noise, as it did on a real Na2O-3SiO2 model. ``g x pair_counts_per_unit_g``
  is checked against an independent pair count. The value and error at the
  point are g as measured, whatever the smoothing, and a smoothing kernel
  wider than the grid is refused before any frame, or (grid capped by the
  box) leaves that pair without a cutoff and with the reason.
* **Closed forms for Q^n(mX) and Q3**: an Si4O10 cage built from the quartz
  Si-O length and Si-O-Si angle (Q3(3Si) x 4), and, when the reference
  collection holds them, iodosodalite (Si Q4(4Al), Al Q4(4Si)) and Na5P3O10
  (Q1 : Q2 = 2 : 1).
* **Fractions** of every result sum to 1 (``md_stats.check_fractions``).
* **Two identical frames** give a spread of exactly 0 everywhere.
* **sum_n n N(Q^n)** equals the bridges counted from the anion side, on a
  random box with triclusters, where the identity is not trivial.
* **No verdict words** in any note, message or string of the module.

All synthetic geometries are test inputs, not reference values for any
material.
"""
from __future__ import annotations

import ast
import dataclasses
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import _index, sample_cif
from facet.core import (bulk, bv, coordination, glass, md_model, md_stats, pdf,
                        readers, utilities)
from facet.core.neighbors import NeighborFinder, search_radius_for
from facet.core.quality import parse_formula

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data" / "crystals"
QUARTZ = DATA / "quartz_SiO2_cod9013321.cif"

# Method choices for the tests, not physical values: the valley rule on g as
# measured, a difference counting when it exceeds 2 standard errors of g.
VALLEY = glass.MinimumMethod("valley", None, "midpoint", 2.0)
# The margin for noise-free arrays: the values compared as they stand.
EXACT = 0.0
# Bin widths for the tests: method choices, not physical values.
BINS = glass.HistogramBins(angle_deg=1.0, phi=0.01, plateau_decades=0.05,
                           length_ang=0.01)
TETRAHEDRAL_DEG = math.degrees(math.acos(-1.0 / 3.0))


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _crystal_frame(path: Path, reps=(2, 2, 2)):
    """A supercell frame with the crystal path's resolved states per row."""
    structure = readers.read(path)
    coordination.analyse_structure(structure, cations_only=False)
    frame, parent = md_model.supercell_frame(structure, reps)
    ox = np.array([structure.sites[structure.atoms[p].site_index].ox
                   for p in parent], dtype=np.int64)
    return structure, frame, parent, ox


def _bonds(frame, ox, v_bond=bv.V_BOND_DEFAULT):
    table, results = bulk.analyse_frame(frame, ox)
    return table, results, bulk.bonds_at(table, v_bond)


_QUARTZ_SI_O: list[float] = []


def _quartz_si_o_ang() -> float:
    """The mean Si-O bond length of the quartz analysis, Å (not typed)."""
    if not _QUARTZ_SI_O:
        _, frame, _, ox = _crystal_frame(QUARTZ)
        _, _, bonds = _bonds(frame, ox)
        _QUARTZ_SI_O.append(float(np.mean(bonds.d_ang)))
    return _QUARTZ_SI_O[0]


def _tetrahedron() -> np.ndarray:
    """Four unit vectors of a regular tetrahedron (geometry, no parameter)."""
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    return t / math.sqrt(3.0)


def _molecule_frame(symbols, cart_ang, *, box_side_ang):
    """A cubic box with a molecule at its centre."""
    cart = np.asarray(cart_ang, float)
    cart = cart - cart.mean(axis=0) + box_side_ang / 2.0
    return md_model.frame_from_arrays(list(symbols), cart,
                                      box_ang=np.eye(3) * box_side_ang)


def _isolated_tetrahedron(centre="Si"):
    d = _quartz_si_o_ang()
    cart = [np.zeros(3)] + [d * t for t in _tetrahedron()]
    # a box many bond lengths wide: a choice of test geometry
    return _molecule_frame([centre, "O", "O", "O", "O"], cart,
                           box_side_ang=16.0 * d)


def _dimer(first="Si", second="Si"):
    """Two tetrahedra sharing one corner, T-O-T linear."""
    d = _quartz_si_o_ang()
    t = _tetrahedron()
    a = np.zeros(3)
    bridge = d * t[0]
    b = 2.0 * d * t[0]
    cart = [a, b, bridge] + [a + d * t[k] for k in (1, 2, 3)] \
        + [b - d * t[k] for k in (1, 2, 3)]
    symbols = [first, second] + ["O"] * 7
    return _molecule_frame(symbols, cart, box_side_ang=16.0 * d)


def _random_box(side=8, spacing_ang=2.3, seed=7):
    """Si/O/Na on a jittered grid: nonsense chemistry, many triclusters."""
    rng = np.random.default_rng(seed)
    grid = np.indices((side,) * 3).reshape(3, -1).T.astype(float)
    cart = (grid + 0.5) * spacing_ang + rng.uniform(-0.1, 0.1, grid.shape)
    symbols = rng.choice(np.array(["Si", "O", "O", "Na"]), size=len(grid),
                         p=[0.25, 0.30, 0.30, 0.15])
    return md_model.frame_from_arrays(symbols, cart,
                                      box_ang=np.eye(3) * side * spacing_ang)


def _ox(frame):
    return md_model.model_oxidation(frame.species)


def _analyse(models, *, formers=frozenset({"Si"}), r_max=6.0, **kw):
    models = models if isinstance(models, list) else [models]
    traj = kw.pop("trajectory", None) or md_model.MemoryTrajectory(models)
    ox = kw.pop("ox", None) or _ox(models[0])
    return glass.analyse_trajectory(traj, ox, formers=formers,
                                    rdf_r_max_ang=r_max,
                                    minimum=kw.pop("minimum", VALLEY),
                                    bins=BINS, **kw)


# ---------------------------------------------------------------------------
# quartz: fully polymerised, by both definitions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def quartz_result():
    structure = readers.read(QUARTZ)
    frame, _ = md_model.supercell_frame(structure, (3, 3, 3))
    return _analyse([frame, frame])


def test_quartz_is_q4_and_fully_bridged_by_both_definitions(quartz_result):
    for definition in ("bv", "distance"):
        found = getattr(quartz_result, definition)
        assert found["Qn Si"].as_dict()[4] == (1.0, 0.0), definition
        assert all(m == 0.0 for k, (m, _) in found["Qn Si"].as_dict().items()
                   if k != 4)
        assert found["O speciation"].as_dict()["BO"] == (1.0, 0.0)
        assert found["CN Si"].as_dict() == {4: (1.0, 0.0)}, definition
        assert found["CN O"].as_dict() == {2: (1.0, 0.0)}, definition
        assert found["Qn(mSi) Si"].as_dict() == {"Q4(4Si)": (1.0, 0.0)}
        assert found["connectivity Si"].mean == 4.0


def test_quartz_bv_and_distance_cn_agree_atom_by_atom(quartz_result):
    """The cross-table of the two CN on the same atoms is reported."""
    table = quartz_result.comparison
    assert table["CN Si (BV, distance)"].as_dict() == {(4, 4): (1.0, 0.0)}
    assert table["CN O (BV, distance)"].as_dict() == {(2, 2): (1.0, 0.0)}
    assert table["CN Si BV-distance differences"].mean == 0.0


def test_quartz_distance_cutoff_comes_from_the_g_r_minimum(quartz_result):
    cutoff = quartz_result.cutoffs_ang[("Si", "O")]
    source = quartz_result.cutoff_sources[("Si", "O")]
    assert source.startswith("first minimum of the frame-averaged g_SiO(r)")
    minimum = quartz_result.minima[("Si", "O")]
    assert minimum.pair == ("Si", "O") and minimum.found
    assert quartz_result.minima[("O", "Si")].pair == ("O", "Si")
    assert minimum.r_ang == cutoff
    # below the cutoff lie exactly the Si-O distances the bond-valence
    # definition bonds, and the next Si-O distance lies above it
    _, frame, _, ox = _crystal_frame(QUARTZ, (3, 3, 3))
    pairs = bulk.find_pairs(frame, 6.0)
    si_o = (frame.elements[pairs.i] == "Si") & (frame.elements[pairs.j] == "O")
    distances = np.unique(pairs.d_ang[si_o])
    _, _, bonds = _bonds(frame, ox)
    assert np.array_equal(distances[distances <= cutoff],
                          np.unique(bonds.d_ang))
    assert distances[distances > cutoff].size
    assert quartz_result.provenance.method_parameters[
        "pair searches per frame"] == 2


def test_two_identical_frames_give_a_spread_of_exactly_zero(quartz_result):
    checked = 0
    for section, name, item in quartz_result.results():
        std = np.atleast_1d(np.asarray(item.std, dtype=np.float64))
        mean = np.atleast_1d(np.asarray(item.mean, dtype=np.float64))
        finite = np.isfinite(mean)
        assert (std[finite] == 0.0).all(), (section, name)
        checked += 1
    assert checked > 30


def test_every_fraction_set_sums_to_one(quartz_result):
    rows = 0
    for _, name, item in quartz_result.results():
        if isinstance(item, md_stats.Distribution) and item.kind == "fraction":
            md_stats.check_fractions(item.mean, name)
            for row in item.per_frame:
                md_stats.check_fractions(row, name)
                rows += 1
    assert rows > 10
    with pytest.raises(ValueError, match="sum to"):
        md_stats.check_fractions([0.5, 0.49], "doctored")


# ---------------------------------------------------------------------------
# hand-built units: exact integers
# ---------------------------------------------------------------------------

def test_an_isolated_tetrahedron_is_q0_with_four_nbo():
    frame = _isolated_tetrahedron()
    _, _, bonds = _bonds(frame, _ox(frame).per_atom(frame.elements))
    assert glass.qn_counts(bonds, frame, {"Si"}, {"O"}) == {"Si": {0: 1}}
    assert glass.anion_speciation(bonds, frame, {"Si"}, "O") == {
        "free": 0, "NBO": 4, "BO": 0, "tricluster": 0}
    angles = glass.bond_angles_deg(bonds, frame, "Si")[("O", "Si", "O")]
    assert angles.size == 6
    assert np.abs(angles - TETRAHEDRAL_DEG).max() < 1e-10


def test_a_corner_sharing_dimer_is_q1_twice_with_one_bo_and_six_nbo():
    frame = _dimer()
    _, _, bonds = _bonds(frame, _ox(frame).per_atom(frame.elements))
    assert glass.qn_counts(bonds, frame, {"Si"}, {"O"}) == {"Si": {1: 2}}
    assert glass.anion_speciation(bonds, frame, {"Si"}, "O") == {
        "free": 0, "NBO": 6, "BO": 1, "tricluster": 0}
    assert glass.qn_mx_counts(bonds, frame, {"Si"}, {"O"}) == {
        ("Si", "Si"): {"Q1(1Si)": 2}}
    assert glass.connectivity(bonds, frame, {"Si"}, {"O"}) == {
        "Si": 1.0, "all formers": 1.0}
    assert glass.linkage_counts(bonds, frame, "O") == {("Si", "O", "Si"): 1}
    tot = glass.bond_angles_deg(bonds, frame, "O", ends={"Si"})
    assert abs(tot[("Si", "O", "Si")][0] - 180.0) < 1e-10


def test_the_dimer_through_the_trajectory_gives_the_same_integers():
    result = _analyse(_dimer())
    qn = result.bv["Qn Si"]
    assert qn.keys == (0, 1) and qn.mean.tolist() == [0.0, 1.0]
    assert np.isnan(qn.std).all()                 # one frame: no spread
    counts = result.bv["O speciation"]
    assert counts.n_items.tolist() == [7]
    assert (counts.per_frame[0] * 7).tolist() == [0.0, 6.0, 1.0, 0.0]
    assert result.composition.net_charge_e == 2 * 4 - 7 * 2
    assert not result.composition.neutral


def test_si_o_al_and_al_o_al_linkages_and_qn_mx():
    frame = _dimer("Si", "Al")
    _, _, bonds = _bonds(frame, _ox(frame).per_atom(frame.elements))
    formers = {"Si", "Al"}
    assert glass.qn_counts(bonds, frame, formers, {"O"}) == {
        "Al": {1: 1}, "Si": {1: 1}}
    qnm = glass.qn_mx_counts(bonds, frame, formers, {"O"})
    assert qnm[("Si", "Al")] == {"Q1(1Al)": 1}
    assert qnm[("Si", "Si")] == {"Q1(0Si)": 1}
    assert qnm[("Al", "Si")] == {"Q1(1Si)": 1}
    assert glass.linkage_counts(bonds, frame, "O") == {
        ("Al", "O", "Al"): 0, ("Al", "O", "Si"): 1, ("Si", "O", "Si"): 0}
    frame = _dimer("Al", "Al")
    _, _, bonds = _bonds(frame, _ox(frame).per_atom(frame.elements))
    assert glass.linkage_counts(bonds, frame, "O") == {("Al", "O", "Al"): 1}


def test_a_halide_environment_is_labelled_by_its_bonded_cations():
    """F bonded to one Al and two Na, at cryolite's mean Al-F and Na-F."""
    _, frame, _, ox = _crystal_frame(DATA / "cryolite_Na3AlF6_cod9004097.cif")
    _, _, bonds = _bonds(frame, ox)
    lengths = glass.bond_lengths_ang(bonds, frame)
    d_al, d_na = float(lengths[("Al", "F")].mean()), \
        float(lengths[("Na", "F")].mean())
    cart = [np.zeros(3), np.array([d_al, 0, 0]),
            d_na * np.array([-0.5, math.sqrt(3) / 2, 0]),
            d_na * np.array([-0.5, -math.sqrt(3) / 2, 0])]
    unit = _molecule_frame(["F", "Al", "Na", "Na"], cart,
                           box_side_ang=16.0 * d_na)
    _, _, unit_bonds = _bonds(unit, _ox(unit).per_atom(unit.elements))
    assert glass.anion_environments(unit_bonds, unit, "F") == {"F-Al1Na2": 1}


# ---------------------------------------------------------------------------
# crystals with known polyhedra
# ---------------------------------------------------------------------------

def test_eulytite_an_orthosilicate_is_all_q0_and_nbo():
    _, frame, _, ox = _crystal_frame(DATA / "eulytite_Bi4SiO4_3_cod9012894.cif")
    _, results, bonds = _bonds(frame, ox)
    n_si = int((frame.elements == "Si").sum())
    n_o = int((frame.elements == "O").sum())
    assert glass.qn_counts(bonds, frame, {"Si"}, {"O"}) == {"Si": {0: n_si}}
    assert glass.anion_speciation(bonds, frame, {"Si"}, "O") == {
        "free": 0, "NBO": n_o, "BO": 0, "tricluster": 0}


def test_cryolite_holds_alf6_octahedra():
    _, frame, _, ox = _crystal_frame(DATA / "cryolite_Na3AlF6_cod9004097.cif")
    _, results, bonds = _bonds(frame, ox)
    n_al = int((frame.elements == "Al").sum())
    assert glass.aluminium_cn(results.cn, frame) == {
        "Al4": 0, "Al5": 0, "Al6": n_al, "other": 0}
    env = glass.anion_environments(bonds, frame, "F")
    al_bonds = sum(int(re.search(r"Al(\d+)", k).group(1)) * c
                   for k, c in env.items())
    assert al_bonds == 6 * n_al


BIB3O6 = Path(sample_cif("2104291", "2104291_BiB3O6.cif"))


@pytest.mark.skipif(not BIB3O6.exists(), reason="BiB3O6 reference absent")
def test_bib3o6_has_bo3_and_bo4_in_the_ratio_two_to_one():
    _, frame, _, ox = _crystal_frame(BIB3O6)
    _, results, _ = _bonds(frame, ox)
    n4 = glass.boron_n4(results.cn, frame)
    assert n4["other"] == 0 and n4["B3"] == 2 * n4["B4"] > 0


def test_an_si4o10_cage_is_q3_with_six_bo_and_four_nbo():
    """Four SiO4 sharing six corners, the adamantane-like Si4O10 unit, the
    Q3 closed form the collection lacks as a crystal: every Si Q3(3Si),
    connectivity 3, 6 BO, 4 NBO, 6 Si-O-Si linkages. Si sit at a(+-1, +-1,
    +-1) with an even number of minus signs, a bridging O on each axis at b,
    a terminal O radial from each Si. d is the quartz Si-O length and theta
    the mean quartz Si-O-Si angle, both measured on the quartz analysis;
    |Si - O| = d and cos theta = (d^2 - 4 a^2) / d^2 give a = d sqrt(1 -
    cos theta) / 2 and b = a + sqrt(d^2 - 2 a^2), so every Si-O-Si angle is
    theta by construction."""
    _, quartz, _, ox = _crystal_frame(QUARTZ)
    _, _, quartz_bonds = _bonds(quartz, ox)
    theta = float(glass.bond_angles_deg(quartz_bonds, quartz, "O", ends={"Si"})[
        ("Si", "O", "Si")].mean())
    d = _quartz_si_o_ang()
    a = d * math.sqrt(1.0 - math.cos(math.radians(theta))) / 2.0
    b = a + math.sqrt(d * d - 2.0 * a * a)
    si = [a * np.array(v, float) for v in ((1, 1, 1), (1, -1, -1), (-1, 1, -1),
                                           (-1, -1, 1))]
    bridges = [b * np.array(v, float) for v in np.vstack([np.eye(3),
                                                          -np.eye(3)])]
    terminal = [s + d * s / np.linalg.norm(s) for s in si]
    frame = _molecule_frame(["Si"] * 4 + ["O"] * 10, si + bridges + terminal,
                            box_side_ang=16.0 * d)
    _, results, bonds = _bonds(frame, _ox(frame).per_atom(frame.elements))
    assert results.cn[:4].tolist() == [4, 4, 4, 4]
    assert glass.qn_counts(bonds, frame, {"Si"}, {"O"}) == {"Si": {3: 4}}
    assert glass.anion_speciation(bonds, frame, {"Si"}, "O") == {
        "free": 0, "NBO": 4, "BO": 6, "tricluster": 0}
    assert glass.qn_mx_counts(bonds, frame, {"Si"}, {"O"}) == {
        ("Si", "Si"): {"Q3(3Si)": 4}}
    assert glass.connectivity(bonds, frame, {"Si"}, {"O"}) == {
        "Si": 3.0, "all formers": 3.0}
    assert glass.linkage_counts(bonds, frame, "O") == {("Si", "O", "Si"): 6}
    tot = glass.bond_angles_deg(bonds, frame, "O", ends={"Si"})[
        ("Si", "O", "Si")]
    assert tot.size == 6 and np.abs(tot - theta).max() < 1e-9


SODALITE = Path(sample_cif("4030257", "4030257_iodosodalite.cif"))
TRIPHOSPHATE = Path(sample_cif("0", "PDF-04-009-1422_Na5P3O10.cif"))


@pytest.mark.skipif(not SODALITE.exists(), reason="sodalite reference absent")
def test_sodalite_alternates_si_and_al_tetrahedra():
    """Iodosodalite's framework alternates SiO4 and AlO4 (structure type,
    every occupancy 1): Si Q4(4Al), Al Q4(4Si), no Si-O-Si or Al-O-Al, every
    O a BO. With Si alone as former, the same bonds make every Si Q0 and
    every O an NBO: the former set is chemistry the caller states."""
    _, frame, _, ox = _crystal_frame(SODALITE, (1, 1, 1))
    _, _, bonds = _bonds(frame, ox)
    n_si, n_al, n_o = (int((frame.elements == e).sum())
                       for e in ("Si", "Al", "O"))
    both = {"Si", "Al"}
    assert glass.qn_counts(bonds, frame, both, {"O"}) == {
        "Al": {4: n_al}, "Si": {4: n_si}}
    qnm = glass.qn_mx_counts(bonds, frame, both, {"O"})
    assert qnm[("Si", "Al")] == {"Q4(4Al)": n_si}
    assert qnm[("Al", "Si")] == {"Q4(4Si)": n_al}
    assert qnm[("Si", "Si")] == {"Q4(0Si)": n_si}
    assert qnm[("Al", "Al")] == {"Q4(0Al)": n_al}
    links = glass.linkage_counts(bonds, frame, "O")
    assert (links[("Al", "O", "Al")], links[("Si", "O", "Si")],
            links[("Al", "O", "Si")]) == (0, 0, n_o)
    assert glass.anion_speciation(bonds, frame, both, "O") == {
        "free": 0, "NBO": 0, "BO": n_o, "tricluster": 0}
    assert glass.qn_counts(bonds, frame, {"Si"}, {"O"}) == {"Si": {0: n_si}}
    assert glass.anion_speciation(bonds, frame, {"Si"}, "O")["NBO"] == n_o


@pytest.mark.skipif(not TRIPHOSPHATE.exists(),
                    reason="Na5P3O10 reference absent")
def test_a_triphosphate_chain_is_two_q1_to_one_q2():
    """Na5P3O10 holds P3O10 chains (structure type): two end P are Q1, the
    middle P is Q2, and each chain has 2 BO and 8 NBO."""
    _, frame, _, ox = _crystal_frame(TRIPHOSPHATE, (1, 1, 1))
    _, _, bonds = _bonds(frame, ox)
    n_p = int((frame.elements == "P").sum())
    assert n_p % 3 == 0
    assert glass.qn_counts(bonds, frame, {"P"}, {"O"}) == {
        "P": {1: 2 * n_p // 3, 2: n_p // 3}}
    assert glass.anion_speciation(bonds, frame, {"P"}, "O") == {
        "free": 0, "NBO": 8 * n_p // 3, "BO": 2 * n_p // 3, "tricluster": 0}


def _q3_candidates() -> list[Path]:
    """Silicates M_xSi2O5 (O/Si = 5/2) whose only former is Si."""
    found = []
    for path in sorted(set(_index().values())):
        text = Path(path).read_text(errors="replace")
        match = re.search(r"_chemical_formula_sum\s+(?:'([^']*)'|\"([^\"]*)\"|"
                          r"(\S[^\n]*))", text)
        formula = next((g for g in match.groups() if g), "") if match else ""
        counts = parse_formula(formula)
        if counts.get("Si", 0) and counts.get("O", 0) * 2 == counts["Si"] * 5 \
                and not set(counts) & {"Al", "B", "P", "Ge", "H"}:
            found.append(Path(path))
    return found


def test_a_q3_reference_if_the_collection_holds_one():
    candidates = _q3_candidates() if _index() else []
    if not candidates:
        pytest.skip("no Q3 silicate (M_xSi2O5 with Si the only former) in the "
                    "reference collection; none is made up in its place")
    for path in candidates:
        _, frame, _, ox = _crystal_frame(path)
        _, _, bonds = _bonds(frame, ox)
        n_si = int((frame.elements == "Si").sum())
        assert glass.qn_counts(bonds, frame, {"Si"}, {"O"}) == {
            "Si": {3: n_si}}, path.name


# ---------------------------------------------------------------------------
# counting identities on a random box with triclusters
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def random_frame():
    frame = _random_box()
    ox = _ox(frame).per_atom(frame.elements)
    table, results, bonds = _bonds(frame, ox)
    return frame, ox, table, results, bonds


def test_bond_cn_from_the_bonds_is_the_cn(random_frame):
    frame, _, _, results, bonds = random_frame
    assert np.array_equal(glass.bond_cn(bonds, frame.n_atoms), results.cn)


def test_qn_bridges_equal_the_bridges_counted_from_the_anions(random_frame):
    frame, _, _, _, bonds = random_frame
    formers = {"Si"}
    per_anion = glass.former_bond_counts(bonds, frame, formers)
    is_o = frame.elements == "O"
    bridging = per_anion[is_o & (per_anion >= 2)]
    assert (per_anion[is_o] >= 3).any(), "the box needs triclusters"
    qn = glass.qn_counts(bonds, frame, formers, {"O"})["Si"]
    assert sum(n * c for n, c in qn.items()) == int(bridging.sum())
    assert sum(qn.values()) == int((frame.elements == "Si").sum())
    # every pair of formers on a bridging anion is one m count from each end
    qnm = glass.qn_mx_counts(bonds, frame, formers, {"O"})[("Si", "Si")]
    m_total = sum(int(re.match(r"Q\d+\((\d+)Si\)", k).group(1)) * c
                  for k, c in qnm.items())
    assert m_total == int((bridging * (bridging - 1)).sum())
    spec = glass.anion_speciation(bonds, frame, formers, "O")
    assert sum(spec.values()) == int(is_o.sum())
    assert spec["BO"] + spec["tricluster"] == bridging.size


def test_linkages_and_environments_count_every_bond(random_frame):
    frame, _, _, results, bonds = random_frame
    is_o = frame.elements == "O"
    k = results.cn[is_o]
    links = glass.linkage_counts(bonds, frame, "O")
    assert sum(links.values()) == int((k * (k - 1) // 2).sum())
    env = glass.anion_environments(bonds, frame, "O")
    assert sum(env.values()) == int(is_o.sum())
    bonded = sum(sum(int(n) for n in re.findall(r"[A-Z][a-z]?(\d+)", key)) * c
                 for key, c in env.items())
    assert bonded == int(k.sum())
    assert "O-none" in env, "the box needs an anion bonded to no cation"


def test_distance_bonds_agree_with_bulk_distance_cn(random_frame):
    """Two implementations of the distance-cut CN on the same pairs."""
    frame, ox, _, _, _ = random_frame
    cutoffs = {("Si", "O"): 2.4, ("Na", "O"): 2.9}
    pairs = bulk.find_pairs(frame, 3.0)
    dist = glass.distance_bonds(frame, pairs, ox, cutoffs)
    cn = glass.bond_cn(dist, frame.n_atoms)
    reference = bulk.distance_cn(pairs, frame, {
        ("Si", "O"): 2.4, ("Na", "O"): 2.9, ("O", "Si"): 2.4,
        ("O", "Na"): 2.9})
    si, na, o = (frame.elements == e for e in ("Si", "Na", "O"))
    assert np.array_equal(cn[si], reference[("Si", "O")][si])
    assert np.array_equal(cn[na], reference[("Na", "O")][na])
    assert np.array_equal(cn[o], reference[("O", "Si")][o]
                          + reference[("O", "Na")][o])
    assert dist.missing == ()


def test_the_cross_table_matches_an_independent_count(random_frame):
    frame, ox, _, results, _ = random_frame
    result = _analyse(frame, cutoffs_ang={("Si", "O"): 2.4,
                                          ("Na", "O"): 2.9})
    pairs = bulk.find_pairs(frame, 3.0)
    reference = bulk.distance_cn(pairs, frame, {("Si", "O"): 2.4})
    si = frame.elements == "Si"
    expected = {}
    for a, b in zip(results.cn[si], reference[("Si", "O")][si]):
        expected[(int(a), int(b))] = expected.get((int(a), int(b)), 0) + 1
    got = result.comparison["CN Si (BV, distance)"]
    assert {k: round(m * si.sum()) for k, (m, _) in got.as_dict().items()
            if m} == expected
    assert result.comparison["CN Si BV-distance differences"].mean == sum(
        c for (a, b), c in expected.items() if a != b)
    assert result.provenance.method_parameters["pair searches per frame"] == 1
    assert result.cutoff_sources == {("Na", "O"): "user", ("Si", "O"): "user"}


# ---------------------------------------------------------------------------
# partial g(r): against pdf.pair_distribution and bulk.distance_cn
# ---------------------------------------------------------------------------

def test_partials_recombine_into_pdf_pair_distribution():
    """sum_ab c_a b_a b_b / <b>^2 R_ab, broadened as pdf.py broadens, is
    pdf.pair_distribution's R(r): its own pair search, its own weights."""
    structure = readers.read(QUARTZ)
    frame, _ = md_model.supercell_frame(structure, (3, 3, 3))
    ox = _ox(frame)
    crystal = md_model.frame_to_structure(frame, ox)
    r_max, dr = 6.0, 0.01
    grid, _ = glass.rdf_grid_ang(frame, r_max, dr)
    partial = glass.partial_rdf(frame, bulk.find_pairs(frame, grid[-1] + dr),
                                r_max_ang=r_max, dr_ang=dr)
    reference = pdf.pair_distribution(crystal, r_max=r_max, dr=dr, u_iso=0.0)
    weights = pdf.scattering_weights(crystal, "X-ray")
    n = frame.n_atoms
    total = np.zeros_like(grid)
    for (a, b), p in partial.items():
        deposit = p.g * 4 * math.pi * grid ** 2 * dr * p.n_centre \
            * p.n_neighbour / p.volume_ang3
        total += weights.b[a] * weights.b[b] * deposit
    radial = pdf._broaden(total, dr, pdf.MIN_SIGMA) / (n * weights.b_mean ** 2)
    inside = grid <= r_max - 0.1          # pdf.py stops its pairs at r_max
    assert np.array_equal(reference.r, grid)
    worst = np.abs(radial[inside] - reference.R[inside]).max()
    assert worst <= 1e-12 * reference.R.max(), worst


def test_running_cn_is_the_exact_distance_cn_at_each_grid_point():
    _, frame, _, ox = _crystal_frame(QUARTZ)
    grid, _ = glass.rdf_grid_ang(frame, 4.0)
    pairs = bulk.find_pairs(frame, grid[-1] + glass.RDF_DR_ANG)
    partial = glass.partial_rdf(frame, pairs, r_max_ang=4.0)
    assert partial[("Si", "O")].g is partial[("O", "Si")].g
    for k in (160, 161, 250, 330, len(grid) - 1):
        cut = float(grid[k])
        cn = bulk.distance_cn(pairs, frame, {("Si", "O"): cut,
                                             ("O", "Si"): cut})
        si, o = frame.elements == "Si", frame.elements == "O"
        assert partial[("Si", "O")].n_cum[k] == cn[("Si", "O")][si].sum() / si.sum()
        assert partial[("O", "Si")].n_cum[k] == cn[("O", "Si")][o].sum() / o.sum()


def test_the_grid_index_equals_searchsorted_ties_included():
    """The arithmetic index against numpy's binary search, on distances drawn
    at random, placed exactly on grid values, and outside the grid."""
    grid = np.arange(0.01, 10.0 + 0.005, 0.01)
    rng = np.random.default_rng(5)
    d = np.concatenate([rng.uniform(0.0, 10.2, 200_000), grid,
                        np.nextafter(grid, 0.0), np.nextafter(grid, 20.0),
                        [0.0, 0.004, 10.005, 10.02]])
    assert np.array_equal(glass._first_at_or_above(grid, d),
                          np.searchsorted(grid, d, side="left"))


def test_the_grid_stops_at_half_the_box_width_with_a_note():
    _, frame, _, _ = _crystal_frame(QUARTZ)
    grid, notes = glass.rdf_grid_ang(frame, 6.0)
    half = frame.perpendicular_widths_ang.min() / 2
    assert grid[-1] <= half < grid[-1] + glass.RDF_DR_ANG
    assert notes and "half the smallest perpendicular width" in notes[0]
    with pytest.raises(ValueError, match="searched to"):
        glass.partial_rdf(frame, bulk.find_pairs(frame, 3.0), r_max_ang=6.0)


# ---------------------------------------------------------------------------
# the first minimum
# ---------------------------------------------------------------------------

def test_a_flat_floor_gives_its_first_point_or_its_midpoint():
    r = 0.01 * np.arange(1, 10)
    g = np.array([0, 0, 5, 0.5, 0, 0, 0, 2, 1], float)
    for rule in glass.MINIMUM_RULES:
        first = glass.first_minimum(r, g, glass.MinimumMethod(
            rule, None, "first", EXACT))
        middle = glass.first_minimum(r, g, glass.MinimumMethod(
            rule, None, "midpoint", EXACT))
        assert (first.index, middle.index) == (4, 5), rule
        assert middle.r_ang == r[5] and middle.g_value == 0.0
        assert middle.floor_index == (4, 6) and middle.level == 1.0
        assert middle.g_std_error is None            # no pair counts given
        assert "taken as exact" in middle.method


def test_a_fragmented_floor_is_one_floor():
    """Zeros separated by small values (single pairs in the tail of a Si-O
    peak) are one floor from its first zero to its last: the midpoint is the
    middle of that span, not the middle of the first run of zeros (which, as
    a single point here, put the cutoff at its first zero). With pair counts,
    a point holding one or two pairs is within 2 standard errors of a zero
    and joins the floor."""
    r = 0.01 * np.arange(1, 13)
    g = np.array([0, 3, 0.5, 0, 0.01, 0, 0, 0.02, 0, 0.4, 2, 1], float)
    exact = glass.first_minimum(r, g, glass.MinimumMethod(
        "valley", None, "midpoint", EXACT))
    assert exact.floor_index == (3, 8) and exact.index == 5
    first = glass.first_minimum(r, g, glass.MinimumMethod(
        "valley", None, "first", EXACT))
    assert first.index == 3
    per_unit = np.full(r.shape, 100.0)        # g = 0.01 is one pair
    counted = glass.first_minimum(r, g, VALLEY, pairs_per_unit_g=per_unit)
    assert counted.floor_index == (3, 8) and counted.index == 5
    assert counted.g_std_error == 0.0                  # a zero holds no pair
    # depth: (1 - 0) / sqrt(1 / 100) standard errors below the level
    assert counted.depth_std_errors == pytest.approx(10.0, rel=1e-12)


def test_the_valley_rule_reads_past_a_wiggle_above_one():
    """A local minimum at g = 1.2 is not a valley: the two rules differ, and
    the first local minimum above the level carries a note saying so."""
    r = 0.01 * np.arange(1, 10)
    g = np.array([0, 3, 1.2, 1.3, 0.4, 0.6, 0.3, 1.5, 1.0])
    valley = glass.first_minimum(r, g, glass.MinimumMethod("valley", None,
                                                           "first", EXACT))
    local = glass.first_minimum(r, g, glass.MinimumMethod(
        "first local minimum", None, "first", EXACT))
    assert valley.index == 6 and local.index == 2
    assert valley.notes == ()
    assert local.notes and "above its uncorrelated level 1" in local.notes[0]


def test_no_minimum_comes_with_its_reason():
    r = 0.01 * np.arange(1, 8)
    method = glass.MinimumMethod("valley", None, "first", EXACT)
    flat = glass.first_minimum(r, np.full(7, 0.5), method, pair=("Si", "O"))
    assert not flat.found and "does not exceed 1" in flat.reason
    open_valley = glass.first_minimum(r, np.array([0, 4, 0, 0, 0, 0, 0.]),
                                      method)
    assert open_valley.r_ang is None and "no upper end" in open_valley.reason


def test_smoothing_is_named_and_keeps_a_constant_constant():
    flat = np.full(50, 1.7)
    assert np.allclose(glass._smooth(flat, 0.01, 0.03), 1.7, rtol=0,
                       atol=1e-15)
    with pytest.raises(ValueError, match="more than"):
        glass._smooth(flat, 0.01, 1.0)
    text = glass.MinimumMethod("valley", 0.03, "first", EXACT).describe()
    assert "sigma 0.03" in text and "compared as they stand" in text
    assert "2 standard error(s)" in VALLEY.describe()
    with pytest.raises(ValueError):
        glass.MinimumMethod("lowest", None, "first", EXACT)
    with pytest.raises(ValueError):
        glass.MinimumMethod("valley", -0.1, "first", EXACT)
    for margin in (-1.0, float("nan"), float("inf"), True, None):
        with pytest.raises(ValueError, match="margin_std_errors"):
            glass.MinimumMethod("valley", None, "first", margin)
    with pytest.raises(TypeError):                   # no default margin
        glass.MinimumMethod("valley", None, "first")


def test_smoothing_carries_the_variance_through_the_squared_kernel():
    """Away from the grid ends, a constant variance v smoothed by weights w
    is v sum(w^2) / (sum w)^2, the variance of a weighted mean of
    independent points; the ends, with fewer points, keep more of it."""
    step, sigma = 0.01, 0.03
    var = np.full(200, 0.25)
    out = glass._smooth_variance(var, step, sigma)
    half = int(math.ceil(glass.SMOOTH_HALF_WIDTH_SIGMAS * sigma / step))
    w = np.exp(-0.5 * ((np.arange(-half, half + 1) * step) / sigma) ** 2)
    inner = slice(half, 200 - half)
    assert np.allclose(out[inner], 0.25 * (w ** 2).sum() / w.sum() ** 2,
                       rtol=1e-12, atol=0)
    assert out[0] > out[100]


def test_an_isolated_molecule_has_no_cutoff_and_says_why():
    """Four Si-O pairs in a box 16 bond lengths wide: at 2 standard errors
    the one Si-O distance, holding at most 4 pairs, is not above the level
    (it would need more than about 4), and a run of empty grid points is not
    below it either (fewer than 4 pairs would be expected there at g = 1).
    Compared as they stand, the valley has no upper end."""
    frame = _isolated_tetrahedron()
    result = _analyse(frame)
    assert ("Si", "O") not in result.cutoffs_ang
    assert any("no distance cutoff for Si-O" in n
               and "by more than 2 standard error(s)" in n
               for n in result.notes)
    assert result.distance == {} and result.comparison == {}
    assert result.bv["Qn Si"].as_dict()[0][0] == 1.0
    exact = _analyse(frame, minimum=glass.MinimumMethod(
        "valley", None, "midpoint", EXACT))
    assert any("no distance cutoff for Si-O" in n and "no upper end" in n
               for n in exact.notes)


def test_a_user_cutoff_overrides_the_rule_and_makes_one_pass():
    structure = readers.read(QUARTZ)
    frame, _ = md_model.supercell_frame(structure, (3, 3, 3))
    result = _analyse(frame, cutoffs_ang={("O", "Si"): 2.0})
    assert result.cutoffs_ang == {("Si", "O"): 2.0}
    assert result.cutoff_sources == {("Si", "O"): "user"}
    # minima agree with the cutoffs used, and keep the rule's own point
    for pair in (("Si", "O"), ("O", "Si")):
        given = result.minima[pair]
        assert given.source == "user" and given.r_ang == 2.0, pair
        assert given.pair == pair
        assert "given by the user; the rule's own point is" in given.notes[0]
    assert result.minima[("Si", "Si")].source == "auto"
    cn = result.distance["CN Si"]
    assert cn.keys == (4,) and cn.mean.tolist() == [1.0]
    assert result.provenance.method_parameters["pair searches per frame"] == 1
    with pytest.raises(ValueError, match="two cations"):
        _analyse(frame, cutoffs_ang={("Si", "Si"): 3.0})
    with pytest.raises(ValueError, match="two different cutoffs"):
        _analyse(frame, cutoffs_ang={("Si", "O"): 2.0, ("O", "Si"): 2.1})


# A g(r) with what a sodium silicate's g_NaO holds: a first peak at 2.37 Å, a
# valley whose floor, 0.6, lies at 3.25 Å, and a second shell. Test inputs,
# not reference values for any material; the noise is Poisson counting at the
# pair counts of a 3 000-atom model (500 Na, 1 750 O, 41 500 Å^3), as in the
# models the module docstring reports.
_NOISY_R = np.arange(0.01, 8.0 + 0.005, 0.01)


def _noisy_truth(r_ang):
    r_ang = np.asarray(r_ang, float)
    core = 1.0 / (1.0 + np.exp(-(r_ang - 2.15) / 0.03))
    return core * (1.0 + 3.5 * np.exp(-(r_ang - 2.37) ** 2 / (2 * 0.12 ** 2))
                   - 0.4 * np.exp(-(r_ang - 3.25) ** 2 / (2 * 0.25 ** 2))
                   + 0.25 * np.exp(-(r_ang - 4.4) ** 2 / (2 * 0.3 ** 2)))


def _noisy_g(seed: int, n_frames: int):
    per_unit = glass.pair_counts_per_unit_g(_NOISY_R, 0.01, 500, 1750,
                                            41500.0, same_element=False,
                                            n_frames=n_frames)
    rng = np.random.default_rng(seed)
    counts = rng.poisson(_noisy_truth(_NOISY_R) * per_unit)
    return counts / per_unit, per_unit


def test_a_noisy_valley_is_read_in_its_floor_not_at_a_crossing():
    """Over 20 seeds, on one frame's counts and on 20 frames': at 2 standard
    errors the valley rule lands where the noise-free g is below 0.7 (the
    floor; it is 1 at about 2.75 Å), and the floor it reports holds the
    noise-free minimum. Compared as they stand, the same arrays put the point
    at a noise crossing of 1 (noise-free g above 1 there) on some seeds: the
    failure seen on a real Na2O-3SiO2 model, Na-O at 2.77 Å instead of the
    floor at 3.0-3.4 Å."""
    exact = glass.MinimumMethod("valley", None, "midpoint", EXACT)
    for n_frames in (1, 20):
        at_crossing = []
        for seed in range(20):
            g, per_unit = _noisy_g(seed, n_frames)
            found = glass.first_minimum(_NOISY_R, g, VALLEY,
                                        pairs_per_unit_g=per_unit)
            assert found.found, (n_frames, seed)
            assert _noisy_truth(found.r_ang) < 0.7, (n_frames, seed)
            assert found.floor_r_ang[0] <= 3.25 <= found.floor_r_ang[1]
            assert found.depth_std_errors > 2.0
            plain = glass.first_minimum(_NOISY_R, g, exact,
                                        pairs_per_unit_g=per_unit)
            at_crossing.append(float(_noisy_truth(plain.r_ang)))
        assert max(at_crossing) > 1.0, n_frames


def test_a_gap_is_judged_by_all_the_pairs_it_lacks():
    """Two distances of one shell 3 grid points apart, a gap of 41 empty
    points, then the next shell; 1.5 pairs per grid point at g = 1, the
    count of a small crystal cell. No empty point alone is 2 standard errors
    below 1 (that needs more than 4 pairs expected), but the 41-point gap
    lacks 61.5 pairs, 7.8 standard errors: it is the valley. The 2 empty
    points inside the shell lack 3, 1.7 standard errors: they are not.
    Compared as they stand, those 2 points are read as the valley."""
    r = 0.01 * np.arange(1, 61)
    per_unit = np.full(60, 1.5)
    g = np.zeros(60)
    g[[5, 8, 50]] = 20 / 1.5                   # 20 pairs at each distance
    found = glass.first_minimum(r, g, VALLEY, pairs_per_unit_g=per_unit)
    assert found.floor_index == (9, 49) and found.index == 29
    exact = glass.first_minimum(r, g, glass.MinimumMethod(
        "valley", None, "midpoint", EXACT), pairs_per_unit_g=per_unit)
    assert exact.floor_index == (6, 7)
    # the floor begins where the shell ends: a point of one or two pairs
    # (g 1.05, 0.05 above 1, under 2 standard errors) between short empty
    # stretches is inside the gap, and the gap starts after the last point
    # above 1, not where the first held stretch below 1 starts
    g = np.zeros(60)
    g[[5, 50]] = 20 / 1.5
    g[8] = 1.05
    found = glass.first_minimum(r, g, glass.MinimumMethod(
        "valley", None, "first", 2.0), pairs_per_unit_g=per_unit)
    assert found.floor_index == (6, 49) and found.index == 6


def test_a_small_crystal_cell_keeps_its_shells_whole():
    """Cryolite 2 x 2 x 2, one frame. Al has six F at 1.81 Å and none
    nearer than 3.97 Å, Na-F distances begin at 2.227 and 2.263 Å (measured
    on the model here). At 2 standard errors the Al-F cutoff falls in the
    Al-F gap (every Al CN 6 by distance, as by bond valence) and the Na-F
    cutoff above both of the first two Na-F distances, so no Na loses all
    its F; compared as they stand the Na-F cutoff fell between them and two
    Na in three had no F."""
    _, frame, _, ox = _crystal_frame(DATA / "cryolite_Na3AlF6_cod9004097.cif")
    pairs = bulk.find_pairs(frame, 4.0)

    def distances(a, b):
        pick = (frame.elements[pairs.i] == a) & (frame.elements[pairs.j] == b)
        return np.unique(pairs.d_ang[pick])

    al_f, na_f = distances("Al", "F"), distances("Na", "F")
    result = _analyse(frame, formers={"Al"}, ox=_ox(frame), r_max=5.0)
    cut_al = result.cutoffs_ang[("Al", "F")]
    assert al_f[al_f < cut_al].max() < 1.9 < 3.9 < al_f[al_f > cut_al].min()
    assert result.distance["CN Al"].keys == (6,)
    assert result.cutoffs_ang[("Na", "F")] > na_f[1]
    assert 0 not in result.distance["CN Na"].keys
    exact = _analyse(frame, formers={"Al"}, ox=_ox(frame), r_max=5.0,
                     minimum=glass.MinimumMethod("valley", None, "midpoint",
                                                 EXACT))
    assert na_f[0] < exact.cutoffs_ang[("Na", "F")] < na_f[1]
    assert exact.distance["CN Na"].as_dict()[0][0] == pytest.approx(2 / 3)


def test_the_first_local_minimum_rule_on_a_noisy_g():
    """The textbook rule follows the local shape of g, so on noise it needs
    smoothing as well as a margin: smoothed by 0.03 Å and judged at 2
    standard errors it lands in the floor on every seed, one frame or 20;
    with the same smoothing compared as they stand it stops on a wiggle of
    the falling flank (noise-free g above 0.8) on some seeds."""
    judged = glass.MinimumMethod("first local minimum", 0.03, "midpoint", 2.0)
    plain = glass.MinimumMethod("first local minimum", 0.03, "midpoint", EXACT)
    for n_frames in (1, 20):
        flank = []
        for seed in range(20):
            g, per_unit = _noisy_g(seed, n_frames)
            found = glass.first_minimum(_NOISY_R, g, judged,
                                        pairs_per_unit_g=per_unit)
            assert _noisy_truth(found.r_ang) < 0.7, (n_frames, seed)
            other = glass.first_minimum(_NOISY_R, g, plain,
                                        pairs_per_unit_g=per_unit)
            flank.append(float(_noisy_truth(other.r_ang)))
        assert max(flank) > 0.8, n_frames


def test_g_times_the_pair_counts_per_unit_g_counts_the_pairs():
    """The deposit is linear: a pair at d between grid points gives weights
    summing to 1, and one in the last step past the grid gives the share
    that lands on the last point. So the sum over the grid of g times
    pair_counts_per_unit_g equals that count over the pairs of an independent
    search, for a pair of two elements and (halved, each pair deposited
    from both ends) for a pair of one element."""
    _, frame, _, _ = _crystal_frame(QUARTZ)
    grid, _ = glass.rdf_grid_ang(frame, 4.0)
    dr = glass.RDF_DR_ANG
    pairs = bulk.find_pairs(frame, grid[-1] + dr)
    partial = glass.partial_rdf(frame, pairs, r_max_ang=4.0)
    for a, b in (("Si", "O"), ("Si", "Si"), ("O", "O")):
        p = partial[(a, b)]
        counted = float((p.g * p.pair_counts_per_unit_g).sum())
        pick = (frame.elements[pairs.i] == a) & (frame.elements[pairs.j] == b)
        d = pairs.d_ang[pick]
        weight = np.where(d <= grid[-1], 1.0, 1.0 - (d - grid[-1]) / dr)
        expected = float(weight.sum()) / (2.0 if a == b else 1.0)
        assert counted == pytest.approx(expected, rel=1e-12), (a, b)
        level = (p.n_centre - 1) / p.n_centre if a == b else 1.0
        assert p.uncorrelated_level == level
    with pytest.raises(ValueError, match="one atom count"):
        glass.pair_counts_per_unit_g(grid, dr, 3, 4, 100.0,
                                     same_element=True, n_frames=1)


def test_a_dilute_pair_of_one_element_is_read_against_its_own_level():
    """g_aa of a species of 20 atoms tends to 0.95 = 1 - 1/20, not 1, so its
    second shell may never exceed 1: read against 1 the valley has no upper
    end; read against 0.95 it is found. analyse_trajectory takes the level
    from the model's own atom count."""
    r = 0.01 * np.arange(1, 11)
    g = np.array([0, 0, 3, 0.3, 0.2, 0.97, 0.98, 0.95, 0.95, 0.95])
    method = glass.MinimumMethod("valley", None, "first", EXACT)
    against_one = glass.first_minimum(r, g, method)
    assert not against_one.found and "no upper end" in against_one.reason
    against_own = glass.first_minimum(r, g, method, level=1 - 1 / 20)
    assert against_own.index == 4 and against_own.level == 0.95


def test_like_pair_minima_carry_the_level_of_the_model(quartz_result):
    n_si = 81                                  # 3 x 3 x 3 cells of 3 Si
    si_si = quartz_result.minima[("Si", "Si")]
    assert si_si.level == (n_si - 1) / n_si
    assert quartz_result.minima[("Si", "O")].level == 1.0
    source = quartz_result.cutoff_sources[("Si", "O")]
    # the gap of a crystal: N_SiO is 4 across the whole floor
    low, high = quartz_result.minima[("Si", "O")].floor_r_ang
    assert f"its floor runs {low:.6g}-{high:.6g} Å" in source
    assert "across which N_SiO(r) runs 4-4" in source
    assert "standard errors below 1" in source


def test_a_jittered_quartz_model_gets_its_cutoff_in_the_si_o_gap():
    """Quartz with every atom moved by a Gaussian of 0.08 Å (seed 1): on one
    frame its g_SiO at 0.01 Å is noisy, as one MD frame's is. Each Si keeps
    four O within 1.94 Å and has its fifth beyond 3.18 Å (measured on the
    model here, not typed), so a cutoff in that gap gives every Si CN 4 by
    distance. At 2 standard errors the valley rule lands in the gap;
    compared as they stand it stops on the noise of the rising edge of the
    peak, inside the first shell, and most Si get no bond at all."""
    structure = readers.read(QUARTZ)
    base, _ = md_model.supercell_frame(structure, (3, 3, 3))
    rng = np.random.default_rng(1)
    frame = md_model.frame_from_arrays(
        base.elements, base.cart_ang + rng.normal(0, 0.08, base.cart_ang.shape),
        box_ang=base.box_ang)
    pairs = bulk.find_pairs(frame, 4.0)
    si_o = (frame.elements[pairs.i] == "Si") & (frame.elements[pairs.j] == "O")
    centre, d = pairs.i[si_o], pairs.d_ang[si_o]
    order = np.lexsort((d, centre))
    _, start = np.unique(centre[order], return_index=True)
    fourth = d[order[start + 3]].max()                 # each Si's 4th O
    fifth = d[order[start + 4]].min()                  # each Si's 5th O
    result = _analyse(frame, r_max=5.0)
    cutoff = result.cutoffs_ang[("Si", "O")]
    assert fourth < cutoff < fifth, (fourth, cutoff, fifth)
    cn = result.distance["CN Si"]
    assert cn.keys == (4,) and cn.mean.tolist() == [1.0]
    assert any("rest on the pair counts of one frame" in n
               for n in result.notes)
    plain = _analyse(frame, r_max=5.0, minimum=glass.MinimumMethod(
        "valley", None, "midpoint", EXACT))
    assert plain.cutoffs_ang[("Si", "O")] < fourth
    assert plain.distance["CN Si"].as_dict()[0][0] > 0.5


def test_the_value_and_error_at_the_point_are_g_as_measured():
    """Smoothed by 0.03 Å the rule reads the smoothed g, but ``g_value`` is g
    as measured at the point, so ``g_std_error`` is its Poisson error,
    sqrt(g / pairs per unit g). The smoothed g's error, given before beside
    the unsmoothed value, is about a third of it (sum w^2 / (sum w)^2 of a
    3-step Gaussian, 0.094, is about 1 / 3^2)."""
    method = glass.MinimumMethod("valley", 0.03, "midpoint", 2.0)
    for seed in range(5):
        g, per_unit = _noisy_g(seed, 1)
        found = glass.first_minimum(_NOISY_R, g, method,
                                    pairs_per_unit_g=per_unit)
        k = found.index
        assert found.g_value == g[k] > 0.0
        raw = math.sqrt(g[k] / per_unit[k])
        assert found.g_std_error == pytest.approx(raw, rel=1e-15)
        smoothed = math.sqrt(glass._smooth_variance(g / per_unit, 0.01,
                                                    0.03)[k])
        assert smoothed < 0.5 * raw


def test_a_lowest_point_above_the_level_is_said_to_lie_above_it():
    """The first-local-minimum rule can stop above the level (between two
    parts of a split first peak). The cutoff source then states the depth as
    a distance above the level, where it wrote '-10 standard errors below
    1'. Here g = 1.5 at that point, 400 pairs per unit g: (1 - 1.5) /
    sqrt(1 / 400) = -10."""
    r = 0.01 * np.arange(1, 10)
    g = np.array([0, 3, 1.5, 2.5, 0.4, 0.6, 0.3, 1.5, 1.0])
    found = glass.first_minimum(r, g, glass.MinimumMethod(
        "first local minimum", None, "first", 2.0), pair=("B", "O"),
        pairs_per_unit_g=np.full(9, 400.0))
    assert found.index == 2
    assert found.depth_std_errors == pytest.approx(-10.0, rel=1e-12)
    assert "above its uncorrelated level 1" in found.notes[0]
    source = glass._auto_source(found, np.arange(9.0))
    assert source.endswith("its lowest point lies 10 standard errors above 1")
    below = glass.first_minimum(r, g, glass.MinimumMethod(
        "valley", None, "first", 2.0), pair=("B", "O"),
        pairs_per_unit_g=np.full(9, 400.0))
    assert below.depth_std_errors > 0
    assert glass._auto_source(below, np.arange(9.0)).endswith(
        f"lies {below.depth_std_errors:.3g} standard errors below 1")


def test_a_smoothing_wider_than_the_capped_grid_gives_a_reason():
    """Quartz 2 x 2 x 2: the g(r) grid stops at half the box width (about
    425 points) below the 1 000 points of 10 Å asked for. A kernel of 0.5 Å
    (601 points) fits the grid asked for but not the capped one: that pair
    gets no cutoff with the reason, and the rest of the analysis is
    reported, where first_minimum's refusal ended the whole run after pass
    1."""
    structure = readers.read(QUARTZ)
    frame, _ = md_model.supercell_frame(structure, (2, 2, 2))
    result = _analyse(frame, r_max=10.0, minimum=glass.MinimumMethod(
        "valley", 0.5, "midpoint", 2.0))
    n_points = result.rdf[("Si", "O")].axis.size
    assert n_points < 601 < 1000
    assert ("Si", "O") not in result.cutoffs_ang
    reason = result.minima[("Si", "O")].reason
    assert (f"the smoothing sigma of 0.5 Å spans 601 grid points, more than "
            f"the {n_points} of the g(r) grid common to the frames used"
            ) in reason
    assert any("no distance cutoff for Si-O" in n for n in result.notes)
    assert result.bv["CN Si"].keys == (4,) and result.distance == {}


# ---------------------------------------------------------------------------
# angles against the crystal path
# ---------------------------------------------------------------------------

def test_angles_equal_the_crystal_path_atom_by_atom():
    """utilities.bond_angles (acos, NeighborFinder) on each parent atom."""
    structure, frame, parent, ox = _crystal_frame(QUARTZ)
    _, _, bonds = _bonds(frame, ox)
    rmax = search_radius_for(structure, bv.DEFAULT, bv.V_LIST_DEFAULT)
    finder = NeighborFinder(structure, rmax=rmax)
    expected = {}
    for k in range(structure.n_atoms):
        result = coordination.analyse_site(structure, finder.contacts(k),
                                           bv.DEFAULT, bv.V_BOND_DEFAULT,
                                           bv.V_LIST_DEFAULT)
        expected[k] = sorted(a.angle for a in utilities.bond_angles(result))
    for centre in ("Si", "O"):
        rows = np.flatnonzero(frame.elements == centre)
        sample = rows[:: max(1, rows.size // 12)]
        for row in sample:
            one = dataclasses.replace(
                bonds, **{f: getattr(bonds, f)[
                    (bonds.cation == row) | (bonds.anion == row)]
                    for f in ("cation", "anion", "image", "vec_ang", "d_ang",
                              "v_vu")})
            got = sorted(np.concatenate(list(glass.bond_angles_deg(
                one, frame, centre).values())).tolist())
            want = expected[int(parent[row])]
            assert len(got) == len(want)
            assert max(abs(a - b) for a, b in zip(got, want)) < 1e-9


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(DATA.glob("*.cif")),
                         ids=lambda p: p.name.split("_")[0])
def test_density_matches_the_density_each_cif_states(path):
    """_exptl_crystal_density_diffrn is rounded to 1e-3 and was computed with
    the atomic weights of its day; measured agreement 0.7e-4 to 1.5e-4
    relative, so 2e-4 relative is allowed."""
    stated = float(re.search(r"_exptl_crystal_density_diffrn\s+([0-9.]+)",
                             path.read_text()).group(1))
    structure = readers.read(path)
    frame, _ = md_model.supercell_frame(structure, (1, 1, 1))
    assert abs(glass.density_g_per_cm3(frame) - stated) <= 2e-4 * stated


def test_composition_matches_composition_summary_and_the_oxide_basis():
    structure = readers.read(QUARTZ)
    frame, _ = md_model.supercell_frame(structure, (2, 2, 2))
    ox = _ox(frame)
    model = glass.composition(frame, ox, {"Si": "SiO2"})
    summary = utilities.composition_summary(
        md_model.frame_to_structure(frame, ox))
    assert model.counts == {k: int(v) for k, v in summary["counts"].items()}
    for element, value in summary["atomic_percent"].items():
        assert abs(model.atomic_percent[element] - value) < 1e-12
    assert model.net_charge_e == 0 and model.neutral
    assert summary["net_charge"] == 0
    assert model.oxide_mol_percent == {"SiO2": 100.0}
    assert model.anion_unassigned == {"O": 0.0} and model.notes == ()


def test_oxide_mol_percent_is_exact_and_leftovers_are_noted():
    rng = np.random.default_rng(3)
    symbols = ["Na"] * 4 + ["Si"] * 4 + ["O"] * 10
    frame = md_model.frame_from_arrays(symbols, frac=rng.random((18, 3)),
                                       box_ang=np.eye(3) * 10.0)
    model = glass.composition(frame, _ox(frame),
                              {"Na": "Na2O", "Si": "SiO2"})
    assert model.oxide_mol_percent["Na2O"] == pytest.approx(100 / 3, abs=1e-12)
    assert model.anion_unassigned == {"O": 0.0} and model.neutral
    dimer = _dimer()
    leftover = glass.composition(dimer, _ox(dimer), {"Si": "SiO2"})
    assert leftover.anion_unassigned == {"O": 3.0}
    assert any("not neutral" in n for n in leftover.notes)
    assert any("left unassigned" in n for n in leftover.notes)
    for basis, message in (({"Si": "SiO2"}, "no formula for Na"),
                           ({"Na": "SiO2", "Si": "SiO2"}, "holds no Na"),
                           ({"O": "O2", "Na": "Na2O", "Si": "SiO2"},
                            "an anion")):
        with pytest.raises(ValueError, match=message):
            glass.composition(frame, _ox(frame), basis)


# ---------------------------------------------------------------------------
# frames: order, skips, progress, cancellation
# ---------------------------------------------------------------------------

def _jittered(frame, seed):
    rng = np.random.default_rng(seed)
    return md_model.frame_from_arrays(
        frame.elements, frame.cart_ang + rng.normal(0, 0.03, frame.cart_ang.shape),
        box_ang=frame.box_ang)


def test_the_order_frames_arrive_in_changes_no_result():
    base = _random_box(side=6)
    frames = [_jittered(base, s) for s in (1, 2, 3)]
    traj = md_model.MemoryTrajectory(frames)
    a = _analyse(frames, trajectory=traj, frames=[0, 1, 2])
    b = _analyse(frames, trajectory=traj, frames=[2, 0, 1])
    assert a.cutoffs_ang == b.cutoffs_ang
    for (sa, na, ia), (sb, nb, ib) in zip(a.results(), b.results()):
        assert (sa, na) == (sb, nb)
        assert np.array_equal(np.asarray(ia.mean), np.asarray(ib.mean),
                              equal_nan=True), na
        assert np.array_equal(np.asarray(ia.std), np.asarray(ib.std),
                              equal_nan=True), na


def test_permuting_or_translating_the_atoms_changes_no_count(random_frame):
    frame = random_frame[0]
    rng = np.random.default_rng(11)
    order = rng.permutation(frame.n_atoms)
    permuted = md_model.frame_from_arrays(
        frame.elements[order], frac=frame.frac[order], box_ang=frame.box_ang,
        atom_id=np.arange(frame.n_atoms))
    shifted = md_model.frame_from_arrays(
        frame.elements, frac=frame.frac + rng.random(3), box_ang=frame.box_ang)
    cut = {("Si", "O"): 2.4, ("Na", "O"): 2.9}
    reference = _analyse(frame, cutoffs_ang=cut)
    for other in (permuted, shifted):
        result = _analyse(other, cutoffs_ang=cut)
        for definition in ("bv", "distance"):
            for name, item in getattr(reference, definition).items():
                if isinstance(item, md_stats.Distribution):
                    other_item = getattr(result, definition)[name]
                    assert other_item.keys == item.keys, name
                    assert np.array_equal(other_item.per_frame,
                                          item.per_frame), name


class _Damaged(md_model.Trajectory):
    """Frames in memory, one of which the 'reader' cannot load."""

    def __init__(self, frames, bad):
        super().__init__(source_path="<damaged>", file_format="memory",
                         n_atoms=frames[0].n_atoms, n_frames=len(frames),
                         type_map_source="in memory")
        self._frames, self._bad = frames, bad

    def _load(self, k):
        if k in self._bad:
            raise ValueError("synthetic damage in this frame")
        return self._frames[k]


def test_a_frame_that_cannot_be_read_is_skipped_with_its_reason():
    frame = _dimer()
    traj = _Damaged([frame, frame, frame], bad={1})
    calls = []
    result = _analyse([frame], trajectory=traj,
                      progress=lambda done, total: calls.append((done, total)))
    assert result.provenance.frames_used == (0, 2)
    assert "synthetic damage" in result.provenance.frames_skipped[1]
    assert result.bv["CN Si"].n_frames == 2
    assert calls[-1][0] == calls[-1][1]
    assert [c[0] for c in calls] == sorted(c[0] for c in calls)


def test_cancelling_returns_what_was_done_and_lists_the_rest():
    frame = _dimer()
    traj = md_model.MemoryTrajectory([frame] * 4)
    asked = []

    def cancelled():
        asked.append(1)
        return len(asked) > 2

    result = _analyse([frame], trajectory=traj, cancelled=cancelled,
                      cutoffs_ang={("Si", "O"): 2.0})
    assert result.provenance.frames_used == (0, 1)
    assert set(result.provenance.frames_skipped) == {2, 3}
    assert any("cancelled after 2 of 4" in n for n in result.notes)
    with pytest.raises(glass.AnalysisCancelled):
        _analyse([frame], trajectory=traj, cancelled=lambda: True)


def test_cancelling_in_pass_two_keeps_pass_one_and_says_so():
    structure = readers.read(QUARTZ)
    frame, _ = md_model.supercell_frame(structure, (3, 3, 3))
    asked = []

    def cancelled():                     # pass 1 asks twice, pass 2 next
        asked.append(1)
        return len(asked) > 2

    result = _analyse([frame, frame], cancelled=cancelled)
    assert result.provenance.frames_used == (0, 1)
    assert result.bv["CN Si"].n_frames == 2 and result.distance == {}
    assert any(n.startswith("cancelled in pass 2") for n in result.notes)


def test_progress_never_goes_back_when_the_first_frames_cannot_be_read():
    """Before any frame is read the analysis cannot know whether it takes one
    pass or two; done / total never falls, whichever it turns out to be."""
    frame = _dimer()
    traj = _Damaged([frame] * 5, bad={0, 1})
    calls = []
    _analyse([frame], trajectory=traj,
             progress=lambda done, total: calls.append((done, total)))
    fractions = [done / total for done, total in calls]
    assert fractions == sorted(fractions), calls
    assert [c[0] for c in calls] == sorted(c[0] for c in calls)
    assert calls[-1][0] == calls[-1][1]
    one_pass = []
    _analyse([frame], trajectory=traj, cutoffs_ang={("Si", "O"): 2.0},
             progress=lambda done, total: one_pass.append((done, total)))
    assert [d / t for d, t in one_pass] == sorted(d / t for d, t in one_pass)
    assert one_pass[-1] == (5, 5)


def test_frames_whose_box_changes_give_the_same_result_in_any_order():
    """Three quartz frames, the same fractions in boxes scaled by 1.00, 0.97
    and 1.03 (an NPT run in miniature): read in any order they give the same
    notes and the same bond-length edges, the box change is noted however
    the frames arrive, and no bond length falls outside the edges (a bond
    is no longer than its cutoff, or than the bond-valence search radius)."""
    structure = readers.read(QUARTZ)
    base, _ = md_model.supercell_frame(structure, (3, 3, 3))
    frames = [md_model.frame_from_arrays(base.elements, frac=base.frac,
                                         box_ang=base.box_ang * scale)
              for scale in (1.0, 0.97, 1.03)]
    traj = md_model.MemoryTrajectory(frames)
    runs = [_analyse(frames, trajectory=traj, frames=order, r_max=7.0)
            for order in ([0, 1, 2], [1, 0, 2], [2, 1, 0])]
    half = [f.perpendicular_widths_ang.min() / 2.0 for f in frames]
    for run in runs:
        assert sorted(run.notes) == sorted(runs[0].notes)
        assert run.cutoffs_ang == runs[0].cutoffs_ang
        for definition in ("bv", "distance"):
            for name, item in getattr(runs[0], definition).items():
                if name.startswith("bond length"):
                    other = getattr(run, definition)[name]
                    assert np.array_equal(other.edges, item.edges), name
                    assert other.n_above.sum() == 0, name
    text = " | ".join(runs[0].notes)
    assert f"the g(r) grid stops at {min(half):.6g} Å" in text
    assert (f"the box changes between frames: half its smallest perpendicular "
            f"width runs from {min(half):.6g} to {max(half):.6g} Å") in text
    assert runs[0].rdf[("Si", "O")].axis[-1] <= min(half)


def test_notes_of_the_partials_the_frames_and_the_threshold_reach_the_result():
    """One Si relabelled Al: g_AlAl has no pair, said by partial_rdf and now
    by the result and by the Al-Al minimum. The frame's own note (atoms
    wrapped into the box) and bulk.at_threshold's note (atoms with no contact
    above v_bond) reach the result's notes too."""
    structure = readers.read(QUARTZ)
    base, _ = md_model.supercell_frame(structure, (2, 2, 2))
    symbols = base.elements.copy()
    symbols[np.flatnonzero(symbols == "Si")[0]] = "Al"
    frame = md_model.frame_from_arrays(
        symbols, base.cart_ang + 0.5 * base.box_ang.sum(axis=0),
        box_ang=base.box_ang)
    assert any("wrapped into it" in n for n in frame.notes)
    result = _analyse(frame, formers=None, r_max=4.0, v_bond_vu=5.0)
    text = " | ".join(result.notes)
    assert "the model holds one Al atom, so g_AlAl(r) has no pair" in text
    assert "holds one Al atom" in result.minima[("Al", "Al")].reason
    assert "wrapped into it" in text
    assert "have no contact above 5 v.u." in text


def test_an_empty_bridging_set_is_noted():
    """No anion may bridge, so every Si is Q0 though one O is bonded to two
    Si: the result says why the two disagree."""
    result = _analyse(_dimer(), bridging_anions=set())
    assert result.bv["Qn Si"].as_dict()[0][0] == 1.0
    assert result.bv["O speciation"].as_dict()["BO"][0] == 1 / 7
    assert any("an empty bridging-anion set was given" in n
               for n in result.notes)


def test_a_grid_too_short_for_a_minimum_is_refused_before_any_frame():
    frame = _dimer()
    traj = _Damaged([frame, frame], bad=set())
    calls = []
    with pytest.raises(ValueError, match=r"rdf_r_max_ang 0.02 Å with "
                       r"rdf_dr_ang 0.01 Å gives a g\(r\) grid of 2 points"):
        _analyse([frame], trajectory=traj, r_max=0.02,
                 progress=lambda done, total: calls.append(done))
    assert calls == []                        # no frame was analysed first


def test_a_smoothing_wider_than_the_grid_is_refused_before_any_frame():
    """A 0.2 Å kernel, 6 sigma each side, spans 243 points; the 1 Å grid
    asked for holds 100. first_minimum refused it only after pass 1 had
    searched every frame; the arguments alone decide it now, so it is
    refused before the first frame is read, whatever frame comes first."""
    frame = _dimer()
    traj = _Damaged([frame, frame], bad=set())
    calls = []
    with pytest.raises(ValueError, match=r"smooth_sigma_ang 0.2 Å spans 243 "
                       r"grid points, more than the 100 that rdf_r_max_ang "
                       r"1 Å and rdf_dr_ang 0.01 Å give"):
        _analyse([frame], trajectory=traj, r_max=1.0,
                 minimum=glass.MinimumMethod("valley", 0.2, "midpoint", 2.0),
                 progress=lambda done, total: calls.append(done))
    assert calls == []


def test_a_frame_selection_with_no_frame_is_refused():
    frame = _dimer()
    traj = md_model.MemoryTrajectory([frame])
    for frames in (slice(5, 5), slice(1, None), []):
        with pytest.raises(ValueError, match="frames selects no frame"):
            _analyse([frame], trajectory=traj, frames=frames)


def test_phi_of_a_one_bond_atom_stays_in_the_last_bin():
    """An atom with one bond has phi = |u| = 1; the division rounds it to
    1 + 2.2e-16 for some directions (and a rigid move flips it back). The
    direction is found by search, so the rounding is shown here, not
    assumed; binned as it stands the atom fell above the last edge."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        cart = np.array([[5.0, 5.0, 5.0], 5.0 + 2.1 * u])
        frame = md_model.frame_from_arrays(["Na", "Cl"], cart,
                                           box_ang=np.eye(3) * 12.0)
        _, results = bulk.analyse_frame(frame,
                                        _ox(frame).per_atom(frame.elements))
        if results.cn[0] == 1 and results.phi[0] > 1.0:
            break
    else:
        pytest.fail("no direction in 200 rounds phi above 1")
    result = _analyse(frame, formers=None, r_max=4.0)
    phi = result.bv["phi Na"]
    assert int(phi.n_above.sum()) == 0
    assert int(phi.counts[0, -1]) == 1 and int(phi.counts.sum()) == 1


# ---------------------------------------------------------------------------
# formers and refusals
# ---------------------------------------------------------------------------

def test_no_former_set_is_assumed():
    assert glass.FORMERS_DEFAULT is None
    frame = _dimer()
    _, _, bonds = _bonds(frame, _ox(frame).per_atom(frame.elements))
    with pytest.raises(ValueError, match="none given"):
        glass.qn_counts(bonds, frame, None, {"O"})
    result = _analyse(frame, formers=None)
    assert "Qn Si" not in result.bv and "O speciation" not in result.bv
    assert any("network formers not given" in n for n in result.notes)
    assert result.provenance.formers is None
    with pytest.raises(ValueError, match="anions of this model"):
        _analyse(frame, formers={"O"})


def test_impossible_values_become_notes_with_their_threshold():
    """A box compressed to 1.8 Å spacing bonds Si to its diagonal neighbours
    too (CN above 6); the 2.3 Å box leaves some O surrounded by O (no
    cation). Both are noted, with v_bond for one definition and the cutoffs
    for the other, and nothing is refused (a slider must be able to cross
    them)."""
    cut = {("Si", "O"): 2.6, ("Na", "O"): 2.6}
    dense = _analyse(_random_box(side=6, spacing_ang=1.8, seed=4),
                     formers=None, r_max=4.0, cutoffs_ang=cut)
    assert max(dense.bv["CN Si"].keys) > 6
    text = " | ".join(dense.notes)
    assert "Si with CN above 6 at v_bond = 0.075 v.u." in text
    assert "Si with CN above 6 within the distance cutoffs" in text
    sparse = _analyse(_random_box(side=6), formers=None, cutoffs_ang=cut)
    text = " | ".join(sparse.notes)
    assert "O atoms bonded to no cation at v_bond = 0.075 v.u." in text
    assert "O atoms bonded to no cation within the distance cutoffs" in text


def test_lone_pair_values_cover_every_atom_of_the_element():
    _, frame, _, ox = _crystal_frame(DATA / "senarmontite_Sb2O3_cod9009747.cif",
                                     (1, 1, 1))
    _, results, _ = _bonds(frame, ox)
    values = glass.lone_pair_values(results, frame, "Sb")
    rows = frame.elements == "Sb"
    assert np.array_equal(values["phi"], results.phi[rows])
    result = _analyse(frame, ox=_ox(frame), formers=None, r_max=4.0)
    phi = result.bv["phi Sb"]
    assert int(phi.counts.sum() + phi.n_nonfinite.sum() + phi.n_below.sum()
               + phi.n_above.sum()) == int(rows.sum())


# ---------------------------------------------------------------------------
# no verdicts, no Qt
# ---------------------------------------------------------------------------

VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def _texts(result) -> list[str]:
    out = list(result.notes) + result.provenance.as_lines()
    out += list(result.composition.notes)
    for _, _, item in result.results():
        out += list(item.notes)
    out += [m.reason for m in result.minima.values()]
    out += [note for m in result.minima.values() for note in m.notes]
    out += [m.method for m in result.minima.values()]
    out += list(result.cutoff_sources.values())
    return out


def test_no_note_or_message_carries_a_verdict(quartz_result):
    texts = _texts(quartz_result)
    texts += _texts(_analyse(_isolated_tetrahedron()))
    texts += _texts(_analyse(_random_box(side=6), formers=None))
    _, frame, _, _ = _crystal_frame(QUARTZ)
    texts += _texts(_analyse(frame))          # the grid cap note
    texts += _texts(_analyse(_dimer(), bridging_anions=set(), minimum=(
        glass.MinimumMethod("first local minimum", 0.03, "first", EXACT))))
    r = 0.01 * np.arange(1, 10)
    for rule in glass.MINIMUM_RULES:          # the wiggle and narrow notes
        found = glass.first_minimum(r, [0, 3, 1.2, 1.3, 0.4, 1.6, 0.3, 1.5, 1],
                                    glass.MinimumMethod(rule, None, "first",
                                                        EXACT))
        texts += list(found.notes) + [found.method]
    for call in (
            lambda: glass.qn_counts(None, ["Si"], None, {"O"}),
            lambda: glass.MinimumMethod("x", None, "first", EXACT),
            lambda: glass.MinimumMethod("valley", None, "first", -1.0),
            lambda: glass.HistogramBins(0, 1, 1, 1),
            lambda: _analyse(_dimer(), formers={"O"}),
            lambda: _analyse(_dimer(), cutoffs_ang={("Si", "Si"): 3.0}),
            lambda: _analyse(_dimer(), r_max=0.02),
            lambda: _analyse(_dimer(), r_max=1.0, minimum=glass.MinimumMethod(
                "valley", 0.2, "first", EXACT)),
            lambda: _analyse(_dimer(), frames=slice(5, 5)),
            lambda: glass.pair_counts_per_unit_g([0.1], 0.01, 2, 3, 10.0,
                                                 same_element=True,
                                                 n_frames=1),
            lambda: glass.first_minimum(r, np.ones(9), VALLEY,
                                        pairs_per_unit_g=np.zeros(9)),
            lambda: glass.composition(_dimer(), _ox(_dimer()), {"Si": "X"}),
            lambda: glass.rdf_grid_ang(_dimer(), 0.001)):
        try:
            call()
        except ValueError as error:
            texts.append(str(error))
        else:
            raise AssertionError("expected a refusal")
    assert len(texts) > 40
    assert [t for t in texts if VERDICT.search(t)] == []


def test_no_string_in_the_module_carries_a_verdict():
    source = (ROOT / "facet" / "core" / "glass.py").read_text(encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 100
    assert [s for s in strings if VERDICT.search(s)] == []


def test_the_module_imports_without_qt():
    code = ("import sys, facet.core.glass;"
            "print('QT' if any(m.startswith(('PySide6', 'matplotlib')) "
            "for m in sys.modules) else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
