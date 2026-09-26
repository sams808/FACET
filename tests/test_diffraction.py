"""Powder diffraction: structure factors, absences, profiles.

Five layers, the same shape as test_geometry_verification.py:

1. analytic     -- closed forms worked out by hand for rocksalt
2. independent  -- the vectorised path against the scalar one
3. invariance   -- enlarging the enumeration must change nothing
4. external     -- |F| against gemmi's own structure factor calculator
5. real data    -- the Bi CIF collection, when it is present

The external layer is the strongest of these: gemmi computes F by applying the
space group operators to the asymmetric unit, while FACET sums over a cell it
expanded itself. Two unrelated routes to the same number.
"""
from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from facet.core import cif
from facet.core import diffraction as dif
from facet.core.utilities import metric_tensor

BI_CIF_DIR = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi")


# ---------------------------------------------------------------------------
# fixtures built from text, so the tests do not depend on any file
# ---------------------------------------------------------------------------

def _build(a, b, c, alpha, beta, gamma, spacegroup, rows, with_u=False):
    header = ["data_test",
              f"_cell_length_a {a}", f"_cell_length_b {b}",
              f"_cell_length_c {c}", f"_cell_angle_alpha {alpha}",
              f"_cell_angle_beta {beta}", f"_cell_angle_gamma {gamma}",
              f"_symmetry_space_group_name_H-M '{spacegroup}'",
              "loop_", "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy"]
    if with_u:
        header.append("_atom_site_U_iso_or_equiv")
    text = "\n".join(header + list(rows)) + "\n"
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "test.cif")
    Path(path).write_text(text, encoding="utf-8")
    return cif.read(path)


@pytest.fixture(scope="module")
def rocksalt():
    """NaCl, a = 5.6402 A. Coordinates are exact, so absences cancel exactly."""
    return _build(5.6402, 5.6402, 5.6402, 90, 90, 90, "F m -3 m",
                  ["Na1 Na 0.0 0.0 0.0 1.0", "Cl1 Cl 0.5 0.5 0.5 1.0"])


@pytest.fixture(scope="module")
def tungsten():
    return _build(3.1652, 3.1652, 3.1652, 90, 90, 90, "I m -3 m",
                  ["W1 W 0.0 0.0 0.0 1.0"])


@pytest.fixture(scope="module")
def hexagonal():
    return _build(5.44, 5.44, 12.9, 90, 90, 120, "P 6_3/m m c",
                  ["Bi1 Bi 0.3333 0.6667 0.25 1.0",
                   "O1 O 0.0 0.0 0.0 1.0"])


@pytest.fixture(scope="module")
def monoclinic():
    return _build(5.85, 8.17, 7.51, 90, 113.0, 90, "P 1 21/c 1",
                  ["Bi1 Bi 0.02 0.04 0.24 1.0", "O1 O 0.78 0.31 0.21 1.0",
                   "Na1 Na 0.44 0.66 0.51 0.83"])


def _bi_files(limit=None):
    if not BI_CIF_DIR.is_dir():
        return []
    files = sorted(BI_CIF_DIR.rglob("*.cif"))
    return files[:limit] if limit else files


# ===========================================================================
# layer 1: analytic
# ===========================================================================

def test_rocksalt_intensities_match_the_closed_form(rocksalt):
    """For rocksalt, |F|^2 is 16(f_Na +- f_Cl)^2 and nothing else.

    All-odd indices subtract, all-even add. Worked out by hand: the four Na of
    the F lattice are always in phase with each other, likewise the four Cl,
    and the Cl sublattice is displaced by (1/2,1/2,1/2) so it carries
    exp(i pi (h+k+l)) relative to Na -- which is -1 for all-odd and +1 for
    all-even.
    """
    pattern = dif.powder_pattern(rocksalt, two_theta_max=90.0, b_iso=0.0)
    assert pattern.reflections

    for r in pattern.reflections:
        stol2 = 1.0 / (4.0 * r.d * r.d)
        f_na = float(dif.form_factor("Na", stol2))
        f_cl = float(dif.form_factor("Cl", stol2))
        all_odd = all(x % 2 == 1 for x in (r.h, r.k, r.l))
        expected = 16.0 * (f_na - f_cl if all_odd else f_na + f_cl) ** 2
        assert r.f_squared == pytest.approx(expected, rel=1e-9)


def test_rocksalt_neutron_intensities_match_the_closed_form(rocksalt):
    """The same closed form with scattering lengths instead of form factors.

    b_Na = +3.63 fm and b_Cl = +9.577 fm, both positive, so all-odd lines are
    weak and all-even strong -- the reverse of what happens when one length is
    negative, which is the case the neutron option exists for.
    """
    pattern = dif.powder_pattern(rocksalt, radiation="neutron",
                                 wavelength=1.54, b_iso=0.0)
    b_na = float(dif.form_factor("Na", 0.0, "neutron"))
    b_cl = float(dif.form_factor("Cl", 0.0, "neutron"))
    assert b_na == pytest.approx(3.63, abs=1e-3)
    assert b_cl == pytest.approx(9.577, abs=1e-3)

    for r in pattern.reflections:
        all_odd = all(x % 2 == 1 for x in (r.h, r.k, r.l))
        expected = 16.0 * (b_na - b_cl if all_odd else b_na + b_cl) ** 2
        assert r.f_squared == pytest.approx(expected, rel=1e-9)


def test_rocksalt_multiplicities_are_the_textbook_values(rocksalt):
    """Cubic powder multiplicities, which fall out of the merge."""
    expected = {"1 1 1": 8, "2 0 0": 6, "2 2 0": 12, "3 1 1": 24,
                "2 2 2": 8, "4 0 0": 6, "3 3 1": 24, "4 2 0": 24,
                "4 2 2": 24}
    pattern = dif.powder_pattern(rocksalt, two_theta_max=90.0)
    found = {r.hkl: r.multiplicity for r in pattern.reflections}
    assert found == expected


def test_rocksalt_line_positions_follow_bragg(rocksalt):
    """2-theta must be exactly what Bragg's law gives for the d-spacing."""
    pattern = dif.powder_pattern(rocksalt, two_theta_max=90.0)
    for r in pattern.reflections:
        sin_theta = pattern.wavelength / (2.0 * r.d)
        expected = 2.0 * math.degrees(math.asin(sin_theta))
        assert r.two_theta == pytest.approx(expected, abs=1e-9)
        # and d itself from the cubic cell
        n = math.sqrt(r.h ** 2 + r.k ** 2 + r.l ** 2)
        assert r.d == pytest.approx(5.6402 / n, rel=1e-9)


def test_strongest_rocksalt_line_is_200(rocksalt):
    """The 200 is the strongest line of rocksalt NaCl, at about 31.7 degrees.

    Pinned because it is the one number about this pattern that anyone who has
    run a diffractometer knows by heart, and it would catch a scale error that
    the relative closed-form checks above would not.
    """
    pattern = dif.powder_pattern(rocksalt, two_theta_max=90.0)
    strongest = pattern.strongest(1)[0]
    assert strongest.hkl == "2 0 0"
    assert strongest.two_theta == pytest.approx(31.70, abs=0.02)
    assert strongest.intensity == pytest.approx(100.0)


def test_lorentz_polarisation_matches_the_closed_form():
    for two_theta in (5.0, 10.0, 30.0, 45.0, 90.0, 120.0, 160.0, 179.0):
        theta = math.radians(two_theta / 2.0)
        expected = ((1.0 + math.cos(math.radians(two_theta)) ** 2)
                    / (math.sin(theta) ** 2 * math.cos(theta)))
        assert dif.lorentz_polarisation(two_theta) == pytest.approx(expected,
                                                                    rel=1e-12)


def test_lorentz_polarisation_with_a_monochromator():
    k = math.cos(math.radians(26.6)) ** 2
    for two_theta in (20.0, 40.0, 100.0):
        theta = math.radians(two_theta / 2.0)
        polarisation = ((1.0 + k * math.cos(math.radians(two_theta)) ** 2)
                        / (1.0 + k))
        expected = polarisation / (math.sin(theta) ** 2 * math.cos(theta))
        got = dif.lorentz_polarisation(two_theta, 26.6)
        assert got == pytest.approx(expected, rel=1e-12)


def test_lorentz_polarisation_is_finite_at_the_singularities():
    """2-theta of 0 and 180 both divide by zero; neither may return a nan."""
    for two_theta in (0.0, 180.0):
        value = dif.lorentz_polarisation(two_theta)
        assert math.isfinite(value)
        assert value == 0.0


def test_debye_waller_matches_the_u_form():
    """exp(-B s^2) and exp(-2 pi^2 U / d^2) are the same statement."""
    for u_iso, d in ((0.005, 3.0), (0.02, 1.5), (0.05, 1.05), (0.0, 2.0)):
        b = dif.b_from_u(u_iso)
        assert b == pytest.approx(8.0 * math.pi ** 2 * u_iso)
        stol2 = 1.0 / (4.0 * d * d)
        assert float(dif.debye_waller(b, stol2)) == pytest.approx(
            math.exp(-2.0 * math.pi ** 2 * u_iso / (d * d)), rel=1e-12)


def test_form_factor_at_zero_angle_is_the_electron_count():
    """f(0) must be Z for a neutral atom -- the check that the table is sane.

    Not exactly Z: the four-Gaussian fit leaves a residual, largest for heavy
    elements (Bi comes out 82.955 against 83). A part in a thousand is the fit,
    not an error; a factor of two would be a wrong table or a wrong argument.
    """
    import gemmi

    for symbol in ("H", "O", "Na", "Si", "P", "Cl", "Fe", "I", "Bi"):
        z = gemmi.Element(symbol).atomic_number
        assert float(dif.form_factor(symbol, 0.0)) == pytest.approx(
            z, rel=1e-3, abs=0.01)


def test_form_factors_fall_away_with_angle_but_neutron_lengths_do_not():
    angles = np.array([0.0, 0.05, 0.2, 0.5])
    xray = dif.form_factor("Bi", angles, "X-ray")
    assert np.all(np.diff(xray) < 0)

    neutron = dif.form_factor("Bi", angles, "neutron")
    assert np.allclose(neutron, neutron[0])

    electron = dif.form_factor("Bi", angles, "electron")
    assert np.all(np.diff(electron) < 0)


def test_negative_neutron_lengths_are_preserved():
    """H, Ti and Mn scatter neutrons with the opposite sign.

    If this were taken as an absolute value, a neutron pattern would lose the
    contrast that makes the technique worth using.
    """
    for symbol in ("H", "Ti", "Mn"):
        assert float(dif.form_factor(symbol, 0.0, "neutron")) < 0
    for symbol in ("Bi", "O", "Na", "Si"):
        assert float(dif.form_factor(symbol, 0.0, "neutron")) > 0


def test_unknown_element_scatters_nothing_instead_of_raising():
    assert float(dif.form_factor("Xx", 0.1)) == 0.0


# ===========================================================================
# layer 2: the vectorised path against the scalar one
# ===========================================================================

@pytest.mark.parametrize("radiation", ["X-ray", "neutron", "electron"])
def test_vectorised_structure_factors_match_the_scalar_routine(monoclinic,
                                                               radiation):
    """Two implementations of the same sum, one per reflection and one in bulk.

    The chunk size is deliberately not a divisor of the reflection count, so a
    chunk-boundary mistake cannot hide.
    """
    fracs, occupancies, b_values, symbols = dif._atom_arrays(monoclinic, 0.7)
    hkl = np.array([[h, k, l]
                    for h in range(-3, 4) for k in range(-3, 4)
                    for l in range(-3, 4) if (h, k, l) != (0, 0, 0)], float)
    reciprocal = np.linalg.inv(metric_tensor(monoclinic))
    stol2 = np.einsum("mi,ij,mj->m", hkl, reciprocal, hkl) / 4.0

    bulk = dif._structure_factors(hkl, stol2, fracs, occupancies, b_values,
                                  symbols, radiation, chunk=37)
    scalar = np.array([
        abs(dif.structure_factor(monoclinic, int(h), int(k), int(l),
                                 float(s2), radiation, 0.7)) ** 2
        for (h, k, l), s2 in zip(hkl, stol2)])
    assert np.allclose(bulk, scalar, rtol=1e-9, atol=1e-9)


def test_chunking_does_not_change_the_answer(monoclinic):
    fracs, occupancies, b_values, symbols = dif._atom_arrays(monoclinic, 0.5)
    hkl = np.array([[h, k, l] for h in range(0, 5) for k in range(0, 5)
                    for l in range(0, 5) if (h, k, l) != (0, 0, 0)], float)
    reciprocal = np.linalg.inv(metric_tensor(monoclinic))
    stol2 = np.einsum("mi,ij,mj->m", hkl, reciprocal, hkl) / 4.0
    reference = dif._structure_factors(hkl, stol2, fracs, occupancies,
                                       b_values, symbols, "X-ray", chunk=10000)
    for chunk in (1, 2, 7, 31, 124):
        got = dif._structure_factors(hkl, stol2, fracs, occupancies, b_values,
                                     symbols, "X-ray", chunk=chunk)
        assert np.allclose(got, reference, rtol=1e-12, atol=1e-12)


def test_structure_factor_infers_stol2_from_the_cell(rocksalt):
    """Calling without stol2 must give what passing the right one gives."""
    from facet.core.utilities import d_spacing

    for hkl in ((1, 1, 1), (2, 0, 0), (3, 1, 1)):
        d = d_spacing(rocksalt, *hkl)
        explicit = dif.structure_factor(rocksalt, *hkl,
                                        stol2=1.0 / (4.0 * d * d))
        implicit = dif.structure_factor(rocksalt, *hkl)
        assert abs(explicit - implicit) < 1e-9


# ===========================================================================
# layer 3: invariance
# ===========================================================================

SYSTEMS = {
    "hexagonal": (5.44, 5.44, 12.9, 90, 90, 120, "P 6_3/m m c",
                  ["Bi1 Bi 0.3333 0.6667 0.25 1.0", "O1 O 0.0 0.0 0.0 1.0"]),
    "trigonal": (5.0, 5.0, 13.8, 90, 90, 120, "R -3 c",
                 ["Bi1 Bi 0.0 0.0 0.0 1.0", "O1 O 0.30 0.0 0.25 1.0"]),
    "cubic": (5.64, 5.64, 5.64, 90, 90, 90, "F m -3 m",
              ["Na1 Na 0.0 0.0 0.0 1.0", "Cl1 Cl 0.5 0.5 0.5 1.0"]),
    "tetragonal": (3.79, 3.79, 9.51, 90, 90, 90, "I 4/m m m",
                   ["Bi1 Bi 0.0 0.0 0.0 1.0", "O1 O 0.0 0.5 0.25 1.0"]),
    "monoclinic": (5.85, 8.17, 7.51, 90, 113.0, 90, "P 1 21/c 1",
                   ["Bi1 Bi 0.02 0.04 0.24 1.0", "O1 O 0.78 0.31 0.21 1.0"]),
    "triclinic": (6.10, 7.20, 8.30, 88, 102, 95, "P -1",
                  ["Bi1 Bi 0.11 0.22 0.33 1.0", "O1 O 0.61 0.42 0.13 1.0"]),
}


@pytest.mark.parametrize("system", sorted(SYSTEMS))
def test_enlarging_the_enumeration_changes_nothing(system, monkeypatch):
    """The completeness property the automatic index bounds exist for.

    If the bounds already hold every symmetry equivalent of every reflection in
    range, then enumerating a cube three times wider must produce exactly the
    same lines with exactly the same multiplicities. If any equivalent were
    being clipped, the wider run would find it and the multiplicity would rise.
    """
    structure = _build(*SYSTEMS[system])
    narrow = dif.powder_pattern(structure, two_theta_max=110.0, b_iso=0.4,
                                min_intensity=0.0)

    real_bounds = dif.index_bounds
    monkeypatch.setattr(dif, "index_bounds",
                        lambda s, d_min, ceiling=200:
                        tuple(3 * b for b in real_bounds(s, d_min)))
    wide = dif.powder_pattern(structure, two_theta_max=110.0, b_iso=0.4,
                              min_intensity=0.0)

    assert len(narrow.reflections) == len(wide.reflections)
    for a, b in zip(narrow.reflections, wide.reflections):
        assert a.d == pytest.approx(b.d, rel=1e-12)
        assert a.multiplicity == b.multiplicity
        assert a.intensity == pytest.approx(b.intensity, rel=1e-9)


def test_index_bounds_are_the_rigorous_limit(hexagonal):
    """|h| <= a / d_min, because h is the projection of r* onto a."""
    for d_min in (0.8, 1.0, 1.5, 2.5):
        bounds = dif.index_bounds(hexagonal, d_min)
        for bound, length in zip(bounds, hexagonal.cell.lengths):
            assert bound >= length / d_min


def test_a_cap_below_the_requirement_undercounts_and_says_so(hexagonal):
    """A fixed index cube is the failure the automatic bounds replace.

    Capped too low, the hexagonal (3 2 0) family loses two thirds of its
    equivalents to the cube's corner and the line comes out correspondingly
    weak. FACET must not do that silently.
    """
    full = dif.powder_pattern(hexagonal, two_theta_max=110.0, b_iso=0.4,
                              min_intensity=0.0)
    capped = dif.powder_pattern(hexagonal, two_theta_max=110.0, b_iso=0.4,
                                min_intensity=0.0, max_index=4)

    by_d = {round(r.d, 6): r for r in capped.reflections}
    undercounted = [r.hkl for r in full.reflections
                    if round(r.d, 6) in by_d
                    and by_d[round(r.d, 6)].multiplicity < r.multiplicity]
    assert undercounted, "expected the cap to clip some equivalents"
    assert len(capped.reflections) < len(full.reflections)
    assert any("capped" in note for note in capped.notes)

    # and the automatic run must not be flagged
    assert not any("capped" in note for note in full.notes)


def test_intensity_is_independent_of_the_origin(monoclinic):
    """Shifting every atom by the same vector cannot change any intensity.

    |F| is invariant under a change of origin: the shift multiplies every term
    by one common phase factor. A sign or index error in the phase would break
    this, which is why it is worth asserting rather than assuming.
    """
    reference = dif.powder_pattern(monoclinic, two_theta_max=80.0, b_iso=0.5)

    import copy
    shifted = copy.deepcopy(monoclinic)
    offset = np.array([0.137, -0.291, 0.408])
    for site in shifted.sites:
        site.frac = (site.frac + offset) % 1.0
    for atom in shifted.atoms:
        atom.frac = (atom.frac + offset) % 1.0
        atom.cart = shifted.cell.to_cartesian(atom.frac)

    moved = dif.powder_pattern(shifted, two_theta_max=80.0, b_iso=0.5)
    assert len(moved.reflections) == len(reference.reflections)
    for a, b in zip(reference.reflections, moved.reflections):
        assert a.d == pytest.approx(b.d, rel=1e-12)
        assert a.intensity == pytest.approx(b.intensity, abs=1e-6)


def test_friedel_pairs_have_equal_intensity(monoclinic):
    """Without anomalous scattering, F(hkl) and F(-h-k-l) are conjugates."""
    for hkl in ((1, 2, 3), (2, 0, 1), (3, 1, 4), (1, 1, 1)):
        forward = dif.structure_factor(monoclinic, *hkl)
        reverse = dif.structure_factor(monoclinic, *(-x for x in hkl))
        assert abs(forward) == pytest.approx(abs(reverse), rel=1e-12)
        assert forward == pytest.approx(reverse.conjugate(), abs=1e-9)


def test_two_theta_window_only_removes_lines(rocksalt):
    everything = dif.powder_pattern(rocksalt, two_theta_max=90.0)
    windowed = dif.powder_pattern(rocksalt, two_theta_min=40.0,
                                  two_theta_max=80.0)
    inside = [r for r in everything.reflections if 40.0 <= r.two_theta <= 80.0]
    assert [r.hkl for r in windowed.reflections] == [r.hkl for r in inside]
    # the window renormalises, so compare ratios rather than absolute values
    if len(inside) > 1:
        a = inside[0].intensity / inside[1].intensity
        b = windowed.reflections[0].intensity / windowed.reflections[1].intensity
        assert a == pytest.approx(b, rel=1e-9)


# ===========================================================================
# systematic absences
# ===========================================================================

def test_f_centring_absences(rocksalt):
    """All even or all odd. Never asserted as a rule -- the sum cancels."""
    pattern = dif.powder_pattern(rocksalt, two_theta_max=100.0,
                                 min_intensity=0.02)
    for r in pattern.reflections:
        parities = {r.h % 2, r.k % 2, r.l % 2}
        assert len(parities) == 1, f"{r.hkl} breaks F centring"


def test_i_centring_absences(tungsten):
    pattern = dif.powder_pattern(tungsten, two_theta_max=140.0,
                                 min_intensity=0.02)
    assert pattern.reflections
    for r in pattern.reflections:
        assert (r.h + r.k + r.l) % 2 == 0, f"{r.hkl} breaks I centring"


def test_absent_reflections_are_numerically_zero(rocksalt):
    """With exact coordinates, an absence is exactly zero, not merely small."""
    for hkl in ((1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 2, 1), (3, 2, 1)):
        assert abs(dif.structure_factor(rocksalt, *hkl)) < 1e-9


def test_screw_axis_absence_in_p21():
    """P2_1 along b forbids 0k0 with k odd. Another rule never coded in."""
    structure = _build(6.0, 7.0, 8.0, 90, 90, 90, "P 1 21 1",
                       ["Bi1 Bi 0.13 0.27 0.41 1.0"])
    assert abs(dif.structure_factor(structure, 0, 1, 0)) < 1e-6
    assert abs(dif.structure_factor(structure, 0, 3, 0)) < 1e-6
    assert abs(dif.structure_factor(structure, 0, 2, 0)) > 1.0


# ===========================================================================
# layer 4: external -- gemmi's own structure factor calculator
# ===========================================================================

def _gemmi_calculator(path):
    """gemmi's calculator, with displacement parameters cleared.

    gemmi applies every symmetry operation without collapsing special
    positions, so a CIF occupancy of 1 on a special position must first be
    converted to the crystallographic convention -- otherwise the comparison
    is off by the order of the site symmetry.
    """
    import gemmi

    small = gemmi.read_small_structure(str(path))
    for site in small.sites:
        site.u_iso = 0.0
        for component in ("u11", "u22", "u33", "u12", "u13", "u23"):
            setattr(site.aniso, component, 0.0)
        assert not site.aniso.nonzero()
    small.change_occupancies_to_crystallographic()
    return small, gemmi.StructureFactorCalculatorX(small.cell)


@pytest.mark.parametrize("path", _bi_files(limit=12),
                         ids=lambda p: p.stem[:24])
def test_structure_factors_agree_with_gemmi(path):
    """FACET sums over a cell it expanded; gemmi applies the operators.

    Compared on the scale that reaches a pattern -- |F|^2 relative to the
    strongest |F|^2 of that structure -- because a systematically absent
    reflection is exactly zero for gemmi and only as close to zero as the
    CIF's coordinate precision allows for FACET. Judging that residual against
    zero would be measuring the file's decimal places, not the code.
    """
    pytest.importorskip("gemmi")
    structure = cif.read(path)
    small, calculator = _gemmi_calculator(path)
    for site in structure.sites:
        site.u_iso = None

    mine, theirs = [], []
    for h in range(0, 4):
        for k in range(0, 4):
            for l in range(0, 4):
                if (h, k, l) == (0, 0, 0):
                    continue
                theirs.append(abs(calculator.calculate_sf_from_small_structure(
                    small, (h, k, l))) ** 2)
                mine.append(abs(dif.structure_factor(
                    structure, h, k, l, b_iso=0.0)) ** 2)

    mine, theirs = np.array(mine), np.array(theirs)
    scale = max(theirs.max(), 1e-12)
    assert np.max(np.abs(mine - theirs)) / scale < 2e-3


@pytest.mark.parametrize("path", _bi_files(limit=12),
                         ids=lambda p: p.stem[:24])
def test_symmetry_expansion_agrees_with_gemmi(path):
    """Every atom FACET generates must be where gemmi puts one, and no more.

    This is the check behind the duplicated-atom report: it compares the whole
    expanded cell, atom count included, against an independent expansion.
    """
    pytest.importorskip("gemmi")
    import gemmi

    structure = cif.read(path)
    small = gemmi.read_small_structure(str(path))
    theirs = np.array([[s.fract.x % 1.0, s.fract.y % 1.0, s.fract.z % 1.0]
                       for s in small.get_all_unit_cell_sites()])
    mine = np.array([a.frac for a in structure.atoms])
    assert len(mine) == len(theirs), "expanded cell has the wrong atom count"

    delta = mine[:, None, :] - theirs[None, :, :]
    delta -= np.round(delta)                       # minimum image
    cart = np.einsum("ij,mnj->mni", structure.cell.orth, delta)
    nearest = np.linalg.norm(cart, axis=-1).min(axis=1)
    assert float(nearest.max()) < 1e-6


# ===========================================================================
# layer 5: real data
# ===========================================================================

@pytest.mark.parametrize("path", _bi_files(limit=8), ids=lambda p: p.stem[:24])
def test_real_patterns_are_well_formed(path):
    structure = cif.read(path)
    pattern = dif.powder_pattern(structure, two_theta_max=80.0)
    assert pattern.reflections, f"{path.name} produced no lines"

    intensities = [r.intensity for r in pattern.reflections]
    assert max(intensities) == pytest.approx(100.0)
    assert min(intensities) >= 0.0
    two_theta = [r.two_theta for r in pattern.reflections]
    assert two_theta == sorted(two_theta)
    for r in pattern.reflections:
        assert r.multiplicity >= 1
        assert r.d > 0
        assert math.isfinite(r.intensity)
    # every pattern must say how it was made
    assert any("lambda" in note for note in pattern.notes)
    assert any("refined" in note for note in pattern.notes)


def test_no_atoms_is_an_error_not_an_empty_pattern(rocksalt):
    import copy

    empty = copy.deepcopy(rocksalt)
    empty.atoms = []
    with pytest.raises(ValueError, match="no atoms"):
        dif.powder_pattern(empty)


def test_bad_arguments_are_rejected(rocksalt):
    with pytest.raises(ValueError, match="wavelength"):
        dif.powder_pattern(rocksalt, wavelength=0.0)
    with pytest.raises(ValueError, match="two_theta_max"):
        dif.powder_pattern(rocksalt, two_theta_max=0.0)
    with pytest.raises(ValueError, match="two_theta_max"):
        dif.powder_pattern(rocksalt, two_theta_max=181.0)


def test_wavelength_too_long_for_any_reflection_gives_an_empty_pattern(rocksalt):
    """A 20 A wavelength cannot satisfy Bragg for a 5.6 A cell."""
    pattern = dif.powder_pattern(rocksalt, wavelength=20.0)
    assert pattern.reflections == []
    assert any("no reflections" in note for note in pattern.notes)


def test_displacement_parameters_are_used_and_declared():
    """A U_iso in the file must change the pattern, and be reported as used."""
    without = _build(5.6402, 5.6402, 5.6402, 90, 90, 90, "F m -3 m",
                     ["Na1 Na 0.0 0.0 0.0 1.0", "Cl1 Cl 0.5 0.5 0.5 1.0"])
    with_u = _build(5.6402, 5.6402, 5.6402, 90, 90, 90, "F m -3 m",
                    ["Na1 Na 0.0 0.0 0.0 1.0 0.030",
                     "Cl1 Cl 0.5 0.5 0.5 1.0 0.025"], with_u=True)
    assert all(s.u_iso for s in with_u.sites)

    plain = dif.powder_pattern(without, two_theta_max=90.0, b_iso=0.0)
    damped = dif.powder_pattern(with_u, two_theta_max=90.0)
    assert any("no displacement parameters" in n for n in plain.notes)
    assert any("for every atom" in n for n in damped.notes)

    # high angle must be relatively weaker once the atoms are allowed to move
    def ratio(pattern):
        high = max(pattern.reflections, key=lambda r: r.two_theta)
        return high.intensity
    assert ratio(damped) < ratio(plain)


def test_a_pattern_never_states_a_verdict(rocksalt):
    """FACET reports; it does not judge. No note may grade the result."""
    pattern = dif.powder_pattern(rocksalt)
    forbidden = {"good", "bad", "poor", "excellent", "acceptable", "correct",
                 "incorrect", "wrong", "reliable", "unreliable", "suspicious",
                 "better", "worse", "should"}
    for note in pattern.notes:
        # whole words only: "correction" is a thing FACET declines to apply,
        # not a verdict on anybody's structure
        words = set(re.findall(r"[a-z]+", note.lower()))
        offending = words & forbidden
        assert not offending, f"note passes judgement ({offending}): {note}"


# ===========================================================================
# the peak profile
# ===========================================================================

def _one_line(two_theta):
    return dif.Pattern(reflections=[dif.Reflection(
        h=1, k=0, l=0, d=1.0, two_theta=two_theta, intensity=100.0,
        multiplicity=1, f_squared=1.0, lorentz_polarisation=1.0)])


@pytest.mark.parametrize("two_theta", [12.0, 40.0, 95.0])
@pytest.mark.parametrize("uvw", [(0.0, 0.0, 0.01), (0.0, 0.0, 0.04),
                                 (0.010, -0.004, 0.006)])
@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_profile_width_follows_caglioti(two_theta, uvw, eta):
    """The generated peak must have the FWHM the parameters ask for.

    Independent of eta, because both components are normalised to unit height
    rather than unit area -- so the mixing parameter changes the tails and not
    the width.
    """
    u, v, w = uvw
    step = 0.0002
    x, y = _one_line(two_theta).profile(
        two_theta_min=two_theta - 6, two_theta_max=two_theta + 6,
        step=step, u=u, v=v, w=w, eta=eta)
    above = x[y >= y.max() / 2.0]
    fwhm = float(above.max() - above.min())

    t = math.tan(math.radians(two_theta / 2.0))
    expected = math.sqrt(max(u * t * t + v * t + w, 1e-6))
    assert fwhm == pytest.approx(expected, abs=3 * step)
    assert x[int(np.argmax(y))] == pytest.approx(two_theta, abs=step)


def test_lorentzian_has_the_heavier_tail():
    """At five widths out, a Lorentzian is at 1/101 of its peak; a Gaussian is
    at 10^-30. The mixing parameter must actually select between them."""
    pattern = _one_line(40.0)
    heights = {}
    for eta in (0.0, 1.0):
        x, y = pattern.profile(two_theta_min=34, two_theta_max=46, step=0.0005,
                               u=0.0, v=0.0, w=0.04, eta=eta)
        heights[eta] = float(np.interp(40.0 + 5 * 0.2, x, y) / y.max())
    assert heights[1.0] == pytest.approx(1.0 / 101.0, rel=0.02)
    assert heights[0.0] < 1e-20


def test_zero_shift_moves_every_line_by_the_shift():
    pattern = _one_line(40.0)
    for shift in (-0.25, 0.0, 0.31):
        x, y = pattern.profile(two_theta_min=34, two_theta_max=46,
                               step=0.0002, u=0.0, v=0.0, w=0.04, eta=0.5,
                               zero_shift=shift)
        assert x[int(np.argmax(y))] == pytest.approx(40.0 + shift, abs=3e-4)


def test_profile_is_normalised_to_a_hundred(rocksalt):
    pattern = dif.powder_pattern(rocksalt, two_theta_max=90.0)
    x, y = pattern.profile(5.0, 90.0, step=0.02)
    assert y.max() == pytest.approx(100.0)
    assert y.min() >= 0.0
    assert len(x) == len(y)


def test_profile_of_an_empty_pattern_is_flat_not_a_crash():
    x, y = dif.Pattern().profile(10.0, 20.0, step=0.1)
    assert len(x) == len(y)
    assert np.all(y == 0.0)


# ===========================================================================
# measured patterns
# ===========================================================================

def _write(text, suffix=".xy"):
    handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    handle.write(text)
    handle.close()
    return handle.name


def test_read_pattern_handles_the_usual_exports():
    body = "\n".join(f"{5 + 0.02 * i:.4f} {100 + i}" for i in range(50))
    for text in (body,
                 "# 2theta intensity\n" + body,
                 "! comment\n' another\n; third\n" + body,
                 body.replace(" ", ","),
                 body.replace(" ", "\t"),
                 "\n".join(line + " 3.2" for line in body.splitlines())):
        x, y = dif.read_pattern(_write(text))
        assert len(x) == 50
        assert x[0] == pytest.approx(5.0)
        assert y[0] == pytest.approx(100.0)
        assert list(x) == sorted(x)


def test_read_pattern_sorts_by_angle():
    text = "30 5\n10 1\n20 3\n"
    x, y = dif.read_pattern(_write(text))
    assert list(x) == [10.0, 20.0, 30.0]
    assert list(y) == [1.0, 3.0, 5.0]


def test_read_pattern_rejects_a_file_with_no_data():
    for text in ("", "# only a comment\n", "just words here\n", "5.0\n6.0\n"):
        with pytest.raises(ValueError, match="two-column"):
            dif.read_pattern(_write(text))


def test_scale_to_measured_recovers_a_known_factor():
    x = np.linspace(10, 80, 700)
    calculated = np.exp(-((x - 30) / 0.4) ** 2) * 100 + 3.0
    for factor in (0.25, 1.0, 7.5, 1200.0):
        _, found = dif.scale_to_measured(x, calculated, x, calculated * factor)
        assert found == pytest.approx(factor, rel=1e-6)


def test_scale_to_measured_with_no_overlap_leaves_the_data_alone():
    a = np.linspace(10, 20, 50)
    b = np.linspace(60, 70, 50)
    y = np.ones(50)
    scaled, factor = dif.scale_to_measured(a, y, b, y)
    assert factor == 1.0
    assert np.array_equal(scaled, y)


def test_difference_curve_of_a_pattern_against_itself_is_zero():
    x = np.linspace(10, 80, 900)
    y = np.exp(-((x - 45) / 0.3) ** 2) * 80 + np.exp(-((x - 60) / 0.3) ** 2) * 40
    grid, difference, factor = dif.difference_curve(x, y, x, y * 3.0)
    assert factor == pytest.approx(3.0, rel=1e-6)
    assert np.max(np.abs(difference)) < 1e-6 * float(np.max(y * 3.0))
    assert len(grid) == len(difference) == len(x)


def test_difference_curve_is_on_the_measured_grid():
    calculated_x = np.linspace(10, 80, 300)
    measured_x = np.linspace(15, 75, 901)
    grid, difference, _ = dif.difference_curve(
        calculated_x, np.ones(300), measured_x, np.ones(901) * 2.0)
    assert np.array_equal(grid, measured_x)
    assert len(difference) == len(measured_x)


# ===========================================================================
# housekeeping
# ===========================================================================

def test_wavelength_table_is_plausible():
    for name, value in dif.WAVELENGTHS.items():
        assert 0.4 < value < 2.5, name
    assert dif.WAVELENGTHS["Cu Ka1"] == pytest.approx(1.540598)
    assert dif.WAVELENGTHS["Mo Ka1"] == pytest.approx(0.709317)
    # the doublet mean must sit between its components
    assert (dif.WAVELENGTHS["Cu Ka1"] < dif.WAVELENGTHS["Cu Ka"]
            < dif.WAVELENGTHS["Cu Ka2"])


def test_reflection_reports_q_consistently(rocksalt):
    pattern = dif.powder_pattern(rocksalt, two_theta_max=60.0)
    for r in pattern.reflections:
        assert r.q == pytest.approx(2.0 * math.pi / r.d, rel=1e-12)


def test_representative_index_prefers_all_positive():
    candidates = np.array([[-1, 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0]])
    assert list(dif._representative(candidates)) == [1, 0, 0]
    only_negative = np.array([[-1, -2, -3], [-3, -2, -1]])
    assert list(dif._representative(only_negative)) == [-1, -2, -3]


def test_strongest_returns_lines_in_descending_order(rocksalt):
    pattern = dif.powder_pattern(rocksalt, two_theta_max=90.0)
    top = pattern.strongest(5)
    assert len(top) == 5
    assert [r.intensity for r in top] == sorted(
        [r.intensity for r in top], reverse=True)


def test_pattern_arrays_line_up_with_the_reflections(rocksalt):
    pattern = dif.powder_pattern(rocksalt, two_theta_max=90.0)
    assert list(pattern.two_theta) == [r.two_theta for r in pattern.reflections]
    assert list(pattern.intensity) == [r.intensity for r in pattern.reflections]
    assert list(pattern.d) == [r.d for r in pattern.reflections]


def test_radiations_all_produce_a_pattern(rocksalt):
    for radiation in dif.RADIATIONS:
        pattern = dif.powder_pattern(rocksalt, radiation=radiation,
                                     two_theta_max=70.0)
        assert pattern.reflections
        assert pattern.radiation == radiation
        assert any(radiation in note for note in pattern.notes)
