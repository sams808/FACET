"""The pair distribution function, checked against things outside the code.

Five layers, following tests/test_diffraction.py:

1. **Analytic** -- NaCl's first peak area is 6 b_Na b_Cl / <b>^2 exactly, its
   position is a/2, the pair weights sum to 1, g(r) tends to 1.
2. **Two unrelated routes** -- the closed-form truncation kernel against an
   explicit forward sine transform, truncation, and transform back. This pins
   the 2/pi, the 4*pi*r*rho0 baseline and the kernel normalisation at once, and
   it is the PDF equivalent of the gemmi structure-factor check.
3. **Invariance** -- an origin shift and a 2x2x2 supercell are two ways of
   writing the same crystal, so G(r) must not notice.
4. **Cross-module** -- peak degeneracies against utilities.radial_shells, which
   counts neighbours by its own search, and the U fallback against
   diffraction.b_from_u.
5. **Real data** -- the CIF collection, asserting only that the output is well
   formed and that the notes declare where U came from.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from facet.core import cif
from facet.core import diffraction as dif
from facet.core import pdf as P

BI_CIF_DIR = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi")


def _build(a, b, c, alpha, beta, gamma, spacegroup, rows):
    header = ["data_test",
              f"_cell_length_a {a}", f"_cell_length_b {b}",
              f"_cell_length_c {c}", f"_cell_angle_alpha {alpha}",
              f"_cell_angle_beta {beta}", f"_cell_angle_gamma {gamma}",
              f"_symmetry_space_group_name_H-M '{spacegroup}'",
              "loop_", "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy"]
    text = "\n".join(header + list(rows)) + "\n"
    path = os.path.join(tempfile.mkdtemp(), "test.cif")
    Path(path).write_text(text, encoding="utf-8")
    return cif.read(path)


A_NACL = 5.6402
FCC_FRACS = (((0, 0, 0), "Na"), ((0, .5, .5), "Na"), ((.5, 0, .5), "Na"),
             ((.5, .5, 0), "Na"), ((.5, .5, .5), "Cl"), ((.5, 0, 0), "Cl"),
             ((0, .5, 0), "Cl"), ((0, 0, .5), "Cl"))


@pytest.fixture(scope="module")
def rocksalt():
    return _build(A_NACL, A_NACL, A_NACL, 90, 90, 90, "F m -3 m",
                  ["Na1 Na 0.0 0.0 0.0 1.0", "Cl1 Cl 0.5 0.5 0.5 1.0"])


@pytest.fixture(scope="module")
def rocksalt_p1():
    """The same crystal written out atom by atom, so it can be shifted."""
    return _build(A_NACL, A_NACL, A_NACL, 90, 90, 90, "P 1", [
        f"{e}{i} {e} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f} 1.0"
        for i, (f, e) in enumerate(FCC_FRACS)])


# --- 1: analytic ------------------------------------------------------------

def test_the_pair_weights_sum_to_one(rocksalt):
    for radiation in dif.RADIATIONS:
        w = P.scattering_weights(rocksalt, radiation)
        assert sum(w.pairs.values()) == pytest.approx(1.0, abs=1e-12), radiation


def test_the_first_peak_area_is_the_weighted_coordination_number(rocksalt):
    """Six Cl about each Na: the area under the peak must be 6 w_NaCl.

    This is the whole reason R(r) carries the 1/r at the field point rather than
    at r_ij -- only that form keeps a peak's area equal to the coordination
    number.
    """
    w = P.scattering_weights(rocksalt, "X-ray")
    out = P.pair_distribution(rocksalt, r_max=12.0, dr=0.005, u_iso=0.004)
    closed = 6.0 * w.b["Na"] * w.b["Cl"] / (w.b_mean ** 2)
    assert out.coordination_in(2.3, 3.3) == pytest.approx(closed, rel=1e-5)


def test_the_second_peak_is_the_like_pairs(rocksalt):
    """12 Na-Na and 12 Cl-Cl at a/sqrt(2), from a different closed form."""
    w = P.scattering_weights(rocksalt, "X-ray")
    out = P.pair_distribution(rocksalt, r_max=12.0, dr=0.005, u_iso=0.004)
    closed = 6.0 * (w.b["Na"] ** 2 + w.b["Cl"] ** 2) / (w.b_mean ** 2)
    assert out.coordination_in(3.5, 4.4) == pytest.approx(closed, rel=1e-5)


def test_the_first_peak_sits_at_half_the_cell_edge(rocksalt):
    out = P.pair_distribution(rocksalt, r_max=8.0, dr=0.005, u_iso=0.004)
    band = (out.r > 2.0) & (out.r < 3.5)
    peak = float(out.r[band][int(np.argmax(out.R[band]))])
    assert peak == pytest.approx(A_NACL / 2.0, abs=0.006)   # one grid step


def test_g_tends_to_one_and_G_oscillates_about_zero(rocksalt):
    out = P.pair_distribution(rocksalt, r_max=14.0, dr=0.01, u_iso=0.006)
    tail = out.r > 8.0
    assert out.g[tail].mean() == pytest.approx(1.0, abs=0.05)
    assert abs(out.G[tail].mean()) < 0.2 * np.abs(out.G[tail]).max()
    assert (out.g >= -1e-9).all(), "g(r) cannot be negative"


def test_the_three_curves_are_one_curve_in_three_forms(rocksalt):
    """R, G and g are related by definitions, not by three calculations."""
    out = P.pair_distribution(rocksalt, r_max=10.0, dr=0.01, u_iso=0.006)
    baseline = 4.0 * math.pi * out.r * out.rho0
    assert out.G == pytest.approx(out.R / out.r - baseline, abs=1e-9)
    assert out.g == pytest.approx(1.0 + out.G / baseline, abs=1e-9)
    assert out.g == pytest.approx(
        out.R / (4.0 * math.pi * out.r ** 2 * out.rho0), abs=1e-9)


def test_the_density_is_the_cell_contents(rocksalt):
    out = P.pair_distribution(rocksalt, r_max=6.0, dr=0.02)
    assert out.n_cell == pytest.approx(8.0)
    assert out.rho0 == pytest.approx(8.0 / rocksalt.cell.volume, rel=1e-12)


def test_occupancy_enters_the_density_and_the_weights():
    half = _build(A_NACL, A_NACL, A_NACL, 90, 90, 90, "P 1", [
        f"{e}{i} {e} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f} "
        f"{0.5 if e == 'Na' else 1.0}"
        for i, (f, e) in enumerate(FCC_FRACS)])
    out = P.pair_distribution(half, r_max=6.0, dr=0.02)
    assert out.n_cell == pytest.approx(4 * 0.5 + 4 * 1.0)


# --- 2: two unrelated routes ------------------------------------------------

def test_the_truncation_kernel_matches_an_explicit_transform(rocksalt):
    """Convolve with the closed-form kernel, or transform, cut and come back.

    Two routes with no code in common. Agreement pins the kernel's
    normalisation, the 2/pi of the inverse transform and the odd extension all
    at once.
    """
    out = P.pair_distribution(rocksalt, r_max=20.0, dr=0.01, u_iso=0.006)
    q_max = 25.0
    by_kernel = P.apply_termination(out.r, out.G, 0.0, q_max)
    q = np.linspace(1e-6, q_max, 6000)
    by_transform = P.inverse_transform(q, P.forward_transform(out.r, out.G, q),
                                       out.r)
    inside = out.r > 0.8
    scale = np.abs(out.G).max()
    difference = np.abs(by_kernel - by_transform)[inside].max()
    assert difference < 0.01 * scale, (difference, scale)


def test_the_kernel_integrates_to_one_only_when_q_min_is_zero():
    """The integral of sin(a r)/(pi r) is 1 for every a > 0.

    So with Qmin = 0 the kernel preserves the mean of G, and with Qmin > 0 the
    two terms cancel and it removes it -- which is what excluding the small-Q
    region of a measurement does. Convergence is slow (the kernel falls off as
    1/r), so the integral is taken over a long range.
    """
    r = np.arange(-400.0, 400.0, 0.002)
    full = P.termination_kernel(r, 0.0, 20.0)
    assert float(np.trapezoid(full, r)) == pytest.approx(1.0, abs=0.01)
    cut = P.termination_kernel(r, 0.7, 30.0)
    assert float(np.trapezoid(cut, r)) == pytest.approx(0.0, abs=0.01)
    for q_min, q_max in ((0.0, 20.0), (0.7, 30.0)):
        assert P.termination_kernel(np.array([0.0]), q_min, q_max)[0] == \
            pytest.approx((q_max - q_min) / math.pi)


def test_truncation_costs_amplitude_and_makes_a_ripple(rocksalt):
    """The familiar behaviour of a laboratory-Qmax dataset."""
    heights = {}
    for q_max in (0.0, 10.0, 18.0, 30.0):
        out = P.pair_distribution(rocksalt, r_max=12.0, dr=0.01, u_iso=0.006,
                                  q_max=q_max)
        band = (out.r > 2.3) & (out.r < 3.3)
        heights[q_max] = float(out.G[band].max())
    assert heights[10.0] < heights[18.0] < heights[30.0] <= heights[0.0]
    assert heights[10.0] < 0.8 * heights[0.0]

    # And below the first peak the truncated PDF rings while the untruncated one
    # does not. The comparison has to be against the baseline, not against zero:
    # there are no pairs below the first peak, so R = 0 and G is exactly
    # -4 pi r rho0 there, which is not small. G + baseline is R/r, and it is
    # identically zero until the first peak -- so any deviation is the ripple.
    plain = P.pair_distribution(rocksalt, r_max=12.0, dr=0.01, u_iso=0.006)
    cut = P.pair_distribution(rocksalt, r_max=12.0, dr=0.01, u_iso=0.006,
                              q_max=12.0)
    below = (plain.r > 0.6) & (plain.r < 2.1)
    baseline = 4.0 * math.pi * plain.r * plain.rho0
    assert np.abs(plain.G + baseline)[below].max() < 1e-9
    assert np.abs(cut.G + baseline)[below].max() > 0.1


def test_q_damp_is_a_multiplication_in_r(rocksalt):
    """The dual of truncation, and the one most easily applied the wrong way."""
    out = P.pair_distribution(rocksalt, r_max=20.0, dr=0.01, u_iso=0.006)
    damped = P.pair_distribution(rocksalt, r_max=20.0, dr=0.01, u_iso=0.006,
                                 q_damp=0.05)
    envelope = np.exp(-0.5 * (out.r * 0.05) ** 2)
    assert damped.G == pytest.approx(out.G * envelope, abs=1e-9)
    # it damps the far end and leaves the near end alone
    assert abs(damped.G[5] - out.G[5]) < 1e-3 * max(abs(out.G[5]), 1.0)
    assert np.abs(damped.G[-50:]).max() < 0.7 * np.abs(out.G[-50:]).max()


def test_a_peak_position_does_not_depend_on_the_grid(rocksalt):
    """Binning to the nearest grid point would move it by up to dr/2.

    Someone reads a bond length off this plot, so the peak has to sit where the
    distance is however the grid falls. Each distance is therefore deposited
    linearly between its two neighbouring points.
    """
    positions = []
    for dr in (0.02, 0.011, 0.007, 0.005):
        out = P.pair_distribution(rocksalt, r_max=8.0, dr=dr, u_iso=0.004)
        band = (out.r > 2.0) & (out.r < 3.5)
        # the centroid of the peak, which is not quantised by the grid
        r, R = out.r[band], out.R[band]
        positions.append(float((r * R).sum() / R.sum()))
    assert max(positions) - min(positions) < 0.002
    assert positions[0] == pytest.approx(A_NACL / 2.0, abs=0.01)


# --- 3: invariance ----------------------------------------------------------

def test_an_origin_shift_changes_nothing(rocksalt_p1):
    shifted = _build(A_NACL, A_NACL, A_NACL, 90, 90, 90, "P 1", [
        f"{e}{i} {e} {(f[0] + 0.137) % 1:.8f} {(f[1] + 0.42) % 1:.8f} "
        f"{(f[2] + 0.911) % 1:.8f} 1.0"
        for i, (f, e) in enumerate(FCC_FRACS)])
    a = P.pair_distribution(rocksalt_p1, r_max=10.0, dr=0.01, u_iso=0.006)
    b = P.pair_distribution(shifted, r_max=10.0, dr=0.01, u_iso=0.006)
    assert b.G == pytest.approx(a.G, abs=1e-10)


def test_a_supercell_gives_the_same_pdf(rocksalt_p1):
    """Eight cells of one crystal are the same crystal.

    The strongest invariance available here: it exercises the periodic image
    search, the density, the weights and the normalisation together, and a
    2x2x2 cell is a different number of atoms and a different image list.
    """
    rows, n = [], 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for frac, element in FCC_FRACS:
                    x, y, z = [(f + o) / 2.0 for f, o in zip(frac, (i, j, k))]
                    n += 1
                    rows.append(f"{element}{n} {element} "
                                f"{x:.8f} {y:.8f} {z:.8f} 1.0")
    big = _build(2 * A_NACL, 2 * A_NACL, 2 * A_NACL, 90, 90, 90, "P 1", rows)
    small = P.pair_distribution(rocksalt_p1, r_max=10.0, dr=0.01, u_iso=0.006)
    large = P.pair_distribution(big, r_max=10.0, dr=0.01, u_iso=0.006)
    assert len(big.atoms) == 8 * len(rocksalt_p1.atoms)
    assert large.G == pytest.approx(small.G, abs=1e-9)
    assert large.rho0 == pytest.approx(small.rho0, rel=1e-12)


def test_a_wider_r_max_does_not_change_the_part_in_common(rocksalt):
    short = P.pair_distribution(rocksalt, r_max=8.0, dr=0.01, u_iso=0.006)
    long = P.pair_distribution(rocksalt, r_max=20.0, dr=0.01, u_iso=0.006)
    n = len(short.r)
    assert long.G[:n - 60] == pytest.approx(short.G[:n - 60], abs=1e-6)


# --- 4: cross-module --------------------------------------------------------

def test_the_shell_degeneracies_agree_with_radial_shells(rocksalt):
    """A separate neighbour search, counting the same neighbours."""
    from facet.core import utilities

    shells = utilities.radial_shells(rocksalt, 0, 4.5)
    assert shells[0]["element"] == "Cl"
    assert shells[0]["count"] == 6
    assert shells[0]["distance"] == pytest.approx(A_NACL / 2.0, abs=1e-3)
    assert shells[1]["count"] == 12

    w = P.scattering_weights(rocksalt, "X-ray")
    out = P.pair_distribution(rocksalt, r_max=6.0, dr=0.005, u_iso=0.004)
    # the same 6 and 12, reached through the PDF's own normalisation
    assert out.coordination_in(2.3, 3.3) == pytest.approx(
        shells[0]["count"] * w.b["Na"] * w.b["Cl"] / w.b_mean ** 2, rel=1e-4)


def test_the_u_fallback_is_the_diffraction_constant():
    """One number, so peak widths here and B there cannot drift apart."""
    assert dif.b_from_u(P.DEFAULT_U_ISO) == pytest.approx(dif.DEFAULT_B_ISO)


def test_sigma_is_the_uncorrelated_sum_and_the_correlation_narrows_it():
    plain = P.sigma_pair(0.01, 0.02, 2.0)
    assert float(plain) == pytest.approx(math.sqrt(0.03))
    narrowed = P.sigma_pair(0.01, 0.02, 2.0, delta1=0.5)
    assert float(narrowed) < float(plain)
    # and it cannot be driven imaginary at small r
    clamped = P.sigma_pair(0.01, 0.02, 0.2, delta2=5.0)
    assert np.isfinite(clamped).all() and (clamped >= P.MIN_SIGMA).all()
    # q_broad widens with r, the other way from delta1
    assert float(P.sigma_pair(0.01, 0.02, 10.0, q_broad=0.02)) > float(plain)


def test_the_correlated_path_and_the_grouped_path_agree(rocksalt):
    """Turning the correlation on changes the algorithm, not the definitions.

    With delta1 = 0 the r-dependent path must reproduce the grouped one, or the
    two branches are computing different things.
    """
    grouped = P.pair_distribution(rocksalt, r_max=8.0, dr=0.01, u_iso=0.006)
    per_pair = P.pair_distribution(rocksalt, r_max=8.0, dr=0.01, u_iso=0.006,
                                   q_broad=1e-12)
    scale = np.abs(grouped.G).max()
    assert np.abs(per_pair.G - grouped.G).max() < 0.002 * scale


# --- notes and refusals -----------------------------------------------------

def test_the_notes_say_where_u_came_from(rocksalt):
    out = P.pair_distribution(rocksalt, r_max=6.0, dr=0.02)
    joined = " ".join(out.notes)
    assert "took U from the file" in joined
    assert "was assumed" in joined
    assert "average crystal" in joined
    assert "untruncated" in joined


def test_the_notes_never_call_a_number_good_or_bad(rocksalt):
    import re

    out = P.pair_distribution(rocksalt, r_max=6.0, dr=0.02, q_max=20.0)
    text = " ".join(out.notes).lower()
    forbidden = {"good", "bad", "poor", "excellent", "correct", "incorrect",
                 "wrong", "reliable", "should"}
    assert not (set(re.findall(r"[a-z]+", text)) & forbidden)


def test_an_impossible_setting_is_refused_with_a_message(rocksalt):
    for kwargs in ({"r_max": 0.0}, {"dr": 0.0}, {"radiation": "muon"},
                   {"window": "hann"}, {"q_min": 20.0, "q_max": 10.0}):
        with pytest.raises(ValueError):
            P.pair_distribution(rocksalt, **kwargs)


def test_an_empty_structure_reports_rather_than_raising():
    """The CIF reader refuses a file with no atom sites, so this is built.

    A structure can still reach here empty -- every site hidden by an override,
    or a disorder configuration that selects nothing -- and the answer has to be
    an empty pattern with a note, not an exception in a panel.
    """
    from facet.core.structure import Cell, Structure

    empty = Structure(name="none", cell=Cell(5.0, 5.0, 5.0, 90, 90, 90),
                      sites=[], atoms=[])
    out = P.pair_distribution(empty, r_max=5.0, dr=0.02)
    assert not out.R.any()
    assert "no atoms" in " ".join(out.notes).lower()


# --- 5: real data -----------------------------------------------------------

def _bi_files(limit=6):
    if not BI_CIF_DIR.is_dir():
        return []
    return sorted(BI_CIF_DIR.rglob("*.cif"))[:limit]


@pytest.mark.parametrize("path", _bi_files() or [None])
def test_real_structures_produce_a_well_formed_pdf(path):
    if path is None:
        pytest.skip("the CIF collection is not present")
    try:
        structure = cif.read(str(path))
    except Exception:
        pytest.skip(f"{path.name} does not read")
    out = P.pair_distribution(structure, r_max=12.0, dr=0.01)
    assert np.isfinite(out.G).all()
    assert np.isfinite(out.g).all()
    assert np.isfinite(out.R).all()
    assert out.rho0 > 0
    assert out.n_pairs > 0
    assert out.n_from_file + out.n_assumed == len(structure.atoms)
    assert any("U from the file" in note for note in out.notes)


def test_the_heavy_pairs_dominate_an_xray_pdf_and_not_a_neutron_one():
    """The quantitative form of a claim the diffraction module makes in words.

    Bi2O3: three quarters of the X-ray PDF is Bi-Bi, and half the neutron PDF is
    Bi-O. This is why the trustworthy Bi-O structures are neutron refinements,
    and it is computed here rather than asserted.
    """
    sample = BI_CIF_DIR / "cifs" / "1526458_Bi2O3.cif"
    if not sample.is_file():
        pytest.skip("the sample structure is not present")
    structure = cif.read(str(sample))
    xray = P.scattering_weights(structure, "X-ray")
    neutron = P.scattering_weights(structure, "neutron")
    assert xray.pairs[("Bi", "Bi")] > 0.7
    assert xray.pairs[("O", "O")] < 0.03
    assert neutron.pairs[("Bi", "O")] > 0.4
    assert neutron.pairs[("O", "O")] > 0.2


def test_csv_export_carries_the_assumptions(rocksalt, tmp_path):
    out = P.pair_distribution(rocksalt, r_max=6.0, dr=0.02, q_max=20.0)
    path = P.to_csv(out, tmp_path / "pdf.csv", provenance=["FACET test"])
    text = Path(path).read_text(encoding="utf-8")
    assert "# FACET test" in text
    assert "average crystal" in text
    assert "Qmax = 20" in text
    body = [line for line in text.splitlines() if not line.startswith("#")]
    assert len(body) == len(out.r)
    first = body[0].split(",")
    assert float(first[0]) == pytest.approx(out.r[0])
