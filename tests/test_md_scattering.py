"""md_scattering, checked against things outside the module.

Each test says what it pins. The layers, as in tests/test_pdf.py:

1. **Another implementation in FACET.** The total G(r) of a NaCl and a quartz
   supercell, built from the frame's partials, equals ``pdf.pair_distribution``
   of the crystal, which shares no pair search, histogram or normalisation
   code with this module, to 1e-8 (measured: 2e-11 and 1.5e-12), for three
   radiations, untruncated and terminated. The reciprocal-lattice route at
   Bragg vectors equals ``diffraction.structure_factor`` of the crystal.
2. **Published numbers.** Peterson and Keen (J. Appl. Cryst. 54 (2021) 1542,
   Tables 3 and 4) list <b^2>, <b>^2 and the low-Q limit of S(Q) for SiO2,
   MnO and BaTiO3 from Sears' lengths; FACET's factors give the same.
3. **Laws.** G(r) = -4 pi rho0 r below the first contact, with rho0 from the
   CIF cell; Faber-Ziman weights sum to 1 at every Q; S(Q) -> 1 at high Q;
   the Bhatia-Thornton functions tend to 1, 0 and c_A c_B; a Matern type-I
   hard-core model (derivation in :func:`_matern_g`) has an exact g(r), and
   the partials follow it within counting statistics, with g = 0 exactly
   inside the hard core.
4. **Two routes.** The sine transform of g(r) and the sum over the
   reciprocal lattice of the box agree within the second route's own
   statistical error over 1.0 <= Q < 4.0 Å^-1 (the range and the tolerance
   are stated in that test); Bhatia-Thornton from the Faber-Ziman partials
   equals Bhatia and Thornton's own definition from the number and
   concentration modes; the sine kernel equals ``pdf.forward_transform``;
   the Lorch matrix equals ``pdf._lorch_round_trip``; G(r) from S(Q) equals
   the exact discrete termination sum.
5. **Invariance**, **refusals**, the FSDP and R_chi on curves with known
   answers, the trajectory loop's bookkeeping, no verdict words, no Qt.
"""
from __future__ import annotations

import ast
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from facet.core import bulk, cif, diffraction, md_model, pdf, readers
from facet.core import md_scattering as S

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data" / "crystals"
A_NACL = 5.6402


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def _nacl():
    """Rock salt from a hand-written CIF with no displacement parameters, so
    pdf.py uses one width for every pair (as tests/test_pdf.py builds it)."""
    text = "\n".join([
        "data_nacl", f"_cell_length_a {A_NACL}", f"_cell_length_b {A_NACL}",
        f"_cell_length_c {A_NACL}", "_cell_angle_alpha 90",
        "_cell_angle_beta 90", "_cell_angle_gamma 90",
        "_symmetry_space_group_name_H-M 'F m -3 m'", "loop_",
        "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
        "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_occupancy",
        "Na1 Na 0.0 0.0 0.0 1.0", "Cl1 Cl 0.5 0.5 0.5 1.0"]) + "\n"
    path = os.path.join(tempfile.mkdtemp(), "nacl.cif")
    Path(path).write_text(text, encoding="utf-8")
    return cif.read(path)


def _quartz_without_u():
    """The bundled quartz with its U set to 0, so pdf.py takes the fallback
    U = 0 (width MIN_SIGMA) for every atom, like the NaCl file."""
    structure = readers.read(DATA / "quartz_SiO2_cod9013321.cif")
    for site in structure.sites:
        site.u_iso = 0.0
    return structure


def _matern(lam_p: float, h_ang: float, side_ang: float, seed: int,
            species=("O", "Si"), probs=(2 / 3, 1 / 3)) -> md_model.Frame:
    """A Matern type-I hard-core model in a periodic cube.

    Poisson points of intensity lam_p; every point with another within h of
    it (periodic distance) is removed. Elements are drawn independently of
    position, so every partial g_ab(r) is the same function.
    """
    rng = np.random.default_rng(seed)
    n_parent = rng.poisson(lam_p * side_ang ** 3)
    points = rng.uniform(0.0, side_ang, size=(n_parent, 3))
    close = cKDTree(points, boxsize=side_ang).query_pairs(
        h_ang, output_type="ndarray")
    removed = np.zeros(n_parent, dtype=bool)
    removed[close.ravel()] = True
    kept = points[~removed]
    symbols = rng.choice(np.array(species), size=len(kept), p=probs)
    return md_model.frame_from_arrays(symbols, kept,
                                      box_ang=np.eye(3) * side_ang)


def _matern_g(r_ang, lam_p: float, h_ang: float) -> np.ndarray:
    """The exact g(r) of a Matern type-I process.

    Two parent points at distance r >= h both survive when no other parent
    lies in the union of their h-balls, with probability
    exp(-lam_p (2 V - V_lens(r))); one survives with exp(-lam_p V). So
    g(r) = exp(lam_p V_lens(r)) for r >= h and 0 below, where the lens of two
    h-balls at distance r has volume pi (4h + r)(2h - r)^2 / 12 for r < 2h and
    0 beyond. g is exactly 1 from r = 2h on, so S(Q) is a finite integral.
    """
    r = np.asarray(r_ang, dtype=np.float64)
    lens = np.where(r < 2 * h_ang,
                    math.pi / 12.0 * (4 * h_ang + r) * (2 * h_ang - r) ** 2,
                    0.0)
    return np.where(r < h_ang, 0.0, np.exp(lam_p * lens))


LAM_P, H_ANG = 0.1, 1.0       # number density of the survivors about 0.066 /Å^3


@pytest.fixture(scope="module")
def nacl():
    structure = _nacl()
    frame, _ = md_model.supercell_frame(structure, (4, 4, 4))
    return structure, frame, S.frame_partials(frame, r_max_ang=10.0,
                                              dr_ang=0.01)


@pytest.fixture(scope="module")
def quartz():
    structure = _quartz_without_u()
    frame, _ = md_model.supercell_frame(structure, (4, 4, 4))
    return structure, frame, S.frame_partials(frame, r_max_ang=8.0,
                                              dr_ang=0.01)


@pytest.fixture(scope="module")
def glassy():
    """A 1 794-atom disordered O/Si model (seed 100) and its partials to
    r_max = 14.9 Å, half its 30 Å box less one bin."""
    frame = _matern(LAM_P, H_ANG, 30.0, 100)
    return frame, S.frame_partials(frame, r_max_ang=14.9, dr_ang=0.01)


# ---------------------------------------------------------------------------
# 1. the crystal path: pdf.pair_distribution and diffraction.structure_factor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radiation", S.RADIATIONS)
@pytest.mark.parametrize("model", ["nacl", "quartz"])
def test_total_g_equals_pdf_pair_distribution_of_the_crystal(model, radiation,
                                                             request):
    """The supercell's G(r), broadened as pdf.py broadens, is the crystal's.

    pdf.pair_distribution searches the crystal's own images by brute force,
    weights every pair and divides by N <b>^2; this module deposits the
    frame's partial histograms from bulk's pair search and weights the
    partials. Same G(r) to 1e-8 absolute (|G| reaches 150 Å^-2); measured
    2.2e-11 (NaCl) and 1.5e-12 (quartz).
    """
    structure, frame, partials = request.getfixturevalue(model)
    r_max = float(partials.r_ang[-1])
    ref = pdf.pair_distribution(structure, r_max=r_max, dr=0.01, u_iso=0.0,
                                radiation=radiation)
    mine = S.real_space_from_partials(partials, radiation,
                                      sigma_ang=pdf.MIN_SIGMA)
    assert mine.r_ang.shape == ref.r.shape
    assert np.array_equal(mine.r_ang, ref.r)
    assert np.abs(mine.pdf_g - ref.G).max() < 1e-8
    assert mine.rho0_per_ang3 == pytest.approx(ref.rho0, rel=1e-12)


@pytest.mark.parametrize("window", ["boxcar", "Lorch"])
def test_terminated_g_equals_pdf_pair_distribution_with_the_same_qmax(nacl,
                                                                       window):
    """Qmax = 20 Å^-1 applied by both, boxcar and Lorch: still 1e-8."""
    structure, _, partials = nacl
    ref = pdf.pair_distribution(structure, r_max=10.0, dr=0.01, u_iso=0.0,
                                radiation="neutron", q_max=20.0, window=window)
    mine = S.real_space_from_partials(partials, "neutron",
                                      sigma_ang=pdf.MIN_SIGMA,
                                      q_max_inv_ang=20.0, q_window=window)
    assert np.abs(mine.pdf_g - ref.G).max() < 1e-8


def test_the_lorch_operator_is_pdf_lorch_round_trip(glassy):
    """The trajectory path builds pdf._lorch_round_trip's two sine matrices
    once; the two give the same G(r) to 1e-12 of its largest value."""
    _, partials = glassy
    g_pdf = S.real_space_from_partials(partials, "neutron").pdf_g
    r = partials.r_ang
    ref = pdf._lorch_round_trip(r, g_pdf, 0.5, 22.0)
    mine = S._lorch_matrix(r, 0.5, 22.0) @ g_pdf
    assert np.abs(mine - ref).max() < 1e-12 * np.abs(ref).max()


def test_density_modes_at_bragg_vectors_equal_the_crystal_structure_factor():
    """At the 2x2x2 supercell's vector (2h, 2k, 2l), sum_a f_a rho_a(q) is
    8 F(hkl) of the cell, F from diffraction.structure_factor (B = 0)."""
    structure = _nacl()
    frame, _ = md_model.supercell_frame(structure, (2, 2, 2))
    cell_hkl = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1],
                         [1, 0, 0], [2, 1, 1]])
    vectors, rho = S.density_modes(frame, 2 * cell_hkl)
    for (h, k, l), q_vec in zip(cell_hkl, vectors):
        index = int(np.flatnonzero((cell_hkl == (h, k, l)).all(axis=1))[0])
        stol2 = (np.linalg.norm(q_vec) / (4 * math.pi)) ** 2
        amplitude = sum(diffraction.form_factor(e, stol2, "X-ray")
                        * rho[e][index] for e in rho)
        crystal = diffraction.structure_factor(structure, int(h), int(k),
                                               int(l), stol2=stol2,
                                               b_iso=0.0)
        assert abs(amplitude - 8 * crystal) < 1e-9 * max(1.0, abs(crystal))
    # (1 0 0) is extinct in F m -3 m; the supercell knows it too
    assert abs(rho["Na"][4]) < 1e-9 and abs(rho["Cl"][4]) < 1e-9


# ---------------------------------------------------------------------------
# 2. published numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("composition, b2_barn, b_sq_barn, s_low_q", [
    ({"Si": 1 / 3, "O": 2 / 3}, 0.282, 0.276, -0.022),
    ({"Mn": 0.5, "O": 0.5}, 0.238, 0.011, -21.1),
    ({"Ba": 0.2, "Ti": 0.2, "O": 0.6}, 0.277, 0.145, -0.911),
])
def test_neutron_averages_match_peterson_and_keen(composition, b2_barn,
                                                  b_sq_barn, s_low_q):
    """Peterson and Keen, J. Appl. Cryst. 54 (2021) 1542, Table 3 (<b_coh^2>
    and <b_coh>^2 in barn, from Sears 1992) and Table 4 (S(Q -> 0) =
    1 - <b^2>/<b>^2 at eta = 0), to the digits printed. This pins gemmi's
    lengths, the averaging and the fm^2 -> barn factor of 0.01 at once."""
    lengths = S.scattering_lengths(composition, "neutron")
    _, b_mean, b2_mean = S.faber_ziman_weights(composition, lengths)
    assert b2_mean * S.BARN_PER_FM2 == pytest.approx(b2_barn, abs=5e-4)
    assert b_mean ** 2 * S.BARN_PER_FM2 == pytest.approx(b_sq_barn, abs=5e-4)
    digits = 10 ** (math.floor(math.log10(abs(s_low_q))) - 2)
    assert 1.0 - b2_mean / b_mean ** 2 == pytest.approx(s_low_q, abs=digits)


# ---------------------------------------------------------------------------
# 3. laws
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model, contact_ang", [("nacl", 2.5), ("quartz", 1.4)])
def test_g_below_the_first_contact_is_minus_4_pi_rho0_r(model, contact_ang,
                                                        request):
    """Where g(r) = 0, G(r) = 4 pi r rho0 (g - 1) = -4 pi rho0 r exactly.
    rho0 comes from the CIF cell (atoms per cell over a^3, or over
    sqrt(3)/2 a^2 c for quartz), not from the frame."""
    structure, _, partials = request.getfixturevalue(model)
    cell = structure.cell
    if model == "nacl":
        volume = A_NACL ** 3
    else:
        volume = math.sqrt(3.0) / 2.0 * cell.a * cell.a * cell.c
    rho0 = len(structure.atoms) / volume
    below = partials.r_ang < contact_ang
    for radiation in S.RADIATIONS:
        g_pdf = S.real_space_from_partials(partials, radiation).pdf_g
        slope = g_pdf[below] / partials.r_ang[below]
        assert np.abs(slope + 4 * math.pi * rho0).max() < 1e-12


@pytest.mark.parametrize("radiation", S.RADIATIONS)
def test_faber_ziman_weights_sum_to_one_at_every_q(radiation):
    q = np.linspace(0.0, 40.0, 401)
    composition = {"Si": 0.25, "O": 0.6, "Na": 0.15}
    weights, _, _ = S.faber_ziman_weights(
        composition, S.scattering_lengths(composition, radiation, q))
    total = sum(weights.values())
    assert np.abs(total - 1.0).max() < 1e-12
    assert set(weights) == {("Na", "Na"), ("Na", "O"), ("Na", "Si"),
                            ("O", "O"), ("O", "Si"), ("Si", "Si")}


def test_the_hard_core_model_follows_its_exact_g(glassy):
    """Matern type-I law against the deposited pair counts.

    Inside the hard core there is no pair at all: the deposited count is
    exactly 0 below h - dr (a distance just above h deposits into the grid
    point below it). Outside, in 0.1 Å bins from 1.05 to 3.05 Å, the ordered
    pair count of all elements together against N rho0 sum 4 pi r^2 g dr:
    each z = (observed - expected) / sqrt(2 expected) within 5 (Poisson
    counting on unordered pairs, each counted twice). Measured on this model:
    worst |z| = 2.08 over the 20 bins.
    """
    frame, partials = glassy
    r, dr = partials.r_ang, partials.dr_ang
    counts = sum(partials.pair_hist.values())
    assert counts[r < H_ANG - dr].max() == 0.0
    expected_per_point = (frame.n_atoms * frame.number_density_per_ang3
                          * 4 * math.pi * r * r * _matern_g(r, LAM_P, H_ANG)
                          * dr)
    z = []
    for lo in np.arange(1.05, 3.05, 0.1):
        inside = (r >= lo) & (r < lo + 0.1)
        expected = expected_per_point[inside].sum()
        z.append((counts[inside].sum() - expected) / math.sqrt(2 * expected))
    assert np.abs(z).max() < 5.0, np.round(z, 2)


@pytest.mark.parametrize("radiation", S.RADIATIONS)
def test_s_tends_to_one_at_high_q(glassy, radiation):
    """S(Q) - 1 of a g(r) with a step at contact falls as 1/Q^2; for this
    model the law's leading term 4 pi rho0 h g(h+) / Q^2 is 0.0015 at
    Q = 25 Å^-1. The sine route's |S - 1| over 20-30 Å^-1 stays under 0.02
    (measured 0.008-0.010, counting noise included) while the structure at
    0.5-5 Å^-1 reaches 0.28-0.35. The 0.02 bound is this hard-core model's:
    a glass with a sharp Si-O peak keeps |S - 1| at 0.05-0.08 over the same
    range (an outside check on four LAMMPS glass models), which is its
    structure, not a defect."""
    _, partials = glassy
    q = np.arange(0.0, 30.0001, 0.02)
    sq = S.partial_structure_factors(partials, q, r_window="none")
    total = S.total_structure_factor(sq, partials.concentrations, q, radiation)
    high, low = q >= 20.0, (q > 0.5) & (q < 5.0)
    assert np.abs(total.s[high] - 1.0).max() < 0.02
    assert np.abs(total.s[low] - 1.0).max() > 0.2
    assert total.f_reduced_inv_ang == pytest.approx(q * (total.s - 1.0))


def test_bhatia_thornton_tends_to_its_high_q_limits(glassy):
    """S_NN -> 1, S_NC -> 0, S_CC -> c_A c_B as every S_ab -> 1."""
    _, partials = glassy
    q = np.arange(20.0, 30.0001, 0.05)
    sq = S.partial_structure_factors(partials, q, r_window="none")
    bt = S.bhatia_thornton(sq, partials.concentrations)
    assert bt.elements == ("O", "Si")
    assert np.abs(bt.s_nn - 1.0).max() < 0.02
    assert np.abs(bt.s_nc).max() < 0.02
    assert np.abs(bt.s_cc - bt.c_a * bt.c_b).max() < 0.02


# ---------------------------------------------------------------------------
# 4. two routes
# ---------------------------------------------------------------------------

def test_the_reciprocal_lattice_route_agrees_with_the_sine_route(glassy):
    """The two routes share no code: one transforms g(r) on [0, r_max], the
    other sums exp(i q.r) over the atoms at every lattice vector.

    Range: 1.0 <= Q < 4.0 Å^-1 in 0.1 Å^-1 shells (30 shells, 132 to 2 142
    vectors each in this 30 Å box), r_window 'none', r_max = 14.9 Å, on
    1 794 atoms. Below 1 Å^-1 a shell holds a few dozen vectors and the sine
    route's cut at r_max (first zero of its transform at pi / r_max =
    0.21 Å^-1) is no longer small beside the features. Tolerance:
    z = (direct - sine) / se, se the direct route's spread over the shell's
    vectors / sqrt(n / 2) (q and -q are one value); every |z| < 4 and the RMS
    of z < 1.5, for the neutron and X-ray totals and each partial. Over 20
    seeds of this model (100-119) the worst |z| and RMS were 2.37 and 0.86
    (neutron), 2.54 and 1.00 (X-ray), 3.11 and 1.17 (worst partial). An RMS
    below 1 means the two routes differ by less than the second route's own
    noise: on one configuration their errors are correlated. On an 8 109-atom
    model in a 50 Å box the same comparison gave |z| <= 1.75 from 0.36 to
    4 Å^-1 (scratch measurement, not run here).
    """
    frame, partials = glassy
    edges = np.arange(1.0, 4.0001, 0.1)
    direct = S.sq_from_positions(frame, edges, radiations=("neutron", "X-ray"))
    q = direct.q_mean_inv_ang
    assert np.isfinite(q).all() and direct.n_vectors.min() >= 100
    sine = S.partial_structure_factors(partials, q, r_window="none")
    half = np.sqrt(direct.n_vectors / 2.0)
    zs = {}
    for radiation in ("neutron", "X-ray"):
        total = S.total_structure_factor(sine, partials.concentrations, q,
                                         radiation)
        zs[radiation] = (direct.total_s[radiation] - total.s) / (
            direct.total_std[radiation] / half)
    for pair in sine:
        zs[pair] = (direct.partial_s[pair] - sine[pair]) / (
            direct.partial_std[pair] / half)
    for key, z in zs.items():
        assert np.abs(z).max() < 4.0, (key, np.round(z, 2))
        assert math.sqrt(float(np.mean(z * z))) < 1.5, (key, np.round(z, 2))


def test_bhatia_thornton_from_partials_equals_the_mode_definition(glassy):
    """Bhatia and Thornton define S_NN, S_NC, S_CC from the number mode
    N(q) = rho_A + rho_B and the concentration mode C(q) = c_B rho_A -
    c_A rho_B; sq_from_positions computes those directly at every vector.
    The conversion from the Faber-Ziman partials gives the same numbers."""
    frame, _ = glassy
    direct = S.sq_from_positions(frame, np.arange(0.5, 3.0001, 0.25))
    bt = S.bhatia_thornton(direct.partial_s, direct.concentrations)
    assert np.abs(bt.s_nn - direct.bt_modes["S_NN"]).max() < 1e-12
    assert np.abs(bt.s_nc - direct.bt_modes["S_NC"]).max() < 1e-12
    assert np.abs(bt.s_cc - direct.bt_modes["S_CC"]).max() < 1e-12


def test_f_keen_from_faber_ziman_equals_its_bhatia_thornton_form():
    """F_K = <b>^2 (S_NN - 1) + 2 <b> db S_NC + db^2 (S_CC - c_A c_B), with
    db = b_A - b_B, for arbitrary partials (Salmon 1992's decomposition)."""
    rng = np.random.default_rng(1)
    s_aa, s_ab, s_bb = rng.normal(1.0, 0.3, size=(3, 50))
    c_a, b_a, b_b = 0.37, 4.1, -2.3
    c_b = 1.0 - c_a
    bt = S.bhatia_thornton({("A", "A"): s_aa, ("A", "B"): s_ab,
                            ("B", "B"): s_bb}, {"A": c_a, "B": c_b})
    f_k = (c_a * c_a * b_a * b_a * (s_aa - 1) + c_b * c_b * b_b * b_b
           * (s_bb - 1) + 2 * c_a * c_b * b_a * b_b * (s_ab - 1))
    b_mean, db = c_a * b_a + c_b * b_b, b_a - b_b
    other = (b_mean ** 2 * (bt.s_nn - 1) + 2 * b_mean * db * bt.s_nc
             + db * db * (bt.s_cc - c_a * c_b))
    assert np.abs(f_k - other).max() < 1e-12


def test_the_sine_kernel_is_pdf_forward_transform(glassy):
    """S_ab(Q) - 1 = forward_transform(4 pi rho0 r (g_ab - 1), Q) / Q for
    Q > 0, and at Q = 0 the limit sum 4 pi rho0 r^2 (g_ab - 1) dr."""
    _, partials = glassy
    q = np.array([0.0, 0.7, 2.3, 11.0])
    sq = S.partial_structure_factors(partials, q, r_window="none")
    r, dr = partials.r_ang, partials.dr_ang
    for pair, g in partials.g.items():
        g_pdf = 4 * math.pi * partials.rho0_per_ang3 * r * (g - 1.0)
        ref = pdf.forward_transform(r, g_pdf, q[1:]) / q[1:]
        assert np.abs(sq[pair][1:] - 1.0 - ref).max() < 1e-12
        assert sq[pair][0] - 1.0 == pytest.approx(float(np.sum(g_pdf * r)) * dr,
                                                  abs=1e-12)


def test_the_lorch_r_window_damps_the_cut_at_r_max(glassy):
    """With M(r) = sinc(r / r_max) the transform of the cut no longer rings:
    on this model the sine route's Lorch S(Q) lies closer to the exact S(Q)
    of the Matern law than the unwindowed one over 0.5-5 Å^-1."""
    frame, partials = glassy
    q = np.arange(0.5, 5.0, 0.02)
    exact_r = np.linspace(0.0, 2 * H_ANG, 20001)
    kernel = np.sinc(np.outer(q, exact_r) / math.pi) * exact_r ** 2 * (
        _matern_g(exact_r, LAM_P, H_ANG) - 1.0)
    exact = 1.0 + 4 * math.pi * frame.number_density_per_ang3 * np.trapezoid(
        kernel, exact_r, axis=1)
    error = {}
    for window in S.R_WINDOWS:
        sq = S.partial_structure_factors(partials, q, r_window=window)
        total = S.total_structure_factor(sq, partials.concentrations, q,
                                         "neutron")
        error[window] = float(np.sqrt(np.mean((total.s - exact) ** 2)))
    assert error["Lorch"] < error["none"], error


def test_g_from_s_equals_the_exact_discrete_termination_sum(glassy):
    """real_space_from_sq (pdf.inverse_transform of Q [S - 1], boxcar at
    Qmax) against sum_k G_k dr [K(r - r_k) - K(r + r_k)], K the closed-form
    kernel pdf.termination_kernel: the continuous-Q limit of the same
    operation, written in the test. Within 2e-4 of max |G| at dQ = 0.002
    (measured 8.6e-5; 2.1e-4 at dQ = 0.005, so the residue is the Q
    quadrature); see the module docstring for why pdf.apply_termination
    itself is not the reference near r_max."""
    _, partials = glassy
    r, dr = partials.r_ang, partials.dr_ang
    q_max = 20.0
    q = np.arange(0.0, q_max + 0.001, 0.002)
    sq = S.partial_structure_factors(partials, q, r_window="none")
    total = S.total_structure_factor(sq, partials.concentrations, q, "neutron")
    back = S.real_space_from_sq(q, total.s, r, q_min_inv_ang=0.0,
                                q_max_inv_ang=q_max, q_window="boxcar")
    g_pdf = S.real_space_from_partials(partials, "neutron").pdf_g
    kernel = (pdf.termination_kernel(r[:, None] - r[None, :], 0.0, q_max)
              - pdf.termination_kernel(r[:, None] + r[None, :], 0.0, q_max))
    exact = kernel @ g_pdf * dr
    assert np.abs(back - exact).max() < 2e-4 * np.abs(g_pdf).max()


def test_g_from_s_of_the_lorch_r_window_is_m_times_g_terminated(glassy):
    """The sine route with r_window 'Lorch' multiplies g(r) - 1 by
    M(r) = sinc(r / r_max) before the transform, so the G(r) transformed
    back from its S(Q) (what analyse_trajectory returns as pdf_g_from_sq)
    is pdf.apply_termination of M(r) G(r), not of G(r): within 3e-3 of
    max |G| below r = 13 Å (measured 1.4e-3 at dQ = 0.005 Å^-1; the residue
    is the sinc^2(Q dr / 2) of the deposition and the Q sum), against 0.185
    from G(r) itself. With r_window 'none' both references agree with it.
    Until 2026-10-07 the note said only that the r-window 'is in it'; it
    now states M(r) G(r)."""
    _, partials = glassy
    r = partials.r_ang
    q = np.round(np.arange(0.0, 25.0001, 0.005), 6)
    g_pdf = S.real_space_from_partials(partials, "neutron").pdf_g
    scale = np.abs(g_pdf).max()
    inner = r < 13.0
    damped = pdf.apply_termination(r, np.sinc(r / r[-1]) * g_pdf, 0.0, 25.0)
    plain = pdf.apply_termination(r, g_pdf, 0.0, 25.0)
    for window, near, far in (("Lorch", damped, plain), ("none", plain, None)):
        sq = S.partial_structure_factors(partials, q, r_window=window)
        total = S.total_structure_factor(sq, partials.concentrations, q,
                                         "neutron")
        back = S.real_space_from_sq(q, total.s, r, q_min_inv_ang=0.0,
                                    q_max_inv_ang=25.0, q_window="boxcar")
        assert np.abs(back - near)[inner].max() < 3e-3 * scale, window
        if far is not None:
            assert np.abs(back - far)[inner].max() > 0.1 * scale


def test_shared_pair_blocks_give_the_frames_own_result(glassy):
    """Blocks searched once (as glass.py's single pass would share them) and
    the search frame_partials runs itself give the same g(r); with the bulk
    default d_min of 0.4 Å nothing is lost on a model with a 1 Å core."""
    frame, _ = glassy
    own = S.frame_partials(frame, r_max_ang=6.0, dr_ang=0.01)
    blocks = list(bulk.iter_pairs(frame, 6.01))
    shared = S.frame_partials(frame, blocks, r_max_ang=6.0, dr_ang=0.01)
    assert shared.n_below_d_min == 0 and shared.d_min_ang == bulk.D_MIN_ANG
    assert own.d_min_ang == 0.0
    for pair in own.g:
        assert np.array_equal(shared.g[pair], own.g[pair])


# ---------------------------------------------------------------------------
# 5. invariance
# ---------------------------------------------------------------------------

def test_translation_and_atom_order_change_nothing(glassy):
    """A rigid shift (re-wrapped) and a permutation of the rows: the same
    partials to 1e-12, the same S(q) from the positions to 1e-9."""
    frame, _ = glassy
    partials = S.frame_partials(frame, r_max_ang=6.0, dr_ang=0.01)
    rng = np.random.default_rng(3)
    shifted = md_model.frame_from_arrays(
        frame.elements, frac=frame.frac + rng.uniform(0, 1, 3),
        box_ang=frame.box_ang)
    order = rng.permutation(frame.n_atoms)
    permuted = md_model.frame_from_arrays(
        frame.elements[order], frac=frame.frac[order], box_ang=frame.box_ang)
    for other in (shifted, permuted):
        theirs = S.frame_partials(other, r_max_ang=6.0, dr_ang=0.01)
        for pair in partials.g:
            assert np.abs(theirs.g[pair] - partials.g[pair]).max() < 1e-12
    edges = np.arange(1.0, 2.0001, 0.25)
    one = S.sq_from_positions(frame, edges, radiations=("neutron",))
    for other in (shifted, permuted):
        two = S.sq_from_positions(other, edges, radiations=("neutron",))
        assert np.abs(two.total_s["neutron"] - one.total_s["neutron"]).max() \
            < 1e-9


def test_the_fast_lattice_sum_equals_the_direct_sum_on_a_tilted_box():
    """sq_from_positions factorises the sum by axis and halves it by q -> -q;
    density_modes sums directly. On 60 atoms in a strongly tilted box, every
    lattice vector enumerated here with independently built reciprocal
    vectors (a* = b x c / V ...), shell means equal to 1e-10."""
    rng = np.random.default_rng(11)
    box = np.array([[9.0, 0.0, 0.0], [3.1, 8.2, 0.0], [-2.4, 2.9, 10.5]])
    symbols = rng.choice(np.array(["O", "Si", "Na"]), size=60)
    frame = md_model.frame_from_arrays(symbols, frac=rng.uniform(0, 1, (60, 3)),
                                       box_ang=box)
    edges = np.array([0.4, 1.0, 1.7, 2.5])
    fast = S.sq_from_positions(frame, edges, radiations=("X-ray",))
    a, b, c = box
    volume = abs(np.dot(a, np.cross(b, c)))
    recip = 2 * math.pi * np.array([np.cross(b, c), np.cross(c, a),
                                    np.cross(a, b)]) / np.dot(a, np.cross(b, c))
    assert abs(volume - frame.volume_ang3) < 1e-9
    span = np.arange(-12, 13)
    hkl = np.array(np.meshgrid(span, span, span, indexing="ij")).reshape(3, -1).T
    q_norm = np.linalg.norm(hkl @ recip, axis=1)
    keep = (q_norm >= edges[0]) & (q_norm < edges[-1])
    assert (np.abs(hkl[keep]) < 12).all()          # the span was wide enough
    vectors, rho = S.density_modes(frame, hkl[keep])
    assert np.allclose(np.linalg.norm(vectors, axis=1), q_norm[keep])
    n = frame.n_atoms
    conc = {e: np.sum(frame.elements == e) / n for e in rho}
    shell = np.searchsorted(edges, q_norm[keep], side="right") - 1
    for k in range(edges.size - 1):
        inside = shell == k
        assert fast.n_vectors[k] == inside.sum()
        for (x, y), value in fast.partial_s.items():
            cross = (rho[x][inside] * np.conj(rho[y][inside])).real / n
            if x == y:
                cross = cross - conc[x]
            ref = np.mean(1.0 + cross / (conc[x] * conc[y]))
            assert value[k] == pytest.approx(ref, abs=1e-10)
        stol2 = (q_norm[keep][inside] / (4 * math.pi)) ** 2
        f = {e: diffraction.form_factor(e, stol2, "X-ray") for e in rho}
        amp = sum(f[e] * rho[e][inside] for e in rho)
        fm = sum(conc[e] * f[e] for e in rho)
        f2 = sum(conc[e] * f[e] ** 2 for e in rho)
        ref_total = np.mean(1.0 + ((amp * np.conj(amp)).real / n - f2) / fm ** 2)
        assert fast.total_s["X-ray"][k] == pytest.approx(ref_total, abs=1e-10)


# ---------------------------------------------------------------------------
# 6. refusals: an explanation, never a silent zero
# ---------------------------------------------------------------------------

def test_an_element_without_a_neutron_length_is_refused_not_zeroed():
    """gemmi's neutron92 returns 0 for Po (and diffraction.form_factor passes
    it on); here that is a refusal naming the element and the TODO."""
    with pytest.raises(ValueError, match=r"Po.*TODO: need reference"):
        S.scattering_lengths(["Po", "O"], "neutron")
    frame = md_model.frame_from_arrays(["Po", "O", "O"],
                                       [[0, 0, 0], [2, 0, 0], [0, 2.1, 0]],
                                       box_ang=np.eye(3) * 12.0)
    partials = S.frame_partials(frame, r_max_ang=5.0, dr_ang=0.05)
    with pytest.raises(ValueError, match="TODO: need reference"):
        S.real_space_from_partials(partials, "neutron")
    q = np.linspace(0, 10, 11)
    sq = S.partial_structure_factors(partials, q, r_window="none")
    with pytest.raises(ValueError, match="TODO: need reference"):
        S.total_structure_factor(sq, partials.concentrations, q, "neutron")
    with pytest.raises(ValueError, match="TODO: need reference"):
        S.sq_from_positions(frame, [0.5, 1.0], radiations=("neutron",))
    # the X-ray factor of Po exists, so the same frame goes through
    S.total_structure_factor(sq, partials.concentrations, q, "X-ray")


def test_an_element_without_an_xray_table_is_refused():
    with pytest.raises(ValueError, match=r"Es.*TODO: need reference"):
        S.scattering_lengths(["Es", "O"], "X-ray")
    with pytest.raises(ValueError, match=r"Es.*TODO: need reference"):
        S.scattering_lengths(["Es"], "electron")


def test_a_zero_mean_length_is_refused():
    with pytest.raises(ValueError, match="normalisation"):
        S.faber_ziman_weights({"A": 0.5, "B": 0.5}, {"A": 1.0, "B": -1.0})


def test_concentrations_that_are_not_fractions_are_refused():
    with pytest.raises(ValueError, match="sum"):
        S.faber_ziman_weights({"Si": 0.3, "O": 0.6}, {"Si": 4.1, "O": 5.8})


def test_r_max_beyond_half_the_box_is_refused(glassy):
    frame, _ = glassy
    with pytest.raises(ValueError, match="half the smallest perpendicular"):
        S.frame_partials(frame, r_max_ang=15.0, dr_ang=0.01)


def test_pair_blocks_of_another_frame_or_too_short_are_refused(glassy):
    frame, _ = glassy
    other = _matern(LAM_P, H_ANG, 30.0, 101)
    short = list(bulk.iter_pairs(frame, 6.0))
    with pytest.raises(ValueError, match="needs them to"):
        S.frame_partials(frame, short, r_max_ang=6.0, dr_ang=0.01)
    wrong = list(bulk.iter_pairs(other, 6.02))
    with pytest.raises(ValueError, match="pair table|another frame"):
        S.frame_partials(frame, wrong, r_max_ang=6.0, dr_ang=0.01)
    blocks = list(bulk.iter_pairs(frame, 6.02, block_atoms=500))
    assert len(blocks) > 1
    with pytest.raises(ValueError, match="cover"):
        S.frame_partials(frame, blocks[:1], r_max_ang=6.0, dr_ang=0.01)


def test_methods_without_a_stated_choice_are_refused(glassy):
    _, partials = glassy
    q = np.linspace(0, 10, 11)
    with pytest.raises(ValueError, match="no default"):
        S.partial_structure_factors(partials, q, r_window="hann")
    with pytest.raises(ValueError, match="q_window is needed"):
        S.real_space_from_partials(partials, "neutron", q_max_inv_ang=20.0)
    with pytest.raises(ValueError, match="no default"):
        S.real_space_from_sq(np.array([0.0, 1.0, 3.0]), np.ones(3),
                             partials.r_ang[:100], q_min_inv_ang=0.0,
                             q_max_inv_ang=3.0, q_window="hann")
    with pytest.raises(ValueError, match="binary"):
        S.bhatia_thornton({}, {"Si": 0.25, "O": 0.6, "Na": 0.15})


# ---------------------------------------------------------------------------
# 7. FSDP and the comparison with a measured curve, on known answers
# ---------------------------------------------------------------------------

def _gaussian_peak(q, q0=1.5, height=2.0, sigma=0.2):
    return 1.0 + height * np.exp(-0.5 * ((q - q0) / sigma) ** 2)


def test_fsdp_of_a_gaussian_peak_has_its_analytic_position_and_width():
    """On 1 + A exp(-(Q - Q0)^2 / 2 s^2): vertex at Q0; above the 'minima'
    baseline (S = 1 at the window edges) FWHM = 2 sqrt(2 ln 2) s; above
    'zero' the half height is (1 + A)/2, reached where A exp(...) = (A - 1)/2,
    so FWHM = 2 s sqrt(2 ln(2A / (A - 1)))."""
    q = np.arange(0.0, 4.0, 0.001)
    s = _gaussian_peak(q)
    above = S.fsdp(q, s, q_window_inv_ang=(0.2, 2.8), baseline="minima")
    assert above.q_peak_inv_ang == pytest.approx(1.5, abs=1e-6)
    assert above.height == pytest.approx(3.0, abs=1e-6)
    assert above.fwhm_inv_ang == pytest.approx(
        2 * math.sqrt(2 * math.log(2)) * 0.2, abs=1e-5)
    zero = S.fsdp(q, s, q_window_inv_ang=(0.2, 2.8), baseline="zero")
    assert zero.fwhm_inv_ang == pytest.approx(
        2 * 0.2 * math.sqrt(2 * math.log(2 * 2.0 / (2.0 - 1.0))), abs=1e-5)
    assert zero.repeat_distance_ang == pytest.approx(2 * math.pi / 1.5,
                                                     rel=1e-6)
    assert above.coherence_length_ang == pytest.approx(
        2 * math.pi / above.fwhm_inv_ang)


def test_fsdp_reports_rather_than_guesses():
    q = np.arange(0.0, 4.0, 0.01)
    s = _gaussian_peak(q)
    edge = S.fsdp(q, s, q_window_inv_ang=(0.2, 1.2), baseline="zero")
    assert math.isnan(edge.q_peak_inv_ang) and "edge" in edge.notes[0]
    narrow = S.fsdp(q, s, q_window_inv_ang=(1.4, 1.6), baseline="zero")
    assert narrow.q_peak_inv_ang == pytest.approx(1.5, abs=1e-6)
    assert math.isnan(narrow.fwhm_inv_ang)
    assert any("outside the window" in n for n in narrow.notes)
    with pytest.raises(ValueError, match="no default"):
        S.fsdp(q, s, q_window_inv_ang=(0.2, 2.8), baseline="median")


def test_fsdp_of_a_maximum_below_its_baseline_has_no_width():
    """A maximum of -2 over the 'zero' baseline (a partial S_ab or S_NC can
    be negative there) has no half height. Its position and height are the
    parabola vertex, 1.5 Å^-1 and -2; the width is NaN with a note. Before
    2026-10-07 the crossing search started on the wrong side of the level
    and extrapolated: FWHM -16.01 Å^-1 from 'crossings' at 9.51 and
    -6.51 Å^-1, outside both the window and the data, with no note. A
    maximum at exactly 0 gave a width of 0.0; it is NaN with a note too."""
    q = np.arange(0.0, 4.0, 0.01)
    s = -3.0 + np.exp(-0.5 * ((q - 1.5) / 0.2) ** 2)
    peak = S.fsdp(q, s, q_window_inv_ang=(0.5, 2.5), baseline="zero")
    assert peak.q_peak_inv_ang == pytest.approx(1.5, abs=1e-6)
    assert peak.height == pytest.approx(-2.0, abs=1e-6)
    for value in (peak.fwhm_inv_ang, peak.q_left_inv_ang,
                  peak.q_right_inv_ang, peak.coherence_length_ang):
        assert math.isnan(value)
    assert any("0 or less, so it has no half height" in n for n in peak.notes)
    flat = S.fsdp(q, np.where(np.abs(q - 1.5) < 0.005, 0.0, -1.0),
                  q_window_inv_ang=(0.5, 2.5), baseline="zero")
    assert math.isnan(flat.fwhm_inv_ang) and flat.notes
    # a positive peak over 'minima' is unchanged by the guard
    above = S.fsdp(q, -3.0 + 0.5 * s + 1.5, q_window_inv_ang=(0.5, 2.5),
                   baseline="minima")
    assert above.fwhm_inv_ang == pytest.approx(
        2 * math.sqrt(2 * math.log(2)) * 0.2, abs=2e-3)


def test_r_chi_on_curves_with_known_answers():
    """R_chi = sqrt(sum (y - s m)^2 / sum y^2): 0 for the curve itself, 0 for
    half the curve with the fitted s = 2, and 0.5 for half the curve at a
    given s = 1."""
    x = np.linspace(0.5, 20.0, 400)
    y = np.sin(x) / x + 1.0
    same = S.compare_with_measured(x, y, x, y, axis_name="q_inv_ang",
                                   quantity="S(Q) neutron")
    assert same.r_chi == pytest.approx(0.0, abs=1e-14)
    assert same.scale == pytest.approx(1.0)
    half = S.compare_with_measured(x, 0.5 * y, x, y, axis_name="q_inv_ang",
                                   quantity="S(Q) neutron")
    assert half.scale == pytest.approx(2.0) and half.r_chi < 1e-12
    given = S.compare_with_measured(x, 0.5 * y, x, y, axis_name="q_inv_ang",
                                    quantity="S(Q) neutron", scale=1.0)
    assert given.r_chi == pytest.approx(0.5)
    assert given.scale_source == "given"
    assert "R_chi = sqrt" in given.definition


def test_measured_points_outside_the_model_are_counted_not_used():
    x_model = np.linspace(1.0, 10.0, 91)
    x_meas = np.linspace(0.0, 12.0, 121)
    out = S.compare_with_measured(x_model, np.ones_like(x_model), x_meas,
                                  2 * np.ones_like(x_meas), axis_name="r_ang",
                                  quantity="G(r)", axis_range=(2.0, 8.0))
    assert out.n_points == int(((x_meas >= 2.0) & (x_meas <= 8.0)).sum())
    assert out.n_points + out.n_outside == x_meas.size
    assert out.axis_range == (2.0, 8.0)
    assert out.axis_name == "r_ang"
    assert out.scale == pytest.approx(2.0)


def test_read_measured_counts_the_lines_it_sets_aside(tmp_path):
    path = tmp_path / "sq.dat"
    path.write_text("# S(Q) of a glass\nQ S\n0.5 0.1\n1.0 0.9\n1.5 nan\n"
                    "2.0 1.2\n", encoding="utf-8")
    measured = S.read_measured(path, axis_name="q_inv_ang")
    assert measured.axis_values.tolist() == [0.5, 1.0, 2.0]
    assert measured.axis_name == "q_inv_ang"
    assert any("2 non-comment line" in n for n in measured.notes)


# ---------------------------------------------------------------------------
# 8. the trajectory loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_trajectory():
    """Frame 0: a 20 Å Matern model; frame 1: the same atoms rigidly shifted;
    frame 2: the same fractions in a box 0.45 times as large, too small for
    r_max = 9.9 Å."""
    first = _matern(LAM_P, H_ANG, 20.0, 5)
    shifted = md_model.frame_from_arrays(
        first.elements, frac=first.frac + np.array([0.31, 0.77, 0.05]),
        box_ang=first.box_ang)
    shrunk = md_model.frame_from_arrays(first.elements, frac=first.frac,
                                        box_ang=first.box_ang * 0.45)
    return md_model.MemoryTrajectory([first, shifted, shrunk])


def test_the_trajectory_average_is_the_frames_and_the_skips_are_recorded(
        small_trajectory):
    calls = []
    q = np.arange(0.0, 20.0001, 0.05)
    result = S.analyse_trajectory(
        small_trajectory, r_max_ang=9.9, dr_ang=0.01, q_inv_ang=q,
        r_window="Lorch", radiations=("neutron", "X-ray"),
        q_max_inv_ang=18.0, q_window="Lorch",
        fsdp_window_inv_ang=(1.0, 4.0), fsdp_baseline="minima",
        progress=lambda done, total: calls.append((done, total)))
    assert calls == [(1, 3), (2, 3), (3, 3)]
    prov = result.provenance
    assert prov.frames_used == (0, 1)
    assert "half the smallest perpendicular" in prov.frames_skipped[2]
    assert prov.method_parameters["r_window"] == "Lorch"
    # frame 1 is frame 0 shifted: the spread is rounding only
    first = S.frame_partials(small_trajectory.frame(0), r_max_ang=9.9,
                             dr_ang=0.01)
    for pair, series in result.partial_g.items():
        assert series.n_frames == 2
        assert np.abs(series.mean - first.g[pair]).max() < 1e-12
        assert np.nanmax(series.std) < 1e-12
    sq = S.partial_structure_factors(first, q, r_window="Lorch")
    total = S.total_structure_factor(sq, first.concentrations, q, "X-ray")
    assert np.abs(result.totals["X-ray"].s.mean - total.s).max() < 1e-12
    real = S.real_space_from_partials(first, "neutron", q_max_inv_ang=18.0,
                                      q_window="Lorch")
    neutron = result.totals["neutron"]
    assert np.abs(neutron.keen_t.mean - real.keen_t).max() < 1e-10
    assert neutron.keen_d.value_unit == "barn/Å^2"
    assert set(neutron.fsdp) == {"position", "height", "fwhm"}
    assert set(result.bhatia_thornton) == {"S_NN", "S_NC", "S_CC"}
    assert result.rho0.mean == pytest.approx(
        small_trajectory.frame(0).number_density_per_ang3)
    lines = prov.as_lines()
    assert any("frames used" in line for line in lines)


def test_a_cancelled_run_records_the_frames_it_did_not_analyse(
        small_trajectory):
    asked = []

    def cancelled():
        asked.append(1)
        return len(asked) > 1

    result = S.analyse_trajectory(
        small_trajectory, r_max_ang=9.9, dr_ang=0.01,
        q_inv_ang=np.arange(0.0, 10.0, 0.1), r_window="none",
        radiations=("X-ray",), cancelled=cancelled)
    assert result.provenance.frames_used == (0,)
    assert result.provenance.frames_skipped == {
        1: "not analysed: the run was cancelled",
        2: "not analysed: the run was cancelled"}


def test_trajectory_arguments_without_a_stated_choice_are_refused(
        small_trajectory):
    common = dict(r_max_ang=9.9, dr_ang=0.01, q_inv_ang=np.arange(0, 5, 0.1),
                  r_window="none")
    with pytest.raises(ValueError, match="radiations"):
        S.analyse_trajectory(small_trajectory, radiations=(), **common)
    with pytest.raises(ValueError, match="q_window is needed"):
        S.analyse_trajectory(small_trajectory, radiations=("X-ray",),
                             q_max_inv_ang=20.0, **common)
    with pytest.raises(ValueError, match="fsdp_baseline"):
        S.analyse_trajectory(small_trajectory, radiations=("X-ray",),
                             fsdp_window_inv_ang=(1.0, 3.0), **common)
    with pytest.raises(ValueError, match="outside"):
        S.analyse_trajectory(small_trajectory, radiations=("X-ray",),
                             frames=[0, 7], **common)
    with pytest.raises(ValueError, match="no frame could be analysed"):
        S.analyse_trajectory(small_trajectory, radiations=("X-ray",),
                             frames=[2], **common)


# ---------------------------------------------------------------------------
# 9. from the outside checks of 2026-10-07: each test fails on the code they
#    checked
# ---------------------------------------------------------------------------

def test_noble_gases_and_heavy_elements_keep_their_own_factors():
    """elements.normalise turns He into H, Ne into N, Kr into K, Bk into B
    and Cf into C (the symbols are missing from its tables), and the factors
    used to go through it. Here they are read for the symbol as given.
    Neutron lengths against the NIST table (Sears 1992): He 3.26, Ne 4.566,
    Kr 7.81 fm (they were -3.739, 9.36 and 3.67, the lengths of H, N and K).
    X-ray f(0) is Z for a neutral atom: He 2, Kr 36, Bk 97, Cf 98 (they were
    1, 19, 5 and 6). In a frame of SiO2 with one He, the He-O neutron weight
    follows b(He) = 3.26 fm and is positive (it was -0.077)."""
    lengths = S.scattering_lengths(["He", "Ne", "Kr"], "neutron")
    for symbol, b_fm in (("He", 3.26), ("Ne", 4.566), ("Kr", 7.81)):
        assert lengths[symbol] == pytest.approx(b_fm, abs=1e-9)
    f0 = S.scattering_lengths(["He", "Kr", "Bk", "Cf"], "X-ray")
    for symbol, z in (("He", 2), ("Kr", 36), ("Bk", 97), ("Cf", 98)):
        assert f0[symbol] == pytest.approx(z, abs=0.05)
    electron = S.scattering_lengths(["He", "H"], "electron")
    assert electron["He"] != pytest.approx(electron["H"], rel=0.05)
    frame = md_model.frame_from_arrays(
        ["He", "O", "O", "Si"], [[0, 0, 0], [2, 0, 0], [0, 2, 0], [3, 3, 3]],
        box_ang=np.eye(3) * 12.0)
    partials = S.frame_partials(frame, r_max_ang=5.0, dr_ang=0.05)
    real = S.real_space_from_partials(partials, "neutron")
    own = S.scattering_lengths(["O", "Si"], "neutron")
    own["He"] = 3.26
    weights, _, _ = S.faber_ziman_weights(partials.concentrations, own)
    assert real.weights[("He", "O")] == pytest.approx(weights[("He", "O")])
    assert real.weights[("He", "O")] > 0.0


def test_factors_equal_diffraction_form_factor_wherever_it_does_not_relabel():
    """For every element gemmi knows whose symbol elements.normalise leaves
    as it is (all but 15 of Z = 1-118), the factor here equals
    diffraction.form_factor's exactly, at every Q and for the three
    radiations, so a crystal's G(r) here still equals pdf.pair_distribution
    (section 1). Elements without a table are refused by both paths'
    callers and skipped here."""
    import gemmi

    from facet.core import elements
    q = np.linspace(0.0, 30.0, 31)
    stol2 = (q / (4 * math.pi)) ** 2
    compared = 0
    for z in range(1, 119):
        symbol = gemmi.Element(z).name
        if elements.normalise(symbol) != symbol:
            continue
        for radiation in S.RADIATIONS:
            try:
                mine = S.scattering_lengths([symbol], radiation, q)[symbol]
            except ValueError:
                continue
            ref = np.broadcast_to(diffraction.form_factor(symbol, stol2,
                                                          radiation), q.shape)
            assert np.array_equal(mine, ref), (symbol, radiation)
            compared += 1
    assert compared > 250


def test_given_neutron_lengths_enter_the_weights_and_are_recorded():
    """An 11B-enriched borate is measured with b(11B) = 6.65 fm, not natural
    boron's 5.30 fm (NIST table, Sears 1992). A given length replaces
    gemmi's for the element named, needs a stated source, and is named in
    the notes with that source; natural B carries a note on its imaginary
    part, -0.213i fm, which describes absorption. Before, no length could be
    given and no result said which lengths it used."""
    composition = {"B": 0.16, "Na": 0.14, "O": 0.6, "Si": 0.1}
    q = np.linspace(0.0, 10.0, 11)
    partial = {p: np.full(q.shape, 1.5)
               for p in S._unordered_pairs(sorted(composition))}
    natural = S.total_structure_factor(partial, composition, q, "neutron")
    source = "11B, NIST table (Sears 1992)"
    enriched = S.total_structure_factor(partial, composition, q, "neutron",
                                        neutron_lengths_fm={"B": 6.65},
                                        lengths_source=source)
    by_hand = dict(S.scattering_lengths(composition, "neutron"), B=6.65)
    weights, _, _ = S.faber_ziman_weights(composition, by_hand)
    assert enriched.weights[("B", "O")][0] == pytest.approx(
        weights[("B", "O")], rel=1e-12)
    assert natural.weights[("B", "O")][0] < 0.95 * weights[("B", "O")]
    assert any("B 6.65 (given)" in n and source in n for n in enriched.notes)
    assert any("Na 3.63" in n and "natural isotopic abundance" in n
               for n in natural.notes)
    assert any("5.3 - 0.213i fm" in n for n in natural.notes)
    assert not any("0.213i" in n for n in enriched.notes)
    with pytest.raises(ValueError, match="lengths_source"):
        S.total_structure_factor(partial, composition, q, "neutron",
                                 neutron_lengths_fm={"B": 6.65})
    with pytest.raises(ValueError, match="does not hold"):
        S.scattering_lengths(composition, "neutron",
                             neutron_lengths_fm={"Li": -2.22},
                             lengths_source="7Li")
    # a given length lets through an element gemmi has no length for
    assert S.scattering_lengths(
        ["Po"], "neutron", neutron_lengths_fm={"Po": 1.0},
        lengths_source="a test value, not a physical length") == {"Po": 1.0}


def test_the_grid_sum_mirrors_about_pi_over_dr_so_beyond_it_is_refused(glassy):
    """On r_k = k dr, F(Q) = Q [S(Q) - 1] = 4 pi rho0 sum_k dr r_k (g_k - 1)
    sin(Q r_k) obeys F(pi/dr + x) = -F(pi/dr - x): above pi / dr the sum
    returns a lower Q's value with its sign turned. Pinned with the module's
    own kernel; then every entry point refuses a Q or a Qmax beyond pi / dr
    (before, Q to 39 Å^-1 at dr = 0.1 Å was returned with the wrong sign;
    sine_route_notes stated an attenuation at such a Q until 2026-10-07).
    A boxcar Qmax of 2 pi / dr samples the termination kernel as 2/dr at lag
    0 and 0 elsewhere, so pdf.apply_termination returns 2 G(r), the factor
    an outside check measured on a glass. real_space_from_sq refuses r
    beyond pi / dQ, its mirror point in r."""
    frame, _ = glassy
    dr = 0.1
    partials = S.frame_partials(frame, r_max_ang=6.0, dr_ang=dr)
    nyquist = math.pi / dr
    x = np.array([1.0, 3.3, 7.9])
    q_pair = np.concatenate([nyquist - x, nyquist + x])
    sq = S._partial_sq(partials, S._sine_kernel(q_pair, partials.r_ang, dr,
                                                "none"))
    for values in sq.values():
        f_reduced = q_pair * (values - 1.0)
        assert np.abs(f_reduced[:3] + f_reduced[3:]).max() < \
            1e-9 * np.abs(f_reduced).max()
    with pytest.raises(ValueError, match="pi / dr"):
        S.partial_structure_factors(partials, [0.0, nyquist + 0.5],
                                    r_window="none")
    S.partial_structure_factors(partials, [0.0, nyquist - 0.5], r_window="none")
    with pytest.raises(ValueError, match="pi / dr"):
        S.real_space_from_partials(partials, "neutron", q_max_inv_ang=40.0,
                                   q_window="boxcar")
    g_pdf = S.real_space_from_partials(partials, "neutron").pdf_g
    doubled = pdf.apply_termination(partials.r_ang, g_pdf, 0.0, 2 * nyquist)
    assert np.abs(doubled - 2 * g_pdf).max() < 1e-9 * np.abs(g_pdf).max()
    with pytest.raises(ValueError, match="pi / dr"):
        S.sine_route_notes(partials, [0.0, nyquist + 0.5], r_window="none")
    q = np.arange(0.0, 10.0001, 0.5)
    with pytest.raises(ValueError, match="pi / dQ"):
        S.real_space_from_sq(q, np.ones_like(q), np.arange(0.01, 8.0, 0.01),
                             q_min_inv_ang=0.0, q_max_inv_ang=10.0,
                             q_window="boxcar")


def test_linear_deposition_attenuates_s_minus_one_by_sinc_squared(glassy):
    """A distance split linearly between two grid points is a triangle of
    half-width dr, whose transform is sinc^2(Q dr / 2): on average over the
    pairs, the grid route's pair term is the grid-free one times that factor.
    Reference: every pair within r_max from scipy's periodic cKDTree, summed
    exactly, (2/N) sum_{i<j} M(d) sin(Qd)/(Qd), M the Lorch window (0 at
    r_max, so no pair at the cut matters); all factors 1, so the total is
    the number-number S(Q). dr = 0.05 Å, r_max 6 Å. Least-squares factor
    over 10-20 and 20-30 Å^-1 between the grid route and law x reference:
    1 within 1 % (measured 1.0017 and 0.9970); without the law it follows
    sinc^2 at mid-band (measured 0.963 and 0.878, law 0.954 and 0.876). The
    notes state the factor at the grid's last Q."""
    frame, _ = glassy
    dr = 0.05
    partials = S.frame_partials(frame, r_max_ang=6.0, dr_ang=dr)
    r_end = float(partials.r_ang[-1])
    q = np.arange(10.0, 30.0001, 0.25)
    sq = S.partial_structure_factors(partials, q, r_window="Lorch")
    weights, _, _ = S.faber_ziman_weights(
        partials.concentrations, {e: 1.0 for e in partials.elements})
    n = frame.n_atoms
    uniform = 4 * math.pi * n / frame.volume_ang3 * S._sine_kernel(
        q, partials.r_ang, dr, "Lorch").sum(axis=1)
    grid_pairs = sum(weights[p] * (sq[p] - 1.0) for p in weights) + uniform
    side = frame.box_ang[0, 0]
    positions = frame.cart_ang % side
    pairs = cKDTree(positions, boxsize=side).query_pairs(
        r_end, output_type="ndarray")
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    distance = np.linalg.norm((delta + side / 2) % side - side / 2, axis=1)
    exact = 2.0 / n * (np.sinc(np.outer(q, distance) / math.pi)
                       @ np.sinc(distance / r_end))
    law = np.sinc(q * dr / (2 * math.pi)) ** 2
    for low, high in ((10.0, 20.0), (20.0, 30.0)):
        band = (q >= low) & (q <= high)
        with_law = law[band] * exact[band]
        alpha = np.dot(grid_pairs[band], with_law) / np.dot(with_law, with_law)
        beta = np.dot(grid_pairs[band], exact[band]) / np.dot(exact[band],
                                                              exact[band])
        middle = np.sinc((low + high) / 2 * dr / (2 * math.pi)) ** 2
        assert alpha == pytest.approx(1.0, abs=0.01), (low, alpha)
        assert beta == pytest.approx(middle, abs=0.02), (low, beta)
    assert beta < 0.9
    note = S.sine_route_notes(partials, q, r_window="Lorch")[2]
    assert f"{law[-1]:.4g} at the grid's last Q" in note


@pytest.mark.parametrize("window", S.R_WINDOWS)
def test_sine_route_notes_give_the_window_widths_of_the_closed_forms(glassy,
                                                                      window):
    """'none': F(Q) is convolved with sin(x R)/(pi x), first zero pi / R and
    full width at half height 2 x 1.8955 / R (sin u / u = 1/2); the Q -> 0
    response is the 3-D ball 3 (sin u - u cos u) / u^3, first zero
    4.4934 / R (tan u = u), FWHM 2 x 2.4983 / R. 'Lorch': K(Q, 0) is
    proportional to (1/Q) Int_0^R sin(pi r/R) sin(Q r) dr, exactly half its
    Q = 0 value at Q = pi / R and 0 at Q = 2 pi / R. Roots by
    scipy.optimize.brentq; the module measures on its own grid sums, which
    end at R + dr/2 in effect, hence 2e-3 relative. The old note gave pi / R
    as the first zero of S(Q)'s kernel; that is F(Q)'s."""
    from scipy.optimize import brentq
    _, partials = glassy
    r_end = float(partials.r_ang[-1])
    widths = S._window_widths(partials.r_ang, partials.dr_ang, window)
    if window == "none":
        expected = {
            "conv_zero": math.pi / r_end,
            "conv_fwhm": 2 * brentq(lambda u: math.sin(u) / u - 0.5, 1.0,
                                    2.5) / r_end,
            "k0_zero": brentq(lambda u: math.tan(u) - u, 4.0, 4.6) / r_end,
            "k0_fwhm": 2 * brentq(lambda u: 3 * (math.sin(u) - u * math.cos(u))
                                  / u ** 3 - 0.5, 1.0, 4.0) / r_end}
    else:
        expected = {"k0_zero": 2 * math.pi / r_end,
                    "k0_fwhm": 2 * math.pi / r_end}
    for key, value in expected.items():
        assert widths[key] == pytest.approx(value, rel=2e-3), key
    text = " ".join(S.sine_route_notes(partials, np.arange(0.0, 20.0, 0.01),
                                       r_window=window))
    for key in ("conv_fwhm", "conv_zero", "k0_fwhm", "k0_zero"):
        assert f"{widths[key]:.4g} Å^-1" in text, key


def test_low_q_of_a_dense_model_keeps_the_closed_box_q0_point():
    """Why like pairs stay normalised by N_a N_b / V (module docstring, 'The
    q = 0 coefficient'). A jittered simple-cubic lattice (1 728 atoms of one
    element, a = 2.5 Å, sigma = 0.15 Å, 30 Å box) has almost no number
    fluctuation: its lattice-route S(q) stays below 0.01 from 0.2 to
    0.45 Å^-1 (measured 0.002-0.007). The sine route with the Lorch window
    stays within 0.05 of 0 from Q = 0 to 0.3 Å^-1 (measured -0.039 to
    0.014), while the N_a (N_a - 1) normalisation that an outside check
    proposed would add K(Q, 0) / V, which at Q = 0 is 4 R^3 / (pi V) = 0.156
    for the Lorch window (closed form), and so put S(0) near 0.12. That
    proposal holds for a dilute model (its <dN^2>/N is near 1) and not here.
    sine_route_notes states the term at the grid's first Q."""
    rng = np.random.default_rng(7)
    side, spacing, sigma = 12, 2.5, 0.15
    nodes = np.array(np.meshgrid(*[np.arange(side)] * 3, indexing="ij")
                     ).reshape(3, -1).T
    length = side * spacing
    frame = md_model.frame_from_arrays(
        np.array(["Si"] * len(nodes)),
        (nodes * spacing + rng.normal(0.0, sigma, nodes.shape)) % length,
        box_ang=np.eye(3) * length)
    direct = S.sq_from_positions(frame, np.arange(0.15, 0.4501, 0.05),
                                 radiations=("neutron",))
    filled = direct.n_vectors > 0
    assert filled.sum() >= 4
    assert np.abs(direct.total_s["neutron"][filled]).max() < 0.01
    partials = S.frame_partials(frame, r_max_ang=14.9, dr_ang=0.01)
    q = np.array([0.0, 0.02, 0.1, 0.2, 0.3])
    s_sine = S.partial_structure_factors(partials, q, r_window="Lorch")[
        ("Si", "Si")]
    assert np.abs(s_sine).max() < 0.05
    r_end = float(partials.r_ang[-1])
    k0 = 4 * r_end ** 3 / (math.pi * frame.volume_ang3)
    assert s_sine[0] + k0 > 0.09
    # the module docstring's exact form of the term (one element, c_a = 1):
    # g x N / (N - 1) adds N / (N - 1) K(Q, 0) / V + (S - 1) / (N - 1)
    n = frame.n_atoms
    kernel = S._sine_kernel(q, partials.r_ang, partials.dr_ang, "Lorch")
    four_pi_rho0 = 4 * math.pi * partials.rho0_per_ang3
    other = 1.0 + four_pi_rho0 * kernel @ (
        partials.g[("Si", "Si")] * n / (n - 1.0) - 1.0)
    k_over_v = 4 * math.pi * kernel.sum(axis=1) / frame.volume_ang3
    added = n / (n - 1.0) * k_over_v + (s_sine - 1.0) / (n - 1.0)
    assert np.abs(other - s_sine - added).max() < 1e-12
    note = S.sine_route_notes(partials, q, r_window="Lorch")[1]
    stated = float(re.search(r"K\(Q, 0\) / V = ([-0-9.e+]+) at Q = 0 ",
                             note).group(1))
    assert stated == pytest.approx(k0, rel=1e-3)


def test_a_total_below_its_lower_bound_is_noted():
    """|sum_j f_j exp(i q.r_j)|^2 >= 0 makes S(Q) >= 1 - <f^2>/<f>^2 for a
    Faber-Ziman total; for SiO2 and neutrons that bound is -0.022 (Peterson
    and Keen's S(Q -> 0)). Partials that put the total below it, here at
    Q = 0-0.3 Å^-1, get a note naming the Q range and the bound; partials
    that do not, get none. Before, nothing marked such values."""
    q = np.linspace(0.0, 1.0, 11)
    composition = {"O": 2 / 3, "Si": 1 / 3}
    low = {("O", "O"): np.where(q < 0.35, -1.0, 1.0),
           ("O", "Si"): np.ones_like(q), ("Si", "Si"): np.ones_like(q)}
    total = S.total_structure_factor(low, composition, q, "neutron")
    _, b_mean, b2_mean = S.faber_ziman_weights(
        composition, S.scattering_lengths(composition, "neutron"))
    bound = 1.0 - b2_mean / b_mean ** 2
    assert bound == pytest.approx(-0.022, abs=5e-4)
    notes = [n for n in total.notes if n.startswith("S(Q) lies below")]
    assert len(notes) == 1
    assert "4 Q point(s) from 0 to 0.3" in notes[0]
    assert f"{bound:.4g}" in notes[0]
    fine = S.total_structure_factor({p: np.ones_like(q) for p in low},
                                    composition, q, "neutron")
    assert not any(n.startswith("S(Q) lies below") for n in fine.notes)


def test_a_fitted_xray_factor_at_or_below_zero_is_noted():
    """gemmi's it92 factor of B is 0 or below from Q = 38.22 Å^-1, because
    the fit's constant term (-0.1932) remains when the Gaussians have died
    away; an atom's form factor is positive, so a note names the element and
    the first Q on the grid. O stays positive to 45 Å^-1 and gets no note."""
    q = np.arange(0.0, 45.0, 0.01)
    composition = {"B": 0.4, "O": 0.6}
    partial = {p: np.ones_like(q) for p in (("B", "B"), ("B", "O"),
                                            ("O", "O"))}
    total = S.total_structure_factor(partial, composition, q, "X-ray")
    notes = [n for n in total.notes if "factor of B is 0 or below" in n]
    assert len(notes) == 1 and "Q = 38.22 Å^-1" in notes[0]
    assert not any("factor of O is 0" in n for n in total.notes)
    # the electron note's limit is 4 pi x ELECTRON_S_MAX_INV_ANG, computed
    electron = S.total_structure_factor(partial, composition, q, "electron")
    assert any("above Q = 25.13 Å^-1 they are extrapolated" in n
               for n in electron.notes)


def test_real_space_from_sq_weights_each_q_point_by_its_own_spacing(glassy):
    """On a uniform grid the sum is pdf.inverse_transform's rectangle rule
    (equal to 1e-12 of max |G|). A grid printed with 6 significant digits
    (spacing varying by more than 0.5 %) was refused before; it now gives
    the same G(r) within 1e-3 of max |G|. On Q = [0, 1, 3] each point carries
    its local spacing (1, 1.5, 2), written out by hand here."""
    _, partials = glassy
    r = partials.r_ang
    q = np.linspace(0.3, 25.0, 1200)
    sq = S.partial_structure_factors(partials, q, r_window="Lorch")
    s = S.total_structure_factor(sq, partials.concentrations, q, "neutron").s
    mine = S.real_space_from_sq(q, s, r, q_min_inv_ang=0.3, q_max_inv_ang=25.0,
                                q_window="boxcar")
    ref = pdf.inverse_transform(q, q * (s - 1.0), r)
    assert np.abs(mine - ref).max() < 1e-12 * np.abs(ref).max()
    printed = np.array([float(f"{v:.6g}") for v in q])
    steps = np.diff(printed)
    assert steps.max() / steps.min() > 1.005
    again = S.real_space_from_sq(printed, s, r, q_min_inv_ang=0.3,
                                 q_max_inv_ang=25.0, q_window="boxcar")
    assert np.abs(again - mine).max() < 1e-3 * np.abs(mine).max()
    x = np.array([0.4, 0.9])
    tiny = S.real_space_from_sq([0.0, 1.0, 3.0], [2.0, 3.0, 0.5], x,
                                q_min_inv_ang=0.0, q_max_inv_ang=3.0,
                                q_window="boxcar")
    by_hand = (2 / math.pi) * (1.5 * 1.0 * 2.0 * np.sin(1.0 * x)
                               + 2.0 * 3.0 * -0.5 * np.sin(3.0 * x))
    assert np.abs(tiny - by_hand).max() < 1e-14


def test_real_space_from_sq_refuses_ranges_and_values_the_data_lack():
    """Before: Qmax = 25 asked of data ending at 20 ran to 20 with a Lorch
    window built for 25, Qmin = 0 asked of data from 0.8 ran from 0.8, and
    one NaN made every G(r) NaN; none of it was said. Now each is refused
    with the range the data hold; a NaN outside [Qmin, Qmax] is not used."""
    q = np.arange(0.8, 20.0001, 0.02)
    s = 1.0 + np.sin(q) / q
    r = np.arange(0.01, 10.0, 0.01)
    common = dict(q_window="Lorch")
    with pytest.raises(ValueError, match="beyond the last Q"):
        S.real_space_from_sq(q, s, r, q_min_inv_ang=0.8, q_max_inv_ang=25.0,
                             **common)
    with pytest.raises(ValueError, match="below the first Q"):
        S.real_space_from_sq(q, s, r, q_min_inv_ang=0.0, q_max_inv_ang=20.0,
                             **common)
    g_pdf = S.real_space_from_sq(q, s, r, q_min_inv_ang=0.8,
                                 q_max_inv_ang=20.0, **common)
    assert np.isfinite(g_pdf).all()
    bad = s.copy()
    bad[100] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        S.real_space_from_sq(q, bad, r, q_min_inv_ang=0.8, q_max_inv_ang=20.0,
                             **common)
    bad = s.copy()
    bad[0] = np.nan
    assert np.isfinite(S.real_space_from_sq(
        q, bad, r, q_min_inv_ang=1.0, q_max_inv_ang=20.0, **common)).all()


def test_read_measured_refuses_decimal_commas_and_reads_bom_utf16_semicolons(
        tmp_path):
    """'0,50<tab>1,234' is two numbers written with decimal commas, which
    diffraction.read_pattern splits into four (0, 50, 1, 234), so x = 0 and
    y = 50, with no note; here the line is refused and named. A UTF-8
    byte-order mark (Excel 'CSV UTF-8') made read_pattern lose the first
    point; here the point is read and the mark noted. UTF-16 and
    semicolon-separated files are read. On a plain file the points equal
    read_pattern's. A missing file, a directory or a one-column file gets a
    sentence, not an OSError or the 2-theta message of the diffraction
    panel."""
    comma = tmp_path / "comma.txt"
    comma.write_text("0,50\t1,234\n0,52\t1,301\n0,54\t1,402\n1,10\t0,95\n",
                     encoding="utf-8")
    assert diffraction.read_pattern(comma)[1].tolist()[:3] == [50.0, 52.0,
                                                               54.0]
    with pytest.raises(ValueError, match="line 1 .*decimal comma"):
        S.read_measured(comma, axis_name="q_inv_ang")
    semicolon_comma = tmp_path / "fr.csv"
    semicolon_comma.write_text("Q;S\n0,5;1,2\n0,6;1,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 .*decimal comma"):
        S.read_measured(semicolon_comma, axis_name="q_inv_ang")
    bom = tmp_path / "bom.csv"
    bom.write_bytes(b"\xef\xbb\xbf0.5,1.0\n0.6,1.1\n0.7,1.2\n")
    assert diffraction.read_pattern(bom)[0].tolist() == [0.6, 0.7]
    measured = S.read_measured(bom, axis_name="q_inv_ang")
    assert measured.axis_values.tolist() == [0.5, 0.6, 0.7]
    assert any("byte-order mark" in n for n in measured.notes)
    utf16 = tmp_path / "u16.txt"
    utf16.write_text("Q S\n0.5 1.0\n0.6 1.1\n", encoding="utf-16")
    assert S.read_measured(utf16, axis_name="q_inv_ang").values.tolist() == \
        [1.0, 1.1]
    semicolons = tmp_path / "semi.csv"
    semicolons.write_text("0.5;1.0\n0.6;1.1\n", encoding="utf-8")
    assert S.read_measured(semicolons, axis_name="r_ang").values.tolist() == \
        [1.0, 1.1]
    plain = tmp_path / "plain.xy"
    plain.write_text("# a comment\nr G\n1.0 2.0 0.1\n0.5, 3.0\n2.0\t4.0\n",
                     encoding="utf-8")
    x, y = diffraction.read_pattern(plain)
    measured = S.read_measured(plain, axis_name="r_ang")
    assert np.array_equal(measured.axis_values, x)
    assert np.array_equal(measured.values, y)
    for target in (tmp_path / "absent.dat", tmp_path):
        with pytest.raises(ValueError, match="could not be read"):
            S.read_measured(target, axis_name="r_ang")
    one = tmp_path / "one.txt"
    one.write_text("1.0\n2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two columns") as error:
        S.read_measured(one, axis_name="r_ang")
    assert "2-theta" not in str(error.value)
    with pytest.raises(ValueError, match="no default"):
        S.read_measured(plain, axis_name="Q")


def test_compare_with_measured_records_its_inputs_and_explains_refusals():
    """The comparison stores the measured file, the model's notes, the axis
    and both grid steps. Comparing a curve with itself gives R_chi 0 on its
    own grid and more than 0 when the model comes on a grid 10 times
    coarser (its linear interpolation enters R_chi), and that case is noted.
    Bad arguments get a sentence naming the argument (before: 'too many
    values to unpack', 'could not convert string to float', and True
    accepted as a scale of 1; an axis_range beyond the model was reported
    as the empty interval '[25, 20]' until 2026-10-07)."""
    x = np.linspace(0.5, 20.0, 976)
    y = 1.0 + np.sin(3 * x) / x
    kwargs = dict(axis_name="q_inv_ang", quantity="S(Q) neutron")
    same = S.compare_with_measured(x, y, x, y, measured_source="sq.dat",
                                   model_notes=("sine route, Lorch",),
                                   **kwargs)
    assert same.r_chi < 1e-14
    assert same.measured_source == "sq.dat"
    assert same.model_notes == ("sine route, Lorch",)
    assert same.model_step == pytest.approx(same.measured_step, rel=1e-9)
    assert not any("linear interpolation between" in n for n in same.notes)
    coarse = S.compare_with_measured(x[::10], y[::10], x, y, **kwargs)
    assert coarse.r_chi > 1e-3
    assert coarse.model_step == pytest.approx(10 * coarse.measured_step,
                                              rel=1e-6)
    assert any("linear interpolation between its points" in n
               for n in coarse.notes)
    for extra, pattern in ((dict(axis_range=(1, 2, 3)), "two numbers"),
                           (dict(axis_range=("a", "b")), "is not a number"),
                           (dict(axis_range=(25.0, 30.0)),
                            r"does not overlap the model's axis, which spans "
                            r"0\.5 \.\. 20"),
                           (dict(scale=True), "scale True is not a number"),
                           (dict(scale="abc"), "scale 'abc' is not a number")):
        with pytest.raises(ValueError, match=pattern):
            S.compare_with_measured(x, y, x, y, **kwargs, **extra)
    with pytest.raises(ValueError, match="no default"):
        S.compare_with_measured(x, y, x, y, axis_name="x",
                                quantity="S(Q) neutron")


def test_partials_of_an_element_left_out_of_the_concentrations_are_refused(
        glassy):
    """Before, partials for O and Si with concentrations for O alone gave a
    total over O-O only, without a note."""
    _, partials = glassy
    q = np.linspace(0.0, 5.0, 6)
    sq = S.partial_structure_factors(partials, q, r_window="none")
    with pytest.raises(ValueError, match="concentrations hold only O"):
        S.total_structure_factor(sq, {"O": 1.0}, q, "neutron")


def test_an_element_with_one_atom_is_noted():
    """One Na among other atoms has no Na-Na pair: g Na-Na is 0 at every r
    and the sine route's S Na-Na is the window's own transform. Before, a
    value such as -146.6 at Q = 0 came with no note."""
    frame = md_model.frame_from_arrays(
        ["B", "O", "O", "Na", "O"],
        [[0, 0, 0], [1.4, 0, 0], [0, 1.4, 0], [3, 3, 3], [4, 0, 4]],
        box_ang=np.eye(3) * 12.0)
    partials = S.frame_partials(frame, r_max_ang=5.0, dr_ang=0.05)
    assert not partials.g[("Na", "Na")].any()
    assert any(n.startswith("Na has one atom") for n in partials.notes)
    assert any(n.startswith("B has one atom") for n in partials.notes)
    assert not any(n.startswith("O has one atom") for n in partials.notes)


def test_trajectory_refusals_come_before_any_pair_search(small_trajectory,
                                                         monkeypatch):
    """Each refusal here is raised before frame_partials runs (it is replaced
    by a function that fails the test). Before, the FSDP baseline and window
    were checked only inside the loop after frame 0's pair search,
    frames=[0.7, 1.2] ran frames 0 and 1 without a note, q_min without q_max
    was dropped from the provenance, radiations='neutron' was split into
    letters, and a Q beyond pi / dr ran."""
    def no_search(*args, **kwargs):
        raise AssertionError("a pair search ran before the refusal")

    monkeypatch.setattr(S, "frame_partials", no_search)
    common = dict(r_max_ang=9.9, dr_ang=0.01, q_inv_ang=np.arange(0, 5, 0.1),
                  r_window="none")
    xray = ("X-ray",)
    cases = [
        (dict(common, radiations=xray, fsdp_window_inv_ang=(1.0, 3.0),
              fsdp_baseline="median"), "no default"),
        (dict(common, radiations=xray, fsdp_window_inv_ang=(3.0, 1.0),
              fsdp_baseline="zero"), "low < high"),
        (dict(common, radiations=xray, fsdp_baseline="zero"),
         "without fsdp_window"),
        (dict(common, radiations=xray, q_min_inv_ang=5.0),
         "without q_max_inv_ang"),
        (dict(common, radiations=xray, q_min_inv_ang=-3.0), "0 or more"),
        (dict(common, radiations=xray, frames=[0.7, 1.2]),
         "not a whole number"),
        (dict(common, radiations="neutron"), "bare text"),
        (dict(common, radiations=("neutron",),
              neutron_lengths_fm={"Li": -2.22}, lengths_source="7Li"),
         "does not hold"),
        (dict(common, radiations=xray, dr_ang=0.5,
              q_inv_ang=np.arange(0, 8, 0.1)), "pi / dr"),
        (dict(common, radiations=xray, q_max_inv_ang=400.0,
              q_window="boxcar"), "pi / dr"),
    ]
    for kwargs, pattern in cases:
        with pytest.raises(ValueError, match=pattern):
            S.analyse_trajectory(small_trajectory, **kwargs)
    polonium = md_model.MemoryTrajectory([md_model.frame_from_arrays(
        ["Po", "O", "O"], [[0, 0, 0], [2, 0, 0], [0, 2.1, 0]],
        box_ang=np.eye(3) * 12.0)])
    with pytest.raises(ValueError, match="Po.*TODO: need reference"):
        S.analyse_trajectory(polonium, radiations=("neutron",),
                             **dict(common, r_max_ang=5.0))


def test_a_run_cancelled_before_its_first_frame_says_so(small_trajectory):
    """Before: 'no frame could be analysed', with no reason at all."""
    with pytest.raises(ValueError, match="cancelled before any frame"):
        S.analyse_trajectory(small_trajectory, r_max_ang=9.9, dr_ang=0.01,
                             q_inv_ang=np.arange(0.0, 5.0, 0.1),
                             r_window="none", radiations=("X-ray",),
                             cancelled=lambda: True)


def test_trajectory_results_carry_their_reasons_and_inputs(small_trajectory):
    """FSDP: with a window the Q grid does not reach, each frame's value is
    NaN and the reason ('fewer than three Q points', frames 0 and 1) is in
    the Scalar's notes (it was dropped). The provenance states r_max as
    asked (9.9, not 9.900000000000002) and the neutron length given, with
    its source. pdf_g_from_sq equals real_space_from_sq of a frame's total
    S(Q) (frames 0 and 1 are the same atoms shifted); without a Qmax it is
    None and the notes say why; with r_window 'Lorch' its note states that
    it is M(r) G(r) terminated. G(r) keeps the termination's edge note, and
    the S(Q) notes give the window widths, not pi / r_max as the first zero
    of S(Q)'s kernel; they are sine_route_notes' statements for the first
    frame (since 2026-10-07 the widths are measured once per run)."""
    q = np.arange(0.0, 20.0001, 0.05)
    source = "a test value, not a physical length"
    result = S.analyse_trajectory(
        small_trajectory, r_max_ang=9.9, dr_ang=0.01, q_inv_ang=q,
        r_window="Lorch", radiations=("neutron", "X-ray"), frames=[0, 1],
        q_max_inv_ang=18.0, q_window="Lorch",
        fsdp_window_inv_ang=(25.0, 30.0), fsdp_baseline="zero",
        neutron_lengths_fm={"Si": 4.0}, lengths_source=source)
    position = result.totals["neutron"].fsdp["position"]
    assert math.isnan(position.mean)
    assert any(n.startswith("frame(s) 0, 1: fewer than three Q points")
               for n in position.notes)
    assert any("r-window's convolution" in n for n in position.notes)
    method = result.provenance.method_parameters
    assert method["r_max_ang"] == 9.9 and method["r_last_ang"] == 9.9
    assert method["neutron_lengths_fm"] == ("Si 4",)
    assert method["lengths_source"] == source
    first = S.frame_partials(small_trajectory.frame(0), r_max_ang=9.9,
                             dr_ang=0.01)
    sq = S.partial_structure_factors(first, q, r_window="Lorch")
    for radiation, extra in (("X-ray", {}), ("neutron", dict(
            neutron_lengths_fm={"Si": 4.0}, lengths_source=source))):
        total = S.total_structure_factor(sq, first.concentrations, q,
                                         radiation, **extra)
        g_pdf = S.real_space_from_sq(q, total.s, first.r_ang,
                                     q_min_inv_ang=0.0, q_max_inv_ang=18.0,
                                     q_window="Lorch")
        from_sq = result.totals[radiation].pdf_g_from_sq
        assert np.abs(from_sq.mean - g_pdf).max() < 1e-10 * np.abs(g_pdf).max()
    neutron = result.totals["neutron"]
    _, b_mean, _ = S.faber_ziman_weights(
        first.concentrations,
        {"O": S.scattering_lengths(["O"], "neutron")["O"], "Si": 4.0})
    assert neutron.b_mean_sq_q0 == pytest.approx(S.BARN_PER_FM2 * b_mean ** 2)
    assert any("within a few pi/Qmax" in n for n in neutron.pdf_g.notes)
    assert any("Si 4 (given)" in n for n in neutron.s.notes)
    assert any("so this G(r) is M(r) G(r), terminated" in n
               for n in neutron.pdf_g_from_sq.notes)
    assert any("first zero" in n and "K(Q, 0)" in n for n in result.notes)
    # the trajectory states what sine_route_notes states for its first frame
    assert set(S.sine_route_notes(first, q, r_window="Lorch")) <= \
        set(result.notes)
    assert not any("whose first zero is at pi / r_max" in n
                   for n in result.notes)
    plain = S.analyse_trajectory(
        small_trajectory, r_max_ang=9.9, dr_ang=0.01, q_inv_ang=q,
        r_window="none", radiations=("X-ray",), frames=[0])
    assert plain.totals["X-ray"].pdf_g_from_sq is None
    assert any("needs q_max_inv_ang" in n for n in plain.totals["X-ray"].notes)


def test_the_module_docstring_quotes_the_seed_measurement_of_the_test():
    """The two-route test gives 3.11 standard errors as the worst shell over
    20 seeds (a partial); the module docstring quoted 2.54, the X-ray total,
    as the worst of totals and partials."""
    module = " ".join(S.__doc__.split())
    assert "the worst shell was 3.11 standard errors" in module
    test = " ".join(
        test_the_reciprocal_lattice_route_agrees_with_the_sine_route.__doc__
        .split())
    assert "3.11 and 1.17 (worst partial)" in test


# ---------------------------------------------------------------------------
# no verdicts, no Qt
# ---------------------------------------------------------------------------

VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def _messages(call) -> str:
    try:
        call()
    except ValueError as error:
        return str(error)
    return ""


def test_no_note_or_message_carries_a_verdict(glassy, small_trajectory,
                                              tmp_path):
    frame, partials = glassy
    q = np.arange(0.0, 30.0, 0.1)
    texts = list(partials.notes)
    sq = S.partial_structure_factors(partials, q, r_window="none")
    for radiation in S.RADIATIONS:
        texts += S.total_structure_factor(sq, partials.concentrations, q,
                                          radiation).notes
        texts += S.real_space_from_partials(
            partials, radiation, q_max_inv_ang=20.0, q_window="boxcar",
            sigma_ang=0.02).notes
    texts += S.bhatia_thornton(sq, partials.concentrations).notes
    texts += S.sq_from_positions(frame, [0.0, 0.05, 1.0],
                                 radiations=S.RADIATIONS).notes
    texts += S.fsdp(q, sq[("O", "O")], q_window_inv_ang=(0.1, 0.3),
                    baseline="zero").notes
    x = np.linspace(1, 5, 50)
    texts += S.compare_with_measured(x, x, np.linspace(0, 6, 60),
                                     np.linspace(0, 6, 60), axis_name="r_ang",
                                     quantity="G(r)").notes
    texts += S.compare_with_measured(np.linspace(1, 5, 5), np.arange(5.0),
                                     x, x, axis_name="r_ang",
                                     quantity="G(r)").notes
    path = tmp_path / "m.xy"
    path.write_bytes(b"\xef\xbb\xbfx y\n1 2\n2 3\n")
    texts += S.read_measured(path, axis_name="r_ang").notes
    texts += S.sine_route_notes(partials, q, r_window="Lorch")
    texts += S.total_structure_factor(
        {("O", "O"): np.full(q.shape, -9.0), ("O", "Si"): np.ones(q.shape),
         ("Si", "Si"): np.ones(q.shape)}, partials.concentrations, q,
        "X-ray").notes
    texts += S.total_structure_factor(
        sq, partials.concentrations, q, "neutron",
        neutron_lengths_fm={"Si": 4.0}, lengths_source="a test value").notes
    boron = md_model.frame_from_arrays(["B", "O", "O", "Na"],
                                       [[0, 0, 0], [1.4, 0, 0], [0, 1.4, 0],
                                        [3, 3, 3]], box_ang=np.eye(3) * 12.0)
    texts += S.frame_partials(boron, r_max_ang=5.0, dr_ang=0.05).notes
    texts += S.real_space_from_partials(
        S.frame_partials(boron, r_max_ang=5.0, dr_ang=0.05), "neutron").notes
    result = S.analyse_trajectory(small_trajectory, r_max_ang=9.9,
                                  dr_ang=0.01, q_inv_ang=q, r_window="none",
                                  radiations=S.RADIATIONS,
                                  q_max_inv_ang=20.0, q_window="boxcar",
                                  fsdp_window_inv_ang=(20.0, 25.0),
                                  fsdp_baseline="zero")
    texts += result.notes + tuple(result.provenance.as_lines())
    for total in result.totals.values():
        texts += total.notes + total.s.notes + total.pdf_g.notes
        texts += total.pdf_g_from_sq.notes + total.fsdp["fwhm"].notes
    texts += [_messages(lambda: S.scattering_lengths(["Po"], "neutron")),
              _messages(lambda: S.partial_structure_factors(
                  partials, [0.0, 400.0], r_window="none")),
              _messages(lambda: S.read_measured(tmp_path / "absent.dat",
                                                axis_name="r_ang")),
              _messages(lambda: S.faber_ziman_weights(
                  {"A": 0.5, "B": 0.5}, {"A": 1.0, "B": -1.0})),
              _messages(lambda: S.frame_partials(frame, r_max_ang=20.0,
                                                 dr_ang=0.01)),
              _messages(lambda: S.bhatia_thornton({}, {"A": 0.5, "B": 0.25,
                                                       "C": 0.25}))]
    assert len(texts) >= 30
    assert [t for t in texts if VERDICT.search(t)] == []


def test_no_string_in_the_module_carries_a_verdict():
    """Every string literal of md_scattering.py, docstrings included."""
    source = (ROOT / "facet" / "core" / "md_scattering.py").read_text(
        encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 50
    assert [s for s in strings if VERDICT.search(s)] == []


def test_the_module_imports_without_qt():
    code = ("import sys, facet.core.md_scattering;"
            "print('QT' if any(m.startswith(('PySide6', 'matplotlib')) "
            "for m in sys.modules) else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
