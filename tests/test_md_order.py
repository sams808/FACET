"""Local order and geometry (facet/core/md_order.py), each against an outside check.

What each group pins, and against what:

* **Wigner 3j symbols** against formulas other than the one the module
  uses (Racah's sum, DLMF 34.2.4): the closed form for m1 = m2 = m3 = 0
  (DLMF 34.3.5), the special value (j j 0; m -m 0) (DLMF 34.3.1), the
  orthogonality sums (DLMF 34.3.16, 34.3.18) and the selection rules.
* **Steinhardt q_l** on ideal fcc, bcc (first shell, and first + second),
  sc, hcp and icosahedral shells, against values computed here by two routes
  that share no code with the module and do not use scipy's spherical
  harmonics: harmonics written out from the associated Legendre polynomials
  (numpy's Legendre series, differentiated m times, with the Condon-Shortley
  phase), and the addition theorem, which needs only Legendre polynomials,
  q_l^2 = (1 / N^2) sum_jk P_l(u_j . u_k) (DLMF 14.30.9). For sc, also the
  closed forms worked by hand from the addition theorem: q_4 = sqrt(7/12),
  q_6 = sqrt(1/8).
* **Steinhardt w-hat_l** against a third route with no 3j symbol of the
  module in it: the Gaunt integral (DLMF 34.3.22) of the cubed bond density
  f(n) = (2l + 1) / (4 pi N) sum_j P_l(u_j . n) over the sphere, by a product
  quadrature that is exact for its degree (Gauss-Legendre in cos(theta),
  uniform in phi), divided by the closed form of (l l l; 0 0 0).
* **Laws** for the order parameters: rotating box and atoms together, or
  permuting the atoms, leaves q_l, w-hat_l and q-bar_l unchanged; on a
  perfect lattice q-bar_l = q_l and the global Q_l equals the per-atom q_l;
  q-bar_l on a random frame equals the neighbour average written out here.
  With one neighbour q_l = 1 for any direction (addition theorem): NaN and
  noted. The w_l sum's traced memory at l = 32 stays near two chunks.
* **q_tet**: 1 for a regular tetrahedron, 1/2 for a square-planar four
  (worked by hand: 4 x (1/3)^2 + 2 x (2/3)^2 = 4/3), and a mean of 0 over
  independent random directions (the property Errington and Debenedetti
  built it to have).
* **Polyhedron distortion** atom by atom against ``polyhedra.shape`` and
  ``polyhedra.effective_cn`` (the crystal path's own functions), on random
  polyhedra of CN 2-9, octahedra distorted past the point where their hull
  stops being the octahedron of the trans pairs, and flat ones; and on the
  six crystals unrolled into 2 x 2 x 2 supercells, every atom against
  ``coordination.analyse_site`` on its parent atom. Regular octahedron and
  tetrahedron: quadratic elongation 1, angle variance 0. Which six-ligand
  hulls are the octahedron of their trans pairs, against a classification
  from scipy's ConvexHull edges on 300 random sets of six directions.
* **Voronoi cells**: fcc gives 12 rhombic faces <0,12,0,0> of volume a^3/4,
  bcc 14 faces (6 squares, 8 hexagons) of volume a^3/2, sc 6 squares; the
  cell volumes sum to the box volume (a law, checked on a sheared random
  frame too); each cell's volume against a Monte Carlo assignment of random
  points to their nearest periodic atom, done here by brute force over
  images; the margin never changes a cell. On fcc rattled by 1e-5 Å (a
  generic set) every cell obeys Euler's relation, sum_k (6 - k) n_k = 12,
  and its faces and their edge counts equal the Delaunay duality computed
  here from a 27-copy scipy Delaunay. A duplicated atom is refused; a
  1e-6 Å contact is noted; Voronoi neighbours state their face threshold.
* **Delaunay voids**: on a simple cubic lattice every empty sphere is
  a sqrt(3)/2 - r; on fcc, 8 tetrahedral holes per cell at a sqrt(3)/4 - r
  and 16 tetrahedra of the 4 octahedral holes at a/2 - r; on a random
  sheared frame every radius equals the clearance found by brute force over
  all atoms and images, and every circumsphere is empty. A duplicated atom
  (which Qhull leaves out of the triangulation) is refused, as the Voronoi
  cells refuse it; a 1e-6 Å contact is noted.
* **Free volume**: the grid counts equal a brute-force classification of
  every grid point over every atom and image, for the geometric,
  probe-centre and probe-occupiable sets, on a sheared box and an
  orthogonal one, with a probe large enough that two- and three-sphere
  points decide most points, and through both routes to the three-sphere
  points. The probe-occupiable test written here is exact by another
  criterion than the module's: the highest point of the probe ball outside
  the expanded spheres. On a simple cubic lattice of separated spheres the
  fractions approach the closed forms 1 - (4/3) pi r^3 / a^3 and
  1 - (4/3) pi (r + r_p)^3 / a^3, and the probe-occupiable fraction equals
  the geometric one exactly.
* **Frame averages** refuse frames measured with different radii, radii
  source, grid spacing, probe radius or Voronoi thresholds.
* **Neighbour lists**: from bonds, every atom's count equals the bulk CN; from
  pairs, it equals ``bulk.distance_cn``; refusals for bonds of another frame,
  a cutoff with no source, a cutoff beyond the search radius.
* **No verdicts, units in every public argument name, and no Qt.**

Every structure here is a synthetic test input or one of the six bundled
crystals; nothing here is a reference value for a material.
"""
from __future__ import annotations

import ast
import functools
import itertools
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import legendre as npleg

from facet.core import (bulk, bv, coordination, md_model, md_order, polyhedra,
                        readers)
from facet.core.md_model import frame_from_arrays
from facet.core.neighbors import NeighborFinder, search_radius_for

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data" / "crystals"
EXAMPLES = ("quartz_SiO2_cod9013321.cif",
            "bismuth_phosphate_BiPO4_cod9008088.cif",
            "cryolite_Na3AlF6_cod9004097.cif",
            "eulytite_Bi4SiO4_3_cod9012894.cif",
            "senarmontite_Sb2O3_cod9009747.cif",
            "valentinite_Sb2O3_cod9007587.cif")

SOURCE = "test: between two shells of the ideal structure"


# ---------------------------------------------------------------------------
# independent references, written here and sharing no code with md_order
# ---------------------------------------------------------------------------

def legendre_p(l: int, x):
    """P_l(x) from numpy's Legendre series."""
    return npleg.legval(np.asarray(x, float), [0] * l + [1])


def ylm_by_hand(l: int, m: int, unit: np.ndarray) -> np.ndarray:
    """Y_lm written out: N P_l^|m|(cos theta) e^{i m phi}, Condon-Shortley
    phase, P_l^m(x) = (-1)^m (1 - x^2)^(m/2) d^m P_l / dx^m, and
    Y_l,-m = (-1)^m conj(Y_lm)."""
    am = abs(m)
    x = np.clip(unit[:, 2], -1.0, 1.0)
    phi = np.arctan2(unit[:, 1], unit[:, 0])
    derivative = npleg.legval(x, npleg.legder([0] * l + [1], am)) if am else \
        legendre_p(l, x)
    plm = (-1) ** am * (1.0 - x * x) ** (am / 2.0) * derivative
    norm = math.sqrt((2 * l + 1) / (4 * math.pi)
                     * math.factorial(l - am) / math.factorial(l + am))
    y = norm * plm * np.exp(1j * am * phi)
    return y if m >= 0 else (-1) ** am * np.conj(y)


def q_by_hand(l: int, unit: np.ndarray) -> float:
    qlm = [ylm_by_hand(l, m, unit).mean() for m in range(-l, l + 1)]
    return math.sqrt(4 * math.pi / (2 * l + 1) * sum(abs(v) ** 2 for v in qlm))


def q_by_addition(l: int, unit: np.ndarray) -> float:
    """sqrt of the addition-theorem sum. Where q_l is 0 by symmetry (q_2 of
    a cubic shell) the sum is a rounding residue of either sign, and a
    square root turns 5e-17 into 7e-9: the tests compare q_l^2 there."""
    cos = np.clip(unit @ unit.T, -1.0, 1.0)
    return math.sqrt(max(0.0, legendre_p(l, cos).sum() / unit.shape[0] ** 2))


def assert_q_equal(got, want: float, tol: float = 1e-12) -> None:
    """q_l to tol, compared as q_l^2 (see q_by_addition)."""
    got = np.atleast_1d(np.asarray(got, float))
    assert np.abs(got ** 2 - want ** 2).max() < tol, (got[:3], want)


def three_j_zero(j1: int, j2: int, j3: int) -> float:
    """(j1 j2 j3; 0 0 0), DLMF 34.3.5."""
    big = j1 + j2 + j3
    if big % 2:
        return 0.0
    f = math.factorial
    half = big // 2
    return ((-1) ** half
            * math.sqrt(f(big - 2 * j1) * f(big - 2 * j2) * f(big - 2 * j3)
                        / f(big + 1))
            * f(half) / (f(half - j1) * f(half - j2) * f(half - j3)))


def w_hat_by_gaunt(l: int, unit: np.ndarray) -> float:
    """w-hat_l from the integral of the cubed bond density (module docstring
    of this file); l even, so (l l l; 0 0 0) is not 0."""
    n = unit.shape[0]
    n_mu, n_phi = 2 * l + 2, 3 * l + 3       # exact for degree 3l
    mu, weight = npleg.leggauss(n_mu)
    phi = 2.0 * math.pi * np.arange(n_phi) / n_phi
    s = np.sqrt(1.0 - mu * mu)
    grid = np.stack([np.outer(s, np.cos(phi)), np.outer(s, np.sin(phi)),
                     np.outer(mu, np.ones(n_phi))], axis=-1).reshape(-1, 3)
    w_grid = np.repeat(weight, n_phi) * (2.0 * math.pi / n_phi)
    f = (2 * l + 1) / (4 * math.pi * n) * legendre_p(
        l, np.clip(grid @ unit.T, -1, 1)).sum(axis=1)
    integral = float((w_grid * f ** 3).sum())
    w = integral / (math.sqrt((2 * l + 1) ** 3 / (4 * math.pi))
                    * three_j_zero(l, l, l))
    power = (2 * l + 1) / (4 * math.pi * n * n) * float(
        legendre_p(l, np.clip(unit @ unit.T, -1, 1)).sum())
    return w / power ** 1.5


# ---------------------------------------------------------------------------
# ideal structures
# ---------------------------------------------------------------------------

BASIS = {"sc": [[0, 0, 0]],
         "bcc": [[0, 0, 0], [.5, .5, .5]],
         "fcc": [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]}


def cubic_frame(kind: str, n: int, a: float, shift: float = 0.25,
                element: str = "Si") -> md_model.Frame:
    """An n x n x n block of a cubic lattice, every atom displaced by
    ``shift`` lattice units along each axis so no void centre sits on a box
    face."""
    g = np.indices((n, n, n)).reshape(3, -1).T
    frac = ((g[:, None, :] + np.array(BASIS[kind])[None]).reshape(-1, 3)
            + shift) / n
    return frame_from_arrays([element] * len(frac), frac=frac,
                             box_ang=np.eye(3) * n * a)


def hcp_frame(n: int, a: float) -> md_model.Frame:
    """Ideal hcp (c/a = sqrt(8/3)) in its orthohexagonal cell a, sqrt(3) a, c."""
    c = a * math.sqrt(8.0 / 3.0)
    basis = np.array([[0, 0, 0], [.5, .5, 0], [0, 1 / 3, .5], [.5, 5 / 6, .5]])
    g = np.indices((n, n, n)).reshape(3, -1).T
    frac = ((g[:, None, :] + basis[None]).reshape(-1, 3) + 0.1) / n
    box = np.diag([a, math.sqrt(3) * a, c]) * n
    return frame_from_arrays(["Si"] * len(frac), frac=frac, box_ang=box)


def shell(kind: str) -> np.ndarray:
    """Unit vectors of the ideal neighbour shell, from the definition."""
    if kind == "sc":
        v = [p for p in itertools.product((-1, 0, 1), repeat=3)
             if sum(map(abs, p)) == 1]
    elif kind == "bcc":
        v = list(itertools.product((-1, 1), repeat=3))
    elif kind == "bcc14":
        v = list(itertools.product((-1, 1), repeat=3))
        v = [np.array(p) * 0.5 * math.sqrt(3) / math.sqrt(3) * 1.0 for p in v]
        v = [p / np.linalg.norm(p) * math.sqrt(3) / 2 for p in v]
        v += [np.array(p, float) for p in itertools.product((-1, 0, 1), repeat=3)
              if sum(map(abs, p)) == 1]
        return np.array(v) / np.linalg.norm(np.array(v), axis=1)[:, None]
    elif kind == "fcc":
        v = [p for p in itertools.product((-1, 0, 1), repeat=3)
             if sum(map(abs, p)) == 2]
    elif kind == "hcp":
        ring = [(math.cos(k * math.pi / 3), math.sin(k * math.pi / 3), 0.0)
                for k in range(6)]
        h = math.sqrt(2.0 / 3.0)
        r = math.sqrt(1.0 / 3.0)
        up = [(r * math.cos(math.pi / 6 + k * 2 * math.pi / 3),
               r * math.sin(math.pi / 6 + k * 2 * math.pi / 3), h)
              for k in range(3)]
        down = [(x, y, -z) for x, y, z in up]
        v = ring + up + down
    elif kind == "ico":
        g = (1 + math.sqrt(5)) / 2
        v = []
        for s1, s2 in itertools.product((-1, 1), repeat=2):
            v += [(0, s1, s2 * g), (s1, s2 * g, 0), (s2 * g, 0, s1)]
    else:
        raise KeyError(kind)
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1)[:, None]


def cutoff_neighbours(frame, cutoff_ang: float, pair=("Si", "Si"),
                      r_ang: float | None = None):
    r = r_ang or cutoff_ang + 1.0
    pairs = bulk.find_pairs(frame, r)
    return md_order.neighbours_from_pairs(frame, pairs, {pair: cutoff_ang},
                                          {pair: SOURCE})


def ideal(kind: str):
    """(frame, neighbours) with the shell of ``kind`` around every atom."""
    a = 3.0
    if kind == "hcp":
        frame = hcp_frame(3, a)
        return frame, cutoff_neighbours(frame, 1.1 * a)
    base = "bcc" if kind == "bcc14" else kind
    frame = cubic_frame(base, 3, a)
    cut = {"sc": 1.2, "bcc": 0.95, "bcc14": 1.2, "fcc": 0.8}[kind] * a
    return frame, cutoff_neighbours(frame, cut)


def ico_cluster():
    """An icosahedron of O around one Si, alone in a large box."""
    vec = shell("ico") * 2.0
    cart = np.vstack([[15.0, 15.0, 15.0], 15.0 + vec])
    frame = frame_from_arrays(["Si"] + ["O"] * 12, cart, box_ang=np.eye(3) * 30)
    return frame, cutoff_neighbours(frame, 2.05, pair=("Si", "O"))


def random_frame(n=60, seed=3, sheared=True, elements=("Si", "O", "O", "Na"),
                 spacing=2.4):
    """A jittered grid in a sheared box; the chemistry means nothing."""
    rng = np.random.default_rng(seed)
    side = round(n ** (1 / 3))
    g = np.indices((side,) * 3).reshape(3, -1).T
    frac = (g + 0.5 + rng.uniform(-0.3, 0.3, g.shape)) / side
    length = side * spacing
    box = np.eye(3) * length
    if sheared:
        box[1, 0], box[2, 0], box[2, 1] = 0.3 * length, 0.2 * length, -0.15 * length
    symbols = rng.choice(np.array(elements), size=len(frac))
    return frame_from_arrays(symbols, frac=frac, box_ang=box)


def rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def synthetic_neighbours(vectors_per_atom):
    """Neighbours from explicit ligand vectors, one list per centre atom.

    Uses the module's private constructor: a polyhedron given by its vectors
    needs no frame. Neighbour rows are placeholders."""
    centre, vec = [], []
    for k, v in enumerate(vectors_per_atom):
        centre += [k] * len(v)
        vec += list(v)
    vec = np.array(vec, float).reshape(-1, 3)
    n = len(vectors_per_atom)
    return md_order._neighbours(
        np.array(["Si"] * n), centre, np.zeros(len(centre), int),
        np.zeros((len(centre), 3), int), vec, np.linalg.norm(vec, axis=1),
        definition="test vectors")


# ---------------------------------------------------------------------------
# Wigner 3j
# ---------------------------------------------------------------------------

def test_3j_with_zero_m_equals_the_closed_form():
    """Racah's sum against DLMF 34.3.5, for every triangle up to j = 10,
    both parities of J (odd J is 0)."""
    worst = 0.0
    count = 0
    for j1, j2, j3 in itertools.product(range(11), repeat=3):
        if not abs(j1 - j2) <= j3 <= j1 + j2:
            continue
        got = md_order.wigner_3j(j1, j2, j3, 0, 0, 0)
        want = three_j_zero(j1, j2, j3)
        worst = max(worst, abs(got - want))
        count += 1
    assert count > 300
    assert worst < 1e-14


def test_3j_special_value_j_j_0():
    """(j j 0; m -m 0) = (-1)^(j-m) / sqrt(2j + 1), DLMF 34.3.1."""
    for j in range(13):
        for m in range(-j, j + 1):
            want = (-1) ** (j - m) / math.sqrt(2 * j + 1)
            assert md_order.wigner_3j(j, j, 0, m, -m, 0) == pytest.approx(
                want, abs=1e-15)


@pytest.mark.parametrize("j1,j2", [(2, 3), (4, 4), (6, 5), (6, 6)])
def test_3j_orthogonality(j1, j2):
    """DLMF 34.3.16 for every j3, j3' and m3 allowed (a term is nonzero only
    when m2 = -m1 - m3, so the sum over m1, m2 is a sum over m1, and a pair
    m3 != m3' is 0 by the selection rule, checked separately), and 34.3.18."""
    for j3, j3b in itertools.product(range(abs(j1 - j2), j1 + j2 + 1), repeat=2):
        for m3 in range(-min(j3, j3b), min(j3, j3b) + 1):
            total = sum((2 * j3 + 1)
                        * md_order.wigner_3j(j1, j2, j3, m1, -m1 - m3, m3)
                        * md_order.wigner_3j(j1, j2, j3b, m1, -m1 - m3, m3)
                        for m1 in range(-j1, j1 + 1))
            want = 1.0 if j3 == j3b else 0.0
            assert total == pytest.approx(want, abs=1e-13)
    j3 = j1 + j2 - 1
    squares = sum(md_order.wigner_3j(j1, j2, j3, m1, m2, m3) ** 2
                  for m1 in range(-j1, j1 + 1) for m2 in range(-j2, j2 + 1)
                  for m3 in range(-j3, j3 + 1))
    assert squares == pytest.approx(1.0, abs=1e-13)


def test_3j_selection_rules_and_refusals():
    assert md_order.wigner_3j(2, 2, 2, 1, 1, 1) == 0.0      # sum of m != 0
    assert md_order.wigner_3j(1, 1, 3, 0, 0, 0) == 0.0      # triangle
    assert md_order.wigner_3j(2, 2, 2, 3, -3, 0) == 0.0     # |m| > j
    with pytest.raises(ValueError):
        md_order.wigner_3j(1.5, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError):
        md_order.wigner_3j(-1, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Steinhardt order
# ---------------------------------------------------------------------------

SHELLS = ("sc", "bcc", "bcc14", "fcc", "hcp")


def test_the_independent_routes_agree_with_each_other_and_by_hand():
    """Before they judge the module: the two q routes agree, and the sc
    values equal the closed forms worked out from the addition theorem."""
    for kind in SHELLS + ("ico",):
        u = shell(kind)
        for l in (2, 4, 6, 8, 10):
            assert_q_equal(q_by_hand(l, u), q_by_addition(l, u), tol=1e-13)
    assert q_by_addition(4, shell("sc")) == pytest.approx(math.sqrt(7 / 12),
                                                          abs=1e-15)
    assert q_by_addition(6, shell("sc")) == pytest.approx(math.sqrt(1 / 8),
                                                          abs=1e-15)


@pytest.mark.parametrize("kind", SHELLS)
def test_q_and_w_on_ideal_lattices_equal_the_independent_routes(kind):
    """Every atom of the lattice, l = 2..10 (q) and l = 4, 6, 8 (w-hat)."""
    frame, nb = ideal(kind)
    expected_count = {"sc": 6, "bcc": 8, "bcc14": 14, "fcc": 12, "hcp": 12}
    assert set(nb.count.tolist()) == {expected_count[kind]}
    degrees = (2, 4, 6, 8, 10)
    result = md_order.steinhardt(nb, degrees)
    u = shell(kind)
    for l in degrees:
        want_q = q_by_addition(l, u)
        assert_q_equal(result.values("q", l), want_q)
        assert_q_equal(result.values("q_bar", l), want_q)
        assert_q_equal(result.global_q[degrees.index(l)], want_q)
    for l in (4, 6, 8):
        want_w = w_hat_by_gaunt(l, u)
        assert np.abs(result.values("w_hat", l) - want_w).max() < 1e-12, (kind, l)
        assert np.abs(result.values("w_bar_hat", l) - want_w).max() < 1e-12


def test_icosahedron_and_the_zero_q4_case():
    """q_4 of an icosahedron is 0, where w-hat_4 is 0/0: NaN with a note.
    q_6 and w-hat_6 against the independent routes."""
    frame, nb = ico_cluster()
    result = md_order.steinhardt(nb, (4, 6))
    u = shell("ico")
    assert result.values("q", 4)[0] < 1e-14
    assert np.isnan(result.values("w_hat", 4)[0])
    assert result.values("q", 6)[0] == pytest.approx(q_by_addition(6, u),
                                                     abs=1e-12)
    assert result.values("w_hat", 6)[0] == pytest.approx(w_hat_by_gaunt(6, u),
                                                         abs=1e-12)
    assert any("w-hat_l is 0/0" in n for n in result.notes)
    # the 12 O have no neighbour under an (Si, O) cutoff: NaN, and noted
    assert np.isnan(result.q[:, 1:]).all()
    assert any("no neighbour" in n for n in result.notes)
    # and the Si's average involves them: noted, NaN
    assert np.isnan(result.values("q_bar", 6)[0])
    assert any("neighbour that itself has no neighbour" in n
               for n in result.notes)


def test_steinhardt_on_random_frame_against_explicit_harmonics():
    """Atom by atom on a sheared random frame: q_l from the written-out
    harmonics, and q-bar_l as the explicit Lechner-Dellago average."""
    frame = random_frame(seed=5, elements=("Si",))
    nb = cutoff_neighbours(frame, 3.3)
    result = md_order.steinhardt(nb, (4, 6))
    for l in (4, 6):
        qlm = np.full((nb.n_atoms, 2 * l + 1), np.nan, complex)
        for i in range(nb.n_atoms):
            v = nb.vec_ang[nb.of(i)]
            if len(v):
                u = v / np.linalg.norm(v, axis=1)[:, None]
                qlm[i] = [ylm_by_hand(l, m, u).mean()
                          for m in range(-l, l + 1)]
        q = np.sqrt(4 * math.pi / (2 * l + 1) * (np.abs(qlm) ** 2).sum(1))
        assert np.allclose(result.values("q", l), q, rtol=0, atol=1e-12,
                           equal_nan=True)
        bar = np.array([(qlm[i] + qlm[nb.nbr[nb.of(i)]].sum(0))
                        / (nb.count[i] + 1) for i in range(nb.n_atoms)])
        qb = np.sqrt(4 * math.pi / (2 * l + 1) * (np.abs(bar) ** 2).sum(1))
        assert np.allclose(result.values("q_bar", l), qb, rtol=0, atol=1e-12,
                           equal_nan=True)
        assert np.allclose(result.qlm[result.degrees.index(l)], qlm, rtol=0,
                           atol=1e-13, equal_nan=True)


def test_order_parameters_are_invariant_under_rotation_and_permutation():
    """A law: the same frame rotated rigidly (box and atoms), or with its
    atoms listed in another order, gives the same q, w-hat and q-bar per
    atom."""
    frame = random_frame(seed=7, elements=("Si",))
    base = md_order.steinhardt(cutoff_neighbours(frame, 3.3), (4, 6))
    turned = frame_from_arrays(frame.elements, frac=frame.frac,
                               box_ang=frame.box_ang @ rotation(1).T)
    rot = md_order.steinhardt(cutoff_neighbours(turned, 3.3), (4, 6))
    perm = np.random.default_rng(2).permutation(frame.n_atoms)
    shuffled = frame_from_arrays(frame.elements[perm], frac=frame.frac[perm],
                                 box_ang=frame.box_ang)
    per = md_order.steinhardt(cutoff_neighbours(shuffled, 3.3), (4, 6))
    back = np.argsort(perm)
    for name in ("q", "w_hat", "q_bar", "w_bar_hat"):
        a = getattr(base, name)
        assert np.allclose(getattr(rot, name), a, rtol=0, atol=1e-12,
                           equal_nan=True), name
        assert np.allclose(getattr(per, name)[:, back], a, rtol=0, atol=1e-12,
                           equal_nan=True), name


def test_steinhardt_refusals():
    frame, nb = ideal("sc")
    for bad in ((0,), (4, 4), (md_order.DEGREE_MAX + 1,), (2.5,), ()):
        with pytest.raises(ValueError):
            md_order.steinhardt(nb, bad)
    with pytest.raises(ValueError):
        md_order.steinhardt("not neighbours")
    result = md_order.steinhardt(nb)
    assert result.degrees == md_order.DEGREES_DEFAULT
    with pytest.raises(ValueError):
        result.values("q", 8)


def test_one_neighbour_gives_no_q_value():
    """With one neighbour q_lm is one harmonic, so q_l = 1 and w-hat_l =
    (l l l; 0 0 0) whatever the direction (addition theorem): those are not
    measurements, so they are NaN and noted, while q_lm itself (finite)
    still feeds the neighbours' averages. Before, these atoms came out at
    q_l = 1, the value of perfect order, with no note. A Si-O pair alone in
    a box, and an O with one Si in a SiO3 fragment."""
    cart = np.array([[10.0, 10.0, 10.0], [11.2, 10.7, 10.4],
                     [20.0, 20.0, 20.0], [21.6, 20.0, 20.0],
                     [19.2, 21.4, 20.0], [19.2, 18.6, 20.0]])
    frame = frame_from_arrays(["Si", "O", "Si", "O", "O", "O"], cart,
                              box_ang=np.eye(3) * 30.0)
    pairs = bulk.find_pairs(frame, 4.0)
    cut = {("Si", "O"): 2.0, ("O", "Si"): 2.0}
    nb = md_order.neighbours_from_pairs(frame, pairs, cut,
                                        {k: SOURCE for k in cut})
    assert nb.count.tolist() == [1, 1, 3, 1, 1, 1]
    result = md_order.steinhardt(nb, (4, 6))
    single = nb.count == 1
    for l in (4, 6):
        assert np.isnan(result.values("q", l)[single]).all()
        assert np.isnan(result.values("w_hat", l)[single]).all()
        assert np.isfinite(result.values("q", l)[~single]).all()
        assert np.isfinite(result.qlm[result.degrees.index(l)]).all()
        # the neighbours' averages still use them
        assert np.isfinite(result.values("q_bar", l)).all()
    assert any("exactly one neighbour (O 4, Si 1)" in n for n in result.notes)


def test_w_sum_memory_does_not_grow_with_the_degree():
    """The w_l sum gathers q_lm over every 3j triple; the atoms per chunk
    follow from the triple count, so the traced peak stays near two chunks
    of _W_CHUNK_BYTES at l = 32 (3 169 triples). With a fixed 20 000 atoms
    per chunk the 3 000 atoms here went in one chunk of 152 MB per gathered
    array."""
    import tracemalloc

    rng = np.random.default_rng(4)
    qlm = rng.normal(size=(3000, 65)) + 1j * rng.normal(size=(3000, 65))
    md_order._w_terms(32)                    # cache the 3j table first
    tracemalloc.start()
    try:
        md_order._w_of(qlm, 32)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= 3 * md_order._W_CHUNK_BYTES, peak
    # and the chunked sum is the plain one
    i1, i2, i3, coef = md_order._w_terms(32)
    plain = ((qlm[:50, i1] * qlm[:50, i2] * qlm[:50, i3]) @ coef).real
    assert np.allclose(md_order._w_of(qlm[:50], 32), plain, rtol=1e-13,
                       atol=0)


# ---------------------------------------------------------------------------
# tetrahedral order
# ---------------------------------------------------------------------------

def test_q_tet_regular_tetrahedron_and_square_plane():
    tet = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    square = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    five = tet + [(0, 0, 3.0)]
    nb = synthetic_neighbours([np.array(tet) * 0.93, np.array(square) * 2.1,
                               np.array(five, float), [(1, 0, 0)]])
    exact = md_order.tetrahedral_order(nb)
    assert exact.q_tet[0] == pytest.approx(1.0, abs=1e-15)
    assert exact.q_tet[1] == pytest.approx(0.5, abs=1e-15)
    assert np.isnan(exact.q_tet[2]) and np.isnan(exact.q_tet[3])
    assert any("Si with 5: 1" in n and "Si with 1: 1" in n for n in exact.notes)
    nearest = md_order.tetrahedral_order(nb, selection="four nearest")
    assert nearest.q_tet[2] == pytest.approx(1.0, abs=1e-15)
    # 'four nearest' ranks the entries of the list given, not every atom:
    # the result says so (a bond list leaves out unbonded atoms)
    assert "ranked within the list" in nearest.definition
    assert "ranked within the list" not in exact.definition
    with pytest.raises(ValueError):
        md_order.tetrahedral_order(nb, selection="nearest")


def test_q_tet_of_random_directions_averages_zero():
    """Errington and Debenedetti scaled q_tet so that independent random
    directions average 0. 20 000 random quadruples: the standard error of
    the mean is below 0.004 (measured spread of one q_tet about 0.5), and
    the bound is five times that."""
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(20000, 4, 3))
    nb = synthetic_neighbours(list(vectors))
    q = md_order.tetrahedral_order(nb).q_tet
    assert abs(q.mean()) < 0.02
    assert q.max() <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# polyhedron distortion against the crystal path's functions
# ---------------------------------------------------------------------------

def _random_polyhedra(seed=4):
    rng = np.random.default_rng(seed)
    out = []
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
                     [0, 0, -1]], float)
    tet = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
                   float) / math.sqrt(3)
    for _ in range(60):
        out.append(octa * rng.uniform(1.8, 2.6, (6, 1))
                   + rng.normal(0, 0.15, (6, 3)))
        out.append(tet * rng.uniform(1.5, 1.8, (4, 1))
                   + rng.normal(0, 0.1, (4, 3)))
    for cn in (2, 3, 5, 7, 8, 9):
        for _ in range(10):
            v = rng.normal(size=(cn, 3))
            out.append(v / np.linalg.norm(v, axis=1)[:, None]
                       * rng.uniform(1.9, 2.8, (cn, 1)))
    # octahedra whose hull is not the octahedron of their trans pairs: one
    # ligand moved inside the square pyramid of the other five, or two
    # pushed together
    for _ in range(10):
        v = octa * 2.2 + rng.normal(0, 0.05, (6, 3))
        v[0] *= -0.2
        out.append(v)
        w = octa * 2.2
        w[2] = (w[0] + w[2]) / 2 + rng.normal(0, 0.05, 3)
        out.append(w)
    # a flat six: Qhull refuses it, so there is no volume
    ring = [(2 * math.cos(k * math.pi / 3), 2 * math.sin(k * math.pi / 3), 0)
            for k in range(6)]
    out.append(np.array(ring))
    return out


def test_polyhedron_shape_equals_polyhedra_shape_atom_by_atom():
    polys = _random_polyhedra()
    result = md_order.polyhedron_shape(synthetic_neighbours(polys))
    hull_fallbacks = 0
    for k, v in enumerate(polys):
        # the module sorts each atom's ligands by distance; so does this
        v = v[np.argsort(np.linalg.norm(v, axis=1), kind="stable")]
        want = polyhedra.shape(v)
        assert result.cn[k] == len(v)
        assert result.baur[k] == pytest.approx(want["baur"], rel=1e-12)
        assert result.d_mean_ang[k] == pytest.approx(want["d_mean"], rel=1e-14)
        ecn, d_av = polyhedra.effective_cn(np.linalg.norm(v, axis=1))
        assert result.ecn[k] == pytest.approx(ecn, rel=1e-10)
        assert result.ecn_d_av_ang[k] == pytest.approx(d_av, rel=1e-10)
        if len(v) == 6:
            for got, key in ((result.angle_variance_deg2[k], "angle_variance"),
                             (result.quadratic_elongation[k],
                              "quadratic_elongation"),
                             (result.volume_ang3[k], "volume")):
                if want[key] is None:
                    assert np.isnan(got), key
                else:
                    assert got == pytest.approx(want[key], rel=1e-11), key
        elif len(v) == 4:
            assert result.volume_ang3[k] == pytest.approx(want["volume"],
                                                          rel=1e-12)
        else:
            assert np.isnan(result.angle_variance_deg2[k])
    six = result.cn == 6
    assert (six & ~result.octahedral_hull).sum() >= 10    # hull fallbacks
    assert any("ConvexHull" in n for n in result.notes)
    assert any("no volume" in n for n in result.notes)


def _hull_class(v):
    """'trans = hull', 'other trans' or 'not octahedron' for six ligand
    vectors, from scipy's ConvexHull edges: an octahedron is a hull of six
    vertices of degree 4, and its trans pairs are its three non-edges; the
    largest-angle matching is found by trying every matching here."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(v)
    edges = {frozenset((int(a), int(b))) for s in hull.simplices
             for a, b in itertools.combinations(s, 2)}
    degree = [sum(k in e for e in edges) for k in range(6)]
    if len(hull.vertices) != 6 or degree != [4] * 6:
        return "not octahedron"
    u = v / np.linalg.norm(v, axis=1)[:, None]
    angle = np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))
    matchings = {frozenset(frozenset(p[2 * k:2 * k + 2]) for k in range(3))
                 for p in itertools.permutations(range(6))}
    best = max(matchings, key=lambda m: sum(angle[tuple(p)] for p in m))
    non_edges = {frozenset(p) for p in itertools.combinations(range(6), 2)
                 if frozenset(p) not in edges}
    return "trans = hull" if non_edges == best else "other trans"


def test_six_coordinated_hulls_are_classified_and_counted():
    """polyhedra.shape takes the largest-angle matching as the trans pairs
    of any six ligands; when the hull is not an octahedron, or is one whose
    non-edges are another matching, sigma^2 and <lambda> keep that
    convention (crystal parity) but no longer describe the hull's twelve
    edges. octahedral_hull flags them and the notes count both kinds,
    checked here against a ConvexHull classification on 300 random sets of
    six directions (all three kinds occur). Before, nothing flagged them."""
    rng = np.random.default_rng(13)
    polys = []
    for _ in range(300):
        v = rng.normal(size=(6, 3))
        polys.append(v / np.linalg.norm(v, axis=1)[:, None]
                     * rng.uniform(2.2, 2.6, (6, 1)))
    result = md_order.polyhedron_shape(synthetic_neighbours(polys))
    kinds = [_hull_class(v[np.argsort(np.linalg.norm(v, axis=1),
                                      kind="stable")]) for v in polys]
    counts = {k: kinds.count(k) for k in set(kinds)}
    assert min(counts.values()) >= 10 and len(counts) == 3, counts
    assert result.octahedral_hull.tolist() == [k == "trans = hull"
                                               for k in kinds]
    note = [n for n in result.notes if "six-coordinated" in n]
    assert note and (f"{counts['not octahedron']} have a convex hull that "
                     "is not an octahedron and "
                     f"{counts['other trans']} an octahedral hull") in note[0]
    # the values themselves stay those of polyhedra.shape
    for k in range(0, 300, 37):
        v = polys[k][np.argsort(np.linalg.norm(polys[k], axis=1),
                                kind="stable")]
        want = polyhedra.shape(v)["angle_variance"]
        assert result.angle_variance_deg2[k] == pytest.approx(want, rel=1e-11)


def test_tetrahedral_distortion_by_hand():
    """Robinson et al.'s tetrahedral sigma^2 and <lambda>, written out here
    with Python loops on random tetrahedra; and 0 and 1 for a regular one."""
    rng = np.random.default_rng(9)
    tet = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
                   float) / math.sqrt(3)
    polys = [tet * 1.62] + [tet * 1.6 + rng.normal(0, 0.12, (4, 3))
                            for _ in range(20)]
    result = md_order.polyhedron_shape(synthetic_neighbours(polys))
    assert result.angle_variance_deg2[0] == pytest.approx(0.0, abs=1e-20)
    assert result.quadratic_elongation[0] == pytest.approx(1.0, abs=1e-14)
    theta0 = math.degrees(math.acos(-1 / 3))
    for k, v in enumerate(polys):
        d = [math.sqrt(sum(c * c for c in p)) for p in v]
        angles = []
        for i, j in itertools.combinations(range(4), 2):
            dot = sum(v[i][c] * v[j][c] for c in range(3))
            angles.append(math.degrees(math.acos(max(-1, min(1, dot / (d[i] * d[j]))))))
        variance = sum((a - theta0) ** 2 for a in angles) / 5
        e1, e2, e3 = (v[1] - v[0]), (v[2] - v[0]), (v[3] - v[0])
        volume = abs(e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
                     - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
                     + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0])) / 6
        l0 = (9 * math.sqrt(3) * volume / 8) ** (1 / 3)
        elongation = sum((x / l0) ** 2 for x in d) / 4
        assert result.angle_variance_deg2[k] == pytest.approx(variance,
                                                              rel=1e-10, abs=1e-18)
        assert result.quadratic_elongation[k] == pytest.approx(elongation,
                                                               rel=1e-12)
        assert result.volume_ang3[k] == pytest.approx(volume, rel=1e-12)


def test_regular_octahedron_quadratic_elongation_is_one():
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
                     [0, 0, -1]], float) * 2.05
    result = md_order.polyhedron_shape(synthetic_neighbours([octa]))
    assert result.quadratic_elongation[0] == pytest.approx(1.0, abs=1e-14)
    assert result.angle_variance_deg2[0] == pytest.approx(0.0, abs=1e-20)
    assert result.volume_ang3[0] == pytest.approx(4 / 3 * 2.05 ** 3, rel=1e-14)
    assert result.baur[0] < 1e-15       # the mean of six 2.05 is not 2.05 exactly
    assert result.ecn[0] == pytest.approx(6.0, abs=1e-14)


@functools.lru_cache(maxsize=None)
def _crystal_case(name: str):
    structure = readers.read(DATA / name)
    coordination.analyse_structure(structure, cations_only=False)
    rmax = search_radius_for(structure, bv.DEFAULT, bv.V_LIST_DEFAULT)
    finder = NeighborFinder(structure, rmax=rmax)
    sites = [coordination.analyse_site(structure, finder.contacts(k),
                                       bv.DEFAULT, bv.V_BOND_DEFAULT,
                                       bv.V_LIST_DEFAULT)
             for k in range(structure.n_atoms)]
    frame, parent = md_model.supercell_frame(structure, (2, 2, 2))
    ox = np.array([structure.sites[structure.atoms[p].site_index].ox
                   for p in parent], dtype=np.int64)
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT)
    return frame, parent, sites, table


@pytest.mark.parametrize("name", EXAMPLES, ids=lambda n: n.split("_")[0])
def test_crystal_as_model_gives_the_crystal_polyhedra(name):
    """Every supercell atom against analyse_site on its parent atom: Baur
    index, mean distance, ECoN and its distance, the CN-4 hull volume, and
    for CN 6 the angle variance, quadratic elongation and volume. The two
    paths form a distance from different roundings (test_bulk_equivalence
    measured up to 51 units in the last place), so 1e-10 (1e-8 deg^2 for
    the angle variance, a sum of squared degrees). Measured worst over the
    six files: Baur 4.4e-16, ECoN 2.0e-14, angle variance 5.7e-13 deg^2,
    quadratic elongation 6.7e-16, volume 1.4e-14 Å^3, on 656 CN-4 and 448
    CN-6 atoms (quartz, BiPO4 and valentinite have no CN-6 atom)."""
    frame, parent, sites, table = _crystal_case(name)
    nb = md_order.neighbours_from_bonds(frame, bulk.bonds_at(table))
    got = md_order.polyhedron_shape(nb)
    checked = {4: 0, 6: 0}
    for row, p in enumerate(parent):
        site = sites[p]
        assert got.cn[row] == len(site.bonds)
        if not site.bonds:
            continue
        shape = site.shape
        assert got.baur[row] == pytest.approx(shape["baur"], abs=1e-10)
        assert got.d_mean_ang[row] == pytest.approx(shape["d_mean"], abs=1e-10)
        assert got.ecn[row] == pytest.approx(site.cn_ecoN, abs=1e-9)
        assert got.ecn_d_av_ang[row] == pytest.approx(site.d_ecoN, abs=1e-10)
        if got.cn[row] == 6:
            checked[6] += 1
            assert got.angle_variance_deg2[row] == pytest.approx(
                shape["angle_variance"], abs=1e-8)
            assert got.quadratic_elongation[row] == pytest.approx(
                shape["quadratic_elongation"], abs=1e-10)
            assert got.volume_ang3[row] == pytest.approx(shape["volume"],
                                                         abs=1e-9)
        elif got.cn[row] == 4:
            checked[4] += 1
            assert got.volume_ang3[row] == pytest.approx(shape["volume"],
                                                         abs=1e-10)
    assert checked[4] > 0
    if name.startswith(("cryolite", "eulytite", "senarmontite")):
        assert checked[6] > 0


# ---------------------------------------------------------------------------
# Voronoi cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,faces,index,per_cell", [
    ("fcc", 12, (0, 12, 0, 0), 4), ("bcc", 14, (0, 6, 0, 8), 2),
    ("sc", 6, (0, 6, 0, 0), 1)])
def test_voronoi_cells_of_cubic_lattices(kind, faces, index, per_cell):
    """fcc: rhombic dodecahedra; bcc: truncated octahedra (6 squares and 8
    hexagons); sc: cubes. Every cell holds a^3 / (atoms per cell)."""
    a = 2.9
    frame = cubic_frame(kind, 4, a)
    cells = md_order.voronoi_cells(frame)
    assert set(cells.n_faces.tolist()) == {faces}
    assert set(cells.index()) == {index}
    assert np.allclose(cells.volume_ang3, a ** 3 / per_cell, rtol=1e-12, atol=0)
    assert cells.volume_sum_ang3 == pytest.approx(frame.volume_ang3, rel=1e-13)


def test_voronoi_volumes_sum_to_the_box_and_match_monte_carlo():
    """On a sheared random frame: the volume law, and every cell against the
    fraction of 200 000 random points of the box whose nearest atom, among
    all 125 periodic copies of every atom listed here explicitly, it is
    (an exact nearest-point query, no tessellation). Tolerance: five
    binomial standard errors of that fraction."""
    from scipy.spatial import cKDTree

    frame = random_frame(n=27, seed=21)
    cells = md_order.voronoi_cells(frame)
    assert cells.volume_sum_ang3 == pytest.approx(frame.volume_ang3, rel=1e-12)
    rng = np.random.default_rng(22)
    samples = rng.random((200000, 3)) @ frame.box_ang
    shifts = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    images = ((frame.frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
              @ frame.box_ang)
    owner_of_image = np.tile(np.arange(frame.n_atoms), len(shifts))
    _, nearest = cKDTree(images).query(samples)
    fraction = np.bincount(owner_of_image[nearest],
                           minlength=frame.n_atoms) / len(samples)
    want = cells.volume_ang3 / frame.volume_ang3
    sigma = np.sqrt(want * (1 - want) / len(samples))
    assert (np.abs(fraction - want) < 5 * sigma).all()


def test_voronoi_neighbours_are_symmetric_and_the_margin_changes_nothing():
    frame = random_frame(n=27, seed=23)
    cells = md_order.voronoi_cells(frame)
    tight = md_order.voronoi_cells(frame, margin_ang=0.5)
    assert len(tight.notes[0].split(";")) > 3        # several tessellations
    assert np.allclose(tight.volume_ang3, cells.volume_ang3, rtol=1e-12, atol=0)
    assert np.array_equal(tight.n_faces, cells.n_faces)
    nb = md_order.neighbours_from_voronoi(frame, cells)
    entries = {(int(c), int(n), tuple(i)) for c, n, i in
               zip(nb.centre, nb.nbr, nb.image)}
    assert entries == {(n, c, tuple(-np.array(i))) for c, n, i in entries}
    assert np.array_equal(nb.count, cells.n_faces)


def _delaunay_topology(frame):
    """(faces, index) of every atom from the Delaunay triangulation of 27
    copies of the frame (scipy Delaunay, no margin logic, the middle copy
    read): atom i's Voronoi faces are its Delaunay edges, and the face
    across edge (i, j) has as many edges as there are tetrahedra around
    (i, j) (Voronoi-Delaunay duality). Exact for a generic point set."""
    from scipy.spatial import Delaunay

    n = frame.n_atoms
    shifts = np.array(list(itertools.product(range(-1, 2), repeat=3)))
    points = ((frame.frac[None] + shifts[:, None]).reshape(-1, 3)
              @ frame.box_ang)
    middle = int(np.flatnonzero((shifts == 0).all(axis=1))[0])
    tets = Delaunay(points).simplices
    pairs = np.array([(a, b) for a, b in itertools.combinations(range(4), 2)])
    ends = np.sort(tets[:, pairs].reshape(-1, 2), axis=1)
    edge, around = np.unique(ends, axis=0, return_counts=True)
    faces = np.zeros(n, int)
    index = [Counter() for _ in range(n)]
    for (a, b), k in zip(edge, around):
        for row in (a, b):
            if middle * n <= row < (middle + 1) * n:
                faces[row - middle * n] += 1
                index[row - middle * n][int(k)] += 1
    return faces, index


def test_voronoi_cells_of_a_generic_set_obey_euler_and_duality():
    """fcc rattled by up to 1e-5 Å is a generic point set: every Voronoi
    vertex is simple, so every cell has sum_k (6 - k) n_k = 12 exactly
    (Euler's relation for a simple polyhedron), and its faces and their
    edge counts are those of the Delaunay duality computed separately
    here. The old degeneracy rule (no face of area below 1e-9 spacing^2,
    its edges still counted in the neighbouring faces) dropped the faces
    of about 1e-10 Å^2 between second neighbours, which gives sums below
    12 (any user area threshold does the same, and is noted)."""
    rng = np.random.default_rng(7)
    base = cubic_frame("fcc", 3, 3.0)
    cart = base.frac @ base.box_ang + rng.uniform(-1e-5, 1e-5, (108, 3))
    frame = frame_from_arrays(["Cu"] * 108, cart, box_ang=base.box_ang)
    cells = md_order.voronoi_cells(frame)
    k = np.arange(3, 3 + cells.edge_counts.shape[1])
    assert ((cells.edge_counts * (6 - k)).sum(axis=1) == 12).all()
    faces, index = _delaunay_topology(frame)
    assert np.array_equal(cells.n_faces, faces)
    for row in range(108):
        got = {int(e): int(c) for e, c in zip(k, cells.edge_counts[row]) if c}
        assert got == dict(index[row]), row
    assert (cells.n_faces > 12).any()            # not the ideal fcc topology
    assert not any("below 12" in n for n in cells.notes)
    # an area threshold removes faces whose edges the neighbours keep: the
    # index then describes no convex polyhedron, and a note says so
    thresholded = md_order.voronoi_cells(frame, min_face_area_ang2=1e-9)
    assert any("below 12" in n and "min_face_area_ang2" in n
               for n in thresholded.notes)


def test_voronoi_refuses_coincident_atoms_and_notes_close_ones():
    """A duplicated position has no cell of its own: Qhull returns no face
    for it, which before came out as volume 0 and CN 0 with no note. It is
    refused, naming the rows. Two atoms 1e-6 Å apart do have cells, and a
    note gives the contact, below bulk.D_MIN_ANG."""
    rng = np.random.default_rng(4)
    frac = rng.uniform(0, 1, (30, 3))
    frac[29] = frac[0]
    same = frame_from_arrays(["Si"] * 15 + ["O"] * 15, frac=frac,
                             box_ang=np.eye(3) * 12.0)
    # which of the two copies Qhull leaves without a face is its choice
    with pytest.raises(ValueError, match="row (29 at 0 Å from row 0|0 at 0 "
                                         "Å from row 29)"):
        md_order.voronoi_cells(same)
    frac[29] = frac[0] + 1e-6 / 12.0
    close = frame_from_arrays(["Si"] * 15 + ["O"] * 15, frac=frac,
                              box_ang=np.eye(3) * 12.0)
    cells = md_order.voronoi_cells(close)
    assert (cells.n_faces > 0).all()
    assert any("closer than 0.4 Å" in n and "rows 0, 29" in n
               for n in cells.notes)


def test_voronoi_neighbours_state_their_face_threshold():
    """Every counted face is a full neighbour, so without a face-area
    threshold faces of any size are: the definition says so and a note
    gives the smallest face, which travel into the order parameters."""
    frame = random_frame(n=27, seed=23)
    plain = md_order.neighbours_from_voronoi(frame,
                                             md_order.voronoi_cells(frame))
    assert "no face-area threshold" in plain.definition
    note = [n for n in plain.notes if "smallest face counted" in n]
    assert note
    thresholded = md_order.neighbours_from_voronoi(
        frame, md_order.voronoi_cells(frame, min_face_area_ang2=0.5))
    assert "area above 0.5" in thresholded.definition
    assert not any("smallest face counted" in n for n in thresholded.notes)
    order = md_order.steinhardt(plain)
    assert order.definition == plain.definition


def test_voronoi_thresholds_and_index_length():
    frame = random_frame(n=27, seed=25, sheared=False)
    plain = md_order.voronoi_cells(frame)
    filtered = md_order.voronoi_cells(frame, min_face_area_ang2=0.5,
                                      min_edge_ang=0.3)
    assert np.allclose(filtered.volume_ang3, plain.volume_ang3, rtol=1e-12)
    assert filtered.n_faces.sum() <= plain.n_faces.sum()
    assert (plain.edge_counts.sum(1) <= plain.n_faces).all()
    width = plain.edge_counts.shape[1]
    assert all(len(t) == width for t in plain.index())
    assert all(len(t) == width + 2 for t in plain.index(width + 2))
    if plain.edge_counts[:, -1].any():
        with pytest.raises(ValueError):
            plain.index(width - 1)
    with pytest.raises(ValueError):
        md_order.voronoi_cells(frame, min_face_area_ang2=-1)


# ---------------------------------------------------------------------------
# Delaunay voids
# ---------------------------------------------------------------------------

def test_empty_sphere_on_simple_cubic_is_a_sqrt3_over_2_minus_r():
    a, r = 3.1, 0.7
    frame = cubic_frame("sc", 4, a, shift=0.137)
    voids = md_order.empty_spheres(frame, {"Si": r}, radii_source="test")
    assert np.allclose(voids.radius_ang, a * math.sqrt(3) / 2 - r, rtol=0,
                       atol=1e-12)
    assert voids.volume_sum_ang3 == pytest.approx(frame.volume_ang3, rel=1e-13)


def test_empty_spheres_of_fcc_are_its_two_holes():
    """8 tetrahedral holes per cell at a sqrt(3)/4 - r; the 4 octahedral
    holes, each split into 4 tetrahedra, at a/2 - r. With the lattice
    shifted by a/4, a quarter of the tetrahedral holes sit exactly on box
    faces, so this also pins that a circumcentre on a face is kept once
    (FACE_SNAP_FRAC): without the snap, 7 % of the box went missing."""
    a, r, n = 3.0, 0.4, 3
    frame = cubic_frame("fcc", n, a)
    voids = md_order.empty_spheres(frame, {"Si": r}, radii_source="test")
    tet = np.isclose(voids.radius_ang, a * math.sqrt(3) / 4 - r, atol=1e-12)
    octa = np.isclose(voids.radius_ang, a / 2 - r, atol=1e-12)
    assert (tet | octa).all()
    assert tet.sum() == 8 * n ** 3 and octa.sum() == 16 * n ** 3
    assert voids.volume_sum_ang3 == pytest.approx(frame.volume_ang3, rel=1e-13)
    on_face = np.isclose(voids.centre_frac * n, np.round(voids.centre_frac * n),
                         atol=1e-9) & np.isclose(voids.centre_frac, 0.0)
    assert on_face.any()


def test_empty_spheres_against_brute_force_clearance():
    """Sheared random frame, two radii: every radius equals min over all
    atoms and 125 images of (distance - radius), computed here; every
    circumsphere holds no atom centre; the tetrahedra fill the box."""
    frame = random_frame(n=27, seed=31, elements=("Si", "O"))
    radii = {"Si": 0.4, "O": 1.1}
    voids = md_order.empty_spheres(frame, radii, radii_source="test")
    centre = voids.centre_frac @ frame.box_ang
    shifts = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    images = ((frame.frac[None] + shifts[:, None]).reshape(-1, 3)
              @ frame.box_ang)
    r_image = np.tile([radii[s] for s in frame.elements], len(shifts))
    dist = np.linalg.norm(centre[:, None, :] - images[None, :, :], axis=2)
    assert np.allclose(voids.radius_ang, (dist - r_image).min(axis=1), rtol=0,
                       atol=1e-10)
    assert (dist.min(axis=1) >= voids.circumradius_ang - 1e-9).all()
    assert voids.volume_sum_ang3 == pytest.approx(frame.volume_ang3, rel=1e-12)
    assert ((voids.centre_frac >= 0) & (voids.centre_frac < 1)).all()
    if (voids.radius_ang < 0).any():
        assert any("inside an atom sphere" in n for n in voids.notes)


def test_empty_spheres_refuse_coincident_atoms_and_note_close_ones():
    """Qhull leaves a point that lies on another one out of a Delaunay
    triangulation, while every point of a set of distinct points is a
    vertex of it. Before, a duplicated atom was then simply a vertex of no
    tetrahedron (200 tetrahedra here, against 209 for the same atoms 1e-6 Å
    apart) and nothing said so, where voronoi_cells refuses the same frame.
    It is refused, naming the rows; two atoms 1e-6 Å apart are both
    vertices, and a note gives the contact (the Delaunay edge joining each
    atom to its nearest neighbour), below bulk.D_MIN_ANG."""
    rng = np.random.default_rng(4)
    frac = rng.uniform(0, 1, (30, 3))
    frac[29] = frac[0]
    same = frame_from_arrays(["Si"] * 15 + ["O"] * 15, frac=frac,
                             box_ang=np.eye(3) * 12.0)
    radii = {"Si": 0.5, "O": 0.5}
    # which of the two copies Qhull leaves out is its choice
    with pytest.raises(ValueError, match="vertex of no Delaunay tetrahedron.*"
                       "row (29 at 0 Å from row 0|0 at 0 Å from row 29)"):
        md_order.empty_spheres(same, radii, radii_source="test")
    frac[29] = frac[0] + 1e-6 / 12.0
    close = frame_from_arrays(["Si"] * 15 + ["O"] * 15, frac=frac,
                              box_ang=np.eye(3) * 12.0)
    voids = md_order.empty_spheres(close, radii, radii_source="test")
    assert set(np.unique(voids.vertices).tolist()) == set(range(30))
    want = math.sqrt(3) * 1e-6
    note = [n for n in voids.notes if "closer than 0.4 Å" in n]
    assert note and "rows 0, 29" in note[0] and f"{want:.3g} Å apart" in note[0]
    # a frame with no close contact gets no such note
    plain = md_order.empty_spheres(random_frame(n=27, seed=31,
                                                elements=("Si", "O")),
                                   radii, radii_source="test")
    assert not any("closer than" in n for n in plain.notes)


def test_empty_spheres_refuse_missing_radii():
    frame = random_frame(n=27, seed=31, elements=("Si", "O"))
    with pytest.raises(ValueError):
        md_order.empty_spheres(frame, {"Si": 0.4}, radii_source="test")
    with pytest.raises(ValueError):
        md_order.empty_spheres(frame, {"Si": 0.4, "O": 1.0}, radii_source="")
    with pytest.raises(ValueError):
        md_order.empty_spheres(frame, {"Si": 0.4, "O": -1.0},
                               radii_source="test")


# ---------------------------------------------------------------------------
# free volume
# ---------------------------------------------------------------------------

def _circle_top(a, ra, b, rb):
    """The highest point (largest z) of the circle where spheres (a, ra) and
    (b, rb) meet, or None. A horizontal circle has no single highest point;
    any of its points is returned (the three-sphere points settle that
    case, as below)."""
    d = b - a
    dist = math.sqrt(float(d @ d))
    if dist == 0.0 or dist > ra + rb or dist < abs(ra - rb):
        return None
    u = d / dist
    t = (dist * dist + ra * ra - rb * rb) / (2.0 * dist)
    rc = math.sqrt(max(ra * ra - t * t, 0.0))
    w = np.array([0.0, 0.0, 1.0]) - u[2] * u
    if math.sqrt(float(w @ w)) < 1e-12:
        w = np.cross(u, [1.0, 0.0, 0.0])
        if math.sqrt(float(w @ w)) < 1e-12:
            w = np.cross(u, [0.0, 1.0, 0.0])
    return a + t * u + rc * w / math.sqrt(float(w @ w))


def _three_points(a, ra, b, rb, c, rc):
    """The (up to two) points where three spheres meet."""
    ex = b - a
    d = math.sqrt(float(ex @ ex))
    if d == 0.0:
        return []
    ex = ex / d
    i = float(ex @ (c - a))
    ey = c - a - i * ex
    ney = math.sqrt(float(ey @ ey))
    if ney < 1e-12:
        return []
    ey = ey / ney
    j = float(ey @ (c - a))
    x = (ra * ra - rb * rb + d * d) / (2 * d)
    y = (ra * ra - rc * rc + i * i + j * j) / (2 * j) - i * x / j
    z2 = ra * ra - x * x - y * y
    if z2 < 0:
        return []
    z = math.sqrt(z2)
    base = a + x * ex + y * ey
    ez = np.cross(ex, ey)
    return [base + z * ez, base - z * ez]


def _covered_by_a_probe(x, centres, big_r, probe, tol=1e-9):
    """Whether some probe centre within ``probe`` of x lies outside every
    sphere (centres, big_r): the set S = B(x, probe) minus the open spheres
    is empty or not. Decided by S's highest point, which, when S is not
    empty, is the top of the probe sphere, the top of a circle where the
    probe sphere meets one sphere or where two spheres meet, or a point
    where three of the spheres (the probe sphere among them or not) meet:
    a highest point with one sphere alone active would be the bottom of
    that sphere, from which S rises. A characterisation by highest points,
    where the module's is by nearest points, and written with loops."""
    spheres = [(x, probe)] + list(zip(centres, big_r))

    def in_s(c):
        if float((c - x) @ (c - x)) > (probe + tol) ** 2:
            return False
        diff = centres - c
        return bool((np.einsum("ij,ij->i", diff, diff)
                     >= (big_r - tol) ** 2).all())

    candidates = [x + np.array([0.0, 0.0, probe])]
    for (a, ra), (b, rb) in itertools.combinations(spheres, 2):
        top = _circle_top(a, ra, b, rb)
        if top is not None:
            candidates.append(top)
    for (a, ra), (b, rb), (c, rc) in itertools.combinations(spheres, 3):
        candidates += _three_points(a, ra, b, rb, c, rc)
    return any(in_s(c) for c in candidates)


def _brute_force_free(frame, radii, probe, h):
    """The three fractions on free_volume's grid, every point by brute force
    over all atoms and 125 images; the probe-occupiable test is the
    highest-point one above, independent of the module's."""
    shape = np.maximum(1, np.ceil(np.linalg.norm(frame.box_ang, axis=1) / h)
                       ).astype(int)
    idx = np.stack(np.unravel_index(np.arange(int(np.prod(shape))),
                                    tuple(shape)), axis=1)
    points = ((idx + 0.5) / shape) @ frame.box_ang
    shifts = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    images = ((frame.frac[None] + shifts[:, None]).reshape(-1, 3)
              @ frame.box_ang)
    r_image = np.tile([radii[s] for s in frame.elements], len(shifts))
    core = np.zeros(len(points), bool)
    blocked = np.zeros(len(points), bool)
    for chunk in np.array_split(np.arange(len(points)), 20):
        d = np.linalg.norm(points[chunk, None, :] - images[None], axis=2)
        core[chunk] = (d < r_image).any(axis=1)
        blocked[chunk] = (d < r_image + probe).any(axis=1)
    ok = ~blocked
    occupiable = ok.copy()
    # 125 images hold every atom within r + 2 r_p of a point of the box
    assert max(radii.values()) + 2 * probe < \
        2 * frame.perpendicular_widths_ang.min()
    for p in np.flatnonzero(blocked & ~core):
        d = np.linalg.norm(images - points[p], axis=1)
        near = d < r_image + 2 * probe
        occupiable[p] = _covered_by_a_probe(points[p], images[near],
                                            r_image[near] + probe, probe)
    n = len(points)
    return ((~core).sum() / n, ok.sum() / n, occupiable.sum() / n)


@functools.lru_cache(maxsize=None)
def _brute_force_case(sheared: bool, probe: float):
    frame = random_frame(n=8, seed=41, sheared=sheared, elements=("Si", "O"),
                         spacing=3.2)
    radii = {"Si": 0.9, "O": 1.3}
    return frame, radii, _brute_force_free(frame, radii, probe, 0.4)


@pytest.mark.parametrize("sheared,probe,route", [
    (True, 0.45, "auto"), (False, 0.45, "auto"),
    (False, 0.9, "pair graph"), (False, 0.9, "regular triangulation"),
    (True, 0.9, "regular triangulation")])
def test_free_volume_equals_brute_force_on_the_same_grid(sheared, probe, route,
                                                         monkeypatch):
    """Every grid point classified as the brute force above classifies it:
    the counts are equal, for a sheared and an orthogonal box, and with a
    probe large enough (0.9 Å) that most free points need two- and
    three-sphere points, which come from every triple of meeting spheres or
    from the regular triangulation's triangles (both routes forced here).
    The small chunk and work sizes make the module cut the shell points into
    several chunks and its tests into several blocks, so those seams are
    crossed.
    Before the exact test (a grid dilation of the probe-centre grid points)
    the probe-occupiable count was below this one."""
    monkeypatch.setattr(md_order, "_SHELL_PAIRS_PER_SLAB", 2_000)
    monkeypatch.setattr(md_order, "_EXACT_WORK_PER_CHUNK", 5_000)
    monkeypatch.setattr(md_order, "_TRIPLES_FROM", route)
    frame, radii, want = _brute_force_case(sheared, probe)
    got = md_order.free_volume(frame, radii, radii_source="test",
                               probe_radius_ang=probe, grid_spacing_ang=0.4)
    n = got.n_points
    assert (round(got.geometric_fraction * n), round(got.probe_centre_fraction
            * n), round(got.probe_occupiable_fraction * n)) == \
        tuple(round(v * n) for v in want)
    assert got.geometric_fraction >= got.probe_occupiable_fraction \
        >= got.probe_centre_fraction
    assert got.probe_occupiable_fraction > got.probe_centre_fraction
    if route != "auto":
        assert any(("regular triangulation" in n) == (route != "pair graph")
                   for n in got.notes if "candidate triples" in n)


def test_free_volume_of_separated_spheres_approaches_the_closed_forms():
    """Simple cubic, a = 4 Å, r = 1.0 Å, probe 0.4 Å: the spheres and the
    probe-excluded spheres do not overlap (1.4 < a/2), so the geometric and
    probe-occupiable fractions are 1 - (4/3) pi r^3 / a^3 and the
    probe-centre one 1 - (4/3) pi (r + r_p)^3 / a^3. The spacings (0.207
    and 0.109 Å once fitted to the 12 Å box) are not divisors of a, so the
    27 spheres sit at different offsets from the grid. Over 12 random
    lattice shifts the grid fractions scattered about the closed forms by
    1.5e-4 and 0.9e-4 rms at the coarse spacing, 4.0e-4 at most (measured
    with a scratch script); a sphere missed or counted twice would move the
    fraction by 2.4e-3, so the tolerance is 1e-3. Every point outside the
    atom spheres is within r_p of its own atom's expanded sphere, whose
    point there touches no other expanded sphere (2.8 Å < a), so the
    probe-occupiable points are exactly the free points, on any grid: the
    two fractions are equal, not near. The grid dilation used before put
    the probe-occupiable fraction 0.017 below at the coarse spacing and
    0.0041 below at the fine one."""
    a, r, probe = 4.0, 1.0, 0.4
    frame = cubic_frame("sc", 3, a, shift=0.137)
    law_free = 1 - 4 / 3 * math.pi * r ** 3 / a ** 3
    law_centre = 1 - 4 / 3 * math.pi * (r + probe) ** 3 / a ** 3
    coarse = md_order.free_volume(frame, {"Si": r}, radii_source="test",
                                  probe_radius_ang=probe, grid_spacing_ang=0.21)
    fine = md_order.free_volume(frame, {"Si": r}, radii_source="test",
                                probe_radius_ang=probe, grid_spacing_ang=0.11)
    for result in (coarse, fine):
        assert result.geometric_fraction == pytest.approx(law_free, abs=1e-3)
        assert result.probe_centre_fraction == pytest.approx(law_centre,
                                                             abs=1e-3)
        assert result.probe_occupiable_fraction == result.geometric_fraction
    zero = md_order.free_volume(frame, {"Si": r}, radii_source="test",
                                probe_radius_ang=0.0, grid_spacing_ang=0.21)
    assert zero.geometric_fraction == zero.probe_centre_fraction == \
        zero.probe_occupiable_fraction == coarse.geometric_fraction


def test_free_volume_refusals_and_vdw_radii():
    frame = random_frame(n=8, seed=41, elements=("Si", "O"))
    radii = md_order.vdw_radii_ang(frame)
    assert radii == {"O": 1.52, "Si": 2.1}      # gemmi's table, 2 decimals
    for kwargs in ({"probe_radius_ang": -0.1, "grid_spacing_ang": 0.5},
                   {"probe_radius_ang": 0.1, "grid_spacing_ang": 0.0},
                   {"probe_radius_ang": 0.1, "grid_spacing_ang": 1e-4}):
        with pytest.raises(ValueError):
            md_order.free_volume(frame, radii,
                                 radii_source=md_order.VDW_RADII_SOURCE,
                                 **kwargs)
    with pytest.raises(ValueError):
        md_order.vdw_radii_ang(["Xx"])


# ---------------------------------------------------------------------------
# neighbour lists
# ---------------------------------------------------------------------------

def _quartz_frame():
    frame, parent, sites, table = _crystal_case(EXAMPLES[0])
    return frame, table


def test_neighbours_from_bonds_count_the_cn():
    frame, table = _quartz_frame()
    nb = md_order.neighbours_from_bonds(frame, bulk.bonds_at(table))
    cn = bulk.at_threshold(table).cn
    assert np.array_equal(nb.count, cn)
    entries = {(int(c), int(n), tuple(i)) for c, n, i in
               zip(nb.centre, nb.nbr, nb.image)}
    assert entries == {(n, c, tuple(-np.array(i))) for c, n, i in entries}
    assert nb.v_bond_vu == bv.V_BOND_DEFAULT
    other = frame_from_arrays(frame.elements, frac=(frame.frac + 0.01) % 1,
                              box_ang=frame.box_ang)
    with pytest.raises(ValueError):
        md_order.neighbours_from_bonds(other, bulk.bonds_at(table))


def test_neighbours_from_pairs_count_what_distance_cn_counts():
    frame = random_frame(n=64, seed=51)
    pairs = bulk.find_pairs(frame, 4.0)
    cutoffs = {("Si", "O"): 2.6, ("O", "Si"): 2.6, ("Na", "O"): 3.1}
    nb = md_order.neighbours_from_pairs(
        frame, pairs, cutoffs, {k: SOURCE for k in cutoffs})
    counts = bulk.distance_cn(pairs, frame, cutoffs)
    want = np.zeros(frame.n_atoms, int)
    for key, value in counts.items():
        want += np.where(value >= 0, value, 0)
    assert np.array_equal(nb.count, want)
    assert nb.cutoff_sources == {k: SOURCE for k in cutoffs}
    assert "Si-O 2.6 Å" in nb.definition
    with pytest.raises(ValueError):          # a cutoff without its source
        md_order.neighbours_from_pairs(frame, pairs, {("Si", "O"): 2.6}, {})
    with pytest.raises(ValueError):          # beyond the search radius
        md_order.neighbours_from_pairs(frame, pairs, {("Si", "O"): 4.5},
                                       {("Si", "O"): SOURCE})
    with pytest.raises(ValueError):          # no cutoff at all
        md_order.neighbours_from_pairs(frame, pairs, {}, {})
    no_vectors = bulk.find_pairs(frame, 4.0, vectors_within_ang=0)
    with pytest.raises(ValueError):
        md_order.neighbours_from_pairs(frame, no_vectors, {("Si", "O"): 2.6},
                                       {("Si", "O"): SOURCE})


# ---------------------------------------------------------------------------
# frame averages
# ---------------------------------------------------------------------------

def test_frame_averages_through_md_stats():
    frames = [random_frame(n=27, seed=s, elements=("Si", "O")) for s in (61, 62)]
    cells = [md_order.voronoi_cells(f) for f in frames]
    dist = md_order.voronoi_index_distribution(cells, element="Si",
                                               frames=[0, 1])
    assert np.allclose(np.nansum(dist.per_frame, axis=1), 1.0)
    cn = md_order.voronoi_cn_distribution(cells, frames=[0, 1], kind="count")
    assert np.array_equal(cn.per_frame.sum(axis=1), [27, 27])
    orders = [md_order.steinhardt(cutoff_neighbours(f, 3.3, pair=("Si", "Si")))
              for f in frames]
    hist = md_order.element_histograms(
        [(o.elements, o.values("q", 6)) for o in orders],
        np.linspace(0, 1, 11), name="q6", unit="1", frames=[0, 1])
    assert set(hist) == {"O", "Si"}
    # every O has no Si-Si neighbour: all non-finite, counted not dropped
    assert hist["O"].n_nonfinite.sum() == sum((f.elements == "O").sum()
                                              for f in frames)
    voids = [md_order.empty_spheres(f, {"Si": 0.0, "O": 0.0},
                                    radii_source="test") for f in frames]
    h = md_order.empty_sphere_histogram(voids, np.linspace(0, 4, 9))
    assert (h.counts.sum(1) + h.n_above + h.n_below + h.n_nonfinite
            == [len(v) for v in voids]).all()
    free = [md_order.free_volume(f, {"Si": 0.8, "O": 1.0}, radii_source="t",
                                 probe_radius_ang=0.3, grid_spacing_ang=0.5)
            for f in frames]
    scalars = md_order.free_volume_scalars(free, frames=[0, 1])
    assert scalars["geometric_fraction"].mean == pytest.approx(
        np.mean([f.geometric_fraction for f in free]), abs=1e-15)
    identical = md_order.free_volume_scalars([free[0], free[0]])
    assert identical["probe_centre_fraction"].std == 0.0


def test_frame_averages_refuse_frames_measured_differently():
    """An average over frames needs one method. Before, only the probe
    radius was compared, and the notes gave frame 0's settings: radii, their
    source, the grid spacing asked for and the Voronoi thresholds could
    differ between the frames averaged."""
    frames = [random_frame(n=27, seed=s, elements=("Si", "O")) for s in (61, 62)]

    def free(f, radii, source, h):
        return md_order.free_volume(f, radii, radii_source=source,
                                    probe_radius_ang=0.3, grid_spacing_ang=h)

    base = free(frames[0], {"Si": 0.8, "O": 1.0}, "set A", 0.5)
    for other in (free(frames[1], {"Si": 0.8, "O": 1.1}, "set A", 0.5),
                  free(frames[1], {"Si": 0.8, "O": 1.0}, "set B", 0.5),
                  free(frames[1], {"Si": 0.8, "O": 1.0}, "set A", 0.6)):
        with pytest.raises(ValueError, match="different"):
            md_order.free_volume_scalars([base, other])
    same = md_order.free_volume_scalars(
        [base, free(frames[1], {"Si": 0.8, "O": 1.0}, "set A", 0.5)])
    assert "grid spacing 0.5 Å asked for" in same["geometric_fraction"].notes[0]

    cells = [md_order.voronoi_cells(frames[0]),
             md_order.voronoi_cells(frames[1], min_face_area_ang2=0.5)]
    with pytest.raises(ValueError, match="min_face_area_ang2"):
        md_order.voronoi_index_distribution(cells)
    with pytest.raises(ValueError, match="min_face_area_ang2"):
        md_order.voronoi_cn_distribution(cells)
    cells[1] = md_order.voronoi_cells(frames[1], min_edge_ang=0.2)
    with pytest.raises(ValueError, match="min_edge_ang"):
        md_order.voronoi_index_distribution(cells)

    voids = [md_order.empty_spheres(frames[0], {"Si": 0.0, "O": 0.0},
                                    radii_source="zero"),
             md_order.empty_spheres(frames[1], {"Si": 1.0, "O": 1.4},
                                    radii_source="zero")]
    with pytest.raises(ValueError, match="radii"):
        md_order.empty_sphere_histogram(voids, np.linspace(0, 4, 9))


# ---------------------------------------------------------------------------
# no verdicts, and no Qt
# ---------------------------------------------------------------------------

VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def test_no_note_carries_a_verdict():
    notes = []
    frame, nb = ico_cluster()
    notes += md_order.steinhardt(nb, (4, 6)).notes
    notes += md_order.tetrahedral_order(nb).notes
    notes += md_order.polyhedron_shape(synthetic_neighbours(
        _random_polyhedra())).notes
    rf = random_frame(n=27, seed=71, elements=("Si", "O"))
    notes += md_order.voronoi_cells(rf, margin_ang=0.5, min_face_area_ang2=0.3,
                                    min_edge_ang=0.5).notes
    notes += md_order.empty_spheres(rf, {"Si": 2.0, "O": 1.5},
                                    radii_source="test").notes
    notes += md_order.free_volume(rf, {"Si": 0.5, "O": 0.5}, radii_source="t",
                                  probe_radius_ang=0.2,
                                  grid_spacing_ang=0.5).notes
    assert len(notes) >= 12
    assert [n for n in notes if VERDICT.search(n)] == []


def test_no_string_in_the_module_carries_a_verdict():
    source = (ROOT / "facet" / "core" / "md_order.py").read_text(
        encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 50
    assert [s for s in strings if VERDICT.search(s)] == []
    comments = [line for line in source.splitlines() if "#" in line]
    assert [c for c in comments if VERDICT.search(c.split("#", 1)[1])] == []


def test_public_arguments_carry_their_unit():
    """Units in names: no public function takes a bare length, radius,
    cutoff, margin, spacing or bin edges. empty_sphere_histogram's bin edges
    were a bare 'edges' while they are Å; they are 'edges_ang'.
    element_histograms keeps 'edges' because its unit is its own 'unit'
    argument (the quantity is any per-atom value)."""
    import inspect

    bare = {"r", "d", "x", "h", "freq", "radius", "radii", "cutoff",
            "cutoffs", "margin", "spacing", "probe", "edges"}
    found = []
    for name in md_order.__all__:
        obj = getattr(md_order, name)
        if not inspect.isfunction(obj):
            continue
        params = inspect.signature(obj).parameters
        for p in params:
            if p in bare and not (p == "edges" and "unit" in params):
                found.append(f"{name}({p})")
    assert found == []
    assert "edges_ang" in inspect.signature(
        md_order.empty_sphere_histogram).parameters


def test_the_module_imports_without_qt():
    code = ("import sys, facet.core.md_order;"
            "print('QT' if any(m.startswith(('PySide6', 'matplotlib')) "
            "for m in sys.modules) else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
