"""The pair distribution function, from the structure.

What this computes and what it does not, stated first because the distinction is
the whole reason a PDF gets measured.

**Exact from the structure.** Every interatomic distance and its degeneracy, out
to any radius; the number density rho0 and the cell contents N; the
4*pi*r*rho0 baseline; the conversions between R(r), G(r) and g(r); the peak areas
as scattering-weighted coordination numbers; the widths where the file actually
gives displacement parameters.

**Tabulated, so exact only as the tables are.** The scattering weights, from the
same four-Gaussian X-ray factors and neutron lengths ``diffraction`` uses -- so
the two modules cannot disagree about what an element scatters.

**Assumed, each an argument with a stated default.** The radiation. The
displacement parameter used where the file gives none. The correlation
parameters delta1 and delta2, which default to zero because FACET cannot measure
them from one structure. Qmin, Qmax, the window, Qdamp and Qbroad, which are
properties of an instrument and not of a crystal. And the r grid.

**A modelling limit, and not a small one.** This is the PDF of the *average*
crystal. A fractionally occupied site contributes fractional pair correlations,
and a real disordered material's measured PDF differs from this by design --
that difference is usually why the measurement was made. FACET prints that
sentence in the notes of every pattern it computes.

No refinement, no data corrections. This is not PDFgetX3 and it is not PDFgui.

The definitions, once, so the code below can be read against them::

    R(r) = (1/N) sum_i sum_{j != i} w_ij  Normal(r; r_ij, sigma_ij)
    G(r) = R(r)/r - 4 pi r rho0
    g(r) = 1 + G(r)/(4 pi r rho0) = R(r) / (4 pi r^2 rho0)

with w_ij = (occ_i b_i)(occ_j b_j) / <b>^2 and <b> the occupancy-weighted mean
scattering power. The integral of R(r) over one peak is the scattering-weighted
coordination number of that shell, which is what a PDF is usually read for.

Two details that are easy to get wrong and are therefore stated here.

*The 1/r is taken at the field point, not at r_ij.* The delta-function forms are
identical; the broadened ones are not, and only R(r)/r keeps a peak's area equal
to the coordination number. Using 1/r_ij instead shifts the apparent maximum of a
2 A peak by 0.005 A at sigma = 0.10 A and by 0.020 A at sigma = 0.20 A.

*Truncation and Q-resolution are duals.* Truncating the transform at a finite Q
is a multiplication in Q, so it is a **convolution** in r, and it must be done
over the odd extension G(-r) = -G(r) -- which is what generates the termination
ripple below the first peak, exactly the region someone stares at while asking
whether there is a shorter bond. Finite Q-resolution is a convolution in Q, so it
is a **multiplication** in r.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import diffraction
from .structure import Structure

# The displacement parameter used where a file gives none. Derived from the
# diffraction module's constant, not chosen separately, so a structure's peak
# widths here and its Debye-Waller factor there cannot come from two different
# numbers. B = 8 pi^2 U.
DEFAULT_U_ISO = diffraction.DEFAULT_B_ISO / (8.0 * math.pi * math.pi)

WINDOWS = ("boxcar", "Lorch", "none")

# A lower bound on the peak width, in angstrom. Without it a structure with
# U = 0 in the file and delta1 set would ask for a delta function, and with
# delta2 large enough the correlation term can drive sigma^2 negative at small r.
MIN_SIGMA = 0.01


@dataclass
class Weights:
    """Scattering weights, and the pair table they imply.

    ``pairs`` is the fraction of the total weight each element pair carries. It
    is the most useful line on the screen: for Bi2O3 the X-ray PDF is 76% Bi-Bi
    and 2% O-O, while the neutron PDF is 25 / 50 / 26 -- the quantitative form of
    the statement that the trustworthy Bi-O structures are neutron refinements.
    """

    radiation: str
    b: dict[str, float] = field(default_factory=dict)
    b_mean: float = 0.0
    pairs: dict[tuple[str, str], float] = field(default_factory=dict)

    def describe(self) -> list[str]:
        out = []
        for (a, c), fraction in sorted(self.pairs.items(),
                                       key=lambda kv: -kv[1]):
            out.append(f"{a}-{c} {100.0 * fraction:.1f}%")
        return out


@dataclass
class PDF:
    """A computed pair distribution function and everything behind it."""

    r: np.ndarray
    G: np.ndarray                      # the reduced PDF, what is plotted
    g: np.ndarray                      # the pair distribution function, -> 1
    R: np.ndarray                      # the radial distribution, area = CN

    rho0: float = 0.0                  # atoms per angstrom^3, occupancies in
    n_cell: float = 0.0                # cell contents, occupancies in
    radiation: str = "X-ray"
    weights: Weights | None = None

    u_iso: float = DEFAULT_U_ISO       # the fallback that was available
    n_from_file: int = 0               # sites whose U came from the file
    n_assumed: int = 0                 # sites that used the fallback

    q_min: float = 0.0
    q_max: float = 0.0                 # 0 means no truncation was applied
    q_damp: float = 0.0
    q_broad: float = 0.0
    delta1: float = 0.0
    delta2: float = 0.0
    window: str = "none"

    structure_name: str = ""
    n_pairs: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def dr(self) -> float:
        return float(self.r[1] - self.r[0]) if len(self.r) > 1 else 0.0

    def coordination_in(self, r_lo: float, r_hi: float) -> float:
        """Scattering-weighted coordination number under one peak of R(r).

        The integral of R(r) between two minima. Weighted, so for a compound it
        is not an atom count: it is the count each pair would contribute given
        how strongly its elements scatter. The weight table says by how much.
        """
        if not len(self.r):
            return float("nan")
        inside = (self.r >= float(r_lo)) & (self.r <= float(r_hi))
        if not inside.any():
            return 0.0
        return float(np.trapezoid(self.R[inside], self.r[inside]))


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def scattering_weights(structure: Structure,
                       radiation: str = "X-ray") -> Weights:
    """Per-element scattering power at Q = 0, and the element-pair weights.

    The X-ray weight is taken at Q = 0, which is the Warren convention and is
    what PDFgetX3 does with its <f> normalisation. It is exact only as Q -> 0;
    it is serviceable across the whole range because the *ratio* f_i/f_j varies
    slowly, and it is an approximation either way, so it is said here rather
    than buried.
    """
    counts: dict[str, float] = {}
    for atom in structure.atoms:
        counts[atom.element] = counts.get(atom.element, 0.0) + atom.occupancy
    total = sum(counts.values())
    b = {sym: float(diffraction.form_factor(sym, 0.0, radiation))
         for sym in counts}
    if total <= 0:
        return Weights(radiation=radiation, b=b, b_mean=0.0)

    b_mean = sum(counts[s] * b[s] for s in counts) / total
    weights = Weights(radiation=radiation, b=b, b_mean=float(b_mean))
    if abs(b_mean) < 1e-30:
        return weights

    # the fraction of the total pair weight each element pair carries; sums to 1
    pairs: dict[tuple[str, str], float] = {}
    for i, si in enumerate(sorted(counts)):
        for sj in sorted(counts)[i:]:
            share = (counts[si] * b[si]) * (counts[sj] * b[sj]) / (
                total * total * b_mean * b_mean)
            if si != sj:
                share *= 2.0                # both orderings of the pair
            pairs[(si, sj)] = float(share)
    weights.pairs = pairs
    return weights


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------

def pair_distances(structure: Structure, r_max: float, *,
                   r_min: float = 1e-6, chunk: int = 4096):
    """Every interatomic distance out to ``r_max``, with its weight inputs.

    Yields ``(distances, index_i, index_j)`` in chunks over the home atoms, so
    the pair arrays never all exist at once. Public because it is most of what a
    single-scattering EXAFS calculation needs as well, and burying it inside the
    PDF would mean writing the periodic image search twice.

    Ordered pairs: each i-j is produced once for i and once for j, which is what
    the double sum over i != j asks for.
    """
    from .neighbors import _images_within

    orth = structure.cell.orth
    fracs = np.array([a.frac for a in structure.atoms], float)
    if not len(fracs):
        return
    images = _images_within(orth, float(r_max))
    shifts = images @ orth.T                      # (n_images, 3) cartesian
    home = fracs @ orth.T

    for start in range(0, len(home), chunk):
        stop = min(start + chunk, len(home))
        block = home[start:stop]
        # (block, n_atoms, n_images, 3) would be too large; loop the images,
        # which is the smallest of the three axes for any sensible r_max
        d_list, i_list, j_list = [], [], []
        for shift in shifts:
            delta = (home + shift)[None, :, :] - block[:, None, :]
            dist = np.sqrt((delta * delta).sum(axis=2))
            rows, cols = np.nonzero((dist > r_min) & (dist <= r_max))
            if not len(rows):
                continue
            d_list.append(dist[rows, cols])
            i_list.append(rows + start)
            j_list.append(cols)
        if d_list:
            yield (np.concatenate(d_list), np.concatenate(i_list),
                   np.concatenate(j_list))


def _atom_weights(structure: Structure, weights: Weights) -> np.ndarray:
    """``occ_i * b_i`` per atom."""
    return np.array([a.occupancy * weights.b.get(a.element, 0.0)
                     for a in structure.atoms], float)


def _atom_u(structure: Structure, u_iso: float) -> tuple[np.ndarray, int, int]:
    """Isotropic U per atom, and how many came from the file."""
    out, from_file, assumed = [], 0, 0
    for atom in structure.atoms:
        site = structure.sites[atom.site_index]
        if site.u_iso:
            out.append(float(site.u_iso))
            from_file += 1
        else:
            out.append(float(u_iso))
            assumed += 1
    return np.asarray(out, float), from_file, assumed


def sigma_pair(u_i, u_j, distance, *, delta1: float = 0.0,
               delta2: float = 0.0, q_broad: float = 0.0) -> np.ndarray:
    """Peak width for a pair, in angstrom.

    The uncorrelated part is exact for isotropic displacement parameters::

        sigma^2 = <[(u_i - u_j) . r_hat]^2> = U_i + U_j

    and it is an **upper bound** on what a measurement shows: near neighbours
    move together, and the cross term -2<(u_i.r_hat)(u_j.r_hat)> that this
    omits is precisely why the first PDF peak is sharper than U_i + U_j
    predicts. The correlation is applied in the PDFgui parameterisation::

        sigma' = sigma sqrt(1 - delta1/r - delta2/r^2 + Qbroad^2 r^2)

    with delta1 and delta2 defaulting to zero, because they are fitted against
    data and cannot be measured from a structure. The result is clamped, so a
    large delta2 at small r cannot ask for an imaginary width.
    """
    r = np.maximum(np.asarray(distance, float), 1e-6)
    base = np.sqrt(np.maximum(np.asarray(u_i, float) + np.asarray(u_j, float),
                              0.0))
    factor = (1.0 - float(delta1) / r - float(delta2) / (r * r)
              + float(q_broad) ** 2 * r * r)
    return np.maximum(base * np.sqrt(np.maximum(factor, 0.0)), MIN_SIGMA)


# ---------------------------------------------------------------------------
# finite Q
# ---------------------------------------------------------------------------

def termination_kernel(r, q_min: float, q_max: float) -> np.ndarray:
    """The r-space kernel of a boxcar window on [q_min, q_max].

    ``K(r) = [sin(Qmax r) - sin(Qmin r)] / (pi r)``, with
    ``K(0) = (Qmax - Qmin)/pi``. Truncating a transform is a multiplication in
    Q, so it is this convolution in r -- and the ripple it produces below the
    first peak is a property of the measurement, not of the crystal.

    Its integral is **1 when Qmin = 0, and 0 when Qmin > 0**, because the
    integral of ``sin(a r)/(pi r)`` is 1 for every ``a > 0`` and the two terms
    then cancel. That is not a defect: excluding the small-Q region of a
    measurement removes the mean of G(r) along with it, and a kernel of zero
    integral is what doing so looks like in r.
    """
    r = np.asarray(r, float)
    out = np.empty_like(r)
    small = np.abs(r) < 1e-12
    out[small] = (float(q_max) - float(q_min)) / math.pi
    rr = r[~small]
    out[~small] = (np.sin(float(q_max) * rr)
                   - np.sin(float(q_min) * rr)) / (math.pi * rr)
    return out


def apply_termination(r, G, q_min: float, q_max: float) -> np.ndarray:
    """Convolve ``G`` with the boxcar kernel, over the odd extension.

    The odd extension G(-r) = -G(r) is not optional: without it the convolution
    has nothing to reflect at the origin and the termination ripple below the
    first peak -- which is what a measured PDF shows there -- does not appear.
    """
    r = np.asarray(r, float)
    G = np.asarray(G, float)
    if q_max <= 0 or len(r) < 2:
        return G.copy()
    dr = float(r[1] - r[0])
    n = len(r)

    # The odd extension, on a grid that is actually uniform. r = 0 has to be in
    # it: the r grid starts at dr, so gluing -r[::-1] straight onto r leaves a
    # gap of 2*dr across the origin, and a convolution over a non-uniform
    # sampling is not the operation it looks like. G(0) = 0 because G is odd.
    odd = np.concatenate([-G[::-1], [0.0], G])          # length 2n + 1

    # The kernel on the lag grid, the same length and odd, so that
    # np.convolve(..., "same") lines its centre up with lag = 0.
    lag = np.arange(-n, n + 1) * dr
    kernel = termination_kernel(lag, q_min, q_max) * dr
    convolved = np.convolve(odd, kernel, mode="same")
    # index n is r = 0, so the positive half starts one past it
    return convolved[n + 1:]


def apply_q_damp(r, G, q_damp: float) -> np.ndarray:
    """``G *= exp(-(r Qdamp)^2 / 2)``.

    Finite Q-resolution is a convolution in Q, and its dual is this
    multiplication in r -- the other way round from the truncation above, which
    is the pair of operations most easily swapped by mistake.
    """
    if q_damp <= 0:
        return np.asarray(G, float).copy()
    r = np.asarray(r, float)
    return np.asarray(G, float) * np.exp(-0.5 * (r * float(q_damp)) ** 2)


def _lorch_round_trip(r, G, q_min: float, q_max: float,
                      n_q: int = 4096) -> np.ndarray:
    """Apply a Lorch window by going through Q and back.

    The Lorch window has no closed-form r-space kernel, so it is done the
    numerical way: forward sine transform to F(Q), multiply by
    ``sin(pi Q/Qmax)/(pi Q/Qmax)``, transform back.
    """
    r = np.asarray(r, float)
    G = np.asarray(G, float)
    if q_max <= 0 or len(r) < 2:
        return G.copy()
    q = np.linspace(max(q_min, 1e-6), q_max, n_q)
    dr = float(r[1] - r[0])
    # F(Q) = integral G(r) sin(Q r) dr
    F = (np.sin(np.outer(q, r)) * G[None, :]).sum(axis=1) * dr
    x = math.pi * q / q_max
    F = F * np.sin(x) / x
    dq = float(q[1] - q[0])
    return (2.0 / math.pi) * (np.sin(np.outer(r, q)) * F[None, :]).sum(
        axis=1) * dq


def forward_transform(r, G, q):
    """``F(Q) = integral G(r) sin(Q r) dr`` -- the independent route back.

    Not used by ``pair_distribution``; it exists so the tests can check the
    closed-form truncation kernel against an explicit transform and back again,
    which pins the 2/pi, the 4*pi*r*rho0 baseline and the kernel normalisation
    all at once.
    """
    r = np.asarray(r, float)
    G = np.asarray(G, float)
    q = np.asarray(q, float)
    if len(r) < 2:
        return np.zeros_like(q)
    dr = float(r[1] - r[0])
    return (np.sin(np.outer(q, r)) * G[None, :]).sum(axis=1) * dr


def inverse_transform(q, F, r):
    """``G(r) = (2/pi) integral F(Q) sin(Q r) dQ``."""
    q = np.asarray(q, float)
    F = np.asarray(F, float)
    r = np.asarray(r, float)
    if len(q) < 2:
        return np.zeros_like(r)
    dq = float(q[1] - q[0])
    return (2.0 / math.pi) * (np.sin(np.outer(r, q)) * F[None, :]).sum(
        axis=1) * dq


# ---------------------------------------------------------------------------
# the pattern
# ---------------------------------------------------------------------------

def pair_distribution(structure: Structure, *,
                      r_max: float = 20.0,
                      dr: float = 0.01,
                      radiation: str = "X-ray",
                      u_iso: float = DEFAULT_U_ISO,
                      delta1: float = 0.0,
                      delta2: float = 0.0,
                      q_min: float = 0.0,
                      q_max: float = 0.0,
                      q_damp: float = 0.0,
                      q_broad: float = 0.0,
                      window: str = "boxcar") -> PDF:
    """G(r) for a structure, with every assumption recorded in the result.

    ``q_max = 0`` means no truncation: the ideal, infinite-Q PDF. Set it to the
    Qmax of a measurement to compare with one.
    """
    if r_max <= 0:
        raise ValueError("r_max must be positive")
    if dr <= 0:
        raise ValueError("dr must be positive")
    if radiation not in diffraction.RADIATIONS:
        raise ValueError(f"unknown radiation {radiation!r}; "
                         f"expected one of {diffraction.RADIATIONS}")
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; expected one of {WINDOWS}")
    if q_max and q_min >= q_max:
        raise ValueError(f"q_min {q_min} is not below q_max {q_max}")

    notes: list[str] = []
    r = np.arange(dr, r_max + 0.5 * dr, dr)
    weights = scattering_weights(structure, radiation)
    n_cell = float(sum(a.occupancy for a in structure.atoms))
    volume = float(structure.cell.volume)
    rho0 = n_cell / volume if volume > 0 else 0.0

    out = PDF(r=r, G=np.zeros_like(r), g=np.ones_like(r), R=np.zeros_like(r),
              rho0=rho0, n_cell=n_cell, radiation=radiation, weights=weights,
              u_iso=float(u_iso), q_min=float(q_min), q_max=float(q_max),
              q_damp=float(q_damp), q_broad=float(q_broad),
              delta1=float(delta1), delta2=float(delta2), window=window,
              structure_name=structure.name or "", notes=notes)
    if not structure.atoms or n_cell <= 0 or volume <= 0:
        notes.append("The structure has no atoms, so there are no pairs.")
        return out

    if abs(weights.b_mean) < 1e-30:
        notes.append(
            f"The mean {radiation} scattering power of this composition is "
            "zero, so the weights are undefined and no pattern was computed. "
            "This happens for a neutron PDF of a composition whose scattering "
            "lengths cancel.")
        return out

    bw = _atom_weights(structure, weights)
    u, from_file, assumed = _atom_u(structure, u_iso)
    out.n_from_file, out.n_assumed = from_file, assumed

    # R(r), accumulated by grouping pairs with the same width. CIFs carry few
    # distinct U values, so a few hundred site pairs collapse to a handful of
    # groups, and each group is one histogram and one convolution rather than
    # one Gaussian per pair.
    radial = np.zeros_like(r)
    n_pairs = 0
    correlated = bool(delta1 or delta2 or q_broad)

    if not correlated:
        # sigma depends only on (U_i, U_j), so group and convolve once per group
        buckets: dict[float, np.ndarray] = {}
        for distances, i, j in pair_distances(structure, float(r_max) + 1e-9):
            n_pairs += len(distances)
            sigma = sigma_pair(u[i], u[j], distances)
            keys = np.round(sigma, 4)
            w = bw[i] * bw[j]
            for key in np.unique(keys):
                here = keys == key
                deposited = _deposit(r, distances[here], w[here])
                buckets[float(key)] = buckets.get(
                    float(key), np.zeros_like(r)) + deposited
        for sigma, hist in buckets.items():
            radial += _broaden(hist, dr, sigma)
    else:
        # sigma varies with r, so the single convolution is not available;
        # accumulate each pair over a window of +-6 sigma
        for distances, i, j in pair_distances(structure, float(r_max) + 1e-9):
            n_pairs += len(distances)
            sigma = sigma_pair(u[i], u[j], distances, delta1=delta1,
                               delta2=delta2, q_broad=q_broad)
            w = bw[i] * bw[j]
            radial += _sum_gaussians(r, distances, sigma, w)
        notes.append(
            "delta1, delta2 or Qbroad is non-zero, so the peak width varies "
            "with r and each pair was summed separately.")

    radial /= (n_cell * weights.b_mean * weights.b_mean)
    out.R = radial
    out.n_pairs = n_pairs

    baseline = 4.0 * math.pi * r * rho0
    G = radial / r - baseline

    if q_max > 0 and window == "boxcar":
        G = apply_termination(r, G, q_min, q_max)
    elif q_max > 0 and window == "Lorch":
        G = _lorch_round_trip(r, G, q_min, q_max)
    if q_damp > 0:
        G = apply_q_damp(r, G, q_damp)

    out.G = G
    with np.errstate(divide="ignore", invalid="ignore"):
        out.g = np.where(baseline > 0, 1.0 + G / baseline, 1.0)

    notes.extend(_notes(out, from_file, assumed, len(structure.atoms)))
    return out


def _deposit(r, distances, weights) -> np.ndarray:
    """Spread each distance linearly over its two neighbouring grid points.

    ``np.histogram`` would put the whole weight in one bin, which moves every
    peak by up to dr/2 and makes the peak position depend on where the grid
    happens to fall. Linear deposition conserves the total weight exactly and
    leaves the first moment right to O(dr^2), so a distance read off the plot is
    the distance in the structure.
    """
    out = np.zeros(len(r), float)
    if not len(distances):
        return out
    dr = float(r[1] - r[0]) if len(r) > 1 else 1.0
    position = (np.asarray(distances, float) - float(r[0])) / dr
    low = np.floor(position).astype(np.int64)
    frac = position - low
    weights = np.asarray(weights, float)
    for index, share in ((low, 1.0 - frac), (low + 1, frac)):
        inside = (index >= 0) & (index < len(r))
        np.add.at(out, index[inside], (weights * share)[inside])
    return out


def _broaden(histogram, dr: float, sigma: float) -> np.ndarray:
    """Convolve a histogram with one Gaussian of width ``sigma``."""
    sigma = max(float(sigma), MIN_SIGMA)
    half = max(int(math.ceil(6.0 * sigma / dr)), 1)
    x = np.arange(-half, half + 1) * dr
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum() * dr                       # unit area, per angstrom
    return np.convolve(histogram, kernel * dr, mode="same") / dr


def _sum_gaussians(r, centres, sigmas, amplitudes) -> np.ndarray:
    """Every pair as its own Gaussian, over a +-6 sigma window."""
    out = np.zeros_like(r)
    dr = float(r[1] - r[0]) if len(r) > 1 else 1.0
    for centre, sigma, amp in zip(np.asarray(centres, float),
                                  np.asarray(sigmas, float),
                                  np.asarray(amplitudes, float)):
        lo = int(max((centre - 6.0 * sigma - r[0]) / dr, 0))
        hi = int(min((centre + 6.0 * sigma - r[0]) / dr + 1, len(r)))
        if hi <= lo:
            continue
        window = r[lo:hi]
        out[lo:hi] += amp * np.exp(
            -0.5 * ((window - centre) / sigma) ** 2) / (
                sigma * math.sqrt(2.0 * math.pi))
    return out


def _notes(out: PDF, from_file: int, assumed: int, n_atoms: int) -> list[str]:
    """Statements of fact about how this pattern was produced."""
    notes = [
        f"{out.radiation} weights at Q = 0; the pair weights are "
        + ", ".join((out.weights.describe() if out.weights else [])),
        f"rho0 = {out.rho0:.5f} atoms/A^3 from {out.n_cell:g} atoms per cell, "
        "occupancies included.",
    ]
    if assumed:
        notes.append(
            f"{from_file} of {n_atoms} atoms took U from the file; the other "
            f"{assumed} used U = {out.u_iso:.5f} A^2, which was assumed.")
    else:
        notes.append(f"All {n_atoms} atoms took U from the file.")
    notes.append(
        "Peak widths are the uncorrelated sigma^2 = U_i + U_j"
        + (f", with delta1 = {out.delta1:g}, delta2 = {out.delta2:g} applied."
           if (out.delta1 or out.delta2) else
           ". Near neighbours move together, so a measured first peak is "
           "narrower than this."))
    if out.q_max > 0:
        notes.append(
            f"Truncated at Qmax = {out.q_max:g} A^-1"
            + (f" from Qmin = {out.q_min:g} A^-1" if out.q_min else "")
            + f" with a {out.window} window, applied as a convolution in r "
            "over the odd extension.")
    else:
        notes.append("No Qmax was applied: this is the untruncated PDF.")
    if out.q_damp > 0:
        notes.append(f"Q-resolution damping exp(-(r Qdamp)^2/2) with "
                     f"Qdamp = {out.q_damp:g} A^-1.")
    notes.append(
        "This is the PDF of the average crystal. A fractionally occupied site "
        "contributes fractional pair correlations, and a measured PDF of a "
        "disordered material differs from this by design.")
    return notes


def to_csv(out: PDF, path, provenance=None):
    """Write r, G(r), g(r) and R(r), with the assumptions in the header."""
    from pathlib import Path

    lines = ["# " + line for line in (provenance or [])]
    lines += ["# " + note for note in out.notes]
    lines.append("# r/A,G(r),g(r),R(r)")
    for r, G, g, R in zip(out.r, out.G, out.g, out.R):
        lines.append(f"{r:.5f},{G:.6e},{g:.6e},{R:.6e}")
    target = Path(path)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
