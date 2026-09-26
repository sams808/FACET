"""Powder diffraction from a structure.

Structure factors with real atomic scattering factors, not the geometry-only
hkl table :mod:`facet.core.utilities` already provides. Scattering factors come
from gemmi's implementation of the standard tables, so nothing is retyped here
and nothing new is bundled -- gemmi is already a hard dependency and ships in
the executable:

* X-ray: International Tables Vol C table 6.1.1.4, the four-Gaussian
  Cromer-Mann parameterisation.
* neutron: coherent scattering lengths in fm, angle-independent, signed. Some
  are negative (H -3.74, Ti -3.44, Mn -3.73), which is why a neutron pattern
  sees a light atom beside a heavy one at all. For bismuth oxides that matters:
  Bi so dominates an X-ray pattern that oxygen is nearly invisible, which is
  exactly why the trustworthy Bi-O structures are neutron refinements.
* electron: Peng's five-Gaussian fit, in angstrom.

What this does *not* do: no Rietveld refinement, no preferred orientation, no
absorption correction, no anisotropic broadening. It computes what a pattern
should look like from a structure you already have. Judging how well that
matches a measurement is left to you, which is the point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import elements
from .structure import Structure

# Characteristic wavelengths in angstrom. "Ka" is the intensity-weighted mean of
# the Ka1/Ka2 doublet, which is what an unmonochromated laboratory pattern
# actually contains.
WAVELENGTHS: dict[str, float] = {
    "Cu Ka1": 1.540598,
    "Cu Ka": 1.541838,
    "Cu Ka2": 1.544426,
    "Cu Kb": 1.392250,
    "Mo Ka1": 0.709317,
    "Mo Ka": 0.710730,
    "Co Ka1": 1.788996,
    "Co Ka": 1.790260,
    "Cr Ka1": 2.289700,
    "Fe Ka1": 1.936042,
    "Ag Ka1": 0.559422,
    "synchrotron 0.7": 0.700000,
    "synchrotron 0.5": 0.500000,
    "neutron 1.54": 1.540000,
    "neutron 2.41": 2.410000,
}

RADIATIONS = ("X-ray", "neutron", "electron")

# Debye-Waller factor used when the file gave no displacement parameter. Leaving
# it out makes high-angle reflections far too strong; 0.5 A^2 is an unremarkable
# oxide value. It is an argument everywhere so it is never silently assumed.
DEFAULT_B_ISO = 0.5


@dataclass
class Reflection:
    """One powder line: a family of symmetry-equivalent reflections."""

    h: int
    k: int
    l: int
    d: float                       # angstrom
    two_theta: float               # degrees
    intensity: float               # relative, strongest line = 100
    multiplicity: int              # equivalents merged into this line
    f_squared: float               # |F|^2 per equivalent
    lorentz_polarisation: float

    @property
    def hkl(self) -> str:
        return f"{self.h} {self.k} {self.l}"

    @property
    def q(self) -> float:
        """Scattering vector magnitude, 2 pi / d."""
        return 2.0 * math.pi / self.d if self.d else float("inf")


@dataclass
class Pattern:
    """A calculated pattern: the reflection list, plus how it was made."""

    reflections: list[Reflection] = field(default_factory=list)
    wavelength: float = 1.540598
    radiation: str = "X-ray"
    structure_name: str = ""
    b_iso: float = DEFAULT_B_ISO
    notes: list[str] = field(default_factory=list)

    @property
    def two_theta(self) -> np.ndarray:
        return np.array([r.two_theta for r in self.reflections])

    @property
    def intensity(self) -> np.ndarray:
        return np.array([r.intensity for r in self.reflections])

    @property
    def d(self) -> np.ndarray:
        return np.array([r.d for r in self.reflections])

    def strongest(self, n: int = 10) -> list[Reflection]:
        return sorted(self.reflections, key=lambda r: -r.intensity)[:n]

    def profile(self, two_theta_min: float = 5.0, two_theta_max: float = 90.0,
                step: float = 0.02, u: float = 0.010, v: float = -0.004,
                w: float = 0.006, eta: float = 0.5,
                zero_shift: float = 0.0):
        """Convolve the lines with a peak shape to get a continuous pattern.

        Pseudo-Voigt, with the width following Caglioti's
        ``FWHM^2 = U tan^2(theta) + V tan(theta) + W``. The defaults are a
        typical laboratory diffractometer. They are arguments because the
        instrument is not a property of the structure, and FACET will not
        pretend to know which diffractometer measured the pattern you are
        comparing against.

        Returns ``(two_theta, intensity)`` with the maximum scaled to 100.
        """
        x = np.arange(two_theta_min, two_theta_max + step, step)
        y = np.zeros_like(x)
        for r in self.reflections:
            centre = r.two_theta + zero_shift
            if not (two_theta_min - 2.0 <= centre <= two_theta_max + 2.0):
                continue
            t = math.tan(math.radians(centre / 2.0))
            fwhm = math.sqrt(max(u * t * t + v * t + w, 1.0e-6))
            y += r.intensity * _pseudo_voigt(x, centre, fwhm, eta)
        peak = float(y.max()) if y.size and y.max() > 0.0 else 1.0
        return x, y / peak * 100.0


def _pseudo_voigt(x, centre: float, fwhm: float, eta: float):
    """Gaussian and Lorentzian of equal FWHM, mixed by ``eta``.

    Each component is normalised to unit height, not unit area, so ``eta``
    changes the tails without changing the peak height -- which is how the
    mixing parameter behaves in a refinement program.
    """
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gaussian = np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    gamma = fwhm / 2.0
    lorentzian = gamma * gamma / ((x - centre) ** 2 + gamma * gamma)
    return eta * lorentzian + (1.0 - eta) * gaussian


# ---------------------------------------------------------------------------
# scattering factors
# ---------------------------------------------------------------------------
# gemmi's calculate_sf() takes stol2 = (sin(theta)/lambda)^2, NOT sin(theta)/
# lambda. Passing s where s^2 belongs gives factors wrong by tens of percent at
# high angle that still look plausible, so everything below carries stol2
# explicitly rather than a bare "s".

def scattering_coefficients(element: str, radiation: str = "X-ray"):
    """Gaussian coefficients ``(a, b, c)`` for one element.

    ``f(stol2) = sum_i a_i exp(-b_i stol2) + c``. A neutron length comes back as
    a single term with ``b = 0``, so one expression covers all three radiations.
    """
    import gemmi

    symbol = elements.normalise(element)
    try:
        el = gemmi.Element(symbol)
    except Exception:
        el = None
    if el is None or not el.atomic_number:
        return np.zeros(1), np.zeros(1), 0.0

    kind = radiation.lower()
    if kind.startswith("neutron"):
        coefs = list(el.neutron92.get_coefs())
        length = float(coefs[0]) if coefs else 0.0
        return np.array([length]), np.zeros(1), 0.0
    if kind.startswith("electron"):
        table = el.c4322
        return (np.array(list(table.a), float),
                np.array(list(table.b), float), 0.0)
    table = el.it92
    return (np.array(list(table.a), float), np.array(list(table.b), float),
            float(table.c))


def form_factor(element: str, stol2, radiation: str = "X-ray"):
    """Scattering factor at ``stol2 = (sin(theta)/lambda)^2``.

    Works on a scalar or an array. X-ray factors fall away with angle, which is
    what makes a high-angle reflection weak; neutron lengths do not depend on
    angle at all.
    """
    a, b, c = scattering_coefficients(element, radiation)
    s2 = np.asarray(stol2, float)
    total = np.full(s2.shape, c, float)
    for coefficient, exponent in zip(a, b):
        total = total + coefficient * np.exp(-exponent * s2)
    return total if total.shape else float(total)


def debye_waller(b_iso: float, stol2):
    """``exp(-B (sin(theta)/lambda)^2)``.

    Equivalently ``exp(-2 pi^2 U / d^2)``, since ``(sin theta / lambda)^2 =
    1/(4 d^2)`` and ``B = 8 pi^2 U``.
    """
    return np.exp(-float(b_iso) * np.asarray(stol2, float))


def b_from_u(u_iso: float) -> float:
    """``B = 8 pi^2 U``, the conversion a CIF makes you do by hand."""
    return 8.0 * math.pi * math.pi * float(u_iso)


def lorentz_polarisation(two_theta_deg,
                         monochromator_two_theta: float | None = None):
    """Lorentz-polarisation factor for a powder.

    ``(1 + cos^2 2theta) / (sin^2 theta cos theta)``. With a monochromator the
    polarisation term becomes ``(1 + K cos^2 2theta)/(1 + K)``, where ``K`` is
    ``cos^2`` of the monochromator's own 2-theta. That changes the high-angle
    end noticeably on a laboratory instrument, so it is offered rather than
    assumed.
    """
    two_theta = np.radians(np.asarray(two_theta_deg, float))
    theta = two_theta / 2.0
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    if monochromator_two_theta is None:
        polarisation = 1.0 + np.cos(two_theta) ** 2
    else:
        k = math.cos(math.radians(monochromator_two_theta)) ** 2
        polarisation = (1.0 + k * np.cos(two_theta) ** 2) / (1.0 + k)
    denominator = sin_theta * sin_theta * cos_theta
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(denominator) > 1.0e-12,
                       polarisation / denominator, 0.0)
    return out if out.shape else float(out)


# ---------------------------------------------------------------------------
# structure factors
# ---------------------------------------------------------------------------

def _atom_arrays(structure: Structure, b_iso: float):
    """Per-atom coordinates, weights and displacement parameters.

    The structure is already symmetry-expanded to one full unit cell with no
    boundary duplicates, so summing over its atoms handles symmetry without
    needing the operators -- a systematic absence comes out as a sum that
    cancels, rather than as a rule that has to be predicted.
    """
    fracs, occupancies, b_values, symbols = [], [], [], []
    for atom in structure.atoms:
        site = structure.sites[atom.site_index]
        fracs.append(atom.frac)
        occupancies.append(atom.occupancy)
        u_iso = site.u_iso
        b_values.append(b_from_u(u_iso) if u_iso else float(b_iso))
        symbols.append(atom.element)
    if not fracs:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0), []
    return (np.asarray(fracs, float), np.asarray(occupancies, float),
            np.asarray(b_values, float), symbols)


def structure_factor(structure: Structure, h: int, k: int, l: int,
                     stol2: float | None = None, radiation: str = "X-ray",
                     b_iso: float = DEFAULT_B_ISO) -> complex:
    """``F(hkl)`` summed over every atom of the cell.

    ``stol2`` defaults to the value implied by the cell, so a caller that knows
    only hkl need not compute it.
    """
    from .utilities import d_spacing

    if stol2 is None:
        d = d_spacing(structure, h, k, l)
        stol2 = 1.0 / (4.0 * d * d) if np.isfinite(d) and d > 0 else 0.0

    fracs, occupancies, b_values, symbols = _atom_arrays(structure, b_iso)
    if not len(fracs):
        return 0j

    cache: dict[str, float] = {}
    total = 0j
    hkl = np.array([h, k, l], float)
    for i, symbol in enumerate(symbols):
        if symbol not in cache:
            cache[symbol] = float(form_factor(symbol, stol2, radiation))
        weight = (occupancies[i] * cache[symbol]
                  * float(debye_waller(b_values[i], stol2)))
        phase = 2.0 * math.pi * float(hkl @ fracs[i])
        total += weight * complex(math.cos(phase), math.sin(phase))
    return total


def _structure_factors(hkl: np.ndarray, stol2: np.ndarray,
                       fracs: np.ndarray, occupancies: np.ndarray,
                       b_values: np.ndarray, symbols: list[str],
                       radiation: str, chunk: int = 2048) -> np.ndarray:
    """``|F|^2`` for many reflections at once.

    Chunked over reflections so the phase matrix stays bounded: a large cell
    over a fine index range would otherwise want a reflections-by-atoms array of
    hundreds of megabytes.
    """
    unique = sorted(set(symbols))
    column = {symbol: i for i, symbol in enumerate(unique)}
    index = np.array([column[s] for s in symbols])

    out = np.empty(len(hkl), float)
    for start in range(0, len(hkl), chunk):
        stop = min(start + chunk, len(hkl))
        s2 = stol2[start:stop]                                   # (M,)

        # form factor per element, then spread over the atoms
        per_element = np.stack(
            [np.asarray(form_factor(s, s2, radiation), float).reshape(-1)
             for s in unique], axis=1)                           # (M, E)
        f = per_element[:, index]                                # (M, N)
        f = f * np.exp(-np.outer(s2, b_values))                  # Debye-Waller
        f = f * occupancies

        phase = 2.0 * np.pi * (hkl[start:stop] @ fracs.T)        # (M, N)
        real = np.einsum("mn,mn->m", f, np.cos(phase))
        imaginary = np.einsum("mn,mn->m", f, np.sin(phase))
        out[start:stop] = real * real + imaginary * imaginary
    return out


# ---------------------------------------------------------------------------
# the pattern
# ---------------------------------------------------------------------------

def index_bounds(structure: Structure, d_min: float,
                 ceiling: int = 60) -> tuple[int, int, int]:
    """The smallest index range that misses no reflection above ``d_min``.

    For a reflection ``r* = h a* + k b* + l c*`` the index ``h`` is ``r* . a``,
    so ``|h| <= |r*| |a| = a / d_min`` -- and likewise for k and l.

    Using this instead of one fixed cube matters for the merged multiplicities:
    a fixed cube can hold a reflection while leaving one of its symmetry
    equivalents outside, and the line then comes out weak by exactly the
    fraction of equivalents that were clipped. Trigonal and hexagonal cells are
    where that bites, because the three-fold maps ``(h, k)`` to ``(-h-k, h)``
    and pushes indices out of any cube.
    """
    if d_min <= 0:
        return (ceiling, ceiling, ceiling)
    bounds = [min(int(math.ceil(length / d_min)) + 1, ceiling)
              for length in structure.cell.lengths]
    return (max(bounds[0], 1), max(bounds[1], 1), max(bounds[2], 1))


def powder_pattern(structure: Structure, wavelength: float = 1.540598,
                   two_theta_max: float = 90.0, two_theta_min: float = 0.0,
                   radiation: str = "X-ray", b_iso: float = DEFAULT_B_ISO,
                   min_intensity: float = 0.01,
                   monochromator_two_theta: float | None = None,
                   max_index: int | None = None) -> Pattern:
    """Every powder line within a 2-theta range, with its relative intensity.

    Reflections are enumerated over a full sphere and then merged by d-spacing,
    so the multiplicity falls out of the merge rather than from a symmetry
    lookup -- the same reason the structure factor needs no operators. Lines
    that coincide in d merge, which is what a powder does to them.
    """
    from .utilities import metric_tensor

    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if not 0.0 < two_theta_max <= 180.0:
        raise ValueError("two_theta_max must be in (0, 180]")

    reciprocal = np.linalg.inv(metric_tensor(structure))   # 1/d^2 = hkl G* hkl
    d_min = wavelength / (2.0 * math.sin(math.radians(two_theta_max / 2.0)))
    q2_max = 1.0 / (d_min * d_min)

    bounds = index_bounds(structure, d_min)
    clipped = False
    if max_index is not None:
        clipped = any(b > max_index for b in bounds)
        bounds = (min(bounds[0], max_index), min(bounds[1], max_index),
                  min(bounds[2], max_index))

    grid = np.stack(np.meshgrid(
        np.arange(-bounds[0], bounds[0] + 1),
        np.arange(-bounds[1], bounds[1] + 1),
        np.arange(-bounds[2], bounds[2] + 1), indexing="ij"), axis=-1)
    hkl = grid.reshape(-1, 3).astype(float)
    hkl = hkl[np.any(hkl != 0.0, axis=1)]                  # drop 000

    q2 = np.einsum("mi,ij,mj->m", hkl, reciprocal, hkl)
    keep = (q2 > 0) & (q2 <= q2_max * (1.0 + 1.0e-9))
    hkl, q2 = hkl[keep], q2[keep]

    empty = Pattern(wavelength=wavelength, radiation=radiation,
                    structure_name=structure.name or "", b_iso=b_iso,
                    notes=["no reflections fall in the requested range"])
    if not len(hkl):
        return empty

    d = 1.0 / np.sqrt(q2)
    stol2 = q2 / 4.0
    two_theta = 2.0 * np.degrees(np.arcsin(np.clip(wavelength / (2.0 * d),
                                                   -1.0, 1.0)))
    window = (two_theta >= two_theta_min) & (two_theta <= two_theta_max)
    hkl, q2, d, stol2, two_theta = (hkl[window], q2[window], d[window],
                                    stol2[window], two_theta[window])
    if not len(hkl):
        return empty

    fracs, occupancies, b_values, symbols = _atom_arrays(structure, b_iso)
    if not len(fracs):
        raise ValueError(f"{structure.name}: no atoms to scatter from")
    f2 = _structure_factors(hkl, stol2, fracs, occupancies, b_values,
                            symbols, radiation)

    # Merge symmetry equivalents, which share a d-spacing exactly. Grouping on
    # q2 with a relative tolerance does that robustly; accidental coincidences
    # merge too, which is correct -- a powder cannot resolve them either.
    order = np.argsort(q2)
    hkl, q2, d, two_theta, f2 = (hkl[order], q2[order], d[order],
                                 two_theta[order], f2[order])
    breaks = np.nonzero(np.diff(q2) > 1.0e-9 * np.maximum(q2[:-1], 1.0e-12))[0]
    starts = np.concatenate(([0], breaks + 1))
    stops = np.concatenate((breaks + 1, [len(q2)]))

    reflections: list[Reflection] = []
    for start, stop in zip(starts, stops):
        group = slice(int(start), int(stop))
        multiplicity = int(stop - start)
        total_f2 = float(np.sum(f2[group]))
        centre = float(np.mean(two_theta[group]))
        lp = float(lorentz_polarisation(centre, monochromator_two_theta))
        # total_f2 already sums over the equivalents, so the multiplicity must
        # not be applied a second time
        representative = _representative(hkl[group].astype(int))
        reflections.append(Reflection(
            h=int(representative[0]), k=int(representative[1]),
            l=int(representative[2]), d=float(np.mean(d[group])),
            two_theta=centre, intensity=total_f2 * lp,
            multiplicity=multiplicity, f_squared=total_f2 / multiplicity,
            lorentz_polarisation=lp))

    peak = max((r.intensity for r in reflections), default=0.0)
    if peak > 0:
        for r in reflections:
            r.intensity = 100.0 * r.intensity / peak
    reflections = [r for r in reflections if r.intensity >= min_intensity]
    reflections.sort(key=lambda r: r.two_theta)

    pattern = Pattern(reflections=reflections, wavelength=wavelength,
                      radiation=radiation, b_iso=b_iso,
                      structure_name=structure.name or "")
    pattern.notes.append(
        f"{radiation}, lambda = {wavelength:.6f} A, indices to "
        f"{bounds[0]}/{bounds[1]}/{bounds[2]} (set by d_min = {d_min:.3f} A)")
    with_u = sum(1 for a in structure.atoms
                 if structure.sites[a.site_index].u_iso)
    if with_u and with_u == len(structure.atoms):
        pattern.notes.append(
            "displacement parameters came from the file for every atom")
    elif with_u:
        pattern.notes.append(
            f"displacement parameters came from the file for {with_u} of "
            f"{len(structure.atoms)} atoms; the rest used B = {b_iso} A^2")
    else:
        pattern.notes.append(
            f"the file gave no displacement parameters, so B = {b_iso} A^2 was "
            "used for every atom")
    if monochromator_two_theta is not None:
        pattern.notes.append(
            "polarisation for a monochromator at 2theta = "
            f"{monochromator_two_theta:.2f} deg")
    pattern.notes.append(
        "no preferred orientation, absorption or surface roughness correction "
        "is applied, and nothing here is refined against a measurement")
    if clipped:
        pattern.notes.append(
            f"the index range was capped at {max_index}, below what d_min "
            "requires; some reflections in range were not enumerated")
    return pattern


def _representative(candidates: np.ndarray) -> np.ndarray:
    """Pick the conventional label for a family of equivalent reflections.

    All-positive if the family has one, then largest index first, which is how a
    reflection gets written on a figure.
    """
    positive = candidates[np.all(candidates >= 0, axis=1)]
    pool = positive if len(positive) else candidates
    order = np.lexsort((-pool[:, 2], -pool[:, 1], -pool[:, 0]))
    return pool[order[0]]


# ---------------------------------------------------------------------------
# comparing with a measurement
# ---------------------------------------------------------------------------

PATTERN_FILE_FILTER = (
    "Powder patterns (*.xy *.xye *.dat *.txt *.asc *.chi *.csv);;"
    "Two-column (*.xy *.dat *.txt);;"
    "With uncertainties (*.xye);;"
    "All files (*)"
)


def read_pattern(path):
    """A measured pattern from a text file. Returns ``(x, y)``.

    Handles the usual laboratory exports -- .xy, .xye, .dat, .txt, .chi, .csv --
    with comment lines beginning ``#``, ``!``, ``'``, ``;`` or ``*``, and any
    number of header lines. A third column is taken as an uncertainty and is not
    returned; this comparison does not weight by it.
    """
    from pathlib import Path

    path = Path(path)
    x, y = [], []
    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#!';*/":
            continue
        parts = stripped.replace(",", " ").replace("\t", " ").split()
        if len(parts) < 2:
            continue
        try:
            first, second = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if not (math.isfinite(first) and math.isfinite(second)):
            continue
        x.append(first)
        y.append(second)
    if len(x) < 2:
        raise ValueError(
            f"{path.name}: no two-column numeric data found. FACET expects "
            "2-theta and intensity, one pair per line.")
    order = np.argsort(x)
    return np.asarray(x)[order], np.asarray(y)[order]


def scale_to_measured(calculated_x, calculated_y, measured_x, measured_y):
    """Put a calculated pattern on the scale of a measured one.

    A single least-squares scale factor over the overlapping 2-theta range, and
    nothing else: no background, no peak shifts, no profile fitting. That is the
    honest minimum -- it makes the two curves comparable without implying that
    anything was refined. Returns ``(scaled_y, factor)``.
    """
    calculated_x = np.asarray(calculated_x, float)
    calculated_y = np.asarray(calculated_y, float)
    measured_x = np.asarray(measured_x, float)
    measured_y = np.asarray(measured_y, float)

    low = max(float(calculated_x.min()), float(measured_x.min()))
    high = min(float(calculated_x.max()), float(measured_x.max()))
    if high <= low:
        return calculated_y, 1.0
    grid = np.linspace(low, high, 2000)
    calculated = np.interp(grid, calculated_x, calculated_y)
    measured = np.interp(grid, measured_x, measured_y)
    denominator = float(np.sum(calculated * calculated))
    factor = (float(np.sum(calculated * measured)) / denominator
              if denominator > 0 else 1.0)
    return calculated_y * factor, factor


def difference_curve(calculated_x, calculated_y, measured_x, measured_y):
    """``measured - calculated`` on the measured grid, after scaling.

    Returns ``(x, difference, factor)``. A difference curve is a measurement,
    not a verdict: FACET draws it and says nothing about whether it is good.
    """
    scaled, factor = scale_to_measured(calculated_x, calculated_y,
                                       measured_x, measured_y)
    measured_x = np.asarray(measured_x, float)
    on_measured = np.interp(measured_x, np.asarray(calculated_x, float), scaled)
    return measured_x, np.asarray(measured_y, float) - on_measured, factor
