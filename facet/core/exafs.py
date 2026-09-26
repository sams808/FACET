"""EXAFS: what the structure can say, and what it cannot.

Three things are computed here, and one is deliberately not.

**The shell table.** Per crystallographic shell about an absorber: the element,
its Z, the degeneracy N, the occupancy-weighted N, the mean R with the spread
inside the shell, which sites contribute, and two independently sourced
estimates of sigma^2. This is the parameter list an Artemis or Larch fit starts
from, and every column is a measurement of the file.

**The resolution report.** Over a chosen k range the data's resolution is
dR = pi/(2 dk) and the number of independent points is N_idp = 2 dk dR / pi. So
FACET can state that two shells 0.060 A apart are not separable over
k = 3-14 A^-1, and how many parameters a shell list would need against how many
the range supports. Both are statements about the structure and the range
chosen, never about a spectrum.

**FEFF, read back.** ``files.dat`` and the ``feffNNNN.dat`` path files are plain
text. Parsing them puts FEFF's own degeneracy, effective path length, leg count
and amplitude ratio beside FACET's crystallographic shells, and it shows
something the geometry alone cannot: in Bi2O3 the Bi-Bi path at 3.90 A carries
more than half the first oxygen shell's amplitude, so a count-against-distance
histogram misrepresents what an EXAFS measurement sees. FACET does not bundle
FEFF and does not run it unless pointed at an executable.

**What is not computed: chi(k) from the structure alone.** A single-scattering
chi(k) with no phase shifts puts the peaks of its Fourier transform about 0.4 A
away from the distances that generated it, so a curve produced that way
misstates the one quantity it would be read for. Where FEFF path files are
present, chi(k) *is* assembled -- from FEFF's amplitudes and phases with FACET's
degeneracies, distances and sigma^2 -- and every half of it is labelled with
where it came from. XANES needs full multiple scattering or a DFT calculation;
FACET writes the input for one and does not substitute for it.

The Einstein sigma^2 uses

    sigma^2 = (hbar / (2 mu omega_E)) coth(hbar omega_E / 2 k_B T)
            = 24.2544 / (theta_E mu tanh(theta_E / 2T))   [A^2]

with theta_E in kelvin, mu the reduced mass in atomic mass units and T in
kelvin. The constant folds hbar, k_B and the unit conversions; ``EINSTEIN_C``
below carries its derivation so it can be checked rather than trusted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import elements
from .structure import Structure

# hbar^2 / (2 * k_B * u * 1e-20 m^2) in kelvin*amu*angstrom^-2, so that
# sigma^2 [A^2] = EINSTEIN_C / (theta_E [K] * mu [amu] * tanh(theta_E / 2T)).
# Derivation, in SI: sigma^2 = hbar/(2 mu omega) coth(hbar omega / 2 k_B T), and
# with hbar omega = k_B theta_E this is (hbar^2 / (2 mu k_B theta_E)) coth(...).
#   hbar^2 / (2 k_B u) = (1.054571817e-34)^2 / (2 * 1.380649e-23 * 1.66053907e-27)
#                      = 2.42544e-21 m^2 K amu
# and 1 m^2 = 1e20 A^2, so the constant in A^2 K amu is 24.2544.
EINSTEIN_C = 24.2544

# Planck constant over the electron mass, as the k <-> E conversion needs it:
# k [A^-1] = sqrt(2 m_e (E - E0)) / hbar = 0.5123 * sqrt(E - E0 [eV])
K_PER_SQRT_EV = 0.5123167219534328


@dataclass
class Shell:
    """One crystallographic shell about the absorber."""

    element: str
    z: int
    count: int                       # how many contacts, unweighted
    count_occupancy: float           # the same, weighted by site occupancy
    r_mean: float
    r_min: float
    r_max: float
    r_esd: float = float("nan")      # from the file's coordinate esds, if given
    labels: tuple[str, ...] = ()
    sites: tuple[int, ...] = ()
    sigma2_uncorrelated: float = float("nan")
    sigma2_einstein: float = float("nan")
    reduced_mass: float = float("nan")

    @property
    def spread(self) -> float:
        return self.r_max - self.r_min

    @property
    def is_split(self) -> bool:
        """Whether this shell holds contacts that are not all the same length."""
        return self.spread > 1e-6


@dataclass
class ShellTable:
    """The shells about one absorber, and how they were produced."""

    absorber: str
    absorber_label: str
    site_index: int
    shells: list[Shell] = field(default_factory=list)
    temperature: float = 300.0
    einstein_temperature: float = 500.0
    tolerance: float = 0.05
    r_max: float = 6.0
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.shells)


def einstein_sigma2(theta_e: float, reduced_mass: float,
                    temperature: float) -> float:
    """``sigma^2`` for a correlated pair in the Einstein model, in A^2.

    A model, not a measurement: theta_E is a knob. At theta_E = 500 K and
    T = 300 K a Bi-O pair (mu = 14.4 amu) gives 0.0048 A^2, which is the order
    of magnitude a first-shell fit returns.
    """
    theta_e = float(theta_e)
    reduced_mass = float(reduced_mass)
    if theta_e <= 0 or reduced_mass <= 0:
        return float("nan")
    if temperature <= 0:
        return EINSTEIN_C / (theta_e * reduced_mass)       # the zero-point value
    return EINSTEIN_C / (theta_e * reduced_mass
                         * math.tanh(theta_e / (2.0 * float(temperature))))


def reduced_mass(element_a: str, element_b: str) -> float:
    """``mu = m_a m_b / (m_a + m_b)`` in atomic mass units."""
    ma = elements.info(element_a).weight or 0.0
    mb = elements.info(element_b).weight or 0.0
    if ma <= 0 or mb <= 0:
        return float("nan")
    return ma * mb / (ma + mb)


def k_from_energy(energy_ev, e0_ev: float):
    """``k = 0.5123 sqrt(E - E0)`` in inverse angstrom; NaN below the edge."""
    delta = np.asarray(energy_ev, float) - float(e0_ev)
    out = np.where(delta > 0, K_PER_SQRT_EV * np.sqrt(np.maximum(delta, 0.0)),
                   np.nan)
    return out if out.shape else float(out)


def energy_from_k(k, e0_ev: float):
    """The inverse: ``E = E0 + (k / 0.5123)^2``."""
    k = np.asarray(k, float)
    out = float(e0_ev) + (k / K_PER_SQRT_EV) ** 2
    return out if out.shape else float(out)


# ---------------------------------------------------------------------------
# the shell table
# ---------------------------------------------------------------------------

def shell_table(structure: Structure, site_index: int, *,
                r_max: float = 6.0, tolerance: float = 0.05,
                temperature: float = 300.0,
                einstein_temperature: float = 500.0,
                u_iso_default: float | None = None) -> ShellTable:
    """Group the neighbours of one site into shells, as an EXAFS fit wants them.

    Two things ``utilities.radial_shells`` does that this deliberately does not.

    It merges by chaining: a contact within the tolerance of a shell's *running
    mean* joins it, so three contacts at 3.5555, 3.5787 and 3.6145 A end up in
    one 0.05 A shell although the outer two are 0.059 A apart. Here a contact
    joins only if it is within the tolerance of **every** member, so a shell's
    total spread can never exceed the tolerance.

    It also merges symmetry-inequivalent sites of the same element without
    saying so, and ignores occupancy. Here the contributing site labels are
    listed and the occupancy-weighted count is a column of its own, because FEFF
    has no notion of partial occupancy and a quarter-occupied shell is not four
    atoms.
    """
    from .neighbors import NeighborFinder

    site = structure.sites[site_index]
    table = ShellTable(absorber=site.element, absorber_label=site.label,
                       site_index=site_index, temperature=float(temperature),
                       einstein_temperature=float(einstein_temperature),
                       tolerance=float(tolerance), r_max=float(r_max))

    try:
        finder = NeighborFinder(structure, rmax=float(r_max))
    except ValueError as exc:
        table.notes.append(str(exc))
        return table
    contacts = finder.contacts_for_site(site_index)
    if len(contacts) == 0:
        table.notes.append(
            f"No neighbours of {site.label} within {r_max:g} A.")
        return table

    order = np.argsort(contacts.distance)
    groups: list[list[int]] = []
    for i in order:
        d = float(contacts.distance[i])
        element = contacts.elements[i]
        for group in groups:
            if contacts.elements[group[0]] != element:
                continue
            here = contacts.distance[group]
            # within the tolerance of every member, not of the running mean:
            # otherwise the shell walks outwards one contact at a time
            if (max(float(here.max()), d) - min(float(here.min()), d)
                    <= tolerance):
                group.append(int(i))
                break
        else:
            groups.append([int(i)])

    u_of_site = {}
    for index, s in enumerate(structure.sites):
        u_of_site[index] = (float(s.u_iso) if s.u_iso
                            else (float(u_iso_default)
                                  if u_iso_default is not None else None))
    u_absorber = u_of_site.get(site_index)

    assumed = 0
    for group in groups:
        distances = np.asarray(contacts.distance[group], float)
        element = contacts.elements[group[0]]
        site_indices = tuple(sorted(set(int(contacts.neighbor_site[i])
                                        for i in group)))
        labels = tuple(sorted(set(
            structure.sites[int(contacts.neighbor_site[i])].label
            for i in group)))
        occupancy = float(sum(float(contacts.occupancy[i]) for i in group))

        mu = reduced_mass(site.element, element)
        u_neighbour = [u_of_site.get(i) for i in site_indices]
        if u_absorber is None or any(u is None for u in u_neighbour):
            uncorrelated = float("nan")
            assumed += 1
        else:
            uncorrelated = u_absorber + float(np.mean(
                [u for u in u_neighbour if u is not None]))

        table.shells.append(Shell(
            element=element, z=elements.info(element).z or 0,
            count=len(group), count_occupancy=occupancy,
            r_mean=float(distances.mean()), r_min=float(distances.min()),
            r_max=float(distances.max()), labels=labels, sites=site_indices,
            sigma2_uncorrelated=uncorrelated,
            sigma2_einstein=einstein_sigma2(einstein_temperature, mu,
                                            temperature),
            reduced_mass=mu))

    table.shells.sort(key=lambda s: s.r_mean)
    table.notes.append(
        f"Shells are contacts of one element whose total spread is within "
        f"{tolerance:g} A. A contact joins a shell only if it is within the "
        f"tolerance of every member, so no shell is wider than that.")
    if assumed:
        table.notes.append(
            f"{assumed} of {len(table.shells)} shells have no sigma^2 from "
            "displacement parameters, because the file gives no U for the "
            "absorber or for a contributing site.")
    table.notes.append(
        "sigma^2 (uncorrelated) = U_absorber + U_neighbour, from the file. It "
        "is an upper bound: near neighbours move together, and that "
        "correlation is what it leaves out.")
    table.notes.append(
        f"sigma^2 (Einstein) is a model at theta_E = {einstein_temperature:g} K "
        f"and T = {temperature:g} K, not a measurement of this structure.")
    if any(len(s.sites) > 1 for s in table.shells):
        table.notes.append(
            "Some shells gather more than one crystallographic site; the "
            "labels column says which.")
    if any(abs(s.count_occupancy - s.count) > 1e-6 for s in table.shells):
        table.notes.append(
            "Some contributing sites are partly occupied, so the weighted "
            "count differs from the contact count. FEFF has no partial "
            "occupancy.")
    return table


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    """What a k range can and cannot separate."""

    k_min: float
    k_max: float
    r_min: float
    r_max: float
    delta_r: float                  # pi / (2 dk)
    n_independent: float            # 2 dk dR / pi
    unresolved: list[tuple[int, int, float]] = field(default_factory=list)
    n_parameters: int = 0
    notes: list[str] = field(default_factory=list)


def resolution(table: ShellTable, k_min: float = 3.0, k_max: float = 14.0,
               r_min: float = 1.0, r_max: float | None = None,
               parameters_per_shell: int = 3) -> Resolution:
    """Which shells a k range separates, and how many parameters it supports.

    ``delta_r = pi / (2 dk)`` is the resolution of the transform, and
    ``N_idp = 2 dk dR / pi`` is the number of independent points in the fitting
    window -- the standard Stern criterion. Two shells closer together than
    delta_r cannot be fitted independently over that range, and a shell list
    needing more parameters than N_idp is over-parameterised. Both are arithmetic
    on the range you chose and the distances in your file.
    """
    if k_max <= k_min:
        raise ValueError(f"k_max {k_max} is not above k_min {k_min}")
    if r_max is None:
        r_max = table.r_max
    if r_max <= r_min:
        raise ValueError(f"r_max {r_max} is not above r_min {r_min}")

    dk = float(k_max) - float(k_min)
    dr_window = float(r_max) - float(r_min)
    delta_r = math.pi / (2.0 * dk)
    n_idp = 2.0 * dk * dr_window / math.pi

    inside = [i for i, s in enumerate(table.shells)
              if r_min <= s.r_mean <= r_max]
    unresolved = []
    for a, b in zip(inside, inside[1:]):
        gap = table.shells[b].r_mean - table.shells[a].r_mean
        if gap < delta_r:
            unresolved.append((a, b, float(gap)))

    out = Resolution(k_min=float(k_min), k_max=float(k_max),
                     r_min=float(r_min), r_max=float(r_max),
                     delta_r=delta_r, n_independent=n_idp,
                     unresolved=unresolved,
                     n_parameters=len(inside) * int(parameters_per_shell))
    out.notes.append(
        f"Over k = {k_min:g}-{k_max:g} A^-1 the resolution is "
        f"dR = pi/(2 dk) = {delta_r:.3f} A.")
    out.notes.append(
        f"The window r = {r_min:g}-{r_max:g} A holds "
        f"N_idp = 2 dk dR/pi = {n_idp:.1f} independent points.")
    out.notes.append(
        f"{len(inside)} shells lie in that window; at "
        f"{parameters_per_shell} parameters each that is {out.n_parameters} "
        f"parameters against {n_idp:.1f} independent points.")
    for a, b, gap in unresolved:
        out.notes.append(
            f"{table.shells[a].element} at {table.shells[a].r_mean:.3f} A and "
            f"{table.shells[b].element} at {table.shells[b].r_mean:.3f} A are "
            f"{gap:.3f} A apart, which is below dR = {delta_r:.3f} A.")
    for i in inside:
        shell = table.shells[i]
        if shell.spread > delta_r:
            out.notes.append(
                f"The {shell.element} shell at {shell.r_mean:.3f} A spans "
                f"{shell.spread:.3f} A internally, which is more than "
                f"dR = {delta_r:.3f} A.")
    return out


# ---------------------------------------------------------------------------
# FEFF, read back
# ---------------------------------------------------------------------------

@dataclass
class FeffPath:
    """One scattering path, as FEFF reported it."""

    index: int
    filename: str
    n_legs: int
    degeneracy: float
    r_effective: float
    amplitude_ratio: float = float("nan")   # per cent, as files.dat gives it
    sigma2_feff: float = float("nan")       # what FEFF itself applied, if any
    atoms: list[tuple[str, float]] = field(default_factory=list)
    k: np.ndarray | None = None
    magnitude: np.ndarray | None = None     # |f_eff|
    phase: np.ndarray | None = None         # the total phase, radians
    lambda_k: np.ndarray | None = None      # mean free path
    real_p: np.ndarray | None = None        # Re[p], for the exponent

    @property
    def is_single_scattering(self) -> bool:
        return self.n_legs == 2


def read_files_dat(path: str | Path) -> list[FeffPath]:
    """Parse FEFF's ``files.dat``: one line per path file it wrote.

    The columns, as FEFF8L writes them::

        file        sig2   amp ratio    deg    nlegs  r effective

    The sig2 column is easy to count past, and doing so reads the amplitude
    ratio as the leg count -- which produced a list of hundred-legged paths all
    at 2.0 A. FEFF writes this file only when asked, with
    ``PRINT 0 0 0 0 0 3``, which is why FACET's exported input asks for it.
    """
    path = Path(path)
    out: list[FeffPath] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 6 or not parts[0].lower().startswith("feff"):
            continue
        try:
            sigma2 = float(parts[1])
            amplitude = float(parts[2])
            degeneracy = float(parts[3])
            n_legs = int(float(parts[4]))
            r_eff = float(parts[5])
        except ValueError:
            continue
        digits = "".join(c for c in parts[0] if c.isdigit())
        out.append(FeffPath(index=int(digits) if digits else 0,
                            filename=parts[0], n_legs=n_legs,
                            degeneracy=degeneracy, r_effective=r_eff,
                            amplitude_ratio=amplitude, sigma2_feff=sigma2))
    out.sort(key=lambda p: (p.r_effective, p.index))
    return out


def read_feff_path(path: str | Path) -> FeffPath:
    """Parse one ``feffNNNN.dat``: the amplitude and phase of a single path.

    The columns after the header are
    ``k, real[2*phc], mag[feff], phase[feff], red_factor, lambda, real[p]``.
    The total magnitude is ``red_factor * mag[feff]`` and the total phase is
    ``phase[feff] + real[2*phc]``, which is the combination Artemis uses.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n_legs, degeneracy, r_eff = 2, 1.0, float("nan")
    atoms: list[tuple[str, float]] = []
    data_start = None
    atom_start = None

    for i, line in enumerate(lines):
        text = line.strip()
        # The nleg line is not a header followed by data: it IS the data, with
        # its column names written after the numbers on the same line.
        if "nleg" in text.lower() and "reff" in text.lower():
            parts = text.split()
            try:
                n_legs = int(float(parts[0]))
                degeneracy = float(parts[1])
                r_eff = float(parts[2])
            except (IndexError, ValueError):
                pass
            continue
        if "pot" in text.lower() and text.lower().startswith("x"):
            atom_start = i + 1
            continue
        if "real[2*phc]" in text.replace(" ", ""):
            data_start = i + 1
            break

    if atom_start is not None:
        import math as _math

        for line in lines[atom_start:data_start or len(lines)]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                x, y, z = (float(parts[k]) for k in range(3))
            except ValueError:
                continue
            element = parts[5]
            atoms.append((element, _math.sqrt(x * x + y * y + z * z)))

    if data_start is None:
        # fall back on the first run of seven-column numeric lines
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 7:
                try:
                    [float(p) for p in parts[:7]]
                except ValueError:
                    continue
                data_start = i
                break

    rows = []
    if data_start is not None:
        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                rows.append([float(p) for p in parts[:7]])
            except ValueError:
                continue
    out = FeffPath(index=int("".join(c for c in path.stem if c.isdigit()) or 0),
                   filename=path.name, n_legs=n_legs, degeneracy=degeneracy,
                   r_effective=r_eff, atoms=atoms)
    if rows:
        table = np.asarray(rows, float)
        out.k = table[:, 0]
        phc = table[:, 1]
        magnitude = table[:, 2]
        phase = table[:, 3]
        reduction = table[:, 4]
        out.lambda_k = table[:, 5]
        out.real_p = table[:, 6]
        out.magnitude = reduction * magnitude
        out.phase = phase + phc
    return out


def read_feff_directory(directory: str | Path) -> list[FeffPath]:
    """Every path FEFF wrote in one directory, amplitudes included.

    ``files.dat`` supplies the degeneracies and amplitude ratios; each
    ``feffNNNN.dat`` supplies the k-dependent amplitude and phase. Where both
    exist they are merged, with the listing's degeneracy kept because that is
    the number FEFF used.
    """
    directory = Path(directory)
    listing = {}
    index_file = directory / "files.dat"
    if index_file.is_file():
        for entry in read_files_dat(index_file):
            listing[entry.filename.lower()] = entry

    out: list[FeffPath] = []
    for candidate in sorted(directory.glob("feff[0-9]*.dat")):
        detail = read_feff_path(candidate)
        listed = listing.pop(candidate.name.lower(), None)
        if listed is not None:
            detail.amplitude_ratio = listed.amplitude_ratio
            detail.degeneracy = listed.degeneracy
            if not np.isfinite(detail.r_effective):
                detail.r_effective = listed.r_effective
            detail.n_legs = listed.n_legs
        out.append(detail)
    # paths listed but whose file is absent are still worth reporting
    out.extend(listing.values())
    out.sort(key=lambda p: (p.r_effective, p.index))
    return out


def read_chi(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column ``chi.dat`` or ``xmu.dat``-style file.

    Comment lines beginning with ``#`` or ``*`` are skipped, and the first two
    numeric columns are taken. FEFF's ``xmu.dat`` has five columns, of which the
    first is energy and the third is k; use ``read_xmu`` for that one.
    """
    path = Path(path)
    x, y = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text[0] in "#*":
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        try:
            x.append(float(parts[0]))
            y.append(float(parts[1]))
        except ValueError:
            continue
    return np.asarray(x, float), np.asarray(y, float)


def read_xmu(path: str | Path) -> dict[str, np.ndarray]:
    """Read FEFF's ``xmu.dat``: energy, e-e0, k, mu, mu0, chi."""
    path = Path(path)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text[0] in "#*":
            continue
        parts = text.split()
        if len(parts) < 5:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    if not rows:
        return {}
    width = min(len(r) for r in rows)
    table = np.asarray([r[:width] for r in rows], float)
    names = ["energy", "e_minus_e0", "k", "mu", "mu0", "chi"][:width]
    return {name: table[:, i] for i, name in enumerate(names)}


# ---------------------------------------------------------------------------
# chi(k), assembled from FEFF amplitudes and FACET geometry
# ---------------------------------------------------------------------------

@dataclass
class Chi:
    """chi(k) built from FEFF paths, with the origin of each half recorded."""

    k: np.ndarray
    chi: np.ndarray
    per_path: dict[str, np.ndarray] = field(default_factory=dict)
    s02: float = 1.0
    notes: list[str] = field(default_factory=list)


def chi_from_paths(paths: list[FeffPath], *, k=None, s02: float = 1.0,
                   sigma2: dict[int, float] | None = None,
                   delta_r: dict[int, float] | None = None,
                   e0_shift: float = 0.0,
                   max_paths: int | None = None) -> Chi:
    """Sum FEFF's paths into chi(k), using FACET's sigma^2 where given.

    The standard EXAFS path sum::

        chi(k) = sum_p S0^2 N_p |f_p(k)| / (k R_p^2)
                 exp(-2 R_p / lambda_p(k)) exp(-2 sigma_p^2 k^2)
                 sin(2 k R_p + phi_p(k))

    Every k-dependent quantity -- ``|f|``, ``phi`` and ``lambda`` -- comes from
    FEFF and none of it is invented here; the degeneracies come from FEFF's own
    listing; ``sigma^2`` and ``delta_R`` are yours, and default to FEFF's own
    (zero) if not given. This is not a fit: nothing is adjusted to any data.

    Refuses rather than guesses when there are no paths with amplitudes, because
    a chi(k) without FEFF's phases would be wrong by about 0.4 A in the exact
    quantity someone would read off its transform.
    """
    usable = [p for p in paths
              if p.k is not None and p.magnitude is not None
              and p.phase is not None and len(p.k) > 1]
    notes: list[str] = []
    if not usable:
        return Chi(k=np.zeros(0), chi=np.zeros(0), s02=float(s02), notes=[
            "No FEFF path files with amplitudes were found, so no chi(k) was "
            "assembled. FACET does not compute chi(k) from the structure "
            "alone: without FEFF's phase shifts the peaks of its Fourier "
            "transform land about 0.4 A away from the distances that produced "
            "them."])
    usable.sort(key=lambda p: -abs(p.amplitude_ratio)
                if np.isfinite(p.amplitude_ratio) else 0.0)
    if max_paths:
        dropped = len(usable) - int(max_paths)
        usable = usable[:int(max_paths)]
        if dropped > 0:
            notes.append(f"{dropped} weaker paths were not included.")

    if k is None:
        lo = max(float(min(p.k.min() for p in usable)), 0.5)
        hi = float(min(p.k.max() for p in usable))
        k = np.arange(lo, hi + 0.025, 0.05)
    k = np.asarray(k, float)
    total = np.zeros_like(k)
    per_path: dict[str, np.ndarray] = {}

    sigma2 = sigma2 or {}
    delta_r = delta_r or {}
    for p in usable:
        r = float(p.r_effective) + float(delta_r.get(p.index, 0.0))
        if not np.isfinite(r) or r <= 0:
            continue
        s2 = float(sigma2.get(p.index, 0.0))
        magnitude = np.interp(k, p.k, p.magnitude)
        phase = np.interp(k, p.k, p.phase)
        mean_free = (np.interp(k, p.k, p.lambda_k)
                     if p.lambda_k is not None else None)
        damping = np.ones_like(k)
        if mean_free is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                damping = np.exp(-2.0 * r / np.where(mean_free > 0,
                                                     mean_free, np.inf))
        with np.errstate(divide="ignore", invalid="ignore"):
            amplitude = np.where(
                k > 0,
                float(s02) * p.degeneracy * magnitude / (k * r * r), 0.0)
        contribution = (amplitude * damping
                        * np.exp(-2.0 * s2 * k * k)
                        * np.sin(2.0 * k * r + phase))
        per_path[p.filename] = contribution
        total = total + contribution

    notes.append(
        f"chi(k) summed over {len(per_path)} FEFF paths. The amplitudes, "
        "phases and mean free paths are FEFF's; the degeneracies are FEFF's; "
        "sigma^2 and any delta_R are the values supplied here. Nothing was "
        "fitted to any measurement.")
    if sigma2:
        notes.append(f"sigma^2 was supplied for {len(sigma2)} paths.")
    else:
        notes.append("sigma^2 was not supplied, so every path is at "
                     "sigma^2 = 0 and the amplitude is an upper bound.")
    return Chi(k=k, chi=total, per_path=per_path, s02=float(s02),
               notes=notes + notes[:0])


def compare_to_shells(paths: list[FeffPath], table: ShellTable,
                      tolerance: float = 0.15) -> list[dict]:
    """Put FEFF's paths beside FACET's crystallographic shells.

    Matched by effective path length, which for a single-scattering path is the
    interatomic distance. Multiple-scattering paths match no shell, and that is
    the point of showing them: they are what a shell-by-shell reading of the
    geometry leaves out.
    """
    rows = []
    for p in sorted(paths, key=lambda q: q.r_effective):
        match = None
        if p.is_single_scattering and table.shells:
            # the NEAREST shell within the tolerance, not the first one found:
            # taking the first matched a 2.200 A path to a shell at 2.119 A
            # while a shell at 2.207 A was sitting right there
            gaps = [(abs(shell.r_mean - p.r_effective), i)
                    for i, shell in enumerate(table.shells)]
            gap, best = min(gaps)
            if gap <= tolerance:
                match = best
        shell = table.shells[match] if match is not None else None
        rows.append({
            "feff_file": p.filename,
            "n_legs": p.n_legs,
            "r_effective": p.r_effective,
            "degeneracy": p.degeneracy,
            "amplitude_ratio": p.amplitude_ratio,
            "shell": match,
            "shell_element": shell.element if shell else "",
            "shell_r": shell.r_mean if shell else float("nan"),
            "shell_count": shell.count if shell else 0,
            "kind": ("single scattering" if p.is_single_scattering
                     else f"{p.n_legs}-leg"),
        })
    return rows


def find_feff_executable(hint: str | Path | None = None) -> Path | None:
    """Locate a FEFF executable, if the user has one. Never bundled.

    An absent external program is not a dependency: FACET writes the input and
    reads the output, and runs the calculation only when pointed at a binary.
    """
    import shutil

    if hint:
        candidate = Path(hint)
        if candidate.is_file():
            return candidate
    for name in ("feff8l", "feff8l.bat", "feff6l", "feff85L", "feff"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None
