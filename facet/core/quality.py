"""Structure health: what is wrong with this file, stated as measurements.

Real collections contain broken entries. The reference set used to build FACET
holds an atom loop of ``? ? ? ?``, oxygen on the wrong Wyckoff site, a P–O
distance of 1.10 Å in a diffraction-card export, and coordinate esds of 0.07 in
fractional units. Reading any of those as if it were sound produces a
confident, wrong number.

Each check returns what it measured and the value it compared against. None of
them says whether a structure is usable — that depends on the question, and a
file that is useless for a bond-valence sum may be perfectly good for indexing
a powder pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from . import elements


class Level(IntEnum):
    """How far a measurement is from the ordinary range.

    Not a verdict. NOTE means "worth knowing", CHECK means "outside what is
    usually seen", IMPOSSIBLE means "no structure has ever shown this".
    """

    NOTE = 0
    CHECK = 1
    IMPOSSIBLE = 2


@dataclass
class Finding:
    level: Level
    code: str
    message: str
    where: str = ""
    value: float | None = None
    reference: float | None = None

    def __str__(self) -> str:
        return f"[{self.level.name}] {self.where + ': ' if self.where else ''}{self.message}"


@dataclass
class HealthReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def worst(self) -> Level | None:
        return max((f.level for f in self.findings), default=None)

    def of_level(self, level: Level) -> list[Finding]:
        return [f for f in self.findings if f.level == level]

    def __bool__(self) -> bool:
        return bool(self.findings)

    def __len__(self) -> int:
        return len(self.findings)


# Shortest bond ever observed between two non-metal atoms is about 1.09 A
# (H-H in dihydrogen is 0.74, but that is not a case FACET meets). Below 1.2 A
# between heavy atoms, the coordinates are wrong.
MIN_PHYSICAL_BOND = 1.20
# Below this, two positions are a split site rather than two atoms.
SPLIT_POSITION = 0.80


def check(structure, results=None, v_bond: float = 0.075) -> HealthReport:
    """Every check, over one structure and optionally its analysis."""
    report = HealthReport()
    _check_coordinates(structure, report)
    _check_contacts(structure, report)
    _check_occupancy(structure, report)
    _check_charge(structure, report)
    _check_esds(structure, report)
    if results:
        _check_valences(results, report, v_bond)
    for note in structure.notes:
        report.findings.append(Finding(Level.NOTE, "note", note))
    return report


def _check_coordinates(structure, report: HealthReport) -> None:
    if not structure.atoms:
        report.findings.append(Finding(
            Level.IMPOSSIBLE, "no-atoms",
            "the file declares a cell but no atomic coordinates"))
        return
    for atom in structure.atoms:
        if not np.all(np.isfinite(atom.frac)):
            report.findings.append(Finding(
                Level.IMPOSSIBLE, "bad-coordinate",
                "coordinates are not finite", atom.label))


def _check_contacts(structure, report: HealthReport) -> None:
    """Distances no real structure shows.

    Split positions are reported separately from impossible contacts: a pair
    0.3 A apart is a disorder model, a pair 1.1 A apart is an error.
    """
    if len(structure.atoms) < 2:
        return
    from scipy.spatial import cKDTree

    positions = structure.cart_array()
    tree = cKDTree(positions)
    for i, j in tree.query_pairs(MIN_PHYSICAL_BOND):
        d = float(np.linalg.norm(positions[i] - positions[j]))
        a, b = structure.atoms[i], structure.atoms[j]
        occ = min(a.occupancy, b.occupancy)
        if d < SPLIT_POSITION or occ < 0.99:
            report.findings.append(Finding(
                Level.NOTE, "split-position",
                f"{a.label} and {b.label} are {d:.3f} Å apart with occupancy "
                f"{a.occupancy:.2f} and {b.occupancy:.2f} — a split or shared "
                f"site", f"{a.label}/{b.label}", d, SPLIT_POSITION))
        else:
            report.findings.append(Finding(
                Level.IMPOSSIBLE, "short-contact",
                f"{a.label}–{b.label} is {d:.3f} Å, shorter than any "
                f"interatomic distance ({MIN_PHYSICAL_BOND:.2f} Å)",
                f"{a.label}/{b.label}", d, MIN_PHYSICAL_BOND))


def _check_occupancy(structure, report: HealthReport) -> None:
    for site in structure.sites:
        if site.occupancy > 1.0001:
            report.findings.append(Finding(
                Level.IMPOSSIBLE, "occupancy-above-one",
                f"occupancy {site.occupancy:.3f}", site.label,
                site.occupancy, 1.0))
        elif site.occupancy <= 0.0:
            report.findings.append(Finding(
                Level.CHECK, "occupancy-zero",
                f"occupancy {site.occupancy:.3f}", site.label,
                site.occupancy, 0.0))


def _check_charge(structure, report: HealthReport) -> None:
    charge = structure.net_charge()
    if charge is None:
        return
    total = sum(abs(s.ox or 0) * s.occupancy * (s.multiplicity or 1)
                for s in structure.sites) or 1.0
    if abs(charge) > 0.02 * total:
        report.findings.append(Finding(
            Level.CHECK, "charge-imbalance",
            f"the cell carries a net charge of {charge:+.3f} against "
            f"{total:.1f} of formal charge",
            value=charge, reference=0.0))


def _check_esds(structure, report: HealthReport) -> None:
    """Coordinate uncertainties large enough to swallow a bond length.

    An esd of 0.01 in fractional units on a 10 Å axis is 0.1 Å, which is the
    size of the differences a coordination analysis turns on.
    """
    for site in structure.sites:
        if site.frac_esd is None:
            continue
        lengths = np.array(structure.cell.lengths, float)
        absolute = np.asarray(site.frac_esd, float) * lengths
        worst = float(np.max(absolute))
        if worst > 0.10:
            report.findings.append(Finding(
                Level.CHECK, "large-esd",
                f"coordinate uncertainty reaches {worst:.3f} Å", site.label,
                worst, 0.10))
        elif worst > 0.03:
            report.findings.append(Finding(
                Level.NOTE, "esd",
                f"coordinate uncertainty reaches {worst:.3f} Å", site.label,
                worst, 0.03))


def _check_valences(results, report: HealthReport, v_bond: float) -> None:
    """Bond-valence sums far from the assumed oxidation state.

    The R0 values themselves carry about ±0.02 Å, which is roughly ±6 % in a
    sum, so a few per cent means nothing. A sum off by half its own value means
    the coordinates, the oxidation state or the parameter is wrong, and the
    check does not say which.
    """
    for r in results:
        if r.ox is None or not r.bonds:
            continue
        expected = abs(r.ox)
        if expected == 0:
            continue
        ratio = r.bvs / expected
        if ratio > 1.5 or ratio < 0.5:
            report.findings.append(Finding(
                Level.IMPOSSIBLE, "valence-far-off",
                f"bond-valence sum {r.bvs:.2f} v.u. against an assumed "
                f"{r.ox:+d} ({100 * (ratio - 1):+.0f} %)",
                r.label, r.bvs, float(expected)))
        elif ratio > 1.25 or ratio < 0.75:
            report.findings.append(Finding(
                Level.CHECK, "valence-off",
                f"bond-valence sum {r.bvs:.2f} v.u. against an assumed "
                f"{r.ox:+d} ({100 * (ratio - 1):+.0f} %)",
                r.label, r.bvs, float(expected)))


def summarise(report: HealthReport) -> str:
    """One line per finding, worst first."""
    if not report:
        return "no findings"
    order = sorted(report.findings, key=lambda f: -int(f.level))
    return "\n".join(str(f) for f in order)
