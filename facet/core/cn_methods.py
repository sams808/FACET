"""Every defensible way to count a coordination number.

The application's argument is that a coordination number is produced by a rule,
not measured, so it has to be able to apply the rules people actually use and
show what each one gives. This module holds them.

Each method returns a :class:`CNResult` carrying the number, the parameters it
took, and the neighbours it counted. None of them is presented as correct; they
are presented together.

Native implementations are used where the method is short enough that a
dependency would cost more than it saves. The ones that are not -- CrystalNN's
weighted Voronoi, ChemEnv's continuous symmetry measures -- are taken from
pymatgen, imported lazily and degrading to "unavailable" rather than failing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import elements


@dataclass
class CNResult:
    """One coordination number, and how it was arrived at."""

    method: str
    cn: float
    integer: bool = True
    parameters: dict = field(default_factory=dict)
    counted: list[int] = field(default_factory=list)   # indices into contacts
    weights: list[float] = field(default_factory=list)
    note: str = ""
    available: bool = True

    @property
    def display(self) -> str:
        if not self.available:
            return "—"
        return f"{self.cn:.0f}" if self.integer else f"{self.cn:.2f}"


# ---------------------------------------------------------------------------
# distance-based rules
# ---------------------------------------------------------------------------

def hard_cutoff(distances, cutoff: float) -> CNResult:
    """Everything within a fixed distance. The rule most papers use."""
    d = np.asarray(distances, float)
    counted = [i for i, x in enumerate(d) if x <= cutoff]
    return CNResult("Distance cutoff", float(len(counted)),
                    parameters={"cutoff": cutoff}, counted=counted)


def percent_of_shortest(distances, percent: float = 130.0) -> CNResult:
    """Everything within a percentage of the shortest contact.

    A scale-free version of a hard cutoff, and the rule behind most informal
    statements of the form "plus two longer contacts".
    """
    d = np.asarray(distances, float)
    if d.size == 0:
        return CNResult("Within % of shortest", 0.0,
                        parameters={"percent": percent})
    limit = d.min() * percent / 100.0
    counted = [i for i, x in enumerate(d) if x <= limit]
    return CNResult("Within % of shortest", float(len(counted)),
                    parameters={"percent": percent, "limit": limit},
                    counted=counted)


def sum_of_radii(distances, elements_list, central: str, ox: int | None = None,
                 tolerance: float = 1.20) -> CNResult:
    """Contacts closer than a tolerance times the sum of covalent radii."""
    d = np.asarray(distances, float)
    r_central = elements.info(central).covalent_radius or 1.0
    counted = []
    for i, (x, element) in enumerate(zip(d, elements_list)):
        r = elements.info(element).covalent_radius or 1.0
        if x <= tolerance * (r_central + r):
            counted.append(i)
    return CNResult("Sum of covalent radii", float(len(counted)),
                    parameters={"tolerance": tolerance}, counted=counted)


def maximum_gap(distances) -> CNResult:
    """Brunner's maximum-gap rule: cut at the largest relative step."""
    from .polyhedra import gap_split

    d = np.asarray(distances, float)
    n, ratio = gap_split(d)
    return CNResult("Maximum gap (Brunner)", float(n),
                    parameters={"ratio": ratio},
                    counted=list(range(n)),
                    note=f"largest step d(k+1)/d(k) = {ratio:.3f}")


def reciprocal_gap(distances) -> CNResult:
    """Brunner's reciprocal-distance variant.

    The gap is taken in 1/d rather than in d, which weights the near contacts
    more heavily and therefore rarely cuts far out in a long tail.
    """
    d = np.asarray(distances, float)
    if d.size < 3:
        return CNResult("Maximum gap (1/d)", float(d.size),
                        counted=list(range(d.size)))
    inv = 1.0 / d
    steps = inv[:-1] - inv[1:]
    k = int(np.argmax(steps[1:])) + 1
    return CNResult("Maximum gap (1/d)", float(k + 1),
                    parameters={"step": float(steps[k])},
                    counted=list(range(k + 1)))


# ---------------------------------------------------------------------------
# weighted, non-integer rules
# ---------------------------------------------------------------------------

def econ(distances, occupancies=None) -> CNResult:
    """Hoppe's effective coordination number."""
    from .polyhedra import effective_cn

    cn, d_av = effective_cn(distances, occupancies)
    return CNResult("ECoN (Hoppe)", float(cn), integer=False,
                    parameters={"d_av": d_av},
                    note=f"weighted mean distance {d_av:.4f} Å")


def chardi(distances, occupancies=None, charge: float = 1.0,
           iterations: int = 60) -> CNResult:
    """CHARDI: the charge-distribution coordination number.

    The cation's formal charge is shared among its ligands in proportion to
    ``exp(1 - (d/d_av)**6)``, with ``d_av`` solved self-consistently as in
    ECoN; the coordination number is the sum of the resulting weights
    normalised by the largest. Like ECoN it is non-integer, and like ECoN it
    does not need a cutoff at all -- which is why both are worth showing
    beside a rule that does.
    """
    d = np.asarray(distances, float)
    if d.size == 0:
        return CNResult("CHARDI", 0.0, integer=False)
    occ = (np.ones_like(d) if occupancies is None
           else np.asarray(occupancies, float))

    d_av = float(d.min())
    for _ in range(iterations):
        w = occ * np.exp(1.0 - (d / d_av) ** 6)
        total = float(w.sum())
        if total <= 0:
            break
        new = float(np.sum(d * w) / total)
        if abs(new - d_av) < 1e-12:
            d_av = new
            break
        d_av = new

    w = occ * np.exp(1.0 - (d / d_av) ** 6)
    if w.max() <= 0:
        return CNResult("CHARDI", 0.0, integer=False)
    normalised = w / w.max()
    return CNResult("CHARDI", float(normalised.sum()), integer=False,
                    parameters={"d_av": d_av, "charge": charge},
                    weights=[float(x) for x in normalised])


def valence_weighted(valences) -> CNResult:
    """Coordination number weighted by bond valence, normalised by the
    strongest bond.

    A continuous counterpart to the valence threshold: a contact carrying half
    the valence of the shortest bond counts as half a neighbour, so nothing
    turns on where a line is drawn.
    """
    v = np.asarray([x for x in valences if x is not None], float)
    if v.size == 0 or v.max() <= 0:
        return CNResult("Valence-weighted", 0.0, integer=False)
    return CNResult("Valence-weighted", float((v / v.max()).sum()),
                    integer=False,
                    weights=[float(x) for x in v / v.max()])


def valence_threshold(valences, v_bond: float) -> CNResult:
    """FACET's own default: everything above a bond-valence threshold."""
    counted = [i for i, x in enumerate(valences)
               if x is not None and x > v_bond]
    return CNResult("Bond valence", float(len(counted)),
                    parameters={"v_bond": v_bond}, counted=counted)


# ---------------------------------------------------------------------------
# Voronoi
# ---------------------------------------------------------------------------

def _solid_angle(centre: np.ndarray, polygon: np.ndarray) -> float:
    """Solid angle a polygon subtends at a point, in steradians.

    Fan-decomposed into triangles, each by Van Oosterom and Strackee's formula
    for the solid angle of a tetrahedron. Exact for a planar polygon, which a
    Voronoi facet is.
    """
    v = polygon - centre
    norms = np.linalg.norm(v, axis=1)
    if np.any(norms < 1e-12):
        return 0.0
    v = v / norms[:, None]
    total = 0.0
    for i in range(1, len(v) - 1):
        a, b, c = v[0], v[i], v[i + 1]
        numerator = abs(float(np.dot(a, np.cross(b, c))))
        denominator = (1.0 + float(np.dot(a, b)) + float(np.dot(b, c))
                       + float(np.dot(a, c)))
        total += 2.0 * math.atan2(numerator, denominator)
    return abs(total)


def voronoi(vectors, min_fraction: float = 0.20) -> CNResult:
    """Solid-angle-weighted Voronoi coordination number.

    Every neighbour that shares a Voronoi facet with the central atom is a
    neighbour; each is weighted by the solid angle that facet subtends,
    normalised by the largest. A facet contributing less than `min_fraction` of
    the largest is dropped, which is the usual way this rule is closed -- the
    Voronoi construction alone has no cutoff and would count every atom that
    happens to share a face, however slight.

    The returned number is the sum of the retained weights, so it is
    non-integer, as the method is.
    """
    v = np.asarray(vectors, float)
    if len(v) < 4:
        return CNResult("Voronoi (solid angle)", float(len(v)), integer=False,
                        note="too few neighbours for a Voronoi cell")
    try:
        from scipy.spatial import Voronoi
    except ImportError:                                   # pragma: no cover
        return CNResult("Voronoi (solid angle)", 0.0, available=False)

    points = np.vstack([np.zeros(3), v])
    try:
        diagram = Voronoi(points)
    except Exception:
        return CNResult("Voronoi (solid angle)", 0.0, available=False,
                        note="the Voronoi construction failed")

    angles: dict[int, float] = {}
    for (p, q), ridge in zip(diagram.ridge_points, diagram.ridge_vertices):
        if 0 not in (p, q) or -1 in ridge or len(ridge) < 3:
            continue
        neighbour = q if p == 0 else p
        polygon = diagram.vertices[ridge]
        angles[neighbour - 1] = _solid_angle(np.zeros(3), polygon)

    if not angles:
        return CNResult("Voronoi (solid angle)", 0.0, integer=False,
                        note="no bounded Voronoi facets")

    largest = max(angles.values())
    if largest <= 0:
        return CNResult("Voronoi (solid angle)", 0.0, integer=False)
    kept = {i: a / largest for i, a in angles.items()
            if a / largest >= min_fraction}
    return CNResult("Voronoi (solid angle)", float(sum(kept.values())),
                    integer=False,
                    parameters={"min_fraction": min_fraction},
                    counted=sorted(kept),
                    weights=[kept[i] for i in sorted(kept)],
                    note=f"{len(kept)} facets above {min_fraction:.0%} of the "
                         f"largest")


# ---------------------------------------------------------------------------
# pymatgen, lazily
# ---------------------------------------------------------------------------

_PYMATGEN_METHODS = {
    "CrystalNN": ("CrystalNN", {}),
    "VoronoiNN": ("VoronoiNN", {"tol": 0.2}),
    "MinimumDistanceNN": ("MinimumDistanceNN", {"tol": 0.3}),
    "JmolNN": ("JmolNN", {}),
    "MinimumOKeeffeNN": ("MinimumOKeeffeNN", {}),
    "EconNN": ("EconNN", {}),
    "BrunnerNN_real": ("BrunnerNN_real", {}),
    "BrunnerNN_reciprocal": ("BrunnerNN_reciprocal", {}),
}


def pymatgen_methods(structure, site_index: int,
                     names=None) -> list[CNResult]:
    """Coordination numbers from pymatgen's near-neighbour strategies.

    Several of these are genuinely different rules rather than variants, and
    reimplementing CrystalNN's weighted Voronoi correctly would be a project.
    pymatgen is imported here and nowhere else in the engine; if it is absent
    every entry comes back unavailable rather than raising, because the built
    application ships without it.
    """
    names = list(names or _PYMATGEN_METHODS)
    try:
        from pymatgen.analysis import local_env
        from pymatgen.core import Lattice
        from pymatgen.core import Structure as PmgStructure
    except Exception:
        return [CNResult(n, 0.0, available=False,
                         note="pymatgen is not installed") for n in names]

    try:
        lattice = Lattice(structure.cell.orth.T)
        species = [a.element for a in structure.atoms]
        coords = [a.frac for a in structure.atoms]
        pmg = PmgStructure(lattice, species, coords)
        atoms = structure.atoms_of_site(site_index)
        if not atoms:
            raise ValueError("site generated no atoms")
        target = atoms[0]
    except Exception as exc:
        return [CNResult(n, 0.0, available=False,
                         note=f"could not build the structure: {exc}")
                for n in names]

    out = []
    for name in names:
        class_name, kwargs = _PYMATGEN_METHODS.get(name, (name, {}))
        try:
            strategy = getattr(local_env, class_name)(**kwargs)
            cn = strategy.get_cn(pmg, target, use_weights=False)
            out.append(CNResult(f"{name} (pymatgen)", float(cn),
                                parameters=kwargs))
        except Exception as exc:
            out.append(CNResult(f"{name} (pymatgen)", 0.0, available=False,
                                note=str(exc)[:90]))
    return out


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

def all_methods(result, structure=None, include_pymatgen: bool = False,
                hard_cutoff_distance: float | None = None) -> list[CNResult]:
    """Every method FACET can apply to one site, in a stable order.

    `result` is a :class:`~facet.core.coordination.SiteResult`. The pymatgen
    strategies are off by default because they rebuild the structure and cost
    far more than the native rules.
    """
    contacts = [c for c in result.contacts if c.has_valence]
    distances = [c.distance for c in contacts]
    valences = [c.valence for c in contacts]
    occupancies = [c.occupancy for c in contacts]
    vectors = np.array([c.vector for c in contacts]) if contacts \
        else np.zeros((0, 3))
    ligands = [c.element for c in contacts]

    param = next((c.param for c in contacts if c.param), None)
    if hard_cutoff_distance is None:
        hard_cutoff_distance = (param.distance_for(result.v_bond)
                                if param else 3.0)

    methods = [
        valence_threshold(valences, result.v_bond),
        hard_cutoff(distances, hard_cutoff_distance),
        percent_of_shortest(distances, 130.0),
        sum_of_radii(distances, ligands, result.element, result.ox),
        maximum_gap(distances),
        reciprocal_gap(distances),
        econ(distances, occupancies),
        chardi(distances, occupancies,
               charge=float(result.ox) if result.ox else 1.0),
        valence_weighted(valences),
        voronoi(vectors),
    ]
    if include_pymatgen and structure is not None:
        methods += pymatgen_methods(structure, result.site_index)
    return methods


def spread(methods: list[CNResult]) -> dict:
    """How far the methods disagree, as a number rather than an impression."""
    values = [m.cn for m in methods if m.available]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "range": float(max(values) - min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "integer_values": sorted({int(round(v)) for v in values}),
    }
