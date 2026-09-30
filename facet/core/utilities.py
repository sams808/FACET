"""Crystallographic utilities.

The things VESTA and CrystalMaker put under a Utilities or Analysis menu, plus
the bond-valence quantities that are the point of this application. All pure
functions over a :class:`~facet.core.structure.Structure`; nothing here touches
Qt and nothing here draws.

Every function reports what it computed. None of them says what it means.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import bv, elements
from .structure import Structure


# ---------------------------------------------------------------------------
# cell and lattice
# ---------------------------------------------------------------------------

def reciprocal_cell(structure: Structure) -> dict:
    """The reciprocal lattice parameters, crystallographic convention.

    Uses a* = (b x c)/V without the 2*pi factor, which is the convention
    diffraction uses and therefore the one that makes d-spacings come out
    right. The physics convention carries 2*pi; mixing them is a standard
    source of factor-of-6.28 errors.
    """
    orth = structure.cell.orth
    a, b, c = orth[:, 0], orth[:, 1], orth[:, 2]
    volume = float(np.dot(a, np.cross(b, c)))
    if abs(volume) < 1e-12:
        return {}
    a_s = np.cross(b, c) / volume
    b_s = np.cross(c, a) / volume
    c_s = np.cross(a, b) / volume

    def angle(u, v):
        return math.degrees(math.acos(
            float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)),
                          -1, 1))))

    return {
        "a*": float(np.linalg.norm(a_s)), "b*": float(np.linalg.norm(b_s)),
        "c*": float(np.linalg.norm(c_s)),
        "alpha*": angle(b_s, c_s), "beta*": angle(c_s, a_s),
        "gamma*": angle(a_s, b_s),
        "volume*": 1.0 / abs(volume),
    }


def metric_tensor(structure: Structure) -> np.ndarray:
    orth = structure.cell.orth
    return orth.T @ orth


def d_spacing(structure: Structure, h: int, k: int, l: int) -> float:
    """Interplanar spacing for one reflection, in angstrom."""
    g = np.linalg.inv(metric_tensor(structure))
    hkl = np.array([h, k, l], float)
    q2 = float(hkl @ g @ hkl)
    return float("inf") if q2 <= 0 else 1.0 / math.sqrt(q2)


def two_theta(structure: Structure, h: int, k: int, l: int,
              wavelength: float = 1.5406) -> float:
    """Bragg angle 2-theta in degrees. Default wavelength is Cu K-alpha1."""
    d = d_spacing(structure, h, k, l)
    s = wavelength / (2.0 * d)
    return float("nan") if not -1 <= s <= 1 else 2.0 * math.degrees(math.asin(s))


def reflection_list(structure: Structure, wavelength: float = 1.5406,
                    two_theta_max: float = 90.0,
                    max_index: int = 8) -> list[dict]:
    """Every reflection within a 2-theta limit, with its d-spacing.

    Geometry only -- no structure factors, so no intensities. Those arrive with
    the diffraction phase; this is the hkl/d table VESTA and CrystalMaker both
    offer, and it is what is needed to index a peak by hand.
    """
    out = []
    d_min = wavelength / (2.0 * math.sin(math.radians(two_theta_max / 2.0)))
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == k == l == 0:
                    continue
                d = d_spacing(structure, h, k, l)
                if d < d_min or not np.isfinite(d):
                    continue
                tt = two_theta(structure, h, k, l, wavelength)
                if np.isnan(tt) or tt > two_theta_max:
                    continue
                out.append({"h": h, "k": k, "l": l, "d": d, "two_theta": tt})
    out.sort(key=lambda r: -r["d"])
    return out


def density(structure: Structure) -> dict:
    """Cell mass, volume and density.

    Occupancies are included, so a partially occupied structure gives the
    density of the average cell rather than of a fully occupied one.
    """
    volume = structure.cell.volume
    mass = 0.0
    missing = []
    for site in structure.sites:
        info = elements.info(site.element)
        if info.weight is None:
            missing.append(site.element)
            continue
        mult = site.multiplicity or 1
        mass += info.weight * site.occupancy * mult
    if volume <= 0:
        return {"volume": volume, "mass": mass, "density": float("nan")}
    # 1 amu / A^3 = 1.66053906660 g/cm^3
    return {
        "volume": volume,
        "mass": mass,
        "density": mass * 1.66053906660 / volume,
        "missing_weights": missing,
    }


# ---------------------------------------------------------------------------
# geometry tables
# ---------------------------------------------------------------------------

@dataclass
class AngleRow:
    site: str
    a: str
    b: str
    angle: float
    d_a: float
    d_b: float


def bond_angles(result) -> list[AngleRow]:
    """Every ligand-central-ligand angle of one site.

    The table VESTA prints under "bond angles", limited to the bonded set so
    the list stays the size of a polyhedron rather than of a neighbourhood.
    """
    bonded = result.bonds
    out: list[AngleRow] = []
    for i in range(len(bonded)):
        for j in range(i + 1, len(bonded)):
            u = np.asarray(bonded[i].vector, float)
            v = np.asarray(bonded[j].vector, float)
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < 1e-9 or nv < 1e-9:
                continue
            ang = math.degrees(math.acos(
                float(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))
            out.append(AngleRow(result.label, bonded[i].label, bonded[j].label,
                                ang, float(nu), float(nv)))
    out.sort(key=lambda r: r.angle)
    return out


def torsion_angle(p0, p1, p2, p3) -> float:
    """Dihedral angle A-B-C-D through four points, in degrees, signed.

    Looking along B to C: the angle from the projection of B->A to the
    projection of C->D. Anti is 180 degrees and eclipsed is 0, which is the
    convention every textbook and every other program uses.

    FACET's earlier form took the angle between the two plane normals,
    ``(b0 x b1)`` and ``(b1 x b2)``, and that is the *supplement*: expanding
    the identity ``(a x b).(c x d) = (a.c)(b.d) - (a.d)(b.c)`` gives
    ``n1.n2 = -|b1|^2 (v.w)``, so the cosine came out negated and every angle
    was reported as ``sign(t) * (180 - |t|)``. Anti read as 0 and eclipsed as
    180. Checked against two independent implementations over 2000 random
    quadruples: they agree with each other to 3e-14 degrees, and the old form
    was out by up to 180.

    The perpendicular components are taken directly, which is also better
    conditioned as the four points approach collinearity than a difference of
    cross products is.

    Points rather than atom indices, because both callers have points: the
    viewport measures atoms of a drawn scene, which are periodic images as
    often as they are atoms of the structure.
    """
    a, b, c, d = (np.asarray(point, float) for point in (p0, p1, p2, p3))
    axis = c - b
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        return 0.0
    axis = axis / length
    v = (a - b) - np.dot(a - b, axis) * axis        # B->A, across the bond
    w = (d - c) - np.dot(d - c, axis) * axis        # C->D, across the bond
    return float(np.degrees(np.arctan2(float(np.cross(axis, v) @ w),
                                       float(v @ w))))


def torsions(structure, a: int, b: int, c: int, d: int) -> float:
    """Dihedral angle through four atom indices, in degrees."""
    return torsion_angle(*(structure.atoms[i].cart for i in (a, b, c, d)))


def radial_shells(structure: Structure, site_index: int, rmax: float = 6.0,
                  tolerance: float = 0.05) -> list[dict]:
    """Neighbours grouped into shells by distance, with degeneracies.

    The input an EXAFS fit wants: how many of what, at what radius. Contacts
    within `tolerance` of each other are one shell.
    """
    from .neighbors import NeighborFinder

    finder = NeighborFinder(structure, rmax=rmax)
    contacts = finder.contacts_for_site(site_index)
    if len(contacts) == 0:
        return []

    shells: list[dict] = []
    for i in range(len(contacts)):
        d = float(contacts.distance[i])
        element = contacts.elements[i]
        for s in shells:
            if s["element"] == element and abs(s["distance"] - d) <= tolerance:
                s["count"] += 1
                s["_sum"] += d
                s["distance"] = s["_sum"] / s["count"]
                break
        else:
            shells.append({"element": element, "distance": d, "count": 1,
                           "_sum": d})
    for s in shells:
        s.pop("_sum", None)
    shells.sort(key=lambda s: s["distance"])
    return shells


# ---------------------------------------------------------------------------
# bond valence, beyond the per-site sum
# ---------------------------------------------------------------------------

def global_instability_index(results) -> dict:
    """GII over a set of site results, with the site contributing most.

    GII is the root-mean-square difference between each cation's bond-valence
    sum and its formal charge, over the sites of a structure. Reported as a
    number with its worst contributor; no threshold is applied to it here.
    """
    discrepancies = []
    worst = None
    for r in results:
        if r.valence_discrepancy is None or np.isnan(r.valence_discrepancy):
            continue
        discrepancies.append(r.valence_discrepancy)
        if worst is None or abs(r.valence_discrepancy) > abs(worst[1]):
            worst = (r.label, r.valence_discrepancy)
    if not discrepancies:
        return {"gii": float("nan"), "n_sites": 0, "worst": None}
    return {
        "gii": float(np.sqrt(np.mean(np.square(discrepancies)))),
        "n_sites": len(discrepancies),
        "worst_site": worst[0] if worst else None,
        "worst_discrepancy": worst[1] if worst else None,
    }


def bond_strain_index(results) -> float:
    """BSI: the r.m.s. difference between observed bond valences and the
    valences a strain-free structure would distribute.

    Approximated per site by the equal-share expectation, BVS/CN. The full
    definition solves the network equations; the per-site form captures the
    same thing for a first look and needs no global solve.
    """
    residuals = []
    for r in results:
        bonded = r.bonds
        if not bonded:
            continue
        expected = r.bvs / len(bonded)
        residuals += [c.valence - expected for c in bonded if c.valence]
    if not residuals:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(residuals))))


def valence_map_point(structure: Structure, position, element: str, ox: int,
                      params: bv.ParameterSet | None = None,
                      rmax: float = 6.0, finder=None) -> float:
    """Bond-valence sum a probe cation would have at an arbitrary point.

    The quantity a bond-valence map is made of: place a hypothetical cation
    somewhere and sum its valence to the surrounding anions. Where that sum
    equals the formal charge, the site is a plausible one for that ion.
    """
    from .neighbors import NeighborFinder

    params = params or bv.DEFAULT
    if finder is None:
        finder = NeighborFinder(structure, rmax=rmax)
    position = np.asarray(position, float)

    total = 0.0
    idx = finder._tree.query_ball_point(position, rmax)
    for k in idx:
        atom_index = int(finder._atom[k])
        atom = structure.atoms[atom_index]
        if not structure.sites[atom.site_index].is_anion:
            continue
        d = float(np.linalg.norm(finder._pos[k] - position))
        if d < 0.5:
            continue
        p = params.get(element, ox, atom.element)
        if p is None:
            continue
        total += float(p.valence(d)) * atom.occupancy
    return total


def valence_map(structure: Structure, element: str, ox: int,
                resolution: float = 0.35,
                params: bv.ParameterSet | None = None,
                rmax: float = 6.0):
    """A 3D grid of bond-valence sums over the unit cell.

    Returns ``(grid, shape, step)``. This is the basis of a bond-valence map
    and, with an energy conversion, of a bond-valence energy landscape. Sampled
    on a grid in fractional coordinates so it tiles correctly whatever the cell
    shape.

    Cost grows as the cube of the inverse resolution. A 0.35 A step over a
    300 A^3 cell is roughly 8000 points and takes a second or two; going to
    0.1 A is forty times that, so the caller chooses.
    """
    from .neighbors import NeighborFinder

    params = params or bv.DEFAULT
    finder = NeighborFinder(structure, rmax=rmax)
    cell = structure.cell
    steps = [max(2, int(round(length / resolution)))
             for length in (cell.a, cell.b, cell.c)]

    grid = np.zeros(steps, float)
    for i in range(steps[0]):
        for j in range(steps[1]):
            for k in range(steps[2]):
                frac = np.array([i / steps[0], j / steps[1], k / steps[2]])
                grid[i, j, k] = valence_map_point(
                    structure, cell.to_cartesian(frac), element, ox,
                    params, rmax, finder)
    return grid, tuple(steps), resolution


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def composition_summary(structure: Structure) -> dict:
    """Cell contents, formula weight, and the charge balance."""
    counts = structure.composition()
    total = sum(counts.values()) or 1.0
    weight = 0.0
    for element, n in counts.items():
        info = elements.info(element)
        if info.weight:
            weight += info.weight * n
    charge = structure.net_charge()
    return {
        "counts": counts,
        "atoms_per_cell": total,
        "formula_weight": weight,
        "net_charge": charge,
        "balanced": (charge is not None and abs(charge) < 1e-6),
        "atomic_percent": {e: 100.0 * n / total for e, n in counts.items()},
        "weight_percent": ({e: 100.0 * elements.info(e).weight * n / weight
                            for e, n in counts.items()
                            if elements.info(e).weight} if weight else {}),
    }


def polyhedral_connectivity(structure: Structure, results,
                            v_bond: float) -> list[dict]:
    """How the coordination polyhedra join: corner, edge or face sharing.

    Two polyhedra share a corner if they have one ligand in common, an edge if
    two, a face if three or more. This is the connectivity a framework is
    usually described by, and it is a count of shared ligands rather than an
    interpretation of one.
    """
    ligand_sets: dict[int, set] = {}
    for r in results:
        atoms = structure.atoms_of_site(r.site_index)
        if not atoms:
            continue
        origin = structure.atoms[atoms[0]].cart
        keys = set()
        for c in r.contacts:
            if c.valence is None or c.valence <= v_bond:
                continue
            p = origin + c.vector
            keys.add((round(float(p[0]), 2), round(float(p[1]), 2),
                      round(float(p[2]), 2)))
        ligand_sets[r.site_index] = keys

    names = {r.site_index: r.label for r in results}
    out = []
    indices = list(ligand_sets)
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            a, b = indices[i], indices[j]
            shared = len(ligand_sets[a] & ligand_sets[b])
            if shared == 0:
                continue
            kind = {1: "corner", 2: "edge"}.get(shared, "face")
            out.append({"site_a": names[a], "site_b": names[b],
                        "shared_ligands": shared, "sharing": kind})
    return out
