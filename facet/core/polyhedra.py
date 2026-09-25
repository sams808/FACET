"""Geometry of a coordination polyhedron.

Ported from the validated engine used for the 2026 bismuth survey
(``XRD/cif/Bi/study/bi_core.py``), where every function here was checked against
analytic shapes and the whole pipeline was reproduced by an independent
implementation sharing no code. The element-specific parts have been removed;
the geometry never had any.
"""
from __future__ import annotations

import math

import numpy as np

_FIB_CACHE: dict[int, np.ndarray] = {}


def effective_cn(distances, weights=None) -> tuple[float, float]:
    """Hoppe's effective coordination number, and its weighted mean distance.

    ECoN weights each contact by ``exp(1 - (d/d_av)**6)`` with ``d_av`` solved
    self-consistently, so a contact contributes almost 1 while it is close to
    the shortest bond and fades smoothly out rather than being cut off. It gives
    a non-integer coordination number, which is an honest answer for a site with
    no gap in its distance distribution.
    """
    d = np.asarray(distances, float)
    if d.size == 0:
        return 0.0, float("nan")
    w0 = np.ones_like(d) if weights is None else np.asarray(weights, float)
    d_av = float(d.min())
    for _ in range(300):
        w = w0 * np.exp(1.0 - (d / d_av) ** 6)
        total = float(w.sum())
        if total <= 0:
            break
        new = float(np.sum(d * w) / total)
        if abs(new - d_av) < 1e-11:
            d_av = new
            break
        d_av = new
    w = w0 * np.exp(1.0 - (d / d_av) ** 6)
    return float(w.sum()), d_av


def gap_split(distances) -> tuple[int, float]:
    """Split a sorted contact list at its largest relative step.

    Returns ``(n_primary, ratio)``. This is Brunner's maximum-gap rule: the
    cutoff is placed where ``d[k+1]/d[k]`` is largest, leaving at least two
    contacts inside. It is the most common informal way a coordination number is
    decided, and FACET reports it so that it can be compared against the others
    rather than trusted alone -- on a lone-pair cation with a smooth distance
    distribution the largest step is often not significant.
    """
    d = np.asarray(distances, float)
    if d.size < 3:
        return int(d.size), 1.0
    ratio = d[1:] / d[:-1]
    lo = 1
    k = int(np.argmax(ratio[lo:])) + lo
    return k + 1, float(ratio[k])


def _fibonacci_sphere(n: int = 12000) -> np.ndarray:
    cached = _FIB_CACHE.get(n)
    if cached is not None:
        return cached
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    pts = np.stack([np.cos(theta) * np.sin(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(phi)], axis=1)
    _FIB_CACHE[n] = pts
    return pts


def void_cone(vectors, n_sample: int = 12000) -> tuple[float, np.ndarray]:
    """Half-angle of the largest ligand-free cone at the central atom.

    ``max over directions u of min over ligands i of angle(u, r_i)``, in degrees,
    with the axis of that cone.

    Reference values: regular octahedron 54.74 deg, regular tetrahedron 70.53,
    trigonal planar 90, a single bond 180. A large void cone is the geometric
    signature of a stereochemically active lone pair, and its axis is where the
    lone pair points.

    Deterministic: the local refinement uses a fixed seed, so repeated calls on
    the same site return the same number.
    """
    u = np.asarray(vectors, float)
    if u.shape[0] == 0:
        return 180.0, np.array([0.0, 0.0, 1.0])
    norms = np.linalg.norm(u, axis=1)
    norms[norms == 0] = 1.0
    u = u / norms[:, None]

    sphere = _fibonacci_sphere(n_sample)
    closest = (sphere @ u.T).max(axis=1)        # cosine to the nearest ligand
    best = sphere[int(np.argmin(closest))]

    rng = np.random.default_rng(0)
    for scale in (0.06, 0.02, 0.006, 0.002):
        current = float((best @ u.T).max())
        pert = best + scale * rng.normal(size=(600, 3))
        pert /= np.linalg.norm(pert, axis=1)[:, None]
        c = (pert @ u.T).max(axis=1)
        j = int(np.argmin(c))
        if c[j] < current:
            best = pert[j]

    angle = math.degrees(math.acos(float(np.clip((best @ u.T).max(), -1, 1))))
    return angle, best


def _trans_pairs(vectors, distances) -> tuple[tuple, ...]:
    """The three trans pairs of a six-coordinate polyhedron.

    Taking the three largest of the fifteen inter-ligand angles is *not*
    equivalent and is wrong on a distorted polyhedron, where those three can
    share a vertex. Six vertices admit only fifteen perfect matchings, so they
    are enumerated exhaustively and the one maximising the total angle is taken.
    """
    ang = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            c = float(np.dot(vectors[i], vectors[j]) / (distances[i] * distances[j]))
            ang[i, j] = ang[j, i] = math.degrees(math.acos(np.clip(c, -1, 1)))

    best, best_sum = None, -1.0
    rest = [1, 2, 3, 4, 5]
    for a in range(5):
        p1 = (0, rest[a])
        rem = [x for k, x in enumerate(rest) if k != a]
        for b in (1, 2, 3):
            p2 = (rem[0], rem[b])
            p3 = tuple(x for k, x in enumerate(rem) if k not in (0, b))
            total = ang[p1] + ang[p2] + ang[p3]
            if total > best_sum:
                best_sum, best = total, (p1, p2, p3)
    return best, ang


def shape(vectors) -> dict:
    """Distance statistics, hull volume and the polyhedral distortion indices.

    The octahedral indices (bond-angle variance, quadratic elongation) are only
    computed for six contacts, and are descriptive even then -- a large angle
    variance is normal for a stereoactive lone pair and is not a fault.
    """
    v = np.asarray(vectors, float)
    n = len(v)
    out: dict = {"n": n, "d_min": None, "d_max": None, "d_mean": None,
                 "spread": None, "baur": None, "volume": None,
                 "angle_variance": None, "quadratic_elongation": None,
                 "eccentricity": None}
    if n == 0:
        return out

    d = np.linalg.norm(v, axis=1)
    d_mean = float(d.mean())
    out.update(d_min=float(d.min()), d_max=float(d.max()), d_mean=d_mean,
               spread=float(d.max() - d.min()),
               baur=float(np.mean(np.abs(d - d_mean)) / d_mean) if d_mean else None)

    # how far the centroid of the ligands sits from the central atom
    out["eccentricity"] = float(np.linalg.norm(v.mean(axis=0)))

    if n >= 4:
        try:
            from scipy.spatial import ConvexHull

            out["volume"] = float(ConvexHull(v).volume)
        except Exception:
            pass

    if n == 6:
        trans, ang = _trans_pairs(v, d)
        tr = set(trans)
        cis = [ang[i, j] for i in range(6) for j in range(i + 1, 6)
               if (i, j) not in tr]
        out["angle_variance"] = float(np.sum((np.array(cis) - 90.0) ** 2) / 11.0)
        if out["volume"]:
            d0 = (3.0 * out["volume"] / 4.0) ** (1.0 / 3.0)
            out["quadratic_elongation"] = float(np.mean((d / d0) ** 2))
    return out


def valence_fraction_in_shortest(valences, n: int = 3) -> float:
    """Fraction of the bond-valence sum carried by the n shortest bonds.

    For six coordination a regular octahedron carries exactly 0.500 of its
    valence in its three shortest bonds, so ``f3 - 0.5`` is an absolute measure
    of shell splitting -- zero for a regular octahedron, whatever its size.

    Paired with phi, which measures direction, it separates three things that a
    bond-length spread alone confuses: a 3+3 environment with the long bonds to
    one side (hemidirected, high phi), one with them opposite (the tetradymite
    motif, large splitting but phi near zero), and a regular one.
    """
    v = np.sort(np.asarray(valences, float))[::-1]
    total = float(v.sum())
    if total <= 0 or v.size == 0:
        return float("nan")
    return float(v[:n].sum() / total)
