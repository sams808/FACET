"""Environments: the atoms of one species grouped by what is around them.

A Q³ fraction says how many Si are Q³; it does not say which Q³
environments are alike. This module describes each atom's environment at
the threshold in force -- the sorted first-shell distances, the angles at
the centre, the composition of the second shell -- and groups the atoms of
one species whose descriptions agree within a stated tolerance. A group
such as "Si Q³ environment 1" has a population, members to highlight on
the frame, and a representative (the member nearest the group's mean) to
export as a cluster, so that a shift computed on the representative stands
for the class.

No clustering verdict is made: the tolerances are the user's, and the
counts are reported.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import bv
from .md_environment import neighbour_shells
from .md_model import Frame

TETRAHEDRAL_DEG = 109.4712206


@dataclass
class Environment:
    """One atom's environment at the threshold."""

    row: int
    element: str
    cn: int
    qn: int | None                  # None when the atom is not a former
    distances_ang: np.ndarray       # first-shell distances, sorted
    angles_deg: np.ndarray          # angles at the centre, sorted
    first_shell: tuple[tuple[str, int], ...]    # ((element, count), ...)
    second_shell: tuple[tuple[str, int], ...]   # beyond the first, by element

    @property
    def signature(self) -> tuple:
        """What has to agree exactly for two environments to be compared:
        the species, the coordination number, Qn and both shells'
        compositions."""
        return (self.element, self.cn, self.qn, self.first_shell,
                self.second_shell)


@dataclass
class Group:
    """Environments alike within the tolerance."""

    index: int
    label: str
    signature: tuple
    members: list[int]
    representative: int
    mean_distances_ang: np.ndarray
    mean_angles_deg: np.ndarray
    spread_distance_ang: float
    spread_angle_deg: float
    scalars: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.members)


def _bonds_of(bonds, n_atoms: int):
    cation = np.asarray(bonds.cation, np.int64)
    anion = np.asarray(bonds.anion, np.int64)
    vec = np.asarray(bonds.vec_ang, np.float64)
    centre = np.concatenate([cation, anion])
    other = np.concatenate([anion, cation])
    arms = np.concatenate([vec, -vec])
    order = np.argsort(centre, kind="stable")
    centre, other, arms = centre[order], other[order], arms[order]
    starts = np.searchsorted(centre, np.arange(n_atoms + 1))
    return other, arms, starts


def environments(frame: Frame, bonds, *, element: str | None = None,
                 formers: Iterable[str] = (), qn: np.ndarray | None = None,
                 rows: Sequence[int] | None = None) -> list[Environment]:
    """The environment of every atom of ``element`` (or of ``rows``), from
    the bonds the threshold counts (``bulk.bonds_at``)."""
    symbols = np.asarray(frame.elements)
    n = len(symbols)
    other, arms, starts = _bonds_of(bonds, n)
    former_set = frozenset(formers)
    if rows is None:
        rows = np.flatnonzero(symbols == element) if element else np.arange(n)
    out: list[Environment] = []
    for row in (int(r) for r in rows):
        lo, hi = starts[row], starts[row + 1]
        neighbours = other[lo:hi]
        v = arms[lo:hi]
        norms = np.linalg.norm(v, axis=1)
        d = np.sort(norms)
        angles = np.zeros(0)
        if len(v) >= 2:
            ok = norms > 0
            unit = v[ok] / norms[ok][:, None]
            cos = np.clip(unit @ unit.T, -1.0, 1.0)
            iu = np.triu_indices(len(unit), 1)
            angles = np.sort(np.degrees(np.arccos(cos[iu])))
        first = Counter(str(s) for s in symbols[neighbours])
        second = Counter()
        for nb in neighbours:
            lo2, hi2 = starts[nb], starts[nb + 1]
            for nn in other[lo2:hi2]:
                if nn != row:
                    second[str(symbols[nn])] += 1
        q = None
        if qn is not None and symbols[row] in former_set:
            value = float(qn[row])
            q = int(round(value)) if np.isfinite(value) else None
        out.append(Environment(
            row=row, element=str(symbols[row]), cn=int(len(neighbours)), qn=q,
            distances_ang=d, angles_deg=angles,
            first_shell=tuple(sorted(first.items())),
            second_shell=tuple(sorted(second.items()))))
    return out


def group_environments(envs: Sequence[Environment], *,
                       tol_distance_ang: float = 0.05,
                       tol_angle_deg: float = 5.0,
                       same_second_shell: bool = True) -> list[Group]:
    """Group environments whose signatures agree and whose sorted distances
    and angles agree within the tolerances (the largest absolute difference
    against the group's running mean).

    Greedy, in the order given: an environment joins the first group it
    fits, else starts one. The order can move a borderline member between
    neighbouring groups; the tolerance is the user's statement of what
    'alike' means, and the counts are reported as they come out.
    """
    groups: list[Group] = []
    sums: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for env in envs:
        signature = env.signature if same_second_shell else env.signature[:4]
        home = None
        for g in groups:
            if g.signature != signature:
                continue
            if len(g.mean_distances_ang) != len(env.distances_ang) \
                    or len(g.mean_angles_deg) != len(env.angles_deg):
                continue
            dd = np.abs(g.mean_distances_ang - env.distances_ang).max() \
                if len(env.distances_ang) else 0.0
            da = np.abs(g.mean_angles_deg - env.angles_deg).max() \
                if len(env.angles_deg) else 0.0
            if dd <= tol_distance_ang and da <= tol_angle_deg:
                home = g
                break
        if home is None:
            home = Group(index=len(groups), label="", signature=signature,
                         members=[], representative=env.row,
                         mean_distances_ang=env.distances_ang.copy(),
                         mean_angles_deg=env.angles_deg.copy(),
                         spread_distance_ang=0.0, spread_angle_deg=0.0)
            groups.append(home)
            sums[home.index] = (env.distances_ang.copy(), env.angles_deg.copy())
        home.members.append(env.row)
        sd, sa = sums[home.index]
        if home.n > 1:
            sd = sd + env.distances_ang
            sa = sa + env.angles_deg
            sums[home.index] = (sd, sa)
            home.mean_distances_ang = sd / home.n
            home.mean_angles_deg = sa / home.n
    by_env = {e.row: e for e in envs}
    # labels per species and Qn, numbered by population; representatives
    counters: dict[tuple, int] = {}
    for g in sorted(groups, key=lambda g: -g.n):
        head = g.signature[0]
        q = g.signature[2]
        key = (head, q)
        counters[key] = counters.get(key, 0) + 1
        qtext = f" Q{q}" if q is not None else ""
        g.label = f"{head}{qtext} environment {counters[key]}"
        best, best_score = g.representative, float("inf")
        dists, angs = [], []
        for row in g.members:
            e = by_env[row]
            dd = np.abs(e.distances_ang - g.mean_distances_ang).max() \
                if len(e.distances_ang) else 0.0
            da = np.abs(e.angles_deg - g.mean_angles_deg).max() \
                if len(e.angles_deg) else 0.0
            dists.append(dd)
            angs.append(da)
            score = dd / max(tol_distance_ang, 1e-9) + da / max(tol_angle_deg, 1e-9)
            if score < best_score:
                best, best_score = row, score
        g.representative = best
        g.spread_distance_ang = float(max(dists)) if dists else 0.0
        g.spread_angle_deg = float(max(angs)) if angs else 0.0
    groups.sort(key=lambda g: (g.signature[0], -(g.signature[2] or -1), -g.n))
    for i, g in enumerate(groups):
        g.index = i
    return groups


def group_scalars(groups: Sequence[Group],
                  per_atom: Mapping[str, np.ndarray]) -> None:
    """Average any per-atom arrays (a bond-valence sum, a landscape value,
    a free-volume radius) over each group's members, into ``scalars``."""
    for g in groups:
        for name, values in per_atom.items():
            v = np.asarray(values, float)[g.members]
            v = v[np.isfinite(v)]
            g.scalars[name] = float(v.mean()) if len(v) else float("nan")


def nearest_void_radius(frame: Frame, centres_ang: np.ndarray,
                        radii_ang: np.ndarray) -> np.ndarray:
    """Per atom, the radius of the nearest empty sphere: the free volume
    about it, read from spheres already found (``md_order.empty_spheres``).
    Atoms are wrapped into the box; the spheres are taken as given."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return np.full(len(frame.elements), np.nan)
    if centres_ang is None or len(centres_ang) == 0:
        return np.full(len(frame.elements), np.nan)
    frac = np.asarray(frame.frac, float) % 1.0
    cart = frac @ np.asarray(frame.box_ang, float) + np.asarray(
        getattr(frame, "origin_ang", np.zeros(3)), float)[None, :]
    _d, j = cKDTree(np.asarray(centres_ang, float)).query(cart)
    return np.asarray(radii_ang, float)[j]


def group_rows(groups: Sequence[Group]) -> list[dict]:
    """One row per group, for a table or a CSV."""
    rows = []
    for g in groups:
        row = {
            "group": g.label,
            "element": g.signature[0],
            "cn": g.signature[1],
            "qn": g.signature[2],
            "n": g.n,
            "representative_row": g.representative,
            "first_shell": " ".join(f"{e}{c}" for e, c in g.signature[3]),
            "second_shell": " ".join(f"{e}{c}" for e, c in g.signature[4])
            if len(g.signature) > 4 else "",
            "mean_distances_ang": " ".join(f"{d:.3f}" for d in g.mean_distances_ang),
            "mean_angles_deg": " ".join(f"{a:.1f}" for a in g.mean_angles_deg),
            "spread_distance_ang": round(g.spread_distance_ang, 4),
            "spread_angle_deg": round(g.spread_angle_deg, 2),
        }
        for name, value in g.scalars.items():
            row[name] = value
        rows.append(row)
    return rows


def per_atom_rows(frame: Frame, envs: Sequence[Environment],
                  groups: Sequence[Group],
                  per_atom: Mapping[str, np.ndarray] | None = None) -> list[dict]:
    """One row per atom described: its environment, its group and any
    per-atom values given."""
    group_of = {row: g.label for g in groups for row in g.members}
    ids = np.asarray(frame.atom_id)
    rows = []
    for e in envs:
        row = {"row": e.row, "id": int(ids[e.row]), "element": e.element,
               "cn": e.cn, "qn": e.qn,
               "first_shell": " ".join(f"{el}{c}" for el, c in e.first_shell),
               "second_shell": " ".join(f"{el}{c}" for el, c in e.second_shell),
               "distances_ang": " ".join(f"{d:.3f}" for d in e.distances_ang),
               "mean_angle_deg": float(e.angles_deg.mean()) if len(e.angles_deg) else None,
               "group": group_of.get(e.row, "")}
        for name, values in (per_atom or {}).items():
            v = float(np.asarray(values, float)[e.row])
            row[name] = v if np.isfinite(v) else None
        rows.append(row)
    return rows


def populations(groups: Sequence[Group]) -> dict[str, int]:
    return {g.label: g.n for g in groups}
