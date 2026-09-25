"""Labels: what to write on the picture, and where.

VESTA lets you label atoms and bonds; CrystalMaker adds measurement
annotations. FACET covers both and adds the labels this application exists to
show -- **bond valence per contact**, coordination number, bond-valence sum and
phi per site -- because a figure that says "2.119 Å / 0.924 v.u." next to a bond
carries the argument that a distance alone does not.

Two things here that most implementations skip:

**De-cluttering.** A cell of forty atoms labelled naively is unreadable: the
labels overlap each other and sit on top of the bonds. Labels are placed
nearest-first and one that would collide with a label already placed is dropped,
so what survives is the front of the structure rather than an arbitrary subset.

**Legibility over anything.** Each label is drawn with a contrasting halo, so it
reads over a pale atom, a dark background or a transparent polyhedron without
having to know which it landed on.

Placement is pure: it takes a text-measuring callable rather than a QPainter, so
it can be tested without a window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class AtomLabel(Enum):
    """What to write next to an atom."""

    NONE = "none"
    ELEMENT = "element"                     # Bi
    SITE = "site label"                     # Bi1
    ELEMENT_INDEX = "element + number"      # Bi1 (numbered per element)
    OXIDATION = "element + charge"          # Bi3+
    WYCKOFF = "Wyckoff"                     # 4e
    OCCUPANCY = "occupancy"                 # 0.92
    COORDINATION = "coordination number"    # CN 5
    BVS = "bond-valence sum"                # 2.93 v.u.
    PHI = "phi"                             # 0.433
    COORDINATES = "fractional coordinates"  # 0.777 0.304 0.707


class BondLabel(Enum):
    """What to write along a bond."""

    NONE = "none"
    DISTANCE = "distance"                   # 2.119 Å
    VALENCE = "bond valence"                # 0.924 v.u.
    BOTH = "distance + valence"             # 2.119 Å · 0.924 v.u.
    PERCENT = "percent of shortest"         # 112 %
    FRACTION_OF_BVS = "share of the sum"    # 32 %


class LabelScope(Enum):
    """Which objects get labelled."""

    ALL = "everything"
    CATIONS = "cations only"
    ANIONS = "anions only"
    SELECTED_SITE = "the selected site"
    BONDED_ONLY = "bonds above the threshold"


@dataclass
class LabelSettings:
    atom: AtomLabel = AtomLabel.NONE
    atom_scope: LabelScope = LabelScope.ALL
    bond: BondLabel = BondLabel.NONE
    bond_scope: LabelScope = LabelScope.BONDED_ONLY

    show_axes: bool = False            # a, b, c at the cell origin
    show_measurements: bool = True     # the click-to-measure chain

    font_points: float = 9.0
    bold: bool = False
    decimals: int = 3
    halo: bool = True
    color: tuple[float, float, float] | None = None     # None: follow the theme
    offset: tuple[int, int] = (7, -7)                   # pixels from the anchor
    declutter: bool = True
    max_labels: int = 400              # a hard stop; a wall of text is not a figure

    def any_enabled(self) -> bool:
        return (self.atom is not AtomLabel.NONE
                or self.bond is not BondLabel.NONE
                or self.show_axes)


@dataclass
class PlacedLabel:
    text: str
    x: float
    y: float
    depth: float
    kind: str = "atom"                 # atom | bond | axis | measurement
    emphasis: bool = False


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------

def _fmt(value, decimals: int) -> str:
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        return str(value)
    return f"{value:.{decimals}f}"


def atom_text(kind: AtomLabel, *, element: str, label: str, index: int,
              occupancy: float, result=None, site=None, frac=None,
              decimals: int = 3) -> str:
    """The text for one atom, or an empty string for nothing to say."""
    if kind is AtomLabel.NONE:
        return ""
    if kind is AtomLabel.ELEMENT:
        return element
    if kind is AtomLabel.SITE:
        return label
    if kind is AtomLabel.ELEMENT_INDEX:
        return f"{element}{index}"
    if kind is AtomLabel.OXIDATION:
        ox = getattr(site, "ox", None) if site is not None else None
        if ox is None:
            return element
        return f"{element}{abs(ox)}{'+' if ox > 0 else '-'}"
    if kind is AtomLabel.WYCKOFF:
        w = getattr(site, "wyckoff", None) if site is not None else None
        m = getattr(site, "multiplicity", None) if site is not None else None
        return f"{m or ''}{w}" if w else ""
    if kind is AtomLabel.OCCUPANCY:
        # a full site says nothing interesting; only partial occupancy does
        return "" if occupancy >= 0.999 else _fmt(occupancy, 2)
    if kind is AtomLabel.COORDINATES and frac is not None:
        return " ".join(_fmt(float(c), decimals) for c in frac)

    if result is None:
        return ""
    if kind is AtomLabel.COORDINATION:
        return f"CN {result.cn_valence}"
    if kind is AtomLabel.BVS:
        return f"{_fmt(result.bvs, 2)} v.u."
    if kind is AtomLabel.PHI:
        return f"φ {_fmt(result.phi, decimals)}"
    return ""


def bond_text(kind: BondLabel, *, distance: float, valence: float,
              shortest: float | None = None, bvs: float | None = None,
              decimals: int = 3) -> str:
    if kind is BondLabel.NONE:
        return ""
    if kind is BondLabel.DISTANCE:
        return f"{_fmt(distance, decimals)} Å"
    if kind is BondLabel.VALENCE:
        return f"{_fmt(valence, 3)} v.u."
    if kind is BondLabel.BOTH:
        return f"{_fmt(distance, decimals)} Å · {_fmt(valence, 3)} v.u."
    if kind is BondLabel.PERCENT and shortest:
        return f"{100.0 * distance / shortest:.0f} %"
    if kind is BondLabel.FRACTION_OF_BVS and bvs:
        return f"{100.0 * valence / bvs:.0f} %"
    return ""


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------

def build(scene, camera, settings: LabelSettings, *, width: int, height: int,
          measure, results=None, structure=None,
          selected_site: int | None = None,
          measurement_chain: list[int] | None = None) -> list[PlacedLabel]:
    """Everything to draw, already positioned and de-cluttered.

    `measure` takes a string and returns ``(width, height)`` in pixels, so this
    function needs no painter and can be tested with a stub.
    """
    if not settings.any_enabled() and not measurement_chain:
        return []

    by_site = {r.site_index: r for r in (results or [])}
    candidates: list[PlacedLabel] = []

    if settings.atom is not AtomLabel.NONE and scene.n_atoms:
        candidates += _atom_candidates(scene, camera, settings, width, height,
                                       by_site, structure, selected_site)
    if settings.bond is not BondLabel.NONE and scene.n_bonds:
        candidates += _bond_candidates(scene, camera, settings, width, height,
                                       by_site, selected_site)
    if settings.show_axes and structure is not None:
        candidates += _axis_candidates(structure, camera, width, height)

    # nearest first: when two labels collide the front one is the one that
    # should survive, because it belongs to what the viewer is looking at
    candidates.sort(key=lambda p: p.depth, reverse=True)

    placed: list[PlacedLabel] = []
    boxes: list[tuple[float, float, float, float]] = []
    for label in candidates:
        if len(placed) >= settings.max_labels:
            break
        w, h = measure(label.text)
        x = label.x + settings.offset[0]
        y = label.y + settings.offset[1]
        box = (x, y - h, x + w, y)
        if x < -w or x > width or y < 0 or y > height + h:
            continue
        if settings.declutter and _collides(box, boxes):
            continue
        boxes.append(box)
        label.x, label.y = x, y
        placed.append(label)
    return placed


def _collides(box, boxes) -> bool:
    x0, y0, x1, y1 = box
    for bx0, by0, bx1, by1 in boxes:
        if x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0:
            return True
    return False


def _in_scope(scope: LabelScope, *, is_anion: bool, site_index: int,
              selected_site: int | None) -> bool:
    if scope is LabelScope.ALL:
        return True
    if scope is LabelScope.CATIONS:
        return not is_anion
    if scope is LabelScope.ANIONS:
        return is_anion
    if scope is LabelScope.SELECTED_SITE:
        return selected_site is not None and site_index == selected_site
    return True


def _atom_candidates(scene, camera, settings, width, height, by_site,
                     structure, selected_site) -> list[PlacedLabel]:
    projected = camera.project(scene.atom_position, width, height)
    out: list[PlacedLabel] = []
    per_element: dict[str, int] = {}

    for i in range(scene.n_atoms):
        x, y, z = projected[i]
        if z >= 0:
            continue
        site_index = int(scene.atom_site[i])
        site = (structure.sites[site_index]
                if structure is not None and 0 <= site_index < structure.n_sites
                else None)
        is_anion = bool(site.is_anion) if site is not None else False
        if not _in_scope(settings.atom_scope, is_anion=is_anion,
                         site_index=site_index, selected_site=selected_site):
            continue

        element = scene.atom_element[i]
        per_element[element] = per_element.get(element, 0) + 1
        text = atom_text(
            settings.atom, element=element, label=scene.atom_label[i],
            index=per_element[element],
            occupancy=1.0, result=by_site.get(site_index), site=site,
            frac=(structure.atoms[i].frac
                  if structure is not None and i < structure.n_atoms else None),
            decimals=settings.decimals)
        if not text:
            continue
        out.append(PlacedLabel(text, float(x), float(y), float(z), "atom",
                               emphasis=site_index == selected_site))
    return out


def _bond_candidates(scene, camera, settings, width, height, by_site,
                     selected_site) -> list[PlacedLabel]:
    midpoints = (scene.bond_a + scene.bond_b) * 0.5
    projected = camera.project(midpoints, width, height)
    shortest = float(scene.bond_distance.min()) if scene.n_bonds else None

    out: list[PlacedLabel] = []
    for i in range(scene.n_bonds):
        valence = float(scene.bond_valence[i])
        if (settings.bond_scope is LabelScope.BONDED_ONLY
                and valence < scene.v_bond):
            continue
        source = int(scene.bond_atoms[i, 0])
        site_index = (int(scene.atom_site[source])
                      if 0 <= source < scene.n_atoms else -1)
        if settings.bond_scope is LabelScope.SELECTED_SITE:
            if selected_site is None or site_index != selected_site:
                continue

        x, y, z = projected[i]
        if z >= 0:
            continue
        result = by_site.get(site_index)
        text = bond_text(settings.bond, distance=float(scene.bond_distance[i]),
                         valence=valence, shortest=shortest,
                         bvs=result.bvs if result else None,
                         decimals=settings.decimals)
        if not text:
            continue
        out.append(PlacedLabel(text, float(x), float(y), float(z), "bond",
                               emphasis=valence >= scene.v_bond))
    return out


def _axis_candidates(structure, camera, width, height) -> list[PlacedLabel]:
    """a, b and c written at the ends of the cell edges from the origin."""
    orth = structure.cell.orth
    origin = np.zeros(3)
    out: list[PlacedLabel] = []
    points = [origin + orth @ np.eye(3)[i] for i in range(3)]
    projected = camera.project(np.array([origin] + points), width, height)
    for name, p in zip(("a", "b", "c"), projected[1:]):
        if p[2] >= 0:
            continue
        out.append(PlacedLabel(name, float(p[0]), float(p[1]), float(p[2]),
                               "axis", emphasis=True))
    return out


def measurement_labels(scene, camera, chain: list[int], *, width: int,
                       height: int, decimals: int = 3) -> list[PlacedLabel]:
    """Annotations for the click-to-measure chain.

    Placed at the midpoint of each segment, and at the vertex for an angle, so
    the number sits where the quantity is.
    """
    if not chain or len(chain) < 2:
        return []
    pts = [scene.atom_position[i] for i in chain if i < scene.n_atoms]
    if len(pts) < 2:
        return []
    out: list[PlacedLabel] = []

    for a, b in zip(pts[:-1], pts[1:]):
        mid = (np.asarray(a) + np.asarray(b)) * 0.5
        p = camera.project([mid], width, height)[0]
        if p[2] < 0:
            d = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
            out.append(PlacedLabel(f"{d:.{decimals}f} Å",
                                   float(p[0]), float(p[1]), float(p[2]),
                                   "measurement", emphasis=True))

    if len(pts) >= 3:
        for i in range(1, len(pts) - 1):
            u = np.asarray(pts[i - 1]) - np.asarray(pts[i])
            v = np.asarray(pts[i + 1]) - np.asarray(pts[i])
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < 1e-9 or nv < 1e-9:
                continue
            angle = np.degrees(np.arccos(np.clip(float(u @ v) / (nu * nv), -1, 1)))
            p = camera.project([pts[i]], width, height)[0]
            if p[2] < 0:
                out.append(PlacedLabel(f"{angle:.2f}°",
                                       float(p[0]), float(p[1]), float(p[2]),
                                       "measurement", emphasis=True))
    return out
