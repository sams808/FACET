"""The drawable scene: renderer-agnostic, and the place the science meets the
picture.

Two decisions here carry most of the application's meaning.

**Bond thickness is bond valence.** Not a constant, not a function of distance:
the radius of a drawn bond tracks ``v`` directly, so a 0.03 v.u. contact reads
as the hairline it is beside a 0.5 v.u. bond. Someone looking at a bismuth site
sees the 3+n split before being told about it.

**Sub-threshold contacts are drawn, not hidden.** Everything above the listing
threshold is in the scene. Contacts below the bond threshold are thin and
desaturated rather than absent, because the whole argument is that where the
line falls between them is a choice. A viewer that simply deleted them would be
making the same silent decision every other program makes.

The scene is built once, at the widest threshold, and carries every contact's
valence. Moving the threshold afterwards restyles what is already there --
:meth:`Scene.restyle` -- and never rebuilds or re-searches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..core import bv, elements
from ..core.coordination import SiteResult
from ..core.structure import Structure


class Style(Enum):
    BALL_AND_STICK = "ball-and-stick"
    POLYHEDRAL = "polyhedral"
    SPACE_FILLING = "space-filling"
    STICK = "stick"
    WIREFRAME = "wireframe"


# Colour a sub-threshold contact is blended towards. Neutral grey rather than a
# warning colour: these contacts are not errors, they are the contested ones.
_FADE_TO = np.array([0.42, 0.44, 0.48], np.float32)


@dataclass
class Scene:
    """Everything to draw, as flat arrays ready for a vertex buffer."""

    # --- atoms -----------------------------------------------------------
    atom_position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    atom_radius: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    atom_color: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    atom_index: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    atom_site: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    atom_label: list[str] = field(default_factory=list)
    atom_element: list[str] = field(default_factory=list)

    # --- bonds -----------------------------------------------------------
    bond_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    bond_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    bond_radius: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    bond_color_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    bond_color_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    bond_valence: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    bond_distance: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    # the two atoms, so a click on a bond can name them
    bond_atoms: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))
    # base colours, kept so restyling can re-fade from the original
    _bond_base_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    _bond_base_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))

    # --- coordination polyhedra -------------------------------------------
    poly_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    poly_normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    poly_color: tuple[float, float, float] = (0.30, 0.74, 0.96)
    poly_alpha: float = 0.38

    # --- unit cell --------------------------------------------------------
    cell_segments: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 3), np.float32))

    # --- framing ----------------------------------------------------------
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    radius: float = 1.0

    # --- state ------------------------------------------------------------
    # atoms beyond this index are periodic images added only to close a bond;
    # they are drawn, but they are not part of the cell's contents
    n_cell_atoms: int = 0
    v_bond: float = bv.V_BOND_DEFAULT
    style: Style = Style.BALL_AND_STICK
    selected_atom: int | None = None

    # ------------------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return len(self.atom_radius)

    @property
    def n_bonds(self) -> int:
        return len(self.bond_radius)

    @property
    def n_poly_triangles(self) -> int:
        return len(self.poly_vertices) // 3

    def restyle(self, v_bond: float) -> None:
        """Apply a new bond threshold to a scene that is already built.

        Pure array arithmetic over the cached valences -- no neighbour search,
        no geometry rebuild. This is what makes the cutoff slider free.
        """
        self.v_bond = float(v_bond)
        if self.n_bonds == 0:
            return

        v = self.bond_valence
        is_bond = v >= self.v_bond

        self.bond_radius = bond_radius_for(v, self.v_bond)

        # fade everything below the threshold towards neutral, proportionally,
        # so a contact just under the line still looks nearly like a bond
        frac = np.clip(v / max(self.v_bond, 1e-9), 0.0, 1.0)[:, None]
        weight = np.where(is_bond[:, None], 1.0, 0.25 + 0.75 * frac)
        self.bond_color_a = (self._bond_base_a * weight
                             + _FADE_TO * (1.0 - weight)).astype(np.float32)
        self.bond_color_b = (self._bond_base_b * weight
                             + _FADE_TO * (1.0 - weight)).astype(np.float32)

    def bonds_above_threshold(self) -> int:
        return int((self.bond_valence >= self.v_bond).sum())


def bond_radius_for(valence, v_bond: float) -> np.ndarray:
    """Drawn radius of a bond, in angstrom, as a function of its valence.

    A power law rather than a linear map: bond valences across one site span
    more than a decade, and linear scaling would make everything below the
    strongest bond invisible. Sub-threshold contacts are additionally pinched
    so that the threshold itself is visible as a step in thickness.
    """
    v = np.asarray(valence, np.float32)
    r = 0.030 + 0.115 * np.clip(v, 0.0, 1.0) ** 0.62
    below = v < v_bond
    return np.where(below, r * 0.45, r).astype(np.float32)


# ---------------------------------------------------------------------------

def build_scene(structure: Structure,
                results: list[SiteResult] | None = None,
                style: Style = Style.BALL_AND_STICK,
                v_bond: float = bv.V_BOND_DEFAULT,
                v_list: float = bv.V_LIST_DEFAULT,
                polyhedron_site: int | None = None,
                show_cell: bool = True,
                params: bv.ParameterSet | None = None) -> Scene:
    """Build a drawable scene for a structure.

    `results` is reused when supplied, so opening a structure does not analyse
    it twice. Contacts down to `v_list` are included; `v_bond` only styles them.
    """
    from ..core import coordination

    params = params or bv.DEFAULT
    if results is None:
        results = coordination.analyse_structure(structure, params,
                                                 v_bond=v_bond, v_list=v_list)

    scene = Scene(style=style, v_bond=v_bond)
    _build_atoms(structure, scene, style)
    _build_bonds(structure, results, scene)
    _add_bonded_images(scene, style)
    if polyhedron_site is not None:
        _build_polyhedron(structure, results, scene, polyhedron_site)
    if show_cell:
        scene.cell_segments = unit_cell_segments(structure)
    _frame(scene, structure)
    scene.restyle(v_bond)
    return scene


def _add_bonded_images(scene: Scene, style: Style) -> None:
    """Draw the periodic images that close a bond.

    Without this, every bond that crosses a cell boundary is drawn running out
    to an atom that is not on screen -- a spray of stubs into empty space, and
    a coordination polyhedron that looks broken. Only the images that actually
    terminate a bond are added, so the atom count stays close to the cell's.
    """
    if scene.n_bonds == 0 or scene.n_atoms == 0:
        return

    def key(p):
        return (round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3))

    present = {key(p) for p in scene.atom_position}
    extra_pos, extra_col, extra_rad = [], [], []
    extra_label, extra_element, extra_site = [], [], []

    for i in range(scene.n_bonds):
        end = scene.bond_b[i]
        k = key(end)
        if k in present:
            continue
        present.add(k)
        # the far half of the bond carries the neighbour's colour, so the image
        # atom is coloured and sized from that rather than looked up again
        colour = scene._bond_base_b[i]
        source = int(scene.bond_atoms[i, 1])
        if 0 <= source < scene.n_atoms:
            radius = float(scene.atom_radius[source])
            label = scene.atom_label[source]
            element = scene.atom_element[source]
            site = int(scene.atom_site[source])
        else:
            radius, label, element, site = 0.3, "", "", -1
        extra_pos.append(end)
        extra_col.append(colour)
        extra_rad.append(radius)
        extra_label.append(label)
        extra_element.append(element)
        extra_site.append(site)

    if not extra_pos:
        return

    n0 = scene.n_atoms
    scene.atom_position = np.vstack([scene.atom_position,
                                     np.array(extra_pos, np.float32)])
    scene.atom_color = np.vstack([scene.atom_color,
                                  np.array(extra_col, np.float32)])
    scene.atom_radius = np.concatenate([scene.atom_radius,
                                        np.array(extra_rad, np.float32)])
    scene.atom_site = np.concatenate([scene.atom_site,
                                      np.array(extra_site, np.int32)])
    scene.atom_index = np.arange(len(scene.atom_radius), dtype=np.int32)
    scene.atom_label.extend(extra_label)
    scene.atom_element.extend(extra_element)
    scene.n_cell_atoms = n0


def _build_atoms(structure: Structure, scene: Scene, style: Style) -> None:
    atoms = structure.atoms
    if not atoms:
        return
    scene.atom_position = np.array([a.cart for a in atoms], np.float32)
    scene.atom_index = np.arange(len(atoms), dtype=np.int32)
    scene.atom_site = np.array([a.site_index for a in atoms], np.int32)
    scene.atom_label = [a.label for a in atoms]
    scene.atom_element = [a.element for a in atoms]
    scene.atom_color = np.array(
        [elements.info(a.element).color for a in atoms], np.float32)

    info = [elements.info(a.element) for a in atoms]
    if style is Style.SPACE_FILLING:
        radii = [(i.vdw_radius or (i.display_radius * 2.6)) for i in info]
    elif style in (Style.STICK, Style.WIREFRAME):
        radii = [0.10 for _ in info]
    else:
        radii = [i.display_radius for i in info]
    scene.atom_radius = np.array(radii, np.float32)

    # a partially occupied site is drawn smaller, so disorder is visible
    occ = np.array([a.occupancy for a in atoms], np.float32)
    scene.atom_radius = scene.atom_radius * (0.55 + 0.45 * np.clip(occ, 0, 1))


def _build_bonds(structure: Structure, results: list[SiteResult],
                 scene: Scene) -> None:
    """One drawn bond per contact of every analysed site.

    Contacts are de-duplicated: a cation-anion contact appears in the cation's
    list, and would appear again in the anion's if anions were analysed too.
    """
    a_pos, b_pos, col_a, col_b = [], [], [], []
    valence, distance, pairs = [], [], []
    seen: set[tuple] = set()

    for res in results:
        centres = structure.atoms_of_site(res.site_index)
        if not centres:
            continue
        # every atom of the site, not just the representative: the whole cell
        # is drawn, and each copy needs its own bonds
        for centre in centres:
            origin = structure.atoms[centre].cart
            rep = structure.atoms[structure.atoms_of_site(res.site_index)[0]].cart
            offset = origin - rep
            for c in res.contacts:
                if c.valence is None:
                    continue
                end = rep + c.vector + offset
                key = (round(float(origin[0]), 3), round(float(origin[1]), 3),
                       round(float(origin[2]), 3), round(float(end[0]), 3),
                       round(float(end[1]), 3), round(float(end[2]), 3))
                rev = key[3:] + key[:3]
                if key in seen or rev in seen:
                    continue
                seen.add(key)
                a_pos.append(origin)
                b_pos.append(end)
                col_a.append(elements.info(res.element).color)
                col_b.append(elements.info(c.element).color)
                valence.append(c.valence)
                distance.append(c.distance)
                pairs.append((centre, c.atom_index))

    if not a_pos:
        return
    scene.bond_a = np.array(a_pos, np.float32)
    scene.bond_b = np.array(b_pos, np.float32)
    scene._bond_base_a = np.array(col_a, np.float32)
    scene._bond_base_b = np.array(col_b, np.float32)
    scene.bond_color_a = scene._bond_base_a.copy()
    scene.bond_color_b = scene._bond_base_b.copy()
    scene.bond_valence = np.array(valence, np.float32)
    scene.bond_distance = np.array(distance, np.float32)
    scene.bond_atoms = np.array(pairs, np.int32)
    scene.bond_radius = bond_radius_for(scene.bond_valence, scene.v_bond)


def _build_polyhedron(structure: Structure, results: list[SiteResult],
                      scene: Scene, site_index: int) -> None:
    """Convex hull of the bonded ligands of one site."""
    res = next((r for r in results if r.site_index == site_index), None)
    if res is None:
        return
    bonded = res.bonds
    if len(bonded) < 4:
        return

    atoms = structure.atoms_of_site(site_index)
    if not atoms:
        return
    origin = structure.atoms[atoms[0]].cart
    points = np.array([origin + c.vector for c in bonded], float)

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
    except Exception:
        return

    centroid = points.mean(axis=0)
    verts, norms = [], []
    for simplex in hull.simplices:
        p, q, r = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        n = np.cross(q - p, r - p)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        if np.dot(n, (p + q + r) / 3.0 - centroid) < 0:
            p, q = q, p
            n = -n
        verts += [p, q, r]
        norms += [n, n, n]
    scene.poly_vertices = np.array(verts, np.float32)
    scene.poly_normals = np.array(norms, np.float32)


def unit_cell_segments(structure: Structure) -> np.ndarray:
    """The twelve edges of the unit cell, as (12, 2, 3) cartesian endpoints."""
    orth = structure.cell.orth
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                       float)
    cart = (orth @ corners.T).T
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            # neighbours on the cube differ in exactly one coordinate
            if int(np.abs(corners[i] - corners[j]).sum()) == 1:
                edges.append([cart[i], cart[j]])
    return np.array(edges, np.float32)


def _frame(scene: Scene, structure: Structure) -> None:
    """Bounding sphere of everything drawn, atom radii included."""
    pts = [scene.atom_position] if scene.n_atoms else []
    if len(scene.cell_segments):
        pts.append(scene.cell_segments.reshape(-1, 3))
    if not pts:
        scene.center = np.zeros(3, np.float32)
        scene.radius = 1.0
        return
    allpts = np.concatenate(pts, axis=0)
    centre = allpts.mean(axis=0)
    r = float(np.linalg.norm(allpts - centre, axis=1).max())
    if scene.n_atoms:
        r = max(r, float((np.linalg.norm(scene.atom_position - centre, axis=1)
                          + scene.atom_radius).max()))
    scene.center = centre.astype(np.float32)
    scene.radius = max(r, 1e-3)
