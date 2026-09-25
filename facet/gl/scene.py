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

from ..core import bv, elements, theme as theme_mod
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
    poly_sites: list[int] = field(default_factory=list)

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
    cell_range: tuple[int, int, int] = (1, 1, 1)
    # the theme this scene was built with; the renderer reads its presentation
    # settings, and restyle() needs its sub-threshold colour and bond scale
    theme: object | None = None

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

        # The theme's bond scale has to be reapplied here. restyle() recomputes
        # the radii from scratch, so without this it silently reverts every
        # bond to the unscaled width the moment the threshold is touched.
        scale = float(getattr(self.theme, "bond_scale", 1.0) or 1.0)
        self.bond_radius = bond_radius_for(v, self.v_bond) * scale

        # fade everything below the threshold towards neutral, proportionally,
        # so a contact just under the line still looks nearly like a bond
        fade = np.array(getattr(self.theme, "subthreshold_color", None)
                        or _FADE_TO, np.float32)
        frac = np.clip(v / max(self.v_bond, 1e-9), 0.0, 1.0)[:, None]
        weight = np.where(is_bond[:, None], 1.0, 0.25 + 0.75 * frac)
        self.bond_color_a = (self._bond_base_a * weight
                             + fade * (1.0 - weight)).astype(np.float32)
        self.bond_color_b = (self._bond_base_b * weight
                             + fade * (1.0 - weight)).astype(np.float32)

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
                polyhedron_sites: list[int] | None = None,
                show_cell: bool = True,
                cell_range: tuple[int, int, int] = (1, 1, 1),
                theme=None,
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

    theme = theme or theme_mod.Theme()
    scene = Scene(style=style, v_bond=v_bond, theme=theme)
    scene.poly_color = theme.polyhedron_color
    scene.poly_alpha = theme.polyhedron_alpha
    scene.cell_range = tuple(max(1, int(n)) for n in cell_range)

    _build_atoms(structure, scene, style, theme, results)
    _build_bonds(structure, results, scene, theme)
    _add_bonded_images(scene, style)

    sites = list(polyhedron_sites or [])
    if polyhedron_site is not None and polyhedron_site not in sites:
        sites.append(polyhedron_site)
    scene.poly_sites = sites
    if sites:
        _build_polyhedra(structure, results, scene, sites)

    if show_cell:
        scene.cell_segments = unit_cell_segments(structure, scene.cell_range)
    if scene.cell_range != (1, 1, 1):
        _replicate(structure, scene)
    _frame(scene, structure)
    scene.restyle(v_bond)
    return scene


def _replicate(structure: Structure, scene: Scene) -> None:
    """Repeat the drawn contents across a block of unit cells.

    CrystalMaker and VESTA both call this a cell range, and it is how anyone
    looks at connectivity beyond a single cell. Applied after bonds are built,
    so every copy carries the same bonds including those crossing a boundary.
    """
    nx, ny, nz = scene.cell_range
    if (nx, ny, nz) == (1, 1, 1):
        return
    orth = structure.cell.orth
    shifts = [orth @ np.array([i, j, k], float)
              for i in range(nx) for j in range(ny) for k in range(nz)]
    n_copies = len(shifts)
    if n_copies <= 1:
        return

    base_atoms = scene.n_atoms
    base_bonds = scene.n_bonds
    scene.atom_position = np.vstack(
        [scene.atom_position + sh for sh in shifts]).astype(np.float32)
    scene.atom_radius = np.tile(scene.atom_radius, n_copies)
    scene.atom_color = np.tile(scene.atom_color, (n_copies, 1))
    scene.atom_site = np.tile(scene.atom_site, n_copies)
    scene.atom_label = scene.atom_label * n_copies
    scene.atom_element = scene.atom_element * n_copies
    scene.atom_index = np.arange(len(scene.atom_radius), dtype=np.int32)
    scene.n_cell_atoms = scene.n_cell_atoms * n_copies

    if base_bonds:
        scene.bond_a = np.vstack([scene.bond_a + sh for sh in shifts]).astype(np.float32)
        scene.bond_b = np.vstack([scene.bond_b + sh for sh in shifts]).astype(np.float32)
        scene.bond_valence = np.tile(scene.bond_valence, n_copies)
        scene.bond_distance = np.tile(scene.bond_distance, n_copies)
        scene._bond_base_a = np.tile(scene._bond_base_a, (n_copies, 1))
        scene._bond_base_b = np.tile(scene._bond_base_b, (n_copies, 1))
        scene.bond_radius = np.tile(scene.bond_radius, n_copies)
        offsets = np.repeat(np.arange(n_copies) * base_atoms, base_bonds)
        scene.bond_atoms = (np.tile(scene.bond_atoms, (n_copies, 1))
                            + offsets[:, None]).astype(np.int32)

    if len(scene.poly_vertices):
        scene.poly_vertices = np.vstack(
            [scene.poly_vertices + sh for sh in shifts]).astype(np.float32)
        scene.poly_normals = np.tile(scene.poly_normals, (n_copies, 1))


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
        # Take everything from the atom this is an image OF, including its
        # colour. Using the bond's far-half colour instead looks right only
        # while bonds are coloured by element -- under a uniform or a
        # by-valence bond colouring the image atoms would silently diverge
        # from the cell atoms they duplicate.
        source = int(scene.bond_atoms[i, 1])
        if 0 <= source < scene.n_atoms:
            colour = scene.atom_color[source]
            radius = float(scene.atom_radius[source])
            label = scene.atom_label[source]
            element = scene.atom_element[source]
            site = int(scene.atom_site[source])
        else:
            colour = scene._bond_base_b[i]
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


def _build_atoms(structure: Structure, scene: Scene, style: Style,
                 theme, results) -> None:
    atoms = structure.atoms
    if not atoms:
        return
    scene.atom_position = np.array([a.cart for a in atoms], np.float32)
    scene.atom_index = np.arange(len(atoms), dtype=np.int32)
    scene.atom_site = np.array([a.site_index for a in atoms], np.int32)
    scene.atom_label = [a.label for a in atoms]
    scene.atom_element = [a.element for a in atoms]
    scene.atom_color = _atom_colors(structure, atoms, theme, results)

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
    scene.atom_radius = (scene.atom_radius * (0.55 + 0.45 * np.clip(occ, 0, 1))
                         * float(theme.atom_scale))
    scene.n_cell_atoms = len(atoms)


def _atom_colors(structure: Structure, atoms, theme, results) -> np.ndarray:
    """Apply the theme's colour mode.

    Colouring by element is the default and is what every other program offers.
    The rest put the analysis onto the structure: a site coloured by phi or by
    bond-valence sum shows its chemistry in the picture, not only in a table
    beside it.
    """
    mode = theme.color_mode

    if mode is theme_mod.ColorMode.UNIFORM:
        return np.tile(np.array(theme.uniform_atom_color, np.float32),
                       (len(atoms), 1))

    if mode is theme_mod.ColorMode.SITE:
        cycle = theme_mod.SITE_CYCLE
        return np.array([cycle[a.site_index % len(cycle)] for a in atoms],
                        np.float32)

    if mode is not theme_mod.ColorMode.ELEMENT and results:
        values = theme_mod.site_values(results, mode)
        lo, hi = theme_mod.scale_range(values, theme, mode)
        diverging = mode is theme_mod.ColorMode.VALENCE_DISCREPANCY
        out = []
        for a in atoms:
            if a.site_index in values:
                out.append(theme_mod.color_for_value(values[a.site_index],
                                                     lo, hi, diverging))
            else:
                # anions carry no per-site scalar; keep them elemental so the
                # framework stays readable behind the coloured cations
                out.append(theme.element_color(a.element))
        return np.array(out, np.float32)

    return np.array([theme.element_color(a.element) for a in atoms], np.float32)


def _build_bonds(structure: Structure, results: list[SiteResult],
                 scene: Scene, theme) -> None:
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
                ca, cb = _bond_colors(theme, res.element, c.element, c.valence)
                col_a.append(ca)
                col_b.append(cb)
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
    scene.bond_radius = (bond_radius_for(scene.bond_valence, scene.v_bond)
                         * float(theme.bond_scale))


def _bond_colors(theme, element_a: str, element_b: str, valence: float):
    mode = theme.bond_color_mode
    if mode is theme_mod.BondColorMode.UNIFORM:
        return theme.uniform_bond_color, theme.uniform_bond_color
    if mode is theme_mod.BondColorMode.BY_VALENCE:
        # a ramp from weak to strong, so relative bond strength is legible
        # without having to compare thicknesses
        c = theme_mod.ramp(min(float(valence) / 0.8, 1.0))
        return c, c
    return theme.element_color(element_a), theme.element_color(element_b)


def _build_polyhedra(structure: Structure, results: list[SiteResult],
                     scene: Scene, site_indices: list[int]) -> None:
    """Convex hulls of the bonded ligands, for every requested site.

    Every symmetry copy of a site gets its own polyhedron, not just the
    representative -- a polyhedral view of a framework is the point, and one
    polyhedron floating in a cell is not that.
    """
    verts: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    for site_index in site_indices:
        res = next((r for r in results if r.site_index == site_index), None)
        if res is None or len(res.bonds) < 4:
            continue
        rep_atoms = structure.atoms_of_site(site_index)
        if not rep_atoms:
            continue
        origin = structure.atoms[rep_atoms[0]].cart
        offsets = [structure.atoms[a].cart - origin for a in rep_atoms]
        base = np.array([origin + c.vector for c in res.bonds], float)
        for off in offsets:
            _hull_into(base + off, verts, norms)

    scene.poly_vertices = (np.array(verts, np.float32) if verts
                           else np.zeros((0, 3), np.float32))
    scene.poly_normals = (np.array(norms, np.float32) if norms
                          else np.zeros((0, 3), np.float32))


def _hull_into(points: np.ndarray, verts: list, norms: list) -> None:
    """Append one outward-oriented convex hull to the triangle lists."""
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
    except Exception:
        return
    centroid = points.mean(axis=0)
    for simplex in hull.simplices:
        p, q, r = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        n = np.cross(q - p, r - p)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        if np.dot(n, (p + q + r) / 3.0 - centroid) < 0:
            p, q = q, p
            n = -n
        verts += [p, q, r]
        norms += [n, n, n]


def unit_cell_segments(structure: Structure,
                       cell_range: tuple[int, int, int] = (1, 1, 1)) -> np.ndarray:
    """Unit-cell edges, as (n, 2, 3) cartesian endpoints.

    With a cell range, every cell of the block is outlined rather than only the
    enclosing box -- which is what makes it possible to see where one cell ends
    and the next begins.
    """
    orth = structure.cell.orth
    nx, ny, nz = (max(1, int(n)) for n in cell_range)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                       float)
    edges = []
    for ox in range(nx):
        for oy in range(ny):
            for oz in range(nz):
                shifted = corners + np.array([ox, oy, oz], float)
                cart = (orth @ shifted.T).T
                for i in range(8):
                    for j in range(i + 1, 8):
                        # neighbours on a cube differ in exactly one coordinate
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
