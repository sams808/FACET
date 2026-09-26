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

    # --- isosurface -------------------------------------------------------
    # A triangulated level set of a volumetric field: a bond-valence surface,
    # a charge density, an ELF. Kept separate from the coordination polyhedra
    # because the two are drawn with different colours and opacities and are
    # toggled independently.
    iso_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    iso_normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    iso_color: tuple[float, float, float] = (0.98, 0.78, 0.30)
    iso_alpha: float = 0.55
    iso_label: str = ""

    # --- lattice planes ---------------------------------------------------
    # One entry per drawn plane: (vertices, normals, colour, alpha). A list
    # rather than one concatenated buffer because each plane carries its own
    # colour and opacity, and a handful of draw calls costs nothing beside the
    # atoms. Plane outlines are separate line segments.
    plane_meshes: list = field(default_factory=list)
    plane_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 3), np.float32))
    plane_edge_color: tuple[float, float, float] = (0.62, 0.80, 0.94)

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
    # which merged structure each atom came from, when several are shown at
    # once; empty for a single-structure scene
    structure_of: np.ndarray = field(
        default_factory=lambda: np.zeros(0, np.int32))
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

    @property
    def n_iso_triangles(self) -> int:
        return len(self.iso_vertices) // 3

    @property
    def n_planes(self) -> int:
        return len(self.plane_meshes)

    def clear_planes(self) -> None:
        self.plane_meshes = []
        self.plane_edges = np.zeros((0, 2, 3), np.float32)

    def add_plane(self, vertices, normals, color, alpha: float,
                  edges=None) -> None:
        """Adopt one drawn plane, as a triangle soup plus its outline."""
        vertices = np.asarray(vertices, np.float32)
        if not len(vertices):
            return
        normals = np.asarray(normals, np.float32)
        if len(normals) != len(vertices):
            normals = np.zeros_like(vertices)
        self.plane_meshes.append((vertices, normals,
                                  tuple(float(c) for c in color),
                                  float(alpha)))
        if edges is not None and len(edges):
            edges = np.asarray(edges, np.float32).reshape(-1, 2, 3)
            self.plane_edges = (edges if not len(self.plane_edges)
                                else np.vstack([self.plane_edges, edges])
                                ).astype(np.float32)

    def set_isosurface(self, vertices, faces, normals, *, color=None,
                       alpha: float | None = None, label: str = "") -> None:
        """Adopt a triangulated level set.

        Indexed triangles are flattened to a triangle soup, because that is
        what the renderer's one mesh path takes and an isosurface is uploaded
        once rather than edited.
        """
        vertices = np.asarray(vertices, np.float32)
        faces = np.asarray(faces, int)
        normals = np.asarray(normals, np.float32)
        if len(faces) == 0 or len(vertices) == 0:
            self.iso_vertices = np.zeros((0, 3), np.float32)
            self.iso_normals = np.zeros((0, 3), np.float32)
            self.iso_label = ""
            return
        flat = faces.reshape(-1)
        self.iso_vertices = vertices[flat]
        self.iso_normals = (normals[flat] if len(normals) == len(vertices)
                            else np.zeros_like(self.iso_vertices))
        if color is not None:
            self.iso_color = tuple(float(c) for c in color)
        if alpha is not None:
            self.iso_alpha = float(alpha)
        self.iso_label = label

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
                params: bv.ParameterSet | None = None,
                lattice_planes=None,
                slab=None) -> Scene:
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
    _build_bonds(structure, scene, theme, params, v_list)
    _add_bonded_images(structure, scene)

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

    # Restyle before the slab, not after. _replicate deliberately does not tile
    # the bond colours -- restyle regenerates them from the base colours -- so
    # until this runs, a replicated scene's colour arrays are still the length of
    # one cell while every other bond array is n_copies longer. Filtering that
    # inconsistent state is how the slab first met a shape mismatch.
    scene.restyle(v_bond)

    # The slab comes before the planes, so a plane drawn to show where the slab
    # was cut is not itself cut away; and before framing, so the view frames
    # what is left rather than what was removed.
    if slab is not None and getattr(slab, "enabled", False):
        apply_slab(structure, scene, slab)
    if lattice_planes:
        _build_planes(structure, scene, lattice_planes)

    _frame(scene, structure)
    return scene


def _build_planes(structure: Structure, scene: Scene, lattice_planes) -> None:
    """Cut each visible (hkl) plane against the drawn box."""
    from ..core import planes as planes_mod

    scene.clear_planes()
    for plane in lattice_planes:
        if not getattr(plane, "visible", True) or not plane.is_valid:
            continue
        try:
            polygons = planes_mod.plane_polygons(structure, plane,
                                                 scene.cell_range)
        except ValueError:
            continue
        for polygon in polygons:
            vertices, normals = planes_mod.triangulate(polygon)
            edges = None
            if getattr(plane, "show_edges", True) and len(polygon) >= 3:
                edges = np.stack([polygon, np.roll(polygon, -1, axis=0)],
                                 axis=1)
            scene.add_plane(vertices, normals, plane.color, plane.alpha, edges)


def apply_slab(structure: Structure, scene: Scene, slab) -> None:
    """Keep only the atoms, bonds and polyhedra inside the slab.

    Filtering the built scene rather than clipping in a shader, for two
    reasons. It works identically on all three render tiers, including the
    QPainter fallback that has no shaders at all -- and the promise is that
    FACET runs on a machine with no graphics card. And it means what is on
    screen is what the scene contains, so picking, labels and the atom counts
    all agree with the picture instead of the shader quietly disagreeing with
    the arrays behind it.

    Atom indices are renumbered, so every array indexed by atom -- the bond
    endpoints, the selection, the per-structure map -- is remapped with them.
    """
    from ..core import planes as planes_mod

    if scene.n_atoms == 0:
        return
    keep = planes_mod.slab_mask(structure, scene.atom_position, slab)
    if keep.all():
        return

    index = np.nonzero(keep)[0]
    remap = np.full(scene.n_atoms, -1, np.int64)
    remap[index] = np.arange(len(index))

    # how many of the survivors were cell contents rather than bonded images
    cell_atoms = int(np.count_nonzero(index < scene.n_cell_atoms))

    scene.atom_position = scene.atom_position[index]
    scene.atom_radius = scene.atom_radius[index]
    scene.atom_color = scene.atom_color[index]
    scene.atom_index = scene.atom_index[index]
    scene.atom_site = scene.atom_site[index]
    scene.atom_label = [scene.atom_label[i] for i in index]
    scene.atom_element = [scene.atom_element[i] for i in index]
    if len(scene.structure_of):
        scene.structure_of = scene.structure_of[index]
    scene.n_cell_atoms = cell_atoms

    if scene.selected_atom is not None:
        new_selection = int(remap[scene.selected_atom])             if 0 <= scene.selected_atom < len(remap) else -1
        scene.selected_atom = None if new_selection < 0 else new_selection

    if scene.n_bonds:
        # A bond is kept or dropped by where it is DRAWN, not by which
        # crystallographic atoms it names. The two differ: bond_a is exactly the
        # first atom's position, but bond_b is the contact position, which for a
        # bond crossing the cell edge is a periodic image -- and bond_atoms
        # records the image's home-cell representative, sometimes ten angstrom
        # away. Filtering on bond_atoms would keep bonds whose far end has been
        # cut away, which is the trailing-bond fault again, and drop bonds that
        # lie wholly inside the slab.
        inside_a = planes_mod.slab_mask(structure, scene.bond_a, slab)
        inside_b = planes_mod.slab_mask(structure, scene.bond_b, slab)
        alive = inside_a & inside_b

        for name in ("bond_a", "bond_b", "bond_radius", "bond_color_a",
                     "bond_color_b", "bond_valence", "bond_distance",
                     "_bond_base_a", "_bond_base_b"):
            array = getattr(scene, name)
            # Length-checked rather than filtered blind: a caller may hand in a
            # scene whose colour arrays have not been regenerated yet, and
            # silently indexing the wrong one with the right-sized mask would
            # scramble the colours instead of failing.
            if len(array) == len(alive):
                setattr(scene, name, array[alive])

        # Re-point the named atoms at atoms that are still drawn. The first is
        # simply remapped, since bond_a is that atom and it survived. The second
        # is found from the drawn endpoint, so it names the image actually at the
        # end of the tube rather than a representative that may now be gone --
        # and an image carries the same site, element and label as the atom it
        # came from, so nothing downstream reads differently.
        pairs = scene.bond_atoms[alive]
        first = remap[pairs[:, 0]]
        second = pairs[:, 1].astype(np.int64)
        if len(scene.bond_b):
            from scipy.spatial import cKDTree

            tree = cKDTree(scene.atom_position)
            distance, nearest = tree.query(np.asarray(scene.bond_b, float))
            found = distance < 1.0e-4
            second = np.where(found, nearest, remap[pairs[:, 1]])
        scene.bond_atoms = np.stack([first, second], axis=1).astype(np.int32)

    if scene.n_poly_triangles:
        triangles = scene.poly_vertices.reshape(-1, 3, 3)
        centroids = triangles.mean(axis=1)
        inside = planes_mod.slab_mask(structure, centroids, slab)
        scene.poly_vertices = triangles[inside].reshape(-1, 3).astype(
            np.float32)
        scene.poly_normals = scene.poly_normals.reshape(-1, 3, 3)[
            inside].reshape(-1, 3).astype(np.float32)

    if scene.n_iso_triangles:
        triangles = scene.iso_vertices.reshape(-1, 3, 3)
        inside = planes_mod.slab_mask(structure, triangles.mean(axis=1), slab)
        scene.iso_vertices = triangles[inside].reshape(-1, 3).astype(np.float32)
        scene.iso_normals = scene.iso_normals.reshape(-1, 3, 3)[
            inside].reshape(-1, 3).astype(np.float32)


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
    if len(scene.iso_vertices):
        scene.iso_vertices = np.vstack(
            [scene.iso_vertices + sh for sh in shifts]).astype(np.float32)
        scene.iso_normals = np.tile(scene.iso_normals, (n_copies, 1))
    # Lattice planes are deliberately NOT replicated. A plane is already cut to
    # the whole drawn box, cell range included, so copying it per cell would lay
    # several coincident sheets on top of each other and darken the blend.


def _add_bonded_images(structure: Structure, scene: Scene) -> None:
    """Draw the periodic images that terminate a bond.

    Without these, every bond crossing a cell boundary runs out to an atom that
    is not on screen. Only the images that actually close a bond are added.

    Duplicates are rejected by **distance**, not by rounding coordinates to a
    fixed number of decimals. Rounding has a boundary failure -- two
    representations of the same point either side of a rounding step become two
    different keys -- and that produced pairs of atoms a few thousandths of an
    angstrom apart, drawn as one slightly thickened sphere.
    """
    if scene.n_bonds == 0 or scene.n_atoms == 0:
        return

    from scipy.spatial import cKDTree

    known = list(scene.atom_position)
    tree = cKDTree(np.array(known, float))
    extra_pos, extra_col, extra_rad = [], [], []
    extra_label, extra_element, extra_site = [], [], []
    added: list[np.ndarray] = []

    TOL = 0.05          # angstrom; far below any real interatomic distance

    for i in range(scene.n_bonds):
        end = np.asarray(scene.bond_b[i], float)
        if tree.query(end, distance_upper_bound=TOL)[0] < TOL:
            continue
        if any(np.linalg.norm(end - p) < TOL for p in added):
            continue
        added.append(end)

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
    """Apply the theme's colour mode."""
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


def _bond_key(i: int, j: int, image) -> tuple:
    """A canonical identity for one contact, so it is drawn once.

    A contact between two atoms of the home cell is found twice, once from each
    end, with opposite lattice translations. Ordering the pair and flipping the
    translation with it collapses the two into one key.
    """
    image = tuple(int(x) for x in image)
    return min((i, j, image), (j, i, tuple(-x for x in image)))


def _build_bonds(structure: Structure, scene: Scene, theme,
                 params, v_list: float) -> None:
    """Every drawn contact, taken straight from the periodic neighbour search.

    This used to propagate the representative atom's contact vectors to the
    other atoms of the same site by **translation**, which is wrong for every
    structure whose sites have multiplicity above one: symmetry copies are
    related by rotations, screws and inversions, not by translation. Each copy
    was given the representative's bond directions, producing bonds that
    pointed at nothing and invented image atoms a tenth of an angstrom away
    from real ones.

    Searching per atom is exact and costs little: the neighbour tree is built
    once for the structure, and querying it is milliseconds.
    """
    from ..core.neighbors import NeighborFinder, search_radius_for

    rmax = search_radius_for(structure, params, v_list)
    try:
        finder = NeighborFinder(structure, rmax=rmax)
    except ValueError:
        return

    orth = structure.cell.orth
    a_pos, b_pos, col_a, col_b = [], [], [], []
    valence, distance, pairs = [], [], []
    seen: set[tuple] = set()

    for i, j, d, image in finder.pairs_within(rmax):
        site_i = structure.atoms[i].site_index
        site_j = structure.atoms[j].site_index
        anion_i = structure.sites[site_i].is_anion
        anion_j = structure.sites[site_j].is_anion
        # a bond valence is defined for a cation against an anion; a
        # cation-cation contact is reported separately, not drawn as a bond
        if anion_i == anion_j:
            continue

        key = _bond_key(i, j, image)
        if key in seen:
            continue
        seen.add(key)

        cation = structure.atoms[j if anion_i else i]
        anion = structure.atoms[i if anion_i else j]
        site = structure.sites[cation.site_index]
        param = params.get(site.element, site.ox, anion.element)
        if param is None:
            continue
        v = float(param.valence(d))
        if v <= v_list:
            continue

        shift = orth @ np.array(image, float)
        a_pos.append(structure.atoms[i].cart)
        b_pos.append(structure.atoms[j].cart + shift)
        ca, cb = _bond_colors(theme, structure.atoms[i].element,
                              structure.atoms[j].element, v)
        col_a.append(ca)
        col_b.append(cb)
        valence.append(v)
        distance.append(d)
        pairs.append((i, j))

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
    if len(scene.iso_vertices):
        pts.append(scene.iso_vertices)
    for vertices, _, _, _ in scene.plane_meshes:
        pts.append(vertices)
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


def merge_scenes(scenes: list[Scene], offsets=None) -> Scene:
    """Combine several scenes into one drawable, each shifted by its offset.

    Used to show a set of structures at once. Atom indices are renumbered so
    that picking still identifies a unique atom, and `structure_of` records
    which scene each atom came from so a click can be traced back to a file.
    """
    scenes = [s for s in scenes if s is not None and s.n_atoms]
    if not scenes:
        return Scene()
    if offsets is None:
        offsets = [np.zeros(3) for _ in scenes]

    out = Scene(style=scenes[0].style, v_bond=scenes[0].v_bond,
                theme=scenes[0].theme)
    out.poly_color = scenes[0].poly_color
    out.poly_alpha = scenes[0].poly_alpha

    atom_pos, atom_rad, atom_col, atom_site = [], [], [], []
    structure_of = []
    bond_a, bond_b, base_a, base_b = [], [], [], []
    bond_v, bond_d, bond_pairs = [], [], []
    poly_v, poly_n, cell_seg = [], [], []
    base = 0

    for index, (scene, offset) in enumerate(zip(scenes, offsets)):
        offset = np.asarray(offset, np.float32)
        atom_pos.append(scene.atom_position + offset)
        atom_rad.append(scene.atom_radius)
        atom_col.append(scene.atom_color)
        atom_site.append(scene.atom_site)
        out.atom_label.extend(scene.atom_label)
        out.atom_element.extend(scene.atom_element)
        structure_of.extend([index] * scene.n_atoms)

        if scene.n_bonds:
            bond_a.append(scene.bond_a + offset)
            bond_b.append(scene.bond_b + offset)
            base_a.append(scene._bond_base_a)
            base_b.append(scene._bond_base_b)
            bond_v.append(scene.bond_valence)
            bond_d.append(scene.bond_distance)
            bond_pairs.append(scene.bond_atoms + base)
        if len(scene.poly_vertices):
            poly_v.append(scene.poly_vertices + offset)
            poly_n.append(scene.poly_normals)
        if len(scene.cell_segments):
            cell_seg.append(scene.cell_segments + offset)
        base += scene.n_atoms

    out.atom_position = np.vstack(atom_pos).astype(np.float32)
    out.atom_radius = np.concatenate(atom_rad).astype(np.float32)
    out.atom_color = np.vstack(atom_col).astype(np.float32)
    out.atom_site = np.concatenate(atom_site).astype(np.int32)
    out.atom_index = np.arange(len(out.atom_radius), dtype=np.int32)
    out.structure_of = np.array(structure_of, np.int32)
    out.n_cell_atoms = sum(s.n_cell_atoms for s in scenes)

    if bond_a:
        out.bond_a = np.vstack(bond_a).astype(np.float32)
        out.bond_b = np.vstack(bond_b).astype(np.float32)
        out._bond_base_a = np.vstack(base_a).astype(np.float32)
        out._bond_base_b = np.vstack(base_b).astype(np.float32)
        out.bond_color_a = out._bond_base_a.copy()
        out.bond_color_b = out._bond_base_b.copy()
        out.bond_valence = np.concatenate(bond_v).astype(np.float32)
        out.bond_distance = np.concatenate(bond_d).astype(np.float32)
        out.bond_atoms = np.vstack(bond_pairs).astype(np.int32)
        # n_bonds reads the radius array, and restyle() returns early when it
        # is empty -- so it has to be filled before restyling, not by it
        out.bond_radius = bond_radius_for(out.bond_valence, out.v_bond)
    if poly_v:
        out.poly_vertices = np.vstack(poly_v).astype(np.float32)
        out.poly_normals = np.vstack(poly_n).astype(np.float32)
    if cell_seg:
        out.cell_segments = np.vstack(cell_seg).astype(np.float32)

    _frame_points(out)
    out.restyle(out.v_bond)
    return out


def _frame_points(scene: Scene) -> None:
    """Bounding sphere of a merged scene."""
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
