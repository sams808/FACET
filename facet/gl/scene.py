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

from . import vectors as vectors_mod
from ..core import bv, elements, theme as theme_mod
from ..core.coordination import SiteResult
from ..core.structure import Structure


class Style(Enum):
    BALL_AND_STICK = "ball-and-stick"
    POLYHEDRAL = "polyhedral"
    SPACE_FILLING = "space-filling"
    STICK = "stick"
    WIREFRAME = "wireframe"
    ELLIPSOIDS = "displacement ellipsoids"


# Colour a sub-threshold contact is blended towards. Neutral grey rather than a
# warning colour: these contacts are not errors, they are the contested ones.
_FADE_TO = np.array([0.42, 0.44, 0.48], np.float32)

# What an atom is drawn as in the ellipsoid style when the file gives it no
# displacement parameter at all. Small, and deliberately not the atom's display
# radius: in this style a drawn size is a measurement, and an atom with nothing
# behind it must not look like one with a very isotropic tensor. It is marked
# unmeasured as well, so nothing has to infer that from the size.
FALLBACK_ELLIPSOID_RADIUS = 0.08

# How much thinner bonds are drawn in the ellipsoid style. Measured against the
# ellipsoids themselves: the semi-axes in the reference structures run 0.10 to
# 0.26 A, and an ordinary bond is drawn at about 0.11 A radius, so at full
# width a bond is as thick as the atom it joins.
ELLIPSOID_BOND_SCALE = 0.35


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
    # The displacement ellipsoid of each atom: a 3x3 matrix carrying a unit
    # sphere onto the drawn surface, in world coordinates. None everywhere but
    # the ellipsoid style. Per atom rather than per site because the overlay
    # merge concatenates site indices from different structures without
    # offsetting them, so atom_site is not a key there.
    atom_shape: np.ndarray | None = None
    # Which of those came from an anisotropic tensor, as against an isotropic
    # parameter or nothing at all. It is what decides whether the principal
    # axes are drawn on the ellipsoid: a sphere built from U_iso has no
    # principal axes to draw, and a cross on it would assert a direction the
    # file never gave.
    atom_anisotropic: np.ndarray | None = None
    ellipsoid_probability: float = 0.50

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
    # which end of each bond is the cation: 0 for bond_atoms[:, 0], 1 for the
    # other, -1 for a contact with no cation. Needed because a bond-valence
    # vector points from the cation outwards, and either endpoint may be it.
    bond_cation: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    # the ligand's occupancy, which weights its contribution to a valence sum
    # exactly as it does in coordination.analyse_structure
    bond_occupancy: np.ndarray = field(
        default_factory=lambda: np.ones(0, np.float32))
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

    # --- overlays ---------------------------------------------------------
    # Annotations drawn from the analysis rather than from the crystallography:
    # the bond-valence vector sum on each cation, and the void cone. One entry
    # per kind, each a triangle soup whose parts stay addressable so that a slab
    # can drop a whole lobe instead of slicing one in half.
    overlay_meshes: list = field(default_factory=list)
    show_vectors: bool = False
    show_void_cones: bool = False
    # Length = vector_scale x phi x mean bond length. The only imported number
    # in the construction, and it is a drawing scale rather than a distance to
    # anything, which is why it is here and not in the analysis.
    vector_scale: float = 0.6
    vector_sums: object | None = None

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

    @property
    def n_overlay_triangles(self) -> int:
        return sum(m.n_triangles for m in self.overlay_meshes)

    def clear_overlays(self) -> None:
        self.overlay_meshes = []

    def rebuild_overlays(self, theme=None) -> None:
        """Recompute the analysis overlays for the current threshold.

        Called from ``restyle``, so the lobes follow the cutoff slider like
        everything else does. Working from the cached bond arrays rather than
        from a fresh neighbour search is what makes that affordable, and it is
        also exact for the periodic images that were added to close a bond --
        they carry no site of their own to look an answer up against.
        """
        self.overlay_meshes = []
        theme = theme or self.theme
        if self.n_bonds == 0 or self.n_atoms == 0:
            self.vector_sums = None
            return
        if len(self.bond_cation) != self.n_bonds:
            self.vector_sums = None
            return

        sums = vectors_mod.accumulate(
            self.bond_a, self.bond_b, self.bond_atoms, self.bond_cation,
            self.bond_valence, self.v_bond, self.n_atoms,
            occupancy=self.bond_occupancy)
        self.vector_sums = sums

        if self.show_vectors:
            # phi = 0 draws nothing: a centrosymmetric site has no direction to
            # draw, and inventing one out of rounding noise would be a claim.
            live = (sums.phi > 1e-6) & (sums.count > 0)
            if live.any():
                magnitude = np.linalg.norm(sums.vector[live], axis=1)
                direction = -sums.vector[live] / magnitude[:, None]
                length = (float(self.vector_scale) * sums.phi[live]
                          * sums.mean_distance[live])
                colour = (getattr(theme, "vector_color", None)
                          or vectors_mod.overlay_colors(theme)[0])
                # Start at the drawn surface of the atom, not at its centre.
                # The lobe's own length is still exactly what was computed; it
                # is simply drawn where it can be seen, since a lobe shorter
                # than the sphere it sits inside shows nothing at all.
                start = (self.atom_position[live]
                         + direction * self.atom_radius[live][:, None])
                self.overlay_meshes.append(vectors_mod.lobe_meshes(
                    start, direction, length,
                    atoms=np.nonzero(live)[0], color=colour))

        if self.show_void_cones and self.poly_sites:
            self._add_void_cones(sums, theme)

    def _add_void_cones(self, sums, theme) -> None:
        """A cone of the measured void half-angle on each shown polyhedron.

        Restricted to the sites whose polyhedra are drawn, because the cone is
        found by a search over directions -- a few milliseconds per atom, which
        is nothing for the handful of sites on screen and would be felt on every
        cation of a large cell.
        """
        from ..core import polyhedra

        wanted = set(int(i) for i in self.poly_sites)
        origins, axes, angles, lengths, atoms = [], [], [], [], []
        for atom in range(self.n_atoms):
            if int(self.atom_site[atom]) not in wanted or sums.count[atom] < 2:
                continue
            here = self._ligand_directions(atom)
            if len(here) < 2:
                continue
            angle, axis = polyhedra.void_cone(here)
            origins.append(self.atom_position[atom])
            axes.append(axis)
            angles.append(angle)
            lengths.append(max(float(sums.mean_distance[atom]) * 0.75, 0.3))
            atoms.append(atom)
        if not origins:
            return
        colour = (getattr(theme, "cone_color", None)
                  or vectors_mod.overlay_colors(theme)[1])
        self.overlay_meshes.append(vectors_mod.cone_meshes(
            origins, axes, angles, lengths, atoms=atoms, color=colour))

    def _ligand_directions(self, atom: int) -> np.ndarray:
        """Unit vectors from one drawn atom to its bonded ligands.

        Taken from the drawn bond endpoints, so an atom that is a periodic image
        gets its own directions rather than a representative's.
        """
        if self.n_bonds == 0 or len(self.bond_cation) != self.n_bonds:
            return np.zeros((0, 3))
        pairs = np.asarray(self.bond_atoms, np.int64)
        cation = np.asarray(self.bond_cation, np.int64)
        above = np.asarray(self.bond_valence) >= self.v_bond
        first = above & (cation == 0) & (pairs[:, 0] == atom)
        second = above & (cation == 1) & (pairs[:, 1] == atom)
        out = []
        if first.any():
            out.append(self.bond_b[first] - self.bond_a[first])
        if second.any():
            out.append(self.bond_a[second] - self.bond_b[second])
        if not out:
            return np.zeros((0, 3))
        return np.vstack(out).astype(float)

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
            self.rebuild_overlays()
            return

        v = self.bond_valence
        is_bond = v >= self.v_bond

        # The theme's bond scale has to be reapplied here. restyle() recomputes
        # the radii from scratch, so without this it silently reverts every
        # bond to the unscaled width the moment the threshold is touched.
        scale = float(getattr(self.theme, "bond_scale", 1.0) or 1.0)
        if self.style is Style.ELLIPSOIDS:
            # Thin sticks, as every ellipsoid plot since ORTEP has drawn them.
            # A 50% ellipsoid is a tenth of an angstrom across and an ordinary
            # bond is drawn wider than that, so at full width the bonds are the
            # picture and the ellipsoids are specks hanging off them.
            scale *= ELLIPSOID_BOND_SCALE
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

        # The vector sum is over the contacts above the threshold, so it moves
        # with the threshold -- which is the point of showing it here at all.
        self.rebuild_overlays()

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
                slab=None,
                overrides=None,
                show_vectors: bool = False,
                show_void_cones: bool = False,
                vector_scale: float = 0.6,
                ellipsoid_probability: float = 0.50) -> Scene:
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
    scene.show_vectors = bool(show_vectors)
    scene.show_void_cones = bool(show_void_cones)
    scene.vector_scale = float(vector_scale)
    # set before the atoms are built: it decides how large every ellipsoid is
    scene.ellipsoid_probability = float(ellipsoid_probability)

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

    # Overrides come after the slab so their atom indices address the atoms that
    # are actually drawn, and after restyle so a per-atom colour is not undone by
    # the threshold recolouring.
    if overrides is not None:
        from ..core import overrides as overrides_mod

        overrides_mod.apply_to_scene(scene, overrides)

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
    if scene.atom_shape is not None:
        scene.atom_shape = scene.atom_shape[index]
    if scene.atom_anisotropic is not None:
        scene.atom_anisotropic = scene.atom_anisotropic[index]
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
                     "bond_cation", "bond_occupancy",
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

    # A lobe is kept or dropped whole, by the position of the atom it belongs
    # to. Filtering its triangles by centroid would cut one in half at the slab
    # face, which would read as a measurement about the lobe's length.
    if scene.overlay_meshes:
        kept = []
        for mesh in scene.overlay_meshes:
            if not mesh.n_parts:
                continue
            inside = planes_mod.slab_mask(structure, mesh.anchors, slab)
            trimmed = mesh.keep_parts(inside)
            if trimmed.n_triangles:
                kept.append(trimmed)
        scene.overlay_meshes = kept


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
    if scene.atom_shape is not None:
        # tiled, not transformed: a lattice translation moves an atom and
        # leaves its displacement tensor exactly as it was
        scene.atom_shape = np.tile(scene.atom_shape, (n_copies, 1, 1))
    if scene.atom_anisotropic is not None:
        scene.atom_anisotropic = np.tile(scene.atom_anisotropic, n_copies)
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
        if len(scene.bond_cation) == base_bonds:
            scene.bond_cation = np.tile(scene.bond_cation, n_copies)
        if len(scene.bond_occupancy) == base_bonds:
            scene.bond_occupancy = np.tile(scene.bond_occupancy, n_copies)
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
    extra_shape, extra_anisotropic = [], []
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
            shape = (None if scene.atom_shape is None
                     else scene.atom_shape[source])
            measured = (False if scene.atom_anisotropic is None
                        else bool(scene.atom_anisotropic[source]))
        else:
            colour = scene._bond_base_b[i]
            radius, label, element, site = 0.3, "", "", -1
            # An image with no atom behind it gets the fallback shape and is
            # marked unmeasured, like any other atom whose file said nothing.
            shape = (None if scene.atom_shape is None
                     else np.eye(3, dtype=np.float32)
                     * FALLBACK_ELLIPSOID_RADIUS)
            measured = False

        extra_shape.append(shape)
        extra_anisotropic.append(measured)
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
    if scene.atom_shape is not None:
        scene.atom_shape = np.concatenate(
            [scene.atom_shape, np.array(extra_shape, np.float32)])
        scene.atom_anisotropic = np.concatenate(
            [scene.atom_anisotropic, np.array(extra_anisotropic, bool)])


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
    if style is Style.ELLIPSOIDS:
        _build_ellipsoids(structure, scene)


def _build_ellipsoids(structure: Structure, scene: Scene) -> None:
    """One transform per atom, from the site's displacement parameters.

    Three cases, kept distinguishable because they say different things:

    * an anisotropic tensor -- the ellipsoid it describes;
    * only U_iso -- a sphere of radius sqrt(U_iso) times the probability
      factor, which is what the file supports and no more;
    * a tensor that is not positive definite, or nothing at all -- a small
      fixed sphere, marked as not measured, because the alternative is to
      draw a shape the file does not contain.

    Nothing here is scaled by occupancy or by the theme's atom size, as the
    radii above are. An ellipsoid is a measurement in angstroms, not a display
    radius, and scaling it would make it a picture of something else. The
    probability is the size control for this style.
    """
    from ..core import adp

    scale = adp.scale_for(scene.ellipsoid_probability)
    shapes = np.zeros((len(structure.atoms), 3, 3), np.float32)
    measured = np.zeros(len(structure.atoms), bool)
    # one ellipsoid per site, then handed to that site's atoms: every atom of
    # a site is the same site, and the tensor is in Cartesian axes already
    per_site = {}
    for i, site in enumerate(structure.sites):
        shape = adp.for_site(structure.cell, site)
        if shape is not None and shape.is_ellipsoid:
            per_site[i] = (shape.transform(scene.ellipsoid_probability), True)
        elif site.u_iso:
            radius = float(np.sqrt(max(site.u_iso, 0.0))) * scale
            per_site[i] = (np.eye(3) * radius, False)
        else:
            per_site[i] = (np.eye(3) * FALLBACK_ELLIPSOID_RADIUS, False)

    for j, atom in enumerate(structure.atoms):
        matrix, is_measured = per_site.get(
            atom.site_index, (np.eye(3) * FALLBACK_ELLIPSOID_RADIUS, False))
        shapes[j] = matrix
        measured[j] = is_measured
    scene.atom_shape = shapes
    scene.atom_anisotropic = measured
    # The drawn radius becomes the ellipsoid's own largest semi-axis, so that
    # everything measuring an atom on screen -- picking, the label offset, the
    # billboard -- has a size that matches what is drawn.
    scene.atom_radius = np.maximum(
        np.linalg.norm(shapes, axis=1).max(axis=1), 1e-3).astype(np.float32)


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
    valence, distance, pairs, cation_end, occupancy = [], [], [], [], []
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
        # i is the cation unless i is the anion; recorded rather than recomputed
        # later, where the sites would have to be looked up again
        cation_end.append(1 if anion_i else 0)
        occupancy.append(float(anion.occupancy))

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
    scene.bond_cation = np.array(cation_end, np.int8)
    scene.bond_occupancy = np.array(occupancy, np.float32)
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
    for mesh in scene.overlay_meshes:
        if len(mesh.vertices):
            pts.append(np.asarray(mesh.vertices, np.float32))
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
    # The overlays are not concatenated; they are rebuilt from the merged bond
    # arrays by the restyle at the end, which is both simpler and correct for
    # bonds that were shifted by an offset.
    out.show_vectors = scenes[0].show_vectors
    out.vector_scale = scenes[0].vector_scale
    # Void cones are not drawn on a merged scene: they are selected by site
    # index, and the merged atom_site arrays come from different structures, so
    # an index no longer names one site.
    out.show_void_cones = False

    atom_pos, atom_rad, atom_col, atom_site = [], [], [], []
    atom_shape, atom_anisotropic = [], []
    structure_of = []
    bond_a, bond_b, base_a, base_b = [], [], [], []
    bond_v, bond_d, bond_pairs = [], [], []
    bond_cat, bond_occ = [], []
    poly_v, poly_n, cell_seg = [], [], []
    base = 0

    for index, (scene, offset) in enumerate(zip(scenes, offsets)):
        offset = np.asarray(offset, np.float32)
        atom_pos.append(scene.atom_position + offset)
        atom_rad.append(scene.atom_radius)
        atom_col.append(scene.atom_color)
        atom_site.append(scene.atom_site)
        atom_shape.append(
            scene.atom_shape if scene.atom_shape is not None
            else np.zeros((scene.n_atoms, 3, 3), np.float32))
        atom_anisotropic.append(
            scene.atom_anisotropic if scene.atom_anisotropic is not None
            else np.zeros(scene.n_atoms, bool))
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
            bond_cat.append(scene.bond_cation
                            if len(scene.bond_cation) == scene.n_bonds
                            else np.zeros(scene.n_bonds, np.int8) - 1)
            bond_occ.append(scene.bond_occupancy
                            if len(scene.bond_occupancy) == scene.n_bonds
                            else np.ones(scene.n_bonds, np.float32))
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
    # Only when at least one of the merged scenes had them; a zero matrix drawn
    # as an ellipsoid is an invisible atom, which is why the ones that had none
    # contribute zeros and the whole array is dropped unless someone did.
    if any(sc.atom_shape is not None for sc in scenes):
        out.atom_shape = np.vstack(atom_shape).astype(np.float32)
        out.atom_anisotropic = np.concatenate(atom_anisotropic)
        out.ellipsoid_probability = scenes[0].ellipsoid_probability
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
        out.bond_cation = np.concatenate(bond_cat).astype(np.int8)
        out.bond_occupancy = np.concatenate(bond_occ).astype(np.float32)
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
    for mesh in scene.overlay_meshes:
        if len(mesh.vertices):
            pts.append(np.asarray(mesh.vertices, np.float32))
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
