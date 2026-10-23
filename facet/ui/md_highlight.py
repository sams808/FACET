"""The Highlight page of the Model workspace: rules that change the 3D view
of the frame shown, and the functions that apply them to a built Scene.

The rules edit the arrays of a :class:`~facet.gl.scene.Scene` that
:func:`facet.ui.md_scene.build_md_scene` built (atom colours and radii,
bond colours and radii, the overlay meshes), so every renderer tier draws
them, the QPainter one included, and the SVG and PDF exports carry them:
no renderer change. Three kinds of rule:

* *Show only [element | any] where [CN | former CN | BVS | phi | Qn]
  [≤ | ≥] value*: the atoms outside the rule are dimmed (paler and
  smaller, so the inside of an opaque box stays visible; measured on the
  prototype) or, with "Dim the rest" off, left out of the scene;
* *Voids ≥ V Å³, elongation ≥ e*: the empty spheres of the frame
  (:func:`facet.core.md_order.empty_spheres`, FACET's van der Waals radii)
  whose volume is at least V, drawn as translucent icospheres; the
  elongation criterion reads the void regions of
  :mod:`facet.core.md_channels` when that build exports ``void_regions``
  and is greyed with the reason otherwise (:func:`channel_backend`);
* *Channels [by charge | by modifier density | by voids] …*: the regions
  :mod:`facet.core.md_channels` measures on this frame, each atom coloured
  by its region under "channel membership" and the atoms outside every
  region dimmed; by charge the mismatch isosurface at Δ is drawn as well,
  under the meshing policy stated beside its constants below (what is
  drawn, never what is counted).

Every threshold is what the field shows: nothing physical is assumed, and
the count line states the frame, v_bond and what was drawn with which
values. :class:`FrameData` holds what the rules read of one frame, with
the empty spheres and the bond-valence landscapes computed once, on demand.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core import bulk, elements as element_data, md_order
from ..core import theme as theme_mod
from ..gl.vectors import OverlayMesh
from . import chrome
from .md_dialogs import _hint

__all__ = [
    "FrameData", "Rule", "HighlightResult", "HighlightPage", "apply_rules",
    "channel_backend", "grid_points_note", "landscape_status_note",
    "sphere_mesh", "icosphere", "radius_for_volume",
    "DESCRIPTORS", "COLOUR_MODES", "CHANNEL_MODES", "VOID_COLOUR",
    "CHANNEL_COLOUR",
]

DESCRIPTORS = ("CN", "former CN", "BVS", "phi", "Qn", "mean bond angle",
               "tetrahedral deviation", "smallest ring",
               "distance to channel")
SENSES = ("≤", "≥", "=")
COLOUR_MODES = ("element", "shell from the atom",
                "bond-valence sum", "coordination number",
                "stereoactivity (phi)", "Qn", "modifier-rich O",
                "modifier density", "channel membership",
                "microscopic basicity")
# the hot map's surface: a warm colour the element palettes keep clear of
HOTMAP_COLOUR = (0.95, 0.35, 0.15)
HOTMAP_ALPHA = 0.30
HOTMAP_RADIUS_ANG = 4.0           # the count radius of the colour mode
CHANNEL_MODES = ("by charge", "by modifier density", "by voids")

# Presentation choices of the overlays (an orange the element palettes do
# not use for the voids, a blue for the channels) and of the dimming: the
# dimmed atom keeps a fifth of its colour over the background and 40 % of
# its radius, a bond 35 %; measured on the prototype as what keeps the box
# readable while the kept atoms stand out.
# the atom radii a void rule can subtract: key -> (menu text, words)
VOID_RADII = {
    "ionic": ("ionic (Shannon)", "Shannon's ionic radii by the model's "
              "oxidation states"),
    "vdw": ("van der Waals", "FACET's van der Waals radii"),
    "none": ("none", "no atom radius, spheres to the atom centres"),
}
# at most this many void spheres are drawn (the largest first; the note
# states how many passed the volume), 80 triangles each: a drawing choice
VOID_DRAW_BUDGET = 4000
VOID_COLOUR = (0.98, 0.72, 0.20)
VOID_ALPHA = 0.38
CHANNEL_COLOUR = (0.20, 0.55, 0.85)
CHANNEL_ALPHA = 0.30
DIM_KEEP = 0.20
DIM_ATOM_RADIUS = 0.40
DIM_BOND_RADIUS = 0.35
# the default dimming, as a fraction of the way to the background: 0 leaves
# the atoms as they are, 1 fades them to the background entirely (the
# Highlight page's slider, 0 to 100 %)
DIM_DEFAULT = 1.0 - DIM_KEEP
# the atoms within this of a drawn void's surface are its lining: what a
# void rule keeps when the rest is left out
VOID_LINING_ANG = 0.5
NEUTRAL = (0.60, 0.62, 0.66)       # theme.ramp's colour for NaN

# The meshing policy of a by-charge channel surface: METHOD CHOICES about
# what is drawn, never about what is measured -- the region count, volume
# fraction and percolation always come from the full grid. Only the regions
# of at least CHANNEL_MESH_MIN_VOLUME_ANG3 (the "draw >=" field of the rule
# overrides it per rule; the largest region is always drawn) are meshed, and
# the isosurface grid is subsampled (volume.isosurface's step) by the
# smallest step whose predicted triangle count fits the budget. Why: on the
# 15 486-atom C-N glass the full 0.5 Å mismatch surface was 1 277 322
# triangles and took 74 s of marching tetrahedra, and every render that
# re-sorts transparency walked all of them. The triangle estimate is
# straddling cubes x CHANNEL_MESH_TRIANGLES_PER_CUBE (marching tetrahedra
# on that frame gave 6.17 triangles per straddling cube at step 1, 5.29 at
# step 2 and 4.44 at step 3; the step-1 value is the measured upper bound,
# so the budget holds at whatever step is chosen).
CHANNEL_MESH_MIN_VOLUME_ANG3 = 8.0
CHANNEL_MESH_TRIANGLE_BUDGET = 150_000
CHANNEL_MESH_TRIANGLES_PER_CUBE = 6.2
CHANNEL_MESH_MAX_STEP = 4

# What the landscape engine measured through cKDTree.sparse_distance_matrix
# and the exponential: "about 7-8 million (grid point, anion image) pairs a
# second" (facet.core.md_channels module docstring). Used only for the
# "about N s" of the count line while a landscape is being computed.
LANDSCAPE_PAIRS_PER_S = 7.5e6


# ---------------------------------------------------------------------------
# geometry of the overlays
# ---------------------------------------------------------------------------

def icosphere(subdivisions: int = 1) -> np.ndarray:
    """Unit-sphere triangles, (n, 3, 3): 20 faces, times four per level."""
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
         (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    tri = v[np.array(f)]
    for _ in range(int(subdivisions)):
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        for m in (ab, bc, ca):
            m /= np.linalg.norm(m, axis=1)[:, None]
        tri = np.concatenate([np.stack([a, ab, ca], 1),
                              np.stack([ab, b, bc], 1),
                              np.stack([ca, bc, c], 1),
                              np.stack([ab, bc, ca], 1)])
    return tri


def sphere_mesh(centres, radii, colour=VOID_COLOUR, alpha: float = VOID_ALPHA,
                *, label: str = "voids", subdivisions: int = 1) -> OverlayMesh:
    """One OverlayMesh of icospheres (one part per sphere, so the slab
    filter keeps or drops a whole sphere)."""
    unit = icosphere(subdivisions)
    faces = len(unit)
    centres = np.asarray(centres, np.float64).reshape(-1, 3)
    radii = np.asarray(radii, np.float64).reshape(-1)
    if not len(centres):
        return OverlayMesh(np.zeros((0, 3), np.float32),
                           np.zeros((0, 3), np.float32), tuple(colour),
                           float(alpha), label=label)
    tri = unit[None, :, :, :] * radii[:, None, None, None] \
        + centres[:, None, None, :]
    vertices = tri.reshape(-1, 3).astype(np.float32)
    normals = np.tile(unit.reshape(-1, 3), (len(centres), 1)).astype(np.float32)
    part = np.repeat(np.arange(len(centres)), faces).astype(np.int32)
    return OverlayMesh(vertices, normals, tuple(colour), float(alpha),
                       anchors=centres.astype(np.float32), part_of=part,
                       label=label)


def radius_for_volume(volume_ang3: float) -> float:
    """The radius of a sphere of this volume."""
    return (3.0 * max(float(volume_ang3), 0.0) / (4.0 * math.pi)) ** (1.0 / 3)


def _triangle_mesh(vertices, faces, normals, colour, alpha, label
                   ) -> OverlayMesh:
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, int)
    if not len(faces) or not len(vertices):
        return OverlayMesh(np.zeros((0, 3), np.float32),
                           np.zeros((0, 3), np.float32), tuple(colour),
                           float(alpha), label=label)
    flat = faces.reshape(-1)
    normals = np.asarray(normals, np.float32)
    n = (normals[flat] if len(normals) == len(vertices)
         else np.zeros((len(flat), 3), np.float32))
    return OverlayMesh(vertices[flat], n, tuple(colour), float(alpha),
                       label=label)


# ---------------------------------------------------------------------------
# what the rules read of one frame
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class FrameData:
    """One frame as the rules read it: its table at v_bond, the per-atom
    descriptors, and, computed once on demand, the empty spheres and the
    bond-valence landscapes.

    ``former_cn``, ``qn`` and ``modifier_count`` are None without formers
    (a glass result's formers, or those ticked in Setup): former CN counts
    an anion's bonds to the formers (1: non-bridging), Qn a former cation's
    bonds to anions bonded to two or more formers (NaN off the formers),
    modifier_count an anion's bonds to the cations that are not formers
    (NaN off the anions).
    """

    k: int
    frame: object
    table: object
    results: object
    bonds: object
    elements: np.ndarray
    ox_atom: np.ndarray
    v_bond: float
    formers: frozenset
    cn: np.ndarray
    bvs: np.ndarray
    phi: np.ndarray
    former_cn: np.ndarray | None = None
    qn: np.ndarray | None = None
    modifier_count: np.ndarray | None = None
    params: object = None
    # set from outside when known: the smallest ring each atom is in (from
    # a rings analysis), the distance of each atom to the nearest channel
    # grid point (from a channels rule), and the atom shells are read from
    ring_size: np.ndarray | None = None
    channel_distance: np.ndarray | None = None
    shell_centre: int | None = None
    # the scale of basicity moderating parameters the microscopic-basicity
    # colouring reads (facet.core.basicity.SCALE_KEYS); set from the
    # window's Tools tab, FACET's default otherwise
    basicity_scale: str = "pauling"
    _basicity: dict = field(default_factory=dict, repr=False)
    _angles: tuple | None = field(default=None, repr=False)
    _spheres: object = field(default=None, repr=False)
    _landscapes: dict = field(default_factory=dict, repr=False)
    _void_regions: dict = field(default_factory=dict, repr=False)
    _regions: dict = field(default_factory=dict, repr=False)
    _channel_meshes: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_frame(cls, frame, table, *, k: int = 0,
                   v_bond: float | None = None, formers=(), ox_atom=None,
                   params=None, glass=None) -> "FrameData":
        """From a frame and its ``bulk.ValenceTable``; ``v_bond`` None takes
        FACET's default; ``glass`` (a result with ``formers``) names the
        formers when ``formers`` is empty."""
        from ..core import bv, md_model

        if not isinstance(table, bulk.ValenceTable):
            raise ValueError("FrameData needs the frame's bulk.ValenceTable "
                             "(a frame view drawn without bonds has none)")
        v_bond = bv.V_BOND_DEFAULT if v_bond is None else float(v_bond)
        if not formers and glass is not None:
            formers = getattr(glass, "formers", None) or ()
        former_set = frozenset(str(s) for s in formers
                               if isinstance(s, str))
        symbols = np.asarray(frame.elements)
        if ox_atom is None:
            ox_atom = md_model.model_oxidation(frame.species).per_atom(symbols)
        results = bulk.at_threshold(table, v_bond)
        bonds = bulk.bonds_at(table, v_bond)
        n = frame.n_atoms
        cn = np.asarray(results.cn, np.int64)
        former_cn = qn = modifiers = None
        if former_set:
            cation = np.asarray(bonds.cation, np.int64)
            anion = np.asarray(bonds.anion, np.int64)
            to_former = np.isin(symbols[cation], sorted(former_set))
            former_cn = np.bincount(anion[to_former], minlength=n)
            bridging = former_cn >= 2
            bridge = to_former & bridging[anion]
            n_bridge = np.bincount(cation[bridge], minlength=n)
            is_former = np.isin(symbols, sorted(former_set))
            qn = np.where(is_former, n_bridge, np.nan)
            count = np.bincount(anion[~to_former], minlength=n)
            modifiers = np.where(np.asarray(table.is_anion), count, np.nan)
        return cls(int(k), frame, table, results, bonds, symbols,
                   np.asarray(ox_atom), v_bond, former_set, cn,
                   np.asarray(results.bvs_vu), np.asarray(results.phi),
                   former_cn, qn, modifiers, params)

    def basicity_values(self, scale: str | None = None) -> np.ndarray:
        """The microscopic optical basicity of every atom at this frame's
        v_bond on ``scale`` (the frame data's own when None): lambda for an
        oxide or fluoride anion whose cations the scale covers, NaN for
        every other atom (``facet.core.basicity.frame_basicity``). Computed
        once per scale."""
        from ..core import basicity as basicity_mod

        key = str(scale or self.basicity_scale)
        found = self._basicity.get(key)
        if found is None:
            found = basicity_mod.frame_basicity(
                self.table, key, v_bond=self.v_bond,
                formers=sorted(self.formers) or None)
            self._basicity[key] = found
        return found.value

    @classmethod
    def from_view(cls, view, formers=(), glass=None) -> "FrameData":
        """From a :class:`facet.ui.md_jobs.FrameView` drawn with bonds."""
        return cls.from_frame(view.frame, view.table, k=view.k,
                              v_bond=view.v_bond_vu, formers=formers,
                              ox_atom=view.ox_atom, params=view.params,
                              glass=glass)

    @property
    def n_atoms(self) -> int:
        return int(self.elements.shape[0])

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(self.frame.species)

    @property
    def cations(self) -> tuple[str, ...]:
        return tuple(s for s in self.species
                     if int(self.ox_atom[self.elements == s][0]) > 0)

    @property
    def anions(self) -> tuple[str, ...]:
        return tuple(s for s in self.species
                     if int(self.ox_atom[self.elements == s][0]) < 0)

    @property
    def modifiers(self) -> tuple[str, ...]:
        """The cations that are not formers (every cation without formers)."""
        out = tuple(s for s in self.cations if s not in self.formers)
        return out or self.cations

    def spheres_with(self, radii: str = "ionic"):
        """The empty spheres of the frame under one choice of atom radii
        (:data:`VOID_RADII`), computed once per choice and refined to the
        local maxima of the clearance (md_order.empty_spheres)."""
        radii = str(radii or "ionic")
        if radii not in VOID_RADII:
            raise ValueError(f"radii {radii!r}: one of {tuple(VOID_RADII)}")
        if not isinstance(self._spheres, dict):
            self._spheres = {}
        if radii not in self._spheres:
            if radii == "ionic":
                table, source = md_order.ionic_radii_ang(self.frame, self.ox_atom)
            elif radii == "vdw":
                table, source = (md_order.vdw_radii_ang(self.frame),
                                 md_order.VDW_RADII_SOURCE)
            else:
                table = {s: 0.0 for s in self.frame.species}
                source = "no atom radius: the spheres reach the atom centres"
            self._spheres[radii] = md_order.empty_spheres(
                self.frame, table, radii_source=source)
        return self._spheres[radii]

    @property
    def spheres(self):
        """The empty spheres under ionic radii, the default."""
        return self.spheres_with("ionic")

    def sphere_centres_ang(self, radii: str = "ionic") -> np.ndarray:
        box = np.asarray(self.frame.box_ang, np.float64)
        return np.asarray(self.spheres_with(radii).centre_frac, np.float64) @ box

    def descriptor(self, name: str) -> np.ndarray:
        """The per-atom values of a descriptor of :data:`DESCRIPTORS`
        (ValueError when it needs the formers and there are none)."""
        table = {"CN": self.cn, "former CN": self.former_cn,
                 "BVS": self.bvs, "phi": self.phi, "Qn": self.qn}
        if name in ("mean bond angle", "tetrahedral deviation"):
            mean, dev = self.angle_stats()
            return mean if name == "mean bond angle" else dev
        if name == "smallest ring":
            if self.ring_size is None:
                raise ValueError("smallest ring needs a rings analysis run "
                                 "on this model (Setup > Rings)")
            return np.asarray(self.ring_size, np.float64)
        if name == "distance to channel":
            if self.channel_distance is None:
                raise ValueError("distance to channel needs a channels rule "
                                 "by charge that is on")
            return np.asarray(self.channel_distance, np.float64)
        if name not in table:
            raise ValueError(f"descriptor {name!r}: one of {DESCRIPTORS}")
        values = table[name]
        if values is None:
            raise ValueError(f"{name} needs the network formers (none are "
                             "named)")
        return np.asarray(values, np.float64)

    def angle_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Per atom, the mean angle between its bonds and the mean
        deviation from the tetrahedral angle, in degrees; NaN with fewer
        than two bonds. From the bonds the threshold counts."""
        if self._angles is not None:
            return self._angles
        n = self.n_atoms
        mean = np.full(n, np.nan)
        dev = np.full(n, np.nan)
        bonds = self.bonds
        cation = np.asarray(bonds.cation, np.int64)
        anion = np.asarray(bonds.anion, np.int64)
        vec = np.asarray(bonds.vec_ang, np.float64)
        # every bond seen from both ends
        centre = np.concatenate([cation, anion])
        arms = np.concatenate([vec, -vec])
        order = np.argsort(centre, kind="stable")
        centre, arms = centre[order], arms[order]
        starts = np.searchsorted(centre, np.arange(n + 1))
        for row in range(n):
            lo, hi = starts[row], starts[row + 1]
            if hi - lo < 2:
                continue
            v = arms[lo:hi]
            norms = np.linalg.norm(v, axis=1)
            ok = norms > 0
            v, norms = v[ok], norms[ok]
            if len(v) < 2:
                continue
            unit = v / norms[:, None]
            cos = np.clip(unit @ unit.T, -1.0, 1.0)
            iu = np.triu_indices(len(v), 1)
            angles = np.degrees(np.arccos(cos[iu]))
            mean[row] = angles.mean()
            dev[row] = np.abs(angles - 109.4712206).mean()
        self._angles = (mean, dev)
        return self._angles

    def neighbour_count(self, element: str) -> np.ndarray:
        """Per atom, how many of its bonded neighbours are ``element``
        ('any': all of them)."""
        bonds = self.bonds
        cation = np.asarray(bonds.cation, np.int64)
        anion = np.asarray(bonds.anion, np.int64)
        n = self.n_atoms
        if element == "any":
            return (np.bincount(cation, minlength=n)
                    + np.bincount(anion, minlength=n)).astype(np.float64)
        sym = self.elements
        to_a = np.bincount(cation[sym[anion] == element], minlength=n)
        to_c = np.bincount(anion[sym[cation] == element], minlength=n)
        return (to_a + to_c).astype(np.float64)

    def shells_from(self, centre: int, depth: int = 3) -> np.ndarray:
        """The shell of each atom about ``centre`` (0 at the centre, -1
        beyond ``depth``), over the bonds the threshold counts."""
        from ..core.md_environment import neighbour_shells
        _keep, shell = neighbour_shells(self.bonds, self.n_atoms, centre, depth)
        return shell

    def modifier_counts(self, elements, r_ang: float) -> np.ndarray:
        """Modifier atoms within ``r_ang`` of each atom
        (:func:`facet.core.md_hotmap.modifier_counts`), kept per
        (elements, r)."""
        from ..core import md_hotmap

        key = ("counts", tuple(sorted(str(e) for e in elements)), float(r_ang))
        if key not in self._landscapes:
            self._landscapes[key] = md_hotmap.modifier_counts(
                self.frame, key[1], float(r_ang))
        return self._landscapes[key]

    def hotmap(self, elements, grid_spacing_ang: float, sigma_ang: float):
        """The smoothed modifier density grid
        (:func:`facet.core.md_hotmap.modifier_density_grid`), kept per
        (elements, spacing, sigma)."""
        from ..core import md_hotmap

        key = ("hotmap", tuple(sorted(str(e) for e in elements)),
               float(grid_spacing_ang), float(sigma_ang))
        if key not in self._landscapes:
            self._landscapes[key] = md_hotmap.modifier_density_grid(
                self.frame, key[1], grid_spacing_ang=float(grid_spacing_ang),
                sigma_ang=float(sigma_ang))
        return self._landscapes[key]

    def probe_sum_spread(self, probe: str, r_cut_ang: float):
        """(mean, standard deviation, count) of the landscape's sum at the
        probe element's own atoms (:func:`facet.core.md_channels.probe_sums`),
        kept per (probe, r_cut); (nan, nan, 0) without such an atom."""
        from ..core import md_channels

        key = ("spread", str(probe), float(r_cut_ang))
        if key not in self._landscapes:
            rows = self.elements == probe
            if rows.any():
                probe_ox = int(self.ox_atom[rows][0])
            else:
                probe_ox = element_data.COMMON_OX.get(str(probe))
            if probe_ox is None or not rows.any():
                self._landscapes[key] = (float("nan"), float("nan"), 0)
            else:
                sums = md_channels.probe_sums(
                    self.frame, self.ox_atom, str(probe), probe_ox,
                    r_cut_ang=float(r_cut_ang), params=self.params)
                self._landscapes[key] = (float(sums.mean()), float(sums.std()),
                                         int(sums.size))
        return self._landscapes[key]

    def landscape(self, probe: str, grid_spacing_ang: float,
                  r_cut_ang: float, cancelled=None, target: str = "formal"):
        """The bond-valence landscape of ``probe`` on this frame
        (:func:`facet.core.md_channels.bv_landscape`), kept per
        (probe, spacing, r_cut). ``cancelled`` (a ``() -> bool`` hook) is
        forwarded to the engine, which polls it once per grid chunk; None
        takes the cancel of the job running on this thread
        (:func:`facet.ui.md_jobs.current_cancelled`), so a highlight worker
        stops when its job is cancelled."""
        from ..core import md_channels

        key = (str(probe), float(grid_spacing_ang), float(r_cut_ang),
               str(target))
        if key not in self._landscapes:
            rows = self.elements == probe
            if rows.any():
                probe_ox = int(self.ox_atom[rows][0])
            else:
                probe_ox = element_data.COMMON_OX.get(str(probe))
                if probe_ox is None:
                    raise ValueError(f"no oxidation state for the probe "
                                     f"{probe}")
            self._landscapes[key] = md_channels.bv_landscape(
                self.frame, self.ox_atom, str(probe), probe_ox,
                grid_spacing_ang=float(grid_spacing_ang),
                r_cut_ang=float(r_cut_ang), params=self.params,
                cancelled=_cancel_hook(cancelled), target=str(target))
        return self._landscapes[key]

    def regions(self, probe: str, grid_spacing_ang: float, r_cut_ang: float,
                delta_vu: float, cancelled=None, target: str = "formal"):
        """The accessible regions of :meth:`landscape` at ``delta_vu``
        (:func:`facet.core.md_channels.accessible_regions`), kept per
        (probe, spacing, r_cut, Delta) -- the landscape and its regions do
        not depend on v_bond, so a v_bond move reads them from here."""
        from ..core import md_channels

        key = (str(probe), float(grid_spacing_ang), float(r_cut_ang),
               float(delta_vu), str(target))
        if key not in self._regions:
            landscape = self.landscape(probe, grid_spacing_ang, r_cut_ang,
                                       cancelled, target=target)
            try:
                self._regions[key] = md_channels.accessible_regions(
                    landscape, float(delta_vu),
                    cancelled=_cancel_hook(cancelled))
            except TypeError:
                # an older md_channels without the hook: runs through
                self._regions[key] = md_channels.accessible_regions(
                    landscape, float(delta_vu))
        return self._regions[key]

    def void_regions(self, probe_radius_ang: float, grid_spacing_ang: float,
                     cancelled=None):
        """The void regions of this frame
        (:func:`facet.core.md_channels.void_regions` on :attr:`spheres`,
        lining distance 0), kept per (probe radius, spacing);
        :class:`ChannelUnavailable` when the build lacks them. ``cancelled``
        is forwarded as :meth:`landscape` forwards it."""
        from ..core import md_channels

        ok, reason = channel_backend("by voids")
        if not ok:
            raise ChannelUnavailable(reason)
        key = (float(probe_radius_ang), float(grid_spacing_ang))
        if key not in self._void_regions:
            try:
                self._void_regions[key] = md_channels.void_regions(
                    self.frame, self.spheres, probe_radius_ang=key[0],
                    grid_spacing_ang=key[1], lining_distance_ang=0.0,
                    cancelled=_cancel_hook(cancelled))
            except TypeError as error:
                raise ChannelUnavailable(
                    "md_channels.void_regions takes other arguments than "
                    "this page passes (frame, spheres, probe_radius_ang, "
                    f"grid_spacing_ang, lining_distance_ang): {error}"
                    ) from None
        return self._void_regions[key]


def _thousands(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


def _cancel_hook(cancelled):
    """``cancelled`` itself, or, when None, the cancel of the job whose work
    runs on this thread (:func:`facet.ui.md_jobs.current_cancelled`). The
    highlight job's work reaches the landscapes through FrameData without
    passing a hook down, so the default is what lets its cancel stop a grid
    computation mid-chunk."""
    if cancelled is not None:
        return cancelled
    try:
        from .md_jobs import current_cancelled
    except ImportError:
        return None
    return current_cancelled()


def grid_points_note(fd, grid_spacing_ang: float) -> str:
    """What the spacing asks for over the frame shown, before anything runs:
    the grid's point count, or md_channels's refusal when the count exceeds
    its cap (:data:`facet.core.md_channels.MAX_GRID_POINTS`)."""
    if fd is None:
        return ""
    try:
        from ..core import md_channels
    except ImportError:
        return ""
    guard = getattr(md_channels, "grid_guard", None)
    if not callable(guard):
        return ""
    try:
        shape, n_points, refusal = guard(fd.frame.box_ang,
                                         float(grid_spacing_ang))
    except (ValueError, TypeError):
        return ""
    if refusal:
        return refusal
    return (f"grid {shape[0]} × {shape[1]} × {shape[2]} = "
            f"{_thousands(n_points)} points at {grid_spacing_ang:g} Å")


def landscape_status_note(fd, rules, previous: str = "") -> str:
    """What the count line says while a by-charge landscape is being
    computed: the grid's point count and about how long on this machine
    (:data:`LANDSCAPE_PAIRS_PER_S`, the engine docstring's measured rate),
    with the stale result marked as previous. Empty when every enabled
    by-charge rule's landscape is already cached on ``fd``."""
    if fd is None:
        return ""
    try:
        from ..core import md_channels
    except ImportError:
        return ""
    for rule in rules:
        if not (rule.enabled and rule.kind == "channels"
                and rule.channel_by == "by charge"):
            continue
        probe = rule.probe or (fd.modifiers[0] if fd.modifiers else "")
        if not probe or rule.grid_spacing_ang <= 0 or rule.r_cut_ang <= 0:
            continue
        key = (str(probe), float(rule.grid_spacing_ang),
               float(rule.r_cut_ang), str(rule.target))
        if key in fd._landscapes:
            continue
        try:
            _shape, n_points, refusal = md_channels.grid_guard(
                fd.frame.box_ang, float(rule.grid_spacing_ang))
        except (ValueError, TypeError, AttributeError):
            continue
        if refusal:
            continue
        anions = int((np.asarray(fd.ox_atom) < 0).sum())
        volume = float(fd.frame.volume_ang3)
        per_point = max(1.0, anions / max(volume, 1e-9)
                        * 4.0 / 3.0 * math.pi * float(rule.r_cut_ang) ** 3)
        seconds = n_points * per_point / LANDSCAPE_PAIRS_PER_S
        about = f"{seconds:.0f} s" if seconds >= 1.5 else "a second"
        note = (f"computing the {probe} landscape: "
                f"{_thousands(n_points)} grid points, about {about} on "
                "this machine…")
        if previous:
            note += f" · previous: {previous}"
        return note
    return ""


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """One highlight rule. ``kind`` is 'show-only', 'voids' or 'channels'.

    show-only: ``element`` ('any' or a symbol), ``descriptor``
    (:data:`DESCRIPTORS`), ``sense`` ('≤' or '≥'), ``value``.
    voids: ``min_volume_ang3``; ``min_elongation`` None or 1.0 keeps every
    shape, above 1.0 reads the void regions.
    channels: ``channel_by`` (:data:`CHANNEL_MODES`), ``threshold`` (the
    mismatch Δ in v.u. by charge; the modifiers per anion k by modifier
    density; the least elongation by voids), ``probe`` (the mobile ion by
    charge, the modifier element by modifier density), ``grid_spacing_ang``
    (the landscape or union grid) and ``r_cut_ang`` (the valence cutoff by
    charge, the M–anion cutoff by modifier density, the probe radius by
    voids).
    """

    kind: str
    enabled: bool = True
    element: str = "any"
    descriptor: str = "CN"
    sense: str = "≤"
    value: float = 0.0
    neighbour: str = "any"          # neighbours rule: the element counted
    rows: tuple = ()                # rows rule: the frame rows kept
    rows_label: str = ""            # what those rows are ('Si Q3 environment 1')
    min_volume_ang3: float = 0.0
    min_elongation: float | None = None
    radii: str = "ionic"            # voids: the atom radii subtracted (VOID_RADII)
    # hotmap: the elements counted ('' = every modifier), the smoothing
    # sigma, and the level drawn as a multiple of the mean density
    elements: str = ""
    sigma_ang: float = 1.5
    level_x_mean: float = 1.5
    channel_by: str = "by charge"
    threshold: float = 0.0
    probe: str = ""
    grid_spacing_ang: float = 0.0
    r_cut_ang: float = 0.0
    # by charge only: the volume floor of the DRAWN surface (None takes
    # CHANNEL_MESH_MIN_VOLUME_ANG3); a method choice about the picture, the
    # numbers always come from the full grid
    mesh_min_volume_ang3: float | None = None
    # by charge only: what the sum is matched to, 'frame' (the mean sum at
    # the probe's own atoms) or 'formal' (its formal valence); md_channels
    target: str = "frame"


@dataclass(frozen=True)
class HighlightResult:
    """What :func:`apply_rules` drew: the atoms a show-only rule keeps
    (``n_matching``; the frame's atom count without one), the empty
    spheres drawn, the channel regions found and the atoms in them, and
    the count line."""

    n_matching: int
    n_spheres: int
    note: str
    n_regions: int = 0
    n_in_regions: int = 0
    problems: tuple[str, ...] = ()
    keep: np.ndarray | None = None      # the atoms the rules kept, by row


class ChannelUnavailable(ValueError):
    """A channel backend this build does not offer, with the reason."""


def channel_backend(mode: str) -> tuple[bool, str]:
    """Whether :mod:`facet.core.md_channels` offers the channel ``mode``
    in this build, and the reason when not."""
    try:
        from ..core import md_channels
    except ImportError as error:
        return False, f"facet.core.md_channels is not in this build ({error})"
    if mode == "by charge":
        names = ("bv_landscape", "accessible_regions", "landscape_grid")
    elif mode == "by modifier density":
        names = ("modifier_density",)
    elif mode == "by voids":
        names = ("void_regions", "select_regions")
    else:
        return False, f"{mode!r} is not one of {CHANNEL_MODES}"
    lacking = [n for n in names if not callable(getattr(md_channels, n, None))]
    if lacking:
        return False, (f"md_channels.{', '.join(lacking)} is not in this "
                       "build")
    return True, ""


def _straddling_cubes(values: np.ndarray, level: float, step: int) -> int:
    """How many grid cubes of ``values[::step]`` the level crosses
    (periodic), which is what the marching tetrahedra will walk."""
    v = values[::step, ::step, ::step]
    if min(v.shape) < 2:
        return 0
    inside = v <= level
    any_in = inside.copy()
    all_in = inside.copy()
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                corner = np.roll(inside, (-dx, -dy, -dz), axis=(0, 1, 2))
                any_in |= corner
                all_in &= corner
    return int((any_in & ~all_in).sum())


def _choose_step(values: np.ndarray, level: float) -> tuple[int, int]:
    """The smallest isosurface step whose predicted triangle count fits
    :data:`CHANNEL_MESH_TRIANGLE_BUDGET`, and that prediction."""
    per_cube = CHANNEL_MESH_TRIANGLES_PER_CUBE
    estimate = 0
    for step in range(1, CHANNEL_MESH_MAX_STEP + 1):
        estimate = int(_straddling_cubes(values, level, step) * per_cube)
        if estimate <= CHANNEL_MESH_TRIANGLE_BUDGET:
            return step, estimate
        if min(n // (step + 1) for n in values.shape) < 2:
            break
    return min(step, CHANNEL_MESH_MAX_STEP), estimate


def _channel_mesh_arrays(fd: FrameData, rule: Rule, regions,
                         floor_ang3: float):
    """(vertices, flat normals, policy note) of the by-charge surface:
    the mismatch isosurface at Δ restricted to the regions drawn under the
    meshing policy (volume floor, triangle budget; module constants above).
    Kept per (probe, spacing, r_cut, Δ, floor, budget), so a v_bond move or
    a second apply reads the triangles instead of marching the grid again."""
    from ..core import md_channels, volume

    key = (str(rule.probe), float(rule.grid_spacing_ang),
           float(rule.r_cut_ang), float(rule.threshold), float(floor_ang3),
           int(CHANNEL_MESH_TRIANGLE_BUDGET), str(rule.target))
    hit = fd._channel_meshes.get(key)
    if hit is not None:
        return hit
    level = float(rule.threshold)
    volumes = np.asarray(regions.volume_ang3, np.float64)
    # regions are numbered largest first, so those at or above the floor
    # are a leading run; the largest is always drawn
    n_drawn = max(1, int((volumes >= floor_ang3).sum())) \
        if regions.n_regions else 0
    label = np.asarray(regions.label_grid)
    landscape = fd.landscape(rule.probe, rule.grid_spacing_ang,
                             rule.r_cut_ang, target=rule.target)
    grid = md_channels.landscape_grid(landscape, "mismatch")
    # raise the accessible points outside the drawn regions above the level,
    # so their surface is not meshed; the numbers never read this field
    drawn = (label >= 0) & (label < n_drawn)
    grid.values[~drawn & (grid.values <= level)] = level + abs(level) + 1.0
    hook = _cancel_hook(None)
    if hook is not None and hook():
        from ..core.md_channels import AnalysisCancelled

        raise AnalysisCancelled("cancelled before the channel surface was "
                                "meshed")
    step, _ = _choose_step(grid.values, level)
    vertices, faces, normals = volume.isosurface(grid, level, step=step)
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, int)
    if len(faces) and len(vertices):
        flat = faces.reshape(-1)
        tri_vertices = vertices[flat]
        normals = np.asarray(normals, np.float32)
        tri_normals = (normals[flat] if len(normals) == len(vertices)
                       else np.zeros((len(flat), 3), np.float32))
    else:
        tri_vertices = np.zeros((0, 3), np.float32)
        tri_normals = np.zeros((0, 3), np.float32)
    drawn_volume = float(volumes[:n_drawn].sum()) if n_drawn else 0.0
    open_volume = float(volumes.sum())
    if regions.n_regions == 0:
        policy = ""
    elif n_drawn == regions.n_regions:
        policy = f"all {regions.n_regions} region(s) drawn"
    else:
        share = 100.0 * drawn_volume / open_volume if open_volume else 0.0
        policy = (f"the {n_drawn} of at least {floor_ang3:g} Å³ drawn "
                  f"({share:.0f} % of the open volume); smaller ones left "
                  "out of the picture, not the numbers")
    if step > 1 and policy:
        policy += (f"; surface meshed at grid step {step} to fit the "
                   f"{_thousands(CHANNEL_MESH_TRIANGLE_BUDGET)}-triangle "
                   "budget")
    out = (tri_vertices, tri_normals, policy)
    fd._channel_meshes[key] = out
    return out


def _channels_by_charge(fd: FrameData, rule: Rule):
    """(region of each atom, n_regions, mesh, note) by the bond-valence
    landscape of the probe and its accessible regions at Δ. The NUMBERS
    (region count, volume fraction, percolation, membership) come from the
    full grid; only the drawn surface follows the meshing policy
    (:data:`CHANNEL_MESH_MIN_VOLUME_ANG3` and the constants beside it)."""
    probe = rule.probe or (fd.modifiers[0] if fd.modifiers else "")
    if not probe:
        raise ChannelUnavailable("by charge needs a probe ion: the frame "
                                 "holds no cation")
    if rule.grid_spacing_ang <= 0 or rule.r_cut_ang <= 0:
        raise ChannelUnavailable("by charge needs a grid spacing and an "
                                 "r_cut above 0 (both are shown in the rule)")
    rule = replace(rule, probe=probe)
    regions = fd.regions(probe, rule.grid_spacing_ang, rule.r_cut_ang,
                         rule.threshold, target=rule.target)
    floor = CHANNEL_MESH_MIN_VOLUME_ANG3 \
        if rule.mesh_min_volume_ang3 is None \
        else max(float(rule.mesh_min_volume_ang3), 0.0)
    tri_vertices, tri_normals, policy = _channel_mesh_arrays(
        fd, rule, regions, floor)
    mesh = OverlayMesh(tri_vertices, tri_normals, tuple(CHANNEL_COLOUR),
                       float(CHANNEL_ALPHA), label="channels")
    landscape = fd.landscape(probe, rule.grid_spacing_ang, rule.r_cut_ang,
                             target=rule.target)
    fraction = float(regions.accessible_fraction)
    spans = "".join(axis for axis, on in zip("abc", regions.percolates)
                    if bool(on))
    region_of_atom = np.asarray(regions.region_of_atom)
    probe_rows = np.asarray(fd.elements) == probe
    n_probe = int(probe_rows.sum())
    inside = int((region_of_atom[probe_rows] >= 0).sum()) if n_probe else 0
    sums = landscape.probe_sums_vu
    if landscape.target_mode == "frame" and sums is not None and len(sums):
        target_text = (f"target {landscape.target_vu:.3g} v.u. = the mean sum "
                       f"at the frame's {n_probe} {probe} atoms (spread "
                       f"{float(np.std(sums)):.2g}; formal {abs(landscape.probe_ox):g})")
    else:
        target_text = f"target {landscape.target_vu:.3g} v.u. (the formal valence)"
        if sums is not None and len(sums):
            target_text += (f"; the frame's {n_probe} {probe} atoms sum to "
                            f"{float(np.mean(sums)):.3g} ± {float(np.std(sums)):.2g}")
    note = (f"channels by charge: {landscape.probe_label} probe, |V − target| "
            f"≤ {rule.threshold:g} v.u., {target_text}, on a "
            f"{rule.grid_spacing_ang:g} Å grid (r_cut {rule.r_cut_ang:g} Å): "
            f"{regions.n_regions} region(s), {100 * fraction:.1f} % of the box"
            + (f", spanning {spans}" if spans else ", none spanning the box")
            + (f"; {inside} of {n_probe} {probe} atoms lie inside a region"
               if n_probe else "")
            + (f"; {policy}" if policy else ""))
    return np.asarray(regions.region_of_atom), int(regions.n_regions), mesh, \
        note


def _channels_by_modifier_density(fd: FrameData, rule: Rule):
    """(cluster of each atom, n_clusters, None, note) by the modifier
    counts around the anions within the M–anion cutoff given."""
    from ..core import md_channels

    modifier = rule.probe or (fd.modifiers[0] if fd.modifiers else "")
    if not modifier:
        raise ChannelUnavailable("by modifier density needs a modifier "
                                 "element: the frame holds no cation")
    if rule.r_cut_ang <= 0:
        raise ChannelUnavailable("by modifier density needs the M–anion "
                                 "cutoff in Å (shown in the rule)")
    k = max(int(round(rule.threshold)), 1)
    anions = fd.anions
    if not anions:
        raise ChannelUnavailable("the frame holds no anion")
    cutoffs = {(modifier, a): float(rule.r_cut_ang) for a in anions}
    sources = {key: "given on the Highlight page" for key in cutoffs}
    pairs = bulk.find_pairs(fd.frame, float(rule.r_cut_ang) + 0.01)
    density = md_channels.modifier_density(
        fd.frame, pairs, modifiers={modifier}, anions=set(anions),
        cutoffs_ang=cutoffs, cutoff_sources=sources, k_rich=k,
        bonds=fd.bonds, formers=fd.formers or None)
    label = np.asarray(density.label)
    note = (f"channels by modifier density: anions with ≥ {k} {modifier} "
            f"within {rule.r_cut_ang:g} Å: {density.n_rich} of "
            f"{density.n_anions}, in {density.n_clusters} cluster(s)")
    return label, int(density.n_clusters), None, note


def _channels_by_voids(fd: FrameData, rule: Rule):
    """(region of each atom, n_regions kept, None, note) by the void regions
    (``md_channels.void_regions`` with the probe radius of the rule) of at
    least the rule's elongation (``select_regions``)."""
    from ..core import md_channels

    regions = fd.void_regions(rule.r_cut_ang, rule.grid_spacing_ang or 0.5)
    try:
        chosen = md_channels.select_regions(
            regions, min_volume_ang3=float(rule.min_volume_ang3),
            min_elongation=float(rule.threshold))
    except TypeError as error:
        raise ChannelUnavailable("md_channels.select_regions takes other "
                                 "arguments than this page passes (regions, "
                                 f"min_volume_ang3, min_elongation): {error}"
                                 ) from None
    region_of_atom = np.asarray(getattr(regions, "region_of_atom"))
    kept = np.asarray(getattr(chosen, "regions", chosen), int).reshape(-1)
    member = np.full(region_of_atom.shape, -1, int)
    for new, old in enumerate(kept):
        member[region_of_atom == old] = new
    note = (f"channels by voids: regions of volume ≥ "
            f"{rule.min_volume_ang3:g} Å³ and elongation ≥ "
            f"{rule.threshold:g} (probe {rule.r_cut_ang:g} Å): {len(kept)} "
            f"of {regions.n_regions}")
    return member, int(len(kept)), None, note


_CHANNEL_BACKENDS = {"by charge": _channels_by_charge,
                     "by modifier density": _channels_by_modifier_density,
                     "by voids": _channels_by_voids}


# ---------------------------------------------------------------------------
# colouring
# ---------------------------------------------------------------------------

def _element_colours(scene, theme) -> np.ndarray:
    tokens = sorted(set(scene.atom_element))
    palette = {t: np.array(theme.element_color(t), np.float32) for t in tokens}
    return np.array([palette[e] for e in scene.atom_element],
                    np.float32).reshape(-1, 3)


def _ramp_over(rows_values: dict, lo: float, hi: float, base: np.ndarray,
               sites: np.ndarray, diverging: bool = False) -> np.ndarray:
    """``base`` with the ramp colour of each frame row in ``rows_values``,
    on every scene atom (images included) of that row."""
    out = base.copy()
    if not rows_values:
        return out
    colour_of = {row: np.array(theme_mod.color_for_value(v, lo, hi, diverging),
                               np.float32) for row, v in rows_values.items()}
    for i, row in enumerate(np.asarray(sites, int)):
        c = colour_of.get(int(row))
        if c is not None:
            out[i] = c
    return out


def colour_atoms(scene, fd: FrameData, colour_by: str, theme=None,
                 membership: np.ndarray | None = None) -> str:
    """Set ``scene.atom_color`` by ``colour_by`` (:data:`COLOUR_MODES`);
    returns a note when the mode cannot be drawn on this frame (the atoms
    then keep their element colours)."""
    theme = theme or getattr(scene, "theme", None) or theme_mod.Theme()
    base = _element_colours(scene, theme)
    sites = np.asarray(scene.atom_site, int)
    cations = np.nonzero(~np.asarray(fd.table.is_anion))[0]
    anions = np.nonzero(np.asarray(fd.table.is_anion))[0]
    note = ""
    legend = None                       # (title, lo, hi) of the colour bar
    if colour_by == "element":
        colours = base
    elif colour_by == "shell from the atom":
        centre = fd.shell_centre
        if centre is None:
            note = ("shell from the atom needs an atom under investigation "
                    "(click one in the frame)")
            colours = base
        else:
            shell = fd.shells_from(int(centre), 3)
            rows = np.nonzero(shell >= 0)[0]
            by_atom = {int(i): float(shell[i]) for i in rows}
            colours = _ramp_over(by_atom, 0.0, 3.0, base, sites)
            legend = ("shell from the atom", 0.0, 3.0)
    elif colour_by in ("bond-valence sum", "coordination number",
                       "stereoactivity (phi)"):
        # the crystal window's modes, as facet.ui.md_scene._value_colours
        mode = theme_mod.ColorMode(colour_by)
        values = {"bond-valence sum": fd.bvs,
                  "coordination number": fd.cn.astype(np.float64),
                  "stereoactivity (phi)": fd.phi}[colour_by]
        by_atom = {int(i): float(values[i]) for i in cations}
        lo, hi = theme_mod.scale_range(by_atom, theme, mode)
        colours = _ramp_over(by_atom, lo, hi, base, sites)
        legend = (colour_by, lo, hi)
    elif colour_by == "Qn":
        if fd.qn is None:
            note = "Qn needs the network formers (none are named)"
            colours = base
        else:
            rows = np.nonzero(np.isfinite(fd.qn))[0]
            by_atom = {int(i): float(fd.qn[i]) for i in rows}
            finite = list(by_atom.values())
            hi = max(max(finite) if finite else 0.0, 1.0)
            colours = _ramp_over(by_atom, 0.0, hi, base, sites)
            legend = ("Qn", 0.0, hi)
    elif colour_by == "modifier-rich O":
        if fd.modifier_count is None:
            note = "modifier-rich O needs the network formers (none are named)"
            colours = base
        else:
            by_atom = {int(i): float(fd.modifier_count[i]) for i in anions}
            finite = list(by_atom.values())
            hi = max(max(finite) if finite else 0.0, 1.0)
            colours = _ramp_over(by_atom, 0.0, hi, base, sites)
            legend = ("modifiers on the anion", 0.0, hi)
    elif colour_by == "modifier density":
        modifiers = list(fd.modifiers)
        if not modifiers:
            note = "modifier density needs a modifier (a cation that is not a former)"
            colours = base
        else:
            counts = fd.modifier_counts(modifiers, HOTMAP_RADIUS_ANG)
            by_atom = {int(i): float(counts[i]) for i in range(fd.n_atoms)}
            hi = max(float(counts.max()) if counts.size else 0.0, 1.0)
            colours = _ramp_over(by_atom, 0.0, hi, base, sites)
            legend = (f"modifiers within {HOTMAP_RADIUS_ANG:g} Å", 0.0, hi)
    elif colour_by == "channel membership":
        if membership is None:
            note = "channel membership needs a channels rule that is on"
            colours = base
        else:
            inside = np.nonzero(membership >= 0)[0]
            n_regions = int(membership.max()) + 1 if len(inside) else 0
            by_atom = {int(i): float(membership[i]) for i in inside}
            colours = _ramp_over(by_atom, 0.0, max(n_regions - 1, 1), base,
                                 sites)
            legend = ("channel region", 0.0, float(max(n_regions - 1, 1)))
    elif colour_by == "microscopic basicity":
        # the anions on the ramp by their lambda (facet.core.basicity, at
        # v_bond on the frame data's scale), the cations in their element
        # colour; a pinned theme range holds, else the frame's own span
        values = fd.basicity_values()
        rows = np.nonzero(np.isfinite(values))[0]
        if not len(rows):
            note = ("microscopic basicity: no anion of this frame has a "
                    "lambda (an oxide or fluoride anion whose cations the "
                    f"{fd.basicity_scale} scale covers)")
            colours = base
        else:
            by_atom = {int(i): float(values[i]) for i in rows}
            lo, hi = theme_mod.scale_range(by_atom, theme,
                                           theme_mod.ColorMode.BASICITY)
            colours = _ramp_over(by_atom, lo, hi, base, sites)
            legend = (f"microscopic basicity λ ({fd.basicity_scale})", lo, hi)
    else:
        raise ValueError(f"colour_by {colour_by!r}: one of {COLOUR_MODES}")
    scene.atom_color = np.asarray(colours, np.float32).reshape(-1, 3)
    # the view's colour bar: what the ramp spans, for every value mode
    scene.value_legend = (None if legend is None
                          else (legend[0], float(legend[1]), float(legend[2]),
                                False))
    return note


# ---------------------------------------------------------------------------
# applying the rules to a scene
# ---------------------------------------------------------------------------

def _keep_of_rule(fd: FrameData, rule: Rule) -> np.ndarray:
    if rule.kind == "rows":
        keep = np.zeros(fd.n_atoms, bool)
        rows = np.asarray([int(r) for r in rule.rows], int)
        keep[rows[(rows >= 0) & (rows < fd.n_atoms)]] = True
        return keep
    if rule.kind == "neighbours":
        values = fd.neighbour_count(rule.neighbour)
    else:
        values = fd.descriptor(rule.descriptor)
    with np.errstate(invalid="ignore"):
        if rule.sense == "=":
            # equal to, for the integer descriptors (CN, former CN, Qn): a
            # rule 'Qn = 3' picks exactly the Q3 silicons
            keep = np.isclose(values, rule.value, atol=1e-6)
        elif rule.sense == "≤":
            keep = values <= rule.value
        else:
            keep = values >= rule.value
    keep &= np.isfinite(values)
    if rule.element != "any":
        keep &= fd.elements == rule.element
    return keep


def _void_lining(scene, fd, centres_ang, radii_ang, atom_radii) -> np.ndarray:
    """Which frame rows have an atom whose surface lies within
    VOID_LINING_ANG of the surface of a drawn void sphere (the drawn
    atoms, images included, so a void across the box edge keeps its
    lining)."""
    from scipy.spatial import cKDTree

    keep = np.zeros(fd.n_atoms, bool)
    positions = np.asarray(scene.atom_position, np.float64)
    if positions.shape[0] == 0 or len(centres_ang) == 0:
        return keep
    rows = np.asarray(scene.atom_site, int)
    r_atom = np.asarray([float(atom_radii.get(str(s), 0.0))
                         for s in scene.atom_element], np.float64)
    reach = float(np.max(radii_ang)) + float(r_atom.max()) + VOID_LINING_ANG
    tree = cKDTree(positions)
    for centre, r in zip(np.asarray(centres_ang, np.float64), radii_ang):
        near = tree.query_ball_point(centre, r + reach)
        if not near:
            continue
        near = np.asarray(near, int)
        d = np.linalg.norm(positions[near] - centre, axis=1)
        close = d - r_atom[near] <= r + VOID_LINING_ANG
        keep[rows[near[close]]] = True
    return keep


def _dim(scene, keep_rows: np.ndarray, theme, amount: float = DIM_DEFAULT) -> None:
    """Fade the atoms outside ``keep_rows`` ``amount`` of the way to the
    background (0 none, 1 all the way) and shrink them in step; the bonds
    with no kept end likewise."""
    amount = float(min(max(amount, 0.0), 1.0))
    if amount <= 0.0:
        return
    own = 1.0 - amount
    bg = np.array(getattr(theme, "background", None)
                  or theme_mod.FALLBACK_BACKGROUND, np.float32)
    keep = keep_rows[np.asarray(scene.atom_site, int)]
    pale = amount * bg + own * scene.atom_color
    scene.atom_color = np.where(keep[:, None], scene.atom_color,
                                pale).astype(np.float32)
    # the default amount reproduces DIM_ATOM_RADIUS; 100 % shrinks further
    atom_factor = max(0.0, 1.0 - amount * (1.0 - DIM_ATOM_RADIUS) / DIM_DEFAULT)
    scene.atom_radius = np.where(keep, scene.atom_radius,
                                 scene.atom_radius * atom_factor
                                 ).astype(np.float32)
    if scene.n_bonds:
        ends = np.asarray(scene.bond_atoms, int)
        bond_keep = keep_rows[ends[:, 0]] | keep_rows[ends[:, 1]]
        for name in ("bond_color_a", "bond_color_b"):
            c = getattr(scene, name)
            setattr(scene, name, np.where(bond_keep[:, None], c,
                                          amount * bg + own * c)
                    .astype(np.float32))
        bond_factor = max(0.0, 1.0 - amount * (1.0 - DIM_BOND_RADIUS) / DIM_DEFAULT)
        scene.bond_radius = np.where(bond_keep, scene.bond_radius,
                                     scene.bond_radius * bond_factor
                                     ).astype(np.float32)


_PER_ATOM = ("atom_position", "atom_radius", "atom_color", "atom_site")
_PER_BOND = ("bond_a", "bond_b", "bond_radius", "bond_color_a",
             "bond_color_b", "bond_valence", "bond_distance", "bond_atoms",
             "bond_cation", "bond_occupancy", "_bond_base_a", "_bond_base_b")


def _renumber_bond_atoms(scene, new_index: np.ndarray) -> None:
    """``bond_atoms`` names scene atoms by index; after atoms are dropped
    or merged the indices move, and ``new_index[old]`` says where (-1 for
    an atom that is gone). Without this the arcs, the bond-valence vectors
    and the picking of a cluster read the wrong atoms, or none."""
    if not scene.n_bonds or len(scene.bond_atoms) == 0:
        return
    ends = np.asarray(scene.bond_atoms, int)
    valid = (ends >= 0) & (ends < len(new_index))
    out = np.where(valid, new_index[np.clip(ends, 0, len(new_index) - 1)], -1)
    scene.bond_atoms = out.astype(np.int32)


def _hide(scene, keep_rows: np.ndarray) -> None:
    """Leave the atoms outside ``keep_rows`` (and the bonds to them) out of
    the scene's arrays; the centre and radius are kept, so the camera
    stays. The bond endpoints are renumbered to the atoms that remain."""
    keep = keep_rows[np.asarray(scene.atom_site, int)]
    new_index = np.full(len(keep), -1, np.int64)
    new_index[keep] = np.arange(int(keep.sum()))
    for name in _PER_ATOM:
        value = getattr(scene, name)
        if len(value) == len(keep):
            setattr(scene, name, np.asarray(value)[keep])
    for name in ("atom_label", "atom_element"):
        value = getattr(scene, name)
        if len(value) == len(keep):
            setattr(scene, name, [v for v, k in zip(value, keep) if k])
    for name in ("atom_shape", "atom_anisotropic", "structure_of"):
        value = getattr(scene, name, None)
        if value is not None and len(value) == len(keep):
            setattr(scene, name, np.asarray(value)[keep])
    scene.atom_index = np.arange(int(keep.sum()), dtype=np.int32)
    if scene.n_bonds:
        ends = np.asarray(scene.bond_atoms, int)
        bond_keep = keep_rows[ends[:, 0]] & keep_rows[ends[:, 1]]
        for name in _PER_BOND:
            value = getattr(scene, name)
            if len(value) == len(bond_keep):
                setattr(scene, name, np.asarray(value)[bond_keep])
        _renumber_bond_atoms(scene, new_index)
    scene.selected_atom = None


def _unwrap_about(scene, frame, centre_row: int) -> None:
    """Move every drawn atom to its minimum image about ``centre_row`` and
    the bond ends with them, then drop the periodic images that now sit on
    their originals. For an isolated environment: a cluster across the box
    edge is drawn as one piece, as the XYZ export writes it."""
    from ..core.md_environment import unwrapped_about

    if scene.n_atoms == 0:
        return
    rows = np.asarray(scene.atom_site, int)
    old = np.asarray(scene.atom_position, float)
    new = unwrapped_about(frame, rows, centre_row)
    if scene.n_bonds:
        for name in ("bond_a", "bond_b"):
            ends = np.asarray(getattr(scene, name), float)
            if len(ends) == 0:
                continue
            d = np.linalg.norm(ends[:, None, :] - old[None, :, :], axis=2)
            j = d.argmin(axis=1)
            ok = d[np.arange(len(ends)), j] < 1e-3
            ends = ends.copy()
            ends[ok] = new[j[ok]]
            setattr(scene, name, ends.astype(np.float32))
    scene.atom_position = new.astype(np.float32)
    # frame the cluster, not the box it was cut from
    scene.center = new.mean(axis=0).astype(np.float32)
    scene.radius = float(np.linalg.norm(new - new.mean(axis=0), axis=1).max()
                         + 2.0)
    # the images of a kept atom now coincide with it: keep each once, and
    # point the bonds of a dropped image at the copy that stays
    _, first, inverse = np.unique(np.round(new, 3), axis=0,
                                  return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    keep = np.zeros(len(new), bool)
    keep[np.sort(first)] = True
    if keep.all():
        return
    survivor = np.full(len(new), -1, np.int64)
    survivor[keep] = np.arange(int(keep.sum()))
    new_index = survivor[first[inverse]]
    _renumber_bond_atoms(scene, new_index)
    for name in _PER_ATOM:
        value = getattr(scene, name)
        if len(value) == len(keep):
            setattr(scene, name, np.asarray(value)[keep])
    for name in ("atom_label", "atom_element"):
        value = getattr(scene, name)
        if len(value) == len(keep):
            setattr(scene, name, [v for v, k in zip(value, keep) if k])
    for name in ("atom_shape", "atom_anisotropic", "structure_of"):
        value = getattr(scene, name, None)
        if value is not None and len(value) == len(keep):
            setattr(scene, name, np.asarray(value)[keep])
    scene.atom_index = np.arange(int(keep.sum()), dtype=np.int32)


def _drop_overlays(scene, labels: Sequence[str]) -> None:
    scene.overlay_meshes = [m for m in scene.overlay_meshes
                            if getattr(m, "label", "") not in labels]


def _channel_distance(fd: FrameData) -> np.ndarray | None:
    """Each atom's distance to the nearest grid point of an accessible
    region found for this frame (the channels rules by charge computed
    them); None when none was computed. The atoms are wrapped into the box
    and the grid is not imaged beyond it, so a distance across the box edge
    reads to the nearer copy inside only."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    points = []
    for key, regions in list(fd._regions.items()):
        landscape = None
        if isinstance(key, tuple) and len(key) >= 5:
            # regions are kept per (probe, spacing, r_cut, delta, target),
            # landscapes per (probe, spacing, r_cut, target)
            landscape = fd._landscapes.get((key[0], key[1], key[2], key[4]))
        if landscape is None and isinstance(key, tuple):
            for lk, lv in fd._landscapes.items():
                if isinstance(lk, tuple) and lk[:3] == key[:3] \
                        and not isinstance(lv, tuple):
                    landscape = lv
                    break
        if landscape is None:
            continue
        grid = np.asarray(regions.label_grid)
        idx = np.argwhere(grid >= 0)
        if not len(idx):
            continue
        spacing = np.asarray(landscape.spacing_ang, float)
        origin = np.asarray(landscape.origin_ang, float)
        points.append(origin[None, :] + idx * spacing[None, :])
    if not points:
        return None
    cloud = np.vstack(points)
    frac = np.asarray(fd.frame.frac, float) % 1.0
    cart = frac @ np.asarray(fd.frame.box_ang, float) + np.asarray(
        getattr(fd.frame, "origin_ang", np.zeros(3)), float)[None, :]
    d, _ = cKDTree(cloud).query(cart)
    return np.asarray(d, np.float64)


def apply_rules(scene, fd: FrameData, rules: Sequence[Rule], *,
                dim_rest: bool = True, colour_by: str = "element",
                theme=None, dim_amount: float = DIM_DEFAULT) -> HighlightResult:
    """Edit ``scene``'s arrays in place by ``rules`` (the disabled ones are
    skipped): colour the atoms by ``colour_by``, dim (``dim_rest``) or
    leave out the atoms outside the show-only rules and outside the
    channel regions, draw the empty spheres of the void rules and the
    mismatch isosurface of a by-charge channel rule. Applying again on the
    same scene replaces the 'voids' and 'channels' overlays rather than
    adding to them. A rule the frame cannot serve is reported in
    ``problems`` and in the note, never raised.
    """
    theme = theme or getattr(scene, "theme", None) or theme_mod.Theme()
    n = fd.n_atoms
    active = [r for r in rules if r.enabled]
    problems: list[str] = []
    parts: list[str] = [f"frame {fd.k} at v_bond {fd.v_bond:g} v.u."]
    _drop_overlays(scene, ("voids", "channels", "hotmap"))

    # channels first: their membership feeds the colour and the keep mask
    membership = None
    n_regions = 0
    channel_keep = None
    for rule in [r for r in active if r.kind == "channels"]:
        backend = _CHANNEL_BACKENDS.get(rule.channel_by)
        ok, reason = channel_backend(rule.channel_by)
        if backend is None or not ok:
            problems.append(f"channels {rule.channel_by}: {reason}")
            continue
        try:
            member, count, mesh, note = backend(fd, rule)
        except (ChannelUnavailable, ValueError) as error:
            problems.append(f"channels {rule.channel_by}: {error}")
            continue
        membership = member if membership is None else np.where(
            member >= 0, member, membership)
        n_regions += count
        if mesh is not None and mesh.n_triangles:
            scene.overlay_meshes.append(mesh)
        inside = member >= 0
        channel_keep = inside if channel_keep is None else channel_keep | inside
        parts.append(note)

    colour_note = colour_atoms(scene, fd, colour_by, theme, membership)
    if colour_note:
        problems.append(colour_note)
    parts.append(f"colour: {colour_by}"
                 + (f" ({', '.join(fd.modifiers)} within {HOTMAP_RADIUS_ANG:g} Å "
                    "of each atom)" if colour_by == "modifier density"
                    and fd.modifiers else ""))

    keep = None
    n_matching = n
    wants_channel_distance = any(
        r.kind == "show-only" and r.descriptor == "distance to channel"
        for r in active)
    if wants_channel_distance:
        fd.channel_distance = _channel_distance(fd)
    for rule in [r for r in active if r.kind in ("show-only", "neighbours", "rows")]:
        try:
            this = _keep_of_rule(fd, rule)
        except ValueError as error:
            problems.append(f"show only: {error}")
            continue
        keep = this if keep is None else keep & this
        n_matching = int(keep.sum())
        if rule.kind == "rows":
            parts.append(f"{_thousands(n_matching)} of {_thousands(n)} atoms "
                         f"match ({rule.rows_label or 'the rows given'})")
        elif rule.kind == "neighbours":
            parts.append(f"{_thousands(n_matching)} of {_thousands(n)} atoms "
                         f"match ({rule.element} with bonds to "
                         f"{rule.neighbour} {rule.sense} {rule.value:g})")
        else:
            parts.append(f"{_thousands(n_matching)} of {_thousands(n)} atoms "
                         f"match ({rule.element}, {rule.descriptor} {rule.sense} "
                         f"{rule.value:g})")
    if keep is not None:
        # the count by element, so '52 atoms' reads as '12 Si, 40 O'
        by_element = []
        for symbol in fd.species:
            count = int((keep & (fd.elements == symbol)).sum())
            if count:
                by_element.append(f"{count} {symbol}")
        if by_element:
            parts.append("matched: " + ", ".join(by_element))
    if channel_keep is not None:
        keep = channel_keep if keep is None else keep & channel_keep
        parts.append(f"{_thousands(int(channel_keep.sum()))} atoms in "
                     f"{n_regions} region(s)")
    # the dimming or the hiding comes after the void rules, whose lining
    # atoms are a keep of their own (below)
    n_spheres = 0
    void_keep = None
    for rule in [r for r in active if r.kind == "voids"]:
        r_min = radius_for_volume(rule.min_volume_ang3)
        mode = rule.radii if rule.radii in VOID_RADII else "ionic"
        try:
            spheres = fd.spheres_with(mode)
            radii = np.asarray(spheres.radius_ang, np.float64)
        except ValueError as error:
            problems.append(f"voids: {error}")
            continue
        chosen = radii >= r_min
        if spheres.representative is not None:
            # one sphere per void: the spheres that climbed to the same
            # maximum are the same void seen from neighbouring tetrahedra
            chosen &= np.asarray(spheres.representative, bool)
        if rule.min_elongation is not None and rule.min_elongation > 1.0:
            # a sphere is drawn when its own volume passes and the void
            # region it belongs to (md_channels, probe radius 0) is at
            # least this elongated
            try:
                regions = fd.void_regions(0.0, rule.grid_spacing_ang or 0.5)
                keep_region = np.asarray(regions.elongation, np.float64) \
                    >= float(rule.min_elongation)
                rows = np.asarray(regions.sphere_rows, int)[
                    keep_region[np.asarray(regions.sphere_region, int)]]
                in_region = np.zeros(len(radii), bool)
                in_region[rows] = True
                chosen &= in_region
            except (ChannelUnavailable, AttributeError, ValueError) as error:
                problems.append(f"voids, elongation ≥ "
                                f"{rule.min_elongation:g}: {error}")
        count = int(chosen.sum())
        n_spheres += count
        drawn = chosen
        if count > VOID_DRAW_BUDGET:
            # the largest within the budget; the numbers above are the full count
            order = np.argsort(-radii)
            keep_idx = order[chosen[order]][:VOID_DRAW_BUDGET]
            drawn = np.zeros(len(radii), bool)
            drawn[keep_idx] = True
        centres = fd.sphere_centres_ang(mode)[drawn]
        if count:
            scene.overlay_meshes.append(sphere_mesh(centres, radii[drawn]))
            # the atoms lining the drawn voids: their surface within
            # VOID_LINING_ANG of a drawn sphere's surface
            lining = _void_lining(scene, fd, centres, radii[drawn],
                                  spheres.radii_ang)
            void_keep = lining if void_keep is None else void_keep | lining
            parts.append(f"{_thousands(int(lining.sum()))} atoms line the "
                         f"drawn voids")
        largest = float(radii.max()) if radii.size else float("nan")
        parts.append(f"{count} voids ≥ {rule.min_volume_ang3:g} Å³ "
                     f"(radius ≥ {r_min:.2f} Å) with {VOID_RADII[mode][1]}; "
                     f"largest {largest:.2f} Å"
                     + (f"; the largest {VOID_DRAW_BUDGET} drawn"
                        if count > VOID_DRAW_BUDGET else "")
                     + (f", elongation ≥ {rule.min_elongation:g}"
                        if rule.min_elongation is not None
                        and rule.min_elongation > 1.0 else "")
                     + " drawn")
    for rule in [r for r in active if r.kind == "hotmap"]:
        elements = ([e.strip() for e in rule.elements.split(",") if e.strip()]
                    if rule.elements else list(fd.modifiers))
        if not elements:
            problems.append("hot map: no modifier to count (name the formers, "
                            "or the elements in the rule)")
            continue
        try:
            hot = fd.hotmap(elements, rule.grid_spacing_ang or 0.5,
                            rule.sigma_ang or 1.5)
        except ValueError as error:
            problems.append(f"hot map: {error}")
            continue
        level = float(rule.level_x_mean) * hot.mean_density
        try:
            from ..core import volume
            step, _ = _choose_step(np.asarray(hot.grid.values), level)
            vertices, faces, normals = volume.isosurface(hot.grid, level, step=step)
        except ValueError as error:
            problems.append(f"hot map: {error}")
            continue
        vertices = np.asarray(vertices, np.float32)
        faces = np.asarray(faces, int)
        if len(faces) and len(vertices):
            flat = faces.reshape(-1)
            normals = np.asarray(normals, np.float32)
            tri_normals = (normals[flat] if len(normals) == len(vertices)
                           else np.zeros((len(flat), 3), np.float32))
            scene.overlay_meshes.append(OverlayMesh(
                vertices[flat], tri_normals, tuple(HOTMAP_COLOUR),
                float(HOTMAP_ALPHA), label="hotmap"))
        above = hot.fraction_above(level)
        parts.append(
            f"hot map of {'+'.join(hot.elements)}: {hot.n_atoms} atoms, mean "
            f"{hot.mean_density:.4f} atoms/Å³, highest {hot.max_density / hot.mean_density:.2f} "
            f"× the mean; the surface at {rule.level_x_mean:g} × the mean encloses "
            f"{100 * above:.1f} % of the box (sigma {hot.sigma_ang:g} Å, grid "
            f"{hot.grid_spacing_ang:g} Å" + (f", meshed at step {step}" if step > 1 else "") + ")")
    if void_keep is not None:
        keep = void_keep if keep is None else keep & void_keep
    if keep is not None:
        if dim_rest:
            _dim(scene, keep, theme, dim_amount)
            parts.append(f"the rest dimmed {100 * dim_amount:.0f} %")
        else:
            _hide(scene, keep)
            parts.append("the rest left out")
    if problems:
        parts.append("not drawn: " + "; ".join(problems))
    n_in = int(channel_keep.sum()) if channel_keep is not None else 0
    return HighlightResult(n_matching, n_spheres, " · ".join(parts),
                           n_regions, n_in, tuple(problems), keep)


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def _combo(items, current=0) -> QComboBox:
    box = QComboBox()
    box.addItems(list(items))
    box.setCurrentIndex(current)
    chrome.fit_combo(box)
    return box


def _spin(value, lo, hi, step, decimals=1, suffix="") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setValue(value)
    box.setSuffix(suffix)
    return box


def _grey_item(combo: QComboBox, index: int, reason: str) -> None:
    """Disable one item of a combo (and say why on hover)."""
    model = combo.model()
    item = model.item(index)
    if item is None:
        return
    item.setEnabled(not reason)
    item.setToolTip(reason)


class RuleRow(QFrame):
    """One rule: a tick, its controls, a remove button and a note line."""

    changed = Signal()
    removed = Signal(object)
    kind = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rule")
        self.setFrameShape(QFrame.StyledPanel)
        self.column = QVBoxLayout(self)
        self.column.setContentsMargins(6, 4, 4, 4)
        self.column.setSpacing(2)
        self.row = QHBoxLayout()
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(6)
        self.column.addLayout(self.row)
        self.tick = QCheckBox()
        self.tick.setChecked(False)
        self.tick.setToolTip("On or off; the rule stays listed.")
        self.tick.toggled.connect(lambda _on: self.changed.emit())
        self.row.addWidget(self.tick)
        self.close = QToolButton()
        self.close.setText("×")
        self.close.setAutoRaise(True)
        self.close.setToolTip("Remove this rule")
        self.close.clicked.connect(lambda: self.removed.emit(self))
        self.note = _hint()
        self.note.setContentsMargins(22, 0, 0, 0)
        # the note wraps (a grid refusal is a sentence); without an explicit
        # minimum the wrapped label still asks the layout for its full
        # unwrapped width, and the page no longer fits its column
        self.note.setMinimumWidth(1)
        self.note.hide()

    def _finish(self) -> None:
        self.row.addStretch(1)
        self.row.addWidget(self.close)
        self.column.addWidget(self.note)

    def _add(self, *widgets) -> None:
        for w in widgets:
            if isinstance(w, str):
                w = QLabel(w)
            self.row.addWidget(w)

    def set_note(self, text: str) -> None:
        self.note.setText(text)
        self.note.setVisible(bool(text))

    def set_frame_data(self, fd: FrameData | None) -> None:
        pass

    def rule(self) -> Rule:
        raise NotImplementedError


class ShowOnlyRow(RuleRow):
    kind = "show-only"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.element = _combo(["any"], 0)
        self.descriptor = _combo(DESCRIPTORS, 0)
        self.descriptor.setToolTip(
            "CN: bonds to every counter-ion at v_bond; former CN: bonds to "
            "the formers only (an O with former CN 1 is non-bridging); BVS "
            "and phi at v_bond; Qn: a former's bonds to bridging anions. "
            "former CN and Qn need the formers named in Setup.")
        # the descriptor list grew long names; the box shows the one chosen
        # and opens to the rest, rather than widening the whole column
        self.descriptor.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.descriptor.setMinimumContentsLength(6)
        self.descriptor.setMinimumWidth(0)
        self.sense = _combo(SENSES, 0)
        self.value = _spin(1, 0, 100, 1, 0)
        self.value.setFixedWidth(64)
        self._add("Show only", self.element, "where", self.descriptor,
                  self.sense, self.value)
        self.setToolTip("Atoms outside the rule are dimmed (or left out, "
                        "with Dim the rest off). The count of atoms matched "
                        "is written under the rules.")
        for w in (self.element, self.descriptor, self.sense):
            w.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self.descriptor.currentIndexChanged.connect(self._on_descriptor)
        self.value.valueChanged.connect(lambda _v: self.changed.emit())
        self._finish()

    def _on_descriptor(self, _index: int) -> None:
        name = self.descriptor.currentText()
        self.value.blockSignals(True)
        if name in ("BVS", "phi"):
            self.value.setDecimals(2)
            self.value.setSingleStep(0.05)
            self.value.setSuffix(" v.u." if name == "BVS" else "")
        else:
            self.value.setDecimals(0)
            self.value.setSingleStep(1)
            self.value.setSuffix("")
        self.value.blockSignals(False)

    def set_frame_data(self, fd: FrameData | None) -> None:
        current = self.element.currentText()
        self.element.blockSignals(True)
        self.element.clear()
        self.element.addItems(["any"] + (list(fd.species) if fd else []))
        at = self.element.findText(current)
        self.element.setCurrentIndex(max(0, at))
        chrome.fit_combo(self.element)
        self.element.blockSignals(False)
        reason = "" if fd is None or fd.former_cn is not None else \
            "needs the network formers (none are named in Setup)"
        for name in ("former CN", "Qn"):
            _grey_item(self.descriptor, DESCRIPTORS.index(name), reason)
        if reason and self.descriptor.currentText() in ("former CN", "Qn"):
            self.set_note(f"{self.descriptor.currentText()}: {reason}")
        else:
            self.set_note("")

    def rule(self) -> Rule:
        return Rule("show-only", self.tick.isChecked(),
                    element=self.element.currentText(),
                    descriptor=self.descriptor.currentText(),
                    sense=self.sense.currentText(),
                    value=float(self.value.value()))


class NeighboursRow(RuleRow):
    """Show only [element] bonded to [n] [neighbour element]: 'O bonded to
    exactly one Si and one Na' is two of these rules."""

    kind = "neighbours"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.element = _combo(["any"], 0)
        self.neighbour = _combo(["any"], 0)
        self.sense = _combo(SENSES, 2)
        self.value = _spin(1, 0, 100, 1, 0)
        self.value.setFixedWidth(64)
        self._add("Show only", self.element, "with bonds to", self.neighbour,
                  self.sense, self.value)
        for w in (self.element, self.neighbour, self.sense):
            w.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self.value.valueChanged.connect(lambda _v: self.changed.emit())
        self._finish()

    def set_frame_data(self, fd: FrameData | None) -> None:
        for box in (self.element, self.neighbour):
            current = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(["any"] + (list(fd.species) if fd is not None else []))
            i = box.findText(current)
            box.setCurrentIndex(i if i >= 0 else 0)
            box.blockSignals(False)
            chrome.fit_combo(box)

    def rule(self) -> Rule:
        return Rule(kind="neighbours", enabled=self.tick.isChecked(),
                    element=self.element.currentText(),
                    neighbour=self.neighbour.currentText(),
                    sense=self.sense.currentText(),
                    value=float(self.value.value()))


class RowsRow(RuleRow):
    """Show only a given set of frame rows: a group from the Environments
    tab, or any set handed over by name. The rows are not edited here."""

    kind = "rows"

    def __init__(self, parent=None, rows=(), label: str = ""):
        super().__init__(parent)
        self._rows = tuple(int(r) for r in rows)
        self._label = label
        self.caption = QLabel("")
        self._add("Show only", self.caption)
        self._finish()
        self.set_rows(self._rows, label)

    def set_rows(self, rows, label: str = "") -> None:
        self._rows = tuple(int(r) for r in rows)
        self._label = label
        self.caption.setText(f"{label or 'the rows given'} "
                             f"({len(self._rows)} atoms)")
        self.changed.emit()

    def rule(self) -> Rule:
        return Rule(kind="rows", enabled=self.tick.isChecked(),
                    rows=self._rows, rows_label=self._label)


class VoidsRow(RuleRow):
    kind = "voids"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume = _spin(8.0, 0.0, 100000.0, 1.0, 1, " Å³")
        self.volume.setToolTip("The voids of the frame at or above this "
                               "volume: the largest empty sphere at each "
                               "local maximum of the clearance to the atom "
                               "surfaces, drawn translucent.")
        self.radii = _combo([v[0] for v in VOID_RADII.values()], 0)
        self.radii.setToolTip(
            "The atom radius subtracted from the distance to each atom "
            "centre. Ionic: Shannon's radii by the model's oxidation states "
            "(O2- 1.40 Å, Si4+ 0.40 Å, Ca2+ 1.00 Å), the choice for an oxide "
            "or halide. Van der Waals: gemmi's radii, larger than a bond "
            "length in an ionic solid, so most of its voids come out "
            "negative. None: the spheres reach the atom centres.")
        self.elongation = _spin(1.0, 1.0, 50.0, 0.1, 1)
        self.elongation.setToolTip("Length over width of a void region "
                                   "(merged empty spheres, md_channels); "
                                   "1.0 keeps every shape.")
        self._add("Voids ≥", self.volume, "elongation ≥", self.elongation)
        extra = QHBoxLayout()
        extra.setContentsMargins(22, 0, 0, 0)
        extra.setSpacing(6)
        extra.addWidget(QLabel("radii"))
        extra.addWidget(self.radii)
        extra.addStretch(1)
        self.column.insertLayout(1, extra)
        self.radii.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self.setToolTip("Empty spheres at or above this volume, drawn as "
                        "translucent spheres; the elongation reads the void "
                        "regions of md_channels.")
        self.volume.valueChanged.connect(lambda _v: self.changed.emit())
        self.elongation.valueChanged.connect(lambda _v: self.changed.emit())
        self._finish()
        self.set_frame_data(None)

    def set_frame_data(self, fd: FrameData | None) -> None:
        ok, reason = channel_backend("by voids")
        self.elongation.setEnabled(ok)
        notes = [] if ok else [f"elongation greyed: {reason}"]
        # the union-volume grid this rule would run on (spacing 0.5 Å,
        # rule()), stated before anything runs; a box the cap refuses is
        # refused here in the same words
        grid = grid_points_note(fd, 0.5)
        if grid:
            notes.append(grid)
        self.set_note(" · ".join(notes))

    def rule(self) -> Rule:
        elongation = float(self.elongation.value()) \
            if self.elongation.isEnabled() else None
        keys = list(VOID_RADII)
        return Rule("voids", self.tick.isChecked(),
                    min_volume_ang3=float(self.volume.value()),
                    min_elongation=elongation, grid_spacing_ang=0.5,
                    radii=keys[max(0, self.radii.currentIndex())])


class HotmapRow(RuleRow):
    """Where the modifiers crowd: the smoothed number density of the
    modifier elements, drawn as a translucent surface at a multiple of
    its mean; 'Colour atoms by: modifier density' shows the count around
    each atom."""
    kind = "hotmap"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = _spin(1.5, 0.1, 20.0, 0.1, 1, " × mean")
        self.level.setToolTip("The density the surface is drawn at, as a "
                              "multiple of the mean modifier density of the "
                              "box; 1.5 encloses the regions half again as "
                              "dense as the average.")
        self.sigma = _spin(1.5, 0.3, 6.0, 0.1, 1, " Å")
        self.sigma.setToolTip("The width of the Gaussian each modifier atom "
                              "is spread with; larger smooths more.")
        self.spacing = _spin(0.5, 0.2, 3.0, 0.1, 1, " Å")
        self.spacing.setToolTip("The grid spacing of the density field.")
        self.elements = QComboBox()
        self.elements.setEditable(True)
        self.elements.setToolTip("The elements counted: the modifiers (every "
                                 "cation that is not a former), or type "
                                 "symbols separated by commas.")
        self.elements.setMinimumContentsLength(8)
        self._add("Hot map ≥", self.level, "sigma", self.sigma)
        extra = QHBoxLayout()
        extra.setContentsMargins(22, 0, 0, 0)
        extra.setSpacing(6)
        extra.addWidget(QLabel("of"))
        extra.addWidget(self.elements)
        extra.addWidget(QLabel("grid"))
        extra.addWidget(self.spacing)
        extra.addStretch(1)
        self.column.insertLayout(1, extra)
        for box in (self.level, self.sigma, self.spacing):
            box.valueChanged.connect(lambda _v: self.changed.emit())
        self.elements.currentTextChanged.connect(lambda _t: self.changed.emit())
        self.setToolTip("A hot map of the network modifiers: their smoothed "
                        "number density, the surface where it exceeds a "
                        "multiple of the mean. The count line states the "
                        "mean, the peak and the fraction of the box enclosed.")
        self._finish()
        self.set_frame_data(None)

    def set_frame_data(self, fd: FrameData | None) -> None:
        current = self.elements.currentText()
        self.elements.blockSignals(True)
        self.elements.clear()
        self.elements.addItem("modifiers")
        if fd is not None:
            for s in fd.species:
                self.elements.addItem(s)
        if current:
            i = self.elements.findText(current)
            if i >= 0:
                self.elements.setCurrentIndex(i)
            else:
                self.elements.setEditText(current)
        self.elements.blockSignals(False)
        note = []
        if fd is not None:
            if fd.modifiers:
                note.append("modifiers: " + ", ".join(fd.modifiers))
            else:
                note.append("no modifier known: name the formers in Setup, or "
                            "type the elements")
            grid = grid_points_note(fd, float(self.spacing.value()))
            if grid:
                note.append(grid)
        self.set_note(" · ".join(note))

    def rule(self) -> Rule:
        text = self.elements.currentText().strip()
        return Rule("hotmap", self.tick.isChecked(),
                    elements="" if text in ("", "modifiers") else text,
                    sigma_ang=float(self.sigma.value()),
                    level_x_mean=float(self.level.value()),
                    grid_spacing_ang=float(self.spacing.value()))


class ChannelsRow(RuleRow):
    kind = "channels"
    _THRESHOLD = {"by charge": ("mismatch ≤", 0.30, 0.0, 5.0, 0.05, 2,
                                " v.u."),
                  "by modifier density": ("modifiers ≥", 2, 1, 12, 1, 0, ""),
                  "by voids": ("elongation ≥", 1.5, 1.0, 50.0, 0.1, 1, "")}
    _CUT = {"by charge": ("r_cut", 6.0),
            "by modifier density": ("M–anion cutoff", 3.0),
            "by voids": ("probe radius", 0.0)}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = _combo(CHANNEL_MODES, 0)
        self.mode.setToolTip(
            "by charge: the bond-valence landscape of a probe ion, its "
            "regions where the mismatch is at most Δ, and the isosurface "
            "at Δ; by modifier density: the anions with k or more modifiers "
            "within the cutoff and the clusters they form; by voids: the "
            "void regions (merged empty spheres) of at least this "
            "elongation.")
        self.threshold_label = QLabel("mismatch ≤")
        self.threshold = _spin(0.30, 0.0, 5.0, 0.05, 2, " v.u.")
        self._add("Channels", self.mode, self.threshold_label, self.threshold)
        self._finish()
        second = QHBoxLayout()
        second.setContentsMargins(22, 0, 0, 0)
        second.setSpacing(6)
        self.spread_button = QPushButton("Δ from spread")
        self.spread_button.setToolTip(
            "Set the mismatch to twice the standard deviation of the sum "
            "at the probe's own atoms in this frame (about 95 % of them "
            "then lie within it of the mean); the note states the numbers.")
        self.spread_button.clicked.connect(self._delta_from_spread)
        self.target_label = QLabel("target")
        self.target = _combo(["frame mean", "formal charge"], 0)
        self.target.setToolTip(
            "What the probe's sum is matched to. 'frame mean': the mean of "
            "the same sum at the probe element's own atoms in this frame, "
            "so the regions follow how the model bonds that ion and pass "
            "through its sites; 'formal charge': the ion's formal valence, "
            "the original Adams form. The note states both numbers.")
        self.probe_label = QLabel("probe")
        self.probe = _combo([""], 0)
        self.probe.setToolTip("by charge: the mobile ion whose landscape is "
                              "drawn; by modifier density: the modifier "
                              "element counted.")
        self.spacing = _spin(0.5, 0.1, 5.0, 0.1, 2, " Å")
        self.spacing.setToolTip("grid spacing of the landscape (by charge) "
                                "or of the union volume (by voids)")
        self.cut_label = QLabel("r_cut")
        self.cut = _spin(6.0, 0.0, 20.0, 0.5, 2, " Å")
        self.cut.setToolTip("by charge: the distance beyond which an anion "
                            "adds nothing to the sum; by modifier density: "
                            "the M–anion cutoff; by voids: the probe "
                            "radius.")
        self.floor_label = QLabel("draw regions ≥")
        self.floor = _spin(CHANNEL_MESH_MIN_VOLUME_ANG3, 0.0, 100000.0,
                           1.0, 1, " Å³")
        self.floor.setToolTip(
            "by charge: only the regions of at least this volume are drawn "
            "as a surface (the largest always is); the region count, the "
            "volume fraction and the percolation always come from the full "
            "grid. A drawing choice that keeps the mesh small.")
        second.addWidget(self.probe_label)
        second.addWidget(self.probe)
        second.addWidget(QLabel("grid"))
        second.addWidget(self.spacing)
        second.addWidget(self.cut_label)
        second.addWidget(self.cut)
        second.addStretch(1)
        self.column.insertLayout(1, second)
        third = QHBoxLayout()
        third.setContentsMargins(22, 0, 0, 0)
        third.setSpacing(6)
        third.addWidget(self.floor_label)
        third.addWidget(self.floor)
        third.addStretch(1)
        self.column.insertLayout(2, third)
        fourth = QHBoxLayout()
        fourth.setContentsMargins(22, 0, 0, 0)
        fourth.setSpacing(6)
        fourth.addWidget(self.target_label)
        fourth.addWidget(self.target)
        fourth.addWidget(self.spread_button)
        fourth.addStretch(1)
        self.column.insertLayout(3, fourth)
        self.setToolTip("The regions md_channels measures on the frame "
                        "shown; the atoms outside them are dimmed and, "
                        "under 'channel membership', coloured by region.")
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.probe.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self.target.currentIndexChanged.connect(lambda _i: self.changed.emit())
        for w in (self.threshold, self.spacing, self.cut, self.floor):
            w.valueChanged.connect(lambda _v: self.changed.emit())
        # the note's grid estimate follows the spacing typed
        self.spacing.valueChanged.connect(lambda _v: self._availability())
        self._fd = None
        self._on_mode(0)

    def _on_mode(self, _index: int) -> None:
        mode = self.mode.currentText()
        text, value, lo, hi, step, decimals, suffix = self._THRESHOLD[mode]
        self.threshold_label.setText(text)
        self.threshold.blockSignals(True)
        self.threshold.setDecimals(decimals)
        self.threshold.setRange(lo, hi)
        self.threshold.setSingleStep(step)
        self.threshold.setSuffix(suffix)
        self.threshold.setValue(value)
        self.threshold.blockSignals(False)
        cut_text, cut_value = self._CUT[mode]
        self.cut_label.setText(cut_text)
        self.cut.blockSignals(True)
        self.cut.setValue(cut_value)
        self.cut.blockSignals(False)
        uses_probe = mode != "by voids"
        self.probe_label.setVisible(uses_probe)
        self.probe.setVisible(uses_probe)
        self.target_label.setVisible(mode == "by charge")
        self.target.setVisible(mode == "by charge")
        self.spread_button.setVisible(mode == "by charge")
        self.spacing.setVisible(mode != "by modifier density")
        self.floor_label.setVisible(mode == "by charge")
        self.floor.setVisible(mode == "by charge")
        self._availability()
        self.changed.emit()

    def _delta_from_spread(self) -> None:
        """Twice the standard deviation of the probe's own sums, into the
        mismatch spin; a note when the frame cannot say."""
        fd = self._fd
        probe = self.probe.currentText()
        if fd is None or not probe or self.cut.value() <= 0:
            self.set_note("Δ from spread needs a frame, a probe and an r_cut")
            return
        try:
            mean, sd, n = fd.probe_sum_spread(probe, float(self.cut.value()))
        except (ValueError, TypeError) as error:
            self.set_note(f"Δ from spread: {error}")
            return
        if not n or not math.isfinite(sd):
            self.set_note(f"Δ from spread: the frame holds no {probe} atom")
            return
        delta = max(0.01, round(2.0 * sd, 2))
        self.threshold.setValue(delta)
        self.set_note(f"{probe}: the sum at its {n} atoms is {mean:.3g} ± "
                      f"{sd:.2g} v.u.; Δ set to 2 × {sd:.2g} = {delta:g} v.u.")

    def _availability(self) -> None:
        for i, mode in enumerate(CHANNEL_MODES):
            ok, reason = channel_backend(mode)
            _grey_item(self.mode, i, "" if ok else reason)
        ok, reason = channel_backend(self.mode.currentText())
        self.tick.setEnabled(ok)
        notes = []
        if not ok:
            self.tick.setChecked(False)
            notes.append(f"greyed: {reason}")
        elif self._fd is not None and not self._fd.cations and \
                self.mode.currentText() != "by voids":
            notes.append("greyed: the frame holds no cation to probe with")
            self.tick.setEnabled(False)
        if self.mode.currentText() != "by modifier density":
            # the grid the spacing asks for over the frame shown, stated
            # before anything runs; a spacing the cap refuses shows the
            # engine's refusal here, in the same words
            grid = grid_points_note(self._fd, float(self.spacing.value()))
            if grid:
                notes.append(grid)
        self.set_note(" · ".join(notes))

    def set_frame_data(self, fd: FrameData | None) -> None:
        self._fd = fd
        current = self.probe.currentText()
        self.probe.blockSignals(True)
        self.probe.clear()
        choices = list(fd.modifiers) if fd is not None else []
        self.probe.addItems(choices or [""])
        at = self.probe.findText(current)
        self.probe.setCurrentIndex(max(0, at))
        chrome.fit_combo(self.probe)
        self.probe.blockSignals(False)
        self._availability()

    def rule(self) -> Rule:
        return Rule("channels", self.tick.isChecked() and self.tick.isEnabled(),
                    channel_by=self.mode.currentText(),
                    target="formal" if self.target.currentIndex() == 1 else "frame",
                    threshold=float(self.threshold.value()),
                    probe=self.probe.currentText(),
                    grid_spacing_ang=float(self.spacing.value()),
                    r_cut_ang=float(self.cut.value()),
                    min_volume_ang3=0.0,
                    mesh_min_volume_ang3=(float(self.floor.value())
                                          if self.mode.currentText()
                                          == "by charge" else None))


_ROW_CLASSES = {"show-only": ShowOnlyRow, "neighbours": NeighboursRow,
                "rows": RowsRow, "voids": VoidsRow, "channels": ChannelsRow,
                "hotmap": HotmapRow}


def _rule_stylesheet(theme) -> str:
    ui = chrome.ui_colors(theme)
    return (f"QFrame#rule {{ border: 1px solid {ui.border}; "
            f"border-radius: 4px; background: {ui.base}; }}")


class HighlightPage(QWidget):
    """Rules that change the 3D view of the frame shown.

    ``rulesChanged()`` is emitted after any edit; the window then calls
    :meth:`apply_to` on a freshly built scene (the rules edit the arrays,
    so a scene is built once per frame and the rules applied to a copy or
    a rebuild). :meth:`set_frame_data` gives the page the frame the rules
    read (its species fill the combos; the formers decide what is
    offered).
    """

    rulesChanged = Signal()

    def __init__(self, *, theme=None, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.frame_data: FrameData | None = None
        self.result: HighlightResult | None = None
        self.rows: list[RuleRow] = []
        self.setStyleSheet(_rule_stylesheet(theme))
        column = QVBoxLayout(self)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(6)
        lead = _hint("Rules that change the 3D view of the frame shown; "
                     "each can be switched off or removed, and the view "
                     "follows at once.")
        lead.setToolTip("The rules are drawn on every renderer tier, the "
                        "QPainter one included, and go into the exported "
                        "figure.")
        column.addWidget(lead)

        colour_row = QHBoxLayout()
        colour_row.addWidget(QLabel("Colour atoms by"))
        self.colour = _combo(COLOUR_MODES, 0)
        self.colour.setToolTip(
            "element, bond-valence sum, CN and phi: the crystal window's "
            "colour modes, on the frame at v_bond (cations on the ramp, "
            "anions in their element colour). Qn and modifier-rich O need "
            "the formers named in Setup; channel membership needs a "
            "channels rule that is on. Microscopic basicity puts every "
            "oxide and fluoride anion on the ramp by its lambda (Tools > "
            "Basicity chooses the scale); the colour bar on the view gives "
            "the range.")
        self.colour.currentIndexChanged.connect(lambda _i: self._changed())
        colour_row.addWidget(self.colour)
        colour_row.addStretch(1)
        column.addLayout(colour_row)

        column.addWidget(QLabel("<b>Rules</b>"))
        self.rules_box = QVBoxLayout()
        self.rules_box.setSpacing(4)
        column.addLayout(self.rules_box)
        self.only = self.add_rule("show-only")
        self.voids = self.add_rule("voids")
        self.channels = self.add_rule("channels")

        add_row = QHBoxLayout()
        self.add = QPushButton("Add rule…")
        menu = QMenu(self.add)
        for text, kind in (("Show only atoms where…", "show-only"),
                           ("Show only atoms with bonds to…", "neighbours"),
                           ("Voids above a volume", "voids"),
                           ("Channels", "channels"),
                           ("Hot map of the modifiers", "hotmap")):
            action = menu.addAction(text)
            action.triggered.connect(lambda _c=False, k=kind: self.add_rule(k))
        self.add.setMenu(menu)
        add_row.addWidget(self.add)
        # rule sets, named and kept between runs
        self.sets = QPushButton("Rule sets…")
        self.sets.setToolTip("Save these rules under a name, or load a set "
                             "saved before; kept between runs.")
        sets_menu = QMenu(self.sets)
        sets_menu.aboutToShow.connect(self._fill_sets_menu)
        self.sets.setMenu(sets_menu)
        add_row.addWidget(self.sets)
        add_row.addStretch(1)
        column.addLayout(add_row)

        column.addWidget(QLabel("<b>Then</b>"))
        # What happens to the atoms the rules do not pick: a two-way choice
        # in words. The hidden checkbox is the same state, kept for the
        # callers and tests that read or set ``dim``.
        self.rest = QComboBox()
        self.rest.addItems(["dimmed", "left out"])
        self.rest.setToolTip("Dimmed keeps the context of the box around "
                             "what the rules pick; left out draws only the "
                             "atoms picked, and the bonds among them.")
        chrome.fit_combo(self.rest)
        self.dim = QCheckBox("Dim the rest")
        self.dim.setChecked(True)
        self.dim.hide()
        self.rest.currentIndexChanged.connect(
            lambda i: self.dim.setChecked(i == 0))
        self.dim.toggled.connect(
            lambda on: (self.rest.setCurrentIndex(0 if on else 1),
                        self._changed()))
        self.export = QCheckBox("Add to export")
        self.export.setToolTip("Write the count line into the header of "
                               "every figure and table exported while the "
                               "rules are on (the window reads this box).")
        then = QHBoxLayout()
        then.addWidget(QLabel("Outside the rules:"))
        then.addWidget(self.rest)
        then.addWidget(self.export)
        then.addStretch(1)
        column.addLayout(then)
        # how far the dimmed atoms fade towards the background
        from PySide6.QtWidgets import QSlider
        self.dim_label = QLabel("Dim by")
        self.dim_slider = QSlider(Qt.Horizontal)
        self.dim_slider.setRange(0, 100)
        self.dim_slider.setValue(int(round(100 * DIM_DEFAULT)))
        self.dim_slider.setToolTip("How far the atoms outside the rules "
                                   "fade towards the background: 0 % leaves "
                                   "them as they are, 100 % hides them in "
                                   "it; they shrink in step.")
        self.dim_slider.setMaximumWidth(160)
        self.dim_value = QLabel(f"{self.dim_slider.value()} %")
        self.dim_value.setMinimumWidth(36)
        self.dim_slider.valueChanged.connect(
            lambda v: self.dim_value.setText(f"{v} %"))
        self.dim_slider.sliderReleased.connect(self._changed)
        self.dim_slider.valueChanged.connect(
            lambda _v: None if self.dim_slider.isSliderDown() else self._changed())
        dim_row = QHBoxLayout()
        dim_row.addWidget(self.dim_label)
        dim_row.addWidget(self.dim_slider)
        dim_row.addWidget(self.dim_value)
        dim_row.addStretch(1)
        column.addLayout(dim_row)
        for w in (self.dim_label, self.dim_slider, self.dim_value):
            w.setVisible(True)
        self.rest.currentIndexChanged.connect(
            lambda i: [w.setVisible(i == 0) for w in
                       (self.dim_label, self.dim_slider, self.dim_value)])

        self.summary = _hint()
        self.summary.setToolTip("The frame, v_bond and what was drawn, with "
                                "the values used.")
        column.addWidget(self.summary)
        column.addStretch(1)

    # -- rows ------------------------------------------------------------------
    # -- rule sets ---------------------------------------------------------
    def to_dict(self) -> dict:
        """The rules as plain data, for a saved set."""
        from dataclasses import asdict
        return {"rules": [asdict(r) for r in self.rules()],
                "colour_by": self.colour_by(), "dim_rest": self.dim_rest(),
                "dim_amount": self.dim_amount()}

    def from_dict(self, data: dict) -> None:
        for row in list(self.rows):
            self.remove_rule(row)
        for spec in data.get("rules", []):
            kind = spec.get("kind", "show-only")
            if kind not in _ROW_CLASSES:
                continue
            row = self.add_rule(kind)
            if kind == "rows" and hasattr(row, "set_rows"):
                row.set_rows(spec.get("rows", ()), spec.get("rows_label", ""))
            for name in ("element", "neighbour", "descriptor", "sense",
                         "channel_by", "probe"):
                box = getattr(row, name, None)
                if box is not None and hasattr(box, "findText") and name in spec:
                    i = box.findText(str(spec[name]))
                    if i >= 0:
                        box.setCurrentIndex(i)
            if kind == "hotmap":
                if "level_x_mean" in spec:
                    row.level.setValue(float(spec["level_x_mean"]))
                if "sigma_ang" in spec:
                    row.sigma.setValue(float(spec["sigma_ang"]))
                if spec.get("elements"):
                    row.elements.setEditText(str(spec["elements"]))
            if kind == "voids" and hasattr(row, "radii") and "radii" in spec:
                keys = list(VOID_RADII)
                if spec["radii"] in keys:
                    row.radii.setCurrentIndex(keys.index(spec["radii"]))
            if kind == "channels" and hasattr(row, "target") and "target" in spec:
                row.target.setCurrentIndex(
                    1 if str(spec["target"]) == "formal" else 0)
            for name in ("value", "min_volume_ang3", "threshold",
                         "grid_spacing_ang", "r_cut_ang"):
                if kind == "hotmap" and name == "grid_spacing_ang":
                    spin = getattr(row, "spacing", None)
                    if spin is not None and name in spec:
                        spin.setValue(float(spec[name]))
                    continue
                spin = getattr(row, name, None)
                if spin is not None and hasattr(spin, "setValue") and name in spec:
                    try:
                        spin.setValue(float(spec[name]))
                    except (TypeError, ValueError):
                        pass
            if hasattr(row, "tick"):
                row.tick.setChecked(bool(spec.get("enabled", True)))
        if "colour_by" in data and hasattr(self, "colour"):
            i = self.colour.findText(str(data["colour_by"]))
            if i >= 0:
                self.colour.setCurrentIndex(i)
        if "dim_rest" in data:
            self.dim.setChecked(bool(data["dim_rest"]))
        if "dim_amount" in data:
            try:
                self.dim_slider.setValue(
                    int(round(100 * min(max(float(data["dim_amount"]), 0.0), 1.0))))
            except (TypeError, ValueError):
                pass
        self._changed()

    def save_set(self, name: str) -> None:
        from ..core import prefs
        sets = prefs.get("rule_sets", {}) or {}
        sets[name] = self.to_dict()
        prefs.set("rule_sets", sets)

    def load_set(self, name: str) -> bool:
        from ..core import prefs
        sets = prefs.get("rule_sets", {}) or {}
        if name not in sets:
            return False
        self.from_dict(sets[name])
        return True

    def _fill_sets_menu(self) -> None:
        from ..core import prefs
        menu = self.sets.menu()
        menu.clear()
        save = menu.addAction("Save these rules as…")
        save.triggered.connect(self._ask_save_set)
        sets = prefs.get("rule_sets", {}) or {}
        if sets:
            menu.addSeparator()
            for name in sorted(sets):
                act = menu.addAction(name)
                act.triggered.connect(lambda _=False, n=name: self.load_set(n))
            menu.addSeparator()
            for name in sorted(sets):
                act = menu.addAction(f"Forget {name}")
                act.triggered.connect(lambda _=False, n=name: self._forget_set(n))

    def _ask_save_set(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save these rules",
                                        "A name for this rule set:")
        if ok and name.strip():
            self.save_set(name.strip())

    def _forget_set(self, name: str) -> None:
        from ..core import prefs
        sets = prefs.get("rule_sets", {}) or {}
        sets.pop(name, None)
        prefs.set("rule_sets", sets)

    def show_rows(self, rows, label: str) -> RuleRow:
        """One rows rule for ``rows`` (replacing an earlier rows rule),
        on, so the frame shows that set at once."""
        for existing in list(self.rows):
            if getattr(existing, "kind", "") == "rows":
                self.remove_rule(existing)
        row = self.add_rule("rows")
        row.set_rows(rows, label)
        row.tick.setChecked(True)
        return row

    def add_rule(self, kind: str) -> RuleRow:
        row = _ROW_CLASSES[kind]()
        row.changed.connect(self._changed)
        row.removed.connect(self.remove_rule)
        row.set_frame_data(self.frame_data)
        self.rows.append(row)
        self.rules_box.addWidget(row)
        return row

    def remove_rule(self, row: RuleRow) -> None:
        if row not in self.rows:
            return
        self.rows.remove(row)
        self.rules_box.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        self._changed()

    def _changed(self) -> None:
        self.rulesChanged.emit()

    # -- state -----------------------------------------------------------------
    def set_frame_data(self, fd: FrameData | None) -> None:
        self.frame_data = fd
        for row in self.rows:
            row.set_frame_data(fd)
        self._colour_availability()

    def _colour_availability(self) -> None:
        fd = self.frame_data
        reason = ("needs the network formers (none are named in Setup)"
                  if fd is not None and fd.qn is None else "")
        for name in ("Qn", "modifier-rich O"):
            _grey_item(self.colour, COLOUR_MODES.index(name), reason)

    def rules(self) -> list[Rule]:
        return [row.rule() for row in self.rows]

    def colour_by(self) -> str:
        return self.colour.currentText()

    def dim_rest(self) -> bool:
        return self.dim.isChecked()

    def dim_amount(self) -> float:
        """0 to 1: how far the dimmed atoms fade towards the background."""
        return self.dim_slider.value() / 100.0

    def apply_to(self, scene, theme=None) -> HighlightResult | None:
        """Apply the page's rules to ``scene`` (:func:`apply_rules`) and
        write the count line; None without frame data."""
        if self.frame_data is None:
            self.summary.setText("no frame data yet")
            return None
        result = apply_rules(scene, self.frame_data, self.rules(),
                             dim_rest=self.dim_rest(), colour_by=self.colour_by(),
                             theme=theme or self.theme,
                             dim_amount=self.dim_amount())
        self.show_result(result)
        return result

    def show_result(self, result: HighlightResult) -> None:
        self.result = result
        self.summary.setText(result.note)

    def apply_theme(self, theme) -> None:
        self.theme = theme
        self.setStyleSheet(_rule_stylesheet(theme))
