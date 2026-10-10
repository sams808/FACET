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
  region dimmed; by charge the mismatch isosurface at Δ is drawn as well.

Every threshold is what the field shows: nothing physical is assumed, and
the count line states the frame, v_bond and what was drawn with which
values. :class:`FrameData` holds what the rules read of one frame, with
the empty spheres and the bond-valence landscapes computed once, on demand.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

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
    "channel_backend", "sphere_mesh", "icosphere", "radius_for_volume",
    "DESCRIPTORS", "COLOUR_MODES", "CHANNEL_MODES", "VOID_COLOUR",
    "CHANNEL_COLOUR",
]

DESCRIPTORS = ("CN", "former CN", "BVS", "phi", "Qn")
SENSES = ("≤", "≥")
COLOUR_MODES = ("element", "bond-valence sum", "coordination number",
                "stereoactivity (phi)", "Qn", "modifier-rich O",
                "channel membership")
CHANNEL_MODES = ("by charge", "by modifier density", "by voids")

# Presentation choices of the overlays (an orange the element palettes do
# not use for the voids, a blue for the channels) and of the dimming: the
# dimmed atom keeps a fifth of its colour over the background and 40 % of
# its radius, a bond 35 %; measured on the prototype as what keeps the box
# readable while the kept atoms stand out.
VOID_COLOUR = (0.98, 0.72, 0.20)
VOID_ALPHA = 0.38
CHANNEL_COLOUR = (0.20, 0.55, 0.85)
CHANNEL_ALPHA = 0.30
DIM_KEEP = 0.20
DIM_ATOM_RADIUS = 0.40
DIM_BOND_RADIUS = 0.35
NEUTRAL = (0.60, 0.62, 0.66)       # theme.ramp's colour for NaN


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
    _spheres: object = field(default=None, repr=False)
    _landscapes: dict = field(default_factory=dict, repr=False)
    _void_regions: dict = field(default_factory=dict, repr=False)

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

    @property
    def spheres(self):
        """The empty spheres of the frame (FACET's van der Waals radii),
        computed on the first call."""
        if self._spheres is None:
            self._spheres = md_order.empty_spheres(
                self.frame, md_order.vdw_radii_ang(self.frame),
                radii_source=md_order.VDW_RADII_SOURCE)
        return self._spheres

    def sphere_centres_ang(self) -> np.ndarray:
        box = np.asarray(self.frame.box_ang, np.float64)
        return np.asarray(self.spheres.centre_frac, np.float64) @ box

    def descriptor(self, name: str) -> np.ndarray:
        """The per-atom values of a descriptor of :data:`DESCRIPTORS`
        (ValueError when it needs the formers and there are none)."""
        table = {"CN": self.cn, "former CN": self.former_cn,
                 "BVS": self.bvs, "phi": self.phi, "Qn": self.qn}
        if name not in table:
            raise ValueError(f"descriptor {name!r}: one of {DESCRIPTORS}")
        values = table[name]
        if values is None:
            raise ValueError(f"{name} needs the network formers (none are "
                             "named)")
        return np.asarray(values, np.float64)

    def landscape(self, probe: str, grid_spacing_ang: float,
                  r_cut_ang: float):
        """The bond-valence landscape of ``probe`` on this frame
        (:func:`facet.core.md_channels.bv_landscape`), kept per
        (probe, spacing, r_cut)."""
        from ..core import md_channels

        key = (str(probe), float(grid_spacing_ang), float(r_cut_ang))
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
                r_cut_ang=float(r_cut_ang), params=self.params)
        return self._landscapes[key]

    def void_regions(self, probe_radius_ang: float, grid_spacing_ang: float):
        """The void regions of this frame
        (:func:`facet.core.md_channels.void_regions` on :attr:`spheres`,
        lining distance 0), kept per (probe radius, spacing);
        :class:`ChannelUnavailable` when the build lacks them."""
        from ..core import md_channels

        ok, reason = channel_backend("by voids")
        if not ok:
            raise ChannelUnavailable(reason)
        key = (float(probe_radius_ang), float(grid_spacing_ang))
        if key not in self._void_regions:
            try:
                self._void_regions[key] = md_channels.void_regions(
                    self.frame, self.spheres, probe_radius_ang=key[0],
                    grid_spacing_ang=key[1], lining_distance_ang=0.0)
            except TypeError as error:
                raise ChannelUnavailable(
                    "md_channels.void_regions takes other arguments than "
                    "this page passes (frame, spheres, probe_radius_ang, "
                    f"grid_spacing_ang, lining_distance_ang): {error}"
                    ) from None
        return self._void_regions[key]


def _thousands(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


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
    min_volume_ang3: float = 0.0
    min_elongation: float | None = None
    channel_by: str = "by charge"
    threshold: float = 0.0
    probe: str = ""
    grid_spacing_ang: float = 0.0
    r_cut_ang: float = 0.0


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


def _channels_by_charge(fd: FrameData, rule: Rule):
    """(region of each atom, n_regions, mesh, note) by the bond-valence
    landscape of the probe and its accessible regions at Δ."""
    from ..core import md_channels, volume

    probe = rule.probe or (fd.modifiers[0] if fd.modifiers else "")
    if not probe:
        raise ChannelUnavailable("by charge needs a probe ion: the frame "
                                 "holds no cation")
    if rule.grid_spacing_ang <= 0 or rule.r_cut_ang <= 0:
        raise ChannelUnavailable("by charge needs a grid spacing and an "
                                 "r_cut above 0 (both are shown in the rule)")
    landscape = fd.landscape(probe, rule.grid_spacing_ang, rule.r_cut_ang)
    regions = md_channels.accessible_regions(landscape, rule.threshold)
    grid = md_channels.landscape_grid(landscape, "mismatch")
    vertices, faces, normals = volume.isosurface(grid, float(rule.threshold))
    mesh = _triangle_mesh(vertices, faces, normals, CHANNEL_COLOUR,
                          CHANNEL_ALPHA, "channels")
    fraction = float(regions.accessible_fraction)
    spans = "".join(axis for axis, on in zip("abc", regions.percolates)
                    if bool(on))
    note = (f"channels by charge: {landscape.probe_label} probe, mismatch "
            f"≤ {rule.threshold:g} v.u. on a {rule.grid_spacing_ang:g} Å "
            f"grid (r_cut {rule.r_cut_ang:g} Å): {regions.n_regions} "
            f"region(s), {100 * fraction:.1f} % of the box"
            + (f", spanning {spans}" if spans else ", none spanning the box"))
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
    if colour_by == "element":
        colours = base
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
    elif colour_by == "modifier-rich O":
        if fd.modifier_count is None:
            note = "modifier-rich O needs the network formers (none are named)"
            colours = base
        else:
            by_atom = {int(i): float(fd.modifier_count[i]) for i in anions}
            finite = list(by_atom.values())
            hi = max(max(finite) if finite else 0.0, 1.0)
            colours = _ramp_over(by_atom, 0.0, hi, base, sites)
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
    else:
        raise ValueError(f"colour_by {colour_by!r}: one of {COLOUR_MODES}")
    scene.atom_color = np.asarray(colours, np.float32).reshape(-1, 3)
    return note


# ---------------------------------------------------------------------------
# applying the rules to a scene
# ---------------------------------------------------------------------------

def _keep_of_rule(fd: FrameData, rule: Rule) -> np.ndarray:
    values = fd.descriptor(rule.descriptor)
    with np.errstate(invalid="ignore"):
        keep = (values <= rule.value) if rule.sense == "≤" \
            else (values >= rule.value)
    keep &= np.isfinite(values)
    if rule.element != "any":
        keep &= fd.elements == rule.element
    return keep


def _dim(scene, keep_rows: np.ndarray, theme) -> None:
    bg = np.array(getattr(theme, "background", None)
                  or theme_mod.FALLBACK_BACKGROUND, np.float32)
    keep = keep_rows[np.asarray(scene.atom_site, int)]
    pale = (1 - DIM_KEEP) * bg + DIM_KEEP * scene.atom_color
    scene.atom_color = np.where(keep[:, None], scene.atom_color,
                                pale).astype(np.float32)
    scene.atom_radius = np.where(keep, scene.atom_radius,
                                 scene.atom_radius * DIM_ATOM_RADIUS
                                 ).astype(np.float32)
    if scene.n_bonds:
        ends = np.asarray(scene.bond_atoms, int)
        bond_keep = keep_rows[ends[:, 0]] | keep_rows[ends[:, 1]]
        for name in ("bond_color_a", "bond_color_b"):
            c = getattr(scene, name)
            setattr(scene, name, np.where(bond_keep[:, None], c,
                                          (1 - DIM_KEEP) * bg + DIM_KEEP * c)
                    .astype(np.float32))
        scene.bond_radius = np.where(bond_keep, scene.bond_radius,
                                     scene.bond_radius * DIM_BOND_RADIUS
                                     ).astype(np.float32)


_PER_ATOM = ("atom_position", "atom_radius", "atom_color", "atom_site")
_PER_BOND = ("bond_a", "bond_b", "bond_radius", "bond_color_a",
             "bond_color_b", "bond_valence", "bond_distance", "bond_atoms",
             "bond_cation", "bond_occupancy", "_bond_base_a", "_bond_base_b")


def _hide(scene, keep_rows: np.ndarray) -> None:
    """Leave the atoms outside ``keep_rows`` (and the bonds to them) out of
    the scene's arrays; the centre and radius are kept, so the camera
    stays."""
    keep = keep_rows[np.asarray(scene.atom_site, int)]
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
    scene.selected_atom = None


def _drop_overlays(scene, labels: Sequence[str]) -> None:
    scene.overlay_meshes = [m for m in scene.overlay_meshes
                            if getattr(m, "label", "") not in labels]


def apply_rules(scene, fd: FrameData, rules: Sequence[Rule], *,
                dim_rest: bool = True, colour_by: str = "element",
                theme=None) -> HighlightResult:
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
    _drop_overlays(scene, ("voids", "channels"))

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
    parts.append(f"colour: {colour_by}")

    keep = None
    n_matching = n
    for rule in [r for r in active if r.kind == "show-only"]:
        try:
            this = _keep_of_rule(fd, rule)
        except ValueError as error:
            problems.append(f"show only: {error}")
            continue
        keep = this if keep is None else keep & this
        n_matching = int(keep.sum())
        parts.append(f"{_thousands(n_matching)} of {_thousands(n)} atoms "
                     f"match ({rule.element}, {rule.descriptor} {rule.sense} "
                     f"{rule.value:g})")
    if channel_keep is not None:
        keep = channel_keep if keep is None else keep & channel_keep
        parts.append(f"{_thousands(int(channel_keep.sum()))} atoms in "
                     f"{n_regions} region(s)")
    if keep is not None:
        if dim_rest:
            _dim(scene, keep, theme)
            parts.append("the rest dimmed")
        else:
            _hide(scene, keep)
            parts.append("the rest left out")

    n_spheres = 0
    for rule in [r for r in active if r.kind == "voids"]:
        r_min = radius_for_volume(rule.min_volume_ang3)
        try:
            radii = np.asarray(fd.spheres.radius_ang, np.float64)
        except ValueError as error:
            problems.append(f"voids: {error}")
            continue
        chosen = radii >= r_min
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
        centres = fd.sphere_centres_ang()[chosen]
        count = int(chosen.sum())
        n_spheres += count
        if count:
            scene.overlay_meshes.append(sphere_mesh(centres, radii[chosen]))
        parts.append(f"{count} empty spheres ≥ {rule.min_volume_ang3:g} Å³ "
                     f"(radius ≥ {r_min:.2f} Å)"
                     + (f", elongation ≥ {rule.min_elongation:g}"
                        if rule.min_elongation is not None
                        and rule.min_elongation > 1.0 else "")
                     + " drawn")
    if problems:
        parts.append("not drawn: " + "; ".join(problems))
    n_in = int(channel_keep.sum()) if channel_keep is not None else 0
    return HighlightResult(n_matching, n_spheres, " · ".join(parts),
                           n_regions, n_in, tuple(problems))


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


class VoidsRow(RuleRow):
    kind = "voids"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume = _spin(8.0, 0.0, 100000.0, 1.0, 1, " Å³")
        self.volume.setToolTip("Empty spheres (vdW radii) of the frame at "
                               "or above this volume, drawn as translucent "
                               "spheres.")
        self.elongation = _spin(1.0, 1.0, 50.0, 0.1, 1)
        self.elongation.setToolTip("Length over width of a void region "
                                   "(merged empty spheres, md_channels); "
                                   "1.0 keeps every shape.")
        self._add("Voids ≥", self.volume, "elongation ≥", self.elongation)
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
        self.set_note("" if ok else f"elongation greyed: {reason}")

    def rule(self) -> Rule:
        elongation = float(self.elongation.value()) \
            if self.elongation.isEnabled() else None
        return Rule("voids", self.tick.isChecked(),
                    min_volume_ang3=float(self.volume.value()),
                    min_elongation=elongation, grid_spacing_ang=0.5)


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
        second.addWidget(self.probe_label)
        second.addWidget(self.probe)
        second.addWidget(QLabel("grid"))
        second.addWidget(self.spacing)
        second.addWidget(self.cut_label)
        second.addWidget(self.cut)
        second.addStretch(1)
        self.column.insertLayout(1, second)
        self.setToolTip("The regions md_channels measures on the frame "
                        "shown; the atoms outside them are dimmed and, "
                        "under 'channel membership', coloured by region.")
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.probe.currentIndexChanged.connect(lambda _i: self.changed.emit())
        for w in (self.threshold, self.spacing, self.cut):
            w.valueChanged.connect(lambda _v: self.changed.emit())
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
        self.spacing.setVisible(mode != "by modifier density")
        self._availability()
        self.changed.emit()

    def _availability(self) -> None:
        for i, mode in enumerate(CHANNEL_MODES):
            ok, reason = channel_backend(mode)
            _grey_item(self.mode, i, "" if ok else reason)
        ok, reason = channel_backend(self.mode.currentText())
        self.tick.setEnabled(ok)
        if not ok:
            self.tick.setChecked(False)
            self.set_note(f"greyed: {reason}")
        elif self._fd is not None and not self._fd.cations and \
                self.mode.currentText() != "by voids":
            self.set_note("greyed: the frame holds no cation to probe with")
            self.tick.setEnabled(False)
        else:
            self.set_note("")

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
                    threshold=float(self.threshold.value()),
                    probe=self.probe.currentText(),
                    grid_spacing_ang=float(self.spacing.value()),
                    r_cut_ang=float(self.cut.value()),
                    min_volume_ang3=0.0)


_ROW_CLASSES = {"show-only": ShowOnlyRow, "voids": VoidsRow,
                "channels": ChannelsRow}


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
            "channels rule that is on.")
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
                           ("Voids above a volume", "voids"),
                           ("Channels", "channels")):
            action = menu.addAction(text)
            action.triggered.connect(lambda _c=False, k=kind: self.add_rule(k))
        self.add.setMenu(menu)
        add_row.addWidget(self.add)
        add_row.addStretch(1)
        column.addLayout(add_row)

        column.addWidget(QLabel("<b>Then</b>"))
        self.dim = QCheckBox("Dim the rest")
        self.dim.setToolTip("Atoms and bonds outside the rules are drawn "
                            "paler and smaller rather than left out, so the "
                            "context stays.")
        self.dim.setChecked(True)
        self.dim.toggled.connect(lambda _on: self._changed())
        self.export = QCheckBox("Add to export")
        self.export.setToolTip("Write the count line into the header of "
                               "every figure and table exported while the "
                               "rules are on (the window reads this box).")
        then = QHBoxLayout()
        then.addWidget(self.dim)
        then.addWidget(self.export)
        then.addStretch(1)
        column.addLayout(then)

        self.summary = _hint()
        self.summary.setToolTip("The frame, v_bond and what was drawn, with "
                                "the values used.")
        column.addWidget(self.summary)
        column.addStretch(1)

    # -- rows ------------------------------------------------------------------
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

    def apply_to(self, scene, theme=None) -> HighlightResult | None:
        """Apply the page's rules to ``scene`` (:func:`apply_rules`) and
        write the count line; None without frame data."""
        if self.frame_data is None:
            self.summary.setText("no frame data yet")
            return None
        result = apply_rules(scene, self.frame_data, self.rules(),
                             dim_rest=self.dim_rest(), colour_by=self.colour_by(),
                             theme=theme or self.theme)
        self.show_result(result)
        return result

    def show_result(self, result: HighlightResult) -> None:
        self.result = result
        self.summary.setText(result.note)

    def apply_theme(self, theme) -> None:
        self.theme = theme
        self.setStyleSheet(_rule_stylesheet(theme))
