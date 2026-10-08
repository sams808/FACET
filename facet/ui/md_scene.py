"""The drawable scene of one MD frame, built from the bulk engine's bonds.

``facet.gl.scene.build_scene`` builds a crystal's scene from a ``Structure``:
a ``NeighborFinder`` over 125 image copies, then a Python loop over every pair
it yields within the 6 Å floor radius, with a parameter lookup per pair. On a
3 000-atom glass frame that took 2.4-3.3 s and on a 24 000-atom one 18-25 s
(scout_1, 2026-10-07), 87 % of it in the bond loop, of which 2-3 % of the
pairs became bonds. An MD frame already has its bonds: the bulk engine's
:class:`~facet.core.bulk.ValenceTable` (one vectorised pair search, the same
contacts, the same parameters, the same valences; ``tests/
test_bulk_equivalence.py``) holds every counter-ion contact above v_list of
every atom, which is exactly the set ``build_scene`` draws. This module fills
the :class:`~facet.gl.scene.Scene` arrays from it with array operations, so
the StructureView draws an MD frame on every render tier without a
``Structure`` ever being built.

:func:`build_md_scene` takes the frame and, as ``bonds``,

* a ``bulk.ValenceTable`` (the glass analysis keeps one per frame): every
  listed contact is drawn, the ones below v_bond thin and faded as in the
  crystal window, and :meth:`Scene.restyle` moves the threshold without a
  search;
* a ``bulk.Bonds``: those bonds only (each above its v_bond);
* None: the table is computed here, from a search at the largest
  ``BVParam.distance_for(v_list)`` over the cation-anion species present,
  with no 6 Å floor (the floor only widens a search whose extra pairs are all
  below v_list and are dropped), and the model's oxidation states
  (``ox``; :func:`facet.core.md_model.model_oxidation` of the frame's
  elements when not given).

WHAT IS THE SAME AS build_scene, AND WHAT IS NOT
------------------------------------------------
The same: the atoms (one per row of the frame, at ``frac @ box_ang``, the
box corner at the origin, as ``md_model.frame_to_structure`` places them),
their labels (element + atom id), radii and element colours; the bonds as a
set (cation row, anion row, distance) with bit-identical valences, their
radii (``bond_radius_for``), colours and the sub-threshold fading
(``Scene.restyle``); the box outline. ``tests/test_md_scene.py`` compares the
two on small frames.

Not the same:

* each bond is drawn from its cation's home position to the anion's image;
  ``build_scene`` draws from whichever end its search met first, so the
  periodic images added to close the bonds that cross the box differ (398
  here against 371 there on the 3 000-atom Na2O-3SiO2 frame, scout_1) while
  the bonds are the same;
* the image atoms are found by (anion row, lattice translation), exact
  integers, where ``build_scene`` merges them by a 0.05 Å distance;
* no coordination polyhedra, slab, lattice planes, overrides or
  displacement ellipsoids: an MD frame has no sites to hang them on and no
  displacement parameters (the ellipsoid style is drawn as ball-and-stick);
* the value colour modes (bond-valence sum, CN, phi, valence discrepancy)
  are computed from the table at the v_bond given here
  (``bulk.at_threshold``), on the cations, anions keeping their element
  colour as the crystal window keeps them; they need the table, so with
  ``bulk.Bonds`` the atoms take their element colours.

TIMINGS
-------
See ``tests/test_md_scene.py``, which records them; scout_1's prototype of
the same arrays measured 61 ms at 3 000 atoms, 220-257 ms at 12 000 and
384-431 ms at 24 000 (warm), against 2.4-24.5 s for ``build_scene``.
"""
from __future__ import annotations

import numpy as np

from ..core import bulk, bv, elements, theme as theme_mod
from ..core.md_model import Frame, ModelOxidation, model_oxidation
from ..gl.scene import Scene, Style, bond_radius_for, _frame_points

__all__ = ["build_md_scene", "box_segments"]

# The box corners in fractions, and the twelve edges between them (corners
# that differ in exactly one coordinate).
_CORNERS = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                    dtype=np.float64)
_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
          if int(np.abs(_CORNERS[i] - _CORNERS[j]).sum()) == 1]


def box_segments(box_ang) -> np.ndarray:
    """The twelve edges of the periodic box, as (12, 2, 3) endpoints in Å,
    the box corner at the origin (``unit_cell_segments`` of the crystal
    window, for an MD box)."""
    corners = _CORNERS @ np.asarray(box_ang, dtype=np.float64)
    return np.array([[corners[i], corners[j]] for i, j in _EDGES], np.float32)


def _ox_per_atom(frame: Frame, ox) -> np.ndarray:
    if ox is None:
        ox = model_oxidation(frame.species)
    if isinstance(ox, ModelOxidation):
        return ox.per_atom(frame.elements)
    states = np.asarray(ox, dtype=np.int64)
    if states.shape != (frame.n_atoms,):
        raise ValueError(f"ox: a ModelOxidation or one state per atom "
                         f"({frame.n_atoms}) is needed, not shape "
                         f"{states.shape}")
    return states


def valence_table_for(frame: Frame, ox=None, *,
                      params: bv.ParameterSet | None = None,
                      v_list: float = bv.V_LIST_DEFAULT) -> bulk.ValenceTable:
    """Every counter-ion contact above ``v_list`` of every atom: the contacts
    ``build_scene`` draws, from one search at the largest
    ``distance_for(v_list)`` of the species present (no 6 Å floor)."""
    params = params or bv.DEFAULT
    ox_atom = _ox_per_atom(frame, ox)
    radius = bulk.search_radius_ang(frame.elements, ox_atom, params, v_list,
                                    floor_ang=0.0)
    pairs = bulk.find_pairs(frame, radius)
    return bulk.valence_table(frame, pairs, ox_atom, params, v_list_vu=v_list)


def _contacts(source) -> tuple[np.ndarray, ...]:
    """(cation, anion, image, d, v) of every contact to draw, each once,
    from its cation's row."""
    if isinstance(source, bulk.ValenceTable):
        width = source.v_vu.shape[1]
        listed = np.arange(width)[None, :] < source.n_listed[:, None]
        rows, cols = np.nonzero(listed & ~source.is_anion[:, None])
        return (rows.astype(np.int64),
                source.nbr[rows, cols].astype(np.int64),
                source.image[rows, cols].astype(np.int64),
                source.d_ang[rows, cols].astype(np.float64),
                source.v_vu[rows, cols].astype(np.float64))
    if isinstance(source, bulk.Bonds):
        return (np.asarray(source.cation, np.int64),
                np.asarray(source.anion, np.int64),
                np.asarray(source.image, np.int64).reshape(-1, 3),
                np.asarray(source.d_ang, np.float64),
                np.asarray(source.v_vu, np.float64))
    raise ValueError(f"bonds: a bulk.ValenceTable, a bulk.Bonds or None is "
                     f"needed, not {type(source).__name__}")


def _value_colours(table: bulk.ValenceTable, mode, theme, v_bond: float,
                   base: np.ndarray) -> np.ndarray:
    """The crystal window's continuous colour modes, per atom: the cations
    take the ramp colour of their value at ``v_bond`` (``bulk.at_threshold``,
    field for field the SiteResult value ``theme.site_values`` reads), the
    anions keep ``base``."""
    results = bulk.at_threshold(table, v_bond)
    if mode is theme_mod.ColorMode.BVS:
        values = results.bvs_vu
    elif mode is theme_mod.ColorMode.COORDINATION:
        values = results.cn.astype(np.float64)
    elif mode is theme_mod.ColorMode.PHI:
        values = results.phi
    else:
        values = results.valence_discrepancy_vu
    cations = np.nonzero(~table.is_anion)[0]
    by_atom = {int(i): float(values[i]) for i in cations}
    lo, hi = theme_mod.scale_range(by_atom, theme, mode)
    diverging = mode is theme_mod.ColorMode.VALENCE_DISCREPANCY
    out = base.copy()
    for i, value in by_atom.items():
        out[i] = theme_mod.color_for_value(value, lo, hi, diverging)
    return out


def build_md_scene(frame: Frame, bonds=None, *, theme=None, ox=None,
                   params: bv.ParameterSet | None = None,
                   v_bond: float = bv.V_BOND_DEFAULT,
                   v_list: float = bv.V_LIST_DEFAULT,
                   style: Style = Style.BALL_AND_STICK,
                   show_cell: bool = True,
                   bonded_images: bool = True) -> Scene:
    """The Scene ``StructureView.set_scene`` draws, for one MD frame.

    ``bonds`` is a ``bulk.ValenceTable`` (every contact above its v_list,
    styled at ``v_bond``), a ``bulk.Bonds`` (those bonds) or None (the table
    computed here with ``ox``, ``params`` and ``v_list``; module docstring).
    ``ox`` is a ModelOxidation or one state per atom, read only when
    ``bonds`` is None. ``bonded_images`` adds the periodic image of each
    anion that closes a bond crossing the box, as ``build_scene`` does;
    ``show_cell`` draws the box. The scene's ``atom_site`` is the frame row,
    so a picked atom (an image included) names its row.
    """
    if not isinstance(frame, Frame):
        raise ValueError(f"build_md_scene needs a Frame, not "
                         f"{type(frame).__name__}")
    theme = theme or theme_mod.Theme()
    params = params or bv.DEFAULT
    table = bonds if bonds is not None else valence_table_for(
        frame, ox, params=params, v_list=v_list)
    if isinstance(table, bulk.ValenceTable) and table.n_atoms != frame.n_atoms:
        raise ValueError(f"the valence table holds {table.n_atoms} atoms and "
                         f"the frame {frame.n_atoms}")
    cation, anion, image, d_ang, v_vu = _contacts(table)
    if len(cation) and (int(max(cation.max(), anion.max())) >= frame.n_atoms
                        or int(min(cation.min(), anion.min())) < 0):
        raise ValueError("the bonds name atom rows outside the frame")
    if style is Style.ELLIPSOIDS:
        # no displacement parameters in an MD frame: the module docstring
        style = Style.BALL_AND_STICK

    n = frame.n_atoms
    box = np.asarray(frame.box_ang, np.float64)
    # frac @ box: frame_to_structure's Atom.cart (Cell.orth == box_ang.T),
    # the box corner at the origin
    cart = np.asarray(frame.frac, np.float64) @ box
    symbols = np.asarray(frame.elements)
    tokens, inverse = np.unique(symbols, return_inverse=True)
    inverse = inverse.reshape(-1)
    palette = np.array([theme.element_color(str(t)) for t in tokens],
                       np.float32).reshape(-1, 3)
    info = [elements.info(str(t)) for t in tokens]
    if style is Style.SPACE_FILLING:
        sizes = [(i.vdw_radius or (i.display_radius * 2.6)) for i in info]
    elif style in (Style.STICK, Style.WIREFRAME):
        sizes = [0.10 for _ in info]
    else:
        sizes = [i.display_radius for i in info]
    radius_of = np.array(sizes, np.float32) * float(theme.atom_scale)

    mode = theme.color_mode
    element_colour = palette[inverse]
    if mode is theme_mod.ColorMode.UNIFORM:
        atom_colour = np.tile(np.array(theme.uniform_atom_color, np.float32),
                              (n, 1))
    elif mode is theme_mod.ColorMode.SITE:
        cycle = np.array(theme_mod.SITE_CYCLE, np.float32)
        atom_colour = cycle[np.arange(n) % len(cycle)]
    elif mode is not theme_mod.ColorMode.ELEMENT and \
            isinstance(table, bulk.ValenceTable):
        atom_colour = _value_colours(table, mode, theme, v_bond, element_colour)
    else:
        atom_colour = element_colour
    atom_colour = np.asarray(atom_colour, np.float32)

    # the periodic images that close a bond across the box: one per distinct
    # (anion row, translation), exact integers rather than a distance merge
    if bonded_images and len(anion):
        off = np.any(image != 0, axis=1)
        keys = np.unique(np.column_stack([anion[off], image[off]]), axis=0) \
            if off.any() else np.zeros((0, 4), np.int64)
    else:
        keys = np.zeros((0, 4), np.int64)
    source = np.concatenate([np.arange(n, dtype=np.int64), keys[:, 0]])
    position = np.vstack([cart, cart[keys[:, 0]] + keys[:, 1:] @ box]) \
        if len(keys) else cart

    scene = Scene(style=style, v_bond=float(v_bond), theme=theme)
    scene.poly_color = theme.polyhedron_color
    scene.poly_alpha = theme.polyhedron_alpha
    scene.atom_position = position.astype(np.float32)
    scene.atom_color = np.vstack([atom_colour, atom_colour[keys[:, 0]]]) \
        if len(keys) else atom_colour
    scene.atom_radius = radius_of[inverse][source].astype(np.float32)
    scene.atom_index = np.arange(len(source), dtype=np.int32)
    scene.atom_site = source.astype(np.int32)
    scene.atom_element = [str(s) for s in symbols[source]]
    ids = np.asarray(frame.atom_id)[source]
    scene.atom_label = [f"{e}{int(i)}" for e, i in zip(scene.atom_element, ids)]
    scene.n_cell_atoms = n

    if len(cation):
        a = cart[cation]
        b = cart[anion] + image @ box
        scene.bond_a = a.astype(np.float32)
        scene.bond_b = b.astype(np.float32)
        bond_mode = theme.bond_color_mode
        if bond_mode is theme_mod.BondColorMode.UNIFORM:
            colour = np.tile(np.array(theme.uniform_bond_color, np.float32),
                             (len(cation), 1))
            base_a = base_b = colour
        elif bond_mode is theme_mod.BondColorMode.BY_VALENCE:
            # _bond_colors' ramp, a bond at a time as there
            colour = np.array([theme_mod.ramp(min(float(x) / 0.8, 1.0))
                               for x in v_vu], np.float32).reshape(-1, 3)
            base_a = base_b = colour
        else:
            base_a = palette[inverse[cation]]
            base_b = palette[inverse[anion]]
        scene._bond_base_a = np.array(base_a, np.float32)
        scene._bond_base_b = np.array(base_b, np.float32)
        scene.bond_color_a = scene._bond_base_a.copy()
        scene.bond_color_b = scene._bond_base_b.copy()
        scene.bond_valence = v_vu.astype(np.float32)
        scene.bond_distance = d_ang.astype(np.float32)
        scene.bond_atoms = np.column_stack([cation, anion]).astype(np.int32)
        scene.bond_cation = np.zeros(len(cation), np.int8)
        scene.bond_occupancy = np.ones(len(cation), np.float32)
        # filled before restyle, which returns early on an empty radius array
        scene.bond_radius = bond_radius_for(scene.bond_valence, scene.v_bond)

    if show_cell:
        scene.cell_segments = box_segments(box)
    scene.restyle(float(v_bond))
    _frame_points(scene)
    return scene
