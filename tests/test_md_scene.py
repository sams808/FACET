"""The MD frame's scene, built from the bulk engine's bonds, against build_scene.

``facet.ui.md_scene.build_md_scene`` fills the arrays ``StructureView``
draws straight from a ``bulk.ValenceTable``, without the ``Structure`` and
the ``NeighborFinder`` loop ``facet.gl.scene.build_scene`` needs. These
tests hold it to ``build_scene`` on frames small enough for the latter: the
same bonds as a set (cation row, anion row, distance), bit-identical
valences, the same drawn radii and colours at any threshold, the same atom
positions; and they record how long each took.

The frames are crystals unrolled into supercells (``md_model.
supercell_frame``) and jittered with a fixed seed, so that no two contacts
of a site are equal and every comparison is between distinct numbers.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


import numpy as np  # noqa: E402
import pytest  # noqa: E402

from conftest import dispose  # noqa: E402

CRYSTALS = Path(__file__).parent / "data" / "crystals"
SHOTS = os.environ.get("FACET_MD_SCENE_SHOTS", "")


def _frame(name: str, reps=(2, 2, 2), jitter_ang: float = 0.0, seed: int = 7):
    from facet.core import md_model, readers

    structure = readers.read(CRYSTALS / name)
    frame, _parent = md_model.supercell_frame(structure, reps)
    if jitter_ang:
        rng = np.random.default_rng(seed)
        cart = frame.frac @ frame.box_ang
        cart = cart + rng.normal(0.0, jitter_ang, cart.shape)
        frame = md_model.frame_from_arrays(
            frame.elements, cart, box_ang=frame.box_ang,
            atom_id=frame.atom_id)
    return frame


FRAMES = [
    ("quartz", "quartz_SiO2_cod9013321.cif", (2, 2, 2), 0.0),
    ("quartz jittered", "quartz_SiO2_cod9013321.cif", (2, 2, 2), 0.05),
    ("cryolite jittered", "cryolite_Na3AlF6_cod9004097.cif", (1, 1, 1), 0.04),
    ("BiPO4 jittered", "bismuth_phosphate_BiPO4_cod9008088.cif", (1, 2, 1),
     0.04),
    ("valentinite", "valentinite_Sb2O3_cod9007587.cif", (1, 1, 2), 0.0),
]
IDS = [f[0] for f in FRAMES]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _reference(frame, **kw):
    """build_scene on the same atoms: frame_to_structure, the model's
    oxidation states, no polyhedra, no cell outline."""
    from facet.core import md_model
    from facet.gl.scene import build_scene

    ox = md_model.model_oxidation(frame.species)
    structure = md_model.frame_to_structure(frame, ox)
    return build_scene(structure, results=[], show_cell=False, **kw), ox


def _bond_ids(scene):
    pairs = scene.bond_atoms.astype(np.int64)
    first = scene.bond_cation == 0
    cation = np.where(first, pairs[:, 0], pairs[:, 1])
    anion = np.where(first, pairs[:, 1], pairs[:, 0])
    return sorted(zip(cation.tolist(), anion.tolist(),
                      np.round(scene.bond_distance.astype(np.float64),
                               5).tolist()))


def _by_bond(scene, values):
    """Values of a per-bond array in the order of _bond_ids."""
    pairs = scene.bond_atoms.astype(np.int64)
    first = scene.bond_cation == 0
    cation = np.where(first, pairs[:, 0], pairs[:, 1])
    anion = np.where(first, pairs[:, 1], pairs[:, 0])
    d = np.round(scene.bond_distance.astype(np.float64), 5)
    order = np.lexsort((d, anion, cation))
    return np.asarray(values)[order]


# ---------------------------------------------------------------------------
# the same bonds as build_scene
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,name,reps,jitter", FRAMES, ids=IDS)
def test_the_bond_set_is_build_scenes(label, name, reps, jitter):
    from facet.ui.md_scene import build_md_scene

    frame = _frame(name, reps, jitter)
    ref, ox = _reference(frame)
    scene = build_md_scene(frame, ox=ox, show_cell=False)
    assert scene.n_bonds == ref.n_bonds > 0
    assert _bond_ids(scene) == _bond_ids(ref)
    # bit-identical valences, matched bond for bond
    assert np.array_equal(_by_bond(scene, scene.bond_valence),
                          _by_bond(ref, ref.bond_valence))
    assert np.allclose(_by_bond(scene, scene.bond_distance),
                       _by_bond(ref, ref.bond_distance), atol=1e-6)
    # the drawn widths and colours follow from the valences the same way
    assert np.array_equal(_by_bond(scene, scene.bond_radius),
                          _by_bond(ref, ref.bond_radius))
    assert np.allclose(_by_bond(scene, scene.bond_color_a),
                       _by_bond(ref, np.where(
                           (ref.bond_cation == 0)[:, None], ref.bond_color_a,
                           ref.bond_color_b)), atol=1e-6)


@pytest.mark.parametrize("label,name,reps,jitter", FRAMES[:3], ids=IDS[:3])
def test_the_atoms_are_build_scenes(label, name, reps, jitter):
    from facet.ui.md_scene import build_md_scene

    frame = _frame(name, reps, jitter)
    ref, ox = _reference(frame)
    scene = build_md_scene(frame, ox=ox, show_cell=False)
    n = frame.n_atoms
    assert scene.n_cell_atoms == ref.n_cell_atoms == n
    assert np.allclose(scene.atom_position[:n], ref.atom_position[:n],
                       atol=1e-5)
    assert np.array_equal(scene.atom_radius[:n], ref.atom_radius[:n])
    assert np.array_equal(scene.atom_color[:n], ref.atom_color[:n])
    assert scene.atom_label[:n] == ref.atom_label[:n]
    assert scene.atom_element[:n] == ref.atom_element[:n]


@pytest.mark.parametrize("label,name,reps,jitter", FRAMES, ids=IDS)
def test_every_bond_ends_on_a_drawn_atom(label, name, reps, jitter):
    """What _add_bonded_images guarantees for a crystal: no bond runs out
    to an atom that is not drawn. The image atoms take their row's site."""
    from scipy.spatial import cKDTree

    from facet.ui.md_scene import build_md_scene

    frame = _frame(name, reps, jitter)
    scene = build_md_scene(frame)
    tree = cKDTree(scene.atom_position.astype(np.float64))
    for ends in (scene.bond_a, scene.bond_b):
        d, _ = tree.query(ends.astype(np.float64))
        assert float(d.max()) < 1e-4
    images = scene.atom_site[frame.n_atoms:]
    assert np.all((images >= 0) & (images < frame.n_atoms))
    # an image is drawn as the atom it is an image of
    rows = scene.atom_site
    assert [scene.atom_element[i] for i in range(scene.n_atoms)] == \
        [str(frame.elements[r]) for r in rows]


@pytest.mark.parametrize("v_bond", [0.02, 0.1, 0.3])
def test_moving_the_threshold_restyles_as_build_scene_does(v_bond):
    from facet.ui.md_scene import build_md_scene

    frame = _frame("quartz_SiO2_cod9013321.cif", (2, 2, 2), 0.05)
    ref, ox = _reference(frame)
    scene = build_md_scene(frame, ox=ox, show_cell=False)
    ref.restyle(v_bond)
    scene.restyle(v_bond)
    assert scene.bonds_above_threshold() == ref.bonds_above_threshold()
    assert np.array_equal(_by_bond(scene, scene.bond_radius),
                          _by_bond(ref, ref.bond_radius))
    # and the scene built at that threshold is the restyled one
    direct = build_md_scene(frame, ox=ox, show_cell=False, v_bond=v_bond)
    assert np.array_equal(direct.bond_radius, scene.bond_radius)


# ---------------------------------------------------------------------------
# what build_md_scene takes as its bonds
# ---------------------------------------------------------------------------

def test_a_valence_table_from_the_analysis_gives_the_same_scene():
    """The workspace hands over bulk.analyse_frame's table (searched with
    the 6 A floor); the scene is the one built from nothing."""
    from facet.core import bulk, md_model
    from facet.ui.md_scene import build_md_scene

    frame = _frame("cryolite_Na3AlF6_cod9004097.cif", (1, 1, 1), 0.04)
    ox = md_model.model_oxidation(frame.species)
    table, _results = bulk.analyse_frame(frame, ox.per_atom(frame.elements))
    from_table = build_md_scene(frame, table)
    from_nothing = build_md_scene(frame, ox=ox)
    assert _bond_ids(from_table) == _bond_ids(from_nothing)
    assert np.array_equal(np.sort(from_table.bond_valence),
                          np.sort(from_nothing.bond_valence))
    assert from_table.n_atoms == from_nothing.n_atoms


def test_bonds_at_a_threshold_draw_only_those_bonds():
    from facet.core import bulk, md_model
    from facet.ui.md_scene import build_md_scene

    frame = _frame("quartz_SiO2_cod9013321.cif", (2, 2, 2), 0.05)
    ox = md_model.model_oxidation(frame.species)
    table, _ = bulk.analyse_frame(frame, ox.per_atom(frame.elements))
    bonds = bulk.bonds_at(table, 0.2)
    scene = build_md_scene(frame, bonds, v_bond=0.2)
    assert scene.n_bonds == len(bonds) > 0
    assert float(scene.bond_valence.min()) > 0.2
    assert scene.bonds_above_threshold() == len(bonds)


def test_no_bonds_draws_the_atoms(qapp):
    """md_jobs hands an empty bulk.Bonds above its atoms-only limit."""
    from facet.core import bulk
    from facet.ui.md_scene import build_md_scene

    frame = _frame("quartz_SiO2_cod9013321.cif", (2, 2, 2))
    empty = np.zeros(0, np.int32)
    none = bulk.Bonds(cation=empty, anion=empty.copy(),
                      image=np.zeros((0, 3), np.int16),
                      vec_ang=np.zeros((0, 3)), d_ang=np.zeros(0),
                      v_vu=np.zeros(0), v_bond_vu=0.04)
    scene = build_md_scene(frame, none)
    assert scene.n_bonds == 0
    assert scene.n_atoms == frame.n_atoms
    assert len(scene.cell_segments) == 12


def test_bad_bonds_are_refused():
    from facet.core import bulk, md_model
    from facet.ui.md_scene import build_md_scene

    frame = _frame("quartz_SiO2_cod9013321.cif", (1, 1, 1))
    with pytest.raises(ValueError):
        build_md_scene(frame, [(0, 1)])
    other = _frame("quartz_SiO2_cod9013321.cif", (2, 1, 1))
    ox = md_model.model_oxidation(other.species)
    table, _ = bulk.analyse_frame(other, ox.per_atom(other.elements))
    with pytest.raises(ValueError, match="atoms"):
        build_md_scene(frame, table)
    with pytest.raises(ValueError):
        build_md_scene("not a frame")


# ---------------------------------------------------------------------------
# the theme's colour modes
# ---------------------------------------------------------------------------

def test_bond_colour_modes_match_build_scene():
    from facet.core import theme as theme_mod
    from facet.ui.md_scene import build_md_scene

    frame = _frame("quartz_SiO2_cod9013321.cif", (2, 2, 2), 0.05)
    for mode in theme_mod.BondColorMode:
        theme = theme_mod.Theme()
        theme.bond_color_mode = mode
        ref, ox = _reference(frame, theme=theme)
        scene = build_md_scene(frame, ox=ox, theme=theme, show_cell=False)
        ref_a = np.where((ref.bond_cation == 0)[:, None], ref.bond_color_a,
                         ref.bond_color_b)
        assert np.allclose(_by_bond(scene, scene.bond_color_a),
                           _by_bond(ref, ref_a), atol=1e-6), mode


def test_value_colour_modes_match_build_scene():
    """BVS, CN, phi and valence discrepancy colour the cations from the
    table at v_bond, as build_scene does from the per-site results."""
    from facet.core import coordination, md_model, theme as theme_mod
    from facet.gl.scene import build_scene
    from facet.ui.md_scene import build_md_scene

    frame = _frame("bismuth_phosphate_BiPO4_cod9008088.cif", (1, 1, 1), 0.04)
    ox = md_model.model_oxidation(frame.species)
    structure = md_model.frame_to_structure(frame, ox)
    results = coordination.analyse_structure(structure,
                                             resolve_oxidation=False)
    n = frame.n_atoms
    for mode in (theme_mod.ColorMode.BVS, theme_mod.ColorMode.COORDINATION,
                 theme_mod.ColorMode.PHI,
                 theme_mod.ColorMode.VALENCE_DISCREPANCY,
                 theme_mod.ColorMode.UNIFORM, theme_mod.ColorMode.SITE):
        theme = theme_mod.Theme()
        theme.color_mode = mode
        ref = build_scene(structure, results=results, show_cell=False,
                          theme=theme)
        scene = build_md_scene(frame, ox=ox, theme=theme, show_cell=False)
        assert np.allclose(scene.atom_color[:n], ref.atom_color[:n],
                           atol=1e-5), mode


# ---------------------------------------------------------------------------
# it draws on the QPainter tier, and it is fast
# ---------------------------------------------------------------------------

def test_the_scene_draws_on_the_qpainter_tier(qapp, tmp_path):
    from facet.gl.view import StructureView
    from facet.ui.md_scene import build_md_scene

    frame = _frame("quartz_SiO2_cod9013321.cif", (3, 3, 3), 0.05)
    scene = build_md_scene(frame)
    view = StructureView()
    try:
        view.resize(640, 480)
        view._enter_fallback("no OpenGL in this test")
        view.set_scene(scene)
        image = view.grab_image(640, 480, supersample=1)
        assert not image.isNull()
        colours = {image.pixel(x, y) for x in range(0, 640, 4)
                   for y in range(0, 480, 4)}
        assert len(colours) > 50
        if SHOTS:
            image.save(str(Path(SHOTS) / "md_scene_painter_quartz_3x3x3.png"))
    finally:
        dispose(view)


def test_timing_against_build_scene(record_property, capsys):
    """Recorded, not tuned: the bulk path against the crystal path on the
    same frame. The assertion leaves a margin of ten over measured ratios of
    about a hundred, so that a loaded machine does not fail it."""
    from facet.core import md_model
    from facet.gl.scene import build_scene
    from facet.ui.md_scene import build_md_scene, valence_table_for

    frame = _frame("quartz_SiO2_cod9013321.cif", (4, 4, 4), 0.05)
    ox = md_model.model_oxidation(frame.species)
    build_md_scene(frame, ox=ox)                      # imports, first call
    clock = time.perf_counter()
    build_md_scene(frame, ox=ox)
    md_s = time.perf_counter() - clock
    table = valence_table_for(frame, ox)
    clock = time.perf_counter()
    build_md_scene(frame, table)
    from_table_s = time.perf_counter() - clock
    structure = md_model.frame_to_structure(frame, ox)
    clock = time.perf_counter()
    build_scene(structure, results=[], show_cell=False)
    crystal_s = time.perf_counter() - clock
    record_property("n_atoms", frame.n_atoms)
    record_property("build_md_scene_s", round(md_s, 4))
    record_property("build_md_scene_from_table_s", round(from_table_s, 4))
    record_property("build_scene_s", round(crystal_s, 4))
    with capsys.disabled():
        print(f"\n[md_scene timing] {frame.n_atoms} atoms: build_md_scene "
              f"{md_s * 1e3:.1f} ms (from a table {from_table_s * 1e3:.1f} "
              f"ms), build_scene {crystal_s * 1e3:.1f} ms")
    assert md_s * 10 < crystal_s
