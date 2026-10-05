"""The main window, driven end to end.

These are the tests that were missing when preview.py stopped compiling: every
panel had its own tests, but nothing built the window that assembles them. They
go through the window's own methods rather than synthesising Qt events, which
keeps them fast and still exercises the wiring -- the point is that a context
menu action reaches the scene, and that undo puts it back.

No OpenGL context is needed: the window is built but never shown, and the scene
is checked as data rather than as pixels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import dispose, sample_cif

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")
SECOND = sample_cif("1004091", "1004091_BiNa3O8P2.cif")

pytestmark = pytest.mark.skipif(not SAMPLE.is_file(),
                                reason="the sample structure is not present")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    from facet.ui.preview import PreviewWindow

    w = PreviewWindow()
    w.load(str(SAMPLE))
    yield w
    dispose(w)


# ---------------------------------------------------------------------------
# it assembles and loads
# ---------------------------------------------------------------------------

def test_the_window_loads_a_structure_and_fills_its_panels(window):
    assert window.structure is not None
    assert window.scene is not None
    assert window.scene.n_atoms > 0
    # the site list holds the sites a coordination number is reported for,
    # which is the cations, not every site in the file
    assert window.site_list.count() == len(window.results)
    assert 0 < window.site_list.count() < len(window.structure.sites)
    # every panel got the structure
    assert window.diffraction.pattern is not None
    assert window.planes_panel.structure is window.structure
    assert window.overrides_panel.structure is window.structure
    assert window.overrides_panel.overrides is window.project.current.overrides


def test_loading_starts_a_history_with_nothing_to_undo(window):
    assert len(window.history) == 1
    assert not window.history.can_undo
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()


def test_every_tab_is_present(window):
    from PySide6.QtWidgets import QTabWidget

    tabs = window.findChild(QTabWidget)
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    # A deliberate inventory rather than a subset check: it catches a tab
    # accidentally removed as well as one added, and the order is the reading
    # order of the workflow.
    assert titles == ["Site", "Tools", "Diffraction", "PDF", "EXAFS",
                      "Planes", "Styles", "Volume", "Disorder", "Theme"]
    assert len(set(titles)) == len(titles)


# ---------------------------------------------------------------------------
# overrides through the window
# ---------------------------------------------------------------------------

def test_hiding_a_site_reaches_the_scene_and_records_history(window):
    label = window.structure.sites[0].label
    before = window.scene.n_bonds

    window._apply_override("site", label, "hid", visible=False)

    hidden = [i for i in range(window.scene.n_atoms)
              if window.scene.atom_label[i] == label]
    assert hidden
    assert all(window.scene.atom_radius[i] == 0.0 for i in hidden)
    assert window.scene.n_bonds < before

    assert window.history.can_undo
    assert label in window.history.undo_description
    assert window.undo_action.isEnabled()


def test_undo_restores_the_scene(window):
    label = window.structure.sites[0].label
    radii = window.scene.atom_radius.copy()
    colours = window.scene.atom_color.copy()
    bonds = window.scene.n_bonds

    window._apply_override("site", label, "hid", visible=False)
    assert window.scene.n_bonds != bonds

    window._undo()
    assert np.allclose(window.scene.atom_radius, radii)
    assert np.allclose(window.scene.atom_color, colours)
    assert window.scene.n_bonds == bonds
    assert window.project.current.overrides.is_empty


def test_redo_reapplies_it(window):
    label = window.structure.sites[0].label
    window._apply_override("site", label, "hid", visible=False)
    window._undo()
    assert window.redo_action.isEnabled()
    window._redo()
    hidden = [i for i in range(window.scene.n_atoms)
              if window.scene.atom_label[i] == label]
    assert all(window.scene.atom_radius[i] == 0.0 for i in hidden)


def test_several_steps_undo_in_order(window):
    labels = [s.label for s in window.structure.sites[:3]]
    for label in labels:
        window._apply_override("site", label, "hid", visible=False)
    assert window.project.current.overrides.count() == 3

    for expected in reversed(range(3)):
        window._undo()
        assert window.project.current.overrides.count() == expected
    assert not window.history.can_undo


def test_scaling_compounds_and_undoes_one_step_at_a_time(window):
    label = window.structure.sites[0].label
    window._scale_override("site", label, 1.25)
    window._scale_override("site", label, 1.25)
    style = window.project.current.overrides.by_site[label]
    assert style.radius_scale == pytest.approx(1.25 * 1.25)

    window._undo()
    style = window.project.current.overrides.by_site[label]
    assert style.radius_scale == pytest.approx(1.25)


def test_an_atom_override_reaches_exactly_that_atom(window):
    window._apply_override("atom", 2, "coloured", color=(1.0, 0.0, 1.0))
    assert np.allclose(window.scene.atom_color[2], (1.0, 0.0, 1.0))
    window._undo()
    assert not np.allclose(window.scene.atom_color[2], (1.0, 0.0, 1.0))


def test_clearing_an_override_is_itself_undoable(window):
    label = window.structure.sites[0].label
    window._apply_override("site", label, "coloured", color=(1.0, 0.0, 0.0))
    window._clear_override("site", label)
    assert window.project.current.overrides.is_empty
    window._undo()
    assert not window.project.current.overrides.is_empty


def test_remove_every_override_is_one_undoable_step(window):
    for site in window.structure.sites[:3]:
        window._apply_override("site", site.label, "hid", visible=False)
    window._clear_overrides()
    assert window.project.current.overrides.is_empty
    window._undo()
    assert window.project.current.overrides.count() == 3


def test_the_overrides_panel_lists_what_is_in_force(window):
    label = window.structure.sites[0].label
    window._apply_override("site", label, "hid", visible=False)
    rows = [window.overrides_panel.table.item(r, 1).text()
            for r in range(window.overrides_panel.table.rowCount())]
    assert label in rows
    assert "1 override" in window.overrides_panel.count_label.text()


# ---------------------------------------------------------------------------
# the context menu
# ---------------------------------------------------------------------------

def test_the_context_menu_offers_the_right_actions_on_an_atom(window):
    """Built and inspected without entering its modal event loop."""
    menu = window._build_context_menu(0)
    assert menu is not None, "no menu was built"
    texts = [a.text() for a in menu.actions()]
    assert any("Select this site" in t for t in texts)
    assert any("polyhedron" in t for t in texts)
    assert any("Look along a" in t for t in texts)
    # the first entry names the atom and is not clickable
    header = menu.actions()[0]
    assert window.scene.atom_label[0] in header.text()
    assert not header.isEnabled()

    submenus = [a.text() for a in menu.actions() if a.menu() is not None]
    assert any("Site" in t for t in submenus)
    assert any("This atom only" in t for t in submenus)
    assert any("Every" in t for t in submenus)


def test_the_context_menu_on_empty_space_offers_only_view_actions(window):
    menu = window._build_context_menu(-1)
    texts = [a.text() for a in menu.actions()]
    assert any("Reset the view" in t for t in texts)
    assert not any("Select this site" in t for t in texts)
    assert not any(a.menu() is not None for a in menu.actions())


def test_a_context_menu_action_hides_the_site_it_names(window):
    menu = window._build_context_menu(0)
    site_index = int(window.scene.atom_site[0])
    site_label = window.structure.sites[site_index].label
    site_menu = next(a.menu() for a in menu.actions()
                     if a.menu() is not None and site_label in a.text())
    hide = next(a for a in site_menu.actions() if a.text() == "Hide")
    hide.trigger()

    hidden = [i for i in range(window.scene.n_atoms)
              if window.scene.atom_label[i] == site_label]
    assert hidden
    assert all(window.scene.atom_radius[i] == 0.0 for i in hidden)


def test_the_menu_offers_undo_only_when_there_is_something_to_undo(window):
    def raise_menu():
        return [a.text() for a in window._build_context_menu(-1).actions()]

    assert not any(t.startswith("Undo") for t in raise_menu())
    window._apply_override("site", window.structure.sites[0].label,
                           "hid", visible=False)
    assert any(t.startswith("Undo") for t in raise_menu())


# ---------------------------------------------------------------------------
# planes through the window
# ---------------------------------------------------------------------------

def test_adding_a_plane_reaches_the_scene_and_undoes(window):
    window.planes_panel.add_plane(0, 0, 1)
    assert window.scene.n_planes >= 1
    assert "plane" in window.history.undo_description
    window._undo()
    assert window.scene.n_planes == 0


def test_enabling_a_slab_reduces_the_scene_and_undoes(window):
    before = window.scene.n_atoms
    window.planes_panel.slab_l.setValue(1)
    window.planes_panel.slab_thickness.setValue(1.8)
    window.planes_panel.slab_on.setChecked(True)
    assert window.scene.n_atoms < before

    window._undo()
    assert window.scene.n_atoms == before
    assert not window.planes_panel.slab_on.isChecked()


def test_undoing_a_slab_puts_the_controls_back_without_recording_more(window):
    window.planes_panel.slab_thickness.setValue(2.5)
    window.planes_panel.slab_on.setChecked(True)
    steps = len(window.history)
    window._undo()
    assert len(window.history) == steps, "undo recorded a step of its own"
    assert window.planes_panel.slab_thickness.value() != 2.5 or \
        not window.planes_panel.slab_on.isChecked()


def test_choosing_a_reflection_turns_the_view(window):
    """The plane normal is reciprocal, so this must not use h a + k b + l c."""
    before = window.view.camera.orientation.copy()
    window._on_reflection(1, 1, 1)
    assert not np.allclose(window.view.camera.orientation, before)


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

def test_a_session_round_trips_the_overrides_and_the_planes(window, tmp_path,
                                                           monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    label = window.structure.sites[0].label
    window._apply_override("site", label, "coloured", color=(1.0, 0.0, 0.0))
    window._apply_override("atom", 1, "hid", visible=False)
    window.planes_panel.add_plane(1, 0, 1)
    window.planes_panel.slab_thickness.setValue(3.25)
    window.planes_panel.slab_on.setChecked(True)

    target = tmp_path / "session.json"
    monkeypatch.setattr(window, "_ask", lambda *a, **k: str(target))
    window._save_session()
    assert target.is_file()

    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["entries"][0]["overrides"]["by_site"][label]["color"]
    assert data["presentation"]["planes"][0]["h"] == 1
    assert data["presentation"]["slab"]["enabled"] is True

    # now reopen it into a fresh window
    from facet.ui.preview import PreviewWindow

    other = PreviewWindow()
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    try:
        other._open_session()
        overrides = other.project.current.overrides
        assert overrides.by_site[label].color == pytest.approx((1.0, 0.0, 0.0))
        assert overrides.by_atom[1].visible is False
        assert [p.hkl for p in other.planes_panel.current_planes()] == ["1 0 1"]
        assert other.planes_panel.slab.enabled
        assert other.planes_panel.slab.thickness == pytest.approx(3.25)
        # and the restored state is the start of its own history
        assert not other.history.can_undo
    finally:
        other.close()


def test_overrides_are_per_structure(window, qapp):
    """A label means something different in another file."""
    if not SECOND.is_file():
        pytest.skip("the second sample structure is not present")
    label = window.structure.sites[0].label
    window._apply_override("site", label, "hid", visible=False)
    first = window.project.current

    window.load(str(SECOND))
    second = window.project.current
    assert second is not first
    assert second.overrides.is_empty
    assert not first.overrides.is_empty


def test_a_stale_atom_override_is_pruned_on_load(window):
    """An atom index that no longer addresses an atom must be dropped.

    Keeping it would apply the override to whatever atom happened to land on
    that index in the next structure, which is worse than losing it.
    """
    window.project.current.overrides.set_atom(100000, visible=False)
    window._after_load()
    assert 100000 not in window.project.current.overrides.by_atom


# ---------------------------------------------------------------------------
# the rotation centre
# ---------------------------------------------------------------------------

def _outermost(scene):
    """The atom furthest from the middle -- the case that breaks things."""
    offsets = np.linalg.norm(scene.atom_position - scene.center, axis=1)
    return int(np.argmax(offsets))


def test_an_atom_can_be_made_the_centre_of_rotation(window):
    scene = window.scene
    index = _outermost(scene)
    position = scene.atom_position[index].astype(float).copy()

    window._center_on_atom(index, scene.atom_label[index])
    assert window.view.camera.target == pytest.approx(position, abs=1e-6)
    assert window.view.pivot_caption == scene.atom_label[index]
    assert scene.atom_label[index] in window.pivot_label.text()


def test_choosing_a_centre_does_not_move_the_structure_in_depth(window):
    """A pan, not a zoom: every drawn point keeps the depth it had."""
    scene = window.scene
    camera = window.view.camera
    points = np.asarray(scene.atom_position, float)

    def depths():
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        return -(homogeneous @ camera.view_matrix().T)[:, 2]

    before = depths()
    window._center_on_atom(_outermost(scene), "x")
    assert np.abs(before - depths()).max() < 1e-6


def test_the_centre_survives_every_rebuild(window, qapp):
    """A rebuild happens on nearly every interaction, so it must not disturb it.

    This is why the centre is a position and not an atom index: an index into
    the drawn scene is renumbered by a change of threshold, of cell range or by
    a slab, and would silently come to mean a different atom.
    """
    scene = window.scene
    index = _outermost(scene)
    position = scene.atom_position[index].astype(float).copy()
    window._center_on_atom(index, scene.atom_label[index])

    for act in (lambda: window._on_threshold(0.02),
                lambda: window.style_box.setCurrentIndex(1),
                lambda: window.cell_check.setChecked(False),
                lambda: window.site_list.setCurrentRow(0),
                window._rebuild):
        act()
        qapp.processEvents()
        assert window.view.camera.target == pytest.approx(position, abs=1e-6)
    assert window.view.pivot_caption


def test_a_deliberate_reframe_clears_the_centre(window, qapp):
    """Reset view and a newly framed structure overwrite the pivot.

    They must also clear the record, or the read-out names an atom the camera
    is no longer turning about.
    """
    scene = window.scene
    window._center_on_atom(_outermost(scene), scene.atom_label[0])
    assert window.view.pivot_caption

    window.view.set_scene(window.scene, reframe=True)
    qapp.processEvents()
    assert window.view.pivot_caption is None
    assert window.pivot_label.text() == ""


def test_the_cell_centre_is_the_cell_and_not_the_atoms(window):
    """Two different points, and the action is named for the one it uses.

    `scene.center` is the centroid of everything drawn. For a centrosymmetric
    cell it coincides with the middle of the cell box; for one that is not, the
    two can be nearly 2 A apart, so an action called "the cell centre" that
    used the centroid would be a false label.
    """
    centre = window._cell_centre()
    assert centre is not None
    cell = window.structure.cell
    expected = cell.orth @ np.array([0.5, 0.5, 0.5])
    assert centre == pytest.approx(expected, abs=1e-9)

    window._center_on_cell()
    assert window.view.camera.target == pytest.approx(expected, abs=1e-6)
    assert window.pivot_label.text() == ""


def test_the_cell_centre_follows_the_cell_range(window, qapp):
    for box in window.range_boxes:
        box.setValue(2)
    qapp.processEvents()
    cell = window.structure.cell
    assert window._cell_centre() == pytest.approx(
        cell.orth @ np.array([1.0, 1.0, 1.0]), abs=1e-9)
    for box in window.range_boxes:
        box.setValue(1)
    qapp.processEvents()


def test_escape_clears_the_centre(window, qapp):
    """Escape is the application's "clear what I set" gesture."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    window._center_on_atom(_outermost(window.scene), "Bi1")
    assert window.view.pivot_caption
    window.view.keyPressEvent(
        QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    qapp.processEvents()
    assert window.view.pivot_caption is None


def test_the_camera_learns_the_new_extent_on_every_rebuild(window, qapp):
    """The bug this uncovered, which shipped: a stale radius clips the scene.

    `set_scene` refreshed the camera's idea of the scene only when it reframed,
    and nearly every rebuild does not. Growing the cell range then left the
    camera believing the structure was one cell wide, and the clip planes are
    that radius.
    """
    for box in window.range_boxes:
        box.setValue(3)
    qapp.processEvents()
    assert window.view.camera.scene_radius == pytest.approx(
        window.scene.radius, rel=1e-9)

    points = np.asarray(window.scene.atom_position, float)
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    near, far = window.view.camera.clip_planes()
    depth = -(homogeneous @ window.view.camera.view_matrix().T)[:, 2]
    outside = int(((depth < near) | (depth > far)).sum())
    assert outside == 0, f"{outside} of {len(points)} atoms are clipped away"

    for box in window.range_boxes:
        box.setValue(1)
    qapp.processEvents()


def test_the_context_menu_offers_the_rotation_centre(window):
    scene = window.scene
    menu = window._build_context_menu(_outermost(scene))
    assert menu is not None
    texts = [a.text() for a in menu.actions()]
    assert any(t.startswith("Rotate about") for t in texts), texts
    assert "Rotate about the cell centre" in texts


def test_closing_the_last_structure_clears_the_viewport(window, qapp):
    """The window let go of the scene and the viewport did not.

    `_on_remove` set `self.scene = None` and never told the view, so after
    closing the only open file the 3-D view carried on drawing it -- 44 atoms
    of a structure the application no longer considered open, with the panels
    beside it empty.
    """
    assert window.view.scene is not None
    window._on_remove(0)
    qapp.processEvents()
    assert window.scene is None
    assert window.view.scene is None, (
        "the viewport is still holding the structure that was closed")
    assert window.site_list.count() == 0


# ---------------------------------------------------------------------------
# reaching a site that cannot be clicked
# ---------------------------------------------------------------------------

def test_the_site_list_centres_on_the_copy_nearest_the_cell_centre(window):
    """A site is drawn once per repeat, and they are not interchangeable.

    Turning about a copy at the edge of the block puts the rest of the
    structure off to one side. The copy nearest the middle is the one the eye
    takes as the site.
    """
    scene = window.scene
    centre = np.asarray(window._cell_centre(), float)
    for row in range(window.site_list.count()):
        site = window._rows[row]
        index = window._atom_of_site(site)
        assert index is not None
        assert int(scene.atom_site[index]) == site

        copies = np.flatnonzero(np.asarray(scene.atom_site) == site)
        assert len(copies) > 1, "this site is drawn only once; nothing to choose"
        distances = np.linalg.norm(scene.atom_position[copies] - centre, axis=1)
        chosen = np.linalg.norm(scene.atom_position[index] - centre)
        assert chosen == pytest.approx(distances.min())


def test_double_clicking_a_site_turns_the_view_about_it(window, qapp):
    """The site list names every site; it is how you reach one you cannot see.

    The rotation centre could otherwise only be set by right-clicking the atom
    in the 3-D view, which needs it to be visible, findable and clickable --
    and the site worth turning about is often the one that is none of those.
    """
    for row in range(window.site_list.count()):
        item = window.site_list.item(row)
        index = window._atom_of_site(window._rows[row])
        position = window.scene.atom_position[index].astype(float).copy()

        window._on_site_double_click(item)
        qapp.processEvents()
        assert window.view.camera.target == pytest.approx(position, abs=1e-9)
        assert window.view.pivot_caption == window.scene.atom_label[index]


def test_the_site_list_menu_offers_the_centre(window):
    menu = window._build_site_menu(0)
    assert menu is not None
    texts = [a.text() for a in menu.actions()]
    assert any(t.startswith("Rotate about") for t in texts), texts
    assert "Rotate about the cell centre" in texts

    index = window._atom_of_site(window._rows[0])
    assert texts[0] == f"Rotate about {window.scene.atom_label[index]}"
    assert window._build_site_menu(-1) is None
    assert window._build_site_menu(10_000) is None


def test_showing_an_atom_brings_it_into_view(window, qapp):
    """"Show me this atom" used to draw a ring and nothing else.

    Which shows nothing at all when the atom is behind the structure or outside
    the frame -- the two cases where being shown it is the point. The overrides
    panel asks for this by double-clicking an atom override.
    """
    scene = window.scene
    far = int(np.argmax(np.linalg.norm(
        scene.atom_position - scene.center, axis=1)))
    before = window.view.camera.target.copy()

    window._focus_atom(far)
    qapp.processEvents()

    assert window.view._selected == far
    assert np.linalg.norm(window.view.camera.target - before) > 1e-6, (
        "the camera did not move, so the atom was not shown")
    assert window.view.camera.target == pytest.approx(
        scene.atom_position[far].astype(float), abs=1e-9)


def test_centring_on_a_site_that_is_not_drawn_says_so(window, qapp):
    """Rather than moving the view somewhere arbitrary, or doing nothing."""
    window._center_on_site(10_000)
    qapp.processEvents()
    assert "not drawn" in window.statusBar().currentMessage()


# ---------------------------------------------------------------------------
# the health checks, which never ran
# ---------------------------------------------------------------------------

NPD = sample_cif("7023720", "7023720_BiPO4.cif")


def _file_tab(window):
    """What the File tab shows once it is the tab on screen.

    Shown rather than merely asked for, because the panel computes a tab when
    it is shown and not before -- recomputing all ten on every change cost 3.8
    seconds on a 484-atom structure.
    """
    panel = window.utilities
    titles = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
    assert "File" in titles, titles
    panel.tabs.setCurrentIndex(titles.index("File"))
    return panel.file_report.toPlainText()


def test_the_health_checks_run_at_all(window):
    """They were written, tested, and never called.

    facet/core/quality.py is 350 lines of checks for the things that make a
    number untrustworthy, and `Structure.issues` carries a comment saying the
    module fills it. Until the File tab existed, the module was imported by the
    test suite and by nothing else: no check had ever run in the application,
    so none of it had ever reached anyone using it.
    """
    text = _file_tab(window)
    assert text.strip(), "the File tab is empty"
    assert window.structure.issues is not None


def test_the_file_tab_reports_a_broken_file(window, qapp):
    if not NPD.is_file():
        pytest.skip("the reference structure is not present")
    window.load(str(NPD))
    qapp.processEvents()

    text = _file_tab(window)
    assert "P1" in text
    assert "not positive definite" in text
    codes = {f.code for f in window.structure.issues}
    assert "npd-displacement" in codes


def test_the_file_tab_states_the_provenance(window):
    text = _file_tab(window)
    assert "Where this came from" in text
    assert Path(window.structure.source_path).name in text


def test_the_file_tab_gives_the_displacement_table(window, qapp):
    if not NPD.is_file():
        pytest.skip("the reference structure is not present")
    window.load(str(NPD))
    qapp.processEvents()

    text = _file_tab(window)
    assert "Displacement" in text
    for label in ("Bi1", "O1", "O2", "O3"):
        assert label in text
    # the site whose tensor is not an ellipsoid shows eigenvalues, not an
    # r.m.s. it does not have
    assert "eigenvalues" in text


def test_the_file_tab_passes_no_verdict(window, qapp):
    """The standing rule, on the one panel most tempted to break it."""
    if not NPD.is_file():
        pytest.skip("the reference structure is not present")
    window.load(str(NPD))
    qapp.processEvents()

    low = _file_tab(window).lower()
    for word in ("unusable", "unreliable", "bad file", "wrong", "invalid",
                 "should be", "do not trust", "poor"):
        assert word not in low, word


# ---------------------------------------------------------------------------
# a restored session's theme
# ---------------------------------------------------------------------------

def test_a_new_window_themes_the_panels_that_draw_their_own_text(window):
    """Five panels compose HTML by hand and fall back to near-black.

    That is right for the four light themes, which is why it never showed, and
    unreadable on the two dark ones.
    """
    assert window.utilities.theme is not None
    assert window.exafs_panel.theme is not None
    assert window.view.theme is not None


def test_restoring_a_session_applies_its_theme_everywhere(window, qapp, tmp_path):
    """The restore set `self.theme` and told the theme panel, and stopped.

    The viewport kept the theme it opened with and the HTML panels kept none,
    so a session saved on Dark came back as dark panels with black text in
    them.
    """
    from facet.core import exporters, theme as theme_mod

    dark = theme_mod.PRESETS["Dark"]()
    assert not dark.is_light_background

    window._apply_theme_everywhere(dark)
    window.view.set_theme(dark)
    path = tmp_path / "dark.json"
    exporters.save_session(window.project, path, theme=window.theme)

    from facet.ui.preview import PreviewWindow

    fresh = PreviewWindow()
    try:
        assert fresh.theme.is_light_background      # opens light
        fresh.restore_session(exporters.load_session(path))
        qapp.processEvents()

        assert not fresh.theme.is_light_background
        assert fresh.view.theme.name == dark.name
        assert fresh.utilities.theme.name == dark.name
        assert fresh.exafs_panel.theme.name == dark.name

        # and the ink the panels will actually write with is light
        from facet.ui import chrome
        ink = chrome.text_hex(fresh.utilities.theme)
        ground = chrome.ui_colors(fresh.utilities.theme).base
        assert chrome.contrast(_rgb(ink), _rgb(ground)) > 4.5, (
            f"ink {ink} on {ground}")
    finally:
        dispose(fresh)


def _rgb(hex_colour: str):
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def test_only_the_tab_on_screen_is_computed(window, qapp):
    """Ten tabs recomputed on every change cost 3.8 s on a 484-atom structure.

    `update_for` runs on every move of the bond-valence threshold, every change
    of site and every rebuild, so the window froze for seconds a step over
    figures that were behind another tab. The tab on screen is brought up to
    date at once; the rest when they are shown.
    """
    panel = window.utilities
    titles = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
    assert set(panel.REFRESHERS) == set(titles), (
        "a tab with no refresher would never be computed at all")

    window._on_threshold(0.02)
    qapp.processEvents()
    current = panel.tabs.tabText(panel.tabs.currentIndex())
    assert current not in panel._stale
    assert panel._stale == set(titles) - {current}

    other = next(t for t in titles if t != current)
    panel.tabs.setCurrentIndex(titles.index(other))
    qapp.processEvents()
    assert other not in panel._stale

    panel.refresh_every_tab()
    assert panel._stale == set()


def test_a_tab_can_be_asked_for_without_being_shown(window, qapp):
    """An export needs a tab's contents and must not have to show it."""
    panel = window.utilities
    window._on_threshold(0.03)
    qapp.processEvents()
    assert "Angles" in panel._stale

    panel.refresh_tab("Angles")
    assert "Angles" not in panel._stale
    panel.refresh_tab("Angles")            # asking twice is not an error
    panel.refresh_tab("not a tab")         # nor is asking for one that is gone


# ---------------------------------------------------------------------------
# a priori bond valences in the interface
# ---------------------------------------------------------------------------

def test_the_site_panel_shows_the_split(window, qapp):
    """The two indices, and what each bond would be from the topology alone."""
    window.site_list.setCurrentRow(0)
    qapp.processEvents()
    text = window.analysis.toPlainText()
    assert "topol" in text and "cryst" in text
    assert "a priori" in text

    rows, reason = window.project.network_for(window.project.current)
    assert reason == "", reason
    site = rows[window.site_index]
    assert f"{site.delta_topol:.3f}" in text
    assert f"{site.delta_cryst:.3f}" in text
    # the a priori valences of a site sum to its formal charge
    assert sum(site.a_priori) == pytest.approx(abs(site.ox), abs=1e-9)


def test_the_tools_tab_lists_every_site(window, qapp):
    panel = window.utilities
    titles = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
    assert "A priori" in titles
    panel.tabs.setCurrentIndex(titles.index("A priori"))
    qapp.processEvents()

    rows, _ = window.project.network_for(window.project.current)
    assert panel.apriori.rowCount() == len(rows) > 0
    labels = {panel.apriori.item(i, 0).text()
              for i in range(panel.apriori.rowCount())}
    assert labels == {r.label for r in rows.values()}


def test_a_structure_whose_topology_does_not_close_says_so(window, qapp):
    """Not a blank table and not a number: the reason it cannot be computed.

    gamma-Bi2O3 as COD 2100844 gives it: the cell carries a net charge of +2 e,
    so the valence-sum rule cannot hold at every site at once and the network
    equations have no solution. A real file rather than a seeded cache, because
    the seeding is what the panel's own rebuild undoes.
    """
    unbalanced = sample_cif("2100844", "2100844_Bi2O3.cif")
    if not unbalanced.is_file():
        pytest.skip("no structure with an unbalanced cell to hand")

    window.load(str(unbalanced))
    qapp.processEvents()
    rows, reason = window.project.network_for(window.project.current)
    assert not rows and "net charge" in reason

    window.site_list.setCurrentRow(0)
    qapp.processEvents()
    text = window.analysis.toPlainText()
    assert "No a priori bond valences" in text
    assert "net charge" in text

    panel = window.utilities
    titles = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
    panel.tabs.setCurrentIndex(titles.index("A priori"))
    qapp.processEvents()
    assert panel.apriori.rowCount() == 1
    assert "No a priori bond valences" in panel.apriori.item(0, 0).text()


def test_the_a_priori_analysis_is_computed_once(window, qapp):
    """It needs a second pass over the structure, so it is cached."""
    entry = window.project.current
    first = window.project.network_for(entry)
    again = window.project.network_for(entry)
    assert first is again

    window.project.set_list_threshold(window.project.v_list * 1.5)
    third = window.project.network_for(window.project.current)
    assert third is not first
