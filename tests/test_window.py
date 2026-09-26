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

SAMPLE = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs"
              r"\1526458_Bi2O3.cif")
SECOND = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs"
              r"\1004091_BiNa3O8P2.cif")

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
    w.close()


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
    assert titles == ["Site", "Utilities", "Diffraction", "Planes",
                      "Overrides", "Appearance"]


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
