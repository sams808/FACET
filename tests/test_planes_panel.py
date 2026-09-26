"""The planes panel, and planes drawn on both render tiers.

The panel is tested through its own controls rather than through the window,
because the property that matters is that a control change produces the right
plane in the scene -- and that the panel reports what is on a plane without
saying anything about it.

The drawing is tested on both tiers. The software tier matters as much as the
GL one here: the promise is that FACET runs on a machine with no graphics card,
and a feature that only draws under OpenGL quietly breaks that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SAMPLE = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs"
              r"\1526458_Bi2O3.cif")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def structure():
    if not SAMPLE.is_file():
        pytest.skip("the sample structure is not present")
    from facet.core import cif

    return cif.read(SAMPLE)


@pytest.fixture
def panel(qapp, structure):
    from facet.ui.planes_panel import PlanesPanel

    widget = PlanesPanel()
    widget.set_structure(structure)
    return widget


# ---------------------------------------------------------------------------
# adding and removing planes
# ---------------------------------------------------------------------------

def test_a_new_panel_has_no_planes(panel):
    assert panel.current_planes() == []
    assert not panel.current_slab().enabled


def test_adding_a_plane_emits_a_rebuild(panel):
    seen = []
    panel.changed.connect(lambda: seen.append(1))
    panel.add_plane(1, 0, 1)
    assert len(panel.current_planes()) == 1
    assert panel.current_planes()[0].hkl == "1 0 1"
    assert seen, "the scene was not asked to rebuild"


def test_the_table_row_reports_the_spacing(panel):
    from facet.core import planes as planes_mod

    panel.add_plane(0, 0, 2)
    plane = panel.current_planes()[0]
    expected = planes_mod.spacing(panel.structure, plane)
    shown = float(panel.table.item(0, 2).text())
    assert shown == pytest.approx(expected, abs=5e-5)


def test_zero_indices_are_refused_with_a_reason(panel):
    panel.add_plane(0, 0, 0)
    assert panel.current_planes() == []
    assert "not a plane" in panel.summary.text()


def test_each_new_plane_gets_a_different_colour(panel):
    for hkl in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)):
        panel.add_plane(*hkl)
    colours = [p.color for p in panel.current_planes()]
    assert len(set(colours)) == len(colours)


def test_removing_a_plane_removes_the_right_one(panel):
    for hkl in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        panel.add_plane(*hkl)
    middle = panel.current_planes()[1]
    panel._remove(middle)
    assert [p.hkl for p in panel.current_planes()] == ["1 0 0", "0 0 1"]
    assert panel.table.rowCount() == 2


def test_remove_all_clears_the_table(panel):
    for hkl in ((1, 0, 0), (0, 1, 0)):
        panel.add_plane(*hkl)
    panel._clear()
    assert panel.current_planes() == []
    assert panel.table.rowCount() == 0


def test_a_preset_adds_that_plane(panel):
    from facet.ui.planes_panel import PRESETS

    label, hkl = PRESETS[6]                      # (1 1 1)
    for index in range(panel.preset.count()):
        if panel.preset.itemText(index) == label:
            panel.preset.setCurrentIndex(index)
            break
    assert [p.hkl for p in panel.current_planes()] == ["1 1 1"]
    # and the chooser returns to its prompt, ready for the next one
    assert panel.preset.currentIndex() == 0


def test_the_row_controls_write_through_to_the_plane(panel):
    panel.add_plane(0, 0, 1)
    plane = panel.current_planes()[0]

    panel.table.cellWidget(0, 3).setValue(0.375)         # offset
    assert plane.offset == pytest.approx(0.375)

    panel.table.cellWidget(0, 4).setValue(5)             # repeat
    assert plane.repeat == 5

    panel.table.cellWidget(0, 5).setValue(80)            # opacity
    assert plane.alpha == pytest.approx(0.80)

    panel.table.cellWidget(0, 0).setChecked(False)       # visible
    assert plane.visible is False


def test_editing_a_row_asks_for_a_rebuild(panel):
    panel.add_plane(0, 0, 1)
    seen = []
    panel.changed.connect(lambda: seen.append(1))
    panel.table.cellWidget(0, 3).setValue(0.2)
    assert seen


def test_refreshing_the_table_does_not_fire_rebuilds(panel, structure):
    """Rebuilding the widgets sets their values, which must not look like edits.

    Without the guard, re-reading the table would emit a rebuild per control per
    plane and the view would thrash on every structure change.
    """
    for hkl in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        panel.add_plane(*hkl)
    seen = []
    panel.changed.connect(lambda: seen.append(1))
    panel.set_structure(structure)
    assert seen == []


# ---------------------------------------------------------------------------
# the slab
# ---------------------------------------------------------------------------

def test_the_slab_controls_write_through(panel):
    panel.slab_h.setValue(1)
    panel.slab_k.setValue(0)
    panel.slab_l.setValue(1)
    panel.slab_centre.setValue(0.25)
    panel.slab_thickness.setValue(3.5)
    panel.slab_on.setChecked(True)

    slab = panel.current_slab()
    assert (slab.h, slab.k, slab.l) == (1, 0, 1)
    assert slab.centre == pytest.approx(0.25)
    assert slab.thickness == pytest.approx(3.5)
    assert slab.enabled


def test_the_slab_report_states_the_numbers_and_the_count(panel, structure):
    from facet.core import planes as planes_mod

    panel.slab_l.setValue(1)
    panel.slab_thickness.setValue(3.0)
    panel.slab_on.setChecked(True)
    text = panel.slab_report.text()

    _, d = planes_mod.normal_and_spacing(structure, 0, 0, 1)
    assert f"{d:.4f}" in text
    inside = len(planes_mod.atoms_in_slab(structure, panel.current_slab()))
    assert str(inside) in text
    assert str(len(structure.atoms)) in text


def test_a_disabled_slab_says_it_is_not_applied(panel):
    panel.slab_on.setChecked(False)
    panel.slab_thickness.setValue(2.0)
    assert "not applied" in panel.slab_report.text()


def test_an_invalid_slab_normal_is_explained(panel):
    panel.slab_h.setValue(0)
    panel.slab_k.setValue(0)
    panel.slab_l.setValue(0)
    assert "no normal" in panel.slab_report.text()


# ---------------------------------------------------------------------------
# what the panel says about a plane
# ---------------------------------------------------------------------------

def test_the_report_lists_the_atoms_on_the_selected_plane(panel, structure):
    from facet.core import planes as planes_mod

    panel.add_plane(0, 0, 1)
    panel.table.setCurrentCell(0, 1)
    panel.tolerance.setValue(0.5)

    expected = planes_mod.atoms_on_plane(panel.current_planes()[0], )  \
        if False else planes_mod.atoms_on_plane(
            structure, panel.current_planes()[0], 0.5)
    assert panel.contents.rowCount() == len(expected)
    labels = {panel.contents.item(r, 0).text()
              for r in range(panel.contents.rowCount())}
    assert labels == {structure.atoms[i].label for i in expected}


def test_the_reported_distances_are_to_the_nearest_plane_of_the_family(panel):
    panel.add_plane(0, 0, 2)
    panel.table.setCurrentCell(0, 1)
    panel.tolerance.setValue(0.6)
    for row in range(panel.contents.rowCount()):
        distance = float(panel.contents.item(row, 2).text())
        assert 0.0 <= distance <= 0.6 + 1e-9


def test_a_wider_tolerance_lists_more_atoms(panel):
    panel.add_plane(1, 0, 1)
    panel.table.setCurrentCell(0, 1)
    counts = []
    for tolerance in (0.05, 0.3, 1.0, 2.5):
        panel.tolerance.setValue(tolerance)
        counts.append(panel.contents.rowCount())
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_the_summary_states_the_spacing_and_the_composition(panel, structure):
    from facet.core import planes as planes_mod

    panel.add_plane(0, 0, 2)
    panel.table.setCurrentCell(0, 1)
    panel.tolerance.setValue(0.4)
    text = panel.summary.text()
    _, d = planes_mod.normal_and_spacing(structure, 0, 0, 2)
    assert f"{d:.4f}" in text
    composition = planes_mod.plane_occupancy(structure,
                                            panel.current_planes()[0], 0.4)
    for element in composition:
        assert element in text


def test_the_panel_never_says_what_a_plane_means(panel):
    """It reports the spacing and the contents. It does not call it a layer."""
    import re

    panel.add_plane(0, 0, 2)
    panel.table.setCurrentCell(0, 1)
    panel.slab_on.setChecked(True)
    text = " ".join([panel.summary.text(), panel.slab_report.text()]).lower()
    forbidden = {"layer", "layers", "layered", "good", "bad", "poor",
                 "correct", "incorrect", "wrong", "clearly", "obviously",
                 "should", "suggests", "indicates", "confirms"}
    found = set(re.findall(r"[a-z]+", text)) & forbidden
    assert not found, f"the panel passes judgement: {found}"


def test_the_panel_survives_having_no_structure(qapp):
    from facet.ui.planes_panel import PlanesPanel

    widget = PlanesPanel()
    widget.set_structure(None)
    widget.add_plane(0, 0, 1)
    assert len(widget.current_planes()) == 1
    assert widget.table.item(0, 2).text() == ""       # no spacing to report
    widget.table.setCurrentCell(0, 1)
    assert widget.contents.rowCount() == 0


# ---------------------------------------------------------------------------
# drawing, on both tiers
# ---------------------------------------------------------------------------

def test_the_software_tier_draws_the_planes(qapp, structure):
    """The QPainter fallback must draw planes too, not only OpenGL."""
    from PySide6.QtGui import QImage, QPainter

    from facet.core import planes as planes_mod
    from facet.gl.camera import Camera
    from facet.gl.painter import PainterRenderer
    from facet.gl.scene import build_scene

    def render(scene):
        image = QImage(320, 240, QImage.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        camera = Camera()
        camera.frame(scene.center, scene.radius)
        try:
            PainterRenderer().render(painter, scene, camera, 320, 240)
        finally:
            painter.end()
        return image

    def colours(image, step=5):
        seen = set()
        for y in range(0, image.height(), step):
            for x in range(0, image.width(), step):
                c = image.pixelColor(x, y)
                seen.add((c.red(), c.green(), c.blue()))
        return seen

    plain = render(build_scene(structure))
    with_plane = render(build_scene(structure, lattice_planes=[
        planes_mod.LatticePlane(0, 0, 1, offset=0.5, color=(1.0, 0.0, 0.0),
                                alpha=0.55)]))
    assert colours(with_plane) != colours(plain), "the plane was not drawn"

    # and a strongly red plane must put red on the image
    reds = [c for c in colours(with_plane)
            if c[0] > 110 and c[0] > c[1] + 40 and c[0] > c[2] + 40]
    assert reds, "the plane's colour never reached the image"


def test_the_software_tier_survives_a_slab_that_removes_everything(qapp,
                                                                   structure):
    from PySide6.QtGui import QImage, QPainter

    from facet.core import planes as planes_mod
    from facet.gl.camera import Camera
    from facet.gl.painter import PainterRenderer
    from facet.gl.scene import build_scene

    scene = build_scene(structure, slab=planes_mod.Slab(
        0, 0, 1, centre=0.5, thickness=0.001, enabled=True))
    image = QImage(160, 120, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        PainterRenderer().render(painter, scene, Camera(), 160, 120)
    finally:
        painter.end()
    assert not image.isNull()


class TestGLPlanes:
    """Planes through the real OpenGL widget, skipped where there is no context."""

    @pytest.fixture(scope="class")
    def view(self, qapp, structure):
        from PySide6.QtGui import QSurfaceFormat

        from facet.gl import caps as caps_mod
        from facet.gl.view import StructureView

        QSurfaceFormat.setDefaultFormat(caps_mod.request_format())
        widget = StructureView()
        widget.resize(360, 280)
        widget.show()
        for _ in range(20):
            qapp.processEvents()
        if widget.caps is None:
            pytest.skip("no OpenGL context available in this environment")
        yield widget
        widget.close()

    def test_a_plane_changes_the_image(self, view, qapp, structure):
        from facet.core import planes as planes_mod
        from facet.gl.scene import build_scene

        def grab(scene):
            view.set_scene(scene, reframe=True)
            for _ in range(6):
                qapp.processEvents()
            return view.grab_image(360, 280)

        plain = grab(build_scene(structure))
        with_plane = grab(build_scene(structure, lattice_planes=[
            planes_mod.LatticePlane(0, 0, 1, offset=0.5,
                                    color=(1.0, 0.2, 0.2), alpha=0.6)]))
        assert not plain.isNull() and not with_plane.isNull()

        different = 0
        for y in range(0, 280, 7):
            for x in range(0, 360, 7):
                if plain.pixelColor(x, y) != with_plane.pixelColor(x, y):
                    different += 1
        assert different > 20, "the plane did not reach the image"

    def test_removing_the_planes_restores_the_image(self, view, qapp,
                                                   structure):
        """A scene with fewer planes must not keep drawing the old ones."""
        from facet.core import planes as planes_mod
        from facet.gl.scene import build_scene

        def grab(scene):
            view.set_scene(scene, reframe=True)
            for _ in range(6):
                qapp.processEvents()
            return view.grab_image(360, 280)

        first = grab(build_scene(structure))
        grab(build_scene(structure, lattice_planes=[
            planes_mod.LatticePlane(0, 0, 1, offset=0.5,
                                    color=(1.0, 0.2, 0.2), alpha=0.7),
            planes_mod.LatticePlane(1, 0, 0, offset=0.5,
                                    color=(0.2, 1.0, 0.2), alpha=0.7)]))
        back = grab(build_scene(structure))

        different = 0
        for y in range(0, 280, 7):
            for x in range(0, 360, 7):
                if first.pixelColor(x, y) != back.pixelColor(x, y):
                    different += 1
        assert different < 12, "a removed plane is still being drawn"

    def test_a_slab_reduces_what_is_drawn(self, view, qapp, structure):
        from facet.core import planes as planes_mod
        from facet.gl.scene import build_scene

        whole = build_scene(structure, cell_range=(2, 2, 1))
        view.set_scene(whole, reframe=True)
        for _ in range(6):
            qapp.processEvents()

        thin = build_scene(structure, cell_range=(2, 2, 1),
                          slab=planes_mod.Slab(0, 0, 1, centre=0.25,
                                               thickness=2.0, enabled=True))
        assert thin.n_atoms < whole.n_atoms
        view.set_scene(thin, reframe=True)
        for _ in range(6):
            qapp.processEvents()
        assert not view.grab_image(360, 280).isNull()
