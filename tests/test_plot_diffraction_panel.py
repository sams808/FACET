"""The QPainter plot and the diffraction panel.

The plot is tested by rendering it and checking the geometry it produced, not by
comparing images: a pixel comparison would break on every font, and it would not
say whether the axis mapping is right. The mapping is what matters, because it
is what decides whether a peak is drawn where its 2-theta says.

The vector exports are checked by reading the file back -- an SVG must contain
path data, a PDF must be a PDF -- because "it did not raise" is not evidence
that anything was drawn.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from facet.core import cif
from facet.core import diffraction as dif


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def rocksalt():
    text = "\n".join([
        "data_NaCl", "_cell_length_a 5.6402", "_cell_length_b 5.6402",
        "_cell_length_c 5.6402", "_cell_angle_alpha 90",
        "_cell_angle_beta 90", "_cell_angle_gamma 90",
        "_symmetry_space_group_name_H-M 'F m -3 m'", "loop_",
        "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
        "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_occupancy",
        "Na1 Na 0.0 0.0 0.0 1.0", "Cl1 Cl 0.5 0.5 0.5 1.0"]) + "\n"
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "NaCl.cif")
    Path(path).write_text(text, encoding="utf-8")
    return cif.read(path)


def _render(widget, width=800, height=480):
    """Paint the widget into an image and hand back the image."""
    from PySide6.QtGui import QImage, QPainter

    image = QImage(width, height, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        widget.render_to(painter, width, height)
    finally:
        painter.end()
    return image


# ---------------------------------------------------------------------------
# the plot: axis mapping
# ---------------------------------------------------------------------------

def test_data_and_device_coordinates_are_inverses(qapp):
    from facet.ui.plot import Plot, Series

    plot = Plot()
    plot.set_series([Series(np.linspace(10, 90, 100), np.linspace(0, 100, 100))])
    plot.resize(800, 480)
    rect, _ = plot._metrics(800, 480, plot.font())

    for x, y in ((12.0, 3.0), (50.0, 55.0), (88.0, 97.0)):
        px, py = plot._to_device(rect, x, y)
        back_x, back_y = plot._to_data(rect, float(px), float(py))
        assert back_x == pytest.approx(x, rel=1e-9, abs=1e-9)
        assert back_y == pytest.approx(y, rel=1e-9, abs=1e-9)


def test_the_axis_mapping_is_linear_and_the_right_way_up(qapp):
    """Larger x goes right; larger y goes *up*, meaning a smaller pixel row."""
    from facet.ui.plot import Plot, Series

    plot = Plot()
    plot.set_series([Series(np.array([0.0, 10.0]), np.array([0.0, 10.0]))])
    plot.resize(600, 400)
    rect, _ = plot._metrics(600, 400, plot.font())

    x_low = float(plot._to_device(rect, 1.0, 0.0)[0])
    x_high = float(plot._to_device(rect, 9.0, 0.0)[0])
    assert x_low < x_high

    y_low = float(plot._to_device(rect, 0.0, 1.0)[1])
    y_high = float(plot._to_device(rect, 0.0, 9.0)[1])
    assert y_high < y_low, "the intensity axis is upside down"

    # linearity: equal steps in data must give equal steps in pixels
    steps = [float(plot._to_device(rect, v, 0.0)[0]) for v in (1, 2, 3, 4, 5)]
    gaps = np.diff(steps)
    assert np.allclose(gaps, gaps[0], rtol=1e-9)


def test_fit_frames_every_series(qapp):
    from facet.ui.plot import Plot, Series

    plot = Plot()
    plot.set_series([
        Series(np.array([10.0, 20.0]), np.array([0.0, 50.0])),
        Series(np.array([15.0, 80.0]), np.array([-20.0, 90.0])),
    ])
    low_x, high_x = plot.x_range
    low_y, high_y = plot.y_range
    assert low_x <= 10.0 and high_x >= 80.0
    assert low_y <= -20.0 and high_y >= 90.0


def test_fit_includes_an_offset_series(qapp):
    """A difference curve sits below zero; framing must not cut it off."""
    from facet.ui.plot import Plot, Series

    plot = Plot()
    plot.set_series([
        Series(np.array([10.0, 20.0]), np.array([0.0, 100.0])),
        Series(np.array([10.0, 20.0]), np.array([-5.0, 5.0]), offset=-40.0),
    ])
    assert plot.y_range[0] <= -45.0


def test_fit_survives_degenerate_data(qapp):
    """A single point, or a flat line, must not produce a zero-width axis."""
    from facet.ui.plot import Plot, Series

    plot = Plot()
    for x, y in ((np.array([5.0]), np.array([1.0])),
                 (np.array([1.0, 2.0]), np.array([3.0, 3.0]))):
        plot.set_series([Series(x, y)])
        low_x, high_x = plot.x_range
        low_y, high_y = plot.y_range
        assert high_x > low_x
        assert high_y > low_y


def test_an_empty_plot_renders(qapp):
    from facet.ui.plot import Plot

    plot = Plot()
    plot.set_series([])
    image = _render(plot)
    assert not image.isNull()


def test_mismatched_series_lengths_are_rejected():
    from facet.ui.plot import Series

    with pytest.raises(ValueError, match="x values"):
        Series(np.zeros(5), np.zeros(4), label="wrong")


def test_wheel_zoom_keeps_the_point_under_the_cursor(qapp):
    """Zooming about the cursor must leave the cursor over the same value."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    from facet.ui.plot import Plot, Series

    plot = Plot()
    plot.set_series([Series(np.linspace(0, 100, 50), np.linspace(0, 100, 50))])
    plot.resize(700, 420)
    rect, _ = plot._metrics(700, 420, plot.font())

    position = QPointF(rect.center())
    before = plot._to_data(rect, position.x(), position.y())
    event = QWheelEvent(position, plot.mapToGlobal(position.toPoint()),
                        QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
                        Qt.NoModifier, Qt.ScrollUpdate, False)
    plot.wheelEvent(event)
    after = plot._to_data(rect, position.x(), position.y())
    assert after[0] == pytest.approx(before[0], abs=1e-6)
    assert after[1] == pytest.approx(before[1], abs=1e-6)
    # and it really did zoom in
    assert (plot.x_range[1] - plot.x_range[0]) < 100.0


def test_ctrl_wheel_zooms_only_the_intensity_axis(qapp):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    from facet.ui.plot import Plot, Series

    plot = Plot()
    plot.set_series([Series(np.linspace(0, 100, 50), np.linspace(0, 100, 50))])
    plot.resize(700, 420)
    rect, _ = plot._metrics(700, 420, plot.font())
    x_before = plot.x_range
    y_span_before = plot.y_range[1] - plot.y_range[0]

    position = QPointF(rect.center())
    event = QWheelEvent(position, plot.mapToGlobal(position.toPoint()),
                        QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
                        Qt.ControlModifier, Qt.ScrollUpdate, False)
    plot.wheelEvent(event)
    assert plot.x_range == x_before
    assert (plot.y_range[1] - plot.y_range[0]) < y_span_before


def test_nice_step_gives_readable_intervals(qapp):
    from facet.ui.plot import Plot

    plot = Plot()
    for span, target in ((100.0, 5), (1.0, 4), (0.05, 5), (7000.0, 6)):
        step = plot._nice_step(span, target)
        assert step > 0
        mantissa = step / 10.0 ** math.floor(math.log10(step))
        assert mantissa == pytest.approx(
            min((1.0, 2.0, 2.5, 5.0, 10.0),
                key=lambda m: abs(m - mantissa)), rel=1e-9)
        # and it must not produce an absurd number of gridlines
        assert 1 <= span / step <= 4 * target


# ---------------------------------------------------------------------------
# the plot: what actually gets drawn
# ---------------------------------------------------------------------------

def test_a_peak_is_drawn_at_the_column_its_two_theta_maps_to(qapp):
    """The one property that matters: position on screen follows the data.

    A single narrow peak is rendered and the brightest column of the image is
    compared with where the axis mapping says that 2-theta lands. This is what
    would catch an off-by-a-margin error that the mapping unit tests, which use
    the same margin code, could not.
    """
    from facet.ui.plot import Plot, Series

    x = np.linspace(20.0, 60.0, 4001)
    y = np.exp(-((x - 43.0) / 0.05) ** 2) * 100.0
    plot = Plot()
    plot.set_labels(x="2theta", y="I")
    plot.set_series([Series(x, y, color=(255, 255, 255), width=1.0)])
    plot.resize(800, 480)

    image = _render(plot, 800, 480)
    rect, _ = plot._metrics(800, 480, plot.font())
    expected = float(plot._to_device(rect, 43.0, 0.0)[0])

    # the topmost drawn pixel inside the frame marks the peak
    best_row, best_col = None, None
    for row in range(int(rect.top()) + 2, int(rect.bottom()) - 1):
        for col in range(int(rect.left()) + 2, int(rect.right()) - 1):
            pixel = image.pixelColor(col, row)
            if pixel.red() > 200 and pixel.green() > 200 and pixel.blue() > 200:
                best_row, best_col = row, col
                break
        if best_row is not None:
            break
    assert best_col is not None, "the curve was not drawn"
    assert abs(best_col - expected) <= 3


def test_ticks_are_drawn_below_the_frame(qapp):
    from facet.ui.plot import Plot, Series, Ticks

    plot = Plot()
    plot.set_series([Series(np.array([10.0, 90.0]), np.array([0.0, 100.0]))])
    plot.set_ticks([Ticks(np.array([20.0, 40.0, 60.0]), color=(255, 0, 0))])
    plot.resize(800, 480)
    image = _render(plot, 800, 480)
    rect, _ = plot._metrics(800, 480, plot.font())

    found = 0
    row = int(rect.bottom()) + 4
    for column in range(int(rect.left()), int(rect.right())):
        pixel = image.pixelColor(column, row)
        if pixel.red() > 150 and pixel.green() < 100:
            found += 1
    assert found >= 3, "reflection marks were not drawn under the axis"


def test_a_hidden_series_is_not_drawn(qapp):
    from facet.ui.plot import Plot, Series

    plot = Plot()
    visible = Series(np.linspace(10, 90, 200), np.full(200, 50.0),
                     color=(255, 255, 255))
    plot.set_series([visible])
    lit = _count_light(_render(plot))
    visible.visible = False
    plot.set_series([visible], keep_view=True)
    assert _count_light(_render(plot)) < lit / 4


def _count_light(image) -> int:
    total = 0
    for row in range(0, image.height(), 3):
        for column in range(0, image.width(), 3):
            pixel = image.pixelColor(column, row)
            if pixel.red() > 200 and pixel.green() > 200:
                total += 1
    return total


# ---------------------------------------------------------------------------
# the plot: vector export
# ---------------------------------------------------------------------------

def test_svg_export_contains_vector_paths(qapp):
    from facet.ui.plot import Plot, Series

    plot = Plot()
    plot.set_series([Series(np.linspace(10, 90, 400),
                            np.abs(np.sin(np.linspace(0, 9, 400))) * 100,
                            label="calculated")])
    plot.set_labels(x="2theta / deg", y="intensity", title="test pattern")
    handle = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    handle.close()
    plot.save_svg(handle.name, 900, 520)

    text = Path(handle.name).read_text(encoding="utf-8", errors="replace")
    assert "<svg" in text
    assert "<path" in text or "polyline" in text
    assert "test pattern" in text
    # a vector file, not an embedded bitmap
    assert "image/png" not in text
    assert len(text) > 2000


def test_pdf_export_writes_a_pdf(qapp):
    from facet.ui.plot import Plot, Series

    plot = Plot()
    plot.set_series([Series(np.linspace(10, 90, 300),
                            np.linspace(0, 100, 300), label="calculated")])
    handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    handle.close()
    plot.save_pdf(handle.name)

    data = Path(handle.name).read_bytes()
    assert data[:5] == b"%PDF-"
    assert b"%%EOF" in data[-1024:]
    assert len(data) > 1500


def test_export_uses_a_light_ground_for_print(qapp):
    """A figure for a paper is printed on white, whatever the screen theme."""
    from facet.ui.plot import Plot, Series

    plot = Plot()
    plot.set_series([Series(np.linspace(10, 90, 50), np.linspace(0, 100, 50))])
    plot.resize(600, 360)

    from PySide6.QtGui import QImage, QPainter

    images = {}
    for export in (False, True):
        image = QImage(600, 360, QImage.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        plot.render_to(painter, 600, 360, for_export=export)
        painter.end()
        images[export] = image.pixelColor(3, 3)

    assert images[True].lightness() > images[False].lightness() + 100


# ---------------------------------------------------------------------------
# the diffraction panel
# ---------------------------------------------------------------------------

def test_panel_shows_the_pattern_for_a_structure(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    assert panel.pattern is not None
    assert panel.pattern.reflections
    assert panel.table.rowCount() == len(panel.pattern.reflections)
    # the table rows must agree with the pattern they came from
    for row, line in enumerate(panel.pattern.reflections):
        assert panel.table.item(row, 0).text() == line.hkl
        assert float(panel.table.item(row, 2).text()) == pytest.approx(
            line.two_theta, abs=5e-4)
        assert float(panel.table.item(row, 3).text()) == pytest.approx(
            line.intensity, abs=5e-3)


def test_panel_recomputes_only_for_a_different_structure(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    first = panel.pattern
    panel.set_structure(rocksalt)
    assert panel.pattern is first, "the same structure was recomputed"


def test_panel_clears_when_the_structure_goes_away(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    panel.set_structure(None)
    assert panel.pattern is None
    assert panel.table.rowCount() == 0


def test_changing_the_radiation_changes_the_pattern(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    xray = {r.hkl: r.intensity for r in panel.pattern.reflections}

    panel.radiation.setCurrentText("neutron")
    neutron = {r.hkl: r.intensity for r in panel.pattern.reflections}
    assert panel.pattern.radiation == "neutron"
    assert xray != neutron


def test_choosing_a_named_wavelength_fills_in_the_exact_value(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    for i in range(panel.wavelength_choice.count()):
        if panel.wavelength_choice.itemText(i).startswith("Mo Ka1"):
            panel.wavelength_choice.setCurrentIndex(i)
            break
    assert panel.wavelength.value() == pytest.approx(
        dif.WAVELENGTHS["Mo Ka1"], abs=1e-6)
    assert panel.pattern.wavelength == pytest.approx(
        dif.WAVELENGTHS["Mo Ka1"], abs=1e-6)


def test_the_plot_carries_the_calculated_curve(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    labels = [s.label for s in panel.plot.series]
    assert any("calculated" in label for label in labels)
    assert not any("measured" in label for label in labels)


def test_stick_mode_puts_a_line_at_every_reflection(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    panel.show_sticks.setChecked(True)
    series = panel.plot.series[-1]
    # three points per line: up from zero, the peak, back to zero
    assert len(series.x) == 3 * len(panel.pattern.reflections)
    shift = panel.zero_shift.value()
    for i, line in enumerate(panel.pattern.reflections):
        assert series.x[3 * i] == pytest.approx(line.two_theta + shift)
        assert series.y[3 * i + 1] == pytest.approx(line.intensity)


def test_a_measured_pattern_is_overlaid_and_scaled(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    x, y = panel.pattern.profile(5.0, 90.0, step=0.02)
    panel.measured = (x, y * 640.0)
    panel.measured_name = "synthetic"
    panel.redraw()

    labels = [s.label for s in panel.plot.series]
    assert any("measured" in label for label in labels)
    assert any("difference" in label for label in labels)
    assert panel._scale_factor == pytest.approx(640.0, rel=1e-3)
    assert "scale factor" in panel.plot.footnote


def test_the_difference_curve_sits_below_the_data(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    x, y = panel.pattern.profile(5.0, 90.0, step=0.02)
    panel.measured = (x, y * 3.0)
    panel.measured_name = "synthetic"
    panel.redraw()
    difference = next(s for s in panel.plot.series if s.label == "difference")
    assert difference.offset < 0


def test_clearing_the_measured_pattern_removes_it(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    x, y = panel.pattern.profile(5.0, 90.0, step=0.02)
    panel.measured = (x, y)
    panel.measured_name = "synthetic"
    panel.clear_measured.setEnabled(True)
    panel.redraw()
    panel._clear_measured()
    assert panel.measured is None
    assert not any("measured" in s.label for s in panel.plot.series)
    assert not panel.clear_measured.isEnabled()


def test_hovering_reports_the_nearest_line(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    strongest = panel.pattern.strongest(1)[0]
    panel._on_hover(strongest.two_theta + 0.05, 50.0)
    text = panel.readout.text()
    assert strongest.hkl in text
    assert f"{strongest.d:.4f}" in text


def test_clicking_a_line_selects_its_row_and_reports_it(qapp, rocksalt):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    seen = []
    panel.reflection_selected.connect(lambda h, k, l: seen.append((h, k, l)))
    target = panel.pattern.reflections[2]
    panel._on_pick(target.two_theta, 10.0)
    assert (target.h, target.k, target.l) in seen
    rows = panel.table.selectionModel().selectedRows()
    assert rows and panel.table.item(rows[0].row(), 0).text() == target.hkl


def test_the_panel_reports_notes_but_never_a_verdict(qapp, rocksalt):
    import re

    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    text = panel.notes.text().lower()
    assert "lambda" in text or "λ" in text
    forbidden = {"good", "bad", "poor", "excellent", "correct", "incorrect",
                 "wrong", "reliable", "should"}
    assert not (set(re.findall(r"[a-z]+", text)) & forbidden)


def test_an_impossible_setting_is_reported_not_raised(qapp, rocksalt):
    """2-theta from above 2-theta to must not crash the panel."""
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    panel.two_theta_min.setValue(120.0)
    panel.two_theta_max.setValue(30.0)
    # either an empty pattern or a message, but the panel must still be alive
    assert panel.isEnabled()
    panel.redraw()


def test_the_panel_renders_in_both_themes(qapp, rocksalt):
    from facet.core import theme as theme_mod
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)
    for name in list(theme_mod.PALETTES)[:2]:
        theme = theme_mod.Theme(palette_name=name)
        panel.apply_theme(theme)
        image = _render(panel.plot)
        assert not image.isNull()


def test_export_csv_round_trips_the_reflection_list(qapp, rocksalt, tmp_path):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)

    target = tmp_path / "pattern.csv"
    import facet.ui.diffraction_panel as module
    original = module.QFileDialog.getSaveFileName
    module.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (str(target), ""))
    try:
        panel._export_csv()
    finally:
        module.QFileDialog.getSaveFileName = original

    lines = target.read_text(encoding="utf-8").splitlines()
    body = [line for line in lines if not line.startswith("#")]
    assert body[0].startswith("h,k,l,d,two_theta")
    assert len(body) - 1 == len(panel.pattern.reflections)
    first = body[1].split(",")
    line = panel.pattern.reflections[0]
    assert int(first[0]) == line.h
    assert float(first[4]) == pytest.approx(line.two_theta, abs=1e-4)


def test_export_xy_round_trips_the_profile(qapp, rocksalt, tmp_path):
    from facet.ui.diffraction_panel import DiffractionPanel

    panel = DiffractionPanel()
    panel.set_structure(rocksalt)

    target = tmp_path / "profile.xy"
    import facet.ui.diffraction_panel as module
    original = module.QFileDialog.getSaveFileName
    module.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (str(target), ""))
    try:
        panel._export_csv()
    finally:
        module.QFileDialog.getSaveFileName = original

    x, y = dif.read_pattern(target)
    assert len(x) > 1000
    assert y.max() == pytest.approx(100.0, abs=0.01)
    assert x.min() == pytest.approx(panel.two_theta_min.value(), abs=0.03)
