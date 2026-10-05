"""Contouring a field, and the 2D section view.

The contouring is checked against a field whose contours are known exactly: for
``f = sqrt(u^2 + v^2)`` the level set at r is a circle of radius r and
circumference 2 pi r, so every segment's position and the total length can both be
compared with the closed form. The saddle case is checked separately, because
that is where a marching-squares implementation without a disambiguation rule
draws crossing lines.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import volume as V

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def radial():
    """f = sqrt(u^2 + v^2) on a square grid: contours are circles."""
    n = 401
    extent = (-5.0, 5.0, -5.0, 5.0)
    u = np.linspace(extent[0], extent[1], n)
    v = np.linspace(extent[2], extent[3], n)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.sqrt(uu ** 2 + vv ** 2), extent


# ===========================================================================
# contouring
# ===========================================================================

@pytest.mark.parametrize("level", [1.0, 2.0, 3.0, 4.0])
def test_a_contour_of_the_radial_field_is_a_circle(radial, level):
    values, extent = radial
    segments = V.contour_lines(values, extent, [level])[level]
    assert len(segments) > 100

    points = segments.reshape(-1, 2)
    radii = np.linalg.norm(points, axis=1)
    assert radii.mean() == pytest.approx(level, abs=5e-4)
    assert radii.std() < 1e-3

    length = float(np.sum(np.linalg.norm(segments[:, 1] - segments[:, 0],
                                        axis=1)))
    assert length == pytest.approx(2.0 * math.pi * level, rel=1e-4)


def test_every_contour_point_lies_on_its_level(radial):
    """Interpolation along a cell edge must land on the level, not near it."""
    values, extent = radial
    for level in (0.8, 1.7, 3.3):
        segments = V.contour_lines(values, extent, [level])[level]
        points = segments.reshape(-1, 2)
        # the field at those points, evaluated analytically
        assert np.allclose(np.linalg.norm(points, axis=1), level, atol=2e-3)


def test_the_saddle_is_resolved_without_crossing_lines():
    """f = uv at level 0 is the two axes. A naive implementation crosses them.

    The four corners of a cell straddling the origin alternate in sign, and
    without the centre-value rule the two branches are joined the wrong way
    round, which puts a contour where the field is not at the level.
    """
    n = 201
    extent = (-5.0, 5.0, -5.0, 5.0)
    axis = np.linspace(-5.0, 5.0, n)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    segments = V.contour_lines(uu * vv, extent, [0.0])[0.0]
    assert len(segments) > 100
    points = segments.reshape(-1, 2)
    # every point must satisfy uv = 0, so one coordinate is zero
    assert float(np.abs(points[:, 0] * points[:, 1]).max()) < 1e-9


def test_contours_of_several_levels_come_back_keyed_by_level(radial):
    values, extent = radial
    levels = [1.0, 2.5, 4.0]
    lines = V.contour_lines(values, extent, levels)
    assert set(lines) == {1.0, 2.5, 4.0}
    # a larger circle has more segments
    assert len(lines[4.0]) > len(lines[1.0])


def test_a_level_outside_the_data_gives_nothing(radial):
    values, extent = radial
    for level in (-1.0, 100.0):
        assert len(V.contour_lines(values, extent, [level])[level]) == 0


def test_a_flat_field_has_no_contours():
    extent = (0.0, 1.0, 0.0, 1.0)
    lines = V.contour_lines(np.full((20, 20), 3.0), extent, [3.0, 1.0])
    for segments in lines.values():
        assert len(segments) == 0


def test_a_degenerate_grid_does_not_raise():
    extent = (0.0, 1.0, 0.0, 1.0)
    for shape in ((1, 1), (1, 10), (10, 1)):
        lines = V.contour_lines(np.ones(shape), extent, [0.5])
        assert lines[0.5].shape == (0, 2, 2)


def test_the_contour_follows_the_extent_not_the_grid_indices():
    """A shifted extent must shift the contour, not the picture."""
    n = 101
    axis = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    field = uu                          # a ramp in u

    a = V.contour_lines(field, (0.0, 1.0, 0.0, 1.0), [0.5])[0.5]
    b = V.contour_lines(field, (10.0, 11.0, 0.0, 1.0), [0.5])[0.5]
    assert a.reshape(-1, 2)[:, 0].mean() == pytest.approx(0.5, abs=1e-6)
    assert b.reshape(-1, 2)[:, 0].mean() == pytest.approx(10.5, abs=1e-6)


def test_a_ramp_contours_at_the_right_place_in_both_directions():
    n = 101
    axis = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    extent = (0.0, 1.0, 0.0, 1.0)

    in_u = V.contour_lines(uu, extent, [0.25])[0.25].reshape(-1, 2)
    assert np.allclose(in_u[:, 0], 0.25, atol=1e-6)

    in_v = V.contour_lines(vv, extent, [0.75])[0.75].reshape(-1, 2)
    assert np.allclose(in_v[:, 1], 0.75, atol=1e-6)


# ===========================================================================
# levels
# ===========================================================================

def test_linear_levels_span_the_data_without_touching_the_ends():
    values = np.linspace(2.0, 10.0, 100)
    levels = V.nice_levels(values, 5)
    assert len(levels) == 5
    assert levels.min() > values.min()
    assert levels.max() < values.max()
    gaps = np.diff(levels)
    assert np.allclose(gaps, gaps[0])


def test_logarithmic_levels_are_geometrically_spaced():
    """A valence sum or a charge density spans decades; linear levels all land
    in the divergent region near the nuclei."""
    values = np.exp(np.linspace(0.0, 10.0, 200))
    levels = V.nice_levels(values, 6, log=True)
    assert len(levels) == 6
    ratios = levels[1:] / levels[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-9)


def test_levels_of_an_empty_or_flat_field():
    assert V.nice_levels(np.array([np.nan, np.nan])).size == 0
    assert list(V.nice_levels(np.full(10, 3.0))) == [3.0]


# ===========================================================================
# the section view
# ===========================================================================

def test_the_colormaps_are_monotonic_in_lightness(qapp):
    """Perceptually uniform: an apparent feature must be a feature in the data.

    Checked as a monotonic lightness ramp, which is the property that stops the
    colours inventing a band where the data has none. The diverging map is
    exempt: it is light in the middle by design, because zero has to be visible.
    """
    from facet.ui.section_view import COLORMAPS, sample_colormap

    for name in COLORMAPS:
        rgb = sample_colormap(name, np.linspace(0.0, 1.0, 64)).astype(float)
        luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
        if name == "difference":
            # light in the middle, dark at both ends
            assert luma[32] > luma[0] and luma[32] > luma[-1]
            continue
        assert np.all(np.diff(luma) > -2.0), f"{name} is not a lightness ramp"
        assert luma[-1] > luma[0] + 100


def test_a_colormap_clamps_outside_zero_to_one(qapp):
    from facet.ui.section_view import sample_colormap

    low = sample_colormap("viridis", [-5.0, 0.0])
    high = sample_colormap("viridis", [1.0, 5.0])
    assert np.array_equal(low[0], low[1])
    assert np.array_equal(high[0], high[1])


def test_the_section_view_interpolates_the_value_under_the_pointer(qapp):
    from facet.ui.section_view import SectionView

    view = SectionView()
    axis = np.linspace(0.0, 1.0, 11)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    view.set_section(uu + 2.0 * vv, (0.0, 1.0, 0.0, 1.0))

    for u, v in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.25, 0.75),
                 (0.13, 0.42)):
        assert view.value_at(u, v) == pytest.approx(u + 2.0 * v, abs=1e-6)


def test_a_point_off_the_plane_reads_as_nan(qapp):
    from facet.ui.section_view import SectionView

    view = SectionView()
    view.set_section(np.zeros((10, 10)), (0.0, 1.0, 0.0, 1.0))
    assert math.isnan(view.value_at(-0.1, 0.5))
    assert math.isnan(view.value_at(0.5, 1.2))
    assert not math.isnan(view.value_at(0.5, 0.5))


def test_an_empty_section_reads_as_nan_and_renders(qapp):
    from PySide6.QtGui import QImage, QPainter

    from facet.ui.section_view import SectionView

    view = SectionView()
    assert math.isnan(view.value_at(0.0, 0.0))
    image = QImage(200, 160, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    view.render_to(painter, 200, 160)
    painter.end()
    assert not image.isNull()


def test_the_data_range_follows_the_values_and_the_overrides(qapp):
    from facet.ui.section_view import SectionView

    view = SectionView()
    view.set_section(np.linspace(2.0, 8.0, 100).reshape(10, 10),
                     (0.0, 1.0, 0.0, 1.0))
    assert view.data_range == pytest.approx((2.0, 8.0))
    view.vmin, view.vmax = 3.0, 5.0
    assert view.data_range == pytest.approx((3.0, 5.0))


def test_the_view_maps_data_to_device_and_back(qapp):
    from PySide6.QtCore import QRectF

    from facet.ui.section_view import SectionView

    view = SectionView()
    view.set_section(np.zeros((20, 20)), (-3.0, 3.0, -2.0, 2.0))
    rect = QRectF(40, 20, 300, 200)
    for u, v in ((-3.0, -2.0), (0.0, 0.0), (2.5, 1.5)):
        x, y = view._to_device(rect, u, v)
        back = view._to_data(rect, float(x), float(y))
        assert back[0] == pytest.approx(u, abs=1e-9)
        assert back[1] == pytest.approx(v, abs=1e-9)


def test_v_increases_upwards_on_screen(qapp):
    from PySide6.QtCore import QRectF

    from facet.ui.section_view import SectionView

    view = SectionView()
    view.set_section(np.zeros((10, 10)), (0.0, 1.0, 0.0, 1.0))
    rect = QRectF(0, 0, 100, 100)
    _, low = view._to_device(rect, 0.5, 0.1)
    _, high = view._to_device(rect, 0.5, 0.9)
    assert float(high) < float(low), "the section is drawn upside down"


def test_the_section_renders_a_real_field(qapp):
    from PySide6.QtGui import QImage, QPainter

    from facet.ui.section_view import SectionView

    axis = np.linspace(-3.0, 3.0, 120)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    field = np.exp(-(uu ** 2 + vv ** 2) / 2.0)

    view = SectionView()
    view.set_section(field, (-3.0, 3.0, -3.0, 3.0),
                     levels=V.nice_levels(field, 6), units="v.u.",
                     title="a gaussian")
    image = QImage(520, 420, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    view.render_to(painter, 520, 420)
    painter.end()

    colours = set()
    for y in range(0, 420, 5):
        for x in range(0, 520, 5):
            pixel = image.pixelColor(x, y)
            colours.add((pixel.red(), pixel.green(), pixel.blue()))
    assert len(colours) > 40, "the colour map was not drawn"


def test_the_section_exports_as_vector(qapp, tmp_path):
    from facet.ui.section_view import SectionView

    axis = np.linspace(-2.0, 2.0, 80)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    view = SectionView()
    view.set_section(uu ** 2 + vv ** 2, (-2.0, 2.0, -2.0, 2.0),
                     levels=[0.5, 1.0, 2.0], title="bowl")

    svg = tmp_path / "section.svg"
    view.save_svg(svg)
    text = svg.read_text(encoding="utf-8", errors="replace")
    assert "<svg" in text
    # the contours are lines in the vector output; the map itself is an image
    assert "<path" in text or "<line" in text or "<polyline" in text

    pdf = tmp_path / "section.pdf"
    view.save_pdf(pdf)
    assert pdf.read_bytes()[:5] == b"%PDF-"


# ===========================================================================
# the volume panel
# ===========================================================================

@pytest.fixture(scope="module")
def structure():
    if not SAMPLE.is_file():
        pytest.skip("the sample structure is not present")
    from facet.core import cif

    return cif.read(SAMPLE)


def test_the_panel_builds_a_valence_map_and_cuts_it(qapp, structure):
    from facet.core import bv
    from facet.ui.volume_panel import VolumePanel

    panel = VolumePanel()
    panel.set_context(structure, bv.DEFAULT)
    panel.probe_element.setCurrentText("Bi")
    panel.probe_ox.setValue(3)
    panel.resolution.setValue(0.5)
    panel.compute()

    assert panel.grid is not None
    assert panel.grid.units == "v.u."
    statistics = panel.grid.statistics()
    assert statistics["min"] > 0
    assert statistics["max"] > statistics["min"]

    assert panel.section.values is not None
    assert panel.section.values.shape == (panel.samples.value(),) * 2
    assert panel.section.levels.size == panel.n_levels.value()
    assert "v.u." in panel.stats.text()


def test_a_wide_ranging_field_gets_a_logarithmic_scale(qapp, structure):
    """A valence map diverges at the nuclei; linear levels would be useless."""
    from facet.core import bv
    from facet.ui.volume_panel import VolumePanel

    panel = VolumePanel()
    panel.set_context(structure, bv.DEFAULT)
    panel.probe_element.setCurrentText("Bi")
    panel.probe_ox.setValue(3)
    panel.resolution.setValue(0.5)
    panel.compute()

    statistics = panel.grid.statistics()
    assert statistics["max"] / statistics["min"] > 100
    assert panel.log_scale.isChecked()
    levels = panel.section.levels
    ratios = levels[1:] / levels[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-6)


def test_the_plane_indices_change_the_section(qapp, structure):
    from facet.core import bv
    from facet.ui.volume_panel import VolumePanel

    panel = VolumePanel()
    panel.set_context(structure, bv.DEFAULT)
    panel.resolution.setValue(0.6)
    panel.probe_element.setCurrentText("Bi")
    panel.probe_ox.setValue(3)
    panel.compute()
    first = panel.section.values.copy()

    panel.h.setValue(1)
    panel.l.setValue(0)
    assert not np.allclose(panel.section.values, first)


def test_zero_indices_are_refused_with_a_reason(qapp, structure):
    from facet.core import bv
    from facet.ui.volume_panel import VolumePanel

    panel = VolumePanel()
    panel.set_context(structure, bv.DEFAULT)
    panel.resolution.setValue(0.7)
    panel.probe_element.setCurrentText("Bi")
    panel.probe_ox.setValue(3)
    panel.compute()
    panel.l.setValue(0)              # now (0 0 0)
    assert panel.section.values is None
    assert "not a plane" in panel.readout.text()


def test_atoms_on_the_plane_are_marked(qapp, structure):
    from facet.core import bv
    from facet.ui.volume_panel import VolumePanel

    panel = VolumePanel()
    panel.set_context(structure, bv.DEFAULT)
    panel.resolution.setValue(0.6)
    panel.probe_element.setCurrentText("Bi")
    panel.probe_ox.setValue(3)
    panel.compute()
    assert panel.show_atoms.isChecked()
    assert panel.section.markers, "no atoms were marked"
    labels = {label for _, _, label in panel.section.markers}
    assert labels <= {site.label for site in structure.sites}

    panel.show_atoms.setChecked(False)
    assert panel.section.markers == []


def test_clearing_removes_the_field_and_the_isosurface(qapp, structure):
    from facet.core import bv
    from facet.ui.volume_panel import VolumePanel

    panel = VolumePanel()
    panel.set_context(structure, bv.DEFAULT)
    panel.resolution.setValue(0.7)
    panel.probe_element.setCurrentText("Bi")
    panel.probe_ox.setValue(3)
    panel.compute()

    seen = []
    panel.isosurfaceChanged.connect(lambda grid, level: seen.append(grid))
    panel.clear()
    assert panel.grid is None
    assert panel.section.values is None
    assert seen and seen[-1] is None


def test_the_isosurface_signal_carries_the_grid_and_the_level(qapp, structure):
    from facet.core import bv
    from facet.ui.volume_panel import VolumePanel

    panel = VolumePanel()
    panel.set_context(structure, bv.DEFAULT)
    panel.resolution.setValue(0.7)
    panel.probe_element.setCurrentText("Bi")
    panel.probe_ox.setValue(3)
    panel.compute()

    seen = []
    panel.isosurfaceChanged.connect(lambda grid, level: seen.append((grid, level)))
    panel.level.setValue(3.0)
    panel.show_iso.setChecked(True)
    assert seen[-1][0] is panel.grid
    assert seen[-1][1] == pytest.approx(3.0)

    panel.show_iso.setChecked(False)
    assert seen[-1][0] is None


def test_a_changed_structure_clears_the_field(qapp, structure):
    """A field belongs to one cell; keeping it across a load would be wrong."""
    import copy

    from facet.core import bv
    from facet.ui.volume_panel import VolumePanel

    panel = VolumePanel()
    panel.set_context(structure, bv.DEFAULT)
    panel.resolution.setValue(0.7)
    panel.probe_element.setCurrentText("Bi")
    panel.probe_ox.setValue(3)
    panel.compute()
    assert panel.grid is not None

    panel.set_context(copy.deepcopy(structure), bv.DEFAULT)
    assert panel.grid is None


def test_the_panel_never_calls_a_basin_a_pathway(qapp, structure):
    import re

    from facet.core import bv
    from facet.ui.volume_panel import VolumePanel

    panel = VolumePanel()
    panel.set_context(structure, bv.DEFAULT)
    panel.resolution.setValue(0.7)
    panel.probe_element.setCurrentText("Bi")
    panel.probe_ox.setValue(3)
    panel.compute()

    text = " ".join([panel.stats.text(), panel.section.title,
                     panel.section.footnote]).lower()
    forbidden = {"pathway", "conduction", "conducting", "percolates",
                 "connected", "good", "bad", "should", "proves", "confirms"}
    assert not (set(re.findall(r"[a-z]+", text)) & forbidden)


def test_the_valence_surface_passes_through_the_real_sites(qapp, structure):
    """The strongest check available on the whole valence-map chain.

    A Bi(3+) probe's V = 3 surface should pass through where the bismuth atoms
    actually are, because that is where a Bi(3+) is correctly bonded. It comes out
    within about a tenth of an angstrom -- from the parameters and the geometry,
    with nothing fitted to make it so.
    """
    from facet.core import bv
    from facet.ui.volume_panel import VolumePanel

    panel = VolumePanel()
    panel.set_context(structure, bv.DEFAULT)
    panel.probe_element.setCurrentText("Bi")
    panel.probe_ox.setValue(3)
    panel.resolution.setValue(0.35)
    panel.compute()

    vertices, faces, _ = V.isosurface(panel.grid, 3.0)
    assert len(faces), "no V = 3 surface was found"

    bismuth = [atom.cart for atom in structure.atoms if atom.element == "Bi"]
    assert bismuth
    for position in bismuth:
        nearest = float(np.min(np.linalg.norm(vertices - position, axis=1)))
        assert nearest < 0.25, f"the surface misses a Bi site by {nearest:.3f} A"
