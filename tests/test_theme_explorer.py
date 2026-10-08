"""Colour themes and the cutoff explorer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import theme as T

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


# --- palettes and overrides --------------------------------------------------

def test_every_shipped_palette_covers_the_common_elements():
    for name, factory in T.PALETTES.items():
        p = factory()
        for sym in ("O", "Si", "Na", "Fe", "Bi"):
            assert sym in p, f"{name} has no colour for {sym}"
            assert all(0.0 <= c <= 1.0 for c in p[sym]), f"{name}/{sym} out of range"


def test_an_element_colour_can_be_overridden_and_restored():
    t = T.Theme()
    original = t.element_color("Bi")
    t.set_element_color("Bi", (0.1, 0.2, 0.3))
    assert t.element_color("Bi") == (0.1, 0.2, 0.3)
    t.clear_element_color("Bi")
    assert t.element_color("Bi") == original


def test_overrides_survive_a_palette_change():
    """A user's own colour is theirs; switching palette must not discard it."""
    t = T.Theme()
    t.set_element_color("O", (0.0, 0.0, 1.0))
    t.palette_name = "Jmol / CPK"
    assert t.element_color("O") == (0.0, 0.0, 1.0)


def test_label_normalisation_applies_to_overrides():
    t = T.Theme()
    t.set_element_color("bi", (0.5, 0.5, 0.5))
    assert t.element_color("Bi") == (0.5, 0.5, 0.5)
    assert t.element_color("Bi3+") == (0.5, 0.5, 0.5)


def test_greyscale_palette_is_actually_grey():
    for rgb in T.PALETTES["Greyscale"]().values():
        assert rgb[0] == pytest.approx(rgb[1]) == pytest.approx(rgb[2])


def test_greyscale_orders_by_atomic_number():
    """Heavier must be darker, or the ordering is lost when printed."""
    p = T.PALETTES["Greyscale"]()
    assert p["Bi"][0] < p["O"][0]


def test_high_contrast_avoids_pure_red_against_green():
    """Red against green is the pairing that fails most often, and is exactly
    what CPK uses for oxygen against chlorine."""
    p = T.PALETTES["High contrast"]()
    o, cl = p["O"], p["Cl"]
    assert not (o[0] > 0.7 and o[1] < 0.3 and cl[1] > 0.6 and cl[0] < 0.3)


# --- persistence -------------------------------------------------------------

def test_a_theme_round_trips_through_a_file(tmp_path):
    t = T.publication()
    t.set_element_color("Bi", (0.9, 0.1, 0.4))
    t.color_mode = T.ColorMode.PHI
    t.bond_color_mode = T.BondColorMode.BY_VALENCE
    t.scale_min, t.scale_max = 0.0, 0.6
    path = tmp_path / "theme.json"
    t.save(path)

    back = T.Theme.load(path)
    assert back.palette_name == t.palette_name
    assert back.element_color("Bi") == (0.9, 0.1, 0.4)
    assert back.color_mode is T.ColorMode.PHI
    assert back.bond_color_mode is T.BondColorMode.BY_VALENCE
    assert back.background == t.background
    assert (back.scale_min, back.scale_max) == (0.0, 0.6)
    assert back.fog_amount == t.fog_amount


def test_a_theme_file_is_readable_json(tmp_path):
    path = tmp_path / "t.json"
    T.Theme().save(path)
    json.loads(path.read_text(encoding="utf-8"))


def test_loading_a_theme_with_missing_keys_uses_defaults():
    back = T.Theme.from_dict({"name": "sparse"})
    assert back.name == "sparse"
    assert back.color_mode is T.ColorMode.ELEMENT


def test_loading_a_theme_with_a_bad_mode_does_not_raise():
    back = T.Theme.from_dict({"color_mode": "not a mode"})
    assert back.color_mode is T.ColorMode.ELEMENT


def test_copy_does_not_share_the_override_dictionary():
    a = T.Theme()
    b = a.copy()
    b.set_element_color("O", (1.0, 0.0, 0.0))
    assert "O" not in a.overrides


# --- presets -----------------------------------------------------------------

def test_presets_all_construct_and_differ():
    made = {name: factory() for name, factory in T.PRESETS.items()}
    assert len(made) == len(T.PRESETS)
    backgrounds = {tuple(t.background) for t in made.values()}
    assert len(backgrounds) > 1


def test_light_and_publication_themes_know_they_are_light():
    assert T.publication().is_light_background
    assert T.light().is_light_background
    assert not T.dark().is_light_background


def test_a_light_theme_gives_dark_ink():
    """Switching to a white background must not keep pale grey text."""
    ink = T.publication().contrasting_ink()
    assert sum(ink) < 1.0


def test_publication_theme_turns_off_depth_cueing():
    """Depth cueing reads as haze in print and makes a figure look badly
    reproduced rather than three-dimensional."""
    assert T.publication().fog_amount == 0.0


# --- ramps and scaling -------------------------------------------------------

def test_ramp_is_continuous_and_in_range():
    previous = T.ramp(0.0)
    for t in np.linspace(0, 1, 60)[1:]:
        c = T.ramp(float(t))
        assert all(0.0 <= v <= 1.0 for v in c)
        assert max(abs(a - b) for a, b in zip(c, previous)) < 0.35
        previous = c


def test_ramp_handles_nan():
    assert T.ramp(float("nan")) == T.ramp(None)


def test_diverging_ramp_is_symmetric_about_its_middle():
    lo, hi = T.ramp(0.0, diverging=True), T.ramp(1.0, diverging=True)
    assert lo != hi
    mid = T.ramp(0.5, diverging=True)
    assert abs(mid[0] - mid[2]) < 0.15        # near-neutral at the midpoint


def test_valence_discrepancy_range_is_symmetric_about_zero():
    """A signed quantity must put zero in the middle of the ramp, or the sign
    stops being readable."""
    values = {0: -0.1, 1: 0.4}
    lo, hi = T.scale_range(values, T.Theme(), T.ColorMode.VALENCE_DISCREPANCY)
    assert lo == pytest.approx(-hi)


def test_a_pinned_range_overrides_autoscale():
    """Pinning is what makes two figures comparable."""
    t = T.Theme(scale_min=0.0, scale_max=1.0)
    assert T.scale_range({0: 5.0}, t, T.ColorMode.BVS) == (0.0, 1.0)


def test_scale_range_survives_all_nan():
    lo, hi = T.scale_range({0: float("nan")}, T.Theme(), T.ColorMode.BVS)
    assert lo < hi


def test_identical_values_still_give_a_usable_range():
    lo, hi = T.scale_range({0: 3.0, 1: 3.0}, T.Theme(), T.ColorMode.BVS)
    assert hi > lo


# --- colour modes on a real structure ----------------------------------------

@pytest.mark.skipif(not SAMPLE.exists(), reason="sample structure not present")
class TestColorModesOnAStructure:

    @pytest.fixture(scope="class")
    def loaded(self):
        from facet.core import cif, coordination

        s = cif.read(SAMPLE)
        return s, coordination.analyse_structure(s)

    def test_element_mode_gives_one_colour_per_element(self, loaded):
        from facet.gl.scene import build_scene

        s, res = loaded
        scene = build_scene(s, res, theme=T.Theme(color_mode=T.ColorMode.ELEMENT))
        assert len(np.unique(scene.atom_color, axis=0)) == 2      # Bi and O

    def test_phi_mode_distinguishes_the_two_bismuth_sites(self, loaded):
        """The two Bi sites of alpha-Bi2O3 have phi 0.433 and 0.370, so a
        stereoactivity colouring must tell them apart."""
        from facet.gl.scene import build_scene

        s, res = loaded
        scene = build_scene(s, res, theme=T.Theme(color_mode=T.ColorMode.PHI))
        bi = [i for i, e in enumerate(scene.atom_element) if e == "Bi"]
        assert len(np.unique(scene.atom_color[bi], axis=0)) == 2

    def test_uniform_mode_gives_one_colour(self, loaded):
        """Including the periodic images: an image atom is an atom, and must
        not take its colour from the bond that reached it."""
        from facet.gl.scene import build_scene

        s, res = loaded
        scene = build_scene(s, res, theme=T.Theme(color_mode=T.ColorMode.UNIFORM))
        assert len(np.unique(scene.atom_color, axis=0)) == 1

    def test_image_atoms_match_the_atoms_they_duplicate(self, loaded):
        from facet.gl.scene import build_scene

        s, res = loaded
        t = T.Theme(bond_color_mode=T.BondColorMode.BY_VALENCE)
        scene = build_scene(s, res, theme=t)
        cell = np.unique(scene.atom_color[:scene.n_cell_atoms], axis=0)
        image = np.unique(scene.atom_color[scene.n_cell_atoms:], axis=0)
        for c in image:
            assert any(np.allclose(c, d) for d in cell),                 "an image atom has a colour no cell atom has"

    def test_the_bond_scale_survives_moving_the_threshold(self, loaded):
        """restyle() recomputes the radii, and must reapply the theme's scale
        rather than reverting every bond to the unscaled width."""
        from facet.gl.scene import build_scene

        s, res = loaded
        scene = build_scene(s, res, theme=T.Theme(bond_scale=2.5))
        before = float(scene.bond_radius.mean())
        scene.restyle(scene.v_bond)
        assert float(scene.bond_radius.mean()) == pytest.approx(before, rel=1e-6)

    def test_an_override_reaches_the_scene(self, loaded):
        from facet.gl.scene import build_scene

        s, res = loaded
        t = T.Theme()
        t.set_element_color("Bi", (1.0, 0.0, 1.0))
        scene = build_scene(s, res, theme=t)
        bi = [i for i, e in enumerate(scene.atom_element) if e == "Bi"]
        assert np.allclose(scene.atom_color[bi[0]], [1.0, 0.0, 1.0])

    def test_atom_and_bond_scale_reach_the_scene(self, loaded):
        from facet.gl.scene import build_scene

        s, res = loaded
        small = build_scene(s, res, theme=T.Theme(atom_scale=0.5, bond_scale=0.5))
        big = build_scene(s, res, theme=T.Theme(atom_scale=2.0, bond_scale=2.0))
        assert big.atom_radius.mean() == pytest.approx(
            small.atom_radius.mean() * 4.0, rel=1e-5)
        assert big.bond_radius.mean() > small.bond_radius.mean()

    def test_the_subthreshold_colour_follows_the_theme(self, loaded):
        from facet.gl.scene import build_scene

        s, res = loaded
        t = T.Theme(subthreshold_color=(1.0, 0.0, 0.0))
        scene = build_scene(s, res, theme=t)
        below = scene.bond_valence < scene.v_bond
        assert below.any()
        # faded towards red, so the red channel must exceed the blue
        faded = scene.bond_color_a[below]
        assert (faded[:, 0] > faded[:, 2]).any()

    def test_cell_range_replicates_contents_and_outlines_every_cell(self, loaded):
        from facet.gl.scene import build_scene

        s, res = loaded
        one = build_scene(s, res, theme=T.Theme())
        block = build_scene(s, res, theme=T.Theme(), cell_range=(2, 2, 1))
        assert block.n_cell_atoms == one.n_cell_atoms * 4
        assert block.n_bonds == one.n_bonds * 4
        assert len(block.cell_segments) == 12 * 4

    def test_all_polyhedra_covers_every_cation_site(self, loaded):
        from facet.gl.scene import build_scene

        s, res = loaded
        one = build_scene(s, res, theme=T.Theme(),
                          polyhedron_sites=[res[0].site_index])
        every = build_scene(s, res, theme=T.Theme(),
                            polyhedron_sites=[r.site_index for r in res])
        assert every.n_poly_triangles > one.n_poly_triangles


# --- the cutoff explorer -----------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample structure not present")
class TestCutoffExplorer:

    @pytest.fixture(scope="class")
    def result(self):
        from facet.core import cif, coordination

        s = cif.read(SAMPLE)
        return coordination.analyse_structure(s)[0]

    def _widget(self, qapp, result):
        from facet.ui.cutoff_explorer import CutoffExplorer

        w = CutoffExplorer()
        w.resize(900, 220)
        w.set_result(result)
        return w

    def test_the_valence_axis_round_trips(self, qapp, result):
        w = self._widget(qapp, result)
        w._layout()
        for v in (0.01, 0.075, 0.3):
            assert w._x_to_v(w._v_to_x(v)) == pytest.approx(v, rel=1e-6)

    def test_the_axis_is_monotonic(self, qapp, result):
        w = self._widget(qapp, result)
        w._layout()
        xs = [w._v_to_x(v) for v in (0.005, 0.02, 0.075, 0.2, 0.5)]
        assert all(b > a for a, b in zip(xs, xs[1:]))

    def test_it_renders_without_error_in_both_themes(self, qapp, result):
        for theme in (T.dark(), T.publication()):
            w = self._widget(qapp, result)
            w.set_theme(theme)
            image = w.grab().toImage()
            seen = set()
            for y in range(0, image.height(), 7):
                for x in range(0, image.width(), 7):
                    c = image.pixelColor(x, y)
                    seen.add((c.red(), c.green(), c.blue()))
            assert len(seen) > 8, f"{theme.name} drew nothing"

    def test_it_survives_having_no_result(self, qapp):
        from facet.ui.cutoff_explorer import CutoffExplorer

        w = CutoffExplorer()
        w.resize(400, 200)
        assert not w.grab().isNull()

    def test_double_click_snaps_to_the_middle_of_a_plateau(self, qapp, result):
        """The useful gesture: land on the most defensible threshold rather
        than hunting for it by hand."""
        import math

        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        w = self._widget(qapp, result)
        w.resize(900, 220)
        w._layout()
        widest = max(result.plateaus, key=lambda q: q.width_decades)
        target_v = math.sqrt(widest.v_low * widest.v_high)
        # click somewhere inside that plateau but not at its centre
        click_v = math.sqrt(widest.v_low * target_v)
        x = w._v_to_x(click_v)

        event = QMouseEvent(QMouseEvent.MouseButtonDblClick,
                            QPointF(x, w._plot.center().y()),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        w.mouseDoubleClickEvent(event)
        assert w.v_bond == pytest.approx(target_v, rel=1e-6)

    def test_the_threshold_is_clamped_to_the_visible_axis(self, qapp, result):
        """To the axis this site actually draws, not to a fixed 0.6 v.u.

        The name was always right and the constant was not: the axis now grows
        to hold the site's own contacts, and on a silicate every one of them
        lies above 0.6, so a clamp there stopped the threshold short of where
        the coordination number changes.
        """
        w = self._widget(qapp, result)
        low, high = w._data_range()
        w.set_threshold(1e6)
        assert w.v_bond <= high
        w.set_threshold(-5.0)
        assert w.v_bond >= low
        # the shipped span is a floor, so a site that fitted before still does
        assert low <= 0.004 and high >= 0.6

    def test_the_reference_distance_can_be_cleared(self, qapp, result):
        w = self._widget(qapp, result)
        w.set_reference_distance(None)
        assert not w.grab().isNull()


# --- the cutoff explorer's layout --------------------------------------------

def _explorer_with(result, width=1148, height=230):
    """A laid-out explorer showing one site's result."""
    from facet.ui.cutoff_explorer import CutoffExplorer

    e = CutoffExplorer()
    e.resize(width, height)
    e.set_result(result)
    e._layout()
    return e


def _si_result():
    """A site whose reference marks nearly coincide, which is the hard case.

    The tabulation threshold and the distance convention land within a few
    pixels of each other for a short bond, so their labels collide and have to
    be staggered -- and the stagger is what used to be printed over the
    readout.
    """
    from pathlib import Path as _Path

    import pytest as _pytest

    from facet.core import bv, cif, coordination

    sample = sample_cif("9011871", "9011871_bismutoferrite.cif")
    if not sample.is_file():
        _pytest.skip("the sample structure is not present")
    structure = cif.read(str(sample))
    results = coordination.analyse_structure(structure, bv.DEFAULT)
    return next(r for r in results if r.element == "Si")


def test_the_explorer_leaves_room_for_every_label_row(qapp):
    """A staggered reference label must not be printed over the readout.

    The stagger worked; the layout did not know about it. `bottom` reserved a
    fixed two lines, so the second row landed on the readout at the foot of the
    widget -- visible exactly when two marks nearly coincide, which is the case
    the figure exists to make a point about.
    """
    from PySide6.QtGui import QFontMetricsF

    e = _explorer_with(_si_result())
    small = QFontMetricsF(e._small_font())
    rows = e._reference_rows(small)
    assert rows >= 2, "this site was chosen because its marks collide"

    last_row_bottom = (e._rail.bottom() + 2 + (rows - 1) * (small.height() - 1)
                       + small.height())
    readout_top = e.height() - QFontMetricsF(e.font()).height() - 2
    assert last_row_bottom <= readout_top, (
        f"the last label row ends at {last_row_bottom:.0f} and the readout "
        f"starts at {readout_top:.0f}: they overlap")
    assert last_row_bottom <= e.height()


def test_the_plot_box_is_the_data_box(qapp):
    """The shading, the gridlines and the staircase must share one boundary.

    The headroom above the top step used to be kept inside the plot rectangle,
    so the plateau shading -- which fills the rectangle -- ran 16 px above the
    highest gridline and above the staircase's top step.
    """
    result = _si_result()
    e = _explorer_with(result)
    cn_max = max([q.cn for q in result.plateaus] + [1])
    assert e._cn_to_y(cn_max, cn_max) == pytest.approx(e._plot.top(), abs=0.01)
    assert e._cn_to_y(0, cn_max) == pytest.approx(e._plot.bottom(), abs=0.01)
    # and the headroom is real, above the box
    assert e._plot.top() >= e.HEADROOM


def test_nothing_the_explorer_draws_falls_outside_it(qapp):
    """Every element, at several widget sizes."""
    from PySide6.QtGui import QFontMetricsF

    result = _si_result()
    for width, height in ((1148, 230), (1467, 230), (700, 230), (520, 260)):
        e = _explorer_with(result, width, height)
        small = QFontMetricsF(e._small_font())
        assert e._plot.height() > 0, (width, height)
        assert e._plot.left() >= 0 and e._plot.right() <= width
        assert e._rail.bottom() <= height
        for _v, _label, _col, left, w, _row in e._place_references(small):
            assert left >= 0, (width, _label, left)
            assert left + w <= width + 1, (width, _label, left + w)
        readout_top = e.height() - QFontMetricsF(e.font()).height() - 2
        assert readout_top >= e._rail.bottom(), (width, height)


def test_the_axis_caption_does_not_sit_on_the_top_tick(qapp):
    """"CN" and the highest number were printed on top of each other."""
    result = _si_result()
    e = _explorer_with(result)
    cn_max = max([q.cn for q in result.plateaus] + [1])
    caption_centre = e._plot.top() - e.HEADROOM - 2 + 8
    tick_centre = e._cn_to_y(cn_max, cn_max)
    assert abs(caption_centre - tick_centre) >= 10, (
        "the CN caption and the top tick overlap")


def test_no_contact_is_drawn_outside_the_plot(qapp):
    """The fault the user saw: the axis stopped and the plot did not.

    `_v_to_x` clamped only the low end against a fixed V_MAX of 0.6 v.u., a
    span chosen for Bi-O. A silicon-oxygen bond is about 1.0 v.u. and a
    phosphorus-oxygen bond more, so the tick, the step and the plateau shading
    for every one of those contacts were painted past `plot.right()`, onto the
    widget's margin, and cut off at its edge. Measured over the reference
    collection, 81 of 143 sites had at least one contact outside the box, the
    worst 213 px past the right edge of a 1088 px plot.

    Run over every site of every structure available, because the sites that
    broke were not the ones the explorer was designed around.
    """
    from pathlib import Path as _Path

    from facet.core import bv, cif, coordination
    from facet.ui.cutoff_explorer import CutoffExplorer

    folder = _Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs")
    if not folder.is_dir():
        pytest.skip("the CIF collection is not present")

    widget = CutoffExplorer()
    widget.resize(1148, 230)
    checked = 0
    # 30 files, not 18. The collection was reorganised into folders by
    # chemical system and deduplicated, so the first 18 by sort order are no
    # longer the same 18 and cover fewer sites than they did.
    for path in sorted(p for p in folder.rglob("*.cif")
                       if "_duplicates" not in p.parts)[:30]:
        try:
            structure = cif.read(str(path))
            results = coordination.analyse_structure(structure, bv.DEFAULT)
        except Exception:
            continue
        for result in results:
            values = [c.valence for c in result.contacts if c.has_valence]
            if not values:
                continue
            widget.set_result(result)
            widget._layout()
            checked += 1
            for v in values:
                x = widget._v_to_x(v)
                assert widget._plot.left() - 0.5 <= x <= widget._plot.right() + 0.5, (
                    f"{path.name} {result.label}: a contact at {v:.4f} v.u. is "
                    f"drawn at x={x:.0f}, outside the plot "
                    f"{widget._plot.left():.0f}..{widget._plot.right():.0f}")
    assert checked > 40, f"only {checked} sites were actually checked"


def test_the_axis_holds_the_threshold_wherever_it_is(qapp):
    """One threshold serves the whole structure, and sites differ.

    A value set on a site with strong bonds must not leave the marker off the
    end of the plot when a site with weak ones is selected.
    """
    from facet.ui.cutoff_explorer import CutoffExplorer

    widget = CutoffExplorer()
    widget.resize(1148, 230)
    widget._layout()
    for v in (0.004, 0.02, 0.075, 0.6, 1.5, 3.0):
        widget.v_bond = v
        low, high = widget._v_range()
        assert low <= v <= high, (v, low, high)
        x = widget._v_to_x(v)
        assert widget._plot.left() - 0.5 <= x <= widget._plot.right() + 0.5


# --- the theme panel ---------------------------------------------------------

def test_the_theme_panel_shows_the_window_behind_its_page(qapp):
    """The Theme tab built a scroll area of its own, whose page and viewport
    filled with the application palette's Window colour. With a dark system
    palette that palette does not replace, the theme's dark text sat on
    (30, 30, 30): 1.05:1. The page now shows the styled window behind it."""
    from PySide6.QtGui import QColor, QImage, QPalette
    from PySide6.QtWidgets import QLabel, QMainWindow, QScrollArea

    from facet.ui import chrome
    from facet.ui.theme_panel import ThemePanel

    sheet, palette = qapp.styleSheet(), qapp.palette()
    dark = QPalette()
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        for role in (QPalette.Window, QPalette.Base, QPalette.Button):
            dark.setColor(group, role, QColor(30, 30, 30))
    window = QMainWindow()
    try:
        qapp.setPalette(dark)
        # the sheet alone: the platform's palette left as it is
        qapp.setStyleSheet(chrome.stylesheet(T.Theme()))
        panel = ThemePanel()
        window.setCentralWidget(panel)
        window.resize(420, 700)
        window.show()
        qapp.processEvents()
        area = panel.findChild(QScrollArea)
        assert not area.viewport().autoFillBackground()
        assert not area.widget().autoFillBackground()
        image = window.grab().toImage().convertToFormat(QImage.Format_RGB32)
        dpr = image.devicePixelRatio()
        raw = np.frombuffer(image.constBits(), dtype=np.uint8,
                            count=image.bytesPerLine() * image.height())
        pixels = raw.reshape(image.height(), image.bytesPerLine())[
            :, :image.width() * 4].reshape(image.height(), image.width(),
                                           4)[..., [2, 1, 0]]
        labels = [w for w in panel.findChildren(QLabel)
                  if w.isVisible() and w.text().strip()
                  and not w.visibleRegion().isEmpty()]
        assert labels
        for label in labels[:8]:
            top_left = label.mapTo(window, label.rect().topLeft())
            x0, y0 = int(top_left.x() * dpr), int(top_left.y() * dpr)
            block = pixels[y0:y0 + int(label.height() * dpr),
                           x0:x0 + int(label.width() * dpr)].reshape(-1, 3)
            colours, counts = np.unique(block, axis=0, return_counts=True)
            behind = QColor(*(int(v) for v in colours[np.argmax(counts)]))
            # behind a label: the theme's panel, never the system's
            # (30, 30, 30)
            assert behind.lightness() > 200, (label.text(), behind.name(),
                                              chrome.ui_colors(T.Theme())
                                              .window)
    finally:
        window.close()
        window.deleteLater()
        qapp.setStyleSheet(sheet)
        qapp.setPalette(palette)
        qapp.processEvents()
