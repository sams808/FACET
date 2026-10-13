"""The ways of reading the explorer, from the View menu.

Each option is a way of showing the same contacts; none of them changes a
number. So each is tested for two things: that the drawing follows the
option, and that what it draws is the same running sum the core computes.
"""
from __future__ import annotations

import pytest

from conftest import dispose, sample_cif

from facet.core import bv, cif, coordination, shells
from facet.ui.cutoff_explorer import AXIS_DISTANCE, AXIS_VALENCE, CutoffExplorer

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


@pytest.fixture(scope="module")
def site():
    structure = cif.read(str(SAMPLE))
    return coordination.analyse_structure(structure, bv.DEFAULT, v_bond=0.075,
                                          v_list=0.02, cations_only=True)[0]


@pytest.fixture
def explorer(qapp, site):
    w = CutoffExplorer()
    w.resize(1000, 230)
    w.set_result(site)
    w.set_pin(0.075)
    w.set_threshold(0.2)
    yield w
    dispose(w)


def test_an_unknown_option_is_refused(explorer):
    with pytest.raises(ValueError):
        explorer.set_option("colour_by_mood", True)


def test_the_sum_axis_reads_in_valence_units_or_as_a_fraction(explorer, site):
    total = shells.shell_curve(site)[-1].sum_after
    explorer._layout()
    assert abs(explorer._sum_value(total) - total) < 1e-12
    explorer.set_option("sum_as_fraction", True)
    assert abs(explorer._sum_value(total) - 100.0 * total / abs(site.ox)) < 1e-9
    assert explorer._sum_value(abs(site.ox)) == 100.0
    # the axis top is above the larger of the sum and the charge
    assert explorer._sum_max() >= max(explorer._sum_value(total), 100.0)


def test_stacked_panes_split_the_plot_and_share_the_axis(explorer):
    explorer._layout()
    overlay = (explorer._plot, explorer._sum_plot)
    assert overlay[0] == overlay[1]
    explorer.set_option("stacked", True)
    explorer._layout()
    assert explorer._sum_plot.top() > explorer._plot.bottom()
    assert explorer._sum_plot.left() == explorer._plot.left()
    assert explorer._sum_plot.width() == explorer._plot.width()
    assert explorer.minimumHeight() > 210
    # a contact sits at the same x in both panes: one axis, two scales
    s = shells.shell_curve(explorer.result, explorer.axis)[0]
    x_before = explorer._shell_x(s)
    explorer.set_option("stacked", False)
    explorer._layout()
    assert abs(explorer._shell_x(s) - x_before) < 1e-9


def test_the_bar_strip_takes_its_room_only_when_asked(explorer):
    explorer._layout()
    assert explorer._bars.height() == 0
    explorer.set_option("show_bars", True)
    explorer._layout()
    assert explorer._bars.height() == 34
    assert explorer._bars.top() >= explorer._rail.bottom()
    assert explorer.grab().toImage().width() > 0


def test_the_fill_can_be_turned_off_and_the_picture_still_draws(explorer):
    explorer.set_option("fill_sum", False)
    image = explorer.grab().toImage()
    assert image.width() > 0
    explorer.set_option("fill_sum", True)


def test_every_option_draws_on_both_axes(explorer):
    for axis in (AXIS_VALENCE, AXIS_DISTANCE):
        explorer.set_axis(axis)
        for name in ("fill_sum", "sum_as_fraction", "stacked", "show_bars"):
            explorer.set_option(name, True)
            explorer._layout()
            assert explorer.grab().toImage().width() > 0
        for name in ("sum_as_fraction", "stacked", "show_bars"):
            explorer.set_option(name, False)


def test_the_menu_drives_the_explorer_and_the_site_tab(qapp):
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        win.load_many([str(SAMPLE)])
        qapp.processEvents()
        entry = win.project.current
        entry.selected_site = win.project.results_for(entry)[0].site_index
        win._rebuild()
        qapp.processEvents()

        assert win.explorer.fill_sum
        win.explorer_option_actions["stacked"].setChecked(True)
        assert win.explorer.stacked
        win.explorer_option_actions["show_bars"].setChecked(True)
        assert win.explorer.show_bars
        win.explorer_axis_actions[AXIS_DISTANCE].trigger()
        assert win.explorer.axis == AXIS_DISTANCE
        assert win.explorer_axis_actions[AXIS_DISTANCE].isChecked()
        assert not win.explorer_axis_actions[AXIS_VALENCE].isChecked()

        assert "so far" not in win.analysis.toHtml()
        win.running_sum_column.setChecked(True)
        qapp.processEvents()
        html = win.analysis.toHtml()
        assert "so far" in html
        # the column's last value is the sum to the listing cut
        r = win._current_result()
        assert f"{r.bvs_listed:.3f}" in html

        win._label_bonds_by_share()
        from facet.gl.labels import BondLabel
        assert win.bond_label_box.currentData() is BondLabel.FRACTION_OF_BVS
    finally:
        win.close()
        dispose(win)
