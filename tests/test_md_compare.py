"""The Compare workspace (facet.ui.md_compare).

What these tests pin:

* the pure helpers: the descriptors two ModelResults share, the
  parameter-difference statements, and the value extraction that the basket
  table is built from;
* the basket: a crystal site's row equals the numbers recomputed from its
  SiteResult at the project's v_bond, and a model element's row equals the
  numpy means over that element's atoms of the frame's ValenceTable at the
  model's v_bond;
* the staircase overlay draws one curve per basket row with units on both
  axes and the current v_bond marked; the crystal-vs-glass histogram puts
  the vertical line at the crystal site's value;
* two models' shared descriptors show side by side: overlaid series and
  grouped bars, the legend naming both models; the parameter header states
  a difference in the frames analysed;
* the View menu action exists on a mnemonic no other View item claims; the
  Sites-list menu, the 3D context menu and the Elements-dock menu all fill
  the basket; compare stays open across a row switch and Back leaves it;
* no verdict word appears anywhere in the module.

Default platform, as the other workspace tests.
"""
from __future__ import annotations

import dataclasses
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import dispose  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "md"
QUARTZ = HERE / "data" / "crystals" / "quartz_SiO2_cod9013321.cif"

VERDICT_WORDS = ("good", "bad", "poor", "excellent", "acceptable", "correct",
                 "incorrect", "wrong", "reliable", "unreliable",
                 "trustworthy", "untrustworthy", "unusable", "should",
                 "proves", "confirms")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(qapp, condition, timeout_s: float = 30.0) -> bool:
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(condition())


@pytest.fixture(scope="module")
def glass_results():
    """Two runs of the glass analysis on glass.extxyz: one frame, then
    both frames, so their provenances differ in the frames analysed."""
    from facet.core import md_analysis as ma, md_readers

    trajectory = md_readers.read_trajectory(str(DATA / "glass.extxyz"))
    one = ma.analyse(trajectory, ma.AnalysisRequest(
        analyses=("glass",), formers=frozenset({"Si"}), frames=slice(0, 1)))
    both = ma.analyse(trajectory, ma.AnalysisRequest(
        analyses=("glass",), formers=frozenset({"Si"}), frames=slice(0, 2)))
    assert not one.failed and not both.failed
    return one, both


@pytest.fixture
def warnings(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    seen: list[tuple[str, str]] = []

    def record(_parent, title, text, *args, **kwargs):
        seen.append((title, text))
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(record))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(record))
    return seen


@pytest.fixture
def window(qapp, warnings):
    from facet.ui.preview import PreviewWindow

    w = PreviewWindow()
    yield w
    dispose(w)


def _open_model(qapp, window, path):
    controllers = window.open_md_models([str(path)])
    assert len(controllers) == 1
    controller = controllers[0]
    assert _wait(qapp, lambda: controller.trajectory is not None
                 or controller.problem is not None), "the file was not read"
    assert controller.trajectory is not None, controller.problem
    index = window.project.index_of(controller.entry)
    window.project.set_active(index)
    window.structure_panel.refresh()
    window._show_entry()
    assert _wait(qapp, lambda: controller.entry.frame_view is not None
                 and controller.entry.frame_view.table is not None), \
        "no frame was drawn with its table"
    return controller


# ---------------------------------------------------------------------------
# the pure helpers
# ---------------------------------------------------------------------------

def test_common_descriptors_are_those_in_at_least_two_results(glass_results):
    from facet.ui.md_compare import common_descriptors

    one, both = glass_results
    shared = common_descriptors([one, both])
    everything = {(a, n) for a, n, _c in one.descriptors()} \
        & {(a, n) for a, n, _c in both.descriptors()}
    assert set(shared) == everything and shared

    lonely = SimpleNamespace(descriptors=lambda: iter(
        [("glass", "only here", None)]))
    assert ("glass", "only here") not in common_descriptors([one, lonely])


def test_parameter_differences_state_frames_formers_and_v_bond(
        glass_results):
    from facet.ui.md_compare import parameter_differences

    one, both = glass_results
    assert parameter_differences([("a", one), ("b", one)]) == []
    stated = parameter_differences([("a", one), ("b", both)])
    assert any("frames" in line and "a" in line and "b" in line
               for line in stated)
    moved = dataclasses.replace(
        one, provenance=dataclasses.replace(one.provenance, v_bond_vu=0.31))
    lines = parameter_differences([("a", one), ("b", moved)])
    assert any("v_bond" in line and "0.31" in line for line in lines)
    renamed = dataclasses.replace(
        one, provenance=dataclasses.replace(one.provenance,
                                            formers=frozenset({"P"})))
    lines = parameter_differences([("a", one), ("b", renamed)])
    assert any("formers" in line and "P" in line for line in lines)


def test_crystal_values_equal_the_site_result_they_come_from():
    from facet.core import bv, readers
    from facet.core.coordination import analyse_structure
    from facet.ui.md_compare import crystal_values

    structure = readers.read(QUARTZ)
    result = analyse_structure(structure)[0]
    v = bv.V_BOND_DEFAULT
    row = crystal_values(result, v)
    assert row["cn"] == result.cn_at(v) == 4
    bonded = [c for c in result.contacts
              if c.has_valence and c.valence > v]
    assert row["bvs_vu"] == pytest.approx(
        bv.bvs([c.valence * c.occupancy for c in bonded]))
    assert row["d_mean_ang"] == pytest.approx(
        np.mean([c.distance for c in bonded]))
    plateau = result.current_plateau
    assert row["plateau_decades"] == pytest.approx(plateau.width_decades)
    # moved onto another plateau, the row follows the threshold it was asked
    high = crystal_values(result, 1.2)
    assert high["cn"] == result.cn_at(1.2) < 4


# ---------------------------------------------------------------------------
# the basket in the window
# ---------------------------------------------------------------------------

def test_the_basket_rows_equal_the_numbers_they_come_from(qapp, window):
    from facet.core import bulk
    from facet.ui.md_compare import model_values

    controller = _open_model(qapp, window, DATA / "glass.extxyz")
    window.load(str(QUARTZ))
    assert window.project.current.kind == "crystal"
    site = window.results[0]
    window._add_site_to_compare(site.site_index)
    page = window._compare_page
    assert page is not None and window.compare_on
    assert window.column_stack.currentWidget() is page
    window._add_model_to_compare(controller.entry, "O")
    rows = page.basket_rows()
    assert [r["kind"] for r in rows] == ["crystal", "model"]

    crystal = rows[0]
    assert crystal["site"] == site.label
    assert crystal["cn"] == site.cn_at(window.project.v_bond)

    model = rows[1]
    fv = controller.entry.frame_view
    v_bond = float(controller.strip.v_bond)
    results = bulk.at_threshold(fv.table, v_bond)
    sel = np.asarray(fv.table.elements) == "O"
    assert model["atoms"] == int(sel.sum())
    assert model["cn"] == pytest.approx(
        float(np.asarray(results.cn, float)[sel].mean()))
    bvs = np.asarray(results.bvs_vu, float)[sel]
    assert model["bvs_vu"] == pytest.approx(
        float(bvs[np.isfinite(bvs)].mean()))
    assert model["frame"] == fv.k
    helper = model_values(fv.table, "O", v_bond)
    for key, value in helper.items():
        if key == "element":
            continue
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(model[key]), key
        else:
            assert model[key] == pytest.approx(value), key
    # the table lists one line per reference plus nothing else
    assert page.basket_table.rowCount() == 2

    # compare stays open across a row switch; Back leaves it
    window.structure_panel.list.setCurrentRow(0)
    qapp.processEvents()
    assert window.project.current is controller.entry
    assert window.column_stack.currentWidget() is page
    page.back_button.click()
    assert not window.compare_on
    assert window.column_stack.currentWidget() is controller.column

    # a v_bond move re-reads the crystal side through the existing signal
    window._on_threshold(1e-3)
    row = page.basket_rows()[0]
    assert row["cn"] == site.cn_at(1e-3)


def test_the_staircase_overlay_and_the_crystal_vs_glass_line(qapp, window):
    controller = _open_model(qapp, window, DATA / "glass.extxyz")
    window.load(str(QUARTZ))
    site = window.results[0]                        # a Si site of quartz
    assert site.element == "Si"
    window._add_site_to_compare(site.site_index)
    window._add_model_to_compare(controller.entry, "Si")
    page = window._compare_page

    figure = page.stairs_plot.figure
    assert figure is not None and len(figure.series) == 2
    assert figure.x_label == "v_bond (v.u.)"
    assert figure.y_label == "CN (dimensionless)"
    assert figure.log_x
    labels = [s.label for s in figure.series]
    assert any(site.label in text for text in labels)
    assert any("glass.extxyz" in text and "Si" in text for text in labels)
    v = float(window.project.v_bond)
    assert any(kind == "current" and x == pytest.approx(v)
               for x, _t, kind in figure.vlines)
    # the two curves carry distinct styles, so colour never stands alone
    a, b = (s.style for s in figure.series)
    assert (a.colour, a.marker) != (b.colour, b.marker)

    hist = page.hist_plot.figure
    assert hist is not None and len(hist.series) == 1
    assert hist.series[0].kind == "step"
    assert hist.x_label == "CN (dimensionless)"
    assert "fraction" in hist.y_label
    expected = float(site.cn_at(v))
    assert any(x == pytest.approx(expected) and site.label in text
               for x, text, _k in hist.vlines)

    # an element with no matching crystal site is not binned
    page.clear()
    window._add_model_to_compare(controller.entry, "Na")
    assert page.hist_plot.figure is None


# ---------------------------------------------------------------------------
# the same calculations across models
# ---------------------------------------------------------------------------

def test_two_models_shared_descriptors_show_side_by_side(qapp, window,
                                                         glass_results):
    from PySide6.QtCore import Qt
    from facet.core.md_stats import Distribution, Series

    one, both = glass_results
    first = _open_model(qapp, window, DATA / "glass.extxyz")
    second = _open_model(qapp, window, DATA / "glass.extxyz")
    first.entry.result = one
    second.entry.result = both
    window._show_compare()
    page = window._compare_page
    assert page.models_list.count() == 2
    assert all(page.models_list.item(i).checkState() == Qt.Checked
               for i in range(2))
    assert page.descriptor_list.count() > 0
    # the header states the frame difference, in words, without a verdict
    assert not page.warning.isHidden()
    assert "frames" in page.warning.text()

    def pick(kind) -> bool:
        for i in range(page.descriptor_list.count()):
            item = page.descriptor_list.item(i)
            analysis, name = item.data(Qt.UserRole)
            if isinstance(one.outputs[analysis].tables.get(name), kind):
                page.descriptor_list.setCurrentItem(item)
                return True
        return False

    names = {page.models_list.item(i).text() for i in range(2)}
    assert pick(Series)
    figure = page.view.figure
    assert figure is not None and len(figure.series) == 2
    assert {s.label for s in figure.series} == names
    assert {s.kind for s in figure.series} == {"line"}
    rows = page.view.rows()
    assert {r["model"] for r in rows} == names

    assert pick(Distribution)
    figure = page.view.figure
    assert figure is not None and len(figure.series) == 2
    assert {s.kind for s in figure.series} == {"bars"}
    assert {s.label for s in figure.series} == names

    # the filter narrows the list without touching the data
    page.filter_edit.setText("zzz-no-such-descriptor")
    assert all(page.descriptor_list.item(i).isHidden()
               for i in range(page.descriptor_list.count()))
    page.filter_edit.setText("")


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def test_the_view_menu_action_exists_on_its_own_mnemonic(window):
    from collections import Counter

    action = window.compare_action
    # the bar's action wrappers are kept alive for the whole walk: a menu
    # reached through a collected wrapper is torn down with it (PySide)
    bar_actions = window.menuBar().actions()
    view = next(a.menu() for a in bar_actions if a.text() == "&View")
    items = view.actions()
    assert action in items
    assert action.shortcut().toString() == "M"
    letters = Counter()
    for item in items:
        text = item.text()
        if "&" in text and text.index("&") + 1 < len(text):
            letters[text[text.index("&") + 1].lower()] += 1
    assert letters["m"] == 1
    assert max(letters.values()) == 1


def test_the_three_right_click_menus_fill_the_basket(qapp, window):
    controller = _open_model(qapp, window, DATA / "glass.extxyz")
    window.load(str(QUARTZ))
    qapp.processEvents()

    def action_named(menu, text):
        found = [a for a in menu.actions() if a.text().startswith(text)]
        assert found, [a.text() for a in menu.actions()]
        return found[0]

    site_menu = window._build_site_menu(0)
    action_named(site_menu, "Add to comparison").trigger()
    page = window._compare_page
    assert [kind for kind, _e, _k in page.refs()] == ["crystal"]

    context = window._build_context_menu(0)
    action_named(context, "Add to comparison").trigger()
    assert len(page.refs()) == 1                 # the same site only once

    assert _wait(qapp, lambda: controller.elements.table.rowCount() > 0)
    element_menu = controller.elements._element_menu(0)
    symbol = controller.elements.table.item(0, 0).text()
    action_named(element_menu, f"Add {symbol} to comparison").trigger()
    kinds = [kind for kind, _e, _k in page.refs()]
    assert kinds == ["crystal", "model"]
    assert page.refs()[1][2] == symbol

    # removing the model entry drops its basket row
    row = window.project.index_of(controller.entry)
    window._on_remove(row)
    assert [kind for kind, _e, _k in page.refs()] == ["crystal"]


def test_no_verdict_word_appears_in_the_module(qapp):
    source = (HERE.parent / "facet" / "ui" / "md_compare.py").read_text(
        encoding="utf-8")
    found = [word for word in VERDICT_WORDS
             if re.search(rf"\b{word}\b", source, re.IGNORECASE)]
    assert not found, found
