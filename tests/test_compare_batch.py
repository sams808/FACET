"""The Compare batch: the plateau width as a value kind, two numbers for
how far two distributions lie apart, the difference of staircases, the
Table page's "differ by more than" filter, the 3D difference colouring,
rows pinned to a frame, rows under another parameter set, linked cursors,
and the basket in the session file.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import dispose

from facet.core import bv, cif, coordination, exporters
from facet.core.md_stats import Distribution, Histogram
from facet.ui.md_compare import (
    VALUE_KINDS,
    difference_colours,
    distribution_distances,
    per_atom_values,
    rows_that_differ,
    staircase_difference,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "md"
CRYSTALS = HERE / "data" / "crystals"
QUARTZ = CRYSTALS / "quartz_SiO2_cod9013321.cif"
CRYOLITE = CRYSTALS / "cryolite_Na3AlF6_cod9004097.cif"
BIPO4 = CRYSTALS / "bismuth_phosphate_BiPO4_cod9008088.cif"


# -- pure helpers --------------------------------------------------------------------

def _hist(counts, edges=(0.0, 1.0, 2.0, 3.0)):
    counts = np.asarray([counts], float)
    zero = np.zeros(1, np.int64)
    return Histogram("h", "Å", np.asarray(edges, float), counts, zero, zero, zero)


def test_distribution_distances_are_the_ks_gap_and_the_area_between():
    a = _hist([4, 0, 0])
    b = _hist([0, 0, 4])
    far = distribution_distances(a, b)
    # cumulative fractions on the edges: a = 0,1,1,1  b = 0,0,0,1
    assert far["ks"] == pytest.approx(1.0)
    assert far["w1"] == pytest.approx(2.0)          # the whole shift
    assert far["unit"] == "Å"
    assert distribution_distances(a, a) == {"ks": 0.0, "w1": 0.0, "unit": "Å"}
    # different edges cannot be laid on one axis
    assert distribution_distances(a, _hist([1, 1], edges=(0.0, 1.0, 2.0))) is None
    # a distribution over numeric keys, mean fractions
    d1 = Distribution.from_counts([{3: 2, 4: 2}], name="Q", kind="fraction")
    d2 = Distribution.from_counts([{3: 0, 4: 4}], name="Q", kind="fraction")
    far = distribution_distances(d1, d2)
    assert far["ks"] == pytest.approx(0.5)
    assert far["w1"] == pytest.approx(0.5)


def test_rows_that_differ_keeps_only_the_rows_beyond_the_percentage():
    labels = ["A", "B"]
    rows = [{"descriptor": "x", "A mean": 1.0, "B mean": 1.04},
            {"descriptor": "y", "A mean": 1.0, "B mean": 1.2},
            {"descriptor": "z", "A mean": 0.0, "B mean": 0.001},
            {"descriptor": "w", "A mean": "—", "B mean": 2.0},
            {"descriptor": "v", "A mean": 2.0, "B mean": float("nan")}]
    kept = [r["descriptor"] for r in rows_that_differ(rows, labels, 5.0)]
    assert kept == ["y", "z"]
    assert [r["descriptor"] for r in rows_that_differ(rows, labels, 0.0)] == ["x", "y", "z"]
    # a zero on the first side keeps any non-zero on the other, whatever the percentage
    assert [r["descriptor"] for r in rows_that_differ(rows, labels, 50.0)] == ["z"]


def test_the_staircase_difference_is_zero_against_itself_and_counts_steps():
    sites = coordination.analyse_structure(cif.read(str(QUARTZ)), bv.DEFAULT,
                                           v_bond=0.075, v_list=0.02)
    si = sites[0]
    grid = np.logspace(-2, 0, 50)
    assert not np.any(staircase_difference(si, si, grid))
    cryolite = coordination.analyse_structure(cif.read(str(CRYOLITE)), bv.DEFAULT,
                                              v_bond=0.075, v_list=0.02)
    al = next(r for r in cryolite if r.element == "Al")
    d = staircase_difference(si, al, np.asarray([0.075]))
    assert d[0] == al.cn_at(0.075) - si.cn_at(0.075)


def test_plateau_is_a_value_kind_with_per_atom_values():
    from facet.core import bulk, md_model, md_readers
    assert [k for k, _n, _u in VALUE_KINDS] == ["CN", "BVS", "phi", "plateau"]
    trajectory = md_readers.read_trajectory(str(DATA / "glass.extxyz"))
    frame = trajectory.frame(0)
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT, v_bond_vu=0.075,
                                  v_list_vu=0.02)
    values = per_atom_values(table, "Si", 0.075, "plateau")
    results = bulk.at_threshold(table, 0.075)
    sel = np.asarray(table.elements) == "Si"
    assert np.array_equal(np.isfinite(values),
                          np.isfinite(np.asarray(results.plateau_decades)[sel]))
    assert values[np.isfinite(values)].min() >= 0


def test_difference_colours_follow_shared_site_labels():
    from facet.gl.scene import Scene
    quartz = cif.read(str(QUARTZ))
    results = coordination.analyse_structure(quartz, bv.DEFAULT, v_bond=0.075,
                                             v_list=0.02)
    n = 3
    scene = Scene()
    scene.atom_color = np.zeros((n, 3), np.float32)
    scene.atom_site = np.asarray([results[0].site_index] * 2 + [999], np.int32)
    colours, words = difference_colours(scene, results, scene, results, "BVS")
    assert colours.shape == (n, 3)
    # a zero difference sits at the middle of the ramp, the unmatched site is grey
    assert np.allclose(colours[0], colours[1])
    assert np.allclose(colours[2], [0.72, 0.72, 0.72])
    assert "1 labels" in words and "grey" in words
    other = coordination.analyse_structure(cif.read(str(BIPO4)), bv.DEFAULT,
                                           v_bond=0.075, v_list=0.02)
    none, reason = difference_colours(scene, other, scene, results, "CN")
    assert none is None and "share no site label" in reason
    assert difference_colours(scene, results, scene, results, "nope")[0] is None


# -- the workspace -------------------------------------------------------------------

def _wait(qapp, condition, timeout_s: float = 30.0) -> bool:
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return bool(condition())


@pytest.fixture
def window(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: QMessageBox.Ok))
    from facet.ui.preview import PreviewWindow
    w = PreviewWindow()
    yield w
    dispose(w)


def _open_model(qapp, window, path):
    controller = window.open_md_models([str(path)])[0]
    assert _wait(qapp, lambda: controller.trajectory is not None
                 or controller.problem is not None)
    assert controller.trajectory is not None, controller.problem
    window.project.set_active(window.project.index_of(controller.entry))
    window.structure_panel.refresh()
    window._show_entry()
    assert _wait(qapp, lambda: controller.entry.frame_view is not None
                 and controller.entry.frame_view.table is not None)
    return controller


def test_two_crystal_rows_give_the_difference_of_staircases(qapp, window):
    window.load(str(QUARTZ))
    si = window.results[0]
    window._add_site_to_compare(si.site_index)
    window.load(str(CRYOLITE))
    al = next(r for r in window.results if r.element == "Al")
    window._add_site_to_compare(al.site_index)
    ws = window._compare_ws
    figure = ws.staircase_difference_figure()
    assert figure is not None and len(figure.series) == 1
    assert figure.log_x and figure.y_label.startswith("CN difference")
    s = figure.series[0]
    assert al.label in s.label and si.label in s.label and "minus" in s.label
    i = int(np.argmin(np.abs(np.asarray(s.x) - 0.075)))
    v = float(s.x[i])
    assert s.mean[i] == al.cn_at(v) - si.cn_at(v)
    assert ws.select_site_figure("stairs-diff")
    assert ws.page_shown() == "figure" and ws.view.figure is not None
    # one crystal alone has nothing to subtract
    ws.clear()
    window._add_site_to_compare(al.site_index)
    assert ws.staircase_difference_figure() is None


def test_a_crystal_row_under_another_parameter_set_sits_beside_the_first(qapp, window):
    window.load(str(QUARTZ))
    si = window.results[0]
    window._add_site_to_compare(si.site_index)
    ws = window._compare_ws
    assert ws.add_basket_under_params(bv.DEFAULT, "the same set") == 1
    rows = ws.basket_rows()
    assert len(rows) == 2
    assert rows[1]["structure"].endswith("[the same set]")
    assert rows[1]["cn"] == rows[0]["cn"]
    assert rows[1]["bvs_vu"] == pytest.approx(rows[0]["bvs_vu"])
    # once only
    assert ws.add_basket_under_params(bv.DEFAULT, "the same set") == 0
    # a set without the pair leaves the row stated as not analysed
    empty = bv.ParameterSet(name="empty", fitted={}, allow_estimated=False)
    assert ws.add_basket_under_params(empty, "empty") == 1
    third = ws.basket_rows()[2]
    assert third.get("missing") or third["cn"] == 0
    # the colouring controls exist on the 3D page and refuse without two crystals
    ws.diff_colour_check.setChecked(True)
    ws.show_page("3d")
    qapp.processEvents()
    assert "two crystals" in ws.scene_note.text()


def test_pinned_frames_linked_cursors_and_the_table_filter(qapp, window):
    from PySide6.QtCore import Qt
    controller = _open_model(qapp, window, DATA / "glass.extxyz")
    window._add_model_to_compare(controller.entry, "Si")
    ws = window._compare_ws
    k = controller.entry.frame_view.k
    assert ws.pin_frames() == 1
    rows = ws.basket_rows()
    assert len(rows) == 2 and rows[1]["frame"] == k
    assert rows[1]["structure"].endswith(f"[frame {k}]")
    assert rows[1]["cn"] == pytest.approx(rows[0]["cn"])
    # pinned to a frame other than the one shown: loaded once
    assert ws.add_model_frame(controller.entry, "Si", 1 - k)
    row = ws.basket_rows()[2]
    assert row["frame"] == 1 - k and "missing" not in row
    assert ws.pin_frames() == 0

    # linked cursors: the project's threshold moves the model's strip
    for i in range(ws.models_list.count()):
        ws.models_list.item(i).setCheckState(Qt.Checked)
    ws.link_check.setChecked(True)
    before = float(controller.strip.v_bond)
    target = 0.05 if abs(before - 0.05) > 1e-3 else 0.09
    window._on_threshold(target)
    assert float(controller.strip.v_bond) == pytest.approx(target)
    ws.link_check.setChecked(False)
    window._on_threshold(0.075)
    assert float(controller.strip.v_bond) == pytest.approx(target)

    # the basket round-trips through a session file
    records = ws.basket_records()
    assert len(records) == 3 and records[1]["frame"] == k
    ws.clear()
    assert ws.basket_rows() == []
    assert ws.restore_basket(records, list(window.project.entries)) == 3
    assert [r["frame"] for r in ws.basket_rows()] == [k, k, 1 - k]


def test_the_session_file_carries_the_basket(qapp, window, tmp_path):
    window.load(str(QUARTZ))
    si = window.results[0]
    window._add_site_to_compare(si.site_index)
    path = tmp_path / "s.json"
    window.save_session(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["compare"] == [{"kind": "crystal", "path": str(QUARTZ),
                                "key": si.site_index, "frame": None}]
    window._compare_ws.clear()
    window.restore_session(exporters.load_session(path))
    rows = window._compare_ws.basket_rows()
    assert len(rows) == 1 and rows[0]["site"] == si.label
    bad = dict(data, compare="nope")
    with pytest.raises(ValueError):
        exporters.validate_session(bad)


def test_the_table_filter_keeps_the_rows_that_differ(qapp, window):
    from PySide6.QtCore import Qt
    from facet.core import md_analysis as ma, md_readers
    trajectory = md_readers.read_trajectory(str(DATA / "glass.extxyz"))
    one = ma.analyse(trajectory, ma.AnalysisRequest(
        analyses=("glass",), formers=frozenset({"Si"}), frames=slice(0, 1)))
    both = ma.analyse(trajectory, ma.AnalysisRequest(
        analyses=("glass",), formers=frozenset({"Si"}), frames=slice(0, 2)))
    first = _open_model(qapp, window, DATA / "glass.extxyz")
    second = _open_model(qapp, window, DATA / "glass.extxyz")
    first.entry.result, second.entry.result = one, both
    window._show_compare()
    ws = window._compare_ws
    for i in range(ws.models_list.count()):
        ws.models_list.item(i).setCheckState(Qt.Checked)
    ws.show_page("table")
    qapp.processEvents()
    everything = ws.table.rows()
    assert everything
    labels = [label for label, _r in ws.ticked_results()]
    ws.differ_box.setValue(1.0)
    ws.differ_check.setChecked(True)
    qapp.processEvents()
    kept = ws.table.rows()
    assert kept == rows_that_differ(everything, labels, 1.0)
    assert 0 < len(kept) < len(everything)
    # every kept row has two numbers more than 1 % apart
    for row in kept:
        a, b = row[f"{labels[0]} mean"], row[f"{labels[1]} mean"]
        assert a is not None and b is not None
        assert a == 0 or abs(b - a) / abs(a) > 0.01
    ws.differ_box.setValue(1000.0)
    qapp.processEvents()
    # only the rows whose first mean is zero survive any percentage
    assert all(row[f"{labels[0]} mean"] == 0 for row in ws.table.rows())
    assert len(ws.table.rows()) < len(kept)
    ws.differ_check.setChecked(False)
    qapp.processEvents()
    assert ws.table.rows() == everything
