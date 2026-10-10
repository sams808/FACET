"""The Model workspace inside the main window (facet.ui.md_workspace).

What these tests pin, on the real ``PreviewWindow`` with a real
``ModelController`` per MD model:

* a dropped or opened MD file becomes a model entry of the project and the
  window's five regions switch to its workspace; a crystal loaded beside
  it keeps the crystal workspace exactly as it was;
* a dump without elements shows the read panel in the centre, asks for the
  type map with the reader's evidence, and reads once it is given;
* a run goes through the worker thread and back to slots on the GUI
  thread; a cancel mid-run keeps the finished frames; a request the engine
  refuses is reported, not raised; closing the window during a run
  returns within the wait;
* the frame stepper and View > Frame move the frame; the threshold strip
  re-reads the frame's table (``bulk.at_threshold``, no search) and the
  Elements dock follows; an element clicked is emphasised in the scene;
* the centre switches between the frame and the chosen descriptor's
  figure; the highlight rules are applied on a worker and the count they
  report equals a direct numpy count;
* export, the request as a file (saved, read by the command line, loaded
  back) and the session (a model saved as its file, read options and
  request) round-trip;
* the Model menu follows the model's state, every menu keeps one item per
  mnemonic, no button shares a letter with the menu bar, and no verdict
  word appears in the workspace.

Default platform; no OpenGL context is needed (the 3D view is checked as
data: the scene the controller hands it).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import dispose  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "md"
QUARTZ = HERE / "data" / "crystals" / "quartz_SiO2_cod9013321.cif"

VERDICT_WORDS = ("good", "bad", "poor", "excellent", "acceptable", "correct",
                 "incorrect", "wrong", "reliable", "unreliable", "trustworthy",
                 "untrustworthy", "unusable", "should", "proves", "confirms")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(qapp, condition, timeout_s: float = 30.0) -> bool:
    """Process events until ``condition()`` holds or the time is up."""
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(condition())


def _write_extxyz(path: Path, frames) -> Path:
    with open(path, "w", encoding="ascii") as handle:
        for frame in frames:
            lattice = " ".join(repr(float(x)) for x in frame.box_ang.ravel())
            handle.write(f"{frame.n_atoms}\n")
            handle.write(f'Lattice="{lattice}" '
                         'Properties=species:S:1:pos:R:3 pbc="T T T"\n')
            for symbol, p in zip(frame.elements, frame.cart_ang):
                handle.write(f"{symbol} {float(p[0])!r} {float(p[1])!r} "
                             f"{float(p[2])!r}\n")
    return path


@pytest.fixture(scope="module")
def quartz_model(tmp_path_factory):
    """A 3 x 3 x 3 quartz supercell (243 atoms), six frames, each moved by a
    seeded 0.02 Å Gaussian, as an extended XYZ file."""
    from facet.core import md_model, readers

    base, _ = md_model.supercell_frame(readers.read(QUARTZ), (3, 3, 3))
    rng = np.random.default_rng(11)
    frames = [md_model.frame_from_arrays(
        base.elements, base.cart_ang + rng.normal(0.0, 0.02,
                                                  base.cart_ang.shape),
        box_ang=base.box_ang) for _ in range(6)]
    return _write_extxyz(tmp_path_factory.mktemp("md") / "quartz6.extxyz",
                         frames)


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


def _open(qapp, window, path, read_options=None):
    """List ``path`` as a model and wait for the reader."""
    controllers = window.open_md_models([str(path)], read_options=read_options)
    assert len(controllers) == 1
    controller = controllers[0]
    assert _wait(qapp, lambda: controller.trajectory is not None
                 or controller.problem is not None), "the file was not read"
    index = window.project.index_of(controller.entry)
    window.project.set_active(index)
    window.structure_panel.refresh()
    window._show_entry()
    return controller


def _drawn(qapp, controller, timeout_s: float = 60.0) -> bool:
    """Wait for the frame on show to be drawn with its rules applied."""
    return _wait(qapp, lambda: controller.frame_view is not None
                 and not controller._busy()
                 and controller._highlight_job is not None
                 and controller._highlight_job.done
                 and not controller._redraw_timer.isActive()
                 and controller.view.scene is not None, timeout_s)


def _quartz(qapp, window, quartz_model, analyses=("glass",)):
    controller = _open(qapp, window, quartz_model)
    assert controller.trajectory is not None, controller.problem
    controller.page.panel.formers.set_formers({"Si"})
    controller.page.panel.set_analyses(list(analyses))
    return controller


def _run(qapp, controller, timeout_s=120.0):
    assert controller.run_analysis(), controller.page.panel.problems()
    assert _wait(qapp, lambda: not controller.is_running(), timeout_s), \
        "the run did not end"
    return controller.result


def _crystal_tabs(window):
    tabs = window.crystal_tabs
    return [tabs.tabText(i) for i in range(tabs.count())]


CRYSTAL_TABS = ["Site", "Tools", "Diffraction", "PDF", "EXAFS", "Planes",
                "Styles", "Volume", "Disorder", "Theme"]


# ---------------------------------------------------------------------------
# listing a model, and the crystal beside it
# ---------------------------------------------------------------------------

def test_an_opened_model_becomes_an_entry_with_its_workspace(qapp, window):
    controller = _open(qapp, window, DATA / "glass.extxyz")
    entry = controller.entry
    assert entry.kind == "model" and entry.controller is controller
    assert entry.trajectory is controller.trajectory
    assert entry.summary.n_atoms == 5 and entry.summary.n_frames == 2
    assert entry.name.splitlines() == ["glass.extxyz", "5 atoms · 2 frames"]
    tabs = controller.tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Setup", "Results", "Highlight", "Notes"]
    for widget, stack in zip(controller.widgets, window._stacks()):
        assert stack.currentWidget() is widget
    assert window.sites_dock.windowTitle() == "Elements"
    assert controller.canvas_shown() == "3d"
    assert _drawn(qapp, controller)
    assert controller.elements.elements() == ["Na", "O", "Si"]
    assert "Renderer:" in window.statusBar().currentMessage()
    assert "frame 0 drawn in" in window.statusBar().currentMessage()
    assert "MD model" in window.windowTitle()
    # the crystal side is untouched
    assert _crystal_tabs(window) == CRYSTAL_TABS
    assert not window.model_menu.isEmpty()
    assert window.run_action.isEnabled() is False     # nothing ticked yet


def test_the_crystal_loads_beside_a_model_and_each_keeps_its_state(
        qapp, window):
    controller = _open(qapp, window, DATA / "glass.extxyz")
    window.load(str(QUARTZ))
    assert [e.kind for e in window.project.entries] == ["model", "crystal"]
    assert window.project.current.kind == "crystal"
    assert window.structure is not None and window.scene is not None
    assert window.site_list.count() == len(window.results) > 0
    for stack in window._stacks():
        assert stack.currentIndex() == 0
    assert window.sites_dock.windowTitle() == "Sites"
    # back to the model: its widgets, and the crystal's scene still held
    window.structure_panel.list.setCurrentRow(0)
    qapp.processEvents()
    assert window.project.current is controller.entry
    assert window.column_stack.currentWidget() is controller.column
    assert window.scene is not None
    # crystal-only actions raise nothing with a model selected
    assert window._build_context_menu(0) is None
    assert window._cell_centre() is None
    window._apply_override("site", "Si1", "hid", visible=False)
    snapshot = window._snapshot()
    assert snapshot["overrides"] is None
    window._on_include_anions(True)
    window._on_threshold(0.05)


def test_a_background_models_status_line_is_named_under_a_crystal_view(
        qapp, window):
    """With the crystal current, a model's status line (a frame drawn, a
    read ending) is shown with the model's name and a timeout, never bare
    and permanent as if it were the crystal's; the model on show keeps the
    bare permanent line."""
    controller = _open(qapp, window, DATA / "glass.extxyz")
    assert _drawn(qapp, controller)
    window.load(str(QUARTZ))
    assert window.project.current.kind == "crystal"
    qapp.processEvents()
    controller.statusMessage.emit("frame 0 drawn in 0.34 s (probe)")
    qapp.processEvents()
    assert window.statusBar().currentMessage() == \
        "glass.extxyz: frame 0 drawn in 0.34 s (probe)"
    # back on the model, the line is its own again
    window.structure_panel.list.setCurrentRow(0)
    qapp.processEvents()
    assert window.project.current is controller.entry
    controller.statusMessage.emit("frame 1 drawn in 0.12 s (probe)")
    qapp.processEvents()
    assert window.statusBar().currentMessage() == \
        "frame 1 drawn in 0.12 s (probe)"


def test_a_dump_without_elements_shows_the_read_panel_and_reads_once_given(
        qapp, window):
    controller = _open(qapp, window, DATA / "dump_tri_x.lammpstrj")
    assert controller.trajectory is None
    problem = controller.problem
    assert "type_map" in problem.needs and "masses_from" in problem.needs
    assert controller.canvas_shown() == "read-options"
    assert controller.column.currentIndex() == 0
    assert "reader asks" in controller.pending_label.text()
    assert controller.tabs is None
    assert "not read yet" in controller.entry.name
    panel = controller.read_panel
    assert list(panel.combos) == [1, 2, 3]
    counts = [panel.type_table.item(r, 1).text() for r in range(3)]
    assert counts == ["1", "3", "1"]
    assert panel.type_map() == {}
    assert not panel.read_button.isEnabled()
    panel.set_element(1, "Si")
    panel.set_element(2, "O")
    panel.set_element(3, "Na")
    assert panel.read_button.isEnabled()
    panel.read_button.click()
    assert _wait(qapp, lambda: controller.summary is not None)
    assert controller.trajectory is not None
    assert controller.summary.composition == {"Na": 1, "O": 3, "Si": 1}
    assert controller.entry.read_options["type_map"] == {1: "Si", 2: "O",
                                                         3: "Na"}
    assert controller.canvas_shown() == "3d"
    assert controller.column.currentWidget() is controller.tabs
    assert "5 atoms" in controller.entry.name
    assert _drawn(qapp, controller)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def test_a_glass_run_goes_through_the_worker_and_back(qapp, window,
                                                      quartz_model,
                                                      monkeypatch):
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    from facet.core import md_analysis
    from facet.ui import md_workspace

    seen = {}
    real_analyse = md_analysis.analyse

    def recording_analyse(*args, **kwargs):
        seen["worker"] = QThread.currentThread()
        return real_analyse(*args, **kwargs)

    real_show = md_workspace.ModelController._show_result

    def recording_show(self, result):
        seen["slot"] = QThread.currentThread()
        return real_show(self, result)

    monkeypatch.setattr(md_analysis, "analyse", recording_analyse)
    monkeypatch.setattr(md_workspace.ModelController, "_show_result",
                        recording_show)
    controller = _quartz(qapp, window, quartz_model)
    outcomes = []
    controller.finished.connect(outcomes.append)
    assert window.run_action.isEnabled()
    window.run_action.trigger()
    assert controller.is_running()
    assert window.cancel_action.isEnabled() and not window.run_action.isEnabled()
    # the footer (Run, the progress bar, Cancel) sits under the scroll area
    assert controller.page.cancel_button.isVisibleTo(controller.setup_tab)
    assert not controller.run_analysis()          # a second run is refused
    assert _wait(qapp, lambda: bool(outcomes), 120)
    result = controller.result
    assert outcomes == [result] and result is controller.entry.result
    assert isinstance(result, md_analysis.ModelResult)
    assert not result.cancelled
    cn = result.outputs["glass"].tables["CN Si (BV)"]
    assert cn.mean[list(cn.keys).index(4)] == 1.0
    app_thread = QApplication.instance().thread()
    assert seen["worker"] != app_thread and seen["slot"] == app_thread
    assert all(w.thread() == app_thread for w in QApplication.allWidgets())
    # the results came forward: the tab, the figure in the centre, the notes
    assert controller.current_tab() == "Results"
    assert controller.canvas_shown() == "figure"
    assert controller.browser.result is result
    text = controller.notes_text.toPlainText()
    assert "== glass ==" in text and "== the model, as read ==" in text
    assert window.export_csv_action.isEnabled()
    assert window.export_results_menu.menuAction().isEnabled()
    assert "Run finished" in window.statusBar().currentMessage()
    assert _wait(qapp, lambda: not controller._busy(), 30)


def test_a_cancel_mid_run_keeps_the_finished_frames(qapp, window,
                                                    quartz_model, monkeypatch):
    from facet.core import md_analysis

    real_analyse = md_analysis.analyse

    def slow_analyse(trajectory, request, *, progress=None, cancelled=None):
        def slow(done, total, stage):
            progress(done, total, stage)
            if re.search(r"\bframe\b", stage):
                time.sleep(0.25)
        return real_analyse(trajectory, request, progress=slow,
                            cancelled=cancelled)

    monkeypatch.setattr(md_analysis, "analyse", slow_analyse)
    controller = _quartz(qapp, window, quartz_model)
    assert controller.run_analysis()
    job = controller._run_job
    asked = []

    def on_progress(done, total, stage):
        if re.search(r"\bframe\b", stage) and not asked:
            asked.append(stage)
            window.cancel_action.trigger()

    job.progressed.connect(on_progress)
    assert _wait(qapp, lambda: not controller.is_running(), 120)
    assert asked
    result = controller.result
    assert result is not None and result.cancelled
    assert any("cancelled" in note for note in result.notes)
    assert "cancelled" in window.statusBar().currentMessage()
    assert controller.partial_result is result
    assert not window.show_partial_action.isEnabled()


def test_a_request_the_engine_refuses_is_reported_not_raised(qapp, window,
                                                             quartz_model):
    import dataclasses

    controller = _quartz(qapp, window, quartz_model)
    request = controller.page.request()
    bad = dataclasses.replace(request, formers=frozenset({"O"}))
    outcomes = []
    controller.finished.connect(outcomes.append)
    assert controller.run_analysis(bad)
    assert _wait(qapp, lambda: bool(outcomes), 60)
    failure = outcomes[0]
    assert failure.kind == "request"
    assert any("anions" in m for m in failure.missing)
    assert window.statusBar().currentMessage().startswith("The run needs")
    assert "The run needs" in controller.notes_text.toPlainText()


def _endless(seconds_if_ignored: float | None):
    from facet.core import md_analysis

    def analyse(trajectory, request, *, progress=None, cancelled=None):
        start = time.perf_counter()
        while True:
            time.sleep(0.02)
            if progress is not None:
                progress(0, 10, "frame 1 of 10 (stand-in)")
            elapsed = time.perf_counter() - start
            if seconds_if_ignored is None:
                if cancelled():
                    raise md_analysis.AnalysisCancelled("cancelled")
            elif elapsed > seconds_if_ignored:
                raise md_analysis.AnalysisCancelled("cancelled late")
            if elapsed > 60:
                raise RuntimeError("the stand-in ran a minute")
    return analyse


def test_closing_during_a_run_returns_within_the_wait(qapp, warnings,
                                                      quartz_model,
                                                      monkeypatch):
    from facet.core import md_analysis
    from facet.ui import md_jobs, md_workspace
    from facet.ui.preview import PreviewWindow

    monkeypatch.setattr(md_analysis, "analyse", _endless(None))
    window = PreviewWindow()
    controller = _quartz(qapp, window, quartz_model)
    assert controller.run_analysis()
    job = controller._run_job
    assert _wait(qapp, job.is_running, 10)
    assert window.cancel_action.isEnabled()
    assert not window.run_action.isEnabled()
    assert not controller.page.run_button.isEnabled()
    start = time.perf_counter()
    window.close()
    took = time.perf_counter() - start
    try:
        assert took < md_workspace.CLOSE_WAIT_MS / 1000.0 + 1.0
        assert controller.closing
        assert job.wait(5000)
        assert _wait(qapp, lambda: job.done, 10)
        assert job not in md_jobs.running_jobs()
    finally:
        dispose(window)


def test_work_that_ignores_the_cancel_outlives_the_window(qapp, warnings,
                                                          quartz_model,
                                                          monkeypatch):
    from facet.core import md_analysis
    from facet.ui import md_jobs, md_workspace
    from facet.ui.preview import PreviewWindow

    monkeypatch.setattr(md_analysis, "analyse", _endless(2.0))
    monkeypatch.setattr(md_workspace, "CLOSE_WAIT_MS", 150)
    window = PreviewWindow()
    controller = _quartz(qapp, window, quartz_model)
    assert controller.run_analysis()
    job = controller._run_job
    assert _wait(qapp, job.is_running, 10)
    start = time.perf_counter()
    window.close()
    try:
        assert time.perf_counter() - start < 1.5
        assert job in md_jobs.running_jobs()
        assert _wait(qapp, lambda: job.done, 20)
        assert job not in md_jobs.running_jobs()
    finally:
        dispose(window)


def test_the_last_complete_run_survives_a_cancelled_one(qapp, window,
                                                        quartz_model,
                                                        monkeypatch):
    from facet.core import md_analysis

    controller = _quartz(qapp, window, quartz_model)
    complete = _run(qapp, controller)
    assert controller.complete_result is complete
    real_analyse = md_analysis.analyse

    def cancelling(trajectory, request, *, progress=None, cancelled=None):
        def hook(done, total, stage):
            progress(done, total, stage)
            if re.search(r"\bframe\b", stage):
                controller.cancel()
                time.sleep(0.1)
        return real_analyse(trajectory, request, progress=hook,
                            cancelled=cancelled)

    monkeypatch.setattr(md_analysis, "analyse", cancelling)
    partial = _run(qapp, controller)
    assert partial is complete                     # the complete run stays
    assert controller.partial_result is not None
    assert controller.partial_result.cancelled
    assert window.show_partial_action.isEnabled()
    window.show_partial_action.trigger()
    assert controller.result is controller.partial_result
    assert window.show_complete_action.isEnabled()
    window.show_complete_action.trigger()
    assert controller.result is complete


# ---------------------------------------------------------------------------
# the frame, the strip, the Elements dock
# ---------------------------------------------------------------------------

def test_the_frame_stepper_and_view_frame_move_the_frame(qapp, window,
                                                         quartz_model):
    controller = _quartz(qapp, window, quartz_model)
    assert _drawn(qapp, controller)
    assert controller.frame_view.k == 0
    assert window.frames_menu.menuAction().isEnabled()
    first_scene = controller.view.scene
    controller.step_frame(1)
    assert _wait(qapp, lambda: controller.frame_view.k == 1, 30)
    assert controller.toolbar.frame_box.value() == 1
    assert controller.entry.frame_k == 1
    window.frame_actions["last"].trigger()
    assert _wait(qapp, lambda: controller.frame_view.k == 5, 30)
    assert "Frame " in controller.toolbar.frame_box.prefix()
    assert controller.toolbar.frame_box.suffix() == " of 6"
    window.frame_actions["next"].trigger()               # clamped
    qapp.processEvents()
    assert controller.toolbar.frame_box.value() == 5
    window.frame_actions["first"].trigger()
    assert _wait(qapp, lambda: controller.frame_view.k == 0, 30)
    window.frame_actions["previous"].trigger()
    qapp.processEvents()
    assert controller.toolbar.frame_box.value() == 0
    assert _drawn(qapp, controller)
    assert controller.view.scene is not first_scene
    assert "timestep" in controller.toolbar.frame_note.text() or \
        "bonds" in controller.toolbar.frame_note.text()


def test_the_threshold_strip_re_reads_the_table_and_the_dock_follows(
        qapp, window, quartz_model):
    from facet.core import bulk

    controller = _quartz(qapp, window, quartz_model)
    assert _drawn(qapp, controller)
    fv = controller.frame_view
    strip = controller.strip
    assert strip.table is fv.table                   # no second search
    assert strip.results is not None
    expected = bulk.at_threshold(fv.table, strip.v_bond)
    assert np.array_equal(strip.results.cn, expected.cn)
    strip.set_threshold(0.1)
    assert strip.v_bond == pytest.approx(0.1)
    at = bulk.at_threshold(fv.table, 0.1)
    assert np.array_equal(strip.results.cn, at.cn)
    assert np.allclose(strip.results.bvs_vu, at.bvs_vu, equal_nan=True)
    # the Elements dock lists each element's mean CN at that threshold
    symbols = np.asarray(fv.frame.elements)
    table = controller.elements.table
    rows = {table.item(r, 0).text(): r for r in range(table.rowCount())}
    assert set(rows) == {"O", "Si"}
    for element, r in rows.items():
        mean = float(at.cn[symbols == element].mean())
        assert table.item(r, 3).text() == f"{mean:.2f}"
        assert table.item(r, 1).text() == str(int((symbols == element).sum()))
    assert table.item(rows["Si"], 2).text() == "+4"
    assert table.item(rows["O"], 2).text() == "-2"
    assert "0.1" in controller.elements.note.text()
    # the scene is rebuilt at the new threshold after the pause
    assert _drawn(qapp, controller)
    assert controller.view.scene.v_bond == pytest.approx(0.1)
    assert "v_bond 0.1 v.u." in controller.status_line()
    # the run's threshold is named as the reference once a run exists
    _run(qapp, controller)
    assert strip.analysis_v_bond == pytest.approx(0.075)
    assert any(kind == "reference"
               for _x, _label, kind in strip.stairs_plot.figure.vlines)


def test_clicking_an_element_emphasises_its_atoms(qapp, window, quartz_model):
    from facet.ui.md_highlight import DIM_ATOM_RADIUS

    controller = _quartz(qapp, window, quartz_model)
    assert _drawn(qapp, controller)
    before = controller.view.scene
    radii = np.asarray(before.atom_radius).copy()
    elements = np.asarray(before.atom_element)
    controller.elements.choose("O")
    assert controller.element == "O"
    assert _wait(qapp, lambda: controller.view.scene is not before, 30)
    assert _drawn(qapp, controller)
    after = np.asarray(controller.view.scene.atom_radius)
    assert np.allclose(after[elements == "O"], radii[elements == "O"])
    assert np.allclose(after[elements == "Si"],
                       radii[elements == "Si"] * DIM_ATOM_RADIUS)
    # the same row clicked again shows every atom alike
    item = controller.elements.table.item(
        controller.elements.elements().index("O"), 0)
    scene = controller.view.scene
    controller.elements._on_click(item)
    assert controller.element == ""
    assert _wait(qapp, lambda: controller.view.scene is not scene, 30)
    assert _drawn(qapp, controller)
    assert np.allclose(controller.view.scene.atom_radius, radii)


# ---------------------------------------------------------------------------
# the centre: figure or frame; the highlight rules
# ---------------------------------------------------------------------------

def test_the_centre_switches_between_the_figure_and_the_frame(qapp, window,
                                                              quartz_model):
    controller = _quartz(qapp, window, quartz_model)
    assert _drawn(qapp, controller)
    assert controller.canvas_shown() == "3d"
    controller.show_canvas("figure")                 # no run: stays
    assert controller.canvas_shown() == "3d"
    _run(qapp, controller)
    assert controller.canvas_shown() == "figure"
    assert controller.browser.select("glass", "CN Si (BV)")
    assert controller.browser.view.figure is not None
    assert window.export_figure_action.isEnabled()
    assert window.export_rows_action.isEnabled()
    controller.toolbar.show_3d.click()
    assert controller.canvas_shown() == "3d"
    assert controller.toolbar.show_3d.isChecked()
    assert not window.export_figure_action.isEnabled()
    controller.toolbar.show_figure.click()
    assert controller.canvas_shown() == "figure"
    controller.show_tab("Setup")
    assert controller.canvas_shown() == "3d"
    controller.show_tab("Results")
    assert controller.canvas_shown() == "figure"
    # a descriptor chosen in the tree brings the figure back
    controller.show_tab("Highlight")
    assert controller.canvas_shown() == "3d"
    assert controller.browser.select("glass", "CN O (BV)")
    assert controller.canvas_shown() == "figure"
    assert controller.browser.view.title.text() == "CN O (BV)"
    # and so does a click on the descriptor already current
    controller.show_tab("Highlight")
    assert controller.canvas_shown() == "3d"
    tree = controller.browser.tree
    tree.itemClicked.emit(tree.currentItem(), 0)
    assert controller.canvas_shown() == "figure"
    # the filter hides the names that do not match
    controller.results_page.search.setText("Qn")
    shown = controller.results_page.visible_descriptors()
    assert shown and all("Qn" in name for name in shown)
    controller.results_page.search.setText("")


def test_the_highlight_rules_are_applied_on_a_worker(qapp, window,
                                                     quartz_model, monkeypatch):
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    from facet.ui import md_highlight

    seen = []
    real_apply = md_highlight.apply_rules

    def recording(*args, **kwargs):
        seen.append(QThread.currentThread())
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(md_highlight, "apply_rules", recording)
    controller = _quartz(qapp, window, quartz_model)
    assert _drawn(qapp, controller)
    page = controller.highlight
    assert page.frame_data is not None
    assert page.frame_data.former_cn is not None     # Si is ticked
    # quartz has no non-bridging oxygen: the bridging ones (former CN 2)
    # are the atoms kept, every Si and nothing else is dimmed
    row = page.only
    row.element.setCurrentText("O")
    row.descriptor.setCurrentText("former CN")
    row.sense.setCurrentText("≥")
    row.value.setValue(2)
    before = controller.view.scene
    row.tick.setChecked(True)
    assert _wait(qapp, lambda: controller.view.scene is not before, 60)
    assert _drawn(qapp, controller)
    result = page.result
    fd = controller.frame_data
    keep = (fd.former_cn >= 2) & (fd.elements == "O")
    assert 0 < int(keep.sum()) < fd.n_atoms
    assert result.n_matching == int(keep.sum())
    assert "match" in page.summary.text()
    app_thread = QApplication.instance().thread()
    assert seen and all(t != app_thread for t in seen)
    assert all(w.thread() == app_thread for w in QApplication.allWidgets())
    # the scene carries the rule: the kept atoms keep their radius
    scene = controller.view.scene
    radii = np.asarray(scene.atom_radius)
    sites = np.asarray(scene.atom_site)
    assert np.allclose(radii[keep[sites]], radii[keep[sites]].max())
    assert (radii[~keep[sites]] < radii[keep[sites]].max()).all()
    # View > Highlight… brings the tab forward
    controller.show_tab("Setup")
    window.highlight_action.trigger()
    assert controller.current_tab() == "Highlight"
    assert any(line.startswith("highlight:")
               for line in controller.notes_lines())


# ---------------------------------------------------------------------------
# export, the request as a file, the session
# ---------------------------------------------------------------------------

def test_the_export_writes_every_table_with_its_header(qapp, window,
                                                       quartz_model, tmp_path):
    controller = _quartz(qapp, window, quartz_model)
    _run(qapp, controller)
    job = controller.export_results(tmp_path / "csv")
    assert job is not None
    assert _wait(qapp, lambda: job.done, 60)
    index = tmp_path / "csv" / "index.csv"
    assert index.is_file()
    files = list((tmp_path / "csv").glob("*.csv"))
    assert len(files) > 1
    one = next(f for f in files if f.name != "index.csv")
    assert one.read_text(encoding="utf-8").startswith("#")
    assert "Wrote" in window.statusBar().currentMessage()


def test_the_request_file_round_trips_through_the_command_line_and_back(
        qapp, window, quartz_model, tmp_path):
    import tomllib

    from facet.md import cli

    controller = _quartz(qapp, window, quartz_model)
    panel = controller.page.panel
    panel.field("network", "ring_criterion").set_text("primitive")
    panel.field("network", "ring_max_size").set_text("8")
    panel.field("scattering", "r_window").set_text("Lorch")
    panel.field("scattering", "radiations").set_text("neutron, X-ray")
    panel.field("glass", "cutoffs_ang").set_text("Si-O=2.1")
    panel.field(None, "frame_interval_ps").set_text("0.1")
    panel.field("dynamics", "remove_com_drift").set_text("true")
    panel.frame_range.set_range(1, 5, 2)
    panel.set_analyses(["glass", "rings", "scattering", "msd"])
    gui = panel.request()
    assert gui is not None, panel.problems()
    assert window.save_request_action.isEnabled()
    path = controller.save_request(tmp_path / "request.toml")
    text = path.read_text(encoding="utf-8")
    assert "py -3.11 -m facet.md analyse" in text and "--request" in text
    from_file = cli.request_from_mapping(tomllib.loads(text))
    assert from_file == gui
    assert controller.entry.request_spec["analyses"] == list(gui.analyses)
    # emptied, then filled again from the file: the same request
    panel.set_analyses([])
    panel.field("network", "ring_max_size").set_text("")
    panel.field("glass", "cutoffs_ang").set_text("")
    panel.frame_range.set_range(0, 5, 1)
    assert panel.request() is None
    spec = controller.load_request(path)
    assert spec["frames"] == "1:6:2"
    assert controller.page.preset.currentText() == "Custom"
    again = panel.request()
    assert again is not None, panel.problems()
    assert again == gui
    # a file the command line refuses is refused here, with its reason
    bad = tmp_path / "bad.toml"
    bad.write_text('analyses = ["glass"]\nno_such_key = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no_such_key"):
        controller.load_request(bad)


def test_a_session_saves_a_model_as_its_file_options_and_request(
        qapp, window, warnings, quartz_model, tmp_path):
    from facet.core import exporters
    from facet.ui.preview import PreviewWindow

    controller = _quartz(qapp, window, quartz_model)
    window.load(str(QUARTZ))
    target = tmp_path / "session.json"
    window.save_session(target)
    data = json.loads(target.read_text(encoding="utf-8"))
    kinds = [e.get("kind", "crystal") for e in data["entries"]]
    assert kinds == ["model", "crystal"]
    record = data["entries"][0]
    assert record["path"] == str(quartz_model)
    assert record["request"]["analyses"] == ["glass"]
    assert record["request"]["formers"] == ["Si"]
    assert "result" not in record
    other = PreviewWindow()
    try:
        other.restore_session(exporters.load_session(target))
        assert [e.kind for e in other.project.entries] == ["crystal", "model"]
        restored = other.models[0]
        assert _wait(qapp, lambda: restored.trajectory is not None)
        qapp.processEvents()
        assert restored.page.ticked() == ["glass"]
        assert restored.page.formers() == frozenset({"Si"})
        assert restored.page.preset.currentText() == "Custom"
    finally:
        dispose(other)


# ---------------------------------------------------------------------------
# the Model menu, the presets, the theme
# ---------------------------------------------------------------------------

def test_the_model_menu_follows_the_model_s_state(qapp, window, quartz_model,
                                                  monkeypatch):
    from facet.core import md_analysis

    controller = _open(qapp, window, quartz_model)
    assert window.highlight_action.isEnabled()
    assert window.frames_menu.menuAction().isEnabled()
    assert window.save_request_action.isEnabled()
    assert window.load_request_action.isEnabled()
    assert not window.run_action.isEnabled()         # nothing ticked
    assert not window.export_results_menu.menuAction().isEnabled()
    controller.page.panel.formers.set_formers({"Si"})
    controller.page.panel.set_analyses(["glass"])
    qapp.processEvents()
    assert window.run_action.isEnabled()
    monkeypatch.setattr(md_analysis, "analyse", _endless(None))
    window.run_action.trigger()
    assert window.cancel_action.isEnabled()
    assert not window.run_action.isEnabled()
    assert not window.load_request_action.isEnabled()
    window.cancel_action.trigger()
    assert _wait(qapp, lambda: not controller.is_running(), 30)
    assert not window.cancel_action.isEnabled()
    assert window.run_action.isEnabled()


def test_the_presets_menu_fills_the_setup(qapp, window, quartz_model):
    controller = _open(qapp, window, quartz_model)
    window._fill_presets_menu()
    assert len(window.preset_actions) == 10
    assert all(not a.isChecked() for a in window.preset_actions.values()) \
        or window.preset_actions["Custom"].isChecked()
    window.preset_actions["Silicate"].trigger()
    assert controller.page.preset.currentText() == "Silicate"
    assert "glass" in controller.page.ticked()
    assert controller.page.formers() == frozenset({"Si"})
    assert controller.current_tab() == "Setup"
    window._fill_presets_menu()
    assert window.preset_actions["Silicate"].isChecked()
    assert window.run_action.isEnabled()
    assert "Silicate" in window.statusBar().currentMessage()


def test_a_theme_change_redraws_the_model_with_it(qapp, window, quartz_model):
    from facet.core import theme as theme_mod

    controller = _quartz(qapp, window, quartz_model)
    assert _drawn(qapp, controller)
    dark = theme_mod.PRESETS["Dark"]()
    window._on_theme_structural(dark)
    assert controller.theme is dark
    assert controller.view.theme.name == dark.name
    assert _drawn(qapp, controller)
    assert controller.view.scene.theme.name == dark.name


def test_a_reason_link_reveals_the_field_in_a_dialog(qapp, window,
                                                     quartz_model):
    from facet.ui import md_workspace

    controller = _quartz(qapp, window, quartz_model)
    controller.show_tab("Notes")
    controller.page.fieldRequested.emit("network.ring_max_size")
    qapp.processEvents()
    assert controller.current_tab() == "Setup"
    dialogs = [d for d in controller._field_dialogs if d.isVisible()]
    assert len(dialogs) == 1
    dialog = dialogs[0]
    assert isinstance(dialog, md_workspace._FieldDialog)
    group = dialog.group
    assert ("network", "ring_max_size") in group.fields
    dialog.close()
    qapp.processEvents()
    assert group.parentWidget() is controller.page.panel
    # a field the page itself shows needs no dialog
    controller.page.fieldRequested.emit("frames")
    qapp.processEvents()
    assert not [d for d in controller._field_dialogs if d.isVisible()]


# ---------------------------------------------------------------------------
# wording and mnemonics
# ---------------------------------------------------------------------------

def _texts(widget) -> list[str]:
    from PySide6.QtWidgets import (QAbstractButton, QComboBox, QGroupBox,
                                   QLabel, QLineEdit, QMenu, QTableWidget,
                                   QWidget)

    out = []
    for w in [widget] + widget.findChildren(QWidget):
        out.append(w.toolTip())
        if isinstance(w, QLabel):
            out.append(w.text())
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QLineEdit):
            out.append(w.placeholderText())
        elif isinstance(w, QGroupBox):
            out.append(w.title())
        elif isinstance(w, QComboBox):
            out.extend(w.itemText(i) for i in range(w.count()))
        elif isinstance(w, QTableWidget):
            for r in range(w.rowCount()):
                for c in range(w.columnCount()):
                    item = w.item(r, c)
                    if item is not None:
                        out.append(item.text())
        elif isinstance(w, QMenu):
            out.extend(a.text() for a in w.actions())
    return out


def test_no_verdict_word_appears_in_the_workspace(qapp, window, quartz_model):
    from PySide6.QtWidgets import QMenu

    pattern = re.compile(r"\b(" + "|".join(VERDICT_WORDS) + r")\b",
                         re.IGNORECASE)
    controllers = [_open(qapp, window, DATA / "dump_tri_x.lammpstrj"),
                   _quartz(qapp, window, quartz_model)]
    _run(qapp, controllers[1])
    assert _drawn(qapp, controllers[1])
    window._fill_presets_menu()
    found = {}
    texts = []
    for controller in controllers:
        for widget in controller.widgets:
            texts += _texts(widget)
        texts += controller.notes_lines()
    for menu in window.model_menu.findChildren(QMenu) + [window.model_menu,
                                                         window.frames_menu]:
        for action in menu.actions():
            texts += [action.text(), action.toolTip()]
    texts += [window.highlight_action.text(), window.highlight_action.toolTip(),
              window.statusBar().currentMessage()]
    for text in texts:
        for match in pattern.finditer(text or ""):
            found.setdefault(match.group(0).lower(), text[:160])
    assert not found, found


def _mnemonic(text: str) -> str | None:
    match = re.search(r"&(\w)", text or "")
    return match.group(1).lower() if match else None


def test_no_button_of_the_workspace_shares_a_mnemonic_with_the_menu_bar(
        qapp, window, quartz_model):
    """Alt+F opens the File menu, never a button of a page: the menu bar's
    letters (File, Edit, View, Model, Help) are kept off every visible
    button, group box and tab of the workspace."""
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QTabWidget

    controllers = [_open(qapp, window, DATA / "dump_tri_x.lammpstrj"),
                   _quartz(qapp, window, quartz_model)]
    bar = {_mnemonic(a.text()): a.text() for a in window.menuBar().actions()
           if _mnemonic(a.text())}
    assert set(bar) == {"f", "e", "v", "m", "h"}
    window.show()
    for controller in controllers:
        window.project.set_active(window.project.index_of(controller.entry))
        window._show_entry()
        qapp.processEvents()
        shared = {}
        for widget in controller.widgets:
            for w in widget.findChildren(QAbstractButton) + \
                    widget.findChildren(QGroupBox):
                if not w.isVisibleTo(window):
                    continue
                text = w.text() if isinstance(w, QAbstractButton) else w.title()
                letter = _mnemonic(text)
                if letter in bar:
                    shared[text] = bar[letter]
            for tabs in widget.findChildren(QTabWidget):
                for i in range(tabs.count()):
                    letter = _mnemonic(tabs.tabText(i))
                    if letter in bar:
                        shared[tabs.tabText(i)] = bar[letter]
        assert not shared, shared


def test_every_menu_keeps_one_item_per_mnemonic_with_a_model_shown(
        qapp, window, quartz_model):
    from collections import Counter

    from PySide6.QtWidgets import QMenu

    _quartz(qapp, window, quartz_model)
    window._fill_presets_menu()

    def check(menu, path):
        letters = Counter()
        for action in menu.actions():
            if action.isSeparator():
                continue
            text = action.text()
            if "&" in text and text.index("&") + 1 < len(text):
                letters[text[text.index("&") + 1].lower()] += 1
            if isinstance(action.menu(), QMenu):
                check(action.menu(), f"{path} > {text.replace('&', '')}")
        duplicated = {k: v for k, v in letters.items() if v > 1}
        assert not duplicated, f"{path}: {duplicated}"

    for action in window.menuBar().actions():
        if action.menu() is not None:
            check(action.menu(), action.text().replace("&", ""))
    # the bar's action is held while its menu is read (PySide6 invalidates
    # the menu's wrapper once the action's wrapper is collected)
    for action in window.menuBar().actions():
        rows = [a for a in action.menu().actions() if not a.isSeparator()]
        assert len(rows) <= 12, (action.text(), len(rows))


def test_a_model_shown_widens_the_column_and_each_region_sizes_to_its_page(
        qapp, window, quartz_model):
    """A stacked region used to ask for the largest of every page's
    minimum: the crystal toolbar's row capped the right column at 400 px
    with a model shown (the Setup tree elided its names, the Highlight
    rules clipped), and the figure page raised the window's minimum height
    above a 768 px screen with a crystal shown."""
    window.load(str(QUARTZ))
    window.resize(1440, 900)
    window.show()
    qapp.processEvents()
    crystal_min = window.minimumSizeHint()
    assert crystal_min.height() <= 730 and crystal_min.width() <= 1440
    assert window.column_stack.width() < window.MODEL_COLUMN
    controller = _quartz(qapp, window, quartz_model)
    qapp.processEvents()
    assert window.column_stack.width() >= window.MODEL_COLUMN - 10
    assert window.toolbar_stack.minimumSizeHint().width() \
        == controller.toolbar.minimumSizeHint().width()
    _run(qapp, controller)
    assert controller.canvas_shown() == "figure"
    assert window.minimumSizeHint().height() <= 730
    # back to the crystal: its own minimum, as before
    window.structure_panel.list.setCurrentRow(0)
    qapp.processEvents()
    assert window.project.current.kind == "crystal"
    assert window.minimumSizeHint() == crystal_min


def test_removing_the_last_structure_empties_the_viewport(qapp, window,
                                                          quartz_model):
    """Removing the only structure clears the view instead of raising.

    `_on_remove` hands the viewport `set_scene(None)`, and the OpenGL
    renderer used to read `scene.n_bonds` off that None -- an AttributeError
    on every remove-of-the-last-structure wherever a GL context exists. The
    painter tier never minded, which is why offscreen runs stayed green while
    the deep-scan sweep failed on the GPU in every scenario that ended by
    emptying the window.
    """
    controller = _quartz(qapp, window, quartz_model)
    assert _drawn(qapp, controller)
    assert len(window.project.entries) == 1
    window._on_remove(0)                      # must not raise, on any tier
    qapp.processEvents()
    assert len(window.project.entries) == 0
    assert window.scene is None
    assert window.view.scene is None


def test_removing_the_model_shuts_its_work_and_frees_its_pages(qapp, window,
                                                               quartz_model):
    controller = _quartz(qapp, window, quartz_model)
    assert _drawn(qapp, controller)
    assert all(stack.count() == 2 for stack in window._stacks())
    row = window.project.index_of(controller.entry)
    window._on_remove(row)
    qapp.processEvents()
    assert controller.closing
    assert window.project.models == []
    assert all(stack.count() == 1 for stack in window._stacks())
    assert window.sites_dock.windowTitle() == "Sites"
    assert not window.highlight_action.isEnabled()
