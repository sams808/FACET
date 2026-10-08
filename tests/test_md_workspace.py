"""The Model window (facet.ui.md_workspace), its worker and its setup.

What these tests pin:

* every MD fixture opens a Model window, never an error box or silence: a
  file the reader refuses for want of a type map, a topology or a box
  opens the read panel asking for exactly that (and nothing else), with the
  reader's own evidence per type and no element filled in; a file nothing
  reads is shown with the reader's reason;
* the setup lists the cations present as formers with none ticked, keeps
  an analysis whose inputs are missing disabled with what it needs, and
  reaches every option group of md_analysis.AnalysisRequest;
* a run goes through the worker thread, its results reach slots on the GUI
  thread, and no widget lives on another thread; a cancel mid-run gives
  the finished part with the engine's note; closing a window during a run
  returns within its wait, and work that ignores the cancel is kept alive
  until it ends;
* no verdict word appears in the window, and each menu's mnemonics differ;
* the text of the read and setup pages stays readable under the operating
  system's dark palette (measured from the drawn pixels), and the window
  opens inside its screen's available area, frame included.

Offscreen; no OpenGL context is created (the 3D view is checked as data).
"""
from __future__ import annotations

import os
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


def _measured_sq(folder: Path) -> str:
    """A two-column S(Q) file (test values, not a measurement)."""
    path = folder / "sq.dat"
    q = np.arange(0.5, 15.0, 0.05)
    path.write_text("\n".join(f"{x:.3f} {1.0 + 0.1 * np.sin(x):.6f}"
                              for x in q) + "\n", encoding="ascii")
    return str(path)


def _open(qapp, path, **kwargs):
    from facet.ui.md_workspace import ModelWindow

    window = ModelWindow(str(path), **kwargs)
    assert _wait(qapp, lambda: window.summary is not None
                 or window.problem is not None), "the file was not read"
    return window


def _quartz_window(qapp, quartz_model, analyses=("glass",)):
    window = _open(qapp, quartz_model)
    assert window.trajectory is not None, window.problem
    window.setup.formers.set_formers({"Si"})
    window.setup.set_analyses(list(analyses))
    return window


def _run(qapp, window, timeout_s=120.0):
    assert window.run_analysis(), window.setup.problems()
    assert _wait(qapp, lambda: not window.is_running(), timeout_s), \
        "the run did not end"
    return window.result


# ---------------------------------------------------------------------------
# reading: each fixture opens, and the reader's needs are asked for
# ---------------------------------------------------------------------------

def test_a_dump_without_elements_asks_for_a_type_map_with_its_evidence(qapp):
    window = _open(qapp, DATA / "dump_tri_x.lammpstrj")
    try:
        assert window.trajectory is None
        problem = window.problem
        assert "type_map" in problem.needs and "masses_from" in problem.needs
        assert "topology" not in problem.needs
        assert "box_from" not in problem.needs
        panel = window.read_panel
        assert window.pages.currentWidget() is window.read_page
        assert not panel.type_box.isHidden()
        assert list(panel.combos) == [1, 2, 3]
        # the atoms of each type in frame 0, counted from the file
        counts = [panel.type_table.item(r, 1).text() for r in range(3)]
        assert counts == ["1", "3", "1"]
        # the reader's own words per type, and no element filled in
        assert panel.type_table.item(0, 2).text() == "no source names it"
        assert panel.type_map() == {}
        assert not panel.read_button.isEnabled()
        panel.set_element(1, "Si")
        panel.set_element(2, "O")
        assert not panel.read_button.isEnabled()
        assert "type 3" in panel.reason.text()
        panel.set_element(3, "Xx")
        assert not panel.read_button.isEnabled()
        panel.set_element(3, "Na")
        assert panel.read_button.isEnabled()
        panel.read_button.click()
        assert _wait(qapp, lambda: window.summary is not None)
        assert window.trajectory is not None
        assert window.summary.composition == {"Na": 1, "O": 3, "Si": 1}
        assert window.summary.read_options["type_map"] == {1: "Si", 2: "O",
                                                           3: "Na"}
        assert window.pages.currentWidget() is window.workspace
    finally:
        dispose(window)


def test_a_dump_reads_with_masses_from_its_data_file(qapp):
    window = _open(qapp, DATA / "dump_tri_x.lammpstrj")
    try:
        window.read_panel.set_option("masses_from",
                                     str(DATA / "data_atomic.data"))
        assert window.read_panel.read_button.isEnabled()
        window.read_panel.read_button.click()
        assert _wait(qapp, lambda: window.summary is not None)
        assert window.summary.type_map_source == "data-file masses"
        assert window.summary.composition == {"Na": 1, "O": 3, "Si": 1}
    finally:
        dispose(window)


def test_an_extended_xyz_opens_straight_to_the_setup(qapp):
    window = _open(qapp, DATA / "glass.extxyz")
    try:
        assert window.trajectory is not None
        assert window.setup is not None
        assert window.pages.currentWidget() is window.workspace
        assert window.summary.n_frames == 2
    finally:
        dispose(window)


def test_a_data_file_opens_and_one_frame_leaves_the_dynamics_out(qapp):
    window = _open(qapp, DATA / "data_atomic.data")
    try:
        assert window.trajectory is not None
        assert window.summary.n_frames == 1
        setup = window.setup
        setup.formers.set_formers({"Si"})
        setup.refresh()
        for name in ("msd", "vacf", "bond-lifetimes"):
            assert not setup.box(name).isEnabled()
            assert "two or more chosen frames" in setup.reason_text(name)
        assert setup.box("glass").isEnabled()
    finally:
        dispose(window)


def test_a_dcd_asks_for_its_topology_and_nothing_else(qapp):
    window = _open(qapp, DATA / "lammps_formats" / "traj.dcd")
    try:
        assert window.trajectory is None
        problem = window.problem
        assert "topology" in problem.needs
        assert "box_from" not in problem.needs
        panel = window.read_panel
        # the type map of a DCD applies to its topology's types: no table,
        # and the topology is what is asked for
        assert panel.type_box.isHidden()
        assert panel.asked == ("topology",)
        assert "type_map" in panel.fields       # one click away
        assert not panel.read_button.isEnabled()
        panel.set_option("topology", str(DATA / "lammps_formats" /
                                         "topology.data"))
        assert panel.read_button.isEnabled()
        panel.read_button.click()
        assert _wait(qapp, lambda: window.summary is not None)
        assert window.summary.composition == {"Na": 3, "O": 6, "Si": 3}
    finally:
        dispose(window)


def test_a_plain_xyz_asks_for_a_box(qapp):
    window = _open(qapp, DATA / "recognition" / "nvt.xyz")
    try:
        assert window.trajectory is None
        assert window.problem.needs == ("box_from",)
        panel = window.read_panel
        assert panel.type_box.isHidden()
        panel.set_option("box_from", str(DATA / "recognition" / "nvt.data"))
        panel.read_button.click()
        assert _wait(qapp, lambda: window.summary is not None
                     or window.problem.read_options.get("box_from"))
        assert window.trajectory is not None, window.problem.message
    finally:
        dispose(window)


def test_labels_are_asked_for_by_label_with_the_reader_s_words(qapp):
    window = _open(qapp, DATA / "HISTORY")
    try:
        panel = window.read_panel
        assert list(panel.combos) == ["O_b", "O_nb"]
        assert [panel.type_table.item(r, 1).text() for r in range(2)] == \
            ["2", "1"]
        assert panel.type_table.item(1, 2).text() == \
            "'O_nb' (mass 15.9994 amu in the file): not an element symbol"
        assert panel.type_table.item(0, 2).text().startswith("'O_b' (mass")
    finally:
        dispose(window)


def test_a_file_no_reader_takes_is_shown_with_its_reason(qapp):
    window = _open(qapp, DATA / "recognition" / "NS2-1.cell")
    try:
        problem = window.problem
        assert not problem.can_supply
        panel = window.read_panel
        assert window.pages.currentWidget() is window.read_page
        assert "cell file" in panel.message.text()
        assert panel.read_button.isHidden()
        assert not panel.other_button.isHidden()
    finally:
        dispose(window)


def test_the_diagnosis_keeps_the_reader_s_first_map(qapp):
    """Qt-free: a message offering the types by number and by label asks
    for one map (the numbers), with the labels as evidence."""
    from facet.ui import md_jobs

    problem = md_jobs.open_model(
        str(DATA / "recognition" / "labels_element.lammpstrj"), {})
    assert problem.type_keys == (1, 3, 4)
    assert problem.keys_complete
    assert "'Si_t'" in problem.evidence[1]
    assert problem.counts == {1: 4, 3: 6, 4: 4}
    cut = md_jobs.open_model(str(DATA / "binary" / "lammps_style.nc"), {})
    assert cut.type_keys == (1, 2) and not cut.keys_complete


# ---------------------------------------------------------------------------
# the setup
# ---------------------------------------------------------------------------

def test_the_former_picker_lists_the_cations_and_ticks_none(qapp):
    from facet.core import md_analysis as ma

    window = _open(qapp, DATA / "glass.extxyz")
    try:
        setup = window.setup
        assert setup.formers.cations() == ("Na", "Si")
        assert setup.formers.ticked() == frozenset()
        assert setup.formers.formers() is None
        assert not setup.box("glass").isEnabled()
        assert "network formers" in setup.reason_text("glass")
        assert setup.request() is None
        setup.formers.none_box.setChecked(True)
        setup.refresh()
        assert setup.formers.formers() == ma.NO_FORMERS
        assert setup.box("glass").isEnabled()
        setup.formers.set_formers({"Si"})
        setup.refresh()
        assert not setup.formers.none_box.isChecked()
        setup.set_analyses(["glass"])
        request = setup.request()
        assert request is not None
        assert request.formers == frozenset({"Si"})
        assert request.analyses == ("glass",)
    finally:
        dispose(window)


def test_the_oxidation_states_start_common_and_a_change_is_the_user_s(qapp):
    window = _open(qapp, DATA / "glass.extxyz")
    try:
        table = window.setup.oxidation
        assert table.states() == {"Na": 1, "O": -2, "Si": 4}
        assert table.item(0, 3).text().startswith("common")
        table.set_state("Na", 2)
        assert table.item(0, 3).text() == "user"
        assert table.overrides() == {"Na": 2}
        window.setup.formers.set_formers({"Si"})
        window.setup.set_analyses(["glass"])
        assert window.setup.request().ox_overrides == {"Na": 2}
        table.edits["Na"].setText("")
        assert window.setup.request() is None
        assert any("no oxidation state for Na" in p
                   for p in window.setup.problems())
    finally:
        dispose(window)


def test_the_frame_range_counts_what_it_selects(qapp, quartz_model):
    window = _open(qapp, quartz_model)
    try:
        frames = window.setup.frame_range
        assert frames.count() == 6
        frames.set_range(1, 5, 2)
        assert frames.frames() == slice(1, 6, 2)
        assert frames.count() == len(range(6)[1:6:2]) == 3
        assert "3 of 6 frame(s): 1, 3, 5" in frames.note.text()
    finally:
        dispose(window)


def test_every_option_group_of_the_request_is_reachable(qapp):
    import dataclasses

    from facet.core import md_analysis as ma
    from facet.ui.md_dialogs import group_class

    window = _open(qapp, DATA / "glass.extxyz")
    try:
        setup = window.setup
        fields = setup.all_fields()
        for group in ("glass", "scattering", "network", "order", "voids",
                      "nmr", "exafs", "feff", "dynamics"):
            for f in dataclasses.fields(group_class(group)):
                key = f"{group}.{f.name}"
                if (group, f.name) in fields:
                    continue
                assert setup.extra(key) is not None, key
        for name in ("params", "bridging_anions", "timestep_fs",
                     "frame_interval_ps", "temperature_k", "charges_e"):
            assert (None, name) in fields, name
        for name in ma.ANALYSES:
            assert setup.box(name) is not None
    finally:
        dispose(window)


def test_typed_options_reach_the_request(qapp, quartz_model, tmp_path):
    window = _open(qapp, quartz_model)
    try:
        setup = window.setup
        setup.formers.set_formers({"Si"})
        assert not setup.box("rings").isEnabled()
        assert "network.ring_criterion" in setup.reason_text("rings")
        setup.field("scattering", "r_window").set_text("Lorch")
        setup.field("scattering", "radiations").set_text("neutron, X-ray")
        setup.field("network", "ring_criterion").set_text("primitive")
        setup.field("network", "ring_max_size").set_text("8")
        setup.field("voids", "radii").set_text("vdw")
        setup.field("voids", "probe_radius_ang").set_text("0.5")
        setup.field("voids", "grid_spacing_ang").set_text("0.5")
        setup.field("glass", "cutoffs_ang").set_text("Si-O=2.1")
        setup.field(None, "frame_interval_ps").set_text("0.1")
        setup.field("dynamics", "fit_t_min_ps").set_text("0.1")
        setup.field("dynamics", "fit_t_max_ps").set_text("0.4")
        setup.field("exafs", "absorber").set_text("Si")
        setup.extra("scattering.measured").add_curve(
            _measured_sq(tmp_path), "neutron", "S(Q)")
        setup.set_analyses(["glass", "scattering", "scattering-comparison",
                            "rings", "free-volume", "msd", "exafs"])
        request = setup.request()
        assert request is not None, setup.problems()
        assert set(request.analyses) == {"glass", "scattering",
                                         "scattering-comparison", "rings",
                                         "free-volume", "msd", "exafs"}
        assert request.scattering.r_window == "Lorch"
        assert set(request.scattering.radiations) == {"neutron", "X-ray"}
        assert request.scattering.measured[0].function == "S(Q)"
        assert request.network.ring_criterion == "primitive"
        assert request.network.ring_max_size == 8
        assert request.voids.radii == "vdw"
        assert request.voids.probe_radius_ang == 0.5
        assert request.glass.cutoffs_ang == {("Si", "O"): 2.1}
        assert request.frame_interval_ps == 0.1
        assert request.exafs.absorber == "Si"
        # method choices left blank keep the engine's defaults
        assert request.glass.rdf_dr_ang == \
            type(request.glass)().rdf_dr_ang
        # a value that does not read disables its group, and says why
        setup.field("network", "ring_max_size").set_text("eight")
        setup.refresh()
        assert not setup.box("rings").isEnabled()
        assert "network.ring_max_size" in setup.groups["md_network"] \
            .errors.text()
        assert "rings" not in setup.request().analyses
    finally:
        dispose(window)


def test_scattering_comparison_runs_with_the_scattering_it_reads(qapp,
                                                                  quartz_model,
                                                                  tmp_path):
    window = _open(qapp, quartz_model)
    try:
        setup = window.setup
        setup.field("scattering", "r_window").set_text("none")
        setup.field("scattering", "radiations").set_text("neutron")
        setup.extra("scattering.measured").add_curve(
            _measured_sq(tmp_path), "neutron", "S(Q)")
        setup.set_analyses(["scattering-comparison"])
        request = setup.request()
        assert request is not None
        assert request.analyses == ("scattering", "scattering-comparison")
        assert "runs with scattering" in setup.reason_text(
            "scattering-comparison")
    finally:
        dispose(window)


# ---------------------------------------------------------------------------
# the run, on the worker thread
# ---------------------------------------------------------------------------

def test_a_glass_run_goes_through_the_worker_and_back(qapp, quartz_model,
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

    real_show = md_workspace.ModelWindow._show_result

    def recording_show(self, result):
        seen["slot"] = QThread.currentThread()
        return real_show(self, result)

    monkeypatch.setattr(md_analysis, "analyse", recording_analyse)
    monkeypatch.setattr(md_workspace.ModelWindow, "_show_result",
                        recording_show)
    window = _quartz_window(qapp, quartz_model)
    outcomes = []
    window.finished.connect(outcomes.append)
    progress = []
    try:
        assert window.run_analysis()
        window._run_job.progressed.connect(
            lambda d, t, s: progress.append((d, t)))
        assert _wait(qapp, lambda: bool(outcomes), 120)
        result = window.result
        assert outcomes == [result]
        assert isinstance(result, md_analysis.ModelResult)
        assert not result.cancelled
        output = result.outputs["glass"]
        assert output.ok
        cn = output.tables["CN Si (BV)"]
        assert cn.mean[list(cn.keys).index(4)] == 1.0
        # the engine ran on a thread of its own; the result came back to
        # the GUI thread
        app_thread = QApplication.instance().thread()
        assert seen["worker"] != app_thread
        assert seen["slot"] == app_thread
        assert all(w.thread() == app_thread
                   for w in QApplication.allWidgets())
        # progress never went back
        done = [d for d, _ in progress]
        assert done == sorted(done)
        # the notes tab holds the run's provenance and the model's summary
        text = window.notes_text.toPlainText()
        assert "== glass ==" in text and "== the model, as read ==" in text
        assert window.export_csv_action.isEnabled()
        assert _wait(qapp, lambda: not window._busy(), 30)
    finally:
        dispose(window)


def test_a_cancel_mid_run_shows_the_finished_frames_with_the_note(
        qapp, quartz_model, monkeypatch):
    from facet.core import md_analysis

    real_analyse = md_analysis.analyse

    def slow_analyse(trajectory, request, *, progress=None, cancelled=None):
        def slow(done, total, stage):
            progress(done, total, stage)
            if re.search(r"\bframe\b", stage):
                time.sleep(0.25)    # time for the GUI thread to cancel
        return real_analyse(trajectory, request, progress=slow,
                            cancelled=cancelled)

    monkeypatch.setattr(md_analysis, "analyse", slow_analyse)
    window = _quartz_window(qapp, quartz_model)
    try:
        assert window.run_analysis()
        job = window._run_job
        asked = []

        def on_progress(done, total, stage):
            if re.search(r"\bframe\b", stage) and not asked:
                asked.append(stage)
                window.cancel()

        job.progressed.connect(on_progress)
        assert _wait(qapp, lambda: not window.is_running(), 120)
        assert asked
        result = window.result
        assert result is not None and result.cancelled
        assert any("cancelled" in note for note in result.notes)
        assert "cancelled" in window.stage_label.text()
    finally:
        dispose(window)


def test_a_request_the_engine_refuses_is_reported_not_raised(qapp,
                                                             quartz_model):
    import dataclasses

    window = _quartz_window(qapp, quartz_model)
    try:
        request = window.setup.request()
        # formers that are anions of this model: checked against the first
        # frame, refused with a RequestError on the worker
        bad = dataclasses.replace(request, formers=frozenset({"O"}))
        outcomes = []
        window.finished.connect(outcomes.append)
        assert window.run_analysis(bad)
        assert _wait(qapp, lambda: bool(outcomes), 60)
        failure = outcomes[0]
        assert failure.kind == "request"
        assert any("anions" in m for m in failure.missing)
        assert window.stage_label.text().startswith("The run needs")
    finally:
        dispose(window)


def _endless(seconds_if_ignored: float | None):
    """A stand-in for md_analysis.analyse that runs until cancelled (or,
    with ``seconds_if_ignored``, ignores the cancel for that long)."""
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


def test_closing_during_a_run_returns_and_stops_the_thread(qapp, quartz_model,
                                                           monkeypatch):
    from facet.core import md_analysis
    from facet.ui import md_jobs, md_workspace

    monkeypatch.setattr(md_analysis, "analyse", _endless(None))
    window = _quartz_window(qapp, quartz_model)
    assert window.run_analysis()
    job = window._run_job
    assert _wait(qapp, job.is_running, 10)
    # while it runs: Cancel offered, Run not, and a second run refused
    assert window.cancel_action.isEnabled()
    assert not window.run_action.isEnabled()
    assert not window.setup.run_button.isEnabled()
    assert not window.run_analysis()
    start = time.perf_counter()
    window.close()
    took = time.perf_counter() - start
    try:
        assert took < md_workspace.CLOSE_WAIT_MS / 1000.0 + 1.0
        assert job.wait(5000)
        assert _wait(qapp, lambda: job.done, 10)
        assert job not in md_jobs.running_jobs()
    finally:
        dispose(window)


def test_work_that_ignores_the_cancel_outlives_its_window(qapp, quartz_model,
                                                          monkeypatch):
    from facet.core import md_analysis
    from facet.ui import md_jobs, md_workspace

    monkeypatch.setattr(md_analysis, "analyse", _endless(2.0))
    monkeypatch.setattr(md_workspace, "CLOSE_WAIT_MS", 150)
    window = _quartz_window(qapp, quartz_model)
    assert window.run_analysis()
    job = window._run_job
    assert _wait(qapp, job.is_running, 10)
    start = time.perf_counter()
    window.close()
    try:
        assert time.perf_counter() - start < 1.5
        # still running, and held so that its QThread is not destroyed
        assert job in md_jobs.running_jobs()
        assert _wait(qapp, lambda: job.done, 20)
        assert job not in md_jobs.running_jobs()
    finally:
        dispose(window)


def test_the_export_writes_every_table_with_its_header(qapp, quartz_model,
                                                       tmp_path):
    window = _quartz_window(qapp, quartz_model)
    try:
        _run(qapp, window)
        job = window.export_results(tmp_path / "csv")
        assert job is not None
        assert _wait(qapp, lambda: job.done, 60)
        index = tmp_path / "csv" / "index.csv"
        assert index.is_file()
        head = index.read_text(encoding="utf-8").splitlines()[:5]
        assert all(line.startswith("#") for line in head)
        assert any("FACET" in line for line in head)
        assert "Wrote" in window.stage_label.text()
    finally:
        dispose(window)


# ---------------------------------------------------------------------------
# the 3D view of a frame
# ---------------------------------------------------------------------------

def test_the_frame_view_is_built_from_the_bulk_bonds(qapp, quartz_model):
    window = _quartz_window(qapp, quartz_model)
    try:
        assert _wait(qapp, lambda: window.frame_view is not None, 30)
        view = window.frame_view
        assert view.table is not None and view.n_bonds > 0
        # every Si has four O in quartz: 81 Si, 324 bonds
        assert view.n_bonds == 324
        if view.scene is not None:
            assert window.frame_tab.view.scene is view.scene
            assert view.scene.n_atoms >= 243
    finally:
        dispose(window)


def test_the_painter_tier_draws_atoms_only_above_the_limit(qapp,
                                                           quartz_model,
                                                           monkeypatch):
    from facet.ui import md_workspace

    monkeypatch.setattr(md_workspace, "ATOMS_ONLY_ABOVE", 100)
    window = _quartz_window(qapp, quartz_model)
    try:
        assert window.frame_tab.painter_tier()   # offscreen: no GL context
        assert _wait(qapp, lambda: window.frame_view is not None, 30)
        assert window.frame_view.table is None
        assert "without bonds" in window.frame_tab.note.text()
        assert "QPainter" in window.frame_tab.note.text()
    finally:
        dispose(window)


def test_the_threshold_panel_gets_the_frame_and_moves_the_drawn_bonds(
        qapp, quartz_model, monkeypatch):
    """With a stand-in md_threshold module: the window hands the panel the
    frame, its states and the parameter set, and a threshold the panel
    emits restyles the scene without a search."""
    import sys
    import types

    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QWidget

    calls = []

    class ThresholdPanel(QWidget):
        thresholdChanged = Signal(float)

        def __init__(self, parent=None, *, theme=None):
            super().__init__(parent)

        def set_frame(self, frame, ox_atom, params, **kwargs):
            calls.append((frame, np.asarray(ox_atom), params, kwargs))

    stub = types.ModuleType("facet.ui.md_threshold")
    stub.ThresholdPanel = ThresholdPanel
    monkeypatch.setitem(sys.modules, "facet.ui.md_threshold", stub)
    window = _quartz_window(qapp, quartz_model)
    try:
        assert isinstance(window.threshold_panel, ThresholdPanel)
        assert _wait(qapp, lambda: bool(calls), 30)
        frame, ox_atom, params, kwargs = calls[0]
        assert frame.n_atoms == 243
        assert sorted(set(ox_atom.tolist())) == [-2, 4]
        # the table the 3D view was built from, so the panel searches nothing
        assert kwargs["table"] is window.frame_view.table
        assert kwargs["analysis_v_bond_vu"] is None       # no run yet
        _run(qapp, window)
        assert _wait(qapp, lambda: len(calls) >= 2, 10)
        assert calls[-1][3]["analysis_v_bond_vu"] == 0.075
        restyled = []
        monkeypatch.setattr(window.frame_tab.view, "set_threshold",
                            restyled.append)
        window.threshold_panel.thresholdChanged.emit(0.2)
        assert restyled == [0.2]
    finally:
        dispose(window)


def test_a_list_of_files_opens_as_one_model(qapp):
    series = sorted((DATA / "recognition" / "series").glob("dump.*.lammpstrj"),
                    key=lambda p: int(p.name.split(".")[1]))
    from facet.ui.md_workspace import ModelWindow

    window = ModelWindow([str(p) for p in series])
    try:
        assert _wait(qapp, lambda: window.problem is not None
                     or window.summary is not None)
        panel = window.read_panel
        assert list(panel.combos) == [1, 2, 3]
        for key, symbol in ((1, "Si"), (2, "O"), (3, "Na")):
            panel.set_element(key, symbol)
        panel.read_button.click()
        assert _wait(qapp, lambda: window.summary is not None)
        assert window.summary.n_frames == len(series)
    finally:
        dispose(window)


def test_a_theme_reaches_the_window_s_own_views(qapp, quartz_model):
    from facet.core import theme as theme_mod

    window = _quartz_window(qapp, quartz_model)
    try:
        assert _wait(qapp, lambda: window.frame_view is not None, 30)
        dark = theme_mod.dark()
        window.apply_theme(dark)
        assert window.theme is dark
        assert window.frame_tab.view.theme is dark
    finally:
        dispose(window)


# ---------------------------------------------------------------------------
# the window as a whole
# ---------------------------------------------------------------------------

def test_open_model_window_keeps_it_and_it_has_no_qt_parent(qapp):
    from PySide6.QtWidgets import QMainWindow

    from facet.ui import md_workspace

    opener = QMainWindow()
    opener._model_windows = []
    window = md_workspace.open_model_window(str(DATA / "glass.extxyz"),
                                            opener)
    try:
        assert window.parent() is None
        assert opener.findChildren(md_workspace.ModelWindow) == []
        assert window in md_workspace.model_windows()
        assert window in opener._model_windows
        assert _wait(qapp, lambda: window.summary is not None)
        window.close()
        assert window not in md_workspace.model_windows()
        assert window not in opener._model_windows
    finally:
        dispose(window, opener)


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


def test_no_verdict_word_appears_in_the_window(qapp, quartz_model):
    pattern = re.compile(r"\b(" + "|".join(VERDICT_WORDS) + r")\b",
                         re.IGNORECASE)
    windows = [_open(qapp, DATA / "dump_tri_x.lammpstrj"),
               _open(qapp, DATA / "lammps_formats" / "traj.dcd"),
               _quartz_window(qapp, quartz_model)]
    try:
        _run(qapp, windows[2])
        found = {}
        for window in windows:
            for text in _texts(window):
                for match in pattern.finditer(text or ""):
                    found.setdefault(match.group(0).lower(), text[:160])
        assert not found, found
    finally:
        dispose(*windows)


def test_each_menu_has_one_item_per_mnemonic(qapp):
    from PySide6.QtWidgets import QMenu

    window = _open(qapp, DATA / "glass.extxyz")
    try:
        menus = window.menuBar().findChildren(QMenu)
        assert len(menus) == 3          # File, Run, Help
        tops = [a.text() for a in window.menuBar().actions()]
        assert [t.replace("&", "") for t in tops] == ["File", "Run", "Help"]
        letters = [re.search(r"&(\w)", t).group(1).lower() for t in tops]
        assert len(set(letters)) == len(letters), tops
        for menu in menus:
            keys = []
            for action in menu.actions():
                match = re.search(r"&(\w)", action.text())
                if match:
                    keys.append(match.group(1).lower())
            assert len(set(keys)) == len(keys), [a.text() for a in
                                                 menu.actions()]
    finally:
        dispose(window)


# ---------------------------------------------------------------------------
# the verifiers' findings of 2026-10-07, each pinned
# ---------------------------------------------------------------------------

def _mnemonic(text: str) -> str | None:
    match = re.search(r"&(\w)", text or "")
    return match.group(1).lower() if match else None


def _page_mnemonics(window) -> dict[str, list[str]]:
    """Letter -> the menu-bar titles and the visible buttons, group boxes
    and tabs that underline it, on the page shown."""
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QTabWidget

    seen: dict[str, list[str]] = {}
    for action in window.menuBar().actions():
        letter = _mnemonic(action.text())
        if letter:
            seen.setdefault(letter, []).append(f"menu {action.text()}")
    page = window.pages.currentWidget()
    for w in page.findChildren(QAbstractButton) + page.findChildren(QGroupBox):
        if not w.isVisibleTo(window):
            continue
        text = w.text() if isinstance(w, QAbstractButton) else w.title()
        letter = _mnemonic(text)
        if letter:
            seen.setdefault(letter, []).append(text)
    for tabs in page.findChildren(QTabWidget):
        for i in range(tabs.count()):
            letter = _mnemonic(tabs.tabText(i))
            if letter:
                seen.setdefault(letter, []).append(tabs.tabText(i))
    return seen


def test_no_button_shares_a_mnemonic_with_the_menu_bar(qapp):
    """Alt+F opened no menu on the read page ('Open another MD &file…'),
    and Alt+R reached Read again or the Run button, not the Run menu."""
    windows = [_open(qapp, DATA / "dump_tri_x.lammpstrj"),
               _open(qapp, DATA / "glass.extxyz")]
    try:
        assert windows[0].pages.currentWidget() is windows[0].read_page
        assert windows[1].pages.currentWidget() is windows[1].workspace
        for window in windows:
            window.show()
            qapp.processEvents()
            seen = _page_mnemonics(window)
            shared = {k: v for k, v in seen.items() if len(v) > 1}
            assert not shared, shared
    finally:
        dispose(*windows)


def test_a_box_typed_as_three_rows_reads(qapp):
    """A box typed as rows is a (3, 3) array: open_model compared it with
    {} and the window dead-ended on numpy's 'truth value is ambiguous'."""
    from facet.ui import md_jobs

    rows = "9 0 0; 0.8 8.5 0; -0.6 0.4 9.5"
    from facet.ui.md_dialogs import _parse_read_option

    box = _parse_read_option("box_from", rows)
    assert box.shape == (3, 3)
    outcome = md_jobs.open_model(str(DATA / "recognition" / "nvt.xyz"),
                                 {"box_from": box})
    assert isinstance(outcome, md_jobs.Opened)
    assert outcome.summary.n_frames == 3
    window = _open(qapp, DATA / "recognition" / "nvt.xyz")
    try:
        window.read_panel.set_option("box_from", rows)
        assert window.read_panel.read_button.isEnabled()
        window.read_panel.read_button.click()
        assert _wait(qapp, lambda: window.summary is not None
                     or window.problem.error_type == "error")
        assert window.trajectory is not None, window.problem.message
        assert window.summary.n_atoms == 18
    finally:
        dispose(window)


def test_an_unexpected_read_failure_keeps_the_fields_asked_for(qapp,
                                                              monkeypatch):
    from facet.ui import md_jobs

    window = _open(qapp, DATA / "recognition" / "nvt.xyz")
    try:
        assert window.problem.needs == ("box_from",)

        def broken(source, read_options=None):
            raise TypeError("an exception the read does not sort")

        monkeypatch.setattr(md_jobs, "open_model", broken)
        window.read_panel.set_option("box_from", "9 0 0; 0 9 0; 0 0 9")
        window.read_panel.read_button.click()
        assert _wait(qapp, lambda: "does not sort" in window.problem.message)
        panel = window.read_panel
        # the reason is shown, and so are the fields: not a dead end
        assert "does not sort" in panel.message.text()
        assert "box_from" in panel.fields
        assert not panel.read_button.isHidden()
        assert window.pages.currentWidget() is window.read_page
        # what was tried is in the field; Read again waits for a change
        assert panel.fields["box_from"].text() == "9 0 0; 0 9 0; 0 0 9"
        assert not panel.read_button.isEnabled()
        panel.set_option("box_from", str(DATA / "recognition" / "nvt.data"))
        assert panel.read_button.isEnabled()
    finally:
        dispose(window)


def test_read_again_is_offered_only_when_something_new_would_be_sent(qapp):
    from facet.ui import md_jobs
    from facet.ui.md_dialogs import ReadOptionsPanel

    panel = ReadOptionsPanel()
    try:
        # a format the reader refuses outright: no option changes that
        refused = md_jobs.ReadProblem(
            source="ovito.nc", read_options={}, message="refused (test)",
            error_type="UnsupportedFormat", file_format="amber-netcdf",
            options_taken=frozenset({"type_map", "topology"}), needs=())
        panel.set_problem(refused)
        assert panel.read_button.isHidden()
        # round 2 of a typed XYZ: the box carried over is not a new input,
        # so with no element chosen there is nothing new to read with
        box = str(DATA / "recognition" / "nvt.data")
        round2 = md_jobs.ReadProblem(
            source="nvt_types.xyz", read_options={"box_from": box},
            message="no element for type 1 (test)", error_type="ValueError",
            file_format="extxyz",
            options_taken=frozenset({"type_map", "box_from", "units",
                                     "mass_tol_amu"}),
            needs=("type_map",), type_keys=(1, 2), keys_complete=True,
            counts={1: 6, 2: 12})
        panel.set_problem(round2)
        assert panel.fields["box_from"].text() == box
        assert not panel.read_button.isEnabled()
        assert "an element for each type" in panel.reason.text()
        panel.set_element(1, "Si")
        panel.set_element(2, "O")
        assert panel.read_button.isEnabled()
        assert panel.read_options() == {"box_from": box,
                                        "type_map": {1: "Si", 2: "O"}}
    finally:
        dispose(panel)


def test_the_read_panel_asks_in_words_and_offers_a_file_opened_with_it(qapp):
    window = _open(qapp, DATA / "dump_tri_x.lammpstrj")
    try:
        panel = window.read_panel
        assert panel.lead.text().startswith("Choose the element of each type")
        assert "LAMMPS data file of the run" in panel.lead.text()
        # the reader's own message folded under it, in full
        assert "type_map" in panel.message.text()
        assert not panel.message_fold.button.isChecked()
        from PySide6.QtWidgets import QFormLayout, QLabel
        labels = [panel.asked_form.itemAt(r, QFormLayout.LabelRole).widget()
                  .text() for r in range(panel.asked_form.rowCount())]
        assert labels == ["LAMMPS data file of the run"]
        assert isinstance(panel.asked_form.itemAt(0, QFormLayout.LabelRole)
                          .widget(), QLabel)
        # a data file opened with the dump is offered, never filled in
        data = str(DATA / "data_atomic.data")
        window.set_companions([data])
        assert panel.fields["masses_from"].text() == ""
        button = panel.companion_buttons["masses_from"]
        assert "data_atomic.data" in button.text()
        button.click()
        assert panel.fields["masses_from"].text() == data
        assert panel.read_button.isEnabled()
    finally:
        dispose(window)


def test_a_dump_dropped_with_its_data_file_is_offered_that_file(tmp_path):
    from facet.ui.preview import md_companions

    dump = DATA / "dump_tri_x.lammpstrj"
    data = DATA / "data_atomic.data"
    elsewhere = tmp_path / "other.data"
    elsewhere.write_text("x", encoding="ascii")
    models = [(str(dump), "lammps-dump"), (str(data), "lammps-data"),
              (str(elsewhere), "lammps-data")]
    assert md_companions(str(dump), models) == [str(data)]
    assert md_companions(str(data), models) == []


def test_file_save_figure_and_rows_follow_the_item_shown(qapp,
                                                         quartz_model):
    window = _quartz_window(qapp, quartz_model)
    try:
        _run(qapp, window)
        browser = window.results_view
        messages = []
        browser.statusMessage.connect(messages.append)
        # a descriptor with a figure: both offered
        assert browser.select("glass", "CN Si (BV)")
        qapp.processEvents()
        assert window.export_figure_action.isEnabled()
        assert window.export_rows_action.isEnabled()
        # the run node shows no figure and no rows: neither is offered
        browser.tree.setCurrentItem(browser.tree.topLevelItem(0))
        qapp.processEvents()
        assert browser.current() == ("run",)
        assert not window.export_figure_action.isEnabled()
        assert not window.export_rows_action.isEnabled()
        # reached anyway (a shortcut), the save says why, never silent
        window._save_figure_shown()
        window._save_rows_shown()
        assert any("No figure to save" in m for m in messages)
        assert any("No rows to save" in m for m in messages)
    finally:
        dispose(window)


def test_options_typed_for_an_analysis_not_ticked_do_not_block_the_run(
        qapp, quartz_model):
    window = _quartz_window(qapp, quartz_model)
    try:
        setup = window.setup
        # a termination Q max with no Q window: the engine refuses it, and
        # scattering says so; glass alone runs all the same
        setup.field("scattering", "termination_q_max_inv_ang").set_text("18")
        setup.refresh()
        assert "Q window" in setup.reason_text("scattering") or \
            "q_window" in setup.reason_text("scattering")
        request = setup.request()
        assert request is not None, setup.problems()
        assert request.analyses == ("glass",)
        # the value not read by the run is left out of what is sent
        assert request.scattering.termination_q_max_inv_ang is None
        # a value glass reads stays, and stays refused with its reason
        setup.field("glass", "rdf_dr_ang").set_text("-1")
        setup.refresh()
        assert setup.request() is None
        assert not setup.box("glass").isEnabled()
        assert "glass.rdf_dr_ang" in setup.reason_text("glass")
    finally:
        dispose(window)


def test_a_cancelled_run_keeps_the_last_complete_result(qapp, quartz_model,
                                                        monkeypatch):
    from facet.core import md_analysis

    window = _quartz_window(qapp, quartz_model)
    try:
        complete = _run(qapp, window)
        assert not complete.cancelled
        real_analyse = md_analysis.analyse

        def slow_analyse(trajectory, request, *, progress=None,
                         cancelled=None):
            def slow(done, total, stage):
                progress(done, total, stage)
                if re.search(r"\bframe\b", stage):
                    time.sleep(0.25)
            return real_analyse(trajectory, request, progress=slow,
                                cancelled=cancelled)

        monkeypatch.setattr(md_analysis, "analyse", slow_analyse)
        outcomes = []
        window.finished.connect(outcomes.append)
        assert window.run_analysis()
        asked = []

        def on_progress(done, total, stage):
            if re.search(r"\bframe\b", stage) and not asked:
                asked.append(stage)
                window.cancel()

        window._run_job.progressed.connect(on_progress)
        assert _wait(qapp, lambda: bool(outcomes), 120)
        partial = outcomes[0]
        assert partial.cancelled
        # the complete run stays on show, the partial one is a menu away
        assert window.result is complete
        assert window.results_view.result is complete
        assert window.partial_result is partial
        assert "stays on show" in window.stage_label.text()
        assert window.show_partial_action.isEnabled()
        assert window.show_partial_result()
        assert window.result is partial
        assert window.show_complete_action.isEnabled()
        assert window.show_complete_result()
        assert window.result is complete
    finally:
        dispose(window)


def test_a_closed_model_window_is_freed_with_its_model(qapp):
    """close() (the title bar's X) only: no dispose(). Two lambdas holding
    the window kept every closed one alive, 844 widgets and 27 MB each."""
    import gc
    import weakref

    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    from facet.ui import md_workspace

    before = len(QApplication.allWidgets())
    window = md_workspace.open_model_window(str(DATA / "glass.extxyz"))
    assert _wait(qapp, lambda: window.summary is not None)
    assert _wait(qapp, lambda: window.frame_view is not None, 30)
    assert _wait(qapp, lambda: not window._busy() and not window._jobs(), 30)
    refs = {"window": weakref.ref(window),
            "trajectory": weakref.ref(window.trajectory),
            "frame": weakref.ref(window.frame_view)}
    window.close()
    del window
    for _ in range(3):
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        gc.collect()
    alive = [name for name, ref in refs.items() if ref() is not None]
    assert not alive, alive
    assert len(QApplication.allWidgets()) <= before


def test_a_finished_job_frees_what_its_work_held(qapp):
    """The worker's deleteLater() was posted from the GUI thread to a
    thread already ended, so the worker, its work and what the work held
    (a trajectory, a frame) were never freed."""
    import gc
    import weakref

    from PySide6.QtCore import QCoreApplication, QEvent

    from facet.ui import md_jobs

    class Held:
        pass

    held = Held()
    ref = weakref.ref(held)

    def make_work(thing):
        def work(progress, cancelled):
            return type(thing).__name__
        return work

    job = md_jobs.Job(make_work(held), name="test")
    del held
    job.start()
    assert _wait(qapp, lambda: job.done, 10)
    assert job.outcome == "Held"
    for _ in range(3):
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        gc.collect()
    assert ref() is None


def test_closing_waits_at_most_close_wait_for_the_panels(qapp, quartz_model,
                                                         monkeypatch):
    from facet.ui import md_threshold, md_views, md_workspace

    asked = []
    monkeypatch.setattr(md_threshold.ThresholdPanel, "shutdown",
                        lambda self, ms=0: asked.append(("panel", ms)) or True)
    monkeypatch.setattr(md_views.ResultBrowser, "shutdown",
                        lambda self, ms=0: asked.append(("browser", ms))
                        or True)
    window = _quartz_window(qapp, quartz_model)
    assert _wait(qapp, lambda: window.frame_view is not None, 30)
    window.close()
    try:
        assert {who for who, _ in asked} == {"panel", "browser"}
        assert all(0 <= ms <= md_workspace.CLOSE_WAIT_MS for _, ms in asked)
    finally:
        dispose(window)


def test_the_model_window_lists_the_crystal_window_s_md_names(qapp):
    from facet.ui import md_workspace, preview

    assert md_workspace.md_file_filter() == preview.md_file_filter()
    for pattern in ("*.gro", "*.xsf", "*.axsf", "*.pdb", "POSCAR*",
                    "*.lammpsdump", "*.data.gz"):
        assert pattern in md_workspace.md_file_filter(), pattern


def _drop_on(window, paths) -> None:
    from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent

    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    event = QDropEvent(QPointF(20, 20), Qt.CopyAction, data, Qt.LeftButton,
                       Qt.NoModifier)
    window.dropEvent(event)


def test_a_cif_dropped_on_a_model_window_goes_to_the_crystal_window(qapp):
    from PySide6.QtWidgets import QMainWindow

    from facet.ui import md_workspace

    class Opener(QMainWindow):
        def __init__(self):
            super().__init__()
            self._model_windows = []
            self.loaded = []

        def load_many(self, paths):
            self.loaded.append(list(paths))

    opener = Opener()
    window = md_workspace.open_model_window(str(DATA / "glass.extxyz"),
                                            opener)
    try:
        assert _wait(qapp, lambda: window.summary is not None)
        before = list(md_workspace.model_windows())
        _drop_on(window, [QUARTZ])
        qapp.processEvents()
        assert [[Path(p) for p in paths] for paths in opener.loaded] == \
            [[QUARTZ]]
        assert list(md_workspace.model_windows()) == before
        assert "crystal window" in window.stage_label.text()
    finally:
        dispose(*[w for w in md_workspace.model_windows()
                  if w is not window], window, opener)


def test_the_model_window_has_a_help_menu_at_the_md_section(qapp):
    from facet.ui.help import MD_SECTION

    window = _open(qapp, DATA / "glass.extxyz")
    try:
        titles = [a.text().replace("&", "")
                  for a in window.menuBar().actions()]
        assert "Help" in titles
        assert window.help_action.shortcut().toString() != ""
        window.help_action.trigger()
        qapp.processEvents()
        dialog = window._manual
        assert dialog is not None and dialog.isVisible()
        keys = [k for k, _t, _b in dialog._sections]
        assert dialog.contents.currentRow() == keys.index(MD_SECTION)
    finally:
        dispose(window._manual, window)


def test_the_run_button_stays_in_view_under_the_setup(qapp, quartz_model):
    window = _quartz_window(qapp, quartz_model)
    try:
        window.show()
        qapp.processEvents()
        setup = window.setup
        # under the scroll area, not at the end of the long panel
        assert not window.setup_scroll.isAncestorOf(setup.run_button)
        assert setup.run_button.isVisibleTo(window)
        assert setup.run_reason.isVisibleTo(window)
        # and the empty results say where a run starts
        text = window.results_view.page.toPlainText()
        assert "Tick analyses" in text and "Ctrl+R" in text
    finally:
        dispose(window)


def test_the_kinetic_temperature_needs_velocities_in_the_file(qapp,
                                                              quartz_model):
    window = _quartz_window(qapp, quartz_model)
    try:
        setup = window.setup
        assert not window.summary.has_velocities
        setup.field(None, "frame_interval_ps").set_text("0.1")
        setup.refresh()
        assert setup.box("msd").isEnabled()
        assert not setup.box("kinetic-temperature").isEnabled()
        assert "velocities" in setup.reason_text("kinetic-temperature")
        assert not setup.box("vacf").isEnabled()
        setup.field("dynamics", "velocities").set_text("finite difference")
        setup.refresh()
        assert setup.box("vacf").isEnabled()
        assert not setup.box("kinetic-temperature").isEnabled()
    finally:
        dispose(window)


def test_former_reasons_follow_the_picker(qapp):
    window = _open(qapp, DATA / "glass.extxyz")
    try:
        setup = window.setup
        # before any former: warren-cowley's need is glass's, said so
        text = setup.reason_text("warren-cowley")
        assert text.startswith("needs glass (whose results it reads) needs")
        # 'No former to name' runs glass; the network analyses still need
        # named formers or their graph given, in the engine's words, and
        # are not told to tick 'No former to name'
        setup.formers.none_box.setChecked(True)
        setup.refresh()
        assert setup.box("glass").isEnabled()
        for name in ("rings", "polyhedral-sharing"):
            text = setup.reason_text(name)
            assert "No former to name" not in text, text
            assert "network formers" in text
        assert "graph elements" in setup.reason_text("rings")
    finally:
        dispose(window)


def test_reasons_name_fields_by_their_labels_and_link_to_them(qapp):
    window = _open(qapp, DATA / "glass.extxyz")
    try:
        setup = window.setup
        text = setup.reason_text("scattering")
        assert "r-window M(r) (scattering.r_window)" in text
        html_text = setup.groups["md_scattering"].reasons["scattering"].text()
        assert "href='field:scattering.r_window'" in html_text
        # a link opens the field, unfolding the method choices it sits in
        group = setup.groups["md_dynamics"]
        assert not group.methods_fold.button.isChecked()
        group.reasons["msd"].linkActivated.emit(
            "field:dynamics.step_limit_fraction")
        assert group.methods_fold.button.isChecked()
        assert setup.focus_field("scattering.r_window")
        assert not setup.focus_field("nowhere.nothing")
    finally:
        dispose(window)


def test_field_tips_and_choices_say_what_they_mean(qapp):
    window = _open(qapp, DATA / "glass.extxyz")
    try:
        setup = window.setup
        tip = setup.field("network", "ring_criterion").toolTip()
        assert "the definition of a ring" in tip
        assert "network.ring_criterion" in tip
        # a list to choose from is not 'typed as the command line does'
        assert "command line takes it" not in tip
        assert "command line takes it" in \
            setup.field("network", "ring_max_size").toolTip()
        combo = setup.field("network", "ring_criterion").widget
        from PySide6.QtCore import Qt
        tips = {combo.itemData(i): combo.itemData(i, Qt.ToolTipRole)
                for i in range(combo.count())}
        assert tips["primitive"].startswith("Primitive")
        assert tips["king"].startswith("King")
        assert "Lorch" in setup.field("scattering", "r_window").widget \
            .itemData(setup.field("scattering", "r_window").widget
                      .findData("Lorch"), Qt.ToolTipRole)
        label = setup.groups["md_network"].labels[("network",
                                                   "ring_max_size")]
        assert "graph nodes" in label.text()
        assert "first Q of the grid" in setup.field(
            "scattering", "termination_q_min_inv_ang").placeholder
    finally:
        dispose(window)


def test_no_developer_note_reaches_the_window_or_the_manual(qapp):
    from facet.ui.help import MD_SECTION, _sections

    window = _open(qapp, DATA / "glass.extxyz")
    try:
        text = window.setup.reason_text("nmr")
        assert "nmr.correlation" in text
        assert "TODO" not in text
        assert "none ships with FACET" in text
        assert not any("TODO" in t for t in _texts(window))
    finally:
        dispose(window)
    body = {k: b for k, _t, b in _sections()}[MD_SECTION]
    assert "TODO" not in body


def test_an_analysis_says_what_it_leaves_out_while_a_field_is_blank(
        qapp, quartz_model):
    from facet.ui.help import MD_SECTION, _sections

    window = _quartz_window(qapp, quartz_model)
    try:
        setup = window.setup
        setup.field(None, "frame_interval_ps").set_text("0.1")
        setup.refresh()
        assert setup.box("msd").isEnabled()
        text = setup.reason_text("msd")
        assert "diffusion coefficients" in text and "fit window" in text
        setup.field("dynamics", "fit_t_min_ps").set_text("0.1")
        setup.field("dynamics", "fit_t_max_ps").set_text("0.4")
        setup.refresh()
        assert "diffusion coefficients" not in setup.reason_text("msd")
        # quartz holds two elements: Bhatia-Thornton is not left out
        setup.field("scattering", "r_window").set_text("Lorch")
        setup.field("scattering", "radiations").set_text("neutron")
        setup.refresh()
        assert setup.box("scattering").isEnabled()
        assert "the FSDP" in setup.reason_text("scattering")
        assert "Bhatia-Thornton" not in setup.reason_text("scattering")
        # the drift choice that changes every MSD is beside the inputs
        group = setup.groups["md_dynamics"]
        drift = setup.field("dynamics", "remove_com_drift")
        assert not group.methods_fold.inner.isAncestorOf(drift)
    finally:
        dispose(window)
    body = {k: b for k, _t, b in _sections()}[MD_SECTION]
    assert "one neighbour search per frame" not in body
    assert "Optional:" in body
    assert "velocities in the file" in body


def test_units_can_be_stated_and_the_file_read_again(qapp):
    window = _open(qapp, DATA / "glass.extxyz")
    try:
        setup = window.setup
        assert "units" in setup.read_fields
        setup.read_fields["units"].setText("metal")
        setup.reread_button.click()
        assert _wait(qapp, lambda: window.setup is not setup
                     and window.summary is not None)
        assert window.summary.read_options.get("units") == "metal"
        assert window.setup.read_fields["units"].text() == "metal"
    finally:
        dispose(window)


def _jumping_model(folder: Path) -> Path:
    """The quartz cell over four frames, one atom moved by 0.35 of the cell
    height between frames 1 and 2 (beyond the unwrap's step limit, 0.25)."""
    from facet.core import md_model, readers

    base, _ = md_model.supercell_frame(readers.read(QUARTZ), (2, 2, 2))
    frames = []
    for k in range(4):
        cart = base.cart_ang.copy()
        if k >= 2:
            cart[0, 2] += 0.35 * float(base.box_ang[2, 2])
        frames.append(md_model.frame_from_arrays(base.elements, cart,
                                                 box_ang=base.box_ang))
    return _write_extxyz(folder / "jump.extxyz", frames)


def test_a_dynamics_refusal_comes_before_the_run_not_after_it(
        qapp, tmp_path, monkeypatch):
    """The tracks were refused after every per-frame analysis had run (395 s
    on a hot melt read every 10th frame); now within a second, before any
    frame is analysed, and the dynamics then say why beside their boxes."""
    from facet.core import md_analysis

    calls = []
    real_analyse = md_analysis.analyse

    def recording(*args, **kwargs):
        calls.append(args[1].analyses)
        return real_analyse(*args, **kwargs)

    monkeypatch.setattr(md_analysis, "analyse", recording)
    window = _open(qapp, _jumping_model(tmp_path))
    outcomes = []
    window.finished.connect(outcomes.append)
    try:
        setup = window.setup
        setup.formers.set_formers({"Si"})
        setup.field(None, "frame_interval_ps").set_text("0.1")
        setup.set_analyses(["glass", "msd"])
        request = setup.request()
        assert request is not None and "msd" in request.analyses
        assert window.run_analysis()
        assert _wait(qapp, lambda: bool(outcomes), 60)
        failure = outcomes[0]
        assert failure.kind == "request"
        assert calls == []                   # nothing was analysed
        assert "step_limit_fraction" in window.stage_label.text()
        setup.refresh()
        assert not setup.box("msd").isEnabled()
        assert "tracks" in setup.reason_text("msd")
        assert setup.request().analyses == ("glass",)
        # another frame range is another set of tracks: asked again
        setup.frame_range.set_range(0, 1, 1)
        setup.refresh()
        assert setup.box("msd").isEnabled()
    finally:
        dispose(window)


def test_measured_fractions_read_as_the_command_line_reads_them(qapp,
                                                                tmp_path):
    import json

    one_set = {"model_descriptor": "Qn Si",
               "descriptor": "Si Qn (test values)",
               "values": {"4": 0.9, "3": 0.1}, "source": "test values"}
    single = tmp_path / "one.json"
    single.write_text(json.dumps(one_set), encoding="utf-8")
    listed = tmp_path / "list.json"
    listed.write_text(json.dumps([one_set, dict(one_set)]), encoding="utf-8")
    toml = tmp_path / "sets.toml"
    toml.write_text('[[measured_fractions]]\nmodel_descriptor = "Qn Si"\n'
                    'descriptor = "Si Qn (test values)"\n'
                    'source = "test values"\n'
                    '[measured_fractions.values]\n"4" = 0.9\n"3" = 0.1\n',
                    encoding="utf-8")
    window = _open(qapp, DATA / "glass.extxyz")
    try:
        # the list file --nmr-fractions takes was refused by the window
        setup = window.setup
        setup.extra("nmr.measured_fractions").add_path(str(listed))
        options, errors = setup.groups["md_spectroscopy"].options("nmr")
        assert errors == []
        assert len(options.measured_fractions) == 2
    finally:
        dispose(window)
    from facet.ui.md_dialogs import fractions_from_files

    assert len(fractions_from_files([str(single)])) == 1
    assert len(fractions_from_files([str(listed)])) == 2
    assert len(fractions_from_files([str(toml)])) == 1


def test_a_script_ending_while_a_job_runs_exits_cleanly(tmp_path):
    """No app.exec(), so no aboutToQuit: the jobs are waited for at
    interpreter exit (a QThread destroyed while it runs aborts, exit 127)."""
    import subprocess
    import sys

    script = tmp_path / "ends_early.py"
    script.write_text(
        "import os, sys, time\n"
        "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        f"sys.path.insert(0, {str(HERE.parent)!r})\n"
        "from PySide6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "from facet.ui import md_jobs\n"
        "def work(progress, cancelled):\n"
        "    time.sleep(1.5)\n"
        "    return 1\n"
        "job = md_jobs.Job(work, name='late')\n"
        "job.start()\n"
        "time.sleep(0.2)\n"
        "app.processEvents()\n"
        "print('leaving with the job running', job.is_running())\n",
        encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, timeout=120,
                          env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert "leaving with the job running True" in done.stdout
    assert done.returncode == 0, (done.returncode, done.stderr[-2000:])


def test_the_setup_saves_a_request_the_command_line_reads(qapp, quartz_model,
                                                          tmp_path):
    """A GUI run could be repeated on the command line only by retyping
    each option; File > Save the request writes the file --request reads,
    and the command line builds the same request from it."""
    import tomllib

    from facet.md import cli

    window = _quartz_window(qapp, quartz_model)
    try:
        setup = window.setup
        setup.field("network", "ring_criterion").set_text("primitive")
        setup.field("network", "ring_max_size").set_text("8")
        setup.field("scattering", "r_window").set_text("Lorch")
        setup.field("scattering", "radiations").set_text("neutron, X-ray")
        setup.field("glass", "cutoffs_ang").set_text("Si-O=2.1")
        setup.field(None, "frame_interval_ps").set_text("0.1")
        setup.field("dynamics", "remove_com_drift").set_text("true")
        setup.frame_range.set_range(1, 5, 2)
        setup.set_analyses(["glass", "rings", "scattering", "msd"])
        gui = setup.request()
        assert gui is not None, setup.problems()
        assert window.save_request_action.isEnabled()
        path = window.save_request(tmp_path / "request.toml")
        text = path.read_text(encoding="utf-8")
        assert "py -3.11 -m facet.md analyse" in text and "--request" in text
        from_file = cli.request_from_mapping(tomllib.loads(text))
        assert from_file == gui
    finally:
        dispose(window)


def test_the_notes_tab_lists_each_line_of_an_analysis_once(qapp,
                                                           quartz_model):
    import dataclasses

    window = _quartz_window(qapp, quartz_model)
    try:
        result = _run(qapp, window)
        output = result.outputs["glass"]
        assert output.notes
        doubled = dataclasses.replace(result, outputs={
            "glass": dataclasses.replace(
                output, notes=type(output.notes)(list(output.notes) * 3))})
        lines = window.notes_lines(doubled)
        start = lines.index("== glass ==") + 1
        end = next(i for i in range(start, len(lines))
                   if lines[i].startswith("== "))
        section = [line for line in lines[start:end] if line.strip()]
        assert len(section) == len(set(section))
        assert all(f"note: {n}" in section for n in output.notes)
    finally:
        dispose(window)


# ---------------------------------------------------------------------------
# the final check of the Model window's layout and texts (2026-10-07)
# ---------------------------------------------------------------------------

def test_the_setup_s_tables_and_labels_fit_its_width(qapp):
    """At the window's first size (a 450 px setup) the measured-curves
    table showed two of its five headers, the oxidation table cut its
    source to 'common ...', and the largest-ring label set the label column
    of the whole network box, its fields 165 px wide."""
    from facet.ui.md_dialogs import _CurveTable, label_of

    window = _open(qapp, DATA / "glass.extxyz")
    try:
        window.resize(1440, 920)
        window.show()
        qapp.processEvents()
        setup = window.setup
        curves = setup.extra("scattering.measured")
        assert isinstance(curves, _CurveTable)
        table = curves.table
        width = sum(table.columnWidth(c) for c in range(table.columnCount()))
        assert width <= table.viewport().width() + 1
        tips = [table.horizontalHeaderItem(c).toolTip()
                for c in range(table.columnCount())]
        assert "range (axis unit)" in tips[3]
        assert "scale (blank: fitted)" in tips[4]
        oxidation = setup.oxidation
        width = sum(oxidation.columnWidth(c)
                    for c in range(oxidation.columnCount()))
        assert width <= oxidation.viewport().width() + 1
        source = oxidation.item(0, 3).text()
        assert oxidation.columnWidth(3) >= \
            oxidation.fontMetrics().horizontalAdvance(source)
        assert label_of("ring_max_size") == "largest ring (graph nodes)"
        assert "T atoms on the bridging-anion graph" in setup.field(
            "network", "ring_max_size").placeholder
        # the note points at the box where it is: above it
        assert "box above" in setup.formers.note.text()
    finally:
        dispose(window)


def test_a_run_in_progress_says_so_where_its_results_will_be(qapp,
                                                              quartz_model):
    """The results pane said 'No run yet' while a run was going."""
    window = _quartz_window(qapp, quartz_model)
    try:
        page = window.results_view.page
        assert "No run yet" in page.toPlainText()
        assert window.run_analysis()
        assert "run is in progress" in page.toPlainText()
        assert _wait(qapp, lambda: not window.is_running(), 120)
        assert window.result is not None
    finally:
        dispose(window)


def test_a_refused_run_keeps_the_notes_of_the_run_on_show(qapp,
                                                          quartz_model):
    """A second run refused (or cancelled before its first frame) replaced
    the Notes and provenance tab with its one line, while the Results tab
    still showed the earlier run."""
    from facet.ui import md_jobs

    window = _quartz_window(qapp, quartz_model)
    try:
        result = _run(qapp, window)
        assert result is not None
        window._on_run_failed(md_jobs.JobFailure(
            "cancelled", "cancelled before the first frame"))
        assert window.result is result
        text = window.notes_text.toPlainText()
        assert text.startswith("The run was cancelled before any frame")
        assert "== glass ==" in text
        assert "earlier run" in text
    finally:
        dispose(window)


def test_a_job_s_worker_is_deleted_on_the_gui_thread_not_its_own(qapp):
    """A job's worker was deleted on its own thread as the thread ended; a
    worker made in Python then takes the GIL inside QObject's destructor,
    under Qt's signal-slot locks, and the GUI thread, holding the GIL in a
    connect, waited for ever: a run of this file hung in show_frame's
    connect (py-spy, 2026-10-07). The worker now outlives its thread and is
    deleted on the GUI thread when the job ends."""
    import shiboken6

    from facet.ui import md_jobs

    job = md_jobs.Job(lambda progress, cancelled: 1, name="probe")
    ended = []
    job.ended.connect(lambda: ended.append(True))
    job.start()
    worker, thread = job._worker, job._thread
    assert thread.wait(10000)
    # the thread has ended and nothing ran on the GUI thread yet: the
    # worker was not deleted on its own thread
    assert shiboken6.isValid(worker)
    assert _wait(qapp, lambda: bool(ended), 10)
    assert job.outcome == 1
    assert not shiboken6.isValid(worker)


# ---------------------------------------------------------------------------
# readable under the system's dark palette, and inside the screen
# ---------------------------------------------------------------------------
#
# Measured on the built window with Windows in dark mode (2026-10-07): the
# read page's heading, its first paragraph and the label beside the topology
# field drawn in the theme's text colour (23, 26, 31) on (30, 30, 30), a
# contrast of 1.05, and the same on every label of the setup. The scroll
# area's page and viewport filled with the operating system's Window colour,
# not the theme's. And the window opened 1440 x 950 with its title bar on a
# screen whose available area is 930 px tall.

# The colours of the Windows 11 style's dark palette, as Qt 6.9 reports them
# with Windows set to dark mode for apps.
WINDOWS_DARK = {
    "Window": (30, 30, 30), "WindowText": (255, 255, 255),
    "Base": (45, 45, 45), "AlternateBase": (52, 52, 52),
    "Text": (255, 255, 255), "Button": (60, 60, 60),
    "ButtonText": (255, 255, 255), "BrightText": (166, 216, 255),
    "ToolTipBase": (60, 60, 60), "ToolTipText": (212, 212, 212),
    "Highlight": (135, 100, 184), "HighlightedText": (255, 255, 255),
    "PlaceholderText": (171, 171, 171), "Light": (91, 91, 91),
    "Midlight": (69, 69, 69), "Mid": (40, 40, 40), "Dark": (20, 20, 20),
    "Shadow": (0, 0, 0), "Link": (147, 147, 255),
}


def _dark_palette():
    """A palette with every role set, so that no colour-scheme request
    replaces it: what a platform that ignores the request leaves."""
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        for name, rgb in WINDOWS_DARK.items():
            palette.setColor(group, getattr(QPalette, name), QColor(*rgb))
    return palette


class _DarkSystem:
    """The system's dark palette and FACET's chrome, for one test; both
    undone afterwards (the chrome asks for a colour scheme; the request is
    withdrawn)."""

    def __init__(self, qapp):
        self.app = qapp

    def __enter__(self):
        from facet.core import theme as theme_mod
        from facet.ui import chrome

        self.palette = self.app.palette()
        self.sheet = self.app.styleSheet()
        self.app.setPalette(_dark_palette())
        chrome.apply(self.app, theme_mod.Theme())
        self.app.processEvents()
        return self

    def __exit__(self, *_exc):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication

        self.app.setStyleSheet(self.sheet)
        hints = QGuiApplication.styleHints()
        if hasattr(hints, "setColorScheme"):
            hints.setColorScheme(Qt.ColorScheme.Unknown)
        self.app.setPalette(self.palette)
        self.app.processEvents()
        return False


def _luminance(rgb) -> np.ndarray:
    """WCAG relative luminance of 0..255 sRGB triples."""
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def _pixels(image) -> np.ndarray:
    from PySide6.QtGui import QImage

    image = image.convertToFormat(QImage.Format_RGB32)
    h, line = image.height(), image.bytesPerLine()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8,
                        count=line * h).reshape(h, line)
    return raw[:, :image.width() * 4].reshape(h, image.width(), 4)[
        ..., [2, 1, 0]]


def _ink_contrast(pixels, rect, dpr):
    """(contrast, ink, background) of the text drawn in ``rect``: the
    background is the commonest colour there, the ink the pixel of greatest
    contrast against it."""
    x0, y0 = max(0, int(rect.left() * dpr)), max(0, int(rect.top() * dpr))
    x1 = min(pixels.shape[1], int((rect.right() + 1) * dpr))
    y1 = min(pixels.shape[0], int((rect.bottom() + 1) * dpr))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    block = pixels[y0:y1, x0:x1].reshape(-1, 3)
    colours, counts = np.unique(block, axis=0, return_counts=True)
    background = colours[int(np.argmax(counts))]
    lb, lp = _luminance(background), _luminance(block)
    ratio = (np.maximum(lp, lb) + 0.05) / (np.minimum(lp, lb) + 0.05)
    i = int(np.argmax(ratio))
    return (float(ratio[i]), tuple(int(v) for v in block[i]),
            tuple(int(v) for v in background))


def _text_rect(widget):
    """Where a label's, a check box's or a group box's text is drawn."""
    from PySide6.QtWidgets import (QCheckBox, QGroupBox, QStyle,
                                   QStyleOptionButton, QStyleOptionGroupBox)

    style = widget.style()
    if isinstance(widget, QGroupBox):
        option = QStyleOptionGroupBox()
        option.initFrom(widget)
        option.text = widget.title()
        option.lineWidth = 1
        option.subControls = QStyle.SC_GroupBoxFrame | QStyle.SC_GroupBoxLabel
        return style.subControlRect(QStyle.CC_GroupBox, option,
                                    QStyle.SC_GroupBoxLabel, widget)
    if isinstance(widget, QCheckBox):
        option = QStyleOptionButton()
        option.initFrom(widget)
        option.text = widget.text()
        return style.subElementRect(QStyle.SE_CheckBoxContents, option,
                                    widget)
    return widget.contentsRect()


def _ratio(a, b) -> np.ndarray:
    """WCAG contrast of 0..255 sRGB triples (arrays broadcast)."""
    la, lb = _luminance(a), _luminance(b)
    return (np.maximum(la, lb) + 0.05) / (np.minimum(la, lb) + 0.05)


def _crop(pixels, window, widget, rect, dpr) -> np.ndarray:
    """The pixels of ``rect`` (``widget``'s coordinates) in a grab of
    ``window``, as rows x columns x 3."""
    top_left = widget.mapTo(window, rect.topLeft())
    x0, y0 = int(top_left.x() * dpr), int(top_left.y() * dpr)
    x1 = int((top_left.x() + rect.width()) * dpr)
    y1 = int((top_left.y() + rect.height()) * dpr)
    return np.asarray(pixels[max(0, y0):y1, max(0, x0):x1], dtype=int)


def _commonest(block) -> np.ndarray:
    colours, counts = np.unique(block.reshape(-1, 3), axis=0,
                                return_counts=True)
    return colours[int(np.argmax(counts))]


def _link_contrast(pixels, window, label, dpr):
    """(contrast, link colour, background) of the links of a label that
    carries some, when that colour is on screen there; else None. The
    strongest pixel of such a label is its body text and says nothing of
    its links: links at (233, 212, 242) on (241, 241, 241) passed that way."""
    from PySide6.QtGui import QPalette

    rect = label.contentsRect().intersected(
        label.visibleRegion().boundingRect())
    if rect.isEmpty():
        return None
    block = _crop(pixels, window, label, rect, dpr).reshape(-1, 3)
    if block.size == 0:
        return None
    colour = np.array(label.palette().color(QPalette.Link).getRgb()[:3])
    if int((np.abs(block - colour).max(axis=1) <= 30).sum()) < 6:
        return None
    background = _commonest(block)
    return (float(_ratio(colour, background)),
            tuple(int(v) for v in colour), tuple(int(v) for v in background))


def _page_contrasts(qapp, window, page) -> list[tuple]:
    """(contrast, kind, text, ink, background) of every label, check box and
    group-box title of ``page`` that shows text, and of every empty field's
    placeholder, its scroll area scrolled through. ``kind`` is 'body';
    'muted' for what the chrome draws as secondary text (hints, group
    titles, disabled controls); 'placeholder'; or 'link', for the link colour
    of a label that carries links."""
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import (QCheckBox, QGroupBox, QLabel, QLineEdit,
                                   QScrollArea)

    area = next((a for a in window.findChildren(QScrollArea)
                 if a.widget() is page), None)
    bar = area.verticalScrollBar() if area is not None else None
    if bar is not None:
        step = max(60, int(area.viewport().height() * 0.7))
        positions = list(range(0, bar.maximum() + step, step))
    else:
        positions = [0]
    best: dict = {}
    for value in positions:
        if bar is not None:
            bar.setValue(min(value, bar.maximum()))
            qapp.processEvents()
        image = window.grab().toImage()
        pixels, dpr = _pixels(image), image.devicePixelRatio()
        widgets = [w for kind in (QLabel, QCheckBox, QGroupBox)
                   for w in page.findChildren(kind)]
        for w in widgets:
            if not w.isVisible():
                continue
            text = w.title() if isinstance(w, QGroupBox) else w.text()
            if not text.strip() or (isinstance(w, QLabel)
                                    and not w.pixmap().isNull()):
                continue
            rect = _text_rect(w).intersected(w.visibleRegion().boundingRect())
            if rect.isEmpty():
                continue
            measured = _ink_contrast(
                pixels, QRect(w.mapTo(window, rect.topLeft()), rect.size()),
                dpr)
            if measured is None:
                continue
            muted = isinstance(w, QGroupBox) or not w.isEnabled() or \
                w.objectName() == "hint"
            size = rect.width() * rect.height()
            if id(w) not in best or size > best[id(w)][0]:
                best[id(w)] = (size, (measured[0],
                                      "muted" if muted else "body",
                                      text[:60], measured[1], measured[2]))
            if isinstance(w, QLabel) and "<a " in text:
                link = _link_contrast(pixels, window, w, dpr)
                if link is not None:
                    key = ("link", id(w))
                    if key not in best or size > best[key][0]:
                        best[key] = (size, (link[0], "link", text[:60],
                                            link[1], link[2]))
        for w in page.findChildren(QLineEdit):
            if not w.isVisible() or w.text() or not w.placeholderText():
                continue
            rect = w.rect().adjusted(3, 3, -3, -3).intersected(
                w.visibleRegion().boundingRect())
            if rect.isEmpty():
                continue
            measured = _ink_contrast(
                pixels, QRect(w.mapTo(window, rect.topLeft()), rect.size()),
                dpr)
            if measured is None:
                continue
            size = rect.width() * rect.height()
            key = ("placeholder", id(w))
            if key not in best or size > best[key][0]:
                best[key] = (size, (measured[0], "placeholder",
                                    w.placeholderText()[:60], measured[1],
                                    measured[2]))
    if bar is not None:
        bar.setValue(0)
    return [entry for _, entry in best.values()]


def _assert_readable(found, where):
    """Every kind of text at 4.5:1 or more. Secondary text was softened to
    3.2:1 by design and measured 3.80; it is held to 4.5 now, as are
    placeholders (3.36 before) and links."""
    def listed(rows):
        return "; ".join(f"{r:.2f} {kind} {text!r} ink {ink} on {bg}"
                         for r, kind, text, ink, bg in rows[:6])

    body = [f for f in found if f[1] == "body"]
    assert body, f"{where}: no text was measured"
    low = [f for f in found if f[0] < 4.5]
    assert not low, (f"{where}: text below 4.5:1 against what is behind "
                     f"it: {listed(low)}")


def test_the_read_and_setup_pages_are_readable_on_a_dark_system(qapp):
    """The pages inside scroll areas showed the system's dark Window colour
    under the theme's dark text: 1.05:1 for the read page's heading, its
    first paragraph and the label beside the topology / masses field, and
    for every label of the setup."""
    from PySide6.QtWidgets import QToolButton

    with _DarkSystem(qapp):
        for name in ("dump_tri_x.lammpstrj", "lammps_formats/traj.dcd"):
            window = _open(qapp, DATA / name)
            try:
                window.show()
                qapp.processEvents()
                assert window.pages.currentWidget() is window.read_page
                panel = window.read_panel
                panel.message_fold.button.setChecked(True)
                panel.more.button.setChecked(True)
                qapp.processEvents()
                found = _page_contrasts(qapp, window, panel)
                assert any("needs more information" in f[2] for f in found)
                _assert_readable(found, f"the read page of {name}")
            finally:
                dispose(window)

        window = _open(qapp, DATA / "glass.extxyz")
        try:
            window.show()
            qapp.processEvents()
            setup = window.setup
            for button in setup.findChildren(QToolButton):
                if button.objectName() == "disclosure":
                    button.setChecked(True)
            qapp.processEvents()
            found = _page_contrasts(qapp, window, setup)
            assert len(found) > 50
            # the links and the placeholders were measured, not passed over
            kinds = {f[1] for f in found}
            assert {"link", "placeholder", "muted"} <= kinds, kinds
            _assert_readable(found, "the setup page")
        finally:
            dispose(window)


def _grab_showing(qapp, window, area, widget):
    """A grab of ``window`` (pixels, device pixel ratio) with ``widget``
    scrolled into view in ``area``."""
    if area is not None:
        area.ensureWidgetVisible(widget, 0, 0)
    qapp.processEvents()
    image = window.grab().toImage()
    return _pixels(image), image.devicePixelRatio()


def _strongest(block):
    """(contrast, ink, background): the commonest colour of ``block`` and
    the pixel of greatest contrast against it."""
    flat = np.asarray(block, dtype=int).reshape(-1, 3)
    background = _commonest(flat)
    ratio = _ratio(flat, background)
    i = int(np.argmax(ratio))
    return (float(ratio[i]), tuple(int(v) for v in flat[i]),
            tuple(int(v) for v in background))


def test_what_the_style_draws_is_readable_on_a_dark_system(qapp,
                                                           quartz_model):
    """What the style sheet leaves to the Windows 11 style, measured with
    Windows dark (2026-10-07): a ticked box was a white tick on the panel,
    1.11:1, so ticked and unticked looked alike; a spin box's arrows were
    dots 3 px wide; the pressed Provenance button carried dark text on the
    accent, 2.87; table headers and unselected tabs were 3.80 and a disabled
    menu item 4.29. With a platform that keeps its own dark palette the
    Provenance button was a (60, 60, 60) face under dark text, 1.58."""
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtWidgets import (QAbstractSpinBox, QCheckBox, QHeaderView,
                                   QMenu, QScrollArea, QStyle,
                                   QStyleOptionButton, QStyleOptionSpinBox)

    with _DarkSystem(qapp):
        window = _quartz_window(qapp, quartz_model)
        try:
            window.show()
            qapp.processEvents()
            setup = window.setup
            area = next((a for a in window.findChildren(QScrollArea)
                         if a.widget() is setup), None)

            # a ticked box, inside, against how an unticked one looks there
            boxes = [b for b in setup.findChildren(QCheckBox)
                     if b.isVisible() and b.isEnabled()]
            ticked = next(b for b in boxes if b.isChecked())
            unticked = next(b for b in boxes if not b.isChecked())

            def indicator(box):
                option = QStyleOptionButton()
                option.initFrom(box)
                rect = box.style().subElementRect(
                    QStyle.SE_CheckBoxIndicator, option, box)
                pixels, dpr = _grab_showing(qapp, window, area, box)
                return _crop(pixels, window, box, rect, dpr)

            on, off = indicator(ticked), indicator(unticked)
            h = min(on.shape[0], off.shape[0])
            w = min(on.shape[1], off.shape[1])
            inside = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
            empty = _commonest(off[inside])
            marked = int((_ratio(on[inside].reshape(-1, 3), empty)
                          >= 3.0).sum())
            assert marked >= 8, (
                f"{ticked.text()!r} ticked differs from an unticked box "
                f"{tuple(empty)} by 3:1 in {marked} pixels")

            # a spin box's arrows are arrows, not dots
            spin = next(s for s in setup.findChildren(QAbstractSpinBox)
                        if s.isVisible() and s.isEnabled())
            pixels, dpr = _grab_showing(qapp, window, area, spin)
            option = QStyleOptionSpinBox()
            option.initFrom(spin)
            option.frame = True
            option.buttonSymbols = spin.buttonSymbols()
            option.stepEnabled = (QAbstractSpinBox.StepUpEnabled
                                  | QAbstractSpinBox.StepDownEnabled)
            option.subControls = (QStyle.SC_SpinBoxUp | QStyle.SC_SpinBoxDown
                                  | QStyle.SC_SpinBoxFrame
                                  | QStyle.SC_SpinBoxEditField)
            for control, name in ((QStyle.SC_SpinBoxUp, "up"),
                                  (QStyle.SC_SpinBoxDown, "down")):
                rect = spin.style().subControlRect(
                    QStyle.CC_SpinBox, option, control, spin).adjusted(
                    3, 3, -3, -3)
                block = _crop(pixels, window, spin, rect, dpr)
                ink = (_ratio(block, _commonest(block)) >= 3.0).any(axis=0)
                columns = np.nonzero(ink)[0]
                wide = int(columns.max() - columns.min() + 1) \
                    if columns.size else 0
                assert wide >= 6, f"the spin box's {name} arrow: {wide} px wide"

            _run(qapp, window)
            view = window.results_view
            window.tabs.setCurrentWidget(view)
            qapp.processEvents()

            # the Provenance button, released and pressed
            button = view.provenance_button
            for pressed in (False, True):
                button.setChecked(pressed)
                pixels, dpr = _grab_showing(qapp, window, None, button)
                found = _strongest(_crop(pixels, window, button,
                                         button.rect().adjusted(5, 5, -5, -5),
                                         dpr))
                assert found[0] >= 4.5, ("Provenance", pressed, found)
            button.setChecked(False)

            # unselected tabs and table headers: secondary text, at 4.5
            pixels, dpr = _grab_showing(qapp, window, None, window.tabs)
            bar = window.tabs.tabBar()
            measured = []
            for i in range(bar.count()):
                if i != bar.currentIndex():
                    measured.append((_strongest(_crop(
                        pixels, window, bar, bar.tabRect(i), dpr)),
                        f"tab {bar.tabText(i)}"))
            for header in window.tabs.findChildren(QHeaderView):
                if not header.isVisible():
                    continue
                model = header.model()
                for s in range(header.count()):
                    label = model.headerData(s, header.orientation()) \
                        if model is not None else None
                    if header.isSectionHidden(s) or not str(label or
                                                            "").strip():
                        continue
                    rect = QRect(header.sectionViewportPosition(s), 0,
                                 header.sectionSize(s),
                                 header.height()).intersected(
                        header.viewport().visibleRegion().boundingRect())
                    if rect.width() < 12:
                        continue
                    measured.append((_strongest(_crop(
                        pixels, window, header.viewport(), rect, dpr)),
                        f"header {label}"))
            assert any(m[1].startswith("header") for m in measured)
            low = [m for m in measured if m[0][0] < 4.5]
            assert not low, low

            # a disabled menu item
            menu = next(m for m in window.findChildren(QMenu)
                        if m.title().replace("&", "") == "Run")
            menu.popup(window.mapToGlobal(QPoint(40, 40)))
            try:
                assert _wait(qapp, menu.isVisible, 5)
                qapp.processEvents()
                image = menu.grab().toImage()
                pixels, dpr = _pixels(image), image.devicePixelRatio()
                disabled = [a for a in menu.actions()
                            if a.text() and not a.isSeparator()
                            and not a.isEnabled()]
                assert disabled
                for action in disabled:
                    found = _strongest(_crop(
                        pixels, menu, menu,
                        menu.actionGeometry(action).adjusted(4, 2, -4, -2),
                        dpr))
                    assert found[0] >= 4.5, (action.text(), found)
            finally:
                menu.hide()
        finally:
            dispose(window)


def test_the_chrome_asks_for_the_theme_s_colour_scheme(qapp):
    """What the style sheet does not name -- tick boxes, the results'
    Provenance button -- the Windows 11 style draws from the system's
    scheme: with Windows dark, unticked boxes all but vanished on the white
    theme's panels and the Provenance button was a dark face under dark
    text."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication

    from facet.core import theme as theme_mod
    from facet.ui import chrome

    hints = QGuiApplication.styleHints()
    if not hasattr(hints, "setColorScheme"):
        pytest.skip("Qt before 6.8 cannot be asked for a colour scheme")
    if qapp.platformName() in ("offscreen", "minimal"):
        pytest.skip("this platform has no colour scheme to ask for")
    sheet, palette = qapp.styleSheet(), qapp.palette()
    try:
        for theme, wanted in ((theme_mod.Theme(), Qt.ColorScheme.Light),
                              (theme_mod.dark(), Qt.ColorScheme.Dark),
                              (theme_mod.Theme(), Qt.ColorScheme.Light)):
            chrome.apply(qapp, theme)
            qapp.processEvents()
            assert hints.colorScheme() == wanted, theme.name
        # asked for even when the scheme already matches: the request is
        # what holds it if the system switches while FACET runs
        asked = []
        hints.setColorScheme = lambda scheme: asked.append(scheme)
        try:
            chrome.apply(qapp, theme_mod.Theme())
        finally:
            del hints.setColorScheme
        assert asked == [Qt.ColorScheme.Light]
    finally:
        qapp.setStyleSheet(sheet)
        hints.setColorScheme(Qt.ColorScheme.Unknown)
        qapp.setPalette(palette)
        qapp.processEvents()


def test_the_application_palette_carries_the_theme_s_colours(qapp):
    """What the style sheet does not name came from the system's palette:
    with a platform that keeps its own dark palette, links in the setup at
    (233, 212, 242) on the white theme's panels (1.23:1) and a tool
    button's (60, 60, 60) face under dark text (1.58)."""
    from PySide6.QtGui import QPalette

    from facet.core import theme as theme_mod
    from facet.ui import chrome

    sheet, palette = qapp.styleSheet(), qapp.palette()
    try:
        qapp.setPalette(_dark_palette())
        for theme in (theme_mod.Theme(), theme_mod.dark()):
            chrome.apply(qapp, theme)
            colours = chrome.ui_colors(theme)
            now = qapp.palette()
            for role, expected in (("Window", colours.window),
                                   ("WindowText", colours.text),
                                   ("Base", colours.base),
                                   ("Text", colours.text),
                                   ("Button", colours.window),
                                   ("ButtonText", colours.text),
                                   ("Link", colours.link),
                                   ("PlaceholderText", colours.muted),
                                   ("Highlight", colours.accent),
                                   ("HighlightedText", colours.accent_text)):
                got = now.color(QPalette.Active, getattr(QPalette, role))
                assert got.name() == expected, (theme.name, role, got.name())
            assert now.color(QPalette.Disabled, QPalette.Text).name() == \
                colours.muted
    finally:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication

        qapp.setStyleSheet(sheet)
        hints = QGuiApplication.styleHints()
        if hasattr(hints, "setColorScheme"):
            hints.setColorScheme(Qt.ColorScheme.Unknown)
        qapp.setPalette(palette)
        qapp.processEvents()


def test_the_window_opens_inside_the_screen_frame_included(qapp):
    """A fixed 1440 x 920 opened 1440 x 950 with its title bar on a screen
    whose available area is 930 px tall: the status bar, with the run's
    progress and Cancel run, sat behind the task bar."""
    from PySide6.QtCore import QMargins, QRect

    window = _open(qapp, DATA / "glass.extxyz")
    try:
        window.show()
        for _ in range(5):
            qapp.processEvents()
        available = window.screen().availableGeometry()
        frame = window.frameGeometry()
        assert available.contains(frame), (
            f"frame {frame.getRect()} outside the available area "
            f"{available.getRect()}")
        # any screen: a 1366 x 768 laptop (a 48 px task bar, a 30 px title
        # bar), then a large one beside it
        for screen, margins in (
                (QRect(0, 0, 1366, 720), QMargins(0, 30, 0, 0)),
                (QRect(1920, 0, 2560, 1392), QMargins(8, 31, 8, 8))):
            chosen = window.fit_to_screen(screen, margins)
            assert screen.contains(chosen), (chosen.getRect(),
                                             screen.getRect())
            assert chosen.size().width() == window.width() + 16 * (
                margins.left() == 8)
            assert window.width() <= window.PREFERRED_SIZE[0]
            assert window.height() <= window.PREFERRED_SIZE[1]
        # as large as it asks for where there is room
        assert (window.width(), window.height()) == window.PREFERRED_SIZE
        # the smallest the window can be fits that laptop, title bar and
        # all (the crystal window's own test asks 730 px of its height)
        minimum = window.minimumSizeHint()
        assert minimum.width() <= 1366
        assert minimum.height() + 30 <= 720
    finally:
        dispose(window)
