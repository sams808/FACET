"""The engine batch: an estimate with the progress, partial results on
cancel, the frame cache on disk, workers for the pair searches, bond
valences binned by pair, composition by region and the centre of mass
per element.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from facet.core import bulk, bv, md_analysis as ma, md_cache, md_model, md_readers

GLASS = Path(__file__).resolve().parent / "data" / "md" / "glass.extxyz"


def _request(**extra):
    return ma.AnalysisRequest(analyses=("glass",), formers=frozenset({"Si"}),
                              **extra)


@pytest.fixture(scope="module")
def glass_result():
    trajectory = md_readers.read_trajectory(str(GLASS))
    return ma.analyse(trajectory, _request())


def test_the_glass_pass_records_valences_regions_and_centres(glass_result):
    tables = glass_result.outputs["glass"].tables
    names = set(tables)
    assert "bond valence Si-O (BV)" in names
    assert "Si by region (BV)" in names and "Si segregation (BV)" in names
    assert "centre of mass Na x (BV)" in names
    valence = tables["bond valence Si-O (BV)"]
    assert np.asarray(valence.edges)[0] == 0.0
    region = tables["O by region (BV)"]
    assert len(region.keys) == 8
    # every atom of the element sits in exactly one cell
    assert np.asarray(region.per_frame).sum(axis=1).tolist() == [3, 3]


def test_a_cancel_after_one_frame_keeps_a_partial_result():
    trajectory = md_readers.read_trajectory(str(GLASS))
    calls = []

    def cancelled() -> bool:
        calls.append(1)
        return len(calls) > 1          # true once a frame is in

    result = ma.analyse(trajectory, _request(), cancelled=cancelled)
    notes = " ".join(result.model_notes) if hasattr(result, "model_notes") \
        else " ".join(str(n) for n in getattr(result, "notes", ()))
    assert "partial" in notes
    assert "glass" in result.outputs


def test_the_frame_cache_round_trips_a_table(tmp_path):
    trajectory = md_readers.read_trajectory(str(GLASS))
    frame = trajectory.frame(0)
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT, v_bond_vu=0.075,
                                  v_list_vu=0.02)
    assert md_cache.load(GLASS, 0, bv.DEFAULT, 0.02) is None
    written = md_cache.save(GLASS, 0, table, 0.02)
    assert written is not None and written.exists()
    again = md_cache.load(GLASS, 0, bv.DEFAULT, 0.02)
    assert again is not None
    for name in ("nbr", "d_ang", "v_vu", "cum_v_vu"):
        assert np.array_equal(np.asarray(getattr(again, name)),
                              np.asarray(getattr(table, name)), equal_nan=True)
    assert again.params.params_name == table.params.params_name
    assert again.v_list_vu == pytest.approx(table.v_list_vu)
    # another threshold or frame misses
    assert md_cache.load(GLASS, 1, bv.DEFAULT, 0.02) is None
    assert md_cache.load(GLASS, 0, bv.DEFAULT, 0.03) is None
    # the numbers read from the cached table are the table's
    assert bulk.at_threshold(again, 0.075).cn.tolist() == \
        bulk.at_threshold(table, 0.075).cn.tolist()
    assert md_cache.clear() == 1
    assert md_cache.load(GLASS, 0, bv.DEFAULT, 0.02) is None


def test_the_frame_view_reads_the_cache_the_second_time(qapp):
    from facet.ui import md_jobs
    trajectory = md_readers.read_trajectory(str(GLASS))
    md_cache.clear()
    first = md_jobs.load_frame_view(trajectory, 0, {}, bv.DEFAULT, 0.075, 0.02,
                                    with_bonds=True)
    assert "frame cache" not in first.note
    second = md_jobs.load_frame_view(trajectory, 0, {}, bv.DEFAULT, 0.075, 0.02,
                                     with_bonds=True)
    assert "frame cache" in second.note
    assert np.array_equal(second.table.nbr, first.table.nbr)
    md_cache.clear()


def test_workers_prefetch_the_tables_for_the_own_pass():
    trajectory = md_readers.read_trajectory(str(GLASS))
    request = ma.AnalysisRequest(analyses=("coordination-sequences",),
                                 formers=frozenset({"Si"}), workers=2,
                                 network=ma.NetworkOptions(n_shells=3))
    result = ma.analyse(trajectory, request)
    assert "coordination-sequences" in result.outputs
    serial = ma.analyse(trajectory, ma.AnalysisRequest(
        analyses=("coordination-sequences",), formers=frozenset({"Si"}),
        network=request.network))
    a = result.outputs["coordination-sequences"].tables
    b = serial.outputs["coordination-sequences"].tables
    assert set(a) == set(b)


def test_the_cli_accepts_workers(tmp_path):
    from facet.md.cli import main
    code = main(["analyse", str(GLASS), "--formers", "Si", "--only", "glass",
                 "--workers", "2", "--out", str(tmp_path / "out")])
    assert code == 0
    assert any(p.name.startswith("glass__bond_valence") for p in (tmp_path / "out").iterdir())


def test_the_progress_line_carries_an_estimate(qapp, monkeypatch):
    from facet.core import prefs
    prefs.set("tour_offered", True)
    from facet.ui import preview
    from conftest import dispose
    win = preview.PreviewWindow()
    try:
        win.load_many([str(GLASS)])
        end = time.monotonic() + 30
        c = win.project.entries[0].controller
        while time.monotonic() < end and c.summary is None:
            qapp.processEvents()
            time.sleep(0.05)
        messages = []
        c.statusMessage.connect(messages.append)

        class Job:
            cancel_requested = False

            @staticmethod
            def elapsed_s():
                return 10.0
        c._run_job = Job()
        c._on_run_progress(2, 10, "frames")
        assert any("about" in m and "left" in m for m in messages)
        c._run_job = None
    finally:
        win.close()
        dispose(win)
