"""The Setup page of the Model workspace (facet.ui.md_setup) and its presets
(facet.core.md_presets).

What these tests pin:

* every preset names analyses md_analysis runs and option fields that
  exist in their option dataclass, with text the command line's converter
  reads; Custom changes nothing;
* applying a preset is restricted to what the model allows (one frame, no
  velocities, no frame times), with a note for each restriction, and the
  formers are the preset's candidates among the cations present;
* the page ticks what a preset says, mirrors the real SetupPanel's boxes
  and reasons in its tree, flips to Custom on a manual edit, builds the
  same request as a SetupPanel given the same inputs, shows why a greyed
  analysis is greyed, detaches its footer, reports progress, and shows the
  timestep field inline when a preset needs a time axis the file lacks;
* no verdict word appears on the page, and it fits a 470 px column.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import dispose  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "md"
GLASS = DATA / "glass.extxyz"
NS3 = Path(r"C:\Users\samso\AppData\Local\Temp\claude\C--Users-samso"
           r"\342b7e11-8171-41a0-8eee-4126254c9cc5\scratchpad"
           r"\facet_md_descriptors\models\ns3_pedone\nvt300K.lammpstrj")
NS3_TYPES = {1: "Si", 2: "O", 3: "Na"}

VERDICT_WORDS = ("good", "bad", "poor", "excellent", "acceptable", "correct",
                 "incorrect", "wrong", "reliable", "unreliable", "trustworthy",
                 "untrustworthy", "unusable", "should", "proves", "confirms")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _open(path, options=None):
    from facet.ui import md_jobs

    opened = md_jobs.open_model(str(path), options or {})
    assert isinstance(opened, md_jobs.Opened), opened
    return opened


@pytest.fixture(scope="module")
def glass():
    return _open(GLASS)


@pytest.fixture(scope="module")
def ns3():
    if not NS3.exists():
        pytest.skip(f"{NS3} is not on this machine")
    return _open(NS3, {"type_map": NS3_TYPES})


@dataclasses.dataclass
class _Summary:
    """The few summary fields apply_preset reads, for a model on paper."""

    species: tuple
    n_frames: int = 1
    has_times: bool = False
    has_velocities: bool = False


def _texts(widget) -> list[str]:
    from PySide6.QtWidgets import (QAbstractButton, QComboBox, QLabel,
                                   QLineEdit, QTreeWidget, QWidget)

    out = []
    for w in [widget] + widget.findChildren(QWidget):
        out.append(w.toolTip())
        if isinstance(w, QLabel):
            out.append(w.text())
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QLineEdit):
            out.append(w.placeholderText())
        elif isinstance(w, QComboBox):
            out.extend(w.itemText(i) for i in range(w.count()))
            out.extend(str(w.itemData(i, 3) or "") for i in range(w.count()))
        elif isinstance(w, QTreeWidget):
            for i in range(w.topLevelItemCount()):
                top = w.topLevelItem(i)
                for item in [top] + [top.child(k)
                                     for k in range(top.childCount())]:
                    for column in range(w.columnCount()):
                        out.append(item.text(column))
                        out.append(item.toolTip(column))
    return out


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

def test_every_preset_names_real_analyses_and_real_fields():
    from facet.core import md_analysis as ma, md_presets
    from facet.ui.md_dialogs import field_keys, group_class

    try:
        from facet.md import cli
    except ImportError:          # the converter is checked where it exists
        cli = None
    names = [p.name for p in md_presets.PRESETS]
    assert names[0] == "Silicate" and names[-1] == "Custom"
    assert "Ion conduction / channels" in names and "Everything" in names
    for preset in md_presets.PRESETS:
        unknown = [a for a in preset.analyses if a not in ma.ANALYSES]
        assert not unknown, (preset.name, unknown)
        for key, text in preset.options.items():
            (group, name), = field_keys(key)
            cls = group_class(group)
            fields = {f.name: f for f in dataclasses.fields(cls)}
            assert name in fields, (preset.name, key)
            assert set(md_presets.option_readers(key)) & set(preset.analyses)
            if cli is not None:
                value = cli.convert(fields[name].type, text, key)
                cls(**{name: value})       # the dataclass takes it
        for need in preset.needs:
            assert need in md_presets.NEEDS, (preset.name, need)
    everything = md_presets.preset_named("Everything")
    assert set(everything.analyses) == set(ma.ANALYSES)
    custom = md_presets.apply_preset(md_presets.preset_named("Custom"),
                                     _Summary(("Si", "O")))
    assert custom.changes is False and custom.analyses == () \
        and not custom.options
    with pytest.raises(KeyError):
        md_presets.preset_named("Silicates")


def test_a_preset_is_restricted_to_what_the_model_allows():
    from facet.core import md_analysis as ma, md_presets

    dynamics = md_presets.preset_named("Dynamics")
    # a trajectory with frame times and velocities: everything stays
    full = md_presets.apply_preset(
        dynamics, _Summary(("Na", "Si", "O"), 20, True, True))
    assert set(full.analyses) == set(dynamics.analyses)
    assert full.formers == frozenset({"Si"})
    assert not any("time" in n.lower() for n in full.notes)
    # one frame: the analyses of time are left out, and said so
    one = md_presets.apply_preset(dynamics, _Summary(("Si", "O"), 1, False,
                                                     True))
    assert one.analyses == ()
    assert any("One frame" in n for n in one.notes)
    # no frame times, no velocities: the VACF and the temperature go, the
    # others stay with the timestep named
    bare = md_presets.apply_preset(dynamics, _Summary(("Si", "O"), 20))
    assert set(bare.analyses) == {"msd", "self-correlations",
                                  "bond-lifetimes"}
    assert any("no velocities" in n for n in bare.notes)
    assert any("timestep" in n for n in bare.notes)
    # the formers: candidates among the cations present, or a note
    phosphate = md_presets.preset_named("Phosphate")
    none = md_presets.apply_preset(phosphate, _Summary(("Na", "Si", "O")))
    assert none.formers == frozenset()
    assert any("left as they are" in n for n in none.notes)
    some = md_presets.apply_preset(phosphate, _Summary(("Na", "P", "O")))
    assert some.formers == frozenset({"P"})
    # options only for the analyses kept
    silicate = md_presets.apply_preset(md_presets.preset_named("Silicate"),
                                       _Summary(("Na", "Si", "O")))
    assert silicate.options == {"network.ring_criterion": "king",
                                "network.ring_max_size": "12",
                                "voids.radii": "vdw"}
    assert silicate.analyses == tuple(
        a for a in ma.ANALYSES if a in ("glass", "rings", "components",
                                        "polyhedral-sharing",
                                        "tetrahedral-order", "empty-spheres"))
    # the halide of an oxyfluoride, and its absence
    oxy = md_presets.preset_named("Oxyfluoride")
    assert any("Halide present: F" in n for n in md_presets.apply_preset(
        oxy, _Summary(("Al", "Si", "O", "F"))).notes)
    assert any("No halide" in n for n in md_presets.apply_preset(
        oxy, _Summary(("Al", "Si", "O"))).notes)
    # ion conduction: the void analyses, and a channels note or tick
    ion = md_presets.apply_preset(
        md_presets.preset_named("Ion conduction / channels"),
        _Summary(("Na", "Si", "O"), 20, True, False))
    assert {"voronoi", "empty-spheres", "free-volume"} <= set(ion.analyses)
    if "channels" in ma.ANALYSES:
        assert "channels" in ion.analyses
    else:
        assert any("channels" in n for n in ion.notes)


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def test_the_page_ticks_a_preset_and_an_edit_flips_to_custom(qapp, glass):
    from PySide6.QtCore import Qt
    from facet.ui.md_setup import SetupPage

    page = SetupPage(glass.summary, glass.trajectory)
    try:
        # nothing ticked at first: the formers are a model input
        assert page.preset.currentText() == "Custom"
        assert page.ticked() == [] and page.formers() == frozenset()
        assert not page.chips["Si"].isChecked()
        ok, reason = page.can_run()
        assert not ok and reason.startswith("Run needs")
        applied = page.apply_preset("Silicate")
        page.refresh()
        assert page.preset.currentText() == "Silicate"
        assert set(page.ticked()) == set(applied.analyses)
        assert page.chips["Si"].isChecked() and not page.chips["Na"].isChecked()
        assert page.formers() == frozenset({"Si"})
        assert page.groups["Structure"].text(1) == "5 of 9"
        assert page.groups["Voids and channels"].text(1).startswith("1 of")
        assert page.items["rings"].text(1) == "king, ≤ 12 nodes"
        assert page.items["empty-spheres"].text(1) == "vdw radii"
        ok, line = page.can_run()
        assert ok and line.startswith("Runs glass") and "formers Si" in line
        assert page.run_note.text() == line
        assert page.run_button.isEnabled()
        # a manual tick: the combo says Custom, the panel follows
        page.items["warren-cowley"].setCheckState(0, Qt.Checked)
        assert page.preset.currentText() == "Custom"
        page.refresh()
        assert "warren-cowley" in page.ticked()
        assert page.panel.box("warren-cowley").isChecked()
        # a chip: the picker follows, and the preset stays Custom
        page.apply_preset("Silicate")
        page.chips["Na"].setChecked(True)
        assert page.preset.currentText() == "Custom"
        assert page.panel.formers.ticked() == frozenset({"Si", "Na"})
        # a group tick ticks its enabled children only
        page.apply_preset("Silicate")
        page.groups["Dynamics"].setCheckState(0, Qt.Checked)
        page.refresh()
        timed = [n for n, g in page.question_of.items() if g == "Dynamics"]
        for name in timed:
            item = page.items[name]
            enabled = bool(item.flags() & Qt.ItemIsEnabled)
            assert (item.checkState(0) == Qt.Checked) == enabled, name
        assert page.preset.currentText() == "Custom"
    finally:
        dispose(page)


def test_the_request_equals_the_panel_s_for_the_same_inputs(qapp, glass):
    from facet.core import md_presets
    from facet.ui.md_dialogs import SetupPanel
    from facet.ui.md_setup import SetupPage

    page = SetupPage(glass.summary, glass.trajectory)
    panel = SetupPanel(glass.summary, glass.trajectory,
                       type_map=glass.summary.type_map)
    try:
        applied = page.apply_preset("Silicate")
        panel.formers.set_formers({"Si"})
        for key, text in applied.options.items():
            group, _, name = key.partition(".")
            panel.field(group or None, name).set_text(text)
        panel.set_analyses(list(applied.analyses))
        expected = panel.request()
        assert expected is not None, panel.problems()
        request = page.request()
        assert request == expected
        assert request.network.ring_criterion == "king"
        assert request.network.ring_max_size == 12
        assert request.voids.radii == "vdw"
        assert request.formers == frozenset({"Si"})
        # the page's spec is the panel's (the request file the window saves)
        assert page.panel.request_spec() == panel.request_spec()
        assert set(md_presets.preset_named("Silicate").analyses) == \
            set(request.analyses)
    finally:
        dispose(page, panel)


def test_greyed_analyses_say_why_and_a_refused_run_carries_the_footer_line(
        qapp, glass):
    from PySide6.QtCore import Qt
    from facet.core import md_analysis as ma
    from facet.ui.md_setup import SetupPage

    page = SetupPage(glass.summary, glass.trajectory)
    try:
        page.refresh()
        exafs = page.items["exafs"]
        assert not (exafs.flags() & Qt.ItemIsEnabled)
        assert exafs.text(1) == "needs absorber"
        assert "Greyed: needs" in exafs.toolTip(0)
        assert "absorber" in exafs.toolTip(1)
        comparison = page.items["scattering-comparison"]
        assert comparison.text(1).startswith("needs ")
        assert "measured curves" in comparison.toolTip(0)
        # the panel's reasons are the tree's
        for name, reasons in page.panel.reasons().items():
            item = page.items[name]
            assert bool(item.flags() & Qt.ItemIsEnabled) == (not reasons), name
            if reasons:
                assert item.text(1).startswith("needs "), name
        # nothing ticked: the footer says so, and request() raises it
        ok, line = page.can_run()
        assert not ok and "analysis ticked" in line
        with pytest.raises(ma.RequestError) as caught:
            page.request()
        assert str(caught.value) == line
        assert caught.value.missing
        # without a former, glass is greyed with that need; a tick in the
        # tree cannot reach it, and a chip answers it
        glass_item = page.items["glass"]
        assert not (glass_item.flags() & Qt.ItemIsEnabled)
        assert glass_item.text(1) == "needs network formers"
        glass_item.setCheckState(0, Qt.Checked)
        page.refresh()
        assert glass_item.checkState(0) == Qt.Unchecked
        page.chips["Si"].setChecked(True)
        page.refresh()
        assert bool(glass_item.flags() & Qt.ItemIsEnabled)
        glass_item.setCheckState(0, Qt.Checked)
        page.refresh()
        ok, line = page.can_run()
        assert ok and line.startswith("Runs glass on 2 frame(s); formers Si")
        assert page.request().analyses == ("glass",)
        # the field a double-click asks for
        asked = []
        page.fieldRequested.connect(asked.append)
        page._on_double_click(page.items["exafs"], 1)
        assert asked == ["exafs.absorber"]
        assert page.reveal("frames") and page.reveal("timestep_fs")
        assert page.more.button.isChecked()
        assert not page.reveal("network.ring_criterion")
        assert asked[-1] == "network.ring_criterion"
    finally:
        dispose(page)


def test_the_footer_detaches_and_reports_progress(qapp, glass):
    from PySide6.QtWidgets import QWidget
    from facet.ui.md_setup import SetupPage

    page = SetupPage(glass.summary, glass.trajectory)
    try:
        footer = page.detach_footer()
        assert isinstance(footer, QWidget) and footer.parent() is None
        assert footer.isAncestorOf(page.run_button)
        assert footer.isAncestorOf(page.progress) and \
            footer.isAncestorOf(page.cancel_button) and \
            footer.isAncestorOf(page.run_note)
        assert not page.isAncestorOf(page.run_button)
        page.apply_preset("Silicate")
        page.refresh()
        assert page.run_button.isEnabled()
        page.set_running(True)
        assert not page.run_button.isEnabled()
        assert not page.progress.isHidden() and not page.cancel_button.isHidden()
        assert "in progress" in page.run_note.text()
        ok, line = page.can_run()
        assert not ok and "in progress" in line
        page.set_progress(3, 10, "pass 1, frame 3 of 10")
        assert page.progress.value() == 3 and page.progress.maximum() == 10
        assert page.progress.format() == "30 %  ·  pass 1, frame 3 of 10"
        cancelled = []
        page.cancelRequested.connect(lambda: cancelled.append(True))
        page.cancel_button.click()
        assert cancelled == [True]
        page.set_running(False)
        assert page.run_button.isEnabled() and page.progress.isHidden()
        assert page.run_note.text().startswith("Runs ")
        runs = []
        page.runRequested.connect(lambda: runs.append(True))
        page.run_button.click()
        assert runs == [True]
    finally:
        dispose(page, footer)


def test_the_timestep_field_comes_inline_when_a_preset_needs_a_time_axis(
        qapp, ns3):
    from PySide6.QtCore import Qt
    from facet.ui.md_setup import SetupPage

    assert not ns3.summary.has_times
    page = SetupPage(ns3.summary, ns3.trajectory)
    try:
        page.show()
        qapp.processEvents()
        assert not page._timestep_inline
        assert page.more_time_holder.isAncestorOf(page.timestep_field)
        applied = page.apply_preset("Ion conduction / channels")
        page.refresh()
        assert page._timestep_inline
        assert page.time_holder.isAncestorOf(page.timestep_field)
        assert not page.time_widget.isHidden()
        assert page.time_note.text().startswith("needs a time axis")
        assert any("no frame times" in n for n in applied.notes)
        assert page.preset_notes == applied.notes
        assert {"voronoi", "empty-spheres", "free-volume", "glass"} <= \
            set(page.ticked())
        msd = page.items["msd"]
        assert not (msd.flags() & Qt.ItemIsEnabled)
        assert msd.text(1) == "needs MD timestep (fs)"
        # typing the timestep enables the analyses of time, still wanted
        page.timestep_field.set_text("1.0")
        page.refresh()
        assert page.preset.currentText() == "Custom"
        assert bool(msd.flags() & Qt.ItemIsEnabled)
        assert msd.checkState(0) == Qt.Checked
        assert page.request().timestep_fs == 1.0
        assert "msd" in page.request().analyses
        # a preset without that need puts the field back under More…
        page.apply_preset("Silicate")
        assert not page._timestep_inline
        assert page.more_time_holder.isAncestorOf(page.timestep_field)
        assert page.time_widget.isHidden()
    finally:
        dispose(page)


def test_no_verdict_word_appears_on_the_page_and_it_fits_the_column(qapp,
                                                                     glass):
    from facet.ui import chrome
    from facet.ui.md_setup import SetupPage

    pattern = re.compile(r"\b(" + "|".join(VERDICT_WORDS) + r")\b",
                         re.IGNORECASE)
    page = SetupPage(glass.summary, glass.trajectory)
    area = chrome.in_scroll_area(page)
    try:
        page.apply_preset("Everything")
        page.refresh()
        page.more.button.setChecked(True)
        found = {}
        for text in _texts(page) + _texts(page.footer):
            for match in pattern.finditer(text or ""):
                found.setdefault(match.group(0).lower(), text[:160])
        assert not found, found
        area.resize(470, 700)
        area.show()
        qapp.processEvents()
        assert page.minimumSizeHint().width() <= 470
        tree = page.tree
        assert tree.columnWidth(0) + tree.columnWidth(1) <= \
            tree.viewport().width() + 1
        assert tree.columnWidth(0) >= 200
        assert page.width() <= 470
    finally:
        dispose(area)


def test_apply_theme_recolours_the_page(qapp, glass):
    from facet.core import theme as theme_mod
    from facet.ui.md_setup import SetupPage

    page = SetupPage(glass.summary, glass.trajectory)
    try:
        light = page.styleSheet()
        page.apply_theme(theme_mod.dark())
        assert page.styleSheet() != light
        assert page.panel.theme is not None
        assert np.all([g.theme is page.panel.theme
                       for g in page.panel.groups.values()])
    finally:
        dispose(page)
