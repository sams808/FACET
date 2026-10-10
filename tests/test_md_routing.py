"""MD files reach the Model workspace from everywhere the window opens a file.

A dropped ``.lammpstrj`` or ``.data`` used to be refused by the drag cursor
with no message at all, and File > Open gave a "could not be read" box
pointing at a Python function. These tests drive the real main window
(``PreviewWindow``): a drop through ``QDropEvent``, File > Open and File >
Open MD model through their menu actions with the file dialog answered, and
the command line through the window's constructor, for a LAMMPS dump, a
data file, DCD, XTC, a multi-frame extended XYZ, a plain XYZ trajectory
with no box, and the other formats the reader registers. Each becomes an
MD model entry of the project, listed in the Structures dock, while the
crystal side of the window stays exactly as it was.

Two layers. The routing tests put a recording stand-in for
``md_workspace.ModelController`` in place, so they check exactly what the
window hands over (which file, as one model or a series, with which read
options) whatever the reader does. The tests at the end build the real
controller: the workspace shown, the read panel for a dump without
elements, and a data file offered to the dump dropped with it.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest  # noqa: E402

from conftest import dispose  # noqa: E402

DATA = Path(__file__).parent / "data"
MD = DATA / "md"
CIF = DATA / "crystals" / "quartz_SiO2_cod9013321.cif"

# (label, path): one file of each kind the task names, and the others the
# reader registers, all from tests/data (never the scratchpad models)
MD_FILES = [
    ("lammps dump", MD / "dump_ortho_real.lammpstrj"),
    ("lammps data", MD / "data_charge.data"),
    ("dcd", MD / "lammps_formats" / "traj.dcd"),
    ("xtc", MD / "xtc_lammps.xtc"),
    ("multi-frame extxyz", MD / "glass.extxyz"),
    ("cp2k xyz, no box", MD / "recognition" / "NS2-pos-1.xyz"),
    ("xdatcar", MD / "XDATCAR_fixed"),
    ("dl_poly history", MD / "HISTORY"),
    ("dl_poly config", MD / "CONFIG"),
    ("gzip dump", MD / "dump_tri_x.lammpstrj.gz"),
    ("lammps yaml", MD / "lammps_formats" / "traj.yaml"),
    ("lammps binary", MD / "lammps_formats" / "traj_atom.bin"),
    ("gsd", MD / "binary" / "hoomd_small.gsd"),
    ("netcdf", MD / "binary" / "lammps_style.nc"),
    ("ase traj", MD / "binary" / "ase_md.traj"),
    ("castep md", MD / "text" / "ase.md"),
    ("gro", MD / "text" / "ase_vel.gro"),
    ("axsf", MD / "text" / "ase_var.axsf"),
    ("pdb models", MD / "text" / "ase_models.pdb"),
    ("imd", MD / "text" / "ovito.imd"),
]
IDS = [label for label, _ in MD_FILES]
PATHS = [path for _, path in MD_FILES]

CRYSTAL_TABS = ["Site", "Tools", "Diffraction", "PDF", "EXAFS", "Planes",
                "Styles", "Volume", "Disorder", "Theme"]
MODEL_TABS = ["Setup", "Results", "Highlight", "Notes"]


def _plain_xyz(path: Path, frames: int = 2, comment: str = " Atoms. Timestep: {}"
               ) -> Path:
    """A plain XYZ trajectory as LAMMPS 'dump xyz' writes it: no Lattice=."""
    rows = [("Si", 0.0, 0.0, 0.0), ("O", 1.6, 0.0, 0.0), ("O", 0.0, 1.6, 0.0)]
    lines = []
    for k in range(frames):
        lines += [str(len(rows)), comment.format(100 * k)]
        lines += [f"{e} {x + 0.01 * k:.4f} {y:.4f} {z:.4f}" for e, x, y, z in rows]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

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


class _Recorder:
    """A stand-in for facet.ui.md_workspace: a controller that records what
    it was built with and offers the window the five widgets, the signals
    and the state the window reads, reading no file."""

    def __init__(self):
        from PySide6.QtCore import QObject, Signal
        from PySide6.QtWidgets import QWidget

        recorder = self

        class ModelController(QObject):
            opened = Signal(object)
            finished = Signal(object)
            frameShown = Signal(int)
            statusMessage = Signal(str)
            stateChanged = Signal()

            def __init__(self, entry, window=None, *, theme=None,
                         read_options=None):
                super().__init__(window)
                self.entry = entry
                entry.controller = self
                self.window = window
                self.read_options = (dict(read_options) if read_options
                                     is not None else dict(entry.read_options))
                self.trajectory = self.summary = self.problem = None
                self.page = self.result = None
                self.complete_result = self.partial_result = None
                self.themes = []
                self.shut = False
                self._widgets = tuple(QWidget() for _ in range(5))
                recorder.calls.append((entry.path, window, read_options))
                recorder.controllers.append(self)

            @property
            def widgets(self):
                return self._widgets

            def is_running(self):
                return False

            def can_run(self):
                return False, "no model is read"

            def figure_shown(self):
                return False

            def rows_shown(self):
                return False

            def status_line(self):
                return "stand-in"

            def current_tab(self):
                return ""

            def canvas_shown(self):
                return "reading"

            def apply_theme(self, theme):
                self.themes.append(theme)

            def shutdown(self, wait_ms=None):
                self.shut = True
                return True

            def dispose(self):
                for w in self._widgets:
                    w.deleteLater()

        self.ModelController = ModelController
        self.calls: list[tuple] = []
        self.controllers: list = []

    def opened(self) -> list:
        """The paths of every call, a series as a list of names."""
        return [[Path(p).name for p in paths] if isinstance(paths, list)
                else Path(paths).name for paths, _window, _o in self.calls]


@pytest.fixture
def recorder(monkeypatch, qapp):
    import facet.ui

    rec = _Recorder()
    stub = types.ModuleType("facet.ui.md_workspace")
    stub.ModelController = rec.ModelController
    stub.CLOSE_WAIT_MS = 3000
    stub.ATOMS_ONLY_ABOVE = 5000
    monkeypatch.setitem(sys.modules, "facet.ui.md_workspace", stub)
    monkeypatch.setattr(facet.ui, "md_workspace", stub, raising=False)
    yield rec


@pytest.fixture
def warnings(monkeypatch):
    """Every QMessageBox.warning / information the window raises, recorded
    instead of shown (a modal box would hang the test)."""
    from PySide6.QtWidgets import QMessageBox

    seen: list[tuple[str, str]] = []

    def record(_parent, title, text, *args, **kwargs):
        seen.append((title, text))
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(record))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(record))
    return seen


@pytest.fixture
def window(qapp, recorder, warnings):
    from facet.ui.preview import PreviewWindow

    w = PreviewWindow()
    w.show()
    yield w
    dispose(w)


def _mime(paths):
    from PySide6.QtCore import QMimeData, QUrl

    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return data


def _drag_enter(window, paths) -> bool:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QDragEnterEvent

    data = _mime(paths)
    event = QDragEnterEvent(QPoint(50, 50), Qt.CopyAction, data,
                            Qt.LeftButton, Qt.NoModifier)
    event.ignore()
    window.dragEnterEvent(event)
    return event.isAccepted()


def _drop(window, paths) -> bool:
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QDropEvent

    data = _mime(paths)
    event = QDropEvent(QPointF(50, 50), Qt.CopyAction, data, Qt.LeftButton,
                       Qt.NoModifier)
    event.ignore()
    window.dropEvent(event)
    return event.isAccepted()


def _action(window, menu_title: str, text: str):
    for top in window.menuBar().actions():
        if top.text().replace("&", "") == menu_title:
            for action in top.menu().actions():
                if action.text().replace("&", "") == text:
                    return action
    raise AssertionError(f"no {menu_title} > {text}")


def _answer_dialog(monkeypatch, paths, seen_filters=None):
    from PySide6.QtWidgets import QFileDialog

    def answer(_parent, _caption="", _dir="", filt="", *args, **kwargs):
        if seen_filters is not None:
            seen_filters.append(filt)
        return [str(p) for p in paths], ""

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(answer))


def _crystal_tabs(window):
    tabs = window.crystal_tabs
    return [tabs.tabText(i) for i in range(tabs.count())]


def _models(window):
    return [e for e in window.project.entries if e.kind == "model"]


# ---------------------------------------------------------------------------
# the routing rule itself (no window)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", PATHS, ids=IDS)
def test_every_md_test_file_routes_to_a_model(path):
    from facet.ui.preview import read_or_route

    assert path.is_file(), path
    kind, value = read_or_route(path)
    assert kind == "md", (path.name, kind, value)
    assert isinstance(value, str) and value


def test_a_crystal_still_reads_as_a_crystal():
    from facet.core.structure import Structure
    from facet.ui.preview import read_or_route

    kind, value = read_or_route(CIF)
    assert kind == "structure"
    assert isinstance(value, Structure)


def test_md_routing_agrees_with_the_reader_wherever_the_reader_routes(tmp_path):
    """Every file readers.read refuses with MDModelFile is routed, with the
    format the reader named; the router adds only refusals the workspace
    opens (a boxless XYZ trajectory)."""
    from facet.core import readers
    from facet.ui.preview import read_or_route

    for path in PATHS + [CIF]:
        try:
            readers.read(path)
            expected = None
        except readers.MDModelFile as error:
            expected = error.file_format
        except readers.UnsupportedFormat:
            expected = "refused"
        kind, value = read_or_route(path)
        if expected is None:
            assert kind == "structure", path.name
        elif expected != "refused":
            assert (kind, value) == ("md", expected), path.name


def test_a_boxless_xyz_trajectory_and_a_dump_xyz_frame_route(tmp_path):
    from facet.ui.preview import read_or_route

    assert read_or_route(_plain_xyz(tmp_path / "traj.xyz")) == ("md", "extxyz")
    one = _plain_xyz(tmp_path / "one.xyz", frames=1)
    assert read_or_route(one) == ("md", "extxyz")


def test_an_empty_dump_opens_where_its_reader_can_say_so(tmp_path):
    from facet.ui.preview import read_or_route

    empty = tmp_path / "run.lammpstrj"
    empty.write_bytes(b"")
    assert read_or_route(empty) == ("md", "lammps-dump")


def test_unreadable_files_stay_failures(tmp_path):
    from facet.ui.preview import read_or_route

    junk = tmp_path / "notes.txt"
    junk.write_text("nothing structural here\n")
    kind, why = read_or_route(junk)
    assert kind == "failed" and "notes.txt" in why
    kind, why = read_or_route(tmp_path / "missing.cif")
    assert kind == "failed"


def test_series_grouping_takes_only_one_snapshot_per_file_formats(tmp_path):
    from facet.ui.preview import model_groups

    cfg = sorted((MD / "lammps_formats" / "cfg").glob("dump.*.cfg"),
                 reverse=True)
    groups = model_groups([(str(p), "atomeye-cfg") for p in cfg])
    assert len(groups) == 1
    assert [Path(p).name for p in groups[0]] == [
        "dump.100.cfg", "dump.110.cfg", "dump.120.cfg"]
    runs = [(str(tmp_path / "glass_300K.lammpstrj"), "lammps-dump"),
            (str(tmp_path / "glass_600K.lammpstrj"), "lammps-dump")]
    assert model_groups(runs) == [runs[0][0], runs[1][0]]
    assert model_groups([runs[0], runs[0]]) == [runs[0][0]]


def test_the_md_file_filter_lists_every_registered_format():
    from facet.core import md_readers
    from facet.ui.preview import md_file_filter

    filt = md_file_filter()
    parts = filt.split(";;")
    assert parts[0].startswith("MD models and trajectories (")
    assert parts[-1] == "All files (*)"
    for spec in md_readers.format_specs():
        for ext in spec.extensions:
            assert f"*{ext}" in parts[0], (spec.name, ext)
    for pattern in ("*.lammpstrj", "*.data", "*.extxyz", "XDATCAR*",
                    "HISTORY*", "CONFIG*", "*.dcd", "*.xtc"):
        assert pattern in parts[0], pattern


# ---------------------------------------------------------------------------
# the project's model entry
# ---------------------------------------------------------------------------

def test_a_model_entry_is_listed_and_left_alone_by_the_crystal_paths():
    from facet.core import project as project_mod, readers

    project = project_mod.Project()
    crystal = project.add(readers.read(CIF), str(CIF))
    model = project.add_model(str(MD / "glass.extxyz"), {"units": "metal"})
    assert model.kind == "model" and crystal.kind == "crystal"
    assert model.structure is None and model.selected_site is None
    assert model.name.splitlines() == ["glass.extxyz", "reading…"]
    assert "MD model" in model.label
    assert project.crystals == [crystal] and project.models == [model]
    project.set_active(1)
    assert project.current is model
    assert project.visible_entries == []          # a model is not drawn here
    assert project.results_for(model) == []
    assert all(r["structure"] == crystal.name for r in project.site_table())
    project.set_overlay(True, 5.0)                # lays out the crystals only
    assert project.visible_entries == [crystal]
    series = project.add_model([str(MD / "CONFIG"), str(MD / "HISTORY")])
    assert series.name.startswith("CONFIG .. HISTORY")
    assert series.source_path.endswith("CONFIG")
    project.remove(1)
    assert project.models == [series]


# ---------------------------------------------------------------------------
# drag and drop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", PATHS, ids=IDS)
def test_dragging_an_md_file_is_accepted(window, path):
    assert _drag_enter(window, [path])


def test_dragging_a_renamed_dump_is_accepted_by_its_content(window, tmp_path):
    renamed = tmp_path / "dump.glass"
    renamed.write_bytes((MD / "dump_ortho_real.lammpstrj").read_bytes())
    assert _drag_enter(window, [renamed])


def test_dragging_a_file_nothing_reads_is_still_refused(window, tmp_path):
    junk = tmp_path / "picture.bmp"
    junk.write_bytes(b"BM" + b"\xff" * 64)
    assert not _drag_enter(window, [junk])


@pytest.mark.parametrize("path", PATHS, ids=IDS)
def test_dropping_an_md_file_becomes_a_model_entry(window, recorder, warnings,
                                                  path):
    assert _drop(window, [path])
    assert recorder.opened() == [path.name]
    _paths, parent, read_options = recorder.calls[0]
    assert parent is window and read_options is None
    models = _models(window)
    assert len(models) == 1 and len(window.project) == 1
    assert models[0].controller is recorder.controllers[0]
    # the model is the row shown: its widgets are current in the five stacks
    assert window.project.current is models[0]
    controller = recorder.controllers[0]
    for widget, stack in zip(controller.widgets, window._stacks()):
        assert stack.currentWidget() is widget
    assert window.sites_dock.windowTitle() == "Elements"
    assert warnings == []
    # and the crystal side keeps its ten tabs, untouched
    assert _crystal_tabs(window) == CRYSTAL_TABS
    assert window.structure_panel.list.count() == 1
    assert "reading" in window.structure_panel.list.item(0).text()


def test_dropping_a_plain_xyz_trajectory_lists_a_model(
        window, recorder, warnings, tmp_path):
    path = _plain_xyz(tmp_path / "lammps_dump_xyz.xyz")
    assert _drop(window, [path])
    assert recorder.opened() == [path.name]
    assert warnings == []


def test_a_mixed_drop_loads_the_cif_and_lists_the_dump(window, recorder,
                                                       warnings):
    dump = MD / "dump_ortho_real.lammpstrj"
    assert _drop(window, [CIF, dump])
    assert recorder.opened() == [dump.name]
    assert [e.kind for e in window.project.entries] == ["crystal", "model"]
    # the crystal stays the one shown, as before
    assert window.project.current.kind == "crystal"
    assert window.structure is not None
    assert sorted(window.structure.elements_present) == ["O", "Si"]
    for stack in window._stacks():
        assert stack.currentIndex() == 0
    assert window.sites_dock.windowTitle() == "Sites"
    assert warnings == []
    assert _crystal_tabs(window) == CRYSTAL_TABS


def test_switching_rows_switches_the_five_regions(window, recorder, warnings,
                                                  qapp):
    _drop(window, [CIF, MD / "dump_ortho_real.lammpstrj"])
    controller = recorder.controllers[0]
    window.structure_panel.list.setCurrentRow(1)
    qapp.processEvents()
    assert window.project.current.kind == "model"
    for widget, stack in zip(controller.widgets, window._stacks()):
        assert stack.currentWidget() is widget
    assert window.sites_dock.windowTitle() == "Elements"
    assert "MD model" in window.windowTitle()
    window.structure_panel.list.setCurrentRow(0)
    qapp.processEvents()
    assert window.project.current.kind == "crystal"
    for stack in window._stacks():
        assert stack.currentIndex() == 0
    assert window.sites_dock.windowTitle() == "Sites"
    assert window.scene is not None


def test_removing_a_model_row_shuts_it_and_frees_its_pages(window, recorder,
                                                           warnings, qapp):
    _drop(window, [CIF, MD / "dump_ortho_real.lammpstrj"])
    controller = recorder.controllers[0]
    assert all(stack.count() == 2 for stack in window._stacks())
    window._on_remove(1)
    qapp.processEvents()
    assert controller.shut
    assert _models(window) == []
    assert all(stack.count() == 1 for stack in window._stacks())
    assert window.project.current.kind == "crystal"


def test_a_drop_of_cfg_snapshots_is_one_model(window, recorder, warnings):
    cfg = sorted((MD / "lammps_formats" / "cfg").glob("dump.*.cfg"))
    assert _drop(window, cfg[::-1])
    assert recorder.opened() == [[p.name for p in cfg]]
    assert len(_models(window)) == 1
    assert isinstance(_models(window)[0].path, list)


def test_a_drop_of_two_dumps_is_two_models(window, recorder, warnings):
    dumps = [MD / "dump_ortho_real.lammpstrj", MD / "dump_tri_x.lammpstrj"]
    assert _drop(window, dumps)
    assert recorder.opened() == [p.name for p in dumps]
    assert len(_models(window)) == 2
    assert all(stack.count() == 3 for stack in window._stacks())


def test_a_drop_with_an_unreadable_file_still_says_so(window, recorder,
                                                      warnings, tmp_path):
    junk = tmp_path / "junk.xyz"
    junk.write_text("this is not a structure\n")
    dump = MD / "dump_ortho_real.lammpstrj"
    assert _drop(window, [dump, junk])
    assert recorder.opened() == [dump.name]
    assert len(warnings) == 1
    title, text = warnings[0]
    assert "junk.xyz" in text and "1 listed as MD models" in text


def test_a_data_file_dropped_with_its_dump_is_offered_as_a_companion(
        window, recorder, warnings):
    dump = MD / "dump_tri_x.lammpstrj"
    data = MD / "data_atomic.data"
    assert _drop(window, [dump, data])
    by_name = {Path(e.source_path).name: e for e in _models(window)}
    assert set(by_name) == {dump.name, data.name}
    assert [Path(p) for p in by_name[dump.name].companions] == [data]
    assert by_name[data.name].companions == ()


# ---------------------------------------------------------------------------
# File > Open, File > Open MD model, File > Open MD series
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", PATHS, ids=IDS)
def test_file_open_lists_a_model(window, recorder, warnings, monkeypatch, path):
    _answer_dialog(monkeypatch, [path])
    _action(window, "File", "Open…").trigger()
    assert recorder.opened() == [path.name]
    assert warnings == []
    assert [e.kind for e in window.project.entries] == ["model"]


def test_file_open_md_model_lists_the_md_formats(window, recorder, warnings,
                                                 monkeypatch):
    from facet.ui.preview import md_file_filter

    filters: list[str] = []
    dump = MD / "dump_ortho_real.lammpstrj"
    _answer_dialog(monkeypatch, [dump, MD / "xtc_lammps.xtc"], filters)
    action = _action(window, "File", "Open MD model…")
    assert action is window.open_md_action
    action.trigger()
    assert filters == [md_file_filter()]
    assert recorder.opened() == [dump.name, "xtc_lammps.xtc"]
    assert warnings == []
    # the first of them is the row shown
    assert window.project.current is _models(window)[0]


def test_file_open_md_model_takes_a_one_frame_extxyz_as_a_model(
        window, recorder, warnings, monkeypatch, tmp_path):
    """The crystal reader reads one frame with Lattice= as a crystal; chosen
    as an MD model it is listed as a one-frame model."""
    lines = (MD / "glass.extxyz").read_text().splitlines()
    one = tmp_path / "one_frame.extxyz"
    one.write_text("\n".join(lines[:7]) + "\n")
    _answer_dialog(monkeypatch, [one])
    _action(window, "File", "Open MD model…").trigger()
    assert recorder.opened() == [one.name]


def test_file_open_md_model_sends_a_cif_to_the_crystal_side(
        window, recorder, warnings, monkeypatch):
    _answer_dialog(monkeypatch, [CIF])
    _action(window, "File", "Open MD model…").trigger()
    assert recorder.calls == []
    assert [e.kind for e in window.project.entries] == ["crystal"]


def test_file_open_md_series_reads_the_files_as_one_model(
        window, recorder, warnings, monkeypatch, tmp_path):
    source = (MD / "dump_ortho_real.lammpstrj").read_bytes()
    names = ["dump.100.lammpstrj", "dump.20.lammpstrj", "dump.3.lammpstrj"]
    for name in names:
        (tmp_path / name).write_bytes(source)
    _answer_dialog(monkeypatch, [tmp_path / n for n in names])
    _action(window, "File", "Open MD series as one model…").trigger()
    assert recorder.opened() == [["dump.3.lammpstrj", "dump.20.lammpstrj",
                                  "dump.100.lammpstrj"]]


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [MD / "dump_ortho_real.lammpstrj",
                                  MD / "data_charge.data",
                                  MD / "xtc_lammps.xtc"],
                         ids=["dump", "data", "xtc"])
def test_a_command_line_md_file_is_listed_and_shown(qapp, recorder, warnings,
                                                    path):
    """app.main and preview.main hand their argument to PreviewWindow(path)."""
    from facet.ui.preview import PreviewWindow

    w = PreviewWindow(str(path))
    try:
        assert recorder.opened() == [path.name]
        assert recorder.calls[0][1] is w
        assert [e.kind for e in w.project.entries] == ["model"]
        assert w.project.current.kind == "model"
        assert warnings == []
    finally:
        dispose(w)


def test_a_command_line_cif_still_opens_as_a_crystal(qapp, recorder, warnings):
    from facet.ui.preview import PreviewWindow

    w = PreviewWindow(str(CIF))
    try:
        assert recorder.calls == []
        assert [e.kind for e in w.project.entries] == ["crystal"]
        assert _crystal_tabs(w) == CRYSTAL_TABS
    finally:
        dispose(w)


# ---------------------------------------------------------------------------
# no silence: a workspace that cannot be built says so
# ---------------------------------------------------------------------------

def test_a_build_without_the_workspace_says_so(qapp, monkeypatch, warnings):
    import facet.ui
    from facet.ui.preview import PreviewWindow

    monkeypatch.setitem(sys.modules, "facet.ui.md_workspace", None)
    monkeypatch.delattr(facet.ui, "md_workspace", raising=False)
    w = PreviewWindow()
    try:
        _drop(w, [MD / "dump_ortho_real.lammpstrj"])
        assert len(warnings) == 1
        assert "dump_ortho_real.lammpstrj" in warnings[0][1]
        assert "-m facet.md" in warnings[0][1]
        assert len(w.project) == 0
    finally:
        dispose(w)


def test_a_controller_that_raises_is_reported_and_the_row_removed(
        window, recorder, warnings, monkeypatch):
    def broken(entry, window=None, **kwargs):
        raise RuntimeError("could not build")

    monkeypatch.setattr(sys.modules["facet.ui.md_workspace"],
                        "ModelController", broken)
    _drop(window, [MD / "dump_ortho_real.lammpstrj"])
    assert len(warnings) == 1
    assert "could not build" in warnings[0][1]
    assert len(window.project) == 0


# ---------------------------------------------------------------------------
# the menus and the empty view
# ---------------------------------------------------------------------------

def _bar_action(window, title: str):
    """The menu bar's action for ``title``: held while its menu is read,
    since PySide6 invalidates the menu's wrapper once the action's wrapper
    is collected."""
    for action in window.menuBar().actions():
        if action.text().replace("&", "") == title:
            return action
    raise AssertionError(f"no menu {title}")


def _rows(window, title: str) -> list[str]:
    action = _bar_action(window, title)
    return [a.text().replace("&", "") for a in action.menu().actions()
            if not a.isSeparator()]


def test_the_md_items_sit_after_open_folder_and_in_help(window):
    names = _rows(window, "File")
    at = names.index("Open folder…")
    assert names[at + 1:at + 3] == ["Open MD model…",
                                    "Open MD series as one model…"]
    help_action = _action(window, "Help", "MD models and the Model workspace")
    assert help_action is window.md_help_action


def test_the_menu_bar_holds_a_model_menu_of_at_most_twelve_rows(window):
    titles = [a.text().replace("&", "")
              for a in window.menuBar().actions()]
    assert titles == ["File", "Edit", "View", "Model", "Help"]
    for title in titles:
        rows = _rows(window, title)
        assert len(rows) <= 12, (title, rows)
    model = _rows(window, "Model")
    assert model == ["Run analyses", "Cancel run", "Presets", "Export results",
                     "Save request…", "Load request…",
                     "Show the last complete run",
                     "Show the cancelled run's partial result"]
    assert "Bond-valence parameters" in _rows(window, "File")
    view_rows = _rows(window, "View")
    assert "Frame" in view_rows and "Highlight…" in view_rows
    # every row of the Model menu explains itself on hover
    bar_action = _bar_action(window, "Model")
    for action in bar_action.menu().actions():
        if not action.isSeparator():
            assert action.toolTip(), action.text()


def test_every_menu_keeps_one_item_per_mnemonic(window):
    """Two items in one menu claiming the same letter breaks the keyboard:
    Qt moves the highlight instead of triggering. Checked on every menu,
    the Model menu and its submenus included."""
    from collections import Counter

    from PySide6.QtWidgets import QMenu

    window._fill_presets_menu()

    def check(menu, path):
        letters = Counter()
        for action in menu.actions():
            if action.isSeparator():
                continue
            text = action.text()
            if "&" in text and text.index("&") + 1 < len(text):
                letters[text[text.index("&") + 1].lower()] += 1
            sub = action.menu()
            if isinstance(sub, QMenu):
                check(sub, f"{path} > {text.replace('&', '')}")
        duplicated = {k: v for k, v in letters.items() if v > 1}
        assert not duplicated, f"{path}: {duplicated}"

    bar = Counter(a.text()[a.text().index("&") + 1].lower()
                  for a in window.menuBar().actions() if "&" in a.text())
    assert not {k: v for k, v in bar.items() if v > 1}
    for action in window.menuBar().actions():
        if action.menu() is not None:
            check(action.menu(), action.text().replace("&", ""))


def test_the_model_menu_is_disabled_while_a_crystal_is_shown(window):
    window.load(str(CIF))
    assert not window.run_action.isEnabled()
    assert not window.cancel_action.isEnabled()
    assert not window.highlight_action.isEnabled()
    assert not window.frames_menu.menuAction().isEnabled()
    assert not window.export_results_menu.menuAction().isEnabled()
    # a Model item reached with a crystal shown says so, raising nothing
    window._model_call("run_analysis")
    assert "MD model" in window.statusBar().currentMessage()


def test_the_help_item_opens_the_md_section(window, qapp):
    window.md_help_action.trigger()
    dialog = window._manual_dialog
    try:
        assert "MD models" in dialog.contents.currentItem().text()
        assert "Model workspace" in dialog.browser.toPlainText()
    finally:
        dispose(dialog)


def test_f1_opens_the_manual_at_the_part_of_the_tab_shown(window, recorder,
                                                          qapp):
    from facet.ui.help import md_anchor_for

    assert md_anchor_for("Setup") == "md-setup"
    assert md_anchor_for("Highlight") == "md-highlight"
    assert md_anchor_for("read-options") == "md-open"
    assert md_anchor_for("nothing") == ""
    # a crystal shown: the manual's start
    window.manual_action.trigger()
    dialog = window._manual_dialog
    try:
        assert dialog.contents.currentRow() == 0
        # a model shown (its read panel): the MD section
        _drop(window, [MD / "dump_ortho_real.lammpstrj"])
        recorder.controllers[0].canvas_shown = lambda: "read-options"
        window.manual_action.trigger()
        assert "MD models" in dialog.contents.currentItem().text()
    finally:
        dispose(dialog)


def test_a_theme_change_reaches_the_models(window, recorder):
    from facet.core import theme as theme_mod

    _drop(window, [MD / "dump_ortho_real.lammpstrj"])
    model = recorder.controllers[0]
    dark = theme_mod.Theme()
    window._on_theme_structural(dark)
    assert model.themes == [dark]


def test_the_md_manual_section(qapp):
    """Every analysis and every registered format is named, every preset
    and every question group, the anchors F1 scrolls to, the thresholds
    from the code, and no verdict word."""
    import re

    from facet.core import bv, md_analysis, md_presets
    from facet.ui.help import MD_ANCHORS, MD_SECTION, _sections
    from facet.ui.md_setup import ANALYSES_BY_QUESTION
    from facet.ui.preview import md_filter_entries

    sections = {k: b for k, _t, b in _sections()}
    body = sections[MD_SECTION]
    text = re.sub(r"<[^>]+>", " ", body)
    for name in md_analysis.ANALYSES:
        assert f"<b>{name}</b>" in body, name
    for label, _patterns, _name in md_filter_entries():
        assert label.replace("&", "&amp;") in body, label
    for preset in md_presets.PRESETS:
        assert f"<b>{preset.name}</b>" in body, preset.name
    for group, _members in ANALYSES_BY_QUESTION:
        assert f"<b>{group}</b>" in body, group
    for anchor in set(MD_ANCHORS.values()) | {"md-frames", "md-menu",
                                               "md-export", "md-cli"}:
        assert f'name="{anchor}"' in body, anchor
    assert f"{bv.V_BOND_DEFAULT:g} v.u." in text
    assert f"{bv.V_LIST_DEFAULT:g} v.u." in text
    for claim in ("py -3.11 -m facet.md analyse",
                  "py -3.11 -m facet glass.lammpstrj",
                  "No molecular dynamics is run",
                  "No quadrupolar NMR spectrum is simulated",
                  "by charge", "by modifier density", "by voids",
                  "Save request", "Load request"):
        assert claim in text, claim
    forbidden = {"good", "bad", "poor", "excellent", "acceptable", "correct",
                 "incorrect", "wrong", "reliable", "unreliable",
                 "trustworthy", "untrustworthy", "unusable", "should",
                 "proves", "confirms"}
    assert not set(re.findall(r"[a-z]+", text.lower())) & forbidden
    assert "does not run molecular dynamics" in sections["limits"].lower()
    assert "Model window" not in text


def test_the_empty_view_names_md_models(window):
    from PySide6.QtGui import QImage, QPainter

    assert "Model workspace" in window.view.MD_HINT
    window.view.resize(640, 400)
    image = QImage(640, 400, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        window.view._paint_placeholder(painter)
    finally:
        painter.end()
    background = image.pixel(2, 2)
    lower = {image.pixel(x, y) for y in range(215, 300, 2)
             for x in range(40, 600, 3)}
    assert len(lower - {background}) > 0


# ---------------------------------------------------------------------------
# the real workspace
# ---------------------------------------------------------------------------

@pytest.fixture
def real_window(qapp, warnings):
    from facet.ui.preview import PreviewWindow

    w = PreviewWindow()
    w.show()
    yield w
    dispose(w)


@pytest.mark.parametrize("path", [
    MD / "dump_ortho_real.lammpstrj", MD / "data_charge.data",
    MD / "lammps_formats" / "traj.dcd", MD / "xtc_lammps.xtc",
    MD / "glass.extxyz", MD / "recognition" / "NS2-pos-1.xyz"],
    ids=["dump", "data", "dcd", "xtc", "extxyz", "plain xyz"])
def test_a_drop_opens_the_real_workspace(real_window, qapp, path):
    from facet.ui import md_workspace

    assert _drop(real_window, [path])
    models = _models(real_window)
    assert len(models) == 1
    controller = models[0].controller
    assert isinstance(controller, md_workspace.ModelController)
    assert _wait(qapp, lambda: controller.trajectory is not None
                 or controller.problem is not None)
    assert real_window.column_stack.currentWidget() is controller.column
    assert _crystal_tabs(real_window) == CRYSTAL_TABS
    if controller.trajectory is not None:
        tabs = controller.tabs
        assert [tabs.tabText(i) for i in range(tabs.count())] == MODEL_TABS
        assert controller.canvas_shown() == "3d"
        # the row names its atoms and frames once read
        text = real_window.structure_panel.list.item(0).text()
        assert "atoms" in text and "frame" in text
    else:
        assert controller.canvas_shown() == "read-options"


def test_a_mixed_drop_opens_the_real_workspace_beside_the_crystal(
        real_window, qapp):
    assert _drop(real_window, [CIF, MD / "glass.extxyz"])
    assert [e.kind for e in real_window.project.entries] == ["crystal",
                                                             "model"]
    controller = _models(real_window)[0].controller
    assert _wait(qapp, lambda: controller.trajectory is not None)
    assert real_window.project.current.kind == "crystal"
    assert _crystal_tabs(real_window) == CRYSTAL_TABS
    real_window.structure_panel.list.setCurrentRow(1)
    qapp.processEvents()
    assert real_window.column_stack.currentWidget() is controller.column


def test_a_dump_dropped_with_its_data_file_is_offered_that_file(real_window,
                                                                qapp):
    """Dropped together, a dump and the data file of its run: the dump's
    read panel asks for the Masses file and offers the one dropped with
    it, never filled in."""
    dump = MD / "dump_tri_x.lammpstrj"
    data = MD / "data_atomic.data"
    assert _drop(real_window, [dump, data])
    models = _models(real_window)
    assert len(models) == 2
    entry = next(e for e in models if Path(e.source_path).name == dump.name)
    controller = entry.controller
    assert [Path(p) for p in controller.companions] == [data]
    assert _wait(qapp, lambda: controller.problem is not None)
    assert controller.canvas_shown() == "read-options"
    panel = controller.read_panel
    assert "masses_from" in panel.companion_buttons
    assert panel.fields["masses_from"].text() == ""
    panel.companion_buttons["masses_from"].click()
    assert Path(panel.fields["masses_from"].text()) == data
