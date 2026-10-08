"""MD files reach the Model window from everywhere the crystal window opens a file.

Before this, a dropped ``.lammpstrj`` or ``.data`` was refused by the drag
cursor with no message at all, and File > Open or the command line gave a
"could not be read" box pointing at a Python function. These tests drive the
real crystal window (``PreviewWindow``): a drop through ``QDropEvent``, File >
Open and File > Open MD model through their menu actions with the file dialog
answered, and the command line through the window's constructor, for a LAMMPS
dump, a data file, DCD, XTC, a multi-frame extended XYZ, a plain XYZ
trajectory with no box, and the other formats the reader registers.

Two layers. The routing tests put a recording stand-in for
``md_workspace.open_model_window`` in place, so they check exactly what the
crystal window hands over (which file, as one model or a series, with which
parent) whatever state the Model window itself is in. The tests at the end
open the real ``ModelWindow`` and are skipped while ``facet.ui.md_workspace``
does not exist.
"""
from __future__ import annotations

import sys
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


class _Recorder:
    """A stand-in for facet.ui.md_workspace: records every call and returns
    a real, shown QMainWindow named ModelWindow, as the contract says."""

    def __init__(self):
        from PySide6.QtWidgets import QMainWindow

        class ModelWindow(QMainWindow):
            def __init__(self, paths, parent=None, *, read_options=None):
                super().__init__(parent)
                self.paths = paths
                self.read_options = read_options
                self.themes = []

            def apply_theme(self, theme):
                self.themes.append(theme)

        self.ModelWindow = ModelWindow
        self.calls: list[tuple] = []
        self.windows: list = []

    def open_model_window(self, paths, parent=None, *, read_options=None):
        self.calls.append((paths, parent, read_options))
        window = self.ModelWindow(paths, parent, read_options=read_options)
        window.show()
        self.windows.append(window)
        return window

    def opened(self) -> list:
        """The paths of every call, a series as a list of names."""
        return [[Path(p).name for p in paths] if isinstance(paths, list)
                else Path(paths).name for paths, _parent, _o in self.calls]


@pytest.fixture
def recorder(monkeypatch, qapp):
    import facet.ui

    rec = _Recorder()
    stub = types.ModuleType("facet.ui.md_workspace")
    stub.open_model_window = rec.open_model_window
    stub.ModelWindow = rec.ModelWindow
    monkeypatch.setitem(sys.modules, "facet.ui.md_workspace", stub)
    monkeypatch.setattr(facet.ui, "md_workspace", stub, raising=False)
    yield rec
    dispose(*rec.windows)


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


def _tab_titles(window):
    from PySide6.QtWidgets import QTabWidget

    tabs = window.findChild(QTabWidget)
    return [tabs.tabText(i) for i in range(tabs.count())]


CRYSTAL_TABS = ["Site", "Tools", "Diffraction", "PDF", "EXAFS", "Planes",
                "Styles", "Volume", "Disorder", "Theme"]


# ---------------------------------------------------------------------------
# the routing rule itself (no window)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", PATHS, ids=IDS)
def test_every_md_test_file_routes_to_a_model_window(path):
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
    format the reader named; the router adds only refusals the Model window
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
    # one frame of LAMMPS's dump xyz states no box either; the crystal
    # reader refuses it rather than invent one
    one = _plain_xyz(tmp_path / "one.xyz", frames=1)
    assert read_or_route(one) == ("md", "extxyz")


def test_an_empty_dump_opens_where_its_reader_can_say_so(tmp_path):
    """The crystal reader says 'not recognised'; the MD reader states the
    fault of a file under an MD-only name, so the Model window gets it."""
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
    # two runs whose names differ only in digits stay two models
    runs = [(str(tmp_path / "glass_300K.lammpstrj"), "lammps-dump"),
            (str(tmp_path / "glass_600K.lammpstrj"), "lammps-dump")]
    assert model_groups(runs) == [runs[0][0], runs[1][0]]
    # the same file twice is one model
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
def test_dropping_an_md_file_opens_a_model_window(window, recorder, warnings,
                                                  path):
    assert _drop(window, [path])
    assert recorder.opened() == [path.name]
    _paths, parent, read_options = recorder.calls[0]
    assert parent is window and read_options is None
    assert window.model_windows == recorder.windows
    assert len(window.project) == 0           # nothing went to the crystal side
    assert warnings == []                     # and nothing was refused
    assert _tab_titles(window) == CRYSTAL_TABS


def test_dropping_a_plain_xyz_trajectory_opens_a_model_window(
        window, recorder, warnings, tmp_path):
    path = _plain_xyz(tmp_path / "lammps_dump_xyz.xyz")
    assert _drop(window, [path])
    assert recorder.opened() == [path.name]
    assert warnings == []


def test_a_mixed_drop_loads_the_cif_and_opens_the_dump(window, recorder,
                                                       warnings):
    dump = MD / "dump_ortho_real.lammpstrj"
    assert _drop(window, [CIF, dump])
    assert recorder.opened() == [dump.name]
    assert len(window.project) == 1
    assert window.structure is not None
    assert sorted(window.structure.elements_present) == ["O", "Si"]
    assert warnings == []
    assert _tab_titles(window) == CRYSTAL_TABS


def test_a_drop_of_cfg_snapshots_is_one_model(window, recorder, warnings):
    cfg = sorted((MD / "lammps_formats" / "cfg").glob("dump.*.cfg"))
    assert _drop(window, cfg[::-1])
    assert recorder.opened() == [[p.name for p in cfg]]


def test_a_drop_of_two_dumps_is_two_models(window, recorder, warnings):
    dumps = [MD / "dump_ortho_real.lammpstrj", MD / "dump_tri_x.lammpstrj"]
    assert _drop(window, dumps)
    assert recorder.opened() == [p.name for p in dumps]


def test_a_drop_with_an_unreadable_file_still_says_so(window, recorder,
                                                      warnings, tmp_path):
    junk = tmp_path / "junk.xyz"
    junk.write_text("this is not a structure\n")
    dump = MD / "dump_ortho_real.lammpstrj"
    assert _drop(window, [dump, junk])
    assert recorder.opened() == [dump.name]
    assert len(warnings) == 1
    title, text = warnings[0]
    assert "junk.xyz" in text and "1 opened in a Model window" in text


# ---------------------------------------------------------------------------
# File > Open, File > Open MD model, File > Open MD series
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", PATHS, ids=IDS)
def test_file_open_opens_a_model_window(window, recorder, warnings,
                                        monkeypatch, path):
    _answer_dialog(monkeypatch, [path])
    _action(window, "File", "Open…").trigger()
    assert recorder.opened() == [path.name]
    assert warnings == []
    assert len(window.project) == 0


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


def test_file_open_md_model_takes_a_one_frame_extxyz_as_a_model(
        window, recorder, warnings, monkeypatch, tmp_path):
    """The crystal reader reads one frame with Lattice= as a crystal; chosen
    as an MD model it opens as a one-frame model."""
    lines = (MD / "glass.extxyz").read_text().splitlines()
    one = tmp_path / "one_frame.extxyz"
    one.write_text("\n".join(lines[:7]) + "\n")
    _answer_dialog(monkeypatch, [one])
    _action(window, "File", "Open MD model…").trigger()
    assert recorder.opened() == [one.name]


def test_file_open_md_model_sends_a_cif_to_the_crystal_window(
        window, recorder, warnings, monkeypatch):
    _answer_dialog(monkeypatch, [CIF])
    _action(window, "File", "Open MD model…").trigger()
    assert recorder.calls == []
    assert len(window.project) == 1


def test_file_open_md_series_reads_the_files_as_one_model(
        window, recorder, warnings, monkeypatch, tmp_path):
    source = (MD / "dump_ortho_real.lammpstrj").read_bytes()
    names = ["dump.100.lammpstrj", "dump.20.lammpstrj", "dump.3.lammpstrj"]
    for name in names:
        (tmp_path / name).write_bytes(source)
    _answer_dialog(monkeypatch, [tmp_path / n for n in names])
    _action(window, "File", "Open MD series as one model…").trigger()
    # natural order: digits compare as numbers
    assert recorder.opened() == [["dump.3.lammpstrj", "dump.20.lammpstrj",
                                  "dump.100.lammpstrj"]]


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [MD / "dump_ortho_real.lammpstrj",
                                  MD / "data_charge.data",
                                  MD / "xtc_lammps.xtc"],
                         ids=["dump", "data", "xtc"])
def test_a_command_line_md_file_opens_a_model_window(qapp, recorder, warnings,
                                                     path):
    """app.main and preview.main hand their argument to PreviewWindow(path)."""
    from facet.ui.preview import PreviewWindow

    w = PreviewWindow(str(path))
    try:
        assert recorder.opened() == [path.name]
        assert recorder.calls[0][1] is w
        assert len(w.project) == 0
        assert warnings == []
    finally:
        dispose(w)


def test_a_command_line_cif_still_opens_in_the_crystal_window(qapp, recorder,
                                                              warnings):
    from facet.ui.preview import PreviewWindow

    w = PreviewWindow(str(CIF))
    try:
        assert recorder.calls == []
        assert len(w.project) == 1
        assert _tab_titles(w) == CRYSTAL_TABS
    finally:
        dispose(w)


# ---------------------------------------------------------------------------
# no silence: a Model window that cannot open says so
# ---------------------------------------------------------------------------

def test_a_build_without_the_model_window_says_so(qapp, monkeypatch, warnings):
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


def test_a_model_window_that_raises_is_reported(window, recorder, warnings,
                                                monkeypatch):
    def broken(paths, parent=None, *, read_options=None):
        raise RuntimeError("could not build")

    monkeypatch.setattr(sys.modules["facet.ui.md_workspace"],
                        "open_model_window", broken)
    _drop(window, [MD / "dump_ortho_real.lammpstrj"])
    assert len(warnings) == 1
    assert "could not build" in warnings[0][1]


# ---------------------------------------------------------------------------
# the menus and the empty view
# ---------------------------------------------------------------------------

def test_the_md_items_sit_after_open_folder_and_in_help(window):
    # the bar's action is held while its menu is read: PySide6 invalidates
    # the menu's wrapper once the action's wrapper is collected
    file_action = next(a for a in window.menuBar().actions()
                       if a.text() == "&File")
    names = [a.text().replace("&", "") for a in file_action.menu().actions()
             if not a.isSeparator()]
    at = names.index("Open folder…")
    assert names[at + 1:at + 3] == ["Open MD model…",
                                    "Open MD series as one model…"]
    help_action = _action(window, "Help", "MD models and the Model window")
    assert help_action is window.md_help_action


def test_the_md_items_add_no_mnemonic_collision(window):
    """Each new item takes a letter its menu does not already use. (The other
    session's File > Examples and File > Export share 'e'; that pair is
    theirs and is left out of this count.)"""
    from collections import Counter

    for title, new in (("File", ("Open MD &model…",
                                 "Open MD se&ries as one model…")),
                       ("Help", ("MD models and the Model &window",))):
        bar_action = next(a for a in window.menuBar().actions()
                          if a.text().replace("&", "") == title)
        letters = Counter()
        for action in bar_action.menu().actions():
            text = action.text()
            if "&" in text and text.index("&") + 1 < len(text):
                letters[text[text.index("&") + 1].lower()] += 1
        for text in new:
            letter = text[text.index("&") + 1].lower()
            assert letters[letter] == 1, (title, text, letters)


def test_the_help_item_opens_the_md_section(window, qapp):
    window.md_help_action.trigger()
    dialog = window._manual_dialog
    try:
        assert "MD models" in dialog.contents.currentItem().text()
        assert "Model window" in dialog.browser.toPlainText()
    finally:
        dispose(dialog)


def test_a_theme_change_reaches_the_model_windows(window, recorder):
    from facet.core import theme as theme_mod

    _drop(window, [MD / "dump_ortho_real.lammpstrj"])
    model = recorder.windows[0]
    dark = theme_mod.Theme()
    window._on_theme_structural(dark)
    assert model.themes == [dark]


def test_the_md_manual_section(qapp):
    """Every analysis and every registered format is named, the thresholds
    come from the code, and no verdict word is used."""
    import re

    from facet.core import bv, md_analysis
    from facet.ui.help import MD_SECTION, _sections
    from facet.ui.preview import md_filter_entries

    sections = {k: b for k, _t, b in _sections()}
    body = sections[MD_SECTION]
    text = re.sub(r"<[^>]+>", " ", body)
    for name in md_analysis.ANALYSES:
        assert f"<b>{name}</b>" in body, name
    for label, _patterns, _name in md_filter_entries():
        assert label.replace("&", "&amp;") in body, label
    assert f"{bv.V_BOND_DEFAULT:g} v.u." in text
    assert f"{bv.V_LIST_DEFAULT:g} v.u." in text
    for claim in ("py -3.11 -m facet.md analyse",
                  "py -3.11 -m facet glass.lammpstrj",
                  "No molecular dynamics is run",
                  "No quadrupolar NMR spectrum is simulated"):
        assert claim in text, claim
    forbidden = {"good", "bad", "poor", "excellent", "acceptable", "correct",
                 "incorrect", "wrong", "reliable", "unreliable",
                 "trustworthy", "untrustworthy", "unusable", "should",
                 "proves", "confirms"}
    assert not set(re.findall(r"[a-z]+", text.lower())) & forbidden
    assert "does not run molecular dynamics" in sections["limits"].lower()


def test_the_empty_view_names_md_models(window):
    from PySide6.QtGui import QImage, QPainter

    assert "Model window" in window.view.MD_HINT
    window.view.resize(640, 400)
    image = QImage(640, 400, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        window.view._paint_placeholder(painter)
    finally:
        painter.end()
    # the hint is drawn below the view's own line: ink in the lower half
    background = image.pixel(2, 2)
    lower = {image.pixel(x, y) for y in range(215, 300, 2)
             for x in range(40, 600, 3)}
    assert len(lower - {background}) > 0


# ---------------------------------------------------------------------------
# the real Model window (skipped until facet.ui.md_workspace exists)
# ---------------------------------------------------------------------------

def _real_workspace():
    try:
        import importlib

        return importlib.import_module("facet.ui.md_workspace")
    except ImportError:
        return None


REAL = pytest.mark.skipif(_real_workspace() is None,
                          reason="facet.ui.md_workspace is not built yet")


@pytest.fixture
def real_window(qapp, warnings):
    from facet.ui.preview import PreviewWindow

    w = PreviewWindow()
    w.show()
    yield w
    windows = w.model_windows
    for model in windows:
        try:
            model.cancel()
        except Exception:
            pass
    dispose(*windows)
    dispose(w)


@REAL
@pytest.mark.parametrize("path", [
    MD / "dump_ortho_real.lammpstrj", MD / "data_charge.data",
    MD / "lammps_formats" / "traj.dcd", MD / "xtc_lammps.xtc",
    MD / "glass.extxyz", MD / "recognition" / "NS2-pos-1.xyz"],
    ids=["dump", "data", "dcd", "xtc", "extxyz", "plain xyz"])
def test_a_drop_opens_a_real_model_window(real_window, qapp, path):
    md_workspace = _real_workspace()
    assert _drop(real_window, [path])
    qapp.processEvents()
    windows = real_window.model_windows
    assert len(windows) == 1
    assert isinstance(windows[0], md_workspace.ModelWindow)
    assert windows[0].isVisible()
    assert len(real_window.project) == 0
    assert _tab_titles(real_window) == CRYSTAL_TABS


@REAL
def test_a_mixed_drop_opens_a_real_model_window_beside_the_crystal(
        real_window, qapp):
    md_workspace = _real_workspace()
    assert _drop(real_window, [CIF, MD / "dump_ortho_real.lammpstrj"])
    qapp.processEvents()
    windows = real_window.model_windows
    assert len(windows) == 1
    assert isinstance(windows[0], md_workspace.ModelWindow)
    assert len(real_window.project) == 1
    assert _tab_titles(real_window) == CRYSTAL_TABS


@REAL
def test_a_dump_dropped_with_its_data_file_is_offered_that_file(real_window,
                                                                qapp):
    """Dropped together, a dump and the data file of its run opened two
    windows, and the dump's window asked for the Masses file without
    offering the one dropped with it. It is offered now, never filled in."""
    import time

    dump = MD / "dump_tri_x.lammpstrj"
    data = MD / "data_atomic.data"
    assert _drop(real_window, [dump, data])
    qapp.processEvents()
    windows = real_window.model_windows
    assert len(windows) == 2
    dump_window = next(w for w in windows
                       if Path(str(w.source)).name == dump.name)
    assert [Path(p) for p in dump_window.companions] == [data]
    end = time.perf_counter() + 30
    while dump_window.problem is None and time.perf_counter() < end:
        qapp.processEvents()
        time.sleep(0.01)
    panel = dump_window.read_panel
    assert "masses_from" in panel.companion_buttons
    assert panel.fields["masses_from"].text() == ""
    panel.companion_buttons["masses_from"].click()
    assert Path(panel.fields["masses_from"].text()) == data
