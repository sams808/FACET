"""The reading batch: a CIF's own bond list read and pinned on the explorer,
a multi-block CIF opening every block, bzip2 and xz read by their magic
bytes, a dropped folder, the read options remembered by path, and the data
files beside a lone dump offered as companions.
"""
from __future__ import annotations

import bz2
import lzma
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import dispose, sample_cif

from facet.core import bv, cif, coordination, md_readers, prefs

HERE = Path(__file__).resolve().parent
QUARTZ = HERE / "data" / "crystals" / "quartz_SiO2_cod9013321.cif"
DUMP = HERE / "data" / "md" / "dump_ortho_real.lammpstrj"
DATA = HERE / "data" / "md" / "data_charge.data"
SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")

BOND_LOOP = """
loop_
_geom_bond_atom_site_label_1
_geom_bond_atom_site_label_2
_geom_bond_distance
Si O 1.6052(3)
Si O 1.6109(3)
"""


def _quartz_with_bonds(tmp_path: Path) -> Path:
    text = QUARTZ.read_text(encoding="utf-8")
    out = tmp_path / "quartz_bonds.cif"
    out.write_text(text.rstrip() + "\n" + BOND_LOOP, encoding="utf-8")
    return out


# -- the authors' bonds -----------------------------------------------------------

def test_a_cif_bond_loop_is_read_and_noted(tmp_path):
    s = cif.read(str(_quartz_with_bonds(tmp_path)))
    assert len(s.bond_loops) == 2
    assert s.bond_loops[0][2] == pytest.approx(1.6052)
    assert any("lists 2 bonds" in n for n in s.notes)
    assert cif.read(str(QUARTZ)).bond_loops == []


def test_the_explorer_marks_the_authors_bonds(qapp, tmp_path):
    from facet.ui.cutoff_explorer import CutoffExplorer
    s = cif.read(str(_quartz_with_bonds(tmp_path)))
    r = coordination.analyse_structure(s, bv.DEFAULT, v_bond=0.075,
                                       v_list=0.02, cations_only=True)[0]
    w = CutoffExplorer()
    w.set_result(r)
    assert w.authors_threshold() is None
    w.set_bond_loops(s.bond_loops)
    v = w.authors_threshold()
    assert v is not None
    # the weakest listed contact, just below its valence, so the cut keeps
    # it: each listed bond matches its nearest contact, so the 1.6109 Å bond
    # takes a 1.611 Å contact and not the 1.605 Å one within tolerance
    nearest = min((c for c in r.contacts if c.label == "O"),
                  key=lambda c: abs(c.distance - 1.6109))
    assert v == pytest.approx(nearest.valence * 0.999, abs=2e-4)
    assert any(label == "authors' bonds" for _, label, _ in w._reference_marks())
    dispose(w)


# -- a multi-block CIF -------------------------------------------------------------

def test_every_block_of_a_multi_block_cif_opens(qapp, tmp_path):
    prefs.set("tour_offered", True)
    text = QUARTZ.read_text(encoding="utf-8")
    # two different structures in one file (two identical blocks would be
    # refused as an ASE trajectory written to CIF, by design)
    second = SAMPLE.read_text(encoding="utf-8", errors="replace")
    two = tmp_path / "two_blocks.cif"
    two.write_text(text.rstrip() + "\n\n" + second, encoding="utf-8")
    from facet.ui import preview
    assert len(preview._other_blocks(two, cif.read(str(two)))) == 1
    assert preview._other_blocks(QUARTZ, cif.read(str(QUARTZ))) == []
    from PySide6.QtWidgets import QMessageBox
    warnings = []
    original = QMessageBox.warning
    QMessageBox.warning = staticmethod(lambda *a, **k: warnings.append(a) or QMessageBox.Ok)
    win = preview.PreviewWindow()
    try:
        win.load_many([str(two)])
        qapp.processEvents()
        assert not warnings, warnings[0][2] if warnings else ""
        assert len(win.project) == 2
        names = [e.structure.name for e in win.project.entries]
        assert names[0] != names[1]
    finally:
        QMessageBox.warning = original
        win.close()
        dispose(win)


# -- compression ----------------------------------------------------------------------

@pytest.mark.parametrize("opener, suffix", [(bz2.open, ".bz2"), (lzma.open, ".xz")])
def test_bzip2_and_xz_dumps_read_by_their_magic(tmp_path, opener, suffix):
    plain = md_readers.read_trajectory(str(DUMP)).frame(0)
    packed = tmp_path / ("dump.lammpstrj" + suffix)
    with open(DUMP, "rb") as src, opener(packed, "wb") as dst:
        shutil.copyfileobj(src, dst)
    # and under a name that says nothing about compression
    renamed = tmp_path / "dump_renamed.lammpstrj"
    shutil.copy(packed, renamed)
    for path in (packed, renamed):
        frame = md_readers.read_trajectory(str(path)).frame(0)
        assert frame.n_atoms == plain.n_atoms
        assert np.allclose(frame.cart_ang, plain.cart_ang)
    assert md_readers._compression_of(packed) == ("bz2" if suffix == ".bz2" else "xz")
    assert md_readers._compression_of(DUMP) is None


# -- a dropped folder ------------------------------------------------------------------

def test_a_folder_can_be_dropped(qapp, tmp_path):
    prefs.set("tour_offered", True)
    folder = tmp_path / "cifs"
    folder.mkdir()
    shutil.copy(QUARTZ, folder / "a.cif")
    shutil.copy(SAMPLE, folder / "b.cif")
    (folder / "notes.txt").write_text("not a structure", encoding="utf-8")
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        assert win._droppable(str(folder))
        assert not win._droppable(str(folder / "notes.txt"))

        class Url:
            def __init__(self, p):
                self._p = p

            def toLocalFile(self):
                return self._p

        class Mime:
            def urls(self):
                return [Url(str(folder))]

        class Event:
            accepted = False

            def mimeData(self):
                return Mime()

            def acceptProposedAction(self):
                self.accepted = True
        event = Event()
        win.dropEvent(event)
        qapp.processEvents()
        assert event.accepted and len(win.project) == 2
    finally:
        win.close()
        dispose(win)


# -- remembered read options and companions --------------------------------------------

def test_read_options_are_remembered_by_path(tmp_path):
    from facet.ui import md_workspace as ws
    source = str(tmp_path / "run.lammpstrj")
    assert ws._remembered_read_options(source) == {}
    ws._remember_read_options(source, {"type_map": {1: "Si", 2: "O"},
                                       "units": "metal"})
    back = ws._remembered_read_options(source)
    assert back["units"] == "metal"
    assert back["type_map"] == {1: "Si", 2: "O"}      # keys are integers again
    assert ws._remembered_read_options(str(tmp_path / "other.lammpstrj")) == {}


def test_a_lone_dump_is_offered_the_data_files_beside_it(tmp_path):
    from facet.ui.preview import md_companions
    folder = tmp_path / "run"
    folder.mkdir()
    dump = folder / "run.lammpstrj"
    shutil.copy(DUMP, dump)
    shutil.copy(DATA, folder / "system.data")
    (folder / "README.txt").write_text("x", encoding="utf-8")
    found = md_companions(str(dump), [])
    assert [Path(f).name for f in found] == ["system.data"]
    # opened together, only the dropped data files are companions
    found = md_companions(str(dump), [(str(folder / "system.data"), "lammps-data")])
    assert [Path(f).name for f in found] == ["system.data"]
