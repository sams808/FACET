"""md_formats_text: CASTEP .md, GROMACS .gro, animated XSF, PDB with several
frames, IMD and POSCAR series give back the model their writers were given.

The files in tests/data/md/text were written by the programs that write these
formats (ASE 3.29, mdtraj 1.11, MDAnalysis 2.10, OVITO 3.16; writers.json)
from one six-atom, three-frame model, by make_text_files.py, which imports
nothing from FACET; truth.json holds that model. castep_doc_example.md is the
example step of CASTEP's documented .md layout, copied line by line. What is
pinned, and why:

* **Positions, boxes, velocities and times agree with the model** to the
  precision each writer keeps: 1e-8 Å for CASTEP (ASE converts with CODATA
  2002, FACET with scipy's CODATA 2022), the decimals a .gro or PDB file
  writes, 1e-12 Å for XSF, IMD and POSCAR.
* **Units come from scipy.constants**, and the CASTEP example's numbers are
  converted with them, not re-derived from the reader.
* **Nothing is dropped silently:** a cut last frame, missing ANIMSTEPS steps,
  a POSCAR of another composition in a series each have a skipped position
  and a reason.
* **Every refusal names the file and says what to pass or what FACET reads:**
  force-field atom names, a .gro without a box, an XSF slab, a PDB placeholder
  cell, an untyped IMD file, a binary IMD file, a VASP 4 POSCAR without names.
* **The sniffers claim their own files only**, a one-frame PDB and a single
  POSCAR stay with the crystal reader, and a Markdown file is not CASTEP's.
* **CRLF and gzip copies read identically;** a trajectory pickles.
* **The FormatSpec contract** (md_formats_base) holds, and the module has no
  verdict words and no dependency beyond numpy, scipy, gemmi and stdlib.
* **The defects two verification passes found** each have a test at the end
  of this file that fails without its fix: PDB coordinates read from their
  columns (repeated residue numbers, fields that are not numbers), CRYST1
  and SCALEn records checked against what they can hold, CP2K's step and
  time REMARKs, frames merged by a lost delimiter or found outside any
  MODEL block, XSF step numbers, a one-structure XSF, last lines without a
  line ending, .gro atom counts too large and blank titles, ASE's image
  index on CASTEP's time line, velocities of a run started from rest, the
  file named in a later frame's error, POSCAR routing (gzip names, long
  headers, the VASP 4 layout, a VASP run directory) and content-only
  routing. mda_resid_wrap.pdb (MDAnalysis), pmg_single.xsf (pymatgen) and
  cp2k_layout.pdb (CP2K's Fortran formats) come from make_text_files_more.py
  (writers_more.json).
"""
from __future__ import annotations

import ast
import gzip
import inspect
import json
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy import constants

from facet.core import md_formats_base, md_formats_text as T, md_readers, readers
from facet.core.md_model import NO_TIMESTEP, FrameError
from facet.core.readers import UnsupportedFormat

DATA = Path(__file__).resolve().parent / "data" / "md" / "text"
MD_DATA = DATA.parent
TRUTH = json.loads((DATA / "truth.json").read_text(encoding="utf-8"))
ELEMENTS = TRUTH["elements"]
POS = np.array(TRUTH["positions_ang"])
BOX = np.array(TRUTH["boxes_ang"])
VEL = np.array(TRUTH["velocities_ang_per_ps"])
TIMES = np.array(TRUTH["times_ps"])
IMD_MAP = {1: "Si", 2: "O", 3: "Na"}


def lattice_gap(cart, ref, box) -> float:
    """Largest distance between two position sets, modulo the lattice."""
    d = (np.asarray(cart) - np.asarray(ref)) @ np.linalg.inv(box)
    d -= np.round(d)
    return float(np.linalg.norm(d @ box, axis=1).max())


def copy_with(tmp_path, name, transform, new_name=None) -> Path:
    """A copy of a fixture with its text transformed."""
    text = (DATA / name).read_text(encoding="utf-8")
    out = tmp_path / (new_name or name)
    out.write_text(transform(text), encoding="utf-8", newline="")
    return out


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def test_unit_constants_come_from_scipy():
    pc = constants.physical_constants
    assert T.BOHR_ANG == pc["Bohr radius"][0] / constants.angstrom
    assert T.AU_TIME_PS == pc["atomic unit of time"][0] / constants.pico
    # The atomic unit of velocity is a0 E_h / hbar = Bohr / (atomic time).
    assert T.AU_VELOCITY_ANG_PER_PS == pytest.approx(
        T.BOHR_ANG / T.AU_TIME_PS, rel=1e-9)
    assert T.NM_ANG == pytest.approx(10.0, rel=1e-15)


# ---------------------------------------------------------------------------
# CASTEP
# ---------------------------------------------------------------------------

def test_castep_ase_file_gives_back_the_model():
    with T.read_castep_md(DATA / "ase.md") as traj:
        assert traj.file_format == "castep-md"
        assert (traj.n_atoms, traj.n_frames) == (6, 3)
        assert traj.type_map_source == "file symbols"
        assert traj.ids_track_atoms
        # ASE 3.29 writes the image index as the time line (0, 1, 2 a.u.,
        # 2.4e-5 ps apart): not a time, so no time is given, and a note says
        # why.
        assert traj.times_ps is None
        note = next(n for n in traj.notes if "image index" in n)
        assert "0, 1, 2" in note and "not used as times" in note
        assert (traj.timesteps == NO_TIMESTEP).all()
        assert traj.box_varies is True
        for k in range(3):
            frame = traj.frame(k)
            assert frame.elements.tolist() == ELEMENTS
            assert np.abs(frame.box_ang - BOX[k]).max() < 1e-8
            assert lattice_gap(frame.cart_ang, POS[k], BOX[k]) < 1e-8
            assert frame.vel_ang_per_ps == pytest.approx(VEL[k], rel=1e-6)
        assert any("not read: temperature (T)" in n for n in traj.notes)
        assert "Bohr" in traj.units_note


def test_castep_documented_example_step():
    """The example step of 'The .md file': values copied from the guide."""
    traj = T.read_castep_md(DATA / "castep_doc_example.md")
    assert (traj.n_atoms, traj.n_frames) == (8, 1)
    frame = traj.frame(0)
    assert frame.elements.tolist() == ["Si"] * 8
    assert frame.box_ang[0, 0] == pytest.approx(1.01599045e1 * T.BOHR_ANG)
    assert frame.box_ang[1, 0] == pytest.approx(1.29430839e-17 * T.BOHR_ANG)
    r1 = np.array([1.04834750e-2, 1.15560090e-2, 7.82230990e-3]) * T.BOHR_ANG
    assert frame.cart_ang[0] == pytest.approx(r1, abs=1e-12)
    v8 = np.array([7.97201068e-5, -1.33563298e-4, -7.92824105e-5])
    assert frame.vel_ang_per_ps[7] == pytest.approx(
        v8 * T.AU_VELOCITY_ANG_PER_PS, rel=1e-12)
    assert frame.time_ps == 0.0
    note = next(n for n in traj.notes if n.startswith("not read:"))
    for record in ("forces (F)", "stress (S)", "cell velocities (hv)",
                   "energies (E)", "temperature (T)", "pressure (P)"):
        assert record in note


def test_castep_geom_iterations_are_timesteps():
    traj = T.read_castep_md(DATA / "ase.geom")
    # The registered format name, so a FormatSpec lookup by name finds it;
    # the notes say the file is a .geom file.
    assert traj.file_format == "castep-md"
    assert traj.file_format in md_readers.format_names()
    assert md_readers.sniff_md(DATA / "ase.geom") == traj.file_format
    assert traj.timesteps.tolist() == [0, 1, 2]
    assert traj.times_ps is None
    assert lattice_gap(traj.frame(2).cart_ang, POS[2], BOX[2]) < 1e-8
    assert any("geometry-optimisation" in n for n in traj.notes)


def test_castep_cut_last_step_is_skipped(tmp_path):
    text = (DATA / "ase.md").read_text(encoding="utf-8")
    cut = text[:text.rfind("<-- R")]            # inside the last step's R lines
    path = tmp_path / "cut.md"
    path.write_text(cut, encoding="utf-8", newline="")
    traj = T.read_castep_md(path)
    assert traj.n_frames == 2
    assert list(traj.skipped) == [2]
    assert "truncated" in traj.skipped[2]
    # A file whose last line has no line ending may be cut inside a number.
    path.write_text(text.rstrip("\n").rstrip(), encoding="utf-8", newline="")
    traj = T.read_castep_md(path)
    assert traj.n_frames == 2 and "line ending" in traj.skipped[2]


def test_castep_markdown_is_not_castep(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nBEGIN header is a phrase here.\n",
                    encoding="utf-8")
    head = md_readers._head(path)
    assert not T.sniff_castep_md(head, path)
    with pytest.raises(UnsupportedFormat, match="notes.md"):
        T.read_castep_md(path)


def test_castep_labels_need_a_map(tmp_path):
    path = copy_with(tmp_path, "ase.md",
                     lambda s: s.replace(" Na  ", " NaX "), "labels.md")
    with pytest.raises(ValueError) as caught:
        T.read_castep_md(path)
    message = str(caught.value)
    assert message.startswith("labels.md") and "NaX" in message
    assert "type_map=" in message
    traj = T.read_castep_md(path, type_map={"NaX": "Na"})
    assert traj.frame(0).elements.tolist() == ELEMENTS
    assert traj.type_map == {"NaX": "Na"}


# ---------------------------------------------------------------------------
# GROMACS
# ---------------------------------------------------------------------------

def test_gro_ase_frame_with_velocities():
    traj = T.read_gro(DATA / "ase_vel.gro")
    assert traj.file_format == "gromacs-gro"
    assert (traj.n_atoms, traj.n_frames) == (6, 1)
    frame = traj.frame(0)
    assert frame.elements.tolist() == ELEMENTS
    # 3 decimals in nm: each component within 0.005 Å.
    assert lattice_gap(frame.cart_ang, POS[0], BOX[0]) <= 0.005 * 3 ** 0.5
    assert np.abs(frame.box_ang - BOX[0]).max() < 5e-5
    assert np.abs(frame.vel_ang_per_ps - VEL[0]).max() <= 5e-4
    assert traj.times_ps is None
    assert "nm" in traj.units_note


def test_gro_mdtraj_frames_times_and_wide_fields():
    """mdtraj's 6-decimal fields are 11 wide; the width comes from the
    distance between the decimal points."""
    traj = T.read_gro(DATA / "mdtraj.gro")
    assert (traj.n_atoms, traj.n_frames) == (6, 3)
    assert traj.times_ps == pytest.approx(TIMES)
    for k in range(3):
        frame = traj.frame(k)
        assert lattice_gap(frame.cart_ang, POS[k], BOX[k]) < 1e-5
        assert np.abs(frame.box_ang - BOX[k]).max() < 5e-5
        assert frame.vel_ang_per_ps is None


def test_gro_force_field_names_need_a_map(tmp_path):
    def rename(text):
        return (text.replace("   Si    ", "  SI1    ")
                .replace("    O    ", "   OW    ")
                .replace("   Na    ", "   NA    "))
    path = copy_with(tmp_path, "ase_vel.gro", rename, "names.gro")
    with pytest.raises(ValueError) as caught:
        T.read_gro(path)
    message = str(caught.value)
    assert message.startswith("names.gro")
    assert "'OW'" in message and "'<element>'" in message
    traj = T.read_gro(path, type_map={"SI1": "Si", "OW": "O", "NA": "Na"})
    assert traj.frame(0).elements.tolist() == ELEMENTS
    assert traj.type_map_source == "user"


def test_gro_without_box_is_refused(tmp_path):
    def zero_box(text):
        lines = text.splitlines(keepends=True)
        lines[-1] = "   0.00000   0.00000   0.00000\n"
        return "".join(lines)
    path = copy_with(tmp_path, "ase_vel.gro", zero_box, "nobox.gro")
    with pytest.raises(UnsupportedFormat, match="nobox.gro.*box"):
        T.read_gro(path)


def test_gro_cut_last_frame_is_skipped(tmp_path):
    text = (DATA / "mdtraj.gro").read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    path = tmp_path / "cut.gro"
    path.write_text("".join(lines[:-3]), encoding="utf-8", newline="")
    traj = T.read_gro(path)
    assert traj.n_frames == 2
    assert "truncated" in traj.skipped[2]


def test_gro_frame_with_a_zero_box_is_skipped(tmp_path):
    def zero_second_box(text):
        lines = text.splitlines(keepends=True)
        lines[17] = "   0.00000   0.00000   0.00000\n"      # frame 1's box
        return "".join(lines)
    path = copy_with(tmp_path, "mdtraj.gro", zero_second_box, "zero1.gro")
    traj = T.read_gro(path)
    assert traj.n_frames == 2
    assert "only zeros" in traj.skipped[1]
    assert traj.times_ps == pytest.approx([0.0, 1.0])


def test_gro_count_line_lost_is_resynchronised(tmp_path):
    def damage(text):
        lines = text.splitlines(keepends=True)
        lines[10] = "six\n"                     # frame 1's count line
        return "".join(lines)
    path = copy_with(tmp_path, "mdtraj.gro", damage, "damaged.gro")
    traj = T.read_gro(path)
    assert traj.n_frames == 2
    assert "not an atom count" in traj.skipped[1]
    assert traj.times_ps == pytest.approx([0.0, 1.0])
    assert lattice_gap(traj.frame(1).cart_ang, POS[2], BOX[2]) < 1e-5


# ---------------------------------------------------------------------------
# XSF
# ---------------------------------------------------------------------------

def test_xsf_variable_cell():
    traj = T.read_xsf(DATA / "ase_var.axsf")
    assert (traj.n_atoms, traj.n_frames) == (6, 3)
    assert traj.box_varies is True
    for k in range(3):
        frame = traj.frame(k)
        assert frame.elements.tolist() == ELEMENTS
        assert np.abs(frame.box_ang - BOX[k]).max() < 1e-12
        assert lattice_gap(frame.cart_ang, POS[k], BOX[k]) < 1e-12
    assert any("variable cell" in n for n in traj.notes)


def test_xsf_fixed_cell():
    traj = T.read_xsf(DATA / "ase_fixed.axsf")
    assert traj.box_varies is False
    for k in range(3):
        frame = traj.frame(k)
        assert np.abs(frame.box_ang - BOX[0]).max() < 1e-12
        assert lattice_gap(frame.cart_ang, POS[k], BOX[0]) < 1e-12
    assert any("fixed cell" in n for n in traj.notes)


def test_xsf_slab_is_refused(tmp_path):
    path = copy_with(tmp_path, "ase_var.axsf",
                     lambda s: s.replace("CRYSTAL", "SLAB"), "slab.axsf")
    with pytest.raises(UnsupportedFormat) as caught:
        T.read_xsf(path)
    message = str(caught.value)
    assert message.startswith("slab.axsf") and "SLAB" in message
    assert "CRYSTAL" in message


def test_xsf_step_with_another_count_is_skipped(tmp_path):
    def drop_atom(text):
        head, sep, rest = text.partition("PRIMCOORD 2\n 6 1\n")
        rows = rest.splitlines(keepends=True)
        return head + "PRIMCOORD 2\n 5 1\n" + "".join(rows[:5] + rows[6:])
    path = copy_with(tmp_path, "ase_var.axsf", drop_atom, "five.axsf")
    traj = T.read_xsf(path)
    assert traj.n_frames == 2
    assert "PRIMCOORD gives 5 atoms" in traj.skipped[1]
    assert lattice_gap(traj.frame(1).cart_ang, POS[2], BOX[2]) < 1e-12


def test_xsf_missing_steps_are_counted(tmp_path):
    path = copy_with(tmp_path, "ase_var.axsf",
                     lambda s: s.replace("ANIMSTEPS 3", "ANIMSTEPS 5"),
                     "short.axsf")
    traj = T.read_xsf(path)
    assert traj.n_frames == 3
    assert sorted(traj.skipped) == [3, 4]
    assert all("not in the file" in r for r in traj.skipped.values())


# ---------------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------------

def test_pdb_ase_models_with_a_cryst1_each():
    traj = T.read_pdb_trajectory(DATA / "ase_models.pdb")
    assert traj.file_format == "pdb-models"
    assert (traj.n_atoms, traj.n_frames) == (6, 3)
    assert traj.box_varies is True
    for k in range(3):
        frame = traj.frame(k)
        assert frame.elements.tolist() == ELEMENTS
        # CRYST1 keeps 3 decimals of a, b, c and 2 of the angles.
        assert np.abs(frame.box_ang - BOX[k]).max() < 2e-3
        # Coordinates keep 3 decimals; the wrap uses the CRYST1 box.
        assert lattice_gap(frame.cart_ang, POS[k], frame.box_ang) < 1e-3


def test_pdb_mdanalysis_one_cryst1_for_every_frame():
    traj = T.read_pdb_trajectory(DATA / "mda_models.pdb")
    assert traj.n_frames == 3
    box0 = traj.frame(0).box_ang
    for k in range(3):
        frame = traj.frame(k)
        assert frame.elements.tolist() == ELEMENTS   # 'SI', 'NA' columns
        assert np.array_equal(frame.box_ang, box0)
        assert lattice_gap(frame.cart_ang, POS[k], box0) < 1e-3
    assert traj.box_varies is False
    assert any("one CRYST1 record" in n for n in traj.notes)


def test_pdb_frames_ended_by_end_records(tmp_path):
    def cp2k_style(text):
        out = [x for x in text.splitlines() if not x.startswith("MODEL")]
        return "\n".join("END" if x.startswith("ENDMDL") else x
                         for x in out) + "\n"
    path = copy_with(tmp_path, "ase_models.pdb", cp2k_style, "pos-1.pdb")
    assert T.is_multiframe_pdb(path)
    assert T.sniff_pdb_trajectory(md_readers._head(path), path)
    traj = T.read_pdb_trajectory(path)
    assert traj.n_frames == 3
    assert any("ended by END" in n for n in traj.notes)
    for k in range(3):
        assert lattice_gap(traj.frame(k).cart_ang, POS[k],
                           traj.frame(k).box_ang) < 1e-3


def test_pdb_one_model_is_left_to_the_crystal_reader(tmp_path):
    def first_model(text):
        return text[:text.index("ENDMDL") + len("ENDMDL\n")]
    path = copy_with(tmp_path, "ase_models.pdb", first_model, "one.pdb")
    assert not T.is_multiframe_pdb(path)
    assert not T.sniff_pdb_trajectory(md_readers._head(path), path)
    assert T.read_pdb_trajectory(path).n_frames == 1


def test_pdb_placeholder_cell_is_refused(tmp_path):
    def placeholder(text):
        return re.sub(r"^CRYST1.*$",
                      "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 "
                      "P 1           1", text, flags=re.M)
    path = copy_with(tmp_path, "ase_models.pdb", placeholder, "nmr.pdb")
    with pytest.raises(UnsupportedFormat, match="nmr.pdb.*placeholder"):
        T.read_pdb_trajectory(path)


def test_pdb_blank_element_columns(tmp_path):
    def blank(text):
        return "\n".join(x[:76] if x.startswith("ATOM") else x
                         for x in text.splitlines()) + "\n"
    path = copy_with(tmp_path, "ase_models.pdb", blank, "noel.pdb")
    traj = T.read_pdb_trajectory(path)
    assert traj.frame(0).elements.tolist() == ELEMENTS   # names Si, O, Na
    assert any("element columns (77-78) are blank" in n for n in traj.notes)
    # A PDB atom name such as CA (an alpha carbon) is not read as calcium.
    text = path.read_text(encoding="utf-8").replace("   Na MOL", "   CA MOL")
    calpha = tmp_path / "ca.pdb"
    calpha.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"ca.pdb.*'CA'.*type_map="):
        T.read_pdb_trajectory(calpha)


def test_pdb_model_with_another_count_is_skipped(tmp_path):
    def drop_atom(text):
        lines = text.splitlines(keepends=True)
        second = [i for i, x in enumerate(lines) if x.startswith("MODEL")][1]
        del lines[second + 3]
        return "".join(lines)
    path = copy_with(tmp_path, "ase_models.pdb", drop_atom, "five.pdb")
    traj = T.read_pdb_trajectory(path)
    assert traj.n_frames == 2
    assert traj.skipped == {1: "5 atoms, where frame 0 holds 6"}


def test_pdb_cut_last_model_is_skipped(tmp_path):
    text = (DATA / "ase_models.pdb").read_text(encoding="utf-8")
    path = tmp_path / "cut.pdb"
    path.write_text(text[:text.rindex("ENDMDL")], encoding="utf-8")
    traj = T.read_pdb_trajectory(path)
    assert traj.n_frames == 2
    assert "truncated" in traj.skipped[2]


# ---------------------------------------------------------------------------
# IMD
# ---------------------------------------------------------------------------

def test_imd_with_a_type_map():
    traj = T.read_imd(DATA / "ovito.imd", type_map=IMD_MAP)
    assert traj.file_format == "imd"
    frame = traj.frame(0)
    assert frame.atom_id.tolist() == [1, 2, 3, 4, 5, 6]
    assert frame.elements.tolist() == ELEMENTS
    assert np.array_equal(frame.box_ang, BOX[0])
    assert lattice_gap(frame.cart_ang, POS[0], BOX[0]) < 1e-12
    assert traj.type_map_source == "user"
    assert any("no origin" in n for n in traj.notes)


def test_imd_untyped_refusal_lists_this_files_types():
    with pytest.raises(ValueError) as caught:
        T.read_imd(DATA / "ovito.imd")
    message = str(caught.value)
    assert message.startswith("ovito.imd")
    assert "type_map={1: '<element>', 2: '<element>', 3: '<element>'}" \
        in message


def test_imd_masses_name_the_types():
    traj = T.read_imd(DATA / "ovito_mass.imd")
    assert traj.frame(0).elements.tolist() == ELEMENTS
    assert traj.type_map == IMD_MAP
    assert traj.type_map_source == "data-file masses"
    assert any("mass-column mass" in n for n in traj.notes)
    assert any("vx, vy, vz" in n and "no unit" in n for n in traj.notes)
    assert traj.frame(0).vel_ang_per_ps is None


def test_imd_binary_is_refused(tmp_path):
    path = copy_with(tmp_path, "ovito.imd",
                     lambda s: s.replace("#F A", "#F B"), "binary.imd")
    with pytest.raises(UnsupportedFormat, match="binary.imd.*ASCII"):
        T.read_imd(path)


def test_imd_series(tmp_path):
    paths = []
    for k in range(2):
        paths.append(tmp_path / f"conf.{k}.imd")
        shutil.copy(DATA / "ovito.imd", paths[-1])
    traj = T.read_imd_series(paths, type_map=IMD_MAP)
    assert traj.n_frames == 2
    assert traj.source_paths == tuple(str(p.resolve()) for p in paths)
    assert np.array_equal(traj.frame(1).cart_ang, traj.frame(0).cart_ang)
    # A file of the series without a box is skipped, and named.
    paths.append(tmp_path / "conf.2.imd")
    paths[-1].write_text((DATA / "ovito.imd").read_text(encoding="utf-8")
                         .replace("#Z -0.8 0.6 9.0\n", ""), encoding="utf-8")
    traj = T.read_imd_series(paths, type_map=IMD_MAP)
    assert traj.n_frames == 2
    assert traj.skipped[2].startswith("unreadable: conf.2.imd")
    assert "#Z" in traj.skipped[2]


# ---------------------------------------------------------------------------
# POSCAR series
# ---------------------------------------------------------------------------

def test_poscar_series_ase():
    paths = sorted(DATA.glob("POSCAR_ase_*"))
    traj = T.read_poscar_series(paths)
    assert traj.file_format == "vasp-poscar"
    assert (traj.n_atoms, traj.n_frames) == (6, 3)
    assert traj.box_varies is True
    for k in range(3):
        frame = traj.frame(k)
        # ASE writes one species group per run: Si O Na O Si.
        assert frame.elements.tolist() == ELEMENTS
        assert np.abs(frame.box_ang - BOX[k]).max() < 1e-12
        assert lattice_gap(frame.cart_ang, POS[k], BOX[k]) < 1e-12


def test_poscar_series_ovito_sorted_by_type():
    paths = sorted(DATA.glob("ovito_poscar.*.vasp"))
    traj = T.read_poscar_series(paths)
    assert traj.n_frames == 3
    for k in range(3):
        frame = traj.frame(k)
        assert frame.composition == {"Na": 1, "O": 3, "Si": 2}
        inv = np.linalg.inv(BOX[k])
        for atom in range(6):
            same = np.array(ELEMENTS) == frame.elements[atom]
            d = (POS[k][same] - frame.cart_ang[atom]) @ inv
            d -= np.round(d)
            assert np.linalg.norm(d @ BOX[k], axis=1).min() < 1e-9


def test_poscar_series_other_composition_is_skipped():
    paths = [DATA / "POSCAR_ase_000", DATA / "ovito_poscar.1.vasp",
             DATA / "POSCAR_ase_002"]
    traj = T.read_poscar_series(paths)
    assert traj.n_frames == 2
    assert list(traj.skipped) == [1]
    assert "ovito_poscar.1.vasp" in traj.skipped[1]
    assert len(traj.source_paths) == 3


def test_poscar_series_name_order_note():
    paths = [DATA / "POSCAR_ase_002", DATA / "POSCAR_ase_000",
             DATA / "POSCAR_ase_001"]
    traj = T.read_poscar_series(paths)
    note = next(n for n in traj.notes if "do not increase" in n)
    assert "POSCAR_ase_002 before POSCAR_ase_000" in note
    assert lattice_gap(traj.frame(0).cart_ang, POS[2], BOX[2]) < 1e-12


def test_poscar_vasp4_layout(tmp_path):
    def vasp4(text, title):
        lines = text.splitlines(keepends=True)
        return title + "\n" + "".join(lines[1:5] + lines[6:])
    path = copy_with(tmp_path, "POSCAR_ase_000",
                     lambda s: vasp4(s, "Si O Na O Si"), "POSCAR_v4")
    traj = T.read_poscar_file(path)
    assert traj.frame(0).elements.tolist() == ELEMENTS
    assert lattice_gap(traj.frame(0).cart_ang, POS[0], BOX[0]) < 1e-12
    assert any("VASP 4 layout" in n for n in traj.notes)
    path = copy_with(tmp_path, "POSCAR_ase_000",
                     lambda s: vasp4(s, "glass model"), "POSCAR_v4b")
    with pytest.raises(ValueError, match="POSCAR_v4b.*VASP 4.*element names"):
        T.read_poscar_file(path)


def test_poscar_sniff_takes_poscar_names_only(tmp_path):
    """POSCAR content is claimed only under a name readers.read gives its
    POSCAR reader (and for which it consults no format module); under any
    other name the crystal reader keeps it."""
    spec = next(s for s in T.FORMATS if s.name == "vasp-poscar")
    for path in [*DATA.glob("POSCAR_ase_*"), *DATA.glob("ovito_poscar.*")]:
        head = md_readers._head(path)
        assert spec.sniff(head, path)
        assert T.sniff_poscar(head, path)
    other = tmp_path / "glass_model"
    shutil.copy(DATA / "POSCAR_ase_000", other)
    head = md_readers._head(other)
    assert T.sniff_poscar(head, other) and not spec.sniff(head, other)
    for name in ("XDATCAR_fixed", "XDATCAR_variable"):
        path = MD_DATA / name
        assert not T.sniff_poscar(md_readers._head(path), path)


# ---------------------------------------------------------------------------
# recognition, file variants, the contract
# ---------------------------------------------------------------------------

OWN = {
    "castep-md": {"ase.md", "ase.geom", "castep_doc_example.md"},
    "gromacs-gro": {"ase_vel.gro", "mdtraj.gro"},
    "xsf": {"ase_var.axsf", "ase_fixed.axsf", "pmg_single.xsf"},
    "pdb-models": {"ase_models.pdb", "mda_models.pdb", "mda_resid_wrap.pdb",
                   "cp2k_layout.pdb"},
    "imd": {"ovito.imd", "ovito_mass.imd"},
    "vasp-poscar": {"POSCAR_ase_000", "POSCAR_ase_001", "POSCAR_ase_002",
                    "ovito_poscar.0.vasp", "ovito_poscar.1.vasp",
                    "ovito_poscar.2.vasp"},
}


def test_sniffers_claim_their_own_files_only():
    files = [p for p in DATA.iterdir() if p.is_file()] + \
        [p for p in MD_DATA.iterdir() if p.is_file()]
    for spec in T.FORMATS:
        claimed = {p.name for p in files
                   if spec.sniff(md_readers._head(p), p)}
        assert claimed == OWN[spec.name], spec.name


@pytest.mark.parametrize("name, reader", [
    ("ase.md", T.read_castep_md), ("mdtraj.gro", T.read_gro),
    ("ase_var.axsf", T.read_xsf), ("ase_models.pdb", T.read_pdb_trajectory)])
def test_crlf_and_gzip_copies_read_identically(tmp_path, name, reader):
    raw = (DATA / name).read_bytes()
    crlf = tmp_path / f"crlf_{name}"
    crlf.write_bytes(raw.replace(b"\n", b"\r\n"))
    gz = tmp_path / f"{name}.gz"
    gz.write_bytes(gzip.compress(raw))
    reference = reader(DATA / name)
    for variant in (crlf, gz):
        traj = reader(variant)
        assert traj.n_frames == reference.n_frames
        for k in range(traj.n_frames):
            a, b = traj.frame(k), reference.frame(k)
            assert np.array_equal(a.cart_ang, b.cart_ang)
            assert np.array_equal(a.box_ang, b.box_ang)
        traj.close()


def test_trajectories_pickle():
    traj = T.read_castep_md(DATA / "ase.md")
    clone = pickle.loads(pickle.dumps(traj))
    assert np.array_equal(clone.frame(2).cart_ang, traj.frame(2).cart_ang)
    series = T.read_poscar_series(sorted(DATA.glob("POSCAR_ase_*")))
    clone = pickle.loads(pickle.dumps(series))
    assert np.array_equal(clone.frame(1).cart_ang, series.frame(1).cart_ang)
    for traj in (T.read_pdb_trajectory(DATA / "cp2k_layout.pdb"),
                 T.read_xsf(DATA / "ase_var.axsf")):
        clone = pickle.loads(pickle.dumps(traj))
        assert np.array_equal(clone.frame(2).cart_ang, traj.frame(2).cart_ang)
        assert clone.frame(2).timestep == traj.frame(2).timestep


def test_formatspec_contract():
    names = [spec.name for spec in T.FORMATS]
    assert len(set(names)) == len(names)
    assert "md_formats_text" in md_formats_base.FORMAT_MODULES
    for spec in T.FORMATS:
        assert isinstance(spec, md_formats_base.FormatSpec)
        assert spec.options <= md_formats_base.KNOWN_OPTIONS
        for function in (spec.read, spec.read_series):
            if function is None:
                continue
            keywords = {p.name for p in inspect.signature(function)
                        .parameters.values() if p.kind == p.KEYWORD_ONLY}
            assert keywords == set(spec.options), spec.name


@pytest.mark.parametrize("name, file_format, options", [
    ("ase.md", "castep-md", {}), ("mdtraj.gro", "gromacs-gro", {}),
    ("ase_var.axsf", "xsf", {}), ("ase_models.pdb", "pdb-models", {}),
    ("ovito.imd", "imd", {"type_map": IMD_MAP}),
    ("POSCAR_ase_001", "vasp-poscar", {})])
def test_read_trajectory_reaches_these_formats(name, file_format, options):
    traj = md_readers.read_trajectory(DATA / name, **options)
    assert traj.file_format == file_format
    assert traj.frame(0).elements.tolist() == ELEMENTS


def test_read_trajectory_reads_a_poscar_series_and_readers_keeps_crystals(
        tmp_path):
    from facet.core import readers

    traj = md_readers.read_trajectory(sorted(DATA.glob("POSCAR_ase_*")))
    assert (traj.file_format, traj.n_frames) == ("vasp-poscar", 3)
    assert lattice_gap(traj.frame(2).cart_ang, POS[2], BOX[2]) < 1e-12
    # One POSCAR, under its name or another, is a crystal for readers.read.
    other = tmp_path / "glass_model"
    shutil.copy(DATA / "POSCAR_ase_000", other)
    for path in (DATA / "POSCAR_ase_000", other):
        assert len(readers.read(path).atoms) == 6
    # A PDB of several models is routed to the MD reader as this format.
    with pytest.raises(readers.MDModelFile) as caught:
        readers.read(DATA / "ase_models.pdb")
    assert caught.value.file_format == "pdb-models"


VERDICT = re.compile(r"\b(good|bad|poor|excellent|acceptable|correct|incorrect|"
                     r"wrong|reliable|unreliable|trustworthy|untrustworthy|"
                     r"unusable|should|proves|confirms)\b", re.I)


def test_no_verdict_words_and_no_other_dependencies():
    source = Path(T.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not VERDICT.search(node.value), node.value[:80]
    allowed = {"numpy", "scipy", "gemmi", "__future__", "re", "bisect",
               "threading", "collections", "dataclasses", "pathlib", "gzip"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots = {node.module.split(".")[0]}
        else:
            continue
        assert roots <= allowed, roots


# ---------------------------------------------------------------------------
# defects found by the verification passes: each test fails without its fix
# ---------------------------------------------------------------------------

def _pdb_columns(text, model, atom, start, stop, field):
    """PDB text with columns start+1 .. stop of one ATOM record replaced
    (model and atom counted from 0)."""
    lines = text.splitlines(keepends=True)
    first = [i for i, x in enumerate(lines) if x.startswith("MODEL")][model]
    i = first + 1 + atom
    lines[i] = lines[i][:start] + field + lines[i][stop:]
    return "".join(lines)


def _cryst1(text, values, which=None):
    """PDB text whose CRYST1 records (or the one of model ``which``) give
    a, b, c, alpha, beta, gamma."""
    lines = text.splitlines(keepends=True)
    crysts = [i for i, x in enumerate(lines) if x.startswith("CRYST1")]
    for n, i in enumerate(crysts):
        if which is None or n == which:
            a, b, c, al, be, ga = values
            lines[i] = (f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}"
                        f"{ga:7.2f} P 1\n")
    return "".join(lines)


def _standard_box(a, b, c, alpha, beta, gamma):
    """CRYST1's standard orthogonalisation: a along x, b in the xy plane."""
    al, be, ga = np.radians([alpha, beta, gamma])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.array([[a, 0.0, 0.0], [b * np.cos(ga), b * np.sin(ga), 0.0],
                     [cx, cy, np.sqrt(c * c - cx * cx - cy * cy)]])


def _with_scale(text, cell=None):
    """SCALEn records after each CRYST1, written %10.6f as crystallographic
    programs write them, from the CRYST1 values or from ``cell`` (the
    unrounded cell the CRYST1 record rounds)."""
    out = []
    for line in text.splitlines(keepends=True):
        out.append(line)
        if line.startswith("CRYST1"):
            values = cell if cell is not None else [
                float(line[a:b]) for a, b in ((6, 15), (15, 24), (24, 33),
                                              (33, 40), (40, 47), (47, 54))]
            inverse = np.linalg.inv(_standard_box(*values).T)
            for i in range(3):
                out.append(f"SCALE{i + 1}    {inverse[i, 0]:10.6f}"
                           f"{inverse[i, 1]:10.6f}{inverse[i, 2]:10.6f}     "
                           f"{0.0:10.5f}\n")
    return "".join(out)


def _six_models():
    """ase_models.pdb with its three models written again as models 4-6."""
    text = (DATA / "ase_models.pdb").read_text(encoding="utf-8")
    again = re.sub(r"^MODEL +(\d+)$",
                   lambda m: f"MODEL     {int(m.group(1)) + 3}", text,
                   flags=re.M)
    return text + again


def test_pdb_repeated_residue_numbers_keep_the_file_order():
    """MDAnalysis keeps the last four digits of a residue number, so a model
    of more than 9999 residues repeats them; gemmi files such atoms under the
    first residue of that number, so the coordinates are read from their
    columns in file order."""
    traj = md_readers.read_trajectory(DATA / "mda_resid_wrap.pdb")
    assert (traj.file_format, traj.n_atoms, traj.n_frames) == \
        ("pdb-models", 6, 3)
    assert traj.skipped == {}
    for k in range(3):
        frame = traj.frame(k)
        assert frame.elements.tolist() == ELEMENTS
        assert lattice_gap(frame.cart_ang, POS[k], frame.box_ang) < 1e-3


@pytest.mark.parametrize("field", ["   X.747", "********", "   1.2.3",
                                   "     ---", "    1e99"])
def test_pdb_coordinate_text_that_is_no_coordinate_is_refused(tmp_path,
                                                              field):
    """gemmi reads such a z field as 0 (or 1.2); FACET reads the columns."""
    path = copy_with(tmp_path, "ase_models.pdb",
                     lambda s: _pdb_columns(s, 1, 4, 46, 54, field),
                     "zfield.pdb")
    traj = md_readers.read_trajectory(path)
    for k in (0, 2):
        frame = traj.frame(k)
        assert lattice_gap(frame.cart_ang, POS[k], frame.box_ang) < 1e-3
    with pytest.raises(FrameError) as caught:
        traj.frame(1)
    message = str(caught.value)
    assert "zfield.pdb" in message and "line 16" in message
    assert field.strip() in message
    assert not VERDICT.search(message)


def test_pdb_damaged_first_and_last_frames(tmp_path):
    path = copy_with(tmp_path, "ase_models.pdb",
                     lambda s: _pdb_columns(s, 0, 2, 30, 38, "  ABCDEF"),
                     "first.pdb")
    with pytest.raises(FrameError, match=r"first\.pdb.*line 5.*ABCDEF"):
        md_readers.read_trajectory(path)
    path = copy_with(tmp_path, "ase_models.pdb",
                     lambda s: _pdb_columns(s, 2, 2, 30, 38, "  ABCDEF"),
                     "last.pdb")
    traj = md_readers.read_trajectory(path)
    assert traj.n_frames == 2 and list(traj.skipped) == [2]
    assert "ABCDEF" in traj.skipped[2]
    assert not traj.skipped[2].startswith("truncated")


def test_pdb_short_atom_record_is_described_in_facets_words(tmp_path):
    def cut(text):
        lines = text.splitlines(keepends=True)
        lines[15] = lines[15][:50] + "\n"       # model 2, atom 5: z cut
        return "".join(lines)
    path = copy_with(tmp_path, "ase_models.pdb", cut, "short.pdb")
    traj = md_readers.read_trajectory(path)
    with pytest.raises(FrameError) as caught:
        traj.frame(1)
    message = str(caught.value)
    assert "short.pdb" in message and "line 16" in message
    assert "column 50" in message
    assert not VERDICT.search(message) and "gemmi" not in message


@pytest.mark.parametrize("values, why", [
    ((0, 0, 0, 0, 0, 0), "length"),
    ((-8.0, 7.595, 9.055, 87.05, 95.07, 80.91), "length"),
    ((8.0, 7.595, 9.055, 0.0, 95.07, 80.91), "angle"),
    ((8.0, 7.595, 9.055, 87.05, 95.07, 180.0), "angle"),
    ((8.0, 7.595, 9.055, 150.0, 150.0, 150.0), "angles")])
def test_pdb_cryst1_that_gives_no_cell_is_refused(tmp_path, values, why):
    """gemmi turns CRYST1 zeros into a 1 Å cube and raises RuntimeError
    for a zero angle; the record's values are checked first."""
    path = copy_with(tmp_path, "ase_models.pdb",
                     lambda s: _cryst1(s, values), "cell.pdb")
    with pytest.raises(UnsupportedFormat) as caught:
        md_readers.read_trajectory(path)
    message = str(caught.value)
    assert message.startswith("cell.pdb") and "CRYST1" in message
    assert why in message and "periodic box" in message
    assert "Lattice=" in message                # what to do instead
    assert "Impossible" not in message          # gemmi's own text


def test_pdb_later_cryst1_that_gives_no_cell_skips_its_frame(tmp_path):
    path = copy_with(
        tmp_path, "ase_models.pdb",
        lambda s: _cryst1(s, (8.1, 7.679, 9.141, 0.0, 94.39, 81.76), which=1),
        "cell1.pdb")
    traj = md_readers.read_trajectory(path)
    assert traj.n_frames == 2 and list(traj.skipped) == [1]
    assert "CRYST1" in traj.skipped[1] and "angle" in traj.skipped[1]
    for k in range(traj.n_frames):
        traj.frame(k)


def test_pdb_scale_records_of_the_cryst1_frame_are_accepted(tmp_path):
    # Every model's SCALEn sits within the first 4 kB of this small file.
    path = copy_with(tmp_path, "ase_models.pdb", _with_scale, "scale.pdb")
    traj = md_readers.read_trajectory(path)
    assert traj.n_frames == 3 and traj.skipped == {}
    # SCALEn from the unrounded cell of an elongated box differ from the
    # inverse of the CRYST1 box by the rounding of both records (1.4e-6),
    # more than 1e-4 / 78.9 Å.
    cell = (21.3004, 22.7004, 78.9004, 90.0, 90.0, 90.0)
    path = copy_with(tmp_path, "ase_models.pdb",
                     lambda s: _with_scale(_cryst1(s, cell), cell), "long.pdb")
    traj = md_readers.read_trajectory(path)
    assert traj.n_frames == 3 and traj.skipped == {}
    assert np.abs(traj.frame(2).box_ang
                  - np.diag([21.3, 22.7, 78.9])).max() < 1e-12


def test_pdb_scale_records_of_a_rotated_frame_are_refused(tmp_path):
    def rotated(text):
        swap = {"SCALE1": "SCALE2", "SCALE2": "SCALE1"}
        return "".join(swap.get(x[:6], x[:6]) + x[6:]
                       for x in _with_scale(text).splitlines(keepends=True))
    path = copy_with(tmp_path, "ase_models.pdb", rotated, "rot.pdb")
    with pytest.raises(UnsupportedFormat, match=r"rot\.pdb.*SCALEn"):
        md_readers.read_trajectory(path)


def test_pdb_cp2k_layout_steps_and_times():
    """CP2K writes 'REMARK Step <it>, time = <fs>, E = <etot>' before each
    frame (motion_utils.F)."""
    traj = md_readers.read_trajectory(DATA / "cp2k_layout.pdb")
    assert (traj.file_format, traj.n_frames) == ("pdb-models", 3)
    assert traj.timesteps.tolist() == [0, 250, 500]
    assert traj.times_ps == pytest.approx(TIMES, abs=1e-12)
    for k in range(3):
        frame = traj.frame(k)
        assert frame.timestep == [0, 250, 500][k]
        assert frame.elements.tolist() == ELEMENTS
        assert lattice_gap(frame.cart_ang, POS[k], frame.box_ang) < 1e-3
    assert not any("holds no time" in n for n in traj.notes)
    assert any("REMARK" in n and "fs" in n for n in traj.notes)


def test_pdb_single_cryst1_note_names_both_writers():
    traj = T.read_pdb_trajectory(DATA / "mda_models.pdb")
    note = next(n for n in traj.notes if "one CRYST1 record" in n)
    assert "MDAnalysis" in note and "mdtraj" in note


def test_pdb_atoms_outside_model_blocks_are_accounted_for(tmp_path):
    path = copy_with(tmp_path, "ase_models.pdb",
                     lambda s: re.sub(r"^MODEL +2\n", "", s, flags=re.M),
                     "nomodel2.pdb")
    traj = md_readers.read_trajectory(path)
    # Model 2's records hold the model's atoms and their CRYST1: read as
    # frame 1, with a note.
    assert traj.n_frames == 3 and traj.skipped == {}
    assert any("outside any MODEL" in n for n in traj.notes)
    assert lattice_gap(traj.frame(1).cart_ang, POS[1],
                       traj.frame(1).box_ang) < 1e-3
    # One CP2K-layout frame put before a MODEL trajectory ('cat a b').
    cp2k = (DATA / "cp2k_layout.pdb").read_text(encoding="utf-8")
    path = tmp_path / "cat.pdb"
    path.write_text(cp2k[:cp2k.index("\nEND\n") + 5]
                    + (DATA / "ase_models.pdb").read_text(encoding="utf-8"),
                    encoding="utf-8", newline="")
    traj = md_readers.read_trajectory(path)
    assert traj.n_frames + len(traj.skipped) == 4
    for k in range(traj.n_frames):
        traj.frame(k)
    # A MODEL block with no atom records is a position of its own.
    path = copy_with(tmp_path, "ase_models.pdb",
                     lambda s: s.replace("ENDMDL\n", "ENDMDL\nMODEL     9\n"
                                         "ENDMDL\n", 1), "empty.pdb")
    traj = md_readers.read_trajectory(path)
    assert traj.n_frames == 3 and list(traj.skipped) == [1]
    assert "no ATOM or HETATM records" in traj.skipped[1]


def test_pdb_lost_frame_delimiters_merge_frames_that_are_counted(tmp_path):
    lines = _six_models().splitlines(keepends=True)
    del lines[lines.index("ENDMDL\n")]
    del lines[lines.index("MODEL     2\n")]
    path = tmp_path / "merged.pdb"
    path.write_text("".join(lines), encoding="utf-8", newline="")
    traj = md_readers.read_trajectory(path)
    assert (traj.n_atoms, traj.n_frames) == (6, 4)
    assert sorted(traj.skipped) == [0, 1]
    assert all("merged" in traj.skipped[p] for p in (0, 1))
    for k, truth in enumerate((2, 0, 1, 2)):
        frame = traj.frame(k)
        assert lattice_gap(frame.cart_ang, POS[truth], frame.box_ang) < 1e-3
    # Three models, the first two merged: 12 and 6 atom records.
    lines = (DATA / "ase_models.pdb").read_text(encoding="utf-8") \
        .splitlines(keepends=True)
    del lines[lines.index("ENDMDL\n")]
    del lines[lines.index("MODEL     2\n")]
    path.write_text("".join(lines), encoding="utf-8", newline="")
    traj = md_readers.read_trajectory(path)
    assert (traj.n_atoms, traj.n_frames) == (6, 1)
    assert sorted(traj.skipped) == [0, 1]
    # CP2K's layout with its first END lost.
    text = (DATA / "cp2k_layout.pdb").read_text(encoding="utf-8")
    path.write_text(text.replace("\nEND\n", "\n", 1), encoding="utf-8",
                    newline="")
    traj = md_readers.read_trajectory(path)
    assert (traj.n_atoms, traj.n_frames) == (6, 1)
    assert sorted(traj.skipped) == [0, 1]
    assert traj.timesteps.tolist() == [500]


def test_pdb_sniff_and_readers_read_take_the_same_decision(tmp_path,
                                                           monkeypatch):
    """readers.read routes a .pdb by md_readers.pdb_model_count, which reads
    as far as it needs; the sniff decides the same way (it stopped at 32 MB,
    so a PDB with a larger first frame was routed and then refused), and
    reads END records written ' END' as pdb_model_count does."""
    monkeypatch.setattr(T, "_PDB_SNIFF_BYTES", 300, raising=False)
    files = [DATA / n for n in ("ase_models.pdb", "mda_models.pdb",
                                "cp2k_layout.pdb", "mda_resid_wrap.pdb")]
    files.append(copy_with(tmp_path, "cp2k_layout.pdb",
                           lambda s: s.replace("\nEND\n", "\n END\n"),
                           "space_end.pdb"))
    files.append(copy_with(tmp_path, "ase_models.pdb",
                           lambda s: s[:s.index("ENDMDL") + 7], "one.pdb"))
    for path in files:
        claimed = T.sniff_pdb_trajectory(md_readers._head(path), path)
        assert claimed == (md_readers.pdb_model_count(path) >= 2), path.name
    traj = md_readers.read_trajectory(files[4])
    assert traj.n_frames == 3
    with pytest.raises(readers.MDModelFile) as caught:
        readers.read(files[4])
    assert caught.value.file_format == "pdb-models"


def test_text_formats_route_by_content_not_by_name(tmp_path):
    """read_trajectory recognises these formats by content alone, so a claim
    on a file name could only send a crystal file to a reader that refuses
    it: a POSCAR saved as .gro, .axsf or .imd stays a crystal."""
    for spec in T.FORMATS:
        assert spec.extensions == () and spec.stems == (), spec.name
    for suffix in (".gro", ".axsf", ".imd"):
        path = tmp_path / f"crystal{suffix}"
        shutil.copy(DATA / "POSCAR_ase_000", path)
        assert len(readers.read(path).atoms) == 6


def test_xsf_lost_primvec_of_a_variable_cell_step_is_skipped(tmp_path):
    def drop_primvec_2(text):
        lines = text.splitlines(keepends=True)
        i = lines.index("PRIMVEC 2\n")
        return "".join(lines[:i] + lines[i + 4:])
    path = copy_with(tmp_path, "ase_var.axsf", drop_primvec_2, "nopv2.axsf")
    traj = md_readers.read_trajectory(path)
    assert traj.n_frames == 2 and list(traj.skipped) == [1]
    assert "PRIMVEC 2" in traj.skipped[1]
    assert np.abs(traj.frame(1).box_ang - BOX[2]).max() < 1e-12


def test_xsf_step_numbers_place_the_steps(tmp_path):
    def drop(text, line):
        lines = text.splitlines(keepends=True)
        lines.remove(line)
        return "".join(lines)
    # Fixed cell: step 2's lines run on after step 1's atoms.
    path = copy_with(tmp_path, "ase_fixed.axsf",
                     lambda s: drop(s, "PRIMCOORD 2\n"), "nopc2.axsf")
    traj = md_readers.read_trajectory(path)
    assert traj.file_positions.tolist() == [0, 2]
    assert list(traj.skipped) == [1] and "PRIMCOORD 2" in traj.skipped[1]
    assert lattice_gap(traj.frame(1).cart_ang, POS[2], BOX[0]) < 1e-12
    # Variable cell.
    path = copy_with(tmp_path, "ase_var.axsf",
                     lambda s: drop(s, "PRIMCOORD 2\n"), "nopc2v.axsf")
    traj = md_readers.read_trajectory(path)
    assert traj.file_positions.tolist() == [0, 2]
    assert list(traj.skipped) == [1]
    assert np.abs(traj.frame(1).box_ang - BOX[2]).max() < 1e-12
    # A step number written twice: read as the step its place gives.
    path = copy_with(tmp_path, "ase_fixed.axsf",
                     lambda s: s.replace("PRIMCOORD 3\n", "PRIMCOORD 2\n"),
                     "twice.axsf")
    traj = md_readers.read_trajectory(path)
    assert traj.n_frames == 3 and traj.skipped == {}
    assert any("PRIMCOORD 2" in n and "twice" in n for n in traj.notes)


def test_xsf_animsteps_far_beyond_the_file_is_one_count(tmp_path):
    path = copy_with(tmp_path, "ase_fixed.axsf",
                     lambda s: s.replace("ANIMSTEPS 3", "ANIMSTEPS 1000000"),
                     "huge.axsf")
    traj = md_readers.read_trajectory(path)
    assert traj.n_frames == 3
    # One entry per lost step up to T._XSF_LISTED_MISSING, then a count
    # (an entry for each of the 999 997 took 217 MB).
    assert len(traj.skipped) < 999997
    assert len(traj.skipped) == T._XSF_LISTED_MISSING
    assert sorted(traj.skipped)[:2] == [3, 4]
    note = next(n for n in traj.notes if "ANIMSTEPS" in n and "999997" in n)
    assert str(999997 - T._XSF_LISTED_MISSING) in note


def test_xsf_one_structure_reads_as_one_frame():
    """pymatgen writes a one-structure XSF with no line ending after its last
    line; no crystal reader opens XSF, so the MD reader does."""
    traj = md_readers.read_trajectory(DATA / "pmg_single.xsf")
    assert (traj.file_format, traj.n_frames) == ("xsf", 1)
    frame = traj.frame(0)
    assert frame.elements.tolist() == ELEMENTS
    assert np.abs(frame.box_ang - BOX[0]).max() < 1e-12
    assert lattice_gap(frame.cart_ang, POS[0], BOX[0]) < 1e-12
    assert any("line ending" in n for n in traj.notes)
    with pytest.raises(readers.MDModelFile) as caught:
        readers.read(DATA / "pmg_single.xsf")
    assert caught.value.file_format == "xsf"


def test_last_line_without_a_line_ending(tmp_path):
    raw = (DATA / "pmg_single.xsf").read_bytes()
    cut = tmp_path / "cut.xsf"
    cut.write_bytes(raw[:-3])                   # inside the last number
    with pytest.raises(ValueError, match=r"cut\.xsf.*truncated"):
        T.read_xsf(cut)
    text = (DATA / "mdtraj.gro").read_bytes().rstrip(b"\n")
    whole = tmp_path / "noeol.gro"
    whole.write_bytes(text)
    traj = T.read_gro(whole)
    assert traj.n_frames == 3 and traj.skipped == {}
    part = tmp_path / "cut.gro"
    part.write_bytes(text[:-2])                 # inside the last box value
    traj = T.read_gro(part)
    assert traj.n_frames == 2 and "truncated" in traj.skipped[2]


def test_castep_time_lines_that_are_times_are_kept(tmp_path):
    """A CASTEP run writes its time in a.u. (41.34 a.u. per fs); only ASE's
    0, 1, 2 index pattern is not taken as time."""
    per_fs = 1e-3 / T.AU_TIME_PS
    def spaced(text):
        out, k = [], 0
        for line in text.splitlines(keepends=True):
            if re.fullmatch(r"\s*[-+0-9.E]+\s*", line):
                line = f"{k * per_fs:45.16E}\n"
                k += 1
            out.append(line)
        return "".join(out)
    path = copy_with(tmp_path, "ase.md", spaced, "fs.md")
    traj = T.read_castep_md(path)
    assert traj.times_ps == pytest.approx([0.0, 0.001, 0.002], abs=1e-15)


def test_castep_lost_cell_lines_merge_steps_that_are_counted(tmp_path):
    def drop_h(text, step):
        lines = text.splitlines(keepends=True)
        h = [i for i, x in enumerate(lines) if x.rstrip().endswith("<-- h")]
        for i in reversed(h[3 * step:3 * step + 3]):
            del lines[i]
        return "".join(lines)
    path = copy_with(tmp_path, "ase.md", lambda s: drop_h(s, 1), "noh1.md")
    traj = md_readers.read_trajectory(path)
    assert (traj.n_atoms, traj.n_frames) == (6, 1)
    assert sorted(traj.skipped) == [0, 1]
    assert all("merged" in r for r in traj.skipped.values())
    assert lattice_gap(traj.frame(0).cart_ang, POS[2], BOX[2]) < 1e-8
    path = copy_with(tmp_path, "ase.md", lambda s: drop_h(s, 2), "noh2.md")
    traj = md_readers.read_trajectory(path)
    assert (traj.n_atoms, traj.n_frames) == (6, 1)
    assert sorted(traj.skipped) == [1, 2]
    assert lattice_gap(traj.frame(0).cart_ang, POS[0], BOX[0]) < 1e-8


def test_later_frame_errors_name_the_file(tmp_path):
    def corrupt(text):                          # step 1, atom 2
        lines = text.splitlines(keepends=True)
        r = [i for i, x in enumerate(lines) if x.rstrip().endswith("<-- R")]
        lines[r[7]] = lines[r[7]].replace("E+00", "X+00", 1)
        return "".join(lines)
    path = copy_with(tmp_path, "ase.md", corrupt, "mid.md")
    traj = md_readers.read_trajectory(path)
    with pytest.raises(FrameError, match=r"frame 1 .*mid\.md"):
        traj.frame(1)
    # IMD series: the second file holds one atom fewer.
    paths = [tmp_path / f"conf.{k}.imd" for k in range(3)]
    lines = (DATA / "ovito.imd").read_text(encoding="utf-8") \
        .splitlines(keepends=True)
    for k, path in enumerate(paths):
        path.write_text("".join(lines[:-1] if k == 1 else lines),
                        encoding="utf-8", newline="")
    traj = T.read_imd_series(paths, type_map=IMD_MAP)
    with pytest.raises(FrameError, match=r"conf\.1\.imd"):
        traj.frame(1)
    # POSCAR series: the second file is cut.
    cut = copy_with(tmp_path, "POSCAR_ase_001",
                    lambda s: "".join(s.splitlines(keepends=True)[:11]),
                    "POSCAR_cut_001")
    traj = T.read_poscar_series([DATA / "POSCAR_ase_000", cut,
                                 DATA / "POSCAR_ase_002"])
    with pytest.raises(FrameError, match=r"POSCAR_cut_001"):
        traj.frame(1)


def _zero_velocity_fields(line, start, width):
    return line[:start] + f"{0.0:{width}.4f}" * 3 + line[start + 3 * width:]


def test_velocities_are_kept_when_only_frame_0_is_at_rest(tmp_path):
    """A run started from rest has zero velocities in frame 0 only; the
    velocities of every frame are kept. A file whose first and last frames
    hold only zeros (ASE's writers for a model without velocities) stores
    none."""
    lines = (DATA / "ase_vel.gro").read_text(encoding="utf-8") \
        .splitlines(keepends=True)
    rest = [lines[0], lines[1]] + [
        _zero_velocity_fields(x, 44, 8) for x in lines[2:8]] + [lines[8]]
    path = tmp_path / "rest.gro"
    path.write_text("".join(rest + lines), encoding="utf-8", newline="")
    traj = T.read_gro(path)
    assert traj.n_frames == 2
    assert np.array_equal(traj.frame(0).vel_ang_per_ps, np.zeros((6, 3)))
    assert np.abs(traj.frame(1).vel_ang_per_ps - VEL[0]).max() <= 5e-4
    def zero_step0(text):
        out, step = [], -1
        for line in text.splitlines(keepends=True):
            if line.rstrip().endswith("<-- h") and not out[-1].rstrip() \
                    .endswith("<-- h"):
                step += 1
            if step == 0 and line.rstrip().endswith("<-- V"):
                head, tail = line[:line.index("<--")], "<-- V\n"
                tokens = head.split()
                line = (f" {tokens[0]:<2s}{tokens[1]:>15s}"
                        + "".join(f"{0.0:27.16E}" for _ in range(3))
                        + "  " + tail)
            out.append(line)
        return "".join(out)
    path = copy_with(tmp_path, "ase.md", zero_step0, "rest.md")
    traj = T.read_castep_md(path)
    assert np.array_equal(traj.frame(0).vel_ang_per_ps, np.zeros((6, 3)))
    assert traj.frame(2).vel_ang_per_ps == pytest.approx(VEL[2], rel=1e-6)
    # Every frame at zero: no velocities, and a note.
    def zero_all(text):
        return re.sub(r"(?m)^(.*)<-- V$", lambda m: " ".join(
            m.group(1).split()[:2]) + "  0.0 0.0 0.0  <-- V", text)
    path = copy_with(tmp_path, "ase.md", zero_all, "zeros.md")
    traj = T.read_castep_md(path)
    assert traj.frame(1).vel_ang_per_ps is None
    assert any("exactly 0" in n for n in traj.notes)


def _gro_frames(count):
    lines = (DATA / "mdtraj.gro").read_text(encoding="utf-8") \
        .splitlines(keepends=True)
    return [list(lines[9 * (k % 3):9 * (k % 3) + 9]) for k in range(count)]


@pytest.mark.parametrize("count", ["60", "66666"])
def test_gro_atom_count_too_large_costs_one_frame(tmp_path, count):
    frames = _gro_frames(20)
    frames[5][1] = f" {count}\n"
    path = tmp_path / "count.gro"
    path.write_text("".join("".join(f) for f in frames), encoding="utf-8",
                    newline="")
    traj = T.read_gro(path)
    assert traj.n_frames == 19 and list(traj.skipped) == [5]
    assert not traj.skipped[5].startswith("truncated")
    assert lattice_gap(traj.frame(5).cart_ang, POS[6 % 3], BOX[6 % 3]) < 1e-5


def test_gro_blank_title_lines(tmp_path):
    frames = _gro_frames(3)
    for frame in frames:
        frame[0] = "\n"
    path = tmp_path / "blank.gro"
    path.write_text("".join("".join(f) for f in frames), encoding="utf-8",
                    newline="")
    traj = T.read_gro(path)
    assert traj.n_frames == 3 and traj.skipped == {}
    assert lattice_gap(traj.frame(2).cart_ang, POS[2], BOX[2]) < 1e-5


def test_gzipped_poscar_names_keep_their_poscar_name(tmp_path):
    for k in range(3):
        raw = (DATA / f"ovito_poscar.{k}.vasp").read_bytes()
        (tmp_path / f"ovito_poscar.{k}.vasp.gz").write_bytes(gzip.compress(raw))
    traj = md_readers.read_trajectory(str(tmp_path / "ovito_poscar.*.vasp.gz"))
    assert (traj.file_format, traj.n_frames) == ("vasp-poscar", 3)
    assert md_readers.read_trajectory(
        tmp_path / "ovito_poscar.0.vasp.gz").n_frames == 1


def test_poscar_series_with_long_headers_and_the_vasp4_layout(tmp_path):
    # 1500 atoms, Si and O alternating, one species group per atom, as ASE
    # writes a model in random order: the title (ASE writes the symbols
    # there too), name and count lines run past the 4 kB window.
    rng = np.random.default_rng(3)
    symbols = ["Si", "O"] * 750
    paths = []
    for k in range(2):
        rows = [" ".join(symbols), " 1.0", " 30.0 0.0 0.0", " 0.0 31.0 0.0",
                " 0.0 0.0 32.0", " ".join(symbols), " ".join("1" * 1500),
                "Direct"]
        rows += [" ".join(f"{v:.10f}" for v in row)
                 for row in rng.random((1500, 3))]
        paths.append(tmp_path / f"POSCAR_{k:03d}")
        paths[-1].write_text("\n".join(rows) + "\n", encoding="utf-8")
        assert len("\n".join(rows[:7])) > md_readers.SNIFF_BYTES
    traj = md_readers.read_trajectory(paths)
    assert (traj.file_format, traj.n_frames, traj.n_atoms) == \
        ("vasp-poscar", 2, 1500)
    # VASP 4 layout: element names on the title line, no name line.
    def vasp4(text):
        lines = text.splitlines(keepends=True)
        return "Si O Na O Si\n" + "".join(lines[1:5] + lines[6:])
    v4 = [copy_with(tmp_path, f"POSCAR_ase_{k:03d}", vasp4, f"POSCAR_v4_{k}")
          for k in range(3)]
    traj = md_readers.read_trajectory(v4)
    assert traj.n_frames == 3
    assert traj.frame(0).elements.tolist() == ELEMENTS


def test_vasp_run_directory_poscar_and_contcar_are_not_a_series(tmp_path):
    """POSCAR and CONTCAR are a VASP run's start and end; a directory gives
    them CONTCAR first, so as a series they would run backwards."""
    run = tmp_path / "run"
    run.mkdir()
    shutil.copy(DATA / "POSCAR_ase_000", run / "POSCAR")
    shutil.copy(DATA / "POSCAR_ase_002", run / "CONTCAR")
    shutil.copy(MD_DATA / "XDATCAR_fixed", run / "XDATCAR")
    with pytest.raises(UnsupportedFormat) as caught:
        md_readers.read_trajectory(run)
    message = str(caught.value)
    assert "XDATCAR" in message and "[POSCAR, CONTCAR]" in message
    traj = md_readers.read_trajectory([run / "POSCAR", run / "CONTCAR"])
    assert traj.n_frames == 2
    assert lattice_gap(traj.frame(1).cart_ang, POS[2], BOX[2]) < 1e-12
