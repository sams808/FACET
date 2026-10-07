"""Reading MD models: LAMMPS data and dump, extended XYZ, VASP XDATCAR, DL_POLY.

FACET does not run molecular dynamics; it reads what a simulation wrote and
measures it. This module turns the files a glass simulation leaves behind into
the :class:`~.md_model.Trajectory` of :class:`~.md_model.Frame` objects that the
bulk engine and the glass descriptors measure, and records everything it had to
assume on the way.

WHY A MODULE OF ITS OWN
-----------------------
Every reader in ``readers.py`` returns a crystal ``Structure`` through
``readers._assemble``, which runs spglib on every file, labels the cell P 1 and
sets ``COMMON_OX`` with ``ox_source = 'common'``; the crystal window then
resolves oxidation states from the geometry. On a 9 261-atom frame Step 0
measured the spglib search alone at 7.2 s of a 7.35 s read, and resolving
oxidation states from the geometry is what the MD path must not do (the states
of a model are the force field's inputs). So an MD file never goes through
``_assemble``: ``readers.read`` recognises it and raises
``readers.MDModelFile``, and :func:`read_trajectory` is the way in.

WHAT IS READ
------------
=================  ==========================  ===============================
format             what a frame can carry      source of the layout
=================  ==========================  ===============================
LAMMPS data        positions, box, ids, image  ``read_data`` page [1], and
                   flags, velocities, charges  ``write_data`` source [2]
LAMMPS dump        x | xs | xu | xsu, image    ``dump`` page [3] and
(text or gzip)     flags, velocities, charges, ``dump_custom.cpp`` [4]
                   time, units; type numbers,
                   type labels (also in the
                   type column) or elements
extended XYZ       species or Z, pos, velo,    libAtoms specification [5]
(multi-frame)      id, charge (also ASE's
                   initial_charges), Time,
                   Lattice, Origin; masses
                   check the species
VASP XDATCAR       direct or Cartesian         VASP wiki, POSCAR [6]
(fixed and         positions, a cell per
variable cell)     frame, the scale factor
DL_POLY CONFIG,    positions, velocities,      DL_POLY 4 user manual,
HISTORY            charges, masses, time       §5.1.2, §5.2.1, App. A [7]
=================  ==========================  ===============================

Forces and every other column are listed in the notes as not read, with the
reason when a column was looked at and not used (a lone ``vx``, image flags
beside ``xu yu zu``, ASE's ``momenta``, which would need the masses ASE used
and ASE's unit of time).

BOXES
-----
* **LAMMPS restricted triclinic** [8]: origin (xlo, ylo, zlo), a = (xhi - xlo,
  0, 0), b = (xy, yhi - ylo, 0), c = (xz, yz, zhi - zlo). :func:`box_from_lammps`
  and its inverse :func:`lammps_from_box`.
* **A dump's BOX BOUNDS are not the box.** For a tilted box LAMMPS writes the
  bounding box of the parallelepiped [8]::

      xlo_bound = xlo + MIN(0.0, xy, xz, xy+xz)    xhi_bound = xhi + MAX(0.0, xy, xz, xy+xz)
      ylo_bound = ylo + MIN(0.0, yz)               yhi_bound = yhi + MAX(0.0, yz)
      zlo_bound = zlo                              zhi_bound = zhi

  with xy, xz and yz as the third value of lines 1, 2 and 3 of the block
  (``dump_custom.cpp``, ``header_item_triclinic`` [4]). All three lines are read
  before any bound is inverted, because xlo needs xz from line 2.
  :func:`box_from_dump_bounds` inverts these; :func:`dump_bounds_from_box` is
  the forward form.
* **LAMMPS general triclinic** [8]: a data file gives ``avec``, ``bvec``,
  ``cvec`` and ``abc origin``; a dump writes ``ITEM: BOX BOUNDS abc origin`` and
  three lines ``ax ay az xlo`` / ``bx by bz ylo`` / ``cx cy cz zlo``
  (``header_item_triclinic_general`` [4]). The vectors are the rows as given.
* **Scaled coordinates include the origin.** LAMMPS computes
  ``xs = (x - boxlo) * h_inv`` (``pack_xs_triclinic`` [4]), so
  ``x = origin + xs @ box``; ``xu = x + image @ box`` (``pack_xu_triclinic``);
  ``xsu = xs + image`` (``pack_xsu``). Read from the LAMMPS source, which
  settles what Step 0 left unverified.
* **Extended XYZ** ``Lattice=`` lists the three vectors as rows [5]. An
  ``Origin=`` key, which some writers add and the specification does not
  define, is read when present; otherwise the origin is zero.
* **XDATCAR** follows the POSCAR header [6]: one scale factor multiplies the
  vectors, a negative one is the cell volume (the convention
  ``readers.read_poscar`` already uses, a factor ``(|s| / |det|)^(1/3)``), and
  three positive factors scale the x, y and z Cartesian components. With a
  variable cell VASP repeats the header before each configuration.
* **DL_POLY** places the coordinate origin at the centre of the cell: "Note
  that the atomic coordinate origin is the centre of the MD cell" [7, App. A],
  so ``origin_ang = -(a + b + c) / 2``. Boundary keys 1, 2 and 3 (cubic,
  orthorhombic, parallelepiped) are read; 0 (no box) and 6 (a slab with no
  period along z) are refused, because wrapping into a box that is not
  periodic moves atoms.
* **Non-periodic axes are not handled alike across formats yet:** DL_POLY
  keys 0 and 6 are refused, while LAMMPS boundary flags f, s or m and
  ``pbc=F`` in extended XYZ are noted and the box read as periodic along a,
  b and c. Which rule applies to all of them is Sam's open decision 2.

UNITS
-----
LAMMPS files do not carry their unit style, except through ``ITEM: UNITS``,
which ``dump_modify units yes`` writes once, in the first frame
(``header_item`` [4]), and the first line ``write_data`` writes ("... timestep
= N, units = S" [2]). Lengths are Å only in units ``metal`` and ``real`` [9]; the
other styles are refused. Times go to ps and velocities to Å/ps: unchanged in
``metal``; fs and Å/fs in ``real``, so x 1e-3 and x 1e3. With no stated style
the values are kept as written and ``units_note`` says what that assumes
(ASE's lammps-data writer states no style either). The extended XYZ format
states no unit for ``Time=`` or velocities, and LAMMPS's own ``dump extxyz``
writes them in the run's style (measured against LAMMPS 22 Jul 2025: Time=
in fs and velocities in Å/fs under units real), so ``units`` converts them as
for a dump; without it they are kept as written, with a note. DL_POLY writes
Å, Å/ps and ps [7].

TYPES TO ELEMENTS
-----------------
A LAMMPS type is a number. In order of preference, per type: the user's map; an
``element`` column; a type label (a ``typelabel`` column, labels in the type
column, or the data file's ``Atom Type Labels``) written as files write element
symbols; the data file's ``Masses`` matched to gemmi's standard atomic weights.
DL_POLY atom names, VASP species and extended XYZ species are labels too.

* **A label is a symbol only as files write symbols:** an upper-case letter,
  optionally a lower-case one (:data:`_FILE_SYMBOL`). Force fields write names
  such as HO (a hydroxyl hydrogen), CA or ho; matching them to Ho, Ca and Ho
  in any case gave a hydrogen holmium's identity, so such a label needs a map.
* **The file's own mass checks its labels** (a data file's Masses, DL_POLY's
  atom records, an extended XYZ ``masses`` column): a label whose mass in the
  same file lies nearer another element's weight than its own is refused
  ('Ho' at 1.008 amu); a mass within the tolerance, that weight rounded to a
  whole number, or a mass no other element lies nearer is accepted, and a
  difference beyond the tolerance is noted.
* **Masses alone** name an element only when exactly one lies within
  :data:`MASS_TOL_AMU`, compared with :data:`_MASS_EPS_AMU` of slack so that
  written decimals decide (Ge 72.63 against gemmi's 72.64); a whole-number mass
  is refused, because force fields round masses to integers and 16 or 28 amu
  names no element. gemmi lists elements without a standard atomic weight at a
  whole number (Po at 209), and an isotope's mass lies near it, not on it, so
  when such an element sits within :data:`_MASS_NUMBER_WINDOW_AMU` of a match
  the evidence note names it (208.98 amu is Bi's weight, and a Po isotope's
  mass lies there too; whether to refuse such a match is part of Sam's open
  decision 1).

The source is recorded in ``type_map_source``, the evidence for each mass match
and each label passed over goes into the notes, and two accepted sources that
disagree for a type are noted with the one used. A type no source names is
refused with a message asking for a map: no element is ever guessed.

NOTHING IS DROPPED SILENTLY
---------------------------
A frame that is truncated, holds another atom count, or has a header that does
not parse is listed in ``Trajectory.skipped`` under its position in the file,
with the reason, and every lost frame has a position of its own. In detail:

* **A file cut while it was being written.** LAMMPS, VASP, DL_POLY, ASE and
  OVITO end every line with a line ending, so a last line without one may be
  cut inside its last number ('2.496' read as '2.4'): the last frame is
  skipped, and a one-frame data file or CONFIG refused. A last row with fewer
  values than its columns is skipped as truncated.
* **A restart appended to a cut row.** ``dump_modify append`` after a crash
  writes ``ITEM: TIMESTEP`` on the cut row's line; the cut frame is skipped as
  cut and the restarted run's frames are read.
* **Stray rows.** A dump frame with more lines than NUMBER OF ATOMS has its
  values counted when the file is indexed (blank lines are not rows), so
  ``len()`` counts only frames that load.
* **Lost extended XYZ frames.** Blank lines between frames are passed over and
  noted (one note for the file, naming the first positions); a frame whose
  count disagrees with its rows is skipped, and the walk
  resyncs at the next count line followed by ``Lattice=``, giving each frame
  passed over its own position.
* **XDATCAR configuration lines.** Whether a cell header precedes a
  configuration is decided by parsing it, not by the gap alone; rows with no
  configuration line before them are a lost configuration each.
* **The atom count** is the one most frames hold, so one odd first frame does
  not put every other frame in ``skipped``.
* **Repeated or decreasing timesteps** (a restart appended to the same file)
  are kept in file order, and a note names where the order breaks.

A row whose values do not parse is found when its frame is loaded, and
``frame(k)`` then raises ``FrameError`` naming the frame; frame 0, parsed when
the file is opened, raises it naming the file and the frame. VASP's
fixed-width numbers that touch ('-0.63265286-0.11227753') are split at the
sign, with a frame note. The first note is the load record: the resolved path,
format, atoms, frames, frame 0's box and composition.

A LAMMPS dump without an id column is read frame by frame: LAMMPS re-orders
rows between frames (``atom_modify sort``, on by default), so the rows of each
frame are put in element order, ``ids_track_atoms`` is False, and per-atom
quantities across frames are not available from it.

HOW A FILE IS READ
------------------
The file is indexed once: one pass finds the byte offset of every frame marker
(``ITEM:``, ``configuration=``, ``timestep``), with ``bytes.find`` over 16 MB
blocks rather than a Python loop over lines, and counts the lines between them.
Frames are then read on demand from their byte range, so 100 frames of 10 000
atoms are never held at once. Gzip files are recognised by their first two
bytes and decompressed as a stream; a frame is reached by seeking in the
decompressed stream, which is cheap forwards and restarts from the beginning
backwards, so iterating in order costs one decompression. A gzip trajectory's
one stream is shared, so reaching a frame holds a lock (a worker thread and
the GUI thread may both load frames). A damaged gzip stream (zlib's error, a
failed CRC or length check, bytes after the stream) is refused with
``UnsupportedFormat`` naming the file. A gzip stream that ends without its
end-of-stream marker (a copy taken while the run was writing it) is read up to
the cut, and every trajectory format says so in a note: frames after the cut
are not in the file, so they cannot be counted. Everything is read as bytes,
so CRLF line endings parse like LF; a leading UTF-8 byte-order mark is
ignored. Lines that end with CR alone are refused with that reason by the
readers that index a file by byte offset (dump, extended XYZ, XDATCAR,
HISTORY); a data file or CONFIG is split into lines whole, and reads with
them.

TIMINGS
-------
Measured on this machine (Windows 11, i5-13420H, Python 3.11, numpy 2.4.6, run
pinned to the P-cores while other sessions shared the machine, total load
about 17 %), on a synthetic LAMMPS dump of 10 000 atoms x 20 frames written to
a temporary directory (``id type xu yu zu ix iy iz vx vy vz q``, rows shuffled,
a sheared box changing every frame; 17.7 MB). Medians of 5 repeats, two runs:

* opening the trajectory (the index pass, all 20 headers, and frame 0 parsed
  to check the type map): 0.14-0.16 s, of which the index pass is
  0.05-0.07 s;
* loading one frame (parse, sort by id, wrap, unwrap, validate): 39-52 ms;
* all 20 frames in order: 0.93-0.96 s;
* the same file gzip-compressed (7.7 MB): open 0.40-0.51 s; frame 7 as the
  first frame asked for, 128-137 ms (the stream is decompressed up to it);
  all 20 frames in order 1.06-1.08 s.

Every format, after the fixes of the readers' verification (same machine, not
pinned, other sessions running; 10 000 atoms x 100 frames each, three
repeats; open = index pass + load record, then every frame in order):

==============================  =========  =================  ===========
file                            size       open               per frame
==============================  =========  =================  ===========
LAMMPS dump, plain              36.3 MB    0.17-0.33 s        18-36 ms
LAMMPS dump, gzip               16.8 MB    1.33-1.77 s        40-47 ms
extended XYZ                    32.1 MB    0.72-0.79 s        43-49 ms
VASP XDATCAR                    37.0 MB    0.20-0.28 s        32-37 ms
DL_POLY HISTORY (velocities)    89.8 MB    0.70-0.77 s        78-84 ms
==============================  =========  =================  ===========

Reading is about 0.02-0.08 s per 10 000-atom frame; Step 0 measured the
vectorised bond-valence floor at about 1.0 s per 9 261-atom frame on the same
machine.

REFERENCES
----------
[1] LAMMPS documentation, "read_data command",
    https://docs.lammps.org/read_data.html (the atom-style column table, read
    for the style inference)
[2] LAMMPS source, src/write_data.cpp (WriteData::header, the first line),
    release stable_22Jul2025,
    https://github.com/lammps/lammps/blob/stable_22Jul2025/src/write_data.cpp
[3] LAMMPS documentation, "dump command", https://docs.lammps.org/dump.html
[4] LAMMPS source, src/dump_custom.cpp (header_item, header_item_triclinic,
    header_item_triclinic_general, pack_xs_triclinic, pack_xu_triclinic,
    pack_xsu), release stable_22Jul2025,
    https://github.com/lammps/lammps/blob/stable_22Jul2025/src/dump_custom.cpp;
    the conventions read from it were checked against the output and internal
    state of LAMMPS 22 Jul 2025 in the readers' verification
[5] The extended XYZ specification, libAtoms,
    https://github.com/libAtoms/extxyz
[6] VASP wiki, "POSCAR", https://www.vasp.at/wiki/index.php/POSCAR, and
    "XDATCAR", https://www.vasp.at/wiki/index.php/XDATCAR (the wiki does not
    describe the repeated header of a variable cell; that layout is read from
    VASP output, pymatgen's test files XDATCAR_6 and XDATCAR_monatomic)
[7] I. T. Todorov and W. Smith, *The DL_POLY 4 User Manual*, version 4.03.4,
    STFC Daresbury Laboratory (June 2012): §5.1.2 "The CONFIG File", §5.2.1
    "The HISTORY File", Appendix A "DL_POLY 4 Periodic Boundary Conditions".
    The manual names the DL_POLY website, http://www.ccp5.ac.uk/DL_POLY/
[8] LAMMPS documentation, "Triclinic (non-orthogonal) simulation boxes",
    https://docs.lammps.org/Howto_triclinic.html
[9] LAMMPS documentation, "units command", https://docs.lammps.org/units.html
[10] LAMMPS documentation, "Adiabatic core/shell model",
    https://docs.lammps.org/Howto_coreshell.html (the CS-Info section, read
    with ``read_data ... fix csinfo NULL CS-Info``)
"""
from __future__ import annotations

import functools
import gzip
import math
import re
import threading
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import md_model
from .md_model import (NO_TIMESTEP, Frame, FrameError, MemoryTrajectory,
                       Trajectory, frame_from_arrays, unwrap_with_images,
                       validate_symbol)
from .readers import UnsupportedFormat

__all__ = [
    "MASS_TOL_AMU", "MD_FORMATS", "LAMMPS_ANGSTROM_UNITS", "SNIFF_BYTES",
    "read_trajectory", "read_lammps_data", "read_lammps_dump", "read_extxyz",
    "read_xdatcar", "read_dlpoly", "read_dlpoly_config", "read_dlpoly_history",
    "sniff_md", "is_multiframe_xyz", "xyz_has_lattice", "box_from_lammps",
    "lammps_from_box", "box_from_dump_bounds", "dump_bounds_from_box",
    "type_map_from_masses", "element_weights_amu",
]

# ---------------------------------------------------------------------------
# constants: each one is a choice or a definition, and says which
# ---------------------------------------------------------------------------

# How far a LAMMPS type's mass may sit from an element's standard atomic weight
# for the type to be read as that element. A parsing tolerance, not a physical
# constant: the value proposed in Step 0 and not yet set by Sam. It accepts Bi
# given at 208.9804 (gemmi 208.98) while Po (209.0) lies 0.0196 amu away, which
# a rule refusing a second element within twice the tolerance would not.
# Measured against gemmi 0.7.1's weights: older force-field masses Mo 95.94 and
# Zn 65.39 are accepted (0.01 amu from gemmi's 95.95 and 65.38), Zn 65.409 is
# refused (0.029 amu); gemmi gives Ge 72.64 and Se 78.96, while ASE 3.29 writes
# 72.630 and 78.971 into its lammps-data files, so Ge from such a file is
# accepted (0.01 amu) and Se refused (0.011 amu).
MASS_TOL_AMU: float = 0.01

# What a mass comparison allows beyond the tolerance, so that a mass written
# with the tolerance's own decimals is compared as its decimals say: 72.64 -
# 72.63 is 0.010000000000005 in binary floating point. A parsing choice, far
# below the last digit any force field writes.
_MASS_EPS_AMU = 1e-9

# gemmi lists elements without a standard atomic weight at a whole number (Po
# at 209, Tc at 98); an isotope's mass sits near that whole number, not on it,
# so a mass within this window of one is not compared at MASS_TOL_AMU and the
# evidence note names the element. A choice, half the spacing of whole numbers.
_MASS_NUMBER_WINDOW_AMU = 0.5

# An element symbol as a file writes one: an upper-case letter, optionally a
# lower-case one, optionally a charge. validate_symbol accepts any case, which
# suits a symbol a user types; a label in a file is read as an element only in
# this form, because force fields write names such as HO (a hydroxyl hydrogen),
# CA or ho, which validate_symbol would turn into Ho, Ca and Ho.
_FILE_SYMBOL = re.compile(r"[A-Z][a-z]?(?:[0-9]?[+-]|[+-][0-9]?)?")

# The names read_trajectory and sniff_md use for the formats.
MD_FORMATS = ("lammps-data", "lammps-dump", "extxyz", "vasp-xdatcar",
              "dlpoly-config", "dlpoly-history")

# LAMMPS unit styles whose lengths are Å [9], and the factors that take their
# times to ps and velocities to Å/ps. Definitions of the units, not choices:
# metal is ps and Å/ps; real is fs and Å/fs.
LAMMPS_ANGSTROM_UNITS = ("metal", "real")
_TIME_TO_PS = {"metal": 1.0, "real": 1.0e-3}
_VELOCITY_TO_ANG_PER_PS = {"metal": 1.0, "real": 1.0e3}
_OTHER_LENGTH_UNITS = {"lj": "reduced units (sigma)", "si": "metres",
                       "cgs": "centimetres", "electron": "Bohr",
                       "micro": "micrometres", "nano": "nanometres"}

# How much of a file sniff_md looks at. A choice: every header it recognises
# fits in it (the first configuration line of an XDATCAR is on line 8).
SNIFF_BYTES = 4096

# Block size of the index pass, and how much is kept on each side of a block
# edge so that a marker's neighbouring lines are never cut. Performance
# choices; neither changes what is read.
_CHUNK_BYTES = 1 << 24
_MARGIN_BYTES = 1 << 16

_GZIP_MAGIC = b"\x1f\x8b"
_BOM = b"\xef\xbb\xbf"
_WHITESPACE = frozenset(b" \t\r\n\f\v")
_SOURCE_ORDER = ("user", "element column", "type label", "data-file masses",
                 "file symbols")
# What zlib and gzip raise for a stream whose bytes are damaged (a flipped bit,
# a failed CRC or length check, bytes after the stream that are not gzip).
_GZIP_DAMAGE = (zlib.error, gzip.BadGzipFile)
# The note for a gzip stream with no end-of-stream marker. A cut at a frame
# boundary leaves every indexed frame complete, so without this note the file
# would read as whole.
_GZIP_CUT_NOTE = ("the gzip stream ends before its end-of-stream marker (a copy "
                  "taken while it was being written); what was decompressed "
                  "before the cut was indexed, and frames after the cut are not "
                  "in the file, so they are not counted")


# ---------------------------------------------------------------------------
# file access: plain or gzip, always bytes
# ---------------------------------------------------------------------------

def _existing(path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such file")
    return path


def _is_gzip(path: Path) -> bool:
    with open(path, "rb") as handle:
        return handle.read(2) == _GZIP_MAGIC


def _open_binary(path: Path):
    """The file as a binary stream, decompressed when it starts with gzip's
    magic bytes (whatever its name)."""
    if _is_gzip(path):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _damaged(handle, error: Exception) -> UnsupportedFormat:
    """The refusal for a gzip stream that cannot be decompressed, naming the
    file and what zlib or gzip reported."""
    name = Path(str(getattr(handle, "name", "") or "the file")).name
    source = "zlib" if isinstance(error, zlib.error) else "gzip"
    return UnsupportedFormat(
        f"{name}: the gzip stream is damaged and cannot be decompressed "
        f"({source}: {error}); nothing in it is read")


def _strip_bom(data: bytes) -> bytes:
    """The bytes without a leading UTF-8 byte-order mark (Windows editors and
    PowerShell's Out-File write one), which no MD writer emits."""
    return data[len(_BOM):] if data.startswith(_BOM) else data


def _refuse_cr_only(name: str, head: bytes) -> None:
    """ValueError for text whose lines end with CR alone: the index pass and
    readline split lines at LF, so such a file would read as one line."""
    if b"\r" in head and b"\n" not in head:
        raise ValueError(
            f"{name}: its lines end with CR alone (classic Mac OS line endings); "
            "FACET reads LF and CRLF line endings, so convert the file first")


def _read_some(handle, size: int) -> tuple[bytes, bool, bool]:
    """Up to ``size`` bytes: (data, at_end, cut).

    ``cut`` is True when a gzip stream ended without its end-of-stream marker
    (a file copied while it was being written, or truncated). ``read1`` returns
    what each underlying read produced, so nothing decompressed before the cut
    is lost when gzip raises EOFError. A damaged stream raises
    UnsupportedFormat naming the file, never zlib.error.
    """
    pieces: list[bytes] = []
    got = 0
    while got < size:
        try:
            piece = handle.read1(size - got)
        except EOFError:
            return b"".join(pieces), True, True
        except _GZIP_DAMAGE as error:
            raise _damaged(handle, error) from None
        if not piece:
            return b"".join(pieces), True, False
        pieces.append(piece)
        got += len(piece)
    return b"".join(pieces), False, False


def _readline(handle) -> bytes:
    """``handle.readline()``, with a damaged gzip stream reported as
    UnsupportedFormat naming the file."""
    try:
        return handle.readline()
    except _GZIP_DAMAGE as error:
        raise _damaged(handle, error) from None


def _read_all(path: Path) -> bytes:
    pieces = []
    with _open_binary(path) as handle:
        while True:
            data, at_end, cut = _read_some(handle, _CHUNK_BYTES)
            pieces.append(data)
            if at_end:
                break
    data = b"".join(pieces)
    if cut:
        raise ValueError(f"{path.name}: the gzip stream ends before its end "
                         "marker; the file is incomplete")
    return data


def _head(path: Path, size: int = SNIFF_BYTES) -> bytes:
    """The first ``size`` bytes (decompressed), without a byte-order mark."""
    with _open_binary(path) as handle:
        data, _, _ = _read_some(handle, size)
    return _strip_bom(data)


def _text(raw: bytes) -> str:
    return raw.decode("utf-8", "replace").rstrip("\r\n")


def _lines(raw: bytes) -> list[str]:
    """Lines of a byte string: LF, CRLF and CR all end a line."""
    return [line.decode("utf-8", "replace") for line in raw.splitlines()]


# ---------------------------------------------------------------------------
# the index pass
# ---------------------------------------------------------------------------

@dataclass
class _Mark:
    """One marker line: where it is and the lines around it."""

    offset: int             # byte offset of the line's first byte
    end: int                # byte offset just past its line ending
    line: int               # 0-based line number
    text: str
    before: list[str]       # up to before_lines lines preceding it
    after: list[str]        # up to after_lines lines following it
    glued: bool = False     # the needle starts after other text on its line;
                            # offset and text then start at the needle


@dataclass
class _Scan:
    marks: list[_Mark]
    size: int               # bytes (decompressed)
    n_lines: int
    last_text_line: int     # 0-based index of the last line holding text; -1 if none
    last_line: str          # that line
    gzip_cut: bool
    # Whether a line ending follows the last line holding text. LAMMPS, VASP,
    # DL_POLY, ASE and OVITO end every line with one, so a file whose last
    # text has none was cut, possibly inside its last number.
    last_line_ended: bool = True


def _lines_before(buf: bytes, start: int, count: int) -> list[str]:
    first = start
    for _ in range(count):
        if first <= 0:
            break
        first = buf.rfind(b"\n", 0, first - 1) + 1
    return _lines(buf[first:start])


def _lines_after(buf: bytes, end: int, count: int) -> list[str]:
    stop = end
    for _ in range(count):
        newline = buf.find(b"\n", stop)
        if newline < 0:
            stop = len(buf)
            break
        stop = newline + 1
    return _lines(buf[end:stop])


def _scan(path: Path, needle: bytes, *, line_start: bool,
          before_lines: int = 0, after_lines: int = 0,
          chunk_bytes: int = _CHUNK_BYTES, glued: bool = False) -> _Scan:
    """Every line holding ``needle`` (at its start, after blanks, when
    ``line_start``), with its byte offset and line number, in one pass.

    With ``glued``, a needle found after other text on its line is kept too,
    marked ``glued`` and starting at the needle: a LAMMPS run restarted with
    ``dump_modify append`` onto a file that a crash cut mid-row writes its
    first ``ITEM: TIMESTEP`` on the cut row's line.

    The file is read in blocks; each block is searched only up to its last
    line start that leaves ``_MARGIN_BYTES`` after it, and the rest is carried
    into the next block together with ``_MARGIN_BYTES`` before it, so the lines
    around a marker are always in the buffer. A leading UTF-8 byte-order mark
    counts as blank; lines that end with CR alone are refused.
    """
    marks: list[_Mark] = []
    buf = b""
    base = 0            # file offset of buf[0]
    done = 0            # buf[:done] has been searched; always a line start
    line_at_done = 0
    last_text_line = -1
    ended = True
    cut = False
    first_block = True
    with _open_binary(path) as handle:
        while True:
            data, at_end, cut_now = _read_some(handle, chunk_bytes)
            cut = cut or cut_now
            if first_block:
                _refuse_cr_only(path.name, data)
                first_block = False
            if data:
                buf = buf + data
            if at_end:
                limit = len(buf)
            else:
                edge = len(buf) - _MARGIN_BYTES
                limit = buf.rfind(b"\n", done, edge) + 1 if edge > done else 0
                if limit <= done:
                    continue
            counted, line = done, line_at_done
            position = done
            while True:
                hit = buf.find(needle, position, limit)
                if hit < 0:
                    break
                start = buf.rfind(b"\n", 0, hit) + 1
                newline = buf.find(b"\n", hit)
                end = len(buf) if newline < 0 else newline + 1
                position = end
                prefix = buf[start:hit]
                if base + start == 0:
                    prefix = _strip_bom(prefix)
                stuck = bool(prefix.strip())
                if line_start and stuck and not glued:
                    continue
                line += buf.count(b"\n", counted, start)
                counted = start
                first = hit if stuck and line_start else start
                marks.append(_Mark(
                    offset=base + first, end=base + end, line=line,
                    text=_text(buf[first:end]),
                    before=_lines_before(buf, start, before_lines),
                    after=_lines_after(buf, end, after_lines),
                    glued=stuck and line_start))
            line += buf.count(b"\n", counted, limit)
            tail = limit
            while tail > done and buf[tail - 1] in _WHITESPACE:
                tail -= 1
            if tail > done:
                last_text_line = line_at_done + buf.count(b"\n", done, tail - 1)
                after_text = buf[tail:limit]
                ended = b"\n" in after_text or b"\r" in after_text
            if at_end:
                size = base + len(buf)
                n_lines = line + (1 if buf and not buf.endswith(b"\n") else 0)
                stripped = buf.rstrip()
                last_line = _text(stripped[stripped.rfind(b"\n") + 1:]) \
                    if stripped else ""
                return _Scan(marks, size, n_lines, last_text_line, last_line, cut,
                             ended)
            keep = max(0, limit - _MARGIN_BYTES)
            buf = buf[keep:]
            base += keep
            done = limit - keep
            line_at_done = line


def _rows_between(marks: Sequence[_Mark], index: int, scan: _Scan) -> int:
    """Lines between marker ``index`` and the next one (or the last line of
    the file holding text)."""
    if index + 1 < len(marks):
        return marks[index + 1].line - marks[index].line - 1
    return max(0, scan.last_text_line - marks[index].line)


def _range_end(marks: Sequence[_Mark], index: int, scan: _Scan) -> int:
    return marks[index + 1].offset if index + 1 < len(marks) else scan.size


# ---------------------------------------------------------------------------
# LAMMPS boxes
# ---------------------------------------------------------------------------

def _finite(values, name: str) -> list[float]:
    """Floats from text or numbers; ValueError naming ``name`` for a value that
    is not a number or not finite."""
    out = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} holds {value!r}, which is not a "
                             "number") from None
        if not math.isfinite(number):
            raise ValueError(f"{name} holds a value that is not finite "
                             f"({value!r})")
        out.append(number)
    return out


def box_from_lammps(xlo_ang: float, xhi_ang: float, ylo_ang: float,
                    yhi_ang: float, zlo_ang: float, zhi_ang: float,
                    xy_ang: float = 0.0, xz_ang: float = 0.0,
                    yz_ang: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``(box_ang, origin_ang)`` of a LAMMPS restricted triclinic box [8].

    Rows a = (xhi - xlo, 0, 0), b = (xy, yhi - ylo, 0), c = (xz, yz,
    zhi - zlo); origin (xlo, ylo, zlo). ValueError when a hi bound is not above
    its lo bound or a value is not finite.
    """
    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = _finite(
        (xlo_ang, xhi_ang, ylo_ang, yhi_ang, zlo_ang, zhi_ang, xy_ang, xz_ang,
         yz_ang), "the LAMMPS box")
    for axis, lo, hi in (("x", xlo, xhi), ("y", ylo, yhi), ("z", zlo, zhi)):
        if not hi > lo:
            raise ValueError(f"the box has {axis}hi {hi:g} not above {axis}lo "
                             f"{lo:g}, so it spans no length along {axis}")
    box = np.array([[xhi - xlo, 0.0, 0.0],
                    [xy, yhi - ylo, 0.0],
                    [xz, yz, zhi - zlo]])
    return box, np.array([xlo, ylo, zlo])


def lammps_from_box(box_ang, origin_ang) -> dict[str, float]:
    """The inverse of :func:`box_from_lammps`: xlo .. zhi, xy, xz, yz.

    ValueError for a box that is not in LAMMPS's restricted form (a along +x,
    b in the xy plane with a positive y component, c with a positive z
    component), which needs a rotation first.
    """
    box = np.asarray(box_ang, dtype=np.float64)
    origin = np.asarray(origin_ang, dtype=np.float64)
    if box.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("box_ang (3, 3) and origin_ang (3,) are needed")
    if box[0, 1] != 0.0 or box[0, 2] != 0.0 or box[1, 2] != 0.0 \
            or not (box[0, 0] > 0 and box[1, 1] > 0 and box[2, 2] > 0):
        raise ValueError(
            "the box is not in LAMMPS's restricted triclinic form (a along +x, "
            "b in the xy plane with by > 0, cz > 0); it needs a rotation first")
    xlo, ylo, zlo = (float(v) for v in origin)
    return {"xlo": xlo, "xhi": xlo + float(box[0, 0]),
            "ylo": ylo, "yhi": ylo + float(box[1, 1]),
            "zlo": zlo, "zhi": zlo + float(box[2, 2]),
            "xy": float(box[1, 0]), "xz": float(box[2, 0]),
            "yz": float(box[2, 1])}


def box_from_dump_bounds(bounds_ang, tilt_ang=None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """``(box_ang, origin_ang)`` from a dump's BOX BOUNDS block.

    ``bounds_ang`` is (3, 2): (xlo_bound, xhi_bound), (ylo_bound, yhi_bound),
    (zlo_bound, zhi_bound). ``tilt_ang`` is (xy, xz, yz), the third values of
    the three lines, or None for an orthogonal box. Inverts [8]::

        xlo = xlo_bound - MIN(0, xy, xz, xy+xz)   xhi = xhi_bound - MAX(0, xy, xz, xy+xz)
        ylo = ylo_bound - MIN(0, yz)              yhi = yhi_bound - MAX(0, yz)
    """
    bounds = np.asarray(bounds_ang, dtype=np.float64)
    if bounds.shape != (3, 2):
        raise ValueError("the bounds need three (lo, hi) pairs")
    xy, xz, yz = (0.0, 0.0, 0.0) if tilt_ang is None else _finite(
        tilt_ang, "the tilt factors")
    (xlo_b, xhi_b), (ylo_b, yhi_b), (zlo_b, zhi_b) = bounds
    return box_from_lammps(
        xlo_b - min(0.0, xy, xz, xy + xz), xhi_b - max(0.0, xy, xz, xy + xz),
        ylo_b - min(0.0, yz), yhi_b - max(0.0, yz), zlo_b, zhi_b, xy, xz, yz)


def dump_bounds_from_box(box_ang, origin_ang
                         ) -> tuple[np.ndarray, tuple[float, float, float]]:
    """The forward bound formulas [8]: ((3, 2) bounds, (xy, xz, yz))."""
    p = lammps_from_box(box_ang, origin_ang)
    xy, xz, yz = p["xy"], p["xz"], p["yz"]
    bounds = np.array([
        [p["xlo"] + min(0.0, xy, xz, xy + xz), p["xhi"] + max(0.0, xy, xz, xy + xz)],
        [p["ylo"] + min(0.0, yz), p["yhi"] + max(0.0, yz)],
        [p["zlo"], p["zhi"]]])
    return bounds, (xy, xz, yz)


# ---------------------------------------------------------------------------
# types to elements
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def element_weights_amu() -> tuple[tuple[str, float], ...]:
    """(symbol, standard atomic weight in amu) for every element gemmi knows.

    The weights ``elements.info(symbol).weight`` returns, read from gemmi
    directly, because ``elements.info`` normalises its argument first and
    ``info('He')`` returns hydrogen's data (md_model's
    ``symbols_changed_by_normalise`` lists the 15 symbols affected). A test
    pins the two to each other for every symbol normalise leaves unchanged.
    """
    import gemmi

    out = []
    for symbol in sorted(md_model._canonical_symbols()):
        weight = float(gemmi.Element(symbol).weight)
        if weight > 0:
            out.append((symbol, weight))
    return tuple(out)


def _checked_tol(tol_amu) -> float:
    if isinstance(tol_amu, bool):
        raise ValueError("the mass tolerance needs a number of amu")
    tol = float(tol_amu)
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError(f"the mass tolerance {tol_amu!r} amu needs to be a "
                         "positive number")
    return tol


def _amu_gap(gap: float, tol: float) -> str:
    """A mass difference as text that cannot read as equal to the tolerance
    when it lies outside it ('0.01' printed for 0.0100004 next to 'tolerance
    0.01' would)."""
    text = f"{gap:.3g}"
    if gap > tol + _MASS_EPS_AMU and float(text) <= tol:
        return f"more than {tol:g}"
    return text


def _whole_number_neighbours(value: float, symbol: str) -> list[tuple[str, float]]:
    """Elements other than ``symbol`` that gemmi lists at a whole number within
    :data:`_MASS_NUMBER_WINDOW_AMU` of ``value``."""
    return [(s, w) for s, w in element_weights_amu()
            if s != symbol and w == round(w)
            and abs(w - value) <= _MASS_NUMBER_WINDOW_AMU]


def _match_mass(mass, tol: float) -> tuple[str | None, str]:
    """(element, evidence) for one mass, or (None, why not).

    An element is accepted when it alone lies within ``tol`` (plus
    :data:`_MASS_EPS_AMU`); two within it are refused as ambiguous.
    """
    try:
        value = float(mass)
    except (TypeError, ValueError):
        return None, f"its mass {mass!r} is not a number"
    if not math.isfinite(value) or value <= 0:
        return None, f"its mass {mass!r} is not a positive number"
    if value == round(value):
        return None, (f"its mass {value:g} amu is a whole number, which names "
                      f"no element to within {tol:g} amu (force fields often "
                      "round masses to integers)")
    weights = element_weights_amu()
    distance = sorted((abs(w - value), s, w) for s, w in weights)
    within = [(s, w) for d, s, w in distance if d <= tol + _MASS_EPS_AMU]
    if not within:
        d, s, w = distance[0]
        return None, (f"no standard atomic weight lies within {tol:g} amu of its "
                      f"mass {value:.6g} amu (nearest: {s}, {w:.6g} amu, "
                      f"{_amu_gap(d, tol)} amu away)")
    if len(within) > 1:
        return None, (f"its mass {value:.6g} amu lies within {tol:g} amu of "
                      + " and of ".join(f"{s} ({w:.6g} amu)" for s, w in within))
    symbol, weight = within[0]
    d2, s2, _ = distance[1]
    evidence = (f"mass {value:.6g} amu read as {symbol} (standard atomic "
                f"weight {weight:.6g} amu, {abs(weight - value):.2g} amu "
                f"away; tolerance {tol:g} amu; the next closest element, "
                f"{s2}, is {d2:.3g} amu away)")
    for other, whole in _whole_number_neighbours(value, symbol):
        evidence += (f"; gemmi lists {other} at the whole number {whole:g} amu "
                     f"(no standard atomic weight), so this match does not rule "
                     f"{other} out: an isotope's mass lies near its whole mass "
                     "number, not on it")
    return symbol, evidence


def _label_problem(token: str) -> str | None:
    """Why a label in a file is not read as an element, or None when it is
    one, written as files write symbols (see :data:`_FILE_SYMBOL`)."""
    text = str(token).strip()
    try:
        symbol = validate_symbol(text)
    except ValueError:
        return "not an element symbol"
    if _FILE_SYMBOL.fullmatch(text) is None:
        return (f"matches the symbol {symbol} only in another capitalisation; a "
                "force-field name such as HO or ca is not read as an element")
    return None


def _file_symbol(token: str) -> str:
    """The element a label in a file names, or ValueError saying why not."""
    problem = _label_problem(token)
    if problem is not None:
        raise ValueError(f"{str(token)!r}: {problem}; it needs a type map to an "
                         "element")
    return validate_symbol(str(token).strip())


def _mass_conflict(symbol: str, mass, tol: float, *, where: str = "the file"
                   ) -> str | None:
    """Why the file's own mass contradicts reading a label as ``symbol``, or
    None.

    A mass is consistent with the element when it lies within ``tol`` of its
    standard atomic weight, when it is that weight rounded to a whole number
    (force fields write O as 16), or when no other element's weight lies
    closer (older tables, a heavier isotope). It contradicts the label when
    another element's weight lies closer: 'HO' at 1.008 amu is hydrogen, not
    holmium.
    """
    try:
        value = float(mass)
    except (TypeError, ValueError):
        return f"{where} gives its mass as {mass!r}, which is not a number"
    if not math.isfinite(value) or value <= 0:
        return f"{where} gives its mass as {mass!r}, which is not a positive number"
    weights = dict(element_weights_amu())
    weight = weights.get(symbol)
    if weight is None or abs(weight - value) <= tol + _MASS_EPS_AMU:
        return None
    if value == round(value) and round(weight) == value:
        return None
    nearest, near_weight = min(weights.items(), key=lambda sw: abs(sw[1] - value))
    if nearest == symbol or abs(near_weight - value) >= abs(weight - value):
        return None
    return (f"{where} gives its mass as {value:.6g} amu, nearer to {nearest} "
            f"({near_weight:.6g} amu) than to {symbol} ({weight:.6g} amu)")


def type_map_from_masses(masses_amu: Mapping[int | str, float],
                         tol_amu: float = MASS_TOL_AMU
                         ) -> tuple[dict[int | str, str], list[str]]:
    """Each type's mass matched to gemmi's standard atomic weights.

    A type is read as an element only when exactly one element lies within
    ``tol_amu``; zero or several, a whole-number mass, or a mass that is not a
    positive number raise ValueError naming every such type. Returns the map
    and one evidence sentence per type (its mass, the element, how far away,
    and the next closest element).
    """
    tol = _checked_tol(tol_amu)
    out: dict[int | str, str] = {}
    evidence: list[str] = []
    problems: list[str] = []
    for key in sorted(masses_amu, key=lambda k: (isinstance(k, str), str(k))):
        symbol, detail = _match_mass(masses_amu[key], tol)
        if symbol is None:
            problems.append(f"type {key}: {detail}")
        else:
            out[key] = symbol
            evidence.append(f"type {key}: {detail}")
    if problems:
        raise ValueError("; ".join(problems)
                         + "; give a type map (type -> element) for these")
    return out, evidence


def _normalise_type_map(type_map) -> tuple[dict[int, str], dict[str, str]]:
    """A user's map split into numeric types and text labels, values validated."""
    numeric: dict[int, str] = {}
    labels: dict[str, str] = {}
    if type_map is None:
        return numeric, labels
    if not isinstance(type_map, Mapping):
        raise ValueError("type_map needs a mapping such as {1: 'Si', 2: 'O'}")
    for key, value in type_map.items():
        if not isinstance(value, str):
            raise ValueError(f"type map entry {key!r}: {value!r} is not an "
                             "element symbol")
        symbol = validate_symbol(value)
        if isinstance(key, (bool, np.bool_)):
            raise ValueError(f"type map key {key!r} is neither a type number "
                             "nor a label")
        if isinstance(key, (int, np.integer)):
            target, name = numeric, int(key)
        elif isinstance(key, str) and re.fullmatch(r"\s*[+-]?\d+\s*", key):
            target, name = numeric, int(key)
        elif isinstance(key, str) and key.strip():
            target, name = labels, key.strip()
        else:
            raise ValueError(f"type map key {key!r} is neither a type number "
                             "nor a label")
        if name in target and target[name] != symbol:
            raise ValueError(f"the type map gives {name!r} two elements "
                             f"({target[name]} and {symbol})")
        target[name] = symbol
    return numeric, labels


def _join_sources(used: Sequence[str]) -> str:
    present = [s for s in _SOURCE_ORDER if s in set(used)]
    return " + ".join(present) if present else "user"


def _type_options(type_id: int, *, user: Mapping[int, str],
                  candidates: Sequence[tuple[str, Mapping[int, str]]],
                  masses: Mapping[int, float] | None, tol: float
                  ) -> tuple[tuple[str, str, str] | None, list[str], str | None]:
    """(the source used, notes, why no element) for one numeric type."""
    options: list[tuple[str, str | None, str]] = []
    if type_id in user:
        options.append(("user", user[type_id],
                        f"the type map gives {user[type_id]}"))
    for source, labels in candidates:
        if type_id in labels:
            label = labels[type_id]
            problem = _label_problem(label)
            if problem is None:
                symbol = validate_symbol(label.strip())
                options.append((source, symbol,
                                f"its {source} {label!r} reads as {symbol}"))
            else:
                options.append((source, None,
                                f"its {source} {label!r}: {problem}"))
    if masses is not None and type_id in masses:
        symbol, detail = _match_mass(masses[type_id], tol)
        options.append(("data-file masses", symbol,
                        detail if symbol is None else f"its data-file {detail}"))
    pick = next((o for o in options if o[1] is not None), None)
    if pick is None:
        return None, [], "; ".join(o[2] for o in options) or "no source names it"
    notes: list[str] = []
    if pick[0] not in ("user", "data-file masses") and masses is not None \
            and type_id in masses:
        conflict = _mass_conflict(pick[1], masses[type_id], tol,
                                  where="the data file")
        if conflict is not None:
            return None, [], f"{pick[2]}, and {conflict}"
    if pick[0] == "data-file masses":
        notes.append(f"type {type_id}: {pick[2]}")
    passed_over = [o[2] for o in options
                   if o[1] is None and o[0] != "data-file masses"]
    if passed_over:
        notes.append(f"type {type_id}: {pick[1]} from the {pick[0]}; "
                     + "; ".join(passed_over))
    differing = [o for o in options if o[1] is not None and o[1] != pick[1]]
    if differing:
        notes.append(f"type {type_id}: {pick[1]} from the {pick[0]} is used; "
                     + "; ".join(o[2] for o in differing))
    return pick, notes, None


def _resolve_types(present: Sequence[int], *, user: Mapping[int, str],
                   candidates: Sequence[tuple[str, Mapping[int, str]]],
                   masses: Mapping[int, float] | None, tol: float,
                   how_to: str, later: Sequence[int] = (),
                   absent: str = "no atom has those types"
                   ) -> tuple[dict[int, str], str, list[str]]:
    """Numeric type -> element, each type from the first source that names it.

    Sources in order: the user's map, each ``candidates`` entry (labels that
    are element symbols as files write them, see :data:`_FILE_SYMBOL`), then
    masses. A label whose element the same file's mass for that type
    contradicts (:func:`_mass_conflict`) is refused, not used. Disagreements
    between sources that are both accepted are noted with the element used; a
    type no source names is refused.

    ``later`` are types not in ``present`` (a dump's frame 0) that a source
    names anyway (the user's map, ``masses_from``): each one that resolves is
    kept for later frames, and the notes say so; one that does not is noted,
    and a frame holding it is refused when loaded. ``absent`` ends the note
    for user entries no type uses.
    """
    chosen: dict[int, str] = {}
    used: list[str] = []
    notes: list[str] = []
    problems: list[str] = []
    for type_id in sorted(present):
        pick, type_notes, problem = _type_options(
            type_id, user=user, candidates=candidates, masses=masses, tol=tol)
        if pick is None:
            problems.append(f"type {type_id} ({problem})")
            continue
        chosen[type_id] = pick[1]
        used.append(pick[0])
        notes.extend(type_notes)
    if problems:
        raise ValueError("no element for " + ", ".join(problems) + ". "
                         + how_to)
    kept, missing = [], []
    for type_id in sorted(set(later) - set(present)):
        pick, type_notes, problem = _type_options(
            type_id, user=user, candidates=candidates, masses=masses, tol=tol)
        if pick is None:
            missing.append(f"type {type_id} ({problem})")
            continue
        chosen[type_id] = pick[1]
        used.append(pick[0])
        kept.append(type_id)
        notes.extend(type_notes)
    if kept:
        notes.append(f"type(s) {kept} do not occur in frame 0; their elements "
                     "are kept for later frames")
    if missing:
        notes.append("no element for " + ", ".join(missing) + "; these types do "
                     "not occur in frame 0, and a frame holding them is refused")
    unused = sorted(set(user) - set(chosen))
    if unused:
        notes.append(f"type map entries for type(s) {unused} are not used: "
                     + absent)
    return chosen, _join_sources(used), notes


def _resolve_labels(labels: Sequence[str], user: Mapping[str, str], *,
                    reduce: Callable[[str], tuple[str, str | None]] | None = None,
                    masses: Mapping[str, float] | None = None,
                    tol: float = MASS_TOL_AMU, what: str = "label"
                    ) -> tuple[dict[str, str], dict[str, str], str, list[str]]:
    """Text labels -> elements: the user's map, else the label as a symbol.

    ``reduce`` turns a label into the token to validate and an optional note
    (VASP's 'Na_pv'). A label is read as an element only when it is written as
    files write symbols (:data:`_FILE_SYMBOL`: 'HO', 'CA' and 'ho' are not).
    Returns (per-label element, the part worth listing as a type map, source,
    notes). Masses, when the file gives them, are compared with the element's
    standard atomic weight: a label whose mass lies nearer another element is
    refused (:func:`_mass_conflict`), and any other difference beyond ``tol``
    is noted. A user's map entry is used as given, its mass difference noted.
    """
    out: dict[str, str] = {}
    listed: dict[str, str] = {}
    used: list[str] = []
    notes: list[str] = []
    problems: list[tuple[str, str]] = []
    weights = dict(element_weights_amu())
    for label in sorted(set(labels)):
        mass = None if masses is None or label not in masses else masses[label]
        if label in user:
            out[label] = listed[label] = user[label]
            used.append("user")
        else:
            token, note = (label, None) if reduce is None else reduce(label)
            problem = _label_problem(token)
            if problem is None and mass is not None:
                symbol = validate_symbol(token.strip())
                conflict = _mass_conflict(symbol, mass, tol)
                if conflict is not None:
                    problem = f"reads as {symbol}, and {conflict}"
            if problem is not None:
                given = "" if mass is None else f" (mass {mass:.6g} amu in the file)"
                problems.append((label, f"{label!r}{given}: {problem}"))
                continue
            symbol = validate_symbol(token.strip())
            out[label] = symbol
            used.append("file symbols")
            if symbol != label:
                listed[label] = symbol
            if note:
                notes.append(note)
        if mass is not None:
            weight = weights.get(out[label])
            if weight is not None and abs(weight - mass) > tol + _MASS_EPS_AMU:
                notes.append(
                    f"{what} {label!r} is read as {out[label]} (standard atomic "
                    f"weight {weight:.6g} amu), and the file gives its mass as "
                    f"{mass:.6g} amu, {abs(weight - mass):.3g} amu away")
    if problems:
        example = ", ".join(f"{label!r}: '<element>'" for label, _ in problems[:3])
        raise ValueError(
            f"no element for the {what}(s) "
            + "; ".join(text for _, text in problems)
            + f". Give a type map from {what} to element, for example "
            f"type_map={{{example}}}")
    unused = sorted(set(user) - set(labels))
    if unused:
        notes.append(f"type map entries {unused} are not used: no atom carries "
                     f"those {what}s")
    return out, listed, _join_sources(used), notes


def _numeric_entries_note(user_types: Mapping[int, str], what: str) -> list[str]:
    """The note for type-number entries in a map for a file mapped by text."""
    if not user_types:
        return []
    return [f"type map entries {sorted(user_types)} are type numbers; this file "
            f"is mapped by its {what}, so they are not used"]


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def _checked_units(style: str | None, where: str) -> str | None:
    if style is None:
        return None
    if not isinstance(style, str):
        raise ValueError(f"units {style!r}: a LAMMPS unit style name is needed")
    name = style.strip().lower()
    if name in LAMMPS_ANGSTROM_UNITS:
        return name
    if name in _OTHER_LENGTH_UNITS:
        raise ValueError(
            f"{where} gives LAMMPS units {name}, in which lengths are "
            f"{_OTHER_LENGTH_UNITS[name]}, not Å; FACET reads units metal and "
            "real")
    raise ValueError(f"{where} gives units {style!r}, which is not a LAMMPS "
                     "unit style")


def _combine_units(from_file: str | None, from_user: str | None, file_where: str
                   ) -> tuple[str | None, str]:
    """The unit style to use, and where it came from."""
    if from_file and from_user and from_file != from_user:
        raise ValueError(f"{file_where} says units {from_file}, and the units "
                         f"argument says {from_user}")
    if from_file:
        return from_file, file_where
    if from_user:
        return from_user, "the units argument"
    return None, ""


def _units_note(style: str | None, where: str, *, has_time: bool,
                has_velocity: bool) -> str:
    if style is None:
        text = ("length unit assumed Å (LAMMPS units metal or real); the file "
                "does not say")
        if has_time or has_velocity:
            text += ("; times and velocities are stored as written, which is "
                     "ps and Å/ps under units metal; under units real LAMMPS "
                     "writes fs and Å/fs, so a time read here would be 1000 "
                     "times its value in ps and a velocity 1/1000 of its value "
                     "in Å/ps (files written by ASE or by hand state no style "
                     "either); units='real' or units='metal' states the style")
        return text
    text = f"LAMMPS units {style} ({where}): lengths in Å"
    if style == "real" and (has_time or has_velocity):
        text += "; times converted from fs to ps and velocities from Å/fs to Å/ps"
    return text


# ---------------------------------------------------------------------------
# trajectories read from a file
# ---------------------------------------------------------------------------

class _Closable:
    """``close()`` and the context-manager protocol, for every reader's result.

    ``ids_track_atoms`` is True when ``atom_id`` k is the same atom in every
    frame: the file's own ids, or a fixed atom order (XDATCAR, an extended
    XYZ without ids). It is False for a LAMMPS dump without an id column,
    whose rows LAMMPS re-orders between frames (``atom_modify sort``, on by
    default); there each frame is a valid structure, and per-atom quantities
    across frames (displacements, mean-square displacement) have no meaning.
    """

    ids_track_atoms: bool = True

    def close(self) -> None:
        """Release the open file, if any. Frames already loaded stay valid."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _FileTrajectory(_Closable, Trajectory):
    """Frames read on demand from byte ranges of one file (plain or gzip).

    ``periodic`` is (x, y, z) when the file states the boundary, else None.
    A plain file is opened for each frame; a gzip file keeps one stream open,
    because reaching a frame means decompressing up to it. That stream is
    shared by every caller, so seeking and reading it hold a lock: a worker
    thread loading frame k and the GUI thread loading frame 0 would otherwise
    interleave their seeks on it.
    """

    def __init__(self, path: Path, ranges: Sequence[tuple[int, int]], *,
                 periodic: tuple[bool, bool, bool] | None, **kwargs) -> None:
        super().__init__(source_path=path, **kwargs)
        self._path = Path(path)
        self._ranges = [(int(a), int(b)) for a, b in ranges]
        self._gzip = _is_gzip(self._path)
        self._handle = None
        self._lock = threading.Lock()
        self.periodic = periodic

    def _bytes(self, k: int) -> bytes:
        start, end = self._ranges[k]
        if not self._gzip:
            with open(self._path, "rb") as handle:
                handle.seek(start)
                data = handle.read(end - start)
            cut = False
        else:
            with self._lock:
                if self._handle is None:
                    self._handle = gzip.open(self._path, "rb")
                try:
                    self._handle.seek(start)
                except EOFError:
                    data, cut = b"", True
                except _GZIP_DAMAGE as error:
                    raise _damaged(self._handle, error) from None
                else:
                    data, _, cut = _read_some(self._handle, end - start)
        if len(data) != end - start:
            raise ValueError(
                f"the file holds {len(data)} of the {end - start} bytes indexed "
                "for this frame" + (" (the gzip stream ends early)" if cut else
                                    "; it changed after it was indexed"))
        return data

    def close(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            handle.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __del__(self):
        try:
            self.close()
        except Exception:       # noqa: BLE001 - interpreter shutdown
            pass


class _OneFrame(_Closable, MemoryTrajectory):
    """A file that holds one frame (a LAMMPS data file, a DL_POLY CONFIG)."""

    def __init__(self, frame: Frame, *, periodic=None, **kwargs) -> None:
        super().__init__([frame], **kwargs)
        self.periodic = periodic


# ---------------------------------------------------------------------------
# what every reader records
# ---------------------------------------------------------------------------

def _load_summary(trajectory: Trajectory, frame0: Frame) -> str:
    """The load record, as one sentence: the resolved path, format, atoms,
    frames, frame 0's box and composition. Readers put it first in
    ``notes``, so an export that copies the notes keeps it."""
    cell = frame0.cell()
    skipped = f"; {len(trajectory.skipped)} frame(s) skipped" \
        if trajectory.skipped else ""
    return (f"read {trajectory.source_path} as {trajectory.file_format}: "
            f"{trajectory.n_atoms} atoms x {trajectory.n_frames} frame(s)"
            f"{skipped}; frame 0 box a {cell.a:.6g}, b {cell.b:.6g}, c "
            f"{cell.c:.6g} Å, alpha {cell.alpha:.6g}, beta {cell.beta:.6g}, "
            f"gamma {cell.gamma:.6g} deg, volume {frame0.volume_ang3:.6g} Å^3, "
            f"origin ({', '.join(f'{x:.6g}' for x in frame0.origin_ang)}) Å; "
            "composition (frame 0) " + ", ".join(
                f"{el} {count}" for el, count in frame0.composition.items()))


def _finish(trajectory: Trajectory, frame0: Frame | None = None) -> Trajectory:
    """Put the load record first in the notes, and the timestep-order note
    after the reader's own, loading frame 0 when the reader has not built it.

    A frame 0 that cannot be built raises FrameError naming the file and the
    frame, as a later frame's FrameError names the frame.
    """
    name = Path(trajectory.source_path).name
    if frame0 is None:
        try:
            frame0 = trajectory.frame(0)
        except FrameError as error:
            raise FrameError(f"{name}, {error}") from error
    trajectory.notes.insert(0, _load_summary(trajectory, frame0))
    order = _timestep_order_note(trajectory.timesteps,
                                 trajectory.file_positions)
    if order:
        trajectory.notes.append(order)
    return trajectory


def _frame0(name: str, position: int, build: Callable[[], Frame]) -> Frame:
    """``build()``, with a ValueError reported as FrameError naming the file
    and the frame (frame 0 is parsed when the file is opened)."""
    try:
        return build()
    except FrameError:
        raise
    except ValueError as error:
        raise FrameError(f"{name}, frame 0 (file position {position}): "
                         f"{error}") from error


def _timestep_order_note(timesteps, positions) -> str | None:
    """A note naming repeated or decreasing timesteps, or None.

    A run restarted and appended to the same file (``dump_modify append``)
    writes its first frames again; the frames are kept in file order, and the
    note says where the order breaks, so an average over frames can be read
    knowing it.
    """
    known = [(int(p), int(t)) for p, t in zip(positions, timesteps)
             if int(t) != NO_TIMESTEP]
    breaks = []
    for (_, before), (position, step) in zip(known, known[1:]):
        if step == before:
            breaks.append(f"timestep {step} repeats at file position {position}")
        elif step < before:
            breaks.append(f"timestep {before} is followed by {step} at file "
                          f"position {position}")
    if not breaks:
        return None
    shown = "; ".join(breaks[:5])
    more = f"; {len(breaks) - 5} more" if len(breaks) > 5 else ""
    return ("the timesteps do not increase from frame to frame (" + shown + more
            + "); the frames are kept in file order, as a restart appended to "
            "the same file leaves them")


def _majority_count(counts: Sequence[int]) -> int:
    """The atom count most frames hold (the first frame's on a tie), so one
    odd frame does not put every other frame in ``skipped``."""
    tally = Counter(counts)
    best = max(tally.values())
    return next(c for c in counts if tally[c] == best)


def _count_reason(n: int, n0: int, counts: Sequence[int]) -> str:
    held = sum(1 for c in counts if c == n0)
    return f"{n} atoms, where {held} of the {len(counts)} frames hold {n0}"


# ---------------------------------------------------------------------------
# LAMMPS data files
# ---------------------------------------------------------------------------

# Section keywords read_data knows [1]. A line that is exactly one of these
# (after its comment is removed) starts a section.
_DATA_SECTIONS = frozenset({
    "Atoms", "Velocities", "Masses", "Ellipsoids", "Lines", "Triangles",
    "Bodies", "Bonds", "Angles", "Dihedrals", "Impropers", "Pair Coeffs",
    "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs",
    "Improper Coeffs", "BondBond Coeffs", "BondAngle Coeffs",
    "MiddleBondTorsion Coeffs", "EndBondTorsion Coeffs", "AngleTorsion Coeffs",
    "AngleAngleTorsion Coeffs", "BondBond13 Coeffs", "AngleAngle Coeffs",
    "Atom Type Labels", "Bond Type Labels", "Angle Type Labels",
    "Dihedral Type Labels", "Improper Type Labels"})

# The sections whose rows are read. read_data's fix keyword hands sections the
# input script names to a fix [1], such as the core/shell model's CS-Info
# [10], so a valid file can hold section names that are not in the list
# above. Inside a section that is read (or one a fix defines), a line holding
# no number at all cannot be one of its rows (each starts with an atom id or
# a type and holds numbers) and has the shape of a keyword, so it starts such
# a section; its rows were otherwise read as Atoms, Velocities or Masses rows,
# and the file refused. After a section that is not read, its rows are not
# parsed, so such a section there changes nothing that is read and is left in
# the section before it.
_DATA_READ_SECTIONS = frozenset({"Atoms", "Velocities", "Masses",
                                 "Atom Type Labels"})


def _fix_section_keyword(tokens: Sequence[str]) -> bool:
    return not any(_is_number(token) for token in tokens)


_DATA_HEADER = {
    "atoms": re.compile(r"^(\S+)\s+atoms$"),
    "atom types": re.compile(r"^(\S+)\s+atom\s+types$"),
    "xlo xhi": re.compile(r"^(\S+)\s+(\S+)\s+xlo\s+xhi$"),
    "ylo yhi": re.compile(r"^(\S+)\s+(\S+)\s+ylo\s+yhi$"),
    "zlo zhi": re.compile(r"^(\S+)\s+(\S+)\s+zlo\s+zhi$"),
    "xy xz yz": re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+xy\s+xz\s+yz$"),
    "avec": re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+avec$"),
    "bvec": re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+bvec$"),
    "cvec": re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+cvec$"),
    "abc origin": re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+abc\s+origin$"),
}
_DATA_HEADER_TOPOLOGY = re.compile(
    r"^\S+\s+(bonds|angles|dihedrals|impropers|bond\s+types|angle\s+types|"
    r"dihedral\s+types|improper\s+types|extra\s+bond\s+per\s+atom|"
    r"extra\s+angle\s+per\s+atom|extra\s+dihedral\s+per\s+atom|"
    r"extra\s+improper\s+per\s+atom|extra\s+special\s+per\s+atom|ellipsoids|"
    r"lines|triangles|bodies)$")
_WRITE_DATA_TITLE = re.compile(
    r"^LAMMPS data file via write_data.*?timestep\s*=\s*(-?\d+)"
    r"(?:.*?units\s*=\s*(\w+))?")

# Atom styles read, as (id column, type column, charge column or None, first
# position column, number of columns) in an Atoms line [1]. bond, angle and
# molecular share one layout.
_ATOM_STYLES = {
    "atomic": (0, 1, None, 2, 5),
    "charge": (0, 1, 2, 3, 6),
    "molecular": (0, 2, None, 3, 6),
    "bond": (0, 2, None, 3, 6),
    "angle": (0, 2, None, 3, 6),
    "full": (0, 2, 3, 4, 7),
}
# Columns of an Atoms line for every fixed-width atom style in the read_data
# table [1] (tdpd and hybrid add a variable number). Image flags add three to
# any of them. Only 5 columns name one style (atomic); every other count fits
# several, so without a '# style' comment it is refused, naming them: 7 is
# full but also sphere (atom-ID atom-type diameter density x y z), whose
# diameter would be read as the type.
_STYLE_COLUMNS = {
    "angle": 6, "atomic": 5, "body": 7, "bond": 6, "bpm/sphere": 8,
    "charge": 6, "dielectric": 14, "dipole": 9, "dpd": 6, "edpd": 7,
    "electron": 8, "ellipsoid": 7, "full": 7, "line": 8, "mdpd": 6,
    "molecular": 6, "peri": 7, "rheo": 7, "rheo/thermal": 8, "smd": 13,
    "sph": 8, "sphere": 7, "spin": 9, "template": 8, "tri": 8,
}


def _styles_with(n_cols: int) -> list[str]:
    """Every atom style of the read_data table an Atoms line of ``n_cols``
    values fits, with and without image flags."""
    plain = [s for s, c in _STYLE_COLUMNS.items() if c == n_cols]
    flagged = [f"{s} with image flags" for s, c in _STYLE_COLUMNS.items()
               if c == n_cols - 3]
    return plain + flagged


@dataclass
class _DataFile:
    title: str
    header: dict[str, tuple[int, list[str]]]
    unread_header: list[str]
    sections: dict[str, list[tuple[int, list[str]]]]
    style_hint: str | None
    # The section holding the file's last line of text, when that line has no
    # line ending after it (None when it has one).
    unended_section: str | None = None
    # Section names taken as sections a fix defines (_fix_section_keyword).
    fix_sections: list[str] = field(default_factory=list)


def _parse_data_file(path: Path) -> _DataFile:
    raw = _read_all(path)
    lines = _lines(_strip_bom(raw))
    if not lines:
        raise UnsupportedFormat(f"{path.name}: the file is empty")
    trimmed = raw.rstrip(b" \t\f\v")
    unended = bool(trimmed.strip()) and not trimmed.endswith((b"\n", b"\r"))
    header: dict[str, tuple[int, list[str]]] = {}
    unread: list[str] = []
    sections: dict[str, list[tuple[int, list[str]]]] = {}
    style_hint = None
    current = None
    last_section = "the header"
    fix_sections: list[str] = []
    for number, line in enumerate(lines[1:], start=2):
        content, _, comment = line.partition("#")
        stripped = " ".join(content.split())
        if stripped in _DATA_SECTIONS:
            if stripped in sections:
                raise ValueError(f"{path.name}, line {number}: a second "
                                 f"'{stripped}' section")
            current = stripped
            sections[current] = []
            last_section = current
            if current == "Atoms" and comment.split():
                style_hint = comment.split()[0].lower()
            continue
        if not stripped:
            continue
        if current is not None:
            tokens = stripped.split()
            if (current in _DATA_READ_SECTIONS or current in fix_sections) \
                    and _fix_section_keyword(tokens):
                current = stripped
                if current not in fix_sections:
                    fix_sections.append(current)
                sections.setdefault(current, [])
                last_section = current
                continue
            sections[current].append((number, tokens))
            last_section = current
            continue
        for key, pattern in _DATA_HEADER.items():
            match = pattern.match(stripped)
            if match:
                if key in header:
                    raise ValueError(f"{path.name}, line {number}: a second "
                                     f"'{key}' header line")
                header[key] = (number, list(match.groups()))
                break
        else:
            if not _DATA_HEADER_TOPOLOGY.match(stripped):
                unread.append(f"line {number}: {stripped!r}")
    return _DataFile(lines[0].strip(), header, unread, sections, style_hint,
                     last_section if unended else None, fix_sections)


def _header_numbers(data: _DataFile, key: str, path: Path) -> list[float]:
    number, values = data.header[key]
    return _finite(values, f"{path.name}, line {number} ('{key}')")


def _header_count(data: _DataFile, key: str, path: Path) -> int | None:
    if key not in data.header:
        return None
    number, values = data.header[key]
    try:
        count = int(values[0])
    except ValueError:
        raise ValueError(f"{path.name}, line {number}: '{values[0]} {key}' is "
                         "not a whole number") from None
    if count < 0:
        raise ValueError(f"{path.name}, line {number}: a negative count")
    return count


def _data_box(data: _DataFile, path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    general = [k for k in ("avec", "bvec", "cvec") if k in data.header]
    restricted = [k for k in ("xlo xhi", "ylo yhi", "zlo zhi")
                  if k in data.header]
    if general and (restricted or "xy xz yz" in data.header):
        raise ValueError(f"{path.name}: the header gives both a general "
                         "triclinic box (avec/bvec/cvec) and xlo/xhi bounds")
    if general:
        if len(general) != 3:
            missing = sorted({"avec", "bvec", "cvec"} - set(general))
            raise ValueError(f"{path.name}: a general triclinic box needs "
                             f"avec, bvec and cvec; {missing} missing")
        box = np.array([_header_numbers(data, k, path)
                        for k in ("avec", "bvec", "cvec")])
        if "abc origin" in data.header:
            origin = np.array(_header_numbers(data, "abc origin", path))
            return box, origin, "general triclinic (avec, bvec, cvec, abc origin)"
        return box, np.zeros(3), ("general triclinic (avec, bvec, cvec); no "
                                  "'abc origin' line, so LAMMPS's default "
                                  "origin (0, 0, 0) is used")
    if len(restricted) != 3:
        missing = sorted({"xlo xhi", "ylo yhi", "zlo zhi"} - set(restricted))
        raise ValueError(f"{path.name}: the header has no {missing} line(s), so "
                         "the box is not stated")
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = (
        _header_numbers(data, k, path) for k in ("xlo xhi", "ylo yhi", "zlo zhi"))
    tilt = _header_numbers(data, "xy xz yz", path) \
        if "xy xz yz" in data.header else [0.0, 0.0, 0.0]
    box, origin = box_from_lammps(xlo, xhi, ylo, yhi, zlo, zhi, *tilt)
    kind = "restricted triclinic (xy xz yz)" if "xy xz yz" in data.header \
        else "orthogonal"
    return box, origin, kind


def _data_type_info(data: _DataFile, path: Path
                    ) -> tuple[dict[int, float], dict[int, str]]:
    """(type -> mass, type -> label) from Masses and Atom Type Labels."""
    labels: dict[int, str] = {}
    for number, tokens in data.sections.get("Atom Type Labels", []):
        if len(tokens) < 2:
            raise ValueError(f"{path.name}, line {number}: an Atom Type Labels "
                             "line needs a type and a label")
        try:
            labels[int(tokens[0])] = tokens[1]
        except ValueError:
            raise ValueError(f"{path.name}, line {number}: type {tokens[0]!r} "
                             "is not a whole number") from None
    by_label = {label: type_id for type_id, label in labels.items()}
    masses: dict[int, float] = {}
    for number, tokens in data.sections.get("Masses", []):
        if len(tokens) < 2:
            raise ValueError(f"{path.name}, line {number}: a Masses line needs "
                             "a type and a mass")
        key = tokens[0]
        if key in by_label:
            type_id = by_label[key]
        else:
            try:
                type_id = int(key)
            except ValueError:
                raise ValueError(f"{path.name}, line {number}: type {key!r} is "
                                 "neither a number nor a defined type label") \
                    from None
        try:
            masses[type_id] = float(tokens[1])
        except ValueError:
            raise ValueError(f"{path.name}, line {number}: mass {tokens[1]!r} "
                             "is not a number") from None
    return masses, labels


def _int_column(values, what: str, where: str) -> np.ndarray:
    try:
        return np.asarray(values).astype(np.int64)
    except (ValueError, OverflowError):
        raise ValueError(f"{where}: the {what} column holds a value that is not "
                         "a whole number") from None


def _float_column(values, what: str, where: str) -> np.ndarray:
    try:
        return np.asarray(values).astype(np.float64)
    except ValueError:
        raise ValueError(f"{where}: the {what} column holds a value that is not "
                         "a number") from None


def read_lammps_data(path, *, type_map: Mapping | None = None,
                     atom_style: str | None = None, units: str | None = None,
                     mass_tol_amu: float = MASS_TOL_AMU) -> Trajectory:
    """A LAMMPS data file as a one-frame trajectory.

    Atom styles atomic, charge, molecular, bond, angle and full [1]. The style
    is taken from the ``Atoms # style`` comment, or from ``atom_style``
    (refused when the two differ). Without either it is inferred only from 5
    columns, the one count that names a single style of the read_data table
    (atomic), with a note; any other count is refused, naming the styles it
    fits. Trailing image flags give ``unwrapped_cart_ang``; a Velocities
    section gives velocities (unit per ``units``, or the write_data title, or
    noted as assumed); the charge column gives charges. Types become elements
    as the module docstring describes. A file whose last line has no line
    ending is refused when that line is in a section that is read (it may be
    cut inside its last number), and noted otherwise. A section named by the
    input script for a fix (read_data's fix keyword [1], such as the
    core/shell model's CS-Info [10]) is recognised after a section that is
    read, by a keyword line holding no number, and named in the notes as not
    read.
    """
    path = _existing(path)
    tol = _checked_tol(mass_tol_amu)
    user_types, user_labels = _normalise_type_map(type_map)
    user_units = _checked_units(units, "the units argument")
    data = _parse_data_file(path)
    if "Atoms" not in data.sections:
        raise UnsupportedFormat(f"{path.name}: no Atoms section, so no atoms "
                                "to read")
    where = path.name
    notes: list[str] = []
    if data.unended_section is not None:
        if data.unended_section in ("the header", "Atoms", "Velocities",
                                    "Masses", "Atom Type Labels"):
            raise ValueError(
                f"{where}: the file does not end with a line ending, and its "
                f"last line is in {data.unended_section}, so its last value may "
                "be cut (a file copied while it was being written); add the "
                "line ending if the file is complete")
        notes.append(f"the file does not end with a line ending; its last line "
                     f"is in the {data.unended_section} section, which is not "
                     "read")

    title = _WRITE_DATA_TITLE.match(data.title)
    timestep = int(title.group(1)) if title else None
    title_units = _checked_units(title.group(2), f"{where}'s first line") \
        if title and title.group(2) else None
    style_units, units_where = _combine_units(title_units, user_units,
                                              f"{where}'s first line")

    hint = data.style_hint
    if atom_style is not None:
        if not isinstance(atom_style, str):
            raise ValueError(f"atom_style {atom_style!r} is not a style name")
        atom_style = atom_style.strip().lower()
    if hint and atom_style and hint != atom_style:
        raise ValueError(f"{where}: the Atoms section says '# {hint}' and "
                         f"atom_style says {atom_style!r}")
    rows = data.sections["Atoms"]
    if not rows:
        raise ValueError(f"{where}: the Atoms section is empty")
    counts = sorted({len(tokens) for _, tokens in rows})
    if len(counts) != 1:
        raise ValueError(f"{where}: Atoms lines have {counts} values; every "
                         "line needs the same columns")
    n_cols = counts[0]
    style = hint or atom_style
    if style is None:
        fits = _styles_with(n_cols)
        if fits != ["atomic"]:
            raise ValueError(
                f"{where}: the Atoms section has no '# style' comment, and "
                f"{n_cols} columns fit "
                + (("the atom styles " + ", ".join(fits)) if fits else
                   "no fixed-width atom style of the read_data table")
                + "; atom_style= states which one (FACET reads "
                + ", ".join(sorted(_ATOM_STYLES)) + ")")
        style = "atomic"
        notes.append(f"atom style atomic inferred from the {n_cols} columns of "
                     "the Atoms section (it has no '# style' comment; no other "
                     "style of the read_data table has 5)")
    else:
        notes.append(f"atom style {style} (from "
                     + ("the Atoms section comment" if hint else
                        "the atom_style argument") + ")")
    if style not in _ATOM_STYLES:
        raise ValueError(f"{where}: atom style {style!r} is not read; FACET "
                         f"reads {', '.join(sorted(_ATOM_STYLES))}")
    id_col, type_col, q_col, x_col, base = _ATOM_STYLES[style]
    if n_cols not in (base, base + 3):
        raise ValueError(f"{where}: atom style {style} has {base} columns (or "
                         f"{base + 3} with image flags); the Atoms lines have "
                         f"{n_cols}")
    table = np.array([tokens for _, tokens in rows], dtype=object)
    n = table.shape[0]
    declared = _header_count(data, "atoms", path)
    if declared is not None and declared != n:
        raise ValueError(f"{where}: the header says {declared} atoms and the "
                         f"Atoms section lists {n}")

    masses, labels = _data_type_info(data, path)
    by_label = {label: type_id for type_id, label in labels.items()}
    raw_types = table[:, type_col]
    types = np.empty(n, dtype=np.int64)
    for i, token in enumerate(raw_types):
        if token in by_label:
            types[i] = by_label[token]
        else:
            try:
                types[i] = int(token)
            except ValueError:
                raise ValueError(f"{where}: atom type {token!r} is neither a "
                                 "number nor a defined type label") from None
    n_types = _header_count(data, "atom types", path)
    if n_types is not None and (types.min() < 1 or types.max() > n_types):
        raise ValueError(f"{where}: atom types run from {types.min()} to "
                         f"{types.max()}; the header declares {n_types}")

    how = ("pass type_map={1: 'Si', 2: 'O', ...}, or add an Atom Type Labels "
           "section whose labels are element symbols")
    present = sorted(set(types.tolist()))
    mapping, source, type_notes = _resolve_types(
        present, user=user_types,
        candidates=[("type label", labels)] if labels else [],
        masses=masses or None, tol=tol, how_to=how)
    notes.extend(type_notes)
    if user_labels:
        notes.append(f"type map entries {sorted(user_labels)} are labels; a "
                     "data file's types are numbers (map them by number)")
    elements = np.array([mapping[t] for t in types.tolist()], dtype="<U2")

    ids = _int_column(table[:, id_col].astype(str), "atom-ID", where)
    cart = _float_column(table[:, x_col:x_col + 3].astype(str), "x y z", where)
    charge = None if q_col is None else \
        _float_column(table[:, q_col].astype(str), "q", where)
    box, origin, box_kind = _data_box(data, path)
    notes.append(f"box: {box_kind}")
    unwrapped = None
    if n_cols == base + 3:
        image = _int_column(table[:, base:base + 3].astype(str), "image flag",
                            where)
        unwrapped = unwrap_with_images(cart, image, box)
        notes.append("image flags read: unwrapped positions are x + image @ box")

    id_values, id_counts = np.unique(ids, return_counts=True)
    repeated = id_values[id_counts > 1]
    if repeated.size:
        raise ValueError(f"{where}: the Atoms section lists atom id(s) "
                         f"{repeated[:5].tolist()} more than once; each atom "
                         "needs its own id")
    velocities = None
    if "Velocities" in data.sections:
        vrows = data.sections["Velocities"]
        if not vrows:
            raise ValueError(f"{where}: the Velocities section is empty (the "
                             "file may be cut after its keyword)")
        short = [number for number, tokens in vrows if len(tokens) < 4]
        if short:
            raise ValueError(f"{where}, line {short[0]}: a Velocities line needs "
                             "atom-ID vx vy vz")
        vtable = np.array([tokens[:4] for _, tokens in vrows], dtype=object)
        vids = _int_column(vtable[:, 0].astype(str), "Velocities atom-ID", where)
        if len(vids) != n or set(vids.tolist()) != set(ids.tolist()):
            raise ValueError(f"{where}: the Velocities section lists {len(vids)} "
                             f"atoms that are not the {n} atoms of the Atoms "
                             "section")
        order = {atom: row for row, atom in enumerate(vids.tolist())}
        values = _float_column(vtable[:, 1:4].astype(str), "vx vy vz", where)
        factor = _VELOCITY_TO_ANG_PER_PS[style_units] if style_units else 1.0
        velocities = values[[order[a] for a in ids.tolist()]] * factor

    unread_sections = sorted(set(data.sections) - _DATA_READ_SECTIONS
                             - set(data.fix_sections))
    if unread_sections:
        notes.append("sections not read: " + ", ".join(unread_sections)
                     + " (bonds are found from the geometry, not taken from "
                       "the file)")
    if data.fix_sections:
        notes.append("section(s) " + ", ".join(data.fix_sections) + " are not "
                     "read_data keywords; they are taken as sections an input "
                     "script hands to a fix (read_data's fix keyword, as the "
                     "core/shell model's CS-Info) and are not read")
    if data.unread_header:
        notes.append("header lines not recognised and not read: "
                     + "; ".join(data.unread_header))
    notes.append("the data file does not record the boundary (the input "
                 "script's boundary command sets it); the box is taken as "
                 "periodic along a, b and c")
    frame = _frame0(where, 0, lambda: frame_from_arrays(
        elements, cart, box_ang=box, origin_ang=origin, atom_id=ids,
        timestep=timestep, unwrapped_cart_ang=unwrapped,
        vel_ang_per_ps=velocities, charge_e=charge))
    return _finish(_OneFrame(
        frame, source_path=str(path), file_format="lammps-data",
        type_map=mapping, type_map_source=source, notes=notes,
        units_note=_units_note(style_units, units_where, has_time=False,
                               has_velocity=velocities is not None)), frame)


# ---------------------------------------------------------------------------
# LAMMPS dumps
# ---------------------------------------------------------------------------

_POSITION_SETS = (("x", ("x", "y", "z")), ("xs", ("xs", "ys", "zs")),
                  ("xu", ("xu", "yu", "zu")), ("xsu", ("xsu", "ysu", "zsu")))
_IMAGE_COLUMNS = ("ix", "iy", "iz")
_VELOCITY_COLUMNS = ("vx", "vy", "vz")


@dataclass
class _DumpBlock:
    position: int
    reason: str | None = None
    timestep: int | None = None
    n_atoms: int = 0
    box: np.ndarray | None = None
    origin: np.ndarray | None = None
    periodic: tuple[bool, bool, bool] | None = None
    boundary: str = ""
    columns: tuple[str, ...] = ()
    wrapped: tuple[str, tuple[str, str, str]] | None = None
    unwrapped: tuple[str, tuple[str, str, str]] | None = None
    rows_start: int = 0
    rows_end: int = 0
    extra_lines: int = 0      # lines after the atom rows before the next ITEM
    time: float | None = None
    units: str | None = None
    other: list[str] = field(default_factory=list)


class _Short(ValueError):
    """A header item followed by fewer lines than it needs."""


def _dump_kind(text: str) -> tuple[str, str]:
    body = text.split("ITEM:", 1)[1].strip()
    upper = body.upper()
    for kind in ("TIMESTEP", "TIME", "UNITS"):
        if upper == kind:
            return kind, ""
    if upper.startswith("NUMBER OF ATOMS"):
        return "NUMBER OF ATOMS", ""
    if upper.startswith("BOX BOUNDS"):
        return "BOX BOUNDS", body[len("BOX BOUNDS"):].strip()
    if upper.startswith("ATOMS"):
        return "ATOMS", body[len("ATOMS"):].strip()
    return "OTHER", body


def _body(mark: _Mark, count: int, item: str) -> list[str]:
    lines = mark.after[:count]
    if len(lines) < count:
        raise _Short(f"ITEM: {item} is followed by {len(lines)} of the {count} "
                     "line(s) it needs")
    return lines


def _numbers(line: str, count: int, item: str, hint: str = "") -> list[float]:
    """Exactly ``count`` numbers: LAMMPS writes no more and no fewer on a
    BOX BOUNDS line, so a third value on an orthogonal box's line is a tilt
    whose header keywords are missing, not a value to drop."""
    tokens = line.split()
    if len(tokens) != count:
        raise ValueError(f"a line of ITEM: {item} holds {len(tokens)} values "
                         f"where {count} are expected{hint}")
    return _finite(tokens, f"ITEM: {item}")


def _dump_header(block: _DumpBlock, items: dict, marks: Sequence[_Mark],
                 scan: _Scan, is_last: bool) -> str | None:
    """Fill a block from its ITEM lines; the reason it cannot be read, or None."""
    missing = [k for k in ("TIMESTEP", "NUMBER OF ATOMS", "BOX BOUNDS", "ATOMS")
               if k not in items]
    if missing:
        what = ", ".join("ITEM: " + k for k in missing)
        if is_last:
            return f"truncated: the file ends inside the frame header (no {what})"
        return f"the frame header has no {what}"
    try:
        (line,) = _body(items["TIMESTEP"][1], 1, "TIMESTEP")
        try:
            block.timestep = int(line.split()[0])
        except (ValueError, IndexError):
            return f"unreadable: ITEM: TIMESTEP holds {line.strip()!r}, not a " \
                   "whole number"
        (line,) = _body(items["NUMBER OF ATOMS"][1], 1, "NUMBER OF ATOMS")
        try:
            block.n_atoms = int(line.split()[0])
        except (ValueError, IndexError):
            return f"unreadable: ITEM: NUMBER OF ATOMS holds {line.strip()!r}, " \
                   "not a whole number"
        if block.n_atoms < 1:
            return f"ITEM: NUMBER OF ATOMS is {block.n_atoms}: no atoms"
        if "UNITS" in items:
            (line,) = _body(items["UNITS"][1], 1, "UNITS")
            block.units = line.strip()
        if "TIME" in items:
            (line,) = _body(items["TIME"][1], 1, "TIME")
            try:
                block.time = float(line.split()[0])
            except (ValueError, IndexError):
                return f"unreadable: ITEM: TIME holds {line.strip()!r}, not a " \
                       "number"
            if not math.isfinite(block.time):
                return "unreadable: ITEM: TIME is not finite"
        rest, mark = items["BOX BOUNDS"][0], items["BOX BOUNDS"][1]
        words = rest.split()
        lines = _body(mark, 3, "BOX BOUNDS")
        if words[:3] == ["xy", "xz", "yz"]:
            values = [_numbers(line, 3, "BOX BOUNDS") for line in lines]
            flags = words[3:]
            block.box, block.origin = box_from_dump_bounds(
                [v[:2] for v in values], [values[0][2], values[1][2], values[2][2]])
        elif words[:2] == ["abc", "origin"]:
            values = [_numbers(line, 4, "BOX BOUNDS") for line in lines]
            flags = words[2:]
            block.box = np.array([v[:3] for v in values])
            block.origin = np.array([v[3] for v in values])
        else:
            values = [_numbers(line, 2, "BOX BOUNDS",
                               " (the header names no tilt factors xy xz yz)")
                      for line in lines]
            flags = words
            block.box, block.origin = box_from_dump_bounds(values, None)
        md_model._checked_box(block.box)
        block.boundary = " ".join(flags)
        block.periodic = tuple(f == "pp" for f in flags) if len(flags) == 3 \
            else None
    except _Short as error:
        return f"truncated: {error}" if is_last else f"unreadable: {error}"
    except ValueError as error:
        return f"unreadable: {error}"

    rest, mark, index = items["ATOMS"]
    block.columns = tuple(rest.split())
    if not block.columns:
        return "ITEM: ATOMS names no columns"
    if len(set(block.columns)) != len(block.columns):
        return "ITEM: ATOMS names a column twice"
    complete = {}
    for kind, names in _POSITION_SETS:
        have = [c for c in names if c in block.columns]
        if len(have) == 3:
            complete[kind] = names
        elif have:
            return (f"ITEM: ATOMS has {' '.join(have)} without the rest of "
                    f"{' '.join(names)}")
    if not complete:
        return ("ITEM: ATOMS has no complete set of positions (x y z, xs ys zs, "
                "xu yu zu or xsu ysu zsu)")
    kind = next(k for k, _ in _POSITION_SETS if k in complete)
    block.wrapped = (kind, complete[kind])
    for kind in ("xu", "xsu"):
        if kind in complete:
            block.unwrapped = (kind, complete[kind])
            break
    block.rows_start = mark.end
    block.rows_end = _range_end(marks, index, scan)
    rows = _rows_between(marks, index, scan)
    if index + 1 < len(marks) and marks[index + 1].glued:
        return (f"its last atom row is cut: an 'ITEM:' line starts inside it "
                f"(line {marks[index + 1].line + 1}), as when a run is "
                "appended to a file that a crash ended mid-row")
    if rows < block.n_atoms:
        if is_last:
            return (f"truncated: the file ends after {rows} of {block.n_atoms} "
                    "atom rows")
        return f"{rows} atom rows where ITEM: NUMBER OF ATOMS gives {block.n_atoms}"
    block.extra_lines = rows - block.n_atoms
    if is_last and index + 1 == len(marks):
        if len(scan.last_line.split()) != len(block.columns):
            return (f"truncated: the last atom row holds "
                    f"{len(scan.last_line.split())} of {len(block.columns)} "
                    "values")
        if not scan.last_line_ended:
            return ("truncated: the file does not end with a line ending, so "
                    "the last value of its last atom row may be cut (a file "
                    "copied while it was being written)")
    return None


def _dump_blocks(scan: _Scan) -> tuple[list[_DumpBlock], list[str]]:
    groups: list[list[tuple[str, str, int]]] = []
    seen: set[str] = set()
    other: list[str] = []
    for index, mark in enumerate(scan.marks):
        kind, rest = _dump_kind(mark.text)
        if kind == "OTHER":
            if rest not in other:
                other.append(rest)
        if not groups or "ATOMS" in seen or (kind != "OTHER" and kind in seen):
            groups.append([])
            seen = set()
        groups[-1].append((kind, rest, index))
        seen.add(kind)
    blocks = []
    for position, group in enumerate(groups):
        block = _DumpBlock(position=position)
        items = {kind: (rest, scan.marks[index], index)
                 for kind, rest, index in group if kind != "OTHER"}
        block.reason = _dump_header(block, items, scan.marks, scan,
                                    position == len(groups) - 1)
        blocks.append(block)
    return blocks, other


def _read_range(path: Path, start: int, end: int) -> bytes:
    """Bytes ``start`` .. ``end`` of the (decompressed) file."""
    with _open_binary(path) as handle:
        try:
            handle.seek(start)
        except EOFError:
            return b""
        except _GZIP_DAMAGE as error:
            raise _damaged(handle, error) from None
        data, _, _ = _read_some(handle, end - start)
    return data


def _check_extra_rows(path: Path, blocks: Sequence[_DumpBlock]) -> None:
    """Give a reason to each frame whose ITEM: ATOMS block has more lines than
    NUMBER OF ATOMS and more values than its rows hold.

    The index pass counts lines, blank ones included; only a frame with extra
    lines (none in a file LAMMPS wrote) is read here, so one with blank lines
    stays readable and one with stray rows is skipped at open, not refused
    when it is loaded. One forward pass over the file.
    """
    suspects = [b for b in blocks if b.reason is None and b.extra_lines > 0]
    if not suspects:
        return
    with _open_binary(path) as handle:
        for block in suspects:
            try:
                handle.seek(block.rows_start)
            except EOFError:
                block.reason = "truncated: the gzip stream ends inside this frame"
                continue
            except _GZIP_DAMAGE as error:
                raise _damaged(handle, error) from None
            data, _, _ = _read_some(handle, block.rows_end - block.rows_start)
            values = len(data.split())
            width = len(block.columns)
            if values != block.n_atoms * width:
                block.reason = (
                    f"{block.n_atoms + block.extra_lines} lines with {values} "
                    f"values follow ITEM: ATOMS, where ITEM: NUMBER OF ATOMS "
                    f"gives {block.n_atoms} rows of {width} values")


def _dump_columns(block: _DumpBlock, identity) -> tuple[list[str], list[str]]:
    """(columns read, columns not read with the reason) of a dump frame, from
    what ``_dump_frame`` and ``_dump_identity`` use, not from column names: a
    lone vx, or ix iy iz next to xu yu zu, is listed as not read."""
    cols = block.columns
    read: list[str] = []
    not_read: list[str] = []
    claimed: set[str] = set()

    def take(names):
        read.extend(names)
        claimed.update(names)

    def skip(names, why):
        not_read.append(f"{' '.join(names)} ({why})")
        claimed.update(names)

    if "id" in cols:
        take(["id"])
    if identity[0] == "type":
        take(["type"])
        take([c for c in ("element", "typelabel") if c in cols])
    else:
        take([identity[1]])
        for column in ("element", "typelabel", "type"):
            if column in cols and column != identity[1]:
                skip([column], f"elements come from the {identity[1]} column")
    names = block.wrapped[1]
    take(list(names))
    unwrapped = block.unwrapped[1] if block.unwrapped is not None else None
    if unwrapped is not None and unwrapped != names:
        take(list(unwrapped))
    for _, others in _POSITION_SETS:
        if all(c in cols for c in others) and others[0] not in claimed:
            skip(others, f"positions come from {' '.join(names)}")
    image = [c for c in _IMAGE_COLUMNS if c in cols]
    if len(image) == 3 and unwrapped is None:
        take(image)
    elif len(image) == 3:
        skip(image, f"unwrapped positions come from {' '.join(unwrapped)}")
    elif image:
        skip(image, "image flags need all of ix iy iz")
    velocity = [c for c in _VELOCITY_COLUMNS if c in cols]
    if len(velocity) == 3:
        take(velocity)
    elif velocity:
        skip(velocity, "velocities need all of vx vy vz")
    if "q" in cols:
        take(["q"])
    not_read.extend(c for c in cols if c not in claimed)
    return read, not_read


def _dump_table(data: bytes, block: _DumpBlock) -> np.ndarray:
    tokens = data.split()
    n, width = block.n_atoms, len(block.columns)
    if len(tokens) != n * width:
        raise ValueError(
            f"the ITEM: ATOMS rows hold {len(tokens)} values where {n} rows x "
            f"{width} columns ({' '.join(block.columns)}) = {n * width} are "
            "expected")
    return np.array(tokens).reshape(n, width)


def _columns(table: np.ndarray, block: _DumpBlock, names, kind: str) -> np.ndarray:
    index = [block.columns.index(c) for c in names]
    raw = table[:, index] if len(index) > 1 else table[:, index[0]]
    what = " ".join(names)
    if kind == "int":
        return _int_column(raw, what, "ITEM: ATOMS")
    return _float_column(raw, what, "ITEM: ATOMS")


def _decoded(column: np.ndarray) -> np.ndarray:
    return np.char.decode(column, "utf-8", "replace") if column.dtype.kind == "S" \
        else column.astype(str)


def _dump_elements(table: np.ndarray, block: _DumpBlock, identity) -> np.ndarray:
    if identity[0] == "type":
        if "type" not in block.columns:
            raise ValueError("this frame has no type column")
        types = _columns(table, block, ["type"], "int")
        unique, inverse = np.unique(types, return_inverse=True)
        missing = [int(t) for t in unique if int(t) not in identity[1]]
        if missing:
            raise ValueError(
                f"type(s) {missing} occur in this frame; the type map was built "
                "when the file was opened, from frame 0 and the sources given, "
                "and names no element for them; pass type_map= with these types")
        symbols = np.array([identity[1][int(t)] for t in unique], dtype="<U2")
        return symbols[inverse.reshape(-1)]
    column, mapping = identity[1], identity[2]
    if column not in block.columns:
        raise ValueError(f"this frame has no {column} column")
    tokens = _decoded(table[:, block.columns.index(column)])
    unique, inverse = np.unique(tokens, return_inverse=True)
    symbols = []
    for token in unique:
        token = str(token)
        symbols.append(mapping[token] if token in mapping
                       else _file_symbol(token))
    return np.array(symbols, dtype="<U2")[inverse.reshape(-1)]


def _dump_frame(data: bytes, block: _DumpBlock, identity, velocity_factor: float,
                time_ps: float | None) -> Frame:
    """One dump frame. Without an id column the rows are put in element order
    (stable), so that atom k is an atom of the same element in every frame
    of an unchanged composition; it is not the same atom (``ids_track_atoms``
    is False for such a file)."""
    table = _dump_table(data, block)
    elements = _dump_elements(table, block, identity)
    cols = block.columns
    ids = _columns(table, block, ["id"], "int") if "id" in cols else None
    box, origin = block.box, block.origin
    kind, names = block.wrapped
    raw = _columns(table, block, names, "float")
    unwrapped = None
    if block.unwrapped is not None:
        ukind, unames = block.unwrapped
        values = raw if unames == names else _columns(table, block, unames, "float")
        unwrapped = values if ukind == "xu" else origin + values @ box
    elif all(c in cols for c in _IMAGE_COLUMNS):
        image = _columns(table, block, _IMAGE_COLUMNS, "int")
        absolute = raw if kind == "x" else origin + raw @ box
        unwrapped = unwrap_with_images(absolute, image, box)
    velocities = None
    if all(c in cols for c in _VELOCITY_COLUMNS):
        velocities = _columns(table, block, _VELOCITY_COLUMNS, "float") \
            * velocity_factor
    charge = _columns(table, block, ["q"], "float") if "q" in cols else None
    if ids is None:
        order = np.argsort(elements, kind="stable")
        elements, raw = elements[order], raw[order]
        unwrapped = None if unwrapped is None else unwrapped[order]
        velocities = None if velocities is None else velocities[order]
        charge = None if charge is None else charge[order]
    cart, frac = (raw, None) if kind in ("x", "xu") else (None, raw)
    return frame_from_arrays(elements, cart, frac=frac, box_ang=box,
                             origin_ang=origin, atom_id=ids,
                             timestep=block.timestep, time_ps=time_ps,
                             unwrapped_cart_ang=unwrapped,
                             vel_ang_per_ps=velocities, charge_e=charge)


def _type_column_is_numeric(table: np.ndarray, block: _DumpBlock) -> bool:
    """False when the type column holds labels (``dump_modify types labels``
    writes a type label in place of the number)."""
    try:
        _columns(table, block, ["type"], "int")
    except ValueError:
        return False
    return True


def _dump_identity(table: np.ndarray, block: _DumpBlock, user_types, user_labels,
                   masses_from, tol: float, name: str):
    """How rows get elements: ('type', map) or ('tokens', column, label map)."""
    cols = block.columns
    numeric_type = "type" in cols and _type_column_is_numeric(table, block)
    token_columns = [c for c in ("element", "typelabel") if c in cols]
    if "type" in cols and not numeric_type:
        token_columns.append("type")
    source_of = {"element": "element column", "typelabel": "type label",
                 "type": "type label"}
    notes: list[str] = []
    if numeric_type:
        types = _columns(table, block, ["type"], "int")
        present = sorted(set(types.tolist()))
        candidates = []
        for column in token_columns:
            tokens = _decoded(table[:, cols.index(column)])
            per_type = {}
            for type_id in present:
                values = sorted(set(tokens[types == type_id].tolist()))
                if len(values) == 1:
                    per_type[type_id] = values[0]
                else:
                    notes.append(f"type {type_id} carries {values} in the "
                                 f"{column} column; that column is not used "
                                 "for it")
            candidates.append((source_of[column], per_type))
        masses = None
        if masses_from is not None:
            data_path = _existing(masses_from)
            parsed = _parse_data_file(data_path)
            if parsed.unended_section in ("Masses", "Atom Type Labels"):
                raise ValueError(
                    f"{data_path.name} (masses_from): the file does not end with "
                    f"a line ending, and its last line is in "
                    f"{parsed.unended_section}, so its last value may be cut; "
                    "add the line ending if the file is complete")
            masses, labels = _data_type_info(parsed, data_path)
            if labels:
                candidates.append(("type label", labels))
            notes.append(f"types matched with the Masses and labels of "
                         f"{data_path.resolve()}")
        how = "pass type_map={1: 'Si', 2: 'O', ...}"
        if masses_from is None:
            how += (", or masses_from=<the LAMMPS data file the run read>, "
                    "whose Masses can name the elements")
        if "mass" in cols:
            per_type_mass = {}
            mass_values = _columns(table, block, ["mass"], "float")
            for type_id in present:
                per_type_mass[type_id] = sorted(set(
                    mass_values[types == type_id].tolist()))
            how += ("; the dump's mass column gives "
                    + ", ".join(f"type {t}: {v} amu"
                                for t, v in per_type_mass.items()))
        later = set(user_types)
        if masses is not None:
            later |= set(masses)
        for source_name, labels in candidates:
            if source_name == "type label" and masses_from is not None:
                later |= set(labels)
        mapping, source, type_notes = _resolve_types(
            present, user=user_types, candidates=candidates, masses=masses,
            tol=tol, how_to=f"{name}: {how}", later=sorted(later - set(present)),
            absent="no atom has those types in frame 0 or in the other sources")
        notes.extend(type_notes)
        if user_labels:
            notes.append(f"type map entries {sorted(user_labels)} are labels; "
                         "this dump is mapped by its type numbers")
        return ("type", mapping), mapping, source, notes
    if masses_from is not None:
        raise ValueError(f"{name}: masses_from maps type numbers, and this "
                         "dump has no numeric type column")
    if not token_columns:
        raise ValueError(f"{name}: ITEM: ATOMS has no type, element or typelabel "
                         "column, so its atoms cannot be given elements")
    column = token_columns[0]
    tokens = sorted(set(_decoded(table[:, cols.index(column)]).tolist()))
    per_label, listed, source, label_notes = _resolve_labels(
        tokens, user_labels, what=f"{column} value")
    if source == "file symbols":
        source = source_of[column]
    elif source == "user + file symbols":
        source = "user + " + source_of[column]
    notes.extend(label_notes)
    if column == "type":
        notes.append("the type column holds type labels (dump_modify types "
                     "labels), read as the typelabel column would be")
    if len(token_columns) > 1:
        notes.append(f"elements from the {column} column; the "
                     + " and ".join(c for c in token_columns if c != column)
                     + " column(s) are not used for them")
    notes.extend(_numeric_entries_note(user_types, f"{column} column"))
    return ("tokens", column, per_label), listed, source, notes


class _DumpTrajectory(_FileTrajectory):
    def __init__(self, path: Path, blocks: Sequence[_DumpBlock], *, identity,
                 velocity_factor: float, **kwargs) -> None:
        super().__init__(path, [(b.rows_start, b.rows_end) for b in blocks],
                         **kwargs)
        self._blocks = list(blocks)
        self._identity = identity
        self._velocity_factor = velocity_factor
        # Frame 0's composition, for a dump without ids: its rows are matched
        # by element only, so a frame of another composition is named as such.
        self._composition: dict[str, int] | None = None

    def _load(self, k: int) -> Frame:
        time = None
        if self.times_ps is not None and np.isfinite(self.times_ps[k]):
            time = float(self.times_ps[k])
        frame = _dump_frame(self._bytes(k), self._blocks[k], self._identity,
                            self._velocity_factor, time)
        if self._composition is not None and frame.composition != self._composition:
            raise ValueError(
                f"this frame holds {frame.composition}, frame 0 holds "
                f"{self._composition}; the dump has no id column, so its atoms "
                "are matched between frames by element only")
        return frame


def read_lammps_dump(path, *, type_map: Mapping | None = None,
                     masses_from=None, units: str | None = None,
                     mass_tol_amu: float = MASS_TOL_AMU) -> Trajectory:
    """A LAMMPS text dump (``dump atom`` or ``custom``), plain or gzip.

    Columns are taken from each frame's ``ITEM: ATOMS`` line, in any order:
    ``id``, ``type``, ``element``, ``typelabel``, positions as ``x y z`` |
    ``xs ys zs`` | ``xu yu zu`` | ``xsu ysu zsu`` (the first complete set in
    that order gives the frame; ``xu``/``xsu`` or ``x``/``xs`` with ``ix iy iz``
    give the unwrapped positions), ``vx vy vz`` and ``q``. A type column that
    holds labels (``dump_modify types labels``) is read as labels. ``ITEM:
    UNITS`` and ``ITEM: TIME`` are read when present. ``masses_from`` names a
    LAMMPS data file whose Masses (and Atom Type Labels) can give the types
    their elements, including types frame 0 does not hold. Frames that are
    truncated, cut by an appended restart, hold stray rows or another atom
    count than most frames, or have a header that does not parse are counted
    in ``skipped`` with the reason. Without an id column each frame's rows are
    put in element order and ``ids_track_atoms`` is False (module docstring).
    """
    path = _existing(path)
    tol = _checked_tol(mass_tol_amu)
    user_types, user_labels = _normalise_type_map(type_map)
    user_units = _checked_units(units, "the units argument")
    scan = _scan(path, b"ITEM:", line_start=True, after_lines=3, glued=True)
    if not scan.marks:
        raise UnsupportedFormat(f"{path.name}: no 'ITEM:' lines, so not a "
                                "LAMMPS dump")
    blocks, other = _dump_blocks(scan)
    _check_extra_rows(path, blocks)
    readable = [b for b in blocks if b.reason is None]
    if readable:
        counts = [b.n_atoms for b in readable]
        n0 = _majority_count(counts)
        for block in readable:
            if block.n_atoms != n0:
                block.reason = _count_reason(block.n_atoms, n0, counts)
    readable = [b for b in blocks if b.reason is None]
    skipped = {b.position: b.reason for b in blocks if b.reason is not None}
    if not readable:
        raise ValueError(f"{path.name} holds no readable frame: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))

    file_units = sorted({b.units for b in blocks if b.units})
    if len(file_units) > 1:
        raise ValueError(f"{path.name}: ITEM: UNITS gives {file_units} in "
                         "different frames")
    stated = _checked_units(file_units[0], f"{path.name}'s ITEM: UNITS") \
        if file_units else None
    style, units_where = _combine_units(stated, user_units,
                                        f"{path.name}'s ITEM: UNITS")
    time_factor = _TIME_TO_PS[style] if style else 1.0
    velocity_factor = _VELOCITY_TO_ANG_PER_PS[style] if style else 1.0
    times = None
    if any(b.time is not None for b in readable):
        times = [np.nan if b.time is None else b.time * time_factor
                 for b in readable]

    first = readable[0]
    name = path.name
    notes: list[str] = []
    data = _read_range(path, first.rows_start, first.rows_end)
    table0 = _frame0(name, first.position, lambda: _dump_table(data, first))
    identity, listed, source, type_notes = _dump_identity(
        table0, first, user_types, user_labels, masses_from, tol, name)
    notes.extend(type_notes)

    cols = first.columns
    used, unread = _dump_columns(first, identity)
    notes.append(f"frame positions from {' '.join(first.wrapped[1])}"
                 + (f"; unwrapped positions from {' '.join(first.unwrapped[1])}"
                    if first.unwrapped else
                    "; unwrapped positions from the image flags ix iy iz"
                    if all(c in cols for c in _IMAGE_COLUMNS) else
                    "; no unwrapped positions (neither xu/xsu nor ix iy iz)"))
    notes.append("columns read: " + " ".join(used)
                 + (("; not read: " + ", ".join(unread)) if unread else ""))
    if "id" not in cols:
        notes.append(
            "no id column: in every frame the rows are put in element order and "
            "numbered 0 .. N-1; LAMMPS re-orders rows between frames (atom_modify "
            "sort, on by default, and processor order), so atom k is not the "
            "same atom in two frames (ids_track_atoms is False): each frame is "
            "measured as a structure, and per-atom quantities across frames "
            "(displacements, mean-square displacement) need a dump with an id "
            "column")
    boundaries = sorted({b.boundary for b in readable})
    if any(b.periodic is None for b in readable):
        notes.append("ITEM: BOX BOUNDS states no boundary flags; the box is "
                     "taken as periodic along a, b and c")
    nonperiodic = sorted({b.boundary for b in readable
                          if b.periodic is not None and not all(b.periodic)})
    if nonperiodic:
        notes.append(f"boundary flags {nonperiodic}: not periodic along every "
                     "axis, while FACET measures the box as periodic along a, b "
                     "and c, so atoms near a non-periodic face see images "
                     "across it")
    if other:
        notes.append("ITEM lines not read: " + "; ".join(other))
    if scan.gzip_cut:
        notes.append(_GZIP_CUT_NOTE)
    box_varies = any(not np.array_equal(b.box, first.box)
                     or not np.array_equal(b.origin, first.origin)
                     for b in readable[1:])

    trajectory = _DumpTrajectory(
        path, readable, identity=identity, velocity_factor=velocity_factor,
        periodic=None if first.periodic is None or len(boundaries) != 1
        else first.periodic,
        file_format="lammps-dump", n_atoms=first.n_atoms,
        n_frames=len(readable), type_map_source=source,
        timesteps=[b.timestep for b in readable], times_ps=times,
        type_map=listed, skipped=skipped, notes=notes,
        units_note=_units_note(style, units_where, has_time=times is not None,
                               has_velocity=all(c in cols for c in
                                                _VELOCITY_COLUMNS)),
        box_varies=box_varies)
    frame0 = _frame0(name, first.position, lambda: _dump_frame(
        data, first, identity, velocity_factor,
        None if times is None or not np.isfinite(times[0]) else float(times[0])))
    if "id" not in cols:
        trajectory.ids_track_atoms = False
        trajectory._composition = frame0.composition
    if frame0.charge_e is not None:
        charges, charge_notes = md_model.charges_per_element(frame0.elements,
                                                             frame0.charge_e)
        trajectory.charges_e = charges
        trajectory.notes.extend(charge_notes)
    return _finish(trajectory, frame0)


# ---------------------------------------------------------------------------
# extended XYZ
# ---------------------------------------------------------------------------

_KEY_VALUE = re.compile(r"""
    \s*(?P<key>[^\s="'{}\[\]]+)
    (?:\s*=\s*(?P<value>
        "(?:\\.|[^"\\])*"
      | '(?:\\.|[^'\\])*'
      | \{[^}]*\}
      | \[(?:[^\[\]]|\[[^\]]*\])*\]
      | [^\s]+
    ))?""", re.VERBOSE)

_XYZ_SPECIES = ("species",)
_XYZ_Z = ("z",)
_XYZ_POSITIONS = ("pos",)
_XYZ_VELOCITIES = ("velo", "vel", "velocities", "velocity")
_XYZ_IDS = ("id", "ids")
# initial_charges is ASE's name for the charges set on the atoms (ASE writes
# them so); charges is what an ASE calculator computed.
_XYZ_CHARGES = ("charge", "charges", "q", "initial_charges")
_XYZ_TYPES = ("type",)
# Per-atom masses (ASE writes 'masses' when they differ from its defaults):
# not stored on the frames, compared with the species' standard atomic
# weights, as a data file's Masses are with its type labels.
_XYZ_MASSES = ("masses", "mass")
# The groups read, each from the first matching column in Properties= order.
_XYZ_GROUPS = (("species", _XYZ_SPECIES), ("Z", _XYZ_Z), ("type", _XYZ_TYPES),
               ("positions", _XYZ_POSITIONS), ("ids", _XYZ_IDS),
               ("velocities", _XYZ_VELOCITIES), ("charges", _XYZ_CHARGES))


def _comment_pairs(comment: str) -> dict[str, tuple[str, str | None]]:
    """lower-case key -> (key as written, value without its quotes or None)."""
    out: dict[str, tuple[str, str | None]] = {}
    position = 0
    text = comment.strip()
    while position < len(text):
        match = _KEY_VALUE.match(text, position)
        if match is None or match.end() == position:
            raise ValueError(f"the comment line does not parse as key=value "
                             f"pairs near {text[position:position + 30]!r}")
        position = match.end()
        key, value = match.group("key"), match.group("value")
        if value is not None:
            if value[:1] in "\"'" and value[-1:] == value[:1] and len(value) >= 2:
                value = re.sub(r"\\(.)", r"\1", value[1:-1])
            elif value[:1] == "{" and value[-1:] == "}":
                value = value[1:-1]
        lower = key.lower()
        if lower in out:
            raise ValueError(f"the comment line gives {key!r} twice")
        out[lower] = (key, value)
    return out


def _numbers_of(value: str, count: int, key: str) -> list[float]:
    tokens = value.replace("[", " ").replace("]", " ").replace(",", " ").split()
    if len(tokens) != count:
        raise ValueError(f"{key}= holds {len(tokens)} values where {count} "
                         "numbers are needed")
    return _finite(tokens, f"{key}=")


@dataclass
class _XyzBlock:
    position: int
    rows_start: int
    rows_end: int
    n_atoms: int
    box: np.ndarray | None = None
    origin: np.ndarray | None = None
    properties: list[tuple[str, str, int]] = field(default_factory=list)
    width: int = 0
    pbc: tuple[bool, bool, bool] | None = None
    time: float | None = None
    timestep: int | None = None
    unread_keys: list[str] = field(default_factory=list)
    reason: str | None = None


def _xyz_header(block: _XyzBlock, comment: str) -> str | None:
    try:
        pairs = _comment_pairs(comment)
    except ValueError as error:
        return f"unreadable: {error}"
    if "lattice" not in pairs or pairs["lattice"][1] is None:
        return "no Lattice= in its comment line"
    try:
        rows = _numbers_of(pairs["lattice"][1], 9, "Lattice")
        block.box = np.array(rows).reshape(3, 3)
        md_model._checked_box(block.box)
        block.origin = np.array(_numbers_of(pairs["origin"][1], 3, "Origin")) \
            if "origin" in pairs and pairs["origin"][1] else np.zeros(3)
        spec = pairs["properties"][1] if "properties" in pairs else \
            "species:S:1:pos:R:3"
        parts = (spec or "").split(":")
        if len(parts) % 3 or not parts[0]:
            raise ValueError(f"Properties={spec!r} is not a list of "
                             "name:type:count triplets")
        props = []
        for i in range(0, len(parts), 3):
            name, kind, count = parts[i], parts[i + 1].upper(), parts[i + 2]
            if kind not in ("S", "R", "I", "L") or not count.isdigit() \
                    or int(count) < 1:
                raise ValueError(f"Properties= entry {name}:{parts[i + 1]}:"
                                 f"{count} is not name:S|R|I|L:count")
            props.append((name, kind, int(count)))
        block.properties = props
        block.width = sum(c for _, _, c in props)
        if "pbc" in pairs and pairs["pbc"][1]:
            flags = pairs["pbc"][1].replace(",", " ").replace("[", " ").replace(
                "]", " ").split()
            truth = {"t": True, "true": True, "1": True, "f": False,
                     "false": False, "0": False}
            if len(flags) != 3 or any(f.lower() not in truth for f in flags):
                raise ValueError(f"pbc={pairs['pbc'][1]!r} is not three "
                                 "booleans")
            block.pbc = tuple(truth[f.lower()] for f in flags)
        if "time" in pairs and pairs["time"][1] is not None:
            block.time = _finite([pairs["time"][1]], "Time=")[0]
        for key in ("timestep", "step"):
            if key in pairs and pairs[key][1] is not None:
                value = pairs[key][1].strip()
                if not re.fullmatch(r"[+-]?\d+", value):
                    raise ValueError(f"{pairs[key][0]}={value!r} is not a "
                                     "whole number")
                block.timestep = int(value)
                break
    except ValueError as error:
        return f"unreadable: {error}"
    known = {"lattice", "origin", "properties", "pbc", "time", "timestep", "step"}
    block.unread_keys = [k for low, (k, _) in pairs.items() if low not in known]
    names = {n.lower(): (kind, count) for n, kind, count in props}
    if "pos" not in names or names["pos"] != ("R", 3):
        return "Properties= has no pos:R:3"
    return None


def _xyz_count(line: bytes) -> int | None:
    """The atom count a frame's first line gives: one whole number of 1 or
    more and nothing else, or None. Strict, so that an atom row whose first
    column is a number (a type or Z) is not taken for a count."""
    tokens = line.split()
    if len(tokens) != 1 or not tokens[0].isdigit():
        return None
    n = int(tokens[0])
    return n if n >= 1 else None


class _Lines:
    """readline with one line of look-ahead, counting lines (1-based)."""

    def __init__(self, handle) -> None:
        self.handle = handle
        self.number = 0
        self._back: list[bytes] = []

    def read(self) -> bytes:
        line = self._back.pop() if self._back else _readline(self.handle)
        if line:
            self.number += 1
        return line

    def unread(self, line: bytes) -> None:
        if line:
            self._back.append(line)
            self.number -= 1


def _xyz_index(path: Path) -> tuple[list[_XyzBlock], dict[int, str], list[str]]:
    """Walk the frames by their atom counts; one readline per line.

    Blank lines between frames are passed over and counted in a note (the
    count line delimits frames, so they are unambiguous). When the line after
    a frame's rows is not the next frame's count, the frame's count and rows
    disagree: that frame is skipped, the walk looks for the next count line
    followed by a comment line holding ``Lattice=``, and every comment line
    holding ``Lattice=`` passed on the way gets its own skipped position, so
    each lost frame is counted. Returns (readable blocks, skipped, notes).
    """
    blocks: list[_XyzBlock] = []
    skipped: dict[int, str] = {}
    notes: list[str] = []
    position = 0
    blank_lines = 0
    blank_before: list[int] = []      # file positions with blank lines before
    blank_total = 0
    _refuse_cr_only(path.name, _head(path))
    with _open_binary(path) as handle:
        lines = _Lines(handle)
        try:
            first_line = True
            while True:
                line = lines.read()
                if first_line:
                    line = _strip_bom(line)
                    first_line = False
                if not line:
                    break
                if not line.strip():
                    blank_lines += 1
                    continue
                n = _xyz_count(line)
                if n is None:
                    lost, at = _xyz_resync(lines, line)
                    what = (f"line {at} ({_text(line).strip()[:40]!r}) is not "
                            "an atom count line")
                    if blocks and blocks[-1].position == position - 1:
                        previous = blocks.pop()
                        skipped[previous.position] = (
                            f"its atom count line gives {previous.n_atoms}, and "
                            f"after that many rows {what}, so the count and the "
                            "rows disagree")
                    elif position - 1 in skipped:
                        skipped[position - 1] += (f"; after its rows {what}")
                    elif not lost:
                        skipped[position] = f"unreadable: {what}"
                        position += 1
                    for _ in range(lost):
                        skipped[position] = (
                            "unreadable: its comment line (with Lattice=) has no "
                            "atom count line before it where the frame before "
                            "ends, so its rows cannot be delimited")
                        position += 1
                    continue
                if blank_lines and position > 0:
                    blank_before.append(position)
                    blank_total += blank_lines
                blank_lines = 0
                comment = lines.read()
                if not comment:
                    skipped[position] = ("truncated: the file ends after the "
                                         "atom count")
                    break
                start = handle.tell() - sum(len(x) for x in lines._back)
                got = 0
                last = b""
                for _ in range(n):
                    row = lines.read()
                    if not row:
                        break
                    last = row
                    got += 1
                block = _XyzBlock(position, start, handle.tell(), n)
                block.reason = _xyz_header(block, _text(comment))
                if got < n:
                    skipped[position] = (f"truncated: the file ends after {got} "
                                         f"of {n} atom rows")
                    break
                if block.reason is None and not last.endswith((b"\n", b"\r")):
                    block.reason = (
                        f"truncated: the file does not end with a line ending, "
                        f"so the last value of its last atom row may be cut "
                        f"(the row holds {len(last.split())} of {block.width} "
                        "values)")
                if block.reason is not None:
                    skipped[position] = block.reason
                else:
                    blocks.append(block)
                position += 1
        except EOFError:
            skipped[position] = ("truncated: the gzip stream ends before its "
                                 "end marker")
            notes.append(_GZIP_CUT_NOTE)
    if blank_before:
        # One note for the file: a writer that puts a blank line after every
        # frame would otherwise add a note per frame.
        shown = ", ".join(str(p) for p in blank_before[:5])
        more = f" and {len(blank_before) - 5} more" if len(blank_before) > 5 \
            else ""
        notes.append(f"{blank_total} blank line(s) between frames, before file "
                     f"position(s) {shown}{more}, are passed over")
    return blocks, skipped, notes


def _xyz_resync(lines: _Lines, line: bytes) -> tuple[int, int]:
    """Pass lines until a count line followed by a comment line holding
    ``Lattice=``, and leave both unread. Returns (the number of comment lines
    holding ``Lattice=`` passed, each a frame whose count line was lost; the
    line number where the walk lost step)."""
    at = lines.number
    lost = 1 if b"lattice=" in line.lower() else 0
    while True:
        candidate = lines.read()
        if not candidate:
            return lost, at
        if _xyz_count(candidate) is not None:
            following = lines.read()
            if b"lattice=" in following.lower():
                lines.unread(following)
                lines.unread(candidate)
                return lost, at
            lines.unread(following)
            continue
        if b"lattice=" in candidate.lower():
            lost += 1


def _xyz_symbols_from_z(values: np.ndarray) -> np.ndarray:
    import gemmi

    unique, inverse = np.unique(values, return_inverse=True)
    symbols = []
    for z in unique:
        value = int(z)
        try:
            # gemmi raises TypeError for a number beyond a C int (Z = 10**12)
            element = gemmi.Element(value) if value >= 1 else None
        except (TypeError, OverflowError):
            element = None
        if element is None or element.atomic_number != value:
            raise ValueError(f"Z = {value} is not an atomic number")
        symbols.append(validate_symbol(element.name))
    return np.array(symbols, dtype="<U2")[inverse.reshape(-1)]


def _xyz_columns(block: _XyzBlock, names: Sequence[str], kind: str, count: int
                 ) -> tuple[int, str] | None:
    offset = 0
    for name, ptype, pcount in block.properties:
        if name.lower() in names:
            if pcount != count or (kind and ptype != kind):
                raise ValueError(f"Properties= gives {name}:{ptype}:{pcount}; "
                                 f"{kind or 'S'}:{count} is needed")
            return offset, name
        offset += pcount
    return None


def _xyz_table(data: bytes, block: _XyzBlock) -> np.ndarray:
    tokens = data.split()
    n, width = block.n_atoms, block.width
    if len(tokens) != n * width:
        raise ValueError(f"the atom rows hold {len(tokens)} values where {n} "
                         f"rows x {width} Properties= columns = {n * width} are "
                         "expected")
    return np.array(tokens).reshape(n, width)


def _xyz_frame(data: bytes, block: _XyzBlock, user_types, label_map,
               time_ps, velocity_factor: float = 1.0) -> Frame:
    table = _xyz_table(data, block)

    def take(names, kind, count, what, convert):
        found = _xyz_columns(block, names, kind, count)
        if found is None:
            return None
        offset, _ = found
        raw = table[:, offset:offset + count] if count > 1 else table[:, offset]
        return convert(raw, what, "the atom rows")

    species = _xyz_columns(block, _XYZ_SPECIES, "S", 1)
    if species is not None:
        tokens_ = _decoded(table[:, species[0]])
        unique, inverse = np.unique(tokens_, return_inverse=True)
        symbols = [label_map[str(t)] if str(t) in label_map else
                   _file_symbol(str(t)) for t in unique]
        elements = np.array(symbols, dtype="<U2")[inverse.reshape(-1)]
    else:
        z = take(_XYZ_Z, "I", 1, "Z", _int_column)
        if z is not None:
            elements = _xyz_symbols_from_z(z)
        else:
            types = take(_XYZ_TYPES, "I", 1, "type", _int_column)
            if types is None:
                raise ValueError("Properties= has no species, Z or type column")
            missing = sorted({int(t) for t in types} - set(user_types))
            if missing:
                raise ValueError(f"type(s) {missing} have no element in the "
                                 "type map")
            elements = np.array([user_types[int(t)] for t in types], dtype="<U2")
    cart = take(_XYZ_POSITIONS, "R", 3, "pos", _float_column)
    ids = take(_XYZ_IDS, "I", 1, "id", _int_column)
    velocities = take(_XYZ_VELOCITIES, "R", 3, "velocity", _float_column)
    if velocities is not None:
        velocities = velocities * velocity_factor
    charge = take(_XYZ_CHARGES, "R", 1, "charge", _float_column)
    return frame_from_arrays(elements, cart, box_ang=block.box,
                             origin_ang=block.origin, atom_id=ids,
                             timestep=block.timestep, time_ps=time_ps,
                             vel_ang_per_ps=velocities, charge_e=charge)


def _xyz_group(name: str) -> str | None:
    return next((g for g, names in _XYZ_GROUPS if name.lower() in names), None)


def _xyz_columns_note(properties, *, masses_checked: bool = False
                      ) -> tuple[list[str], list[str]]:
    """(columns read, columns not read with the reason): one column per group
    of :data:`_XYZ_GROUPS`, the first in Properties= order. The elements come
    from species, else Z, else type, whatever their order in Properties=, as
    :func:`_xyz_frame` takes them. With ``masses_checked`` the first masses
    column is listed as read (it checks the species, see read_extxyz)."""
    read: list[str] = []
    not_read: list[str] = []
    taken: dict[str, str] = {}
    identity = None
    for wanted in ("species", "Z", "type"):
        identity = next((f"{n}:{k}:{c}" for n, k, c in properties
                         if _xyz_group(n) == wanted), None)
        if identity is not None:
            break
    for name, kind, count in properties:
        written = f"{name}:{kind}:{count}"
        group = _xyz_group(name)
        if group in ("species", "Z", "type"):
            if written == identity and "identity" not in taken:
                taken["identity"] = written
                read.append(written)
            else:
                not_read.append(f"{written} (elements come from {identity})")
        elif name.lower() in _XYZ_MASSES and masses_checked \
                and "masses" not in taken and (kind, count) == ("R", 1):
            taken["masses"] = written
            read.append(written)
        elif group is not None and group not in taken:
            taken[group] = written
            read.append(written)
        elif group is not None:
            not_read.append(f"{written} ({group} come from {taken[group]})")
        elif name.lower() == "momenta":
            not_read.append(
                f"{written} (ASE writes momenta, mass x velocity in ASE's units; "
                "they are not turned into velocities, which would need the "
                "masses ASE used, its own table unless a masses column gives "
                "them, and ASE's unit of time)")
        else:
            not_read.append(written)
    return read, not_read


class _XyzTrajectory(_FileTrajectory):
    def __init__(self, path: Path, blocks: Sequence[_XyzBlock], *, user_types,
                 label_map, velocity_factor: float = 1.0, **kwargs) -> None:
        super().__init__(path, [(b.rows_start, b.rows_end) for b in blocks],
                         **kwargs)
        self._blocks = list(blocks)
        self._user_types = dict(user_types)
        self._label_map = dict(label_map)
        self._velocity_factor = velocity_factor

    def _load(self, k: int) -> Frame:
        time = None
        if self.times_ps is not None and np.isfinite(self.times_ps[k]):
            time = float(self.times_ps[k])
        return _xyz_frame(self._bytes(k), self._blocks[k], self._user_types,
                          self._label_map, time, self._velocity_factor)


def _xyz_mass_column(block: _XyzBlock) -> int | None:
    """The offset of the first masses column when it is masses:R:1, else
    None (a column of another shape is listed as not read)."""
    offset = 0
    for name, kind, count in block.properties:
        if name.lower() in _XYZ_MASSES:
            return offset if (kind, count) == ("R", 1) else None
        offset += count
    return None


def _species_z_note(table0: np.ndarray, block: _XyzBlock,
                    species_symbols: np.ndarray) -> list[str]:
    """A note when a Z column beside the species names other elements: two
    element sources in one file that disagree, as for LAMMPS types, are
    noted with the one used (the species, the order _xyz_frame takes)."""
    try:
        found = _xyz_columns(block, _XYZ_Z, "I", 1)
        z_values = _int_column(table0[:, found[0]], "Z", "the atom rows")
        z_symbols = _xyz_symbols_from_z(z_values)
    except ValueError as error:
        return [f"the Z column is not compared with the species ({error}); "
                "elements come from the species column"]
    differ = np.flatnonzero(z_symbols != species_symbols)
    if not differ.size:
        return []
    i = int(differ[0])
    return [f"the Z column names another element than the species column for "
            f"{differ.size} of {len(z_symbols)} atoms of frame 0 (first: row "
            f"{i + 1}, species read as {species_symbols[i]}, Z = "
            f"{int(z_values[i])}, {z_symbols[i]}); elements come from the "
            "species column"]


def read_extxyz(path, *, type_map: Mapping | None = None,
                units: str | None = None,
                mass_tol_amu: float = MASS_TOL_AMU) -> Trajectory:
    """Extended XYZ with one or more frames [5].

    Each frame's comment line needs ``Lattice=`` (nine numbers, the vectors as
    rows, in double or single quotes, braces or brackets); without it the
    first frame is refused, because a configuration without a periodic box is
    a molecule, and a later frame is skipped with that reason. ``Properties=``
    name:type:count triplets give the columns (``species:S:1:pos:R:3`` when
    absent). Read: species (or Z, or type with a type map), pos, velo/vel,
    id, charge/charges/q/initial_charges; ``Time=``, ``Timestep=``/``step=``,
    ``pbc=`` and an ``Origin=`` key. Every other column and key is named in
    the notes. A ``masses`` column checks the species as a data file's Masses
    check its labels (``mass_tol_amu``; a species whose mass lies nearer
    another element is refused), and a Z column beside the species is
    compared with it, a difference noted.

    The format states no unit for ``Time=`` or velocities. LAMMPS's own
    ``dump extxyz`` writes both in the run's unit style (fs and Å/fs under
    units real), so ``units`` ('metal' or 'real') converts them as for a
    LAMMPS dump; without it they are stored as written and ``units_note``
    says so.
    """
    path = _existing(path)
    tol = _checked_tol(mass_tol_amu)
    user_types, user_labels = _normalise_type_map(type_map)
    style = _checked_units(units, "the units argument")
    blocks, skipped, index_notes = _xyz_index(path)
    if 0 in skipped and skipped[0].startswith("no Lattice="):
        raise ValueError(
            f"{path.name}: the first frame's comment line has no Lattice=, so "
            "the file states no periodic box. read_trajectory reads periodic "
            "models only; LAMMPS's 'dump xyz' writes no box, while 'dump custom' "
            "and 'dump extxyz' do. A single-frame XYZ opens with readers.read, "
            "which places a molecule in a box")
    if not blocks:
        raise ValueError(f"{path.name} holds no readable frame: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))
    counts = [b.n_atoms for b in blocks]
    n0 = _majority_count(counts)
    for block in blocks:
        if block.n_atoms != n0:
            skipped[block.position] = _count_reason(block.n_atoms, n0, counts)
    blocks = [b for b in blocks if b.position not in skipped]
    first = blocks[0]

    notes: list[str] = list(index_notes)
    data = _read_range(path, first.rows_start, first.rows_end)
    names = [n.lower() for n, _, _ in first.properties]
    listed: dict = {}
    table0 = _frame0(path.name, first.position, lambda: _xyz_table(data, first))
    masses_checked = False
    if "species" in names:
        found = _xyz_columns(first, _XYZ_SPECIES, "S", 1)
        row_species = _decoded(table0[:, found[0]])
        species = sorted(set(row_species.tolist()))
        by_label = None
        mass_offset = _xyz_mass_column(first)
        if mass_offset is not None:
            values = _frame0(path.name, first.position, lambda: _float_column(
                table0[:, mass_offset], "masses", "the atom rows"))
            by_label = {}
            for label, mass in zip(row_species.tolist(), values.tolist(),
                                   strict=True):
                by_label.setdefault(label, mass)
            masses_checked = True
        label_map, listed, source, label_notes = _resolve_labels(
            species, user_labels, masses=by_label, tol=tol, what="species")
        notes.extend(label_notes)
        if masses_checked:
            notes.append("the masses column of frame 0 is compared with each "
                         "species' standard atomic weight (the first mass "
                         "given for it); the frames do not store masses")
        notes.extend(_numeric_entries_note(user_types, "species column"))
        if "z" in names:
            notes.extend(_species_z_note(table0, first, np.array(
                [label_map[str(t)] for t in row_species.tolist()])))
    elif "z" in names:
        label_map, source = {}, "file symbols"
        notes.append("elements from the Z (atomic number) column")
        notes.extend(_numeric_entries_note(user_types, "Z column"))
        if user_labels:
            notes.append(f"type map entries {sorted(user_labels)} are not used: "
                         "the elements come from the Z column")
    elif "type" in names:
        label_map, source = {}, "user"
        if not user_types:
            raise ValueError(f"{path.name}: the atoms carry numeric types and no "
                             "species; pass type_map={1: 'Si', ...}")
        listed = dict(user_types)
        if user_labels:
            notes.append(f"type map entries {sorted(user_labels)} are labels; "
                         "this file is mapped by its numeric type column")
    else:
        raise ValueError(f"{path.name}: Properties= has no species, Z or type "
                         "column, so the atoms cannot be given elements")
    used, unread = _xyz_columns_note(first.properties,
                                     masses_checked=masses_checked)
    notes.append("columns read: " + " ".join(used)
                 + (("; not read: " + ", ".join(unread)) if unread else ""))
    charge_column = next((n for n in names if n in _XYZ_CHARGES), None)
    if charge_column == "initial_charges":
        notes.append("per-atom charges from initial_charges (ASE's name for the "
                     "charges set on the atoms)")
    if first.unread_keys:
        notes.append("comment keys not read: " + ", ".join(first.unread_keys))
    if not any(n in _XYZ_IDS for n in names):
        notes.append("no id column: atoms are numbered 0 .. N-1 in the order "
                     "of the rows, taken as the same atoms in every frame")
    if any(np.any(b.origin) for b in blocks):
        notes.append("box origin from the Origin= key (not part of the extended "
                     "XYZ specification)")
    else:
        notes.append("no Origin= key: the box origin is taken as (0, 0, 0) (LAMMPS's "
                     "extxyz dump writes none, so its positions may differ from "
                     "the run's by whole lattice vectors; the periodic structure "
                     "is the same)")
    # The specification's default: periodic along all three when Lattice= is
    # given and pbc= is not.
    periodic = first.pbc if first.pbc is not None else (True, True, True)
    if any(b.pbc is not None and not all(b.pbc) for b in blocks):
        notes.append("pbc= marks an axis as not periodic, while FACET measures "
                     "the box as periodic along a, b and c, so atoms near that "
                     "face see images across it")
    velocity_group = [n for n in names if n in _XYZ_VELOCITIES]
    has_velocity = bool(velocity_group)
    time_factor = _TIME_TO_PS[style] if style else 1.0
    velocity_factor = _VELOCITY_TO_ANG_PER_PS[style] if style else 1.0
    times = None
    if any(b.time is not None for b in blocks):
        times = [np.nan if b.time is None else b.time * time_factor
                 for b in blocks]
    if style is not None:
        units_text = f"LAMMPS units {style} (the units argument): lengths in Å"
        if style == "real" and (times is not None or has_velocity):
            units_text += ("; Time= converted from fs to ps and velocities from "
                           "Å/fs to Å/ps")
    else:
        units_text = "lengths read as Å (the extended XYZ format states no unit)"
        if times is not None or has_velocity:
            units_text += (
                "; Time= values and velocities are stored as written, which is "
                "ps and Å/ps if the writer used them; LAMMPS's extxyz dump writes "
                "them in the run's unit style, fs and Å/fs under units real (a "
                "time 1000 times its value in ps, a velocity 1/1000 of its value "
                "in Å/ps); units='real' or units='metal' states the style")
    box_varies = any(not np.array_equal(b.box, first.box)
                     or not np.array_equal(b.origin, first.origin)
                     for b in blocks[1:])
    timesteps = [NO_TIMESTEP if b.timestep is None else b.timestep
                 for b in blocks]
    trajectory = _XyzTrajectory(
        path, blocks, user_types=user_types, label_map=label_map,
        velocity_factor=velocity_factor,
        periodic=periodic, file_format="extxyz", n_atoms=n0,
        n_frames=len(blocks), type_map_source=source, timesteps=timesteps,
        times_ps=times, type_map=listed, skipped=skipped, notes=notes,
        units_note=units_text, box_varies=box_varies)
    frame0 = _frame0(path.name, first.position, lambda: _xyz_frame(
        data, first, user_types, label_map,
        None if times is None or not np.isfinite(times[0]) else float(times[0]),
        velocity_factor))
    if frame0.charge_e is not None:
        charges, charge_notes = md_model.charges_per_element(frame0.elements,
                                                             frame0.charge_e)
        trajectory.charges_e = charges
        trajectory.notes.extend(charge_notes)
    return _finish(trajectory, frame0)


# ---------------------------------------------------------------------------
# VASP XDATCAR
# ---------------------------------------------------------------------------

_POTCAR_NAME = re.compile(r"^([A-Za-z]{1,2})[_/].+$")


def _vasp_species(token: str) -> tuple[str, str | None]:
    """'Na_pv' -> ('Na', note); 'Na' -> ('Na', None)."""
    match = _POTCAR_NAME.match(token)
    if match:
        return match.group(1), (f"VASP species {token!r} read as "
                                f"{match.group(1)} (a POTCAR name: the part "
                                "after '_' or '/' names the potential)")
    return token, None


@dataclass
class _VaspHeader:
    box: np.ndarray
    cart_scale: np.ndarray       # multiplies Cartesian positions
    species: tuple[str, ...]
    counts: tuple[int, ...]
    scale_note: str


def _vasp_header(lines: Sequence[str]) -> _VaspHeader:
    """The seven VASP 5 header lines: title, scale, a, b, c, names, counts."""
    if len(lines) != 7:
        raise ValueError(f"a VASP 5 header has 7 lines; {len(lines)} found")
    scale = _finite(lines[1].split(), "the scale-factor line")
    vectors = []
    for i in (2, 3, 4):
        tokens = lines[i].split()
        if len(tokens) < 3:
            raise ValueError(f"lattice vector line {lines[i].strip()!r} holds "
                             "fewer than 3 numbers")
        vectors.append(_finite(tokens[:3], "a lattice vector line"))
    vectors = np.array(vectors)
    species = tuple(lines[5].split())
    if not species or any(_is_number(t) for t in species):
        raise ValueError("line 6 holds no element names (the VASP 4 layout "
                         "puts the counts there); FACET reads the VASP 5 layout")
    try:
        counts = tuple(int(t) for t in lines[6].split())
    except ValueError:
        raise ValueError(f"line 7 {lines[6].strip()!r} is not a list of atom "
                         "counts") from None
    if len(counts) != len(species) or any(c < 0 for c in counts) \
            or sum(counts) < 1:
        raise ValueError(f"{len(species)} element names and the counts "
                         f"{list(counts)} do not match")
    if len(scale) == 1:
        s = scale[0]
        if s == 0:
            raise ValueError("the scale factor is 0")
        if s < 0:
            current = abs(float(np.linalg.det(vectors)))
            if current == 0:
                raise ValueError("the lattice vectors span no volume")
            factor = (abs(s) / current) ** (1.0 / 3.0)
            note = (f"scale factor {s:g} read as the cell volume in Å^3 (a "
                    f"negative scale, as VASP defines it): factor {factor:.10g}")
        else:
            factor = s
            note = f"scale factor {s:g}"
        box = vectors * factor
        cart_scale = np.full(3, factor)
    elif len(scale) == 3:
        if any(v <= 0 for v in scale):
            raise ValueError("three scale factors need to be positive (VASP)")
        box = vectors * np.array(scale)[None, :]
        cart_scale = np.array(scale)
        note = (f"three scale factors {scale} applied to the x, y and z "
                "components")
    else:
        raise ValueError(f"line 2 holds {len(scale)} scale factors; 1 or 3 are "
                         "allowed")
    md_model._checked_box(box)
    return _VaspHeader(box, cart_scale, species, counts, note)


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


@dataclass
class _VaspBlock:
    position: int
    rows_start: int
    rows_end: int
    header: _VaspHeader
    cartesian: bool
    timestep: int | None


# A number as Fortran writes one, exponent letter E or D. VASP writes
# coordinates in fixed-width fields, and a negative value that fills its field
# leaves no blank before its sign ('-0.63265286-0.11227753'); such a token is
# split into the numbers it is made of.
_FORTRAN_NUMBER = re.compile(rb"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?")


def _fused_numbers(row: bytes) -> list[float] | None:
    """The leading numbers of a row whose tokens may be numbers written with
    no blank between them; None when fewer than three lead the row."""
    out: list[float] = []
    for token in row.split():
        if _FORTRAN_NUMBER.sub(b"", token):
            break               # a token that is not only numbers ends the run
        out.extend(float(x.replace(b"D", b"E").replace(b"d", b"e"))
                   for x in _FORTRAN_NUMBER.findall(token))
        if len(out) >= 3:
            return out[:3]
    return None


def _coordinate_rows(rows: Sequence[bytes]) -> tuple[np.ndarray, int]:
    """(N, 3) coordinates from XDATCAR rows, and how many rows needed their
    numbers split at a sign; ValueError naming the first row that holds
    fewer than three numbers."""
    try:
        coords = np.array([row.split()[:3] for row in rows]).astype(np.float64)
        if coords.shape == (len(rows), 3):
            return coords, 0
    except ValueError:
        pass
    out = np.empty((len(rows), 3))
    fused = 0
    for i, row in enumerate(rows):
        tokens = row.split()[:3]
        try:
            values = [float(t) for t in tokens]
        except ValueError:
            values = []
        if len(values) != 3:
            values = _fused_numbers(row)
            if values is None:
                raise ValueError(f"coordinate line {i + 1} of the configuration "
                                 f"({_text(row).strip()[:50]!r}) does not hold "
                                 "three numbers")
            fused += 1
        out[i] = values
    if not np.isfinite(out).all():
        raise ValueError("a coordinate line holds a value that is not finite")
    return out, fused


class _XdatcarTrajectory(_FileTrajectory):
    def __init__(self, path: Path, blocks: Sequence[_VaspBlock], *, elements,
                 **kwargs) -> None:
        super().__init__(path, [(b.rows_start, b.rows_end) for b in blocks],
                         **kwargs)
        self._blocks = list(blocks)
        self._elements = elements

    def _load(self, k: int) -> Frame:
        block = self._blocks[k]
        n = len(self._elements)
        rows = self._bytes(k).splitlines()[:n]
        if len(rows) < n:
            raise ValueError(f"{len(rows)} coordinate lines where the header "
                             f"gives {n} atoms")
        coords, fused = _coordinate_rows(rows)
        notes = [f"{fused} coordinate line(s) hold numbers with no blank between "
                 "them (fixed-width fields); they were split at each sign"] \
            if fused else []
        header = block.header
        if block.cartesian:
            return frame_from_arrays(self._elements, coords * header.cart_scale,
                                     box_ang=header.box, timestep=block.timestep,
                                     notes=notes)
        return frame_from_arrays(self._elements, frac=coords, box_ang=header.box,
                                 timestep=block.timestep, notes=notes)


def read_xdatcar(path, *, type_map: Mapping | None = None) -> Trajectory:
    """A VASP XDATCAR in the VASP 5 layout, fixed or variable cell [6].

    The header (title, scale, three vectors, element names, counts) comes
    first; VASP repeats it before each configuration when the cell varies.
    Each ``Direct configuration= n`` (or Cartesian) block lists every atom.
    The number after ``configuration=`` is kept as the frame's timestep. The
    file holds no time (POTIM and NBLOCK are in the INCAR) and no
    velocities.
    """
    path = _existing(path)
    user_types, user_labels = _normalise_type_map(type_map)
    scan = _scan(path, b"configuration=", line_start=False, before_lines=7)
    if not scan.marks:
        raise UnsupportedFormat(f"{path.name}: no 'configuration=' line, so "
                                "not an XDATCAR")
    marks = scan.marks
    if marks[0].line != 7:
        raise ValueError(
            f"{path.name}: the first 'configuration=' line is line "
            f"{marks[0].line + 1}; the VASP 5 layout puts it on line 8 (title, "
            "scale, three lattice vectors, element names, counts); VASP 4 "
            "files, which have no element names, are not read")
    header0 = _vasp_header(marks[0].before)
    n = sum(header0.counts)
    # Whether VASP repeated the 7-line cell header before a configuration line
    # (a variable cell) is decided by parsing the 7 lines before it, not by the
    # gap alone: with 7 atoms, a fixed-cell file that lost one configuration
    # line has the gap (2 x 7) of a variable cell. With fewer than 7 atoms the
    # lines before a fixed-cell configuration hold the previous configuration
    # line, whose words do not parse as element names.
    gaps = [0] + [marks[k].line - marks[k - 1].line - 1
                  for k in range(1, len(marks))]
    parsed: list[_VaspHeader | None] = [header0]
    errors: list[str | None] = [None]
    for index in range(1, len(marks)):
        header, error = None, None
        if gaps[index] >= n + 7:
            try:
                header = _vasp_header(marks[index].before)
            except ValueError as problem:
                error = str(problem)
        parsed.append(header)
        errors.append(error)
    variable = any(h is not None for h in parsed[1:])
    # In a variable-cell file, a gap of exactly n + 7 whose header does not
    # parse is that configuration's damaged header.
    header_before = [True] + [
        parsed[k] is not None or (variable and gaps[k] == n + 7
                                  and errors[k] is not None)
        for k in range(1, len(marks))]
    period = n + 7 if variable else n      # lines one configuration takes
    blocks: list[_VaspBlock] = []
    skipped: dict[int, str] = {}
    notes: list[str] = []
    current = header0
    position = 0
    for index, mark in enumerate(marks):
        is_last = index + 1 == len(marks)
        reason = None
        if index > 0 and parsed[index] is not None:
            current = parsed[index]
        elif index > 0 and header_before[index]:
            reason = f"unreadable: its cell header: {errors[index]}"
        header = current
        rows = _rows_between(marks, index, scan)
        if not is_last and header_before[index + 1]:
            rows -= 7
        extra = rows - n
        orphans = 0
        if reason is not None:
            pass
        elif header.species != header0.species or header.counts != header0.counts:
            reason = (f"its header lists {list(header.species)} "
                      f"{list(header.counts)}, the first lists "
                      f"{list(header0.species)} {list(header0.counts)}")
        elif extra < 0:
            reason = (f"truncated: the file ends after {rows} of {n} coordinate "
                      "lines") if is_last else \
                f"{rows} coordinate lines where the header gives {n} atoms"
        elif extra > 0 and extra % period == 0 and (not is_last or not variable):
            orphans = extra // period
        elif extra > 0 and not is_last:
            reason = f"{rows} coordinate lines where the header gives {n} atoms"
        elif is_last and extra == 0:
            if len(scan.last_line.split()) < 3:
                reason = (f"truncated: the last coordinate line holds "
                          f"{len(scan.last_line.split())} of 3 values")
            elif not scan.last_line_ended:
                reason = ("truncated: the file does not end with a line ending, "
                          "so the last value of its last coordinate line may be "
                          "cut (a file copied while it was being written)")
        number = mark.text.split("=", 1)[1].split()
        timestep = int(number[0]) if number and re.fullmatch(r"-?\d+", number[0]) \
            else None
        mode = mark.text.strip()[:1]
        if reason is None:
            blocks.append(_VaspBlock(position, mark.end,
                                     _range_end(marks, index, scan), header,
                                     mode in "CcKk", timestep))
        else:
            skipped[position] = reason
        position += 1
        for _ in range(orphans):
            skipped[position] = (
                f"{n} coordinate lines" + (" and a cell header" if variable else "")
                + " with no configuration line before them (a configuration "
                "line is missing)")
            position += 1
        if is_last and extra > 0 and not orphans:
            if variable:
                skipped[position] = (
                    f"truncated: {extra} line(s) follow the last configuration "
                    f"without a configuration line (a cell header takes 7)")
                position += 1
            else:
                notes.append(f"{extra} line(s) after the last configuration "
                             f"are not read (fewer than the {n} of a "
                             "configuration)")
    if not blocks:
        raise ValueError(f"{path.name} holds no readable frame: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))

    tokens = [t for t, c in zip(header0.species, header0.counts, strict=True)
              for _ in range(c)]
    per_label, listed, source, label_notes = _resolve_labels(
        tokens, user_labels, reduce=_vasp_species, what="species")
    elements = np.array([per_label[t] for t in tokens], dtype="<U2")
    notes = list(label_notes) + _numeric_entries_note(user_types, "species") \
        + notes
    notes.append(f"{'variable' if variable else 'fixed'} cell: "
                 + ("the header is repeated before each configuration"
                    if variable else "one header for every configuration")
                 + f"; {header0.scale_note}")
    notes.append("atoms are numbered 0 .. N-1 in the order of the element "
                 "counts; the configuration number is kept as the timestep; the "
                 "XDATCAR holds no time step (POTIM and NBLOCK are in the INCAR) "
                 "and no velocities")
    if scan.gzip_cut:
        notes.append(_GZIP_CUT_NOTE)
    box_varies = any(not np.array_equal(b.header.box, blocks[0].header.box)
                     for b in blocks[1:])
    return _finish(_XdatcarTrajectory(
        path, blocks, elements=elements, periodic=None,
        file_format="vasp-xdatcar", n_atoms=n, n_frames=len(blocks),
        type_map_source=source,
        timesteps=[NO_TIMESTEP if b.timestep is None else b.timestep
                   for b in blocks],
        type_map=listed, skipped=skipped, notes=notes,
        units_note="VASP lengths in Å", box_varies=box_varies))


# ---------------------------------------------------------------------------
# DL_POLY
# ---------------------------------------------------------------------------

_IMCON = {0: "no periodic boundaries", 1: "cubic", 2: "orthorhombic",
          3: "parallelepiped", 4: "truncated octahedral",
          5: "rhombic dodecahedral", 6: "x-y slab with no periodicity along z",
          7: "hexagonal prism"}


def _checked_imcon(imcon: int, where: str) -> str:
    if imcon in (1, 2, 3):
        return _IMCON[imcon]
    meaning = _IMCON.get(imcon, "not a DL_POLY boundary key")
    raise ValueError(
        f"{where}: periodic boundary key imcon = {imcon} ({meaning}); FACET reads "
        "a box periodic along a, b and c (imcon 1, 2 or 3), because wrapping "
        "atoms into a box that is not periodic moves them")


def _dlpoly_ints(line: str, count: int, where: str) -> list[int]:
    tokens = line.split()
    try:
        values = [int(t) for t in tokens[:count]]
    except ValueError:
        raise ValueError(f"{where}: {line.strip()!r} does not start with "
                         f"{count} whole numbers") from None
    if len(values) < count:
        raise ValueError(f"{where}: {line.strip()!r} holds fewer than {count} "
                         "whole numbers")
    return values


def _dlpoly_cell(lines: Sequence[str], where: str) -> tuple[np.ndarray, np.ndarray]:
    if len(lines) < 3:
        raise ValueError(f"{where}: fewer than three cell vector records")
    rows = []
    for line in lines[:3]:
        tokens = line.split()
        if len(tokens) < 3:
            raise ValueError(f"{where}: cell record {line.strip()!r} holds fewer "
                             "than 3 numbers")
        rows.append(_finite(tokens[:3], f"{where}: a cell record"))
    box = md_model._checked_box(np.array(rows))
    return box, -0.5 * box.sum(axis=0)


def _dlpoly_atoms(records: Sequence[bytes], n: int, per: int, where: str):
    """(labels, ids, masses, charges, positions, velocities) of n atom blocks."""
    if len(records) < n * per:
        raise ValueError(f"{where}: {len(records)} atom records where {n} atoms "
                         f"x {per} records = {n * per} are expected")
    heads = [r.split() for r in records[0:n * per:per]]
    if any(not h for h in heads):
        raise ValueError(f"{where}: an atom record has no atom name")
    labels = [h[0].decode("utf-8", "replace") for h in heads]
    ids = None
    if all(len(h) >= 2 for h in heads):
        try:
            ids = np.array([int(h[1]) for h in heads], dtype=np.int64)
        except ValueError:
            ids = None
    masses = charges = None
    if all(len(h) >= 4 for h in heads):
        try:
            masses = np.array([float(h[2]) for h in heads])
            charges = np.array([float(h[3]) for h in heads])
        except ValueError:
            masses = charges = None

    def block(offset, what):
        tokens = b" ".join(records[offset:n * per:per]).split()
        if len(tokens) != 3 * n:
            raise ValueError(f"{where}: the {what} records hold {len(tokens)} "
                             f"values where {3 * n} are expected")
        return _float_column(np.array(tokens).reshape(n, 3), what, where)

    positions = block(1, "position")
    velocities = block(2, "velocity") if per >= 3 else None
    return labels, ids, masses, charges, positions, velocities


def _dlpoly_elements(labels: Sequence[str], user_labels, masses, tol):
    by_label = None
    if masses is not None:
        by_label = {}
        for label, mass in zip(labels, masses, strict=True):
            by_label.setdefault(label, float(mass))
    per_label, listed, source, notes = _resolve_labels(
        labels, user_labels, masses=by_label, tol=tol, what="atom name")
    return np.array([per_label[x] for x in labels], dtype="<U2"), listed, \
        source, notes


def read_dlpoly_config(path, *, type_map: Mapping | None = None,
                       mass_tol_amu: float = MASS_TOL_AMU) -> Trajectory:
    """A DL_POLY CONFIG (or REVCON, CFGMIN) as a one-frame trajectory [7].

    Record 2 gives levcfg (0 positions, 1 and velocities, 2 and forces) and
    imcon; records 3-5 the cell vectors as rows. Each atom is a block of 2 to
    4 records: name (and index), position, velocity, force. Forces are not
    read. Atom names become elements through ``type_map`` (keyed by name) or
    as element symbols.

    Record 2 "may contain more information apart from the mandatory" [7,
    §5.1.2.2]: DL_POLY 4 writes megatm, the atom count, third, and DL_POLY
    Classic writes other items there (the KCl test CONFIG has '2 3 2000
    0.5000000000E-02' over 216 atoms). A third value that differs from the
    atom blocks is refused only when record 2 holds exactly three integers,
    the DL_POLY 4 form; otherwise it is noted, and the count comes from the
    atom blocks.
    """
    path = _existing(path)
    tol = _checked_tol(mass_tol_amu)
    user_types, user_labels = _normalise_type_map(type_map)
    raw = _read_all(path)
    records = _strip_bom(raw).splitlines()
    while records and not records[-1].strip():
        records.pop()
    where = path.name
    if len(records) < 2:
        raise UnsupportedFormat(f"{where}: too short to be a DL_POLY CONFIG")
    levcfg, imcon = _dlpoly_ints(_text(records[1]), 2, f"{where}, record 2")
    if levcfg not in (0, 1, 2):
        raise ValueError(f"{where}: levcfg = {levcfg}; DL_POLY defines 0, 1 "
                         "and 2")
    kind = _checked_imcon(imcon, where)
    box, origin = _dlpoly_cell([_text(r) for r in records[2:5]], where)
    body = records[5:]
    per = 2 + levcfg
    if len(body) % per:
        raise ValueError(f"{where}: {len(body)} atom records do not divide into "
                         f"blocks of {per} (levcfg = {levcfg})")
    n = len(body) // per
    if n < 1:
        raise ValueError(f"{where}: no atom records")
    record2_notes: list[str] = []
    tokens = _text(records[1]).split()
    if len(tokens) >= 3 and re.fullmatch(r"[+-]?\d+", tokens[2]) \
            and int(tokens[2]) != n:
        if len(tokens) == 3:
            raise ValueError(f"{where}: record 2 gives {tokens[2]} atoms "
                             f"(megatm) and the file holds {n} atom blocks")
        record2_notes.append(
            f"record 2 holds {' '.join(tokens)}; its third value {tokens[2]} is "
            f"not the {n} atom blocks (DL_POLY Classic writes other items "
            "there), so the atom count comes from the atom blocks")
    trimmed = raw.rstrip(b" \t\f\v")
    if trimmed.strip() and not trimmed.endswith((b"\n", b"\r")):
        if levcfg < 2:
            raise ValueError(
                f"{where}: the file does not end with a line ending, so the last "
                "value of its last record may be cut (a file copied while it "
                "was being written); add the line ending if the file is complete")
        record2_notes.append("the file does not end with a line ending; its last "
                             "record is a force, which is not read")
    labels, ids, _, _, positions, velocities = _dlpoly_atoms(body, n, per, where)
    elements, listed, source, notes = _dlpoly_elements(labels, user_labels,
                                                       None, tol)
    notes.extend(_numeric_entries_note(user_types, "atom names"))
    notes.extend(record2_notes)
    notes.append(f"DL_POLY boundary key imcon = {imcon} ({kind}); the coordinate "
                 "origin is the centre of the cell (DL_POLY 4 manual, Appendix "
                 "A), so origin_ang = -(a + b + c)/2")
    notes.append(f"levcfg = {levcfg}: "
                 + ("positions" if levcfg == 0 else
                    "positions and velocities" if levcfg == 1 else
                    "positions, velocities and forces (forces not read)"))
    if ids is None:
        notes.append("atom records give no index: atoms are numbered 0 .. N-1 "
                     "in file order")
    frame = _frame0(where, 0, lambda: frame_from_arrays(
        elements, positions, box_ang=box, origin_ang=origin, atom_id=ids,
        vel_ang_per_ps=velocities))
    return _finish(_OneFrame(frame, periodic=(True, True, True),
                             source_path=str(path), file_format="dlpoly-config",
                             type_map=listed, type_map_source=source,
                             notes=notes, units_note="DL_POLY units: Å and Å/ps"),
                   frame)


@dataclass
class _HistoryBlock:
    position: int
    rows_start: int
    rows_end: int
    n_atoms: int
    per: int
    timestep: int
    time_ps: float | None
    box: np.ndarray
    origin: np.ndarray


class _HistoryTrajectory(_FileTrajectory):
    def __init__(self, path: Path, blocks: Sequence[_HistoryBlock], *,
                 label_map: Mapping[str, str], **kwargs) -> None:
        super().__init__(path, [(b.rows_start, b.rows_end) for b in blocks],
                         **kwargs)
        self._blocks = list(blocks)
        self._label_map = dict(label_map)

    def _load(self, k: int) -> Frame:
        block = self._blocks[k]
        records = self._bytes(k).splitlines()[3:]
        labels, ids, _, charges, positions, velocities = _dlpoly_atoms(
            records, block.n_atoms, block.per, "this frame")
        unknown = sorted(set(labels) - set(self._label_map))
        if unknown:
            raise ValueError(f"atom name(s) {unknown} do not occur in the first "
                             "frame")
        elements = np.array([self._label_map[x] for x in labels], dtype="<U2")
        return frame_from_arrays(elements, positions, box_ang=block.box,
                                 origin_ang=block.origin, atom_id=ids,
                                 timestep=block.timestep, time_ps=block.time_ps,
                                 vel_ang_per_ps=velocities, charge_e=charges)


def read_dlpoly_history(path, *, type_map: Mapping | None = None,
                        mass_tol_amu: float = MASS_TOL_AMU) -> Trajectory:
    """A DL_POLY HISTORY trajectory, DL_POLY 4 or Classic layout [7].

    Record 2 gives keytrj, imcon and the atom count. Each frame starts with
    ``timestep nstep natms keytrj imcon tstep [time]`` (time in ps, written by
    DL_POLY 4), then three cell records, then a block per atom: name, index,
    mass, charge [, displacement]; position; velocity (keytrj >= 1); force
    (keytrj = 2, not read).
    """
    path = _existing(path)
    tol = _checked_tol(mass_tol_amu)
    user_types, user_labels = _normalise_type_map(type_map)
    scan = _scan(path, b"timestep", line_start=True, before_lines=2,
                 after_lines=3)
    where = path.name
    # Records 1 and 2 are the header; a title that starts with the word
    # 'timestep' is not a frame.
    marks = [m for m in scan.marks if m.line >= 2]
    if not marks:
        raise UnsupportedFormat(f"{where}: no 'timestep' record, so not a "
                                "DL_POLY HISTORY")
    if marks[0].line != 2 or len(marks[0].before) != 2:
        raise ValueError(f"{where}: the first 'timestep' record is line "
                         f"{marks[0].line + 1}; a HISTORY file has a two-record "
                         "header before it")
    keytrj0, imcon0, _ = _dlpoly_ints(marks[0].before[1], 3,
                                      f"{where}, record 2")
    _checked_imcon(imcon0, where)
    blocks: list[_HistoryBlock] = []
    skipped: dict[int, str] = {}
    last_notes: list[str] = []
    tsteps = set()
    for index, mark in enumerate(marks):
        is_last = index + 1 == len(marks)
        try:
            tokens = mark.text.split()
            if len(tokens) < 6:
                raise ValueError(f"the timestep record {mark.text.strip()!r} "
                                 "holds fewer than 6 items")
            nstep, natms, keytrj, imcon = (int(t) for t in tokens[1:5])
            tstep = _finite([tokens[5]], "the time step (tstep)")[0]
            time = _finite([tokens[6]], "the elapsed time")[0] \
                if len(tokens) >= 7 else None
            _checked_imcon(imcon, "the timestep record")
            if keytrj not in (0, 1, 2):
                raise ValueError(f"keytrj = {keytrj}; DL_POLY defines 0, 1 and 2")
            if natms < 1:
                raise ValueError(f"the timestep record gives {natms} atoms")
            box, origin = _dlpoly_cell(mark.after, "the cell records")
        except ValueError as error:
            skipped[index] = f"unreadable: {error}"
            continue
        per = 2 + keytrj
        expected = 3 + natms * per
        rows = _rows_between(marks, index, scan)
        if rows < expected:
            skipped[index] = (
                f"truncated: the file ends after {rows} of {expected} records"
                if is_last else
                f"{rows} records where {natms} atoms x {per} + 3 cell records = "
                f"{expected} are expected")
            continue
        if rows > expected and not is_last:
            skipped[index] = (f"{rows} records where {natms} atoms x {per} + 3 "
                              f"cell records = {expected} are expected")
            continue
        if is_last and rows == expected:
            # The file's last line is this frame's last record: a position,
            # velocity or force record of three values. Fewer, or no line
            # ending after it, means the file was cut there.
            cut = None
            if len(scan.last_line.split()) < 3:
                cut = (f"its last record holds {len(scan.last_line.split())} "
                       "of 3 values")
            elif not scan.last_line_ended:
                cut = ("the file does not end with a line ending, so the last "
                       "value of its last record may be cut")
            if cut is not None and keytrj == 2:
                last_notes.append(f"the last frame's last record is a force, "
                                  f"which is not read, and {cut}")
            elif cut is not None:
                skipped[index] = f"truncated: {cut}"
                continue
        tsteps.add(tstep)
        blocks.append(_HistoryBlock(index, mark.end, _range_end(marks, index, scan),
                                    natms, per, nstep, time, box, origin))
    if not blocks:
        raise ValueError(f"{where} holds no readable frame: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))
    counts = [b.n_atoms for b in blocks]
    n0 = _majority_count(counts)
    for block in blocks:
        if block.n_atoms != n0:
            skipped[block.position] = _count_reason(block.n_atoms, n0, counts)
    blocks = [b for b in blocks if b.position not in skipped]
    last_rows = _rows_between(marks, len(marks) - 1, scan)
    notes: list[str] = list(last_notes)
    if blocks and blocks[-1].position == len(marks) - 1 \
            and last_rows > 3 + blocks[-1].n_atoms * blocks[-1].per:
        notes.append(f"{last_rows - 3 - blocks[-1].n_atoms * blocks[-1].per} "
                     "line(s) after the last frame are not read")

    first = blocks[0]
    data = _read_range(path, first.rows_start, first.rows_end)
    labels, _, masses, charges, _, _ = _frame0(
        where, first.position, lambda: _dlpoly_atoms(
            data.splitlines()[3:], first.n_atoms, first.per, "the first frame"))
    elements, listed, source, label_notes = _dlpoly_elements(
        labels, user_labels, masses, tol)
    label_map = {x: str(e) for x, e in zip(labels, elements, strict=True)}
    notes.extend(label_notes)
    notes.extend(_numeric_entries_note(user_types, "atom names"))
    notes.append(f"DL_POLY boundary key imcon = {imcon0} ({_IMCON[imcon0]}); "
                 "the coordinate origin is the centre of the cell (DL_POLY 4 "
                 "manual, Appendix A), so origin_ang = -(a + b + c)/2")
    notes.append(f"keytrj = {keytrj0} in the header: "
                 + ("positions" if keytrj0 == 0 else
                    "positions and velocities" if keytrj0 == 1 else
                    "positions, velocities and forces (forces not read)"))
    if len(tsteps) == 1:
        notes.append(f"integration time step {tsteps.pop():g} ps")
    times = None
    if any(b.time_ps is not None for b in blocks):
        times = [np.nan if b.time_ps is None else b.time_ps for b in blocks]
    else:
        notes.append("the timestep records give no elapsed time (the DL_POLY "
                     "Classic layout); frame times are not set")
    if scan.gzip_cut:
        notes.append(_GZIP_CUT_NOTE)
    box_varies = any(not np.array_equal(b.box, first.box) for b in blocks[1:])
    trajectory = _HistoryTrajectory(
        path, blocks, label_map=label_map, periodic=(True, True, True),
        file_format="dlpoly-history", n_atoms=n0, n_frames=len(blocks),
        type_map_source=source, timesteps=[b.timestep for b in blocks],
        times_ps=times, type_map=listed, skipped=skipped, notes=notes,
        units_note="DL_POLY units: Å, Å/ps and ps", box_varies=box_varies)
    if charges is not None:
        values, charge_notes = md_model.charges_per_element(elements, charges)
        trajectory.charges_e = values
        trajectory.notes.extend(charge_notes)
    return _finish(trajectory)


def read_dlpoly(path, *, type_map: Mapping | None = None,
                mass_tol_amu: float = MASS_TOL_AMU) -> Trajectory:
    """A DL_POLY HISTORY or CONFIG, told apart by the 'timestep' record."""
    path = _existing(path)
    lines = _lines(_head(path))
    if len(lines) >= 3 and lines[2].split()[:1] == ["timestep"]:
        return read_dlpoly_history(path, type_map=type_map,
                                   mass_tol_amu=mass_tol_amu)
    return read_dlpoly_config(path, type_map=type_map, mass_tol_amu=mass_tol_amu)


# ---------------------------------------------------------------------------
# recognising a file, and the entry point
# ---------------------------------------------------------------------------

_DATA_ATOMS_LINE = re.compile(r"^\s*\d+\s+atoms\s*(#.*)?$")
_DATA_BOX_LINE = re.compile(r"^\s*\S+\s+\S+\s+xlo\s+xhi\b|^\s*\S+\s+\S+\s+\S+\s+avec\b")


def _all_numbers(tokens: Sequence[str], count: int) -> bool:
    return len(tokens) >= count and all(_is_number(t) for t in tokens[:count])


def _all_ints(tokens: Sequence[str], count: int) -> bool:
    return len(tokens) >= count and all(
        re.fullmatch(r"[+-]?\d+", t) for t in tokens[:count])


def _looks_like_data(lines: Sequence[str]) -> bool:
    body = lines[1:]
    return any(_DATA_ATOMS_LINE.match(x) for x in body) and \
        any(_DATA_BOX_LINE.match(x) for x in body)


def _looks_like_xdatcar(lines: Sequence[str]) -> bool:
    if len(lines) < 8 or "configuration=" not in lines[7]:
        return False
    return (len(lines[1].split()) in (1, 3) and _all_numbers(lines[1].split(), 1)
            and all(_all_numbers(lines[i].split(), 3) for i in (2, 3, 4))
            and _all_ints(lines[6].split(), 1))


def _looks_like_history(lines: Sequence[str]) -> bool:
    return len(lines) >= 3 and _all_ints(lines[1].split(), 3) \
        and lines[2].split()[:1] == ["timestep"]


def _looks_like_config(name: str, lines: Sequence[str]) -> bool:
    """Record 2 levcfg imcon, then (imcon > 0) three cell records, then an
    atom name; only for a file named CONFIG, REVCON or CFGMIN."""
    if not name.upper().startswith(("CONFIG", "REVCON", "CFGMIN")):
        return False
    if len(lines) < 3 or not _all_ints(lines[1].split(), 2):
        return False
    imcon = int(lines[1].split()[1])
    first_atom = 2 if imcon == 0 else 5
    if len(lines) <= first_atom:
        return False
    if imcon != 0 and not all(_all_numbers(lines[i].split(), 3)
                              for i in (2, 3, 4)):
        return False
    tokens = lines[first_atom].split()
    return bool(tokens) and not _is_number(tokens[0])


def _looks_like_xyz(lines: Sequence[str]) -> bool:
    """An atom count alone on line 1, then either an extended XYZ comment
    (Lattice= or Properties=, whatever order its columns take: OVITO writes
    them in the order the user picks) or a plain row 'symbol x y z'."""
    if len(lines) < 3:
        return False
    first, third = lines[0].split(), lines[2].split()
    if len(first) != 1 or not first[0].isdigit() or int(first[0]) < 1:
        return False
    comment = lines[1].lower()
    if "lattice=" in comment or "properties=" in comment:
        return True
    return len(third) >= 4 and all(_is_number(t) for t in third[1:4])


def sniff_md(path) -> str | None:
    """The MD format of a file from its name and first :data:`SNIFF_BYTES`.

    One of :data:`MD_FORMATS`, or None (also for a file that cannot be read or
    decompressed). Gzip files are looked at decompressed, and a leading UTF-8
    byte-order mark is ignored. Only a DL_POLY CONFIG needs its name (CONFIG,
    REVCON or CFGMIN), because its first records resemble other formats;
    everything else is recognised by content. 'extxyz' is returned for any
    XYZ-shaped head, with or without ``Lattice=``: ``read_extxyz`` then states
    what it needs.
    """
    path = Path(path)
    try:
        head = _head(path)
    except (OSError, EOFError, ValueError):
        return None
    return _sniff_head(path.name, head)


def _sniff_head(name: str, head: bytes) -> str | None:
    lines = _lines(head)
    if head and len(head) >= SNIFF_BYTES - len(_BOM) and lines \
            and not head.endswith(b"\n"):
        lines = lines[:-1]             # the last line is cut by the 4 kB window
    text = [x for x in lines if x.strip()]
    if not text:
        return None
    if text[0].lstrip().startswith("ITEM:"):
        return "lammps-dump"
    if _looks_like_data(lines):
        return "lammps-data"
    if _looks_like_xdatcar(lines):
        return "vasp-xdatcar"
    if _looks_like_history(lines):
        return "dlpoly-history"
    if _looks_like_config(name, lines):
        return "dlpoly-config"
    if _looks_like_xyz(lines):
        return "extxyz"
    return None


def is_multiframe_xyz(path) -> bool:
    """True when a second XYZ frame follows the first.

    Reads the first frame, then the next non-blank line, which needs to be an
    atom count followed by a comment line and at least one row, no further: a
    lone number after one frame is not a frame. False for anything that does
    not parse as an XYZ frame or cannot be read.
    """
    try:
        with _open_binary(Path(path)) as handle:
            n = _xyz_count(_strip_bom(_readline(handle)))
            if n is None:
                return False
            for _ in range(n + 1):
                if not _readline(handle):
                    return False
            while True:
                line = _readline(handle)
                if not line:
                    return False
                if line.strip():
                    break
            if _xyz_count(line) is None:
                return False
            return bool(_readline(handle)) and bool(_readline(handle).strip())
    except (OSError, ValueError, EOFError):
        return False


def xyz_has_lattice(path) -> bool:
    """True when the first frame's comment line (line 2) holds ``Lattice=``,
    the key that gives an extended XYZ frame its periodic box."""
    try:
        with _open_binary(Path(path)) as handle:
            _readline(handle)
            return b"lattice=" in _readline(handle).lower()
    except (OSError, ValueError, EOFError):
        return False


_READERS = {
    "lammps-data": read_lammps_data,
    "lammps-dump": read_lammps_dump,
    "extxyz": read_extxyz,
    "vasp-xdatcar": read_xdatcar,
    "dlpoly-config": read_dlpoly_config,
    "dlpoly-history": read_dlpoly_history,
}
_OPTIONS = {
    "lammps-data": {"type_map", "atom_style", "units", "mass_tol_amu"},
    "lammps-dump": {"type_map", "masses_from", "units", "mass_tol_amu"},
    "extxyz": {"type_map", "units", "mass_tol_amu"},
    "vasp-xdatcar": {"type_map"},
    "dlpoly-config": {"type_map", "mass_tol_amu"},
    "dlpoly-history": {"type_map", "mass_tol_amu"},
}


def read_trajectory(path, *, type_map: Mapping | None = None,
                    atom_style: str | None = None, masses_from=None,
                    units: str | None = None,
                    mass_tol_amu: float | None = None) -> Trajectory:
    """Open an MD model or trajectory, whatever its format.

    The format comes from :func:`sniff_md`. ``type_map`` maps LAMMPS type
    numbers, or text labels (DL_POLY atom names, VASP species, dump element
    values), to elements. ``atom_style`` applies to LAMMPS data files,
    ``units`` to LAMMPS files and extended XYZ (LAMMPS's extxyz dump writes in
    the run's unit style), ``masses_from`` to a LAMMPS dump, ``mass_tol_amu``
    (None: :data:`MASS_TOL_AMU`) to LAMMPS, DL_POLY and extended XYZ files
    (the last for a masses column); an option the
    format does not use is refused rather than ignored. The result is a
    :class:`~.md_model.Trajectory` with ``close()`` (it is also a context
    manager); its first note is the load record, and ``describe()`` gives the
    full summary. A gzip file whose stream is damaged is refused with
    UnsupportedFormat naming the file.
    """
    path = _existing(path)
    file_format = _sniff_head(path.name, _head(path))
    if file_format is None:
        raise UnsupportedFormat(
            f"{path.name}: not recognised as an MD model. FACET reads LAMMPS "
            "data and dump files (text or gzip), extended XYZ with Lattice=, "
            "VASP XDATCAR and DL_POLY CONFIG / HISTORY.")
    given = {"type_map": type_map, "atom_style": atom_style,
             "masses_from": masses_from, "units": units,
             "mass_tol_amu": mass_tol_amu}
    extra = sorted(k for k, v in given.items()
                   if v is not None and k not in _OPTIONS[file_format])
    if extra:
        raise ValueError(f"{path.name} is read as {file_format}, which does not "
                         f"use {', '.join(extra)}")
    kwargs = {k: v for k, v in given.items()
              if k in _OPTIONS[file_format] and v is not None}
    return _READERS[file_format](path, **kwargs)
