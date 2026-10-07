"""MD files written as text by CASTEP, GROMACS, XCrySDen, PDB writers and OVITO,
and a series of POSCAR files.

WHY A MODULE OF ITS OWN
-----------------------
:mod:`.md_readers` reads what a LAMMPS, VASP or DL_POLY glass run leaves
behind. Other programs leave other files: an ab initio MD run in CASTEP writes
a ``.md`` file; GROMACS and the tools around it (mdtraj, MDAnalysis) write
``.gro``; XCrySDen's animated XSF holds a trajectory, and pymatgen writes one
structure as XSF; ASE, MDAnalysis and mdtraj write a trajectory as a PDB file
with one MODEL per frame, and CP2K ends each frame with END and no MODEL
record [8]; OVITO exports IMD; pymatgen and OVITO write one POSCAR per frame.
FACET's recognition test (115 files from the real writers) found every one of
them refused with the same generic message, and a multi-model PDB silently cut
to its first model by ``readers.read`` (defects D2 and D8). Each format here
follows :mod:`.md_formats_base`: frames are built with ``frame_from_arrays``,
read on demand from byte ranges found by one index pass, counted in
``Trajectory.skipped`` with a reason when they cannot be read, and given
elements only from the file's own symbols or the user's map; units are
converted with :mod:`scipy.constants` and recorded in ``units_note``. An
error met when a later frame is loaded starts with the name of the file it
comes from (in a series, the file of that frame).

WHAT IS READ
------------
=================  ===============================  ===========================
format             what a frame carries             source of the layout
=================  ===============================  ===========================
CASTEP .md         positions (R), the cell (h),     Quigley, "The .md file"
(also .geom)       velocities (V), time             [1]; CASTEP help [2]
GROMACS .gro       positions, box, velocities,      GROMACS manual, "gro" [3]
(one or more       time and step from the title
frames)            (``t=``, ``step=``)
XSF (ANIMSTEPS,    positions, PRIMVEC once (fixed   XSF specification [4]
or one CRYSTAL)    cell) or per step (variable)
PDB, several       positions, names and elements    wwPDB format 3.3 [5];
MODEL or END       from their columns, CRYST1 per   CP2K source [8]
blocks             model (or the last before it),
                   CP2K's REMARK step and time
IMD (ASCII)        number, type, mass, positions,   OVITO's IMD importer and
                   the cell (#X #Y #Z)              exporter source [6]
POSCAR series      positions, box, one file per     VASP wiki, POSCAR [7]
                   frame (read_series)
=================  ===============================  ===========================

Forces (CASTEP F, XSF force columns), energies, temperature, pressure, stress,
cell velocities, IMD velocity and data columns, PDB occupancies and B-factors,
GROMACS residue names and numbers are not read; each one present is listed in
the notes. A PDB frame is read from the fixed columns of its ATOM / HETATM
records, in file order, not through gemmi's residue hierarchy: writers keep
the last four digits of a residue number, so a model of more than 9999
residues repeats them and gemmi files those atoms under the first residue of
the number (MDAnalysis 2.10 and mdtraj 1.11 files of 10 005 atoms were
refused), and gemmi reads a coordinate field that is not a number as 0.

UNITS
-----
* **CASTEP** writes "in Hartree atomic units" [1]: R and h in Bohr, the time in
  the atomic unit of time (hbar / E_h), V in the atomic unit of velocity.
  :data:`BOHR_ANG`, :data:`AU_TIME_PS` and :data:`AU_VELOCITY_ANG_PER_PS` come
  from ``scipy.constants.physical_constants`` ('Bohr radius', 'atomic unit of
  time', 'atomic unit of velocity'); scipy 1.15 gives CODATA 2022, 1 Bohr =
  0.529177210544 Å. ASE 3.29 writes .md files with CODATA 2002 (0.5291772108
  Å), 5e-10 apart, 8e-9 Å at 16 Å. The h rows are the cell vectors: "Each row
  gives the three Cartesian components of one of the cell vectors" [1]. ASE
  3.29's castep-md writer puts the image index (0, 1, 2, ...) on the time
  line; a file whose time lines are exactly its step positions 0, 1, 2 a.u.
  (2.4e-5 ps apart, a step no MD run takes; a CASTEP-written file, MDANSE's
  PBAnew.md [9], holds 0, 41.34, 82.68 a.u., 1 fs steps) gives no time, and a
  note says why.
* **GROMACS** positions are in nm, velocities in nm/ps, the title's ``t=`` in
  ps [3]; :data:`NM_ANG` is ``scipy.constants.nano / scipy.constants.angstrom``.
  The box line lists v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y), the
  last six optional [3]. Positions are fixed-width fields whose width is the
  distance between the first two decimal points of an atom line ("Upon
  reading, the precision will be inferred from the distance between the
  decimal points" [3]); velocity fields have the same width ("n+5
  positions with n decimal places (n+1 for velocities)" [3]).
* **XSF** "all coordinates are in ANGSTROMS units" [4]; PRIMVEC lists the
  three lattice vectors as rows [4].
* **PDB** coordinates and CRYST1 lengths are in Å [5]. The box is CRYST1's
  standard orthogonalisation (a along x, b in the xy plane [5]), built with
  ``gemmi.UnitCell`` once the record's values are checked (gemmi turns zeros
  into a 1 Å cube); SCALEn records that describe another frame are refused,
  each element compared with the inverse of the box within the rounding both
  records carry (:func:`_pdb_scale_check`). CP2K's REMARK gives the time in
  fs [8], converted with :data:`FS_PS` (``scipy.constants.femto /
  scipy.constants.pico``).
* **IMD** states no unit. Positions are kept as written and read as Å, which is
  what OVITO writes from a LAMMPS metal or real model. The header's ``#X``,
  ``#Y``, ``#Z`` lines are the cell vectors a, b, c (OVITO's importer reads
  ``#X`` into the first cell vector [6]); the file holds no origin (OVITO's
  exporter writes none [6]), so the origin is (0, 0, 0). Velocity columns
  are not read: the file gives them no unit.
* **POSCAR**: Å, with VASP's scale factor (negative: the volume; three: per
  axis) applied by :func:`.md_readers._vasp_header`, the parser XDATCAR
  already uses.

ELEMENTS
--------
CASTEP species, GROMACS atom names, XSF symbols, PDB atom names and POSCAR
species are labels, resolved by :func:`.md_readers._resolve_labels`: a label is
an element only when written as files write symbols ('OW', 'HW1', 'CA' and
'ho' are not), and a ``type_map`` keyed by the labels names the rest. XSF
atomic numbers name their element. The PDB element columns (77-78) hold an
element symbol by definition, right-justified [5]; writers put it in upper case
(ASE and MDAnalysis write 'SI', 'NA') or not (CP2K writes 'Si'), so it is read
as a symbol in any case. When they are blank the atom names are used as
labels. IMD types are numbers: the user's ``type_map`` or, when the file has a
mass column, the mass of each type matched to the standard atomic weights. A
CASTEP or GROMACS file whose first and last frames hold velocities that are
all exactly 0 (ASE 3.29's castep-md and gromacs writers fill the columns with
zeros for a model without velocities) stores no velocities, and says so; a run
started from rest, zeros in its first frame only, keeps them.

NOTHING IS DROPPED SILENTLY
---------------------------
Every frame that cannot be read has a file position in ``skipped`` and a
reason: a last frame cut while the file was being written (fewer rows, or a
last line without a line ending that holds fewer values or fewer decimals than
the line above; pymatgen ends its XSF without a line ending, and such a whole
last line is read), a frame with another atom count, a header or count line
that does not parse. The model's atom count is the count most frames hold, so
an odd first frame does not set it. A CASTEP block that lost the cell lines
between k steps (their (species, number) pairs repeat), and a PDB block that
lost the ENDMDL and MODEL (or END) records between k frames (k times the
model's atom records), take k positions. PDB atom records outside any MODEL
block are read as a frame when they hold the model's atom count, and take a
position otherwise. A GROMACS count line that breaks the walk, or a count
larger than the frame (the walk meets a line that is not an atom line), costs
that frame only: the walk resynchronises at the next frame. An XSF step is
placed by the number after PRIMCOORD, so a lost 'PRIMCOORD i' line is that
step's position (with the lines the section before it holds in excess), and a
variable cell needs 'PRIMVEC i' before 'PRIMCOORD i'; steps that ANIMSTEPS
announces beyond the file take a position each, up to
:data:`_XSF_LISTED_MISSING`, and a count in a note beyond. A POSCAR or IMD
file of a series that holds other species, counts or columns than the first is
skipped with the difference.

ROUTING
-------
No format here claims a file name: read_trajectory recognises them by content
(a sniff), so a CIF or POSCAR saved as .gro, .axsf or .imd stays with
``readers.read``'s crystal sniffing. Each ``sniff`` looks at the first 4 kB:
the CASTEP header (``BEGIN header`` ... ``END header``, then ``<--`` markers;
a Markdown ``.md`` file has none), a GROMACS atom count and fixed-width atom
line, ``ANIMSTEPS`` or ``CRYSTAL`` with PRIMVEC and PRIMCOORD for XSF (no
crystal reader of FACET opens XSF, so one structure is a one-frame model),
``#F`` for IMD. The PDB sniff decides as ``readers.read`` decides for a .pdb
file, with :func:`.md_readers.pdb_model_count` (a second MODEL, or atoms after
an END record; ' END' counts), which reads lines until it finds a second
structure: a one-frame PDB is a crystal structure, which ``readers.read``
opens, and a PDB that readers.read routes here is never refused by the sniff.
A single POSCAR is a crystal structure too. The POSCAR format's sniff
(:func:`sniff_poscar_series`) accepts POSCAR content (VASP 5, or VASP 4 with
the element symbols on the title line) only under the names readers.read gives
its POSCAR reader (``.vasp``, ``.poscar``, ``.contcar``, a name starting with
POSCAR or CONTCAR, a '.gz' after any of them set aside), for which
readers.read consults no format module: read_trajectory then reads one file
as a frame and a list, pattern or directory of them as a series
(``read_series``), while readers.read keeps reading each one as a crystal, as
it reads the same content under any other name. For such a name it reads the
first 9 lines when the 4 kB window holds fewer (ASE writes the symbols of a
model in random order on the title, name and count lines). A directory's
CONTCAR and POSCAR (a VASP run's end and start, in that name order) are
refused as a series, naming the run's XDATCAR.

TIMINGS
-------
Measured on this machine (Windows 11, i5-13420H, Python 3.11, numpy 2.4.6,
gemmi 0.7.1; other sessions kept the CPU 40-70 % busy), on 10 000 atoms x 100
frames per file written on Windows (CRLF line endings) by the real writer:
ASE 3.29, mdtraj 1.11 (3 decimals), OVITO 3.16. Two runs of three repeats;
open = index pass + frame 0 + the load record, then every frame in order:

==============================  =========  ===========  ===========
file                            size       open         per frame
==============================  =========  ===========  ===========
CASTEP .md (T, h, R, V; ASE)    216 MB     0.77-0.93 s  64-79 ms
GROMACS .gro (mdtraj)           46.0 MB    0.04-0.06 s  18-24 ms
animated XSF (ASE)              68.0 MB    0.30-0.34 s  25-33 ms
PDB, 100 MODEL (ASE)            82.0 MB    0.39-0.53 s  36-42 ms
IMD, 1 frame (OVITO)            0.49 MB    0.06-0.07 s  29-43 ms
POSCAR series, 100 files (ASE)  131 MB     1.06-1.24 s  38-56 ms
==============================  =========  ===========  ===========

Three choices came from these measurements. The .gro index first walked every
atom line (1.2-1.4 s to open); it now reaches a frame's box line with one seek
when the frame's atom lines have one length (:func:`_gro_skip_rows`). The
CASTEP step parse first matched regular expressions over the step (0.18 s per
step); it now splits the step into lines by their '<--' tag (0.05 s on the
same step). The PDB index first scanned the file three times (MODEL, END,
CRYST1; 0.77-0.93 s to open, measured beside the present one); one pass that
also counts the atom records (:func:`_pdb_index`) opens it in 0.38-0.62 s. A
PDB of two 420 000-atom frames (69 MB) opens in 5.6 s and reads a frame in
1.8 s.

REFERENCES
----------
[1] D. Quigley, CASTEP molecular dynamics documentation, "The .md file"
    (2005-05-10), https://www.tcm.phy.cam.ac.uk/castep/MD/node13.html
    (header, the time line,
    the E, T, P, h, hv, S, R, V and F records, Hartree atomic units, the
    Fortran formats)
[2] CASTEP help, "CASTEP file formats - task-specific files" (Materials
    Studio 8.0), https://www.tcm.phy.cam.ac.uk/castep/documentation/WebHelp/
    content/modules/castep/expcastepfiletask.htm (.geom, .md and .ts share the
    layout; values in atomic units)
[3] GROMACS reference manual, "File formats: gro",
    https://manual.gromacs.org/current/reference-manual/file-formats.html#gro
[4] XCrySDen, "The XSF Format Specification",
    http://www.xcrysden.org/doc/XSF.html (units, PRIMVEC, PRIMCOORD, animated
    XSF with fixed and variable cell, comment lines)
[5] wwPDB, "Atomic Coordinate Entry Format Version 3.3": "Crystallographic and
    Coordinate Transformation Section" (CRYST1, SCALEn),
    https://www.wwpdb.org/documentation/file-format-content/format33/sect8.html,
    and "Coordinate Section" (MODEL, ATOM, ENDMDL),
    https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
[6] OVITO source, src/ovito/particles/import/imd/IMDImporter.cpp and
    src/ovito/particles/export/imd/IMDExporter.cpp,
    https://gitlab.com/stuko/ovito (master, read 2026-10-07)
[7] VASP wiki, "POSCAR", https://www.vasp.at/wiki/index.php/POSCAR
[8] CP2K source, https://github.com/cp2k/cp2k (master, read 2026-10-07):
    src/motion_utils.F (write_trajectory, dump_pdb: TITLE and AUTHOR once,
    the title 'Step <it>, time = <t>, E = <etot>' per frame, FMT
    "(A,I0,A,F0.3,A,F0.10)"), src/motion/md_energies.F (the time passed as
    ``time*femtoseconds``), src/particle_methods.F (REMARK "(A6,T11,A)",
    CRYST1 "(A6,3F9.3,3F7.2)", ATOM columns, 'END' after each frame)
[9] MDANSE test data, MDANSE/Tests/UnitTests/Data/PBAnew.md (written by
    CASTEP in Materials Studio), https://github.com/ISISNeutronMuon/MDANSE
"""
from __future__ import annotations

import bisect
import gzip
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import constants as _constants

from . import md_model
from .md_formats_base import FormatSpec
from .md_model import (NO_TIMESTEP, Frame, FrameError, Trajectory,
                       frame_from_arrays, validate_symbol)
from .readers import UnsupportedFormat

__all__ = [
    "BOHR_ANG", "AU_TIME_PS", "AU_VELOCITY_ANG_PER_PS", "NM_ANG", "FS_PS",
    "FORMATS",
    "read_castep_md", "read_gro", "read_xsf", "read_pdb_trajectory",
    "read_imd", "read_imd_series", "read_poscar_series", "read_poscar_file",
    "sniff_castep_md", "sniff_gro", "sniff_xsf", "sniff_pdb_trajectory",
    "sniff_imd", "sniff_poscar", "sniff_poscar_series", "is_multiframe_pdb",
]

# ---------------------------------------------------------------------------
# unit conversions: definitions, from scipy.constants
# ---------------------------------------------------------------------------

_PC = _constants.physical_constants
# 1 Bohr in Å, 1 atomic unit of time (hbar / E_h) in ps, and 1 atomic unit of
# velocity (a0 E_h / hbar) in Å/ps: CASTEP's .md units [1].
BOHR_ANG: float = _PC["Bohr radius"][0] / _constants.angstrom
AU_TIME_PS: float = _PC["atomic unit of time"][0] / _constants.pico
AU_VELOCITY_ANG_PER_PS: float = (_PC["atomic unit of velocity"][0]
                                 * _constants.pico / _constants.angstrom)
# 1 nm in Å: GROMACS lengths [3].
NM_ANG: float = _constants.nano / _constants.angstrom

_CASTEP_UNITS_NOTE = (
    "CASTEP writes Hartree atomic units [Quigley, 'The .md file']: positions "
    f"and cell from Bohr with 1 Bohr = {BOHR_ANG:.12g} Å, times from the "
    f"atomic unit of time with 1 a.u. = {AU_TIME_PS:.12g} ps, velocities from "
    f"the atomic unit of velocity with 1 a.u. = {AU_VELOCITY_ANG_PER_PS:.12g} "
    "Å/ps (scipy.constants)")
_GRO_UNITS_NOTE = (
    "GROMACS units [GROMACS manual, gro]: positions and box from nm with 1 nm "
    f"= {NM_ANG:g} Å, velocities from nm/ps to Å/ps (x {NM_ANG:g}), the "
    "title's t= in ps")

# The note for velocity columns that hold only zeros in the first and the
# last frame. A run started from rest has zeros in its first frame only, and
# its velocities are kept.
_ZERO_VELOCITY_NOTE = (
    "every velocity in the first and the last frame is exactly 0, so the "
    "velocity columns are taken to hold no velocities (ASE 3.29's castep-md "
    "and gromacs writers fill them with zeros for a model without "
    "velocities); no velocities are stored for any frame")
_ZERO_VELOCITY_FRAME_NOTE = (
    "this frame holds non-zero velocities, which are not stored because every "
    "velocity in the first and the last frame is 0")


def _mr():
    """:mod:`.md_readers`, imported when first used: it consults this module
    for its formats, so neither imports the other at module level."""
    from . import md_readers
    return md_readers


def _named(path: Path, error: Exception) -> str:
    """The error's text, starting with the file name."""
    text = str(error)
    return text if text.startswith(path.name) else f"{path.name}: {text}"


def _floats(tokens, what: str) -> np.ndarray:
    """float64 from text or bytes tokens; ValueError naming ``what`` for one
    that is not a finite number."""
    array = np.asarray(tokens)
    try:
        values = array.astype(np.float64)
    except ValueError:
        odd = next((t for t in array.ravel()
                    if not _is_number(t)), array.ravel()[0])
        if isinstance(odd, bytes):
            odd = odd.decode("utf-8", "replace")
        raise ValueError(f"{what} holds {str(odd).strip()!r}, which is not a "
                         "number") from None
    if not np.isfinite(values).all():
        raise ValueError(f"{what} holds a value that is not finite")
    return values


def _is_number(token) -> bool:
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def _decode(token) -> str:
    return token.decode("utf-8", "replace") if isinstance(token, bytes) \
        else str(token)


def _labels_to_elements(path: Path, labels: np.ndarray, type_map, *,
                        what: str, reduce=None, masses=None):
    """Per-atom elements from per-atom text labels.

    Returns (elements, listed map, source, notes); a label that is not an
    element as files write them, and that the map does not name, raises
    ValueError naming the file and asking for a map.
    """
    mr = _mr()
    user_types, user_labels = mr._normalise_type_map(type_map)
    unique, inverse = np.unique(labels, return_inverse=True)
    try:
        per_label, listed, source, notes = mr._resolve_labels(
            [str(x) for x in unique], user_labels, reduce=reduce, what=what,
            masses=masses)
    except ValueError as error:
        raise ValueError(_named(path, error)) from None
    symbols = np.array([per_label[str(x)] for x in unique], dtype="<U2")
    notes = list(notes) + mr._numeric_entries_note(user_types, what)
    return symbols[inverse.reshape(-1)], listed, source, notes


# ---------------------------------------------------------------------------
# trajectories: frames from byte ranges of one file, or one file per frame
# ---------------------------------------------------------------------------

class _Base(Trajectory):
    """``close()``, the context manager and ``ids_track_atoms`` (True: every
    format here lists the atoms in the same order in every frame), as the
    readers of :mod:`.md_readers` give them.

    Subclasses read a frame in ``_read_frame``; ``_load`` starts any error
    with the name of the file the frame comes from, because
    ``Trajectory.frame`` adds only the frame and its file position, and a
    series reads each frame from a file of its own.
    """

    ids_track_atoms: bool = True
    periodic: tuple[bool, bool, bool] | None = None

    def _read_frame(self, k: int) -> Frame:      # pragma: no cover - abstract
        raise NotImplementedError

    def _frame_path(self, k: int) -> Path:
        return Path(self.source_path)

    def _load(self, k: int) -> Frame:
        try:
            return self._read_frame(k)
        except ValueError as error:
            raise ValueError(_named(self._frame_path(k), error)) from error

    def close(self) -> None:
        """Release the open file, if any. Frames already loaded stay valid."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _RangeTrajectory(_Base):
    """Frames read on demand from byte ranges of one file, plain or gzip.

    A plain file is opened for each frame; a gzip file keeps one stream,
    shared under a lock, because reaching a frame means decompressing up to it
    (the same scheme as :class:`.md_readers._FileTrajectory`).
    """

    def __init__(self, path: Path, ranges: Sequence[tuple[int, int]],
                 **kwargs) -> None:
        super().__init__(source_path=path, **kwargs)
        self._path = Path(path)
        self._ranges = [(int(a), int(b)) for a, b in ranges]
        self._gzip = _mr()._is_gzip(self._path)
        self._handle = None
        self._lock = threading.Lock()

    def _frame_path(self, k: int) -> Path:
        return self._path

    def _bytes(self, k: int) -> bytes:
        mr = _mr()
        start, end = self._ranges[k]
        cut = False
        if not self._gzip:
            with open(self._path, "rb") as handle:
                handle.seek(start)
                data = handle.read(end - start)
        else:
            with self._lock:
                if self._handle is None:
                    self._handle = gzip.open(self._path, "rb")
                try:
                    self._handle.seek(start)
                except EOFError:
                    data, cut = b"", True
                except mr._GZIP_DAMAGE as error:
                    raise mr._damaged(self._handle, error) from None
                else:
                    data, _, cut = mr._read_some(self._handle, end - start)
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


class _SeriesTrajectory(_Base):
    """One file per frame, each read when its frame is asked for.

    ``source_path`` is the first readable file; ``source_paths`` lists every
    file of the series in the order given, readable or skipped.
    """

    def __init__(self, paths: Sequence[Path], all_paths: Sequence[Path],
                 **kwargs) -> None:
        super().__init__(source_path=paths[0], **kwargs)
        self._paths = [Path(p) for p in paths]
        self.source_paths = tuple(str(Path(p).resolve()) for p in all_paths)

    def _frame_path(self, k: int) -> Path:
        return self._paths[k]


def _finish(trajectory: Trajectory) -> Trajectory:
    """The load record first in the notes (:func:`.md_readers._finish`)."""
    return _mr()._finish(trajectory)


def _frame_count(counts: Sequence[int]) -> int:
    """The atom count of the model: the count most blocks hold. On a tie, the
    smallest tied count that divides the others, because a block that lost
    the delimiter between two frames holds twice a frame's atoms; else the
    first block's (:func:`.md_readers._majority_count`)."""
    tally = Counter(counts)
    best = max(tally.values())
    tied = sorted(c for c in tally if tally[c] == best)
    if len(tied) > 1:
        for c in tied:
            if c > 0 and all(t % c == 0 for t in tied):
                return c
    return _mr()._majority_count(counts)


def _decimals(token: bytes) -> int:
    """Characters after the decimal point of a number as written (its
    decimals, and its exponent when it has one), -1 without a point: a
    number cut short has fewer than the numbers its writer wrote with the
    same format."""
    point = token.find(b".")
    return -1 if point < 0 else len(token) - point - 1


def _whole_last_row(rows: Sequence[Sequence[bytes]], width: int) -> bool:
    """Whether the last row of a table, the file's last line written without
    a line ending, holds every value: ``width`` tokens, the last written with
    as many characters after its decimal point as the same column of the row
    above (or, in a one-row table, as the numeric column before it). A copy
    cut inside the last number fails this; a writer that ends its last line
    without a line ending (pymatgen's XSF) passes it."""
    last = rows[-1]
    if len(last) != width or width < 2:
        return False
    if len(rows) >= 2 and len(rows[-2]) == width:
        reference = rows[-2][-1]
    else:
        reference = last[-2]
    shape = _decimals(last[-1])
    return shape >= 1 and shape == _decimals(reference)


def _series_paths(paths, what: str) -> list[Path]:
    """The paths of a series as a list of existing files, in the order given."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    try:
        out = [Path(p) for p in paths]
    except TypeError:
        raise ValueError(f"a {what} series needs a list of file paths, not "
                         f"{type(paths).__name__}") from None
    if not out:
        raise ValueError(f"a {what} series needs at least one file; none was "
                         "given")
    return [_mr()._existing(p) for p in out]


_NUMBER_IN_NAME = re.compile(r"(\d+)(?!.*\d)")


def _name_order_note(paths: Sequence[Path]) -> str | None:
    """A note when the last number in the file names does not increase along
    the order given: a shell's glob puts frame_10 before frame_2."""
    numbers = []
    for p in paths:
        match = _NUMBER_IN_NAME.search(p.name)
        if match is None:
            return None
        numbers.append(int(match.group(1)))
    breaks = [f"{a.name} before {b.name}" for a, b, x, y in
              zip(paths, paths[1:], numbers, numbers[1:]) if y <= x]
    if not breaks:
        return None
    return ("the numbers in the file names do not increase in the order the "
            "files were given (" + "; ".join(breaks[:3])
            + (f"; {len(breaks) - 3} more" if len(breaks) > 3 else "")
            + "); the frames are kept in the order given, so sort the paths "
            "by that number to read them in time order")


# ---------------------------------------------------------------------------
# CASTEP .md / .geom
# ---------------------------------------------------------------------------

_CASTEP_MARKER = re.compile(r"<--\s*(\S+)\s*$")
_CASTEP_NOT_READ = {"E": "energies (E)", "T": "temperature (T)",
                    "P": "pressure (P)", "hv": "cell velocities (hv)",
                    "S": "stress (S)", "F": "forces (F)",
                    "c": "convergence flags (c)"}


def _castep_species(token: str) -> tuple[str, str | None]:
    """'O:a' -> ('O', note): a CASTEP species with a label after the colon."""
    if ":" in token:
        head = token.split(":", 1)[0]
        return head, (f"CASTEP species {token!r} read as {head} (the part after "
                      "':' labels the species)")
    return token, None


@dataclass
class _CastepBlock:
    position: int
    start: int
    end: int
    line: int
    box_ang: np.ndarray
    time_ps: float | None
    timestep: int | None


@dataclass
class _CastepAtoms:
    labels: np.ndarray          # (n,) species as written
    numbers: np.ndarray         # (n,) the number within the species, as bytes
    cart_bohr: np.ndarray
    vel_au: np.ndarray | None
    box_bohr: np.ndarray
    tags: frozenset


def _castep_records(data: bytes) -> dict[bytes, list[bytes]]:
    """The lines of a step by their tag, each without its '<-- X' marker."""
    out: dict[bytes, list[bytes]] = {}
    for line in data.splitlines():
        cut = line.rfind(b"<--")
        if cut >= 0:
            out.setdefault(line[cut + 3:].strip(), []).append(line[:cut])
    return out


def _castep_table(rows: Sequence[bytes], columns: int, what: str) -> np.ndarray:
    """(rows, columns) tokens; ValueError naming the first row that holds
    another number of values."""
    tokens = b" ".join(rows).split()
    if len(tokens) != columns * len(rows):
        odd = next(r for r in rows if len(r.split()) != columns)
        raise ValueError(f"a {what} line ({_decode(odd).strip()[:60]!r}) holds "
                         f"{len(odd.split())} values where it has {columns}")
    return np.array(tokens).reshape(len(rows), columns)


def _castep_atoms(data: bytes) -> _CastepAtoms:
    """The h, R and V records of one block; ValueError saying what is
    missing or does not parse."""
    records = _castep_records(data)
    h = records.get(b"h", [])
    if len(h) != 3:
        raise ValueError(f"{len(h)} cell lines (<-- h) where a step holds 3")
    box = _floats(_castep_table(h, 3, "cell (<-- h)"), "a cell line (<-- h)")
    rows = records.get(b"R", [])
    if not rows:
        raise ValueError("no position lines (<-- R)")
    table = _castep_table(rows, 5, "position (<-- R)")
    cart = _floats(table[:, 2:5], "a position line (<-- R)")
    vel = None
    if b"V" in records:
        v_rows = records[b"V"]
        if len(v_rows) != len(rows):
            raise ValueError(f"{len(v_rows)} velocity lines (<-- V) for "
                             f"{len(rows)} atoms")
        v_table = _castep_table(v_rows, 5, "velocity (<-- V)")
        if not np.array_equal(v_table[:, :2], table[:, :2]):
            raise ValueError("the velocity lines (<-- V) list the atoms in "
                             "another order than the position lines (<-- R)")
        vel = _floats(v_table[:, 2:5], "a velocity line (<-- V)")
    tags = frozenset(_decode(t) for t in records)
    return _CastepAtoms(table[:, 0].astype(str), table[:, 1], cart, vel, box,
                        tags)


def _castep_preamble(before: Sequence[str]) -> tuple[float | None, int | None]:
    """(time in a.u., iteration) from the lines before a step's first cell
    line: the time is the first line of an .md step, alone on its line [1];
    a .geom step starts with its iteration number and '<-- c'."""
    for line in reversed(before):
        text = line.strip()
        if not text:
            return None, None
        marker = _CASTEP_MARKER.search(text)
        if marker is None:
            tokens = text.split()
            if len(tokens) == 1 and _is_number(tokens[0]):
                return float(tokens[0]), None
            return None, None
        if marker.group(1) == "c":
            first = text.split()[0]
            return None, int(first) if re.fullmatch(r"[+-]?\d+", first) else None
    return None, None


class _CastepTrajectory(_RangeTrajectory):
    def __init__(self, path, blocks, *, elements, labels, numbers,
                 keep_velocities, **kwargs) -> None:
        super().__init__(path, [(b.start, b.end) for b in blocks], **kwargs)
        self._blocks = list(blocks)
        self._elements = elements
        self._labels = labels
        self._numbers = numbers
        self._keep_velocities = keep_velocities

    def _read_frame(self, k: int) -> Frame:
        block = self._blocks[k]
        atoms = _castep_atoms(self._bytes(k))
        if atoms.labels.shape != self._labels.shape:
            raise ValueError(f"{atoms.labels.size} atoms; the model has "
                             f"{self._labels.size}")
        if not (np.array_equal(atoms.labels, self._labels)
                and np.array_equal(atoms.numbers, self._numbers)):
            raise ValueError("the position lines (<-- R) list other species or "
                             "species numbers, or another order, than frame 0's")
        notes = []
        vel = None
        if atoms.vel_au is not None:
            if self._keep_velocities:
                vel = atoms.vel_au * AU_VELOCITY_ANG_PER_PS
            elif np.any(atoms.vel_au):
                notes.append(_ZERO_VELOCITY_FRAME_NOTE)
        return frame_from_arrays(
            self._elements, atoms.cart_bohr * BOHR_ANG, box_ang=block.box_ang,
            timestep=block.timestep, time_ps=block.time_ps,
            vel_ang_per_ps=vel, notes=notes)


def _castep_steps_in(atoms: _CastepAtoms) -> int:
    """How many steps' position lines one block holds. CASTEP numbers the
    atoms of each species 1, 2, ... within a step [1], so a (species,
    number) pair written k times is k steps run together: the cell lines
    (<-- h) that start each of the others are missing."""
    pairs = Counter(zip(atoms.labels.tolist(), atoms.numbers.tolist()))
    return max(pairs.values())


def _is_image_index(times_au: Sequence[float | None],
                    positions: Sequence[int]) -> bool:
    """Whether the time lines are 0, 1, 2, ...: the file position of each
    step, which ASE 3.29's castep-md writer writes there (``_write_time``
    writes the image index). One atomic unit of time is 2.4e-5 ps, a step no
    MD run takes, and CASTEP's own file (MDANSE's PBAnew.md) holds 0,
    41.34, 82.68 a.u. (1 fs steps)."""
    return len(times_au) >= 2 and all(
        t is not None and t == float(p) for t, p in zip(times_au, positions))


def read_castep_md(path, *, type_map: Mapping | None = None) -> Trajectory:
    """A CASTEP ``.md`` (or ``.geom``) trajectory [1].

    Each step is a block: the time (a .geom step: the iteration and '<-- c'),
    optional E, T and P records, three h records (the cell vectors as rows),
    optional hv and S, then one R, V and F record per atom, labelled with the
    species and the atom's number within it. Atoms keep the file's order,
    numbered 0 .. N-1. Lengths, times and velocities are converted from atomic
    units (:data:`BOHR_ANG`, :data:`AU_TIME_PS`,
    :data:`AU_VELOCITY_ANG_PER_PS`); time lines that are the step positions
    0, 1, 2 (ASE's image index) give no time (:func:`_is_image_index`).
    ``type_map`` maps species labels to elements.

    A block runs from one step's cell lines to the next one's. The model's
    atom count is the one most blocks hold; a block whose (species, number)
    pairs repeat k times holds k steps whose cell lines between them are
    lost, and takes k skipped positions. The file format is 'castep-md' for
    a .geom file too (the notes say it is one).
    """
    mr = _mr()
    path = mr._existing(path)
    head = mr._head(path)
    if not sniff_castep_md(head, path):
        raise UnsupportedFormat(
            f"{path.name}: no CASTEP header ('BEGIN header' ... 'END header' "
            "followed by '<--' records), so not a CASTEP .md or .geom file. "
            "FACET reads the .md file CASTEP writes for an MD run")
    scan = mr._scan(path, b"<-- h", line_start=False, before_lines=6,
                    after_lines=2)
    starts = [m for m in scan.marks
              if m.text.rstrip().endswith("<-- h")
              and not (m.before and m.before[-1].rstrip().endswith("<-- h"))]
    if not starts:
        raise UnsupportedFormat(
            f"{path.name}: the CASTEP header is followed by no cell record "
            "('<-- h'), so the file holds no step; FACET reads the .md file "
            "of an MD run, which writes the cell at every step")
    geom = path.suffix.lower() == ".geom" or any(
        line.rstrip().endswith("<-- c") for line in starts[0].before)
    skipped: dict[int, str] = {}
    blocks: list[_CastepBlock] = []
    n_blocks = len(starts)
    last = n_blocks - 1
    ends = [m.offset for m in starts[1:]] + [scan.size]
    gaps = [b.line - a.line for a, b in zip(starts, starts[1:])]
    usual = mr._majority_count(gaps) if gaps else None
    boxes: dict[int, np.ndarray] = {}
    box_errors: dict[int, str] = {}
    for index, mark in enumerate(starts):
        try:
            rows = [mark.text] + list(mark.after[:2])
            if len(rows) < 3 or not all(r.rstrip().endswith("<-- h")
                                        for r in rows):
                raise ValueError("fewer than 3 cell lines (<-- h)")
            box = _floats([r.split()[:3] for r in rows],
                          "a cell line (<-- h)") * BOHR_ANG
            boxes[index] = md_model._checked_box(box)
        except ValueError as error:
            box_errors[index] = str(error)

    # A block runs from one step's cell lines to the next one's. The first
    # and the last block are parsed (the last may be cut), and so is every
    # block whose line count differs from most blocks' (it may hold another
    # atom count, or several steps whose cell lines are lost). The other
    # blocks have the usual length and are taken to hold the atom count of
    # the first such block that parses as one step.
    parsed: dict[int, _CastepAtoms | str] = {}

    def parse(i: int) -> _CastepAtoms | str:
        if i not in parsed:
            try:
                parsed[i] = _castep_atoms(_read_range(path, starts[i].offset,
                                                      ends[i]))
            except ValueError as error:
                parsed[i] = str(error)
        return parsed[i]

    for i in [0, last] + [i for i, g in enumerate(gaps) if g != usual]:
        if i not in box_errors:
            parse(i)
    reference = None
    for i, gap in enumerate(gaps):
        if gap == usual and i not in box_errors:
            atoms = parse(i)
            if isinstance(atoms, _CastepAtoms) and _castep_steps_in(atoms) == 1:
                reference = i
                break
    if reference is None:
        for i in range(n_blocks):
            if i not in box_errors:
                parse(i)

    def steps_in(i: int) -> int:
        atoms = parsed.get(i)
        return _castep_steps_in(atoms) if isinstance(atoms, _CastepAtoms) else 1

    def count(i: int) -> int:
        atoms = parsed.get(i, parsed.get(reference))
        return int(atoms.labels.size)

    readable = [i for i in range(n_blocks) if i not in box_errors
                and not isinstance(parsed.get(i), str)]
    voters = [i for i in readable if steps_in(i) == 1]
    if len(voters) > 1 and last in voters:
        voters.remove(last)                 # the last block may be cut
    if voters:
        n0 = _frame_count([count(i) for i in voters])
    elif readable:
        n0 = count(readable[0]) // steps_in(readable[0])
    else:
        n0 = 0
    times_au: list[float | None] = []
    kept: list[int] = []
    position = 0
    for index, mark in enumerate(starts):
        is_last = index == last
        if index in box_errors:
            skipped[position] = f"unreadable: its cell: {box_errors[index]}"
            position += 1
            continue
        atoms = parsed.get(index)
        if isinstance(atoms, str):
            skipped[position] = ("truncated: " if is_last else
                                 "unreadable: ") + atoms
            position += 1
            continue
        size, k = count(index), steps_in(index)
        if k > 1 and size == k * n0:
            for j in range(k):
                skipped[position + j] = (
                    f"merged: the cell lines (<-- h) that start a step are "
                    f"missing between {k} steps, so one block (from file line "
                    f"{mark.line + 1}) holds their {size} position lines; "
                    "none of them is read")
            position += k
            continue
        if size != n0:
            skipped[position] = (
                f"truncated: the file ends inside this step, after {size} of "
                f"{n0} position lines" if is_last and size < n0 else
                f"{size} atoms, where frame 0 holds {n0}"
                if count(readable[0]) == n0 and steps_in(readable[0]) == 1
                else f"{size} atoms, where most steps hold {n0}")
            position += 1
            continue
        if is_last and not scan.last_line_ended:
            skipped[position] = (
                "truncated: the file does not end with a line ending, so "
                "the last value of its last line may be cut (a file "
                "copied while it was being written)")
            position += 1
            continue
        time_au, iteration = _castep_preamble(mark.before)
        times_au.append(time_au)
        kept.append(index)
        blocks.append(_CastepBlock(
            position, mark.offset, ends[index], mark.line, boxes[index],
            None if time_au is None else time_au * AU_TIME_PS, iteration))
        position += 1
    if not blocks:
        raise ValueError(f"{path.name} holds no readable step: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))

    atoms0 = parse(kept[0])
    atoms_last = parse(kept[-1])
    elements, listed, source, notes = _labels_to_elements(
        path, atoms0.labels, type_map, what="species", reduce=_castep_species)
    if not geom and _is_image_index(times_au, [b.position for b in blocks]):
        for block in blocks:
            block.time_ps = None
        notes.append(
            "the time lines read 0, 1, 2, ..., the file position of each "
            f"step, in atomic units {AU_TIME_PS:.6g} ps apart: ASE's "
            "castep-md writer writes the image index there, so they are "
            "not used as times")
    # A run started from rest has zero velocities in its first step only;
    # zeros in the first and the last step are a writer's filler.
    keep_velocities = atoms0.vel_au is not None and bool(
        np.any(atoms0.vel_au) or (isinstance(atoms_last, _CastepAtoms)
                                  and atoms_last.vel_au is not None
                                  and np.any(atoms_last.vel_au)))
    if atoms0.vel_au is not None and not keep_velocities:
        notes.append(_ZERO_VELOCITY_NOTE)
    # Records of the first step: those after its cell lines, and E, T, P
    # (or a .geom step's c) before them, up to the blank line before the step.
    tags = set(atoms0.tags)
    for line in reversed(starts[0].before):
        if not line.strip():
            break
        marker = _CASTEP_MARKER.search(line.strip())
        if marker is not None:
            tags.add(marker.group(1))
    absent = sorted(t for t in _CASTEP_NOT_READ if t in tags)
    if absent:
        notes.append("not read: " + ", ".join(_CASTEP_NOT_READ[t]
                                              for t in absent))
    notes.append("atoms are numbered 0 .. N-1 in the order of the file; the "
                 "cell lines (<-- h) are the cell vectors as rows, with the "
                 "origin at (0, 0, 0)")
    times = [b.time_ps for b in blocks]
    if geom:
        notes.append("a .geom file (CASTEP's geometry-optimisation layout, "
                     "read as castep-md): the steps are geometry-optimisation "
                     "iterations, kept as timesteps; the file holds no time")
    elif any(t is None for t in times_au):
        notes.append(f"{sum(t is None for t in times_au)} step(s) have no "
                     "time line")
    if scan.gzip_cut:
        notes.append(mr._GZIP_CUT_NOTE)
    box_varies = any(not np.array_equal(b.box_ang, blocks[0].box_ang)
                     for b in blocks[1:])
    return _finish(_CastepTrajectory(
        path, blocks, elements=elements, labels=atoms0.labels,
        numbers=atoms0.numbers, keep_velocities=keep_velocities,
        file_format="castep-md",
        n_atoms=int(atoms0.labels.size), n_frames=len(blocks),
        type_map_source=source,
        timesteps=[NO_TIMESTEP if b.timestep is None else b.timestep
                   for b in blocks],
        times_ps=None if all(t is None for t in times) else
        [np.nan if t is None else t for t in times],
        type_map=listed, skipped=skipped, notes=notes,
        units_note=_CASTEP_UNITS_NOTE, box_varies=box_varies))


def _read_range(path: Path, start: int, end: int) -> bytes:
    """Bytes start .. end of a plain or gzip file."""
    mr = _mr()
    with mr._open_binary(path) as handle:
        try:
            handle.seek(start)
        except EOFError:            # a gzip stream cut before ``start``
            return b""
        except mr._GZIP_DAMAGE as error:
            raise mr._damaged(handle, error) from None
        data, _, _ = mr._read_some(handle, end - start)
    return data


def sniff_castep_md(head: bytes, path: Path) -> bool:
    """A CASTEP .md / .geom head: first text line 'BEGIN header', an
    'END header' line, and a '<--' record after it."""
    try:
        lines = [x.strip() for x in head.decode("utf-8", "replace").splitlines()
                 if x.strip()]
        if not lines or lines[0] != "BEGIN header" or "END header" not in lines:
            return False
        after = lines[lines.index("END header") + 1:]
        return any(_CASTEP_MARKER.search(x) for x in after)
    except Exception:           # noqa: BLE001 - a sniffer never raises
        return False


# ---------------------------------------------------------------------------
# GROMACS .gro
# ---------------------------------------------------------------------------

_GRO_TIME = re.compile(r"\bt=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
_GRO_STEP = re.compile(r"\bstep=\s*(\d+)")


def _gro_count(line: bytes) -> int | None:
    tokens = line.split()
    if len(tokens) != 1 or not tokens[0].isdigit():
        return None
    n = int(tokens[0])
    return n if n >= 1 else None


def _gro_box(line: bytes) -> np.ndarray | None:
    """The box rows in nm from a box line of 3 or 9 values [3], or None."""
    tokens = line.split()
    if len(tokens) not in (3, 9) or not all(_is_number(t) for t in tokens):
        return None
    v = [float(t) for t in tokens] + [0.0] * (9 - len(tokens))
    if not np.isfinite(v).all():
        return None
    # v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
    return np.array([[v[0], v[3], v[4]], [v[5], v[1], v[6]],
                     [v[7], v[8], v[2]]])


def _gro_width(row: bytes) -> int | None:
    """The width of a position field: the distance between the first two
    decimal points after column 20 [3]."""
    first = row.find(b".", 20)
    second = row.find(b".", first + 1) if first >= 0 else -1
    if first < 0 or second < 0 or second - first < 5:
        return None
    return second - first


def _gro_row_ok(row: bytes) -> bool:
    """An atom line: residue number (5), names (5 + 5), atom number (5), then
    three fixed-width positions."""
    width = _gro_width(row)
    if width is None or len(row.rstrip()) < 20 + 3 * width:
        return False
    if not row[15:20].strip().isdigit() or not row[:5].strip().isdigit():
        return False
    return all(_is_number(row[20 + i * width:20 + (i + 1) * width])
               for i in range(3))


def _gro_shape(row: bytes) -> tuple[int, int] | None:
    """(field width, column of the first decimal point) of an atom line, or
    None when the line is not one (:func:`_gro_row_ok`)."""
    if not _gro_row_ok(row):
        return None
    return _gro_width(row), row.find(b".", 20)


def _gro_row_like(row: bytes, shape: tuple[int, int]) -> bool:
    """A cheap test that a line is an atom line of the frame whose first atom
    line has ``shape``: its three position fields put their decimal points in
    the same columns [3]. Title, count and box lines do not."""
    width, dot = shape
    return len(row) > dot + 2 * width and row[dot] == 46 \
        and row[dot + width] == 46 and row[dot + 2 * width] == 46


def _gro_frame_follows(lines: "_Walker") -> bool:
    """Whether the next two lines are an atom count and an atom line (both
    are left unread): a blank line before them is a frame's empty title."""
    count_line = lines.read()
    row = lines.read() if count_line and _gro_count(count_line) else b""
    follows = bool(row) and _gro_row_ok(row)
    lines.unread(row)
    lines.unread(count_line)
    return follows


def _whole_box_line(line: bytes, previous: int | None) -> bool:
    """Whether a box line written without a line ending (the file's last
    line) holds every value: 3 or 9 numbers with the same number of decimals
    (GROMACS writes each '%10.5f' [3]), as many as the frame before gave."""
    tokens = line.split()
    shapes = {_decimals(t) for t in tokens}
    return len(shapes) == 1 and min(shapes) >= 1 and \
        (previous is None or len(tokens) == previous)


@dataclass
class _GroBlock:
    position: int
    start: int
    end: int
    n: int
    box_ang: np.ndarray
    time_ps: float | None
    timestep: int | None
    box_values: int = 9


class _Walker:
    """readline with look-back, counting lines (1-based) and byte offsets."""

    def __init__(self, handle) -> None:
        self.handle = handle
        self.number = 0
        self.offset = 0
        self._back: list[bytes] = []

    def read(self) -> bytes:
        line = self._back.pop() if self._back else _mr()._readline(self.handle)
        if line:
            self.number += 1
            self.offset += len(line)
        return line

    def unread(self, line: bytes) -> None:
        if line:
            self._back.append(line)
            self.number -= 1
            self.offset -= len(line)


def _gro_index(path: Path):
    """Walk the frames by their atom counts. Returns (blocks, skipped,
    notes)."""
    mr = _mr()
    mr._refuse_cr_only(path.name, mr._head(path))
    blocks: list[_GroBlock] = []
    skipped: dict[int, str] = {}
    notes: list[str] = []
    blank = 0
    position = 0
    with mr._open_binary(path) as handle:
        # Offsets count every byte, a leading byte-order mark included: it is
        # in the file, and only the first title line holds it.
        lines = _Walker(handle)
        try:
            while True:
                start_line = lines.number + 1
                title = lines.read()
                if not title:
                    break
                if not title.strip() and not _gro_frame_follows(lines):
                    blank += 1
                    continue
                count_line = lines.read()
                if not count_line:
                    skipped[position] = ("truncated: the file ends after the "
                                         "title line")
                    break
                n = _gro_count(count_line)
                if n is None:
                    found = _gro_resync(lines)
                    skipped[position] = (
                        f"unreadable: line {start_line + 1} "
                        f"({_decode(count_line).strip()[:40]!r}) is not an atom "
                        f"count, so lines {start_line} to {lines.number} do "
                        "not form a frame (a .gro frame is a title line, the "
                        "atom count, one line per atom and the box line)"
                        + ("" if found else "; the file ends there"))
                    position += 1
                    continue
                rows_start = lines.offset
                box_line = _gro_skip_rows(handle, lines, n)
                if box_line is not None:
                    rows_end = lines.offset - len(box_line)
                else:
                    # Line by line, each checked as an atom line: a count
                    # that is too large meets the next frame's title, which
                    # then costs this frame only, not the frames after it.
                    got, shape, odd = 0, None, None
                    for _ in range(n):
                        row = lines.read()
                        if not row:
                            break
                        shape = shape or _gro_shape(row)
                        if shape is None or not _gro_row_like(row, shape):
                            odd = (row, lines.number)
                            lines.unread(row)
                            break
                        got += 1
                    if odd is not None:
                        found = _gro_resync(lines)
                        skipped[position] = (
                            f"its atom count gives {n}, and line {odd[1]} "
                            f"({_decode(odd[0]).strip()[:40]!r}), after {got} "
                            "atom lines, is not an atom line, so the count and "
                            "the lines disagree"
                            + ("" if found else "; the file ends there"))
                        position += 1
                        continue
                    if got < n:
                        skipped[position] = (f"truncated: the file ends after "
                                             f"{got} of {n} atom lines")
                        break
                    rows_end = lines.offset
                    box_line = lines.read()
                if not box_line:
                    skipped[position] = ("truncated: the file ends before the "
                                         "box line")
                    break
                box = _gro_box(box_line)
                if box is None:
                    lines.unread(box_line)
                    _gro_resync(lines)
                    skipped[position] = (
                        f"its atom count gives {n}, and the line after {n} atom "
                        f"lines ({_decode(box_line).strip()[:40]!r}) is not a "
                        "box line of 3 or 9 values, so the count and the lines "
                        "disagree")
                    position += 1
                    continue
                values = len(box_line.split())
                if not box_line.endswith((b"\n", b"\r")):
                    if not _whole_box_line(box_line, blocks[-1].box_values
                                           if blocks else None):
                        skipped[position] = (
                            "truncated: the file does not end with a line "
                            "ending, and its box line holds fewer values, or "
                            "fewer decimals in its last value, than a whole "
                            "box line, so it was cut")
                        break
                    notes.append(
                        "the file does not end with a line ending; its last "
                        f"line, a box line, holds {values} values written with "
                        "the same number of decimals, so its frame is read")
                text = _decode(title)
                t = _GRO_TIME.search(text)
                s = _GRO_STEP.search(text)
                blocks.append(_GroBlock(
                    position, rows_start, rows_end, n, box * NM_ANG,
                    float(t.group(1)) if t else None,
                    int(s.group(1)) if s else None, values))
                position += 1
        except EOFError:
            skipped[position] = ("truncated: the gzip stream ends before its "
                                 "end marker")
            notes.append(mr._GZIP_CUT_NOTE)
    if blank:
        notes.append(f"{blank} blank line(s) between or after frames are "
                     "passed over")
    return blocks, skipped, notes


def _gro_skip_rows(handle, lines: _Walker, n: int) -> bytes | None:
    """The box line after ``n`` atom lines of equal length, reached with one
    seek, or None (and the walker where it was) when the lines differ in
    length or the line found there is not a box line.

    GROMACS, mdtraj and ASE write every atom line of a frame with the same
    fixed-width format, so the box line starts n line lengths after the
    first atom line; that position is accepted only when the line before it
    has the first atom line's length and decimal points (an atom line, ended
    by a line ending) and the line there holds the 3 or 9 values of a box
    [3].
    """
    if lines._back or n < 2:
        return None
    start, number = lines.offset, lines.number
    first = lines.read()
    shape = _gro_shape(first)
    if not first.endswith(b"\n") or shape is None:
        lines.unread(first)
        return None
    target = start + n * len(first)
    handle.seek(target - len(first))
    before = handle.read(len(first))
    line = _mr()._readline(handle) if (
        len(before) == len(first) and before.endswith(b"\n")
        and _gro_row_like(before, shape)) else b""
    if line and _gro_box(line) is not None:
        lines.offset = target + len(line)
        lines.number = number + n + 1
        return line
    handle.seek(start)
    lines.offset, lines.number = start, number
    return None


def _gro_resync(lines: _Walker) -> bool:
    """Pass lines until a title line (blank or not) followed by an atom count
    and an atom line, and leave the three unread. False when the file ends
    first."""
    previous = None
    while True:
        line = lines.read()
        if not line:
            return False
        n = _gro_count(line) if previous is not None else None
        if n is not None:
            row = lines.read()
            if row and _gro_row_ok(row):
                lines.unread(row)
                lines.unread(line)
                lines.unread(previous)
                return True
            lines.unread(row)
        previous = line


class _GroTrajectory(_RangeTrajectory):
    def __init__(self, path, blocks, *, names, elements, keep_velocities,
                 **kwargs) -> None:
        super().__init__(path, [(b.start, b.end) for b in blocks], **kwargs)
        self._blocks = list(blocks)
        self._names = names
        self._elements = elements
        self._keep_velocities = keep_velocities

    def _read_frame(self, k: int) -> Frame:
        block = self._blocks[k]
        names, cart_nm, vel = _gro_rows(self._bytes(k), block.n)
        if not np.array_equal(names, self._names):
            differ = int(np.flatnonzero(names != self._names)[0]) \
                if names.shape == self._names.shape else 0
            raise ValueError(f"atom line {differ + 1} names "
                             f"{names[differ]!r} where frame 0 names "
                             f"{self._names[differ]!r}")
        notes = []
        if vel is not None and not self._keep_velocities:
            if np.any(vel):
                notes.append(_ZERO_VELOCITY_FRAME_NOTE)
            vel = None
        return frame_from_arrays(
            self._elements, cart_nm * NM_ANG, box_ang=block.box_ang,
            timestep=block.timestep, time_ps=block.time_ps,
            vel_ang_per_ps=None if vel is None else vel * NM_ANG, notes=notes)


def _gro_rows(data: bytes, n: int):
    """(atom names, positions in nm, velocities in nm/ps or None) from the
    atom lines of one frame [3]."""
    rows = data.splitlines()
    if len(rows) != n:
        raise ValueError(f"{len(rows)} atom lines where the count gives {n}")
    width = _gro_width(rows[0])
    if width is None:
        raise ValueError(f"the first atom line ({_decode(rows[0]).strip()[:50]!r}) "
                         "holds no two decimal points after column 20, from "
                         "which the field width is read")
    lengths = {len(r.rstrip()) for r in rows}
    names = np.array([_decode(r[10:15]).strip() for r in rows])
    fields = []
    for i in range(3):
        a, b = 20 + i * width, 20 + (i + 1) * width
        fields.append([r[a:b] for r in rows])
    cart = _floats(np.array(fields).T, "a position field").reshape(n, 3)
    vel = None
    v_end = 20 + 6 * width
    if min(lengths) >= v_end:
        v_fields = [[r[20 + j * width:20 + (j + 1) * width] for r in rows]
                    for j in range(3, 6)]
        vel = _floats(np.array(v_fields).T, "a velocity field").reshape(n, 3)
    elif max(lengths) >= v_end:
        raise ValueError("some atom lines hold velocities and others do not")
    return names, cart, vel


def read_gro(path, *, type_map: Mapping | None = None) -> Trajectory:
    """A GROMACS ``.gro`` file of one or more frames [3].

    A frame is a title line (``t=`` gives the time in ps, ``step=`` the
    step), the atom count, one fixed-width line per atom and the box line.
    Positions and velocities are read from fields whose width is the distance
    between the decimal points; nm and nm/ps become Å and Å/ps. Atoms keep the
    file's order, numbered 0 .. N-1; the atom name (columns 11-15) is the
    label ``type_map`` keys, read as an element only when written as files
    write symbols ('OW' and 'HW1' need a map).

    The walk follows the atom counts. A blank title line is a title when an
    atom count and an atom line follow it. A count line that does not parse,
    or a count larger than the frame (the walk meets a line that is not an
    atom line), skips that frame and resynchronises at the next one. A last
    box line without a line ending is read when it holds as many values as
    the box line before it, each with the same decimals.
    """
    mr = _mr()
    path = mr._existing(path)
    blocks, skipped, notes = _gro_index(path)
    if not blocks:
        raise ValueError(f"{path.name} holds no readable frame" + (": " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items()))
            if skipped else ""))
    n0 = mr._majority_count([b.n for b in blocks])
    kept = []
    for block in blocks:
        if block.n != n0:
            skipped[block.position] = mr._count_reason(
                block.n, n0, [b.n for b in blocks])
        else:
            kept.append(block)
    blocks = kept
    boxed = [b for b in blocks if np.any(b.box_ang)]
    if not boxed:
        raise UnsupportedFormat(
            f"{path.name}: the box line holds only zeros, so the file gives no "
            "periodic box (mdtraj writes zeros, and ASE an empty line, for a "
            "model without one). FACET reads periodic models: write the .gro "
            "with its box, the last line of each frame")
    for block in blocks:
        if not np.any(block.box_ang):
            skipped[block.position] = ("its box line holds only zeros, so it "
                                       "gives no periodic box")
    blocks = boxed
    try:
        data = _read_range(path, blocks[0].start, blocks[0].end)
        names, _, vel0 = _gro_rows(data, blocks[0].n)
    except ValueError as error:
        raise FrameError(f"{path.name}, frame 0 (file position "
                         f"{blocks[0].position}): {error}") from error
    elements, listed, source, label_notes = _labels_to_elements(
        path, names, type_map, what="atom name")
    # A run started from rest has zero velocities in its first frame only;
    # zeros in the first and the last frame are a writer's filler.
    vel_last = None
    if vel0 is not None and not np.any(vel0) and len(blocks) > 1:
        try:
            _, _, vel_last = _gro_rows(
                _read_range(path, blocks[-1].start, blocks[-1].end),
                blocks[-1].n)
        except ValueError:
            vel_last = None             # its error comes when it is loaded
    keep_velocities = vel0 is not None and bool(
        np.any(vel0) or (vel_last is not None and np.any(vel_last)))
    notes = label_notes + notes
    if vel0 is not None and not keep_velocities:
        notes.append(_ZERO_VELOCITY_NOTE)
    notes.append("atoms are numbered 0 .. N-1 in the order of the file; "
                 "residue names and numbers and the atom number column are not "
                 "read; the box origin is (0, 0, 0)")
    times = [b.time_ps for b in blocks]
    steps = [b.timestep for b in blocks]
    if all(t is None for t in times):
        notes.append("no title line gives a time (t=)")
    box_varies = any(not np.array_equal(b.box_ang, blocks[0].box_ang)
                     for b in blocks[1:])
    return _finish(_GroTrajectory(
        path, blocks, names=names, elements=elements,
        keep_velocities=keep_velocities, file_format="gromacs-gro",
        n_atoms=n0, n_frames=len(blocks), type_map_source=source,
        timesteps=[NO_TIMESTEP if s is None else s for s in steps],
        times_ps=None if all(t is None for t in times) else
        [np.nan if t is None else t for t in times],
        type_map=listed, skipped=skipped, notes=notes,
        units_note=_GRO_UNITS_NOTE, box_varies=box_varies))


def sniff_gro(head: bytes, path: Path) -> bool:
    """A GROMACS head: a title, a lone atom count, then a fixed-width atom
    line whose three position fields parse."""
    try:
        lines = head.splitlines()
        if len(lines) < 3 or _gro_count(lines[1]) is None:
            return False
        return _gro_row_ok(lines[2])
    except Exception:           # noqa: BLE001 - a sniffer never raises
        return False


# ---------------------------------------------------------------------------
# XCrySDen animated XSF
# ---------------------------------------------------------------------------

_XSF_PERIODICITY = ("CRYSTAL", "SLAB", "POLYMER", "MOLECULE")
# How many steps that ANIMSTEPS announces and the file does not hold get a
# skipped entry each (a choice: a file cut early keeps one entry per lost
# step; beyond this, and beyond the number of sections in the file, the
# rest is counted in a note).
_XSF_LISTED_MISSING = 1000


@dataclass
class _XsfBlock:
    position: int
    start: int
    end: int
    box_ang: np.ndarray


def _xsf_keywords(path: Path) -> tuple[int | None, str | None, bool]:
    """(ANIMSTEPS, the periodicity keyword, whether ATOMS sections appear)
    from the file's keyword lines, read up to the first coordinate section."""
    mr = _mr()
    steps, kind, atoms = None, None, False
    with mr._open_binary(path) as handle:
        while True:
            line = mr._readline(handle)
            if not line:
                break
            tokens = mr._strip_bom(line).split()
            if not tokens or tokens[0].startswith(b"#"):
                continue
            word = tokens[0].decode("ascii", "replace").upper()
            if word == "ANIMSTEPS" and len(tokens) > 1 and tokens[1].isdigit():
                steps = int(tokens[1])
            elif word in _XSF_PERIODICITY:
                kind = word
            elif word == "ATOMS":
                atoms = True
                break
            elif word.startswith("PRIMCOORD"):
                break
    return steps, kind, atoms


class _XsfTrajectory(_RangeTrajectory):
    def __init__(self, path, blocks, *, n, tokens, elements, **kwargs) -> None:
        super().__init__(path, [(b.start, b.end) for b in blocks], **kwargs)
        self._blocks = list(blocks)
        self._n = n
        self._tokens = tokens
        self._elements = elements

    def _read_frame(self, k: int) -> Frame:
        tokens, cart, _, _ = _xsf_rows(self._bytes(k), self._n)
        if not np.array_equal(tokens, self._tokens):
            raise ValueError("the atom lines name other atoms, or list them in "
                             "another order, than frame 0's")
        return frame_from_arrays(self._elements, cart,
                                 box_ang=self._blocks[k].box_ang)


def _xsf_lines(data: bytes) -> list[bytes]:
    """The lines of a section that hold text and are not comments ('#'
    starts a comment line anywhere in an XSF file [4])."""
    return [x for x in data.splitlines()
            if x.strip() and not x.lstrip().startswith(b"#")]


def _xsf_rows(data: bytes, n_expected: int | None):
    """(atom tokens, positions in Å, whether force columns follow, the table
    of tokens) from the lines after a PRIMCOORD keyword: 'n 1', then n lines
    [4]."""
    lines = _xsf_lines(data)
    if not lines:
        raise ValueError("no atom count line after PRIMCOORD")
    head = lines[0].split()
    if len(head) < 1 or not head[0].isdigit() or int(head[0]) < 1:
        raise ValueError(f"the line after PRIMCOORD "
                         f"({_decode(lines[0]).strip()[:40]!r}) is not 'number "
                         "of atoms' and '1'")
    n = int(head[0])
    if n_expected is not None and n != n_expected:
        raise ValueError(f"PRIMCOORD gives {n} atoms; frame 0 gives {n_expected}")
    rows = lines[1:1 + n]
    if len(rows) < n:
        raise ValueError(f"{len(rows)} atom lines where PRIMCOORD gives {n}")
    values = b" ".join(rows).split()
    # 4a + 7b values over a + b = n lines is 4n only when b = 0 and 7n only
    # when a = 0, so the count alone says every line has the same width.
    width = len(values) // n if len(values) in (4 * n, 7 * n) else None
    if width is None:
        raise ValueError("the atom lines hold neither 4 values each (atom, x, "
                         "y, z) nor 7 (with the three force components)")
    table = np.array(values).reshape(n, width)
    tokens = table[:, 0].astype(str)
    cart = _floats(table[:, 1:4], "an atom line")
    return tokens, cart, width == 7, table


def _xsf_elements(path: Path, tokens: np.ndarray, type_map):
    """Elements from XSF atom tokens: an atomic number names its element; a
    symbol is a label (:func:`.md_readers._resolve_labels`)."""
    import gemmi

    numeric = np.array([t.isdigit() for t in tokens])
    elements = np.empty(tokens.shape, dtype="<U2")
    listed: dict = {}
    sources: list[str] = []
    notes: list[str] = []
    for z in np.unique(tokens[numeric]):
        element = gemmi.Element(int(z)) if 0 < int(z) < 1000 else None
        if element is None or element.atomic_number != int(z):
            raise ValueError(
                f"{path.name}: atomic number {z} names no element in gemmi's "
                "periodic table (XCrySDen writes 0 for a dummy atom); FACET "
                "reads atoms of real elements, so remove such atoms or give "
                "them an element")
        elements[tokens == z] = validate_symbol(element.name)
    if numeric.any():
        sources.append("file symbols")
    if (~numeric).any():
        labelled, listed, source, notes = _labels_to_elements(
            path, tokens[~numeric], type_map, what="atom symbol")
        elements[~numeric] = labelled
        sources.extend(source.split(" + "))
    elif type_map:
        notes.append("the type map is not used: the file names its atoms by "
                     "atomic number")
    return elements, listed, _mr()._join_sources(sources), notes


def read_xsf(path, *, type_map: Mapping | None = None) -> Trajectory:
    """An XCrySDen XSF file of a periodic (CRYSTAL) structure, animated
    (ANIMSTEPS) or not [4].

    The cell is PRIMVEC, given once for a fixed cell or as ``PRIMVEC i`` before
    each step of a variable cell; each ``PRIMCOORD i`` gives the atom count
    and one line per atom (atomic number or symbol, x, y, z in Å, optionally
    three force components, not read). SLAB, POLYMER and MOLECULE files are
    refused: they are not periodic along three axes. Atoms keep the file's
    order, numbered 0 .. N-1; the file holds no time or MD step.

    In an animated file the number after PRIMCOORD places each section: a
    number skipped is a skipped position (the lines the section before it
    holds in excess are named in the reason), a number written twice or out
    of order is read at its place in the file with a note, and in a variable
    cell the last PRIMVEC before 'PRIMCOORD i' has to be 'PRIMVEC i'.
    """
    mr = _mr()
    path = mr._existing(path)
    steps, kind, has_atoms = _xsf_keywords(path)
    if kind != "CRYSTAL":
        why = (f"it is a {kind} (periodic along fewer than three axes)" if kind
               else ("it gives no CRYSTAL keyword" + (" and its coordinates are "
                     "ATOMS sections (a molecule)" if has_atoms else "")))
        raise UnsupportedFormat(
            f"{path.name}: {why}. FACET reads models periodic along a, b and "
            "c: write the model as a CRYSTAL with PRIMVEC and PRIMCOORD (ASE "
            "writes one for atoms periodic along all three axes)")
    scan = mr._scan(path, b"PRIM", line_start=True, after_lines=3)
    marks = [m for m in scan.marks
             if m.text.split()[:1] in (["PRIMVEC"], ["PRIMCOORD"])]
    coords = [i for i, m in enumerate(marks) if m.text.split()[0] == "PRIMCOORD"]
    if not coords:
        raise UnsupportedFormat(f"{path.name}: no PRIMCOORD section, so no "
                                "atoms; FACET reads the PRIMCOORD sections of "
                                "a CRYSTAL file")
    animated = steps is not None

    def number(mark) -> int | None:
        """The step number after PRIMVEC or PRIMCOORD in an animated file."""
        words = mark.text.split()
        return int(words[1]) if animated and len(words) > 1 \
            and words[1].isdigit() else None

    vectors = [m for m in marks if m.text.split()[0] == "PRIMVEC"]
    # A variable cell gives 'PRIMVEC i' before each 'PRIMCOORD i'; a fixed
    # cell one PRIMVEC with no number [4].
    numbered_cell = any(number(m) is not None for m in vectors)
    variable = numbered_cell or len(vectors) > 1
    skipped: dict[int, str] = {}
    blocks: list[_XsfBlock] = []
    step_notes: list[str] = []
    box = box_error = None
    box_number = box_line = None
    n0 = None
    tokens0 = None
    forces = False
    position = 0                    # the place of the next section
    # The lines the last section held past its own rows (its atoms, or a
    # PRIMVEC's three vectors): the lines of a step whose 'PRIMCOORD i' line
    # is lost, named in that step's skipped reason.
    loose: str | None = None
    # A step number further ahead than the file has sections is not taken
    # as a gap: one mistyped number would otherwise open that many entries.
    reach = len(coords)
    for index, mark in enumerate(marks):
        words = mark.text.split()
        if words[0] == "PRIMVEC":
            box_number, box_line = number(mark), mark.line + 1
            try:
                rows = list(mark.after[:3])
                if len(rows) < 3:
                    raise ValueError("fewer than 3 lines follow PRIMVEC")
                if any(len(r.split()) < 3 for r in rows):
                    raise ValueError("a PRIMVEC line holds fewer than 3 values")
                box = _floats([r.split()[:3] for r in rows],
                              "a PRIMVEC line")
                md_model._checked_box(box)
                box_error = None
            except ValueError as error:
                box, box_error = None, str(error)
            if mr._rows_between(marks, index, scan) > 3:
                end = marks[index + 1].offset if index + 1 < len(marks) \
                    else scan.size
                more = _xsf_extra(_xsf_lines(_read_range(path, mark.end,
                                                          end))[3:])
                if more:
                    loose = (f"the PRIMVEC section at file line "
                             f"{mark.line + 1} holds {more} more line(s) after "
                             "its three vectors, which are not read")
                    step_notes.append(loose)
            continue
        written = number(mark)
        if written is None or written <= position or \
                (steps is not None and written > steps) or \
                written - 1 - position > reach:
            step = position
            if written is not None:
                step_notes.append(
                    f"'PRIMCOORD {written}' (file line {mark.line + 1}) "
                    f"follows step {position}: a step number written twice, "
                    "out of order or beyond the file, so the section is read "
                    f"as step {step + 1}, its place in the file")
        else:
            step = written - 1
            for missing in range(position, step):
                skipped[missing] = (
                    f"not in the file: no 'PRIMCOORD {missing + 1}' line "
                    f"between step {position} and 'PRIMCOORD {written}' (file "
                    f"line {mark.line + 1})" + (f"; {loose}" if loose else ""))
        loose = None
        position = step + 1
        is_last = index == coords[-1]
        end = marks[index + 1].offset if index + 1 < len(marks) else scan.size
        if box is None:
            skipped[step] = ("unreadable: no readable PRIMVEC before it"
                             + (f" ({box_error})" if box_error else ""))
            continue
        if numbered_cell and box_number is not None and \
                box_number != step + 1:
            skipped[step] = (
                f"the last PRIMVEC before it (file line {box_line}) is "
                f"'PRIMVEC {box_number}', not 'PRIMVEC {step + 1}': the cell "
                f"of step {step + 1} is not in the file")
            continue
        try:
            rows_after = mr._rows_between(marks, index, scan)
            if n0 is None or is_last or rows_after > n0 + 1:
                data = _read_range(path, mark.end, end)
                tokens, cart, has_forces, table = _xsf_rows(data, n0)
                if n0 is None:
                    n0, tokens0, forces = len(tokens), tokens, has_forces
                more = _xsf_extra(_xsf_lines(data)[1 + n0:])
                if more:
                    loose = (f"step {step + 1}'s section (from file line "
                             f"{mark.line + 1}) holds {more} more line(s) after "
                             f"its count line and {n0} atom lines, which are "
                             "not read")
                    step_notes.append(loose)
                if is_last and not scan.last_line_ended and \
                        rows_after <= n0 + 1 + more:
                    if more or not _whole_last_row(table.tolist(),
                                                   table.shape[1]):
                        raise ValueError(
                            "the file does not end with a line ending, and its "
                            "last line holds fewer values, or fewer decimals "
                            "in its last value, than the line above, so it "
                            "was cut")
                    step_notes.append(
                        "the file does not end with a line ending; its last "
                        "line holds every value, with as many decimals as the "
                        "line above, so it is read")
            else:
                # The count line and the number of lines before the next
                # keyword, without reading the step.
                count = next((x.split()[0] for x in mark.after if x.strip()),
                             "")
                if count != str(n0):
                    raise ValueError(f"PRIMCOORD gives {count or 'no'} atoms; "
                                     f"frame 0 gives {n0}")
                if rows_after < n0 + 1:
                    raise ValueError(f"fewer than {n0} atom lines before the "
                                     "next PRIMVEC or PRIMCOORD")
        except ValueError as error:
            skipped[step] = ("truncated: " if is_last else "unreadable: ") \
                + str(error)
            continue
        blocks.append(_XsfBlock(step, mark.end, end, box))
    if not blocks or n0 is None:
        raise ValueError(f"{path.name} holds no readable step: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))
    notes: list[str] = []
    found = position
    if steps is not None and found < steps:
        # One entry per missing step, up to _XSF_LISTED_MISSING (or as many
        # as the file has sections, when more); the rest is a count, so a
        # mistyped ANIMSTEPS costs no memory (one entry each made 999 997
        # entries, +217 MB as a verification pass measured, for 'ANIMSTEPS
        # 1000000' over 3 steps).
        missing = steps - found
        listed = min(missing, max(len(coords), _XSF_LISTED_MISSING))
        for j in range(found, found + listed):
            skipped[j] = (f"not in the file: ANIMSTEPS gives {steps} steps "
                          f"and the file ends after {found}")
        if missing > listed:
            notes.append(
                f"ANIMSTEPS gives {steps} steps and the file ends after "
                f"{found}: the {missing} steps after them are not in the "
                f"file; the first {listed} have a skipped entry each, the "
                f"other {missing - listed} only this count")
    elif steps is not None and found > steps:
        notes.append(f"ANIMSTEPS gives {steps} steps and the file holds {found} "
                     "PRIMCOORD sections; every section is read")
    notes.extend(step_notes)
    elements, listed, source, label_notes = _xsf_elements(path, tokens0,
                                                          type_map)
    notes = label_notes + notes
    notes.append(("variable cell: a PRIMVEC before each step" if variable else
                  "fixed cell: one PRIMVEC for every step" if animated else
                  "one structure (no ANIMSTEPS line), read as one frame")
                 + "; atoms are numbered 0 .. N-1 in the order of the file; "
                 "the box origin is (0, 0, 0); "
                 + ("the step number after PRIMCOORD (and after PRIMVEC in a "
                    "variable cell) places each section among the animation "
                    "steps, and " if animated else "")
                 + "the file holds no time or MD step")
    if forces:
        notes.append("not read: the force columns (Hartree/Å)")
    if b"CONVVEC" in mr._head(path):
        notes.append("CONVVEC (the conventional cell) is not used: the box is "
                     "PRIMVEC, the cell PRIMCOORD's atoms belong to")
    if scan.gzip_cut:
        notes.append(mr._GZIP_CUT_NOTE)
    box_varies = any(not np.array_equal(b.box_ang, blocks[0].box_ang)
                     for b in blocks[1:])
    trajectory = _XsfTrajectory(
        path, blocks, n=n0, tokens=tokens0, elements=elements,
        file_format="xsf", n_atoms=n0, n_frames=len(blocks),
        type_map_source=source, type_map=listed, skipped=skipped, notes=notes,
        units_note="XSF lengths in Å [XSF specification]",
        box_varies=box_varies)
    # CRYSTAL states periodicity along all three vectors [4].
    trajectory.periodic = (True, True, True)
    return _finish(trajectory)


def _xsf_extra(lines: Sequence[bytes]) -> int:
    """How many lines follow a step's atom lines before the next PRIMVEC or
    PRIMCOORD, a CONVVEC section (the keyword and its three rows [4]) not
    counted: lines of a step whose 'PRIMCOORD i' line is lost."""
    count, skip = 0, 0
    for line in lines:
        if skip:
            skip -= 1
        elif line.split()[0].upper() == b"CONVVEC":
            skip = 3
        else:
            count += 1
    return count


def sniff_xsf(head: bytes, path: Path) -> bool:
    """An XSF head [4]: its first keyword line is 'ANIMSTEPS n' (an animated
    file), or 'CRYSTAL' followed by PRIMVEC and PRIMCOORD (one periodic
    structure: no crystal reader of FACET opens XSF, so the MD reader reads
    it as one frame). Comment lines ('#') are passed over."""
    try:
        words: list[bytes] = []
        for line in head.splitlines():
            tokens = line.split()
            if not tokens or tokens[0].startswith(b"#"):
                continue
            word = tokens[0].upper()
            if not words:
                if word == b"ANIMSTEPS":
                    return len(tokens) >= 2 and tokens[1].isdigit()
                if word != b"CRYSTAL":
                    return False
            words.append(word)
        return b"PRIMVEC" in words and b"PRIMCOORD" in words
    except Exception:           # noqa: BLE001 - a sniffer never raises
        return False


# ---------------------------------------------------------------------------
# PDB with several frames
# ---------------------------------------------------------------------------

_PDB_RECORDS = (b"HEADER", b"TITLE ", b"REMARK", b"CRYST1", b"MODEL ",
                b"ATOM  ", b"HETATM", b"COMPND", b"AUTHOR", b"SCALE",
                b"ORIGX", b"EXPDTA", b"KEYWDS", b"SOURCE", b"END")

# CP2K's PDB trajectory title, written as a REMARK before each frame:
# 'Step <it>, time = <t>, E = <etot>' (motion_utils.F, FMT
# "(A,I0,A,F0.3,A,F0.10)"; 'Step <it>, E = <etot>' without the extended
# title), the time in fs (md_energies.F passes time*femtoseconds) [8].
_CP2K_REMARK = re.compile(
    r"^REMARK\s+Step\s+(\d+)\s*,(?:\s*time\s*=\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?))?")
# 1 fs in ps.
FS_PS: float = _constants.femto / _constants.pico
# The largest magnitude a Real(8.3) coordinate field (columns 31-38, 39-46,
# 47-54 [5]) can hold.
_PDB_COORDINATE_MAX = 9999.999


def _is_end(line: bytes) -> bool:
    """An END record: 'END' and nothing else but blanks, leading blanks
    allowed as :func:`.md_readers.pdb_model_count` allows them (a file
    written with ' END')."""
    return line.strip() == b"END"


@dataclass
class _PdbRecord:
    kind: str           # MODEL, ENDMDL, END, CRYST1, SCALE, REMARK
    offset: int
    end: int
    line: int           # 1-based file line
    text: str           # the record from its name on, without line ending
    atoms_before: int   # ATOM / HETATM records since the record before it
    first_atom: int     # the file line of the first of them (0: none)


@dataclass
class _PdbIndex:
    records: list
    atoms_after: int    # ATOM / HETATM records after the last record
    first_atom_after: int
    size: int
    n_lines: int
    gzip_cut: bool


def _pdb_index(path: Path) -> _PdbIndex:
    """One pass over the lines of a PDB file: the MODEL, ENDMDL, END, CRYST1,
    SCALEn and REMARK records with their byte offsets and file lines, and
    the number of ATOM / HETATM records between each and the one before it
    [5]. Counting the atom records lets every frame's atom count be known
    without parsing it, so a frame that two frames ran into, or atoms
    outside any frame, are found when the file is opened."""
    mr = _mr()
    mr._refuse_cr_only(path.name, mr._head(path))
    records: list[_PdbRecord] = []
    atoms = first = 0
    offset = number = 0
    cut = False
    with mr._open_binary(path) as handle:
        try:
            for raw in handle:
                number += 1
                start = offset
                offset += len(raw)
                if raw[:4] == b"ATOM" or raw[:6] == b"HETATM":
                    if not atoms:
                        first = number
                    atoms += 1
                    continue
                key = raw.lstrip(b" \t")
                if key[:6] == b"ENDMDL":
                    kind = "ENDMDL"
                elif key[:3] == b"END":
                    if key.strip() != b"END":
                        continue
                    kind = "END"
                elif key[:5] == b"MODEL":
                    kind = "MODEL"
                elif key[:6] == b"CRYST1":
                    kind = "CRYST1"
                elif key[:5] == b"SCALE" and key[5:6] in (b"1", b"2", b"3"):
                    kind = "SCALE"
                elif key[:6] == b"REMARK":
                    kind = "REMARK"
                else:
                    continue
                records.append(_PdbRecord(kind, start, offset, number,
                                          _decode(key).rstrip("\r\n"), atoms,
                                          first))
                atoms = first = 0
        except EOFError:
            cut = True
        except mr._GZIP_DAMAGE as error:
            raise mr._damaged(handle, error) from None
    return _PdbIndex(records, atoms, first, offset, number, cut)


@dataclass
class _PdbItem:
    """A run of atom records the index found: a MODEL ... ENDMDL block
    ('model'), a MODEL with no ENDMDL ('open'), a block ended by END in a
    file without MODEL records ('end'), the atoms after the last END
    ('tail'), or atoms outside any MODEL block in a file with MODEL records
    ('outside')."""

    kind: str
    start: int
    end: int
    line: int           # file line of its first byte
    last_line: int      # its last file line
    atoms: int
    meta: int           # REMARK records from this offset on belong to it
    reason: str = ""


def _pdb_items(index: _PdbIndex) -> tuple[list[_PdbItem], str]:
    """The runs of atom records in file order, and the delimiter ('MODEL'
    when the file has MODEL records, else 'END')."""
    items: list[_PdbItem] = []
    records = index.records
    seg_start, seg_line, seg_atoms = 0, 1, 0
    if any(r.kind == "MODEL" for r in records):
        model, inside, meta = None, 0, 0
        for r in records:
            if model is not None:
                inside += r.atoms_before
            else:
                seg_atoms += r.atoms_before
            if r.kind == "MODEL":
                if model is not None:
                    items.append(_PdbItem(
                        "open", model.offset, r.offset, model.line, r.line - 1,
                        inside, meta, "no ENDMDL before the next MODEL record"))
                    meta = r.offset
                elif seg_atoms:
                    items.append(_PdbItem("outside", seg_start, r.offset,
                                          seg_line, r.line - 1, seg_atoms,
                                          seg_start))
                    seg_atoms, meta = 0, r.offset
                else:
                    meta = seg_start
                model, inside = r, 0
            elif r.kind in ("ENDMDL", "END"):
                if model is not None and r.kind == "ENDMDL":
                    items.append(_PdbItem("model", model.offset, r.end,
                                          model.line, r.line, inside, meta))
                    model = None
                elif model is None and seg_atoms:
                    items.append(_PdbItem("outside", seg_start, r.end,
                                          seg_line, r.line, seg_atoms,
                                          seg_start))
                    seg_atoms = 0
                if model is None:
                    seg_start, seg_line = r.end, r.line + 1
        if model is not None:
            items.append(_PdbItem(
                "open", model.offset, index.size, model.line, index.n_lines,
                inside + index.atoms_after, meta,
                "truncated: no ENDMDL before the file ends"))
        elif seg_atoms + index.atoms_after:
            items.append(_PdbItem("outside", seg_start, index.size, seg_line,
                                  index.n_lines, seg_atoms + index.atoms_after,
                                  seg_start))
        return items, "MODEL"
    for r in records:
        seg_atoms += r.atoms_before
        if r.kind in ("END", "ENDMDL"):
            if seg_atoms:
                items.append(_PdbItem("end", seg_start, r.offset, seg_line,
                                      r.line - 1, seg_atoms, seg_start))
            seg_start, seg_line, seg_atoms = r.end, r.line + 1, 0
    if seg_atoms + index.atoms_after:
        items.append(_PdbItem("tail", seg_start, index.size, seg_line,
                              index.n_lines, seg_atoms + index.atoms_after,
                              seg_start))
    return items, "END"


@dataclass
class _PdbBlock:
    position: int
    start: int
    end: int
    line: int
    cryst1: str
    box_ang: np.ndarray
    scale: tuple
    timestep: int | None
    time_ps: float | None


def _cryst1_values(text: str) -> tuple[list[float], list[float]]:
    """(a, b, c, alpha, beta, gamma) from a CRYST1 record, columns 7-54 [5]
    (or its first six words after the name, for a record whose columns
    shifted), and half a unit of the last decimal written for each: the
    rounding the record carries."""
    fields = [text[6:15], text[15:24], text[24:33], text[33:40], text[40:47],
              text[47:54]]
    try:
        values = [float(x) for x in fields]
    except ValueError:
        fields = text.split()[1:7]
        if len(fields) < 6 or not all(_is_number(t) for t in fields):
            raise ValueError(f"the CRYST1 record {text.strip()[:60]!r} does "
                             "not hold a, b, c, alpha, beta, gamma") from None
        values = [float(t) for t in fields]
    steps = [0.5 * 10.0 ** -max(_decimals(x.strip().encode()), 0)
             for x in fields]
    return values, steps


def _standard_box(values: Sequence[float]) -> np.ndarray:
    """CRYST1's standard orthogonalisation [5] (a along x, b in the xy
    plane), in numpy: the rounding of the record is carried through it."""
    a, b, c = values[:3]
    alpha, beta, gamma = np.radians(values[3:6])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    return np.array([[a, 0.0, 0.0], [b * np.cos(gamma), b * np.sin(gamma), 0.0],
                     [cx, cy, np.sqrt(c * c - cx * cx - cy * cy)]])


# What to do about a CRYST1 record that gives no periodic box.
_NO_BOX_REMEDY = ("; FACET reads periodic models: write the trajectory with "
                  "its box (CRYST1), or as extended XYZ with Lattice=")


def _cryst1_box(text: str) -> np.ndarray:
    """Box rows in Å from a CRYST1 record: columns 7-54 [5], the standard
    orthogonalisation (gemmi.UnitCell: a along x, b in the xy plane).

    UnsupportedFormat for values no cell has (a length that is not positive,
    an angle outside 0-180°, angles whose metric has no volume) and for the
    1 Å placeholder; gemmi substitutes a unit cube for zeros and raises
    RuntimeError for some of the others, so they are checked first."""
    import gemmi

    record = text.strip()[:60]
    values, _ = _cryst1_values(text)
    if not np.isfinite(values).all():
        raise ValueError(f"the CRYST1 record {record!r} holds a value that is "
                         "not finite")
    if values[:3] == [1.0, 1.0, 1.0] and values[3:] == [90.0, 90.0, 90.0]:
        raise UnsupportedFormat(
            "CRYST1 gives a = b = c = 1 Å and 90° angles, the placeholder the "
            "PDB format uses for a structure that is not a crystal, so the "
            "file gives no periodic box" + _NO_BOX_REMEDY)
    a, b, c, alpha, beta, gamma = values
    if min(a, b, c) <= 0:
        raise UnsupportedFormat(
            f"the CRYST1 record {record!r} gives a cell length of "
            f"{min(a, b, c):g} Å; cell lengths are above 0, so it gives no "
            "periodic box" + _NO_BOX_REMEDY)
    odd = next((x for x in (alpha, beta, gamma) if not 0 < x < 180), None)
    if odd is not None:
        raise UnsupportedFormat(
            f"the CRYST1 record {record!r} gives an angle of {odd:g}°; cell "
            "angles lie between 0 and 180°, so it gives no periodic box"
            + _NO_BOX_REMEDY)
    cos = np.cos(np.radians([alpha, beta, gamma]))
    metric = 1.0 - float((cos ** 2).sum()) + 2.0 * float(cos.prod())
    if not metric > 0:
        raise UnsupportedFormat(
            f"the CRYST1 record {record!r}: its angles alpha, beta, gamma "
            f"({alpha:g}, {beta:g}, {gamma:g}°) describe no cell (1 - cos²α - "
            f"cos²β - cos²γ + 2 cosα cosβ cosγ = {metric:.3g}, not above 0), "
            "so it gives no periodic box" + _NO_BOX_REMEDY)
    try:
        cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
    except (RuntimeError, ValueError):
        raise UnsupportedFormat(
            f"the CRYST1 record {record!r} is not a cell gemmi.UnitCell "
            "builds, so it gives no periodic box" + _NO_BOX_REMEDY) from None
    box = np.array(cell.orth.mat).T.copy()
    built = (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
    if not np.isfinite(box).all() or not np.allclose(built, values,
                                                     rtol=1e-12, atol=0):
        raise UnsupportedFormat(
            f"the CRYST1 record {record!r} is not a cell gemmi.UnitCell "
            "builds as written, so it gives no periodic box" + _NO_BOX_REMEDY)
    return box


def _pdb_scale_check(scale: Sequence[str], cryst1: str,
                     box: np.ndarray) -> None:
    """Refuse SCALEn records that put the coordinates in another frame than
    CRYST1's standard orthogonalisation [5].

    ``scale`` holds the SCALEn records after the frame's CRYST1 record. Their
    matrix is the inverse of the CRYST1 box (transposed) up to the rounding
    of both records, so each element may differ from it by its own SCALEn
    rounding (half a unit of the last decimal written, 5e-7 for F10.6) plus
    the rounding of the six CRYST1 values (5e-4 Å, 0.005°) carried to that
    element to first order (finite differences of the orthogonalisation);
    twice that sum is allowed. Measured on 41 120 random cells (lengths 3-80
    Å, angles 40-140°, two seeds) with SCALEn written F10.6 from the
    unrounded cell and CRYST1 rounded: none refused; the same cells rotated
    by 0.1° or 0.5° about x, y or z: every one refused.
    """
    if not scale:
        return
    rows: dict[str, str] = {}
    for record in scale:
        rows.setdefault(record[5:6], record)
    if sorted(rows) != ["1", "2", "3"]:
        raise ValueError(
            f"the SCALE records after the CRYST1 record are SCALE"
            f"{', SCALE'.join(sorted(rows))}, where the format gives SCALE1, "
            "SCALE2 and SCALE3")
    columns = ((10, 20), (20, 30), (30, 40))
    try:
        matrix = np.array([[float(rows[n][a:b]) for a, b in columns]
                           for n in "123"])
        shift = np.array([float(rows[n][45:55]) if rows[n][45:55].strip()
                          else 0.0 for n in "123"])
    except ValueError:
        raise ValueError("a SCALEn record does not hold its matrix row "
                         "(columns 11-40) and shift (46-55) as numbers") \
            from None
    allowed = np.array([[0.5 * 10.0 ** -max(_decimals(
        rows[n][a:b].strip().encode()), 0) for a, b in columns]
        for n in "123"])
    values, steps = _cryst1_values(cryst1)
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, step in enumerate(steps):
            h = step * 1e-3
            up, down = list(values), list(values)
            up[k] += h
            down[k] -= h
            slope = (np.linalg.inv(_standard_box(up).T)
                     - np.linalg.inv(_standard_box(down).T)) / (2.0 * h)
            allowed = allowed + np.abs(slope) * step
    allowed = 2.0 * allowed
    gap = np.abs(matrix - np.linalg.inv(box.T))
    shifted = bool(np.abs(shift).max() > 5e-6)
    if not np.isfinite(allowed).all() or (gap > allowed).any() or shifted:
        worst = np.unravel_index(int(np.argmax(gap / allowed)), gap.shape)
        raise UnsupportedFormat(
            "its SCALEn records put the coordinates in another frame than the "
            "standard orthogonalisation of CRYST1 (a along x, b in the xy "
            f"plane), which FACET reads: SCALE{worst[0] + 1}'s element "
            f"{worst[1] + 1} differs from the inverse of the CRYST1 box by "
            f"{gap[worst]:.3g}, where the rounding of the two records allows "
            f"{allowed[worst]:.2g}"
            + (", and their shift is not 0" if shifted else "")
            + "; write the trajectory without a rotated frame, or as "
            "extended XYZ with Lattice=")


def _pdb_frame_atoms(data: bytes, first_line: int):
    """(labels, positions in Å, label kind) of one frame, read from the
    fixed columns of its ATOM / HETATM records in file order [5]: the atom
    name (13-16), x, y, z (31-38, 39-46, 47-54, Real(8.3)) and the element
    (77-78). A frame needs nothing else, so no residue hierarchy is built:
    gemmi's would file the atoms of a residue number written twice (writers
    keep its last four digits) under its first residue, and reads a field
    that is not a number as 0. ``first_line`` is the file line of the
    data's first line, which errors name."""
    rows = [(i, x) for i, x in enumerate(data.splitlines())
            if x[:4] == b"ATOM" or x[:6] == b"HETATM"]
    if not rows:
        raise ValueError("no ATOM or HETATM records")
    short = next(((i, x) for i, x in rows if len(x.rstrip()) < 54), None)
    if short is not None:
        i, x = short
        raise ValueError(
            f"file line {first_line + i}: the {_decode(x[:6]).strip()} record "
            f"ends at column {len(x.rstrip())}, before the end of its "
            "coordinate columns (31-54)")
    fields = np.array([(x[30:38], x[38:46], x[46:54]) for _, x in rows])
    try:
        cart = fields.astype(np.float64)
        odd = not np.isfinite(cart).all() or \
            bool((np.abs(cart) > _PDB_COORDINATE_MAX).any())
    except ValueError:
        cart, odd = None, True
    if odd:
        for (i, _), triple in zip(rows, fields):
            for axis, field in enumerate(triple):
                value = float(field) if _is_number(field) else None
                if value is not None and np.isfinite(value) and \
                        abs(value) <= _PDB_COORDINATE_MAX:
                    continue
                first = 31 + 8 * axis
                raise ValueError(
                    f"file line {first_line + i}: the {'xyz'[axis]} field "
                    f"(columns {first}-{first + 7}) holds "
                    f"{_decode(field).strip()!r}, which "
                    + ("is not a number" if value is None else
                       "a Real(8.3) coordinate field cannot hold"))
    names = np.array([_decode(x[12:16]).strip() for _, x in rows])
    columns = [_decode(x[76:78]).strip() for _, x in rows]
    if all(columns):
        return np.array(columns), cart, "element column"
    return names, cart, "atom name"


def _same_labels(labels: np.ndarray, reference: np.ndarray, kind: str) -> bool:
    """Element columns name an element in any case [5] (ASE writes 'SI',
    CP2K 'Si'); atom names compare as written."""
    if labels.shape != reference.shape:
        return False
    if kind == "element column":
        return bool(np.array_equal(np.char.upper(labels),
                                   np.char.upper(reference)))
    return bool(np.array_equal(labels, reference))


class _PdbTrajectory(_RangeTrajectory):
    def __init__(self, path, blocks, *, labels, kind, elements,
                 **kwargs) -> None:
        super().__init__(path, [(b.start, b.end) for b in blocks], **kwargs)
        self._blocks = list(blocks)
        self._labels = labels
        self._kind = kind
        self._elements = elements

    def _read_frame(self, k: int) -> Frame:
        block = self._blocks[k]
        labels, cart, kind = _pdb_frame_atoms(self._bytes(k), block.line)
        if kind != self._kind or not _same_labels(labels, self._labels, kind):
            raise ValueError(f"the {self._kind}s differ from frame 0's (other "
                             "atoms, or another order)")
        return frame_from_arrays(self._elements, cart, box_ang=block.box_ang,
                                 timestep=block.timestep,
                                 time_ps=block.time_ps)


def _pdb_elements(path: Path, labels: np.ndarray, kind: str, type_map):
    """Elements from the element columns (an element symbol in any case [5])
    or, when they are blank, from the atom names as labels."""
    if kind == "atom name":
        elements, listed, source, notes = _labels_to_elements(
            path, labels, type_map, what="atom name")
        return elements, listed, source, ["the element columns (77-78) are "
                                          "blank; elements from the atom "
                                          "names"] + notes
    mr = _mr()
    user_types, user_labels = mr._normalise_type_map(type_map)
    out = np.empty(labels.shape, dtype="<U2")
    listed: dict = {}
    sources: list[str] = []
    problems: list[str] = []
    for label in np.unique(labels):
        if label in user_labels:
            out[labels == label] = listed[label] = user_labels[label]
            sources.append("user")
            continue
        try:
            symbol = validate_symbol(str(label))
        except ValueError:
            problems.append(str(label))
            continue
        out[labels == label] = symbol
        sources.append("file symbols")
    if problems:
        example = ", ".join(f"{p!r}: '<element>'" for p in problems[:3])
        raise ValueError(
            f"{path.name}: the element columns (77-78) hold "
            f"{', '.join(repr(p) for p in problems)}, which name no element; "
            f"give a type map from those values to elements, for example "
            f"type_map={{{example}}}")
    notes = mr._numeric_entries_note(user_types, "element column")
    unused = sorted(set(user_labels) - set(str(x) for x in labels))
    if unused:
        notes.append(f"type map entries {unused} are not used: no atom carries "
                     "those element-column values")
    return out, listed, mr._join_sources(sources), notes


def _pdb_count_reason(item: _PdbItem, n0: int, delimiter: str,
                      first_holds: bool, is_last: bool) -> list[str]:
    """The skipped reasons of a run whose atom count is not the model's: one
    per frame it stands for."""
    lines = f"file lines {item.line}-{item.last_line}"
    k = item.atoms // n0
    if item.kind != "tail" and k >= 2 and item.atoms == k * n0:
        if item.kind == "outside":
            why = ("they lie outside any MODEL ... ENDMDL block, with no "
                   "delimiter between them")
        elif delimiter == "MODEL":
            why = "the ENDMDL and MODEL records between them are missing"
        else:
            why = "the END record between them is missing"
        return [f"merged: {item.atoms} atom records ({lines}) are {k} frames "
                f"of {n0} atoms run together, because {why}; none of them is "
                "read"] * k
    if item.kind == "outside":
        return [f"{item.atoms} atom records outside any MODEL ... ENDMDL block "
                f"({lines}), where a frame holds {n0}; they are not read"]
    if item.kind == "tail" and is_last and item.atoms < n0:
        return [f"truncated: the file ends inside this frame, after "
                f"{item.atoms} of {n0} atom records"]
    if first_holds:
        return [f"{item.atoms} atoms, where frame 0 holds {n0}"]
    return [f"{item.atoms} atoms, where most frames hold {n0}"]


def read_pdb_trajectory(path, *, type_map: Mapping | None = None
                        ) -> Trajectory:
    """A PDB file holding one frame per MODEL ... ENDMDL (ASE, MDAnalysis
    and mdtraj write one), or per block of atoms ended by END and no MODEL
    record (CP2K) [5, 8].

    Each frame's box is the last CRYST1 record before the frame ends (one per
    model, or one for the file), in the PDB's standard orthogonalisation;
    SCALEn records after it are checked against it. Coordinates, atom names
    and elements are read from the fixed columns of the ATOM / HETATM
    records, in file order; elements come from the element columns (77-78),
    or from the atom names when those are blank. Atoms keep the file's
    order, numbered 0 .. N-1; occupancies, B-factors, residues and chains
    are not read. CP2K's 'REMARK Step <it>, time = <t>' gives a frame its MD
    step and its time (fs) [8]; other files hold neither.

    The index pass counts the atom records of every frame, so the model's
    atom count is the count most frames hold (:func:`_frame_count`), and a
    block that lost the delimiter between k frames is k skipped positions.
    Atom records outside any MODEL block are read as a frame when they hold
    the model's atom count, and are a skipped position otherwise.
    """
    mr = _mr()
    path = mr._existing(path)
    index = _pdb_index(path)
    items, delimiter = _pdb_items(index)
    if not items:
        raise UnsupportedFormat(f"{path.name}: no MODEL or ATOM records, so no "
                                "frame; FACET reads PDB files of atoms")
    crysts = [r for r in index.records if r.kind == "CRYST1"]
    if not crysts:
        raise UnsupportedFormat(
            f"{path.name}: no CRYST1 record, so the file gives no periodic box. "
            "FACET reads periodic models: write the trajectory with its box "
            "(CRYST1), or as extended XYZ with Lattice=")
    cryst_offsets = [c.offset for c in crysts]
    scales = [r for r in index.records if r.kind == "SCALE"]
    scale_offsets = [r.offset for r in scales]
    remarks = [r for r in index.records if r.kind == "REMARK"]
    remark_offsets = [r.offset for r in remarks]

    for item in items:
        if item.kind == "model" and not item.atoms:
            item.kind, item.reason = "open", (
                f"no ATOM or HETATM records in the MODEL block (file lines "
                f"{item.line}-{item.last_line})")
    voters = [it.atoms for it in items if it.kind in ("model", "end",
                                                      "outside")] \
        or [it.atoms for it in items if it.kind == "tail"]
    if not voters:
        raise UnsupportedFormat(
            f"{path.name}: no MODEL block holds ATOM or HETATM records, so no "
            "frame: " + "; ".join(f"position {p}: {it.reason}"
                                  for p, it in enumerate(items[:5]))
            + "; FACET reads PDB files of atoms")
    n0 = _frame_count(voters)
    first_holds = items[0].kind != "open" and items[0].atoms == n0
    skipped: dict[int, str] = {}
    notes: list[str] = []
    boxes: dict[str, np.ndarray | Exception] = {}
    candidates: list[tuple[_PdbItem, _PdbBlock]] = []
    with_remark = 0
    position = 0
    for i, item in enumerate(items):
        is_last = i + 1 == len(items)
        if item.kind == "open":
            skipped[position] = item.reason
            position += 1
            continue
        if item.atoms != n0:
            for reason in _pdb_count_reason(item, n0, delimiter, first_holds,
                                            is_last):
                skipped[position] = reason
                position += 1
            continue
        at = bisect.bisect_left(cryst_offsets, item.end)
        if at == 0:
            skipped[position] = "no CRYST1 record before or in it"
            position += 1
            continue
        cryst = crysts[at - 1]
        if cryst.text not in boxes:
            try:
                boxes[cryst.text] = _cryst1_box(cryst.text)
            except ValueError as error:     # UnsupportedFormat included
                boxes[cryst.text] = error
        box = boxes[cryst.text]
        lo = bisect.bisect_right(scale_offsets, cryst.offset)
        hi = bisect.bisect_left(scale_offsets, item.end)
        scale = tuple(r.text for r in scales[lo:hi])
        try:
            if isinstance(box, Exception):
                raise box
            _pdb_scale_check(scale, cryst.text, box)
        except ValueError as error:
            if not candidates:
                if isinstance(error, UnsupportedFormat):
                    raise UnsupportedFormat(_named(path, error)) from None
                raise FrameError(f"{path.name}, frame 0 (file position "
                                 f"{position}): {error}") from None
            skipped[position] = f"its CRYST1 or SCALEn records: {error}"
            position += 1
            continue
        timestep = time_ps = None
        lo = bisect.bisect_left(remark_offsets, item.meta)
        hi = bisect.bisect_left(remark_offsets, item.end)
        for remark in remarks[lo:hi]:
            match = _CP2K_REMARK.match(remark.text)
            if match:
                timestep = int(match.group(1))
                if match.group(2) is not None:
                    time_ps = float(match.group(2).replace("d", "e")
                                    .replace("D", "e")) * FS_PS
                with_remark += 1
                break
        candidates.append((item, _PdbBlock(
            position, item.start, item.end, item.line, cryst.text, box, scale,
            timestep, time_ps)))
        if item.kind == "outside":
            notes.append(
                f"frame at file position {position} is a block of {n0} atom "
                f"records outside any MODEL ... ENDMDL block (file lines "
                f"{item.line}-{item.last_line}), read as a frame: it holds the "
                "model's atom count")
        position += 1
    if not candidates:
        raise ValueError(f"{path.name} holds no readable frame: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))

    # Frame 0 gives the atoms; the last frame may be cut.
    item0, block0 = candidates[0]
    try:
        labels0, _, kind0 = _pdb_frame_atoms(
            _read_range(path, block0.start, block0.end), block0.line)
    except ValueError as error:
        raise FrameError(f"{path.name}, frame 0 (file position "
                         f"{block0.position}): {error}") from error
    if len(candidates) > 1:
        item, block = candidates[-1]
        try:
            labels, _, kind = _pdb_frame_atoms(
                _read_range(path, block.start, block.end), block.line)
            if kind != kind0 or not _same_labels(labels, labels0, kind0):
                raise ValueError(f"the {kind0}s differ from frame 0's (other "
                                 "atoms, or another order)")
        except ValueError as error:
            cut = item.kind == "tail" or index.gzip_cut
            skipped[block.position] = ("truncated: " if cut else
                                       "unreadable: ") + str(error)
            candidates.pop()
    blocks = [block for _, block in candidates]
    elements, listed, source, label_notes = _pdb_elements(path, labels0, kind0,
                                                          type_map)
    notes = label_notes + notes
    notes.append(
        ("frames are MODEL ... ENDMDL blocks" if delimiter == "MODEL"
         else "frames are blocks of atoms ended by END records")
        + "; each frame's box is the last CRYST1 before it ends, in the "
        "standard orthogonalisation (a along x, b in the xy plane), with the "
        "origin at (0, 0, 0); coordinates, atom names and elements are read "
        "from their columns, and atoms are numbered 0 .. N-1 in the order of "
        "the file; "
        + (f"the MD step, and the time where it is given, come from the "
           f"'REMARK Step <it>, time = <t>' record CP2K writes before a frame "
           f"(motion_utils.F), the time in fs (x {FS_PS:g} ps)"
           + ("" if with_remark == len(blocks) else
              f"; {len(blocks) - with_remark} frame(s) have no such record")
           if with_remark else "the file holds no time or MD step"))
    box_varies = any(not np.array_equal(b.box_ang, blocks[0].box_ang)
                     for b in blocks[1:])
    if len(blocks) > 1 and len(crysts) == 1:
        notes.append("the file holds one CRYST1 record, which every frame takes "
                     "as its box; a box that changed during the run is not in "
                     "the file (MDAnalysis 2.10 and mdtraj 1.11 write CRYST1 "
                     "once for a multi-frame PDB)")
    if index.gzip_cut:
        notes.append(mr._GZIP_CUT_NOTE)
    times = [b.time_ps for b in blocks]
    return _finish(_PdbTrajectory(
        path, blocks, labels=labels0, kind=kind0, elements=elements,
        file_format="pdb-models", n_atoms=int(labels0.size),
        n_frames=len(blocks), type_map_source=source, type_map=listed,
        timesteps=[NO_TIMESTEP if b.timestep is None else b.timestep
                   for b in blocks],
        times_ps=None if all(t is None for t in times) else
        [np.nan if t is None else t for t in times],
        skipped=skipped, notes=notes,
        units_note="PDB lengths in Å [wwPDB format 3.3]"
        + (f"; CP2K's REMARK times in fs, 1 fs = {FS_PS:g} ps"
           if with_remark else ""),
        box_varies=box_varies))


def is_multiframe_pdb(path, max_bytes: int | None = None) -> bool:
    """True when a PDB file holds more than one structure, by the test
    ``readers.read`` applies to a .pdb file
    (:func:`.md_readers.pdb_model_count` >= 2): two MODEL records, or ATOM
    records after an END record that ended earlier atoms. With
    ``max_bytes``, no further than that many bytes; False for a file that
    cannot be read."""
    mr = _mr()
    if max_bytes is None:
        return mr.pdb_model_count(path) >= 2
    models = sets = read = 0
    in_set = False
    try:
        with mr._open_binary(Path(path)) as handle:
            while max(models, sets) < 2:
                line = mr._readline(handle)
                read += len(line)
                if not line or read > max_bytes:
                    return False
                if line.startswith(b"MODEL"):
                    models += 1
                elif line.startswith((b"ATOM", b"HETATM")):
                    if not in_set:
                        sets += 1
                        in_set = True
                elif _is_end(line):
                    in_set = False
    except (OSError, ValueError, EOFError):
        return False
    return True


def sniff_pdb_trajectory(head: bytes, path: Path) -> bool:
    """A PDB head (its first record is a PDB record name) of a file holding
    more than one structure, decided as ``readers.read`` decides it for a
    .pdb file (:func:`.md_readers.pdb_model_count`, which reads lines until
    it finds a second structure or the file ends): a one-frame PDB stays
    with the crystal reader, and readers.read never routes a file that this
    sniff then refuses."""
    try:
        lines = [x for x in head.splitlines() if x.strip()]
        if not lines or not lines[0].startswith(_PDB_RECORDS):
            return False
        return _mr().pdb_model_count(path) >= 2
    except Exception:           # noqa: BLE001 - a sniffer never raises
        return False


# ---------------------------------------------------------------------------
# IMD
# ---------------------------------------------------------------------------

@dataclass
class _ImdHeader:
    columns: tuple[str, ...]
    box: np.ndarray
    body_offset: int
    format_line: str


def _imd_header(path: Path) -> _ImdHeader:
    """The header lines up to '#E' [6]: '#F A ...', '#C' column names,
    '#X' '#Y' '#Z' the cell vectors, '##' comments."""
    mr = _mr()
    columns: tuple[str, ...] | None = None
    vectors: dict[str, list[float]] = {}
    format_line = ""
    offset = 0
    with mr._open_binary(path) as handle:
        first = True
        while True:
            raw = mr._readline(handle)
            if not raw:
                raise UnsupportedFormat(
                    f"{path.name}: the IMD header has no '#E' line, so the atom "
                    "lines cannot be found; FACET reads IMD files as OVITO "
                    "writes them ('#F A', '#C', '#X', '#Y', '#Z', '#E')")
            offset += len(raw)
            line = mr._strip_bom(raw) if first else raw
            text = _decode(line).strip()
            if first:
                first = False
                tokens = text.split()
                if not text.startswith("#F") or len(tokens) < 2:
                    raise UnsupportedFormat(
                        f"{path.name}: the first line is not an IMD '#F' line, "
                        "so not an IMD configuration file; FACET reads IMD "
                        "files that start with '#F A'")
                if tokens[1] != "A":
                    raise UnsupportedFormat(
                        f"{path.name}: the '#F' line declares the format "
                        f"{tokens[1]!r}, a binary IMD layout; FACET reads the "
                        "ASCII layout ('#F A'), which OVITO writes when it "
                        "exports IMD")
                format_line = text
                continue
            if not text.startswith("#"):
                raise UnsupportedFormat(
                    f"{path.name}: header line {text[:40]!r} does not start "
                    "with '#' before the '#E' line; FACET reads IMD headers "
                    "as OVITO writes them")
            key = text[1:2]
            if key == "#":
                continue
            if key == "E":
                break
            if key == "C":
                columns = tuple(text[2:].split())
            elif key in "XYZ" and key:
                values = text[2:].split()
                if len(values) != 3 or not all(_is_number(v) for v in values):
                    raise UnsupportedFormat(
                        f"{path.name}: the '#{key}' line {text[:60]!r} does not "
                        "hold three numbers (a cell vector)")
                vectors[key] = [float(v) for v in values]
            else:
                raise UnsupportedFormat(
                    f"{path.name}: the header line {text[:40]!r} has a key "
                    "FACET does not know; IMD headers hold #F, #C, #X, #Y, #Z, "
                    "## and #E")
    if columns is None:
        raise UnsupportedFormat(
            f"{path.name}: no '#C' line names the columns; FACET reads IMD "
            "files whose '#C' line names them (OVITO writes one)")
    missing = [c for c in ("x", "y", "z") if c not in columns]
    if missing:
        raise UnsupportedFormat(f"{path.name}: the '#C' columns {list(columns)} "
                                f"have no {', '.join(missing)} column")
    if set(vectors) != {"X", "Y", "Z"}:
        raise UnsupportedFormat(
            f"{path.name}: the header gives the cell vectors "
            f"{sorted(vectors) or 'none'}, not all of #X, #Y and #Z, so the "
            "file gives no periodic box")
    box = np.array([vectors["X"], vectors["Y"], vectors["Z"]])
    if not np.isfinite(box).all():
        raise UnsupportedFormat(f"{path.name}: a cell vector holds a value that "
                                "is not finite")
    md_model._checked_box(box)
    return _ImdHeader(columns, box, offset, format_line)


def _imd_table(path: Path, header: _ImdHeader) -> np.ndarray:
    """The atom lines as an (n, columns) array of text tokens."""
    data = _mr()._read_all(path)[header.body_offset:]
    lines = [x for x in data.splitlines() if x.strip()]
    if not lines:
        raise ValueError("no atom lines after '#E'")
    if not data.rstrip(b" \t").endswith((b"\n", b"\r")):
        raise ValueError("the file does not end with a line ending, so the last "
                         "value of its last atom line may be cut")
    tokens = b" ".join(lines).split()
    width = len(header.columns)
    if len(tokens) != width * len(lines):
        odd = next(i for i, x in enumerate(lines) if len(x.split()) != width)
        raise ValueError(f"atom line {odd + 1} after '#E' holds "
                         f"{len(lines[odd].split())} values where '#C' names "
                         f"{width} columns")
    return np.array(tokens).reshape(len(lines), width)


def _imd_atoms(path: Path, header: _ImdHeader):
    """(ids or None, types, masses or None, positions) of one IMD file."""
    table = _imd_table(path, header)
    col = {name: i for i, name in enumerate(header.columns)}
    cart = _floats(table[:, [col["x"], col["y"], col["z"]]], "a position "
                   "column")
    ids = None
    if "number" in col:
        values = _floats(table[:, col["number"]], "the number column")
        if (values != np.round(values)).any():
            raise ValueError("the number column holds values that are not whole "
                             "numbers")
        ids = values.astype(np.int64)
    if "type" not in col:
        raise ValueError("the '#C' line names no type column, so no atom has "
                         "a type")
    types_f = _floats(table[:, col["type"]], "the type column")
    if (types_f != np.round(types_f)).any():
        raise ValueError("the type column holds values that are not whole "
                         "numbers")
    masses = _floats(table[:, col["mass"]], "the mass column") \
        if "mass" in col else None
    return ids, types_f.astype(np.int64), masses, cart


class _ImdSeries(_SeriesTrajectory):
    def __init__(self, paths, all_paths, *, headers, type_elements,
                 **kwargs) -> None:
        super().__init__(paths, all_paths, **kwargs)
        self._headers = list(headers)
        self._type_elements = dict(type_elements)

    def _read_frame(self, k: int) -> Frame:
        path, header = self._paths[k], self._headers[k]
        ids, types, _, cart = _imd_atoms(path, header)
        if cart.shape[0] != self.n_atoms:
            raise ValueError(f"{cart.shape[0]} atoms; the first file of the "
                             f"series holds {self.n_atoms}")
        unknown = sorted(set(types.tolist()) - set(self._type_elements))
        if unknown:
            raise ValueError(f"{path.name}: type(s) {unknown} have no element "
                             "(they do not occur in the first file)")
        elements = np.array([self._type_elements[t] for t in types.tolist()])
        return frame_from_arrays(elements, cart, box_ang=header.box,
                                 atom_id=ids)


def _imd_type_map(path: Path, types: np.ndarray, masses, type_map,
                  mass_tol_amu):
    """Type number -> element: the user's map, else each type's mass."""
    mr = _mr()
    user_types, user_labels = mr._normalise_type_map(type_map)
    present = sorted(set(types.tolist()))
    type_masses = None
    notes: list[str] = []
    if masses is not None:
        type_masses = {}
        for t in present:
            values = np.unique(masses[types == t])
            if values.size == 1:
                type_masses[t] = float(values[0])
            else:
                notes.append(f"type {t}: the mass column holds {values.size} "
                             f"masses ({values.min():.6g} to {values.max():.6g} "
                             "amu), so the masses do not name its element")
    tol = mr.MASS_TOL_AMU if mass_tol_amu is None else mr._checked_tol(
        mass_tol_amu)
    example = ", ".join(f"{t}: '<element>'" for t in present[:6])
    how_to = (f"Give a type map from this file's type numbers to elements, "
              f"for example type_map={{{example}}}")
    try:
        chosen, source, type_notes = mr._resolve_types(
            present, user=user_types, candidates=[], masses=type_masses,
            tol=tol, how_to=how_to, absent="no atom has those types")
    except ValueError as error:
        raise ValueError(_named(path, error)) from None
    type_notes = [x.replace("its data-file mass", "its mass-column mass")
                  for x in type_notes]
    if user_labels:
        notes.append(f"type map entries {sorted(user_labels)} are labels; IMD "
                     "types are numbers, so they are not used")
    return chosen, source, notes + type_notes


def read_imd_series(paths, *, type_map: Mapping | None = None,
                    mass_tol_amu: float | None = None) -> Trajectory:
    """IMD configuration files (ASCII, '#F A'), one frame per file, in the
    order given [6].

    The '#C' line names the columns: ``number`` (the atom id), ``type`` (a
    number), ``mass`` (amu), ``x y z``; velocity and other columns are listed
    and not read. ``#X``, ``#Y``, ``#Z`` are the cell vectors a, b, c; the
    file holds no origin. Elements: ``type_map`` (type number -> element),
    else the mass of each type. A file whose columns or cell vectors cannot be
    read, or whose atoms differ from the first file's, is skipped with the
    reason.
    """
    all_paths = _series_paths(paths, "IMD")
    first = all_paths[0]
    header0 = _imd_header(first)
    try:
        ids0, types0, masses0, cart0 = _imd_atoms(first, header0)
    except ValueError as error:
        raise FrameError(f"{first.name}, frame 0 (file position 0): "
                         f"{error}") from error
    type_elements, source, notes = _imd_type_map(first, types0, masses0,
                                                 type_map, mass_tol_amu)
    readable, headers, skipped = [first], [header0], {}
    for position, path in enumerate(all_paths[1:], start=1):
        try:
            header = _imd_header(path)
        except UnsupportedFormat as error:
            skipped[position] = f"unreadable: {error}"
            continue
        if header.columns != header0.columns:
            skipped[position] = (f"{path.name}: its columns {list(header.columns)} "
                                 f"differ from the first file's "
                                 f"{list(header0.columns)}")
            continue
        readable.append(path)
        headers.append(header)
    others = [c for c in header0.columns
              if c not in ("number", "type", "mass", "x", "y", "z")]
    if others:
        notes.append("not read: the columns " + ", ".join(others)
                     + (" (velocities: the IMD format gives them no unit)"
                        if {"vx", "vy", "vz"} & set(others) else ""))
    if ids0 is None:
        notes.append("no number column: atoms are numbered 0 .. N-1 in the "
                     "order of the file")
    notes.append("IMD headers give the cell vectors (#X #Y #Z) and no origin, "
                 "so the origin is (0, 0, 0) and positions are wrapped into "
                 "that box; the file holds no time or MD step")
    if len(all_paths) > 1:
        notes.append(f"a series of {len(all_paths)} files, {all_paths[0].name} "
                     f"to {all_paths[-1].name}, one frame each")
        order = _name_order_note(all_paths)
        if order:
            notes.append(order)
    box_varies = any(not np.array_equal(h.box, header0.box)
                     for h in headers[1:])
    return _finish(_ImdSeries(
        readable, all_paths, headers=headers, type_elements=type_elements,
        file_format="imd", n_atoms=int(cart0.shape[0]),
        n_frames=len(readable), type_map_source=source,
        type_map=dict(type_elements), skipped=skipped, notes=notes,
        units_note=("the IMD format states no unit; positions and cell "
                    "vectors are kept as written and read as Å"),
        box_varies=box_varies))


def read_imd(path, *, type_map: Mapping | None = None,
             mass_tol_amu: float | None = None) -> Trajectory:
    """One IMD configuration file as a one-frame trajectory
    (:func:`read_imd_series`)."""
    return read_imd_series([path], type_map=type_map,
                           mass_tol_amu=mass_tol_amu)


def sniff_imd(head: bytes, path: Path) -> bool:
    """An IMD head: the first line starts with '#F ' (OVITO's test [6])."""
    try:
        return _mr()._strip_bom(head).startswith(b"#F ")
    except Exception:           # noqa: BLE001 - a sniffer never raises
        return False


# ---------------------------------------------------------------------------
# a series of POSCAR files
# ---------------------------------------------------------------------------

@dataclass
class _PoscarHeader:
    header: object              # md_readers._VaspHeader
    species: tuple[str, ...]
    counts: tuple[int, ...]
    cartesian: bool
    rows_from: int              # 0-based line index of the first position
    notes: tuple[str, ...]
    vasp4: bool = False         # element names taken from the title


def _poscar_lines(path: Path, count: int) -> list[str]:
    """The first ``count`` lines, decoded, a byte-order mark removed; read
    line by line, because ASE writes one species group per run of atoms, so
    the name and count lines can be longer than any fixed head."""
    mr = _mr()
    out: list[str] = []
    with mr._open_binary(path) as handle:
        for _ in range(count):
            line = mr._readline(handle)
            if not line:
                break
            out.append(_decode(mr._strip_bom(line) if not out else line)
                       .rstrip("\r\n"))
    return out


def _poscar_header(path: Path) -> _PoscarHeader:
    """The POSCAR header [7]: title, scale, three vectors, element names
    (VASP 5; VASP 4 takes them from the title), counts, optionally 'Selective
    dynamics', then Direct or Cartesian."""
    mr = _mr()
    lines = _poscar_lines(path, 9)
    if len(lines) < 8:
        raise ValueError(f"{path.name}: {len(lines)} lines, fewer than a POSCAR "
                         "header and one position")
    notes: list[str] = []
    names = lines[5].split()
    if names and all(_is_number(t) for t in names):
        title = lines[0].split()[:len(names)]
        if len(title) == len(names) and all(
                mr._label_problem(t) is None for t in title):
            lines = lines[:5] + [" ".join(title)] + lines[5:]
            notes.append("VASP 4 layout (no element-name line): the element "
                         f"names {title} are taken from the title line, as "
                         "facet.core.readers.read_poscar does")
        else:
            raise ValueError(
                f"{path.name}: line 6 holds the atom counts and no element "
                "names (the VASP 4 layout), and the title line does not list "
                "one element symbol per count; FACET reads the VASP 5 layout: "
                "add a line of element names before the counts")
    try:
        header = mr._vasp_header(lines[:7])
    except ValueError as error:
        raise ValueError(_named(path, error)) from None
    cursor = 7
    if lines[cursor].strip()[:1] in ("s", "S"):
        cursor += 1
        notes.append("selective-dynamics flags are not read")
    if cursor >= len(lines):
        raise ValueError(f"{path.name}: the file ends before the Direct / "
                         "Cartesian line")
    mode = lines[cursor].strip()[:1]
    vasp4 = bool(notes) and notes[0].startswith("VASP 4")
    # rows_from counts lines of the file: the VASP 4 title line inserted
    # above as a name line is not one of them.
    return _PoscarHeader(header, tuple(header.species), tuple(header.counts),
                         mode in ("c", "C", "k", "K"),
                         cursor + 1 - (1 if vasp4 else 0), tuple(notes), vasp4)


def _poscar_coords(path: Path, info: _PoscarHeader):
    """(positions as written, rows split at a sign) of one POSCAR file."""
    mr = _mr()
    raw = mr._strip_bom(mr._read_all(path)).splitlines()
    n = sum(info.counts)
    start = info.rows_from
    rows = raw[start:start + n]
    if len(rows) < n or any(not x.strip() for x in rows):
        got = sum(1 for x in rows if x.strip())
        raise ValueError(f"{got} position lines where the counts give {n}")
    return mr._coordinate_rows(rows)


class _PoscarSeries(_SeriesTrajectory):
    def __init__(self, paths, all_paths, *, headers, elements,
                 **kwargs) -> None:
        super().__init__(paths, all_paths, **kwargs)
        self._headers = list(headers)
        self._elements = elements

    def _read_frame(self, k: int) -> Frame:
        info = self._headers[k]
        coords, fused = _poscar_coords(self._paths[k], info)
        notes = [f"{fused} position line(s) hold numbers with no blank between "
                 "them (fixed-width fields); they were split at each sign"] \
            if fused else []
        if info.cartesian:
            return frame_from_arrays(self._elements,
                                     coords * info.header.cart_scale,
                                     box_ang=info.header.box, notes=notes)
        return frame_from_arrays(self._elements, frac=coords,
                                 box_ang=info.header.box, notes=notes)


def _refuse_run_ends(paths: Sequence[Path]) -> None:
    """UnsupportedFormat when a directory's CONTCAR comes before its POSCAR.

    VASP reads a run's start from POSCAR and writes its end to CONTCAR [7].
    A directory or a wildcard pattern lists them in name order, CONTCAR
    first (and read_trajectory, given a VASP run directory, picks them over
    its one XDATCAR because they are two files), so as a series their two
    frames would run backwards in time. Given in time order, as a list
    [POSCAR, CONTCAR], they are read.
    """
    place = {(str(p.resolve().parent), p.name): i for i, p in enumerate(paths)
             if p.name in ("POSCAR", "CONTCAR")}
    for (folder, name), i in place.items():
        if name != "CONTCAR" or place.get((folder, "POSCAR"), -1) < i:
            continue
        xdatcar = Path(folder) / "XDATCAR"
        raise UnsupportedFormat(
            f"{Path(folder).name}: its CONTCAR and POSCAR are the end and the "
            "start of one VASP run, and they come here in that order, so as "
            "a series their two frames would run backwards in time. "
            + (f"The run's trajectory is its XDATCAR ({xdatcar}), which "
               "FACET reads: pass that file. " if xdatcar.is_file() else
               "The run's trajectory is the XDATCAR VASP writes beside them, "
               "which FACET reads. ")
            + "To read the two structures as two frames, pass them as a list "
            "in time order, [POSCAR, CONTCAR]")


def read_poscar_series(paths, *, type_map: Mapping | None = None
                       ) -> Trajectory:
    """POSCAR / CONTCAR files, one frame per file, in the order given [7].

    Each file is parsed as :func:`.md_readers.read_xdatcar` parses a VASP
    header (scale factor, negative volume, three scales) and as
    ``readers.read_poscar`` handles selective dynamics, Direct / Cartesian and
    the VASP 4 layout (element names from the title), without a symmetry
    search. Every file has to list the same element names and counts as the
    first; one that does not is skipped with the difference. Atoms are
    numbered 0 .. N-1 in the order of the counts. Lines after the positions
    (a CONTCAR's velocities) are not read.
    """
    mr = _mr()
    all_paths = _series_paths(paths, "POSCAR")
    _refuse_run_ends(all_paths)
    first = all_paths[0]
    info0 = _poscar_header(first)
    try:
        _poscar_coords(first, info0)
    except ValueError as error:
        raise FrameError(f"{first.name}, frame 0 (file position 0): "
                         f"{error}") from error
    readable, headers, skipped = [first], [info0], {}
    for position, path in enumerate(all_paths[1:], start=1):
        try:
            info = _poscar_header(path)
        except ValueError as error:
            skipped[position] = f"unreadable: {error}"
            continue
        if info.species != info0.species or info.counts != info0.counts:
            skipped[position] = (f"{path.name} lists {list(info.species)} "
                                 f"{list(info.counts)}; the first file lists "
                                 f"{list(info0.species)} {list(info0.counts)}")
            continue
        readable.append(path)
        headers.append(info)
    tokens = [s for s, c in zip(info0.species, info0.counts) for _ in range(c)]
    elements, listed, source, notes = _labels_to_elements(
        first, np.array(tokens), type_map, what="species",
        reduce=mr._vasp_species)
    for note in info0.notes:
        notes.append(note)
    notes.append("atoms are numbered 0 .. N-1 in the order of the element "
                 f"counts; {info0.header.scale_note}; a POSCAR holds no time "
                 "or MD step; lines after the positions (velocities of a "
                 "CONTCAR) are not read")
    if len(all_paths) > 1:
        notes.append(f"a series of {len(all_paths)} files, {all_paths[0].name} "
                     f"to {all_paths[-1].name}, one frame each")
        order = _name_order_note(all_paths)
        if order:
            notes.append(order)
    box_varies = any(not np.array_equal(h.header.box, info0.header.box)
                     for h in headers[1:])
    return _finish(_PoscarSeries(
        readable, all_paths, headers=headers, elements=elements,
        file_format="vasp-poscar", n_atoms=len(tokens), n_frames=len(readable),
        type_map_source=source, type_map=listed, skipped=skipped, notes=notes,
        units_note="VASP lengths in Å", box_varies=box_varies))


def read_poscar_file(path, *, type_map: Mapping | None = None) -> Trajectory:
    """One POSCAR or CONTCAR as a one-frame trajectory
    (:func:`read_poscar_series`)."""
    return read_poscar_series([path], type_map=type_map)


def sniff_poscar(head: bytes, path: Path) -> bool:
    """Whether a head is a POSCAR [7]: a scale line, three vector lines,
    element names and counts (VASP 5), or counts alone with one element
    symbol per count on the title line (VASP 4, read as
    :func:`_poscar_header` reads it), then (after an optional 'Selective
    dynamics' line) Direct or Cartesian, and no XDATCAR 'configuration='
    line, whatever the file's name (:func:`sniff_poscar_series` adds the
    name, see ROUTING, and reads a header longer than the window)."""
    try:
        mr = _mr()
        raw = mr._strip_bom(head)
        lines = mr._lines(raw)
        if len(raw) >= mr.SNIFF_BYTES - len(mr._BOM) and not raw.endswith(
                (b"\n", b"\r")):
            lines = lines[:-1]          # the window cuts the last line
        if len(lines) < 7 or any("configuration=" in x for x in lines[:9]):
            return False
        scale = lines[1].split()
        if len(scale) not in (1, 3) or not all(_is_number(t) for t in scale):
            return False
        if not all(len(lines[i].split()) >= 3 and all(
                _is_number(t) for t in lines[i].split()[:3]) for i in (2, 3, 4)):
            return False
        names, counts = lines[5].split(), lines[6].split()
        if names and all(t.isdigit() for t in names):
            # VASP 4: line 6 holds the counts, the title the names.
            title = lines[0].split()[:len(names)]
            if len(title) != len(names) or any(
                    mr._label_problem(t) is not None for t in title):
                return False
            cursor = 6
        elif not names or any(_is_number(t) for t in names) or \
                len(counts) != len(names) or not all(t.isdigit() for t in counts):
            return False
        else:
            cursor = 7
        if cursor < len(lines) and lines[cursor].strip()[:1] in ("s", "S"):
            cursor += 1
        return cursor < len(lines) and \
            lines[cursor].strip()[:1] in ("d", "D", "c", "C", "k", "K")
    except Exception:           # noqa: BLE001 - a sniffer never raises
        return False


# The names readers.read gives to its POSCAR reader: these extensions, and
# names starting with POSCAR or CONTCAR in any case. readers.read consults no
# format module's sniff for such a name, so a POSCAR under one stays a crystal
# structure there, while read_trajectory can take it as a frame of a series.
_POSCAR_SUFFIXES = (".vasp", ".poscar", ".contcar")
_POSCAR_STEMS = ("POSCAR", "CONTCAR")


def _poscar_named(path: Path) -> bool:
    """A POSCAR name, a trailing '.gz' set aside (the content of a gzip file
    is read decompressed, whatever its name)."""
    name = Path(path).name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    return Path(name).suffix.lower() in _POSCAR_SUFFIXES or \
        name.upper().startswith(_POSCAR_STEMS)


def sniff_poscar_series(head: bytes, path: Path) -> bool:
    """The POSCAR format's sniff: POSCAR content (:func:`sniff_poscar`) under
    a name readers.read gives its POSCAR reader (see ROUTING). The same
    content under any other name is left to readers.read, which reads it as a
    crystal structure."""
    try:
        if not _poscar_named(path):
            return False
        mr = _mr()
        raw = mr._strip_bom(head)
        if len(raw) >= mr.SNIFF_BYTES - len(mr._BOM) and \
                len(raw.splitlines()) < 10:
            # ASE writes the element symbols of every run of atoms on the
            # title, name and count lines, so the header of a model in random
            # order runs past the window: its first 9 lines are read (8 for
            # a file that holds no more), as md_readers._vasp_long_header
            # reads an XDATCAR's, for POSCAR names only.
            first = mr._first_lines(Path(path), 9, mr._VASP_HEADER_MAX_BYTES) \
                or mr._first_lines(Path(path), 8, mr._VASP_HEADER_MAX_BYTES)
            if first is not None:
                head = ("\n".join(first) + "\n").encode("utf-8")
        return sniff_poscar(head, path)
    except Exception:           # noqa: BLE001 - a sniffer never raises
        return False


# ---------------------------------------------------------------------------
# the formats this module adds (md_formats_base.FORMAT_MODULES)
# ---------------------------------------------------------------------------

# No format here claims a file name (extensions, stems): read_trajectory
# recognises them by content alone, so a name claim could only route to the
# MD reader a file that readers.read's crystal sniffing reads and
# read_trajectory refuses (a CIF or a POSCAR saved as .gro, .axsf or .imd).
FORMATS = (
    FormatSpec(
        name="castep-md",
        description="CASTEP .md (or .geom) trajectory, Hartree atomic units",
        extensions=(), stems=(), sniff=sniff_castep_md, read=read_castep_md,
        options=frozenset({"type_map"})),
    FormatSpec(
        name="gromacs-gro",
        description="GROMACS .gro, one or more frames, nm",
        extensions=(), stems=(), sniff=sniff_gro, read=read_gro,
        options=frozenset({"type_map"})),
    FormatSpec(
        name="xsf",
        description="XCrySDen XSF, animated (ANIMSTEPS) or one CRYSTAL",
        extensions=(), stems=(), sniff=sniff_xsf, read=read_xsf,
        options=frozenset({"type_map"})),
    FormatSpec(
        name="pdb-models",
        description="PDB trajectory: one MODEL (or END-terminated block) per "
                    "frame, CRYST1 box",
        extensions=(), stems=(), sniff=sniff_pdb_trajectory,
        read=read_pdb_trajectory, options=frozenset({"type_map"})),
    FormatSpec(
        name="imd",
        description="IMD configuration (ASCII, as OVITO writes it)",
        extensions=(), stems=(), sniff=sniff_imd, read=read_imd,
        options=frozenset({"type_map", "mass_tol_amu"}),
        read_series=read_imd_series),
    FormatSpec(
        name="vasp-poscar",
        description="a series of VASP POSCAR / CONTCAR files, one frame each",
        extensions=(), stems=(), sniff=sniff_poscar_series,
        read=read_poscar_file,
        options=frozenset({"type_map"}),
        read_series=read_poscar_series),
)
