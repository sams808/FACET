"""LAMMPS's other trajectory formats: DCD, binary dump, YAML dump, AtomEye CFG.

WHY A MODULE OF ITS OWN
-----------------------
:mod:`.md_readers` reads the text dump and the data file. A LAMMPS run of an
oxide glass also leaves trajectories in four other layouts, and the
recognition test of the MD readers (115 files written by LAMMPS 22 Jul 2025,
ASE 3.29, OVITO 3.16 and pymatgen) found every one of them refused with the
same generic message:

* ``dump dcd`` -- the CHARMM / NAMD / VMD binary trajectory, the usual input
  of VMD and MDAnalysis. Positions only: no element, type or id.
* ``dump custom`` / ``dump atom`` to a ``*.bin`` file -- the binary dump that
  large runs write, otherwise converted with LAMMPS's ``tools/binary2txt``.
* ``dump yaml`` -- one YAML document per snapshot.
* ``dump cfg`` -- AtomEye's extended CFG, one file per snapshot (LAMMPS
  refuses a single file); ASE writes the same format.

Each is read here into the same :class:`~.md_model.Trajectory` the other
readers return, frame by frame with :func:`~.md_model.frame_from_arrays`, and
listed in :data:`FORMATS` (:mod:`.md_formats_base` describes the contract).
Binary files are read with numpy and :mod:`struct`, YAML with the standard
library: PyYAML is not a dependency, and LAMMPS writes one fixed flow-style
layout that a short parser follows exactly.

WHAT EACH FILE HOLDS, AND WHERE THAT IS WRITTEN DOWN
----------------------------------------------------
**DCD** [1, 2, 3]. Fortran unformatted records: each record is a length
marker, the payload, the same marker. The byte order follows from the first
marker, which is 84: MDAnalysis's ``readdcd.h`` (from VMD's dcdplugin) [3]
swaps it when it is not; markers written as int64, which some Fortran
compilers write, are recognised here the same way (84 as an int64). Record 1
is ``CORD`` and 20 int32 ``ICNTRL``: [0] NSET frames, [1] ISTART, [2] NSAVC,
[3] the timestep of the last frame (LAMMPS) or NSTEP (CHARMM), [8] NAMNF
fixed atoms, [9] DELTA as a float32 (a float64 over [9..10] when [19] is 0,
the X-PLOR layout), [10] non-zero when each frame has a unit-cell record,
[11] 1 for a fourth coordinate, [19] the CHARMM version (LAMMPS writes 24;
readdcd.h reads [10] and [11] only when it is non-zero). Record 2 holds
titles (int32 count, 80 characters each; LAMMPS's first is ``Written by
LAMMPS``), record 3 NATOM. Each frame is an optional unit-cell record of six
float64, then X, Y and Z records of NATOM float32.

* *The unit cell.* LAMMPS writes ``[a, cos gamma, b, cos beta, cos alpha,
  c]``: lengths and angle cosines (``DumpDCD::write_header`` [1]: ``dim[1] =
  gamma = cosine of angle between a and b`` ...; an orthogonal box writes 0
  for the three cosines). NAMD 2.5 wrote degrees in the same slots; every
  angle slot within (1, 180) is read as degrees, as MDAnalysis's DCD reader
  (after VMD's dcdplugin) does [4]. Newer CHARMM versions (at least since
  c36b2, per [4]) write the symmetric box matrix there instead, which matches
  the cosine reading only for an orthogonal box: a non-LAMMPS file with
  non-zero cosines carries a note, and slots that fit neither reading are
  refused. MDAnalysis writes a zeroed cell for a trajectory without
  dimensions [15]: lengths 0, 0, 0 are read as no unit cell, and
  ``box_from`` gives the box. The box is rebuilt with a along x and b in
  the xy plane, the frame LAMMPS's positions are in. A DCD has **no origin**:
  positions are read relative to (0, 0, 0), so each atom keeps its position
  modulo the box vectors and the cell shows another set of images than
  LAMMPS's [xlo, xhi) box.
* *Time.* LAMMPS writes ISTART = the first dumped timestep, NSAVC = the dump
  interval, ICNTRL[3] = the step of the last frame written (updated after
  every frame) and DELTA = ``update->dt`` in the run's time unit (ps under
  units metal, fs under real) [1]; frame k is timestep ISTART + k NSAVC, at
  time timestep x DELTA, the reading MDAnalysis also uses [4]. With
  ``dump_modify first yes`` the first snapshot falls on the step the run
  starts from and the later ones on multiples of NSAVC (Output::setup [14]),
  so ISTART is not a multiple of NSAVC and ISTART + k NSAVC misses every
  later frame (125, 145 for 120, 140); the layout whose last step is
  ICNTRL[3] is used, and when neither is, the frames get no timestep. LAMMPS
  refuses ``reset_timestep`` and ``dump_modify every`` while a DCD dump is
  open [14, 1], so no other layout occurs. DELTA is a float32,
  so the shortest decimal that rounds to it is used (0.001 rather than
  0.0010000000474974513). CHARMM and NAMD write DELTA in AKMA time units [4]:
  sqrt(amu Å^2 / (kcal/mol)), whose inverse in ps^-1 is
  sqrt(4184 J/kcal x 1e20 Å^2/m^2 x 1000 g/kg) x 1e-12 [5], 4.888821e-14 s;
  :data:`AKMA_TIME_PS` computes it from :mod:`scipy.constants` (the molar
  mass constant ``m_u N_A`` and the thermochemical calorie). ``timestep_fs``
  replaces DELTA. The time assumes DELTA was the timestep size since step 0
  (a DCD holds one DELTA; dump yaml or a custom dump with ``dump_modify time
  yes`` records each frame's time). VMD's molfile DCD plugin, which mdtraj
  uses too, writes ISTART 0, NSAVC 1 and DELTA 1.0 for every file
  (``open_dcd_write`` [13]) under the title ``Created by DCD plugin``: such a
  file has no timestep and no time, and ``timestep_fs`` then gives the time
  between frames.
* *Damage.* Every frame's record markers and unit cell are read when the
  file is opened (a memory map touches only those bytes; a gzip file is
  decompressed once), so a frame whose markers or cell do not fit is in
  ``skipped`` with its reason, and a record length larger than the file is
  read in bounded pieces, never allocated whole.
* *Elements.* None in the file, so ``topology=`` names an MD file of the same
  atoms -- the LAMMPS data file the run read -- read with
  :func:`.md_readers.read_trajectory` (``type_map``, ``atom_style``,
  ``masses_from``, ``units`` and ``mass_tol_amu`` go to it). LAMMPS writes a
  DCD sorted by atom id (``init_style`` refuses any other order [1]), so row
  k is the topology's k-th atom in id order. A DCD of a group holds fewer
  atoms than the topology and is refused: which atoms it holds is not in it.
* *Wrapped or unwrapped.* ``dump_modify unwrap yes`` writes x + image @ box
  (``DumpDCD::pack`` [1]); nothing in the file says so. Up to 64 frames are
  sampled when the file is opened: an atom set spanning more than 1.5 box
  lengths along a box vector (a wrapped file spans at most one length plus
  the distance atoms move between re-neighbourings) marks the positions
  unwrapped, and they are kept as ``unwrapped_cart_ang``; an atom whose
  fractional coordinate changes by more than 0.5 between consecutive frames
  (a jump across the box, which continuous positions never make) marks them
  wrapped. The note says which, and when neither shows, that it is not
  known and no unwrapped positions are set.
* *Precision.* float32 keeps 24 significant bits: a position x is stored to
  within 2^-24 |x| (half a unit in the last place), 1.8e-6 Å at 30 Å.

**LAMMPS binary dump** [6, 7, 8]. Per frame, in the writing machine's byte
order: since the magic-string layout (``DumpCustom::header_binary`` [6],
``DumpAtom`` alike), int64 -len, the magic string ``DUMPCUSTOM`` or
``DUMPATOM``, int32 endian flag 1, int32 revision; then (both layouts) int64
timestep, int64 atom count, int32 triclinic flag (0, 1, or 2 for a general
triclinic box), int32 boundary[3][2] (0 p, 1 f, 2 s, 3 m), the box (6, 9 or
12 float64: bounds as a text dump writes them, plus xy xz yz; or avec bvec
cvec and the origin), int32 column count; from revision 2: int32 length and
the unit style (first frame, with ``units yes``), a char time flag and a
float64 time, int32 length and the column names; then int32 chunk count and
per chunk int32 n and n float64 (one chunk per writing processor).
``tools/binary2txt.cpp`` [8] reads both layouts this way, and so does this
module. A file of the older layout (no magic string) or of revision 1 names
no columns: ``columns=`` gives them, the list the dump command named; for a
file that names them, ``columns=`` replaces the names one for one (a dump
written with ``dump_modify colname`` [16], whose renamed columns FACET does
not know). Types are numbers: ``element`` and ``typelabel`` columns hold the
type number in a binary file (``pack_type`` fills them [7]; the names are
written only to text dumps), so they are read as types. A frame of 0 atoms
(``dump_modify thresh``, an empty group) has a header and one empty chunk;
it is in ``skipped`` and the frames after it are read. In the magic-string
layout a header that does not parse, chunks whose counts disagree with it,
or a length past the end of the file is passed over to the next frame's
magic string, the stretch taking one position in ``skipped`` with its byte
range; the older layout has no marker, so reading stops there. A gzip
stream cut inside a chunk ends the frames at the cut frame, which is listed
as truncated, and a stream without its end-of-stream marker is noted.

**LAMMPS YAML dump** [9, 10]. Per frame: ``---``, ``creator: LAMMPS``,
``timestep:``, ``units:`` and ``time:`` (``dump_modify units/time yes``),
``natoms:``, ``boundary: [ p, p, ... ]`` (six flags), optional ``thermo:``
(``dump_modify thermo yes``; not read), ``box:`` with three ``[ lo, hi ]``
rows (a text dump's bounds) and a fourth ``[ xy, xz, yz ]`` for a tilted box,
``keywords: [ ... ]``, ``data:`` and one ``  - [ v, v, ..., ]`` row per atom,
then ``...``. The rows are read as the text dump's rows are, with the same
columns, type maps, units and image flags. ``DumpYAML::write_data`` [10]
prints nothing for a ``typelabel`` value (it handles element names, not
type labels), so such a column is empty in every row; it is dropped with a
note, and a file that has no other source of elements is refused. Rows are
counted from line numbers; a frame whose count does not match natoms (a
header without its ``---``, a file cut inside the next header) is read to
count its rows, and a header the file ends inside takes a position in
``skipped``. With ``dump_modify triclinic/general yes``, which the
documentation lists for the atom and custom styles only [16], LAMMPS 22 Jul
2025 still writes a yaml dump: the header holds the restricted box and the
rows general-frame positions, which nothing in the file marks. Frame 0's
atoms lying outside the stated box are counted when the file is opened, and
the notes give the count, the farthest distance beyond a face and this
cause. A run's own dump holds atoms up to about half the neighbour skin
beyond a face (LAMMPS wraps them when it re-neighbours, which ``neigh_modify
check yes``, the default, does once an atom has moved half the skin [17]),
and the general-frame files written while testing this reach 0.56 and
1.07 Å, so no distance separates the two and the file is read, not refused.

**AtomEye CFG** [11, 12]. ``Number of particles = N``, ``A = <length scale>
Angstrom``, ``H0(i,j)`` (edge i as row i, in units of A), optional
``Transform(i,j)`` (H = H0 Transform) and ``eta(i,j)`` (H = H0 sqrt(1 +
2 eta)), ``R`` (the rate scale), ``.NO_VELOCITY.``, ``entry_count``,
``auxiliary[k] = name``; then blocks of a mass line, an element line and
rows of reduced coordinates s (x = s H), velocities unless ``.NO_VELOCITY.``,
and the auxiliary values. Without ``entry_count`` it is the standard CFG: one
row ``mass symbol s1 s2 s3 v1 v2 v3`` per atom. LAMMPS writes a mass and an
element line before every atom (the type name of ``dump_modify element``,
``C`` for every type without it), the box with ``%g`` (6 significant
digits), and the auxiliary columns the dump command names (``id``, ``q``,
``ix iy iz`` are read). With ``xsu ysu zsu`` LAMMPS writes s' = (s - 0.5) / 10
+ 0.5 and A = 10 (``UNWRAPEXPAND`` in ``dump_cfg.cpp`` [12]) so that AtomEye
shows molecules whole in a box ten times larger: a file with A = 10 and one
atom per mass/element block (LAMMPS's layout; ASE writes A = 1, and OVITO
3.16 writes no CFG) is read back as LAMMPS wrote it (box H0, unwrapped s),
with a note; any other A scales H0 as AtomEye does. The %g printing then
leaves s with five significant digits, not six. The velocity columns are not
read: the specification gives ds/dt in R (ns^-1), and ASE writes Cartesian
velocities in its own units in the same columns. LAMMPS writes its velocities
as auxiliary ``vx vy vz`` columns instead, in the run's velocity unit, and
those are read, with ``units`` naming the style as for a text dump. A CFG
has no origin, no timestep and no time; for a series
(:func:`read_cfg_series`) the timesteps come from the file names when they
differ in one number only (LAMMPS replaces ``*`` in ``dump.*.cfg`` with the
timestep), and ``timestep_fs`` then gives times. When a series is opened
every file is checked: one cut short (rows missing, a last row cut, no line
ending after it), with other auxiliary columns or another atom count than
most files is in ``skipped``, and frame 0 is the first file that reads.
Names that follow more than one pattern with several files each (two
series in one directory) are refused with the patterns. ``entry_count`` is
held to the first atom row's width before a name is made for each column.

ELEMENTS AND REFUSALS
---------------------
Elements follow :mod:`.md_readers`' rules (its module docstring): a user map,
then the file's own labels written as files write symbols, then masses; a
label the file's own mass contradicts is refused; nothing is guessed. Every
refusal names the file, says why and says what to pass or how to rewrite the
dump. Frames that are truncated, hold another atom count than most frames, or
have a header that does not parse are listed in ``Trajectory.skipped`` with
the reason; a binary file of the older layout whose header cannot be
followed past some frame says how many bytes after it could not be located.
An option given with a value the reader refuses (``timestep_fs``,
``units``, ``type_map``, ``mass_tol_amu``) is refused naming the file, and
an empty file is refused as empty.

TIMINGS
-------
Measured on this machine (Windows 11, i5-13420H, Python 3.11, numpy 2.4.6,
not pinned, other sessions running: about 40 % CPU load) on synthetic files
of 10 000 atoms written to the layouts above (a sheared box changing every
frame, rows shuffled; binary and YAML with ``id type x y z ix iy iz vx vy vz
q``, the binary in four chunks). Open = index pass (for the DCD every
frame's markers and cell, for a CFG series every file's rows), topology or
type map, the DCD's wrapped/unwrapped scan, and frame 0; per frame = the
median over the frames read in order. Three runs of three repeats after the
checks of every frame and file were added (2026-10-07; two runs for the CFG
row, after its check was made faster), the files freshly written by the
timing script:

=================================  ========  ===========  ========
file                               size      open         per frame
=================================  ========  ===========  ========
DCD, 100 frames (data-file topo.)  12.0 MB   0.19-0.42 s  4-5 ms
binary dump, 100 frames            96.0 MB   0.03-0.06 s  8-10 ms
YAML dump, 100 frames              100.6 MB  0.37-0.52 s  55-63 ms
CFG series, 20 files               10.1 MB   0.13-1.74 s  72-82 ms
=================================  ========  ===========  ========

The first open of each run reads files written a moment before and is the
upper end of each open range (1.7 s for the CFG series, whose files are all
read; 0.13-0.16 s on the repeats). Of the DCD's open, the check of every
frame's markers and cell took 2 ms; reading the 10 000-atom data file took
0.06-0.13 s and the wrapped/unwrapped scan (64 sampled frames and the
frames after them) 0.10-0.14 s, as measured before the check was added. A
CFG file is checked by counting its line endings against a file whose rows
were counted in full (numpy over the bytes, about 10 ms for 0.5 MB). The
YAML and CFG rows are text, parsed as a text dump's rows are; the DCD and
binary frames are read with ``np.frombuffer``.

REFERENCES
----------
[1] LAMMPS source, src/EXTRA-DUMP/dump_dcd.cpp (write_dcd_header,
    write_header, pack, write_frame, init_style), release stable_22Jul2025,
    https://github.com/lammps/lammps/blob/stable_22Jul2025/src/EXTRA-DUMP/dump_dcd.cpp
[2] LAMMPS documentation, "dump command" (styles dcd, yaml, cfg; binary
    files), https://docs.lammps.org/dump.html
[3] MDAnalysis, package/MDAnalysis/lib/formats/include/readdcd.h (from VMD's
    dcdplugin: header offsets, byte order, CHARMM flags),
    https://github.com/MDAnalysis/mdanalysis/blob/develop/package/MDAnalysis/lib/formats/include/readdcd.h
[4] MDAnalysis user guide, "DCD (CHARMM, NAMD, or LAMMPS trajectory)" (AKMA
    time, unit-cell conventions),
    https://userguide.mdanalysis.org/stable/formats/reference/dcd.html, and
    MDAnalysis 2.10 coordinates/DCD.py (DCDReader._frame_to_ts)
[5] NAMD mailing list, "Re: Internal Unit system" (V. Ovchinnikov, 2012),
    https://tcbg.illinois.edu/Research/namd/mailing_list/namd-l.2012-2013/0317.html
[6] LAMMPS source, src/dump_custom.cpp and src/dump_atom.cpp
    (format_magic_string_binary, header_binary, header_binary_triclinic,
    header_binary_triclinic_general, header_unit_style_binary,
    header_time_binary, header_columns_binary, write_binary), release
    stable_22Jul2025,
    https://github.com/lammps/lammps/blob/stable_22Jul2025/src/dump_custom.cpp
[7] LAMMPS source, src/dump_custom.cpp, DumpCustom::parse_fields
    (``element`` and ``typelabel`` packed with pack_type), and src/dump.cpp
    (Dump::write: tilted boxes write boxlo_bound / boxhi_bound; ``.bin`` and
    ``.lammpsbin`` select binary output),
    https://github.com/lammps/lammps/blob/stable_22Jul2025/src/dump.cpp
[8] LAMMPS tools/binary2txt.cpp, release stable_22Jul2025,
    https://github.com/lammps/lammps/blob/stable_22Jul2025/tools/binary2txt.cpp
[9] LAMMPS documentation, "dump command", the yaml example and the box rule
    ("three lines for orthogonal boxes and four lines for triclinic boxes"),
    https://docs.lammps.org/dump.html
[10] LAMMPS source, src/EXTRA-DUMP/dump_yaml.cpp (write_header, write_data,
    write_footer), release stable_22Jul2025,
    https://github.com/lammps/lammps/blob/stable_22Jul2025/src/EXTRA-DUMP/dump_yaml.cpp
[11] J. Li, AtomEye configuration file formats (standard and extended CFG),
    http://li.mit.edu/Archive/Graphics/A/ (the https address does not answer;
    read over http on 2026-10-07)
[12] LAMMPS source, src/dump_cfg.cpp (UNWRAPEXPAND, write_header,
    write_lines), release stable_22Jul2025,
    https://github.com/lammps/lammps/blob/stable_22Jul2025/src/dump_cfg.cpp
[13] VMD molfile plugin dcdplugin.c as shipped with mdtraj (open_dcd_write:
    ``istart = 0; nsavc = 1; delta = 1.0;`` and the title "Created by DCD
    plugin"),
    https://github.com/mdtraj/mdtraj/blob/master/mdtraj/formats/dcd/src/dcdplugin.c
[14] LAMMPS source, src/output.cpp (Output::setup: a dump writes on the
    setup step when ``last_dump < 0 && first_flag == 1`` or the step is a
    multiple of N, and next_dump = (step / N) * N + N; reset_timestep:
    "Cannot reset timestep with active dump"), release stable_22Jul2025,
    https://github.com/lammps/lammps/blob/stable_22Jul2025/src/output.cpp
[15] MDAnalysis 2.10, coordinates/DCD.py, DCDWriter ("When writing out
    timesteps without dimensions ... a zeroed unitcell"),
    https://github.com/MDAnalysis/mdanalysis/blob/develop/package/MDAnalysis/coordinates/DCD.py
[16] LAMMPS documentation, "dump_modify command" (colname; first;
    triclinic/general "only applies to the dump atom and custom styles"),
    https://docs.lammps.org/dump_modify.html
[17] LAMMPS documentation, "neigh_modify command" (check yes: "only build
    if at least one atom has moved half the skin distance or more"),
    https://docs.lammps.org/neigh_modify.html

The layouts read from these sources were checked against files written by
LAMMPS 22 Jul 2025 update 4 and ASE 3.29.0 (tests/data/md/lammps_formats and
the recognition corpus), with LAMMPS's internal state as the reference.
"""
from __future__ import annotations

import dataclasses
import gzip
import math
import re
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import constants as _constants

from . import md_model
from . import md_readers as _mr
from .md_formats_base import FormatSpec
from .md_model import (NO_TIMESTEP, Frame, Trajectory, frame_from_arrays,
                       unwrap_with_images)
from .readers import UnsupportedFormat

__all__ = [
    "FORMATS", "AKMA_TIME_PS", "DCD_FORMAT", "BINARY_FORMAT", "YAML_FORMAT",
    "CFG_FORMAT", "read_dcd", "read_lammps_binary_dump", "read_lammps_yaml_dump",
    "read_cfg", "read_cfg_series", "sniff_dcd", "sniff_lammps_binary",
    "sniff_lammps_yaml", "sniff_cfg", "dcd_cell_to_box", "timesteps_from_names",
]

# ---------------------------------------------------------------------------
# constants: each one is a definition or a choice, and says which
# ---------------------------------------------------------------------------

# The names read_trajectory and sniff_md give these formats.
DCD_FORMAT = "dcd"
BINARY_FORMAT = "lammps-dump-binary"
YAML_FORMAT = "lammps-dump-yaml"
CFG_FORMAT = "atomeye-cfg"

# The AKMA time unit in ps, a definition: sqrt(amu * Å^2 / (kcal/mol)) [5].
# amu * N_A is the molar mass constant (0.99999999965e-3 kg/mol since 2019);
# kcal is the thermochemical kilocalorie, 4184 J (scipy.constants.calorie).
AKMA_TIME_PS: float = math.sqrt(
    _constants.physical_constants["atomic mass constant"][0] * _constants.N_A
    * 1e-20 / (1e3 * _constants.calorie)) * 1e12

# The record markers a DCD can start with: the first record is 84 bytes.
_DCD_FIRST_RECORD = 84
_DCD_MAGIC = (b"CORD", b"VELD")

# How many frames of a DCD are sampled when it is opened to tell wrapped from
# unwrapped positions. A cost bound, not a property of the file: 64 frames of
# 10 000 atoms are about 8 MB of float32.
_DCD_SAMPLE_FRAMES = 64
# An atom set spanning more than this many box lengths along a box vector
# holds unwrapped positions. A choice: a wrapped LAMMPS file spans one box
# length plus the distance atoms move between re-neighbourings (about half
# the neighbour skin on each side), far below half a box.
_UNWRAPPED_SPAN = 1.5
# A fractional coordinate that changes by more than this between consecutive
# frames has jumped across the box. A choice: half a box, which continuous
# positions do not cover between two dumps of an MD run.
_JUMP_FRACTION = 0.5

# The magic strings of LAMMPS's binary dumps [6] (DumpCustom, DumpAtom).
_BINARY_MAGIC = (b"DUMPCUSTOM", b"DUMPATOM")
# The suffixes that make LAMMPS write a binary dump [7]. A file of the older
# layout (no magic string) is recognised only under one of these names.
_BINARY_SUFFIXES = (".bin", ".lammpsbin")
_BOUNDARY_FLAGS = {0: "p", 1: "f", 2: "s", 3: "m"}
# Plausibility limits for the integer fields of a binary header, used to tell
# a header from other bytes and from a damaged one. Parsing choices.
_MAX_COLUMNS = 100_000
_MAX_CHUNKS = 10_000_000

# The bytes that separate values on a text line (space, tab, CR, LF, VT, FF),
# as a lookup table over byte values.
_BLANK_BYTE = np.zeros(256, dtype=bool)
_BLANK_BYTE[list(b" \t\r\n\v\f")] = True

# LAMMPS's cfg writer: the unwrapped-coordinate expansion factor [12].
_CFG_UNWRAP_EXPAND = 10.0
# How much of a CFG file is read to find the end of its header. A choice: an
# AtomEye header is a few hundred bytes; a longer one is read whole.
_CFG_HEAD_BYTES = 65536

_GZIP_DAMAGE = (zlib.error, gzip.BadGzipFile, EOFError)

# The most bytes asked of a file in one read. A choice: a length field that a
# damaged or foreign file fills with a large number then costs at most this
# much memory before the read meets the end of the file (a 948-byte DCD whose
# title marker read 2^31 - 1 allocated 2.1 GB when the read asked for it all).
_READ_PIECE = 1 << 24

# VMD's molfile DCD plugin (also inside mdtraj) titles its files so and writes
# ISTART 0, NSAVC 1 and DELTA 1.0 whatever the run [13]: placeholders, not a
# time axis.
_MOLFILE_TITLE = "Created by DCD plugin"

# The route for a dump whose columns carry names other than LAMMPS's keywords
# (dump_modify colname renames them in the header, [16]).
_COLNAME_ADVICE = (
    "FACET reads the columns by LAMMPS's own keywords (id, type, element, x y "
    "z, xs ys zs, xu yu zu, xsu ysu zsu, ix iy iz, vx vy vz, q); a dump "
    "written with dump_modify colname is read with columns='<LAMMPS's "
    "keywords, in the file's order>' (for example columns='id type x y z'), "
    "or can be written again without colname")


def _what_facet_reads() -> str:
    return ("FACET reads LAMMPS data files and text, binary and YAML dumps, "
            "DCD, AtomEye CFG, extended XYZ, VASP XDATCAR and DL_POLY "
            "CONFIG / HISTORY")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _shortest_float32(value: float) -> float:
    """The shortest decimal that a float32 holding ``value`` rounds to, as a
    float64: 0.001 for the float32 nearest 0.001, not 0.0010000000474974513."""
    return float(str(np.float32(value)))


def _checked_positive(value, what: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{what} needs a positive number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} {value!r} is not a number") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{what} {value!r} needs to be a positive number")
    return number


def _named(name: str, check, *args):
    """``check(*args)``, with a ValueError's message starting with the file
    name (the shared option checks of :mod:`.md_readers` do not know it)."""
    try:
        return check(*args)
    except ValueError as error:
        text = str(error)
        raise ValueError(text if text.startswith(name)
                         else f"{name}: {text}") from None


def _refuse_empty(path: Path, name: str, what: str) -> None:
    """UnsupportedFormat for a file that holds no byte (after gzip)."""
    try:
        empty = not _mr._head(path, 1)
    except (OSError, EOFError):
        return
    if empty:
        raise UnsupportedFormat(f"{name}: the file is empty, so it holds no "
                                f"{what}; {_what_facet_reads()}")


def _finite_box(box, what: str) -> np.ndarray:
    """``box`` checked as :func:`.md_model._checked_box` does, and refused
    when its edge lengths or volume overflow to a number that is not finite
    (finite rows of 1e300 Å pass the row check)."""
    box = np.asarray(box, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        lengths = np.linalg.norm(box, axis=1) if box.shape == (3, 3) else None
        volume = abs(float(np.linalg.det(box))) if box.shape == (3, 3) else 0.0
    if lengths is not None and np.isfinite(box).all() and not (
            np.isfinite(lengths).all() and math.isfinite(volume)):
        raise ValueError(f"{what} gives a box whose edge lengths or volume are "
                         "not finite numbers")
    return md_model._checked_box(box)


def _whole(values: np.ndarray, what: str, where: str) -> np.ndarray:
    """int64 from floats that hold whole numbers, or ValueError naming them."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (values != np.round(values)).any():
        raise ValueError(f"{where}: the {what} column holds a value that is not "
                         "a whole number")
    return values.astype(np.int64)


def _positions_of(columns: Sequence[str]):
    """((kind, names) wrapped, (kind, names) or None unwrapped) of a dump's
    columns, or a string saying why there are none; the text dump's rule
    (:func:`.md_readers._dump_header`): x, xs, xu, xsu in that order."""
    complete = {}
    for kind, names in _mr._POSITION_SETS:
        have = [c for c in names if c in columns]
        if len(have) == 3:
            complete[kind] = names
        elif have:
            return (f"the columns have {' '.join(have)} without the rest of "
                    f"{' '.join(names)}")
    if not complete:
        return ("the columns hold no complete set of positions (x y z, xs ys "
                "zs, xu yu zu or xsu ysu zsu)")
    kind = next(k for k, _ in _mr._POSITION_SETS if k in complete)
    unwrapped = None
    for ukind in ("xu", "xsu"):
        if ukind in complete:
            unwrapped = (ukind, complete[ukind])
            break
    return (kind, complete[kind]), unwrapped


class _Cut:
    """Set when a gzip stream ends before its end-of-stream marker (a copy
    taken while the file was being written)."""

    def __init__(self) -> None:
        self.seen = False


def _read_exact(handle, size: int, name: str, cut: _Cut | None = None) -> bytes:
    """``size`` bytes, fewer only at the end of the file, read in pieces of
    at most :data:`_READ_PIECE` so a length no file holds never allocates
    it; a damaged gzip stream is refused naming the file, and a cut one
    (no end-of-stream marker) ends the bytes and sets ``cut.seen``."""
    pieces: list[bytes] = []
    got = 0
    while got < size:
        try:
            data, at_end, was_cut = _mr._read_some(
                handle, min(size - got, _READ_PIECE))
        except UnsupportedFormat:
            raise
        except _GZIP_DAMAGE as error:
            raise UnsupportedFormat(f"{name}: the gzip stream is damaged and "
                                    f"cannot be decompressed ({error})") from None
        if was_cut and cut is not None:
            cut.seen = True
        if data:
            pieces.append(data)
            got += len(data)
        if at_end:
            break
    return b"".join(pieces)


def _skip(handle, size: int, name: str, cut: _Cut | None = None) -> bool:
    """Move ``size`` bytes forward; False when the file ends first. A gzip
    stream is read through :func:`_read_exact`, so a cut or damaged one is
    reported as for any other read."""
    if size <= 0:
        return True
    if isinstance(handle, gzip.GzipFile):
        remaining = size
        while remaining > 0:
            want = min(remaining, _READ_PIECE)
            piece = _read_exact(handle, want, name, cut)
            remaining -= len(piece)
            if len(piece) < want:
                return False
        return True
    start = handle.tell()
    handle.seek(0, 2)
    end = handle.tell()
    if start + size > end:
        handle.seek(end)
        return False
    handle.seek(start + size)
    return True


def _file_size(path: Path, name: str) -> int:
    """Bytes of the file, decompressed when it is gzip."""
    if not _mr._is_gzip(path):
        return path.stat().st_size
    total = 0
    with _mr._open_binary(path) as handle:
        while True:
            data = _read_exact(handle, 1 << 24, name)
            if not data:
                return total
            total += len(data)


def _boundary_note(periodic_sets, boundaries: Sequence[str], what: str) -> list[str]:
    notes = []
    nonperiodic = sorted({b for b, p in zip(boundaries, periodic_sets)
                          if p is not None and not all(p)})
    if nonperiodic:
        notes.append(f"boundary flags {nonperiodic}: not periodic along every "
                     "axis, while FACET measures the box as periodic along a, b "
                     "and c, so atoms near a non-periodic face see images "
                     "across it")
    if any(p is None for p in periodic_sets):
        notes.append(f"{what} states no boundary flags; the box is taken as "
                     "periodic along a, b and c")
    return notes


# ---------------------------------------------------------------------------
# DCD
# ---------------------------------------------------------------------------

def _dcd_layout(head: bytes) -> tuple[str, int, bytes] | None:
    """(byte order, marker size, magic) of a DCD head, or None."""
    for order in "<>":
        if len(head) >= 8 and struct.unpack(order + "i", head[:4])[0] \
                == _DCD_FIRST_RECORD and head[4:8] in _DCD_MAGIC:
            return order, 4, head[4:8]
    for order in "<>":
        if len(head) >= 12 and struct.unpack(order + "q", head[:8])[0] \
                == _DCD_FIRST_RECORD and head[8:12] in _DCD_MAGIC:
            return order, 8, head[8:12]
    return None


def sniff_dcd(head: bytes, path: Path) -> bool:
    """A DCD: the first record is 84 bytes and starts with CORD (or VELD)."""
    try:
        return _dcd_layout(head) is not None
    except Exception:       # noqa: BLE001 - a sniffer never raises
        return False


@dataclass
class _DcdHeader:
    order: str
    marker: int
    magic: bytes
    nset: int
    istart: int
    nsavc: int
    last_step: int
    namnf: int
    delta: float
    delta_bits: int            # 32 or 64
    has_cell: bool
    four_d: bool
    charmm: int
    titles: list[str]
    natom: int
    header_bytes: int

    @property
    def lammps(self) -> bool:
        return bool(self.titles) and self.titles[0].startswith("Written by LAMMPS")


def _dcd_record(handle, header: tuple[str, int], what: str, name: str) -> bytes:
    order, size = header
    code = order + ("i" if size == 4 else "q")
    raw = _read_exact(handle, size, name)
    if len(raw) < size:
        raise ValueError(f"{name}: the file ends before the {what} record")
    length = struct.unpack(code, raw)[0]
    if not 0 <= length <= 1 << 31:
        raise ValueError(f"{name}: the {what} record gives a length of {length} "
                         "bytes; the file is damaged or not a DCD")
    payload = _read_exact(handle, length, name)
    tail = _read_exact(handle, size, name)
    if len(payload) < length or len(tail) < size:
        raise ValueError(f"{name}: the file ends inside the {what} record")
    closing = struct.unpack(code, tail)[0]
    if closing != length:
        raise ValueError(f"{name}: the {what} record's two length markers differ "
                         f"({length} and {closing} bytes); the file is damaged")
    return payload


def _dcd_header(path: Path) -> _DcdHeader:
    name = path.name
    with _mr._open_binary(path) as handle:
        head = _read_exact(handle, 12, name)
        layout = _dcd_layout(head)
        if layout is None:
            raise UnsupportedFormat(
                f"{name}: not a DCD file (its first record is not 84 bytes "
                f"starting with CORD); {_what_facet_reads()}")
        order, msize, magic = layout
    with _mr._open_binary(path) as handle:
        first = _dcd_record(handle, (order, msize), "header", name)
        icntrl = struct.unpack(order + "20i", first[4:84])
        charmm = icntrl[19]
        if charmm != 0:
            delta = struct.unpack(order + "f", first[40:44])[0]
            bits = 32
            has_cell = icntrl[10] != 0
            four_d = icntrl[11] == 1
        else:
            delta = struct.unpack(order + "d", first[40:48])[0]
            bits = 64
            has_cell = False
            four_d = False
        titles_raw = _dcd_record(handle, (order, msize), "title", name)
        titles = []
        if len(titles_raw) >= 4:
            count = struct.unpack(order + "i", titles_raw[:4])[0]
            body = titles_raw[4:]
            for k in range(max(0, min(count, len(body) // 80))):
                text = body[80 * k:80 * (k + 1)].split(b"\0")[0]
                titles.append(text.decode("latin-1").strip())
        natom_raw = _dcd_record(handle, (order, msize), "atom-count", name)
        if len(natom_raw) != 4:
            raise ValueError(f"{name}: the atom-count record holds "
                             f"{len(natom_raw)} bytes where 4 are expected")
        natom = struct.unpack(order + "i", natom_raw)[0]
        header_bytes = 3 * 2 * msize + len(first) + len(titles_raw) + 4
    if natom < 1:
        raise ValueError(f"{name}: the DCD gives {natom} atoms")
    return _DcdHeader(order, msize, magic, icntrl[0], icntrl[1], icntrl[2],
                      icntrl[3], icntrl[8], float(delta), bits, has_cell, four_d,
                      charmm, titles, natom, header_bytes)


def _dcd_dtype(header: _DcdHeader) -> np.dtype:
    marker = header.order + ("i4" if header.marker == 4 else "i8")
    fields = []
    if header.has_cell:
        fields += [("cell0", marker), ("cell", header.order + "f8", (6,)),
                   ("cell1", marker)]
    for axis in ("x", "y", "z") + (("w",) if header.four_d else ()):
        fields += [(axis + "0", marker),
                   (axis, header.order + "f4", (header.natom,)),
                   (axis + "1", marker)]
    return np.dtype(fields)


def dcd_cell_to_box(cell) -> tuple[np.ndarray, str]:
    """``(box_ang, convention)`` from a DCD unit-cell record.

    ``cell`` is the six values as written: ``[a, cos gamma, b, cos beta,
    cos alpha, c]`` (LAMMPS [1], NAMD > 2.5), or the same with the angles in
    degrees (NAMD 2.5, detected as every angle slot within (1, 180)). The box
    rows are a = (a, 0, 0), b in the xy plane, c above it -- LAMMPS's restricted
    triclinic frame, so for a LAMMPS file the box is xprd, xy, yprd, xz, yz,
    zprd again. ValueError when the values fit neither reading or give no box.
    """
    values = np.asarray(cell, dtype=np.float64).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("the unit-cell record needs six finite values")
    a, s_gamma, b, s_beta, s_alpha, c = values
    slots = (s_alpha, s_beta, s_gamma)
    if all(-1.0 <= s <= 1.0 for s in slots):
        cos_alpha, cos_beta, cos_gamma = slots
        convention = "lengths and angle cosines"
    elif all(1.0 < s < 180.0 for s in slots):
        # cos(t) as sin(90 - t), so 90 degrees gives exactly 0 (as VMD does)
        cos_alpha, cos_beta, cos_gamma = (math.sin(math.radians(90.0 - s))
                                          for s in slots)
        convention = "lengths and angles in degrees"
    else:
        raise ValueError(
            f"the unit-cell record [{', '.join(f'{v:.6g}' for v in values)}] "
            "holds angle values that are neither all cosines (within [-1, 1]) "
            "nor all angles in degrees (within (1, 180)); CHARMM's symmetric "
            "shape matrix is not read")
    if not (a > 0 and b > 0 and c > 0):
        raise ValueError(f"the unit-cell record gives lengths {a:.6g}, {b:.6g}, "
                         f"{c:.6g}; all three need to be positive")
    bx = b * cos_gamma
    by2 = b * b - bx * bx
    if by2 <= 0:
        raise ValueError("the unit-cell record gives a and b parallel")
    by = math.sqrt(by2)
    cx = c * cos_beta
    cy = (b * c * cos_alpha - bx * cx) / by
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError("the unit-cell record gives angles that span no "
                         "volume")
    box = np.array([[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, math.sqrt(cz2)]])
    _finite_box(box, "the unit-cell record")
    return box, convention


def _dcd_cells_fit(cells: np.ndarray, convention: str) -> np.ndarray:
    """Which unit-cell records (rows of six values) give a box under
    ``convention``: the checks of :func:`dcd_cell_to_box` and
    :func:`.md_model._checked_box`, on every row at once."""
    with np.errstate(all="ignore"):
        a, b, c = cells[:, 0], cells[:, 2], cells[:, 5]
        slots = cells[:, [4, 3, 1]]                 # alpha, beta, gamma
        finite = np.isfinite(cells).all(axis=1)
        if convention == "lengths and angle cosines":
            fits = np.all((slots >= -1.0) & (slots <= 1.0), axis=1)
            cosines = slots
        else:
            fits = np.all((slots > 1.0) & (slots < 180.0), axis=1)
            cosines = np.sin(np.radians(90.0 - slots))
        ca, cb, cg = cosines[:, 0], cosines[:, 1], cosines[:, 2]
        bx = b * cg
        by2 = b * b - bx * bx
        by = np.sqrt(np.where(by2 > 0, by2, 1.0))
        cx = c * cb
        cy = (b * c * ca - bx * cx) / by
        cz2 = c * c - cx * cx - cy * cy
        volume = a * by * np.sqrt(np.where(cz2 > 0, cz2, 0.0))
        ratio = volume / (a * b * c)
        return (finite & fits & (a > 0) & (b > 0) & (c > 0) & (by2 > 0)
                & (cz2 > 0) & np.isfinite(volume) & np.isfinite(a * b * c)
                & (ratio >= md_model.BOX_MIN_NORMALISED_VOLUME))


def _dcd_records_ok(record, header: _DcdHeader) -> str | None:
    """Why a frame's record markers do not fit the header, or None."""
    expected = {"x": 4 * header.natom, "y": 4 * header.natom,
                "z": 4 * header.natom}
    if header.four_d:
        expected["w"] = 4 * header.natom
    if header.has_cell:
        expected["cell"] = 48
    for name, size in expected.items():
        first, last = int(record[name + "0"]), int(record[name + "1"])
        if first != size or last != size:
            return (f"its {name} record has length markers {first} and {last} "
                    f"where {size} bytes are expected")
    return None


@dataclass
class _DcdScan:
    n_full: int                      # complete frames
    remainder: int                   # bytes after the last complete frame
    damaged: dict                    # frame -> why its record markers do not fit
    cells: np.ndarray | None         # (n_full, 6) unit-cell records
    cut: bool                        # a gzip stream without end-of-stream marker


def _dcd_marker_problems(records: np.ndarray, header: _DcdHeader, base: int,
                         damaged: dict) -> None:
    """Add to ``damaged`` the frames (``base`` + row) whose record markers do
    not fit the header, checked on every frame at once."""
    expected = {"x": 4 * header.natom, "y": 4 * header.natom,
                "z": 4 * header.natom}
    if header.four_d:
        expected["w"] = 4 * header.natom
    if header.has_cell:
        expected["cell"] = 48
    for field_name, size in expected.items():
        first = records[field_name + "0"].astype(np.int64)
        last = records[field_name + "1"].astype(np.int64)
        for row in np.flatnonzero((first != size) | (last != size)).tolist():
            damaged.setdefault(base + row, (
                f"its {field_name} record has length markers {first[row]} and "
                f"{last[row]} where {size} bytes are expected; the file is "
                "damaged or holds another layout than its header states"))


def _dcd_scan(path: Path, header: _DcdHeader, dtype: np.dtype,
              name: str) -> _DcdScan:
    """Every frame's record markers and unit cell, read when the file is
    opened: a plain file through a memory map (only the marker and cell bytes
    are touched), a gzip file decompressed once in pieces of whole frames."""
    size = dtype.itemsize
    damaged: dict[int, str] = {}
    if not _mr._is_gzip(path):
        body = path.stat().st_size - header.header_bytes
        n_full = max(0, body // size)
        remainder = body - n_full * size if body > 0 else 0
        cells = None
        if n_full:
            mapped = np.memmap(path, dtype=dtype, mode="r",
                               offset=header.header_bytes, shape=(n_full,))
            try:
                _dcd_marker_problems(mapped, header, 0, damaged)
                if header.has_cell:
                    cells = np.array(mapped["cell"], dtype=np.float64)
            finally:
                del mapped
        return _DcdScan(n_full, remainder, damaged, cells, False)
    cut = _Cut()
    batch = max(1, _READ_PIECE // size)
    n_full = 0
    remainder = 0
    pieces = []
    with _mr._open_binary(path) as handle:
        _read_exact(handle, header.header_bytes, name, cut)
        while True:
            data = _read_exact(handle, batch * size, name, cut)
            whole = len(data) // size
            if whole:
                records = np.frombuffer(data, dtype=dtype, count=whole)
                _dcd_marker_problems(records, header, n_full, damaged)
                if header.has_cell:
                    pieces.append(np.array(records["cell"], dtype=np.float64))
                n_full += whole
            if len(data) < batch * size:
                remainder = len(data) - whole * size
                break
    cells = None
    if header.has_cell:
        cells = np.concatenate(pieces) if pieces else np.zeros((0, 6))
    return _DcdScan(n_full, remainder, damaged, cells, cut.seen)


def _dcd_steps(header: _DcdHeader, n: int) -> tuple[list[int] | None, str]:
    """The timestep of each of the first ``n`` frames, and how they were
    found; None when the header does not give them.

    LAMMPS writes ISTART = the first dumped step, NSAVC = the dump interval
    and, after every frame, the step of that frame in ICNTRL[3] [1]. Its
    snapshots fall on multiples of NSAVC, except the first one under
    ``dump_modify first yes``, which is written on the step the run starts
    from (output.cpp, Output::setup [14]): ISTART, then the multiples of
    NSAVC after it. The layout whose last step is ICNTRL[3] is used; when
    neither is, the timesteps are not known. LAMMPS refuses ``reset_timestep``
    and ``dump_modify every`` while a DCD dump is open [14, 1], so no other
    layout is written.
    """
    if header.nsavc <= 0:
        return None, (f"the header gives NSAVC = {header.nsavc}, so the frames "
                      "have no timesteps")
    arithmetic = [header.istart + k * header.nsavc for k in range(n)]
    text = (f"timesteps from the header: ISTART {header.istart} + k x NSAVC "
            f"{header.nsavc}")
    if not header.lammps or header.nset <= 1:
        return arithmetic, text
    last = header.istart + (header.nset - 1) * header.nsavc
    if header.last_step == last:
        return arithmetic, text
    base = header.istart // header.nsavc
    first_yes = [header.istart] + [(base + k) * header.nsavc
                                   for k in range(1, max(n, header.nset))]
    if header.istart % header.nsavc and \
            first_yes[header.nset - 1] == header.last_step:
        return first_yes[:n], (
            f"timesteps from the header as dump_modify first yes writes them: "
            f"ISTART {header.istart} (a step that is not a multiple of NSAVC "
            f"{header.nsavc}), then the multiples of NSAVC after it; the "
            f"header's last-timestep field holds {header.last_step}, which this "
            f"layout gives and ISTART + (NSET - 1) x NSAVC ({last}) does not")
    return None, (
        f"the header's last-timestep field holds {header.last_step}, which "
        f"neither ISTART + (NSET - 1) x NSAVC ({last}) nor the dump_modify "
        f"first yes layout ({first_yes[header.nset - 1]}) gives, with ISTART "
        f"{header.istart}, NSAVC {header.nsavc} and NSET {header.nset}; the "
        "frames' timesteps are not known, so they have none")


class _DcdTrajectory(_mr._FileTrajectory):
    def __init__(self, path: Path, ranges, *, header: _DcdHeader, dtype,
                 elements, ids, fixed_box, convention: str | None,
                 unwrapped: bool, **kwargs) -> None:
        super().__init__(path, ranges, **kwargs)
        self._header = header
        self._dtype = dtype
        self._elements = elements
        self._ids = ids
        self._fixed_box = fixed_box
        self._convention = convention
        self._unwrapped = unwrapped

    def _raw(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(positions as written, box) of readable frame k."""
        record = np.frombuffer(self._bytes(k), dtype=self._dtype, count=1)[0]
        problem = _dcd_records_ok(record, self._header)
        if problem is not None:
            raise ValueError(f"{problem}; the file is damaged or holds another "
                             "layout than its header states")
        if self._fixed_box is not None:
            box = self._fixed_box
        else:
            box, convention = dcd_cell_to_box(record["cell"])
            if convention != self._convention:
                raise ValueError(f"its unit cell is written as {convention}, "
                                 f"frame 0's as {self._convention}")
        positions = np.stack([record["x"], record["y"], record["z"]],
                             axis=1).astype(np.float64)
        return positions, box

    def _load(self, k: int) -> Frame:
        positions, box = self._raw(k)
        step = int(self.timesteps[k])
        time = None
        if self.times_ps is not None and np.isfinite(self.times_ps[k]):
            time = float(self.times_ps[k])
        return frame_from_arrays(
            self._elements, positions, box_ang=box, atom_id=self._ids,
            timestep=None if step == NO_TIMESTEP else step, time_ps=time,
            unwrapped_cart_ang=positions if self._unwrapped else None)


def _topology_title_units(path: Path) -> str | None:
    """The unit style a write_data title names ("... units = metal"), or
    None."""
    try:
        head = _mr._head(path, 512)
    except (OSError, ValueError, EOFError):
        return None
    first = head.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    match = _mr._WRITE_DATA_TITLE.match(first)
    if match and match.group(2):
        return match.group(2).lower()
    return None


@dataclass
class _Topology:
    path: Path
    file_format: str
    elements: np.ndarray
    ids: np.ndarray
    type_map: dict
    source: str
    charges_e: dict | None
    title_units: str | None


def _read_topology(topology, name: str, n_atoms: int, options: dict) -> _Topology:
    path = Path(str(topology))
    if not path.is_file():
        raise ValueError(f"{name}: the topology {topology} is not a file; give "
                         "the LAMMPS data file the run read")
    file_format = _mr.sniff_md(path)
    if file_format is None:
        raise ValueError(
            f"{name}: the topology {path.name} is not recognised as an MD model; "
            "give the LAMMPS data file the run read (or a LAMMPS dump or "
            "extended XYZ file of the same atoms)")
    allowed = getattr(_mr, "_OPTIONS", {}).get(file_format)
    kwargs = {}
    for key, value in options.items():
        if value is None:
            continue
        if key == "units" and allowed is not None and key not in allowed:
            continue
        kwargs[key] = value
    try:
        trajectory = _mr.read_trajectory(path, **kwargs)
    except (UnsupportedFormat, ValueError) as error:
        raise type(error)(f"{name}: its topology {path.name} could not be read: "
                          f"{error}") from error
    try:
        frame = trajectory.frame(0)
    finally:
        close = getattr(trajectory, "close", None)
        if close is not None:
            close()
    if frame.n_atoms != n_atoms:
        raise ValueError(
            f"{name}: the DCD holds {n_atoms} atoms and its topology "
            f"{path.name} {frame.n_atoms}; a DCD written for a group holds a "
            "subset of the atoms and does not say which, so give a topology "
            "holding exactly the dumped atoms (a data file written for that "
            "group), or dump the group 'all'")
    if file_format == "lammps-dump" and not getattr(trajectory, "ids_track_atoms",
                                                    True):
        raise ValueError(
            f"{name}: its topology {path.name} is a dump without an id column, "
            "so its rows are not in atom-id order; give the LAMMPS data file "
            "the run read, or a dump with an id column")
    return _Topology(path, file_format, frame.elements, frame.atom_id,
                     dict(trajectory.type_map), trajectory.type_map_source,
                     trajectory.charges_e,
                     _topology_title_units(path) if file_format == "lammps-data"
                     else None)


def _dcd_box_from(box_from, name: str) -> tuple[np.ndarray, str]:
    if isinstance(box_from, (str, Path)):
        path = Path(str(box_from))
        if not path.is_file():
            raise ValueError(f"{name}: box_from {box_from} is not a file")
        try:
            if _mr.sniff_md(path) == "lammps-data":
                # only the box is needed, so the types need no elements
                box, _, kind = _mr._data_box(_mr._parse_data_file(path), path)
                md_model._checked_box(box)
                return box, f"the {kind} box of {path.resolve()}"
            with _mr.read_trajectory(path) as trajectory:
                frame = trajectory.frame(0)
        except (UnsupportedFormat, ValueError) as error:
            raise type(error)(f"{name}: box_from {path.name} could not be read: "
                              f"{error}") from error
        return np.array(frame.box_ang), f"frame 0 of {path.resolve()}"
    box = np.asarray(box_from, dtype=np.float64)
    if box.shape != (3, 3):
        raise ValueError(f"{name}: box_from needs an MD file or three box rows "
                         "(3, 3) in Å")
    md_model._checked_box(box)
    return box, "the box_from rows"


def _dcd_wrap_state(trajectory: _DcdTrajectory
                    ) -> tuple[str, str, bool | None]:
    """('unwrapped' | 'wrapped' | 'unknown', the evidence as a sentence,
    whether the sampled boxes differ: True, False when every frame was
    sampled and none differs, else None)."""
    n = trajectory.n_frames
    picks = sorted(set(np.linspace(0, n - 1, min(n, _DCD_SAMPLE_FRAMES))
                       .round().astype(int).tolist()))
    pairs = [(k, k + 1) for k in picks if k + 1 < n]
    needed = sorted(set(picks) | {b for _, b in pairs})
    span_hits: list[tuple[int, float]] = []
    jump_hits: list[tuple[int, int]] = []
    fracs: dict[int, np.ndarray] = {}
    boxes: dict[int, np.ndarray] = {}
    unreadable: list[int] = []
    for k in needed:
        try:
            positions, box = trajectory._raw(k)
        except ValueError:
            unreadable.append(k)
            continue
        boxes[k] = box
        frac = np.linalg.solve(box.T, positions.T).T
        fracs[k] = frac
        if k in picks:
            span = float((frac.max(axis=0) - frac.min(axis=0)).max())
            if span > _UNWRAPPED_SPAN:
                span_hits.append((k, span))
    for a, b in pairs:
        if a in fracs and b in fracs:
            moved = np.abs(fracs[b] - fracs[a]) > _JUMP_FRACTION
            count = int(moved.any(axis=1).sum())
            if count:
                jump_hits.append((a, count))
    first = boxes.get(min(boxes)) if boxes else None
    differ = any(not np.array_equal(b, first) for b in boxes.values())
    box_varies = True if differ else (False if len(boxes) == n else None)
    checked = (f"{len(picks)} of {n} frame(s) sampled for the span and "
               f"{len(pairs)} consecutive pair(s) for jumps"
               + (f"; frame(s) {unreadable[:5]} could not be read for it"
                  if unreadable else ""))
    if span_hits and not jump_hits:
        k, span = span_hits[0]
        return "unwrapped", (
            f"positions read as unwrapped (as dump_modify unwrap yes writes "
            f"them): in frame {k} the atoms span {span:.3g} box lengths along a "
            f"box vector, more than a wrapped file spans ({_UNWRAPPED_SPAN:g}); "
            f"no atom jumps across the box between frames ({checked}); "
            "the positions as written are kept as unwrapped_cart_ang"), box_varies
    if jump_hits and not span_hits:
        a, count = jump_hits[0]
        return "wrapped", (
            f"positions read as wrapped (LAMMPS's default, no dump_modify "
            f"unwrap): {count} atom(s) jump across the box between frames {a} "
            f"and {a + 1} ({checked}); no unwrapped positions are set"), box_varies
    if span_hits and jump_hits:
        return "unknown", (
            "the positions span more than a wrapped file spans and also jump "
            f"across the box between frames ({checked}); whether they were "
            "written wrapped or unwrapped is not known, so no unwrapped "
            "positions are set"), box_varies
    return "unknown", (
        "the DCD does not record whether its positions were written wrapped "
        "(LAMMPS's default) or unwrapped (dump_modify unwrap yes), and the "
        f"positions do not show it ({checked}: no span above "
        f"{_UNWRAPPED_SPAN:g} box lengths, no jump across the box); no "
        "unwrapped positions are set"), box_varies


def read_dcd(path, *, topology=None, type_map: Mapping | None = None,
             atom_style: str | None = None, masses_from=None,
             units: str | None = None, mass_tol_amu: float | None = None,
             timestep_fs: float | None = None, box_from=None) -> Trajectory:
    """A DCD trajectory (LAMMPS ``dump dcd``; CHARMM, NAMD, VMD).

    ``topology`` is required: an MD file of the same atoms, normally the
    LAMMPS data file the run read, which gives each atom's element and id in
    atom-id order. ``type_map``, ``atom_style``, ``masses_from`` and
    ``mass_tol_amu`` are passed to the reading of the topology, and ``units``
    too when the topology's format takes it. ``units`` (or the write_data
    title of a data-file topology) states the run's unit style, which gives
    DELTA's time unit; ``timestep_fs`` replaces DELTA (for a file of VMD's
    molfile plugin, whose header holds placeholders, it is the time between
    frames). ``box_from`` gives the box of a DCD written without unit cells,
    or with zeroed ones (MDAnalysis without dimensions): an MD file, whose
    frame 0's box is used, or three rows in Å. Every frame's record markers
    and unit cell are checked when the file is opened, and a frame whose
    markers or cell do not fit is listed in ``skipped``. Positions are
    float32 as written; see the module docstring for the unit cell, the time,
    the missing origin and how wrapped and unwrapped files are told apart.
    """
    path = _mr._existing(path)
    name = path.name
    _refuse_empty(path, name, "DCD frame")
    header = _dcd_header(path)
    if header.magic == b"VELD":
        raise UnsupportedFormat(
            f"{name}: a velocity DCD (VELD), which holds velocities and no "
            "positions; give the coordinate DCD (CORD) of the run")
    if header.namnf > 0:
        raise UnsupportedFormat(
            f"{name}: the DCD has {header.namnf} fixed atoms (CHARMM's NAMNF), "
            "whose frames after the first hold only the free atoms; FACET reads "
            "DCD files without fixed atoms -- write the trajectory again with "
            "every atom in every frame (for example through VMD or MDAnalysis)")
    if topology is None:
        raise ValueError(
            f"{name}: a DCD holds positions only, with no element, type or atom "
            "id; pass topology=<the LAMMPS data file the run read> (FACET takes "
            "each atom's element from it in atom-id order), with "
            "type_map={<type>: '<element>', ...} when that file's Masses do not "
            "name the elements")
    user_units = _named(name, _mr._checked_units, units, "the units argument")
    step_fs = None if timestep_fs is None else _named(
        name, _checked_positive, timestep_fs, "timestep_fs")
    topo = _read_topology(topology, name, header.natom, {
        "type_map": type_map, "atom_style": atom_style,
        "masses_from": masses_from, "units": units, "mass_tol_amu": mass_tol_amu})

    notes: list[str] = []
    writer = ("LAMMPS (title 'Written by LAMMPS', CHARMM version field "
              f"{header.charmm})" if header.lammps else
              f"another program (title {header.titles[0]!r}, CHARMM version "
              f"field {header.charmm})" if header.titles else
              f"another program (no title, CHARMM version field {header.charmm})")
    notes.append(
        f"DCD written by {writer}: {header.natom} atoms, "
        f"{'int64' if header.marker == 8 else 'int32'} record markers, "
        f"{'big' if header.order == '>' else 'little'}-endian, "
        + ("a unit-cell record per frame" if header.has_cell else
           "no unit-cell records")
        + ("; a fourth coordinate per atom (CHARMM 4D), not read"
           if header.four_d else ""))
    notes.append(
        f"elements and atom ids from the topology {topo.path.resolve()} "
        f"({topo.file_format}), matched to the DCD's rows in atom-id order, the "
        "order LAMMPS writes a DCD in (dump dcd requires sorting by atom id)")

    # every frame's record markers and unit cell, checked at open
    dtype = _dcd_dtype(header)
    frame_size = dtype.itemsize
    scan = _dcd_scan(path, header, dtype, name)
    n_full = scan.n_full
    skipped: dict[int, str] = dict(scan.damaged)
    if scan.remainder:
        skipped[n_full] = (
            f"truncated: the file holds {scan.remainder} bytes after its last "
            f"complete frame, fewer than the {frame_size} a frame needs (a copy "
            "taken while the run was writing it ends so, and so do bytes "
            "appended to the file)")
    if n_full == 0:
        raise ValueError(f"{name} holds no complete frame"
                         + (f": {skipped[0]}" if skipped else ""))
    marker_ok = [k for k in range(n_full) if k not in skipped]
    convention = None
    zeroed = False
    if header.has_cell and marker_ok:
        lengths = scan.cells[:, [0, 2, 5]]
        zeroed = bool(np.all(lengths[marker_ok[0]] == 0.0))
        if zeroed:
            for k in marker_ok:
                if not np.all(lengths[k] == 0.0):
                    skipped[k] = (
                        "its unit-cell record gives lengths "
                        + ", ".join(f"{v:.6g}" for v in lengths[k])
                        + f", while frame {marker_ok[0]}'s gives 0, 0, 0 (no "
                        "unit cell)")
        else:
            first_cell = None
            for k in marker_ok:
                try:
                    _, convention = dcd_cell_to_box(scan.cells[k])
                except ValueError as error:
                    skipped[k] = str(error)
                    continue
                first_cell = k
                break
            if convention is not None:
                fits = _dcd_cells_fit(scan.cells, convention)
                for k in marker_ok:
                    if k in skipped or fits[k]:
                        continue
                    try:
                        _, other = dcd_cell_to_box(scan.cells[k])
                    except ValueError as error:
                        skipped[k] = str(error)
                    else:
                        skipped[k] = (f"its unit cell is written as {other}, "
                                      f"frame {first_cell}'s as {convention}")
    readable = [k for k in range(n_full) if k not in skipped]
    if not readable:
        raise ValueError(f"{name} holds no readable frame: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))

    fixed_box = None
    if not header.has_cell or zeroed:
        if box_from is None:
            if zeroed:
                raise ValueError(
                    f"{name}: the DCD's unit-cell records give lengths 0, 0, 0 "
                    "(MDAnalysis writes such a zeroed cell for a trajectory "
                    "without dimensions), so the file holds no box; pass "
                    "box_from=<an MD file whose frame 0 holds the box> or "
                    "box_from=<three box rows in Å>, which is then used for "
                    "every frame")
            raise ValueError(
                f"{name}: the DCD has no unit-cell records, so no box; pass "
                "box_from=<an MD file whose frame 0 holds the box> or "
                "box_from=<three box rows in Å>, which is then used for every "
                "frame")
        fixed_box, box_source = _dcd_box_from(box_from, name)
        if zeroed:
            notes.append(
                "the DCD's unit-cell records give lengths 0, 0, 0 (MDAnalysis "
                "writes such a zeroed cell for a trajectory without dimensions "
                "[15]), read as no unit cell; the box of every frame is "
                f"{box_source}")
        else:
            notes.append(f"the DCD has no unit cell; the box of every frame is "
                         f"{box_source}")
    elif box_from is not None:
        raise ValueError(f"{name}: the DCD has its own unit cell in every frame; "
                         "box_from is for a DCD without one")
    if header.nset != n_full:
        notes.append(f"the header gives NSET = {header.nset} frame(s); the file "
                     f"holds {n_full} complete frame(s)"
                     + (f", of which {len(readable)} are read"
                        if len(readable) != n_full else ", which are read"))
    if scan.cut:
        notes.append(_mr._GZIP_CUT_NOTE)

    # timesteps and times
    placeholder = (not header.lammps and bool(header.titles)
                   and header.titles[0].startswith(_MOLFILE_TITLE)
                   and header.istart == 0 and header.nsavc == 1
                   and header.delta == 1.0)
    if placeholder:
        molfile = (
            f"the DCD was written by VMD's molfile DCD plugin (title "
            f"{_MOLFILE_TITLE!r}, which VMD and mdtraj write), whose writer "
            "stores ISTART 0, NSAVC 1 and DELTA 1.0 whatever the run "
            "(dcdplugin.c, open_dcd_write [13]): placeholders, so the file "
            "holds no timestep and no time")
        if step_fs is None:
            all_steps = None
            notes.append(molfile + "; timestep_fs=<the time between frames in "
                         "fs> gives the frames times")
        else:
            all_steps = list(range(n_full))
            notes.append(molfile + "; the frame index stands for the timestep, "
                         "and timestep_fs gives the time between frames")
    else:
        all_steps, step_text = _dcd_steps(header, n_full)
        notes.append(step_text)
        if all_steps is None and step_fs is not None:
            raise ValueError(f"{name}: timestep_fs needs timesteps, and "
                             f"{step_text}")
    delta = _shortest_float32(header.delta) if header.delta_bits == 32 \
        else header.delta
    style, units_where = _mr._combine_units(
        _mr._checked_units(topo.title_units, f"{topo.path.name}'s first line")
        if topo.title_units else None, user_units,
        f"the topology {topo.path.name}'s first line")
    times = None
    if all_steps is not None:
        if placeholder:
            dt_ps = step_fs * 1e-3
            time_text = (f"times are frame index x timestep_fs {step_fs:g} fs; "
                         "the file's DELTA 1.0 is the molfile plugin's "
                         "placeholder")
        elif step_fs is not None:
            dt_ps = step_fs * 1e-3
            time_text = (f"times are timestep x timestep_fs {step_fs:g} fs; the "
                         f"file's DELTA is {delta:g}")
        elif header.lammps:
            if style is not None:
                dt_ps = delta * _mr._TIME_TO_PS[style]
                time_text = (f"times are timestep x DELTA {delta:g} (LAMMPS "
                             f"writes dt in the run's time unit; units {style} "
                             f"from {units_where}, so DELTA is "
                             f"{'ps' if style == 'metal' else 'fs'})")
            else:
                dt_ps = delta
                time_text = (
                    f"times are timestep x DELTA {delta:g} as written: LAMMPS "
                    "writes dt in the run's time unit, ps under units metal and "
                    "fs under units real, and neither units= nor a write_data "
                    "title in the topology states which, so under units real a "
                    "time read here is 1000 times its value in ps; units='metal' "
                    "or 'real', or timestep_fs=, states it")
        else:
            dt_ps = delta * AKMA_TIME_PS
            time_text = (f"times are timestep x DELTA {delta:g} in AKMA time "
                         f"units ({AKMA_TIME_PS:.7g} ps each), the unit CHARMM "
                         "and NAMD write; timestep_fs= replaces it")
        if header.delta_bits == 32 and step_fs is None:
            time_text += (f"; DELTA is stored as a 32-bit float "
                          f"({header.delta!r}), and the shortest decimal that "
                          "rounds to it is used")
        if not placeholder:
            time_text += (
                "; the time assumes this timestep size since step 0: a DCD "
                "holds one DELTA, so for a run that changed the timestep size "
                "while the dump was open, or ran at another size before it, "
                "the times differ from the run's (dump yaml, or dump custom with "
                "dump_modify time yes, records each frame's time)")
        times = [float(all_steps[k]) * dt_ps for k in readable]
        steps = [all_steps[k] for k in readable]
    else:
        time_text = "no frame times (the file gives no timesteps)"
        steps = [NO_TIMESTEP] * len(readable)

    ranges = [(header.header_bytes + k * frame_size,
               header.header_bytes + (k + 1) * frame_size) for k in readable]
    units_note = ("lengths in Å, assumed (a DCD states no unit; LAMMPS units "
                  "metal and real write Å)" if style is None else
                  f"LAMMPS units {style} ({units_where}): lengths in Å")
    units_note += "; " + time_text
    if fixed_box is not None:
        box_varies = False
    else:
        cells = scan.cells[readable]
        box_varies = bool(np.any(cells != cells[0]))
    trajectory = _DcdTrajectory(
        path, ranges, header=header, dtype=dtype, elements=topo.elements,
        ids=topo.ids, fixed_box=fixed_box, convention=convention,
        unwrapped=False, periodic=None, file_format=DCD_FORMAT,
        n_atoms=header.natom, n_frames=len(readable),
        type_map_source=topo.source, timesteps=steps, times_ps=times,
        type_map=topo.type_map, charges_e=topo.charges_e, skipped=skipped,
        notes=notes, units_note=units_note, box_varies=box_varies)
    if fixed_box is None:
        box0, _ = dcd_cell_to_box(scan.cells[readable[0]])
        trajectory.notes.append(
            f"unit cell read as {convention}"
            + (" (dump_dcd.cpp writes a, cos gamma, b, cos beta, cos alpha, c)"
               if header.lammps else "")
            + "; the box is rebuilt with a along x and b in the xy plane")
        if not header.lammps and convention.endswith("cosines") \
                and np.any(box0 - np.diag(np.diag(box0))):
            trajectory.notes.append(
                "newer CHARMM versions write the symmetric box matrix in the "
                "unit-cell slots, which reads as these cosines only for an "
                "orthogonal box; this box is not orthogonal, so it holds only "
                "if the writer used lengths and cosines (NAMD, VMD, MDAnalysis "
                "and LAMMPS do)")
    state, evidence, _ = _dcd_wrap_state(trajectory)
    trajectory._unwrapped = state == "unwrapped"
    trajectory.notes.append(evidence)
    frame0 = _mr._frame0(name, readable[0], lambda: trajectory.frame(0))
    biggest = float(np.abs(trajectory._raw(0)[0]).max())
    trajectory.notes.append(
        "positions are stored as 32-bit floats: each is within 2^-24 of its "
        f"magnitude of the value written, {biggest * 2.0 ** -24:.2g} Å at frame "
        f"0's largest coordinate ({biggest:.4g} Å)")
    trajectory.notes.append(
        "the DCD holds no box origin; positions are read relative to (0, 0, 0) "
        "and wrapped into the box with that corner, so each atom keeps its "
        "position modulo the box vectors, and the cell holds other images of "
        "the atoms than LAMMPS's xlo..xhi box")
    trajectory.notes.append(
        "the DCD does not record the boundary; the box is taken as periodic "
        "along a, b and c")
    if topo.charges_e is not None:
        trajectory.notes.append(f"charges per element from the topology "
                                f"{topo.path.name}; a DCD holds positions only")
    return _mr._finish(trajectory, frame0)


# ---------------------------------------------------------------------------
# LAMMPS binary dump
# ---------------------------------------------------------------------------

@dataclass
class _BinFrame:
    position: int
    reason: str | None = None
    timestep: int = 0
    n_atoms: int = 0
    triclinic: int = 0
    boundary: str = ""
    periodic: tuple[bool, bool, bool] | None = None
    box: np.ndarray | None = None
    origin: np.ndarray | None = None
    size_one: int = 0
    units: str | None = None
    time: float | None = None
    columns: tuple[str, ...] | None = None
    revision: int = 0
    body_start: int = 0
    body_end: int = 0


def _binary_layout(head: bytes, path: Path | None) -> tuple[str, bytes | None] | None:
    """(byte order, magic or None for the older layout), or None."""
    for order in "<>":
        if len(head) < 8:
            return None
        value = struct.unpack(order + "q", head[:8])[0]
        if -16 <= value < 0:
            length = -value
            magic = head[8:8 + length]
            if magic in _BINARY_MAGIC and len(head) >= 16 + length:
                endian = struct.unpack(order + "i", head[8 + length:12 + length])[0]
                if endian == 1:
                    return order, magic
            elif magic in _BINARY_MAGIC:
                # the file ends inside its first header: the reader says so
                # (truncated), where a generic refusal would not
                return order, magic
    name = "" if path is None else path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    if not name.endswith(_BINARY_SUFFIXES):
        return None
    for order in "<>":
        if _old_header_plausible(head, order):
            return order, None
    return None


def _old_header_plausible(head: bytes, order: str) -> bool:
    """The older binary layout's first header (no magic string) parses with
    values a LAMMPS run writes."""
    try:
        step, natoms = struct.unpack(order + "qq", head[:16])
        triclinic = struct.unpack(order + "i", head[16:20])[0]
        boundary = struct.unpack(order + "6i", head[20:44])
        if not (0 <= step < 1 << 62 and 0 < natoms < 1 << 40
                and triclinic in (0, 1) and all(b in _BOUNDARY_FLAGS
                                                 for b in boundary)):
            return False
        nbox = 6 if triclinic == 0 else 9
        box = struct.unpack(order + f"{nbox}d", head[44:44 + 8 * nbox])
        at = 44 + 8 * nbox
        size_one, nchunk = struct.unpack(order + "ii", head[at:at + 8])
        first_n = struct.unpack(order + "i", head[at + 8:at + 12])[0]
    except struct.error:
        return False
    if not all(math.isfinite(v) for v in box):
        return False
    if not (box[0] < box[1] and box[2] < box[3] and box[4] < box[5]):
        return False
    return 0 < size_one <= _MAX_COLUMNS and 0 < nchunk <= _MAX_CHUNKS \
        and first_n >= 0 and first_n % size_one == 0


def sniff_lammps_binary(head: bytes, path: Path) -> bool:
    """A LAMMPS binary dump: the magic string DUMPCUSTOM or DUMPATOM after a
    negative int64; or, for a file named ``*.bin`` / ``*.lammpsbin``, an older
    header whose values parse."""
    try:
        return _binary_layout(head, Path(path)) is not None
    except Exception:       # noqa: BLE001 - a sniffer never raises
        return False


class _Ends(Exception):
    """The file ends inside a binary header or chunk."""


class _Reader:
    """A binary file read forward, for the index pass of a binary dump: the
    offset of the next byte, bytes pushed back after a search, and whether a
    gzip stream was cut. Every read goes through :func:`_read_exact`, so a
    cut or damaged gzip stream is reported as for any other read."""

    def __init__(self, handle, name: str) -> None:
        self.handle = handle
        self.name = name
        self.gzip = isinstance(handle, gzip.GzipFile)
        self.offset = 0
        self.pending = b""
        self.cut = _Cut()

    def read(self, size: int) -> bytes:
        head = b""
        if self.pending:
            head, self.pending = self.pending[:size], self.pending[size:]
        rest = b"" if len(head) >= size else _read_exact(
            self.handle, size - len(head), self.name, self.cut)
        data = head + rest
        self.offset += len(data)
        return data

    def skip(self, size: int) -> bool:
        """Move ``size`` bytes forward; False when the file ends first."""
        if self.pending and size > 0:
            used = min(size, len(self.pending))
            self.pending = self.pending[used:]
            self.offset += used
            size -= used
        if size <= 0:
            return True
        if self.gzip:
            while size > 0:
                want = min(size, _READ_PIECE)
                piece = _read_exact(self.handle, want, self.name, self.cut)
                self.offset += len(piece)
                size -= len(piece)
                if len(piece) < want:
                    return False
            return True
        start = self.handle.tell()
        self.handle.seek(0, 2)
        end = self.handle.tell()
        if start + size > end:
            self.offset += end - start
            return False
        self.handle.seek(start + size)
        self.offset += size
        return True

    def find(self, pattern: bytes, start: int) -> int | None:
        """Move to the next ``pattern`` at or after byte ``start`` (a plain
        file) or after the bytes already read (a gzip stream, which is not
        decompressed twice); its offset, or None, with the reader at the end
        of the file."""
        if not self.gzip:
            self.handle.seek(start)
            self.pending = b""
            self.offset = start
        tail = b""
        base = self.offset
        while True:
            data = self.read(1 << 20)
            if not data:
                return None
            buf = tail + data
            hit = buf.find(pattern)
            if hit >= 0:
                found = base - len(tail) + hit
                self.pending = buf[hit:] + self.pending
                self.offset = found
                return found
            tail = buf[-(len(pattern) - 1):] if len(pattern) > 1 else b""
            base += len(data)


def _unpack(reader: _Reader, fmt: str):
    size = struct.calcsize(fmt)
    data = reader.read(size)
    if len(data) < size:
        raise _Ends()
    return struct.unpack(fmt, data)


def _binary_frame_header(reader: _Reader, lead: bytes, order: str,
                         magic: bytes | None, frame: _BinFrame) -> None:
    """Fill ``frame`` from its header, whose first 8 bytes are ``lead``.
    Raises _Ends at the end of the file and ValueError for values no LAMMPS
    run writes. A frame of 0 atoms (``dump_modify thresh`` or a group that
    holds no atom at that step) is a valid header; its chunks hold 0 values."""
    (value,) = struct.unpack(order + "q", lead)
    if magic is not None:
        if value != -len(magic):
            raise ValueError(f"the frame starts with {value} where the magic "
                             f"string's negative length -{len(magic)} is "
                             "expected")
        text = reader.read(len(magic))
        if len(text) < len(magic):
            raise _Ends()
        if text != magic:
            raise ValueError(f"the frame's magic string is {text!r}, not "
                             f"{magic!r}")
        endian, revision = _unpack(reader, order + "ii")
        if endian != 1:
            raise ValueError(f"the endian flag reads {endian}, not 1")
        frame.revision = revision
        (step,) = _unpack(reader, order + "q")
    else:
        step = value
    (natoms,) = _unpack(reader, order + "q")
    (triclinic,) = _unpack(reader, order + "i")
    boundary = _unpack(reader, order + "6i")
    if not 0 <= natoms < 1 << 40:
        raise ValueError(f"the atom count reads {natoms}")
    if triclinic not in (0, 1, 2) or (magic is None and triclinic == 2):
        raise ValueError(f"the triclinic flag reads {triclinic}")
    if any(b not in _BOUNDARY_FLAGS for b in boundary):
        raise ValueError(f"the boundary flags read {list(boundary)}")
    nbox = {0: 6, 1: 9, 2: 12}[triclinic]
    box = _unpack(reader, order + f"{nbox}d")
    (size_one,) = _unpack(reader, order + "i")
    if not 0 < size_one <= _MAX_COLUMNS:
        raise ValueError(f"the column count reads {size_one}")
    frame.timestep, frame.n_atoms, frame.triclinic = int(step), int(natoms), triclinic
    frame.size_one = size_one
    flags = ["".join(_BOUNDARY_FLAGS[b] for b in boundary[2 * i:2 * i + 2])
             for i in range(3)]
    frame.boundary = " ".join(flags)
    frame.periodic = tuple(f == "pp" for f in flags)
    if triclinic == 0:
        frame.box, frame.origin = _mr.box_from_dump_bounds(
            [box[0:2], box[2:4], box[4:6]], None)
    elif triclinic == 1:
        frame.box, frame.origin = _mr.box_from_dump_bounds(
            [box[0:2], box[2:4], box[4:6]], box[6:9])
    else:
        frame.box = np.array(box[0:9], dtype=np.float64).reshape(3, 3)
        frame.origin = np.array(box[9:12], dtype=np.float64)
        md_model._checked_box(frame.box)
    if magic is not None and frame.revision > 1:
        (length,) = _unpack(reader, order + "i")
        if length > 0:
            if length > 256:
                raise ValueError(f"the unit-style length reads {length}")
            text = reader.read(length)
            if len(text) < length:
                raise _Ends()
            frame.units = text.decode("ascii", "replace")
        flag = reader.read(1)
        if len(flag) < 1:
            raise _Ends()
        if flag != b"\0":
            (frame.time,) = _unpack(reader, order + "d")
        (length,) = _unpack(reader, order + "i")
        if not 0 < length <= 1 << 20:
            raise ValueError(f"the column-name length reads {length}")
        text = reader.read(length)
        if len(text) < length:
            raise _Ends()
        frame.columns = tuple(text.decode("ascii", "replace").split())
    if not 0 <= step < 1 << 62:
        raise ValueError(f"the timestep reads {step}")


def _binary_index(path: Path, order: str, magic: bytes | None
                  ) -> tuple[list[_BinFrame], list[str]]:
    """Every frame's header and the byte range of its chunks, in one forward
    pass (chunks are skipped, so a gzip file is decompressed once).

    In the magic-string layout every frame starts with int64 -len and the
    magic string, so after a header that does not parse, chunks whose counts
    disagree with the header, or a length that runs past the end of the file,
    the pass searches for the next frame's magic string and goes on from
    there; the bytes passed over take one position in ``skipped``, with their
    byte range. (A plain file is searched from the byte after the damaged
    frame's start; a gzip stream from where the reading stopped, so it is not
    decompressed twice.) The older layout has no such marker, and the pass
    stops at the first header that does not parse.
    """
    name = path.name
    frames: list[_BinFrame] = []
    notes: list[str] = []
    marker = None if magic is None else \
        struct.pack(order + "q", -len(magic)) + magic
    with _mr._open_binary(path) as handle:
        reader = _Reader(handle, name)
        while True:
            start = reader.offset
            lead = reader.read(8)
            if not lead:
                break
            frame = _BinFrame(position=len(frames))
            frames.append(frame)
            try:
                if len(lead) < 8:
                    raise _Ends()
                _binary_frame_header(reader, lead, order, magic, frame)
                (nchunk,) = _unpack(reader, order + "i")
                if not 0 < nchunk <= _MAX_CHUNKS:
                    raise ValueError(f"the chunk count reads {nchunk}")
                frame.body_start = reader.offset
                total = 0
                for _ in range(nchunk):
                    (count,) = _unpack(reader, order + "i")
                    if count < 0 or count % frame.size_one:
                        raise ValueError(f"a chunk holds {count} values, not a "
                                         f"multiple of the {frame.size_one} "
                                         "columns")
                    if not reader.skip(8 * count):
                        raise _Ends()
                    total += count
                frame.body_end = reader.offset
            except _Ends:
                found = None
                if marker is not None and not reader.cut.seen:
                    found = reader.find(marker, start + 1)
                if found is not None:
                    frame.reason = (
                        "unreadable: a length in its header or chunks reaches "
                        "past the end of the file, while a frame's magic string "
                        f"follows at byte {found}; bytes {start} to {found} are "
                        "not read, and any frame in them is not counted")
                    continue
                frame.reason = (
                    f"truncated: the file ends inside this frame, "
                    f"{reader.offset - start} bytes from its start at byte "
                    f"{start} (a copy taken while the run was writing it ends "
                    "so, and so do bytes appended after the last frame)")
                break
            except (ValueError, struct.error) as error:
                if marker is not None:
                    found = reader.find(marker, start + 1)
                    if found is not None:
                        frame.reason = (
                            f"unreadable header ({error}); bytes {start} to "
                            f"{found}, up to the next frame's magic string, are "
                            "not read, and any frame in them is not counted")
                        continue
                    rest = reader.offset - start
                    after = (" and no frame's magic string follows it, so the "
                             "rest of the file")
                else:
                    rest = _file_size(path, name) - start
                    after = (", so the rest of the file")
                frame.reason = (f"unreadable header ({error}); a binary dump "
                                "cannot be followed past a header that does not "
                                f"parse{after} ({max(rest, 0)} bytes from byte "
                                f"{start}) is not located and any frames in it "
                                "are not counted")
                break
            if total != frame.n_atoms * frame.size_one:
                frame.reason = (f"its chunks hold {total} values where "
                                f"{frame.n_atoms} atoms x {frame.size_one} "
                                f"columns = {frame.n_atoms * frame.size_one}"
                                " are expected")
                if marker is not None:
                    # the counts disagree, so where the next frame starts is
                    # not known from them: the next magic string tells
                    found = reader.find(marker, start + 1)
                    if found is None:
                        frame.reason += (
                            f"; no frame's magic string follows it, so the rest "
                            f"of the file ({max(reader.offset - frame.body_end, 0)}"
                            f" bytes from byte {frame.body_end}) is not located "
                            "and any frames in it are not counted")
                        break
                    if found != frame.body_end:
                        frame.reason += (
                            f"; the next frame's magic string is at byte {found}, "
                            f"not at byte {frame.body_end} where its chunks end, "
                            "and reading goes on from there")
            elif frame.n_atoms == 0:
                frame.reason = "the atom count is 0: no atoms"
    if reader.cut.seen:
        notes.append(_mr._GZIP_CUT_NOTE)
    return frames, notes


def _binary_table(data: bytes, frame: _BinFrame, order: str) -> np.ndarray:
    """(n_atoms, size_one) float64 from a frame's chunks."""
    pieces = []
    position = 0
    while position < len(data):
        if position + 4 > len(data):
            raise ValueError("a chunk is cut inside its length")
        count = struct.unpack(order + "i", data[position:position + 4])[0]
        position += 4
        end = position + 8 * count
        if count < 0 or end > len(data):
            raise ValueError("a chunk is cut inside its values")
        pieces.append(np.frombuffer(data, dtype=order + "f8", count=count,
                                    offset=position))
        position = end
    values = np.concatenate(pieces) if pieces else np.zeros(0)
    if values.size != frame.n_atoms * frame.size_one:
        raise ValueError(f"the chunks hold {values.size} values where "
                         f"{frame.n_atoms * frame.size_one} are expected")
    return values.astype(np.float64).reshape(frame.n_atoms, frame.size_one)


def _identity_columns(columns: Sequence[str]
                      ) -> tuple[tuple[str, ...], dict, list[str]]:
    """Columns as the identity code sees them, the names to show for renamed
    ones, and a note: in a binary dump element and typelabel hold type
    numbers."""
    view = list(columns)
    shown: dict[str, str] = {}
    holders = [c for c in ("element", "typelabel") if c in view]
    notes = []
    if not holders:
        return tuple(view), shown, notes
    if "type" in view:
        renamed = holders
    else:
        view[view.index(holders[0])] = "type"
        shown["type"] = holders[0]
        renamed = holders[1:]
    for column in renamed:
        alias = f"{column}#number"
        view[view.index(column)] = alias
        shown[alias] = column
    notes.append(
        "the " + " and ".join(holders) + " column(s) of a binary dump hold the "
        "type number (LAMMPS packs the type for them and writes element names "
        "and type labels only to text dumps), so they are read as type numbers"
        + ("" if "type" not in columns else
           "; the type column gives the types"))
    return tuple(view), shown, notes


def _shown(text: str, shown: Mapping[str, str]) -> str:
    for alias, original in shown.items():
        if alias != "type":
            text = text.replace(alias, f"{original} (type numbers, not read)")
    return text


def _numeric_frame(table: np.ndarray, columns: Sequence[str], block, identity,
                   velocity_factor: float, time_ps: float | None,
                   where: str) -> Frame:
    """One frame from a numeric table: the text dump's
    :func:`.md_readers._dump_frame` on float64 columns."""
    index = {c: i for i, c in enumerate(columns)}

    def col(names):
        return table[:, [index[n] for n in names]]

    types = _whole(table[:, index["type"]], "type", where)
    mapping = identity[1]
    unique, inverse = np.unique(types, return_inverse=True)
    missing = [int(t) for t in unique if int(t) not in mapping]
    if missing:
        raise ValueError(
            f"type(s) {missing} occur in this frame; the type map was built when "
            "the file was opened, from frame 0 and the sources given, and names "
            "no element for them; pass type_map= with these types")
    elements = np.array([mapping[int(t)] for t in unique],
                        dtype="<U2")[inverse.reshape(-1)]
    ids = _whole(table[:, index["id"]], "id", where) if "id" in index else None
    kind, names = block.wrapped
    raw = col(names)
    origin, box = block.origin, block.box
    unwrapped = None
    if block.unwrapped is not None:
        ukind, unames = block.unwrapped
        values = raw if unames == names else col(unames)
        unwrapped = values if ukind == "xu" else origin + values @ box
    elif all(c in index for c in _mr._IMAGE_COLUMNS):
        image = _whole(col(_mr._IMAGE_COLUMNS), "ix iy iz", where)
        absolute = raw if kind == "x" else origin + raw @ box
        unwrapped = unwrap_with_images(absolute, image, box)
    velocities = None
    if all(c in index for c in _mr._VELOCITY_COLUMNS):
        velocities = col(_mr._VELOCITY_COLUMNS) * velocity_factor
    charge = table[:, index["q"]].copy() if "q" in index else None
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


class _BinaryTrajectory(_mr._FileTrajectory):
    def __init__(self, path: Path, frames: Sequence[_BinFrame], *, order: str,
                 view: tuple[str, ...], blocks, identity, velocity_factor: float,
                 **kwargs) -> None:
        super().__init__(path, [(f.body_start, f.body_end) for f in frames],
                         **kwargs)
        self._frames = list(frames)
        self._order = order
        self._view = view
        self._blocks = list(blocks)
        self._identity = identity
        self._velocity_factor = velocity_factor
        self._composition: dict[str, int] | None = None

    def _load(self, k: int) -> Frame:
        time = None
        if self.times_ps is not None and np.isfinite(self.times_ps[k]):
            time = float(self.times_ps[k])
        table = _binary_table(self._bytes(k), self._frames[k], self._order)
        frame = _numeric_frame(table, self._view, self._blocks[k], self._identity,
                               self._velocity_factor, time, "the chunks")
        if self._composition is not None and frame.composition != self._composition:
            raise ValueError(
                f"this frame holds {frame.composition}, frame 0 holds "
                f"{self._composition}; the dump has no id column, so its atoms "
                "are matched between frames by element only")
        return frame


def _columns_option(columns, name: str) -> tuple[str, ...] | None:
    if columns is None:
        return None
    if isinstance(columns, str):
        names = tuple(columns.split())
    elif isinstance(columns, Sequence) and all(isinstance(c, str) for c in columns):
        names = tuple(c.strip() for c in columns)
    else:
        raise ValueError(f"{name}: columns needs the column names, as text "
                         "('id type x y z') or a list of names")
    if not names or any(not c for c in names):
        raise ValueError(f"{name}: columns names no column")
    if len(set(names)) != len(names):
        raise ValueError(f"{name}: columns names a column twice")
    return names


def read_lammps_binary_dump(path, *, type_map: Mapping | None = None,
                            masses_from=None, units: str | None = None,
                            mass_tol_amu: float | None = None,
                            columns=None) -> Trajectory:
    """A LAMMPS binary dump (``dump atom`` or ``custom`` to ``*.bin``).

    Both layouts ``tools/binary2txt`` reads (module docstring), in either byte
    order. The columns come from the file (revision 2 of the magic-string
    layout); a file of the older layout or of revision 1 names none, and
    ``columns`` gives them ('id type x y z', as the dump command listed them).
    Columns, type maps (``type_map``, ``masses_from``), units, image flags,
    velocities and charges are read as :func:`.md_readers.read_lammps_dump`
    reads a text dump's; ``element`` and ``typelabel`` columns hold type
    numbers in a binary file and are read as such.
    """
    path = _mr._existing(path)
    name = path.name
    tol = _named(name, _mr._checked_tol, _mr.MASS_TOL_AMU if mass_tol_amu is None
                 else mass_tol_amu)
    user_types, user_labels = _named(name, _mr._normalise_type_map, type_map)
    user_units = _named(name, _mr._checked_units, units, "the units argument")
    given_columns = _columns_option(columns, name)
    _refuse_empty(path, name, "binary dump frame")
    layout = _binary_layout(_mr._head(path, 4096), path)
    if layout is None:
        raise UnsupportedFormat(
            f"{name}: not a LAMMPS binary dump (no DUMPCUSTOM or DUMPATOM magic "
            "string, and no older header that parses under a .bin name); "
            f"{_what_facet_reads()}")
    order, magic = layout
    frames, notes = _binary_index(path, order, magic)

    file_columns = [f.columns for f in frames if f.columns]
    unnamed = [f for f in frames if f.reason is None and not f.columns]
    stated = None
    if file_columns:
        stated = file_columns[0]
        names = stated
        source = "the file's header"
        if given_columns is not None and tuple(given_columns) != stated:
            if len(given_columns) != len(stated):
                raise ValueError(
                    f"{name}: the file names its columns ({' '.join(stated)}; "
                    f"{len(stated)} of them), and columns= gives "
                    f"{len(given_columns)} ({' '.join(given_columns)}); columns= "
                    "replaces the stated names one for one")
            names = tuple(given_columns)
            source = "the columns argument"
            notes.append(f"columns= replaces the column names the file states "
                         f"({' '.join(stated)}) with {' '.join(names)}, in the "
                         "same order")
    elif not unnamed:
        # no frame got as far as naming its columns: the reason is damage or
        # a cut, not a layout without column names
        skipped = {f.position: f.reason for f in frames if f.reason is not None}
        raise ValueError(f"{name} holds no readable frame: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))
    else:
        if given_columns is None:
            sizes = sorted({f.size_one for f in unnamed if f.size_one})
            layout_name = ("the layout without a magic string" if magic is None
                           else f"revision {unnamed[0].revision} of the "
                           "magic-string layout")
            raise ValueError(
                f"{name}: a LAMMPS binary dump in {layout_name} does not name "
                f"its {'/'.join(str(s) for s in sizes) or 'per-atom'} columns; "
                "pass columns='<the per-atom keywords of the dump command, in "
                "order>' (for example columns='id type x y z' for dump custom, "
                "or columns='id type xs ys zs' for dump atom with its default "
                "scaled positions)")
        names = given_columns
        source = "the columns argument"
    for frame in frames:
        if frame.reason is not None:
            continue
        if frame.columns is None:
            frame.columns = tuple(names)
        elif stated is not None and frame.columns == stated:
            frame.columns = tuple(names)
        if frame.columns != tuple(names):
            frame.reason = (f"its columns ({' '.join(frame.columns)}) differ "
                            f"from frame 0's ({' '.join(stated or names)})")
        elif frame.size_one != len(names):
            frame.reason = (f"it holds {frame.size_one} columns where "
                            f"{source} names {len(names)}")
    readable = [f for f in frames if f.reason is None]
    if readable:
        counts = [f.n_atoms for f in readable]
        n0 = _mr._majority_count(counts)
        for frame in readable:
            if frame.n_atoms != n0:
                frame.reason = _mr._count_reason(frame.n_atoms, n0, counts)
    readable = [f for f in frames if f.reason is None]
    skipped = {f.position: f.reason for f in frames if f.reason is not None}
    if not readable:
        raise ValueError(f"{name} holds no readable frame: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))

    view, shown, column_notes = _identity_columns(names)
    positions = _positions_of(view)
    if isinstance(positions, str):
        raise ValueError(f"{name}: {positions} ({source}: {' '.join(names)}); "
                         + _COLNAME_ADVICE)
    if "type" not in view:
        raise ValueError(f"{name}: the columns ({' '.join(names)}) hold no type, "
                         "element or typelabel column, so the atoms cannot be "
                         "given elements; dump the type column (id type x y z)")
    blocks = []
    for frame in readable:
        block = _mr._DumpBlock(position=frame.position, timestep=frame.timestep,
                               n_atoms=frame.n_atoms, box=frame.box,
                               origin=frame.origin, periodic=frame.periodic,
                               boundary=frame.boundary, columns=view,
                               wrapped=positions[0], unwrapped=positions[1],
                               time=frame.time, units=frame.units)
        blocks.append(block)

    file_units = sorted({f.units for f in frames if f.units})
    if len(file_units) > 1:
        raise ValueError(f"{name}: the unit style reads {file_units} in "
                         "different frames")
    stated = _mr._checked_units(file_units[0], f"{name}'s unit-style field") \
        if file_units else None
    style, units_where = _mr._combine_units(stated, user_units,
                                            f"{name}'s unit-style field")
    time_factor = _mr._TIME_TO_PS[style] if style else 1.0
    velocity_factor = _mr._VELOCITY_TO_ANG_PER_PS[style] if style else 1.0
    times = None
    if any(f.time is not None for f in readable):
        times = [np.nan if f.time is None else f.time * time_factor
                 for f in readable]

    first, block0 = readable[0], blocks[0]
    data0 = _mr._read_range(path, first.body_start, first.body_end)
    table0 = _mr._frame0(name, first.position,
                         lambda: _binary_table(data0, first, order))
    for column in ("type", "id"):
        if column in view:
            _mr._frame0(name, first.position, lambda c=column: _whole(
                table0[:, view.index(c)], shown.get(c, c), "the chunks"))
    identity, listed, type_source, type_notes = _mr._frame0(
        name, first.position, lambda: _mr._dump_identity(
            table0, block0, user_types, user_labels, masses_from, tol, name))
    notes.extend(column_notes)
    notes.extend(type_notes)
    notes.append(
        f"LAMMPS binary dump, {'big' if order == '>' else 'little'}-endian, "
        + (f"magic string {magic.decode()} revision {first.revision}"
           if magic is not None else "the layout without a magic string")
        + f"; columns from {source}")
    used, unread = _mr._dump_columns(block0, identity)
    used = [shown.get(c, c) for c in used]
    unread = [_shown(c, shown) for c in unread]
    notes.append(f"frame positions from {' '.join(block0.wrapped[1])}"
                 + (f"; unwrapped positions from {' '.join(block0.unwrapped[1])}"
                    if block0.unwrapped else
                    "; unwrapped positions from the image flags ix iy iz"
                    if all(c in view for c in _mr._IMAGE_COLUMNS) else
                    "; no unwrapped positions (neither xu/xsu nor ix iy iz)"))
    notes.append("columns read: " + " ".join(used)
                 + (("; not read: " + ", ".join(unread)) if unread else ""))
    if "id" not in view:
        notes.append(
            "no id column: in every frame the rows are put in element order and "
            "numbered 0 .. N-1; LAMMPS re-orders atoms between frames, so atom "
            "k is not the same atom in two frames (ids_track_atoms is False)")
    if any(f.triclinic == 2 for f in readable):
        notes.append("general triclinic box: the box rows are avec, bvec, cvec "
                     "and the origin as written")
    notes.extend(_boundary_note([f.periodic for f in readable],
                                [f.boundary for f in readable],
                                "the binary header"))
    boundaries = sorted({f.boundary for f in readable})
    box_varies = any(not np.array_equal(f.box, first.box)
                     or not np.array_equal(f.origin, first.origin)
                     for f in readable[1:])
    trajectory = _BinaryTrajectory(
        path, readable, order=order, view=view, blocks=blocks,
        identity=identity, velocity_factor=velocity_factor,
        periodic=None if len(boundaries) != 1 else first.periodic,
        file_format=BINARY_FORMAT, n_atoms=first.n_atoms,
        n_frames=len(readable), type_map_source=type_source,
        timesteps=[f.timestep for f in readable], times_ps=times,
        type_map=listed, skipped=skipped, notes=notes,
        units_note=_mr._units_note(style, units_where,
                                   has_time=times is not None,
                                   has_velocity=all(c in view for c in
                                                    _mr._VELOCITY_COLUMNS)),
        box_varies=box_varies)
    frame0 = _mr._frame0(name, first.position, lambda: _numeric_frame(
        table0, view, block0, identity, velocity_factor,
        None if times is None or not np.isfinite(times[0]) else float(times[0]),
        "the chunks"))
    if "id" not in view:
        trajectory.ids_track_atoms = False
        trajectory._composition = frame0.composition
    if frame0.charge_e is not None:
        charges, charge_notes = md_model.charges_per_element(frame0.elements,
                                                             frame0.charge_e)
        trajectory.charges_e = charges
        trajectory.notes.extend(charge_notes)
    return _mr._finish(trajectory, frame0)


# ---------------------------------------------------------------------------
# LAMMPS YAML dump
# ---------------------------------------------------------------------------

_YAML_FOOTER = re.compile(rb"^(?:\.\.\.|---)[ \t]*\r?$", re.MULTILINE)


def sniff_lammps_yaml(head: bytes, path: Path) -> bool:
    """A LAMMPS YAML dump: a '---' line, then 'creator: LAMMPS' (or the
    timestep, natoms and box keys a LAMMPS dump writes)."""
    try:
        text = _mr._strip_bom(head).decode("utf-8", "replace")
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if not lines or lines[0] != "---":
            return False
        keys = {x.split(":", 1)[0] for x in lines[1:20] if ":" in x}
        return "creator: LAMMPS" in lines[1:4] or \
            {"timestep", "natoms", "box"} <= keys
    except Exception:       # noqa: BLE001 - a sniffer never raises
        return False


def _flow_list(value: str, key: str) -> list[str]:
    """The items of a YAML flow sequence as LAMMPS writes one: '[ a, b, ]'."""
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"{key} is {value.strip()!r}, not a [ ... ] list")
    items = [item.strip() for item in text[1:-1].split(",")]
    while items and items[-1] == "":
        items.pop()
    out = []
    for item in items:
        if len(item) >= 2 and item[0] == item[-1] and item[0] in "'\"":
            item = item[1:-1]
        out.append(item)
    return out


@dataclass
class _YamlHeader:
    lines: int                       # header lines from '---' to 'keywords:'
    footer_before: bool              # a '...' line precedes the '---'
    values: dict = field(default_factory=dict)


def _yaml_line(line: str) -> str:
    """A line stripped of blanks and of a leading byte-order mark (Windows
    editors and PowerShell's Out-File write one; the scan keeps it)."""
    return line.strip().lstrip("\ufeff").strip()


def _yaml_header(before: Sequence[str]) -> _YamlHeader | None:
    """The header of the frame whose 'data:' line follows ``before``: from
    the last '---' line, which has to come after the previous frame's '...'
    footer and 'data:' line; None when no such '---' line is there."""
    start = None
    for i in range(len(before) - 1, -1, -1):
        text = _yaml_line(before[i])
        if text == "---":
            start = i
            break
        if text == "..." or text.startswith("data:"):
            break
    if start is None:
        return None
    header = _YamlHeader(lines=len(before) - start,
                         footer_before=start > 0 and before[start - 1].strip()
                         == "...")
    section = None
    for line in before[start + 1:]:
        if not line.strip():
            continue
        if line[:1] in (" ", "\t"):
            if section is not None:
                header.values.setdefault(section, []).append(line.strip())
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        section = key if not value.strip() else None
        if value.strip():
            header.values[key] = value.strip()
        else:
            header.values.setdefault(key, [])
    return header


def _yaml_block(header: _YamlHeader, position: int,
                given: Sequence[str] | None = None) -> tuple[object, str | None]:
    """A text-dump block from a YAML header, and why it cannot be read;
    ``given`` (the columns option) replaces the keywords name for name."""
    block = _mr._DumpBlock(position=position)
    values = header.values
    missing = [k for k in ("timestep", "natoms", "box", "keywords")
               if k not in values]
    if missing:
        return block, f"the frame header has no {', '.join(missing)}"
    try:
        block.timestep = int(values["timestep"])
    except ValueError:
        return block, f"unreadable: timestep is {values['timestep']!r}"
    try:
        block.n_atoms = int(values["natoms"])
    except ValueError:
        return block, f"unreadable: natoms is {values['natoms']!r}"
    if block.n_atoms < 1:
        return block, f"natoms is {block.n_atoms}: no atoms"
    if "units" in values:
        block.units = str(values["units"])
    if "time" in values:
        try:
            block.time = float(values["time"])
        except ValueError:
            return block, f"unreadable: time is {values['time']!r}"
        if not math.isfinite(block.time):
            return block, "unreadable: time is not finite"
    try:
        rows = values["box"]
        if not isinstance(rows, list) or len(rows) not in (3, 4):
            raise ValueError(f"box has {len(rows) if isinstance(rows, list) else 0} "
                             "rows where 3 (orthogonal) or 4 (tilted) are expected")
        parsed = [_mr._finite(_flow_list(r.lstrip("- ").strip()
                                         if r.startswith("-") else r, "a box row"),
                              "box") for r in rows]
        if any(len(p) != 2 for p in parsed[:3]) or \
                (len(parsed) == 4 and len(parsed[3]) != 3):
            raise ValueError("a box row holds another number of values than "
                             "[ lo, hi ] or [ xy, xz, yz ]")
        block.box, block.origin = _mr.box_from_dump_bounds(
            parsed[:3], parsed[3] if len(parsed) == 4 else None)
        md_model._checked_box(block.box)
        if "boundary" in values:
            flags = _flow_list(values["boundary"], "boundary")
            if len(flags) == 6:
                pairs = ["".join(flags[2 * i:2 * i + 2]) for i in range(3)]
                block.boundary = " ".join(pairs)
                block.periodic = tuple(p == "pp" for p in pairs)
        block.columns = tuple(_flow_list(values["keywords"], "keywords"))
    except ValueError as error:
        return block, f"unreadable: {error}"
    if not block.columns:
        return block, "keywords names no columns"
    if given is not None:
        if len(given) != len(block.columns):
            return block, (f"keywords names {len(block.columns)} columns "
                           f"({' '.join(block.columns)}), and columns= gives "
                           f"{len(given)}; columns= replaces the keywords one "
                           "for one")
        block.columns = tuple(given)
    if len(set(block.columns)) != len(block.columns):
        return block, "keywords names a column twice"
    positions = _positions_of(block.columns)
    if isinstance(positions, str):
        return block, positions
    block.wrapped, block.unwrapped = positions
    return block, None


def _yaml_rows(data: bytes) -> bytes:
    """The data rows of one frame, up to its '...' footer or the next '---'."""
    match = _YAML_FOOTER.search(data)
    return data if match is None else data[:match.start()]


def _yaml_text(rows: bytes) -> bytes:
    """The rows as a text dump's rows: '- [ 1 , 3 , O , -1.7, ]' -> '1 3 O -1.7'."""
    return rows.replace(b"- [", b" ").replace(b",", b" ").replace(b"]", b" ")


def _yaml_fields(rows: bytes, width: int, n: int) -> list[list[bytes]]:
    """Each row's fields split at commas, empty fields kept."""
    out = []
    for line in rows.splitlines():
        text = line.strip()
        if not text:
            continue
        if not (text.startswith(b"- [") and text.endswith(b"]")):
            raise ValueError(f"a data row reads {text[:60]!r}, not '- [ ... ]'")
        fields = [f.strip() for f in text[3:-1].split(b",")]
        if fields and fields[-1] == b"":
            fields.pop()
        if len(fields) != width:
            raise ValueError(f"a data row holds {len(fields)} values where "
                             f"keywords names {width}")
        out.append(fields)
    if len(out) != n:
        raise ValueError(f"the data holds {len(out)} rows where natoms gives {n}")
    return out


class _YamlTrajectory(_mr._DumpTrajectory):
    """The text dump's trajectory over YAML rows, turned into a text dump's
    rows when each frame is loaded."""

    def __init__(self, *args, drop: Sequence[int] = (), width: int = 0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drop = tuple(drop)
        self._width = width

    def _bytes(self, k: int) -> bytes:
        rows = _yaml_rows(super()._bytes(k))
        if not self._drop:
            return _yaml_text(rows)
        block = self._blocks[k]
        fields = _yaml_fields(rows, self._width, block.n_atoms)
        keep = [i for i in range(self._width) if i not in self._drop]
        for row in fields:
            if any(row[i] for i in self._drop):
                raise ValueError("a column that is empty in frame 0 holds values "
                                 "in this frame")
        return b"\n".join(b" ".join(row[i] for i in keep) for row in fields)


def _yaml_count(data: bytes) -> tuple[int, str, bool]:
    """For the bytes after a 'data:' line: the data rows before the frame
    ends, how they end ('footer' at a '...' line, 'header' at a '---' line,
    'end' at the end of the bytes), and whether a '---' line comes after
    them (in the last frame's bytes, a header the file ends inside)."""
    lines = data.splitlines()
    rows = 0
    for i, line in enumerate(lines):
        text = line.strip()
        if text in (b"...", b"---"):
            how = "footer" if text == b"..." else "header"
            later = any(x.strip() == b"---" for x in lines[i:])
            return rows, how, later
        if text:
            rows += 1
    return rows, "end", False


def _outside_box(values: np.ndarray, kind: str, box: np.ndarray,
                 origin: np.ndarray) -> tuple[int, float, float]:
    """(atoms beyond a face of the box, the largest distance beyond one in
    Å, the same as a fraction of the spacing of that face pair) for positions
    x (Cartesian) or xs (fractional), counting only distances above half a
    unit in the sixth significant digit (LAMMPS's default %g)."""
    if kind == "x":
        frac = np.linalg.solve(box.T, (values - origin).T).T
        tol_ang = 5e-6 * float(np.abs(values).max())
    else:
        frac = values
        tol_ang = 5e-6 * float(np.abs(values).max()) \
            * float(np.linalg.norm(box, axis=1).max())
    over = np.maximum(np.maximum(-frac, frac - 1.0), 0.0)
    spacing = 1.0 / np.linalg.norm(np.linalg.inv(box), axis=0)
    beyond = over * spacing
    outside = (beyond > tol_ang).any(axis=1)
    if not outside.any():
        return 0, 0.0, 0.0
    worst = np.unravel_index(int(np.argmax(beyond)), beyond.shape)
    return int(outside.sum()), float(beyond[worst]), float(over[worst])


def read_lammps_yaml_dump(path, *, type_map: Mapping | None = None,
                          masses_from=None, units: str | None = None,
                          mass_tol_amu: float | None = None,
                          columns=None) -> Trajectory:
    """A LAMMPS YAML dump (``dump yaml``), plain or gzip.

    Each frame's header gives the timestep, atom count, boundary, box (a text
    dump's bounds, plus the tilt factors), units and time when written, and
    the column keywords; the rows are then read exactly as
    :func:`.md_readers.read_lammps_dump` reads a text dump's (columns, type
    maps, ``masses_from``, units, image flags, velocities, charges, frames
    skipped with their reason). ``columns`` replaces the keywords name for
    name (a dump written with ``dump_modify colname``). The header's thermo
    data is not read. A frame whose rows the line count does not match is
    read to count them, and a header the file ends inside is listed in
    ``skipped``.
    """
    path = _mr._existing(path)
    name = path.name
    tol = _named(name, _mr._checked_tol, _mr.MASS_TOL_AMU if mass_tol_amu is None
                 else mass_tol_amu)
    user_types, user_labels = _named(name, _mr._normalise_type_map, type_map)
    user_units = _named(name, _mr._checked_units, units, "the units argument")
    given_columns = _columns_option(columns, name)
    _refuse_empty(path, name, "YAML dump frame")
    scan = _mr._scan(path, b"data:", line_start=True, before_lines=48)
    if not scan.marks:
        raise UnsupportedFormat(f"{name}: no 'data:' line, so not a LAMMPS YAML "
                                f"dump; {_what_facet_reads()}")
    headers = [_yaml_header(mark.before) for mark in scan.marks]
    blocks = []
    other: list[str] = []
    creators = set()
    unended = False
    thermo = False
    cut_header = False
    stated_keywords = None
    for index, (mark, header) in enumerate(zip(scan.marks, headers)):
        if header is None:
            block = _mr._DumpBlock(position=index)
            block.reason = "no '---' line opens this frame's header"
            blocks.append(block)
            continue
        block, reason = _yaml_block(header, index, given_columns)
        block.reason = reason
        if stated_keywords is None and "keywords" in header.values:
            try:
                stated_keywords = tuple(_flow_list(header.values["keywords"],
                                                   "keywords"))
            except ValueError:
                pass
        creators.add(str(header.values.get("creator", "")))
        thermo = thermo or "thermo" in header.values
        for key in header.values:
            if key not in ("creator", "timestep", "units", "time", "natoms",
                           "boundary", "box", "keywords", "thermo") \
                    and key not in other:
                other.append(key)
        block.rows_start = mark.end
        block.rows_end = _mr._range_end(scan.marks, index, scan)
        is_last = index + 1 == len(scan.marks)
        if block.reason is None:
            if not is_last:
                nxt = headers[index + 1]
                between = scan.marks[index + 1].line - mark.line - 1
                rows = between - (nxt.lines if nxt else 0) \
                    - (1 if nxt and nxt.footer_before else 0)
                how = "footer"
            else:
                rows = _mr._rows_between(scan.marks, index, scan)
                footer = _yaml_line(scan.last_line) == "..."
                if footer:
                    rows -= 1
                how = "footer" if footer else "end"
            if rows != block.n_atoms:
                # the line count disagrees (a header without '---', a header
                # the file ends inside, damage): count the rows themselves
                rows, how, later = _yaml_count(_mr._read_range(
                    path, block.rows_start, block.rows_end))
                cut_header = cut_header or (is_last and later)
            if rows < block.n_atoms:
                block.reason = (f"truncated: the file ends after {rows} of "
                                f"{block.n_atoms} data rows" if how == "end"
                                else f"{rows} data rows where natoms gives "
                                f"{block.n_atoms}")
            elif rows > block.n_atoms:
                block.reason = (f"{rows} data rows where natoms gives "
                                f"{block.n_atoms}")
            elif is_last and how == "end":
                if not scan.last_line_ended:
                    block.reason = ("truncated: the file does not end with a "
                                    "line ending, so its last value may be cut")
                else:
                    unended = True
        blocks.append(block)
    if cut_header:
        block = _mr._DumpBlock(position=len(scan.marks))
        block.reason = ("truncated: the file ends inside this frame's header, "
                        "before its 'data:' line")
        blocks.append(block)
    readable = [b for b in blocks if b.reason is None]
    if readable:
        counts = [b.n_atoms for b in readable]
        n0 = _mr._majority_count(counts)
        for block in readable:
            if block.n_atoms != n0:
                block.reason = _mr._count_reason(block.n_atoms, n0, counts)
    readable = [b for b in blocks if b.reason is None]
    skipped = {b.position: b.reason for b in blocks if b.reason is not None}
    if not readable:
        columns_named = any(r.startswith("the columns ") for r in skipped.values())
        raise ValueError(f"{name} holds no readable frame: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items()))
            + (f"; {_COLNAME_ADVICE}" if columns_named else ""))
    first = readable[0]
    differing = [b.position for b in readable if b.columns != first.columns]
    if differing:
        raise ValueError(f"{name}: frames at positions {differing[:5]} name other "
                         f"keywords than frame 0 ({' '.join(first.columns)})")

    file_units = sorted({b.units for b in blocks if b.units})
    if len(file_units) > 1:
        raise ValueError(f"{name}: units reads {file_units} in different frames")
    stated = _mr._checked_units(file_units[0], f"{name}'s units key") \
        if file_units else None
    style, units_where = _mr._combine_units(stated, user_units,
                                            f"{name}'s units key")
    time_factor = _mr._TIME_TO_PS[style] if style else 1.0
    velocity_factor = _mr._VELOCITY_TO_ANG_PER_PS[style] if style else 1.0
    times = None
    if any(b.time is not None for b in readable):
        times = [np.nan if b.time is None else b.time * time_factor
                 for b in readable]

    notes: list[str] = []
    if given_columns is not None:
        notes.append(f"columns= replaces the keywords the file states ("
                     f"{' '.join(stated_keywords or ())}) with "
                     f"{' '.join(given_columns)}, in the same order")
    rows0 = _yaml_rows(_mr._read_range(path, first.rows_start, first.rows_end))
    width = len(first.columns)
    drop: list[int] = []
    text0 = _yaml_text(rows0)
    if len(text0.split()) != first.n_atoms * width:
        fields = _mr._frame0(name, first.position,
                             lambda: _yaml_fields(rows0, width, first.n_atoms))
        empty = [i for i in range(width) if not any(row[i] for row in fields)]
        partly = [first.columns[i] for i in range(width)
                  if i not in empty and not all(row[i] for row in fields)]
        if partly:
            raise ValueError(f"{name}: the {', '.join(partly)} column(s) are "
                             "empty in some rows of frame 0 and not in others")
        drop = empty
        dropped = [first.columns[i] for i in drop]
        notes.append(
            f"the {' '.join(dropped)} column(s) are empty in every row (LAMMPS's "
            "dump yaml prints no value for a typelabel column) and are not read")
        kept = tuple(c for i, c in enumerate(first.columns) if i not in drop)
        for block in readable:
            block.columns = kept
        text0 = b"\n".join(b" ".join(row[i] for i in range(width) if i not in drop)
                           for row in fields)
        if not any(c in kept for c in ("type", "element", "typelabel")):
            raise ValueError(
                f"{name}: with the empty {' '.join(dropped)} column(s) set aside "
                "the rows hold no type or element column, so the atoms cannot "
                "be given elements; write the dump with the type column (id "
                "type x y z) and pass type_map= or masses_from=<the LAMMPS data "
                "file the run read>, or with dump_modify element and the element "
                "column")
    table0 = _mr._frame0(name, first.position,
                         lambda: _mr._dump_table(text0, first))
    identity, listed, type_source, type_notes = _mr._frame0(
        name, first.position, lambda: _mr._dump_identity(
            table0, first, user_types, user_labels, masses_from, tol, name))
    notes.extend(type_notes)
    cols = first.columns
    used, unread = _mr._dump_columns(first, identity)
    notes.append(f"frame positions from {' '.join(first.wrapped[1])}"
                 + (f"; unwrapped positions from {' '.join(first.unwrapped[1])}"
                    if first.unwrapped else
                    "; unwrapped positions from the image flags ix iy iz"
                    if all(c in cols for c in _mr._IMAGE_COLUMNS) else
                    "; no unwrapped positions (neither xu/xsu nor ix iy iz)"))
    notes.append("columns read: " + " ".join(used)
                 + (("; not read: " + ", ".join(unread)) if unread else ""))
    if "id" not in cols:
        notes.append(
            "no id column: in every frame the rows are put in element order and "
            "numbered 0 .. N-1; LAMMPS re-orders atoms between frames, so atom "
            "k is not the same atom in two frames (ids_track_atoms is False)")
    notes.extend(_boundary_note([b.periodic for b in readable],
                                [b.boundary for b in readable], "the YAML header"))
    kind, position_names = first.wrapped
    if kind in ("x", "xs"):
        values = _mr._frame0(name, first.position, lambda: _mr._columns(
            table0, first, position_names, "float"))
        n_out, beyond_ang, beyond_frac = _outside_box(
            np.asarray(values, dtype=np.float64), kind, first.box, first.origin)
        if n_out:
            notes.append(
                f"{n_out} of {first.n_atoms} atoms of frame 0 lie outside the "
                f"box the header states, the farthest {beyond_ang:.3g} Å "
                f"({beyond_frac:.3g} of the spacing of that pair of faces) "
                "beyond a face; each frame is read with its atoms wrapped into "
                "the box. LAMMPS wraps atoms into the box when it re-neighbours, "
                "so a run's dump holds atoms up to about half the neighbour skin "
                "beyond a face")
            notes.append(
                "LAMMPS's dump yaml writes the restricted box (bounds and tilt "
                "factors) even with dump_modify triclinic/general yes, which "
                "the dump_modify documentation lists for the atom and custom "
                "styles only [16]; the rows of such a file hold general-frame "
                "positions that do not fit the stated box, so the distances "
                "between atoms read from it differ from the run's. A dump "
                "custom written with triclinic/general yes states the general "
                "box (ITEM: BOX BOUNDS abc origin) and is read with it")
    if creators - {"LAMMPS"}:
        notes.append(f"creator {sorted(creators - {'LAMMPS'})}: the layout is "
                     "read as LAMMPS writes it")
    if thermo:
        notes.append("the thermo data in the frame headers (dump_modify thermo "
                     "yes) is not read")
    if other:
        notes.append("header keys not read: " + "; ".join(other))
    if unended:
        notes.append("no '...' line ends the last frame (LAMMPS writes one after "
                     "every frame); its rows are complete and are read")
    if scan.gzip_cut:
        notes.append(_mr._GZIP_CUT_NOTE)
    boundaries = sorted({b.boundary for b in readable})
    box_varies = any(not np.array_equal(b.box, first.box)
                     or not np.array_equal(b.origin, first.origin)
                     for b in readable[1:])
    trajectory = _YamlTrajectory(
        path, readable, identity=identity, velocity_factor=velocity_factor,
        drop=drop, width=width,
        periodic=None if first.periodic is None or len(boundaries) != 1
        else first.periodic,
        file_format=YAML_FORMAT, n_atoms=first.n_atoms, n_frames=len(readable),
        type_map_source=type_source, timesteps=[b.timestep for b in readable],
        times_ps=times, type_map=listed, skipped=skipped, notes=notes,
        units_note=_mr._units_note(style, units_where, has_time=times is not None,
                                   has_velocity=all(c in cols for c in
                                                    _mr._VELOCITY_COLUMNS)),
        box_varies=box_varies)
    frame0 = _mr._frame0(name, first.position, lambda: _mr._dump_frame(
        text0, first, identity, velocity_factor,
        None if times is None or not np.isfinite(times[0]) else float(times[0])))
    if "id" not in cols:
        trajectory.ids_track_atoms = False
        trajectory._composition = frame0.composition
    if frame0.charge_e is not None:
        charges, charge_notes = md_model.charges_per_element(frame0.elements,
                                                             frame0.charge_e)
        trajectory.charges_e = charges
        trajectory.notes.extend(charge_notes)
    return _mr._finish(trajectory, frame0)


# ---------------------------------------------------------------------------
# AtomEye CFG
# ---------------------------------------------------------------------------

_CFG_KEY = re.compile(r"^\s*(?P<key>[A-Za-z_][\w ]*?(?:\(\s*\d\s*,\s*\d\s*\)|"
                      r"\[\s*\d+\s*\])?)\s*=\s*(?P<value>\S+)\s*(?P<rest>.*)$")
_CFG_FIRST = re.compile(r"^\s*Number of particles\s*=\s*\d+\s*$")
_CFG_MATRIX = re.compile(r"(H0|Transform|eta)\(([1-3]),([1-3])\)")


def sniff_cfg(head: bytes, path: Path) -> bool:
    """An AtomEye CFG: its first line (after comments) is 'Number of
    particles = N', which the format requires first."""
    try:
        text = _mr._strip_bom(head).decode("utf-8", "replace")
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            return _CFG_FIRST.match(line) is not None
        return False
    except Exception:       # noqa: BLE001 - a sniffer never raises
        return False


@dataclass
class _CfgHeader:
    n_atoms: int
    scale: float
    h0: np.ndarray
    transform: np.ndarray | None
    eta: np.ndarray | None
    rate: str | None
    velocities: bool
    entry_count: int | None          # None: the standard CFG
    auxiliary: list[str]
    body_line: int                   # 0-based index of the first body line
    unread: list[str]


def _cfg_header(lines: Sequence[str], name: str) -> _CfgHeader:
    n_atoms = None
    scale = 1.0
    h0 = np.full((3, 3), np.nan)
    transform = np.full((3, 3), np.nan)
    eta = {}
    rate = None
    no_velocity = False
    entry = None
    aux: dict[int, str] = {}
    unread: list[str] = []
    body = None
    for number, line in enumerate(lines):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text == ".NO_VELOCITY.":
            no_velocity = True
            continue
        match = _CFG_KEY.match(text)
        if match is None:
            body = number
            break
        key = " ".join(match.group("key").split())
        value, rest = match.group("value"), match.group("rest").strip()
        where = f"{name}, line {number + 1}"
        try:
            if key == "Number of particles":
                n_atoms = int(value)
            elif key == "A":
                scale = float(value)
                unit = rest.split()[0] if rest.split() else "Angstrom"
                if unit.lower() not in ("angstrom", "a", "å"):
                    raise ValueError(f"{where}: A is given in {unit!r}; FACET "
                                     "reads A in Angstrom")
            elif _CFG_MATRIX.fullmatch(key.replace(" ", "")):
                matrix, i, j = _CFG_MATRIX.fullmatch(key.replace(" ", "")).groups()
                i, j = int(i) - 1, int(j) - 1
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"{key} is not finite")
                if matrix == "H0":
                    h0[i, j] = number
                elif matrix == "Transform":
                    transform[i, j] = number
                else:
                    eta[(min(i, j), max(i, j))] = number
            elif key == "R":
                rate = f"{value} {rest}".strip()
            elif key == "entry_count":
                entry = int(value)
            elif key.startswith("auxiliary"):
                indexed = re.fullmatch(r"auxiliary\s*\[\s*(\d+)\s*\]", key)
                if indexed is None:
                    raise ValueError("an auxiliary column needs its index, "
                                     "auxiliary[k] = name")
                aux[int(indexed.group(1))] = value
            else:
                unread.append(text)
        except ValueError as error:
            if str(error).startswith(name):
                raise
            raise ValueError(f"{where}: {text!r} does not parse ({error})") \
                from None
    if n_atoms is None:
        raise ValueError(f"{name}: no 'Number of particles' line")
    if n_atoms < 1:
        raise ValueError(f"{name}: Number of particles is {n_atoms}")
    if np.isnan(h0).any():
        missing = [f"H0({i + 1},{j + 1})" for i in range(3) for j in range(3)
                   if np.isnan(h0[i, j])]
        raise ValueError(f"{name}: the header has no {', '.join(missing)}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{name}: A is {scale}; a positive length scale is "
                         "needed")
    if body is None:
        body = len(lines)
    tr = None
    if not np.isnan(transform).all():
        if np.isnan(transform).any():
            raise ValueError(f"{name}: some Transform(i,j) lines are missing")
        tr = transform if not np.array_equal(transform, np.eye(3)) else None
    et = None
    if eta:
        matrix = np.zeros((3, 3))
        for (i, j), v in eta.items():
            matrix[i, j] = matrix[j, i] = v
        et = matrix if np.any(matrix) else None
    if entry is not None:
        n_aux = entry - 3 - (0 if no_velocity else 3)
        if n_aux < 0:
            raise ValueError(f"{name}: entry_count {entry} is below the 3 "
                             "coordinates" + ("" if no_velocity else
                                              " and 3 velocities"))
        # entry_count is the width of every atom row: compare it with the
        # first one before building a name per column, so a large number in
        # a damaged header costs nothing
        width, row_line = _cfg_first_row(lines, body)
        if width is not None and width != entry:
            raise ValueError(f"{name}, line {row_line + 1}: the first atom row "
                             f"holds {width} values where entry_count gives "
                             f"{entry}")
        if entry > _MAX_COLUMNS:
            raise ValueError(f"{name}: entry_count {entry} is above "
                             f"{_MAX_COLUMNS} values per atom row, a parsing "
                             "limit of FACET's (the binary dump's column limit)")
        names = [aux.get(k, f"auxiliary[{k}]") for k in range(n_aux)]
    else:
        names = []
    return _CfgHeader(n_atoms, scale, h0, tr, et, rate, not no_velocity, entry,
                      names, body, unread)


def _cfg_box(header: _CfgHeader, name: str) -> np.ndarray:
    """H = A H0 (Transform) (sqrt(1 + 2 eta)), the rows the edges [11]."""
    box = header.h0.copy()
    if header.transform is not None and header.eta is not None:
        raise ValueError(f"{name}: the header gives both Transform and eta; "
                         "FACET reads one of them, not both")
    if header.transform is not None:
        box = box @ header.transform
    if header.eta is not None:
        w, v = np.linalg.eigh(np.eye(3) + 2.0 * header.eta)
        if (w <= 0).any():
            raise ValueError(f"{name}: 1 + 2 eta is not positive definite")
        box = box @ (v @ np.diag(np.sqrt(w)) @ v.T)
    try:
        _finite_box(header.scale * box, "A x H0")
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from None
    return box


def _cfg_first_row(lines: Sequence[str], body: int) -> tuple[int | None, int]:
    """(values on the first atom row of an extended CFG, its line index):
    the first body line holding more than one value (mass and element lines
    hold one); None when the lines given end before one."""
    for number in range(body, len(lines)):
        tokens = lines[number].split()
        if len(tokens) > 1 and not tokens[0].startswith("#"):
            return len(tokens), number
    return None, -1


def _cfg_body(lines: Sequence[str], header: _CfgHeader, name: str):
    """(per-atom block index, blocks [(mass, label)], rows (N, width))."""
    width = header.entry_count
    blocks: list[tuple[float, str]] = []
    block_of: list[int] = []
    rows: list[list[str]] = []
    mass = label = None
    standard = width is None
    for number in range(header.body_line, len(lines)):
        tokens = lines[number].split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if standard:
            if len(tokens) < 5:
                raise ValueError(f"{name}, line {number + 1}: a standard CFG row "
                                 "needs mass, symbol and three coordinates")
            try:
                mass_value = float(tokens[0])
            except ValueError:
                raise ValueError(f"{name}, line {number + 1}: the mass "
                                 f"{tokens[0]!r} is not a number") from None
            key = (mass_value, tokens[1])
            if not blocks or blocks[-1] != key:
                blocks.append(key)
            block_of.append(len(blocks) - 1)
            rows.append(tokens[2:])
            continue
        if len(tokens) == 1:
            if _is_float(tokens[0]):
                mass = float(tokens[0])
                label = None
            else:
                if mass is None:
                    raise ValueError(f"{name}, line {number + 1}: the element "
                                     f"{tokens[0]!r} follows no mass line")
                label = tokens[0]
                blocks.append((mass, label))
            continue
        if len(tokens) != width:
            raise ValueError(f"{name}, line {number + 1}: {len(tokens)} values "
                             f"where entry_count gives {width}")
        if label is None:
            raise ValueError(f"{name}, line {number + 1}: an atom row with no "
                             "mass and element line before it")
        block_of.append(len(blocks) - 1)
        rows.append(tokens)
    if len(rows) != header.n_atoms:
        raise ValueError(f"{name}: {len(rows)} atom rows where Number of "
                         f"particles gives {header.n_atoms}" +
                         (" (the file may be cut)" if len(rows) < header.n_atoms
                          else ""))
    if standard:
        widths = {len(r) for r in rows}
        if widths != {6}:
            raise ValueError(f"{name}: standard CFG rows hold mass, symbol, three "
                             "coordinates and three velocities; rows here hold "
                             f"{sorted(w + 2 for w in widths)} values")
    try:
        table = np.array(rows, dtype=np.float64)
    except ValueError:
        raise ValueError(f"{name}: an atom row holds a value that is not a "
                         "number") from None
    if not np.isfinite(table).all():
        raise ValueError(f"{name}: an atom row holds a value that is not finite")
    return np.array(block_of, dtype=np.int64), blocks, table


def _is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _cfg_elements(blocks, user_labels: Mapping[str, str], tol: float, name: str
                  ) -> tuple[list[str], str, list[str]]:
    """The element of each (mass, label) block, the source, and notes."""
    out: list[str] = []
    used: list[str] = []
    notes: list[str] = []
    problems: list[str] = []
    weights = dict(_mr.element_weights_amu())
    for mass, label in sorted(set(blocks), key=lambda b: (b[1], b[0])):
        if label in user_labels:
            symbol = user_labels[label]
            used.append("user")
        else:
            problem = _mr._label_problem(label)
            if problem is None:
                symbol = md_model.validate_symbol(label.strip())
                conflict = _mr._mass_conflict(symbol, mass, tol,
                                              where="its mass line")
                if conflict is not None:
                    problem = f"reads as {symbol}, and {conflict}"
            if problem is not None:
                problems.append(f"{label!r} at {mass:.6g} amu: {problem}")
                continue
            used.append("file symbols")
        weight = weights.get(symbol)
        if weight is not None and abs(weight - mass) > tol + _mr._MASS_EPS_AMU:
            notes.append(f"element line {label!r} is read as {symbol} (standard "
                         f"atomic weight {weight:.6g} amu), and its mass line "
                         f"gives {mass:.6g} amu, {abs(weight - mass):.3g} amu "
                         "away")
    if problems:
        labels = sorted({b[1] for b in blocks})
        masses_per_label = {lab: sorted({m for m, l in blocks if l == lab})
                            for lab in labels}
        hint = ""
        if labels == ["C"] and len(masses_per_label["C"]) > 1:
            hint = (" Every atom is written 'C', which LAMMPS's dump cfg writes "
                    "for every type when dump_modify element does not name the "
                    "types; write the dump again with dump_modify <dump-ID> "
                    "element <one symbol per type>.")
        example = ", ".join(f"{lab!r}: '<element>'" for lab in labels[:3])
        raise ValueError(f"{name}: no element for the element line(s) "
                         + "; ".join(problems) + "." + hint
                         + f" A label that names one element can be mapped with "
                         f"type_map={{{example}}}")
    lookup = {}
    for mass, label in set(blocks):
        lookup[(mass, label)] = user_labels.get(label) or \
            md_model.validate_symbol(label.strip())
    out = [lookup[b] for b in blocks]
    unused = sorted(set(user_labels) - {b[1] for b in blocks})
    if unused:
        notes.append(f"type map entries {unused} are not used: no element line "
                     "carries those labels")
    return out, _mr._join_sources(used), notes


@dataclass
class _CfgFile:
    path: Path
    position: int
    reason: str | None = None
    header: _CfgHeader | None = None
    box: np.ndarray | None = None
    timestep: int = NO_TIMESTEP


def _cfg_read_header(path: Path) -> tuple[_CfgHeader, list[str]]:
    name = path.name
    head = _mr._head(path, _CFG_HEAD_BYTES)
    lines = _mr._lines(head)
    if len(head) >= _CFG_HEAD_BYTES - 3:
        lines = lines[:-1]
    header = _cfg_header(lines, name)
    if header.body_line >= len(lines):
        lines = _mr._lines(_mr._strip_bom(_mr._read_all(path)))
        header = _cfg_header(lines, name)
    return header, lines


def timesteps_from_names(paths: Sequence) -> list[int] | None:
    """The number that differs between the names of a file series, or None.

    LAMMPS replaces the ``*`` of ``dump.*.cfg`` with the timestep. When the
    names split into the same text and numbers, and exactly one number differs
    between them, that number is returned for each file; otherwise None (one
    file, two numbers that differ, names of different shapes).
    """
    names = [Path(str(p)).name for p in paths]
    if len(names) < 2:
        return None
    parts = [re.split(r"(\d+)", n) for n in names]
    if len({len(p) for p in parts}) != 1:
        return None
    shape = parts[0]
    for p in parts[1:]:
        if any(p[i] != shape[i] for i in range(0, len(p), 2)):
            return None
    varying = [i for i in range(1, len(shape), 2)
               if len({int(p[i]) for p in parts}) > 1]
    if len(varying) != 1:
        return None
    return [int(p[varying[0]]) for p in parts]


class _CfgTrajectory(_mr._Closable, Trajectory):
    def __init__(self, files: Sequence[_CfgFile], *, user_labels, tol: float,
                 lammps_unwrapped: bool, element_order: bool,
                 velocity_factor: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._files = list(files)
        self._user_labels = dict(user_labels)
        self._tol = tol
        self._lammps_unwrapped = lammps_unwrapped
        self._element_order = element_order
        self._velocity_factor = velocity_factor
        self._composition: dict[str, int] | None = None
        self.periodic = None

    def _load(self, k: int) -> Frame:
        time = None
        if self.times_ps is not None and np.isfinite(self.times_ps[k]):
            time = float(self.times_ps[k])
        frame, _ = _cfg_frame(self._files[k], self._user_labels, self._tol,
                              self._lammps_unwrapped, self._element_order, time,
                              self._velocity_factor)
        if self._composition is not None and frame.composition != self._composition:
            raise ValueError(
                f"{self._files[k].path.name} holds {frame.composition}, the first "
                f"file {self._composition}; without an id column the atoms are "
                "matched between files by element only")
        return frame


def _cfg_lammps_unwrapped(header: _CfgHeader, blocks, table) -> bool:
    """A LAMMPS ``xsu ysu zsu`` file: A = 10 and one atom per mass/element
    block, as dump_cfg.cpp writes it [12]."""
    return header.scale == _CFG_UNWRAP_EXPAND and len(blocks) == header.n_atoms


def _cfg_parse(path: Path):
    """(header, per-atom block index, blocks, rows) of one CFG file, or
    ValueError naming it."""
    name = path.name
    lines = _mr._lines(_mr._strip_bom(_mr._read_all(path)))
    header = _cfg_header(lines, name)
    block_of, blocks, table = _cfg_body(lines, header, name)
    return header, block_of, blocks, table


def _cfg_rows_problem(path: Path, header: _CfgHeader,
                      known: dict | None = None) -> str | None:
    """Why a CFG file's atom rows are not complete, found without parsing
    them, or None.

    A file is cut at its end when copied while the run wrote it, so what is
    checked is the end: the lines after the header, the last line's values
    and the line ending after it. ``known`` carries, between the files of a
    series, the line count after the header of a file whose rows were
    counted in full; a file with that count and a complete last row is taken
    as complete (one read and one count of line endings). Any other file has
    its rows counted with numpy over its bytes -- lines holding more than one
    value are atom rows (mass and element lines hold one) -- and a file that
    count questions is parsed in full, which gives the reason (or reads,
    when only comment lines made the count differ). A row damaged in the
    middle of a file is found when that frame is read."""
    raw = _mr._strip_bom(path.read_bytes() if not _mr._is_gzip(path)
                         else _mr._read_all(path))
    start = 0
    for _ in range(header.body_line):
        newline = raw.find(b"\n", start)
        if newline < 0:
            start = len(raw)
            break
        start = newline + 1
    width = header.entry_count if header.entry_count is not None else 8
    tail = raw[start:].rstrip(b" \t")
    ended = tail.endswith((b"\n", b"\r"))
    stripped = tail.rstrip()
    last = stripped[stripped.rfind(b"\n") + 1:].split()
    lines = raw.count(b"\n", start)
    if known is not None and known.get("lines") == lines and ended \
            and len(last) == width:
        return None
    problem = None
    body = np.frombuffer(raw, dtype=np.uint8)[start:]
    if body.size:
        space = _BLANK_BYTE[body]
        starts = np.flatnonzero(~space[1:] & space[:-1]) + 1
        if not space[0]:
            starts = np.concatenate(([0], starts))
        line_of = np.searchsorted(np.flatnonzero(body == 10), starts)
        per_line = np.bincount(line_of)
        rows = per_line[per_line > 1]
        if rows.size != header.n_atoms:
            problem = (f"{rows.size} atom rows where Number of particles gives "
                       f"{header.n_atoms}")
        elif (rows != width).any():
            problem = f"an atom row holds another number of values than {width}"
        elif not ended:
            problem = ("the file does not end with a line ending, so its last "
                       "value may be cut")
    elif header.n_atoms:
        problem = f"0 atom rows where Number of particles gives {header.n_atoms}"
    if problem is None:
        if known is not None and "lines" not in known:
            known["lines"] = lines
        return None
    try:
        _cfg_parse(path)
    except ValueError as error:
        text = str(error)
        for lead in (f"{path.name}: ", f"{path.name}, "):
            if text.startswith(lead):
                text = text[len(lead):]
                break
        if "may be cut" in text or (not ended or len(last) != width) \
                and "values where entry_count" in text:
            return "truncated: " + text
        return text
    if problem.startswith("the file does not end"):
        return "truncated: " + problem
    return None


def _cfg_frame(entry: _CfgFile, user_labels, tol: float,
               lammps_unwrapped: bool | None, element_order: bool,
               time_ps: float | None, velocity_factor: float,
               parsed=None) -> tuple[Frame, dict]:
    """One CFG file as a frame, and what was found (for the notes).

    ``lammps_unwrapped`` None decides from this file whether it is LAMMPS's
    xsu layout (the first file of a series); True or False applies the first
    file's decision, so every frame of a series is read alike. ``parsed`` is
    :func:`_cfg_parse`'s result when the file was parsed already."""
    path, name = entry.path, entry.path.name
    header, block_of, blocks, table = parsed if parsed is not None \
        else _cfg_parse(path)
    labels, source, label_notes = _cfg_elements(
        [blocks[i] for i in block_of], user_labels, tol, name)
    elements = np.array(labels, dtype="<U2")
    h = _cfg_box(header, name)
    standard = header.entry_count is None
    aux0 = 3 + (3 if header.velocities else 0)
    aux = {n: aux0 + i for i, n in enumerate(header.auxiliary)}
    if lammps_unwrapped is None:
        is_unwrapped = _cfg_lammps_unwrapped(header, blocks, table)
    else:
        is_unwrapped = lammps_unwrapped
        if is_unwrapped and (header.scale != _CFG_UNWRAP_EXPAND
                             or len(blocks) != header.n_atoms):
            raise ValueError(f"{name}: the first file of the series is LAMMPS's "
                             "unwrapped layout (A = 10, one atom per block) and "
                             f"this one is not (A = {header.scale:g})")
    s = table[:, :3]
    unwrapped = None
    if is_unwrapped:
        box = h
        s = (s - 0.5) * _CFG_UNWRAP_EXPAND + 0.5
        unwrapped = s @ box
    else:
        box = header.scale * h
    ids = None
    if "id" in aux:
        ids = _whole(table[:, aux["id"]], "id", name)
    charge = table[:, aux["q"]].copy() if "q" in aux else None
    velocities = None
    if all(c in aux for c in _mr._VELOCITY_COLUMNS):
        velocities = table[:, [aux[c] for c in _mr._VELOCITY_COLUMNS]] \
            * velocity_factor
    if unwrapped is None and all(c in aux for c in _mr._IMAGE_COLUMNS):
        image = _whole(table[:, [aux[c] for c in _mr._IMAGE_COLUMNS]],
                       "ix iy iz", name)
        unwrapped = unwrap_with_images(s @ box, image, box)
    if ids is None and element_order:
        order = np.argsort(elements, kind="stable")
        elements, s = elements[order], s[order]
        unwrapped = None if unwrapped is None else unwrapped[order]
        charge = None if charge is None else charge[order]
        velocities = None if velocities is None else velocities[order]
    step = None if entry.timestep == NO_TIMESTEP else entry.timestep
    frame = frame_from_arrays(elements, None, frac=s, box_ang=box, atom_id=ids,
                              timestep=step, time_ps=time_ps,
                              unwrapped_cart_ang=unwrapped,
                              vel_ang_per_ps=velocities, charge_e=charge)
    found = {"header": header, "source": source, "label_notes": label_notes,
             "unwrapped": is_unwrapped, "standard": standard,
             "listed": {b[1]: lab for b, lab in zip([blocks[i] for i in block_of],
                                                    labels) if b[1] != lab},
             "aux": list(header.auxiliary), "per_atom_blocks":
             len(blocks) == header.n_atoms}
    return frame, found


def _name_pattern(path: Path) -> str:
    """A file name with each run of digits written '*' (dump.100.cfg ->
    dump.*.cfg), the pattern LAMMPS's dump writes a series under."""
    return re.sub(r"\d+", "*", path.name)


def read_cfg_series(paths, *, type_map: Mapping | None = None,
                    mass_tol_amu: float | None = None,
                    timestep_fs: float | None = None,
                    units: str | None = None) -> Trajectory:
    """AtomEye CFG files, one frame each, in the order given.

    LAMMPS's ``dump cfg`` writes one file per snapshot (``dump.*.cfg``); ASE
    writes one file. Elements come from each block's element line, checked
    against its mass line (``type_map`` maps a label to an element); the
    ``id``, ``q``, ``ix iy iz`` and ``vx vy vz`` auxiliary columns are read
    (LAMMPS writes the last in the run's velocity unit, which ``units``
    names). Timesteps come from the file names when exactly one number
    differs between them (:func:`timesteps_from_names`), and ``timestep_fs``
    turns them into times. Every file is checked when the series is opened: a
    file whose header does not parse, whose atom rows are cut or short, whose
    atom count differs from most files' or whose auxiliary columns differ
    from most files' is listed in ``skipped``, and frame 0 is the first file
    that reads. Files whose names follow more than one pattern, each held by
    several files (two series in one directory), are refused. Without an id
    column, each file's rows are put in element order and
    ``ids_track_atoms`` is False when there is more than one file.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = [_mr._existing(p) for p in paths]
    if not files:
        raise ValueError("read_cfg_series needs at least one CFG file")
    name = files[0].name if len(files) == 1 else \
        f"the CFG series {files[0].name} .. {files[-1].name}"
    tol = _named(name, _mr._checked_tol, _mr.MASS_TOL_AMU if mass_tol_amu is None
                 else mass_tol_amu)
    user_types, user_labels = _named(name, _mr._normalise_type_map, type_map)
    user_units = _named(name, _mr._checked_units, units, "the units argument")
    step_fs = None if timestep_fs is None else _named(
        name, _checked_positive, timestep_fs, "timestep_fs")
    if len(files) == 1:
        _refuse_empty(files[0], files[0].name, "CFG frame")
    patterns: dict[str, list[str]] = {}
    for path in files:
        patterns.setdefault(_name_pattern(path), []).append(path.name)
    series = {p: n for p, n in patterns.items() if len(n) > 1}
    if len(series) > 1:
        raise ValueError(
            f"{name}: the file names follow {len(patterns)} patterns ("
            + "; ".join(f"{p}: {len(n)} files" for p, n in patterns.items())
            + "), as the files of more than one series do; a trajectory is "
            "read from one series, so pass a wildcard pattern (for example "
            f"'{files[0].parent / next(iter(series))}') or a list of the files "
            "of one series")
    steps = timesteps_from_names(files)
    if step_fs is not None and steps is None:
        raise ValueError(
            f"{name}: timestep_fs needs timesteps, and a CFG file holds none; "
            "the names of a series give them when exactly one number differs "
            "between them (LAMMPS's dump.*.cfg)" + (
                "" if len(files) > 1 else ", and a single file has no series"))
    entries = []
    known: dict = {}         # a complete file's line count, for the next ones
    for position, path in enumerate(files):
        entry = _CfgFile(path, position,
                         timestep=steps[position] if steps else NO_TIMESTEP)
        try:
            entry.header, _ = _cfg_read_header(path)
            entry.box = _cfg_box(entry.header, path.name)
        except (ValueError, UnsupportedFormat) as error:
            text = str(error)
            if text.startswith(f"{path.name}: "):
                text = text[len(path.name) + 2:]
            elif text.startswith(f"{path.name}, "):
                text = text[len(path.name) + 2:]
            entry.reason = f"{path.name}: unreadable header ({text})"
        else:
            if len(files) > 1:
                problem = _cfg_rows_problem(path, entry.header, known)
                if problem is not None:
                    entry.reason = (problem if problem.startswith(path.name)
                                    else f"{path.name}: {problem}")
        entries.append(entry)
    readable = [e for e in entries if e.reason is None]
    if readable:
        counts = [e.header.n_atoms for e in readable]
        n0 = _mr._majority_count(counts)
        for entry in readable:
            if entry.header.n_atoms != n0:
                entry.reason = (f"{entry.path.name}: "
                                + _mr._count_reason(entry.header.n_atoms, n0,
                                                    counts))
    readable = [e for e in entries if e.reason is None]
    if readable:
        layouts = [(e.header.velocities, tuple(e.header.auxiliary))
                   for e in readable]
        common = _mr._majority_count(layouts)
        held = sum(1 for x in layouts if x == common)
        def shown(layout) -> str:
            return (f"auxiliary {' '.join(layout[1]) or 'none'}, "
                    + ("velocities" if layout[0] else "no velocities"))

        for entry, layout in zip(readable, layouts):
            if layout != common:
                entry.reason = (f"{entry.path.name}: its columns ({shown(layout)}) "
                                f"differ from those of {held} of the "
                                f"{len(layouts)} files ({shown(common)})")
    # frame 0: the first file whose atom rows parse
    parsed = None
    for entry in entries:
        if entry.reason is not None:
            continue
        try:
            parsed = _cfg_parse(entry.path)
        except ValueError as error:
            text = str(error)
            entry.reason = text if text.startswith(entry.path.name) \
                else f"{entry.path.name}: {text}"
            continue
        break
    readable = [e for e in entries if e.reason is None]
    skipped = {e.position: e.reason for e in entries if e.reason is not None}
    if not readable:
        if len(files) == 1:
            raise ValueError(skipped[0])
        raise ValueError(f"{name} holds no readable file: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items())))
    style, units_where = _mr._combine_units(None, user_units, "")
    velocity_factor = _mr._VELOCITY_TO_ANG_PER_PS[style] if style else 1.0
    first = readable[0]
    frame0, found = _mr._frame0(first.path.name, first.position, lambda: _cfg_frame(
        first, user_labels, tol, None, len(files) > 1, None, velocity_factor,
        parsed))
    header = found["header"]
    has_ids = "id" in header.auxiliary
    times = None
    if step_fs is not None:
        times = [e.timestep * step_fs * 1e-3 for e in readable]
    notes: list[str] = list(found["label_notes"])
    if len(files) > 1:
        notes.append(f"{len(files)} CFG files read as frames in the order given: "
                     + ", ".join(f.name for f in files[:3])
                     + (f", ..., {files[-1].name}" if len(files) > 3 else ""))
    notes.append(
        ("standard CFG (no entry_count)" if found["standard"] else
         f"extended CFG, entry_count {header.entry_count}")
        + (f", auxiliary columns {' '.join(header.auxiliary)}"
           if header.auxiliary else ", no auxiliary columns")
        + ("; one mass and element line per atom, as LAMMPS's dump cfg writes"
           if found["per_atom_blocks"] else ""))
    aux = header.auxiliary
    read_aux = [c for c in ("id", "q") if c in aux]
    if all(c in aux for c in _mr._IMAGE_COLUMNS) and not found["unwrapped"]:
        read_aux += list(_mr._IMAGE_COLUMNS)
    has_velocity = all(c in aux for c in _mr._VELOCITY_COLUMNS)
    if has_velocity:
        read_aux += list(_mr._VELOCITY_COLUMNS)
    unread_aux = [c for c in aux if c not in read_aux]
    notes.append("auxiliary columns read: " + (" ".join(read_aux) or "none")
                 + ("; not read: " + " ".join(unread_aux) if unread_aux else ""))
    if header.velocities:
        notes.append(
            "the velocity columns are not read: the CFG specification gives "
            "ds/dt in the rate scale R (ns^-1 by default), and ASE writes "
            "Cartesian velocities in its own units in the same columns, so their "
            "unit depends on the writer")
    if found["unwrapped"]:
        notes.append(
            "A = 10 with one mass and element line per atom: read as LAMMPS's "
            "dump cfg with xsu ysu zsu, which writes s' = (s - 0.5) / 10 + 0.5 "
            "and A = 10 (dump_cfg.cpp, UNWRAPEXPAND) so that AtomEye shows "
            "molecules whole; s = (s' - 0.5) x 10 + 0.5 in the box H0 (not the "
            "box 10 x H0 AtomEye draws), kept as unwrapped_cart_ang")
    elif header.scale != 1.0:
        notes.append(f"A = {header.scale:g}: the box is A x H0, as AtomEye reads "
                     "it")
    if header.transform is not None:
        notes.append("the box is H0 x Transform")
    if header.eta is not None:
        notes.append("the box is H0 x sqrt(1 + 2 eta)")
    if header.unread:
        notes.append("header lines not read: " + "; ".join(header.unread))
    notes.append("a CFG holds no box origin; positions are s @ H, with the box "
                 "corner at (0, 0, 0)")
    notes.append("a CFG does not record the boundary; the box is taken as "
                 "periodic along a, b and c")
    if steps is not None:
        notes.append("timesteps from the file names (the one number that "
                     "differs between them; LAMMPS replaces * in dump.*.cfg "
                     "with the timestep)")
    elif len(files) > 1:
        notes.append("the file names do not give timesteps (no single number "
                     "differs between them), so the frames have none")
    if not has_ids and len(files) > 1:
        notes.append(
            "no id column: in every file the rows are put in element order and "
            "numbered 0 .. N-1; LAMMPS writes atoms in processor order, so atom "
            "k is not the same atom in two files (ids_track_atoms is False)")
    listed = dict(found["listed"])
    notes.extend(_mr._numeric_entries_note(user_types, "element lines"))
    boxes = [e.box * (1.0 if found["unwrapped"] else e.header.scale)
             for e in readable]
    box_varies = any(not np.array_equal(b, boxes[0]) for b in boxes[1:])
    units_note = "lengths in Å (the CFG's A is in Angstrom)"
    if has_velocity:
        if style is None:
            units_note += (
                "; velocities from the auxiliary vx vy vz columns are stored as "
                "written, which is Å/ps under LAMMPS units metal; under units "
                "real LAMMPS writes Å/fs, so a velocity read here would be "
                "1/1000 of its value in Å/ps (a CFG states no unit style); "
                "units='real' or units='metal' states the style")
        else:
            units_note += (f"; velocities from the auxiliary vx vy vz columns "
                           f"in LAMMPS units {style} ({units_where})"
                           + (", converted from Å/fs to Å/ps"
                              if style == "real" else ", Å/ps"))
    elif style is not None:
        units_note += (f"; units {style} ({units_where}) is not used: the files "
                       "hold no velocity columns")
    units_note += (f"; times are timestep x timestep_fs {step_fs:g} fs"
                   if step_fs is not None else "; no times")
    source_path = files[0] if len(files) == 1 else files[0].parent
    trajectory = _CfgTrajectory(
        readable, user_labels=user_labels, tol=tol,
        lammps_unwrapped=found["unwrapped"], element_order=len(files) > 1,
        velocity_factor=velocity_factor,
        source_path=source_path, file_format=CFG_FORMAT, n_atoms=frame0.n_atoms,
        n_frames=len(readable), type_map_source=found["source"],
        timesteps=[e.timestep for e in readable], times_ps=times,
        type_map=listed, skipped=skipped, notes=notes, units_note=units_note,
        box_varies=box_varies)
    if not has_ids and len(files) > 1:
        trajectory.ids_track_atoms = False
        trajectory._composition = frame0.composition
    if frame0.charge_e is not None:
        charges, charge_notes = md_model.charges_per_element(frame0.elements,
                                                             frame0.charge_e)
        trajectory.charges_e = charges
        trajectory.notes.extend(charge_notes)
    if times is not None:
        frame0 = dataclasses.replace(frame0, time_ps=times[0])
    return _mr._finish(trajectory, frame0)


def read_cfg(path, *, type_map: Mapping | None = None,
             mass_tol_amu: float | None = None,
             timestep_fs: float | None = None,
             units: str | None = None) -> Trajectory:
    """One AtomEye CFG file as a one-frame trajectory
    (:func:`read_cfg_series` of one file)."""
    return read_cfg_series([path], type_map=type_map, mass_tol_amu=mass_tol_amu,
                           timestep_fs=timestep_fs, units=units)


# ---------------------------------------------------------------------------
# the formats this module adds
# ---------------------------------------------------------------------------

FORMATS = (
    FormatSpec(
        name=DCD_FORMAT,
        description="DCD trajectory (LAMMPS dump dcd, as CHARMM, NAMD and VMD "
                    "write it; elements from a topology file)",
        extensions=(".dcd",), stems=(), sniff=sniff_dcd, read=read_dcd,
        options=frozenset({"topology", "type_map", "atom_style", "masses_from",
                           "units", "mass_tol_amu", "timestep_fs", "box_from"}),
        binary=True),
    FormatSpec(
        name=BINARY_FORMAT,
        description="LAMMPS binary dump (dump atom / custom to *.bin)",
        extensions=_BINARY_SUFFIXES, stems=(), sniff=sniff_lammps_binary,
        read=read_lammps_binary_dump,
        options=frozenset({"type_map", "masses_from", "units", "mass_tol_amu",
                           "columns"}),
        binary=True),
    FormatSpec(
        name=YAML_FORMAT,
        description="LAMMPS YAML dump (dump yaml)",
        extensions=(".yaml", ".yml"), stems=(), sniff=sniff_lammps_yaml,
        read=read_lammps_yaml_dump,
        options=frozenset({"type_map", "masses_from", "units", "mass_tol_amu",
                           "columns"})),
    FormatSpec(
        name=CFG_FORMAT,
        description="AtomEye extended or standard CFG (LAMMPS dump cfg, one file "
                    "per snapshot; ASE)",
        extensions=(".cfg",), stems=(), sniff=sniff_cfg, read=read_cfg,
        options=frozenset({"type_map", "mass_tol_amu", "timestep_fs", "units"}),
        read_series=read_cfg_series),
)
