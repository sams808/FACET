"""GROMACS XTC trajectories (also LAMMPS ``dump xtc``), read with numpy.

WHY FACET READS XTC
-------------------
XTC is GROMACS's compressed trajectory, and LAMMPS writes it too (``dump
xtc``), so a glass run analysed in VMD, MDAnalysis or GROMACS tools often
leaves one behind. Before this module FACET refused the file with the generic
"not recognised" message (defect D8 of the readers' recognition test). The
reference decoder is C code (libxdrfile, carried by MDAnalysis and mdtraj);
FACET depends only on numpy, scipy, gemmi and spglib, so the format is decoded
here from its published description, in numpy and plain Python.

WHAT A FRAME HOLDS
------------------
All numbers are XDR [1]: big-endian 32-bit two's-complement integers, IEEE
single-precision floats, and counted opaque data padded with zero bytes to a
multiple of four. One frame (``xtc_header`` and ``xtc_coord`` of [2] and [3])::

    int    magic        1995, or 2023 for GROMACS's large-system variant [4]
    int    natoms
    int    step         32 bits: GROMACS keeps only the low 32 bits [3]
    float  time         ps
    float  box[3][3]    nm; the rows are the cell vectors a, b, c
    int    natoms       again, at the head of the coordinate block
    -- natoms <= 9: 3 x natoms floats, nm, not compressed; otherwise:
    float  precision    positions are stored as round(x * precision)
    int    minint[3], maxint[3]   the smallest and largest integer per axis
    int    smallidx     the starting width of the small differences
    int    nbytes       (int64, as two 32-bit words, high first, for magic 2023)
    opaque bytes[nbytes], padded to a multiple of 4

THE COMPRESSED BLOCK (``xdr3dfcoord``)
--------------------------------------
Described from libxdrfile's ``xdrfile_decompress_coord_float`` [2] and GROMACS's
``xdr3dfcoord`` [3]; the code below is written from this description, not
copied. The block is a bit stream, read most significant bit first. Atoms come
in groups; each group is

1. one atom written in full: the three integers minus ``minint``. When every
   axis range ``maxint - minint + 1`` is at most 0xffffff the three are packed
   into one number ``(x * size_y + y) * size_z + z`` of ``bitsize`` bits (the
   bit length of the product of the three ranges), whose bytes are stored
   least significant first, each byte read most significant bit first and the
   last, partial byte holding the remaining high bits. Otherwise each axis is
   a plain unsigned field of ``bit_length(range)`` bits;
2. one bit: 1 says a new run length follows in the next 5 bits, ``v``. Then
   ``v mod 3 - 1`` (-1, 0 or +1) is added to ``smallidx`` after this group,
   and ``v - v mod 3`` is three times the number of small atoms. 0 keeps the
   previous group's run length, and ``smallidx`` stays;
3. the small atoms: each is three integers ``d + smallnum`` packed as above
   into ``smallidx`` bits, with all three ranges ``magicints[smallidx]``
   (a table of about 2^(i/3), from 8 at index 9 to 2^24 at index 72) and
   ``smallnum = magicints[smallidx] // 2``. The first small atom is placed at
   the full atom plus its ``d`` and comes FIRST in the output, before the full
   atom; each further small atom is the previous small atom plus its ``d``.
   (The writer swaps the first two atoms of a run: for water O, H1, H2 it
   writes H1 in full, then O as a difference from H1 and H2 as one from O.)

After the group ``smallidx`` moves by the step, and ``smallnum`` follows it.
The writer's search for the starting ``smallidx`` can end at 73, one past
the table, when every difference between neighbours exceeds 2^24 (precision
1e6 over tens of nm); its own code then reads past the table, so that width
is accepted while no small atom uses it, and a frame whose small atoms do is
refused.

HOW IT IS DECODED HERE
----------------------
The widths of the fields depend only on the run bits, never on the coordinate
values, so a Python loop walks the groups reading just the 1-bit flags and
5-bit run lengths and records where each coordinate field starts. While the
run length is 0, a group whose flag is 0 is one full atom and changes
nothing, so after 8 such groups in a row the rest of the stretch is counted
with numpy (the flags sit a fixed number of bits apart). numpy then reads
every field of one width at once: nine bytes from each field's first byte
form a 72-bit window, which is shifted and cut to the field; the bytes of a
packed number are put back in order with a byte swap, and the number is split
by the axis ranges. The small atoms are rebuilt with one cumulative sum per
group. Fields wider than 64 bits (axis ranges above about 2.6 million, e.g.
precision 1e5 over 30 nm) go through a scalar decoder with Python integers,
:func:`_decode_scalar`, which follows the reference step by step and serves
the tests as a second implementation.

Every frame is checked as it is decoded: a value outside its axis range, a
group that runs past ``natoms``, a ``smallidx`` outside 9 .. 73, or a bit
count that disagrees with ``nbytes`` (the writer stores exactly
``ceil(bits / 8)``) raises FrameError naming the file, the frame and its byte
offset, and no position is returned for it. So does a frame whose positions
decode 2^52 box lengths or more from the box (a precision word changed towards
zero, say): beyond 2^52 a float64 (52-bit significand) keeps no digit of the
position within the box, so wrapping would put every atom on one point.

WHAT THE CHECKS CANNOT SEE
--------------------------
**XTC holds no checksum**, so the checks above catch damage only where it
breaks the structure of the stream. Measured: one random byte changed inside
frame 1's compressed bytes, 300 times per file, in a GROMACS-written file
(cobrotoxin.xtc of MDAnalysisTests 2.10.0, 19 385 atoms) and two LAMMPS 22 Jul
2025 files (600 and 4096 atoms). 9 %, 18 % and 17 % of the changes raised
FrameError; the others decoded, moving a median of 1 to 3 atoms by a median
of 0.12 to 0.62 nm (the smallest move 0.001 nm), and the xdrfile C reader of
MDAnalysis 2.10.0 returned the same moved positions, bit for bit, on all 60
such frames compared. A frame changed that way cannot be told from an intact
one by the file alone. The same holds for a changed precision word that stays
positive and finite; when the frames of a file carry more than one precision,
a note lists which file positions carry which (a writer keeps one precision
through a run: LAMMPS ``dump_modify precision`` [7], GROMACS
``compressed-x-precision`` [8]).

UNITS AND WHAT THE FILE DOES NOT HOLD
-------------------------------------
* **Lengths are nm and times ps**, the GROMACS convention [5]; LAMMPS converts
  to them with ``sfactor = 0.1 / force->angstrom`` and ``tfactor = 0.001 /
  force->femtosecond`` (``dump_xtc.cpp`` [6], [7]), except under ``units lj``,
  where it writes reduced units unscaled and warns, and except where
  ``dump_modify sfactor`` or ``tfactor`` replaced the factors ([6], [7]). The
  file records neither, so such a file would be read here as nm and ps; the
  units note says so. nm -> Å is ``scipy.constants.nano /
  scipy.constants.angstrom`` (= 10, a definition).
* **Positions sit on a grid** of 1/precision nm: the writer rounds
  ``x * precision`` half away from zero, so a position read here lies within
  0.5/precision nm (0.005 Å at the default 1000) of what the program held,
  plus float32 rounding. Files with 9 atoms or fewer store plain float32.
* **Times are 32-bit floats.** At 10 000 ps one float step is about 0.001
  ps, which is 1 % of a 0.1 ps frame interval; ``timestep_fs`` rebuilds the
  times from the integer steps, checked against the file's times. A start
  time (GROMACS ``tinit``) is kept as an offset, taken from the frame whose
  stored time has the finest float32 rounding (the smallest absolute time),
  since a start time of 0.001 ps vanishes in the rounding of a frame at
  40 000 ps. A stored time that is not a finite number leaves that frame's
  time unknown (NaN), and the frame is kept.
* **No box origin.** LAMMPS writes absolute positions (box corner at xlo, ylo,
  zlo) and a box without its corner [6]; GROMACS boxes start at 0. FACET puts
  the corner at (0, 0, 0) and wraps the positions into that box, so they match
  the writer's modulo the box vectors.
* **No elements, types or ids.** They come from ``topology`` (any file FACET
  reads with the same atoms in the same order; row k of the XTC is row k of
  its frame 0, which FACET sorts by id; LAMMPS sorts ``dump xtc`` rows by id,
  ``sort_flag = 1`` in [6]) or from ``type_map`` keyed by atom number 1 .. N in
  file order. When the topology's frame 0 states a step that an XTC frame
  shares, the two are compared row by row modulo the box, and a note gives
  the largest difference beside the XTC grid's half step: row k is the same
  atom in both files only while the difference stays within the two files'
  rounding. Otherwise rows are matched by the atom count alone, and a note
  says that.
* **No boundary.** The file does not record which axes are periodic (LAMMPS:
  the ``boundary`` command; GROMACS: ``pbc`` in the .mdp file [8]).
  ``periodic`` is the topology's where its file states it (the flags of a
  LAMMPS dump's BOX BOUNDS line, an extended XYZ ``pbc``), else None, and the
  box is measured as periodic along a, b and c either way. A note says so: a
  slab (``boundary p p f``, ``pbc = xy``) counts its vacuum in the box volume
  that densities and g(r) normalisation use.
* **Whether positions are continuous is not stated.** LAMMPS writes wrapped
  positions unless ``dump_modify unwrap yes`` [7]; GROMACS writes what mdrun
  holds. So ``unwrapped_cart_ang`` is left None.
* **No velocities or charges.** Per-atom charges of a topology that has them
  are carried to every frame (a fixed-charge model's inputs). A topology
  without atom ids (a LAMMPS dump without an id column, whose rows FACET puts
  in element order) is refused: its row k is not atom k.
* **A zero box** (GROMACS without periodic boundaries) has no cell;
  ``box_from`` supplies one: a file FACET reads (another XTC file's frame 0
  header too, read by :func:`read_xtc_box` without elements), a Frame, or
  three cell vectors.

A FILE THAT IS CUT OR DAMAGED
-----------------------------
The file is indexed once by walking the frame headers and searching each
compressed block for a frame header inside it (no decoding). A frame whose
bytes run past the end of the file (a copy taken while it was being written)
is listed in ``Trajectory.skipped`` with the byte counts. A frame whose stated
extent holds another frame header (the magic number, this frame's atom count,
the count again 52 bytes on) has a damaged byte count: it is listed in
``skipped`` and reading resumes at that header, so no frame hides inside
another's extent. Bytes where a header is expected but the magic number is
not found -- the file's first bytes included -- are passed over to the next
header with the same atom count (four bytes 1995 or 2023, the count, and the
count again 52 bytes later); the gap is one position in ``skipped``, which
says how many median-sized frames it could have held. So is a header whose
compressed block states fewer bytes than its atoms take at the format's
minimum of 2 bits per atom (one bit of a full atom's coordinates and its run
flag): its atom count or byte count is damaged, and no memory is set aside
for the count it states. A frame whose header parses but holds a non-finite
box, a precision that is not positive, or ``maxint < minint`` is skipped by
its stated length. Frames holding an atom count other than the one most
frames hold are skipped with both counts. A file in which no frame can be
read is refused with the reason for each position; an empty file, and one
shorter than a frame header, are refused saying so.

TIMINGS
-------
Measured on this machine (Windows 11, i5-13420H, Python 3.11, numpy 2.4.6,
other sessions running, so single runs vary by up to 3x), medians of 5
repeats in two runs. Files written by MDAnalysis 2.10.0's libmdaxdr,
precision 1000, 10 000 atoms per frame:

=====================================  ==================  =================
                                       random glass-like   chain, neighbours
                                       model, 5.2 nm box   close in file
                                       (100 frames,        order (3 frames,
                                       4.9 MB; 99 % full   35 kB each; half
                                       atoms)              small atoms)
=====================================  ==================  =================
decode one frame (decode_xtc_frame)    3.6-7.0 ms          13.7-14.3 ms
  of which the Python group walk       1.4-3.2 ms          5.9-7.9 ms
one frame through Trajectory.frame     7.3-10.7 ms         18.3-19.3 ms
  (read, decode, wrap check, wrap,
  validate)
the index pass alone (headers, and     11 ms               0.3-0.5 ms
  each block searched for a header)
open the file (index pass, load        22-32 ms            29-32 ms
  record with frame 0)
every frame in order                   1.09-1.11 s
                                       (10.9-11.1 ms each)
the scalar decoder on one frame        78-86 ms            50-71 ms
=====================================  ==================  =================

The search of each block for a frame header inside it reads the whole file
once at open: the index pass of the 4.9 MB file took 3 ms without it and 11
ms with it. The wrap check (:func:`_check_wrap`, one 3 x 3 solve per frame)
takes 0.3 ms of a 10 000-atom frame. An earlier run of the same table, before
both were added and on a less loaded machine, gave 3.5-6 ms per decode and
0.80-0.86 s for every frame in order. A slower first version (a bit matrix
per field, one list entry per group) took 18-24 ms per glass-like frame; the
text readers of :mod:`.md_readers` take 18-84 ms per 10 000-atom frame on
this machine.

REFERENCES
----------
[1] M. Eisler (ed.), "XDR: External Data Representation Standard", RFC 4506,
    STD 67 (May 2006), https://www.rfc-editor.org/rfc/rfc4506 -- §4.1 integer,
    §4.5 hyper integer, §4.6 float, §4.10 variable-length opaque data
[2] E. Lindahl and D. van der Spoel, libxdrfile (xdrfile.c,
    xdrfile_xtc.c), as carried by MDAnalysis, BSD 2-clause licence,
    https://github.com/MDAnalysis/mdanalysis/blob/develop/package/MDAnalysis/lib/formats/src/xdrfile.c
    and .../src/xdrfile_xtc.c (``xdrfile_decompress_coord_float``,
    ``decodebits``, ``decodeints``, ``magicints``, ``xtc_header``)
[3] GROMACS source, release-2025: src/gromacs/fileio/libxdrf.cpp
    (``xdr3dfcoord``), xtcio.cpp (``xtc_header``), xdrd.cpp (``xdr_int64``),
    xdrf.h (``XTC_MAGIC`` 1995, ``XTC_NEW_MAGIC`` 2023), LGPL 2.1 or later,
    https://gitlab.com/gromacs/gromacs/-/tree/release-2025/src/gromacs/fileio
[4] GROMACS 2023.2 release notes, "Fixes for gmx tools: XTC files for large
    systems", https://manual.gromacs.org/2023.2/release-notes/2023/2023.2.html
[5] GROMACS manual, "File formats: xtc",
    https://manual.gromacs.org/current/reference-manual/file-formats.html
[6] LAMMPS source, src/EXTRA-DUMP/dump_xtc.cpp, release stable_22Jul2025,
    https://github.com/lammps/lammps/blob/stable_22Jul2025/src/EXTRA-DUMP/dump_xtc.cpp
[7] LAMMPS documentation, "dump_modify command" (unwrap, precision, sfactor,
    tfactor), https://docs.lammps.org/dump_modify.html
[8] GROMACS manual, "Molecular dynamics parameters (.mdp options)": pbc,
    compressed-x-precision, compressed-x-grps,
    https://manual.gromacs.org/current/user-guide/mdp-options.html
"""
from __future__ import annotations

import math
import os
import struct
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import constants

from . import md_model
from .md_formats_base import FormatSpec
from .md_model import (Frame, FrameError, Trajectory, charges_per_element,
                       frame_from_arrays, validate_symbol)
from .readers import UnsupportedFormat, cell_from_vectors
from .structure import Structure

__all__ = ["XTC_MAGIC", "XTC_NEW_MAGIC", "ANG_PER_NM", "XtcFrame",
           "decode_xtc_frame", "sniff_xtc", "read_xtc", "read_xtc_box",
           "FORMATS"]

# ---------------------------------------------------------------------------
# constants: each is a definition of the format or of a unit, and says so
# ---------------------------------------------------------------------------

# The magic numbers of a frame header (GROMACS xdrf.h [3]); 2023 marks a frame
# whose byte count is 64-bit (GROMACS 2023.2 and later, above 298 261 617
# atoms [4]).
XTC_MAGIC = 1995
XTC_NEW_MAGIC = 2023
_MAGICS = (XTC_MAGIC, XTC_NEW_MAGIC)

# nm -> Å, a definition: 1 nm = 1e-9 m and 1 Å = 1e-10 m.
ANG_PER_NM: float = constants.nano / constants.angstrom

# fs -> ps, a definition, for timestep_fs.
_PS_PER_FS: float = constants.femto / constants.pico

# The table of small-difference ranges (``magicints`` of [2] and [3]): entry i
# is about 2^(i/3), so three values below it fit in i bits. Part of the
# format's definition; indices below 9 are unused.
_MAGICINTS = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
    80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
    1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
    16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
    131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
    832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
    4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216)
_FIRSTIDX = 9                      # the first non-zero entry
_LASTIDX = len(_MAGICINTS)         # 73

# Frames with this many atoms or fewer are stored as plain floats ([2], [3]).
_UNCOMPRESSED_MAX_ATOMS = 9

# An axis range above this is written as a plain bit field per axis ([2]).
_PACKED_RANGE_MAX = 0xFFFFFF

# The fewest bits one atom takes in a compressed block, a consequence of the
# format: a full atom holds at least one bit of coordinates (all three axis
# ranges 1 give a product of 1, whose bit length is 1) plus its 1-bit run
# flag; a small atom takes smallidx >= 9 bits.
_MIN_BITS_PER_ATOM = 2

# Fractional coordinates at or beyond this magnitude keep no digit within the
# box once wrapped: a float64 has a 52-bit significand, so its spacing there
# is 1 or more.
_WRAP_LIMIT = 2.0 ** np.finfo(np.float64).nmant

# Byte layout of a frame: the fixed header (magic, natoms, step, time, box,
# natoms again), and the compressed block's own header before the byte count.
_HEAD_BYTES = 56
_BLOCK_HEAD_BYTES = 32             # precision, minint[3], maxint[3], smallidx

# The widest field the numpy path decodes: a field is cut from a 64-bit
# window held in uint64. Wider fields go through the scalar decoder.
_NUMPY_MAX_BITS = 64

# How many plain groups (one atom, run flag 0) the group walk steps over one
# at a time before it counts the rest of the stretch with numpy. A
# performance choice, set from timings on this machine; it does not change
# what is decoded.
_PLAIN_STEPS = 8

# How much of the file the resync search reads at a time. A performance
# choice; it does not change what is found.
_SCAN_BYTES = 1 << 20

_GZIP_MAGIC = b"\x1f\x8b"

FORMAT_NAME = "gromacs-xtc"


# ---------------------------------------------------------------------------
# decoding one frame
# ---------------------------------------------------------------------------

class XtcFrame(NamedTuple):
    """One decoded frame, in the file's own units.

    ``coords_nm`` (N, 3) float64 is ``ints / precision`` for a compressed
    frame and the stored float32 values otherwise; ``ints`` is the integer
    grid (None for an uncompressed frame); ``precision`` is None for an
    uncompressed frame.
    """

    magic: int
    natoms: int
    step: int
    time_ps: float
    box_nm: np.ndarray
    precision: float | None
    coords_nm: np.ndarray
    ints: np.ndarray | None


def _pad4(n: int) -> int:
    """n rounded up to a multiple of four (XDR opaque padding [1, §4.10])."""
    return (n + 3) & ~3


def _bits_at(buf: bytes, pos: int, width: int) -> int:
    """``width`` bits starting at bit ``pos``, most significant bit first
    (``buf`` holds the field's bytes)."""
    first = pos >> 3
    last = (pos + width + 7) >> 3
    value = int.from_bytes(buf[first:last], "big")
    return (value >> (8 * (last - first) - (pos & 7) - width)) & ((1 << width) - 1)


def _packed_at(buf: bytes, pos: int, width: int) -> int:
    """A packed number of ``width`` bits at ``pos``: its bytes come least
    significant first, each read most significant bit first, and the last
    group holds the remaining (width - 8k) high bits."""
    full = (width - 1) // 8
    value = 0
    for j in range(full):
        value |= _bits_at(buf, pos + 8 * j, 8) << (8 * j)
    return value | (_bits_at(buf, pos + 8 * full, width - 8 * full) << (8 * full))


def _split3(value: int, size_y: int, size_z: int) -> tuple[int, int, int]:
    """(x, y, z) from ``(x * size_y + y) * size_z + z``."""
    value, z = divmod(value, size_z)
    x, y = divmod(value, size_y)
    return x, y, z


def _fields(octets: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    """The ``width``-bit fields (width <= 64) starting at bit ``starts``, read
    most significant bit first, as uint64: nine bytes from each start's byte,
    shifted so the field's first bit is the top bit, then cut to ``width``."""
    first = starts >> 3
    rows = octets[first[:, None] + np.arange(9)]
    high = np.ascontiguousarray(rows[:, :8]).view(">u8").ravel().astype(
        np.uint64)
    low = rows[:, 8].astype(np.uint64)
    shift = (starts & 7).astype(np.uint64)
    window = (high << shift) | (low >> (np.uint64(8) - shift))
    return window >> np.uint64(64 - width)


def _packed_values(plain: np.ndarray, width: int) -> np.ndarray:
    """Packed numbers (see :func:`_packed_at`) from their fields read most
    significant bit first: the full bytes come in reverse order of
    significance, and the last ``width - 8k`` bits are the top part."""
    full = (width - 1) // 8
    if full == 0:
        return plain
    rest = width - 8 * full
    top = plain >> np.uint64(rest)              # the full bytes, first on top
    reversed_ = top.byteswap() >> np.uint64(8 * (8 - full))
    return reversed_ | ((plain & np.uint64((1 << rest) - 1))
                        << np.uint64(8 * full))


def _plain_stretch(bits: np.ndarray, flag_bit: int, stride: int,
                   limit: int) -> int:
    """How many groups from the one whose run flag sits at ``flag_bit`` have
    a flag of 0 (at most ``limit``), groups being ``stride`` bits apart."""
    room = (bits.size - 1 - flag_bit) // stride + 1 if flag_bit < bits.size \
        else 0
    limit = min(limit, room)
    count, chunk = 0, 16
    while count < limit:
        take = min(chunk, limit - count)
        flags = bits[flag_bit + stride * (count + np.arange(take))]
        hit = np.flatnonzero(flags)
        if hit.size:
            return count + int(hit[0])
        count += take
        chunk = min(4 * chunk, 4096)
    return count


class _Groups(NamedTuple):
    """Where the fields of a compressed block lie (bit offsets)."""

    large_pos: np.ndarray       # (G,) start of each full atom
    n_small: np.ndarray         # (G,) small atoms after it
    run_group: np.ndarray       # (R,) the groups with small atoms
    small_pos: np.ndarray       # (R,) start of each such group's first one
    small_idx: np.ndarray       # (R,) their smallidx (bit width)
    small_num: np.ndarray       # (R,) their smallnum
    end_bit: int                # the first bit not used
    run_changes: int            # groups that carried a new run length
    idx_steps: tuple[int, int]  # smallidx decreases, increases


def _walk_groups(buf: bytes, natoms: int, large_bits: int, smallidx: int,
                 total_bits: int, bits: np.ndarray | None = None) -> _Groups:
    """Walk the groups reading only the run bits; raise ValueError on a
    stream that cannot be the encoder's.

    While the run length is 0, a group whose flag is 0 holds one atom and
    leaves the state as it is; after :data:`_PLAIN_STEPS` such groups in a
    row the rest of the stretch is counted with numpy (:func:`_plain_stretch`)
    instead of one group at a time.
    """
    _check_width(smallidx)
    smaller = _half(max(_FIRSTIDX, smallidx - 1))
    smallnum = _half(smallidx)
    stride = large_bits + 1
    marks: list[tuple[int, int]] = []        # (first bit, groups) per stretch
    runs: list[tuple[int, int, int, int, int]] = []
    mark = marks.append
    note_run = runs.append
    pos = done = run = changes = down = up = group = 0
    while done < natoms:
        if pos > total_bits:
            raise ValueError(f"the compressed block ({total_bits // 8} bytes) "
                             f"ends after {done} of {natoms} atoms")
        flag_bit = pos + large_bits
        flag = (buf[flag_bit >> 3] >> (7 - (flag_bit & 7))) & 1
        if run == 0 and not flag:
            start, count = pos, 1
            pos += stride
            while count < _PLAIN_STEPS and done + count < natoms:
                flag_bit = pos + large_bits
                if (buf[flag_bit >> 3] >> (7 - (flag_bit & 7))) & 1:
                    break
                pos += stride
                count += 1
            else:
                if count == _PLAIN_STEPS and done + count < natoms:
                    if bits is None:
                        bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))
                    more = _plain_stretch(bits, pos + large_bits, stride,
                                          natoms - done - count)
                    pos += more * stride
                    count += more
            mark((start, count))
            done += count
            group += count
            continue
        mark((pos, 1))
        pos = flag_bit + 1
        step = 0
        if flag:
            q = pos >> 3
            v = (((buf[q] << 8) | buf[q + 1]) >> (11 - (pos & 7))) & 31
            pos += 5
            step = v % 3
            run = v - step
            step -= 1
            changes += 1
        n = run // 3
        if n:
            _check_run_width(smallidx)
            if done + 1 + n > natoms:
                raise ValueError(f"a run of {n} small atoms takes the count to "
                                 f"{done + 1 + n}, past the {natoms} atoms of "
                                 "the frame; the block is damaged")
            note_run((group, n, pos, smallidx, smallnum))
            pos += n * smallidx
        done += 1 + n
        group += 1
        if step:
            smallidx += step
            _check_width(smallidx)
            if step < 0:
                down += 1
                smallnum = smaller
                smaller = _half(smallidx - 1) if smallidx > _FIRSTIDX else 0
            else:
                up += 1
                smaller = smallnum
                smallnum = _half(smallidx)
    if pos > total_bits:
        raise ValueError(f"the compressed block ({total_bits // 8} bytes) "
                         f"ends before all {natoms} atoms are decoded")
    table = np.array(marks, dtype=np.int64).reshape(-1, 2)
    counts = table[:, 1]
    first = np.cumsum(counts) - counts
    large_pos = (np.repeat(table[:, 0], counts)
                 + stride * (np.arange(group, dtype=np.int64)
                             - np.repeat(first, counts)))
    found = np.array(runs, dtype=np.int64).reshape(-1, 5)
    n_small = np.zeros(group, dtype=np.int64)
    n_small[found[:, 0]] = found[:, 1]
    return _Groups(large_pos, n_small, found[:, 0], found[:, 2], found[:, 3],
                   found[:, 4], pos, changes, (down, up))


def _assemble(large: np.ndarray, groups: _Groups, deltas: np.ndarray,
              natoms: int) -> np.ndarray:
    """Output order from full atoms (G, 3) and small differences (S, 3):
    per group [A0, full, A1, A2, ...] with A0 = full + d0 and A_j = A_{j-1}
    + d_j."""
    n = groups.n_small
    counts = 1 + n
    start = np.cumsum(counts) - counts
    out = np.empty((natoms, 3), dtype=np.int64)
    out[start + (n > 0)] = large
    if deltas.shape[0]:
        group = np.repeat(np.arange(n.size), n)
        first = np.cumsum(n) - n
        running = np.cumsum(deltas, axis=0)
        before = np.zeros((n.size, 3), dtype=np.int64)
        has = (n > 0) & (first > 0)
        before[has] = running[first[has] - 1]
        within = np.arange(deltas.shape[0]) - first[group]
        slot = start[group] + np.where(within == 0, 0, 1 + within)
        out[slot] = large[group] + running - before[group]
    return out


def _decode_numpy(buf: bytes, natoms: int, sizes: tuple[int, int, int],
                  bitsize: int, axis_bits: tuple[int, int, int],
                  smallidx: int, total_bits: int) -> np.ndarray | None:
    """The vectorised decoder; None when a field is wider than
    :data:`_NUMPY_MAX_BITS` (the caller then uses :func:`_decode_scalar`)."""
    large_bits = bitsize if bitsize else sum(axis_bits)
    octets = np.frombuffer(buf, dtype=np.uint8)
    groups = _walk_groups(buf, natoms, large_bits, smallidx, total_bits)
    _check_byte_count(groups.end_bit, total_bits)
    n_runs = groups.run_group.size
    if bitsize > _NUMPY_MAX_BITS or (
            n_runs and int(groups.small_idx.max()) > _NUMPY_MAX_BITS):
        return None
    starts = groups.large_pos
    if bitsize:
        value = _packed_values(_fields(octets, starts, bitsize), bitsize)
        size_y, size_z = np.uint64(sizes[1]), np.uint64(sizes[2])
        z = value % size_z
        value //= size_z
        y = value % size_y
        x = value // size_y
        large = np.stack([x, y, z], axis=1)
    else:
        columns, offset = [], 0
        for width in axis_bits:
            columns.append(_fields(octets, starts + offset, width))
            offset += width
        large = np.stack(columns, axis=1)
    over = large >= np.asarray(sizes, dtype=np.uint64)
    if over.any():
        atom = int(np.flatnonzero(over.any(axis=1))[0])
        raise ValueError(f"full atom {atom} of the stream lies outside the "
                         "range minint..maxint the frame states; the block "
                         "is damaged")
    large = large.astype(np.int64)

    per_run = groups.n_small[groups.run_group]
    total = int(per_run.sum())
    deltas = np.empty((total, 3), dtype=np.int64)
    if total:
        run = np.repeat(np.arange(n_runs), per_run)
        within = np.arange(total) - np.repeat(np.cumsum(per_run) - per_run,
                                              per_run)
        widths = groups.small_idx[run]
        starts = groups.small_pos[run] + within * widths
        for width in np.unique(widths).tolist():
            pick = np.flatnonzero(widths == width)
            value = _packed_values(_fields(octets, starts[pick], width), width)
            size = np.uint64(_MAGICINTS[width])
            z = value % size
            value //= size
            y = value % size
            x = value // size
            if (x >= size).any():
                raise ValueError("a small difference lies outside its range "
                                 f"(width {width}); the block is damaged")
            deltas[pick] = (np.stack([x, y, z], axis=1).astype(np.int64)
                            - groups.small_num[run[pick]][:, None])
    return _assemble(large, groups, deltas, natoms)


def _decode_scalar(buf: bytes, natoms: int, sizes: tuple[int, int, int],
                   bitsize: int, axis_bits: tuple[int, int, int],
                   smallidx: int, total_bits: int) -> np.ndarray:
    """The reference walk, one field at a time with Python integers. Used for
    fields wider than 64 bits, and by the tests as a second implementation."""
    _check_width(smallidx)
    out = np.empty((natoms, 3), dtype=np.int64)
    smaller = _half(max(_FIRSTIDX, smallidx - 1))
    smallnum = _half(smallidx)
    pos = done = run = 0
    while done < natoms:
        if pos > total_bits:
            raise ValueError(f"the compressed block ({total_bits // 8} bytes) "
                             f"ends after {done} of {natoms} atoms")
        if bitsize:
            full = list(_split3(_packed_at(buf, pos, bitsize), sizes[1],
                                sizes[2]))
            pos += bitsize
        else:
            full = []
            for width in axis_bits:
                full.append(_bits_at(buf, pos, width))
                pos += width
        if any(v >= s for v, s in zip(full, sizes)):
            raise ValueError(f"full atom {done} lies outside the range "
                             "minint..maxint the frame states; the block is "
                             "damaged")
        flag = _bits_at(buf, pos, 1)
        pos += 1
        step = 0
        if flag:
            v = _bits_at(buf, pos, 5)
            pos += 5
            step = v % 3
            run = v - step
            step -= 1
        n = run // 3
        if done + 1 + n > natoms:
            raise ValueError(f"a run of {n} small atoms takes the count to "
                             f"{done + 1 + n}, past the {natoms} atoms of the "
                             "frame; the block is damaged")
        if n:
            _check_run_width(smallidx)
        if n == 0:
            out[done] = full
            done += 1
        prev = full
        for k in range(n):
            size = _MAGICINTS[smallidx]
            x, y, z = _split3(_packed_at(buf, pos, smallidx), size, size)
            pos += smallidx
            if x >= size:
                raise ValueError("a small difference lies outside its range "
                                 f"(width {smallidx}); the block is damaged")
            here = [prev[0] + x - smallnum, prev[1] + y - smallnum,
                    prev[2] + z - smallnum]
            out[done] = here
            if k == 0:                     # the first small atom comes first
                out[done + 1] = full
                done += 2
            else:
                done += 1
            prev = here
        if pos > total_bits:
            raise ValueError(f"the compressed block ({total_bits // 8} bytes) "
                             f"ends before all {natoms} atoms are decoded")
        if step:
            smallidx += step
            _check_width(smallidx)
            if step < 0:
                smallnum = smaller
                smaller = _half(smallidx - 1) if smallidx > _FIRSTIDX else 0
            else:
                smaller = smallnum
                smallnum = _half(smallidx)
    _check_byte_count(pos, total_bits)
    return out


def _half(index: int) -> int:
    """``magicints[index] // 2``, or -1 one past the table: the writer's
    search for the starting width ends at index 73 when every difference
    between neighbours exceeds 2^24, and its own code then reads past the
    table, so no small atom can be decoded at that width
    (:func:`_check_run_width`)."""
    return _MAGICINTS[index] // 2 if index < _LASTIDX else -1


def _check_width(smallidx: int) -> None:
    """The small-difference width stays within what the writer can reach,
    9 to 73 (one past the table, see :func:`_half`)."""
    if not _FIRSTIDX <= smallidx <= _LASTIDX:
        raise ValueError(f"the small-difference width (smallidx {smallidx}) "
                         f"lies outside {_FIRSTIDX} to {_LASTIDX}, the widths "
                         "the writer uses; the block is damaged")


def _check_run_width(smallidx: int) -> None:
    """Small atoms need a width inside the table."""
    if smallidx >= _LASTIDX:
        raise ValueError(
            f"small atoms use width {smallidx}, one past the writer's table of "
            f"{_LASTIDX} entries; the reference code reads past its table "
            "there, so these differences cannot be decoded (a smaller "
            "precision in the writer avoids it)")


def _check_byte_count(end_bit: int, total_bits: int) -> None:
    used = (end_bit + 7) // 8
    if used != total_bits // 8:
        raise ValueError(
            f"decoding every atom used {used} bytes of the compressed block, "
            f"which states {total_bits // 8}; the writer stores exactly the "
            "bytes it used, so the block is damaged")


# Bytes added after a compressed block so that a group's reads never index
# past the buffer before its end is checked: one group reads at most 72 + 6 +
# 8 x 72 bits (about 82 bytes). A buffer size, not part of the format.
_READ_PAD = bytes(128)


def _decode_block(block: bytes, natoms: int, minint, maxint, smallidx: int,
                  *, scalar: bool = False) -> np.ndarray:
    """The integer grid (natoms, 3) of one compressed block."""
    sizes = tuple(int(b) - int(a) + 1 for a, b in zip(minint, maxint))
    if any(s < 1 for s in sizes):
        raise ValueError(f"maxint {list(maxint)} lies below minint "
                         f"{list(minint)}")
    if max(sizes) > _PACKED_RANGE_MAX:
        bitsize = 0
        axis_bits = tuple(min(s.bit_length(), 32) for s in sizes)
    else:
        bitsize = (sizes[0] * sizes[1] * sizes[2]).bit_length()
        axis_bits = (0, 0, 0)
    total_bits = 8 * len(block)
    buf = block + _READ_PAD
    ints = None
    if not scalar:
        ints = _decode_numpy(buf, natoms, sizes, bitsize, axis_bits, smallidx,
                             total_bits)
    if ints is None:
        ints = _decode_scalar(buf, natoms, sizes, bitsize, axis_bits,
                              smallidx, total_bits)
    return ints + np.asarray(minint, dtype=np.int64)


def _floats(data: bytes, offset: int, count: int) -> np.ndarray:
    """``count`` big-endian float32 values from ``offset``, as float64. A
    signalling NaN among them (a damaged byte can make one) makes numpy warn
    on the cast; the callers test every value for finiteness right after, so
    the warning is silenced here rather than let out to the caller."""
    with np.errstate(invalid="ignore"):
        return np.frombuffer(data, dtype=">f4", count=count,
                             offset=offset).astype(np.float64)


def decode_xtc_frame(data: bytes, *, scalar: bool = False) -> XtcFrame:
    """Decode one XTC frame from its bytes (header included).

    Raises ValueError saying what in the frame could not be read. ``scalar``
    forces the step-by-step decoder (for tests and comparisons).
    """
    if len(data) < _HEAD_BYTES:
        raise ValueError(f"the frame holds {len(data)} bytes; its header alone "
                         f"takes {_HEAD_BYTES}")
    magic, natoms, step = struct.unpack_from(">iii", data, 0)
    if magic not in _MAGICS:
        raise ValueError(f"the frame starts with {magic}, not the XTC magic "
                         f"number {XTC_MAGIC} (or {XTC_NEW_MAGIC})")
    (time_ps,) = struct.unpack_from(">f", data, 12)
    box = _floats(data, 16, 9).reshape(3, 3)
    (lsize,) = struct.unpack_from(">i", data, 52)
    if natoms < 1 or lsize != natoms:
        raise ValueError(f"the header gives {natoms} atoms and the coordinate "
                         f"block {lsize}")
    if natoms <= _UNCOMPRESSED_MAX_ATOMS:
        need = _HEAD_BYTES + 12 * natoms
        if len(data) < need:
            raise ValueError(f"the frame holds {len(data)} of its {need} bytes")
        coords = _floats(data, _HEAD_BYTES, 3 * natoms)
        return XtcFrame(magic, natoms, step, float(time_ps), box, None,
                        coords.reshape(natoms, 3), None)
    head = _HEAD_BYTES + _BLOCK_HEAD_BYTES
    count_bytes = 8 if magic == XTC_NEW_MAGIC else 4
    if len(data) < head + count_bytes:
        raise ValueError(f"the frame holds {len(data)} bytes, fewer than its "
                         "compressed block's header")
    (precision,) = struct.unpack_from(">f", data, _HEAD_BYTES)
    minint = struct.unpack_from(">3i", data, _HEAD_BYTES + 4)
    maxint = struct.unpack_from(">3i", data, _HEAD_BYTES + 16)
    (smallidx,) = struct.unpack_from(">i", data, _HEAD_BYTES + 28)
    (nbytes,) = struct.unpack_from(">q" if count_bytes == 8 else ">i", data,
                                   head)
    if not (math.isfinite(precision) and precision > 0):
        raise ValueError(f"the precision {precision!r} is not a positive number")
    start = head + count_bytes
    if nbytes < 0 or start + nbytes > len(data):
        raise ValueError(f"the compressed block states {nbytes} bytes and the "
                         f"frame holds {len(data) - start} after its header")
    ints = _decode_block(data[start:start + nbytes], natoms, minint, maxint,
                         smallidx, scalar=scalar)
    return XtcFrame(magic, natoms, step, float(time_ps), box,
                    float(precision), ints / float(precision), ints)


# ---------------------------------------------------------------------------
# recognising and indexing a file
# ---------------------------------------------------------------------------

def sniff_xtc(head: bytes, path=None) -> bool:
    """True when ``head`` starts like an XTC frame: the magic number 1995 or
    2023, a positive atom count, and the same count again at byte 52 (the
    head of the coordinate block). Never raises."""
    try:
        if len(head) < _HEAD_BYTES:
            return False
        magic, natoms = struct.unpack_from(">ii", head, 0)
        (lsize,) = struct.unpack_from(">i", head, 52)
        return magic in _MAGICS and natoms >= 1 and lsize == natoms
    except Exception:                      # noqa: BLE001 - a sniff never raises
        return False


class _Record(NamedTuple):
    position: int
    offset: int
    end: int
    magic: int
    natoms: int
    step: int
    time_ps: float
    box_nm: np.ndarray
    precision: float | None
    block: int = -1    # first byte of the compressed data, -1 for plain floats
    nbytes: int = -1   # the byte count the frame states, -1 for plain floats


class _Problem(NamedTuple):
    kind: str          # 'cut' (runs past the end), 'gap' (no header), 'skip'
    reason: str
    end: int           # where the next frame starts, for 'skip'
    block: int = -1    # for 'skip': as in _Record
    nbytes: int = -1
    natoms: int = 0
    step: int = 0


def _frame_extent(head: bytes, offset: int, file_size: int
                  ) -> _Record | _Problem:
    """The frame at ``offset`` from its first bytes: a record, or why not."""
    left = file_size - offset
    if len(head) < 4:
        return _Problem("cut", f"the last {left} byte(s) of the file hold no "
                        "frame header (fewer than 4 bytes)", file_size)
    (magic,) = struct.unpack_from(">i", head, 0)
    if magic not in _MAGICS:
        return _Problem("gap", f"the 4 bytes at byte {offset} read {magic}, "
                        f"not the XTC magic number {XTC_MAGIC} (or "
                        f"{XTC_NEW_MAGIC})", file_size)
    if len(head) < _HEAD_BYTES:
        return _Problem("cut", f"the frame header at byte {offset} is cut: the "
                        f"file ends {left} bytes into its {_HEAD_BYTES} (a copy "
                        "taken while it was being written)", file_size)
    natoms, step = struct.unpack_from(">ii", head, 4)
    (lsize,) = struct.unpack_from(">i", head, 52)
    if natoms < 1 or lsize != natoms:
        return _Problem("gap", f"the header at byte {offset} gives {natoms} "
                        f"atoms and its coordinate block {lsize}", file_size)
    (time_ps,) = struct.unpack_from(">f", head, 12)
    box = _floats(head, 16, 9).reshape(3, 3)
    problem = ""
    block = nbytes = -1
    if natoms <= _UNCOMPRESSED_MAX_ATOMS:
        end = offset + _HEAD_BYTES + 12 * natoms
        precision = None
    else:
        count_bytes = 8 if magic == XTC_NEW_MAGIC else 4
        need = _HEAD_BYTES + _BLOCK_HEAD_BYTES + count_bytes
        if len(head) < need:
            return _Problem("cut", f"the frame at byte {offset} (step {step}) "
                            f"is cut: the file ends {left} bytes into it, "
                            f"inside the {need}-byte header of its coordinate "
                            "block (a copy taken while it was being written)",
                            file_size)
        (precision,) = struct.unpack_from(">f", head, _HEAD_BYTES)
        minint = struct.unpack_from(">3i", head, _HEAD_BYTES + 4)
        maxint = struct.unpack_from(">3i", head, _HEAD_BYTES + 16)
        (nbytes,) = struct.unpack_from(">q" if count_bytes == 8 else ">i",
                                       head, _HEAD_BYTES + _BLOCK_HEAD_BYTES)
        if nbytes < 0:
            return _Problem("gap", f"the frame at byte {offset} states a "
                            f"negative byte count ({nbytes})", file_size)
        least = -(-natoms * _MIN_BITS_PER_ATOM // 8)
        if nbytes < least:
            return _Problem(
                "gap", f"the frame at byte {offset} (step {step}) states "
                f"{natoms} atoms and a compressed block of {nbytes} bytes, "
                f"fewer than the {least} bytes {natoms} atoms take at the "
                f"format's minimum of {_MIN_BITS_PER_ATOM} bits each (one bit "
                "of a full atom's coordinates and its run flag), so its atom "
                "count or its byte count is damaged", file_size)
        block = offset + need
        end = block + _pad4(nbytes)
        precision = float(precision)
        if not (math.isfinite(precision) and precision > 0):
            problem = f"its precision {precision!r} is not a positive number"
        elif any(b < a for a, b in zip(minint, maxint)):
            problem = (f"its maxint {list(maxint)} lies below minint "
                       f"{list(minint)}")
    if end > file_size:
        return _Problem("cut", f"the frame at byte {offset} (step {step}) "
                        f"needs {end - offset} bytes and the file holds "
                        f"{left} from there (a copy taken while it was being "
                        "written)", file_size)
    if not problem and not np.isfinite(box).all():
        problem = "it holds a box value that is not a finite number"
    if problem:
        return _Problem("skip", f"the frame at byte {offset} (step {step}): "
                        f"{problem}", end, block, nbytes, natoms, step)
    # A time that is not finite leaves the frame's time unknown; the frame is
    # kept (read_xtc notes it, and timestep_fs can rebuild it from the step).
    return _Record(-1, offset, end, magic, natoms, step, float(time_ps), box,
                   precision, block, nbytes)


def _resync(handle, start: int, file_size: int, natoms: int | None,
            stop: int | None = None) -> int | None:
    """The offset of the next frame header at or after ``start`` and before
    ``stop`` (the end of the file when None): the magic number, then
    ``natoms`` (any positive count when None), then the same count 52 bytes
    after the magic. None when there is none."""
    patterns = [struct.pack(">i", m) + (struct.pack(">i", natoms)
                                        if natoms else b"") for m in _MAGICS]
    longest = max(len(p) for p in patterns)
    stop = file_size if stop is None else min(stop, file_size)
    offset = start
    while offset < stop:
        span = min(_SCAN_BYTES, stop - offset)
        handle.seek(offset)
        # A pattern that starts inside the span may end past it.
        chunk = handle.read(span + longest)
        hits = []
        for pattern in patterns:
            at = chunk.find(pattern)
            while -1 < at < span:
                hits.append(at)
                at = chunk.find(pattern, at + 1)
        for at in sorted(hits):
            handle.seek(offset + at)
            if sniff_xtc(handle.read(_HEAD_BYTES)):
                return offset + at
        offset += span
    return None


def _index(path: Path) -> tuple[list[_Record], dict[int, str]]:
    """Walk the frame headers: the readable frames and the skipped positions.

    A frame that runs past the end of the file is the file's last (a copy
    taken while the writer was at work) unless another frame header follows
    inside it, in which case its byte count is damaged and the walk goes on
    from that header. A frame that ends within the file is searched the same
    way: a frame header inside the compressed bytes it states (an enlarged
    byte count landing on a later frame's header, say) means its byte count
    is damaged, and the walk goes on from that header, so that the frames
    the stated extent covers are read, not hidden.
    """
    file_size = path.stat().st_size
    records: list[_Record] = []
    skipped: dict[int, str] = {}
    gaps: list[tuple[int, int, int]] = []        # position, start, end
    position = 0
    offset = 0
    head_bytes = _HEAD_BYTES + _BLOCK_HEAD_BYTES + 8
    with open(path, "rb") as handle:
        while offset < file_size:
            handle.seek(offset)
            head = handle.read(head_bytes)
            found = _frame_extent(head, offset, file_size)
            if found.block >= 0 and (isinstance(found, _Record)
                                     or found.kind == "skip"):
                # Any atom count: a hidden frame may hold another count.
                inner = _resync(handle, found.block, file_size, None,
                                stop=found.end)
                if inner is not None:
                    skipped[position] = (
                        f"the frame at byte {offset} (step {found.step}): its "
                        f"byte count ({found.nbytes}) is damaged: a frame "
                        f"header follows at byte {inner}, inside the "
                        f"{found.end - offset} bytes the frame states; reading "
                        "resumed at that header")
                    position += 1
                    offset = inner
                    continue
            if isinstance(found, _Record):
                records.append(found._replace(position=position))
                position += 1
                offset = found.end
                continue
            if found.kind == "skip":
                skipped[position] = found.reason
                position += 1
                offset = found.end
                continue
            expected = records[-1].natoms if records else None
            after = _resync(handle, offset + 4, file_size, expected)
            if found.kind == "cut" and after is None:
                skipped[position] = found.reason
                break
            end = file_size if after is None else after
            reason = found.reason
            if found.kind == "cut":
                reason = reason.split(" (a copy")[0] + (
                    f", yet another frame header follows at byte {after}: "
                    "its byte count is damaged")
            gaps.append((position, offset, end))
            skipped[position] = (
                f"bytes {offset} to {end} ({end - offset} bytes): {reason}; "
                + ("no further frame header follows, so the rest of the file "
                   "is not read" if after is None else
                   f"reading resumed at the next frame header, at byte {after}"))
            position += 1
            if after is None:
                break
            offset = after
    if gaps and records:
        median = float(np.median([r.end - r.offset for r in records]))
        for pos, start, end in gaps:
            skipped[pos] += (f"; the gap could have held about "
                             f"{(end - start) / median:.3g} frame(s) of the "
                             f"median size here, {median:.0f} bytes")
    return records, skipped


# ---------------------------------------------------------------------------
# elements, boxes and times
# ---------------------------------------------------------------------------

_CRYSTAL_NOT_TOPOLOGY = (
    "a crystal structure (CIF, POSCAR, PDB) is not taken as a topology: it "
    "lists the atoms of one cell after its symmetry is applied, not the run's "
    "atoms in the order the XTC file holds them; pass the LAMMPS data file the "
    "run read, or a dump or extended XYZ file of the same atoms")


def _model_frame(source, what: str, name: str, options: dict) -> tuple[
        Frame, str, dict, str, tuple | None]:
    """Frame 0 of ``source`` (a path FACET reads, a Trajectory or a Frame),
    its type_map_source, its type map, how to name it, and the periodic
    flags its file states (None when it states none)."""
    if isinstance(source, Frame):
        if options:
            raise ValueError(
                f"{name}: {', '.join(sorted(options))} applies to a "
                f"{what} file that FACET reads; the {what} given is already a "
                "frame")
        return source, "in memory", {}, f"the {what} frame given", None
    if isinstance(source, Trajectory):
        if options:
            raise ValueError(
                f"{name}: {', '.join(sorted(options))} applies to a "
                f"{what} file that FACET reads; the {what} given is already a "
                "trajectory")
        _check_row_order(source, what, name)
        return (source.frame(0), source.type_map_source, dict(source.type_map),
                f"{what} {Path(source.source_path).name}",
                getattr(source, "periodic", None))
    if not isinstance(source, (str, os.PathLike)):
        extra = f"; {_CRYSTAL_NOT_TOPOLOGY}" if isinstance(source, Structure) \
            else ""
        raise ValueError(
            f"{name}: {what} needs a file FACET reads (a path), a Trajectory "
            f"or a Frame, not {type(source).__name__}{extra}")
    from . import md_readers                # lazy: md_readers lists this module

    try:
        model = md_readers.read_trajectory(source, **options)
    except UnsupportedFormat as error:
        raise UnsupportedFormat(
            f"{name}: {what}={Path(source).name} cannot be opened as an MD "
            f"model: {error}"
            + (f" As for topology=, {_CRYSTAL_NOT_TOPOLOGY}."
               if what == "topology" else "")) from error
    try:
        _check_row_order(model, what, name)
        frame = model.frame(0)
    finally:
        close = getattr(model, "close", None)
        if close is not None:
            close()
    return (frame, model.type_map_source, dict(model.type_map),
            f"{what} {Path(source).name}", getattr(model, "periodic", None))


def _check_row_order(model: Trajectory, what: str, name: str) -> None:
    """Refuse a topology whose rows are not its atoms in a fixed order: a
    LAMMPS dump without an id column, whose rows FACET puts in element order
    (``ids_track_atoms`` False)."""
    if what == "topology" and getattr(model, "ids_track_atoms", True) is False:
        raise ValueError(
            f"{name}: its topology {Path(model.source_path).name} has no atom "
            "ids (a LAMMPS dump without an id column), so its row k is not "
            "atom k of the XTC; give the LAMMPS data file the run read, or a "
            "dump with an id column")


def _per_atom_map(type_map, n: int, name: str) -> np.ndarray:
    """Elements from a map of atom number (1 .. n, file order) -> element."""
    if not isinstance(type_map, Mapping):
        raise ValueError(f"{name}: type_map needs a mapping of atom number "
                         "(1 to N, in file order) to element, such as "
                         "{1: 'Si', 2: 'O', 3: 'O'}")
    # Nothing is allocated in proportion to n before the map is known to
    # cover every atom: n comes from the file, whose header may be damaged.
    given: dict[int, str] = {}
    for key, value in type_map.items():
        if isinstance(key, (bool, np.bool_)) or not isinstance(
                key, (int, np.integer)):
            raise ValueError(
                f"{name}: an XTC file has no atom types, so type_map is keyed "
                f"by atom number (1 to {n}, in file order); {key!r} is not "
                "one. A map of LAMMPS types goes with topology= (the data file "
                "that holds the types)")
        if not 1 <= int(key) <= n:
            raise ValueError(f"{name}: type_map names atom {int(key)}; the "
                             f"frames hold atoms 1 to {n}")
        if not isinstance(value, str):
            raise ValueError(f"{name}: type_map entry {key!r}: {value!r} is "
                             "not an element symbol")
        given[int(key)] = validate_symbol(value)
    if len(given) < n:
        first: list[int] = []
        number = 1
        while len(first) < 5 and number <= n:   # ends within len(given) + 5
            if number not in given:
                first.append(number)
            number += 1
        raise ValueError(
            f"{name}: an XTC file has no atom types, so type_map is keyed by "
            f"atom number (1 to {n}, in file order) and needs every atom; "
            f"{n - len(given)} of {n} have no entry (first: {first}). With a "
            "LAMMPS data file or another model of the same atoms, pass "
            "topology= instead")
    return np.array([given[number] for number in range(1, n + 1)])


def _explicit_box(box_from, name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """(box_ang, origin_ang, note) from box_from: a Frame or Trajectory (its
    frame 0), or what :func:`.md_readers.box_from_source` takes (the path of
    an MD file whose box is read, or (3, 3) rows in Å)."""
    if isinstance(box_from, (Frame, Trajectory)):
        frame, _, _, label, _ = _model_frame(box_from, "box_from", name, {})
        return (frame.box_ang, frame.origin_ang,
                f"box and origin from frame 0 of {label}")
    if isinstance(box_from, (str, os.PathLike)) and Path(box_from).is_file():
        with open(box_from, "rb") as handle:
            head = handle.read(_HEAD_BYTES)
        if sniff_xtc(head):
            # Another XTC file: its header holds the box, and no elements are
            # needed for it (md_readers.box_from_source would open the file
            # with read_xtc, which needs elements).
            box, origin, where = read_xtc_box(box_from)
            cell = cell_from_vectors(box)
            return box, origin, (
                f"box from {where}: a {cell.a:.6g}, b {cell.b:.6g}, c "
                f"{cell.c:.6g} Å, alpha {cell.alpha:.6g}, beta "
                f"{cell.beta:.6g}, gamma {cell.gamma:.6g} deg; {name} states "
                "no box in these frames, so this one box holds each of them, "
                "which assumes a constant volume (an NVT or NVE run); "
                f"{name} cannot show whether its volume changed")
    from . import md_readers                # lazy: md_readers lists this module

    return md_readers.box_from_source(box_from, for_name=name)


def read_xtc_box(path) -> tuple[np.ndarray, np.ndarray, str]:
    """(box_ang rows, origin_ang, where from) of an XTC file's first frame,
    read from its header alone, so no elements are needed.

    The box is the header's nm x :data:`ANG_PER_NM`; XTC holds no box origin,
    so the origin is (0, 0, 0). Refused, naming the file, when the file does
    not start with an XTC frame header, or when that frame's box is all zeros
    (GROMACS without periodic boundaries), not finite or degenerate.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such file")
    with open(path, "rb") as handle:
        head = handle.read(_HEAD_BYTES)
    if not sniff_xtc(head):
        raise UnsupportedFormat(
            f"{path.name}: does not start with an XTC frame header (the magic "
            f"number {XTC_MAGIC} or {XTC_NEW_MAGIC}, then the atom count, "
            "repeated at byte 52), so no box is read from it; give a file "
            "whose first frame states the box, or the box as (3, 3) rows in Å")
    (step,) = struct.unpack_from(">i", head, 8)
    box_nm = _floats(head, 16, 9).reshape(3, 3)
    if not np.isfinite(box_nm).all():
        raise ValueError(f"{path.name}: the box of its first frame holds a "
                         "value that is not a finite number, so no box is "
                         "read from it")
    if not box_nm.any():
        raise ValueError(f"{path.name}: the box of its first frame is all "
                         "zeros (GROMACS writes that for a run without "
                         "periodic boundaries), so it gives no box; give "
                         "another file, or the box as (3, 3) rows in Å")
    try:
        box = md_model._checked_box(box_nm * ANG_PER_NM)
    except ValueError as error:
        raise ValueError(f"{path.name}: the box of its first frame: "
                         f"{error}") from error
    return box, np.zeros(3), (
        f"the header of frame 0 of {path.resolve()} (gromacs-xtc, step "
        f"{step}; nm x {ANG_PER_NM:g}; the file holds no box origin, so the "
        "corner is at (0, 0, 0))")


def _times(records: list[_Record], timestep_fs, name: str
           ) -> tuple[np.ndarray, list[str], str]:
    """times_ps, notes, and the time part of units_note.

    A stored time that is not finite is unknown (NaN). With ``timestep_fs``
    the times are ``offset + step x dt``; the offset (a GROMACS start time)
    is taken from the frame whose stored time has the finest float32
    rounding, the smallest absolute time, where it is resolved best (a start
    time of 0.001 ps is below the rounding of a time near 40 000 ps), and
    every frame with a known time is checked against it.
    """
    file_times = np.array([r.time_ps for r in records], dtype=np.float64)
    file_times[~np.isfinite(file_times)] = np.nan       # inf: unknown too
    known = np.isfinite(file_times)
    steps = np.array([r.step for r in records], dtype=np.int64)
    with np.errstate(invalid="ignore"):
        ulp = np.spacing(np.abs(file_times).astype(np.float32)).astype(
            np.float64)
    notes: list[str] = []
    unknown = [r.position for r, k in zip(records, known) if not k]
    if unknown:
        notes.append(
            f"the stored time of {len(unknown)} frame(s) is not a finite "
            f"number (file position(s) {_positions_text(unknown)}); "
            + ("their time is unknown (NaN), and timestep_fs= gives them one "
               "from the step" if timestep_fs is None else
               "timestep_fs gives them one from the step"))
    if timestep_fs is None:
        text = ("times are the file's 32-bit floats, in ps")
        if known.any():
            worst = float(ulp[known].max())
            text += f" (one float step at the latest time is {worst:.3g} ps"
            if known.sum() > 1:
                spacing = np.diff(file_times[known])
                typical = float(np.median(np.abs(spacing)))
                if typical > 0:
                    text += f", {worst / typical:.2g} of the median spacing"
            text += "; timestep_fs= rebuilds them from the integer steps)"
        return file_times, notes, text
    if isinstance(timestep_fs, (bool, np.bool_)) or not isinstance(
            timestep_fs, (int, float, np.integer, np.floating)) \
            or not math.isfinite(timestep_fs) or timestep_fs <= 0:
        raise ValueError(f"{name}: timestep_fs needs a positive number of fs, "
                         f"not {timestep_fs!r}")
    dt_ps = float(timestep_fs) * _PS_PER_FS
    if not known.any():
        rebuilt = steps * dt_ps
        notes.append(
            f"times are step x timestep_fs={timestep_fs} ({dt_ps:g} ps per "
            "step) with no offset: no frame holds a finite time, so a start "
            "time (GROMACS tinit) cannot be read and nothing is checked "
            "against the file")
        return rebuilt, notes, (f"times in ps from the steps and timestep_fs="
                                f"{timestep_fs} fs")
    candidates = np.flatnonzero(known)
    finest = int(candidates[np.argmin(ulp[candidates])])
    offset = float(file_times[finest] - steps[finest] * dt_ps)
    if abs(offset) <= float(ulp[finest]):
        # Within the rounding of the finest stored time: no offset can be
        # told from zero (LAMMPS writes time = step x dt; GROMACS adds tinit).
        offset = 0.0
    rebuilt = offset + steps * dt_ps
    gap = np.abs(rebuilt - file_times)
    allowed = ulp + float(np.spacing(np.float32(abs(offset)))) + 1e-12
    with np.errstate(invalid="ignore"):
        bad = np.flatnonzero(known & (gap > allowed))
    where = (f"time - step x dt at file position {records[finest].position}, "
             "the frame whose 32-bit time has the finest rounding")
    if bad.size:
        k = int(bad[0])
        raise ValueError(
            f"{name}: timestep_fs={timestep_fs} with the offset {offset:.9g} "
            f"ps ({where}) gives {rebuilt[k]:.9g} ps at step {steps[k]} (file "
            f"position {records[k].position}), where the file holds "
            f"{file_times[k]:.9g} ps; they differ by {gap[k]:.3g} ps, more "
            f"than the file's 32-bit rounding there ({allowed[k]:.3g} ps), in "
            f"{bad.size} of {int(known.sum())} frames with a stored time. No "
            f"one start time plus step x {dt_ps:g} ps holds every frame: the "
            "time step of the run differs from the one given, or it changed "
            "during the run; without timestep_fs the file's times are used")
    shift = (f"plus the offset {offset:.9g} ps ({where})" if offset else
             f"with no offset ({where}, lies within that rounding)")
    note = (f"times rebuilt from the integer steps with timestep_fs="
            f"{timestep_fs} ({dt_ps:g} ps per step), {shift}; every frame "
            f"with a stored time agrees with it within its 32-bit rounding "
            f"(largest difference {float(np.nanmax(gap)):.3g} ps)")
    notes.append(note)
    return rebuilt, notes, (f"times in ps from the steps and timestep_fs="
                            f"{timestep_fs} fs")


def _positions_text(positions: list[int], shown: int = 10) -> str:
    """File positions as text, the first ``shown`` of them and a count."""
    text = ", ".join(str(p) for p in positions[:shown])
    if len(positions) > shown:
        text += f" and {len(positions) - shown} more"
    return text


# ---------------------------------------------------------------------------
# the trajectory
# ---------------------------------------------------------------------------

def _check_wrap(cart_ang: np.ndarray, box_ang: np.ndarray,
                precision: float | None) -> None:
    """Raise ValueError when positions lie so far from the box that wrapping
    them into it keeps no digit (:data:`_WRAP_LIMIT` box lengths): every atom
    would land on one point. Non-finite positions are left to
    frame_from_arrays, which refuses them by name."""
    if not np.isfinite(cart_ang).all():
        return
    frac = np.linalg.solve(box_ang.T, cart_ang.T)
    far = float(np.abs(frac).max())
    if far < _WRAP_LIMIT:
        return
    stored = (f"its precision is {precision:g}, a grid step of "
              f"{1.0 / precision:.3g} nm" if precision is not None
              else "it stores plain 32-bit floats")
    raise ValueError(
        f"the positions decode to {far:.3g} box lengths from the box ({stored}"
        f"); beyond 2^52 box lengths a float64 keeps no digit of the position "
        "within the box, so every atom would wrap onto one point, and no "
        "position is returned for this frame. The units note lists the "
        "precision every frame states")


class _XtcTrajectory(Trajectory):
    """Frames decoded on demand from byte ranges of one XTC file."""

    ids_track_atoms = True

    def __init__(self, path: Path, records: list[_Record], *, elements,
                 atom_id, boxes, origins, charge, periodic=None,
                 **kwargs) -> None:
        super().__init__(source_path=path, **kwargs)
        self._path = Path(path)
        self._records = records
        self._elements = elements
        self._atom_id = atom_id
        self._boxes = boxes
        self._origins = origins
        self._charge = charge
        # (x, y, z) when a topology's file states the boundary, else None: the
        # XTC file does not record it (the md_readers convention).
        self.periodic = periodic

    def _load(self, k: int) -> Frame:
        record = self._records[k]
        where = (f"{self._path.name}: frame {k} (file position "
                 f"{record.position}, byte {record.offset})")
        with open(self._path, "rb") as handle:
            handle.seek(record.offset)
            data = handle.read(record.end - record.offset)
        if len(data) != record.end - record.offset:
            raise FrameError(f"{where}: the file holds {len(data)} of the "
                             f"{record.end - record.offset} bytes indexed for "
                             "this frame; it changed after it was indexed")
        try:
            decoded = decode_xtc_frame(data)
            cart = decoded.coords_nm * ANG_PER_NM
            box = md_model._checked_box(self._boxes[k])
            _check_wrap(cart - self._origins[k], box, decoded.precision)
            time_ps = float(self.times_ps[k])
            return frame_from_arrays(
                self._elements, cart, box_ang=box, origin_ang=self._origins[k],
                atom_id=self._atom_id, timestep=record.step,
                time_ps=time_ps if math.isfinite(time_ps) else None,
                charge_e=self._charge)
        except FrameError:
            raise
        except ValueError as error:
            raise FrameError(f"{where}: {error}") from error

    def close(self) -> None:
        """Nothing stays open: each frame opens the file for its own read."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _check_start(path: Path, head: bytes, size: int) -> None:
    """For a file whose first bytes are not an XTC frame header: refuse an
    empty file, one shorter than a header, and one that neither carries the
    .xtc name nor starts with the magic number; let any other through to the
    index, which passes over a damaged first header to the next one."""
    name = path.name
    if size == 0:
        raise UnsupportedFormat(
            f"{name}: the file is empty (0 bytes), so it holds no XTC frame (a "
            "run that had not written a frame yet, or a copy that did not "
            "complete); give the file once the writer has written frames to it")
    starts = len(head) >= 4 and struct.unpack_from(">i", head, 0)[0] in _MAGICS
    if not (starts or path.suffix.lower() == ".xtc"):
        raise UnsupportedFormat(
            f"{name}: does not start with an XTC frame header (the magic "
            f"number {XTC_MAGIC} or {XTC_NEW_MAGIC}, then the atom count, "
            "repeated at byte 52); a file in another format opens with "
            "md_readers.read_trajectory, which recognises the formats FACET "
            "reads by their content")
    if size < _HEAD_BYTES:
        raise UnsupportedFormat(
            f"{name}: the file holds {size} bytes, fewer than one "
            f"{_HEAD_BYTES}-byte XTC frame header, so it holds no frame (a copy "
            "taken as the writer started, or a cut file); give the whole file")


def read_xtc(path, *, topology=None, type_map: Mapping | None = None,
             mass_tol_amu: float | None = None, box_from=None,
             timestep_fs: float | None = None) -> Trajectory:
    """Open a GROMACS / LAMMPS XTC trajectory.

    An XTC frame holds positions, the box, the step and the time; elements
    come from ``topology`` (a file FACET reads -- LAMMPS data or dump,
    extended XYZ, ... -- or a Trajectory or Frame, whose frame 0 lists the same
    atoms in the same order; ``type_map`` and ``mass_tol_amu`` are passed on
    to its reader) or, without one, from ``type_map`` keyed by atom number 1
    .. N in file order. ``box_from`` gives a box to frames whose box is all
    zeros; ``timestep_fs`` rebuilds the times from the integer steps.
    ``periodic`` is the topology's where its file states the boundary, else
    None (the XTC file does not record it). A file named .xtc, or starting
    with the magic number, whose first header is damaged is indexed like any
    other: the damaged bytes are one position in ``skipped`` and the frames
    after them read.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such file")
    name = path.name
    size = path.stat().st_size
    with open(path, "rb") as handle:
        head = handle.read(_HEAD_BYTES)
    if head[:2] == _GZIP_MAGIC:
        raise UnsupportedFormat(
            f"{name}: a gzip-compressed file; FACET reads XTC files as "
            "written (XTC is already compressed), so decompress it first "
            "(gunzip, or 7-Zip on Windows) and open the result")
    if not sniff_xtc(head):
        _check_start(path, head, size)

    records, skipped = _index(path)
    if not records:
        reasons = "; ".join(f"position {p}: {r}" for p, r in sorted(skipped.items()))
        raise UnsupportedFormat(
            f"{name} holds no readable XTC frame ({reasons}). An XTC file is a "
            f"series of frames, each opening with the magic number {XTC_MAGIC} "
            f"(or {XTC_NEW_MAGIC}) and the atom count, repeated at byte 52; "
            "a file in another format opens with md_readers.read_trajectory, "
            "which recognises the formats FACET reads by their content")
    counts = [r.natoms for r in records]
    tally = Counter(counts)
    n_atoms = next(c for c in counts if tally[c] == max(tally.values()))
    kept = []
    for record in records:
        if record.natoms == n_atoms:
            kept.append(record)
        else:
            skipped[record.position] = (
                f"{record.natoms} atoms, where {tally[n_atoms]} of the "
                f"{len(records)} frames hold {n_atoms}")
    records = kept

    notes: list[str] = []
    charge = charges_e = None
    periodic = frame0 = label = None
    forwarded = {k: v for k, v in (("type_map", type_map),
                                   ("mass_tol_amu", mass_tol_amu))
                 if v is not None}
    if topology is not None:
        frame0, source, used_map, label, periodic = _model_frame(
            topology, "topology", name, forwarded)
        if frame0.n_atoms != n_atoms:
            raise ValueError(
                f"{name}: the XTC frames hold {n_atoms} atoms and {label} "
                f"holds {frame0.n_atoms}; the topology needs the same atoms in "
                "the same order (LAMMPS dump xtc of a group, or GROMACS "
                "compressed-x-grps, writes only those atoms, so the topology "
                "has to hold just them)")
        elements, atom_id = frame0.elements, frame0.atom_id
        charge = frame0.charge_e
        type_map_source = source
        notes.append(
            f"elements and atom ids from {label} (frame 0, {frame0.n_atoms} "
            f"atoms, elements from {source}): row k of every XTC frame is "
            "taken as row k of that frame, sorted by atom id (LAMMPS writes "
            "dump xtc rows sorted by id; GROMACS writes topology order)")
        if charge is not None:
            charges_e, charge_notes = charges_per_element(elements, charge)
            notes.append(f"per-atom charges from {label}, the same in every "
                         "frame (an XTC file holds positions only)")
            notes.extend(charge_notes)
    elif mass_tol_amu is not None:
        raise ValueError(f"{name}: mass_tol_amu applies to the masses of a "
                         "topology file, and none was given")
    elif type_map is not None:
        elements = _per_atom_map(type_map, n_atoms, name)
        atom_id = np.arange(1, n_atoms + 1, dtype=np.int64)
        used_map, type_map_source = {}, "user"
        notes.append("elements from type_map, by atom number in file order; "
                     "atom ids are those numbers (1 to N)")
    else:
        raise ValueError(
            f"{name}: an XTC file holds positions, the box, the step and the "
            "time, but no elements, atom types or ids. Pass topology= a file "
            "FACET reads that holds the same atoms in the same order (the "
            "LAMMPS data file the run started from, a dump, an extended XYZ), "
            "or type_map={1: '<element>', 2: '<element>', ...} giving the "
            f"element of every atom by its number in the file (1 to "
            f"{n_atoms})")

    zero = [r for r in records if not r.box_nm.any()]
    if box_from is not None and not zero:
        raise ValueError(f"{name}: box_from applies to frames without a box; "
                         "every frame of the file states its box")
    boxes, origins = [], []
    if zero:
        if box_from is None:
            if len(zero) == len(records):
                raise ValueError(
                    f"{name}: every frame's box is all zeros (GROMACS writes "
                    "that for a run without periodic boundaries); FACET "
                    "measures periodic models, so pass box_from= a file FACET "
                    "reads, a Frame, or three cell vectors as rows in Å")
            for record in zero:
                skipped[record.position] = (
                    "its box is all zeros (no periodic cell); box_from= "
                    "supplies one")
            records = [r for r in records if r.box_nm.any()]
            explicit = None
        else:
            explicit = _explicit_box(box_from, name)
            notes.append(f"{len(zero)} of {len(records)} frames have a box of "
                         f"zeros (no periodic cell in the file) and take the "
                         f"box_from box: {explicit[2]}")
    else:
        explicit = None
    for record in records:
        if not record.box_nm.any() and explicit is not None:
            boxes.append(explicit[0])
            origins.append(explicit[1])
        else:
            boxes.append(record.box_nm * ANG_PER_NM)
            origins.append(np.zeros(3))

    times_ps, time_notes, time_text = _times(records, timestep_fs, name)
    notes.extend(time_notes)
    precisions = sorted({r.precision for r in records if r.precision})
    grid = ("; positions on a grid of " + ", ".join(
        f"1/{p:g} nm ({ANG_PER_NM / p:g} Å)" for p in precisions)
        + " (the file's precision), rounded half away from zero by the writer"
        if precisions else "")
    plain = sum(1 for r in records if r.precision is None)
    if plain:
        grid += (f"; {plain} frame(s) of {_UNCOMPRESSED_MAX_ATOMS} atoms or "
                 "fewer store plain 32-bit floats")
    units_note = (
        f"XTC lengths in nm and times in ps (the GROMACS convention; LAMMPS "
        f"dump xtc converts to them except under units lj, which it writes "
        f"unscaled, and except where dump_modify sfactor or tfactor replaced "
        f"its factors; the file records neither, and such a file reads as nm "
        f"and ps here); positions and box x {ANG_PER_NM:g} to Å{grid}; "
        f"{time_text}")
    precision_note = _precision_note(records)
    if precision_note:
        notes.append(precision_note)
    notes.append(_boundary_note(periodic, label))
    notes.append("the file stores no box origin: the box corner is placed at "
                 "(0, 0, 0) Å and positions are wrapped into that box, so "
                 "they match the writer's (LAMMPS: corner at xlo, ylo, zlo) "
                 "modulo the box vectors")
    notes.append("the file does not say whether its positions are continuous "
                 "(LAMMPS writes wrapped positions unless dump_modify unwrap "
                 "yes; GROMACS writes what mdrun holds), so no unwrapped "
                 "positions are kept; it holds no velocities and no charges "
                 "of its own")
    if any(r.step < 0 for r in records):
        notes.append("some frames have a negative step: GROMACS stores only "
                     "the low 32 bits of the step, so a step above "
                     "2 147 483 647 wraps")
    magic_new = sum(1 for r in records if r.magic == XTC_NEW_MAGIC)
    if magic_new:
        notes.append(f"{magic_new} frame(s) use GROMACS's large-system header "
                     f"(magic {XTC_NEW_MAGIC}, 64-bit byte count)")

    box_varies = any(not np.array_equal(b, boxes[0]) for b in boxes[1:]) \
        if boxes else None
    trajectory = _XtcTrajectory(
        path, records, elements=elements, atom_id=atom_id, boxes=boxes,
        origins=origins, charge=charge, periodic=periodic,
        charges_e=charges_e, file_format=FORMAT_NAME,
        n_atoms=n_atoms, n_frames=len(records),
        timesteps=np.array([r.step for r in records], dtype=np.int64),
        times_ps=times_ps, type_map=used_map, type_map_source=type_map_source,
        skipped=skipped, notes=notes, units_note=units_note,
        box_varies=box_varies)
    if frame0 is not None:
        # Right after the topology note (notes[0] until _finish puts the
        # load record before it).
        trajectory.notes.insert(
            1, _topology_agreement(trajectory, records, frame0, label))
    return _finish(trajectory)


def _precision_note(records: list[_Record]) -> str | None:
    """A note naming which file positions carry which precision, when the
    frames carry more than one."""
    by_precision: dict[float, list[int]] = {}
    for record in records:
        if record.precision is not None:
            by_precision.setdefault(record.precision, []).append(
                record.position)
    if len(by_precision) < 2:
        return None
    parts = [f"{p:g} at file position(s) {_positions_text(positions)}"
             for p, positions in sorted(by_precision.items(),
                                        key=lambda item: -len(item[1]))]
    return ("the frames state more than one precision: " + "; ".join(parts)
            + ". A writer keeps one precision through a run (LAMMPS dump_modify "
            "precision, a power of 10 from 10 to 1e6; GROMACS "
            "compressed-x-precision), so frames of another precision come from "
            "another run written to the same file, or hold a changed "
            "precision word (the file holds no checksum to tell)")


def _boundary_note(periodic, label: str | None) -> str:
    """What the file does not say about periodicity, and where FACET's
    ``periodic`` comes from."""
    slab = ("; a slab (LAMMPS boundary p p f, GROMACS pbc = xy) then counts "
            "its vacuum in the box volume that densities and g(r) "
            "normalisation use")
    if periodic is None:
        return ("the file does not record the boundary (LAMMPS: the boundary "
                "command; GROMACS: pbc in the .mdp file); the box is taken as "
                "periodic along a, b and c" + slab)
    text = (f"the file does not record the boundary; {label} states periodic "
            f"{tuple(bool(p) for p in periodic)} (x, y, z), taken here as the "
            "XTC run's")
    if all(periodic):
        return text
    return (text + ": not periodic along every axis, while FACET measures the "
            "box as periodic along a, b and c, so atoms near a non-periodic "
            "face see images across it" + slab)


def _topology_agreement(trajectory: Trajectory, records: list[_Record],
                        frame0: Frame, label: str) -> str:
    """A note comparing the topology's frame 0 with the XTC frame of the same
    step, row by row modulo the XTC box: row k is the same atom in both files
    only while they agree within the two files' rounding. When no XTC frame
    shares the topology's step, the note says that rows are matched by the
    atom count alone."""
    step = frame0.timestep
    k = next((i for i, r in enumerate(records) if r.step == step), None) \
        if step is not None else None
    if k is None:
        stated = f"is at step {step}" if step is not None else \
            "states no step"
        return (f"frame 0 of {label} {stated}, which no XTC frame shares, so "
                "its positions are not compared with the XTC frames: rows are "
                "matched by the atom count alone")
    where = f"the XTC frame at file position {records[k].position}"
    try:
        frame = trajectory.frame(k)
    except FrameError as error:
        return (f"frame 0 of {label} and {where} share step {step}, and that "
                f"frame cannot be decoded ({error}), so the positions are not "
                "compared: rows are matched by the atom count alone")
    diff = frame.cart_ang - frame0.cart_ang
    frac = np.linalg.solve(frame.box_ang.T, diff.T).T
    frac -= np.round(frac)
    per_axis = np.abs(frac @ frame.box_ang)
    largest = float(per_axis.max())
    precision = records[k].precision
    if precision is not None:
        half = 0.5 / precision * ANG_PER_NM
        grid = f"the XTC grid's half step is {half:.3g} Å"
    else:
        half = 0.0
        grid = "the XTC frame stores plain 32-bit floats"
    # float32 rounding of the written value and of value x precision.
    scale = max(float(np.abs(frame.cart_ang).max()),
                float(np.abs(frame0.cart_ang).max()))
    beyond = np.flatnonzero(per_axis.max(axis=1) > half + 2 * scale * 2.0 ** -23)
    box_diff = float(np.abs(frame.box_ang - frame0.box_ang).max())
    text = (f"frame 0 of {label} and {where} share step {step}: row by row, "
            "modulo the "
            f"XTC box, their positions differ by up to {largest:.3g} Å per "
            f"axis ({grid}), and their boxes by up to {box_diff:.3g} Å")
    if beyond.size:
        ids = frame.atom_id[beyond[:5]].tolist()
        text += (f"; {beyond.size} of {frame.n_atoms} rows differ by more than "
                 f"that half step and the float32 rounding (first atom ids "
                 f"{ids})")
    return (text + ". Row k is the same atom in both files only while the "
            "difference stays within the two files' rounding (the XTC grid, "
            "and the digits the topology file prints)")


def _finish(trajectory: Trajectory) -> Trajectory:
    """The load record first in the notes and the timestep-order note last,
    as :mod:`.md_readers` does; frame 0 is decoded here, and its FrameError
    names the file."""
    from . import md_readers                # lazy: md_readers lists this module

    frame0 = trajectory.frame(0)
    trajectory.notes.insert(0, md_readers._load_summary(trajectory, frame0))
    order = md_readers._timestep_order_note(trajectory.timesteps,
                                            trajectory.file_positions)
    if order:
        trajectory.notes.append(order)
    return trajectory


FORMATS = (
    FormatSpec(
        name=FORMAT_NAME,
        description="GROMACS XTC compressed trajectory (also LAMMPS dump xtc)",
        extensions=(".xtc",),
        stems=(),
        sniff=sniff_xtc,
        read=read_xtc,
        options=frozenset({"topology", "type_map", "mass_tol_amu",
                           "box_from", "timestep_fs"}),
        binary=True,
    ),
)
