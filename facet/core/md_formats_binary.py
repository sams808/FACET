"""Binary MD trajectories: ASE .traj, HOOMD GSD and AMBER NetCDF.

WHY A MODULE OF ITS OWN
-----------------------
Three binary formats reach an oxide-glass group from the tools around the MD
engine rather than from the engine itself: ASE writes ``.traj`` by default,
and MD driven through ASE with a machine-learned potential (MACE, NequIP)
leaves one behind; OVITO exports all three (ase/traj, gsd/hoomd,
netcdf/amber); LAMMPS writes AMBER NetCDF through ``dump netcdf``. Each one is
read here with numpy, ``struct`` and ``json`` only: the libraries their
authors provide (ase, gsd, netCDF4) are not FACET dependencies, and every
layout below is read from the format's own specification or the writer's
source, cited in REFERENCES. The module keeps the contract of
:mod:`.md_formats_base`: frames are built with
:func:`.md_model.frame_from_arrays` and read on demand, a frame that cannot be
read is counted in ``Trajectory.skipped`` with its reason, and a file that
cannot be read is refused with a message naming the file and saying what to
do.

WHAT IS READ
------------
=============  =================================  ===============================
format         what a frame can carry             source of the layout
=============  =================================  ===============================
ASE .traj      atomic numbers, positions, cell,   ase/io/ulm.py and
(ULM)          pbc, momenta (with the masses      ase/io/trajectory.py [1, 2]
               ASE stores, or gemmi's weights),
               initial charges; OVITO's info
               keys Timestep and cell_origin
GSD (schema    box, positions, type names and     GSD file layer [3, 4], HOOMD
hoomd)         type ids, image flags, velocities, schema [5]
               charges, masses (to check the
               type names), step
AMBER NetCDF   coordinates (also unwrapped or     netCDF classic, 64-bit offset
(CDF-1, CDF-2, scaled), cell lengths and angles,  and CDF-5 grammars [6, 7];
CDF-5)         cell_origin, time, velocities,     AMBER convention [8]; LAMMPS
               ids, numeric types, atomic         dump netcdf [9]
               numbers, element names, charges
=============  =================================  ===============================

Forces, energies, stresses, constraints, tags, magnetic moments, logged
quantities and every other array are named in the notes as not read.

ASE TRAJECTORY (ULM)
--------------------
``ase/io/ulm.py`` [1] lays the file out as: 8 bytes of magic (``- of Ulm``;
``AFFormat`` before ULM version 3), a 16-byte tag (``ASE-Trajectory``), then
int64 version, item count and the position of the table of item offsets. The
writer byte-swaps these on a big-endian machine, so they are always
little-endian. Each item is an int64 length followed by a JSON dictionary;
``name.`` keys hold ``{"ndarray": [shape, dtype, offset]}`` references to raw
arrays elsewhere in the file, in the writing machine's byte order
(``_little_endian: false`` in the JSON says big-endian). One item is one frame.

``ase/io/trajectory.py`` [2] writes the atomic numbers, pbc, constraints and
masses (only when masses were set on the Atoms) in the first item, and again
in every item once they change; an item without ``numbers`` takes them from
item 0, as ASE's reader does. Positions are absolute Cartesian Å; ASE has no
box origin, so the origin is zero, except where OVITO's ase/traj export
stores ``info['cell_origin']`` (and ``info['Timestep']``), which are read.
ASE's MD integrators do not wrap atoms into the cell, but the file does not
say whether the positions were wrapped, so continuous positions are not
claimed. Atomic number 0 (ASE's placeholder 'X') names no element and is
refused.

ASE stores momenta, mass x velocity in ASE units. ASE's unit of time is
derived from Å, eV and amu: ``ase/units.py`` [10] sets ``second = 1e10 *
sqrt(e / amu)`` with CODATA 2014 values (``__codata_version__ = '2014'``,
e = 1.6021766208e-19 C, amu = 1.660539040e-27 kg; read from the ASE source by
WebFetch and checked against ASE 3.29's ``units.second``). One ASE unit of
velocity is therefore ``second * 1e-12`` Å/ps = 98.2269478846 Å/ps
(:data:`ASE_VELOCITY_TO_ANG_PER_PS`; scipy's CODATA 2022 constants give a
value 4.6e-9 smaller). Velocities are momenta / masses x that factor. When the
file stores no masses, ASE divided by its own table (``ase.data.
atomic_masses``, IUPAC 2016 in ASE 3.29), which FACET does not carry: the
momenta are divided by gemmi's standard atomic weights instead, and the note
says so. Measured against ASE 3.29's table, gemmi's weights differ by up to
1.6e-4 relative for the 84 elements gemmi gives a standard atomic weight
(S 1.56e-4, Li 1.44e-4, Se 1.39e-4, Ge 1.38e-4, B 9.3e-5, Cl 8.5e-5, H 6.0e-5,
O 2.5e-5, C 2.5e-5, Si 1.8e-5; Na, Al, P below 4e-8; 43 of the 84, Mg, K, Ca,
Ti, Fe, Zn, Sr, Zr, Ba and Pb among them, equal), and by up to 7.8e-3 for
elements gemmi lists at a whole mass number; velocities carry the same
relative difference. When only some frames' headers store masses (ASE writes
them from the frame on which they were set), the note names the frames of
each kind.

GSD
---
The file layer [3, 4]: a 256-byte header (uint64 magic 0x65DF65DF65DF65DF,
index location and allocated entries, name-list location and allocated
entries, uint32 schema and file-layer versions, 64-byte application and schema
names); an index of 32-byte entries (uint64 frame, uint64 N, int64 location,
uint32 M, uint16 name id, uint8 type, uint8 flags), whose first entry at
location 0 ends it; a name list of 64-byte names (version 1.x) or names
separated by zero bytes (2.x), ``64 x allocated entries`` bytes long in both;
chunks of N x M values in C order. Type ids follow ``enum gsd_type`` in
gsd.h: 1-4 uint8-64, 5-8 int8-64, 9 float, 10 double, 11 character. The file
is read little-endian; a header whose magic reads only big-endian is refused.
An entry the gsd library would refuse (unknown type, non-zero flags, a name id
outside the list, data past the end of the file) puts its frame in
``skipped``; an entry whose frame number is not below the index's entry count
makes the gsd library refuse the whole file as corrupt (gsd.c,
gsd_is_entry_valid), and FACET refuses it too, so that a damaged frame
number cannot make it count millions of frames. A frame number the index
lists no chunk for (the gsd library writes none for a frame equal to frame 0
in every value, step included; a damaged frame number leaves such gaps too)
is skipped with that reason rather than read as a copy of frame 0.

The HOOMD schema [5]: when a chunk is absent from frame i, frame 0's chunk of
that name applies (when the particle count is equal), and a chunk absent from
both has the schema's default; the gsd library omits a chunk equal to its
default (the image flags of an unwrapped frame 0, measured), so image flags,
velocities and charges that other frames hold are taken as zero there, with
a note naming the frames. The library also omits a chunk equal to frame 0's
(measured: a frame whose positions and box equal frame 0's holds only
``configuration/step``), so a frame without ``particles/position`` or
``configuration/box`` repeats frame 0's, and a note names those frames. The
gsd library rounds ``configuration/box`` to float32 whatever the precision
of the file (gsd/hoomd.py, ConfigurationData: ``numpy.ascontiguousarray(
self.box, dtype=numpy.float32)``), so a double-precision file's box carries
float32 rounding (3.8e-7 Å at 15 Å, measured). ``configuration/box``
is (Lx, Ly, Lz, xy, xz, yz); the schema's unwrapping formula fixes the box
rows as a = (Lx, 0, 0), b = (xy Ly, Ly, 0), c = (xz Lz, yz Lz, Lz)
(:func:`box_from_hoomd`), and its position bounds centre the box on zero, so
the origin is -(a + b + c) / 2. ``particles/image`` gives continuous positions
x + i_x a + i_y b + i_z c. GSD stores no particle ids: atoms are numbered
0 .. N-1 in file order (HOOMD-blue writes particles in tag order).

**OVITO's GSD exporter writes another box.** OVITO's ``GSDExporter.cpp`` [11]
builds HOOMD's box matrix from the cell, then writes ``(Lx, |b|, |c|,
b_x / |b|, c_x / |c|, c_y / |c|)``: for a tilted box, the lengths of b and c
where the schema puts its diagonal. Read with the schema, such a box differs
by up to 0.18 Å in a box vector in the recognition corpus (OVITO 3.16.1;
OVITO's own GSD importer reads it back with the schema, so a round trip
through OVITO changes the box). For a file whose application is OVITO and
whose box is tilted, the box is read as the exporter writes it, and the
positions are checked against both readings: the exporter wraps every atom
into its box, so atoms it did not shift (image flags 0) all lie inside the
box it meant. When they all lie inside the schema's box and some outside
OVITO's, the schema's reading is used instead; when they lie inside both
(the boxes differ only in a layer about 0.01 Å thick at the faces, so a
270-atom model gives no such atom, nor did a 1,800-atom one whose two readings
differ by 0.52 Å, measured), the note says the positions do not decide, that
the file records no OVITO version, and by how much the readings differ.

The same source wraps coordinate d alone by its own cell vector
(``transformation.prodrow(p - s * cell->matrix().column(dim), dim)``) and
writes s in ``particles/image``: in a tilted box an atom wrapped along c is
moved by (0, 0, c_z), not by c. Adding s times the box's diagonal back
restores the position OVITO held, exactly (:func:`_undo_ovito_shifts`);
FACET then wraps it by whole box vectors. Those image flags are OVITO's
wrapping shifts, not the run's (measured on the corpus file: 0-7 non-zero
rows per frame where the run held 88-99), so no continuous positions are set
for an OVITO file. OVITO 3.16.1 writes the particles sorted by identifier
(measured).

GSD names no unit: HOOMD's native units are whatever the simulation used.
Lengths are read as Å (OVITO writes the units of the data it was given, Å for
LAMMPS metal or real, VASP and extended XYZ input). Velocities and charges
are stored as written unless ``units`` states the LAMMPS style they came in,
as for a LAMMPS dump without a unit style: 'metal' or 'real', or 'nano'
(LAMMPS: distance in nanometers, velocity in nanometers/nanosecond, charge in
multiples of the electron charge [14]), whose lengths are converted with
scipy.constants (nano / angstrom = 10) and velocities by 10 / 1000 to Å/ps.
mdtraj's GSD writer keeps mdtraj's nm [15]. A frame 0 holding more than 0.5
atoms per Å^3 is refused with that option named: no substance holds as many
(diamond, which holds the most atoms per unit volume, has 3520 kg/m^3 [16],
0.176 atoms/Å^3 with 12.011 g/mol), while nm read as Å multiply a number
density by 1000 and reduced units hold about one particle per sigma^3.

mdtraj's GSD writer also writes the tilts as the lengths b_x, c_x, c_y
(``lengths_and_angles_to_tilt_factors``) where the schema puts b_x/Ly,
c_x/Lz, c_y/Lz, and the positions as it was given them, and its application
name is the gsd library's own, so such a file cannot be told from HOOMD-blue's
by its header. For a tilted box whose frame 0 holds positions outside the
schema's centred box (HOOMD-blue keeps every position inside), the notes say
the writer does not follow the schema in full and name mdtraj's tilts; the
box is read with the schema's definition.

Type names are labels: a name is an element only when written as files write
symbols. A ``particles/mass`` chunk checks such a name (``mass_tol_amu``),
and gives a name that is not a symbol ('Si_t') the element whose standard
atomic weight alone lies within the tolerance of its mass, as
:func:`.md_readers.type_map_from_masses` does (``type_map_source`` 'data-file
masses'). Without ``particles/mass`` (the gsd library omits it when every
mass is HOOMD's default, 1) nothing tells a letter used as a label from a
symbol: when any name in use is not a symbol (HOOMD's 'A'), every name is
taken as a label and the type map needs them all, so that 'B' beside 'A' is
not read as boron; when every name is a symbol, the note says no mass checked
them.

AMBER NETCDF
------------
The classic grammar [6] (CDF-1, 32-bit offsets; CDF-2, 64-bit offsets) and
CDF-5 [7] (64-bit sizes and the unsigned and 64-bit integer types) are
parsed here: big-endian throughout, names and attribute values padded to 4
bytes, non-record variables at their ``begin`` offsets, record variables
interleaved record by record. The record size is the sum of the record
variables' sizes each padded to 4 bytes, except that a file with exactly one
record variable packs it unpadded (netCDF-C's ``NC_computeshapes``; the
classic specification states the same for byte, char and short data). A
record count of all ones bytes (streaming) is replaced by the complete records
the file holds. NetCDF-4 files are HDF5 underneath and are refused with the
conversion that makes them classic (``nccopy -k cdf5``) [12].
``scipy.io.netcdf_file`` reads CDF-1 and CDF-2 but not CDF-5, which OVITO
writes; the tests use it as a check on this parser.

The AMBER convention [8]: ``coordinates`` (frame, atom, spatial) in Å,
``cell_lengths`` in Å and ``cell_angles`` in degrees per frame, ``time`` in
ps, ``velocities`` in Å/ps, and "If a scale_factor attribute exists for a
variable, readers shall multiply data values by the value of the scale_factor
attribute" (AMBER's own velocities carry 20.455). A ``units`` attribute that
names another unit (nm, fs) is converted with scipy.constants. The
convention gives the cell as lengths and angles, not vectors, and states only
that for right angles a, b and c lie along x, y and z; the box is built with a
along x and b in the xy plane (:func:`box_from_cell_parameters`), LAMMPS's
restricted orientation, as ASE's and OVITO's readers build it.
``cell_origin`` (LAMMPS, OVITO, ASE) gives the origin. A restart
(``AMBERRESTART``) holds one frame and declares "double time units =
'picosecond'" with no dimension; ParmEd and mdtraj write it with a dimension
``time`` of length 1 (measured), and both are read as the frame's time. A
``scale_factor`` of zero, or one that is not a finite number, on a variable
that is read refuses the file, naming the variable.

Writers depart from the convention, each read as its source or output shows:

* LAMMPS ``dump netcdf`` [9] (``program`` 'LAMMPS') writes the step number in
  ``time`` (``time = update->ntimestep``, line 660) and the step length as
  its float ``scale_factor`` (``(float) update->dt``, line 473), in the run's
  time unit. ``time`` has the dump's real type: NC_FLOAT (line 195) unless
  ``dump_modify double yes`` makes it NC_DOUBLE (line 861). A LAMMPS file's
  whole-number times are read as the steps, and step x scale_factor as the
  times; float32 holds every whole number only up to 2**24 = 16777216, so a
  larger step is stored rounded, and the note says so. Ids and types are in
  ``id`` and ``type`` variables.
* OVITO 3.16.1 writes ``Timestep``, ``identifier``, ``atom_types`` (numeric
  types, no names) and ``Charge``, and fills ``time`` with its animation frame
  number (0, 1, 2, ...; measured on the corpus and test files), which is not
  read as a time. Its ``cell_angles`` are the angles of the cell with the xy
  and yz tilts exchanged, b' = (yz, Ly, 0) and c' = (xz, xy, Lz), while
  ``cell_lengths`` are the cell's own: measured on six cells with one or more
  tilts (the generator of the test files records them, and the written angles
  match the exchanged cell to the printed digits) and on the corpus model;
  OVITO's own importer reads such a file back into the exchanged cell. The
  exporter's source is not public, so the cell is not reconstructed from
  this: an OVITO file whose alpha or gamma is not 90 degrees is refused with
  the export that works, and one tilted in xz only (alpha = gamma = 90,
  written as the cell's own) is read.
* ASE's NetCDFTrajectory [13] writes atomic numbers in ``atom_types``, and
  momenta / masses (ASE's unit of velocity) in ``velocities`` without a units
  attribute, while setting ``units = 'Angstrom/Femtosecond'`` on
  ``coordinates``; a file whose ``program`` is ASE is read that way. It
  stores the coordinates in float32 even when asked for double (2.1e-6 Å,
  measured), and writes the cell as lengths and angles with the positions
  as they are, so a cell whose a does not lie along x (or b outside the xy
  plane) is read 10-12 Å off by ASE's own reader and by FACET alike
  (measured on ASE 3.29); the same model as ``.traj`` reads to 5e-15 Å.
* MDAnalysis's NCDFWriter and AMBER itself name no atoms (AMBER keeps them in
  the topology file): ``topology`` gives the elements, as another MD file
  FACET reads, a trajectory ``read_trajectory`` returned (to read that file
  with options of its own), or one symbol per atom (a list or a numpy array,
  as MDAnalysis's ``atoms.elements``).

NOTHING IS DROPPED SILENTLY
---------------------------
A frame whose data ends beyond the end of the file (a copy taken while the
file was being written), whose record does not parse, whose atoms differ from
the other frames', or whose last coordinates are still the netCDF fill value
is listed in ``skipped`` with the reason. A header that counts more frames
than the file holds (a cut copy, or a damaged count) lists each lost frame,
up to 1000 in a row; a longer run is one entry that names its count (the
records are counted from the offsets, so a count damaged into millions costs
no memory). A value that does not parse when a frame loads raises FrameError
naming the file and the frame, as do the checks that a later frame holds the
atoms of frame 0; bytes that raise another error while parsing (a shape
too large for an integer, JSON nested past the recursion limit) are reported
as a ValueError naming the file. A gzip-compressed copy is refused with how
to decompress it, because these formats are read by seeking.

TIMINGS
-------
Measured on this machine (Windows 11, i5-13420H, Python 3.11, numpy 2.4.6;
not pinned, other sessions running, so the ranges are wide), 10 000 atoms x
100 frames per file in a tilted box, written by the programs themselves; three
runs of 3, 5 and 5 repeats (the third after the checks for damaged counts,
labels without masses and frames repeating frame 0 were added, on
2026-10-07); open = header, index, frame 0 and the load record, per frame =
every frame in order, divided by 100:

=========================================  =======  ============  ===========
file                                       size     open          per frame
=========================================  =======  ============  ===========
ASE .traj with momenta (ASE 3.29)          48.1 MB  0.03-0.18 s   4.7-23 ms
GSD, float32 (gsd 5.0.1, gsd.hoomd)        35.9 MB  0.02-0.09 s   5.2-30 ms
GSD, tilted box (OVITO 3.16.1)             40.0 MB  0.23-2.1 s    5.1-33 ms
NetCDF CDF-5, LAMMPS layout, double        56.0 MB  0.016-0.19 s  7.3-43 ms
(netCDF-C 4.9.3)
NetCDF CDF-2, float32 (MDAnalysis 2.10)    24.0 MB  0.014-0.25 s  4.5-33 ms
=========================================  =======  ============  ===========

Opening an OVITO file with a tilted box reads every frame's positions, to
test both readings of its box; the 10 000-atom file was decided by them (box
and positions to 1e-14 Å). Reading is about 0.01-0.04 s per 10 000-atom
frame, the range md_readers measured for the text formats.

REFERENCES
----------
[1] ASE source, ase/io/ulm.py ("File layout", Writer, read_header),
    https://gitlab.com/ase/ase/-/blob/master/ase/io/ulm.py (read from ASE
    3.29.0)
[2] ASE source, ase/io/trajectory.py (TrajectoryWriter._write_atoms,
    write_atoms, read_atoms), https://gitlab.com/ase/ase/-/blob/master/ase/io/
    trajectory.py
[3] GSD documentation, "File layer",
    https://gsd.readthedocs.io/en/stable/file-layer.html
[4] GSD source, gsd/gsd.h (enum gsd_type, struct gsd_header, struct
    gsd_index_entry) and gsd/gsd.c (gsd_is_entry_valid, name-list reading),
    https://github.com/glotzerlab/gsd
[5] GSD documentation, "HOOMD Schema",
    https://gsd.readthedocs.io/en/stable/schema-hoomd.html
[6] Unidata, "NetCDF File Format Specifications" (classic and 64-bit offset),
    https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html;
    fill values from include/netcdf.h, https://github.com/Unidata/netcdf-c
[7] PnetCDF, "CDF-5 file format specification",
    https://parallel-netcdf.github.io/doc/c-reference/pnetcdf-c/CDF_002d5-file-format-specification.html
[8] AMBER, "AMBER Trajectory (and Restart) NetCDF Conventions", version 1.0,
    https://ambermd.org/netcdf/nctraj.xhtml
[9] LAMMPS source, src/NETCDF/dump_netcdf.cpp, release stable_22Jul2025,
    https://github.com/lammps/lammps/blob/stable_22Jul2025/src/NETCDF/dump_netcdf.cpp
[10] ASE source, ase/units.py, https://gitlab.com/ase/ase/-/blob/master/ase/units.py
[11] OVITO source, src/ovito/particles/export/gsd/GSDExporter.cpp,
     https://gitlab.com/stuko/ovito/-/blob/master/src/ovito/particles/export/gsd/GSDExporter.cpp
[12] Unidata, "NetCDF Utilities" (nccopy -k),
     https://docs.unidata.ucar.edu/nug/current/netcdf_utilities_guide.html
[13] ASE source, ase/io/netcdftrajectory.py (NetCDFTrajectory.write,
     _add_velocities), https://gitlab.com/ase/ase/-/blob/master/ase/io/netcdftrajectory.py
[14] LAMMPS documentation, "units command" (style nano),
     https://docs.lammps.org/units.html
[15] mdtraj source, mdtraj/formats/gsd.py (write_gsd) and
     mdtraj/utils/unitcell.py (lengths_and_angles_to_tilt_factors),
     https://github.com/mdtraj/mdtraj (read from mdtraj 1.11.1)
[16] Wikipedia, "Diamond": "of all known substances, diamond has the
     greatest number of atoms per unit volume"; 3520 kg/m^3 in pure diamond,
     https://en.wikipedia.org/wiki/Diamond
"""
from __future__ import annotations

import importlib
import json
import math
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import constants

from . import md_model
from .md_formats_base import FormatSpec
from .md_model import (NO_TIMESTEP, Frame, FrameError, Trajectory,
                       frame_from_arrays, validate_symbol)
from .readers import UnsupportedFormat

__all__ = [
    "FORMATS", "ASE_VELOCITY_TO_ANG_PER_PS", "read_ase_traj", "read_gsd",
    "read_amber_netcdf", "box_from_hoomd", "box_from_cell_parameters",
    "netcdf_header", "NetcdfHeader", "NetcdfVariable",
]


def _mr():
    """facet.core.md_readers, imported when first used: it consults this
    module through md_formats_base.FORMAT_MODULES, so a top-level import
    could run while it is still initialising."""
    return importlib.import_module(".md_readers", __package__)


# ---------------------------------------------------------------------------
# constants: each one is a definition from a cited source, or a choice
# ---------------------------------------------------------------------------

_GZIP_MAGIC = b"\x1f\x8b"

# ASE's unit of time [10]: ase/units.py sets u['second'] = 1e10 * sqrt(e /
# amu) with the CODATA 2014 values it names (__codata_version__ = '2014').
_ASE_ELEMENTARY_CHARGE_C = 1.6021766208e-19
_ASE_ATOMIC_MASS_KG = 1.660539040e-27
_ASE_TIME_UNITS_PER_SECOND = 1e10 * math.sqrt(_ASE_ELEMENTARY_CHARGE_C
                                              / _ASE_ATOMIC_MASS_KG)
# One ASE unit of velocity (Å per ASE time unit) in Å/ps.
ASE_VELOCITY_TO_ANG_PER_PS: float = _ASE_TIME_UNITS_PER_SECOND * constants.pico

# Unit names a NetCDF units attribute may carry, as factors to Å and ps.
# Definitions of the SI prefixes, from scipy.constants.
_NM_TO_ANG = constants.nano / constants.angstrom
_FS_TO_PS = constants.femto / constants.pico
_NS_TO_PS = constants.nano / constants.pico
_LENGTH_UNITS = {
    "angstrom": 1.0, "angstroms": 1.0, "angstroem": 1.0, "ang": 1.0,
    "a": 1.0, "å": 1.0, "Å": 1.0,
    "nanometer": _NM_TO_ANG, "nanometers": _NM_TO_ANG,
    "nanometre": _NM_TO_ANG, "nanometres": _NM_TO_ANG, "nm": _NM_TO_ANG,
}
_TIME_UNITS = {
    "picosecond": 1.0, "picoseconds": 1.0, "ps": 1.0,
    "femtosecond": _FS_TO_PS, "femtoseconds": _FS_TO_PS, "fs": _FS_TO_PS,
    "nanosecond": _NS_TO_PS, "nanoseconds": _NS_TO_PS, "ns": _NS_TO_PS,
}

# What FACET reads, for refusals.
_READS = ("FACET reads ASE .traj, HOOMD GSD and AMBER NetCDF (classic, 64-bit "
          "offset and CDF-5) trajectories, LAMMPS data and dump files, "
          "extended XYZ, VASP XDATCAR and DL_POLY CONFIG / HISTORY")


# ---------------------------------------------------------------------------
# what the three readers share
# ---------------------------------------------------------------------------

class _BinaryTrajectory(Trajectory):
    """Frames read on demand from a binary file.

    Each frame opens the file, reads its byte ranges and closes it, so frames
    load from several threads without a shared file position. ``periodic``
    is (x, y, z) when the file states it, else None; ``ids_track_atoms`` is
    True, because each of these formats keeps one atom order, or ids.
    A ValueError raised while a frame loads names the file, so FrameError
    for any frame names it too.
    """

    ids_track_atoms: bool = True

    def __init__(self, path: Path, *, periodic=None, **kwargs) -> None:
        super().__init__(source_path=path, **kwargs)
        self._path = Path(path)
        self._name = self._path.name
        self.periodic = periodic

    def close(self) -> None:
        """No file stays open between frames; kept for the readers' protocol."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _build(self, k: int) -> Frame:
        raise NotImplementedError

    def _load(self, k: int) -> Frame:
        try:
            return self._build(k)
        except FrameError:
            raise
        except ValueError as error:
            raise ValueError(f"{self._name}: {error}") from error
        except OSError as error:
            raise ValueError(f"{self._name}: the file could not be read "
                             f"({error})") from error
        except _PARSE_ERRORS as error:
            raise ValueError(f"{self._name}: {_parse_error_text(error)}") \
                from error

    def frame(self, k: int) -> Frame:
        """Trajectory.frame, with the file named in every FrameError: the
        checks Trajectory.frame makes after loading (atom count, ids,
        elements, timestep) do not name it."""
        try:
            return super().frame(k)
        except FrameError as error:
            if self._name in str(error):
                raise
            raise FrameError(f"{self._name}, {error}") from error

    def _hold_to(self, frame0: Frame) -> None:
        """Make frame 0, built when the file was opened, the frame every
        later frame is held to (Trajectory.frame otherwise takes the first
        frame a caller loads, so frame 0 loaded after frame 2 would be
        compared with frame 2)."""
        if getattr(self, "_reference", ()) is None:
            self._reference = (frame0.atom_id, frame0.elements, 0)


# What parsing damaged bytes can raise besides ValueError and OSError (a
# shape too large for an integer, JSON nested past the recursion limit, a
# type error from a value of an unexpected kind); a reader reports them as a
# ValueError naming the file, as the readers' contract says.
_PARSE_ERRORS = (TypeError, OverflowError, RecursionError, IndexError,
                 KeyError, struct.error, ZeroDivisionError)


def _parse_error_text(error: BaseException) -> str:
    return (f"its bytes do not parse ({type(error).__name__}: {error}); "
            "if the program that wrote it still opens it, write it again, "
            "or export it in a format FACET reads")


def _guarded(path, call):
    """``call()``, with the errors damaged bytes raise outside the readers'
    contract (:data:`_PARSE_ERRORS`) reported as a ValueError naming the
    file."""
    try:
        return call()
    except _PARSE_ERRORS as error:
        raise ValueError(f"{Path(path).name}: {_parse_error_text(error)}") \
            from error


# The longest run of frames listed one by one in ``skipped`` when a header
# counts frames the file does not hold. A longer run (a count damaged into
# millions) is one entry that names how many frames it stands for, so that a
# damaged count cannot fill memory. A choice: a cut copy of a long run lists
# up to this many frames individually.
_LISTED_SKIPS = 1000


def _skip_run(skipped: dict[int, str], notes: list[str], start: int,
              stop: int, reason: str) -> None:
    """File positions start .. stop-1, all lost for ``reason``, into
    ``skipped``: one entry each, or one entry at ``start`` that counts them
    when there are more than :data:`_LISTED_SKIPS` (and a note saying so)."""
    count = stop - start
    if count <= 0:
        return
    if count <= _LISTED_SKIPS:
        for position in range(start, stop):
            skipped[position] = reason
        return
    text = (f"{reason}; this one entry stands for {count} frames (file "
            f"positions {start} to {stop - 1} as the header counts them), "
            f"more than the {_LISTED_SKIPS} listed one by one")
    skipped[start] = text
    notes.append(f"the header counts {count} frames from file position "
                 f"{start} on that the file does not hold; skipped lists them "
                 "as one entry")


def _read_exact(handle, offset: int, nbytes: int, what: str) -> bytes:
    handle.seek(offset)
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(
            f"{what}: bytes {offset} to {offset + nbytes} lie beyond the end "
            f"of the file ({len(data)} of the {nbytes} bytes are there)")
    return data


def _refuse_gzip(path: Path, what: str) -> None:
    with open(path, "rb") as handle:
        if handle.read(2) == _GZIP_MAGIC:
            raise UnsupportedFormat(
                f"{path.name}: a gzip-compressed {what}. FACET reads such "
                "files uncompressed, because a frame is reached by seeking to "
                "its offset in the file; decompress it first (gzip -d, or "
                "7-Zip on Windows) and open the result")


def _checked_timestep_fs(value, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or \
            not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name}: timestep_fs {value!r}: a number of "
                         "femtoseconds is needed")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name}: timestep_fs {value!r}: a positive number of "
                         "femtoseconds is needed")
    return out


def _named(name: str, check, *args):
    """``check(*args)``, with a ValueError's message prefixed by the file
    name (an option checked against a file names the file)."""
    try:
        return check(*args)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from None


def _times_from_steps(name: str, steps: Sequence[int], timestep_fs: float
                      ) -> tuple[list[float], str]:
    """Each frame's time from its step and the user's step length."""
    if all(int(s) == NO_TIMESTEP for s in steps):
        raise ValueError(f"{name}: timestep_fs gives the time of a step, and "
                         "the file holds no step numbers; it cannot be used "
                         "for this file")
    times = [np.nan if int(s) == NO_TIMESTEP else
             int(s) * timestep_fs * _FS_TO_PS for s in steps]
    return times, (f"frame times are step x timestep_fs ({timestep_fs:g} fs, "
                   "the timestep_fs argument)")


def _symbols_from_numbers(numbers, where: str) -> np.ndarray:
    """Element symbols for atomic numbers, refusing 0 and unknown numbers."""
    z = np.asarray(numbers)
    if z.dtype.kind not in "iu":
        raise ValueError(f"{where}: the atomic numbers are stored as "
                         f"{z.dtype}, not integers")
    unique = np.unique(z)
    table: dict[int, str] = {}
    for value in unique.tolist():
        count = int((z == value).sum())
        if value == 0:
            raise ValueError(
                f"{where}: {count} atom(s) carry atomic number 0 (ASE's "
                "placeholder 'X'), which names no element; give them an "
                "element before writing the file")
        element = _element_of_number(int(value))
        if element is None or element.atomic_number != value \
                or element.name not in md_model._canonical_symbols():
            raise ValueError(f"{where}: {count} atom(s) carry atomic number "
                             f"{value}, which is not an element gemmi knows")
        table[int(value)] = validate_symbol(element.name)
    return np.array([table[int(v)] for v in z.tolist()], dtype="<U2")


def _element_of_number(value: int):
    """gemmi's Element for an atomic number, or None outside its table."""
    import gemmi

    if not 0 < value < 256:
        return None
    try:
        return gemmi.Element(int(value))
    except (RuntimeError, ValueError, TypeError):
        return None


def _listed(values: Sequence, limit: int = 10) -> str:
    """Values as a list, the first ``limit`` of them and how many more."""
    values = list(values)
    shown = ", ".join(str(v) for v in values[:limit])
    more = f", and {len(values) - limit} more" if len(values) > limit else ""
    return f"[{shown}{more}]"


def _not_read(names: Sequence[str], what: str) -> list[str]:
    names = sorted(set(names))
    if not names:
        return []
    return [f"{what} not read: " + ", ".join(names)]


# ---------------------------------------------------------------------------
# ASE trajectory (ULM)
# ---------------------------------------------------------------------------

_ULM_MAGICS = (b"- of Ulm", b"AFFormat")
_ULM_TAG = "ASE-Trajectory"
_ULM_VERSIONS = (1, 2, 3)
_ULM_HEADER_BYTES = 48
# Keys an item holds that this reader uses; everything else is noted.
_ASE_READ = {"positions", "cell", "numbers", "pbc", "masses", "momenta",
             "charges", "info", "version", "ase_version"}
_ASE_INFO_READ = {"Timestep", "cell_origin"}


@dataclass(frozen=True)
class _UlmArray:
    shape: tuple[int, ...]
    dtype: np.dtype
    offset: int

    @property
    def nbytes(self) -> int:
        # Python integers: a damaged shape such as [1e20, 3] overflows int64
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass
class _AseItem:
    position: int
    arrays: dict[str, _UlmArray]
    plain: dict
    children: dict
    header: int = 0                      # position of the item whose header applies
    cell: np.ndarray | None = None
    origin: np.ndarray | None = None
    origin_unread: bool = False          # info['cell_origin'] not three numbers
    timestep: int | None = None
    pbc: tuple[bool, bool, bool] | None = None
    pbc_unread: bool = False             # pbc present, not three flags


def _ulm_array(value, little_endian: bool, key: str) -> _UlmArray:
    try:
        shape, dtype_name, offset = value["ndarray"]
        dtype = np.dtype(str(dtype_name))
        shape = tuple(int(s) for s in shape)
        offset = int(offset)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"its {key} array reference does not parse "
                         f"({error})") from None
    if dtype.kind not in "biuf" or any(s < 0 for s in shape) or offset < 0:
        raise ValueError(f"its {key} array reference names dtype {dtype}, "
                         f"shape {shape}, offset {offset}")
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if little_endian else ">")
    return _UlmArray(shape, dtype, offset)


def _info_vector(value) -> np.ndarray | None:
    """A 3-vector from info: a list, or ASE's {"__ndarray__": [shape, dtype,
    values]} JSON encoding (ase/io/jsonio.py)."""
    if isinstance(value, dict) and "__ndarray__" in value:
        try:
            value = value["__ndarray__"][2]
        except (TypeError, IndexError, KeyError):
            return None
    try:
        out = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if out.shape != (3,) or not np.isfinite(out).all():
        return None
    return out


def _ase_pbc(value) -> tuple[bool, bool, bool] | None:
    """pbc as ASE writes it (three booleans), or None for anything else."""
    if not isinstance(value, list) or len(value) != 3 or \
            not all(isinstance(v, (bool, int)) for v in value):
        return None
    return tuple(bool(v) for v in value)


def _ase_item(handle, size: int, offset: int, position: int
              ) -> tuple[_AseItem | None, str | None]:
    if offset < _ULM_HEADER_BYTES:
        return None, (f"its entry in the table of frame offsets names byte "
                      f"{offset}, inside the {_ULM_HEADER_BYTES}-byte ULM header, "
                      "where no record starts")
    if offset + 8 > size:
        return None, ("truncated: its record starts beyond the end of the file "
                      f"(byte {offset} of {size})")
    handle.seek(offset)
    (length,) = struct.unpack("<q", handle.read(8))
    if length < 0 or offset + 8 + length > size:
        return None, (f"truncated: its {length}-byte record ends beyond the "
                      f"end of the file ({size} bytes)")
    raw = handle.read(length)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        return None, f"its JSON record does not parse ({error})"
    if not isinstance(data, dict):
        return None, "its JSON record is not a dictionary"
    little = bool(data.pop("_little_endian", True))
    arrays, children, plain = {}, {}, {}
    try:
        for key, value in data.items():
            if key.endswith(".") and isinstance(value, dict) and "ndarray" in value:
                arrays[key[:-1]] = _ulm_array(value, little, key[:-1])
            elif key.endswith(".") and isinstance(value, dict):
                children[key[:-1]] = value
            else:
                plain[key] = value
    except ValueError as error:
        return None, str(error)
    for key in ("positions", "numbers", "masses", "momenta", "charges"):
        array = arrays.get(key)
        if array is not None and array.offset + array.nbytes > size:
            return None, (f"truncated: its {key} array ends beyond the end of "
                          f"the file ({size} bytes)")
    if "positions" not in arrays:
        return None, "it holds no positions array"
    item = _AseItem(position, arrays, plain, children)
    try:
        cell = np.asarray(plain.get("cell"), dtype=np.float64)
    except (TypeError, ValueError):
        cell = None
    if cell is None or cell.shape != (3, 3) or not np.isfinite(cell).all():
        return None, f"its cell {plain.get('cell')!r} is not three vectors"
    item.cell = cell
    if "pbc" in plain:
        item.pbc = _ase_pbc(plain["pbc"])
        item.pbc_unread = item.pbc is None
    info = plain.get("info") if isinstance(plain.get("info"), dict) else {}
    if "cell_origin" in info:
        item.origin = _info_vector(info["cell_origin"])
        item.origin_unread = item.origin is None
    step = info.get("Timestep")
    if isinstance(step, int) and not isinstance(step, bool) and \
            abs(step) < 2 ** 63:
        item.timestep = step
    return item, None


def _ulm_values(handle, array: _UlmArray, shape: tuple[int, ...], what: str
                ) -> np.ndarray:
    if array.shape != shape:
        raise ValueError(f"its {what} array has shape {array.shape}; "
                         f"{shape} is needed")
    data = _read_exact(handle, array.offset, array.nbytes, what)
    return np.frombuffer(data, dtype=array.dtype).reshape(shape)


class _AseTrajectory(_BinaryTrajectory):
    def __init__(self, path: Path, items: Sequence[_AseItem], *, elements,
                 masses: Mapping[int, _UlmArray | None],
                 weights: np.ndarray | None, velocities: bool, times,
                 **kwargs) -> None:
        super().__init__(path, **kwargs)
        self._items = list(items)
        self._elements = elements
        self._masses = dict(masses)
        self._weights = weights
        self._velocities = velocities
        self._times = times

    def _build(self, k: int) -> Frame:
        item = self._items[k]
        n = len(self._elements)
        with open(self._path, "rb") as handle:
            positions = _ulm_values(handle, item.arrays["positions"], (n, 3),
                                    "positions")
            velocities = None
            if self._velocities and "momenta" in item.arrays:
                stored = self._masses.get(item.header)
                masses = self._weights if stored is None else np.asarray(
                    _ulm_values(handle, stored, (n,), "masses"),
                    dtype=np.float64)
                if masses is None or not (masses > 0).all():
                    raise ValueError("the masses that turn its momenta into "
                                     "velocities are not all positive")
                momenta = _ulm_values(handle, item.arrays["momenta"], (n, 3),
                                      "momenta")
                velocities = (np.asarray(momenta, dtype=np.float64)
                              / masses[:, None] * ASE_VELOCITY_TO_ANG_PER_PS)
            charges = None
            if "charges" in item.arrays:
                charges = _ulm_values(handle, item.arrays["charges"], (n,),
                                      "charges")
        if not np.any(item.cell):
            raise ValueError(
                "its cell is zero: ASE stores a zero cell for a model without "
                "a periodic box, and FACET reads periodic models only")
        time = None
        if self._times is not None and np.isfinite(self._times[k]):
            time = float(self._times[k])
        return frame_from_arrays(
            self._elements, np.asarray(positions, dtype=np.float64),
            box_ang=item.cell, origin_ang=item.origin, timestep=item.timestep,
            time_ps=time, vel_ang_per_ps=velocities, charge_e=charges)


def read_ase_traj(path, *, timestep_fs: float | None = None) -> Trajectory:
    """An ASE trajectory (``.traj``, ULM binary) [1, 2].

    Elements come from the atomic numbers; velocities from the momenta and
    the masses the file stores, or gemmi's standard atomic weights when it
    stores none (ASE stores masses only when they were set), converted from
    ASE's unit with :data:`ASE_VELOCITY_TO_ANG_PER_PS`. ASE's initial charges
    are read as per-atom charges. OVITO's ase/traj export adds
    ``info['Timestep']`` and ``info['cell_origin']``, which give the step and
    the box origin; ``timestep_fs`` turns those steps into times. A frame
    whose atoms differ from the other frames' (ASE then writes a header in
    every later frame) is skipped with the reason.
    """
    return _guarded(path, lambda: _read_ase_traj(path, timestep_fs=timestep_fs))


def _read_ase_traj(path, *, timestep_fs) -> Trajectory:
    mr = _mr()
    path = mr._existing(path)
    name = path.name
    _refuse_gzip(path, "ASE trajectory")
    step_fs = _checked_timestep_fs(timestep_fs, name)
    size = path.stat().st_size
    skipped: dict[int, str] = {}
    notes: list[str] = []
    with open(path, "rb") as handle:
        head = handle.read(_ULM_HEADER_BYTES)
        if len(head) < 8 or head[:8] not in _ULM_MAGICS:
            raise UnsupportedFormat(f"{name}: not an ASE ULM file (its first "
                                    f"bytes are {head[:8]!r}). {_READS}")
        if len(head) < _ULM_HEADER_BYTES:
            raise ValueError(f"{name}: the file ends inside its {_ULM_HEADER_BYTES}"
                             f"-byte ULM header ({size} bytes); it was cut, and "
                             "holds no frame")
        tag = head[8:24].decode("ascii", "replace").rstrip()
        if tag != _ULM_TAG:
            raise UnsupportedFormat(
                f"{name}: an ASE ULM file with the tag {tag!r}, not "
                f"{_ULM_TAG!r} (GPAW and other tools write ULM files of their "
                "own); FACET reads ASE trajectories, which ase.io.write(..., "
                "format='traj') and ase.io.Trajectory write")
        version, nitems, pos0 = struct.unpack_from("<qqq", head, 24)
        if version not in _ULM_VERSIONS:
            raise UnsupportedFormat(
                f"{name}: ULM version {version}; FACET reads versions 1 to 3 "
                "(ASE 3.29 writes 3). Write the trajectory again with ASE as "
                "extended XYZ (ase.io.write(..., format='extxyz')), which "
                "FACET reads")
        if nitems <= 0:
            raise ValueError(f"{name}: the ASE trajectory holds no frame (its "
                             f"header counts {nitems} items)")
        if pos0 < _ULM_HEADER_BYTES or pos0 + 8 > size:
            raise ValueError(
                f"{name}: the table of frame offsets starts at byte {pos0}, "
                f"beyond the end of the file ({size} bytes); the file was cut "
                "while it was being written, and its frames cannot be found")
        listed = min(int(nitems), (size - pos0) // 8)
        handle.seek(pos0)
        offsets = np.frombuffer(handle.read(8 * listed), dtype="<i8")
        items: list[_AseItem] = []
        first_at: dict[int, int] = {}
        for position, offset in enumerate(offsets.tolist()):
            if offset in first_at:
                # ASE gives every frame a record of its own
                skipped[position] = (
                    f"its entry in the table of frame offsets names byte "
                    f"{offset}, the record of file position "
                    f"{first_at[offset]}; the table names one record twice")
                continue
            first_at[offset] = position
            item, reason = _ase_item(handle, size, int(offset), position)
            if item is None:
                skipped[position] = reason
            else:
                items.append(item)
        # entries past the end of the file: those frames' records cannot be
        # found (ase/io/ulm.py, Writer.sync, writes the frames after the
        # table it allocates after the first, so they lie past the end too)
        _skip_run(skipped, notes, listed, int(nitems),
                  "truncated: its entry in the table of frame offsets lies "
                  "beyond the end of the file")
        if not items or items[0].position != 0:
            reason = skipped.get(0, "it cannot be read")
            raise ValueError(
                f"{name}: frame 0 cannot be read ({reason}); ASE stores the "
                "atomic numbers in frame 0, so no frame can be read")
        if "numbers" not in items[0].arrays:
            raise ValueError(f"{name}: frame 0 holds no atomic numbers, so the "
                             "atoms have no elements")
        numbers: dict[int, np.ndarray] = {}
        for item in items:
            if "numbers" in item.arrays:
                item.header = item.position
                shape = item.arrays["numbers"].shape
                if len(shape) != 1:
                    skipped[item.position] = (f"its numbers array has shape "
                                              f"{shape}, not (N,)")
                    continue
                try:
                    numbers[item.position] = np.asarray(_ulm_values(
                        handle, item.arrays["numbers"], shape, "numbers"))
                except ValueError as error:
                    skipped[item.position] = str(error)
        if 0 not in numbers:
            raise ValueError(f"{name}: frame 0's atomic numbers cannot be read "
                             f"({skipped.get(0)})")
    by_position = {i.position: i for i in items}
    items = [i for i in items if i.position not in skipped
             and i.header in numbers]

    # The atoms most frames hold (ASE writes a header in every frame after one
    # changes them); a frame holding others is skipped.
    keys = [numbers[i.header].tobytes() for i in items]
    tally = Counter(keys)
    best = max(tally.values())
    reference_key = next(k for k in keys if tally[k] == best)
    reference = next(numbers[i.header] for i, k in zip(items, keys)
                     if k == reference_key)
    counts = [len(numbers[i.header]) for i in items]
    n0 = len(reference)
    kept = []
    for item, key in zip(items, keys):
        if key == reference_key:
            kept.append(item)
            continue
        own = numbers[item.header]
        if len(own) != n0:
            skipped[item.position] = mr._count_reason(len(own), n0, counts)
        else:
            row = int(np.flatnonzero(own != reference)[0])
            skipped[item.position] = (
                f"its atomic numbers differ from those most frames hold "
                f"({int((own != reference).sum())} atom(s); first: atom {row}, "
                f"Z {int(reference[row])} -> {int(own[row])})")
    items = kept
    elements = _symbols_from_numbers(reference, f"{name}, frame "
                                     f"{items[0].position}")

    ase_version = by_position[0].plain.get("ase_version")
    notes[:0] = [
        f"written by ASE {ase_version or '(version not recorded)'}, ULM version "
        f"{version}, tag {_ULM_TAG}",
        "elements from the atomic numbers; ASE trajectories store no atom ids, "
        "so atoms are numbered 0 .. N-1 in the order of the file, the same "
        "atoms in every frame"]

    # masses for the velocities: those the file stores with the header that
    # applies to each frame, else gemmi's standard atomic weights
    header_masses = {i.header: by_position[i.header].arrays.get("masses")
                     for i in items}
    has_momenta = any("momenta" in i.arrays for i in items)
    weights = None
    units = ("ASE units: lengths in Å (positions and cell); the file holds no "
             "time")
    if has_momenta:
        with_masses = [i.position for i in items
                       if header_masses.get(i.header) is not None]
        without_masses = [i.position for i in items
                          if header_masses.get(i.header) is None]
        lacking = ("the file stores no masses" if not with_masses else
                   f"the frame(s) at file position(s) "
                   f"{_listed(without_masses)} store no masses")
        if without_masses:
            table = dict(mr.element_weights_amu())
            missing = sorted({str(e) for e in elements} - set(table))
            if missing:
                notes.append(
                    f"momenta not read: {lacking}, and gemmi gives no standard "
                    f"atomic weight for {', '.join(missing)}")
                has_momenta = False
            else:
                weights = np.array([table[str(e)] for e in elements])
        if has_momenta:
            units = (
                "ASE units: lengths in Å; velocities are momenta / masses in "
                "ASE's unit of velocity, Å per (Å sqrt(amu/eV)), x "
                f"{ASE_VELOCITY_TO_ANG_PER_PS:.10g} = Å/ps (ase/units.py: "
                "second = 1e10 sqrt(e/amu), CODATA 2014 e and amu); the file "
                "holds no time")
            gemmi_text = (
                "ASE divided by its own table (ase.data.atomic_masses), which "
                "differs from gemmi's by up to 1.6e-4 relative for elements "
                "with a standard atomic weight (measured against ASE 3.29: O "
                "2.5e-5, Si 1.8e-5, Na below 4e-8), so those velocities carry "
                "that relative difference")
            if not without_masses:
                notes.append("velocities from the momenta divided by the masses "
                             "the file stores")
            elif not with_masses:
                notes.append(
                    "velocities from the momenta divided by gemmi's standard "
                    "atomic weights: the file stores no masses (ASE stores them "
                    f"only when they were set), and {gemmi_text}")
            else:
                notes.append(
                    f"velocities of the frame(s) at file position(s) "
                    f"{_listed(with_masses)} from the momenta divided by the "
                    "masses the file stores; the frame(s) at file position(s) "
                    f"{_listed(without_masses)} store no masses (ASE stores "
                    "them only once they are set), and their velocities are the "
                    "momenta divided by gemmi's standard atomic weights: "
                    + gemmi_text)
            without = [i.position for i in items if "momenta" not in i.arrays]
            if without:
                notes.append(f"frame(s) at file position(s) {_listed(without)} "
                             "store no momenta and carry no velocities")
    if any("charges" in i.arrays for i in items):
        notes.append("per-atom charges from ASE's initial charges (the charges "
                     "array), taken as e")
    headers = [by_position[i.header] for i in items]
    pbcs = {h.pbc for h in headers}
    periodic = next(iter(pbcs)) if len(pbcs) == 1 else None
    if periodic is not None and not all(periodic):
        notes.append(f"pbc is {list(periodic)}: an axis is marked as not "
                     "periodic, while FACET measures the box as periodic along "
                     "a, b and c, so atoms near that face see images across it")
    odd_pbc = [i for i, h in zip(items, headers) if h.pbc_unread]
    if odd_pbc:
        notes.append(f"pbc not read for the frame(s) at file position(s) "
                     f"{_listed(i.position for i in odd_pbc)}: it is not three "
                     "flags "
                     f"({by_position[odd_pbc[0].header].plain.get('pbc')!r})")
    odd_origin = [i.position for i in items if i.origin_unread]
    if any(i.origin is not None for i in items):
        notes.append("box origin from info['cell_origin'], which OVITO's "
                     "ase/traj export writes (ASE itself stores no origin)")
    elif not odd_origin:
        notes.append("ASE stores no box origin: the origin is (0, 0, 0)")
    if odd_origin:
        notes.append(f"info['cell_origin'] of the frame(s) at file position(s) "
                     f"{_listed(odd_origin)} is not three numbers, so it is not "
                     "read; those frames take the origin (0, 0, 0)")
    steps = [NO_TIMESTEP if i.timestep is None else i.timestep for i in items]
    if any(i.timestep is not None for i in items):
        notes.append("steps from info['Timestep'], which OVITO's ase/traj "
                     "export writes")
    notes.append("ASE's MD integrators do not wrap atoms into the cell, and the "
                 "file does not say whether these positions were wrapped, so no "
                 "continuous (unwrapped) positions are set; the frames hold the "
                 "positions wrapped into the box")
    unread: list[str] = []
    info_unread: list[str] = []
    for item in items:
        unread += [k for k in list(item.plain) + list(item.arrays)
                   if k not in _ASE_READ]
        for key, child in item.children.items():
            inner = sorted(str(v).rstrip(".") for v in child
                           if v not in ("name", "parameters"))
            unread.append(f"{key} ({', '.join(inner)})" if inner else key)
        info = item.plain.get("info")
        if isinstance(info, dict):
            info_unread += [k for k in info if k not in _ASE_INFO_READ]
    notes += _not_read(unread, "arrays and keys")
    notes += _not_read(info_unread, "info keys")
    times = None
    if step_fs is not None:
        times, note = _times_from_steps(name, steps, step_fs)
        notes.append(note)
    boxes = [i.cell for i in items]
    origins = [np.zeros(3) if i.origin is None else i.origin for i in items]
    box_varies = any(not np.array_equal(b, boxes[0]) or
                     not np.array_equal(o, origins[0])
                     for b, o in zip(boxes[1:], origins[1:]))
    trajectory = _AseTrajectory(
        path, items, elements=elements, masses=header_masses, weights=weights,
        velocities=has_momenta,
        times=None if times is None else np.asarray(times, dtype=float),
        periodic=periodic, file_format="ase-traj", n_atoms=n0,
        n_frames=len(items), type_map_source="file symbols", timesteps=steps,
        times_ps=times, skipped=skipped, notes=notes, units_note=units,
        box_varies=box_varies)
    frame0 = mr._frame0(name, items[0].position, lambda: trajectory._build(0))
    trajectory._hold_to(frame0)
    if frame0.charge_e is not None:
        charges, charge_notes = md_model.charges_per_element(frame0.elements,
                                                             frame0.charge_e)
        trajectory.charges_e = charges
        trajectory.notes.extend(charge_notes)
    return mr._finish(trajectory, frame0)


# ---------------------------------------------------------------------------
# GSD (HOOMD schema)
# ---------------------------------------------------------------------------

_GSD_MAGIC = 0x65DF65DF65DF65DF
_GSD_HEADER = struct.Struct("<QQQQQII")
_GSD_HEADER_BYTES = 256
_GSD_NAME_BYTES = 64
_GSD_INDEX = np.dtype([("frame", "<u8"), ("N", "<u8"), ("location", "<i8"),
                       ("M", "<u4"), ("id", "<u2"), ("type", "u1"),
                       ("flags", "u1")])
# enum gsd_type in gsd.h [4]
_GSD_TYPES = {1: "u1", 2: "<u2", 3: "<u4", 4: "<u8", 5: "i1", 6: "<i2",
              7: "<i4", 8: "<i8", 9: "<f4", 10: "<f8", 11: "S1"}
_GSD_FLOAT = (9, 10)
_GSD_INT = (1, 2, 3, 4, 5, 6, 7, 8)
_GSD_READ = {"configuration/step", "configuration/box",
             "configuration/dimensions", "particles/N", "particles/position",
             "particles/types", "particles/typeid", "particles/velocity",
             "particles/image", "particles/charge", "particles/mass"}


@dataclass(frozen=True)
class _GsdChunk:
    name: str
    n: int
    m: int
    type_code: int
    location: int

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(_GSD_TYPES[self.type_code])

    @property
    def nbytes(self) -> int:
        return self.n * self.m * self.dtype.itemsize


@dataclass
class _GsdIndex:
    application: str
    schema: str
    schema_version: tuple[int, int]
    version: tuple[int, int]
    names: list[str]
    frames: dict[int, dict[str, _GsdChunk]]  # the frames the index lists chunks for
    problems: dict[int, list[str]]
    n_frames: int                            # the largest frame number + 1


def box_from_hoomd(box6) -> tuple[np.ndarray, np.ndarray]:
    """(rows, origin) of a HOOMD box (Lx, Ly, Lz, xy, xz, yz) [5].

    a = (Lx, 0, 0), b = (xy Ly, Ly, 0), c = (xz Lz, yz Lz, Lz): the vectors the
    schema's unwrapping formula adds per image (x_u = x + i_x Lx + xy i_y Ly +
    xz i_z Lz, y_u = y + i_y Ly + yz i_z Lz, z_u = z + i_z Lz). The schema's
    position bounds centre the box on zero, so the origin is -(a + b + c) / 2.
    """
    lx, ly, lz, xy, xz, yz = (float(v) for v in np.asarray(box6).reshape(6))
    rows = np.array([[lx, 0.0, 0.0], [xy * ly, ly, 0.0],
                     [xz * lz, yz * lz, lz]])
    return rows, -rows.sum(axis=0) / 2.0


def _box_from_ovito_gsd(box6) -> tuple[np.ndarray, np.ndarray] | None:
    """(rows, origin) of the box OVITO's GSDExporter.cpp [11] meant: it writes
    (Lx, |b|, |c|, b_x / |b|, c_x / |c|, c_y / |c|) of its HOOMD box matrix.
    None when the numbers cannot be such a box."""
    lx, lb, lc, xy, xz, yz = (float(v) for v in np.asarray(box6).reshape(6))
    bx, cx, cy = xy * lb, xz * lc, yz * lc
    by2, cz2 = lb * lb - bx * bx, lc * lc - cx * cx - cy * cy
    if lx <= 0 or by2 <= 0 or cz2 <= 0:
        return None
    rows = np.array([[lx, 0.0, 0.0], [bx, math.sqrt(by2), 0.0],
                     [cx, cy, math.sqrt(cz2)]])
    return rows, -rows.sum(axis=0) / 2.0


def _gsd_index(path: Path) -> _GsdIndex:
    name = path.name
    size = path.stat().st_size
    with open(path, "rb") as handle:
        head = handle.read(_GSD_HEADER_BYTES)
        if len(head) < 8 or struct.unpack_from("<Q", head)[0] != _GSD_MAGIC:
            if len(head) >= 8 and struct.unpack_from(">Q", head)[0] == _GSD_MAGIC:
                raise UnsupportedFormat(
                    f"{name}: a big-endian GSD file; FACET reads GSD files in "
                    "little-endian byte order, as x86 and ARM machines write "
                    "them. Read it with the gsd package on the machine that "
                    "wrote it and export it again")
            raise UnsupportedFormat(f"{name}: not a GSD file (no GSD magic "
                                    f"number). {_READS}")
        if len(head) < _GSD_HEADER_BYTES:
            raise ValueError(f"{name}: the file ends inside its 256-byte GSD "
                             f"header ({size} bytes); it was cut, and holds no "
                             "frame")
        (_, index_location, index_allocated, namelist_location,
         namelist_allocated, schema_version, gsd_version) = _GSD_HEADER.unpack_from(head)
        application = head[48:112].split(b"\0")[0].decode("utf-8", "replace")
        schema = head[112:176].split(b"\0")[0].decode("utf-8", "replace")
        major, minor = gsd_version >> 16, gsd_version & 0xFFFF
        if major not in (1, 2) and gsd_version != 3:
            raise UnsupportedFormat(
                f"{name}: GSD file layer version {major}.{minor}; FACET reads "
                "versions 1.x and 2.x, as the gsd library does. Read it with "
                "the gsd package and write it again")
        if schema != "hoomd":
            raise UnsupportedFormat(
                f"{name}: a GSD file with schema {schema!r}; FACET reads the "
                "'hoomd' schema (HOOMD-blue and OVITO write it)")
        if index_location < _GSD_HEADER_BYTES or index_location >= size:
            raise ValueError(
                f"{name}: the GSD index starts at byte {index_location}, beyond "
                f"the end of the file ({size} bytes); the file was cut while it "
                "was being written, and its frames cannot be found")
        available = min(int(index_allocated), (size - index_location) // 32)
        handle.seek(index_location)
        index = np.frombuffer(handle.read(32 * available), dtype=_GSD_INDEX)
        ends = np.flatnonzero(index["location"] == 0)
        if ends.size:
            index = index[:ends[0]]
        elif available < index_allocated:
            raise ValueError(
                f"{name}: the GSD index runs past the end of the file ({size} "
                "bytes); the file was cut while it was being written, and its "
                "frames cannot be found")
        names_end = namelist_location + _GSD_NAME_BYTES * namelist_allocated
        if namelist_location < _GSD_HEADER_BYTES or names_end > size:
            raise ValueError(
                f"{name}: the GSD name list (bytes {namelist_location} to "
                f"{names_end}) lies beyond the end of the file ({size} bytes); "
                "the file was cut while it was being written")
        handle.seek(namelist_location)
        block = handle.read(names_end - namelist_location)
    names: list[str] = []
    if major == 1 or gsd_version == 3:
        pieces = [block[i:i + _GSD_NAME_BYTES].split(b"\0")[0]
                  for i in range(0, len(block), _GSD_NAME_BYTES)]
    else:
        pieces = block.split(b"\0")
    for piece in pieces:
        if not piece:
            break
        names.append(piece.decode("utf-8", "replace"))
    if index.size == 0:
        raise ValueError(f"{name}: the GSD index lists no data chunk, so the "
                         "file holds no frame")
    # gsd.c, gsd_is_entry_valid: "frame cannot be more than the number of
    # index entries"; the gsd library refuses such a file as corrupt
    beyond = np.flatnonzero(index["frame"] >= np.uint64(available))
    if beyond.size:
        entry = index[beyond[0]]
        raise ValueError(
            f"{name}: GSD index entry {int(beyond[0])} names frame "
            f"{int(entry['frame'])}, and the index has room for {available} "
            "entries; the gsd library refuses an entry whose frame number is "
            "not below that count (gsd.c, gsd_is_entry_valid), so the index is "
            "damaged. If the program that wrote the file still opens it, "
            "write it again")
    n_frames = int(index["frame"].max()) + 1
    frames: dict[int, dict[str, _GsdChunk]] = {}
    problems: dict[int, list[str]] = {}
    for entry in index:
        frame = int(entry["frame"])
        frames.setdefault(frame, {})
        type_code = int(entry["type"])
        label = names[entry["id"]] if entry["id"] < len(names) else \
            f"name id {int(entry['id'])}"
        problem = None
        if int(entry["flags"]) != 0:
            problem = f"chunk {label} has flags {int(entry['flags'])}, not 0"
        elif type_code not in _GSD_TYPES:
            problem = f"chunk {label} has the unknown type id {type_code}"
        elif entry["id"] >= len(names):
            problem = f"a chunk refers to {label}, outside the name list"
        else:
            chunk = _GsdChunk(label, int(entry["N"]), int(entry["M"]),
                              type_code, int(entry["location"]))
            if chunk.location < 0 or chunk.location + chunk.nbytes > size:
                problem = (f"truncated: chunk {label} ends beyond the end of the "
                           f"file ({size} bytes)")
            else:
                frames[frame][label] = chunk
        if problem is not None:
            problems.setdefault(frame, []).append(problem)
    return _GsdIndex(application, schema,
                     (schema_version >> 16, schema_version & 0xFFFF),
                     (major, minor), names, frames, problems, n_frames)


def _gsd_values(handle, chunk: _GsdChunk, what: str) -> np.ndarray:
    data = _read_exact(handle, chunk.location, chunk.nbytes, what)
    return np.frombuffer(data, dtype=chunk.dtype).reshape(chunk.n, chunk.m)


def _gsd_type_names(handle, chunk: _GsdChunk) -> list[str]:
    raw = _gsd_values(handle, chunk, "particles/types")
    out = []
    for row in raw:
        text = row.tobytes().split(b"\0")[0]
        out.append(text.decode("utf-8", "replace").strip())
    return out


class _GsdTrajectory(_BinaryTrajectory):
    def __init__(self, path: Path, index: _GsdIndex, frames: Sequence[int], *,
                 counts: Mapping[int, int], boxes, label_map: dict,
                 strict_labels: bool, length_factor: float,
                 velocity_factor: float, charges: bool, unwrap: bool,
                 ovito_shifts: bool, defaults: frozenset, times,
                 **kwargs) -> None:
        super().__init__(path, **kwargs)
        self._defaults = defaults           # chunks some frame holds
        self._gsd = index
        self._frames = list(frames)
        self._counts = dict(counts)
        self._boxes = list(boxes)
        self._label_map = dict(label_map)
        self._strict_labels = strict_labels  # every type name needs the map
        self._length_factor = length_factor
        self._velocity_factor = velocity_factor
        self._charges = charges
        self._unwrap = unwrap
        self._ovito_shifts = ovito_shifts
        self._times = times

    def _chunk(self, f: int, name: str) -> _GsdChunk | None:
        return _gsd_chunk(self._gsd, f, name)

    def _build(self, k: int) -> Frame:
        f = self._frames[k]
        n = self._counts[f]
        rows, origin = self._boxes[k]
        with open(self._path, "rb") as handle:
            positions = _gsd_per_particle(handle, self._chunk(f, "particles/position"),
                                          n, 3, "particles/position", float) \
                * self._length_factor
            types_chunk = self._chunk(f, "particles/types")
            type_names = ["A"] if types_chunk is None else \
                _gsd_type_names(handle, types_chunk)
            typeid_chunk = self._chunk(f, "particles/typeid")
            typeid = np.zeros(n, dtype=np.int64) if typeid_chunk is None else \
                _gsd_per_particle(handle, typeid_chunk, n, 1, "particles/typeid",
                                  int).reshape(n)
            # A chunk neither this frame nor frame 0 holds takes the schema's
            # default (zero) when other frames hold it, so that every frame
            # carries the same quantities (the gsd library omits a chunk
            # equal to its default).
            velocities = None
            chunk = self._chunk(f, "particles/velocity")
            if chunk is not None:
                velocities = _gsd_per_particle(handle, chunk, n, 3,
                                               "particles/velocity", float) \
                    * self._velocity_factor
            elif "particles/velocity" in self._defaults:
                velocities = np.zeros((n, 3))
            image = None
            if self._unwrap or self._ovito_shifts:
                chunk = self._chunk(f, "particles/image")
                if chunk is not None:
                    image = _gsd_per_particle(handle, chunk, n, 3,
                                              "particles/image", int)
                elif "particles/image" in self._defaults:
                    image = np.zeros((n, 3), dtype=np.int64)
            charge = None
            if self._charges:
                chunk = self._chunk(f, "particles/charge")
                if chunk is not None:
                    charge = _gsd_per_particle(handle, chunk, n, 1,
                                               "particles/charge", float) \
                        .reshape(n)
                elif "particles/charge" in self._defaults:
                    charge = np.zeros(n)
        if image is not None and self._ovito_shifts:
            positions = _undo_ovito_shifts(positions, image, rows)
            image = None
        if typeid.size and (typeid.min() < 0 or typeid.max() >= len(type_names)):
            raise ValueError(f"particles/typeid holds type {int(typeid.max())}, "
                             f"and particles/types names {len(type_names)}")
        lookup = np.full(len(type_names), "", dtype="<U2")
        for t in np.unique(typeid).tolist():
            label = type_names[t]
            if self._strict_labels and label not in self._label_map:
                raise ValueError(
                    f"type name {label!r} has no entry in the type map; the "
                    "file stores no particles/mass and some of its type names "
                    "are not element symbols, so its type names are labels and "
                    "the map needs every one of them")
            lookup[t] = _label_element(self._label_map, label)
        elements = lookup[typeid]
        unwrapped = None
        if image is not None:
            unwrapped = md_model.unwrap_with_images(positions, image, rows)
        time = None
        if self._times is not None and np.isfinite(self._times[k]):
            time = float(self._times[k])
        step = int(self.timesteps[k])
        return frame_from_arrays(
            elements, positions, box_ang=rows, origin_ang=origin,
            timestep=None if step == NO_TIMESTEP else step, time_ps=time,
            unwrapped_cart_ang=unwrapped, vel_ang_per_ps=velocities,
            charge_e=charge)


def _undo_ovito_shifts(positions: np.ndarray, image: np.ndarray,
                       rows: np.ndarray) -> np.ndarray:
    """The positions OVITO was given, from what its GSD exporter wrote.

    GSDExporter.cpp [11] wraps coordinate d alone by its own cell vector,
    ``transformation.prodrow(p - s * cell->matrix().column(dim), dim)``, and
    writes s as the image flag: coordinate d moves by s_d times the d-th
    diagonal element of the HOOMD box, so in a tilted box an atom wrapped
    along c moves by (0, 0, s c_z) rather than by s c. Adding s_d times the
    diagonal back restores the position exactly; frame_from_arrays then wraps
    it by whole box vectors.
    """
    return positions + image * np.diag(rows)[None, :]


def _label_element(label_map: dict, label: str) -> str:
    """The element of a label: from the map built when the file was opened
    (the user's map and frame 0's labels), else the label read as a symbol
    (refused with the reason when it is not one), remembered for later
    frames."""
    if label not in label_map:
        label_map[label] = _mr()._file_symbol(label)
    return label_map[label]


def _gsd_chunk(index: _GsdIndex, f: int, name: str) -> _GsdChunk | None:
    """Frame f's chunk, or frame 0's when frame f has none [5]."""
    chunk = index.frames.get(f, {}).get(name)
    if chunk is None and f != 0:
        chunk = index.frames.get(0, {}).get(name)
    return chunk


def _gsd_per_particle(handle, chunk: _GsdChunk, n: int, m: int, what: str,
                      kind) -> np.ndarray:
    if chunk.n != n or chunk.m != m:
        raise ValueError(f"{what} holds {chunk.n} x {chunk.m} values for {n} "
                         f"particles; {n} x {m} are needed")
    if kind is float and chunk.type_code not in _GSD_FLOAT:
        raise ValueError(f"{what} is stored as {chunk.dtype}, not as floats")
    if kind is int and chunk.type_code not in _GSD_INT:
        raise ValueError(f"{what} is stored as {chunk.dtype}, not as integers")
    values = _gsd_values(handle, chunk, what)
    return values.astype(np.float64 if kind is float else np.int64)


def _gsd_scalar(handle, chunk: _GsdChunk, what: str) -> int:
    if chunk.type_code not in _GSD_INT or chunk.n * chunk.m != 1:
        raise ValueError(f"{what} is {chunk.n} x {chunk.m} {chunk.dtype}, not "
                         "one integer")
    return int(_gsd_values(handle, chunk, what).reshape(-1)[0])


def _gsd_frame_problem(index: _GsdIndex, f: int, n: int) -> str | None:
    """Why frame f cannot be read, from its chunks' shapes alone."""
    if n <= 0:
        return "it holds no particles (particles/N is 0)"
    position = _gsd_chunk(index, f, "particles/position")
    if position is None:
        return "neither it nor frame 0 holds particles/position"
    checks = (("particles/position", 3, _GSD_FLOAT),
              ("particles/typeid", 1, _GSD_INT),
              ("particles/velocity", 3, _GSD_FLOAT),
              ("particles/image", 3, _GSD_INT),
              ("particles/charge", 1, _GSD_FLOAT))
    for name, m, types in checks:
        chunk = _gsd_chunk(index, f, name)
        if chunk is None:
            continue
        where = "frame 0's " if name not in index.frames.get(f, {}) else ""
        if chunk.n != n or chunk.m != m:
            return (f"{where}{name} holds {chunk.n} x {chunk.m} values where the "
                    f"frame has {n} particles ({n} x {m} needed)")
        if chunk.type_code not in types:
            return f"{where}{name} is stored as {chunk.dtype}"
    box = _gsd_chunk(index, f, "configuration/box")
    if box is None:
        return ("neither it nor frame 0 holds configuration/box (the HOOMD "
                "schema's default is a 1 x 1 x 1 box, which FACET does not take "
                "as the model's box)")
    if box.n * box.m != 6 or box.type_code not in _GSD_FLOAT:
        return (f"configuration/box holds {box.n} x {box.m} {box.dtype}, not 6 "
                "floats")
    return None


def _gsd_box_problem(box: np.ndarray) -> str | None:
    """Why (Lx, Ly, Lz, xy, xz, yz) is not a box, or None."""
    if not np.isfinite(box).all():
        return f"configuration/box holds {box.tolist()}"
    if (box[:3] <= 0).any():
        return (f"configuration/box gives the lengths Lx, Ly, Lz as "
                f"{box[:3].tolist()}; a periodic box has three positive lengths")
    return None


def _gsd_inside(positions: np.ndarray, rows: np.ndarray, origin: np.ndarray,
                tol: float) -> int:
    """How many positions lie outside the box (fractional coordinate below
    -tol or at least 1 + tol)."""
    frac = np.linalg.solve(rows.T, (positions - origin).T).T
    return int(((frac < -tol) | (frac >= 1.0 + tol)).any(axis=1).sum())


def _gsd_box_reading(path: Path, index: _GsdIndex, frames: Sequence[int],
                     counts: Mapping[int, int], box6: Sequence[np.ndarray]
                     ) -> tuple[str, list[str]]:
    """'schema' or 'ovito', and the notes that say why (see the module
    docstring, "OVITO's GSD exporter writes another box")."""
    tilted = [float(np.abs(np.asarray(b)[3:6]).max()) > 0 for b in box6]
    if not index.application.lower().startswith("ovito") or not any(tilted):
        return "schema", []
    impossible = [f for f, b in zip(frames, box6)
                  if _box_from_ovito_gsd(b) is None]
    if impossible:
        return "schema", [
            "box read with the HOOMD schema's definition, although the file "
            "names OVITO as its writer: the box numbers of the frame(s) at "
            f"file position(s) {_listed(impossible)} cannot be what OVITO's "
            "GSD exporter (GSDExporter.cpp) writes, (Lx, |b|, |c|, b_x/|b|, "
            "c_x/|c|, c_y/|c|): there |xy|, or the length of (xz, yz), is 1 or "
            "more, which b_x/|b| and (c_x, c_y)/|c| never reach"]
    outside = {"schema": 0, "ovito": 0}
    largest = 0.0
    precision = 0.0
    total = 0
    with open(path, "rb") as handle:
        for f, b in zip(frames, box6):
            rows_s, origin_s = box_from_hoomd(b)
            ovito = _box_from_ovito_gsd(b)
            chunk = _gsd_chunk(index, f, "particles/position")
            positions = _gsd_per_particle(handle, chunk, counts[f], 3,
                                          "particles/position", float)
            # An atom OVITO shifted while wrapping (a non-zero image flag) moved
            # by its box's diagonal, not by box vectors (_undo_ovito_shifts),
            # so only the atoms it did not shift are evidence.
            image_chunk = _gsd_chunk(index, f, "particles/image")
            if image_chunk is not None:
                image = _gsd_per_particle(handle, image_chunk, counts[f], 3,
                                          "particles/image", int)
                positions = positions[~image.any(axis=1)]
            total += len(positions)
            tol = 16 * float(np.finfo(chunk.dtype).eps)
            box_chunk = _gsd_chunk(index, f, "configuration/box")
            precision = max(precision, 4 * float(np.finfo(box_chunk.dtype).eps)
                            * float(np.abs(rows_s).max()))
            outside["schema"] += _gsd_inside(positions, rows_s, origin_s, tol)
            if ovito is None:
                outside["ovito"] += len(positions)
            else:
                outside["ovito"] += _gsd_inside(positions, ovito[0], ovito[1], tol)
                largest = max(largest, float(np.abs(ovito[0] - rows_s).max()))
    if largest <= precision:
        return "schema", []
    source = ("OVITO's GSD exporter (GSDExporter.cpp) writes a tilted box as "
              "(Lx, |b|, |c|, b_x/|b|, c_x/|c|, c_y/|c|), while the HOOMD schema "
              "defines Ly and Lz as the box's diagonal and the tilts as b_x/Ly, "
              "c_x/Lz, c_y/Lz; the two readings of this file's box differ by up "
              f"to {largest:.3g} Å in a box vector")
    counted = (f"{total} atom positions OVITO did not shift while wrapping "
               f"(image flags 0) in the {len(frames)} frame(s)")
    if outside["ovito"] == 0 and outside["schema"] > 0:
        return "ovito", [
            f"box read as OVITO's exporter writes it: {source}. Of the "
            f"{counted}, {outside['schema']} lie outside the box with the "
            "schema's reading, which the schema does not allow and OVITO's "
            "exporter does not write (it wraps every position into its box); "
            "with OVITO's reading none do"]
    if outside["schema"] == 0 and outside["ovito"] > 0:
        return "schema", [
            f"box read with the HOOMD schema's definition, although the file "
            f"names OVITO as its writer: {source}. Of the {counted}, "
            f"{outside['ovito']} lie outside the box with OVITO's reading, none "
            "with the schema's"]
    if outside["schema"] == 0 and outside["ovito"] == 0:
        return "ovito", [
            f"box read as OVITO's exporter writes it: {source}. All {counted} "
            "lie inside both boxes (the two differ only in a thin layer at the "
            "faces), so the positions do not decide; the file names OVITO as "
            "its writer, and this is the box OVITO's exporter writes (OVITO "
            "3.16.1, measured, and its current source). The file does not "
            "record OVITO's version: had a version that writes the schema's "
            "box written it, the box would differ by the amount above"]
    return "ovito", [
        f"box read as OVITO's exporter writes it: {source}. Of the {counted}, "
        f"{outside['schema']} lie outside the box with the schema's reading and "
        f"{outside['ovito']} with OVITO's, so the positions do not decide; the "
        "file names OVITO as its writer, whose exporter writes this box. "
        "Positions outside the box are wrapped into it"]


# The most atoms per Å^3 a GSD frame read in Å may hold before the file is
# refused as being in other length units. A choice. Diamond holds the most
# atoms per unit volume of any known substance (Wikipedia, "Diamond": 3520
# kg/m^3 for pure diamond), 3.520 g/cm^3 / 12.011 g/mol x N_A = 0.176
# atoms/Å^3; lengths in nm read as Å multiply a number density by 1000, and
# a model in reduced units holds about one particle per sigma^3.
_GSD_MAX_ATOMS_PER_ANG3 = 0.5


def _gsd_units(mr, units, name: str) -> tuple[str | None, float, float]:
    """(LAMMPS style or None, length factor to Å, velocity factor to Å/ps)
    for the units argument of read_gsd: 'metal' and 'real' as for a LAMMPS
    dump, and 'nano' (LAMMPS: distance nanometers, velocity
    nanometers/nanosecond, charge multiple of electron charge [14]) for a
    file in nm, converted with scipy.constants."""
    if isinstance(units, str) and units.strip().lower() == "nano":
        return "nano", _NM_TO_ANG, _NM_TO_ANG / _NS_TO_PS
    try:
        style = mr._checked_units(units, "the units argument")
    except ValueError as error:
        raise ValueError(f"{name}: {error}; for a GSD file FACET also reads "
                         "units='nano' (lengths in nm, as mdtraj's GSD writer "
                         "keeps them)") from None
    if style is None:
        return None, 1.0, 1.0
    return style, 1.0, mr._VELOCITY_TO_ANG_PER_PS[style]


def read_gsd(path, *, type_map: Mapping | None = None, units: str | None = None,
             timestep_fs: float | None = None,
             mass_tol_amu: float | None = None) -> Trajectory:
    """A GSD file in the HOOMD schema, as HOOMD-blue and OVITO write it [3-5].

    Elements come from ``particles/types`` (labels; ``type_map`` maps labels
    such as HOOMD's default 'A' to elements). A ``particles/mass`` chunk
    checks a label read as a symbol, and names a label that is not a symbol
    when its mass matches one element alone, within ``mass_tol_amu``; without
    it, a file whose type names include one that is not a symbol needs a map
    for every type name. Lengths are read as Å, and velocities and charges as
    written, unless ``units`` names the LAMMPS style the data came in
    ('metal', 'real', or 'nano' for lengths in nm); a frame 0 denser than
    any substance read in Å is refused. ``timestep_fs`` turns
    ``configuration/step`` into times. A chunk absent from a frame is taken
    from frame 0, as the schema defines, and the notes name the frames that
    repeat frame 0's positions or box. Continuous positions come from
    ``particles/image`` (not for OVITO, which writes only its own wrapping
    shifts there). See the module docstring for OVITO's encoding of a tilted
    box.
    """
    return _guarded(path, lambda: _read_gsd(
        path, type_map=type_map, units=units, timestep_fs=timestep_fs,
        mass_tol_amu=mass_tol_amu))


def _read_gsd(path, *, type_map, units, timestep_fs, mass_tol_amu
              ) -> Trajectory:
    mr = _mr()
    path = mr._existing(path)
    name = path.name
    _refuse_gzip(path, "GSD file")
    tol = _named(name, mr._checked_tol, mr.MASS_TOL_AMU if mass_tol_amu is None
                 else mass_tol_amu)
    user_types, user_labels = _named(name, mr._normalise_type_map, type_map)
    style, length_factor, velocity_factor = _gsd_units(mr, units, name)
    step_fs = _checked_timestep_fs(timestep_fs, name)
    index = _gsd_index(path)
    is_ovito = index.application.lower().startswith("ovito")
    skipped: dict[int, str] = {}
    n_of: dict[int, int] = {}
    steps_of: dict[int, int] = {}
    box_of: dict[int, np.ndarray] = {}
    empty = ("the GSD index lists no chunk for it. The gsd library writes no "
             "chunk for a frame equal to frame 0 in every value, step "
             "included, and a damaged frame number leaves such a gap too; a "
             "frame holding nothing of its own is not read as a copy of "
             "frame 0")
    with open(path, "rb") as handle:
        dims = _gsd_chunk(index, 0, "configuration/dimensions")
        if dims is not None and _named(name, _gsd_scalar, handle, dims,
                                       "configuration/dimensions") == 2:
            raise UnsupportedFormat(
                f"{name}: a two-dimensional model (configuration/dimensions is "
                "2); FACET measures three-dimensional periodic models")
        for f in range(index.n_frames):
            if f in index.problems:
                skipped[f] = "; ".join(index.problems[f])
                continue
            if not index.frames.get(f):
                skipped[f] = empty
                continue
            try:
                chunk = _gsd_chunk(index, f, "particles/N")
                n = 0 if chunk is None else _gsd_scalar(handle, chunk,
                                                        "particles/N")
                problem = _gsd_frame_problem(index, f, n)
                if problem is None:
                    box_chunk = _gsd_chunk(index, f, "configuration/box")
                    box = _gsd_values(handle, box_chunk, "configuration/box")
                    box = box.reshape(6).astype(np.float64)
                    problem = _gsd_box_problem(box)
                    step_chunk = _gsd_chunk(index, f, "configuration/step")
                    step = NO_TIMESTEP if step_chunk is None else \
                        _gsd_scalar(handle, step_chunk, "configuration/step")
                    if problem is None and step >= 2 ** 63:
                        problem = (f"configuration/step is {step}, outside "
                                   "the 64-bit signed range FACET keeps steps "
                                   "in (the schema writes it as uint64)")
            except ValueError as error:
                problem = str(error)
            if problem is not None:
                skipped[f] = problem
                continue
            n_of[f], box_of[f], steps_of[f] = n, box, step
        if 0 in skipped:
            raise ValueError(f"{name}: frame 0 cannot be read ({skipped[0]}); "
                             "later frames take what they do not hold from "
                             "frame 0, so no frame can be read")
        frames = sorted(n_of)
        counts = [n_of[f] for f in frames]
        n0 = mr._majority_count(counts)
        for f in frames:
            if n_of[f] != n0:
                skipped[f] = mr._count_reason(n_of[f], n0, counts)
        frames = [f for f in frames if f not in skipped]
        first = frames[0]
        types_chunk = _gsd_chunk(index, first, "particles/types")
        type_names = ["A"] if types_chunk is None else \
            _gsd_type_names(handle, types_chunk)
        typeid_chunk = _gsd_chunk(index, first, "particles/typeid")
        typeid0 = np.zeros(n0, dtype=np.int64) if typeid_chunk is None else \
            mr._frame0(name, first, lambda: _gsd_per_particle(
                handle, typeid_chunk, n0, 1, "particles/typeid", int).reshape(n0))
        mass_chunk = _gsd_chunk(index, first, "particles/mass")
        masses = None
        if mass_chunk is not None:
            values = mr._frame0(name, first, lambda: _gsd_per_particle(
                handle, mass_chunk, n0, 1, "particles/mass", float).reshape(n0))
            masses = {}
            for t, mass in zip(typeid0.tolist(), values.tolist()):
                if 0 <= t < len(type_names):
                    masses.setdefault(type_names[t], mass)
    if typeid0.size and (typeid0.min() < 0 or typeid0.max() >= len(type_names)):
        raise ValueError(f"{name}, frame {first}: particles/typeid holds type "
                         f"{int(typeid0.max())}, and particles/types names "
                         f"{len(type_names)}")
    used = sorted({type_names[t] for t in typeid0.tolist()})
    notes: list[str] = [
        f"written by {index.application or '(application not recorded)'}, GSD "
        f"file layer {index.version[0]}.{index.version[1]}, schema hoomd "
        f"{index.schema_version[0]}.{index.schema_version[1]}"]

    # Type names. Without particles/mass nothing tells a letter used as a
    # label from an element symbol: when any name in use is not a symbol
    # (HOOMD's 'A'), the names are labels and the map needs all of them, so
    # that 'B' beside 'A' is never read as boron. A name that is not a symbol
    # takes its element from particles/mass when exactly one element's
    # standard atomic weight lies within the tolerance.
    strict_labels = False
    if masses is None:
        not_symbols = [t for t in used if mr._label_problem(t) is not None]
        unmapped = [t for t in used if t not in user_labels]
        if not_symbols and unmapped:
            example = ", ".join(f"{t!r}: '<element>'" for t in used)
            given = (f"; the map given has no entry for {unmapped}"
                     if user_labels else "")
            raise ValueError(
                f"{name}: no element for the type name(s) "
                f"{', '.join(repr(t) for t in unmapped)}. The file stores no "
                "particles/mass to tell a letter used as a label from an "
                "element symbol, and "
                f"{', '.join(repr(t) for t in not_symbols)} "
                f"{'is not an element symbol' if len(not_symbols) == 1 else 'are not element symbols'}"
                f", so all its type names ({', '.join(repr(t) for t in used)}) "
                "are taken as labels: give a type map from every type name in "
                f"use to its element, type_map={{{example}}}{given}")
        strict_labels = bool(not_symbols)
        symbols = [t for t in used if t not in user_labels]
        if symbols and not strict_labels:
            notes.append(
                f"type name(s) {symbols} read as element symbols without a "
                "mass to check them: the file stores no particles/mass (the "
                "gsd library leaves it out when every mass is 1, HOOMD's "
                "default), so a letter used as a label would read as an "
                "element; type_map states the elements")
        if mass_tol_amu is not None:
            notes.append("mass_tol_amu not used: the file stores no "
                         "particles/mass")
    by_mass: dict[str, str] = {}
    if masses is not None:
        for label in used:
            if label in user_labels or label not in masses or \
                    mr._label_problem(label) is None:
                continue
            symbol, detail = mr._match_mass(masses[label], tol)
            if symbol is not None:
                by_mass[label] = symbol
                notes.append(f"type name {label!r} is not an element symbol, "
                             f"and frame 0's particles/mass gives it a {detail}")
    try:
        label_map, listed, _, label_notes = mr._resolve_labels(
            used, {**by_mass, **user_labels}, masses=masses, tol=tol,
            what="type name")
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from None
    sources = {"user" if t in user_labels else
               "data-file masses" if t in by_mass else "file symbols"
               for t in used}
    source = " + ".join(s for s in ("user", "data-file masses", "file symbols")
                        if s in sources)
    notes += label_notes + mr._numeric_entries_note(user_types, "type names")
    if masses is not None:
        notes.append("particles/mass of frame 0 is compared with each type "
                     "name's standard atomic weight (the first mass given for "
                     "it); the frames do not store masses")
    if is_ovito:
        notes.append("GSD stores no particle ids: atoms are numbered 0 .. N-1 "
                     "in file order, the same particles in every frame (OVITO "
                     "3.16.1's GSD export writes them sorted by particle "
                     "identifier, measured)")
    else:
        notes.append("GSD stores no particle ids: atoms are numbered 0 .. N-1 "
                     "in file order, the same particles in every frame "
                     "(HOOMD-blue writes particles in tag order; a converter "
                     "that reordered them between frames could not be "
                     "detected)")
    boxes6 = [box_of[f] for f in frames]
    reading, box_notes = _gsd_box_reading(path, index, frames, n_of, boxes6)
    notes += box_notes
    boxes = []
    for b in boxes6:
        rows, origin = _box_from_ovito_gsd(b) if reading == "ovito" else \
            box_from_hoomd(b)
        boxes.append((rows * length_factor, origin * length_factor))
    notes.append("box origin at -(a + b + c) / 2: the HOOMD schema centres the "
                 "box on zero")
    if reading == "schema" and not is_ovito and \
            float(np.abs(boxes6[0][3:6]).max()) > 0:
        # HOOMD-blue keeps every position inside the centred box; a writer
        # that does not may not follow the schema's tilts either
        with open(path, "rb") as handle:
            chunk = _gsd_chunk(index, first, "particles/position")
            raw = _gsd_per_particle(handle, chunk, n0, 3, "particles/position",
                                    float)
        rows_s, origin_s = box_from_hoomd(boxes6[0])
        outside = _gsd_inside(raw, rows_s, origin_s,
                              16 * float(np.finfo(chunk.dtype).eps))
        if outside:
            notes.append(
                f"{outside} of the {n0} positions of frame 0 lie outside the "
                "box the HOOMD schema defines (centred on zero; HOOMD-blue "
                "keeps every position inside it), so the program that wrote "
                "the file does not follow the schema in full, and the file "
                "does not name it. The tilted box is read with the schema's "
                "tilt factors xy = b_x/Ly, xz = c_x/Lz, yz = c_y/Lz; mdtraj's "
                "GSD writer (mdtraj 1.11, utils/unitcell.py, "
                "lengths_and_angles_to_tilt_factors) writes positions as given "
                "and the tilts as the lengths b_x, c_x, c_y, so for a file it "
                "wrote the box differs from the one mdtraj held in its tilts "
                "(Lx, Ly and Lz agree)")
    volume = abs(float(np.linalg.det(boxes[0][0])))
    density = n0 / volume
    if density > _GSD_MAX_ATOMS_PER_ANG3:
        read_as = ("with units='nano', lengths read in nm" if style == "nano"
                   else "with lengths read as Å")
        raise ValueError(
            f"{name}: {read_as}, frame 0 holds {n0} atoms in {volume:.4g} Å^3, "
            f"{density:.3g} atoms per Å^3, more than {_GSD_MAX_ATOMS_PER_ANG3:g} "
            "(diamond, the substance with the most atoms per volume, holds "
            "0.176). GSD names no unit, and lengths in nm (mdtraj's GSD writer "
            "keeps nm) or in reduced units (HOOMD-blue's own) give such a "
            "number. Pass units='nano' when the lengths are in nm; FACET does "
            "not read reduced units")
    for chunk_name, what in (("particles/position", "positions"),
                             ("configuration/box", "box")):
        repeat = [f for f in frames
                  if f != 0 and chunk_name not in index.frames.get(f, {})]
        if repeat:
            notes.append(
                f"the frame(s) at file position(s) {_listed(repeat)} hold no "
                f"{chunk_name} and repeat frame 0's {what}, as the HOOMD schema "
                "defines (the gsd library leaves a chunk out of a frame when it "
                "equals frame 0's)")
    has_image = any(_gsd_chunk(index, f, "particles/image") is not None
                    for f in frames)
    unwrap = has_image and not is_ovito
    if has_image and is_ovito:
        notes.append(
            "particles/image holds only the shifts OVITO's GSD exporter made "
            "while wrapping (GSDExporter.cpp), not the run's image flags, so no "
            "continuous positions are set. The exporter moves each coordinate "
            "by its own box vector's diagonal element alone, so in a tilted box "
            "a wrapped atom is not moved by whole box vectors; adding those "
            "shifts back restores the positions OVITO held, which are then "
            "wrapped into the box")
    elif unwrap:
        notes.append("continuous positions from particles/image: x + i_x a + "
                     "i_y b + i_z c, the HOOMD schema's definition")
    has_velocity = any(_gsd_chunk(index, f, "particles/velocity") is not None
                       for f in frames)
    has_charge = any(_gsd_chunk(index, f, "particles/charge") is not None
                     for f in frames)
    if style == "nano":
        units_text = ("LAMMPS units nano (the units argument): lengths in nm, "
                      f"x {_NM_TO_ANG:g} to Å (scipy.constants nano / angstrom)"
                      + ("; velocities in nm/ns, x "
                         f"{_NM_TO_ANG / _NS_TO_PS:g} to Å/ps" if has_velocity
                         else "")
                      + ("; charges in e" if has_charge else ""))
    elif style is not None:
        units_text = (f"LAMMPS units {style} (the units argument): lengths in Å"
                      + ("; velocities converted from Å/fs to Å/ps" if
                         style == "real" and has_velocity else "")
                      + ("; charges in e" if has_charge else ""))
    else:
        units_text = ("GSD names no unit (HOOMD's native units are whatever the "
                      "simulation used); lengths are read as Å, as OVITO writes "
                      "data it read from LAMMPS metal or real, VASP or extended "
                      "XYZ files, and a model in nm or in reduced units is not "
                      "in Å (units='nano' reads nm)")
        if has_velocity or has_charge:
            units_text += ("; velocities and charges are stored as written, "
                           "which is Å/ps and e for LAMMPS units metal (Å/fs "
                           "and e under units real); units='metal' or "
                           "units='real' states the style")
    defaults = set()
    for chunk_name in ("particles/velocity", "particles/image",
                       "particles/charge"):
        lacking = [f for f in frames
                   if _gsd_chunk(index, f, chunk_name) is None]
        if lacking and len(lacking) < len(frames):
            defaults.add(chunk_name)
            others = [f for f in lacking if f != 0]
            notes.append(
                f"frame 0 holds no {chunk_name}"
                + (f", nor do the frame(s) at file position(s) "
                   f"{_listed(others)}" if others else "")
                + ": those frames take the HOOMD schema's default, zero (the "
                "gsd library leaves out a chunk equal to its default)")
    unread = set()
    for f in frames:
        unread |= set(index.frames.get(f, {})) - _GSD_READ
    notes += _not_read(sorted(unread), "chunks")
    steps = [steps_of[f] for f in frames]
    times = None
    if step_fs is not None:
        times, note = _times_from_steps(name, steps, step_fs)
        notes.append(note)
    else:
        notes.append("GSD stores the step of each frame, not its time; "
                     "timestep_fs gives the length of a step")
    box_varies = any(not np.array_equal(b[0], boxes[0][0]) or
                     not np.array_equal(b[1], boxes[0][1]) for b in boxes[1:])
    trajectory = _GsdTrajectory(
        path, index, frames, counts=n_of, boxes=boxes,
        label_map={**user_labels, **label_map}, strict_labels=strict_labels,
        length_factor=length_factor, velocity_factor=velocity_factor,
        charges=True, unwrap=unwrap, ovito_shifts=has_image and is_ovito,
        defaults=frozenset(defaults),
        times=None if times is None else np.asarray(times, dtype=float),
        periodic=(True, True, True), file_format="gsd", n_atoms=n0,
        n_frames=len(frames), type_map_source=source, timesteps=steps,
        times_ps=times, type_map=listed, skipped=skipped, notes=notes,
        units_note=units_text, box_varies=box_varies)
    frame0 = mr._frame0(name, first, lambda: trajectory._build(0))
    trajectory._hold_to(frame0)
    if frame0.charge_e is not None:
        charges, charge_notes = md_model.charges_per_element(frame0.elements,
                                                             frame0.charge_e)
        trajectory.charges_e = charges
        trajectory.notes.extend(charge_notes)
    return mr._finish(trajectory, frame0)


# ---------------------------------------------------------------------------
# NetCDF: classic, 64-bit offset and CDF-5 headers
# ---------------------------------------------------------------------------

_CDF_VERSIONS = {b"CDF\x01": 1, b"CDF\x02": 2, b"CDF\x05": 5}
_CDF_NAMES = {1: "classic (CDF-1)", 2: "64-bit offset (CDF-2)", 5: "CDF-5"}
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_NC_DIMENSION, _NC_VARIABLE, _NC_ATTRIBUTE = 10, 11, 12
# nc_type codes [6, 7]: big-endian on disk
_NC_TYPES = {1: ">i1", 2: "S1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8",
             7: "u1", 8: ">u2", 9: ">u4", 10: ">i8", 11: ">u8"}
# NC_FILL_FLOAT and NC_FILL_DOUBLE, include/netcdf.h [6]
_NC_DEFAULT_FILL = {5: float(np.float32(9.9692099683868690e+36)),
                    6: 9.9692099683868690e+36}
# How much of a file is read for its header at first, and at most. A
# performance choice: AMBER-convention headers are a few kB.
_NC_HEADER_FIRST_READ = 1 << 20
_NC_HEADER_MAX = 1 << 26
_NC_EXTENSIONS = (".nc", ".ncdf", ".netcdf", ".ncrst")
# The dimension the AMBER convention counts frames along.
_NC_FRAME = "frame"


@dataclass
class NetcdfVariable:
    """One variable of a NetCDF classic / CDF-5 header.

    ``framed`` is True for a variable whose first dimension is the record
    (unlimited) dimension, or a fixed dimension named 'frame'; its values for
    frame k are one slab of ``shape[1:]``.
    """

    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]           # the record dimension counts as 0 here
    nc_type: int
    attributes: dict
    begin: int
    is_record: bool

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(_NC_TYPES[self.nc_type])

    @property
    def framed(self) -> bool:
        return self.is_record or (bool(self.dimensions)
                                  and self.dimensions[0] == _NC_FRAME)

    @property
    def per_frame_dimensions(self) -> tuple[str, ...]:
        return self.dimensions[1:] if self.framed else self.dimensions

    @property
    def slab_count(self) -> int:
        """Values in one frame (a framed variable) or in the whole variable."""
        dims = self.shape[1:] if self.framed else self.shape
        return math.prod(dims)          # Python integers: no int64 overflow


@dataclass
class NetcdfHeader:
    """A NetCDF classic (CDF-1), 64-bit offset (CDF-2) or CDF-5 header [6, 7].

    ``numrecs`` is the record count the header gives, or, when it gives the
    streaming marker (``streaming`` True), the complete records the file holds.
    ``record_size`` is the stride from one record to the next.
    """

    version: int
    numrecs: int
    streaming: bool
    dimensions: dict[str, int]       # the record dimension has length 0
    record_dimension: str | None
    attributes: dict
    variables: dict[str, NetcdfVariable]
    record_size: int
    file_size: int
    header_bytes: int


class _Short(Exception):
    """The header runs past the bytes read."""


class _Cursor:
    def __init__(self, data: bytes, version: int) -> None:
        self.data = data
        self.pos = 4
        self.wide = version == 5
        self.offset64 = version in (2, 5)

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise _Short
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def int32(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def nonneg(self, what: str) -> int:
        value = struct.unpack(">q" if self.wide else ">i",
                              self.take(8 if self.wide else 4))[0]
        if value < 0:
            raise ValueError(f"{what} is {value}, a negative count")
        return value

    def offset(self) -> int:
        return struct.unpack(">q" if self.offset64 else ">i",
                             self.take(8 if self.offset64 else 4))[0]

    def pad(self, n: int) -> None:
        self.take((-n) % 4)

    def name(self) -> str:
        n = self.nonneg("a name length")
        raw = self.take(n)
        self.pad(n)
        return raw.decode("utf-8", "replace")

    def attributes(self) -> dict:
        tag = self.int32()
        count = self.nonneg("an attribute count")
        if tag == 0 and count == 0:
            return {}
        if tag != _NC_ATTRIBUTE:
            raise ValueError(f"an attribute list starts with tag {tag}, not "
                             f"{_NC_ATTRIBUTE} (NC_ATTRIBUTE)")
        out = {}
        for _ in range(count):
            key = self.name()
            nc_type = self.int32()
            if nc_type not in _NC_TYPES:
                raise ValueError(f"attribute {key!r} has the unknown type "
                                 f"{nc_type}")
            n = self.nonneg(f"attribute {key!r}'s length")
            dtype = np.dtype(_NC_TYPES[nc_type])
            raw = self.take(n * dtype.itemsize)
            self.pad(n * dtype.itemsize)
            if nc_type == 2:
                out[key] = raw.split(b"\0")[0].decode("utf-8", "replace")
            else:
                out[key] = np.frombuffer(raw, dtype=dtype).astype(
                    dtype.newbyteorder("="))
        return out


def _netcdf_parse(data: bytes, version: int, size: int) -> NetcdfHeader:
    cur = _Cursor(data, version)
    raw = cur.take(8 if version == 5 else 4)
    streaming = raw == b"\xff" * len(raw)
    numrecs = 0 if streaming else \
        struct.unpack(">q" if version == 5 else ">i", raw)[0]
    if numrecs < 0:
        raise ValueError(f"the record count is {numrecs}")
    tag, count = cur.int32(), cur.nonneg("the dimension count")
    dims: list[tuple[str, int]] = []
    if not (tag == 0 and count == 0):
        if tag != _NC_DIMENSION:
            raise ValueError(f"the dimension list starts with tag {tag}, not "
                             f"{_NC_DIMENSION} (NC_DIMENSION)")
        for _ in range(count):
            dim_name = cur.name()
            dims.append((dim_name, cur.nonneg(f"dimension {dim_name!r}")))
    attributes = cur.attributes()
    tag, count = cur.int32(), cur.nonneg("the variable count")
    variables: dict[str, NetcdfVariable] = {}
    record = next((n for n, length in dims if length == 0), None)
    if not (tag == 0 and count == 0):
        if tag != _NC_VARIABLE:
            raise ValueError(f"the variable list starts with tag {tag}, not "
                             f"{_NC_VARIABLE} (NC_VARIABLE)")
        for _ in range(count):
            var_name = cur.name()
            ndims = cur.nonneg(f"variable {var_name!r}'s rank")
            ids = [cur.nonneg(f"variable {var_name!r}'s dimension id")
                   for _ in range(ndims)]
            if any(i >= len(dims) for i in ids):
                raise ValueError(f"variable {var_name!r} refers to dimension "
                                 f"id {max(ids)} of {len(dims)}")
            vatts = cur.attributes()
            nc_type = cur.int32()
            if nc_type not in _NC_TYPES:
                raise ValueError(f"variable {var_name!r} has the unknown type "
                                 f"{nc_type}")
            cur.nonneg(f"variable {var_name!r}'s vsize")
            begin = cur.offset()
            if begin < 0:
                raise ValueError(f"variable {var_name!r} begins at byte {begin}")
            names = tuple(dims[i][0] for i in ids)
            shape = tuple(dims[i][1] for i in ids)
            if record is not None and record in names[1:]:
                raise ValueError(f"variable {var_name!r} uses the record "
                                 "dimension after its first dimension")
            variables[var_name] = NetcdfVariable(
                var_name, names, shape, nc_type, vatts, begin,
                bool(names) and names[0] == record)
    records = [v for v in variables.values() if v.is_record]
    if len(records) == 1:
        # one record variable is packed without padding (netCDF-C)
        record_size = records[0].slab_count * records[0].dtype.itemsize
    else:
        record_size = sum(-(-v.slab_count * v.dtype.itemsize // 4) * 4
                          for v in records)
    if streaming:
        first = min((v.begin for v in records), default=size)
        numrecs = 0 if record_size == 0 else max(0, (size - first) // record_size)
    return NetcdfHeader(version, numrecs, streaming, dict(dims), record,
                        attributes, variables, record_size, size, cur.pos)


def netcdf_header(path) -> NetcdfHeader:
    """The header of a NetCDF classic, 64-bit offset or CDF-5 file [6, 7].

    Refuses a NetCDF-4 (HDF5) file with the command that converts it, and a
    file whose header does not parse, naming the file.
    """
    path = Path(path)
    name = path.name
    size = path.stat().st_size
    with open(path, "rb") as handle:
        data = handle.read(min(size, _NC_HEADER_FIRST_READ))
        if data[:8] == _HDF5_MAGIC:
            raise UnsupportedFormat(
                f"{name}: a NetCDF-4 file (HDF5 underneath). FACET reads NetCDF "
                "classic, 64-bit offset and CDF-5 files; convert it with the "
                f"netCDF utilities: nccopy -k cdf5 {name} converted.nc")
        version = _CDF_VERSIONS.get(data[:4])
        if version is None:
            raise UnsupportedFormat(f"{name}: not a NetCDF file (its first "
                                    f"bytes are {data[:4]!r}). {_READS}")
        for attempt in (0, 1):
            try:
                return _netcdf_parse(data, version, size)
            except _Short:
                if len(data) >= _NC_HEADER_MAX:
                    raise ValueError(
                        f"{name}: the NetCDF header is longer than the "
                        f"{_NC_HEADER_MAX} bytes FACET reads of a header (AMBER-"
                        "convention headers are a few kB; a large attribute "
                        "makes it longer). Copy the file without its large "
                        "attributes (ncks or nccopy from the netCDF utilities) "
                        "and open the copy") from None
                if len(data) >= size:
                    raise ValueError(
                        f"{name}: the NetCDF header runs past the end of the "
                        f"file ({size} bytes); the file was cut") from None
                handle.seek(0)
                data = handle.read(min(size, _NC_HEADER_MAX))
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"{name}: the NetCDF header does not parse "
                                 f"({error})") from None
    raise ValueError(f"{name}: the NetCDF header does not parse")


def _nc_offset(header: NetcdfHeader, var: NetcdfVariable, k: int) -> int:
    if var.is_record:
        return var.begin + k * header.record_size
    if var.framed:
        return var.begin + k * var.slab_count * var.dtype.itemsize
    return var.begin


def _nc_values(handle, header: NetcdfHeader, var: NetcdfVariable, k: int
               ) -> np.ndarray:
    """Variable ``var`` in frame k (a framed variable) or whole, shaped
    ``shape[1:]`` or ``shape``, numbers in native byte order (text as S1)."""
    nbytes = var.slab_count * var.dtype.itemsize
    data = _read_exact(handle, _nc_offset(header, var, k), nbytes, var.name)
    shape = var.shape[1:] if var.framed else var.shape
    values = np.frombuffer(data, dtype=var.dtype).reshape(shape)
    if var.nc_type == 2:
        return values
    return values.astype(var.dtype.newbyteorder("="))


def _nc_text_rows(values: np.ndarray, n: int) -> list[str]:
    rows = np.asarray(values).reshape(n, -1)
    return [b"".join(r.tolist()).split(b"\0")[0].decode("utf-8", "replace")
            .strip() for r in rows]


def _attr_text(var: NetcdfVariable | None, key: str) -> str | None:
    if var is None:
        return None
    value = var.attributes.get(key)
    return value if isinstance(value, str) else None


def _attr_number(var: NetcdfVariable, key: str) -> float | None:
    value = var.attributes.get(key)
    if value is None or isinstance(value, str) or np.asarray(value).size != 1:
        return None
    return float(np.asarray(value).reshape(-1)[0])


def _fill_value(var: NetcdfVariable) -> float | None:
    value = _attr_number(var, "_FillValue")
    return _NC_DEFAULT_FILL.get(var.nc_type) if value is None else value


def _unit_factor(text: str | None, table: Mapping[str, float]) -> float | None:
    if text is None:
        return None
    return table.get(text.strip().lower())


def _velocity_unit_factor(text: str) -> float | None:
    parts = text.replace(" per ", "/").split("/")
    if len(parts) != 2:
        return None
    length = _unit_factor(parts[0], _LENGTH_UNITS)
    time = _unit_factor(parts[1], _TIME_UNITS)
    if length is None or time is None:
        return None
    return length / time


def box_from_cell_parameters(lengths, angles_deg) -> np.ndarray:
    """Box rows from (a, b, c) and (alpha, beta, gamma) in degrees.

    a along x, b in the xy plane: a = (a, 0, 0), b = (b cos g, b sin g, 0),
    c = (c cos be, c (cos al - cos be cos g) / sin g, c_z), with c_z from
    |c|. A right angle's cosine is taken as exactly 0, so a rectangular box
    has exact zeros off the diagonal.
    """
    a, b, c = (float(x) for x in np.asarray(lengths).reshape(3))
    alpha, beta, gamma = (float(x) for x in np.asarray(angles_deg).reshape(3))

    def cos(angle):
        return 0.0 if angle == 90.0 else math.cos(math.radians(angle))

    ca, cb, cg = cos(alpha), cos(beta), cos(gamma)
    sg = 1.0 if gamma == 90.0 else math.sin(math.radians(gamma))
    if sg <= 0:
        raise ValueError(f"the cell angle gamma {gamma:g} deg does not form a "
                         "cell")
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError(f"the cell angles {alpha:g}, {beta:g}, {gamma:g} deg "
                         "do not form a cell")
    return np.array([[a, 0.0, 0.0], [b * cg, b * sg, 0.0],
                     [cx, cy, math.sqrt(cz2)]])


# ---------------------------------------------------------------------------
# AMBER NetCDF trajectories
# ---------------------------------------------------------------------------

_NC_POSITIONS = ("coordinates", "unwrapped_coordinates", "scaled_coordinates")
_NC_IDS = ("id", "identifier")
_NC_CHARGES = ("charge", "charges", "Charge", "q")
_NC_LABELS = ("element", "species")
_NC_DESCRIPTIVE = {"spatial", "cell_spatial", "cell_angular"}


@dataclass
class _NcPlan:
    """What a NetCDF trajectory's frames are built from."""

    n: int
    positions: NetcdfVariable
    position_kind: str                      # 'cart', 'unwrapped', 'scaled'
    length_factor: float
    unwrapped: NetcdfVariable | None
    lengths: NetcdfVariable
    angles: NetcdfVariable
    cell_factor: float
    origin: NetcdfVariable | None
    origin_factor: float
    velocities: NetcdfVariable | None
    velocity_factor: float
    ids: NetcdfVariable | None
    charges: NetcdfVariable | None
    identity: str                   # 'labels', 'numbers', 'types', 'topology'
    identity_var: NetcdfVariable | None
    user_types: dict = field(default_factory=dict)
    label_map: dict = field(default_factory=dict)
    by_id: dict | None = None               # topology: id -> element
    by_row: np.ndarray | None = None        # topology: element per row


class _NetcdfTrajectory(_BinaryTrajectory):
    def __init__(self, path: Path, header: NetcdfHeader, records: Sequence[int],
                 plan: _NcPlan, *, times, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self._header = header
        self._records = list(records)
        self._plan = plan
        self._times = times

    def _build(self, k: int) -> Frame:
        step = int(self.timesteps[k])
        time = None
        if self._times is not None and np.isfinite(self._times[k]):
            time = float(self._times[k])
        return _nc_frame(self._path, self._header, self._plan,
                         self._records[k], None if step == NO_TIMESTEP else step,
                         time)


def _nc_scaled(handle, header: NetcdfHeader, var: NetcdfVariable, record: int,
               shape: tuple[int, ...], factor: float = 1.0) -> np.ndarray:
    """Values as float64 times the variable's scale_factor and ``factor``."""
    values = _nc_values(handle, header, var, record).reshape(shape) \
        .astype(np.float64)
    scale = _attr_number(var, "scale_factor")
    return values * ((1.0 if scale is None else scale) * factor)


def _nc_scale_problem(var: NetcdfVariable) -> str | None:
    """Why a variable's scale_factor cannot scale it (zero, not finite, or
    not one number), or None."""
    if "scale_factor" not in var.attributes:
        return None
    value = var.attributes["scale_factor"]
    if isinstance(value, str) or np.asarray(value).size != 1:
        return f"{var.name}'s scale_factor is {value!r}, not one number"
    number = float(np.asarray(value).reshape(-1)[0])
    if not math.isfinite(number) or number == 0:
        return (f"{var.name}'s scale_factor is {number:g}, which would make "
                "every value " + ("0" if number == 0 else "not finite"))
    return None


def _nc_integers(handle, header: NetcdfHeader, var: NetcdfVariable,
                 record: int, n: int) -> np.ndarray:
    values = _nc_values(handle, header, var, record).reshape(n)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{var.name} is stored as {var.dtype} (text), not as "
                         "numbers")
    if values.dtype.kind not in "iu":
        if not np.isfinite(values).all() or (values != np.round(values)).any():
            raise ValueError(f"{var.name} holds values that are not whole "
                             "numbers")
    return values.astype(np.int64)


def _nc_elements(handle, header: NetcdfHeader, plan: _NcPlan, record: int,
                 ids: np.ndarray | None) -> np.ndarray:
    n = plan.n
    if plan.identity == "topology":
        if plan.by_id is not None and ids is not None:
            missing = sorted(set(ids.tolist()) - set(plan.by_id))
            if missing:
                raise ValueError(
                    f"atom ids {missing[:5]} are not among the topology's ids "
                    f"({min(plan.by_id)} to {max(plan.by_id)}); a topology "
                    "file needs the same atom ids, or topology can list one "
                    "element per atom")
            return np.array([plan.by_id[i] for i in ids.tolist()], dtype="<U2")
        return plan.by_row
    if plan.identity == "labels":
        labels = _nc_text_rows(_nc_values(handle, header, plan.identity_var,
                                          record), n)
        table = {label: _label_element(plan.label_map, label)
                 for label in set(labels)}
        return np.array([table[label] for label in labels], dtype="<U2")
    values = _nc_integers(handle, header, plan.identity_var, record, n)
    if plan.identity == "numbers":
        return _symbols_from_numbers(values, plan.identity_var.name)
    missing = sorted(set(values.tolist()) - set(plan.user_types))
    if missing:
        raise ValueError(
            f"type(s) {missing} ({plan.identity_var.name} variable) have no "
            "element in the type map")
    return np.array([plan.user_types[int(t)] for t in values.tolist()],
                    dtype="<U2")


def _nc_frame(path: Path, header: NetcdfHeader, plan: _NcPlan, record: int,
              step: int | None, time: float | None) -> Frame:
    n = plan.n
    with open(path, "rb") as handle:
        raw = _nc_values(handle, header, plan.positions, record)
        fill = _fill_value(plan.positions)
        if fill is not None and (raw == fill).any():
            raise ValueError(f"{plan.positions.name} holds the netCDF fill value "
                             f"{fill:g} (a record the writer had not filled)")
        positions = _nc_scaled(handle, header, plan.positions, record, (n, 3))
        lengths = _nc_scaled(handle, header, plan.lengths, record, (3,),
                             plan.cell_factor)
        angles = _nc_scaled(handle, header, plan.angles, record, (3,))
        if (lengths <= 0).any() or not np.isfinite(lengths).all():
            raise ValueError(
                f"cell_lengths is {lengths.tolist()}: a length of 0 marks a "
                "direction that is not periodic (ASE writes 0 there), and FACET "
                "reads periodic models only")
        rows = box_from_cell_parameters(lengths, angles)
        origin = None if plan.origin is None else _nc_scaled(
            handle, header, plan.origin, record, (3,), plan.origin_factor)
        ids = None if plan.ids is None else \
            _nc_integers(handle, header, plan.ids, record, n)
        elements = _nc_elements(handle, header, plan, record, ids)
        unwrapped = None if plan.unwrapped is None else _nc_scaled(
            handle, header, plan.unwrapped, record, (n, 3), plan.length_factor)
        velocities = None if plan.velocities is None else _nc_scaled(
            handle, header, plan.velocities, record, (n, 3),
            plan.velocity_factor)
        charges = None if plan.charges is None else _nc_scaled(
            handle, header, plan.charges, record, (n,))
    common = dict(box_ang=rows, origin_ang=origin, atom_id=ids, timestep=step,
                  time_ps=time, unwrapped_cart_ang=unwrapped,
                  vel_ang_per_ps=velocities, charge_e=charges)
    if plan.position_kind == "scaled":
        return frame_from_arrays(elements, frac=positions, **common)
    return frame_from_arrays(elements, positions * plan.length_factor, **common)


def _nc_var(header: NetcdfHeader, name: str, per_frame: Sequence[str],
            path_name: str) -> NetcdfVariable | None:
    """The variable, None when absent; refused when its dimensions per frame
    are not ``per_frame``."""
    var = header.variables.get(name)
    if var is None:
        return None
    if tuple(var.per_frame_dimensions) != tuple(per_frame):
        raise ValueError(
            f"{path_name}: variable {name!r} has dimensions {var.dimensions}; "
            f"the AMBER convention gives it (frame, {', '.join(per_frame)})")
    return var


def _nc_atom_var(header: NetcdfHeader, names: Sequence[str], *,
                 types: Sequence[int] | None = None) -> NetcdfVariable | None:
    """The first of ``names`` holding one value per atom (per frame or for
    the file), of one of ``types`` when given."""
    for candidate in names:
        var = header.variables.get(candidate)
        if var is None or var.per_frame_dimensions != ("atom",):
            continue
        if types is None or var.nc_type in types:
            return var
    return None


_TOPOLOGY_TAKES = (
    "topology takes an MD file FACET reads, a trajectory read_trajectory "
    "returned (to read the topology file with options of its own, such as "
    "type_map, pass topology=read_trajectory(<file>, type_map=...)), or one "
    "element symbol per atom")


def _topology_label(topology) -> str:
    if isinstance(topology, (str, Path)):
        return Path(topology).name
    if isinstance(topology, Trajectory):
        return f"<trajectory of {Path(str(topology.source_path)).name}>"
    try:
        return f"<{type(topology).__name__} of {len(topology)}>"
    except TypeError:
        return f"<{type(topology).__name__}>"


def _topology_elements(topology, n: int, name: str):
    """(by_id, by_row, description) from an MD file, a Trajectory, or one
    element symbol per atom (a list, a tuple or a numpy array, as
    MDAnalysis's ``atoms.elements``). Every refusal names this file and the
    topology argument."""
    where = f"{name}: topology={_topology_label(topology)}"
    if isinstance(topology, (str, Path, Trajectory)):
        try:
            if isinstance(topology, Trajectory):
                first = topology.frame(0)
            else:
                source = _mr().read_trajectory(topology)
                try:
                    first = source.frame(0)
                finally:
                    close = getattr(source, "close", None)
                    if close is not None:
                        close()
        except (ValueError, OSError) as error:
            raise ValueError(f"{where}: {error} ({_TOPOLOGY_TAKES})") \
                from error
        label = _topology_label(topology)
        if first.n_atoms != n:
            raise ValueError(
                f"{name}: the topology {label} holds {first.n_atoms} atoms, "
                f"and this file {n}")
        return (dict(zip(first.atom_id.tolist(), first.elements.tolist())),
                np.asarray(first.elements, dtype="<U2"),
                f"the topology argument (frame 0 of {label})")
    if isinstance(topology, (Mapping, bytes)) or \
            not isinstance(topology, (Sequence, np.ndarray)):
        raise ValueError(f"{where}: {_TOPOLOGY_TAKES}")
    symbols = []
    for i, entry in enumerate(np.asarray(topology, dtype=object).reshape(-1)
                              if isinstance(topology, np.ndarray)
                              else topology):
        if not isinstance(entry, str):
            raise ValueError(f"{where}: entry {i} is {entry!r}, not an element "
                             f"symbol; {_TOPOLOGY_TAKES}")
        try:
            symbols.append(validate_symbol(entry))
        except ValueError:
            raise ValueError(f"{where}: entry {i}, {entry!r}, is not an "
                             f"element symbol gemmi knows; {_TOPOLOGY_TAKES}") \
                from None
    if len(symbols) != n:
        raise ValueError(f"{name}: topology lists {len(symbols)} elements for "
                         f"{n} atoms")
    return None, np.array(symbols, dtype="<U2"), \
        "the topology argument (one symbol per atom, in row order)"


def _nc_records_held(header: NetcdfHeader, var: NetcdfVariable,
                     total: int) -> int:
    """How many of records 0 .. total-1 hold ``var``'s values inside the
    file, computed from the offsets (a damaged record count costs nothing)."""
    nbytes = var.slab_count * var.dtype.itemsize
    if var.begin + nbytes > header.file_size:
        return 0
    if not var.framed:
        return total
    stride = header.record_size if var.is_record else nbytes
    if stride == 0:
        return total
    return min(total, (header.file_size - var.begin - nbytes) // stride + 1)


def _nc_per_frame(header: NetcdfHeader, var: NetcdfVariable | None,
                  framed_file: bool) -> tuple[NetcdfVariable | None, str | None]:
    """A time or step variable when it gives one number per frame (per
    record, or one number for a restart's single frame), else (None, why)."""
    if var is None:
        return None, None
    if var.dtype.kind not in "iuf":
        return None, f"{var.name} is stored as {var.dtype} (text), not numbers"
    if framed_file:
        if var.framed and var.per_frame_dimensions == ():
            return var, None
        return None, (f"{var.name} has dimensions {var.dimensions}, not one "
                      "value per frame (frame)")
    if var.slab_count == 1:
        return var, None
    return None, (f"{var.name} has dimensions {var.dimensions}, not the one "
                  "value of a restart's single frame")


def read_amber_netcdf(path, *, type_map: Mapping | None = None,
                      timestep_fs: float | None = None,
                      topology=None) -> Trajectory:
    """An AMBER-convention NetCDF trajectory or restart [6-9, 13].

    CDF-1, CDF-2 and CDF-5 files are read; NetCDF-4 (HDF5) is refused with
    the conversion. Elements come, in this order, from an ``element`` /
    ``species`` text variable (labels; ``type_map`` maps labels), from atomic
    numbers (``Z``, or ``atom_types`` when ASE wrote the file), from numeric
    ``type`` / ``atom_types`` through ``type_map``, or, for a file that names
    none, from ``topology``: another MD file FACET reads (matched by atom id
    when this file has ids, else by row), a trajectory ``read_trajectory``
    returned, or one element symbol per atom in the order of the returned
    rows (a list or a numpy array). A restart's single frame takes its time
    from a scalar ``time`` (or one of length 1). LAMMPS's ``time`` holds the
    step, and its ``scale_factor`` the step length. ``timestep_fs`` gives
    times to a file with steps and no time.
    """
    return _guarded(path, lambda: _read_amber_netcdf(
        path, type_map=type_map, timestep_fs=timestep_fs, topology=topology))


def _read_amber_netcdf(path, *, type_map, timestep_fs, topology) -> Trajectory:
    mr = _mr()
    path = mr._existing(path)
    name = path.name
    _refuse_gzip(path, "NetCDF file")
    step_fs = _checked_timestep_fs(timestep_fs, name)
    user_types, user_labels = _named(name, mr._normalise_type_map, type_map)
    header = netcdf_header(path)
    attrs = header.attributes
    conventions = str(attrs.get("Conventions", ""))
    if "AMBER" not in conventions.replace(",", " ").upper():
        raise UnsupportedFormat(
            f"{name}: a NetCDF file whose Conventions attribute is "
            f"{conventions!r}, not AMBER; FACET reads NetCDF trajectories that "
            "follow AMBER's convention (coordinates, cell_lengths, cell_angles), "
            "as AMBER, LAMMPS (dump netcdf), OVITO, ASE and MDAnalysis write "
            "them")
    program = str(attrs.get("program", "")).strip()
    is_ase = program == "ASE"
    is_ovito = program.upper().startswith("OVITO")
    is_lammps = program == "LAMMPS"
    n = int(header.dimensions.get("atom", 0))
    if n <= 0:
        raise ValueError(f"{name}: the file has no 'atom' dimension (or it is "
                         "0), so it holds no atoms")
    for dim in ("spatial", "cell_spatial", "cell_angular"):
        length = header.dimensions.get(dim)
        if length is not None and length != 3:
            raise ValueError(
                f"{name}: the dimension {dim!r} has length {length}; the AMBER "
                "convention gives it 3 (x, y, z; a, b, c; alpha, beta, "
                "gamma), so the coordinates or the cell cannot be read")
    per_atom3 = ("atom", "spatial")
    position_name = next((v for v in _NC_POSITIONS if v in header.variables),
                         None)
    if position_name is None:
        raise ValueError(f"{name}: the file holds no coordinates variable "
                         f"({', '.join(_NC_POSITIONS)}), so no positions")
    positions = _nc_var(header, position_name, per_atom3, name)
    kind = {"coordinates": "cart", "unwrapped_coordinates": "unwrapped",
            "scaled_coordinates": "scaled"}[position_name]
    version = f"{program} {attrs.get('programVersion', '')}".strip()
    notes: list[str] = [
        f"written by {version or '(program not recorded)'}; NetCDF "
        f"{_CDF_NAMES[header.version]}; Conventions {conventions}"]

    # lengths
    length_factor = 1.0
    if kind != "scaled":
        unit = _attr_text(positions, "units")
        if unit is None:
            notes.append(f"{position_name} has no units attribute: read as Å, "
                         "the AMBER convention's unit")
        else:
            factor = _unit_factor(unit, _LENGTH_UNITS)
            if factor is None and is_ase and \
                    unit.strip().lower() == "angstrom/femtosecond":
                factor = 1.0
                notes.append(
                    "coordinates carry units = 'Angstrom/Femtosecond', which "
                    "ASE's NetCDFTrajectory sets on the coordinates when it "
                    "adds velocities (netcdftrajectory.py, _add_velocities); "
                    "they are read as Å")
            if factor is None:
                raise ValueError(f"{name}: {position_name} is in {unit!r}; "
                                 "FACET reads lengths in Å or nm")
            length_factor = factor
            if factor != 1.0:
                notes.append(f"lengths converted from {unit} to Å")
    unwrapped = None
    if kind == "cart" and "unwrapped_coordinates" in header.variables:
        unwrapped = _nc_var(header, "unwrapped_coordinates", per_atom3, name)
        notes.append("continuous positions from unwrapped_coordinates")
    elif kind == "unwrapped":
        unwrapped = positions
        notes.append("positions and continuous positions from "
                     "unwrapped_coordinates")
    if positions.dtype.itemsize == 4:
        notes.append(f"{position_name} is stored in single precision (float32, "
                     "about 7 significant digits: 1e-6 Å at 10 Å)")
    lengths = _nc_var(header, "cell_lengths", ("cell_spatial",), name)
    angles = _nc_var(header, "cell_angles", ("cell_angular",), name)
    if lengths is None or angles is None:
        raise ValueError(
            f"{name}: the file states no periodic box (no cell_lengths and "
            "cell_angles); FACET reads periodic models only")
    cell_unit = _attr_text(lengths, "units")
    cell_factor = 1.0 if cell_unit is None else \
        _unit_factor(cell_unit, _LENGTH_UNITS)
    if cell_factor is None:
        raise ValueError(f"{name}: cell_lengths is in {cell_unit!r}; FACET "
                         "reads lengths in Å or nm")
    angle_unit = _attr_text(angles, "units")
    if angle_unit is not None and angle_unit.strip().lower() not in \
            ("degree", "degrees", "deg"):
        raise ValueError(f"{name}: cell_angles is in {angle_unit!r}; the "
                         "AMBER convention and FACET use degrees")
    origin = _nc_var(header, "cell_origin", ("cell_spatial",), name)
    origin_factor = 1.0
    if origin is not None:
        unit = _attr_text(origin, "units")
        origin_factor = 1.0 if unit is None else _unit_factor(unit,
                                                              _LENGTH_UNITS)
        if origin_factor is None:
            raise ValueError(f"{name}: cell_origin is in {unit!r}; FACET reads "
                             "lengths in Å or nm")
        notes.append("box origin from cell_origin (written by LAMMPS, OVITO "
                     "and ASE; not part of the AMBER convention)")
    else:
        notes.append("no cell_origin: the box origin is (0, 0, 0)")

    # velocities
    velocities = None
    velocity_factor = 1.0
    if "velocities" in header.variables:
        velocities = _nc_var(header, "velocities", per_atom3, name)
        unit = _attr_text(velocities, "units")
        if is_ase:
            velocity_factor = ASE_VELOCITY_TO_ANG_PER_PS
            notes.append(
                "velocities in ASE's unit of velocity: ASE's NetCDFTrajectory "
                "writes momenta / masses there (netcdftrajectory.py), "
                f"converted with x {ASE_VELOCITY_TO_ANG_PER_PS:.10g} to Å/ps "
                "(ase/units.py, CODATA 2014)")
        elif unit is None:
            notes.append("velocities have no units attribute: read as Å/ps, "
                         "the AMBER convention's unit")
        else:
            factor = _velocity_unit_factor(unit)
            if factor is None:
                notes.append(f"velocities not read: their unit {unit!r} is not "
                             "a length over a time FACET converts")
                velocities = None
            else:
                velocity_factor = factor
                if factor != 1.0:
                    notes.append(f"velocities converted from {unit} to Å/ps")
        scale = None if velocities is None else \
            _attr_number(velocities, "scale_factor")
        if scale is not None and scale != 1.0:
            notes.append(f"velocities multiplied by their scale_factor "
                         f"{scale:g}, as the AMBER convention directs")

    ids = _nc_atom_var(header, _NC_IDS)
    if ids is None:
        notes.append("no id variable: atoms are numbered 0 .. N-1 in file "
                     "order, the same atoms in every frame")
    else:
        notes.append(f"atom ids from {ids.name}")
    charges = _nc_atom_var(header, _NC_CHARGES, types=(5, 6))
    if charges is not None:
        notes.append(f"per-atom charges from {charges.name}, taken as e (the "
                     "file names no charge unit)")

    # elements
    label_var = next((v for v in (header.variables.get(x) for x in _NC_LABELS)
                      if v is not None and v.nc_type == 2
                      and len(v.per_frame_dimensions) == 2
                      and v.per_frame_dimensions[0] == "atom"), None)
    numbers_var = _nc_atom_var(header, ("Z",) + (("atom_types",) if is_ase
                                                 else ()))
    types_var = None if is_ase else _nc_atom_var(header, ("type", "atom_types"))
    named_by = label_var or numbers_var or types_var
    if topology is not None and named_by is not None:
        raise ValueError(f"{name}: the file names its atoms ({named_by.name} "
                         "variable); topology is for a file that names none")
    source, listed = "user", {}
    by_id = by_row = None
    # A missing type map is reported after the cell checks below: a file
    # refused for its cell (a tilted OVITO cell) is not read with one either,
    # so asking for it first would send the user round twice.
    missing_types = None
    if label_var is not None:
        identity, identity_var = "labels", label_var
        notes += mr._numeric_entries_note(user_types, "element labels")
    elif numbers_var is not None:
        identity, identity_var = "numbers", numbers_var
        source = "file symbols"
        notes.append(f"elements from the atomic numbers in {numbers_var.name}"
                     + (" (ASE's NetCDFTrajectory stores atomic numbers there)"
                        if numbers_var.name == "atom_types" else ""))
        notes += mr._numeric_entries_note(user_types, "atomic numbers")
        if user_labels:
            notes.append(f"type map entries {sorted(user_labels)} are not "
                         "used: the elements come from the atomic numbers in "
                         f"{numbers_var.name}")
    elif types_var is not None:
        identity, identity_var = "types", types_var
        if not user_types:
            missing_types = ValueError(
                f"{name}: the atoms carry numeric types ({types_var.name} "
                "variable) and the file names no element for them; pass "
                "type_map={1: '<element>', 2: '<element>', ...} with one entry "
                "per type")
        listed = dict(user_types)
        if user_labels:
            notes.append(f"type map entries {sorted(user_labels)} are labels; "
                         f"this file is mapped by its numeric {types_var.name}")
    elif topology is not None:
        identity, identity_var = "topology", None
        by_id, by_row, described = _topology_elements(topology, n, name)
        notes.append(f"elements from {described}")
        if ids is None and by_id is not None:
            by_id = None
            notes.append("this file carries no atom ids, so the topology's "
                         "atoms (in its row order) are taken for this file's "
                         "atoms in file order")
        if user_types or user_labels:
            notes.append("type map not used: the elements come from the "
                         "topology argument")
    else:
        raise ValueError(
            f"{name}: the file names no element or type for its atoms (AMBER "
            "keeps them in the topology file, and MDAnalysis's writer stores "
            "none); pass topology=<an MD file of the same atoms that FACET "
            "reads, such as the LAMMPS data file>, or topology=[one element "
            "symbol per atom]")

    # time and step variables, read when they give one number per frame
    step_var, _ = _nc_per_frame(header, header.variables.get("Timestep"),
                                positions.framed)
    if step_var is not None and step_var.dtype.kind not in "iu":
        step_var = None
    time_var, time_problem = _nc_per_frame(header, header.variables.get("time"),
                                           positions.framed)

    used = [v for v in (positions, lengths, angles, origin, velocities, ids,
                        charges, identity_var, unwrapped) if v is not None]
    for var in used + [v for v in (time_var,) if v is not None]:
        problem = _nc_scale_problem(var)
        if problem is not None:
            raise ValueError(
                f"{name}: {problem}. The AMBER convention multiplies a "
                "variable by its scale_factor; if the program that wrote the "
                "file still opens it, write it again")

    # frames: the records the file holds, from the offsets
    if positions.framed:
        total = header.numrecs if positions.is_record else \
            int(header.dimensions.get(positions.dimensions[0], 0))
        if header.streaming and positions.is_record:
            notes.append(f"the record count is not written (a streaming file); "
                         f"{total} complete record(s) were found")
    else:
        total = 1
    checked = used + [v for v in (step_var, time_var) if v is not None]
    held, short = total, None
    for var in checked:
        count = _nc_records_held(header, var, total)
        if count < held:
            held, short = count, var
    skipped: dict[int, str] = {}
    if short is not None:
        _skip_run(skipped, notes, held, total,
                  f"truncated: its {short.name} data ends beyond the end of the "
                  f"file ({header.file_size} bytes)")
    records = list(range(held))
    if not records:
        raise ValueError(f"{name} holds no readable frame: " + "; ".join(
            f"position {p}: {r}" for p, r in sorted(skipped.items()))
            if skipped else f"{name} holds no frame (the record count is 0)")
    with open(path, "rb") as handle:
        if len(records) > 1:
            last = records[-1]
            fill = _fill_value(positions)
            values = _nc_values(handle, header, positions, last)
            if fill is not None and (values == fill).all():
                skipped[last] = ("unwritten: its coordinates are all the netCDF "
                                 "fill value (the writer had not filled the "
                                 "record)")
                records = records[:-1]
        steps = [NO_TIMESTEP] * len(records)
        times = None
        if step_var is not None:
            steps = [int(_nc_values(handle, header, step_var, r).reshape(-1)[0])
                     for r in records]
            notes.append("steps from the Timestep variable (OVITO's)")
        if time_var is None:
            notes.append("no time variable in the file"
                         if "time" not in header.variables
                         else f"time not read: {time_problem}")
        elif is_ovito:
            notes.append("time not read: OVITO writes its animation frame "
                         "number (0, 1, 2, ...) there, not a time in ps "
                         "(measured on OVITO 3.16.1 files)")
        else:
            raw = np.array([_nc_values(handle, header, time_var, r)
                            .reshape(-1)[0] for r in records])
            unit = _attr_text(time_var, "units")
            factor = 1.0 if unit is None else _unit_factor(unit, _TIME_UNITS)
            scale = _attr_number(time_var, "scale_factor")
            if is_lammps and scale is not None and \
                    all(s == NO_TIMESTEP for s in steps) and \
                    np.isfinite(raw).all() and (raw == np.round(raw)).all() \
                    and np.abs(raw).max() < 2 ** 63:
                # dump_netcdf.cpp [9]: time = update->ntimestep, in a variable
                # of the dump's real type, scale_factor = (float) dt
                steps = [int(v) for v in np.round(raw).tolist()]
                notes.append(
                    "steps from the time variable: LAMMPS's dump netcdf "
                    "writes the step number there (time = update->ntimestep) "
                    f"and the step length as its scale_factor ({scale:g}"
                    + (f" {unit}" if unit else "") + "), so the times are "
                    "step x scale_factor")
                if raw.dtype.itemsize == 4 and np.abs(raw).max() > 2 ** 24:
                    notes.append(
                        "the time variable is float32 (LAMMPS writes it so "
                        "unless dump_modify double yes), which holds every "
                        "whole number only up to 2**24 = 16777216: steps "
                        "above that are stored rounded (16777217 as "
                        "16777216), and are read as stored")
            if factor is None:
                notes.append(f"time not read: its unit {unit!r} is not one "
                             "FACET converts")
            else:
                times = (raw.astype(np.float64)
                         * ((1.0 if scale is None else scale) * factor)).tolist()
                notes.append("frame times from the time variable"
                             + (" (no units attribute: ps, the AMBER "
                                "convention's unit)" if unit is None
                                else f" ({unit})"))
        boxes = []
        nonorthogonal = False
        exchanged = False
        for r in records:
            a = _nc_scaled(handle, header, angles, r, (3,))
            le = _nc_values(handle, header, lengths, r).reshape(3)
            o = None if origin is None else \
                _nc_values(handle, header, origin, r).reshape(3).tobytes()
            nonorthogonal |= bool((a != 90.0).any())
            exchanged |= bool(a[0] != 90.0 or a[2] != 90.0)
            boxes.append((le.tobytes(), a.tobytes(), o))
        if is_ovito and exchanged:
            raise UnsupportedFormat(
                f"{name}: written by OVITO with a cell tilted in xy or yz. "
                "OVITO's netcdf/amber exporter (3.16.1) writes as cell_angles "
                "the angles of the cell with its xy and yz tilts exchanged, "
                "while cell_lengths are the cell's own (measured on test cells "
                "with one tilt at a time and on a LAMMPS model; OVITO's own "
                "importer reads such a file back into the exchanged cell), so "
                "this file's lengths and angles do not describe the cell OVITO "
                "held. Export the model from OVITO as a LAMMPS dump or an "
                "extended XYZ file, which FACET reads")
        if missing_types is not None:
            raise missing_types
        label_names = None
        if identity == "labels":
            label_names = sorted(set(_nc_text_rows(
                _nc_values(handle, header, label_var, records[0]), n)))
    if step_fs is not None:
        if times is not None:
            raise ValueError(
                f"{name}: the file gives each frame's time (time variable); "
                "timestep_fs is for a file with steps and no time")
        times, note = _times_from_steps(name, steps, step_fs)
        notes.append(note)
    label_map: dict = {}
    if label_names is not None:
        try:
            label_map, listed, source, label_notes = mr._resolve_labels(
                label_names, user_labels, what="element label")
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from None
        label_map = {**user_labels, **label_map}
        notes += label_notes
    if nonorthogonal:
        notes.append("the AMBER convention gives the cell as lengths and angles; "
                     "the box is built with a along x and b in the xy plane "
                     "(LAMMPS's restricted orientation, as ASE and OVITO build "
                     "it), and the coordinates are taken to be in that frame")
    unread = [v for v in header.variables if v not in
              {x.name for x in used} | _NC_DESCRIPTIVE | {"time", "Timestep"}]
    notes += _not_read(unread, "variables")
    plan = _NcPlan(n=n, positions=positions, position_kind=kind,
                   length_factor=length_factor, unwrapped=unwrapped,
                   lengths=lengths, angles=angles, cell_factor=cell_factor,
                   origin=origin, origin_factor=origin_factor,
                   velocities=velocities, velocity_factor=velocity_factor,
                   ids=ids, charges=charges, identity=identity,
                   identity_var=identity_var, user_types=user_types,
                   label_map=label_map, by_id=by_id, by_row=by_row)
    units_text = ("AMBER NetCDF: lengths in Å, times in ps and velocities in "
                  "Å/ps, after the conversions in the notes")
    trajectory = _NetcdfTrajectory(
        path, header, records, plan,
        times=None if times is None else np.asarray(times, dtype=float),
        periodic=(True, True, True), file_format="amber-netcdf", n_atoms=n,
        n_frames=len(records), type_map_source=source, timesteps=steps,
        times_ps=times, type_map=listed, skipped=skipped, notes=notes,
        units_note=units_text,
        box_varies=any(b != boxes[0] for b in boxes[1:]))
    frame0 = mr._frame0(name, records[0], lambda: trajectory._build(0))
    trajectory._hold_to(frame0)
    if frame0.charge_e is not None:
        charges_e, charge_notes = md_model.charges_per_element(frame0.elements,
                                                               frame0.charge_e)
        trajectory.charges_e = charges_e
        trajectory.notes.extend(charge_notes)
    return mr._finish(trajectory, frame0)


# ---------------------------------------------------------------------------
# recognising the formats
# ---------------------------------------------------------------------------

def _sniff_ase_traj(head: bytes, path: Path) -> bool:
    """An ASE ULM file: its first 8 bytes are ULM's magic. Another tag (a
    GPAW file) is claimed too, so that read_ase_traj names the tag."""
    return head[:8] in _ULM_MAGICS


def _sniff_gsd(head: bytes, path: Path) -> bool:
    """A GSD file: the uint64 magic 0x65DF65DF65DF65DF, either byte order (a
    big-endian file is then refused naming its byte order)."""
    if len(head) < 8:
        return False
    return _GSD_MAGIC in (struct.unpack_from("<Q", head)[0],
                          struct.unpack_from(">Q", head)[0])


def _sniff_netcdf(head: bytes, path: Path) -> bool:
    """A NetCDF classic, 64-bit offset or CDF-5 file by its magic; a
    NetCDF-4 (HDF5) file only by a NetCDF name, so that it is refused with the
    conversion rather than as an unknown file."""
    if head[:4] in _CDF_VERSIONS:
        return True
    return head[:8] == _HDF5_MAGIC and \
        Path(path).suffix.lower() in _NC_EXTENSIONS


FORMATS = (
    FormatSpec(
        name="ase-traj",
        description="ASE trajectory (.traj, ULM binary)",
        extensions=(".traj",), stems=(), sniff=_sniff_ase_traj,
        read=read_ase_traj, options=frozenset({"timestep_fs"}), binary=True),
    FormatSpec(
        name="gsd",
        description="HOOMD-blue GSD (schema hoomd), as HOOMD-blue and OVITO "
                    "write it",
        extensions=(".gsd",), stems=(), sniff=_sniff_gsd, read=read_gsd,
        options=frozenset({"type_map", "units", "timestep_fs",
                           "mass_tol_amu"}),
        binary=True),
    FormatSpec(
        name="amber-netcdf",
        description="AMBER-convention NetCDF trajectory (classic, 64-bit "
                    "offset or CDF-5), as LAMMPS, OVITO, ASE, MDAnalysis and "
                    "AMBER write it",
        extensions=_NC_EXTENSIONS, stems=(), sniff=_sniff_netcdf,
        read=read_amber_netcdf,
        options=frozenset({"type_map", "timestep_fs", "topology"}),
        binary=True),
)
