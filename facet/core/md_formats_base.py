"""The contract by which further MD file formats join :mod:`.md_readers`.

:mod:`.md_readers` reads the formats an oxide-glass MD run most often leaves
behind -- LAMMPS data and dump files, extended XYZ, VASP XDATCAR and DL_POLY
CONFIG / HISTORY. Others turn up in a group's directories too: LAMMPS's DCD,
XTC, binary and YAML dumps, AtomEye CFG, ASE trajectories, GSD, AMBER NetCDF,
CASTEP .md, GROMACS .gro, animated XSF, multi-model PDB, a series of POSCAR
files. Each of those lives in a module of its own, ``md_formats_*.py``, which
exports ``FORMATS``: a tuple of :class:`FormatSpec`. :func:`.md_readers.
read_trajectory` and :func:`.md_readers.sniff_md` consult the modules listed in
:data:`FORMAT_MODULES` after their own formats, so a new format is added
without touching the reader that already works.

The rules a format module keeps are the ones :mod:`.md_readers` keeps:

* every frame is built with :func:`.md_model.frame_from_arrays`, so the same
  checks apply to every format;
* frames are read on demand -- a long trajectory is never held at once;
* a frame that cannot be read is counted in ``Trajectory.skipped`` with its
  reason, never dropped silently, and a file that cannot be read at all is
  refused with :class:`.readers.UnsupportedFormat` (or a ValueError) whose
  message names the file and says what FACET does read;
* elements come from the file's own symbols, a type map from the user, or
  masses that match exactly one element (:func:`.md_readers.
  type_map_from_masses`), and the source used is recorded in
  ``type_map_source``;
* units are converted with constants from :mod:`scipy.constants` or with the
  format's own stated definition, cited in the module docstring, and the
  conversion is recorded in ``units_note``;
* only numpy, scipy, gemmi and the standard library are used: a binary format
  is read with numpy, not with the library its authors provide.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Keyword options read_trajectory accepts and may pass on to a format's reader.
# A format lists the ones it uses in FormatSpec.options; read_trajectory
# refuses any other one rather than ignoring it.
KNOWN_OPTIONS = frozenset({
    "type_map",      # Mapping[int | str, str]: type number or label -> element
    "atom_style",    # LAMMPS data files
    "masses_from",   # a LAMMPS data file whose Masses name the types
    "units",         # LAMMPS unit style when the file does not say
    "mass_tol_amu",  # tolerance of the mass -> element match
    "box_from",      # a box for a file that has none: another MD file, or (3, 3) rows in A
    "topology",      # elements / ids for a file that has none (DCD): another MD file
    "timestep_fs",   # time between steps when the file gives steps but no time
    "columns",       # per-atom column names for a file that does not state them
                     # (a LAMMPS binary dump in the layout without a magic
                     # string, which holds no column names)
})

# The modules that hold further formats, in the order they are consulted.
FORMAT_MODULES = (
    "md_formats_lammps",    # DCD, LAMMPS binary dump, YAML dump, extended CFG
    "md_formats_xtc",       # GROMACS / LAMMPS XTC
    "md_formats_binary",    # ASE .traj, GSD, AMBER NetCDF
    "md_formats_text",      # CASTEP .md, GROMACS .gro, animated XSF, multi-model PDB, IMD, POSCAR series
)


def _imports_for_frozen_builds() -> None:  # pragma: no cover - never called
    """Never called: it names the modules of :data:`FORMAT_MODULES` to a
    frozen build.

    :func:`.md_readers.format_registry` imports the modules by name at run
    time (``importlib.import_module``), which PyInstaller's import analysis
    cannot follow, while it does follow import statements in a function's
    body. Measured with PyInstaller 6.11.1's module graph started from
    ``facet.core.readers`` and ``facet.core.md_readers`` (2026-10-07): without
    these statements none of the four modules was found, so a frozen FACET
    would list them all as absent and read none of their 14 formats; with
    them, all four are found. The import cannot run at module level, because
    the format modules import :mod:`.md_readers`, which imports this module.
    tests/test_md_formats_dispatch.py checks that this list and
    :data:`FORMAT_MODULES` name the same modules.
    """
    from . import (md_formats_binary, md_formats_lammps,  # noqa: F401
                   md_formats_text, md_formats_xtc)


@dataclass(frozen=True)
class FormatSpec:
    """One file format a module adds.

    ``sniff(head, path)`` decides from the first bytes of the file (the
    caller passes :data:`.md_readers.SNIFF_BYTES` of them, decompressed if the
    file is gzip) and the path; it never reads the whole file and never
    raises. ``read(path, **options)`` returns a :class:`.md_model.
    Trajectory` and accepts exactly the keywords in ``options``.
    ``extensions`` (lower case, with the dot) and ``stems`` (exact names or
    name prefixes, compared case-sensitively as VASP and DL_POLY name their
    files) let :func:`.readers.read` route a file without sniffing it.
    ``binary`` says the format is not text, so a crystal reader never sees it.
    """

    name: str
    description: str
    extensions: tuple[str, ...]
    stems: tuple[str, ...]
    sniff: Callable[[bytes, Path], bool]
    read: Callable[..., object]
    options: frozenset[str] = field(default_factory=frozenset)
    binary: bool = False
    # For a format written one file per snapshot (LAMMPS dump cfg, a POSCAR
    # series): read_series(paths, **options) -> Trajectory over the files in
    # the given order. read_trajectory calls it for a directory, a wildcard
    # pattern or a list of paths.
    read_series: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        unknown = set(self.options) - KNOWN_OPTIONS
        if unknown:
            raise ValueError(f"format {self.name!r} lists unknown options "
                             f"{sorted(unknown)}; known: {sorted(KNOWN_OPTIONS)}")
        if any(not e.startswith(".") or e != e.lower() for e in self.extensions):
            raise ValueError(f"format {self.name!r}: extensions are lower case "
                             f"and start with a dot: {self.extensions}")
