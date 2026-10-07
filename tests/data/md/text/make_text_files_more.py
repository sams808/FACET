"""Write the further md_formats_text test files in this folder, from the
six-atom, three-frame model of make_text_files.py (imported from there).

Two Pythons are needed, because pymatgen and MDAnalysis were installed in
different environments here; each run writes the files its packages can
write and records their versions in writers_more.json:

    <python with MDAnalysis> tests/data/md/text/make_text_files_more.py
    <python with pymatgen>   tests/data/md/text/make_text_files_more.py

The files are committed; tests/test_md_formats_text.py reads them and never
runs this script. This script imports nothing from FACET.

WHAT WRITES WHAT
----------------
* ``mda_resid_wrap.pdb``: MDAnalysis ``PDBWriter`` with ``multiframe=True``,
  one residue per atom with residue numbers 1, 2, 3, 10001, 10002, 10003.
  The writer keeps the last four digits of a residue number (the resSeq field
  is four columns wide), so the file holds 1, 2, 3, 1, 2, 3: residue numbers
  that repeat, as in every MDAnalysis or mdtraj PDB of more than 9999
  residues (an ionic model with one residue per ion).
* ``pmg_single.xsf``: pymatgen ``Structure.to(fmt='xsf')`` of frame 0, a
  one-structure XSF (CRYSTAL, PRIMVEC, PRIMCOORD); pymatgen writes no line
  ending after the last line.
* ``cp2k_layout.pdb``: no CP2K was available, so this file is composed by
  this script with the Fortran formats of CP2K's PDB trajectory writer
  (https://github.com/cp2k/cp2k, master, read 2026-10-07):
  src/motion_utils.F writes 'TITLE' and 'AUTHOR' once (FMT "(A6,T11,A)")
  and, per frame, the title 'Step <it>, time = <t>, E = <etot>' (FMT
  "(A,I0,A,F0.3,A,F0.10)"), where src/motion/md_energies.F passes the time as
  ``time*femtoseconds``; src/particle_methods.F writes that title as
  'REMARK' (FMT "(A6,T11,A)"), CRYST1 (FMT "(A6,3F9.3,3F7.2)", no space
  group), each atom into a blank line buffer ('ATOM  ' in 1-6, the serial
  (I5) in 7-11, the name (A4, left-adjusted) in 13-16, x y z (3F8.3) in
  31-54, the charge (F6.2) in 55-60 and 61-66, the element (A2,
  right-adjusted) in 77-78), and 'END' after each frame. Steps 0, 250, 500 at
  the model's times (0, 0.5, 1 ps = 0, 500, 1000 fs); the energies are test
  inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_text_files as M  # noqa: E402  (the model, no FACET import)

CP2K_STEPS = [0, 250, 500]
CP2K_ENERGIES = [-1234.5678901234, -1234.4678901234, -1234.3678901234]


def cellpar(box):
    """a, b, c (Å) and alpha, beta, gamma (degrees) of a box given as rows."""
    a, b, c = (np.linalg.norm(v) for v in box)

    def angle(u, v):
        return np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u)
                                    / np.linalg.norm(v)))
    return a, b, c, angle(box[1], box[2]), angle(box[0], box[2]), \
        angle(box[0], box[1])


def write_mdanalysis() -> dict:
    import MDAnalysis as mda
    from MDAnalysis.lib.mdamath import triclinic_box

    n = len(M.ELEMENTS)
    u = mda.Universe.empty(n, n_residues=n, atom_resindex=np.arange(n),
                           trajectory=True)
    u.add_TopologyAttr("names", M.ELEMENTS)
    u.add_TopologyAttr("elements", M.ELEMENTS)
    u.add_TopologyAttr("resnames", ["ION"] * n)
    u.add_TopologyAttr("resids", [1, 2, 3, 10001, 10002, 10003])
    with mda.Writer(str(HERE / "mda_resid_wrap.pdb"), n,
                    multiframe=True) as writer:
        for k in range(3):
            u.atoms.positions = M.POSITIONS[k]
            u.dimensions = triclinic_box(*M.BOXES[k])
            writer.write(u.atoms)
    return {"MDAnalysis": f"{mda.__version__} (numpy {np.__version__})"}


def write_pymatgen() -> dict:
    import pymatgen.core
    from pymatgen.core import Lattice, Structure

    structure = Structure(Lattice(M.BOXES[0]), list(M.ELEMENTS),
                          M.POSITIONS[0], coords_are_cartesian=True,
                          to_unit_cell=False)
    structure.to(filename=str(HERE / "pmg_single.xsf"), fmt="xsf")
    return {"pymatgen": f"{pymatgen.core.__version__} (numpy {np.__version__})"}


def write_cp2k_layout() -> None:
    lines = [f"{'TITLE ':6s}    PDB file created by CP2K version 2025.2 "
             "(revision git:0000000)",
             f"{'AUTHOR':6s}    user@host 2026-10-07 12:00:00"]
    for k in range(3):
        time_fs = M.TIMES_PS[k] * 1000.0
        title = (f"Step {CP2K_STEPS[k]}, time = {time_fs:.3f}, "
                 f"E = {CP2K_ENERGIES[k]:.10f}")
        lines.append(f"REMARK    {title}")
        a, b, c, alpha, beta, gamma = cellpar(M.BOXES[k])
        lines.append(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}"
                     f"{gamma:7.2f}")
        for i, (element, r) in enumerate(zip(M.ELEMENTS, M.POSITIONS[k]),
                                         start=1):
            line = [" "] * 80
            line[0:6] = "ATOM  "
            line[6:11] = f"{i % 100000:5d}"
            line[12:16] = f"{element:<4s}"
            line[30:54] = f"{r[0]:8.3f}{r[1]:8.3f}{r[2]:8.3f}"
            line[54:60] = f"{0.0:6.2f}"
            line[60:66] = f"{0.0:6.2f}"
            line[76:78] = f"{element:>2s}"
            lines.append("".join(line).rstrip())
        lines.append("END")
    (HERE / "cp2k_layout.pdb").write_text("\n".join(lines) + "\n",
                                          encoding="ascii", newline="\n")


def main() -> None:
    record_path = HERE / "writers_more.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) \
        if record_path.exists() else {}
    wrote = []
    try:
        record.update(write_mdanalysis())
        wrote.append("mda_resid_wrap.pdb")
    except ImportError:
        pass
    try:
        record.update(write_pymatgen())
        wrote.append("pmg_single.xsf")
    except ImportError:
        pass
    write_cp2k_layout()
    wrote.append("cp2k_layout.pdb")
    record["cp2k_layout.pdb"] = ("composed by make_text_files_more.py from the "
                                 "Fortran formats of CP2K master "
                                 "(motion_utils.F, particle_methods.F)")
    record.pop("numpy", None)
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                           encoding="utf-8", newline="\n")
    # Windows text mode makes writers end lines with CRLF; keep LF, as the
    # same writers produce on Linux (the tests make CRLF copies themselves).
    for name in wrote:
        path = HERE / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    print("wrote", ", ".join(wrote))


if __name__ == "__main__":
    main()
