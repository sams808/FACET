"""Write the edge-case DCD, binary-dump and YAML-dump files in ``edges/``.

Each file is written by the program that writes it in the wild:

* LAMMPS 22 Jul 2025 update 4 (the ``lammps`` pip wheel 2025.7.22.4.0,
  EXTRA-DUMP package) for ``first_yes.dcd``, ``thresh.bin``, ``thresh.yaml``,
  ``colname.bin``, ``colname.yaml`` and ``general.yaml``;
* mdtraj 1.11 (``DCDTrajectoryFile``, VMD's molfile DCD plugin) for
  ``plugin.dcd``;
* MDAnalysis 2.10 (``DCDWriter``) for ``mda_nocell.dcd``;

on Python 3.11 with numpy 2.4.6. They are committed; tests/
test_md_formats_lammps.py reads them. Regenerate with a Python that has
lammps, mdtraj and MDAnalysis (the files of make_lammps_formats.py must exist
first: plugin.dcd and mda_nocell.dcd take their positions from truth.npz)::

    <python> tests/data/md/lammps_formats/make_lammps_edges.py

``edges.json`` holds LAMMPS's own state, read through the library interface
right after each step that wrote a frame (never from a dump), and the writer
versions.

THE FILES
---------
* ``first_yes.dcd`` with ``first_yes.data`` (write_data before the dump):
  ``run 105``, then ``dump dcd 20`` with ``dump_modify first yes`` and runs to
  steps 120 and 140, so the frames are at steps 105, 120, 140 (output.cpp
  writes the first snapshot on the setup step, the later ones on multiples of
  20).
* ``thresh.bin`` / ``thresh.yaml``: 8 atoms, a dump every 10 steps, with
  ``dump_modify thresh id > 100000`` during steps 21-40 so the frames at steps
  30 and 40 hold 0 atoms; steps 0, 10, 20, 50, 60 hold all 8.
* ``colname.bin`` / ``colname.yaml``: ``id type x y z`` with ``dump_modify
  colname x X``; from the same 6-atom run, ``cfg_vel/dump.*.cfg`` (``mass
  type xs ys zs id vx vy vz`` with element names), whose auxiliary vx vy vz
  are the run's velocities in Å/ps (units metal).
* ``general.yaml``: a general triclinic box (``lattice custom ...
  triclinic/general``) and ``dump_modify triclinic/general yes`` on a yaml
  dump, which LAMMPS accepts; the header holds the restricted box and the rows
  general-frame positions.
* ``plugin.dcd``: the 12-atom metal model of truth.npz (3 frames) written by
  mdtraj, whose molfile plugin writes ISTART 0, NSAVC 1 and DELTA 1.0.
* ``mda_nocell.dcd``: the same positions written by MDAnalysis with no
  dimensions, which writes a zeroed unit cell.
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "edges"

MASS = {1: 28.0855, 2: 22.98977, 3: 15.9994}
CHARGE = {1: 2.4, 2: 0.6, 3: -1.2}


def new():
    from lammps import lammps

    return lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"])


def state(lmp) -> dict:
    nlocal = int(lmp.extract_global("nlocal"))
    ids = np.array(lmp.numpy.extract_atom("id"), dtype=np.int64)[:nlocal].copy()
    order = np.argsort(ids)
    x = np.array(lmp.numpy.extract_atom("x"))[:nlocal, :3].copy()[order]
    v = np.array(lmp.numpy.extract_atom("v"))[:nlocal, :3].copy()[order]
    lo, hi, xy, yz, xz = lmp.extract_box()[:5]
    return {"step": int(lmp.extract_global("ntimestep")),
            "time": float(lmp.get_thermo("time")),
            "ids": ids[order].tolist(), "x": x.tolist(), "v": v.tolist(),
            "lo": list(lo), "H": [[hi[0] - lo[0], 0.0, 0.0],
                                  [xy, hi[1] - lo[1], 0.0],
                                  [xz, yz, hi[2] - lo[2]]]}


SMALL = """
units metal
atom_style charge
atom_modify map array
boundary p p p
region box block 0.5 8.5 -1.0 7.5 0.0 9.0
create_box 3 box
mass 1 28.0855
mass 2 22.98977
mass 3 15.9994
create_atoms 1 single 1.2 1.1 1.3
create_atoms 3 single 2.9 1.4 2.2
create_atoms 2 single 5.1 4.0 6.3
create_atoms 3 single 6.6 5.2 3.1
create_atoms 1 single 3.8 6.1 7.4
create_atoms 3 single 7.7 2.6 5.5
set type 1 charge 2.4
set type 2 charge 0.6
set type 3 charge -1.2
pair_style soft 1.5
pair_coeff * * 3.0
velocity all create 1500.0 5521 dist gaussian mom yes rot no
neighbor 1.0 bin
fix 1 all nve/limit 0.05
timestep 0.001
"""


def first_yes() -> dict:
    lmp = new()
    lmp.commands_string(SMALL)
    lmp.command("run 105")
    lmp.commands_string(f"""
write_data {(OUT / 'first_yes.data').as_posix()}
dump d all dcd 20 {(OUT / 'first_yes.dcd').as_posix()}
dump_modify d first yes
""")
    states = []
    for cmd in ("run 0", "run 15", "run 20"):
        lmp.command(cmd)
        states.append(state(lmp))
    lmp.close()
    return {"steps": [s["step"] for s in states],
            "times": [s["time"] for s in states]}


def colname() -> dict:
    (OUT / "cfg_vel").mkdir()
    lmp = new()
    lmp.commands_string(SMALL)
    lmp.commands_string(f"""
dump b all custom 10 {(OUT / 'colname.bin').as_posix()} id type x y z
dump_modify b colname x X
dump y all yaml 10 {(OUT / 'colname.yaml').as_posix()} id type x y z
dump_modify y colname x X
dump c all cfg 10 {(OUT / 'cfg_vel' / 'dump.*.cfg').as_posix()} mass type xs ys zs id vx vy vz
dump_modify c element Si Na O
""")
    states = []
    for cmd in ("run 0", "run 10"):
        lmp.command(cmd)
        states.append(state(lmp))
    lmp.close()
    return {"steps": [s["step"] for s in states],
            "ids": states[0]["ids"], "x": [s["x"] for s in states],
            "v": [s["v"] for s in states], "H": [s["H"] for s in states],
            "lo": [s["lo"] for s in states]}


def thresh() -> dict:
    lmp = new()
    lmp.commands_string(f"""
units metal
atom_style charge
boundary p p p
lattice sc 3.0
region box block 0 2 0 2 0 2
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.5 12345
mass 1 28.0855
mass 2 15.9994
set type 1 charge 2.4
set type 2 charge -1.2
pair_style zero 5.0
pair_coeff * *
velocity all create 600 4928459
fix 1 all nve
dump b all custom 10 {(OUT / 'thresh.bin').as_posix()} id type x y z
dump y all yaml 10 {(OUT / 'thresh.yaml').as_posix()} id type x y z
""")
    states = []

    def run(n):
        # a state after every dumped step
        for _ in range(n // 10):
            lmp.command("run 10")
            states.append(state(lmp))

    lmp.command("run 0")
    states.append(state(lmp))
    run(20)
    lmp.command("dump_modify b thresh id > 100000")
    lmp.command("dump_modify y thresh id > 100000")
    run(20)
    lmp.command("dump_modify b thresh none")
    lmp.command("dump_modify y thresh none")
    run(20)
    lmp.close()
    return {"steps": [s["step"] for s in states],
            "empty_steps": [30, 40], "ids": states[0]["ids"],
            "x": [s["x"] for s in states]}


def general() -> dict:
    a1 = np.array([13.0, 1.1, -0.9])
    a2 = np.array([2.4, 12.6, 0.5])
    a3 = np.array([-1.4, 0.9, 14.2])
    lmp = new()
    lmp.commands_string(f"""
units metal
atom_style charge
atom_modify map array
boundary p p p
lattice custom 1.0 a1 {a1[0]} {a1[1]} {a1[2]} a2 {a2[0]} {a2[1]} {a2[2]} a3 {a3[0]} {a3[1]} {a3[2]} basis 0 0 0 triclinic/general
create_box 3 NULL -0.4 0.6 0.2 1.2 -0.3 0.7
mass 1 28.0855
mass 2 22.98977
mass 3 15.9994
create_atoms 1 random 8 11 NULL
create_atoms 2 random 6 12 NULL
create_atoms 3 random 18 13 NULL
set type 1 charge 2.4
set type 2 charge 0.6
set type 3 charge -1.2
pair_style soft 1.2
pair_coeff * * 4.0
velocity all create 2000.0 991 dist gaussian mom yes rot no
timestep 0.001
neighbor 1.0 bin
fix 1 all nve/limit 0.05
dump y all yaml 10 {(OUT / 'general.yaml').as_posix()} id type x y z
dump_modify y triclinic/general yes
""")
    states = []
    for cmd in ("run 0", "run 10"):
        lmp.command(cmd)
        states.append(state(lmp))
    lmp.close()
    return {"steps": [s["step"] for s in states],
            "restricted_H": [s["H"] for s in states],
            "restricted_lo": [s["lo"] for s in states]}


def other_dcds() -> dict:
    truth = np.load(HERE / "truth.npz")
    x, H, lo = truth["metal_x"], truth["metal_H"], truth["metal_lo"]
    positions = (x - lo[:, None, :]).astype(np.float32)
    lengths = np.linalg.norm(H, axis=2)
    a, b, c = H[:, 0], H[:, 1], H[:, 2]

    def angle(u, v):
        cos = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1)
                                       * np.linalg.norm(v, axis=1))
        return np.degrees(np.arccos(cos))

    angles = np.stack([angle(b, c), angle(a, c), angle(a, b)], axis=1)
    from mdtraj.formats import DCDTrajectoryFile

    with DCDTrajectoryFile(str(OUT / "plugin.dcd"), "w") as handle:
        handle.write(positions, cell_lengths=lengths.astype(np.float32),
                     cell_angles=angles.astype(np.float32))
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    warnings.filterwarnings("ignore")
    universe = mda.Universe.empty(positions.shape[1], trajectory=True)
    universe.load_new(positions, format=MemoryReader)
    with mda.Writer(str(OUT / "mda_nocell.dcd"), positions.shape[1]) as writer:
        for _ in universe.trajectory:
            writer.write(universe)
    return {"positions_written": "metal_x - metal_lo of truth.npz as float32",
            "plugin_cell_lengths": lengths.tolist(),
            "plugin_cell_angles_deg": angles.tolist()}


def main() -> None:
    import lammps
    import MDAnalysis
    import mdtraj
    from importlib.metadata import version

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    record = {"first_yes": first_yes(), "colname": colname(),
              "thresh": thresh(), "general": general(),
              "other_dcds": other_dcds()}
    probe = lammps.lammps(cmdargs=["-screen", "none", "-log", "none"])
    record["writers"] = (
        f"LAMMPS {probe.version()} (pip wheel lammps {version('lammps')}), "
        f"mdtraj {mdtraj.__version__}, MDAnalysis {MDAnalysis.__version__}, "
        f"numpy {np.__version__}, Python {sys.version.split()[0]}")
    probe.close()
    (OUT / "edges.json").write_text(json.dumps(record, indent=1))
    for path in sorted(OUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
