"""Write the XTC fixtures of tests/test_md_formats_xtc.py with real writers.

Run once with a Python that has MDAnalysis and the LAMMPS Python module; the
test suite itself needs neither. The fixtures in this folder were written by

* MDAnalysis 2.10.0, ``MDAnalysis.lib.formats.libmdaxdr.XTCFile`` (the
  xdrfile C library), Python 3.11, numpy 2.4.6:
  xtc_mda_glass.xtc, xtc_mda_runs.xtc, xtc_mda_wide.xtc, xtc_mda_large.xtc,
  xtc_mda_small.xtc, xtc_mda_zerobox.xtc;
* LAMMPS 22 Jul 2025 update 4 (pip wheel ``lammps`` 2025.7.22.4.0),
  ``dump xtc`` (dump_xtc.cpp): xtc_lammps.xtc, with xtc_lammps.data
  (``write_data ... nocoeff`` before the run, the topology) and
  xtc_lammps_ref.lammpstrj (``dump custom id type x y z``, ``%.10g``, the same
  steps: LAMMPS's own positions in Å, the ground truth).

The MDAnalysis positions come from closed formulas (no random generator), so
the tests recompute what was written; the same formulas are in the test file.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def tri_box(a, b, c, xy=0.0, xz=0.0, yz=0.0):
    """Cell vectors as rows, nm."""
    return np.array([[a, 0.0, 0.0], [xy, b, 0.0], [xz, yz, c]])


GLASS_BOX = tri_box(1.52, 1.46, 1.58, 0.21, -0.13, 0.09)


def glass_positions(n, k, box):
    """Atoms spread by irrational steps: no atom lies near the one before it
    in file order, so every atom is written in full."""
    i = np.arange(n)[:, None]
    frac = np.mod(np.array([0.6180339887, 0.4142135624, 0.7320508076]) * (i + 1)
                  + 0.013 * k * np.array([1.0, -1.0, 0.5]), 1.0)
    return frac @ box


def chain_positions(n):
    """A chain whose step size changes every 12 atoms (1 pm to 0.3 nm), so the
    writer's runs reach 8 small atoms, repeat a run length, and move the
    small-difference width both ways."""
    i = np.arange(n)
    scale = np.array([0.001, 0.004, 0.02, 0.3])[(i // 12) % 4]
    steps = np.stack([np.sin(1.3 * i), np.cos(2.1 * i), np.sin(0.7 * i + 1.0)],
                     axis=1) * scale[:, None]
    return 1.5 + np.cumsum(steps, axis=0)


def spread_positions(n, span):
    """n atoms across a cube of ``span`` nm, the first and second at opposite
    corners so every axis range is nearly ``span``."""
    pos = glass_positions(n, 0, tri_box(span, span, span))
    pos[0] = 0.5
    pos[1] = span - 0.5
    return pos


def write_mda():
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile

    def write(name, frames, precision):
        with XTCFile(str(HERE / name), "w") as handle:
            for xyz, box, step, time in frames:
                handle.write(np.asarray(xyz, np.float32),
                             np.asarray(box, np.float32), step, time,
                             precision=precision)

    write("xtc_mda_glass.xtc",
          [(glass_positions(30, k, GLASS_BOX), GLASS_BOX * (1 + 0.01 * k),
            100 * k, 0.2 * k) for k in range(3)], 1000.0)
    write("xtc_mda_runs.xtc",
          [(chain_positions(240), tri_box(3.0, 3.0, 3.0), 7, 1.5)], 1000.0)
    write("xtc_mda_wide.xtc",
          [(spread_positions(12, 30.0), tri_box(30.0, 30.0, 30.0), 0, 0.0)],
          1.0e5)
    write("xtc_mda_large.xtc",
          [(spread_positions(12, 20.0), tri_box(20.0, 20.0, 20.0), 0, 0.0)],
          1.0e6)
    write("xtc_mda_small.xtc",
          [(glass_positions(6, k, GLASS_BOX), GLASS_BOX, k, 0.5 * k)
           for k in range(2)], 1000.0)
    write("xtc_mda_zerobox.xtc",
          [(glass_positions(12, 0, GLASS_BOX), np.zeros((3, 3)), 0, 0.0)],
          1000.0)


# The LAMMPS model: 24 atoms (Si 6, Na 6, O 12) in a restricted triclinic box
# with a corner away from the origin, no forces (pair_style zero), velocities
# at 3000 K and a box stretched along x, so positions and the box change.
LAMMPS_INPUT = """
units metal
atom_style atomic
boundary p p p
region box prism -1.5 9.5 0.5 10.5 -2.0 8.0 1.1 -0.6 0.4
create_box 3 box
mass 1 28.0855
mass 2 22.98977
mass 3 15.9994
"""

LAMMPS_RUN = """
pair_style zero 4.0
pair_coeff * *
velocity all create 3000 4928459 dist gaussian
timestep 0.001
write_data {data} nocoeff
dump x all xtc 10 {xtc}
dump t all custom 10 {ref} id type x y z
dump_modify t format float %.10g sort id
fix 1 all nve
fix 2 all deform 1 x erate 1.0 remap x
run 20
"""


def write_lammps():
    from lammps import lammps

    lmp = lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"])
    lmp.commands_string(LAMMPS_INPUT)
    lo = np.array([-1.5, 0.5, -2.0])
    h = np.array([[11.0, 0.0, 0.0], [1.1, 10.0, 0.0], [-0.6, 0.4, 10.0]])
    frac = np.mod(np.array([0.6180339887, 0.4142135624, 0.7320508076])
                  * (np.arange(24)[:, None] + 1), 1.0)
    x = lo + frac @ h
    types = [1] * 6 + [2] * 6 + [3] * 12
    lmp.create_atoms(24, list(range(1, 25)), types,
                     [float(v) for v in x.ravel()])
    lmp.commands_string(LAMMPS_RUN.format(
        data=HERE / "xtc_lammps.data", xtc=HERE / "xtc_lammps.xtc",
        ref=HERE / "xtc_lammps_ref.lammpstrj"))
    lmp.close()


if __name__ == "__main__":
    write_mda()
    write_lammps()
    for path in sorted(HERE.glob("xtc_*")):
        print(f"{path.name:28s} {path.stat().st_size:6d} bytes")
