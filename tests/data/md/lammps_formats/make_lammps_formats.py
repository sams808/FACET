"""Write the DCD, binary-dump, YAML-dump and CFG test files in this folder.

The files are written by the programs that write them in the wild, not by
hand: LAMMPS 22 Jul 2025 update 4 (the ``lammps`` pip wheel 2025.7.22.4.0,
LAMMPS version 20250722, EXTRA-DUMP package) and ASE 3.29.0 (``ase.io``
``write(..., format='cfg')``), on Python 3.11 with numpy 2.4.6. They are
committed; tests/test_md_formats_lammps.py reads them. Regenerate with a
Python that has those two packages::

    <python with lammps and ase> tests/data/md/lammps_formats/make_lammps_formats.py

GROUND TRUTH
------------
``truth.npz`` is LAMMPS's own state, read through the library interface
(``extract_atom``, ``extract_box``, ``get_thermo``) right after each step that
wrote a frame, never from a dump file. Arrays (F frames, N atoms, rows sorted
by atom id), with a ``metal_`` or ``real_`` prefix for the two sessions:
``ids``, ``types``, ``x`` (F, N, 3), ``image`` (F, N, 3), ``v`` (F, N, 3) in
the run's velocity unit, ``q``, ``H`` (F, 3, 3) box rows, ``lo`` (F, 3) box
origin, ``steps``, ``time`` (in the run's time unit) and ``dt``.

THE MODELS
----------
The numbers are test inputs, not a material.

* **metal** -- 12 atoms (3 Si, 3 Na, 6 O as types 1, 2, 3, charges 2.4, 0.6,
  -1.2) in a restricted triclinic box with a non-zero origin that fix deform
  changes every step; three frames at steps 100, 110, 120 (dt = 0.001 ps).
  Four atoms start with non-zero image flags (up to 2), so unwrapped positions
  lie more than a box away, and atom 1 starts 0.2 A below the x face moving
  out, so it crosses the boundary before the second frame.
* **real** -- 6 atoms in an orthogonal box with a non-zero origin, units
  real (dt = 1 fs), three frames at steps 0, 5, 10: a DCD whose DELTA is in fs.

FILES
-----
metal: ``topology.data`` (write_data before the run, the file a run reads),
``traj.dcd``, ``traj_unwrap.dcd`` (dump_modify unwrap yes),
``traj_custom.bin`` (id type x y z ix iy iz vx vy vz q, time yes units yes),
``traj_atom.bin`` (dump atom: id type xs ys zs), ``traj_element.bin``
(id element x y z with dump_modify element), ``traj.yaml`` (id type element
x y z ix iy iz vx vy vz q, element names, time yes units yes thermo yes),
``traj_typelabel.yaml`` (id typelabel x y z, labelmap Si Na O),
``cfg/dump.*.cfg`` (mass type xs ys zs id q ix iy iz, element names),
``cfg_unwrap/dump.*.cfg`` (mass type xsu ysu zsu id), ``cfg_noelement/
dump.100.cfg`` (no dump_modify element, so every type is written 'C');
ASE: ``ase.cfg`` (frame 0 of the metal model); real: ``topology_real.data``,
``traj_real.dcd``.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

MASS = {1: 28.0855, 2: 22.98977, 3: 15.9994}
CHARGE = {1: 2.4, 2: 0.6, 3: -1.2}
ELEMENT = {1: "Si", 2: "Na", 3: "O"}

# metal model: restricted triclinic box with origin (-2.5, 1.25, -3.0)
LO = np.array([-2.5, 1.25, -3.0])
LX, LY, LZ, XY, XZ, YZ = 7.5, 7.0, 7.75, 1.1, -0.6, 0.45
H0 = np.array([[LX, 0.0, 0.0], [XY, LY, 0.0], [XZ, YZ, LZ]])
TYPES = np.array([1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3])
FRAC = np.array([
    [0.973, 0.50, 0.50], [0.10, 0.10, 0.10], [0.35, 0.20, 0.70],
    [0.60, 0.15, 0.30], [0.20, 0.55, 0.25], [0.45, 0.45, 0.55],
    [0.80, 0.60, 0.15], [0.15, 0.85, 0.75], [0.55, 0.80, 0.85],
    [0.75, 0.35, 0.80], [0.40, 0.95, 0.40], [0.85, 0.85, 0.50]])
IMAGE = np.zeros((12, 3), dtype=int)
IMAGE[1] = (2, 0, -1)
IMAGE[4] = (0, -2, 1)
IMAGE[7] = (-1, 1, 0)
IMAGE[10] = (1, 0, 2)

# real model: orthogonal box with origin (1.5, -2.0, 0.5)
LO_REAL = np.array([1.5, -2.0, 0.5])
L_REAL = np.array([6.5, 6.0, 7.0])
TYPES_REAL = np.array([1, 3, 3, 2, 3, 3])
FRAC_REAL = np.array([[0.2, 0.3, 0.4], [0.7, 0.2, 0.1], [0.4, 0.8, 0.6],
                      [0.9, 0.6, 0.8], [0.1, 0.9, 0.2], [0.6, 0.5, 0.9]])


def lmp_session(units: str):
    from lammps import lammps

    lmp = lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"])
    return lmp


def state(lmp) -> dict:
    nlocal = int(lmp.extract_global("nlocal"))
    ids = np.array(lmp.numpy.extract_atom("id"), dtype=np.int64)[:nlocal].copy()
    order = np.argsort(ids)

    def get(name):
        return np.array(lmp.numpy.extract_atom(name))[:nlocal].copy()[order]

    packed = get("image")
    image = np.array([lmp.decode_image_flags(int(p)) for p in packed])
    lo, hi, xy, yz, xz = lmp.extract_box()[:5]
    H = np.array([[hi[0] - lo[0], 0.0, 0.0], [xy, hi[1] - lo[1], 0.0],
                  [xz, yz, hi[2] - lo[2]]])
    return {"ids": ids[order], "types": get("type"), "x": get("x")[:, :3],
            "image": image, "v": get("v")[:, :3], "q": get("q"), "H": H,
            "lo": np.array(lo), "step": int(lmp.extract_global("ntimestep")),
            "time": float(lmp.get_thermo("time"))}


def create(lmp, types, x, image):
    n = len(types)
    img = [lmp.encode_image_flags(int(a), int(b), int(c)) for a, b, c in image]
    made = lmp.create_atoms(n, list(range(1, n + 1)), [int(t) for t in types],
                            [float(v) for v in np.asarray(x).ravel()], None, img)
    assert made == n, made


def clean(names):
    for name in names:
        target = HERE / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def metal_session() -> list[dict]:
    lmp = lmp_session("metal")
    hi = LO + np.array([LX, LY, LZ])
    lmp.commands_string(f"""
units metal
atom_style charge
boundary p p p
region box prism {LO[0]} {hi[0]} {LO[1]} {hi[1]} {LO[2]} {hi[2]} {XY} {XZ} {YZ}
create_box 3 box
mass 1 {MASS[1]}
mass 2 {MASS[2]}
mass 3 {MASS[3]}
labelmap atom 1 Si 2 Na 3 O
""")
    create(lmp, TYPES, LO + FRAC @ H0, IMAGE)
    lmp.commands_string(f"""
set type 1 charge {CHARGE[1]}
set type 2 charge {CHARGE[2]}
set type 3 charge {CHARGE[3]}
pair_style soft 1.0
pair_coeff * * 5.0
velocity all create 2000.0 4928459 dist gaussian mom yes rot no
set atom 1 vx 60.0
timestep 0.001
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
fix 1 all nve/limit 0.08
fix 2 all deform 1 x erate 0.3 xy erate 0.2 z erate -0.1 remap x
thermo 10
thermo_style custom step time temp pe lx ly lz xy xz yz
reset_timestep 100
write_data {HERE / 'topology.data'}
dump d1 all dcd 10 {HERE / 'traj.dcd'}
dump d2 all dcd 10 {HERE / 'traj_unwrap.dcd'}
dump_modify d2 unwrap yes
dump b1 all custom 10 {HERE / 'traj_custom.bin'} id type x y z ix iy iz vx vy vz q
dump_modify b1 time yes units yes
dump b2 all atom 10 {HERE / 'traj_atom.bin'}
dump b3 all custom 10 {HERE / 'traj_element.bin'} id element x y z
dump_modify b3 element Si Na O
dump y1 all yaml 10 {HERE / 'traj.yaml'} id type element x y z ix iy iz vx vy vz q
dump_modify y1 element Si Na O time yes units yes thermo yes
dump y2 all yaml 10 {HERE / 'traj_typelabel.yaml'} id typelabel x y z
dump c1 all cfg 10 {HERE / 'cfg' / 'dump.*.cfg'} mass type xs ys zs id q ix iy iz
dump_modify c1 element Si Na O
dump c2 all cfg 10 {HERE / 'cfg_unwrap' / 'dump.*.cfg'} mass type xsu ysu zsu id
dump_modify c2 element Si Na O
dump c3 all cfg 100 {HERE / 'cfg_noelement' / 'dump.*.cfg'} mass type xs ys zs id
""")
    states = []
    lmp.command("run 0")
    states.append(state(lmp))
    for _ in range(2):
        lmp.command("run 10")
        states.append(state(lmp))
    lmp.close()
    return states


def real_session() -> list[dict]:
    lmp = lmp_session("real")
    hi = LO_REAL + L_REAL
    lmp.commands_string(f"""
units real
atom_style charge
boundary p p p
region box block {LO_REAL[0]} {hi[0]} {LO_REAL[1]} {hi[1]} {LO_REAL[2]} {hi[2]}
create_box 3 box
mass 1 {MASS[1]}
mass 2 {MASS[2]}
mass 3 {MASS[3]}
""")
    create(lmp, TYPES_REAL, LO_REAL + FRAC_REAL * L_REAL,
           np.zeros((len(TYPES_REAL), 3), dtype=int))
    lmp.commands_string(f"""
set type 1 charge {CHARGE[1]}
set type 2 charge {CHARGE[2]}
set type 3 charge {CHARGE[3]}
pair_style soft 1.0
pair_coeff * * 5.0
velocity all create 1500.0 87287 dist gaussian mom yes rot no
timestep 1.0
neighbor 1.0 bin
fix 1 all nve/limit 0.05
write_data {HERE / 'topology_real.data'}
dump d1 all dcd 5 {HERE / 'traj_real.dcd'}
""")
    states = []
    lmp.command("run 0")
    states.append(state(lmp))
    for _ in range(2):
        lmp.command("run 5")
        states.append(state(lmp))
    lmp.close()
    return states


def ase_cfg(first: dict) -> None:
    from ase import Atoms
    from ase.io import write

    atoms = Atoms(symbols=[ELEMENT[int(t)] for t in first["types"]],
                  positions=first["x"] - first["lo"], cell=first["H"], pbc=True)
    write(HERE / "ase.cfg", atoms, format="cfg")


def main() -> None:
    import ase
    import lammps

    clean(["cfg", "cfg_unwrap", "cfg_noelement"])
    for sub in ("cfg", "cfg_unwrap", "cfg_noelement"):
        (HERE / sub).mkdir()
    metal = metal_session()
    real = real_session()
    ase_cfg(metal[0])
    arrays = {}
    for prefix, states, dt, units in (("metal_", metal, 0.001, "metal"),
                                      ("real_", real, 1.0, "real")):
        arrays.update({
            prefix + "ids": states[0]["ids"], prefix + "types": states[0]["types"],
            prefix + "q": states[0]["q"],
            prefix + "x": np.stack([s["x"] for s in states]),
            prefix + "image": np.stack([s["image"] for s in states]),
            prefix + "v": np.stack([s["v"] for s in states]),
            prefix + "H": np.stack([s["H"] for s in states]),
            prefix + "lo": np.stack([s["lo"] for s in states]),
            prefix + "steps": np.array([s["step"] for s in states]),
            prefix + "time": np.array([s["time"] for s in states]),
            prefix + "dt": np.array(dt), prefix + "units": np.array(units)})
    from importlib.metadata import version

    probe = lammps.lammps(cmdargs=["-screen", "none", "-log", "none"])
    arrays["writers"] = np.array(
        f"LAMMPS {probe.version()} (pip wheel lammps {version('lammps')}),"
        f" ASE {ase.__version__}, numpy {np.__version__}, Python "
        f"{sys.version.split()[0]}")
    probe.close()
    np.savez(HERE / "truth.npz", **arrays)
    for path in sorted(HERE.rglob("*")):
        if path.is_file() and path.suffix != ".py":
            print(f"{path.relative_to(HERE)}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
