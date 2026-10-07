"""Write the binary MD test files in this folder with the programs that write them.

Run with a Python that has ASE, gsd, netCDF4, MDAnalysis and OVITO installed
(not FACET's own environment; FACET reads these formats with numpy only)::

    <venv>/python tests/data/md/binary/make_binary_files.py

The files are committed. Each ``<file>.npz`` beside a file holds what its
writer was given, in the row order FACET is expected to return, and
``MANIFEST.json`` records the writer and its version for every file.
tests/test_md_formats_binary.py compares FACET's reading with both.

THE MODEL
---------
Eight atoms (Si2 Na2 O4, ids 3 5 8 11 12 17 20 21) in a restricted triclinic
box with a non-zero origin that changes every frame (three frames, steps 1000,
1010, 1020; a 1 fs step, so 1.000, 1.010, 1.020 ps). ``held`` positions are
what an MD code holds (two atoms sit just outside the box in some frames);
``image`` flags count the crossings, so ``held + image @ box`` is continuous.
Velocities are in Å/ps, charges in e. The numbers are test inputs, not a
material.

WHAT EACH FILE IS
-----------------
Written by the program itself (versions in MANIFEST.json):

* ``ase_md.traj``: ASE Trajectory, continuous positions, momenta (masses not
  set, so ASE stores none), initial charges, a SinglePointCalculator, info.
* ``ase_masses.traj``: ASE Trajectory with masses set (stored), a FixAtoms
  constraint, the held positions.
* ``ase_changed.traj``: ASE Trajectory whose frame 1 holds Ge for one Si, so
  ASE writes a header in frames 1 and 2.
* ``ovito_small.traj``, ``ovito_ambiguous.gsd``, ``ovito_small.nc``: OVITO's
  ase/traj, gsd/hoomd and netcdf/amber exporters, the atoms handed to OVITO in
  a shuffled order (the NetCDF file carries the ids; the ase/traj export keeps
  the order given, the GSD export sorts by id). In the GSD file every
  position lies well inside the box, so it cannot say which of two readings
  of OVITO's tilted box holds; in the NetCDF file OVITO's cell angles are
  those of the cell with its xy and yz tilts exchanged (MANIFEST.json,
  ``ovito_netcdf_angles``, records what OVITO writes for six simple cells).
* ``ovito_face.gsd``: as ``ovito_ambiguous.gsd`` with atom 11 near an x face,
  inside the box OVITO held and outside the schema's reading of what it wrote.
* ``ovito_xz.nc``: OVITO's netcdf/amber export of the model in a cell tilted
  in xz only, whose angles OVITO writes as the cell's own.
* ``hoomd_small.gsd``: the gsd package's hoomd module (HOOMD-blue's own file
  library): positions wrapped into the centred box with image flags, types,
  masses and charges in frame 0 only (later frames inherit them), a log chunk.
* ``ase_small.nc``: ASE NetCDFTrajectory (netCDF-C classic format).
* ``mda_small.nc``: MDAnalysis NCDFWriter (64-bit offset format), no types.
* ``ase_masses_later.traj``: ASE Trajectory whose masses are set from frame 1
  on, so frame 0 stores none and ASE writes a header with masses in frame 1.
* ``hoomd_letters.gsd``: gsd.hoomd with the type names 'A', 'B', 'C' (the
  letters HOOMD-blue's examples use) and the default mass 1, which the
  library leaves out: the file holds no particles/mass.
* ``hoomd_mass_labels.gsd``: gsd.hoomd with the type names 'Si_t', 'Na_m',
  'O_b' (not element symbols) and the standard atomic weights as masses.
* ``hoomd_static.gsd``: gsd.hoomd, three frames whose positions and box equal
  frame 0's, so the library writes only configuration/step in frames 1 and 2.
* ``mdtraj_nm.gsd``: mdtraj's save_gsd, which writes lengths in nm.
* ``mdtraj.ncrst``: mdtraj's save_netcdfrst, an AMBER restart (one frame, a
  time of length 1).

Written with the netCDF-C library through netCDF4-python, in a layout coded
here from a source (a construction, not a program's output):

* ``lammps_style.nc``: CDF-5, the layout of LAMMPS's dump netcdf
  (src/NETCDF/dump_netcdf.cpp, stable_22Jul2025) under ``dump_modify double
  yes``: ``program`` 'LAMMPS', a double ``time`` holding the step
  (``time = update->ntimestep``) with a float ``scale_factor`` = dt,
  ``cell_origin``, ``id`` and ``type`` variables, atoms in a different order
  in every frame.
* ``lammps_float.nc``: the same layout with LAMMPS's default precision
  (NC_FLOAT for every real, ``time`` included) and steps past 2**24, which
  float32 stores rounded (16777217 as 16777216, 20000001 as 20000000).
* ``amber_style.nc``: CDF-1, AMBER's convention
  (https://ambermd.org/netcdf/nctraj.xhtml): float coordinates, velocities in
  AMBER's internal unit with ``scale_factor`` = 20.455, no atom types.
* ``gsd_v1_synthetic.gsd``: ``hoomd_small.gsd`` with its header and name list
  rewritten in the GSD 1.0 layout (64-byte names), which the current gsd
  package no longer writes.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# No __pycache__ beside the fixtures.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent

SYMBOLS = np.array(["Si", "O", "Na", "O", "Si", "O", "Na", "O"])
IDS = np.array([3, 5, 8, 11, 12, 17, 20, 21])
TYPE_OF = {"Si": 1, "Na": 2, "O": 3}
MASS_OF = {"Si": 28.0855, "Na": 22.98977, "O": 15.9994}
CHARGE_OF = {"Si": 2.4, "Na": 0.6, "O": -1.5}
STEPS = np.array([1000, 1010, 1020])
DT_PS = 0.001
F = len(STEPS)
N = len(SYMBOLS)


def model():
    rng = np.random.default_rng(20261007)
    box = np.array([[[7.1 * (1 + 0.004 * k), 0.0, 0.0],
                     [1.3 + 0.02 * k, 6.8, 0.0],
                     [-0.9, 0.7, 7.4 * (1 - 0.002 * k)]] for k in range(F)])
    lo = np.array([[-1.5 - 0.01 * k, 0.8, -2.2 + 0.005 * k] for k in range(F)])
    frac = rng.uniform(0.08, 0.92, (N, 3))
    held_frac = []
    for k in range(F):
        f = frac + 0.01 * k * rng.uniform(-1, 1, (N, 3))
        if k >= 1:
            f[1, 0] = 1.012            # stored just outside the box
        if k == 2:
            f[6, 2] = -0.015
        held_frac.append(f)
    held_frac = np.array(held_frac)
    image = np.zeros((F, N, 3), dtype=np.int64)
    image[1, 2] = (1, 0, 0)
    image[2, 2] = (1, 0, 0)
    image[2, 5] = (0, -1, 2)
    image[1:, 7] = (0, 1, 0)
    held = np.array([lo[k] + held_frac[k] @ box[k] for k in range(F)])
    unwrapped = np.array([held[k] + image[k] @ box[k] for k in range(F)])
    vel = rng.normal(0.0, 6.0, (F, N, 3))
    charge = np.array([CHARGE_OF[s] for s in SYMBOLS])
    return dict(box=box, lo=lo, held=held, unwrapped=unwrapped, image=image,
                held_frac=held_frac, vel=vel, charge=charge)


M = model()
# programVersion of the files built in LAMMPS's dump netcdf layout: LAMMPS
# writes its version there; these say how they were made.
LAMMPS_LAYOUT_VERSION = ("22 Jul 2025 layout (src/NETCDF/dump_netcdf.cpp), "
                         "written by netCDF4-python for FACET's tests")
MANIFEST: dict[str, str] = {}       # file -> writer and version
RECORDS: dict[str, dict] = {}       # what the writers were measured to do


def truth(name, writer, **arrays):
    clean = {k: np.asarray(v) for k, v in arrays.items() if v is not None}
    np.savez_compressed(HERE / f"{name}.npz", **clean)
    MANIFEST[name] = writer


def versions():
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for module in ("ase", "gsd", "netCDF4", "MDAnalysis", "ovito", "mdtraj"):
        try:
            mod = __import__(module)
        except ImportError:
            continue
        if module == "gsd":
            import gsd.version
            out[module] = gsd.version.version
            continue
        out[module] = getattr(mod, "__version__", None) or \
            getattr(mod, "version_string", None)
    try:
        import netCDF4
        out["netcdf-c"] = netCDF4.__netcdf4libversion__
    except ImportError:
        pass
    return out


def cellpar(box):
    a, b, c = (np.linalg.norm(v) for v in box)
    alpha = np.degrees(np.arccos(np.dot(box[1], box[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(box[0], box[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(box[0], box[1]) / (a * b)))
    return np.array([a, b, c]), np.array([alpha, beta, gamma])


# ---------------------------------------------------------------------------
# ASE
# ---------------------------------------------------------------------------

def write_ase():
    import ase
    from ase import Atoms, units
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.constraints import FixAtoms
    from ase.io import Trajectory, write

    def atoms(k, positions, symbols=SYMBOLS):
        a = Atoms(symbols=list(symbols), positions=positions, cell=M["box"][k],
                  pbc=True)
        a.set_velocities(M["vel"][k] / (1000.0 * units.fs))
        return a

    ver = f"ASE {ase.__version__}"
    RECORDS["ase_units"] = {"fs": units.fs, "second": units.second,
                             "codata": units.__codata_version__}
    # 1. continuous positions, momenta, charges, a calculator, info
    with Trajectory(HERE / "ase_md.traj", "w") as traj:
        for k in range(F):
            a = atoms(k, M["unwrapped"][k])
            a.set_initial_charges(M["charge"])
            a.calc = SinglePointCalculator(a, energy=-12.5 - k,
                                           forces=np.full((N, 3), 0.1 * k))
            a.info["temperature_K"] = 300.0 + k
            traj.write(a)
    masses_used = atoms(0, M["held"][0]).get_masses()
    truth("ase_md.traj", ver, ids=np.arange(N), elements=SYMBOLS,
          pos=M["unwrapped"], box=M["box"], origin=np.zeros((F, 3)),
          vel=M["vel"], charge=M["charge"], ase_masses=masses_used)

    # 2. masses set, a constraint, held positions
    masses = np.array([MASS_OF[s] for s in SYMBOLS])
    with Trajectory(HERE / "ase_masses.traj", "w") as traj:
        for k in range(F):
            a = atoms(k, M["held"][k])
            a.set_masses(masses)
            a.set_velocities(M["vel"][k] / (1000.0 * units.fs))
            a.set_constraint(FixAtoms(indices=[0]))
            traj.write(a)
    truth("ase_masses.traj", ver, ids=np.arange(N), elements=SYMBOLS,
          pos=M["held"], box=M["box"], origin=np.zeros((F, 3)), vel=M["vel"],
          masses=masses)

    # 2b. masses set from frame 1 on: frame 0 stores none (ASE divided its
    # momenta by its own table), frames 1 and 2 store them
    with Trajectory(HERE / "ase_masses_later.traj", "w") as traj:
        for k in range(F):
            a = atoms(k, M["held"][k])
            if k >= 1:
                a.set_masses(masses)
            a.set_velocities(M["vel"][k] / (1000.0 * units.fs))
            traj.write(a)
    truth("ase_masses_later.traj", ver, ids=np.arange(N), elements=SYMBOLS,
          pos=M["held"], box=M["box"], origin=np.zeros((F, 3)), vel=M["vel"],
          masses=masses, ase_masses=masses_used)

    # 3. frame 1 holds another element for atom 0
    changed = SYMBOLS.copy()
    changed[0] = "Ge"
    with Trajectory(HERE / "ase_changed.traj", "w") as traj:
        for k in range(F):
            traj.write(atoms(k, M["held"][k], changed if k == 1 else SYMBOLS))
    keep = [0, 2]
    truth("ase_changed.traj", ver, ids=np.arange(N), elements=SYMBOLS,
          pos=M["held"][keep], box=M["box"][keep], origin=np.zeros((2, 3)),
          vel=M["vel"][keep])

    # 4. ASE's NetCDF trajectory
    frames = [atoms(k, M["unwrapped"][k]) for k in range(F)]
    write(HERE / "ase_small.nc", frames, format="netcdftrajectory")
    truth("ase_small.nc", ver + " (netCDF4 " + versions().get("netCDF4", "?")
          + ", netcdf-c " + versions().get("netcdf-c", "?") + ")",
          ids=np.arange(N), elements=SYMBOLS, pos=M["unwrapped"],
          box=M["box"], origin=np.zeros((F, 3)), vel=M["vel"],
          ase_masses=masses_used)


# ---------------------------------------------------------------------------
# OVITO
# ---------------------------------------------------------------------------

def schema_reading_of_ovito_gsd(rows):
    """The box the HOOMD schema reads from what OVITO's GSDExporter.cpp writes
    for ``rows``: (Lx, |b|, |c|, b_x/|b|, c_x/|c|, c_y/|c|) read as
    (Lx, Ly, Lz, xy, xz, yz) gives b = (b_x, |b|, 0), c = (c_x, c_y, |c|)."""
    a, b, c = rows
    return np.array([a, [b[0], np.linalg.norm(b), 0.0],
                     [c[0], c[1], np.linalg.norm(c)]])


def face_fraction():
    """Fractional coordinates inside the true box of every frame that lie
    outside the box the schema reads from OVITO's GSD export, so that the
    positions in ovito_face.gsd say which reading holds."""
    for fx in (0.001, 0.999):
        for fy in (0.02, 0.98):
            for fz in (0.02, 0.5, 0.98):
                f = np.array([fx, fy, fz])
                outside = True
                for k in range(F):
                    rows = M["box"][k]
                    spec = schema_reading_of_ovito_gsd(rows)
                    p = f @ rows - rows.sum(axis=0) / 2
                    g = np.linalg.solve(spec.T, p + spec.sum(axis=0) / 2)
                    outside &= bool(((g < 0) | (g >= 1)).any())
                if outside:
                    return f
    raise RuntimeError("no fractional position separates the two readings")


def ovito_netcdf_angles(export_file, Pipeline, PythonSource, Source0):
    """What OVITO's netcdf/amber exporter writes as cell_angles for simple
    cells, against the cells' own angles (kept in MANIFEST.json)."""
    import tempfile

    import netCDF4

    def angle(u, v):
        return float(np.degrees(np.arccos(
            u @ v / np.linalg.norm(u) / np.linalg.norm(v))))

    def angles_of(rows):
        a, b, c = rows
        return [angle(b, c), angle(a, c), angle(a, b)]

    cells = {
        "orthogonal": [[10, 0, 0], [0, 11, 0], [0, 0, 12]],
        "xy": [[10, 0, 0], [2, 11, 0], [0, 0, 12]],
        "xz": [[10, 0, 0], [0, 11, 0], [2, 0, 12]],
        "yz": [[10, 0, 0], [0, 11, 0], [0, 2, 12]],
        "xz+yz": [[10, 0, 0], [0, 11, 0], [2, 3, 12]],
        "xy+xz+yz": [[10, 0, 0], [2, 11, 0], [-1, 3, 12]],
    }
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, rows in cells.items():
            rows = np.array(rows, dtype=float)
            pipe = Pipeline(source=PythonSource(delegate=Source0(rows)))
            path = Path(tmp) / "cell.nc"
            export_file(pipe, str(path), "netcdf/amber",
                        columns=["Particle Type", "Position.X", "Position.Y",
                                 "Position.Z"])
            ds = netCDF4.Dataset(path)
            ds.set_auto_mask(False)
            out[label] = {"rows": rows.tolist(), "cell_angles": angles_of(rows),
                          "written_lengths": np.asarray(ds["cell_lengths"][0]).tolist(),
                          "written_angles": np.asarray(ds["cell_angles"][0]).tolist()}
            ds.close()
    return out


def write_ovito():
    import ovito
    from ovito.data import DataCollection, ParticleType
    from ovito.io import export_file
    from ovito.pipeline import Pipeline, PipelineSourceInterface, PythonSource

    order = np.array([5, 2, 7, 0, 3, 6, 1, 4])     # the order OVITO is given

    class Source(PipelineSourceInterface):
        def __init__(self, held, box):
            self.held, self.box = held, box

        def compute_trajectory_length(self, **kwargs):
            return F

        def create(self, data: DataCollection, *, frame: int, **kwargs):
            mat = np.zeros((3, 4))
            mat[:, :3] = self.box[frame].T
            mat[:, 3] = M["lo"][frame]
            data.create_cell(mat, pbc=(True, True, True))
            p = data.create_particles(count=N)
            p.create_property("Position", data=self.held[frame][order])
            p.create_property("Particle Identifier", data=IDS[order])
            tp = p.create_property("Particle Type",
                                   data=[TYPE_OF[s] for s in SYMBOLS[order]])
            for name, t in TYPE_OF.items():
                tp.types.append(ParticleType(id=t, name=name, mass=MASS_OF[name]))
            p.create_property("Velocity", data=M["vel"][frame][order])
            p.create_property("Charge", data=M["charge"][order])
            p.create_property("Periodic Image", data=M["image"][frame][order])
            data.attributes["Timestep"] = int(STEPS[frame])

    class OneCell(PipelineSourceInterface):
        def __init__(self, rows):
            self.rows = rows

        def create(self, data: DataCollection, **kwargs):
            mat = np.zeros((3, 4))
            mat[:, :3] = self.rows.T
            data.create_cell(mat, pbc=(True, True, True))
            p = data.create_particles(count=1)
            p.create_property("Position", data=[[1.0, 1.0, 1.0]])
            tp = p.create_property("Particle Type", data=[1])
            tp.types.append(ParticleType(id=1, name="Si"))

    ver = f"OVITO {ovito.version_string}"
    pipe = Pipeline(source=PythonSource(delegate=Source(M["held"], M["box"])))
    export_file(pipe, str(HERE / "ovito_small.traj"), "ase/traj",
                multiple_frames=True)
    truth("ovito_small.traj", ver, ids=np.arange(N), elements=SYMBOLS[order],
          pos=M["held"][:, order], box=M["box"], origin=M["lo"],
          steps=STEPS)

    # GSD: positions centred on zero; truth in the frame OVITO writes
    def gsd_truth(held):
        corner = -M["box"].sum(axis=1) / 2.0
        return held - M["lo"][:, None, :] + corner[:, None, :], corner

    # OVITO's GSD exporter writes the particles sorted by identifier
    # (measured: the rows come back in id order, not in the order given).
    # Every position well inside the box: the file cannot say which box.
    export_file(pipe, str(HERE / "ovito_ambiguous.gsd"), "gsd/hoomd",
                multiple_frames=True)
    pos, corner = gsd_truth(M["held"])
    truth("ovito_ambiguous.gsd", ver, ids=np.arange(N), elements=SYMBOLS,
          pos=pos, box=M["box"], origin=corner, steps=STEPS, vel=M["vel"],
          charge=M["charge"])
    # atom 3 near an x face, inside the true box and outside the schema's
    # reading of OVITO's box: the positions decide
    held_face = M["held"].copy()
    f = face_fraction()
    for k in range(F):
        held_face[k, 3] = M["lo"][k] + f @ M["box"][k]
    pipe_face = Pipeline(source=PythonSource(delegate=Source(held_face, M["box"])))
    export_file(pipe_face, str(HERE / "ovito_face.gsd"), "gsd/hoomd",
                multiple_frames=True)
    pos, corner = gsd_truth(held_face)
    truth("ovito_face.gsd", ver, ids=np.arange(N), elements=SYMBOLS,
          pos=pos, box=M["box"], origin=corner, steps=STEPS, vel=M["vel"],
          charge=M["charge"], face_fraction=f)

    cols = ["Particle Identifier", "Particle Type", "Position.X", "Position.Y",
            "Position.Z", "Velocity.X", "Velocity.Y", "Velocity.Z", "Charge"]
    # NetCDF, the tilted cell: OVITO writes exchanged angles
    export_file(pipe, str(HERE / "ovito_small.nc"), "netcdf/amber",
                columns=cols, multiple_frames=True)
    truth("ovito_small.nc", ver, ids=IDS, elements=SYMBOLS, pos=M["held"],
          box=M["box"], origin=M["lo"], steps=STEPS, vel=M["vel"],
          charge=M["charge"], types=np.array([TYPE_OF[s] for s in SYMBOLS]))
    # NetCDF, a cell tilted in xz only, which OVITO's angles describe
    box_xz = M["box"].copy()
    box_xz[:, 1, 0] = 0.0
    box_xz[:, 2, 1] = 0.0
    held_xz = np.array([M["lo"][k] + M["held_frac"][k] @ box_xz[k]
                        for k in range(F)])
    pipe_xz = Pipeline(source=PythonSource(delegate=Source(held_xz, box_xz)))
    export_file(pipe_xz, str(HERE / "ovito_xz.nc"), "netcdf/amber",
                columns=cols, multiple_frames=True)
    truth("ovito_xz.nc", ver, ids=IDS, elements=SYMBOLS, pos=held_xz,
          box=box_xz, origin=M["lo"], steps=STEPS, vel=M["vel"],
          charge=M["charge"], types=np.array([TYPE_OF[s] for s in SYMBOLS]))
    RECORDS["ovito_netcdf_angles"] = ovito_netcdf_angles(
        export_file, Pipeline, PythonSource, OneCell)


# ---------------------------------------------------------------------------
# gsd (HOOMD-blue's file library)
# ---------------------------------------------------------------------------

def hoomd_box(box):
    """The HOOMD box (Lx, Ly, Lz, xy, xz, yz) of rows a = (Lx, 0, 0),
    b = (xy Ly, Ly, 0), c = (xz Lz, yz Lz, Lz) [HOOMD schema, particles/image]."""
    lx, ly, lz = box[0, 0], box[1, 1], box[2, 2]
    return [lx, ly, lz, box[1, 0] / ly, box[2, 0] / lz, box[2, 1] / lz]


def write_gsd():
    import gsd
    import gsd.hoomd
    import gsd.version

    pos, img, centre = [], [], []
    with gsd.hoomd.open(HERE / "hoomd_small.gsd", "w") as traj:
        for k in range(F):
            box = M["box"][k]
            frac = M["held_frac"][k]
            shift = np.floor(frac)
            corner = -box.sum(axis=0) / 2.0
            p = corner + (frac - shift) @ box
            i = M["image"][k] + shift.astype(np.int64)
            f = gsd.hoomd.Frame()
            f.configuration.step = int(STEPS[k])
            f.configuration.box = hoomd_box(box)
            f.particles.N = N
            f.particles.types = ["Si", "Na", "O"]
            f.particles.typeid = [["Si", "Na", "O"].index(s) for s in SYMBOLS]
            f.particles.mass = [MASS_OF[s] for s in SYMBOLS]
            f.particles.charge = M["charge"]
            f.particles.position = p
            f.particles.velocity = M["vel"][k]
            f.particles.image = i
            f.log["md/compute/ThermodynamicQuantities/potential_energy"] = \
                [-12.5 - k]
            traj.append(f)
            pos.append(p)
            img.append(i)
            centre.append(corner)
    with gsd.hoomd.open(HERE / "hoomd_small.gsd", "r") as traj:
        oracle_pos = np.array([traj[k].particles.position for k in range(F)])
        oracle_box = np.array([traj[k].configuration.box for k in range(F)])
        oracle_img = np.array([traj[k].particles.image for k in range(F)])
        oracle_vel = np.array([traj[k].particles.velocity for k in range(F)])
    truth("hoomd_small.gsd", f"gsd {gsd.version.version} (gsd.hoomd)",
          ids=np.arange(N), elements=SYMBOLS, pos=np.array(pos),
          box=M["box"], origin=np.array(centre), steps=STEPS, vel=M["vel"],
          charge=M["charge"],
          unwrapped=M["unwrapped"] - M["lo"][:, None, :] + np.array(centre)[:, None, :],
          oracle_pos=oracle_pos, oracle_box=oracle_box, oracle_img=oracle_img,
          oracle_vel=oracle_vel)
    # the same file in the GSD 1.0 layout: names in 64-byte segments
    raw = bytearray((HERE / "hoomd_small.gsd").read_bytes())
    (magic, index_location, index_allocated, namelist_location,
     namelist_allocated, schema_version, gsd_version) = \
        struct.unpack_from("<QQQQQII", raw, 0)
    block = bytes(raw[namelist_location:namelist_location + 64 * namelist_allocated])
    names = []
    for name in block.split(b"\0"):
        if not name:
            break
        names.append(name)
    v1 = b"".join(n.ljust(64, b"\0") for n in names)
    if len(v1) > 64 * namelist_allocated:
        raise RuntimeError("the names do not fit the allocated name list")
    raw[namelist_location:namelist_location + 64 * namelist_allocated] = \
        v1.ljust(64 * namelist_allocated, b"\0")
    struct.pack_into("<I", raw, 44, 1 << 16)          # gsd_version 1.0
    (HERE / "gsd_v1_synthetic.gsd").write_bytes(bytes(raw))
    shutil.copy(HERE / "hoomd_small.gsd.npz", HERE / "gsd_v1_synthetic.gsd.npz")
    MANIFEST["gsd_v1_synthetic.gsd"] = (
        "hoomd_small.gsd rewritten by this script in the GSD 1.0 layout "
        "(synthetic)")

    # the same frames under other type names: letters with the default mass
    # (the library then writes no particles/mass), and labels that are not
    # element symbols with the standard atomic weights
    def variant(name, types, with_mass):
        with gsd.hoomd.open(HERE / name, "w") as traj:
            for k in range(F):
                f = gsd.hoomd.Frame()
                f.configuration.step = int(STEPS[k])
                f.configuration.box = hoomd_box(M["box"][k])
                f.particles.N = N
                f.particles.types = types
                f.particles.typeid = [["Si", "Na", "O"].index(s) for s in SYMBOLS]
                if with_mass:
                    f.particles.mass = [MASS_OF[s] for s in SYMBOLS]
                f.particles.charge = M["charge"]
                f.particles.position = pos[k]
                f.particles.velocity = M["vel"][k]
                f.particles.image = img[k]
                traj.append(f)
        shutil.copy(HERE / "hoomd_small.gsd.npz", HERE / f"{name}.npz")
        MANIFEST[name] = f"gsd {gsd.version.version} (gsd.hoomd)"

    variant("hoomd_letters.gsd", ["A", "B", "C"], False)
    variant("hoomd_mass_labels.gsd", ["Si_t", "Na_m", "O_b"], True)

    # positions and box equal to frame 0's in every frame: the library
    # writes only configuration/step in frames 1 and 2. It compares a frame
    # with frame 0 as read back from the file (float32), so the values are
    # given in float32.
    with gsd.hoomd.open(HERE / "hoomd_static.gsd", "w") as traj:
        for k in range(F):
            f = gsd.hoomd.Frame()
            f.configuration.step = int(STEPS[k])
            f.configuration.box = np.array(hoomd_box(M["box"][0]),
                                           dtype=np.float32)
            f.particles.N = N
            f.particles.types = ["Si", "Na", "O"]
            f.particles.typeid = [["Si", "Na", "O"].index(s) for s in SYMBOLS]
            f.particles.mass = np.array([MASS_OF[s] for s in SYMBOLS],
                                        dtype=np.float32)
            f.particles.position = pos[0].astype(np.float32)
            traj.append(f)
    truth("hoomd_static.gsd", f"gsd {gsd.version.version} (gsd.hoomd)",
          ids=np.arange(N), elements=SYMBOLS, pos=np.array([pos[0]] * F),
          box=np.array([M["box"][0]] * F), origin=np.array([centre[0]] * F),
          steps=STEPS)


# ---------------------------------------------------------------------------
# NetCDF
# ---------------------------------------------------------------------------

def write_mdanalysis():
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    u = mda.Universe.empty(N, trajectory=True, velocities=True)
    dims = []
    for k in range(F):
        lengths, angles = cellpar(M["box"][k])
        dims.append(np.concatenate([lengths, angles]))
    u.load_new(M["held"].astype(np.float32), format=MemoryReader,
               velocities=M["vel"].astype(np.float32),
               dimensions=np.array(dims, dtype=np.float32), dt=0.01)
    with mda.Writer(str(HERE / "mda_small.nc"), n_atoms=N,
                    velocities=True) as writer:
        for _ in u.trajectory:
            writer.write(u)
    truth("mda_small.nc", f"MDAnalysis {mda.__version__} NCDFWriter",
          ids=np.arange(N), elements=SYMBOLS, pos=M["held"], box=M["box"],
          origin=np.zeros((F, 3)), vel=M["vel"], time_ps=np.arange(F) * 0.01)


def write_netcdf_layouts():
    import netCDF4

    lib = f"netCDF4 {netCDF4.__version__}, netcdf-c {netCDF4.__netcdf4libversion__}"
    rng = np.random.default_rng(5)
    # LAMMPS dump netcdf layout, CDF-5
    ds = netCDF4.Dataset(HERE / "lammps_style.nc", "w",
                         format="NETCDF3_64BIT_DATA")
    ds.Conventions = "AMBER"
    ds.ConventionVersion = "1.0"
    ds.program = "LAMMPS"
    ds.programVersion = LAMMPS_LAYOUT_VERSION
    ds.createDimension("frame", None)
    ds.createDimension("spatial", 3)
    ds.createDimension("atom", N)
    ds.createDimension("cell_spatial", 3)
    ds.createDimension("cell_angular", 3)
    ds.createDimension("label", 10)
    ds.createVariable("spatial", "S1", ("spatial",))[:] = list("xyz")
    ds.createVariable("cell_spatial", "S1", ("cell_spatial",))[:] = list("abc")
    angular = ds.createVariable("cell_angular", "S1", ("cell_angular", "label"))
    for i, word in enumerate(("alpha", "beta", "gamma")):
        angular[i, :len(word)] = list(word)
    # dump_netcdf.cpp: time = update->ntimestep (line 660) in a variable of
    # type_nc_real (line 385), NC_DOUBLE under dump_modify double yes
    # (line 861), with a float scale_factor = dt (line 473)
    time = ds.createVariable("time", "f8", ("frame",))
    time.units = "picosecond"
    time.scale_factor = np.float32(DT_PS)
    origin = ds.createVariable("cell_origin", "f8", ("frame", "cell_spatial"))
    origin.units = "Angstrom"
    lengths = ds.createVariable("cell_lengths", "f8", ("frame", "cell_spatial"))
    lengths.units = "Angstrom"
    angles = ds.createVariable("cell_angles", "f8", ("frame", "cell_angular"))
    angles.units = "degree"
    coords = ds.createVariable("coordinates", "f8", ("frame", "atom", "spatial"))
    coords.units = "Angstrom"
    velocities = ds.createVariable("velocities", "f8", ("frame", "atom", "spatial"))
    velocities.units = "Angstrom/picosecond"
    ids = ds.createVariable("id", "i4", ("frame", "atom"))
    types = ds.createVariable("type", "i4", ("frame", "atom"))
    ds.set_auto_scale(False)
    for k in range(F):
        perm = rng.permutation(N)
        time[k] = float(STEPS[k])
        origin[k] = M["lo"][k]
        lengths[k], angles[k] = cellpar(M["box"][k])
        coords[k] = M["held"][k][perm]
        velocities[k] = M["vel"][k][perm]
        ids[k] = IDS[perm]
        types[k] = [TYPE_OF[s] for s in SYMBOLS[perm]]
    ds.close()
    truth("lammps_style.nc", lib, ids=IDS, elements=SYMBOLS, pos=M["held"],
          box=M["box"], origin=M["lo"], steps=STEPS, vel=M["vel"],
          time_ps=STEPS * np.float64(np.float32(DT_PS)),
          types=np.array([TYPE_OF[s] for s in SYMBOLS]))

    # AMBER convention, CDF-1, velocities in AMBER's unit with scale_factor
    ds = netCDF4.Dataset(HERE / "amber_style.nc", "w", format="NETCDF3_CLASSIC")
    ds.Conventions = "AMBER"
    ds.ConventionVersion = "1.0"
    ds.program = "FACET test fixture (AMBER convention)"
    ds.programVersion = "1"
    ds.createDimension("frame", None)
    ds.createDimension("spatial", 3)
    ds.createDimension("atom", N)
    ds.createDimension("cell_spatial", 3)
    ds.createDimension("cell_angular", 3)
    ds.createDimension("label", 5)
    ds.createVariable("spatial", "S1", ("spatial",))[:] = list("xyz")
    time = ds.createVariable("time", "f4", ("frame",))
    time.units = "picosecond"
    coords = ds.createVariable("coordinates", "f4", ("frame", "atom", "spatial"))
    coords.units = "angstrom"
    lengths = ds.createVariable("cell_lengths", "f8", ("frame", "cell_spatial"))
    lengths.units = "angstrom"
    angles = ds.createVariable("cell_angles", "f8", ("frame", "cell_angular"))
    angles.units = "degree"
    velocities = ds.createVariable("velocities", "f4", ("frame", "atom", "spatial"))
    velocities.units = "angstrom/picosecond"
    velocities.scale_factor = 20.455
    ds.set_auto_scale(False)
    for k in range(F):
        time[k] = STEPS[k] * DT_PS
        coords[k] = M["held"][k]
        lengths[k], angles[k] = cellpar(M["box"][k])
        velocities[k] = M["vel"][k] / 20.455
    ds.close()
    truth("amber_style.nc", lib, ids=np.arange(N), elements=SYMBOLS,
          pos=M["held"], box=M["box"], origin=np.zeros((F, 3)), vel=M["vel"],
          time_ps=(STEPS * DT_PS).astype(np.float32).astype(np.float64))

    # LAMMPS dump netcdf with its default precision: every real NC_FLOAT
    # (dump_netcdf.cpp line 195), the step in a float time; steps past 2**24
    steps = np.array([16777200, 16777217, 20000001])
    ds = netCDF4.Dataset(HERE / "lammps_float.nc", "w",
                         format="NETCDF3_64BIT_DATA")
    ds.Conventions = "AMBER"
    ds.ConventionVersion = "1.0"
    ds.program = "LAMMPS"
    ds.programVersion = LAMMPS_LAYOUT_VERSION
    ds.createDimension("frame", None)
    ds.createDimension("spatial", 3)
    ds.createDimension("atom", N)
    ds.createDimension("cell_spatial", 3)
    ds.createDimension("cell_angular", 3)
    ds.createDimension("label", 10)
    ds.createVariable("spatial", "S1", ("spatial",))[:] = list("xyz")
    ds.createVariable("cell_spatial", "S1", ("cell_spatial",))[:] = list("abc")
    time = ds.createVariable("time", "f4", ("frame",))
    time.units = "picosecond"
    time.scale_factor = np.float32(DT_PS)
    origin = ds.createVariable("cell_origin", "f4", ("frame", "cell_spatial"))
    origin.units = "angstrom"
    lengths = ds.createVariable("cell_lengths", "f4", ("frame", "cell_spatial"))
    lengths.units = "angstrom"
    angles = ds.createVariable("cell_angles", "f4", ("frame", "cell_angular"))
    angles.units = "degree"
    coords = ds.createVariable("coordinates", "f4", ("frame", "atom", "spatial"))
    coords.units = "angstrom"
    ids = ds.createVariable("id", "i4", ("frame", "atom"))
    types = ds.createVariable("type", "i4", ("frame", "atom"))
    ds.set_auto_scale(False)
    for k in range(F):
        time[k] = float(steps[k])
        origin[k] = M["lo"][k]
        lengths[k], angles[k] = cellpar(M["box"][k])
        coords[k] = M["held"][k]
        ids[k] = IDS
        types[k] = [TYPE_OF[s] for s in SYMBOLS]
    ds.close()
    stored = steps.astype(np.float32).astype(np.int64)
    truth("lammps_float.nc", lib, ids=IDS, elements=SYMBOLS, pos=M["held"],
          box=M["box"], origin=M["lo"], steps=steps, stored_steps=stored,
          time_ps=stored * np.float64(np.float32(DT_PS)),
          types=np.array([TYPE_OF[s] for s in SYMBOLS]))


# ---------------------------------------------------------------------------
# mdtraj
# ---------------------------------------------------------------------------

def write_mdtraj():
    """mdtraj keeps lengths in nm: its GSD writer stores them as they are
    (mdtraj/formats/gsd.py, write_gsd), its AMBER restart writer converts
    them to Å. The order of mdtraj's GSD type names follows a Python set,
    which is why main() runs this script with PYTHONHASHSEED=0."""
    import mdtraj as md

    ver = f"mdtraj {md.__version__}"
    top = md.Topology()
    chain = top.add_chain()
    for symbol in SYMBOLS:
        top.add_atom(symbol, md.element.get_by_symbol(str(symbol)),
                     top.add_residue("X", chain))
    cells = [cellpar(M["box"][k]) for k in range(F)]
    traj = md.Trajectory(M["held"] / 10.0, top,
                         time=STEPS * DT_PS,
                         unitcell_lengths=np.array([c[0] for c in cells]) / 10.0,
                         unitcell_angles=np.array([c[1] for c in cells]))
    traj.save_gsd(str(HERE / "mdtraj_nm.gsd"))
    truth("mdtraj_nm.gsd", ver + " save_gsd (lengths in nm)",
          ids=np.arange(N), elements=SYMBOLS, pos=M["held"], box=M["box"],
          origin=-M["box"].sum(axis=1) / 2.0)
    traj[0].save_netcdfrst(str(HERE / "mdtraj.ncrst"))
    truth("mdtraj.ncrst", ver + " save_netcdfrst", ids=np.arange(N),
          elements=SYMBOLS, pos=M["held"][:1], box=M["box"][:1],
          origin=np.zeros((1, 3)), time_ps=np.array([STEPS[0] * DT_PS]))


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        # mdtraj orders its GSD type names as a set of strings does
        env = dict(os.environ, PYTHONHASHSEED="0")
        sys.exit(subprocess.run([sys.executable, "-B", __file__],
                                env=env).returncode)
    write_ase()
    write_ovito()
    write_gsd()
    write_mdanalysis()
    write_netcdf_layouts()
    write_mdtraj()
    doc = {"versions": versions(), "files": MANIFEST, "records": RECORDS,
           "model": {"symbols": SYMBOLS.tolist(), "ids": IDS.tolist(),
                     "steps": STEPS.tolist(), "dt_ps": DT_PS}}
    (HERE / "MANIFEST.json").write_text(json.dumps(doc, indent=1) + "\n")
    for path in sorted(HERE.iterdir()):
        if path.suffix != ".py" and path.is_file():
            print(f"{path.name:28s} {path.stat().st_size:7d} bytes")


if __name__ == "__main__":
    main()
