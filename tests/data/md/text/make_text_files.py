"""Write the md_formats_text test files in this folder with the programs that
write them, from one six-atom model.

Run with a Python that has ASE, mdtraj, MDAnalysis and OVITO installed (the
files here were written with the versions in writers.json):

    <python with those packages> tests/data/md/text/make_text_files.py

The files are committed; tests/test_md_formats_text.py reads them and
truth.json, and never runs this script. This script imports nothing from
FACET.

THE MODEL
---------
Six atoms (Si, O, O, Na, O, Si) in three frames. Each frame's box is lower
triangular (a along x, b in the xy plane), the orientation PDB's CRYST1 and
ASE's standard form use, so every writer stores the same Cartesian frame, and
the box changes between frames (an NPT run). Frame 1 places two atoms outside
the box, so a reader has to wrap them. Velocities are in Å/ps, times in ps.
The numbers are test inputs, not a material.

WHAT WRITES WHAT
----------------
* ``ase.md``: ASE ``castep-md`` (ASE writes the image index, 0, 1, 2, as the
  time line, and its CODATA 2002 Bohr); ``ase.geom``: ASE ``castep-geom``
  (the iteration index on each step's '<-- c' line).
* ``castep_doc_example.md``: the one-step example of "The .md file" in D.
  Quigley's CASTEP MD guide (https://www.tcm.phy.cam.ac.uk/castep/MD/
  node13.html), copied line by line with the documented Fortran formats: the
  only CASTEP-written layout available here (E, T, P, h, hv, S, R, V, F).
* ``ase_vel.gro``: ASE ``gromacs``, frame 0 with velocities (3 and 4
  decimals); ``mdtraj.gro``: mdtraj ``save_gro``, all frames, 6 decimals,
  ``t=`` in the title.
* ``ase_var.axsf``: ASE ``xsf``, all frames (variable cell, PRIMVEC per
  step); ``ase_fixed.axsf``: ASE ``xsf`` with frame 0's cell for every frame
  (one PRIMVEC).
* ``ase_models.pdb``: ASE ``proteindatabank`` (CRYST1 before each MODEL);
  ``mda_models.pdb``: MDAnalysis ``PDBWriter`` with ``multiframe=True``.
* ``ovito.imd`` and ``ovito_mass.imd``: OVITO ``imd`` export of frame 0 (the
  second with a mass column and velocities).
* ``POSCAR_ase_000`` .. ``002``: ASE ``vasp`` (direct); ``ovito_poscar.0.vasp``
  .. ``2``: OVITO ``vasp`` with a wildcard, from an extended XYZ ASE wrote.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ELEMENTS = ["Si", "O", "O", "Na", "O", "Si"]
MASSES = {"Si": 28.0855, "O": 15.9994, "Na": 22.98977}
# Lower-triangular boxes, rows a, b, c (Å).
BOXES = np.array([
    [[8.0, 0.0, 0.0], [1.2, 7.5, 0.0], [-0.8, 0.6, 9.0]],
    [[8.1, 0.0, 0.0], [1.1, 7.6, 0.0], [-0.7, 0.5, 9.1]],
    [[8.2, 0.0, 0.0], [1.0, 7.7, 0.0], [-0.6, 0.4, 9.2]],
])
FRAC = np.array([
    [[0.10, 0.20, 0.30], [0.25, 0.15, 0.45], [0.60, 0.70, 0.20],
     [0.85, 0.40, 0.90], [0.35, 0.90, 0.65], [0.55, 0.45, 0.75]],
    [[0.12, 0.22, 0.28], [1.05, 0.10, 0.50], [0.58, -0.05, 0.22],
     [0.87, 0.42, 0.88], [0.33, 0.88, 0.67], [0.57, 0.47, 0.73]],
    [[0.14, 0.24, 0.26], [0.95, 0.12, 0.52], [0.56, 0.93, 0.24],
     [0.89, 0.44, 0.86], [0.31, 0.86, 0.69], [0.59, 0.49, 0.71]],
])
POSITIONS = np.einsum("fij,fjk->fik", FRAC, BOXES)      # Å, any image
VELOCITIES = np.array([                                    # Å/ps
    [[1.5, -2.0, 0.5], [-3.0, 1.0, 2.0], [0.25, 0.75, -1.5],
     [4.0, -0.5, 1.0], [-1.0, -1.0, 3.0], [2.0, 2.5, -0.5]],
    [[1.4, -2.1, 0.6], [-2.9, 1.1, 2.1], [0.35, 0.65, -1.4],
     [3.9, -0.6, 1.1], [-1.1, -0.9, 2.9], [2.1, 2.4, -0.6]],
    [[1.3, -2.2, 0.7], [-2.8, 1.2, 2.2], [0.45, 0.55, -1.3],
     [3.8, -0.7, 1.2], [-1.2, -0.8, 2.8], [2.2, 2.3, -0.7]],
])
TIMES_PS = [0.0, 0.5, 1.0]

# The example step of "The .md file" (Quigley, CASTEP MD guide), as printed
# there; the header is the documented one.
CASTEP_DOC_EXAMPLE = """\
 BEGIN header

 This is 8 atom cubic Si cell
 END header

               0.00000000E+000
              -3.18206146E+001     -3.18108683E+001      9.74270683E-003  <-- E
                9.27876841E-04                                            <-- T
                5.85402338E-06                                            <-- P
               1.01599045E+001      0.00000000E+000      0.00000000E+000  <-- h
               1.29430839E-017      1.01599045E+001      0.00000000E+000  <-- h
               1.29430839E-017      1.29430839E-017      1.01599045E+001  <-- h
               2.80052926E-008     -1.42751448E-007     -1.35787248E-007  <-- hv
              -1.42751448E-007      2.76907508E-008     -1.44694219E-007  <-- hv
              -1.35787248E-007     -1.44694219E-007      2.71532850E-008  <-- hv
              -6.21684372E-006      3.03062374E-005      3.23291890E-005  <-- S
               3.03062374E-005     -6.06171719E-006      3.31426773E-005  <-- S
               3.23291890E-005      3.31426773E-005     -5.79666096E-006  <-- S
 Si     1      1.04834750E-002      1.15560090E-002      7.82230990E-003  <-- R
 Si     2     -3.20400394E-003      5.07172565E+000      5.10986361E+000  <-- R
 Si     3      5.07271954E+000      5.11656710E+000     -2.45225140E-003  <-- R
 Si     4      5.10789066E+000     -2.63240242E-002      5.09753015E+000  <-- R
 Si     5      7.57309188E+000      2.52845762E+000      7.57612741E+000  <-- R
 Si     6      2.55271297E+000      2.54057876E+000      2.53924276E+000  <-- R
 Si     7      2.53422324E+000      7.64091983E+000      7.62428191E+000  <-- R
 Si     8      7.63167143E+000      7.59610368E+000      2.52717191E+000  <-- R
 Si     1      5.77278549E-005      7.23673746E-005      4.30349159E-005  <-- V
 Si     2     -1.72415752E-005     -5.23270551E-005      1.75385181E-004  <-- V
 Si     3     -4.17085102E-005      2.14539848E-004     -1.78769096E-005  <-- V
 Si     4      1.58756714E-004     -1.60916056E-004      9.49147966E-005  <-- V
 Si     5     -2.70431102E-004     -7.18757382E-005     -2.43440176E-004  <-- V
 Si     6      7.51663795E-005      8.70332331E-008     -8.82271461E-006  <-- V
 Si     7     -4.19898677E-005      1.31687892E-004      3.60873179E-005  <-- V
 Si     8      7.97201068E-005     -1.33563298E-004     -7.92824105E-005  <-- V
 Si     1     -4.23569381E-003      2.52214252E-003     -2.46018145E-003  <-- F
 Si     2      3.06338418E-003     -2.16206923E-003     -5.29928454E-003  <-- F
 Si     3      7.37259995E-004     -4.73479365E-003     -2.20030945E-003  <-- F
 Si     4     -6.89050954E-003     -2.25153379E-003     -8.37626756E-003  <-- F
 Si     5      4.53551291E-003      3.34757083E-003      1.02791157E-002  <-- F
 Si     6      2.06737343E-003      2.05175887E-003      2.70990363E-003  <-- F
 Si     7     -1.41457164E-003     -9.15433604E-004      1.54850432E-003  <-- F
 Si     8      2.13724448E-003      2.14235805E-003      3.79851935E-003  <-- F

"""


def ase_frames():
    from ase import Atoms, units

    frames = []
    for k in range(3):
        atoms = Atoms(ELEMENTS, positions=POSITIONS[k], cell=BOXES[k], pbc=True)
        atoms.set_masses([MASSES[e] for e in ELEMENTS])
        # ASE velocity unit: Å per ASE time unit; 1 ps = 1000 * units.fs.
        atoms.set_velocities(VELOCITIES[k] / (1000.0 * units.fs))
        frames.append(atoms)
    return frames


def write_ase():
    from ase.io import write

    frames = ase_frames()
    write(HERE / "ase.md", frames, format="castep-md")
    write(HERE / "ase.geom", frames, format="castep-geom")
    write(HERE / "ase_vel.gro", frames[0], format="gromacs")
    write(HERE / "ase_var.axsf", frames, format="xsf")
    fixed = [a.copy() for a in frames]
    for a, k in zip(fixed, range(3)):
        a.set_cell(BOXES[0], scale_atoms=False)
        a.positions = POSITIONS[k]
    write(HERE / "ase_fixed.axsf", fixed, format="xsf")
    write(HERE / "ase_models.pdb", frames, format="proteindatabank")
    for k, a in enumerate(frames):
        write(HERE / f"POSCAR_ase_{k:03d}", a, format="vasp", direct=True)
    write(HERE / "_frames.extxyz", frames, format="extxyz")


def write_mdtraj():
    import mdtraj

    top = mdtraj.Topology()
    chain = top.add_chain()
    residue = top.add_residue("GLS", chain)
    for e in ELEMENTS:
        top.add_atom(e, mdtraj.element.get_by_symbol(e), residue)
    traj = mdtraj.Trajectory(POSITIONS / 10.0, top, time=TIMES_PS)
    traj.unitcell_vectors = BOXES / 10.0
    traj.save_gro(str(HERE / "mdtraj.gro"), precision=6)


def write_mdanalysis():
    import MDAnalysis as mda
    from MDAnalysis.lib.mdamath import triclinic_box

    u = mda.Universe.empty(len(ELEMENTS), trajectory=True)
    u.add_TopologyAttr("names", ELEMENTS)
    u.add_TopologyAttr("elements", ELEMENTS)
    u.add_TopologyAttr("resnames", ["GLS"])
    with mda.Writer(str(HERE / "mda_models.pdb"), len(ELEMENTS),
                    multiframe=True) as writer:
        for k in range(3):
            u.atoms.positions = POSITIONS[k]
            u.dimensions = triclinic_box(*BOXES[k])
            writer.write(u.atoms)


def write_ovito():
    from ovito.io import export_file, import_file
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import Pipeline, StaticSource

    frame0 = ase_frames()[0]
    data = ase_to_ovito(frame0)
    data.particles_.create_property(
        "Particle Identifier", data=np.arange(1, len(ELEMENTS) + 1))
    pipeline = Pipeline(source=StaticSource(data=data))
    export_file(pipeline, str(HERE / "ovito.imd"), "imd",
                columns=["Particle Identifier", "Particle Type",
                         "Position.X", "Position.Y", "Position.Z"])
    data = ase_to_ovito(frame0)
    data.particles_.create_property(
        "Particle Identifier", data=np.arange(1, len(ELEMENTS) + 1))
    data.particles_.create_property(
        "Mass", data=[MASSES[e] for e in ELEMENTS])
    data.particles_.create_property("Velocity", data=VELOCITIES[0])
    pipeline = Pipeline(source=StaticSource(data=data))
    export_file(pipeline, str(HERE / "ovito_mass.imd"), "imd",
                columns=["Particle Identifier", "Particle Type", "Mass",
                         "Position.X", "Position.Y", "Position.Z",
                         "Velocity.X", "Velocity.Y", "Velocity.Z"])
    pipeline = import_file(str(HERE / "_frames.extxyz"))
    export_file(pipeline, str(HERE / "ovito_poscar.*.vasp"), "vasp",
                multiple_frames=True)


def main():
    import ase
    import MDAnalysis
    import mdtraj
    import ovito

    write_ase()
    write_mdtraj()
    write_mdanalysis()
    write_ovito()
    (HERE / "_frames.extxyz").unlink()
    (HERE / "castep_doc_example.md").write_text(CASTEP_DOC_EXAMPLE,
                                                encoding="ascii", newline="\n")
    truth = {
        "elements": ELEMENTS, "boxes_ang": BOXES.tolist(),
        "positions_ang": POSITIONS.tolist(),
        "velocities_ang_per_ps": VELOCITIES.tolist(), "times_ps": TIMES_PS,
        "masses_amu": MASSES,
    }
    (HERE / "truth.json").write_text(json.dumps(truth, indent=1) + "\n",
                                     encoding="utf-8")
    writers = {
        "ase": ase.__version__, "mdtraj": mdtraj.__version__,
        "MDAnalysis": MDAnalysis.__version__,
        "ovito": ".".join(str(x) for x in ovito.version),
        "numpy": np.__version__,
    }
    (HERE / "writers.json").write_text(json.dumps(writers, indent=1) + "\n",
                                       encoding="utf-8")
    # Windows text mode makes the writers end lines with CRLF; the files keep
    # LF, as the same writers produce on Linux, and the tests make the CRLF
    # copies themselves.
    for path in HERE.iterdir():
        if path.is_file() and path.suffix != ".py":
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


if __name__ == "__main__":
    main()
