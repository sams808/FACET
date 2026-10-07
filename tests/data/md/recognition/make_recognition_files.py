"""Write the recognition fixtures with the programs that write such files.

Run with a Python that has the LAMMPS and ASE wheels (the files here were
written by LAMMPS 22 Jul 2025 update 4, pip wheel 2025.7.22.4.0, and ASE
3.29.0, on Python 3.11.9); this script imports nothing from FACET, so the
tests check FACET against what the writers were given or held. Every number
is a test input, not a property of any material.

Each file pins one recognition defect of the recognition test (corpus of 115
files from real programs, 2026-10-07) or one new input path:

* XDATCAR_long          ASE vasp-xdatcar of 130 atoms not grouped by element:
                        lines 6-7 are 2210 characters, line 8 lies past
                        byte 4096 (D1)
* three_models.pdb      ASE proteindatabank, 3 MODEL blocks (D2)
* cp2k_style.pdb        REMARK / CRYST1 / ATOM / END per frame, no MODEL,
                        the layout CP2K's PDB trajectory has (C2; synthetic)
* xdatcar_saved.vasp    ASE vasp-xdatcar, 3 configurations, saved as .vasp (D3)
* dump_saved_as.xyz     LAMMPS dump custom id type element x y z (D4)
* glass.config          ASE dlp4 CONFIG under a name DL_POLY does not use (D5)
* labels_typelabel.lammpstrj, labels_element.lammpstrj
                        LAMMPS, labelmap Si_t Na+ O_b O_nb, a numeric type
                        column beside a typelabel / element column (D6)
* atom_default.lammpstrj
                        LAMMPS dump atom: types only (D7)
* ase_unwrapped.extxyz  ASE MD (VelocityVerlet, a negligible LJ potential),
                        positions as ASE wrote them, never wrapped (b)
* lammps_wrapped.extxyz LAMMPS dump extxyz of the same kind of run: wrapped
                        positions, origin not written (b, the other side)
* nvt.xyz, nvt_types.xyz, nvt.data
                        LAMMPS dump xyz (dump_modify element, and the default
                        type numbers) of an NVE run in a fixed box, and the
                        data file write_data writes at its end (box_from)
* NS2-pos-1.xyz, NS2-1.cell, NS2-1_npt.cell
                        CP2K's XMOL trajectory and cell files, written here
                        with CP2K's format strings (motion_utils.F): synthetic,
                        as no CP2K is installed
* series/dump.0.lammpstrj, dump.50.lammpstrj, dump.100.lammpstrj
                        LAMMPS dump custom with '*' in the file name, one file
                        per snapshot: text order puts 100 before 50

recognition_truth.json holds what the writers were given or held (positions
in Å, boxes as rows, origins, timesteps, elements by atom id), and the
versions.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def write_truth(truth: dict) -> None:
    (HERE / "recognition_truth.json").write_text(
        json.dumps(truth, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")


def rows(array) -> list:
    """Values to 1e-10, below the last digit any of these writers keeps
    (ASE 8 decimals, LAMMPS %g, CP2K 10 decimals), so the truth file stays
    small."""
    return np.round(np.asarray(array, dtype=float), 10).tolist()


# ---------------------------------------------------------------------------
# LAMMPS
# ---------------------------------------------------------------------------

LO = np.array([-1.5, 0.75, -2.25])
LX, LY, LZ, XY, XZ, YZ = 9.0, 8.5, 9.5, 0.8, -0.6, 0.4
BOX = np.array([[LX, 0, 0], [XY, LY, 0], [XZ, YZ, LZ]])
TYPES = [1] * 4 + [2] * 4 + [3] * 10           # Si, Na, O
SYMBOLS = {1: "Si", 2: "Na", 3: "O"}
MASSES = {1: 28.0855, 2: 22.98977, 3: 15.9994}


def model_frac(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.05, 0.95, (n, 3))


def lammps_session(log: str):
    from lammps import lammps

    return lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"])


def lammps_model(lmp, types=TYPES, ntypes=3, masses=MASSES) -> None:
    hi = LO + np.array([LX, LY, LZ])
    lmp.commands_string(f"""
units metal
atom_style atomic
boundary p p p
region box prism {LO[0]} {hi[0]} {LO[1]} {hi[1]} {LO[2]} {hi[2]} {XY} {XZ} {YZ}
create_box {ntypes} box
""")
    frac = model_frac(len(types), 11)
    cart = LO + frac @ BOX
    lmp.create_atoms(len(types), list(range(1, len(types) + 1)), types,
                     [float(v) for v in cart.ravel()])
    for t, m in masses.items():
        lmp.command(f"mass {t} {m}")
    lmp.commands_string("""
pair_style zero 3.0
pair_coeff * *
velocity all create 6000.0 4928459 dist gaussian mom yes
fix 1 all nve
timestep 0.001
""")


def lammps_state(lmp) -> dict:
    n = lmp.get_natoms()
    ids = np.ctypeslib.as_array(lmp.extract_atom("id"), shape=(n,)).copy()
    x = np.ctypeslib.as_array(lmp.extract_atom("x")[0], shape=(n, 3)).copy()
    order = np.argsort(ids)
    return {"ids": ids[order].tolist(), "x": rows(x[order]),
            "step": int(lmp.get_thermo("step"))}


def write_lammps() -> dict:
    truth: dict = {}
    import lammps as lammps_module

    # one run: dump custom saved as .xyz (D4), dump atom (D7), dump xyz with
    # and without elements (box_from), dump extxyz (b), a dump series, and
    # the data file at the end
    lmp = lammps_session("run")
    lammps_model(lmp)
    series = HERE / "series"
    if series.exists():
        shutil.rmtree(series)
    series.mkdir()
    lmp.commands_string(f"""
dump d1 all custom 50 {HERE / 'dump_saved_as.xyz'} id type element x y z
dump_modify d1 element Si Na O sort id
dump d2 all atom 50 {HERE / 'atom_default.lammpstrj'}
dump d3 all xyz 50 {HERE / 'nvt.xyz'}
dump_modify d3 element Si Na O sort id
dump d4 all xyz 50 {HERE / 'nvt_types.xyz'}
dump_modify d4 sort id
dump d5 all extxyz 50 {HERE / 'lammps_wrapped.extxyz'}
dump_modify d5 element Si Na O sort id
dump d6 all custom 50 {series / 'dump.*.lammpstrj'} id type x y z ix iy iz
dump_modify d6 sort id
""")
    states = []
    for _ in range(3):
        lmp.command("run 0 post no" if not states else "run 50 post no")
        states.append(lammps_state(lmp))
    lmp.command(f"write_data {HERE / 'nvt.data'}")
    lmp.close()
    box = rows(BOX)
    truth["lammps_run"] = {
        "ids": states[0]["ids"], "types": TYPES,
        "elements": [SYMBOLS[t] for t in TYPES],
        "timesteps": [s["step"] for s in states],
        "positions": [s["x"] for s in states], "box": box,
        "origin": LO.tolist(),
        "files": ["dump_saved_as.xyz", "atom_default.lammpstrj", "nvt.xyz",
                  "nvt_types.xyz", "lammps_wrapped.extxyz", "nvt.data",
                  "series/dump.*.lammpstrj"]}

    # labels that are not element symbols, a numeric type column beside them
    lmp = lammps_session("labels")
    types4 = [1] * 4 + [2] * 4 + [3] * 6 + [4] * 4
    lammps_model(lmp, types=types4, ntypes=4,
                 masses={1: 28.0855, 2: 22.98977, 3: 15.9994, 4: 15.9994})
    lmp.commands_string(f"""
labelmap atom 1 Si_t 2 Na+ 3 O_b 4 O_nb
dump l1 all custom 50 {HERE / 'labels_typelabel.lammpstrj'} id type typelabel x y z
dump_modify l1 sort id
dump l2 all custom 50 {HERE / 'labels_element.lammpstrj'} id type element x y z
dump_modify l2 element Si_t Na+ O_b O_nb sort id
run 50 post no
""")
    lmp.close()
    truth["labels"] = {"types": types4,
                       "labels": {"1": "Si_t", "2": "Na+", "3": "O_b",
                                  "4": "O_nb"},
                       "elements": [{1: "Si", 2: "Na", 3: "O", 4: "O"}[t]
                                    for t in types4]}
    truth["lammps_version"] = str(lammps_module.__version__)
    return truth


# ---------------------------------------------------------------------------
# ASE
# ---------------------------------------------------------------------------

def write_ase() -> dict:
    import ase
    from ase import Atoms, units
    from ase.calculators.lj import LennardJones
    from ase.io import write
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    truth: dict = {"ase_version": ase.__version__}
    cell = BOX.copy()

    # D1: 130 atoms, Si and O alternating, so ASE writes 130 symbol groups
    n = 130
    symbols = ["Si" if i % 2 == 0 else "O" for i in range(n)]
    frames, fracs = [], []
    for k in range(2):
        frac = model_frac(n, 21 + k)
        frames.append(Atoms(symbols, scaled_positions=frac, cell=cell, pbc=True))
        fracs.append(rows(frac))
    write(HERE / "XDATCAR_long", frames, format="vasp-xdatcar")
    truth["XDATCAR_long"] = {"symbols": symbols, "frac": fracs, "box": rows(cell)}

    # D3: a grouped XDATCAR, three configurations, saved as .vasp
    small = ["Si"] * 4 + ["Na"] * 4 + ["O"] * 10
    frames, fracs = [], []
    for k in range(3):
        frac = model_frac(len(small), 31 + k)
        frames.append(Atoms(small, scaled_positions=frac, cell=cell, pbc=True))
        fracs.append(rows(frac))
    write(HERE / "xdatcar_saved.vasp", frames, format="vasp-xdatcar")
    truth["xdatcar_saved.vasp"] = {"symbols": small, "frac": fracs,
                                   "box": rows(cell)}

    # D2: three MODEL blocks
    write(HERE / "three_models.pdb", frames, format="proteindatabank")
    truth["three_models.pdb"] = {
        "symbols": small, "positions": [rows(a.positions) for a in frames]}

    # D5: a CONFIG under a name DL_POLY does not give its files
    with open(HERE / "glass.config", "w") as handle:
        write(handle, frames[0], format="dlp4")
    truth["glass.config"] = {"symbols": small,
                             "positions": rows(frames[0].positions)}

    # (b): ASE MD never wraps positions. Free flight (a negligible LJ
    # potential) at high temperature takes atoms several box widths out.
    atoms = Atoms(small, scaled_positions=model_frac(len(small), 41), cell=cell,
                  pbc=True)
    atoms.calc = LennardJones(epsilon=1e-8, sigma=1.0, rc=2.5)
    MaxwellBoltzmannDistribution(atoms, temperature_K=30000,
                                 rng=np.random.default_rng(5))
    md = VelocityVerlet(atoms, timestep=5 * units.fs)
    md_frames, positions = [], []
    for k in range(3):
        if k:
            md.run(100)
        md_frames.append(atoms.copy())
        positions.append(rows(atoms.positions))
    write(HERE / "ase_unwrapped.extxyz", md_frames, format="extxyz")
    truth["ase_unwrapped.extxyz"] = {"symbols": small, "positions": positions,
                                     "box": rows(cell)}
    return truth


# ---------------------------------------------------------------------------
# CP2K's layouts, written with its format strings (synthetic)
# ---------------------------------------------------------------------------

def write_cp2k(truth: dict) -> None:
    run = truth["lammps_run"]
    symbols = run["elements"]
    box = np.array(run["box"])
    lines = []
    times_fs = [0.0, 25.0, 50.0]
    for k, positions in enumerate(run["positions"]):
        lines.append(f"{len(symbols):8d}")
        # FMT="(A,I8,A,F12.3,A,F20.10)": " i = ", it, ", time = ", t, ", E = ", e
        lines.append(f" i = {50 * k:8d}, time = {times_fs[k]:12.3f}, E = "
                     f"{-1234.5678901234 - k:20.10f}")
        lines += [f"{s:>4s} {x:20.10f} {y:20.10f} {z:20.10f}"
                  for s, (x, y, z) in zip(symbols, positions)]
    (HERE / "NS2-pos-1.xyz").write_text("\n".join(lines) + "\n")
    header = ("#   Step   Time [fs]" + "".join(
        f"       {c} [Angstrom]" for c in ("Ax", "Ay", "Az", "Bx", "By", "Bz",
                                           "Cx", "Cy", "Cz"))
        + "      Volume [Angstrom^3]")

    def cell_rows(scale):
        out = [header]
        for k in range(3):
            b = box * scale(k)
            # FMT="(I8,F12.3,9(1X,F19.10),1X,F24.10)"
            out.append(f"{50 * k:8d}{times_fs[k]:12.3f}"
                       + "".join(f" {v:19.10f}" for v in b.ravel())
                       + f" {abs(np.linalg.det(b)):24.10f}")
        return "\n".join(out) + "\n"

    (HERE / "NS2-1.cell").write_text(cell_rows(lambda k: 1.0))
    (HERE / "NS2-1_npt.cell").write_text(cell_rows(lambda k: 1.0 + 0.01 * k))
    # C2: one REMARK / CRYST1 / ATOM ... END block per frame, no MODEL
    a, b, c = np.linalg.norm(box, axis=1)
    alpha = np.degrees(np.arccos(box[1] @ box[2] / (b * c)))
    beta = np.degrees(np.arccos(box[0] @ box[2] / (a * c)))
    gamma = np.degrees(np.arccos(box[0] @ box[1] / (a * b)))
    pdb = []
    for k, positions in enumerate(run["positions"]):
        pdb.append(f"REMARK Step {50 * k}, E = {-1234.5678901234 - k:.10f}")
        pdb.append(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}"
                   f"{gamma:7.2f} P 1           1")
        for i, (s, (x, y, z)) in enumerate(zip(symbols, positions), start=1):
            pdb.append(f"ATOM  {i:5d} {s:<4s} MOL     1    {x:8.3f}{y:8.3f}"
                       f"{z:8.3f}  0.00  0.00          {s:>2s}")
        pdb.append("END")
    (HERE / "cp2k_style.pdb").write_text("\n".join(pdb) + "\n")
    truth["cp2k"] = {"times_fs": times_fs, "steps": [0, 50, 100],
                     "note": "synthetic: CP2K's format strings, not CP2K"}


def main() -> None:
    truth = write_lammps()
    truth.update(write_ase())
    write_cp2k(truth)
    write_truth(truth)


if __name__ == "__main__":
    main()
