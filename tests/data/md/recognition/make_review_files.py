"""Write the fixtures of the core readers' review (2026-10-07) with the
programs that write such files.

Run with a Python that has the LAMMPS and ASE wheels (the files in review/
were written by LAMMPS 22 Jul 2025 update 4, pip wheel 2025.7.22.4.0, and ASE
3.29.0, on Python 3.11.9); this script imports nothing from FACET, so the
tests check FACET against what the writers were given or held. Every number
is a test input, not a property of any material.

* general.data, restricted.data, general_run.xyz, general_dump.lammpstrj
                        LAMMPS read a general triclinic data file (avec, bvec,
                        cvec, abc origin) and wrote it back with 'write_data
                        ... triclinic/general' and with plain 'write_data'
                        (restricted: a along x, b in the xy plane), a 'dump
                        xyz' of three steps (LAMMPS writes its restricted
                        coordinates there) and one 'dump custom' frame with
                        'dump_modify triclinic/general yes' (BOX BOUNDS abc
                        origin, positions in the general frame)
* XDATCAR_long_title    ASE vasp-xdatcar with label= the symbol runs of a
                        shuffled 2970-atom Na2Si2O5 model (what ASE writes as
                        the title of such a model): line 1 is longer than
                        4096 bytes
* long_comment.extxyz   ASE extxyz, 2 frames, an info value of 4200
                        characters, so the comment line passes byte 4096
* three_images.cif      ASE cif of 3 images of one 18-atom model (ASE writes
                        data_image0, data_image1, data_image2)
* NS2-pos-1.extxyz      CP2K's EXTXYZ trajectory layout, written here with
                        CP2K's format strings (src/motion_utils.F,
                        write_trajectory, dump_extxyz): synthetic, as no CP2K
                        is installed; steps 0, 20, 40 at 0.5 fs, so Time=
                        0.000, 10.000, 20.000 fs

review_truth.json holds what the writers were given or held (LAMMPS's
restricted box and fractional coordinates by atom id, ASE's positions and
cells, the CP2K steps and times) and the versions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "review"


def rows(array) -> list:
    """Values to 1e-10, below the last digit any of these writers keeps."""
    return np.round(np.asarray(array, dtype=float), 10).tolist()


# A general triclinic cell: a, b, c rotated away from LAMMPS's restricted
# orientation, right-handed (det > 0), with an origin away from 0.
GENERAL = np.array([[8.0, 2.0, 1.0], [-1.0, 8.5, 1.5], [1.2, -0.8, 9.0]])
GENERAL_ORIGIN = np.array([1.0, -2.0, 0.5])
TYPES = [1] * 4 + [2] * 4 + [3] * 10           # Si, Na, O
MASSES = {1: 28.0855, 2: 22.98977, 3: 15.9994}


def write_lammps_general() -> dict:
    import lammps as lammps_module
    from lammps import lammps

    assert np.linalg.det(GENERAL) > 0
    frac = np.random.default_rng(7).uniform(0.05, 0.95, (len(TYPES), 3))
    cart = GENERAL_ORIGIN + frac @ GENERAL
    source = OUT / "general_input.data"
    lines = ["general triclinic input for make_review_files.py", "",
             f"{len(TYPES)} atoms", "3 atom types", ""]
    for vector, key in zip(GENERAL, ("avec", "bvec", "cvec")):
        lines.append(" ".join(f"{v:.10f}" for v in vector) + f" {key}")
    lines.append(" ".join(f"{v:.10f}" for v in GENERAL_ORIGIN) + " abc origin")
    lines += ["", "Masses", ""] + [f"{t} {m}" for t, m in MASSES.items()]
    lines += ["", "Atoms # atomic", ""]
    lines += [f"{i} {t} {x:.10f} {y:.10f} {z:.10f}"
              for i, (t, (x, y, z)) in enumerate(zip(TYPES, cart), start=1)]
    source.write_text("\n".join(lines) + "\n")

    lmp = lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"])
    lmp.commands_string(f"""
units metal
atom_style atomic
boundary p p p
read_data {source.as_posix()}
pair_style zero 3.0
pair_coeff * *
velocity all create 3000.0 87287 dist gaussian mom yes
fix 1 all nve
timestep 0.001
write_data {(OUT / 'general.data').as_posix()} triclinic/general
write_data {(OUT / 'restricted.data').as_posix()}
dump x1 all xyz 50 {(OUT / 'general_run.xyz').as_posix()}
dump_modify x1 element Si Na O sort id
dump g1 all custom 1000 {(OUT / 'general_dump.lammpstrj').as_posix()} id type x y z
dump_modify g1 triclinic/general yes sort id
""")
    source.unlink()
    states = []
    for k in range(3):
        lmp.command("run 0 post no" if k == 0 else "run 50 post no")
        n = lmp.get_natoms()
        ids = np.ctypeslib.as_array(lmp.extract_atom("id"), shape=(n,)).copy()
        x = np.ctypeslib.as_array(lmp.extract_atom("x")[0],
                                  shape=(n, 3)).copy()
        lo, hi, xy, yz, xz, _, _ = lmp.extract_box()
        box = np.array([[hi[0] - lo[0], 0, 0], [xy, hi[1] - lo[1], 0],
                        [xz, yz, hi[2] - lo[2]]])
        order = np.argsort(ids)
        states.append({"step": int(lmp.get_thermo("step")),
                       "frac": rows((x[order] - np.array(lo))
                                    @ np.linalg.inv(box)),
                       "box": rows(box), "origin": rows(lo)})
    lmp.close()
    return {"lammps_version": str(lammps_module.__version__),
            "general": {"avec_bvec_cvec": rows(GENERAL),
                        "abc_origin": rows(GENERAL_ORIGIN),
                        "elements": [{1: "Si", 2: "Na", 3: "O"}[t]
                                     for t in TYPES],
                        "states": states}}


def write_ase() -> dict:
    import ase
    from ase import Atoms
    from ase.io import write

    truth: dict = {"ase_version": ase.__version__}
    small = ["Si"] * 4 + ["Na"] * 4 + ["O"] * 10
    cell = np.array([[9.0, 0, 0], [0.8, 8.5, 0], [-0.6, 0.4, 9.5]])

    # an XDATCAR whose title, line 1, is longer than the 4096-byte window
    big = np.array(["Na"] * 2 * 330 + ["Si"] * 2 * 330 + ["O"] * 5 * 330)
    np.random.default_rng(3).shuffle(big)
    runs = [big[0]] + [s for s, before in zip(big[1:], big[:-1]) if s != before]
    label = " ".join(runs)
    assert len(label) > 4096
    frames, fracs = [], []
    for k in range(2):
        frac = np.random.default_rng(50 + k).uniform(0.05, 0.95, (18, 3))
        frames.append(Atoms(small, scaled_positions=frac, cell=cell, pbc=True))
        fracs.append(rows(frac))
    write(OUT / "XDATCAR_long_title", frames, format="vasp-xdatcar",
          label=label)
    truth["XDATCAR_long_title"] = {"symbols": small, "frac": fracs,
                                   "title_bytes": len(label)}

    # an extended XYZ whose comment line passes the 4096-byte window
    images = []
    for k in range(2):
        atoms = Atoms(["Si", "O", "O"], scaled_positions=np.random.default_rng(
            60 + k).uniform(0.05, 0.95, (3, 3)), cell=cell, pbc=True)
        atoms.info["calc_params"] = "x" * 4200
        images.append(atoms)
    write(OUT / "long_comment.extxyz", images, format="extxyz")
    truth["long_comment.extxyz"] = {"symbols": ["Si", "O", "O"],
                                    "positions": [rows(a.positions)
                                                  for a in images]}

    # a trajectory saved as CIF: one data block per image
    images = [Atoms(small, scaled_positions=np.array(fracs[k % 2]), cell=cell,
                    pbc=True) for k in range(3)]
    write(OUT / "three_images.cif", images, format="cif")
    truth["three_images.cif"] = {"n_images": 3, "n_atoms": 18}
    return truth


def write_cp2k_extxyz() -> dict:
    """CP2K's dump_extxyz title (motion_utils.F, write_trajectory):
    'Lattice="'//cell_str//'" Properties=species:S:1:pos:R:3 pbc="T T T"
    Step=<I8> Time=<F12.3> Energy=<F20.10>', each number TRIM(ADJUSTL)ed,
    cell_str FMT="(9(1X,F19.10))" of hmat columns (the vectors) in Å; the
    time is in fs (md_energies.F passes time*femtoseconds)."""
    cell = np.array([[9.0, 0, 0], [0.8, 8.5, 0], [-0.6, 0.4, 9.5]])
    symbols = ["Si"] * 4 + ["Na"] * 4 + ["O"] * 10
    steps, times_fs, positions = [0, 20, 40], [0.0, 10.0, 20.0], []
    lines = []
    for k, (step, time) in enumerate(zip(steps, times_fs)):
        frac = np.random.default_rng(70 + k).uniform(0.05, 0.95, (18, 3))
        cart = frac @ cell
        positions.append(rows(cart))
        # TRIM(ADJUSTL(...)) of each field is Python's strip()
        cell_str = "".join(f" {v:19.10f}" for v in cell.ravel()).strip()
        step_str = f"{step:8d}".strip()
        time_str = f"{time:12.3f}".strip()
        etot_str = f"{-1234.5678901234 - k:20.10f}".strip()
        lines.append(f"{len(symbols):8d}")
        lines.append(f'Lattice="{cell_str}" Properties=species:S:1:pos:R:3 '
                     f'pbc="T T T" Step={step_str} Time={time_str} '
                     f'Energy={etot_str}')
        lines += [f"{s:>4s} {x:20.10f} {y:20.10f} {z:20.10f}"
                  for s, (x, y, z) in zip(symbols, cart)]
    text = "\n".join(lines) + "\n"
    (OUT / "NS2-pos-1.extxyz").write_text(text)
    return {"NS2-pos-1.extxyz": {"steps": steps, "times_fs": times_fs,
                                 "positions": positions, "box": rows(cell),
                                 "note": "synthetic: CP2K's format strings, "
                                         "not CP2K"}}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    truth = write_lammps_general()
    truth.update(write_ase())
    truth.update(write_cp2k_extxyz())
    (OUT / "review_truth.json").write_text(
        json.dumps(truth, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
