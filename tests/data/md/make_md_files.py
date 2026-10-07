"""Write the MD test files in this folder from one five-atom model.

Run ``py -3.11 tests/data/md/make_md_files.py`` to regenerate them. The files
are committed; tests/test_md_readers.py loads MODEL from this script and
requires every reader to return it.

THE MODEL
---------
Five atoms (ids 2, 4, 5, 7, 9: Si, O, O, Na, O; LAMMPS types 1, 2, 2, 3, 2) in
a restricted triclinic LAMMPS box with a non-zero origin, two frames, and a
box that changes between them (an NPT step). Fractional positions sit at least
0.05 from a face, so no conversion can wrap an atom to the opposite side.
Frame 1 has atoms that crossed the boundary, so the image flags are not zero.
The numbers are test inputs, not a material: Si-O and Na-O distances are
whatever these fractions give.

WHY A SCRIPT
------------
Every position is written with Python's shortest round-trip repr, so a file
holds exactly the double the model computed, and the comparison in the tests
can be 1e-10 Å. Each file is written with the forward formulas of its own
format (LAMMPS bounds, ``x = origin + xs @ box``, ``xu = x + image @ box``,
the VASP scale factor, DL_POLY's centred origin), coded here independently of
facet.core.md_readers, which inverts them. This script imports nothing from
FACET.
"""
from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

IDS = np.array([2, 4, 5, 7, 9])
ELEMENTS = ["Si", "O", "O", "Na", "O"]
TYPES = np.array([1, 2, 2, 3, 2])
MASSES = {1: 28.0855, 2: 15.9994, 3: 22.98977}
CHARGE = {"Si": 2.4, "O": -1.2, "Na": 0.6}

# LAMMPS restricted triclinic parameters of each frame [Howto triclinic].
BOX_PARAMS = [
    dict(xlo=-3.25, xhi=6.75, ylo=1.5, yhi=10.5, zlo=0.75, zhi=8.75,
         xy=2.5, xz=-1.5, yz=1.25),
    dict(xlo=-3.35, xhi=6.85, ylo=1.45, yhi=10.55, zlo=0.725, zhi=8.775,
         xy=2.4, xz=-1.4, yz=1.3),
]
FRAC = [
    np.array([[0.10, 0.20, 0.30], [0.25, 0.15, 0.45], [0.60, 0.70, 0.20],
              [0.85, 0.40, 0.90], [0.35, 0.90, 0.65]]),
    np.array([[0.12, 0.22, 0.28], [0.95, 0.10, 0.50], [0.58, 0.05, 0.22],
              [0.07, 0.42, 0.06], [0.33, 0.88, 0.95]]),
]
IMAGE = [
    np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [-1, 0, 0], [0, 2, 0], [1, 0, 1], [0, 0, -1]]),
]
# Velocities in Å/ps (LAMMPS units metal).
VELOCITY = [
    np.array([[1.5, -2.25, 0.5], [-0.75, 3.0, 1.25], [2.0, 0.25, -1.75],
              [-3.5, 1.0, 0.125], [0.625, -0.5, 2.75]]),
    np.array([[1.25, -2.0, 0.75], [-1.0, 2.5, 1.5], [1.75, 0.5, -1.5],
              [-3.0, 1.25, 0.375], [0.875, -0.25, 2.5]]),
]
TIMESTEPS = [0, 500]
TIMES_PS = [0.0, 0.5]
# The order the rows are written in: a dump is not sorted by id.
ROW_ORDER = [[3, 0, 4, 2, 1], [1, 4, 0, 3, 2]]


def box_rows(p: dict) -> tuple[np.ndarray, np.ndarray]:
    box = np.array([[p["xhi"] - p["xlo"], 0.0, 0.0],
                    [p["xy"], p["yhi"] - p["ylo"], 0.0],
                    [p["xz"], p["yz"], p["zhi"] - p["zlo"]]])
    return box, np.array([p["xlo"], p["ylo"], p["zlo"]])


BOX = [box_rows(p)[0] for p in BOX_PARAMS]
ORIGIN = [box_rows(p)[1] for p in BOX_PARAMS]


def cart(k: int) -> np.ndarray:
    return ORIGIN[k] + FRAC[k] @ BOX[k]


def unwrapped(k: int) -> np.ndarray:
    return ORIGIN[k] + (FRAC[k] + IMAGE[k]) @ BOX[k]


def _rotation() -> np.ndarray:
    """A fixed rotation (30 degrees about z, then 20 about x) that takes the
    restricted box to a general triclinic one."""
    a, b = math.radians(30.0), math.radians(20.0)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0],
                   [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)],
                   [0, math.sin(b), math.cos(b)]])
    return rx @ rz


GENERAL_BOX = [BOX[k] @ _rotation().T for k in range(2)]
GENERAL_ORIGIN = [np.array([1.0, -2.0, 0.5]), np.array([1.25, -2.0, 0.5])]


def general_cart(k: int) -> np.ndarray:
    return GENERAL_ORIGIN[k] + FRAC[k] @ GENERAL_BOX[k]


# DL_POLY puts the coordinate origin at the centre of the cell (DL_POLY 4
# manual, Appendix A).
def dlpoly_origin(k: int) -> np.ndarray:
    return -0.5 * BOX[k].sum(axis=0)


def dlpoly_cart(k: int) -> np.ndarray:
    return dlpoly_origin(k) + FRAC[k] @ BOX[k]


# The XDATCAR lists atoms by element, in the order of its species line.
VASP_ORDER = [0, 1, 2, 4, 3]           # Si, O, O, O, Na
VASP_SPECIES = ["Si", "O", "Na"]
VASP_COUNTS = [1, 3, 1]

MODEL = dict(IDS=IDS, ELEMENTS=ELEMENTS, TYPES=TYPES, MASSES=MASSES,
             CHARGE=CHARGE, BOX_PARAMS=BOX_PARAMS, BOX=BOX, ORIGIN=ORIGIN,
             FRAC=FRAC, IMAGE=IMAGE, VELOCITY=VELOCITY, TIMESTEPS=TIMESTEPS,
             TIMES_PS=TIMES_PS, GENERAL_BOX=GENERAL_BOX,
             GENERAL_ORIGIN=GENERAL_ORIGIN, VASP_ORDER=VASP_ORDER)


def f(value) -> str:
    return repr(float(value))


def row(values) -> str:
    return " ".join(f(v) for v in values)


# ---------------------------------------------------------------------------
# LAMMPS dumps
# ---------------------------------------------------------------------------

def dump_bounds(p: dict) -> list[str]:
    """The three BOX BOUNDS lines LAMMPS writes for a restricted triclinic box
    (xlo_bound xhi_bound xy / ylo_bound yhi_bound xz / zlo_bound zhi_bound yz)."""
    xy, xz, yz = p["xy"], p["xz"], p["yz"]
    xlo_b = p["xlo"] + min(0.0, xy, xz, xy + xz)
    xhi_b = p["xhi"] + max(0.0, xy, xz, xy + xz)
    ylo_b = p["ylo"] + min(0.0, yz)
    yhi_b = p["yhi"] + max(0.0, yz)
    return [f"{f(xlo_b)} {f(xhi_b)} {f(xy)}", f"{f(ylo_b)} {f(yhi_b)} {f(xz)}",
            f"{f(p['zlo'])} {f(p['zhi'])} {f(yz)}"]


def dump_frame(k: int, columns: str, values, *, box_lines=None,
               box_kind="xy xz yz pp pp pp", units=None, time=None,
               n_rows=None) -> str:
    """One frame. ``values(k, i)`` gives the row values after id and type."""
    lines = []
    if units:
        lines += ["ITEM: UNITS", units]
    if time is not None:
        lines += ["ITEM: TIME", f(time)]
    lines += ["ITEM: TIMESTEP", str(TIMESTEPS[k]), "ITEM: NUMBER OF ATOMS",
              str(len(IDS)), f"ITEM: BOX BOUNDS {box_kind}"]
    lines += box_lines if box_lines is not None else dump_bounds(BOX_PARAMS[k])
    lines.append(f"ITEM: ATOMS {columns}")
    order = ROW_ORDER[k] if n_rows is None else ROW_ORDER[k][:n_rows]
    for i in order:
        lines.append(" ".join(str(v) for v in values(k, i)))
    return "\n".join(lines) + "\n"


def write_dumps() -> None:
    def ident(k, i):
        return [int(IDS[i]), int(TYPES[i])]

    variants = {
        "dump_tri_x.lammpstrj": ("id type x y z",
                                 lambda k, i: ident(k, i) + [f(v) for v in cart(k)[i]]),
        "dump_tri_xs.lammpstrj": ("id type xs ys zs",
                                  lambda k, i: ident(k, i) + [f(v) for v in FRAC[k][i]]),
        "dump_tri_xu.lammpstrj": ("id type xu yu zu",
                                  lambda k, i: ident(k, i) + [f(v) for v in unwrapped(k)[i]]),
        "dump_tri_xsu.lammpstrj": ("id type xsu ysu zsu",
                                   lambda k, i: ident(k, i) + [
                                       f(v) for v in (FRAC[k] + IMAGE[k])[i]]),
        "dump_tri_x_image.lammpstrj": ("id type x y z ix iy iz",
                                       lambda k, i: ident(k, i)
                                       + [f(v) for v in cart(k)[i]]
                                       + [int(v) for v in IMAGE[k][i]]),
    }
    for name, (columns, values) in variants.items():
        text = "".join(dump_frame(k, columns, values) for k in range(2))
        (HERE / name).write_bytes(text.encode("ascii"))
        if name == "dump_tri_x.lammpstrj":
            with open(HERE / (name + ".gz"), "wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                   mtime=0) as packed:
                    packed.write(text.encode("ascii"))

    # Orthogonal box, units real (times in fs, velocities in Å/fs), element
    # column, unwrapped positions, velocities and charges.
    def ortho_box(k):
        p = BOX_PARAMS[k]
        return [f"{f(p['xlo'])} {f(p['xhi'])}", f"{f(p['ylo'])} {f(p['yhi'])}",
                f"{f(p['zlo'])} {f(p['zhi'])}"]

    def ortho_values(k, i):
        box = np.diag(np.diag(BOX[k]))
        position = ORIGIN[k] + (FRAC[k][i] + IMAGE[k][i]) @ box
        return ([int(IDS[i]), ELEMENTS[i]] + [f(v) for v in position]
                + [f(v / 1000.0) for v in VELOCITY[k][i]] + [f(CHARGE[ELEMENTS[i]])])

    text = "".join(dump_frame(k, "id element xu yu zu vx vy vz q", ortho_values,
                              box_lines=ortho_box(k), box_kind="pp pp pp",
                              units="real" if k == 0 else None,
                              time=TIMES_PS[k] * 1000.0) for k in range(2))
    (HERE / "dump_ortho_real.lammpstrj").write_bytes(text.encode("ascii"))

    # General triclinic box ('abc origin'), typelabel column.
    def general_box(k):
        return [f"{row(GENERAL_BOX[k][r])} {f(GENERAL_ORIGIN[k][r])}"
                for r in range(3)]

    text = "".join(dump_frame(
        k, "id typelabel x y z",
        lambda k, i: [int(IDS[i]), ELEMENTS[i]] + [f(v) for v in general_cart(k)[i]],
        box_lines=general_box(k), box_kind="abc origin pp pp pp")
        for k in range(2))
    (HERE / "dump_general.lammpstrj").write_bytes(text.encode("ascii"))

    # Two complete frames, then a third cut off after two of its five rows.
    # ITEM: UNITS is written once, in the first frame, as LAMMPS does.
    values = variants["dump_tri_x.lammpstrj"][1]
    text = (dump_frame(0, "id type x y z", values, units="metal")
            + dump_frame(1, "id type x y z", values)
            + dump_frame(1, "id type x y z", values, n_rows=2).replace(
                "ITEM: TIMESTEP\n500\n", "ITEM: TIMESTEP\n1000\n"))
    (HERE / "dump_truncated.lammpstrj").write_bytes(text.encode("ascii"))


# ---------------------------------------------------------------------------
# LAMMPS data files (frame 0)
# ---------------------------------------------------------------------------

def data_header(p: dict, title: str, tilt: bool = True) -> list[str]:
    lines = [title, "", f"{len(IDS)} atoms", "3 atom types", "",
             f"{f(p['xlo'])} {f(p['xhi'])} xlo xhi",
             f"{f(p['ylo'])} {f(p['yhi'])} ylo yhi",
             f"{f(p['zlo'])} {f(p['zhi'])} zlo zhi"]
    if tilt:
        lines.append(f"{f(p['xy'])} {f(p['xz'])} {f(p['yz'])} xy xz yz")
    return lines


def masses_section(comments: bool) -> list[str]:
    names = {1: "Si", 2: "O", 3: "Na"}
    return ["", "Masses", ""] + [
        f"{t} {m!r}" + (f"  # {names[t]}" if comments else "")
        for t, m in MASSES.items()]


def write_data_files() -> None:
    p = BOX_PARAMS[0]
    lines = data_header(p, "five-atom test model, atomic style")
    lines += masses_section(comments=False)
    lines += ["", "Atoms # atomic", ""]
    for i in ROW_ORDER[0]:
        lines.append(f"{IDS[i]} {TYPES[i]} {row(cart(0)[i])}")
    (HERE / "data_atomic.data").write_bytes(("\n".join(lines) + "\n").encode())

    # write_data's first line states the timestep and the unit style.
    lines = data_header(p, "LAMMPS data file via write_data, version 29 Aug "
                           "2024, timestep = 1000, units = metal")
    lines += masses_section(comments=True)
    lines += ["", "Atoms # charge", ""]
    for i in ROW_ORDER[0]:
        lines.append(f"{IDS[i]} {TYPES[i]} {f(CHARGE[ELEMENTS[i]])} "
                     f"{row(cart(0)[i])} {' '.join(str(v) for v in IMAGE[0][i])}")
    lines += ["", "Velocities", ""]
    for i in ROW_ORDER[1]:
        lines.append(f"{IDS[i]} {row(VELOCITY[0][i])}")
    (HERE / "data_charge.data").write_bytes(("\n".join(lines) + "\n").encode())

    # General triclinic header, Atom Type Labels, labels in the type column.
    lines = ["five-atom test model, general triclinic", "", f"{len(IDS)} atoms",
             "3 atom types", "",
             f"{row(GENERAL_BOX[0][0])} avec", f"{row(GENERAL_BOX[0][1])} bvec",
             f"{row(GENERAL_BOX[0][2])} cvec",
             f"{row(GENERAL_ORIGIN[0])} abc origin",
             "", "Atom Type Labels", "", "1 Si", "2 O", "3 Na",
             "", "Atoms # atomic", ""]
    for i in ROW_ORDER[0]:
        kind = ELEMENTS[i] if i % 2 else str(TYPES[i])
        lines.append(f"{IDS[i]} {kind} {row(general_cart(0)[i])}")
    (HERE / "data_general.data").write_bytes(("\n".join(lines) + "\n").encode())


# ---------------------------------------------------------------------------
# extended XYZ
# ---------------------------------------------------------------------------

def write_extxyz() -> None:
    lines = []
    for k in range(2):
        lattice = " ".join(f(v) for v in BOX[k].reshape(-1))
        origin = " ".join(f(v) for v in ORIGIN[k])
        lines += [str(len(IDS)),
                  f'Lattice="{lattice}" Origin="{origin}" '
                  "Properties=species:S:1:pos:R:3:velo:R:3:id:I:1:charge:R:1 "
                  f'Time={f(TIMES_PS[k])} energy=-12.5 pbc="T T T"']
        for i in ROW_ORDER[k]:
            lines.append(f"{ELEMENTS[i]} {row(cart(k)[i])} {row(VELOCITY[k][i])} "
                         f"{IDS[i]} {f(CHARGE[ELEMENTS[i]])}")
    (HERE / "glass.extxyz").write_bytes(("\n".join(lines) + "\n").encode())

    # The same frames with the columns in another order (OVITO writes them in
    # the order the user picks): id first, then the species, so the third line
    # does not read 'symbol x y z'.
    lines = []
    for k in range(2):
        lattice = " ".join(f(v) for v in BOX[k].reshape(-1))
        origin = " ".join(f(v) for v in ORIGIN[k])
        lines += [str(len(IDS)),
                  f'Lattice="{lattice}" Origin="{origin}" '
                  "Properties=id:I:1:species:S:1:pos:R:3:velo:R:3 "
                  f'Time={f(TIMES_PS[k])} pbc="T T T"']
        for i in ROW_ORDER[k]:
            lines.append(f"{IDS[i]} {ELEMENTS[i]} {row(cart(k)[i])} "
                         f"{row(VELOCITY[k][i])}")
    (HERE / "glass_reordered.extxyz").write_bytes(
        ("\n".join(lines) + "\n").encode())


# ---------------------------------------------------------------------------
# VASP XDATCAR
# ---------------------------------------------------------------------------

def vasp_header(title: str, scale: str, lattice: np.ndarray,
                species: list[str]) -> list[str]:
    return [title, scale] + [row(v) for v in lattice] + [
        " ".join(species), " ".join(str(c) for c in VASP_COUNTS)]


def write_xdatcar() -> None:
    lines = vasp_header("five-atom test model", "1.0", BOX[0], VASP_SPECIES)
    for k in range(2):
        lines.append(f"Direct configuration=     {k + 1}")
        lines += [row(FRAC[k][i]) for i in VASP_ORDER]
    (HERE / "XDATCAR_fixed").write_bytes(("\n".join(lines) + "\n").encode())

    # Variable cell: the header before every configuration. Frame 0 uses a
    # scale of 2 on halved vectors; frame 1 a negative scale, which VASP reads
    # as the cell volume, on vectors shrunk by 0.9.
    volume1 = abs(float(np.linalg.det(BOX[1])))
    lines = vasp_header("five-atom test model", "2.0", BOX[0] / 2.0,
                        ["Si", "O", "Na_pv"])
    lines.append("Direct configuration=     1")
    lines += [row(FRAC[0][i]) for i in VASP_ORDER]
    lines += vasp_header("five-atom test model", f(-volume1), BOX[1] * 0.9,
                         ["Si", "O", "Na_pv"])
    lines.append("Direct configuration=     2")
    lines += [row(FRAC[1][i]) for i in VASP_ORDER]
    (HERE / "XDATCAR_variable").write_bytes(("\n".join(lines) + "\n").encode())


# ---------------------------------------------------------------------------
# DL_POLY
# ---------------------------------------------------------------------------

def write_dlpoly() -> None:
    lines = ["five-atom test model CONFIG", "1 3 5"]
    lines += [row(v) for v in BOX[0]]
    for i in ROW_ORDER[0]:
        lines += [f"{ELEMENTS[i]} {IDS[i]}", row(dlpoly_cart(0)[i]),
                  row(VELOCITY[0][i])]
    (HERE / "CONFIG").write_bytes(("\n".join(lines) + "\n").encode())

    # HISTORY with force-field names: O_b and O_nb are not element symbols,
    # so a type map is needed for them.
    names = {1: "O_b", 2: "O_nb", 4: "O_b"}
    lines = ["five-atom test model HISTORY", "1 3 5 2 42"]
    for k in range(2):
        lines.append(f"timestep {TIMESTEPS[k]} 5 1 3 0.001 {f(TIMES_PS[k])}")
        lines += [row(v) for v in BOX[k]]
        for i in ROW_ORDER[k]:
            lines += [f"{names.get(i, ELEMENTS[i])} {IDS[i]} "
                      f"{MASSES[int(TYPES[i])]!r} {f(CHARGE[ELEMENTS[i]])} 0.0",
                      row(dlpoly_cart(k)[i]), row(VELOCITY[k][i])]
    (HERE / "HISTORY").write_bytes(("\n".join(lines) + "\n").encode())


if __name__ == "__main__":
    write_dumps()
    write_data_files()
    write_extxyz()
    write_xdatcar()
    write_dlpoly()
    print("written:", sorted(p.name for p in HERE.iterdir()
                             if p.name != Path(__file__).name))
