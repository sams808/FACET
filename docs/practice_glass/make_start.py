"""A random starting configuration for the practice glass.

    20 Na2O - 10 CaO - 5 Al2O3 - 65 SiO2 (mol %), 1200 atoms:
    700 O, 260 Si, 40 Al, 160 Na, 40 Ca

Atoms are placed one by one at random in a cubic box at 2.45 g/cm3, none
closer than 1.7 angstrom to another, and written as a LAMMPS data file
(atom_style charge) with the charges of the Pedone 2006 potential.
"""
import numpy as np

SEED = 20261112
SPECIES = [  # type, element, count, mass, charge
    (1, "O", 700, 15.999, -1.2),
    (2, "Si", 260, 28.085, 2.4),
    (3, "Al", 40, 26.982, 1.8),
    (4, "Na", 160, 22.990, 0.6),
    (5, "Ca", 40, 40.078, 1.2),
]
DENSITY = 2.45                       # g/cm3, the starting guess
MIN_DISTANCE = 1.7                   # angstrom

rng = np.random.default_rng(SEED)
mass = sum(n * m for _, _, n, m, _ in SPECIES)
side = (mass / 6.02214076e23 / DENSITY) ** (1 / 3) * 1e8
types = np.concatenate([np.full(n, t) for t, _, n, _, _ in SPECIES])
rng.shuffle(types)
placed = np.zeros((0, 3))
for _ in types:
    while True:
        p = rng.random(3) * side
        d = placed - p
        d -= side * np.round(d / side)
        if not len(placed) or (np.einsum("ij,ij->i", d, d) > MIN_DISTANCE ** 2).all():
            placed = np.vstack([placed, p])
            break
charge = {t: q for t, _, _, _, q in SPECIES}
assert abs(sum(charge[t] for t in types)) < 1e-9
with open("start.data", "w") as out:
    out.write(f"practice glass 20Na2O-10CaO-5Al2O3-65SiO2, random start, seed {SEED}\n\n")
    out.write(f"{len(types)} atoms\n{len(SPECIES)} atom types\n\n")
    for axis in "xyz":
        out.write(f"0.0 {side:.6f} {axis}lo {axis}hi\n")
    out.write("\nMasses\n\n")
    for t, _, _, m, _ in SPECIES:
        out.write(f"{t} {m}\n")
    out.write("\nAtoms # charge\n\n")
    for k, (t, p) in enumerate(zip(types, placed), start=1):
        out.write(f"{k} {t} {charge[t]} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
print(f"{len(types)} atoms, box {side:.4f} A, mass {mass:.1f} g/mol")
