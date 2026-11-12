# The practice glass

A small model of a soda-lime aluminosilicate glass, made so that the examples
of the library manual (`docs/library_manual.md`) and the library's tests run
on a model anyone can have. It is bundled with FACET as
`facet/data/examples/practice_glass.data` (the final frame, a LAMMPS data
file) and `practice_glass.lammpstrj.gz` (six frames, 2 ps apart), and opened
with `facet.example("glass")` and `facet.example("glass-trajectory")`.

It is a model for trying the library, with 1200 atoms and a quench of
10 K/ps. It is released into the public domain (CC0 1.0), as the bundled
crystal structures are.

## What it is

| | |
|---|---|
| Composition | 20 Na₂O · 10 CaO · 5 Al₂O₃ · 65 SiO₂ (mol %) |
| Atoms | 1200: 700 O, 260 Si, 40 Al, 160 Na, 40 Ca |
| Potential | A. Pedone, G. Malavasi, M. C. Menziani, A. N. Cormack and U. Segre, *J. Phys. Chem. B* **110** (2006) 11780: partial charges (O −1.2, Si +2.4, Al +1.8, Na +0.6, Ca +1.2), a Morse term and a C/r¹² repulsion for each cation–oxygen pair and for O–O |
| Program | LAMMPS, version 22 July 2025, `pair_style hybrid/overlay pedone 8.0 coul/long 10.0`, PPPM to 10⁻⁵, time step 1 fs |
| Start | atoms placed at random in a cubic box at 2.45 g/cm³, none within 1.7 Å of another (`make_start.py`, seed 20261112) |
| Melt | 60 ps at 3500 K, constant volume |
| Quench | 3500 K to 300 K at 10 K/ps, constant volume |
| Relaxation | 60 ps at 300 K and zero pressure, then 10 ps more for the six frames |

The pair parameters are in `in.practice_glass`, which is the complete LAMMPS
input, and `log.lammps` is the log of the run that made the bundled files.

## What it measures as

Measured with FACET 2.1.0 at the default thresholds (bond 0.075 v.u., listing
0.02 v.u., Brese and O'Keeffe parameters), over the six frames unless stated:

| Quantity | Value |
|---|---|
| Density at 300 K and zero pressure | 2.665 g/cm³ (sd 0.006 over the frames; 2.661 for the final frame, as LAMMPS printed) |
| First peak of g(r): Si–O, Al–O, Na–O, Ca–O | 1.60, 1.73, 2.29, 2.36 Å |
| First peak of g(r): O–O, Si–Si | 2.59, 3.08 Å |
| Si coordination | 4 for 99.6 % of the Si, 5 for 0.4 % |
| Al coordination | 4 for 93.3 % of the Al, 5 for 6.7 % |
| Mean coordination of Na and of Ca | 4.2 and 5.5 |
| Oxygen | 71.8 % bridging, 27.9 % non-bridging, 0.14 % free, 0.17 % bonded to three formers |
| Qn of Si | Q1 2.3 %, Q2 12.2 %, Q3 43.1 %, Q4 42.0 %, five-fold 0.4 % |
| Most frequent O–Si–O and Si–O–Si angles | 108.5° and 153.5° |
| Net charge | 0 |

The composition fixes the count of non-bridging oxygens when every Al is a
charge-compensated tetrahedral former: Na + 2 Ca − Al = 200 of the 700
oxygens (28.6 %). The model has 27.9 % non-bridging and 0.14 % free oxygens,
the difference being the oxygens bonded to three formers and the five-fold
aluminium.

The density is that of the potential at zero pressure with this quench. Real
soda-lime silicate glasses are near 2.5 g/cm³.

## Making it again

```
python make_start.py                               # writes start.data
mpiexec -n 6 lmp -in in.practice_glass             # about 40 minutes on six cores
```

This writes `practice_glass.data` and `practice_glass.lammpstrj`. The random
velocities and the thermostat make a second run a different glass of the same
kind: the numbers above then change in their last digits, and the tests that
state the atom counts, the stoichiometry of the oxygens and the density
printed in `log.lammps` still hold for the files that are bundled.
