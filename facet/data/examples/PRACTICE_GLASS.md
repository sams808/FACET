# The practice glass model

`practice_glass.data` and `practice_glass.lammpstrj.gz` are a small
molecular-dynamics model of a glass, bundled so that the examples of the
library manual run on a fresh installation: `facet.example("glass")` opens
the first (one frame, a LAMMPS data file) and
`facet.example("glass-trajectory")` the second (six frames, 2 ps apart).

- Composition 20 Na₂O · 10 CaO · 5 Al₂O₃ · 65 SiO₂ (mol %): 1200 atoms, 700 O,
  260 Si, 40 Al, 160 Na, 40 Ca. LAMMPS types 1 to 5 are O, Si, Al, Na, Ca.
- Made for FACET with LAMMPS (22 July 2025) and the interatomic potential of
  A. Pedone, G. Malavasi, M. C. Menziani, A. N. Cormack and U. Segre,
  *J. Phys. Chem. B* **110** (2006) 11780: melted at 3500 K, quenched to 300 K
  at 10 K/ps, relaxed at zero pressure.
- Released into the public domain under the CC0 1.0 Universal dedication. No
  restriction is placed on passing FACET on with it included.

It is a model for trying the library. Its input files, the log of the run
and what it measures as are in `docs/practice_glass` of the repository.
