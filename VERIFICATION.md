# How FACET is verified

FACET exists because published coordination numbers are not reliable. A program
making that argument has to be held to a higher standard than the numbers it
criticises, and "the tests pass" is not that standard: a program that only checks
itself can be consistently wrong, and was.

So every quantity is checked against something outside the code that produces it.
Five kinds of check, each answering a different question.

| | The question | What it caught |
|---|---|---|
| 1 | Does the source hold together? | — |
| 2 | Does an independent implementation agree? | the symmetry expansion losing three quarters of one structure's oxygen |
| 3 | Is the answer a property of the crystal, or of how it was written down? | the gap coordination number changing under an origin shift |
| 4 | Does the program contradict itself? | the cutoff table double-counting every mixed-valence structure |
| 5 | What happens when the input is wrong? | four errors that surfaced as tracebacks instead of explanations |

Run over the whole bismuth collection — 90 CIF files, 89 readable — not one
structure, because a single agreement can be luck.

The output of molecular dynamics is checked the same five ways, on model files
written by LAMMPS and other programs, in "MD models" at the end.

---

## 1. Static integrity

81 source and test files compile. Every module imports. The engine still pulls in
no Qt, and loads in under 300 ms. No bare `except`, no mutable default arguments,
no debugger or `print` left behind outside the build script. The thirteen places
that swallow an exception were each read: all are optional-feature probes where
failure genuinely means "not available", and every caller handles absence.

One was changed anyway. A failed symmetry determination used to vanish silently,
which would have removed the space-group cross-check from every non-CIF file with
nothing to show that it had. It now leaves a note.

## 2. Against independent implementations

gemmi, spglib and pymatgen have no code in common with FACET.

| Quantity | Compared with | Worst over 89 structures |
|---|---|---|
| Cell volume | gemmi | 9 × 10⁻¹⁶ relative |
| Reciprocal cell lengths / angles | gemmi | 4 × 10⁻¹⁶ / 1 × 10⁻¹⁴ ° |
| d-spacing | gemmi, and FACET's own second route | 5 × 10⁻¹⁶ relative |
| Structure factor \|F\|² | gemmi's own calculator | 2 × 10⁻⁴ of the strongest line |
| Space group | spglib | no disagreement |
| Density | the composition, independently | 3 × 10⁻¹⁰ relative |
| Neighbour distances | pymatgen | exactly zero |
| Grid interpolation | an analytic linear field | 3 × 10⁻¹⁵ |

Two comparisons deserve a note.

**The structure-factor residual** is finite CIF coordinate precision, not
arithmetic. Reflections with real intensity agree to seven figures; the residual
is on reflections that are systematically absent, which gemmi gets exactly zero
by applying operators and FACET gets only as close to zero as the file's decimal
places allow. With exact coordinates — a synthetic NaCl — the absences come out
at exactly zero.

**One structure now disagrees with gemmi deliberately.** See below.

## 3. Invariance

Each of these is a different way to write the same crystal. A number that changes
under one of them is not measuring the crystal.

| Transformation | Worst change in any reported number |
|---|---|
| Moving the origin | 2 × 10⁻¹⁴ |
| Rotating the whole structure | 2 × 10⁻¹⁴ |
| Relabelling the axes | 2 × 10⁻¹⁴ |
| Building a supercell | 2 × 10⁻¹⁴ |
| A different symmetry copy as the representative | 7 × 10⁻⁴ \* |
| Shifting every R₀ (for φ) | 3 × 10⁻¹⁶ |

\* One file. Its coordinates satisfy its own space group only to 0.0006 Å, so
symmetry-equivalent bonds genuinely differ. FACET reports that rather than
hiding it.

The φ index is invariant to a shift in R₀ to 3 × 10⁻¹⁶ over 2000 random
environments, while \|BVV\| scales by exactly exp(δ/b). That is the whole reason
φ is what gets reported: R₀ carries ±0.02 Å even when fitted.

## 4. Internal consistency

Twenty-two places where two parts of the program compute the same thing. All
agree: a contact's valence against exp((R₀−d)/b) exactly; bond-valence sums
against the sum of their own contacts to 9 × 10⁻¹⁶; coordination numbers against
the contacts above the threshold exactly; Bragg's law and d-spacings across three
modules to 6 × 10⁻¹⁴; every drawn atom on a real position to float32 precision;
every coordination label saying what the analysis says; no method reporting more
neighbours than there are contacts.

The strongest of these is a physical law rather than a comparison. The cation and
anion bond-valence totals are the same bonds counted from opposite ends by two
separate neighbour searches, so they must agree — and they do, to machine
precision, on 23 of 26 structures. Where they do not, the difference measures how
far the deposited coordinates fall short of the declared symmetry.

## 5. Robustness

Empty files, prose, binary, zero-length axes, collinear axes, unknown elements,
non-numeric coordinates, atoms on top of each other, zero occupancy, nonsense
symmetry operations, 500 Å and 0.5 Å cells; structures with no anions, no
cations, no atoms; parameter files that are not parameter files; every reader
against every file including the wrong reader for the format. Nothing raises an
exception type a caller cannot catch, and nothing returns a `nan` as a result.

---

## What the verification found

Five faults, none of them visible from inside the program. Every structure was
self-consistent, the suite was green, and the numbers looked reasonable.

**The symmetry expansion was losing atoms.** FACET took gemmi's
`get_all_unit_cell_sites()`, which merges positions close in *fractional*
coordinates. For an Fm-3m entry with a = 5.542 Å and oxygen on the 32-fold
(0.266, 0.266, 0.266), it returned 8 oxygen atoms instead of 32, because the
F-centred images land 0.032 fractional away — 0.18 Å in that cell, a real
separation. Three quarters of the oxygen, and of the anion charge, were missing.

FACET now applies the operations itself and de-duplicates by distance in
angstrom, which cannot make that mistake at any cell size. Two checks that would
have caught it are permanent: the computed multiplicity against
`_atom_site_symmetry_multiplicity`, and the expanded cell against
`_chemical_formula_sum` — the one statement in a CIF that comes from neither the
coordinates nor the symmetry. The formula check flags two structures across the
collection, and both are chemically real: sillenite γ-Bi₂O₃, and
Bi₂Sr₂CaCu₂O₈₊δ, whose excess oxygen is the reason anyone studies it.

**The gap coordination number was not a function of the structure.** Brunner's
rule takes the largest step in the sorted distances, and a site whose contacts
are all one distance — a regular octahedron of one symmetry-equivalent oxygen,
which is common — has every ratio exactly 1. `argmax` over equal values returns
whichever index rounding favours, so the same structure written with its origin
elsewhere gave 3, 4, 5 or 6. There is no gap, so there is nothing to truncate:
the answer is every contact.

**The cutoff table was double-counting mixed-valence structures.** Bi(3+)–O and
Bi(5+)–O are separate rows with different R₀ and different cutoffs, but the
measured columns were keyed by element pair alone. Bi2212's bond count came out
529 against an actual 378 — and the structures affected are Bi₄O₇, Bi₂O₄, BaBiO₃
and Bi2212, every mixed-valence bismuth compound in the collection.

**Four errors surfaced as tracebacks rather than explanations**: an empty file as
`IndexError: invalid vector subscript`, a valence threshold of zero as
`math domain error` from inside a logarithm, a ligand on the central atom as a
`nan` and a `RuntimeWarning`, and a broken parameter file as a `JSONDecodeError`
naming nothing. A stated b of zero was silently replaced by 0.37, which would
have changed every valence in the file without saying so.

**A shared site was reported sixteen times**, once per symmetry copy — the same
information made unreadable.

And one performance fault: the volume panel's default grid spacing took 31.6
seconds, because the sum evaluated every grid point against every anion in every
periodic image, of which about thirty of 175 are ever in range. Asking a tree
which are near gives the same sum over the same terms, 40× faster, agreeing to
6 × 10⁻¹⁶.

---

## The one place FACET now disagrees with gemmi on purpose

For `Bi0.92 O1.54 Si0.08`, FACET produces 40 atoms where gemmi's small-structure
expansion produces 16. FACET is right, and the file says so: its formula is
`Bi0.92 O1.54 Si0.08` with Z = 4, and FACET's cell gives Bi 0.92, Si 0.08,
O 1.536 per formula unit. Before the fix it gave O 0.384 — a quarter.

Two further confirmations. The file declares the oxygen site as Wyckoff 32f and
lists 192 symmetry operations; applying them gives 32 positions. And the
alternative disagreement in the collection runs the other way: one file declares
multiplicity 1 for a general position in an eight-fold group, where FACET's 8
gives O:Bi = 1.5000 exactly for Bi₂O₃. The file is wrong there, and FACET says so.

---

## The later features, and what checking them found

The pair distribution function, the EXAFS shell table and the bond-valence
vector overlay were added after the five passes above, and each was checked the
same way. Three of the four faults below were invisible from inside the program:
the code ran, produced numbers, and the numbers were wrong.

### The bond-valence vector, against the analysis that never sees it

The drawn lobe's direction and length come from φ and the vector sum computed
from the scene's cached bond arrays. `coordination.analyse_structure` reaches the
same φ through its own neighbour search and `bv.phi_index`. Comparing the two
over the collection:

| | Before | After |
|---|---|---|
| Worst \|Δφ\| over 67 sites | **0.130** | 8 × 10⁻⁸ |

The 0.130 was on a partially occupied phosphorus site: the analysis weights every
valence by its ligand's occupancy and the drawn sum did not. Two further checks
now hold — every atom of one site gets the same φ to 5 × 10⁻⁷, which is what
would fail if directions were propagated between symmetry copies by translation,
and the drawn axis is exactly antiparallel to the weighted sum.

### The PDF, against closed forms and a second transform route

| Check | Result |
|---|---|
| NaCl first-peak area against 6·b(Na)·b(Cl)/⟨b⟩² | agrees to 2 × 10⁻⁷ |
| Second peak against 6(b(Na)²+b(Cl)²)/⟨b⟩² | agrees to 2 × 10⁻⁵ |
| Shell degeneracies against `utilities.radial_shells` | 6 at 2.8201 Å, 12 at 3.9882 Å |
| Truncation kernel against an explicit forward-and-back sine transform | 0.16 % of the peak height |
| Origin shift | exactly zero |
| 2×2×2 supercell of the same crystal | 3 × 10⁻¹³ |
| Element-pair weights, X-ray / neutron / electron | 76.3 / 24.5 / 66.4 % Bi–Bi, computed twice independently |

Two faults came out of it. The odd extension G(−r) = −G(r) was built by gluing
`-r[::-1]` onto `r`, which leaves a gap of 2·dr across the origin — so the array
was not a uniform sampling and the convolution over it meant nothing. Including
r = 0, where an odd function is zero, fixes it. And binning each distance to its
nearest grid point shifted every peak by up to dr/2; the error scaled exactly
linearly with dr (0.58, 0.30, 0.16, 0.070 as dr halved). Depositing each distance
linearly between its two neighbouring points cut it 38-fold at dr = 0.01 Å and
made the peak position independent of where the grid falls — which matters
because someone reads a bond length off that plot.

### EXAFS, against FEFF itself

The exported `feff.inp` was **refused by FEFF8L**. No internal check could have
found that: the file was well formed, self-consistent, and rejected. Two separate
reasons, both fixed and both now asserted by tests.

- An atom of the absorber's element that was not the absorber was given potential
  index 0, which FEFF reserves for the absorber alone.
- Annotations were written as extra columns in `POTENTIALS` and `ATOMS`, which
  FEFF reads positionally: a trailing word is parsed as the next number.

With those fixed FEFF8L runs FACET's output end to end, producing 50 path files,
`files.dat`, `chi.dat` and `xmu.dat`. Then:

| Check | Result |
|---|---|
| FEFF's single-scattering path lengths against FACET's shell radii | every one within 0.06 Å, most exact to 4 decimals |
| FACET's path sum against FEFF's own `chi.dat` | 2.5 % of the peak, correlation 0.99982 |
| ΔR against π/(2Δk), N_idp against 2ΔkΔR/π | exact |
| Einstein σ² at T → 0 and T → ∞ | the zero-point and classical limits, to 1 % |
| The Einstein and k↔E constants | recomputed from ℏ, k_B, m_e and u |

Both FEFF readers were also wrong at first, and silently: `files.dat` has a
`sig2` column between the file name and the amplitude ratio, and counting past it
reported paths with a hundred legs at a uniform 2.0 Å. In `feffNNNN.dat` the
nleg/deg/reff line is not a header followed by data — it *is* the data, with its
column names written after the numbers.

### What is still not computed

χ(k) from the structure alone, and XANES. Both are declined in the module
docstring, in the manual and in `FEATURES.md`, with the reason: a
single-scattering χ(k) without phase shifts puts the peaks of its Fourier
transform about 0.4 Å from the distances that generated it, which is the one
quantity anyone would read off such a curve.

---

## Reproducing it

The permanent tests live in `tests/`. Those that correspond
directly to the faults above are in `tests/test_expansion_verification.py` and
`tests/test_robustness.py`, each naming what was wrong and how it was found.

```
py -3.11 -m pytest tests/ -q
```

The five passes were written as scripts over the whole collection rather than as
unit tests, because their value is breadth: a single structure agreeing with
gemmi proves much less than eighty-nine doing so. What they established has been
turned into tests where a test can carry it.

---

## MD models

FACET reads and measures the output of molecular dynamics. It does not run MD.
A glass frame holds thousands of atoms and no symmetry, and none of the checks
above reaches the code that handles one: the MD readers, the per-atom
bond-valence engine (`bulk.py`), the glass descriptors, the network, order,
scattering, spectroscopy and dynamics modules, the driver that runs them over a
trajectory, and the Model window. Each was checked the same five ways.

Two sets of input were made for this, both kept outside the repository:

- **A recognition corpus of 115 files written by the real programs**: LAMMPS
  22 Jul 2025, ASE 3.29, OVITO 3.16 and pymatgen, in every style each one
  writes (126 files on disk, a per-frame series counting once). All hold one
  270-atom model in a tilted box that deforms every frame, and the ground truth
  is LAMMPS's own internal state, read through its library interface after each
  step rather than from any file.
- **Five LAMMPS models** made with the published SHIK and Pedone potentials:
  SiO₂ (SHIK), Na₂O·3SiO₂ (NS3, Pedone), 15Na₂O–10Al₂O₃–75SiO₂ (NAS, SHIK) and
  0.5Na₂O·B₂O₃·2SiO₂ (NBS, SHIK), each quenched from the melt to 300 K and
  sampled in 20 frames 1 ps apart, and an NS3 melt held near 3000 K. Each holds
  2880 to 3000 atoms. Their content is known; they are not reference glasses
  (see "What is not verified").

| | The question | What it caught |
|---|---|---|
| 1 | Does the source hold together? | the format modules invisible to PyInstaller, so a frozen build would have read none of their 14 formats |
| 2 | Does an independent implementation agree? | the automatic g(r) minimum landing on noise: Na–O cut at 2.15 Å on one NS3 frame, before the first peak, where LAMMPS's frame-averaged g(r) has its minimum at 3.23 Å |
| 3 | Is the answer a property of the model, or of how it was written down? | S(Q) and the bond-length histograms depending on the order of frames whose box changes |
| 4 | Does the program contradict itself? | Voronoi cells that broke Euler's law, Σ(6−k)n_k = 9 instead of 12 |
| 5 | What happens with damaged or impossible input? | a file cut inside its last number, read with the shortened value and no note |

### 1. Static integrity

The 17 MD modules under `facet/core` and the command line `facet.md.cli` are in
the Qt-free import test (`tests/test_imports.py`), which passes. All 162 Python
files under `facet/` and `tests/` compile.

The 27 MD files (17 in the engine, 3 for the command line, 7 for the Model
window; 56 211 lines) have no bare `except`, no mutable default argument and no
debugger call. `print` appears only in the command line, which writes to the
terminal, and in the usage line of `md_workspace`'s own entry point. The 19
exception handlers in the engine and the command line that end in `pass`,
`continue` or `return None` were each read: format and element probes that
answer "not this", a fast parse that falls back to a slower one, header fields,
records and text lines skipped while scanning, two destructors at interpreter
shutdown and a terminal-encoding setting. The 17 such handlers in the Model
window's modules were not read for this section. `ruff --select F,E9` finds no
syntax error in the MD files and their tests, and lists 10 warnings: four unused
imports, one f-string with no placeholder, one unused local in a test, and four
reads of a name that a test deletes after its lambdas have run.

No verdict word appears in the text FACET writes about an MD model. The scans
covered:

- 12 402 texts from the spectroscopy runs on the models, and 1 627 from the
  glass runs;
- about 2 600 refusals and notes over the corpus;
- in the Model window, 12 306 texts on its read pages and workspace, 1.95
  million while visiting all 258 result items of an NS3 run, and the
  15 674-character manual section.

One fault showed up only when PyInstaller's module graph was checked. The
format modules are imported by name at run time, which PyInstaller cannot
follow: PyInstaller 6.11.1's module graph found none of the four, so a built
FACET would have read none of the 14 formats they hold. A never-called
function with plain import statements fixes this, and a test keeps that
function in step with the module list.

### 2. Against independent implementations

None of the outside codes shares code with FACET. Each ran in a throwaway
environment outside the repository; they never import FACET, and FACET and its
tests never import them.

**Reading the files.**

| Input | Compared with | Worst |
|---|---|---|
| LAMMPS dumps and data files in every coordinate convention: tilted box with an origin, deforming every frame, image flags, metal and real units | LAMMPS's internal state | fractional 3.3 × 10⁻¹⁶; unwrapped from `xsu` 7.4 × 10⁻¹³ Å; general triclinic 1.4 × 10⁻¹⁴ Å; atoms stored outside the box 7.1 × 10⁻¹⁵ Å; velocities exact; times 5.6 × 10⁻¹⁷ ps |
| A 45 MB dump and its gzip twin (20 000 atoms × 16 frames), read forwards and then backwards | LAMMPS's internal state | fractional 3.3 × 10⁻¹⁶; unwrapped 1.1 × 10⁻¹³ Å; velocities exact |
| FACET's 17 reader test files | OVITO 3.16 reading the same files | 4 × 10⁻¹⁵ Å |
| A DL_POLY 4 HISTORY (KCl, 216 atoms); VASP XDATCARs from pymatgen's test data | OVITO and ASE | 5.3 × 10⁻¹⁵ Å; fractional 2.2 × 10⁻¹⁶ |
| Fractional coordinates, volume and cell of the frame model | ASE 3.29 | bitwise; perpendicular widths 1.5 × 10⁻¹⁵ relative |

FACET reads 114 of the 115 corpus files: 70 with no option, 43 with the option
their refusal names (a type map, the data file a DCD needs for its elements, or
a box for a plain XYZ), and one whose velocities need a `units=` the file does
not state. The 115th, an OVITO NetCDF of a tilted cell, is refused with its
reason (see "What the verification found"). Each file read agrees with the truth
to within its writer's own rounding:

| Writer | Format | Worst position error (Å) | Writer's rounding (Å) |
|---|---|---|---|
| LAMMPS | data | 1.95 × 10⁻¹⁴ | 10⁻⁹ |
| LAMMPS | dump, `%g` | 1.11 × 10⁻⁴ | 2.3 × 10⁻⁴ |
| LAMMPS | DCD (float32) | 2.56 × 10⁻⁶ | 10⁻⁵ |
| LAMMPS | XTC | 8.48 × 10⁻³ | 8.7 × 10⁻³ (its 0.01 Å grid) |
| LAMMPS | binary dump | 4.35 × 10⁻¹⁵ | — |
| ASE | .traj | 5.6 × 10⁻¹⁵ | 10⁻⁴ |
| ASE | XDATCAR | 6.41 × 10⁻⁷ | 10⁻⁴ |
| ASE | multi-model PDB | 2.99 × 10⁻³ | 3.67 × 10⁻³ |
| OVITO | dump, data, extended XYZ | 8.53 × 10⁻¹¹ | 10⁻⁶ |
| OVITO | GSD | 7.1 × 10⁻¹⁵ | 10⁻⁵ |
| pymatgen | XDATCAR | 1.28 × 10⁻⁵ | 2.3 × 10⁻⁵ |

506 crystal files from local collections open as they did before, and none is
sent to the MD reader.

**The per-atom engine: the crystal-as-glass equality.** Eleven crystals, the six
bundled examples and five ordered structures from the reference collection, are
expanded to P1, tiled 2×2×2 and analysed as if each were an MD frame
(`tests/test_bulk_equivalence.py`). Every atom equals `analyse_site` on the
crystal atom it came from: CN, listed CN, BVS, φ and plateau width, at three
thresholds in the test and at fifteen in the re-check. The worst difference is
2.49 × 10⁻¹⁴, and every CN is equal. Equality per site, rather than per atom,
cannot be reached on two of the files with any engine: quartz and BiPO₄ round
their special positions, so symmetry copies of one site already differ in BVS
by 1.1 × 10⁻⁴ and 5.9 × 10⁻⁵ v.u. inside the crystal. That residual is recorded,
not asserted.

| Quantity | Compared with | Worst |
|---|---|---|
| Pair sets (i, j, image) and distances | ASE 3.29 `neighbor_list`, 6 frames including left-handed and strongly tilted cells | identical pairs; 1.4 × 10⁻¹⁴ Å |
| same | OVITO 3.16.1 `CutoffNeighborFinder`, 4 frames | identical pairs; 1.8 × 10⁻¹⁴ Å |
| same | a brute-force engine with explicit lattice loops and no tree | identical pairs; 4.4 × 10⁻¹⁵ Å |
| Every per-atom field, on random, thin and left-handed non-crystalline frames, at five thresholds | the crystal path, atom by atom | equal to 10⁻¹⁰ |

**The glass descriptors.** vitrum 1.1.0 and amorphouspy 0.8.1 read the same
dumps as FACET, for the four 300 K glasses, 20 frames each. FACET's distance
cutoffs came from its own 20-frame result and were passed to both tools as exact
floats.

| Quantity | Compared with | Worst |
|---|---|---|
| CN per atom, every cation and O | vitrum, amorphouspy | 0 of 237 600 atom-frames differ |
| Qⁿ per atom (formers Si; Si and Al; Si and B) | vitrum, amorphouspy | 0 of 71 300 former-frames differ |
| Formers bonded to each O | vitrum (the count), amorphouspy (free, NBO, BO, tricluster) | 0 differ |
| Running CN N_ab(r) | amorphouspy `cn_cumulative`; a count on vitrum's distances | 0 at every grid point and every frame |
| Partial g(r), every ordered pair | an independent linear deposit with its own dump parser and tree | 3.1 × 10⁻¹³ |
| Partial g(r), NS3 frame 0 | OVITO 3.16 `CoordinationAnalysisModifier` | 1.9 × 10⁻¹³ |
| Running N(r) | LAMMPS `compute rdf` on full-precision positions | 2 × 10⁻⁴ to 1.1 × 10⁻³, the dump's six digits |
| CN, O speciation, Qⁿ, Qⁿ(mX), linkages, anion environments, angle and bond-length histograms, under both bond definitions | a brute-force counter written from the definitions | exact, every frame |
| Hand-built units, crystals of known Qⁿ, rock-salt shells | closed forms | exact; shells 6, 8, 24, 30 to 10⁻¹² |
| Density | LAMMPS | 5 × 10⁻⁷ to 1.7 × 10⁻⁶ relative |

The CN and Qⁿ agreement is exact again at the cutoffs FACET picks from the last
frame alone (0 of 11 880 atoms and 0 of 3 565 formers differ) and at the
distances the bond-valence cut amounts to (0 of 237 600 atom-frames and
0 of 71 300 former-frames). The codes differ on a pair exactly at a cutoff:
vitrum counts d < r, FACET and amorphouspy d ≤ r. That never came into play
here. The closest cation–O pair to any distance cutoff is 1.9 × 10⁻⁵ Å away,
and 1.2 × 10⁻⁶ Å at the bond-valence distances.

The g(r) curves as each code publishes them do differ: at most 0.85 on the
20-frame average (NAS Al–Al at 3.285 Å, 4.73 against FACET's 3.88) and 4.18 on
one frame, again NAS Al–Al. The cause is how a distance is binned. FACET
deposits each distance linearly between its two nearest grid points, for the
reason given in the PDF section above; the tools count it in one bin, offset by
half a step. Two further checks account for the whole difference:

- the tools' counts at a twentieth of a bin, re-spread the way FACET spreads
  them, leave at most 0.034 on the average and 0.15 on one frame;
- FACET's estimator applied to vitrum's own distance matrix gives 2.9 × 10⁻¹².

**Network, order, scattering, spectroscopy, dynamics.**

| Quantity | Compared with | Worst |
|---|---|---|
| Primitive (Franzblau) ring counts, sizes 4–24 | matscipy 1.3.0, four glasses | equal at every size |
| Guttman rings | vitrum 1.1.0 | identical ring for ring: 1066, 546, 841 and 926 |
| King and Guttman rings | an oracle written from the R.I.N.G.S. definitions | identical |
| Coordination sequences, 10 shells | RCSR, every vertex of 64 nets | exact |
| Vertex symbols | RCSR | 24 of 24 vertex types on 18 nets; 48 of 50 in a random sweep, the other 2 listed in another order in RCSR's own data file |
| Steinhardt q_l and w_l, l = 2–12 | pyscal3 4.0.0, 6 frames | 1.0 × 10⁻¹⁴ |
| Voronoi index, every atom | OVITO 3.16, frame 0 of the four glasses | equal; volumes to 6.8 × 10⁻¹⁵ relative |
| Empty spheres | a brute-force minimum over atoms and images | 3.6 × 10⁻¹⁵ Å |
| Probe-occupiable free volume | an exact per-point oracle | equal (NS3: 0.74258) |
| Partial S(Q) by the sine route | an exact Debye sum, SiO₂ frame, 0.3–25 Å⁻¹ | 5.3 × 10⁻⁴ (Lorch window) |
| S(q) on the reciprocal lattice | a brute-force sum | 10⁻¹⁰ |
| `feff.inp` for 15 clusters (Si, Al and Na absorbers) | FEFF8L | every input accepted; single-scattering path lengths within 5.04 × 10⁻⁵ Å of the distances in the dump |
| χ(k) from FEFF's path files, per cluster, k 3–14 Å⁻¹ | FEFF's own `chi.dat` | 0.42 % (Al), 0.94 % (Na), 1.95 % (Si) |
| Shell cumulants | closed forms by Gauss quadrature | 4.7 × 10⁻¹⁵ relative |
| MSD over every time origin | a direct O(T²) sum | 3.2 × 10⁻¹² Å² |
| MSD from one origin | LAMMPS `compute msd` | 4.8 × 10⁻⁶ relative, the dump's six digits |
| MSD and VACF | harmonic oscillators; ballistic motion; an Ornstein–Uhlenbeck particle | 6.2 × 10⁻¹⁴ (MSD) and 1.1 × 10⁻¹³ (VACF) relative; MSD 4.2 × 10⁻¹⁴ relative; MSD within 0.54 % against a block scatter of 0.65 % |
| The driver | each module called directly, NS3, 5 frames | identical; g(r) to 1.3 × 10⁻¹⁴ |

The automatic EXAFS first-shell limits against the first minima of LAMMPS's
`compute rdf`:

| Pair | FACET (Å) | LAMMPS (Å) |
|---|---|---|
| Al–O | 2.37 | 2.41 |
| Al–Si | 3.62 | 3.61 |
| Si–Si | 3.48 | 3.47 |
| Si–O | 2.27 | 2.03 |

For Si–O, FACET reports a valley floor of 1.91–2.64 Å, which contains the
LAMMPS minimum, and N is 4.007 against LAMMPS's 4.005.

#### Where the codes differ, and why

**Like-pair g(r) normalisation.** FACET divides by N_a², so a like pair's g
tends to 1 − 1/N_a, as its docstring states; vitrum and amorphouspy divide by
N_a(N_a − 1). The integral of their g over FACET's, from 4 to 5.9 Å, is 1.00046
to 1.0057 for like pairs, against an expected N/(N − 1) of 1.0005 to 1.0056;
unlike pairs give 0.9996 to 1.0003, the size of the window-edge noise.

The N_a(N_a − 1) form was tried for S(Q) and not adopted. In a closed box, g
tends to 1 − ⟨δN_a δN_b⟩/(N_a N_b) (Lebowitz and Percus, Phys. Rev. 122, 1675):
that is −1/N_a only for an ideal gas, and about 0 for a dense glass. At low Q
on the SiO₂ glass:

| Route | Low-Q S(Q) |
|---|---|
| reciprocal lattice | 0.12–0.21 |
| FACET's normalisation | 0.133–0.144 |
| N_a(N_a − 1) | 0.158–0.295 |

A note states the size of the term at the grid's first Q.

**King's rings.** Under King's criterion FACET bars only the node itself;
vitrum also bars the node's other neighbours. SiO₂ gives 2891 rings against
2869; NS3, which has no chords, gives 1430 in both. Which rule R.I.N.G.S.
itself uses is marked "reference to verify".

**The outside codes are not ground truth either**, which is why more than one
was used:

- amorphouspy 0.8.1 has its cumulative-CN keys reversed in the `compute_rdf`
  docstring (`rdf.py:303-304`).
- amorphouspy and vitrum differ on the smallest Guttman rings in SiO₂:
  amorphouspy gives 70 three-membered rings, vitrum 6 two-membered and 54
  three-membered. Neither was used as a reference below four members.
- ASE's DL_POLY writer converts velocities with the atomic unit of velocity,
  so its files hold 222.7 times the velocities it was given.
- OVITO's MSD at its default settings gives Na 194.84 Å² where LAMMPS gives
  223.96 Å².

### 3. Invariance

| Transformation | Data | Worst change |
|---|---|---|
| Translation | 11 crystals as frames | 1.2 × 10⁻¹³ |
| Origin from 10² to 10⁷ Å, same fractions | quartz, eulytite | exactly 0 |
| Permuting the rows | 11 crystals; an NS3 frame of 3000 atoms | 2.1 × 10⁻¹⁴; no difference |
| Rotation, reflection, relabelling the axes | 11 crystals | 6.0 × 10⁻¹⁴ |
| Supercells 1×1×1 to 3×3×3, 2×3×1, 1×1×3, 3×1×2 | 11 crystals | 4.6 × 10⁻¹⁴ |
| Another cell of the same lattice | an NS3 frame | no difference |
| A 2×2×2 supercell (24 000 atoms) | an NS3 frame | fractions identical; counts exactly 8×; g(r) and N(r) within 10⁻¹² |
| Wrapped against unwrapped (r + k·box) | crystals | 1.6 × 10⁻¹³ for \|k\| ≤ 3; 4.9 × 10⁻¹¹ for \|k\| ≤ 1000, lost when the frame wraps them |
| Unwrapped image-flag positions given as input | NS3 frames | no difference |
| Frame order, with a fixed box and with box-changing frames | NS3 frames | no difference; S(Q) to 2.2 × 10⁻¹⁶ |
| Translation; rotation (Voronoi volumes) | 20 frames | 1.1 × 10⁻¹⁴; 8.1 × 10⁻¹⁵ relative, faces and index identical |
| A 2×2×1 replicate (rings) | SiO₂ | counts exactly 4× |

These checks found three faults, all fixed:

- **φ of a one-bond atom** came out as 1 + 2.2 × 10⁻¹⁶, above the histogram's
  last edge, and a rigid move could flip it back. φ is now clipped at 1.
- **Results drifted with the origin.** The drift passed 10⁻¹⁰ near
  \|origin\| = 3 × 10⁵ Å, because pair vectors were built from Cartesian
  positions that include the origin. The search now reads the fractions and the
  box.
- **Box-changing frames were order-dependent.** The bond-length histogram
  edges and the scattering grid depended on which frame came first. The edges
  now come from the bond-valence radius or the largest cutoff, and the
  scattering grid is the one the smallest box in the run holds.

### 4. Internal consistency

**Bond valence against distance, on the same atoms.** This is the comparison
FACET exists to make, applied to glass. The bond-valence cut keeps a contact
above 0.075 v.u., with Brese–O'Keeffe R₀ and b = 0.37, which is exactly a
distance cut at R₀ − b ln 0.075:

| Pair | Distance cut (Å) |
|---|---|
| Si–O | 2.5824 |
| Na–O | 2.7584 |
| Al–O | 2.6094 |
| B–O | 2.3294 |

| Model | Mean CN, bond valence − distance (20 frames) | Atoms whose CN differs |
|---|---|---|
| SiO₂ | Si +0.0022 | 0.2 % of Si |
| NS3 | Na −1.92 (5.63 → 3.71); O −0.55; Si +0.003 | 87 % of Na |
| NAS | Na −0.83; O −0.12; Al +0.021; Si +0.003 | 59 % of Na |
| NBS | Na −0.79; O −0.10; Si +0.017; B +0.017 | 57 % of Na |

The two cuts disagree in opposite directions:

- **Si, Al and B:** the bond-valence distance lies beyond FACET's g(r) cut
  (Si–O 2.18–2.29, Al–O 2.37, B–O 2.01 Å) and takes in a few more O. The Qⁿ
  fractions move by at most 0.021 on the 20-frame average (NAS Al Q⁴) and 0.044
  on the last frame (NAS Al Q⁵).
- **Na:** the bond-valence distance, 2.76 Å, lies short of the g(r) cut, which
  is 3.26, 3.04 and 2.99 Å in NS3, NAS and NBS. The Na–O valley is wide:
  FACET states its floor as 2.95–3.58, 2.81–3.28 and 2.75–3.24 Å, and
  across the NS3 floor N_NaO(r) runs from 4.45 to 7.09.

**Qⁿ against the bridging and non-bridging O.** The Qⁿ distributions count
bridges from the formers, the O speciation counts formers from the O; both must
imply the same bridging bonds. On FACET's published result for the four
glasses, 20 frames each, under both bond definitions (160 frame–definition
pairs), Σ n·N(Qⁿ) over the formers equals the bridging bonds counted from the O
side in every one, and BO + tricluster equals the number of O bonded to two or
more formers. On frame 0 of SiO₂ by distance, Σ n·N(Qⁿ) = 3991 = 2 × 1961 BO +
3 × 23 triclusters. That shortcut, 2·BO + 3·tricluster, holds on 158 of the
160; the other two are NBS frames under the bond-valence cut, where one O is
bonded to four formers and the speciation files it, by definition, as a
tricluster. `tests/test_glass.py::test_qn_bridges_equal_the_bridges_counted_from_the_anions`
holds the same identities, and the Qⁿ(mX) linkage count, on a random box with
triclusters.

**Identities between two parts of the program:**

- **Cation and anion bonds.** The bonds read from the cation side and from the
  anion side are bit-identical sets at 23 thresholds on 11 crystals and 3
  synthetic frames, and at 4 thresholds on 9 261- and 29 791-atom frames, on
  both search paths. The cation and anion valence totals differ by at most
  1.1 × 10⁻¹³ on the crystals and 1.8 × 10⁻¹² at 30 000 atoms.
- **Cutoff and running CN.** The distance CN at each automatic cutoff equals
  N_ab(r) at that grid point exactly, for 9 cation–O pairs.
- **Pair counts.** N_a n_ab(r) = N_b n_ba(r) holds to 1.2 × 10⁻¹⁰.
- **Scattering.** Below the first contact, G(r)/r = −4πρ₀ to 2.2 × 10⁻¹⁶ on four
  glasses for X-rays, neutrons and electrons. The Faber–Ziman weights sum to 1
  to 7 × 10⁻¹⁶, and at Q = 0 equal `pdf.scattering_weights` to 1.7 × 10⁻¹⁶.
- **Voronoi cells.** Cell volumes sum to the box volume to 4.1 × 10⁻¹⁶ on 20
  frames; every face appears from both sides with equal area, to
  7.6 × 10⁻¹⁴ Å²; the Delaunay counts satisfy E_D = N + T exactly.
- **One search per frame.** The driver makes exactly one pair search per frame
  per pass, counted on the search function itself.

### 5. Robustness

The readers were fuzzed with 11 489 truncations and 5 100 byte substitutions
over their 17 test files, each damaged file through both entry points and every
frame: nothing raised anything but `ValueError` or `OSError`. A second fuzz of
1 617 cuts and 1 200 substitutions covered the code added afterwards, with the
same result. Damaged copies of every binary fixture give no crash, and every
message names the file first.

| Input | What happens |
|---|---|
| A numeric-type dump with no type map | refused, naming the file, asking for an element for each type or the data file the run read |
| A frame with another atom count | skipped with its reason ("4 atoms; the first readable frame has 5") and counted |
| An empty file, a header-only dump, `NUMBER OF ATOMS 0`, a zero-length or inverted box | refused with a reason |
| A model that is not charge-neutral | a note in the composition, the provenance and every exported header |
| An element with no bond-valence parameter (Tc⁴⁺) | no valence, never zero; the gap is counted per atom and per pair and named (`Tc4+-O`: 6), as the crystal path counts it |
| An element with no scattering factor | refused; across Z = 1–118 no factor of 0 is accepted |
| An impossible or contradictory request: v_bond below v_list, a reversed window, a former absent from the model, r_max past half the box | every problem named at once, before any frame is read; 34 such cases gave no traceback, and the command line exits 2 |
| Missing, repeated or irregular frames given to dynamics | refused, naming the frame |
| An oxidation state of 10²⁰ | refused (it had turned a cation into an anion) |
| A search radius far beyond the box | refused with the estimated tree size (it had grown past 17 GB) |

### The Model window

The window was checked from outside its widgets, headless, driven as a user
drives it.

- **Every corpus file.** All 126 corpus files opened in a Model window: 81
  reach the setup, 44 the panel asking for the input the reader needs, and
  `ovito.nc` the reader's refusal in full, with no uncaught exception. Through
  the crystal window, 111 of the 115 entries open a Model window; the other 4,
  a one-frame extended XYZ and three single POSCARs, open as crystals, as
  `readers.read` intends. 30 files covering 28 formats reach the same state by
  File > Open, File > Open MD model…, a drop and the command line. Answering
  each read panel as a user would read 10 of 12; the other 2 were the typed-box
  fault below.
- **Against the engine called directly.** SiO₂, NS3, NAS and the NS3 melt,
  each with all 25 analyses run through the window: 270 of 270, 203 of 203, 364 of 364 and 221
  of 221 descriptors are bit-identical to `md_analysis.analyse` on an
  independently read trajectory (1058 in all, every per-frame column), with the
  same g(r) minima, searches, notes and failures.
- **Against numbers from outside FACET.** On NBS, the Qⁿ, N₄ (B3 0.663958, B4
  0.336042) and the MSD at every lag, recomputed from the raw dump with numpy
  and scipy, equal the window's export to 2.6 × 10⁻¹⁵. On the NS3 melt, the MSD
  at 19.8 ps is Na 223.9644, Si 18.35853 and O 22.86761 Å² against LAMMPS's
  `compute msd` 223.964, 18.3586 and 22.8676, and the kinetic temperature
  2976.4 ± 36.5 K against LAMMPS's 2978.8 ± 35.2 K.
- **Against the command line.** One request from the window and from
  `py -3.11 -m facet.md`: 163 descriptor files, 278 040 numeric cells,
  bit-identical. Only the timestamps and the order of radiations in the
  provenance differ.
- **Exports.** The CSV folder read back 1 951 662 values bit for bit, and the
  workbook the same values within 10⁻¹⁵ relative (openpyxl writes 16
  significant digits). Figures save as SVG of vector paths at 180 × 110 mm, as
  PNG of 4252 × 2598 px at 600 dpi, and as PDF.
- **Threshold panel.** At 9 slider positions on each of those models, CN and BVS equal
  `bulk.at_threshold` on the frame's table and a fresh `bulk.analyse_frame`
  search. At the run's v_bond, the panel's mean Na CN on NS3, 3.702, equals the
  run's own frame 0 value.
- **3D view.** The scene is built from the engine's bonds, and gives the same
  bond set, with bit-identical valences, as the crystal scene builder on the
  3000-atom SiO₂ and NS3 frames and five test frames.
- **Threads.** A profile hook on every worker thread recorded no Qt widget call
  off the GUI thread during open, run, frame load, threshold search and both
  exports, and every window slot ran on the GUI thread.

---

### What the verification found

Each fault below was found after the code's own tests passed, by a check
outside that code: another program, a closed form, the input's own statement of
itself, a fresh clone, or the window driven as a user drives it.

**Titanium was filed under thallium, and hydrogen at +3.** The fault was in the
data FACET already held for crystals. Its transcription of Brese and O'Keeffe
(1991) Table 2 held Ti(IV) under a Tl(IV) row the paper does not have, had no
Ti(III), and held H(I) as H(III). Every titanium and hydrogen valence therefore
came from the estimator instead of the table:

| Bond | FACET used (Å) | The paper prints (Å) |
|---|---|---|
| Ti–O | 1.8144 | 1.815 for Ti(IV), 1.791 for Ti(III) |
| H–O | 0.9388 | 0.95 |

The MD engine's verification found it, when a Ti-bearing model came back
flagged as estimated. The paper's text layer reads Ti and Tl alike, and the
existing page test asked only whether each printed number appears somewhere on
the page; a relabelled row keeps its numbers, so it passed. Two transcriptions
of the whole table were then made blind, from rendered images of the page. They
agree on all 109 rows, 327 values and 48 italic marks; against them, FACET's
table differed in the three row labels and in five italic marks, and in no
printed number. The page test now compares the rows one by one. Over 282 CIFs,
no oxidation state or coordination number moved, and 29 sites in 8 files
changed their BVS: Ti(IV) sites by +0.006 to +0.007 v.u., H(I) sites by up to
+0.039, and the oxygens around them by up to +0.059.

**A fresh clone failed 16 tests that passed here.** This machine has
`core.autocrlf=true`, so a Windows checkout turned every LF in `tests/data` into
CRLF, which broke the tests that cut a file at a known byte; a copy meant to be
CRLF came out as CR CR LF, which no writer produces. The other 726 tests passed
in the same clone. `tests/data/**` is now marked `-text` in `.gitattributes`,
and those tests start from LF whatever the checkout did; either change alone is
enough.

**Files read silently, or read with other numbers.** Before the corpus work,
FACET read 84 of the 115 files exactly and gave no other number for any of
them. Six others were cut to one frame without a note, raised a bare error, or
were not recognised:

- a five-model PDB came back as MODEL 1 alone, a 270-atom crystal, with no
  note;
- an XDATCAR saved as `.vasp` gave its first configuration as a crystal;
- a dump saved as `.xyz` raised a bare `ValueError`;
- an XDATCAR whose header ran past 4 kB was not recognised;
- a DL_POLY CONFIG not named CONFIG was not recognised;
- a type map keyed by the file's own labels was ignored.

The readers' verifiers found four more:

- LAMMPS's own `dump extxyz` in units real gave times 1000 times LAMMPS's own
  and velocities a thousandth of them;
- force-field labels became elements by their letters: ClayFF `ho` and DL_POLY
  `HO` became holmium, `CA` calcium, and a file mass 164 amu from the element
  was only noted;
- a file cut inside its last number was read with the shortened value;
- a LAMMPS data file with 7 columns was taken as `atom_style full`, which gave
  every atom another type and charge.

Each of these is now read, or refused with its reason.

**One writer's own fault.** OVITO 3.16.1 writes a tilted cell to NetCDF with its
xy and yz tilts exchanged in the angles, while the lengths are the cell's own,
so the file's lengths and angles describe a cell OVITO never held. FACET refuses
such a file and says to export a LAMMPS dump or an extended XYZ instead.

**Measurements that came out differently on real models.**

- **The automatic first minimum of g(r) landed on noise.** The rule now
  compares points with their Poisson counting errors, with a margin of 2
  standard errors. Several checks found the same fault independently:

  | Pair and data | Before | Now | LAMMPS |
  |---|---|---|---|
  | NS3 Na–O, 20 frames | 2.77 Å | 3.26 Å, floor 2.95–3.58 | 3.23 Å |
  | NS3, frame 19 alone | Na–O 2.15 Å, Si–O 1.80 Å, mean Na CN 0.14 | 3.24 Å, 2.42 Å, mean Na CN 5.55 | — |
  | NAS Al–O, frames 17 and 19 alone | 1.68 and 1.89 Å | 2.64 and 2.59 Å | — |
  | Cryolite Na–F, 2×2×2 | 2.24 Å, between two distances of one shell | 2.46 Å | — |

  On all nine cation–anion pairs of the four glasses, the LAMMPS minimum now
  lies inside the floor FACET reports. The EXAFS shell limits had the same
  fault: Si–Si N came out as 1.97 or 2.35 instead of 4.04.
- **FEFF χ(k) was averaged over more paths than FEFF sums.** It included paths
  FEFF leaves out of its own `chi.dat`, which put it 8.1 % from `chi.dat` (Al
  mean). It now uses the paths `chi.dat` lists and is 0.42–1.95 % from it.
- **The exported g(r) ended at about half its value.** The search stopped at
  r_max, so the last grid point lacked the pairs just beyond it: for Al–O it read
  0.53 where searching one step further gives 1.14.
- **Six elements were scattered as other elements.** He, Ne, Kr, Cm, Bk and Cf
  were given the factors of H, N, K, C, B and C, because the symbol went through
  `elements.normalise`. The factors now come from gemmi for the exact symbol.
- **Nothing stopped Q above π/dr.** There the sine route returns a mirror image
  of lower Q. Terminated at 60 Å⁻¹ on a 0.1 Å grid, G(r)'s first peak came out
  at 16.67 where the untruncated value on that grid is 8.34. Such a grid is now
  refused.
- **The vertex symbol was not the published one.** For quartz FACET gave King's
  shortest cycles, 6_1.6_1.6_2.6_2.8_9.8_9, where RCSR gives
  6.6.6(2).6(2).8(7).8(7). It now comes from primitive rings in RCSR notation.
- **Voronoi cells dropped real faces** while their edges still counted, which
  broke Euler's law. The index now equals OVITO's for every atom.
- **The probe-occupiable free volume read low.** On a real glass at the usual
  grid spacings it was 0.06 to 0.09 under the exact value (NS3: 0.656 against
  0.74258), and a synthetic test box gave 0.512 against 0.690.
- **Two dynamics inputs were accepted without a check:** wrapped positions
  given as tracks, which gave an MSD of 42.61 Å² where the continuous positions
  give 16.89 Å², and a frame interval that ignored the file's own timesteps, so
  a missing frame or a restart went unnoticed. Both are now refused.
- **A used-up pair generator gave CN 0.** Passed to `distance_cn`, it returned
  CN 0 for every atom, with no note. It is now refused.

**The Model window held on to what it was given, and under load it hung.**

- A closed window was never freed: two menu actions were connected to lambdas
  that held the window. Each close left 844 widgets and about 27 MB behind;
  resident memory went from 136 to 244 MB over five windows.
- A worker that met an exception it did not expect never stopped its thread.
  The status stayed at "Writing …", closing the window blocked for 60.08 s, and
  the process ended with exit code 127.
- Worker objects were never deleted, because their deletion was posted to a
  thread that had already ended. After three windows, 18 workers were alive,
  holding three trajectories and three results.
- The first fix deleted each worker on its own thread, which deadlocked against
  the GUI thread: a hung run caught with py-spy showed each thread waiting for a
  lock the other held. A second path released a worker before waiting for its
  thread, and aborted on a double delete. Under load, about 15 runs gave 3 hangs
  and 2 aborts.

Workers are now deleted on the GUI thread once their thread has ended, and every
worker catches every exception and stops its thread in a `finally` clause.
Under the same load, 8 of 8 runs pass; closing during a stuck export takes
0.009 s, and during a stuck threshold search 0.014 s.

The window's other faults included:

- a box typed as three rows never read, and the window dead-ended on a numpy
  error;
- a dump in LAMMPS real units that does not say so had its velocities read as
  metal units, so the kinetic temperature came out as 0.000297 K against
  296.70 K with the units stated, and the window had no way to state them. The
  setup now shows the units assumed and reads the file again with others;
- the kinetic temperature was offered on a dump with no velocities, and failed
  only after Run;
- dynamics on the NS3 melt at a stride of 10 frames was refused after a 395 s
  run; the refusal now comes in 0.22–0.26 s, before any frame is analysed;
- the MSD figure did not show what the engine's note said: on NBS at 19 ps the
  centre of mass's own MSD, 0.514324 Å² (recomputed from the dump's unwrapped
  columns), is 0.69 (Na) to 0.91 (Si) of each element's MSD. The figure now
  carries the note.

**Dark text on a dark page, found by the built application.** Every check
above ran offscreen, where Qt's palette is never dark. The built application,
started on a Windows desktop set to dark mode for apps, drew the Model window's
opening page in (23, 26, 31) on (30, 30, 30): a contrast of 1.05:1. A scroll
area fills its page and its viewport with the system's window colour, and an
application style sheet hands a widget no palette from its parent, so the
light theme's dark text landed on Windows' dark grey. The crystal window had
the same fault on six tabs (1.03–1.05:1), unseen until then. Measured from the
drawn pixels after the fix, on the real platform with Windows dark, Windows
light and forced palettes: body text 15.44:1 on every page of both windows,
secondary text 4.53:1 or more. The same pass found ticked check boxes drawn as
a white tick on the panel at 1.11:1, which a ticked box now fills with the
theme's highlight colour. The window also opened 1057 px tall where the area
above the taskbar is 1027 px; it is now sized within the screen's available
area, frame included.

**Performance.**

- **The image search held 125 copies of every atom.** It now keeps only the
  images that can reach the box. On 29 791 sheared atoms the tree went from 3.72
  million points to 50 653 and the search from 3.03 s to 1.56 s; on about
  100 000 atoms, memory went from 692 MB to 269 MB.
- **The primitive ring search held every candidate cycle to the end.** Memory
  grew by 4.2 MB per node, which extrapolates to 12.6 GB for a 3000-atom NS3
  frame with its Na–O bonds. It now rises by at most 60 MB.
- **One histogram had 830 000 bins.** A polyhedron of nearly zero volume had a
  quadratic elongation near 830, and the workbook export took 374 s. Histograms
  are now capped at 2000 bins, with a note, and values outside the edges are
  counted.

**How the fixes were checked.** Each module was reviewed by verifiers who had
not written it: three for the engine, the readers, each analysis module and the
Model window; two for each format module and for the driver. Each finding was
fixed with a regression test, and each test was checked by undoing its fix in a
scratch copy. For the driver alone that meant 28 breaks; two went unnoticed at
first, and their tests were strengthened until all 28 were caught. The engine,
the readers, each analysis module and the Model window were then read again
after their fixes, and every one of those reads found more:

| Module | Found on the second read |
|---|---|
| bulk engine | 1 |
| readers | 8 |
| glass | 4 |
| spectroscopy | 6 |
| scattering | 5 |
| network | 5 |
| order | 2 code defects, and 6 docstring statements that did not match the code |
| dynamics | 8 |
| Model window | 11: the deadlock and the abort above, and 9 in layout, labels and the text shown |

---

### What is not verified

**No model from the group has been analysed yet.** Every number above measured
on a glass model comes from the five models made for this verification, with
published potentials. The checks do not depend on the models resembling a
real glass, because each compares FACET with another code, a closed form or
itself on the same frames; whether a real glass looks like them was not
tested. Their potentials are not fully checked:

- the Pedone Si–O and Na–O rows were not checked against the 2006 paper, which
  is paywalled;
- the Pedone model's Coulomb term was summed by DSF, not by the paper's
  Ewald sum;
- SHIK's smoothing was applied to the short-range part only, because the
  paper's wording is ambiguous.

The dumps are deliberately LAMMPS's default six significant digits, in spatial
sort order, with image flags and a non-zero origin.

**What six-digit dumps cannot decide.** A few cation–O pairs lie within
1.7 × 10⁻⁴ Å of a cutoff, the worst-case rounding of a distance. Over the 20
frames:

| Cut | NS3 | NAS | NBS |
|---|---|---|---|
| FACET's cutoffs, Na–O | 8 | 6 | 3 |
| Bond-valence distances, Na–O | 20 | 5 | 9 |
| Bond-valence distances, Si–O | — | 1 | 2 |

These pairs could be bonded or not in the full-precision trajectory, which is at
most about 0.002 on a 20-frame mean Na CN.

**Bond-valence CN against an outside code.** Neither vitrum nor amorphouspy
computes one, so it was checked against them only through its exact equality
with a distance cut. That equality holds only when each element has one
oxidation state and each pair one parameter, which leaves two cases out: mixed
valence has no outside check on a glass, and pairs whose parameter comes from
the O'Keeffe–Brese estimator are exercised only through a stand-in in the tests.

**The automatic cutoff still misses rarely, and moves on one frame.** On a
seeded Poisson-sampled g(r) it lands on the falling flank for 1 seed in 200
with 20 frames and 4 in 200 with one frame; unsmoothed, the "first local
minimum" rule misses in 134 of 200. Where the valley floor is wide, a cutoff
taken from one frame differs from the 20-frame one: on the last NBS frame,
Na–O is cut at 2.99 Å from all 20 frames and at 3.26 Å from that frame alone,
and that frame's mean Na CN is 5.71 and 6.83. Every cutoff states its floor and
the CN range across it.

**Files not in the corpus.** LAMMPS per-processor dumps were not tested (the
LAMMPS build used has no MPI), nor output from LAMMPS versions other than
22 Jul 2025, nor data files from other builders (atomsk, Moltemplate, VMD
TopoTools, msi2lmp).

**Limits of the vitrum and amorphouspy comparison:**

- the NS3 melt was not used;
- amorphouspy has no public per-atom Qⁿ, so one was assembled from its own
  classify and neighbour functions; its public histograms match exactly;
- vitrum's g(r) was taken from `partial_pdf`, not from its `Scattering` class;
- the tools' frame averages were taken as means of per-frame values, as
  amorphouspy's `average_over_frames` does, without calling that function.

**The application side.**

- **The OpenGL tiers of the MD 3D view.** No OpenGL context can be created
  offscreen, so the Model window's 3D view was checked only on its QPainter
  tier, including the redraw with bonds that a frame above 5000 atoms gets once
  the view reports an OpenGL tier. The renderer itself was measured on its
  OpenGL tiers with frames of 3000 to 24 000 atoms built by the crystal scene
  builder; the Model window's path on a desktop with a GPU remains to be run
  once.
- **On screen.** Every Model window check above ran offscreen, and its
  screenshots were read. The only on-screen session is the launch of the
  built application below, which reached the window's first page.
- **The built application.** `tools/build_exe.py --app` was run on
  2026-10-07 on the working tree (PyInstaller 6.11.1, Python 3.11.9,
  numpy 2.4.6, PySide6 6.9.1) and exited with code 0. The module list read
  out of the built `FACET.exe` holds the 17 MD modules of the engine,
  `facet.md.cli` and the 7 modules of the Model window. Started on screen,
  once per file, it opened a Model window with no error box for three files:
  the SiO₂ model's LAMMPS dump, whose type-map page lists type 1 with 1000
  atoms and type 2 with 2000, the Si and O counts of the model's manifest;
  and a DCD and an XTC from `tests/data/md`, each asking for a topology file.
  No analysis was run in the built application.

**Open in the code:**

- **Core/shell models give other numbers.** A type map that sends the shells to
  O doubles the O count, 300 O for 150 O ions, and gives 159 Si–O pairs within
  2.0 Å against 79 for the cores alone, with no note.
- **Bi and Po masses.** A Po-209 mass (208.98243 amu) reads as Bi
  (208.9804 amu), 0.002 amu away; the mass tolerance is not yet settled. The
  same choice decides whether Se at 78.971 amu is read.
- **Boundaries and box-less XYZ.** There is no single rule yet for non-periodic
  boundaries across formats. A box-less multi-frame XYZ takes one box for every
  frame; in the deforming-box run that box differs from the true one by up to
  0.245 Å on frames 1–4.
- **No Q³ reference crystal.** Of the 47 CIFs in the reference collection that
  hold Si and O, none is an M_xSi₂O₅ with Si as the only former. The test skips,
  and no structure was made up in its place.
- **NMR and EXAFS.** No NMR shift correlation ships, and nucleus mass numbers
  are not checked; the optional cubic interpolation of FEFF paths is not done.
- **Symbol relabelling on the crystal side.** `elements.normalise` still
  relabels 15 symbols, He, Ne, Kr and Cm among them, for `pdf.py` and
  `diffraction.py`. Only the MD scattering module was changed.
- **Dynamics.** A lone O–O bond row is accepted. Six-digit `xu` columns cost
  about 5 × 10⁻⁷ Å² in the MSD, and nothing notes it. The jump check also
  applies to positions the file gives as unwrapped: on the NS3 melt at a stride
  of 10 frames, atom 2807 moves 0.343 of a box width between frames and every
  dynamics analysis is refused.
- **Cancel.** A run stops after the frame or analysis in progress, which took
  up to 57.4 s on SiO₂ on a loaded machine: free volume, Voronoi and empty
  spheres do not look for a cancel inside a frame.
- **Vacuum slabs in Voronoi** have no per-axis margin; a 60 Å gap ends at a
  91 Å margin and 2.4 GB of commit.
- **The Model window** saves a request file for the command line but does not
  load one. Provenance text prints some derived radii with float noise
  (17.310000000000002 Å).
- **Contrast of single strokes.** Three crystal-window labels measure below
  4.5:1 from the drawn pixels although their colours meet it: a table header
  "I" (3.11:1), a Planes label "l" (3.74:1) and an EXAFS header cut off at the
  table's edge (3.82:1). A lone vertical stroke never draws a pixel in its full
  colour on this screen; pure black "l" peaks at 9.90:1. The cutoff explorer's
  painted labels sit at about 3.2–3.5:1 by their colours and were not changed.

---

### Reproducing it

The permanent tests are in `tests/`: `test_md_*.py`, `test_bulk.py`,
`test_bulk_equivalence.py` (the crystal-as-glass equality) and `test_glass.py`,
with the Model window in `test_md_workspace.py`, `test_md_views.py`,
`test_md_routing.py` and `test_md_scene.py`. Their fixtures are in
`tests/data/md/` and `tests/data/crystals/`, the latter being the six example
CIFs with their attribution.

```
py -3.11 -m pytest tests/ -q
py -3.11 -m facet.md analyses
```

The second command lists every analysis, what it measures and the inputs it
needs.

The outside codes are not dependencies of FACET, and no test imports them. The
cross-checks ran as scripts in two throwaway environments:

- **Python 3.11:** LAMMPS 22 Jul 2025 (pip wheel), ASE 3.29, OVITO 3.16.1;
- **Python 3.12:** vitrum 1.1.0, amorphouspy 0.8.1, OVITO 3.16.0, ASE 3.29.0,
  matscipy 1.3.0, freud 3.6.1, pyscal3 4.0.0.

The vitrum and amorphouspy comparison runs FACET's side and the tools' side in
separate processes, then compares their saved output with a script that imports
neither. The five models were made by scripts whose manifest records, for each
model, the composition, type map and charges; the potential, its references and
its parameter source; the LAMMPS version and command; and a sha256 for every
file. The models, the corpus and these scripts are not in the repository.
