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
