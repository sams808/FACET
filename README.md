# FACET

Coordination analysis from crystal structure files, for people who do not write
code. Windows desktop application; no Python installation required.

## What it is for

A coordination number is not measured. It is produced by choosing a cutoff, and
for a cation with a stereochemically active lone pair there is no gap in the
contact distribution where a cutoff naturally falls. Published coordination
numbers for such sites therefore differ by two or three units between papers
describing the same structure, and the difference is a reporting artefact.

FACET sets cutoffs by **partial bond valence** rather than by distance, so every
anion is cut at the same minimum bond strength instead of at a distance chosen
for oxygen. It then reports not one coordination number but the whole function:
what coordination number each threshold produces, and how wide a range of
threshold each one survives.

A coordination number that holds across a wide range of threshold is a property
of the structure. One that occupies a narrow step is a property of whoever chose
the cutoff. FACET shows which it is.

Nothing in the design is specific to any element.

## The model

Bond valence is `v = exp((R0 - d) / b)`. A contact counts as a bond above a
threshold in valence units rather than within a distance:

| | threshold | Bi–O | Bi–I |
|---|---|---|---|
| bond | 0.075 v.u. | 3.05 Å | 3.72 Å |
| tabulated | 0.020 v.u. | 3.54 Å | 4.21 Å |

The Bi–O figures are the conventional literature cutoffs, recovered rather than
assumed. The Bi–I figures are what the same physical criterion gives for a much
larger anion, and no oxygen-derived convention would have produced them.

**The plateau identity.** Sorting a site's contacts by valence, the coordination
number is `k` for any threshold between the valence of the k-th and (k+1)-th
contact, and the width of that interval in log-valence is

```
ln(v_k / v_k+1) = (d_k+1 - d_k) / b
```

so a plateau width *is* a distance gap, divided by b. The two statements are the
same statement.

**Stereoactivity.** FACET reports `phi = |sum v_i u_i| / sum v_i`, the
bond-valence vector sum normalised by the bond-valence sum. It is dimensionless,
bounded 0 to 1, and exactly invariant to an error in R0 — shifting R0 by delta
scales every `v_i` by `exp(delta/b)`, which cancels. The unnormalised vector sum
has no such property, so only `phi` can be compared across anions or across
parameter sets.

## Parameter provenance

Every parameter says where it came from, and the distinction survives into every
exported report.

* **Fitted** — refined against experimental structures, quoted with its
  citation.
* **Estimated** — computed from the O'Keeffe & Brese electronegativity
  expression. Covers 75 elements including every lanthanide, and reproduces
  fitted values to about 0.03–0.08 Å, against the 0.02 Å a fitted value itself
  carries. Always labelled as estimated.

Oxidation states are resolved by bond-valence self-consistency where the file
does not state them, which is what separates the sites of a mixed-valence
structure. The route used is recorded per site.

## What else it does

The coordination analysis is the argument; these are the tools that go with it.

**Structures.** Reads CIF, VASP POSCAR/CONTCAR, XYZ, VESTA, SHELX .res/.ins, PDB
and CrystalMaker .cmtx. Several at once, shown together or one at a time. Writes
CIF, POSCAR, XYZ, VESTA, FEFF, CSV and XLSX.

**The 3D view.** Ball-and-stick, space-filling, stick and wireframe; coordination
polyhedra; lattice planes and slabs by Miller indices; per-atom, per-site and
per-element colour, size and visibility; labels for atoms, bonds, distances and
bond valences; stereo pairs and red-cyan anaglyph. It runs on a machine with no
graphics card — there are three render tiers, and the lowest is pure Qt.

**Bond valence, beyond one site.** The cutoff table: the distance each pair gets
from a stated valence threshold, which is this program's argument as numbers. A
threshold scan, showing how far the threshold can move before each coordination
number changes. Anion bond-valence sums, from their own neighbour search. A
charge balance that checks the two against each other. Bond-valence maps and
energy landscapes in 3D, and 2D sections through them.

**Diffraction.** Powder patterns with real scattering factors — X-ray, neutron and
electron — and a measured pattern overlaid, scaled by one least-squares factor
and nothing else.

**Figures.** SVG and PDF for the 3D view, the diffraction pattern and the
sections. Vector, so they stay sharp at any size.

**Disorder.** Where a file describes alternative configurations, FACET reads the
assembly and group tags, reports how far apart the alternatives sit, and shows
one configuration rather than all of them at once — because drawing them together
puts atoms a fraction of an angstrom apart and makes every coordination number in
the structure wrong.

Nothing in the program states a verdict. It reports measurements, says where each
number came from, and leaves the reading of them to you.

## Status

Complete and verified. `FEATURES.md` tracks all 120 features against CrystalMaker
and VESTA: 99 done, 10 deliberately out of scope, 11 remaining and listed.

The engine is a generalisation of the validated code behind a 2026 survey of
105 bismuth sites across 74 structures, which was itself checked by an
independent recomputation sharing no code. That survey's results are kept as a
regression fixture: FACET reproduces its bond distances, coordination numbers,
bond-valence sums and `phi` across 92 sites, with one deliberate and documented
departure.

## How it is checked

A program arguing that published numbers are unreliable has to be held to a
higher standard than the numbers it criticises, and "the tests pass" is not that
standard. Every quantity is verified against something outside the code that
produces it: independent implementations (gemmi, spglib, pymatgen), closed forms,
physical laws, invariance under every way of rewriting the same crystal, and the
file's own statement of what it contains.

That last check found an expansion fault that had been losing three quarters of
one structure's oxygen while every test passed. **`VERIFICATION.md`** records what
was checked, what the agreement was, and what was found.

## Running the tests

```
py -3.11 -m pytest tests/ -q
```

Tests that depend on the author's structure library skip cleanly when it is
absent.

## Building the application

```
py -3.11 tools/build_exe.py
```

Produces the application folder, an installer to send to a collaborator, and a
zip for anyone who would rather not run one. **`BUILDING.md`** covers what comes
out, what the installer does, the SmartScreen warning worth mentioning in the
message that carries it, and the checklist to work through before passing a build
on.

The output lands in `dist/`, which is not in the repository: the installer is over
GitHub's 100 MB per-file limit, and a repository carrying its own binaries is slow
to clone for no benefit. Binaries belong in a GitHub Release.

## Licence

MIT for FACET's own source. The built application bundles Qt under the LGPL,
which constrains how a binary may be redistributed — **read
`THIRD_PARTY_NOTICES.md` before passing a build on.** In short: build as
`onedir`, ship the `licenses/` directory, and Qt stays replaceable.
