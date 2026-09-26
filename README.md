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

The window opens on a white ground, and *View ▸ Theme* switches the whole
application — viewport, menus, panels and tables — between six presets, from a
greyscale one for print to two darker grounds for working at night.

**The bond-valence vector.** On request, a lobe on each cation along
−**V**/|**V**|, where **V** = Σ v_i **û**_i over the bonded contacts. That
direction is where the cation is *not* bonded, and for a cation with an ns²
configuration it is where a lone pair is conventionally described as pointing —
so this is what people mean when they ask to see the lone pair. FACET draws it as
the vector sum, because that is the thing that can be measured: its length is
φ × the mean bond length × a display scale, so a centrosymmetric site draws
nothing rather than being given an arbitrary direction. The measured void cone can
be drawn beside it.

**Bond valence, beyond one site.** The cutoff table: the distance each pair gets
from a stated valence threshold, which is this program's argument as numbers. A
threshold scan, showing how far the threshold can move before each coordination
number changes. Anion bond-valence sums, from their own neighbour search. A
charge balance that checks the two against each other. Bond-valence maps and
energy landscapes in 3D, and 2D sections through them.

**Diffraction.** Powder patterns with real scattering factors — X-ray, neutron and
electron — and a measured pattern overlaid, scaled by one least-squares factor
and nothing else.

**Pair distribution function.** G(r), g(r) and R(r) from every interatomic
distance in the structure. The area under a peak of R(r) is the
scattering-weighted coordination number of that shell, and it is reported for the
peak you select. Finite-Q effects are offered and are off until you set them:
truncating at Qmax is a convolution in r, applied over the odd extension so the
termination ripple appears below the first peak where a measurement shows it, and
the Q-resolution damping is its dual. The element-pair weight table is on screen,
because the useful fact about the PDF of a heavy-element compound is how little of
it is the light atoms — three quarters of the X-ray PDF of Bi₂O₃ is Bi–Bi, and
half of its neutron PDF is Bi–O.

**EXAFS.** The shell list a fit starts from: N, the occupancy-weighted N, R, the
spread inside each shell, the contributing sites, and σ² from two separately
labelled sources — the file's own displacement parameters, and an Einstein model
at a temperature you set. Beside it, what a k range can do with that list:
ΔR = π/(2Δk), the number of independent points, which shells are closer together
than ΔR, and how many parameters the list would need against how many the range
supports. FACET writes a FEFF input, does not bundle FEFF, and reads a
calculation back — FEFF's own degeneracies, path lengths and amplitude ratios
beside FACET's shells, and χ(k) assembled from FEFF's amplitudes with FACET's
geometry.

χ(k) is not computed from the structure alone, and XANES is not computed at all.
Both refusals are stated in the program, with the reason.

**Figures.** SVG and PDF for the 3D view, the diffraction pattern and the
sections. Vector, so they stay sharp at any size.

**Disorder.** Where a file describes alternative configurations, FACET reads the
assembly and group tags, reports how far apart the alternatives sit, and shows
one configuration rather than all of them at once — because drawing them together
puts atoms a fraction of an angstrom apart and makes every coordination number in
the structure wrong.

**Documentation.** A manual of fourteen sections under *Help*, compiled into the
program rather than read from a file beside the executable, so it cannot go
missing from a shared folder. One of its sections is what FACET does not do.

Nothing in the program states a verdict. It reports measurements, says where each
number came from, and leaves the reading of them to you.

## Status

Complete and verified. `FEATURES.md` tracks all 141 features against CrystalMaker
and VESTA: 118 done, 14 deliberately out of scope, 9 remaining and listed.

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
