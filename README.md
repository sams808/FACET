# FACET

Coordination analysis from crystal structure files and from molecular-dynamics
models of glasses, for people who do not write code. Windows desktop
application; no Python installation required.

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
CIF, POSCAR, XYZ, VESTA, FEFF, CSV and XLSX. MD models and trajectories open in
a window of their own (see *Glass models from molecular dynamics* below).

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

## Glass models from molecular dynamics

FACET reads the models and trajectories that molecular-dynamics programs write,
and measures them. It runs no simulation, and builds or edits no model.

The argument carries over unchanged. The descriptors reported for a glass model
— coordination numbers, Qⁿ, N₄, bridging and non-bridging oxygen — all depend on
where a bond is cut. FACET cuts it two ways on the same atoms: by bond valence,
and by distance at the first minimum of the model's own partial g(r). On a sodium
trisilicate model made for the verification, the bond-valence threshold of
0.075 v.u. falls at 2.76 Å for Na–O and the g(r) minimum at 3.26 Å, and the mean
Na coordination number comes out 3.71 and 5.63. Both are reported, side by side.

An MD frame is not analysed as a crystal. It has thousands of atoms and no
symmetry, so it goes to an engine of its own, which searches each frame once for
every atom: 0.51 s for 9 261 atoms, where the crystal analysis took 26 s. Given a
crystal tiled into a supercell, that engine returns, atom for atom, what the
crystal analysis returns.

**Opening a model.** Drop the file on FACET's window, or use *File ▸ Open MD
model…*; it opens in a Model window of its own. *File ▸ Open MD series as one
model…* reads several files of one run as one trajectory. A file is recognised
by its content, so a renamed dump still opens as a dump. Where the file leaves
out something only the person who ran the simulation knows — which element a
LAMMPS type number stands for, the data file a DCD needs for its elements, the
box of a plain XYZ — the window shows the reader's message and asks for exactly
that. No element is guessed.

The setup then takes the oxidation states (the common ones, each editable, never
resolved from the geometry), the network formers (none ticked: which cations
form the network is a chemical decision, and FACET does not make it), the frames
(first, last and stride), the thresholds and the analyses. The run goes in the
background with progress and cancel. Each descriptor comes back as a figure
beside its rows, as the mean over frames with the spread across them; a
threshold slider moves the bond threshold on one frame without a new search; one
frame can be turned in 3D. *Help ▸ MD models and the Model window* is the
manual's section on all of it.

**From the command line.** `python -m facet.md` runs the same analyses with no
window, on a workstation or a cluster node. Unlike the application, it needs
Python and FACET's source.

```
py -3.11 -m facet.md describe dump.lammpstrj --type-map 1=Si,2=O,3=Na
py -3.11 -m facet.md analyse dump.lammpstrj --type-map 1=Si,2=O,3=Na \
    --formers Si --frames 0:100:5 --out results.xlsx
py -3.11 -m facet.md template > request.toml
```

`describe` says what the reader finds in a file, `analyses` lists every analysis
with the inputs it needs, and `template` writes a request file holding every
option, which `analyse --request` takes; the Model window can save its setup as
one. An input with no default — the network formers, a timestep, a temperature,
charges, an NMR correlation — is never filled in: a request that lacks one stops
before any frame is read, naming each one.

**Formats.** LAMMPS data files and dumps (text, gzip, binary and YAML), DCD, XTC
and AtomEye CFG; extended XYZ, and plain XYZ trajectories once a box is given
(another file of the run, a CP2K cell file or three vectors); VASP XDATCAR and
series of POSCAR files; DL_POLY CONFIG and HISTORY; ASE .traj; HOOMD GSD; AMBER
NetCDF; CASTEP .md; GROMACS .gro; animated XSF; multi-model PDB; IMD. None of
them needs the library its authors provide.

**What is measured.** Each structural analysis comes as a mean over the frames
chosen, with the spread across them; the dynamics follow the trajectory in time.

- *Glass:* CN by bond valence and by distance; bridging, non-bridging, free and
  tricluster oxygen; Qⁿ and Qⁿ(mX); N₄; Al CN; linkages; halide environments;
  φ and plateau widths; bond angles and lengths; partial g(r) and N(r) with
  their first minima; composition and density.
- *Scattering:* partial and total S(Q), F(Q) and G(r) for X-rays, neutrons and
  electrons; Bhatia–Thornton; the first sharp diffraction peak; a measured curve
  overlaid with one scale factor and R_χ.
- *Rings:* King's, Guttman's or the primitive criterion, with the R.I.N.G.S.
  normalisations.
- *Network:* coordination sequences; corner, edge and face sharing; connected
  pieces and whether they span the box; Warren–Cowley short-range order.
- *Local order:* Steinhardt and tetrahedral order; polyhedron distortion and
  ECoN; Voronoi cells.
- *Voids:* the empty sphere of every Delaunay tetrahedron; free volume for a
  probe radius.
- *NMR:* the model's Qⁿ, N₄, Al CN and speciation beside fractions from an NMR
  fit; shifts and spectra from a published correlation the user supplies.
- *EXAFS:* shell cumulants N, R, σ², C₃ and C₄; FEFF inputs for absorbers drawn
  across frames, and FEFF's results averaged.
- *Dynamics:* mean-square displacement and diffusion; van Hove functions and
  the self intermediate scattering function; velocity autocorrelation and the
  vibrational density of states; kinetic temperature; ionic conductivity and the
  Haven ratio; bond and coordination lifetimes.

The results go out as one XLSX workbook with a sheet per descriptor, or a folder
of CSV files, each headed by its provenance: the file, the frames used and
skipped, the type map and where it came from, the oxidation states, the
parameter set, the thresholds, the g(r) minima, every method parameter and the
FACET version. Figures save as SVG, PDF or 600 dpi PNG.

**What it does not do.** It does not run, build or edit a simulation. It fills in
no timestep, temperature, charge or network former, and does not resolve a
model's oxidation states from its geometry. It ships no NMR correlation and does
not simulate quadrupolar lineshapes. For dynamics it does not interpolate over a
missing or unevenly spaced frame: it refuses the run and names the frame. The
Model window does not yet load a saved request file, and its 3D view has not yet
been checked on the OpenGL tiers with a graphics card.

**How it was checked.** The *MD models* section of `VERIFICATION.md`: 115 files
written by LAMMPS, ASE, OVITO and pymatgen, five models made with LAMMPS for
the purpose (four quenched glasses and a melt), and outside codes run on the
same frames — LAMMPS itself, ASE, OVITO, vitrum, amorphouspy, matscipy,
pyscal3 and FEFF. No model from the group's own work has been analysed yet.

## Status

`FEATURES.md` tracks 203 features: 131 against CrystalMaker and VESTA, and 72 for
glass models from molecular dynamics. 181 are done, 12 deliberately out of
scope, and 10 remain and are listed: 6 partial and 4 planned.

The engine is a generalisation of the validated code behind a 2026 survey of
105 bismuth sites across 74 structures, which was itself checked by an
independent recomputation sharing no code. That survey's results are kept as a
regression fixture: FACET reproduces its bond distances, coordination numbers,
bond-valence sums and `phi` across 92 sites, with one deliberate and documented
departure.

## How it is checked

A program arguing that a published coordination number can be a property of the
cutoff rather than of the structure has to be held to a higher standard than the
numbers it questions, and "the tests pass" is not that standard. Every quantity
is verified against something outside the code that produces it: independent
implementations (gemmi, spglib, pymatgen; for MD models, LAMMPS itself, ASE,
OVITO, vitrum, amorphouspy, matscipy, pyscal3 and FEFF), closed forms, physical
laws, invariance under every way of rewriting the same crystal or model, and the
file's own statement of what it contains.

That last check found an expansion fault that had been losing three quarters of
one structure's oxygen while every test passed. The MD checks found a fault in
data the crystal side already held: Brese and O'Keeffe's titanium parameters had
been filed under thallium, and hydrogen's at +3, so every Ti and H bond valence
had come from the estimator. **`VERIFICATION.md`** records what was checked, what
the agreement was, and what was found; its *MD models* section covers the MD
side.

## Running the tests

```
py -3.11 -m pytest tests/ -q
```

The suite collected 3 208 tests on 2026-10-07. Tests that depend on the author's
structure library skip cleanly when it is absent.

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
