# Feature parity: CrystalMaker and VESTA

A tracked inventory rather than a claim. "Match CrystalMaker and VESTA" is on
the order of sixty features across ten areas, several of which (volumetric data,
structure editing, diffraction simulation) are projects in themselves. This
file records every one of them, what FACET does today, and what is deliberately
out of scope — so that what is missing is visible rather than discovered.

Legend: **done** · **partial** — usable but incomplete · **planned** — accepted,
not started · **out of scope** — with a reason.

Last updated after adding the pair distribution function, the EXAFS shell
and resolution report, the bond-valence vector overlay, the white default theme
and the in-application manual. Every feature selected for implementation is
done; what remains planned is listed as such. Section 14, the analysis of glass
models from molecular dynamics, is accepted and not yet started.

---

## 1. Display styles

| Feature | CM | VESTA | FACET | Note |
|---|:--:|:--:|---|---|
| Ball-and-stick | ✓ | ✓ | **done** | |
| Space-filling | ✓ | ✓ | **done** | van der Waals radii |
| Stick / licorice | ✓ | ✓ | **done** | |
| Wireframe | ✓ | ✓ | **partial** | draws as thin sticks; no true line mode |
| Polyhedral | ✓ | ✓ | **done** | per site, or every cation at once |
| Mixed polyhedral + ball-and-stick | ✓ | ✓ | **done** | the default when a site is selected |
| Thermal ellipsoids | ✓ | ✓ | **done** | any probability from 10 % to 99 %, with octant boundaries; reported per site on the File tab |
| Dot-surface / mesh | ✓ | ✓ | out of scope | superseded by ambient occlusion for depth reading |
| Per-atom style overrides | ✓ | ✓ | **done** | colour, size, visibility and label, per single atom; right-click it |

## 2. Rendering quality

| Feature | CM | VESTA | FACET | Note |
|---|:--:|:--:|---|---|
| Depth cueing | ✓ | – | **done** | adjustable, off by default in the print theme |
| Perspective / orthographic | ✓ | ✓ | **done** | |
| Adjustable lens (field of view) | ✓ | – | **done** | framing preserved when changed |
| Ambient occlusion | – | – | **done** | neither reference application has it |
| Silhouette outlines | – | – | **done** | |
| Analytic sphere silhouettes | – | – | **done** | ray-traced impostors; exact at any zoom |
| Order-independent transparency | – | – | **done** | triangles sorted back to front per view and cached against the view direction; per-buffer blending would need OpenGL 4.0, and this has to work with no graphics card |
| Stereo pairs / anaglyph | ✓ | ✓ | **done** | toe-in stereo; red-cyan, greyscale red-cyan, parallel and crossed pairs |
| Software rendering fallback | – | – | **done** | three tiers, to GL 3.0 and to pure QPainter |

## 3. Colour

| Feature | CM | VESTA | FACET | Note |
|---|:--:|:--:|---|---|
| Per-element colours | ✓ | ✓ | **done** | click a swatch |
| Multiple palettes | ✓ | ✓ | **done** | muted, Jmol/CPK, greyscale, high contrast |
| Colour-vision-safe palette | – | – | **done** | avoids the red/green pairing CPK relies on |
| Background, cell, label, selection colours | ✓ | ✓ | **done** | |
| Polyhedron colour and opacity | ✓ | ✓ | **done** | |
| Colour by site | ✓ | ✓ | **done** | |
| **Colour by bond-valence sum, CN, φ, valence discrepancy** | – | – | **done** | puts the analysis onto the structure |
| Pinned colour scales across structures | – | – | **done** | what makes two figures comparable |
| Save and share a theme | ✓ | ✓ | **done** | JSON |
| Per-site colour overrides | ✓ | ✓ | **done** | element, then site, then atom; only the fields actually set apply |

| Application-wide theming | ✓ | ✓ | **done** | menus, docks, panels and tables follow the theme; the interface roles are derived from it, so a custom background themes the whole window |
| White default, VESTA-like | – | ✓ | **done** | the shipped default; six presets under *View ▸ Theme* |

## 4. Structure display

| Feature | CM | VESTA | FACET | Note |
|---|:--:|:--:|---|---|
| Unit cell outline | ✓ | ✓ | **done** | every cell of a range, not just the box |
| Cell range / supercell | ✓ | ✓ | **done** | up to 6×6×6 |
| Atom labels | ✓ | ✓ | **done** | 11 kinds, de-cluttered, haloed |
| Bond labels / distances on screen | ✓ | ✓ | **done** | 6 kinds, including **bond valence per contact** |
| View down a crystallographic axis | ✓ | ✓ | **done** | a, b, c |
| Lattice planes, Miller planes | ✓ | ✓ | **done** | normal is h a\* + k b\* + l c\*, which is 55° from h a + k b + l c for hexagonal (111) |
| Slab / clipping plane | ✓ | ✓ | **done** | filters the built scene, so picking, labels and the counts agree with the picture |
| Boundary modes (whole molecules, packing) | ✓ | ✓ | **partial** | periodic images that close a bond are drawn |
| Show partial occupancy | – | ✓ | **done** | drawn as a smaller sphere |
| Disorder groups | – | ✓ | **done** | assembly and group read; one configuration chosen by default, and the overlap between alternatives reported |
| Multiple structures at once | ✓ | ✓ | **done** | load many or a whole folder; tick to show; overlay superimposed or laid out in a row |

## 5. Measurement and geometry

| Feature | CM | VESTA | FACET | Note |
|---|:--:|:--:|---|---|
| Click to measure distance | ✓ | ✓ | **done** | shift-click two atoms |
| Angle | ✓ | ✓ | **done** | three atoms |
| Torsion | ✓ | ✓ | **done** | four atoms |
| Bond-length table | ✓ | ✓ | **done** | per site, with valence per contact |
| Bond-angle table | ✓ | ✓ | **done** | Utilities ▸ Angles |
| Polyhedral distortion indices | ✓ | ✓ | **done** | Baur, bond-angle variance, quadratic elongation, volume |
| Effective coordination number | – | ✓ | **done** | Hoppe ECoN |
| Bond-valence sum | – | ✓ | **done** | with parameter provenance |
| Bond-valence map | – | ✓ | **done** | grid, isosurface in the 3D view, and 2D sections; the V = 3 surface passes within 0.1 Å of the real Bi sites |
| Uncertainty on the bond-valence sum | – | – | **done** | systematic (R0) and random (coordinates) reported separately |
| Structure health checks | – | – | **done** | impossible contacts, split sites, charge, esds, valence |
| Bond strain index | – | – | **done** | |
| Void / free-volume analysis | ✓ | ✓ | **partial** | the void cone half-angle; no cavity mapping |
| Centroid and eccentricity | ✓ | ✓ | **done** | |

## 6. Coordination analysis — where FACET goes past both

| Feature | CM | VESTA | FACET |
|---|:--:|:--:|---|
| Bond search by distance | ✓ | ✓ | **done** |
| **Bond search by partial bond valence** | – | – | **done** |
| **CN as a function of threshold (the staircase)** | – | – | **done** |
| **Plateau analysis — is this CN a result or a choice?** | – | – | **done** |
| **Sub-threshold contacts drawn, not hidden** | – | – | **done** |
| **Bond thickness proportional to bond valence** | – | – | **done** |
| Several CN definitions side by side | – | – | **done** | ten native rules plus eight pymatgen strategies, with the spread |
| φ, the scale-free stereoactivity index | – | – | **done** |
| Oxidation state by bond-valence self-consistency | – | – | **done** |
| **Anions analysed as sites, not only as ligands** | – | – | **done** | coordination number, contacts, plateau, φ and the anion's own bond-valence sum, counted from the cations around it |
| Parameter-set switching with citations | – | – | **done** | reads the IUCr bvparm distribution, softBV-style tables and its own JSON; per-pair b is kept, and every value carries its provenance |
| **Bond-valence vector drawn (the "lone pair")** | – | – | **done** | a lobe along −V/\|V\| with V = Σ v_i û_i, length φ × mean bond length × a display scale; φ = 0 draws nothing. Presented as the vector sum, which is what can be measured |
| **Void cone drawn** | – | – | **done** | the measured half-angle about the void axis, on the sites whose polyhedra are shown; the degeneracy of the axis is stated |

## 7. Symmetry

| Feature | CM | VESTA | FACET | Note |
|---|:--:|:--:|---|---|
| Space group from the file | ✓ | ✓ | **done** | |
| Independent symmetry determination | – | ✓ | **done** | spglib, cross-checked and reported when it disagrees |
| Wyckoff positions | ✓ | ✓ | **done** | |
| Site symmetry | ✓ | ✓ | **done** | |
| Symmetry operator list | ✓ | ✓ | **planned** | |
| Reciprocal cell, metric tensor | ✓ | ✓ | **done** | Utilities ▸ Cell |
| Density, composition, charge balance | ✓ | ✓ | **done** | atomic and weight per cent |
| d-spacing and hkl table | ✓ | ✓ | **done** | Utilities ▸ Reflections; geometry only, no intensities |
| Polyhedral connectivity | ✓ | ✓ | **done** | corner / edge / face, by shared ligands |
| Radial shells with degeneracies | – | – | **done** | Utilities ▸ Shells |
| Transform to another setting / cell | ✓ | ✓ | **planned** | |
| Search for higher symmetry | – | ✓ | **planned** | |

## 8. Volumetric data and surfaces

The core is in place: a periodic Grid type, marching-tetrahedra isosurfaces
verified against an analytic sphere, planar sections, and bond-valence maps.

| Feature | CM | VESTA | FACET |
|---|:--:|:--:|---|
| Import charge density (CHGCAR, CUBE, XSF) | – | ✓ | **done** |
| Isosurfaces | – | ✓ | **done** |
| 2D sections and contour maps | – | ✓ | **done** | marching squares with the saddle case resolved; colour map plus contours, on any Miller plane |
| Fourier / difference maps | – | ✓ | **done** | any loaded field, and the difference between two grids |
| Bond-valence energy landscapes | – | ✓ | **done** | the V=3 surface passes within 0.07 Å of the real Bi sites |
| Hirshfeld surfaces | – | – | out of scope | CrystalExplorer does this well |

## 9. Diffraction, PDF and EXAFS

| Feature | CM | VESTA | FACET |
|---|:--:|:--:|---|
| Powder pattern simulation | ✓ (CrystalDiffract) | ✓ | **done** | real form factors for X-rays, neutrons and electrons; pseudo-Voigt profile |
| d-spacing and hkl table | ✓ | ✓ | **done** (see §7) |
| Overlay on a measured pattern | ✓ | – | **done** | one least-squares scale factor and a difference curve; no fit quality is reported |
| **Pair distribution function G(r)** | – | – | **done** | R(r), G(r) and g(r) for X-rays, neutrons or electrons; peak areas as scattering-weighted coordination numbers; the element-pair weight table |
| **Finite-Q effects on the PDF** | – | – | **done** | Qmax truncation as a convolution in r over the odd extension, so the termination ripple appears where a measurement shows it; Qdamp as its dual; a Lorch window |
| **Correlated-motion peak widths** | – | – | **done** | σ² = U_i + U_j from the file, with the PDFgui δ₁/δ₂ parameterisation as explicit inputs defaulting to zero |
| **EXAFS shell table** | – | – | **done** | per shell: N, occupancy-weighted N, R, internal spread, contributing sites, and σ² from the file's U and from an Einstein model, labelled separately |
| **EXAFS resolution report** | – | – | **done** | ΔR = π/(2Δk), N_idp, which shells a k range cannot separate, and the parameter count against N_idp |
| **FEFF input** | – | – | **done** | runs in FEFF8L; the absorber's element gets its own scatterer potential, the cluster is larger than RPATH, and partial occupancy is stated rather than passed over |
| **FEFF output read back** | – | – | **done** | `files.dat` and `feffNNNN.dat` parsed; FEFF's degeneracies, path lengths, leg counts and amplitude ratios shown beside FACET's shells |
| **χ(k) from FEFF paths** | – | – | **done** | assembled from FEFF's amplitudes and phases with FACET's degeneracies and σ²; reproduces FEFF's own `chi.dat` to 2.5 % |
| χ(k) from geometry alone | – | – | out of scope | without phase shifts its transform peaks land ≈0.4 Å from the distances that produced them |
| XANES simulation | – | – | out of scope | needs full multiple scattering or DFT; FACET writes the input and does not substitute for it |
| Single-crystal / reciprocal lattice | ✓ (SingleCrystal) | – | out of scope |
| Electron diffraction | ✓ | – | out of scope |

## 10. File formats

| Format | Read | Write | Note |
|---|:--:|:--:|---|
| CIF | **done** | **done** | reads multi-block with esds; writes P1 with the analysis as comments |
| VASP POSCAR / CONTCAR | **done** | **done** | VASP 4 and 5 layouts, selective dynamics, negative scale |
| XYZ | **done** | **done** | extended XYZ; invents a box and says so when there is no `Lattice=` |
| VESTA `.vesta` | **done** | **done** | reads CELLP and STRUC; writes bond rules at the distance the valence threshold corresponds to |
| CrystalMaker | **partial** | – | `.cmtx` text is read; `.cmdf` is an undocumented binary and is refused with a way forward |
| SHELX `.res` / `.ins` | **done** | – | applies LATT and SYMM, so the analysis is not run on the asymmetric unit alone |
| PDB / mmCIF | **done** | – | via gemmi; invents a box when the file has no real cell |
| FEFF input | – | **done** | absorber first at the origin; distances asserted to match the analysis |
| CSV / XLSX of results | – | **done** | every file carries the parameter set and both thresholds |
| PNG image | – | **done** | supersampled, any resolution |
| Vector (SVG / PDF) | – | **done** | the QPainter tier doubles as the vector exporter: the 3D view, the pattern and the sections |

## 11. Editing

Not implemented, and a deliberate decision rather than an oversight: editing a
structure is a large surface, and FACET's purpose is to interrogate structures
that already exist. Revisit only if it is actually wanted.

| Feature | CM | VESTA | FACET |
|---|:--:|:--:|---|
| Add, move, delete atoms | ✓ | ✓ | out of scope for now |
| Edit cell parameters | ✓ | ✓ | out of scope for now |
| Build from scratch | ✓ | ✓ | out of scope for now |
| Override an oxidation state | – | – | **done** | the one edit that changes an answer |
| Override a bond-valence parameter | – | – | **done** | recorded as a user override |

## 12. Interaction

| Feature | CM | VESTA | FACET | Note |
|---|:--:|:--:|---|---|
| Rotate, pan, zoom | ✓ | ✓ | **done** | quaternion arcball |
| Click to select an atom | ✓ | ✓ | **done** | exact, by id buffer |
| Site list | ✓ | ✓ | **done** | |
| Drag and drop a file | ✓ | ✓ | **done** | |
| Right-click context menus | ✓ | ✓ | **done** | on an atom: select, polyhedron, and per-atom, per-site and per-element style |
| Undo | ✓ | – | **done** | snapshots of the presentation state rather than inverse operations |
| Session save and restore | ✓ | ✓ | **done** | paths, thresholds, theme, camera, labels |
| Animation, movies | ✓ | – | out of scope |
| Scripting / batch | – | – | **planned** | the Library workspace |

---

## 13. Documentation and provenance

| Feature | CM | VESTA | FACET |
|---|:--:|:--:|---|
| In-application manual | ✓ | ✓ | **done** | fourteen sections under *Help ▸ Manual*, compiled into the program so it cannot go missing from a shared folder; the thresholds and b in the text come from the code |
| About, with the running configuration | ✓ | ✓ | **done** | version, renderer tier actually in use, parameter set and its citation |
| Keyboard and mouse reference | ✓ | – | **done** | *Help ▸ Keyboard and mouse* |
| Licences and redistribution terms | – | – | **done** | *Help ▸ Licences*, including what the Qt LGPL requires of a build that is passed on |
| A written statement of what the program does not do | – | – | **done** | a manual section of its own, and `VERIFICATION.md` in the repository |

## 14. Glass models from molecular dynamics

FACET reads and measures the output of a molecular-dynamics run. It does not
build, run or edit one. This section is not a parity comparison, so it has no
CrystalMaker or VESTA columns. The order of the work is Phase MD in
`ROADMAP.md`. Nothing here has been checked against a real model yet.

| Feature | FACET | Note |
|---|---|---|
| LAMMPS `data` file | **planned** | atomic and charge atom styles; `Masses` and the box, including the `xy xz yz` tilt line |
| LAMMPS `dump`, one frame or a trajectory | **planned** | scaled, unscaled and unwrapped coordinates; column order taken from the `ITEM: ATOMS` header; triclinic boxes as the LAMMPS manual defines them |
| Multi-frame extended XYZ | **planned** | `Lattice=` and `Properties=`; single-frame XYZ reads as it does now |
| Atom type → element, with its source recorded | **planned** | a map from the user, then an `element` column, then `Masses`; an ambiguous mass match is refused |
| Skipped frames counted and reported | **planned** | truncated, a different atom count, unreadable: each one is counted, with its reason |
| Per-atom bond-valence analysis of a whole frame | **planned** | CN at the threshold, bond-valence sum, φ and plateau width for every atom, from one neighbour search per frame; the threshold moves without a new search. Oxidation states are model inputs, not resolved |
| Distance-cut CN beside the bond-valence CN | **planned** | cutoffs measured from the model, at the first minimum of the partial g(r) |
| Partial g(r) and N(r), coordination by integration | **planned** | first minimum found automatically, with the method stated; the user can override it |
| CN distribution per element | **planned** | cut by bond valence and by distance, shown side by side |
| Bridging, non-bridging, free and tricluster oxygen | **planned** | counted against a set of network formers that the user chooses |
| Qⁿ per former, and Qⁿ(mX) | **planned** | how many bridges each former has, and to which element they go, e.g. Si–O–Al |
| N₄ for boron, Al CN 4/5/6 | **planned** | fractions per frame, then averaged |
| Halide environments | **planned** | which cations each F or Cl is bonded to, as a frequency table of environments |
| Lone-pair cations | **planned** | distributions of φ, CN and plateau width over every Bi atom: the link to the crystal work |
| Bond-angle distributions | **planned** | T–O–T and O–T–O per former pair, with the same bond definition as the CN |
| Frame averaging | **planned** | every descriptor as a mean and a standard deviation across frames, with the number of frames used |
| Composition and charge of a model | **planned** | oxide mol % when the oxide basis is given; a model that is not neutral with the declared oxidation states is reported |
| X-ray and neutron G(r), S(q) and F(q) from a model | **planned** | the existing PDF machinery, averaged over frames, overlaid on a measured curve with an R-factor defined in the manual; an element with no scattering length is refused, not set to zero |
| Model fractions beside NMR fractions | **planned** | Qⁿ, N₄ and Al CN beside fractions the user enters from NMR fits, with the difference |
| Model workspace | **planned** | type map, frame range and stride, a computation that runs in the background with progress and cancel; histograms and distribution tables in place of the site table |
| CSV and XLSX export with provenance | **planned** | one sheet per descriptor; file, frames used, type map and its source, oxidation states, parameter set, threshold, g(r) minima and FACET version |
| Ring statistics | out of scope this round | a documented hook that names R.I.N.G.S. as a program that computes them |
| Simulated NMR spectra | out of scope this round | requires published shift correlations and their references before it computes anything |
| Running, building or editing molecular dynamics | out of scope | FACET reads MD output; producing it is a different program |

## What is left

113 of the 155 tracked features are done, 13 are deliberately out of scope,
and 29 remain: 4 partial and 25 planned, 21 of the planned ones being the glass
work of §14. Every feature selected for the crystal side is implemented; the
rest are listed so that what is missing stays visible rather than being
discovered. (Counted from the tables, one row per feature.)

**Partial — usable, incomplete**

- *Wireframe* draws as thin sticks rather than as true lines. Cosmetic.
- *Boundary modes*: periodic images are drawn where they close a bond, but there
  is no whole-molecule or whole-polyhedron completion mode.
- *Void analysis* reports and now draws the void cone half-angle, which is what
  a lone pair needs, but does not map cavities.
- *CrystalMaker files*: `.cmtx` text is read. `.cmdf` is an undocumented binary
  and is refused with an explanation rather than guessed at.

**Planned**

- *Symmetry tools*: the operator list, transforming to another setting or cell,
  and searching for higher symmetry. Not selected for this round.
- *Scripting and batch*: the Library workspace, for running an analysis over a
  folder and comparing the results.
- *Glass models from molecular dynamics* (§14): readers for LAMMPS and
  extended XYZ, a per-atom engine for whole frames, the descriptors glass work
  reports, and their comparison with scattering and NMR. Phase MD in
  `ROADMAP.md`.

**Out of scope, with reasons**

Structure editing, running molecular dynamics, animation and movie export.
FACET reads and measures structures, including the output of a
molecular-dynamics run (§14); it does not build, run or edit one. Building and
evolving structures are different programs, and half of one of them is worse
than none. Ring statistics and simulated NMR spectra are out of scope for the
first round of the glass work: the first is a program of its own, and the
second needs published shift correlations to stand on.

---

## How the numbers here were checked

Every quantity is verified against something outside the code that computes it,
because a program that only checks itself can be consistently wrong:

- **against independent implementations** — gemmi, spglib and pymatgen, over the
  whole Bi collection: cell volumes to 9e-16, d-spacings to 5e-16 by two routes,
  structure factors to 2e-4 of the strongest intensity, neighbour distances to
  exactly zero, and no disagreement with spglib on any space group;
- **against closed forms** — rocksalt intensities as 16(f_Na ± f_Cl)², contour
  circumferences as 2πr, cubic interplanar angles, the Debye-Waller identity;
- **against physical laws** — the cation and anion bond-valence totals are the
  same bonds counted from opposite ends by two separate searches, and agree to
  machine precision on 23 of 26 structures;
- **against invariance** — moving the origin, rotating the structure, relabelling
  the axes, building a supercell and choosing a different symmetry copy all leave
  every reported number unchanged to 2e-14;
- **against the file's own statements** — computed site multiplicities against
  `_atom_site_symmetry_multiplicity`, and the expanded cell against
  `_chemical_formula_sum`. That last check is what caught an expansion fault
  that had been losing three quarters of the oxygen in one structure while every
  test passed.
