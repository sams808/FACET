# Feature parity: CrystalMaker and VESTA

A tracked inventory rather than a claim. "Match CrystalMaker and VESTA" is on
the order of sixty features across ten areas, several of which (volumetric data,
structure editing, diffraction simulation) are projects in themselves. This
file records every one of them, what FACET does today, and what is deliberately
out of scope — so that what is missing is visible rather than discovered.

Legend: **done** · **partial** — usable but incomplete · **planned** — accepted,
not started · **out of scope** — with a reason.

Last updated after the analysis of glass models from molecular dynamics
(§14): readers for the files MD programs write, a per-atom engine for whole
frames, the descriptors glass work reports, network and local order,
scattering, NMR, EXAFS and dynamics, a Model window and a command line. Every
feature selected for implementation is done or partial, and each partial row
says what is missing; what remains planned is listed as such.

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
CrystalMaker or VESTA columns. The work is Phase MD in `ROADMAP.md`, and its
checks are the *MD models* section of `VERIFICATION.md`: 115 files written by
LAMMPS, ASE, OVITO and pymatgen, five models made with LAMMPS for the purpose
(four quenched glasses and a melt), and outside codes run on the same frames.
No model from the group's own work has been analysed yet.

### Reading a model

| Feature | FACET | Note |
|---|---|---|
| LAMMPS `data` file | **done** | atomic, charge and the other atom styles; `Masses`, `Velocities` and type labels; tilted and general triclinic boxes, with their origin |
| LAMMPS `dump`, text or gzip, one frame or a trajectory | **done** | scaled, unscaled and unwrapped coordinates, image flags, velocities and charges, in the column order of the `ITEM: ATOMS` header; element and type-label columns; units `metal` and `real`; triclinic boxes as the LAMMPS manual defines them |
| LAMMPS binary dump | **done** | the old and the new layout |
| LAMMPS YAML dump | **done** | read with the standard library |
| DCD | **done** | positions only: the elements come from the LAMMPS data file the run read; a DCD has no box origin |
| XTC | **done** | GROMACS's format, also LAMMPS's `dump xtc`; decoded with numpy from its published description; positions on the format's 0.01 Å grid |
| AtomEye extended CFG | **done** | one file per snapshot, read as a series |
| Extended XYZ, one frame or many | **done** | `Lattice=`, `Properties=`, `Origin=` and `Time=`; positions a writer never wrapped are kept as unwrapped. A single-frame XYZ dropped on the crystal window still opens as a crystal |
| Plain multi-frame XYZ, with the box given | **done** | LAMMPS `dump xyz`, CP2K `pos.xyz`: the box comes from another file of the run, a CP2K `.cell` file or three vectors; one box then holds every frame, which assumes a constant volume, and the notes say so |
| VASP XDATCAR | **done** | fixed and variable cell |
| A series of POSCAR files | **done** | one file per frame, as pymatgen and OVITO write them; a lone POSCAR opens as a crystal |
| DL_POLY `CONFIG` and `HISTORY` | **done** | the DL_POLY 4 layout; boundary keys 1–3, a box periodic along a, b and c, are read, and every other key is refused with its meaning named |
| ASE `.traj` | **done** | read with numpy, `struct` and `json`; ASE is not a dependency |
| HOOMD GSD | **done** | schema `hoomd` |
| AMBER NetCDF | **done** | CDF-1, CDF-2 and CDF-5, as LAMMPS `dump netcdf` and OVITO write it. A tilted cell from OVITO 3.16.1 is refused, because that exporter writes the cell angles with the xy and yz tilts exchanged; the message says to export a LAMMPS dump or an extended XYZ instead |
| CASTEP `.md` | **done** | positions, cell, velocities and time |
| GROMACS `.gro` | **done** | one or more frames |
| Animated XSF | **done** | a cell given once or at every step |
| Multi-model PDB | **done** | one frame per `MODEL`, or per `END` block as CP2K writes it; no longer read as its first model alone |
| IMD | **done** | the ASCII layout OVITO writes |
| Several files as one trajectory | **done** | a directory, a wildcard or a list; names taken in natural order, so `dump.20` comes before `dump.100` |
| A file recognised by its content, not its name | **done** | 114 of 115 files written by LAMMPS, ASE, OVITO and pymatgen are read to the precision of their format; the 115th is the tilted-cell NetCDF above, refused with its reason |
| Atom type → element, with its source recorded | **done** | the user's map, an `element` column, type labels, then `Masses` that match exactly one element within 0.01 amu; a whole-number mass, and a force-field label such as `HO`, need a map. No element is guessed |
| Oxidation states as model inputs | **done** | the common states, each one overridable, with its source recorded; never resolved from the geometry |
| Skipped frames counted and reported | **done** | truncated, another atom count, unreadable: each one is counted with its reason, never dropped |

### Measuring a frame

| Feature | FACET | Note |
|---|---|---|
| Per-atom bond-valence analysis of a whole frame | **done** | CN at the threshold, bond-valence sum, φ and plateau width for every atom, from one neighbour search per frame: 0.51 s for 9 261 atoms, where the crystal path took 26 s. The threshold moves without a new search, in 4–10 ms for 10 000 atoms. Atom for atom, what the crystal analysis gives, to 2.5 × 10⁻¹⁴ |
| Distance-cut CN beside the bond-valence CN | **done** | every bond-based descriptor computed from both bond definitions, with the per-atom cross-table of the two CN |
| Partial g(r) and running coordination N(r) | **done** | N(r) counted, not integrated; the first minimum found by a stated rule that judges g against its counting error, reported with its floor and the CN range across it; the user can override it |
| CN distribution per element | **done** | cut by bond valence and by distance, shown side by side |
| Bridging, non-bridging, free and tricluster oxygen | **done** | counted against the network formers the user chooses; there is no default set |
| Qⁿ per former, Qⁿ(mX) and network connectivity | **done** | how many bridges each former has, and to which element they go, e.g. Q⁴(2Al); the connectivity measured on the model's bonds |
| N₄ for boron, Al CN 4/5/6 | **done** | any other CN is reported, never folded in |
| Linkages X–O–Y | **done** | every pair of cations bonded to one anion; Al–O–Al counted whatever the Al coordination |
| Halide environments | **done** | which cations each anion is bonded to, as a frequency table of environments such as F–Al₁Na₂ |
| Lone-pair cations | **done** | distributions of φ, CN and plateau width over every atom of an element: the link to the crystal work |
| Bond-angle distributions | **done** | T–O–T, O–T–O and any A–B–C, from the bonds the CN counts |
| Bond-length distributions | **done** | from the same bonds |
| Composition, charge and density of a model | **done** | oxide mol % for a basis the user gives; a model that is not neutral with the declared oxidation states is reported |
| Frame averaging | **done** | every descriptor as a mean and the sample standard deviation across frames, with the frames used; one frame gives no spread and says so; frame order changes no result |

### Network and local order

| Feature | FACET | Note |
|---|---|---|
| Ring statistics | **done** | King's, Guttman's and the primitive criterion, with the R.I.N.G.S. normalisations, on a node set the user chooses; a ring that winds through the periodic boundary is not counted; vertex symbols in RCSR notation. Out of scope in the first plan |
| Coordination sequences | **done** | the number of nodes k bonds away, shell by shell |
| Polyhedral corner, edge and face sharing | **done** | |
| Connected components | **done** | the dimensionality of each piece, whether it spans the box, and modifier clustering |
| Warren–Cowley chemical short-range order | **done** | α_ij on a distance graph |
| Bond-orientational order | **done** | Steinhardt q_l and w_l per atom, their averaged forms, and the global Q_l |
| Tetrahedral order | **done** | Errington–Debenedetti q_tet |
| Polyhedron distortion per atom | **done** | Baur's index, bond-angle variance, quadratic elongation, volume and ECoN, as the crystal side computes them |
| Voronoi cells | **done** | the periodic tessellation: volume, face count and Voronoi index |
| Empty spheres and free volume | **done** | the empty sphere of every Delaunay tetrahedron; geometric, probe-centre and probe-occupiable free volume for a probe radius and a set of atomic radii the user states |

### Against experiment

| Feature | FACET | Note |
|---|---|---|
| X-ray, neutron and electron S(Q), F(Q) and G(r) from a model | **done** | Faber–Ziman partials by sine transform; D(r) and T(r) in Keen's conventions; Bhatia–Thornton; the first sharp diffraction peak; S(q) on the reciprocal lattice of the box as a second route. An element with no scattering length or form factor is refused, never set to zero |
| A measured curve overlaid | **done** | one scale factor and Wright's R_χ, its definition carried in every result |
| Model fractions beside NMR fractions | **done** | Qⁿ, N₄, Al CN and oxygen speciation beside fractions the user enters from NMR fits, with the difference; the spread across frames and the measured uncertainty are listed side by side, never combined |
| NMR spectra from structure–shift correlations | **done** | isotropic shifts and a broadened spectrum from a published correlation the user supplies with its reference; none ships with FACET, and nothing is computed without one. Out of scope in the first plan |
| Quadrupolar NMR lineshapes | out of scope | each atom gives one line at its isotropic shift; a second-order quadrupolar shape needs the electric-field gradient at each nucleus, which FACET does not compute |
| EXAFS shells from a model | **done** | absorber-centred g(r), first-shell limits from a measured minimum, and the cumulants N, R, σ², C₃ and C₄ of any shell |
| FEFF over a trajectory | **done** | `feff.inp` for absorbers drawn across frames; FEFF's results read back and χ(k) averaged, within 0.42–1.95 % of FEFF's own `chi.dat` per cluster |

### Dynamics

| Feature | FACET | Note |
|---|---|---|
| Mean-square displacement and diffusion coefficients | **done** | every time origin; diffusion over a stated window. The time axis comes from the file or from a timestep the user gives; irregular or missing frames are refused with the frame named, not interpolated over |
| Non-Gaussian parameter, self van Hove and self intermediate scattering functions | **done** | |
| Distinct van Hove function | **done** | |
| Velocity autocorrelation and vibrational density of states | **done** | with Green–Kubo diffusion; velocities from the file, or by finite difference when asked |
| Kinetic temperature | **done** | from the file's velocities |
| Ionic conductivity and Haven ratio | **done** | Nernst–Einstein and collective; the charges and the temperature are required inputs |
| Bond and coordination lifetimes | **done** | continuous and intermittent correlation functions, and residence times |

### Running, viewing and exporting

| Feature | FACET | Note |
|---|---|---|
| Every analysis from one request | **done** | 25 analyses, with one neighbour search per frame shared between them; frames streamed, never held together; every missing or contradictory input named before any frame is read. 7.5–9.7 s per frame with all 25 on models of 2 880–3 000 atoms, 0.35 s with the glass analysis alone |
| Command line, `python -m facet.md` | **done** | `describe`, `analyses`, `template` and `analyse`, with no Qt; a request file holds every option; the exit code says what happened |
| CSV and XLSX export with provenance | **done** | one file or sheet per descriptor, the unit in every column name, numbers in full; each headed by the file, the frames used and skipped, the type map and its source, oxidation states, parameter set, thresholds, g(r) minima, method parameters and FACET version. The workbook needs openpyxl and is refused without it |
| Model workspace in the main window | **done** | a model is an entry in the Structures dock beside the crystals; choosing one switches the Sites dock to an Elements table, the toolbar to frame navigation and a 3D-view/figure switch, the bottom strip to the coordination-against-threshold staircase, and the right column to four tabs (Setup, Results, Highlight, Notes). Opened by dropping the file or by *File ▸ Open MD model…*; asks for a type map, topology or box only when the reader needs one, and shows the units it assumed, with a way to read the file again with others; runs in the background with progress and cancel |
| Presets | **done** | silicate, aluminosilicate, borosilicate, phosphate, oxyfluoride, ion conduction, scattering against experiment, dynamics, everything; a preset ticks the analyses, ticks the network formers among the cations present and fills the method inputs those analyses read, and states what the model cannot give; any edit afterwards leaves the combo on Custom |
| Analyses grouped by question | **done** | structure, voids and channels, comparison with experiment, dynamics; each row states the inputs it will use, or, greyed, what it still needs, linked to the field that would supply it |
| Highlight rules on the 3D view | **done** | colour by element, bond-valence sum, coordination number, φ, Qⁿ, modifier-rich anions or channel membership; keep only the atoms a rule matches (element, CN, CN to formers, BVS, φ, Qⁿ); draw voids above a volume and an elongation, and channels, as translucent surfaces; dim the rest fades and shrinks what is left. The rules edit the scene's arrays, so the no-GPU tier and the vector export draw them |
| Channels by charge | **done** | the bond-valence landscape of a probe ion on a grid, its mismatch against the formal valence, the accessible volume fraction against the mismatch, the connected regions at one mismatch with volume, elongation and dimensionality, and the mismatch at which a path first crosses the box along each axis, by bisection over the grid's own values. An optional exclusion radius about the cations stands in for the repulsion term of a softBV model, and is stated |
| Channels by modifier density | **done** | modifier cations within the measured first minimum of each anion's partial g(r), the anions rich at a count the user sets, their clusters with the modifiers and whether those span the box; crossed with the speciation when the glass analysis runs |
| Void regions | **done** | the empty spheres of the Delaunay tetrahedra joined where a probe of a stated radius passes between them: union volume, elongation, extent, dimensionality, percolation and the atoms lining each region |
| Results as figures beside their rows | **done** | the spread across frames drawn on every figure; SVG, PDF or 600 dpi PNG, with colour never the only cue |
| Threshold slider on one frame | **done** | moves v_bond without a new search; the staircase of mean CN against threshold, clicked to set it |
| 3D view of one frame | **partial** | drawn from the bulk engine's bonds, which equal the crystal scene builder's on 3 000-atom SiO₂ and NS3 frames and five small test frames; no polyhedra or slab; above 5 000 atoms the QPainter tier draws the atoms without bonds. Looked at offscreen on the QPainter tier; the OpenGL tiers are not yet checked with an MD frame on a graphics card |
| Request file in the Model workspace | **done** | the workspace saves its setup as a request file that `python -m facet.md analyse --request` runs, and loads one back |
| Running, building or editing molecular dynamics | out of scope | FACET reads MD output; producing it is a different program |

## What is left

181 of the 203 tracked features are done, 12 are deliberately out of scope,
and 10 remain: 6 partial and 4 planned. Sections 1–13 hold 131 of the rows
(113 done, 4 partial, 4 planned, 10 out of scope); §14 holds 72 (68 done,
2 partial, 2 out of scope). Every feature selected for implementation is
implemented, the six partial ones in part; the rest are listed so that what is
missing stays visible rather than being discovered. (Counted from the tables,
one row per feature.)

**Partial — usable, incomplete**

- *Wireframe* draws as thin sticks rather than as true lines. Cosmetic.
- *Boundary modes*: periodic images are drawn where they close a bond, but there
  is no whole-molecule or whole-polyhedron completion mode.
- *Void analysis* reports and now draws the void cone half-angle, which is what
  a lone pair needs, but does not map cavities.
- *CrystalMaker files*: `.cmtx` text is read. `.cmdf` is an undocumented binary
  and is refused with an explanation rather than guessed at.
- *3D view of an MD frame* (§14): drawn on every tier from the bulk engine's
  bonds, without polyhedra or a slab, and on the QPainter tier above 5 000
  atoms without bonds. The OpenGL tiers have not yet been checked drawing an MD
  frame on a machine with a graphics card.
- *Request file in the Model window* (§14): the window writes one and the
  command line runs it; the window does not yet load one back.

**Planned**

- *Symmetry tools*: the operator list, transforming to another setting or cell,
  and searching for higher symmetry. Not selected for this round.
- *Scripting and batch*: the Library workspace, for running an analysis over a
  folder and comparing the results. The MD command line (§14) runs over the
  frames of one model, not over a folder of crystals.

**Out of scope, with reasons**

Structure editing, running molecular dynamics, animation and movie export.
FACET reads and measures structures, including the output of a
molecular-dynamics run (§14); it does not build, run or edit one. Building and
evolving structures are different programs, and half of one of them is worse
than none. Quadrupolar NMR lineshapes are out of scope too: a lineshape needs
the electric-field gradient at each nucleus, which FACET does not compute.
Ring statistics and NMR spectra from published correlations, out of scope in
the first plan for the glass work, are done.

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
