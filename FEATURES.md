# Feature parity: CrystalMaker and VESTA

A tracked inventory rather than a claim. "Match CrystalMaker and VESTA" is on
the order of sixty features across ten areas, several of which (volumetric data,
structure editing, diffraction simulation) are projects in themselves. This
file records every one of them, what FACET does today, and what is deliberately
out of scope — so that what is missing is visible rather than discovered.

Legend: **done** · **partial** — usable but incomplete · **planned** — accepted,
not started · **out of scope** — with a reason.

Last updated at the end of Phase 4.

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
| Thermal ellipsoids | ✓ | ✓ | **planned** | ADPs are parsed; the ellipsoid geometry is not built |
| Dot-surface / mesh | ✓ | ✓ | out of scope | superseded by ambient occlusion for depth reading |
| Per-atom style overrides | ✓ | ✓ | **planned** | |

## 2. Rendering quality

| Feature | CM | VESTA | FACET | Note |
|---|:--:|:--:|---|---|
| Depth cueing | ✓ | – | **done** | adjustable, off by default in the print theme |
| Perspective / orthographic | ✓ | ✓ | **done** | |
| Adjustable lens (field of view) | ✓ | – | **done** | framing preserved when changed |
| Ambient occlusion | – | – | **done** | neither reference application has it |
| Silhouette outlines | – | – | **done** | |
| Analytic sphere silhouettes | – | – | **done** | ray-traced impostors; exact at any zoom |
| Order-independent transparency | – | – | **planned** | matters once many polyhedra overlap |
| Stereo pairs / anaglyph | ✓ | ✓ | **planned** | |
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
| Per-site colour overrides | ✓ | ✓ | **planned** | |

## 4. Structure display

| Feature | CM | VESTA | FACET | Note |
|---|:--:|:--:|---|---|
| Unit cell outline | ✓ | ✓ | **done** | every cell of a range, not just the box |
| Cell range / supercell | ✓ | ✓ | **done** | up to 6×6×6 |
| Atom labels | ✓ | ✓ | **done** | 11 kinds, de-cluttered, haloed |
| Bond labels / distances on screen | ✓ | ✓ | **done** | 6 kinds, including **bond valence per contact** |
| View down a crystallographic axis | ✓ | ✓ | **done** | a, b, c |
| Lattice planes, Miller planes | ✓ | ✓ | **planned** | |
| Slab / clipping plane | ✓ | ✓ | **planned** | |
| Boundary modes (whole molecules, packing) | ✓ | ✓ | **partial** | periodic images that close a bond are drawn |
| Show partial occupancy | – | ✓ | **done** | drawn as a smaller sphere |
| Disorder groups | – | ✓ | **planned** | |
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
| Several CN definitions side by side | – | – | **partial** | valence, ECoN, max-gap; Voronoi and CrystalNN still to add |
| φ, the scale-free stereoactivity index | – | – | **done** |
| Oxidation state by bond-valence self-consistency | – | – | **done** |
| Parameter-set switching with citations | – | – | **partial** | one fitted set plus the estimator; more to ship |

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

Nothing in this area is implemented. It is the largest single gap and is
accepted as a later phase.

| Feature | CM | VESTA | FACET |
|---|:--:|:--:|---|
| Import charge density (CHGCAR, CUBE, XSF) | – | ✓ | **planned** |
| Isosurfaces | – | ✓ | **planned** |
| 2D sections and contour maps | – | ✓ | **planned** |
| Fourier / difference maps | – | ✓ | **planned** |
| Bond-valence energy landscapes | – | ✓ | **planned** | a natural fit for this application |
| Hirshfeld surfaces | – | – | out of scope | CrystalExplorer does this well |

## 9. Diffraction

| Feature | CM | VESTA | FACET |
|---|:--:|:--:|---|
| Powder pattern simulation | ✓ (CrystalDiffract) | ✓ | **planned** |
| d-spacing and hkl table | ✓ | ✓ | **planned** |
| Overlay on a measured pattern | ✓ | – | **planned** |
| Single-crystal / reciprocal lattice | ✓ (SingleCrystal) | – | out of scope |
| Electron diffraction | ✓ | – | out of scope |

## 10. File formats

| Format | Read | Write | Note |
|---|:--:|:--:|---|
| CIF | **done** | **done** | reads multi-block with esds; writes P1 with the analysis as comments |
| VASP POSCAR / CONTCAR | **planned** | **done** | |
| XYZ | **planned** | **done** | extended XYZ with `Lattice=` |
| VESTA `.vesta` | **planned** | **done** | bond rules written at the distance the valence threshold corresponds to |
| CrystalMaker `.cmdf` | **planned** | – | format is not openly documented |
| SHELX `.res` / `.ins` | **planned** | – | |
| PDB / mmCIF | **planned** | – | gemmi already parses both |
| FEFF input | – | **done** | absorber first at the origin; distances asserted to match the analysis |
| CSV / XLSX of results | – | **done** | every file carries the parameter set and both thresholds |
| PNG image | – | **done** | supersampled, any resolution |
| Vector (SVG / PDF) | – | **planned** | needed for a publication figure |

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
| Right-click context menus | ✓ | ✓ | **planned** | |
| Undo | ✓ | – | **planned** | |
| Session save and restore | ✓ | ✓ | **done** | paths, thresholds, theme, camera, labels |
| Animation, movies | ✓ | – | out of scope |
| Scripting / batch | – | – | **planned** | the Library workspace |

---

## Where this leaves Phase 4 and beyond

Ordered by how much each would be used here, not by how hard it is:

1. **Export formats** — CIF, VASP, XYZ, VESTA, and CSV of results. Cheap, and it
   is what makes FACET fit alongside everything else in the group.
2. **The Library workspace** — batch analysis over a folder, structure health,
   deduplication. The original Phase 4.
3. **Remaining CN definitions** — Voronoi, CrystalNN, CHARDI, so the
   multi-definition panel is complete rather than indicative.
4. **Vector export** — SVG or PDF, for a figure that goes into a paper.
5. **Powder diffraction** — simulation and overlay.
6. **Lattice planes and slabs** — the main remaining display gap.
7. **Thermal ellipsoids** — the data is already parsed.
8. **Volumetric data and isosurfaces** — the largest gap, and the point at which
   bond-valence energy landscapes become possible.
