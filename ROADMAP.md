# Roadmap to CrystalMaker and VESTA parity

`FEATURES.md` is the inventory — what exists, what does not. This is the plan:
the order the remaining work is done in, why that order, and what each phase
delivers.

Ordered by **how much each would be used here**, not by how hard it is. A
feature that closes a gap in the bismuth work outranks one that merely ticks a
box on a comparison table.

Phases 1–3 are done. Each phase below ends with a runnable `.exe`, a green test
suite, and a commit.

---

## Where the effort actually goes

Counting from `FEATURES.md`: 62 features, of which 34 are done or partial. The
28 remaining are not evenly sized — roughly:

| | share of remaining work |
|---|---|
| Volumetric data, isosurfaces, bond-valence landscapes | ~35 % |
| Diffraction simulation | ~15 % |
| File formats, in and out | ~12 % |
| Library, batch and reporting | ~12 % |
| Remaining display (planes, slabs, ellipsoids) | ~10 % |
| Remaining analysis (CN definitions, parameter sets) | ~8 % |
| Interaction polish (undo, sessions, context menus) | ~8 % |

Volumetric data is a third of what is left and is the only part that needs
machinery FACET does not already have. Everything before it is incremental.

---

## Phase 4 — Interchange *(next)*

**Why first:** it is cheap, and it is what makes FACET fit alongside everything
else in the group rather than being a dead end. Right now an analysis can only
leave as a PNG.

- **Read**: VASP POSCAR/CONTCAR, XYZ, SHELX `.res`/`.ins`, PDB/mmCIF (gemmi
  already parses the last two), `.vesta`
- **Write**: CIF (with the analysis as comments), POSCAR, XYZ, `.vesta` so a
  collaborator can open the same view in VESTA
- **Results out**: CSV and XLSX of the site table, the contact table and the
  plateau table — the thing that actually goes into a paper's SI
- **FEFF input** for the Bi L₃ EXAFS work, with the radial shells already computed
- **Vector export**: SVG of the cutoff explorer, and of the 3D view via a
  painter-based path, because a raster figure in a paper is a compromise
- Session save and restore: structure, camera, theme, threshold, labels

## Phase 5 — The Library workspace

**Why:** the original Phase 4 of the plan, and the answer to "I downloaded 40
CIFs of Bi phosphates, which are trustworthy?" — a real question from the
bismuth survey that currently needs a script.

- SQLite-backed project library; drag a folder in
- The structure-health pipeline from `PLAN.md` §6, with the known-broken COD
  entries as its regression set
- Deduplication by content hash and structure fingerprint
- Batch analysis over a library, with a sortable results table
- Saved analysis presets
- Compare workspace: N structures or sites side by side, and the reduced-axis
  (d − R₀) atlas plot ported from `bi_pubfig.py`

## Phase 6 — Completing the analysis

**Why:** the multi-definition CN panel is currently indicative rather than
complete, which undercuts the application's central claim.

- Voronoi–Dirichlet solid-angle CN, CrystalNN, CHARDI, Brunner's reciprocal
  and maximum-gap variants, `MinimumOKeeffeNN`
- ChemEnv continuous symmetry measures for geometry assignment
- More bond-valence parameter sets: Gagné & Hawthorne 2015, Brown's accumulated
  IUCr table, softBV — downloaded on demand, never bundled (see
  `THIRD_PARTY_NOTICES.md`)
- Uncertainty propagation: coordinate esds through to a BVS ± error
- Global Instability Index per structure, surfaced in the interface
- The auto-written methods paragraph, with every parameter and threshold cited

## Phase 7 — Remaining display

**Why:** these are the visible gaps a CrystalMaker user notices first.

- Lattice planes and Miller planes
- Slabs and clipping planes
- Thermal ellipsoids — the tensor is read, diagonalised and reported; the geometry is what remains
- Per-atom and per-site style and colour overrides
- Order-independent transparency, once many polyhedra overlap
- Boundary modes: whole molecules, packing diagrams
- Stereo pairs and anaglyph
- Undo, right-click context menus everywhere

## Phase 8 — Diffraction

**Why:** it closes CrystalDiffract's half of CrystalMaker, and HT-XRD is live
work here.

- Powder pattern simulation: wavelength, Lorentz–polarisation, Debye–Waller,
  preferred orientation, peak shape
- d-spacing and hkl table
- Overlay on a measured pattern
- Lattice parameter against temperature, for the HT-XRD series

## Phase 9 — Volumetric data

**Why last:** a third of the remaining work, and the only part needing
machinery that does not exist yet — a 3D grid type, marching cubes, a volume
renderer, and a second transfer-function UI.

- Import CHGCAR, CUBE, XSF
- Isosurfaces, with marching cubes
- 2D sections and contour maps
- Fourier and difference maps
- **Bond-valence energy landscapes.** The pay-off: FACET already has the
  bond-valence machinery, so once there is a grid type this becomes a natural
  extension rather than a new subject — and it is the one item in this whole
  list that VESTA does and that would directly serve the glass work.

## Deliberately out of scope

Each with a reason, so the decision can be revisited rather than rediscovered:

| | Why |
|---|---|
| Structure editing, building from scratch | A large surface. FACET interrogates structures that exist. Overriding an oxidation state or a bond-valence parameter — the two edits that change an answer — is already supported. |
| Single-crystal and electron diffraction | CrystalMaker sells these separately; no use here. |
| Hirshfeld surfaces | CrystalExplorer does this well and is free. |
| Animation and movies | Presentation, not analysis. |
| Dot surfaces | Superseded by ambient occlusion for reading depth. |
| `.cmdf` writing | The format is not openly documented. |

---

## How this gets built

Unchanged from the first three phases, because it has been working:

1. **The engine is pure.** `facet/core` imports no Qt, no plotting, nothing
   heavy. It imports in 0.3 s and is tested without a window.
2. **Pure logic is separated from painting**, so it can be tested with a stub —
   label placement takes a text-measuring callable, not a painter.
3. **Look at the output.** Three rendering bugs in Phase 2 and the invisible
   label layer in Phase 4 all passed every test while being plainly wrong on
   screen. Render it, open the image, look at it.
4. **Verify before changing either side.** Three camera tests failed in Phase 2
   and all three were the test being wrong. Check which is wrong first.
5. **A departure from a reference gets a test that pins it**, with the reason —
   as with the Bi₉⁵⁺ cluster reporting CN 0 where the survey floors it at 1.
