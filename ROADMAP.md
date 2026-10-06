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

## Phase MD — Glass models from molecular dynamics *(planned)*

**Why:** the glass work produces MD models, and the descriptors reported for
them (CN distributions, Qⁿ, N₄, bridging and non-bridging oxygen) all depend on
where a bond is cut, which is the question FACET exists to make visible. FACET
reads and measures MD output. It does not run, build or edit MD.

The crystal path analyses one site at a time, and an MD frame has one site per
atom. Measured with `tools/bench_md.py` on Windows (Intel i5-13420H, pinned to
the performance cores), a synthetic 9 261-atom box takes 7 s to read, 98 % of it
the spglib symmetry search, and 26 s to analyse, 14–24 % of that in a site
lookup that scans every atom once per site. A hundred frames would take close
to an hour. With the site lookup bypassed the analysis still takes 17–25 s,
half of it per-site geometry that a frame-wide count does not need. So the MD
path gets an engine of its own, and the crystal path is left as it is.

0. **Scope and plan.** `FEATURES.md` §14 lists the features as planned;
   `tools/bench_md.py` reproduces the timing above.
1. **Readers and the frame model.** LAMMPS `data` and `dump` files (scaled,
   unscaled and unwrapped coordinates; triclinic boxes converted as the LAMMPS
   manual defines them) and multi-frame extended XYZ. Atom type → element with
   its source recorded. Oxidation states are model inputs, not resolved.
   Skipped frames are counted and reported.
2. **A vectorised bulk engine.** One neighbour search per frame, every atom at
   once: CN at the threshold, bond-valence sum, φ and plateau width,
   re-thresholded without a new search. Proved by the crystal-as-glass
   equality: the bundled examples as 2×2×2 P1 supercells give, atom for atom,
   what the crystal analysis gives for the site, to 1e-10.
3. **Glass descriptors.** Partial g(r) and N(r) with their first minima; CN
   distributions cut by bond valence and by distance, side by side; bridging,
   non-bridging, free and tricluster oxygen against a user-chosen set of network
   formers; Qⁿ and Qⁿ(mX); N₄; Al CN; halide environments; φ distributions for
   lone-pair cations; bond-angle distributions. Each one averaged over frames,
   with its spread.
4. **Comparison with experiment.** X-ray and neutron G(r) and S(q) from the
   model, overlaid on a measurement; the model's Qⁿ, N₄ and Al fractions beside
   fractions from NMR fits. No simulated spectra.
5. **A Model workspace.** Type map, frame range, computation in the background
   with progress and cancel, histograms in place of the site table, the
   threshold slider, and CSV and XLSX export with provenance.
6. **Verification.** An MD section in `VERIFICATION.md` with the same five kinds
   of check, including an independent implementation run outside FACET on a
   real model.

Ring statistics and simulated NMR spectra are out of scope for this round. Rings
get a documented hook that names R.I.N.G.S.; NMR gets a stub that computes
nothing until published correlations and their references are supplied.

## Deliberately out of scope

Each with a reason, so the decision can be revisited rather than rediscovered:

| | Why |
|---|---|
| Structure editing, building from scratch | A large surface. FACET interrogates structures that exist. Overriding an oxidation state or a bond-valence parameter — the two edits that change an answer — is already supported. |
| Single-crystal and electron diffraction | CrystalMaker sells these separately; no use here. |
| Hirshfeld surfaces | CrystalExplorer does this well and is free. |
| Animation and movies | Presentation, not analysis. |
| Running, building or editing molecular dynamics | FACET reads and measures MD output (Phase MD). Producing it is a different program. |
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
