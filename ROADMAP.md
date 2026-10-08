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

## Phase MD — Glass models from molecular dynamics *(done)*

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

What each step delivered (`FEATURES.md` §14 has the row-by-row inventory):

0. **Scope and plan.** `FEATURES.md` §14 and this phase; `tools/bench_md.py`
   reproduces the timing above.
1. **Readers and the frame model.** `md_model.py` and `md_readers.py`: LAMMPS
   data files and dumps in every coordinate convention, triclinic boxes as the
   LAMMPS manual defines them, multi-frame extended XYZ, VASP XDATCAR and
   DL_POLY. An element comes only from the user's map, an element column, type
   labels or a mass that matches one element; frames are found by byte offset
   and loaded on demand, and a frame that cannot be read is counted with its
   reason. Reading takes 0.02–0.08 s per 10 000-atom frame. Later in the phase,
   14 more formats in four modules (DCD, XTC, LAMMPS binary and YAML dumps,
   CFG, ASE `.traj`, GSD, AMBER NetCDF, CASTEP, GRO, XSF, multi-model PDB, IMD,
   POSCAR series), recognition by content, and series of files: 114 of 115
   files written by LAMMPS, ASE, OVITO and pymatgen are read to the precision
   of their format, and the 115th is refused with its reason.
2. **A vectorised bulk engine.** `bulk.py`: one neighbour search per frame, then
   numpy. A 9 261-atom frame takes 0.51 s where the crystal analysis took 26 s,
   and a new threshold 4–10 ms for 10 000 atoms. The crystal-as-glass equality
   holds atom for atom, to 2.5 × 10⁻¹⁴, on six bundled and five reference
   crystals tiled 2×2×2 in P1.
3. **Glass descriptors, and more than the plan asked for.** `glass.py` and
   `md_stats.py`: partial g(r) and N(r) with first minima judged against their
   counting error; CN cut by bond valence and by distance, side by side;
   speciation, Qⁿ, Qⁿ(mX), N₄, Al CN, linkages, anion environments, φ, bond
   angles and lengths, composition; every descriptor a mean over frames with
   its spread. `md_network.py`: ring statistics, coordination sequences,
   polyhedral sharing, connected components, Warren–Cowley order.
   `md_order.py`: Steinhardt and tetrahedral order, polyhedron distortion,
   Voronoi cells, empty spheres and free volume.
4. **Comparison with experiment, and dynamics.** `md_scattering.py`: X-ray,
   neutron and electron S(Q), F(Q) and G(r), with a measured curve overlaid and
   R_χ. `md_spectroscopy.py`: the model's fractions beside NMR fractions; NMR
   shifts and spectra from correlations the user supplies, none shipped; EXAFS
   cumulants, and FEFF over a trajectory. `md_dynamics.py`: mean-square
   displacement, van Hove functions, VACF and vibrational density of states,
   conductivity and lifetimes.
5. **A Model workspace.** `md_analysis.py` runs 25 analyses from one request,
   with one neighbour search per frame: 7.5–9.7 s per frame with all 25 on
   models of 2 880–3 000 atoms, 0.35 s with the glass analysis alone, about
   1.1 GB at peak. `md_export.py` writes CSV or XLSX with the provenance on
   every file, and `python -m facet.md` runs the same with no window. The Model
   window (`facet/ui/md_*.py`) opens a dropped or chosen model, asks for what
   the reader lacks, runs in the background with progress and cancel, shows
   each descriptor as a figure beside its rows, moves the threshold on one
   frame without a new search, and draws one frame in 3D from the bulk
   engine's bonds.
6. **Verification.** The *MD models* section of `VERIFICATION.md`, in the same
   five kinds of check, on 115 files written by four programs and five
   models made with LAMMPS for the purpose (four quenched glasses and a
   melt). vitrum 1.1.0 and amorphouspy 0.8.1, run outside FACET on the same
   frames and cut at the same distances, give the same CN on all 237 600
   atom-frames and the same Qⁿ on all 71 300 former-frames.

Ring statistics, out of scope in the first plan, were done in Step 3. NMR
spectra are computed only from a correlation the user supplies with its
reference; quadrupolar lineshapes are not simulated.

What remains:

- **A model from the group's own work.** None has been analysed yet.
- **The built application.** The `--app` build of 2026-10-07 opened a LAMMPS
  dump, a DCD and an XTC in a Model window with no error box; no analysis has
  been run in a built application yet. That build found the window's pages
  unreadable on a Windows desktop in dark mode (1.05:1) and its status bar
  behind the taskbar; both are fixed, in the crystal window as well.
- **The Model window.** It writes a request file and does not yet load one.
  The OpenGL tiers have not been checked drawing an MD frame on a machine with
  a graphics card. Faster bond drawing for large frames was proposed and not
  built: impostor bonds on the OpenGL tiers (estimated, not measured) and
  cached sprites on the QPainter tier (prototyped). Above 5 000 atoms the
  QPainter tier draws atoms only.
- **Open in the code.** A core/shell model whose shells are mapped to O
  doubles the O count with no note; a Po-209 mass, 0.002 amu from Bi's weight,
  reads as Bi, and the mass tolerance that decides it is not settled;
  non-periodic boundaries are not handled alike across formats; none of the 47
  CIFs in the reference collection that hold Si and O is an M_xSi₂O₅ with Si
  as the only former, so the Q³ reference test skips.

## Deliberately out of scope

Each with a reason, so the decision can be revisited rather than rediscovered:

| | Why |
|---|---|
| Structure editing, building from scratch | A large surface. FACET interrogates structures that exist. Overriding an oxidation state or a bond-valence parameter — the two edits that change an answer — is already supported. |
| Single-crystal and electron diffraction | CrystalMaker sells these separately; no use here. |
| Hirshfeld surfaces | CrystalExplorer does this well and is free. |
| Animation and movies | Presentation, not analysis. |
| Running, building or editing molecular dynamics | FACET reads and measures MD output (Phase MD). Producing it is a different program. |
| Quadrupolar NMR lineshapes | A lineshape needs the electric-field gradient at each nucleus, which FACET does not compute; the NMR analysis of a model gives isotropic shifts. |
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
