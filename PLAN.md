# CIF coordination explorer — build plan

Working name: **FACET** (alternatives at the end). Windows desktop application,
shipped as a signed-or-unsigned `.exe`, no Python installation required of the
user.

Status of this document: plan, agreed scope not yet fixed. Section 9 lists the
decisions that need an answer before Phase 1 starts.

---

## 1. What the application is for

A coordination number is not measured. It is produced by choosing a cutoff, and
for a cation with a stereochemically active lone pair there is no gap in the
contact distribution where a cutoff naturally falls. Published CNs for Bi(III)
therefore vary by two or three units across papers describing the same site, and
the variation is a reporting artefact rather than a chemical difference.

The application makes that visible and makes the alternative usable: cutoffs set
by **partial bond valence** rather than by a distance, so that every anion is cut
at the same minimum bond strength instead of at a distance chosen for oxygen.

Three things follow, and they are the product:

1. Any coordination number the application reports is accompanied by the rule
   that produced it and by the range of cutoffs over which it is stable.
2. Several independent definitions of CN are computed at once and shown side by
   side, so agreement or disagreement between them is itself a result.
3. Everything is traceable to a file, a parameter set with its citation, and a
   threshold, and can be exported as a methods paragraph.

Bismuth is the immediate subject. Nothing in the design is specific to it.

---

## 2. Versatility: the generalisation from the existing Bi code

The existing engine (`XRD\cif\Bi\study\bi_core.py`, `bi_survey.py`) is validated
— an independent zero-shared-code recomputation agrees on 94 of 94 sites, and
`test_core.py` checks the geometric primitives against analytic shapes. It is
the starting point, not a reference to reimplement. What has to change:

| Hardcoded now | Becomes |
|---|---|
| `BV` dict: Bi(III) and Bi(V) only, 9 anions | full IUCr accumulated bond-valence table, every cation–anion pair, plus alternative parameter sets (§4) |
| `LIGANDS` list of 9 anions | anion/cation assignment by electronegativity, with per-structure user override |
| `OX` formal-charge table, `charge_balance_bi()` | general oxidation-state resolution: stated in CIF → charge balance → BVS self-consistency → user. The route used is displayed |
| per-structure exclusion lists in `bi_survey.py` | a data-quality pipeline producing verdicts and an explicit user override, recorded with a reason (§6) |
| `CIF_FILES` registries | a project library backed by SQLite |
| Bi-specific report and figure text | element-agnostic templates |
| analysis of one element | analysis of any selected site, any element, or every cation at once |

`phi_index`, `ecoN`, `void_cone`, `polyhedron`, `gap_split` and `bv_of` are
already element-agnostic and move across unchanged. `Bi_sites.csv`'s 57 columns
are effectively the per-site data model already and become the schema.

A per-element-family preset keeps the common case one click: selecting a lone-pair
cation loads the thresholds and displays appropriate to it, selecting a d⁰
transition metal loads different ones, and the preset is visible and editable.

---

## 3. The 3D viewport — decided, and verified on this machine

**Verdict: PySide6 alone, a custom OpenGL 3.3 core renderer. No VTK, no pyvista,
no PyOpenGL, no QWebEngine.** A working offscreen prototype is in
`proto\gl_proto.py` and renders α-Bi₂O₃ and BiPO₄.

### Why not the alternatives

| Option | Why not |
|---|---|
| VTK / pyvista in a QtInteractor | +100–200 MB in the bundle, chronic PyInstaller hidden-import problems, depth-sorted transparency that glitches on overlapping polyhedra, and a generic look that cannot be pushed to impostors or SSAO |
| three.js / 3Dmol.js in QWebEngineView | mature renderer, but ~130 MB of Chromium in the bundle, a Python↔JS bridge in the path of every pick, and debugging split across two languages |
| Qt3D / Qt Quick 3D | a scene-graph API fighting a domain that wants one instanced draw call per atom type; little control over the fragment stage |
| matplotlib 3D | already used for the survey figures; keep as the static fallback only |

### The technique stack, which is what makes it look expensive

* **Ray-traced sphere impostors.** One instanced quad per atom; the sphere is
  intersected analytically in the fragment shader and `gl_FragDepth` written.
  Mathematically exact silhouettes at any zoom, no tessellation, and the atom
  ceiling is set by fill rate rather than triangle count.
* **Instanced tube meshes for bonds**, split at the midpoint so each half carries
  its own atom colour, with **radius proportional to bond valence** — so bond
  strength is visible, and a 0.03 v.u. contact reads as the hairline it is next
  to a 0.5 v.u. bond.
* **Screen-space ambient occlusion** from a G-buffer of view position and normal.
  This is the single largest contributor to the "solid object" impression and is
  the main thing VESTA's renderer lacks.
* **Depth cueing** towards the background — CrystalMaker's signature look.
* **Silhouette pass** from view-position discontinuity.
* **Supersampled rendering**, downsampled at the end. This driver declines MSAA
  on the default framebuffer, and supersampling is what a 600 dpi figure export
  wants anyway.
* **Order-independent transparency** (weighted blended) for coordination
  polyhedra — to be added; the prototype uses simple alpha blending, which is
  adequate for one polyhedron and wrong for many overlapping ones.
* **Picking by ID buffer**, not raycasting: render an integer atom id to an
  offscreen attachment and read one pixel. Exact, and free of near-miss errors.
* **Labels and measurements** as a `QPainter` overlay on the GL widget, so text
  stays crisp at any DPI and needs no texture atlas.

### Measured on this machine

| | |
|---|---|
| Context granted | OpenGL 4.3 core on the RTX 4050; **3.3 core on the Intel UHD** |
| Offscreen contexts | default to the **integrated GPU** — the discrete adapter must be requested explicitly, or Intel accepted |
| Target | **GL 3.3 core**, which the Intel UHD supports and which has everything needed (instancing, MRT, FBOs) |
| Geometry pass | 43–105 ms for 58 spheres + 66 bonds at 4500×3300 — i.e. supersampled *stills*. At 1× window resolution this is well inside a frame |
| SSAO pass | 26–209 ms at 4500×3300, 24 samples |
| CIF parse (gemmi) | **0.8 ms** — 60× faster than pymatgen's 49 ms |
| `import pymatgen` | **1.66 s** — must stay lazy, it alone would double launch time |
| Neighbour search to 6 Å | 75 ms — the only expensive step, and it runs once |
| `void_cone` (12000-point sphere + polish) | 1.0 ms |
| `polyhedron` (angles, hull, trans-pair matching) | 2.7 ms |
| `ecoN` | 0.3 ms |

**The consequence for the interface.** Run the neighbour search once at the
widest tabulation radius and cache the contact list. Dragging the cutoff is then
pure in-memory filtering plus about 4 ms of recomputation — over 200 fps. The
central interaction of the application costs nothing.

### PySide6 OpenGL binding traps (all hit in the prototype)

* `QOpenGLFunctions_3_3_Core` exposes 218 GL calls including
  `glDrawArraysInstanced`, `glVertexAttribDivisor` and `glDrawBuffers`. The
  plain `QOpenGLFunctions` does not — it is ES2-level.
* `glVertexAttribPointer`'s `normalized` argument is bound as an **int**, not a
  bool.
* Its buffer offset must be a **`shiboken6.VoidPtr`**. An `int` and `None` are
  rejected outright, but **`ctypes.c_void_p` is accepted and silently passes the
  wrong address**, so the draw call succeeds, raises no GL error, and renders
  nothing.
* `glUniformMatrix4fv` wants a flat `Sequence[float]`, not `bytes`. Row-major
  numpy matrices acting on column vectors need `transpose = 1`.
* `glGenVertexArrays` and `glGenFramebuffers` are absent; use
  `QOpenGLVertexArrayObject` and `QOpenGLFramebufferObject` instead.
* `QOpenGLFramebufferObject.toImage()` leaves the **default** framebuffer bound.
  Any draw issued after a readback goes to the wrong target.

---

## 4. Bond valence: what ships

`pymatgen.analysis.bond_valence.BV_PARAMS` holds only 75 entries in the
O'Keeffe–Brese electronegativity-estimated `c`/`r` form. That is not enough for a
general application, so parameter tables are shipped as data files:

* **Brese & O'Keeffe 1991** — the set the existing Bi work uses. R₀(Bi³⁺–O) =
  2.09, R₀(Bi⁵⁺–O) = 2.06, b = 0.37 throughout.
* **Brown's accumulated IUCr table** (`bvparm*.cif`) — the broad coverage set,
  thousands of cation–anion pairs, many with their own fitted `b`.
* **Gagné & Hawthorne 2015** — the modern refit for oxides, per oxidation state.
* **softBV (Adams)** — Morse-type parameters, needed if bond-valence energy
  landscapes are added later.

The parameter set is a visible, switchable setting, never a hidden default, and
the active set with its citation appears on every report and figure. Switching
sets re-runs everything, and the difference between sets is itself displayable —
which is the honest way to show that R₀ carries about ±0.02 Å, i.e. roughly ±6 %
in a bond-valence sum, and that sites should not be flagged on a few per cent.

Derived quantities: BVS per site, **Global Instability Index** per structure,
bond-valence vector sum, and **φ = |Σvᵢûᵢ| / Σvᵢ** — the normalised vector sum,
dimensionless, bounded 0–1, and exactly invariant to an error in R₀ because
shifting R₀ by δ scales every vᵢ by exp(δ/b), which cancels. φ, not bare |BVV|,
is what allows stereoactivity to be compared across anions.

---

## 5. Coordination number: the multi-definition panel

The application never reports a single CN without context. For the selected site
it computes, side by side:

| Definition | Note |
|---|---|
| Bond-valence threshold | the default; threshold shown, currently 0.075 v.u. for a bond and 0.02 v.u. for tabulation |
| Hard distance cutoff | user-set, plus the conventional literature value for the pair |
| Sum of ionic radii, scaled | Shannon radii |
| Maximum-gap / largest ratio step | Brunner's method; the existing `gap_split` |
| Hoppe ECoN | effective coordination number, already implemented |
| CHARDI | charge distribution |
| Voronoi–Dirichlet, solid-angle weighted | via `pymatgen.analysis.local_env.VoronoiNN` |
| pymatgen `CrystalNN`, `EconNN`, `MinimumDistanceNN`, `JmolNN`, `MinimumOKeeffeNN` | eight independent implementations available for free |
| ChemEnv continuous symmetry measures | `LocalGeometryFinder`, for geometry assignment rather than count |

Plus the object that makes the argument: the **CN-versus-cutoff staircase**, with
plateau widths marked. A CN that holds over a wide plateau is defensible; one
that occupies a narrow step is a reporting choice. This is the single most
important plot in the application.

---

## 6. CIF ingestion and structure health

The prior survey found genuinely broken database entries — an atom loop of
`? ? ? ?`, oxygen on the wrong Wyckoff site, 40 contacts below 2.00 Å in a
modulated supercell, z esds of 0.07 fractional, misnamed files, and the same
refinement redistributed by two databases. Those become the regression set.

Ingest: **gemmi** for parsing (0.8 ms), **spglib** for symmetry cross-check.
Stages, each producing a plain-language verdict rather than an exception:

1. Parse — dialects, multi-block, encodings, esds retained for propagation.
2. Symmetry — stated H-M symbol versus stated symop list versus spglib's
   determination; disagreement is reported, not silently resolved.
3. Occupancy and disorder — partial occupancy, split positions, the
   contacts-below-1.6 Å rule.
4. Chemistry — charge balance, BVS of **every** cation, GII.
5. Geometry — impossible contacts, coordinate esds too large for the conclusion.
6. Library — dedup by content hash ignoring `_audit_*` stamps, and by a structure
   fingerprint of cell plus sorted element/occupancy/coordinates hashing
   *element symbols* rather than charge-decorated labels.

Each structure carries a health state and its reasons. A user can override and
proceed; the override and its reason are recorded and appear in the report.

---

## 7. Layout

One window. A left rail of workspaces, matching the pattern already familiar
from PRISM. The three-zone Structure workspace is where most time is spent.

```
┌──┬────────────────────────────────────────────────┬──────────────────────┐
│  │  BiPO4 · P2₁/n · COD 7023719      [health ●]   │  SITE  Bi1           │
│L │                                                │  Wyckoff 4e · sym 1  │
│i │                                                │  ─────────────────── │
│b │                                                │  CN (0.075 v.u.)  9  │
│r │                                                │  ECoN           7.4  │
│a │            3D VIEWPORT                         │  Voronoi          9  │
│r │            impostors · SSAO · depth cue        │  CrystalNN        8  │
│y │            click an atom to select a site      │  gap split      5+4  │
│  │            drag to rotate · scroll to zoom     │  ─────────────────── │
│S │                                                │  BVS       3.08 v.u. │
│t │                                                │  φ              0.29 │
│r │                                                │  ⟨R₁⟩       2.41 Å   │
│u │                                                │  void cone     72°   │
│c │                                                │  ─────────────────── │
│t │                                                │  ▸ contacts (11)     │
│u │                                                │  ▸ distortion        │
│r ├────────────────────────────────────────────────┤  ▸ radial shells     │
│e │ CUTOFF EXPLORER                                │  ▸ provenance        │
│  │  CN ┤    ┌───┐                                 │                      │
│⋯ │  12 ┤────┘   └──┐                              │  [copy methods text] │
│  │   9 ┤           └──────────┐   ← plateau       │  [export site CSV]   │
│  │   6 ┤                      └────────           │                      │
│  │     └─┬────┬────────┬──────────┬────────┬──    │                      │
│  │      0.02 0.05    0.075      0.15     0.3 v.u. │                      │
│  │       ▲ tabulate   ▲ bond      ▲ literature 3.0 Å                     │
│  │      ╎ ╎  ╎╎ ╎ ╎        ╎   ╎        ╎  ← every contact of this site  │
└──┴────────────────────────────────────────────────┴──────────────────────┘
```

The bottom strip is the spine of the application and is always visible in the
Structure workspace. Each tick on its axis is one contact of the selected site,
placed at its bond valence. The draggable threshold sets what counts as a bond;
the staircase above shows what CN that choice produces, and how wide the plateau
around it is. The 3D view, the site panel and the contact table all follow the
threshold live — which the timings in §3 make free.

Marked on the axis, always: the tabulation threshold, the bond threshold, and
the conventional literature distance cutoff for the pair, converted to valence.
Seeing 3.00 Å land mid-plateau for Bi–O and mid-step for Bi–I is the argument.

### Workspaces

| | Workspace | For |
|---|---|---|
| 1 | **Library** | the CIF collection: ingest, health traffic lights, dedup, metadata, batch runs |
| 2 | **Structure** | the layout above — 3D, site analysis, cutoff explorer |
| 3 | **Coordination** | every site in the structure, or across the library, as a multi-definition CN table with staircases |
| 4 | **Compare** | N structures or sites side by side; the reduced-axis (d − R₀) atlas plot from the existing publication figures |
| 5 | **Shells** | radial shells with degeneracies; EXAFS path planning and FEFF input export |
| 6 | **Symmetry** | Wyckoff positions, site symmetry, whether an EFG vanishes by symmetry, factor-group mode counting for Raman/IR |
| 7 | **Diffraction** | powder pattern simulation, d-spacing/hkl table, overlay on a measured pattern |
| 8 | **Report** | auto-written methods paragraph, figure recipes, export to CSV/XLSX/PNG/PDF |

Workspaces 1–3 are release 1. The rest are phased in §8.

### Non-coder requirements, concretely

* Drag and drop a CIF or a folder of them onto the window.
* Nothing requires typing a number to get a first answer: presets per element
  family, sensible defaults, and every default visible and overridable.
* Right-click context menus everywhere — on an atom, a site row, a contact, a
  structure in the library.
* No modal alarm on an unknown or unusual fact. Structure health is a calm
  coloured dot with reasons on hover, never a popup.
* Every number has a provenance affordance: which file, which parameter set with
  its citation, which threshold, which formula. "Show the arithmetic" expands
  the actual calculation for the selected contact.
* A methods paragraph is generated and copyable, stating the parameter set,
  thresholds, software version and structure source — so an SI section is a
  paste rather than a reconstruction.
* Session save and restore; last folder remembered; everything exportable to
  CSV/XLSX.

---

## 8. Phases

Each phase ends with a runnable `.exe` and a green test suite.

**Phase 0 — skeleton (foundation).**
Repository, `xtal_core` / `xtal_gl` / `xtal_ui` / `tests` split, PySide6 shell
with the left rail, splash before heavy imports, PyInstaller build script with
the exclusion list, version stamping, crash log. No science yet.

**Phase 1 — engine.**
Port `bi_core` into `xtal_core`, generalised per §2. gemmi ingest, spglib
symmetry, cell-list neighbour search, full bond-valence tables, all CN
definitions, polyhedral indices, φ. Ported tests plus new ones; the existing
`Bi_sites.csv` becomes a regression fixture — every one of its 105 sites must
reproduce.

**Phase 2 — the 3D viewport.**
Promote `proto\gl_proto.py` to an interactive `QOpenGLWidget`: trackball, pick by
ID buffer, ball-and-stick / polyhedral / space-filling styles, order-independent
transparency, labels and click-to-measure via a `QPainter` overlay, image export
at 600 dpi. Software fallback path when GL 3.3 cannot be created.

**Phase 3 — Structure workspace and the cutoff explorer.**
The layout of §7, wired live. This is the release-defining feature and the point
at which the application is worth using.

**Phase 4 — Library and structure health.**
SQLite-backed project library, the §6 validation pipeline with the known-broken
entries as the regression set, dedup, batch analysis, results table.

**Phase 5 — Coordination workspace and reporting.**
Multi-definition tables across a library, plateau analysis, the auto-written
methods paragraph, CSV/XLSX export, publication figure recipes ported from
`bi_pubfig.py`.

→ **Release 1** here. Workspaces 1–3 plus Report.

**Phase 6 onwards, in the order the work demands them.**
Compare workspace and the reduced-axis atlas · radial shells and FEFF export ·
symmetry, EFG-by-symmetry and factor-group mode counting · powder diffraction
simulation and overlay · bond-valence energy landscapes · interchange with the
Bi L₃ XAS pipeline, PRISM and the glass-speciation work.

---

## 9. Decisions needed before Phase 1

1. **Name.** FACET is the working choice — a real word, crystallographic, and it
   sits beside PRISM in the existing family. Alternatives: HEDRA (from
   polyhedra), PLUMB (to measure depth precisely; also *plumbum*, Bi's lone-pair
   neighbour), FATHOM (to measure, and to understand), CORONA, NIMBUS.
2. **Default bond-valence parameter set** when a structure offers no reason to
   prefer one — Brese & O'Keeffe 1991 for continuity with the existing Bi work,
   or Gagné & Hawthorne 2015 for oxides.
3. **Scope of release 1**: workspaces 1–3 as above, or Structure alone shipped
   sooner with the library deferred.
4. **Distribution**: personal use, the research group, or public. This sets
   whether code signing, a licence and a public repository are needed, and
   whether an unsigned exe's SmartScreen warning is acceptable.
5. **Whether the application writes structures** or is read-plus-analyse only.
   Structure editing is a large surface and is currently out of scope.
