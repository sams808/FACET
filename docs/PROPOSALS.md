# A hundred improvements worth making

Proposals for FACET, written 2026-10-14 against version 0.9.0. Each one names
what would change and why it is worth the work; none is scheduled. They are
grouped by where in the application they land. Items marked **(W)** come from
Wyatt Kiff's notes after running FACET beside the Pedone group's MD code.

## The cutoff explorer and the Site tab

1. **Plateau table under the explorer.** Every tread of the staircase as a row
   (CN, threshold range, decades, Å gap), so a site's choices can be read
   without hovering.
2. **Two sites on one explorer.** Overlay the staircase of a second site in a
   second colour, for a polymorph pair or a cation in two hosts, without
   opening Compare.
3. **The distance axis read against each anion.** With two kinds of anion the
   distance axis follows the first parameter found; a selector for which pair
   the axis is read against would make a fluoride-oxide site honest.
4. **Threshold bookmarks.** Named thresholds (the opening one, a published
   convention, one chosen by hand) as pins the cursor snaps to.
5. **ECoN and the max-gap split as marks on the axis.** They are reported in
   the Site tab; drawn on the explorer they would show how far each rule sits
   from the cursor.
6. **Per-contact sensitivity.** A tooltip on each rail tick giving how much the
   bond-valence sum and φ change if that contact crosses the cut.
7. **Uncertainty band on the running sum.** The R0 uncertainty is systematic;
   drawn as a band around the green curve it would show that a 0.05 v.u. gap
   to the charge is inside it.
8. **Angle-based descriptors in the Site tab.** The bond angles at the site
   (now drawn as arcs) summarised: mean, spread, the largest and smallest,
   and the quadratic elongation already computed.
9. **Site tab export.** The Site tab as a one-page PDF or HTML with the
   explorer image, for a figure caption or a thesis appendix.
10. **Keyboard on the explorer.** Left and right arrows nudge the threshold by
    one tick; Home returns to the pin; Page Up/Down jump to the next plateau.

## The 3D view

11. **Angle arcs that avoid each other.** When a hub has more than six bonds,
    place the labels by a small layout pass so no two overlap at any zoom.
12. **Torsion drawn as a dihedral.** The four-atom measurement reports a
    torsion; drawing the two planes and the dihedral arc would make it
    readable.
13. **Dimming by depth.** Fog is a renderer option; a dimming that follows the
    focus sphere's depth would make the inside of a dense MD box legible.
14. **Clip planes from the slab panel on the MD frame.** The crystal
    workspace's slab applies to a frame only through the scene filter; a
    proper slab control on the Model toolbar would cut a box open.
15. **Polyhedra for the focused atom and its neighbours.** The focus mode
    keeps the atoms inside the sphere; drawing their polyhedra at once would
    show the connectivity the cut implies.
16. **Click a bond.** Bonds are pickable in the GL tier but not in the painter
    tier; a bond click there, giving its two atoms, distance and valence, would
    close the gap between tiers.
17. **Stereo for the MD frame.** The side-by-side modes exist on the crystal
    view; the Model workspace reuses the view class but does not expose them.
18. **A measurement that persists.** Measurement chains vanish on a rebuild;
    keeping them by atom identity through a style change or a threshold drag
    would let a figure be set up without redoing them.
19. **A legend for atom colours.** Colour-by-value modes have a colour bar
    only in the exported figure; a small on-screen bar with the range would
    make the view self-describing.
20. **Screenshot with annotations.** The PNG export omits the angle arcs,
    the measurement chain and the focus sphere; the vector export draws
    them; the two should agree.

## Highlight and selection

21. **Rules by angle. (W)** A show-only rule on the mean bond angle at an atom,
    or on the deviation from the tetrahedral angle, to find distorted sites.
22. **Rules by neighbour composition. (W)** "O bonded to exactly one Si and
    one Na" as a rule, which is what a non-bridging oxygen next to a modifier
    means.
23. **Rules by ring membership.** An atom in a 3-ring or a 4-ring, from the
    ring statistics already computed.
24. **Rules by distance to a channel.** Atoms within a stated distance of a
    channel region, for the ions lining a percolation path.
25. **Saved rule sets.** A set of rules named and stored with the project, so
    a figure's highlight can be reproduced on another frame or another file.
26. **A rule from a histogram bin.** Dragging across two bars of a histogram in
    the Results figure sets the two bounds of a rule; today only the mode bin
    is used.
27. **Selection as a set.** Shift-click to add atoms to a selection, then
    "isolate the selection", "export the selection as XYZ", "add the
    selection to the comparison".
28. **Highlight counts per element.** The count line says how many atoms match;
    broken down by element it would read "12 Si, 40 O".
29. **Colour by shell from the investigated atom.** The shells found for the
    environment, as a colour ramp on the whole frame, show how far the
    network reaches in two or three bonds.
30. **Highlight on the Figure page too.** A highlighted set drawn as a second
    series on the running figure (its own CN distribution against the
    frame's).

## Environments and statistics (Wyatt's direction)

31. **Environment descriptor per atom. (W)** First-shell distances sorted,
    the angles at the centre, and the second shell's composition, as a vector
    per atom at the threshold in force.
32. **Grouping within a tolerance. (W)** Environments of one species and Qⁿ
    grouped by that descriptor within a stated tolerance; the groups counted
    and listed, the tolerance the user's.
33. **A representative per group. (W)** The member closest to the group's
    mean descriptor, exportable as XYZ, so one cluster stands for a class.
34. **Group highlight. (W)** Each group as a highlight rule, so "Si Q³
    environment 2" lights up on the frame.
35. **Electrostatics per group.** The bond-valence landscape already computed
    for channels, sampled at each group's centres, as a per-group number.
36. **Free volume per group.** The empty-sphere radius nearest each centre,
    averaged per group.
37. **Group evolution across frames.** How the group populations change along
    a trajectory, as a series on the Figure page.
38. **Environment similarity between two models.** Compare's descriptor view
    extended with the group populations of two runs side by side.
39. **A per-atom table export.** Every atom with its CN, sum, φ, Qⁿ, shell
    composition and group, as CSV, for analysis outside FACET.
40. **A per-atom lookup.** Type an atom id in the Elements dock to jump to it,
    select it and open its Atom tab.

## The MD engine

41. **Per-frame progress with an estimate.** The analyses report progress;
    an estimated time to completion from the frames done so far would make a
    long run plannable.
42. **Partial results on cancel.** A cancelled run keeps the frames already
    analysed, marked as partial in the provenance.
43. **A frame cache on disk.** The valence table of a frame cached beside the
    file, keyed by the file's hash and the parameters, so a second session
    opens at once.
44. **Analyses in parallel.** Independent analyses on one frame in separate
    processes; the engine is Qt-free, so a process pool fits.
45. **Rings on a chosen sub-network.** Rings through Si and O only, or through
    Si, Al and O, as a choice rather than through every former.
46. **Bond-angle distributions per pair triple.** O–Si–O, Si–O–Si and
    Si–O–Na as separate histograms; today only the former-centred ones are
    reported.
47. **Coordination sequences to a stated depth.** The sequence is computed to
    a fixed depth; a user-chosen depth would reach the ring sizes of interest.
48. **A density-of-states-like plot of bond valences.** Every bond's valence
    as a histogram by pair, which is the distance distribution on the axis
    the cut is made on.
49. **Composition by region.** The box split into cells and each cell's
    composition reported, to see segregation.
50. **A drift check.** The centre of mass of each element against frame, to
    catch a run that drifts, before any dynamics analysis is read.

## Reading files

51. **A format wizard that remembers.** The type map and units answered for a
    file remembered by its path and format, so the next frame of the same
    run asks nothing.
52. **Topology from a LAMMPS data file for a dump.** A dump that lacks element
    columns could take its types from a data file beside it, offered rather
    than asked.
53. **CIF loops the window never reads.** `_geom_bond` and `_geom_angle`
    loops read and offered as "the authors' bonds" on the explorer, so the
    published coordination can be pinned.
54. **Multi-block CIFs as one project.** A CIF with several data blocks opens
    every block as its own entry rather than the first.
55. **mmCIF assemblies.** The assembly operators applied, so a biological CIF
    opens as its assembly and not as the asymmetric unit.
56. **A dropped folder of CIFs.** Dropping a folder on the window opens every
    CIF in it, as File ▸ Open folder does.
57. **Recent files.** A File ▸ Recent list; the window keeps nothing between
    runs today, which is a limitation rather than a choice.
58. **A session that reopens itself.** An option to restore the last session
    at start, with the files, thresholds and camera.
59. **Reading a file while another is analysed.** Opening a second large
    model while the first is read blocks the window; the readers could run
    on the job thread too.
60. **Compressed trajectories throughout.** gzip is read for dumps; xz and
    zstd, and gzip for every text format, would spare a decompress step.

## Reports and exports

61. **A report template.** Which tables, which columns and which thresholds a
    report carries, saved and reused, so a group's standard report is one
    command.
62. **A report that compares two thresholds.** The same files at two cuts,
    with the sites whose CN changes listed first.
63. **A report with figures.** The HTML report with an explorer image per
    site and a 3D image per structure.
64. **Markdown and LaTeX tables of the Site tab.** For a manuscript: one site,
    one table, ready to paste.
65. **FEFF input for an MD atom.** The crystal workspace writes a FEFF input
    around a site; the Atom tab could write one around a frame atom from its
    unwrapped environment.
66. **GSD and extended XYZ export of an isolated environment.** XYZ is the
    first format; a format with a box and velocities would serve a restart.
67. **An XYZ of the whole frame unwrapped about an atom.** For a figure of a
    region of a glass without the box edge cutting it.
68. **A CIF of the isolated cluster as P1.** So a cluster can be opened in any
    crystallographic program.
69. **Provenance as JSON in every export.** The provenance header exists as
    text; a machine-readable copy would let a pipeline check what it reads.
70. **A command line for the crystal analyses.** `facet.report` writes tables;
    a `facet.site FILE SITE --v-bond` printing one site's analysis to the
    terminal would complete it.

## Compare

71. **Difference maps on the 3D view.** Two crystals of the same structure
    type overlaid with per-site colour by the difference in BVS or CN.
72. **Compare two frames of one run.** Frame k and frame m of the same model
    side by side, with the atoms that changed CN listed.
73. **Compare a crystal site with its MD equivalents.** The crystal site's
    staircase over the distribution of the frame's atoms of that element,
    which exists for one descriptor, extended to all of them.
74. **A table of what differs.** Compare's table export lists everything;
    a view that lists only the rows where the models differ by more than a
    stated amount.
75. **Linked cursors.** Dragging the threshold in Compare moves every model's
    cursor; today each model keeps its own.
76. **A basket that persists.** The comparison basket saved with the session.
77. **Compare across parameter sets.** The same structure analysed with two
    bond-valence parameter sets, side by side, to show what the parameters
    decide.
78. **Statistical tests stated, not judged.** For two distributions, the
    Kolmogorov–Smirnov distance and the Wasserstein distance reported as
    numbers, with no verdict attached.
79. **Overlay of radial distribution functions.** The pair distribution
    functions of two runs on one axis, with the difference below.
80. **A difference of staircases.** The CN-against-threshold curves of two
    sites subtracted, to show over which thresholds they disagree.

## Verification and trust

81. **A self-check command.** `facet.check` re-runs the reference comparisons
    (vitrum, amorphouspy, FEFF read-back) where the references are present
    and prints what agreed.
82. **Reference values in the bundled examples.** Each example CIF with the
    numbers the tutorials state, checked by a test so the tutorials never go
    stale.
83. **A parity harness for an external MD code. (W)** Read the Pedone code's
    outputs for a frame and lay them beside FACET's descriptor by descriptor.
84. **Parameter provenance per contact.** Each contact's R0 and b shown with
    the row of the table they came from, so a disagreement can be traced to
    its source.
85. **A changelog in the application.** Help ▸ What changed, built from the
    commit messages, so a user on 0.8 knows what 0.9 did.
86. **Stated numerical tolerances.** The tolerances used in the tests
    (1e-12 on sums, 1e-6 on angles) written in VERIFICATION.md as the
    precision the numbers are checked to.
87. **A warning when a parameter is estimated.** The Site tab marks estimated
    parameters with "est."; the explorer and the report could carry the same
    mark on the affected contacts.
88. **Oxidation states from the file, visibly.** Where a CIF gives oxidation
    states and where FACET assumed them, as a column in the report.
89. **A test that opens every bundled example on every renderer tier.** The
    GL tiers are checked on the developer's GPU; a CI machine without a GPU
    would check the painter tier on every example at every commit.
90. **Build reproducibility.** The build script records the versions of every
    package in the bundle, so a build can be rebuilt from its record.

## The window

91. **Settings that survive a restart.** Theme, layout, the explorer's
    defaults and the recent files, kept in QSettings with a "forget
    everything" button.
92. **A dark theme audit on Windows.** The dark palette is checked offscreen;
    an on-screen check of every panel on a dark Windows session would catch
    what offscreen cannot.
93. **Dock layouts that can be saved.** The docks are movable; a saved layout
    per task (figure making, analysis, comparison) would spare setting them
    up each time.
94. **Keyboard access to every menu.** The mnemonics are unique per menu; a
    full keyboard pass (Tab order, Enter on lists, Escape on dialogs) is not
    yet checked by a test.
95. **A status line that is a log.** The status bar's messages collected in a
    panel, with timestamps, so a message missed is not lost.
96. **Undo for the threshold.** The history covers how a structure is drawn;
    a threshold drag could be a step in it.
97. **A smaller installer.** Qt's unused plugins and locales still ship; an
    audit of the bundle could shed tens of megabytes.
98. **Start-up time.** The exe takes about ten seconds to the first window; a
    lazy import of the MD engine and the report module would cut the time
    for a user who opens a CIF.
99. **A first-run tour.** Six tutorials exist; a first-run prompt offering
    tutorial 1 would put a new user on it.
100. **A feedback channel in Help.** Help ▸ Report a problem that writes the
     version, the renderer line and the last status messages to the
     clipboard, ready to paste into an issue.
