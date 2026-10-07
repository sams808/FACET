"""Local order and geometry of every atom of an MD frame.

Coordination numbers and Q^n say how many neighbours an atom has and how its
polyhedra link. They do not say how those neighbours are arranged, how far a
polyhedron departs from a regular one, how much space each atom holds, or
where the empty space of the model is. A glass paper reports those too, and a
crystal-side quantity (the polyhedron distortion of ``polyhedra.shape``) has to
come out the same when the crystal is read as a model. This module measures,
for every atom of a :class:`~facet.core.md_model.Frame` at once:

* **bond-orientational order**: Steinhardt's q_l and normalised w_l per atom
  [1], their neighbour-averaged forms q-bar_l and w-bar_l [2], and Steinhardt's
  original global Q_l, W_l over every bond of the frame (:func:`steinhardt`);
* **tetrahedral order** q_tet of Errington and Debenedetti [3], a rescaling of
  the order parameter of Chau and Hardwick [4] (:func:`tetrahedral_order`);
* **polyhedron distortion**: Baur's bond-length distortion index [5], the
  bond-angle variance and quadratic elongation of Robinson, Gibbs and Ribbe
  [6], the polyhedron volume, and Hoppe's effective coordination number [7]
  (:func:`polyhedron_shape`);
* **Voronoi cells** of the periodic frame: volume, face count (the Voronoi CN)
  and Voronoi index <n3, n4, n5, n6, ...> [8, 9] (:func:`voronoi_cells`);
* **voids**: the empty sphere of every Delaunay tetrahedron [10]
  (:func:`empty_spheres`), and the geometric, probe-centre and
  probe-occupiable free-volume fractions for a probe radius [11]
  (:func:`free_volume`).

The per-atom results of one frame become frame-averaged distributions with
their spread through :mod:`facet.core.md_stats` (:func:`element_histograms`,
:func:`voronoi_index_distribution`, :func:`voronoi_cn_distribution`,
:func:`empty_sphere_histogram`, :func:`free_volume_scalars`).

WHO COUNTS AS A NEIGHBOUR
-------------------------
Every per-atom order parameter is a function of a neighbour list, and the
number it gives depends on that list as much as on the structure. So the list
is an explicit input, a :class:`Neighbours`, built one of three ways, and its
definition travels with every result:

* :func:`neighbours_from_bonds`: the bonds ``bulk.bonds_at`` counts at
  v_bond, the same bond definition as the CN, Q^n and the bond angles, so the
  polyhedron a distortion index measures is the polyhedron the CN counts;
* :func:`neighbours_from_pairs`: a distance cutoff per element pair, taken
  from the pairs of the frame's one search (``bulk.find_pairs`` /
  ``bulk.iter_pairs``). No cutoff is set here: each one is the caller's, with
  its source (the first minimum of a partial g(r) measured on the model, as a
  rule), and both are recorded;
* :func:`neighbours_from_voronoi`: the faces of the Voronoi cell, a
  definition with no parameter beyond the face-area threshold. Each counted
  face is a full neighbour: without a threshold, faces of 1e-8 Å^2 count,
  and on four glass frames such faces moved q_4 by up to 0.12 (the
  function says how this was measured; the result notes the smallest
  face).

None of the three searches for neighbours again: the first two read what the
bulk engine found, the third reads the tessellation. The tessellations, the
empty spheres and the free volume do query atom positions themselves
(Qhull on the atoms and their periodic images; KD-trees for the clearance of
a void and for the expanded spheres that meet a grid point or each other),
as geometry of cells and voids, at radii set by the cells, the atom radii
and the probe; apart from the Voronoi faces, none of those queries defines
a neighbour list.

STEINHARDT ORDER, AND THE CONVENTIONS IT DEPENDS ON
---------------------------------------------------
For atom i with N_b(i) neighbours along unit vectors r_ij,
q_lm(i) = (1 / N_b) sum_j Y_lm(r_ij), q_l(i) = sqrt(4 pi / (2l + 1) sum_m
|q_lm|^2), and w_l(i) = sum_{m1+m2+m3=0} (l l l; m1 m2 m3) q_lm1 q_lm2 q_lm3,
reported normalised, w-hat_l = w_l / (sum_m |q_lm|^2)^(3/2) [1, 2]. The averaged
forms replace q_lm(i) by the mean of q_lm over i and its neighbours,
q-bar_lm(i) = (q_lm(i) + sum_k q_lm(k)) / (N_b(i) + 1) [2]. The global Q_l and
W-hat_l are Steinhardt's original definition [1]: Y_lm averaged over every
(centre, neighbour) entry of the list. A list holding each bond from both ends
holds r and -r, so odd l give zero there.

* The spherical harmonics are scipy's, with the Condon-Shortley phase:
  ``scipy.special.sph_harm_y(l, m, polar, azimuth)`` from scipy 1.15, where it
  replaced ``sph_harm(m, l, azimuth, polar)`` (deprecated in 1.15, removed in
  1.17; it is used only where ``sph_harm_y`` does not exist, and the two
  agreed to 0.0 on this machine, scipy 1.15.1). q_l and w-hat_l do not depend
  on the phase convention, as long as it is one convention throughout.
* The azimuth is ``atan2(y, x)`` taken into [0, 2 pi), the range scipy states.
* The Wigner 3j symbols are computed exactly, in integer and rational
  arithmetic, from Racah's formula (DLMF 34.2.4 [12]) and rounded once
  (:func:`wigner_3j`). The tests check them against the closed form for
  m1 = m2 = m3 = 0 (DLMF 34.3.5), the special value (j j 0; m -m 0), and the
  orthogonality sums (DLMF 34.3.16, 34.3.18).
* w-hat_l is 0 / 0 when q_l is 0 (q_4 of an icosahedron), and a rounding
  residue there is not a value: below :data:`Q_ZERO_TOL` it is NaN, counted
  in the notes.
* With one neighbour, q_lm is one harmonic, and the addition theorem gives
  q_l = 1 and w-hat_l = (l l l; 0 0 0) whatever the direction: the values
  of perfect order, measuring nothing. They are NaN for such atoms (a
  non-bridging O with one bonded Si, in a bond list), counted in the notes;
  the q_lm themselves are kept and enter the neighbours' averaged forms.

The ideal-shell values that are usually quoted were not copied into the tests.
They are computed in the tests by two routes that share no code with this
module: spherical harmonics written out from the associated Legendre
polynomials, and the addition theorem, which needs only Legendre polynomials
(q_l^2 = (1 / N_b^2) sum_jk P_l(r_j . r_k), DLMF 14.30.9 [12]); w_l by the
Gaunt integral of the bond density cubed over the sphere (DLMF 34.3.22).

TETRAHEDRAL ORDER
-----------------
q_tet = 1 - (3 / 8) sum_{j<k} (cos psi_jk + 1/3)^2 over the six angles the four
neighbours make at the centre [3]: 1 for a regular tetrahedron, 0 on average
for four independent random directions. Chau and Hardwick's S_g [4] is the
same sum times 3 / 32, so q_tet = 1 - 4 S_g. It is defined for four
neighbours. ``selection='exactly four'`` takes the atoms whose list holds
exactly four; ``'four nearest'`` takes the four nearest entries of the list
of every atom with four or more. Errington and Debenedetti took the four
nearest molecules of water [3], which is this with a list from
:func:`neighbours_from_pairs` whose cutoff holds at least four; a bond list
holds bonded counter-ions only, and the ranking is within it (the
definition of the result says so). The other atoms are NaN and are counted,
per element and per neighbour count.

POLYHEDRON DISTORTION, AND HOW IT MATCHES THE CRYSTAL SIDE
----------------------------------------------------------
The polyhedron of an atom is its neighbour list. Each formula is that of
``polyhedra.shape`` or ``polyhedra.effective_cn``, the functions the crystal
path calls on the bonded contacts of a site (``coordination.py:385-387``), so
a crystal read as a model gives the same numbers;
``tests/test_md_order.py`` compares every atom against those functions to
1e-12. The crystal functions loop in Python, one site at a time, and
measured 3.1 ms (``shape``, octahedron) and 0.22 ms (``effective_cn``) per
call on this machine; here the same arithmetic runs on all atoms of one CN at
once.

* Baur's distortion index [5]: mean |d_i - d_mean| / d_mean, every CN.
* Bond-angle variance and quadratic elongation [6], for six neighbours
  exactly as ``polyhedra.shape``: the three trans pairs are the perfect
  matching of largest total angle among the fifteen (``_trans_pairs``, the
  same enumeration order, so a tie resolves the same way), sigma^2 =
  sum (theta_cis - 90)^2 / 11, <lambda> = mean (d_i / d0)^2 with d0 the
  centre-to-vertex distance of the regular octahedron of the same volume,
  (3 V / 4)^(1/3). ``polyhedra.shape`` stops at six. For four neighbours this
  module uses Robinson, Gibbs and Ribbe's tetrahedral form [6]: sigma^2 =
  sum (theta - theta_0)^2 / 5 over the six angles, theta_0 = arccos(-1/3)
  (109.4712 deg; Robinson et al. print 109.47), and d0 = (9 sqrt(3) V / 8)^(1/3)
  for the regular tetrahedron. Other CNs give NaN for both.
* The trans pairs of six neighbours are those of ``polyhedra.shape`` for
  every polyhedron, so a crystal read as a model gives the crystal's
  numbers. In a glass the hull of six neighbours is often not the
  octahedron of those pairs: it is not an octahedron at all, or it is one
  whose three non-edges are another matching (bonds at the default v_bond,
  frames 0 and 19 of three Na-bearing glass models: 16 of 29 and 13 of 22
  six-coordinated Na of a Na-aluminosilicate had a hull that is not an
  octahedron, 17 of 45 and 18 of 48 of a Na-borosilicate, 1 of 11 and 3 of
  15 of Na2O-3SiO2, and two Na of the borosilicate's frame 19 an octahedral
  hull of other trans pairs; measured 2026-10-07). sigma^2 and <lambda> are
  then still computed with the largest-angle matching, and the twelve
  angles used are not the hull's edges; ``octahedral_hull`` is False for
  those atoms and the notes count both kinds (the hull test below, applied
  to all fifteen matchings).
* The volume is the convex hull of the ligands. ``polyhedra.shape`` calls
  scipy's ``ConvexHull`` (Qhull [13]), 1.8 ms a call here. For four ligands
  the hull is the tetrahedron, |det| / 6. For six, the octahedron spanned by
  the three trans pairs is tested: when every one of its eight faces has the
  other three ligands strictly on one side, it is convex and is the hull, and
  its volume is the sum of the eight tetrahedra on the ligand centroid. A
  polyhedron that fails the test (a ligand inside the others' hull, or a
  hull of another shape) is passed to ``polyhedra.shape`` itself, which
  calls ``ConvexHull``.
  A volume of 0, or a Qhull failure, gives NaN quadratic elongation, as
  ``polyhedra.shape``'s ``None``.
* ECoN [7]: the self-consistent iteration of ``polyhedra.effective_cn``, run
  on every atom of one CN together, each stopping at the step where its own
  iteration stops (``|d_av' - d_av| < 1e-11``, at most 300 steps).

VORONOI CELLS OF A PERIODIC FRAME
---------------------------------
scipy's ``Voronoi`` (Qhull [13]) is not periodic, so it is given the atoms and
their periodic images within a margin m of the box, the copies
``bulk._image_points`` builds for the pair search, and only the cells of the
atoms themselves are kept. **Why a cell from the padded set is the periodic
cell.** The cell of atom i is cut only by atoms within twice its circumradius
rho_i (the largest distance from i to a vertex of its cell): the bisector of
a farther atom lies beyond every point of the cell. The padded set holds every
image within m of the box, hence within m of atom i. A cell computed from a
subset is never smaller than the true one, so if its rho_i satisfies
2 rho_i <= m, no missing atom can cut it, and it is the true cell. This is
checked for every atom, after every tessellation; when it fails, or a cell is
open, the margin is enlarged towards the reach measured, by a factor of 1.25
to 2 per step (a cell at the edge of too thin a padded set can be far larger
than any true cell: on a 27-atom, 7 Å box, the cells of a 1.0 Å margin
asked for 139 Å, 1.8 million image points, where the true cells needed
4.5 Å; measured), and the tessellation is redone. The margin starts at
:data:`MARGIN_START_SPACINGS` mean atomic spacings (V / N)^(1/3) times two,
a choice that only sets how often a second tessellation is needed: on 16
frames of four oxide-glass models of 2 880 to 3 000 atoms (SiO2,
Na2O-3SiO2, NAS, NBS) the Voronoi cells needed 2.8 to 3.2 spacings and the
Delaunay spheres 1.4 to 1.6, and at 1.7 every frame took one tessellation
of each kind
(measured 2026-10-07; the jittered grid of the benchmark box below needs
1.8 and 0.9). The margin is one length on every axis, so a slab with a
vacuum gap takes several, each larger, and the results are unchanged: the
3 000-atom Na2O-3SiO2 model with a 60 Å gap took four Voronoi
tessellations, the last at a 91 Å margin with about 115 times the atoms in
the padded set, 47 s and 2.4 GB of commit above the process, and four
Delaunay ones, 4.8 s and 0.53 GB (measured 2026-10-07 at the margin start
of 1.7; at the 1.2 used before, the Voronoi cells ended at a 64 Å margin,
179 s and 1.2 GB, measured by the verification of that day under heavier
load). A margin per axis would cut this; it is not done.

A cell's volume is sum over faces of area x (d / 2) / 3, d the distance to
the neighbour across the face, since every face lies on the bisector plane.
The cell volumes summing to the box volume is a law, and it is checked on
every call: a difference above :data:`VOLUME_LAW_TOL` (relative) is refused.

**Faces and edges.** A face polygon is ordered by angle about its centroid in
its own plane; its area is the shoelace sum. Qhull merges the Voronoi
vertices of an exactly degenerate arrangement (several atoms on one sphere,
as in a perfect lattice: no two vertices of one face were closer than 1.0 Å
on fcc, bcc, sc and hcp, measured), but positions a rounding away from such
an arrangement give vertices a rounding apart, joined by edges of that
size. An edge at or below the degeneracy floor, :data:`DEGENERATE_REL_TOL`
times the mean spacing, is contracted: its two vertices, and every vertex
joined to them by such an edge, become one vertex. A face's edges are then
those between distinct vertices, and a face left with fewer than three is
not a face. **Why contraction:** it keeps every cell a polyhedron, so
Euler's relation holds for it, sum_k (6 - k) n_k = 12 when every vertex
joins three edges, more when some join more, never less (12 for every cell
of fcc rattled by 1e-5 Å, checked in the tests against the Delaunay
duality). A floor on face area does not: a face dropped for its area keeps
its edges in the neighbouring faces' counts. Such a floor (faces below
DEGENERATE_REL_TOL spacing^2) dropped real faces of 5e-9 Å^2 with edges of
6e-5 Å on glass models, leaving cells whose sum was 9, which no polyhedron
has; with contraction those cells, and every cell of four glass frames and
of the 3 375-atom benchmark box, have the index OVITO gives (measured
2026-10-07). A cell whose sum is still below 12 is noted. The two
thresholds of the Voronoi index literature [9] are arguments,
``min_face_area_ang2`` and ``min_edge_ang``, 0 by default (no filtering),
and both are stored with the result. They change the face count and the
index, never the volume. An area threshold removes faces whose edges the
neighbouring faces still count, so a thresholded index can have sums
below 12, and the notes count those cells. A face with fewer than three
edges above ``min_edge_ang`` stays in ``n_faces`` and in no index column;
``edge_counts.sum(axis=1)`` leaves it out, as OVITO's coordination does
(:func:`voronoi_cn_distribution`). An atom on another atom has no cell
(Qhull returns no face for it) and is refused; a Voronoi neighbour nearer
than ``bulk.D_MIN_ANG`` is noted.

DELAUNAY VOIDS AND FREE VOLUME
------------------------------
The Delaunay tetrahedra come from scipy's ``Delaunay`` on the same padded
set. A tetrahedron's circumsphere holds no atom centre, so its circumcentre
is a local void [10]. The tetrahedra kept are those whose circumcentre lies in
the box, one copy of each periodic tetrahedron (copies are also removed by a
key independent of the image, and counted); **why they are the periodic
Delaunay tetrahedra**: one whose circumsphere lies within the padded region
is empty of every atom of the periodic frame, so it is a true Delaunay
tetrahedron, and every true one has an image with its circumcentre in the
box. A circumcentre in the box with circumradius R <= m has its sphere inside
the padded region; R > m enlarges the margin. The tetrahedra filling the
box volume is checked as for the Voronoi cells. Tetrahedra of zero volume
(Qhull's triangulation of cospherical atoms) have no circumsphere and are
counted, not kept. Every point of a set of distinct points is a vertex of
its Delaunay triangulation, and Qhull leaves out a point that lies on
another one, so an atom that is a vertex of no tetrahedron is refused, as
:func:`voronoi_cells` refuses it; each atom's nearest neighbour is joined to
it by a Delaunay edge, so an edge shorter than ``bulk.D_MIN_ANG`` is noted
with its rows.

The empty sphere of a tetrahedron is the largest sphere centred on its
circumcentre that overlaps no atom sphere: min over all atoms of
(distance - radius), from a KD-tree over every atom within reach, not only
the four vertices. With one radius r for every atom this is R - r, the
interstitial sphere of the tetrahedron. A negative value means the
circumcentre lies inside an atom sphere, by that much; it is kept, signed,
and counted. With radii 0 it is R itself. That the nearest atom centre lies
at R is the Delaunay property itself, and it is checked.

Free volume is measured on a grid of points, the box divided into
ceil(|a_k| / h) cells along each cell vector. A point is outside an atom
when its distance to the atom exceeds the atom's radius. The *geometric*
fraction is the part of the box outside every atom sphere; the *probe-centre*
fraction is the part where a probe sphere of radius r_p can be centred
without overlapping any atom (outside every expanded sphere, of radius
R = r + r_p: the set C of probe centres); the *probe-occupiable* fraction is
the part a probe sphere can cover, the points within r_p of a point of C.
The terms follow Ongari et al. [11], whose abstract names geometric,
probe-centre and probe-occupiable pore volumes; no connectivity analysis is
made, so every pocket counts, reachable from another or not. Each grid
point is classified exactly for all three fractions; the grid is the only
approximation, and the result states its spacing.

**Why the probe-occupiable test is exact.** A free point x outside C is
covered when the point of C nearest to x is within r_p. That point lies on
the boundary of C, made of patches of expanded spheres, arcs of the circles
where two meet, and points where three meet. On a patch it is the radial
projection of x onto the sphere, on an arc the point of the circle nearest
to x (the only local minima of the distance on a sphere and on a circle),
and otherwise a three-sphere point. So x is covered exactly when one of
those candidates lies within r_p of x and outside every expanded sphere,
each to DEGENERATE_REL_TOL spacings. The three-sphere points of C are found
once per frame, from every triple of expanded spheres that meet pairwise,
or, when the spheres are large and each meets a hundred others, from the
triangles of the regular (power) triangulation only: a point of C on
spheres i, j and k has power |v - a|^2 - R^2 equal to 0 for those three and
0 or more for all others, so it lies in all three power cells and (i, j, k)
is a triangle of that triangulation [18], the lower hull of the centres
lifted to (a, |a|^2 - R^2) (scipy's ``ConvexHull``). On a Na2O-3SiO2 glass
with van der Waals radii and a 1.4 Å probe that is 26 820 triples against
1 837 100 (measured). Only spheres that touch C give the one- and
two-sphere candidates. A point on the axis of a circle sees the whole
circle at one distance and takes one of its points: an arc of C on that
circle ends at three-sphere points, which are tried anyway.

The grid dilation used before (grid points within r_p of a probe-centre
*grid* point) lost the thin wedges and small pockets of C between grid
points: on a Na2O-3SiO2 glass model (half van der Waals radii, r_p = 0.5 Å)
it read 0.656 where the exact fraction of the same 343 000 points is 0.743,
moved irregularly with h, and the same structure in a sheared cell basis
read up to 0.03 differently (the verification of 2026-10-07). The exact
test gives 0.74258 on those points, the value of the verification's own
per-point oracle, and 0.7426, 0.7423, 0.7424 at h = 0.5, 0.35, 0.25 Å in
the cubic basis against 0.7425, 0.7424, 0.7425 in the sheared one; on a
dense random box it equals that oracle, to the five decimals the oracle
printed, at h = 0.45, 0.3 and 0.15 Å (measured 2026-10-07). On the
benchmark box below the grid dilation read 0.512 where the exact value of
the same points is 0.690 (h = 0.5 Å, half van der Waals radii,
r_p = 0.5 Å). The tests check all three fractions against a
classification of every grid point by brute force, written there with a
different exact criterion (the highest point of the probe ball outside the
spheres), in a sheared and an orthogonal box, with a probe large enough
that most points need two- and three-sphere candidates and through both
routes to the three-sphere points, and against the closed forms of
separated spheres, where the probe-occupiable points are exactly the free
ones.

**Radii.** No radius has a default. Either the caller gives one per element
with its source, or :func:`vdw_radii_ang` reads FACET's van der Waals radii
(``elements.info().vdw_radius``, which is gemmi's ``Element.vdw_r``), whose
source is :data:`VDW_RADII_SOURCE`. They are non-bonded contact radii: a
bonded Si-O at 1.61 Å is 2.0 Å shorter than the sum of the two radii in that
table (2.10 + 1.52 Å), so on an oxide the spheres overlap heavily and the
free fractions come out small. That is a property of the radii, and the
choice of radii is the caller's.

NOT DONE HERE
-------------
* Radical (Laguerre) Voronoi cells, which weight the bisector by atom radii:
  scipy has no power diagram (the regular triangulation of the free-volume
  step lists triangles only, never cells), and FACET holds no radius set
  that has been checked for that purpose. Not implemented.
* Face-area-weighted q_l (Minkowski structure metrics [19]) and the
  solid-bond counts built on q_6 . q_6: the per-atom q_lm are returned
  (:attr:`BondOrder.qlm`), so either can be formed from them; neither is
  computed here.
* Connectivity or percolation of the free volume.

OUTSIDE CHECKS (throwaway environments, never imported by FACET or its tests)
-----------------------------------------------------------------------------
Run 2026-10-06 on two synthetic frames, the Step 0 benchmark box at 3 375
atoms (cubic, a jittered 2.3 Å grid) and a 1 000-atom box sheared by
0.30, 0.20, 0.25 L, with every pair within 3.0 Å a neighbour:

* **freud 3.4.0** [16]: the same neighbour count for every atom; q_l and
  w-hat_l for l = 4, 6, 8 agreed to 4.7e-5 at worst (median 4e-7), the
  averaged forms to 7.6e-6; the Voronoi volumes to 3.1e-6 (relative). freud
  holds positions as 32-bit floats, which is the size of these differences.
  freud's Voronoi (voro++) listed one face fewer than this module for 180 of
  the 3 375 nearly-cubic atoms. Those cells were then computed a third way,
  as the intersection of the bisector half-spaces of every image within
  12 Å (scipy ``HalfspaceIntersection``, 64-bit): face for face, the same
  faces and areas as this module (to 1e-8 Å^2) and volumes to 3e-15.
* **OVITO 3.16.1** [17] ``VoronoiAnalysisModifier``: volumes equal to 3.6e-15
  (relative); the face count and the whole Voronoi index equal for all
  1 000 atoms of the sheared box and 3 373 of the 3 375 of the cubic one.
  The two others shared a real face of 2.1e-11 Å^2 that OVITO counts and
  the face-area floor of the time left out (the half-space intersection
  found it too; 2.1e-11 rounds to 0.0 at eight decimals). Since the edge
  contraction above, all 3 375 equal OVITO's (measured 2026-10-07).

Run 2026-10-07 on frame 0 of four LAMMPS oxide-glass models of 2 880 to
3 000 atoms (SiO2, Na2O-3SiO2, Na-aluminosilicate and Na-borosilicate):

* **OVITO 3.16** ``VoronoiAnalysisModifier``: the Voronoi index equal for
  every atom of all four frames with no threshold, and on SiO2 and NAS with
  (area, edge) thresholds (0.1 Å^2, 0), (0, 0.1 Å), (0.1 Å^2, 0.1 Å) and
  (0.5 Å^2, 0.2 Å); OVITO's coordination equal to
  ``edge_counts.sum(axis=1)`` in every case.
* The verification of the same day (three independent reviews) also
  compared, with tools of its own: q_l, w-hat_l and their averages with
  pyscal3 4.0.0 in double precision (to 1e-14, l = 2 to 12, six frames)
  and freud 3.6.1; the Delaunay tetrahedra with scipy's Delaunay of a
  27-copy replica; the polyhedron distortion with a separate neighbour list
  and ``ConvexHull``; the free-volume fractions with Monte Carlo and an
  exact per-point oracle (above).

TIMINGS
-------
Measured 2026-10-06 on Windows 11, Intel i5-13420H, Python 3.11.9, numpy
2.4.6, scipy 1.15.1, unpinned, with other sessions keeping the machine at
100 % load (single runs there vary by 30 % or more); median [min-max] of 5
runs after one untimed run, on ``tools/bench_md.make_box(22)``: 10 648
atoms, a jittered 2.3 Å grid in a 50.6 Å cubic box, Si/O/O/Na at random
(the chemistry means nothing), bonds from ``bulk.analyse_frame`` at the
default thresholds (15 222 bonds, 30 444 list entries):

    neighbours_from_bonds                  0.019 s [0.018-0.021]
    neighbours_from_pairs (2 cutoffs)      0.032 s [0.027-0.034]
    steinhardt, l = 4 and 6                1.00 s  [0.79-1.20]
    steinhardt, l = 12                     1.95 s  [1.59-4.05]
    tetrahedral_order                      0.021 s [0.019-0.023]
    polyhedron_shape                       0.046 s [0.046-0.050]
    voronoi_cells                          2.29 s  [2.19-2.41]
    empty_spheres                          1.72 s  [1.61-1.88]

Re-measured 2026-10-07 after the changes of that day, the same machine and
box under heavier load from other sessions (the verification of that day
could not reproduce the rows above either: 6.8-10 s for ``steinhardt`` and
22.7 s for ``voronoi_cells`` in its runs), median [min-max] of 3 runs after
one untimed run, single runs where no range is given:

    steinhardt, l = 4 and 6                1.65 s  [0.71-2.31]
    steinhardt, l = 12                     3.18 s  [2.90-4.74]
    polyhedron_shape                       0.25 s  [0.13-0.29]
    voronoi_cells                          5.01 s  [4.26-5.09]
    empty_spheres                          3.90 s  [3.82-4.09]
    free_volume, h = 0.5 Å  (1.06 M pts)   8.14 s  [8.06-10.51]
    free_volume, h = 0.25 Å (8.37 M pts)   45.1 s

(free_volume with half the van der Waals radii and a 0.5 Å probe; the grid
dilation it replaced took 1.0-1.5 s at h = 0.5 Å, back to back, for a
probe-occupiable fraction of 0.512 where the exact one is 0.690.) On the
3 000-atom Na2O-3SiO2 glass model at h = 0.25 Å, single runs, cubic cell
(2.69 M points) and the same structure in a sheared basis (3.66 M points):

    half vdW radii, r_p = 0.5 Å            10.0 s     14.9 s
    half vdW radii, r_p = 0.9 Å            38.1 s     49.8 s
    full vdW radii, r_p = 1.4 Å            34.8 s     34.2 s

where the grid dilation took 6.5 s (cubic) and 105 s (the verification's
own sheared basis, 6.54 M points, a KD-tree query per grid point) for the
first row, measured by the verification of 2026-10-07. Every Voronoi and
Delaunay call on 16 frames of four glass models took one tessellation,
1.1-2.3 s and 0.5-1.2 s. Under a profiler,
scipy's ``sph_harm_y`` was half of ``steinhardt``; Qhull was 1.2 s of
``voronoi_cells`` (17 576 points with the images: one tessellation, at a
5.5 Å margin where the cells needed 4.1 Å; 2026-10-06, margin start 1.2
spacings) and 0.8 s of ``empty_spheres`` (2.8 Å margin, 2.0 Å needed;
74 804 tetrahedra). Before the polygons were grouped by vertex count the
face geometry took 1.6 s. In the exact probe-occupiable step the
heavy-overlap case (full radii, 1.4 Å probe) spent most of its time
gathering each shell point's spheres: walking every grid point within
r + 2 r_p of every atom took 142 s (cubic) and 317 s (sheared) for the row
above, a KD-tree of the shell points 35 s. For comparison, the crystal
path's ``polyhedra.shape`` took 3.1 ms per octahedron and ``effective_cn``
0.22 ms per site here: 30 s and 2.3 s for these atoms one by one.

REFERENCES
----------
[1] P. J. Steinhardt, D. R. Nelson and M. Ronchetti, "Bond-orientational
    order in liquids and glasses", *Physical Review B* 28 (1983) 784-805,
    https://doi.org/10.1103/PhysRevB.28.784
[2] W. Lechner and C. Dellago, "Accurate determination of crystal structures
    based on averaged local bond order parameters", *The Journal of Chemical
    Physics* 129 (2008) 114707, https://doi.org/10.1063/1.2977970
[3] J. R. Errington and P. G. Debenedetti, "Relationship between structural
    order and the anomalies of liquid water", *Nature* 409 (2001) 318-321,
    https://doi.org/10.1038/35053024
[4] P.-L. Chau and A. J. Hardwick, "A new order parameter for tetrahedral
    configurations", *Molecular Physics* 93 (1998) 511-518,
    https://doi.org/10.1080/002689798169195
[5] W. H. Baur, "The geometry of polyhedral distortions. Predictive
    relationships for the phosphate group", *Acta Crystallographica* B30
    (1974) 1195-1215, https://doi.org/10.1107/S0567740874004560
[6] K. Robinson, G. V. Gibbs and P. H. Ribbe, "Quadratic elongation: a
    quantitative measure of distortion in coordination polyhedra", *Science*
    172 (1971) 567-570, https://doi.org/10.1126/science.172.3983.567
[7] R. Hoppe, "Effective coordination numbers (ECoN) and mean fictive ionic
    radii (MEFIR)", *Zeitschrift für Kristallographie* 150 (1979) 23-52,
    https://doi.org/10.1524/zkri.1979.150.1-4.23
[8] J. L. Finney, "Random packings and the structure of simple liquids. I.
    The geometry of random close packing", *Proceedings of the Royal Society
    of London A* 319 (1970) 479-493, https://doi.org/10.1098/rspa.1970.0189
[9] H. W. Sheng, W. K. Luo, F. M. Alamgir, J. M. Bai and E. Ma, "Atomic
    packing and short-to-medium-range order in metallic glasses", *Nature*
    439 (2006) 419-425, https://doi.org/10.1038/nature04421
[10] S. Sastry, D. S. Corti, P. G. Debenedetti and F. H. Stillinger,
    "Statistical geometry of particle packings. I. Algorithm for exact
    determination of connectivity, volume, and surface areas of void space in
    monodisperse and polydisperse sphere packings", *Physical Review E* 56
    (1997) 5524-5532, https://doi.org/10.1103/PhysRevE.56.5524
[11] D. Ongari, P. G. Boyd, S. Barthel, M. Witman, M. Haranczyk and B. Smit,
    "Accurate characterization of the pore volume in microporous crystalline
    materials", *Langmuir* 33 (2017) 14529-14538,
    https://doi.org/10.1021/acs.langmuir.7b01682
[12] NIST Digital Library of Mathematical Functions, §14.30 (spherical
    harmonics), §34.2-34.3 (3j symbols), https://dlmf.nist.gov/
[13] C. B. Barber, D. P. Dobkin and H. Huhdanpaa, "The quickhull algorithm
    for convex hulls", *ACM Transactions on Mathematical Software* 22 (1996)
    469-483, https://doi.org/10.1145/235815.235821
[14] A. Bondi, "van der Waals volumes and radii", *The Journal of Physical
    Chemistry* 68 (1964) 441-451, https://doi.org/10.1021/j100785a001
[15] M. Mantina, A. C. Chamberlin, R. Valero, C. J. Cramer and D. G. Truhlar,
    "Consistent van der Waals radii for the whole main group", *The Journal
    of Physical Chemistry A* 113 (2009) 5806-5812,
    https://doi.org/10.1021/jp8111556
[16] V. Ramasubramani, B. D. Dice, E. S. Harper, M. P. Spellings, J. A.
    Anderson and S. C. Glotzer, "freud: A software suite for high throughput
    analysis of particle simulation data", *Computer Physics Communications*
    254 (2020) 107275, https://doi.org/10.1016/j.cpc.2020.107275
[17] A. Stukowski, "Visualization and analysis of atomistic simulation data
    with OVITO - the Open Visualization Tool", *Modelling and Simulation in
    Materials Science and Engineering* 18 (2010) 015012,
    https://doi.org/10.1088/0965-0393/18/1/015012
[18] H. Edelsbrunner, "The union of balls and its dual shape", *Discrete &
    Computational Geometry* 13 (1995) 415-440,
    https://doi.org/10.1007/BF02574053
[19] W. Mickel, S. C. Kapfer, G. E. Schröder-Turk and K. Mecke,
    "Shortcomings of the bond orientational order parameters for the
    analysis of disordered particulate matter", *The Journal of Chemical
    Physics* 138 (2013) 044501, https://doi.org/10.1063/1.4774084
"""
from __future__ import annotations

import functools
import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import bulk, polyhedra
from . import elements as element_data
from .md_model import Frame
from .md_stats import Distribution, Histogram, Scalar

__all__ = [
    "DEGREES_DEFAULT", "DEGREE_MAX", "Q_ZERO_TOL", "TETRAHEDRAL_ANGLE_DEG",
    "OCTAHEDRAL_CIS_ANGLE_DEG", "QTET_SELECTIONS", "DEGENERATE_REL_TOL",
    "VOLUME_LAW_TOL", "MARGIN_START_SPACINGS", "MARGIN_ATTEMPTS",
    "FACE_SNAP_FRAC",
    "GRID_POINTS_LIMIT", "VDW_RADII_SOURCE",
    "Neighbours", "neighbours_from_bonds", "neighbours_from_pairs",
    "neighbours_from_voronoi",
    "wigner_3j", "BondOrder", "steinhardt",
    "TetrahedralOrder", "tetrahedral_order",
    "PolyhedronShape", "polyhedron_shape",
    "VoronoiCells", "voronoi_cells",
    "EmptySpheres", "empty_spheres",
    "FreeVolume", "free_volume", "vdw_radii_ang",
    "element_histograms", "voronoi_index_distribution",
    "voronoi_cn_distribution", "empty_sphere_histogram",
    "free_volume_scalars",
]

# ---------------------------------------------------------------------------
# constants: each one is a method choice or a numerical tolerance, and says so
# ---------------------------------------------------------------------------

# The degrees l computed when none are given. A method choice, not a physical
# value: l = 4 and 6 are the pair Steinhardt et al. tabulate for the cubic,
# hexagonal and icosahedral shells [1], and the ones most papers report.
DEGREES_DEFAULT: tuple[int, ...] = (4, 6)

# The largest degree accepted. A cost bound: w_l sums about (2l + 1)^2 * 3 / 4
# products per atom and Y_lm is evaluated 2l + 1 times per bond.
DEGREE_MAX: int = 32

# Below this q_l, w-hat_l (a ratio of two quantities that vanish together) is
# NaN. Numerical: the q_4 of an exact icosahedron came out at 1e-16 here.
Q_ZERO_TOL: float = 1e-10

# arccos(-1/3): the angle of a regular tetrahedron, exact rather than the
# 109.47 printed by Robinson et al. [6].
TETRAHEDRAL_ANGLE_DEG: float = math.degrees(math.acos(-1.0 / 3.0))
OCTAHEDRAL_CIS_ANGLE_DEG: float = 90.0

QTET_SELECTIONS = ("exactly four", "four nearest")

# The degeneracy floor, a length in mean atomic spacings: a Voronoi face
# edge at or below it is contracted (its vertices merge; module docstring),
# a Delaunay tetrahedron below it times spacing^3 (a slab that thin over a
# spacing^2 face) is Qhull's rendering of a degeneracy, and the candidate
# probe centres of the free volume are tested to it. Numerical: on perfect
# fcc, bcc, sc and hcp lattices no two vertices of one Voronoi face were
# closer than 1.0 Å, and the degenerate tetrahedra were at 1e-17 spacing^3
# (measured); positions within about this of a degenerate arrangement are
# read as that arrangement.
DEGENERATE_REL_TOL: float = 1e-9

# The cell volumes, or the tetrahedron volumes, differing from the box volume
# by more than this (relative) is refused: the tessellation would not be the
# periodic one. Numerical: the measured differences were 1e-15 to 1e-13.
VOLUME_LAW_TOL: float = 1e-9

# The first margin is this many mean atomic spacings (V / N)^(1/3), doubled
# for Voronoi cells (which need 2 rho), and enlarged when the check fails. A
# cost choice: it never changes a result, only how often a tessellation is
# redone. Set from four oxide-glass models (SiO2, Na2O-3SiO2, NAS, NBS):
# their Voronoi cells needed up to 3.2 spacings and their Delaunay spheres
# up to 1.55, so 1.2 (the first value, tuned on a jittered grid that needed
# 1.8 and 0.9) made every glass frame tessellate twice (measured).
MARGIN_START_SPACINGS: float = 1.7

# How many tessellations are tried before the margin search gives up. Each
# retry at most doubles the margin (:func:`_next_margin`), so the last one
# tried is up to 2^7 = 128 times the first.
MARGIN_ATTEMPTS: int = 8

# A circumcentre whose fractional coordinate lies within this of a whole
# number is taken to be on that box face. Numerical: the copies of one
# tetrahedron give fractions that differ from whole numbers apart by
# rounding (1e-15 here), and without the snap a centre on a face could be
# kept from neither copy (an fcc lattice with a hole on the face lost 7 % of
# the box volume that way, measured) or from both.
FACE_SNAP_FRAC: float = 1e-9

# Bonds per chunk when evaluating Y_lm, bytes of one gathered q_lm array in
# the w_l sum (the atoms per chunk follow from the number of 3j triples, so
# the peak does not grow with l: before, a fixed 20 000 atoms per chunk
# peaked at 1.6 GB at l = 32, measured), and work elements per chunk in the
# free-volume grid: memory choices, never in a result.
_BONDS_PER_CHUNK: int = 100_000
_W_CHUNK_BYTES: int = 32_000_000
_GRID_WORK_PER_CHUNK: int = 2_000_000
_RIDGES_PER_CHUNK: int = 50_000

# (grid point, sphere) pairs gathered per chunk of shell points for the
# exact probe-occupiable test, and (candidate, sphere) tests per block in
# it: memory choices, never in a result (a pair holds about 60 bytes with
# its vector and distance, and sorting copies it once; GRID_POINTS_LIMIT
# gives the peaks measured at these values).
_SHELL_PAIRS_PER_SLAB: int = 500_000
_EXACT_WORK_PER_CHUNK: int = 1_000_000

# Where the candidate three-sphere points of the probe-centre set come from
# (_candidate_triples): "auto" takes every triple of meeting spheres (the
# pair graph) while that tries at most _PAIR_GRAPH_PAIRS_PER_ATOM pairs of
# neighbours per atom, and the triangles of the regular triangulation (a
# lifted 4-D hull) above; "pair graph" and "regular triangulation" force
# one. Cost choices: both routes give the same points (the tests run both).
# Measured on a 3 000-atom Na2O-3SiO2 glass: with half van der Waals radii
# and a 0.5 Å probe the pair graph formed 2 923 triples and the hull, which
# cost about 3 s alone, 2 705; with full radii and a 1.4 Å probe the pair
# graph formed 1 837 100 and the hull 26 820 (115 s against 41 s for the
# whole call, under load).
_TRIPLES_FROM: str = "auto"
_PAIR_GRAPH_PAIRS_PER_ATOM: int = 30

# The most grid points the free-volume grid may hold. A memory choice: per
# grid point the call holds four boolean masks at most (core, blocked, its
# complement for a moment, covered) and the index of each shell point
# (8 bytes for the 30 to 40 % of points in the shell), about 10 bytes; the
# work arrays are bounded by the chunk sizes above. Measured peaks above
# the process before the call (NS3 glass, half van der Waals radii, 0.5 Å
# probe, 2026-10-07): 78 MB at 0.34 M points, 148 MB at 2.7 M, 243 MB at
# 12.3 M (10 bytes per point between the last two), so about 1.1 GB at
# this limit, extrapolated; 215 MB at 0.97 M points with full radii and a
# 1.4 Å probe. (The grid dilation used before peaked at 50 to 140 bytes per
# point, measured by the verification of 2026-10-07; this comment then
# claimed 2.)
GRID_POINTS_LIMIT: int = 100_000_000

VDW_RADII_SOURCE: str = (
    "van der Waals radii from gemmi's Element.vdw_r (elements.info), whose "
    "source (include/gemmi/elem.hpp, gemmi 0.7.1) is the Wikipedia data page "
    "'Atomic radii of the elements', citing A. Bondi, J. Phys. Chem. 68 (1964) "
    "441, https://doi.org/10.1021/j100785a001, and M. Mantina et al., J. Phys. "
    "Chem. A 113 (2009) 5806, https://doi.org/10.1021/jp8111556; values "
    "missing there were taken from cctbx van_der_waals_radii.py. gemmi does not "
    "record which of the three each value came from. Stored by gemmi as 32-bit "
    "floats and rounded here to the two decimals of its table")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _check_frame(frame) -> Frame:
    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}")
    return frame


def _frozen(array) -> np.ndarray:
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def _nonnegative(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name}: a number is needed, not {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: a number is needed, not {value!r}") from None
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} = {value!r}: a finite number of 0 or more is "
                         "needed")
    return out


def _positive(value, name: str) -> float:
    out = _nonnegative(value, name)
    if out == 0.0:
        raise ValueError(f"{name} = {value!r}: a number above 0 is needed")
    return out


def _by_element(symbols: np.ndarray, mask: np.ndarray) -> str:
    """'Si 3, O 2' for the atoms selected by mask, alphabetical."""
    found, counts = np.unique(symbols[mask], return_counts=True)
    return ", ".join(f"{s} {c}" for s, c in zip(found, counts))


def _rows_text(rows: np.ndarray, limit: int = 20) -> str:
    """'3, 17, 40' for a few rows; the first ``limit`` and a count beyond."""
    rows = [int(r) for r in rows]
    shown = ", ".join(str(r) for r in rows[:limit])
    return shown if len(rows) <= limit else \
        f"{shown} and {len(rows) - limit} more"


def _coincident_message(frame: Frame, rows: np.ndarray,
                        what: str = "have no Voronoi cell: Qhull returned no "
                                    "face for them",
                        refused: str = "has no cell of its own; it is "
                                       "refused rather than given volume 0"
                        ) -> str:
    """Why Qhull left atoms out: each lies on another atom. The rows named
    are few (each is a refusal), so the other atom is found by the
    minimum-image distance to every atom, for the message only."""
    parts = []
    for row in rows[:20]:
        others = np.delete(np.arange(frame.n_atoms), row)
        df = frame.frac[others] - frame.frac[row]
        df -= np.round(df)
        d = np.linalg.norm(df @ frame.box_ang, axis=1)
        k = int(np.argmin(d))
        parts.append(f"row {int(row)} at {d[k]:.3g} Å from row {int(others[k])}")
    return (f"{rows.size} atom(s) {what}, which happens when an atom lies "
            f"on another one ({'; '.join(parts)}). A duplicated position "
            f"{refused}")


def _spacing_ang(frame: Frame) -> float:
    """The mean atomic spacing (V / N)^(1/3), the length scale tolerances use."""
    return (frame.volume_ang3 / frame.n_atoms) ** (1.0 / 3.0)


def _frac_of(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Fractional coordinates of origin-free Cartesian points (rows)."""
    return np.linalg.solve(box.T, points.T).T


def _padded(frame: Frame, margin_ang: float):
    """(atom, image, position): the atoms first, in row order, then every
    image within ``margin_ang`` of the box (``bulk._image_points``).
    Positions are origin-free, ``(frac + image) . box``, as in the bulk
    engine."""
    atom, image, position = bulk._image_points(frame, margin_ang)
    home = ~image.any(axis=1)
    order = np.concatenate((np.flatnonzero(home), np.flatnonzero(~home)))
    atom, image, position = atom[order], image[order], position[order]
    n = frame.n_atoms
    if not np.array_equal(atom[:n], np.arange(n)):
        raise RuntimeError("the padded point set does not start with the "
                           "frame's own atoms in row order")
    return atom, image.astype(np.int64), position


def _next_margin(margin: float, need: float) -> float:
    """The margin for the next tessellation: what the last one measured it
    needed, but at least 1.25 and at most 2 times the last margin. The reach
    measured on too thin a margin belongs to the cells at the edge of the
    padded set, which can be far larger than any true cell, so it is never
    taken at face value beyond a doubling."""
    return min(max(need * (1.0 + 1e-6), 1.25 * margin), 2.0 * margin)


def _margin_note(kind: str, attempts: list[tuple[float, float]],
                 given: float | None) -> str:
    tried = "; ".join(f"{m:.4g} Å (reach {r:.4g} Å)" for m, r in attempts)
    start = (f"the margin given, {given:.4g} Å" if given is not None
             else f"a first margin of {attempts[0][0]:.4g} Å")
    return (f"{kind}: periodic images within the margin of the box were "
            f"tessellated with the atoms; {start}; tessellations tried, with "
            f"the reach each needed: {tried}")


# ---------------------------------------------------------------------------
# neighbour lists
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Neighbours:
    """Who each atom's neighbours are, as flat arrays sorted by centre.

    Entry p says that ``nbr[p]``, translated by ``image[p]`` (a lattice
    vector in cell units), is a neighbour of ``centre[p]``, along
    ``vec_ang[p]`` (centre -> neighbour image) at ``d_ang[p]``. Within each
    centre the entries are in ascending distance, ties by neighbour row and
    then image, so the order depends on the geometry only. Atom r's entries
    are ``offsets[r]`` to ``offsets[r + 1]``. ``definition`` says in words
    what made an entry; ``cutoffs_ang`` and ``cutoff_sources`` hold the
    distance cutoffs and where each came from, ``v_bond_vu`` the bond-valence
    threshold, whichever the definition used.
    """

    elements: np.ndarray           # (N,)
    centre: np.ndarray             # (P,) int64 row
    nbr: np.ndarray                # (P,) int64 row
    image: np.ndarray              # (P, 3) int64
    vec_ang: np.ndarray            # (P, 3)
    d_ang: np.ndarray              # (P,)
    offsets: np.ndarray            # (N + 1,) int64
    definition: str
    v_bond_vu: float | None = None
    cutoffs_ang: Mapping = field(default_factory=dict)
    cutoff_sources: Mapping = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def n_atoms(self) -> int:
        return int(self.elements.shape[0])

    @property
    def count(self) -> np.ndarray:
        """(N,) neighbours per atom."""
        return np.diff(self.offsets)

    def __len__(self) -> int:
        return int(self.centre.shape[0])

    def of(self, row: int) -> slice:
        """The slice of the flat arrays that holds atom ``row``'s entries."""
        return slice(int(self.offsets[row]), int(self.offsets[row + 1]))


def _neighbours(elements: np.ndarray, centre, nbr, image, vec, d, *,
                definition: str, v_bond_vu=None, cutoffs_ang=None,
                cutoff_sources=None, notes=()) -> Neighbours:
    n = elements.shape[0]
    centre = np.asarray(centre, np.int64)
    nbr = np.asarray(nbr, np.int64)
    image = np.asarray(image, np.int64).reshape(-1, 3)
    vec = np.asarray(vec, np.float64).reshape(-1, 3)
    d = np.asarray(d, np.float64)
    order = np.lexsort((image[:, 2], image[:, 1], image[:, 0], nbr, d, centre))
    centre, nbr, image, vec, d = (a[order] for a in (centre, nbr, image, vec, d))
    count = np.bincount(centre, minlength=n)
    offsets = np.concatenate(([0], np.cumsum(count))).astype(np.int64)
    note_list = list(notes)
    lonely = count == 0
    if lonely.any():
        note_list.append(
            f"{int(lonely.sum())} of {n} atoms have no neighbour under this "
            f"definition ({_by_element(elements, lonely)}); every quantity "
            "that needs a neighbour is NaN for them")
    return Neighbours(
        elements=_frozen(elements), centre=_frozen(centre), nbr=_frozen(nbr),
        image=_frozen(image), vec_ang=_frozen(vec), d_ang=_frozen(d),
        offsets=_frozen(offsets), definition=definition, v_bond_vu=v_bond_vu,
        cutoffs_ang=dict(cutoffs_ang or {}),
        cutoff_sources=dict(cutoff_sources or {}), notes=tuple(note_list))


def neighbours_from_bonds(frame: Frame, bonds: bulk.Bonds) -> Neighbours:
    """Each atom's bonded counter-ions, from ``bulk.bonds_at``.

    A cation's neighbours are the anions it is bonded to and an anion's the
    cations, so the list holds every bond from both ends and is symmetric.
    The bonds are checked to be this frame's: every bond vector is recomputed
    from the frame's fractions as the bulk engine forms it, and a difference
    above 1e-9 Å is refused.
    """
    frame = _check_frame(frame)
    if not isinstance(bonds, bulk.Bonds):
        raise ValueError(f"bulk.Bonds is needed (from bulk.bonds_at), not "
                         f"{type(bonds).__name__}")
    n = frame.n_atoms
    cation = np.asarray(bonds.cation, np.int64)
    anion = np.asarray(bonds.anion, np.int64)
    image = np.asarray(bonds.image, np.int64)
    if cation.size and (max(cation.max(), anion.max()) >= n
                        or min(cation.min(), anion.min()) < 0):
        raise ValueError(f"the bonds name atom rows outside 0..{n - 1}; they "
                         "are not this frame's")
    vec, d = bulk._pair_geometry(frame.frac, frame.box_ang, cation, anion,
                                 image)
    if cation.size:
        worst = float(np.abs(vec - bonds.vec_ang).max())
        if not worst <= 1e-9:
            raise ValueError(
                f"the bond vectors differ by up to {worst:.3g} Å from this "
                "frame's geometry; the bonds were measured on another frame")
    definition = (f"bonds with valence above v_bond = {bonds.v_bond_vu:g} v.u. "
                  "(bulk.bonds_at, the CN's bond definition): each cation to "
                  "its bonded anions, each anion to its bonded cations")
    return _neighbours(
        frame.elements, np.concatenate((cation, anion)),
        np.concatenate((anion, cation)), np.concatenate((image, -image)),
        np.concatenate((bonds.vec_ang, -np.asarray(bonds.vec_ang))),
        np.concatenate((bonds.d_ang, bonds.d_ang)), definition=definition,
        v_bond_vu=float(bonds.v_bond_vu))


def neighbours_from_pairs(frame: Frame, pairs,
                          cutoff_ang: Mapping[tuple[str, str], float],
                          cutoff_sources: Mapping[tuple[str, str], str]
                          ) -> Neighbours:
    """Neighbours within a distance cutoff per (centre, neighbour) element pair.

    ``pairs`` is the frame's one pair search (``bulk.find_pairs``, or the
    blocks of ``bulk.iter_pairs`` as a list), with vectors kept to at least
    the largest cutoff. A neighbour counts when d <= cutoff, inclusive as
    ``bulk.distance_cn``. Every cutoff needs its source in ``cutoff_sources``
    (the g(r) minimum it was read from, say), and no cutoff has a default.
    The pairs are checked to be this frame's and to cover every atom once.
    A cutoff for (A, B) and none for (B, A) gives a list that is not
    symmetric, which the averaged q-bar_l notices (a note counts it).
    """
    frame = _check_frame(frame)
    if not isinstance(cutoff_ang, Mapping) or not cutoff_ang:
        raise ValueError("cutoff_ang needs at least one (centre element, "
                         "neighbour element) -> cutoff in Å; none is assumed")
    if not isinstance(cutoff_sources, Mapping):
        raise ValueError("cutoff_sources needs a mapping of the same keys to "
                         "the source of each cutoff")
    cutoffs: dict[tuple[str, str], float] = {}
    sources: dict[tuple[str, str], str] = {}
    for key, value in cutoff_ang.items():
        if not (isinstance(key, tuple) and len(key) == 2
                and all(isinstance(k, str) for k in key)):
            raise ValueError(f"cutoff key {key!r}: a (centre element, "
                             "neighbour element) pair of symbols is needed")
        cutoffs[key] = _positive(value, f"the cutoff for {key[0]}-{key[1]}")
        source = cutoff_sources.get(key)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"the cutoff for {key[0]}-{key[1]} has no source; "
                             "cutoff_sources needs one for every cutoff")
        sources[key] = source.strip()
    extra = set(cutoff_sources) - set(cutoffs)
    if extra:
        raise ValueError(f"cutoff_sources names pairs with no cutoff: "
                         f"{sorted(extra)}")

    symbols = frame.elements
    species = frame.species
    code = np.searchsorted(np.array(species), symbols)
    table = np.full((len(species), len(species)), -np.inf)
    notes = []
    for (a, b), value in cutoffs.items():
        if a in species and b in species:
            table[species.index(a), species.index(b)] = value
        else:
            notes.append(f"the cutoff for {a}-{b} selects nothing: the frame "
                         f"holds no {a if a not in species else b}")

    coverage = bulk._Coverage(frame, frame.n_atoms)
    parts = []
    for block in bulk._blocks_of(pairs):
        coverage.add(block)
        largest = max(cutoffs.values())
        if largest > block.r_ang:
            raise ValueError(f"a cutoff of {largest} Å is beyond the "
                             f"{block.r_ang} Å the pairs were searched to")
        sel = block.d_ang <= table[code[block.i], code[block.j]]
        if not sel.any():
            continue
        if block.vec_ang is None:
            raise ValueError("the pairs were searched without vectors "
                             "(vectors_within_ang=0); the order parameters "
                             "need them")
        vec = block.vec_ang[sel]
        if not np.isfinite(vec).all():
            raise ValueError(
                f"the pairs keep vectors only to {block.vectors_within_ang} Å, "
                f"below the largest cutoff {largest} Å")
        parts.append((block.i[sel], block.j[sel], block.image[sel], vec,
                      block.d_ang[sel]))
    coverage.check()
    if parts:
        i, j, image, vec, d = (np.concatenate(p) for p in zip(*parts))
    else:
        i = j = np.zeros(0, np.int64)
        image, vec, d = np.zeros((0, 3), np.int64), np.zeros((0, 3)), np.zeros(0)
    definition = "distance cutoffs (d <= cutoff): " + "; ".join(
        f"{a}-{b} {v:g} Å ({sources[(a, b)]})" for (a, b), v in cutoffs.items())
    return _neighbours(symbols, i, j, image, vec, d, definition=definition,
                       cutoffs_ang=cutoffs, cutoff_sources=sources, notes=notes)


# ---------------------------------------------------------------------------
# Wigner 3j symbols and spherical harmonics
# ---------------------------------------------------------------------------

def _whole(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or \
            not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} = {value!r}: a whole number is needed")
    return int(value)


@functools.lru_cache(maxsize=4096)
def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """The Wigner 3j symbol (j1 j2 j3; m1 m2 m3) for whole-number arguments.

    Racah's formula (DLMF 34.2.4) summed exactly in rational arithmetic; the
    square root is taken once, of the exact square, so the only rounding is
    the last one. 0 when the selection rules fail (m1 + m2 + m3 != 0, a
    |m| above its j, or the triangle condition).
    """
    j1, j2, j3 = (_whole(v, n) for v, n in ((j1, "j1"), (j2, "j2"), (j3, "j3")))
    m1, m2, m3 = (_whole(v, n) for v, n in ((m1, "m1"), (m2, "m2"), (m3, "m3")))
    if min(j1, j2, j3) < 0:
        raise ValueError("the j of a 3j symbol are 0 or more")
    if m1 + m2 + m3 != 0 or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if not abs(j1 - j2) <= j3 <= j1 + j2:
        return 0.0
    f = math.factorial
    delta2 = Fraction(f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3),
                      f(j1 + j2 + j3 + 1))
    prefactor2 = delta2 * (f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2)
                           * f(j3 + m3) * f(j3 - m3))
    low = max(0, j2 - j3 - m1, j1 - j3 + m2)
    high = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    total = Fraction(0)
    for s in range(low, high + 1):
        total += Fraction((-1) ** s,
                          f(s) * f(j1 + j2 - j3 - s) * f(j1 - m1 - s)
                          * f(j2 + m2 - s) * f(j3 - j2 + m1 + s)
                          * f(j3 - j1 - m2 + s))
    if total == 0:
        return 0.0
    sign = (-1) ** ((j1 - j2 - m3) % 2) * (1 if total > 0 else -1)
    return sign * math.sqrt(total * total * prefactor2)


@functools.lru_cache(maxsize=None)
def _w_terms(degree: int):
    """Index triples into m = -l..l and their 3j coefficients, m1+m2+m3 = 0."""
    i1, i2, i3, coef = [], [], [], []
    for m1 in range(-degree, degree + 1):
        for m2 in range(-degree, degree + 1):
            m3 = -m1 - m2
            if abs(m3) > degree:
                continue
            c = wigner_3j(degree, degree, degree, m1, m2, m3)
            if c == 0.0:
                continue
            i1.append(m1 + degree)
            i2.append(m2 + degree)
            i3.append(m3 + degree)
            coef.append(c)
    return (np.array(i1), np.array(i2), np.array(i3), np.array(coef))


def _harmonics(degree: int, vec: np.ndarray) -> np.ndarray:
    """Y_lm(r) for m = -l..l, (P, 2l + 1) complex, scipy's convention."""
    from scipy import special

    d = np.sqrt(np.einsum("ij,ij->i", vec, vec))
    if (d <= 0.0).any():
        raise ValueError("a neighbour lies on its centre (zero vector); it has "
                         "no direction")
    polar = np.arccos(np.clip(vec[:, 2] / d, -1.0, 1.0))
    azimuth = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), 2.0 * math.pi)
    m = np.arange(-degree, degree + 1)
    sph_harm_y = getattr(special, "sph_harm_y", None)
    if sph_harm_y is not None:
        return sph_harm_y(degree, m[None, :], polar[:, None], azimuth[:, None])
    return special.sph_harm(m[None, :], degree, azimuth[:, None],
                            polar[:, None])


def _w_of(qlm: np.ndarray, degree: int) -> np.ndarray:
    """sum_{m1+m2+m3=0} 3j q_lm1 q_lm2 q_lm3 per row, real part.

    The atoms are taken in chunks sized so that one gathered (atoms, triples)
    complex array holds at most :data:`_W_CHUNK_BYTES`; the product is formed
    in place, so the peak is about two such arrays whatever l is.
    """
    i1, i2, i3, coef = _w_terms(degree)
    out = np.empty(qlm.shape[0])
    per_chunk = max(1, _W_CHUNK_BYTES // (16 * max(1, coef.size)))
    for start in range(0, qlm.shape[0], per_chunk):
        q = qlm[start:start + per_chunk]
        product = q[:, i1]
        product *= q[:, i2]
        product *= q[:, i3]
        out[start:start + q.shape[0]] = (product @ coef).real
        del product
    return out


def _normalised_w(qlm: np.ndarray, degree: int):
    """(q_l, w-hat_l, n_zero) for (N, 2l + 1) q_lm; NaN rows stay NaN."""
    power = (qlm.real ** 2 + qlm.imag ** 2).sum(axis=1)
    q = np.sqrt(4.0 * math.pi / (2 * degree + 1) * power)
    w = _w_of(qlm, degree)
    zero = q <= Q_ZERO_TOL
    with np.errstate(invalid="ignore", divide="ignore"):
        w_hat = np.where(zero, np.nan, w / power ** 1.5)
    return q, w_hat, int((zero & np.isfinite(q)).sum())


# ---------------------------------------------------------------------------
# Steinhardt bond-orientational order
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class BondOrder:
    """Steinhardt q_l and w-hat_l per atom, their averages, and the global Q_l.

    Arrays indexed (degree, atom) follow ``degrees``; :meth:`values` picks one.
    NaN for an atom with no neighbour, q_l and w-hat_l for an atom with one
    (q_l = 1 for any direction), w-hat where q_l <= Q_ZERO_TOL, and the
    averaged forms where any neighbour of the atom has no q_lm. ``qlm`` holds
    the per-atom complex q_lm, (N, 2l + 1) per degree, m = -l..l (defined
    for one neighbour too).
    """

    degrees: tuple[int, ...]
    elements: np.ndarray           # (N,)
    n_neighbours: np.ndarray       # (N,)
    q: np.ndarray                  # (n_degrees, N)
    w_hat: np.ndarray
    q_bar: np.ndarray
    w_bar_hat: np.ndarray
    qlm: tuple[np.ndarray, ...]
    global_q: np.ndarray           # (n_degrees,)
    global_w_hat: np.ndarray
    definition: str
    notes: tuple[str, ...] = ()

    FIELDS = ("q", "w_hat", "q_bar", "w_bar_hat")

    def values(self, name: str, degree: int) -> np.ndarray:
        """(N,) values of ``name`` ('q', 'w_hat', 'q_bar', 'w_bar_hat') at l."""
        if name not in self.FIELDS:
            raise ValueError(f"{name!r} is not one of {self.FIELDS}")
        if degree not in self.degrees:
            raise ValueError(f"l = {degree} was not computed; the degrees are "
                             f"{self.degrees}")
        return getattr(self, name)[self.degrees.index(degree)]

    @property
    def method_parameters(self) -> dict:
        return {"steinhardt degrees l": self.degrees,
                "neighbours": self.definition, "q_zero_tol": Q_ZERO_TOL}


def _degrees(degrees) -> tuple[int, ...]:
    if isinstance(degrees, (int, np.integer)) and not isinstance(degrees, bool):
        degrees = (degrees,)
    try:
        out = tuple(_whole(d, "a degree l") for d in degrees)
    except TypeError:
        raise ValueError(f"degrees {degrees!r}: whole numbers l are "
                         "needed") from None
    if not out:
        raise ValueError("no degree l was given")
    if len(set(out)) != len(out):
        raise ValueError(f"a degree occurs twice in {out}")
    for d in out:
        if not 1 <= d <= DEGREE_MAX:
            raise ValueError(f"l = {d}: 1 <= l <= {DEGREE_MAX} (DEGREE_MAX, a "
                             "cost bound)")
    return out


def _segment_sum(index: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """sum of complex (P, K) values into (n, K) rows by index, via bincount."""
    out = np.zeros((n, values.shape[1]), dtype=np.complex128)
    for k in range(values.shape[1]):
        out[:, k] = (np.bincount(index, weights=values[:, k].real, minlength=n)
                     + 1j * np.bincount(index, weights=values[:, k].imag,
                                        minlength=n))
    return out


def steinhardt(neighbours: Neighbours,
               degrees: Sequence[int] = DEGREES_DEFAULT) -> BondOrder:
    """q_l, w-hat_l, q-bar_l and w-bar-hat_l of every atom, and the global Q_l.

    The module docstring gives the definitions and conventions. ``degrees``
    is a method choice (default :data:`DEGREES_DEFAULT`, stated in the
    result).
    """
    if not isinstance(neighbours, Neighbours):
        raise ValueError(f"Neighbours are needed, not "
                         f"{type(neighbours).__name__}")
    degrees = _degrees(degrees)
    n = neighbours.n_atoms
    count = neighbours.count
    has = count > 0
    single = count == 1
    n_entries = len(neighbours)
    q_all, w_all, qb_all, wb_all, qlm_all, gq, gw = [], [], [], [], [], [], []
    notes = list(neighbours.notes)
    if single.any():
        notes.append(
            f"{int(single.sum())} atom(s) have exactly one neighbour "
            f"({_by_element(neighbours.elements, single)}): their q_l is 1 and "
            "their w-hat_l is the 3j symbol (l l l; 0 0 0) for any "
            "direction, so both are NaN for them; their q_lm (one harmonic) "
            "still enter their neighbours' q-bar_l, and their own q-bar_l "
            "averages it with their neighbour's")
    for degree in degrees:
        width = 2 * degree + 1
        total = np.zeros((n, width), dtype=np.complex128)
        global_sum = np.zeros(width, dtype=np.complex128)
        for start in range(0, n_entries, _BONDS_PER_CHUNK):
            chunk = slice(start, start + _BONDS_PER_CHUNK)
            y = _harmonics(degree, neighbours.vec_ang[chunk])
            total += _segment_sum(neighbours.centre[chunk], y, n)
            global_sum += y.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            qlm = total / count[:, None]
        qlm[~has] = np.nan
        q, w_hat, n_zero = _normalised_w(qlm, degree)
        # one neighbour: q_lm is one harmonic, so q_l = 1 and w-hat_l is a
        # constant for every direction (the addition theorem); not a value.
        # q_lm itself is kept, for the neighbours' averages.
        q[single] = np.nan
        w_hat[single] = np.nan
        # Lechner-Dellago: the centre and its neighbours, each once per entry
        around = _segment_sum(neighbours.centre, qlm[neighbours.nbr], n)
        qlm_bar = (qlm + around) / (count + 1)[:, None]
        qlm_bar[~has] = np.nan
        q_bar, w_bar_hat, n_zero_bar = _normalised_w(qlm_bar, degree)
        if n_entries:
            g = global_sum / n_entries
            g_q, g_w, _ = _normalised_w(g[None, :], degree)
            gq.append(float(g_q[0]))
            gw.append(float(g_w[0]))
        else:
            gq.append(math.nan)
            gw.append(math.nan)
        if n_zero:
            notes.append(f"l = {degree}: {n_zero} atom(s) have q_l <= "
                         f"{Q_ZERO_TOL:g}, where w-hat_l is 0/0; it is NaN "
                         "for them")
        if n_zero_bar:
            notes.append(f"l = {degree}: {n_zero_bar} atom(s) have q-bar_l <= "
                         f"{Q_ZERO_TOL:g}; w-bar-hat_l is NaN for them")
        q_all.append(q)
        w_all.append(w_hat)
        qb_all.append(q_bar)
        wb_all.append(w_bar_hat)
        qlm.setflags(write=False)
        qlm_all.append(qlm)
    blind = has & ~np.isfinite(qb_all[0])
    if blind.any():
        notes.append(
            f"{int(blind.sum())} atom(s) have a neighbour that itself has no "
            "neighbour under this definition "
            f"({_by_element(neighbours.elements, blind)}), "
            "so their averaged q-bar_l and w-bar-hat_l are NaN; a cutoff "
            "given for (A, B) and not for (B, A) does this")
    return BondOrder(
        degrees=degrees, elements=neighbours.elements,
        n_neighbours=_frozen(count), q=_frozen(np.array(q_all)),
        w_hat=_frozen(np.array(w_all)), q_bar=_frozen(np.array(qb_all)),
        w_bar_hat=_frozen(np.array(wb_all)), qlm=tuple(qlm_all),
        global_q=_frozen(np.array(gq)), global_w_hat=_frozen(np.array(gw)),
        definition=neighbours.definition, notes=tuple(notes))


# ---------------------------------------------------------------------------
# tetrahedral order
# ---------------------------------------------------------------------------

_SIX_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True, eq=False)
class TetrahedralOrder:
    """q_tet per atom: 1 for a regular tetrahedron; NaN where not defined."""

    elements: np.ndarray
    n_neighbours: np.ndarray
    q_tet: np.ndarray
    selection: str
    definition: str
    notes: tuple[str, ...] = ()

    @property
    def method_parameters(self) -> dict:
        return {"q_tet selection": self.selection,
                "neighbours": self.definition}


def tetrahedral_order(neighbours: Neighbours, *,
                      selection: str = "exactly four") -> TetrahedralOrder:
    """Errington and Debenedetti's q_tet [3] of every atom it is defined for.

    ``selection``: 'exactly four' (atoms with four neighbours) or 'four
    nearest' (the four nearest entries of the neighbour list, for atoms with
    four or more). A method choice, stated in the result; the atoms left out
    are counted per element and per neighbour count.

    'four nearest' ranks within the list given, not among all atoms: a list
    from :func:`neighbours_from_bonds` holds bonded counter-ions only, so an
    unbonded atom nearer than the fourth entry is not taken (frame 19 of a
    Na-borosilicate glass model: one of its 1 800 O, bonded to two B and two
    Na, had an unbonded B at 2.48 Å, nearer than both Na at 2.50 and
    2.55 Å; frame 0 had none; measured 2026-10-07). Errington and
    Debenedetti's four nearest molecules [3] are the four nearest atoms of a
    list from :func:`neighbours_from_pairs` whose cutoff holds at least four
    of them.
    """
    if not isinstance(neighbours, Neighbours):
        raise ValueError(f"Neighbours are needed, not "
                         f"{type(neighbours).__name__}")
    if selection not in QTET_SELECTIONS:
        raise ValueError(f"selection {selection!r}: one of {QTET_SELECTIONS}")
    count = neighbours.count
    take = count == 4 if selection == "exactly four" else count >= 4
    rows = np.flatnonzero(take)
    q = np.full(neighbours.n_atoms, np.nan)
    notes = list(neighbours.notes)
    if rows.size:
        idx = neighbours.offsets[rows][:, None] + np.arange(4)
        unit = neighbours.vec_ang[idx] / neighbours.d_ang[idx][..., None]
        total = np.zeros(rows.size)
        for a, b in _SIX_PAIRS:
            cos = np.einsum("ij,ij->i", unit[:, a], unit[:, b])
            total += (cos + 1.0 / 3.0) ** 2
        q[rows] = 1.0 - 3.0 / 8.0 * total
        if selection == "four nearest":
            more = rows[count[rows] > 4]
            if more.size:
                fourth = neighbours.d_ang[neighbours.offsets[more] + 3]
                fifth = neighbours.d_ang[neighbours.offsets[more] + 4]
                tied = int((fourth == fifth).sum())
                if tied:
                    notes.append(
                        f"{tied} atom(s) have their fourth and fifth neighbours "
                        "at the same distance; the fourth taken is the lower "
                        "row (then image), so q_tet depends on that order")
    left = ~take
    if left.any():
        detail = Counter((str(e), int(c)) for e, c in
                         zip(neighbours.elements[left], count[left]))
        listed = ", ".join(f"{e} with {c}: {k}"
                           for (e, c), k in sorted(detail.items()))
        notes.append(f"q_tet ({selection}) is NaN for {int(left.sum())} atom(s) "
                     f"by element and neighbour count: {listed}")
    definition = neighbours.definition
    if selection == "four nearest":
        definition += ("; q_tet from the four nearest entries of this list "
                       "(ranked within the list, not among all atoms)")
    return TetrahedralOrder(
        elements=neighbours.elements, n_neighbours=_frozen(count),
        q_tet=_frozen(q), selection=selection,
        definition=definition, notes=tuple(notes))


# ---------------------------------------------------------------------------
# polyhedron distortion
# ---------------------------------------------------------------------------

def _trans_matchings() -> tuple[tuple[tuple[int, int], ...], ...]:
    """The fifteen perfect matchings of six ligands, in the order
    ``polyhedra._trans_pairs`` visits them, so a tie resolves the same way."""
    out = []
    rest = [1, 2, 3, 4, 5]
    for a in range(5):
        p1 = (0, rest[a])
        rem = [x for k, x in enumerate(rest) if k != a]
        for b in (1, 2, 3):
            p2 = (rem[0], rem[b])
            p3 = tuple(x for k, x in enumerate(rem) if k not in (0, b))
            out.append((p1, p2, p3))
    return tuple(out)


_MATCHINGS = _trans_matchings()
_PAIRS6 = tuple((i, j) for i in range(6) for j in range(i + 1, 6))


@dataclass(frozen=True, eq=False)
class PolyhedronShape:
    """Distortion of every atom's polyhedron (its neighbour list).

    ``angle_variance_deg2``, ``quadratic_elongation`` and ``volume_ang3`` are
    defined for CN 4 and 6 and NaN otherwise; ``baur`` and the ECoN for every
    CN >= 1. NaN where the atom has no neighbour.

    ``octahedral_hull`` is True for a CN-6 atom whose convex hull is the
    octahedron of the trans pairs used (the matching of largest total angle,
    ``polyhedra.shape``'s convention), and False for every other atom. For a
    CN-6 atom where it is False, sigma^2 and <lambda> are still computed with
    that matching, as ``polyhedra.shape`` computes them, but the twelve
    "cis" angles are then not the twelve edges of its hull: either the hull
    is not an octahedron, or its three non-edges are another matching (the
    notes count both).
    """

    elements: np.ndarray
    cn: np.ndarray
    d_mean_ang: np.ndarray
    d_min_ang: np.ndarray
    d_max_ang: np.ndarray
    baur: np.ndarray
    angle_variance_deg2: np.ndarray
    quadratic_elongation: np.ndarray
    volume_ang3: np.ndarray
    ecn: np.ndarray
    ecn_d_av_ang: np.ndarray
    octahedral_hull: np.ndarray
    definition: str
    notes: tuple[str, ...] = ()

    @property
    def method_parameters(self) -> dict:
        return {"neighbours": self.definition,
                "tetrahedral angle (deg)": TETRAHEDRAL_ANGLE_DEG,
                "CN-6 trans pairs": "matching of largest total angle "
                                    "(polyhedra.shape)"}


def _angles_deg(vec: np.ndarray, d: np.ndarray, pairs) -> np.ndarray:
    """(M, len(pairs)) angles at the centre, as ``polyhedra`` forms them."""
    out = np.empty((vec.shape[0], len(pairs)))
    for k, (i, j) in enumerate(pairs):
        cos = np.einsum("ij,ij->i", vec[:, i], vec[:, j]) / (d[:, i] * d[:, j])
        out[:, k] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return out


def _ecn_rows(d: np.ndarray):
    """``polyhedra.effective_cn`` on every row of (M, c) distances at once.

    The same iteration, row by row: d_av starts at the shortest distance;
    each step sets d_av to the weighted mean, and a row stops at the step
    where that moved it by less than 1e-11, or when its weights sum to 0, or
    after 300 steps.
    """
    d_av = d.min(axis=1).copy()
    active = np.ones(d.shape[0], dtype=bool)
    for _ in range(300):
        rows = np.flatnonzero(active)
        if not rows.size:
            break
        w = np.exp(1.0 - (d[rows] / d_av[rows, None]) ** 6)
        total = w.sum(axis=1)
        stop = ~(total > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            new = (d[rows] * w).sum(axis=1) / total
        converged = np.abs(new - d_av[rows]) < 1e-11
        moved = ~stop
        d_av[rows[moved]] = new[moved]
        active[rows[stop | converged]] = False
    w = np.exp(1.0 - (d / d_av[:, None]) ** 6)
    return w.sum(axis=1), d_av


def _octahedron_volume(vec: np.ndarray, matching_index: np.ndarray,
                       scale: np.ndarray):
    """(volume, convex): the volume of the octahedron of the trans pairs of
    each row, and whether that octahedron is convex, in which case it is the
    hull of the six ligands and the volume is the hull volume."""
    m = vec.shape[0]
    pairs = np.array(_MATCHINGS)[matching_index]            # (M, 3, 2)
    rows = np.arange(m)
    centroid = vec.mean(axis=1)
    volume = np.zeros(m)
    convex = np.ones(m, dtype=bool)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                ia, ib, ic = pairs[:, 0, x], pairs[:, 1, y], pairs[:, 2, z]
                pa, pb, pc = vec[rows, ia], vec[rows, ib], vec[rows, ic]
                normal = np.cross(pb - pa, pc - pa)
                others = np.stack([pairs[:, 0, 1 - x], pairs[:, 1, 1 - y],
                                   pairs[:, 2, 1 - z]], axis=1)   # (M, 3)
                side = np.einsum("mkj,mj->mk", vec[rows[:, None], others]
                                 - pa[:, None, :], normal)
                tol = DEGENERATE_REL_TOL * scale ** 3
                same = ((side > tol[:, None]).all(axis=1)
                        | (side < -tol[:, None]).all(axis=1))
                convex &= same
                volume += np.abs(np.einsum("mj,mj->m", centroid - pa,
                                           normal)) / 6.0
    return volume, convex


def polyhedron_shape(neighbours: Neighbours) -> PolyhedronShape:
    """Baur index, angle variance, quadratic elongation, volume and ECoN.

    The module docstring gives each definition and the line of
    ``polyhedra.py`` it reproduces.
    """
    if not isinstance(neighbours, Neighbours):
        raise ValueError(f"Neighbours are needed, not "
                         f"{type(neighbours).__name__}")
    n = neighbours.n_atoms
    cn = neighbours.count
    out = {name: np.full(n, np.nan) for name in (
        "d_mean", "d_min", "d_max", "baur", "angle_variance",
        "quadratic_elongation", "volume", "ecn", "ecn_d_av")}
    notes = list(neighbours.notes)
    octahedral = np.zeros(n, dtype=bool)
    hull_failures = flat = other_trans = not_octahedron = 0
    for c in np.unique(cn[cn > 0]):
        rows = np.flatnonzero(cn == c)
        idx = neighbours.offsets[rows][:, None] + np.arange(c)
        vec = neighbours.vec_ang[idx]                        # (M, c, 3)
        d = neighbours.d_ang[idx]                            # (M, c)
        d_mean = d.mean(axis=1)
        out["d_mean"][rows] = d_mean
        out["d_min"][rows] = d.min(axis=1)
        out["d_max"][rows] = d.max(axis=1)
        out["baur"][rows] = np.abs(d - d_mean[:, None]).mean(axis=1) / d_mean
        ecn, d_av = _ecn_rows(d)
        out["ecn"][rows] = ecn
        out["ecn_d_av"][rows] = d_av
        if c == 4:
            ang = _angles_deg(vec, d, _SIX_PAIRS)
            out["angle_variance"][rows] = (
                ((ang - TETRAHEDRAL_ANGLE_DEG) ** 2).sum(axis=1) / 5.0)
            edges = vec[:, 1:] - vec[:, :1]
            volume = np.abs(np.linalg.det(edges)) / 6.0
            out["volume"][rows] = volume
            positive = volume > 0
            flat += int((~positive).sum())
            d0 = (9.0 * math.sqrt(3.0) * volume / 8.0) ** (1.0 / 3.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                qe = ((d / d0[:, None]) ** 2).mean(axis=1)
            out["quadratic_elongation"][rows] = np.where(positive, qe, np.nan)
        elif c == 6:
            ang = _angles_deg(vec, d, _PAIRS6)               # (M, 15)
            index = {pair: k for k, pair in enumerate(_PAIRS6)}
            totals = np.stack([ang[:, index[p1]] + ang[:, index[p2]]
                               + ang[:, index[p3]]
                               for p1, p2, p3 in _MATCHINGS], axis=1)
            best = np.argmax(totals, axis=1)
            trans = np.zeros((rows.size, 15), dtype=bool)
            for k, matching in enumerate(_MATCHINGS):
                hit = best == k
                for pair in matching:
                    trans[hit, index[pair]] = True
            sq = np.where(trans, 0.0, (ang - OCTAHEDRAL_CIS_ANGLE_DEG) ** 2)
            out["angle_variance"][rows] = sq.sum(axis=1) / 11.0
            volume, convex = _octahedron_volume(vec, best, d_mean)
            octahedral[rows] = convex
            fallback = np.flatnonzero(~convex)
            if fallback.size:
                # is the hull an octahedron of another matching? (its three
                # non-edges are then that matching; the same convexity test)
                any_other = np.zeros(fallback.size, dtype=bool)
                for k in range(len(_MATCHINGS)):
                    _, ok = _octahedron_volume(
                        vec[fallback], np.full(fallback.size, k),
                        d_mean[fallback])
                    any_other |= ok
                other_trans += int(any_other.sum())
                not_octahedron += int((~any_other).sum())
            for k in fallback:
                # the crystal path's own function, ConvexHull and its
                # handling of a set Qhull refuses included
                hull = polyhedra.shape(vec[k])["volume"]
                if hull is None:
                    volume[k] = np.nan
                    hull_failures += 1
                else:
                    volume[k] = hull
            out["volume"][rows] = volume
            positive = volume > 0
            flat += int((~positive).sum())
            d0 = (3.0 * volume / 4.0) ** (1.0 / 3.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                qe = ((d / d0[:, None]) ** 2).mean(axis=1)
            out["quadratic_elongation"][rows] = np.where(positive, qe, np.nan)
    if other_trans or not_octahedron:
        notes.append(
            f"of the six-coordinated polyhedra, {not_octahedron} have a "
            f"convex hull that is not an octahedron and {other_trans} an "
            "octahedral hull whose trans pairs (its three non-edges) are not "
            "the matching of largest total angle; for both, angle variance "
            "and quadratic elongation are computed as polyhedra.shape "
            "computes them, with the largest-angle matching as trans pairs, "
            "so the twelve angles used are not the edges of the hull; their "
            "volume is the hull's (polyhedra.shape, scipy's ConvexHull), and "
            "octahedral_hull is False for them")
    if hull_failures or flat:
        notes.append(f"{flat} polyhedra of CN 4 or 6 have no volume (flat, or "
                     "Qhull refused them); their quadratic elongation is NaN")
    other = (cn > 0) & (cn != 4) & (cn != 6)
    if other.any():
        notes.append(f"angle variance, quadratic elongation and volume are "
                     f"defined here for CN 4 and 6 only; NaN for "
                     f"{int(other.sum())} atom(s) of other CN")
    return PolyhedronShape(
        elements=neighbours.elements, cn=_frozen(cn),
        d_mean_ang=_frozen(out["d_mean"]), d_min_ang=_frozen(out["d_min"]),
        d_max_ang=_frozen(out["d_max"]), baur=_frozen(out["baur"]),
        angle_variance_deg2=_frozen(out["angle_variance"]),
        quadratic_elongation=_frozen(out["quadratic_elongation"]),
        volume_ang3=_frozen(out["volume"]), ecn=_frozen(out["ecn"]),
        ecn_d_av_ang=_frozen(out["ecn_d_av"]),
        octahedral_hull=_frozen(octahedral),
        definition=neighbours.definition, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Voronoi cells
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class VoronoiCells:
    """The Voronoi cell of every atom of a periodic frame.

    ``n_faces`` counts the faces that are faces (three or more edges once
    edges at or below ``degeneracy_floor_ang`` are contracted) with area
    above ``area_threshold_ang2`` (the caller's ``min_face_area_ang2``; 0 is
    no area threshold); ``edge_counts[:, k]`` how many of those have k + 3
    edges longer than ``edge_threshold_ang`` (the Voronoi index n3, n4, ...),
    and ``edge_counts.sum(axis=1)`` how many have three or more such edges.
    The face table (``face_*``, one row per face of each atom, sorted by
    atom, neighbour and image) holds every face Qhull returned, with
    ``face_counted`` marking those in ``n_faces``.
    """

    elements: np.ndarray
    volume_ang3: np.ndarray
    n_faces: np.ndarray
    edge_counts: np.ndarray            # (N, E) faces with 3, 4, ... edges
    face_atom: np.ndarray
    face_nbr: np.ndarray
    face_image: np.ndarray
    face_distance_ang: np.ndarray
    face_area_ang2: np.ndarray
    face_edges: np.ndarray
    face_counted: np.ndarray
    margin_ang: float
    reach_ang: float                   # largest 2 rho over the cells
    min_face_area_ang2: float
    min_edge_ang: float
    area_threshold_ang2: float
    edge_threshold_ang: float
    degeneracy_floor_ang: float
    volume_sum_ang3: float
    box_volume_ang3: float
    notes: tuple[str, ...] = ()

    @property
    def n_atoms(self) -> int:
        return int(self.elements.shape[0])

    def index(self, length: int | None = None) -> list[tuple[int, ...]]:
        """The Voronoi index of every atom, (n3, n4, ..., n_{length+2}).

        ``length`` defaults to all columns (at least n3..n6); a shorter one
        is refused when it would drop a face.
        """
        width = self.edge_counts.shape[1]
        length = width if length is None else _whole(length, "length")
        if length < width and self.edge_counts[:, length:].any():
            raise ValueError(f"an index of length {length} would drop faces "
                             f"with more than {length + 2} edges")
        counts = np.zeros((self.n_atoms, max(length, width)), np.int64)
        counts[:, :width] = self.edge_counts
        return [tuple(int(v) for v in row[:length]) for row in counts]

    @property
    def method_parameters(self) -> dict:
        return {"voronoi margin (Å)": self.margin_ang,
                "min_face_area_ang2": self.min_face_area_ang2,
                "min_edge_ang": self.min_edge_ang,
                "degenerate_rel_tol": DEGENERATE_REL_TOL}


def _ridge_geometry(vertices: np.ndarray, flat: np.ndarray,
                    lengths: np.ndarray, pa: np.ndarray, pb: np.ndarray):
    """(area, reach_a, reach_b, ordered, following, edge) of ridge polygons
    given as flat vertex lists, each ordered by angle about its centroid in
    its own plane. ``ordered`` holds the vertex ids of each polygon in that
    order (flat, the layout of ``flat``), ``following`` the id after each one
    around the polygon, and ``edge`` the length of that edge.

    Polygons with the same number of vertices are handled together, so no
    array is padded.
    """
    s = lengths.size
    area = np.empty(s)
    reach_a = np.empty(s)
    reach_b = np.empty(s)
    ordered = np.empty(flat.size, np.int64)
    following = np.empty(flat.size, np.int64)
    edge_len = np.empty(flat.size)
    start = np.cumsum(lengths) - lengths
    for size in np.unique(lengths):
        rows = np.flatnonzero(lengths == size)
        slots = start[rows][:, None] + np.arange(size)
        ids = flat[slots]
        xyz = vertices[ids]
        a, b = pa[rows], pb[rows]
        normal = b - a
        normal /= np.sqrt(np.einsum("ij,ij->i", normal, normal))[:, None]
        helper = np.eye(3)[np.argmin(np.abs(normal), axis=1)]
        e1 = np.cross(normal, helper)
        e1 /= np.sqrt(np.einsum("ij,ij->i", e1, e1))[:, None]
        e2 = np.cross(normal, e1)
        rel = xyz - xyz.mean(axis=1)[:, None, :]
        x = np.einsum("skj,sj->sk", rel, e1)
        y = np.einsum("skj,sj->sk", rel, e2)
        order = np.argsort(np.arctan2(y, x), axis=1, kind="stable")
        xs = np.take_along_axis(x, order, axis=1)
        ys = np.take_along_axis(y, order, axis=1)
        ps = np.take_along_axis(xyz, order[..., None], axis=1)
        x2, y2 = np.roll(xs, -1, axis=1), np.roll(ys, -1, axis=1)
        area[rows] = 0.5 * np.abs((xs * y2 - x2 * ys).sum(axis=1))
        step = np.roll(ps, -1, axis=1) - ps
        edge_len[slots] = np.sqrt(np.einsum("skj,skj->sk", step, step))
        ids_sorted = np.take_along_axis(ids, order, axis=1)
        ordered[slots] = ids_sorted
        following[slots] = np.roll(ids_sorted, -1, axis=1)
        da = xyz - a[:, None, :]
        db = xyz - b[:, None, :]
        reach_a[rows] = np.sqrt(np.einsum("skj,skj->sk", da, da).max(axis=1))
        reach_b[rows] = np.sqrt(np.einsum("skj,skj->sk", db, db).max(axis=1))
    return area, reach_a, reach_b, ordered, following, edge_len


def _voronoi_faces(points: np.ndarray, n: int, edge_tol: float,
                   floor_tol: float):
    """Faces of the cells of the first n points, or None if one is open.

    Edges at or below ``floor_tol`` are contracted: their two vertices, and
    every vertex joined to them by such an edge, become one vertex (the
    connected components of the graph of short edges). A face's edge count
    is then the number of edges whose ends lie in different components, so
    every cell is a polyhedron again and Euler's relation holds for it;
    ``n_floor`` counts those edges, ``n_edges`` those also longer than
    ``edge_tol``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import Voronoi

    vor = Voronoi(points)
    rp = vor.ridge_points
    sel = np.flatnonzero((rp[:, 0] < n) | (rp[:, 1] < n))
    ridge_vertices = vor.ridge_vertices
    parts = []
    for start in range(0, sel.size, _RIDGES_PER_CHUNK):
        chunk = sel[start:start + _RIDGES_PER_CHUNK]
        lists = [ridge_vertices[k] for k in chunk]
        lengths = np.fromiter(map(len, lists), np.int64, count=len(lists))
        flat = np.fromiter(itertools.chain.from_iterable(lists), np.int64,
                           count=int(lengths.sum()))
        if (flat < 0).any():
            return None
        pa, pb = points[rp[chunk, 0]], points[rp[chunk, 1]]
        parts.append((chunk, lengths) + _ridge_geometry(vor.vertices, flat,
                                                        lengths, pa, pb))
    (chunk, lengths, area, reach_a, reach_b, ordered, following,
     edge_len) = (np.concatenate(p) for p in zip(*parts))
    short = edge_len <= floor_tol
    n_vertices = vor.vertices.shape[0]
    graph = coo_matrix((np.ones(int(short.sum())),
                        (ordered[short], following[short])),
                       shape=(n_vertices, n_vertices))
    _, label = connected_components(graph, directed=False)
    distinct = label[ordered] != label[following]
    ridge = np.repeat(np.arange(lengths.size), lengths)
    n_floor = np.bincount(ridge, weights=distinct,
                          minlength=lengths.size).astype(np.int64)
    n_edges = np.bincount(ridge, weights=distinct & (edge_len > edge_tol),
                          minlength=lengths.size).astype(np.int64)
    a, b = rp[chunk, 0], rp[chunk, 1]
    first, second = a < n, b < n
    atom = np.concatenate((a[first], b[second]))
    other = np.concatenate((b[first], a[second]))
    return (atom, other, np.concatenate((area[first], area[second])),
            np.concatenate((n_edges[first], n_edges[second])),
            np.concatenate((n_floor[first], n_floor[second])),
            np.concatenate((reach_a[first], reach_b[second])))


def voronoi_cells(frame: Frame, *, min_face_area_ang2: float = 0.0,
                  min_edge_ang: float = 0.0,
                  margin_ang: float | None = None) -> VoronoiCells:
    """Volume, face count and Voronoi index of every atom's periodic cell.

    ``min_face_area_ang2`` and ``min_edge_ang`` are the face and edge
    thresholds of the Voronoi index (method choices, 0 = no filtering,
    stated in the result). ``margin_ang`` is the first margin of periodic
    images tried (default: :data:`MARGIN_START_SPACINGS` x 2 mean spacings);
    it is enlarged until every cell passes the reach check of the module
    docstring, and the margins tried are noted. ValueError when the cell
    volumes do not add up to the box volume within :data:`VOLUME_LAW_TOL`,
    and for an atom that lies on another one (it has no cell).
    """
    frame = _check_frame(frame)
    min_area = _nonnegative(min_face_area_ang2, "min_face_area_ang2")
    min_edge = _nonnegative(min_edge_ang, "min_edge_ang")
    n = frame.n_atoms
    spacing = _spacing_ang(frame)
    # the degeneracy floor is one length: an edge at or below it is
    # contracted, and a face left with fewer than three edges is not a face
    # (_voronoi_faces, module docstring); the areas are thresholded only by
    # the caller's min_face_area_ang2
    floor_tol = DEGENERATE_REL_TOL * spacing
    edge_tol = max(min_edge, floor_tol)
    given = None if margin_ang is None else _positive(margin_ang, "margin_ang")
    margin = given if given is not None else \
        2.0 * MARGIN_START_SPACINGS * spacing
    attempts: list[tuple[float, float]] = []
    for _ in range(MARGIN_ATTEMPTS):
        atom_p, image_p, points = _padded(frame, margin)
        faces = _voronoi_faces(points, n, edge_tol, floor_tol)
        if faces is None:
            attempts.append((margin, math.inf))
            margin = _next_margin(margin, math.inf)
            continue
        atom, other, area, n_edges, n_floor, reach = faces
        rho = np.zeros(n)
        np.maximum.at(rho, atom, reach)
        need = 2.0 * float(rho.max())
        attempts.append((margin, need))
        if need <= margin:
            break
        margin = _next_margin(margin, need)
    else:
        raise ValueError(
            f"the Voronoi cells did not pass the reach check after "
            f"{MARGIN_ATTEMPTS} tessellations (margins and reaches: "
            f"{attempts}); a cell this open is not resolved")
    faceless = np.flatnonzero(np.bincount(atom, minlength=n) == 0)
    if faceless.size:
        raise ValueError(_coincident_message(frame, faceless))
    distance = np.linalg.norm(points[other] - points[atom], axis=1)
    volume = np.bincount(atom, weights=area * distance / 6.0, minlength=n)
    nbr = atom_p[other]
    image = image_p[other]
    order = np.lexsort((image[:, 2], image[:, 1], image[:, 0], nbr, atom))
    atom, nbr, image, distance, area, n_edges, n_floor = (
        v[order] for v in (atom, nbr, image, distance, area, n_edges, n_floor))
    degenerate = n_floor < 3
    counted = ~degenerate & (area > min_area if min_area > 0.0 else True)
    n_faces = np.bincount(atom[counted], minlength=n)
    shaped = counted & (n_edges >= 3)
    width = max(4, int(n_edges[shaped].max()) - 2 if shaped.any() else 4)
    edge_counts = np.zeros((n, width), np.int64)
    np.add.at(edge_counts, (atom[shaped], n_edges[shaped] - 3), 1)

    box_volume = frame.volume_ang3
    total = math.fsum(volume)
    relative = abs(total - box_volume) / box_volume
    notes = [_margin_note("Voronoi cells", attempts, given),
             f"the cell volumes sum to {total:.12g} Å^3 and the box holds "
             f"{box_volume:.12g} Å^3 (relative difference {relative:.2e})"]
    if relative > VOLUME_LAW_TOL:
        raise ValueError(
            f"the Voronoi cell volumes sum to {total:.12g} Å^3 but the box "
            f"holds {box_volume:.12g} Å^3 (relative difference "
            f"{relative:.2e}, above VOLUME_LAW_TOL {VOLUME_LAW_TOL:g}); the "
            "tessellation is not the periodic one")
    closest = float(distance.min())
    if closest < bulk.D_MIN_ANG:
        close = np.unique(atom[distance < bulk.D_MIN_ANG])
        notes.append(f"{close.size} atom(s) have a Voronoi neighbour closer "
                     f"than {bulk.D_MIN_ANG:g} Å (bulk.D_MIN_ANG, below which "
                     "the pair search counts a contact as an atom in its own "
                     f"image); the closest pair is {closest:.3g} Å apart "
                     f"(rows {_rows_text(close)})")
    few = counted & (n_edges < 3)
    if few.any():
        notes.append(f"{int(few.sum())} face(s) above the area threshold have "
                     f"fewer than 3 edges longer than {edge_tol:.3g} Å; they "
                     "count in n_faces and in no column of the index "
                     "(edge_counts.sum(axis=1) leaves them out)")
    if degenerate.any():
        notes.append(f"{int(degenerate.sum())} face(s) have fewer than 3 "
                     "edges once the edges at or below the degeneracy floor "
                     f"{floor_tol:.3g} Å (DEGENERATE_REL_TOL x the mean "
                     "spacing) are contracted; they are Qhull's rendering of "
                     "a degenerate arrangement and are left out of the face "
                     "count and the index (their area still counts in the "
                     "volume)")
    thresholded = ~degenerate & ~counted
    if thresholded.any():
        notes.append(f"{int(thresholded.sum())} face(s) at or below "
                     f"min_face_area_ang2 = {min_area:g} Å^2 are left out of "
                     "the face count and the index (their area still counts "
                     "in the volume)")
    euler = (edge_counts * (6 - np.arange(3, 3 + width))).sum(axis=1)
    short = np.flatnonzero(euler < 12)
    if short.size:
        cause = ("min_face_area_ang2 or min_edge_ang removed a face or an edge "
                 "that the neighbouring faces still count"
                 if min_area > 0.0 or min_edge > floor_tol else
                 "no threshold was given, and contracting the edges at or "
                 "below the degeneracy floor did not resolve the arrangement")
        notes.append(f"{short.size} cell(s) have sum_k (6 - k) n_k below 12 "
                     "over their index, which no convex polyhedron has "
                     "(Euler's relation gives 12 or more): "
                     f"{cause} (rows {_rows_text(short)})")
    return VoronoiCells(
        elements=frame.elements, volume_ang3=_frozen(volume),
        n_faces=_frozen(n_faces), edge_counts=_frozen(edge_counts),
        face_atom=_frozen(atom), face_nbr=_frozen(nbr),
        face_image=_frozen(image), face_distance_ang=_frozen(distance),
        face_area_ang2=_frozen(area), face_edges=_frozen(n_edges),
        face_counted=_frozen(counted), margin_ang=float(margin),
        reach_ang=float(attempts[-1][1]), min_face_area_ang2=min_area,
        min_edge_ang=min_edge, area_threshold_ang2=min_area,
        degeneracy_floor_ang=floor_tol,
        edge_threshold_ang=edge_tol, volume_sum_ang3=total,
        box_volume_ang3=box_volume, notes=tuple(notes))


def neighbours_from_voronoi(frame: Frame, cells: VoronoiCells) -> Neighbours:
    """Each atom's Voronoi neighbours: the atoms across its counted faces.

    Every counted face is a full neighbour, whatever its area. Without a
    face-area threshold (``min_face_area_ang2 = 0`` in :func:`voronoi_cells`)
    that includes faces of 1e-8 Å^2: frame 0 of each of four oxide-glass
    models of 2 880 to 3 000 atoms (SiO2, Na2O-3SiO2, NAS, NBS) held 26 to
    45 faces of 1e-5 Å^2 or less, the smallest 3.4e-9 Å^2, and leaving them
    out changed an atom's q_4 by up to 0.12, its q_6 by up to 0.09 and its
    q-bar_6 by up to 0.02 (measured 2026-10-07). A neighbour list jumps when
    a face appears or vanishes, and q_l with it; Mickel et al. [19] weight
    each face by its area so that it does not (Minkowski structure metrics,
    not computed here, module docstring). A face-area threshold leaves the
    smallest faces out. With no threshold the definition and a note say so,
    and give the smallest face counted.
    """
    frame = _check_frame(frame)
    if not isinstance(cells, VoronoiCells):
        raise ValueError(f"VoronoiCells are needed, not {type(cells).__name__}")
    if cells.n_atoms != frame.n_atoms or \
            not np.array_equal(cells.elements, frame.elements):
        raise ValueError("the Voronoi cells are of another frame")
    keep = cells.face_counted
    centre, nbr = cells.face_atom[keep], cells.face_nbr[keep]
    image = cells.face_image[keep]
    vec, d = bulk._pair_geometry(frame.frac, frame.box_ang, centre, nbr, image)
    if not np.allclose(d, cells.face_distance_ang[keep], rtol=0, atol=1e-9):
        raise ValueError("the Voronoi cells are of another frame (the face "
                         "distances differ from this frame's geometry)")
    notes = []
    if cells.area_threshold_ang2 > 0.0:
        definition = (f"Voronoi faces with area above "
                      f"{cells.area_threshold_ang2:.3g} Å^2, each a full "
                      "neighbour")
    else:
        definition = ("every Voronoi face, with no face-area threshold, each "
                      "a full neighbour however small its area")
        if keep.any():
            areas = cells.face_area_ang2[keep]
            notes.append(
                "no face-area threshold: every face counts as a full "
                f"neighbour; the smallest face counted is {areas.min():.3g} "
                f"Å^2 and the median face {float(np.median(areas)):.3g} Å^2 "
                "(min_face_area_ang2 in voronoi_cells sets a threshold)")
    return _neighbours(frame.elements, centre, nbr, image, vec, d,
                       definition=definition, notes=notes)


# ---------------------------------------------------------------------------
# Delaunay voids
# ---------------------------------------------------------------------------

def _checked_radii(frame: Frame, radii_ang, radii_source) -> dict[str, float]:
    if not isinstance(radii_ang, Mapping):
        raise ValueError("radii_ang needs a mapping element -> radius in Å "
                         "(vdw_radii_ang gives FACET's van der Waals radii); "
                         "none is assumed")
    if not isinstance(radii_source, str) or not radii_source.strip():
        raise ValueError("radii_source needs the source of the radii; a "
                         "radius without one is not used")
    missing = [s for s in frame.species if s not in radii_ang]
    if missing:
        raise ValueError(f"no radius for {', '.join(missing)}; every element "
                         "of the frame needs one")
    return {s: _nonnegative(radii_ang[s], f"the radius of {s}")
            for s in frame.species}


@dataclass(frozen=True, eq=False)
class EmptySpheres:
    """The empty sphere of every Delaunay tetrahedron of a periodic frame.

    Tetrahedron t has vertices ``vertices[t]`` (atom rows) translated by
    ``vertex_image[t]``, circumcentre ``centre_frac[t]`` (in [0, 1)),
    circumradius ``circumradius_ang[t]``, and ``radius_ang[t]``, the largest
    sphere centred there that overlaps no atom sphere of ``radii_ang`` (signed:
    negative when the circumcentre lies inside an atom sphere).
    """

    vertices: np.ndarray               # (T, 4)
    vertex_image: np.ndarray           # (T, 4, 3)
    centre_frac: np.ndarray            # (T, 3)
    circumradius_ang: np.ndarray       # (T,)
    radius_ang: np.ndarray             # (T,)
    volume_ang3: np.ndarray            # (T,)
    radii_ang: Mapping[str, float]
    radii_source: str
    margin_ang: float
    volume_sum_ang3: float
    box_volume_ang3: float
    n_degenerate: int
    notes: tuple[str, ...] = ()

    def __len__(self) -> int:
        return int(self.radius_ang.shape[0])

    @property
    def method_parameters(self) -> dict:
        return {"delaunay margin (Å)": self.margin_ang,
                "radii source": self.radii_source,
                "degenerate_rel_tol": DEGENERATE_REL_TOL}


def _delaunay_in_box(frame: Frame, margin: float):
    """The periodic Delaunay tetrahedra, one copy each (module docstring)."""
    from scipy.spatial import Delaunay

    spacing = _spacing_ang(frame)
    attempts: list[tuple[float, float]] = []
    for _ in range(MARGIN_ATTEMPTS):
        atom, image, points = _padded(frame, margin)
        simplices = Delaunay(points).simplices
        # every point of a set of distinct points is a Delaunay vertex;
        # Qhull leaves out one that lies on another (the atoms come first)
        vertex = np.zeros(points.shape[0], dtype=bool)
        vertex[simplices.ravel()] = True
        left_out = np.flatnonzero(~vertex[:frame.n_atoms])
        if left_out.size:
            raise ValueError(_coincident_message(
                frame, left_out,
                what="are a vertex of no Delaunay tetrahedron: Qhull left "
                     "them out of the triangulation",
                refused="is refused rather than left out of the tetrahedra"))
        p0 = points[simplices[:, 0]]
        edges = points[simplices[:, 1:]] - p0[:, None, :]       # (T, 3, 3)
        volume = np.abs(np.linalg.det(edges)) / 6.0
        solid = volume > DEGENERATE_REL_TOL * spacing ** 3
        rhs = 0.5 * (edges[solid] ** 2).sum(axis=2)
        offset = np.linalg.solve(edges[solid], rhs[..., None])[..., 0]
        centre = p0[solid] + offset
        radius = np.linalg.norm(offset, axis=1)
        frac = _frac_of(centre, frame.box_ang)
        whole = np.round(frac)
        frac = np.where(np.abs(frac - whole) < FACE_SNAP_FRAC, whole, frac)
        inside = (np.floor(frac) == 0).all(axis=1)
        need = float(radius[inside].max()) if inside.any() else math.inf
        attempts.append((margin, need))
        if need <= margin:
            break
        margin = _next_margin(margin, need)
    else:
        raise ValueError(
            f"the Delaunay tetrahedra did not pass the reach check after "
            f"{MARGIN_ATTEMPTS} tessellations (margins and circumradii: "
            f"{attempts})")
    kept = np.flatnonzero(solid)[inside]
    flat_cells = simplices[~solid]
    flat_centroid = _frac_of(points[flat_cells].mean(axis=1), frame.box_ang)
    n_degenerate = int((np.floor(flat_centroid) == 0).all(axis=1).sum())
    verts = atom[simplices[kept]]                               # (T, 4)
    images = image[simplices[kept]]                             # (T, 4, 3)
    # one copy per periodic tetrahedron: a key that does not depend on the
    # image, vertices sorted by (row, image), images relative to the first
    t = verts.shape[0]
    order = np.lexsort((images[..., 2].ravel(), images[..., 1].ravel(),
                        images[..., 0].ravel(), verts.ravel(),
                        np.repeat(np.arange(t), 4)))     # row is the first key
    sv = verts.ravel()[order].reshape(t, 4)
    si = images.reshape(-1, 3)[order].reshape(t, 4, 3)
    key = np.concatenate((sv, (si - si[:, :1]).reshape(t, 12)), axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    duplicates = t - first.size
    sel = np.flatnonzero(inside)[first]
    return (verts[first], images[first], frac[sel], radius[sel],
            volume[solid][sel], centre[sel], margin, attempts, n_degenerate,
            duplicates)


def empty_spheres(frame: Frame, radii_ang: Mapping[str, float], *,
                  radii_source: str,
                  margin_ang: float | None = None) -> EmptySpheres:
    """The empty sphere at the circumcentre of every Delaunay tetrahedron.

    ``radii_ang`` (element -> Å) and ``radii_source`` are required: no radius
    is assumed (:func:`vdw_radii_ang` gives FACET's van der Waals radii and
    :data:`VDW_RADII_SOURCE` their source). Radii of 0 give the circumradii.
    ValueError when the tetrahedra do not fill the box volume within
    :data:`VOLUME_LAW_TOL`, and for an atom that lies on another one (it is
    a vertex of no tetrahedron); a contact closer than ``bulk.D_MIN_ANG`` is
    noted.
    """
    from scipy.spatial import cKDTree

    frame = _check_frame(frame)
    radii = _checked_radii(frame, radii_ang, radii_source)
    given = None if margin_ang is None else _positive(margin_ang, "margin_ang")
    spacing = _spacing_ang(frame)
    margin = given if given is not None else MARGIN_START_SPACINGS * spacing
    (verts, images, centre_frac, circum, volume, centre, margin, attempts,
     n_degenerate, duplicates) = _delaunay_in_box(frame, margin)

    # every atom within reach of a circumcentre: R + r of the largest atom
    r_max = max(radii.values())
    reach = float(circum.max()) + r_max
    atom_q, _, points_q = _padded(frame, reach)
    symbols = frame.elements[atom_q]
    nearest = np.full(circum.size, np.inf)
    clearance = np.full(circum.size, np.inf)
    for symbol, r in radii.items():
        tree = cKDTree(points_q[symbols == symbol])
        dist, _ = tree.query(centre, k=1, distance_upper_bound=float(
            np.nextafter(reach + 1e-9 * spacing, np.inf)))
        nearest = np.minimum(nearest, dist)
        clearance = np.minimum(clearance, dist - r)

    box_volume = frame.volume_ang3
    total = math.fsum(volume)
    relative = abs(total - box_volume) / box_volume
    notes = [_margin_note("Delaunay tetrahedra", attempts, given),
             f"{circum.size} tetrahedra; their volumes sum to {total:.12g} Å^3 "
             f"and the box holds {box_volume:.12g} Å^3 (relative difference "
             f"{relative:.2e})",
             f"radii: " + ", ".join(f"{s} {r:g} Å" for s, r in radii.items())
             + f" ({radii_source.strip()})"]
    if relative > VOLUME_LAW_TOL:
        raise ValueError(
            f"the Delaunay tetrahedra fill {total:.12g} Å^3 of a "
            f"{box_volume:.12g} Å^3 box (relative difference {relative:.2e}, "
            f"above VOLUME_LAW_TOL {VOLUME_LAW_TOL:g}); a circumcentre on a "
            "box face, or a degenerate (cospherical) arrangement, can do this")
    if duplicates:
        notes.append(f"{duplicates} copies of tetrahedra already kept were "
                     "removed (a circumcentre within rounding of a box face)")
    # each atom's nearest neighbour is joined to it by a Delaunay edge, and
    # every periodic edge has a copy in a kept tetrahedron (about 80 ms for
    # the 74 804 tetrahedra of tools/bench_md.make_box(22), measured)
    corner = ((frame.frac[verts.ravel()] + images.reshape(-1, 3))
              @ frame.box_ang).reshape(verts.shape[0], 4, 3)
    ends = np.array(_SIX_PAIRS)
    span = corner[:, ends[:, 1]] - corner[:, ends[:, 0]]           # (T, 6, 3)
    edge_ang = np.sqrt(np.einsum("tkj,tkj->tk", span, span))
    if edge_ang.size and float(edge_ang.min()) < bulk.D_MIN_ANG:
        short_t, short_k = np.nonzero(edge_ang < bulk.D_MIN_ANG)
        close = np.unique(np.concatenate(
            (verts[short_t, ends[short_k, 0]], verts[short_t, ends[short_k, 1]])))
        notes.append(f"{close.size} atom(s) have a Delaunay neighbour closer "
                     f"than {bulk.D_MIN_ANG:g} Å (bulk.D_MIN_ANG, below which "
                     "the pair search counts a contact as an atom in its own "
                     "image); the closest pair is "
                     f"{float(edge_ang.min()):.3g} Å apart (rows "
                     f"{_rows_text(close)})")
    if n_degenerate:
        notes.append(f"{n_degenerate} tetrahedra of zero volume (Qhull's "
                     "triangulation of atoms on one sphere) have no "
                     "circumsphere and are not listed")
    breach = nearest < circum - 1e-9 * spacing
    if breach.any():
        worst = float((circum - nearest)[breach].max())
        notes.append(f"{int(breach.sum())} circumsphere(s) hold an atom centre, "
                     f"by up to {worst:.3g} Å: not the Delaunay property; the "
                     "radius given is still the measured clearance")
    inside = clearance < 0
    if inside.any():
        notes.append(f"{int(inside.sum())} of {circum.size} circumcentres lie "
                     "inside an atom sphere; their radius is negative, the "
                     "depth inside")
    return EmptySpheres(
        vertices=_frozen(verts), vertex_image=_frozen(images),
        centre_frac=_frozen(centre_frac), circumradius_ang=_frozen(circum),
        radius_ang=_frozen(clearance), volume_ang3=_frozen(volume),
        radii_ang=dict(radii), radii_source=radii_source.strip(),
        margin_ang=float(margin), volume_sum_ang3=total,
        box_volume_ang3=box_volume, n_degenerate=n_degenerate,
        notes=tuple(notes))


# ---------------------------------------------------------------------------
# free volume
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FreeVolume:
    """Free-volume fractions of one frame, measured on a grid.

    ``geometric_fraction``: outside every atom sphere. ``probe_centre_fraction``:
    where a probe of ``probe_radius_ang`` can be centred without overlapping
    an atom. ``probe_occupiable_fraction``: what such probes can cover.
    """

    probe_radius_ang: float
    grid_shape: tuple[int, int, int]
    grid_spacing_ang: tuple[float, float, float]
    requested_grid_spacing_ang: float
    n_points: int
    geometric_fraction: float
    probe_centre_fraction: float
    probe_occupiable_fraction: float
    box_volume_ang3: float
    radii_ang: Mapping[str, float]
    radii_source: str
    notes: tuple[str, ...] = ()

    @property
    def method_parameters(self) -> dict:
        return {"probe radius (Å)": self.probe_radius_ang,
                "grid spacing requested (Å)": self.requested_grid_spacing_ang,
                "grid spacing (Å)": self.grid_spacing_ang,
                "radii (Å)": dict(self.radii_ang),
                "radii source": self.radii_source}


def _grid_shape(frame: Frame, spacing_ang: float) -> np.ndarray:
    lengths = np.linalg.norm(frame.box_ang, axis=1)
    return np.maximum(1, np.ceil(lengths / spacing_ang)).astype(np.int64)


def _stencil(box: np.ndarray, shape: np.ndarray, widths: np.ndarray,
             reach: float) -> np.ndarray:
    """Integer grid offsets o that can hold a point within ``reach`` of an
    atom, counted from the grid cell the atom sits in.

    A grid point at offset o lies at (o - t) / n in cell units from an atom
    whose position within its grid cell is t in [0, 1)^3, so its distance is
    at least |(o / n) . box| - max_t |(t / n) . box|, and the largest of the
    latter is at a corner of the unit cube. Every offset that can be within
    ``reach`` is therefore in |(o / n) . box| <= reach + that corner length;
    each (atom, point) pair kept is then tested exactly.
    """
    corners = np.array(list(itertools.product((0, 1), repeat=3)), float)
    corner = float(np.linalg.norm((corners / shape) @ box, axis=1).max())
    bound = reach + corner
    half = np.ceil(bound * shape / widths).astype(np.int64) + 1
    axes = [np.arange(-h, h + 1) for h in half]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    length = np.linalg.norm((grid / shape) @ box, axis=1)
    return grid[length <= bound]


def _mark_within(frame: Frame, rows: np.ndarray, shape: np.ndarray,
                 radii: Sequence[float], masks: Sequence[np.ndarray]) -> None:
    """Set masks[k] at every grid point strictly within radii[k] of an atom of
    ``rows`` (periodic); every (atom, grid point) pair in reach is tested.

    The grid point at offset o from the cell of an atom whose position in
    that cell is t lies at ((o - t) / n) . box from it, the difference of two
    vectors computed once each; the index wraps modulo the grid.
    """
    reach = max(radii)
    if reach <= 0.0 or not rows.size:
        return
    box = frame.box_ang
    offsets = _stencil(box, shape, frame.perpendicular_widths_ang, reach)
    offset_xyz = (offsets / shape) @ box
    u = frame.frac[rows] * shape - 0.5
    base = np.floor(u).astype(np.int64)
    t_xyz = ((u - base) / shape) @ box
    per_chunk = max(1, _GRID_WORK_PER_CHUNK // offsets.shape[0])
    order = np.argsort(radii)[::-1]            # widest first: the others nest
    squared = [radii[k] * radii[k] for k in order]
    for start in range(0, rows.size, per_chunk):
        vec = offset_xyz[None, :, :] - t_xyz[start:start + per_chunk, None, :]
        d2 = np.einsum("aok,aok->ao", vec, vec)
        atom, off = np.nonzero(d2 < squared[0])
        if not atom.size:
            continue
        near = d2[atom, off]
        index = np.mod(base[start + atom] + offsets[off], shape)
        flat = np.ravel_multi_index((index[:, 0], index[:, 1], index[:, 2]),
                                    tuple(shape))
        for k, r2 in zip(order, squared):
            if r2 > 0.0:
                masks[k][flat[near < r2]] = True


def _admitted(cand: np.ndarray, owner: np.ndarray, centres: np.ndarray,
              big_r: np.ndarray, probe: float, tol: float) -> np.ndarray:
    """(M,) whether any candidate probe centre of each grid point lies within
    the probe radius of the point and outside every expanded sphere of the
    point's neighbourhood, each to ``tol``. ``cand`` (Q, 3) are relative to
    the point ``owner`` (Q,) of ``centres`` (M, s, 3) and ``big_r`` (M, s);
    the candidates are tested in blocks of :data:`_EXACT_WORK_PER_CHUNK`
    (candidate, sphere) pairs. |c - a|^2 is formed as |c|^2 - 2 c.a + |a|^2:
    with |c| <= r_p and |a| < r + 2 r_p its rounding is near 1e-14 Å^2,
    far below the tolerance."""
    m, s = big_r.shape
    hit = np.zeros(m, dtype=bool)
    floor = np.maximum(big_r - tol, 0.0) ** 2
    norm_a = np.einsum("msj,msj->ms", centres, centres)
    norm_c = np.einsum("qj,qj->q", cand, cand)
    rows = np.flatnonzero(norm_c <= (probe + tol) ** 2)
    step = max(1, _EXACT_WORK_PER_CHUNK // max(1, s))
    for start in range(0, rows.size, step):
        part = rows[start:start + step]
        who = owner[part]
        c = cand[part]
        d2 = (norm_c[part, None] + norm_a[who]
              - 2.0 * np.einsum("qj,qsj->qs", c, centres[who]))
        hit[who[(d2 >= floor[who]).all(axis=1)]] = True
    return hit


def _circle_nearest(ai, aj, ri, rj):
    """Nearest point to the origin on the circle where spheres (ai, ri) and
    (aj, rj) meet, for (K, 3) centres of spheres that do meet. When the
    origin lies on the circle's axis every point of the circle is as near,
    and one is taken (module docstring: the three-sphere points cover that
    case)."""
    d = aj - ai
    dd = np.sqrt(np.einsum("kj,kj->k", d, d))
    axis = d / dd[:, None]
    along = (dd * dd + ri * ri - rj * rj) / (2.0 * dd)
    radius = np.sqrt(np.maximum(ri * ri - along * along, 0.0))
    centre = ai + along[:, None] * axis
    w = np.einsum("kj,kj->k", centre, axis)[:, None] * axis - centre
    nw = np.sqrt(np.einsum("kj,kj->k", w, w))
    on_axis = nw <= 1e-12 * np.maximum(radius, 1.0)
    if on_axis.any():
        rows = np.flatnonzero(on_axis)
        helper = np.eye(3)[np.argmin(np.abs(axis[rows]), axis=1)]
        w[rows] = np.cross(axis[rows], helper)
        nw[rows] = np.sqrt(np.einsum("kj,kj->k", w[rows], w[rows]))
    return centre + (radius / nw)[:, None] * w


def _three_sphere_points(a1, a2, a3, r1, r2, r3):
    """The two points where three spheres meet, (Q, 2, 3), and whether they
    do (centres not collinear and the spheres meeting)."""
    d12 = a2 - a1
    dist = np.sqrt(np.einsum("qj,qj->q", d12, d12))
    ok = dist > 0.0
    ex = d12 / np.where(ok, dist, 1.0)[:, None]
    d13 = a3 - a1
    i = np.einsum("qj,qj->q", ex, d13)
    ey = d13 - i[:, None] * ex
    ney = np.sqrt(np.einsum("qj,qj->q", ey, ey))
    ok &= ney > 1e-12 * np.maximum(dist, 1.0)
    ey = ey / np.where(ok, ney, 1.0)[:, None]
    ez = np.cross(ex, ey)
    j = np.einsum("qj,qj->q", ey, d13)
    safe_d = np.where(ok, dist, 1.0)
    safe_j = np.where(ok, j, 1.0)
    x = (r1 * r1 - r2 * r2 + safe_d * safe_d) / (2.0 * safe_d)
    y = (r1 * r1 - r3 * r3 + i * i + j * j) / (2.0 * safe_j) - i * x / safe_j
    z2 = r1 * r1 - x * x - y * y
    ok &= z2 >= 0.0
    z = np.sqrt(np.maximum(z2, 0.0))
    base = a1 + x[:, None] * ex + y[:, None] * ey
    return np.stack((base + z[:, None] * ez, base - z[:, None] * ez),
                    axis=1), ok


def _copies_within(frac: np.ndarray, box: np.ndarray, widths: np.ndarray,
                   reach: float) -> np.ndarray:
    """Positions of the points and of their periodic copies within ``reach``
    of the box: the construction of ``bulk._image_points``, for any points
    (fractions in [0, 1))."""
    margin = reach / widths
    lo = np.ceil(-margin - frac).astype(np.int64)
    hi = np.floor(1.0 + margin - frac).astype(np.int64)
    count = hi - lo + 1
    per = np.prod(count, axis=1)
    point = np.repeat(np.arange(frac.shape[0]), per)
    offset = np.arange(int(per.sum())) - np.repeat(np.cumsum(per) - per, per)
    image = np.empty((point.size, 3), np.int64)
    for axis in (2, 1, 0):
        size = count[point, axis]
        image[:, axis] = lo[point, axis] + offset % size
        offset //= size
    return (frac[point] + image) @ box


def _spheres_meet(pos: np.ndarray, big: np.ndarray, i: np.ndarray,
                  j: np.ndarray) -> np.ndarray:
    """Whether spheres i and j (rows of ``pos``, radii ``big``) meet in a
    circle (or touch): distinct centres no farther apart than the sum of
    the radii and no nearer than their difference."""
    gap = pos[j] - pos[i]
    dist = np.sqrt(np.einsum("pj,pj->p", gap, gap))
    return ((dist > 0.0) & (dist <= big[i] + big[j])
            & (dist >= np.abs(big[i] - big[j])))


def _candidate_triples(pos: np.ndarray, big: np.ndarray, n: int,
                       r_max: float):
    """(triangles (T, 3), edges (E, 2), how) of expanded spheres that can
    carry a vertex, or an arc, of the probe-centre set C; each has at least
    one atom of the box (index < n), and the spheres of each meet pairwise.

    The pairs of spheres that meet come from one KD-tree query. The
    triples are every triple of such pairs (the pair graph's triangles)
    while those are few, at most :data:`_PAIR_GRAPH_PAIRS_PER_ATOM` pairs of
    neighbours to try per atom of the box. Above that (large spheres, which
    each meet a hundred others) only the triangles of the regular
    triangulation are tried: a point v of C on spheres i, j and k has power
    |v - a|^2 - R^2 equal to 0 for those three and 0 or more for every other
    sphere, so v lies in the power cells of all three and (i, j, k) is a
    triangle of the regular (power-weighted Delaunay) triangulation [18];
    likewise a point of C on the circle of i and j makes (i, j) one of its
    edges. That triangulation is the lower hull of the centres lifted to
    (a, |a|^2 - R^2) in four dimensions (scipy's ConvexHull, Qhull [13],
    which triangulates its facets); where Qhull refuses the lifted set (all
    atoms in one plane), the pair graph is used. :data:`_TRIPLES_FROM` can
    force either route. Each candidate point is then tested exactly either
    way; the route only decides how many cannot qualify.
    """
    from scipy import spatial

    edges = spatial.cKDTree(pos).query_pairs(2.0 * r_max,
                                             output_type="ndarray")
    edges = np.sort(edges.reshape(-1, 2), axis=1)
    edges = edges[_spheres_meet(pos, big, edges[:, 0], edges[:, 1])]
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    first = np.searchsorted(edges[:, 0], np.arange(n + 1))
    degree = np.diff(first)
    to_try = float((degree * (degree - 1) // 2).sum())
    hull_route = (_TRIPLES_FROM == "regular triangulation"
                  or (_TRIPLES_FROM == "auto"
                      and to_try > _PAIR_GRAPH_PAIRS_PER_ATOM * n))
    if hull_route:
        qhull_error = getattr(spatial, "QhullError", None) or \
            spatial._qhull.QhullError
        try:
            lifted = np.column_stack((pos, np.einsum("pj,pj->p", pos, pos)
                                      - big * big))
            hull = spatial.ConvexHull(lifted)
        except qhull_error:
            hull = None
        if hull is not None:
            tets = hull.simplices[hull.equations[:, 3] < 0.0]
            faces = np.sort(np.concatenate(
                [tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]],
                 tets[:, [1, 2, 3]]]), axis=1)
            faces = np.unique(faces[faces[:, 0] < n], axis=0)
            faces = faces[_spheres_meet(pos, big, faces[:, 0], faces[:, 1])
                          & _spheres_meet(pos, big, faces[:, 0], faces[:, 2])
                          & _spheres_meet(pos, big, faces[:, 1], faces[:, 2])]
            tri_edges = np.sort(np.concatenate(
                [tets[:, [a, b]] for a, b in _SIX_PAIRS]), axis=1)
            tri_edges = np.unique(tri_edges[tri_edges[:, 0] < n], axis=0)
            tri_edges = tri_edges[_spheres_meet(pos, big, tri_edges[:, 0],
                                                tri_edges[:, 1])]
            return faces, tri_edges, "regular triangulation (lifted hull)"
    n_points = pos.shape[0]
    keys = edges[:, 0] * n_points + edges[:, 1]
    found = []
    for i in np.flatnonzero(degree >= 2):
        nbr = edges[first[i]:first[i + 1], 1]
        a, b = _pair_indices(nbr.size)
        j, k = nbr[a], nbr[b]
        at = np.minimum(np.searchsorted(keys, j * n_points + k),
                        keys.size - 1)
        edge = keys[at] == j * n_points + k
        if edge.any():
            found.append(np.stack((np.full(int(edge.sum()), i), j[edge],
                                   k[edge]), axis=1))
    faces = np.concatenate(found) if found else np.zeros((0, 3), np.int64)
    return faces, edges[edges[:, 0] < n], "every triple of meeting spheres"


@dataclass(frozen=True)
class _ProbeCentreSet:
    """What :func:`_probe_centre_set` measures of the probe-centre set C of
    a frame: ``vertex_tree``, a KD-tree of every point of C where three
    expanded spheres meet (with their periodic copies within r_p of the
    box), or None; ``live``, per atom row, whether any point of its
    expanded sphere is in C; and the counts and route, for the notes."""

    vertex_tree: object
    live: np.ndarray
    n_vertices: int
    n_triples: int
    route: str


def _probe_centre_set(frame: Frame, radii: Mapping[str, float], probe: float,
                      tol: float) -> _ProbeCentreSet:
    """The three-sphere points of C, and the spheres that touch C.

    C is the set of probe centres, outside every expanded sphere (atom
    radius + r_p). Its boundary is made of patches of the spheres, arcs of
    the circles where two meet, and the points where three meet. The
    candidate triples and pairs come from :func:`_candidate_triples`, on the
    atoms and their images within 2 R_max of the box; each triple gives up
    to two points, and those outside every expanded sphere (a KD-tree per
    element, to ``tol``; every sphere that can hold a point within R_max of
    the box is in the set) are the vertices of C. A sphere touches C when
    one of its vertices does, when a point of one of its circles is in C (a
    circle in C with no vertex on it lies wholly in C), or when a point of
    the sphere itself is (a sphere no other cuts). The one- and two-sphere
    candidates of :func:`_occupiable_group` come only from spheres that
    touch C: no point of C lies on the others.
    """
    from scipy.spatial import cKDTree

    n = frame.n_atoms
    big_of = {s: float(r + probe) for s, r in radii.items()}
    r_max = max(big_of.values())
    atom, _, pos = _padded(frame, 2.0 * r_max + 2.0 * tol)
    symbols = frame.elements[atom]
    big = np.array([big_of[str(s)] for s in symbols])
    trees = [(cKDTree(pos[symbols == s]), value)
             for s, value in big_of.items() if (symbols == s).any()]

    def in_c(points: np.ndarray) -> np.ndarray:
        ok = np.ones(points.shape[0], dtype=bool)
        for tree, value in trees:
            if value - tol <= 0.0 or not points.size:
                continue
            d, _ = tree.query(points, k=1, distance_upper_bound=value)
            ok &= ~(d < value - tol)
        return ok

    live = np.zeros(n, dtype=bool)
    # a point of each sphere of the box
    live |= in_c(pos[:n] + big[:n, None] * np.array([0.0, 0.0, 1.0]))
    faces, edges, route = _candidate_triples(pos, big, n, r_max)
    # a point of each circle that can carry an arc of C
    if edges.size:
        gap = pos[edges[:, 1]] - pos[edges[:, 0]]
        dist = np.sqrt(np.einsum("pj,pj->p", gap, gap))
        bi, bj = big[edges[:, 0]], big[edges[:, 1]]
        meet = (dist > 0.0) & (dist <= bi + bj) & (dist >= np.abs(bi - bj))
        edges, gap, dist, bi, bj = (v[meet] for v in
                                    (edges, gap, dist, bi, bj))
        axis = gap / dist[:, None]
        along = (dist * dist + bi * bi - bj * bj) / (2.0 * dist)
        radius = np.sqrt(np.maximum(bi * bi - along * along, 0.0))
        side = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis), axis=1)])
        side /= np.sqrt(np.einsum("pj,pj->p", side, side))[:, None]
        hit = in_c(pos[edges[:, 0]] + along[:, None] * axis
                   + radius[:, None] * side)
        live[atom[edges[hit]].ravel()] = True
    # the points where three meet
    vertices = []
    step = max(1, _EXACT_WORK_PER_CHUNK // 8)
    for start in range(0, faces.shape[0], step):
        tri = faces[start:start + step]
        pts, ok = _three_sphere_points(
            pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]],
            big[tri[:, 0]], big[tri[:, 1]], big[tri[:, 2]])
        both = np.repeat(ok, 2)
        pts = pts.reshape(-1, 3)[both]
        rows = np.repeat(tri, 2, axis=0)[both]
        keep = in_c(pts)
        if keep.any():
            vertices.append(pts[keep])
            live[atom[rows[keep]].ravel()] = True
    tree = None
    n_vertices = 0
    if vertices:
        points = np.concatenate(vertices)
        n_vertices = points.shape[0]
        frac = np.mod(_frac_of(points, frame.box_ang), 1.0)
        frac[frac >= 1.0] = 0.0
        tree = cKDTree(_copies_within(frac, frame.box_ang,
                                      frame.perpendicular_widths_ang,
                                      probe + 2.0 * tol))
    return _ProbeCentreSet(vertex_tree=tree, live=live,
                           n_vertices=n_vertices,
                           n_triples=int(faces.shape[0]), route=route)


@functools.lru_cache(maxsize=256)
def _pair_indices(size: int):
    a, b = np.triu_indices(size, k=1)
    a.setflags(write=False)
    b.setflags(write=False)
    return a, b


def _occupiable_group(centres: np.ndarray, big_r: np.ndarray,
                      live: np.ndarray, positions: np.ndarray,
                      c_set: _ProbeCentreSet, probe: float,
                      tol: float) -> np.ndarray:
    """(M,) whether a probe sphere can cover each grid point, for points that
    each have the same number s of expanded spheres within r_p
    (``centres`` (M, s, 3) relative to the point, ``big_r`` (M, s), ``live``
    (M, s) whether each sphere touches C; ``positions`` (M, 3) the points).

    Exact (module docstring): the point is covered when a point of the
    probe-centre set C lies within r_p of it, and the point of C nearest to
    it is the radial projection onto one sphere, the nearest point of the
    circle where two meet, or a point where three meet. Each kind is tried
    on the points the previous kinds did not settle: one sphere, then the
    three-sphere points of C (a KD-tree query), then two spheres; only
    spheres that touch C give candidates, and every candidate is tested
    against every sphere of the point's neighbourhood.
    """
    m, s = big_r.shape
    found = np.zeros(m, dtype=bool)
    # one sphere: the radial projection of the point onto it
    pm, ps = np.nonzero(live)
    if pm.size:
        a = centres[pm, ps]
        length = np.sqrt(np.einsum("kj,kj->k", a, a))
        zero = length <= 0.0
        unit = np.where(zero[:, None], np.array([0.0, 0.0, -1.0]),
                        -a / np.where(zero, 1.0, length)[:, None])
        found |= _admitted(a + big_r[pm, ps][:, None] * unit, pm, centres,
                           big_r, probe, tol)
    # three spheres: a vertex of C within r_p
    rest = np.flatnonzero(~found)
    if rest.size and c_set.vertex_tree is not None:
        d, _ = c_set.vertex_tree.query(positions[rest], k=1,
                                       distance_upper_bound=probe + tol)
        found[rest[d <= probe + tol]] = True
    if s < 2:
        return found
    # two spheres: the nearest point of their circle
    rest = np.flatnonzero(~found)
    if not rest.size:
        return found
    ii, jj = _pair_indices(s)
    a, r, lv = centres[rest], big_r[rest], live[rest]
    gap = a[:, jj] - a[:, ii]
    dd = np.sqrt(np.einsum("mpj,mpj->mp", gap, gap))
    meet = (lv[:, ii] & lv[:, jj] & (dd > 0.0) & (dd <= r[:, ii] + r[:, jj])
            & (dd >= np.abs(r[:, ii] - r[:, jj])))
    pm, pp = np.nonzero(meet)
    if pm.size:
        point = _circle_nearest(a[pm, ii[pp]], a[pm, jj[pp]], r[pm, ii[pp]],
                                r[pm, jj[pp]])
        close_by = np.einsum("kj,kj->k", point, point) <= (probe + tol) ** 2
        if close_by.any():
            found[rest] |= _admitted(point[close_by], pm[close_by], a, r,
                                     probe, tol)
    return found


def _occupiable_shell(frame: Frame, shape: np.ndarray,
                      radii: Mapping[str, float], probe: float,
                      shell: np.ndarray, tol: float):
    """The shell points (blocked for a probe centre, outside every atom)
    that a probe sphere can cover, decided exactly per point
    (:func:`_occupiable_group`), and the :class:`_ProbeCentreSet` used.

    Each shell point needs the expanded spheres that pass within r_p of
    it: atoms nearer than r + 2 r_p. They are found by a KD-tree of the
    shell points against one of the atoms and their images within
    R_max + r_p of the box, so the cost follows the shell points and their
    spheres; walking every grid point near every atom instead (as the
    atom masks do) visits the points inside atoms too, which with van der
    Waals radii and a 1.4 Å probe were nine in ten (measured). The shell
    points are taken in chunks of about :data:`_SHELL_PAIRS_PER_SLAB`
    (point, sphere) pairs.
    """
    from scipy.spatial import cKDTree

    covered = np.zeros(shell.size, dtype=bool)
    c_set = _probe_centre_set(frame, radii, probe, tol)
    flat = np.flatnonzero(shell)
    if not flat.size:
        return covered, c_set
    big_of = {s: float(r + probe) for s, r in radii.items()}
    reach = max(big_of.values()) + probe
    atom, _, pos = _padded(frame, reach + 2.0 * tol)
    big_all = np.array([big_of[str(s)] for s in frame.elements[atom]])
    live_all = c_set.live[atom]
    atoms_tree = cKDTree(pos)
    density = frame.n_atoms / frame.volume_ang3
    mean_reach = sum((r + 2.0 * probe) ** 3 * np.count_nonzero(
        frame.elements == s) for s, r in radii.items()) / frame.n_atoms
    per_point = max(1.0, 4.0 / 3.0 * math.pi * density * mean_reach)
    per_chunk = max(1, int(_SHELL_PAIRS_PER_SLAB / per_point))
    for start in range(0, flat.size, per_chunk):
        part = flat[start:start + per_chunk]
        where = np.stack(np.unravel_index(part, tuple(shape)), axis=1)
        xyz = ((where + 0.5) / shape) @ frame.box_ang
        found = cKDTree(xyz).sparse_distance_matrix(
            atoms_tree, reach, output_type="ndarray")
        point, sphere = found["i"], found["j"]
        vec = xyz[point] - pos[sphere]
        dist = np.sqrt(np.einsum("pj,pj->p", vec, vec))
        keep = dist < big_all[sphere] + probe
        point, sphere, vec = point[keep], sphere[keep], vec[keep]
        order = np.argsort(point, kind="stable")
        point, sphere, vec = point[order], sphere[order], vec[order]
        unique, first, count = np.unique(point, return_index=True,
                                         return_counts=True)
        for s in np.unique(count):
            rows = np.flatnonzero(count == s)
            # (the pairs of a point cost about s^3 / 2 sphere tests)
            step = max(1, _EXACT_WORK_PER_CHUNK // (int(s) ** 3 + 1))
            for lo in range(0, rows.size, step):
                sub = rows[lo:lo + step]
                idx = first[sub][:, None] + np.arange(int(s))
                covered[part[unique[sub]]] = _occupiable_group(
                    -vec[idx], big_all[sphere[idx]], live_all[sphere[idx]],
                    xyz[unique[sub]], c_set, probe, tol)
    return covered, c_set


def free_volume(frame: Frame, radii_ang: Mapping[str, float], *,
                radii_source: str, probe_radius_ang: float,
                grid_spacing_ang: float) -> FreeVolume:
    """Geometric, probe-centre and probe-occupiable free-volume fractions.

    Every argument is required: the radii and their source (none assumed),
    the probe radius (0 gives the geometric fraction three times), and the
    grid spacing h, a method choice whose effect the result states (the box
    is cut into ceil(|a_k| / h) cells along each cell vector, a point at the
    centre of each). Each grid point is classified exactly for all three
    fractions (module docstring); the grid is the only approximation.
    """
    frame = _check_frame(frame)
    radii = _checked_radii(frame, radii_ang, radii_source)
    probe = _nonnegative(probe_radius_ang, "probe_radius_ang")
    h = _positive(grid_spacing_ang, "grid_spacing_ang")
    shape = _grid_shape(frame, h)
    total = int(np.prod(shape))
    if total > GRID_POINTS_LIMIT:
        raise ValueError(f"a grid spacing of {h:g} Å gives {total} points, "
                         f"above GRID_POINTS_LIMIT ({GRID_POINTS_LIMIT}, a "
                         "memory choice); a coarser grid is needed")
    core = np.zeros(total, dtype=bool)
    blocked = np.zeros(total, dtype=bool)
    for symbol, r in radii.items():
        rows = np.flatnonzero(frame.elements == symbol)
        _mark_within(frame, rows, shape, (r, r + probe), (core, blocked))
    n_core = int(np.count_nonzero(core))
    n_centre = total - int(np.count_nonzero(blocked))
    if probe == 0.0:
        n_occupiable = n_centre
        method = "probe radius 0: the probe-centre points themselves"
    else:
        blocked &= ~core                      # now the shell points
        tol = DEGENERATE_REL_TOL * _spacing_ang(frame)
        covered, c_set = _occupiable_shell(frame, shape, radii, probe,
                                           blocked, tol)
        n_occupiable = n_centre + int(np.count_nonzero(covered))
        method = ("an exact test per point: the probe-centre point nearest "
                  "to it is a radial projection onto one expanded sphere, the "
                  "nearest point of a circle where two meet, or a point where "
                  "three meet, each tested against every expanded sphere "
                  f"within reach (to {tol:.3g} Å); {c_set.n_vertices} "
                  f"three-sphere points of {c_set.n_triples} candidate "
                  f"triples ({c_set.route}) lie among the probe centres, "
                  f"and {int(c_set.live.sum())} of {frame.n_atoms} expanded "
                  "spheres touch them")
    spacing = tuple(float(v) for v in
                    np.linalg.norm(frame.box_ang, axis=1) / shape)
    notes = [f"grid of {shape[0]} x {shape[1]} x {shape[2]} = {total} points, "
             f"spacing {spacing[0]:.4g}, {spacing[1]:.4g}, {spacing[2]:.4g} Å "
             "along a, b, c; the fractions are those of the grid points",
             "radii: " + ", ".join(f"{s} {r:g} Å" for s, r in radii.items())
             + f" ({radii_source.strip()})",
             "no connectivity analysis: every pocket counts, connected to "
             "another or not",
             f"probe-occupiable points: grid points within the probe radius of "
             f"a probe-centre point (any point, not only grid points), found "
             f"by {method}"]
    return FreeVolume(
        probe_radius_ang=probe, grid_shape=tuple(int(v) for v in shape),
        grid_spacing_ang=spacing, requested_grid_spacing_ang=h,
        n_points=total,
        geometric_fraction=(total - n_core) / total,
        probe_centre_fraction=n_centre / total,
        probe_occupiable_fraction=n_occupiable / total,
        box_volume_ang3=frame.volume_ang3, radii_ang=dict(radii),
        radii_source=radii_source.strip(), notes=tuple(notes))


def vdw_radii_ang(symbols) -> dict[str, float]:
    """FACET's van der Waals radius of each element, in Å.

    From ``elements.info().vdw_radius`` (gemmi's ``Element.vdw_r``), rounded to
    the two decimals gemmi's table holds; the source is
    :data:`VDW_RADII_SOURCE`. ValueError for an element with none.
    """
    if isinstance(symbols, Frame):
        symbols = symbols.species
    out = {}
    for symbol in symbols:
        info = element_data.info(str(symbol))
        if info.symbol != str(symbol) or not info.vdw_radius:
            raise ValueError(f"no van der Waals radius for {symbol!r} in "
                             "FACET's element data")
        out[str(symbol)] = round(float(info.vdw_radius), 2)
    return out


# ---------------------------------------------------------------------------
# frame averages through md_stats
# ---------------------------------------------------------------------------

def _threshold_note(cells: VoronoiCells) -> str:
    area = (f"face area above {cells.min_face_area_ang2:g} Å^2"
            if cells.min_face_area_ang2 > 0.0 else "no face-area threshold")
    edge = (f"edges longer than {cells.min_edge_ang:g} Å"
            if cells.min_edge_ang > 0.0 else "no edge threshold")
    return (f"{area}; {edge} (min_face_area_ang2, min_edge_ang); edges at or "
            "below the degeneracy floor (DEGENERATE_REL_TOL x the mean "
            "spacing) contracted")


def _radii_note(radii: Mapping[str, float], source: str) -> str:
    return ("radii: " + ", ".join(f"{s} {r:g} Å" for s, r in radii.items())
            + f" ({source})")


def element_histograms(per_frame: Sequence[tuple[np.ndarray, np.ndarray]],
                       edges, *, name: str, unit: str,
                       frames: Sequence[int] | None = None,
                       density: bool = False,
                       elements: Sequence[str] | None = None,
                       notes: Sequence[str] = ()) -> dict[str, Histogram]:
    """One frame-averaged histogram per element of a per-atom quantity.

    ``per_frame`` holds one (elements, values) pair of (N,) arrays per frame,
    for example ``(result.elements, result.values('q', 6))``. NaN values (an
    atom the quantity is not defined for) are counted by the histogram as
    non-finite, never dropped silently. ``elements`` limits and orders the
    output; by default every element seen, alphabetically.

    The values arrive without their method, so one definition across the
    frames (the same neighbour list definition, degrees and selection) is
    the caller's to keep, and ``notes`` is where to state it, for example the
    result's ``definition``. The functions below, which receive the results
    themselves, refuse a mix.
    """
    if isinstance(per_frame, (str, bytes, Mapping)):
        raise ValueError("one (elements, values) pair per frame is needed")
    rows = []
    for k, item in enumerate(per_frame):
        try:
            symbols, values = item
        except (TypeError, ValueError):
            raise ValueError(f"frame {k}: an (elements, values) pair is "
                             "needed") from None
        symbols = np.asarray(symbols)
        values = np.asarray(values, dtype=np.float64)
        if symbols.shape != values.shape or symbols.ndim != 1:
            raise ValueError(f"frame {k}: elements {symbols.shape} and values "
                             f"{values.shape} are not the same 1-D length")
        rows.append((symbols, values))
    if not rows:
        raise ValueError("no frame was given")
    seen = sorted(set().union(*(set(s.tolist()) for s, _ in rows)))
    chosen = seen if elements is None else [str(e) for e in elements]
    out = {}
    for symbol in chosen:
        out[symbol] = Histogram.from_samples(
            [values[symbols == symbol] for symbols, values in rows], edges,
            name=f"{name} ({symbol})", unit=unit, density=density,
            frames=frames, notes=notes)
    return out


def _one_setting(results, what: str, **getters) -> None:
    """ValueError when the frames' results differ in any named method
    parameter: an average over frames needs one method."""
    for label, get in getters.items():
        seen = []
        for r in results:
            value = get(r)
            if value not in seen:
                seen.append(value)
        if len(seen) > 1:
            raise ValueError(f"{what}: the frames were measured with "
                             f"different {label} ({seen}); an average over "
                             "frames needs one")


def _index_counts(cells: VoronoiCells, element: str | None,
                  length: int) -> Counter:
    mask = np.ones(cells.n_atoms, bool) if element is None else \
        cells.elements == element
    index = cells.index(length)
    return Counter(index[i] for i in np.flatnonzero(mask))


def voronoi_index_distribution(cells: Sequence[VoronoiCells], *,
                               element: str | None = None,
                               frames: Sequence[int] | None = None,
                               kind: str = "fraction") -> Distribution:
    """Frame-averaged distribution of the Voronoi index <n3, n4, n5, n6, ...>.

    The index runs to the largest edge count found in any frame (at least
    n6), the same length in every frame, so the keys compare across frames.
    ``element`` restricts it to atoms of one element.
    """
    cells = list(cells)
    if not cells or not all(isinstance(c, VoronoiCells) for c in cells):
        raise ValueError("one VoronoiCells per frame is needed")
    _one_setting(cells, "Voronoi index",
                 **{"min_face_area_ang2": lambda c: c.min_face_area_ang2,
                    "min_edge_ang": lambda c: c.min_edge_ang})
    length = max(c.edge_counts.shape[1] for c in cells)
    rows = [_index_counts(c, element, length) for c in cells]
    label = "all atoms" if element is None else element
    return Distribution.from_counts(
        rows, name=f"Voronoi index <n3..n{length + 2}> ({label})", kind=kind,
        frames=frames, notes=(_threshold_note(cells[0]),))


def voronoi_cn_distribution(cells: Sequence[VoronoiCells], *,
                            element: str | None = None,
                            frames: Sequence[int] | None = None,
                            kind: str = "fraction") -> Distribution:
    """Frame-averaged distribution of the Voronoi face count (Voronoi CN).

    The count is ``n_faces``: every face above the area threshold, including
    faces with fewer than three edges above ``min_edge_ang``. OVITO's
    VoronoiAnalysisModifier leaves those out of its coordination when an
    edge threshold is set: on a 3 000-atom SiO2 glass frame with
    min_edge_ang = 0.1 Å, ``n_faces`` equalled OVITO 3.16's coordination
    for 1 509 atoms, while ``edge_counts.sum(axis=1)`` equalled it for all
    3 000, as it did with no threshold and with (area, edge) thresholds of
    (0.1 Å^2, 0), (0.1 Å^2, 0.1 Å) and (0.5 Å^2, 0.2 Å); the indices were
    equal for all 3 000 each time (measured 2026-10-07).
    """
    cells = list(cells)
    if not cells or not all(isinstance(c, VoronoiCells) for c in cells):
        raise ValueError("one VoronoiCells per frame is needed")
    _one_setting(cells, "Voronoi CN",
                 **{"min_face_area_ang2": lambda c: c.min_face_area_ang2,
                    "min_edge_ang": lambda c: c.min_edge_ang})
    rows = []
    for c in cells:
        mask = np.ones(c.n_atoms, bool) if element is None else \
            c.elements == element
        rows.append(Counter(int(v) for v in c.n_faces[mask]))
    label = "all atoms" if element is None else element
    return Distribution.from_counts(rows, name=f"Voronoi CN ({label})",
                                    kind=kind, frames=frames,
                                    notes=(_threshold_note(cells[0]),))


def empty_sphere_histogram(spheres: Sequence[EmptySpheres], edges_ang, *,
                           frames: Sequence[int] | None = None,
                           density: bool = False) -> Histogram:
    """Frame-averaged histogram of the empty-sphere radii, one per tetrahedron.

    ``edges_ang``: the bin edges, in Å. The frames must share one set of
    atom radii and its source."""
    spheres = list(spheres)
    if not spheres or not all(isinstance(s, EmptySpheres) for s in spheres):
        raise ValueError("one EmptySpheres per frame is needed")
    _one_setting(spheres, "empty spheres",
                 **{"radii (Å)": lambda s: dict(s.radii_ang),
                    "radii source": lambda s: s.radii_source})
    return Histogram.from_samples(
        [s.radius_ang for s in spheres], edges_ang, name="empty-sphere radius",
        unit="Å", density=density, frames=frames,
        notes=(_radii_note(spheres[0].radii_ang, spheres[0].radii_source),))


def free_volume_scalars(results: Sequence[FreeVolume], *,
                        frames: Sequence[int] | None = None
                        ) -> dict[str, Scalar]:
    """The three free-volume fractions, each averaged over frames.

    The frames must share the probe radius, the atom radii and their source,
    and the grid spacing asked for (each frame's grid then follows its own
    box)."""
    results = list(results)
    if not results or not all(isinstance(r, FreeVolume) for r in results):
        raise ValueError("one FreeVolume per frame is needed")
    _one_setting(results, "free volume",
                 **{"probe radii (Å)": lambda r: r.probe_radius_ang,
                    "radii (Å)": lambda r: dict(r.radii_ang),
                    "radii source": lambda r: r.radii_source,
                    "grid spacings asked for (Å)":
                        lambda r: r.requested_grid_spacing_ang})
    note = (f"probe radius {results[0].probe_radius_ang:g} Å; grid spacing "
            f"{results[0].requested_grid_spacing_ang:g} Å asked for; "
            + _radii_note(results[0].radii_ang, results[0].radii_source))
    out = {}
    for key, label in (("geometric_fraction", "geometric free-volume fraction"),
                       ("probe_centre_fraction", "probe-centre fraction"),
                       ("probe_occupiable_fraction",
                        "probe-occupiable fraction")):
        out[key] = Scalar(label, "1", np.array([getattr(r, key)
                                                for r in results]),
                          frames=frames, notes=(note,))
    return out
