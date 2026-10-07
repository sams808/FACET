"""Medium-range order of an MD frame: rings, coordination sequences, polyhedral
sharing, connected networks and chemical short-range order.

Short-range descriptors (the coordination number, Q^n, bridging and
non-bridging anions) stop at the first or second neighbour. Two glasses with
the same Q^n can still differ in how their units link beyond that: the sizes
of the rings the network closes, how fast the number of neighbours grows shell
by shell, whether polyhedra share corners, edges or faces, whether the formers
make one network that spans the box or several pieces, and whether modifiers
gather in clusters. This module measures those quantities on one frame; the
frame averages go through :mod:`facet.core.md_stats` like every other MD
descriptor.

THE GRAPH, AND WHERE ITS EDGES COME FROM
----------------------------------------
Everything here runs on a :class:`NetworkGraph`: one node per atom of the box
that the caller selects, one edge per bond, and on every edge the lattice
translation of its far end. That is the labelled quotient graph of a periodic
net [1, 2]. The infinite network the box stands for is its lift, whose
vertices are (atom, cell) pairs; every path search below runs on the lift, so
a closed walk inside the box whose translations do not cancel (one that winds
once around the periodic box) is a path to another copy of its start, not a
ring.

The edges never come from a second neighbour search. There are three
builders, each reading what the bulk engine already found:

* :func:`graph_from_bonds`: the cation-anion bonds that the coordination
  number counts at ``v_bond`` (:func:`facet.core.bulk.bonds_at`), so a ring,
  a shared corner and a CN are read from the same bonds;
* :func:`graph_from_pairs`: the pairs of one :func:`facet.core.bulk.iter_pairs`
  search kept where d <= a cutoff for their element pair. A cutoff is the
  caller's (the first minimum of a partial g(r) measured from the model);
  each one needs a source text, and none is set here;
* :func:`bridged_graph`: formers linked when they share an anion bonded to
  both (the T-O-T link of a silicate, B-O-Si of a borosilicate). The formers
  and the bridging anions are arguments with no default.

RINGS
-----
Three definitions are implemented, each as published:

* **King's criterion** [3], as R.I.N.G.S. states it: "a ring as the shortest
  path between two of the nearest neighbors of a given node" [7]. For every
  node and every pair of its neighbours, the shortest closed paths through
  both bonds. Only the node itself is barred from the path, as that wording
  says: the path may pass through another neighbour of the node, and the
  ring then has a chord at the node (a five-fold Si, an O bonded to three
  Si). vitrum 1.1.0 also bars the node's other neighbours; run outside
  FACET on one frame of the 3000-atom SiO2 glass (Si-O graph within its
  measured g(r) minimum, rings of up to 24 atoms) that rule gave 2869 King
  rings against 2891 here: the 40 found only here pass through another
  neighbour of their node, and the 18 found only there are the longer paths
  that rule takes at those angles instead. On the Na2O-3SiO2 glass, which
  has no such chord, both gave the same 1430 (2026-10-07). Which rule the
  R.I.N.G.S. program applies was not checked (reference to verify).
* **Guttman's criterion** [4]: "a ring as the shortest path which comes back
  to a given node (or atom) from one of its nearest neighbors" [7]. For every
  bond, the shortest closed paths through it.
* **Primitive rings** [5, 6]: a ring is primitive "if it can not be
  decomposed into two smaller rings" [7]. The test applied, as in Yuan and
  Cormack's algorithm [6] and the manual's figure 5.17 [7], is the shortcut:
  a path between two members shorter than the ring path between them splits
  the ring into two smaller rings, so a ring is kept when it has none. The
  strong-ring criterion (no decomposition into any number of smaller rings)
  is not implemented: the manual limits it to "relatively simple cases, like
  crystals", for its CPU time on amorphous systems and because it cannot be
  searched to the same depth as the other criteria [7].

When several closed paths tie for the shortest, every one of them is a ring
here, and R.I.N.G.S. counts them the same way: case a) of the manual's
figure 5.22 [7] (two hexagons sharing a five-node path whose ends a square
closes, 16 nodes) has R_N(6) = 10/16 under King's and Guttman's criteria in
its Table 5.4, which needs both tied shortest paths at the three inner
nodes of the shared path (one path per angle would give 7/16). The tests
rebuild that network and pin its R_C, R_N, P_N, P_max and P_min from
Tables 5.4, 5.7 and 5.8 under all three criteria, but for one entry: the
table prints P_max(4) = 0.5 for primitive rings, while its own P_N(6) =
7/16 puts all four nodes of the square on a hexagon as well, so no node has
4 as its largest size and P_max(4) is 0, which is what this module gives
(the printed 0.5 equals the King and Guttman entry). A ring is stored once,
in a canonical form: the cyclic sequence of its lifted members, started at
the lowest atom row with that member in cell (0, 0, 0), in the direction
that gives the smaller sequence.
A ring and its lattice translates are one ring, a ring read backwards is the
same ring, and a ring may pass through two images of one atom in a small box.

**How primitive rings are found.** If a ring of n members has a shortcut
between a and b (graph distance d_G(a, b) < ring distance k <= n/2), then a
member's distance to its antipode is short too: for even n, a's antipode a'
lies n/2 along the ring, and d_G(a, a') <= d_G(a, b) + n/2 - k < n/2; for odd
n, the same holds for the antipode on b's side. So a ring is primitive exactly
when every member reaches its antipode(s) in floor(n/2) steps, which is the
statement that, seen from every member, the ring is two shortest paths to its
far point (even n) or to the two ends of its far edge (odd n). The search runs
one breadth-first search per node to depth floor(max_size / 2), enumerates
every such pair of internally disjoint shortest paths, and keeps a ring when
it was produced from all n of its members. No distance matrix is built, and
the test needs no second search. A ring of at most ``max_size`` members is
primitive or not whatever ``max_size`` is. The nodes are searched in
breadth-first order over the graph, so the members of a ring come close
together in that order, and a candidate cycle is held only from its first
member in the order until a member that did not produce it (it is dropped)
or its last member (it is kept): the memory follows the candidates open at
once, which the result reports (``peak_open_candidates``), not every
candidate ever formed (``n_candidates``).

**Normalisation** follows R.I.N.G.S. [7, 8], over the N nodes of the graph
(isolated nodes included), every node initiating the search:

* R_C(n): distinct rings of n members, divided by N;
* R_N(n): rings of n members found from each node, summed over nodes and
  divided by N. For King and Guttman, a ring is found from the node whose
  angle or bond it closes; for primitive rings, from each of its members.
  Rings are counted on the lift, through the node's copy in cell (0, 0, 0):
  in a small box a ring can pass through two images of one atom, and it is
  then two rings through that copy. Counted that way, a one-atom simple
  cubic cell and a 27-atom one give the same 12 squares per atom (counted
  once per atom, the one-atom cell had given 3, a measured case);
* P_N(n): the fraction of nodes from which at least one ring of n members
  is found;
* P_max(n) = P_Nmax(n) / P_N(n) and P_min(n) = P_Nmin(n) / P_N(n), where
  P_Nmax(n) (P_Nmin(n)) is the fraction of nodes for which n is the largest
  (smallest) ring size found from them. NaN where P_N(n) = 0. Both depend
  on ``max_size``: the largest size found from a node is the largest of at
  most ``max_size``, so P_max(max_size) = 1 wherever P_N(max_size) > 0, by
  construction (the manual's "Pmax(smax) = 1"), and P_max at smaller sizes
  falls as ``max_size`` grows and larger rings are found. The result's notes
  say so; R_C, R_N, P_N and P_min at sizes of at most ``max_size`` do not
  depend on it.

**Ring size.** A size is the number of nodes of the graph the ring passes
through. On a graph that holds formers and anions (Si and O) that is the size
in all atoms, and with ``t_elements`` the members of those elements are also
counted, giving the size in T atoms (an n-membered SiO2 ring in the usual
sense has 2n atoms and n T atoms). On a :func:`bridged_graph` or a T-T distance
graph the nodes are T atoms, and the size is the size in T atoms; the size in
all atoms is not defined there, because a T-T link does not record which anion
a ring would pass through. The small rings of a bridged graph follow its bond
set, since an anion bonded to three formers closes a T triangle: on the
first frame of the 300 K trajectory of the 3000-atom SiO2 glass, v_bond
0.075 v.u. keeps 3 Si-O contacts at 2.03-2.57 Å, beyond the gap of the
measured Si-O g(r), and the bridged graph then has 25 triclusters and 79
three-membered rings, against 22 and 76 with the bonds within the gap
(v_bond 0.362 v.u. there gives that bond set), whose Si-O atom graph has 54
rings of three Si (outside check, 2026-10-07); on its last frame, 26 and 80
at 0.075 v.u., 20 and 74 at 0.362 v.u., 54 again on the atom graph
(2026-10-07). Each tricluster adds one three-membered ring here.

A ring longer than ``max_size`` is never seen. The King angles, Guttman bond
ends and (primitive) angles through which no ring of at most ``max_size``
members passes are counted (``n_bases_unclosed``), and among them those
shown to lie on no cycle of any size (``n_bases_acyclic``): a neighbour
bonded to nothing else (a non-bridging O on an Si-O graph), or a search that
ran out of vertices, so the piece of the lift on one side is finite. The
rest may lie on a ring larger than ``max_size``, which is not searched; the
notes give both numbers, the number of nodes from which no ring is found,
and, for primitive rings, that rings above ``max_size`` are not counted. An
angle or bond at a dangling neighbour costs no search, and one on a finite
piece a search that stops when the piece runs out of vertices.

**Vertex symbols.** :meth:`RingStatistics.vertex_symbol` gives, per angle of
a node, the size of the smallest ring through it and their number, as RCSR
writes it [17, 18, 19]. A ring there is read as a cycle with no shortcut, a
primitive ring, so the symbol comes from a primitive search (that reading
gives RCSR's published symbols, CHECKED AGAINST below). King's
shortest closed path through an angle need not be one: on the quartz Si net
2 of the 9 shortest 8-membered paths through an 8-ring angle have a shortcut
(vertex symbol 8(7)), and the straight angles of the simple cubic lattice
lie on no ring at all ('*'). :meth:`RingStatistics.shortest_cycle_symbol`
gives the King and Guttman records in the same layout.

COORDINATION SEQUENCES
----------------------
N_k, the number of nodes k bonds away in the lift [9, 10]; N_1 is the degree.
Shell k + 1 is the set of neighbours of shell k less shells k and k - 1, which
holds on any undirected graph and needs no visited set. Known closed forms
check the code: diamond ``floor(5 k^2 / 2) + 2`` (OEIS A008253 [11]), simple
cubic ``4 k^2 + 2`` (OEIS A005899 [12]), and the quartz Si net 4, 12, 30, 52,
80, 116, 156, 204, 258, 318 (OEIS A008261 [13], after Grosse-Kunstleve,
Brunner & Sloane [10]; the same ten terms are RCSR's cs1-cs10 of qtz [18]).

POLYHEDRAL CONNECTIVITY
-----------------------
Two centre polyhedra (cations with their ligands) share a corner when they
have one ligand in common, an edge when two, a face when three or more:
Pauling's third rule [14] is why the distinction matters. Pairs are found
through the ligands (every pair of centres bonded to one ligand), counted with
their periodic images, and reported per centre-pair type as fractions that
pass :func:`facet.core.md_stats.check_fractions`. "Edge" means two shared
ligands, whether or not those two ligands are joined by an edge of either
polyhedron (the two trans vertices of an octahedron count).

CONNECTED COMPONENTS AND PERCOLATION
------------------------------------
Components are those of the quotient graph. Within one, a spanning tree
(grown depth-first) places every node in a cell of the lift; a non-tree edge
then closes a cycle whose net translation is cell(u) + shift - cell(v). Those
translations span the lattice of translations under which the component maps
onto itself, and its rank is the dimensionality of the periodic piece
(0 a finite cluster, 1 a chain, 2 a layer, 3 a framework) [1, 2]. A component
spans box axis a when one of those translations has a nonzero a component:
the piece then reaches every cell along that axis. One component of the box
can stand for k disjoint pieces of the periodic network that a box
translation maps onto each other, two interpenetrating nets as in cuprite
Cu2O: k is the index of the translation lattice in the integer points of the
space it spans (|det| of its Hermite normal form basis in 3-D, the gcd of the
2x2 minors of a 2-D basis, of the components of a 1-D one), reported as
``copies``. The largest fraction and the number of pieces count those
pieces, so a supercell gives the same values: the 6-atom cuprite cell is one
component of two pieces, its 2x2x2 supercell two components of one piece
each, both a largest fraction of 1/2 (tests). The cell of each node also
unwraps a finite cluster into one piece. Modifier clustering is the same
function on a :func:`graph_from_pairs` graph of modifier-modifier pairs within
a cutoff taken from their measured g(r).

WARREN-COWLEY CHEMICAL SHORT-RANGE ORDER
----------------------------------------
alpha_ij = 1 - Z_ij / (Z_i c_j) [15, 16], with Z_ij the mean number of j
neighbours (graph edges) of an i node, Z_i = sum_j Z_ij and c_j the fraction
of j among the nodes. alpha_ij = 0 for a random arrangement, negative when i
prefers j, and sum_j c_j alpha_ij = 0 for every i with Z_i > 0 (the tests
check this). The neighbour shell is whatever the graph's edges are, and a
pair of elements that the graph's definition can never join (two cations in a
cation-anion bond graph, a pair with no cutoff) has alpha = 1 by
construction; a note names those pairs.

WHAT IS NOT DONE HERE
---------------------
No cutoff, former set, bridging anion, ring size limit or shell count has a
default: each is an argument. Frame averages, spreads and the fraction checks
are md_stats's. Nothing here runs or alters a simulation.

CHECKED AGAINST
---------------
``tests/test_md_network.py`` pins the closed forms above (the quartz
sequence to ten shells), RCSR's published vertex symbols of dia
6(2).6(2).6(2).6(2).6(2).6(2), pcu 4 x 12 with * x 3, qtz
6.6.6(2).6(2).8(7).8(7) and gis 4.4.4.8(2).8.8 [18], the R.I.N.G.S.
manual's figure 5.22 case a) and Table 5.0 network [7], and the rings of
the simple cubic lattice against a second enumeration written in the test
(on that lattice the graph distance is the L1 norm). Outside FACET:

* :meth:`RingStatistics.vertex_symbol` on primitive rings to 12 nodes equals
  RCSR's published vertex symbol, compared as a multiset, at every vertex
  orbit of 18 nets (24 orbits: dia, qtz, pcu, lon, crb, sod, nbo, srs, gis,
  coe, wkx, zfa, zmc, uol, bne, wju, bbm, dmh), each net built from the RCSR
  data file (rcsr.net/data/3dall.txt) with gemmi in a throwaway script
  (2026-10-07); King's shortest closed paths (:meth:`shortest_cycle_symbol`)
  differ from it on 13 of the 18. On 50 further nets drawn at random from
  the file (at most 3 vertex orbits and 12 vertices per cell), 48 orbits
  with a published symbol match orbit for orbit and the other 2, both of
  xaq, match each other's: the file lists xaq's three symbols in another
  order than its vertices (its 3-connected vertex V1 has the 3-entry symbol
  4.12(2).12(2), listed second); 23 orbits have no published symbol;
* in a throwaway environment (neither package is a dependency, and no test
  imports them), ASE 3.29.0 and networkx 3.6.1 computed the same quantities
  from the published definitions on explicit supercells of the quartz Si
  net (8x8x8): King 6_1.6_1.6_2.6_2.8_9.8_9 at every Si, Guttman 6_3 at
  every bond, primitive rings through each Si 6 of six members and 40 of
  eight, none of ten or twelve; the same here. The 6 six-membered rings
  agree with RCSR's qtz vertex symbol (1 + 1 + 2 + 2 over the angles); the
  40 eight-membered primitive rings are computed only (here, by that script
  and by a second outside enumeration with a pairwise shortcut test), no
  published value was found. On a 6x6x6 supercell the same script counted 2
  primitive twelve-membered rings per Si: chains along a close on themselves
  after 12 steps through six cells, a cycle that winds around that box,
  which the lift never takes for a ring;
* a 3000-atom Na2O-3SiO2 melt-quenched glass (Si-O edges within the
  1.795 Å first minimum of the Si-O distance histogram, which the outside
  script measured itself), rings of up to 20 atoms, 12 Si and 8 O chosen at
  random: per angle (King) and per bond (Guttman) the size and the number of
  shortest rings, and the primitive rings through each atom by size, all
  identical (0 mismatches over 20 atoms);
* a 3000-atom SiO2 melt-quenched glass (Si-O edges within its measured
  1.80 Å first minimum), rings of up to 20 atoms, 10 Si chosen at random:
  the same three comparisons, identical (0 mismatches over 10 atoms);
* a separate verification on 2026-10-07, on four 3000-atom glasses (SiO2,
  Na2O-3SiO2, a sodium borosilicate and a sodium aluminosilicate) with
  their g(r)-minimum Si-O, B-O and Al-O edges: matscipy 1.3.0 shortest-path
  (Franzblau) rings equal the primitive rings here at every size from 4 to
  24 atoms; vitrum 1.1.0 Guttman rings equal these ring for ring; OVITO
  3.16.0 cluster analysis gives the same components and ASE 3.29.0 the same
  dimensionalities; an ASE neighbour list gives the same Warren-Cowley alpha
  on the Na2O-3SiO2 and aluminosilicate glasses; a brute-force enumeration
  of every simple cycle on 122 random periodic graphs gives the same King,
  Guttman and primitive rings and normalisations. That run used the version
  before the primitive search held its candidates in search order; this
  version gives the same rings, per-node counts and normalisations as that
  one on the SiO2, Na2O-3SiO2 and borosilicate glasses (Si and Si-O
  graphs, and the Na2O-3SiO2 graph with its Na-O bonds, compared ring for
  ring, 2026-10-07).

The glasses were made with LAMMPS for the MD verification, outside FACET;
nothing here has been checked on a model supplied by a user.

TIMINGS
-------
Measured 2026-10-07 on this machine (Windows 11, i5-13420H, Python 3.11.9,
numpy 2.4.6), one fresh process per operation, while other jobs ran: wall
time ran up to 3.4 times the process CPU time, so CPU time is given, with
the wall time in brackets where the two differ by more than a third. The
models: a 3000-atom SiO2 glass (1000 Si), a 3000-atom Na2O-3SiO2 glass (750
Si), both melt-quenched with LAMMPS outside FACET for the MD verification,
and the SiO2 glass repeated 2x2x1 (12 000 atoms; its rings are the 3000-atom
box's four times over, and its cost is a glass's: the 10 648-atom random box
of ``tools/bench_md.make_box`` timed before, with Si CN 0-6 and 1052
triclusters, took 2.4 to 10 times less CPU time per search than this
replica in the verification). Rings to 12 T atoms are max_size 12 on the
:func:`bridged_graph` of Si through O ("Si graph") and max_size 24 on the
Si-O :func:`graph_from_bonds` ("Si-O graph"); CPU seconds::

                          SiO2, 3000     Na2O-3SiO2, 3000   SiO2 x4, 12 000
    King, Si graph        0.39           0.25               2.0 (3.3)
    Guttman, Si graph     0.13           0.06               0.44
    primitive, Si graph   2.6 (5.5)      0.44               12.4 (23.8)
    King, Si-O graph      1.1            0.77               9.5 (17.5)
    Guttman, Si-O graph   0.75           0.36               3.0 (4.2)
    primitive, Si-O graph 15.9 (32.7)    2.9 (3.6)          71.8 (244)

The verification measured the CPU time of each search 4.0-4.9 times larger
for 4 times the nodes, alternating the two sizes in one process. The
Na2O-3SiO2 graph that also holds the Na-O bonds (:func:`graph_from_bonds`
with every element, its default; 3000 nodes, 4858 edges), rings to 24
nodes: King 1.6 s, Guttman 0.81 s, primitive 428 s (1499 s wall), and 250 s
(337 s wall) in a second run on a less loaded machine: the Na-O bonds close
15.1 million candidate cycles for 5098 primitive rings, against 256 104 for
2085 on the SiO2 Si-O graph. Holding candidates only until they are decided
costs CPU time on some graphs: run one after the other in one process, the
version before and this one took 1.9 / 1.9 s (SiO2, Si graph, rings to 12),
3.7 / 3.8 s (SiO2, Si-O graph, to 20), 1.1 / 1.4 s (Na2O-3SiO2, Si-O graph,
to 20) and 2.1 / 2.6 s (Na2O-3SiO2 with its Na-O bonds, to 12).

Memory, as the peak of the Python allocations during the call
(tracemalloc): 12 MB for King on the SiO2 Si-O graph; 6.5 / 11.5 MB for the
primitive search on the SiO2 Si / Si-O graph, 27 / 47 MB on the 12 000-atom
replica (the verification had measured a 230 MB commit for the latter with
the version before). The primitive search on the Na2O-3SiO2 graph with its
Na-O bonds raised the process commit by at most 60 MB (51 MB in the second
run); the version before,
which held every candidate to the end, had raised it 1.8 GB after 450 of its
3000 nodes (about 12 GB at the end, extrapolated). The result reports the
open candidates of a primitive search (``peak_open_candidates``): 2593 and
2820 on the SiO2 glass (Si / Si-O graph), 5481 and 5694 on the replica,
120 569 on the Na2O-3SiO2 graph with its Na-O bonds.

The rest, SiO2 3000 / 12 000, CPU time: a graph builder 3 ms / 31 ms;
coordination sequences to 10 shells on the Si graph 0.59 s / 3.9 s (14 s
wall); polyhedral connectivity 7 ms / 47 ms; components 16 ms / 78 ms;
Warren-Cowley 3 ms / 16 ms. A 20-frame trajectory of the SiO2 glass, with
bulk.analyse_frame, the bridged graph, primitive rings to 12 T atoms, six
coordination shells, components and polyhedral sharing, took 4.9 s CPU per
frame (20 s wall; 3.7 s CPU with the version of 2026-10-06).

The primitive search spends its time on candidates. Under cProfile (version
of 2026-10-06, SiO2 Si-O graph) the canonical forms took 29 % of the time,
the breadth-first searches 20 %, the pair loop 21 % and the path enumeration
18 %; building the canonical form from list slices rather than element by
element changed the CPU time of the Si-graph search from 3.2-3.9 s to
3.1-3.2 s (interleaved runs). A bond with a dangling end, such as an Si-O
bond to a non-bridging O on an Si-O graph, is now decided without a search:
before, a Guttman search started from the Si end of such a bond ran to the
full depth, so the time depended on the row order (Na2O-3SiO2 Si-O graph,
rings to 40 nodes: 1.2 / 10.7 / 19.8 s for the file order, a random order
and the O rows first; now 1.1 / 1.3 / 1.5 s).

REFERENCES
----------
[1] S. J. Chung, Th. Hahn and W. E. Klee, "Nomenclature and generation of
    three-periodic nets: the vector method", *Acta Crystallographica* A40
    (1984) 42-50, https://doi.org/10.1107/S0108767384000088
[2] O. Delgado-Friedrichs and M. O'Keeffe, "Identification of and symmetry
    computation for crystal nets", *Acta Crystallographica* A59 (2003)
    351-360, https://doi.org/10.1107/S0108767303012017
[3] S. V. King, "Ring configurations in a random network model of vitreous
    silica", *Nature* 213 (1967) 1112-1113, https://doi.org/10.1038/2131112a0
[4] L. Guttman, "Ring structure of the crystalline and amorphous forms of
    silicon dioxide", *Journal of Non-Crystalline Solids* 116 (1990) 145-147,
    https://doi.org/10.1016/0022-3093(90)90686-G
[5] K. Goetzke and H.-J. Klein, "Properties and efficient algorithmic
    determination of different classes of rings in finite and infinite
    polyhedral networks", *Journal of Non-Crystalline Solids* 127 (1991)
    215-220, https://doi.org/10.1016/0022-3093(91)90145-V
[6] X. Yuan and A. N. Cormack, "Efficient algorithm for primitive ring
    statistics in topological networks", *Computational Materials Science*
    24 (2002) 343-360, https://doi.org/10.1016/S0927-0256(01)00256-7
[7] S. Le Roux, I.S.A.A.C.S. manual, "Definitions", "Description of a
    network using ring statistics - existing tools" (Table 5.0) and "Rings
    and connectivity: the new R.I.N.G.S. method implemented in I.S.A.A.C.S."
    (Tables 5.4, 5.7, 5.8) (2011),
    https://isaacs.sourceforge.io/manual/page35_ct.html,
    https://isaacs.sourceforge.io/manual/page36_ct.html and
    https://isaacs.sourceforge.io/manual/page37_ct.html
[8] S. Le Roux and P. Jund, "Ring statistics analysis of topological
    networks: New approach and application to amorphous GeS2 and SiO2
    systems", *Computational Materials Science* 49 (2010) 70-83,
    https://doi.org/10.1016/j.commatsci.2010.04.023
[9] J. H. Conway and N. J. A. Sloane, "Low-dimensional lattices. VII.
    Coordination sequences", *Proceedings of the Royal Society of London A*
    453 (1997) 2369-2389, https://doi.org/10.1098/rspa.1997.0126
[10] R. W. Grosse-Kunstleve, G. O. Brunner and N. J. A. Sloane, "Algebraic
    description of coordination sequences and exact topological densities
    for zeolites", *Acta Crystallographica* A52 (1996) 879-889,
    https://doi.org/10.1107/S0108767396007519
[11] OEIS A008253, "Coordination sequence for diamond", https://oeis.org/A008253
[12] OEIS A005899, "... coordination sequence for cubic lattice: a(0) = 1;
    for n > 0, a(n) = 4n^2 + 2", https://oeis.org/A005899
[13] OEIS A008261, "Coordination sequence for quartz", https://oeis.org/A008261
[14] L. Pauling, "The principles determining the structure of complex ionic
    crystals", *Journal of the American Chemical Society* 51 (1929)
    1010-1026, https://doi.org/10.1021/ja01379a006
[15] J. M. Cowley, "An approximate theory of order in alloys", *Physical
    Review* 77 (1950) 669-675, https://doi.org/10.1103/PhysRev.77.669
[16] D. de Fontaine, "The number of independent pair-correlation functions in
    multicomponent systems", *Journal of Applied Crystallography* 4 (1971)
    15-19, https://doi.org/10.1107/S0021889871006174
[17] V. A. Blatov, M. O'Keeffe and D. M. Proserpio, "Vertex-, face-, point-,
    Schlafli-, and Delaney-symbols in nets, polyhedra and tilings:
    recommended terminology", *CrystEngComm* 12 (2010) 44-48,
    https://doi.org/10.1039/B910671E
[18] M. O'Keeffe, M. A. Peskov, S. J. Ramsden and O. M. Yaghi, "The
    Reticular Chemistry Structure Resource (RCSR) database of, and symbols
    for, crystal nets", *Accounts of Chemical Research* 41 (2008) 1782-1789,
    https://doi.org/10.1021/ar800124u; the net data (vertex symbols, cs1-cs10)
    from http://rcsr.net/data/3dall.txt, read 2026-10-07
[19] M. O'Keeffe and S. T. Hyde, "Vertex symbols for zeolite nets",
    *Zeolites* 19 (1997) 370-374,
    https://doi.org/10.1016/S0144-2449(97)00133-4
"""
from __future__ import annotations

import functools
import itertools
import math
import re
import struct
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import bulk, bv
from .bulk import _blocks_of, _Coverage, _frame_digest, _pair_geometry
from .md_model import Frame, validate_symbol
from .md_stats import Distribution, Scalar, Series, check_fractions

__all__ = [
    "CRITERIA", "SHARING_KINDS", "GRAPH_SOURCES", "VECTOR_TOL_ANG",
    "PROGRESS_INTERVAL_S", "Cancelled", "NetworkGraph", "graph_from_bonds",
    "graph_from_pairs", "bridged_graph", "Ring", "RingStatistics", "ring_statistics",
    "CoordinationSequences", "coordination_sequences",
    "PolyhedralConnectivity", "polyhedral_connectivity",
    "Components", "components", "WarrenCowley", "warren_cowley",
    "average_rings", "average_coordination_sequences",
    "average_polyhedral_connectivity", "average_components",
    "average_warren_cowley",
]

CRITERIA = ("king", "guttman", "primitive")
SHARING_KINDS = ("corner", "edge", "face")
GRAPH_SOURCES = ("bond valence", "distance", "bridging anion")

# How far a bond vector handed in may sit from the vector this frame's
# fractions and box give. A numerical tolerance, not a physical one: both are
# formed by the same expression (bulk._pair_geometry) and agree to the bit on
# the frame they came from; a table from another frame differs by far more.
VECTOR_TOL_ANG: float = 1e-9

# The longest a search goes without calling progress(), besides the call at
# every 1 % of its units. A choice for the user interface, not a physical
# number: 1 % of the 3000 units of a primitive search to 24 nodes on a
# Na2O-3SiO2 graph that holds the Na-O bonds took 5-6 s on an idle machine
# and 49 s under full load (measured, 2026-10-07).
PROGRESS_INTERVAL_S: float = 1.0
_clock = time.monotonic


class Cancelled(RuntimeError):
    """Raised when the ``cancelled`` callback of a search returns True."""


# ---------------------------------------------------------------------------
# argument checks
# ---------------------------------------------------------------------------

def _check_frame(frame) -> Frame:
    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}; "
                         "md_model.frame_from_arrays builds one")
    return frame


def _symbols(values, what: str) -> frozenset[str]:
    """A set of validated element symbols; no default, no bare string."""
    if values is None:
        raise ValueError(f"{what} has no default: give the element symbols "
                         "explicitly")
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{what} needs a collection of element symbols such "
                         f"as {{'Si'}}, not the text {values!r}")
    try:
        items = list(values)
    except TypeError:
        raise ValueError(f"{what} needs a collection of element symbols") \
            from None
    out = frozenset(validate_symbol(item) for item in items)
    if not out:
        raise ValueError(f"{what} is empty")
    return out


def _whole(value, what: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or \
            not isinstance(value, (int, np.integer)):
        raise ValueError(f"{what} is {value!r}; a whole number is needed")
    if value < minimum:
        raise ValueError(f"{what} is {value}; at least {minimum} is needed")
    return int(value)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _absent_note(wanted: frozenset[str], frame_elements, what: str) -> list[str]:
    present = {str(e) for e in np.unique(frame_elements)}
    missing = sorted(wanted - present)
    if not missing:
        return []
    return [f"{what}: {', '.join(missing)} not in this frame"]


def _frozen(array) -> np.ndarray:
    out = np.ascontiguousarray(array)
    out.setflags(write=False)
    return out


class _Poller:
    """Polls ``cancelled`` at every unit of work and calls ``progress`` at
    every 1 % of the units and whenever PROGRESS_INTERVAL_S has passed since
    the last call."""

    def __init__(self, cancelled, progress, total: int):
        self.cancelled = cancelled
        self.progress = progress
        self.total = total
        self.step = max(1, total // 100)
        self.last = _clock()

    def __call__(self, done: int) -> None:
        if self.cancelled is not None and self.cancelled():
            raise Cancelled(f"cancelled after {done} of {self.total}")
        if self.progress is None:
            return
        now = _clock()
        if done % self.step == 0 or done == self.total or \
                now - self.last >= PROGRESS_INTERVAL_S:
            self.last = now
            self.progress(done, self.total)


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class NetworkGraph:
    """An undirected periodic graph on the atoms of one frame.

    Edge e joins atom row ``u[e]`` in cell (0, 0, 0) to atom row ``v[e]`` in
    cell ``shift[e]``: in fractions, ``frac[v] + shift - frac[u]`` is the bond.
    Each edge is stored once, with ``u <= v`` (and, for an atom bonded to its
    own image, the shift whose first nonzero component is positive), sorted.
    ``multiplicity`` is the number of bridging anions behind a link of a
    :func:`bridged_graph` and 1 otherwise. ``joinable`` lists the element pairs
    (alphabetical) that the definition can join at all.

    Build one with :func:`graph_from_bonds`, :func:`graph_from_pairs` or
    :func:`bridged_graph`.
    """

    elements: np.ndarray           # (N,) symbols of every atom of the frame
    in_graph: np.ndarray           # (N,) bool: the atom is a node
    u: np.ndarray                  # (E,) int64
    v: np.ndarray                  # (E,) int64
    shift: np.ndarray              # (E, 3) int64
    d_ang: np.ndarray              # (E,)
    multiplicity: np.ndarray       # (E,) int64
    source: str                    # one of GRAPH_SOURCES
    definition: str
    node_elements: frozenset
    joinable: frozenset
    v_bond_vu: float | None
    cutoffs_ang: Mapping
    cutoff_sources: Mapping
    frame_digest: str
    notes: tuple[str, ...] = ()

    @property
    def n_atoms(self) -> int:
        return int(self.elements.shape[0])

    @property
    def nodes(self) -> np.ndarray:
        """Atom rows of the nodes, ascending."""
        return np.flatnonzero(self.in_graph)

    @property
    def n_nodes(self) -> int:
        return int(self.in_graph.sum())

    @property
    def n_edges(self) -> int:
        return int(self.u.shape[0])

    @property
    def degree(self) -> np.ndarray:
        """(N,) bonds per atom in the lift (an edge to the atom's own image
        counts twice); 0 for atoms outside the graph."""
        n = self.n_atoms
        return (np.bincount(self.u, minlength=n)
                + np.bincount(self.v, minlength=n)).astype(np.int64)

    def __len__(self) -> int:
        return self.n_edges

    def __repr__(self) -> str:
        return (f"<NetworkGraph {self.source}: {self.n_nodes} nodes, "
                f"{self.n_edges} edges>")


def _orient(u, v, shift):
    """Each undirected edge as (lower row, higher row, shift)."""
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64).reshape(-1, 3)
    same = u == v
    if same.any():
        s = shift[same]
        if not s.any(axis=1).all():
            raise ValueError("an edge joins an atom to itself in the same "
                             "cell; there is no such bond")
    swap = u > v
    if same.any():
        rows = np.flatnonzero(same)
        s = shift[rows]
        first = np.argmax(s != 0, axis=1)
        negative = s[np.arange(rows.size), first] < 0
        swap[rows[negative]] = True
    out_u = np.where(swap, v, u)
    out_v = np.where(swap, u, v)
    out_shift = np.where(swap[:, None], -shift, shift)
    return out_u, out_v, out_shift


def _unique_edges(u, v, shift):
    """Sorted unique oriented edges, the index of each one's first copy, and
    the number of copies."""
    u, v, shift = _orient(u, v, shift)
    if u.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros((0, 3), dtype=np.int64), empty, empty
    rows = np.column_stack([u, v, shift])
    unique, first, counts = np.unique(rows, axis=0, return_index=True,
                                      return_counts=True)
    return (unique[:, 0].copy(), unique[:, 1].copy(), unique[:, 2:].copy(),
            first, counts)


def _make_graph(frame: Frame, nodes_mask, u, v, shift, multiplicity, *,
                source, definition, node_elements, joinable, v_bond_vu=None,
                cutoffs=None, cutoff_sources=None, notes=()) -> NetworkGraph:
    _, d = _pair_geometry(frame.frac, frame.box_ang, u, v,
                          shift.astype(np.int64))
    if u.size and not (d > 0).all():
        raise ValueError("an edge has length 0; two atoms of the frame sit "
                         "on one position")
    return NetworkGraph(
        elements=frame.elements, in_graph=_frozen(nodes_mask),
        u=_frozen(u.astype(np.int64)), v=_frozen(v.astype(np.int64)),
        shift=_frozen(shift.astype(np.int64)), d_ang=_frozen(d),
        multiplicity=_frozen(np.asarray(multiplicity, dtype=np.int64)),
        source=source, definition=definition,
        node_elements=frozenset(node_elements), joinable=frozenset(joinable),
        v_bond_vu=v_bond_vu, cutoffs_ang=dict(cutoffs or {}),
        cutoff_sources=dict(cutoff_sources or {}),
        frame_digest=_frame_digest(frame), notes=tuple(notes))


def _bonds_of(frame: Frame, table, v_bond_vu):
    """The bonds of ``table`` at ``v_bond_vu``, checked to be this frame's."""
    if not isinstance(table, bulk.ValenceTable):
        raise ValueError(f"a bulk.ValenceTable is needed, not "
                         f"{type(table).__name__}")
    if table.n_atoms != frame.n_atoms or \
            not np.array_equal(table.elements, frame.elements):
        raise ValueError("the valence table was built on another frame (its "
                         "atoms or elements differ from this frame's)")
    bonds = bulk.bonds_at(table, v_bond_vu)
    if len(bonds):
        vec, _ = _pair_geometry(frame.frac, frame.box_ang,
                                bonds.cation.astype(np.int64),
                                bonds.anion.astype(np.int64),
                                bonds.image.astype(np.int64))
        worst = float(np.max(np.abs(vec - bonds.vec_ang)))
        if not worst <= VECTOR_TOL_ANG:
            raise ValueError(
                f"the valence table was built on another frame: its bond "
                f"vectors differ by up to {worst:.3g} Å from this frame's "
                "positions")
    # what the table left out (contacts with no parameter, pairs below
    # d_min) is left out of the bonds, so its notes are the graph's too
    notes = [f"valence table: {note}" for note in table.notes]
    if bonds.v_bond_vu < table.v_list_vu:
        notes.append(
            f"v_bond {bonds.v_bond_vu:g} v.u. lies below the table's v_list "
            f"{table.v_list_vu:g} v.u.; contacts below v_list are not in the "
            "table, so the bonds are those above v_list")
    return bonds, notes


def graph_from_bonds(frame: Frame, table, v_bond_vu: float = bv.V_BOND_DEFAULT,
                     *, elements: Iterable[str] | None = None) -> NetworkGraph:
    """The bond graph: the cation-anion bonds the CN counts at ``v_bond_vu``.

    ``table`` is the frame's :class:`facet.core.bulk.ValenceTable`; the bonds
    are :func:`facet.core.bulk.bonds_at` of it, checked against this frame's
    positions. ``elements`` selects the nodes (None: every element of the
    frame); a bond with an end outside the nodes is not an edge, and the
    number left out per element pair is noted. The table's own notes
    (contacts with no bond-valence parameter, which are never bonds; pairs
    below its d_min) are the graph's notes too, prefixed 'valence table:'.
    """
    frame = _check_frame(frame)
    bonds, notes = _bonds_of(frame, table, v_bond_vu)
    symbols = frame.elements
    if elements is None:
        node_set = frozenset(str(s) for s in np.unique(symbols))
    else:
        node_set = _symbols(elements, "elements")
        notes += _absent_note(node_set, symbols, "elements")
    mask = np.isin(symbols, sorted(node_set))
    c = bonds.cation.astype(np.int64)
    a = bonds.anion.astype(np.int64)
    keep = mask[c] & mask[a]
    if (~keep).any():
        pairs, counts = np.unique(
            [f"{symbols[i]}-{symbols[j]}" for i, j in zip(c[~keep], a[~keep])],
            return_counts=True)
        notes.append(
            f"{int((~keep).sum())} bond(s) end on atoms outside the nodes and "
            "are not edges: " + ", ".join(f"{p} {n}" for p, n in
                                         zip(pairs, counts)))
    u, v, shift, _, copies = _unique_edges(c[keep], a[keep],
                                           bonds.image[keep])
    if (copies != 1).any():
        raise ValueError("bulk.bonds_at listed a bond twice")
    is_anion = np.asarray(table.is_anion, dtype=bool)
    cations = {str(s) for s in np.unique(symbols[~is_anion])} & node_set
    anions = {str(s) for s in np.unique(symbols[is_anion])} & node_set
    joinable = {_pair_key(x, y) for x in cations for y in anions}
    definition = (f"bond valence: cation-anion bonds with v > "
                  f"{bonds.v_bond_vu:g} v.u. (bulk.bonds_at, parameters "
                  f"{table.params.params_name}); nodes: "
                  f"{', '.join(sorted(node_set))}")
    return _make_graph(frame, mask, u, v, shift, np.ones(u.size, np.int64),
                       source="bond valence", definition=definition,
                       node_elements=node_set, joinable=joinable,
                       v_bond_vu=bonds.v_bond_vu, notes=notes)


def _checked_cutoffs(cutoffs_ang, cutoff_sources):
    if not isinstance(cutoffs_ang, Mapping) or not cutoffs_ang:
        raise ValueError("cutoffs_ang needs at least one (element, element) "
                         "-> cutoff in Å; none is assumed")
    if not isinstance(cutoff_sources, Mapping):
        raise ValueError("cutoff_sources needs one source text per cutoff "
                         "(for example the first minimum of the measured "
                         "partial g(r))")
    sources = {}
    for key, text in cutoff_sources.items():
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(f"cutoff source key {key!r}: an (element, "
                             "element) pair is needed")
        pair = _pair_key(validate_symbol(key[0]), validate_symbol(key[1]))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"the source of the {pair[0]}-{pair[1]} cutoff "
                             "is empty")
        sources[pair] = text.strip()
    cutoffs = {}
    for key, value in cutoffs_ang.items():
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(f"cutoff key {key!r}: an (element, element) pair "
                             "is needed")
        pair = _pair_key(validate_symbol(key[0]), validate_symbol(key[1]))
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"the {pair[0]}-{pair[1]} cutoff {value!r} is not "
                             "a number")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"the {pair[0]}-{pair[1]} cutoff {value!r} is not "
                             "a number") from None
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"the {pair[0]}-{pair[1]} cutoff is {value!r}; a "
                             "finite distance above 0 is needed")
        if pair in cutoffs and cutoffs[pair] != number:
            raise ValueError(f"two cutoffs are given for {pair[0]}-{pair[1]} "
                             f"({cutoffs[pair]} and {number} Å)")
        cutoffs[pair] = number
        if pair not in sources:
            raise ValueError(
                f"the {pair[0]}-{pair[1]} cutoff ({number} Å) has no source; "
                "each cutoff needs one (TODO: need reference -- a cutoff is "
                "measured from the model, for example the first minimum of "
                "its partial g(r), not typed in)")
    unused = sorted(set(sources) - set(cutoffs))
    if unused:
        raise ValueError("cutoff_sources names pairs with no cutoff: "
                         + ", ".join(f"{a}-{b}" for a, b in unused))
    return cutoffs, sources


def graph_from_pairs(frame: Frame, pairs, cutoffs_ang: Mapping,
                     cutoff_sources: Mapping, *,
                     elements: Iterable[str] | None = None) -> NetworkGraph:
    """The distance graph: pairs of one search kept where d <= their cutoff.

    ``pairs`` is a :class:`facet.core.bulk.PairTable` or the blocks
    :func:`facet.core.bulk.iter_pairs` yields for this frame (all of them,
    each once). ``cutoffs_ang`` maps an (element, element) pair, in either
    order, to a cutoff in Å; a pair counts when d <= cutoff, inclusive as
    :func:`facet.core.bulk.distance_cn`. ``cutoff_sources`` gives the
    provenance of each cutoff (how it was measured); a cutoff without one is
    refused. ``elements`` selects the nodes (None: the elements the cutoffs
    name). ValueError for a cutoff beyond the radius the pairs were searched
    to. Pairs the search left out at or below its ``d_min_ang`` are not
    edges, and their number is noted.
    """
    frame = _check_frame(frame)
    cutoffs, sources = _checked_cutoffs(cutoffs_ang, cutoff_sources)
    symbols = frame.elements
    named = frozenset(x for pair in cutoffs for x in pair)
    node_set = named if elements is None else _symbols(elements, "elements")
    notes = _absent_note(node_set, symbols, "elements")
    lonely = sorted(node_set - named)
    if lonely:
        notes.append(f"{', '.join(lonely)} atoms are nodes, but no cutoff "
                     "names them, so they have no edge")
    mask = np.isin(symbols, sorted(node_set))
    codes = {s: k for k, s in enumerate(sorted(set(str(x) for x in
                                                   np.unique(symbols))))}
    atom_code = np.array([codes[str(s)] for s in symbols], dtype=np.int64)
    n_codes = len(codes)
    limit = np.full((n_codes, n_codes), -np.inf)
    for (a, b), value in cutoffs.items():
        if a in codes and b in codes:
            limit[codes[a], codes[b]] = value
            limit[codes[b], codes[a]] = value
    coverage = _Coverage(frame, frame.n_atoms)
    kept_i, kept_j, kept_image = [], [], []
    below, d_min = 0, None
    for block in _blocks_of(pairs):
        coverage.add(block)
        below += int(block.n_below_d_min)
        d_min = block.d_min_ang
        for (a, b), value in cutoffs.items():
            if value > block.r_ang:
                raise ValueError(
                    f"the {a}-{b} cutoff is {value} Å, beyond the "
                    f"{block.r_ang} Å the pairs were searched to")
        i = block.i.astype(np.int64)
        j = block.j.astype(np.int64)
        take = (block.d_ang <= limit[atom_code[i], atom_code[j]]) \
            & mask[i] & mask[j]
        kept_i.append(i[take])
        kept_j.append(j[take])
        kept_image.append(block.image[take].astype(np.int64))
    coverage.check()
    if below:
        notes.append(f"{below} ordered pair(s) at or below {d_min:g} Å were "
                     "left out by the pair search (bulk.iter_pairs, its "
                     "d_min_ang), so they are not edges")
    u, v, shift, _, copies = _unique_edges(
        np.concatenate(kept_i), np.concatenate(kept_j),
        np.concatenate(kept_image))
    if (copies != 2).any():
        raise ValueError("a pair was found from one end only; the pair "
                         "blocks are not one complete search of this frame")
    used = {k: v for k, v in cutoffs.items() if k[0] in node_set
            and k[1] in node_set}
    definition = ("distance: pairs with d <= cutoff ("
                  + "; ".join(f"{a}-{b} {value:g} Å, {sources[(a, b)]}"
                              for (a, b), value in sorted(cutoffs.items()))
                  + f"); nodes: {', '.join(sorted(node_set))}")
    return _make_graph(frame, mask, u, v, shift, np.ones(u.size, np.int64),
                       source="distance", definition=definition,
                       node_elements=node_set, joinable=set(used),
                       cutoffs=cutoffs, cutoff_sources=sources, notes=notes)


def bridged_graph(frame: Frame, table, v_bond_vu: float = bv.V_BOND_DEFAULT,
                  *, formers: Iterable[str], anions: Iterable[str]
                  ) -> NetworkGraph:
    """The former network: two formers linked when they share an anion.

    The bonds are :func:`facet.core.bulk.bonds_at` of ``table``; an anion of
    ``anions`` bonded to k formers of ``formers`` links each of the k(k-1)/2
    pairs (with their periodic images). Two formers that share two or more
    anions (edge sharing) get one link whose ``multiplicity`` counts the
    anions; an anion bonded to three or more formers links all of them. Both
    are counted in the notes, as are the former bonds to anions of other
    elements (they link nothing here) and any former or anion element whose
    oxidation state puts it on the other side of every bond. Nodes: every
    atom of a former element, linked or not. No default for either set.
    """
    frame = _check_frame(frame)
    former_set = _symbols(formers, "formers")
    anion_set = _symbols(anions, "anions")
    both = sorted(former_set & anion_set)
    if both:
        raise ValueError(f"{', '.join(both)} given both as former and as "
                         "bridging anion")
    bonds, notes = _bonds_of(frame, table, v_bond_vu)
    symbols = frame.elements
    notes += _absent_note(former_set, symbols, "formers")
    notes += _absent_note(anion_set, symbols, "anions")
    mask = np.isin(symbols, sorted(former_set))
    is_anion = np.asarray(table.is_anion, dtype=bool)
    for what, wanted, side, role in (
            ("formers", former_set, is_anion, "anions (negative oxidation "
             "state), so they are the anion end of each of their bonds"),
            ("anions", anion_set, ~is_anion, "cations (oxidation state of 0 "
             "or more), so they are the cation end of each of their bonds")):
        flipped = sorted({str(s) for s in symbols[side]} & wanted)
        if flipped:
            notes.append(f"{what} {', '.join(flipped)} are {role} in the "
                         "valence table and link no formers here")
    c = bonds.cation.astype(np.int64)
    a = bonds.anion.astype(np.int64)
    listed = np.isin(symbols[a], sorted(anion_set))
    other = mask[c] & ~listed
    if other.any():
        pairs, counts = np.unique(
            [f"{symbols[i]}-{symbols[j]}" for i, j in zip(c[other], a[other])],
            return_counts=True)
        notes.append(
            f"{int(other.sum())} bond(s) from formers to anions outside "
            "'anions' link no formers: "
            + ", ".join(f"{p} {k}" for p, k in zip(pairs, counts)))
    take = mask[c] & listed
    c, a = c[take], a[take]
    cell = -bonds.image[take].astype(np.int64)      # former's cell, anion at 0
    order = np.lexsort((c, a))
    c, a, cell = c[order], a[order], cell[order]
    pu, pv, ps = [], [], []
    tricluster = 0
    if a.size:
        starts = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
        sizes = np.diff(np.r_[starts, a.size])
        tricluster = int((sizes >= 3).sum())
        for k in np.unique(sizes[sizes >= 2]):
            members = starts[sizes == k][:, None] + np.arange(k)[None, :]
            for p, q in itertools.combinations(range(int(k)), 2):
                pu.append(c[members[:, p]])
                pv.append(c[members[:, q]])
                ps.append(cell[members[:, q]] - cell[members[:, p]])
    if pu:
        u, v, shift, _, counts = _unique_edges(
            np.concatenate(pu), np.concatenate(pv), np.concatenate(ps))
    else:
        u = v = counts = np.zeros(0, dtype=np.int64)
        shift = np.zeros((0, 3), dtype=np.int64)
    multiple = int((counts >= 2).sum())
    if multiple:
        notes.append(
            f"{multiple} former pair(s) share two or more anions (edge or face "
            "sharing); each is one link here, its multiplicity counting the "
            "anions, so a two-membered ring is not a ring of this graph "
            "(polyhedral_connectivity reports the sharing)")
    if tricluster:
        notes.append(
            f"{tricluster} anion(s) are bonded to three or more formers "
            "(triclusters); each links every pair of its formers, so each "
            "closes a three-membered ring of this graph that the atom graph "
            "(graph_from_bonds) does not hold")
    definition = (f"bridging anion: formers {', '.join(sorted(former_set))} "
                  f"linked through a shared {', '.join(sorted(anion_set))} "
                  f"(bonds with v > {bonds.v_bond_vu:g} v.u., bulk.bonds_at, "
                  f"parameters {table.params.params_name}); nodes: "
                  f"{', '.join(sorted(former_set))}")
    joinable = {_pair_key(x, y) for x in former_set for y in former_set}
    return _make_graph(frame, mask, u, v, shift, counts, source="bridging anion",
                       definition=definition, node_elements=former_set,
                       joinable=joinable, v_bond_vu=bonds.v_bond_vu,
                       notes=notes)


def _check_graph(graph) -> NetworkGraph:
    if not isinstance(graph, NetworkGraph):
        raise ValueError(f"a NetworkGraph is needed, not "
                         f"{type(graph).__name__}; graph_from_bonds, "
                         "graph_from_pairs or bridged_graph builds one")
    if graph.n_nodes == 0:
        raise ValueError("the graph has no node; there is nothing to measure")
    return graph


# ---------------------------------------------------------------------------
# the lift: (atom, cell) as one integer
# ---------------------------------------------------------------------------

class _Lift:
    """Vertices of the lift as integers ``atom + n * code(cell)``.

    ``code`` is a mixed-radix number of the cell with base ``width``, linear in
    the cell, so stepping along an edge adds a constant (``deltas[atom]``) and
    the difference of two keys encodes the atom and the cell difference. The
    base exceeds four times the largest cell component a search of ``depth``
    steps can reach, so a cell difference decodes uniquely.
    """

    def __init__(self, graph: NetworkGraph, depth: int):
        n = graph.n_atoms
        largest = int(np.abs(graph.shift).max()) if graph.n_edges else 0
        reach = depth * max(largest, 1) + 1
        width = 4 * reach + 3
        self.n = n
        self.reach = reach
        self.width = width
        self.origin = (reach * width + reach) * width + reach
        weights = np.array([width * width, width, 1], dtype=np.int64)
        code = graph.shift @ weights
        u, v = graph.u, graph.v
        atoms = np.concatenate([u, v])
        steps = np.concatenate([v - u + n * code, u - v - n * code])
        order = np.argsort(atoms, kind="stable")
        atoms, steps = atoms[order], steps[order]
        bounds = np.searchsorted(atoms, np.arange(n + 1))
        flat = steps.tolist()
        self.deltas = [tuple(flat[bounds[k]:bounds[k + 1]]) for k in range(n)]

    def key(self, atom: int) -> int:
        """The atom in cell (0, 0, 0)."""
        return int(atom) + self.n * self.origin

    def decode_cell(self, code: int) -> tuple[int, int, int]:
        """A cell difference from its code difference."""
        width = self.width
        half = width // 2
        z = (code + half) % width - half
        code = (code - z) // width
        y = (code + half) % width - half
        x = (code - y) // width
        return (int(x), int(y), int(z))


def _bfs(deltas, n: int, start: int, max_depth: int, *, banned: int | None = None,
         banned_edge: tuple[int, int] | None = None, targets=None):
    """Shortest-path layers of the lift from ``start`` up to ``max_depth``.

    ``banned``: a vertex never entered. ``banned_edge``: the one edge (two
    keys) never crossed. ``targets``: stop after the first layer at which
    every target has been reached. Returns (dist, preds, exhausted): the
    depth of each vertex reached, the list of its predecessors one layer
    closer, and whether the search ran out of vertices (its frontier
    emptied) before ``max_depth``: then every vertex it did not reach is
    unreachable at any depth.
    """
    dist = {start: 0}
    preds: dict[int, list[int]] = {start: []}
    frontier = [start]
    remaining = None if targets is None else set(targets)
    edge_a, edge_b = banned_edge if banned_edge is not None else (None, None)
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        layer = []
        for x in frontier:
            skip = edge_b if x == edge_a else (edge_a if x == edge_b else None)
            for step in deltas[x % n]:
                y = x + step
                if y == banned or y == skip:
                    continue
                seen = dist.get(y)
                if seen is None:
                    dist[y] = depth
                    preds[y] = [x]
                    layer.append(y)
                elif seen == depth:
                    preds[y].append(x)
        frontier = layer
        if remaining is not None:
            remaining.difference_update(layer)
            if not remaining:
                break
    return dist, preds, not frontier


def _finite_within(deltas, n: int, start: int, max_depth: int, *,
                   banned: int | None = None,
                   banned_edge: tuple[int, int] | None = None) -> bool:
    """True when a breadth-first search from ``start`` (same bans as
    :func:`_bfs`) runs out of vertices within ``max_depth`` steps: the piece
    of the lift it explores is then finite, and it holds every vertex
    reachable from ``start`` at any depth."""
    seen = {start}
    frontier = [start]
    edge_a, edge_b = banned_edge if banned_edge is not None else (None, None)
    depth = 0
    while frontier:
        if depth == max_depth:
            return False
        depth += 1
        layer = []
        for x in frontier:
            skip = edge_b if x == edge_a else (edge_a if x == edge_b else None)
            for step in deltas[x % n]:
                y = x + step
                if y == banned or y == skip or y in seen:
                    continue
                seen.add(y)
                layer.append(y)
        frontier = layer
    return True


def _angle_acyclic(deltas, n: int, start: int, first: int, other: int,
                   depth: int) -> bool:
    """True when the angle first-start-other is shown to lie on no cycle of
    any size: a neighbour bonded to nothing but ``start``, or a finite piece
    of the lift on one side once ``start`` is removed. False when a path
    first -> other avoiding ``start`` exists within ``depth`` steps, or when
    neither side ran out of vertices within ``depth`` (not decided)."""
    if len(deltas[first % n]) == 1 or len(deltas[other % n]) == 1:
        return True
    dist, _, exhausted = _bfs(deltas, n, first, depth, banned=start,
                              targets=(other,))
    if other in dist:
        return False
    return exhausted or _finite_within(deltas, n, other, depth, banned=start)


def _paths(preds, start: int, end: int, memo: dict) -> list[tuple[int, ...]]:
    """Every shortest path start -> end on the BFS dag, as key tuples."""
    got = memo.get(end)
    if got is not None:
        return got
    if end == start:
        out = [(start,)]
    else:
        out = [path + (end,) for p in preds[end]
               for path in _paths(preds, start, p, memo)]
    memo[end] = out
    return out


def _canonical(keys: Sequence[int], n: int):
    """(canonical tuple, where, sign) for one ring.

    The ring is started at a member with the lowest atom row, its keys taken
    relative to that member's cell (key - n * code(start)), in the direction
    that gives the smaller tuple. Translation, rotation and reversal of the
    ring give the same tuple. ``keys[q]`` sits at position
    ``(where + sign * q) % len(keys)`` of the canonical tuple.
    """
    size = len(keys)
    atoms = [k % n for k in keys]
    low = min(atoms)
    best = None
    where, sign = 0, 1
    p = -1
    for _ in range(atoms.count(low)):
        p = atoms.index(low, p + 1)
        base = keys[p] - low
        rel = [k - base for k in keys]
        forward = tuple(rel[p:] + rel[:p])
        backward = tuple(rel[p::-1] + rel[:p:-1])
        if best is None or forward < best:
            best, where, sign = forward, (-p) % size, 1
        if backward < best:
            best, where, sign = backward, p % size, -1
    return best, where, sign


# ---------------------------------------------------------------------------
# rings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ring:
    """One ring: its members in order, each with its lattice cell.

    ``cells[0]`` is (0, 0, 0); member k sits at ``frac[atoms[k]] + cells[k]``
    in fractions, so the members form one closed loop. ``frame_digest``
    names the frame the ring was found on.
    """

    atoms: tuple[int, ...]
    cells: tuple[tuple[int, int, int], ...]
    elements: tuple[str, ...]
    frame_digest: str = field(default="", repr=False, compare=False)

    @property
    def size(self) -> int:
        return len(self.atoms)

    def count(self, elements: Iterable[str]) -> int:
        """Members whose element is one of ``elements``."""
        wanted = _symbols(elements, "elements")
        return sum(1 for e in self.elements if e in wanted)

    def positions_ang(self, frame: Frame) -> np.ndarray:
        """(size, 3) Cartesian positions of the members as one closed loop.
        ValueError for a frame other than the one the ring was found on."""
        frame = _check_frame(frame)
        if self.frame_digest and _frame_digest(frame) != self.frame_digest:
            raise ValueError("this ring was found on another frame (its "
                             "fractional positions or box differ)")
        frac = frame.frac[list(self.atoms)] + np.array(self.cells, dtype=float)
        return frame.origin_ang + frac @ frame.box_ang


@dataclass(frozen=True, eq=False)
class RingStatistics:
    """The rings of one frame under one criterion, normalised as R.I.N.G.S.

    ``sizes`` runs from 3 to ``max_size`` (nodes of the graph). Per size:
    ``ring_counts`` distinct rings, ``rc`` R_C(n), ``rn`` R_N(n), ``pn``
    P_N(n), ``pn_max`` / ``pn_min`` P_Nmax(n) / P_Nmin(n), ``p_max`` /
    ``p_min`` the normalised P_max(n) / P_min(n) (NaN where P_N(n) = 0).
    ``found`` (n_nodes, n_sizes) holds the rings of each size found from each
    node (rows in the order of ``nodes``); it is built on access from
    ``found_compact``, which stores the columns up to the largest ring size
    found (every larger size is 0), so a ``max_size`` far above the rings
    costs no memory per node. ``base_rings`` holds, per node, one
    (size, number) record per angle (King, primitive) or per bond (Guttman),
    (0, 0) where no ring of at most ``max_size`` passes through it: for King
    and Guttman the shortest closed paths through the angle or bond, for
    primitive rings the smallest primitive rings through the angle (the
    entries of the vertex symbol). ``n_bases`` counts those angles or bond
    ends, ``n_bases_unclosed`` the ones with no ring of at most ``max_size``,
    and ``n_bases_acyclic`` the part of those shown to lie on no cycle of
    any size (a neighbour bonded to nothing else, or a finite piece of the
    graph); the rest may lie on a larger ring, which is not searched.
    ``t_sizes`` is each ring's member count over ``t_elements`` (None when no
    T set was given). For primitive rings, ``n_candidates`` counts the
    candidate cycles the search formed and ``peak_open_candidates`` the most
    it held at once while deciding them (0 for King and Guttman).
    """

    criterion: str
    max_size: int
    graph_definition: str
    graph_source: str
    node_elements: frozenset
    nodes: np.ndarray
    sizes: np.ndarray
    rings: tuple[Ring, ...]
    ring_counts: np.ndarray
    found_compact: np.ndarray
    rc: np.ndarray
    rn: np.ndarray
    pn: np.ndarray
    pn_max: np.ndarray
    pn_min: np.ndarray
    p_max: np.ndarray
    p_min: np.ndarray
    base_rings: tuple
    n_bases: int
    n_bases_unclosed: int
    n_bases_acyclic: int
    n_nodes_without_ring: int
    t_elements: frozenset | None
    t_sizes: np.ndarray | None
    n_candidates: int = 0
    peak_open_candidates: int = 0
    frame_digest: str = ""
    notes: tuple[str, ...] = ()

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_rings(self) -> int:
        return len(self.rings)

    @functools.cached_property
    def found(self) -> np.ndarray:
        """(n_nodes, n_sizes) rings of each size found from each node, built
        on first access."""
        out = np.zeros((self.n_nodes, self.sizes.size), dtype=np.int64)
        out[:, :self.found_compact.shape[1]] = self.found_compact
        out.setflags(write=False)
        return out

    def rings_through(self, atom: int) -> int:
        """Distinct rings found from one node (its row in the frame)."""
        hit = np.flatnonzero(self.nodes == int(atom))
        if hit.size == 0:
            raise ValueError(f"atom row {atom} is not a node of the graph")
        return int(self.found_compact[hit[0]].sum())

    def _base_records(self, atom: int) -> list[tuple[int, int]]:
        hit = np.flatnonzero(self.nodes == int(atom))
        if hit.size == 0:
            raise ValueError(f"atom row {atom} is not a node of the graph")
        return sorted(self.base_rings[hit[0]],
                      key=lambda sc: (sc[0] == 0, sc[0], sc[1]))

    def shortest_cycle_symbol(self, atom: int) -> str:
        """King: per angle, the length of the shortest closed path through
        it and the number of such paths (ties counted); Guttman: the same
        per bond. Written 'size_count', joined by '.', sorted; '*' where no
        closed path of at most max_size nodes passes through.

        This is not a vertex symbol. A shortest closed path may have a
        shortcut, and a vertex symbol counts only rings, cycles without one
        [17, 18, 19]: on the quartz Si net this gives 8_9 at the 8-ring angles,
        where the vertex symbol has 8(7), and on the simple cubic lattice
        6_4 at the straight angles, which lie on no ring ('*' in its vertex
        symbol). :meth:`vertex_symbol` on primitive rings gives the vertex
        symbol."""
        if self.criterion == "primitive":
            raise ValueError("shortest closed paths are found per angle "
                             "(King) or per bond (Guttman); for primitive "
                             "rings, vertex_symbol gives the per-angle record")
        return ".".join("*" if size == 0 else f"{size}_{count}"
                        for size, count in self._base_records(atom))

    def vertex_symbol(self, atom: int) -> str:
        """The vertex symbol of one node [17, 18, 19]: for every angle, the size
        of the smallest ring through it and, when there are several, their
        number, as RCSR writes it ('6(2)' for two six-membered rings), joined
        by '.'. A ring here is a primitive ring, a cycle with no shortcut
        [6, 7]; read that way, the symbol equals RCSR's published one at the
        24 vertex orbits of 18 named nets and at 48 of 50 orbits of a random
        sweep, the other 2 being one net's symbols listed in swapped order
        (module docstring, CHECKED AGAINST).
        Only a result of ``ring_statistics(graph, 'primitive', max_size)``
        gives one.

        Two differences from the published text. The entries are sorted by
        size and number, while RCSR lists them in an order of its own (RCSR
        gis 4.4.4.8(2).8.8 is 4.4.4.8.8.8(2) here; that the order pairs
        opposite angles is a reference to verify); compare the entries as a
        multiset. '*' marks an angle on no ring of at most max_size nodes;
        the published '*' is an angle on no ring of any size, which a
        bounded search cannot establish, so the two agree when no ring
        through the angle is larger than max_size (the straight angles of
        the simple cubic lattice are '*' in both). ``n_bases_acyclic``
        counts the angles shown to lie on no cycle at all."""
        if self.criterion != "primitive":
            raise ValueError("a vertex symbol counts rings, cycles without a "
                             "shortcut, which King's and Guttman's shortest "
                             "closed paths need not be; run ring_statistics "
                             "with criterion 'primitive' for it "
                             "(shortest_cycle_symbol gives the King or "
                             "Guttman record)")
        return ".".join("*" if size == 0 else
                        (str(size) if count == 1 else f"{size}({count})")
                        for size, count in self._base_records(atom))

    def t_size_counts(self) -> dict[tuple[int, int], int]:
        """(size in nodes, size in T atoms) -> distinct rings."""
        if self.t_sizes is None:
            raise ValueError("no t_elements were given to ring_statistics")
        out: dict[tuple[int, int], int] = {}
        for ring, t in zip(self.rings, self.t_sizes.tolist()):
            out[(ring.size, t)] = out.get((ring.size, t), 0) + 1
        return dict(sorted(out.items()))

    def as_rows(self) -> list[dict]:
        """One row per ring size, for exporters.write_csv / write_xlsx."""
        by_t: dict[int, set[int]] = {}
        if self.t_sizes is not None:
            for ring, t in zip(self.rings, self.t_sizes.tolist()):
                by_t.setdefault(ring.size, set()).add(t)
        rows = []
        for k, size in enumerate(self.sizes.tolist()):
            if self.t_sizes is None:
                t_text = "not given"
            elif size not in by_t:
                t_text = ""
            else:
                ts = sorted(by_t[size])
                t_text = str(ts[0]) if len(ts) == 1 else \
                    ", ".join(str(t) for t in ts)
            rows.append({
                "descriptor": f"rings ({self.criterion})",
                "ring size (nodes)": size, "ring size (T atoms)": t_text,
                "rings": int(self.ring_counts[k]),
                "R_C": float(self.rc[k]), "R_N": float(self.rn[k]),
                "P_N": float(self.pn[k]), "P_max": float(self.p_max[k]),
                "P_min": float(self.p_min[k])})
        return rows

    def __repr__(self) -> str:
        return (f"<RingStatistics {self.criterion} <= {self.max_size}: "
                f"{self.n_rings} rings over {self.n_nodes} nodes>")


def ring_statistics(graph: NetworkGraph, criterion: str, max_size: int, *,
                    t_elements: Iterable[str] | None = None,
                    progress: Callable[[int, int], None] | None = None,
                    cancelled: Callable[[], bool] | None = None
                    ) -> RingStatistics:
    """Rings of at most ``max_size`` nodes under one criterion.

    ``criterion`` is 'king', 'guttman' or 'primitive' (module docstring).
    ``max_size`` has no default: the search depth, and its cost, follow from
    it. ``t_elements`` (optional, no default set) names the elements counted
    as T atoms in each ring. ``progress(done, total)`` and ``cancelled()``
    are polled per node (King), per bond (Guttman), or per node in each of
    two passes (primitive: the ring search, then the per-angle records of
    the vertex symbol, so ``total`` is twice the nodes); ``progress`` is
    called at every 1 % of ``total`` and at least every
    ``PROGRESS_INTERVAL_S`` of work, and a True from ``cancelled`` raises
    :class:`Cancelled`.

    The cost grows with the number of closed paths of at most ``max_size``
    nodes, so with the graph's density as well as its size: a graph that
    holds the modifier-anion bonds (``graph_from_bonds`` with every element,
    its default) closes far more of them than the former-anion graph of the
    same frame (module docstring, TIMINGS).
    """
    graph = _check_graph(graph)
    if criterion not in CRITERIA:
        raise ValueError(f"criterion {criterion!r} is not one of {CRITERIA}")
    max_size = _whole(max_size, "max_size", 3)
    t_set = None if t_elements is None else _symbols(t_elements, "t_elements")
    n = graph.n_atoms
    lift = _Lift(graph, max_size)
    deltas = lift.deltas
    nodes = graph.nodes
    node_list = nodes.tolist()
    ring_index: dict[tuple, int] = {}
    ring_keys: list[tuple] = []
    finders: list[set[int]] = []
    base: dict[int, list[tuple[int, int]]] = {a: [] for a in node_list}
    counts = {"bases": 0, "unclosed": 0, "acyclic": 0}
    search = None

    def add(keys, *at):
        """Record a ring found from the members at positions ``at`` of
        ``keys``. A finder is (atom, position in the canonical ring): a ring
        through two images of one atom is two rings through its copy in
        cell (0, 0, 0), as on the lift."""
        canon, where, sign = _canonical(keys, n)
        index = ring_index.get(canon)
        if index is None:
            index = len(ring_keys)
            ring_index[canon] = index
            ring_keys.append(canon)
            finders.append(set())
        size = len(keys)
        finders[index].update((keys[q] % n, (where + sign * q) % size)
                              for q in at)

    if criterion == "king":
        poll = _Poller(cancelled, progress, len(node_list))
        for done, atom in enumerate(node_list, 1):
            _king_node(atom, lift, deltas, n, max_size, base[atom], counts,
                       add)
            poll(done)
    elif criterion == "guttman":
        total = graph.n_edges
        poll = _Poller(cancelled, progress, total)
        us, vs = graph.u.tolist(), graph.v.tolist()
        weights = np.array([lift.width ** 2, lift.width, 1], dtype=np.int64)
        codes = (graph.shift @ weights).tolist()
        for e in range(total):
            counts["bases"] += 2
            if len(deltas[us[e]]) == 1 or len(deltas[vs[e]]) == 1:
                # an end bonded to nothing else: the bond is on no cycle
                counts["unclosed"] += 2
                counts["acyclic"] += 2
                base[us[e]].append((0, 0))
                base[vs[e]].append((0, 0))
                poll(e + 1)
                continue
            x = lift.key(us[e])
            y = vs[e] + n * (lift.origin + codes[e])
            dist, preds, exhausted = _bfs(deltas, n, y, max_size - 1,
                                          banned_edge=(x, y), targets=(x,))
            length = dist.get(x)
            if length is None:
                counts["unclosed"] += 2
                if exhausted or _finite_within(deltas, n, x, max_size - 1,
                                               banned_edge=(x, y)):
                    counts["acyclic"] += 2
                record = (0, 0)
                found = []
            else:
                found = _paths(preds, y, x, {})
                record = (length + 1, len(found))
            base[us[e]].append(record)
            base[vs[e]].append(record)
            for path in found:                  # path: (v at shift) ... (u)
                add(path, 0, len(path) - 1)
            poll(e + 1)
    else:
        poll = _Poller(cancelled, progress, 2 * len(node_list))
        accepted, search = _primitive_search(graph, node_list, lift, deltas,
                                             n, max_size, poll)
        for canon in accepted:
            ring_index[canon] = len(ring_keys)
            ring_keys.append(canon)
            finders.append({(key % n, q) for q, key in enumerate(canon)})
        _primitive_angles(accepted, node_list, lift, deltas, n, max_size,
                          base, counts, poll)

    return _ring_result(graph, lift, criterion, max_size, ring_keys, finders,
                        base, counts, t_set, search)


def _king_node(atom, lift, deltas, n, max_size, records, counts, add) -> None:
    """King's rings through every angle of one node (module docstring).

    A neighbour bonded to nothing but this node closes no cycle, so its
    angles are recorded as acyclic with no search; a search that runs out of
    vertices establishes the same for the neighbours it did not reach."""
    start = lift.key(atom)
    neighbours = [start + d for d in deltas[atom]]
    leaf = [len(deltas[x % n]) == 1 for x in neighbours]
    depth = max_size - 2
    for a in range(len(neighbours) - 1):
        first = neighbours[a]
        rest = range(a + 1, len(neighbours))
        targets = [neighbours[b] for b in rest if not leaf[b]]
        dist = preds = None
        exhausted = True
        if not leaf[a] and targets:
            dist, preds, exhausted = _bfs(deltas, n, first, depth,
                                          banned=start, targets=targets)
        memo: dict = {}
        for b in rest:
            other = neighbours[b]
            counts["bases"] += 1
            if leaf[a] or leaf[b]:
                counts["unclosed"] += 1
                counts["acyclic"] += 1
                records.append((0, 0))
                continue
            length = dist.get(other)
            if length is None:
                counts["unclosed"] += 1
                if exhausted or _finite_within(deltas, n, other, depth,
                                               banned=start):
                    counts["acyclic"] += 1
                records.append((0, 0))
                continue
            found = _paths(preds, first, other, memo)
            records.append((length + 2, len(found)))
            for path in found:
                add((start,) + path, 0)


def _search_order(graph: NetworkGraph, node_list: list[int]) -> list[int]:
    """The nodes in breadth-first order over the quotient graph: each
    component from its lowest row, the neighbours of a node in ascending row
    order. Bonded nodes then sit close together in the order, so the members
    of a ring are searched within a short stretch of it."""
    n = graph.n_atoms
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for a, b in zip(graph.u.tolist(), graph.v.tolist()):
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    seen = [False] * n
    order: list[int] = []
    for root in node_list:
        if seen[root]:
            continue
        seen[root] = True
        k = len(order)
        order.append(root)
        while k < len(order):
            x = order[k]
            k += 1
            for y in sorted(adjacency[x]):
                if not seen[y]:
                    seen[y] = True
                    order.append(y)
    return order


def _key_codec(lift: _Lift):
    """(pack, unpack) between a canonical ring tuple and a bytes key of
    fixed-width integers, or (None, None) when the keys do not fit in 64
    bits (a max_size far beyond any ring search). A 20-member ring takes 836
    bytes as a tuple of Python integers and 113 (32-bit) or 193 (64-bit)
    bytes packed (sys.getsizeof, measured)."""
    # a member of a canonical ring is its atom row plus n times the code of
    # its cell less the first member's; two members of a ring of at most
    # depth nodes are at most depth // 2 bonds apart, so each component of
    # that cell difference is below reach and its code below this
    width = lift.width
    bound = lift.n * (1 + (lift.reach - 1) * (width * width + width + 1))
    if bound < 2 ** 31:
        letter = "i"
    elif bound < 2 ** 63:
        letter = "q"
    else:
        return None, None
    item = struct.calcsize(f"<{letter}")
    codecs: dict[int, struct.Struct] = {}

    def codec(size):
        got = codecs.get(size)
        if got is None:
            got = codecs[size] = struct.Struct(f"<{size}{letter}")
        return got

    def pack(canon):
        return codec(len(canon)).pack(*canon)

    def unpack(key):
        return codec(len(key) // item).unpack(key)

    return pack, unpack


def _primitive_search(graph, node_list, lift, deltas, n, max_size,
                      poll) -> tuple[list, dict]:
    """The canonical forms of the primitive rings of at most ``max_size``
    members (module docstring, "How primitive rings are found"), and the
    numbers of candidate cycles formed, entered and open at most at once.

    A candidate is kept only while it can still be primitive. The nodes are
    searched in :func:`_search_order`, and a candidate is entered only when
    its first member in that order produces it (one not produced there is
    not primitive); after each node the candidates waiting on it are
    checked: one this node did not produce at each of its positions is
    dropped, the others wait on their next member in the order, and one with
    no member left is primitive. Memory then follows the candidates still
    open, not every candidate ever formed, and each is held as a packed key
    (:func:`_key_codec`). Measured on a 3000-atom Na2O-3SiO2 graph holding
    the Na-O bonds (2026-10-07): rings to 16 nodes, 238 406 distinct
    candidates, all held to the end before, at most 26 657 open at once with
    the nodes in row order and 9 425 in breadth-first order; rings to 24
    nodes, 15 127 020 candidates formed, at most 300 786 open in row order
    and 120 569 in breadth-first order (holding every one to the end had
    committed 1.8 GB after 450 of the 3000 nodes)."""
    half = max_size // 2
    order = _search_order(graph, node_list)
    rank = [n] * n
    for k, atom in enumerate(order):
        rank[atom] = k
    pack, unpack = _key_codec(lift)
    masks: dict = {}
    waiting: dict[int, list] = {}
    accepted: list[tuple] = []
    tally = {"formed": 0, "entered": 0, "peak open": 0}
    for done, atom in enumerate(order, 1):
        here = rank[atom]
        start = lift.key(atom)
        dist, preds, _ = _bfs(deltas, n, start, half)
        layers: dict[int, list[int]] = {}
        for key, depth in dist.items():
            layers.setdefault(depth, []).append(key)
        memo: dict = {}
        inner: dict[tuple, frozenset] = {}
        entered = waiting.setdefault(atom, [])

        def interior(path, drop_last):
            got = inner.get((path, drop_last))
            if got is None:
                got = frozenset(path[1:-1] if drop_last else path[1:])
                inner[(path, drop_last)] = got
            return got

        def record(keys):
            tally["formed"] += 1
            canon, position, _ = _canonical(keys, n)
            key = canon if pack is None else pack(canon)
            mask = masks.get(key)
            if mask is None:
                if min([rank[k % n] for k in keys]) != here:
                    # its first member in the order, searched before, did
                    # not produce it (or it was dropped since): not primitive
                    return
                masks[key] = 1 << position
                entered.append(key)
                tally["entered"] += 1
            else:
                masks[key] = mask | (1 << position)

        for depth in range(1, half + 1):
            layer = layers.get(depth, ())
            if 2 * depth <= max_size and depth >= 2:       # even rings
                for far in layer:
                    if len(preds[far]) < 2:
                        continue
                    found = _paths(preds, start, far, memo)
                    for p, q in itertools.combinations(found, 2):
                        if p[1] == q[1] or p[-2] == q[-2]:
                            continue
                        if interior(p, True).isdisjoint(interior(q, True)):
                            record(p + q[-2:0:-1])
            if 2 * depth + 1 <= max_size:                  # odd rings
                for far in layer:
                    for d in deltas[far % n]:
                        other = far + d
                        if other <= far or dist.get(other) != depth:
                            continue
                        for p in _paths(preds, start, far, memo):
                            for q in _paths(preds, start, other, memo):
                                if p[1] == q[1]:
                                    continue
                                if interior(p, False).isdisjoint(
                                        interior(q, False)):
                                    record(p + q[:0:-1])
        if len(masks) > tally["peak open"]:
            tally["peak open"] = len(masks)
        for key in waiting.pop(atom):
            mask = masks[key]
            canon = key if unpack is None else unpack(key)
            following = None
            produced = True
            for q, member_key in enumerate(canon):
                member = member_key % n
                if member == atom:
                    if not mask >> q & 1:
                        produced = False
                        break
                elif rank[member] > here and (
                        following is None or rank[member] < rank[following]):
                    following = member
            if not produced:
                del masks[key]
            elif following is not None:
                waiting.setdefault(following, []).append(key)
            else:
                del masks[key]
                if mask == (1 << len(canon)) - 1:
                    accepted.append(tuple(canon))
        poll(done)
    return accepted, tally


def _primitive_angles(accepted, node_list, lift, deltas, n, max_size, base,
                      counts, poll) -> None:
    """Per angle of every node, the smallest primitive ring through it and
    the number of them (the vertex symbol), (0, 0) where there is none of at
    most ``max_size``; those are then tested for lying on no cycle at all."""
    position: dict[int, dict[int, int]] = {}
    best: dict[int, dict[tuple[int, int], list[int]]] = \
        {atom: {} for atom in node_list}
    for canon in accepted:
        size = len(canon)
        for q in range(size):
            atom = canon[q] % n
            steps = position.get(atom)
            if steps is None:
                steps = {d: k for k, d in enumerate(deltas[atom])}
                position[atom] = steps
            a = steps[canon[q - 1] - canon[q]]
            b = steps[canon[(q + 1) % size] - canon[q]]
            angle = (a, b) if a < b else (b, a)
            got = best[atom].get(angle)
            if got is None or size < got[0]:
                best[atom][angle] = [size, 1]
            elif size == got[0]:
                got[1] += 1
    offset = len(node_list)
    for done, atom in enumerate(node_list, 1):
        start = lift.key(atom)
        steps = deltas[atom]
        records = base[atom]
        for a in range(len(steps) - 1):
            for b in range(a + 1, len(steps)):
                counts["bases"] += 1
                got = best[atom].get((a, b))
                if got is not None:
                    records.append((got[0], got[1]))
                    continue
                records.append((0, 0))
                counts["unclosed"] += 1
                if _angle_acyclic(deltas, n, start, start + steps[a],
                                  start + steps[b], max_size - 2):
                    counts["acyclic"] += 1
        poll(offset + done)


def _ring_result(graph, lift, criterion, max_size, ring_keys, finders, base,
                 tally, t_set, search=None) -> RingStatistics:
    n = graph.n_atoms
    nodes = graph.nodes
    n_nodes = nodes.size
    sizes = np.arange(3, max_size + 1)
    node_index = np.full(n, -1, dtype=np.int64)
    node_index[nodes] = np.arange(n_nodes)
    order = sorted(range(len(ring_keys)),
                   key=lambda r: (len(ring_keys[r]), ring_keys[r]))
    rings = []
    # columns up to the largest ring found: every larger size is 0, and a
    # max_size far above the rings would otherwise cost n_nodes x max_size
    columns = max((len(keys) for keys in ring_keys), default=2) - 2
    found = np.zeros((n_nodes, columns), dtype=np.int64)
    symbols = graph.elements
    for r in order:
        keys = ring_keys[r]
        atoms = tuple(int(k % n) for k in keys)
        cells = tuple(lift.decode_cell(k // n) for k in keys)
        rings.append(Ring(atoms, cells, tuple(str(symbols[a]) for a in atoms),
                          frame_digest=graph.frame_digest))
        column = len(keys) - 3
        for atom, _ in finders[r]:
            found[node_index[atom], column] += 1
    counts = np.bincount([ring.size - 3 for ring in rings],
                         minlength=sizes.size).astype(np.int64)
    has = found > 0
    rc = counts / n_nodes
    rn = np.zeros(sizes.size)
    rn[:columns] = found.sum(axis=0) / n_nodes
    pn = np.zeros(sizes.size)
    pn[:columns] = has.sum(axis=0) / n_nodes
    any_ring = has.any(axis=1)
    if columns:
        largest = np.where(any_ring, columns - 1 - np.argmax(has[:, ::-1],
                                                             axis=1), -1)
        smallest = np.where(any_ring, np.argmax(has, axis=1), -1)
    else:                                       # no ring at all
        largest = smallest = np.full(n_nodes, -1, dtype=np.int64)
    pn_max = np.bincount(largest[any_ring], minlength=sizes.size) / n_nodes
    pn_min = np.bincount(smallest[any_ring], minlength=sizes.size) / n_nodes
    with np.errstate(invalid="ignore", divide="ignore"):
        p_max = np.where(pn > 0, pn_max / np.where(pn > 0, pn, 1.0), np.nan)
        p_min = np.where(pn > 0, pn_min / np.where(pn > 0, pn, 1.0), np.nan)
    t_sizes = None
    if t_set is not None:
        t_sizes = np.array([sum(1 for e in ring.elements if e in t_set)
                            for ring in rings], dtype=np.int64)
    notes = [f"rings of at most {max_size} nodes ({criterion}); every node "
             f"initiates the search; normalised over the {n_nodes} nodes of "
             "the graph",
             f"P_max(n) counts the nodes whose largest ring found is of n "
             f"nodes, so it depends on max_size: rings above {max_size} nodes "
             f"are not searched, and P_max({max_size}) = 1 wherever "
             f"P_N({max_size}) > 0, by construction (P_max = 1 at the largest "
             "size found, as the R.I.N.G.S. manual states)"]
    if criterion == "primitive":
        notes.append(f"primitive rings of more than {max_size} nodes are not "
                     "searched; their number is not known")
    if search is not None:
        notes.append(
            f"primitive search: {search['formed']} candidate cycles formed "
            f"from the nodes, {search['entered']} of them held until decided, "
            f"at most {search['peak open']} at once")
    without = int((~any_ring).sum())
    if without:
        notes.append(f"{without} of {n_nodes} nodes are in no ring of at most "
                     f"{max_size} nodes found by this criterion")
    n_bases, unclosed, acyclic = (tally["bases"], tally["unclosed"],
                                  tally["acyclic"])
    if unclosed:
        what = "bond ends" if criterion == "guttman" else "angles"
        ring = "primitive ring" if criterion == "primitive" else "ring"
        notes.append(
            f"{unclosed} of {n_bases} {what} lie on no {ring} of at most "
            f"{max_size} nodes: {acyclic} of them on no cycle of any size (a "
            "neighbour bonded to nothing else, or a finite piece of the "
            f"graph), and {unclosed - acyclic} may lie on a larger "
            f"{ring}, which is not searched")
    if (pn == 0).any():
        notes.append("P_max and P_min are NaN at sizes where P_N is 0 "
                     "(no node finds a ring of that size)")
    if graph.source == "bridging anion":
        notes.append("sizes are in T atoms (the nodes are formers); the size in "
                     "all atoms is not defined on a bridged graph")
    if t_set is not None:
        present = {str(s) for s in np.unique(symbols[nodes])}
        missing = sorted(t_set - present)
        if missing:
            notes.append(f"t_elements: {', '.join(missing)} not among the "
                         "graph's nodes, so they count 0 in every ring")
    notes += [f"graph: {note}" for note in graph.notes]
    base_rings = tuple(tuple(base.get(int(a), ())) for a in nodes.tolist())
    return RingStatistics(
        criterion=criterion, max_size=max_size,
        graph_definition=graph.definition, graph_source=graph.source,
        node_elements=graph.node_elements, nodes=_frozen(nodes),
        sizes=_frozen(sizes), rings=tuple(rings), ring_counts=_frozen(counts),
        found_compact=_frozen(found), rc=_frozen(rc), rn=_frozen(rn),
        pn=_frozen(pn), pn_max=_frozen(pn_max), pn_min=_frozen(pn_min),
        p_max=_frozen(p_max), p_min=_frozen(p_min), base_rings=base_rings,
        n_bases=n_bases, n_bases_unclosed=unclosed, n_bases_acyclic=acyclic,
        n_nodes_without_ring=without, t_elements=t_set,
        t_sizes=None if t_sizes is None else _frozen(t_sizes),
        n_candidates=0 if search is None else search["formed"],
        peak_open_candidates=0 if search is None else search["peak open"],
        frame_digest=graph.frame_digest, notes=tuple(notes))


# ---------------------------------------------------------------------------
# coordination sequences
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class CoordinationSequences:
    """N_k for k = 1 .. n_shells from every centre node.

    ``counts`` (n_centres, n_shells); ``centres`` the atom rows and
    ``elements`` their symbols.
    """

    n_shells: int
    centres: np.ndarray
    elements: np.ndarray
    counts: np.ndarray
    graph_definition: str
    notes: tuple[str, ...] = ()

    @property
    def shells(self) -> np.ndarray:
        return np.arange(1, self.n_shells + 1)

    @property
    def cumulative(self) -> np.ndarray:
        """1 + N_1 + ... + N_k, per centre (the node itself included)."""
        return 1 + np.cumsum(self.counts, axis=1)

    def by_element(self) -> dict[str, np.ndarray]:
        """Element -> (n_atoms_of_element, n_shells) counts."""
        return {str(e): self.counts[self.elements == e]
                for e in np.unique(self.elements)}

    def mean_by_element(self) -> dict[str, np.ndarray]:
        """Element -> mean N_k over its centres (math.fsum per shell)."""
        out = {}
        for element, rows in self.by_element().items():
            out[element] = np.array([math.fsum(col) / rows.shape[0]
                                     for col in rows.T.tolist()])
        return out

    def as_rows(self) -> list[dict]:
        rows = []
        for element, mean in self.mean_by_element().items():
            group = self.counts[self.elements == element]
            for k in range(self.n_shells):
                rows.append({"descriptor": "coordination sequence",
                             "element": element, "shell k": k + 1,
                             "mean N_k": float(mean[k]),
                             "min N_k": int(group[:, k].min()),
                             "max N_k": int(group[:, k].max()),
                             "centres": int(group.shape[0])})
        return rows


def coordination_sequences(graph: NetworkGraph, n_shells: int, *,
                           centres: Iterable[str] | None = None,
                           progress: Callable[[int, int], None] | None = None,
                           cancelled: Callable[[], bool] | None = None
                           ) -> CoordinationSequences:
    """N_k on the lift from every node of the ``centres`` elements (None:
    every node). ``n_shells`` has no default."""
    graph = _check_graph(graph)
    n_shells = _whole(n_shells, "n_shells", 1)
    nodes = graph.nodes
    notes = []
    if centres is not None:
        wanted = _symbols(centres, "centres")
        outside = sorted(wanted - graph.node_elements)
        if outside:
            raise ValueError(f"{', '.join(outside)} are not node elements of "
                             "this graph")
        nodes = nodes[np.isin(graph.elements[nodes], sorted(wanted))]
        if nodes.size == 0:
            raise ValueError("no node of the centre elements is in this frame")
    n = graph.n_atoms
    lift = _Lift(graph, n_shells)
    deltas = lift.deltas
    counts = np.zeros((nodes.size, n_shells), dtype=np.int64)
    poll = _Poller(cancelled, progress, nodes.size)
    for row, atom in enumerate(nodes.tolist()):
        previous: set[int] = set()
        shell = {lift.key(atom)}
        for k in range(n_shells):
            reached = {x + d for x in shell for d in deltas[x % n]}
            reached -= shell
            reached -= previous
            counts[row, k] = len(reached)
            previous, shell = shell, reached
            if not shell:
                break
        poll(row + 1)
    notes += [f"graph: {note}" for note in graph.notes]
    return CoordinationSequences(
        n_shells=n_shells, centres=_frozen(nodes),
        elements=_frozen(graph.elements[nodes]), counts=_frozen(counts),
        graph_definition=graph.definition, notes=tuple(notes))


# ---------------------------------------------------------------------------
# polyhedral connectivity
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class PolyhedralConnectivity:
    """Pairs of centre polyhedra sharing ligands, by centre-pair type.

    ``counts[type][kind]``: pairs sharing one ligand ('corner'), two ('edge')
    or three or more ('face'); ``fractions`` the same over the pairs of the
    type that share any ligand (each set passes check_fractions);
    ``shared[type]``: pairs by the exact number of shared ligands.
    ``per_centre`` (N, 3): corner-, edge- and face-sharing partners of each
    centre atom (-1 rows for atoms that are not centres); a partner that is
    the atom's own image counts once per image. ``isolated[element]``:
    centres sharing no ligand with any centre. ``pairs``: (P, 6) rows
    (centre, centre, shift x, y, z, shared ligands).
    """

    centres: frozenset
    ligands: frozenset
    counts: dict
    fractions: dict
    shared: dict
    per_centre: np.ndarray
    isolated: dict
    pairs: np.ndarray
    graph_definition: str
    notes: tuple[str, ...] = ()

    def as_rows(self) -> list[dict]:
        rows = []
        for pair_type, kinds in self.counts.items():
            total = sum(kinds.values())
            for kind in SHARING_KINDS:
                rows.append({"descriptor": "polyhedral connectivity",
                             "centre pair": f"{pair_type[0]}-{pair_type[1]}",
                             "sharing": kind, "pairs": kinds[kind],
                             "fraction": self.fractions[pair_type][kind]
                             if total else float("nan")})
        return rows


def polyhedral_connectivity(graph: NetworkGraph, *, centres: Iterable[str],
                            ligands: Iterable[str]) -> PolyhedralConnectivity:
    """Corner, edge and face sharing between the polyhedra of ``centres``.

    A polyhedron is a centre atom with its graph neighbours of the
    ``ligands`` elements. Both sets are required and may not overlap. A
    centre's edges to atoms of other elements (another centre on a distance
    graph, an F when the ligands are O) are counted in the notes.
    """
    graph = _check_graph(graph)
    centre_set = _symbols(centres, "centres")
    ligand_set = _symbols(ligands, "ligands")
    overlap = sorted(centre_set & ligand_set)
    if overlap:
        raise ValueError(f"{', '.join(overlap)} given both as centre and as "
                         "ligand")
    symbols = graph.elements
    notes = _absent_note(centre_set, symbols, "centres")
    notes += _absent_note(ligand_set, symbols, "ligands")
    for what, wanted in (("centres", centre_set), ("ligands", ligand_set)):
        outside = sorted((wanted - graph.node_elements)
                         & {str(s) for s in np.unique(symbols)})
        if outside:
            raise ValueError(f"{what} {', '.join(outside)} are not node "
                             "elements of this graph")
    centre_names = sorted(centre_set)
    ligand_names = sorted(ligand_set)
    u, v, shift = graph.u, graph.v, graph.shift
    centre_u, centre_v = (np.isin(symbols[u], centre_names),
                          np.isin(symbols[v], centre_names))
    ligand_u, ligand_v = (np.isin(symbols[u], ligand_names),
                          np.isin(symbols[v], ligand_names))
    forward = centre_u & ligand_v
    backward = centre_v & ligand_u
    stray = (centre_u & ~ligand_v) | (centre_v & ~ligand_u)
    if stray.any():
        names = []
        for x, y, cx in zip(symbols[u[stray]].tolist(),
                            symbols[v[stray]].tolist(),
                            centre_u[stray].tolist()):
            names.append(f"{x}-{y}" if cx else f"{y}-{x}")
        pairs, numbers = np.unique(names, return_counts=True)
        notes.append(
            f"{int(stray.sum())} edge(s) of the graph join a centre to an "
            "atom outside the ligands, and that atom is not a vertex of its "
            "polyhedron: " + ", ".join(f"{p} {k}" for p, k in
                                       zip(pairs, numbers)))
    cen = np.concatenate([u[forward], v[backward]])
    lig = np.concatenate([v[forward], u[backward]])
    # the centre's cell with the ligand in cell (0, 0, 0)
    cell = np.concatenate([-shift[forward], shift[backward]])
    order = np.lexsort((cen, lig))
    cen, lig, cell = cen[order], lig[order], cell[order]
    pu, pv, ps = [], [], []
    if lig.size:
        starts = np.flatnonzero(np.r_[True, lig[1:] != lig[:-1]])
        sizes = np.diff(np.r_[starts, lig.size])
        for k in np.unique(sizes[sizes >= 2]):
            members = starts[sizes == k][:, None] + np.arange(k)[None, :]
            for p, q in itertools.combinations(range(int(k)), 2):
                pu.append(cen[members[:, p]])
                pv.append(cen[members[:, q]])
                ps.append(cell[members[:, q]] - cell[members[:, p]])
    if pu:
        a, b, s, _, shared = _unique_edges(np.concatenate(pu),
                                           np.concatenate(pv),
                                           np.concatenate(ps))
    else:
        a = b = shared = np.zeros(0, dtype=np.int64)
        s = np.zeros((0, 3), dtype=np.int64)
    kind = np.minimum(shared, 3) - 1                 # 0 corner, 1 edge, 2 face
    centre_rows = np.flatnonzero(graph.in_graph & np.isin(symbols,
                                                          centre_names))
    per_centre = np.full((graph.n_atoms, 3), -1, dtype=np.int64)
    per_centre[centre_rows] = 0
    for column in range(3):
        sel = kind == column
        per_centre[:, column] += (np.bincount(a[sel], minlength=graph.n_atoms)
                                  + np.bincount(b[sel],
                                                minlength=graph.n_atoms))
    per_centre[np.setdiff1d(np.arange(graph.n_atoms), centre_rows)] = -1
    present = sorted({str(x) for x in symbols[centre_rows]})
    types = sorted({_pair_key(x, y) for x in present for y in present})
    counts = {t: {k: 0 for k in SHARING_KINDS} for t in types}
    shared_by = {t: {} for t in types}
    for x, y, m in zip(symbols[a].tolist(), symbols[b].tolist(),
                       shared.tolist()):
        t = _pair_key(x, y)
        counts[t][SHARING_KINDS[min(m, 3) - 1]] += 1
        shared_by[t][m] = shared_by[t].get(m, 0) + 1
    fractions = {}
    for t in types:
        total = sum(counts[t].values())
        if total:
            fractions[t] = {k: counts[t][k] / total for k in SHARING_KINDS}
            check_fractions(fractions[t], f"polyhedral sharing {t[0]}-{t[1]}")
        else:
            fractions[t] = {k: float("nan") for k in SHARING_KINDS}
            notes.append(f"no {t[0]}-{t[1]} pair of polyhedra shares a ligand")
    alone = centre_rows[(per_centre[centre_rows] == 0).all(axis=1)]
    isolated = {e: int((symbols[alone] == e).sum()) for e in present}
    no_ligand = centre_rows[~np.isin(centre_rows, cen)]
    if no_ligand.size:
        names, numbers = np.unique(symbols[no_ligand], return_counts=True)
        notes.append("centres with no ligand edge (polyhedra of no vertex): "
                     + ", ".join(f"{x} {c}" for x, c in zip(names, numbers)))
    notes.append("'edge' counts two shared ligands, whether or not the two "
                 "are joined by an edge of either polyhedron; 'face' counts "
                 "three or more")
    notes += [f"graph: {note}" for note in graph.notes]
    rows = np.column_stack([a, b, s, shared]) if a.size else \
        np.zeros((0, 6), dtype=np.int64)
    return PolyhedralConnectivity(
        centres=centre_set, ligands=ligand_set, counts=counts,
        fractions=fractions, shared={t: dict(sorted(d.items()))
                                     for t, d in shared_by.items()},
        per_centre=_frozen(per_centre), isolated=isolated,
        pairs=_frozen(rows.astype(np.int64)),
        graph_definition=graph.definition, notes=tuple(notes))


# ---------------------------------------------------------------------------
# connected components and percolation
# ---------------------------------------------------------------------------

def _lattice_basis(vectors) -> list[tuple[int, int, int]]:
    """The Hermite normal form basis of the integer lattice that
    ``vectors`` generate (rows, exact integer arithmetic): echelon form,
    positive pivots, entries above a pivot reduced to 0 <= entry < pivot.
    The basis depends only on the lattice, not on the generators given."""
    rows = [list(v) for v in vectors if any(v)]
    basis: list[list[int]] = []
    for col in range(3):
        while True:
            live = [r for r in rows if r[col] != 0]
            if len(live) <= 1:
                break
            pivot = min(live, key=lambda r: abs(r[col]))
            for r in live:
                if r is not pivot:
                    q = r[col] // pivot[col]
                    for k in range(3):
                        r[k] -= q * pivot[k]
            rows = [r for r in rows if any(r)]
        live = [r for r in rows if r[col] != 0]
        if live:
            pivot = live[0]
            if pivot[col] < 0:
                pivot = [-x for x in pivot]
            for upper in basis:
                q = upper[col] // pivot[col]
                for k in range(3):
                    upper[k] -= q * pivot[k]
            basis.append(pivot)
            rows = [r for r in rows if r[col] == 0]
    return [tuple(r) for r in basis]


def _copies(basis) -> int:
    """The index of the lattice spanned by ``basis`` in the integer points
    of the space it spans: the gcd of its maximal minors (|det| in 3-D)."""
    if len(basis) == 0:
        return 1
    if len(basis) == 1:
        return math.gcd(*basis[0])
    if len(basis) == 2:
        p, q = basis
        return math.gcd(p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2],
                        p[0] * q[1] - p[1] * q[0])
    p, q, r = basis
    return abs(p[0] * (q[1] * r[2] - q[2] * r[1])
               - p[1] * (q[0] * r[2] - q[2] * r[0])
               + p[2] * (q[0] * r[1] - q[1] * r[0]))


@dataclass(frozen=True, eq=False)
class Components:
    """Connected pieces of the graph, largest first.

    ``label`` (N,) the component of each atom (-1 outside the graph);
    ``sizes`` nodes per component; ``dimensionality`` 0-3 (rank of the
    translations that map the piece onto itself); ``spans`` (n, 3) whether
    the piece reaches every cell along box axis a, b, c; ``translations`` per
    component, a basis of those translations (Hermite normal form rows, in
    cells); ``copies`` per component, the number of disjoint pieces of the
    periodic network it stands for (module docstring); ``node_cell`` (N, 3)
    the cell that places each node in its piece's connected copy.

    A component is one of the box's quotient graph. When ``copies`` is k > 1,
    the network holds k interpenetrating pieces that a box translation maps
    onto each other, so the box shows them as one component and a supercell
    of it as several; each piece holds sizes / k of the nodes per box. The
    order (largest first), ``largest_fraction`` and ``n_pieces`` count those
    pieces, so they do not change when the box is replaced by a supercell.
    """

    label: np.ndarray
    sizes: np.ndarray
    dimensionality: np.ndarray
    spans: np.ndarray
    translations: tuple
    copies: np.ndarray
    node_cell: np.ndarray
    n_nodes: int
    graph_definition: str
    frame_digest: str
    notes: tuple[str, ...] = ()

    @property
    def n_components(self) -> int:
        """Components of the box's quotient graph."""
        return int(self.sizes.shape[0])

    @property
    def n_pieces(self) -> int:
        """Disjoint pieces of the network per box: the sum of ``copies``."""
        return int(self.copies.sum())

    @property
    def piece_nodes(self) -> np.ndarray:
        """Nodes per box in one piece of each component: sizes / copies."""
        return self.sizes / self.copies

    @property
    def largest_fraction(self) -> float:
        """The largest piece's share of the nodes."""
        return float(self.piece_nodes[0]) / self.n_nodes

    def size_counts(self) -> dict[int, int]:
        """Component size -> number of components."""
        values, counts = np.unique(self.sizes, return_counts=True)
        return {int(s): int(c) for s, c in zip(values, counts)}

    def nodes_by_size(self) -> dict[int, int]:
        """Component size -> nodes in components of that size."""
        return {s: s * c for s, c in self.size_counts().items()}

    def unwrapped_cart_ang(self, frame: Frame) -> np.ndarray:
        """(N, 3) positions with each finite piece in one connected copy."""
        frame = _check_frame(frame)
        if _frame_digest(frame) != self.frame_digest:
            raise ValueError("these components were found on another frame")
        return frame.cart_ang + self.node_cell @ frame.box_ang

    def as_rows(self) -> list[dict]:
        rows = []
        for c in range(self.n_components):
            rows.append({"descriptor": "connected component",
                         "component": c, "nodes": int(self.sizes[c]),
                         "fraction of nodes": float(self.sizes[c]) / self.n_nodes,
                         "dimensionality": int(self.dimensionality[c]),
                         "interpenetrating copies": int(self.copies[c]),
                         "spans a": bool(self.spans[c, 0]),
                         "spans b": bool(self.spans[c, 1]),
                         "spans c": bool(self.spans[c, 2])})
        return rows


def components(graph: NetworkGraph) -> Components:
    """Connected components, their dimensionality, their interpenetrating
    copies, and whether they span the box along each axis (module
    docstring)."""
    graph = _check_graph(graph)
    n = graph.n_atoms
    nodes = graph.nodes
    order = np.argsort(np.concatenate([graph.u, graph.v]), kind="stable")
    other = np.concatenate([graph.v, graph.u])[order]
    step = np.concatenate([graph.shift, -graph.shift])[order]
    bounds = np.searchsorted(np.concatenate([graph.u, graph.v])[order],
                             np.arange(n + 1))
    other_l = other.tolist()
    step_l = [tuple(r) for r in step.tolist()]
    raw = np.full(n, -1, dtype=np.int64)
    cell = np.zeros((n, 3), dtype=np.int64)
    cells = [None] * n
    count = 0
    for root in nodes.tolist():
        if raw[root] >= 0:
            continue
        raw[root] = count
        cells[root] = (0, 0, 0)
        queue = [root]
        while queue:
            x = queue.pop()
            cx = cells[x]
            for k in range(bounds[x], bounds[x + 1]):
                y = other_l[k]
                if raw[y] < 0:
                    raw[y] = count
                    t = step_l[k]
                    cells[y] = (cx[0] + t[0], cx[1] + t[1], cx[2] + t[2])
                    queue.append(y)
        count += 1
    for x in nodes.tolist():
        cell[x] = cells[x]
    # every closed walk's net translation is a sum of these (one per
    # non-tree edge), so they generate the component's translation lattice
    defect = cell[graph.u] + graph.shift - cell[graph.v]
    edge_comp = raw[graph.u]
    nonzero = defect.any(axis=1)
    basis_raw: list[list] = [[] for _ in range(count)]
    if nonzero.any():
        rows = np.unique(np.column_stack([edge_comp[nonzero],
                                          defect[nonzero]]), axis=0)
        for comp in np.unique(rows[:, 0]).tolist():
            vecs = [tuple(r) for r in rows[rows[:, 0] == comp][:, 1:].tolist()]
            basis_raw[comp] = _lattice_basis(vecs)
    copies_raw = np.array([_copies(b) for b in basis_raw], dtype=np.int64)
    sizes_raw = np.bincount(raw[nodes], minlength=count)
    first_row = np.full(count, n, dtype=np.int64)
    np.minimum.at(first_row, raw[nodes], nodes)
    rank = np.lexsort((first_row, -sizes_raw, -(sizes_raw / copies_raw)))
    relabel = np.empty(count, dtype=np.int64)
    relabel[rank] = np.arange(count)
    label = np.where(raw >= 0, relabel[np.maximum(raw, 0)], -1)
    sizes = sizes_raw[rank]
    copies = copies_raw[rank]
    translations = [basis_raw[r] for r in rank.tolist()]
    dimensionality = np.array([len(t) for t in translations], dtype=np.int64)
    spans = np.array([[any(vec[a] != 0 for vec in t) for a in range(3)]
                      for t in translations], dtype=bool).reshape(count, 3)
    pieces = sizes / copies
    notes = [f"{count} component(s) over {nodes.size} nodes; the largest "
             f"piece holds {pieces[0]:.6g} nodes per box ("
             f"{pieces[0] / nodes.size:.6g} of the nodes)"]
    split = np.flatnonzero(copies > 1)
    if split.size:
        notes.append(
            f"{split.size} component(s) are each several interpenetrating "
            "pieces of the periodic network that a box translation maps onto "
            "each other, so the box shows them as one component and a "
            "supercell as several: copies "
            + ", ".join(f"{int(copies[c])} (component {c}, {int(sizes[c])} "
                        "nodes)" for c in split[:10])
            + ("" if split.size <= 10 else ", ...")
            + f"; {int(copies.sum())} pieces in all, each holding "
            "sizes / copies of the nodes per box")
    notes += [f"graph: {note}" for note in graph.notes]
    return Components(
        label=_frozen(label), sizes=_frozen(sizes),
        dimensionality=_frozen(dimensionality), spans=_frozen(spans),
        translations=tuple(np.array(t, dtype=np.int64).reshape(-1, 3)
                           for t in translations),
        copies=_frozen(copies), node_cell=_frozen(cell),
        n_nodes=int(nodes.size), graph_definition=graph.definition,
        frame_digest=graph.frame_digest, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Warren-Cowley chemical short-range order
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class WarrenCowley:
    """alpha_ij = 1 - Z_ij / (Z_i c_j) over the graph's neighbour shell.

    Rows and columns follow ``elements``. ``concentration`` c_j over the
    nodes; ``partial_cn`` Z_ij, the mean number of j neighbours of an i node;
    ``total_cn`` Z_i; ``alpha`` NaN in the rows of an element with Z_i = 0;
    ``joinable`` whether the graph's definition can join i and j at all.
    """

    elements: tuple[str, ...]
    counts: np.ndarray
    concentration: np.ndarray
    partial_cn: np.ndarray
    total_cn: np.ndarray
    alpha: np.ndarray
    joinable: np.ndarray
    graph_definition: str
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[tuple[str, str], float]:
        return {(a, b): float(self.alpha[i, j])
                for i, a in enumerate(self.elements)
                for j, b in enumerate(self.elements)}

    def as_rows(self) -> list[dict]:
        rows = []
        for i, a in enumerate(self.elements):
            for j, b in enumerate(self.elements):
                rows.append({"descriptor": "Warren-Cowley alpha",
                             "centre": a, "neighbour": b,
                             "alpha": float(self.alpha[i, j]),
                             "Z_ij": float(self.partial_cn[i, j]),
                             "Z_i": float(self.total_cn[i]),
                             "c_j": float(self.concentration[j]),
                             "joinable": bool(self.joinable[i, j])})
        return rows


def warren_cowley(graph: NetworkGraph) -> WarrenCowley:
    """Warren-Cowley parameters of the graph's first neighbour shell."""
    graph = _check_graph(graph)
    symbols = graph.elements
    nodes = graph.nodes
    names = sorted({str(s) for s in symbols[nodes]})
    index = {s: k for k, s in enumerate(names)}
    m = len(names)
    code = np.full(graph.n_atoms, -1, dtype=np.int64)
    for s, k in index.items():
        code[(symbols == s) & graph.in_graph] = k
    counts = np.bincount(code[nodes], minlength=m).astype(np.int64)
    links = np.zeros((m, m), dtype=np.int64)
    np.add.at(links, (code[graph.u], code[graph.v]), 1)
    np.add.at(links, (code[graph.v], code[graph.u]), 1)
    partial = links / counts[:, None]
    total = partial.sum(axis=1)
    conc = counts / counts.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = 1.0 - partial / (total[:, None] * conc[None, :])
    alpha[total == 0, :] = np.nan
    joinable = np.array([[_pair_key(a, b) in graph.joinable for b in names]
                         for a in names], dtype=bool).reshape(m, m)
    notes = ["the neighbour shell is the graph's edges; c_j is the fraction "
             "of j among the nodes"]
    empty = [names[k] for k in range(m) if total[k] == 0]
    if empty:
        notes.append(f"{', '.join(empty)} nodes have no edge, so their alpha "
                     "is undefined (NaN)")
    never = sorted({_pair_key(names[i], names[j]) for i in range(m)
                    for j in range(m) if not joinable[i, j]})
    if never:
        notes.append("alpha = 1 by construction for pairs the graph's "
                     "definition never joins: "
                     + ", ".join(f"{a}-{b}" for a, b in never))
    notes += [f"graph: {note}" for note in graph.notes]
    return WarrenCowley(
        elements=tuple(names), counts=_frozen(counts),
        concentration=_frozen(conc), partial_cn=_frozen(partial),
        total_cn=_frozen(total), alpha=_frozen(alpha),
        joinable=_frozen(joinable), graph_definition=graph.definition,
        notes=tuple(notes))


# ---------------------------------------------------------------------------
# averages over frames
# ---------------------------------------------------------------------------

def _results(results, kind, what: str) -> list:
    if isinstance(results, (str, bytes, Mapping)):
        raise ValueError(f"{what}: one result per frame is needed")
    out = list(results)
    if not out:
        raise ValueError(f"{what}: no frame; there is nothing to average")
    for r in out:
        if not isinstance(r, kind):
            raise ValueError(f"{what}: a {kind.__name__} per frame is needed, "
                             f"not {type(r).__name__}")
    return out


def _definition_notes(results, what: str) -> list[str]:
    definitions = sorted({r.graph_definition for r in results})
    if len(definitions) == 1:
        return [f"graph: {definitions[0]}"]
    return [f"{what}: the frames' graphs were built with {len(definitions)} "
            "different definitions: " + " | ".join(definitions)]


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _carried_notes(results) -> list[str]:
    """The frames' own notes, each wording once.

    Notes that differ only in their numbers are one note: a number that is
    the same on every frame is kept, one that differs is given as its range
    over the frames ('75 to 80'), and a note not made on every frame says on
    how many it was."""
    groups: dict[tuple[str, int], list[list[str]]] = {}
    for r in results:
        seen: dict[str, int] = {}
        for note in r.notes:
            template = _NUMBER.sub("\0", note)
            occurrence = seen.get(template, 0)
            seen[template] = occurrence + 1
            groups.setdefault((template, occurrence), []).append(
                _NUMBER.findall(note))
    out = []
    for (template, _), rows in groups.items():
        parts = template.split("\0")
        text = parts[0]
        for slot, part in enumerate(parts[1:]):
            values = [row[slot] for row in rows]
            if len(set(values)) == 1:
                text += values[0]
            else:
                low = min(values, key=float)
                high = max(values, key=float)
                text += f"{low} to {high}"
            text += part
        if len(rows) < len(results):
            text += f" (in {len(rows)} of {len(results)} frames)"
        out.append(text)
    return out


def average_rings(results: Sequence[RingStatistics], *,
                  frames: Sequence[int] | None = None) -> dict:
    """R_C, R_N, P_N, P_max and P_min as md_stats Series over ring size, and
    the ring-size distribution (fractions of the distinct rings) as an
    md_stats Distribution; also by size in T atoms when every frame had the
    same ``t_elements``. Every frame needs the same criterion, max_size,
    graph source and node elements. Each result carries the frames' own
    notes (the search scope, the angles or bonds left unclosed, the graph's
    caveats), with numbers that differ between frames given as ranges."""
    what = "average_rings"
    results = _results(results, RingStatistics, what)
    first = results[0]
    for r in results[1:]:
        if (r.criterion, r.max_size, r.graph_source, r.node_elements) != \
                (first.criterion, first.max_size, first.graph_source,
                 first.node_elements):
            raise ValueError(f"{what}: the frames differ in criterion, "
                             "max_size, graph source or node elements")
    notes = _definition_notes(results, what) + _carried_notes(results)
    sizes_note = (f"the fractions are over the rings of at most "
                  f"{first.max_size} nodes; larger rings are not searched, so "
                  "the fractions depend on max_size")
    axis = first.sizes
    out: dict = {}
    for name, attr in (("R_C", "rc"), ("R_N", "rn"), ("P_N", "pn"),
                       ("P_max", "p_max"), ("P_min", "p_min")):
        out[name] = Series.from_frames(
            axis, [getattr(r, attr) for r in results],
            name=f"{name}(n), {first.criterion} rings", axis_name="n_ring_nodes",
            axis_unit="nodes", value_unit="1", frames=frames, notes=notes)
    keys = tuple(int(s) for s in axis)
    per_frame = [{int(s): int(c) for s, c in zip(r.sizes, r.ring_counts)}
                 for r in results]
    out["ring counts"] = Distribution.from_counts(
        per_frame, name=f"distinct rings by size (nodes), {first.criterion}",
        kind="count", keys=keys, frames=frames, notes=notes)
    if not any(r.n_rings for r in results):
        out["ring counts"] = out["ring counts"].with_notes(
            f"no frame holds a ring of at most {first.max_size} nodes, so the "
            "ring-size fractions are undefined and not given")
        return out
    out["ring sizes"] = Distribution.from_counts(
        per_frame, name=f"ring sizes (nodes), {first.criterion}",
        kind="fraction", keys=keys, frames=frames,
        notes=notes + [sizes_note])
    t_sets = {r.t_elements for r in results}
    if len(t_sets) == 1 and first.t_elements is not None:
        rows = []
        for r in results:
            values, counts = np.unique(r.t_sizes, return_counts=True)
            rows.append({int(t): int(c) for t, c in zip(values, counts)})
        out["ring sizes (T atoms)"] = Distribution.from_counts(
            rows, name=f"ring sizes (T atoms: "
                       f"{', '.join(sorted(first.t_elements))}), "
                       f"{first.criterion}",
            kind="fraction", frames=frames, notes=notes + [sizes_note])
    return out


def average_coordination_sequences(results: Sequence[CoordinationSequences], *,
                                   frames: Sequence[int] | None = None
                                   ) -> dict[str, Series]:
    """Element -> mean N_k per frame, averaged over frames (md_stats Series
    on the shell index k), with the frames' notes."""
    what = "average_coordination_sequences"
    results = _results(results, CoordinationSequences, what)
    shells = {r.n_shells for r in results}
    if len(shells) != 1:
        raise ValueError(f"{what}: the frames have different numbers of "
                         "shells")
    notes = _definition_notes(results, what) + _carried_notes(results)
    elements = sorted(set().union(*(set(r.mean_by_element()) for r in results)))
    out = {}
    for element in elements:
        rows = []
        for r in results:
            means = r.mean_by_element()
            if element not in means:
                raise ValueError(f"{what}: a frame has no {element} centre")
            rows.append(means[element])
        out[element] = Series.from_frames(
            results[0].shells, rows, name=f"coordination sequence, {element}",
            axis_name="k_shell", axis_unit="bonds", value_unit="nodes",
            frames=frames, notes=notes)
    return out


def average_polyhedral_connectivity(results: Sequence[PolyhedralConnectivity],
                                    *, frames: Sequence[int] | None = None
                                    ) -> dict[tuple[str, str], Distribution]:
    """Centre-pair type -> corner / edge / face fractions over frames, with
    the frames' notes (what 'edge' and 'face' count among them)."""
    what = "average_polyhedral_connectivity"
    results = _results(results, PolyhedralConnectivity, what)
    notes = _definition_notes(results, what) + _carried_notes(results)
    types = sorted(set().union(*(set(r.counts) for r in results)))
    out = {}
    for t in types:
        rows = [dict(r.counts.get(t, {})) for r in results]
        out[t] = Distribution.from_counts(
            rows, name=f"polyhedral sharing {t[0]}-{t[1]}", kind="fraction",
            keys=SHARING_KINDS, frames=frames, notes=notes)
    return out


def average_components(results: Sequence[Components], *,
                       frames: Sequence[int] | None = None) -> dict:
    """The largest piece's fraction of the nodes, the number of components
    and the number of interpenetrating pieces (md_stats Scalar), and the
    nodes by component dimensionality and by component size (md_stats
    Distribution, fractions), with the frames' notes."""
    what = "average_components"
    results = _results(results, Components, what)
    notes = _definition_notes(results, what) + _carried_notes(results)
    out = {
        "largest fraction": Scalar(
            "largest piece, fraction of nodes", "1",
            [r.largest_fraction for r in results], frames=frames, notes=notes),
        "components": Scalar(
            "number of components", "1",
            [float(r.n_components) for r in results], frames=frames,
            notes=notes),
        "pieces": Scalar(
            "number of interpenetrating pieces", "1",
            [float(r.n_pieces) for r in results], frames=frames, notes=notes),
        "nodes by dimensionality": Distribution.from_counts(
            [{int(d): int(r.sizes[r.dimensionality == d].sum())
              for d in range(4)} for r in results],
            name="nodes by component dimensionality", kind="fraction",
            keys=(0, 1, 2, 3), frames=frames, notes=notes),
        "nodes by size": Distribution.from_counts(
            [r.nodes_by_size() for r in results],
            name="nodes by component size", kind="fraction", frames=frames,
            notes=notes),
    }
    return out


def average_warren_cowley(results: Sequence[WarrenCowley], *,
                          frames: Sequence[int] | None = None
                          ) -> dict[tuple[str, str], Scalar]:
    """(centre, neighbour) -> alpha over frames (md_stats Scalar), with the
    frames' notes; a pair the graph's definition never joins (alpha = 1 by
    construction) says so in its own notes."""
    what = "average_warren_cowley"
    results = _results(results, WarrenCowley, what)
    if len({r.elements for r in results}) != 1:
        raise ValueError(f"{what}: the frames hold different node elements")
    notes = _definition_notes(results, what) + _carried_notes(results)
    names = results[0].elements
    out = {}
    for pair in results[0].as_dict():
        i, j = names.index(pair[0]), names.index(pair[1])
        never = sum(1 for r in results if not r.joinable[i, j])
        extra = []
        if never:
            extra.append(
                f"alpha = 1 by construction: the graph's definition never "
                f"joins {pair[0]} and {pair[1]}"
                + ("" if never == len(results) else
                   f" (in {never} of {len(results)} frames)"))
        out[pair] = Scalar(f"Warren-Cowley alpha {pair[0]}-{pair[1]}", "1",
                           [r.as_dict()[pair] for r in results], frames=frames,
                           notes=notes + extra)
    return out
