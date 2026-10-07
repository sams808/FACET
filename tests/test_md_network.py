"""Tests for facet/core/md_network.py: rings, coordination sequences,
polyhedral sharing, components and Warren-Cowley order.

What is pinned, and against what:

* **Closed forms and published sequences.** Diamond (built here from its
  definition, fcc plus a (1/4, 1/4, 1/4) basis) has only six-membered rings,
  12 through every atom, two chair hexagons on every angle, and the
  coordination sequence floor(5 k^2 / 2) + 2 (OEIS A008253). Simple cubic
  gives 4 k^2 + 2 (OEIS A005899) and the quartz Si net 4, 12, 30, 52, 80,
  116, 156, 204, 258, 318 (OEIS A008261, the same as RCSR qtz cs1-cs10).
* **Published vertex symbols.** RCSR (O'Keeffe, Peskov, Ramsden & Yaghi,
  Acc. Chem. Res. 41 (2008) 1782, data file rcsr.net/data/3dall.txt read
  2026-10-07) lists dia 6(2).6(2).6(2).6(2).6(2).6(2), pcu
  4.4.4.4.4.4.4.4.4.4.4.4.*.*.*, qtz 6.6.6(2).6(2).8(7).8(7) and gis
  4.4.4.8(2).8.8; vertex_symbol on primitive rings gives each of them
  (entries sorted, so gis reads 4.4.4.8.8.8(2)). King's shortest closed
  paths give 8_9 on qtz and 6_4 at the straight angles of pcu, and 6-rings
  on gis where the vertex symbol has 8-rings: those closed paths have
  shortcuts, so they are not the rings of a vertex symbol.
* **The R.I.N.G.S. manual's worked example.** Case a) of figure 5.22 of the
  I.S.A.A.C.S. manual (isaacs.sourceforge.io/manual/page37_ct.html), rebuilt
  from its description and numbers, gives the R_C, R_N, P_N, P_max and P_min
  of Tables 5.4, 5.7 and 5.8, ties counted as the manual counts them, but for
  the primitive P_max(4), which the manual prints as 0.5 and its own P_N(6)
  makes 0.
* **A second implementation inside the test.** On the simple cubic lattice
  the graph distance is the L1 norm, so the rings through one atom can be
  enumerated by a plain depth-first search over lattice points and checked
  for shortcuts with |x - y|_1, with no FACET code involved.
* **An outside implementation, run once.** The quartz numbers pinned here
  (King shortest closed paths 6_1.6_1.6_2.6_2.8_9.8_9, Guttman 6_3 x 4,
  primitive rings through a Si 6 of 6 and 40 of 8 members, none of 10 or 12)
  were also produced by ASE 3.29.0 + networkx 3.6.1 on an explicit 8x8x8
  supercell, in a throwaway environment outside FACET (neither is a
  dependency or imported here). The 6 six-membered rings per Si agree with
  RCSR's qtz vertex symbol; the 40 eight-membered primitive rings are a
  computed number only (no published value was found). The same script on a
  6x6x6 supercell reported 2 primitive 12-rings per Si: the a-axis chains
  close on themselves after 12 steps through six cells, a cycle that winds
  around that box. On the lift used here those are never rings.
* **The periodic boundary.** A one-cell diamond (8 atoms) and a one-atom
  simple cubic cell (an atom bonded to its own images) give, atom for atom,
  what a supercell gives; a cycle that winds once around a small box is not a
  ring; a real ring cut by the box face is counted once; cuprite's two
  interpenetrating nets are two pieces in the 6-atom cell and in a supercell.
* **What a bounded search leaves open.** Angles and bonds with no ring of at
  most max_size are split into those shown to lie on no cycle (a dangling
  neighbour, a finite piece) and those that may lie on a larger ring; the
  notes say that P_max and the ring-size fractions depend on max_size, and
  the frame averages carry every frame's notes.
* **Polyhedral sharing.** Quartz: corners only. A rutile-type SnO2 built here
  (bond-valence bonds): edge 1/5, corner 4/5, two edge and eight corner
  partners per Sn. A confacial pair: one face. A centre's edges to atoms
  outside the ligands, and the bonds a bridged graph does not use, are
  counted in the notes.
* **Bookkeeping.** Fractions pass check_fractions, Warren-Cowley obeys
  sum_j c_j alpha_ij = 0, two identical frames give a spread of exactly 0,
  permuting or translating the atoms changes no count, every refusal names
  its reason, nothing has a default physical parameter, and no message
  carries a verdict word. What the pair search or the valence table left
  out (pairs below d_min) reaches the graph's notes.
* **Memory and progress.** The primitive search holds under a tenth of its
  distinct candidate cycles at once (216-atom diamond), as packed keys that
  round-trip; a max_size far above the rings stores nothing per extra size;
  a bond with a dangling end is not searched; progress() is also called on
  a clock, not only at every 1 %.
"""
from __future__ import annotations

import ast
import itertools
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from facet.core import bulk, md_model, md_network as mn, readers
from facet.core.md_stats import check_fractions

ROOT = Path(__file__).resolve().parent.parent
QUARTZ = Path(__file__).resolve().parent / "data" / "crystals" / \
    "quartz_SiO2_cod9013321.cif"

FCC = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
DIAMOND = FCC + [tuple(np.add(p, 0.25)) for p in FCC]


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _lattice(basis, reps, edge_ang, symbols):
    """A cubic cell of ``edge_ang`` with ``basis`` repeated ``reps`` times."""
    cells = np.indices(reps).reshape(3, -1).T
    basis = np.asarray(basis, dtype=float)
    frac = ((cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)
            / np.asarray(reps, dtype=float))
    if isinstance(symbols, str):
        symbols = [symbols] * len(basis)
    names = list(symbols) * len(cells)
    box = np.diag(np.asarray(reps, dtype=float) * edge_ang)
    return md_model.frame_from_arrays(names, frac=frac, box_ang=box)


def _first_gap_cutoff(frame, pairs, centre, neighbour, rank):
    """Midway between the rank-th and (rank+1)-th distinct centre-neighbour
    distance seen from the first centre atom: a cutoff measured from the
    model, never typed in."""
    first = int(np.flatnonzero(frame.elements == centre)[0])
    sel = (pairs.i == first) & (frame.elements[pairs.j] == neighbour)
    d = np.sort(pairs.d_ang[sel])
    assert d[rank] - d[rank - 1] > 0.1, "no gap at the stated shell"
    return 0.5 * (d[rank - 1] + d[rank])


def _distance_graph(frame, r_ang, symbol, rank):
    pairs = bulk.find_pairs(frame, r_ang)
    cut = _first_gap_cutoff(frame, pairs, symbol, symbol, rank)
    return mn.graph_from_pairs(
        frame, pairs, {(symbol, symbol): cut},
        {(symbol, symbol): f"test: midway between neighbour {rank} and "
                           f"{rank + 1} of the ideal lattice"})


def _diamond(reps):
    return _distance_graph(_lattice(DIAMOND, reps, 4.0, "C"), 3.0, "C", 4)


def _cubic(reps):
    return _distance_graph(_lattice([(0, 0, 0)], reps, 2.0, "C"), 3.5, "C", 6)


def _quartz():
    structure = readers.read(QUARTZ)
    frame, _ = md_model.supercell_frame(structure, (2, 2, 2))
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox)
    return frame, table


def _rutile(reps=(2, 2, 3)):
    """Rutile type, P4_2/mnm: Sn at 2a, O at 4f (u, u, 0). a, c and u are
    illustrative; the test checks that each Sn has exactly six O above v_bond
    rather than relying on them."""
    u = 0.307
    basis = [(0, 0, 0), (0.5, 0.5, 0.5), (u, u, 0), (1 - u, 1 - u, 0),
             (0.5 + u, 0.5 - u, 0.5), (0.5 - u, 0.5 + u, 0.5)]
    symbols = ["Sn", "Sn", "O", "O", "O", "O"]
    cells = np.indices(reps).reshape(3, -1).T
    frac = ((cells[:, None, :] + np.asarray(basis)[None]).reshape(-1, 3)
            / np.asarray(reps, dtype=float))
    box = np.diag(np.array(reps, dtype=float) * np.array([4.74, 4.74, 3.19]))
    frame = md_model.frame_from_arrays(symbols * len(cells), frac=frac,
                                       box_ang=box)
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, results = bulk.analyse_frame(frame, ox)
    return frame, table, results


# ---------------------------------------------------------------------------
# diamond: only 6-rings, 12 per atom, CS A008253
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reps", [(1, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("criterion", mn.CRITERIA)
def test_diamond_has_only_six_rings_twelve_through_each_atom(criterion, reps):
    """Every criterion; the 8-atom cell (rings through the box faces only)
    and the 64-atom supercell give the same per-atom numbers."""
    graph = _diamond(reps)
    assert set(graph.degree.tolist()) == {4}
    stats = mn.ring_statistics(graph, criterion, 12)
    nonzero = {int(s): int(c) for s, c in zip(stats.sizes, stats.ring_counts)
               if c}
    assert nonzero == {6: 2 * graph.n_nodes}            # N * 12 / 6 rings
    six = int(np.flatnonzero(stats.sizes == 6)[0])
    assert (stats.found[:, six] == 12).all()
    assert stats.found.sum() == 12 * graph.n_nodes
    assert stats.rc[six] == 2.0 and stats.pn[six] == 1.0
    assert stats.p_max[six] == 1.0 and stats.p_min[six] == 1.0
    assert all(set(ring.cells) and ring.size == 6 for ring in stats.rings)
    if criterion == "king":
        assert {stats.shortest_cycle_symbol(a) for a in graph.nodes} == \
            {".".join(["6_2"] * 6)}
    if criterion == "guttman":                       # every bond in 6 rings
        assert {stats.shortest_cycle_symbol(a) for a in graph.nodes} == \
            {".".join(["6_6"] * 4)}
    if criterion == "primitive":                     # RCSR dia
        assert {stats.vertex_symbol(a) for a in graph.nodes} == \
            {"6(2).6(2).6(2).6(2).6(2).6(2)"}


def test_diamond_coordination_sequence_is_oeis_a008253():
    """floor(5 k^2 / 2) + 2, from the 8-atom cell, eight shells (the lift
    reaches cells far outside the box)."""
    cs = mn.coordination_sequences(_diamond((1, 1, 1)), 8)
    want = [5 * k * k // 2 + 2 for k in range(1, 9)]
    assert want[:6] == [4, 12, 24, 42, 64, 92]
    assert (cs.counts == np.array(want)).all()
    assert (cs.cumulative[:, -1] == 1 + sum(want)).all()


# ---------------------------------------------------------------------------
# simple cubic: 4-rings, 12 per atom; CS 4k^2 + 2; an L1 oracle
# ---------------------------------------------------------------------------

def _l1_rings_through_origin(max_size):
    """Every simple cycle of the cubic lattice graph through the origin with
    at most max_size members, as frozensets, by depth-first search; with the
    ones that have no shortcut (graph distance = L1 norm). Independent of
    FACET."""
    steps = [s for s in itertools.product((-1, 0, 1), repeat=3)
             if sum(map(abs, s)) == 1]
    origin = (0, 0, 0)
    found = set()

    def walk(path):
        last = path[-1]
        for s in steps:
            nxt = (last[0] + s[0], last[1] + s[1], last[2] + s[2])
            if nxt == origin and len(path) >= 3:
                found.add(tuple(path))
            elif nxt not in path and len(path) < max_size and \
                    sum(map(abs, nxt)) <= max_size // 2:
                walk(path + [nxt])
    walk([origin])
    rings = {}
    for cyc in found:
        key = frozenset(cyc)
        n = len(cyc)
        isometric = all(sum(abs(a - b) for a, b in zip(cyc[p], cyc[q]))
                        == min(q - p, n - q + p)
                        for p in range(n) for q in range(p + 1, n))
        rings[key] = (n, isometric)
    return rings


def test_simple_cubic_rings_against_an_l1_oracle():
    """Through one atom: 12 squares under every criterion. Guttman finds
    nothing else; King adds, for each of the 3 straight angles, the four
    2x1 rectangles that are the shortest cycles through it (4_1 x 12, 6_4 x
    3); primitive adds the 4 skew hexagons of each of the 8 cubes at the
    atom with that atom on them, 24 in all. The oracle enumerates the cycles
    on the lattice itself. A rectangle has a shortcut, so no ring passes
    through a straight angle: the vertex symbol is RCSR's pcu, 4 x 12 and
    '*' x 3, and those three angles are open (on cycles, not shown
    acyclic)."""
    oracle = _l1_rings_through_origin(6)
    primitive = {}
    for n, iso in oracle.values():
        if iso:
            primitive[n] = primitive.get(n, 0) + 1
    assert primitive == {4: 12, 6: 24}
    rectangles = sum(1 for n, iso in oracle.values() if n == 6 and not iso)
    assert rectangles > 0
    graph = _cubic((3, 3, 3))
    want = {"guttman": {4: 12}, "king": {4: 12, 6: 12},
            "primitive": {4: 12, 6: 24}}
    for criterion, per_atom in want.items():
        stats = mn.ring_statistics(graph, criterion, 6)
        for row in stats.found:
            got = {int(s): int(c) for s, c in zip(stats.sizes, row) if c}
            assert got == per_atom, criterion
    king = mn.ring_statistics(graph, "king", 6)
    assert king.shortest_cycle_symbol(0) == \
        ".".join(["4_1"] * 12 + ["6_4"] * 3)
    four, six = 1, 3                                  # sizes 3, 4, 5, 6
    assert king.rc[four] == 3.0 and king.rc[six] == 6.0
    assert king.p_max[four] == 0.0 and king.p_min[four] == 1.0
    assert king.p_max[six] == 1.0 and king.p_min[six] == 0.0
    prim = mn.ring_statistics(graph, "primitive", 6)
    assert prim.rc[four] == 3.0 and prim.rc[six] == 4.0
    assert {prim.vertex_symbol(a) for a in graph.nodes} == \
        {".".join(["4"] * 12 + ["*"] * 3)}
    assert prim.n_bases == 15 * graph.n_nodes
    assert prim.n_bases_unclosed == 3 * graph.n_nodes
    assert prim.n_bases_acyclic == 0


def test_simple_cubic_one_atom_cell_matches_the_supercell():
    """One atom bonded to its own six images: three self-loop edges."""
    one, many = _cubic((1, 1, 1)), _cubic((3, 3, 3))
    assert one.n_edges == 3 and (one.u == one.v).all()
    for criterion in mn.CRITERIA:
        a = mn.ring_statistics(one, criterion, 6)
        b = mn.ring_statistics(many, criterion, 6)
        assert np.array_equal(a.found[0], b.found[0])
        assert np.array_equal(a.rc, b.rc) and np.array_equal(a.pn, b.pn)
    cs = mn.coordination_sequences(one, 8)
    assert cs.counts[0].tolist() == [4 * k * k + 2 for k in range(1, 9)]


# ---------------------------------------------------------------------------
# quartz: CS A008261, King/Guttman/primitive as cross-checked
# ---------------------------------------------------------------------------

def test_quartz_si_net_coordination_sequence_is_oeis_a008261():
    """Ten shells: OEIS A008261 a(1)..a(10), also RCSR qtz cs1-cs10."""
    frame, table = _quartz()
    graph = mn.bridged_graph(frame, table, formers={"Si"}, anions={"O"})
    assert set(graph.degree[graph.nodes].tolist()) == {4}
    assert (graph.multiplicity == 1).all()
    cs = mn.coordination_sequences(graph, 10)
    assert {tuple(r) for r in cs.counts.tolist()} == \
        {(4, 12, 30, 52, 80, 116, 156, 204, 258, 318)}


def test_quartz_rings_match_the_outside_implementation():
    """King's shortest closed paths at the 8-ring angles number 9, but 2 of
    them have a shortcut (graph distance 2 between members 4 apart on the
    path), so the vertex symbol, which counts rings only, has 8(7) there:
    RCSR qtz 6.6.6(2).6(2).8(7).8(7)."""
    frame, table = _quartz()
    graph = mn.bridged_graph(frame, table, formers={"Si"}, anions={"O"})
    king = mn.ring_statistics(graph, "king", 12)
    gutt = mn.ring_statistics(graph, "guttman", 12)
    prim = mn.ring_statistics(graph, "primitive", 12)
    assert {king.shortest_cycle_symbol(a) for a in graph.nodes} == \
        {"6_1.6_1.6_2.6_2.8_9.8_9"}
    assert {gutt.shortest_cycle_symbol(a) for a in graph.nodes} == \
        {".".join(["6_3"] * 4)}
    assert {prim.vertex_symbol(a) for a in graph.nodes} == \
        {"6.6.6(2).6(2).8(7).8(7)"}
    for stats in (king, gutt):
        with pytest.raises(ValueError, match="cycles without a shortcut"):
            stats.vertex_symbol(int(graph.nodes[0]))
    with pytest.raises(ValueError, match="vertex_symbol gives"):
        prim.shortest_cycle_symbol(int(graph.nodes[0]))

    def per_atom(stats):
        return {tuple((int(s), int(c)) for s, c in zip(stats.sizes, row) if c)
                for row in stats.found}
    assert per_atom(king) == {((6, 6), (8, 18))}
    assert per_atom(gutt) == {((6, 6),)}
    assert per_atom(prim) == {((6, 6), (8, 40))}


def _gis(scale=2.0):
    """The gis net from its RCSR entry: I4_1/amd (origin choice 2), a =
    3.334, c = 2.981 (edge length 1, scaled here to 2 Å), one vertex orbit
    16g at (0.15, 0.40, 0.875), expanded with gemmi's symmetry operations."""
    import gemmi
    group = gemmi.find_spacegroup_by_name("I 41/a m d:2")
    sites = set()
    for op in group.operations():
        p = np.mod(op.apply_to_xyz([0.15, 0.40, 0.875]), 1.0)
        sites.add(tuple(np.round(p, 6) % 1.0))
    frac = np.array(sorted(sites))
    assert frac.shape == (16, 3)
    box = np.diag([3.334, 3.334, 2.981]) * scale
    frame = md_model.frame_from_arrays(["C"] * 16, frac=frac, box_ang=box)
    return _distance_graph(frame, 3.5, "C", 4)


def test_gis_vertex_symbol_is_rcsrs_and_differs_from_king_in_ring_size():
    """RCSR gis: vertex symbol 4.4.4.8(2).8.8 and cs1-cs10 4, 9, 18, 32,
    48, 67, 92, 120, 150, 185. King's shortest closed paths through two of
    the angles are 6-membered with a shortcut, so there the smallest ring is
    an 8-ring: the entry changes size, not only count; at a third angle 2 of
    the 3 shortest 8-membered paths are rings."""
    graph = _gis()
    assert set(graph.degree.tolist()) == {4}
    cs = mn.coordination_sequences(graph, 10)
    assert {tuple(r) for r in cs.counts.tolist()} == \
        {(4, 9, 18, 32, 48, 67, 92, 120, 150, 185)}
    prim = mn.ring_statistics(graph, "primitive", 10)
    assert {prim.vertex_symbol(a) for a in graph.nodes} == {"4.4.4.8.8.8(2)"}
    king = mn.ring_statistics(graph, "king", 10)
    assert {king.shortest_cycle_symbol(a) for a in graph.nodes} == \
        {"4_1.4_1.4_1.6_1.6_1.8_3"}


def test_quartz_all_atom_rings_are_the_t_rings_doubled():
    """On the Si + O bond graph every ring has twice the members and the
    same number of Si; the distinct rings are the same in number."""
    frame, table = _quartz()
    atoms = mn.graph_from_bonds(frame, table)
    tnet = mn.bridged_graph(frame, table, formers={"Si"}, anions={"O"})
    for criterion in mn.CRITERIA:
        a = mn.ring_statistics(atoms, criterion, 24, t_elements={"Si"})
        t = mn.ring_statistics(tnet, criterion, 12)
        want = {(2 * int(s), int(s)): int(c)
                for s, c in zip(t.sizes, t.ring_counts) if c}
        assert a.t_size_counts() == want, criterion
        rows = a.as_rows()
        assert {r["ring size (T atoms)"] for r in rows if r["rings"]} == \
            {str(s) for (_, s) in want}


# ---------------------------------------------------------------------------
# the R.I.N.G.S. manual's worked examples (I.S.A.A.C.S. manual, pages 36-37)
# ---------------------------------------------------------------------------

def _abstract_graph(n_nodes, edges, definition):
    """A graph given by its edge list alone, as the manual's figures are:
    every edge inside cell (0, 0, 0), no positions."""
    pairs = sorted((min(a, b), max(a, b)) for a, b in edges)
    u = np.array([a for a, _ in pairs], dtype=np.int64)
    v = np.array([b for _, b in pairs], dtype=np.int64)
    return mn.NetworkGraph(
        elements=np.array(["C"] * n_nodes), in_graph=np.ones(n_nodes, bool),
        u=u, v=v, shift=np.zeros((len(pairs), 3), dtype=np.int64),
        d_ang=np.ones(len(pairs)),
        multiplicity=np.ones(len(pairs), dtype=np.int64), source="distance",
        definition=definition, node_elements=frozenset({"C"}),
        joinable=frozenset({("C", "C")}), v_bond_vu=None, cutoffs_ang={},
        cutoff_sources={}, frame_digest="")


def test_rings_manual_table_5_0():
    """Figure 5.19 / Table 5.0 (page36_ct.html): 10 nodes, 7 bonds, one
    triangle and one square; R_C(3) = R_C(4) = 1/10, R_N(3) = 3/10,
    R_N(4) = 4/10, under every criterion."""
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 3)]
    graph = _abstract_graph(10, edges, "R.I.N.G.S. manual figure 5.19")
    for criterion in mn.CRITERIA:
        stats = mn.ring_statistics(graph, criterion, 6)
        assert stats.rc[:2].tolist() == pytest.approx([1 / 10, 1 / 10])
        assert stats.rn[:2].tolist() == pytest.approx([3 / 10, 4 / 10])
        assert stats.n_nodes_without_ring == 3


def test_a_king_ring_may_pass_through_another_neighbour_of_its_node():
    """Node 0 bonded to 1, 2 and 3, with 1-3 and 3-2 bonded too. The
    shortest path from 1 to 2 that avoids node 0 is 1-3-2, through node 0's
    third neighbour: King's ring of that angle is the square 0-1-3-2, with
    the chord 0-3. Only the node itself is barred, as the R.I.N.G.S.
    wording says; vitrum 1.1.0 also bars the node's other neighbours, and
    would find no ring at that angle here (module docstring)."""
    graph = _abstract_graph(4, [(0, 1), (0, 2), (0, 3), (1, 3), (3, 2)],
                            "a node with a chorded ring")
    king = mn.ring_statistics(graph, "king", 6)
    assert king.shortest_cycle_symbol(0) == "3_1.3_1.4_1"
    prim = mn.ring_statistics(graph, "primitive", 6)
    assert prim.vertex_symbol(0) == "3.3.*"           # the square has a shortcut


def test_rings_manual_figure_5_22_case_a_counts_ties():
    """Case a) of figure 5.22 (page37_ct.html): two hexagons sharing the
    path a-b-c-d-e, closed by x and by y, which also close the square
    a-x-e-y; 16 nodes, 9 of them isolated. Tables 5.4, 5.7 and 5.8. Under
    King's and Guttman's criteria R_N(6) = 10/16 needs both tied shortest
    paths at b, c and d (one per angle or bond would give 7/16), so the
    manual counts ties as this module does. The primitive P_max(4) is
    printed 0.5 in Table 5.8, but the same table's P_N(6) = 7/16 puts a, e,
    x and y (the square) on a hexagon too, so no node has 4 as its largest
    size and P_max(4) = 0; the printed 0.5 equals the King entry."""
    a, b, c, d, e, x, y = range(7)
    edges = [(a, b), (b, c), (c, d), (d, e), (a, x), (x, e), (a, y), (y, e)]
    graph = _abstract_graph(16, edges, "R.I.N.G.S. manual figure 5.22 a)")
    published = {   # n: (R_C, R_N, P_N, P_max, P_min)
        "shortest": {4: (1 / 16, 4 / 16, 4 / 16, 0.5, 1.0),
                     6: (2 / 16, 10 / 16, 5 / 16, 1.0, 0.6)},
        "primitive": {4: (1 / 16, 4 / 16, 4 / 16, 0.0, 1.0),
                      6: (2 / 16, 12 / 16, 7 / 16, 1.0, 3 / 7)}}
    for criterion in mn.CRITERIA:
        stats = mn.ring_statistics(graph, criterion, 8)
        want = published["primitive" if criterion == "primitive"
                         else "shortest"]
        assert {int(s): int(n) for s, n in zip(stats.sizes, stats.ring_counts)
                if n} == {4: 1, 6: 2}, criterion
        for size, values in want.items():
            k = size - 3
            got = (stats.rc[k], stats.rn[k], stats.pn[k], stats.p_max[k],
                   stats.p_min[k])
            assert got == pytest.approx(values, abs=1e-12), (criterion, size)
    king = mn.ring_statistics(graph, "king", 8)
    assert king.shortest_cycle_symbol(c) == "6_2"          # the tie at c


# ---------------------------------------------------------------------------
# single ring, winding cycle, chain
# ---------------------------------------------------------------------------

def _hexagon_frame(offset_frac=(0.0, 0.0, 0.0)):
    angles = np.arange(6) * np.pi / 3
    ring = np.column_stack([1.5 * np.cos(angles), 1.5 * np.sin(angles),
                            np.zeros(6)])
    box = np.eye(3) * 20.0
    frac = ring / 20.0 + np.asarray(offset_frac)
    return md_model.frame_from_arrays(["C"] * 6, frac=frac, box_ang=box)


@pytest.mark.parametrize("offset", [(0.5, 0.5, 0.5), (0.0, 0.02, 0.0)])
def test_a_single_ring_is_one_ring_wherever_the_box_cuts_it(offset):
    """The second offset puts the hexagon across the box corner: three of its
    atoms are wrapped to the far faces."""
    frame = _hexagon_frame(offset)
    graph = _distance_graph(frame, 4.0, "C", 2)
    for criterion in mn.CRITERIA:
        stats = mn.ring_statistics(graph, criterion, 10)
        assert stats.n_rings == 1 and stats.rings[0].size == 6
        six = 3
        assert stats.rc[six] == pytest.approx(1 / 6)
        assert (stats.found[:, six] == 1).all() and stats.pn[six] == 1.0
        loop = stats.rings[0].positions_ang(frame)
        sides = np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1)
        assert np.allclose(sides, 1.5)
    cs = mn.coordination_sequences(graph, 5)
    assert {tuple(r) for r in cs.counts.tolist()} == {(2, 2, 1, 0, 0)}
    comp = mn.components(graph)
    assert comp.sizes.tolist() == [6] and comp.dimensionality.tolist() == [0]
    whole = comp.unwrapped_cart_ang(frame)
    assert np.ptp(whole, axis=0).max() <= 3.0 + 1e-9


def test_a_cycle_winding_around_the_box_is_not_a_ring():
    """Three atoms 2 Å apart along x in a 6 Å box: in the box they close a
    triangle of bonds, but the closing bond goes to the next cell, so the
    walk returns to a copy one box length away. A one-atom chain bonded to
    its own images is the same in a smaller box."""
    box = np.diag([6.0, 20.0, 20.0])
    frac = np.array([[0.0, 0.5, 0.5], [1 / 3, 0.5, 0.5], [2 / 3, 0.5, 0.5]])
    frame = md_model.frame_from_arrays(["C"] * 3, frac=frac, box_ang=box)
    graph = _distance_graph(frame, 4.5, "C", 2)
    assert graph.n_edges == 3 and set(graph.degree.tolist()) == {2}
    for criterion in mn.CRITERIA:
        stats = mn.ring_statistics(graph, criterion, 12)
        assert stats.n_rings == 0
        assert stats.n_nodes_without_ring == 3
        assert stats.n_bases_unclosed == stats.n_bases > 0
        # an endless chain: no search of bounded depth runs out of vertices,
        # so nothing is shown acyclic and the note says rings may be larger
        assert stats.n_bases_acyclic == 0
        assert any("may lie on a larger" in note for note in stats.notes)
        symbol = (stats.vertex_symbol(0) if criterion == "primitive"
                  else stats.shortest_cycle_symbol(0))
        assert set(symbol.split(".")) == {"*"}
    comp = mn.components(graph)
    assert comp.dimensionality.tolist() == [1]
    assert comp.spans.tolist() == [[True, False, False]]
    assert comp.translations[0].tolist() in ([[1, 0, 0]], [[-1, 0, 0]])
    single = md_model.frame_from_arrays(["C"], frac=[[0.0, 0.5, 0.5]],
                                        box_ang=np.diag([2.0, 20.0, 20.0]))
    pairs = bulk.find_pairs(single, 2.5)
    one = mn.graph_from_pairs(single, pairs, {("C", "C"): 2.2},
                              {("C", "C"): "test: between 2.0 and 4.0 Å, the "
                                           "only two distances of the chain"})
    assert one.n_edges == 1 and one.u[0] == one.v[0]
    for criterion in mn.CRITERIA:
        assert mn.ring_statistics(one, criterion, 12).n_rings == 0


def test_an_open_chain_has_no_ring_and_a_finite_component():
    box = np.eye(3) * 30.0
    cart = np.column_stack([np.arange(5) * 1.5 + 5.0, np.full(5, 5.0),
                            np.full(5, 5.0)])
    frame = md_model.frame_from_arrays(["C"] * 5, cart, box_ang=box)
    graph = _distance_graph(frame, 4.0, "C", 1)
    for criterion in mn.CRITERIA:
        stats = mn.ring_statistics(graph, criterion, 8)
        assert stats.n_rings == 0
        # every angle and bond of a finite chain is shown to close nothing
        assert stats.n_bases_acyclic == stats.n_bases_unclosed == \
            stats.n_bases > 0
        assert any(f"{stats.n_bases} of them on no cycle" in note
                   for note in stats.notes)
    cs = mn.coordination_sequences(graph, 4)
    assert cs.counts.tolist() == [[1, 1, 1, 1], [2, 1, 1, 0], [2, 2, 0, 0],
                                  [2, 1, 1, 0], [1, 1, 1, 1]]
    comp = mn.components(graph)
    assert comp.sizes.tolist() == [5] and comp.dimensionality.tolist() == [0]
    assert not comp.spans.any()


def _hexagon_with_pendant():
    """A 1.5 Å hexagon with one more atom 1.5 Å out from vertex 0 (row 6),
    like a non-bridging O on a ring: the next distance is 2.6 Å."""
    frame = _hexagon_frame((0.5, 0.5, 0.5))
    cart = np.vstack([frame.cart_ang, frame.cart_ang[0]
                      + 1.5 * (frame.cart_ang[0] - frame.cart_ang.mean(axis=0))
                      / 1.5])
    frame = md_model.frame_from_arrays(["C"] * 7, cart, box_ang=frame.box_ang)
    pairs = bulk.find_pairs(frame, 4.0)
    assert np.isclose(np.sort(np.unique(np.round(pairs.d_ang, 6)))[:2],
                      [1.5, 2.598076]).all()
    return mn.graph_from_pairs(frame, pairs, {("C", "C"): 2.0},
                               {("C", "C"): "test: between 1.5 Å, the bonds, "
                                            "and 2.6 Å, the next distance"})


def test_a_dangling_atom_closes_nothing_and_is_not_searched(monkeypatch):
    """The two angles at vertex 0 that hold the pendant atom, and its bond,
    lie on no cycle: they are counted as unclosed and shown acyclic, apart
    from what a larger max_size might close. With max_size 5 the hexagon's
    own angles are unclosed too, but not acyclic. A Guttman bond with a
    dangling end is decided without a search (a search from its other end
    would run to the full depth; it did, once per such bond, before)."""
    graph = _hexagon_with_pendant()
    king = mn.ring_statistics(graph, "king", 8)
    assert king.n_bases == 6 + 2 and king.n_bases_unclosed == 2
    assert king.n_bases_acyclic == 2
    assert king.shortest_cycle_symbol(0) == "6_1.*.*"
    gutt = mn.ring_statistics(graph, "guttman", 8)
    assert gutt.n_bases == 2 * 7
    assert gutt.n_bases_unclosed == gutt.n_bases_acyclic == 2
    prim = mn.ring_statistics(graph, "primitive", 8)
    assert prim.vertex_symbol(0) == "6.*.*"
    assert prim.n_bases_unclosed == prim.n_bases_acyclic == 2
    small = mn.ring_statistics(graph, "king", 5)
    assert small.n_bases_unclosed == 8 and small.n_bases_acyclic == 2
    assert any("6 may lie on a larger ring" in note for note in small.notes)

    calls = []
    real = mn._bfs

    def counting(*args, **kwargs):
        calls.append(args[2])
        return real(*args, **kwargs)
    monkeypatch.setattr(mn, "_bfs", counting)
    mn.ring_statistics(graph, "guttman", 8)
    assert len(calls) == 6                 # the six hexagon bonds only


def test_a_tricluster_closes_a_t_triangle_only_on_the_bridged_graph():
    """One O bonded to three Si (a star of bonds): no ring in the atom
    graph, but the bridged graph links each pair of the three Si, a
    three-membered ring the notes account for. On the last frame of the
    3000-atom SiO2 model used for the timings the same difference was
    80 - 54 = 26 three-rings, its 26 triclusters (25 on its first frame)."""
    angles = np.radians([90, 210, 330])
    cart = np.array([[0.0, 0.0, 0.0]] + [[1.62 * np.cos(t), 1.62 * np.sin(t),
                                          0.0] for t in angles]) + 10.0
    frame = md_model.frame_from_arrays(["O", "Si", "Si", "Si"], cart,
                                       box_ang=np.eye(3) * 20.0)
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, results = bulk.analyse_frame(frame, ox)
    assert results.cn.tolist() == [3, 1, 1, 1]
    atoms = mn.graph_from_bonds(frame, table)
    tnet = mn.bridged_graph(frame, table, formers={"Si"}, anions={"O"})
    assert atoms.n_edges == 3 and tnet.n_edges == 3
    assert any("1 anion(s) are bonded to three or more formers" in n
               and "three-membered ring" in n for n in tnet.notes)
    for criterion in mn.CRITERIA:
        assert mn.ring_statistics(atoms, criterion, 8).n_rings == 0
        tri = mn.ring_statistics(tnet, criterion, 8)
        assert [r.size for r in tri.rings] == [3]


# ---------------------------------------------------------------------------
# polyhedral connectivity
# ---------------------------------------------------------------------------

def test_quartz_tetrahedra_share_corners_only():
    frame, table = _quartz()
    graph = mn.graph_from_bonds(frame, table)
    pc = mn.polyhedral_connectivity(graph, centres={"Si"}, ligands={"O"})
    n_si = int((frame.elements == "Si").sum())
    assert pc.counts == {("Si", "Si"): {"corner": 2 * n_si, "edge": 0,
                                        "face": 0}}
    assert pc.fractions[("Si", "Si")] == {"corner": 1.0, "edge": 0.0,
                                          "face": 0.0}
    rows = pc.per_centre[frame.elements == "Si"]
    assert (rows == [4, 0, 0]).all()
    assert (pc.per_centre[frame.elements == "O"] == -1).all()
    assert pc.isolated == {"Si": 0}


def test_polyhedral_connectivity_notes_centre_edges_to_non_ligands():
    """A distance graph that also joins each Si to its four Si neighbours
    (3.06 Å in quartz; the next Si is at 4.37 Å): those 48 Si-Si edges are
    not polyhedron vertices, so the sharing stays the corner sharing of the
    bond graph, and the notes count the edges left out. Before, they were
    left out without a word."""
    frame, table = _quartz()
    pairs = bulk.find_pairs(frame, 5.0)
    cutoffs = {("Si", "O"): _first_gap_cutoff(frame, pairs, "Si", "O", 4),
               ("Si", "Si"): _first_gap_cutoff(frame, pairs, "Si", "Si", 4)}
    sources = {("Si", "O"): "test: between the 4th and 5th Si-O distance",
               ("Si", "Si"): "test: between the 4th and 5th Si-Si distance"}
    graph = mn.graph_from_pairs(frame, pairs, cutoffs, sources)
    n_si = int((frame.elements == "Si").sum())
    pc = mn.polyhedral_connectivity(graph, centres={"Si"}, ligands={"O"})
    assert pc.counts == {("Si", "Si"): {"corner": 2 * n_si, "edge": 0,
                                        "face": 0}}
    assert (f"{2 * n_si} edge(s) of the graph join a centre to an atom "
            "outside the ligands, and that atom is not a vertex of its "
            f"polyhedron: Si-Si {2 * n_si}") in pc.notes
    plain = mn.polyhedral_connectivity(mn.graph_from_bonds(frame, table),
                                       centres={"Si"}, ligands={"O"})
    assert not any("outside the ligands" in note for note in plain.notes)


def test_bridged_graph_notes_the_bonds_it_does_not_use():
    """A former's bonds to anions outside ``anions`` link nothing, and a
    former given with a negative oxidation state (an anion given with a
    positive one) is never the cation (anion) end of a bond: each is counted
    in the notes rather than giving an edgeless graph without a word.
    Quartz 2x2x2: 24 Si x 4 = 96 Si-O bonds."""
    frame, table = _quartz()
    unused = mn.bridged_graph(frame, table, formers={"Si"}, anions={"F"})
    assert unused.n_edges == 0
    assert ("96 bond(s) from formers to anions outside 'anions' link no "
            "formers: Si-O 96") in unused.notes
    flipped = mn.bridged_graph(frame, table, formers={"O"}, anions={"Si"})
    assert flipped.n_edges == 0
    assert any(note.startswith("formers O are anions (negative oxidation")
               for note in flipped.notes)
    assert any(note.startswith("anions Si are cations (oxidation state of 0")
               for note in flipped.notes)
    plain = mn.bridged_graph(frame, table, formers={"Si"}, anions={"O"})
    assert plain.notes == ()


def test_rutile_type_octahedra_share_two_edges_and_eight_corners():
    """Each SnO6 octahedron shares an edge with the two along c and a corner
    with eight in the neighbouring chains: 1/5 of the sharing pairs are
    edges. The bridged graph sees each edge-sharing pair as one link of
    multiplicity 2, and every O bonded to three Sn."""
    frame, table, results = _rutile()
    sn = frame.elements == "Sn"
    assert (results.cn[sn] == 6).all() and (results.cn[~sn] == 3).all()
    graph = mn.graph_from_bonds(frame, table)
    pc = mn.polyhedral_connectivity(graph, centres={"Sn"}, ligands={"O"})
    n_sn = int(sn.sum())
    assert pc.counts[("Sn", "Sn")] == {"corner": 4 * n_sn, "edge": n_sn,
                                       "face": 0}
    assert pc.fractions[("Sn", "Sn")]["edge"] == pytest.approx(0.2)
    assert (pc.per_centre[sn] == [8, 2, 0]).all()
    check_fractions(pc.fractions[("Sn", "Sn")], "rutile")
    tnet = mn.bridged_graph(frame, table, formers={"Sn"}, anions={"O"})
    assert set(tnet.degree[sn].tolist()) == {10}
    assert int((tnet.multiplicity == 2).sum()) == n_sn
    assert any("triclusters" in note and str(int((~sn).sum())) in note
               for note in tnet.notes)


def test_a_confacial_pair_shares_a_face():
    h, r = 1.2, 1.6
    bridge = [(r * np.cos(t), r * np.sin(t), 0.0)
              for t in np.radians([0, 120, 240])]
    top = [(r * np.cos(t), r * np.sin(t), 2 * h)
           for t in np.radians([60, 180, 300])]
    bottom = [(x, y, -z) for x, y, z in top]
    cart = np.array([(0, 0, h), (0, 0, -h)] + bridge + top + bottom) + 10.0
    frame = md_model.frame_from_arrays(["Fe", "Fe"] + ["Cl"] * 9, cart,
                                       box_ang=np.eye(3) * 20.0)
    pairs = bulk.find_pairs(frame, 5.0)
    cut = _first_gap_cutoff(frame, pairs, "Fe", "Cl", 6)
    graph = mn.graph_from_pairs(frame, pairs, {("Fe", "Cl"): cut},
                                {("Fe", "Cl"): "test: between the 6th and "
                                               "7th Fe-Cl distance"})
    pc = mn.polyhedral_connectivity(graph, centres={"Fe"}, ligands={"Cl"})
    assert pc.counts[("Fe", "Fe")] == {"corner": 0, "edge": 0, "face": 1}
    assert pc.shared[("Fe", "Fe")] == {3: 1}
    assert pc.pairs.tolist() == [[0, 1, 0, 0, 0, 3]]


# ---------------------------------------------------------------------------
# components, percolation, clustering
# ---------------------------------------------------------------------------

def test_quartz_network_is_one_three_dimensional_component():
    frame, table = _quartz()
    comp = mn.components(mn.graph_from_bonds(frame, table))
    assert comp.n_components == 1 and comp.largest_fraction == 1.0
    assert comp.dimensionality.tolist() == [3]
    assert comp.spans.tolist() == [[True, True, True]]


def test_a_layer_spans_two_axes_and_clusters_are_counted():
    """A square net in the ab plane with vacuum along c is two-dimensional;
    Na atoms in two groups of 3 and 2 make two finite clusters."""
    net = _lattice([(0, 0, 0)], (3, 3, 1), 2.0, "C")
    frac = net.frac * [1, 1, 0.1]
    layer = md_model.frame_from_arrays(net.elements, frac=frac,
                                       box_ang=np.diag([6.0, 6.0, 20.0]))
    comp = mn.components(_distance_graph(layer, 3.5, "C", 4))
    assert comp.dimensionality.tolist() == [2]
    assert comp.spans.tolist() == [[True, True, False]]
    cart = np.array([[5, 5, 5], [7.5, 5, 5], [10, 5, 5], [5, 15, 15],
                     [7.5, 15, 15], [20, 20, 20]], dtype=float)
    frame = md_model.frame_from_arrays(["Na"] * 5 + ["O"], cart,
                                       box_ang=np.eye(3) * 25.0)
    pairs = bulk.find_pairs(frame, 6.0)
    graph = mn.graph_from_pairs(frame, pairs, {("Na", "Na"): 3.0},
                                {("Na", "Na"): "test: between 2.5 and 5.0 Å, "
                                               "the two Na-Na distances"})
    assert graph.n_nodes == 5                      # O is not a node
    comp = mn.components(graph)
    assert comp.sizes.tolist() == [3, 2]
    assert comp.largest_fraction == pytest.approx(0.6)
    assert comp.nodes_by_size() == {2: 2, 3: 3}
    assert comp.label[5] == -1


def _cuprite(reps):
    """Cu2O, Pn-3m: O at (0, 0, 0) and (1/2, 1/2, 1/2), Cu at the four
    (1/4, 1/4, 1/4)-type sites; a = 4.27 Å is illustrative. Each Cu has two
    O at a sqrt(3) / 4 = 1.85 Å; the next O is at a sqrt(11) / 4 = 3.54 Å."""
    basis = [(0, 0, 0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25),
             (0.75, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.75, 0.75)]
    frame = _lattice(basis, reps, 4.27, ["O", "O", "Cu", "Cu", "Cu", "Cu"])
    pairs = bulk.find_pairs(frame, 4.0)
    cut = _first_gap_cutoff(frame, pairs, "Cu", "O", 2)
    return mn.graph_from_pairs(frame, pairs, {("Cu", "O"): cut},
                               {("Cu", "O"): "test: between the 2nd and 3rd "
                                             "Cu-O distance"})


def test_interpenetrating_nets_count_the_same_in_a_cell_and_a_supercell():
    """Cuprite's Cu-O network is two interpenetrating nets that the body
    centring maps onto each other. The 6-atom cell shows them as one
    component whose translations have index 2 (two pieces); the 2x2x2
    supercell shows two components of index 1. The number of pieces, the
    largest piece's fraction (1/2) and the dimensionality agree; counted per
    component, the cell had given a largest fraction of 1."""
    one, eight = mn.components(_cuprite((1, 1, 1))), \
        mn.components(_cuprite((2, 2, 2)))
    assert one.n_components == 1 and one.copies.tolist() == [2]
    assert abs(int(round(np.linalg.det(one.translations[0])))) == 2
    assert eight.n_components == 2 and eight.copies.tolist() == [1, 1]
    for comp in (one, eight):
        assert comp.n_pieces == 2
        assert comp.largest_fraction == 0.5
        assert comp.dimensionality.tolist() == [3] * comp.n_components
    assert any("interpenetrating" in note and "copies 2" in note
               for note in one.notes)
    avg = mn.average_components([one, eight])
    assert avg["largest fraction"].mean == 0.5
    assert avg["pieces"].mean == 2.0 and avg["pieces"].std == 0.0


# ---------------------------------------------------------------------------
# Warren-Cowley
# ---------------------------------------------------------------------------

def _rocksalt_on_cubic(reps=(4, 4, 4)):
    frame = _lattice([(0, 0, 0)], reps, 2.0, "Na")
    cells = np.rint(frame.frac * np.asarray(reps, float)).astype(int)
    odd = cells.sum(axis=1) % 2 == 1
    symbols = np.where(odd, "Cl", "Na")
    return md_model.frame_from_arrays(symbols, frac=frame.frac,
                                      box_ang=frame.box_ang)


def test_warren_cowley_of_a_fully_ordered_and_a_pure_arrangement():
    """Alternating A/B on the simple cubic lattice: every first neighbour is
    unlike, so alpha_AB = 1 - 1/0.5 = -1 and alpha_AA = 1. One element:
    alpha = 0. The sum rule sum_j c_j alpha_ij = 0 holds in both."""
    frame = _rocksalt_on_cubic()
    pairs = bulk.find_pairs(frame, 3.5)
    cut = _first_gap_cutoff(frame, pairs, "Na", "Cl", 6)
    cutoffs = {("Na", "Cl"): cut, ("Na", "Na"): cut, ("Cl", "Cl"): cut}
    sources = {k: "test: midway between the first and second shell"
               for k in cutoffs}
    wc = mn.warren_cowley(mn.graph_from_pairs(frame, pairs, cutoffs, sources))
    assert wc.elements == ("Cl", "Na")
    assert wc.alpha.tolist() == [[1.0, -1.0], [-1.0, 1.0]]
    assert wc.joinable.all()
    assert np.allclose(wc.alpha @ wc.concentration, 0.0, atol=1e-15)
    pure = mn.warren_cowley(_cubic((3, 3, 3)))
    assert pure.alpha.tolist() == [[0.0]]


def test_warren_cowley_sum_rule_and_unjoinable_pairs_on_quartz():
    frame, table = _quartz()
    wc = mn.warren_cowley(mn.graph_from_bonds(frame, table))
    assert np.allclose(wc.alpha @ wc.concentration, 0.0, atol=1e-14)
    assert wc.as_dict()[("Si", "Si")] == 1.0
    assert not wc.joinable[1, 1] and wc.joinable[0, 1]
    assert any("never joins" in n and "Si-Si" in n for n in wc.notes)


# ---------------------------------------------------------------------------
# frame averages, invariance
# ---------------------------------------------------------------------------

def test_two_identical_frames_average_with_zero_spread():
    frame, table = _quartz()
    graph = mn.bridged_graph(frame, table, formers={"Si"}, anions={"O"})
    stats = mn.ring_statistics(graph, "primitive", 8, t_elements={"Si"})
    rings = mn.average_rings([stats, stats])
    for name in ("R_C", "R_N", "P_N"):
        assert (rings[name].std == 0).all() and \
            np.array_equal(rings[name].mean, getattr(
                stats, {"R_C": "rc", "R_N": "rn", "P_N": "pn"}[name]))
    sizes = rings["ring sizes"]
    check_fractions(sizes.mean, "ring sizes")
    assert sizes.as_dict()[6][0] == pytest.approx(24 / (24 + 120))
    assert set(rings["ring sizes (T atoms)"].keys) == {6, 8}
    cs = mn.coordination_sequences(graph, 4)
    avg = mn.average_coordination_sequences([cs, cs])["Si"]
    assert avg.mean.tolist() == [4, 12, 30, 52] and (avg.std == 0).all()
    pc = mn.polyhedral_connectivity(mn.graph_from_bonds(frame, table),
                                    centres={"Si"}, ligands={"O"})
    shared = mn.average_polyhedral_connectivity([pc, pc])[("Si", "Si")]
    assert shared.as_dict()["corner"] == (1.0, 0.0)
    comp = mn.components(graph)
    avg = mn.average_components([comp, comp])
    assert avg["largest fraction"].mean == 1.0
    assert avg["nodes by dimensionality"].as_dict()[3] == (1.0, 0.0)
    wc = mn.warren_cowley(mn.graph_from_bonds(frame, table))
    assert mn.average_warren_cowley([wc, wc])[("O", "Si")].std == 0.0


def test_frames_without_rings_average_as_counts_only():
    box = np.diag([6.0, 20.0, 20.0])
    frac = np.array([[0.0, 0.5, 0.5], [1 / 3, 0.5, 0.5], [2 / 3, 0.5, 0.5]])
    frame = md_model.frame_from_arrays(["C"] * 3, frac=frac, box_ang=box)
    stats = mn.ring_statistics(_distance_graph(frame, 4.5, "C", 2), "king", 6)
    out = mn.average_rings([stats, stats])
    assert "ring sizes" not in out
    assert (out["ring counts"].mean == 0).all()
    assert any("undefined" in n for n in out["ring counts"].notes)


def test_the_search_scope_is_stated_where_it_shapes_a_number():
    """P_max(n) counts the nodes whose largest ring is of n nodes, so
    P_max(max_size) = 1 wherever P_N(max_size) > 0, by construction, and it
    falls as max_size grows: simple cubic primitive rings, max_size 4 gives
    P_max(4) = 1, max_size 6 gives 0. Both results say so, and the
    primitive one says that larger rings are not counted."""
    graph = _cubic((3, 3, 3))
    four = mn.ring_statistics(graph, "primitive", 4)
    six = mn.ring_statistics(graph, "primitive", 6)
    assert four.p_max[1] == 1.0 and six.p_max[1] == 0.0
    assert four.rc[1] == six.rc[1] and four.pn[1] == six.pn[1]
    for stats in (four, six):
        m = stats.max_size
        assert any(f"P_max({m}) = 1 wherever P_N({m}) > 0" in note
                   for note in stats.notes)
        assert any(f"primitive rings of more than {m} nodes are not "
                   "searched" in note for note in stats.notes)
    sizes = mn.average_rings([four, four])["ring sizes"]
    assert any("depend on max_size" in note for note in sizes.notes)


def test_frame_averages_carry_the_frames_notes():
    """Two open chains of 5 and 6 atoms (no ring; every angle unclosed and
    acyclic): the averaged series carry the unclosed-angle note with the
    numbers that differ given as a range. Warren-Cowley carries 'alpha = 1 by
    construction' on the pair it concerns; polyhedral sharing what 'edge'
    and 'face' count. Before, every average carried the graph definition
    only."""
    def chain(n_atoms):
        cart = np.column_stack([np.arange(n_atoms) * 1.5 + 5.0,
                                np.full(n_atoms, 5.0), np.full(n_atoms, 5.0)])
        frame = md_model.frame_from_arrays(["C"] * n_atoms, cart,
                                           box_ang=np.eye(3) * 30.0)
        pairs = bulk.find_pairs(frame, 4.0)
        return mn.graph_from_pairs(frame, pairs, {("C", "C"): 2.0},
                                   {("C", "C"): "test: between 1.5 and 3.0 Å"})
    stats = [mn.ring_statistics(chain(k), "king", 8) for k in (5, 6)]
    rn = mn.average_rings(stats)["R_N"]
    assert any("3 to 4 of 3 to 4 angles lie on no ring of at most 8 nodes"
               in note for note in rn.notes)
    assert any("are in no ring" in note for note in rn.notes)
    frame, table = _quartz()
    wc = mn.warren_cowley(mn.graph_from_bonds(frame, table))
    avg = mn.average_warren_cowley([wc, wc])
    assert any("by construction" in n and "Si and Si" in n
               for n in avg[("Si", "Si")].notes)
    assert not any("never joins O and Si" in n for n in avg[("O", "Si")].notes)
    pc = mn.polyhedral_connectivity(mn.graph_from_bonds(frame, table),
                                    centres={"Si"}, ligands={"O"})
    shared = mn.average_polyhedral_connectivity([pc, pc])[("Si", "Si")]
    assert any("'edge' counts two shared ligands" in n for n in shared.notes)


def test_t_elements_absent_and_rings_on_another_frame():
    frame, table = _quartz()
    tnet = mn.bridged_graph(frame, table, formers={"Si"}, anions={"O"})
    stats = mn.ring_statistics(tnet, "king", 6, t_elements={"Al"})
    assert set(stats.t_sizes.tolist()) == {0}
    assert any("t_elements: Al not among the graph's nodes" in note
               for note in stats.notes)
    other = md_model.frame_from_arrays(frame.elements,
                                       frac=(frame.frac + 0.01) % 1.0,
                                       box_ang=frame.box_ang)
    ring = stats.rings[0]
    assert ring.positions_ang(frame).shape == (6, 3)
    with pytest.raises(ValueError, match="another frame"):
        ring.positions_ang(other)


def test_primitive_search_holds_few_candidates_at_once(monkeypatch):
    """The search forms every candidate cycle (pairs of shortest paths to a
    far point) from every node; it used to hold each distinct candidate to
    the end, 4.2 MB per node on a 3000-atom Na2O-3SiO2 graph with its Na-O
    bonds (12.6 GB extrapolated). It now holds a candidate only until it is
    decided. On a 216-atom diamond to 12 nodes: 20 520 distinct candidates,
    at most 1 702 held at once (counted 2026-10-07); the bound pinned here
    is a tenth of the distinct ones. The rings are unchanged."""
    graph = _diamond((3, 3, 3))
    distinct = set()
    real = mn._canonical

    def counting(keys, n):
        out = real(keys, n)
        distinct.add(out[0])
        return out
    monkeypatch.setattr(mn, "_canonical", counting)
    stats = mn.ring_statistics(graph, "primitive", 12)
    assert stats.n_rings == 2 * graph.n_nodes
    assert stats.n_candidates >= len(distinct) > 0
    assert 0 < stats.peak_open_candidates * 10 < len(distinct)
    assert any(f"at most {stats.peak_open_candidates} at once" in note
               for note in stats.notes)


def test_packed_candidate_keys_round_trip():
    """The open candidates are held as packed integers: 32-bit while every
    key fits (3000 atoms, rings to 24 nodes), 64-bit beyond (12 000 atoms),
    tuples when not even 64 bits hold them. Canonical keys can be negative
    (a member in a cell below the first member's)."""
    from types import SimpleNamespace
    small = SimpleNamespace(n=3000, reach=25, width=4 * 25 + 3)   # max_size 24
    big = SimpleNamespace(n=12000, reach=25, width=4 * 25 + 3)
    huge = SimpleNamespace(n=12000, reach=10 ** 6, width=4 * 10 ** 6 + 3)
    for lift, size in ((small, 4), (big, 8)):
        pack, unpack = mn._key_codec(lift)
        w = lift.width
        edge = lift.n - 1 + lift.n * (lift.reach - 1) * (w * w + w + 1)
        canon = (5, -edge, edge, 2999, -1, 7, 0, 11)
        key = pack(canon)
        assert isinstance(key, bytes) and len(key) == size * len(canon)
        assert unpack(key) == canon
    assert mn._key_codec(huge) == (None, None)


def test_a_max_size_far_above_the_rings_costs_no_memory_per_node():
    """found is built on access; what is stored stops at the largest ring
    found (6 on diamond), so max_size 2000 stores 4 columns per node, not
    1998. Every per-size array still runs to max_size."""
    graph = _diamond((1, 1, 1))
    stats = mn.ring_statistics(graph, "king", 2000)
    assert stats.found_compact.shape == (8, 4)
    assert stats.found.shape == (8, 1998)
    assert stats.found[:, 3].tolist() == [12] * 8 and \
        stats.found[:, 4:].sum() == 0
    assert stats.rn.shape == stats.pn.shape == (1998,)
    assert stats.rings_through(0) == 12
    assert len(stats.as_rows()) == 1998


def test_permuting_and_translating_the_atoms_changes_no_count():
    frame, table = _quartz()
    rng = np.random.default_rng(7)
    order = rng.permutation(frame.n_atoms)
    moved = md_model.frame_from_arrays(
        frame.elements[order], frac=frame.frac[order] + rng.random(3),
        box_ang=frame.box_ang)
    ox = md_model.model_oxidation(moved.species).per_atom(moved.elements)
    table2, _ = bulk.analyse_frame(moved, ox)
    for f, t in ((frame, table), (moved, table2)):
        graph = mn.graph_from_bonds(f, t)
        tnet = mn.bridged_graph(f, t, formers={"Si"}, anions={"O"})
        got = []
        for criterion in mn.CRITERIA:
            s = mn.ring_statistics(tnet, criterion, 10)
            got.append((s.ring_counts.tolist(), s.rc.tolist(), s.pn.tolist(),
                        [None if np.isnan(x) else x
                         for x in s.p_max.tolist()]))
        cs = mn.coordination_sequences(graph, 5)
        got.append(sorted(map(tuple, cs.counts.tolist())))
        got.append(mn.components(graph).sizes.tolist())
        if f is frame:
            first = got
    assert got == first


# ---------------------------------------------------------------------------
# refusals and provenance
# ---------------------------------------------------------------------------

def test_nothing_physical_has_a_default():
    frame, table = _quartz()
    with pytest.raises(ValueError, match="no default"):
        mn.bridged_graph(frame, table, formers=None, anions={"O"})
    with pytest.raises(ValueError, match="not the text"):
        mn.bridged_graph(frame, table, formers="Si", anions={"O"})
    pairs = bulk.find_pairs(frame, 3.5)
    with pytest.raises(ValueError, match="no source"):
        mn.graph_from_pairs(frame, pairs, {("Si", "O"): 1.8}, {})
    with pytest.raises(ValueError, match="at least one"):
        mn.graph_from_pairs(frame, pairs, {}, {})
    with pytest.raises(ValueError, match="beyond"):
        mn.graph_from_pairs(frame, pairs, {("Si", "O"): 4.0},
                            {("Si", "O"): "test"})
    with pytest.raises(TypeError):
        mn.ring_statistics(mn.graph_from_bonds(frame, table), "king")
    with pytest.raises(TypeError):
        mn.coordination_sequences(mn.graph_from_bonds(frame, table))
    with pytest.raises(ValueError, match="no default"):
        mn.polyhedral_connectivity(mn.graph_from_bonds(frame, table),
                                   centres={"Si"}, ligands=None)


def test_refusals_name_their_reason():
    frame, table = _quartz()
    graph = mn.graph_from_bonds(frame, table)
    other = _rutile()[1]
    calls = {
        "criterion": lambda: mn.ring_statistics(graph, "strong", 8),
        "at least 3": lambda: mn.ring_statistics(graph, "king", 2),
        "whole number": lambda: mn.ring_statistics(graph, "king", 6.0),
        "another frame": lambda: mn.graph_from_bonds(frame, other),
        "both as centre": lambda: mn.polyhedral_connectivity(
            graph, centres={"Si", "O"}, ligands={"O"}),
        "both as former": lambda: mn.bridged_graph(
            frame, table, formers={"O"}, anions={"O"}),
        "two cutoffs": lambda: mn.graph_from_pairs(
            frame, bulk.find_pairs(frame, 3.0),
            {("Si", "O"): 1.8, ("O", "Si"): 1.9}, {("Si", "O"): "t"}),
        "NetworkGraph is needed": lambda: mn.components(table),
        "not node elements": lambda: mn.coordination_sequences(
            mn.bridged_graph(frame, table, formers={"Si"}, anions={"O"}), 3,
            centres={"O"}),
    }
    for reason, call in calls.items():
        with pytest.raises(ValueError, match=reason):
            call()


def test_pairs_of_another_frame_are_refused():
    frame, _ = _quartz()
    other = md_model.frame_from_arrays(frame.elements,
                                       frac=(frame.frac + 0.01) % 1.0,
                                       box_ang=frame.box_ang)
    with pytest.raises(ValueError, match="another frame"):
        mn.graph_from_pairs(frame, bulk.find_pairs(other, 3.0),
                            {("Si", "O"): 1.8}, {("Si", "O"): "test"})


def test_the_graph_records_its_definition_and_the_bonds_left_out():
    frame, table = _quartz()
    graph = mn.graph_from_bonds(frame, table, elements={"Si", "O"})
    assert graph.n_edges == len(bulk.bonds_at(table))
    assert "v > 0.075 v.u." in graph.definition
    assert "Brese" in graph.definition
    only_si = mn.graph_from_bonds(frame, table, elements={"Si"})
    assert only_si.n_edges == 0
    assert any("outside the nodes" in n and "Si-O" in n for n in only_si.notes)
    pairs = bulk.find_pairs(frame, 3.5)
    dist = mn.graph_from_pairs(frame, pairs, {("O", "Si"): 1.8},
                               {("Si", "O"): "first minimum of g_SiO(r)"})
    assert "first minimum of g_SiO(r)" in dist.definition
    assert dist.n_edges == graph.n_edges
    assert np.array_equal(dist.u, graph.u) and np.array_equal(dist.v, graph.v)


def test_pairs_the_search_left_out_are_noted_on_the_graph():
    """Two O 0.3 Å apart, below the pair search's d_min (0.4 Å, the
    NeighborFinder dmin): the search leaves the pair out from both ends and
    the valence table notes it. A graph built on that table, or on those
    pairs, has no edge for it and now says so (before, the graph's notes
    were silent: the table's notes stopped at the table)."""
    cart = np.array([[0.0, 0.0, 0.0], [1.62, 0.0, 0.0], [1.62, 0.3, 0.0]])
    frame = md_model.frame_from_arrays(["Si", "O", "O"], cart + 10.0,
                                       box_ang=np.eye(3) * 20.0)
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox)
    assert table.n_below_d_min == 2
    assert any(note.startswith("2 ordered pair(s) at or below 0.4")
               for note in table.notes)
    for graph in (mn.graph_from_bonds(frame, table),
                  mn.bridged_graph(frame, table, formers={"Si"},
                                   anions={"O"})):
        assert [f"valence table: {note}" for note in table.notes] == \
            [note for note in graph.notes if note.startswith("valence table")]
    pairs = bulk.find_pairs(frame, 3.0)
    assert pairs.n_below_d_min == 2
    dist = mn.graph_from_pairs(frame, pairs, {("Si", "O"): 2.0},
                               {("Si", "O"): "test: between the Si-O "
                                             "distances and 3 Å"})
    assert dist.n_edges == 2
    assert ("2 ordered pair(s) at or below 0.4 Å were left out by the pair "
            "search (bulk.iter_pairs, its d_min_ang), so they are not "
            "edges") in dist.notes


def test_cancel_and_progress_hooks():
    """The primitive search reports two passes over the nodes (the rings,
    then the per-angle records of the vertex symbol)."""
    graph = _diamond((2, 2, 2))
    calls = []
    mn.ring_statistics(graph, "primitive", 6,
                       progress=lambda d, t: calls.append((d, t)))
    assert calls[-1] == (2 * graph.n_nodes, 2 * graph.n_nodes)
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)
    with pytest.raises(mn.Cancelled):
        mn.ring_statistics(graph, "king", 6, cancelled=lambda: True)
    with pytest.raises(mn.Cancelled):
        mn.coordination_sequences(graph, 3, cancelled=lambda: True)


def test_progress_is_also_called_on_a_clock(monkeypatch):
    """Besides every 1 % of the units, progress() is called whenever
    PROGRESS_INTERVAL_S has passed: on a slow search the 1 % steps can be a
    minute apart (49 s measured on a Na2O-3SiO2 graph with its Na-O bonds).
    A clock that advances two intervals per reading makes every unit report;
    a 512-node graph has a 1 % step of 5 units."""
    graph = _diamond((4, 4, 4))
    ticks = iter(range(10 ** 6))
    monkeypatch.setattr(mn, "_clock",
                        lambda: 2 * mn.PROGRESS_INTERVAL_S * next(ticks))
    calls = []
    mn.ring_statistics(graph, "king", 6,
                       progress=lambda d, t: calls.append(d))
    assert calls == list(range(1, graph.n_nodes + 1))


# ---------------------------------------------------------------------------
# no verdicts, no Qt
# ---------------------------------------------------------------------------

VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def test_no_string_in_the_module_carries_a_verdict():
    source = (ROOT / "facet" / "core" / "md_network.py").read_text(
        encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 50
    assert [s for s in strings if VERDICT.search(s)] == []


def test_no_note_or_message_carries_a_verdict():
    frame, table = _quartz()
    notes = []
    graph = mn.graph_from_bonds(frame, table, elements={"Si"})
    notes += graph.notes
    tnet = mn.bridged_graph(_rutile()[0], _rutile()[1], formers={"Sn"},
                            anions={"O"})
    notes += tnet.notes
    stats = mn.ring_statistics(tnet, "king", 4)
    notes += stats.notes
    notes += mn.warren_cowley(mn.graph_from_bonds(frame, table)).notes
    notes += mn.polyhedral_connectivity(
        mn.graph_from_bonds(frame, table), centres={"Si"},
        ligands={"O"}).notes
    notes += mn.components(tnet).notes
    try:
        mn.graph_from_pairs(frame, bulk.find_pairs(frame, 3.0),
                            {("Si", "O"): 1.8}, {})
    except ValueError as error:
        notes.append(str(error))
    assert len(notes) >= 8
    assert [n for n in notes if VERDICT.search(n)] == []


def test_the_module_imports_without_qt():
    code = ("import sys, facet.core.md_network;"
            "print('QT' if any(m.startswith(('PySide6', 'matplotlib')) "
            "for m in sys.modules) else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
