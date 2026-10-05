"""A priori bond valences, against the paper that defines them.

Gagné & Hawthorne (IUCrJ 7 (2020) 581) solve the network equations by writing
out the valence-sum and loop rules for each structure. This module solves the
equivalent resistor network instead, which needs no cycle basis. The two have to
agree, and the place to show it is the example the paper works through in full.

The internal checks matter as much: the a priori valences must satisfy the
valence-sum rule exactly at every site, and the loop rule around every cycle.
Those are the equations being solved, so a solver that fails them is not wrong
by a little.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import bv, cif, coordination, network

CIFS = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif")
BI2O3 = sample_cif("1526458", "1526458_Bi2O3.cif")
CRYOLITE = sample_cif("9004097", "9004097_Cryolite.cif")


# ---------------------------------------------------------------------------
# against the paper
# ---------------------------------------------------------------------------

def test_the_worked_perovskite_example():
    """§4.1.2, GdMnO3 in Pnma, the one the paper sets out in full.

    "A makes two bonds to X1 and six bonds to X2, B makes two bonds to X1 and
    four bonds to X2" -- giving 2a + 6b = 3, 2c + 4d = 3, 2a + 2c = 2,
    3b + 2d = 2 and the path equation a - c + d - b = 0.
    """
    # the paper's own equations, solved directly
    matrix = np.array([[2, 6, 0, 0], [0, 0, 2, 4],
                       [2, 0, 2, 0], [1, -1, -1, 1]], float)
    theirs = np.linalg.solve(matrix, np.array([3, 3, 2, 0], float))

    # A=0, B=1, X1=2, X2=3; multiplicities 4, 4, 4, 8 in Pnma
    topology = network.Topology(
        bonds={(0, 2): 4 * 2, (0, 3): 4 * 6, (1, 2): 4 * 2, (1, 3): 4 * 4},
        charge={0: 4 * 3, 1: 4 * 3, 2: 4 * -2, 3: 8 * -2},
        multiplicity={0: 4, 1: 4, 2: 4, 3: 8},
        labels={0: "A", 1: "B", 2: "X1", 3: "X2"})
    ours = network.a_priori_valences(topology)

    assert ours[(0, 2)] == pytest.approx(theirs[0], abs=1e-12)   # a
    assert ours[(0, 3)] == pytest.approx(theirs[1], abs=1e-12)   # b
    assert ours[(1, 2)] == pytest.approx(theirs[2], abs=1e-12)   # c
    assert ours[(1, 3)] == pytest.approx(theirs[3], abs=1e-12)   # d
    # and the values themselves, which are exact fractions
    assert ours[(0, 2)] == pytest.approx(15 / 34)
    assert ours[(0, 3)] == pytest.approx(6 / 17)


@pytest.mark.parametrize("a_priori,observed,topol,cryst", [
    # PbCu(CuTeO7), ICSD 405329, quoted on p.42
    ([0.265, 0.372, 0.399, 0.399, 0.565],
     [0.060, 0.342, 0.498, 0.498, 0.590], 0.066, 0.091),
    # Na5NbO5, ICSD 24819, quoted on p.39
    ([0.981, 0.981, 0.981, 1.029, 1.029],
     [0.797, 0.797, 1.083, 1.028, 1.028], 0.023, 0.094),
    # Cs[Mo2O3(PO4)2], ICSD 79517, quoted on p.37
    ([0.915, 1.026, 1.078, 1.491, 1.491],
     [0.850, 0.801, 0.735, 1.735, 1.891], 0.232, 0.256),
])
def test_the_indices_reproduce_the_published_values(a_priori, observed,
                                                    topol, cryst):
    """Polyhedra the paper gives both the inputs and the answers for.

    To 0.0015 v.u., which is as close as the published inputs allow: the
    valences are printed to three decimals and the indices were computed from
    the unrounded ones. Reproducing 0.0916 where the paper prints 0.091 is
    agreement, not disagreement.
    """
    got_topol, got_cryst = network.indices(a_priori, observed)
    assert got_topol == pytest.approx(topol, abs=1.5e-3)
    assert got_cryst == pytest.approx(cryst, abs=1.5e-3)


def test_delta_topol_is_zero_for_a_regular_polyhedron():
    """A site whose bonds the topology makes identical has nothing to spread."""
    assert network.indices([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])[0] == 0.0
    assert network.indices([0.5] * 4, [0.4, 0.6, 0.45, 0.55])[0] == 0.0


def test_delta_cryst_is_zero_when_the_structure_is_what_the_topology_says():
    assert network.indices([0.3, 0.7], [0.3, 0.7])[1] == 0.0


# ---------------------------------------------------------------------------
# the equations being solved
# ---------------------------------------------------------------------------

def _structure(path):
    if not path.is_file():
        pytest.skip(f"{path.name} is not present")
    return cif.read(path)


@pytest.mark.parametrize("path", [BI2O3, CRYOLITE])
def test_the_valence_sum_rule_holds_exactly(path):
    """Every site's a priori valences sum to its formal charge.

    One of the two rules being solved. A residual here is not an approximation
    -- it means the system solved was not the one intended.
    """
    structure = _structure(path)
    results = coordination.analyse_structure(structure, bv.DEFAULT,
                                             cations_only=False)
    topology = network.topology_of(structure, results, 0.075)
    assert topology.closes, "; ".join(topology.reasons)
    valences = network.a_priori_valences(topology)

    for site in topology.sites:
        total = 0.0
        for (cation, anion), count in topology.bonds.items():
            if site == cation:
                total += count * valences[(cation, anion)]
            elif site == anion:
                total -= count * valences[(cation, anion)]
        assert total == pytest.approx(topology.charge[site], abs=1e-9), (
            f"{topology.labels[site]}: {total} against "
            f"{topology.charge[site]}")


@pytest.mark.parametrize("path", [BI2O3, CRYOLITE])
def test_the_loop_rule_holds_around_every_cycle(path):
    """The other rule: alternating sums around a closed path vanish.

    Taken over a cycle basis of the bond graph, which is where the published
    method has to make a choice and this one does not -- the potentials make it
    true by construction, so this is a check that the construction is right.
    """
    structure = _structure(path)
    results = coordination.analyse_structure(structure, bv.DEFAULT,
                                             cations_only=False)
    topology = network.topology_of(structure, results, 0.075)
    assert topology.closes
    valences = network.a_priori_valences(topology)

    import networkx  # noqa: F401  -- optional
    pytest.importorskip("networkx")
    import networkx as nx

    graph = nx.Graph()
    for (cation, anion) in topology.bonds:
        graph.add_edge(cation, anion)
    cycles = nx.cycle_basis(graph)
    assert cycles, "this structure has no cycles to check"
    for cycle in cycles:
        total = 0.0
        for i, node in enumerate(cycle):
            nxt = cycle[(i + 1) % len(cycle)]
            if (node, nxt) in valences:
                total += valences[(node, nxt)]
            else:
                total -= valences[(nxt, node)]
        assert total == pytest.approx(0.0, abs=1e-9)


def test_a_regular_structure_has_one_a_priori_valence():
    """Cryolite's AlF6 is a single bond type: the topology requires no spread.

    Na3AlF6 -- every Al-F bond is equivalent by topology, so Δ_topol must be
    zero and whatever variation the structure has is crystallographic.
    """
    structure = _structure(CRYOLITE)
    results = coordination.analyse_structure(structure, bv.DEFAULT,
                                             cations_only=False)
    rows = {r.label: r for r in network.analyse(structure, results, 0.075)}
    aluminium = rows["Al"]
    assert aluminium.coordination == 6
    assert aluminium.delta_topol == pytest.approx(0.0, abs=1e-9)
    assert aluminium.a_priori == pytest.approx([0.5] * 6, abs=1e-9)
    assert aluminium.delta_cryst < 0.05


# ---------------------------------------------------------------------------
# when it cannot answer
# ---------------------------------------------------------------------------

def test_an_unbalanced_cell_is_refused_rather_than_answered():
    topology = network.Topology(
        bonds={(0, 1): 4.0},
        charge={0: 3.0, 1: -2.0},          # does not balance
        multiplicity={0: 1, 1: 1}, labels={0: "A", 1: "X"})
    with pytest.raises(ValueError, match="net charge"):
        network.a_priori_valences(topology)


def test_a_topology_that_does_not_close_says_why():
    structure = _structure(CRYOLITE)
    cations = coordination.analyse_structure(structure, bv.DEFAULT)
    topology = network.topology_of(structure, cations, 0.075)
    assert not topology.closes
    assert any("only the cations" in r for r in topology.reasons)

    with pytest.raises(ValueError):
        network.analyse(structure, cations, 0.075)


def test_bismuth_oxide_separates_into_the_two_parts():
    """alpha-Bi2O3: a lone-pair structure, which is the point of the exercise.

    The numbers themselves are a measurement, not a claim -- what is asserted
    here is only that the split is computed and that the crystallographic part
    is the larger one, which is what a stereoactive lone pair does.
    """
    structure = _structure(BI2O3)
    results = coordination.analyse_structure(structure, bv.DEFAULT,
                                             cations_only=False)
    rows = [r for r in network.analyse(structure, results, 0.075)
            if r.element == "Bi"]
    assert len(rows) == 2
    for row in rows:
        assert row.delta_cryst > row.delta_topol
        assert np.isfinite(row.delta_topol) and np.isfinite(row.delta_cryst)
        assert sum(row.a_priori) == pytest.approx(abs(row.ox), abs=1e-9)
