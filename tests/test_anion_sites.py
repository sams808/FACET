"""An anion analysed as a site, not only as somebody else's ligand.

The engine took `cations_only=False` from the start and handed back the anion
sites with a coordination number of zero, no contacts and a sum of nan: the
contact list was filtered to anions, which for an anion centre removes
everything bonded to it, and the parameter was looked up cation-first, which
for an anion centre finds nothing.

The arithmetic is checked against `bv_report.anion_sums`, which has done the
reversed lookup correctly all along by a different route -- a separate neighbour
search, a separate loop, the same number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import bv, bv_report, cif, coordination

CRYOLITE = sample_cif("9004097", "9004097_Cryolite.cif")
PHOSPHATE = sample_cif("1004091", "1004091_BiNa3O8P2.cif")

pytestmark = pytest.mark.skipif(not CRYOLITE.is_file(),
                                reason="the reference structures are not present")


@pytest.fixture(scope="module")
def cryolite():
    return cif.read(CRYOLITE)


def _anions(results):
    return [r for r in results if r.ox is not None and r.ox < 0]


def test_an_anion_site_gets_contacts_and_a_coordination_number(cryolite):
    results = coordination.analyse_structure(cryolite, bv.DEFAULT,
                                             cations_only=False)
    anions = _anions(results)
    assert len(anions) == 3                       # F1, F2, F3
    for r in anions:
        assert r.cn_valence > 0, f"{r.label} has no coordination number"
        assert r.bonds, f"{r.label} has no contacts"
        assert np.isfinite(r.bvs)
        # every contact of an anion is a cation
        for c in r.bonds:
            assert cryolite.sites[c.site_index].is_anion is False


def test_the_sum_matches_the_independent_anion_report(cryolite):
    """Two routes to the same number: a different search, a different loop."""
    results = coordination.analyse_structure(cryolite, bv.DEFAULT,
                                             cations_only=False)
    theirs = {a.label: a for a in bv_report.anion_sums(cryolite, bv.DEFAULT)}
    checked = 0
    for r in _anions(results):
        other = theirs[r.label]
        assert r.bvs == pytest.approx(other.bvs, abs=1e-9)
        assert r.cn_valence == other.n_contacts
        checked += 1
    assert checked == 3


def test_the_cations_are_unchanged_by_asking_for_the_anions(cryolite):
    """Adding sites to the list must not alter the ones already in it."""
    only = coordination.analyse_structure(cryolite, bv.DEFAULT)
    both = coordination.analyse_structure(cryolite, bv.DEFAULT,
                                          cations_only=False)
    by_label = {r.label: r for r in both}
    for r in only:
        other = by_label[r.label]
        assert other.cn_valence == r.cn_valence
        assert other.bvs == pytest.approx(r.bvs, abs=1e-12)
        assert other.phi == pytest.approx(r.phi, abs=1e-12)


def test_the_fluorine_sums_are_near_one(cryolite):
    """Na3AlF6: each F takes one Al and three Na, and should come to about 1.

    Not a verdict about the structure -- a check that the reversed lookup is
    reaching the right parameters at all. A sum of 2 would mean it had picked
    up the anion-anion contacts, and one near 0.5 that it was using the
    estimator.
    """
    results = coordination.analyse_structure(cryolite, bv.DEFAULT,
                                             cations_only=False)
    for r in _anions(results):
        assert 0.85 < r.bvs < 1.15, f"{r.label}: {r.bvs}"
        assert r.cn_valence == 4


def test_the_discrepancy_is_measured_against_the_size_of_the_charge():
    """A bond-valence sum is a sum of positive terms.

    An anion at -1 whose sum is 1.05 is 0.05 over, not 2.05. The signed
    subtraction was invisible while only cations were analysed, and it feeds
    the global instability index and the valence-discrepancy colouring.
    """
    structure = cif.read(CRYOLITE)
    results = coordination.analyse_structure(structure, bv.DEFAULT,
                                             cations_only=False)
    for r in results:
        if r.ox is None or r.valence_discrepancy is None:
            continue
        assert r.valence_discrepancy == pytest.approx(r.bvs - abs(r.ox))
        assert abs(r.valence_discrepancy) < 0.6, (
            f"{r.label} {r.ox:+d}: discrepancy {r.valence_discrepancy}")


@pytest.mark.skipif(not PHOSPHATE.is_file(), reason="structure not present")
def test_an_oxide_structure_too():
    """Phosphate oxygens: three to six cations each, sums near 2."""
    structure = cif.read(PHOSPHATE)
    results = coordination.analyse_structure(structure, bv.DEFAULT,
                                             cations_only=False)
    anions = _anions(results)
    assert len(anions) >= 8
    sums = np.array([r.bvs for r in anions])
    assert (sums > 1.5).all() and (sums < 2.5).all(), sums
    assert all(2 <= r.cn_valence <= 7 for r in anions)


def test_the_project_asks_for_them_only_when_told(cryolite, tmp_path):
    from facet.core import project as project_mod

    project = project_mod.Project()
    project.add(cryolite)
    assert project.include_anions is False
    assert not _anions(project.results_for(project.current))

    project.set_include_anions(True)
    assert _anions(project.results_for(project.current))

    project.set_include_anions(False)
    assert not _anions(project.results_for(project.current))
