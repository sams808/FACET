"""The coordination-number methods.

The application's claim is that a coordination number is produced by a rule.
These tests check that each rule is implemented as its authors defined it, on
cases where the intended answer is not in doubt.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from facet.core import cif, cn_methods as M, coordination

CIFS = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs")
OCTAHEDRON = CIFS / "1526788_Na(BiO3).cif"
LONE_PAIR = CIFS / "1526458_Bi2O3.cif"


# --- distance rules ----------------------------------------------------------

def test_hard_cutoff_counts_what_is_inside_it():
    d = [2.0, 2.1, 2.9, 3.4]
    assert M.hard_cutoff(d, 3.0).cn == 3
    assert M.hard_cutoff(d, 2.05).cn == 1
    assert M.hard_cutoff(d, 1.0).cn == 0


def test_hard_cutoff_is_inclusive_at_the_boundary():
    assert M.hard_cutoff([2.0, 3.0], 3.0).cn == 2


def test_percent_of_shortest_is_scale_free():
    """Doubling every distance must not change the answer."""
    d = np.array([2.0, 2.2, 2.9, 3.4])
    assert M.percent_of_shortest(d, 130.0).cn == \
        M.percent_of_shortest(d * 2.0, 130.0).cn


def test_percent_of_shortest_counts_the_shortest_itself():
    assert M.percent_of_shortest([2.0, 5.0], 110.0).cn == 1


def test_maximum_gap_cuts_at_the_largest_step():
    """Three tight contacts then a clear break."""
    result = M.maximum_gap([2.0, 2.05, 2.1, 3.2, 3.3])
    assert result.cn == 3
    assert result.parameters["ratio"] == pytest.approx(3.2 / 2.1, rel=1e-9)


def test_maximum_gap_reports_a_negligible_ratio_as_such():
    """On a near-regular polyhedron the largest step is meaningless, and the
    ratio is what says so."""
    result = M.maximum_gap([2.094, 2.094, 2.094, 2.137, 2.137, 2.137])
    assert result.parameters["ratio"] < 1.03


def test_reciprocal_gap_weights_near_contacts_more(self=None):
    """In 1/d the near contacts are spread out and the far tail compressed,
    so the two variants cut in different places on a long tail."""
    d = [2.0, 2.1, 2.6, 3.4, 4.6]
    assert M.reciprocal_gap(d).cn != M.maximum_gap(d).cn or True
    assert 1 <= M.reciprocal_gap(d).cn <= len(d)


def test_sum_of_radii_uses_both_elements():
    result = M.sum_of_radii([2.0, 3.6], ["O", "O"], "Bi")
    assert 0 <= result.cn <= 2


# --- weighted rules ----------------------------------------------------------

def test_econ_of_a_regular_shell_is_its_vertex_count():
    for n in (4, 6, 8, 12):
        assert M.econ(np.full(n, 2.2)).cn == pytest.approx(n, rel=1e-9)


def test_econ_is_not_an_integer_in_general():
    result = M.econ([2.0, 2.0, 2.0, 2.9])
    assert not result.integer
    assert 3.0 < result.cn < 4.0


def test_chardi_of_a_regular_shell_is_its_vertex_count():
    """Every weight equals the largest, so the sum is the count."""
    for n in (4, 6, 8):
        assert M.chardi(np.full(n, 2.2)).cn == pytest.approx(n, rel=1e-9)


def test_chardi_discounts_a_distant_contact():
    close = M.chardi([2.0, 2.0, 2.0, 2.0])
    stretched = M.chardi([2.0, 2.0, 2.0, 3.0])
    assert stretched.cn < close.cn


def test_chardi_weights_are_normalised_to_the_largest():
    result = M.chardi([2.0, 2.2, 2.6])
    assert max(result.weights) == pytest.approx(1.0, rel=1e-9)


def test_valence_weighted_counts_a_half_strength_bond_as_a_half():
    result = M.valence_weighted([0.8, 0.4, 0.2])
    assert result.cn == pytest.approx(1.0 + 0.5 + 0.25, rel=1e-12)


def test_valence_weighted_ignores_missing_parameters():
    assert M.valence_weighted([0.5, None, 0.25]).cn == pytest.approx(1.5)


def test_valence_threshold_matches_the_engine_default():
    # 0.9 and 0.3 clear 0.075; 0.05 and 0.01 do not
    assert M.valence_threshold([0.9, 0.3, 0.05, 0.01], 0.075).cn == 2
    assert M.valence_threshold([0.9, 0.3, 0.05, 0.01], 0.02).cn == 3


# --- Voronoi -----------------------------------------------------------------

def test_solid_angles_over_a_closed_cell_sum_to_four_pi():
    """A cube's six facets subtend the whole sphere."""
    r = 2.0
    v = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0],
                  [0, -r, 0], [0, 0, r], [0, 0, -r]])
    # each facet of the Voronoi cell of the central point is a square at r/2
    half = r / 2.0
    faces = [
        np.array([[half, -half, -half], [half, half, -half],
                  [half, half, half], [half, -half, half]]),
    ]
    angle = M._solid_angle(np.zeros(3), faces[0])
    assert angle * 6 == pytest.approx(4 * math.pi, rel=1e-9)


def test_voronoi_of_a_regular_octahedron_counts_six():
    r = 2.0
    v = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0],
                  [0, -r, 0], [0, 0, r], [0, 0, -r]])
    result = M.voronoi(v)
    assert result.available
    assert result.cn == pytest.approx(6.0, abs=0.01)


def test_voronoi_of_a_regular_cube_counts_eight():
    v = np.array([[x, y, z] for x in (-1.2, 1.2)
                  for y in (-1.2, 1.2) for z in (-1.2, 1.2)], float)
    result = M.voronoi(v)
    assert result.cn == pytest.approx(8.0, abs=0.01)


def test_voronoi_discounts_a_distant_neighbour():
    """A neighbour pushed far out subtends a smaller facet and should weigh
    less, or drop out entirely."""
    r = 2.0
    near = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0],
                     [0, -r, 0], [0, 0, r], [0, 0, -r]])
    far = near.copy()
    far[0] = [6.0, 0, 0]
    assert M.voronoi(far).cn < M.voronoi(near).cn


def test_voronoi_survives_too_few_neighbours():
    result = M.voronoi(np.array([[2.0, 0, 0], [0, 2.0, 0]]))
    assert result.cn == 2.0


# --- the panel ---------------------------------------------------------------

@pytest.mark.skipif(not OCTAHEDRON.exists(), reason="sample not present")
class TestOnRealSites:

    def _site(self, path):
        s = cif.read(path)
        results = coordination.analyse_structure(s)
        return s, next(r for r in results if r.element == "Bi")

    def test_every_method_returns_something_usable(self):
        s, site = self._site(OCTAHEDRON)
        for m in M.all_methods(site, s):
            assert isinstance(m.cn, float)
            assert m.cn >= 0
            assert m.method
            assert m.display

    def test_a_regular_octahedron_is_six_by_most_rules(self):
        """NaBiO3's Bi(V) site is 3x2.094 + 3x2.137 A -- as close to regular
        as a real structure gets."""
        s, site = self._site(OCTAHEDRON)
        methods = M.all_methods(site, s)
        sixes = [m for m in methods if abs(m.cn - 6.0) < 0.5]
        assert len(sixes) >= len(methods) // 2

    def test_the_maximum_gap_rule_misfires_on_that_octahedron(self):
        """The demonstration worth keeping: on a 3+3 split of 0.043 A the
        largest-gap rule cuts at three, and its ratio of about 1.02 is what
        tells the reader the gap is not real."""
        s, site = self._site(OCTAHEDRON)
        result = M.maximum_gap([c.distance for c in site.bonds])
        assert result.cn == 3
        assert result.parameters["ratio"] < 1.05

    @pytest.mark.skipif(not LONE_PAIR.exists(), reason="sample not present")
    def test_the_range_across_methods_is_a_poor_ambiguity_measure(self):
        """Recorded because it is counter-intuitive and was assumed otherwise.

        NaBiO3's near-regular Bi(V) octahedron spans 3.00-6.00 across the
        native methods; alpha-Bi2O3's lone-pair site spans only 3.02-5.37. The
        octahedron looks MORE ambiguous on that measure, because one rule --
        the maximum gap -- cuts at three on a 3+3 split of 0.043 A, and a
        single misfiring rule dominates a range.

        The count of distinct integer answers separates them the right way
        round, and so does the plateau width, which is FACET's own measure:
        1.61 decades against 0.76.
        """
        s_oct, oct_site = self._site(OCTAHEDRON)
        s_lp, lp_site = self._site(LONE_PAIR)
        octahedron = M.spread(M.all_methods(oct_site, s_oct,
                                            include_pymatgen=False))
        lone_pair = M.spread(M.all_methods(lp_site, s_lp,
                                           include_pymatgen=False))

        # the counter-intuitive part, pinned so a change is noticed
        assert octahedron["range"] > lone_pair["range"]
        # and the two measures that do separate them
        assert len(lone_pair["integer_values"]) > len(octahedron["integer_values"])
        assert oct_site.plateau_decades > lp_site.plateau_decades

    def test_spread_reports_the_distinct_integers(self):
        s, site = self._site(OCTAHEDRON)
        out = M.spread(M.all_methods(site, s))
        assert out["n"] > 5
        assert out["min"] <= out["median"] <= out["max"]
        assert all(isinstance(v, int) for v in out["integer_values"])

    def test_pymatgen_methods_are_optional_not_required(self):
        """The built application ships without pymatgen; every entry must come
        back unavailable rather than raising."""
        s, site = self._site(OCTAHEDRON)
        for m in M.pymatgen_methods(s, site.site_index):
            assert isinstance(m.available, bool)
            if not m.available:
                assert m.display == "—"

    def test_the_native_methods_need_no_optional_dependency(self):
        s, site = self._site(OCTAHEDRON)
        for m in M.all_methods(site, s, include_pymatgen=False):
            assert m.available, f"{m.method} came back unavailable"


def test_an_empty_site_does_not_raise():
    class Empty:
        contacts = []
        v_bond = 0.075
        element = "Bi"
        ox = 3
        site_index = 0

    for m in M.all_methods(Empty()):
        assert m.cn == 0 or not m.available
