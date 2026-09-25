"""Checks on the bond-valence model.

The important one is :func:`test_fitted_table_agrees_with_estimator`. The fitted
table is typed from the literature, and a transcription error in it would be
invisible -- a wrong R0 produces a perfectly plausible bond-valence sum. The
estimator is an independent route to the same quantity, so requiring the two to
agree to within 0.20 A catches a mistyped digit without requiring the estimator
itself to be accurate.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from facet.core import bv, elements

# Pairs where the O'Keeffe-Brese electronegativity estimate is known to be poor
# and the fitted value is nonetheless right. These are chemistry, not typos, and
# they are exactly the cases where FACET's "estimated" label earns its keep.
ESTIMATOR_KNOWN_POOR = {
    # d9 Cu(II) is Jahn-Teller distorted: 4 short bonds plus 2 long ones. The
    # estimator, which knows only electronegativity and size, has no way to see
    # that and returns an R0 about 0.21 A too short. Fitted value 1.679 is the
    # standard one (Brown & Altermatt 1985; Brese & O'Keeffe 1991).
    ("Cu", 2, "O"),
}


# --- the model itself --------------------------------------------------------

def test_valence_and_cutoff_are_inverse():
    r0, b = 2.09, 0.37
    for v in (0.02, 0.075, 0.2, 1.0):
        d = bv.cutoff_for_valence(r0, v, b)
        assert bv.valence(d, r0, b) == pytest.approx(v, rel=1e-12)


def test_valence_at_r0_is_unity():
    assert bv.valence(2.09, 2.09) == pytest.approx(1.0)


def test_documented_cutoff_arithmetic():
    """b*ln(1/0.075) = 0.958 A and b*ln(1/0.02) = 1.447 A, as documented."""
    b = 0.37
    assert -b * math.log(0.075) == pytest.approx(0.958, abs=0.001)
    assert -b * math.log(0.02) == pytest.approx(1.447, abs=0.001)


def test_bi_o_cutoffs_reproduce_the_literature_conventions():
    """Cutting Bi-O at the default valences must land on the conventional
    3.05 / 3.54 A, which is what justifies using the same valence elsewhere."""
    p = bv.DEFAULT.get("Bi", 3, "O")
    assert p.fitted and p.r0 == 2.09
    assert p.distance_for(bv.V_BOND_DEFAULT) == pytest.approx(3.05, abs=0.01)
    assert p.distance_for(bv.V_LIST_DEFAULT) == pytest.approx(3.54, abs=0.01)


def test_bi_i_cutoff_is_far_beyond_any_oxygen_convention():
    """The point of cutting by valence: Bi-I lands at 3.72 / 4.21 A."""
    p = bv.DEFAULT.get("Bi", 3, "I")
    assert p.distance_for(bv.V_BOND_DEFAULT) == pytest.approx(3.72, abs=0.02)
    assert p.distance_for(bv.V_LIST_DEFAULT) == pytest.approx(4.21, abs=0.02)


# --- phi ---------------------------------------------------------------------

def test_phi_vanishes_for_a_regular_octahedron():
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    _, phi, _ = bv.phi_index(v, np.ones(6))
    assert phi == pytest.approx(0.0, abs=1e-12)


def test_phi_is_one_for_a_single_bond():
    _, phi, _ = bv.phi_index(np.array([[0, 0, 1.0]]), [0.5])
    assert phi == pytest.approx(1.0)


def test_phi_is_exactly_invariant_to_an_error_in_r0():
    """The property that makes phi, not |BVV|, the reportable quantity.

    Shifting R0 scales every v_i by a common exp(delta/b), which cancels in the
    ratio. A 0.40 A error moves |BVV| by nearly a factor three and leaves phi
    unchanged to machine precision.
    """
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(7, 3))
    d = 2.0 + rng.random(7)
    r0, b = 2.09, 0.37

    mag_a, phi_a, _ = bv.phi_index(vecs, bv.valence(d, r0, b))
    mag_b, phi_b, _ = bv.phi_index(vecs, bv.valence(d, r0 + 0.40, b))

    assert phi_b == pytest.approx(phi_a, rel=1e-12)
    assert mag_b / mag_a == pytest.approx(math.exp(0.40 / b), rel=1e-12)
    assert mag_b / mag_a > 2.9          # and it really is a large shift


def test_phi_handles_the_empty_case():
    mag, phi, vec = bv.phi_index(np.zeros((0, 3)), [])
    assert mag == 0.0 and math.isnan(phi) and vec.shape == (3,)


# --- the fitted table --------------------------------------------------------

def test_fitted_table_agrees_with_estimator():
    """Every hand-typed R0 must be within 0.20 A of the independent estimate.

    Not a test of the estimator's accuracy -- the tolerance is far wider than
    the estimator's typical 0.03-0.08 A error. It is a transcription check: a
    dropped or transposed digit moves R0 by a lot more than 0.20 A.
    """
    if not bv._okb_params():
        pytest.skip("O'Keeffe-Brese parameters unavailable (pymatgen missing)")

    bad = []
    for (cation, ox, anion), r0 in bv._FITTED.items():
        if (cation, ox, anion) in ESTIMATOR_KNOWN_POOR:
            continue
        est = bv.estimate_r0(cation, anion)
        if est is None:
            continue
        if abs(est - r0) > 0.20:
            bad.append(f"{cation}{ox:+d}-{anion}: table {r0:.3f}, "
                       f"estimate {est:.3f}, diff {est - r0:+.3f}")
    assert not bad, "fitted values far from the estimate:\n  " + "\n  ".join(bad)


def test_known_poor_estimates_really_are_poor():
    """Guard the exemption list: if the estimator ever agrees for one of these,
    the exemption is stale and should be removed rather than left standing."""
    if not bv._okb_params():
        pytest.skip("O'Keeffe-Brese parameters unavailable")
    for key in ESTIMATOR_KNOWN_POOR:
        cation, ox, anion = key
        est = bv.estimate_r0(cation, anion)
        assert est is not None
        assert abs(est - bv._FITTED[key]) > 0.20, (
            f"{cation}{ox:+d}-{anion} now agrees with the estimator; "
            "drop it from ESTIMATOR_KNOWN_POOR")


def test_fitted_values_are_physically_plausible():
    for (cation, ox, anion), r0 in bv._FITTED.items():
        assert 1.0 < r0 < 3.5, f"{cation}{ox:+d}-{anion} R0 = {r0}"


def test_fitted_beats_estimated_and_both_are_labelled():
    fitted = bv.DEFAULT.get("Si", 4, "O")
    assert fitted.fitted and "Brese" in fitted.source

    # Rh-O has no fitted entry here, so the estimator must cover it and say so
    est = bv.DEFAULT.get("Rh", 3, "O")
    assert est is not None and not est.fitted and "estimated" in est.source


def test_estimator_can_be_refused():
    strict = bv.ParameterSet(allow_estimated=False)
    assert strict.get("Si", 4, "O") is not None
    assert strict.get("Rh", 3, "O") is None


def test_user_override_is_recorded_as_such():
    ps = bv.ParameterSet()
    ps.override("Bi", 3, "O", 2.094, source="Brown & Altermatt 1985")
    p = ps.get("Bi", 3, "O")
    assert p.r0 == 2.094 and p.fitted and "Altermatt" in p.source
    ps.clear_overrides()
    assert ps.get("Bi", 3, "O").r0 == 2.09


def test_unknown_element_returns_none_rather_than_guessing():
    assert bv.DEFAULT.get("Xx", 3, "O") is None


# --- GII ---------------------------------------------------------------------

def test_gii_is_rms_of_discrepancies():
    assert bv.global_instability_index([0.1, -0.1]) == pytest.approx(0.1)
    assert bv.global_instability_index([0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert math.isnan(bv.global_instability_index([]))


# --- element helpers ---------------------------------------------------------

@pytest.mark.parametrize("label,expect", [
    ("Na", "Na"), ("Na1", "Na"), ("Na+", "Na"), ("Na+1", "Na"), ("Na1+", "Na"),
    ("NA", "Na"), ("O", "O"), ("O2-", "O"), ("Bi3+", "Bi"), ("BI1", "Bi"),
    ("Si4+", "Si"), ("c", "C"),
])
def test_label_normalisation(label, expect):
    assert elements.normalise(label) == expect


def test_anion_assignment_is_by_electronegativity():
    assert elements.more_electronegative("Bi", "O") == "O"
    assert elements.more_electronegative("Na", "F") == "F"
    assert elements.more_electronegative("Si", "O") == "O"
    assert elements.is_anion_like("O") and not elements.is_anion_like("Na")


def test_family_is_used_only_for_defaults_but_must_be_right():
    assert elements.family("Bi", 3) == "lone-pair"
    assert elements.family("Bi", 5) != "lone-pair"
    assert elements.family("Pb", 2) == "lone-pair"
    assert elements.family("W", 6) == "d0-transition-metal"
    assert elements.family("Na", 1) == "alkali"
    assert elements.family("O", -2) == "anion"
