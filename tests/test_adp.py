"""Anisotropic displacement parameters: the tensor, and what it says.

The conversion from the CIF's six numbers to a Cartesian tensor is the kind of
arithmetic that is wrong silently. A swapped pair of off-diagonal components
leaves U_eq exactly right, because U_eq is a trace, and tilts every ellipsoid;
a missing reciprocal-length factor scales the whole tensor by a constant, which
in a cell whose axes are similar looks entirely plausible. So these check
against things outside the code: gemmi's own formula for U_eq, invariance under
a relabelling of the axes, the isotropic case in four crystal systems, and the
tabulated probability factors every drawing program quotes.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import adp
from facet.core.readers import cell_from_parameters

ANISO = sample_cif("1004091", "1004091_BiNa3O8P2.cif")
NPD = sample_cif("7023720", "7023720_BiPO4.cif")


# ---------------------------------------------------------------------------
# the conversion
# ---------------------------------------------------------------------------

def test_an_orthogonal_cell_leaves_the_components_alone():
    """The identity that pins the component order.

    With orthogonal axes the reciprocal lengths are the inverses of the direct
    ones, so the transform is exactly the identity and the CIF components *are*
    the Cartesian ones. Six distinct values therefore say precisely where each
    one belongs -- which is what catches a swap of U13 and U23, the error that
    Voigt order would introduce and that leaves U_eq correct.
    """
    orth = np.diag([7.0, 11.0, 13.0])
    u = np.array([0.011, 0.022, 0.033, 0.0044, 0.0055, 0.0066])
    out = adp.cartesian_tensor(orth, u)

    assert out[0, 0] == pytest.approx(0.011)
    assert out[1, 1] == pytest.approx(0.022)
    assert out[2, 2] == pytest.approx(0.033)
    assert out[0, 1] == pytest.approx(0.0044)     # U12
    assert out[0, 2] == pytest.approx(0.0055)     # U13, not U23
    assert out[1, 2] == pytest.approx(0.0066)     # U23
    assert out == pytest.approx(out.T)


def test_an_oblique_cell_does_not_leave_them_alone():
    """The transform has to be doing something, or the test above proves nothing."""
    cell = cell_from_parameters(19.86, 5.353, 13.96, 90, 110.64, 90)
    u = np.array([0.011, 0.022, 0.033, 0.0044, 0.0055, 0.0066])
    raw = np.array([[u[0], u[3], u[4]], [u[3], u[1], u[5]], [u[4], u[5], u[2]]])
    out = adp.cartesian_tensor(cell.orth, u)
    assert np.abs(out - raw).max() > 1e-4


def test_u_equivalent_against_gemmi():
    """gemmi reaches the same number without building a Cartesian tensor.

    Its ``calculate_u_eq`` sums U_ij a*_i a*_j (a_i . a_j) / 3 directly. Two
    implementations of different formulae agreeing to machine precision is what
    makes this a check rather than a restatement.
    """
    gemmi = pytest.importorskip("gemmi")
    if not ANISO.is_file():
        pytest.skip("the reference structure is not present")
    from facet.core import cif

    ours = cif.read(ANISO)
    theirs = gemmi.read_small_structure(str(ANISO))
    by_label = {s.label: s for s in ours.sites}

    checked = 0
    for site in theirs.sites:
        mine = by_label.get(site.label)
        if mine is None or mine.u_aniso is None:
            continue
        shape = adp.for_site(ours.cell, mine)
        assert shape.u_equivalent == pytest.approx(
            theirs.cell.calculate_u_eq(site.aniso), rel=1e-12)
        checked += 1
    assert checked >= 4, f"only {checked} sites carried a tensor"


@pytest.mark.parametrize("system,parameters", [
    ("cubic", (5.0, 5.0, 5.0, 90, 90, 90)),
    ("hexagonal", (4.0, 4.0, 6.0, 90, 90, 120)),
    ("monoclinic", (19.86, 5.353, 13.96, 90, 110.64, 90)),
    ("triclinic", (6.1, 7.3, 9.7, 78.0, 84.5, 101.2)),
])
def test_an_isotropic_tensor_comes_back_spherical(system, parameters):
    """A sphere is a sphere in any basis, and this is the whole transform.

    The six components are built by carrying an isotropic Cartesian tensor
    *back* to the CIF basis, so a transform that is wrong in one direction
    cannot cancel itself: the test uses the inverse and the forward map.
    """
    cell = cell_from_parameters(*parameters)
    u_iso = 0.017
    m = cell.orth @ np.diag(adp.reciprocal_lengths(cell.orth))
    inverse = np.linalg.inv(m)
    matrix = inverse @ (np.eye(3) * u_iso) @ inverse.T
    six = np.array([matrix[0, 0], matrix[1, 1], matrix[2, 2],
                    matrix[0, 1], matrix[0, 2], matrix[1, 2]])

    shape = adp.ellipsoid(cell.orth, six)
    assert shape.rms == pytest.approx(math.sqrt(u_iso), abs=1e-12)
    assert shape.anisotropy == pytest.approx(1.0, abs=1e-9)
    assert shape.u_equivalent == pytest.approx(u_iso, rel=1e-12)


def test_relabelling_the_axes_does_not_change_the_shape():
    """The tensor is a physical object; the axis names are not.

    Orthorhombic, so a permutation of a, b, c is an exact relabelling of the
    same crystal and the principal displacements must come back identical.
    """
    lengths = np.array([7.0, 11.0, 13.0])
    diagonal = np.array([0.010, 0.020, 0.030])
    base = adp.ellipsoid(np.diag(lengths),
                         np.concatenate([diagonal, np.zeros(3)]))
    for perm in ((1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)):
        order = list(perm)
        other = adp.ellipsoid(
            np.diag(lengths[order]),
            np.concatenate([diagonal[order], np.zeros(3)]))
        assert np.sort(other.rms) == pytest.approx(np.sort(base.rms))
        assert other.u_equivalent == pytest.approx(base.u_equivalent)


# ---------------------------------------------------------------------------
# the shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("probability,published", [
    (0.50, 1.5382), (0.90, 2.5003), (0.95, 2.7955), (0.99, 3.3682)])
def test_the_probability_factors_match_the_published_table(probability,
                                                           published):
    """The values every crystallographic drawing program quotes."""
    assert adp.scale_for(probability) == pytest.approx(published, abs=5e-5)


def test_a_fifty_per_cent_ellipsoid_is_larger_than_the_rms_surface():
    """Worth pinning, because the intuition runs the other way.

    "50%" sounds like half of something. It is the surface enclosing half the
    probability, and for three degrees of freedom that is half again as far out
    as the r.m.s. displacement.
    """
    assert adp.scale_for(0.50) > 1.5
    assert adp.scale_for(0.10) < adp.scale_for(0.50) < adp.scale_for(0.99)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_a_probability_outside_the_range_is_refused(bad):
    with pytest.raises(ValueError):
        adp.scale_for(bad)


def test_the_transform_carries_a_unit_sphere_onto_the_ellipsoid():
    """What the renderer will multiply its sphere vertices by."""
    orth = np.diag([7.0, 11.0, 13.0])
    shape = adp.ellipsoid(orth, np.array([0.01, 0.04, 0.09, 0, 0, 0]))
    matrix = shape.transform(0.50)

    rng = np.random.default_rng(0)
    points = rng.normal(size=(500, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    mapped = points @ matrix.T

    # every mapped point is on the surface: x^T U^-1 x = scale^2
    inverse = np.linalg.inv(shape.u_cart)
    quadratic = np.einsum("ij,jk,ik->i", mapped, inverse, mapped)
    assert quadratic == pytest.approx(adp.scale_for(0.50) ** 2, rel=1e-9)

    # and the semi-axes are the radii, in ascending order
    assert np.sort(np.linalg.norm(matrix, axis=0)) == pytest.approx(
        np.sort(shape.radii(0.50)))


def test_an_isotropic_sphere_is_offered_as_the_same_type():
    shape = adp.sphere(0.02)
    assert shape.is_ellipsoid
    assert shape.anisotropy == pytest.approx(1.0)
    assert shape.rms == pytest.approx(math.sqrt(0.02))
    assert shape.u_equivalent == pytest.approx(0.02)


def test_a_site_without_a_tensor_gets_none_rather_than_a_sphere():
    """The two cases have to stay distinguishable.

    A file that refined every site anisotropically says something different
    from one that gave only U_iso, and turning the second into a sphere here
    would be a drawing decision dressed up as a measurement.
    """
    from facet.core.structure import Site

    cell = cell_from_parameters(5, 5, 5, 90, 90, 90)
    assert adp.for_site(cell, Site("O1", "O", [0, 0, 0], u_iso=0.02)) is None
    assert adp.for_site(cell, Site("O1", "O", [0, 0, 0],
                                   u_aniso=np.zeros(6))) is None


# ---------------------------------------------------------------------------
# what it says about the file
# ---------------------------------------------------------------------------

def test_a_negative_eigenvalue_is_reported_and_has_no_rms():
    orth = np.diag([5.0, 5.0, 5.0])
    shape = adp.ellipsoid(orth, np.array([0.005, 0.006, 0.0001, 0, -0.0018, 0]))
    assert not shape.is_ellipsoid
    assert shape.anisotropy == float("inf")
    assert np.isnan(shape.rms).any()
    # U_eq is still a trace and still computable, which is exactly why it
    # cannot be used to detect this
    assert shape.u_equivalent > 0


def test_the_sign_of_the_eigenvalues_does_not_depend_on_the_cell():
    """Sylvester's law of inertia, which is why the health check needs no transform.

    ``U_cart = M U M^T`` is a congruence, so it cannot change how many
    eigenvalues are positive, negative or zero. A non-positive-definite tensor
    is a fact about the file, not about the conversion.
    """
    u = np.array([0.005, 0.006, 0.0001, 0.0, -0.0018, 0.0])
    raw = np.array([[u[0], u[3], u[4]], [u[3], u[1], u[5]], [u[4], u[5], u[2]]])
    signs = np.sign(np.linalg.eigvalsh(raw))
    for parameters in ((5, 5, 5, 90, 90, 90),
                       (19.86, 5.353, 13.96, 90, 110.64, 90),
                       (6.1, 7.3, 9.7, 78.0, 84.5, 101.2)):
        cell = cell_from_parameters(*parameters)
        assert np.array_equal(
            np.sign(adp.ellipsoid(cell.orth, u).eigenvalues), signs)


def test_the_health_check_finds_the_npd_site_in_a_real_file():
    """A structure in the reference collection with a genuine NPD atom.

    7023720 refines P1 with U_33 = 0.0001 and U_13 = -0.0018, which makes the
    tensor indefinite in the file's own components before any conversion.
    """
    if not NPD.is_file():
        pytest.skip("the reference structure is not present")
    from facet.core import cif, quality

    report = quality.check(cif.read(NPD))
    npd = [f for f in report.findings if f.code == "npd-displacement"]
    assert len(npd) == 1
    assert npd[0].where == "P1"
    assert npd[0].level == quality.Level.IMPOSSIBLE
    assert npd[0].value < 0


def test_the_health_check_says_nothing_about_a_sound_structure():
    """No finding at all where the displacement parameters are ordinary."""
    if not ANISO.is_file():
        pytest.skip("the reference structure is not present")
    from facet.core import cif, quality

    report = quality.check(cif.read(ANISO))
    assert [f for f in report.findings if "displacement" in f.code] == []


def test_the_health_check_reports_and_does_not_judge():
    """The standing rule: measurements, never verdicts."""
    if not NPD.is_file():
        pytest.skip("the reference structure is not present")
    from facet.core import cif, quality

    report = quality.check(cif.read(NPD))
    banned = ("bad", "good", "wrong", "unusable", "unreliable", "should",
              "must be", "invalid", "poor", "suspicious")
    for finding in report.findings:
        low = finding.message.lower()
        assert not any(word in low for word in banned), finding.message
