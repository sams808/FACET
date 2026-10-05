"""The torsion angle, against the convention and against itself.

FACET reported the supplement of the torsion angle from the beginning, in two
places: `utilities.torsions`, which nothing called, and the viewport's
four-atom measurement, which was the same eight lines written out again. Anti
read as 0 and eclipsed as 180. Nothing caught it, because there was no test and
because the two copies agreed with each other.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import sample_cif

from facet.core.utilities import torsion_angle

# The central bond along +x, with the two outer atoms placed at chosen
# azimuths about it, so that the torsion is the azimuth difference by
# construction rather than by calculation.
B = np.array([0.0, 0.0, 0.0])
C = np.array([1.5, 0.0, 0.0])


def _outer(base, sign, azimuth_degrees):
    t = np.radians(azimuth_degrees)
    return base + np.array([sign * 0.6, np.cos(t), np.sin(t)])


def _chain(azimuth):
    return _outer(B, -1, 0.0), B, C, _outer(C, +1, azimuth)


@pytest.mark.parametrize("azimuth,expected,name", [
    (0.0, 0.0, "eclipsed"),
    (60.0, 60.0, "gauche +"),
    (-60.0, -60.0, "gauche -"),
    (90.0, 90.0, "perpendicular"),
    (120.0, 120.0, "anticlinal"),
    (180.0, 180.0, "anti"),
])
def test_the_named_conformations(azimuth, expected, name):
    """The values every textbook states.

    Anti is 180 and eclipsed is 0. FACET had them the other way round: its
    formula took the angle between the two plane normals, which is the
    supplement -- expanding (a x b).(c x d) gives n1.n2 = -|b1|^2 (v.w), so the
    cosine was negated and every angle came back as sign(t) * (180 - |t|).
    """
    assert torsion_angle(*_chain(azimuth)) == pytest.approx(expected, abs=1e-9)


def test_it_agrees_with_an_independent_implementation():
    """Straight from the definition, on random geometry.

    The signed angle from the projection of B->A to the projection of C->D
    about the B-C axis, built without any cross-product identity. Over two
    thousand quadruples the two must not differ at all.
    """
    def from_the_definition(p0, p1, p2, p3):
        axis = np.asarray(p2, float) - np.asarray(p1, float)
        axis = axis / np.linalg.norm(axis)
        a = np.asarray(p0, float) - np.asarray(p1, float)
        d = np.asarray(p3, float) - np.asarray(p2, float)
        a_perp = a - np.dot(a, axis) * axis
        d_perp = d - np.dot(d, axis) * axis
        a_perp /= np.linalg.norm(a_perp)
        d_perp /= np.linalg.norm(d_perp)
        return float(np.degrees(np.arctan2(
            float(np.dot(np.cross(a_perp, d_perp), axis)),
            float(np.clip(np.dot(a_perp, d_perp), -1, 1)))))

    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(2000):
        points = rng.normal(size=(4, 3))
        difference = ((torsion_angle(*points) - from_the_definition(*points)
                       + 180) % 360) - 180
        worst = max(worst, abs(difference))
    assert worst < 1e-9, f"worst disagreement {worst} degrees"


def test_reversing_the_chain_keeps_the_angle():
    """D-C-B-A is the same dihedral as A-B-C-D, sign included."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        p = rng.normal(size=(4, 3))
        assert torsion_angle(*p[::-1]) == pytest.approx(torsion_angle(*p),
                                                        abs=1e-9)


def test_mirroring_the_chain_reverses_the_sign():
    """A torsion is chiral: reflect the four points and it changes sign.

    The property that distinguishes a signed torsion from an unsigned angle,
    and the one a formula that lost a sign would break.
    """
    rng = np.random.default_rng(11)
    for _ in range(200):
        p = rng.normal(size=(4, 3))
        mirrored = p * np.array([1.0, 1.0, -1.0])
        assert torsion_angle(*mirrored) == pytest.approx(
            -torsion_angle(*p), abs=1e-9)


def test_collinear_points_do_not_raise():
    """Four points on a line have no dihedral; they must not crash one out."""
    line = [np.array([float(i), 0.0, 0.0]) for i in range(4)]
    value = torsion_angle(*line)
    assert np.isfinite(value)

    degenerate = [np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3)]
    assert torsion_angle(*degenerate) == 0.0


def test_the_structure_form_and_the_point_form_are_the_same_function():
    """`torsions` takes atom indices and must not be a second implementation.

    It was one, and so was the viewport's measurement: three copies of eight
    lines, all agreeing on the wrong answer.
    """
    from pathlib import Path

    from facet.core import cif, utilities

    sample = sample_cif("1004091", "1004091_BiNa3O8P2.cif")
    if not sample.is_file():
        pytest.skip("the reference structure is not present")
    structure = cif.read(sample)
    indices = (0, 1, 2, 3)
    points = [structure.atoms[i].cart for i in indices]
    assert utilities.torsions(structure, *indices) == pytest.approx(
        torsion_angle(*points), abs=1e-12)


def test_the_viewport_reports_the_same_angle():
    """The measurement a user reads off the screen."""
    import re

    from facet.gl.scene import Scene
    from facet.gl.view import StructureView

    points = np.array([_outer(B, -1, 0.0), B, C, _outer(C, +1, 60.0)])
    scene = Scene()
    scene.atom_position = points.astype(np.float32)
    scene.atom_label = ["A", "B", "C", "D"]

    widget = StructureView.__new__(StructureView)     # no GL context needed
    widget.scene = scene
    widget._measure_chain = [0, 1, 2, 3]
    text = widget._describe_measurement()

    assert "torsion" in text
    value = float(re.search(r"(-?\d+\.\d+)\s*°", text).group(1))
    assert value == pytest.approx(60.0, abs=1e-6)
