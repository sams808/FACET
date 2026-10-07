"""Links from an MD model to NMR and EXAFS (facet.core.md_spectroscopy).

What is pinned, and against what:

* **The comparison table is arithmetic, and nothing else.** model - measured
  key by key, the model mean and spread checked against Python's
  ``statistics`` module, a measured key with no model counterpart refused, a
  model key with no measurement left empty (not 0), measured sets that do
  not sum to 1 refused with the sum and the tolerance. N4 renormalised over
  B3 + B4 equals the ratio formed by hand from the counts.
* **No correlation ships, and none computes without its reference.** A
  correlation lacking a coefficient, its intercept, any term or its
  reference is refused with 'TODO: need reference'; the module source holds
  no Correlation or Term built with numbers.
* **The descriptors are the geometry.** On a hand-built corner-sharing Si2O7
  dimer the bridge angle, Qn, CN, BVS and bond length come out as built, and
  on the quartz supercell every Si-O-Si angle equals one computed by an
  independent brute-force search (nearest two Si of each O over 27 images,
  arccos), to 1e-9 degrees.
* **The law of the spectrum.** Each line has unit area: on a wide grid a
  Gaussian spectrum integrates to the number of atoms with a shift, the
  exact in-grid area of a Lorentzian is (2/pi) arctan(L/gamma), and the
  half maximum sits at +-FWHM/2 for both shapes. Atoms outside a stated
  range are counted and their part of the spectrum integrates to their
  number.
* **Cumulants against known distributions.** Gauss-Hermite nodes and
  weights reproduce a Gaussian's moments exactly (C3 = C4 = 0); generalised
  Gauss-Laguerre nodes reproduce a gamma distribution's (C2 = k, C3 = 2k,
  C4 = 6k, scaled); random samples agree with directly computed central
  moments.
* **The EXAFS partials are the pair search's.** The absorber-centred
  distances equal ``pdf.pair_distances`` on the same frame, the deposited
  histogram conserves the pair count, N(r) is an exact count, the quartz
  Si-O first minimum lies in the empty gap after the first shell, and N = 4
  with a spread of exactly 0 over two identical frames.
* **The FEFF input is write_feff's.** For a tiny frame the POTENTIALS and
  ATOMS blocks equal those ``exporters.write_feff`` writes from
  ``frame_to_structure`` of the same frame, and the format rules of
  tests/test_exafs.py hold. FEFF is not run here (it was run outside the
  suite; the module docstring gives the result). chi(k) averaged from
  FEFF-format path files equals the mean of ``exafs.chi_from_paths``.
* **No verdicts and no Qt.**

Pinned after an independent review (each test fails on the module before
that review's fixes):

* **Rounding cannot leave [0, 1], and a stated rounding has a bound.** A
  value of -0.01 or 1.01 is refused whatever ``sum_tol`` is; a ``sum_tol``
  wider than n x half the finest decimal place of the values is refused; a
  complete measured set that leaves out a model category of non-zero mean is
  noted as a comparison of different totals; the spread note says where the
  model std came from.
* **The inputs are this frame's and mean something.** Bonds of another frame
  (same atom count) are refused by recomputing their vectors; a 'user' term
  named like a computed descriptor, 'bridges to' a non-former, 'bonds to' a
  partner on the same side of every bond, and a bridge descriptor in a
  frame without the bridging anion are refused; the placeholder 'TODO: need
  reference' is not a reference; a negative power is allowed and a value of
  0 under it is undefined and counted.
* **The spectrum says how its grid samples it.** A grid step at or above
  the FWHM is noted with the area difference; shifts beyond the grid are
  counted; a decreasing grid is refused with the reversal named; the
  histogram refuses mixed correlations; provenance travels with every
  result that can be exported, or its absence is noted.
* **One first-minimum rule for one g(r).** md_exafs takes the glass module's
  ``MinimumMethod`` and returns glass.first_minimum's cutoff; on a noisy
  synthetic first shell the 'valley' rule keeps all four neighbours, the
  'first local minimum' rule stops inside the peak and says that g is not
  below 1 there.
* **Shells.** A repeated shell is refused, a user shell equal to an
  automatic first shell is kept as a second shell, and every radius past
  half the box is refused before any pair search; pass 1 searches to the
  FEFF cluster radius only in a frame that holds a cluster.
* **FEFF files.** A second draw into an out_dir is refused, overwrite
  removes the old FEFF output and notes the stray directories, the feff.inp
  header names its source and no bond-valence setting, more than 10
  potentials is commented; average_feff_chi sums the paths of chi.dat's
  header by default and leaves out incomplete, mixed or stale runs.

Pinned after a second review (each test fails on the module before it):

* **The last g(r) point holds the pairs one step past it**, which md_exafs
  had not searched for (a real glass's Al-O g ended at half its value);
  pass 1 searches only as far as the g(r) grid needs when the shells wait
  for pass 2, and a MinimumMethod that located no limit is not recorded as
  used.
* **A limit before the main peak is noted** with both maxima.
* **A FEFF run counts once**, and a run with no feff.inp beside it is named
  as unchecked.

Every coefficient here is a synthetic test input, not a published value.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from facet.core import bulk, bv, cif, exafs, exporters, glass, md_model, pdf
from facet.core import md_spectroscopy as ms
from facet.core.md_stats import Distribution, Provenance

# The first-minimum methods the tests use: rules of glass.MinimumMethod,
# chosen per test (no default rule exists). FIRST_LOCAL compares the values
# as they stand (margin 0), the reading of a noise-free g(r); VALLEY and
# LOCAL_2SE ask a difference to exceed 1 and 2 counting errors of g.
FIRST_LOCAL = glass.MinimumMethod("first local minimum", None, "midpoint",
                                  0.0)
VALLEY = glass.MinimumMethod("valley", None, "midpoint", 1.0)
LOCAL_2SE = glass.MinimumMethod("first local minimum", None, "midpoint", 2.0)

ROOT = Path(__file__).resolve().parent.parent
QUARTZ = ROOT / "tests" / "data" / "crystals" / "quartz_SiO2_cod9013321.cif"


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def quartz():
    return cif.read(str(QUARTZ))


@pytest.fixture(scope="module")
def quartz_frame(quartz):
    frame, _ = md_model.supercell_frame(quartz, (2, 2, 2))
    return frame


def _bonds(frame):
    ox = md_model.model_oxidation(frame.species)
    table, _ = bulk.analyse_frame(frame, ox.per_atom(frame.elements))
    return bulk.bonds_at(table)


@pytest.fixture(scope="module")
def quartz_bonds(quartz_frame):
    return _bonds(quartz_frame)


def _tetrahedral_directions(a):
    """The three directions at the tetrahedral angle (cos = -1/3) from a."""
    a = a / np.linalg.norm(a)
    p = np.cross(a, [0.0, 0.0, 1.0])
    p /= np.linalg.norm(p)
    q = np.cross(a, p)
    return [-a / 3.0 + (2.0 * math.sqrt(2.0) / 3.0)
            * (math.cos(phi) * p + math.sin(phi) * q)
            for phi in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)]


THETA_DEG = 150.0          # the built bridge angle: a test input


@pytest.fixture(scope="module")
def dimer():
    """Two SiO4 tetrahedra sharing one O at THETA_DEG, alone in a 30 Å box.

    The Si-O length is where the bond valence of Si4+-O is exactly 1 v.u. in
    FACET's parameter set (distance_for(1.0)), so no length is typed in.
    """
    d = bv.DEFAULT.get("Si", 4, "O").distance_for(1.0)
    half = math.radians(THETA_DEG) / 2.0
    centre = np.array([15.0, 15.0, 15.0])
    u = [np.array([-math.sin(half), -math.cos(half), 0.0]),
         np.array([math.sin(half), -math.cos(half), 0.0])]
    symbols, positions = ["O"], [centre]
    for direction in u:
        silicon = centre + d * direction
        symbols.append("Si")
        positions.append(silicon)
        for w in _tetrahedral_directions(-direction):
            symbols.append("O")
            positions.append(silicon + d * w)
    frame = md_model.frame_from_arrays(symbols, np.array(positions),
                                       box_ang=np.eye(3) * 30.0)
    return frame, d


# ---------------------------------------------------------------------------
# 1. fractions beside an NMR fit
# ---------------------------------------------------------------------------

def _qn_distribution():
    counts = [{3: 10, 4: 90}, {3: 12, 4: 88}, {3: 17, 4: 83}]
    return Distribution.from_counts(counts, name="Si Qn", kind="fraction",
                                    keys=(2, 3, 4)), counts


def test_the_table_is_model_minus_measured_key_by_key():
    dist, counts = _qn_distribution()
    measured = ms.MeasuredFractions(
        "Si Qn", {3: 0.10, 4: 0.90}, "29Si MAS fit (test input)",
        uncertainties={4: 0.02})
    table = ms.compare_fractions(dist, measured)
    assert table.keys == (2, 3, 4)
    for c, key in enumerate(table.keys):
        per_frame = [row.get(key, 0) / 100 for row in counts]
        assert table.model_mean[c] == pytest.approx(
            statistics.fmean(per_frame), abs=1e-15)
        assert table.model_std[c] == pytest.approx(
            statistics.stdev(per_frame), abs=1e-15)
    assert np.isnan(table.measured[0]) and np.isnan(table.difference[0])
    assert table.difference[1] == table.model_mean[1] - 0.10
    assert table.difference[2] == table.model_mean[2] - 0.90
    assert np.isnan(table.uncertainty[1]) and table.uncertainty[2] == 0.02
    assert table.model_frames == 3
    rows = table.as_rows()
    assert [r["key"] for r in rows] == ["2", "3", "4"]
    assert rows[2]["model - measured"] == table.difference[2]
    assert rows[0]["measured source"] == "29Si MAS fit (test input)"
    assert any("not as 0" in n for n in table.notes)
    assert any("not combined" in n for n in table.notes)


def test_a_measured_key_with_no_model_counterpart_is_refused():
    dist, _ = _qn_distribution()
    measured = ms.MeasuredFractions("Si Qn", {"Q3": 0.1, "Q4": 0.9}, "fit")
    with pytest.raises(ValueError, match=r"Q3, Q4 match no key .*2, 3, 4"):
        ms.compare_fractions(dist, measured)


def test_measured_sets_that_do_not_sum_to_one_are_refused():
    with pytest.raises(ValueError, match=r"0\.9.*tolerance 1e-12"):
        ms.MeasuredFractions("B N4", {"B3": 0.5, "B4": 0.4}, "fit")
    # rounded values from a paper: refused at the default, accepted at the
    # stated rounding, and the comparison says so
    with pytest.raises(ValueError):
        ms.MeasuredFractions("Al CN", {4: 0.33, 5: 0.33, 6: 0.33}, "paper")
    # three values rounded to 0.01 each: up to 3 x 0.005 off
    rounded = ms.MeasuredFractions("Al CN", {4: 0.33, 5: 0.33, 6: 0.33},
                                   "paper", sum_tol=0.015)
    model = {4: 1 / 3, 5: 1 / 3, 6: 1 / 3}
    out = ms.compare_fractions(model, rounded, model_name="Al CN model")
    assert any("stated rounding 0.015" in n for n in out.notes)
    # a partial set (N4 alone) is bounded, not summed to 1
    partial = ms.MeasuredFractions("B N4", {"B4": 0.42}, "11B MAS fit",
                                   complete=False)
    assert partial.values == {"B4": 0.42}
    with pytest.raises(ValueError, match="above 1"):
        ms.MeasuredFractions("B", {"B3": 0.7, "B4": 0.6}, "fit",
                             complete=False)
    for bad in (dict(values={}, source="s"),
                dict(values={"a": 1.0}, source=""),
                dict(values={"a": 1.0}, source="s", uncertainties={"b": 0.1}),
                dict(values={"a": 1.0}, source="s", uncertainties={"a": -0.1}),
                dict(values={"a": float("nan")}, source="s")):
        with pytest.raises(ValueError):
            ms.MeasuredFractions("d", **bad)


def test_the_model_side_can_be_a_mapping():
    measured = ms.MeasuredFractions("O", {"BO": 0.6, "NBO": 0.4}, "17O fit")
    plain = ms.compare_fractions({"BO": 0.75, "NBO": 0.25}, measured)
    assert plain.difference.tolist() == [0.75 - 0.6, 0.25 - 0.4]
    assert np.isnan(plain.model_std).all() and plain.model_frames is None
    paired = ms.compare_fractions({"BO": (0.75, 0.01), "NBO": (0.25, 0.01)},
                                  measured)
    assert paired.model_std.tolist() == [0.01, 0.01]
    with pytest.raises(ValueError, match="sum"):
        ms.compare_fractions({"BO": 0.75, "NBO": 0.2}, measured)
    with pytest.raises(ValueError, match="mixture"):
        ms.compare_fractions({"BO": 0.75, "NBO": (0.25, 0.0)}, measured)
    rows = ms.comparison_rows([plain, paired])
    assert len(rows) == 4 and rows[0]["descriptor"] == "O"


def test_n4_renormalised_over_b3_and_b4_is_the_nmr_ratio():
    counts = [{"B3": 6, "B4": 3, "other": 1}, {"B3": 5, "B4": 5}]
    dist = Distribution.from_counts(counts, name="B CN", kind="fraction",
                                    keys=("B3", "B4", "other"))
    n4 = ms.renormalised(dist, ["B3", "B4"])
    expected = statistics.fmean([3 / 9, 5 / 10])
    assert n4.keys == ("B3", "B4")
    assert n4.mean[1] == pytest.approx(expected, abs=1e-15)
    assert n4.per_frame[0].tolist() == pytest.approx([6 / 9, 3 / 9])
    assert any("other (mean fraction 0.05" in n for n in n4.notes)
    # the same from the count distribution
    from_counts = ms.renormalised(Distribution.from_counts(
        counts, name="B CN", kind="count", keys=("B3", "B4", "other")),
        ["B3", "B4"])
    assert from_counts.mean.tolist() == n4.mean.tolist()
    with pytest.raises(ValueError, match="no key"):
        ms.renormalised(dist, ["B5"])


def test_rounding_cannot_leave_0_1_and_has_a_bound():
    """sum_tol, meant for rounding, had widened the [0, 1] bound of each
    value and had no upper limit (a set summing to 0.6 passed at 0.5)."""
    with pytest.raises(ValueError, match=r"-0\.01, outside \[0, 1\]"):
        ms.MeasuredFractions("x", {"b": -0.01, "a": 1.01}, "s", sum_tol=0.015)
    with pytest.raises(ValueError, match=r"1\.01, outside \[0, 1\]"):
        ms.MeasuredFractions("x", {"a": 1.01}, "s", complete=False,
                             sum_tol=0.015)
    # two values written to 0.1 miss 1 by at most 2 x 0.05
    with pytest.raises(ValueError, match=r"at most 2 x 0\.05 = 0\.1$"):
        ms.MeasuredFractions("Si Qn", {3: 0.3, 4: 0.3}, "fit", sum_tol=0.5)
    # the refusal prints |sum - 1| with every digit: 0.99 misses by a hair
    # more than 0.01
    with pytest.raises(ValueError, match=r"0\.010000000000000009"):
        ms.MeasuredFractions("Al CN", {4: 0.33, 5: 0.33, 6: 0.33}, "paper",
                             sum_tol=0.01)
    # a model mapping rounded to 0.001 takes a stated rounding too, bounded
    measured = ms.MeasuredFractions("Si Qn", {2: 0.33, 3: 0.33, 4: 0.34},
                                    "fit")
    rounded = {2: 0.333, 3: 0.333, 4: 0.333}
    with pytest.raises(ValueError, match="sum"):
        ms.compare_fractions(rounded, measured)
    out = ms.compare_fractions(rounded, measured, model_sum_tol=0.0015)
    assert any("stated rounding 0.0015" in n for n in out.notes)
    with pytest.raises(ValueError, match="wider than rounding"):
        ms.compare_fractions(rounded, measured, model_sum_tol=0.01)
    dist, _ = _qn_distribution()
    with pytest.raises(ValueError, match="model_sum_tol"):
        ms.compare_fractions(dist, measured, model_sum_tol=0.0015)


def test_a_complete_set_over_fewer_categories_is_named_as_such():
    """A complete 11B fit over B3, B4 against a model with 'other' compared
    fractions of different totals with no word (the B4 difference even
    changed sign once renormalised)."""
    counts = [{"B3": 109, "B4": 81, "other": 10}] * 2
    dist = Distribution.from_counts(counts, name="B CN", kind="fraction",
                                    keys=("B3", "B4", "other"))
    measured = ms.MeasuredFractions("B N4", {"B3": 0.58, "B4": 0.42},
                                    "11B MAS fit (test input)")
    out = ms.compare_fractions(dist, measured)
    note = [n for n in out.notes if "different totals" in n]
    assert len(note) == 1
    assert "matches sum to 0.95" in note[0] and "other (mean fraction 0.05" \
        in note[0]
    assert "renormalised(model, ['B3', 'B4'])" in note[0]
    # over the measured categories the note has nothing to say
    over = ms.compare_fractions(ms.renormalised(dist, ["B3", "B4"]), measured)
    assert not any("different totals" in n for n in over.notes)
    assert over.difference[1] == pytest.approx(81 / 190 - 0.42, abs=1e-15)


def test_the_spread_note_says_where_the_model_std_came_from():
    """The ddof = 1 note had been attached to a std the caller typed and to
    a NaN std alike."""
    measured = ms.MeasuredFractions("O", {"BO": 0.6, "NBO": 0.4}, "17O fit")
    given = ms.compare_fractions({"BO": (0.75, 0.004), "NBO": (0.25, 0.004)},
                                 measured)
    plain = ms.compare_fractions({"BO": 0.75, "NBO": 0.25}, measured)
    for result in (given, plain):
        assert not any("ddof" in n for n in result.notes)
    assert any("as given by the caller" in n for n in given.notes)
    assert any("no model spread was given" in n for n in plain.notes)
    dist, _ = _qn_distribution()
    from_frames = ms.compare_fractions(dist, ms.MeasuredFractions(
        "Si Qn", {3: 0.1, 4: 0.9}, "fit"))
    assert any("ddof = 1" in n for n in from_frames.notes)


def test_a_comparison_carries_its_provenance(quartz_frame):
    traj = md_model.MemoryTrajectory([quartz_frame])
    header = Provenance.from_trajectory(traj, frames_used=[0])
    measured = ms.MeasuredFractions("O", {"BO": 0.6, "NBO": 0.4}, "17O fit")
    out = ms.compare_fractions({"BO": 0.75, "NBO": 0.25}, measured,
                               provenance=header)
    assert out.provenance is header
    assert set(out.as_sheets()) == {"provenance", "comparison", "notes"}
    bare = ms.compare_fractions({"BO": 0.75, "NBO": 0.25}, measured)
    assert any("no provenance was given" in n for n in bare.notes)
    assert "provenance" not in bare.as_sheets()


# ---------------------------------------------------------------------------
# 2. correlations: refusals
# ---------------------------------------------------------------------------

def _angle_term(coefficient=-0.5, valid_range=None):
    return ms.Term("T-O-T angle", coefficient, angle_function="deg",
                   valid_range=valid_range)


def test_a_correlation_without_coefficients_or_reference_is_refused(
        dimer):
    frame, _ = dimer
    bonds = _bonds(frame)
    cases = {
        "the intercept": ms.Correlation("29Si", None, (_angle_term(),),
                                        reference="test"),
        "coefficient of mean T-O-T angle": ms.Correlation(
            "29Si", 1.0, (_angle_term(None),), reference="test"),
        "at least one term": ms.Correlation("29Si", 1.0, (),
                                            reference="test"),
        "literature reference": ms.Correlation("29Si", 1.0,
                                               (_angle_term(),),
                                               reference=None),
    }
    for lacking, correlation in cases.items():
        assert correlation.missing()
        with pytest.raises(ValueError) as error:
            ms.predict_shifts(correlation, frame, bonds, formers={"Si"})
        assert ms.TODO_REFERENCE in str(error.value)
        assert lacking in str(error.value)
    blank = ms.Correlation("29Si", 1.0, (_angle_term(),), reference="  ")
    assert "the literature reference of the coefficients" in blank.missing()


def test_md_nmr_refuses_before_reading_a_frame():
    class Unreadable:
        n_frames = 3

        def frame(self, k):
            raise AssertionError("a frame was read")

    correlation = ms.Correlation("29Si", 1.0, (_angle_term(),),
                                 reference=None)
    ox = md_model.model_oxidation(["Si", "O"])
    with pytest.raises(ValueError, match=ms.TODO_REFERENCE):
        ms.md_nmr(Unreadable(), ox, correlation, formers={"Si"},
                  delta_ppm=[0.0, 1.0], fwhm_ppm=1.0, lineshape="gaussian")


def test_the_module_ships_no_correlation():
    source = (ROOT / "facet" / "core" / "md_spectroscopy.py").read_text(
        encoding="utf-8")
    built = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) in ("Correlation", "Term")]
    assert built == []


def test_terms_and_correlations_refuse_what_they_cannot_mean():
    for bad in (dict(descriptor="angle", coefficient_ppm=1.0),
                dict(descriptor="T-O-T angle", coefficient_ppm=1.0),
                dict(descriptor="T-O-T angle", coefficient_ppm=1.0,
                     angle_function="tan"),
                dict(descriptor="CN", coefficient_ppm=1.0,
                     angle_function="cos"),
                dict(descriptor="bonds to", coefficient_ppm=1.0),
                dict(descriptor="CN", coefficient_ppm=1.0, partner="Al"),
                dict(descriptor="user", coefficient_ppm=1.0, name="x"),
                dict(descriptor="CN", coefficient_ppm=1.0, power=0),
                dict(descriptor="CN", coefficient_ppm=1.0,
                     valid_range=(6.0, 4.0)),
                dict(descriptor="CN", coefficient_ppm=float("inf"))):
        with pytest.raises(ValueError):
            ms.Term(**bad)
    for nucleus in ("Si29", "29", "0Si", "29Xx"):
        with pytest.raises(ValueError):
            ms.Correlation(nucleus, 0.0, (), reference="r")
    with pytest.raises(ValueError, match="two terms"):
        ms.Correlation("27Al", 0.0, (ms.Term("CN", 1.0), ms.Term("CN", 2.0)),
                       reference="r")
    with pytest.raises(ValueError, match="different ranges"):
        ms.Correlation("27Al", 0.0, (ms.Term("CN", 1.0, valid_range=(4, 6)),
                                     ms.Term("CN", 1.0, power=2)),
                       reference="r")
    assert ms.Correlation("27al", 0.0, (), reference="r").nucleus == "27Al"


def test_a_user_term_named_like_a_descriptor_is_refused():
    """A 'user' term named 'CN' beside a CN term shared one set of values
    with it: the user's values were dropped without a note."""
    with pytest.raises(ValueError, match="share the label 'CN'"):
        ms.Correlation("27Al", 0.0, (ms.Term("CN", 1.0), ms.Term(
            "user", 1.0, power=2, name="CN", unit="1")), reference="r")
    with pytest.raises(ValueError, match="share the label"):
        ms.Correlation("27Al", 0.0, (
            ms.Term("user", 1.0, name="x", unit="Å"),
            ms.Term("user", 1.0, power=2, name="x", unit="1")),
            reference="r")
    # one descriptor in two powers is one label, as before
    ms.Correlation("27Al", 0.0, (ms.Term("CN", 1.0), ms.Term("CN", 1.0,
                                                             power=2)),
                   reference="r")


def test_the_reference_placeholder_is_not_a_reference(dimer):
    """'TODO: need reference' had passed the reference gate."""
    frame, _ = dimer
    for text in (ms.TODO_REFERENCE, "todo: need reference (Smith?)"):
        correlation = ms.Correlation("29Si", 0.0, (ms.Term("CN", 1.0),),
                                     reference=text)
        assert any("placeholder" in m for m in correlation.missing())
        with pytest.raises(ValueError, match="placeholder"):
            ms.predict_shifts(correlation, frame, _bonds(frame),
                              formers=None)


def test_a_negative_power_divides_and_a_zero_is_undefined():
    """Power had to be 1 or more, so 1/x correlations could not be entered;
    now x = 0 under a negative power has no value and is counted."""
    frame = md_model.frame_from_arrays(
        ["Al"] * 5, np.arange(15.0).reshape(5, 3), box_ang=np.eye(3) * 20.0)
    term = ms.Term("user", 4.0, power=-1, name="x", unit="1")
    correlation = ms.Correlation("27Al", 10.0, (term,), reference="test")
    assert correlation.describe().endswith("(4.0) * [x]^-1")
    pred = ms.predict_shifts(correlation, frame, None, formers=None,
                             user_values={"x": [0.5, 1.0, 2.0, 0.0, np.nan]})
    assert pred.shift_ppm[:3].tolist() == [18.0, 14.0, 12.0]
    assert np.isnan(pred.shift_ppm[3:]).all()
    assert pred.undefined_by_term == {"x": 2}
    assert any("2 of 5 Al atoms have no value of x (NaN among the values "
               "given or a value of 0, which a negative power leaves "
               "undefined)" in n for n in pred.notes)
    with pytest.raises(ValueError, match="other than 0"):
        ms.Term("CN", 1.0, power=0)


def test_no_physical_parameter_has_a_default():
    def no_default(function, *names):
        parameters = inspect.signature(function).parameters
        for name in names:
            assert parameters[name].default is inspect.Parameter.empty, name

    no_default(ms.predict_shifts, "formers")
    no_default(ms.md_nmr, "formers", "delta_ppm", "fwhm_ppm", "lineshape")
    no_default(ms.nmr_spectrum, "fwhm_ppm", "lineshape")
    no_default(ms.broadened_spectrum, "fwhm_ppm", "lineshape")
    no_default(ms.average_feff_chi, "k_inv_ang", "s02")
    no_default(ms.md_exafs, "r_max_ang", "dr_ang", "minimum")
    no_default(ms.first_minimum, "method")
    no_default(ms.absorber_rdf, "r_max_ang", "dr_ang")
    no_default(ms.shell_cumulants, "r_lo_ang", "r_hi_ang", "weighting")
    no_default(ms.write_feff_input, "edge")
    no_default(ms.sample_absorbers, "n_clusters", "seed")
    no_default(ms.FeffRequest, "n_clusters", "seed", "cluster_radius_ang",
               "edge")
    no_default(ms.Correlation, "intercept_ppm", "terms", "reference")


# ---------------------------------------------------------------------------
# 2. correlations: the descriptors are the geometry
# ---------------------------------------------------------------------------

def test_the_dimer_descriptors_are_what_was_built(dimer):
    frame, d = dimer
    bonds = _bonds(frame)
    terms = (_angle_term(-0.5), ms.Term("Qn", 2.0), ms.Term("CN", 0.25),
             ms.Term("BVS", 0.125), ms.Term("mean bond length", 1.0),
             ms.Term("bonds to", 0.5, partner="O"),
             ms.Term("bridges to", 3.0, partner="Si"))
    correlation = ms.Correlation("29Si", -10.0, terms, reference="test")
    pred = ms.predict_shifts(correlation, frame, bonds, formers={"Si"})
    v = pred.values
    assert pred.n_atoms == 2 and pred.n_defined == 2
    assert v["mean T-O-T angle"] == pytest.approx([THETA_DEG] * 2, abs=1e-9)
    assert v["Qn (bridging O)"].tolist() == [1.0, 1.0]
    assert v["bridges to Si (via O)"].tolist() == [1.0, 1.0]
    assert v["CN"].tolist() == [4.0, 4.0]
    assert v["bonds to O"].tolist() == [4.0, 4.0]
    assert v["BVS"] == pytest.approx([4.0, 4.0], abs=1e-12)
    assert v["mean bond length"] == pytest.approx([d, d], abs=1e-12)
    expected = (-10.0 - 0.5 * THETA_DEG + 2.0 + 1.0 + 0.5 + d + 2.0 + 3.0)
    assert pred.shift_ppm == pytest.approx([expected] * 2, abs=1e-9)
    assert pred.n_outside == 0
    assert any(ms.TODO_REFERENCE in n and "range of validity" in n
               for n in pred.notes)
    assert ms.QUADRUPOLAR_NOTE in pred.notes

    # centred on the anion: one O bridges, six do not
    oxygen = ms.Correlation("17O", 0.0, (ms.Term(
        "T-O-T angle", 1.0, angle_function="cos"),), reference="test")
    po = ms.predict_shifts(oxygen, frame, bonds, formers={"Si"})
    defined = np.isfinite(po.shift_ppm)
    assert defined.sum() == 1 and po.n_atoms == 7
    assert po.shift_ppm[defined][0] == pytest.approx(
        math.cos(math.radians(THETA_DEG)), abs=1e-12)
    assert any("6 of 7 O atoms have no value of mean cos(T-O-T)" in n
               for n in po.notes)


def test_formers_are_never_assumed(dimer):
    frame, _ = dimer
    bonds = _bonds(frame)
    correlation = ms.Correlation("29Si", 0.0, (ms.Term("Qn", 1.0),),
                                 reference="test")
    with pytest.raises(ValueError, match="formers is None"):
        ms.predict_shifts(correlation, frame, bonds, formers=None)
    with pytest.raises(ValueError, match="not among the formers"):
        ms.predict_shifts(correlation, frame, bonds, formers={"B"})
    with pytest.raises(ValueError, match="cannot also be a network former"):
        ms.predict_shifts(correlation, frame, bonds, formers={"Si", "O"})
    with pytest.raises(ValueError, match="one string"):
        ms.predict_shifts(correlation, frame, bonds, formers="Si")
    # a descriptor that needs no former takes None
    cn = ms.Correlation("29Si", 0.0, (ms.Term("CN", 1.0),), reference="test")
    assert ms.predict_shifts(cn, frame, bonds, formers=None).n_defined == 2


def test_bonds_of_another_frame_are_refused(quartz_frame, quartz_bonds):
    """Only the largest row was checked, so the bonds of another frame with
    the same atoms gave shifts from the wrong geometry. Every bond vector is
    now formed again from this frame's fractions."""
    correlation = ms.Correlation("29Si", 0.0, (_angle_term(1.0),),
                                 reference="t")
    rng = np.random.default_rng(3)
    moved = md_model.frame_from_arrays(
        quartz_frame.elements,
        quartz_frame.cart_ang + rng.normal(0.0, 0.05, quartz_frame.cart_ang.shape),
        box_ang=quartz_frame.box_ang)
    with pytest.raises(ValueError, match="measured on another frame"):
        ms.predict_shifts(correlation, moved, quartz_bonds, formers={"Si"})
    # the frame's own bonds pass: the vectors agree to the bit
    ms.predict_shifts(correlation, quartz_frame, quartz_bonds, formers={"Si"})


def test_descriptors_that_cannot_exist_are_refused(dimer):
    """'bridges to' a non-former, 'bonds to' a cation from a cation, and a
    bridge descriptor without its anion had returned zeros with no note."""
    frame, _ = dimer
    bonds = _bonds(frame)

    def run(term, formers, anion="O", nucleus="29Si"):
        correlation = ms.Correlation(nucleus, 0.0, (term,), reference="t",
                                     anion=anion)
        return ms.predict_shifts(correlation, frame, bonds, formers=formers)

    with pytest.raises(ValueError, match="Al is not among the formers"):
        run(ms.Term("bridges to", 1.0, partner="Al"), {"Si"})
    with pytest.raises(ValueError, match="both bonded only as cations"):
        run(ms.Term("bonds to", 1.0, partner="Si"), None)
    with pytest.raises(ValueError, match="both bonded only as cations"):
        run(ms.Term("mean bond length", 1.0, partner="Si"), None)
    with pytest.raises(ValueError, match="holds no F, the bridging anion"):
        run(ms.Term("Qn", 1.0), {"Si"}, anion="F")
    # a partner absent from the frame: zeros, said so
    absent = run(ms.Term("bridges to", 1.0, partner="B"), {"Si", "B"})
    assert absent.values["bridges to B (via O)"].tolist() == [0.0, 0.0]
    assert any("the frame holds no B" in n for n in absent.notes)
    lengths = run(ms.Term("mean bond length", 1.0, partner="F"), None)
    assert any("the frame holds no F: every Si atom has no value of" in n
               for n in lengths.notes)


def _brute_force_angles(frame):
    """Per O: the angle to its two nearest Si over the 27 nearest images."""
    box = frame.box_ang
    shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1)
                       for c in (-1, 0, 1)], float) @ box
    si = frame.cart_ang[frame.elements == "Si"]
    images = (si[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    out = {}
    for row in np.flatnonzero(frame.elements == "O"):
        delta = images - frame.cart_ang[row]
        dist = np.linalg.norm(delta, axis=1)
        a, b = delta[np.argsort(dist)[:2]]
        cosine = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        out[int(row)] = math.degrees(math.acos(cosine))
    return out


def test_quartz_angles_equal_an_independent_brute_force(quartz_frame,
                                                        quartz_bonds):
    oracle = _brute_force_angles(quartz_frame)
    oxygen = ms.Correlation("17O", 0.0, (_angle_term(1.0),), reference="t")
    po = ms.predict_shifts(oxygen, quartz_frame, quartz_bonds,
                           formers={"Si"})
    assert po.n_defined == len(oracle) == 48
    for row, angle in zip(po.rows, po.values["mean T-O-T angle"]):
        assert angle == pytest.approx(oracle[int(row)], abs=1e-9)
    # each Si: the mean of the angles of its four bridges
    silicon = ms.Correlation("29Si", 0.0, (_angle_term(1.0),
                                           ms.Term("Qn", 1.0)),
                             reference="t")
    ps = ms.predict_shifts(silicon, quartz_frame, quartz_bonds,
                           formers={"Si"})
    b = quartz_bonds
    for row, angle in zip(ps.rows, ps.values["mean T-O-T angle"]):
        oxygens = b.anion[b.cation == row]
        assert angle == pytest.approx(
            statistics.fmean(oracle[int(o)] for o in oxygens), abs=1e-9)
    assert ps.values["Qn (bridging O)"].tolist() == [4.0] * 24
    # the bridge count from the Si side equals twice the bridging O
    assert ps.values["Qn (bridging O)"].sum() == 2 * po.n_defined


# ---------------------------------------------------------------------------
# 2. the spectrum
# ---------------------------------------------------------------------------

def test_each_line_has_unit_area_and_the_stated_width():
    grid = np.arange(-60.0, 60.0 + 1e-9, 0.005)
    shifts = [-3.0, 0.0, 2.5, 7.25]
    gauss = ms.broadened_spectrum(shifts, grid, fwhm_ppm=2.0,
                                  lineshape="gaussian")
    assert np.trapezoid(gauss, grid) == pytest.approx(4.0, rel=1e-12)
    assert ms.line_area_in_grid(shifts, grid, fwhm_ppm=2.0,
                                lineshape="gaussian") == pytest.approx(
        4.0, rel=1e-14)
    # a Lorentzian loses its tails: the exact area of one centred line
    # inside [-L, L] is (2/pi) arctan(L / gamma)
    lorentz = ms.broadened_spectrum([0.0], grid, fwhm_ppm=2.0,
                                    lineshape="lorentzian")
    exact = 2.0 / math.pi * math.atan(60.0 / 1.0)
    assert ms.line_area_in_grid([0.0], grid, fwhm_ppm=2.0,
                                lineshape="lorentzian") == pytest.approx(
        exact, rel=1e-14)
    assert np.trapezoid(lorentz, grid) == pytest.approx(exact, rel=1e-6)
    # half maximum at +-FWHM/2, for both shapes
    for shape in ms.LINESHAPES:
        line = ms.broadened_spectrum([0.0], np.array([-1.0, 0.0, 1.0]),
                                     fwhm_ppm=2.0, lineshape=shape)
        assert line[0] / line[1] == pytest.approx(0.5, rel=1e-14)
        assert line[2] / line[1] == pytest.approx(0.5, rel=1e-14)
    # the 40-sigma window and the chunking change no value: against every
    # line evaluated directly on the whole grid
    rng = np.random.default_rng(5)
    many = rng.normal(0.0, 15.0, 300)
    sigma = 0.5 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    direct = (np.exp(-0.5 * ((grid[None, :] - many[:, None]) / sigma) ** 2)
              .sum(axis=0) / (sigma * math.sqrt(2.0 * math.pi)))
    windowed = ms.broadened_spectrum(many, grid, fwhm_ppm=0.5,
                                     lineshape="gaussian")
    assert np.abs(windowed - direct).max() <= 1e-12 * direct.max()
    gamma = 0.25
    direct = (gamma / math.pi / ((grid[None, :] - many[:, None]) ** 2
                                 + gamma ** 2)).sum(axis=0)
    assert np.abs(ms.broadened_spectrum(many, grid, fwhm_ppm=0.5,
                                        lineshape="lorentzian")
                  - direct).max() <= 1e-12 * direct.max()
    with pytest.raises(ValueError, match="NaN"):
        ms.broadened_spectrum([np.nan], grid, fwhm_ppm=1.0,
                              lineshape="gaussian")
    with pytest.raises(ValueError):
        ms.broadened_spectrum([0.0], grid[::-1], fwhm_ppm=1.0,
                              lineshape="gaussian")
    with pytest.raises(ValueError):
        ms.broadened_spectrum([0.0], grid, fwhm_ppm=0.0,
                              lineshape="gaussian")


def _five_atoms():
    frame = md_model.frame_from_arrays(
        ["Al"] * 5, np.arange(15.0).reshape(5, 3), box_ang=np.eye(3) * 20.0)
    term = ms.Term("user", 4.0, name="x", unit="1", valid_range=(1.0, 3.0))
    correlation = ms.Correlation("27Al", 10.0, (term,), reference="test",
                                 shift_reference="test compound")
    values = {"x": [0.5, 1.5, 2.5, np.nan, 3.5]}
    return frame, correlation, values


def test_outside_and_undefined_atoms_are_counted_never_dropped():
    frame, correlation, values = _five_atoms()
    pred = ms.predict_shifts(correlation, frame, None, formers=None,
                             user_values=values)
    assert pred.shift_ppm[[0, 1, 2, 4]].tolist() == [12.0, 16.0, 20.0, 24.0]
    assert np.isnan(pred.shift_ppm[3])
    assert pred.outside.tolist() == [True, False, False, False, True]
    assert pred.outside_by_term == {"x": 2}
    assert pred.undefined_by_term == {"x": 1}
    grid = np.arange(-40.0, 80.0 + 1e-9, 0.01)
    spec = ms.nmr_spectrum([pred, pred], grid, fwhm_ppm=1.0,
                           lineshape="gaussian", frames=[4, 9])
    assert spec.counts.keys == ("in range", "outside range", "undefined")
    assert spec.counts.per_frame.tolist() == [[2, 2, 1], [2, 2, 1]]
    assert spec.area_numeric == pytest.approx([4.0, 4.0], rel=1e-12)
    assert spec.area_in_grid == pytest.approx([4.0, 4.0], rel=1e-14)
    assert np.trapezoid(spec.outside.mean, grid) == pytest.approx(2.0,
                                                                  rel=1e-12)
    assert spec.spectrum.std.max() == 0.0
    assert spec.spectrum.frames.tolist() == [4, 9]
    assert spec.mean_shift.mean == 18.0
    notes = " | ".join(spec.notes)
    assert "2 of 10 Al atoms over 2 frames" in notes
    assert "4 of 8 Al atoms with a shift over 2 frames" in notes
    assert "relative to test compound" in notes
    assert ms.QUADRUPOLAR_NOTE in spec.notes
    hist = ms.shift_histogram([pred], [0.0, 15.0, 30.0])
    assert hist.counts.tolist() == [[1, 3]]
    assert hist.n_nonfinite.tolist() == [1]
    with pytest.raises(ValueError, match="user_values"):
        ms.predict_shifts(correlation, frame, None, formers=None)


def test_md_nmr_over_two_identical_frames(quartz_frame):
    traj = md_model.MemoryTrajectory([quartz_frame, quartz_frame])
    ox = md_model.model_oxidation(quartz_frame.species)
    correlation = ms.Correlation(
        "29Si", -20.0, (_angle_term(-0.5, valid_range=(130.0, 160.0)),),
        reference="test input", shift_reference="TMS")
    grid = np.arange(-120.0, -60.0, 0.02)
    spec = ms.md_nmr(traj, ox, correlation, formers={"Si"}, delta_ppm=grid,
                     fwhm_ppm=1.0, lineshape="lorentzian")
    assert spec.spectrum.n_frames == 2
    assert spec.spectrum.std.max() == 0.0
    assert spec.mean_shift.std == 0.0
    lines = spec.provenance.as_lines()
    assert "network formers: Si" in lines
    assert any(line.startswith("method parameter: lineshape = lorentzian")
               for line in lines)
    sheets = spec.as_sheets()
    assert set(sheets) >= {"provenance", "spectrum", "atoms counted"}
    assert not any("no provenance" in n for n in spec.notes)
    assert "method parameter: pair search method = auto" in lines


def _user_prediction(shifts, nucleus="29Si", element="Si"):
    """One frame of len(shifts) atoms whose shift is the user value."""
    n = len(shifts)
    frame = md_model.frame_from_arrays(
        [element] * n, np.arange(3.0 * n).reshape(n, 3),
        box_ang=np.eye(3) * 40.0)
    correlation = ms.Correlation(nucleus, 0.0, (ms.Term(
        "user", 1.0, name="x", unit="ppm"),), reference="test")
    return ms.predict_shifts(correlation, frame, None, formers=None,
                             user_values={"x": list(shifts)})


def test_the_spectrum_states_how_its_grid_samples_it():
    """A FWHM at or below the grid step gave a sampled spectrum that missed
    the lines, with nothing but the two area columns to show it; lines
    beyond the grid were counted 'in range' and not noted."""
    pred = _user_prediction([-100.13, -90.0, -80.31])
    grid = np.arange(-120.0, -59.9, 0.5)
    spec = ms.nmr_spectrum(pred, grid, fwhm_ppm=0.1, lineshape="gaussian")
    assert spec.area_in_grid[0] == pytest.approx(3.0, abs=1e-12)
    assert abs(spec.area_numeric[0] - 3.0) > 0.1
    relative = abs(spec.area_numeric[0] - spec.area_in_grid[0]) \
        / spec.area_in_grid[0]
    notes = " | ".join(spec.notes)
    assert "grid step up to 0.5 ppm (5 x the FWHM)" in notes
    assert "is at least the FWHM 0.1 ppm" in notes
    assert f"by up to {relative:.3g} (relative)" in notes
    fine = ms.nmr_spectrum(pred, np.arange(-120.0, -59.99, 0.01),
                           fwhm_ppm=1.0, lineshape="gaussian")
    assert not any("at least the FWHM" in n for n in fine.notes)
    # lines beyond the grid: counted per frame and noted
    far = _user_prediction([88.0, 90.0, -90.0])
    out = ms.nmr_spectrum(far, grid, fwhm_ppm=1.0, lineshape="gaussian")
    assert out.n_above_grid.tolist() == [2] and out.n_below_grid.tolist() == [0]
    assert any("0 predicted shift(s) lie below the grid's first point" in n
               and "2 above its last point" in n for n in out.notes)
    rows = out.as_sheets()["areas"]
    assert rows[0]["shifts above the grid"] == 2
    # NMR plotting order is refused with the reversal named
    with pytest.raises(ValueError, match=r"delta_ppm\[::-1\]"):
        ms.nmr_spectrum(pred, grid[::-1], fwhm_ppm=1.0, lineshape="gaussian")


def test_mixed_correlations_are_refused_and_provenance_travels(quartz_frame):
    """shift_histogram had binned a 27Al prediction under the 29Si name;
    nmr_spectrum had no provenance route for user terms."""
    si = _user_prediction([-90.0, -100.0])
    al = _user_prediction([60.0], nucleus="27Al", element="Al")
    with pytest.raises(ValueError, match="different correlations"):
        ms.shift_histogram([si, al], [-120.0, 0.0, 120.0])
    with pytest.raises(ValueError, match="ShiftPrediction items, not str"):
        ms.shift_histogram(["not a prediction"], [-120.0, 0.0])
    grid = np.arange(-120.0, -60.0, 0.05)
    bare = ms.nmr_spectrum(si, grid, fwhm_ppm=1.0, lineshape="lorentzian")
    assert bare.provenance is None and "provenance" not in bare.as_sheets()
    assert any("no provenance was given" in n for n in bare.notes)
    header = Provenance.from_trajectory(
        md_model.MemoryTrajectory([quartz_frame]), frames_used=[0])
    kept = ms.nmr_spectrum(si, grid, fwhm_ppm=1.0, lineshape="lorentzian",
                           provenance=header)
    assert kept.provenance is header and "provenance" in kept.as_sheets()
    assert kept.as_sheets()["areas"][0]["timestep"] == si.timestep
    assert not any("no provenance" in n for n in kept.notes)


def test_md_nmr_refuses_a_missing_element_before_the_pair_search(
        quartz_frame, monkeypatch):
    """An element absent from the model was refused only after the first
    frame's pair search."""
    def no_search(*args, **kwargs):
        raise AssertionError("the pair search ran before the refusal")

    monkeypatch.setattr(ms, "analyse_frame", no_search)
    traj = md_model.MemoryTrajectory([quartz_frame])
    ox = md_model.model_oxidation(quartz_frame.species)
    for nucleus, formers, match in (("27Al", {"Si"}, "holds no Al"),
                                    ("29Si", {"Al"}, "not among the formers")):
        correlation = ms.Correlation(nucleus, 0.0, (ms.Term("Qn", 1.0),),
                                     reference="test")
        with pytest.raises(ValueError, match=match):
            ms.md_nmr(traj, ox, correlation, formers=formers,
                      delta_ppm=[0.0, 1.0], fwhm_ppm=1.0,
                      lineshape="gaussian")


# ---------------------------------------------------------------------------
# 3. cumulants
# ---------------------------------------------------------------------------

def test_a_gaussian_has_c3_and_c4_zero():
    """Gauss-Hermite nodes and weights reproduce a normal distribution's
    moments up to degree 2n - 1, so the cumulants are exact."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(12)
    r0, sigma = 2.05, 0.08
    c1, c2, c3, c4 = ms.distance_cumulants(r0 + sigma * nodes, weights)
    assert c1 == pytest.approx(r0, abs=1e-15)
    assert c2 == pytest.approx(sigma ** 2, rel=1e-13)
    assert abs(c3) < 1e-16
    assert abs(c4) < 1e-17


def test_a_skewed_distribution_has_its_known_cumulants():
    """A gamma distribution of shape k: cumulants k, k, 2k, 6k (scale 1);
    generalised Gauss-Laguerre quadrature reproduces its moments exactly."""
    from scipy.special import roots_genlaguerre

    alpha = 1.5
    k = alpha + 1.0
    nodes, weights = roots_genlaguerre(10, alpha)
    r0, scale = 1.9, 0.03
    c1, c2, c3, c4 = ms.distance_cumulants(r0 + scale * nodes, weights)
    assert c1 == pytest.approx(r0 + scale * k, rel=1e-13)
    assert c2 == pytest.approx(scale ** 2 * k, rel=1e-11)
    assert c3 == pytest.approx(2.0 * scale ** 3 * k, rel=1e-9)
    assert c4 == pytest.approx(6.0 * scale ** 4 * k, rel=1e-8)


def test_cumulants_equal_direct_moments_of_samples():
    rng = np.random.default_rng(11)
    d = 2.0 + rng.gamma(2.0, 0.05, size=5000)
    m = d.mean()
    mu = [np.mean((d - m) ** n) for n in (2, 3, 4)]
    c1, c2, c3, c4 = ms.distance_cumulants(d)
    assert c1 == pytest.approx(m, rel=1e-15)
    assert c2 == pytest.approx(mu[0], rel=1e-12)
    assert c3 == pytest.approx(mu[1], rel=1e-10)
    assert c4 == pytest.approx(mu[2] - 3.0 * mu[0] ** 2, rel=1e-9)
    # 1/r^2 weighting is the weighted distribution, nothing else
    w = 1.0 / d ** 2
    wm = np.sum(w * d) / np.sum(w)
    assert ms.distance_cumulants(d, w)[0] == pytest.approx(wm, rel=1e-14)
    assert ms.distance_cumulants([]) == pytest.approx((np.nan,) * 4,
                                                      nan_ok=True)
    with pytest.raises(ValueError):
        ms.distance_cumulants([1.0, 2.0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# 3. absorber-centred partials and shells
# ---------------------------------------------------------------------------

def test_the_absorber_pairs_are_the_pair_distances_of_pdf(quartz_frame):
    ox = md_model.model_oxidation(quartz_frame.species)
    structure = md_model.frame_to_structure(quartz_frame, ox)
    ap = ms.absorber_pairs(quartz_frame,
                           bulk.iter_pairs(quartz_frame, 4.0), "Si")
    oracle = []
    for distances, i, _ in pdf.pair_distances(structure, 4.0):
        oracle.append(distances[quartz_frame.elements[i] == "Si"])
    oracle = np.sort(np.concatenate(oracle))
    assert ap.d_ang.size == oracle.size
    assert np.abs(np.sort(ap.d_ang) - oracle).max() < 1e-12
    assert set(ap.i.tolist()) == set(ap.rows.tolist())
    with pytest.raises(ValueError, match="another frame"):
        moved = md_model.frame_from_arrays(
            quartz_frame.elements, quartz_frame.cart_ang + 0.3,
            box_ang=quartz_frame.box_ang)
        ms.absorber_pairs(moved, bulk.find_pairs(quartz_frame, 4.0), "Si")


def test_the_partial_conserves_the_pairs_and_counts_exactly(quartz_frame):
    ap = ms.absorber_pairs(quartz_frame, bulk.iter_pairs(quartz_frame, 4.0),
                           "Si")
    # r_max in the empty gap after the first Si-O shell: every first-shell
    # pair is inside the grid
    rdf = ms.absorber_rdf(ap, r_max_ang=2.5, dr_ang=0.01)["O"]
    assert np.sum(rdf.rdf_per_ang) * rdf.dr_ang == pytest.approx(4.0,
                                                                 rel=1e-13)
    assert rdf.n_cum[-1] == 4.0
    first = ap.d_ang[ap.elements[ap.j] == "O"].min()
    assert rdf.n_cum[rdf.r_ang < first].max() == 0.0
    rho = 48 / quartz_frame.volume_ang3
    at = 160
    assert rdf.g[at] == pytest.approx(
        rdf.rdf_per_ang[at] / (4.0 * math.pi * rdf.r_ang[at] ** 2 * rho),
        rel=1e-15)
    wide = ms.absorber_pairs(quartz_frame,
                             bulk.iter_pairs(quartz_frame, 5.0), "Si")
    with pytest.raises(ValueError, match="half the smallest"):
        ms.absorber_rdf(wide, r_max_ang=4.5, dr_ang=0.01)
    with pytest.raises(ValueError, match="searched to"):
        ms.absorber_rdf(ms.absorber_pairs(
            quartz_frame, bulk.iter_pairs(quartz_frame, 2.0), "Si"),
            r_max_ang=3.0, dr_ang=0.01)


def test_the_first_minimum_rule():
    """The rule is glass.first_minimum's, under the caller's MinimumMethod;
    the shell limit adds the first-shell maximum."""
    r = np.arange(1, 9) * 0.1
    curve = [0, 2, 3, 2, 1, 0.5, 1, 1.2]
    lim = ms.first_minimum(r, curve, method=FIRST_LOCAL)
    assert lim.r_ang == pytest.approx(0.6) and lim.r_peak_ang == pytest.approx(
        0.3)
    assert lim.g_peak == 3.0 and lim.g_at_limit == 0.5 and lim.notes == ()
    assert lim.method.startswith(FIRST_LOCAL.describe())
    assert lim.level == 1.0 and lim.floor_r_ang == pytest.approx((0.6, 0.6))
    flat = ms.first_minimum(r, [0, 2, 0, 0, 0, 1, 1, 1], method=FIRST_LOCAL)
    assert flat.r_ang == pytest.approx(0.4)          # midpoint of 0.3 .. 0.5
    none = ms.first_minimum(r, [0, 0.5, 0.9, 0.8, 0.5, 0.9, 1, 1],
                            method=FIRST_LOCAL)
    assert none.r_ang is None and "does not exceed 1" in none.reason
    assert ms.first_minimum(r, [0, 2, 1.5, 1.2, 1.1, 1.0, 0.9, 0.8],
                            method=FIRST_LOCAL).r_ang is None
    # smoothing keeps the minimum of a valley symmetric about 3.0 Å (the
    # grid starts at the minimum at 1 Å, so the first maximum is at 2 Å)
    grid = np.arange(1.0, 6.0, 0.01)
    g = 1.0 + np.cos(2.0 * math.pi * (grid - 2.0) / 2.0)
    smooth = ms.first_minimum(grid, g, method=glass.MinimumMethod(
        "valley", 0.05, "midpoint", 0.0))
    assert smooth.r_ang == pytest.approx(3.0, abs=1e-9)
    assert "sigma 0.05" in smooth.method
    with pytest.raises(ValueError, match="MinimumMethod"):
        ms.first_minimum(r, curve, method=None)


def test_a_limit_before_the_main_peak_is_noted():
    """At margin 0 the rule stopped at the foot of the real Al-Na peak of a
    glass (2.74 Å, N 0.085 of about 2.7) with g below 1 there, so the one
    note (g at the limit not below the level) said nothing. A rise at the
    foot, a dip below 1, then the peak: the second note gives both maxima."""
    r = np.arange(1, 14) * 0.1
    g = [0, 0, 1.2, 0.8, 1.5, 3, 5, 3, 0.5, 0.2, 0.5, 1, 1]
    lim = ms.first_minimum(r, g, method=FIRST_LOCAL, absorber="Al",
                           neighbour="Na")
    assert lim.r_ang == pytest.approx(0.4) and lim.g_at_limit == 0.8
    assert lim.g_peak == 1.2
    assert len(lim.notes) == 1
    assert lim.notes[0].startswith("Al-Na: g(r) beyond the limit")
    assert "reaches 5 at 0.7" in lim.notes[0] and "above 1.2" in lim.notes[0]
    # a first shell that is the highest maximum: no note
    plain = ms.first_minimum(r, [0, 2, 6, 2, 1, 0.5, 1, 1.4, 1.2, 1, 1, 1, 1],
                             method=FIRST_LOCAL)
    assert plain.notes == ()


def _noisy_shell_frames(n_frames, seed):
    """Al on a 7 Å lattice in a 28 Å box, each with 4 O at 1.75 +- 0.1 Å.

    The distances are uniform in [1.65, 1.85] Å (a test input), so the
    first shell has sharp edges and a noisy top: every O of an Al's own
    shell is within 1.85 Å and every other O at least 7 - 1.85 Å away.
    """
    rng = np.random.default_rng(seed)
    side, spacing = 4, 7.0
    centres = np.array([[a, b, c] for a in range(side) for b in range(side)
                        for c in range(side)], float) * spacing + 1.0
    frames = []
    for _ in range(n_frames):
        u = rng.normal(size=(centres.shape[0], 4, 3))
        u /= np.linalg.norm(u, axis=2, keepdims=True)
        d = rng.uniform(1.65, 1.85, size=(centres.shape[0], 4, 1))
        oxygens = (centres[:, None, :] + d * u).reshape(-1, 3)
        frames.append(md_model.frame_from_arrays(
            ["Al"] * len(centres) + ["O"] * len(oxygens),
            np.vstack([centres, oxygens]),
            box_ang=np.eye(3) * side * spacing))
    return md_model.MemoryTrajectory(frames)


def test_a_noisy_first_shell_keeps_its_neighbours_under_the_valley_rule():
    """The rule written here first (the first point after the first maximum
    where g rises, values as they stand) put the Al-O limit of a real glass
    on a noise wiggle of the peak (N = 1.79 of about 4). md_exafs now takes
    glass's MinimumMethod, with the level and counting errors glass passes,
    and returns the cutoff glass.first_minimum gives the same g(r)."""
    traj = _noisy_shell_frames(3, seed=7)
    volume = 28.0 ** 3
    valley = ms.md_exafs(traj, "Al", r_max_ang=8.0, dr_ang=0.01,
                         minimum=VALLEY, neighbours=["O"])
    limit = valley.limits["O"]
    assert 1.85 < limit.r_ang < 7.0 - 1.85
    first = valley.shells[0]
    assert first.per_frame["N"].mean == 4.0 and first.per_frame["N"].std == 0
    assert limit.notes == () and limit.g_at_limit == 0.0
    axis = valley.g["O"].axis
    per_unit = glass.pair_counts_per_unit_g(axis, 0.01, 64, 256, volume,
                                            same_element=False, n_frames=3)
    assert limit.r_ang == glass.first_minimum(
        axis, valley.g["O"].mean, VALLEY, level=1.0,
        pairs_per_unit_g=per_unit).r_ang
    assert limit.g_std_error == 0.0 and limit.depth_std_errors > 0.0
    assert 1.85 < limit.floor_r_ang[0] <= limit.r_ang <= limit.floor_r_ang[1]
    # the textbook rule, asking a drop and a rise of 2 counting errors
    aware = ms.md_exafs(traj, "Al", r_max_ang=8.0, dr_ang=0.01,
                        minimum=LOCAL_2SE, neighbours=["O"])
    assert 1.85 < aware.limits["O"].r_ang < 7.0 - 1.85
    assert aware.shells[0].per_frame["N"].mean == 4.0
    # the textbook rule on the raw, noisy g(r), values as they stand, stops
    # inside the peak, and says so with both heights
    local = ms.md_exafs(traj, "Al", r_max_ang=8.0, dr_ang=0.01,
                        minimum=FIRST_LOCAL, neighbours=["O"])
    cut = local.limits["O"]
    assert 1.65 < cut.r_ang < 1.85 and cut.g_at_limit > 1.0
    assert cut.r_ang == glass.first_minimum(
        axis, local.g["O"].mean, FIRST_LOCAL, level=1.0,
        pairs_per_unit_g=per_unit).r_ang
    assert local.shells[0].per_frame["N"].mean < 4.0
    assert any("not below its uncorrelated level 1" in n
               and "highest point of g below the limit" in n
               for n in local.notes)
    assert cut.notes and cut.g_peak >= cut.g_at_limit
    rows = local.provenance.as_lines()
    assert any("first minimum rule = first local minimum" in line
               for line in rows)


def test_quartz_first_shell(quartz_frame):
    traj = md_model.MemoryTrajectory([quartz_frame, quartz_frame])
    res = ms.md_exafs(traj, "Si", r_max_ang=4.2, dr_ang=0.01,
                      minimum=FIRST_LOCAL, neighbours=["O", "Si"],
                      shells=[("Si", 2.9, 3.2)])
    assert res.searches_per_frame == 2
    ap = ms.absorber_pairs(quartz_frame, bulk.iter_pairs(quartz_frame, 4.2),
                           "Si")
    d_o = np.sort(ap.d_ang[ap.elements[ap.j] == "O"])
    last_first, next_shell = d_o[95], d_o[96]           # 24 Si x 4 O
    limit = res.limits["O"].r_ang
    assert last_first < limit < next_shell
    assert res.limits["O"].g_at_limit == 0.0
    by = {(s.neighbour, s.r_hi_ang): s for s in res.shells}
    first = by[("O", limit)]
    assert first.per_frame["N"].mean == 4.0
    assert first.per_frame["N"].std == 0.0
    assert first.per_frame["R"].std == 0.0
    assert first.pooled["R"] == pytest.approx(statistics.fmean(d_o[:96]),
                                              abs=1e-15)
    assert first.pooled["sigma^2"] == pytest.approx(
        statistics.pvariance(d_o[:96]), rel=1e-12)
    assert by[("Si", 3.2)].pooled["N"] == 4.0
    assert by[("Si", 3.2)].limit_source == "user"
    assert res.limits["Si"].r_ang is None              # no rise before 4.2 Å
    assert any("no first-shell limit" in n for n in res.notes)
    # nothing to measure in a second pass: none is made, and that is said
    alone = ms.md_exafs(traj, "Si", r_max_ang=4.2, dr_ang=0.01,
                        minimum=FIRST_LOCAL, neighbours=["Si"])
    assert alone.shells == () and alone.searches_per_frame == 1
    assert any("no shell was measured" in n for n in alone.notes)
    # with the limit given, one search per frame and the same numbers
    again = ms.md_exafs(traj, "Si", r_max_ang=4.2, dr_ang=0.01,
                        minimum=None, neighbours=["O"],
                        first_shell_limits_ang={"O": limit})
    assert again.searches_per_frame == 1
    assert again.shells[0].pooled == first.pooled
    assert again.limits["O"].source == "user"
    lines = again.provenance.as_lines()
    assert any(line.startswith("distance cutoff: Si-O") for line in lines)
    assert "g Si-O" in again.as_sheets()


def test_progress_is_reported_and_cancel_stops(quartz_frame):
    traj = md_model.MemoryTrajectory([quartz_frame, quartz_frame])
    seen = []
    ms.md_exafs(traj, "Si", r_max_ang=4.0, dr_ang=0.02, minimum=FIRST_LOCAL,
                neighbours=["O"], progress=lambda done, total:
                seen.append((done, total)))
    assert seen == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]   # two passes
    with pytest.raises(ms.AnalysisCancelled):
        ms.md_exafs(traj, "Si", r_max_ang=4.0, dr_ang=0.02,
                    minimum=FIRST_LOCAL, cancelled=lambda: True)
    with pytest.raises(ValueError, match="minimum is None"):
        ms.md_exafs(traj, "Si", r_max_ang=4.0, dr_ang=0.02, minimum=None,
                    neighbours=["O"])


def test_shell_refusals(quartz_frame):
    ap = ms.absorber_pairs(quartz_frame, bulk.iter_pairs(quartz_frame, 3.0),
                           "Si")
    with pytest.raises(ValueError, match="searched to"):
        ms.shell_cumulants(ap, "O", r_lo_ang=0.0, r_hi_ang=3.5,
                           weighting="none")
    with pytest.raises(ValueError, match="weighting"):
        ms.shell_cumulants(ap, "O", r_lo_ang=0.0, r_hi_ang=2.0,
                           weighting="1/r")
    empty = ms.shell_cumulants(ap, "O", r_lo_ang=2.0, r_hi_ang=2.5,
                               weighting="none")
    assert empty.n_per_absorber == 0.0 and np.isnan(empty.r_mean_ang)
    averaged = ms.average_shells([empty, empty])
    assert any("hold no pair" in n for n in averaged.notes)
    with pytest.raises(ValueError, match="holds no"):
        ms.absorber_pairs(quartz_frame, bulk.iter_pairs(quartz_frame, 3.0),
                          "Al")


def _half_width(frame):
    return float(frame.perpendicular_widths_ang.min() / 2.0)


def test_a_repeated_shell_is_refused_and_an_equal_one_is_kept_apart(
        quartz_frame):
    """A repeated shell crashed the two-pass path (2n rows for n frames) and
    was merged without a word in the one-pass path."""
    traj = md_model.MemoryTrajectory([quartz_frame] * 3)
    twice = [("Si", 2.9, 3.2), ("Si", 2.9, 3.2)]
    for limits in (None, {"O": 2.0}):
        with pytest.raises(ValueError, match="given twice"):
            ms.md_exafs(traj, "Si", r_max_ang=4.0, dr_ang=0.02,
                        minimum=FIRST_LOCAL, neighbours=["O"], shells=twice,
                        first_shell_limits_ang=limits)
    with pytest.raises(ValueError, match="first shell of the limit given"):
        ms.md_exafs(traj, "Si", r_max_ang=4.0, dr_ang=0.02, minimum=None,
                    neighbours=["O"], first_shell_limits_ang={"O": 2.0},
                    shells=[("O", 0.0, 2.0)])
    # a user shell that equals the automatic first shell: two shells, each
    # with one row per frame
    auto = ms.md_exafs(traj, "Si", r_max_ang=4.0, dr_ang=0.02,
                       minimum=FIRST_LOCAL, neighbours=["O"])
    limit = auto.limits["O"].r_ang
    both = ms.md_exafs(traj, "Si", r_max_ang=4.0, dr_ang=0.02,
                       minimum=FIRST_LOCAL, neighbours=["O"],
                       shells=[("O", 0.0, limit)])
    assert [s.limit_source for s in both.shells] == [
        "first minimum of the frame-averaged g(r)", "user"]
    assert both.shells[0].pooled == both.shells[1].pooled
    assert all(s.per_frame["N"].n_frames == 3 for s in both.shells)
    assert any("both are listed" in n for n in both.notes)


def test_every_radius_past_half_the_box_is_refused_before_a_search(
        quartz_frame, monkeypatch):
    """A shell past half the box counted pairs once per periodic image, with
    no note; r_max there was refused only after the first search."""
    half = _half_width(quartz_frame)
    ap = ms.absorber_pairs(quartz_frame,
                           bulk.iter_pairs(quartz_frame, half + 0.5), "Si")
    with pytest.raises(ValueError, match="half the smallest perpendicular"):
        ms.shell_cumulants(ap, "O", r_lo_ang=0.0, r_hi_ang=half + 0.4,
                           weighting="none")

    def no_search(*args, **kwargs):
        raise AssertionError("a pair search ran before the refusal")

    monkeypatch.setattr(ms, "iter_pairs", no_search)
    traj = md_model.MemoryTrajectory([quartz_frame] * 2)
    for kwargs in (dict(r_max_ang=half + 0.2),
                   dict(r_max_ang=3.0, shells=[("O", 3.0, half + 0.2)]),
                   dict(r_max_ang=3.0,
                        first_shell_limits_ang={"O": half + 0.2})):
        with pytest.raises(ValueError, match="half the smallest"):
            ms.md_exafs(traj, "Si", dr_ang=0.02, minimum=FIRST_LOCAL,
                        neighbours=["O"], **kwargs)


def test_pass_one_searches_to_the_cluster_radius_only_where_a_cluster_is(
        quartz_frame, tmp_path, monkeypatch):
    """One drawn cluster made every frame search to the cluster radius; the
    provenance said 'search radius: not given'."""
    radii = []
    real = bulk.iter_pairs

    def spy(frame, r_ang, **kwargs):
        radii.append(float(r_ang))
        return real(frame, r_ang, **kwargs)

    monkeypatch.setattr(ms, "iter_pairs", spy)
    traj = md_model.MemoryTrajectory([quartz_frame] * 3)
    request = ms.FeffRequest(tmp_path / "feff", 1, 5, 4.0, "K")
    res = ms.md_exafs(traj, "Si", r_max_ang=2.5, dr_ang=0.02, minimum=None,
                      neighbours=["O"], first_shell_limits_ang={"O": 2.0},
                      feff=request)
    drawn = res.feff.clusters[0].frame
    # elsewhere: the g(r) grid's last point plus dr (pdf.py's grid)
    need = float(np.arange(0.02, 2.5 + 0.01, 0.02)[-1] + 0.02)
    assert radii == [4.0 if k == drawn else need for k in range(3)]
    assert res.provenance.r_search_ang == 4.0
    lines = res.provenance.as_lines()
    assert "search radius: 4.0 Å" in lines
    assert f"method parameter: pass 1 search radius (Å) = {need}" in lines
    assert "method parameter: pair search method = auto" in lines


def test_the_last_grid_point_holds_the_pairs_just_past_it(tmp_path,
                                                          monkeypatch):
    """md_exafs searched to r_max, so a pair in (r_last, r_last + dr) never
    reached the last grid point it deposits on: the last point of a real
    glass's Al-O g(r) came out at about half its value. One Al-O pair 0.4
    of a step past r_max: 0.6 of it belongs on the last point."""
    r_max, dr = 3.0, 0.1
    frame = md_model.frame_from_arrays(
        ["Al", "O"], np.array([[5.0, 5.0, 5.0], [5.0 + r_max + 0.4 * dr, 5.0,
                                                 5.0]]),
        box_ang=np.eye(3) * 20.0)
    with pytest.raises(ValueError, match="its last point plus dr_ang"):
        ms.absorber_rdf(ms.absorber_pairs(frame, bulk.iter_pairs(frame, r_max),
                                          "Al"), r_max_ang=r_max, dr_ang=dr)
    rdf = ms.absorber_rdf(ms.absorber_pairs(
        frame, bulk.iter_pairs(frame, r_max + dr), "Al"),
        r_max_ang=r_max, dr_ang=dr)["O"]
    last = float(rdf.r_ang[-1])
    expected = 0.6 * frame.volume_ang3 / (4.0 * math.pi * last ** 2 * dr)
    assert rdf.g[-1] == pytest.approx(expected, rel=1e-9)
    assert rdf.n_cum[-1] == 0.0                   # the pair lies past r_max
    res = ms.md_exafs(md_model.MemoryTrajectory([frame]), "Al",
                      r_max_ang=r_max, dr_ang=dr, minimum=None,
                      neighbours=["O"], first_shell_limits_ang={"O": 2.0})
    assert res.g["O"].mean[-1] == pytest.approx(expected, rel=1e-9)
    # with every limit known, pass 1 still searches to the grid's need when
    # that lies past the shells
    assert res.provenance.method_parameters["pass 1 search radius (Å)"] == \
        pytest.approx(last + dr, abs=1e-12)


def test_pass_one_searches_only_as_far_as_the_grid_when_shells_wait(
        quartz_frame, monkeypatch):
    """With a limit to locate, every shell is measured in pass 2, yet pass 1
    searched to the user shells' outer edges as well."""
    radii = []
    real = bulk.iter_pairs

    def spy(frame, r_ang, **kwargs):
        radii.append(float(r_ang))
        return real(frame, r_ang, **kwargs)

    monkeypatch.setattr(ms, "iter_pairs", spy)
    traj = md_model.MemoryTrajectory([quartz_frame] * 2)
    res = ms.md_exafs(traj, "Si", r_max_ang=4.0, dr_ang=0.02,
                      minimum=FIRST_LOCAL, neighbours=["O"],
                      shells=[("Si", 2.9, 4.2)])
    need = float(np.arange(0.02, 4.0 + 0.01, 0.02)[-1] + 0.02)
    assert need < 4.2
    assert radii == [need, need, 4.2, 4.2]
    assert res.searches_per_frame == 2
    assert [(s.neighbour, s.r_hi_ang) for s in res.shells] == [
        ("O", res.limits["O"].r_ang), ("Si", 4.2)]
    assert res.shells[1].per_frame["N"].n_frames == 2


def test_an_unused_minimum_method_is_not_recorded_as_used(quartz_frame):
    """With every first-shell limit given, a MinimumMethod passed as well
    was written into the provenance as if a limit had come from it."""
    traj = md_model.MemoryTrajectory([quartz_frame])
    res = ms.md_exafs(traj, "Si", r_max_ang=2.5, dr_ang=0.02,
                      minimum=FIRST_LOCAL, neighbours=["O"],
                      first_shell_limits_ang={"O": 2.0})
    assert not any("first minimum" in key
                   for key in res.provenance.method_parameters)
    assert any("no limit was located with it" in n for n in res.notes)
    located = ms.md_exafs(traj, "Si", r_max_ang=2.5, dr_ang=0.02,
                          minimum=FIRST_LOCAL, neighbours=["O"])
    assert located.provenance.method_parameters["first minimum rule"] == \
        "first local minimum"


# ---------------------------------------------------------------------------
# 3. FEFF
# ---------------------------------------------------------------------------

def _blocks(text, name, end):
    body = text.split(name)[1].split(end)[0]
    return [line.split() for line in body.splitlines()
            if line.strip() and not line.strip().startswith("*")]


def test_the_feff_input_is_write_feffs(quartz, tmp_path):
    frame, _ = md_model.supercell_frame(quartz, (1, 1, 1))
    ox = md_model.model_oxidation(frame.species)
    structure = md_model.frame_to_structure(frame, ox)
    ap = ms.absorber_pairs(frame, bulk.iter_pairs(frame, 5.0), "Si")
    for row in ap.rows[:2]:
        cluster = ms.feff_cluster(ap, int(row), cluster_radius_ang=5.0,
                                  frame_label=0)
        mine = ms.write_feff_input(cluster, tmp_path / f"m{row}.inp",
                                   edge="K").read_text(encoding="utf-8")
        theirs = exporters.write_feff(structure, int(row),
                                      tmp_path / f"t{row}.inp", rmax=5.0,
                                      edge="K").read_text(encoding="utf-8")
        assert _blocks(mine, "POTENTIALS", "ATOMS") == \
            _blocks(theirs, "POTENTIALS", "ATOMS")

        def atoms(text):
            rows = _blocks(text, "\nATOMS", "END")
            return sorted((r[4], round(float(r[5]), 4), round(float(r[0]), 4),
                           round(float(r[1]), 4), round(float(r[2]), 4), r[3])
                          for r in rows)

        assert atoms(mine) == atoms(theirs)
        for card in ("EDGE", "PRINT", "RPATH", "CONTROL", "EXCHANGE", "S02"):
            line = [x for x in mine.splitlines() if x.startswith(card)]
            assert line == [x for x in theirs.splitlines()
                            if x.startswith(card)], card
        # read back: every atom of the cluster, at its vector and distance,
        # with the potential of its own element
        potentials = {r[0]: r[2] for r in _blocks(mine, "POTENTIALS",
                                                  "ATOMS")}
        written = _blocks(mine, "\nATOMS", "END")[1:]
        assert len(written) == len(cluster.elements)
        for r, element, v, d in zip(written, cluster.elements,
                                    cluster.vec_ang, cluster.d_ang):
            assert r[4] == element and potentials[r[3]] == element
            assert [float(x) for x in r[:3]] == pytest.approx(v, abs=5e-6)
            assert float(r[5]) == pytest.approx(d, abs=5e-6)
        # the rules tests/test_exafs.py holds write_feff to
        zeros = [r for r in _blocks(mine, "\nATOMS", "END") if r[3] == "0"]
        assert len(zeros) == 1
        assert all(len(r) <= 3 for r in _blocks(mine, "POTENTIALS", "ATOMS"))
        assert all(len(r) <= 6 for r in _blocks(mine, "\nATOMS", "END"))
    with pytest.raises(ValueError, match="RPATH"):
        ms.write_feff_input(cluster, tmp_path / "x.inp", edge="K",
                            r_path_ang=5.0)
    with pytest.raises(ValueError, match="edge"):
        ms.write_feff_input(cluster, tmp_path / "x.inp", edge="K4")
    with pytest.raises(ValueError, match="vectors"):
        ms.feff_cluster(ms.absorber_pairs(frame, bulk.iter_pairs(
            frame, 5.0, vectors_within_ang=2.0), "Si"), int(ap.rows[0]),
            cluster_radius_ang=5.0)


def test_sampling_is_seeded_and_without_replacement():
    rows = np.array([3, 7, 11, 20])
    a = ms.sample_absorbers([0, 5, 10], rows, n_clusters=6, seed=42)
    assert a == ms.sample_absorbers([0, 5, 10], rows, n_clusters=6, seed=42)
    assert len(set(a)) == 6 and list(a) == sorted(a)
    assert all(f in (0, 5, 10) and r in rows for f, r in a)
    every = ms.sample_absorbers([0, 5, 10], rows, n_clusters=12, seed=1)
    assert len(set(every)) == 12
    with pytest.raises(ValueError, match="13 clusters"):
        ms.sample_absorbers([0, 5, 10], rows, n_clusters=13, seed=1)
    with pytest.raises(ValueError):
        ms.sample_absorbers([0, 0], rows, n_clusters=1, seed=1)


def test_export_writes_the_clusters_and_their_list(quartz_frame, tmp_path):
    traj = md_model.MemoryTrajectory([quartz_frame, quartz_frame])
    request = ms.FeffRequest(tmp_path / "feff", 5, 3, 4.5, "K")
    export = ms.export_feff_inputs(traj, "Si", request)
    assert len(export.clusters) == 5
    for record in export.clusters:
        text = Path(record.directory, "feff.inp").read_text(encoding="utf-8")
        assert f"TITLE MD frame {record.frame} Si{record.atom_id}" in text
        assert record.n_neighbours == len(_blocks(text, "\nATOMS", "END")) - 1
    manifest = export.manifest.read_text(encoding="utf-8").splitlines()
    assert any("method parameter: seed = 3" in line for line in manifest)
    # 4.5 Å is beyond half the 8.51 Å width of this box: stated
    assert any("periodic images of atoms already in it" in n
               for n in export.notes)
    assert sum(1 for line in manifest if not line.startswith("#")) == 6
    with pytest.raises(ValueError, match="already exist"):
        ms.export_feff_inputs(traj, "Si", request)
    with pytest.raises(ValueError, match="RPATH"):
        ms.FeffRequest(tmp_path, 1, 0, 2.0, "K")     # default RPATH 2.0 Å
    # the header names the trajectory and no bond-valence setting (the
    # cluster is cut by distance alone, and clusters.csv says 'not given')
    text = Path(export.clusters[0].directory, "feff.inp").read_text(
        encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("*")]
    assert "* source: <memory>" in header
    assert not any("ond-valence" in line or "threshold" in line
                   for line in header)
    assert export.provenance.as_lines() == [
        line[2:] for line in manifest if line.startswith("# ")][
            :len(export.provenance.as_lines())]
    assert all(r.n_potentials == 3 for r in export.clusters)  # Si0, Si, O


def test_a_second_draw_is_refused_and_overwrite_clears_feff_output(
        quartz_frame, tmp_path):
    """A second export into one out_dir rewrote clusters.csv and dropped the
    first draw from the record; overwrite=True kept FEFF's old output beside
    the new feff.inp."""
    traj = md_model.MemoryTrajectory([quartz_frame, quartz_frame])
    first = ms.export_feff_inputs(
        traj, "Si", ms.FeffRequest(tmp_path, 3, 1, 4.0, "K"))
    other = ms.FeffRequest(tmp_path, 3, 2, 4.0, "K")
    drawn = {(f, int(quartz_frame.atom_id[r])) for f, r in
             ms.sample_absorbers([0, 1], np.flatnonzero(
                 quartz_frame.elements == "Si"), n_clusters=3, seed=2)}
    assert drawn != {(c.frame, c.atom_id) for c in first.clusters}
    with pytest.raises(ValueError, match="clusters.csv already exists"):
        ms.export_feff_inputs(traj, "Si", other)
    # FEFF output of the first draw, then the first draw again on overwrite
    target = Path(first.clusters[0].directory)
    for name in ("files.dat", "feff0001.dat", "chi.dat", "pot.pad",
                 "log1.dat", "notes.txt"):
        (target / name).write_text("x", encoding="utf-8")
    again = ms.export_feff_inputs(
        traj, "Si", ms.FeffRequest(tmp_path, 3, 1, 4.0, "K", overwrite=True))
    assert sorted(p.name for p in target.iterdir()) == ["feff.inp",
                                                        "notes.txt"]
    assert any("FEFF output removed beside 1 rewritten feff.inp (5 files)"
               in n for n in again.notes)
    # another draw on overwrite: the first draw's directories are named
    third = ms.export_feff_inputs(
        traj, "Si", ms.FeffRequest(tmp_path, 3, 2, 4.0, "K", overwrite=True))
    stray = {Path(c.directory).name for c in first.clusters} - \
        {Path(c.directory).name for c in third.clusters}
    assert stray and any(f"{len(stray)} cluster director(ies) of an earlier "
                         "draw" in n for n in third.notes)


def test_a_cluster_with_many_potentials_is_commented(tmp_path):
    """FEFF8L stopped beyond 10 unique potentials and returned exit code 0;
    the input gave no sign of it."""
    elements = ("O", "Si", "B", "Na", "Al", "Li", "Ca", "Fe", "Mg", "K")
    vectors = np.array([[2.0 + 0.1 * n, 0.0, 0.0] for n in range(10)])
    cluster = ms.FeffCluster(
        absorber="Al", absorber_row=0, absorber_id=1, frame_label=0,
        timestep=0, elements=elements, vec_ang=vectors,
        d_ang=np.linalg.norm(vectors, axis=1), cluster_radius_ang=4.0)
    assert cluster.n_potentials == 11
    text = ms.write_feff_input(cluster, tmp_path / "feff.inp",
                               edge="K").read_text(encoding="utf-8")
    assert "holds 11 unique potentials (ipot 0..10)" in text
    few = dataclasses.replace(cluster, elements=("O",) * 10)
    assert "unique potentials" not in ms.write_feff_input(
        few, tmp_path / "few.inp", edge="K").read_text(encoding="utf-8")
    # the same through an export: one Zr with the ten elements 2.5 Å away
    # on a Fibonacci sphere; the list records it and the notes say it
    turn = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - (np.arange(10) + 0.5) * 2.0 / 10
    directions = np.column_stack([np.sqrt(1 - z * z) * np.cos(turn * np.arange(
        10)), np.sqrt(1 - z * z) * np.sin(turn * np.arange(10)), z])
    frame = md_model.frame_from_arrays(
        ("Zr",) + elements, np.vstack([[10.0, 10.0, 10.0],
                                       10.0 + 2.5 * directions]),
        box_ang=np.eye(3) * 20.0)
    export = ms.export_feff_inputs(
        md_model.MemoryTrajectory([frame]), "Zr",
        ms.FeffRequest(tmp_path / "many", 1, 0, 4.0, "K"))
    assert export.clusters[0].n_potentials == 11
    assert any("1 of 1 clusters hold more than 10 unique potentials (up to "
               "11)" in n for n in export.notes)


FILES_DAT = """ test                                    Feff8L (EXAFS)       0.1
 PATH  Rmax= 5.000,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%
 -----------------------------------------------------------------------
    file        sig2   amp ratio    deg    nlegs  r effective
 feff0001.dat 0.00000   100.000     4.000     2   1.6100
"""

FILES_DAT_TWO = FILES_DAT + \
    " feff0002.dat 0.00000     3.000     2.000     3   3.2000\n"

# chi.dat as FEFF8L's ff2x writes its header: path 2 (3 %) is below its 4 %
# filter, so only path 1 is summed
CHI_DAT = """# test                                        Feff8L (EXAFS)       0.1
# PATH  Rmax= 5.000,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%
#  S02=1.000                                        Global_sig2= 0.00000
#  Curved wave amplitude ratio filter   4.000%
#     file         sig2 tot  cw amp ratio   deg  nlegs   reff  inp sig2
#           1       0.00000    100.00      4.00     2   1.6100
#     1/   2 paths used
#  -----------------------------------------------------------------------
#       k          chi          mag           phase @#
    1.0000   1.000000E-02  1.000000E-02  0.0000
"""


def _feff_path(magnitude: float) -> str:
    rows = "\n".join(
        f"  {k:.3f}  1.2000E+01  {magnitude * k:.4E} -1.4000E+01  9.500E-01"
        f"  6.0000E+00  2.5000E+00" for k in (1.0, 2.0, 3.0, 4.0))
    return (" test\n"
            "   2   4.000   1.6100    2.6314    2.34296 nleg, deg, reff, "
            "rnrmav(bohr), edge\n"
            "        x         y         z   pot at#\n"
            "    -0.0000   -0.0000   -0.0000  0  14 Si       absorbing atom\n"
            "     0.8487   -1.0000    0.8836  1   8 O\n"
            "    k   real[2*phc]   mag[feff]  phase[feff] red factor   lambda"
            "     real[p]@#\n" + rows + "\n")


def test_chi_is_the_mean_of_the_clusters_paths(tmp_path):
    directories = []
    for n, magnitude in enumerate((0.3, 0.5)):
        directory = tmp_path / f"c{n}"
        directory.mkdir()
        (directory / "files.dat").write_text(FILES_DAT, encoding="utf-8")
        (directory / "feff0001.dat").write_text(_feff_path(magnitude),
                                                encoding="utf-8")
        directories.append(directory)
    k = np.array([1.5, 2.0, 2.5, 3.0])
    avg = ms.average_feff_chi(directories + [tmp_path / "absent"],
                              k_inv_ang=k, s02=0.8, path_set="all")
    single = [exafs.chi_from_paths(exafs.read_feff_directory(d), k=k,
                                   s02=0.8).chi for d in directories]
    assert avg.chi_mean == pytest.approx(np.mean(single, axis=0), rel=1e-14)
    assert avg.chi_std == pytest.approx(np.std(single, axis=0, ddof=1),
                                        rel=1e-12)
    assert avg.missing == {str(tmp_path / "absent"): "directory not found"}
    assert any("sigma^2 = 0 on every path" in n for n in avg.notes)
    assert not any("upper bound" in n for n in avg.notes)
    # numpy.interp would hold FEFF's last value beyond its k range: refused
    with pytest.raises(ValueError, match=r"no directory holds FEFF output.*"
                                         r"does not cover the grid"):
        ms.average_feff_chi(directories, k_inv_ang=[1.5, 4.5], s02=1.0,
                            path_set="all")
    one = ms.average_feff_chi(directories[:1], k_inv_ang=k, s02=0.8,
                              path_set="all")
    assert np.isnan(one.chi_std).all()
    assert any("one cluster" in n for n in one.notes)
    with pytest.raises(ValueError):
        ms.average_feff_chi(directories, k_inv_ang=k, s02=0.0, path_set="all")
    # without chi.dat the default path set has nothing to read
    with pytest.raises(ValueError, match="no chi.dat"):
        ms.average_feff_chi(directories, k_inv_ang=k, s02=0.8)


def _feff_run(directory, *, files_dat=FILES_DAT_TWO, chi_dat=CHI_DAT,
              paths=(1, 2), feff_inp=None):
    """A FEFF8L-format directory: feff.inp first, then files.dat, the path
    files and chi.dat, in the order FEFF writes them."""
    directory.mkdir()
    if feff_inp is not None:
        (directory / "feff.inp").write_text(feff_inp, encoding="utf-8")
    (directory / "files.dat").write_text(files_dat, encoding="utf-8")
    for index in paths:
        (directory / f"feff{index:04d}.dat").write_text(
            _feff_path(0.1 * index), encoding="utf-8")
    if chi_dat is not None:
        (directory / "chi.dat").write_text(chi_dat, encoding="utf-8")
    return directory


def test_chi_sums_feffs_own_path_set_and_leaves_out_broken_runs(tmp_path):
    """Every path file had been summed, including those FEFF's ff2x leaves
    out of chi.dat (8 % of max|chi| over k = 3-14 on a real glass); an
    incomplete or stale FEFF run was averaged without a word; the path count
    in the notes was the first cluster's."""
    k = np.array([1.5, 2.0, 2.5, 3.0])
    run = _feff_run(tmp_path / "run")
    paths = {p.index: p for p in exafs.read_feff_directory(run)}
    feff_own = ms.average_feff_chi([run], k_inv_ang=k, s02=1.0)
    assert feff_own.chi_mean == pytest.approx(exafs.chi_from_paths(
        [paths[1]], k=k, s02=1.0).chi, rel=1e-15)
    assert feff_own.n_paths == {str(run): 1}
    assert feff_own.n_listed == {str(run): 2}
    notes = " | ".join(feff_own.notes)
    assert "summed into chi.dat" in notes and "1 to 1 paths per cluster, " \
        "of 2 to 2 path files" in notes
    assert "amplitude ratio filter in chi.dat: 4 %" in notes
    every = ms.average_feff_chi([run], k_inv_ang=k, s02=1.0, path_set="all")
    assert every.chi_mean == pytest.approx(exafs.chi_from_paths(
        list(paths.values()), k=k, s02=1.0).chi, rel=1e-15)
    strong = ms.average_feff_chi([run], k_inv_ang=k, s02=1.0, path_set=50)
    assert strong.chi_mean.tolist() == feff_own.chi_mean.tolist()
    assert "amplitude ratio 50 % or more" in strong.path_set
    # broken runs are left out, each with its reason, beside a sound one
    inp = "TITLE t\nRPATH     4.00\nEND\n"
    broken = {
        "absent": _feff_run(tmp_path / "absent", paths=(1,)),
        "unlisted": _feff_run(tmp_path / "unlisted", files_dat=FILES_DAT,
                              chi_dat=CHI_DAT.replace("1/   2", "1/   1")),
        "rpath": _feff_run(tmp_path / "rpath", feff_inp=inp.replace(
            "4.00", "2.50").replace("RPATH", "RPATH")),
        "older": _feff_run(tmp_path / "older",
                           feff_inp=inp.replace("4.00", "5.00")),
        "count": _feff_run(tmp_path / "count",
                           chi_dat=CHI_DAT.replace("1/   2", "2/   2")),
    }
    stamp = (broken["older"] / "feff.inp").stat().st_mtime
    os.utime(broken["older"] / "files.dat", (stamp - 60.0, stamp - 60.0))
    second = _feff_run(tmp_path / "second")
    out = ms.average_feff_chi([run, second] + list(broken.values()),
                              k_inv_ang=k, s02=1.0)
    assert out.used == (str(run), str(second))
    reasons = {name: out.missing[str(path)] for name, path in broken.items()}
    assert "lists 1 path file(s) that are not in the directory (first: " \
        "feff0002.dat)" in reasons["absent"]
    assert "1 path file(s) are not in files.dat" in reasons["unlisted"]
    assert "Rmax = 5.000 Å while feff.inp says RPATH 2.5 Å" in reasons["rpath"]
    assert "older than feff.inp" in reasons["older"]
    assert "chi.dat says 2/2 paths used" in reasons["count"]
    assert any("5 director(ies) left out" in n for n in out.notes)
    rows = out.cluster_rows()
    assert [r["paths summed"] for r in rows if r["used"]] == [1, 1]
    with pytest.raises(ValueError, match="path_set"):
        ms.average_feff_chi([run], k_inv_ang=k, s02=1.0, path_set="some")


def test_a_run_counts_once_and_an_unchecked_run_is_named(tmp_path):
    """A directory given twice counted twice in the mean and narrowed the
    spread; a run with no feff.inp beside it was used with its staleness
    unchecked and no word."""
    k = np.array([1.5, 2.0, 2.5, 3.0])
    run = _feff_run(tmp_path / "run")
    (tmp_path / "x").mkdir()
    for twice in ([run, run], [run, tmp_path / "x" / ".." / "run"]):
        with pytest.raises(ValueError, match="given twice"):
            ms.average_feff_chi(twice, k_inv_ang=k, s02=1.0)
    bare = ms.average_feff_chi([run], k_inv_ang=k, s02=1.0)
    assert any("1 of 1 directories used hold no feff.inp" in n
               for n in bare.notes)
    # FEFF's Rmax (5.000 in files.dat) is the RPATH of the input beside it
    checked = _feff_run(tmp_path / "checked",
                        feff_inp="TITLE t\nRPATH     5.00\nEND\n")
    out = ms.average_feff_chi([checked], k_inv_ang=k, s02=1.0)
    assert out.used == (str(checked),)
    assert not any("hold no feff.inp" in n for n in out.notes)


def test_the_export_provenance_reaches_the_averaged_chi(quartz_frame,
                                                       tmp_path):
    traj = md_model.MemoryTrajectory([quartz_frame])
    export = ms.export_feff_inputs(
        traj, "Si", ms.FeffRequest(tmp_path / "x", 1, 0, 4.0, "K"))
    run = _feff_run(tmp_path / "run")
    k = np.array([1.5, 2.0])
    avg = ms.average_feff_chi([run], k_inv_ang=k, s02=1.0,
                              provenance=export.provenance)
    assert avg.as_sheets()["provenance"] == export.provenance.as_rows()
    bare = ms.average_feff_chi([run], k_inv_ang=k, s02=1.0)
    assert any("no provenance was given" in n for n in bare.notes)


# ---------------------------------------------------------------------------
# no verdicts, and no Qt
# ---------------------------------------------------------------------------

VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def test_no_note_or_message_carries_a_verdict(dimer, quartz_frame, tmp_path):
    frame, _ = dimer
    bonds = _bonds(frame)
    texts = []
    correlation = ms.Correlation("29Si", 0.0, (
        _angle_term(1.0, valid_range=(0.0, 10.0)), ms.Term("Qn", 1.0)),
        reference="t")
    pred = ms.predict_shifts(correlation, frame, bonds, formers={"Si"})
    texts += pred.notes
    five, user, values = _five_atoms()
    texts += ms.nmr_spectrum([ms.predict_shifts(
        user, five, None, formers=None, user_values=values)],
        np.linspace(0, 40, 50), fwhm_ppm=1.0, lineshape="lorentzian").notes
    dist, _ = _qn_distribution()
    texts += ms.compare_fractions(dist, ms.MeasuredFractions(
        "Si Qn", {3: 0.2}, "fit", complete=False)).notes
    traj = md_model.MemoryTrajectory([quartz_frame])
    texts += ms.md_exafs(traj, "Si", r_max_ang=4.2, dr_ang=0.02,
                         minimum=glass.MinimumMethod(
                             "first local minimum", 0.03, "midpoint", 2.0),
                         neighbours=["Si"]).notes
    # the notes added after the review: noisy limits, different totals,
    # grid sampling, lines beyond the grid, overwrite and path sets
    texts += ms.md_exafs(_noisy_shell_frames(2, seed=7), "Al", r_max_ang=8.0,
                         dr_ang=0.01, minimum=FIRST_LOCAL,
                         neighbours=["O"]).notes
    texts += ms.compare_fractions(
        Distribution.from_counts([{"B3": 5, "B4": 4, "other": 1}],
                                 name="B", kind="fraction",
                                 keys=("B3", "B4", "other")),
        ms.MeasuredFractions("B", {"B3": 0.6, "B4": 0.4}, "fit")).notes
    texts += ms.nmr_spectrum(_user_prediction([-90.0, 50.0]),
                             np.arange(-100.0, -80.0, 0.5), fwhm_ppm=0.2,
                             lineshape="gaussian").notes
    export_dir = tmp_path / "feff"
    ms.export_feff_inputs(traj, "Si", ms.FeffRequest(export_dir, 2, 0, 4.0,
                                                     "K"))
    texts += ms.export_feff_inputs(traj, "Si", ms.FeffRequest(
        export_dir, 2, 1, 4.0, "K", overwrite=True)).notes
    texts += ms.average_feff_chi(
        [_feff_run(tmp_path / "run"),
         _feff_run(tmp_path / "partial", paths=(1,))],
        k_inv_ang=[1.5, 2.0], s02=1.0).notes
    texts += ms.predict_shifts(ms.Correlation(
        "29Si", 0.0, (ms.Term("bridges to", 1.0, partner="B"),),
        reference="t"), frame, bonds, formers={"Si", "B"}).notes
    near = ms.absorber_pairs(quartz_frame, bulk.iter_pairs(quartz_frame, 3.0),
                             "Si")
    small = ms.feff_cluster(near, int(near.rows[0]), cluster_radius_ang=2.0)
    for call in (
            lambda: ms.predict_shifts(ms.Correlation("29Si", None, (),
                                                     reference=None),
                                      frame, bonds, formers={"Si"}),
            lambda: ms.MeasuredFractions("d", {"a": 0.5}, "s"),
            lambda: ms.compare_fractions(dist, ms.MeasuredFractions(
                "d", {"Q9": 1.0}, "s")),
            lambda: ms.absorber_rdf(ms.absorber_pairs(
                quartz_frame, bulk.iter_pairs(quartz_frame, 6.0), "Si"),
                r_max_ang=6.0, dr_ang=0.01),
            lambda: ms.write_feff_input(small, tmp_path / "f", edge="K"),
            lambda: ms.average_feff_chi([tmp_path], k_inv_ang=[1.0, 2.0],
                                        s02=1.0),
            lambda: ms.sample_absorbers([0], [1], n_clusters=2, seed=0),
            lambda: ms.MeasuredFractions("d", {"a": 0.3, "b": 0.3}, "s",
                                         sum_tol=0.5),
            lambda: ms.MeasuredFractions("d", {"a": 1.01, "b": -0.01}, "s",
                                         sum_tol=0.015),
            lambda: ms.Correlation("27Al", 0.0, (ms.Term("CN", 1.0), ms.Term(
                "user", 1.0, name="CN", unit="1")), reference="r"),
            lambda: ms.predict_shifts(ms.Correlation(
                "29Si", 0.0, (ms.Term("bonds to", 1.0, partner="Si"),),
                reference="t"), frame, bonds, formers=None),
            lambda: ms.predict_shifts(ms.Correlation(
                "29Si", 0.0, (ms.Term("bridges to", 1.0, partner="Al"),),
                reference="t"), frame, bonds, formers={"Si"}),
            lambda: ms.broadened_spectrum([0.0], [1.0, 0.0], fwhm_ppm=1.0,
                                          lineshape="gaussian"),
            lambda: ms.export_feff_inputs(traj, "Si", ms.FeffRequest(
                export_dir, 1, 3, 4.0, "K")),
            lambda: ms.shift_histogram(["x"], [0.0, 1.0])):
        try:
            call()
        except ValueError as error:
            texts.append(str(error))
        else:
            raise AssertionError("expected a refusal")
    assert len(texts) >= 20
    assert [t for t in texts if VERDICT.search(t)] == []


def test_no_string_in_the_module_carries_a_verdict():
    source = (ROOT / "facet" / "core" / "md_spectroscopy.py").read_text(
        encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value,
                                                                str)]
    assert len(strings) > 100
    assert [s for s in strings if VERDICT.search(s)] == []


def test_the_module_imports_without_qt():
    code = ("import sys, facet.core.md_spectroscopy;"
            "print('QT' if any(m.startswith(('PySide6', 'matplotlib')) "
            "for m in sys.modules) else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
