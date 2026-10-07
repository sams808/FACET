"""The frame-averaged result containers every MD analysis module shares.

Glass descriptors, rings, scattering, NMR from correlations and dynamics all
report through :mod:`facet.core.md_stats`, so a mistake here shows up in every
table of the MD path at once. What is pinned, and why:

* **The spread across frames.** Sample standard deviation, ddof = 1, checked
  against Python's ``statistics`` module, an independent implementation that
  sums in exact rational arithmetic. Two identical frames give exactly 0, one
  frame gives NaN with a note, never 0.
* **Frame order never changes a result.** Permuted frames give bit-identical
  means and spreads, on data where a plain sum in frame order does change in
  its last bits (shown in the test, so the check can fail). With frame labels
  the whole container, rows included, is identical.
* **Nothing is dropped.** A key missing from a frame is a 0 there, an
  unlisted key is appended, a sample outside the edges or not finite is
  counted, a frame that counted nothing is listed and left out of the mean
  with a note.
* **Fractions sum to 1 or are refused**, with a message naming the sum and
  the tolerance.
* **Binning follows numpy.histogram** for the samples inside the edges
  (checked against it), and the density integrates to the in-range fraction.
* **Provenance** states every input, unset ones as 'not given', and feeds
  ``exporters.write_csv``.
* **No verdicts and no Qt.**

All numbers here are synthetic test inputs, not reference values for any
material.
"""
from __future__ import annotations

import ast
import dataclasses
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from facet.core import md_model, md_stats
from facet.core.md_stats import (Distribution, Histogram, Provenance, Scalar,
                                 Series, check_fractions, mean_and_spread)

ROOT = Path(__file__).resolve().parent.parent


def _oracle(column) -> tuple[float, float]:
    """Mean and ddof = 1 spread from the stdlib, which sums exact fractions."""
    values = [float(v) for v in column]
    return statistics.fmean(values), statistics.stdev(values)


# ---------------------------------------------------------------------------
# check_fractions
# ---------------------------------------------------------------------------

def test_fraction_sets_that_sum_to_one_pass():
    check_fractions([0.25, 0.25, 0.5], "quarters")
    check_fractions([0.1] * 10, "tenths")                 # fsum: exactly 1.0
    check_fractions([1 / 3, 1 / 3, 1 / 3], "thirds")
    check_fractions({"Q3": 0.4, "Q4": 0.6}, "a mapping")
    check_fractions(np.array([1.0]), "one key")
    counts = np.array([7, 11, 13, 17, 19, 23, 29])
    check_fractions(counts / counts.sum(), "count / total")
    assert check_fractions([0.5, 0.5], "returns None") is None


def test_a_set_that_does_not_sum_to_one_names_the_sum_and_the_tolerance():
    with pytest.raises(ValueError) as error:
        check_fractions([0.25, 0.25, 0.4], "doctored Q^n")
    text = str(error.value)
    assert "doctored Q^n" in text
    assert repr(math.fsum([0.25, 0.25, 0.4])) in text      # the sum
    assert "1e-12" in text                                  # the tolerance
    # Just outside and just inside a caller's tolerance.
    with pytest.raises(ValueError, match="tolerance 1e-09"):
        check_fractions([0.5, 0.5 + 2e-9], "tight", tol=1e-9)
    check_fractions([0.5, 0.5 + 2e-9], "loose", tol=1e-8)


@pytest.mark.parametrize("values, match", [
    ([], "empty"),
    ([0.5, float("nan"), 0.5], "NaN or infinite"),
    ([1.5, -0.5], "below 0"),
    ([1.0 + 1e-6, 0.0], "above 1"),
    ([[0.5, 0.5], [0.5, 0.5]], "1-D"),
    ("0.5", "not numbers"),
    ([0.5, "half"], "not numbers"),
])
def test_impossible_fraction_sets_are_refused(values, match):
    with pytest.raises(ValueError, match=match):
        check_fractions(values, "set")


@pytest.mark.parametrize("tol", [-1e-12, float("nan"), float("inf"), "x", True])
def test_the_tolerance_is_checked(tol):
    with pytest.raises(ValueError, match="tol"):
        check_fractions([1.0], "set", tol=tol)


# ---------------------------------------------------------------------------
# mean_and_spread
# ---------------------------------------------------------------------------

def test_mean_and_spread_agree_with_the_statistics_module():
    rng = np.random.default_rng(2026)
    data = rng.normal(3.0, 0.7, size=(37, 5))
    mean, spread = mean_and_spread(data)
    for c in range(5):
        expected_mean, expected_std = _oracle(data[:, c])
        # fmean is fsum / n as well, so the means agree to the bit.
        assert mean[c] == expected_mean
        assert spread[c] == pytest.approx(expected_std, rel=1e-13)
    # The same numbers as a 1-D array (a Scalar) give the same bits as a
    # column of a 2-D one (a Distribution); numpy's own sums differ here.
    scalar_mean, scalar_std = mean_and_spread(data[:, 0])
    assert scalar_mean.shape == () and float(scalar_mean) == mean[0]
    assert float(scalar_std) == spread[0]


def test_identical_frames_give_exactly_zero_spread():
    row = np.array([0.1, 1 / 3, 2 / 3, 0.7, 1e-300, 123456.789])
    for n_frames in (2, 3, 7):
        mean, spread = mean_and_spread(np.tile(row, (n_frames, 1)))
        assert np.array_equal(mean, row)        # the value itself, not a sum / n
        assert np.array_equal(spread, np.zeros_like(row))


def test_a_column_of_signed_zeros_averages_to_plus_zero_in_either_order():
    for column in ([[0.0], [-0.0]], [[-0.0], [0.0]]):
        mean, spread = mean_and_spread(column)
        assert mean[0] == 0.0 and not np.signbit(mean[0]) and spread[0] == 0.0


def test_one_frame_has_no_spread():
    mean, spread = mean_and_spread([[0.2, 0.8]])
    assert np.array_equal(mean, [0.2, 0.8])
    assert np.isnan(spread).all()


def test_frame_order_never_changes_the_mean_or_the_spread_to_the_last_bit():
    rng = np.random.default_rng(7)
    data = rng.random((50, 40)) * 10.0 ** rng.integers(-3, 4, size=(50, 40))
    mean, spread = mean_and_spread(data)
    naive = data.sum(axis=0)
    order_mattered = False
    for _ in range(20):
        shuffled = data[rng.permutation(50)]
        other_mean, other_spread = mean_and_spread(shuffled)
        assert np.array_equal(other_mean, mean)
        assert np.array_equal(other_spread, spread)
        # A plain running sum in frame order does change with the order.
        running = np.zeros(40)
        for row in shuffled:
            running = running + row
        order_mattered |= not np.array_equal(running, naive)
    assert order_mattered


def test_nan_propagates_and_infinity_is_refused():
    mean, spread = mean_and_spread([[1.0, np.nan], [2.0, 3.0]])
    assert mean[0] == 1.5 and np.isnan(mean[1]) and np.isnan(spread[1])
    with pytest.raises(ValueError, match="infinite"):
        mean_and_spread([[1.0, np.inf], [2.0, 3.0]])
    with pytest.raises(ValueError, match="no frame"):
        mean_and_spread(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="no frame"):
        mean_and_spread(4.0)


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

def test_a_key_missing_from_a_frame_counts_zero_and_is_not_dropped():
    frames = [{4: 18, 3: 2}, {4: 20}, {4: 17, 3: 2, 5: 1}]
    counts = Distribution.from_counts(frames, name="Si CN", kind="count")
    assert counts.keys == (3, 4, 5)
    assert np.array_equal(counts.per_frame, [[2, 18, 0], [0, 20, 0],
                                             [2, 17, 1]])
    assert np.array_equal(counts.n_items, [20, 20, 20])
    fractions = Distribution.from_counts(frames, name="Si CN", kind="fraction")
    assert np.array_equal(fractions.per_frame,
                          [[0.1, 0.9, 0.0], [0.0, 1.0, 0.0],
                           [0.1, 0.85, 0.05]])
    # The mean is over all three frames, a 0 where the key was absent.
    for c in range(3):
        expected_mean, expected_std = _oracle(fractions.per_frame[:, c])
        assert fractions.mean[c] == pytest.approx(expected_mean, rel=1e-14)
        assert fractions.std[c] == pytest.approx(expected_std, rel=1e-13)
    assert fractions.mean[2] == pytest.approx(0.05 / 3, rel=1e-14)
    check_fractions(fractions.mean, "mean")
    assert fractions.n_frames == fractions.n_frames_used == 3
    assert fractions.notes == ()


def test_to_fractions_equals_from_counts_as_fractions():
    frames = [{"B3": 7, "B4": 3}, {"B3": 6, "B4": 4}]
    direct = Distribution.from_counts(frames, name="N4", kind="fraction")
    via = Distribution.from_counts(frames, name="N4", kind="count").to_fractions()
    for label in ("keys", "kind", "n_frames", "frames_empty", "notes"):
        assert getattr(direct, label) == getattr(via, label)
    for label in ("per_frame", "n_items", "frames", "mean", "std"):
        assert np.array_equal(getattr(direct, label), getattr(via, label))
    with pytest.raises(ValueError, match="already holds fractions"):
        direct.to_fractions()


def test_given_keys_fix_the_order_and_unlisted_keys_are_appended_with_a_note():
    frames = [{"Q4": 5, "Q3": 1}, {"Q4": 4, "Q5": 1, "Q3": 1}]
    dist = Distribution.from_counts(frames, name="Si Qn", kind="fraction",
                                    keys=("Q0", "Q1", "Q2", "Q3", "Q4"))
    assert dist.keys == ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5")
    assert np.array_equal(dist.per_frame[:, :3], np.zeros((2, 3)))
    assert any("Q5" in n and "appended" in n for n in dist.notes)
    with pytest.raises(ValueError, match="more than once"):
        Distribution.from_counts(frames, name="Si Qn", kind="count",
                                 keys=("Q3", "Q3", "Q4"))


def test_keys_sort_numbers_then_text_then_tuples():
    frames = [{("Si", "Al"): 1, 5: 1, "other": 1, 4: 2, np.int64(6): 1,
               ("Si", "Si"): 3}]
    dist = Distribution.from_counts(frames, name="mixed", kind="count")
    assert dist.keys == (4, 5, 6, "other", ("Si", "Al"), ("Si", "Si"))
    assert type(dist.keys[2]) is int                    # numpy scalar unwrapped
    labels = [row["key"] for row in dist.as_rows()]
    assert labels == ["4", "5", "6", "other", "Si-Al", "Si-Si"]


def test_two_identical_frames_give_zero_spread_and_permuted_frames_are_identical():
    frame_a = {"F-Al1Na2": 3, "F-Al2": 5, "F-Na3": 2}
    frame_b = {"F-Al1Na2": 4, "F-Al2": 5, "F-Na3": 1}
    frame_c = {"F-Al1Na2": 2, "F-Al2": 6, "F-Na3": 2}
    twice = Distribution.from_counts([frame_a, frame_a], name="F env",
                                     kind="fraction")
    assert np.array_equal(twice.std, np.zeros(3))
    assert np.array_equal(twice.mean, twice.per_frame[0])

    forward = Distribution.from_counts([frame_a, frame_b, frame_c],
                                       name="F env", kind="fraction",
                                       frames=[10, 20, 30])
    backward = Distribution.from_counts([frame_c, frame_a, frame_b],
                                        name="F env", kind="fraction",
                                        frames=[30, 10, 20])
    for label in ("per_frame", "n_items", "frames", "mean", "std"):
        assert np.array_equal(getattr(forward, label), getattr(backward, label))
    assert forward.frames.tolist() == [10, 20, 30]
    # Without labels the rows stay in the order given; the mean does not move.
    unlabelled = Distribution.from_counts([frame_c, frame_a, frame_b],
                                          name="F env", kind="fraction")
    assert np.array_equal(unlabelled.mean, forward.mean)
    assert np.array_equal(unlabelled.std, forward.std)


def test_one_frame_gives_nan_spread_with_a_note():
    dist = Distribution.from_counts([{4: 10}], name="Si CN", kind="fraction")
    assert np.array_equal(dist.mean, [1.0]) and np.isnan(dist.std).all()
    assert any("one frame" in n and "NaN" in n for n in dist.notes)
    blocks = Distribution.from_counts([{4: 10}], name="Si CN", kind="count",
                                      row_kind="block")
    assert any("one block" in n for n in blocks.notes)


def test_a_frame_that_counted_nothing_is_listed_and_left_out_of_the_mean():
    dist = Distribution.from_counts([{3: 1, 4: 3}, {}, {3: 2, 4: 2}],
                                    name="halide env", kind="fraction",
                                    frames=[0, 5, 9])
    assert dist.frames_empty == (5,)
    assert np.isnan(dist.per_frame[1]).all()
    assert dist.n_frames == 3 and dist.n_frames_used == 2
    assert np.array_equal(dist.mean, [0.375, 0.625])
    assert any("counted no item" in n and "frames 5" in n for n in dist.notes)
    # As counts, an empty frame is a frame of zeros, in the mean.
    counts = Distribution.from_counts([{3: 1, 4: 3}, {}, {3: 2, 4: 2}],
                                      name="halide env", kind="count")
    assert counts.frames_empty == () and counts.mean[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no frame counted any item"):
        Distribution.from_counts([{}, {}], name="empty", kind="fraction",
                                 keys=("Q4",))
    with pytest.raises(ValueError, match="no category"):
        Distribution.from_counts([{}, {}], name="empty", kind="count")


def test_a_fraction_table_built_directly_is_checked_row_by_row():
    table = Distribution("Al CN", ("Al4", "Al5", "Al6"), "fraction",
                         [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]], n_items=[10, 10])
    check_fractions(table.mean, "mean")
    with pytest.raises(ValueError, match=r"Al CN, frame 1: .*sum to"):
        Distribution("Al CN", ("Al4", "Al5", "Al6"), "fraction",
                     [[0.7, 0.2, 0.1], [0.6, 0.3, 0.2]], n_items=[10, 10])
    with pytest.raises(ValueError, match=r"frame 0: a fraction is -0.1, below 0"):
        Distribution("Al CN", ("Al4", "Al5"), "fraction", [[1.1, -0.1]],
                     n_items=[10])
    with pytest.raises(ValueError, match=r"frame 3: 1 of 2 fractions are NaN"):
        Distribution("Al CN", ("Al4", "Al5"), "fraction", [[1.0, 0.0],
                                                           [0.5, np.nan]],
                     n_items=[10, 10], frames=[2, 3])
    with pytest.raises(ValueError, match="need n_items"):
        Distribution("Al CN", ("Al4",), "fraction", [[1.0]])
    with pytest.raises(ValueError, match="not NaN"):
        Distribution("Al CN", ("Al4",), "fraction", [[1.0], [0.0]],
                     n_items=[3, 0])
    with pytest.raises(ValueError, match="row sums"):
        Distribution("Al CN", ("Al4", "Al5"), "count", [[3, 1]], n_items=[5])


@pytest.mark.parametrize("rows, match", [
    ([{4: -1}], "negative"),
    ([{4: 2.5}], "whole number"),
    ([{4: float("nan")}], "NaN or infinite"),
    ([{4: True}], "not a number"),
    ([{4: "3"}], "whole numbers"),
    ([{float("nan"): 3}], "not finite"),
    ([[4, 3]], "not a mapping"),
    ({4: 3}, "single mapping"),
    ([], "no rows"),
])
def test_impossible_counts_are_refused(rows, match):
    with pytest.raises(ValueError, match=match):
        Distribution.from_counts(rows, name="CN", kind="fraction")


def test_bad_arguments_of_a_distribution_are_refused():
    with pytest.raises(ValueError, match="kind"):
        Distribution.from_counts([{4: 1}], name="CN", kind="percent")
    with pytest.raises(ValueError, match="row_kind"):
        Distribution.from_counts([{4: 1}], name="CN", kind="count",
                                 row_kind="snapshot")
    with pytest.raises(ValueError, match="name"):
        Distribution.from_counts([{4: 1}], name="", kind="count")
    with pytest.raises(ValueError, match="occur more than once"):
        Distribution.from_counts([{4: 1}, {4: 1}], name="CN", kind="count",
                                 frames=[3, 3])
    with pytest.raises(ValueError, match="one label per row"):
        Distribution.from_counts([{4: 1}, {4: 1}], name="CN", kind="count",
                                 frames=[3])
    with pytest.raises(ValueError, match="whole number of 0 or more"):
        Distribution.from_counts([{4: 1}], name="CN", kind="count",
                                 frames=[-1])
    with pytest.raises(ValueError, match="2 columns for 1 keys"):
        Distribution("CN", (4,), "count", [[1, 2]])
    with pytest.raises(ValueError, match="not hashable"):
        Distribution.from_counts([{4: 1}], name="CN", kind="count",
                                 keys=([4],))


def test_a_distribution_is_read_only_and_exports_rows():
    dist = Distribution.from_counts([{4: 9, 5: 1}, {4: 10}], name="Si CN",
                                    kind="fraction", frames=[4, 8])
    with pytest.raises(dataclasses.FrozenInstanceError):
        dist.kind = "count"
    for label in ("per_frame", "n_items", "frames", "mean", "std"):
        with pytest.raises(ValueError):
            getattr(dist, label)[0] = 0
    rows = dist.as_rows(per_frame=True)
    assert [r["key"] for r in rows] == ["4", "5"]
    assert rows[1]["mean"] == pytest.approx(0.05)
    assert rows[0]["frame 4"] == 0.9 and rows[0]["frame 8"] == 1.0
    assert rows[0]["frames used"] == 2 and rows[0]["items per frame"] == "10"
    assert dist.as_dict()[5][0] == pytest.approx(0.05)
    noted = dist.with_notes("BV-cut at v_bond = 0.075 v.u.")
    assert noted.notes[-1] == "BV-cut at v_bond = 0.075 v.u."
    assert np.array_equal(noted.mean, dist.mean)
    assert noted.with_notes("BV-cut at v_bond = 0.075 v.u.").notes == noted.notes


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

EDGES_DEG = np.linspace(0.0, 180.0, 37)                 # 5 deg bins, a test choice


def test_binning_matches_numpy_histogram_inside_the_edges():
    rng = np.random.default_rng(11)
    frames = [rng.uniform(-20.0, 200.0, size=n) for n in (500, 731, 1)]
    frames[0][:3] = [0.0, 180.0, 5.0]                     # both outer edges, a bin edge
    hist = Histogram.from_samples(frames, EDGES_DEG, name="Si-O-Si",
                                  unit="deg")
    for k, samples in enumerate(frames):
        expected, _ = np.histogram(samples, bins=EDGES_DEG)
        assert np.array_equal(hist.counts[k], expected)
        assert hist.n_below[k] == (samples < 0.0).sum()
        assert hist.n_above[k] == (samples > 180.0).sum()
    # Nothing is lost: every sample is in a bin or in a tally.
    assert np.array_equal(hist.n_samples, [500, 731, 1])
    assert np.array_equal(hist.counts.sum(1) + hist.n_below + hist.n_above
                          + hist.n_nonfinite, hist.n_samples)
    assert hist.counts[0, 0] >= 1 and hist.counts[0, -1] >= 1
    assert hist.value_unit == "count"
    assert np.array_equal(hist.per_frame, hist.counts.astype(float))


def test_samples_outside_the_edges_and_undefined_samples_are_counted_and_noted():
    hist = Histogram.from_samples(
        [np.array([-1.0, 10.0, 181.0, 190.0, np.nan]),
         np.array([90.0, np.inf, -np.inf])],
        EDGES_DEG, name="O-T-O", unit="deg")
    assert hist.n_below.tolist() == [1, 0]
    assert hist.n_above.tolist() == [2, 0]
    assert hist.n_nonfinite.tolist() == [1, 2]
    assert hist.counts.sum() == 2
    outside = [n for n in hist.notes if "below the first edge" in n]
    assert len(outside) == 1
    assert "1 sample(s) below" in outside[0] and "2 above" in outside[0]
    assert "0.0 deg" in outside[0] and "180.0 deg" in outside[0]
    assert any("3 sample(s) are NaN or infinite" in n for n in hist.notes)


def test_the_density_integrates_to_the_in_range_fraction():
    rng = np.random.default_rng(3)
    inside = [rng.uniform(0.0, 180.0, size=400) for _ in range(3)]
    hist = Histogram.from_samples(inside, EDGES_DEG, name="angles",
                                  unit="deg", density=True)
    assert hist.value_unit == "1/deg"
    for row in hist.per_frame:
        assert math.fsum(row * hist.widths) == pytest.approx(1.0, abs=1e-14)
    assert math.fsum(hist.mean * hist.widths) == pytest.approx(1.0, abs=1e-14)

    with_outside = Histogram.from_samples(
        [np.array([10.0, 20.0, 200.0, np.nan])], EDGES_DEG, name="angles",
        unit="deg", density=True)
    # Normalised by the 3 finite samples; the NaN is in no denominator.
    assert math.fsum(with_outside.per_frame[0] * with_outside.widths) == \
        pytest.approx(2 / 3, abs=1e-15)
    assert any("integrates" in n and repr(2 / 3) in n
               for n in with_outside.notes)
    phi = Histogram.from_samples([[0.1, 0.3]], [0.0, 0.5, 1.0], name="phi",
                                 unit="1", density=True)
    assert phi.value_unit == "1"


def test_a_density_frame_with_no_finite_sample_is_left_out_with_a_note():
    hist = Histogram.from_samples([[10.0, 20.0], [np.nan], [30.0]], EDGES_DEG,
                                  name="phi Bi", unit="deg", density=True)
    assert hist.frames_empty == (1,) and hist.n_frames_used == 2
    assert np.isnan(hist.per_frame[1]).all()
    assert not np.isnan(hist.mean).any()
    assert any("hold no finite sample" in n for n in hist.notes)
    with pytest.raises(ValueError, match="no frame holds a finite sample"):
        Histogram.from_samples([[np.nan]], EDGES_DEG, name="phi", unit="1",
                               density=True)


def test_histogram_frames_identical_permuted_and_single():
    rng = np.random.default_rng(5)
    frames = [rng.uniform(0, 180, size=n) for n in (300, 280, 310, 299)]
    twice = Histogram.from_samples([frames[0], frames[0]], EDGES_DEG,
                                   name="a", unit="deg", density=True)
    assert np.array_equal(twice.std, np.zeros(36))
    assert np.array_equal(twice.mean, twice.per_frame[0])
    one = Histogram.from_samples(frames, EDGES_DEG, name="a", unit="deg",
                                 density=True, frames=[0, 1, 2, 3])
    other = Histogram.from_samples([frames[i] for i in (2, 0, 3, 1)],
                                   EDGES_DEG, name="a", unit="deg",
                                   density=True, frames=[2, 0, 3, 1])
    for label in ("counts", "per_frame", "mean", "std", "frames", "n_samples"):
        assert np.array_equal(getattr(one, label), getattr(other, label))
    single = Histogram.from_samples([frames[0]], EDGES_DEG, name="a",
                                    unit="deg")
    assert np.isnan(single.std).all()
    assert any("one frame" in n for n in single.notes)


@pytest.mark.parametrize("edges, match", [
    ([1.0], "at least two"),
    ([0.0, 1.0, 1.0], "increase strictly"),
    ([0.0, np.nan], "NaN or infinite"),
    ([[0.0, 1.0]], "at least two"),
])
def test_bad_edges_are_refused(edges, match):
    with pytest.raises(ValueError, match=match):
        Histogram.from_samples([[0.5]], edges, name="h", unit="1")


def test_bad_histogram_input_is_refused():
    with pytest.raises(ValueError, match="1-D"):
        Histogram.from_samples([[[1.0]]], [0, 1], name="h", unit="1")
    with pytest.raises(ValueError, match="no rows"):
        Histogram.from_samples([], [0, 1], name="h", unit="1")
    with pytest.raises(ValueError, match="True or False"):
        Histogram.from_samples([[0.5]], [0, 1], name="h", unit="1",
                               density="yes")
    with pytest.raises(ValueError, match="shape"):
        Histogram("h", "1", [0, 1, 2], [[1]], [0], [0], [0])


def test_histogram_rows_carry_the_unit_and_edges():
    hist = Histogram.from_samples([[1.0, 7.0], [2.0, 8.0]], [0.0, 5.0, 10.0],
                                  name="d", unit="deg")
    rows = hist.as_rows(per_frame=True)
    assert rows[0]["bin low (deg)"] == 0.0 and rows[1]["bin high (deg)"] == 10.0
    assert rows[0]["mean (count)"] == 1.0 and rows[0]["std (count)"] == 0.0
    assert rows[1]["frame 1"] == 1.0
    assert np.array_equal(hist.centres, [2.5, 7.5])


# ---------------------------------------------------------------------------
# Series and Scalar
# ---------------------------------------------------------------------------

R_ANG = np.arange(0.01, 10.0, 0.01)                       # a grid, a test choice


def test_series_mean_spread_identical_and_permuted_frames():
    rng = np.random.default_rng(9)
    rows = [rng.random(R_ANG.size) for _ in range(4)]
    series = Series.from_frames(R_ANG, rows, name="g Si-O", axis_name="r_ang",
                                axis_unit="Å", value_unit="1",
                                frames=[0, 1, 2, 3])
    for p in (0, 500, R_ANG.size - 1):
        expected_mean, expected_std = _oracle([row[p] for row in rows])
        assert series.mean[p] == pytest.approx(expected_mean, rel=1e-14)
        assert series.std[p] == pytest.approx(expected_std, rel=1e-13)
    permuted = Series.from_frames(R_ANG, [rows[i] for i in (3, 1, 0, 2)],
                                  name="g Si-O", axis_name="r_ang",
                                  axis_unit="Å", value_unit="1",
                                  frames=[3, 1, 0, 2])
    for label in ("per_frame", "frames", "mean", "std"):
        assert np.array_equal(getattr(series, label), getattr(permuted, label))
    twice = Series.from_frames(R_ANG, [rows[0], rows[0]], name="g",
                               axis_name="r_ang", axis_unit="Å",
                               value_unit="1")
    assert np.array_equal(twice.std, np.zeros(R_ANG.size))
    single = Series.from_frames(R_ANG, [rows[0]], name="MSD Na",
                                axis_name="t_ps", axis_unit="ps",
                                value_unit="Å^2", row_kind="time origin")
    assert np.isnan(single.std).all()
    assert any("one time origin" in n for n in single.notes)
    rows_out = series.as_rows()
    assert rows_out[0]["r_ang"] == pytest.approx(0.01)
    assert "mean (1)" in rows_out[0] and rows_out[0]["frames used"] == 4


def test_series_nan_is_noted_and_bad_input_refused():
    series = Series.from_frames([1.0, 2.0, 3.0], [[1, 2, 3], [1, np.nan, 3]],
                                name="S(q)", axis_name="q_inv_ang",
                                axis_unit="1/Å", value_unit="1")
    assert np.isnan(series.mean[1]) and series.mean[0] == 1.0
    assert any("1 value(s) are NaN, in 1 frame;" in n for n in series.notes)
    for axis, rows, match in (
            ([1.0, 1.0], [[1, 2]], "increase strictly"),
            ([2.0, 1.0], [[1, 2]], "increase strictly"),
            ([1.0, np.nan], [[1, 2]], "NaN or infinite"),
            ([1.0, 2.0], [[1, 2, 3]], "values per row"),
            ([1.0, 2.0], [[1, 2], [1]], "every row needs"),
            ([1.0, 2.0], [[1, np.inf]], "infinite"),
            ([1.0, 2.0], [], "no rows")):
        with pytest.raises(ValueError, match=match):
            Series.from_frames(axis, rows, name="s", axis_name="q_inv_ang",
                               axis_unit="1/Å", value_unit="1")
    with pytest.raises(ValueError, match="axis_name"):
        Series.from_frames([1.0], [[1.0]], name="s", axis_name="",
                           axis_unit="Å", value_unit="1")


def test_scalar_mean_spread_permutation_and_single_frame():
    values = [4.02, 4.05, 3.98, 4.01]
    scalar = Scalar("mean Si CN", "1", values, frames=[1, 2, 3, 4])
    expected_mean, expected_std = _oracle(values)
    assert scalar.mean == pytest.approx(expected_mean, rel=1e-14)
    assert scalar.std == pytest.approx(expected_std, rel=1e-13)
    permuted = Scalar("mean Si CN", "1", values[::-1], frames=[4, 3, 2, 1])
    assert (permuted.mean, permuted.std) == (scalar.mean, scalar.std)
    assert np.array_equal(permuted.per_frame, scalar.per_frame)
    twice = Scalar("x", "1", [0.1, 0.1])
    assert twice.mean == 0.1 and twice.std == 0.0
    single = Scalar("density", "g/cm^3", [2.2])
    assert single.mean == 2.2 and math.isnan(single.std)
    assert any("one frame" in n for n in single.notes)
    assert single.as_rows()[0]["unit"] == "g/cm^3"
    with pytest.raises(ValueError, match="infinite"):
        Scalar("x", "1", [1.0, np.inf])
    with pytest.raises(ValueError, match="1-dimensional"):
        Scalar("x", "1", [[1.0]])


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _two_frame_trajectory():
    box = np.eye(3) * 10.0
    frames = [md_model.frame_from_arrays(["Si", "O", "O"],
                                         [[1.0, 1, 1], [2.6, 1, 1],
                                          [1.0, 2.6, 1]], box_ang=box,
                                         timestep=step, time_ps=step / 1000)
              for step in (100, 200)]
    return md_model.MemoryTrajectory(frames, source_path="synthetic.dump",
                                     file_format="lammps-dump",
                                     type_map={1: "Si", 2: "O"},
                                     type_map_source="user",
                                     notes=["2 frames, 3 atoms"])


def test_provenance_from_a_trajectory_states_every_input():
    traj = _two_frame_trajectory()
    ox = md_model.model_oxidation(["Si", "O"])
    prov = Provenance.from_trajectory(
        traj, frames_used=[1, 0], ox=ox,
        params=type("Params", (), {"name": "Brese & O'Keeffe 1991",
                                   "source": "Acta Cryst. B47 (1991) 192"})(),
        v_bond_vu=0.075, v_list_vu=0.02, r_search_ang=6.0,
        formers={"Si"}, cutoffs_ang={("Si", "O"): 2.2},
        cutoff_sources={("Si", "O"): "first minimum of g(r), raw"},
        method_parameters={"angle_bin_deg": 5.0, "smooth_sigma_ang": None,
                           "edges": [0.0, 90.0, 180.0]},
        estimated_pairs={"Ti4+-O": 12}, notes=["test note"],
        created="2026-10-06 12:00")
    assert prov.frames_used == (0, 1) and prov.timesteps == (100, 200)
    assert prov.method_parameters["edges"] == (0.0, 90.0, 180.0)
    lines = prov.as_lines()
    text = "\n".join(lines)
    for expected in ("program: FACET ", "created: 2026-10-06 12:00",
                     "source file: synthetic.dump",
                     "file format: lammps-dump", "frames used: 2: 0, 1",
                     "timesteps: 100, 200", "time (ps): 0.1 to 0.2",
                     "frames skipped: none",
                     "type map: 1 -> Si, 2 -> O", "type map source: user",
                     "oxidation states: O2- (common), Si4+ (common)",
                     "bond-valence parameters: Brese & O'Keeffe 1991",
                     "bond threshold v_bond: 0.075 v.u.",
                     "tabulation threshold v_list: 0.02 v.u.",
                     "search radius: 6.0 Å", "network formers: Si",
                     "distance cutoff: Si-O: 2.2 Å (first minimum of g(r), raw)",
                     "method parameter: angle_bin_deg = 5.0",
                     "method parameter: smooth_sigma_ang = None",
                     "method parameter: edges = (0.0, 90.0, 180.0)",
                     "estimated bond-valence parameter: Ti4+-O: 12 contacts",
                     "spread: std is the spread", "note: 2 frames, 3 atoms",
                     "note: test note"):
        assert expected in text, expected
    assert prov.as_rows()[0] == {"field": "program",
                                 "value": f"FACET {md_stats.FACET_VERSION}"}


def test_unset_inputs_read_as_not_given_and_formers_are_never_assumed():
    prov = Provenance(source_path="m.xyz", file_format="extxyz",
                      frames_used=(0,), created="now")
    text = "\n".join(prov.as_lines())
    assert "network formers: not given" in text
    assert "oxidation states: not given" in text
    assert "bond threshold v_bond: not given" in text
    assert "timesteps: not given in the file" in text
    empty = Provenance(source_path="m.xyz", file_format="extxyz",
                       frames_used=(0,), formers=(), created="now")
    assert "network formers: none (an empty set was given)" in \
        "\n".join(empty.as_lines())


def test_frames_used_are_compacted_and_skips_carry_their_reasons():
    prov = Provenance(source_path="m.dump", file_format="lammps-dump",
                      frames_used=list(range(0, 50, 5)) + [51, 52, 53, 70],
                      frames_skipped={60: "unreadable: truncated line"},
                      file_blocks_skipped={99: "truncated: 3 of 6 atoms"},
                      created="now")
    text = "\n".join(prov.as_lines())
    assert "frames used: 14: 0-45 step 5, 51-53, 70" in text
    assert "frame skipped: 60: unreadable: truncated line" in text
    assert ("file block skipped by the reader: position 99: truncated: 3 of 6 "
            "atoms") in text


@pytest.mark.parametrize("changes, match", [
    ({"frames_used": ()}, "empty"),
    ({"frames_used": (1, 1)}, "more than once"),
    ({"frames_used": (0, 1), "frames_skipped": {1: "x"}}, "both used and "
                                                          "skipped"),
    ({"frames_skipped": {2: ""}}, "reason"),
    ({"formers": "SiO"}, "not one string"),
    ({"formers": {"Xx"}}, "not an element symbol"),
    ({"cutoffs_ang": {("Si", "O"): 2.0}}, "needs its source"),
    ({"cutoffs_ang": {("Si", "O"): -2.0},
      "cutoff_sources": {("Si", "O"): "user"}}, "above 0"),
    ({"method_parameters": {"edges": np.zeros(3)}}, "ndarray"),
    ({"v_bond_vu": float("nan")}, "finite"),
    ({"timesteps": (1, 2, 3)}, "one value per frame"),
    ({"ox": {"Si": 4}}, "ModelOxidation"),
    ({"source_path": ""}, "source_path"),
])
def test_impossible_provenance_is_refused(changes, match):
    base = {"source_path": "m.dump", "file_format": "lammps-dump",
            "frames_used": (0,), "created": "now"}
    base.update(changes)
    with pytest.raises(ValueError, match=match):
        Provenance(**base)


def test_from_trajectory_refuses_a_frame_it_does_not_hold():
    with pytest.raises(ValueError, match="outside"):
        Provenance.from_trajectory(_two_frame_trajectory(), frames_used=[2])


def test_the_header_and_rows_feed_write_csv(tmp_path):
    from facet.core import exporters
    dist = Distribution.from_counts([{"Q4": 9, "Q3": 1}, {"Q4": 10}],
                                    name="Si Qn", kind="fraction")
    prov = Provenance.from_trajectory(_two_frame_trajectory(),
                                      frames_used=[0, 1], formers={"Si"},
                                      created="2026-10-06 12:00")
    path = exporters.write_csv(dist.as_rows(per_frame=True),
                               tmp_path / "qn.csv", provenance=prov.as_lines())
    text = path.read_text(encoding="utf-8").splitlines()
    header = [line for line in text if line.startswith("# ")]
    assert header == [f"# {line}" for line in prov.as_lines()]
    body = [line for line in text if not line.startswith("#")]
    assert body[0].startswith("descriptor,key,kind,mean,std")
    assert body[1].startswith("Si Qn,Q3,fraction,0.05,")


# ---------------------------------------------------------------------------
# no verdicts, and no Qt
# ---------------------------------------------------------------------------

VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def _every_note_and_message() -> list[str]:
    out: list[str] = []
    out += Distribution.from_counts([{3: 1}, {}], name="d", kind="fraction",
                                    keys=(4,)).notes
    out += Distribution.from_counts([{3: 1}], name="d", kind="count").notes
    out += Histogram.from_samples([[-1.0, 200.0, np.nan], [np.nan]], EDGES_DEG,
                                  name="h", unit="deg", density=True).notes
    out += Series.from_frames([1.0], [[np.nan]], name="s", axis_name="r_ang",
                              axis_unit="Å", value_unit="1").notes
    out += Scalar("c", "1", [np.nan]).notes
    out += Provenance.from_trajectory(
        _two_frame_trajectory(), frames_used=[0], created="now").as_lines()
    for call in (
            lambda: check_fractions([0.5, 0.4], "f"),
            lambda: check_fractions([1.5, -0.5], "f"),
            lambda: check_fractions([], "f"),
            lambda: check_fractions([np.nan], "f"),
            lambda: mean_and_spread([np.inf]),
            lambda: Distribution.from_counts([{4: -1}], name="d", kind="count"),
            lambda: Distribution.from_counts([{4: 0.5}], name="d",
                                             kind="count"),
            lambda: Distribution.from_counts([{}], name="d", kind="fraction",
                                             keys=(1,)),
            lambda: Distribution("d", (1,), "fraction", [[0.5]], n_items=[2]),
            lambda: Histogram.from_samples([[1.0]], [1.0, 1.0], name="h",
                                           unit="1"),
            lambda: Series.from_frames([2.0, 1.0], [[1, 2]], name="s",
                                       axis_name="t_ps", axis_unit="ps",
                                       value_unit="Å^2"),
            lambda: Provenance(source_path="m", file_format="f",
                               frames_used=(0,),
                               cutoffs_ang={("Si", "O"): 2.0}),
            lambda: Provenance(source_path="m", file_format="f",
                               frames_used=(0,), formers="Si")):
        try:
            call()
        except ValueError as error:
            out.append(str(error))
        else:
            raise AssertionError("expected a refusal")
    return out


def test_no_note_or_message_carries_a_verdict():
    notes = _every_note_and_message()
    assert len(notes) >= 30
    assert [n for n in notes if VERDICT.search(n)] == []


def test_no_string_in_the_module_carries_a_verdict():
    """Every string literal in md_stats.py, docstrings included."""
    source = (ROOT / "facet" / "core" / "md_stats.py").read_text(
        encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 50
    assert [s for s in strings if VERDICT.search(s)] == []


def test_the_module_imports_without_qt():
    code = ("import sys, facet.core.md_stats;"
            "print('QT' if any(m.startswith(('PySide6', 'matplotlib')) "
            "for m in sys.modules) else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
