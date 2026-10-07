"""Frame-averaged results with their spread: the containers every MD analysis shares.

A crystal gives one number per site. An MD model gives one number per frame,
and what a glass paper reports is the mean over frames with the spread around
it: the Q^n fractions of a silicate, the fraction of four-coordinated boron,
an O-Si-O angle distribution, a partial g(r), a mean-square displacement. Every
analysis module of the MD path (glass descriptors, rings, scattering, NMR from
correlations, dynamics) produces one of four shapes, and this module holds them
so that the averaging, the spread, the bookkeeping of what was left out, and
the export rows are written once and behave the same everywhere:

* :class:`Distribution`: categorical, one value per key per frame (CN 4/5/6,
  Q^0..Q^4, 'F-Al1Na2'), as fractions or as counts;
* :class:`Histogram`: a continuous quantity binned on fixed edges per frame
  (angle_deg, phi, plateau_decades);
* :class:`Series`: a function of a continuous axis per frame (g(r) on r_ang,
  S(q) on q_inv_ang, MSD on t_ps);
* :class:`Scalar`: one number per frame (a mean CN, a density, an R-factor);

plus :class:`Provenance`, the header every export carries, and
:func:`check_fractions`, the one test every fraction set passes.

THE AVERAGE AND THE SPREAD, AND WHY THEY ARE COMPUTED THIS WAY
--------------------------------------------------------------
**Every frame weighs the same.** The mean is the plain mean of the per-frame
values, so a fraction is averaged as a fraction, not pooled from the counts.
For atoms of a fixed model (every CN, Q^n, N4 or speciation fraction) the
number of items is the same in every frame and the two agree; ``n_items`` is
kept per frame so that a pooled value can be formed from the export when it is
wanted.

**The spread is the sample standard deviation across frames, ddof = 1**
(:data:`STD_DDOF`). It describes how much the frames differ. It is not the
standard error of the mean: frames of one trajectory are correlated in time,
and dividing by sqrt(n_frames) would treat them as independent [1]. The
standard error of correlated data needs block averaging [2], which is not done
here. With one frame the spread is undefined and is NaN, with a note saying so;
it is never reported as 0.

**Frame order never changes a result, to the last bit.** Each column is
summed with ``math.fsum`` (:func:`mean_and_spread`), which keeps the partial
sums exactly and rounds once at the end, so the sum depends on the values
only: permuting the frames gives bit-identical means and spreads, where a
floating-point sum taken in frame order differs in its last bits when the
order changes (``tests/test_md_stats.py`` shows both). The same holds across
array shapes: numpy reduces a 1-D array pairwise and a 2-D column in
sequence, and on 37 values the two had already differed in the last bit
(measured), so a :class:`Scalar` and a :class:`Distribution` column holding
the same numbers would not have agreed. The variance is computed in two
passes, mean first and then the squared deviations from it, the algorithm
Chan, Golub and LeVeque recommend over the one-pass sum of squares, which
loses digits by cancellation [3]. A column whose values are all equal has its
mean set to that value and its spread to exactly 0, so two identical frames
give std = 0, not a rounding residue. When frame labels are given, the rows
are also stored in label order, so the whole container, not only its mean, is
the same whatever order the frames arrived in.

NOTHING IS DROPPED WITHOUT BEING COUNTED
----------------------------------------
* A key absent from a frame's counts is a count of 0 in that frame, never a
  missing column, so a Q^4 fraction averaged over frames where some frames had
  no Q^4 is not inflated.
* A key found in the counts but not among the keys given is appended after
  them, with a note; it is not dropped (a Q^5 can appear at a low threshold).
* A fraction needs at least one item. A frame that counted none has undefined
  fractions: its row is NaN, it is listed in ``frames_empty``, the mean and
  spread are over the other frames, and a note says so. If no frame counted
  anything, the result is refused.
* A histogram sample outside the edges is counted in ``n_below`` or
  ``n_above``, and a NaN or infinite sample (phi of an atom with no bond is NaN
  by design) in ``n_nonfinite``; each is noted. ``numpy.histogram`` itself
  ignores values outside its range without a word [4], which is the reason this
  module bins the samples itself.
* NaN values of a :class:`Series` or :class:`Scalar` propagate into the mean
  at that point, with a note counting them. An infinite value is refused.

FRACTIONS SUM TO 1, OR THE RESULT IS REFUSED
--------------------------------------------
:func:`check_fractions` raises ValueError, naming the sum and the tolerance,
when a fraction set does not sum to 1 within ``tol`` (default
:data:`FRACTION_TOL`, 1e-12) or holds a value outside [0, 1]. The sum is taken
with ``math.fsum``, which is exact before its final rounding, so a set formed
as count / total departs from 1 only by the roundings of the divisions
(about 1e-16 per key). It is a raise, not an ``assert``: facet/core holds no
assert statement, and asserts vanish under ``python -O``. Every fraction row of
a :class:`Distribution` and its mean pass through it.

NOT IN THIS MODULE
------------------
The containers measure nothing. They hold what another module measured, and
they carry no physical parameter, so they have no default for one. The
method parameters they do take (the histogram edges, whether a histogram is a
density) are arguments, stated in the result.

TIMINGS
-------
Measured 2026-10-06 on this machine (Windows 11, i5-13420H, Python 3.11.9,
numpy 2.4.6), unpinned, on a machine shared with other sessions where single
runs vary by about 30 %; the median of 7 runs after one untimed run, and the
range over two such sessions:

* :func:`mean_and_spread` over 100 frames x 1 000 points (a g(r) to 10 Å at
  0.01 Å): 8.8 ms; over 1 000 x 1 000: 108-129 ms; over 100 x 10 000:
  158-205 ms. ``math.fsum`` and the conversion to Python floats are nearly
  all of it, about 0.1-0.2 µs per value; a frame of the bulk engine takes
  0.5 s, so averaging is never the slow step.
* :meth:`Distribution.from_counts` from 100 per-frame dicts of 6 keys, as
  fractions: 0.9-1.5 ms (37 ms before the per-row fraction check and the
  per-count validation were vectorised).
* :meth:`Histogram.from_samples`, 100 frames x 10 000 samples on 180 bins:
  146-192 ms, 1.5-1.9 ms a frame, of which ``numpy.searchsorted`` takes about
  0.8 ms (profiled).
* :meth:`Series.from_frames`, 100 frames x 1 000 points: 23-34 ms.

REFERENCES
----------
[1] M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
    Oxford University Press, Oxford (2017).
[2] H. Flyvbjerg and H. G. Petersen, "Error estimates on averages of
    correlated data", *The Journal of Chemical Physics* 91 (1989) 461-466,
    https://doi.org/10.1063/1.457480
[3] T. F. Chan, G. H. Golub and R. J. LeVeque, "Algorithms for computing the
    sample variance: analysis and recommendations", *The American
    Statistician* 37 (1983) 242-247,
    https://doi.org/10.1080/00031305.1983.10483115
[4] NumPy documentation, ``numpy.histogram``: "All but the last
    (righthand-most) bin is half-open ... Values outside the range are
    ignored." https://numpy.org/doc/stable/reference/generated/numpy.histogram.html;
    C. R. Harris et al., "Array programming with NumPy", *Nature* 585 (2020)
    357-362, https://doi.org/10.1038/s41586-020-2649-2
"""
from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..version import __version__ as FACET_VERSION
from .md_model import NO_TIMESTEP, ModelOxidation, validate_symbol

__all__ = [
    "FRACTION_TOL", "STD_DDOF", "KINDS", "ROW_KINDS", "SPREAD_NOTE",
    "check_fractions", "mean_and_spread",
    "Distribution", "Histogram", "Series", "Scalar", "Provenance",
]

# ---------------------------------------------------------------------------
# constants: each one is a choice, and says so
# ---------------------------------------------------------------------------

# How far a fraction set may sum from 1. A numerical tolerance, not a physical
# one: fractions formed as count / total and summed with math.fsum depart from
# 1 by about 1e-16 per key, so 1e-12 leaves room for thousands of keys and
# still catches a single missing atom in a model of 10^12.
FRACTION_TOL: float = 1e-12

# The spread across frames is the sample standard deviation (ddof = 1). The
# reading taken in Step 0 (reply section 5, item 7): a spread, not a standard
# error, because the frames of one trajectory are correlated.
STD_DDOF: int = 1

KINDS = ("fraction", "count")

# What one row of a container is. Only the wording of notes and export columns
# changes with it: a mean-square displacement is averaged over time origins or
# blocks, not over frames.
ROW_KINDS = ("frame", "block", "time origin")

SPREAD_NOTE = ("std is the spread of the per-frame values across the frames "
               "(sample standard deviation, ddof = 1), not the standard error "
               "of the mean: frames of one trajectory are correlated in time")


# ---------------------------------------------------------------------------
# small checks shared by every container
# ---------------------------------------------------------------------------

def _plural(row_kind: str) -> str:
    return row_kind + "s"


def _how_many(count: int, row_kind: str) -> str:
    """'1 frame', '3 frames'."""
    return f"{count} {row_kind if count == 1 else _plural(row_kind)}"


def _name(value, what: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} needs a non-empty text, not {value!r}")
    return value


def _row_kind(value) -> str:
    if value not in ROW_KINDS:
        raise ValueError(f"row_kind {value!r} is not one of {ROW_KINDS}")
    return value


def _tolerance(tol) -> float:
    if isinstance(tol, (bool, np.bool_)):
        raise ValueError(f"tol {tol!r} is not a number")
    try:
        out = float(tol)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"tol {tol!r} is not a number") from None
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"tol is {tol!r}; a finite number of 0 or more is "
                         "needed")
    return out


def _notes(notes, what: str) -> tuple[str, ...]:
    if isinstance(notes, str):
        raise ValueError(f"{what}: notes needs a sequence of sentences, not "
                         "one string")
    out = []
    for note in notes or ():
        if not isinstance(note, str):
            raise ValueError(f"{what}: a note must be text, not "
                             f"{type(note).__name__}")
        out.append(note)
    return tuple(out)


def _merged(notes: Sequence[str], *extra: str) -> tuple[str, ...]:
    """notes followed by every extra note not already among them."""
    out = list(notes)
    for note in extra:
        if note and note not in out:
            out.append(note)
    return tuple(out)


def _frozen(array: np.ndarray) -> np.ndarray:
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def _compact(values: Iterable[int]) -> str:
    """'0-9, 20-40 step 5, 47': runs of three or more with one step joined."""
    ordered = sorted(int(v) for v in values)
    parts: list[str] = []
    i = 0
    while i < len(ordered):
        if i + 2 < len(ordered):
            step = ordered[i + 1] - ordered[i]
            j = i + 1
            while j + 1 < len(ordered) and ordered[j + 1] - ordered[j] == step:
                j += 1
            if step > 0 and j - i >= 2:
                parts.append(f"{ordered[i]}-{ordered[j]}"
                             + ("" if step == 1 else f" step {step}"))
                i = j + 1
                continue
        parts.append(str(ordered[i]))
        i += 1
    return ", ".join(parts)


def _frame_labels(frames, n_rows: int, what: str) -> np.ndarray:
    """(n_rows,) int64 labels; 0 .. n_rows-1 when none are given."""
    if frames is None:
        return np.arange(n_rows, dtype=np.int64)
    if isinstance(frames, (str, bytes)):
        raise ValueError(f"{what}: frames needs one integer label per row")
    labels = list(frames.tolist() if isinstance(frames, np.ndarray) else frames)
    if len(labels) != n_rows:
        raise ValueError(f"{what}: {len(labels)} frame labels for {n_rows} "
                         "rows; one label per row is needed")
    for label in labels:
        if isinstance(label, (bool, np.bool_)) or \
                not isinstance(label, (int, np.integer)) or label < 0:
            raise ValueError(f"{what}: frame label {label} is not a whole "
                             "number of 0 or more")
    out = np.array([int(v) for v in labels], dtype=np.int64)
    unique, counts = np.unique(out, return_counts=True)
    if (counts > 1).any():
        raise ValueError(f"{what}: frame label(s) "
                         f"{unique[counts > 1].tolist()} occur more than once")
    return out


def _float_rows(values, what: str, ndim: int) -> np.ndarray:
    try:
        out = np.array(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what}: the values are not numbers (a ragged list "
                         "of rows is refused; every row needs the same "
                         "length)") from None
    if out.ndim != ndim:
        raise ValueError(f"{what}: {ndim}-dimensional values are needed "
                         f"(rows first), not shape {out.shape}")
    if out.shape[0] == 0:
        raise ValueError(f"{what}: no rows; there is nothing to average")
    return out


def _whole_numbers(values, what: str) -> np.ndarray:
    """int64 copy of values that are finite, whole and not negative."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what}: the counts are not numbers") from None
    if raw.dtype.kind == "b" or raw.dtype.kind not in "iuf":
        raise ValueError(f"{what}: counts are whole numbers, not "
                         f"{raw.dtype} values")
    numbers = raw.astype(np.float64)
    if not np.isfinite(numbers).all():
        raise ValueError(f"{what}: a count is NaN or infinite")
    if (numbers < 0).any():
        raise ValueError(f"{what}: a count is negative "
                         f"({float(numbers[numbers < 0][0])!r})")
    if (numbers != np.floor(numbers)).any():
        first = numbers[numbers != np.floor(numbers)][0]
        raise ValueError(f"{what}: a count is not a whole number ({float(first)!r})")
    if (numbers > 2.0 ** 53).any():
        raise ValueError(f"{what}: a count exceeds 2**53, beyond which a float "
                         "no longer holds every whole number")
    return raw.astype(np.int64) if raw.dtype.kind in "iu" else \
        numbers.astype(np.int64)


def _key(value, what: str):
    """A hashable category key, numpy scalars turned into Python ones."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{what}: key {float(value)!r} is not finite")
        return float(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, tuple):
        return tuple(_key(v, what) for v in value)
    try:
        hash(value)
    except TypeError:
        raise ValueError(f"{what}: key {value!r} cannot label a category "
                         "(it is not hashable)") from None
    return value


def _key_rank(key):
    """A sort key that orders numbers numerically, then text, then tuples."""
    if isinstance(key, (bool, int, float)):
        return (0, key)
    if isinstance(key, str):
        return (1, key)
    if isinstance(key, tuple):
        return (2, tuple(_key_rank(k) for k in key))
    return (3, repr(key))


def _key_label(key) -> str:
    if isinstance(key, tuple):
        return "-".join(_key_label(k) for k in key)
    return str(key)


# ---------------------------------------------------------------------------
# the two primitives
# ---------------------------------------------------------------------------

def check_fractions(values, what: str, tol: float = FRACTION_TOL) -> None:
    """Refuse a fraction set that does not sum to 1, or holds a value outside [0, 1].

    ``values`` is one set: a 1-D sequence or array, or a mapping whose values
    are the fractions. ValueError, naming ``what``, the sum (``math.fsum``) and
    the tolerance, when ``|sum - 1| > tol``, when a value lies below ``-tol`` or
    above ``1 + tol``, when a value is NaN or infinite, or when the set is
    empty. Returns None when the set passes.
    """
    what = _name(what, "what")
    tol = _tolerance(tol)
    items = list(values.values()) if isinstance(values, Mapping) else values
    if isinstance(items, (str, bytes)):
        raise ValueError(f"{what}: the fractions are not numbers")
    try:
        array = np.asarray(items, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what}: the fractions are not numbers") from None
    if array.ndim != 1:
        raise ValueError(f"{what}: one set of fractions (1-D) is needed, not "
                         f"shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{what}: the fraction set is empty; it sums to 0, "
                         f"not 1 (tolerance {tol:g})")
    if not np.isfinite(array).all():
        raise ValueError(f"{what}: {int((~np.isfinite(array)).sum())} of "
                         f"{array.size} fractions are NaN or infinite, so the "
                         "set has no sum")
    total = math.fsum(array.tolist())
    tail = (f"the fractions sum to {total!r} (|sum - 1| = "
            f"{abs(total - 1.0):.3g}; tolerance {tol:g})")
    if array.min() < -tol:
        raise ValueError(f"{what}: a fraction is {float(array.min())!r}, below 0; "
                         + tail)
    if array.max() > 1.0 + tol:
        raise ValueError(f"{what}: a fraction is {float(array.max())!r}, above 1; "
                         + tail)
    if abs(total - 1.0) > tol:
        raise ValueError(f"{what}: {tail}")


def mean_and_spread(per_frame) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample standard deviation (ddof = 1) over the first axis.

    ``per_frame`` is (n_frames,) or (n_frames, ...). Each column is summed
    with ``math.fsum``, so the result is bit-identical under any permutation
    of the frames and for any shape; the variance is two-pass [3]. A column
    whose values are all
    equal gets that value as its mean and exactly 0 as its spread. One frame
    gives a NaN spread. A NaN in a column makes that column's mean and spread
    NaN. An infinite value, or no frame at all, raises ValueError.
    """
    try:
        values = np.asarray(per_frame, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("the per-frame values are not numbers (a ragged list "
                         "of rows is refused)") from None
    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError("no frame to average")
    if np.isinf(values).any():
        raise ValueError("a per-frame value is infinite; there is no mean")
    n_rows = values.shape[0]
    columns = values.reshape(n_rows, -1)
    with np.errstate(invalid="ignore"):
        low = columns.min(axis=0)           # NaN wherever a column holds one
        high = columns.max(axis=0)
        constant = low == high
    mean = _exact_column_sums(columns) / n_rows
    mean = np.minimum(np.maximum(mean, low), high)
    # low + 0.0 turns -0.0 into 0.0, so a column of 0.0 and -0.0 gives 0.0
    # whichever comes first.
    mean = np.where(constant, low + 0.0, mean)
    if n_rows == 1:
        spread = np.full(mean.shape, np.nan)
    else:
        deviation = columns - mean
        spread = np.sqrt(_exact_column_sums(deviation * deviation)
                         / (n_rows - STD_DDOF))
        spread = np.where(constant, 0.0, spread)
    shape = values.shape[1:]
    return (np.asarray(mean, dtype=np.float64).reshape(shape),
            np.asarray(spread, dtype=np.float64).reshape(shape))


def _exact_column_sums(columns: np.ndarray) -> np.ndarray:
    """math.fsum down each column: exact before one final rounding, so the
    result depends on the values only, not on their order or the array shape."""
    return np.array([math.fsum(column) for column in columns.T.tolist()],
                    dtype=np.float64)


def _one_row_note(row_kind: str, n_used: int) -> str:
    if n_used != 1:
        return ""
    return (f"one {row_kind}: the spread across {_plural(row_kind)} (std, "
            f"ddof = {STD_DDOF}) is undefined and given as NaN")


def _sorted_rows(labels: np.ndarray, *arrays: np.ndarray):
    order = np.argsort(labels, kind="stable")
    return (labels[order],) + tuple(a[order] for a in arrays)


# ---------------------------------------------------------------------------
# categorical: Distribution
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Distribution:
    """A categorical descriptor per frame and over frames.

    ``per_frame`` is (n_frames, n_keys): fractions (``kind='fraction'``) or
    counts (``kind='count'``) of each key in each frame. ``n_items`` (n_frames,)
    is how many atoms, bonds or units were counted in each frame; for counts it
    is the row sum and may be omitted, for fractions it is required. ``frames``
    labels the rows (trajectory indices; 0 .. n-1 when omitted) and the rows
    are stored in label order.

    Computed on construction, from the rows that counted something:
    ``mean`` and ``std`` (n_keys,), ``n_frames`` (all rows), ``frames_empty``
    (labels of fraction rows that counted nothing, NaN and left out of the mean)
    and the notes. Every fraction row with items, and the mean, pass
    :func:`check_fractions`. Build one with :meth:`from_counts` from per-frame
    dicts, or directly from a table.
    """

    name: str
    keys: tuple
    kind: str
    per_frame: np.ndarray
    n_items: np.ndarray | None = None
    frames: np.ndarray | Sequence[int] | None = None
    row_kind: str = "frame"
    notes: tuple[str, ...] = ()
    mean: np.ndarray = field(init=False, repr=False)
    std: np.ndarray = field(init=False, repr=False)
    n_frames: int = field(init=False)
    frames_empty: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        name = _name(self.name)
        if self.kind not in KINDS:
            raise ValueError(f"{name}: kind {self.kind!r} is not one of {KINDS}")
        row_kind = _row_kind(self.row_kind)
        notes = _notes(self.notes, name)
        if isinstance(self.keys, (str, bytes)) or self.keys is None:
            raise ValueError(f"{name}: keys needs a sequence of category keys")
        keys = tuple(_key(k, name) for k in self.keys)
        if not keys:
            raise ValueError(f"{name}: no keys; a distribution needs at least "
                             "one category")
        if len(set(keys)) != len(keys):
            raise ValueError(f"{name}: a key occurs more than once in {keys}")
        values = _float_rows(self.per_frame, name, 2)
        n_rows = values.shape[0]
        if values.shape[1] != len(keys):
            raise ValueError(f"{name}: {values.shape[1]} columns for "
                             f"{len(keys)} keys")
        if np.isinf(values).any():
            raise ValueError(f"{name}: a per-frame value is infinite")
        labels = _frame_labels(self.frames, n_rows, name)

        if self.kind == "count":
            counts = _whole_numbers(values, name)
            sums = counts.sum(axis=1)
            if self.n_items is None:
                n_items = sums
            else:
                n_items = _whole_numbers(self.n_items, f"{name}, n_items")
                if n_items.shape != (n_rows,) or not np.array_equal(n_items,
                                                                    sums):
                    raise ValueError(f"{name}: n_items has to be the row sums "
                                     f"of the counts, {sums.tolist()}")
            used = np.ones(n_rows, dtype=bool)
        else:
            if self.n_items is None:
                raise ValueError(f"{name}: fractions need n_items, the number "
                                 "of items counted in each frame")
            n_items = _whole_numbers(self.n_items, f"{name}, n_items")
            if n_items.shape != (n_rows,):
                raise ValueError(f"{name}: n_items needs {n_rows} values, one "
                                 "per row")
            used = n_items > 0
            if not used.any():
                raise ValueError(f"{name}: no {row_kind} counted any item, so "
                                 "no fraction is defined")
            if not np.isnan(values[~used]).all():
                raise ValueError(f"{name}: a {row_kind} with no item has "
                                 "fractions that are not NaN")
            # check_fractions on every row, vectorised: the same fsum and the
            # same bounds; the row-by-row call only runs to word the refusal.
            rows = values[used]
            with np.errstate(invalid="ignore"):
                failed = (not np.isfinite(rows).all()
                          or rows.min() < -FRACTION_TOL
                          or rows.max() > 1.0 + FRACTION_TOL
                          or (np.abs(_exact_column_sums(rows.T) - 1.0)
                              > FRACTION_TOL).any())
            if failed:
                for label, row in zip(labels[used], rows):
                    check_fractions(row, f"{name}, {row_kind} {int(label)}")

        labels, values, n_items, used = _sorted_rows(labels, values, n_items,
                                                     used)
        mean, spread = mean_and_spread(values[used])
        if self.kind == "fraction":
            check_fractions(mean, f"{name}, mean over {_plural(row_kind)}")
        empty = tuple(int(v) for v in labels[~used])
        extra = [_one_row_note(row_kind, int(used.sum()))]
        if empty:
            extra.append(
                f"{name}: {len(empty)} of {n_rows} {_plural(row_kind)} counted "
                f"no item ({_plural(row_kind)} {_compact(empty)}); their "
                "fractions are undefined (NaN), and the mean and spread are "
                f"over the other {n_rows - len(empty)}")
        put = object.__setattr__
        put(self, "name", name)
        put(self, "keys", keys)
        put(self, "row_kind", row_kind)
        put(self, "per_frame", _frozen(values))
        put(self, "n_items", _frozen(n_items))
        put(self, "frames", _frozen(labels))
        put(self, "mean", _frozen(mean))
        put(self, "std", _frozen(spread))
        put(self, "n_frames", n_rows)
        put(self, "frames_empty", empty)
        put(self, "notes", _merged(notes, *extra))

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_counts(cls, per_frame_counts: Sequence[Mapping], *, name: str,
                    kind: str, keys: Sequence | None = None,
                    frames: Sequence[int] | None = None,
                    row_kind: str = "frame",
                    notes: Sequence[str] = ()) -> "Distribution":
        """One mapping key -> count per frame, as fractions or counts.

        A key absent from a frame's mapping counts 0 there. ``keys`` fixes the
        order and lets a category with no occurrence still show (Q^0..Q^4);
        a key found in the counts but not in ``keys`` is appended after them,
        with a note. Without ``keys`` the order is numbers, then text, then
        tuples, each ascending. Counts are whole numbers of 0 or more.
        """
        name = _name(name)
        if kind not in KINDS:
            raise ValueError(f"{name}: kind {kind!r} is not one of {KINDS}")
        if isinstance(per_frame_counts, Mapping):
            raise ValueError(f"{name}: one mapping per frame is needed (a "
                             "sequence of mappings), not a single mapping")
        rows = []
        for index, row in enumerate(per_frame_counts):
            if not isinstance(row, Mapping):
                raise ValueError(f"{name}: row {index} is "
                                 f"{type(row).__name__}, not a mapping of key "
                                 "to count")
            rows.append({_key(k, name): v for k, v in row.items()})
        if not rows:
            raise ValueError(f"{name}: no rows; there is nothing to average")
        seen = set().union(*(row.keys() for row in rows))
        extra_notes = []
        if keys is None:
            order = sorted(seen, key=_key_rank)
        else:
            if isinstance(keys, (str, bytes)):
                raise ValueError(f"{name}: keys needs a sequence of category "
                                 "keys, not one string")
            order = [_key(k, name) for k in keys]
            if len(set(order)) != len(order):
                raise ValueError(f"{name}: a key occurs more than once in "
                                 f"{order}")
            unlisted = sorted(seen - set(order), key=_key_rank)
            if unlisted:
                extra_notes.append(
                    f"{name}: key(s) {', '.join(_key_label(k) for k in unlisted)}"
                    " occur in the counts but not among the keys given; "
                    "appended after them")
                order += unlisted
        if not order:
            raise ValueError(f"{name}: every row is empty and no keys were "
                             "given, so there is no category to report")
        column = {k: c for c, k in enumerate(order)}
        counts = np.zeros((len(rows), len(order)), dtype=np.int64)
        for index, row in enumerate(rows):
            for key, value in row.items():
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"{name}, row {index}: the count of "
                                     f"{_key_label(key)} is {value!r}, not a "
                                     "number")
                if isinstance(value, (int, np.integer)) and \
                        0 <= value <= 2 ** 53:
                    counts[index, column[key]] = value
                else:   # checked and worded there, or a whole-valued float
                    counts[index, column[key]] = _whole_numbers(
                        value, f"{name}, row {index}, key {_key_label(key)}")
        result = cls(name, tuple(order), "count", counts, frames=frames,
                     row_kind=row_kind,
                     notes=_merged(_notes(notes, name), *extra_notes))
        return result if kind == "count" else result.to_fractions()

    def to_fractions(self) -> "Distribution":
        """The same counts as fractions of each frame's items."""
        if self.kind != "count":
            raise ValueError(f"{self.name}: to_fractions needs counts; this "
                             "distribution already holds fractions")
        with np.errstate(invalid="ignore", divide="ignore"):
            fractions = self.per_frame / self.n_items[:, None]
        fractions[self.n_items == 0] = np.nan
        return Distribution(self.name, self.keys, "fraction", fractions,
                            n_items=self.n_items, frames=self.frames,
                            row_kind=self.row_kind, notes=self.notes)

    def with_notes(self, *extra: str) -> "Distribution":
        """A copy with sentences added to its notes."""
        return dataclasses.replace(self, notes=_merged(self.notes, *_notes(
            extra, self.name)))

    # -- reading ------------------------------------------------------------
    @property
    def n_frames_used(self) -> int:
        """Rows in the mean: every row, less the fraction rows with no item."""
        return self.n_frames - len(self.frames_empty)

    def as_dict(self) -> dict:
        """key -> (mean, std)."""
        return {k: (float(m), float(s))
                for k, m, s in zip(self.keys, self.mean, self.std)}

    def as_rows(self, *, per_frame: bool = False) -> list[dict]:
        """One row per key, for ``exporters.write_csv`` / ``write_xlsx``.

        Columns: descriptor, key, kind, mean, std, the number of rows in the
        mean, the items per row (one value, or min-max when it varies), and,
        with ``per_frame``, one column per row label.
        """
        used = self.n_items[np.isin(self.frames, self.frames_empty,
                                    invert=True)]
        items = (str(int(used.min())) if used.min() == used.max()
                 else f"{int(used.min())}-{int(used.max())}")
        rows = []
        for c, key in enumerate(self.keys):
            row = {"descriptor": self.name, "key": _key_label(key),
                   "kind": self.kind, "mean": float(self.mean[c]),
                   "std": float(self.std[c]),
                   f"{_plural(self.row_kind)} used": self.n_frames_used,
                   f"items per {self.row_kind}": items}
            if per_frame:
                for r, label in enumerate(self.frames):
                    row[f"{self.row_kind} {int(label)}"] = \
                        float(self.per_frame[r, c])
            rows.append(row)
        return rows

    def __repr__(self) -> str:
        return (f"<Distribution {self.name!r}: {self.kind}, "
                f"{len(self.keys)} keys, {self.n_frames} "
                f"{_plural(self.row_kind)}>")


# ---------------------------------------------------------------------------
# continuous, binned: Histogram
# ---------------------------------------------------------------------------

def _edges(edges, what: str) -> np.ndarray:
    try:
        out = np.array(edges, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what}: the bin edges are not numbers") from None
    if out.ndim != 1 or out.size < 2:
        raise ValueError(f"{what}: at least two bin edges (1-D) are needed")
    if not np.isfinite(out).all():
        raise ValueError(f"{what}: a bin edge is NaN or infinite")
    if not (np.diff(out) > 0).all():
        raise ValueError(f"{what}: the bin edges have to increase strictly")
    return out


def _bin(samples: np.ndarray, edges: np.ndarray):
    """(counts, n_below, n_above, n_nonfinite) of one frame's samples.

    numpy.histogram's convention [4]: [e_k, e_k+1) for every bin but the last,
    which is closed, [e_n-1, e_n].
    """
    finite = np.isfinite(samples)
    values = samples[finite]
    below = values < edges[0]
    above = values > edges[-1]
    inside = values[~(below | above)]
    n_bins = edges.size - 1
    index = np.searchsorted(edges, inside, side="right") - 1
    index[index == n_bins] = n_bins - 1          # a value on the last edge
    counts = np.bincount(index, minlength=n_bins).astype(np.int64)
    return (counts, int(below.sum()), int(above.sum()),
            int(samples.size - values.size))


@dataclass(frozen=True, eq=False)
class Histogram:
    """A continuous descriptor binned on fixed edges, per frame and averaged.

    ``counts`` (n_frames, n_bins) holds the samples in each bin, with the
    convention of ``numpy.histogram``: every bin [low, high) but the last,
    which is [low, high]. Samples below the first edge, above the last, or not
    finite are counted per frame in ``n_below``, ``n_above`` and
    ``n_nonfinite``, never dropped. ``unit`` is the unit of the samples
    ('deg', 'v.u.', '1').

    ``density=False``: ``per_frame`` is the counts. ``density=True``: it is
    counts / (n_finite * bin width), with n_finite every finite sample of the
    frame, in range or not, so it integrates over the edges to the fraction of
    the frame's finite samples inside them (1 when none falls outside); a frame
    with no finite sample is NaN, listed in ``frames_empty`` and left out of the
    mean. ``value_unit`` says which ('count' or '1/<unit>'). ``mean`` and ``std``
    are over frames (:func:`mean_and_spread`).
    """

    name: str
    unit: str
    edges: np.ndarray
    counts: np.ndarray
    n_below: np.ndarray
    n_above: np.ndarray
    n_nonfinite: np.ndarray
    density: bool = False
    frames: np.ndarray | Sequence[int] | None = None
    row_kind: str = "frame"
    notes: tuple[str, ...] = ()
    per_frame: np.ndarray = field(init=False, repr=False)
    mean: np.ndarray = field(init=False, repr=False)
    std: np.ndarray = field(init=False, repr=False)
    n_samples: np.ndarray = field(init=False, repr=False)
    n_frames: int = field(init=False)
    frames_empty: tuple[int, ...] = field(init=False)
    value_unit: str = field(init=False)

    def __post_init__(self) -> None:
        name = _name(self.name)
        if not isinstance(self.unit, str):
            raise ValueError(f"{name}: unit needs a text such as 'deg' or "
                             "'1' (dimensionless)")
        if not isinstance(self.density, (bool, np.bool_)):
            raise ValueError(f"{name}: density is True or False, not "
                             f"{self.density!r}")
        density = bool(self.density)
        row_kind = _row_kind(self.row_kind)
        notes = _notes(self.notes, name)
        edges = _edges(self.edges, name)
        counts = _whole_numbers(self.counts, f"{name}, counts")
        if counts.ndim != 2 or counts.shape[1] != edges.size - 1 \
                or counts.shape[0] == 0:
            raise ValueError(f"{name}: counts needs shape (n_frames, "
                             f"{edges.size - 1}), not {counts.shape}")
        n_rows = counts.shape[0]
        tallies = []
        for label in ("n_below", "n_above", "n_nonfinite"):
            tally = _whole_numbers(getattr(self, label), f"{name}, {label}")
            if tally.shape != (n_rows,):
                raise ValueError(f"{name}: {label} needs {n_rows} values, one "
                                 "per row")
            tallies.append(tally)
        below, above, nonfinite = tallies
        labels = _frame_labels(self.frames, n_rows, name)
        labels, counts, below, above, nonfinite = _sorted_rows(
            labels, counts, below, above, nonfinite)

        in_range = counts.sum(axis=1)
        finite = in_range + below + above
        n_samples = finite + nonfinite
        widths = np.diff(edges)
        if density:
            used = finite > 0
            if not used.any():
                raise ValueError(f"{name}: no {row_kind} holds a finite "
                                 "sample, so no density is defined")
            values = np.full(counts.shape, np.nan)
            values[used] = counts[used] / (finite[used, None] * widths)
            value_unit = "1" if self.unit in ("", "1") else f"1/{self.unit}"
        else:
            used = np.ones(n_rows, dtype=bool)
            values = counts.astype(np.float64)
            value_unit = "count"
        mean, spread = mean_and_spread(values[used])

        plural = _plural(row_kind)
        extra = [_one_row_note(row_kind, int(used.sum()))]
        if below.sum() or above.sum():
            extra.append(
                f"{name}: {int(below.sum())} sample(s) below the first edge "
                f"({float(edges[0])!r} {self.unit}) and {int(above.sum())} above the "
                f"last edge ({float(edges[-1])!r} {self.unit}), out of "
                f"{int(n_samples.sum())} over {_how_many(n_rows, row_kind)}; "
                "counted in "
                "n_below and n_above, in no bin")
        if nonfinite.sum():
            extra.append(
                f"{name}: {int(nonfinite.sum())} sample(s) are NaN or infinite "
                "(an undefined value, such as phi of an atom with no bond); "
                "counted in n_nonfinite, in no bin, and left out of any "
                "normalisation")
        if density and (below.sum() or above.sum()):
            extra.append(
                f"{name}: the density is per finite sample, so it integrates "
                "over the edges to the fraction of finite samples inside them, "
                f"{float(in_range.sum() / finite.sum())!r} over all {plural}, not to 1")
        empty = tuple(int(v) for v in labels[~used])
        if empty:
            extra.append(
                f"{name}: {len(empty)} of {n_rows} {plural} hold no finite "
                f"sample ({plural} {_compact(empty)}); their densities are "
                "undefined (NaN), and the mean and spread are over the other "
                f"{n_rows - len(empty)}")
        put = object.__setattr__
        put(self, "name", name)
        put(self, "density", density)
        put(self, "row_kind", row_kind)
        put(self, "edges", _frozen(edges))
        put(self, "counts", _frozen(counts))
        put(self, "n_below", _frozen(below))
        put(self, "n_above", _frozen(above))
        put(self, "n_nonfinite", _frozen(nonfinite))
        put(self, "frames", _frozen(labels))
        put(self, "per_frame", _frozen(values))
        put(self, "mean", _frozen(mean))
        put(self, "std", _frozen(spread))
        put(self, "n_samples", _frozen(n_samples))
        put(self, "n_frames", n_rows)
        put(self, "frames_empty", empty)
        put(self, "value_unit", value_unit)
        put(self, "notes", _merged(notes, *extra))

    @classmethod
    def from_samples(cls, per_frame_samples: Sequence, edges, *, name: str,
                     unit: str, density: bool = False,
                     frames: Sequence[int] | None = None,
                     row_kind: str = "frame",
                     notes: Sequence[str] = ()) -> "Histogram":
        """Bin one 1-D array of samples per frame on ``edges``.

        The frames may hold different numbers of samples. ``edges`` and
        ``density`` are method choices with no physical default; both are
        stored in the result.
        """
        name = _name(name)
        edges = _edges(edges, name)
        if isinstance(per_frame_samples, (str, bytes, Mapping)):
            raise ValueError(f"{name}: one array of samples per frame is "
                             "needed")
        binned = []
        for index, samples in enumerate(per_frame_samples):
            try:
                array = np.asarray(samples, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{name}, row {index}: the samples are not "
                                 "numbers") from None
            if array.ndim != 1:
                raise ValueError(f"{name}, row {index}: one 1-D array of "
                                 f"samples per row is needed, not shape "
                                 f"{array.shape}")
            binned.append(_bin(array, edges))
        if not binned:
            raise ValueError(f"{name}: no rows; there is nothing to average")
        counts = np.stack([b[0] for b in binned])
        return cls(name, unit, edges, counts,
                   n_below=np.array([b[1] for b in binned], dtype=np.int64),
                   n_above=np.array([b[2] for b in binned], dtype=np.int64),
                   n_nonfinite=np.array([b[3] for b in binned],
                                        dtype=np.int64),
                   density=density, frames=frames, row_kind=row_kind,
                   notes=notes)

    def with_notes(self, *extra: str) -> "Histogram":
        """A copy with sentences added to its notes."""
        return dataclasses.replace(self, notes=_merged(self.notes, *_notes(
            extra, self.name)))

    @property
    def n_bins(self) -> int:
        return self.edges.size - 1

    @property
    def centres(self) -> np.ndarray:
        """Bin midpoints, in ``unit``."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self) -> np.ndarray:
        """Bin widths, in ``unit``."""
        return np.diff(self.edges)

    @property
    def n_frames_used(self) -> int:
        return self.n_frames - len(self.frames_empty)

    def as_rows(self, *, per_frame: bool = False) -> list[dict]:
        """One row per bin: its edges, mean, std, and optionally every row."""
        unit = f" ({self.unit})" if self.unit else ""
        rows = []
        for b in range(self.n_bins):
            row = {"descriptor": self.name,
                   f"bin low{unit}": float(self.edges[b]),
                   f"bin high{unit}": float(self.edges[b + 1]),
                   f"mean ({self.value_unit})": float(self.mean[b]),
                   f"std ({self.value_unit})": float(self.std[b]),
                   f"{_plural(self.row_kind)} used": self.n_frames_used}
            if per_frame:
                for r, label in enumerate(self.frames):
                    row[f"{self.row_kind} {int(label)}"] = \
                        float(self.per_frame[r, b])
            rows.append(row)
        return rows

    def __repr__(self) -> str:
        return (f"<Histogram {self.name!r}: {self.n_bins} bins "
                f"[{self.edges[0]:g}, {self.edges[-1]:g}] {self.unit}, "
                f"{self.value_unit}, {self.n_frames} "
                f"{_plural(self.row_kind)}>")


# ---------------------------------------------------------------------------
# continuous axis: Series; one number: Scalar
# ---------------------------------------------------------------------------

def _nan_note(name: str, values: np.ndarray, row_kind: str,
              where: str) -> str:
    missing = np.isnan(values)
    if not missing.any():
        return ""
    rows = int(missing.reshape(missing.shape[0], -1).any(axis=1).sum())
    text = (f"{name}: {int(missing.sum())} value(s) are NaN, in "
            f"{_how_many(rows, row_kind)}; the mean and spread are NaN")
    if values.ndim == 2:
        text += f" at the {int(missing.any(axis=0).sum())} {where} they fall on"
    return text


@dataclass(frozen=True, eq=False)
class Series:
    """A function of a continuous axis, per frame and averaged over frames.

    ``axis`` (n_points,) is shared by every frame, finite and strictly
    increasing; ``axis_name`` carries its unit the way FACET names variables
    ('r_ang', 'q_inv_ang', 't_ps'), and ``axis_unit`` is the unit as text
    ('Å', '1/Å', 'ps'). ``per_frame`` is (n_frames, n_points) in
    ``value_unit``. ``mean`` and ``std`` are over frames
    (:func:`mean_and_spread`); a NaN value makes that point's mean NaN, with a
    note; an infinite value is refused.
    """

    name: str
    axis: np.ndarray
    axis_name: str
    axis_unit: str
    value_unit: str
    per_frame: np.ndarray
    frames: np.ndarray | Sequence[int] | None = None
    row_kind: str = "frame"
    notes: tuple[str, ...] = ()
    mean: np.ndarray = field(init=False, repr=False)
    std: np.ndarray = field(init=False, repr=False)
    n_frames: int = field(init=False)

    def __post_init__(self) -> None:
        name = _name(self.name)
        axis_name = _name(self.axis_name, f"{name}: axis_name")
        for label in ("axis_unit", "value_unit"):
            if not isinstance(getattr(self, label), str):
                raise ValueError(f"{name}: {label} needs a text ('1' when "
                                 "dimensionless)")
        row_kind = _row_kind(self.row_kind)
        notes = _notes(self.notes, name)
        try:
            axis = np.array(self.axis, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name}: the {axis_name} values are not "
                             "numbers") from None
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError(f"{name}: {axis_name} needs a 1-D array of at "
                             "least one value")
        if not np.isfinite(axis).all():
            raise ValueError(f"{name}: an {axis_name} value is NaN or infinite")
        if not (np.diff(axis) > 0).all():
            raise ValueError(f"{name}: the {axis_name} values have to "
                             "increase strictly")
        values = _float_rows(self.per_frame, name, 2)
        if values.shape[1] != axis.size:
            raise ValueError(f"{name}: {values.shape[1]} values per row for "
                             f"{axis.size} {axis_name} points")
        if np.isinf(values).any():
            raise ValueError(f"{name}: a value is infinite")
        labels = _frame_labels(self.frames, values.shape[0], name)
        labels, values = _sorted_rows(labels, values)
        mean, spread = mean_and_spread(values)
        extra = (_one_row_note(row_kind, values.shape[0]),
                 _nan_note(name, values, row_kind, f"{axis_name} points"))
        put = object.__setattr__
        put(self, "name", name)
        put(self, "axis_name", axis_name)
        put(self, "row_kind", row_kind)
        put(self, "axis", _frozen(axis))
        put(self, "per_frame", _frozen(values))
        put(self, "frames", _frozen(labels))
        put(self, "mean", _frozen(mean))
        put(self, "std", _frozen(spread))
        put(self, "n_frames", values.shape[0])
        put(self, "notes", _merged(notes, *extra))

    @classmethod
    def from_frames(cls, axis, per_frame_values: Sequence, *, name: str,
                    axis_name: str, axis_unit: str, value_unit: str,
                    frames: Sequence[int] | None = None,
                    row_kind: str = "frame",
                    notes: Sequence[str] = ()) -> "Series":
        """One 1-D array of values per frame, all on the same ``axis``."""
        name = _name(name)
        if isinstance(per_frame_values, (str, bytes, Mapping)):
            raise ValueError(f"{name}: one array of values per frame is needed")
        rows = []
        for index, values in enumerate(per_frame_values):
            try:
                row = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{name}, row {index}: the values are not "
                                 "numbers") from None
            if row.ndim != 1:
                raise ValueError(f"{name}, row {index}: a 1-D array is needed, "
                                 f"not shape {row.shape}")
            rows.append(row)
        if not rows:
            raise ValueError(f"{name}: no rows; there is nothing to average")
        lengths = {row.size for row in rows}
        if len(lengths) != 1:
            raise ValueError(f"{name}: the rows hold {sorted(lengths)} values; "
                             "every row needs one value per axis point")
        return cls(name, axis, axis_name, axis_unit, value_unit,
                   np.stack(rows), frames=frames, row_kind=row_kind,
                   notes=notes)

    def with_notes(self, *extra: str) -> "Series":
        """A copy with sentences added to its notes."""
        return dataclasses.replace(self, notes=_merged(self.notes, *_notes(
            extra, self.name)))

    def as_rows(self, *, per_frame: bool = False) -> list[dict]:
        """One row per axis point: the axis value, mean, std, optionally every row."""
        rows = []
        for p in range(self.axis.size):
            row = {"descriptor": self.name,
                   self.axis_name: float(self.axis[p]),
                   f"mean ({self.value_unit})": float(self.mean[p]),
                   f"std ({self.value_unit})": float(self.std[p]),
                   f"{_plural(self.row_kind)} used": self.n_frames}
            if per_frame:
                for r, label in enumerate(self.frames):
                    row[f"{self.row_kind} {int(label)}"] = \
                        float(self.per_frame[r, p])
            rows.append(row)
        return rows

    def __repr__(self) -> str:
        return (f"<Series {self.name!r}: {self.axis.size} points of "
                f"{self.axis_name}, {self.n_frames} "
                f"{_plural(self.row_kind)}>")


@dataclass(frozen=True, eq=False)
class Scalar:
    """One number per frame, averaged over frames (a mean CN, a density, an R-factor).

    ``per_frame`` is (n_frames,) in ``unit``. ``mean`` and ``std`` are floats
    (:func:`mean_and_spread`); a NaN value makes them NaN, with a note; an
    infinite value is refused.
    """

    name: str
    unit: str
    per_frame: np.ndarray
    frames: np.ndarray | Sequence[int] | None = None
    row_kind: str = "frame"
    notes: tuple[str, ...] = ()
    mean: float = field(init=False)
    std: float = field(init=False)
    n_frames: int = field(init=False)

    def __post_init__(self) -> None:
        name = _name(self.name)
        if not isinstance(self.unit, str):
            raise ValueError(f"{name}: unit needs a text ('1' when "
                             "dimensionless)")
        row_kind = _row_kind(self.row_kind)
        notes = _notes(self.notes, name)
        values = _float_rows(self.per_frame, name, 1)
        if np.isinf(values).any():
            raise ValueError(f"{name}: a value is infinite")
        labels = _frame_labels(self.frames, values.shape[0], name)
        labels, values = _sorted_rows(labels, values)
        mean, spread = mean_and_spread(values)
        extra = (_one_row_note(row_kind, values.shape[0]),
                 _nan_note(name, values, row_kind, ""))
        put = object.__setattr__
        put(self, "name", name)
        put(self, "row_kind", row_kind)
        put(self, "per_frame", _frozen(values))
        put(self, "frames", _frozen(labels))
        put(self, "mean", float(mean))
        put(self, "std", float(spread))
        put(self, "n_frames", values.shape[0])
        put(self, "notes", _merged(notes, *extra))

    def with_notes(self, *extra: str) -> "Scalar":
        """A copy with sentences added to its notes."""
        return dataclasses.replace(self, notes=_merged(self.notes, *_notes(
            extra, self.name)))

    def as_rows(self, *, per_frame: bool = False) -> list[dict]:
        """One row: name, unit, mean, std, optionally every row."""
        row = {"descriptor": self.name, "unit": self.unit, "mean": self.mean,
               "std": self.std,
               f"{_plural(self.row_kind)} used": self.n_frames}
        if per_frame:
            for r, label in enumerate(self.frames):
                row[f"{self.row_kind} {int(label)}"] = float(self.per_frame[r])
        return [row]

    def __repr__(self) -> str:
        return (f"<Scalar {self.name!r}: {self.mean!r} +/- {self.std!r} "
                f"{self.unit}, {self.n_frames} {_plural(self.row_kind)}>")


# ---------------------------------------------------------------------------
# provenance: the export header
# ---------------------------------------------------------------------------

def _now() -> str:
    return f"{datetime.now():%Y-%m-%d %H:%M}"


def _optional_float(value, what: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} {value!r} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what} {value!r} is not a number") from None
    if not math.isfinite(out):
        raise ValueError(f"{what} is {value!r}; a finite number is needed")
    return out


def _reasons(mapping, what: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for key, reason in (mapping or {}).items():
        if isinstance(key, (bool, np.bool_)) or \
                not isinstance(key, (int, np.integer)) or key < 0:
            raise ValueError(f"{what} is keyed by a whole number of 0 or more, "
                             f"not {key!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{what}: frame {key} needs a reason as text")
        out[int(key)] = reason
    return dict(sorted(out.items()))


_PARAMETER_TYPES = (type(None), bool, int, float, str)


def _parameter(value, key: str):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, _PARAMETER_TYPES):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_parameter(v, key) for v in value)
    raise ValueError(f"method parameter {key!r} is {type(value).__name__}; a "
                     "number, text, True/False, None or a tuple of them is "
                     "needed, so that the export header can state it")


def _format(value) -> str:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, tuple):
        return "(" + ", ".join(_format(v) for v in value) + ")"
    return str(value)


@dataclass(frozen=True)
class Provenance:
    """Where a set of frame-averaged results came from, as an export header.

    ``frames_used`` are readable-frame indices (what ``Trajectory.frame(k)``
    takes), stored ascending with ``timesteps`` / ``times_ps`` moved with them;
    ``frames_skipped`` maps an index that was chosen but not analysed to the
    reason; ``file_blocks_skipped`` is the reader's own ``Trajectory.skipped``,
    keyed by file position. ``formers`` is None when the analysis took none,
    and is reported as such; no former set is ever assumed. ``cutoffs_ang``
    maps an element pair to a distance cutoff, and ``cutoff_sources`` says for
    each where it came from (a g(r) minimum and its method, or the user).
    ``method_parameters`` holds every non-physical choice (bin widths,
    smoothing) by name. :meth:`as_rows` and :meth:`as_lines` give the header.
    """

    source_path: str
    file_format: str
    frames_used: tuple[int, ...]
    frames_skipped: Mapping[int, str] = field(default_factory=dict)
    file_blocks_skipped: Mapping[int, str] = field(default_factory=dict)
    timesteps: tuple[int, ...] = ()
    times_ps: tuple[float, ...] = ()
    type_map: Mapping[int | str, str] = field(default_factory=dict)
    type_map_source: str = ""
    ox: ModelOxidation | None = None
    params_name: str | None = None
    params_source: str | None = None
    v_bond_vu: float | None = None
    v_list_vu: float | None = None
    r_search_ang: float | None = None
    formers: frozenset[str] | None = None
    cutoffs_ang: Mapping[tuple[str, str], float] = field(default_factory=dict)
    cutoff_sources: Mapping[tuple[str, str], str] = field(default_factory=dict)
    method_parameters: Mapping[str, object] = field(default_factory=dict)
    estimated_pairs: Mapping[str, int] = field(default_factory=dict)
    missing_pairs: Mapping[str, int] = field(default_factory=dict)
    units_note: str = ""
    facet_version: str = FACET_VERSION
    created: str = field(default_factory=_now)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        put = object.__setattr__
        for label in ("source_path", "file_format"):
            value = getattr(self, label)
            put(self, label, _name(os.fspath(value) if isinstance(
                value, os.PathLike) else value, label))
        if isinstance(self.frames_used, (str, bytes)):
            raise ValueError("frames_used needs a sequence of frame indices")
        used = list(self.frames_used)
        if not used:
            raise ValueError("frames_used is empty: a result needs at least "
                             "one frame")
        labels = _frame_labels(used, len(used), "frames_used")
        order = np.argsort(labels, kind="stable")
        put(self, "frames_used", tuple(int(v) for v in labels[order]))
        for label, cast in (("timesteps", int), ("times_ps", float)):
            values = list(getattr(self, label))
            if values and len(values) != len(used):
                raise ValueError(f"{label} needs one value per frame used "
                                 f"({len(used)}), not {len(values)}")
            if cast is int and any(isinstance(v, (bool, np.bool_)) or
                                   not isinstance(v, (int, np.integer))
                                   for v in values):
                raise ValueError("timesteps are whole numbers (NO_TIMESTEP = "
                                 f"{NO_TIMESTEP} where the file gives none)")
            put(self, label, tuple(cast(values[i]) for i in order)
                if values else ())
        skipped = _reasons(self.frames_skipped, "frames_skipped")
        both = sorted(set(skipped) & set(self.frames_used))
        if both:
            raise ValueError(f"frame(s) {both} are both used and skipped")
        put(self, "frames_skipped", skipped)
        put(self, "file_blocks_skipped",
            _reasons(self.file_blocks_skipped, "file_blocks_skipped"))
        type_map = {}
        for key, symbol in dict(self.type_map).items():
            type_map[key] = validate_symbol(str(symbol))
        put(self, "type_map", type_map)
        if self.ox is not None and not isinstance(self.ox, ModelOxidation):
            raise ValueError("ox needs a ModelOxidation (md_model."
                             "model_oxidation builds one), or None")
        for label in ("v_bond_vu", "v_list_vu", "r_search_ang"):
            put(self, label, _optional_float(getattr(self, label), label))
        if self.formers is not None:
            if isinstance(self.formers, (str, bytes)):
                raise ValueError("formers needs a collection of element "
                                 "symbols, not one string")
            put(self, "formers", frozenset(validate_symbol(str(s))
                                           for s in self.formers))
        cutoffs = {}
        for pair, value in dict(self.cutoffs_ang).items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"cutoff key {pair!r} is not an element pair")
            pair = (validate_symbol(str(pair[0])), validate_symbol(str(pair[1])))
            distance = _optional_float(value, f"the {pair[0]}-{pair[1]} cutoff")
            if distance is None or distance <= 0.0:
                raise ValueError(f"the {pair[0]}-{pair[1]} cutoff is {value!r}; "
                                 "a distance above 0 Å is needed")
            cutoffs[pair] = distance
        sources = {}
        for pair, text in dict(self.cutoff_sources).items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"cutoff source key {pair!r} is not an "
                                 "element pair")
            pair = (validate_symbol(str(pair[0])), validate_symbol(str(pair[1])))
            sources[pair] = _name(text, f"the source of the {pair[0]}-"
                                        f"{pair[1]} cutoff")
        unsourced = sorted(set(cutoffs) - set(sources))
        if unsourced:
            raise ValueError("every cutoff needs its source (a g(r) minimum "
                             "and its method, or 'user'); none for "
                             + ", ".join(f"{a}-{b}" for a, b in unsourced))
        put(self, "cutoffs_ang", dict(sorted(cutoffs.items())))
        put(self, "cutoff_sources", dict(sorted(sources.items())))
        method = {}
        for key, value in dict(self.method_parameters).items():
            _name(key, "a method parameter name")
            method[key] = _parameter(value, key)
        put(self, "method_parameters", method)
        for label in ("estimated_pairs", "missing_pairs"):
            tallies = {}
            for pair, count in dict(getattr(self, label)).items():
                tallies[str(pair)] = int(_whole_numbers(count, f"{label}, "
                                                               f"{pair}"))
            put(self, label, tallies)
        put(self, "notes", _notes(self.notes, "provenance"))

    @classmethod
    def from_trajectory(cls, trajectory, *, frames_used: Sequence[int],
                        frames_skipped: Mapping[int, str] | None = None,
                        params=None, notes: Sequence[str] = (),
                        **fields_) -> "Provenance":
        """Fill the file-side fields from a ``md_model.Trajectory``.

        Takes source_path, file_format, type_map, type_map_source, units_note,
        the reader's skipped blocks, the timesteps and times of the frames
        used, and the reader's notes (load summary first, then ``notes``).
        ``params`` (a ``bv.ParameterSet``) fills params_name and
        params_source. Every other field passes through ``fields_``.
        """
        indices = [int(k) for k in frames_used]
        outside = [k for k in indices if not 0 <= k < trajectory.n_frames]
        if outside:
            raise ValueError(f"frame(s) {outside} are outside the "
                             f"{trajectory.n_frames} readable frames")
        times = getattr(trajectory, "times_ps", None)
        if params is not None:
            fields_.setdefault("params_name", params.name)
            fields_.setdefault("params_source", params.source)
        return cls(
            source_path=trajectory.source_path,
            file_format=trajectory.file_format,
            frames_used=tuple(indices),
            frames_skipped=dict(frames_skipped or {}),
            file_blocks_skipped=dict(trajectory.skipped),
            timesteps=tuple(int(trajectory.timesteps[k]) for k in indices),
            times_ps=() if times is None else
            tuple(float(times[k]) for k in indices),
            type_map=dict(trajectory.type_map),
            type_map_source=trajectory.type_map_source,
            units_note=trajectory.units_note,
            notes=tuple(trajectory.notes) + _notes(notes, "provenance"),
            **fields_)

    def with_notes(self, *extra: str) -> "Provenance":
        """A copy with sentences added to its notes."""
        return dataclasses.replace(self, notes=_merged(self.notes, *_notes(
            extra, "provenance")))

    def as_rows(self) -> list[dict[str, str]]:
        """The header as ``{'field', 'value'}`` rows (a provenance sheet).

        Every field appears, unset ones as 'not given', so a reader of the file
        can tell an absent input from a forgotten one.
        """
        rows: list[dict[str, str]] = []

        def add(label: str, value) -> None:
            rows.append({"field": label, "value": str(value)})

        add("program", f"FACET {self.facet_version}")
        add("created", self.created)
        add("source file", self.source_path)
        add("file format", self.file_format)
        if self.units_note:
            add("length unit", self.units_note)
        add("frames used", f"{len(self.frames_used)}: "
                           f"{_compact(self.frames_used)}")
        known = [t for t in self.timesteps if t != NO_TIMESTEP]
        if not known:
            add("timesteps", "not given in the file")
        else:
            add("timesteps", _compact(known)
                + (f" ({len(self.timesteps) - len(known)} frames without one)"
                   if len(known) < len(self.timesteps) else ""))
        finite = [t for t in self.times_ps if math.isfinite(t)]
        if finite:
            add("time (ps)", f"{min(finite)!r} to {max(finite)!r}")
        if self.frames_skipped:
            for k, reason in self.frames_skipped.items():
                add("frame skipped", f"{k}: {reason}")
        else:
            add("frames skipped", "none")
        if self.file_blocks_skipped:
            for position, reason in self.file_blocks_skipped.items():
                add("file block skipped by the reader",
                    f"position {position}: {reason}")
        else:
            add("file blocks skipped by the reader", "none")
        add("type map", ", ".join(f"{k} -> {v}" for k, v in
                                  self.type_map.items())
            or "none (the file names the elements)")
        add("type map source", self.type_map_source or "not given")
        add("oxidation states",
            self.ox.describe() if self.ox is not None else "not given")
        add("bond-valence parameters", self.params_name or "not given")
        if self.params_source:
            add("bond-valence parameter source", self.params_source)
        for label, value, unit in (("bond threshold v_bond", self.v_bond_vu,
                                    "v.u."),
                                   ("tabulation threshold v_list",
                                    self.v_list_vu, "v.u."),
                                   ("search radius", self.r_search_ang, "Å")):
            add(label, "not given" if value is None else f"{value!r} {unit}")
        if self.formers is None:
            add("network formers", "not given (no former-dependent result)")
        else:
            add("network formers", ", ".join(sorted(self.formers))
                or "none (an empty set was given)")
        if self.cutoffs_ang:
            for (a, b), value in self.cutoffs_ang.items():
                add("distance cutoff", f"{a}-{b}: {value!r} Å "
                                       f"({self.cutoff_sources[(a, b)]})")
        else:
            add("distance cutoffs", "none")
        for key, value in self.method_parameters.items():
            add("method parameter", f"{key} = {_format(value)}")
        for label, tallies in (("estimated bond-valence parameter",
                                self.estimated_pairs),
                               ("pair without a bond-valence parameter",
                                self.missing_pairs)):
            for pair, count in tallies.items():
                add(label, f"{pair}: {count} contacts")
        add("spread", SPREAD_NOTE)
        for note in self.notes:
            add("note", note)
        return rows

    def as_lines(self) -> list[str]:
        """The header as ``'field: value'`` lines, for ``exporters.write_csv``'s
        ``provenance`` argument."""
        return [f"{row['field']}: {row['value']}" for row in self.as_rows()]
