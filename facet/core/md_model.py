"""The array model of a periodic MD frame, and the one road into it.

Everything FACET does with a crystal goes through :class:`~.structure.Structure`:
one ``Site`` object per symmetry-distinct position, one ``Atom`` object per
expanded position, spglib run on every file that is read. That model suits a
20-site crystal and does not scale to a 10 000-atom glass. Measured
on this machine in Step 0 of the MD work (9 261 atoms, pinned to the P-cores):
the spglib annotation alone took 7.2 s of a 7.35 s read, and the per-site
analysis 26 s, part of it the O(N^2) ``Structure.atoms_of_site`` scan. So an MD
frame is held here as read-only arrays, and the readers, the vectorised bulk
engine and the glass descriptors all take the same :class:`Frame`.

This module is the contract those three share. It owns:

* :class:`Frame` and :func:`frame_from_arrays`, the single validated
  constructor every reader calls;
* :class:`Trajectory`, the lazy container a reader returns, and
  :class:`MemoryTrajectory` for synthetic models and tests;
* :class:`ModelOxidation`, the oxidation states an MD model is analysed with;
* :func:`frame_to_structure` and :func:`supercell_frame`, the two bridges to
  and from the crystal model.

CONVENTIONS, AND WHY EACH ONE
-----------------------------
**The box is three rows.** ``box_ang[0]``, ``[1]``, ``[2]`` are the cell vectors
a, b and c, which is how both MD formats write them: LAMMPS defines a restricted
triclinic box as a = (lx, 0, 0), b = (xy, ly, 0), c = (xz, yz, lz) [1], and the
extended XYZ ``Lattice="R1x R1y R1z R2x R2y R2z R3x R3y R3z"`` lists the three
vectors in order [2]. FACET's crystal side stores the same vectors as *columns*:
``readers.cell_from_vectors`` builds ``Cell.orth`` with ``np.column_stack``, and
``Cell.to_cartesian(f)`` is ``orth @ f``. So ``Cell.orth == box_ang.T`` and
``Cell.to_cartesian(frac) == frac @ box_ang``; both are checked at runtime by
the tests on a strongly tilted box.

**A box has an origin.** A LAMMPS box spans (xlo, ylo, zlo) to (xhi, yhi, zhi),
and unscaled ``x`` / ``xu`` columns are absolute positions, so a box without its
origin cannot be written back out. ``cart_ang = origin_ang + frac @ box_ang``.
``Cell`` has no origin, so :func:`frame_to_structure` drops it and says so.

**Fractional coordinates are wrapped exactly into [0, 1).** The neighbour
search can use ``cKDTree(boxsize=)``, which refuses a coordinate equal to the
box length, and ``(-1e-17) % 1.0`` is ``1.0`` in floating point (measured).
:func:`wrap_frac` computes ``f - floor(f)`` and sets any 1.0 produced by that
rounding to 0.0, the same atom.

**Rows are sorted by atom id.** A LAMMPS dump lists atoms in whatever order the
processors wrote them, which changes from frame to frame. Sorting by the file's
id makes row k the same atom in every frame. ``atom_id`` is a label carried
from the file; every array in FACET is indexed by *row*, never by id.

**Element symbols are validated, never normalised.** ``elements.normalise`` is
built to recover an element from a CIF label, so it reduces whatever it is
given: ``'1'`` -> ``''``, ``'type1'`` -> ``'Type'``, ``'Ob'`` -> ``'O'``, and 15
real symbols to a different element (He -> H, Ne -> N, Kr -> K, Og -> O, and 11
others; measured over gemmi's periodic table). On an MD file, where the column
might hold a type number or a force-field label, that turns a missing type map
into a silently relabelled model. :func:`validate_symbol` accepts an exact
symbol (any case, an optional charge such as ``Na+`` or ``O2-``) and refuses
everything else. :func:`symbols_changed_by_normalise` lists the symbols the
crystal tools would turn into another element, because ``Site.__post_init__``
and ``bv.ParameterSet.get`` both call ``normalise``.

**Oxidation states are inputs.** For a crystal, FACET resolves them from the
geometry (``oxidation.resolve``). For a model, the states are what the person
who built it put in the force field, so they are taken from
``elements.COMMON_OX`` overlaid by the user's map, with the source of each
recorded (:class:`ModelOxidation`), and never resolved.

**Dynamics need more than positions.** Mean-square displacement needs
continuous (unwrapped) positions, velocity autocorrelation needs velocities,
and both need the time of each frame [3]. A frame carries them when the file
has them (``unwrapped_cart_ang``, ``vel_ang_per_ps``, ``time_ps``), with the
unit in the name, and ``None`` when it does not. Nothing is filled in.

TIMINGS
-------
Measured on this machine (Windows 11, i5-13420H, pinned to the P-cores, on a
shared machine where Step 0 found single runs vary by about 30 %). The boxes
are 2.3 Å jittered grids with a strong shear, Si/O/Na drawn at random, a third
of the atoms placed one cell outside the box and the ids shuffled, so the
wrap, the solve and the sort all run. Medians of 3 to 7 runs.

* ``frame_from_arrays`` from Cartesian positions: 0.6-0.9 ms at 1 000 atoms,
  4-7 ms at 9 261 and 10 648. Building the frame is not where the time goes.
* ``frame_to_structure`` with the symmetry search skipped: 5 ms at 1 000
  atoms, 26 ms at 3 375, 70-84 ms at 9 261 (one Site and one Atom per atom).
* The spglib search inside it added 0.02 s at 1 000 atoms, 0.20 s at 3 375 and
  1.75 s at 9 261, close to N^2. Step 0 measured 0.10, 0.70 and 7.2 s on its
  cubic Si/O/Na boxes through ``readers.read_xyz``; the box and the
  composition change the constant, not the trend. :data:`SPGLIB_MAX_ATOMS`
  keeps the search where it costs a fraction of a second.
* ``supercell_frame`` 2x2x2 of a 1 000-atom structure: 3-6 ms.

REFERENCES
----------
[1] LAMMPS documentation, "Triclinic (non-orthogonal) simulation boxes",
    https://docs.lammps.org/Howto_triclinic.html
[2] The extended XYZ specification, libAtoms, https://github.com/libAtoms/extxyz
[3] M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
    Oxford University Press, Oxford (2017).
"""
from __future__ import annotations

import functools
import itertools
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from . import elements as element_data
from .structure import Atom, Site, Structure

__all__ = [
    "SPGLIB_MAX_ATOMS", "BOX_MIN_NORMALISED_VOLUME", "NO_TIMESTEP",
    "NOTE_SYMMETRY_NOT_SEARCHED", "TYPE_MAP_SOURCES",
    "FrameError", "Frame", "frame_from_arrays", "wrap_frac", "validate_symbol",
    "symbols_changed_by_normalise", "unwrap_with_images", "charges_per_element",
    "Trajectory", "MemoryTrajectory", "ModelOxidation", "model_oxidation",
    "frame_to_structure", "supercell_frame",
]

# ---------------------------------------------------------------------------
# constants: each one is a choice, and says so
# ---------------------------------------------------------------------------

# Above this many atoms frame_to_structure does not run the spglib search. A
# choice, not a property of spglib, set from measurements on this machine:
# spglib took 0.10 s at 1 000 atoms, 0.70 s at 3 375 and 7.2 s at 9 261 in
# Step 0 (cubic boxes), and 0.02 / 0.20 / 1.75 s on this module's sheared
# boxes (TIMINGS above). The value proposed in Step 0; Sam has not set it yet.
SPGLIB_MAX_ATOMS: int = 1000

# A box is refused as degenerate when |det(box)| / (|a| |b| |c|) is below this.
# A parsing choice, not a physical limit: the ratio is 1 for a rectangular box
# and goes to 0 as the three rows become coplanar, and at 1e-6 the inverse
# used for the fractional coordinates has lost about six significant figures.
BOX_MIN_NORMALISED_VOLUME: float = 1e-6

# Trajectory.timesteps holds this where the file gives no timestep for a frame.
# A placeholder, never a time: LAMMPS timesteps are not negative.
NO_TIMESTEP: int = -1

# The note frame_to_structure adds when it skipped the symmetry search, worded
# as the MD prompt gives it. Added only when the search was skipped: below the
# threshold the search runs, and the note would be untrue.
NOTE_SYMMETRY_NOT_SEARCHED = ("amorphous model, P1 by construction; "
                              "symmetry not searched")

# Where a trajectory's type -> element map came from. A reader that combines
# two sources joins them with ' + ' (for example 'user + data-file masses').
TYPE_MAP_SOURCES = ("user", "element column", "type label", "data-file masses",
                    "file symbols", "in memory")

# How far Frame.cart_ang may sit from origin_ang + frac @ box_ang when a Frame is
# built directly instead of through frame_from_arrays (which makes them equal).
# A consistency check on construction, not a physical value.
_CART_CONSISTENCY_ANG = 1e-9


class FrameError(ValueError):
    """One frame of a trajectory could not be read, or differs from the others.

    A subclass of ValueError, so the readers' unreadable-file contract holds, and a
    class of its own, so a caller averaging over frames can catch it, record the
    frame and its reason, and go on.
    """


# ---------------------------------------------------------------------------
# element symbols
# ---------------------------------------------------------------------------

# One or two letters, then optionally a charge written as '2-', '+', '+1' or '-'.
# Digits without a sign ('Si1') are a site label, not a charge, and are refused.
_SYMBOL = re.compile(r"([A-Za-z]{1,2})(?:[0-9]?[+-]|[+-][0-9]?)?")


@functools.lru_cache(maxsize=1)
def _canonical_symbols() -> frozenset[str]:
    """Every element symbol in gemmi's periodic table, H to the last it knows.

    gemmi rather than facet.core.elements, because elements.info normalises its
    argument first, so info('He') returns hydrogen's data. Built by walking the
    atomic numbers until gemmi stops recognising them, so no count is assumed;
    gemmi.Element(1).name is 'H', which keeps the isotope label 'D' out.
    """
    import gemmi

    out = set()
    z = 1
    while True:
        el = gemmi.Element(z)
        if el.atomic_number != z:
            break
        out.add(el.name)
        z += 1
    return frozenset(out)


@functools.lru_cache(maxsize=4096)
def validate_symbol(token: str) -> str:
    """The element symbol ``token`` names, or ValueError naming the token.

    Accepted: an exact element symbol in any case (``'si'``, ``'SI'`` and
    ``'Si'`` all give ``'Si'``), optionally followed by a charge (``'Na+'``,
    ``'O2-'``, ``'Na+1'``). The charge is not used: oxidation states come from
    :func:`model_oxidation`.

    Refused, each with a message: ``'1'``, ``'type1'``, ``'Si1'``, ``'Ob'``,
    ``'O_b'``, ``'X'``, ``'D'``, an empty string, and anything that is not text.
    ``elements.normalise`` would have turned several of these into a real
    element ('Ob' into O, 'He' into H); this function never relabels.
    """
    if not isinstance(token, str):
        raise ValueError(
            f"{token!r} is not an element symbol: a text symbol is needed, and "
            "numeric atom types need a type map to elements")
    text = token.strip()
    match = _SYMBOL.fullmatch(text)
    if match is None:
        raise ValueError(
            f"{token!r} is not an element symbol; atom types, site labels such "
            "as 'Si1' or 'Ob', and force-field names need an explicit type map "
            "to elements")
    letters = match.group(1)
    symbol = letters[0].upper() + letters[1:].lower()
    if symbol not in _canonical_symbols():
        raise ValueError(
            f"{token!r} is not an element symbol in gemmi's periodic table "
            "(isotope labels such as 'D' and placeholders such as 'X' included); "
            "it needs an explicit type map to an element")
    return symbol


def symbols_changed_by_normalise(symbols: Iterable[str]) -> dict[str, str]:
    """Each validated symbol that ``elements.normalise`` turns into another one.

    ``{'He': 'H', 'Kr': 'K'}`` for a model holding He and Kr. The crystal model
    calls normalise in ``Site.__post_init__`` and in ``bv.ParameterSet.get``,
    so such an element would silently take another element's identity and
    bond-valence parameters there. :func:`frame_to_structure` refuses these;
    the bulk engine is expected to report their contacts as unparameterised.
    """
    out: dict[str, str] = {}
    for token in sorted({str(s) for s in symbols}):
        symbol = validate_symbol(token)
        reduced = element_data.normalise(symbol)
        if reduced != symbol:
            out[symbol] = reduced
    return out


# ---------------------------------------------------------------------------
# small array helpers
# ---------------------------------------------------------------------------

def wrap_frac(frac) -> np.ndarray:
    """``f - floor(f)``, with any exact 1.0 produced by rounding set to 0.0.

    For finite f the difference lies in [0, 1] and reaches 1.0 only by rounding,
    for a tiny negative f: ``-1e-17 - floor(-1e-17)`` is ``1.0``. That value is
    the same atom as 0.0, and is set to it, so every result is in [0, 1).
    Returns a new float64 array; non-finite input raises ValueError.
    """
    f = np.array(frac, dtype=np.float64)
    finite = np.isfinite(f)
    if not finite.all():
        raise ValueError(
            f"{int(f.size - finite.sum())} fractional coordinate value(s) are "
            "not finite (NaN or inf) and cannot be wrapped into the box")
    out = f - np.floor(f)
    out[out >= 1.0] = 0.0
    return out


def unwrap_with_images(cart_ang, image, box_ang) -> np.ndarray:
    """Continuous positions from file positions and image flags.

    ``cart_ang + image @ box_ang``: the position the atom would have if it had
    never been wrapped back into the box. ``cart_ang`` here is the coordinate
    as the file wrote it (LAMMPS ``x y z`` with ``ix iy iz``), not a position
    FACET has already re-wrapped, because the image flags count the crossings
    made to reach that coordinate. One definition, so every reader unwraps the
    same way.
    """
    cart = _float_array(cart_ang, "cart_ang")
    if cart.ndim != 2 or cart.shape[1] != 3:
        raise ValueError(f"cart_ang has shape {cart.shape}; (N, 3) is needed")
    box = _checked_box(box_ang)
    flags = np.asarray(image)
    if flags.shape != cart.shape:
        raise ValueError(
            f"image flags have shape {flags.shape} but the positions have "
            f"{cart.shape}")
    if flags.dtype.kind not in "iu":
        as_float = _float_array(flags, "image")
        if not np.isfinite(as_float).all() or \
                (as_float != np.round(as_float)).any():
            raise ValueError("image flags are not whole numbers")
        flags = as_float
    _require_finite(cart, "cart_ang")
    return cart + flags.astype(np.float64) @ box


def charges_per_element(elements, charge_e
                        ) -> tuple[dict[str, float] | None, list[str]]:
    """Per-element charges when every atom of each element carries the same q.

    Compared exactly: two atoms written with the same digits parse to the same
    float, and any other difference is a real difference (a charge-equilibration
    model, for instance). When any element's charges differ, the result is
    None, with a note giving each such element's range; the per-atom charges
    stay on the frames.
    """
    symbols = np.asarray(elements)
    charges = _float_array(charge_e, "charge_e")
    if charges.shape != symbols.shape:
        raise ValueError(
            f"{charges.size} charges for {symbols.size} atoms")
    _require_finite(charges, "charge_e")
    out: dict[str, float] = {}
    differing = []
    for symbol in sorted({str(s) for s in symbols}):
        values = np.unique(charges[symbols == symbol])
        if values.size == 1:
            out[symbol] = float(values[0])
        else:
            differing.append(f"{symbol} ({values.min():.6g} to "
                             f"{values.max():.6g} e)")
    if differing:
        return None, [
            "per-atom charges differ within " + ", ".join(differing)
            + "; no per-element charge is stated and the per-atom charges stay "
            "on each frame"]
    return out, []


def _float_array(value, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} could not be read as numbers ({error})") \
            from None


def _require_finite(array: np.ndarray, name: str) -> None:
    finite = np.isfinite(array)
    if not finite.all():
        raise ValueError(
            f"{name} holds {int(array.size - finite.sum())} value(s) that are "
            "not finite (NaN or inf)")


def _checked_box(box_ang) -> np.ndarray:
    """A (3, 3) float64 box with three rows that span a volume, or ValueError."""
    box = _float_array(box_ang, "box_ang")
    if box.shape != (3, 3):
        raise ValueError(
            f"box_ang has shape {box.shape}; it needs the three cell vectors "
            "a, b, c as the rows of a (3, 3) array")
    _require_finite(box, "box_ang")
    lengths = np.linalg.norm(box, axis=1)
    for name, length in zip("abc", lengths):
        if length == 0.0:
            raise ValueError(
                f"box vector {name} has zero length, so the box spans no volume")
    volume = abs(float(np.linalg.det(box)))
    ratio = volume / float(np.prod(lengths))
    if ratio < BOX_MIN_NORMALISED_VOLUME:
        raise ValueError(
            f"the box is degenerate: its volume {volume:.6g} Å^3 is "
            f"{ratio:.3g} of |a||b||c|, below {BOX_MIN_NORMALISED_VOLUME:g} "
            "(BOX_MIN_NORMALISED_VOLUME, a parsing choice), so the rows a, b, c "
            "are coplanar or nearly so and fractional coordinates cannot be "
            "computed from it")
    return box


def _frozen(array: np.ndarray) -> np.ndarray:
    """A read-only, C-contiguous array that owns its data (copied if needed)."""
    if array.flags.writeable or array.base is not None \
            or not array.flags.c_contiguous:
        array = np.array(array, copy=True, order="C")
    array.setflags(write=False)
    return array


# ---------------------------------------------------------------------------
# the frame
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Frame:
    """One snapshot of a periodic model, as read-only arrays.

    Build it with :func:`frame_from_arrays`, which validates, sorts and wraps.
    Constructing a Frame directly re-checks the invariants below and refuses
    any that fail, but does not sort, wrap or convert.

    Invariants (N atoms, rows sorted by ``atom_id``):

    * ``elements`` (N,) ``'<U2'``, every entry a symbol :func:`validate_symbol`
      returns unchanged;
    * ``frac`` (N, 3) float64, every value in [0, 1);
    * ``cart_ang`` (N, 3) float64, ``origin_ang + frac @ box_ang``;
    * ``box_ang`` (3, 3) float64, cell vectors a, b, c as ROWS
      (``Cell.orth == box_ang.T``);
    * ``origin_ang`` (3,) float64, the box corner (LAMMPS xlo, ylo, zlo);
    * ``atom_id`` (N,) int64, strictly increasing, the file's own ids
      (0 .. N-1 when the file has none);
    * ``timestep`` int or None; ``time_ps`` float or None, in picoseconds;
    * ``unwrapped_cart_ang`` (N, 3) or None: continuous positions, for dynamics;
    * ``vel_ang_per_ps`` (N, 3) or None: velocities in Å/ps;
    * ``charge_e`` (N,) or None: per-atom charges in units of e;
    * ``notes``: provenance, as sentences.

    Every array is read-only. A frame pickles and copies through its own
    constructor, so a copy is frozen and checked like the original.
    """

    elements: np.ndarray
    frac: np.ndarray
    cart_ang: np.ndarray
    box_ang: np.ndarray
    origin_ang: np.ndarray
    atom_id: np.ndarray
    timestep: int | None = None
    time_ps: float | None = None
    unwrapped_cart_ang: np.ndarray | None = None
    vel_ang_per_ps: np.ndarray | None = None
    charge_e: np.ndarray | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_frame(self)

    def __reduce__(self):
        return (Frame, tuple(getattr(self, f.name) for f in fields(self)))

    # -- derived quantities ---------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return int(self.elements.shape[0])

    @property
    def volume_ang3(self) -> float:
        """|det(box_ang)|, in Å^3."""
        return abs(float(np.linalg.det(self.box_ang)))

    @property
    def number_density_per_ang3(self) -> float:
        """Atoms per Å^3: n_atoms / volume_ang3."""
        return self.n_atoms / self.volume_ang3

    @property
    def composition(self) -> dict[str, int]:
        """Element -> number of atoms, in alphabetical order of the symbol."""
        symbols, counts = np.unique(self.elements, return_counts=True)
        return {str(s): int(c) for s, c in zip(symbols, counts)}

    @property
    def species(self) -> tuple[str, ...]:
        """The elements present, in alphabetical order."""
        return tuple(str(s) for s in np.unique(self.elements))

    @property
    def perpendicular_widths_ang(self) -> np.ndarray:
        """The box widths perpendicular to the bc, ca and ab faces, in Å.

        V/|b x c|, V/|c x a|, V/|a x b|: the formula
        ``neighbors._images_within`` uses to count images, so the two agree. Half
        the smallest width is the largest radius at which the minimum-image
        convention still finds every pair once.
        """
        a, b, c = self.box_ang
        volume = self.volume_ang3
        return np.array([volume / np.linalg.norm(np.cross(b, c)),
                         volume / np.linalg.norm(np.cross(c, a)),
                         volume / np.linalg.norm(np.cross(a, b))])

    @property
    def box_is_diagonal(self) -> bool:
        """True when every off-diagonal entry of box_ang is exactly 0."""
        off = self.box_ang - np.diag(np.diag(self.box_ang))
        return not np.any(off)

    def cell(self):
        """The crystal model's Cell for this box (``Cell.orth == box_ang.T``)."""
        from . import readers

        return readers.cell_from_vectors(self.box_ang)

    def with_notes(self, *extra: str) -> "Frame":
        """A copy with sentences appended to ``notes``; the arrays are shared."""
        for note in extra:
            if not isinstance(note, str):
                raise ValueError(f"a note must be text, not {type(note).__name__}")
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["notes"] = tuple(self.notes) + tuple(extra)
        return Frame(**values)

    def __repr__(self) -> str:
        step = "" if self.timestep is None else f", timestep {self.timestep}"
        return (f"<Frame {self.n_atoms} atoms, "
                f"{self.volume_ang3:.6g} Å^3{step}>")


def _check_frame(frame: Frame) -> None:
    """Validate a Frame's invariants and freeze its arrays in place."""
    def put(name, value):
        object.__setattr__(frame, name, value)

    symbols = np.asarray(frame.elements)
    if symbols.ndim != 1 or symbols.dtype.kind != "U":
        raise ValueError(
            "Frame.elements needs a one-dimensional array of element symbols; "
            "frame_from_arrays builds one from symbols or text tokens")
    n = symbols.shape[0]
    if n == 0:
        raise ValueError("the frame holds no atoms; an empty box has nothing "
                         "to measure")
    for token in np.unique(symbols):
        if validate_symbol(str(token)) != str(token):
            raise ValueError(
                f"Frame.elements holds {str(token)!r}, which is not in the form "
                "validate_symbol returns; build frames with frame_from_arrays")
    put("elements", _frozen(symbols.astype("<U2", copy=False)))

    box = _checked_box(frame.box_ang)
    origin = _float_array(frame.origin_ang, "origin_ang")
    if origin.shape != (3,):
        raise ValueError(f"origin_ang has shape {origin.shape}; (3,) is needed")
    _require_finite(origin, "origin_ang")
    put("box_ang", _frozen(box))
    put("origin_ang", _frozen(origin))

    frac = _float_array(frame.frac, "frac")
    if frac.shape != (n, 3):
        raise ValueError(f"frac has shape {frac.shape} for {n} atoms")
    _require_finite(frac, "frac")
    if (frac < 0.0).any() or (frac >= 1.0).any():
        raise ValueError(
            "Frame.frac holds values outside [0, 1); frame_from_arrays wraps "
            "them")
    put("frac", _frozen(frac))

    cart = _float_array(frame.cart_ang, "cart_ang")
    if cart.shape != (n, 3):
        raise ValueError(f"cart_ang has shape {cart.shape} for {n} atoms")
    _require_finite(cart, "cart_ang")
    gap = float(np.abs(cart - (origin + frac @ box)).max())
    if gap > _CART_CONSISTENCY_ANG:
        raise ValueError(
            f"cart_ang differs from origin_ang + frac @ box_ang by up to "
            f"{gap:.3g} Å; frame_from_arrays computes one from the other")
    put("cart_ang", _frozen(cart))

    ids = np.asarray(frame.atom_id)
    if ids.shape != (n,) or ids.dtype.kind not in "iu":
        raise ValueError(f"atom_id needs {n} integers")
    ids = ids.astype(np.int64, copy=False)
    if n > 1 and not (np.diff(ids) > 0).all():
        raise ValueError(
            "Frame.atom_id is not strictly increasing (duplicated or unsorted "
            "ids); frame_from_arrays sorts rows and refuses duplicates")
    put("atom_id", _frozen(ids))

    put("timestep", _checked_timestep(frame.timestep))
    put("time_ps", _checked_time(frame.time_ps))
    for name, shape in (("unwrapped_cart_ang", (n, 3)),
                        ("vel_ang_per_ps", (n, 3)), ("charge_e", (n,))):
        value = getattr(frame, name)
        if value is not None:
            put(name, _frozen(_checked_optional(value, shape, name)))

    notes = frame.notes
    if isinstance(notes, str) or not all(isinstance(x, str) for x in notes):
        raise ValueError("Frame.notes needs a sequence of sentences (text)")
    put("notes", tuple(notes))


def _checked_timestep(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or \
            not isinstance(value, (int, np.integer)):
        raise ValueError(f"timestep {value!r} is not an integer")
    return int(value)


def _checked_time(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or \
            not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"time_ps {value!r} is not a number")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"time_ps {value!r} is not finite")
    return out


def _checked_optional(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = _float_array(value, name)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; {shape} is needed")
    _require_finite(array, name)
    return array


def _symbol_array(elements) -> np.ndarray:
    """(N,) '<U2' validated symbols from symbols, text or bytes tokens."""
    if isinstance(elements, (str, bytes)):
        raise ValueError(
            "elements needs one symbol per atom, not a single string")
    raw = np.asarray(elements)
    if raw.ndim != 1:
        raise ValueError(
            f"elements has shape {raw.shape}; one symbol per atom is needed")
    if raw.size == 0:                   # [] arrives as float64; no atoms, no type
        return np.zeros(0, dtype="<U2")
    if raw.dtype.kind in "biuf":
        raise ValueError(
            "elements holds numbers, not symbols: numeric atom types need a "
            "type map to elements")
    if raw.dtype.kind == "S":
        raw = np.char.decode(raw, "ascii")
    if raw.dtype.kind == "O":
        if not all(isinstance(x, str) for x in raw):
            raise ValueError(
                "elements holds values that are not text; numeric atom types "
                "need a type map to elements")
        raw = raw.astype(str)
    tokens, inverse = np.unique(raw, return_inverse=True)
    symbols = np.array([validate_symbol(str(t)) for t in tokens], dtype="<U2")
    return symbols[inverse.reshape(-1)]


def _atom_id_array(atom_id, n: int) -> np.ndarray:
    if atom_id is None:
        return np.arange(n, dtype=np.int64)
    ids = np.asarray(atom_id)
    if ids.shape != (n,):
        raise ValueError(f"{ids.size} atom ids for {n} atoms")
    if ids.dtype.kind == "f":
        if not np.isfinite(ids).all() or (ids != np.round(ids)).any():
            raise ValueError("atom ids are not whole numbers")
    elif ids.dtype.kind not in "iu":
        raise ValueError(f"atom ids of type {ids.dtype} are not integers")
    ids = ids.astype(np.int64)
    values, counts = np.unique(ids, return_counts=True)
    repeated = values[counts > 1]
    if repeated.size:
        raise ValueError(
            f"{repeated.size} atom id(s) occur more than once (first: "
            f"{repeated[:5].tolist()}); each atom needs its own id")
    return ids


def frame_from_arrays(elements, cart_ang=None, *, frac=None, box_ang,
                      origin_ang=None, atom_id=None, timestep: int | None = None,
                      time_ps: float | None = None, unwrapped_cart_ang=None,
                      vel_ang_per_ps=None, charge_e=None,
                      notes: Sequence[str] = ()) -> Frame:
    """The validated constructor every reader uses.

    Give exactly one of ``cart_ang`` (absolute Cartesian positions, Å) or
    ``frac`` (fractional coordinates of ``box_ang``). Either may lie outside the
    box; the frame stores ``frac = wrap_frac(...)`` and
    ``cart_ang = origin_ang + frac @ box_ang``, and a note counts the atoms that
    were moved into the box.

    Refused, each with a ValueError saying why: symbols that are not elements
    (numbers included), no atoms, both or neither of ``cart_ang`` / ``frac``,
    arrays whose lengths disagree, non-finite positions, velocities, charges or
    box, a degenerate box (:data:`BOX_MIN_NORMALISED_VOLUME`), duplicate atom
    ids, a non-integer timestep.

    Rows are sorted by ``atom_id`` (stable), and every per-atom array moves with
    them; ``atom_id=None`` numbers the atoms 0 .. N-1 in the order given.
    ``unwrapped_cart_ang`` is stored as given (it is continuous by definition).
    """
    symbols = _symbol_array(elements)
    n = symbols.shape[0]
    if n == 0:
        raise ValueError("the frame holds no atoms; an empty box has nothing "
                         "to measure")
    if (cart_ang is None) == (frac is None):
        raise ValueError("give exactly one of cart_ang and frac")
    if isinstance(notes, str):
        raise ValueError("notes needs a sequence of sentences, not one string")
    note_list = []
    for note in notes:
        if not isinstance(note, str):
            raise ValueError(f"a note must be text, not {type(note).__name__}")
        note_list.append(note)

    box = _checked_box(box_ang)
    if origin_ang is None:
        origin = np.zeros(3)
    else:
        origin = _float_array(origin_ang, "origin_ang")
        if origin.shape != (3,):
            raise ValueError(
                f"origin_ang has shape {origin.shape}; (3,) is needed")
        _require_finite(origin, "origin_ang")
    ids = _atom_id_array(atom_id, n)

    name = "cart_ang" if cart_ang is not None else "frac"
    given = _float_array(cart_ang if cart_ang is not None else frac, name)
    if given.shape != (n, 3):
        raise ValueError(
            f"{name} has shape {given.shape} but there are {n} element "
            "symbols; one (x, y, z) row per atom is needed")
    finite = np.isfinite(given).all(axis=1)
    if not finite.all():
        non_finite = ids[~finite]
        raise ValueError(
            f"{non_finite.size} atom(s) have non-finite coordinates (NaN or "
            f"inf) in {name}; first atom ids: {non_finite[:5].tolist()}")
    if cart_ang is not None:
        raw = np.linalg.solve(box.T, (given - origin).T).T
    else:
        raw = given

    extras = {}
    for key, value, shape in (("unwrapped_cart_ang", unwrapped_cart_ang, (n, 3)),
                              ("vel_ang_per_ps", vel_ang_per_ps, (n, 3)),
                              ("charge_e", charge_e, (n,))):
        if value is not None:
            extras[key] = _checked_optional(value, shape, key)

    if n > 1 and not (np.diff(ids) > 0).all():
        order = np.argsort(ids, kind="stable")
        symbols, ids, raw = symbols[order], ids[order], raw[order]
        extras = {key: value[order] for key, value in extras.items()}
        note_list.append("rows sorted by atom id; the input listed the atoms "
                         "in a different order")

    outside = int(((raw < 0.0) | (raw >= 1.0)).any(axis=1).sum())
    wrapped = wrap_frac(raw)
    if outside:
        note_list.append(f"{outside} of {n} atoms lay outside the box and were "
                         "wrapped into it")
    if np.linalg.det(box) < 0:
        note_list.append("the box rows a, b, c form a left-handed set "
                         "(det(box_ang) < 0); the volume is |det|")
    cart = origin + wrapped @ box

    # Fresh copies before freezing: np.asarray passes a caller's float64 array
    # through unchanged, and setting the flag on it would freeze their array.
    arrays = {key: np.array(value, copy=True, order="C") for key, value in
              {"elements": symbols, "frac": wrapped, "cart_ang": cart,
               "box_ang": box, "origin_ang": origin, "atom_id": ids,
               **extras}.items()}
    for value in arrays.values():
        value.setflags(write=False)
    return Frame(timestep=_checked_timestep(timestep),
                 time_ps=_checked_time(time_ps), notes=tuple(note_list),
                 **arrays)


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------

class Trajectory:
    """A model file: frames loaded on demand, plus everything that was assumed.

    A reader subclasses this, indexes the file once (where each frame starts,
    which frames are complete), passes what it learned to ``__init__`` and
    implements :meth:`_load`. ``frame(k)`` then loads one frame at a time, so
    100 frames of 10 000 atoms are never held at once.

    Two index spaces, kept apart:

    * ``k``, 0 .. n_frames-1, over the readable frames; what ``frame(k)`` and
      ``timesteps[k]`` use;
    * the *file position*, 0-based over every frame block in the file,
      readable or not; what ``skipped`` is keyed by. ``file_positions[k]`` maps
      one to the other. Every block is either readable or skipped, so the
      positions are the complement of ``skipped`` and are derived, not passed.

    Attributes: ``source_path``, ``file_format``, ``n_atoms`` (every readable
    frame has exactly this many), ``n_frames``, ``timesteps`` ((n_frames,)
    int64, :data:`NO_TIMESTEP` where the file gives none), ``times_ps``
    ((n_frames,) float64 with NaN where unknown, or None when no frame has a
    time), ``file_positions``, ``type_map`` (LAMMPS integer types and text labels
    such as DL_POLY's ``'O_b'`` -> element; empty when the file names the
    elements), ``type_map_source`` (one or more of :data:`TYPE_MAP_SOURCES`
    joined by ' + '), ``charges_e`` (per element, only when identical for every
    atom of the element; see :func:`charges_per_element`), ``skipped``
    (file position -> reason), ``notes``, ``units_note`` (the unit assumption,
    as a sentence), ``box_varies`` (True / False when the reader checked every
    frame's box, None when it did not).
    """

    def __init__(self, *, source_path: str | Path, file_format: str,
                 n_atoms: int, n_frames: int, type_map_source: str,
                 timesteps: Sequence[int] | np.ndarray | None = None,
                 times_ps: Sequence[float] | np.ndarray | None = None,
                 type_map: Mapping[int | str, str] | None = None,
                 charges_e: Mapping[str, float] | None = None,
                 skipped: Mapping[int, str] | None = None,
                 notes: Sequence[str] | None = None, units_note: str = "",
                 box_varies: bool | None = None) -> None:
        path = Path(str(source_path))
        try:
            exists = path.exists()
        except OSError:
            exists = False
        self.source_path = str(path.resolve()) if exists else str(source_path)
        if not isinstance(file_format, str) or not file_format:
            raise ValueError("file_format needs a name such as 'lammps-dump'")
        self.file_format = file_format

        self.skipped: dict[int, str] = {}
        for position, reason in (skipped or {}).items():
            if isinstance(position, (bool, np.bool_)) or \
                    not isinstance(position, (int, np.integer)):
                raise ValueError(f"skipped is keyed by file position (an "
                                 f"integer), not {position!r}")
            self.skipped[int(position)] = str(reason)

        for name, value in (("n_atoms", n_atoms), ("n_frames", n_frames)):
            if isinstance(value, (bool, np.bool_)) or \
                    not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} {value!r} is not an integer")
        if n_atoms < 1:
            raise ValueError("the model holds no atoms; an empty box has "
                             "nothing to measure")
        if n_frames < 1:
            reasons = "; ".join(f"position {p}: {r}"
                                for p, r in sorted(self.skipped.items()))
            raise ValueError(
                f"{source_path} holds no readable frame"
                + (f" ({reasons})" if reasons else ""))
        self.n_atoms = int(n_atoms)
        self.n_frames = int(n_frames)

        total = self.n_frames + len(self.skipped)
        outside = [p for p in self.skipped if not 0 <= p < total]
        if outside:
            raise ValueError(
                f"skipped names file positions {outside} outside 0 .. "
                f"{total - 1} ({self.n_frames} readable + {len(self.skipped)} "
                "skipped frames)")
        positions = np.array([p for p in range(total) if p not in self.skipped],
                             dtype=np.int64)
        positions.setflags(write=False)
        self.file_positions = positions

        if timesteps is None:
            steps = np.full(self.n_frames, NO_TIMESTEP, dtype=np.int64)
        else:
            raw = np.asarray(timesteps)
            if raw.shape != (self.n_frames,) or raw.dtype.kind not in "iu":
                raise ValueError(
                    f"timesteps needs {self.n_frames} integers, one per readable "
                    f"frame (NO_TIMESTEP = {NO_TIMESTEP} where none)")
            steps = raw.astype(np.int64)
        steps.setflags(write=False)
        self.timesteps = steps

        if times_ps is None:
            self.times_ps = None
        else:
            times = _float_array(times_ps, "times_ps")
            if times.shape != (self.n_frames,):
                raise ValueError(
                    f"times_ps needs {self.n_frames} values, one per readable "
                    "frame (NaN where unknown)")
            if np.isinf(times).any():
                raise ValueError("times_ps holds an infinite time")
            times.setflags(write=False)
            self.times_ps = times

        self.type_map: dict[int | str, str] = {}
        for key, value in (type_map or {}).items():
            if isinstance(key, (bool, np.bool_)) or \
                    not isinstance(key, (int, np.integer, str)):
                raise ValueError(
                    f"type map key {key!r} is neither an integer type nor a "
                    "text label")
            self.type_map[int(key) if isinstance(key, (int, np.integer))
                          else key] = validate_symbol(value)
        parts = [p.strip() for p in str(type_map_source).split("+")]
        unknown = [p for p in parts if p not in TYPE_MAP_SOURCES]
        if unknown:
            raise ValueError(
                f"type_map_source {type_map_source!r}: {unknown} not among "
                f"{TYPE_MAP_SOURCES} (several may be joined with ' + ')")
        self.type_map_source = " + ".join(parts)

        if charges_e is None:
            self.charges_e = None
        else:
            self.charges_e = {}
            for key, value in charges_e.items():
                charge = float(value)
                if not np.isfinite(charge):
                    raise ValueError(f"the charge of {key} is not finite")
                self.charges_e[validate_symbol(key)] = charge

        if notes is not None and isinstance(notes, str):
            raise ValueError("notes needs a sequence of sentences, not a string")
        self.notes: list[str] = [str(x) for x in (notes or [])]
        self.units_note = str(units_note)
        if box_varies not in (None, True, False):
            raise ValueError("box_varies is True, False or None")
        self.box_varies = box_varies

        # The atoms of the first frame loaded; every later frame is held to it.
        self._reference: tuple[np.ndarray, np.ndarray, int] | None = None

    # -- what a reader implements ---------------------------------------------
    def _load(self, k: int) -> Frame:
        """Readable frame k, built with frame_from_arrays. Subclasses only."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _load")

    # -- access ---------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_frames

    def file_position(self, k: int) -> int:
        """Where readable frame k sits among all the file's frame blocks."""
        return int(self.file_positions[self._index(k)])

    def frame(self, k: int) -> Frame:
        """Readable frame k (negative k counts from the end).

        Raises IndexError outside the range, and :class:`FrameError` when the
        frame cannot be read or holds different atoms (count, ids or elements)
        from the first frame loaded, or a timestep other than the index says.
        """
        k = self._index(k)
        where = f"frame {k} (file position {self.file_positions[k]})"
        try:
            frame = self._load(k)
        except FrameError:
            raise
        except ValueError as error:
            raise FrameError(f"{where}: {error}") from error
        if not isinstance(frame, Frame):
            raise TypeError(f"{type(self).__name__}._load returned "
                            f"{type(frame).__name__}, not a Frame")
        if frame.n_atoms != self.n_atoms:
            raise FrameError(f"{where} has {frame.n_atoms} atoms; the model has "
                             f"{self.n_atoms}")
        expected = int(self.timesteps[k])
        if expected != NO_TIMESTEP and frame.timestep is not None \
                and frame.timestep != expected:
            raise FrameError(f"{where} has timestep {frame.timestep}; the index "
                             f"of the file gives {expected}")
        if self._reference is None:
            self._reference = (frame.atom_id, frame.elements, k)
        else:
            ids, symbols, first = self._reference
            if not np.array_equal(frame.atom_id, ids):
                raise FrameError(f"{where} holds different atom ids from frame "
                                 f"{first}")
            differ = np.flatnonzero(frame.elements != symbols)
            if differ.size:
                raise FrameError(
                    f"{where}: {differ.size} atom(s) carry a different element "
                    f"from frame {first} (first atom id "
                    f"{int(frame.atom_id[differ[0]])}: "
                    f"{symbols[differ[0]]} -> {frame.elements[differ[0]]})")
        return frame

    def __getitem__(self, k: int) -> Frame:
        return self.frame(k)

    def frame_indices(self, start: int = 0, stop: int | None = None,
                      step: int = 1) -> range:
        """The readable indices ``iter_frames`` would visit, as a range."""
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) \
                or step < 1:
            raise ValueError(f"step {step!r}: a whole number of 1 or more is "
                             "needed")
        return range(self.n_frames)[slice(start, stop, int(step))]

    def iter_frames(self, start: int = 0, stop: int | None = None,
                    step: int = 1) -> Iterator[Frame]:
        """One frame at a time, as ``range(n_frames)[start:stop:step]``.

        A frame that cannot be read raises :class:`FrameError` and ends the
        iteration. A caller that wants to go on past it loops over
        :meth:`frame_indices` and calls :meth:`frame` itself, recording the
        index and the message.
        """
        for k in self.frame_indices(start, stop, step):
            yield self.frame(k)

    def __iter__(self) -> Iterator[Frame]:
        return self.iter_frames()

    def _index(self, k: int) -> int:
        if isinstance(k, (bool, np.bool_)) or \
                not isinstance(k, (int, np.integer)):
            raise TypeError(f"frame index {k!r} is not an integer")
        k = int(k)
        if not -self.n_frames <= k < self.n_frames:
            raise IndexError(f"frame {k} is outside 0 .. {self.n_frames - 1}")
        return k % self.n_frames

    # -- the load summary -------------------------------------------------------
    def describe(self) -> list[str]:
        """What was read and what was assumed, one sentence per line.

        The resolved path, the format, n_atoms, n_frames and what was skipped,
        the timesteps and times, frame 0's box and composition, the type map
        and its source, charges, the unit assumption, then every note. Loads
        frame 0 for the box and the composition.
        """
        lines = [f"source: {self.source_path}",
                 f"format: {self.file_format}",
                 f"{self.n_atoms} atoms per frame; {self.n_frames} readable "
                 f"frame(s)"]
        if self.skipped:
            lines.append(
                f"{len(self.skipped)} frame(s) skipped: " + "; ".join(
                    f"position {p}: {r}" for p, r in sorted(self.skipped.items())))
        known = self.timesteps[self.timesteps != NO_TIMESTEP]
        if known.size:
            lines.append(f"timesteps {int(known.min())} to {int(known.max())}"
                         + ("" if known.size == self.n_frames else
                            f" ({self.n_frames - known.size} frame(s) without "
                            "one)"))
        else:
            lines.append("no timesteps in the file")
        if self.times_ps is not None and np.isfinite(self.times_ps).any():
            times = self.times_ps[np.isfinite(self.times_ps)]
            lines.append(f"time {times.min():.6g} to {times.max():.6g} ps")
        else:
            lines.append("no frame times in the file")
        try:
            first = self.frame(0)
        except (FrameError, OSError) as error:
            lines.append(f"frame 0 could not be loaded: {error}")
        else:
            cell = first.cell()
            lines.append(
                f"box (frame 0): a {cell.a:.6g}, b {cell.b:.6g}, c {cell.c:.6g} Å; "
                f"alpha {cell.alpha:.6g}, beta {cell.beta:.6g}, gamma "
                f"{cell.gamma:.6g} deg; volume {first.volume_ang3:.6g} Å^3; "
                f"origin ({', '.join(f'{x:.6g}' for x in first.origin_ang)}) Å")
            lines.append(f"number density (frame 0) "
                         f"{first.number_density_per_ang3:.6g} atoms/Å^3")
            lines.append("composition (frame 0): " + ", ".join(
                f"{el} {count}" for el, count in first.composition.items()))
        if self.box_varies is True:
            lines.append("the box changes between frames")
        elif self.box_varies is False:
            lines.append("the box is the same in every frame")
        if self.type_map:
            lines.append(f"type map ({self.type_map_source}): " + ", ".join(
                f"{key} -> {value}" for key, value in self.type_map.items()))
        else:
            lines.append(f"elements from: {self.type_map_source}")
        if self.charges_e is not None:
            lines.append("charges per element: " + ", ".join(
                f"{el} {q:+.6g} e" for el, q in self.charges_e.items()))
        if self.units_note:
            lines.append(self.units_note)
        lines.extend(self.notes)
        return lines

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} {self.file_format} "
                f"{Path(self.source_path).name!r}: {self.n_atoms} atoms x "
                f"{self.n_frames} frames>")


class MemoryTrajectory(Trajectory):
    """Frames already in memory: synthetic models, tests, a frame list built
    by a tool. Every frame must hold the same atoms (ids and elements)."""

    def __init__(self, frames: Sequence[Frame], *,
                 source_path: str = "<memory>", file_format: str = "memory",
                 type_map: Mapping[int | str, str] | None = None,
                 type_map_source: str = "in memory",
                 notes: Sequence[str] | None = None,
                 units_note: str = "") -> None:
        frames = list(frames)
        if not frames:
            raise ValueError("a trajectory needs at least one frame")
        for index, frame in enumerate(frames):
            if not isinstance(frame, Frame):
                raise ValueError(f"item {index} is {type(frame).__name__}, "
                                 "not a Frame")
        first = frames[0]
        for index, frame in enumerate(frames[1:], start=1):
            if frame.n_atoms != first.n_atoms \
                    or not np.array_equal(frame.atom_id, first.atom_id) \
                    or not np.array_equal(frame.elements, first.elements):
                raise ValueError(
                    f"frame {index} holds different atoms (count, ids or "
                    "elements) from frame 0")
        charges, charge_notes = (None, [])
        if first.charge_e is not None:
            charges, charge_notes = charges_per_element(first.elements,
                                                        first.charge_e)
        times = [f.time_ps for f in frames]
        super().__init__(
            source_path=source_path, file_format=file_format,
            n_atoms=first.n_atoms, n_frames=len(frames),
            type_map_source=type_map_source,
            timesteps=[NO_TIMESTEP if f.timestep is None else f.timestep
                       for f in frames],
            times_ps=None if all(t is None for t in times) else
            [np.nan if t is None else t for t in times],
            type_map=type_map, charges_e=charges,
            notes=list(notes or []) + charge_notes, units_note=units_note,
            box_varies=any(not np.array_equal(f.box_ang, first.box_ang)
                           or not np.array_equal(f.origin_ang, first.origin_ang)
                           for f in frames[1:]))
        self._frames = frames

    def _load(self, k: int) -> Frame:
        return self._frames[k]


# ---------------------------------------------------------------------------
# oxidation states as model inputs
# ---------------------------------------------------------------------------

def _ox_label(symbol: str, ox: int) -> str:
    if ox == 0:
        return f"{symbol}0"
    return f"{symbol}{abs(ox)}{'+' if ox > 0 else '-'}"


@dataclass(frozen=True)
class ModelOxidation:
    """Formal oxidation states as model inputs; never resolved from geometry.

    ``ox`` and ``source`` are keyed by element symbol, in alphabetical order;
    ``source`` is ``'common'`` (``elements.COMMON_OX``) or ``'user'``. Treat the
    dicts as read-only: build a new one with :func:`model_oxidation` to change
    a state.
    """

    ox: dict[str, int]
    source: dict[str, str]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if set(self.ox) != set(self.source):
            raise ValueError("ox and source need the same elements")
        for symbol, origin in self.source.items():
            if origin not in ("common", "user"):
                raise ValueError(f"the source of {symbol}'s state is "
                                 f"{origin!r}; 'common' or 'user' is expected")

    def __hash__(self) -> int:
        # The generated hash would hash the dicts and fail; hash their items.
        return hash((tuple(self.ox.items()), tuple(self.source.items()),
                     self.notes))

    def per_atom(self, elements) -> np.ndarray:
        """(N,) int64 states, one per atom, for the bulk engine."""
        symbols = np.asarray(elements)
        if symbols.ndim != 1:
            raise ValueError("per_atom needs a one-dimensional array of symbols")
        if symbols.size == 0:
            return np.zeros(0, dtype=np.int64)
        tokens, inverse = np.unique(symbols, return_inverse=True)
        missing = [str(t) for t in tokens if str(t) not in self.ox]
        if missing:
            raise ValueError(
                f"no oxidation state for {', '.join(missing)} in this model "
                "oxidation map; model_oxidation over every element of the "
                "model provides one")
        values = np.array([self.ox[str(t)] for t in tokens], dtype=np.int64)
        return values[inverse.reshape(-1)]

    def describe(self) -> str:
        """``'O2- (common), Si4+ (common)'``: the states and their sources."""
        return ", ".join(f"{_ox_label(s, v)} ({self.source[s]})"
                         for s, v in self.ox.items())


def model_oxidation(elements_present: Iterable[str],
                    overrides: Mapping[str, int] | None = None
                    ) -> ModelOxidation:
    """``elements.COMMON_OX`` overlaid by the user's map, with each source.

    ValueError naming every element present that has neither a common state
    nor a user one: no state is invented (TODO: need reference). Notes are
    added for:

    * every model: the states are inputs, and ``oxidation.resolve`` is not run;
    * an element whose state is 0, which is neither cation nor anion in the
      bond-valence split;
    * an element taking a negative default state in a model that also holds O
      or F (Te, Se, S in a tellurite, selenite or sulfate glass): it is then an
      anion beside them, and its contacts to them carry no bond valence;
    * a user state for an element the model does not hold (not used).
    """
    if isinstance(elements_present, str):
        raise ValueError("elements_present needs a collection of symbols, not "
                         "one string")
    present = sorted({validate_symbol(str(e)) for e in elements_present})
    if not present:
        raise ValueError("no elements given")

    user: dict[str, int] = {}
    for key, value in (overrides or {}).items():
        symbol = validate_symbol(str(key))
        if isinstance(value, (bool, np.bool_)) or \
                not isinstance(value, (int, np.integer)):
            raise ValueError(f"the oxidation state given for {symbol} is "
                             f"{value!r}; a whole number is needed")
        if symbol in user and user[symbol] != int(value):
            raise ValueError(f"two different states are given for {symbol} "
                             f"({user[symbol]} and {int(value)})")
        user[symbol] = int(value)

    ox: dict[str, int] = {}
    source: dict[str, str] = {}
    missing = []
    for symbol in present:
        if symbol in user:
            ox[symbol], source[symbol] = user[symbol], "user"
        elif symbol in element_data.COMMON_OX:
            ox[symbol], source[symbol] = element_data.COMMON_OX[symbol], "common"
        else:
            missing.append(symbol)
    if missing:
        raise ValueError(
            f"no oxidation state for {', '.join(missing)}: not in "
            "elements.COMMON_OX and not given in the oxidation-state map "
            "(TODO: need reference); give a state for each")

    result = ModelOxidation(ox, source)
    notes = [
        "oxidation states are model inputs, not resolved from the geometry "
        "(oxidation.resolve is not run on MD frames): " + result.describe()]
    for symbol in present:
        if ox[symbol] == 0:
            notes.append(f"{symbol} has oxidation state 0, so it is neither a "
                         "cation nor an anion in the bond-valence split")
    beside = [a for a in ("O", "F") if a in ox]
    for symbol in present:
        if beside and symbol not in ("O", "F") and source[symbol] == "common" \
                and ox[symbol] < 0:
            others = " and ".join(beside)
            notes.append(
                f"{symbol} takes its default state {_ox_label(symbol, ox[symbol])} "
                f"from elements.COMMON_OX, so it counts as an anion beside "
                f"{others}; its contacts to {others} are anion-anion contacts "
                f"and carry no bond valence. A cationic state for {symbol} can "
                "be given in the oxidation-state map")
    for symbol in sorted(set(user) - set(present)):
        notes.append(f"the state given for {symbol} "
                     f"({_ox_label(symbol, user[symbol])}) is not used: the "
                     f"model holds no {symbol}")
    return ModelOxidation(ox, source, tuple(notes))


# ---------------------------------------------------------------------------
# bridges to and from the crystal model
# ---------------------------------------------------------------------------

def frame_to_structure(frame: Frame, ox: ModelOxidation, *,
                       name: str | None = None, source_path: str | None = None,
                       symmetry_max_atoms: int = SPGLIB_MAX_ATOMS) -> Structure:
    """One frame as a crystal-model Structure, for the tools that need one.

    For the 3D view and the per-atom panels on one frame, and for comparing the
    bulk engine with ``coordination.analyse_site`` on small frames. One Site
    (multiplicity 1, label element + atom id) and one Atom per atom, built
    without ``readers._assemble``, which would run spglib unconditionally and
    replace the oxidation states with ``'common'`` ones.

    * Each Site carries the model's state, ``ox_source`` ``'common'`` or
      ``'user'``. Callers pass ``resolve_oxidation=False`` to
      ``analyse_structure``, or the crystal resolver replaces them.
    * ``Atom.cart = Cell.to_cartesian(frac)``. ``Cell`` has no origin, so the
      positions are relative to the box corner; a note says so when the frame's
      origin is not zero.
    * ``spacegroup_hm`` is None. ``readers._annotate`` (spglib) runs only at or
      below ``symmetry_max_atoms`` atoms; above it ``spacegroup_number`` stays
      None and the note :data:`NOTE_SYMMETRY_NOT_SEARCHED` is added.
    * Elements that ``elements.normalise`` turns into another element (He, Ne,
      Kr, ...; :func:`symbols_changed_by_normalise`) are refused, because the
      Site would silently become that other element.
    """
    from . import readers

    if not isinstance(frame, Frame):
        raise ValueError(f"frame_to_structure needs a Frame, not "
                         f"{type(frame).__name__}")
    if not isinstance(ox, ModelOxidation):
        raise ValueError(f"frame_to_structure needs a ModelOxidation, not "
                         f"{type(ox).__name__}")
    if isinstance(symmetry_max_atoms, bool) or \
            not isinstance(symmetry_max_atoms, (int, np.integer)) or \
            symmetry_max_atoms < 0:
        raise ValueError(f"symmetry_max_atoms {symmetry_max_atoms!r}: a whole "
                         "number of 0 or more is needed")
    changed = symbols_changed_by_normalise(frame.species)
    if changed:
        raise ValueError(
            "the crystal model reduces " + ", ".join(
                f"{s} to {r}" for s, r in changed.items())
            + " (elements.normalise, called by Site and by the bond-valence "
            "lookup), so a Structure built from this frame would hold another "
            "element; it is refused")
    ox_atom = ox.per_atom(frame.elements)

    n = frame.n_atoms
    cell = readers.cell_from_vectors(frame.box_ang)
    cart = cell.to_cartesian(frame.frac)
    sites: list[Site] = []
    atoms: list[Atom] = []
    for i in range(n):
        symbol = str(frame.elements[i])
        label = f"{symbol}{int(frame.atom_id[i])}"
        frac = np.array(frame.frac[i], dtype=np.float64)
        sites.append(Site(label, symbol, frac.copy(), 1.0, multiplicity=1,
                          ox=int(ox_atom[i]), ox_source=ox.source[symbol]))
        atoms.append(Atom(element=symbol, frac=frac,
                          cart=np.array(cart[i], dtype=np.float64),
                          site_index=i, label=label, occupancy=1.0))

    if name is None:
        name = f"MD frame ({n} atoms" + (
            "" if frame.timestep is None else f", timestep {frame.timestep}") + ")"
    structure = Structure(name=name, cell=cell, sites=sites, atoms=atoms,
                          spacegroup_hm=None, spacegroup_number=None,
                          source_path=source_path)
    structure.notes.extend(frame.notes)
    structure.notes.extend(ox.notes)
    structure.notes.append(
        f"built from an MD frame of {n} atoms: one site per atom "
        "(multiplicity 1, labelled element + atom id); oxidation states are "
        "the model's inputs, not resolved from the geometry")
    if np.any(frame.origin_ang):
        structure.notes.append(
            "the box origin (" + ", ".join(f"{x:.6g}" for x in frame.origin_ang)
            + ") Å is dropped: the crystal model's Cell has none, so positions "
            "here are relative to the box corner")
    left_out = [label for label, value in (
        ("unwrapped positions", frame.unwrapped_cart_ang),
        ("velocities", frame.vel_ang_per_ps),
        ("per-atom charges", frame.charge_e)) if value is not None]
    if left_out:
        structure.notes.append(
            "not carried into the Structure (it has no place for them): "
            + ", ".join(left_out))
    if n <= symmetry_max_atoms:
        readers._annotate(structure)
    else:
        structure.notes.append(NOTE_SYMMETRY_NOT_SEARCHED)
        structure.notes.append(
            f"the spglib search runs only up to {symmetry_max_atoms} atoms "
            f"(symmetry_max_atoms); this frame has {n}")
    return structure


def supercell_frame(structure: Structure, reps: tuple[int, int, int] = (2, 2, 2)
                    ) -> tuple[Frame, np.ndarray]:
    """``structure.atoms`` tiled ``reps`` times along a, b, c, as one Frame.

    The tiles are the structure's existing P1 expansion, not a new one: the
    crystal-as-glass equality compares each row with the very atom it came
    from. Returns the frame and ``parent_index`` ((N,) int64, read-only): row
    ``r`` is ``structure.atoms[parent_index[r]]`` translated by a whole number
    of cells. Rows run image by image in ``itertools.product`` order (the a
    index slowest), the parent atoms in order within each image, so
    ``parent_index[r] == r % len(structure.atoms)``; ``atom_id`` is the row.

    The supercell box is ``Cell.orth.T`` with each row multiplied by its
    repeat, and the origin is zero. ValueError for an empty structure, a
    repeat that is not a whole number of 1 or more, any atom with occupancy
    other than 1 (a partly present atom cannot be tiled into a model), or an
    atom outside the home cell.
    """
    if len(reps) != 3 or any(isinstance(r, (bool, np.bool_))
                             or not isinstance(r, (int, np.integer)) or r < 1
                             for r in reps):
        raise ValueError(f"reps {tuple(reps)!r}: three whole numbers of 1 or "
                         "more are needed")
    counts = np.array([int(r) for r in reps], dtype=np.int64)
    atoms = structure.atoms
    if not atoms:
        raise ValueError(f"{structure.name!r} has no atoms to tile")
    partial = [a.label for a in atoms if a.occupancy != 1.0]
    if partial:
        raise ValueError(
            f"{len(partial)} of {len(atoms)} atoms of {structure.name!r} have "
            f"occupancy other than 1 (first: {partial[:5]}); a supercell model "
            "needs every atom fully present")
    away = [a.label for a in atoms if tuple(a.image) != (0, 0, 0)]
    if away:
        raise ValueError(
            f"{len(away)} atoms of {structure.name!r} lie outside the home cell "
            f"(first: {away[:5]}); only the home-cell expansion can be tiled")

    n_parent = len(atoms)
    symbols = np.array([validate_symbol(a.element) for a in atoms], dtype="<U2")
    frac0 = wrap_frac(np.array([a.frac for a in atoms], dtype=np.float64))
    rows = np.asarray(structure.cell.orth, dtype=np.float64).T
    shifts = np.array(list(itertools.product(*(range(r) for r in counts))),
                      dtype=np.float64)
    frac = ((frac0[None, :, :] + shifts[:, None, :]) / counts).reshape(-1, 3)
    box = rows * counts[:, None]
    n_rows = n_parent * len(shifts)
    parent = np.tile(np.arange(n_parent, dtype=np.int64), len(shifts))
    parent.setflags(write=False)
    frame = frame_from_arrays(
        np.tile(symbols, len(shifts)), frac=frac, box_ang=box,
        atom_id=np.arange(n_rows, dtype=np.int64),
        notes=(f"{counts[0]}x{counts[1]}x{counts[2]} supercell of "
               f"{structure.name!r}: {n_parent} atoms in its expanded cell, "
               f"{n_rows} here; row r comes from "
               "structure.atoms[parent_index[r]]",))
    return frame, parent
