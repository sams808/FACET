"""The MD frame model: the contract the readers and the bulk engine share.

Every MD file FACET reads becomes a :class:`facet.core.md_model.Frame`, and the
bulk engine measures nothing but Frames, so a mistake here is a mistake
everywhere downstream, and a quiet one. What is pinned, and why:

* **The box convention.** Rows a, b, c, with ``Cell.orth == box_ang.T``. Getting
  that transposed gives a box that still has the right volume and lengths on
  an orthogonal cell and puts every atom somewhere else on a tilted one, so it
  is tested on a strongly tilted box, against the crystal model's own Cell.
* **Exact wrapping.** ``(-1e-17) % 1.0`` is 1.0; a cKDTree with ``boxsize``
  refuses such a coordinate. The edge cases are tested value by value.
* **Symbols validated, never normalised.** ``elements.normalise`` turns 'Ob'
  into O and 'He' into H; on an MD file that relabels a model silently.
* **Every refusal says why.** Physically impossible input (non-finite
  positions, a degenerate box, duplicate ids) raises a ValueError with a
  message, never a numpy traceback.
* **The two bridges.** ``frame_to_structure`` has to give a Structure the
  crystal analysis runs on without resolving oxidation states, and spglib must
  not run above the threshold. ``supercell_frame`` has to be a pure tiling of
  the existing expansion, because the crystal-as-glass equality test compares
  each tile with the atom it came from.
* **No verdicts.** Notes and messages report measurements and provenance.

Every structure here is synthetic and built in the test. The lengths in them
are test inputs, not reference values for any material.
"""
from __future__ import annotations

import ast
import copy
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from facet.core import coordination, elements, md_model, readers
from facet.core.md_model import (Frame, FrameError, MemoryTrajectory,
                                 Trajectory, frame_from_arrays,
                                 frame_to_structure, model_oxidation,
                                 supercell_frame, validate_symbol, wrap_frac)
from facet.core.structure import Atom, Site, Structure

ROOT = Path(__file__).resolve().parent.parent

# A strongly tilted box: b leans 9 Å along a on a 4 Å height, c leans back.
# |det| / (|a||b||c|) is about 0.18, far from rectangular and far from the
# degenerate threshold.
TILTED = np.array([[10.0, 0.0, 0.0],
                   [9.0, 4.0, 0.0],
                   [-7.0, 3.5, 5.0]])
ORIGIN = np.array([-3.25, 1.5, 0.75])


def _rock_salt(edge_ang: float = 5.6, origin=None) -> Frame:
    """Eight atoms of the rock-salt arrangement, built from fractions.

    The edge is a test input: it puts Na-Cl at half of it (2.8 Å) and the next
    Na-Cl shell at sqrt(3) times that. It is not a reference lattice constant.
    """
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    frac = np.vstack([fcc, (fcc + 0.5) % 1.0])
    return frame_from_arrays(["Na"] * 4 + ["Cl"] * 4, frac=frac,
                             box_ang=np.eye(3) * edge_ang, origin_ang=origin)


# ---------------------------------------------------------------------------
# the box convention
# ---------------------------------------------------------------------------

def test_the_box_rows_are_the_transpose_of_the_crystal_cell():
    """Cell.orth holds a, b, c as columns; a Frame holds them as rows.

    readers.cell_from_vectors builds orth with column_stack and to_cartesian is
    orth @ f. If that ever changes, every MD frame bridged to the crystal model
    would be transposed; this is where it shows.
    """
    cell = readers.cell_from_vectors(TILTED)
    assert np.array_equal(cell.orth, TILTED.T)
    frac = np.random.default_rng(3).random((20, 3))
    assert np.allclose(cell.to_cartesian(frac), frac @ TILTED, rtol=0,
                       atol=1e-12)
    frame = frame_from_arrays(["Si"] * 20, frac=frac, box_ang=TILTED,
                              origin_ang=ORIGIN)
    assert np.array_equal(frame.cell().orth, TILTED.T)


def test_cartesian_and_fractional_round_trip_on_a_tilted_box():
    """Positions in, positions out, to 1e-12, from either side and from
    positions several cells away.

    The input is placed up to three cells outside the box: an MD file's
    unwrapped or lagging coordinates look like that. The stored frac must be the
    same atom (equal modulo 1), and cart_ang must be origin + frac @ box.
    """
    rng = np.random.default_rng(2026)
    n = 200
    frac = rng.random((n, 3))
    symbols = rng.choice(["Si", "O", "Na"], size=n)
    cart = ORIGIN + frac @ TILTED

    from_cart = frame_from_arrays(symbols, cart, box_ang=TILTED,
                                  origin_ang=ORIGIN)
    gap = from_cart.frac - frac
    gap -= np.round(gap)
    assert np.abs(gap).max() <= 1e-12
    assert np.abs(from_cart.cart_ang - cart).max() <= 1e-12

    from_frac = frame_from_arrays(symbols, frac=frac, box_ang=TILTED,
                                  origin_ang=ORIGIN)
    assert np.abs(from_frac.cart_ang - cart).max() <= 1e-12

    shifts = rng.integers(-3, 4, size=(n, 3))
    far = frame_from_arrays(symbols, cart + shifts @ TILTED, box_ang=TILTED,
                            origin_ang=ORIGIN)
    gap = far.frac - frac
    gap -= np.round(gap)
    assert np.abs(gap).max() <= 1e-12
    assert np.abs(far.cart_ang - cart).max() <= 1e-12
    assert any("were wrapped into it" in note for note in far.notes)

    # the stored pair is consistent by construction, not by tolerance
    assert np.array_equal(far.cart_ang, far.origin_ang + far.frac @ far.box_ang)


def test_the_perpendicular_widths_come_from_the_reciprocal_lattice_too():
    """V/|b x c| (the neighbour search's formula) against 1/|a*|.

    Two derivations of one number: the plane spacing of the bc faces is the
    inverse length of the reciprocal vector a*, which is a column of
    inv(box_ang). Half the smallest width bounds the minimum-image radius.
    """
    frame = frame_from_arrays(["O"], frac=[[0.1, 0.2, 0.3]], box_ang=TILTED)
    reciprocal = np.linalg.inv(TILTED)          # columns are a*, b*, c*
    expected = 1.0 / np.linalg.norm(reciprocal, axis=0)
    assert np.allclose(frame.perpendicular_widths_ang, expected, rtol=1e-13,
                       atol=0)


def test_volume_density_and_composition():
    frame = frame_from_arrays(["Si", "O", "O", "Na", "O"],
                              frac=np.random.default_rng(1).random((5, 3)),
                              box_ang=TILTED)
    a, b, c = TILTED
    assert frame.volume_ang3 == pytest.approx(abs(np.dot(a, np.cross(b, c))),
                                              rel=1e-14)
    assert frame.number_density_per_ang3 == pytest.approx(
        5 / frame.volume_ang3, rel=1e-15)
    assert frame.composition == {"Na": 1, "O": 3, "Si": 1}
    assert frame.species == ("Na", "O", "Si")
    assert frame.n_atoms == 5
    assert not frame.box_is_diagonal
    assert _rock_salt().box_is_diagonal


# ---------------------------------------------------------------------------
# wrapping
# ---------------------------------------------------------------------------

def test_the_rounding_that_makes_the_guard_necessary():
    """Why wrap_frac does more than % 1.0: a tiny negative wraps to 1.0."""
    assert (-1e-17) % 1.0 == 1.0
    assert -1e-17 - np.floor(-1e-17) == 1.0


@pytest.mark.parametrize("value,expected", [
    (-1e-17, 0.0),                       # rounds to 1.0, which is 0.0
    (1.0, 0.0),
    (2.0000000001, 2.0000000001 - 2.0),  # about 1e-10, not 0 and not 1
    (-0.0, 0.0),
    (0.5, 0.5),
    (-0.25, 0.75),
    (float(np.nextafter(1.0, 0.0)), float(np.nextafter(1.0, 0.0))),
    (-3.0, 0.0),
])
def test_wrap_edge_cases(value, expected):
    wrapped = wrap_frac(np.array([value]))
    assert wrapped[0] == expected
    assert 0.0 <= wrapped[0] < 1.0
    assert not np.signbit(wrapped[0])


def test_a_frame_stores_the_wrapped_edge_values():
    frame = frame_from_arrays(["O"], frac=[[-1e-17, 1.0, 2.0000000001]],
                              box_ang=np.eye(3) * 10.0)
    assert frame.frac[0, 0] == 0.0
    assert frame.frac[0, 1] == 0.0
    assert frame.frac[0, 2] == 2.0000000001 - 2.0
    assert (frame.frac < 1.0).all()


def test_wrap_refuses_non_finite_values():
    with pytest.raises(ValueError, match="not finite"):
        wrap_frac([0.2, np.nan])


# ---------------------------------------------------------------------------
# element symbols
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token,symbol", [
    ("Si", "Si"), ("si", "Si"), ("SI", "Si"), ("O", "O"), ("o", "O"),
    ("Na+", "Na"), ("O2-", "O"), ("Fe3+", "Fe"), ("Na+1", "Na"), ("Cl-", "Cl"),
    ("Bi", "Bi"), (" Al ", "Al"),
    # real symbols that elements.normalise would turn into another element
    ("He", "He"), ("Ne", "Ne"), ("Kr", "Kr"), ("Og", "Og"),
])
def test_symbols_that_pass(token, symbol):
    assert validate_symbol(token) == symbol


@pytest.mark.parametrize("token", [
    "1", "type1", "Si1", "Ob", "OB", "O_b", "X", "D", "Xx", "Uuo", "", "  ",
    "Si 1", "Na++", "2+",
])
def test_symbols_that_never_pass(token):
    with pytest.raises(ValueError, match="element symbol"):
        validate_symbol(token)


@pytest.mark.parametrize("token", [1, 8.0, None, b"Si"])
def test_non_text_never_passes(token):
    with pytest.raises(ValueError, match="element symbol"):
        validate_symbol(token)


def test_why_normalise_alone_is_not_enough():
    """The crystal-side reduction, on the tokens an MD file can hold.

    Pinned so that if normalise changes, the reason for validate_symbol is
    re-read rather than assumed.
    """
    assert elements.normalise("1") == ""
    assert elements.normalise("type1") == "Type"
    assert elements.normalise("Ob") == "O"
    assert elements.normalise("He") == "H"
    assert elements.normalise("Kr") == "K"


def test_symbols_the_crystal_model_would_change():
    assert md_model.symbols_changed_by_normalise(["He", "Si", "Kr", "O"]) \
        == {"He": "H", "Kr": "K"}
    common = ["O", "F", "Cl", "Si", "Al", "B", "P", "Na", "K", "Ca", "Mg",
              "Bi", "Pb", "Ge", "Ti", "Zr", "Li", "Ba", "Sr", "Zn"]
    assert md_model.symbols_changed_by_normalise(common) == {}


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def _valid() -> dict:
    return dict(elements=["Si", "O", "O"],
                cart_ang=np.array([[0.0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]]),
                box_ang=np.eye(3) * 10.0)


def _with(**changes) -> dict:
    out = _valid()
    out.update(changes)
    return out


@pytest.mark.parametrize("arguments,message", [
    (_with(cart_ang=np.array([[0.0, 0, 0], [np.nan, 0, 0], [0, 1.6, 0]])),
     "non-finite"),
    (_with(cart_ang=np.array([[0.0, 0, 0], [np.inf, 0, 0], [0, 1.6, 0]])),
     "non-finite"),
    (_with(box_ang=np.array([[10.0, 0, 0], [0, 10, 0], [10, 10, 0]])),
     "degenerate"),
    (_with(box_ang=np.array([[10.0, 0, 0], [0, 10, 0], [10, 10, 1e-7]])),
     "degenerate"),
    (_with(box_ang=np.array([[10.0, 0, 0], [0, 0, 0], [0, 0, 10]])),
     "zero length"),
    (_with(box_ang=np.eye(3)[:2] * 10), "shape"),
    (_with(box_ang=np.array([[10.0, 0, 0], [0, np.nan, 0], [0, 0, 10]])),
     "not finite"),
    (_with(elements=["Si", "O"]), "shape"),
    (_with(elements=[14, 8, 8]), "type map"),
    (_with(elements=["Si", "Ob", "O"]), "element symbol"),
    (_with(elements=[], cart_ang=np.zeros((0, 3))), "no atoms"),
    (_with(atom_id=[1, 2]), "atom ids for 3 atoms"),
    (_with(atom_id=[4, 7, 4]), "more than once"),
    (_with(atom_id=[1.0, 2.5, 3.0]), "whole numbers"),
    (_with(frac=np.zeros((3, 3))), "exactly one"),
    (_with(cart_ang=None), "exactly one"),
    (_with(vel_ang_per_ps=np.zeros((3, 2))), "vel_ang_per_ps has shape"),
    (_with(charge_e=np.array([1.0, np.nan, -1.0])), "not finite"),
    (_with(unwrapped_cart_ang=np.zeros((2, 3))), "unwrapped_cart_ang has shape"),
    (_with(timestep=1.5), "not an integer"),
    (_with(timestep=True), "not an integer"),
    (_with(time_ps=float("inf")), "not finite"),
    (_with(notes="one string"), "sequence"),
    (_with(origin_ang=[0.0, 0.0]), "origin_ang has shape"),
])
def test_each_refusal_says_why(arguments, message):
    elements_ = arguments.pop("elements")
    with pytest.raises(ValueError, match=message):
        frame_from_arrays(elements_, **arguments)


def test_a_directly_built_frame_is_held_to_the_same_invariants():
    built = _rock_salt()
    fields = dict(elements=built.elements, frac=built.frac,
                  cart_ang=built.cart_ang, box_ang=built.box_ang,
                  origin_ang=built.origin_ang, atom_id=built.atom_id)
    Frame(**fields)                                       # passes as built
    with pytest.raises(ValueError, match=r"outside \[0, 1\)"):
        Frame(**{**fields, "frac": built.frac + 1.0})
    with pytest.raises(ValueError, match="differs"):
        Frame(**{**fields, "cart_ang": built.cart_ang + 0.01})
    with pytest.raises(ValueError, match="strictly increasing"):
        Frame(**{**fields, "atom_id": built.atom_id[::-1].copy()})
    with pytest.raises(ValueError, match="validate_symbol"):
        Frame(**{**fields, "elements": np.array(["na"] * 4 + ["Cl"] * 4)})


# ---------------------------------------------------------------------------
# ordering, immutability, copies
# ---------------------------------------------------------------------------

def test_rows_are_sorted_by_atom_id_and_every_array_moves_with_them():
    cart = np.array([[1.0, 1, 1], [2.0, 2, 2], [3.0, 3, 3]])
    frame = frame_from_arrays(
        ["Si", "O", "Na"], cart, box_ang=np.eye(3) * 10.0, atom_id=[30, 10, 20],
        vel_ang_per_ps=cart * 10, charge_e=[2.4, -1.2, 0.6],
        unwrapped_cart_ang=cart + 10.0, timestep=500, time_ps=0.5)
    assert frame.atom_id.tolist() == [10, 20, 30]
    assert frame.elements.tolist() == ["O", "Na", "Si"]
    expected = cart[[1, 2, 0]]
    assert np.allclose(frame.cart_ang, expected, rtol=0, atol=1e-14)
    assert np.array_equal(frame.vel_ang_per_ps, expected * 10)
    assert np.array_equal(frame.unwrapped_cart_ang, expected + 10.0)
    assert frame.charge_e.tolist() == [-1.2, 0.6, 2.4]
    assert frame.timestep == 500 and frame.time_ps == 0.5
    assert any("sorted by atom id" in note for note in frame.notes)


def test_ids_default_to_the_input_order():
    frame = frame_from_arrays(["Si", "O"], frac=[[0.1, 0, 0], [0.2, 0, 0]],
                              box_ang=np.eye(3))
    assert frame.atom_id.tolist() == [0, 1]
    assert frame.atom_id.dtype == np.int64
    assert frame.elements.dtype == np.dtype("<U2")


def test_the_frame_is_read_only_and_the_callers_arrays_are_not():
    """Frozen frame, frozen arrays; the caller's own inputs untouched.

    np.asarray hands a float64 input straight through, so freezing it in place
    would have frozen the caller's array; the constructor copies first.
    """
    box = np.eye(3) * 10.0
    vel = np.ones((2, 3))
    frame = frame_from_arrays(["Si", "O"], frac=[[0.1, 0, 0], [0.2, 0, 0]],
                              box_ang=box, vel_ang_per_ps=vel)
    for name in ("elements", "frac", "cart_ang", "box_ang", "origin_ang",
                 "atom_id", "vel_ang_per_ps"):
        with pytest.raises(ValueError):
            getattr(frame, name)[0] = 0
    with pytest.raises(AttributeError):
        frame.timestep = 3
    assert box.flags.writeable and vel.flags.writeable
    box[0, 0] = 99.0                        # does not reach the frame
    assert frame.box_ang[0, 0] == 10.0


def test_pickle_and_copy_rebuild_a_frozen_checked_frame():
    """A copy goes through the constructor, so it is frozen like the original.

    Plain unpickling of a numpy array gives a writeable one; a frame sent to a
    worker process would have lost its read-only arrays without this.
    """
    frame = _rock_salt(origin=[1.0, 2.0, 3.0]).with_notes("from a test")
    for clone in (pickle.loads(pickle.dumps(frame)), copy.copy(frame),
                  copy.deepcopy(frame)):
        assert np.array_equal(clone.cart_ang, frame.cart_ang)
        assert not clone.cart_ang.flags.writeable
        assert clone.notes == frame.notes
    assert frame.notes[-1] == "from a test"


# ---------------------------------------------------------------------------
# helpers the readers share
# ---------------------------------------------------------------------------

def test_unwrapping_with_image_flags():
    cart = np.array([[1.0, 2.0, 3.0], [9.5, 0.5, 4.0]])
    image = np.array([[0, 0, 0], [-1, 2, 1]])
    out = md_model.unwrap_with_images(cart, image, TILTED)
    assert np.allclose(out[1], cart[1] - TILTED[0] + 2 * TILTED[1] + TILTED[2],
                       rtol=0, atol=1e-13)
    assert np.array_equal(out[0], cart[0])
    with pytest.raises(ValueError, match="whole numbers"):
        md_model.unwrap_with_images(cart, image + 0.5, TILTED)


def test_charges_per_element():
    charges, notes = md_model.charges_per_element(
        ["Si", "O", "O", "Na"], [2.4, -1.2, -1.2, 0.6])
    assert charges == {"Na": 0.6, "O": -1.2, "Si": 2.4} and notes == []
    charges, notes = md_model.charges_per_element(
        ["Si", "O", "O", "Na"], [2.4, -1.2, -1.1, 0.6])
    assert charges is None
    assert "O (-1.2 to -1.1 e)" in notes[0]


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------

def _moved(frame: Frame, shift_ang: float, timestep: int, box=None) -> Frame:
    return frame_from_arrays(frame.elements, frame.cart_ang + shift_ang,
                             box_ang=frame.box_ang if box is None else box,
                             atom_id=frame.atom_id, timestep=timestep,
                             time_ps=timestep * 0.001)


def test_a_memory_trajectory_indexes_iterates_and_describes():
    first = _moved(_rock_salt(), 0.0, 0)
    frames = [first, _moved(first, 0.1, 1000), _moved(first, 0.2, 2000)]
    traj = MemoryTrajectory(frames)

    assert len(traj) == 3 and traj.n_atoms == 8 and traj.n_frames == 3
    assert traj.frame(-1) is frames[2] and traj[1] is frames[1]
    assert traj.timesteps.tolist() == [0, 1000, 2000]
    assert traj.times_ps.tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert [f.timestep for f in traj.iter_frames(0, None, 2)] == [0, 2000]
    assert list(traj.frame_indices(1)) == [1, 2]
    assert traj.file_positions.tolist() == [0, 1, 2]
    assert traj.box_varies is False
    with pytest.raises(IndexError):
        traj.frame(3)
    with pytest.raises(ValueError, match="step"):
        traj.frame_indices(0, None, 0)

    text = "\n".join(traj.describe())
    for expected in ("source: <memory>", "format: memory",
                     "8 atoms per frame; 3 readable frame(s)",
                     "timesteps 0 to 2000", "time 0 to 2 ps",
                     "box (frame 0): a 5.6, b 5.6, c 5.6 Å",
                     "composition (frame 0): Cl 4, Na 4",
                     "the box is the same in every frame"):
        assert expected in text


def test_a_varying_box_is_reported():
    first = _rock_salt()
    traj = MemoryTrajectory([first, _moved(first, 0.0, 1,
                                           box=np.eye(3) * 5.7)])
    assert traj.box_varies is True
    assert "the box changes between frames" in traj.describe()


def test_frames_with_different_atoms_are_refused():
    first = _rock_salt()
    swapped = frame_from_arrays(first.elements[::-1].copy(), first.cart_ang,
                                box_ang=first.box_ang)
    with pytest.raises(ValueError, match="different atoms"):
        MemoryTrajectory([first, swapped])
    with pytest.raises(ValueError, match="at least one frame"):
        MemoryTrajectory([])


class _FromList(Trajectory):
    """A reader stand-in: frames from a list, one of them unreadable."""

    def __init__(self, frames, unreadable=(), **kw):
        kw.setdefault("type_map_source", "in memory")
        super().__init__(source_path="glass.dump", file_format="test",
                         n_atoms=8, n_frames=len(frames), **kw)
        self._frames, self._unreadable = frames, set(unreadable)

    def _load(self, k):
        if k in self._unreadable:
            raise ValueError("line 12 has 3 columns; the header names 5")
        return self._frames[k]


def test_a_frame_that_fails_to_load_raises_frame_error_with_its_position():
    first = _rock_salt()
    traj = _FromList([first, first], unreadable={1}, skipped={1: "truncated"})
    # file positions 0 and 2 are readable; 1 was skipped while indexing
    assert traj.file_positions.tolist() == [0, 2]
    assert traj.file_position(1) == 2
    with pytest.raises(FrameError, match=r"frame 1 \(file position 2\): line 12"):
        traj.frame(1)
    assert isinstance(FrameError("x"), ValueError)
    assert "1 frame(s) skipped: position 1: truncated" in traj.describe()


def test_a_frame_holding_other_atoms_raises_frame_error():
    first = _rock_salt()
    other = frame_from_arrays(["Na"] * 8, first.cart_ang, box_ang=first.box_ang)
    traj = _FromList([first, other])
    traj.frame(0)
    with pytest.raises(FrameError, match="different element"):
        traj.frame(1)
    fewer = frame_from_arrays(["Na"] * 4, first.cart_ang[:4],
                              box_ang=first.box_ang)
    with pytest.raises(FrameError, match="has 4 atoms"):
        _FromList([first, fewer]).frame(1)


def test_trajectory_bookkeeping_is_checked():
    with pytest.raises(ValueError, match="no readable frame.*truncated"):
        _FromList([], skipped={0: "truncated"})
    with pytest.raises(ValueError, match="outside"):
        _FromList([_rock_salt()], skipped={5: "truncated"})
    with pytest.raises(ValueError, match="type_map_source"):
        _FromList([_rock_salt()], type_map_source="guessed")
    with pytest.raises(ValueError, match="element symbol"):
        _FromList([_rock_salt()], type_map={1: "Ob"})
    traj = _FromList([_rock_salt()], type_map={1: "na", "Cl_x": "Cl"},
                     type_map_source="user + data-file masses",
                     units_note="length unit assumed Å")
    assert traj.type_map == {1: "Na", "Cl_x": "Cl"}
    lines = traj.describe()
    assert "type map (user + data-file masses): 1 -> Na, Cl_x -> Cl" in lines
    assert "length unit assumed Å" in lines
    assert traj.timesteps.tolist() == [md_model.NO_TIMESTEP]
    assert "no timesteps in the file" in lines


# ---------------------------------------------------------------------------
# oxidation states
# ---------------------------------------------------------------------------

def test_common_states_overlaid_by_the_users():
    ox = model_oxidation(["Bi", "O", "Na"], {"Bi": 5})
    assert ox.ox == {"Bi": 5, "Na": 1, "O": -2}
    assert ox.source == {"Bi": "user", "Na": "common", "O": "common"}
    assert ox.per_atom(np.array(["O", "Bi", "O", "Na"])).tolist() == [-2, 5, -2, 1]
    assert "oxidation.resolve is not run" in ox.notes[0]
    assert "Bi5+ (user)" in ox.notes[0]
    hash(ox)


def test_an_element_with_no_state_anywhere_is_refused():
    with pytest.raises(ValueError, match="He.*TODO: need reference"):
        model_oxidation(["He", "O", "Si"])
    assert model_oxidation(["He", "O"], {"He": 0}).ox["He"] == 0


def test_the_notes_on_states_that_need_a_second_look():
    tellurite = model_oxidation(["Te", "O", "Na"])
    assert any(note.startswith("Te takes its default state Te2-")
               for note in tellurite.notes)
    assert not any("Te takes" in note for note in
                   model_oxidation(["Te", "O"], {"Te": 4}).notes)
    # an oxyfluoride: O and F are the anions the rule refers to, never noted
    oxyfluoride = model_oxidation(["O", "F", "Al", "Si"])
    assert not any("takes its default" in note for note in oxyfluoride.notes)
    with_xenon = model_oxidation(["Xe", "O", "Si"])
    assert any(note.startswith("Xe has oxidation state 0")
               for note in with_xenon.notes)
    unused = model_oxidation(["O", "Si"], {"Bi": 5})
    assert any("Bi" in note and "not used" in note for note in unused.notes)
    assert "Bi" not in unused.ox


@pytest.mark.parametrize("overrides,message", [
    ({"Bi": 5.0}, "whole number"), ({"Bi": True}, "whole number"),
    ({"Na": 1, "Na+": 2}, "two different states"), ({"Ob": -2}, "element symbol"),
])
def test_override_refusals(overrides, message):
    with pytest.raises(ValueError, match=message):
        model_oxidation(["Bi", "Na", "O"], overrides)


def test_per_atom_refuses_an_element_it_has_no_state_for():
    with pytest.raises(ValueError, match="no oxidation state for Al"):
        model_oxidation(["O", "Si"]).per_atom(np.array(["Si", "Al", "O"]))


# ---------------------------------------------------------------------------
# frame_to_structure
# ---------------------------------------------------------------------------

def test_a_frame_becomes_a_structure_the_crystal_analysis_runs_on(monkeypatch):
    """One site per atom, the model's states kept, spglib run below the limit.

    spglib finding Fm-3m (225) on eight atoms placed from fractions is an
    outside check that the conversion kept the geometry. The analysis then runs
    with resolve_oxidation=False and leaves every ox_source as it was.
    """
    calls = []
    original = readers._annotate
    monkeypatch.setattr(readers, "_annotate",
                        lambda st: (calls.append(st.n_atoms), original(st)))
    frame = _rock_salt(origin=[2.0, -1.0, 0.5]).with_notes("a reader's note")
    ox = model_oxidation(frame.species)
    structure = frame_to_structure(frame, ox)

    assert calls == [8]
    assert structure.spacegroup_hm is None
    assert structure.spacegroup_number == 225
    assert md_model.NOTE_SYMMETRY_NOT_SEARCHED not in structure.notes
    assert structure.n_sites == structure.n_atoms == 8
    for i, (site, atom) in enumerate(zip(structure.sites, structure.atoms)):
        assert site.multiplicity == 1 and atom.site_index == i
        assert site.label == atom.label == f"{frame.elements[i]}{frame.atom_id[i]}"
        assert site.ox == ox.ox[site.element] and site.ox_source == "common"
    cart = np.array([a.cart for a in structure.atoms])
    assert np.allclose(cart, frame.cart_ang - frame.origin_ang, rtol=0,
                       atol=1e-12)
    assert np.allclose(cart, structure.cell.to_cartesian(frame.frac), rtol=0,
                       atol=0)
    assert "a reader's note" in structure.notes
    assert ox.notes[0] in structure.notes
    assert any(note.startswith("the box origin (2, -1, 0.5) Å is dropped")
               for note in structure.notes)

    results = coordination.analyse_structure(structure, resolve_oxidation=False)
    assert [r.cn_valence for r in results] == [6, 6, 6, 6]
    assert np.ptp([r.bvs for r in results]) <= 1e-12
    assert {s.ox_source for s in structure.sites} == {"common"}


def test_above_the_limit_spglib_never_runs(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the symmetry search ran above the limit")

    import spglib

    monkeypatch.setattr(readers, "_annotate", refuse)
    monkeypatch.setattr(spglib, "get_symmetry_dataset", refuse)
    frame = _rock_salt()
    structure = frame_to_structure(frame, model_oxidation(frame.species),
                                   symmetry_max_atoms=4)
    assert md_model.NOTE_SYMMETRY_NOT_SEARCHED in structure.notes
    assert structure.spacegroup_hm is None
    assert structure.spacegroup_number is None
    results = coordination.analyse_structure(structure, resolve_oxidation=False)
    assert len(results) == 4


def test_user_states_and_dropped_arrays_are_recorded():
    base = _rock_salt()
    frame = frame_from_arrays(base.elements, base.cart_ang, box_ang=base.box_ang,
                              vel_ang_per_ps=np.zeros((8, 3)),
                              charge_e=np.zeros(8))
    structure = frame_to_structure(frame, model_oxidation(frame.species,
                                                          {"Na": 1}))
    assert {s.ox_source for s in structure.sites if s.element == "Na"} == {"user"}
    assert any(note.startswith("not carried into the Structure") and
               "velocities" in note and "per-atom charges" in note
               for note in structure.notes)
    assert not any("origin" in note for note in structure.notes)


def test_elements_the_crystal_model_would_rename_are_refused():
    frame = frame_from_arrays(["Kr", "O"], frac=[[0, 0, 0], [0.5, 0.5, 0.5]],
                              box_ang=np.eye(3) * 6.0)
    ox = model_oxidation(frame.species, {"Kr": 0})
    with pytest.raises(ValueError, match="Kr to K"):
        frame_to_structure(frame, ox)


# ---------------------------------------------------------------------------
# supercell_frame
# ---------------------------------------------------------------------------

def _tiny_structure(occupancy: float = 1.0) -> Structure:
    """Two atoms in a tilted cell, built by hand (no reader, no spglib).

    The Cl x coordinate is exactly 1.0, as ``% 1.0`` can leave it, to show the
    tiling wraps it to 0.0 rather than placing an atom on the far face.
    """
    cell = readers.cell_from_vectors(TILTED)
    sites = [Site("Na1", "Na", [0.1, 0.2, 0.3], occupancy, multiplicity=1,
                  ox=1, ox_source="cif"),
             Site("Cl1", "Cl", [1.0, 0.7, 0.95], 1.0, multiplicity=1, ox=-1,
                  ox_source="cif")]
    atoms = [Atom(s.element, s.frac.copy(), cell.to_cartesian(s.frac), i,
                  s.label, s.occupancy) for i, s in enumerate(sites)]
    return Structure("tiny", cell, sites, atoms)


def test_a_supercell_is_a_pure_tiling_of_the_expanded_cell():
    structure = _tiny_structure()
    frame, parent = supercell_frame(structure, (2, 3, 1))

    assert frame.n_atoms == 12 and parent.tolist() == [0, 1] * 6
    assert not parent.flags.writeable
    assert frame.atom_id.tolist() == list(range(12))
    assert frame.elements.tolist() == ["Na", "Cl"] * 6
    assert np.array_equal(frame.box_ang, TILTED * np.array([[2], [3], [1]]))
    assert np.array_equal(frame.origin_ang, np.zeros(3))
    assert (frame.frac >= 0).all() and (frame.frac < 1).all()

    # each row is its parent atom moved by a whole number of cells
    parent_cart = np.array([structure.atoms[p].cart for p in parent])
    cells = np.linalg.solve(TILTED.T, (frame.cart_ang - parent_cart).T).T
    assert np.abs(cells - np.round(cells)).max() <= 1e-12
    # and every one of the 2 x 3 x 1 translations occurs once per parent
    for p in (0, 1):
        found = {tuple(int(x) for x in np.round(c)) for c in cells[parent == p]}
        expected = {(i, j, 0) for i in range(2) for j in range(3)}
        assert {(i % 2, j % 3, k) for i, j, k in found} == expected
    assert frame.notes[0].startswith("2x3x1 supercell of 'tiny'")


def test_a_one_cell_supercell_is_the_structure_itself():
    structure = _tiny_structure()
    frame, parent = supercell_frame(structure, (1, 1, 1))
    cart = np.array([a.cart for a in structure.atoms])
    gap = frame.cart_ang - cart
    gap = np.linalg.solve(TILTED.T, gap.T).T
    assert np.abs(gap - np.round(gap)).max() <= 1e-12
    assert frame.frac[1, 0] == 0.0          # the 1.0 of the input, wrapped


@pytest.mark.parametrize("reps", [(0, 1, 1), (2, 2), (2.0, 2, 2), (True, 1, 1),
                                  (-1, 1, 1)])
def test_supercell_repeat_refusals(reps):
    with pytest.raises(ValueError, match="whole numbers"):
        supercell_frame(_tiny_structure(), reps)


def test_a_partly_occupied_structure_is_not_tiled():
    with pytest.raises(ValueError, match="occupancy other than 1.*Na1"):
        supercell_frame(_tiny_structure(occupancy=0.5))
    empty = _tiny_structure()
    empty.atoms = []
    with pytest.raises(ValueError, match="no atoms"):
        supercell_frame(empty)


# ---------------------------------------------------------------------------
# no verdicts, and no Qt
# ---------------------------------------------------------------------------

VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def _every_note_the_module_produces() -> list[str]:
    """Drive each path that writes a note or a message, and collect them."""
    out: list[str] = []
    left_handed = TILTED[[1, 0, 2]]
    frame = frame_from_arrays(
        ["Si", "O", "Na"], np.array([[30.0, 1, 1], [-2.0, 2, 2], [3.0, 3, 3]]),
        box_ang=left_handed, origin_ang=ORIGIN, atom_id=[3, 1, 2],
        vel_ang_per_ps=np.zeros((3, 3)), charge_e=[2.4, -1.2, 0.6],
        unwrapped_cart_ang=np.zeros((3, 3)), timestep=7)
    out += frame.notes
    ox = model_oxidation(["Si", "O", "Na", "Te", "Xe", "Cl", "F"], {"Bi": 5})
    out += ox.notes
    out += frame_to_structure(frame, model_oxidation(frame.species),
                              symmetry_max_atoms=0).notes
    out += frame_to_structure(_rock_salt(), model_oxidation(["Na", "Cl"])).notes
    out += supercell_frame(_tiny_structure(), (2, 1, 1))[0].notes
    out += md_model.charges_per_element(["O", "O"], [-1.0, -1.1])[1]
    first = _rock_salt()
    traj = MemoryTrajectory([first, _moved(first, 0.0, 1, box=np.eye(3) * 5.7)],
                            units_note="length unit assumed Å")
    out += traj.describe()
    out += _FromList([first, first], unreadable={1}, skipped={1: "truncated"},
                     type_map={1: "Na"}).describe()

    def message(call):
        try:
            call()
        except (ValueError, IndexError) as error:
            return str(error)
        raise AssertionError("expected a refusal")

    for arguments, _ in [(_with(cart_ang=np.full((3, 3), np.nan)), None),
                         (_with(box_ang=np.zeros((3, 3))), None),
                         (_with(box_ang=np.array([[10.0, 0, 0], [0, 10, 0],
                                                  [10, 10, 0]])), None),
                         (_with(atom_id=[4, 7, 4]), None),
                         (_with(elements=[14, 8, 8]), None),
                         (_with(elements=[], cart_ang=np.zeros((0, 3))), None)]:
        elements_ = arguments.pop("elements")
        out.append(message(lambda: frame_from_arrays(elements_, **arguments)))
    out.append(message(lambda: validate_symbol("Ob")))
    out.append(message(lambda: validate_symbol("D")))
    out.append(message(lambda: model_oxidation(["He", "O"])))
    out.append(message(lambda: supercell_frame(_tiny_structure(0.5))))
    out.append(message(lambda: _FromList([first, first], unreadable={1}).frame(1)))
    out.append(message(lambda: traj.frame(9)))
    return out


def test_no_note_or_message_carries_a_verdict():
    notes = _every_note_the_module_produces()
    assert len(notes) > 30
    offending = [n for n in notes if VERDICT.search(n)]
    assert offending == []


def test_no_string_in_the_module_carries_a_verdict():
    """Every string literal in md_model.py, docstrings and f-string parts
    included, since any of them can reach a user."""
    source = (ROOT / "facet" / "core" / "md_model.py").read_text(encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 100
    offending = [s for s in strings if VERDICT.search(s)]
    assert offending == []


def test_the_model_imports_without_qt():
    code = ("import sys, facet.core.md_model;"
            "print('QT' if any(m.startswith('PySide6') for m in sys.modules) "
            "else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
