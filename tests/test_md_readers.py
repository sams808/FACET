"""The MD readers: every format gives back the model it was written from.

The files in tests/data/md were written by tests/data/md/make_md_files.py from
one five-atom model (MODEL below, loaded from that script), each with the
forward formulas of its own format. That script imports nothing from FACET, so
these tests check the readers against an independent statement of what each
file holds. What is pinned, and why:

* **Positions agree to 1e-10 Å across conventions.** The same atoms written as
  x, xs, xu, xsu, and x with image flags, in a tilted box with a non-zero
  origin. Leaving the origin out of the scaled conversion, or inverting the
  dump bounds with the wrong tilt, shifts positions by Å, far above 1e-10.
* **The box round-trips,** through LAMMPS's restricted parameters and through
  a dump's bounding box, on a hand-built case whose bounds are worked out by
  hand below, and against the cell-parameter formulas of the LAMMPS manual.
* **Type-map provenance is recorded,** and a type no source names is refused:
  a silently guessed element changes every number downstream.
* **Nothing is dropped silently.** A truncated last frame, a frame with
  another atom count and a header that does not parse are each in
  ``skipped`` with the reason; a row that does not parse raises FrameError
  naming its frame.
* **Every refusal says why,** in words, never a traceback from numpy or
  zlib, and a frame 0 that does not parse names the file and the frame.
* **A file cut while it was being written is not read as complete:** a last
  line with no line ending, a row an appended restart glued onto, stray rows,
  a damaged gzip stream. Every lost frame has its own skipped position.
* **A force-field name is not an element** because it matches one in another
  capitalisation ('HO' is not holmium), and a label the same file's mass
  contradicts is refused.
* **Formats a student meets** read as the files their programs write: LAMMPS
  extxyz in units real, ASE's initial_charges, labels in a dump's type
  column, an id-less dump LAMMPS re-sorted, VASP's fused fixed-width numbers,
  DL_POLY Classic's record 2.
* **CRLF copies parse identically** (core.autocrlf is true on this machine).
* **The readers.read hook** raises MDModelFile for MD files and leaves a
  single-frame XYZ exactly as it was.
* **No verdicts** in any note or message, and no Qt in the module.

Every number in the model is a test input, not a property of any material.
"""
from __future__ import annotations

import ast
import gzip
import importlib.util
import itertools
import pickle
import re
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from facet.core import elements, md_readers, readers
from facet.core.md_model import FrameError
from facet.core.md_readers import (box_from_dump_bounds, box_from_lammps,
                                   dump_bounds_from_box, lammps_from_box,
                                   read_trajectory, sniff_md,
                                   type_map_from_masses)

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data" / "md"

_SPEC = importlib.util.spec_from_file_location("make_md_files",
                                               DATA / "make_md_files.py")
GEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GEN)

TYPES = {1: "Si", 2: "O", 3: "Na"}
HISTORY_MAP = {"O_b": "O", "O_nb": "O"}
TOL_ANG = 1e-10

DUMP_CONVENTIONS = ["dump_tri_x.lammpstrj", "dump_tri_xs.lammpstrj",
                    "dump_tri_xu.lammpstrj", "dump_tri_xsu.lammpstrj",
                    "dump_tri_x_image.lammpstrj", "dump_tri_x.lammpstrj.gz"]

# file -> (read_trajectory keyword arguments, expected format)
FILES = {
    **{name: ({"type_map": TYPES}, "lammps-dump") for name in DUMP_CONVENTIONS},
    "dump_ortho_real.lammpstrj": ({}, "lammps-dump"),
    "dump_general.lammpstrj": ({}, "lammps-dump"),
    "dump_truncated.lammpstrj": ({"type_map": TYPES}, "lammps-dump"),
    "data_atomic.data": ({}, "lammps-data"),
    "data_charge.data": ({}, "lammps-data"),
    "data_general.data": ({}, "lammps-data"),
    "glass.extxyz": ({}, "extxyz"),
    "glass_reordered.extxyz": ({}, "extxyz"),
    "XDATCAR_fixed": ({}, "vasp-xdatcar"),
    "XDATCAR_variable": ({}, "vasp-xdatcar"),
    "CONFIG": ({}, "dlpoly-config"),
    "HISTORY": ({"type_map": HISTORY_MAP}, "dlpoly-history"),
}


def _open(name: str, **kwargs):
    defaults = dict(FILES[name][0])
    defaults.update(kwargs)
    return read_trajectory(DATA / name, **defaults)


def _frames(traj) -> list:
    with traj:
        return [traj.frame(k) for k in range(len(traj))]


def _gap_mod_one(a: np.ndarray, b: np.ndarray) -> float:
    gap = a - b
    gap -= np.round(gap)
    return float(np.abs(gap).max())


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# boxes
# ---------------------------------------------------------------------------

def test_the_hand_built_triclinic_bounds():
    """Frame 0's box, worked by hand from the Howto triclinic formulas.

    xlo = -3.25, xy = 2.5, xz = -1.5: MIN(0, 2.5, -1.5, 1.0) = -1.5, so
    xlo_bound = -4.75; MAX = 2.5, so xhi_bound = 6.75 + 2.5 = 9.25.
    yz = 1.25: ylo_bound = 1.5, yhi_bound = 10.5 + 1.25 = 11.75. z unchanged.
    Every value is exact in binary, so the comparison is exact too.
    """
    box, origin = box_from_lammps(-3.25, 6.75, 1.5, 10.5, 0.75, 8.75,
                                  2.5, -1.5, 1.25)
    assert box.tolist() == [[10.0, 0.0, 0.0], [2.5, 9.0, 0.0],
                            [-1.5, 1.25, 8.0]]
    assert origin.tolist() == [-3.25, 1.5, 0.75]
    bounds, tilt = dump_bounds_from_box(box, origin)
    assert bounds.tolist() == [[-4.75, 9.25], [1.5, 11.75], [0.75, 8.75]]
    assert tilt == (2.5, -1.5, 1.25)
    back_box, back_origin = box_from_dump_bounds(bounds, tilt)
    assert back_box.tolist() == box.tolist()
    assert back_origin.tolist() == origin.tolist()
    assert lammps_from_box(box, origin) == {
        "xlo": -3.25, "xhi": 6.75, "ylo": 1.5, "yhi": 10.5, "zlo": 0.75,
        "zhi": 8.75, "xy": 2.5, "xz": -1.5, "yz": 1.25}


@pytest.mark.parametrize("signs", list(itertools.product((-1, 1), repeat=3)))
def test_the_box_round_trips_for_every_tilt_sign(signs):
    """MIN and MAX in the bound formulas pick different terms for each sign
    of xy, xz and yz; all eight combinations round-trip."""
    rng = np.random.default_rng(sum((s + 1) * 3 ** i for i, s in enumerate(signs)))
    lo = rng.uniform(-5, 5, 3)
    lengths = rng.uniform(5, 15, 3)
    tilts = np.array(signs) * rng.uniform(0.1, 2.0, 3)
    box, origin = box_from_lammps(lo[0], lo[0] + lengths[0], lo[1],
                                  lo[1] + lengths[1], lo[2], lo[2] + lengths[2],
                                  *tilts)
    params = lammps_from_box(box, origin)
    assert np.allclose([params[k] for k in ("xlo", "ylo", "zlo")], lo,
                       rtol=0, atol=1e-12)
    assert np.allclose([params[k] for k in ("xy", "xz", "yz")], tilts,
                       rtol=0, atol=1e-12)
    bounds, tilt = dump_bounds_from_box(box, origin)
    back_box, back_origin = box_from_dump_bounds(bounds, tilt)
    assert np.abs(back_box - box).max() <= 1e-12
    assert np.abs(back_origin - origin).max() <= 1e-12


def test_cell_parameters_match_the_howto_formulas():
    """An independent route to the same box: the manual's a, b, c and cosines
    from lx, ly, lz, xy, xz, yz, against readers.cell_from_vectors on the rows."""
    lx, ly, lz, xy, xz, yz = 10.0, 9.0, 8.0, 2.5, -1.5, 1.25
    box, _ = box_from_lammps(0, lx, 0, ly, 0, lz, xy, xz, yz)
    cell = readers.cell_from_vectors(box)
    a = lx
    b = np.sqrt(ly ** 2 + xy ** 2)
    c = np.sqrt(lz ** 2 + xz ** 2 + yz ** 2)
    alpha = np.degrees(np.arccos((xy * xz + ly * yz) / (b * c)))
    beta = np.degrees(np.arccos(xz / c))
    gamma = np.degrees(np.arccos(xy / b))
    assert np.allclose([cell.a, cell.b, cell.c], [a, b, c], rtol=0, atol=1e-12)
    assert np.allclose([cell.alpha, cell.beta, cell.gamma], [alpha, beta, gamma],
                       rtol=0, atol=1e-10)


@pytest.mark.parametrize("call, message", [
    (lambda: box_from_lammps(1.0, 1.0, 0, 1, 0, 1), "xhi 1 not above xlo"),
    (lambda: box_from_lammps(0, 1, 0, float("nan"), 0, 1), "not finite"),
    (lambda: box_from_lammps(0, 1, 0, 1, 0, "x"), "not a number"),
    (lambda: lammps_from_box(GEN.GENERAL_BOX[0], np.zeros(3)), "rotation"),
    (lambda: box_from_dump_bounds([[0, 1], [0, 1]], None), "three"),
])
def test_box_refusals_say_why(call, message):
    with pytest.raises(ValueError, match=message):
        call()


# ---------------------------------------------------------------------------
# positions across conventions
# ---------------------------------------------------------------------------

def _check_against_model(frames, *, box=None, origin=None, unwrapped=False):
    for k, frame in enumerate(frames):
        expected_box = GEN.BOX[k] if box is None else box[k]
        expected_origin = GEN.ORIGIN[k] if origin is None else origin[k]
        assert np.abs(frame.box_ang - expected_box).max() <= 1e-12
        assert np.abs(frame.origin_ang - expected_origin).max() <= 1e-12
        assert frame.atom_id.tolist() == GEN.IDS.tolist()
        assert frame.elements.tolist() == GEN.ELEMENTS
        assert _gap_mod_one(frame.frac, GEN.FRAC[k]) <= TOL_ANG
        cart = expected_origin + GEN.FRAC[k] @ expected_box
        assert np.abs(frame.cart_ang - cart).max() <= TOL_ANG
        if unwrapped:
            full = expected_origin + (GEN.FRAC[k] + GEN.IMAGE[k]) @ expected_box
            assert np.abs(frame.unwrapped_cart_ang - full).max() <= TOL_ANG


@pytest.mark.parametrize("name", DUMP_CONVENTIONS)
def test_every_dump_convention_gives_the_model(name):
    traj = _open(name)
    assert traj.file_format == "lammps-dump"
    assert traj.timesteps.tolist() == GEN.TIMESTEPS
    assert traj.box_varies is True
    assert traj.periodic == (True, True, True)
    frames = _frames(traj)
    has_unwrapped = any(t in name for t in ("xu", "xsu", "image"))
    _check_against_model(frames, unwrapped=has_unwrapped)
    for frame in frames:
        assert (frame.unwrapped_cart_ang is not None) == has_unwrapped


def test_the_conventions_agree_with_each_other():
    """Pairwise, not only against the model: the largest difference between
    any two conventions, frame by frame."""
    cart = {name: [f.cart_ang for f in _frames(_open(name))]
            for name in DUMP_CONVENTIONS}
    worst = max(float(np.abs(a[k] - b[k]).max())
                for a, b in itertools.combinations(cart.values(), 2)
                for k in range(2))
    assert worst <= TOL_ANG
    unwrapped = [[f.unwrapped_cart_ang for f in _frames(_open(name))]
                 for name in ("dump_tri_xu.lammpstrj", "dump_tri_xsu.lammpstrj",
                              "dump_tri_x_image.lammpstrj")]
    worst = max(float(np.abs(a[k] - b[k]).max())
                for a, b in itertools.combinations(unwrapped, 2)
                for k in range(2))
    assert worst <= TOL_ANG


def test_the_gzip_dump_equals_the_plain_one():
    plain = _frames(_open("dump_tri_x.lammpstrj"))
    packed = _frames(_open("dump_tri_x.lammpstrj.gz"))
    for a, b in zip(plain, packed, strict=True):
        assert np.array_equal(a.cart_ang, b.cart_ang)
        assert np.array_equal(a.box_ang, b.box_ang)


def test_a_gzip_dump_read_out_of_order():
    """Frame 1 then frame 0: the stream rewinds, and the frames are the same."""
    with _open("dump_tri_x.lammpstrj.gz") as traj:
        second = traj.frame(1)
        first = traj.frame(0)
        again = traj.frame(1)
    assert np.array_equal(second.cart_ang, again.cart_ang)
    _check_against_model([first, second])


def test_the_general_triclinic_dump_and_data_file():
    """'abc origin' boxes: the rows as written, not rotated to LAMMPS's
    restricted form, and the origin from the fourth column."""
    frames = _frames(_open("dump_general.lammpstrj"))
    _check_against_model(frames, box=GEN.GENERAL_BOX, origin=GEN.GENERAL_ORIGIN)
    (frame,) = _frames(_open("data_general.data"))
    _check_against_model([frame], box=GEN.GENERAL_BOX, origin=GEN.GENERAL_ORIGIN)


def test_an_orthogonal_dump_in_real_units():
    """ITEM: UNITS real: TIME in fs and velocities in Å/fs, converted to ps
    and Å/ps; elements from the element column; charges per element."""
    traj = _open("dump_ortho_real.lammpstrj")
    assert traj.type_map_source == "element column"
    assert traj.type_map == {}
    assert np.allclose(traj.times_ps, GEN.TIMES_PS, rtol=0, atol=1e-15)
    assert "units real" in traj.units_note
    assert traj.charges_e == GEN.CHARGE
    ortho = [np.diag(np.diag(b)) for b in GEN.BOX]
    frames = _frames(traj)
    _check_against_model(frames, box=ortho, unwrapped=True)
    for k, frame in enumerate(frames):
        assert frame.time_ps == GEN.TIMES_PS[k]
        assert np.allclose(frame.vel_ang_per_ps, GEN.VELOCITY[k], rtol=1e-14,
                           atol=0)
        assert frame.charge_e.tolist() == [GEN.CHARGE[e] for e in GEN.ELEMENTS]


# ---------------------------------------------------------------------------
# frames that are not read
# ---------------------------------------------------------------------------

def test_the_truncated_last_frame_is_reported_not_dropped():
    traj = _open("dump_truncated.lammpstrj")
    assert traj.n_frames == 2
    assert traj.skipped == {2: "truncated: the file ends after 2 of 5 atom rows"}
    assert traj.file_positions.tolist() == [0, 1]
    assert any("1 frame(s) skipped" in line for line in traj.describe())
    # ITEM: UNITS is written once, in the first frame, and applies to all.
    assert "units metal" in traj.units_note
    _check_against_model(_frames(traj))


def _dump_text(frames: list[tuple[int, list[str]]], *, columns="id type x y z",
               bounds=("0 10", "0 10", "0 10")) -> str:
    out = []
    for step, rows in frames:
        out += ["ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS",
                str(len(rows)), "ITEM: BOX BOUNDS pp pp pp", *bounds,
                f"ITEM: ATOMS {columns}", *rows]
    return "\n".join(out) + "\n"


ROWS = ["1 1 1.0 1.0 1.0", "2 2 2.5 1.0 1.0", "3 2 1.0 2.5 1.0"]


def test_a_frame_with_another_atom_count_is_skipped(tmp_path):
    path = _write(tmp_path, "count.lammpstrj",
                  _dump_text([(0, ROWS), (10, ROWS[:2]), (20, ROWS)]))
    with read_trajectory(path, type_map={1: "Si", 2: "O"}) as traj:
        assert traj.n_frames == 2
        assert traj.skipped == {1: "2 atoms, where 2 of the 3 frames hold 3"}
        assert traj.timesteps.tolist() == [0, 20]
        assert traj.frame(1).timestep == 20


def test_the_atom_count_is_the_one_most_frames_hold(tmp_path):
    """One odd FIRST frame used to set the count, so every later frame was
    skipped; the count most frames hold is kept, and the odd frame skipped."""
    frames = [(0, ROWS[:2])] + [(10 * k, ROWS) for k in range(1, 5)]
    path = _write(tmp_path, "odd_first.lammpstrj", _dump_text(frames))
    with read_trajectory(path, type_map={1: "Si", 2: "O"}) as traj:
        assert traj.n_atoms == 3 and traj.n_frames == 4
        assert traj.skipped == {0: "2 atoms, where 4 of the 5 frames hold 3"}


def test_a_header_that_does_not_parse_is_skipped(tmp_path):
    text = _dump_text([(0, ROWS), (10, ROWS)]).replace(
        "ITEM: TIMESTEP\n10\n", "ITEM: TIMESTEP\nten\n")
    with read_trajectory(_write(tmp_path, "bad.lammpstrj", text),
                         type_map={1: "Si", 2: "O"}) as traj:
        assert traj.n_frames == 1
        assert traj.skipped == {1: "unreadable: ITEM: TIMESTEP holds 'ten', not "
                                   "a whole number"}


def test_a_middle_frame_with_missing_rows_is_skipped(tmp_path):
    """Fewer rows than NUMBER OF ATOMS in a frame followed by another: not a
    truncation, and the next frame is still read."""
    text = _dump_text([(0, ROWS), (10, ROWS), (20, ROWS)])
    text = text.replace("ITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n"
                        "0 10\n0 10\n0 10\nITEM: ATOMS id type x y z\n"
                        + "\n".join(ROWS) + "\nITEM: TIMESTEP\n20",
                        "ITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n"
                        "0 10\n0 10\n0 10\nITEM: ATOMS id type x y z\n"
                        + "\n".join(ROWS[:2]) + "\nITEM: TIMESTEP\n20")
    with read_trajectory(_write(tmp_path, "gap.lammpstrj", text),
                         type_map={1: "Si", 2: "O"}) as traj:
        assert traj.skipped == {1: "2 atom rows where ITEM: NUMBER OF ATOMS "
                                   "gives 3"}
        assert traj.timesteps.tolist() == [0, 20]


def test_a_row_that_does_not_parse_raises_frame_error_naming_the_frame(tmp_path):
    rows = ["1 1 1.0 1.0 1.0", "2 2 2.5 one 1.0", "3 2 1.0 2.5 1.0"]
    path = _write(tmp_path, "row.lammpstrj", _dump_text([(0, ROWS), (10, rows)]))
    with read_trajectory(path, type_map={1: "Si", 2: "O"}) as traj:
        traj.frame(0)
        with pytest.raises(FrameError, match=r"frame 1 \(file position 1\).*"
                                             r"not a number"):
            traj.frame(1)


def test_a_cut_gzip_stream_is_indexed_up_to_the_cut(tmp_path):
    """A gzip file copied while it was being written: no EOFError escapes,
    the cut frame is in skipped, and the note says the stream was cut."""
    text = _dump_text([(0, ROWS), (10, ROWS), (20, ROWS)])
    packed = gzip.compress(text.encode())
    path = tmp_path / "cut.lammpstrj.gz"
    path.write_bytes(packed[:-12])
    with read_trajectory(path, type_map={1: "Si", 2: "O"}) as traj:
        assert any("end-of-stream marker" in note for note in traj.notes)
        assert traj.n_frames + len(traj.skipped) == 3
        for k in range(traj.n_frames):
            traj.frame(k)


# ---------------------------------------------------------------------------
# types to elements
# ---------------------------------------------------------------------------

def test_type_map_provenance():
    assert _open("dump_tri_x.lammpstrj").type_map_source == "user"
    assert _open("dump_ortho_real.lammpstrj").type_map_source == "element column"
    assert _open("dump_general.lammpstrj").type_map_source == "type label"
    assert _open("data_general.data").type_map_source == "type label"
    traj = _open("data_atomic.data")
    assert traj.type_map_source == "data-file masses"
    assert traj.type_map == TYPES
    evidence = [n for n in traj.notes if "read as" in n]
    assert len(evidence) == 3
    assert "type 3:" in evidence[2] and "Na" in evidence[2] and "Mg" in evidence[2]
    traj = read_trajectory(DATA / "dump_tri_x.lammpstrj",
                           masses_from=DATA / "data_atomic.data")
    assert traj.type_map_source == "data-file masses"
    traj = read_trajectory(DATA / "dump_tri_x.lammpstrj", type_map={1: "Si"},
                           masses_from=DATA / "data_atomic.data")
    assert traj.type_map_source == "user + data-file masses"
    assert traj.type_map == TYPES


def test_two_sources_that_disagree_are_noted_and_the_users_is_used():
    traj = read_trajectory(DATA / "data_atomic.data",
                           type_map={1: "Si", 2: "F", 3: "Na"})
    assert traj.type_map[2] == "F"
    (note,) = [n for n in traj.notes if n.startswith("type 2:")]
    assert "F from the user is used" in note and "read as O" in note


def test_numeric_types_without_a_source_are_refused():
    with pytest.raises(ValueError) as raised:
        read_trajectory(DATA / "dump_tri_x.lammpstrj")
    message = str(raised.value)
    assert "no element for type 1" in message
    assert "type_map=" in message and "masses_from=" in message


def test_a_label_that_is_not_a_symbol_needs_a_map(tmp_path):
    text = _dump_text([(0, ["1 Si 1 1 1", "2 Ob 2.5 1 1", "3 Ob 1 2.5 1"])],
                      columns="id typelabel x y z")
    path = _write(tmp_path, "labels.lammpstrj", text)
    with pytest.raises(ValueError, match="'Ob'.*not an element symbol"):
        read_trajectory(path)
    traj = read_trajectory(path, type_map={"Ob": "O"})
    assert traj.type_map_source == "user + type label"
    assert traj.frame(0).elements.tolist() == ["Si", "O", "O"]


def test_mass_matching():
    """Bi given at 208.9804 is accepted although Po (209.0) is 0.0196 amu away;
    a whole-number mass, a mass near no element and a mass within the
    tolerance of two elements are refused."""
    mapping, evidence = type_map_from_masses({1: 208.9804, 2: 15.9994})
    assert mapping == {1: "Bi", 2: "O"}
    assert "Po" in evidence[0]
    for masses, message in [({1: 28.0}, "whole number"),
                            ({1: 1000.5}, "no standard atomic weight"),
                            ({1: -3.5}, "not a positive number")]:
        with pytest.raises(ValueError, match=message):
            type_map_from_masses(masses)
    # Bi (208.98) and Po (209.0) are 0.02 apart, so 208.99 sits on the edge of
    # 0.01 in floating point; at 0.015 it lies inside both.
    with pytest.raises(ValueError, match=r"within 0.015 amu of Po .* and of Bi"):
        type_map_from_masses({1: 208.99}, tol_amu=0.015)
    with pytest.raises(ValueError, match="positive"):
        type_map_from_masses({1: 28.0855}, tol_amu=0.0)


def test_the_weights_are_the_ones_elements_info_gives():
    """The table comes from gemmi directly because elements.info normalises
    (info('He') is hydrogen); where normalise keeps the symbol, the two agree."""
    weights = dict(md_readers.element_weights_amu())
    checked = 0
    for symbol, weight in weights.items():
        if elements.normalise(symbol) == symbol:
            assert elements.info(symbol).weight == weight
            checked += 1
    assert checked > 80
    assert weights["He"] != elements.info("He").weight


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

VEL_ROWS = ["1 1 1.0 1.0 1.0 0.5 -0.25 0.125", "2 2 2.5 1.0 1.0 1 2 3",
            "3 2 1.0 2.5 1.0 -1 0 1"]


def _velocity_dump(tmp_path, units=None):
    text = _dump_text([(0, VEL_ROWS)], columns="id type x y z vx vy vz")
    text = "ITEM: TIME\n250\n" + text
    if units:
        text = f"ITEM: UNITS\n{units}\n" + text
    return _write(tmp_path, "vel.lammpstrj", text)


def test_without_a_unit_style_values_are_kept_and_the_assumption_stated(tmp_path):
    traj = read_trajectory(_velocity_dump(tmp_path), type_map={1: "Si", 2: "O"})
    assert traj.times_ps.tolist() == [250.0]
    assert traj.frame(0).vel_ang_per_ps[0].tolist() == [0.5, -0.25, 0.125]
    assert "assumed" in traj.units_note and "units real" in traj.units_note


def test_the_units_argument_converts_real(tmp_path):
    traj = read_trajectory(_velocity_dump(tmp_path), type_map={1: "Si", 2: "O"},
                           units="real")
    assert traj.times_ps.tolist() == [0.25]
    assert traj.frame(0).vel_ang_per_ps[0].tolist() == [500.0, -250.0, 125.0]


@pytest.mark.parametrize("units, argument, message", [
    ("lj", None, "sigma"),
    ("nano", None, "nanometres"),
    (None, "si", "metres"),
    ("metal", "real", "says units metal, and the units argument says real"),
    (None, "furlong", "not a LAMMPS unit style"),
])
def test_unit_refusals(tmp_path, units, argument, message):
    with pytest.raises(ValueError, match=message):
        read_trajectory(_velocity_dump(tmp_path, units),
                        type_map={1: "Si", 2: "O"}, units=argument)


# ---------------------------------------------------------------------------
# LAMMPS data files
# ---------------------------------------------------------------------------

def test_an_atomic_data_file():
    traj = _open("data_atomic.data")
    (frame,) = _frames(traj)
    _check_against_model([frame])
    assert frame.timestep is None and frame.vel_ang_per_ps is None
    assert "atom style atomic (from the Atoms section comment)" in traj.notes
    assert "assumed" in traj.units_note


def test_a_charge_data_file_from_write_data():
    """write_data's first line gives the timestep and units metal; trailing
    image flags give the unwrapped positions; the Velocities section, listed in
    another order, is matched by id."""
    traj = _open("data_charge.data")
    (frame,) = _frames(traj)
    _check_against_model([frame], unwrapped=True)
    assert frame.timestep == 1000
    assert "units metal" in traj.units_note
    assert np.array_equal(frame.vel_ang_per_ps, GEN.VELOCITY[0])
    assert traj.charges_e == GEN.CHARGE


def _data_text(atoms_header: str, rows: list[str], *, extra: str = "") -> str:
    return ("test\n\n3 atoms\n2 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n"
            f"0 10 zlo zhi\n{extra}\nMasses\n\n1 28.0855\n2 15.9994\n\n"
            f"{atoms_header}\n\n" + "\n".join(rows) + "\n")


def test_a_full_style_file_and_an_inferred_style(tmp_path):
    """Only 5 columns name one style of the read_data table (atomic). Seven
    fit full and also sphere (atom-ID atom-type diameter density x y z), whose
    diameter, read as full, would become every atom's type: refused, naming
    both, and read once atom_style says which."""
    rows = ["1 7 1 2.4 1 1 1", "2 7 2 -1.2 2.5 1 1", "3 8 2 -1.2 1 2.5 1"]
    traj = read_trajectory(_write(tmp_path, "full.data",
                                  _data_text("Atoms # full", rows)))
    assert traj.frame(0).charge_e.tolist() == [2.4, -1.2, -1.2]
    path = _write(tmp_path, "seven.data", _data_text("Atoms", rows))
    with pytest.raises(ValueError, match="7 columns fit the atom styles.*full.*"
                                         "sphere.*atom_style="):
        read_trajectory(path)
    assert read_trajectory(path, atom_style="full").frame(0).charge_e.tolist() \
        == [2.4, -1.2, -1.2]
    rows8 = ["1 1 1 1 1 0 0 0", "2 2 2.5 1 1 0 0 0", "3 2 1 2.5 1 0 0 0"]
    with pytest.raises(ValueError, match="8 columns fit .*electron.*atomic with "
                                         "image flags"):
        read_trajectory(_write(tmp_path, "eight.data", _data_text("Atoms", rows8)))
    traj = read_trajectory(_write(tmp_path, "five.data", _data_text(
        "Atoms", ["1 1 1 1 1", "2 2 2.5 1 1", "3 2 1 2.5 1"])))
    assert any("atom style atomic inferred" in n for n in traj.notes)


@pytest.mark.parametrize("header, rows, kwargs, extra, message", [
    ("Atoms", ["1 1 2.4 1 1 1", "2 2 -1.2 2.5 1 1", "3 2 -1.2 1 2.5 1"], {}, "",
     "6 columns fit the atom styles angle, bond, charge"),
    ("Atoms # charge", ["1 1 2.4 1 1 1", "2 2 -1.2 2.5 1 1", "3 2 -1.2 1 2.5 1"],
     {"atom_style": "molecular"}, "", "says '# charge' and atom_style says"),
    ("Atoms # sphere", ["1 1 1 1 1 1 1", "2 2 1 1 2.5 1 1", "3 2 1 1 1 2.5 1"],
     {}, "", "'sphere' is not read"),
    ("Atoms # atomic", ["1 1 1 1 1", "2 2 2.5 1 1"], {}, "",
     "header says 3 atoms and the Atoms section lists 2"),
    ("Atoms # atomic", ["1 1 1 1 1", "2 2 2.5 1 1", "3 2 1 2.5"], {}, "",
     "every line needs the same columns"),
    ("Atoms # atomic", ["1 1 1 1 1", "2 3 2.5 1 1", "3 2 1 2.5 1"], {}, "",
     "the header declares 2"),
    ("Atoms # atomic", ["1 1 1 1 1", "2 2 2.5 1 1", "3 2 1 2.5 1"], {},
     "0 0 0 avec\n", "both a general triclinic box"),
])
def test_data_file_refusals(tmp_path, header, rows, kwargs, extra, message):
    path = _write(tmp_path, "bad.data", _data_text(header, rows, extra=extra))
    with pytest.raises(ValueError, match=message):
        read_trajectory(path, **kwargs)


def test_whole_number_masses_are_refused_with_a_way_forward(tmp_path):
    text = _data_text("Atoms # atomic", ["1 1 1 1 1", "2 2 2.5 1 1",
                                         "3 2 1 2.5 1"])
    text = text.replace("1 28.0855\n2 15.9994", "1 28.0\n2 16.0")
    path = _write(tmp_path, "rounded.data", text)
    with pytest.raises(ValueError, match="whole number.*type_map="):
        read_trajectory(path)
    assert read_trajectory(path, type_map={1: "Si", 2: "O"}).type_map_source \
        == "user"


def test_a_data_file_without_a_box_is_refused(tmp_path):
    text = _data_text("Atoms # atomic", ["1 1 1 1 1", "2 2 2.5 1 1",
                                         "3 2 1 2.5 1"]).replace(
        "0 10 zlo zhi\n", "")
    with pytest.raises(ValueError, match="zlo zhi.*not stated"):
        read_trajectory(_write(tmp_path, "nobox.data", text))


# ---------------------------------------------------------------------------
# extended XYZ
# ---------------------------------------------------------------------------

def test_extended_xyz_frames():
    traj = _open("glass.extxyz")
    assert traj.type_map_source == "file symbols"
    assert traj.charges_e == GEN.CHARGE
    assert traj.times_ps.tolist() == GEN.TIMES_PS
    assert any("energy" in n for n in traj.notes)
    assert any("Origin=" in n for n in traj.notes)
    frames = _frames(traj)
    _check_against_model(frames)
    for k, frame in enumerate(frames):
        assert np.array_equal(frame.vel_ang_per_ps, GEN.VELOCITY[k])


def _lattice(k: int) -> str:
    return " ".join(repr(float(v)) for v in GEN.BOX[k].reshape(-1))


@pytest.mark.parametrize("style", ["single", "braces", "brackets"])
def test_every_lattice_quote_style(tmp_path, style):
    text = (DATA / "glass.extxyz").read_text()
    for k in range(2):
        rows = GEN.BOX[k]
        new = {"single": f"Lattice='{_lattice(k)}'",
               "braces": f"Lattice={{{_lattice(k)}}}",
               "brackets": "Lattice=[" + ", ".join(
                   "[" + ", ".join(repr(float(v)) for v in r) + "]"
                   for r in rows) + "]"}[style]
        text = text.replace(f'Lattice="{_lattice(k)}"', new)
    assert "Lattice=\"" not in text
    frames = _frames(read_trajectory(_write(tmp_path, "q.extxyz", text)))
    _check_against_model(frames)


def test_xyz_without_a_lattice_is_a_molecule_and_refused(tmp_path):
    text = "2\nwater-ish\nO 0 0 0\nH 0.9 0 0\n2\nagain\nO 0 0 0\nH 0.9 0.1 0\n"
    path = _write(tmp_path, "molecule.xyz", text)
    with pytest.raises(ValueError, match="molecule"):
        read_trajectory(path)


def test_a_later_frame_without_a_lattice_is_skipped(tmp_path):
    lines = (DATA / "glass.extxyz").read_text().splitlines()
    lines[8] = re.sub(r'Lattice="[^"]*" ', "", lines[8])
    traj = read_trajectory(_write(tmp_path, "nolat.extxyz",
                                  "\n".join(lines) + "\n"))
    assert traj.n_frames == 1
    assert traj.skipped == {1: "no Lattice= in its comment line"}


def test_a_truncated_extended_xyz(tmp_path):
    text = (DATA / "glass.extxyz").read_text().splitlines()
    traj = read_trajectory(_write(tmp_path, "cut.extxyz",
                                  "\n".join(text[:-2]) + "\n"))
    assert traj.skipped == {1: "truncated: the file ends after 3 of 5 atom rows"}


# ---------------------------------------------------------------------------
# VASP XDATCAR
# ---------------------------------------------------------------------------

def _vasp_order(array):
    return np.asarray(array)[GEN.VASP_ORDER]


def test_a_fixed_cell_xdatcar():
    traj = _open("XDATCAR_fixed")
    assert traj.timesteps.tolist() == [1, 2]
    assert traj.box_varies is False
    for k, frame in enumerate(_frames(traj)):
        assert frame.elements.tolist() == _vasp_order(GEN.ELEMENTS).tolist()
        assert np.abs(frame.box_ang - GEN.BOX[0]).max() <= 1e-12
        assert _gap_mod_one(frame.frac, _vasp_order(GEN.FRAC[k])) <= TOL_ANG


def test_a_variable_cell_xdatcar_and_its_scale_factors():
    """Frame 0: scale 2 on halved vectors. Frame 1: a negative scale, the cell
    volume, on vectors shrunk by 0.9. Both give the model's box."""
    traj = _open("XDATCAR_variable")
    assert traj.box_varies is True
    assert traj.type_map == {"Na_pv": "Na"}
    assert any("'Na_pv' read as Na" in n for n in traj.notes)
    for k, frame in enumerate(_frames(traj)):
        assert np.abs(frame.box_ang - GEN.BOX[k]).max() <= 1e-10
        assert _gap_mod_one(frame.frac, _vasp_order(GEN.FRAC[k])) <= TOL_ANG
        assert frame.elements.tolist() == _vasp_order(GEN.ELEMENTS).tolist()


def test_the_negative_scale_follows_read_poscar(tmp_path):
    """FACET's POSCAR reader already reads a negative scale as a volume; the
    XDATCAR reader gives the same box from the same header."""
    lines = (DATA / "XDATCAR_variable").read_text().splitlines()
    header = lines[13:20]
    poscar = header + ["Direct"] + lines[21:26]
    path = _write(tmp_path, "POSCAR", "\n".join(poscar) + "\n")
    structure = readers.read_poscar(path)
    frame = _frames(_open("XDATCAR_variable"))[1]
    assert np.abs(structure.cell.orth.T - frame.box_ang).max() <= 1e-12


def test_cartesian_configurations_and_three_scale_factors(tmp_path):
    """Three positive factors scale the x, y and z components of the vectors
    and of Cartesian positions (VASP wiki, POSCAR). Written here from the
    model: vectors and positions divided by the factors, so reading must
    multiply them back."""
    scale = np.array([2.0, 1.0, 0.5])
    lattice = GEN.BOX[0] / scale
    cart = GEN.FRAC[0] @ GEN.BOX[0]
    lines = ["three scales", " ".join(repr(float(s)) for s in scale)]
    lines += [" ".join(repr(float(v)) for v in r) for r in lattice]
    lines += ["Si O Na", "1 3 1", "Cartesian configuration=     1"]
    lines += [" ".join(repr(float(v)) for v in (cart[i] / scale))
              for i in GEN.VASP_ORDER]
    traj = read_trajectory(_write(tmp_path, "XDATCAR", "\n".join(lines) + "\n"))
    (frame,) = _frames(traj)
    assert np.abs(frame.box_ang - GEN.BOX[0]).max() <= 1e-12
    assert _gap_mod_one(frame.frac, _vasp_order(GEN.FRAC[0])) <= TOL_ANG
    assert any("three scale factors" in n for n in traj.notes)


def test_dlpoly_files_with_forces(tmp_path):
    """levcfg 2 and keytrj 2 add a force record to every atom block; the
    forces are not read, and the positions and velocities around them are."""
    config = (DATA / "CONFIG").read_text().splitlines()
    out = [config[0], "2 3 5", *config[2:5]]
    for i in range(5):
        out += config[5 + 3 * i:8 + 3 * i] + ["0.1 0.2 0.3"]
    (frame,) = _frames(md_readers.read_dlpoly_config(
        _write(tmp_path, "CONFIG", "\n".join(out) + "\n")))
    _check_against_model([frame], origin=[-0.5 * GEN.BOX[0].sum(axis=0)])
    history = (DATA / "HISTORY").read_text().splitlines()
    out = [history[0], "2 3 5 2 52"]
    cursor = 2
    for _ in range(2):
        out.append(history[cursor].replace(" 5 1 3 ", " 5 2 3 "))
        out += history[cursor + 1:cursor + 4]
        cursor += 4
        for _ in range(5):
            out += history[cursor:cursor + 3] + ["0.1 0.2 0.3"]
            cursor += 3
    traj = md_readers.read_dlpoly_history(
        _write(tmp_path, "HISTORY", "\n".join(out) + "\n"), type_map=HISTORY_MAP)
    frames = _frames(traj)
    _check_against_model(frames, origin=[-0.5 * b.sum(axis=0) for b in GEN.BOX])
    assert np.array_equal(frames[1].vel_ang_per_ps, GEN.VELOCITY[1])
    assert any("forces not read" in n for n in traj.notes)


def test_extended_xyz_with_atomic_numbers_or_numeric_types(tmp_path):
    base = (DATA / "glass.extxyz").read_text().splitlines()
    z = {"Si": "14", "O": "8", "Na": "11"}
    numbers = []
    types = []
    for line in base:
        parts = line.split()
        if parts and parts[0] in z:
            numbers.append(" ".join([z[parts[0]], *parts[1:]]))
            types.append(" ".join([{"Si": "1", "O": "2", "Na": "3"}[parts[0]],
                                   *parts[1:]]))
        else:
            numbers.append(line)
            types.append(line)
    as_z = "\n".join(numbers).replace("species:S:1", "Z:I:1") + "\n"
    _check_against_model(_frames(read_trajectory(
        _write(tmp_path, "z.extxyz", as_z))))
    as_types = "\n".join(types).replace("species:S:1", "type:I:1") + "\n"
    path = _write(tmp_path, "t.extxyz", as_types)
    with pytest.raises(ValueError, match="numeric types"):
        read_trajectory(path)
    traj = read_trajectory(path, type_map=TYPES)
    assert traj.type_map_source == "user"
    _check_against_model(_frames(traj))


def test_a_dump_with_type_and_element_columns(tmp_path):
    """The element column names each type; a user map that disagrees is used,
    and the disagreement is in the notes."""
    rows = ["1 1 Si 1 1 1", "2 2 O 2.5 1 1", "3 2 O 1 2.5 1"]
    path = _write(tmp_path, "both.lammpstrj",
                  _dump_text([(0, rows)], columns="id type element x y z"))
    traj = read_trajectory(path)
    assert traj.type_map_source == "element column"
    assert traj.frame(0).elements.tolist() == ["Si", "O", "O"]
    traj = read_trajectory(path, type_map={1: "Ge", 2: "O"})
    assert traj.type_map_source == "user"
    assert traj.frame(0).elements.tolist() == ["Ge", "O", "O"]
    assert any("type 1: Ge from the user is used" in n and "'Si'" in n
               for n in traj.notes)


def test_a_vasp4_xdatcar_is_refused(tmp_path):
    lines = (DATA / "XDATCAR_fixed").read_text().splitlines()
    del lines[5]
    with pytest.raises(ValueError, match="VASP 4"):
        md_readers.read_xdatcar(_write(tmp_path, "XDATCAR", "\n".join(lines)))


# ---------------------------------------------------------------------------
# DL_POLY
# ---------------------------------------------------------------------------

def test_a_dlpoly_config_has_its_origin_at_the_centre():
    traj = _open("CONFIG")
    (frame,) = _frames(traj)
    origin = [-0.5 * GEN.BOX[0].sum(axis=0)]
    _check_against_model([frame], origin=origin)
    assert np.array_equal(frame.vel_ang_per_ps, GEN.VELOCITY[0])
    assert traj.type_map_source == "file symbols"


def test_a_dlpoly_history():
    traj = _open("HISTORY")
    assert traj.type_map_source == "user + file symbols"
    assert traj.type_map == HISTORY_MAP
    assert traj.timesteps.tolist() == GEN.TIMESTEPS
    assert traj.times_ps.tolist() == GEN.TIMES_PS
    assert traj.charges_e == GEN.CHARGE
    frames = _frames(traj)
    origins = [-0.5 * b.sum(axis=0) for b in GEN.BOX]
    _check_against_model(frames, origin=origins)
    for k, frame in enumerate(frames):
        assert np.array_equal(frame.vel_ang_per_ps, GEN.VELOCITY[k])


def test_dlpoly_names_that_are_not_symbols_need_a_map():
    with pytest.raises(ValueError, match=r"'O_b' \(mass 15.9994 amu.*type_map="):
        read_trajectory(DATA / "HISTORY")


@pytest.mark.parametrize("imcon, message", [(0, "no periodic boundaries"),
                                            (6, "slab"), (9, "not a DL_POLY")])
def test_dlpoly_boundaries_that_are_not_periodic_are_refused(tmp_path, imcon,
                                                             message):
    text = (DATA / "CONFIG").read_text().replace("1 3 5", f"1 {imcon} 5", 1)
    with pytest.raises(ValueError, match=message):
        md_readers.read_dlpoly_config(_write(tmp_path, "CONFIG", text))


# ---------------------------------------------------------------------------
# every format, the same model
# ---------------------------------------------------------------------------

def test_every_format_gives_the_same_fractional_positions():
    """Frame 0 of every file, in the model's atom order, against FRAC: the
    origin conventions differ (LAMMPS xlo, DL_POLY centre, VASP zero), the
    fractional positions may not."""
    worst = 0.0
    for name in FILES:
        frame = _frames(_open(name))[0]
        frac = frame.frac
        if name.startswith("XDATCAR"):
            frac = frac[np.argsort(GEN.VASP_ORDER)]
        worst = max(worst, _gap_mod_one(frac, GEN.FRAC[0]))
    assert worst <= TOL_ANG


@pytest.mark.parametrize("name", sorted(FILES))
def test_crlf_copies_parse_identically(tmp_path, name):
    raw = (DATA / name).read_bytes()
    # From LF, whatever the checkout did: converting an already-CRLF file
    # would make CR CR LF, which no writer produces.
    if name.endswith(".gz"):
        crlf = gzip.compress(gzip.decompress(raw).replace(b"\r\n", b"\n")
                             .replace(b"\n", b"\r\n"))
    else:
        crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    copy = tmp_path / name
    copy.write_bytes(crlf)
    kwargs = FILES[name][0]
    original, converted = _open(name), read_trajectory(copy, **kwargs)
    assert converted.skipped == original.skipped
    assert converted.timesteps.tolist() == original.timesteps.tolist()
    for a, b in zip(_frames(original), _frames(converted), strict=True):
        for field in ("cart_ang", "frac", "box_ang", "origin_ang", "atom_id",
                      "elements"):
            assert np.array_equal(getattr(a, field), getattr(b, field)), field


@pytest.mark.parametrize("name", sorted(FILES))
def test_sniff_md_names_each_format(name):
    assert sniff_md(DATA / name) == FILES[name][1]


def test_sniff_md_leaves_crystal_files_alone(tmp_path):
    poscar = (DATA / "XDATCAR_fixed").read_text().splitlines()[:7] + [
        "Direct", "0.1 0.2 0.3", "0.25 0.15 0.45", "0.6 0.7 0.2",
        "0.35 0.9 0.65", "0.85 0.4 0.9"]
    assert sniff_md(_write(tmp_path, "POSCAR", "\n".join(poscar))) is None
    cif = ROOT / "tests" / "data" / "crystals" / "quartz_SiO2_cod9013321.cif"
    assert sniff_md(cif) is None
    config = shutil.copy(DATA / "CONFIG", tmp_path / "model.txt")
    assert sniff_md(config) is None, "a CONFIG is recognised only by its name"


def test_a_trajectory_pickles_for_worker_processes():
    for name in ("dump_tri_x.lammpstrj", "dump_tri_x.lammpstrj.gz",
                 "glass.extxyz", "HISTORY"):
        traj = _open(name)
        traj.frame(1)
        copy = pickle.loads(pickle.dumps(traj))
        assert np.array_equal(copy.frame(1).cart_ang, traj.frame(1).cart_ang)
        traj.close()
        copy.close()


# ---------------------------------------------------------------------------
# the entry point and the readers.read hook
# ---------------------------------------------------------------------------

def test_read_trajectory_refuses_options_the_format_does_not_use():
    with pytest.raises(ValueError, match="does not use atom_style"):
        read_trajectory(DATA / "dump_tri_x.lammpstrj", type_map=TYPES,
                        atom_style="atomic")
    with pytest.raises(ValueError, match="does not use masses_from$"):
        read_trajectory(DATA / "glass.extxyz", masses_from="x", units="metal")
    with pytest.raises(ValueError, match="does not use units"):
        read_trajectory(DATA / "XDATCAR_fixed", units="metal")


def test_read_trajectory_on_files_that_are_not_models(tmp_path):
    with pytest.raises(readers.UnsupportedFormat, match="not recognised as an "
                                                        "MD model"):
        read_trajectory(_write(tmp_path, "junk.txt", "not a model\n"))
    with pytest.raises(FileNotFoundError):
        read_trajectory(tmp_path / "missing.lammpstrj")


@pytest.mark.parametrize("name, copy_as", [
    *[(name, None) for name in sorted(FILES)],
    ("dump_tri_x.lammpstrj", "dump.glass"),
    ("data_atomic.data", "data.glass"),
    ("data_atomic.data", "glass.lmp"),
    ("dump_tri_x.lammpstrj", "glass.dump"),
    ("dump_tri_x.lammpstrj.gz", "glass.lammpstrj.gz"),
    ("CONFIG", "REVCON"),
    ("glass.extxyz", "glass.xyz"),
])
def test_readers_read_hands_md_files_to_the_md_readers(tmp_path, name, copy_as):
    path = DATA / name if copy_as is None else \
        Path(shutil.copy(DATA / name, tmp_path / copy_as))
    with pytest.raises(readers.MDModelFile) as raised:
        readers.read(path)
    assert raised.value.file_format == FILES[name][1]
    assert raised.value.path == str(path)
    assert "read_trajectory" in str(raised.value)
    assert isinstance(raised.value, ValueError)


# What readers.read gave for frame 0 of glass.extxyz saved as one.xyz before
# the hook existed: the integration check imported `git show HEAD`'s
# readers.py beside the working tree and found the two outputs identical bit
# for bit, so these values are HEAD's. read_xyz ignores Origin=, so the
# fractions are of the Cartesian positions as written.
ONE_XYZ_CELL = (10.0, 9.340770846134703, 8.234834546000302, 84.40452631810119,
                100.49519244424717, 74.47588900324574)
ONE_XYZ_FRAC = [[0.5006510416666667, 0.5536458333333333, 0.99375],
                [0.7506510416666666, 0.35364583333333327, 0.39375],
                [0.0006510416666667032, 0.05364583333333317, 0.74375],
                [0.2506510416666667, 0.8536458333333333, 0.29375],
                [0.9006510416666667, 0.3036458333333333, 0.54375]]


def test_a_single_frame_xyz_reads_as_before(tmp_path):
    """The hook only looks for a second frame; a one-frame file goes to the
    crystal read_xyz and gives what HEAD gave (stored above), with its spglib
    annotation and its note, so a change on the crystal path fails here."""
    lines = (DATA / "glass.extxyz").read_text().splitlines()[:7]
    path = _write(tmp_path, "one.xyz", "\n".join(lines) + "\n")
    structure = readers.read(path)
    assert structure.n_atoms == 5
    assert [a.label for a in structure.atoms] == ["Na1", "Si2", "O3", "O4", "O5"]
    cell = structure.cell
    assert np.allclose([cell.a, cell.b, cell.c, cell.alpha, cell.beta,
                        cell.gamma], ONE_XYZ_CELL, rtol=0, atol=1e-12)
    assert np.allclose([a.frac.tolist() for a in structure.atoms], ONE_XYZ_FRAC,
                       rtol=0, atol=1e-12)
    assert structure.spacegroup_number == 1
    assert structure.notes == [
        "read with explicit coordinates; the symmetry reported is what spglib "
        "finds in them, not what the file declared"]


def test_multi_frame_xyz_without_a_lattice_opens_nowhere_and_says_so(tmp_path):
    """LAMMPS 'dump xyz' and a molecule's trajectory: no box for the MD
    reader, more frames than the crystal reader takes. Both refusals now say
    the same thing instead of pointing at each other, and keeping frame 0
    silently (what the crystal reader did before the hook) stays refused."""
    path = _write(tmp_path, "two.xyz", "1\na\nO 0 0 0\n1\nb\nO 0 0 0.1\n")
    with pytest.raises(readers.UnsupportedFormat,
                       match="more than one XYZ frame and no periodic box") \
            as raised:
        readers.read(path)
    assert not isinstance(raised.value, readers.MDModelFile)
    assert "dump custom" in str(raised.value)
    with pytest.raises(ValueError, match="no periodic box.*read_trajectory reads "
                                         "periodic models only"):
        read_trajectory(path)


def test_a_lone_number_after_one_frame_is_not_a_second_frame(tmp_path):
    """is_multiframe_xyz used to count a trailing integer line as a frame, so a
    one-frame file that ends in '7' was refused by readers.read."""
    path = _write(tmp_path, "trailing.xyz", "3\nmolecule\nO 0 0 0\nH 0.9 0 0\n"
                                            "H 0 0.9 0\n\n7\n")
    assert md_readers.is_multiframe_xyz(path) is False


def test_files_that_are_not_models_still_reach_the_crystal_sniffer(tmp_path):
    script = _write(tmp_path, "in.lmp", "units metal\nread_data glass.data\n"
                                        "pair_style buck/coul/long 10.0\nrun 0\n")
    junk = _write(tmp_path, "junk.dat", "not a structure\n")
    for path in (script, junk):
        with pytest.raises(readers.UnsupportedFormat) as raised:
            readers.read(path)
        assert not isinstance(raised.value, readers.MDModelFile)
        assert "CIF" in str(raised.value)


def test_garbage_never_escapes_as_an_unexpected_exception(tmp_path):
    """The readers' contract (test_robustness): a file that is not what it
    claims fails with ValueError or OSError, never IndexError or EOFError."""
    rng = np.random.default_rng(7)
    good = (DATA / "dump_tri_x.lammpstrj").read_bytes()
    cases = {
        "empty.lammpstrj": b"",
        "zeros.dump": bytes(64),
        "random.lammpstrj": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes(),
        "items.lammpstrj": b"ITEM:\nITEM: ATOMS\nITEM: BOX BOUNDS\n",
        "half.lammpstrj": good[:len(good) // 2],
        "badgz.lammpstrj.gz": b"\x1f\x8b" + bytes(30),
        # a valid gzip header and a corrupt deflate body: zlib.error, which is
        # neither ValueError nor OSError, used to escape from here
        **{f"{kind}.lammpstrj.gz": data
           for kind, data in _gzip_damage(good).items()},
        "corrupt.xyz": _gzip_damage(good)["body"],
        "data.bad": b"title\n\n3 atoms\n0 1 xlo xhi\n\nAtoms\n\n1 1\n",
        "XDATCAR": b"t\n1\n1 0 0\n0 1 0\n0 0 1\nSi\n2\nDirect configuration= 1\n0 0\n",
        "HISTORY": b"t\n1 3 2\ntimestep 0 2 1 3 0.001\n1 0\n",
        "CONFIG": b"t\n0 3\n1 0 0\n0 1 0\n0 0 1\nSi 1\n0.1 0.1\n",
        "glass.extxyz": b'2\nLattice="1 0 0 0 1 0 0 0 1"\nSi 0 0\n',
    }
    for name, content in cases.items():
        path = tmp_path / name
        path.write_bytes(content)
        try:
            with read_trajectory(path, **({"type_map": TYPES}
                                          if "lammpstrj" in name else {})) as t:
                for k in range(len(t)):
                    t.frame(k)
        except (ValueError, OSError):
            pass
        sniff_md(path)                       # answers, never raises
        md_readers.is_multiframe_xyz(path)
        try:
            readers.read(path)
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# files damaged, cut or glued while being written
# ---------------------------------------------------------------------------

def _gzip_damage(raw: bytes) -> dict[str, bytes]:
    """Four damaged copies of one gzip stream: a valid header followed by a
    deflate body zlib rejects (zlib.error), a flipped CRC32 trailer and a
    flipped ISIZE trailer (gzip.BadGzipFile, an OSError without the file
    name), and bytes after the stream that are not gzip."""
    intact = gzip.compress(raw, mtime=0)
    body = bytearray(intact)
    for i in range(12, 40):
        body[i] ^= 0xFF
    crc = bytearray(intact)
    crc[-6] ^= 0x01
    size = bytearray(intact)
    size[-2] ^= 0x01
    return {"body": bytes(body), "crc": bytes(crc), "isize": bytes(size),
            "trailing": intact + b"garbage after the stream"}


@pytest.mark.parametrize("damage", ["body", "crc", "isize", "trailing"])
def test_a_damaged_gzip_is_refused_naming_the_file(tmp_path, damage):
    """zlib.error (neither ValueError nor OSError) and BadGzipFile used to
    escape from read_trajectory, sniff_md and readers.read. Now
    read_trajectory raises UnsupportedFormat naming the file and the damage,
    the sniffers answer None and False as their docstrings say, and
    readers.read gives the UnsupportedFormat it gave before the hook."""
    raw = (DATA / "dump_tri_x.lammpstrj").read_bytes()
    path = tmp_path / "damaged.lammpstrj.gz"
    path.write_bytes(_gzip_damage(raw)[damage])
    with pytest.raises(readers.UnsupportedFormat,
                       match=r"damaged\.lammpstrj\.gz: the gzip stream is "
                             r"damaged"):
        read_trajectory(path, type_map=TYPES)
    assert sniff_md(path) is None
    assert md_readers.is_multiframe_xyz(path) is False
    with pytest.raises(readers.UnsupportedFormat) as raised:
        readers.read(path)
    assert not isinstance(raised.value, readers.MDModelFile)


def test_a_gzip_damaged_after_indexing_raises_frame_error(tmp_path):
    """The frame is decompressed when it is loaded: damage found then is a
    FrameError for that frame, not a zlib.error."""
    raw = (DATA / "dump_tri_x.lammpstrj").read_bytes()
    path = tmp_path / "later.lammpstrj.gz"
    path.write_bytes(gzip.compress(raw, mtime=0))
    traj = read_trajectory(path, type_map=TYPES)
    path.write_bytes(_gzip_damage(raw)["body"])
    with pytest.raises(FrameError, match="gzip stream is damaged"):
        traj.frame(1)
    traj.close()


@pytest.mark.parametrize("name, kwargs, digits", [
    ("dump_tri_x.lammpstrj", {"type_map": TYPES}, "2.496"),
    ("glass.extxyz", {}, "-1.2"),
    ("XDATCAR_fixed", {}, "0.06"),
    ("HISTORY", {"type_map": HISTORY_MAP}, "-1.5"),
])
def test_a_last_number_cut_with_its_line_ending_skips_the_frame(tmp_path, name,
                                                                 kwargs, digits):
    """Every writer ends each line with a line ending, so a file whose last
    line has none was cut, possibly inside its last number. Cutting the last
    two characters and the line ending leaves every column present ('2.496'
    becomes '2.4'), and the frame used to be read with the shortened value.
    Now it is skipped with the reason; the frame before it still reads."""
    raw = (DATA / name).read_bytes().replace(b"\r\n", b"\n")
    assert raw.endswith(digits.encode() + b"\n")
    path = tmp_path / name
    path.write_bytes(raw[:-3])
    traj = read_trajectory(path, **kwargs)
    assert traj.n_frames == 1
    (reason,) = traj.skipped.values()
    assert reason.startswith("truncated") and "line ending" in reason
    _frames(traj)
    path.write_bytes(raw[:-1])          # complete digits, no line ending
    assert read_trajectory(path, **kwargs).n_frames == 1


@pytest.mark.parametrize("name", ["data_atomic.data", "CONFIG"])
def test_a_one_frame_file_cut_without_its_line_ending_is_refused(tmp_path, name):
    raw = (DATA / name).read_bytes()
    path = tmp_path / name
    path.write_bytes(raw[:-3])
    with pytest.raises(ValueError, match="does not end with a line ending"):
        read_trajectory(path)


def test_a_restart_appended_to_a_row_a_crash_cut(tmp_path):
    """A run killed mid-row and restarted with dump_modify append writes its
    first ITEM: TIMESTEP on the cut row's line. The cut frame is skipped as
    cut, and the restarted run's frames are read (they used to be lost as a
    header with no TIMESTEP, and the cut frame counted readable)."""
    first = _dump_text([(0, ROWS), (10, ROWS)])
    assert first.endswith("2.5 1.0\n")
    text = first[:-len(" 1.0\n")] + _dump_text([(10, ROWS), (20, ROWS)])
    with read_trajectory(_write(tmp_path, "restart.lammpstrj", text),
                         type_map={1: "Si", 2: "O"}) as traj:
        assert list(traj.skipped) == [1]
        assert "its last atom row is cut: an 'ITEM:' line starts inside it" \
            in traj.skipped[1]
        assert traj.timesteps.tolist() == [0, 10, 20]
        assert traj.file_positions.tolist() == [0, 2, 3]
        _frames(traj)


def test_stray_rows_skip_a_frame_when_indexed_and_blank_lines_do_not(tmp_path):
    """A frame with more rows than NUMBER OF ATOMS used to be counted
    readable and fail when loaded, so len() overstated the frames. Its values
    are counted when the file is indexed; blank lines are not rows."""
    text = _dump_text([(0, ROWS), (10, ROWS + ["4 1 3.0 3.0 3.0"]), (20, ROWS)])
    text = text.replace("ITEM: NUMBER OF ATOMS\n4\n", "ITEM: NUMBER OF ATOMS\n3\n")
    with read_trajectory(_write(tmp_path, "stray.lammpstrj", text),
                         type_map={1: "Si", 2: "O"}) as traj:
        assert traj.skipped == {1: "4 lines with 20 values follow ITEM: ATOMS, "
                                   "where ITEM: NUMBER OF ATOMS gives 3 rows of "
                                   "5 values"}
        assert len(_frames(traj)) == 2
    text = _dump_text([(0, ROWS), (10, ROWS)]).replace(
        "2.5 1.0\nITEM: TIMESTEP", "2.5 1.0\n\nITEM: TIMESTEP", 1)
    with read_trajectory(_write(tmp_path, "blank.lammpstrj", text),
                         type_map={1: "Si", 2: "O"}) as traj:
        assert traj.skipped == {} and len(_frames(traj)) == 2


def _xyz_blocks() -> list[list[str]]:
    lines = (DATA / "glass.extxyz").read_text().splitlines()
    return [lines[0:7], lines[7:14], lines[0:7], lines[7:14]]


@pytest.mark.parametrize("case, n_frames, skipped", [
    ("blank", 4, []),
    ("count too small", 3, [1]),
    ("count too large", 2, [1, 2]),
])
def test_every_lost_extended_xyz_frame_is_counted(tmp_path, case, n_frames,
                                                   skipped):
    """The walk used to stop at the first blank line or count that disagreed
    with the rows, with one skipped entry for all the frames after it. Blank
    lines between frames are passed over (the count line delimits frames);
    a frame whose count disagrees with its rows is skipped, the walk resyncs
    at the next count line followed by Lattice=, and a frame whose count line
    the frame before swallowed gets its own position."""
    blocks = _xyz_blocks()
    if case == "count too small":
        blocks[1][0] = "4"
    if case == "count too large":
        blocks[1][0] = "6"
    separator = "\n\n" if case == "blank" else "\n"
    text = separator.join("\n".join(b) for b in blocks) + "\n"
    traj = read_trajectory(_write(tmp_path, "lost.extxyz", text))
    assert traj.n_frames == n_frames
    assert sorted(traj.skipped) == skipped
    assert traj.n_frames + len(traj.skipped) == 4
    if case == "blank":
        assert any("blank line(s) between frames" in n for n in traj.notes)
    _frames(traj)


def _xdatcar(n_configs: int) -> list[str]:
    lines = (DATA / "XDATCAR_fixed").read_text().splitlines()
    out = lines[:7]
    for k in range(n_configs):
        out += [f"Direct configuration=     {k + 1}"] + lines[8 + 6 * (k % 2):
                                                            13 + 6 * (k % 2)]
    return out


def test_a_missing_xdatcar_configuration_line_is_one_lost_frame(tmp_path):
    """Three configurations, the second's configuration line removed: the
    first keeps its own five rows, the orphaned rows are one skipped
    position, the third reads (one skipped entry used to stand for two lost
    configurations and the first was skipped too)."""
    lines = _xdatcar(3)
    lines.remove("Direct configuration=     2")
    traj = read_trajectory(_write(tmp_path, "XDATCAR", "\n".join(lines) + "\n"))
    assert traj.timesteps.tolist() == [1, 3]
    assert traj.skipped == {1: "5 coordinate lines with no configuration line "
                               "before them (a configuration line is missing)"}
    _frames(traj)


def test_a_seven_atom_fixed_cell_is_not_taken_for_a_variable_cell(tmp_path):
    """With 7 atoms, a missing configuration line leaves a gap of 14 lines,
    the gap of a repeated 7-line header: the gap alone called the file
    variable-cell. The lines before the configuration are parsed instead."""
    rows = [f"0.{i} 0.{i} 0.{i}" for i in range(1, 8)]
    header = ["seven", "1.0", "8.0 0.0 0.0", "0.0 8.0 0.0", "0.0 0.0 8.0",
              "Si", "7"]
    # configuration 1, the rows of 2 (its configuration line lost), 3 and 4
    lines = (header + ["Direct configuration=     1"] + rows + rows
             + ["Direct configuration=     3"] + rows
             + ["Direct configuration=     4"] + rows)
    traj = read_trajectory(_write(tmp_path, "XDATCAR", "\n".join(lines) + "\n"))
    assert any(n.startswith("fixed cell") for n in traj.notes)
    assert traj.timesteps.tolist() == [1, 3, 4]
    assert traj.skipped == {1: "7 coordinate lines with no configuration line "
                               "before them (a configuration line is missing)"}
    _frames(traj)


def test_xdatcar_trailing_lines_and_a_cut_last_row(tmp_path):
    lines = _xdatcar(2)
    traj = read_trajectory(_write(tmp_path, "XDATCAR",
                                  "\n".join(lines + ["0.1 0.1 0.1"]) + "\n"))
    assert traj.skipped == {} and traj.n_frames == 2
    assert any("1 line(s) after the last configuration are not read" in n
               for n in traj.notes)
    cut = lines[:-1] + [" ".join(lines[-1].split()[:2])]
    traj = read_trajectory(_write(tmp_path, "XDATCAR", "\n".join(cut) + "\n"))
    assert traj.skipped == {1: "truncated: the last coordinate line holds 2 of "
                               "3 values"}


def test_fused_fixed_width_vasp_numbers_are_split_at_the_sign(tmp_path):
    """VASP writes coordinates in fixed-width fields, so a negative value
    that fills its field touches the one before ('0.6-0.3'; pymatgen's
    XDATCAR.bad_fmt has such a row). It used to raise FrameError. -0.3 is
    0.7 wrapped, and 8.5D-01 is Fortran for 0.85, so the frame is the model."""
    text = (DATA / "XDATCAR_fixed").read_text()
    assert "0.6 0.7 0.2" in text and "0.85 0.4 0.9" in text
    text = text.replace("0.6 0.7 0.2", "0.6-0.3 0.2").replace(
        "0.85 0.4 0.9", "8.5D-01 0.4 0.9")
    frames = _frames(read_trajectory(_write(tmp_path, "XDATCAR", text)))
    assert _gap_mod_one(frames[0].frac, _vasp_order(GEN.FRAC[0])) <= TOL_ANG
    assert any("split at each sign" in n for n in frames[0].notes)
    with pytest.raises(FrameError, match="coordinate line 3.*does not hold "
                                         "three numbers"):
        read_trajectory(_write(tmp_path, "XDATCAR", text.replace(
            "0.6-0.3 0.2", "0.6-0.3x 0.2"))).frame(0)


def test_history_records_cut_or_unreadable(tmp_path):
    """A title starting with 'timestep' is not a frame; a non-finite time
    skips its frame at indexing (it used to refuse the whole file, or fail
    when loaded); a last record with fewer values is a cut frame."""
    lines = (DATA / "HISTORY").read_text().splitlines()
    retitled = ["timestep 0 written as a title"] + lines[1:]
    traj = read_trajectory(_write(tmp_path, "HISTORY", "\n".join(retitled) + "\n"),
                           type_map=HISTORY_MAP)
    assert traj.n_frames == 2
    for value in ("inf", "nan"):
        text = "\n".join(lines).replace("0.001 0.5", f"0.001 {value}") + "\n"
        traj = read_trajectory(_write(tmp_path, "HISTORY", text),
                               type_map=HISTORY_MAP)
        assert traj.skipped == {1: "unreadable: the elapsed time holds a value "
                                   f"that is not finite ('{value}')"}
    cut = lines[:-1] + ["1.75 0.5"]
    traj = read_trajectory(_write(tmp_path, "HISTORY", "\n".join(cut) + "\n"),
                           type_map=HISTORY_MAP)
    assert traj.skipped == {1: "truncated: its last record holds 2 of 3 values"}


def test_dlpoly_record_2_with_more_than_megatm(tmp_path):
    """DL_POLY Classic writes other items third on record 2 (the KCl test
    CONFIG: '2 3 2000 0.5000000000E-02' over 216 atoms), which used to be
    refused as an atom count. Exactly three integers is DL_POLY 4's megatm,
    still held to the atom blocks."""
    text = (DATA / "CONFIG").read_text()
    traj = read_trajectory(_write(tmp_path, "CONFIG", text.replace(
        "1 3 5", "1 3 2000 0.5000000000E-02", 1)))
    assert any("its third value 2000 is not the 5 atom blocks" in n
               for n in traj.notes)
    _check_against_model(_frames(traj), origin=[-0.5 * GEN.BOX[0].sum(axis=0)])
    with pytest.raises(ValueError, match="record 2 gives 7 atoms"):
        read_trajectory(_write(tmp_path, "CONFIG", text.replace("1 3 5", "1 3 7", 1)))


# ---------------------------------------------------------------------------
# labels, masses and units as the student's files write them
# ---------------------------------------------------------------------------

def test_force_field_names_are_not_case_folded_into_elements(tmp_path):
    """'HO' (a hydroxyl hydrogen), 'CA' (a carbon) and 'SI' used to become Ho,
    Ca and Si; a label is an element only as files write symbols. A user map
    still reads them."""
    lines = (DATA / "HISTORY").read_text().replace("Na 7", "NA 7")
    path = _write(tmp_path, "HISTORY", lines)
    with pytest.raises(ValueError, match="'NA'.*matches the symbol Na only in "
                                         "another capitalisation"):
        read_trajectory(path, type_map=HISTORY_MAP)
    traj = read_trajectory(path, type_map={**HISTORY_MAP, "NA": "Na"})
    assert traj.frame(0).elements.tolist() == GEN.ELEMENTS
    config = (DATA / "CONFIG").read_text().replace("Si 2", "SI 2")
    with pytest.raises(ValueError, match="'SI'.*another capitalisation"):
        read_trajectory(_write(tmp_path, "CONFIG", config))


def test_a_label_the_files_own_mass_contradicts_is_refused(tmp_path):
    """'Ho' with the mass of hydrogen used to be read as holmium, the 164 amu
    difference only noted. A mass nearer another element than the label's
    refuses the label; an older table's mass or a whole-number mass does not."""
    text = (DATA / "HISTORY").read_text().replace("Si 2 28.0855", "Ho 2 1.00794")
    with pytest.raises(ValueError, match="'Ho'.*reads as Ho, and the file gives "
                                         "its mass as 1.00794 amu, nearer to H"):
        read_trajectory(_write(tmp_path, "HISTORY", text), type_map=HISTORY_MAP)
    whole = (DATA / "HISTORY").read_text().replace("Si 2 28.0855", "Si 2 28.0")
    assert read_trajectory(_write(tmp_path, "HISTORY", whole),
                           type_map=HISTORY_MAP).frame(0).composition \
        == {"Na": 1, "O": 3, "Si": 1}


def _labelled_data(labels: tuple[str, str], masses: tuple[str, str]) -> str:
    return ("labels\n\n3 atoms\n2 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n"
            "0 10 zlo zhi\n\nAtom Type Labels\n\n"
            f"1 {labels[0]}\n2 {labels[1]}\n\nMasses\n\n1 {masses[0]}\n"
            f"2 {masses[1]}\n\nAtoms # atomic\n\n1 1 1 1 1\n2 2 2 1 1\n"
            "3 2 1 2 1\n")


def test_lammps_type_labels_against_the_masses(tmp_path):
    """ClayFF writes 'st' and 'ho'; 'ho' used to become Ho. Now the label is
    passed over (noted) and the mass names hydrogen. A label written as a
    symbol that the type's own mass contradicts is refused."""
    traj = read_trajectory(_write(tmp_path, "clayff.data",
                                  _labelled_data(("st", "ho"), ("28.0855", "1.008"))))
    assert traj.type_map == {1: "Si", 2: "H"}
    assert any("type 2: H from the data-file masses" in n
               and "another capitalisation" in n for n in traj.notes)
    with pytest.raises(ValueError, match="type 2 .*reads as Ho, and the data "
                                         "file gives its mass as 1.008 amu"):
        read_trajectory(_write(tmp_path, "ho.data",
                               _labelled_data(("Si", "Ho"), ("28.0855", "1.008"))))


def test_mass_comparisons_follow_the_written_decimals():
    """72.64 - 72.63 is 0.010000000000005 in binary, so Ge written to IUPAC
    2013 was refused with 'nearest: Ge, 0.01 amu away' beside a 0.01
    tolerance; 208.99, 0.01 from both Bi and Po, was read as Po by rounding.
    A difference outside the tolerance is never printed as equal to it."""
    assert type_map_from_masses({1: 72.63, 2: 95.94, 3: 65.39})[0] == \
        {1: "Ge", 2: "Mo", 3: "Zn"}
    with pytest.raises(ValueError, match=r"within 0.01 amu of Po \(209 amu\) and "
                                         r"of Bi"):
        type_map_from_masses({1: 208.99})
    with pytest.raises(ValueError, match="Se, 78.96 amu, 0.011 amu away"):
        type_map_from_masses({1: 78.971})
    with pytest.raises(ValueError, match="Ge, 72.64 amu, more than 0.01 amu away"):
        type_map_from_masses({1: 72.64 - 0.0100001})
    _, evidence = type_map_from_masses({1: 208.9804})
    assert "gemmi lists Po at the whole number 209 amu" in evidence[0]


def test_lammps_extxyz_in_units_real(tmp_path):
    """LAMMPS's own extxyz dump writes Time= in fs and velocities in Å/fs
    under units real; they were stored 1000 times off and units= was refused
    for extxyz. units='real' converts them as for a dump."""
    traj = read_trajectory(DATA / "glass.extxyz", units="real")
    assert traj.times_ps.tolist() == [t * 1e-3 for t in GEN.TIMES_PS]
    for k, frame in enumerate(_frames(traj)):
        assert np.array_equal(frame.vel_ang_per_ps, GEN.VELOCITY[k] * 1e3)
    plain = read_trajectory(DATA / "glass.extxyz")
    assert "fs and Å/fs under units real" in plain.units_note


def test_ase_extxyz_charges_and_momenta(tmp_path):
    """ASE writes per-atom charges as initial_charges (read) and momenta
    rather than velocities (named in the notes, with why they are not
    converted: that needs the masses ASE used and ASE's unit of time)."""
    text = ('2\nLattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3:'
            'momenta:R:3:initial_charges:R:1 pbc="T T T"\n'
            "Na 0 0 0 1 2 3 0.6\nCl 2.5 2.5 2.5 -1 -2 -3 -0.6\n")
    traj = read_trajectory(_write(tmp_path, "ase.extxyz", text))
    assert traj.charges_e == {"Na": 0.6, "Cl": -0.6}
    assert traj.frame(0).vel_ang_per_ps is None
    assert any("momenta:R:3 (ASE writes momenta" in n for n in traj.notes)


def test_the_reordered_properties_file_is_the_same_model():
    """Properties= with id first and pos before species was sniffed as
    nothing, so read_trajectory refused a file read_extxyz read."""
    a, b = _frames(_open("glass.extxyz")), _frames(_open("glass_reordered.extxyz"))
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.cart_ang, y.cart_ang)
        assert np.array_equal(x.vel_ang_per_ps, y.vel_ang_per_ps)


def test_type_labels_in_the_type_column(tmp_path):
    """dump_modify types labels writes the type label in the type column;
    it used to be refused as a type that is not a whole number."""
    rows = ["1 Si 1.0 1.0 1.0", "2 O 2.5 1.0 1.0", "3 O 1.0 2.5 1.0"]
    traj = read_trajectory(_write(tmp_path, "labels.lammpstrj",
                                  _dump_text([(0, rows)])))
    assert traj.type_map_source == "type label"
    assert traj.frame(0).elements.tolist() == ["Si", "O", "O"]
    assert any("dump_modify types labels" in n for n in traj.notes)


# ---------------------------------------------------------------------------
# dumps across frames, and what the notes say
# ---------------------------------------------------------------------------

def test_a_dump_without_ids_that_lammps_resorted(tmp_path):
    """LAMMPS re-sorts atoms (atom_modify sort, on by default), so in a dump
    without ids row k is a different atom in each frame; every frame after a
    re-sort used to fail as atoms 'changing element'. Rows are put in element
    order, so frames load as structures; ids_track_atoms says atom k is not
    the same atom across frames, and a change of composition is named."""
    rows0 = ["1 1.0 1.0 1.0", "2 2.5 1.0 1.0", "2 1.0 2.5 1.0"]
    rows1 = [rows0[1], rows0[0], rows0[2]]
    path = _write(tmp_path, "noid.lammpstrj",
                  _dump_text([(0, rows0), (10, rows1)], columns="type x y z"))
    with read_trajectory(path, type_map={1: "Si", 2: "O"}) as traj:
        assert traj.ids_track_atoms is False
        frames = _frames(traj)
    for frame in frames:
        assert frame.elements.tolist() == ["O", "O", "Si"]
        assert sorted(map(tuple, frame.cart_ang.tolist())) == sorted(
            [(1.0, 1.0, 1.0), (2.5, 1.0, 1.0), (1.0, 2.5, 1.0)])
    assert _open("dump_tri_x.lammpstrj").ids_track_atoms is True
    rows2 = ["1 1.0 1.0 1.0", "1 2.5 1.0 1.0", "2 1.0 2.5 1.0"]
    path = _write(tmp_path, "noid2.lammpstrj",
                  _dump_text([(0, rows0), (10, rows2)], columns="type x y z"))
    with read_trajectory(path, type_map={1: "Si", 2: "O"}) as traj:
        with pytest.raises(FrameError, match="matched between frames by "
                                             "element only"):
            traj.frame(1)


def test_timesteps_that_repeat_or_go_back_are_noted(tmp_path):
    """A restart appended to the same file writes earlier timesteps again;
    the frames are kept in file order and the note says where."""
    steps = [0, 100, 200, 100, 200, 300]
    traj = read_trajectory(_write(tmp_path, "back.lammpstrj",
                                  _dump_text([(s, ROWS) for s in steps])),
                           type_map={1: "Si", 2: "O"})
    assert traj.timesteps.tolist() == steps
    assert any("timestep 200 is followed by 100 at file position 3" in n
               for n in traj.notes)
    traj = read_trajectory(_write(tmp_path, "same.lammpstrj",
                                  _dump_text([(0, ROWS), (100, ROWS), (100, ROWS)])),
                           type_map={1: "Si", 2: "O"})
    assert any("timestep 100 repeats at file position 2" in n for n in traj.notes)


def test_the_columns_read_are_the_columns_used(tmp_path):
    """'columns read' listed vx without vy vz, and ix iy iz beside xu yu zu,
    although neither was used."""
    rows = ["1 1 1.0 1.0 1.0 0 0 0 0.5", "2 2 2.5 1.0 1.0 0 0 0 0.5",
            "3 2 1.0 2.5 1.0 0 0 0 0.5"]
    traj = read_trajectory(_write(tmp_path, "cols.lammpstrj", _dump_text(
        [(0, rows)], columns="id type xu yu zu ix iy iz vx")),
        type_map={1: "Si", 2: "O"})
    assert "columns read: id type xu yu zu; not read: ix iy iz (unwrapped " \
           "positions come from xu yu zu), vx (velocities need all of vx vy " \
           "vz)" in traj.notes
    assert traj.frame(0).vel_ang_per_ps is None


def test_a_mapped_type_absent_from_frame_0_is_kept(tmp_path):
    """The user's type 3 used to be dropped as 'no atom has those types' when
    frame 0 had none, and a later frame holding it was refused."""
    rows3 = ["1 1 1.0 1.0 1.0", "2 2 2.5 1.0 1.0", "3 3 1.0 2.5 1.0"]
    traj = read_trajectory(_write(tmp_path, "late.lammpstrj", _dump_text(
        [(0, ROWS), (10, ROWS)])), type_map=TYPES)
    assert traj.type_map == TYPES
    assert any("type(s) [3] do not occur in frame 0" in n for n in traj.notes)
    assert not any("no atom has those types" in n for n in traj.notes)
    path = _write(tmp_path, "new.lammpstrj", _dump_text([(0, ROWS), (10, rows3)]))
    with pytest.raises(FrameError, match=r"type\(s\) \[3\] occur in this frame"):
        read_trajectory(path, type_map={1: "Si", 2: "O"}).frame(1)


def test_box_bounds_with_an_unnamed_third_value(tmp_path):
    """'0 10 2.5' under BOX BOUNDS pp pp pp: the tilt was dropped and the box
    read as orthogonal. The frame is skipped with the reason."""
    text = _dump_text([(0, ROWS)]) + _dump_text(
        [(10, ROWS)], bounds=("0 10 2.5", "0 10", "0 10"))
    traj = read_trajectory(_write(tmp_path, "tilt.lammpstrj", text),
                           type_map={1: "Si", 2: "O"})
    assert traj.skipped == {1: "unreadable: a line of ITEM: BOX BOUNDS holds 3 "
                               "values where 2 are expected (the header names no "
                               "tilt factors xy xz yz)"}


@pytest.mark.parametrize("name", ["dump_tri_x.lammpstrj", "glass.extxyz",
                                  "data_atomic.data", "XDATCAR_fixed"])
def test_a_byte_order_mark_is_ignored(tmp_path, name):
    """PowerShell's Out-File and Windows editors write a UTF-8 byte-order
    mark; a dump or extxyz starting with one was 'not recognised'."""
    path = tmp_path / name
    path.write_bytes(b"\xef\xbb\xbf" + (DATA / name).read_bytes())
    assert sniff_md(path) == FILES[name][1]
    for a, b in zip(_frames(_open(name)),
                    _frames(read_trajectory(path, **FILES[name][0])), strict=True):
        assert np.array_equal(a.cart_ang, b.cart_ang)


def test_cr_only_line_endings_are_refused_with_the_reason(tmp_path):
    path = tmp_path / "mac.lammpstrj"
    path.write_bytes((DATA / "dump_tri_x.lammpstrj").read_bytes().replace(
        b"\n", b"\r"))
    with pytest.raises(ValueError, match="CR alone"):
        read_trajectory(path, type_map=TYPES)


@pytest.mark.parametrize("name", sorted(FILES))
def test_the_load_record_is_the_first_note(name):
    """Step 1 asks for the path, atoms, frames, box and composition to be
    recorded on load; describe() computed them on request only, so an export
    copying the notes lost them."""
    traj = _open(name)
    first = traj.notes[0]
    assert first.startswith(f"read {traj.source_path} as {traj.file_format}: "
                            f"{traj.n_atoms} atoms x {traj.n_frames} frame(s)")
    assert "volume" in first and "composition (frame 0)" in first


def test_frame_0_problems_name_the_file_and_the_frame(tmp_path):
    rows = ["1 1 1.0 1.0 1.0", "1 2 2.5 1.0 1.0", "3 2 1.0 2.5 1.0"]
    path = _write(tmp_path, "dup.lammpstrj", _dump_text([(0, rows), (10, ROWS)]))
    with pytest.raises(FrameError, match=r"dup\.lammpstrj, frame 0 \(file "
                                         r"position 0\): 1 atom id\(s\) occur "
                                         r"more than once"):
        read_trajectory(path, type_map={1: "Si", 2: "O"})


def test_data_file_velocity_sections_cut_or_with_repeated_ids(tmp_path):
    """A file cut after its Velocities keyword raised IndexError; repeated
    Atoms ids were blamed on the Velocities section."""
    text = (DATA / "data_charge.data").read_text()
    cut = text[:text.index("Velocities") + len("Velocities\n")]
    with pytest.raises(ValueError, match="the Velocities section is empty"):
        read_trajectory(_write(tmp_path, "cut.data", cut))
    repeated = text.replace("\n4 2 -1.2 ", "\n2 2 -1.2 ")
    with pytest.raises(ValueError, match=r"lists atom id\(s\) \[2\] more than "
                                         "once"):
        read_trajectory(_write(tmp_path, "dup.data", repeated))


@pytest.mark.parametrize("name", ["XDATCAR_fixed", "CONFIG", "glass.extxyz"])
def test_type_numbers_in_a_map_for_a_labelled_file_are_noted(name):
    traj = read_trajectory(DATA / name, type_map={1: "K", 2: "S"})
    assert any("type map entries [1, 2] are type numbers" in n
               for n in traj.notes)


def test_a_gzipped_single_frame_extxyz_goes_to_the_md_reader(tmp_path):
    """The crystal reader reads plain text only, so readers.read used to say
    'not recognised' for a file read_trajectory reads."""
    lines = (DATA / "glass.extxyz").read_text().splitlines()[:7]
    path = tmp_path / "one.xyz.gz"
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0))
    with pytest.raises(readers.MDModelFile, match="gzip-compressed"):
        readers.read(path)
    assert read_trajectory(path).n_frames == 1


def test_gzip_frames_read_from_several_threads(tmp_path):
    """A gzip trajectory shares one decompressing stream; a lock keeps two
    threads from interleaving their seeks on it. Every frame read from four
    threads at once equals the frame read alone."""
    from concurrent.futures import ThreadPoolExecutor

    raw = (DATA / "dump_tri_x.lammpstrj").read_bytes()
    path = tmp_path / "many.lammpstrj.gz"
    # 40 frames: the two model frames twenty times, timesteps 0 .. 39.
    text = b"".join(raw.replace(b"ITEM: TIMESTEP\n0\n",
                                f"ITEM: TIMESTEP\n{2 * k}\n".encode()).replace(
        b"ITEM: TIMESTEP\n500\n", f"ITEM: TIMESTEP\n{2 * k + 1}\n".encode())
        for k in range(20))
    path.write_bytes(gzip.compress(text, mtime=0))
    traj = read_trajectory(path, type_map=TYPES)
    alone = [traj.frame(k).cart_ang for k in range(len(traj))]
    order = [k for _ in range(3) for k in (*range(len(traj)),
                                           *reversed(range(len(traj))))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(lambda k: (k, traj.frame(k).cart_ang), order))
    traj.close()
    assert all(np.array_equal(cart, alone[k]) for k, cart in got)


# ---------------------------------------------------------------------------
# found by the re-check after the fixes
# ---------------------------------------------------------------------------

def _gzip_cut(raw: bytes) -> bytes:
    """A gzip member holding ``raw``, flushed but with no end-of-stream marker
    and no trailer: what a copy taken while the run was writing leaves."""
    compressor = zlib.compressobj(9, zlib.DEFLATED, 31)
    return compressor.compress(raw) + compressor.flush(zlib.Z_SYNC_FLUSH)


@pytest.mark.parametrize("name, marker", [
    ("dump_tri_x.lammpstrj", b"ITEM: TIMESTEP"),
    ("glass.extxyz", b"5\nLattice"),
    ("XDATCAR_fixed", b"Direct configuration"),
    ("HISTORY", b"timestep"),
])
def test_a_gzip_stream_cut_at_a_frame_boundary_is_noted(tmp_path, name, marker):
    """Cut where the second frame starts, the stream leaves frame 0 complete.
    The dump reader said so in a note; XDATCAR and HISTORY read the cut file
    as if it were whole, with no word that frames after the cut are gone."""
    raw = (DATA / name).read_bytes().replace(b"\r\n", b"\n")
    second = raw.find(marker, raw.find(marker) + 1)
    if name == "XDATCAR_fixed":
        second = raw.rfind(b"\n", 0, second) + 1
    path = tmp_path / name
    path.write_bytes(_gzip_cut(raw[:second]))
    traj = read_trajectory(path, **FILES[name][0])
    assert traj.n_frames == 1
    assert any("the gzip stream ends before its end-of-stream marker" in n
               and "frames after the cut are not in the file" in n
               for n in traj.notes)
    _frames(traj)


def test_blank_lines_between_extxyz_frames_make_one_note(tmp_path):
    """A writer that puts a blank line after every frame gave one note per
    frame (1000 notes for 1000 frames). One note now counts them and names
    the first positions."""
    frame = ('2\nLattice="10 0 0 0 10 0 0 0 10"\nSi 1 1 1\nO 2 2 2\n')
    traj = read_trajectory(_write(tmp_path, "blank.extxyz",
                                  "\n".join([frame] * 12)))
    assert traj.n_frames == 12 and traj.skipped == {}
    blank = [n for n in traj.notes if "blank line" in n]
    assert blank == ["11 blank line(s) between frames, before file position(s) "
                     "1, 2, 3, 4, 5 and 6 more, are passed over"]


def test_the_elements_note_names_the_column_the_elements_come_from(tmp_path):
    """With Z listed before species in Properties=, the elements came from
    the species column (as read_extxyz documents) while 'columns read' named
    Z, and a Z column naming other elements went unremarked."""
    head = '2\nLattice="10 0 0 0 10 0 0 0 10" Properties=Z:I:1:species:S:1:pos:R:3\n'
    traj = read_trajectory(_write(tmp_path, "z.extxyz",
                                  head + "14 Na 1 1 1\n8 O 2 2 2\n"))
    assert traj.frame(0).elements.tolist() == ["Na", "O"]
    assert "columns read: species:S:1 pos:R:3; not read: Z:I:1 (elements come " \
           "from species:S:1)" in traj.notes
    assert any("the Z column names another element than the species column for "
               "1 of 2 atoms" in n and "Z = 14, Si" in n for n in traj.notes)
    agree = read_trajectory(_write(tmp_path, "z2.extxyz",
                                   head + "11 Na 1 1 1\n8 O 2 2 2\n"))
    assert not any("the Z column" in n for n in agree.notes)


def test_an_extxyz_masses_column_checks_the_species(tmp_path):
    """The module states that a file's own masses check its labels; the
    extended XYZ reader ignored a masses column, so 'Ho' at 1.008 amu read
    as holmium. It is refused as in a data file; a heavier isotope's mass
    (D written as H) is read and noted; mass_tol_amu reaches the reader."""
    head = ('2\nLattice="10 0 0 0 10 0 0 0 10" '
            'Properties=species:S:1:pos:R:3:masses:R:1\n')
    path = _write(tmp_path, "ho.extxyz", head + "Si 1 1 1 28.0855\nHo 2 2 2 1.008\n")
    with pytest.raises(ValueError, match=r"'Ho' \(mass 1\.008 amu in the file\): "
                                         r"reads as Ho, and the file gives its mass "
                                         r"as 1\.008 amu, nearer to H"):
        read_trajectory(path)
    traj = read_trajectory(_write(tmp_path, "d.extxyz",
                                  head + "Si 1 1 1 28.0855\nH 2 2 2 2.014\n"),
                           mass_tol_amu=0.01)
    assert traj.frame(0).elements.tolist() == ["Si", "H"]
    assert any("species 'H' is read as H" in n and "2.014 amu" in n
               for n in traj.notes)
    assert "columns read: species:S:1 pos:R:3 masses:R:1" in traj.notes


def test_data_file_sections_a_fix_defines(tmp_path):
    """read_data's fix keyword hands sections named by the input script to a
    fix, such as the core/shell model's CS-Info (atom-ID, core/shell ID).
    After Atoms, Velocities or Masses its rows were read as rows of that
    section and the file refused ('Atoms lines have [1, 2, 9] values').
    Each placement now reads the same model and names the section."""
    base = (DATA / "data_charge.data").read_text().replace("\r\n", "\n")
    cs_info = "CS-Info\n\n7 1\n2 2\n9 3\n5 4\n4 5\n\n"
    reference = _frames(_open("data_charge.data"))[0]
    for text in (base.replace("Atoms # charge", cs_info + "Atoms # charge"),
                 base.replace("Velocities", cs_info + "Velocities"),
                 base.rstrip("\n") + "\n\n" + cs_info):
        traj = read_trajectory(_write(tmp_path, "cs.data", text))
        frame = _frames(traj)[0]
        assert np.array_equal(frame.cart_ang, reference.cart_ang)
        assert np.array_equal(frame.elements, reference.elements)
        assert np.array_equal(frame.vel_ang_per_ps, reference.vel_ang_per_ps)
        assert any(n.startswith("section(s) CS-Info are not read_data keywords")
                   for n in traj.notes)
    # a row of the Atoms section is still a row, and a bad one still refused
    bad = base.replace("\n9 2 -1.2 ", "\nx9 2 -1.2 ")
    with pytest.raises(ValueError, match="atom-ID column"):
        read_trajectory(_write(tmp_path, "bad.data", bad))


def test_masses_from_a_file_cut_in_its_masses(tmp_path):
    """The data file named by masses_from was not checked for a cut last
    line, which a file holding only a header and Masses ends with."""
    text = "masses\n\n2 atom types\n\nMasses\n\n1 28.0855\n2 15.9994"
    dump = _write(tmp_path, "d.lammpstrj", _dump_text([(0, ROWS)]))
    with pytest.raises(ValueError, match=r"m\.data \(masses_from\): the file does "
                                         r"not end with a line ending"):
        read_trajectory(dump, masses_from=_write(tmp_path, "m.data", text))
    traj = read_trajectory(dump, masses_from=_write(tmp_path, "m.data",
                                                    text + "\n"))
    assert traj.type_map == {1: "Si", 2: "O"}


def test_an_atomic_number_beyond_a_c_int_is_refused_in_words(tmp_path):
    """gemmi.Element raises TypeError for Z = 10**12, which escaped
    read_trajectory (the readers' contract allows ValueError and OSError);
    it is refused as any other number that is not an atomic number."""
    text = ('2\nLattice="10 0 0 0 10 0 0 0 10" Properties=Z:I:1:pos:R:3\n'
            "14 1 1 1\n1000000000000 2 2 2\n")
    with pytest.raises(FrameError, match=r"z\.extxyz, frame 0 .*Z = "
                                         r"1000000000000 is not an atomic number"):
        read_trajectory(_write(tmp_path, "z.extxyz", text))


def test_a_gzip_compressed_xyz_named_xyz_goes_to_the_md_reader(tmp_path):
    """Named .xyz, a gzip file went to the crystal read_xyz, which reads
    plain text only and raised on the compressed bytes; named .xyz.gz it
    already went to the MD reader."""
    lines = (DATA / "glass.extxyz").read_text().splitlines()[:7]
    path = tmp_path / "one.xyz"
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0))
    with pytest.raises(readers.MDModelFile, match="gzip-compressed"):
        readers.read(path)
    assert read_trajectory(path).n_frames == 1


# ---------------------------------------------------------------------------
# no verdicts, and no Qt
# ---------------------------------------------------------------------------

VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def _every_note_and_message(tmp_path) -> list[str]:
    out: list[str] = []
    for name in FILES:
        traj = _open(name)
        out += traj.describe()
        out += [n for f in _frames(traj) for n in f.notes]
    out += read_trajectory(DATA / "data_atomic.data",
                           type_map={1: "Si", 2: "F", 3: "Na"}).notes
    out += read_trajectory(_velocity_dump(tmp_path), type_map=TYPES).describe()

    def message(call):
        try:
            call()
        except (ValueError, OSError) as error:
            return str(error)
        raise AssertionError("expected a refusal")

    out.append(message(lambda: read_trajectory(DATA / "dump_tri_x.lammpstrj")))
    out.append(message(lambda: read_trajectory(DATA / "HISTORY")))
    out.append(message(lambda: type_map_from_masses({1: 28.0, 2: 208.99})))
    out.append(message(lambda: readers.read(DATA / "CONFIG")))
    out.append(message(lambda: readers.read(DATA / "glass.extxyz")))
    out.append(message(lambda: read_trajectory(
        _velocity_dump(tmp_path, "lj"), type_map=TYPES)))
    # the refusals and notes added for damaged, cut and mislabelled files
    damaged = tmp_path / "damaged.lammpstrj.gz"
    damaged.write_bytes(_gzip_damage(
        (DATA / "dump_tri_x.lammpstrj").read_bytes())["crc"])
    out.append(message(lambda: read_trajectory(damaged, type_map=TYPES)))
    cut = tmp_path / "cut.data"
    cut.write_bytes((DATA / "data_atomic.data").read_bytes()[:-3])
    out.append(message(lambda: read_trajectory(cut)))
    upper = _write(tmp_path, "HISTORY", (DATA / "HISTORY").read_text()
                   .replace("Na 7", "NA 7").replace("Si 2 28.0855", "Ho 2 1.00794"))
    out.append(message(lambda: read_trajectory(upper, type_map=HISTORY_MAP)))
    two = _write(tmp_path, "two.xyz", "1\na\nO 0 0 0\n1\nb\nO 0 0 0.1\n")
    out.append(message(lambda: readers.read(two)))
    out.append(message(lambda: read_trajectory(two)))
    out.append(message(lambda: type_map_from_masses({1: 72.64 - 0.0100001})))
    out.append(message(lambda: read_trajectory(_write(
        tmp_path, "seven.data", _data_text("Atoms", [
            "1 7 1 2.4 1 1 1", "2 7 2 -1.2 2.5 1 1", "3 8 2 -1.2 1 2.5 1"])))))
    out += read_trajectory(_write(tmp_path, "clayff.data", _labelled_data(
        ("st", "ho"), ("28.0855", "1.008")))).notes
    out += read_trajectory(DATA / "glass.extxyz", units="real").describe()
    out += read_trajectory(DATA / "XDATCAR_fixed", type_map={1: "K"}).notes
    steps = [(s, ROWS) for s in (0, 100, 100, 50)]
    out += read_trajectory(_write(tmp_path, "back.lammpstrj", _dump_text(steps)),
                           type_map={1: "Si", 2: "O"}).notes
    noid = _write(tmp_path, "noid.lammpstrj", _dump_text(
        [(0, ["1 1.0 1.0 1.0 0.5", "2 2.5 1.0 1.0 0.5"])],
        columns="type x y z vx"))
    out += read_trajectory(noid, type_map={1: "Si", 2: "O"}).notes
    config = (DATA / "CONFIG").read_text().replace("1 3 5", "1 3 2000 0.5E-02", 1)
    out += read_trajectory(_write(tmp_path, "CONFIG", config)).notes
    # the notes and refusals added by the re-check after the fixes
    raw = (DATA / "HISTORY").read_bytes().replace(b"\r\n", b"\n")
    cut = tmp_path / "cut" / "HISTORY"
    cut.parent.mkdir()
    cut.write_bytes(_gzip_cut(raw[:raw.find(b"timestep", raw.find(b"timestep") + 1)]))
    out += read_trajectory(cut, type_map=HISTORY_MAP).notes
    frame = '2\nLattice="10 0 0 0 10 0 0 0 10"\nSi 1 1 1\nO 2 2 2\n'
    out += read_trajectory(_write(tmp_path, "blank.extxyz",
                                  "\n".join([frame] * 3))).notes
    head = ('2\nLattice="10 0 0 0 10 0 0 0 10" '
            'Properties=Z:I:1:species:S:1:pos:R:3:masses:R:1\n')
    out += read_trajectory(_write(tmp_path, "zm.extxyz", head + "14 Na 1 1 1 "
                                  "22.99\n8 H 2 2 2 2.014\n")).notes
    out.append(message(lambda: read_trajectory(_write(
        tmp_path, "ho.extxyz", head + "67 Ho 1 1 1 1.008\n8 O 2 2 2 16.0\n"))))
    text = (DATA / "data_charge.data").read_text().replace(
        "Velocities", "CS-Info\n\n7 1\n2 2\n9 3\n5 4\n4 5\n\nVelocities")
    out += read_trajectory(_write(tmp_path, "cs.data", text)).notes
    out.append(message(lambda: read_trajectory(
        _write(tmp_path, "d.lammpstrj", _dump_text([(0, ROWS)])),
        masses_from=_write(tmp_path, "m.data", "m\n\nMasses\n\n1 28.0855"))))
    return out


def test_no_note_or_message_carries_a_verdict(tmp_path):
    notes = _every_note_and_message(tmp_path)
    assert len(notes) > 150
    assert [n for n in notes if VERDICT.search(n)] == []


def test_no_string_in_the_module_carries_a_verdict():
    """Every string literal in md_readers.py, docstrings included."""
    source = (ROOT / "facet" / "core" / "md_readers.py").read_text(
        encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 200
    assert [s for s in strings if VERDICT.search(s)] == []


def test_the_readers_import_without_qt():
    code = ("import sys, facet.core.md_readers, facet.core.readers;"
            "print('QT' if any(m.startswith('PySide6') for m in sys.modules) "
            "else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
