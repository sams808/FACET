"""The XTC reader: what the writers stored comes back, and nothing is lost.

The fixtures in tests/data/md were written by real programs
(tests/data/md/make_xtc_files.py): MDAnalysis 2.10.0's libmdaxdr (the xdrfile
C library) from closed-form positions, which these tests recompute from the
same script, and LAMMPS 22 Jul 2025 ``dump xtc`` beside a ``dump custom`` of
the same steps at 10 significant figures (LAMMPS's own positions). What is
pinned, and why:

* **Positions within the writer's rounding.** The writer stores
  round(x * precision) half away from zero, so a decoded position lies within
  0.5 / precision nm of what was written, plus float32 rounding of the
  written value (|x| x 2^-23 nm, taken twice). Every comparison uses that
  bound, never a looser one; LAMMPS's positions are compared modulo the box
  vectors, because XTC holds no box origin.
* **Every branch of the decompressor is exercised by a written file:** runs of
  1 to 8 small atoms, a repeated run length (flag 0), the small-difference
  width moving down and up, per-axis bit fields (ranges above 2^24), a packed
  field of 65 bits and small fields of 65 bits (the scalar decoder), the
  writer's starting width of 73 (one past its table), and plain floats for
  nine atoms or fewer. The numpy decoder and the scalar decoder agree on every
  fixture, and the numpy bit extraction equals the scalar one on random bytes.
* **Nothing is dropped silently.** A file cut inside a frame, garbage between
  frames, a damaged magic number (the first frame's too) or byte count (one
  enlarged onto a later frame's header included), a frame with another atom
  count, a non-finite box: each lost frame has a position in ``skipped`` with
  the reason, and the frames around it read. A frame whose only non-finite
  value is its time is kept, its time unknown. A compressed block whose
  damage breaks the stream's structure raises FrameError naming the file and
  the frame; XTC holds no checksum, so damage that keeps the structure
  decodes, and the module docstring says so (pinned here).
* **A damaged header costs no memory:** a header stating more atoms than its
  block can hold at 2 bits per atom is a counted gap, and a type_map is
  checked without arrays the size of the stated count.
* **Elements come only from what is given:** a topology (a file, a
  Trajectory or a Frame: elements, ids and charges, row k for row k) or a map
  of every atom number; a topology without atom ids, a topology of another
  atom count, a crystal structure and a map of LAMMPS types are refused. A
  topology at a step an XTC frame shares is compared with it row by row.
* **What the file does not record is said:** the boundary (``periodic`` is the
  topology's, else None), a unit change by dump_modify sfactor / tfactor, and
  frames of more than one precision.
* **md_readers dispatches** an XTC file here and readers.read routes it.
* **Every refusal names the file and says what to pass.**
* **No verdicts** in any note, message or string of the module.
"""
from __future__ import annotations

import ast
import gzip
import importlib.util
import pickle
import re
import struct
import subprocess
import sys
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest

from facet.core import md_formats_xtc as X
from facet.core import md_readers, readers
from facet.core.md_formats_base import KNOWN_OPTIONS, FormatSpec
from facet.core.md_model import FrameError, frame_from_arrays
from facet.core.readers import UnsupportedFormat

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data" / "md"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_xtc_files", DATA / "make_xtc_files.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _generator()
GLASS = DATA / "xtc_mda_glass.xtc"
GLASS_ELEMENTS = ["Si" if i % 3 == 0 else "O" for i in range(30)]
GLASS_MAP = {i + 1: e for i, e in enumerate(GLASS_ELEMENTS)}
LAMMPS_TYPES = {1: "Si", 2: "Na", 3: "O"}


def _bound_nm(written_nm: np.ndarray, precision: float) -> float:
    """The writer's rounding: half a grid step plus float32 rounding of the
    written value and of value x precision."""
    return 0.5 / precision + 2 * float(np.abs(written_nm).max()) * 2.0 ** -23


def _mod_lattice(cart: np.ndarray, ref: np.ndarray, box: np.ndarray) -> float:
    frac = np.linalg.solve(box.T, (cart - ref).T).T
    frac -= np.round(frac)
    return float(np.abs(frac @ box).max())


def _blocks(path: Path) -> list[tuple[int, int]]:
    records, skipped = X._index(path)
    assert skipped == {}
    return [(r.offset, r.end) for r in records]


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _refusal(call) -> str:
    try:
        call()
    except (ValueError, OSError) as error:
        return str(error)
    raise AssertionError("expected a refusal")


def _walk(data: bytes):
    """The group walk of one compressed frame, and its full-atom width."""
    minint = struct.unpack_from(">3i", data, 60)
    maxint = struct.unpack_from(">3i", data, 72)
    smallidx = struct.unpack_from(">i", data, 84)[0]
    nbytes = struct.unpack_from(">i", data, 88)[0]
    sizes = [b - a + 1 for a, b in zip(minint, maxint)]
    if max(sizes) > 0xFFFFFF:
        bits = sum(s.bit_length() for s in sizes)
    else:
        bits = (sizes[0] * sizes[1] * sizes[2]).bit_length()
    natoms = struct.unpack_from(">i", data, 4)[0]
    groups = X._walk_groups(data[92:92 + nbytes] + bytes(128), natoms, bits,
                            smallidx, 8 * nbytes)
    return groups, bits, sizes, smallidx


# ---------------------------------------------------------------------------
# what the writers stored comes back
# ---------------------------------------------------------------------------

def test_mdanalysis_glass_positions_box_step_and_time():
    traj = X.read_xtc(GLASS, type_map=GLASS_MAP)
    assert (traj.n_atoms, traj.n_frames, traj.skipped) == (30, 3, {})
    assert traj.file_format == "gromacs-xtc"
    assert traj.timesteps.tolist() == [0, 100, 200]
    assert np.allclose(traj.times_ps, [0.0, 0.2, 0.4], rtol=0, atol=1e-7)
    assert traj.box_varies is True
    for k, frame in enumerate(traj):
        box_nm = GEN.GLASS_BOX * (1 + 0.01 * k)
        written = GEN.glass_positions(30, k, GEN.GLASS_BOX)
        assert np.abs(frame.box_ang - box_nm * 10).max() < 1e-5
        assert np.array_equal(frame.origin_ang, np.zeros(3))
        error_ang = _mod_lattice(frame.cart_ang, written * 10, frame.box_ang)
        assert error_ang <= 10 * _bound_nm(written, 1000.0)
        assert frame.elements.tolist() == GLASS_ELEMENTS
        assert frame.atom_id.tolist() == list(range(1, 31))
        assert frame.unwrapped_cart_ang is None and frame.vel_ang_per_ps is None


def test_runs_fixture_exercises_every_branch_of_the_decompressor():
    data = (DATA / "xtc_mda_runs.xtc").read_bytes()
    groups, _, _, _ = _walk(data)
    runs = np.bincount(groups.n_small)
    assert runs[1] and runs[2] and runs[8]           # 1, 2 ... 8 small atoms
    with_small = int(np.count_nonzero(groups.n_small))
    assert groups.run_changes < groups.large_pos.size  # some flags are 0
    assert with_small > groups.run_changes - runs[0]   # a run length repeated
    down, up = groups.idx_steps
    assert down > 0 and up > 0
    frame = X.decode_xtc_frame(data)
    assert np.array_equal(frame.ints, X.decode_xtc_frame(data, scalar=True).ints)
    written = GEN.chain_positions(240)
    assert np.abs(frame.coords_nm - written).max() <= _bound_nm(written, 1000.0)
    assert (frame.step, frame.precision) == (7, 1000.0)
    assert frame.time_ps == pytest.approx(1.5, abs=1e-7)


def test_fields_wider_than_64_bits_go_through_the_scalar_decoder():
    data = (DATA / "xtc_mda_wide.xtc").read_bytes()
    groups, bits, _, smallidx = _walk(data)
    assert bits == 65 and smallidx == 65 and groups.n_small.sum() > 0
    frame = X.decode_xtc_frame(data)
    assert np.array_equal(frame.ints, X.decode_xtc_frame(data, scalar=True).ints)
    written = GEN.spread_positions(12, 30.0)
    assert np.abs(frame.coords_nm - written).max() <= _bound_nm(written, 1e5)


def test_ranges_above_2_24_use_a_bit_field_per_axis():
    data = (DATA / "xtc_mda_large.xtc").read_bytes()
    _, _, sizes, smallidx = _walk(data)
    assert max(sizes) > 0xFFFFFF
    assert smallidx == 73                 # the writer's search ends past its table
    frame = X.decode_xtc_frame(data)
    assert np.array_equal(frame.ints, X.decode_xtc_frame(data, scalar=True).ints)
    written = GEN.spread_positions(12, 20.0)
    assert np.abs(frame.coords_nm - written).max() <= _bound_nm(written, 1e6)


def test_nine_atoms_or_fewer_are_plain_floats():
    path = DATA / "xtc_mda_small.xtc"
    traj = X.read_xtc(path, type_map={i: "O" for i in range(1, 7)})
    assert "2 frame(s) of 9 atoms or fewer store plain 32-bit floats" \
        in traj.units_note
    for k in range(2):
        start, end = _blocks(path)[k]
        frame = X.decode_xtc_frame(path.read_bytes()[start:end])
        assert frame.precision is None and frame.ints is None
        written = GEN.glass_positions(6, k, GEN.GLASS_BOX).astype(np.float32)
        assert np.array_equal(frame.coords_nm, written.astype(np.float64))


def test_lammps_dump_xtc_against_lammps_positions():
    traj = X.read_xtc(DATA / "xtc_lammps.xtc", topology=DATA / "xtc_lammps.data")
    ref = md_readers.read_trajectory(DATA / "xtc_lammps_ref.lammpstrj",
                                     type_map=LAMMPS_TYPES)
    assert (traj.n_atoms, traj.n_frames) == (24, 3)
    assert traj.type_map_source == "data-file masses"
    assert traj.timesteps.tolist() == ref.timesteps.tolist() == [0, 10, 20]
    assert np.abs(traj.times_ps - traj.timesteps * 0.001).max() < 1e-8
    assert traj.box_varies is True
    for k in range(3):
        mine, theirs = traj.frame(k), ref.frame(k)
        assert mine.atom_id.tolist() == theirs.atom_id.tolist()
        assert mine.elements.tolist() == theirs.elements.tolist()
        assert mine.composition == {"Na": 6, "O": 12, "Si": 6}
        assert np.abs(mine.box_ang - theirs.box_ang).max() < 1e-5
        bound_ang = 10 * _bound_nm(theirs.cart_ang / 10, 1000.0)
        assert _mod_lattice(mine.cart_ang, theirs.cart_ang,
                            theirs.box_ang) <= bound_ang
    assert any("no box origin" in n for n in traj.notes)
    assert traj.notes[0].startswith("read ")


def test_timestep_fs_rebuilds_the_times_and_refuses_a_conflict():
    path = DATA / "xtc_lammps.xtc"
    traj = X.read_xtc(path, topology=DATA / "xtc_lammps.data", timestep_fs=1.0)
    assert traj.times_ps.tolist() == [0.0, 0.01, 0.02]
    assert any("with no offset" in n for n in traj.notes)
    message = _refusal(lambda: X.read_xtc(
        path, topology=DATA / "xtc_lammps.data", timestep_fs=2.0))
    assert "xtc_lammps.xtc" in message and "timestep_fs=2.0" in message
    message = _refusal(lambda: X.read_xtc(
        path, topology=DATA / "xtc_lammps.data", timestep_fs=-1))
    assert "positive" in message


def test_a_gromacs_start_time_is_kept_as_an_offset(tmp_path):
    """GROMACS writes time = tinit + step x dt; shift every time by 5 ps."""
    raw = bytearray(GLASS.read_bytes())
    for start, _ in _blocks(GLASS):
        step = struct.unpack_from(">i", raw, start + 8)[0]
        struct.pack_into(">f", raw, start + 12, 5.0 + step * 0.002)
    path = _write(tmp_path, "tinit.xtc", bytes(raw))
    traj = X.read_xtc(path, type_map=GLASS_MAP, timestep_fs=2.0)
    assert np.abs(traj.times_ps - [5.0, 5.2, 5.4]).max() < 1e-6
    assert any("plus the offset 5" in n for n in traj.notes)


# ---------------------------------------------------------------------------
# the numpy decoder against the scalar one
# ---------------------------------------------------------------------------

def test_magicints_table_is_the_formats():
    table = X._MAGICINTS
    assert len(table) == 73 and table[:9] == (0,) * 9
    assert table[9] == 8 and table[-1] == 2 ** 24
    for index in range(9, 73):
        assert table[index] ** 3 <= 2 ** index      # three values fit in index bits
    assert all(a < b for a, b in zip(table[9:], table[10:]))


def test_numpy_bit_fields_equal_the_scalar_reads_on_random_bytes():
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, 400, dtype=np.uint8).tobytes() + bytes(16)
    octets = np.frombuffer(data, dtype=np.uint8)
    for width in range(1, 65):
        starts = rng.integers(0, 8 * 380, 50)
        plain = X._fields(octets, starts, width)
        packed = X._packed_values(plain, width)
        assert plain.tolist() == [X._bits_at(data, int(s), width) for s in starts]
        assert packed.tolist() == [X._packed_at(data, int(s), width)
                                   for s in starts]


def _hand_encoded_frame(ints: np.ndarray, minint, maxint) -> bytes:
    """An XTC frame whose atoms are all written in full (run flag 0), encoded
    here from the format's description: (x * size_y + y) * size_z + z in
    bitsize bits, bytes least significant first, each byte most significant
    bit first, the last partial byte holding the high bits."""
    sizes = [b - a + 1 for a, b in zip(minint, maxint)]
    width = (sizes[0] * sizes[1] * sizes[2]).bit_length()
    full, rest = (width - 1) // 8, width - 8 * ((width - 1) // 8)
    bits: list[int] = []
    for x, y, z in ints - np.asarray(minint):
        value = (int(x) * sizes[1] + int(y)) * sizes[2] + int(z)
        for j in range(full):
            byte = (value >> (8 * j)) & 0xFF
            bits += [(byte >> (7 - t)) & 1 for t in range(8)]
        top = value >> (8 * full)
        bits += [(top >> (rest - 1 - t)) & 1 for t in range(rest)]
        bits.append(0)                                   # run flag: unchanged
    block = np.packbits(np.array(bits, dtype=np.uint8)).tobytes()
    natoms = len(ints)
    head = struct.pack(">iiif", 1995, natoms, 0, 0.0)
    head += np.diag([5.0, 5.0, 5.0]).astype(">f4").tobytes()
    head += struct.pack(">if3i3iii", natoms, 1000.0, *minint, *maxint, 9,
                        len(block))
    return head + block + bytes(-len(block) % 4)


def test_a_hand_encoded_block_decodes_and_an_out_of_range_atom_is_refused():
    rng = np.random.default_rng(3)
    minint, maxint = (-1200, 30, 7), (3999, 2600, 4100)
    ints = rng.integers(minint, np.add(maxint, 1), size=(40, 3))
    data = _hand_encoded_frame(ints, minint, maxint)
    for scalar in (False, True):
        frame = X.decode_xtc_frame(data, scalar=scalar)
        assert np.array_equal(frame.ints, ints)
    ints[17, 0] = maxint[0] + 1                 # one past the stated range
    wider = (maxint[0] + 1, *maxint[1:])
    widths = [int(np.prod(np.subtract(m, minint) + 1)).bit_length()
              for m in (maxint, wider)]
    assert widths[0] == widths[1]               # the same field width either way
    data = _hand_encoded_frame(ints, minint, wider)
    head = bytearray(data)
    struct.pack_into(">i", head, 72, maxint[0])  # the header states the old one
    for scalar in (False, True):
        with pytest.raises(ValueError, match="outside the range minint..maxint"):
            X.decode_xtc_frame(bytes(head), scalar=scalar)


def test_every_fixture_decodes_alike_with_both_decoders():
    for path in sorted(DATA.glob("xtc_*.xtc")):
        raw = path.read_bytes()
        for start, end in _blocks(path):
            fast = X.decode_xtc_frame(raw[start:end])
            slow = X.decode_xtc_frame(raw[start:end], scalar=True)
            assert np.array_equal(fast.coords_nm, slow.coords_nm), path.name


# ---------------------------------------------------------------------------
# elements, boxes and options
# ---------------------------------------------------------------------------

def test_without_elements_the_refusal_says_what_to_pass():
    message = _refusal(lambda: X.read_xtc(GLASS))
    assert "xtc_mda_glass.xtc" in message
    assert "topology=" in message and "type_map=" in message
    assert "1 to 30" in message
    # placeholders, not element names that read like advice for this file
    # (D7 of the recognition test; integration re-run, 2026-10-07)
    assert "type_map={1: '<element>', 2: '<element>', ...}" in message
    assert "'Si'" not in message


def test_type_map_is_keyed_by_atom_number():
    message = _refusal(lambda: X.read_xtc(GLASS, type_map={1: "Si", 2: "O"}))
    assert "28 of 30 have no entry" in message and "topology=" in message
    message = _refusal(lambda: X.read_xtc(GLASS, type_map={"Si": "Si"}))
    assert "atom number" in message and "xtc_mda_glass.xtc" in message
    message = _refusal(lambda: X.read_xtc(GLASS, type_map={31: "O"}))
    assert "atoms 1 to 30" in message
    bad = dict(GLASS_MAP)
    bad[3] = "Xx"
    with pytest.raises(ValueError):
        X.read_xtc(GLASS, type_map=bad)


def test_topology_as_a_frame_or_a_trajectory():
    elements = np.array(GLASS_ELEMENTS)
    frame = frame_from_arrays(elements, np.zeros((30, 3)) + 1.0,
                              box_ang=np.eye(3) * 15, atom_id=np.arange(101, 131))
    traj = X.read_xtc(GLASS, topology=frame)
    assert traj.frame(0).atom_id.tolist() == list(range(101, 131))
    assert traj.type_map_source == "in memory"
    model = X.read_xtc(GLASS, type_map=GLASS_MAP)
    again = X.read_xtc(GLASS, topology=model)
    assert again.frame(1).elements.tolist() == GLASS_ELEMENTS
    message = _refusal(lambda: X.read_xtc(GLASS, topology=frame,
                                          type_map={1: "O"}))
    assert "already a frame" in message
    short = frame_from_arrays(elements[:20], np.ones((20, 3)),
                              box_ang=np.eye(3) * 15)
    message = _refusal(lambda: X.read_xtc(GLASS, topology=short))
    assert "30 atoms" in message and "20" in message
    message = _refusal(lambda: X.read_xtc(GLASS, type_map=GLASS_MAP,
                                          mass_tol_amu=0.02))
    assert "topology" in message


def test_topology_charges_are_carried_and_a_topology_without_ids_refused(
        tmp_path):
    elements = np.array(GLASS_ELEMENTS)
    charge = np.where(elements == "Si", 2.4, -1.2)
    frame = frame_from_arrays(elements, np.ones((30, 3)), box_ang=np.eye(3) * 15,
                              charge_e=charge)
    traj = X.read_xtc(GLASS, topology=frame)
    assert np.array_equal(traj.frame(2).charge_e, charge)
    assert traj.charges_e == {"O": -1.2, "Si": 2.4}
    rows = "\n".join(f"{1 if e == 'Si' else 2} {i % 7}.5 {i % 5}.5 {i % 3}.5"
                     for i, e in enumerate(GLASS_ELEMENTS))
    dump = _write(tmp_path, "noid.lammpstrj", (
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n30\nITEM: BOX BOUNDS pp pp "
        "pp\n0 15\n0 15\n0 15\nITEM: ATOMS type x y z\n" + rows + "\n").encode())
    message = _refusal(lambda: X.read_xtc(GLASS, topology=dump,
                                          type_map={1: "Si", 2: "O"}))
    assert "noid.lammpstrj has no atom ids" in message and "id column" in message


def test_a_zero_box_needs_box_from():
    path = DATA / "xtc_mda_zerobox.xtc"
    elements = {i: "O" for i in range(1, 13)}
    message = _refusal(lambda: X.read_xtc(path, type_map=elements))
    assert "xtc_mda_zerobox.xtc" in message and "box_from=" in message
    box = np.eye(3) * 20.0
    traj = X.read_xtc(path, type_map=elements, box_from=box)
    assert np.array_equal(traj.frame(0).box_ang, box)
    assert any("box of zeros" in n for n in traj.notes)
    message = _refusal(lambda: X.read_xtc(GLASS, type_map=GLASS_MAP,
                                          box_from=box))
    assert "states its box" in message
    from_file = X.read_xtc(path, type_map=elements,
                           box_from=DATA / "xtc_lammps.data")
    lammps_box = md_readers.read_trajectory(DATA / "xtc_lammps.data").frame(0)
    assert np.array_equal(from_file.frame(0).box_ang, lammps_box.box_ang)
    assert np.array_equal(from_file.frame(0).origin_ang, lammps_box.origin_ang)


def test_md_readers_dispatches_xtc_files_and_readers_read_routes_them():
    path = DATA / "xtc_lammps.xtc"
    assert md_readers.sniff_md(path) == "gromacs-xtc"
    direct = X.read_xtc(path, topology=DATA / "xtc_lammps.data")
    routed = md_readers.read_trajectory(path, topology=DATA / "xtc_lammps.data")
    assert routed.file_format == "gromacs-xtc"
    assert np.array_equal(routed.frame(2).cart_ang, direct.frame(2).cart_ang)
    message = _refusal(lambda: md_readers.read_trajectory(path, units="metal"))
    assert "xtc_lammps.xtc" in message and "units" in message
    with pytest.raises(readers.MDModelFile):
        readers.read(path)


# ---------------------------------------------------------------------------
# cut and damaged files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("keep, words", [
    (-10, "needs"), (20, "frame header at byte"), (70, "inside the"),
    (2, "fewer than 4 bytes")])
def test_a_cut_last_frame_is_counted(tmp_path, keep, words):
    raw = GLASS.read_bytes()
    start, end = _blocks(GLASS)[2]
    cut = raw[:end + keep] if keep < 0 else raw[:start + keep]
    traj = X.read_xtc(_write(tmp_path, "cut.xtc", cut), type_map=GLASS_MAP)
    assert traj.n_frames == 2 and list(traj.skipped) == [2]
    assert words in traj.skipped[2]
    assert traj.file_positions.tolist() == [0, 1]
    assert any("1 frame(s) skipped" in line for line in traj.describe())


def test_garbage_between_frames_is_one_counted_gap(tmp_path):
    raw = GLASS.read_bytes()
    _, end0 = _blocks(GLASS)[0]
    path = _write(tmp_path, "gap.xtc", raw[:end0] + b"\xab" * 37 + raw[end0:])
    traj = X.read_xtc(path, type_map=GLASS_MAP)
    assert traj.n_frames == 3 and traj.file_positions.tolist() == [0, 2, 3]
    assert f"bytes {end0} to {end0 + 37} (37 bytes)" in traj.skipped[1]
    assert "reading resumed" in traj.skipped[1]
    original = X.read_xtc(GLASS, type_map=GLASS_MAP)
    for k in range(3):
        assert np.array_equal(traj.frame(k).cart_ang, original.frame(k).cart_ang)


def test_a_damaged_magic_number_loses_only_its_frame(tmp_path):
    raw = bytearray(GLASS.read_bytes())
    start, end = _blocks(GLASS)[1]
    raw[start:start + 4] = b"\x00\x00\x00\x00"
    traj = X.read_xtc(_write(tmp_path, "magic.xtc", bytes(raw)),
                      type_map=GLASS_MAP)
    assert traj.n_frames == 2 and traj.timesteps.tolist() == [0, 200]
    assert "read 0, not the XTC magic number 1995" in traj.skipped[1]
    assert "about 1 frame(s)" in traj.skipped[1]


def test_a_damaged_byte_count_is_passed_over_to_the_next_header(tmp_path):
    raw = bytearray(GLASS.read_bytes())
    struct.pack_into(">i", raw, 88, 10 ** 6)
    traj = X.read_xtc(_write(tmp_path, "count.xtc", bytes(raw)),
                      type_map=GLASS_MAP)
    assert traj.timesteps.tolist() == [100, 200]
    assert "its byte count is damaged" in traj.skipped[0]


def test_bytes_with_no_further_header_are_one_position(tmp_path):
    path = _write(tmp_path, "tail.xtc", GLASS.read_bytes() + b"\x01" * 100)
    traj = X.read_xtc(path, type_map=GLASS_MAP)
    assert traj.n_frames == 3
    assert "no further frame header follows" in traj.skipped[3]


@pytest.mark.parametrize("fill, words", [
    (b"\xff", "past the 30 atoms of the frame"),
    (b"\x00", "the writer stores exactly the bytes it used")])
def test_a_damaged_block_raises_frame_error_naming_file_and_frame(
        tmp_path, fill, words):
    raw = bytearray(GLASS.read_bytes())
    start, end = _blocks(GLASS)[1]
    nbytes = struct.unpack_from(">i", raw, start + 88)[0]
    raw[start + 92:start + 92 + nbytes] = fill * nbytes
    traj = X.read_xtc(_write(tmp_path, "bits.xtc", bytes(raw)),
                      type_map=GLASS_MAP)
    traj.frame(0)
    traj.frame(2)
    with pytest.raises(FrameError) as caught:
        traj.frame(1)
    message = str(caught.value)
    assert "bits.xtc" in message and "frame 1" in message and words in message


def test_a_damaged_frame_0_names_the_file(tmp_path):
    raw = bytearray(GLASS.read_bytes())
    nbytes = struct.unpack_from(">i", raw, 88)[0]
    raw[92:92 + nbytes] = b"\xff" * nbytes
    message = _refusal(lambda: X.read_xtc(
        _write(tmp_path, "zero.xtc", bytes(raw)), type_map=GLASS_MAP))
    assert "zero.xtc: frame 0" in message


def test_a_non_finite_box_skips_its_frame(tmp_path):
    raw = bytearray(GLASS.read_bytes())
    start, _ = _blocks(GLASS)[1]
    struct.pack_into(">f", raw, start + 16, float("nan"))
    traj = X.read_xtc(_write(tmp_path, "nan.xtc", bytes(raw)),
                      type_map=GLASS_MAP)
    assert traj.timesteps.tolist() == [0, 200]
    assert "not a finite number" in traj.skipped[1]


def test_a_frame_with_another_atom_count_is_skipped(tmp_path):
    small = DATA / "xtc_mda_small.xtc"
    start, end = _blocks(small)[0]
    path = _write(tmp_path, "mixed.xtc",
                  GLASS.read_bytes() + small.read_bytes()[start:end])
    traj = X.read_xtc(path, type_map=GLASS_MAP)
    assert traj.n_frames == 3
    assert traj.skipped == {3: "6 atoms, where 3 of the 4 frames hold 30"}


def test_large_system_frames_with_magic_2023_read_alike(tmp_path):
    """GROMACS 2023.2+ writes magic 2023 and a 64-bit byte count (xtcio.cpp,
    libxdrf.cpp); built here from the fixture by that layout."""
    raw = GLASS.read_bytes()
    out = b""
    for start, end in _blocks(GLASS):
        frame = bytearray(raw[start:end])
        struct.pack_into(">i", frame, 0, 2023)
        nbytes = struct.unpack_from(">i", frame, 88)[0]
        out += bytes(frame[:88]) + struct.pack(">q", nbytes) + bytes(frame[92:])
    traj = X.read_xtc(_write(tmp_path, "new.xtc", out), type_map=GLASS_MAP)
    original = X.read_xtc(GLASS, type_map=GLASS_MAP)
    assert traj.n_frames == 3
    for k in range(3):
        assert np.array_equal(traj.frame(k).cart_ang, original.frame(k).cart_ang)
    assert any("magic 2023" in n for n in traj.notes)


def test_files_that_are_not_xtc_are_refused_with_the_reason(tmp_path):
    text = _write(tmp_path, "notes.xtc", b"ITEM: TIMESTEP\n0\n" * 10)
    message = _refusal(lambda: X.read_xtc(text, type_map=GLASS_MAP))
    assert "notes.xtc" in message and "1995" in message
    packed = _write(tmp_path, "glass.xtc.gz", gzip.compress(GLASS.read_bytes()))
    with pytest.raises(UnsupportedFormat, match="decompress it first"):
        X.read_xtc(packed, type_map=GLASS_MAP)
    with pytest.raises(FileNotFoundError):
        X.read_xtc(tmp_path / "missing.xtc")


def _six_frames() -> tuple[bytes, list[int]]:
    """The glass fixture twice, steps rewritten 0 .. 500, and frame starts."""
    raw = bytearray(GLASS.read_bytes() * 2)
    size = GLASS.stat().st_size
    starts = [s for s, _ in _blocks(GLASS)]
    starts += [size + s for s in starts]
    for k, start in enumerate(starts):
        struct.pack_into(">i", raw, start + 8, 100 * k)
    return bytes(raw), starts


@pytest.mark.parametrize("onto", ["exactly frame 3", "past frame 5"])
def test_a_byte_count_enlarged_over_later_frames_hides_none(tmp_path, onto):
    """Frame 1's byte count enlarged so that its stated extent covers frames
    2 .. 3 (landing exactly on frame 3's header) or 2 .. 5: before, those
    frames vanished with no position in skipped."""
    raw, starts = _six_frames()
    raw = bytearray(raw)
    nbytes = (starts[3] - starts[1] - 92 if onto == "exactly frame 3"
              else starts[5] - starts[1] - 92 + 5)
    struct.pack_into(">i", raw, starts[1] + 88, nbytes)
    traj = X.read_xtc(_write(tmp_path, "count.xtc", bytes(raw)),
                      type_map=GLASS_MAP)
    assert traj.n_frames + len(traj.skipped) == 6
    assert traj.timesteps.tolist() == [0, 200, 300, 400, 500]
    assert list(traj.skipped) == [1]
    assert f"byte count ({nbytes}) is damaged" in traj.skipped[1]
    assert f"a frame header follows at byte {starts[2]}" in traj.skipped[1]
    intact, _ = _six_frames()
    whole = X.read_xtc(_write(tmp_path, "whole.xtc", intact), type_map=GLASS_MAP)
    for k, position in enumerate(traj.file_positions.tolist()):
        assert np.array_equal(traj.frame(k).cart_ang,
                              whole.frame(position).cart_ang)


def test_a_changed_precision_word_is_refused_or_named(tmp_path):
    """One bit of frame 1's precision (1000.0) flipped: bit 30 gives 2.9e-36,
    whose positions lie 3e38 box lengths away and would all wrap onto one
    point; bit 28 gives 4.3e12, which decodes (no checksum to tell), and a
    note names the file position of each precision."""
    start = _blocks(GLASS)[1][0]
    for bit in (30, 28):
        raw = bytearray(GLASS.read_bytes())
        word = struct.unpack_from(">I", raw, start + 56)[0]
        struct.pack_into(">I", raw, start + 56, word ^ (1 << bit))
        flipped = struct.unpack_from(">f", raw, start + 56)[0]
        traj = X.read_xtc(_write(tmp_path, f"prec{bit}.xtc", bytes(raw)),
                          type_map=GLASS_MAP)
        note = next(n for n in traj.notes if "more than one precision" in n)
        assert "1000 at file position(s) 0, 2" in note
        assert f"{flipped:g} at file position(s) 1" in note
        if bit == 30:
            with pytest.raises(FrameError) as caught:
                traj.frame(1)
            message = str(caught.value)
            assert f"prec{bit}.xtc: frame 1" in message
            assert "box lengths from the box" in message
            assert f"precision is {flipped:g}" in message
        else:
            traj.frame(1)
        traj.frame(2)


def test_a_header_stating_a_huge_atom_count_sets_aside_no_memory(tmp_path):
    """A 96-byte file whose header states 20 000 000 atoms and a 4-byte block,
    read with a 30-atom type_map: it took 340 MB before its refusal."""
    n = 20_000_000
    head = struct.pack(">iiif", 1995, n, 0, 0.0)
    head += np.diag([5.0, 5.0, 5.0]).astype(">f4").tobytes()
    head += struct.pack(">if3i3iii", n, 1000.0, 0, 0, 0, 10, 10, 10, 9, 4)
    path = _write(tmp_path, "huge.xtc", head + bytes(4))
    tracemalloc.start()
    try:
        message = _refusal(lambda: X.read_xtc(path, type_map=GLASS_MAP))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert "huge.xtc" in message and "states 20000000 atoms" in message
    assert "fewer than the 5000000 bytes" in message and "2 bits" in message
    assert peak < n // 20                  # under 0.05 bytes per stated atom
    # The map check alone: 17 bytes per stated atom before (340 MB here).
    tracemalloc.start()
    try:
        message = _refusal(lambda: X._per_atom_map({1: "O"}, n, "x.xtc"))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert "19999999 of 20000000 have no entry (first: [2, 3, 4, 5, 6])" \
        in message
    assert peak < n // 20


@pytest.mark.parametrize("damage", ["atom count", "magic number"])
def test_a_damaged_first_header_loses_only_its_frame(tmp_path, damage):
    raw = bytearray(GLASS.read_bytes())
    if damage == "atom count":
        struct.pack_into(">i", raw, 4, 31)          # byte 52 still says 30
    else:
        raw[0:4] = bytes(4)
    traj = X.read_xtc(_write(tmp_path, "first.xtc", bytes(raw)),
                      type_map=GLASS_MAP)
    assert traj.timesteps.tolist() == [100, 200]
    assert list(traj.skipped) == [0]
    assert "reading resumed at the next frame header, at byte 224" \
        in traj.skipped[0]
    original = X.read_xtc(GLASS, type_map=GLASS_MAP)
    assert np.array_equal(traj.frame(0).cart_ang, original.frame(1).cart_ang)


def test_an_empty_or_short_file_is_refused_saying_so(tmp_path):
    message = _refusal(lambda: X.read_xtc(_write(tmp_path, "empty.xtc", b""),
                                          type_map=GLASS_MAP))
    assert "empty.xtc" in message and "empty (0 bytes)" in message
    short = _write(tmp_path, "short.xtc", GLASS.read_bytes()[:40])
    message = _refusal(lambda: X.read_xtc(short, type_map=GLASS_MAP))
    assert "short.xtc" in message
    assert "40 bytes, fewer than one 56-byte XTC frame header" in message
    other = _write(tmp_path, "notes.txt", b"ITEM: TIMESTEP\n0\n" * 10)
    message = _refusal(lambda: X.read_xtc(other, type_map=GLASS_MAP))
    assert "notes.txt" in message and "does not start with an XTC" in message
    assert "md_readers.read_trajectory" in message


def test_box_from_takes_the_header_box_of_another_xtc_file():
    path = DATA / "xtc_mda_zerobox.xtc"
    elements = {i: "O" for i in range(1, 13)}
    traj = md_readers.read_trajectory(path, type_map=elements, box_from=GLASS)
    glass0 = X.read_xtc(GLASS, type_map=GLASS_MAP).frame(0)
    assert np.array_equal(traj.frame(0).box_ang, glass0.box_ang)
    assert any("header of frame 0 of" in n and "xtc_mda_glass.xtc" in n
               for n in traj.notes)
    box, origin, where = X.read_xtc_box(GLASS)
    assert np.array_equal(box, glass0.box_ang) and not origin.any()
    assert "step 0" in where and "no box origin" in where
    message = _refusal(lambda: X.read_xtc_box(path))
    assert "xtc_mda_zerobox.xtc" in message and "all zeros" in message


def test_a_frame_whose_only_non_finite_value_is_its_time_is_kept(tmp_path):
    starts = [s for s, _ in _blocks(GLASS)]
    raw = bytearray(GLASS.read_bytes())
    struct.pack_into(">f", raw, starts[1] + 12, float("inf"))
    path = _write(tmp_path, "inf.xtc", bytes(raw))
    traj = X.read_xtc(path, type_map=GLASS_MAP)
    assert traj.n_frames == 3 and traj.skipped == {}
    assert np.isnan(traj.times_ps[1]) and traj.frame(1).time_ps is None
    assert any("not a finite number (file position(s) 1)" in n
               for n in traj.notes)
    rebuilt = X.read_xtc(path, type_map=GLASS_MAP, timestep_fs=2.0)
    assert np.allclose(rebuilt.times_ps, [0.0, 0.2, 0.4], rtol=0, atol=1e-12)
    for start in starts:
        struct.pack_into(">f", raw, start + 12, float("nan"))
    path = _write(tmp_path, "nan.xtc", bytes(raw))
    traj = X.read_xtc(path, type_map=GLASS_MAP)
    assert traj.n_frames == 3 and np.isnan(traj.times_ps).all()
    rebuilt = X.read_xtc(path, type_map=GLASS_MAP, timestep_fs=2.0)
    assert np.allclose(rebuilt.times_ps, [0.0, 0.2, 0.4], rtol=0, atol=1e-12)
    assert any("no frame holds a finite time" in n for n in rebuilt.notes)


def test_a_signalling_nan_raises_no_numpy_warning(tmp_path):
    """0x7F800001 is a signalling NaN; casting it to float64 made numpy warn,
    which escapes as an exception under -W error."""
    raw = bytearray(GLASS.read_bytes())
    struct.pack_into(">I", raw, 32, 0x7F800001)            # frame 0's box
    small = bytearray((DATA / "xtc_mda_small.xtc").read_bytes())
    struct.pack_into(">I", small, 56, 0x7F800001)          # a plain coordinate
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        traj = X.read_xtc(_write(tmp_path, "snan.xtc", bytes(raw)),
                          type_map=GLASS_MAP)
        assert list(traj.skipped) == [0]
        assert "not a finite number" in traj.skipped[0]
        with pytest.raises(FrameError, match="non-finite coordinates"):
            X.read_xtc(_write(tmp_path, "snan_small.xtc", bytes(small)),
                       type_map={i: "O" for i in range(1, 7)})


def test_a_crystal_structure_as_topology_is_refused_naming_both():
    cif = ROOT / "tests" / "data" / "crystals" / "quartz_SiO2_cod9013321.cif"
    message = _refusal(lambda: X.read_xtc(GLASS, topology=cif))
    assert message.startswith(
        "xtc_mda_glass.xtc: topology=quartz_SiO2_cod9013321.cif")
    assert "not taken as a topology" in message
    assert "LAMMPS data file the run read" in message
    message = _refusal(lambda: X.read_xtc(GLASS, topology=readers.read(cif)))
    assert "not Structure" in message and "not taken as a topology" in message


def test_periodicity_is_the_topologys_where_its_file_states_it(tmp_path):
    xtc = DATA / "xtc_lammps.xtc"
    traj = X.read_xtc(xtc, topology=DATA / "xtc_lammps.data")
    assert traj.periodic is None
    note = next(n for n in traj.notes if "does not record the boundary" in n)
    assert "taken as periodic along a, b and c" in note and "vacuum" in note
    ref = DATA / "xtc_lammps_ref.lammpstrj"
    traj = X.read_xtc(xtc, topology=ref, type_map=LAMMPS_TYPES)
    assert traj.periodic == (True, True, True)
    assert any("xtc_lammps_ref.lammpstrj states periodic (True, True, True)"
               in n for n in traj.notes)
    slab = _write(tmp_path, "slab.lammpstrj",
                  ref.read_bytes().replace(b"pp pp pp", b"pp pp ff"))
    traj = X.read_xtc(xtc, topology=slab, type_map=LAMMPS_TYPES)
    assert traj.periodic == (True, True, False)
    note = next(n for n in traj.notes if "slab.lammpstrj states periodic" in n)
    assert "not periodic along every axis" in note and "vacuum" in note


def test_a_start_time_below_the_rounding_of_late_frames_is_kept(tmp_path):
    """Times 0.001 + step x 0.002 ps at steps 0, 1e7, 2e7: in float32 the
    0.001 ps start is lost at 40 000 ps, and the median of time - step x dt
    put the offset at 0 and refused the file."""
    raw = bytearray(GLASS.read_bytes())
    for k, (start, _) in enumerate(_blocks(GLASS)):
        step = 10_000_000 * k
        struct.pack_into(">i", raw, start + 8, step)
        struct.pack_into(">f", raw, start + 12, 0.001 + step * 0.002)
    traj = X.read_xtc(_write(tmp_path, "tinit.xtc", bytes(raw)),
                      type_map=GLASS_MAP, timestep_fs=2.0)
    assert np.abs(traj.times_ps - [0.001, 20000.001, 40000.001]).max() < 1e-9
    assert any("plus the offset 0.00100000005 ps" in n
               and "file position 0" in n for n in traj.notes)


def test_the_units_note_names_sfactor_and_tfactor():
    traj = X.read_xtc(DATA / "xtc_lammps.xtc", topology=DATA / "xtc_lammps.data")
    assert "units lj" in traj.units_note
    assert "dump_modify sfactor or tfactor" in traj.units_note


def test_a_topology_at_a_shared_step_is_compared_row_by_row():
    traj = X.read_xtc(DATA / "xtc_lammps.xtc", topology=DATA / "xtc_lammps.data")
    note = next(n for n in traj.notes if "share step 0" in n)
    largest = float(re.search(r"differ by up to (\S+) Å per axis", note)[1])
    assert largest <= 0.005 + 1e-6                     # the grid's half step
    assert "rows differ by more" not in note
    frame = X.read_xtc(GLASS, type_map=GLASS_MAP).frame(0)
    order = np.roll(np.arange(30), 1)                 # the rows in another order
    shuffled = frame_from_arrays(frame.elements[order], frame.cart_ang[order],
                                 box_ang=frame.box_ang, timestep=0)
    note = next(n for n in X.read_xtc(GLASS, topology=shuffled).notes
                if "share step 0" in n)
    assert "30 of 30 rows differ by more than that half step" in note
    later = frame_from_arrays(frame.elements, frame.cart_ang,
                              box_ang=frame.box_ang, timestep=7)
    note = next(n for n in X.read_xtc(GLASS, topology=later).notes
                if "no XTC frame shares" in n)
    assert "is at step 7" in note and "atom count alone" in note


def test_damage_that_keeps_the_stream_whole_decodes_and_is_documented():
    """XTC holds no checksum: a changed bit inside a full atom's field moves
    that atom and decodes, here as in the xdrfile C reader. The module
    docstring says so, with the measured share of changes caught."""
    data = (DATA / "xtc_mda_glass.xtc").read_bytes()[:_blocks(GLASS)[0][1]]
    intact = X.decode_xtc_frame(data).ints
    nbytes = struct.unpack_from(">i", data, 88)[0]
    moved = 0
    for offset in range(92, 92 + nbytes):
        bad = bytearray(data)
        bad[offset] ^= 0x01
        try:
            ints = X.decode_xtc_frame(bytes(bad)).ints
        except ValueError:
            continue
        moved += int((ints != intact).any())
    assert moved > 0
    assert "XTC holds no checksum" in X.__doc__
    assert "9 %, 18 % and 17 %" in X.__doc__


# ---------------------------------------------------------------------------
# the format contract
# ---------------------------------------------------------------------------

def test_format_spec_and_sniff():
    (spec,) = X.FORMATS
    assert isinstance(spec, FormatSpec)
    assert spec.name == "gromacs-xtc" and spec.extensions == (".xtc",)
    assert spec.binary and spec.read is X.read_xtc
    assert spec.options <= KNOWN_OPTIONS
    for path in DATA.glob("xtc_*.xtc"):
        assert spec.sniff(path.read_bytes()[:4096], path)
    assert not spec.sniff(b"ITEM: TIMESTEP\n0\n", Path("x.xtc"))
    assert not spec.sniff(b"\x00\x00\x07\xcb", Path("x.xtc"))
    assert not spec.sniff(None, Path("x.xtc"))           # never raises


def test_the_trajectory_api():
    with X.read_xtc(GLASS, type_map=GLASS_MAP) as traj:
        lines = traj.describe()
        assert lines[1] == "format: gromacs-xtc"
        # The file does not record the boundary: None, the md_readers
        # convention for a file that does not state it (a LAMMPS data file).
        assert traj.ids_track_atoms and traj.periodic is None
        assert any("does not record the boundary" in n for n in traj.notes)
        copy = pickle.loads(pickle.dumps(traj))
        assert np.array_equal(copy.frame(2).cart_ang, traj.frame(2).cart_ang)
        assert "1/1000 nm (0.01 Å)" in traj.units_note


def test_the_module_imports_without_md_readers_and_without_qt():
    """md_readers may import this module at its top level: no cycle."""
    code = ("import sys, facet.core.md_formats_xtc;"
            "print(sorted(m for m in ('facet.core.md_readers', 'PySide6') "
            "if m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def test_no_note_message_or_string_carries_a_verdict(tmp_path):
    texts = []
    texts += X.read_xtc(GLASS, type_map=GLASS_MAP).describe()
    texts += X.read_xtc(DATA / "xtc_lammps.xtc", topology=DATA / "xtc_lammps.data",
                        timestep_fs=1.0).describe()
    raw = GLASS.read_bytes()
    _, end0 = _blocks(GLASS)[0]
    texts += X.read_xtc(_write(tmp_path, "g.xtc", raw[:end0] + b"\xab" * 9
                               + raw[end0:-5]), type_map=GLASS_MAP).describe()
    texts.append(_refusal(lambda: X.read_xtc(GLASS)))
    texts.append(_refusal(lambda: X.read_xtc(GLASS, type_map={1: "O"})))
    texts.append(_refusal(lambda: X.read_xtc(DATA / "xtc_mda_zerobox.xtc",
                                             type_map={1: "O"})))
    texts.append(_refusal(lambda: X.read_xtc(
        DATA / "xtc_lammps.xtc", topology=DATA / "xtc_lammps.data",
        timestep_fs=3.0)))
    assert [t for t in texts if VERDICT.search(t)] == []
    source = (ROOT / "facet" / "core" / "md_formats_xtc.py").read_text(
        encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 80
    assert [s for s in strings if VERDICT.search(s)] == []
