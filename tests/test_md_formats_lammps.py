"""DCD, LAMMPS binary dump, YAML dump and AtomEye CFG give back LAMMPS's state.

The files in tests/data/md/lammps_formats were written by LAMMPS 22 Jul 2025
update 4 (pip wheel) and ASE 3.29.0 from two small models
(make_lammps_formats.py beside them); ``truth.npz`` is LAMMPS's own state,
read through its library interface after each dumped step, never from a dump.
What is pinned, and with which tolerance:

* **DCD positions are float32.** A float32 holds 24 significant bits, so a
  value x is stored within 2^-24 |x| (half a unit in the last place);
  positions are compared modulo the box vectors (a DCD has no origin) to
  ``2**-24 * max|x|`` plus 1e-12 Å for the float64 arithmetic after it. The
  box is rebuilt from lengths and cosines written as float64: 1e-12 Å.
  MDAnalysis 2.10 read the same files bit-identically (checked outside the
  suite; MDAnalysis is not a dependency).
* **Binary dumps are float64:** positions, unwrapped positions, velocities to
  1e-12.
* **Text values carry LAMMPS's ``%g``** (6 significant digits) in YAML and CFG
  rows and CFG boxes: a value x is within 5e-6 |x| of the double; positions
  are compared to ``5e-6 * max|x|`` per coordinate set, times 10 for LAMMPS's
  unwrapped CFG, whose s' = (s - 0.5) / 10 + 0.5 is printed with %g and
  multiplied by 10 again. ASE writes ``%e`` (7 digits) and the box with
  ``%f``: 1e-5 Å.
* **Byte order and record-marker size** do not change a number: big-endian
  and int64-marker DCDs, big-endian, older-layout, revision-1 and
  multi-chunk binary dumps are built here by byte surgery on the LAMMPS files
  and read to the same values.
* **Nothing is dropped silently:** a cut last frame is in ``skipped``; a
  binary header that does not parse says how many bytes after it are not
  located.
* **Refusals say what to pass:** a DCD without topology, a binary dump without
  column names, LAMMPS's empty YAML typelabel column, a CFG whose every type
  is 'C'.
* **No verdict words** in any note, message or string of the module.
* **Edge cases from two verification passes** (the last section) read files
  in tests/data/md/lammps_formats/edges, written by LAMMPS 22 Jul 2025,
  mdtraj 1.11 and MDAnalysis 2.10 (make_lammps_edges.py beside them, with
  LAMMPS's own steps, times, positions and velocities in edges.json):
  ``dump_modify first yes`` and molfile-plugin DCD time axes, MDAnalysis's
  zeroed DCD cell, 0-atom binary frames, ``dump_modify colname``, a yaml
  dump written with ``triclinic/general yes``, LAMMPS's CFG velocities; and
  byte surgery for damage: resynchronising binary frames, gzip cuts, cut
  YAML headers, a byte-order mark, cut CFG files, CFG header fields, record
  lengths beyond the file (memory measured with tracemalloc).
"""
from __future__ import annotations

import ast
import gzip
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tracemalloc
import zlib
from pathlib import Path

import numpy as np
import pytest

from facet.core import md_formats_base, md_formats_lammps as M, md_readers
from facet.core.md_model import FrameError
from facet.core.readers import UnsupportedFormat

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data" / "md" / "lammps_formats"
EDGES = DATA / "edges"     # make_lammps_edges.py: LAMMPS, mdtraj, MDAnalysis
MD_DATA = DATA.parent
T = np.load(DATA / "truth.npz")
E = json.loads((EDGES / "edges.json").read_text())
TYPES = {1: "Si", 2: "Na", 3: "O"}
F32 = 2.0 ** -24          # half a unit in the last place of a float32
G6 = 5e-6                 # half a unit in the 6th significant digit (%g)


def truth(prefix: str = "metal_"):
    return {k[len(prefix):]: T[k] for k in T.files if k.startswith(prefix)}


METAL = truth()
REAL = truth("real_")
ELEMENTS = np.array([TYPES[int(t)] for t in METAL["types"]])


def modlat(cart, reference, box) -> float:
    """Largest distance between two position sets modulo the box vectors."""
    d = np.asarray(cart) - np.asarray(reference)
    f = np.linalg.solve(np.asarray(box).T, d.T).T
    f -= np.round(f)
    return float(np.abs(f @ box).max())


def unwrapped_truth(model, k):
    return model["x"][k] + model["image"][k] @ model["H"][k]


# ---------------------------------------------------------------------------
# DCD byte surgery: Fortran records, re-encoded
# ---------------------------------------------------------------------------

def records(raw: bytes, order: str = "<", marker: int = 4) -> list[bytes]:
    code = order + ("i" if marker == 4 else "q")
    out, pos = [], 0
    while pos < len(raw):
        n = struct.unpack(code, raw[pos:pos + marker])[0]
        out.append(raw[pos + marker:pos + marker + n])
        pos += 2 * marker + n
    return out


def join(recs, order: str = "<", marker: int = 4) -> bytes:
    code = order + ("i" if marker == 4 else "q")
    return b"".join(struct.pack(code, len(r)) + r + struct.pack(code, len(r))
                    for r in recs)


def swap(data: bytes, width: int) -> bytes:
    return np.frombuffer(data, dtype=f"<u{width}").byteswap().tobytes()


def is_cell(index: int) -> bool:
    """Record ``index`` of a LAMMPS DCD is a unit cell: after the three header
    records, each frame is cell, x, y, z (by position, not by length: 12
    float32 coordinates are 48 bytes too)."""
    return index >= 3 and (index - 3) % 4 == 0


def big_endian_dcd(raw: bytes, marker: int = 4) -> bytes:
    recs = records(raw)
    out = [recs[0][:4] + swap(recs[0][4:], 4), swap(recs[1][:4], 4) + recs[1][4:],
           swap(recs[2], 4)]
    for index, r in enumerate(recs[3:], start=3):
        out.append(swap(r, 8) if is_cell(index) else swap(r, 4))
    return join(out, ">", marker)


def dcd_with(raw: bytes, *, icntrl=None, title=None, cells=None, drop_cells=False):
    """The LAMMPS DCD with header fields, the title or the cells replaced."""
    recs = records(raw)
    head = bytearray(recs[0])
    for index, value in (icntrl or {}).items():
        struct.pack_into("<i", head, 4 + 4 * index, value)
    recs[0] = bytes(head)
    if title is not None:
        recs[1] = struct.pack("<i", 1) + title.encode().ljust(80, b" ")
    out = recs[:3]
    k = 0
    for index, r in enumerate(recs[3:], start=3):
        if is_cell(index):
            if drop_cells:
                continue
            if cells is not None:
                r = np.asarray(cells(np.frombuffer(r, "<f8").copy(), k),
                               dtype="<f8").tobytes()
            k += 1
        out.append(r)
    return join(out)


def write(tmp_path, name, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def dcd_values(trajectory):
    return [trajectory._raw(k) for k in range(len(trajectory))]


# ---------------------------------------------------------------------------
# DCD
# ---------------------------------------------------------------------------

def test_dcd_positions_box_time_and_elements_match_lammps():
    traj = M.read_dcd(DATA / "traj.dcd", topology=DATA / "topology.data")
    assert traj.file_format == "dcd" and len(traj) == 3 and traj.n_atoms == 12
    assert traj.timesteps.tolist() == METAL["steps"].tolist() == [100, 110, 120]
    # units metal from the data file's write_data title: DELTA 0.001 ps
    assert np.allclose(traj.times_ps, METAL["time"], rtol=0, atol=1e-12)
    for k in range(3):
        frame = traj.frame(k)
        assert frame.atom_id.tolist() == METAL["ids"].tolist()
        assert frame.elements.tolist() == ELEMENTS.tolist()
        bound = F32 * np.abs(METAL["x"][k]).max() + 1e-12
        assert modlat(frame.cart_ang, METAL["x"][k], METAL["H"][k]) <= bound
        assert np.abs(frame.box_ang - METAL["H"][k]).max() < 1e-12
        assert np.array_equal(frame.origin_ang, np.zeros(3))
        assert frame.unwrapped_cart_ang is None
    assert traj.box_varies is True
    assert traj.type_map == TYPES and traj.type_map_source == "type label"
    assert any("read as wrapped" in n and "jump across the box" in n
               for n in traj.notes)
    assert any("no box origin" in n for n in traj.notes)
    assert traj.charges_e == {"Si": 2.4, "Na": 0.6, "O": -1.2}


def test_dcd_unwrap_yes_is_recognised_and_kept_as_unwrapped():
    traj = M.read_dcd(DATA / "traj_unwrap.dcd", topology=DATA / "topology.data")
    assert any("read as unwrapped" in n for n in traj.notes)
    for k in range(3):
        frame = traj.frame(k)
        expected = unwrapped_truth(METAL, k)
        bound = F32 * np.abs(expected).max() + 1e-12
        assert np.abs(frame.unwrapped_cart_ang - expected).max() <= bound
        assert modlat(frame.cart_ang, METAL["x"][k], METAL["H"][k]) <= bound


def test_dcd_units_real_gives_delta_in_fs():
    traj = M.read_dcd(DATA / "traj_real.dcd", topology=DATA / "topology_real.data")
    assert traj.timesteps.tolist() == [0, 5, 10]
    assert np.allclose(traj.times_ps, REAL["time"] * 1e-3, rtol=0, atol=1e-15)
    assert "units real" in traj.units_note and "fs" in traj.units_note
    assert traj.type_map_source == "data-file masses"
    for k in range(3):
        assert modlat(traj.frame(k).cart_ang, REAL["x"][k], REAL["H"][k]) <= \
            F32 * np.abs(REAL["x"][k]).max() + 1e-12
    # the units argument and timestep_fs
    assert np.allclose(M.read_dcd(DATA / "traj_real.dcd",
                                  topology=DATA / "topology_real.data",
                                  timestep_fs=2.0).times_ps, [0, 0.01, 0.02])
    with pytest.raises(ValueError, match="units real"):
        M.read_dcd(DATA / "traj_real.dcd", topology=DATA / "topology_real.data",
                   units="metal")


def test_dcd_without_topology_or_with_other_atoms_is_refused():
    with pytest.raises(ValueError) as error:
        M.read_dcd(DATA / "traj.dcd")
    assert "traj.dcd" in str(error.value) and "topology=" in str(error.value)
    with pytest.raises(ValueError, match="holds 12 atoms and its topology "
                                         "topology_real.data 6"):
        M.read_dcd(DATA / "traj.dcd", topology=DATA / "topology_real.data")


def test_dcd_type_map_reaches_the_topology(tmp_path):
    text = (DATA / "topology_real.data").read_text()
    stripped = re.sub(r"\nMasses\n\n(.*\n)+?\n", "\n", text, count=1)
    assert "Masses" not in stripped
    topo = tmp_path / "nomass.data"
    topo.write_text(stripped)
    with pytest.raises(ValueError, match="nomass.data could not be read"):
        M.read_dcd(DATA / "traj_real.dcd", topology=topo)
    traj = M.read_dcd(DATA / "traj_real.dcd", topology=topo, type_map=TYPES)
    assert traj.type_map_source == "user"
    assert sorted(traj.frame(0).composition.items()) == [("Na", 1), ("O", 4),
                                                         ("Si", 1)]


@pytest.mark.parametrize("variant", ["big-int32", "little-int64", "big-int64"])
def test_dcd_byte_order_and_marker_size_change_no_value(tmp_path, variant):
    raw = (DATA / "traj.dcd").read_bytes()
    data = {"big-int32": lambda: big_endian_dcd(raw),
            "little-int64": lambda: join(records(raw), "<", 8),
            "big-int64": lambda: big_endian_dcd(raw, 8)}[variant]()
    path = write(tmp_path, f"{variant}.dcd", data)
    assert M.sniff_dcd(data[:4096], path)
    ref = M.read_dcd(DATA / "traj.dcd", topology=DATA / "topology.data")
    traj = M.read_dcd(path, topology=DATA / "topology.data")
    for (p0, b0), (p1, b1) in zip(dcd_values(ref), dcd_values(traj)):
        assert np.array_equal(p0, p1) and np.array_equal(b0, b1)
    assert traj.timesteps.tolist() == ref.timesteps.tolist()
    assert np.array_equal(traj.times_ps, ref.times_ps)
    assert any(("big" if "big" in variant else "little") + "-endian" in n
               for n in traj.notes)


def test_dcd_cut_last_frame_is_skipped(tmp_path):
    raw = (DATA / "traj.dcd").read_bytes()
    path = write(tmp_path, "cut.dcd", raw[:-100])
    traj = M.read_dcd(path, topology=DATA / "topology.data")
    assert len(traj) == 2 and list(traj.skipped) == [2]
    assert traj.skipped[2].startswith("truncated")
    assert any("NSET = 3" in n for n in traj.notes)


def test_dcd_damaged_marker_names_the_frame(tmp_path):
    """Every frame's record markers are checked when the file is opened: a
    frame whose markers do not fit is in ``skipped`` with the reason (it
    was counted as readable, and frame(1) raised, before), and the frames
    after it keep their own timesteps and values."""
    recs = records((DATA / "traj.dcd").read_bytes())
    raw = bytearray(join(recs))
    header = sum(len(r) + 8 for r in recs[:3])
    frame = sum(len(r) + 8 for r in recs[3:7])
    struct.pack_into("<i", raw, header + frame + 56 + 4 + 48, 999)   # frame 1, x
    path = write(tmp_path, "damaged.dcd", bytes(raw))
    traj = M.read_dcd(path, topology=DATA / "topology.data")
    assert len(traj) == 2 and list(traj.skipped) == [1]
    assert "length markers 48 and 999" in traj.skipped[1]
    assert traj.timesteps.tolist() == [100, 120]
    assert traj.file_positions.tolist() == [0, 2]
    ref = M.read_dcd(DATA / "traj.dcd", topology=DATA / "topology.data")
    assert np.array_equal(traj.frame(1).cart_ang, ref.frame(2).cart_ang)
    # a frame damaged after the file was opened still names the frame
    later = M.read_dcd(DATA / "traj.dcd", topology=DATA / "topology.data")
    later._ranges[1] = (later._ranges[1][0] + 4, later._ranges[1][1] + 4)
    with pytest.raises(FrameError, match="frame 1.*length markers"):
        later.frame(1)


def test_dcd_without_cells_needs_box_from(tmp_path):
    raw = (DATA / "traj_real.dcd").read_bytes()
    path = write(tmp_path, "nocell.dcd", dcd_with(raw, icntrl={10: 0},
                                                   drop_cells=True))
    with pytest.raises(ValueError, match="box_from="):
        M.read_dcd(path, topology=DATA / "topology_real.data")
    traj = M.read_dcd(path, topology=DATA / "topology_real.data",
                      box_from=DATA / "topology_real.data")
    assert np.abs(traj.frame(2).box_ang - REAL["H"][2]).max() < 1e-12
    assert traj.box_varies is False
    rows = M.read_dcd(path, topology=DATA / "topology_real.data",
                      box_from=REAL["H"][0])
    assert np.array_equal(rows.frame(1).box_ang, REAL["H"][0])
    with pytest.raises(ValueError, match="its own unit cell"):
        M.read_dcd(DATA / "traj_real.dcd", topology=DATA / "topology_real.data",
                   box_from=REAL["H"][0])


def test_dcd_angles_in_degrees_and_akma_time_of_other_writers(tmp_path):
    raw = (DATA / "traj.dcd").read_bytes()

    def degrees(cell, k):
        out = cell.copy()
        for slot in (1, 3, 4):
            out[slot] = math.degrees(math.acos(cell[slot]))
        return out

    path = write(tmp_path, "namd25.dcd", dcd_with(raw, title="Created by NAMD",
                                                  cells=degrees))
    ref = M.read_dcd(DATA / "traj.dcd", topology=DATA / "topology.data")
    traj = M.read_dcd(path, topology=DATA / "topology.data")
    for k in range(3):
        assert np.abs(traj.frame(k).box_ang - ref.frame(k).box_ang).max() < 1e-12
    assert any("angles in degrees" in n for n in traj.notes)
    # a writer other than LAMMPS writes DELTA in AKMA time units
    assert np.allclose(traj.times_ps, np.array([100, 110, 120]) * 0.001
                       * M.AKMA_TIME_PS, rtol=1e-12, atol=0)
    assert "AKMA" in traj.units_note
    assert not any("box matrix" in n for n in traj.notes)
    # cosines of a tilted box from a writer other than LAMMPS: CHARMM's box
    # matrix would fill the same slots otherwise, and the note says so
    other = write(tmp_path, "vmd.dcd", dcd_with(raw, title="Created by VMD"))
    traj = M.read_dcd(other, topology=DATA / "topology.data")
    assert np.abs(traj.frame(0).box_ang - ref.frame(0).box_ang).max() == 0
    assert any("symmetric box matrix" in n for n in traj.notes)
    assert not any("box matrix" in n for n in ref.notes)


def test_akma_time_unit_matches_its_published_values():
    # MDAnalysis user guide: 1 AKMA = 4.888821e-14 s; NAMD list [5]: the
    # inverse in ps^-1 is sqrt(4184 * 1e20 * 1000) * 1e-12 (molar mass 1 g/mol)
    assert M.AKMA_TIME_PS == pytest.approx(4.888821e-2, rel=1e-7)
    assert M.AKMA_TIME_PS == pytest.approx(
        1.0 / (math.sqrt(4184.0 * 1e20 * 1000.0) * 1e-12), rel=1e-9)


@pytest.mark.parametrize("kind", ["VELD", "fixed"])
def test_dcd_velocity_files_and_fixed_atoms_are_refused(tmp_path, kind):
    raw = bytearray((DATA / "traj.dcd").read_bytes())
    if kind == "VELD":
        raw[4:8] = b"VELD"
        match = "VELD"
    else:
        raw = bytearray(dcd_with(bytes(raw), icntrl={8: 3}))
        match = "3 fixed atoms"
    path = write(tmp_path, f"{kind}.dcd", bytes(raw))
    with pytest.raises(UnsupportedFormat, match=match) as error:
        M.read_dcd(path, topology=DATA / "topology.data")
    assert f"{kind}.dcd" in str(error.value)


def test_dcd_cell_to_box_inverts_lammps_dim():
    xy, xz, yz, lx, ly, lz = 1.1, -0.6, 0.45, 7.5, 7.0, 7.75
    b, c = math.hypot(xy, ly), math.sqrt(xz * xz + yz * yz + lz * lz)
    # dump_dcd.cpp: dim = a, cos gamma, b, cos beta, cos alpha, c
    dim = [lx, lx * xy / (lx * b), b, lx * xz / (lx * c), (xy * xz + ly * yz) / (b * c),
           c]
    box, convention = M.dcd_cell_to_box(dim)
    assert convention == "lengths and angle cosines"
    assert np.abs(box - [[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]]).max() < 1e-14
    ortho, _ = M.dcd_cell_to_box([5.0, 0.0, 6.0, 0.0, 0.0, 7.0])
    assert np.array_equal(ortho, np.diag([5.0, 6.0, 7.0]))
    degrees, kind = M.dcd_cell_to_box([5.0, 90.0, 6.0, 90.0, 90.0, 7.0])
    assert kind == "lengths and angles in degrees"
    assert np.array_equal(degrees, np.diag([5.0, 6.0, 7.0]))
    for bad in ([5.0, 0.5, 6.0, 120.0, 0.0, 7.0], [-5.0, 0, 6.0, 0, 0, 7.0]):
        with pytest.raises(ValueError):
            M.dcd_cell_to_box(bad)


# ---------------------------------------------------------------------------
# LAMMPS binary dump, and its byte surgery
# ---------------------------------------------------------------------------

def parse_binary(raw: bytes) -> list[dict]:
    """The magic-string layout (revision 2) as binary2txt.cpp reads it, coded
    here independently of the module."""
    frames, pos = [], 0

    def take(fmt):
        nonlocal pos
        values = struct.unpack_from("<" + fmt, raw, pos)
        pos += struct.calcsize("<" + fmt)
        return values

    while pos < len(raw):
        (neg,) = take("q")
        magic = raw[pos:pos - neg]
        pos -= neg
        endian, revision, step, natoms, tri = take("iiqqi")
        boundary = take("6i")
        box = take({0: "6d", 1: "9d", 2: "12d"}[tri])
        (size_one,) = take("i")
        (ulen,) = take("i")
        units = raw[pos:pos + ulen]
        pos += ulen
        flag = raw[pos:pos + 1]
        pos += 1
        time = take("d")[0] if flag != b"\0" else None
        (clen,) = take("i")
        columns = raw[pos:pos + clen]
        pos += clen
        (nchunk,) = take("i")
        values = []
        for _ in range(nchunk):
            (n,) = take("i")
            values.extend(take(f"{n}d"))
        frames.append(dict(magic=magic, revision=revision, step=step,
                           natoms=natoms, tri=tri, boundary=boundary, box=box,
                           size_one=size_one, units=units, time=time,
                           columns=columns, values=values))
    return frames


def emit_binary(frames, *, order="<", layout="rev2", chunks=1) -> bytes:
    out = []
    for f in frames:
        p = lambda fmt, *v: struct.pack(order + fmt, *v)    # noqa: E731
        if layout != "old":
            out.append(p("q", -len(f["magic"])) + f["magic"] + p("ii", 1,
                       2 if layout == "rev2" else 1))
        out.append(p("qqi", f["step"], f["natoms"], f["tri"]) + p("6i", *f["boundary"])
                   + p(f"{len(f['box'])}d", *f["box"]) + p("i", f["size_one"]))
        if layout == "rev2":
            out.append(p("i", len(f["units"])) + f["units"])
            out.append(b"\0" if f["time"] is None else b"\1" + p("d", f["time"]))
            out.append(p("i", len(f["columns"])) + f["columns"])
        values = f["values"]
        rows = len(values) // f["size_one"]
        cut = [0, (rows // 2) * f["size_one"], len(values)] if chunks == 2 \
            else [0, len(values)]
        out.append(p("i", len(cut) - 1))
        for a, b in zip(cut, cut[1:]):
            out.append(p("i", b - a) + p(f"{b - a}d", *values[a:b]))
    return b"".join(out)


def test_binary_custom_dump_matches_lammps_to_float64():
    traj = M.read_lammps_binary_dump(DATA / "traj_custom.bin", type_map=TYPES)
    assert traj.file_format == "lammps-dump-binary" and len(traj) == 3
    assert traj.timesteps.tolist() == [100, 110, 120]
    assert np.allclose(traj.times_ps, METAL["time"], rtol=0, atol=1e-15)
    assert "units metal" in traj.units_note
    for k in range(3):
        frame = traj.frame(k)
        assert frame.atom_id.tolist() == METAL["ids"].tolist()
        assert frame.elements.tolist() == ELEMENTS.tolist()
        assert modlat(frame.cart_ang, METAL["x"][k], METAL["H"][k]) < 1e-12
        assert np.abs(frame.box_ang - METAL["H"][k]).max() < 1e-12
        assert np.abs(frame.origin_ang - METAL["lo"][k]).max() < 1e-12
        assert np.abs(frame.unwrapped_cart_ang - unwrapped_truth(METAL, k)).max() < 1e-12
        assert np.abs(frame.vel_ang_per_ps - METAL["v"][k]).max() < 1e-12
        assert np.array_equal(frame.charge_e, METAL["q"])
    assert traj.periodic == (True, True, True)


def test_binary_atom_dump_with_masses_from():
    traj = M.read_lammps_binary_dump(DATA / "traj_atom.bin",
                                     masses_from=DATA / "topology.data")
    for k in range(3):
        frame = traj.frame(k)
        assert modlat(frame.cart_ang, METAL["x"][k], METAL["H"][k]) < 1e-12
        assert frame.elements.tolist() == ELEMENTS.tolist()
    assert any("magic string DUMPATOM revision 2" in n for n in traj.notes)


def test_binary_element_column_holds_type_numbers():
    traj = M.read_lammps_binary_dump(DATA / "traj_element.bin",
                                     masses_from=DATA / "topology.data")
    assert traj.frame(1).elements.tolist() == ELEMENTS.tolist()
    assert any("hold the type number" in n for n in traj.notes)
    assert any(n.startswith("columns read: id element x y z") for n in traj.notes)
    with pytest.raises(ValueError, match="type_map="):
        M.read_lammps_binary_dump(DATA / "traj_element.bin")


@pytest.mark.parametrize("layout,order,chunks", [
    ("rev2", ">", 1), ("rev2", "<", 2), ("old", "<", 1), ("old", ">", 2),
    ("rev1", "<", 1)])
def test_binary_layouts_byte_orders_and_chunks_change_no_value(
        tmp_path, layout, order, chunks):
    frames = parse_binary((DATA / "traj_custom.bin").read_bytes())
    data = emit_binary(frames, order=order, layout=layout, chunks=chunks)
    path = write(tmp_path, "variant.bin", data)
    assert M.sniff_lammps_binary(data[:4096], path)
    columns = frames[0]["columns"].decode()
    ref = M.read_lammps_binary_dump(DATA / "traj_custom.bin", type_map=TYPES)
    if layout != "rev2":
        with pytest.raises(ValueError, match="columns=") as error:
            M.read_lammps_binary_dump(path, type_map=TYPES)
        assert "variant.bin" in str(error.value)
        traj = M.read_lammps_binary_dump(path, type_map=TYPES, columns=columns,
                                         units="metal")
    else:
        traj = M.read_lammps_binary_dump(path, type_map=TYPES)
    for k in range(3):
        a, b = ref.frame(k), traj.frame(k)
        for name in ("cart_ang", "box_ang", "origin_ang", "unwrapped_cart_ang",
                     "vel_ang_per_ps", "charge_e"):
            assert np.array_equal(getattr(a, name), getattr(b, name)), name


def test_binary_old_layout_is_recognised_only_under_a_binary_name(tmp_path):
    frames = parse_binary((DATA / "traj_custom.bin").read_bytes())
    data = emit_binary(frames, layout="old")
    assert M.sniff_lammps_binary(data[:4096], tmp_path / "dump.lammpsbin")
    assert not M.sniff_lammps_binary(data[:4096], tmp_path / "dump.dat")
    with pytest.raises(ValueError, match="names its columns"):
        M.read_lammps_binary_dump(DATA / "traj_custom.bin", type_map=TYPES,
                                  columns="id type x y z")


def test_binary_cut_and_damaged_frames_are_listed(tmp_path):
    raw = (DATA / "traj_custom.bin").read_bytes()
    cut = M.read_lammps_binary_dump(write(tmp_path, "cut.bin", raw[:-50]),
                                    type_map=TYPES)
    assert len(cut) == 2 and cut.skipped[2].startswith("truncated")
    frames = parse_binary(raw)
    first = len(emit_binary(frames[:1]))
    second = len(emit_binary(frames[:2]))
    damaged = bytearray(raw)
    damaged[first + 8:first + 12] = b"XXXX"          # frame 1's magic string
    traj = M.read_lammps_binary_dump(write(tmp_path, "damaged.bin", bytes(damaged)),
                                     type_map=TYPES)
    # the magic-string layout: the pass finds frame 2's magic string and
    # reads on, and the damaged stretch takes one position with its bytes
    assert len(traj) == 2 and traj.timesteps.tolist() == [100, 120]
    reason = traj.skipped[1]
    assert "unreadable header" in reason and f"bytes {first} to {second}" in reason
    ref = M.read_lammps_binary_dump(DATA / "traj_custom.bin", type_map=TYPES)
    assert np.array_equal(traj.frame(1).cart_ang, ref.frame(2).cart_ang)
    # the older layout has no marker to search for: the rest is not located
    old = bytearray(emit_binary(frames, layout="old"))
    old_first = len(emit_binary(frames[:1], layout="old"))
    struct.pack_into("<q", old, old_first + 8, -1)   # frame 1's atom count
    traj = M.read_lammps_binary_dump(write(tmp_path, "old.bin", bytes(old)),
                                     type_map=TYPES, columns=frames[0]["columns"]
                                     .decode(), units="metal")
    assert len(traj) == 1
    reason = traj.skipped[1]
    assert "unreadable header" in reason
    assert f"{len(old) - old_first} bytes" in reason


def test_binary_gzip_reads_the_same(tmp_path):
    path = tmp_path / "traj.bin.gz"
    path.write_bytes(gzip.compress((DATA / "traj_custom.bin").read_bytes()))
    ref = M.read_lammps_binary_dump(DATA / "traj_custom.bin", type_map=TYPES)
    traj = M.read_lammps_binary_dump(path, type_map=TYPES)
    assert np.array_equal(ref.frame(2).cart_ang, traj.frame(2).cart_ang)


# ---------------------------------------------------------------------------
# YAML dump
# ---------------------------------------------------------------------------

def test_yaml_dump_matches_lammps_to_its_printed_digits():
    traj = M.read_lammps_yaml_dump(DATA / "traj.yaml")
    assert traj.file_format == "lammps-dump-yaml" and len(traj) == 3
    assert traj.timesteps.tolist() == [100, 110, 120]
    assert np.allclose(traj.times_ps, METAL["time"], rtol=0, atol=1e-15)
    assert traj.type_map_source == "element column"
    for k in range(3):
        frame = traj.frame(k)
        assert frame.elements.tolist() == ELEMENTS.tolist()
        bound = G6 * np.abs(METAL["x"][k]).max()
        assert modlat(frame.cart_ang, METAL["x"][k], METAL["H"][k]) <= bound
        # the box is printed with fmt's shortest round-trip form: exact
        assert np.abs(frame.box_ang - METAL["H"][k]).max() < 1e-12
        expected = unwrapped_truth(METAL, k)
        assert np.abs(frame.unwrapped_cart_ang - expected).max() <= bound
        assert np.abs(frame.vel_ang_per_ps - METAL["v"][k]).max() <= \
            G6 * np.abs(METAL["v"][k]).max()
    assert any("thermo data" in n for n in traj.notes)


def test_yaml_typelabel_column_is_empty_and_refused_with_advice():
    with pytest.raises(ValueError) as error:
        M.read_lammps_yaml_dump(DATA / "traj_typelabel.yaml")
    text = str(error.value)
    assert "traj_typelabel.yaml" in text and "typelabel" in text
    assert "type_map=" in text and "dump_modify element" in text


def test_yaml_typelabel_column_is_dropped_when_types_remain(tmp_path):
    text = (DATA / "traj.yaml").read_text()
    text = text.replace("keywords: [ id, type, element,",
                        "keywords: [ id, type, typelabel,")
    text = re.sub(r"(\d+ , \d+ , )(?:Si|Na|O) ,", r"\1,", text)
    path = tmp_path / "tl.yaml"
    path.write_text(text)
    traj = M.read_lammps_yaml_dump(path, type_map=TYPES)
    assert any("empty in every row" in n for n in traj.notes)
    ref = M.read_lammps_yaml_dump(DATA / "traj.yaml")
    assert np.array_equal(traj.frame(2).cart_ang, ref.frame(2).cart_ang)
    assert traj.frame(2).elements.tolist() == ELEMENTS.tolist()


def test_yaml_cut_last_frame_and_gzip(tmp_path):
    raw = (DATA / "traj.yaml").read_bytes()
    cut = M.read_lammps_yaml_dump(write(tmp_path, "cut.yaml",
                                        raw[:raw.rfind(b"  - [") - 200]))
    assert len(cut) == 2 and cut.skipped[2].startswith("truncated")
    zipped = M.read_lammps_yaml_dump(write(tmp_path, "z.yaml.gz",
                                           gzip.compress(raw)))
    assert np.array_equal(zipped.frame(1).cart_ang,
                          M.read_lammps_yaml_dump(DATA / "traj.yaml").frame(1).cart_ang)


def test_yaml_crlf_copy_parses_identically(tmp_path):
    raw = (DATA / "traj.yaml").read_bytes().replace(b"\r\n", b"\n")
    traj = M.read_lammps_yaml_dump(write(tmp_path, "crlf.yaml",
                                         raw.replace(b"\n", b"\r\n")))
    ref = M.read_lammps_yaml_dump(DATA / "traj.yaml")
    for k in range(3):
        assert np.array_equal(traj.frame(k).cart_ang, ref.frame(k).cart_ang)


# ---------------------------------------------------------------------------
# AtomEye CFG
# ---------------------------------------------------------------------------

def cfg_paths(folder: str) -> list[Path]:
    return sorted((DATA / folder).glob("dump.*.cfg"),
                  key=lambda p: int(p.name.split(".")[1]))


def test_cfg_series_matches_lammps_relative_to_the_origin():
    traj = M.read_cfg_series(cfg_paths("cfg"), timestep_fs=1.0)
    assert traj.file_format == "atomeye-cfg" and len(traj) == 3
    assert traj.timesteps.tolist() == [100, 110, 120]
    assert np.allclose(traj.times_ps, [0.1, 0.11, 0.12], rtol=0, atol=1e-15)
    assert traj.charges_e == {"Si": 2.4, "Na": 0.6, "O": -1.2}
    for k in range(3):
        frame = traj.frame(k)
        assert frame.atom_id.tolist() == METAL["ids"].tolist()
        assert frame.elements.tolist() == ELEMENTS.tolist()
        relative = METAL["x"][k] - METAL["lo"][k]      # a CFG has no origin
        bound = G6 * (np.abs(relative).max() + np.abs(METAL["H"][k]).max())
        assert modlat(frame.cart_ang, relative, METAL["H"][k]) <= bound
        assert np.abs(frame.box_ang - METAL["H"][k]).max() <= \
            G6 * np.abs(METAL["H"][k]).max()
        expected = unwrapped_truth(METAL, k) - METAL["lo"][k]
        assert np.abs(frame.unwrapped_cart_ang - expected).max() <= \
            bound * (1 + np.abs(METAL["image"][k]).max())


def test_cfg_lammps_unwrapped_layout_is_read_back():
    traj = M.read_cfg_series(cfg_paths("cfg_unwrap"))
    assert any("UNWRAPEXPAND" in n for n in traj.notes)
    for k in range(3):
        frame = traj.frame(k)
        assert np.abs(frame.box_ang - METAL["H"][k]).max() <= \
            G6 * np.abs(METAL["H"][k]).max()
        expected = unwrapped_truth(METAL, k) - METAL["lo"][k]
        # s' is printed with %g and multiplied by 10: one digit less
        bound = 10 * G6 * np.abs(METAL["H"][k]).sum(axis=0).max() * 10
        assert np.abs(frame.unwrapped_cart_ang - expected).max() <= bound


def test_cfg_without_element_names_is_refused_with_the_dump_modify_advice():
    with pytest.raises(ValueError) as error:
        M.read_cfg(DATA / "cfg_noelement" / "dump.100.cfg")
    text = str(error.value)
    assert "dump.100.cfg" in text and "dump_modify" in text and "element" in text


def test_ase_cfg_matches_the_model():
    traj = M.read_cfg(DATA / "ase.cfg")
    frame = traj.frame(0)
    relative = METAL["x"][0] - METAL["lo"][0]
    assert frame.elements.tolist() == ELEMENTS.tolist()
    assert modlat(frame.cart_ang, relative, METAL["H"][0]) < 1e-5
    assert np.abs(frame.box_ang - METAL["H"][0]).max() < 1e-5
    assert any("velocity columns are not read" in n for n in traj.notes)
    with pytest.raises(ValueError, match="timestep_fs needs timesteps"):
        M.read_cfg(DATA / "ase.cfg", timestep_fs=1.0)


def test_cfg_standard_layout_transform_and_label_map(tmp_path):
    head = ("Number of particles = 2\nA = 2.0 Angstrom (basic length-scale)\n"
            "H0(1,1) = 5 A\nH0(1,2) = 0 A\nH0(1,3) = 0 A\nH0(2,1) = 0 A\n"
            "H0(2,2) = 5 A\nH0(2,3) = 0 A\nH0(3,1) = 0 A\nH0(3,2) = 0 A\n"
            "H0(3,3) = 5 A\n")
    path = tmp_path / "std.cfg"
    path.write_text(head + "28.0855 Si 0.1 0.2 0.3 0 0 0\n"
                    "15.9994 O 0.5 0.5 0.5 0 0 0\n")
    frame = M.read_cfg(path).frame(0)
    assert np.array_equal(frame.box_ang, np.diag([10.0, 10.0, 10.0]))
    assert np.abs(frame.cart_ang[0] - [1.0, 2.0, 3.0]).max() < 1e-12
    transform = head + "".join(f"Transform({i},{j}) = {2 if i == j == 3 else int(i == j)}\n"
                               for i in (1, 2, 3) for j in (1, 2, 3))
    path.write_text(transform + ".NO_VELOCITY.\nentry_count = 3\n28.0855\nSi\n"
                    "0.1 0.2 0.3\n0.5 0.5 0.5\n")
    frame = M.read_cfg(path).frame(0)
    assert np.array_equal(frame.box_ang, np.diag([10.0, 10.0, 20.0]))
    path.write_text(head + ".NO_VELOCITY.\nentry_count = 3\n15.9994\nOb\n"
                    "0.1 0.2 0.3\n0.5 0.5 0.5\n")
    with pytest.raises(ValueError, match="type_map="):
        M.read_cfg(path)
    traj = M.read_cfg(path, type_map={"Ob": "O"})
    assert traj.frame(0).elements.tolist() == ["O", "O"]
    assert traj.type_map_source == "user" and traj.type_map == {"Ob": "O"}


def test_timesteps_from_names():
    assert M.timesteps_from_names(["d.0.cfg", "d.10.cfg", "d.100.cfg"]) == [0, 10, 100]
    assert M.timesteps_from_names(["run2.000100.cfg", "run2.000200.cfg"]) == [100, 200]
    assert M.timesteps_from_names(["a1.cfg", "b2.cfg"]) is None
    assert M.timesteps_from_names(["d.1.2.cfg", "d.2.3.cfg"]) is None
    assert M.timesteps_from_names(["d.100.cfg"]) is None


# ---------------------------------------------------------------------------
# the contract, recognition, and words
# ---------------------------------------------------------------------------

FIXTURES = {
    "dcd": ["traj.dcd", "traj_unwrap.dcd", "traj_real.dcd"],
    "lammps-dump-binary": ["traj_custom.bin", "traj_atom.bin", "traj_element.bin"],
    "lammps-dump-yaml": ["traj.yaml", "traj_typelabel.yaml"],
    "atomeye-cfg": ["ase.cfg", "cfg/dump.100.cfg", "cfg_unwrap/dump.110.cfg",
                    "cfg_noelement/dump.100.cfg"],
}
OTHERS = ["topology.data", "truth.npz", "make_lammps_formats.py"]
MD_OTHERS = ["dump_tri_x.lammpstrj", "glass.extxyz", "XDATCAR_fixed", "CONFIG",
             "HISTORY", "data_charge.data", "dump_tri_x.lammpstrj.gz"]


def test_formats_follow_the_contract():
    names = [spec.name for spec in M.FORMATS]
    assert names == ["dcd", "lammps-dump-binary", "lammps-dump-yaml", "atomeye-cfg"]
    for spec in M.FORMATS:
        assert isinstance(spec, md_formats_base.FormatSpec)
        assert spec.options <= md_formats_base.KNOWN_OPTIONS
    assert [s.binary for s in M.FORMATS] == [True, True, False, False]
    assert M.FORMATS[3].read_series is M.read_cfg_series
    assert "md_formats_lammps" in md_formats_base.FORMAT_MODULES


def test_each_sniffer_takes_its_own_files_only():
    head = lambda p: md_readers._head(p, md_readers.SNIFF_BYTES)    # noqa: E731
    files = {name: DATA / name for group in FIXTURES.values() for name in group}
    others = [DATA / n for n in OTHERS] + [MD_DATA / n for n in MD_OTHERS]
    for spec in M.FORMATS:
        for name, path in files.items():
            assert spec.sniff(head(path), path) == (name in FIXTURES[spec.name]), \
                (spec.name, name)
        for path in others:
            assert not spec.sniff(head(path), path), (spec.name, path.name)
        assert spec.sniff(b"", Path("x")) is False


def test_read_through_the_spec_with_options():
    spec = {s.name: s for s in M.FORMATS}
    traj = spec["dcd"].read(DATA / "traj.dcd", topology=DATA / "topology.data")
    assert len(traj) == 3
    traj = spec["atomeye-cfg"].read_series(cfg_paths("cfg"))
    assert len(traj) == 3
    with traj:
        pass


def test_read_trajectory_reaches_these_formats():
    """Through md_readers.read_trajectory, the entry point the app uses: the
    format comes from the content, options reach the reader, a CFG directory
    or pattern is read as one series in natural order of the names."""
    assert md_readers.sniff_md(DATA / "traj.dcd") == "dcd"
    with md_readers.read_trajectory(DATA / "traj.dcd",
                                    topology=DATA / "topology.data") as traj:
        assert traj.file_format == "dcd" and len(traj) == 3
    with pytest.raises(ValueError, match="topology="):
        md_readers.read_trajectory(DATA / "traj.dcd")
    with md_readers.read_trajectory(DATA / "traj_custom.bin", type_map=TYPES) as traj:
        assert traj.file_format == "lammps-dump-binary"
    with md_readers.read_trajectory(DATA / "traj.yaml") as traj:
        assert traj.file_format == "lammps-dump-yaml"
    for series in (DATA / "cfg", str(DATA / "cfg" / "dump.*.cfg")):
        with md_readers.read_trajectory(series, timestep_fs=1.0) as traj:
            assert traj.file_format == "atomeye-cfg"
            assert traj.timesteps.tolist() == [100, 110, 120]
            assert np.allclose(traj.times_ps, [0.1, 0.11, 0.12])


VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def _every_note_and_message(tmp_path) -> list[str]:
    out: list[str] = []
    for traj in (M.read_dcd(DATA / "traj.dcd", topology=DATA / "topology.data"),
                 M.read_dcd(DATA / "traj_unwrap.dcd", topology=DATA / "topology.data"),
                 M.read_dcd(DATA / "traj_real.dcd", topology=DATA / "topology_real.data"),
                 M.read_lammps_binary_dump(DATA / "traj_custom.bin", type_map=TYPES),
                 M.read_lammps_binary_dump(DATA / "traj_element.bin",
                                           masses_from=DATA / "topology.data"),
                 M.read_lammps_yaml_dump(DATA / "traj.yaml"),
                 M.read_cfg_series(cfg_paths("cfg")),
                 M.read_cfg_series(cfg_paths("cfg_unwrap")),
                 M.read_cfg(DATA / "ase.cfg")):
        out += traj.describe()
        out += [n for k in range(len(traj)) for n in traj.frame(k).notes]

    def message(call):
        try:
            call()
        except (ValueError, OSError) as error:
            return str(error)
        raise AssertionError("expected a refusal")

    out.append(message(lambda: M.read_dcd(DATA / "traj.dcd")))
    out.append(message(lambda: M.read_dcd(DATA / "traj.dcd",
                                          topology=DATA / "topology_real.data")))
    out.append(message(lambda: M.read_lammps_binary_dump(DATA / "traj_element.bin")))
    out.append(message(lambda: M.read_lammps_yaml_dump(DATA / "traj_typelabel.yaml")))
    out.append(message(lambda: M.read_cfg(DATA / "cfg_noelement" / "dump.100.cfg")))
    frames = parse_binary((DATA / "traj_custom.bin").read_bytes())
    old = write(tmp_path, "old.bin", emit_binary(frames, layout="old"))
    out.append(message(lambda: M.read_lammps_binary_dump(old)))
    out.append(message(lambda: M.read_dcd(DATA / "traj.yaml",
                                          topology=DATA / "topology.data")))
    return out


def test_no_note_or_message_carries_a_verdict(tmp_path):
    notes = _every_note_and_message(tmp_path)
    assert len(notes) > 100
    assert [n for n in notes if VERDICT.search(n)] == []


def test_no_string_in_the_module_carries_a_verdict():
    source = (ROOT / "facet" / "core" / "md_formats_lammps.py").read_text(
        encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 150
    assert [s for s in strings if VERDICT.search(s)] == []


def test_the_module_imports_without_qt():
    code = ("import sys, facet.core.md_formats_lammps;"
            "print('QT' if any(m.startswith('PySide6') for m in sys.modules) "
            "else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


# ---------------------------------------------------------------------------
# edge cases from the verification passes (files in edges/, written by
# LAMMPS 22 Jul 2025, mdtraj 1.11 and MDAnalysis 2.10: make_lammps_edges.py)
# ---------------------------------------------------------------------------

NO_STEP = -1              # md_model.NO_TIMESTEP


def message_of(call) -> str:
    try:
        call()
    except (ValueError, OSError) as error:
        return str(error)
    raise AssertionError("expected a refusal")


def edge_truth(name: str, k: int) -> np.ndarray:
    """LAMMPS's positions (rows in id order) after the k-th dumped step."""
    return np.array(E[name]["x"][k])


# -- DCD ---------------------------------------------------------------------

def test_dcd_first_yes_timesteps_and_times_match_lammps(tmp_path):
    """dump_modify first yes writes the first snapshot on a step that is not
    a multiple of N (105) and the later ones on multiples (120, 140); the
    header holds ISTART 105, NSAVC 20 and the last step 140, which tell the
    layouts apart (ISTART + k NSAVC gave 125 and 145)."""
    traj = M.read_dcd(EDGES / "first_yes.dcd", topology=EDGES / "first_yes.data")
    assert traj.timesteps.tolist() == E["first_yes"]["steps"] == [105, 120, 140]
    assert np.allclose(traj.times_ps, E["first_yes"]["times"], rtol=0, atol=1e-12)
    assert any("dump_modify first yes" in n for n in traj.notes)
    assert "time yes" in traj.units_note      # where a changed dt is recorded
    # a last-step field that neither layout gives: no timesteps and no times
    odd = bytearray((EDGES / "first_yes.dcd").read_bytes())
    struct.pack_into("<i", odd, 4 + 4 + 4 * 3, 150)          # ICNTRL[3]
    path = write(tmp_path, "odd_last.dcd", bytes(odd))
    traj = M.read_dcd(path, topology=EDGES / "first_yes.data")
    assert traj.timesteps.tolist() == [NO_STEP] * 3 and traj.times_ps is None
    assert any("timesteps are not known" in n for n in traj.notes)
    text = message_of(lambda: M.read_dcd(path, topology=EDGES / "first_yes.data",
                                         timestep_fs=1.0))
    assert text.startswith("odd_last.dcd") and "timestep_fs needs timesteps" in text


def test_dcd_of_the_molfile_plugin_has_no_time_axis():
    """mdtraj (VMD's molfile plugin) writes ISTART 0, NSAVC 1, DELTA 1.0 for
    every file; read as AKMA they gave a 48.9 fs frame spacing."""
    traj = M.read_dcd(EDGES / "plugin.dcd", topology=DATA / "topology.data")
    assert traj.timesteps.tolist() == [NO_STEP] * 3 and traj.times_ps is None
    assert any("molfile" in n and "placeholders" in n for n in traj.notes)
    assert "AKMA" not in traj.units_note
    for k in range(3):
        frame = traj.frame(k)
        written = (METAL["x"][k] - METAL["lo"][k]).astype(np.float32)
        # the positions as written, modulo the box the file states
        assert modlat(frame.cart_ang, written, frame.box_ang) < 1e-12
        # the cell is written as float32 lengths and angles
        assert np.abs(frame.box_ang - METAL["H"][k]).max() < 1e-5
    timed = M.read_dcd(EDGES / "plugin.dcd", topology=DATA / "topology.data",
                       timestep_fs=2.0)
    assert timed.timesteps.tolist() == [0, 1, 2]
    assert np.allclose(timed.times_ps, [0.0, 0.002, 0.004], rtol=0, atol=1e-15)


def test_dcd_with_a_zeroed_cell_reads_with_box_from():
    """MDAnalysis writes [0, 1, 0, 1, 1, 0] for a trajectory without
    dimensions: no box. It was refused bare (lengths 0) and with box_from
    ('has its own unit cell')."""
    text = message_of(lambda: M.read_dcd(EDGES / "mda_nocell.dcd",
                                         topology=DATA / "topology.data"))
    assert text.startswith("mda_nocell.dcd") and "box_from=" in text
    traj = M.read_dcd(EDGES / "mda_nocell.dcd", topology=DATA / "topology.data",
                      box_from=METAL["H"][0])
    assert traj.box_varies is False
    assert any("zeroed cell" in n for n in traj.notes)
    for k in range(3):
        frame = traj.frame(k)
        assert np.array_equal(frame.box_ang, METAL["H"][0])
        written = (METAL["x"][k] - METAL["lo"][k]).astype(np.float32)
        assert modlat(frame.cart_ang, written, METAL["H"][0]) < 1e-12


def test_dcd_cell_damage_is_found_at_open(tmp_path):
    raw = (DATA / "traj.dcd").read_bytes()

    def nan_in_two(cell, k):
        return cell * np.nan if k == 2 else cell

    traj = M.read_dcd(write(tmp_path, "nan.dcd", dcd_with(raw, cells=nan_in_two)),
                      topology=DATA / "topology.data")
    assert len(traj) == 2 and list(traj.skipped) == [2]
    assert "finite" in traj.skipped[2]
    # bytes appended after the last frame: the reason gives their count
    traj = M.read_dcd(write(tmp_path, "tail.dcd", raw + b"\0" * 100),
                      topology=DATA / "topology.data")
    assert len(traj) == 3
    assert "100 bytes after its last complete frame" in traj.skipped[3]


def test_dcd_record_length_beyond_the_file_allocates_little(tmp_path):
    """A title marker of 2^31 - 1 in a 948-byte file: the read asked for
    2 GB at once (peak working set 2.2 GB) before finding the end."""
    raw = (DATA / "traj.dcd").read_bytes()
    path = write(tmp_path, "title_len.dcd",
                 raw[:92] + struct.pack("<i", 2 ** 31 - 1) + raw[96:])
    tracemalloc.start()
    try:
        text = message_of(lambda: M.read_dcd(path, topology=DATA / "topology.data"))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert text.startswith("title_len.dcd") and "ends inside the title" in text
    assert peak < 64 * 2 ** 20


def test_option_refusals_and_empty_files_name_the_file(tmp_path):
    text = message_of(lambda: M.read_dcd(DATA / "traj.dcd",
                                         topology=DATA / "topology.data",
                                         timestep_fs=-1))
    assert text.startswith("traj.dcd: timestep_fs")
    text = message_of(lambda: md_readers.read_trajectory(DATA / "cfg",
                                                         timestep_fs="abc"))
    assert "dump.100.cfg" in text and "timestep_fs 'abc'" in text
    text = message_of(lambda: M.read_lammps_binary_dump(
        DATA / "traj_custom.bin", type_map={1: "Xx"}))
    assert text.startswith("traj_custom.bin")
    for name, reader in (("e.dcd", lambda p: M.read_dcd(p, topology=DATA /
                                                        "topology.data")),
                         ("e.bin", M.read_lammps_binary_dump),
                         ("e.yaml", M.read_lammps_yaml_dump),
                         ("e.cfg", M.read_cfg)):
        path = write(tmp_path, name, b"")
        text = message_of(lambda: reader(path))
        assert text.startswith(name) and "the file is empty" in text


# -- binary dump ---------------------------------------------------------------

def test_binary_zero_atom_frames_are_skipped_and_the_rest_read():
    """dump_modify thresh wrote 0-atom frames at steps 30 and 40; the header
    check refused natoms 0, so steps 50 and 60 were lost."""
    tmap = {1: "Si", 2: "O"}
    traj = M.read_lammps_binary_dump(EDGES / "thresh.bin", type_map=tmap)
    yaml = M.read_lammps_yaml_dump(EDGES / "thresh.yaml", type_map=tmap)
    assert traj.timesteps.tolist() == yaml.timesteps.tolist() == [0, 10, 20, 50, 60]
    assert sorted(traj.skipped) == [3, 4]
    assert all("0 atoms" in r or "is 0" in r for r in traj.skipped.values())
    for k, step_index in ((3, 5), (4, 6)):
        assert np.abs(traj.frame(k).cart_ang
                      - edge_truth("thresh", step_index)).max() < 1e-12


def test_binary_gzip_cut_inside_a_chunk_keeps_the_frames_before(tmp_path):
    """A .bin.gz copied while LAMMPS wrote it: a cut inside a chunk raised a
    raw EOFError naming no file."""
    raw = (DATA / "traj_custom.bin").read_bytes() * 40
    packed = gzip.compress(raw)
    path = write(tmp_path, "cut.bin.gz", packed[:len(packed) // 2])
    traj = M.read_lammps_binary_dump(path, type_map=TYPES)
    last = max(traj.skipped)
    assert traj.skipped[last].startswith("truncated") and last == len(traj)
    assert any("end-of-stream marker" in n for n in traj.notes)
    ref = M.read_lammps_binary_dump(DATA / "traj_custom.bin", type_map=TYPES)
    assert np.array_equal(traj.frame(len(traj) - 1).cart_ang,
                          ref.frame((len(traj) - 1) % 3).cart_ang)
    # a damaged stream: refused naming the file, never zlib.error or a raw
    # gzip error without the name
    for at in range(len(packed) // 10, len(packed) - 20, len(packed) // 9):
        flipped = bytearray(packed)
        flipped[at] ^= 0x5A
        path = write(tmp_path, "flip.bin.gz", bytes(flipped))
        try:
            M.read_lammps_binary_dump(path, type_map=TYPES)
        except UnsupportedFormat as error:
            assert "flip.bin.gz" in str(error)
        except ValueError as error:
            assert "flip.bin.gz" in str(error)


def test_binary_gzip_cut_at_a_frame_boundary_says_so(tmp_path):
    raw = (DATA / "traj_custom.bin").read_bytes()
    frames = parse_binary(raw)
    two = len(emit_binary(frames[:2]))
    stream = zlib.compressobj(9, zlib.DEFLATED, 31)
    path = write(tmp_path, "sync.bin.gz",
                 stream.compress(raw[:two]) + stream.flush(zlib.Z_SYNC_FLUSH))
    traj = M.read_lammps_binary_dump(path, type_map=TYPES)
    assert len(traj) == 2 and not traj.skipped
    assert md_readers._GZIP_CUT_NOTE in traj.notes


def test_binary_damaged_header_resyncs_at_the_next_magic_string(tmp_path):
    raw = (DATA / "traj_custom.bin").read_bytes()
    frames = parse_binary(raw)
    first, second = len(emit_binary(frames[:1])), len(emit_binary(frames[:2]))
    ref = M.read_lammps_binary_dump(DATA / "traj_custom.bin", type_map=TYPES)
    for at, kept, steps in ((first, [0, 2], [100, 120]), (0, [1, 2], [110, 120])):
        damaged = bytearray(raw)
        struct.pack_into("<q", damaged, at + 34, -1)          # the atom count
        traj = M.read_lammps_binary_dump(
            write(tmp_path, f"bad{at}.bin", bytes(damaged)), type_map=TYPES)
        assert traj.timesteps.tolist() == steps
        position = 1 if at else 0
        end = second if at else first
        assert f"bytes {at} to {end}" in traj.skipped[position]
        for k, original in enumerate(kept):
            assert np.array_equal(traj.frame(k).cart_ang, ref.frame(original).cart_ang)
    # frame 0's header damaged: not taken for a layout without column names
    damaged = bytearray(raw)
    struct.pack_into("<q", damaged, 34, -1)
    traj = M.read_lammps_binary_dump(write(tmp_path, "f0.bin", bytes(damaged)),
                                     type_map=TYPES)
    assert not any("revision 1" in n for n in traj.notes)
    text = message_of(lambda: M.read_lammps_binary_dump(
        write(tmp_path, "cut60.bin", raw[:60]), type_map=TYPES))
    assert "holds no readable frame" in text and "truncated" in text
    assert "columns=" not in text
    # cut inside the magic string's header: still recognised, and the
    # refusal says the file ends there (not 'tools/binary2txt')
    cut = write(tmp_path, "cut20.bin", raw[:20])
    assert md_readers.sniff_md(cut) == "lammps-dump-binary"
    text = message_of(lambda: md_readers.read_trajectory(cut, type_map=TYPES))
    assert text.startswith("cut20.bin") and "truncated" in text


def test_binary_and_yaml_colname_dumps_read_with_columns():
    """dump_modify colname x X: the refusal said nothing about what to do,
    and columns= was refused for a file that names its columns."""
    for name, reader in (("colname.bin", M.read_lammps_binary_dump),
                         ("colname.yaml", M.read_lammps_yaml_dump)):
        text = message_of(lambda: reader(EDGES / name, type_map=TYPES))
        assert text.startswith(name) and "columns=" in text and "colname" in text
        traj = reader(EDGES / name, type_map=TYPES, columns="id type x y z")
        assert any("columns= replaces" in n and "X" in n for n in traj.notes)
        tol = 1e-12 if name.endswith(".bin") else G6 * 9.0
        for k in range(2):
            frame = traj.frame(k)
            assert frame.atom_id.tolist() == E["colname"]["ids"]
            assert np.abs(frame.cart_ang - edge_truth("colname", k)).max() <= tol
        text = message_of(lambda: reader(EDGES / name, type_map=TYPES,
                                         columns="id type x y z q"))
        assert "columns=" in text


# -- YAML dump ---------------------------------------------------------------

def test_yaml_cut_inside_the_last_header_keeps_the_frame_before(tmp_path):
    """Cut after the last frame's natoms and boundary lines: the complete
    frame before it was skipped as '14 data rows where natoms gives 8', and
    the cut header was not counted."""
    raw = (EDGES / "thresh.yaml").read_bytes()
    cut = raw[:raw.index(b"box:", raw.rindex(b"---"))]
    traj = M.read_lammps_yaml_dump(write(tmp_path, "hdr.yaml", cut),
                                   type_map={1: "Si", 2: "O"})
    assert traj.timesteps.tolist() == [0, 10, 20, 50]
    assert sorted(traj.skipped) == [3, 4, 6]
    assert traj.skipped[6].startswith("truncated") and "header" in traj.skipped[6]
    assert np.abs(traj.frame(3).cart_ang - edge_truth("thresh", 5)).max() <= \
        G6 * np.abs(edge_truth("thresh", 5)).max()


def test_yaml_frame_without_its_dashes_leaves_the_frame_before_whole(tmp_path):
    raw = (DATA / "traj.yaml").read_bytes()
    third = raw.index(b"---", raw.index(b"---", raw.index(b"---") + 3) + 3)
    damaged = raw[:third] + raw[third:].replace(b"---", b"", 1)
    traj = M.read_lammps_yaml_dump(write(tmp_path, "nodash.yaml", damaged))
    assert traj.timesteps.tolist() == [100, 110]
    assert list(traj.skipped) == [2] and "'---'" in traj.skipped[2]
    assert not any("-" in r.split()[0] for r in traj.skipped.values())


def test_yaml_with_a_byte_order_mark_reads_every_frame(tmp_path):
    raw = (EDGES / "thresh.yaml").read_bytes()
    traj = M.read_lammps_yaml_dump(write(tmp_path, "bom.yaml", b"\xef\xbb\xbf" + raw),
                                   type_map={1: "Si", 2: "O"})
    assert traj.timesteps.tolist() == [0, 10, 20, 50, 60]
    assert sorted(traj.skipped) == [3, 4]


def test_yaml_general_triclinic_rows_outside_the_box_are_noted():
    """dump_modify triclinic/general yes on a yaml dump: LAMMPS writes the
    restricted box and general-frame rows; the trajectory notes now say how
    far atoms lie outside the stated box and why that can happen."""
    tmap = {1: "Si", 2: "Na", 3: "O"}
    traj = M.read_lammps_yaml_dump(EDGES / "general.yaml", type_map=tmap)
    outside = [n for n in traj.notes if "lie outside the box" in n]
    assert len(outside) == 1 and "Å" in outside[0]
    beyond = float(re.search(r"the farthest ([0-9.e+-]+) Å", outside[0]).group(1))
    assert beyond > 0.5
    assert any("triclinic/general" in n for n in traj.notes)
    ref = M.read_lammps_yaml_dump(DATA / "traj.yaml")
    assert not any("lie outside the box" in n for n in ref.notes)


# -- CFG -----------------------------------------------------------------------

def test_cfg_auxiliary_without_index_is_refused_naming_the_file(tmp_path):
    text = (DATA / "cfg" / "dump.100.cfg").read_text()
    path = tmp_path / "aux.cfg"
    path.write_text(text.replace("auxiliary[0] = id", "auxiliary = id"))
    message = message_of(lambda: md_readers.read_trajectory(path))
    assert "aux.cfg" in message and "auxiliary[k]" in message


def test_cfg_entry_count_is_held_to_the_rows(tmp_path):
    """entry_count 10^7 in an 877-byte file built 10^7 column names (1.9 GB)
    before any row was compared."""
    text = (DATA / "cfg" / "dump.100.cfg").read_text()
    path = tmp_path / "entry.cfg"
    path.write_text(re.sub(r"entry_count = \d+", "entry_count = 10000000", text))
    tracemalloc.start()
    try:
        message = message_of(lambda: M.read_cfg(path))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert message.startswith("entry.cfg")
    assert "entry_count gives 10000000" in message
    assert peak < 64 * 2 ** 20


def test_cfg_series_lists_cut_files_at_open(tmp_path):
    for cut, kept, steps in (("dump.120.cfg", 2, [100, 110]),
                             ("dump.100.cfg", 2, [110, 120])):
        folder = tmp_path / cut
        shutil.copytree(DATA / "cfg", folder)
        data = (folder / cut).read_bytes()
        (folder / cut).write_bytes(data[:len(data) * 3 // 4])
        traj = md_readers.read_trajectory(folder)
        assert len(traj) == kept and traj.timesteps.tolist() == steps
        position = 2 if cut == "dump.120.cfg" else 0
        assert list(traj.skipped) == [position]
        assert traj.skipped[position].startswith(cut)
        for k in range(len(traj)):
            traj.frame(k)


def test_cfg_two_series_in_one_folder_are_refused(tmp_path):
    folder = tmp_path / "mix"
    shutil.copytree(DATA / "cfg", folder)
    for path in sorted((DATA / "cfg").glob("*.cfg")):
        step = path.name.split(".")[1]
        shutil.copy(path, folder / f"run2_{step}.cfg")
    message = message_of(lambda: md_readers.read_trajectory(folder))
    assert "dump.*.cfg: 3 files" in message and "run*_*.cfg: 3 files" in message
    assert "wildcard pattern" in message
    with md_readers.read_trajectory(str(folder / "dump.*.cfg")) as traj:
        assert traj.timesteps.tolist() == [100, 110, 120]
    # a file whose auxiliary columns differ from the others' is skipped
    other = tmp_path / "cols"
    shutil.copytree(DATA / "cfg", other)
    text = (other / "dump.110.cfg").read_text()
    (other / "dump.110.cfg").write_text(text.replace("auxiliary[1] = q",
                                                     "auxiliary[1] = charge"))
    traj = md_readers.read_trajectory(other)
    assert traj.timesteps.tolist() == [100, 120] and list(traj.skipped) == [1]
    assert "auxiliary" in traj.skipped[1] and "charge" in traj.skipped[1]


def test_cfg_auxiliary_velocities_are_read_in_the_run_units():
    traj = M.read_cfg_series(sorted((EDGES / "cfg_vel").glob("*.cfg"),
                                    key=lambda p: int(p.name.split(".")[1])))
    assert traj.timesteps.tolist() == [0, 10]
    assert "stored as written" in traj.units_note
    for k in range(2):
        frame = traj.frame(k)
        truth = np.array(E["colname"]["v"][k])
        assert frame.atom_id.tolist() == E["colname"]["ids"]
        assert np.abs(frame.vel_ang_per_ps - truth).max() <= \
            G6 * np.abs(truth).max()
    real = M.read_cfg_series(sorted((EDGES / "cfg_vel").glob("*.cfg"),
                                    key=lambda p: int(p.name.split(".")[1])),
                             units="real")
    assert np.allclose(real.frame(1).vel_ang_per_ps,
                       traj.frame(1).vel_ang_per_ps * 1000.0, rtol=1e-12, atol=0)
    assert any("vx vy vz" in n for n in traj.notes if n.startswith("auxiliary"))


def test_cfg_box_that_overflows_is_refused(tmp_path):
    text = (DATA / "cfg" / "dump.100.cfg").read_text()
    path = tmp_path / "huge.cfg"
    path.write_text(text.replace("A = 1 Angstrom", "A = 1e300 Angstrom"))
    message = message_of(lambda: M.read_cfg(path))
    assert "huge.cfg" in message and "not finite" in message


def test_new_notes_and_messages_carry_no_verdict(tmp_path):
    tmap = {1: "Si", 2: "Na", 3: "O"}
    texts: list[str] = []
    for traj in (M.read_dcd(EDGES / "first_yes.dcd", topology=EDGES / "first_yes.data"),
                 M.read_dcd(EDGES / "plugin.dcd", topology=DATA / "topology.data"),
                 M.read_dcd(EDGES / "mda_nocell.dcd", topology=DATA / "topology.data",
                            box_from=METAL["H"][0]),
                 M.read_lammps_binary_dump(EDGES / "thresh.bin", type_map=tmap),
                 M.read_lammps_binary_dump(EDGES / "colname.bin", type_map=tmap,
                                           columns="id type x y z"),
                 M.read_lammps_yaml_dump(EDGES / "general.yaml", type_map=tmap),
                 M.read_cfg_series(sorted((EDGES / "cfg_vel").glob("*.cfg")))):
        texts += traj.describe() + list(traj.skipped.values())
    texts.append(message_of(lambda: M.read_dcd(EDGES / "mda_nocell.dcd",
                                               topology=DATA / "topology.data")))
    texts.append(message_of(lambda: M.read_lammps_yaml_dump(EDGES / "colname.yaml",
                                                            type_map=tmap)))
    assert [t for t in texts if VERDICT.search(t)] == []
