"""Binary MD formats: ASE .traj, HOOMD GSD and AMBER NetCDF.

The files in tests/data/md/binary were written by the programs themselves
(ASE 3.29, OVITO 3.16.1, the gsd package, MDAnalysis, the netCDF-C library;
versions in MANIFEST.json) by tests/data/md/binary/make_binary_files.py, and
each ``<file>.npz`` holds what the writer was given. What is pinned, and why:

* **Every file gives back its writer's input.** Positions (compared as
  fractional coordinates modulo 1, so the box origin's convention cannot hide
  a shift), box, origin, elements, ids, steps, times, velocities, charges and
  continuous positions, at the precision the file stores: 1e-12 Å for double
  precision, 1e-6 Å for single.
* **Independent readers agree.** scipy.io.netcdf_file (a FACET dependency)
  reads the classic and 64-bit offset files variable by variable as the
  NetCDF parser does; the gsd package's own reading of the GSD files is
  stored beside them.
* **Unit conversions are the writers' own definitions:** ASE's unit of
  velocity against the value ASE 3.29 reports, AMBER's scale_factor, LAMMPS's
  time holding the step with the step length as its scale_factor
  (dump_netcdf.cpp, float or double), nm to Å (a NetCDF units attribute, and
  units='nano' for a GSD file mdtraj wrote in nm).
* **Type names are labels:** without particles/mass, a GSD file whose names
  include one that is not a symbol needs a map for every name; with it, a
  name that is not a symbol takes the element its mass alone names.
* **Damaged bytes give named ValueErrors and bounded memory:** a damaged
  frame count, a shape too large for an integer, JSON nested past the
  recursion limit, a text id variable, a box of length 0.
* **Writer defects are handled as measured:** OVITO's GSD box encoding and its
  coordinate-wise wrapping are undone; OVITO's NetCDF cell angles, which
  describe the cell with its xy and yz tilts exchanged (recorded in
  MANIFEST.json), make a tilted OVITO NetCDF file a refusal.
* **Nothing is dropped silently:** a file cut while it was written loses its
  last frame to ``skipped`` with the reason, a frame with other atoms is
  skipped, a damaged later frame raises FrameError naming the file.
* **Every refusal names the file and says what to do,** and no note or
  message carries a verdict word.
"""
from __future__ import annotations

import ast
import gzip
import inspect
import json
import pickle
import re
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from facet.core import md_formats_base, md_formats_binary as mb, md_readers
from facet.core.md_model import NO_TIMESTEP, FrameError
from facet.core.readers import UnsupportedFormat

DATA = Path(__file__).resolve().parent / "data" / "md" / "binary"
TEXT_DATA = DATA.parent
MANIFEST = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))
TYPES = {1: "Si", 2: "Na", 3: "O"}
STEPS = np.array(MANIFEST["model"]["steps"])
SYMBOLS = MANIFEST["model"]["symbols"]
F = len(STEPS)
N = len(SYMBOLS)
DOUBLE_ANG = 1e-12
SINGLE_ANG = 1e-6

VERDICTS = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|correct|incorrect|wrong|reliable|"
    r"unreliable|trustworthy|untrustworthy|unusable|should|proves|confirms)\b",
    re.IGNORECASE)


def truth(name: str) -> dict:
    with np.load(DATA / f"{name}.npz") as data:
        return {k: data[k] for k in data.files}


def copy(tmp_path: Path, name: str, new: str | None = None) -> Path:
    target = tmp_path / (new or name)
    shutil.copy(DATA / name, target)
    return target


def frac_error_ang(frame, tr: dict, k: int, rows) -> float:
    """Largest distance between a frame's atoms and the truth's, measured as
    fractional coordinates modulo 1 in the truth box."""
    box = tr["box"][k]
    want = (tr["pos"][k][rows] - tr["origin"][k]) @ np.linalg.inv(box)
    d = np.asarray(frame.frac) - want
    d -= np.round(d)
    return float(np.linalg.norm(d @ box, axis=1).max())


def rows_of(frame, tr: dict, by_id: bool) -> np.ndarray:
    if not by_id:
        return np.arange(frame.n_atoms)
    lookup = {int(v): i for i, v in enumerate(tr["ids"])}
    return np.array([lookup[int(v)] for v in frame.atom_id])


def assert_matches(traj, name: str, *, by_id: bool, pos_tol: float,
                   box_tol: float = DOUBLE_ANG, vel_rel: float | None = None,
                   charges: bool = False, steps: bool = False,
                   origin: bool = True) -> None:
    tr = truth(name)
    assert traj.n_frames == tr["pos"].shape[0]
    assert traj.n_atoms == N
    for k in range(traj.n_frames):
        frame = traj.frame(k)
        rows = rows_of(frame, tr, by_id)
        assert frame.elements.tolist() == tr["elements"][rows].tolist()
        if by_id:
            assert frame.atom_id.tolist() == sorted(tr["ids"].tolist())
        assert np.abs(frame.box_ang - tr["box"][k]).max() <= box_tol
        if origin:
            assert np.abs(frame.origin_ang - tr["origin"][k]).max() <= box_tol
        assert frac_error_ang(frame, tr, k, rows) <= pos_tol
        if vel_rel is not None:
            want = tr["vel"][k][rows]
            assert frame.vel_ang_per_ps is not None
            assert np.abs(frame.vel_ang_per_ps - want).max() \
                <= vel_rel * np.abs(want).max()
        if charges:
            assert np.array_equal(frame.charge_e, tr["charge"][rows])
    if steps:
        assert traj.timesteps.tolist() == tr["steps"].tolist()


def messages(traj) -> list[str]:
    return list(traj.notes) + [traj.units_note] + list(traj.skipped.values())


def assert_no_verdicts(texts) -> None:
    for text in texts:
        assert VERDICTS.search(str(text)) is None, text


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------

def test_formats_keep_the_contract():
    names = [spec.name for spec in mb.FORMATS]
    assert names == ["ase-traj", "gsd", "amber-netcdf"]
    assert "md_formats_binary" in md_formats_base.FORMAT_MODULES
    for spec in mb.FORMATS:
        assert spec.binary
        assert spec.options <= md_formats_base.KNOWN_OPTIONS
        keywords = {p.name for p in inspect.signature(spec.read).parameters.values()
                    if p.kind is inspect.Parameter.KEYWORD_ONLY}
        assert keywords == set(spec.options), spec.name


def test_module_imports_numpy_scipy_gemmi_and_stdlib_only():
    tree = ast.parse(Path(mb.__file__).read_text(encoding="utf-8"))
    allowed = {"numpy", "scipy", "gemmi", "importlib", "json", "math", "struct",
               "collections", "dataclasses", "pathlib", "__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            assert node.module.split(".")[0] in allowed, node.module


SNIFFED = {".traj": "ase-traj", ".gsd": "gsd", ".nc": "amber-netcdf",
           ".ncrst": "amber-netcdf"}


@pytest.mark.parametrize("name", sorted(MANIFEST["files"]))
def test_each_file_is_recognised_by_its_format_alone(name):
    path = DATA / name
    head = path.read_bytes()[:md_readers.SNIFF_BYTES]
    claimed = [spec.name for spec in mb.FORMATS if spec.sniff(head, path)]
    assert claimed == [SNIFFED[path.suffix]]
    # through md_readers: no built-in text format and no earlier format
    # module claims the file first
    assert md_readers.sniff_md(path) == SNIFFED[path.suffix]


@pytest.mark.parametrize("name,options", [
    ("ase_md.traj", {}), ("hoomd_small.gsd", {"timestep_fs": 1.0}),
    ("ovito_xz.nc", {"type_map": TYPES})])
def test_public_entry_points_route_the_files(name, options):
    from facet.core import readers

    direct = {".traj": mb.read_ase_traj, ".gsd": mb.read_gsd,
              ".nc": mb.read_amber_netcdf}[Path(name).suffix]
    through = md_readers.read_trajectory(DATA / name, **options)
    own = direct(DATA / name, **options)
    assert through.file_format == own.file_format == SNIFFED[Path(name).suffix]
    assert np.array_equal(through.frame(2).cart_ang, own.frame(2).cart_ang)
    assert through.notes == own.notes
    with pytest.raises(readers.MDModelFile) as caught:
        readers.read(DATA / name)
    assert caught.value.file_format == SNIFFED[Path(name).suffix]
    spec = next(s for s in mb.FORMATS if s.name == own.file_format)
    if "type_map" not in spec.options:
        with pytest.raises(ValueError, match="does not use type_map"):
            md_readers.read_trajectory(DATA / name, type_map=TYPES)


def test_text_md_files_are_not_claimed():
    for path in sorted(TEXT_DATA.iterdir()):
        if not path.is_file():
            continue
        head = path.read_bytes()[:md_readers.SNIFF_BYTES]
        if head[:2] == b"\x1f\x8b":
            head = gzip.decompress(path.read_bytes())[:md_readers.SNIFF_BYTES]
        assert not any(spec.sniff(head, path) for spec in mb.FORMATS), path.name


# ---------------------------------------------------------------------------
# ASE trajectories
# ---------------------------------------------------------------------------

def test_ase_velocity_unit_is_ases_own():
    ase_units = MANIFEST["records"]["ase_units"]
    assert ase_units["codata"] == "2014"
    # one ASE unit of velocity is 1 Å per ASE time unit = second * 1e-12 Å/ps
    assert mb.ASE_VELOCITY_TO_ANG_PER_PS == pytest.approx(
        ase_units["second"] * 1e-12, rel=1e-15)
    assert mb.ASE_VELOCITY_TO_ANG_PER_PS == pytest.approx(
        ase_units["fs"] * 1000.0, rel=1e-15)
    # scipy's CODATA 2022 values give the same unit to 1e-8
    from scipy import constants
    assert mb.ASE_VELOCITY_TO_ANG_PER_PS == pytest.approx(
        1e-2 * np.sqrt(constants.e / constants.atomic_mass), rel=1e-8)


def test_ase_md_trajectory_without_masses():
    traj = mb.read_ase_traj(DATA / "ase_md.traj")
    tr = truth("ase_md.traj")
    assert traj.file_format == "ase-traj"
    assert traj.type_map_source == "file symbols"
    assert_matches(traj, "ase_md.traj", by_id=False, pos_tol=DOUBLE_ANG,
                   charges=True)
    assert (traj.timesteps == NO_TIMESTEP).all() and traj.times_ps is None
    weights = dict(md_readers.element_weights_amu())
    gemmi_mass = np.array([weights[s] for s in SYMBOLS])
    for k in range(F):
        frame = traj.frame(k)
        # ASE divided by its own masses; FACET by gemmi's: the ratio is exact
        restored = frame.vel_ang_per_ps * (gemmi_mass / tr["ase_masses"])[:, None]
        assert np.abs(restored - tr["vel"][k]).max() <= 1e-12 * np.abs(tr["vel"][k]).max()
    assert traj.charges_e == {"Na": 0.6, "O": -1.5, "Si": 2.4}
    text = " ".join(traj.notes)
    assert "gemmi's standard atomic weights" in text
    assert "calculator (energy, forces)" in text
    assert "temperature_K" in text
    assert traj.frame(0).unwrapped_cart_ang is None
    assert_no_verdicts(messages(traj))


def test_ase_masses_give_exact_velocities_and_constraints_are_noted():
    traj = mb.read_ase_traj(DATA / "ase_masses.traj")
    assert_matches(traj, "ase_masses.traj", by_id=False, pos_tol=DOUBLE_ANG,
                   vel_rel=1e-14)
    assert "divided by the masses the file stores" in " ".join(traj.notes)
    assert "constraints" in " ".join(traj.notes)


def test_ase_frame_with_other_atoms_is_skipped():
    traj = mb.read_ase_traj(DATA / "ase_changed.traj")
    assert traj.n_frames == 2
    assert traj.file_positions.tolist() == [0, 2]
    assert set(traj.skipped) == {1}
    assert "atomic numbers differ" in traj.skipped[1]
    assert "Z 14 -> 32" in traj.skipped[1]
    assert_matches(traj, "ase_changed.traj", by_id=False, pos_tol=DOUBLE_ANG)


def test_ovito_ase_traj_origin_and_steps():
    traj = mb.read_ase_traj(DATA / "ovito_small.traj")
    assert_matches(traj, "ovito_small.traj", by_id=False, pos_tol=DOUBLE_ANG,
                   steps=True)
    assert traj.times_ps is None
    timed = mb.read_ase_traj(DATA / "ovito_small.traj", timestep_fs=2.0)
    assert np.allclose(timed.times_ps, STEPS * 2.0e-3, rtol=0, atol=1e-15)
    assert "info['cell_origin']" in " ".join(traj.notes)


def test_ase_timestep_fs_needs_steps():
    with pytest.raises(ValueError, match="ase_md.traj: timestep_fs .* no step"):
        mb.read_ase_traj(DATA / "ase_md.traj", timestep_fs=1.0)
    with pytest.raises(ValueError, match="ovito_small.traj: timestep_fs -1.0: "
                       "a positive number of femtoseconds"):
        mb.read_ase_traj(DATA / "ovito_small.traj", timestep_fs=-1.0)


def _ulm_item0(path: Path) -> tuple[int, dict]:
    raw = path.read_bytes()
    _, _, pos0 = struct.unpack_from("<qqq", raw, 24)
    (offset,) = struct.unpack_from("<q", raw, pos0)
    (length,) = struct.unpack_from("<q", raw, offset)
    return offset + 8, json.loads(raw[offset + 8:offset + 8 + length])


def test_ase_cut_file_loses_its_last_frame(tmp_path):
    path = copy(tmp_path, "ase_md.traj")
    raw = path.read_bytes()
    path.write_bytes(raw[:-60])                 # inside the last JSON record
    traj = mb.read_ase_traj(path)
    assert traj.n_frames == 2
    assert set(traj.skipped) == {2}
    assert traj.skipped[2].startswith("truncated")
    _, pos0 = struct.unpack_from("<qq", raw, 32)
    path.write_bytes(raw[:pos0 - 8])            # before the offset table
    with pytest.raises(ValueError, match="ase_md.traj: the table of frame "
                       "offsets .* cut"):
        mb.read_ase_traj(path)
    path.write_bytes(raw[:30])
    with pytest.raises(ValueError, match="ase_md.traj: the file ends inside"):
        mb.read_ase_traj(path)


def test_ase_refusals_name_the_file_and_say_what_to_do(tmp_path):
    path = copy(tmp_path, "ase_md.traj")
    raw = bytearray(path.read_bytes())
    raw[8:24] = b"GPAW".ljust(16)
    path.write_bytes(bytes(raw))
    with pytest.raises(UnsupportedFormat, match="ase_md.traj: .*tag 'GPAW'.*"
                       "ase.io.write"):
        mb.read_ase_traj(path)
    raw = bytearray(copy(tmp_path, "ase_md.traj").read_bytes())
    struct.pack_into("<q", raw, 24, 9)
    path.write_bytes(bytes(raw))
    with pytest.raises(UnsupportedFormat, match="ULM version 9.*extxyz"):
        mb.read_ase_traj(path)
    # atomic number 0, ASE's placeholder 'X'
    raw = bytearray(copy(tmp_path, "ase_md.traj").read_bytes())
    _, item = _ulm_item0(path)
    offset = item["numbers."]["ndarray"][2]
    struct.pack_into("<q", raw, offset, 0)
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="ase_md.traj.*atomic number 0"):
        mb.read_ase_traj(path)
    # a zero cell (a molecule): the same JSON length, zeros for the vectors
    raw = copy(tmp_path, "ase_md.traj").read_bytes()
    start, item = _ulm_item0(path)
    old = b'"cell": [[7.1, 0.0, 0.0], [1.3, 6.8, 0.0], [-0.9, 0.7, 7.4]]'
    new = b'"cell": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.0, 0.0, 0.0]]'
    assert raw.count(old) == 1 and len(old) == len(new)
    path.write_bytes(raw.replace(old, new))
    with pytest.raises(FrameError, match="ase_md.traj, frame 0.*cell is zero"):
        mb.read_ase_traj(path)


# ---------------------------------------------------------------------------
# GSD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["hoomd_small.gsd", "gsd_v1_synthetic.gsd"])
def test_gsd_written_by_the_gsd_library(name):
    traj = mb.read_gsd(DATA / name)
    tr = truth(name)
    assert traj.file_format == "gsd"
    # single-precision positions: the gsd package's own reading, exactly ...
    for k in range(F):
        frame = traj.frame(k)
        box, origin = mb.box_from_hoomd(tr["oracle_box"][k])
        oracle = (tr["oracle_pos"][k].astype(float) - origin) @ np.linalg.inv(box)
        d = frame.frac - oracle
        d -= np.round(d)
        assert np.abs(d @ box).max() <= DOUBLE_ANG
        assert np.array_equal(frame.vel_ang_per_ps,
                              tr["oracle_vel"][k].astype(float))
        unwrapped = tr["oracle_pos"][k].astype(float) + \
            tr["oracle_img"][k] @ box
        assert np.abs(frame.unwrapped_cart_ang - unwrapped).max() <= DOUBLE_ANG
        assert np.abs(frame.unwrapped_cart_ang - tr["unwrapped"][k]).max() \
            <= SINGLE_ANG
    # ... and the writer's double-precision input to float32 rounding
    assert_matches(traj, name, by_id=False, pos_tol=SINGLE_ANG,
                   box_tol=SINGLE_ANG, vel_rel=1e-6, steps=True)
    frame = traj.frame(0)
    assert np.allclose(frame.charge_e, tr["charge"], atol=1e-6)
    text = " ".join(traj.notes)
    assert "continuous positions from particles/image" in text
    assert "log/md/compute/ThermodynamicQuantities/potential_energy" in text
    assert "particles/mass of frame 0 is compared" in text
    # the gsd library left frame 0's all-zero image flags out; the note says
    # so without naming frame 0 twice
    assert "frame 0 holds no particles/image: those frames take the HOOMD " \
        "schema's default, zero" in text
    assert "nor frame 0" not in text
    assert traj.times_ps is None
    assert_no_verdicts(messages(traj))


def test_gsd_inherits_frame_0_chunks():
    # the gsd library writes types, typeid and charges in frame 0 only
    index = mb._gsd_index(DATA / "hoomd_small.gsd")
    assert "particles/types" in index.frames[0]
    assert "particles/types" not in index.frames[2]
    assert "particles/typeid" not in index.frames[2]
    traj = mb.read_gsd(DATA / "hoomd_small.gsd")
    assert traj.frame(2).elements.tolist() == SYMBOLS


def test_gsd_options():
    timed = mb.read_gsd(DATA / "hoomd_small.gsd", timestep_fs=1.0)
    assert np.allclose(timed.times_ps, STEPS * 1e-3, rtol=0, atol=1e-15)
    real = mb.read_gsd(DATA / "hoomd_small.gsd", units="real")
    plain = mb.read_gsd(DATA / "hoomd_small.gsd")
    assert np.allclose(real.frame(1).vel_ang_per_ps,
                       1000.0 * plain.frame(1).vel_ang_per_ps, rtol=1e-15)
    assert "LAMMPS units real" in real.units_note
    with pytest.raises(ValueError, match="hoomd_small.gsd: .*lj.*not Å"):
        mb.read_gsd(DATA / "hoomd_small.gsd", units="lj")
    with pytest.raises(ValueError, match="hoomd_small.gsd: type map entry"):
        mb.read_gsd(DATA / "hoomd_small.gsd", type_map={"Si": 14})


def test_gsd_type_names_are_labels(tmp_path):
    path = copy(tmp_path, "hoomd_small.gsd")
    raw = path.read_bytes()
    old = b"Si\0Na\0O\0\0"
    assert raw.count(old) == 1
    path.write_bytes(raw.replace(old, b"A\0\0B\0\0C\0\0"))
    # 'A' names no element, and its mass (28.0855 amu) names silicon alone;
    # 'B' and 'C' are symbols that the file's own masses (22.99 and 15.999
    # amu) put nearer sodium and oxygen, so they are refused
    with pytest.raises(ValueError) as caught:
        mb.read_gsd(path)
    text = str(caught.value)
    assert text.startswith("hoomd_small.gsd: ")
    assert "'B' (mass 22.9898 amu in the file)" in text
    assert "'C' (mass 15.9994 amu in the file)" in text
    assert "'A'" not in text
    assert "<element>" in text
    traj = mb.read_gsd(path, type_map={"A": "Si", "B": "Na", "C": "O"})
    assert traj.frame(0).elements.tolist() == SYMBOLS
    assert traj.type_map_source == "user"
    # 'A' from its mass, 'B' and 'C' from the map
    traj = mb.read_gsd(path, type_map={"B": "Na", "C": "O"})
    assert traj.frame(0).elements.tolist() == SYMBOLS
    assert traj.type_map_source == "user + data-file masses"
    assert traj.type_map == {"A": "Si", "B": "Na", "C": "O"}
    assert "type name 'A' is not an element symbol, and frame 0's " \
        "particles/mass gives it a mass 28.0855 amu read as Si" \
        in " ".join(traj.notes)


def test_ovito_gsd_box_is_read_as_ovito_writes_it():
    traj = mb.read_gsd(DATA / "ovito_face.gsd")
    assert_matches(traj, "ovito_face.gsd", by_id=False, pos_tol=DOUBLE_ANG,
                   vel_rel=1e-15, charges=True, steps=True)
    text = " ".join(traj.notes)
    assert "box read as OVITO's exporter writes it" in text
    assert "with OVITO's reading none do" in text
    assert "particles/image holds only the shifts" in text
    assert traj.frame(0).unwrapped_cart_ang is None
    # the schema's reading of the same numbers is another box
    index = mb._gsd_index(DATA / "ovito_face.gsd")
    with open(DATA / "ovito_face.gsd", "rb") as handle:
        box6 = mb._gsd_values(handle, index.frames[0]["configuration/box"],
                              "box").reshape(6)
    schema_rows, _ = mb.box_from_hoomd(box6)
    assert np.abs(schema_rows - truth("ovito_face.gsd")["box"][0]).max() > 0.1


def test_ovito_gsd_whose_positions_do_not_decide():
    traj = mb.read_gsd(DATA / "ovito_ambiguous.gsd")
    assert_matches(traj, "ovito_ambiguous.gsd", by_id=False, pos_tol=DOUBLE_ANG)
    assert "positions do not decide" in " ".join(traj.notes)


def test_ovito_coordinate_wise_wrapping_is_undone():
    # frame 2 holds atom 17 just below the c face; OVITO moved it by
    # (0, 0, c_z) and wrote image flag -1 along c
    rows = truth("ovito_face.gsd")["box"][2]
    written = np.array([[1.0, 2.0, 3.0]])
    restored = mb._undo_ovito_shifts(written, np.array([[0, 0, -1]]), rows)
    assert np.array_equal(restored, [[1.0, 2.0, 3.0 - rows[2, 2]]])
    traj = mb.read_gsd(DATA / "ovito_face.gsd")
    tr = truth("ovito_face.gsd")
    assert frac_error_ang(traj.frame(2), tr, 2, np.arange(N)) <= DOUBLE_ANG


def test_box_from_hoomd_is_the_schema_unwrapping_formula():
    lx, ly, lz, xy, xz, yz = 10.0, 11.0, 12.0, 0.2, -0.1, 0.3
    rows, origin = mb.box_from_hoomd([lx, ly, lz, xy, xz, yz])
    image = np.array([2, -1, 3])
    x, y, z = 0.5, -0.25, 1.5
    unwrapped = np.array([x + image[0] * lx + xy * image[1] * ly + xz * image[2] * lz,
                          y + image[1] * ly + yz * image[2] * lz,
                          z + image[2] * lz])
    assert np.allclose(np.array([x, y, z]) + image @ rows, unwrapped,
                       rtol=0, atol=1e-13)
    assert np.allclose(origin, -rows.sum(axis=0) / 2)


def test_gsd_cut_file_loses_its_last_frame(tmp_path):
    path = copy(tmp_path, "hoomd_small.gsd")
    raw = path.read_bytes()
    path.write_bytes(raw[:-10])
    traj = mb.read_gsd(path)
    assert traj.n_frames == 2 and set(traj.skipped) == {2}
    assert "truncated" in traj.skipped[2]
    path.write_bytes(raw[:200])
    with pytest.raises(ValueError, match="hoomd_small.gsd: the file ends inside "
                       "its 256-byte GSD header"):
        mb.read_gsd(path)
    path.write_bytes(raw[:300])
    with pytest.raises(ValueError, match="hoomd_small.gsd: the GSD index runs "
                       "past the end"):
        mb.read_gsd(path)


def test_gsd_refusals(tmp_path):
    path = copy(tmp_path, "hoomd_small.gsd")
    raw = bytearray(path.read_bytes())
    raw[112:176] = b"other".ljust(64, b"\0")
    path.write_bytes(bytes(raw))
    with pytest.raises(UnsupportedFormat, match="schema 'other'.*'hoomd'"):
        mb.read_gsd(path)
    raw = bytearray(copy(tmp_path, "hoomd_small.gsd").read_bytes())
    raw[0:8] = raw[0:8][::-1]
    path.write_bytes(bytes(raw))
    assert mb.FORMATS[1].sniff(bytes(raw[:64]), path)
    with pytest.raises(UnsupportedFormat, match="big-endian GSD"):
        mb.read_gsd(path)
    raw = bytearray(copy(tmp_path, "hoomd_small.gsd").read_bytes())
    struct.pack_into("<I", raw, 44, 3 << 16)
    path.write_bytes(bytes(raw))
    with pytest.raises(UnsupportedFormat, match="version 3.0; FACET reads "
                       "versions 1.x and 2.x"):
        mb.read_gsd(path)


# ---------------------------------------------------------------------------
# NetCDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["amber_style.nc", "mda_small.nc", "ase_small.nc"])
def test_netcdf_parser_agrees_with_scipy(name):
    from scipy.io import netcdf_file

    header = mb.netcdf_header(DATA / name)
    with netcdf_file(DATA / name, "r", mmap=False) as oracle, \
            open(DATA / name, "rb") as handle:
        assert set(header.variables) == set(oracle.variables)
        assert header.numrecs == oracle._recs
        for key, var in oracle.variables.items():
            mine = header.variables[key]
            assert mine.dimensions == var.dimensions
            if mine.is_record:
                for k in range(header.numrecs):
                    assert np.array_equal(mb._nc_values(handle, header, mine, k),
                                          var.data[k])
            else:
                assert np.array_equal(mb._nc_values(handle, header, mine, 0),
                                      var.data)
            for attr, value in var._attributes.items():
                if isinstance(value, bytes):
                    assert mine.attributes[attr] == value.decode()
                else:
                    assert np.array_equal(np.ravel(mine.attributes[attr]),
                                          np.ravel(value))


def test_lammps_layout_steps_times_ids():
    traj = mb.read_amber_netcdf(DATA / "lammps_style.nc", type_map=TYPES)
    tr = truth("lammps_style.nc")
    assert_matches(traj, "lammps_style.nc", by_id=True, pos_tol=DOUBLE_ANG,
                   vel_rel=1e-15, steps=True)
    assert np.array_equal(traj.times_ps, tr["time_ps"])
    assert traj.type_map == TYPES and traj.type_map_source == "user"
    # LAMMPS writes the step in a real-typed time (dump_netcdf.cpp: time =
    # update->ntimestep), double here (dump_modify double yes)
    header = mb.netcdf_header(DATA / "lammps_style.nc")
    assert header.variables["time"].dtype == np.dtype(">f8")
    assert header.attributes["program"] == "LAMMPS"
    assert "steps from the time variable: LAMMPS's dump netcdf writes the " \
        "step number there" in " ".join(traj.notes)
    assert "float32" not in " ".join(traj.notes)
    with pytest.raises(ValueError, match="lammps_style.nc: the file gives each "
                       "frame's time"):
        mb.read_amber_netcdf(DATA / "lammps_style.nc", type_map=TYPES,
                             timestep_fs=1.0)


def test_numeric_types_without_a_map_are_refused():
    with pytest.raises(ValueError) as caught:
        mb.read_amber_netcdf(DATA / "lammps_style.nc")
    text = str(caught.value)
    assert text.startswith("lammps_style.nc: ")
    assert "type_map={1: '<element>'" in text


def test_amber_scale_factor_and_single_precision():
    elements = truth("amber_style.nc")["elements"].tolist()
    traj = mb.read_amber_netcdf(DATA / "amber_style.nc", topology=elements)
    tr = truth("amber_style.nc")
    assert_matches(traj, "amber_style.nc", by_id=False, pos_tol=SINGLE_ANG,
                   vel_rel=1e-6)
    assert np.array_equal(traj.times_ps, tr["time_ps"])
    assert "scale_factor 20.455" in " ".join(traj.notes)
    assert "single precision" in " ".join(traj.notes)


def test_mdanalysis_file_needs_a_topology(tmp_path):
    with pytest.raises(ValueError) as caught:
        mb.read_amber_netcdf(DATA / "mda_small.nc")
    assert str(caught.value).startswith("mda_small.nc: the file names no element")
    assert "topology=" in str(caught.value)
    # a topology file of the same atoms, read by FACET's own reader
    tr = truth("mda_small.nc")
    lines = [f"{N}", 'Lattice="' + " ".join(str(x) for x in tr["box"][0].ravel())
             + '" Properties=species:S:1:pos:R:3']
    lines += [f"{s} {x} {y} {z}" for s, (x, y, z) in zip(SYMBOLS, tr["pos"][0])]
    topology = tmp_path / "model.extxyz"
    topology.write_text("\n".join(lines) + "\n")
    traj = mb.read_amber_netcdf(DATA / "mda_small.nc", topology=topology)
    assert_matches(traj, "mda_small.nc", by_id=False, pos_tol=SINGLE_ANG,
                   box_tol=SINGLE_ANG, vel_rel=1e-6)
    assert np.allclose(traj.times_ps, tr["time_ps"], rtol=0, atol=1e-8)
    assert "model.extxyz" in " ".join(traj.notes)
    with pytest.raises(ValueError, match="topology lists 3 elements for 8"):
        mb.read_amber_netcdf(DATA / "mda_small.nc", topology=["Si", "O", "O"])
    with pytest.raises(ValueError, match="names its atoms .*topology is for"):
        mb.read_amber_netcdf(DATA / "lammps_style.nc", topology=SYMBOLS)


def test_ase_netcdf_atomic_numbers_and_velocities():
    traj = mb.read_amber_netcdf(DATA / "ase_small.nc")
    tr = truth("ase_small.nc")
    assert traj.type_map_source == "file symbols"
    assert_matches(traj, "ase_small.nc", by_id=False, pos_tol=SINGLE_ANG,
                   vel_rel=1e-6)
    text = " ".join(traj.notes)
    assert "Angstrom/Femtosecond" in text
    assert "ASE's unit of velocity" in text
    del tr


def test_ovito_netcdf_xz_cell_is_read():
    traj = mb.read_amber_netcdf(DATA / "ovito_xz.nc", type_map=TYPES)
    assert_matches(traj, "ovito_xz.nc", by_id=True, pos_tol=DOUBLE_ANG,
                   vel_rel=1e-15, charges=True, steps=True)
    assert traj.times_ps is None
    assert "animation frame number" in " ".join(traj.notes)


def test_ovito_netcdf_angles_describe_the_exchanged_cell():
    """The measurement that makes a tilted OVITO NetCDF file a refusal."""
    def angles(rows):
        a, b, c = rows
        cos = [np.dot(b, c) / np.linalg.norm(b) / np.linalg.norm(c),
               np.dot(a, c) / np.linalg.norm(a) / np.linalg.norm(c),
               np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)]
        return np.degrees(np.arccos(cos))

    for label, record in MANIFEST["records"]["ovito_netcdf_angles"].items():
        a, b, c = np.array(record["rows"])
        exchanged = np.array([a, [c[1], b[1], 0.0], [c[0], b[0], c[2]]])
        assert np.allclose(record["written_angles"], angles(exchanged),
                           rtol=0, atol=1e-9), label
        assert np.allclose(record["written_lengths"],
                           np.linalg.norm(record["rows"], axis=1)), label
        own = np.allclose(record["written_angles"], record["cell_angles"],
                          rtol=0, atol=1e-9)
        assert own == (label in ("orthogonal", "xz")), label
    with pytest.raises(UnsupportedFormat) as caught:
        mb.read_amber_netcdf(DATA / "ovito_small.nc", type_map=TYPES)
    text = str(caught.value)
    assert text.startswith("ovito_small.nc: written by OVITO")
    assert "LAMMPS dump or an extended XYZ" in text


def test_a_tilted_ovito_netcdf_is_refused_for_its_cell_before_its_types():
    """Integration re-run of the recognition corpus (2026-10-07): the bare
    read of OVITO's tilted NetCDF asked for type_map=, and the read with it
    then refused the cell, so the user went round twice for a file no
    option reads. The cell refusal now comes first; an untilted file still
    asks for the type map."""
    with pytest.raises(UnsupportedFormat) as caught:
        mb.read_amber_netcdf(DATA / "ovito_small.nc")
    assert str(caught.value).startswith("ovito_small.nc: written by OVITO")
    with pytest.raises(ValueError, match=r"^ovito_xz\.nc: the atoms carry "
                                         r"numeric types"):
        mb.read_amber_netcdf(DATA / "ovito_xz.nc")


def _scipy_netcdf(path: Path, *, version=1, conventions="AMBER", units=None,
                  frame_unlimited=True, labels=None, numbers=None,
                  velocity_units=None, restart=False, restart_time=None,
                  char_ids=False, comment=None):
    """A small AMBER-style NetCDF file written by scipy.io.netcdf_file.

    ``restart_time``: 'scalar' (the AMBER convention's restart time),
    'length 1' (a dimension time of length 1, as ParmEd and mdtraj write)
    or 'per atom' (a variable no frame time can be read from)."""
    from scipy.io import netcdf_file

    tr = truth("lammps_style.nc")
    with netcdf_file(path, "w", version=version) as f:
        if conventions is not None:
            f.Conventions = conventions
        f.program = "scipy.io.netcdf_file test"
        if comment is not None:
            f.comment = comment
        if not restart:
            f.createDimension("frame", None if frame_unlimited else F)
        f.createDimension("atom", N)
        f.createDimension("spatial", 3)
        f.createDimension("cell_spatial", 3)
        f.createDimension("cell_angular", 3)
        f.createDimension("label", 2)
        lead = () if restart else ("frame",)
        coords = f.createVariable("coordinates", "d", lead + ("atom", "spatial"))
        if units is not None:
            coords.units = units
        lengths = f.createVariable("cell_lengths", "d", lead + ("cell_spatial",))
        angles = f.createVariable("cell_angles", "d", lead + ("cell_angular",))
        origin = f.createVariable("cell_origin", "d", lead + ("cell_spatial",))
        factor = {None: 1.0, "nanometer": 0.1, "angstrom": 1.0}[units]
        cells = []
        for k in range(F):
            box = tr["box"][k]
            norms = np.linalg.norm(box, axis=1)
            cos = [box[1] @ box[2] / norms[1] / norms[2],
                   box[0] @ box[2] / norms[0] / norms[2],
                   box[0] @ box[1] / norms[0] / norms[1]]
            cells.append((norms, np.degrees(np.arccos(cos))))
        if velocity_units is not None:
            vel = f.createVariable("velocities", "d", lead + ("atom", "spatial"))
            vel.units = velocity_units
        if labels is not None:
            element = f.createVariable("element", "c", ("atom", "label"))
            element[:] = np.array([list(s.ljust(2, "\0").encode()) for s in labels],
                                  dtype="u1").view("S1")
        if numbers is not None:
            z = f.createVariable("Z", "i", ("atom",))
            z[:] = numbers
        if char_ids:
            ids = f.createVariable("id", "c", ("atom",))
            ids[:] = np.array([b"x"] * N, dtype="S1")
        if restart_time == "scalar":
            f.createVariable("time", "d", ())[()] = 2.5
        elif restart_time == "length 1":
            f.createDimension("time", 1)
            f.createVariable("time", "d", ("time",))[:] = [2.5]
        elif restart_time == "per atom":
            f.createVariable("time", "d", ("atom",))[:] = np.arange(N)
        if restart:
            coords[:] = tr["pos"][0] * factor
            lengths[:], angles[:] = cells[0]
            origin[:] = tr["origin"][0]
        else:
            for k in range(F):
                coords[k] = tr["pos"][k] * factor
                lengths[k], angles[k] = cells[k]
                origin[k] = tr["origin"][k]
                if velocity_units is not None:
                    vel[k] = tr["vel"][k] * 1e-3
    return tr


def test_netcdf_variants_written_by_scipy(tmp_path):
    numbers = {"Si": 14, "Na": 11, "O": 8}
    # element names in a text variable, nm lengths, a fixed frame dimension
    path = tmp_path / "labels.nc"
    tr = _scipy_netcdf(path, labels=SYMBOLS, units="nanometer",
                       frame_unlimited=False,
                       velocity_units="angstrom/femtosecond")
    traj = mb.read_amber_netcdf(path)
    assert traj.n_frames == F
    for k in range(F):
        frame = traj.frame(k)
        assert frame.elements.tolist() == SYMBOLS
        tr_k = {"pos": tr["pos"], "origin": tr["origin"], "box": tr["box"]}
        assert frac_error_ang(frame, tr_k, k, np.arange(N)) <= DOUBLE_ANG
        assert np.allclose(frame.vel_ang_per_ps, tr["vel"][k], rtol=1e-14)
    text = " ".join(traj.notes)
    assert "converted from nanometer to Å" in text
    assert "converted from angstrom/femtosecond to Å/ps" in text
    # atomic numbers in Z, CDF-2
    path = tmp_path / "z.nc"
    _scipy_netcdf(path, version=2, numbers=[numbers[s] for s in SYMBOLS])
    traj = mb.read_amber_netcdf(path)
    assert traj.frame(1).elements.tolist() == SYMBOLS
    assert traj.type_map_source == "file symbols"
    # a restart: no frame dimension
    path = tmp_path / "restart.ncrst"
    _scipy_netcdf(path, restart=True, labels=SYMBOLS)
    traj = mb.read_amber_netcdf(path)
    assert traj.n_frames == 1
    assert frac_error_ang(traj.frame(0), tr, 0, np.arange(N)) <= DOUBLE_ANG


def test_netcdf_refusals(tmp_path):
    path = tmp_path / "other.nc"
    _scipy_netcdf(path, conventions="CF-1.6", labels=SYMBOLS)
    with pytest.raises(UnsupportedFormat, match="other.nc: .*Conventions .*"
                       "'CF-1.6', not AMBER"):
        mb.read_amber_netcdf(path)
    path = tmp_path / "parsec.nc"
    _scipy_netcdf(path, units="angstrom", labels=SYMBOLS)
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"angstrom", b"parsecs!"))
    with pytest.raises(ValueError, match="parsec.nc: coordinates is in "
                       "'parsecs!'; FACET reads lengths in Å or nm"):
        mb.read_amber_netcdf(path)
    hdf5 = tmp_path / "netcdf4.nc"
    hdf5.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(64))
    assert mb.FORMATS[2].sniff(hdf5.read_bytes(), hdf5)
    assert not mb.FORMATS[2].sniff(hdf5.read_bytes(), tmp_path / "model.h5")
    with pytest.raises(UnsupportedFormat, match="nccopy -k cdf5 netcdf4.nc"):
        mb.read_amber_netcdf(hdf5)


def test_netcdf_cut_streaming_and_fill(tmp_path):
    path = copy(tmp_path, "lammps_style.nc")
    raw = path.read_bytes()
    path.write_bytes(raw[:-100])
    traj = mb.read_amber_netcdf(path, type_map=TYPES)
    assert traj.n_frames == 2 and set(traj.skipped) == {2}
    assert traj.skipped[2].startswith("truncated")
    path.write_bytes(raw[:200])
    with pytest.raises(ValueError, match="lammps_style.nc: the NetCDF header "
                       "runs past"):
        mb.read_amber_netcdf(path, type_map=TYPES)
    # the streaming record count: the complete records are counted
    elements = truth("amber_style.nc")["elements"].tolist()
    path = copy(tmp_path, "amber_style.nc")
    raw = bytearray(path.read_bytes())
    raw[4:8] = b"\xff" * 4
    path.write_bytes(bytes(raw[:-10]))
    traj = mb.read_amber_netcdf(path, topology=elements)
    assert traj.n_frames == 2
    assert "streaming" in " ".join(traj.notes)
    # a last record still holding the fill value
    path = copy(tmp_path, "amber_style.nc")
    header = mb.netcdf_header(path)
    var = header.variables["coordinates"]
    raw = bytearray(path.read_bytes())
    start = var.begin + 2 * header.record_size
    raw[start:start + 4 * 3 * N] = np.full(3 * N, 9.9692099683868690e+36,
                                           dtype=">f4").tobytes()
    path.write_bytes(bytes(raw))
    traj = mb.read_amber_netcdf(path, topology=elements)
    assert traj.n_frames == 2
    assert traj.skipped[2].startswith("unwritten")


def test_damaged_later_frame_names_the_file(tmp_path):
    path = copy(tmp_path, "lammps_style.nc")
    header = mb.netcdf_header(path)
    var = header.variables["coordinates"]
    raw = bytearray(path.read_bytes())
    start = var.begin + 1 * header.record_size
    raw[start:start + 8] = np.array([np.nan], dtype=">f8").tobytes()
    path.write_bytes(bytes(raw))
    traj = mb.read_amber_netcdf(path, type_map=TYPES)
    with pytest.raises(FrameError, match=r"frame 1 \(file position 1\): "
                       r"lammps_style.nc: .*non-finite coordinates"):
        traj.frame(1)


def test_box_from_cell_parameters_round_trips_lower_triangular_rows():
    rng = np.random.default_rng(3)
    for _ in range(20):
        rows = np.array([[rng.uniform(5, 20), 0, 0],
                         [rng.uniform(-3, 3), rng.uniform(5, 20), 0],
                         [rng.uniform(-3, 3), rng.uniform(-3, 3),
                          rng.uniform(5, 20)]])
        norms = np.linalg.norm(rows, axis=1)
        cos = [rows[1] @ rows[2] / norms[1] / norms[2],
               rows[0] @ rows[2] / norms[0] / norms[2],
               rows[0] @ rows[1] / norms[0] / norms[1]]
        rebuilt = mb.box_from_cell_parameters(norms, np.degrees(np.arccos(cos)))
        assert np.abs(rebuilt - rows).max() <= 1e-12
    square = mb.box_from_cell_parameters([3, 4, 5], [90, 90, 90])
    assert np.array_equal(square, np.diag([3.0, 4.0, 5.0]))


# ---------------------------------------------------------------------------
# what every reader shares
# ---------------------------------------------------------------------------

READS = [
    ("ase_md.traj", mb.read_ase_traj, {}),
    ("hoomd_small.gsd", mb.read_gsd, {}),
    ("lammps_style.nc", mb.read_amber_netcdf, {"type_map": TYPES}),
]


@pytest.mark.parametrize("name,reader,options", READS)
def test_gzip_copies_are_refused_with_how_to_decompress(tmp_path, name, reader,
                                                        options):
    path = tmp_path / (name + ".gz")
    path.write_bytes(gzip.compress((DATA / name).read_bytes()))
    with pytest.raises(UnsupportedFormat, match=f"{re.escape(path.name)}: a "
                       "gzip-compressed .*decompress it first") as raised:
        reader(path, **options)
    # 'FACET reads GSD file files' read as a typo (integration, 2026-10-07)
    assert "file files" not in str(raised.value)


@pytest.mark.parametrize("name,reader,options", READS)
def test_trajectories_pickle_describe_and_close(name, reader, options):
    traj = reader(DATA / name, **options)
    clone = pickle.loads(pickle.dumps(traj))
    assert np.array_equal(clone.frame(1).frac, traj.frame(1).frac)
    lines = traj.describe()
    assert lines[1] == f"format: {traj.file_format}"
    assert traj.notes[0].startswith(f"read {traj.source_path} as ")
    with traj as opened:
        assert opened.frame(-1).n_atoms == N
    assert traj.ids_track_atoms is True
    assert_no_verdicts(messages(traj) + lines)


def test_unknown_options_are_refused():
    with pytest.raises(TypeError):
        mb.read_ase_traj(DATA / "ase_md.traj", type_map=TYPES)
    with pytest.raises(TypeError):
        mb.read_gsd(DATA / "hoomd_small.gsd", topology=SYMBOLS)


def test_refusal_messages_carry_no_verdicts(tmp_path):
    texts = []
    for call in (lambda: mb.read_amber_netcdf(DATA / "mda_small.nc"),
                 lambda: mb.read_amber_netcdf(DATA / "lammps_style.nc"),
                 lambda: mb.read_amber_netcdf(DATA / "ovito_small.nc",
                                              type_map=TYPES),
                 lambda: mb.read_ase_traj(DATA / "ase_md.traj", timestep_fs=1.0)):
        with pytest.raises(ValueError) as caught:
            call()
        texts.append(str(caught.value))
    assert_no_verdicts(texts)
    for name in ("ovito_face.gsd", "ovito_ambiguous.gsd", "ase_small.nc",
                 "ovito_xz.nc"):
        reader = {".gsd": mb.read_gsd, ".nc": mb.read_amber_netcdf}[Path(name).suffix]
        options = {"type_map": TYPES} if name == "ovito_xz.nc" else {}
        assert_no_verdicts(messages(reader(DATA / name, **options)))


# ---------------------------------------------------------------------------
# what the verification of this module found: each test failed before the fix
# ---------------------------------------------------------------------------

def _ulm_with_item0(raw: bytes, record: bytes) -> bytes:
    """The ULM file with frame 0's record replaced by ``record`` (JSON),
    appended at the end of the file and pointed to by the table, as a writer
    appending a record would place it."""
    _, _, pos0 = struct.unpack_from("<qqq", raw, 24)
    out = bytearray(raw) + struct.pack("<q", len(record)) + record
    struct.pack_into("<q", out, pos0, len(raw))
    return bytes(out)


_GSD_CODES = {np.dtype("u1"): 1, np.dtype("<u2"): 2, np.dtype("<u4"): 3,
              np.dtype("<u8"): 4, np.dtype("i1"): 5, np.dtype("<i2"): 6,
              np.dtype("<i4"): 7, np.dtype("<i8"): 8, np.dtype("<f4"): 9,
              np.dtype("<f8"): 10}


def _write_gsd(path: Path, frames, application="FACET test (GSD file layer 2.0)"):
    """A GSD file (file layer 2.0, schema hoomd 1.4) of ``frames``, each a
    dict of chunk name -> array, written by hand for chunks no writer
    produces (a float dimensions chunk, a step past 2**63, a box of length
    0). The gsd 5.0.1 library reads files written this way (checked once:
    gsd.hoomd returns the steps, boxes, types and positions given)."""
    names = sorted({n for f in frames for n in f})
    data = bytearray()
    entries = []
    for number, chunks in enumerate(frames):
        for name in sorted(chunks):
            array = np.ascontiguousarray(chunks[name])
            array = array.reshape(len(array), -1)
            entries.append((number, array.shape[0], 256 + len(data),
                            array.shape[1], names.index(name),
                            _GSD_CODES[array.dtype], 0))
            data += array.tobytes()
    index = np.zeros(len(entries) + 1, dtype=mb._GSD_INDEX)
    for i, entry in enumerate(entries):
        index[i] = entry
    index_location = 256 + len(data)
    block = b"\0".join(n.encode() for n in names) + b"\0"
    allocated = -(-len(block) // 64)
    header = struct.pack("<QQQQQII", mb._GSD_MAGIC, index_location, len(index),
                         index_location + index.nbytes, allocated,
                         (1 << 16) | 4, 2 << 16)
    header += application.encode().ljust(64, b"\0") + \
        b"hoomd".ljust(64, b"\0") + bytes(80)
    path.write_bytes(header + bytes(data) + index.tobytes()
                     + block.ljust(64 * allocated, b"\0"))


def _gsd_frame(step, box=(5.0, 5.0, 5.0, 0.0, 0.0, 0.0), **extra):
    out = {"configuration/step": np.array([step], dtype="<u8"),
           "configuration/box": np.array(box, dtype="<f4"),
           "particles/N": np.array([2], dtype="<u4"),
           "particles/types": np.frombuffer(b"Si\0", dtype="i1").reshape(1, 3),
           "particles/typeid": np.array([0, 0], dtype="<u4"),
           "particles/position": np.array([[0.5, 0.5, 0.5], [-1.0, 1.0, 0.0]],
                                          dtype="<f4")}
    out.update(extra)
    return out


def test_gsd_letters_without_masses_need_a_map_for_every_name():
    # 'A', 'B', 'C' with HOOMD's default mass, which the gsd library leaves
    # out: nothing tells 'B' the label from 'B' boron
    assert "particles/mass" not in mb._gsd_index(DATA / "hoomd_letters.gsd").frames[0]
    with pytest.raises(ValueError) as caught:
        mb.read_gsd(DATA / "hoomd_letters.gsd")
    text = str(caught.value)
    assert text.startswith("hoomd_letters.gsd: no element for the type name(s) "
                           "'A', 'B', 'C'. ")
    assert "'A' is not an element symbol" in text
    assert "type_map={'A': '<element>', 'B': '<element>', 'C': '<element>'}" in text
    # a map for 'A' alone, which the refusal used to suggest, read the
    # sodium atoms as boron and the oxygen atoms as carbon
    with pytest.raises(ValueError) as partial:
        mb.read_gsd(DATA / "hoomd_letters.gsd", type_map={"A": "Si"})
    assert str(partial.value).startswith("hoomd_letters.gsd: no element for "
                                         "the type name(s) 'B', 'C'. ")
    assert "the map given has no entry for ['B', 'C']" in str(partial.value)
    traj = mb.read_gsd(DATA / "hoomd_letters.gsd",
                       type_map={"A": "Si", "B": "Na", "C": "O"})
    assert traj.type_map_source == "user"
    assert_matches(traj, "hoomd_letters.gsd", by_id=False, pos_tol=SINGLE_ANG,
                   box_tol=SINGLE_ANG, vel_rel=1e-6, steps=True)
    assert_no_verdicts([text, str(partial.value)] + messages(traj))


def test_gsd_symbols_without_masses_are_noted():
    traj = mb.read_gsd(DATA / "ovito_face.gsd")
    assert "type name(s) ['Na', 'O', 'Si'] read as element symbols without a " \
        "mass to check them" in " ".join(traj.notes)
    assert "mass_tol_amu not used" not in " ".join(traj.notes)
    traj = mb.read_gsd(DATA / "ovito_face.gsd", mass_tol_amu=0.5)
    assert "mass_tol_amu not used: the file stores no particles/mass" in traj.notes


def test_gsd_labels_take_their_element_from_the_mass():
    traj = mb.read_gsd(DATA / "hoomd_mass_labels.gsd")
    assert traj.type_map_source == "data-file masses"
    assert traj.type_map == {"Na_m": "Na", "O_b": "O", "Si_t": "Si"}
    assert_matches(traj, "hoomd_mass_labels.gsd", by_id=False, pos_tol=SINGLE_ANG,
                   box_tol=SINGLE_ANG, vel_rel=1e-6, steps=True)
    assert "type name 'Si_t' is not an element symbol, and frame 0's " \
        "particles/mass gives it a mass 28.0855 amu read as Si" \
        in " ".join(traj.notes)
    # a mass two elements lie within names neither
    with pytest.raises(ValueError, match="hoomd_mass_labels.gsd: no element for "
                       r"the type name\(s\) .*'Si_t' \(mass 28.0855 amu"):
        mb.read_gsd(DATA / "hoomd_mass_labels.gsd", mass_tol_amu=1.5)
    assert_no_verdicts(messages(traj))


def test_gsd_in_nm_is_refused_unless_units_nano():
    with pytest.raises(ValueError) as caught:
        mb.read_gsd(DATA / "mdtraj_nm.gsd")
    text = str(caught.value)
    assert text.startswith("mdtraj_nm.gsd: with lengths read as Å, frame 0 "
                           "holds 8 atoms")
    assert "Pass units='nano' when the lengths are in nm" in text
    with pytest.raises(ValueError, match="hoomd_small.gsd: .*lj.*units='nano'"):
        mb.read_gsd(DATA / "hoomd_small.gsd", units="lj")
    traj = md_readers.read_trajectory(DATA / "mdtraj_nm.gsd", units="nano")
    tr = truth("mdtraj_nm.gsd")
    index = mb._gsd_index(DATA / "mdtraj_nm.gsd")
    with open(DATA / "mdtraj_nm.gsd", "rb") as handle:
        for k in range(F):
            frame = traj.frame(k)
            box6 = mb._gsd_values(handle, index.frames[k]["configuration/box"],
                                  "box").reshape(6)
            stored = mb._gsd_values(handle,
                                    index.frames[k]["particles/position"],
                                    "position").astype(float)
            rows, origin = mb.box_from_hoomd(box6)
            # the file's own statement, in nm, times 10
            assert np.abs(frame.box_ang - 10.0 * rows).max() <= 1e-12
            want = (10.0 * (stored - origin)) @ np.linalg.inv(10.0 * rows)
            d = frame.frac - want
            assert np.abs(d - np.round(d)).max() <= 1e-12
            # mdtraj's box: its Lx, Ly, Lz are the model's, to float32 in nm
            assert np.abs(np.diag(frame.box_ang) - np.diag(tr["box"][k])).max() \
                <= 2e-6
    assert traj.units_note.startswith("LAMMPS units nano (the units argument): "
                                      "lengths in nm, x 10 to Å")
    # mdtraj writes its tilts as lengths and its positions as given
    assert "lie outside the box the HOOMD schema defines" in " ".join(traj.notes)
    assert "mdtraj's GSD writer" in " ".join(traj.notes)
    assert "outside the box" not in " ".join(
        mb.read_gsd(DATA / "hoomd_small.gsd").notes)
    assert_no_verdicts([text] + messages(traj))


def test_netcdf_restart_reads_its_time(tmp_path):
    traj = mb.read_amber_netcdf(DATA / "mdtraj.ncrst", topology=SYMBOLS)
    tr = truth("mdtraj.ncrst")
    assert traj.n_frames == 1
    assert np.array_equal(traj.times_ps, tr["time_ps"])
    assert frac_error_ang(traj.frame(0), tr, 0, np.arange(N)) <= 5 * SINGLE_ANG
    assert "frame times from the time variable (picosecond)" in traj.notes
    for layout in ("scalar", "length 1"):
        path = tmp_path / "restart.ncrst"
        _scipy_netcdf(path, restart=True, labels=SYMBOLS, restart_time=layout)
        traj = mb.read_amber_netcdf(path)
        assert traj.times_ps.tolist() == [2.5], layout
    _scipy_netcdf(path, restart=True, labels=SYMBOLS, restart_time="per atom")
    traj = mb.read_amber_netcdf(path)
    assert traj.times_ps is None
    assert "time not read: time has dimensions ('atom',)" in " ".join(traj.notes)
    assert "no time variable in the file" not in traj.notes


def test_lammps_float_time_holds_steps_rounded_past_2_24():
    traj = mb.read_amber_netcdf(DATA / "lammps_float.nc", type_map=TYPES)
    tr = truth("lammps_float.nc")
    assert mb.netcdf_header(DATA / "lammps_float.nc").variables["time"].dtype \
        == np.dtype(">f4")
    assert traj.timesteps.tolist() == tr["stored_steps"].tolist() \
        == [16777200, 16777216, 20000000]
    assert np.array_equal(traj.times_ps, tr["time_ps"])
    text = " ".join(traj.notes)
    assert "steps from the time variable" in text
    assert "steps above that are stored rounded (16777217 as 16777216)" in text
    assert_matches(traj, "lammps_float.nc", by_id=True, pos_tol=2e-5,
                   box_tol=2e-5)


def test_whole_scaled_times_are_steps_only_for_lammps(tmp_path):
    path = copy(tmp_path, "lammps_style.nc")
    raw = path.read_bytes()
    assert raw.count(b"LAMMPS") == 1
    path.write_bytes(raw.replace(b"LAMMPS", b"LAMMPX"))
    traj = mb.read_amber_netcdf(path, type_map=TYPES)
    assert (traj.timesteps == NO_TIMESTEP).all()
    assert np.array_equal(traj.times_ps, truth("lammps_style.nc")["time_ps"])


def test_damaged_ase_item_count_costs_no_memory(tmp_path):
    # an item count of 10**6 + 3 for three frames
    path = copy(tmp_path, "ase_md.traj")
    raw = bytearray(path.read_bytes())
    struct.pack_into("<q", raw, 32, 10 ** 6 + 3)
    path.write_bytes(bytes(raw))
    traj = mb.read_ase_traj(path)
    assert traj.n_frames == 3
    assert len(traj.skipped) < 10 ** 4          # not one entry per counted frame
    assert any("this one entry stands for" in r for r in traj.skipped.values())


def test_damaged_netcdf_record_count_costs_no_memory(tmp_path):
    # a record count of 10**5 + 3 for three records
    path = copy(tmp_path, "lammps_style.nc")
    raw = bytearray(path.read_bytes())
    struct.pack_into(">q", raw, 4, 10 ** 5 + 3)
    path.write_bytes(bytes(raw))
    traj = mb.read_amber_netcdf(path, type_map=TYPES)
    assert traj.n_frames == 3 and list(traj.skipped) == [3]
    assert "this one entry stands for 100000 frames" in traj.skipped[3]
    assert "the header counts 100000 frames from file position 3 on" \
        in " ".join(traj.notes)


def test_damaged_gsd_frame_number_is_refused_as_gsd_refuses_it(tmp_path):
    # an index entry naming frame 10**5, which the gsd library refuses
    path = copy(tmp_path, "hoomd_small.gsd")
    raw = bytearray(path.read_bytes())
    index_location, allocated = struct.unpack_from("<QQ", raw, 8)
    index = np.frombuffer(bytes(raw[index_location:index_location
                                    + 32 * allocated]), dtype=mb._GSD_INDEX)
    entries = int(np.flatnonzero(index["location"] == 0)[0])
    assert index["frame"][entries - 1] == 2
    struct.pack_into("<Q", raw, index_location + 32 * (entries - 1), 10 ** 5)
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match=r"hoomd_small.gsd: GSD index entry \d+ "
                       "names frame 100000, and the index has room for"):
        mb.read_gsd(path)
    # ... and frame 5: frames 3 and 4 hold no chunk, and are not read as
    # copies of frame 0
    struct.pack_into("<Q", raw, index_location + 32 * (entries - 1), 5)
    path.write_bytes(bytes(raw))
    traj = mb.read_gsd(path)
    assert traj.file_positions.tolist() == [0, 1, 2, 5]
    assert set(traj.skipped) == {3, 4}
    assert all(r.startswith("the GSD index lists no chunk for it")
               for r in traj.skipped.values())
    assert_no_verdicts(messages(traj))


def test_later_frame_errors_name_the_file_and_hold_to_frame_0(tmp_path):
    # LAMMPS fix atom/swap: one atom's type changes in frame 2
    path = copy(tmp_path, "lammps_style.nc")
    header = mb.netcdf_header(path)
    offset = mb._nc_offset(header, header.variables["type"], 2)
    raw = bytearray(path.read_bytes())
    types = np.frombuffer(bytes(raw[offset:offset + 4 * N]), dtype=">i4").copy()
    types[0] = 3 if types[0] != 3 else 1
    raw[offset:offset + 4 * N] = types.tobytes()
    path.write_bytes(bytes(raw))
    traj = md_readers.read_trajectory(path, type_map=TYPES)
    traj.frame(1)
    with pytest.raises(FrameError, match=r"^lammps_style.nc, frame 2 \(file "
                       r"position 2\): 1 atom\(s\) carry a different element "
                       "from frame 0"):
        traj.frame(2)
    # frame 0 is the reference even when frame 2 is the first one loaded
    traj = mb.read_amber_netcdf(path, type_map=TYPES)
    with pytest.raises(FrameError, match="from frame 0"):
        traj.frame(2)
    assert traj.frame(0).n_atoms == N


def test_damaged_ase_records_give_named_value_errors(tmp_path):
    raw = (DATA / "ase_md.traj").read_bytes()
    _, item = _ulm_item0(DATA / "ase_md.traj")
    path = tmp_path / "ase_md.traj"
    path.write_bytes(_ulm_with_item0(raw, json.dumps(dict(item, pbc=5)).encode()))
    traj = mb.read_ase_traj(path)
    assert traj.periodic is None
    assert "pbc not read for the frame(s) at file position(s) [0, 1, 2]: it " \
        "is not three flags (5)" in traj.notes
    shaped = dict(item, **{"positions.": {"ndarray": [[10 ** 20, 3], "float64",
                                                      120]}})
    path.write_bytes(_ulm_with_item0(raw, json.dumps(shaped).encode()))
    with pytest.raises(ValueError, match="ase_md.traj: frame 0 cannot be read "
                       r"\(truncated: its positions array ends beyond"):
        mb.read_ase_traj(path)
    deep = json.dumps(item)[:-1] + ', "d": ' + "[" * 100000 + "]" * 100000 + "}"
    path.write_bytes(_ulm_with_item0(raw, deep.encode()))
    with pytest.raises(ValueError, match=r"ase_md.traj: frame 0 cannot be read "
                       r"\(its JSON record does not parse"):
        mb.read_ase_traj(path)
    origin = dict(item, info=dict(item["info"], cell_origin=[1.0, 2.0]))
    path.write_bytes(_ulm_with_item0(raw, json.dumps(origin).encode()))
    text = " ".join(mb.read_ase_traj(path).notes)
    assert "info['cell_origin'] of the frame(s) at file position(s) [0] is not " \
        "three numbers" in text
    assert "ASE stores no box origin" not in text


def test_ase_offset_table_damage_is_named(tmp_path):
    raw = bytearray((DATA / "ase_md.traj").read_bytes())
    _, _, pos0 = struct.unpack_from("<qqq", raw, 24)
    first = struct.unpack_from("<q", raw, pos0)[0]
    path = tmp_path / "ase_md.traj"
    damaged = bytearray(raw)
    struct.pack_into("<q", damaged, pos0 + 8, first)      # frame 1 -> frame 0's
    struct.pack_into("<q", damaged, pos0 + 16, 0)         # frame 2 -> byte 0
    path.write_bytes(bytes(damaged))
    traj = mb.read_ase_traj(path)
    assert traj.n_frames == 1
    assert "the record of file position 0; the table names one record twice" \
        in traj.skipped[1]
    assert "names byte 0, inside the 48-byte ULM header" in traj.skipped[2]
    assert_no_verdicts(messages(traj))


def test_ase_masses_set_later_are_named_by_frame():
    traj = mb.read_ase_traj(DATA / "ase_masses_later.traj")
    tr = truth("ase_masses_later.traj")
    note = next(n for n in traj.notes if n.startswith("velocities"))
    assert note.startswith("velocities of the frame(s) at file position(s) "
                           "[1, 2] from the momenta divided by the masses the "
                           "file stores; the frame(s) at file position(s) [0] "
                           "store no masses")
    assert "the file stores no masses" not in note
    weights = dict(md_readers.element_weights_amu())
    gemmi_mass = np.array([weights[s] for s in SYMBOLS])
    for k in range(F):
        velocity = traj.frame(k).vel_ang_per_ps
        if k == 0:          # ASE divided by its own table, FACET by gemmi's
            velocity = velocity * (gemmi_mass / tr["ase_masses"])[:, None]
        assert np.abs(velocity - tr["vel"][k]).max() \
            <= 1e-12 * np.abs(tr["vel"][k]).max()


def test_gsd_frames_repeating_frame_0_are_named():
    index = mb._gsd_index(DATA / "hoomd_static.gsd")
    assert [sorted(index.frames[k]) for k in (1, 2)] == \
        [["configuration/step"]] * 2
    traj = mb.read_gsd(DATA / "hoomd_static.gsd")
    assert_matches(traj, "hoomd_static.gsd", by_id=False, pos_tol=SINGLE_ANG,
                   box_tol=SINGLE_ANG, steps=True)
    text = " ".join(traj.notes)
    assert "the frame(s) at file position(s) [1, 2] hold no particles/position " \
        "and repeat frame 0's positions" in text
    assert "the frame(s) at file position(s) [1, 2] hold no configuration/box " \
        "and repeat frame 0's box" in text
    assert "repeat frame 0's" not in " ".join(
        mb.read_gsd(DATA / "ovito_face.gsd").notes)


def test_damaged_gsd_chunks_give_named_reasons(tmp_path):
    path = tmp_path / "hand.gsd"
    # configuration/dimensions as a float
    _write_gsd(path, [_gsd_frame(0, **{"configuration/dimensions":
                                       np.array([3.0], dtype="<f4")})])
    with pytest.raises(ValueError, match="^hand.gsd: configuration/dimensions "
                       "is 1 x 1 float32, not one integer"):
        mb.read_gsd(path)
    # a step past the int64 range in frame 1
    _write_gsd(path, [_gsd_frame(0), _gsd_frame(2 ** 63), _gsd_frame(20)])
    traj = mb.read_gsd(path)
    assert traj.timesteps.tolist() == [0, 20]
    assert "configuration/step is 9223372036854775808, outside the 64-bit " \
        "signed range" in traj.skipped[1]
    # a box of length 0 in frame 0
    _write_gsd(path, [_gsd_frame(0, box=(0.0, 5.0, 5.0, 0.1, 0.0, 0.0))],
               application="OVITO 3.16.1")
    with pytest.raises(ValueError, match=r"^hand.gsd: frame 0 cannot be read "
                       r"\(configuration/box gives the lengths Lx, Ly, Lz as "
                       r"\[0.0, 5.0, 5.0\]; a periodic box has three positive"):
        mb.read_gsd(path)


def test_ovito_gsd_box_that_cannot_be_ovitos_is_read_with_the_schema(tmp_path):
    path = copy(tmp_path, "ovito_face.gsd")
    chunk = mb._gsd_index(path).frames[0]["configuration/box"]
    raw = bytearray(path.read_bytes())
    end = chunk.location + chunk.nbytes
    box = np.frombuffer(bytes(raw[chunk.location:end]), dtype=chunk.dtype).copy()
    box[3] = 1.5                     # xy = b_x/|b| = 1.5: no box gives it
    raw[chunk.location:end] = box.tobytes()
    path.write_bytes(bytes(raw))
    traj = mb.read_gsd(path)
    assert "file position(s) [0] cannot be what OVITO's GSD exporter " \
        "(GSDExporter.cpp) writes" in " ".join(traj.notes)
    rows, _ = mb.box_from_hoomd(box)
    assert np.allclose(traj.frame(0).box_ang, rows, rtol=0, atol=1e-12)


def test_damaged_netcdf_headers_give_named_reasons(tmp_path, monkeypatch):
    # an id variable of characters
    path = tmp_path / "chars.nc"
    _scipy_netcdf(path, labels=SYMBOLS, char_ids=True)
    with pytest.raises(FrameError, match=r"^chars.nc, frame 0 \(file position "
                       r"0\): id is stored as \|S1 \(text\), not as numbers"):
        mb.read_amber_netcdf(path)
    # a cell_spatial dimension of length 2
    path = tmp_path / "cell2.nc"
    _scipy_netcdf(path, labels=SYMBOLS)
    raw = bytearray(path.read_bytes())
    at = raw.index(b"cell_spatial") + len(b"cell_spatial")
    assert struct.unpack_from(">i", raw, at)[0] == 3
    struct.pack_into(">i", raw, at, 2)
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="^cell2.nc: the dimension "
                       "'cell_spatial' has length 2"):
        mb.read_amber_netcdf(path)
    # time data placed past the end of the file
    path = copy(tmp_path, "lammps_style.nc")
    header = mb.netcdf_header(path)
    raw = bytearray(path.read_bytes())
    begin = struct.pack(">q", header.variables["time"].begin)
    assert raw[:header.header_bytes].count(begin) == 1
    at = raw.index(begin)
    struct.pack_into(">q", raw, at, 1 << 40)
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="^lammps_style.nc holds no readable "
                       "frame: position 0: truncated: its time data ends "
                       "beyond the end of the file"):
        mb.read_amber_netcdf(path, type_map=TYPES)
    # a scale_factor of 0 on the velocities
    path = copy(tmp_path, "amber_style.nc")
    raw = path.read_bytes()
    assert raw.count(struct.pack(">d", 20.455)) == 1
    path.write_bytes(raw.replace(struct.pack(">d", 20.455), struct.pack(">d", 0.0)))
    with pytest.raises(ValueError, match="^amber_style.nc: velocities's "
                       "scale_factor is 0, which would make every value 0"):
        mb.read_amber_netcdf(path, topology=SYMBOLS)
    # a header longer than FACET reads (the limit lowered for the test)
    monkeypatch.setattr(mb, "_NC_HEADER_FIRST_READ", 512)
    monkeypatch.setattr(mb, "_NC_HEADER_MAX", 1024)
    path = tmp_path / "long.nc"
    _scipy_netcdf(path, labels=SYMBOLS, comment="x" * 4000)
    with pytest.raises(ValueError, match="^long.nc: the NetCDF header is longer "
                       "than the 1024 bytes FACET reads of a header"):
        mb.netcdf_header(path)


def test_read_exact_names_the_bytes_it_lacks():
    import io

    with pytest.raises(ValueError, match=r"^what: bytes 10 to 20 lie beyond the "
                       r"end of the file \(2 of the 10 bytes are there\)$"):
        mb._read_exact(io.BytesIO(bytes(12)), 10, 10, "what")


def test_topology_takes_arrays_and_trajectories_and_names_the_file(tmp_path):
    traj = mb.read_amber_netcdf(DATA / "mda_small.nc", topology=np.array(SYMBOLS))
    assert traj.frame(0).elements.tolist() == SYMBOLS
    # a trajectory read_trajectory returned
    tr = truth("mda_small.nc")
    lines = [f"{N}", 'Lattice="' + " ".join(str(x) for x in tr["box"][0].ravel())
             + '" Properties=species:S:1:pos:R:3']
    lines += [f"{s} {x} {y} {z}" for s, (x, y, z) in zip(SYMBOLS, tr["pos"][0])]
    model = tmp_path / "model.extxyz"
    model.write_text("\n".join(lines) + "\n")
    traj = mb.read_amber_netcdf(DATA / "mda_small.nc",
                                topology=md_readers.read_trajectory(model))
    assert traj.frame(0).elements.tolist() == SYMBOLS
    assert "frame 0 of <trajectory of model.extxyz>" in " ".join(traj.notes)
    cif = Path(__file__).resolve().parent / "data" / "crystals" / \
        "quartz_SiO2_cod9013321.cif"
    texts = []
    for topology, start in (
            (cif, "mda_small.nc: topology=quartz_SiO2_cod9013321.cif: "),
            (SYMBOLS[:-1] + ["Xx"], "mda_small.nc: topology=<list of 8>: "
             "entry 7, 'Xx', is not an element symbol"),
            ([14, 8, 11, 8, 14, 8, 11, 8], "mda_small.nc: topology=<list of "
             "8>: entry 0 is 14, not an element symbol"),
            (tmp_path / "missing.data", "mda_small.nc: topology=missing.data: "),
            ({"Si": 1}, "mda_small.nc: topology=<dict of 1>: topology takes")):
        with pytest.raises(ValueError) as caught:
            mb.read_amber_netcdf(DATA / "mda_small.nc", topology=topology)
        assert str(caught.value).startswith(start), str(caught.value)
        assert "type map to an element" not in str(caught.value)
        texts.append(str(caught.value))
    assert_no_verdicts(texts)


def test_options_a_file_does_not_use_are_noted():
    traj = mb.read_amber_netcdf(DATA / "mda_small.nc", topology=SYMBOLS,
                                type_map={"Si": "Ge"})
    assert traj.frame(0).elements.tolist() == SYMBOLS
    assert "type map not used: the elements come from the topology argument" \
        in traj.notes
    traj = mb.read_amber_netcdf(DATA / "ase_small.nc", type_map={"Zz": "Si"})
    assert "type map entries ['Zz'] are not used: the elements come from the " \
        "atomic numbers in atom_types" in traj.notes


def test_text_files_under_binary_names_reach_their_own_reader(tmp_path):
    from facet.core import readers

    cif = Path(__file__).resolve().parent / "data" / "crystals" / \
        "quartz_SiO2_cod9013321.cif"
    for suffix in (".nc", ".traj", ".gsd", ".ncrst"):
        path = tmp_path / f"quartz{suffix}"
        shutil.copy(cif, path)
        assert not any(spec.sniff(path.read_bytes()[:md_readers.SNIFF_BYTES],
                                  path) for spec in mb.FORMATS)
        assert readers.read(path) is not None
    path = tmp_path / "md.traj"
    shutil.copy(TEXT_DATA / "dump_general.lammpstrj", path)
    assert md_readers.read_trajectory(path).file_format == "lammps-dump"


def test_no_bytecode_beside_the_fixtures():
    assert not (DATA / "__pycache__").exists()
