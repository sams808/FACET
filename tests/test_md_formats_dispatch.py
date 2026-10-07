"""Dispatch to the format modules, series input, and routing by readers.read.

What is pinned, and why:

* **Every module md_formats_base.FORMAT_MODULES lists imports** and exports a
  tuple of FormatSpec with names of its own: a module that is listed and
  missing (or left out of a frozen build) loses its formats with no other
  sign than this test.
* **The registry passes over a module that is not present** and names it,
  and names a module that is present and cannot be used, so one module's
  fault never stops the others; a sniff that raises recognises nothing.
* **Options are checked against FormatSpec.options**: an option a format does
  not take is refused, never ignored, and one read_trajectory does not know
  is a TypeError.
* **A directory, a wildcard pattern or a list of paths** reads as one
  trajectory: natural order for a directory and a pattern (dump.50 before
  dump.100, where text order puts 100 first), the list's own order for a
  list, the order in the notes; a format's own read_series is called with
  the files; built-in formats are joined frame after frame, a file that
  cannot be read is one skipped position, and nothing is dropped silently.
* **readers.read routes** every registered extension and stem, and every
  binary format whatever its name, to MDModelFile; a format module's sniff
  is not consulted for a crystal name, so a POSCAR stays a crystal.
* **Every format is reached** (integration, 2026-10-07): no listed module is
  absent, the registry holds every FormatSpec of every module, and for each
  one a committed fixture is named by sniff_md, opened by that format's
  reader through read_trajectory, and routed by readers.read under the same
  name (a lone POSCAR stays a crystal); a format with a series reader reads a
  pattern of its files.

The fake modules are built here; the real ones are exercised where present.
The LAMMPS series was written by LAMMPS 22 Jul 2025
(tests/data/md/recognition/make_recognition_files.py).
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from facet.core import md_formats_base, md_readers, readers
from facet.core.md_formats_base import FORMAT_MODULES, FormatSpec
from facet.core.md_model import MemoryTrajectory, frame_from_arrays

DATA = Path(__file__).resolve().parent / "data" / "md"
RECOGNITION = DATA / "recognition"
SERIES = RECOGNITION / "series"
TRUTH = json.loads((RECOGNITION / "recognition_truth.json").read_text())
TYPES = {1: "Si", 2: "Na", 3: "O"}
PACKAGE = "facet.core"


# ---------------------------------------------------------------------------
# the real modules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", FORMAT_MODULES)
def test_every_listed_format_module_imports(name):
    module = importlib.import_module(f"{PACKAGE}.{name}")
    formats = module.FORMATS
    assert isinstance(formats, tuple) and formats
    assert all(isinstance(spec, FormatSpec) for spec in formats)


def test_the_registry_holds_every_format_of_the_present_modules_once():
    registry = md_readers.format_registry()
    assert registry.failed == ()
    names = [spec.name for spec in registry.specs]
    assert len(names) == len(set(names))
    assert not set(names) & set(md_readers.MD_FORMATS)
    assert md_readers.format_names() == md_readers.MD_FORMATS + tuple(names)
    for spec in registry.specs:
        assert set(spec.options) <= md_formats_base.KNOWN_OPTIONS


def _crystal_name(suffix: str) -> bool:
    return suffix in readers.READERS


@pytest.mark.parametrize("spec", md_readers.format_specs(),
                         ids=lambda s: s.name)
def test_readers_read_routes_each_registered_extension(tmp_path, spec):
    """A file named with a format module's extension goes to the MD reader
    even when its content is not that format's (the MD reader then says
    why): the crystal reader never sees it. Crystal extensions are routed
    by content only, and stems compare as prefixes."""
    names = [f"model{e}" for e in spec.extensions if not _crystal_name(e)]
    names += [f"{stem}_1" for stem in spec.stems]
    if not names:
        pytest.skip(f"{spec.name} is recognised by content only")
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"\x00not a model\n" if spec.binary else b"not a model\n")
        with pytest.raises(readers.MDModelFile) as raised:
            readers.read(path)
        found = raised.value.file_format
        claimed = (md_readers.binary_md_format(path) if spec.binary
                   else md_readers.named_md_format(path))
        assert found == claimed.name
        assert "read_trajectory" in str(raised.value)


# ---------------------------------------------------------------------------
# fake modules
# ---------------------------------------------------------------------------

def _frame(z: float):
    return frame_from_arrays(["Si", "O"], [[1.0, 1.0, z], [2.5, 1.0, 1.0]],
                             box_ang=np.eye(3) * 10.0)


class _Calls:
    def __init__(self) -> None:
        self.read: list[tuple[Path, dict]] = []
        self.series: list[tuple[list[Path], dict]] = []


def _fake_text_module(calls: _Calls, *, accept_all: bool = False):
    module = types.ModuleType(f"{PACKAGE}.md_formats_fake_text")

    def sniff(head: bytes, path: Path) -> bool:
        return accept_all or head.startswith(b"FAKE-MD")

    def read(path, *, type_map=None, timestep_fs=None):
        calls.read.append((Path(path), {"type_map": type_map,
                                        "timestep_fs": timestep_fs}))
        return MemoryTrajectory([_frame(1.0)], source_path=str(path),
                                file_format="fake-text",
                                type_map_source="file symbols")

    def read_series(paths, *, type_map=None, timestep_fs=None):
        calls.series.append(([Path(p) for p in paths],
                             {"type_map": type_map, "timestep_fs": timestep_fs}))
        return MemoryTrajectory([_frame(1.0 + k) for k in range(len(paths))],
                                source_path=str(paths[0]),
                                file_format="fake-text",
                                type_map_source="file symbols")

    module.FORMATS = (FormatSpec(
        name="fake-text", description="the fake text format of the tests",
        extensions=(".fakemd",), stems=("FAKETRAJ",), sniff=sniff, read=read,
        options=frozenset({"type_map", "timestep_fs"}), read_series=read_series),)
    return module


def _fake_binary_module(calls: _Calls):
    module = types.ModuleType(f"{PACKAGE}.md_formats_fake_bin")

    def read(path, *, topology=None):
        calls.read.append((Path(path), {"topology": topology}))
        return MemoryTrajectory([_frame(2.0)], source_path=str(path),
                                file_format="fake-bin",
                                type_map_source="file symbols")

    module.FORMATS = (FormatSpec(
        name="fake-bin", description="the fake binary format of the tests",
        extensions=(".fakebin",), stems=(),
        sniff=lambda head, path: head.startswith(b"\x00FAKEBIN"), read=read,
        options=frozenset({"topology"}), binary=True),)
    return module


@pytest.fixture
def fake(monkeypatch):
    """Two fake modules and one that is not written, in place of the real
    list; the real modules are left out so that only the fakes answer."""
    calls = _Calls()
    text = _fake_text_module(calls)
    binary = _fake_binary_module(calls)
    monkeypatch.setitem(sys.modules, text.__name__, text)
    monkeypatch.setitem(sys.modules, binary.__name__, binary)
    monkeypatch.setattr(md_formats_base, "FORMAT_MODULES",
                        ("md_formats_fake_text", "md_formats_fake_bin",
                         "md_formats_not_written_yet"))
    return calls


def test_the_registry_passes_over_a_module_that_is_not_present(fake):
    registry = md_readers.format_registry()
    assert [s.name for s in registry.specs] == ["fake-text", "fake-bin"]
    assert registry.absent == ("md_formats_not_written_yet",)
    assert registry.failed == ()


def test_a_module_that_cannot_be_used_is_named_and_the_others_still_answer(
        fake, monkeypatch, tmp_path):
    broken = types.ModuleType(f"{PACKAGE}.md_formats_fake_broken")
    broken.FORMATS = ["not a FormatSpec"]
    twin = types.ModuleType(f"{PACKAGE}.md_formats_fake_twin")
    twin.FORMATS = (FormatSpec(name="lammps-dump", description="a clash",
                               extensions=(), stems=(),
                               sniff=lambda h, p: True, read=lambda p: None),)
    monkeypatch.setitem(sys.modules, broken.__name__, broken)
    monkeypatch.setitem(sys.modules, twin.__name__, twin)
    monkeypatch.setattr(md_formats_base, "FORMAT_MODULES",
                        md_formats_base.FORMAT_MODULES
                        + ("md_formats_fake_broken", "md_formats_fake_twin"))
    registry = md_readers.format_registry()
    assert [s.name for s in registry.specs] == ["fake-text", "fake-bin"]
    assert [m for m, _ in registry.failed] == ["md_formats_fake_broken",
                                               "md_formats_fake_twin"]
    assert "already taken" in registry.failed[1][1]
    path = tmp_path / "junk.txt"
    path.write_text("not a model\n")
    with pytest.raises(readers.UnsupportedFormat) as raised:
        md_readers.read_trajectory(path)
    assert "md_formats_fake_broken" in str(raised.value)
    assert md_readers.sniff_md(DATA / "dump_tri_x.lammpstrj") == "lammps-dump"


def test_a_sniff_that_raises_recognises_nothing(fake, monkeypatch, tmp_path):
    def boom(head, path):
        raise RuntimeError("a fault in a format module")

    spec = sys.modules[f"{PACKAGE}.md_formats_fake_text"].FORMATS[0]
    monkeypatch.setattr(sys.modules[f"{PACKAGE}.md_formats_fake_text"],
                        "FORMATS", (FormatSpec(**{**spec.__dict__,
                                                  "sniff": boom}),))
    path = tmp_path / "model.dat"
    path.write_bytes(b"\x00FAKEBIN rest")
    assert md_readers.sniff_md(path) == "fake-bin"


def test_built_in_formats_come_before_the_modules(fake, monkeypatch):
    calls = _Calls()
    everything = _fake_text_module(calls, accept_all=True)
    monkeypatch.setitem(sys.modules, everything.__name__, everything)
    assert md_readers.sniff_md(DATA / "dump_tri_x.lammpstrj") == "lammps-dump"
    assert md_readers.sniff_md(DATA / "XDATCAR_fixed") == "vasp-xdatcar"


def test_read_trajectory_dispatches_with_the_options_the_format_takes(
        fake, tmp_path):
    path = tmp_path / "model.dat"
    path.write_bytes(b"FAKE-MD 1\n")
    assert md_readers.sniff_md(path) == "fake-text"
    traj = md_readers.read_trajectory(path, type_map={"X": "Si"},
                                      timestep_fs=2.0)
    assert traj.file_format == "fake-text"
    assert fake.read == [(path, {"type_map": {"X": "Si"}, "timestep_fs": 2.0})]
    with pytest.raises(ValueError, match=r"model\.dat is read as fake-text "
                                         r"\(options it takes: timestep_fs, "
                                         r"type_map\), which does not use "
                                         r"units$"):
        md_readers.read_trajectory(path, units="metal")
    with pytest.raises(TypeError, match="has no option colour"):
        md_readers.read_trajectory(path, colour="red")
    binary = tmp_path / "model.bin2"
    binary.write_bytes(b"\x00FAKEBIN")
    md_readers.read_trajectory(binary, topology="glass.data")
    assert fake.read[-1] == (binary, {"topology": "glass.data"})
    with pytest.raises(ValueError, match="does not use box_from$"):
        md_readers.read_trajectory(binary, box_from=np.eye(3))


def test_the_refusal_names_the_module_formats(fake, tmp_path):
    path = tmp_path / "junk.txt"
    path.write_text("not a model\n")
    with pytest.raises(readers.UnsupportedFormat) as raised:
        md_readers.read_trajectory(path)
    message = str(raised.value)
    assert message.startswith("junk.txt: not recognised as an MD model")
    assert "the fake text format of the tests" in message
    assert "the fake binary format of the tests" in message
    assert "readers.read" in message


@pytest.mark.parametrize("name", ["model.fakemd", "FAKETRAJ_0001"])
def test_readers_read_routes_a_registered_extension_or_stem(fake, tmp_path,
                                                            name):
    path = tmp_path / name
    path.write_text("anything\n")
    with pytest.raises(readers.MDModelFile) as raised:
        readers.read(path)
    assert raised.value.file_format == "fake-text"
    assert "the fake text format of the tests" in str(raised.value)


@pytest.mark.parametrize("name", ["model.fakebin", "glass.xyz", "POSCAR",
                                  "model.pdb", "model.cif", "noext"])
def test_a_binary_format_never_reaches_a_crystal_reader(fake, tmp_path, name):
    """By its extension, or by its first bytes under any name, crystal
    names included."""
    path = tmp_path / name
    path.write_bytes(b"\x00FAKEBIN" + bytes(64) if name != "model.fakebin"
                     else b"\x00not the magic")
    with pytest.raises(readers.MDModelFile) as raised:
        readers.read(path)
    assert raised.value.file_format == "fake-bin"


def test_a_module_sniff_is_not_consulted_for_a_crystal_name(fake, monkeypatch,
                                                            tmp_path):
    """A format module whose sniff accepts anything does not take a POSCAR
    from the crystal reader: the crystal names are routed by the built-in
    content tests only."""
    calls = _Calls()
    everything = _fake_text_module(calls, accept_all=True)
    monkeypatch.setitem(sys.modules, everything.__name__, everything)
    poscar = (DATA / "XDATCAR_fixed").read_text().splitlines()[:7] + [
        "Direct", "0.1 0.2 0.3", "0.25 0.15 0.45", "0.6 0.7 0.2",
        "0.35 0.9 0.65", "0.85 0.4 0.9"]
    for name in ("POSCAR", "model.vasp"):
        path = tmp_path / name
        path.write_text("\n".join(poscar) + "\n")
        structure = readers.read(path)
        assert len(structure.atoms) == 5


# ---------------------------------------------------------------------------
# series
# ---------------------------------------------------------------------------

def test_a_format_reads_its_own_series_in_natural_order(fake, tmp_path):
    for step in (100, 5, 50):
        (tmp_path / f"snap.{step}.fake").write_bytes(b"FAKE-MD\n")
    traj = md_readers.read_trajectory(str(tmp_path / "snap.*.fake"),
                                      type_map={"X": "Si"})
    (paths, options), = fake.series
    assert [p.name for p in paths] == ["snap.5.fake", "snap.50.fake",
                                       "snap.100.fake"]
    assert options == {"type_map": {"X": "Si"}, "timestep_fs": None}
    assert traj.n_frames == 3
    assert ("3 file(s) read as one trajectory, the files the pattern matches "
            "in natural order of their names (digits compared as numbers): "
            "snap.5.fake, snap.50.fake, snap.100.fake") in traj.notes


def test_a_list_keeps_its_order(fake, tmp_path):
    paths = []
    for step in (100, 5):
        path = tmp_path / f"snap.{step}.fake"
        path.write_bytes(b"FAKE-MD\n")
        paths.append(path)
    md_readers.read_trajectory(paths)
    assert [p.name for p in fake.series[0][0]] == ["snap.100.fake",
                                                   "snap.5.fake"]


def _dump_series(**options):
    return md_readers.read_trajectory(str(SERIES / "dump.*.lammpstrj"),
                                      type_map=TYPES, **options)


def test_a_lammps_dump_series_in_natural_order():
    """LAMMPS wrote dump.0, dump.50, dump.100 (one file per snapshot); text
    order puts 100 before 50. The frames are the run's, file by file."""
    with _dump_series() as traj:
        assert traj.file_format == "lammps-dump"
        assert traj.timesteps.tolist() == [0, 50, 100]
        assert [Path(p).name for p in traj.source_paths] == [
            "dump.0.lammpstrj", "dump.50.lammpstrj", "dump.100.lammpstrj"]
        assert traj.notes[1].startswith("3 file(s) read as one trajectory")
        assert not any("do not increase" in n for n in traj.notes)
        run = TRUTH["lammps_run"]
        box, origin = np.array(run["box"]), np.array(run["origin"])
        for k in range(len(traj)):
            frame = traj.frame(k)
            single = md_readers.read_trajectory(
                SERIES / f"dump.{run['timesteps'][k]}.lammpstrj",
                type_map=TYPES).frame(0)
            assert np.array_equal(frame.cart_ang, single.cart_ang)
            assert np.array_equal(frame.unwrapped_cart_ang,
                                  single.unwrapped_cart_ang)
            # against LAMMPS's own state, modulo the lattice (%g rounding)
            gap = np.linalg.solve(box.T, (frame.cart_ang
                                          - np.array(run["positions"][k])).T).T
            gap -= np.round(gap)
            assert np.abs(gap @ box).max() < 1e-5
            assert np.abs(frame.origin_ang - origin).max() < 1e-12
            assert frame.elements.tolist() == run["elements"]
        assert traj.box_varies is False


def test_a_directory_reads_its_files_of_one_format_and_names_the_others(
        tmp_path):
    for name in ("dump.0.lammpstrj", "dump.50.lammpstrj", "dump.100.lammpstrj"):
        shutil.copy(SERIES / name, tmp_path / name)
    (tmp_path / "in.glass").write_text("units metal\natom_style atomic\n"
                                       "read_data glass.data\nrun 100\n")
    shutil.copy(RECOGNITION / "nvt.data", tmp_path / "nvt.data")
    (tmp_path / ".hidden").write_text("x")
    with md_readers.read_trajectory(tmp_path, type_map=TYPES) as traj:
        assert traj.timesteps.tolist() == [0, 50, 100]
        (note,) = [n for n in traj.notes if "of the directory not read" in n]
        assert "in.glass (not recognised as an MD model)" in note
        assert "nvt.data (lammps-data)" in note
        assert ".hidden" not in note


def test_a_list_in_text_order_is_kept_and_its_steps_are_noted():
    """C5 of the recognition test: the order a list gives is the order
    read, and steps that go back are noted, never re-sorted silently."""
    files = sorted(str(p) for p in SERIES.iterdir())
    with md_readers.read_trajectory(files, type_map=TYPES) as traj:
        assert traj.timesteps.tolist() == [0, 100, 50]
        assert any("timestep 100 is followed by 50" in n for n in traj.notes)
        assert any("in the order of the list given" in n for n in traj.notes)


def test_a_file_of_the_series_that_cannot_be_read_is_one_skipped_position(
        tmp_path):
    shutil.copy(SERIES / "dump.0.lammpstrj", tmp_path / "dump.0.lammpstrj")
    shutil.copy(SERIES / "dump.100.lammpstrj", tmp_path / "dump.100.lammpstrj")
    text = (SERIES / "dump.50.lammpstrj").read_text()
    (tmp_path / "dump.50.lammpstrj").write_text(text[:len(text) // 2])
    with md_readers.read_trajectory(str(tmp_path / "dump.*.lammpstrj"),
                                    type_map=TYPES) as traj:
        assert traj.n_frames == 2
        assert list(traj.skipped) == [1]
        assert traj.skipped[1].startswith("dump.50.lammpstrj")
        assert traj.file_positions.tolist() == [0, 2]
        assert traj.timesteps.tolist() == [0, 100]


def test_a_file_with_another_atom_count_has_every_frame_skipped(tmp_path):
    for name in ("dump.0.lammpstrj", "dump.50.lammpstrj"):
        shutil.copy(SERIES / name, tmp_path / name)
    lines = (SERIES / "dump.100.lammpstrj").read_text().splitlines()
    lines[3] = "17"
    (tmp_path / "dump.100.lammpstrj").write_text("\n".join(lines[:-1]) + "\n")
    with md_readers.read_trajectory(str(tmp_path / "dump.*.lammpstrj"),
                                    type_map=TYPES) as traj:
        assert traj.n_frames == 2
        assert traj.skipped == {2: "dump.100.lammpstrj: 17 atoms, where 2 of "
                                   "the 3 files hold 18"}


@pytest.mark.parametrize("source, error, message", [
    ([], ValueError, "empty list"),
    ([3], ValueError, "not a path"),
])
def test_series_refusals(source, error, message):
    with pytest.raises(error, match=message):
        md_readers.read_trajectory(source)


def test_a_pattern_that_matches_nothing_and_mixed_formats(tmp_path):
    with pytest.raises(FileNotFoundError, match="no file matches this pattern"):
        md_readers.read_trajectory(str(tmp_path / "dump.*.lammpstrj"))
    shutil.copy(SERIES / "dump.0.lammpstrj", tmp_path / "a.1")
    shutil.copy(RECOGNITION / "nvt.data", tmp_path / "a.2")
    with pytest.raises(readers.UnsupportedFormat,
                       match="a.1 is lammps-dump and a.2 is lammps-data"):
        md_readers.read_trajectory(str(tmp_path / "a.*"))
    (tmp_path / "a.3").write_text("junk\n")
    with pytest.raises(readers.UnsupportedFormat,
                       match=r"a\.3: not recognised as an MD model.*It is one "
                             r"of the 3 files of"):
        md_readers.read_trajectory([tmp_path / "a.3", tmp_path / "a.1",
                                    tmp_path / "a.2"])


def test_a_series_without_its_type_map_says_what_to_pass():
    with pytest.raises(ValueError, match=r"no file of the series holds a "
                                         r"readable frame.*type_map="
                                         r"\{1: '<element>'"):
        md_readers.read_trajectory(SERIES)


def test_natural_sort_key():
    names = ["dump.100", "dump.20", "dump.3", "dump.20b", "Dump.4"]
    assert sorted(names, key=md_readers.natural_sort_key) == [
        "dump.3", "Dump.4", "dump.20", "dump.20b", "dump.100"]


def test_a_series_pickles_and_reads_after_unpickling():
    import pickle

    with _dump_series() as traj:
        copy = pickle.loads(pickle.dumps(traj))
        assert np.array_equal(copy.frame(2).cart_ang, traj.frame(2).cart_ang)
        copy.close()


# ---------------------------------------------------------------------------
# the core readers' review (2026-10-07): each test failed before its fix
# ---------------------------------------------------------------------------

def _fake_typed_module(*, varies: bool = False):
    """A text format whose reader needs a type map, as a format module's
    reader of numeric types does."""
    module = types.ModuleType(f"{PACKAGE}.md_formats_fake_typed")

    def read(path, *, type_map=None):
        if not type_map:
            raise ValueError(f"{Path(path).name}: the atoms carry numeric "
                             "types; pass type_map={1: '<element>'}")
        frames = [frame_from_arrays(["Si", "O"], [[1.0, 1.0, 1.0],
                                                  [2.5, 1.0, 1.0]],
                                    box_ang=np.eye(3) * (10.0 + k))
                  for k in range(2 if varies else 1)]
        return MemoryTrajectory(frames, source_path=str(path),
                                file_format="fake-typed",
                                type_map_source="user")

    module.FORMATS = (FormatSpec(
        name="fake-typed", description="the fake typed format of the tests",
        extensions=(), stems=(), sniff=lambda head, path: head.startswith(
            b"FAKE-TYPED"), read=read, options=frozenset({"type_map"})),)
    return module


@pytest.mark.parametrize("varies", [False, True])
def test_box_from_a_module_file_that_needs_a_type_map(monkeypatch, tmp_path,
                                                       varies):
    """box_from opened a format module's file with no option, so a reader
    that needs a type map asked for one, and passing type_map= to
    read_trajectory gave the same refusal (it went to the XYZ only). The
    type map now reaches that reader too; without one, the refusal says
    what box_from takes. A source whose box changes is refused."""
    module = _fake_typed_module(varies=varies)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(md_formats_base, "FORMAT_MODULES",
                        ("md_formats_fake_typed",))
    source = tmp_path / "run.typed"
    source.write_bytes(b"FAKE-TYPED\n")
    xyz = RECOGNITION / "nvt_types.xyz"
    with pytest.raises(ValueError) as raised:
        md_readers.read_trajectory(xyz, box_from=source)
    message = str(raised.value)
    # the file being read is named first (integration re-run, 2026-10-07)
    assert message.startswith("nvt_types.xyz: box_from=run.typed: no box was "
                              "taken from it")
    assert "give the box as (3, 3) rows in Å" in message
    if varies:
        with pytest.raises(ValueError, match="the box changes from frame to "
                                             "frame"):
            md_readers.read_trajectory(xyz, box_from=source, type_map=TYPES)
        return
    with md_readers.read_trajectory(xyz, box_from=source,
                                    type_map=TYPES) as traj:
        assert np.array_equal(traj.frame(0).box_ang, np.eye(3) * 10.0)


def _damaged_gzip(source: Path, target: Path) -> None:
    import gzip

    data = bytearray(gzip.compress(source.read_bytes()))
    data[30:40] = b"\xff" * 10
    target.write_bytes(bytes(data))


def test_a_series_member_that_cannot_be_decoded_is_a_skipped_position(
        tmp_path):
    """A damaged gzip member of a directory was named 'not recognised as an
    MD model', its reason lost, and took no skipped position; a hidden file
    was passed over with no note."""
    for name in ("dump.0.lammpstrj", "dump.50.lammpstrj"):
        shutil.copy(SERIES / name, tmp_path / name)
    _damaged_gzip(SERIES / "dump.100.lammpstrj",
                  tmp_path / "dump.100.lammpstrj.gz")
    shutil.copy(SERIES / "dump.50.lammpstrj", tmp_path / ".dump.150.lammpstrj")
    with md_readers.read_trajectory(tmp_path, type_map=TYPES) as traj:
        assert traj.timesteps.tolist() == [0, 50]
        assert list(traj.skipped) == [2]
        assert traj.skipped[2].startswith("dump.100.lammpstrj.gz: the gzip "
                                          "stream is damaged")
        assert any(n.startswith("1 hidden file(s) of the directory")
                   and ".dump.150.lammpstrj" in n for n in traj.notes)
        assert not any("of the directory not read" in n for n in traj.notes)


def test_an_empty_member_of_a_pattern_is_a_skipped_position(tmp_path):
    """A 0-byte member (a run still writing it) refused the whole pattern
    with the generic 'not recognised as an MD model'."""
    for name in ("dump.0.lammpstrj", "dump.50.lammpstrj", "dump.100.lammpstrj"):
        shutil.copy(SERIES / name, tmp_path / name)
    (tmp_path / "dump.150.lammpstrj").write_bytes(b"")
    with md_readers.read_trajectory(str(tmp_path / "dump.*.lammpstrj"),
                                    type_map=TYPES) as traj:
        assert traj.timesteps.tolist() == [0, 50, 100]
        assert traj.skipped == {3: "dump.150.lammpstrj: the file is empty "
                                   "(0 bytes), as a run that was writing it "
                                   "leaves it"}


def test_series_messages_of_the_review(tmp_path):
    with pytest.raises(OSError, match="is a directory, not a file"):
        md_readers.read_trajectory([SERIES / "dump.0.lammpstrj", tmp_path],
                                   type_map=TYPES)
    with pytest.raises(FileNotFoundError, match=r"glob\.escape"):
        md_readers.read_trajectory(str(tmp_path / "run[1]" / "dump.*"))


# ---------------------------------------------------------------------------
# integration (2026-10-07): every listed module is used, every format reached
# ---------------------------------------------------------------------------

# One committed fixture per format of the format modules, with the options its
# reader needs; each was written by the program its folder's generator names
# (lammps_formats/make_lammps_formats.py: LAMMPS 22 Jul 2025 update 4 and ASE
# 3.29.0; xtc: make_xtc_files.py; binary/make_binary_files.py and
# text/make_text_files.py: ASE, OVITO and the other writers they list).
REACH = {
    "dcd": ("lammps_formats/traj.dcd",
            {"topology": DATA / "lammps_formats" / "topology.data"}),
    "lammps-dump-binary": ("lammps_formats/traj_custom.bin",
                           {"type_map": TYPES}),
    "lammps-dump-yaml": ("lammps_formats/traj.yaml", {"type_map": TYPES}),
    "atomeye-cfg": ("lammps_formats/ase.cfg", {}),
    "gromacs-xtc": ("xtc_lammps.xtc", {"topology": DATA / "xtc_lammps.data"}),
    "ase-traj": ("binary/ase_md.traj", {}),
    "gsd": ("binary/hoomd_small.gsd", {}),
    "amber-netcdf": ("binary/ase_small.nc", {}),
    "castep-md": ("text/ase.md", {}),
    "gromacs-gro": ("text/ase_vel.gro", {}),
    "xsf": ("text/ase_var.axsf", {}),
    "pdb-models": ("text/ase_models.pdb", {}),
    "imd": ("text/ovito_mass.imd", {}),
    "vasp-poscar": ("text/POSCAR_ase_000", {}),
}
# A series of each format whose module reads one (FormatSpec.read_series):
# a wildcard pattern under tests/data/md, or a file copied to make one.
SERIES_OF = {
    "atomeye-cfg": ("lammps_formats/cfg/dump.*.cfg", None),
    "vasp-poscar": ("text/POSCAR_ase_*", None),
    "imd": (None, "text/ovito_mass.imd"),
}
# The formats readers.read keeps as a crystal by design: a lone POSCAR under a
# VASP name is one structure (md_formats_text's module docstring).
CRYSTAL_BY_DESIGN = {"vasp-poscar"}


def test_every_listed_module_is_present_and_each_of_its_formats_is_used():
    """No listed module is absent or failed (a frozen build that leaves one
    out shows here as 'absent'), and the registry holds every FormatSpec of
    every module, in the order of FORMAT_MODULES then FORMATS."""
    registry = md_readers.format_registry()
    assert registry.absent == ()
    assert registry.failed == ()
    expected = []
    for name in FORMAT_MODULES:
        expected += list(importlib.import_module(f"{PACKAGE}.{name}").FORMATS)
    assert list(registry.specs) == expected
    assert {spec.read.__module__ for spec in registry.specs} == {
        f"{PACKAGE}.{name}" for name in FORMAT_MODULES}


def test_a_frozen_build_is_shown_every_listed_module():
    """format_registry imports the modules by name, which PyInstaller cannot
    follow; md_formats_base._imports_for_frozen_builds names them in import
    statements it does follow (measured with PyInstaller 6.11.1: none of the
    four was in the module graph before). That list and FORMAT_MODULES name
    the same modules, so a module added to one is added to the other."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(
        md_formats_base._imports_for_frozen_builds))
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.level == 1
                for alias in node.names}
    assert imported == set(FORMAT_MODULES)


def test_every_format_has_a_reachability_fixture():
    names = {spec.name for spec in md_readers.format_specs()}
    assert set(REACH) == names
    assert set(SERIES_OF) == {spec.name for spec in md_readers.format_specs()
                              if spec.read_series is not None}


@pytest.mark.parametrize("spec", md_readers.format_specs(),
                         ids=lambda s: s.name)
def test_dispatch_reaches_every_format(spec):
    """sniff_md names the format, read_trajectory opens the file with that
    format's reader (its file_format, a frame that loads), and readers.read
    routes it to the MD reader under the same name -- except a lone POSCAR,
    which readers.read keeps a crystal."""
    rel, options = REACH[spec.name]
    path = DATA / rel
    assert md_readers.sniff_md(path) == spec.name
    with md_readers.read_trajectory(path, **options) as traj:
        assert traj.file_format == spec.name
        assert traj.n_frames >= 1
        frame = traj.frame(0)
        assert frame.n_atoms == traj.n_atoms > 0
        assert np.isfinite(frame.cart_ang).all()
    if spec.name in CRYSTAL_BY_DESIGN:
        structure = readers.read(path)
        assert len(structure.atoms) == traj.n_atoms
        return
    with pytest.raises(readers.MDModelFile) as raised:
        readers.read(path)
    assert raised.value.file_format == spec.name


@pytest.mark.parametrize("name", sorted(SERIES_OF))
def test_dispatch_reaches_every_format_series_reader(name, tmp_path):
    pattern, copy = SERIES_OF[name]
    if pattern is None:
        source = DATA / copy
        for k in range(3):
            shutil.copy(source, tmp_path / f"run.{k}{source.suffix}")
        pattern = str(tmp_path / f"run.*{source.suffix}")
        count = 3
    else:
        count = len(list(DATA.glob(pattern)))
        pattern = str(DATA / pattern)
    assert count >= 2
    options = REACH[name][1]
    with md_readers.read_trajectory(pattern, **options) as traj:
        assert traj.file_format == name
        assert traj.n_frames + len(traj.skipped) == count
        assert not traj.skipped
        assert any(note.startswith(f"{count} file(s) read as one trajectory")
                   for note in traj.notes)


# The items the format modules' reviews left to the md_readers / readers
# owner, each failing before its fix (integration re-run, 2026-10-07).

CRYSTAL_NAMES = ("model.cif", "model.xyz", "model.pdb", "POSCAR", "model.vasp",
                 "model.res", "model.vesta")


@pytest.mark.parametrize("rel, name", [
    ("lammps_formats/traj.yaml", "lammps-dump-yaml"),
    ("lammps_formats/ase.cfg", "atomeye-cfg"),
    ("text/ase.md", "castep-md"),
    ("text/ase_var.axsf", "xsf"),
    ("text/ovito_mass.imd", "imd"),
    ("text/ase_vel.gro", "gromacs-gro"),
], ids=lambda x: x if "/" not in x else None)
def test_a_format_module_text_file_under_a_crystal_name_is_routed(
        tmp_path, rel, name):
    """A YAML dump saved as .vasp gave a bare ValueError ('could not convert
    string to float') naming no file, and as .pdb, .res or .vesta 'no
    atoms', 'no CELL instruction', 'no CELLP section'; as .cif or .xyz it was
    told to convert a format FACET reads. readers.read now gives such a file
    to the MD reader once the crystal reader stops on it."""
    for crystal_name in CRYSTAL_NAMES:
        path = tmp_path / crystal_name
        shutil.copy(DATA / rel, path)
        with pytest.raises(readers.MDModelFile) as raised:
            readers.read(path)
        assert raised.value.file_format == name, crystal_name
        with md_readers.read_trajectory(path, **REACH[name][1]) as traj:
            assert traj.file_format == name


def test_a_crystal_file_under_a_crystal_name_still_reaches_its_reader(
        tmp_path, monkeypatch):
    """The routing above happens only after the crystal reader stopped: a
    format module whose sniff accepts everything takes no CIF or POSCAR, and
    a POSCAR the reader stops on keeps the crystal reader's refusal."""
    calls = _Calls()
    everything = _fake_text_module(calls, accept_all=True)
    monkeypatch.setitem(sys.modules, everything.__name__, everything)
    monkeypatch.setattr(md_formats_base, "FORMAT_MODULES",
                        ("md_formats_fake_text",))
    cif = tmp_path / "quartz.cif"
    shutil.copy(Path(__file__).resolve().parent / "data" / "crystals"
                / "quartz_SiO2_cod9013321.cif", cif)
    assert len(readers.read(cif).atoms) > 0
    monkeypatch.undo()
    broken = tmp_path / "model.vasp"
    broken.write_text("title\n1.0\n5 0 0\n0 5 0\n0 0 5\nSi\nx\nDirect\n0 0 0\n")
    with pytest.raises(readers.UnsupportedFormat) as raised:
        readers.read(broken)
    assert not isinstance(raised.value, readers.MDModelFile)
    # a bare ValueError of a crystal reader is named and says what FACET reads
    assert str(raised.value).startswith(
        "model.vasp: the structure reader stopped on it (ValueError: ")
    assert "read_trajectory" in str(raised.value)


def test_a_cif_saved_as_pdb_is_told_the_name_it_opens_under(tmp_path):
    """gemmi's RuntimeError ('perhaps it is cif not pdb?') came back with
    no advice; the refusal now says the .cif name opens it."""
    path = tmp_path / "quartz.pdb"
    shutil.copy(Path(__file__).resolve().parent / "data" / "crystals"
                / "quartz_SiO2_cod9013321.cif", path)
    with pytest.raises(readers.UnsupportedFormat) as raised:
        readers.read(path)
    text = str(raised.value)
    assert text.startswith("quartz.pdb: the structure reader stopped on it")
    assert "saved with the .cif extension it opens with readers.read" in text


def test_a_damaged_first_xtc_header_is_read_past_through_read_trajectory(
        tmp_path):
    """The XTC reader indexes past a damaged first frame header, but
    read_trajectory refused the file before calling it (no sniff took it);
    a file under a binary format's extension whose content shows no other
    format now goes to that format's reader."""
    raw = bytearray((DATA / "xtc_lammps.xtc").read_bytes())
    raw[0:4] = bytes(4)                     # frame 0's magic number
    path = tmp_path / "run.xtc"
    path.write_bytes(bytes(raw))
    assert md_readers.sniff_md(path) is None
    with md_readers.read_trajectory(
            path, topology=DATA / "xtc_lammps.data") as traj:
        assert traj.file_format == "gromacs-xtc"
        assert list(traj.skipped) == [0] and traj.n_frames == 2
    junk = tmp_path / "junk.xtc"
    junk.write_bytes(b"\x00not a model\n")
    with pytest.raises(readers.UnsupportedFormat) as raised:
        md_readers.read_trajectory(junk)
    text = str(raised.value)
    assert text.startswith("junk.xtc: not recognised as an MD model")
    assert "Opened as gromacs-xtc by its name, the reader of that format " \
           "stopped: junk.xtc: " in text


def test_a_file_like_a_format_facet_reads_is_told_so(tmp_path):
    """The descriptions of unread files predate the format modules: an XTC
    whose header the sniff passed over was told only to convert it, as if
    FACET read no XTC."""
    raw = bytearray((DATA / "xtc_lammps.xtc").read_bytes())
    raw[52:56] = b"\x7f\xff\xff\xff"        # the atom count's repeat
    path = tmp_path / "run.dat"
    path.write_bytes(bytes(raw))
    with pytest.raises(readers.UnsupportedFormat) as raised:
        md_readers.read_trajectory(path)
    assert ("it looks like a GROMACS XTC trajectory (as LAMMPS 'dump xtc' "
            "writes); FACET's MD reader reads GROMACS XTC") in str(raised.value)
    assert "(gromacs-xtc), and this file's first bytes are not in the " \
           "layout that reader takes" in str(raised.value)
