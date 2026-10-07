"""The command line, ``py -3.11 -m facet.md``, run as a student would run it.

What these tests pin:

* ``--help`` and ``analyses`` list every analysis, the inputs each needs
  and the exit codes;
* ``describe`` reads the files under tests/data/md and says what it found;
  a dump with numeric types and no type map stops with exit code 3 and the
  flag to give;
* ``analyse`` with an input missing stops with exit code 2 before reading
  any frame, naming every missing input (formers, temperature, charges),
  and writes nothing;
* ``analyse`` on a model writes the workbook and the CSV directory, reports
  one pair search per frame per pass, runs every analysis whose inputs are
  given when ``--only`` is left out and lists the others with what they
  lack;
* the request file printed by ``template`` reads back as a request, and the
  value syntax of ``--set`` gives typed values;
* exit codes 4 (cancelled, the finished part written) and 5 (an output that
  cannot be written, the other outputs still written);
* what the review of 2026-10-07 found: a value no analysis can take, a
  frame selection the model cannot give and v_bond below v_list exit 2
  before the model is read, every one named; --ox adds to a request file's
  ox_overrides; a blank formers value is not 'none'; without a time axis
  the analyses of time are left out of the default selection;
* ``python -m facet.md`` imports no Qt.

The commands run in a subprocess from the repository root, as
``py -3.11 -m facet.md`` does; the cancel test calls ``cli.main`` in this
process to stop the run at a known point.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import openpyxl
import pytest

from facet.core import md_analysis as ma, md_model, readers
from facet.md import cli

ROOT = Path(__file__).resolve().parent.parent
MD = Path(__file__).resolve().parent / "data" / "md"
QUARTZ = Path(__file__).resolve().parent / "data" / "crystals" / \
    "quartz_SiO2_cod9013321.cif"


def run(*args, cwd=ROOT):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "facet.md", *map(str, args)],
                          cwd=str(cwd), capture_output=True, text=True,
                          encoding="utf-8", env=env)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """Three jittered frames of a 3 x 3 x 3 quartz supercell, as a
    multi-frame extended XYZ file with velocities."""
    structure = readers.read(QUARTZ)
    base, _ = md_model.supercell_frame(structure, (3, 3, 3))
    rng = np.random.default_rng(11)
    path = tmp_path_factory.mktemp("model") / "quartz.extxyz"
    lattice = " ".join(f"{v:.10f}" for v in base.box_ang.reshape(-1))
    lines = []
    for _ in range(3):
        cart = base.cart_ang + rng.normal(0.0, 0.03, base.cart_ang.shape)
        vel = rng.normal(0.0, 5.0, cart.shape)
        lines.append(str(base.n_atoms))
        lines.append(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:'
                     f'velo:R:3 pbc="T T T"')
        for symbol, (x, y, z), (u, v, w) in zip(base.elements, cart, vel,
                                                strict=True):
            lines.append(f"{symbol} {x:.10f} {y:.10f} {z:.10f} "
                         f"{u:.6f} {v:.6f} {w:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# help, analyses, describe
# ---------------------------------------------------------------------------

def test_help_lists_every_analysis_and_the_exit_codes():
    result = run("--help")
    assert result.returncode == 0, result.stderr
    for name in ma.ANALYSES:
        assert f"  {name} (" in result.stdout, name
    assert "exit codes" in result.stdout and "cancelled" in result.stdout
    listing = run("analyses")
    assert listing.returncode == 0
    assert "needs: formers, network.ring_criterion, network.ring_max_size" \
        in listing.stdout
    assert "needs: charges_e" in listing.stdout


def test_describe_reads_a_dump_and_states_the_oxidation_states():
    result = run("describe", MD / "dump_ortho_real.lammpstrj")
    assert result.returncode == cli.EXIT_OK, result.stderr
    assert "5 atoms per frame; 2 readable frame(s)" in result.stdout
    assert "Na1+ (common), O2- (common), Si4+ (common)" in result.stdout
    assert "LAMMPS units real" in result.stdout


def test_describe_without_a_type_map_names_the_flag_and_exits_3():
    result = run("describe", MD / "dump_tri_x.lammpstrj")
    assert result.returncode == cli.EXIT_READ
    assert "--type-map" in result.stderr
    mapped = run("describe", MD / "dump_tri_x.lammpstrj", "--type-map",
                 "1=Si,2=O,3=Na")
    assert mapped.returncode == cli.EXIT_OK, mapped.stderr
    assert "type map (user): 1 -> Si, 2 -> O, 3 -> Na" in mapped.stdout
    missing = run("describe", MD / "no_such_file.lammpstrj")
    assert missing.returncode == cli.EXIT_READ


# ---------------------------------------------------------------------------
# missing inputs: exit 2, nothing written
# ---------------------------------------------------------------------------

def test_analyse_without_formers_names_them_and_writes_nothing(tmp_path):
    out = tmp_path / "r.xlsx"
    result = run("analyse", MD / "dump_ortho_real.lammpstrj", "--only",
                 "glass", "--out", out)
    assert result.returncode == cli.EXIT_USAGE
    assert "glass: formers:" in result.stderr
    assert not out.exists()


def test_every_missing_physical_input_is_named_at_once(tmp_path):
    result = run("analyse", MD / "dump_ortho_real.lammpstrj", "--only",
                 "conductivity,nmr", "--out", tmp_path / "r")
    assert result.returncode == cli.EXIT_USAGE
    for name in ("temperature_k", "charges_e", "dynamics.fit_t_min_ps",
                 "nmr.correlation", "nmr.fwhm_ppm"):
        assert name in result.stderr, name
    assert not (tmp_path / "r").exists()


def test_an_unknown_option_lists_the_known_ones(tmp_path):
    result = run("analyse", MD / "dump_ortho_real.lammpstrj", "--formers",
                 "Si", "--set", "glass.minimum_rul=valley", "--out",
                 tmp_path / "r.xlsx")
    assert result.returncode == cli.EXIT_USAGE
    assert "unknown option glass.minimum_rul" in result.stderr
    assert "minimum_rule" in result.stderr


def test_a_csv_path_is_refused_for_the_output(tmp_path):
    result = run("analyse", MD / "dump_ortho_real.lammpstrj", "--formers",
                 "Si", "--out", tmp_path / "one.csv")
    assert result.returncode == cli.EXIT_USAGE
    assert "directory" in result.stderr


# ---------------------------------------------------------------------------
# a run
# ---------------------------------------------------------------------------

def test_analyse_writes_the_workbook_and_the_csv_directory(model, tmp_path):
    book, folder = tmp_path / "run.xlsx", tmp_path / "csv"
    result = run("analyse", model, "--formers", "Si", "--only",
                 "glass,rings,voronoi,msd,vacf", "--frame-interval-ps", "0.1",
                 "--set", "network.ring_criterion=guttman",
                 "--set", "network.ring_max_size=10",
                 "--set", "dynamics.fit_t_min_ps=0.1",
                 "--set", "dynamics.fit_t_max_ps=0.2",
                 "--out", book, "--out", folder, "--quiet")
    assert result.returncode == cli.EXIT_OK, result.stdout + result.stderr
    assert "bulk pair searches per frame (pass 1, pass 2): 1, 1" in \
        result.stdout
    for name in ("glass", "rings", "voronoi", "msd", "vacf"):
        assert f"  {name}" in result.stdout
    workbook = openpyxl.load_workbook(book, read_only=True)
    assert workbook.sheetnames[:3] == ["provenance", "index", "notes"]
    workbook.close()
    with open(folder / "index.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(line for line in handle
                                   if not line.startswith("#")))
    analyses = {row["analysis"] for row in rows}
    assert analyses == {"glass", "rings", "voronoi", "msd", "vacf"}
    first = folder / rows[0]["file"]
    header = first.read_text(encoding="utf-8").splitlines()[:40]
    assert any("quartz.extxyz" in line for line in header)


def test_without_only_every_runnable_analysis_runs(model, tmp_path):
    result = run("analyse", model, "--formers", "Si", "--frame-interval-ps",
                 "0.1", "--frames", "0:3", "--set", "voids.radii=vdw",
                 "--out", tmp_path / "r")
    assert result.returncode == cli.EXIT_OK, result.stdout + result.stderr
    assert "left out: rings (needs network.ring_criterion, " \
        "network.ring_max_size)" in result.stderr
    assert "left out: conductivity" in result.stderr
    for name in ("glass", "empty-spheres", "warren-cowley", "bond-order",
                 "kinetic-temperature"):
        assert f"  {name}" in result.stdout, name
    assert "  rings" not in result.stdout


def test_the_template_reads_back_as_a_request(tmp_path):
    result = run("template")
    assert result.returncode == cli.EXIT_OK
    spec = tomllib.loads(result.stdout)
    request = cli.request_from_mapping(spec)
    assert request.glass.minimum_rule == "valley"
    assert request.formers is None and request.temperature_k is None
    written = tmp_path / "request.toml"
    assert run("template", "--out", written).returncode == cli.EXIT_OK
    assert tomllib.loads(written.read_text(encoding="utf-8")) == spec


def test_a_request_file_and_the_flags_combine(model, tmp_path):
    request_file = tmp_path / "request.toml"
    request_file.write_text(
        'analyses = ["glass", "rings"]\nformers = ["Si"]\n'
        '[network]\nring_criterion = "king"\nring_max_size = 8\n'
        '[glass]\ncutoffs_ang = {"Si-O" = 2.2}\n', encoding="utf-8")
    result = run("analyse", model, "--request", request_file, "--set",
                 "network.ring_max_size=10", "--out", tmp_path / "r",
                 "--quiet")
    assert result.returncode == cli.EXIT_OK, result.stdout + result.stderr
    # every glass cutoff given and the rings on the bonds: one pass
    assert "bulk pair searches per frame (pass 1, pass 2): 1, 0" in \
        result.stdout
    provenance = (tmp_path / "r" / "provenance.csv").read_text(
        encoding="utf-8")
    assert "network.ring_max_size = 10" in provenance
    assert "Si-O: 2.2 Å (user)" in provenance


def test_an_nmr_correlation_and_measured_fractions_from_files(model,
                                                              tmp_path):
    """The correlation and the measured fractions as JSON files: the
    numbers are test values, not published ones (none ships)."""
    import json

    correlation = tmp_path / "corr.json"
    correlation.write_text(json.dumps({
        "nucleus": "29Si", "intercept_ppm": -10.0, "reference": "test values",
        "terms": [{"descriptor": "T-O-T angle", "coefficient_ppm": -0.5,
                   "angle_function": "deg"}]}), encoding="utf-8")
    fractions = tmp_path / "fractions.json"
    fractions.write_text(json.dumps([{
        "model_descriptor": "Qn Si", "descriptor": "Si Qn (test values)",
        "values": {"4": 0.9, "3": 0.1}, "source": "test values"}]),
        encoding="utf-8")
    result = run("analyse", model, "--formers", "Si", "--only",
                 "glass,nmr,nmr-comparison", "--nmr-correlation", correlation,
                 "--nmr-fractions", fractions, "--set",
                 "nmr.delta_ppm=-130,-50,0.5", "--set", "nmr.fwhm_ppm=2",
                 "--set", "nmr.lineshape=gaussian", "--out", tmp_path / "r",
                 "--quiet")
    assert result.returncode == cli.EXIT_OK, result.stdout + result.stderr
    names = (tmp_path / "r" / "index.csv").read_text(encoding="utf-8")
    assert "Si Qn (test values) vs Qn Si (BV)" in names
    assert ",nmr," in names and "spectrum areas" in names
    unreferenced = tmp_path / "bare.json"
    unreferenced.write_text(json.dumps({
        "nucleus": "29Si", "intercept_ppm": -10.0, "terms": [
            {"descriptor": "Qn", "coefficient_ppm": 1.0}]}), encoding="utf-8")
    refused = run("analyse", model, "--formers", "Si", "--only", "nmr",
                  "--nmr-correlation", unreferenced, "--set",
                  "nmr.delta_ppm=-130,-50,0.5", "--set", "nmr.fwhm_ppm=2",
                  "--set", "nmr.lineshape=gaussian", "--out", tmp_path / "s")
    assert refused.returncode == cli.EXIT_USAGE
    assert "literature reference" in refused.stderr


def test_set_values_are_typed_from_the_option_annotations():
    request = cli.request_from_mapping({
        "analyses": "glass,exafs", "formers": "Si,B", "frames": "0:100:5",
        "ox": {"Na": "1"}, "charges_e": "Na=0.6,O=-1.2",
        "glass": {"cutoffs_ang": "Si-O=2.2,B-O=1.9",
                  "minimum_smooth_sigma_ang": "none",
                  "oxide_basis": "Si=SiO2,B=B2O3"},
        "exafs": {"absorber": "B", "shells": "O:0:2.0;Si:2.5:3.5"},
        "dynamics": {"lag_t_ps": "1,2.5", "remove_com_drift": "yes",
                     "distinct_pairs": "Na-Na,Na-O"},
        "type_map": "1=Si,2=O,lab=Na"})
    assert request.analyses == ("glass", "exafs")
    assert request.formers == frozenset({"Si", "B"})
    assert request.frames == slice(0, 100, 5)
    assert request.ox_overrides == {"Na": 1}
    assert request.charges_e == {"Na": 0.6, "O": -1.2}
    assert request.glass.cutoffs_ang == {("Si", "O"): 2.2, ("B", "O"): 1.9}
    assert request.glass.minimum_smooth_sigma_ang is None
    assert request.exafs.shells == (("O", 0.0, 2.0), ("Si", 2.5, 3.5))
    assert request.dynamics.lag_t_ps == (1.0, 2.5)
    assert request.dynamics.remove_com_drift is True
    assert request.dynamics.distinct_pairs == (("Na", "Na"), ("Na", "O"))
    assert request.type_map == {1: "Si", 2: "O", "lab": "Na"}
    assert cli.request_from_mapping({"formers": "none"}).formers == \
        ma.NO_FORMERS
    with pytest.raises(cli.UsageError, match="not a number"):
        cli.request_from_mapping({"temperature_k": "warm"})


# ---------------------------------------------------------------------------
# exit codes 4 and 5
# ---------------------------------------------------------------------------

def test_an_output_that_cannot_be_written_exits_5(model, tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("not a directory", encoding="utf-8")
    result = run("analyse", model, "--formers", "Si", "--only", "voronoi",
                 "--out", blocker / "inside", "--quiet")
    assert result.returncode == cli.EXIT_WRITE
    assert "could not be written" in result.stderr


def test_a_cancel_writes_the_finished_part_and_exits_4(model, tmp_path,
                                                      monkeypatch, capsys):
    state = {"stop": False}

    def progress(self, done, total, stage):
        if done >= 1:
            state["stop"] = True
    monkeypatch.setattr(cli._Progress, "__call__", progress)
    monkeypatch.setattr(cli._CancelOnInterrupt, "__call__",
                        lambda self: state["stop"])
    code = cli.main(["analyse", str(model), "--formers", "Si", "--only",
                     "glass,voronoi", "--out", str(tmp_path / "r"),
                     "--quiet"])
    assert code == cli.EXIT_CANCELLED
    assert (tmp_path / "r" / "index.csv").exists()
    provenance = (tmp_path / "r" / "provenance.csv").read_text(
        encoding="utf-8")
    assert "the run was cancelled" in provenance
    assert "(cancelled)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# what the review of the command line found (2026-10-07)
# ---------------------------------------------------------------------------

def test_a_request_files_ox_overrides_survive_the_ox_flag(tmp_path):
    """The file's own field name, ox_overrides, used to be replaced by the
    flag's alias key 'ox' (Fe3+ lost, every valence changed); now --ox adds
    to it, and a file giving both spellings is refused."""
    import json

    parser = cli.build_parser()
    for key in ("ox_overrides", "ox"):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({key: {"Fe": 3}}), encoding="utf-8")
        args = parser.parse_args(["analyse", "x.lammpstrj", "--request",
                                  str(path), "--ox", "Na=1", "--out", "o/"])
        request = cli.request_from_mapping(cli._spec_from_args(args))
        assert request.ox_overrides == {"Fe": 3, "Na": 1}, key
    both = tmp_path / "both.json"
    both.write_text(json.dumps({"ox": {"Fe": 3}, "ox_overrides": {"Fe": 2}}),
                    encoding="utf-8")
    args = parser.parse_args(["analyse", "x", "--request", str(both),
                              "--out", "o/"])
    with pytest.raises(cli.UsageError, match="twice"):
        cli._spec_from_args(args)
    with pytest.raises(cli.UsageError, match="twice"):
        cli.request_from_mapping({"v_bond": 0.1, "v_bond_vu": 0.2})


def test_only_the_word_none_states_no_formers():
    """A blank --set formers= used to state that the model has no formers;
    it now leaves them not given, which glass refuses."""
    args = cli.build_parser().parse_args(["analyse", "x", "--set", "formers=",
                                          "--out", "o/"])
    assert cli.request_from_mapping(cli._spec_from_args(args)).formers is None
    assert cli.request_from_mapping({"formers": ""}).formers is None
    assert cli.request_from_mapping({"formers": "None"}).formers == \
        ma.NO_FORMERS


def test_v_bond_below_the_default_v_list_exits_2(model, tmp_path):
    result = run("analyse", model, "--formers", "Si", "--only", "glass",
                 "--v-bond", "0.01", "--out", tmp_path / "r", "--quiet")
    assert result.returncode == cli.EXIT_USAGE, result.stdout + result.stderr
    assert "v_bond_vu 0.01" in result.stderr and "v_list_vu 0.02" in \
        result.stderr
    assert not (tmp_path / "r").exists()


def test_impossible_values_exit_2_naming_every_one_with_no_traceback(
        model, tmp_path):
    """Zero steps and bin widths used to end in a ZeroDivisionError
    traceback (some after every frame), bad names and ranges in exit 1 once
    per frame; each is now named at once, before the model is read."""
    result = run("analyse", model, "--formers", "Si", "--only",
                 "scattering,bond-order,nmr,rings,exafs",
                 "--set", "scattering.r_window=none",
                 "--set", "scattering.radiations=gamma",
                 "--set", "scattering.dr_ang=0",
                 "--set", "order.q_bin=0",
                 "--set", "network.ring_criterion=bogus",
                 "--set", "network.ring_max_size=0",
                 "--set", "exafs.weighting=bogus", "--absorber", "Si",
                 "--nmr-correlation", tmp_path / "missing.json",
                 "--out", tmp_path / "r")
    # the correlation file is read with the request: its absence alone
    assert result.returncode == cli.EXIT_USAGE
    assert "missing.json" in result.stderr
    result = run("analyse", model, "--formers", "Si", "--only",
                 "scattering,bond-order,rings,exafs",
                 "--set", "scattering.r_window=none",
                 "--set", "scattering.radiations=gamma",
                 "--set", "scattering.dr_ang=0",
                 "--set", "order.q_bin=0",
                 "--set", "network.ring_criterion=bogus",
                 "--set", "network.ring_max_size=0",
                 "--set", "exafs.weighting=bogus", "--absorber", "Si",
                 "--timestep-fs", "1", "--frame-interval-ps", "1",
                 "--out", tmp_path / "r")
    assert result.returncode == cli.EXIT_USAGE, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    for name in ("scattering.radiations", "scattering.dr_ang", "order.q_bin",
                 "network.ring_criterion", "network.ring_max_size",
                 "exafs.weighting", "timestep_fs"):
        assert name in result.stderr, name
    assert not (tmp_path / "r").exists()


def test_frames_the_model_does_not_hold_exit_2(model, tmp_path):
    for frames in ("50:60", "3:3", "7"):
        result = run("analyse", model, "--formers", "Si", "--only", "glass",
                     "--frames", frames, "--out", tmp_path / "r", "--quiet")
        assert result.returncode == cli.EXIT_USAGE, (frames, result.stderr)
        assert "frame" in result.stderr
    assert not (tmp_path / "r").exists()


def test_without_a_time_axis_the_analyses_of_time_are_left_out(model,
                                                               tmp_path):
    """The extended XYZ states no frame times: without --only and without a
    time axis, the analyses of time used to run and fail at the end (exit
    1); they are now left out, naming what they lack, and the run exits 0."""
    result = run("analyse", model, "--formers", "Si", "--out",
                 tmp_path / "r", "--quiet")
    assert result.returncode == cli.EXIT_OK, result.stdout + result.stderr
    for name in ("msd", "vacf", "kinetic-temperature", "bond-lifetimes"):
        assert f"  {name} " not in result.stdout, name
    explicit = run("analyse", model, "--formers", "Si", "--only",
                   "glass,msd", "--out", tmp_path / "s", "--quiet")
    assert explicit.returncode == cli.EXIT_USAGE
    assert "timestep_fs or frame_interval_ps" in explicit.stderr
    assert not (tmp_path / "s").exists()


def test_a_failed_output_does_not_lose_the_others(model, tmp_path, capsys):
    """A workbook that cannot be written (a locked file, here a directory
    in its place) used to stop the export before the CSV directory: every
    target is now tried, CSV first; with none written, the results go to a
    CSV directory beside the first target."""
    locked = tmp_path / "locked.xlsx"
    locked.mkdir()
    code = cli.main(["analyse", str(model), "--formers", "Si", "--only",
                     "voronoi", "--frames", "0:2", "--out", str(locked),
                     "--out", str(tmp_path / "csvdir"), "--quiet"])
    assert code == cli.EXIT_WRITE
    assert (tmp_path / "csvdir" / "index.csv").exists()
    assert "locked.xlsx: could not be written" in capsys.readouterr().err
    code = cli.main(["analyse", str(model), "--formers", "Si", "--only",
                     "voronoi", "--frames", "0:2", "--out", str(locked),
                     "--quiet"])
    assert code == cli.EXIT_WRITE
    assert (tmp_path / "locked_csv" / "index.csv").exists()
    assert "written instead" in capsys.readouterr().err
    # a CSV directory that cannot be made (under a file), tried first: the
    # workbook after it is still written
    blocker = tmp_path / "a_file"
    blocker.write_text("not a directory", encoding="utf-8")
    code = cli.main(["analyse", str(model), "--formers", "Si", "--only",
                     "voronoi", "--frames", "0:2", "--out",
                     str(tmp_path / "kept.xlsx"), "--out",
                     str(blocker / "inside"), "--quiet"])
    assert code == cli.EXIT_WRITE
    assert (tmp_path / "kept.xlsx").exists()


def test_python_m_facet_md_imports_no_qt():
    code = ("import runpy, sys\n"
            "sys.argv = ['facet.md', 'analyses']\n"
            "try:\n"
            "    runpy.run_module('facet.md', run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
            "print('QT' if [m for m in sys.modules if m.startswith('PySide6')]"
            " else 'CLEAN')\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True, encoding="utf-8",
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("CLEAN")
