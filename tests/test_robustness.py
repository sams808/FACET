"""What the program does when it is handed something it was not designed for.

The standard is not that everything works -- some inputs have no sensible answer --
but that FACET says what is wrong in its own terms, rather than raising from three
layers down, returning a number that looks fine, or not finishing.

Every case here was found by feeding the program malformed, degenerate or
deliberately awkward input and looking at how it failed.
"""
from __future__ import annotations

import math
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import (bv, bv_files, bv_report, cif, coordination, diffraction,
                        planes, polyhedra, quality, readers, volume)
from facet.core.structure import Atom, Cell, Site, Structure

BI_DIR = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi")
SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


def _write(text, suffix=".cif"):
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "probe" + suffix)
    Path(path).write_text(text, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def structure():
    if not SAMPLE.is_file():
        pytest.skip("the sample structure is not present")
    return cif.read(SAMPLE)


# ===========================================================================
# files that are not structures
# ===========================================================================

@pytest.mark.parametrize("label,text", [
    ("empty", ""),
    ("whitespace", "   \n\n  \t\n"),
    ("a comment only", "# just a comment\n"),
])
def test_a_file_with_no_data_block_is_reported_not_raised(label, text):
    """Dropping an empty file on the window is an ordinary accident.

    It used to reach gemmi's sole_block() and come back as
    ``IndexError: invalid vector subscript`` -- a subscript error from three
    layers down, with nothing in it about files or CIFs.
    """
    with pytest.raises(ValueError, match="no CIF data block"):
        cif.read(_write(text))


def test_asking_for_a_block_that_is_not_there_says_what_is():
    text = ("data_first\n_cell_length_a 5\n_cell_length_b 5\n"
            "_cell_length_c 5\n_cell_angle_alpha 90\n_cell_angle_beta 90\n"
            "_cell_angle_gamma 90\nloop_\n_atom_site_label\n"
            "_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
            "_atom_site_fract_z\nC1 C 0 0 0\n")
    with pytest.raises(KeyError, match="first"):
        cif.read(_write(text), block="second")


@pytest.mark.parametrize("label,text", [
    ("prose", "this is a shopping list\nbread\nmilk\n"),
    ("binary", "\x00\x01\x02" * 100),
    ("a cell with no atoms",
     "data_x\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n"
     "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"),
    ("a zero-length axis",
     "data_x\n_cell_length_a 0\n_cell_length_b 5\n_cell_length_c 5\n"
     "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
     "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
     "_atom_site_fract_y\n_atom_site_fract_z\nC1 C 0 0 0\n"),
    ("coordinates that are not numbers",
     "data_x\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n"
     "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
     "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
     "_atom_site_fract_y\n_atom_site_fract_z\nC1 C abc def ghi\n"),
])
def test_an_unreadable_file_gives_a_reported_error(label, text):
    """A ValueError with something in it, not an arbitrary exception type."""
    with pytest.raises((ValueError, KeyError)) as caught:
        cif.read(_write(text))
    assert str(caught.value).strip(), "raised with no message"


@pytest.mark.parametrize("label,text", [
    ("atoms with no cell",
     "data_x\nloop_\n_atom_site_label\n_atom_site_type_symbol\n"
     "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nC1 C 0 0 0\n"),
    ("an unknown element",
     "data_x\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n"
     "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
     "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
     "_atom_site_fract_y\n_atom_site_fract_z\nQq1 Qq 0 0 0\n"),
    ("coordinates far outside the cell",
     "data_x\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n"
     "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
     "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
     "_atom_site_fract_y\n_atom_site_fract_z\nC1 C 1e6 -1e6 5.5\n"),
    ("two atoms at the same place",
     "data_x\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n"
     "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
     "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
     "_atom_site_fract_y\n_atom_site_fract_z\nC1 C 0 0 0\nC2 C 0 0 0\n"),
    ("an occupancy of zero",
     "data_x\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n"
     "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
     "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
     "_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n"
     "C1 C 0 0 0 0.0\n"),
    ("a nonsense symmetry operation",
     "data_x\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n"
     "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
     "loop_\n_symmetry_equiv_pos_as_xyz\n  'nonsense'\n  'x, y, z'\n"
     "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
     "_atom_site_fract_y\n_atom_site_fract_z\nC1 C 0.1 0.2 0.3\n"),
])
def test_an_odd_but_readable_file_gets_all_the_way_through(label, text):
    """Read, analysed and health-checked without an exception or a nan."""
    structure = cif.read(_write(text))
    results = coordination.analyse_structure(structure, bv.DEFAULT)
    report = quality.check(structure, results)
    assert isinstance(report.findings, list)
    for result in results:
        assert not math.isinf(result.bvs) if result.bvs == result.bvs else True
        assert result.cn_valence >= 0


# ===========================================================================
# extreme arguments
# ===========================================================================

def test_a_non_positive_valence_threshold_is_refused_in_its_own_terms(structure):
    """It used to surface as ``math domain error`` from inside a logarithm."""
    for threshold in (0.0, -0.1):
        with pytest.raises(ValueError, match="greater than zero"):
            coordination.analyse_structure(structure, bv.DEFAULT,
                                          v_bond=threshold, v_list=threshold)
    with pytest.raises(ValueError, match="greater than zero"):
        bv.cutoff_for_valence(2.09, 0.0)
    with pytest.raises(ValueError, match="greater than zero"):
        bv.BVParam("Bi", 3, "O", -2, 2.09, 0.37, "test", True).distance_for(0.0)


def test_a_grid_that_would_not_finish_is_refused_with_the_numbers(structure):
    """A hang is not an acceptable answer to a small number in a spin box.

    The cost is points times anions times periodic images, not points alone: 2.9
    million points on a twenty-atom cell is six billion distance evaluations.
    """
    with pytest.raises(ValueError) as caught:
        volume.bond_valence_grid(structure, "Na", 1, resolution=0.01)
    message = str(caught.value)
    assert "terms" in message
    assert "would fit" in message
    # the estimate must describe the path that will actually run: the tree costs
    # points x anions x the fraction of the cell within rmax, not points x anions
    # x images, and estimating the wrong one refused requests that take a second
    assert "anion contacts each" in message

    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            volume.bond_valence_grid(structure, "Na", 1, resolution=bad)

    # and a reasonable spacing still works
    grid = volume.bond_valence_grid(structure, "Na", 1, resolution=0.5)
    assert grid.n_points > 100


def test_extreme_diffraction_settings_either_work_or_say_why(structure):
    assert diffraction.powder_pattern(structure, two_theta_max=179.9).reflections
    assert diffraction.powder_pattern(structure, two_theta_max=0.5) is not None
    assert diffraction.powder_pattern(structure, wavelength=0.1,
                                      two_theta_max=20.0) is not None
    assert diffraction.powder_pattern(structure, wavelength=20.0).reflections == []
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="wavelength"):
            diffraction.powder_pattern(structure, wavelength=bad)
    # a huge displacement parameter damps everything but must not break
    pattern = diffraction.powder_pattern(structure, b_iso=100.0)
    assert all(math.isfinite(r.intensity) for r in pattern.reflections)


def test_a_zero_index_plane_is_refused_and_a_huge_one_is_not(structure):
    with pytest.raises(ValueError, match="not a plane"):
        planes.normal_and_spacing(structure, 0, 0, 0)
    normal, d = planes.normal_and_spacing(structure, 99, -99, 99)
    assert math.isfinite(d) and d > 0
    assert abs(float(np.linalg.norm(normal)) - 1.0) < 1e-12


# ===========================================================================
# degenerate geometry
# ===========================================================================

def test_a_ligand_on_the_central_atom_is_reported_not_divided_by():
    """It produced a nan, a RuntimeWarning, and an angle variance that was
    neither a number nor an error."""
    cases = [np.zeros((1, 3)), np.zeros((6, 3)),
             np.array([[0, 0, 0], [2.2, 0, 0], [0, 2.2, 0], [0, 0, 2.2],
                       [-2.2, 0, 0], [0, -2.2, 0]], float)]
    for points in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            shape = polyhedra.shape(points)
        assert shape["d_min"] is None
        assert "undefined" in shape.get("note", "")
        for key in ("volume", "angle_variance", "quadratic_elongation"):
            assert shape[key] is None


def test_a_polyhedron_of_coplanar_ligands_still_gives_what_it_can():
    """No volume, because there is none; the distances are still distances."""
    angles = np.linspace(0, 2 * math.pi, 6, endpoint=False)
    flat = np.stack([2.2 * np.cos(angles), 2.2 * np.sin(angles),
                     np.zeros(6)], axis=1)
    shape = polyhedra.shape(flat)
    assert shape["d_min"] == pytest.approx(2.2)
    assert shape["volume"] is None
    assert shape["n"] == 6


def test_gap_split_of_too_few_distances():
    assert polyhedra.gap_split([]) == (0, 1.0)
    assert polyhedra.gap_split([2.0]) == (1, 1.0)
    assert polyhedra.gap_split([2.0, 2.4]) == (2, 1.0)


def test_phi_of_no_contacts_is_not_a_number_rather_than_an_error():
    magnitude, phi, vector = bv.phi_index(np.zeros((0, 3)), np.zeros(0))
    assert magnitude == 0.0
    assert math.isnan(phi)
    assert vector.shape == (3,)


def test_contours_of_an_empty_level_list():
    assert volume.contour_lines(np.zeros((8, 8)), (0, 1, 0, 1), []) == {}


# ===========================================================================
# structures built to be awkward
# ===========================================================================

def _made(sites):
    orth = np.eye(3) * 6.0
    cell = Cell(6.0, 6.0, 6.0, 90, 90, 90, orth=orth)
    atoms = [Atom(s.element, np.asarray(s.frac, float),
                  orth @ np.asarray(s.frac, float), i, s.label, s.occupancy)
             for i, s in enumerate(sites)]
    return Structure("made up", cell, sites, atoms)


AWKWARD = {
    "no sites at all": _made([]),
    "one atom with nothing to bond to": _made([Site("C1", "C", [0, 0, 0])]),
    "only anions": _made([Site("O1", "O", [0, 0, 0], ox=-2),
                          Site("O2", "O", [0.5, 0.5, 0.5], ox=-2)]),
    "only cations": _made([Site("Bi1", "Bi", [0, 0, 0], ox=3),
                           Site("Bi2", "Bi", [0.5, 0.5, 0.5], ox=3)]),
    "a cation with no parameter": _made([Site("Xx1", "Xx", [0, 0, 0], ox=3),
                                         Site("O1", "O", [0.3, 0, 0], ox=-2)]),
    "an impossible oxidation state": _made([Site("Bi1", "Bi", [0, 0, 0], ox=99),
                                            Site("O1", "O", [0.3, 0, 0],
                                                 ox=-2)]),
}


@pytest.mark.parametrize("label", sorted(AWKWARD))
def test_every_report_survives_an_awkward_structure(label):
    """None of these has a useful answer; none may raise something unexpected."""
    structure = AWKWARD[label]
    allowed = (ValueError, KeyError)

    def attempt(fn):
        try:
            return fn()
        except allowed as error:
            assert str(error).strip(), "raised with no message"
            return None

    results = attempt(lambda: coordination.analyse_structure(structure,
                                                            bv.DEFAULT))
    attempt(lambda: quality.check(structure, results))
    attempt(lambda: bv_report.cutoff_table(structure, bv.DEFAULT))
    attempt(lambda: bv_report.anion_sums(structure, bv.DEFAULT))
    attempt(lambda: diffraction.powder_pattern(structure, two_theta_max=40.0))
    if results:
        attempt(lambda: bv_report.threshold_scan(results))
        attempt(lambda: bv_report.charge_balance(structure, results,
                                                bv.DEFAULT))


@pytest.mark.parametrize("label", sorted(AWKWARD))
def test_a_scene_can_be_built_from_an_awkward_structure(label):
    from facet.gl.scene import build_scene

    structure = AWKWARD[label]
    try:
        scene = build_scene(structure)
    except ValueError as error:
        assert str(error).strip()
        return
    assert scene.n_atoms == len(structure.atoms)
    assert len(scene.atom_radius) == scene.n_atoms
    assert math.isfinite(scene.radius) and scene.radius > 0


@pytest.mark.parametrize("label", sorted(AWKWARD))
def test_the_exporters_survive_an_awkward_structure(label, tmp_path):
    from facet.core import exporters

    structure = AWKWARD[label]
    for name, writer in (("cif", exporters.write_cif),
                         ("vasp", exporters.write_poscar),
                         ("xyz", exporters.write_xyz),
                         ("vesta", exporters.write_vesta)):
        try:
            writer(structure, tmp_path / f"out_{label[:8]}.{name}")
        except (ValueError, KeyError) as error:
            assert str(error).strip(), f"{name} raised with no message"


# ===========================================================================
# parameter files that are not parameter files
# ===========================================================================

@pytest.mark.parametrize("label,text,suffix", [
    ("empty", "", ".txt"),
    ("prose", "the quick brown fox\n", ".txt"),
    ("headers only", "cation,charge,anion,charge,R0,b\n", ".csv"),
    ("a truncated cif loop", "data_x\nloop_\n_valence_param_atom_1\nBi\n",
     ".cif"),
    ("valid json, wrong shape", '{"hello": "world"}', ".json"),
    ("broken json", "{not json", ".json"),
])
def test_a_bad_parameter_file_is_refused_with_a_reason(label, text, suffix):
    with pytest.raises(ValueError) as caught:
        bv_files.load(_write(text, suffix))
    message = str(caught.value)
    assert message.strip()
    # and it names the file, so the message can be shown to someone
    assert "probe" in message, f"the message does not name the file: {message}"


@pytest.mark.parametrize("text", [
    "Bi 3 O -2 -5.0 0.37\n",           # a negative R0
    "Bi 3 O -2 1e9 0.37\n",            # an absurd R0
])
def test_an_impossible_r0_is_not_accepted(text):
    """A negative or enormous R0 would make every valence meaningless."""
    with pytest.raises(ValueError):
        bv_files.load(_write(text, ".txt"))


@pytest.mark.parametrize("b", ["0.0", "-0.37"])
def test_a_b_of_zero_or_less_is_refused_rather_than_replaced(b):
    """exp((R0 - d)/b) has no meaning at b = 0.

    Substituting the default quietly would be worse than refusing: it would
    change every valence the file produces, without saying so anywhere.
    """
    with pytest.raises(ValueError, match="no meaning"):
        bv_files.load(_write(f"Bi 3 O -2 2.09 {b}\n", ".txt"))


def test_a_valid_b_is_kept():
    params, report = bv_files.load(_write("Bi 3 O -2 2.09 0.40\n", ".txt"))
    assert params.get("Bi", 3, "O").b == pytest.approx(0.40)
    assert report.with_own_b == 1


# ===========================================================================
# every reader against every file
# ===========================================================================

def test_no_reader_raises_an_unexpected_exception_type():
    """A reader handed the wrong format must fail in a way callers can catch."""
    if not BI_DIR.is_dir():
        pytest.skip("the collection is not present")
    sample = sorted(BI_DIR.rglob("*.cif"))[:6]
    every = [readers.read, readers.read_poscar, readers.read_xyz,
             readers.read_vesta, readers.read_shelx, readers.read_pdb]
    for path in sample:
        for reader in every:
            try:
                reader(path)
            except (ValueError, OSError, KeyError, IndexError, TypeError,
                    UnicodeDecodeError):
                pass
            except Exception as error:      # noqa: BLE001 - that is the point
                pytest.fail(f"{reader.__name__} on {path.name} raised "
                            f"{type(error).__name__}: {error}")
