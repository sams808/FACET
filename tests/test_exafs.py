"""The EXAFS module: shells, resolution, and FEFF read back.

The checks that carry weight here are against FEFF itself. One runs a FEFF
executable if the machine has one and skips otherwise, because a FEFF input that
FEFF refuses is the failure that matters and no amount of internal consistency
finds it -- the exported file was refused outright until the absorber's own
element was given a second potential.

The other compares FACET's path sum against FEFF's own ``chi.dat``: the same
paths added up twice, once by FEFF's ff2x module and once here, which is two
unrelated routes to one curve.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from facet.core import cif, exafs, exporters

BI_CIF_DIR = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi")
SAMPLE = BI_CIF_DIR / "cifs" / "1526458_Bi2O3.cif"
LARCH_FEFF = Path(r"C:\Users\samso\AppData\Local\Programs\Python\Python311"
                  r"\Lib\site-packages\larch\bin\win64\feff8l.bat")


@pytest.fixture(scope="module")
def structure():
    if not SAMPLE.is_file():
        pytest.skip("the sample structure is not present")
    return cif.read(str(SAMPLE))


@pytest.fixture(scope="module")
def bismuth_site(structure):
    return next(i for i, s in enumerate(structure.sites) if s.element == "Bi")


@pytest.fixture(scope="module")
def table(structure, bismuth_site):
    return exafs.shell_table(structure, bismuth_site, r_max=6.0,
                             temperature=300.0, einstein_temperature=500.0)


# --- the constants ----------------------------------------------------------

def test_the_einstein_constant_is_what_its_derivation_says():
    """hbar^2 / (2 k_B u), in A^2 K amu. Recomputed here from SI constants."""
    hbar = 1.054571817e-34
    k_b = 1.380649e-23
    u = 1.66053906892e-27
    expected = hbar * hbar / (2.0 * k_b * u) * 1e20
    assert exafs.EINSTEIN_C == pytest.approx(expected, rel=2e-5)


def test_the_k_conversion_constant_is_what_its_derivation_says():
    """k = sqrt(2 m_e E)/hbar, with E in eV and k in inverse angstrom."""
    hbar = 1.054571817e-34
    m_e = 9.1093837015e-31
    ev = 1.602176634e-19
    expected = math.sqrt(2.0 * m_e * ev) / hbar * 1e-10
    assert exafs.K_PER_SQRT_EV == pytest.approx(expected, rel=1e-9)


def test_k_and_energy_round_trip():
    k = np.array([2.0, 5.5, 14.0, 20.0])
    back = exafs.k_from_energy(exafs.energy_from_k(k, 13419.0), 13419.0)
    assert back == pytest.approx(k, abs=1e-10)
    # below the edge there is no k
    assert np.isnan(exafs.k_from_energy(13000.0, 13419.0))


def test_the_einstein_sigma2_has_the_right_limits():
    mu = exafs.reduced_mass("Bi", "O")
    assert mu == pytest.approx(14.86, abs=0.05)

    # at T = 0 it is the zero-point value hbar/(2 mu omega)
    zero_point = exafs.EINSTEIN_C / (500.0 * mu)
    assert exafs.einstein_sigma2(500.0, mu, 0.0) == pytest.approx(zero_point)
    assert exafs.einstein_sigma2(500.0, mu, 1e-9) == pytest.approx(zero_point)

    # at high T it becomes classical: sigma^2 -> 2 C T / (theta^2 mu)
    for T in (2400.0, 4800.0):
        classical = 2.0 * exafs.EINSTEIN_C * T / (500.0 ** 2 * mu)
        assert exafs.einstein_sigma2(500.0, mu, T) == pytest.approx(
            classical, rel=0.01)

    # it increases with temperature and decreases with mass and with theta
    assert (exafs.einstein_sigma2(500.0, mu, 300.0)
            < exafs.einstein_sigma2(500.0, mu, 600.0))
    assert (exafs.einstein_sigma2(500.0, mu, 300.0)
            > exafs.einstein_sigma2(700.0, mu, 300.0))
    assert (exafs.einstein_sigma2(500.0, mu, 300.0)
            > exafs.einstein_sigma2(500.0, 2 * mu, 300.0))

    # and the order of magnitude a Bi-O first-shell fit returns
    assert 0.003 < exafs.einstein_sigma2(500.0, mu, 300.0) < 0.007


def test_a_nonsense_einstein_input_gives_nan():
    assert np.isnan(exafs.einstein_sigma2(0.0, 14.0, 300.0))
    assert np.isnan(exafs.einstein_sigma2(500.0, 0.0, 300.0))
    assert np.isnan(exafs.reduced_mass("Bi", "Xx"))


# --- the shell table --------------------------------------------------------

def test_no_shell_is_wider_than_the_tolerance(structure, bismuth_site):
    """The chaining fault: a shell that walks outwards one contact at a time.

    Merging against a running mean let three Bi contacts at 3.5555, 3.5787 and
    3.6145 A land in one 0.05 A shell, although the outer two are 0.059 A apart.
    """
    for tolerance in (0.01, 0.05, 0.12):
        table = exafs.shell_table(structure, bismuth_site, r_max=6.0,
                                  tolerance=tolerance)
        for shell in table.shells:
            assert shell.spread <= tolerance + 1e-9, (tolerance, shell)


def test_it_splits_shells_the_old_grouping_chained_together(structure,
                                                            bismuth_site):
    from facet.core import utilities

    old = utilities.radial_shells(structure, bismuth_site, 6.0, 0.05)
    new = exafs.shell_table(structure, bismuth_site, r_max=6.0, tolerance=0.05)
    assert len(new) >= len(old)


def test_every_contact_is_in_exactly_one_shell(structure, bismuth_site):
    from facet.core.neighbors import NeighborFinder

    contacts = NeighborFinder(structure, rmax=6.0).contacts_for_site(
        bismuth_site)
    table = exafs.shell_table(structure, bismuth_site, r_max=6.0)
    assert sum(s.count for s in table.shells) == len(contacts)


def test_the_shells_are_sorted_and_carry_their_sites(table):
    distances = [s.r_mean for s in table.shells]
    assert distances == sorted(distances)
    for shell in table.shells:
        assert shell.labels, "a shell with no contributing site"
        assert shell.z > 0
        assert shell.r_min <= shell.r_mean <= shell.r_max


def test_occupancy_is_a_column_and_not_silently_folded_in():
    """FEFF has no partial occupancy, so the difference has to be visible."""
    import os
    import tempfile

    rows = ["Bi1 Bi 0.0 0.0 0.0 1.0", "O1 O 0.25 0.25 0.25 0.5"]
    header = ["data_t", "_cell_length_a 5.0", "_cell_length_b 5.0",
              "_cell_length_c 5.0", "_cell_angle_alpha 90",
              "_cell_angle_beta 90", "_cell_angle_gamma 90",
              "_symmetry_space_group_name_H-M 'P 1'", "loop_",
              "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy"]
    path = os.path.join(tempfile.mkdtemp(), "t.cif")
    Path(path).write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    st = cif.read(path)
    table = exafs.shell_table(st, 0, r_max=5.0)
    oxygen = [s for s in table.shells if s.element == "O"]
    assert oxygen
    assert oxygen[0].count_occupancy == pytest.approx(0.5 * oxygen[0].count)
    assert any("partly occupied" in note for note in table.notes)


def test_the_notes_state_where_sigma2_came_from(table):
    joined = " ".join(table.notes)
    assert "uncorrelated" in joined
    assert "upper bound" in joined
    assert "not a measurement" in joined


def test_the_notes_never_call_a_number_good_or_bad(table):
    import re

    text = " ".join(table.notes).lower()
    forbidden = {"good", "bad", "poor", "excellent", "correct", "incorrect",
                 "wrong", "reliable", "trustworthy"}
    assert not (set(re.findall(r"[a-z]+", text)) & forbidden)


def test_an_empty_neighbourhood_is_reported(structure, bismuth_site):
    table = exafs.shell_table(structure, bismuth_site, r_max=0.5)
    assert len(table) == 0
    assert table.notes


# --- resolution -------------------------------------------------------------

def test_the_resolution_is_the_closed_form(table):
    res = exafs.resolution(table, k_min=3.0, k_max=14.0, r_min=1.0, r_max=4.5)
    assert res.delta_r == pytest.approx(math.pi / (2.0 * 11.0), rel=1e-12)
    assert res.n_independent == pytest.approx(
        2.0 * 11.0 * 3.5 / math.pi, rel=1e-12)


def test_a_wider_k_range_resolves_more(table):
    narrow = exafs.resolution(table, k_min=3.0, k_max=8.0)
    wide = exafs.resolution(table, k_min=3.0, k_max=18.0)
    assert wide.delta_r < narrow.delta_r
    assert wide.n_independent > narrow.n_independent
    assert len(wide.unresolved) <= len(narrow.unresolved)


def test_every_unresolved_pair_really_is_closer_than_delta_r(table):
    res = exafs.resolution(table, k_min=3.0, k_max=14.0, r_min=1.0, r_max=4.5)
    for a, b, gap in res.unresolved:
        assert gap < res.delta_r
        assert gap == pytest.approx(
            table.shells[b].r_mean - table.shells[a].r_mean, abs=1e-9)
    # and the ones it does not list are not closer than dR
    inside = [i for i, s in enumerate(table.shells)
              if 1.0 <= s.r_mean <= 4.5]
    listed = {(a, b) for a, b, _ in res.unresolved}
    for a, b in zip(inside, inside[1:]):
        if (a, b) not in listed:
            assert (table.shells[b].r_mean
                    - table.shells[a].r_mean) >= res.delta_r


def test_the_parameter_count_is_against_n_idp(table):
    res = exafs.resolution(table, k_min=3.0, k_max=14.0, r_min=1.0, r_max=4.5,
                           parameters_per_shell=3)
    inside = sum(1 for s in table.shells if 1.0 <= s.r_mean <= 4.5)
    assert res.n_parameters == 3 * inside
    assert any("independent points" in note for note in res.notes)


def test_an_impossible_range_is_refused(table):
    with pytest.raises(ValueError):
        exafs.resolution(table, k_min=14.0, k_max=3.0)
    with pytest.raises(ValueError):
        exafs.resolution(table, r_min=5.0, r_max=1.0)


# --- the FEFF input ---------------------------------------------------------

def test_the_absorber_is_the_only_atom_with_potential_zero(structure,
                                                           bismuth_site,
                                                           tmp_path):
    """The fault that made FEFF8L refuse the file outright."""
    path = exporters.write_feff(structure, bismuth_site, tmp_path / "feff.inp",
                               rmax=6.0)
    text = Path(path).read_text(encoding="utf-8")
    atoms = text.split("ATOMS")[1]
    zeros = [line for line in atoms.splitlines()
             if line.strip() and not line.strip().startswith("*")
             and len(line.split()) >= 4 and line.split()[3] == "0"]
    assert len(zeros) == 1, "only the absorber may have ipot 0"

    # and the absorber's element has a second potential for the other atoms
    potentials = text.split("POTENTIALS")[1].split("ATOMS")[0]
    bismuth = [line for line in potentials.splitlines()
               if line.strip() and not line.strip().startswith("*")
               and line.split()[-1] == "Bi"]
    assert len(bismuth) == 2, "the absorber's element needs its own scatterer"


def test_no_line_of_the_input_carries_an_extra_column(structure,
                                                      bismuth_site, tmp_path):
    """FEFF reads POTENTIALS and ATOMS positionally.

    A trailing word is read as the next number, and the file is refused -- which
    is what happened when the roles were written as a fourth column.
    """
    path = exporters.write_feff(structure, bismuth_site, tmp_path / "f.inp")
    text = Path(path).read_text(encoding="utf-8")
    potentials = text.split("POTENTIALS")[1].split("ATOMS")[0]
    atoms = text.split("ATOMS")[1].split("END")[0]
    for block, body, fields in (("POTENTIALS", potentials, 3),
                                ("ATOMS", atoms, 6)):
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("*"):
                continue
            assert len(stripped.split()) <= fields, (block, line)


def test_the_path_radius_is_inside_the_cluster(structure, bismuth_site,
                                               tmp_path):
    path = exporters.write_feff(structure, bismuth_site, tmp_path / "f.inp",
                               rmax=7.0)
    text = Path(path).read_text(encoding="utf-8")
    rpath = float(next(line for line in text.splitlines()
                       if line.startswith("RPATH")).split()[1])
    assert rpath < 7.0


def test_the_input_asks_for_the_path_files(structure, bismuth_site, tmp_path):
    """PRINT 1 0 0 0 0 0 writes no feffNNNN.dat, which is what a fit needs."""
    path = exporters.write_feff(structure, bismuth_site, tmp_path / "f.inp")
    text = Path(path).read_text(encoding="utf-8")
    print_line = next(line for line in text.splitlines()
                      if line.startswith("PRINT"))
    assert print_line.split()[1:] == ["0", "0", "0", "0", "0", "3"]


def test_partial_occupancy_is_stated_in_the_header():
    """FEFF cannot represent it, so it must not be passed over silently."""
    import os
    import tempfile

    rows = ["Bi1 Bi 0.0 0.0 0.0 1.0", "O1 O 0.25 0.25 0.25 0.25"]
    header = ["data_t", "_cell_length_a 5.0", "_cell_length_b 5.0",
              "_cell_length_c 5.0", "_cell_angle_alpha 90",
              "_cell_angle_beta 90", "_cell_angle_gamma 90",
              "_symmetry_space_group_name_H-M 'P 1'", "loop_",
              "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy"]
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "t.cif")
    Path(path).write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    st = cif.read(path)
    out = exporters.write_feff(st, 0, os.path.join(directory, "feff.inp"),
                              rmax=5.0)
    text = Path(out).read_text(encoding="utf-8")
    assert "PARTIAL OCCUPANCY" in text
    assert "0.2500 occupied" in text


# --- FEFF read back ---------------------------------------------------------

FILES_DAT = """ test  Bi1                                    Feff8L (EXAFS)       0.1
 PATH  Rmax= 5.000,  Keep_limit= 0.00
 -----------------------------------------------------------------------
    file        sig2   amp ratio    deg    nlegs  r effective
 feff0001.dat 0.00000   100.000     1.000     2   2.1194
 feff0002.dat 0.00000    91.140     2.000     2   2.2001
 feff0031.dat 0.00000    12.340     4.000     3   4.5678
"""

FEFF_PATH = """ test  Bi1                                    Feff8L (EXAFS)       0.1
 Path    1      icalc       2
 -----------------------------------------------------------------------
   2   1.000   2.1194    2.6314    2.34296 nleg, deg, reff, rnrmav(bohr), edge
        x         y         z   pot at#
    -0.0000   -0.0000   -0.0000  0  83 Bi       absorbing atom
     0.8487   -1.7294    0.8836  1   8 O
    k   real[2*phc]   mag[feff]  phase[feff] red factor   lambda     real[p]@#
  1.000  1.2000E+01  3.0000E-01 -1.4000E+01  9.500E-01  6.0000E+00  2.5000E+00
  2.000  1.2100E+01  4.0000E-01 -1.5000E+01  9.600E-01  7.0000E+00  3.0000E+00
  3.000  1.2200E+01  5.0000E-01 -1.6000E+01  9.700E-01  8.0000E+00  3.5000E+00
"""


def test_files_dat_columns_are_not_counted_past(tmp_path):
    """The sig2 column sits between the file name and the amplitude ratio.

    Reading past it gave a list of hundred-legged paths all at 2.0 A -- numbers,
    and all of them wrong.
    """
    path = tmp_path / "files.dat"
    path.write_text(FILES_DAT, encoding="utf-8")
    paths = exafs.read_files_dat(path)
    assert len(paths) == 3
    first = paths[0]
    assert first.filename == "feff0001.dat"
    assert first.index == 1
    assert first.sigma2_feff == pytest.approx(0.0)
    assert first.amplitude_ratio == pytest.approx(100.0)
    assert first.degeneracy == pytest.approx(1.0)
    assert first.n_legs == 2
    assert first.r_effective == pytest.approx(2.1194)
    assert first.is_single_scattering

    triple = [p for p in paths if p.filename == "feff0031.dat"][0]
    assert triple.n_legs == 3
    assert triple.degeneracy == pytest.approx(4.0)
    assert triple.r_effective == pytest.approx(4.5678)
    assert not triple.is_single_scattering


def test_a_path_file_is_read_with_its_amplitude_and_phase(tmp_path):
    """The nleg line is data with its own column names written after it."""
    path = tmp_path / "feff0001.dat"
    path.write_text(FEFF_PATH, encoding="utf-8")
    p = exafs.read_feff_path(path)
    assert p.n_legs == 2
    assert p.degeneracy == pytest.approx(1.0)
    assert p.r_effective == pytest.approx(2.1194)
    assert p.k == pytest.approx([1.0, 2.0, 3.0])
    # the magnitude is red_factor * mag[feff]
    assert p.magnitude == pytest.approx([0.95 * 0.3, 0.96 * 0.4, 0.97 * 0.5])
    # and the phase is phase[feff] + real[2*phc]
    assert p.phase == pytest.approx([-2.0, -2.9, -3.8])
    assert p.lambda_k == pytest.approx([6.0, 7.0, 8.0])
    # the two atoms of the path, with their distances from the absorber
    assert len(p.atoms) == 2
    assert p.atoms[0][0] == "Bi" and p.atoms[0][1] == pytest.approx(0.0)
    assert p.atoms[1][0] == "O"
    assert p.atoms[1][1] == pytest.approx(2.1194, abs=1e-3)


def test_a_directory_merges_the_listing_with_the_path_files(tmp_path):
    (tmp_path / "files.dat").write_text(FILES_DAT, encoding="utf-8")
    (tmp_path / "feff0001.dat").write_text(FEFF_PATH, encoding="utf-8")
    paths = exafs.read_feff_directory(tmp_path)
    assert len(paths) == 3                    # two listed only, one with data
    detailed = [p for p in paths if p.k is not None]
    assert len(detailed) == 1
    assert detailed[0].amplitude_ratio == pytest.approx(100.0)


def test_chi_is_refused_without_feff_amplitudes():
    """The one thing this module declines to invent."""
    listed = exafs.read_files_dat.__doc__ and []
    chi = exafs.chi_from_paths([exafs.FeffPath(
        index=1, filename="feff0001.dat", n_legs=2, degeneracy=1.0,
        r_effective=2.12)])
    assert not len(chi.k)
    assert any("0.4 A" in note for note in chi.notes)
    assert any("does not compute chi(k) from the structure" in note
               for note in chi.notes)


def test_chi_scales_with_s02_and_damps_with_sigma2(tmp_path):
    (tmp_path / "files.dat").write_text(FILES_DAT, encoding="utf-8")
    (tmp_path / "feff0001.dat").write_text(FEFF_PATH, encoding="utf-8")
    paths = exafs.read_feff_directory(tmp_path)
    k = np.array([1.0, 2.0, 3.0])
    plain = exafs.chi_from_paths(paths, s02=1.0, k=k)
    doubled = exafs.chi_from_paths(paths, s02=2.0, k=k)
    assert doubled.chi == pytest.approx(2.0 * plain.chi)

    damped = exafs.chi_from_paths(paths, s02=1.0, k=k, sigma2={1: 0.01})
    assert damped.chi == pytest.approx(plain.chi * np.exp(-2 * 0.01 * k * k))
    assert np.abs(damped.chi[-1]) < np.abs(plain.chi[-1])


def test_a_path_matches_its_nearest_shell_not_the_first_one(table):
    """Taking the first within tolerance matched 2.200 A to a 2.119 A shell."""
    paths = [exafs.FeffPath(index=2, filename="feff0002.dat", n_legs=2,
                            degeneracy=1.0, r_effective=2.2001)]
    rows = exafs.compare_to_shells(paths, table)
    assert rows[0]["shell"] is not None
    matched = table.shells[rows[0]["shell"]]
    gaps = [abs(s.r_mean - 2.2001) for s in table.shells]
    assert abs(matched.r_mean - 2.2001) == pytest.approx(min(gaps))


def test_a_multiple_scattering_path_matches_no_shell(table):
    paths = [exafs.FeffPath(index=31, filename="feff0031.dat", n_legs=3,
                            degeneracy=4.0, r_effective=table.shells[0].r_mean)]
    rows = exafs.compare_to_shells(paths, table)
    assert rows[0]["shell"] is None
    assert "leg" in rows[0]["kind"]


def test_chi_and_xmu_readers_skip_comments(tmp_path):
    (tmp_path / "chi.dat").write_text(
        "# a comment\n* another\n 1.0  0.01\n 2.0 -0.02\n", encoding="utf-8")
    k, chi = exafs.read_chi(tmp_path / "chi.dat")
    assert k == pytest.approx([1.0, 2.0])
    assert chi == pytest.approx([0.01, -0.02])

    (tmp_path / "xmu.dat").write_text(
        "# header\n 100.0 -10.0 0.0 1.0 1.0 0.0\n"
        " 200.0  90.0 4.9 1.2 1.1 0.09\n", encoding="utf-8")
    xmu = exafs.read_xmu(tmp_path / "xmu.dat")
    assert set(xmu) == {"energy", "e_minus_e0", "k", "mu", "mu0", "chi"}
    assert xmu["k"] == pytest.approx([0.0, 4.9])


# --- against FEFF itself ----------------------------------------------------

def _feff_executable():
    if LARCH_FEFF.is_file():
        return LARCH_FEFF
    found = exafs.find_feff_executable()
    return found


@pytest.fixture(scope="module")
def feff_run(structure, bismuth_site, tmp_path_factory):
    """Run a real FEFF on FACET's exported input, or skip."""
    executable = _feff_executable()
    if executable is None:
        pytest.skip("no FEFF executable on this machine")
    directory = tmp_path_factory.mktemp("feff")
    exporters.write_feff(structure, bismuth_site, directory / "feff.inp",
                         rmax=6.0, edge="L3")
    try:
        subprocess.run([str(executable)], cwd=str(directory),
                       capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"FEFF did not run: {exc}")
    if not (directory / "files.dat").is_file():
        log = (directory / "log.dat")
        tail = log.read_text(errors="replace")[-600:] if log.is_file() else ""
        pytest.fail(f"FEFF ran but wrote no files.dat.\n{tail}")
    return directory


def test_feff_accepts_the_exported_input(feff_run):
    """The check no internal consistency can make: does FEFF run the file?"""
    paths = exafs.read_feff_directory(feff_run)
    assert len(paths) > 10
    assert any(p.is_single_scattering for p in paths)
    assert any(p.k is not None for p in paths)
    assert (feff_run / "chi.dat").is_file()
    assert (feff_run / "xmu.dat").is_file()


def test_feffs_own_distances_match_facets_shells(feff_run, table):
    """FEFF computed these from the cluster FACET wrote, so they must agree.

    A cross-check on the neighbour search: every single-scattering path FEFF
    reports must land on a FACET shell, to the precision FEFF prints.
    """
    paths = exafs.read_feff_directory(feff_run)
    single = [p for p in paths if p.is_single_scattering]
    assert single
    for p in single:
        gaps = [abs(s.r_mean - p.r_effective) for s in table.shells]
        assert min(gaps) < 0.06, (p.filename, p.r_effective)


def test_facets_path_sum_reproduces_feffs_own_chi(feff_run):
    """The same paths added up twice: by FEFF's ff2x, and here.

    FEFF's chi.dat is its own sum over the same feffNNNN.dat files at S02 = 1
    and sigma^2 = 0, so reproducing it is a check on the path-sum formula --
    the 1/(k R^2), the mean-free-path damping, the 2kR + phi argument and the
    degeneracies all at once.
    """
    paths = exafs.read_feff_directory(feff_run)
    k, feff_chi = exafs.read_chi(feff_run / "chi.dat")
    assert len(k) > 100
    mine = exafs.chi_from_paths(paths, s02=1.0, k=k)
    band = (k > 3.0) & (k < 14.0)
    scale = float(np.abs(feff_chi[band]).max())
    difference = float(np.abs(mine.chi[band] - feff_chi[band]).max())
    assert difference < 0.06 * scale, (difference, scale)
    correlation = float(np.corrcoef(mine.chi[band], feff_chi[band])[0, 1])
    assert correlation > 0.99


def test_a_heavy_scatterer_can_rival_the_first_shell(feff_run, table):
    """Why a count-against-distance histogram misleads about EXAFS.

    In Bi2O3 the Bi-Bi path near 3.9 A carries a larger amplitude ratio than
    most of the oxygen contacts inside it -- the geometry cannot say that, and
    FEFF can.
    """
    paths = exafs.read_feff_directory(feff_run)
    heavy = [p for p in paths if p.is_single_scattering
             and 3.7 < p.r_effective < 4.1
             and np.isfinite(p.amplitude_ratio)]
    if not heavy:
        pytest.skip("no path in that range in this calculation")
    strongest_heavy = max(abs(p.amplitude_ratio) for p in heavy)
    outer_oxygen = [p for p in paths if p.is_single_scattering
                    and 2.5 < p.r_effective < 3.5
                    and np.isfinite(p.amplitude_ratio)]
    assert outer_oxygen
    assert strongest_heavy > min(abs(p.amplitude_ratio) for p in outer_oxygen)
