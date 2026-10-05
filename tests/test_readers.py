"""Reading structures that are not CIFs.

The strongest available check is a round trip: write a structure out in a
format and read it back, then require that the *coordination analysis* is
unchanged. Agreeing on atom count is easy; agreeing on every bond-valence sum
to three decimals is not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import cif, coordination, exporters, readers

CIFS = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs")
SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")

pytestmark = pytest.mark.skipif(not SAMPLE.exists(),
                                reason="sample structure not present")


@pytest.fixture(scope="module")
def reference():
    s = cif.read(SAMPLE)
    results = coordination.analyse_structure(s)
    fingerprint = sorted((r.element, r.cn_valence, round(r.bvs, 3))
                         for r in results)
    return s, fingerprint


def _fingerprint(structure) -> list:
    """Distinct site environments, so an expanded P1 file compares against a
    symmetric one."""
    return sorted({(r.element, r.cn_valence, round(r.bvs, 3))
                   for r in coordination.analyse_structure(structure)})


# --- symmetry operators ------------------------------------------------------

@pytest.mark.parametrize("text,rotation,translation", [
    ("x, y, z", np.eye(3), [0, 0, 0]),
    ("-x, -y, -z", -np.eye(3), [0, 0, 0]),
    ("-X, 0.5+Y, 0.5-Z",
     [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0, 0.5, 0.5]),
    ("-x, y+1/2, -z+1/2",
     [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0, 0.5, 0.5]),
    ("1/2-x, -y, 1/2+z",
     [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [0.5, 0, 0.5]),
    ("y, x, -z", [[0, 1, 0], [1, 0, 0], [0, 0, -1]], [0, 0, 0]),
])
def test_symmetry_operators_parse(text, rotation, translation):
    got = readers.parse_symop(text)
    assert got is not None
    assert np.allclose(got[0], rotation)
    assert np.allclose(got[1], translation)


def test_a_malformed_operator_returns_none():
    assert readers.parse_symop("x, y") is None


def test_expansion_collapses_a_special_position():
    """An atom at the origin is its own image under inversion."""
    rows = [("A", "Na", np.zeros(3), 1.0)]
    out = readers.expand_symmetry(rows, [], latt=1)     # P, centrosymmetric
    assert len(out) == 1


def test_expansion_doubles_a_general_position_under_inversion():
    rows = [("A", "Na", np.array([0.1, 0.2, 0.3]), 1.0)]
    out = readers.expand_symmetry(rows, [], latt=1)
    assert len(out) == 2


def test_body_centring_doubles_the_contents():
    rows = [("A", "Na", np.array([0.1, 0.2, 0.3]), 1.0)]
    out = readers.expand_symmetry(rows, [], latt=-2)    # I, non-centrosymmetric
    assert len(out) == 2
    assert any(np.allclose(frac, [0.6, 0.7, 0.8]) for _, _, frac, _, _ in out)


def test_face_centring_quadruples_the_contents():
    rows = [("A", "Na", np.array([0.1, 0.2, 0.3]), 1.0)]
    assert len(readers.expand_symmetry(rows, [], latt=-4)) == 4


def test_expansion_records_which_site_each_atom_came_from():
    rows = [("A", "Na", np.array([0.1, 0.2, 0.3]), 1.0),
            ("B", "Cl", np.array([0.4, 0.5, 0.6]), 1.0)]
    out = readers.expand_symmetry(rows, [], latt=1)
    assert {origin for *_, origin in out} == {0, 1}


# --- round trips -------------------------------------------------------------

def test_poscar_round_trips_the_analysis(reference, tmp_path):
    s, fingerprint = reference
    exporters.write_poscar(s, tmp_path / "POSCAR")
    back = readers.read_poscar(tmp_path / "POSCAR")
    assert back.n_atoms == s.n_atoms
    assert _fingerprint(back) == sorted(set(fingerprint))
    for a, b in ((s.cell.a, back.cell.a), (s.cell.beta, back.cell.beta)):
        assert a == pytest.approx(b, abs=1e-4)


def test_xyz_round_trips_the_analysis(reference, tmp_path):
    s, fingerprint = reference
    exporters.write_xyz(s, tmp_path / "out.xyz")
    back = readers.read_xyz(tmp_path / "out.xyz")
    assert back.n_atoms == s.n_atoms
    assert _fingerprint(back) == sorted(set(fingerprint))


def test_vesta_round_trips_the_analysis(reference, tmp_path):
    s, fingerprint = reference
    exporters.write_vesta(s, tmp_path / "out.vesta")
    back = readers.read_vesta(tmp_path / "out.vesta")
    assert back.n_atoms == s.n_atoms, "the section terminator may be read as an atom"
    assert _fingerprint(back) == sorted(set(fingerprint))


def test_xyz_without_a_lattice_invents_a_box_and_says_so(tmp_path):
    (tmp_path / "mol.xyz").write_text(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n",
        encoding="utf-8")
    s = readers.read_xyz(tmp_path / "mol.xyz")
    assert s.n_atoms == 3
    assert any("invented" in n for n in s.notes)
    assert s.cell.volume > 100


# --- SHELX -------------------------------------------------------------------

def _shelx_from(structure, path: Path) -> Path:
    order = list(dict.fromkeys(st.element for st in structure.sites))
    lines = [
        "TITL written by the test suite",
        "CELL 0.71073 %.4f %.4f %.4f %.3f %.3f %.3f" % (
            structure.cell.a, structure.cell.b, structure.cell.c,
            structure.cell.alpha, structure.cell.beta, structure.cell.gamma),
        "LATT 1",
        "SYMM -X, 0.5+Y, 0.5-Z",
        "SFAC " + " ".join(e.upper() for e in order),
        "FVAR 0.15",
    ]
    for st in structure.sites:
        lines.append("%-5s %d %11.6f %11.6f %11.6f  11.00000  0.01000" % (
            st.label, order.index(st.element) + 1,
            st.frac[0], st.frac[1], st.frac[2]))
    lines += ["HKLF 4", "END", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_shelx_applies_its_symmetry_and_reproduces_the_cif(reference, tmp_path):
    """SHELX gives the asymmetric unit. Reading it without applying LATT and
    SYMM would report a coordination number computed from a fraction of the
    structure -- silently, and wrongly."""
    s, fingerprint = reference
    back = readers.read_shelx(_shelx_from(s, tmp_path / "rt.res"))
    assert back.n_atoms == s.n_atoms
    assert _fingerprint(back) == sorted(set(fingerprint))


def test_shelx_reports_how_far_it_expanded(reference, tmp_path):
    s, _ = reference
    back = readers.read_shelx(_shelx_from(s, tmp_path / "rt.res"))
    assert any("expanded to" in n for n in back.notes)


def test_shelx_keeps_the_site_count_of_the_asymmetric_unit(reference, tmp_path):
    """Results must be reported per crystallographic site, not per atom."""
    s, _ = reference
    back = readers.read_shelx(_shelx_from(s, tmp_path / "rt.res"))
    assert back.n_sites == s.n_sites


def test_shelx_full_occupancy_encoding_is_not_flagged(reference, tmp_path):
    """11.00000 is the ordinary way SHELX writes 'fixed at full occupancy' and
    is not a refined free variable."""
    s, _ = reference
    back = readers.read_shelx(_shelx_from(s, tmp_path / "rt.res"))
    assert all(abs(site.occupancy - 1.0) < 1e-9 for site in back.sites)
    assert not any("free-variable" in n for n in back.notes)


def test_shelx_partial_occupancy_survives(tmp_path):
    (tmp_path / "part.res").write_text(
        "TITL x\nCELL 0.71 5.0 5.0 5.0 90 90 90\nLATT -1\nSFAC NA CL\n"
        "NA1 1 0.000000 0.000000 0.000000 10.50000 0.01\n"
        "CL1 2 0.500000 0.500000 0.500000 11.00000 0.01\nEND\n",
        encoding="utf-8")
    s = readers.read_shelx(tmp_path / "part.res")
    occupancies = sorted(site.occupancy for site in s.sites)
    assert occupancies == pytest.approx([0.5, 1.0])


def test_shelx_without_a_cell_is_refused(tmp_path):
    (tmp_path / "bad.res").write_text("TITL nothing\nSFAC NA\nEND\n",
                                      encoding="utf-8")
    with pytest.raises(readers.UnsupportedFormat):
        readers.read_shelx(tmp_path / "bad.res")


# --- CrystalMaker ------------------------------------------------------------

def test_crystalmaker_text_is_read(tmp_path):
    (tmp_path / "s.cmtx").write_text(
        "! comment\nTITL test\nCELL 5.0 5.0 5.0 90 90 90\n"
        "ATOM Na NA1 0.0 0.0 0.0\nATOM Cl CL1 0.5 0.5 0.5\n", encoding="utf-8")
    s = readers.read_crystalmaker(tmp_path / "s.cmtx")
    assert s.n_atoms == 2
    assert s.cell.a == pytest.approx(5.0)


def test_crystalmaker_binary_is_refused_with_a_way_forward(tmp_path):
    """Guessing at an undocumented binary layout could return a plausible but
    wrong structure, which is worse than refusing."""
    (tmp_path / "s.cmdf").write_bytes(b"\x00\x01binary nonsense")
    with pytest.raises(readers.UnsupportedFormat) as raised:
        readers.read_crystalmaker(tmp_path / "s.cmdf")
    assert "CIF" in str(raised.value)


# --- dispatch ----------------------------------------------------------------

def test_a_cif_still_goes_to_the_cif_reader(reference):
    s, fingerprint = reference
    assert _fingerprint(readers.read(SAMPLE)) == sorted(set(fingerprint))


def test_poscar_is_recognised_without_an_extension(reference, tmp_path):
    s, _ = reference
    exporters.write_poscar(s, tmp_path / "POSCAR")
    assert readers.read(tmp_path / "POSCAR").n_atoms == s.n_atoms


def test_a_numerically_titled_poscar_is_not_mistaken_for_an_xyz(reference,
                                                                tmp_path):
    """A POSCAR whose title is a COD number begins with a bare integer, which
    is also how an XYZ begins."""
    s, _ = reference
    exporters.write_poscar(s, tmp_path / "unnamed.txt")
    text = (tmp_path / "unnamed.txt").read_text(encoding="utf-8").splitlines()
    text[0] = "1526458"
    (tmp_path / "unnamed.txt").write_text("\n".join(text), encoding="utf-8")
    assert readers.read(tmp_path / "unnamed.txt").n_atoms == s.n_atoms


def test_a_vesta_file_is_sniffed_by_its_header(reference, tmp_path):
    s, _ = reference
    exporters.write_vesta(s, tmp_path / "unnamed.dat")
    assert readers.read(tmp_path / "unnamed.dat").n_atoms == s.n_atoms


def test_an_unrecognisable_file_names_what_is_supported(tmp_path):
    (tmp_path / "junk.dat").write_text("not a structure\n", encoding="utf-8")
    with pytest.raises(readers.UnsupportedFormat) as raised:
        readers.read(tmp_path / "junk.dat")
    assert "CIF" in str(raised.value)


def test_the_file_filter_covers_every_reader():
    for extension in (".cif", ".xyz", ".vesta", ".res", ".ins", ".pdb",
                      ".cmtx"):
        assert extension[1:] in readers.FILE_FILTER
