"""Faults the whole-application verification found, each pinned.

Every test here corresponds to something that was wrong and is now right. They
are kept together because they share a character: none of them was visible from
inside the program. The structures were self-consistent, the suite was green, and
the numbers looked reasonable. Each was caught by comparing against something
outside the code -- an independent implementation, a physical law, or the file's
own statement of what it contains.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import bv, bv_report, cif, coordination, polyhedra, quality

BI_DIR = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi")
FM3M = sample_cif("04-008-3529", "Bi0.92 O1.54 Si0.08 - 04-008-3529.cif")
EQUAL_SHELL = sample_cif("1010311", "1010311_Bi2O3.cif")


def _build(spacegroup, rows, a=5.6402, number=None, with_ops=None):
    """A CIF from text, with or without an explicit operation loop."""
    import os
    import tempfile

    lines = ["data_test", f"_cell_length_a {a}", f"_cell_length_b {a}",
             f"_cell_length_c {a}", "_cell_angle_alpha 90",
             "_cell_angle_beta 90", "_cell_angle_gamma 90",
             f"_symmetry_space_group_name_H-M '{spacegroup}'"]
    if number:
        lines.append(f"_symmetry_Int_Tables_number {number}")
    if with_ops:
        lines.append("loop_")
        lines.append("_symmetry_equiv_pos_as_xyz")
        lines.extend(f"  '{op}'" for op in with_ops)
    lines += ["loop_", "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy"]
    lines.extend(rows)
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "test.cif")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cif.read(path)


# ===========================================================================
# the symmetry expansion
# ===========================================================================

def test_a_32_fold_position_expands_to_32_atoms():
    """gemmi's small-structure expansion merged these; FACET must not.

    The ICDD entry for Bi0.92Si0.08O1.54 is Fm-3m with a = 5.542 A and oxygen on
    the 32-fold (0.266, 0.266, 0.266). gemmi's get_all_unit_cell_sites() returns
    8 of those 32, because the F-centred images land 0.032 in *fractional*
    coordinates away -- which in this small cell is 0.18 A, a real separation.
    Three quarters of the oxygen, and of the anion charge, went missing, and
    nothing inside the program noticed: the structure was self-consistent, just
    not the structure in the file.
    """
    if not FM3M.is_file():
        pytest.skip("the ICDD entry is not present")
    structure = cif.read(FM3M)

    oxygen = next(s for s in structure.sites if s.element == "O")
    assert oxygen.multiplicity == 32
    assert len([a for a in structure.atoms if a.element == "O"]) == 32
    assert len(structure.atoms) == 40          # 32 O + 4 Bi + 4 Si


def test_the_expanded_cell_matches_the_formula_the_file_states():
    """The check that catches an expansion fault: the file's own formula.

    It is the one statement in a CIF that does not come from the coordinates or
    the symmetry, which is exactly what makes it a test rather than a restatement.
    """
    if not FM3M.is_file():
        pytest.skip("the ICDD entry is not present")
    structure = cif.read(FM3M)
    composition = structure.composition()

    # Z = 4, and the file says Bi0.92 O1.54 Si0.08 per formula unit
    assert composition["Bi"] / 4 == pytest.approx(0.92, abs=0.01)
    assert composition["Si"] / 4 == pytest.approx(0.08, abs=0.01)
    assert composition["O"] / 4 == pytest.approx(1.54, abs=0.01)


def test_the_multiplicity_agrees_with_the_files_own_declaration():
    """Across the collection, where the file declares one.

    The only disagreement left is a file that declares multiplicity 1 for a
    general position in an eight-fold group, which cannot be right -- and FACET's
    own value of 8 is confirmed by the O:Bi ratio coming to exactly 1.5 for
    Bi2O3.
    """
    import gemmi

    if not BI_DIR.is_dir():
        pytest.skip("the collection is not present")

    checked = 0
    disagreements = []
    for path in sorted(BI_DIR.rglob("*.cif")):
        try:
            block = gemmi.cif.read(str(path)).sole_block()
            table = block.find(["_atom_site_label",
                                "_atom_site_symmetry_multiplicity"])
            declared = {}
            for row in table:
                try:
                    declared[row[0].strip("'\"")] = int(row[1])
                except (ValueError, IndexError):
                    continue
            if not declared:
                continue
            structure = cif.read(path)
        except Exception:
            continue
        checked += 1
        for site in structure.sites:
            want = declared.get(site.label)
            if want is not None and site.multiplicity != want:
                disagreements.append((path.name, site.label,
                                      site.multiplicity, want))

    assert checked > 20, "not enough files declare a multiplicity to test"
    offenders = {row[0] for row in disagreements}
    assert len(offenders) <= 1, f"unexpected disagreements: {disagreements}"


def test_the_space_group_is_found_by_number_and_by_symbol_spelling():
    """A file with no operation loop depends on this, and a failed lookup is
    invisible: the structure expands as P1 and simply has fewer atoms."""
    for spelling in ("F m -3 m", "Fm-3m", "F M -3 M"):
        structure = _build(spelling, ["Na1 Na 0 0 0 1.0",
                                      "Cl1 Cl 0.5 0.5 0.5 1.0"])
        assert len(structure.atoms) == 8, f"{spelling} expanded to P1"
    # and by number, even with a symbol nothing recognises
    structure = _build("not a symbol", ["Na1 Na 0 0 0 1.0",
                                        "Cl1 Cl 0.5 0.5 0.5 1.0"],
                       number=225)
    assert len(structure.atoms) == 8


def test_a_structure_records_how_it_was_expanded():
    """A silent fallback to P1 is the failure mode; the note makes it loud."""
    structure = _build("F m -3 m", ["Na1 Na 0 0 0 1.0",
                                    "Cl1 Cl 0.5 0.5 0.5 1.0"])
    note = next(n for n in structure.notes if "expansion" in n)
    assert "192" in note
    assert "F m -3 m" in note


def test_an_explicit_operation_loop_of_one_is_honoured():
    """A file listing only 'x, y, z' is saying its coordinates are the whole cell.

    Requiring more than one operation before trusting the loop made such a file
    fall through to its declared symbol and expand a second time -- quadrupling
    the atoms of anything FACET itself had written.
    """
    structure = _build("P 1 21/c 1", ["C1 C 0.1 0.2 0.3 1.0",
                                      "C2 C 0.4 0.5 0.6 1.0"],
                       number=14, with_ops=["x, y, z"])
    assert len(structure.atoms) == 2
    note = next(n for n in structure.notes if "expansion" in n)
    assert "1 operations from the file" in note
    # and it says the symbol implies more, because one of the two is wrong
    assert "shorter than its symbol implies" in note


def test_a_written_cif_reads_back_unchanged(tmp_path):
    """FACET writes the expanded cell, so the file it writes is P1 and says so.

    Declaring the original space group beside a P1 operation list makes a file
    that contradicts itself, and a reader that believes the symbol expands the
    already-expanded coordinates again.
    """
    from facet.core import exporters

    original = _build("F m -3 m", ["Na1 Na 0 0 0 1.0",
                                   "Cl1 Cl 0.5 0.5 0.5 1.0"])
    path = exporters.write_cif(original, tmp_path / "round.cif")
    text = Path(path).read_text(encoding="utf-8")
    assert "_space_group_name_H-M_alt  'P 1'" in text
    assert "# the source structure was F m -3 m" in text

    back = cif.read(path)
    assert back.n_atoms == original.n_atoms
    positions = np.array([a.frac for a in back.atoms])
    expected = np.array([a.frac for a in original.atoms])
    # each atom against its nearest counterpart, allowing for wrapping: the two
    # files need not list them in the same order
    delta = positions[:, None, :] - expected[None, :, :]
    delta -= np.round(delta)
    nearest = np.linalg.norm(delta, axis=-1).min(axis=1)
    assert float(nearest.max()) < 1e-6


# ===========================================================================
# the gap rule
# ===========================================================================

def test_a_site_with_no_gap_gets_every_contact():
    """Six contacts at one distance have no step to cut at.

    argmax over equal ratios returns whichever index rounding favours, so the
    same structure written with its origin somewhere else gave 3, 4, 5 or 6. A
    regular octahedron of one symmetry-equivalent oxygen is common, so this was
    not an edge case.
    """
    count, ratio = polyhedra.gap_split([2.3924] * 6)
    assert count == 6
    assert ratio == pytest.approx(1.0)

    count, ratio = polyhedra.gap_split([2.5] * 12)
    assert count == 12


def test_the_gap_rule_is_a_function_of_the_distances():
    """The same distances, perturbed at the last bit, must give one answer."""
    rng = np.random.default_rng(0)
    for distances in ([2.3924] * 6,
                      [2.1, 2.1, 2.1, 2.9, 2.9, 2.9],
                      [2.0, 2.0, 2.05, 2.1, 3.2, 3.3],
                      [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
                      [2.5] * 4 + [2.5] * 4):
        expected = polyhedra.gap_split(distances)[0]
        for _ in range(200):
            jittered = np.asarray(distances) * (
                1.0 + rng.normal(size=len(distances)) * 1e-15)
            assert polyhedra.gap_split(sorted(jittered))[0] == expected


def test_a_real_gap_is_still_found():
    """The tolerance must not stop the rule working where there is a step."""
    assert polyhedra.gap_split([2.1, 2.1, 2.1, 2.9, 2.9, 2.9])[0] == 3
    assert polyhedra.gap_split([2.0, 2.0, 2.05, 2.1, 3.2, 3.3])[0] == 4
    count, ratio = polyhedra.gap_split([1.9, 1.9, 1.95, 3.5, 3.6])
    assert count == 3
    assert ratio > 1.5


def test_the_gap_coordination_number_survives_an_origin_shift():
    """The whole point: a structure written differently is the same structure."""
    if not EQUAL_SHELL.is_file():
        pytest.skip("the structure is not present")
    import copy

    structure = cif.read(EQUAL_SHELL)
    reference = {r.label: r.cn_gap
                 for r in coordination.analyse_structure(structure, bv.DEFAULT)}

    for shift in (np.array([0.137, -0.291, 0.408]),
                  np.array([0.5, 0.5, 0.5])):
        moved = copy.deepcopy(structure)
        for site in moved.sites:
            site.frac = (site.frac + shift) % 1.0
        for atom in moved.atoms:
            atom.frac = (atom.frac + shift) % 1.0
            atom.cart = moved.cell.to_cartesian(atom.frac)
        for result in coordination.analyse_structure(moved, bv.DEFAULT):
            assert result.cn_gap == reference[result.label]


# ===========================================================================
# the health report
# ===========================================================================

def test_repeated_findings_collapse_to_one_with_a_count():
    """A shared site is found once per symmetry copy: sixteen identical lines.

    That is not more information, it is the same information made unreadable.
    """
    path = sample_cif("1519099", "1519099_Na3Bi(PO4)2.cif")
    if not path.is_file():
        pytest.skip("the structure is not present")
    report = quality.check(cif.read(path))

    messages = [f.message for f in report.findings]
    assert len(messages) == len(set(messages)), "the report repeats itself"
    close = [m for m in messages if "0.192" in m]
    assert close, "the shared site was not reported at all"
    assert "symmetry-equivalent occurrences" in close[0]


def test_the_formula_check_finds_real_non_stoichiometry():
    """Reported, not corrected: a cell that differs from its formula is a fact.

    Across the collection it flags exactly two structures, and both are
    chemically real -- sillenite gamma-Bi2O3 and Bi2Sr2CaCu2O8+delta, whose excess
    oxygen is the reason anyone studies it.
    """
    if not BI_DIR.is_dir():
        pytest.skip("the collection is not present")
    # Counted as structures, not as files. The collection holds some of them
    # under more than one path -- `_duplicates` keeps the redundant copies of
    # the reorganised folder, and `study/` has its own copy of several -- so a
    # count of paths would report the same structure more than once.
    flagged = set()
    for path in sorted(BI_DIR.rglob("*.cif")):
        if "_duplicates" in path.parts:
            continue
        try:
            structure = cif.read(path)
        except Exception:
            continue
        if not structure.formula:
            continue
        report = quality.check(structure)
        if any(f.code == "formula-mismatch" for f in report.findings):
            flagged.add((structure.formula, structure.spacegroup_hm))
    assert len(flagged) <= 3, f"too many structures flagged: {sorted(flagged)}"


def test_the_formula_parser_handles_the_forms_these_files_use():
    assert quality.parse_formula("Bi2 O3") == {"Bi": 2.0, "O": 3.0}
    assert quality.parse_formula("Bi0.92 O1.54 Si0.08") == pytest.approx(
        {"Bi": 0.92, "O": 1.54, "Si": 0.08})
    # a bracketed group, which several of these files use
    assert quality.parse_formula("Na3 Bi (P O4)2") == pytest.approx(
        {"Na": 3.0, "Bi": 1.0, "P": 2.0, "O": 8.0})
    assert quality.parse_formula("") == {}
    assert quality.parse_formula(None) == {}


# ===========================================================================
# the charge balance as a symmetry check
# ===========================================================================

def test_the_charge_balance_closes_for_a_well_formed_file():
    """Two searches from opposite directions over the same bonds."""
    path = sample_cif("1526458", "1526458_Bi2O3.cif")
    if not path.is_file():
        pytest.skip("the structure is not present")
    structure = cif.read(path)
    results = coordination.analyse_structure(structure, bv.DEFAULT)
    balance = bv_report.charge_balance(structure, results, bv.DEFAULT)
    assert abs(balance["difference"]) < 1e-9
    assert "machine precision" in balance["note"]


def test_the_phi_index_is_exactly_invariant_to_a_shift_in_r0():
    """The claim the whole phi index exists for, over random environments.

    Shifting R0 by delta multiplies every valence by exp(delta/b); phi is a ratio
    of two quantities that both carry that factor, so it cancels exactly. |BVV|
    does not, which is why phi is what gets reported.
    """
    rng = np.random.default_rng(3)
    worst_phi = 0.0
    worst_scale = 0.0
    for _ in range(2000):
        n = int(rng.integers(3, 12))
        vectors = rng.normal(size=(n, 3))
        valences = rng.random(n) * 0.9 + 0.01
        magnitude, phi, _ = bv.phi_index(vectors, valences)
        factor = math.exp(rng.normal() * 0.5)
        scaled_magnitude, scaled_phi, _ = bv.phi_index(vectors,
                                                       valences * factor)
        worst_phi = max(worst_phi, abs(scaled_phi - phi))
        if magnitude:
            worst_scale = max(worst_scale,
                              abs(scaled_magnitude / magnitude - factor))
    assert worst_phi < 1e-12, "phi is not invariant"
    assert worst_scale < 1e-12, "|BVV| does not scale with the valences"


def test_a_failed_symmetry_determination_leaves_a_note():
    """A quiet downgrade is what this program is meant to make impossible."""
    import facet.core.readers as readers
    from facet.core.structure import Cell, Site, Structure

    structure = Structure("bare", Cell(5.0, 5.0, 5.0, 90, 90, 90,
                                       orth=np.eye(3) * 5.0),
                          [Site("C1", "C", [0, 0, 0])], [])

    def explode(_):
        raise RuntimeError("spglib is unavailable")

    original = readers._annotate
    import facet.core.cif as cif_module
    saved = cif_module._annotate_symmetry
    cif_module._annotate_symmetry = explode
    try:
        readers._annotate(structure)
    finally:
        cif_module._annotate_symmetry = saved

    assert any("could not be determined" in note for note in structure.notes)
