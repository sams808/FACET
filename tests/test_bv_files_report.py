"""Loading published parameter sets, and the whole-structure bond-valence reports.

The charge balance is the interesting test here. The cation and anion sums are
two separate neighbour searches running in opposite directions over the same
bonds, so they must agree -- and across the Bi collection they agree to machine
precision. Where they do not, the difference measures how far a file's
coordinates fall short of the symmetry it declares, which is verified directly
against the offending file.
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import bv, bv_files, bv_report, cif, coordination

BI_DIR = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi")
SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")
BII3_P3 = sample_cif("1010622", "1010622_BiI3-P3.cif")
BII3_R3 = sample_cif("1010622", "1010622_BiI3-R-3H.cif")


@pytest.fixture(scope="module")
def structure():
    if not SAMPLE.is_file():
        pytest.skip("the sample structure is not present")
    return cif.read(SAMPLE)


@pytest.fixture(scope="module")
def results(structure):
    return coordination.analyse_structure(structure, bv.DEFAULT)


# ===========================================================================
# loading parameter files
# ===========================================================================

BVPARM = """data_bvparm
loop_
_valence_param_atom_1
_valence_param_atom_1_valence
_valence_param_atom_2
_valence_param_atom_2_valence
_valence_param_Ro
_valence_param_B
_valence_param_ref_id
_valence_param_details
Bi 3 O  -2 2.094 0.37 a 'recommended'
Bi 3 O  -2 2.090 0.37 b '.'
Bi 5 O  -2 2.06  0.37 a '.'
Bi 3 I  -1 2.72  0.40 a '.'
Na 1 O  -2 1.803 0.37 a '.'
Si 4 O  -2 1.624 0.37 a '.'
?  . O  -2 1.00  0.37 a 'unreadable'
"""

TABLE = """# a plain table, softBV shape
# cation charge anion charge R0 b
Bi  3  O  -2  2.094  0.370
Bi  5  O  -2  2.060  0.370
Na  1  O  -2  1.803  0.370
Si  4  O  -2  1.624  0.370
Bi  3  I  -1  2.720  0.400
"""

CHARGED_TABLE = """Bi3+  O2-  2.094  0.370
Na1+  O2-  1.803  0.370
Si4+  O2-  1.624  0.370
"""


def _write(text, suffix):
    import tempfile

    directory = tempfile.mkdtemp()
    path = os.path.join(directory, f"params{suffix}")
    Path(path).write_text(text, encoding="utf-8")
    return path


def test_the_iucr_bvparm_loop_is_read():
    params, report = bv_files.load(_write(BVPARM, ".cif"))
    assert report.format == "IUCr bvparm CIF"
    assert report.pairs == 5
    assert report.with_own_b == 5
    assert report.skipped == 1           # the '?' row
    assert report.duplicates == 1        # Bi3-O appears twice

    assert params.get("Bi", 3, "O").r0 == pytest.approx(2.094)
    assert params.get("Bi", 5, "O").r0 == pytest.approx(2.06)
    assert params.get("Na", 1, "O").r0 == pytest.approx(1.803)


def test_the_recommended_row_wins_over_the_other():
    """A pair may appear from several authors; the marked row is the one used."""
    params, _ = bv_files.load(_write(BVPARM, ".cif"))
    assert params.get("Bi", 3, "O").r0 == pytest.approx(2.094)


def test_a_per_pair_b_is_kept_rather_than_replaced():
    """0.37 is a fitted average, not a law.

    Using it where a compilation states 0.40 changes every valence for that
    pair by exp(dR/b) -- so a loaded set's own b has to survive the load.
    """
    params, _ = bv_files.load(_write(BVPARM, ".cif"))
    assert params.get("Bi", 3, "I").b == pytest.approx(0.40)
    assert params.get("Bi", 3, "O").b == pytest.approx(0.37)


def test_a_plain_table_is_read():
    params, report = bv_files.load(_write(TABLE, ".txt"))
    assert report.format == "plain table"
    assert report.pairs == 5
    assert params.get("Bi", 3, "O").r0 == pytest.approx(2.094)
    assert params.get("Bi", 3, "I").b == pytest.approx(0.40)


def test_a_table_carrying_the_charge_on_the_symbol_is_read():
    """Real tables are not consistent about where the charge goes."""
    params, report = bv_files.load(_write(CHARGED_TABLE, ".dat"))
    assert report.pairs == 3
    assert params.get("Bi", 3, "O").r0 == pytest.approx(2.094)
    assert params.get("Si", 4, "O").r0 == pytest.approx(1.624)


def test_a_comma_separated_table_is_read():
    text = TABLE.replace("  ", ",").replace(",,", ",")
    params, report = bv_files.load(_write(text, ".csv"))
    assert report.pairs >= 4
    assert params.get("Bi", 3, "O") is not None


def test_the_format_is_chosen_by_content_not_only_by_extension():
    """A bvparm distribution saved as .txt, and a table saved as .cif."""
    params, report = bv_files.load(_write(BVPARM, ".txt"))
    assert report.format == "IUCr bvparm CIF"


def test_a_structure_cif_is_refused_with_a_reason():
    if not SAMPLE.is_file():
        pytest.skip("the sample structure is not present")
    with pytest.raises(ValueError, match="structure rather than a parameter"):
        bv_files.load(SAMPLE)


def test_an_empty_file_is_refused():
    with pytest.raises(ValueError):
        bv_files.load(_write("# nothing here\n", ".txt"))


def test_charges_are_parsed_however_they_are_written():
    for text, expected in (("3", 3), ("+3", 3), ("3+", 3), ("-2", -2),
                           ("2-", -2), ("1", 1), (".", None), ("?", None),
                           ("", None), (None, None), ("abc", None)):
        assert bv_files._parse_charge(text) == expected


def test_a_value_with_an_esd_is_read():
    assert bv_files._number("1.973(4)") == pytest.approx(1.973)
    assert bv_files._number("2.09") == pytest.approx(2.09)
    assert bv_files._number(".") is None


def test_a_loaded_set_carries_its_provenance():
    params, report = bv_files.load(_write(TABLE, ".txt"), name="my set")
    assert params.name == "my set"
    assert "params.txt" in params.source
    assert any("loaded from" in note for note in params.notes)
    # and every parameter it hands out names that source
    assert "params.txt" in params.get("Bi", 3, "O").source


def test_a_set_round_trips_through_json():
    params, _ = bv_files.load(_write(BVPARM, ".cif"))
    path = _write("{}", ".json")
    bv_files.write_json(path, params)

    restored, report = bv_files.load(path)
    assert report.format == "FACET JSON"
    for key in params.pairs():
        original = params.get(*key)
        copy = restored.get(*key)
        assert copy is not None
        assert copy.r0 == pytest.approx(original.r0)
        assert copy.b == pytest.approx(original.b)


def test_loading_rubbish_json_is_refused():
    with pytest.raises(ValueError):
        bv_files.load(_write(json.dumps({"parameters": []}), ".json"))


def test_comparing_two_sets_reports_the_valence_ratio():
    """A 0.02 A difference in R0 is a five per cent difference in every valence.

    That is the number that matters, so compare() reports it rather than leaving
    the reader to exponentiate.
    """
    first, _ = bv_files.load(_write(TABLE, ".txt"))
    shifted = TABLE.replace("2.094", "2.114")
    second, _ = bv_files.load(_write(shifted, ".txt"))

    rows = bv_files.compare(first, second)
    top = rows[0]
    assert top[0] == "Bi" and top[2] == "O"
    assert top[5] == pytest.approx(0.020, abs=1e-6)
    assert top[6] == pytest.approx(math.exp(0.020 / 0.37), rel=1e-9)
    assert top[6] == pytest.approx(1.0555, abs=1e-3)


def test_with_b_overrides_every_per_pair_value():
    params, _ = bv_files.load(_write(BVPARM, ".cif"))
    assert params.get("Bi", 3, "I").b == pytest.approx(0.40)
    uniform = params.with_b(0.37)
    assert uniform.get("Bi", 3, "I").b == pytest.approx(0.37)
    assert uniform.get("Bi", 3, "I").r0 == pytest.approx(
        params.get("Bi", 3, "I").r0)


# ===========================================================================
# the cutoff table
# ===========================================================================

def test_the_cutoff_table_reproduces_the_bismuth_oxide_numbers(structure):
    """The figures this program was designed around.

    With R0 = 2.09 for Bi(III)-O and b = 0.37, a threshold of 0.075 v.u. puts the
    bond cutoff at 3.05 A and 0.02 v.u. puts the listing cutoff at 3.54 A. Those
    are the distances the literature picks by hand; here they follow from the
    threshold.
    """
    table = bv_report.cutoff_table(structure, bv.DEFAULT)
    row = next(r for r in table.rows if r.cation == "Bi" and r.anion == "O")
    assert row.r0 == pytest.approx(2.09, abs=0.001)
    assert row.d_bond == pytest.approx(3.05, abs=0.01)
    assert row.d_list == pytest.approx(3.54, abs=0.01)
    assert row.fitted


def test_the_cutoff_follows_the_closed_form(structure):
    """d = R0 - b ln(v), for every row."""
    for v_bond, v_list in ((0.075, 0.02), (0.03, 0.01), (0.15, 0.05)):
        table = bv_report.cutoff_table(structure, bv.DEFAULT, v_bond, v_list)
        for row in table.rows:
            assert row.d_bond == pytest.approx(
                row.r0 - row.b * math.log(v_bond), rel=1e-12)
            assert row.d_list == pytest.approx(
                row.r0 - row.b * math.log(v_list), rel=1e-12)
            assert row.window == pytest.approx(row.d_list - row.d_bond)


def test_a_lower_threshold_gives_a_longer_cutoff(structure):
    distances = []
    for v_bond in (0.2, 0.1, 0.075, 0.03, 0.01):
        table = bv_report.cutoff_table(structure, bv.DEFAULT, v_bond, 0.005)
        row = next(r for r in table.rows if r.anion == "O")
        distances.append(row.d_bond)
    assert distances == sorted(distances)


def test_two_pairs_with_different_r0_get_different_cutoffs():
    """The point of stating a valence instead of a distance."""
    params, _ = bv_files.load(_write(TABLE, ".txt"))
    from facet.core.structure import Cell, Site, Structure

    cell = Cell(8.0, 8.0, 8.0, 90, 90, 90, orth=np.eye(3) * 8.0)
    sites = [Site("Bi1", "Bi", [0, 0, 0], ox=3),
             Site("Na1", "Na", [0.5, 0, 0], ox=1),
             Site("O1", "O", [0.25, 0, 0], ox=-2)]
    structure = Structure("made up", cell, sites, [])

    table = bv_report.cutoff_table(structure, params)
    cutoffs = {row.cation: row.d_bond for row in table.rows}
    assert cutoffs["Bi"] != cutoffs["Na"]
    assert cutoffs["Bi"] - cutoffs["Na"] == pytest.approx(2.094 - 1.803,
                                                         abs=1e-9)


def test_the_table_fills_in_what_was_actually_found(structure, results):
    bare = bv_report.cutoff_table(structure, bv.DEFAULT)
    filled = bv_report.cutoff_table(structure, bv.DEFAULT, results=results)

    assert all(row.shortest is None for row in bare.rows)
    assert any("no analysis was supplied" in note for note in bare.notes)

    row = next(r for r in filled.rows if r.anion == "O")
    assert row.shortest is not None
    assert 1.8 < row.shortest < 2.5
    assert row.n_bonds > 0
    assert row.n_listed >= row.n_bonds


def test_the_counts_agree_with_the_analysis(structure, results):
    """Whatever the table says cleared a threshold, the analysis must too."""
    table = bv_report.cutoff_table(structure, bv.DEFAULT, results=results)
    total_bonds = sum(row.n_bonds for row in table.rows)
    expected = sum(1 for r in results for c in r.contacts
                   if c.valence is not None and c.valence >= table.v_bond)
    assert total_bonds == expected


def test_the_table_says_when_a_value_was_estimated():
    from facet.core.structure import Cell, Site, Structure

    cell = Cell(8.0, 8.0, 8.0, 90, 90, 90, orth=np.eye(3) * 8.0)
    # a pair with no fitted parameter in FACET's small table
    sites = [Site("Rb1", "Rb", [0, 0, 0], ox=1),
             Site("Te1", "Te", [0.3, 0, 0], ox=-2)]
    structure = Structure("estimated", cell, sites, [])
    table = bv_report.cutoff_table(structure, bv.DEFAULT)
    if table.rows:
        assert any(not row.fitted for row in table.rows)
        assert any("estimated" in note for note in table.notes)


def test_the_table_exports_as_csv_and_text(structure, results):
    table = bv_report.cutoff_table(structure, bv.DEFAULT, results=results)
    text = table.as_text()
    assert "d(bond)" in text and "R0" in text
    assert "Bi" in text

    csv = table.as_csv()
    lines = [line for line in csv.splitlines() if not line.startswith("#")]
    assert lines[0].startswith("cation,cation_ox,anion,R0,b,d_bond,d_list")
    assert len(lines) - 1 == len(table.rows)
    first = lines[1].split(",")
    assert float(first[5]) == pytest.approx(table.rows[0].d_bond, abs=1e-6)


def test_a_negative_threshold_is_refused(structure):
    with pytest.raises(ValueError, match="positive"):
        bv_report.cutoff_table(structure, bv.DEFAULT, v_bond=0.0)


def test_the_table_states_that_a_cutoff_is_derived_not_chosen(structure):
    table = bv_report.cutoff_table(structure, bv.DEFAULT)
    assert any("not chosen" in note for note in table.notes)


# ===========================================================================
# the threshold scan
# ===========================================================================

def test_the_scan_reproduces_the_coordination_numbers(results):
    """At the threshold the analysis used, the scan must agree with it."""
    scan = bv_report.threshold_scan(results, low=0.005, high=0.30, steps=200)
    for result in results:
        counts = scan.per_site[result.label]
        index = int(np.argmin(np.abs(scan.thresholds - bv.V_BOND_DEFAULT)))
        assert counts[index] == result.cn_valence


def test_the_count_falls_as_the_threshold_rises(results):
    scan = bv_report.threshold_scan(results)
    for counts in scan.per_site.values():
        assert np.all(np.diff(counts) <= 0), "CN rose with a higher threshold"


def test_a_stable_range_is_where_the_count_does_not_change(results):
    scan = bv_report.threshold_scan(results, low=0.005, high=0.30, steps=300)
    for label, counts in scan.per_site.items():
        span = scan.stable_range(label, bv.V_BOND_DEFAULT)
        assert span is not None
        low, high = span
        assert low <= bv.V_BOND_DEFAULT <= high
        inside = counts[(scan.thresholds >= low) & (scan.thresholds <= high)]
        assert len(set(inside.tolist())) == 1, "the range is not stable"
        # and it is maximal: just outside, the count differs
        index = int(np.argmin(np.abs(scan.thresholds - low)))
        if index > 0:
            assert counts[index - 1] != inside[0]


def test_the_widths_say_which_sites_have_a_defensible_number(results):
    """A wide plateau means the count does not depend on where the line is drawn.

    Reported as a width; no site is labelled reliable or unreliable.
    """
    scan = bv_report.threshold_scan(results)
    widths = scan.widths(bv.V_BOND_DEFAULT)
    assert set(widths) == set(scan.per_site)
    for width in widths.values():
        assert width >= 0


def test_an_unknown_label_has_no_range(results):
    scan = bv_report.threshold_scan(results)
    assert scan.stable_range("nothing like this", 0.075) is None


def test_the_scan_exports_as_csv(results):
    scan = bv_report.threshold_scan(results, steps=20)
    lines = [line for line in scan.as_csv().splitlines()
             if not line.startswith("#")]
    assert lines[0].startswith("v_bond,")
    assert len(lines) - 1 == 20
    assert len(lines[1].split(",")) == len(scan.per_site) + 1


def test_the_scan_costs_no_neighbour_search(results):
    """Its whole point: a contact's valence does not depend on the threshold."""
    scan = bv_report.threshold_scan(results)
    assert any("no neighbour search was repeated" in note
               for note in scan.notes)


# ===========================================================================
# anion sums and the charge balance
# ===========================================================================

def test_the_anion_sums_come_out_near_the_formal_charge(structure):
    """alpha-Bi2O3: every oxygen should reach about 2 v.u.

    This is the test that caught the tempting shortcut. Re-gathering the cation
    contact lists instead of searching around the anions gave 0.66 v.u. for an
    O(2-) -- low enough to look like a finding rather than an error -- because a
    cation site's contacts belong to one representative atom, and an anion is
    reached by cation atoms from every equivalent position.
    """
    rows = bv_report.anion_sums(structure, bv.DEFAULT)
    assert rows
    for row in rows:
        assert row.element == "O"
        assert row.expected == 2
        assert row.bvs == pytest.approx(2.0, abs=0.15), row.label
        assert abs(row.discrepancy) < 0.15
        assert 2 <= row.n_contacts <= 6


def test_the_anion_sums_report_both_thresholds(structure):
    rows = bv_report.anion_sums(structure, bv.DEFAULT)
    for row in rows:
        assert row.bvs_listed >= row.bvs - 1e-12
        assert row.n_listed >= row.n_contacts


@pytest.mark.parametrize("path", sorted(BI_DIR.rglob("*.cif"))[:24]
                         if BI_DIR.is_dir() else [],
                         ids=lambda p: p.stem[:22])
def test_the_cation_and_anion_totals_agree(path):
    """Two searches, opposite directions, same bonds: they must agree.

    Where they do not, the difference measures how far the deposited coordinates
    fall short of the symmetry the file declares -- so the tolerance here is set
    by that, not by the arithmetic.
    """
    structure = cif.read(path)
    results = coordination.analyse_structure(structure, bv.DEFAULT)
    if not results:
        pytest.skip("nothing to analyse")
    balance = bv_report.charge_balance(structure, results, bv.DEFAULT)
    scale = max(abs(balance["cation_valence"]), 1e-12)
    assert abs(balance["difference"]) / scale < 1e-3


def test_a_symmetric_file_balances_exactly():
    """R-3H BiI3: the symmetry-equivalent distances are exactly equal."""
    if not BII3_R3.is_file():
        pytest.skip("the file is not present")
    structure = cif.read(BII3_R3)
    results = coordination.analyse_structure(structure, bv.DEFAULT)
    balance = bv_report.charge_balance(structure, results, bv.DEFAULT)
    assert abs(balance["difference"]) < 1e-9
    assert "machine precision" in balance["note"]


def test_a_file_that_breaks_its_own_symmetry_is_reported(structure):
    """P3 BiI3: bonds symmetry requires to be equal differ by 0.0006 A.

    The residual is a measurement of the file, not an error in the arithmetic --
    which is checked here by measuring the inequality directly and then seeing
    the note say so.
    """
    if not BII3_P3.is_file():
        pytest.skip("the file is not present")
    from facet.core.neighbors import NeighborFinder, search_radius_for

    offender = cif.read(BII3_P3)
    finder = NeighborFinder(
        offender, rmax=search_radius_for(offender, bv.DEFAULT,
                                         bv.V_LIST_DEFAULT))
    spreads = []
    for index, site in enumerate(offender.sites):
        if site.is_anion:
            continue
        contacts = finder.contacts_for_site(index)
        near = sorted(float(contacts.distance[i])
                      for i in range(len(contacts))
                      if offender.sites[contacts.neighbor_site[i]].is_anion
                      and contacts.distance[i] < 3.2)
        if len(near) >= 2:
            spreads.append(near[-1] - near[0])
    # the shortest shell is not a single distance, as P3 would require
    assert max(spreads) > 1e-5

    results = coordination.analyse_structure(offender, bv.DEFAULT)
    balance = bv_report.charge_balance(offender, results, bv.DEFAULT)
    assert 0 < abs(balance["difference"]) < 0.05
    assert "fall short of the symmetry" in balance["note"]


def test_a_structure_with_no_usable_parameters_is_not_a_nan(structure):
    """Bi metal has no anions; the totals must be zero and say why."""
    for name in ("1010834_Mg3Bi2_Zintl.cif", "4002410_Bi_metal.cif"):
        path = BI_DIR / "cifs" / name
        if not path.is_file():
            continue
        try:
            metal = cif.read(path)
        except Exception:
            continue
        results = coordination.analyse_structure(metal, bv.DEFAULT)
        balance = bv_report.charge_balance(metal, results, bv.DEFAULT)
        assert math.isfinite(balance["cation_valence"])
        assert math.isfinite(balance["difference"])


def test_a_structure_with_no_atoms_reports_nothing():
    """A structure whose sites were never expanded is a legitimate input."""
    from facet.core.structure import Cell, Site, Structure

    cell = Cell(8.0, 8.0, 8.0, 90, 90, 90, orth=np.eye(3) * 8.0)
    structure = Structure("unexpanded", cell,
                          [Site("O1", "O", [0, 0, 0], ox=-2)], [])
    assert bv_report.anion_sums(structure, bv.DEFAULT) == []


def test_an_anion_with_no_cations_near_it_sums_to_zero():
    """Not an error: a measurement, and a very informative one."""
    from facet.core.structure import Atom, Cell, Site, Structure

    orth = np.eye(3) * 12.0
    cell = Cell(12.0, 12.0, 12.0, 90, 90, 90, orth=orth)
    sites = [Site("O1", "O", [0.0, 0.0, 0.0], ox=-2),
             Site("O2", "O", [0.5, 0.5, 0.5], ox=-2)]
    atoms = [Atom("O", np.array(s.frac), orth @ np.array(s.frac), i, s.label)
             for i, s in enumerate(sites)]
    structure = Structure("only anions", cell, sites, atoms)

    rows = bv_report.anion_sums(structure, bv.DEFAULT)
    assert len(rows) == 2
    for row in rows:
        assert row.bvs == 0.0
        assert row.n_contacts == 0
        assert row.discrepancy == pytest.approx(-2.0)


def test_the_balance_note_never_grades_the_structure(structure, results):
    import re

    balance = bv_report.charge_balance(structure, results, bv.DEFAULT)
    forbidden = {"good", "bad", "poor", "excellent", "correct", "incorrect",
                 "wrong", "reliable", "unreliable", "should", "suspicious"}
    words = set(re.findall(r"[a-z]+", balance["note"].lower()))
    assert not (words & forbidden)
