"""Optical basicity, checked against the papers rather than against itself.

* Duffy and Ingram's basicity moderating parameters (1976, Table 1) come
  back from Pauling's 1960 electronegativities through their relation;
* Duffy's oxide and fluoride Lambda values (1989, Table 1) are 1 / gamma
  and 1 / (2.3 gamma);
* the microscopic basicities of their Table 3 come back when the shares are
  the Pauling bond strengths, which is what the bond valences generalise;
* the compounds of Duffy's 1996 Table 2 come back from their stoichiometry
  on his spectroscopic scale;
* the bond-valence mean over the anions equals the composition value
  exactly when every cation delivers its formal charge, and otherwise
  differs by the departure -- an identity, not a tolerance;
* an MD frame's per-atom lambda equals the crystal path's lambda of the same
  atom, through md_site, so the two workspaces cannot disagree.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from facet.core import basicity as B, bv, coordination, readers
from facet.core.coordination import ContactRow, SiteResult

EXAMPLES = Path(__file__).resolve().parent.parent / "facet" / "data" / "examples"
QUARTZ = EXAMPLES / "quartz_SiO2_cod9013321.cif"
BIPO4 = EXAMPLES / "bismuth_phosphate_BiPO4_cod9008088.cif"
CRYOLITE = EXAMPLES / "cryolite_Na3AlF6_cod9004097.cif"


# -- the relation and the tables -------------------------------------------------

def test_the_relation_reproduces_duffy_and_ingrams_table_1():
    """J. Non-Cryst. Solids 21 (1976) 373, Table 1. Aluminium is the one
    entry the paper itself did not take from the relation (1.65 printed,
    1.69 from chi = 1.5); every other one comes back to 0.01."""
    table = {"Cl": 3.73, "N": 3.73, "S": 3.04, "C": 3.04, "P": 2.50,
             "H": 2.50, "B": 2.36, "Si": 2.09, "Zn": 1.82, "Al": 1.65,
             "Mg": 1.28, "Ca": 1.00, "Li": 1.00, "Na": 0.87, "K": 0.73,
             "Rb": 0.73, "Cs": 0.60}
    for element, gamma in table.items():
        m = B.PAULING.moderator(element, None)
        assert m is not None and not m.measured
        assert m.gamma == pytest.approx(gamma, abs=0.04 if element == "Al"
                                        else 0.011), element


def test_oxide_and_fluoride_values_of_duffy_1989():
    """J. Non-Cryst. Solids 109 (1989) 35, Table 1: Lambda of the oxide is
    1 / gamma and of the fluoride 1 / (2.3 gamma). Strontium's oxide value
    (1.10) is a measured one, 0.99 from chi = 1.0, so it is left to the
    duffy scale below; the alkali values are printed to 0.05."""
    oxides = {"Na": 1.15, "Ba": 1.15, "Li": 1.00, "Ca": 1.00, "Mg": 0.78,
              "Al": 0.60, "Si": 0.48, "B": 0.42, "P": 0.40}
    fluorides = {"Na": 0.50, "Ba": 0.50, "Li": 0.43, "Ca": 0.43, "Mg": 0.34,
                 "Al": 0.26, "Si": 0.21, "B": 0.18, "P": 0.17}
    for element, value in oxides.items():
        assert B.oxide_basicity(element, None, "pauling", "O") == \
            pytest.approx(value, abs=0.011), element
    for element, value in fluorides.items():
        assert B.oxide_basicity(element, None, "pauling", "F") == \
            pytest.approx(value, abs=0.011), element
    assert B.oxide_basicity("Cs", None, "pauling", "O") == pytest.approx(1.7, abs=0.05)
    assert B.oxide_basicity("K", None, "pauling", "O") == pytest.approx(1.4, abs=0.05)
    assert B.oxide_basicity("Sr", 2, "duffy", "O") == pytest.approx(1.10, abs=0.011)
    assert B.oxide_basicity("Sr", 2, "duffy", "F") == pytest.approx(0.48, abs=0.011)


def test_a_measured_scale_falls_back_to_the_relation_and_says_so():
    bi = B.LEBOUTEILLER.moderator("Bi", 3)
    assert bi.measured and bi.basicity == pytest.approx(1.19)
    assert bi.coordination == 6 and "Lebouteiller" in bi.source
    # the oxidation state is part of the key: Fe3+ and Fe2+ differ
    assert B.LEBOUTEILLER.moderator("Fe", 3).basicity == pytest.approx(0.77)
    assert B.LEBOUTEILLER.moderator("Fe", 2).basicity == pytest.approx(1.00)
    # Bi5+ is not in the table: derived, and marked
    bi5 = B.LEBOUTEILLER.moderator("Bi", 5)
    assert not bi5.measured and "Pauling" in bi5.source
    assert bi5.gamma == pytest.approx(B.gamma_from_electronegativity(1.9))
    # an element Pauling did not place has nothing at all
    assert B.PAULING.moderator("Xx", 2) is None
    with pytest.raises(ValueError):
        B.scale_named("nonesuch")
    assert B.DIMITROV.moderator("Bi", 3).basicity == pytest.approx(1.19)
    assert B.DUFFY.moderator("P", 5).gamma == pytest.approx(3.0)


# -- one anion: Table 3 of 1976 ------------------------------------------------------

def _anion(cations, charge=2, element="O", v_bond=0.075):
    """An anion-centred SiteResult whose contacts carry the Pauling bond
    strength z / CN as their valence: Duffy and Ingram's shares."""
    rows = []
    for k, (el, z, cn) in enumerate(cations):
        param = bv.BVParam(el, z, element, -charge, 1.8, 0.37, "test", True)
        rows.append(ContactRow(f"{el}{k}", el, 10 + k, 10 + k, 1.8,
                               np.array([1.0, 0.0, 0.0]), (0, 0, 0), 1.0,
                               z / cn, param))
    return SiteResult(0, "O1", element, -charge, "test", None, None, 1,
                      rows, v_bond, 0.03)


@pytest.mark.parametrize("name, cations, printed", [
    ("Si-O", [("Si", 4, 4)], 0.74),
    ("Si-O-Si", [("Si", 4, 4)] * 2, 0.48),
    ("Si-O-Al(4)", [("Si", 4, 4), ("Al", 3, 4)], 0.59),
    ("Si-O-Al(6)", [("Si", 4, 4), ("Al", 3, 6)], 0.65),
    ("Si-O-B(3)", [("Si", 4, 4), ("B", 3, 3)], 0.45),
    ("Si-O-B(4)", [("Si", 4, 4), ("B", 3, 4)], 0.52),
    ("Si-O-P", [("Si", 4, 4), ("P", 5, 4)], 0.36),
    ("Si-O-H", [("Si", 4, 4), ("H", 1, 1)], 0.44),
    ("B(3)-O", [("B", 3, 3)], 0.71),
    ("B(4)-O", [("B", 3, 4)], 0.78),
    ("B-O-B(3)", [("B", 3, 3)] * 2, 0.42),
    ("B(3)-O-Mg(6)", [("B", 3, 3), ("Mg", 2, 6)], 0.68),
    ("P-O", [("P", 5, 4)], 0.63),
    ("P-O-P", [("P", 5, 4)] * 2, 0.25),
    ("Al(4)-O", [("Al", 3, 4)], 0.85),
    ("Al(6)-O", [("Al", 3, 6)], 0.91),
    ("Al(4)-O-Al(4)", [("Al", 3, 4)] * 2, 0.70),
    ("Al(4)-O-H", [("Al", 3, 4), ("H", 1, 1)], 0.55),
])
def test_the_microscopic_basicities_of_table_3(name, cations, printed):
    """Duffy and Ingram 1976, Table 3, with the shares they used (z / CN
    over the charge of 2). Two of the table's rows, P-O-Al(6) 0.54 and
    Al(6)-O-Al(6) 0.83, do not follow from the paper's own gamma values
    (0.53 and 0.80 do) and are not pinned."""
    m = B.micro_basicity(_anion(cations), "pauling")
    assert m.defined and not m.missing
    assert m.value == pytest.approx(printed, abs=0.015), name
    assert m.value_listed == pytest.approx(m.value)
    assert sum(s.share for s in m.shares) == pytest.approx(
        sum(z / cn for _, z, cn in cations) / 2)


def test_the_remainder_counts_as_unmoderated_and_the_threshold_decides_the_set():
    """A terminal Si-O carries half the charge; the other half is at
    gamma = 1 and the oxygen comes to 0.74, not 0.48. A contact under the
    bond threshold is in the listed value, not the bonded one."""
    terminal = B.micro_basicity(_anion([("Si", 4, 4)]), "pauling")
    assert terminal.unmoderated == pytest.approx(0.5)
    assert terminal.value == pytest.approx(0.74, abs=0.01)
    # a magnesium at 0.05 v.u. (below the 0.075 bond threshold): gamma above
    # 1, so it lowers the listed value and not the bonded one
    weak = _anion([("Si", 4, 4), ("Mg", 2, 40)])
    m = B.micro_basicity(weak, "pauling")
    assert [s.bonded for s in m.shares] == [True, False]
    assert m.value == pytest.approx(terminal.value)
    assert m.value_listed < m.value
    at_lower = B.micro_basicity(weak, "pauling", v_bond=0.01)
    assert at_lower.value == pytest.approx(m.value_listed)
    # an alkali has gamma below 1 and raises the basicity of its oxygen,
    # which is the sign Duffy and Ingram's equation gives it
    sodium = B.micro_basicity(_anion([("Si", 4, 4), ("Na", 1, 20)]), "pauling")
    assert sodium.value_listed > sodium.value


def test_a_fluoride_is_on_the_common_scale_and_a_sulphide_is_not():
    f = B.micro_basicity(_anion([("Na", 1, 6)] * 6, charge=1, element="F"),
                         "pauling")
    assert f.defined and f.factor == 2.3
    assert f.value == pytest.approx(0.50, abs=0.01)         # NaF, Duffy 1989
    s = B.micro_basicity(_anion([("Ca", 2, 6)] * 6, charge=2, element="S"),
                         "pauling")
    assert not s.defined and math.isnan(s.value)
    assert any("oxide" in n and "fluoride" in n for n in s.notes)


def test_a_cation_without_a_parameter_leaves_the_anion_without_a_value():
    m = B.micro_basicity(_anion([("Si", 4, 4), ("Xx", 2, 6)]), "pauling")
    assert m.missing == ["Xx2+"] and math.isnan(m.value)
    assert any("Xx2+" in n for n in m.notes)


# -- a composition: Table 2 of 1996 ------------------------------------------------

@pytest.mark.parametrize("name, composition, printed", [
    ("Na2Si2O5", {("Na", 1): 2, ("Si", 4): 2, ("O", -2): 5}, 0.61),
    ("Na2SiO3", {("Na", 1): 2, ("Si", 4): 1, ("O", -2): 3}, 0.70),
    ("Na4SiO4", {("Na", 1): 4, ("Si", 4): 1, ("O", -2): 4}, 0.82),
    ("NaPO3", {("Na", 1): 1, ("P", 5): 1, ("O", -2): 3}, 0.47),
    ("Na4P2O7", {("Na", 1): 4, ("P", 5): 2, ("O", -2): 7}, 0.56),
    ("Na2SO4", {("Na", 1): 2, ("S", 6): 1, ("O", -2): 4}, 0.48),
    ("Na2B4O7", {("Na", 1): 2, ("B", 3): 4, ("O", -2): 7}, 0.52),
    ("Na2B2O4", {("Na", 1): 2, ("B", 3): 2, ("O", -2): 4}, 0.62),
    ("Na4B2O5", {("Na", 1): 4, ("B", 3): 2, ("O", -2): 5}, 0.71),
    ("H3BO3", {("H", 1): 3, ("B", 3): 1, ("O", -2): 3}, 0.41),
    ("H4SiO4", {("H", 1): 4, ("Si", 4): 1, ("O", -2): 4}, 0.44),
    ("H3PO4", {("H", 1): 3, ("P", 5): 1, ("O", -2): 4}, 0.36),
    ("H2SO4", {("H", 1): 2, ("S", 6): 1, ("O", -2): 4}, 0.29),
    ("Ca3(PO4)2", {("Ca", 2): 3, ("P", 5): 2, ("O", -2): 8}, 0.58),
    ("3Na2O.7SiO2", {("Na", 1): 6, ("Si", 4): 7, ("O", -2): 17}, 0.60),
    ("H2O", {("H", 1): 2, ("O", -2): 1}, 0.40),
])
def test_the_compounds_of_duffy_1996_table_2(name, composition, printed):
    """J. Chem. Educ. 73 (1996) 1138, Table 2, on the scale of its
    Table 1: equivalent fractions over the cation charge, each at
    1 / gamma."""
    b = B.bulk_basicity(composition, "duffy")
    assert b.value == pytest.approx(printed, abs=0.016), name
    assert sum(r.fraction for r in b.rows) == pytest.approx(1.0)
    assert not b.notes


def test_a_fluoride_composition_and_a_charged_one():
    assert B.bulk_basicity({("Na", 1): 1, ("F", -1): 1}, "duffy").value == \
        pytest.approx(0.50, abs=0.005)                     # NaF, Duffy 1989
    cryolite = B.bulk_basicity({("Na", 1): 3, ("Al", 3): 1, ("F", -1): 6},
                               "duffy")
    # Duffy's eq. (4) for an all-fluoride: 0.5 Lambda(NaF) + 0.5 Lambda(AlF3)
    assert cryolite.value == pytest.approx(0.5 * 0.50 + 0.5 * 0.264, abs=0.005)
    charged = B.bulk_basicity({("Si", 4): 1, ("O", -2): 3}, "pauling")
    assert any("net charge" in n for n in charged.notes)
    assert charged.value == pytest.approx(B.oxide_basicity("Si", 4))
    sulphide = B.bulk_basicity({("Ca", 2): 1, ("S", -2): 1}, "pauling")
    assert sulphide.excluded == ["S"] and math.isnan(sulphide.value)


# -- a structure ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def quartz():
    s = readers.read(str(QUARTZ))
    full = coordination.analyse_structure(s, bv.DEFAULT, cations_only=False)
    return s, full


def test_quartz_oxygens_are_bridging_and_the_mean_follows_the_sums(quartz):
    s, full = quartz
    sb = B.structure_basicity(s, full, "pauling")
    assert sb.bulk.value == pytest.approx(B.oxide_basicity("Si", 4, "pauling"))
    (o,) = sb.anions.values()
    assert o.defined and len([x for x in o.shares if x.bonded]) == 2
    # Si-O-Si at the bond valences Brese and O'Keeffe give quartz: the sum
    # is 2.09 rather than 2, so the oxygen is lowered a little more than the
    # 0.48 of the ideal shares
    assert o.sum_bonded == pytest.approx(2.09, abs=0.01)
    assert 0.44 < o.value < 0.48
    assert sb.mean == pytest.approx(o.value)
    # the identity: mean - bulk = -(sum over cations of (BVS - z) x
    # (1 - 1/gamma) x atoms) / (2 x oxygen atoms)
    si = next(r for r in full if r.element == "Si")
    gamma = B.PAULING.moderator("Si", 4).gamma
    n_si = sum(x.occupancy * (x.multiplicity or 1) for x in s.sites
               if x.element == "Si")
    n_o = sum(x.occupancy * (x.multiplicity or 1) for x in s.sites
              if x.element == "O")
    predicted = -(si.bvs - 4) * (1 - 1 / gamma) * n_si / (2 * n_o)
    assert sb.difference == pytest.approx(predicted, abs=2e-3)
    # the cation's environment: its oxygens, and nothing else bonds them
    env = sb.environments[si.site_index]
    assert env.n_anions == 4 and env.value == pytest.approx(o.value)
    assert env.value_host == pytest.approx(1.0)


def test_the_mean_equals_the_composition_value_when_the_sums_are_the_charges():
    """A synthetic structure analysed from both ends: every bond carries
    exactly its ideal share, so the two routes to Lambda coincide."""
    from facet.core.structure import Cell, Site, Structure

    class _FakeSite:
        pass

    # one Mg site (6 O at 1/3 v.u.) and three O sites (2 Mg each): MgO-like
    mg_contacts = [ContactRow(f"O{k}", "O", 1 + (k % 3), 1 + (k % 3), 2.1,
                              np.array([1.0, 0, 0]), (0, 0, 0), 1.0, 1 / 3,
                              bv.BVParam("Mg", 2, "O", -2, 1.693, 0.37, "t", True))
                   for k in range(6)]
    mg = SiteResult(0, "Mg", "Mg", 2, "t", None, None, 1, mg_contacts, 0.075, 0.03)
    o_sites = []
    for k in range(3):
        contacts = [ContactRow("Mg", "Mg", 0, 0, 2.1, np.array([1.0, 0, 0]),
                               (0, 0, 0), 1.0, 1 / 3,
                               bv.BVParam("Mg", 2, "O", -2, 1.693, 0.37, "t", True))
                    for _ in range(6)]
        o_sites.append(SiteResult(1 + k, f"O{k}", "O", -2, "t", None, None, 2,
                                  contacts, 0.075, 0.03))
    sites = [Site("Mg", "Mg", [0, 0, 0], multiplicity=6, ox=2)] + [
        Site(f"O{k}", "O", [0.5, 0.5, 0.5], multiplicity=2, ox=-2)
        for k in range(3)]
    structure = Structure("fake", Cell(4, 4, 4, 90, 90, 90, np.eye(3) * 4),
                          sites, [])
    sb = B.structure_basicity(structure, [mg] + o_sites, "pauling")
    assert sb.bulk.value == pytest.approx(B.oxide_basicity("Mg", 2))
    assert sb.mean == pytest.approx(sb.bulk.value, abs=1e-12)
    assert sb.difference == pytest.approx(0.0, abs=1e-12)


def test_bismuth_phosphate_on_the_four_scales():
    """Bi3+ is where the scales part: 0.45 from the electronegativity
    relation, 1.19 measured. Each value names its source."""
    s = readers.read(str(BIPO4))
    full = coordination.analyse_structure(s, bv.DEFAULT, cations_only=False)
    by_scale = {key: B.structure_basicity(s, full, key) for key in B.SCALE_KEYS}
    assert by_scale["lebouteiller"].bulk.value > by_scale["pauling"].bulk.value + 0.2
    bi = next(r for r in full if r.element == "Bi")
    for key, sb in by_scale.items():
        assert len(sb.anions) == 2 and all(m.defined for m in sb.anions.values())
        env = sb.environments[bi.site_index]
        assert env.n_anions == bi.cn_valence
        # the host value removes Bi's own lowering: the oxygens as P leaves them
        assert env.value_host != pytest.approx(env.value)
    duffy = by_scale["duffy"]
    assert any("Bi3+" in n for n in duffy.bulk.notes)      # derived, said so
    assert not by_scale["lebouteiller"].bulk.notes
    # the Site-tab helper hands back the right kind for each site
    o = next(r for r in full if r.element == "O")
    assert isinstance(B.site_basicity(o, full, "pauling"), B.MicroBasicity)
    env = B.site_basicity(bi, full, "pauling")
    assert isinstance(env, B.EnvironmentBasicity)
    assert env.value == pytest.approx(
        by_scale["pauling"].environments[bi.site_index].value)


def test_cryolite_fluorines_and_the_composition_agree():
    s = readers.read(str(CRYOLITE))
    full = coordination.analyse_structure(s, bv.DEFAULT, cations_only=False)
    sb = B.structure_basicity(s, full, "pauling")
    assert set(sb.by_element) == {"F"}
    assert all(m.factor == 2.3 for m in sb.anions.values())
    assert sb.bulk.value == pytest.approx(0.38, abs=0.01)
    assert abs(sb.difference) < 0.02


# -- an MD frame: the same number by the other path ----------------------------------

def test_a_frames_lambda_equals_the_crystal_paths_lambda_atom_for_atom():
    from facet.core import bulk, md_model, md_site

    structure = readers.read(str(QUARTZ))
    frame, _ = md_model.supercell_frame(structure, (2, 2, 2))
    ox = md_model.model_oxidation(frame.species)
    ox_atom = ox.per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox_atom, bv.DEFAULT, v_list_vu=0.03)
    fb = B.frame_basicity(table, "pauling", v_bond=0.075, formers=("Si",))
    anions = np.flatnonzero(np.asarray(table.is_anion))
    assert np.isfinite(fb.value[anions]).all()
    assert np.isnan(fb.value[~np.asarray(table.is_anion)]).all()
    assert (fb.classes[anions] == 2).all()                 # every O bridging
    assert fb.bulk.value == pytest.approx(B.oxide_basicity("Si", 4))
    for row in anions[:5]:
        site = md_site.site_result_for_atom(frame, table, int(row), v_bond=0.075)
        m = B.micro_basicity(site, "pauling")
        assert fb.value[row] == pytest.approx(m.value, abs=1e-12)
        assert fb.value_listed[row] == pytest.approx(m.value_listed, abs=1e-12)
    assert B.class_name(0) == "free" and B.class_name(7) == "tricluster"


# -- through the driver, the report, the command line and the window ----------------

def test_the_md_driver_runs_basicity_on_quartz_frames():
    from facet.core import md_analysis as ma, md_model

    structure = readers.read(str(QUARTZ))
    base, _ = md_model.supercell_frame(structure, (2, 2, 2))
    rng = np.random.default_rng(3)
    frames = []
    for k in range(2):
        cart = base.cart_ang + rng.normal(0.0, 0.02, base.cart_ang.shape)
        frames.append(md_model.frame_from_arrays(
            base.elements, cart, box_ang=base.box_ang, timestep=k))
    trajectory = md_model.MemoryTrajectory(frames)
    request = ma.AnalysisRequest(analyses=("basicity",),
                                 formers=frozenset({"Si"}))
    assert "basicity" in ma.available_analyses(request)[0]
    result = ma.analyse(trajectory, request)
    out = result.outputs["basicity"]
    assert out.ok, out.error
    tables = out.tables
    assert tables["optical basicity, composition"].mean == pytest.approx(
        B.oxide_basicity("Si", 4))
    mean = tables["mean micro basicity O (BV)"]
    assert 0.40 < mean.mean < 0.50
    assert "mean micro basicity O to the listing cut" in tables
    assert "mean micro basicity O BO (BV)" in tables          # every O bridging
    assert "mean micro basicity O NBO (BV)" not in tables
    hist = tables["micro basicity (O)"]
    n_o = int((np.asarray(frames[0].elements) == "O").sum())
    assert int(hist.counts[0].sum() + hist.n_below[0] + hist.n_above[0]) == n_o
    assert out.provenance.method_parameters["scale"] == "pauling"
    params = tables["basicity moderating parameters"].as_rows()
    assert any(r["species"] == "Si4+" and r["from"] == "from electronegativity"
               for r in params)
    # a scale the engine does not know is refused before any frame is read
    bad = ma.AnalysisRequest(analyses=("basicity",), formers=frozenset({"Si"}),
                             basicity=ma.BasicityOptions(scale="nonesuch"))
    assert any(m.name == "basicity.scale" for m in ma.invalid_inputs(bad))


def test_the_md_names_and_the_setup_place_the_analysis():
    from facet.ui.md_names import display_name, group_of

    assert group_of("basicity") == "Structure"
    assert display_name("mean micro basicity O (BV)") == \
        "Mean microscopic basicity λ of O — bond-valence cut"
    assert display_name("micro basicity (O)") == "Microscopic basicity λ of O"
    assert display_name("mean micro basicity O NBO (BV)") == \
        "Mean microscopic basicity λ of O, NBO — bond-valence cut"
    assert display_name("optical basicity, composition") == \
        "Optical basicity Λ from the composition"


def test_the_report_carries_a_basicity_table_and_the_balance_columns(tmp_path):
    from facet.core import report

    rep = report.analyse_files([QUARTZ, BIPO4], basicity_scale="duffy")
    rows = rep.basicity_rows()
    assert {r["kind"] for r in rows} == {"anion", "cation"}
    quartz_o = [r for r in rows
                if r["structure"] == QUARTZ.name and r["kind"] == "anion"]
    assert len(quartz_o) == 1 and 0.44 < quartz_o[0]["lambda"] < 0.48
    assert "Si" in quartz_o[0]["shares"]
    balance = {r["structure"]: r for r in rep.balance_rows()}
    assert balance[QUARTZ.name]["basicity_scale"] == "duffy"
    assert balance[QUARTZ.name]["optical_basicity_composition"] == \
        pytest.approx(B.oxide_basicity("Si", 4, "duffy"), abs=1e-6)
    assert any("optical basicity" in line for line in rep.provenance())
    files = report.write_report(rep, tmp_path / "out.csv", "csv")
    assert any(f.name.endswith("basicity.csv") for f in files)
    none = report.analyse_files([QUARTZ], basicity_scale=None)
    assert none.basicity_rows() == [] and none.structures[0].basicity is None
    assert not any("optical basicity" in line for line in none.provenance())


def test_the_site_command_prints_the_basicity_rows(capsys):
    from facet.site.cli import main

    assert main([str(BIPO4), "--site", "Bi", "--basicity-scale", "duffy"]) == 0
    out = capsys.readouterr().out
    assert "Optical basicity about this site" in out and "duffy scale" in out
    assert main([str(BIPO4), "--anions", "--all", "--format", "md"]) == 0
    out = capsys.readouterr().out
    assert "Microscopic optical basicity" in out
    assert "Shares of the anion charge" in out
    assert main([str(QUARTZ), "--basicity-scale", "none"]) == 0
    assert "basicity" not in capsys.readouterr().out.lower()


def test_the_crystal_scene_colours_the_anions_by_basicity_with_a_legend():
    from facet.core import theme as theme_mod
    from facet.gl.scene import build_scene

    s = readers.read(str(BIPO4))
    full = coordination.analyse_structure(s, bv.DEFAULT, cations_only=False)
    sb = B.structure_basicity(s, full, "pauling")
    scalars = {index: m.value for index, m in sb.anions.items()}
    theme = theme_mod.Theme()
    theme.color_mode = theme_mod.ColorMode.BASICITY
    cations = [r for r in full if r.ox is not None and r.ox > 0]
    scene = build_scene(s, cations, theme=theme, site_scalars=scalars)
    assert scene.value_legend is not None
    title, lo, hi, diverging = scene.value_legend
    assert title == "microscopic basicity" and not diverging
    assert lo <= min(scalars.values()) and hi >= max(scalars.values())
    assert hi - lo >= 0.02                          # O1 0.464, O2 0.440
    sites = np.asarray(scene.atom_site)
    colours = np.asarray(scene.atom_color)
    for index, value in scalars.items():
        expected = np.array(theme_mod.color_for_value(value, lo, hi), np.float32)
        rows = np.flatnonzero(sites == index)
        assert rows.size and np.allclose(colours[rows], expected, atol=1e-6)
    # the two oxygens differ in colour, and the cations keep their element
    o1, o2 = list(scalars)
    assert not np.allclose(colours[sites == o1][0], colours[sites == o2][0])
    bi = next(r for r in cations if r.element == "Bi")
    assert np.allclose(colours[sites == bi.site_index][0],
                       theme.element_color("Bi"), atol=1e-6)
    # without the scalars the mode draws element colours and no legend
    plain = build_scene(s, cations, theme=theme)
    assert plain.value_legend is None
    theme.color_mode = theme_mod.ColorMode.ELEMENT
    assert build_scene(s, cations, theme=theme).value_legend is None


def test_the_md_highlight_colours_a_frame_by_basicity():
    from facet.core import bulk, md_model, theme as theme_mod
    from facet.ui.md_highlight import COLOUR_MODES, FrameData, colour_atoms
    from facet.ui.md_scene import build_md_scene

    assert "microscopic basicity" in COLOUR_MODES
    structure = readers.read(str(QUARTZ))
    frame, _ = md_model.supercell_frame(structure, (2, 2, 2))
    ox = md_model.model_oxidation(frame.species)
    ox_atom = ox.per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox_atom, bv.DEFAULT, v_list_vu=0.03)
    fd = FrameData.from_frame(frame, table, v_bond=0.075, formers=("Si",),
                              ox_atom=ox_atom)
    theme = theme_mod.Theme()
    scene = build_md_scene(frame, table, theme=theme, v_bond=0.075,
                           v_list=0.03)
    note = colour_atoms(scene, fd, "microscopic basicity", theme)
    assert note == ""
    title, lo, hi, _ = scene.value_legend
    assert title.startswith("microscopic basicity") and "pauling" in title
    values = fd.basicity_values()
    anions = np.flatnonzero(np.asarray(table.is_anion))
    assert np.isfinite(values[anions]).all()
    assert lo <= values[anions].min() and hi >= values[anions].max()
    sites = np.asarray(scene.atom_site)
    colours = np.asarray(scene.atom_color)
    row = int(anions[0])
    expected = np.array(theme_mod.color_for_value(float(values[row]), lo, hi),
                        np.float32)
    assert np.allclose(colours[sites == row], expected, atol=1e-6)
    si = int(np.flatnonzero(~np.asarray(table.is_anion))[0])
    assert np.allclose(colours[sites == si][0], theme.element_color("Si"),
                       atol=1e-6)
    # the scale is the frame data's, and a switch recolours
    fd.basicity_scale = "duffy"
    colour_atoms(scene, fd, "microscopic basicity", theme)
    assert "duffy" in scene.value_legend[0]
    assert fd.basicity_values("duffy")[row] != pytest.approx(values[row])
    # element colouring clears the legend
    colour_atoms(scene, fd, "element", theme)
    assert scene.value_legend is None


def test_the_window_rebuilds_the_basicity_colouring_on_a_scale_change(qapp):
    from conftest import dispose
    from facet.core import theme as theme_mod
    from facet.ui import preview

    win = preview.PreviewWindow()
    try:
        win.load_many([str(BIPO4)])
        qapp.processEvents()
        win.theme.color_mode = theme_mod.ColorMode.BASICITY
        win._rebuild()
        qapp.processEvents()
        legend = win.scene.value_legend
        assert legend is not None and legend[0] == "microscopic basicity"
        before = (legend[1], legend[2])
        win.utilities.set_basicity_scale("lebouteiller")
        qapp.processEvents()
        after = win.scene.value_legend
        assert after is not None and (after[1], after[2]) != before
        assert after[1] > 0.6                       # Bi3+ at 1.19 lifts every O
    finally:
        dispose(win)


def test_the_tools_basicity_tab_and_the_site_tab_follow_the_scale(qapp):
    from conftest import dispose
    from facet.ui import preview

    win = preview.PreviewWindow()
    try:
        win.load_many([str(BIPO4)])
        qapp.processEvents()
        panel = win.utilities
        titles = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
        assert "Basicity" in titles
        panel.tabs.setCurrentIndex(titles.index("Basicity"))
        qapp.processEvents()
        text = panel.basicity_text.toPlainText()
        assert "from the composition" in text
        assert "0.418" in text                      # BiPO4 on the Pauling scale
        assert panel.basicity_anions.rowCount() == 2
        assert panel.basicity_cations.rowCount() == 2
        # the Site tab of the bismuth: the anions about it
        entry = win.project.current
        bi = next(r for r in win.project.results_for(entry)
                  if r.element == "Bi")
        entry.selected_site = bi.site_index
        win._update_analysis()
        qapp.processEvents()
        page = win.analysis.toPlainText()
        assert "Optical basicity about this site" in page
        assert "pauling scale" in page
        # the scale chosen on the Tools tab reaches the Site tab
        panel.set_basicity_scale("lebouteiller")
        qapp.processEvents()
        assert "lebouteiller scale" in win.analysis.toPlainText()
        assert "0.653" in panel.basicity_text.toPlainText()
        # an anion analysed as a site lists every cation's share
        win.anion_check.setChecked(True)
        qapp.processEvents()
        entry = win.project.current
        o = next(r for r in win.project.results_for(entry) if r.element == "O")
        entry.selected_site = o.site_index
        win._update_analysis()
        qapp.processEvents()
        page = win.analysis.toPlainText()
        assert "Microscopic optical basicity" in page
        assert "Optical basicity of this anion" in page
        # Lebouteiller's table lists both P5+ and Bi3+ (in the order of the
        # shares): nothing is derived
        assert "measured for P5+, Bi3+" in page and "marked chi" not in page
        # Duffy's table has P5+ and not Bi3+: the derived one is marked
        panel.set_basicity_scale("duffy")
        qapp.processEvents()
        page = win.analysis.toPlainText()
        assert "measured for P5+" in page
        assert "electronegativity for Bi3+" in page and "marked chi" in page
    finally:
        dispose(win)
