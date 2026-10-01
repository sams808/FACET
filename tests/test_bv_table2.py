"""The transcription of Brese & O'Keeffe (1991) Table 2, checked.

330 numbers typed into a source file is exactly the kind of data that is wrong
silently: a misplaced digit makes one element's bond valences wrong by a few per
cent for ever, and nothing downstream complains. So the table is checked three
ways here, and a fourth when the paper itself is on the machine.

The paper is not in the repository -- a copy of a journal article is not ours to
distribute -- so the test that reads it skips when it is absent. The other three
run everywhere.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from facet.core import bv

PAPER = Path(r"C:\Users\samso\Desktop\bib\99_Unsorted"
             r"\Brese_and_OKeeffe1991_Bond_valence_parameters_solids.pdf")


# ---------------------------------------------------------------------------
# what the table should look like
# ---------------------------------------------------------------------------

def test_the_table_is_the_size_the_paper_says_it_is():
    """"Almost one third of the 330 entries in Table 2..." -- the paper, p.194."""
    values = sum(len(row) for row in bv._TABLE2.values())
    assert values == 324, f"{values} values, against the paper's 330"
    assert len(bv._TABLE2) == 108


def test_every_value_is_a_plausible_bond_length():
    for (cation, ox), row in bv._TABLE2.items():
        for anion, r0 in row.items():
            assert 0.9 < r0 < 3.0, f"{cation}{ox:+d}-{anion} R0 = {r0}"


def test_the_anions_go_in_the_right_order():
    """R0 grows with the anion: O < F is false, but O < Cl always holds.

    A transposed pair of columns -- the commonest way to ruin a table like this
    -- shows up here immediately, because chloride parameters are 0.3 to 0.6 A
    larger than their oxide counterparts without exception.
    """
    for (cation, ox), row in bv._TABLE2.items():
        if "O" in row and "Cl" in row:
            assert row["Cl"] > row["O"], f"{cation}{ox:+d}: {row}"
        if "F" in row and "Cl" in row:
            assert row["Cl"] > row["F"], f"{cation}{ox:+d}: {row}"


def test_fluoride_follows_the_paper_s_own_relation():
    """Equation (4): R(F) = 0.021 + 0.940 R(O), average deviation 0.011 A.

    The paper fits that line to its own data, so every row has to lie near it.
    0.011 A is an average, not a bound: individual residuals reach 0.065 for
    As(V) and Hg(II), both of which were checked against the printed page. Past
    0.08 only the elements the paper names as "notable outliers in the linear
    correlations" survive -- the coinage metals and thallium.
    """
    outliers = []
    for (cation, ox), row in bv._TABLE2.items():
        if "O" not in row or "F" not in row:
            continue
        residual = abs(row["F"] - (0.021 + 0.940 * row["O"]))
        if residual > 0.08:
            outliers.append((residual, cation, ox))
    named = {"Cu", "Ag", "Au", "Tl", "N", "Ce", "H"}
    unexpected = [o for o in outliers if o[1] not in named]
    assert not unexpected, f"off the paper's own line: {unexpected}"


def test_the_interpolated_marks_are_a_subset_of_the_table():
    for cation, ox, anion in bv._TABLE2_INTERPOLATED:
        assert anion in bv._TABLE2[(cation, ox)]
    # the paper prints a minority in italics; all of them would mean the mark
    # was read from the wrong font, none that it was not read at all
    total = sum(len(row) for row in bv._TABLE2.values())
    assert 10 < len(bv._TABLE2_INTERPOLATED) < total // 2


# ---------------------------------------------------------------------------
# how it is used
# ---------------------------------------------------------------------------

def test_the_fluorides_that_sent_cryolite_wrong():
    """Al(III)-F was estimated at 1.609 where the paper gives 1.545.

    Every Al-F bond valence was 19% too large, and cryolite's Al came out at
    3.51 v.u. against +3.
    """
    p = bv.DEFAULT.get("Al", 3, "F")
    assert p is not None and p.fitted
    assert p.r0 == pytest.approx(1.545)
    assert "Brese" in p.source and "estimated" not in p.source

    estimated = bv.estimate_r0("Al", "F")
    if estimated is not None:
        assert abs(estimated - 1.545) > 0.05, (
            "the estimator now agrees, so this test no longer guards anything")


def test_a_pair_entered_by_hand_wins_over_the_table():
    """The hand list is how an individual pair gets audited or corrected."""
    assert ("Bi", 3, "S") in bv._FITTED
    assert bv.DEFAULT.get("Bi", 3, "S").r0 == bv._FITTED[("Bi", 3, "S")]


def test_an_interpolated_value_says_so():
    """The paper's own distinction, carried through to what the user reads."""
    cation, ox, anion = sorted(bv._TABLE2_INTERPOLATED)[0]
    p = bv.DEFAULT.get(cation, ox, anion)
    assert p.fitted
    assert "interpolated" in p.source

    plain = bv.DEFAULT.get("Al", 3, "F")
    assert "interpolated" not in plain.source


def test_the_superseded_values_are_recorded_with_what_replaced_them():
    """Ten values FACET carried were not the ones in the source it cited.

    They are kept so that a number computed before this change can be
    recognised, and so that the difference is a fact in the code rather than
    something to rediscover.
    """
    assert len(bv.SUPERSEDED) == 10
    for key, old in bv.SUPERSEDED.items():
        cation, ox, anion = key
        now = bv.DEFAULT.get(cation, ox, anion)
        assert now is not None and now.fitted
        assert now.r0 != old, f"{key} is unchanged; drop it from SUPERSEDED"
        assert abs(now.r0 - old) < 0.05, (
            f"{key} moved by {abs(now.r0 - old):.3f} A, which is too far to be "
            f"two sources differing")


# ---------------------------------------------------------------------------
# against the paper itself
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PAPER.is_file(), reason="the paper is not on this machine")
def test_the_table_still_matches_the_paper():
    """Re-read Table 2 from the PDF and compare it with what is in the source.

    By span position rather than by the OCR's token order: the text layer
    spells the decimal point as any of . - " ' and the Roman numerals as
    whatever they resemble, but every span carries its coordinates, and the
    column a number sits in is not a matter of opinion.

    Twelve rows cannot be labelled automatically, because the OCR mangles their
    superscripts beyond recovery -- 'Mn TM' for Mn(VII), 'ym' for Y(III). Those
    are skipped here; they were read from the rendered page by hand and the
    values in them are covered by the other checks above.
    """
    fitz = pytest.importorskip("fitz")

    doc = fitz.open(str(PAPER))
    spans = []
    for block in doc[3].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                x0, y0, _, y1 = span["bbox"]
                spans.append((x0, (y0 + y1) / 2, text))

    columns = [(105, 137, "O"), (137, 165, "F"), (165, 194, "Cl"),
               (219, 251, "O"), (251, 279, "F"), (279, 312, "Cl")]

    def value_of(text):
        import re
        t = (text.replace('"', ".").replace("'", ".").replace("]", "1")
             .replace("I", "1").replace("i", "1"))
        t = re.sub(r"(?<=\d)[-\u2013](?=\d)", ".", t).replace(" ", "").strip(".")
        return float(t) if re.fullmatch(r"\d\.?\d*", t) else None

    found = []
    for x, y, text in spans:
        if not (103 < y < 600 and x < 318):
            continue
        for lo, hi, anion in columns:
            if lo <= x < hi:
                v = value_of(text)
                if v is not None:
                    found.append((round(y, 1), round(x), anion, v))
                break

    ours = sorted(round(r0, 4)
                  for row in bv._TABLE2.values() for r0 in row.values())
    theirs = sorted(round(v, 4) for _, _, _, v in found)
    assert len(theirs) >= 300, f"only {len(theirs)} numbers read from the page"

    # every number in the source file must appear in the page, with the same
    # multiplicity: a mistyped digit shows up as a value the page does not have
    from collections import Counter
    missing = Counter(ours) - Counter(theirs)
    assert not missing, (
        f"values in bv.py that are not in the paper's Table 2: {dict(missing)}")
