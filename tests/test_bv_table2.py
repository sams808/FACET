"""The transcription of Brese & O'Keeffe (1991) Table 2, checked.

327 numbers typed into a source file fail silently: a misplaced digit moves one
element's bond valences by a few per cent for ever, a row filed under the next
element sends that element to the estimator, and nothing downstream complains.
So the table is checked several ways here, and once more, row by row, when the
paper itself is on the machine.

The paper is not in the repository -- a copy of a journal article is not ours to
distribute -- so the test that reads it skips when it is absent. The others run
everywhere.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from facet.core import bv

PAPER = Path(r"C:\Users\samso\Desktop\bib\99_Unsorted"
             r"\Brese_and_OKeeffe1991_Bond_valence_parameters_solids.pdf")


# ---------------------------------------------------------------------------
# what the table should look like
# ---------------------------------------------------------------------------

# The row labels as printed, left band top to bottom and then the right band,
# read from the rendered page in the double-entry audit of 2026-10-06. The text
# layer cannot supply them: it reads H(I) as "H m" and both Ti and Tl as "Ti"
# or "TI", and two misreadings of exactly that kind had filed H(I) under
# ("H", 3) and Ti(IV) under ("Tl", 4), with Ti(III) dropped.
PRINTED = """
    Ac3 Ag1 Al3 Am3 As3 As5 Au3 B3 Ba2 Be2 Bi3 Bi5 Bk3 Br7 C4 Ca2 Cd2 Ce3
    Ce4 Cf3 Cl7 Cm3 Co2 Co3 Cr2 Cr3 Cr6 Cs1 Cu1 Cu2 Dy3 Er3 Eu2 Eu3 Fe2 Fe3
    Ga3 Gd3 Ge4 H1 Hf4 Hg1 Hg2 Ho3 I5 I7 In3 Ir5 K1 La3 Li1 Lu3 Mg2 Mn2 Mn3

    Mn4 Mn7 Mo6 N3 N5 Na1 Nb5 Nd3 Ni2 Os4 P5 Pb2 Pb4 Pd2 Pr3 Pt2 Pt4 Pu3
    Rb1 Re7 Rh3 Ru4 S4 S6 Sb3 Sb5 Sc3 Se4 Se6 Si4 Sm3 Sn2 Sn4 Sr2 Ta5 Tb3
    Te4 Te6 Th4 Ti3 Ti4 Tl1 Tl3 Tm3 U4 U6 V3 V4 V5 W6 Y3 Yb3 Zn2 Zr4
"""
PRINTED_LABELS = [(m[1], int(m[2]))
                  for m in re.finditer(r"([A-Z][a-z]?)(\d)", PRINTED)]
ROWS_PER_BAND = (55, 54)


def test_the_table_is_the_size_the_page_prints():
    """109 rows of O, F and Cl: 327 values, 48 of them printed in italics.

    The paper's text (p. 194) speaks of "the 330 entries in Table 2". The table
    is set as two bands of 55 row slots, and 2 x 55 x 3 = 330 counts the empty
    last slot of the right-hand band, after Zr(IV). Until the audit of
    2026-10-06 this file held 108 rows and 324 values; the three values of the
    dropped Ti(III) row are the whole of that difference.
    """
    values = sum(len(row) for row in bv._TABLE2.values())
    assert len(bv._TABLE2) == 109 == sum(ROWS_PER_BAND)
    assert values == 327, f"{values} values, against the 327 printed"
    assert all(set(row) == {"O", "F", "Cl"} for row in bv._TABLE2.values())
    assert len(bv._TABLE2_INTERPOLATED) == 48


def test_the_rows_are_the_ones_the_page_prints():
    """Every printed label is a key, and every key is a printed label.

    The values alone cannot show a relabelled row: H(I) filed as ("H", 3) kept
    its place in the order and its numbers.
    """
    assert len(PRINTED_LABELS) == sum(ROWS_PER_BAND)
    # the paper's order: alphabetical by element symbol, then ascending
    # oxidation state -- which is what sorted() gives on the keys
    assert PRINTED_LABELS == sorted(PRINTED_LABELS)
    absent = sorted(set(PRINTED_LABELS) - set(bv._TABLE2))
    extra = sorted(set(bv._TABLE2) - set(PRINTED_LABELS))
    assert not absent and not extra, (
        f"printed but not in _TABLE2: {absent}; in _TABLE2 but not printed: "
        f"{extra}")


def test_every_row_is_an_oxidation_state_the_element_takes():
    """A check that owes nothing to the page: H(III) and Tl(IV) fail it."""
    core = pytest.importorskip("pymatgen.core")
    unknown = [(cation, ox) for cation, ox in bv._TABLE2
               if ox not in core.Element(cation).oxidation_states]
    assert not unknown, (
        f"not among pymatgen's oxidation states for the element: {unknown}")


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

def _value_of(text):
    """A number from the OCR text layer, which spells the decimal point as any
    of . - " ' and turns a 1 into I, i or ] at times ('2. I 12' is 2.112)."""
    t = (text.replace('"', ".").replace("'", ".").replace("]", "1")
         .replace("I", "1").replace("i", "1"))
    t = re.sub(r"(?<=\d)[-\u2013](?=\d)", ".", t).replace(" ", "").strip(".")
    return float(t) if re.fullmatch(r"\d\.?\d*", t) else None


def _printed_rows(page):
    """Table 2's rows as printed, in reading order.

    Each item is ``(band, y, OCR label, {anion: value}, {italic anions})``. The
    page is two bands side by side, each headed Cation | O | F | Cl; a printed
    row is the set of spans at one height within a band, and the bands are
    read left, then right.
    """
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                x0, y0, _, y1 = span["bbox"]
                y = (y0 + y1) / 2
                if text and 103 < y < 495 and x0 < 318:
                    italic = bool(span["flags"] & 2) or "Italic" in span["font"]
                    spans.append((x0, y, text, italic))

    bands = [
        [(78, 105, "label"), (105, 137, "O"), (137, 165, "F"), (165, 194, "Cl")],
        [(192, 219, "label"), (219, 251, "O"), (251, 279, "F"), (279, 312, "Cl")],
    ]
    printed = []
    for band, columns in enumerate(bands):
        lo, hi = columns[0][0], columns[-1][1]
        rows = []
        for span in sorted((s for s in spans if lo <= s[0] < hi),
                           key=lambda s: s[1]):
            # rows are 7 pt apart; spans of one row agree to under 1 pt
            if rows and span[1] - rows[-1][0] < 2.5:
                rows[-1][1].append(span)
            else:
                rows.append([span[1], [span]])
        for y, row in rows:
            label, values, italic = "", {}, set()
            for x, _, text, is_italic in sorted(row):
                column = next(c for a, b, c in columns if a <= x < b)
                if column == "label":
                    label = label or text
                    continue
                v = _value_of(text)
                assert v is not None and column not in values, (
                    f"band {band + 1}, y = {y:.1f}: {text!r} does not read as "
                    f"the row's one {column} value")
                values[column] = v
                if is_italic:
                    italic.add(column)
            printed.append((band, y, label, values, italic))
    return printed


@pytest.mark.skipif(not PAPER.is_file(), reason="the paper is not on this machine")
def test_the_table_still_matches_the_paper():
    """Re-read Table 2 from the PDF, row by row, and compare it with the source.

    The k-th printed row has to carry the k-th row of _TABLE2 in the paper's
    order -- alphabetical by element symbol, then ascending oxidation state,
    which is sorted() on the keys -- with the same three numbers and the same
    italics. A dropped row shifts every row after it and a row filed under
    another element sits out of place, so both fail here. The test this
    replaced compared the two collections of numbers as a whole, and passed
    with Ti(III) absent and Ti(IV) filed as Tl(IV) (found by the double-entry
    visual audit of 2026-10-06).

    The oxidation states cannot be read from the text layer, so the labels are
    checked by test_the_rows_are_the_ones_the_page_prints; here the first
    letter of each OCR label is compared, after the OCR's l and 1 for I. The
    italic font flag does survive the OCR, on exactly the 48 italic values.
    """
    fitz = pytest.importorskip("fitz")
    printed = _printed_rows(fitz.open(str(PAPER))[3])

    per_band = tuple(sum(1 for p in printed if p[0] == b) for b in (0, 1))
    assert per_band == ROWS_PER_BAND, f"rows per band on the page: {per_band}"

    keys = sorted(bv._TABLE2)
    problems = []
    if len(keys) != len(printed):
        problems.append(f"{len(printed)} rows on the page, {len(keys)} in _TABLE2")
    for k, ((band, y, label, values, italic), key) in enumerate(
            zip(printed, keys)):
        cation, ox = key
        where = (f"printed row {k + 1} (band {band + 1}, y = {y:.1f}, "
                 f"OCR label {label!r}) against _TABLE2[{key}]")
        first = {"1": "I", "l": "I"}.get(label[:1], label[:1].upper())
        if first != cation[0]:
            problems.append(f"{where}: the label starts with {first!r}")
        if values != bv._TABLE2[key]:
            problems.append(f"{where}: page {values}, _TABLE2 {bv._TABLE2[key]}")
        marked = {a for a in values if (cation, ox, a) in bv._TABLE2_INTERPOLATED}
        if italic != marked:
            problems.append(f"{where}: italic on the page {sorted(italic)}, "
                            f"in _TABLE2_INTERPOLATED {sorted(marked)}")
    assert not problems, (f"{len(problems)} disagreements with the page; "
                          "the first ones:\n" + "\n".join(problems[:10]))
