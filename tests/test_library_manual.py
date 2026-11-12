"""The library manual states what the library does.

docs/library_manual.md is built by tools/build_library_manual.py, which runs
every example of docs/library_manual.src.md and writes what it printed under
it. This rebuilds it and compares: a change to the library that changes what
an example prints, or breaks one, fails here until the manual is rebuilt.

Numbers are compared to three significant figures, so the last digits a
different platform or numpy prints do not count as a difference.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import facet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("openpyxl")
pytestmark = pytest.mark.skipif("glass" not in facet.examples(),
                                reason="the practice glass is not bundled")

NUMBER = re.compile(r"-?\d+\.\d+(?:e[+-]?\d+)?")


def _loose(text: str) -> list[str]:
    """The text with every decimal number at three significant figures and
    runs of spaces and rule characters collapsed (a column one character
    wider is not a difference)."""
    def short(match):
        value = float(match.group(0))
        return f"{value:.3g}" if value == value else "nan"

    lines = []
    for line in text.splitlines():
        line = NUMBER.sub(short, line)
        line = re.sub(r"-{2,}", "-", line)
        lines.append(re.sub(r"\s+", " ", line).strip())
    return lines


@pytest.fixture(scope="module")
def rebuilt():
    import build_library_manual as builder

    return builder, builder.run(builder.SOURCE.read_text(encoding="utf-8"))


def test_every_example_runs_and_prints_what_the_manual_shows(rebuilt):
    builder, text = rebuilt
    committed = builder.OUT_MD.read_text(encoding="utf-8")
    mine, theirs = _loose(text), _loose(committed)
    different = [(i, a, b) for i, (a, b) in enumerate(zip(mine, theirs)) if a != b]
    assert len(mine) == len(theirs) and not different, (
        "docs/library_manual.md is out of date (py -3.11 tools/"
        f"build_library_manual.py): {different[:3]}")


def test_the_manual_names_this_version_and_shows_its_figures(rebuilt):
    builder, text = rebuilt
    assert f"version {facet.__version__}" in text
    shown = set(re.findall(r"\]\(library_manual/([^)]+)\)", text))
    assert len(shown) >= 5
    for name in shown:
        assert (builder.FIGURES / name).stat().st_size > 3000, name
    assert builder.OUT_PDF.is_file() and builder.OUT_PDF.read_bytes()[:5] == b"%PDF-"


def test_every_public_name_is_in_the_manual(rebuilt):
    """A function added to the library and not to the manual is found here."""
    _, text = rebuilt
    import facet.library as lib

    skipped = {"V_BOND", "V_LIST", "GLASS_TYPE_MAP", "SiteView", "PlaneView",
               "Results", "row_of", "request"}
    missing = [name for name in lib.__all__
               if name not in skipped and f"facet.{name}" not in text
               and f"`{name}`" not in text]
    assert not missing, missing
    for cls in (lib.Crystal, lib.Model, lib.Results, lib.SiteView, lib.PlaneView,
                lib.Table):
        for name in vars(cls):
            if name.startswith("_") or not callable(getattr(cls, name)):
                continue
            if f"{name}(" not in text and f".{name}" not in text:
                missing.append(f"{cls.__name__}.{name}")
    assert not missing, missing
