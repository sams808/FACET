"""The structures that ship with FACET, and what each one is for.

A tutorial that begins "download a CIF" is a tutorial most people do not
finish, and one written around a file the reader does not have cannot be
checked by them at all. These six structures travel with the application, so
every worked example in the manual can be followed on a fresh installation
with nothing downloaded and no network.

They are chosen to disagree with each other. Quartz has a coordination number
nobody argues about; eulytite has one that depends on where the line is drawn.
Senarmontite and valentinite are the same compound twice over and do not give
the same answer. Cryolite is a fluoride, which is where a cutoff chosen for
oxygen stops being defensible, and its Na2 site has a plateau so narrow that
its coordination number is a property of the choice rather than of the
structure. BiPO4 is the same Bi(III) ion as in eulytite, in a host that treats
it quite differently.

All six come from the Crystallography Open Database, whose contents are placed
in the public domain. ``ATTRIBUTION.md`` beside the files records the COD entry
and the original publication for each.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    """One bundled structure and the reason it is bundled."""

    key: str
    title: str                  # what the menu shows
    formula: str
    filename: str
    cod: str
    shows: str                  # one line: what it is for
    detail: str                 # the paragraph the manual and tooltip use

    @property
    def path(self) -> Path:
        return folder() / self.filename

    @property
    def exists(self) -> bool:
        return self.path.is_file()


CATALOGUE: tuple[Example, ...] = (
    Example(
        key="quartz",
        title="Quartz",
        formula="SiO2",
        filename="quartz_SiO2_cod9013321.cif",
        cod="9013321",
        shows="A coordination number nobody argues about",
        detail="Silicon sits in a tetrahedron of four oxygens at 1.605 to "
               "1.611 angstrom, and the next contact is far enough away that "
               "the answer four survives a threshold moved over more than a "
               "hundredfold range. This is what a plateau looks like when a "
               "structure has nothing to hide, and it is the comparison every "
               "other example here is read against.",
    ),
    Example(
        key="eulytite",
        title="Eulytite",
        formula="Bi4(SiO4)3",
        filename="eulytite_Bi4SiO4_3_cod9012894.cif",
        cod="9012894",
        shows="Two sites in one structure, one settled and one not",
        detail="The silicon is as clean as quartz. The bismuth in the same "
               "crystal has three contacts near 2.16 angstrom and three near "
               "2.61, with no gap between the two groups that a cutoff could "
               "be placed in without choosing it. Both sites are in the same "
               "file and the same refinement, so the difference is not a "
               "question of data quality.",
    ),
    Example(
        key="senarmontite",
        title="Senarmontite",
        formula="Sb2O3",
        filename="senarmontite_Sb2O3_cod9009747.cif",
        cod="9009747",
        shows="One composition, two structures: the first",
        detail="Antimony here is six-coordinate, three short bonds at 1.978 "
               "angstrom and three long at 2.902. Open it beside valentinite, "
               "which is the same compound, and compare the coordination "
               "numbers the two give at the same threshold.",
    ),
    Example(
        key="valentinite",
        title="Valentinite",
        formula="Sb2O3",
        filename="valentinite_Sb2O3_cod9007587.cif",
        cod="9007587",
        shows="One composition, two structures: the second",
        detail="The same Sb2O3, in a different arrangement. The antimony is "
               "five-coordinate and its plateau is wider than senarmontite's. "
               "Two polymorphs of one compound need not report the same "
               "coordination number, and neither answer is wrong.",
    ),
    Example(
        key="cryolite",
        title="Cryolite",
        formula="Na3AlF6",
        filename="cryolite_Na3AlF6_cod9004097.cif",
        cod="9004097",
        shows="A fluoride, and a coordination number that is not safe to quote",
        detail="Nothing in the method is specific to oxygen, and this is "
               "where that matters: a 3.0 angstrom cutoff chosen for an oxide "
               "has no standing in a fluoride. Two of the three sites behave; "
               "the Na2 site has a plateau around a tenth of a decade wide, "
               "which means its coordination number is a property of the "
               "threshold rather than of the structure. FACET says so rather "
               "than picking one.",
    ),
    Example(
        key="bipo4",
        title="Bismuth phosphate",
        formula="BiPO4",
        filename="bismuth_phosphate_BiPO4_cod9008088.cif",
        cod="9008088",
        shows="The same ion in a different host",
        detail="The Bi(III) of eulytite again, with phosphorus competing for "
               "the same oxygens instead of silicon. The coordination number "
               "rises, the bonds lengthen, and the bond-valence vector "
               "shortens considerably. Useful for asking what belongs to the "
               "ion and what belongs to the compound.",
    ),
)

BY_KEY = {e.key: e for e in CATALOGUE}


def folder() -> Path:
    """Where the bundled structures are, frozen or from source.

    PyInstaller unpacks data under ``sys._MEIPASS`` keeping the paths it was
    given, so the same relative layout works in both cases.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "facet" / "data" / "examples"
    return Path(__file__).resolve().parent.parent / "data" / "examples"


def available() -> list[Example]:
    """The examples whose files are actually present.

    A build that failed to bundle them should show an empty menu rather than
    a list of entries that raise when clicked.
    """
    return [e for e in CATALOGUE if e.exists]
