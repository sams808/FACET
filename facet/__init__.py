"""FACET -- coordination analysis from crystal structure files and from
molecular-dynamics models of glasses.

As a library::

    import facet

    c = facet.read("structure.cif")              # a crystal structure
    print(c.sites())                             # CN, valence sum, phi, plateau
    m = facet.model("glass.data", type_map={1: "O", 2: "Si", 3: "Na"})
    r = m.analyse("glass", formers=["Si"])       # Qn, speciation, angles, g(r)
    facet.basicity.composition("30Na2O-70SiO2")  # optical basicity

The names are those of :mod:`facet.library` (``docs/library_manual.md`` is the
manual); the engine underneath is :mod:`facet.core`. Nothing is imported
until a name is used: ``import facet`` stays instant, and neither the window
toolkit, pymatgen nor matplotlib is loaded unless a call needs it.
"""

from .version import NAME, TAGLINE, __version__

_LIBRARY = (
    "Crystal", "SiteView", "PlaneView", "Model", "Results", "Table",
    "read", "read_all", "model", "load", "example", "examples",
    "analyses", "presets", "request", "options", "report_on", "compare",
    "parameters",
    "bond_valence", "cutoff_distance", "read_volume", "selfcheck",
    "table", "row_of", "plot", "V_BOND", "V_LIST", "GLASS_TYPE_MAP",
)

__all__ = ["__version__", "NAME", "TAGLINE", "basicity", *_LIBRARY]


# A name here must not be that of a sub-package (facet.report, facet.site,
# facet.md, facet.check are the commands): importing the sub-package would
# put the module in the function's place. tests/test_library.py holds this.
def __getattr__(name):
    import importlib

    if name in _LIBRARY:
        return getattr(importlib.import_module(".library", __name__), name)
    if name in ("basicity", "library"):
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
