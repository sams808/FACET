"""FACET as a Python library: everything the window computes, without it.

    import facet

    c = facet.read("structure.cif")              # a crystal structure
    m = facet.model("glass.data", type_map=...)  # an MD model
    facet.basicity.composition("30Na2O-70SiO2")  # optical basicity
    facet.plot.cn_function(c.site("Bi1"))        # quick figures

Every name below is also reachable as ``facet.<name>``. The crystal side is
:mod:`facet.library.crystal`, the model side :mod:`facet.library.model`;
``docs/library_manual.md`` is the manual.
"""
from .crystal import (GLASS_TYPE_MAP, V_BOND, V_LIST, Crystal, PlaneView,
                      SiteView, bond_valence, compare, cutoff_distance, example,
                      examples, parameters, read, read_all, read_volume, report_on,
                      selfcheck)
from .model import (Model, Results, analyses, load, model, options, presets,
                    request)
from .tables import Table, row_of, table_of as table

__all__ = [
    "Crystal", "SiteView", "PlaneView", "Model", "Results", "Table",
    "read", "read_all", "model", "load", "example", "examples",
    "analyses", "presets", "request", "options", "report_on", "compare",
    "parameters",
    "bond_valence", "cutoff_distance", "read_volume", "selfcheck",
    "table", "row_of", "plot", "V_BOND", "V_LIST", "GLASS_TYPE_MAP",
]


def __getattr__(name):
    # the figures import matplotlib, so they load on first use only
    if name == "plot":
        import importlib

        return importlib.import_module(".plot", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
