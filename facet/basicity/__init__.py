"""Optical basicity from Python and from the command line.

The calculations behind FACET's Basicity tab, Site rows, MD analysis and
basicity maps, without the window::

    from facet import basicity as fb

    fb.composition("30Na2O-70SiO2").value          # Lambda of a composition
    s = fb.structure("albite.cif", formers=["Si", "Al"])
    print(s)                                       # every anion site's lambda
    g = fb.frame("glass.data", formers=["Si", "B"])
    g.values                                       # lambda per atom
    g.field("Ca").to_cube("lambda_Ca.cube")        # the map, for VESTA
    fb.plot.histogram(g, save="lambda.png")        # a quick figure

and the same from a terminal::

    python -m facet.basicity composition "30Na2O-70SiO2"
    python -m facet.basicity structure albite.cif --formers Si,Al
    python -m facet.basicity plot slice glass.data --probe Ca -o slice.png
    python -m facet.basicity view glass.data

See ``docs/basicity_library.md`` for every call and command.
"""
from . import api
from .api import (CRYSTAL_SUFFIXES, DEFAULT_SCALE, PROBE_IONS, SCALE_KEYS,
                  CompositionError, CompositionResult, FieldResult,
                  FrameResult, StructureResult, basicity_from_band,
                  basicity_from_polarisability, composition, frame, frames,
                  is_md_file, linkage_rows, linkage_table, load, moderator,
                  oxide, oxide_polarisability, probe_band, probe_wavelength,
                  scales, series, structure, text_table)

__all__ = [
    "CRYSTAL_SUFFIXES", "DEFAULT_SCALE", "PROBE_IONS", "SCALE_KEYS",
    "CompositionError", "CompositionResult", "FieldResult", "FrameResult",
    "StructureResult", "api", "basicity_from_band",
    "basicity_from_polarisability", "composition", "frame", "frames",
    "is_md_file", "linkage_rows", "linkage_table", "load", "moderator",
    "oxide", "oxide_polarisability", "plot", "probe_band", "probe_wavelength",
    "scales", "series", "structure", "text_table",
]


def __getattr__(name):
    # the figures import matplotlib, so they load on first use only
    if name == "plot":
        import importlib

        return importlib.import_module(".plot", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
