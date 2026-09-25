"""FACET -- coordination analysis from crystal structure files.

The public surface is :mod:`facet.core`. Nothing here imports Qt, pymatgen or
matplotlib at module level: the engine must stay importable, and fast to
import, without any of them.
"""

from .version import NAME, TAGLINE, __version__

__all__ = ["__version__", "NAME", "TAGLINE"]
