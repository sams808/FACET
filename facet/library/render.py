"""A picture of a crystal structure, drawn as the FACET window draws it.

    c.render("quartz.png")                               # 1600 x 1200 px
    c.render("quartz.svg", style="polyhedral", along="c")
    c.render("cell.png", cell_range=(2, 2, 1), polyhedra=["Si"])

This is the one part of the library that needs the window toolkit (Qt, which
FACET installs): the view is built and never shown. PNG pictures are painted
by FACET's software renderer, so no graphics card is involved; SVG and PDF
keep every atom and bond as a shape.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

STYLES = ("ball-and-stick", "polyhedral", "space-filling", "stick", "wireframe",
          "displacement ellipsoids")


def _application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(["facet"])


def _direction(crystal, along) -> np.ndarray | None:
    """A viewing direction: 'a', 'b', 'c', their reciprocals 'a*', 'b*',
    'c*', or three Cartesian numbers."""
    if along is None:
        return None
    if isinstance(along, str):
        cell = crystal.cell
        vectors = np.asarray(cell.to_cartesian(np.eye(3)), float)
        key = along.strip().lower()
        if key in ("a", "b", "c"):
            return vectors["abc".index(key)]
        if key in ("a*", "b*", "c*"):
            i = "abc".index(key[0])
            return np.cross(vectors[(i + 1) % 3], vectors[(i + 2) % 3])
        raise ValueError(f"along {along!r}: a, b, c, a*, b*, c* or a vector")
    vector = np.asarray(along, float).reshape(3)
    return vector


def render(crystal, path, *, size: Sequence[int] = (1600, 1200),
           style: str = "ball-and-stick", polyhedra=True, cell: bool = True,
           cell_range: Sequence[int] = (1, 1, 1), along=None,
           background: str = "white") -> Path:
    """Draw ``crystal`` to ``path`` (.png, .svg or .pdf).

    ``style``: one of :data:`STYLES`. ``polyhedra``: True draws the
    coordination polyhedron of every cation site, a list of site labels
    those of the sites named, False none. ``cell_range``: how many cells
    along a, b and c. ``along``: the viewing direction ('a', 'b', 'c', 'a*',
    'b*', 'c*' or a Cartesian vector); the default is the window's opening
    view. ``background``: 'white', 'theme' or 'transparent' (SVG and PDF).
    """
    from ..gl import vector_export
    from ..gl.scene import Style, build_scene
    from ..gl.view import StructureView

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".png", ".svg", ".pdf"):
        raise ValueError(f"{path.name}: a picture is written as .png, .svg or .pdf")
    try:
        chosen = Style(style)
    except ValueError:
        raise ValueError(f"style {style!r}: one of {', '.join(STYLES)}") from None
    structure = crystal.structure
    if polyhedra is True:
        sites = list(structure.cation_sites)
    elif not polyhedra:
        sites = []
    else:
        sites = []
        for label in polyhedra:
            index = structure.site_by_label(str(label))
            if index is None:
                raise KeyError(f"no site labelled {label!r}")
            sites.append(index)
    width, height = (int(v) for v in size)

    grounds = {"white": vector_export.Background.WHITE,
               "theme": vector_export.Background.THEME,
               "transparent": vector_export.Background.NONE}
    if background not in grounds:
        raise ValueError("background: 'white', 'theme' or 'transparent'")

    from ..core import theme as theme_mod

    app = _application()
    view = StructureView()
    theme = theme_mod.Theme()
    try:
        # the vector export is laid out at the widget's logical size
        view.resize(max(width // 2, 200), max(height // 2, 150))
        # a view that is never shown has no OpenGL context: it is painted by
        # the software renderer, which is also what a machine without a
        # graphics card uses
        view._enter_fallback("picture drawn by the library")
        view.set_theme(theme)
        scene = build_scene(structure, crystal.results(), style=chosen,
                            v_bond=crystal.v_bond, v_list=crystal.v_list,
                            polyhedron_sites=sites, show_cell=bool(cell),
                            cell_range=tuple(int(n) for n in cell_range),
                            params=crystal.params)
        view.set_scene(scene)
        direction = _direction(crystal, along)
        if direction is not None:
            view.view_along(direction)
        if suffix == ".png":
            from PySide6.QtGui import QColor, QImage, QPainter

            drawn = view.grab_image(width, height, supersample=1)
            if drawn is None or drawn.isNull():
                raise OSError(f"{path}: the picture could not be drawn")
            image = QImage(drawn.width(), drawn.height(),
                           QImage.Format_ARGB32_Premultiplied)
            if background == "transparent":
                image.fill(0)
            elif background == "theme":
                r, g, b = (round(255 * c) for c in theme.background[:3])
                image.fill(QColor(r, g, b))
            else:
                image.fill(QColor(255, 255, 255))
            painter = QPainter(image)
            try:
                painter.drawImage(0, 0, drawn)
            finally:
                painter.end()
            if not image.save(str(path)):
                raise OSError(f"{path}: the picture could not be written")
        else:
            view.save_vector(path, background=grounds[background],
                             title=crystal.name)
    finally:
        view.close()
        view.deleteLater()
        app.processEvents()
    return path
