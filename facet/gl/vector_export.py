"""Vector export of the 3D view: SVG and PDF.

A raster export of a structure is a picture of a picture. Put it in a paper at
600 dpi and the atom edges are soft; scale it for a poster and they are visibly
square. A vector export has no resolution at all -- an atom is a filled circle
with a gradient, a bond is a filled quadrilateral, and they stay sharp at any
size a journal or a printer asks for.

FACET can do this without a second renderer, because it already has one that
draws with QPainter: the software tier, written so the application opens on a
machine with no graphics hardware. A QPainter pointed at a QSvgGenerator or a
QPdfWriter emits drawing commands instead of pixels, so the same code that keeps
the program usable without a graphics card is also what produces the figure for
the paper. The depth sorting it does for transparency is exactly the painter's
algorithm an SVG needs, since SVG has no depth buffer either.

What this cannot do is carry over the OpenGL tier's ambient occlusion and
outlines: those are screen-space effects with no vector equivalent. Nor does it
draw a displacement ellipsoid's octant boundaries the way the OpenGL tier does
-- that is surface shading, and here they become the ellipse's two principal
axes drawn across it, which says the same thing with the lines a vector format
has. An exported
figure is therefore the flat-shaded version of the view, which is what a journal
figure usually wants anyway. Say so rather than let it be a surprise --
:func:`describe_differences` is there to be shown to the user.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

from PySide6.QtCore import QMarginsF, QRectF, QSize, QSizeF
from PySide6.QtGui import QColor, QPageSize, QPainter, QPdfWriter

from .camera import Camera
from .painter import PainterRenderer
from .scene import Scene
from ..core import theme as theme_mod


class Background(Enum):
    THEME = "as on screen"
    WHITE = "white"
    NONE = "transparent"


# A4 width less 2 cm margins each side, which is the usual single-column-and-a-half
# figure. Only a default; the caller may ask for anything.
DEFAULT_WIDTH_MM = 170.0


def describe_differences() -> list[str]:
    """How a vector export differs from the screen, in plain terms."""
    return [
        "Atoms, bonds, polyhedra, planes, the cell and the labels are drawn as "
        "vector shapes, so the figure is sharp at any size.",
        "Ambient occlusion and contact outlines are screen-space effects and "
        "have no vector equivalent, so the export is flat-shaded.",
        "Everything is drawn back to front, which is how transparency comes out "
        "right without a depth buffer -- an SVG has none.",
    ]


def _render(painter: QPainter, scene: Scene, camera: Camera,
            width: float, height: float, theme=None, overlay=None,
            selected: int | None = None) -> None:
    """Draw one frame of the structure onto an open painter."""
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    renderer = PainterRenderer()
    if theme is not None:
        renderer.apply_theme(theme)
    renderer.render(painter, scene, camera, int(width), int(height),
                    selected=selected)
    if overlay is not None:
        overlay(painter)


def _fill(painter: QPainter, width: float, height: float, background: Background,
          theme) -> None:
    if background is Background.NONE:
        return
    if background is Background.WHITE:
        colour = QColor(255, 255, 255)
    else:
        rgb = getattr(theme, "background", theme_mod.FALLBACK_BACKGROUND)
        colour = QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    painter.fillRect(QRectF(0, 0, width, height), colour)


def save_svg(path, scene: Scene, camera: Camera, width: int, height: int,
             theme=None, overlay=None, selected: int | None = None,
             background: Background = Background.THEME,
             scale: float = 1.0, title: str = "") -> Path:
    """Write the view as SVG.

    ``width`` and ``height`` are the logical size the scene is composed at -- the
    widget's own size, so the figure has the layout that was on screen -- and
    ``scale`` multiplies the output. Scaling the painter rather than re-composing
    at a larger size is what keeps the label positions and sizes in the same
    places relative to the structure; re-composing would reflow them.
    """
    from PySide6.QtSvg import QSvgGenerator

    path = Path(path)
    out_width = int(round(width * scale))
    out_height = int(round(height * scale))

    generator = QSvgGenerator()
    generator.setFileName(str(path))
    generator.setSize(QSize(out_width, out_height))
    generator.setViewBox(QRectF(0, 0, out_width, out_height))
    generator.setTitle(title or "FACET structure")
    generator.setDescription("; ".join(describe_differences()))

    painter = QPainter(generator)
    try:
        painter.scale(scale, scale)
        _fill(painter, width, height, background, theme)
        _render(painter, scene, camera, width, height, theme, overlay, selected)
    finally:
        painter.end()
    return path


def save_pdf(path, scene: Scene, camera: Camera, width: int, height: int,
             theme=None, overlay=None, selected: int | None = None,
             background: Background = Background.THEME,
             width_mm: float = DEFAULT_WIDTH_MM, resolution: int = 600,
             title: str = "") -> Path:
    """Write the view as a single-page PDF at a stated physical width.

    The page is sized to the view's aspect ratio, so the figure fills it with no
    margin to crop off. A PDF is what most journals ask for and what most
    drawing programs open without argument.
    """
    path = Path(path)
    aspect = height / max(width, 1)
    height_mm = width_mm * aspect

    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QSizeF(width_mm, height_mm),
                                 QPageSize.Millimeter))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0))
    writer.setResolution(int(resolution))
    writer.setTitle(title or "FACET structure")

    painter = QPainter(writer)
    try:
        # device pixels across the page, over the logical width the scene is
        # composed at
        device_width = width_mm / 25.4 * writer.resolution()
        scale = device_width / max(width, 1)
        painter.scale(scale, scale)
        _fill(painter, width, height, background, theme)
        _render(painter, scene, camera, width, height, theme, overlay, selected)
    finally:
        painter.end()
    return path


FILE_FILTER = "SVG (*.svg);;PDF (*.pdf)"


def save(path, scene: Scene, camera: Camera, width: int, height: int,
         **options) -> Path:
    """Write SVG or PDF, chosen from the file name."""
    suffix = Path(path).suffix.lower()
    if suffix == ".svg":
        options.pop("width_mm", None)
        options.pop("resolution", None)
        return save_svg(path, scene, camera, width, height, **options)
    if suffix == ".pdf":
        options.pop("scale", None)
        return save_pdf(path, scene, camera, width, height, **options)
    raise ValueError(
        f"{Path(path).name}: FACET writes vector figures as .svg or .pdf")
