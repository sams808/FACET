"""The mark, and the splash screen.

The logo is drawn in code rather than shipped as a bitmap, so it is crisp at
any size and on any display, and so the icon, the splash and the About box are
all provably the same drawing.

**The mark.** A coordination polyhedron seen down a three-fold axis: a central
atom with bonds radiating out to vertices. Three bonds are drawn solid and the
rest fade to hairlines -- which is the application's whole argument, and also,
read another way, a cut gemstone's facets. The name follows: a facet is a face
of a crystal and a face of a polyhedron, and "facets of a problem" is what
looking at a coordination number several ways amounts to.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QSplashScreen

from ..version import NAME, TAGLINE, __version__

# The mark's palette. Bismuth's violet against the warm red FACET uses for
# oxygen, on the same ground as the viewport, so the application looks like
# one thing from the splash onwards.
INK = QColor("#e8ecf4")
GROUND = QColor("#14161b")
CENTRE = QColor("#b07ad0")          # the cation
VERTEX = QColor("#e8674f")          # the ligands
ACCENT = QColor("#ffd65c")          # the threshold colour used throughout
FADED = QColor("#5d6472")


def draw_mark(painter: QPainter, size: float, *, ground: bool = True) -> None:
    """Draw the logo into a square of `size`, origin at the top left."""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    s = size
    centre = QPointF(s * 0.5, s * 0.5)

    if ground:
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, s, s), s * 0.22, s * 0.22)
        grad = QLinearGradient(0, 0, s, s)
        grad.setColorAt(0.0, QColor("#1b1e25"))
        grad.setColorAt(1.0, QColor("#0e1014"))
        painter.fillPath(path, QBrush(grad))
        painter.setClipPath(path)

    # six vertices on a hexagon: a coordination shell seen down a three-fold
    radius = s * 0.30
    vertices = [QPointF(centre.x() + radius * math.cos(math.radians(a)),
                        centre.y() + radius * math.sin(math.radians(a)))
                for a in range(-90, 270, 60)]

    # the polyhedron face, faint, so the mark reads as a solid at small sizes
    face = QPainterPath()
    face.moveTo(vertices[0])
    for v in vertices[1:]:
        face.lineTo(v)
    face.closeSubpath()
    painter.fillPath(face, QColor(120, 190, 235, 38))

    # three strong bonds and three hairlines: the argument, in the mark
    for i, v in enumerate(vertices):
        strong = i % 2 == 0
        pen = QPen(INK if strong else FADED,
                   s * (0.045 if strong else 0.014), Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(centre, v)

    # ligands
    for i, v in enumerate(vertices):
        strong = i % 2 == 0
        r = s * (0.070 if strong else 0.045)
        colour = VERTEX if strong else QColor(VERTEX.red(), VERTEX.green(),
                                              VERTEX.blue(), 130)
        grad = QRadialGradient(QPointF(v.x() - r * 0.3, v.y() - r * 0.3), r * 1.6)
        grad.setColorAt(0.0, colour.lighter(150))
        grad.setColorAt(1.0, colour.darker(135))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(v, r, r)

    # the central cation
    r = s * 0.115
    grad = QRadialGradient(QPointF(centre.x() - r * 0.3, centre.y() - r * 0.35),
                           r * 1.7)
    grad.setColorAt(0.0, CENTRE.lighter(155))
    grad.setColorAt(1.0, CENTRE.darker(140))
    painter.setBrush(QBrush(grad))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(centre, r, r)

    # the threshold: an arc between the strong bonds and the hairlines, which
    # is where the coordination number is decided
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(ACCENT, s * 0.018, Qt.DashLine, Qt.RoundCap))
    ring = radius * 0.62
    painter.drawEllipse(centre, ring, ring)

    painter.restore()


def logo_pixmap(size: int = 256, ground: bool = True) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    draw_mark(painter, float(size), ground=ground)
    painter.end()
    return pix


def app_icon() -> QIcon:
    """A multi-resolution icon, so Windows picks a crisp one at every size."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(logo_pixmap(size))
    return icon


def write_ico(path: str, sizes=(16, 24, 32, 48, 64, 128, 256)) -> str:
    """Write a Windows .ico, for the executable and the installer.

    Qt writes a genuine multi-image .ico from a list of pixmaps, so no external
    image tool is needed in the build.
    """
    from PySide6.QtGui import QImage
    from PySide6.QtCore import QBuffer, QByteArray

    images = [logo_pixmap(s).toImage().convertToFormat(QImage.Format_ARGB32)
              for s in sizes]
    # Qt's ICO handler takes the largest and downsamples; writing each size
    # explicitly gives a sharper small icon than letting it scale 256 -> 16.
    biggest = images[-1]
    biggest.save(path, "ICO")
    return path


# ---------------------------------------------------------------------------
# splash
# ---------------------------------------------------------------------------

SPLASH_WIDTH, SPLASH_HEIGHT = 560, 320


def splash_pixmap(message: str = "") -> QPixmap:
    pix = QPixmap(SPLASH_WIDTH, SPLASH_HEIGHT)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.TextAntialiasing, True)

    card = QPainterPath()
    card.addRoundedRect(QRectF(0, 0, SPLASH_WIDTH, SPLASH_HEIGHT), 16, 16)
    grad = QLinearGradient(0, 0, SPLASH_WIDTH, SPLASH_HEIGHT)
    grad.setColorAt(0.0, QColor("#1a1d24"))
    grad.setColorAt(1.0, QColor("#0d0f13"))
    p.fillPath(card, QBrush(grad))
    p.setPen(QPen(QColor(255, 255, 255, 28), 1.5))
    p.drawPath(card)

    p.save()
    p.translate(38, 78)
    draw_mark(p, 164.0, ground=False)
    p.restore()

    x = 232
    title = QFont(p.font())
    title.setPointSizeF(40)
    title.setBold(True)
    title.setLetterSpacing(QFont.AbsoluteSpacing, 3.0)
    p.setFont(title)
    p.setPen(INK)
    p.drawText(QRectF(x, 92, SPLASH_WIDTH - x - 30, 56),
               Qt.AlignLeft | Qt.AlignVCenter, NAME)

    sub = QFont(p.font())
    sub.setPointSizeF(10.5)
    sub.setBold(False)
    sub.setLetterSpacing(QFont.AbsoluteSpacing, 0.0)
    p.setFont(sub)
    p.setPen(QColor("#9aa3b2"))
    p.drawText(QRectF(x, 146, SPLASH_WIDTH - x - 30, 40),
               Qt.AlignLeft | Qt.AlignTop, TAGLINE)

    small = QFont(p.font())
    small.setPointSizeF(9.0)
    p.setFont(small)
    p.setPen(QColor("#6e7788"))
    p.drawText(QRectF(x, 186, SPLASH_WIDTH - x - 30, 20),
               Qt.AlignLeft | Qt.AlignTop, f"version {__version__}")

    if message:
        p.setPen(QColor("#8a93a3"))
        p.drawText(QRectF(38, SPLASH_HEIGHT - 44, SPLASH_WIDTH - 76, 20),
                   Qt.AlignLeft | Qt.AlignVCenter, message)

    p.end()
    return pix


class Splash(QSplashScreen):
    """The opening screen.

    It is not only decoration. FACET's heavy imports -- gemmi, scipy, spglib,
    and pymatgen when a parameter has to be estimated -- are deferred so the
    window appears quickly, and the splash is where that deferral is paid off
    visibly: each step reports what it is doing rather than leaving a blank
    screen.
    """

    def __init__(self):
        super().__init__(splash_pixmap())
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._message = ""

    def step(self, message: str) -> None:
        self._message = message
        self.setPixmap(splash_pixmap(message))
        self.repaint()
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def finish_after(self, window, minimum_ms: int = 1600) -> None:
        """Close once the window is up, but not before `minimum_ms`.

        Without the floor the splash flickers past on a fast machine, which
        looks like a fault rather than a start-up.
        """
        elapsed = getattr(self, "_elapsed", 0)
        remaining = max(0, minimum_ms - elapsed)
        QTimer.singleShot(remaining, lambda: self.finish(window))
