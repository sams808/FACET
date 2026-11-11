"""Application start-up.

The order matters. Qt's surface format has to be chosen before the
QApplication exists, the splash has to appear before anything slow is imported,
and the heavy imports have to happen while the splash is up rather than behind a
blank screen.

FACET's engine imports in about 0.3 s because nothing heavy is imported at
module level. The splash is therefore not hiding a slow start -- it is a
deliberate few seconds, held to a minimum duration so it does not flicker past
and read as a fault.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ..version import NAME, __version__


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    from ..gl import caps as caps_mod

    # before the QApplication: a format requested afterwards is ignored
    QSurfaceFormat.setDefaultFormat(caps_mod.request_format())

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(NAME)
    app.setApplicationDisplayName(NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("FACET")

    from .branding import Splash, app_icon

    app.setWindowIcon(app_icon())

    # The window's colours come from the theme, and the default theme is a white
    # ground. Applied to the application rather than to the window because menus
    # and tool tips are top-level windows of their own and inherit nothing from
    # it. Done before the window is built so that widgets measure themselves
    # with the padding they will actually have.
    from ..core import theme as theme_mod
    from . import chrome

    chrome.apply(app, theme_mod.Theme())

    started = time.perf_counter()
    splash = Splash()
    splash.show()
    app.processEvents()

    splash.step("Loading the crystallographic engine…")
    from ..core import cif, coordination  # noqa: F401  (warm the imports)

    splash.step("Preparing the renderer…")
    from .preview import PreviewWindow

    splash.step("Reading bond-valence parameters…")
    from ..core import bv

    bv.DEFAULT.get("Si", 4, "O")

    path = next((a for a in argv[1:] if not a.startswith("-")), None)
    if path:
        splash.step(f"Opening {Path(path).name}…")

    window = PreviewWindow(path)
    window.setWindowIcon(app_icon())
    window.show()
    if "--color-by-basicity" in argv[1:]:
        _colour_by_basicity(window)

    splash._elapsed = int((time.perf_counter() - started) * 1000)
    splash.finish_after(window, minimum_ms=1200)
    return app.exec()


def _colour_by_basicity(window, tries: int = 240) -> None:
    """Open with the atoms coloured by microscopic optical basicity: what
    ``python -m facet.basicity view FILE`` asks for. A crystal's colour mode
    is set at once (with the anion sites, which carry the lambda, shown); an
    MD model's when its Highlight page exists, which is after the read, so
    this looks again every half second for two minutes."""
    from PySide6.QtCore import QTimer

    from ..core import theme as theme_mod
    from .md_highlight import COLOUR_MODES

    if window.project.current is not None and not window.models:
        window.anion_check.setChecked(True)
        box = window.theme_panel.mode_box
        at = box.findData(theme_mod.ColorMode.BASICITY)
        if at >= 0:
            box.setCurrentIndex(at)
            window.theme_panel._apply_mode()    # setCurrentIndex is silent
        return
    waiting = False
    for controller in window.models:
        page = controller.highlight
        if page is None:
            waiting = True
        else:
            page.colour.setCurrentIndex(
                COLOUR_MODES.index("microscopic basicity"))
    if (waiting or not window.models) and tries > 0:
        QTimer.singleShot(500, lambda: _colour_by_basicity(window, tries - 1))


if __name__ == "__main__":
    raise SystemExit(main())
