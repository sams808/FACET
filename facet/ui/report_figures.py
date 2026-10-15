"""Figures for the HTML report: the cutoff explorer of every site, and a
3D picture of every structure, rendered off screen and embedded.

The report itself is Qt-free (``facet.core.report``); this module adds
pictures to its HTML when the window asks for them. The explorer is the
same widget the window shows, grabbed at a fixed size, so the figure is
the picture a reader of the manual knows.
"""
from __future__ import annotations

import base64
from pathlib import Path

from PySide6.QtCore import QBuffer, QIODevice


def _png_base64(image) -> str:
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    return base64.b64encode(bytes(buffer.data())).decode()


def explorer_png(result, v_bond: float, theme=None, *, width: int = 900,
                 height: int = 300) -> str:
    """The cutoff explorer for one site, as a base64 PNG."""
    from .cutoff_explorer import CutoffExplorer
    w = CutoffExplorer()
    try:
        w.resize(width, height)
        if theme is not None:
            w.set_theme(theme)
        w.set_result(result)
        w.set_pin(v_bond)
        w.set_threshold(v_bond)
        return _png_base64(w.grab().toImage())
    finally:
        w.deleteLater()


def structure_png(structure, results, v_bond: float, v_list: float, theme=None,
                  *, width: int = 640, height: int = 480) -> str | None:
    """A ball-and-stick picture of the structure on the QPainter tier, as a
    base64 PNG; None when it cannot be built."""
    try:
        from ..gl.scene import build_scene
        from ..gl.view import StructureView
    except Exception:                                # noqa: BLE001
        return None
    view = StructureView()
    try:
        view.resize(width, height)
        if theme is not None:
            view.set_theme(theme)
        scene = build_scene(structure, results, v_bond=v_bond, v_list=v_list,
                            theme=theme)
        view.set_scene(scene, reframe=True)
        image = view.grab_image(width, height)
        if image is None or image.isNull():
            return None
        return _png_base64(image)
    except Exception:                                # noqa: BLE001
        return None
    finally:
        view.deleteLater()


def add_figures(html_path: str | Path, report, theme=None,
                *, structures: bool = True) -> Path:
    """Append the figures to a written HTML report: one explorer per site
    under a heading per structure, and the structure's picture."""
    path = Path(html_path)
    text = path.read_text(encoding="utf-8")
    parts = ["<h2>figures</h2>"]
    for s in report.structures:
        if s.error or not s.results:
            continue
        parts.append(f"<h3>{s.name}</h3>")
        if structures:
            try:
                from ..core import cif
                structure = cif.read(s.path)
                png = structure_png(structure, s.results, report.v_bond,
                                    report.v_list, theme)
                if png:
                    parts.append(f"<img src='data:image/png;base64,{png}' "
                                 f"style='max-width:100%'>")
            except Exception:                        # noqa: BLE001
                pass
        for r in s.results:
            png = explorer_png(r, report.v_bond, theme)
            parts.append(f"<h4>{r.label}</h4>"
                         f"<img src='data:image/png;base64,{png}' "
                         f"style='max-width:100%'>")
    block = "\n".join(parts)
    if "</body>" in text:
        text = text.replace("</body>", block + "\n</body>", 1)
    else:
        text += block
    path.write_text(text, encoding="utf-8")
    return path
