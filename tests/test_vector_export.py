"""Vector export of the 3D view.

The thing to verify is that the output really is vector: an SVG full of shapes,
not an SVG with one embedded bitmap in it. That failure mode produces a file
that opens correctly, looks right, and is useless for the purpose -- so it is
checked by reading the file back and looking for drawing commands and for the
absence of an embedded raster.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SAMPLE = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs"
              r"\1526458_Bi2O3.cif")

pytestmark = pytest.mark.skipif(not SAMPLE.is_file(),
                                reason="the sample structure is not present")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def scene_and_camera(qapp):
    from facet.core import cif
    from facet.gl.camera import Camera
    from facet.gl.scene import build_scene

    structure = cif.read(SAMPLE)
    scene = build_scene(structure, polyhedron_sites=structure.cation_sites,
                        cell_range=(2, 1, 1))
    camera = Camera()
    camera.frame(scene.center, scene.radius)
    return scene, camera, structure


def test_svg_is_made_of_shapes_and_not_an_embedded_bitmap(scene_and_camera,
                                                         tmp_path):
    """The failure mode that looks fine and is useless.

    An exporter that rendered to a pixmap and wrapped it in an <svg> element
    would pass any test that only checked the file opens. So the test looks for
    drawing commands, and for the absence of an embedded raster.
    """
    from facet.gl import vector_export as V

    scene, camera, structure = scene_and_camera
    path = V.save_svg(tmp_path / "structure.svg", scene, camera, 640, 480,
                      title=structure.name)
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    assert "<svg" in text
    assert "<path" in text or "<polygon" in text or "<ellipse" in text
    assert "image/png" not in text, "the figure is an embedded bitmap"
    assert "base64" not in text, "the figure is an embedded bitmap"
    assert len(text) > 20_000, "too small to contain a whole structure"
    assert structure.name in text


def test_the_svg_has_a_gradient_for_every_sphere(scene_and_camera, tmp_path):
    """Atoms are shaded with a radial gradient, which survives as a gradient."""
    from facet.gl import vector_export as V

    scene, camera, _ = scene_and_camera
    path = V.save_svg(tmp_path / "gradients.svg", scene, camera, 500, 400)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    assert "radialGradient" in text


def test_pdf_is_a_pdf(scene_and_camera, tmp_path):
    from facet.gl import vector_export as V

    scene, camera, _ = scene_and_camera
    path = V.save_pdf(tmp_path / "structure.pdf", scene, camera, 640, 480)
    data = Path(path).read_bytes()
    assert data[:5] == b"%PDF-"
    assert b"%%EOF" in data[-4096:]
    assert len(data) > 10_000


def test_the_pdf_page_matches_the_view_aspect_ratio(scene_and_camera, tmp_path):
    """No margin to crop: the page is the figure."""
    from facet.gl import vector_export as V

    scene, camera, _ = scene_and_camera
    for width, height in ((640, 480), (900, 300), (400, 800)):
        path = V.save_pdf(tmp_path / f"page_{width}x{height}.pdf", scene,
                          camera, width, height, width_mm=100.0)
        data = Path(path).read_bytes()
        assert data[:5] == b"%PDF-"
        # the MediaBox records the page size in points
        import re

        match = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+"
                          rb"([\d.]+)\s+([\d.]+)\s*\]", data)
        assert match, "no MediaBox in the PDF"
        box = [float(v) for v in match.groups()]
        page_width = box[2] - box[0]
        page_height = box[3] - box[1]
        assert page_height / page_width == pytest.approx(height / width,
                                                         rel=0.02)


def test_scale_changes_the_svg_size_but_not_the_layout(scene_and_camera,
                                                      tmp_path):
    """Scaling must not reflow the labels relative to the structure."""
    import re

    from facet.gl import vector_export as V

    scene, camera, _ = scene_and_camera
    sizes = {}
    for scale in (1.0, 2.0):
        path = V.save_svg(tmp_path / f"scale_{scale}.svg", scene, camera,
                          400, 300, scale=scale)
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        # the viewBox, not the width attribute: Qt writes the width in
        # millimetres, and a regex for digits would match the viewBox anyway
        match = re.search(r'viewBox="0 0 (\d+) (\d+)"', text)
        assert match, "no viewBox on the svg element"
        sizes[scale] = (int(match.group(1)), int(match.group(2)))

    assert sizes[2.0][0] == pytest.approx(2 * sizes[1.0][0], abs=2)
    assert sizes[2.0][1] == pytest.approx(2 * sizes[1.0][1], abs=2)


def test_the_background_choice_is_honoured(scene_and_camera, tmp_path):
    from facet.gl import vector_export as V

    from facet.core import theme as theme_mod

    scene, camera, _ = scene_and_camera
    # a theme whose ground is not white, or THEME and WHITE are the same request
    theme = theme_mod.dark()
    texts = {}
    for background in V.Background:
        path = V.save_svg(tmp_path / f"bg_{background.name}.svg", scene,
                          camera, 300, 220, background=background, theme=theme)
        texts[background] = Path(path).read_text(encoding="utf-8",
                                                 errors="replace")

    # white must appear as a full-bleed white fill; transparent must not fill
    assert "#ffffff" in texts[V.Background.WHITE].lower()
    assert texts[V.Background.NONE] != texts[V.Background.WHITE]
    assert texts[V.Background.THEME] != texts[V.Background.WHITE]
    # the theme's own ground, not a guess at one
    ground = "#%02x%02x%02x" % tuple(int(c * 255) for c in theme.background)
    assert ground in texts[V.Background.THEME].lower()
    # and with no theme at all the fallback is stated once, in core.theme
    plain = V.save_svg(tmp_path / "bg_default.svg", scene, camera, 300, 220,
                       background=V.Background.THEME)
    fallback = "#%02x%02x%02x" % tuple(
        int(c * 255) for c in theme_mod.FALLBACK_BACKGROUND)
    assert fallback in Path(plain).read_text(encoding="utf-8",
                                            errors="replace").lower()


def test_save_chooses_the_format_from_the_name(scene_and_camera, tmp_path):
    from facet.gl import vector_export as V

    scene, camera, _ = scene_and_camera
    svg = V.save(tmp_path / "pick.svg", scene, camera, 320, 240)
    assert Path(svg).read_bytes()[:5] != b"%PDF-"
    pdf = V.save(tmp_path / "pick.pdf", scene, camera, 320, 240)
    assert Path(pdf).read_bytes()[:5] == b"%PDF-"


def test_an_unknown_extension_is_refused(scene_and_camera, tmp_path):
    from facet.gl import vector_export as V

    scene, camera, _ = scene_and_camera
    with pytest.raises(ValueError, match="svg or .pdf"):
        V.save(tmp_path / "nope.eps", scene, camera, 320, 240)


def test_an_empty_scene_still_writes_a_file(qapp, tmp_path):
    from facet.gl.camera import Camera
    from facet.gl.scene import Scene
    from facet.gl import vector_export as V

    path = V.save_svg(tmp_path / "empty.svg", Scene(), Camera(), 200, 150)
    assert Path(path).is_file()
    assert "<svg" in Path(path).read_text(encoding="utf-8", errors="replace")


def test_the_differences_are_stated_rather_than_hidden():
    """A vector figure is flat-shaded; the user is told, not surprised."""
    from facet.gl import vector_export as V

    notes = V.describe_differences()
    assert notes
    joined = " ".join(notes).lower()
    assert "ambient occlusion" in joined
    assert "flat" in joined
    assert "vector" in joined


def test_the_view_exports_through_its_own_method(qapp, tmp_path):
    from facet.core import cif
    from facet.gl.scene import build_scene
    from facet.gl.view import StructureView

    structure = cif.read(SAMPLE)
    widget = StructureView()
    widget.resize(480, 360)
    widget.set_scene(build_scene(structure,
                                 polyhedron_sites=structure.cation_sites))
    try:
        svg = widget.save_vector(tmp_path / "from_view.svg")
        assert Path(svg).is_file()
        text = Path(svg).read_text(encoding="utf-8", errors="replace")
        assert "<path" in text
        assert "image/png" not in text

        pdf = widget.save_vector(tmp_path / "from_view.pdf")
        assert Path(pdf).read_bytes()[:5] == b"%PDF-"
    finally:
        widget.close()


def test_exporting_with_nothing_loaded_is_refused(qapp, tmp_path):
    from facet.gl.view import StructureView

    widget = StructureView()
    try:
        with pytest.raises(ValueError, match="nothing to export"):
            widget.save_vector(tmp_path / "nothing.svg")
    finally:
        widget.close()


def test_the_export_includes_the_labels_when_they_are_on(qapp, tmp_path):
    """The overlay is part of the figure, not just of the screen."""
    from facet.core import cif
    from facet.gl import labels as labels_mod
    from facet.gl.scene import build_scene
    from facet.gl.view import StructureView

    structure = cif.read(SAMPLE)
    widget = StructureView()
    widget.resize(520, 400)
    scene = build_scene(structure)
    widget.set_scene(scene)
    widget.set_label_context(structure, None)

    try:
        settings = labels_mod.LabelSettings()
        settings.atom = labels_mod.AtomLabel.ELEMENT
        settings.atom_scope = labels_mod.LabelScope.ALL
        widget.set_labels(settings)
        with_labels = Path(widget.save_vector(
            tmp_path / "labelled.svg")).read_text(encoding="utf-8",
                                                  errors="replace")

        settings.atom = labels_mod.AtomLabel.NONE
        widget.set_labels(settings)
        without = Path(widget.save_vector(
            tmp_path / "plain.svg")).read_text(encoding="utf-8",
                                               errors="replace")
        assert len(with_labels) > len(without), "the labels were not exported"
    finally:
        widget.close()
