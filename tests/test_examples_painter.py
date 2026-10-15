"""Every bundled example opens in the window and draws on the QPainter tier
(the no-GPU path): the image rendered from the fallback renderer holds
atoms, not a blank. The GL tiers need a GPU and are checked by hand; this
is the tier a machine without one gets, and the one a CI machine can
check at every commit.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import dispose

from facet.core import examples, prefs


@pytest.fixture(scope="module")
def window(qapp):
    prefs.set("tour_offered", True)
    from facet.ui.preview import PreviewWindow
    w = PreviewWindow()
    yield w
    w.close()
    dispose(w)


def _pixels(image):
    ptr = image.constBits()
    arr = np.frombuffer(ptr, np.uint8).reshape(image.height(), image.width(), 4)
    return arr


@pytest.mark.parametrize("example", examples.available(),
                         ids=lambda ex: ex.path.name)
def test_the_example_draws_on_the_painter_tier(qapp, window, example):
    window.project.clear()
    window.load(str(example.path))
    qapp.processEvents()
    scene = window.view.scene
    assert scene is not None and scene.n_atoms > 0
    # the window's own scene, drawn by a view on the painter tier: the
    # path a machine without OpenGL takes (a never-initialised GL widget
    # must not be forced into the fallback and then painted by events)
    from facet.gl.view import StructureView
    view = StructureView()
    try:
        view.resize(320, 240)
        view._enter_fallback("test: painter tier forced")
        if window.theme is not None:
            view.set_theme(window.theme)
        view.set_scene(scene, reframe=True)
        image = view.grab_image(320, 240, supersample=1)
    finally:
        view.deleteLater()
    px = _pixels(image)
    opaque = px[..., 3] > 0
    assert opaque.any(), "nothing was drawn"
    colours = {tuple(c) for c in px[opaque][:, :3][::7]}
    assert len(colours) > 4, "the drawing is a flat colour"
    # the site list and the Site tab are filled for the first cation
    assert window.results
    assert window.project.current.structure.sites
