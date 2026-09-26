"""Stereo viewing.

Two images, one per eye, combined into something a pair of eyes can fuse. The
per-eye rendering is entirely the camera's business -- :meth:`Camera.for_eye`
hands back a displaced camera and every render path draws it as usual -- so this
module only combines the results.

Four ways, because which one works depends on what you have to hand:

* **anaglyph** with red-cyan glasses, the only method needing no special screen;
* **greyscale anaglyph**, which loses the colours but removes the retinal
  rivalry that makes a strongly coloured anaglyph uncomfortable;
* **side by side**, for a stereoscope or parallel free-viewing;
* **cross-eyed**, the same pair swapped, which many people find easier to fuse
  unaided.

The combination is done on the pixels with numpy rather than in a shader. That
is deliberate: it is identical on all three render tiers including the software
fallback, it works the same for an exported figure as for the screen, and it is
testable without a GL context. Stereo is a mode you turn on to look at
something, not something that has to hold sixty frames a second.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from PySide6.QtGui import QImage


class Mode(Enum):
    OFF = "off"
    ANAGLYPH = "anaglyph (red-cyan)"
    ANAGLYPH_GREY = "anaglyph, greyscale (red-cyan)"
    SIDE_BY_SIDE = "side by side (parallel)"
    CROSS_EYED = "side by side (crossed)"

    @property
    def needs_two_eyes(self) -> bool:
        return self is not Mode.OFF

    @property
    def is_anaglyph(self) -> bool:
        return self in (Mode.ANAGLYPH, Mode.ANAGLYPH_GREY)


# Luminance weights. Rec. 709, the same ones a display uses, so a greyscale
# anaglyph keeps the relative brightness the colour image had.
_LUMA = (0.2126, 0.7152, 0.0722)


def _array(image: QImage) -> np.ndarray:
    """An (h, w, 4) uint8 view of a QImage, as RGBA."""
    converted = image.convertToFormat(QImage.Format_RGBA8888)
    width, height = converted.width(), converted.height()
    pointer = converted.constBits()
    raw = np.frombuffer(memoryview(pointer), np.uint8, count=height * width * 4)
    return raw.reshape(height, width, 4).copy()


def _image(array: np.ndarray) -> QImage:
    """A QImage from an (h, w, 4) uint8 RGBA array.

    The array is kept alive on the image, because QImage does not copy the buffer
    it is given -- returning without that reference yields an image whose pixels
    are freed memory, which reads as noise or a crash rather than as an error.
    """
    array = np.ascontiguousarray(array, np.uint8)
    height, width = array.shape[:2]
    image = QImage(array.data, width, height, 4 * width,
                   QImage.Format_RGBA8888)
    image = image.copy()          # own the pixels outright
    return image


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * _LUMA[0] + rgb[..., 1] * _LUMA[1]
            + rgb[..., 2] * _LUMA[2])


def anaglyph(left: QImage, right: QImage, greyscale: bool = False) -> QImage:
    """Red-cyan anaglyph: red from the left eye, green and blue from the right.

    In colour, a saturated red object is nearly invisible to the right eye and a
    cyan one to the left, so the pair will not fuse and the image shimmers. The
    greyscale form gives both eyes the same luminance and avoids that entirely,
    at the cost of the colours -- which for judging a polyhedron's shape is often
    the better trade.
    """
    a, b = _array(left), _array(right)
    if a.shape != b.shape:
        raise ValueError(
            f"the two eyes must be the same size: {a.shape[1]}x{a.shape[0]} "
            f"and {b.shape[1]}x{b.shape[0]}")

    out = np.zeros_like(a)
    if greyscale:
        out[..., 0] = np.clip(_luminance(a.astype(np.float32)), 0, 255)
        grey = np.clip(_luminance(b.astype(np.float32)), 0, 255)
        out[..., 1] = grey
        out[..., 2] = grey
    else:
        out[..., 0] = a[..., 0]
        out[..., 1] = b[..., 1]
        out[..., 2] = b[..., 2]
    # opaque wherever either eye drew something
    out[..., 3] = np.maximum(a[..., 3], b[..., 3])
    return _image(out)


def side_by_side(left: QImage, right: QImage, crossed: bool = False,
                 gap: int = 8, gap_color=(0, 0, 0)) -> QImage:
    """The pair placed next to each other, at the original size of each.

    ``crossed`` swaps them, which is what cross-eyed free-viewing needs. The gap
    is there to give the eyes a seam to lock onto; without one the two halves
    read as a single wide picture.
    """
    a, b = _array(left), _array(right)
    if a.shape != b.shape:
        raise ValueError(
            f"the two eyes must be the same size: {a.shape[1]}x{a.shape[0]} "
            f"and {b.shape[1]}x{b.shape[0]}")
    if crossed:
        a, b = b, a

    height, width = a.shape[:2]
    gap = max(int(gap), 0)
    out = np.zeros((height, width * 2 + gap, 4), np.uint8)
    out[:, :, 0] = gap_color[0]
    out[:, :, 1] = gap_color[1]
    out[:, :, 2] = gap_color[2]
    out[:, :, 3] = 255
    out[:, :width] = a
    out[:, width + gap:] = b
    return _image(out)


def combine(left: QImage, right: QImage, mode: Mode, **options) -> QImage:
    """Combine a stereo pair according to ``mode``."""
    if mode is Mode.OFF:
        return left
    if mode is Mode.ANAGLYPH:
        return anaglyph(left, right, greyscale=False)
    if mode is Mode.ANAGLYPH_GREY:
        return anaglyph(left, right, greyscale=True)
    if mode is Mode.SIDE_BY_SIDE:
        return side_by_side(left, right, crossed=False, **options)
    if mode is Mode.CROSS_EYED:
        return side_by_side(left, right, crossed=True, **options)
    raise ValueError(f"unknown stereo mode: {mode!r}")


def output_size(mode: Mode, width: int, height: int,
                gap: int = 8) -> tuple[int, int]:
    """How large the combined image will be, for one eye of this size."""
    if mode in (Mode.SIDE_BY_SIDE, Mode.CROSS_EYED):
        return width * 2 + max(int(gap), 0), height
    return width, height
