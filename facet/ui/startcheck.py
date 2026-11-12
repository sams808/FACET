"""``FACET.exe --selfcheck [--out report.json]``: this copy of FACET checks
that it is whole, then exits.

    py -3.11 -m facet --selfcheck
    FACET.exe --selfcheck --out report.json

The test suite checks the arithmetic, from a source checkout. What it cannot
see is what a build or an installation left out: the bond-valence table, the
bundled structures, the change log, a Qt plugin, a module the packer did not
follow. Version 2.0.0's wheel lacked the bond-valence table, and every test
passed. This runs in the copy being checked -- the frozen application, or a
``pip install`` -- and uses what that copy carries:

* the bond-valence table answers for a pair;
* every bundled structure is present;
* the numbers the manual states for them are recomputed (``facet.check``);
* the basicity library reads a composition and a bundled structure;
* the library opens a bundled crystal and the practice glass model;
* the manual and the change log are there;
* the window opens on a bundled structure, lists its sites and draws.

Exit code 0 when every line holds, 1 otherwise. The frozen application has
no console, so ``--out`` names a JSON file for the same lines, with the
versions of what is running.
"""
from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from pathlib import Path

from ..version import NAME, __version__

#: how long the window may take to report its renderer, in seconds; the
#: offscreen platform has no OpenGL context to report and is not waited for
READY_TIMEOUT_S = 20.0


def _environment() -> dict:
    import numpy
    import scipy
    from PySide6 import __version__ as pyside
    from PySide6.QtCore import qVersion
    from PySide6.QtGui import QGuiApplication

    out = {
        "facet": __version__,
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "qt": qVersion(),
        "pyside": pyside,
        "qt_platform": QGuiApplication.platformName(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }
    for name in ("gemmi", "spglib", "matplotlib", "pymatgen"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


# --- the checks. Each returns the line to print; raising is a failure. -----

def _bond_valence_table() -> str:
    from ..core import bv

    found = bv.DEFAULT.get("Si", 4, "O")
    if found is None:
        raise RuntimeError("no parameters for Si(4+)-O")
    return f"Si(4+)-O answers, {type(found).__name__}"


def _examples() -> str:
    from ..core import examples

    missing = [e.filename for e in examples.CATALOGUE if not e.exists]
    if missing:
        raise RuntimeError("missing: " + ", ".join(missing))
    return f"{len(examples.CATALOGUE)} structures in {examples.folder()}"


def _reference_numbers() -> str:
    from ..core import selfcheck

    reference = selfcheck.load_reference()
    rows = selfcheck.check_examples(reference) + selfcheck.check_sums(reference)
    outside = [r for r in rows if not r.agrees]
    if outside or not rows:
        first = outside[0] if outside else None
        raise RuntimeError(
            selfcheck.summary(rows) + (
                f"; first outside: {first.example} {first.site} "
                f"{first.quantity} {first.computed} for {first.reference}"
                if first else ""))
    return selfcheck.summary(rows)


def _basicity_library() -> str:
    import math

    from .. import basicity as fb
    from ..core import examples

    glass = fb.composition("30Na2O-70SiO2").value
    quartz = next(e for e in examples.CATALOGUE if e.key == "quartz")
    crystal = fb.structure(str(quartz.path))
    anions = crystal.anions()
    if not (math.isfinite(glass) and math.isfinite(crystal.value) and anions):
        raise RuntimeError(f"composition {glass}, quartz {crystal.value}, "
                           f"{len(anions)} anion sites")
    return (f"30Na2O-70SiO2 {glass:.3f}; quartz {crystal.value:.3f} over "
            f"{len(anions)} anion site(s)")


def _library() -> str:
    """The library front door on what this copy carries: a bundled crystal
    and, when it is there, the practice glass."""
    from .. import library

    quartz = library.example("quartz")
    row = quartz.sites()[0]
    if row["CN"] != 4:
        raise RuntimeError(f"quartz Si reads CN {row['CN']}")
    text = f"quartz Si: CN {row['CN']}, valence sum {row['BVS / v.u.']:.3f} v.u."
    if "glass" not in library.examples():
        raise RuntimeError("the practice glass model is missing")
    with library.example("glass") as glass:
        atoms = glass.atoms()
        silicon = atoms.where(element="Si").column("CN")
        if glass.n_atoms != 1200 or not len(silicon):
            raise RuntimeError(f"the practice glass holds {glass.n_atoms} atoms")
        text += (f"; practice glass: {glass.n_atoms} atoms, mean CN of Si "
                 f"{silicon.mean():.2f}")
    return text


def _documents() -> str:
    from ..core import examples
    from . import help as help_mod

    manual = help_mod.manual_html()
    if len(manual) < 10_000:
        raise RuntimeError(f"the manual is {len(manual)} characters")
    for candidate in (examples.folder().parent / "CHANGELOG.md",
                      Path(__file__).resolve().parents[1] / "data" / "CHANGELOG.md"):
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
            # the file is written from the commit history, one commit after
            # the one that names a version: it may not name this one yet
            versions = text.count("\n## ")
            if versions < 1:
                raise RuntimeError(f"{candidate} lists no version")
            return (f"manual {len(manual)} characters; change log of "
                    f"{versions} version(s)")
    raise RuntimeError("no change log in this copy")


def _window(app) -> str:
    from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
    from PySide6.QtGui import QGuiApplication

    from ..core import examples
    from .preview import PreviewWindow

    def spin(ms: int) -> None:
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    example = next(e for e in examples.CATALOGUE if e.key == "quartz")
    window = PreviewWindow(str(example.path))
    try:
        window.show()
        offscreen = QGuiApplication.platformName() == "offscreen"
        deadline = time.monotonic() + (1.0 if offscreen else READY_TIMEOUT_S)
        while window.view.caps is None and time.monotonic() < deadline:
            spin(50)
        spin(300)                                # one settled frame
        if window.structure is None:
            raise RuntimeError(f"{example.filename} did not load")
        sites = window.site_list.count()
        if sites < 1:
            raise RuntimeError("the site list is empty")
        if NAME not in window.windowTitle() or __version__ not in window.windowTitle():
            raise RuntimeError(f"window title {window.windowTitle()!r}")
        image = window.grab().toImage()
        if image.isNull() or image.width() < 200 or image.height() < 200:
            raise RuntimeError("the window did not draw")
        step_x, step_y = max(1, image.width() // 64), max(1, image.height() // 64)
        colours = {image.pixel(x, y)
                   for x in range(0, image.width(), step_x)
                   for y in range(0, image.height(), step_y)}
        if len(colours) < 4:
            raise RuntimeError(f"the window drew {len(colours)} colour(s)")
        caps = window.view.caps
        renderer = caps.describe() if caps is not None else \
            "no renderer reported within the time allowed"
        return (f"{sites} site(s) listed, {image.width()} x {image.height()} px "
                f"drawn; renderer: {renderer}")
    finally:
        # destroyed here, with its OpenGL context, and not at interpreter
        # exit: processEvents() alone does not deliver a deferred delete
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()


CHECKS = (
    ("bond-valence table", _bond_valence_table, False),
    ("bundled structures", _examples, False),
    ("reference numbers", _reference_numbers, False),
    ("basicity library", _basicity_library, False),
    ("library", _library, False),
    ("manual and change log", _documents, False),
    ("window", _window, True),
)


def run(app, out: str | None = None) -> int:
    """Run every check with the QApplication ``app``; write ``out`` if named."""
    rows = []
    for name, check, wants_app in CHECKS:
        started = time.perf_counter()
        try:
            detail, held = (check(app) if wants_app else check()), True
        except Exception as error:               # a check that raises has failed
            detail = f"{type(error).__name__}: {error}"
            held = False
            trace = traceback.format_exc()
        else:
            trace = ""
        rows.append({"check": name, "holds": held, "detail": detail,
                     "seconds": round(time.perf_counter() - started, 2),
                     **({"traceback": trace} if trace else {})})
    holds = all(r["holds"] for r in rows)
    report = {"holds": holds, "environment": _environment(), "checks": rows}

    lines = [f"{NAME} {__version__} self-check "
             f"({'frozen' if report['environment']['frozen'] else 'from source or pip'}, "
             f"Qt platform {report['environment']['qt_platform']})"]
    for r in rows:
        lines.append(f"  {'holds' if r['holds'] else 'FAILS':5s}  {r['check']}: {r['detail']}")
        if r.get("traceback"):
            lines.append(r["traceback"].rstrip())
    lines.append(f"{sum(r['holds'] for r in rows)} of {len(rows)} checks hold")
    if out:
        Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if sys.stdout is not None:                   # a windowed build has none
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
        print("\n".join(lines), flush=True)
    return 0 if holds else 1


def main(app, argv: list[str]) -> int:
    """Run the checks inside the event loop, as the application itself runs,
    and return the exit code."""
    from PySide6.QtCore import QTimer

    code = [1]

    def go() -> None:
        try:
            code[0] = run(app, out_path(argv))
        finally:
            app.exit(code[0])

    QTimer.singleShot(0, go)
    app.exec()
    return code[0]


def out_path(argv: list[str]) -> str | None:
    """The file named after ``--out``, if any."""
    if "--out" in argv:
        at = argv.index("--out")
        if at + 1 < len(argv):
            return argv[at + 1]
    return None
