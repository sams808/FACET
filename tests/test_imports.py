"""Every module must compile and import.

This exists because it did not. A stray newline inside an f-string in
``facet/ui/preview.py`` made the main window unimportable, and the whole suite
stayed green through it -- the panels each have tests, but nothing imported the
window that assembles them, so the one file a user meets first was the one file
never loaded.

Two levels, because they fail differently. Compiling catches a syntax error in a
file nothing imports. Importing catches a missing name, a circular import, or a
typo in a decorator -- things that only appear when the module body runs.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "facet"


def _python_files():
    return sorted(PACKAGE.rglob("*.py")) + sorted((ROOT / "tools").rglob("*.py"))


def _module_names():
    names = []
    for finder, name, _ in pkgutil.walk_packages([str(PACKAGE)], "facet."):
        names.append(name)
    return sorted(names)


@pytest.mark.parametrize("path", _python_files(),
                         ids=lambda p: str(p.relative_to(ROOT)).replace("\\", "/"))
def test_every_file_compiles(path):
    """A syntax error anywhere, in a file any test imports or not."""
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")


@pytest.mark.parametrize("name", _module_names())
def test_every_module_imports(name):
    """Import each module for real, so a missing name is caught too.

    A QApplication is created first: importing a widget module is harmless
    without one, but any module-level Qt object would not be, and it costs
    nothing to be safe.
    """
    if name.startswith("facet.gl") or name.startswith("facet.ui"):
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
    importlib.import_module(name)


def test_the_main_window_class_can_be_reached():
    """The one import a user's first click depends on."""
    from PySide6.QtWidgets import QApplication, QMainWindow

    QApplication.instance() or QApplication([])
    from facet.ui.preview import PreviewWindow

    assert issubclass(PreviewWindow, QMainWindow)


def test_the_entry_point_exists():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from facet.ui import app

    assert callable(getattr(app, "main", None))


def test_the_core_engine_imports_without_qt():
    """The engine must not pull Qt in.

    It is what makes the core testable headless and the import fast, and it is
    easy to break by reaching for a Qt colour type in a core module.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "import facet.core.coordination, facet.core.diffraction,"
        "facet.core.planes, facet.core.volume, facet.core.readers,"
        "facet.core.exporters, facet.core.utilities,"
        "facet.core.md_model, facet.core.md_readers, facet.core.bulk;"
        "bad=[m for m in sys.modules if m.startswith('PySide6')];"
        "print('QT' if bad else 'CLEAN')"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout, f"Qt was imported by the core: {result.stdout}"
