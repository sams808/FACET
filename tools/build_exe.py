"""Build FACET into a distributable Windows application.

Produces, under ``dist/``:

* ``FACET/``              the application folder. Run ``FACET.exe`` from here.
* ``FACET-Setup-x.y.z.exe``  a single-file installer to hand to a collaborator.
* ``FACET-x.y.z.zip``     the same folder zipped, for anyone who would rather
                          not run an installer.

**onedir, not onefile, and that is a licence requirement rather than a
preference.** Qt is used under the LGPL-3.0, which obliges the distributor to
let a recipient replace the Qt libraries with their own build. In a onedir
bundle the Qt DLLs sit beside the executable as ordinary files, which satisfies
that directly. A onefile bundle hides them inside a self-extracting archive and
makes the obligation awkward to argue. See THIRD_PARTY_NOTICES.md.

The installer is built from Python rather than from Inno Setup or NSIS, so the
build needs nothing installed beyond what FACET already depends on. It extracts
the application folder to the user's local application data, writes Start Menu
and optional desktop shortcuts, and registers an uninstaller -- all per-user, so
it never needs administrator rights, which is what makes it safe to send to
someone who cannot install software on their own machine.

Usage:
    py -3.11 tools/build_exe.py            # everything
    py -3.11 tools/build_exe.py --app      # just the application folder
    py -3.11 tools/build_exe.py --no-clean # keep the previous build
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
ASSETS = ROOT / "assets"

sys.path.insert(0, str(ROOT))
from facet.version import NAME, __version__  # noqa: E402

APP_DIR = DIST / NAME
ICON = ASSETS / "facet.ico"

# PyInstaller pulls in whatever it finds importable. The shared site-packages on
# this machine carries torch, transformers, llvmlite and botocore among others,
# and without these exclusions the bundle goes from roughly 350 MB to nearly
# 900 MB of libraries FACET never touches.
EXCLUDES = [
    "torch", "torchvision", "torchaudio", "transformers", "tokenizers",
    "tensorflow", "keras", "jax", "sklearn", "scikit_learn", "numba",
    "llvmlite", "botocore", "boto3", "awscli", "IPython", "jupyter",
    "jupyter_client", "jupyter_core", "notebook", "nbconvert", "nbformat",
    "ipykernel", "ipywidgets", "qtconsole", "sphinx", "docutils", "pytest",
    "setuptools", "pip", "wheel", "cython", "Cython", "sympy", "networkx",
    "PIL.ImageQt", "PyQt5", "PyQt6", "tkinter", "test", "distutils",
    "plotly", "dash", "seaborn", "statsmodels", "h5py", "tables",
    "pymongo", "sqlalchemy", "zmq", "tornado",
    # pandas drags in pyarrow, which is 80 MB on its own and is the single
    # largest thing in the bundle. Neither is used by the application: the
    # cutoff explorer is drawn with QPainter and the engine returns plain
    # numpy arrays. They stay optional extras for the Python route.
    "pandas", "pyarrow", "matplotlib", "PIL", "Pillow",
    # pulled in transitively by things that are themselves excluded
    "cryptography", "OpenSSL", "lxml", "Pythonwin", "win32com", "pywin32",
    "pythoncom", "pywintypes", "win32ui",
    # Qt modules FACET does not use; each is tens of megabytes
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick", "PySide6.QtQuick3D", "PySide6.Qt3DCore",
    "PySide6.Qt3DRender", "PySide6.Qt3DExtras", "PySide6.Qt3DAnimation",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtBluetooth", "PySide6.QtNfc",
    "PySide6.QtPositioning", "PySide6.QtSerialPort", "PySide6.QtSql",
    "PySide6.QtTest", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtDesigner",
    "PySide6.QtHelp", "PySide6.QtUiTools", "PySide6.QtWebSockets",
    "PySide6.QtWebChannel", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtSensors",
    "PySide6.QtSpatialAudio", "PySide6.QtStateMachine", "PySide6.QtTextToSpeech",
]

# pymatgen is imported lazily and only to estimate a bond-valence parameter for
# a pair with no fitted entry. Bundling it would roughly double the download for
# a fallback most sessions never reach, so the exe ships without it and says so
# when a pair is uncovered.
EXCLUDES.append("pymatgen")

HIDDEN = [
    "gemmi", "spglib", "scipy.spatial.transform._rotation_groups",
    # The SVG and PDF exports draw through QSvgGenerator and QPdfWriter. Qt's
    # own hook does not always pull QtSvg in, and a missing one shows up only
    # when a user clicks Export in the built application, which is the worst
    # place to find out.
    "PySide6.QtSvg", "PySide6.QtPrintSupport",
]

# PyInstaller 6.11's numpy hook does not fully cover numpy 2.4: the frozen
# application fails at import with "No module named numpy._core._exceptions".
# Collecting the private core package explicitly fixes it, and costs nothing
# because those modules are pure Python.
COLLECT_SUBMODULES = ["numpy._core", "numpy._utils"]


def log(message: str) -> None:
    print(f"  {message}", flush=True)


def make_icon() -> Path:
    """Draw the .ico from the same code that draws the logo everywhere else."""
    ASSETS.mkdir(exist_ok=True)
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from facet.ui.branding import logo_pixmap, splash_pixmap

    logo_pixmap(256).save(str(ASSETS / "facet_logo.png"))
    logo_pixmap(256, ground=False).save(str(ASSETS / "facet_mark.png"))
    splash_pixmap().save(str(ASSETS / "facet_splash.png"))
    logo_pixmap(256).save(str(ICON), "ICO")
    log(f"icon -> {ICON.relative_to(ROOT)}")
    return ICON


def write_licences(target: Path) -> None:
    """The licence texts and the Qt replacement note the LGPL requires."""
    folder = target / "licenses"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, folder / name)
    (folder / "README.txt").write_text(
        f"""{NAME} {__version__} -- licences
=====================================

{NAME}'s own source code is MIT licensed; see LICENSE.

This application uses the Qt toolkit through PySide6 under the LGPL-3.0.
The LGPL requires that you be able to replace those libraries with your own
build and re-run the application. That is why {NAME} is distributed as a
folder rather than a single file: the Qt DLLs sit beside {NAME}.exe as
ordinary files.

To replace them, build or obtain matching PySide6/Qt binaries of the same
major version and overwrite the PySide6 folder and the Qt6*.dll files in this
directory. No other change to the application is needed.

The full text of the LGPL-3.0 is available at
https://www.gnu.org/licenses/lgpl-3.0.html and the corresponding Qt sources at
https://download.qt.io/.

Other components, with their licences, are listed in
THIRD_PARTY_NOTICES.md.
""", encoding="utf-8")
    log(f"licences -> {folder.relative_to(ROOT)}")


def build_app(clean: bool = True) -> Path:
    if clean:
        for path in (APP_DIR, BUILD):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

    icon = make_icon()
    entry = ROOT / "facet" / "__main__.py"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onedir",                       # LGPL: Qt must stay replaceable
        "--windowed",                     # no console window
        "--name", NAME,
        "--icon", str(icon),
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
        "--paths", str(ROOT),
    ]
    for module in EXCLUDES:
        cmd += ["--exclude-module", module]
    for module in HIDDEN:
        cmd += ["--hidden-import", module]
    for package in COLLECT_SUBMODULES:
        cmd += ["--collect-submodules", package]
    cmd.append(str(entry))

    log("running PyInstaller (this takes a few minutes)")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"PyInstaller failed with {result.returncode}")

    write_licences(APP_DIR)
    size = sum(f.stat().st_size for f in APP_DIR.rglob("*") if f.is_file())
    log(f"application -> {APP_DIR.relative_to(ROOT)}  ({size / 1e6:.0f} MB)")
    return APP_DIR


def zip_app() -> Path:
    archive = DIST / f"{NAME}-{__version__}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in APP_DIR.rglob("*"):
            if path.is_file():
                z.write(path, Path(NAME) / path.relative_to(APP_DIR))
    log(f"archive -> {archive.name}  ({archive.stat().st_size / 1e6:.0f} MB)")
    return archive


def build_installer(archive: Path) -> Path:
    """Freeze the installer with the zipped application carried inside it."""
    installer_src = ROOT / "tools" / "installer.py"
    out_name = f"{NAME}-Setup-{__version__}"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onefile",                  # an installer is one file by nature
        "--windowed",
        "--name", out_name,
        "--icon", str(ICON),
        "--distpath", str(DIST),
        "--workpath", str(BUILD / "installer"),
        "--specpath", str(BUILD / "installer"),
        "--paths", str(ROOT),
        "--add-data", f"{archive}{os.pathsep}payload",
        "--exclude-module", "numpy", "--exclude-module", "scipy",
        "--exclude-module", "gemmi", "--exclude-module", "spglib",
        "--exclude-module", "matplotlib", "--exclude-module", "pandas",
        str(installer_src),
    ]
    log("building the installer")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"installer build failed with {result.returncode}")

    exe = DIST / f"{out_name}.exe"
    log(f"installer -> {exe.name}  ({exe.stat().st_size / 1e6:.0f} MB)")
    return exe


def write_dist_readme() -> None:
    """The note beside the artefacts, naming this version's files.

    It used to be a static file that nothing updated, so it named the previous
    version's installer -- in a folder that keeps every build ever made, where
    the names differ by one character. It is the note read when choosing which
    file to send to someone, so it is written by the build that makes them.
    """
    (DIST / "README.md").write_text(f"""\
# dist/ - the built application

This folder is where the build puts everything, and it keeps **every version
ever built**. The files below are the ones from the most recent build; anything
else here is older, and is not what you want to hand to anybody.

Its contents are **not in the repository**: `{NAME}-Setup-{__version__}.exe` is
over GitHub's 100 MB per-file limit, and a repository that carries its own
binaries becomes slow to clone for no benefit. Binaries belong in a GitHub
**Release**, which allows 2 GB per file.

## Building

```
py -3.11 tools/build_exe.py
```

Takes a few minutes. This build produced:

| | For |
|---|---|
| `{NAME}/` | **your own use.** Run `{NAME}.exe` from inside it. |
| `{NAME}-Setup-{__version__}.exe` | **what you send to a collaborator.** |
| `{NAME}-{__version__}.zip` | for anyone who would rather not run an installer. |

`py -3.11 tools/build_exe.py --app` builds only the application folder, which is
the quicker loop while developing.

## Before handing a build to anyone

`THIRD_PARTY_NOTICES.md` in the repository root has the licence checklist, and
`BUILDING.md` explains why the application is a folder rather than a single file
-- it is an LGPL obligation, not a preference. The SmartScreen warning on first
run is expected: the executable is not code-signed. Say so in the message that
carries it, or it reads as a virus warning.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Build {NAME}")
    parser.add_argument("--app", action="store_true",
                        help="build only the application folder")
    parser.add_argument("--no-clean", action="store_true",
                        help="keep the previous build directory")
    args = parser.parse_args()

    print(f"Building {NAME} {__version__}")
    DIST.mkdir(exist_ok=True)

    build_app(clean=not args.no_clean)
    if args.app:
        return 0

    archive = zip_app()
    build_installer(archive)

    write_dist_readme()

    print()
    print("Done. In dist/:")
    older = []
    for item in sorted(DIST.iterdir()):
        if not item.is_file():
            print(f"  {item.name + '/':40s} (run {NAME}.exe from here)")
            continue
        # Every build ever made stays here, and the names differ by one
        # character. Whichever is handed on is chosen by reading this list.
        stale = item.name.startswith(NAME) and __version__ not in item.name
        note = "  <- an older build" if stale else ""
        if stale:
            older.append(item.name)
        print(f"  {item.name:40s} {item.stat().st_size / 1e6:6.0f} MB{note}")
    if older:
        print()
        print(f"  {len(older)} file(s) from an earlier version are still here. "
              f"Send {NAME}-Setup-{__version__}.exe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
