# Building and distributing FACET

FACET is meant to be handed to people who do not write code, so the deliverable
is a Windows application rather than a Python package. This is how it is built,
what comes out, and what has to be true before a build is passed on.

```
py -3.11 tools/build_exe.py            # everything
py -3.11 tools/build_exe.py --app      # only the application folder
py -3.11 tools/build_exe.py --no-clean # keep the previous build
```

Takes a few minutes. Everything lands in `dist/`, which is **not in the
repository** — the installer is over GitHub's 100 MB per-file limit, and a
repository that carries its own binaries becomes slow to clone for no benefit.
Binaries belong in a GitHub **Release**, which allows 2 GB per file.

| | Size | For |
|---|---|---|
| `dist/FACET/` | 254 MB | local use. Run `FACET.exe` from inside it. |
| `dist/FACET-Setup-<version>.exe` | 156 MB | what a collaborator receives. |
| `dist/FACET-<version>.zip` | 109 MB | for anyone who would rather not run an installer. |

## What the installer does

Per-user, so it **never asks for administrator rights**. That matters: a research
group is exactly the setting where people cannot install software on their own
machines.

* unpacks to `%LOCALAPPDATA%\Programs\FACET`
* writes a Start Menu shortcut, and a desktop shortcut if asked
* optionally associates `.cif` files
* registers an uninstaller in Add/Remove Programs
* offers to start FACET when it finishes

Uninstalling removes the folder, the registry entry, and the shortcuts the
installation made or that point at its executable. A `FACET.lnk` that points
somewhere else, such as a developer's link to a checkout, is left alone.

### Checked on 2026-10-13

The installer was driven through its own window, as a recipient uses it, on a
machine where FACET 0.3.1 had been installed by that version's installer (which
registered itself as 0.1.0, the constant its file carried at the time):

* **Upgrade.** The 0.5.1 installer opened as *Upgrade FACET*, named the copy it
  would replace, and installed into the same folder. Afterwards every one of
  the payload's 416 files was present with the CRC recorded in the archive,
  nothing else was in the folder but `uninstall.bat`, and the registry entry
  read 0.5.1.
* **Refusal.** Run again while the installed FACET was open, it stopped before
  touching anything -- *FACET is running from … Close it and run this installer
  again* -- and the folder was unchanged afterwards.
* **Uninstall.** `uninstall.bat` removed the folder and the registry entry,
  left alone a desktop `FACET.lnk` that pointed at a checkout, and ended on
  *Done.* with no error line.
* **Fresh install.** With no entry registered, the same installer opened as
  *Install FACET*, proposed `%LOCALAPPDATA%\Programs\FACET`, and the installed
  folder matched the payload the same way.

Not covered: a second user account on the same machine, a machine without
the Visual C++ runtime, and the SmartScreen prompt, which a locally built
file never triggers.

## The SmartScreen warning

The executable is not code-signed, so the first person to run it sees *"Windows
protected your PC"*. They have to click **More info → Run anyway**.

This is expected for any unsigned application, and it is worth saying so in the
message that carries the installer, because it otherwise reads as a virus
warning. Signing requires an Authenticode certificate from a commercial
certificate authority, of the order of a few hundred dollars a year — worth it
only if FACET goes beyond the group.

## Why a folder rather than a single file

`onedir`, not `onefile`, and that is a licence requirement rather than a
preference.

Qt is used under the LGPL-3.0, which obliges the distributor to let a recipient
replace the Qt libraries with their own build. In the `onedir` layout the Qt DLLs
sit beside `FACET.exe` as ordinary, replaceable files, which satisfies that
directly. A `onefile` build hides them inside a self-extracting archive and makes
the obligation awkward to argue.

`licenses/README.txt` inside the application folder spells out how to do the
replacement. **`THIRD_PARTY_NOTICES.md` has the full picture and the checklist to
run through before handing a build to anyone.**

## Size, and what is in it

264 MB installed. The large items, and why they are there:

| | |
|---|---|
| PySide6 / Qt | 100 MB — the interface and the GL bindings |
| scipy | 78 MB — KD-tree neighbour search and convex hulls |
| numpy | 27 MB |
| `opengl32sw.dll` | 20 MB — **the software renderer.** This is what lets FACET run on a machine with no graphics card, so it stays. |

`pandas`, `pyarrow` (80 MB on its own), `matplotlib` and `pymatgen` are
deliberately excluded, because the application does not use them. The cutoff
explorer, the diffraction plot and the 2D sections are all drawn with QPainter,
and the engine returns plain numpy arrays. `pymatgen` is needed only to *estimate*
a bond-valence parameter for a pair with no fitted entry, and the built
application says so rather than silently guessing.

Two hidden imports have to be declared explicitly, and both are recorded in
`tools/build_exe.py` next to the reason:

* `numpy._core` and `numpy._utils`, because PyInstaller's numpy hook does not
  fully cover numpy 2.4 and the frozen application otherwise dies at import with
  *"No module named numpy._core._exceptions"*;
* `PySide6.QtSvg` and `PySide6.QtPrintSupport`, because the SVG and PDF exports
  draw through `QSvgGenerator` and `QPdfWriter`. A missing one of those shows up
  only when a user clicks Export in the built application, which is the worst
  place to find out.

## Before passing a build on

1. `py -3.11 -m pytest tests/ -q` — the suite must be green.
2. Build, then actually run `dist/FACET/FACET.exe` on a structure. A missing
   hidden import is invisible until the application starts.
3. Check `dist/FACET/licenses/` exists and is populated.
4. Work through the checklist in `THIRD_PARTY_NOTICES.md`.
5. Attach the installer to a GitHub Release rather than committing it.
