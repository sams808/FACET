# dist/ — the built application

This folder is where the build puts everything. Its **contents are not in the
repository**: `FACET-Setup-0.2.0.exe` is 154 MB and GitHub rejects any file
over 100 MB. Binaries belong in a GitHub **Release**, not in the tree — a
release has a 2 GB per-file limit, and it keeps the repository small enough to
clone quickly.

## Building

```
py -3.11 tools/build_exe.py
```

Takes a few minutes. Produces:

| | Size | For |
|---|---|---|
| `FACET/` | 264 MB | **your own use.** Run `FACET.exe` from inside it. |
| `FACET-Setup-0.2.0.exe` | 154 MB | **what you send to a collaborator.** |
| `FACET-0.2.0.zip` | 100 MB | for anyone who would rather not run an installer |

`py -3.11 tools/build_exe.py --app` builds only the application folder, which
is the quicker loop while developing.

## What the installer does

Per-user, so it **never asks for administrator rights** — which matters,
because a research group is exactly the setting where people cannot install
software on their own machines.

* unpacks to `%LOCALAPPDATA%\Programs\FACET`
* writes a Start Menu shortcut, and a desktop shortcut if asked
* optionally associates `.cif` files
* registers an uninstaller in Add/Remove Programs
* offers to start FACET when it finishes

Uninstalling removes the folder, the shortcuts and the registry entry.

## The SmartScreen warning

The executable is not code-signed, so the first person to run it will see
*"Windows protected your PC"*. They must click **More info → Run anyway**. This
is expected for any unsigned application and is worth saying in the email that
carries it, because it otherwise looks like a virus warning.

Signing requires an Authenticode certificate from a commercial CA (of the order
of a few hundred dollars a year). Worth it only if FACET goes beyond the group.

## Why a folder rather than a single file

Qt is used under the LGPL-3.0, which requires that a recipient be able to
replace the Qt libraries with their own build. In this `onedir` layout the Qt
DLLs sit beside `FACET.exe` as ordinary, replaceable files, which satisfies
that directly. A `onefile` build would hide them inside a self-extracting
archive and make the obligation awkward to argue.

`licenses/README.txt` inside the application folder spells out how to do the
replacement. See `../THIRD_PARTY_NOTICES.md` for the full picture, and the
checklist to run through before handing a build to anyone.

## Size

264 MB installed. The large items, and why they are there:

| | |
|---|---|
| PySide6 / Qt | 100 MB — the interface and the GL bindings |
| scipy | 78 MB — KD-tree neighbour search and convex hulls |
| `opengl32sw.dll` | 20 MB — **the software renderer.** This is what lets FACET run on a machine with no graphics card, so it stays. |
| numpy | 27 MB | 

`pandas`, `pyarrow` (80 MB on its own), `matplotlib` and `pymatgen` are
deliberately excluded: the application does not use them. `pymatgen` is needed
only to *estimate* a bond-valence parameter for a pair with no fitted entry,
and the built application says so rather than silently guessing.
