# Third-party components and redistribution

FACET's own source code is MIT licensed (see `LICENSE`), which places no
restriction on passing it on. The **built application** bundles third-party
components, and two of them carry obligations that must be met before a binary
is handed to anyone outside the group. This file records what those are and how
the build satisfies them.

## Summary

| Component | Licence | Obligation on redistribution |
|---|---|---|
| PySide6 / shiboken6 (Qt for Python) | **LGPL-3.0** (or GPL, or commercial) | see *Qt and the LGPL* below — this is the binding one |
| Qt 6 libraries (bundled inside PySide6) | **LGPL-3.0** | as above |
| gemmi | MPL-2.0 | keep the licence text; if gemmi's own source is modified, publish those files |
| spglib | BSD-3-Clause | keep the copyright notice |
| NumPy, SciPy, pandas | BSD-3-Clause | keep the copyright notices |
| matplotlib | matplotlib licence (PSF-derived, BSD-compatible) | keep the licence text |
| pymatgen | MIT | keep the copyright notice |
| PyInstaller | GPL-2.0-or-later **with a bootloader exception** | the exception explicitly permits distributing a bundled application under any licence; no obligation follows onto FACET |
| Mesa llvmpipe (`opengl32sw.dll`, shipped by Qt) | MIT | keep the copyright notice |

Nothing here forces FACET to become GPL, and nothing prevents the recipients of
a build from passing it on further — **provided the Qt condition below is met.**

## Qt and the LGPL — the one that constrains the build

Qt is used under the LGPL-3.0. That is allowed for a closed or permissively
licensed application, but LGPL-3.0 §4 requires that a recipient be able to
**replace the Qt libraries with their own modified version and re-run the
application**. Two consequences for FACET:

1. **The application must be built as a PyInstaller `onedir` bundle, not
   `onefile`.** In a `onedir` build the Qt DLLs sit beside the executable as
   ordinary replaceable files, which satisfies the relinking requirement
   directly. A `onefile` build hides them inside a self-extracting archive and
   makes compliance awkward to argue. `tools/build_exe.py` enforces `onedir`.

2. **The distribution must carry the LGPL-3.0 text and a note saying how to
   replace Qt.** The build writes `licenses/` next to the executable containing
   the full text of every licence above, and `licenses/README.txt` explaining
   the substitution.

Qt must also not be statically linked. PySide6's wheels ship dynamic libraries,
so this is satisfied as long as the build is not modified to do otherwise.

## Bond-valence parameter data

Parameter values are physical constants published in the literature, but a
*compilation* of them can attract its own protection, and different sources
carry different terms. FACET is deliberate about this:

* **Shipped with the application.** A curated table of fitted R₀ and b values
  for the pairs FACET has been validated on, each row carrying its citation.
  These are individually attributed literature values, not a copied compilation.

* **Computed, not shipped.** For any pair without a fitted entry, R₀ is
  *estimated* from the O'Keeffe & Brese (1991) electronegativity-based
  expression using their tabulated per-element r and c parameters. This covers
  75 elements including every lanthanide, and reproduces the fitted values to
  roughly ±0.03–0.08 Å — two to four times the ±0.02 Å the fitted values
  themselves carry. **Estimated parameters are labelled as estimated everywhere
  they are used**, in the interface and in every exported report.

* **Downloaded on request, never bundled.** The full IUCr accumulated table
  (`bvparm*.cif`), the Gagné & Hawthorne (2015) refit and the softBV set are
  fetched from their canonical sources by the user, on demand, into the user's
  own data directory. FACET does not redistribute them. This keeps the
  distribution clean of any compilation whose terms FACET has not verified.

## Crystal structure files

FACET ships no structural data. Sample structures used for testing are read
from the user's own files. Any CIF a user loads remains theirs, and FACET does
not transmit it anywhere.

## Checklist before handing a build to anyone

- [ ] built with `tools/build_exe.py` (which forces `onedir`)
- [ ] `licenses/` directory present next to `FACET.exe`
- [ ] `licenses/README.txt` present, describing Qt replacement
- [ ] no downloaded parameter tables inside the distribution directory
- [ ] version and build date shown in Help ▸ About
