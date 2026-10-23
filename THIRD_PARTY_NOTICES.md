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

* **Bundled under its own notice.** The IUCr accumulated table
  `bvparm2020.cif` (I. David Brown, Brockhouse Institute for Materials
  Research, McMaster University; dated 2020-11-25) ships in `facet/data/`
  exactly as distributed. Its notice allows use and distribution without fee
  for non-profit purposes provided (1) the notice is included, (2) no fee is
  charged for the table and (3) any change by anyone other than the copyright
  owner is recorded in the file. FACET charges no fee, includes the notice
  (it is the file's own header) and changes nothing in the file. Its rows cite
  their sources (Brown & Altermatt 1985, Brese & O'Keeffe 1991, Gagné &
  Hawthorne 2015, Adams 2001, Palenik and others); FACET uses the first row
  of each pair, the file's own order. **Anyone distributing FACET for a fee,
  or with the table altered, must consult the copyright owner first.**

* **Read from a file, never bundled.** The softBV set carries its own terms
  and is read from the user's own copy; so is any later bvparm release.

* **Optical-basicity tables, transcribed.** `facet/core/basicity.py` carries
  Pauling's 1960 electronegativity scale (one decimal, *The Nature of the
  Chemical Bond*, 3rd ed., Table 3-8), Duffy's basicity moderating parameters
  (J. Chem. Educ. 73 (1996) 1138, Table 1, sixteen cations), Lebouteiller and
  Courtine's known Λ values by cation (J. Solid State Chem. 137 (1998) 94,
  Table 2, thirty cations) and Dimitrov and Sakka's Λ from refractivity (J.
  Appl. Phys. 79 (1996) 1736, Table I, thirty-three oxides) — short tables of
  published numerical facts, cited at every use, with Duffy and Ingram's
  relation γ = 1.36 (χ − 0.26) (J. Non-Cryst. Solids 21 (1976) 373) behind
  them. `tests/test_basicity.py` re-derives the papers' own worked values
  from them.

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
